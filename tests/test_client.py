"""Tests for the external anti-spoofing HTTP client."""

from pathlib import Path

import httpx
import pytest
from src.api.anti_spoofing.client import AntispoofingClient, AntispoofingClientError
from src.api.anti_spoofing.schema import AntiSpoofingResponse, HealthResponse
from tests.conftest import BASE_URL, make_response_json


@pytest.fixture
def audio_file(tmp_path: Path) -> Path:
    path = tmp_path / "sample.wav"
    path.write_bytes(b"RIFF....WAVEfmt ")
    return path


# --- pure helpers ---------------------------------------------------------


def test_guess_content_type_known_extensions():
    assert AntispoofingClient._guess_content_type(Path("a.wav")) == "audio/wav"
    assert AntispoofingClient._guess_content_type(Path("a.MP3")) == "audio/mpeg"
    assert AntispoofingClient._guess_content_type(Path("a.flac")) == "audio/flac"


def test_guess_content_type_unknown_extension():
    assert AntispoofingClient._guess_content_type(Path("a.xyz")) == "application/octet-stream"


def test_build_error_with_string_detail():
    response = httpx.Response(404, json={"detail": "Not found"})
    error = AntispoofingClient._build_error(response)
    assert isinstance(error, AntispoofingClientError)
    assert str(error) == "Not found"
    assert error.status_code == 404


def test_build_error_with_validation_detail_list():
    response = httpx.Response(
        422,
        json={"detail": [{"loc": ["body", "audio"], "msg": "field required"}]},
    )
    error = AntispoofingClient._build_error(response)
    assert str(error) == "body.audio: field required"
    assert error.status_code == 422


def test_build_error_with_non_json_body_falls_back():
    response = httpx.Response(500, text="upstream exploded")
    error = AntispoofingClient._build_error(response)
    assert "HTTP 500" in str(error)
    assert error.status_code == 500


# --- predict --------------------------------------------------------------


async def test_predict_missing_file_raises(tmp_path: Path):
    client = AntispoofingClient(base_url=BASE_URL)
    with pytest.raises(FileNotFoundError):
        await client.predict(tmp_path / "does-not-exist.wav")


async def test_predict_success(httpx_mock, audio_file: Path):
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE_URL}/v1/auth/token",
        json={"access_token": "tok-1"},
    )
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE_URL}/v1/antispoofing/predict",
        json=make_response_json(label="bonafide"),
    )

    client = AntispoofingClient(base_url=BASE_URL)
    result = await client.predict(audio_file)

    assert isinstance(result, AntiSpoofingResponse)
    assert result.global_prediction is not None
    assert result.global_prediction.label == "bonafide"
    # Bearer token from the auth step is attached to subsequent requests.
    predict_request = httpx_mock.get_requests()[-1]
    assert predict_request.headers["Authorization"] == "Bearer tok-1"


async def test_predict_reauthenticates_on_401(httpx_mock, audio_file: Path):
    # 1) initial auth, 2) predict -> 401, 3) re-auth, 4) predict -> 200
    httpx_mock.add_response(
        method="POST", url=f"{BASE_URL}/v1/auth/token", json={"access_token": "tok-old"}
    )
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE_URL}/v1/antispoofing/predict",
        status_code=401,
        json={"detail": "token expired"},
    )
    httpx_mock.add_response(
        method="POST", url=f"{BASE_URL}/v1/auth/token", json={"access_token": "tok-new"}
    )
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE_URL}/v1/antispoofing/predict",
        json=make_response_json(label="spoof", spoof=0.8, bonafide=0.2),
    )

    client = AntispoofingClient(base_url=BASE_URL)
    result = await client.predict(audio_file)

    assert result.global_prediction is not None
    assert result.global_prediction.label == "spoof"
    # The retried predict call carries the refreshed token.
    assert httpx_mock.get_requests()[-1].headers["Authorization"] == "Bearer tok-new"


async def test_predict_gives_up_after_second_401(httpx_mock, audio_file: Path):
    httpx_mock.add_response(
        method="POST", url=f"{BASE_URL}/v1/auth/token", json={"access_token": "tok-old"}
    )
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE_URL}/v1/antispoofing/predict",
        status_code=401,
        json={"detail": "still unauthorized"},
    )
    httpx_mock.add_response(
        method="POST", url=f"{BASE_URL}/v1/auth/token", json={"access_token": "tok-new"}
    )
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE_URL}/v1/antispoofing/predict",
        status_code=401,
        json={"detail": "still unauthorized"},
    )

    client = AntispoofingClient(base_url=BASE_URL)
    with pytest.raises(AntispoofingClientError) as exc_info:
        await client.predict(audio_file)
    assert exc_info.value.status_code == 401


async def test_predict_auth_failure_raises(httpx_mock, audio_file: Path):
    httpx_mock.add_response(
        method="POST",
        url=f"{BASE_URL}/v1/auth/token",
        status_code=403,
        json={"detail": "bad credentials"},
    )

    client = AntispoofingClient(base_url=BASE_URL)
    with pytest.raises(AntispoofingClientError) as exc_info:
        await client.predict(audio_file)
    assert exc_info.value.status_code == 403


# --- health ---------------------------------------------------------------


async def test_health_success(httpx_mock):
    httpx_mock.add_response(
        method="GET",
        url=f"{BASE_URL}/v1/antispoofing/health",
        json={"status": "ok", "model_loaded": True},
    )

    client = AntispoofingClient(base_url=BASE_URL)
    result = await client.health()

    assert isinstance(result, HealthResponse)
    assert result.status == "ok"
    assert result.model_loaded is True
