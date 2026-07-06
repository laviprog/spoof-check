"""Tests for the external-window aggregation and formatting logic."""

import pytest
from src.api.anti_spoofing.schema import AntiSpoofingResponse
from src.web.gradio_app import GradioApp
from tests.conftest import make_response_json


def _response(windows: list[dict]) -> AntiSpoofingResponse:
    return AntiSpoofingResponse.model_validate(
        make_response_json(with_global=False, windows=windows)
    )


def test_aggregate_windows_averages_and_labels_spoof():
    response = _response(
        [
            {
                "window_index": 0,
                "start_sec": 0.0,
                "end_sec": 2.0,
                "spoof_probability": 0.8,
                "bonafide_probability": 0.2,
            },
            {
                "window_index": 1,
                "start_sec": 2.0,
                "end_sec": 4.0,
                "spoof_probability": 0.6,
                "bonafide_probability": 0.4,
            },
        ]
    )

    prediction = GradioApp._aggregate_external_windows(response)

    assert prediction.spoof_probability == pytest.approx(0.7)
    assert prediction.bonafide_probability == pytest.approx(0.3)
    assert prediction.label == "spoof"
    assert prediction.confidence == pytest.approx(0.7)


def test_aggregate_windows_labels_bonafide():
    response = _response(
        [
            {
                "window_index": 0,
                "start_sec": 0.0,
                "end_sec": 2.0,
                "spoof_probability": 0.1,
                "bonafide_probability": 0.9,
            }
        ]
    )

    prediction = GradioApp._aggregate_external_windows(response)

    assert prediction.label == "bonafide"
    assert prediction.confidence == pytest.approx(0.9)


def test_aggregate_windows_without_windows_raises():
    response = _response([])
    with pytest.raises(ValueError, match="no window predictions"):
        GradioApp._aggregate_external_windows(response)


@pytest.mark.parametrize(
    ("classification", "expected"),
    [
        ("bonafide", "🟢 **Подлинный**"),
        ("BONAFIDE", "🟢 **Подлинный**"),
        ("spoof", "🔴 **Поддельный**"),
    ],
)
def test_format_classification(classification: str, expected: str):
    assert GradioApp._format_classification(classification) == expected
