from pathlib import Path
from typing import Any

import httpx

from src.api.anti_spoofing.schema import AntiSpoofingResponse, HealthResponse
from src.api.base_client import BaseClient
from src.config import settings


class AntispoofingClientError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        code: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.details = details


class AntispoofingClient(BaseClient):
    def __init__(self, base_url: str | None = None):
        super().__init__(base_url or settings.ANTISPOOFING_BASE_URL)
        self._token: str | None = None

    async def _ensure_authenticated(self) -> None:
        if self._token is not None:
            return

        try:
            response = await self._post(
                "/v1/auth/token",
                data={
                    "username": settings.ANTISPOOFING_USERNAME,
                    "password": settings.ANTISPOOFING_PASSWORD,
                    "grant_type": "password",
                },
            )
        except httpx.HTTPStatusError as exc:
            raise AntispoofingClientError(
                f"Authentication failed: HTTP {exc.response.status_code}",
                status_code=exc.response.status_code,
            ) from exc
        except httpx.HTTPError as exc:
            raise AntispoofingClientError(
                f"Authentication request failed: {exc}"
            ) from exc

        self._token = response.json()["access_token"]
        self._headers["Authorization"] = f"Bearer {self._token}"

    async def predict(
        self,
        audio_file_path: str | Path,
        include_global_prediction: bool = True,
    ) -> AntiSpoofingResponse:
        audio_path = Path(audio_file_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

        await self._ensure_authenticated()

        for attempt in range(2):
            with audio_path.open("rb") as audio_file:
                try:
                    response = await self._post(
                        "/v1/antispoofing/predict",
                        files={
                            "audio": (
                                audio_path.name,
                                audio_file,
                                self._guess_content_type(audio_path),
                            )
                        },
                        data={
                            "include_global_prediction": str(include_global_prediction).lower(),
                        },
                    )
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code == 401 and attempt == 0:
                        self._token = None
                        self._headers.pop("Authorization", None)
                        await self._ensure_authenticated()
                        continue
                    raise self._build_error(exc.response) from exc
                except httpx.HTTPError as exc:
                    raise AntispoofingClientError(
                        f"External anti-spoofing service request failed: {exc}"
                    ) from exc

            return AntiSpoofingResponse.model_validate(response.json())

        raise AntispoofingClientError("Authentication retry exhausted")

    async def health(self) -> HealthResponse:
        try:
            response = await self._get("/v1/antispoofing/health", timeout=10.0)
        except httpx.HTTPStatusError as exc:
            raise self._build_error(exc.response) from exc
        except httpx.HTTPError as exc:
            raise AntispoofingClientError(
                f"External anti-spoofing health check failed: {exc}"
            ) from exc

        return HealthResponse.model_validate(response.json())

    @staticmethod
    def _guess_content_type(audio_path: Path) -> str:
        content_types = {
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".flac": "audio/flac",
            ".ogg": "audio/ogg",
            ".m4a": "audio/mp4",
        }
        return content_types.get(audio_path.suffix.lower(), "application/octet-stream")

    @staticmethod
    def _build_error(response: httpx.Response) -> AntispoofingClientError:
        try:
            body = response.json()
            detail = body.get("detail", "")
            if isinstance(detail, list):
                message = "; ".join(
                    f"{'.'.join(str(loc) for loc in d.get('loc', []))}: {d.get('msg', '')}"
                    for d in detail
                )
            else:
                message = str(detail)
        except Exception:
            message = ""

        return AntispoofingClientError(
            message or f"External anti-spoofing service returned HTTP {response.status_code}",
            status_code=response.status_code,
        )
