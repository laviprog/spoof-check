"""Shared test configuration and fixtures."""

import os

os.environ.setdefault("ANTISPOOFING_BASE_URL", "https://test.example.com")
os.environ.setdefault("ANTISPOOFING_USERNAME", "test-user")
os.environ.setdefault("ANTISPOOFING_PASSWORD", "test-pass")

BASE_URL = "https://test.example.com"


def make_response_json(
    *,
    label: str = "bonafide",
    spoof: float = 0.1,
    bonafide: float = 0.9,
    with_global: bool = True,
    windows: list[dict] | None = None,
) -> dict:
    """Build a valid ``AntiSpoofingResponse`` payload for the external API."""
    if windows is None:
        windows = [
            {
                "window_index": 0,
                "start_sec": 0.0,
                "end_sec": 2.0,
                "spoof_probability": spoof,
                "bonafide_probability": bonafide,
            }
        ]

    payload: dict = {
        "request_id": "req-123",
        "audio": {
            "filename": "test.wav",
            "duration_sec": 4.0,
            "sample_rate": 16000,
            "channels": 1,
        },
        "windows": windows,
        "processing": {
            "model_name": "spectra0",
            "model_version": "1.0",
            "processing_time_ms": 42,
        },
    }
    if with_global:
        payload["global_prediction"] = {
            "label": label,
            "confidence": max(spoof, bonafide),
            "spoof_probability": spoof,
            "bonafide_probability": bonafide,
        }
    return payload
