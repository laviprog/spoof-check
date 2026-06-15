from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class PredictionLabel(str):
    SPOOF = "spoof"
    BONAFIDE = "bonafide"


class AudioMetadata(BaseModel):
    filename: str
    duration_sec: float
    sample_rate: int | None = None
    channels: int | None = None


class PredictionResult(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)
    spoof_probability: float = Field(ge=0, le=1)
    bonafide_probability: float = Field(ge=0, le=1)
    antispoofing_score: float | None = None
    prediction_by_logits: dict[str, Any] | None = None
    prediction_by_score: dict[str, Any] | None = None


class WindowPrediction(BaseModel):
    window_index: int
    start_sec: float
    end_sec: float
    spoof_probability: float
    bonafide_probability: float
    antispoofing_score: float | None = None


class ProcessingMetadata(BaseModel):
    model_name: str
    model_version: str
    processing_time_ms: int


class AntiSpoofingResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    request_id: str
    audio: AudioMetadata
    windows: list[WindowPrediction]
    processing: ProcessingMetadata
    global_prediction: PredictionResult | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "unavailable"]
    model_loaded: bool
