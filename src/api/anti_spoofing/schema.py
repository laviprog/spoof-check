from enum import StrEnum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class PredictionLabel(StrEnum):
    SPOOF = "spoof"
    BONAFIDE = "bonafide"


class AudioMetadata(BaseModel):
    filename: str
    duration_sec: float
    sample_rate: int | None = None
    channels: int | None = None


class PredictionResult(BaseModel):
    label: PredictionLabel
    confidence: float = Field(ge=0, le=1)
    spoof_probability: float = Field(ge=0, le=1)
    bonafide_probability: float = Field(ge=0, le=1)


class WindowPrediction(PredictionResult):
    window_index: int
    start_sec: float
    end_sec: float


class ProcessingMetadata(BaseModel):
    model_name: str
    model_version: str
    fixed_window_size_sec: float
    processing_time_ms: int
    device: str | None = None


class AntiSpoofingResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    request_id: UUID
    audio: AudioMetadata
    windows: list[WindowPrediction]
    processing: ProcessingMetadata
    global_prediction: PredictionResult | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "unavailable"]
    model_loaded: bool
    model_name: str | None = None
    model_version: str | None = None


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
