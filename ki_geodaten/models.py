# ki_geodaten/models.py
from __future__ import annotations
from enum import StrEnum
from typing import Literal
from pydantic import BaseModel, Field, model_validator

class TilePreset(StrEnum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"

class JobStatus(StrEnum):
    PENDING = "PENDING"
    DOWNLOADING = "DOWNLOADING"
    INFERRING = "INFERRING"
    READY_FOR_REVIEW = "READY_FOR_REVIEW"
    EXPORTED = "EXPORTED"
    FAILED = "FAILED"

class ErrorReason(StrEnum):
    DOP_TIMEOUT = "DOP_TIMEOUT"
    DOP_HTTP_ERROR = "DOP_HTTP_ERROR"
    OOM = "OOM"
    INFERENCE_ERROR = "INFERENCE_ERROR"
    WORKER_RESTARTED = "WORKER_RESTARTED"
    EXPORT_ERROR = "EXPORT_ERROR"
    INVALID_GEOMETRY = "INVALID_GEOMETRY"

class NoDataReason(StrEnum):
    OOM = "OOM"
    INFERENCE_ERROR = "INFERENCE_ERROR"
    INVALID_GEOMETRY = "INVALID_GEOMETRY"
    NODATA_PIXELS = "NODATA_PIXELS"

Validation = Literal["ACCEPTED", "REJECTED"]

class BBox(BaseModel):
    minx: float
    miny: float
    maxx: float
    maxy: float

    @model_validator(mode="after")
    def _check_ordered(self) -> "BBox":
        if self.minx >= self.maxx:
            raise ValueError("minx must be < maxx")
        if self.miny >= self.maxy:
            raise ValueError("miny must be < maxy")
        return self

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.minx, self.miny, self.maxx, self.maxy)

class CreateJobRequest(BaseModel):
    prompt: str = Field(min_length=1)
    bbox_wgs84: list[float] = Field(min_length=4, max_length=4)
    tile_preset: TilePreset = TilePreset.MEDIUM

class ValidationUpdate(BaseModel):
    pid: int
    validation: Validation

class ValidateBulkRequest(BaseModel):
    updates: list[ValidationUpdate]
