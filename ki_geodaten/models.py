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

class ModalityFilter(BaseModel):
    """Per-job thresholds for post-segmentation modality filtering.

    Each bound is optional. If both bounds are None, the channel is not
    used as a filter. The pipeline computes a per-mask aggregate (mean over
    the mask pixels) and keeps the polygon iff every active bound passes.
    NDVI is derived from DOP20 band 4 (NIR); nDSM is derived locally from
    OpenData DOM/DGM tiles and is therefore additionally gated by tile
    availability.
    """
    ndvi_min: float | None = Field(default=None, ge=-1.0, le=1.0)
    ndvi_max: float | None = Field(default=None, ge=-1.0, le=1.0)
    ndsm_min: float | None = Field(default=None)
    ndsm_max: float | None = Field(default=None)

    @model_validator(mode="after")
    def _check_ranges(self) -> "ModalityFilter":
        if (
            self.ndvi_min is not None
            and self.ndvi_max is not None
            and self.ndvi_min > self.ndvi_max
        ):
            raise ValueError("ndvi_min must be <= ndvi_max")
        if (
            self.ndsm_min is not None
            and self.ndsm_max is not None
            and self.ndsm_min > self.ndsm_max
        ):
            raise ValueError("ndsm_min must be <= ndsm_max")
        return self

    def is_active(self) -> bool:
        return any(
            v is not None
            for v in (self.ndvi_min, self.ndvi_max, self.ndsm_min, self.ndsm_max)
        )

    def needs_nir(self) -> bool:
        return self.ndvi_min is not None or self.ndvi_max is not None

    def needs_ndsm(self) -> bool:
        return self.ndsm_min is not None or self.ndsm_max is not None


class CreateJobRequest(BaseModel):
    prompt: str = Field(min_length=1)
    bbox_wgs84: list[float] = Field(min_length=4, max_length=4)
    tile_preset: TilePreset = TilePreset.MEDIUM
    modality_filter: ModalityFilter = Field(default_factory=ModalityFilter)

class ValidationUpdate(BaseModel):
    pid: int
    validation: Validation

class ValidateBulkRequest(BaseModel):
    updates: list[ValidationUpdate]

class MissedEstimateRequest(BaseModel):
    missed_estimate: int | None = Field(default=None, ge=0)

class JobLabelRequest(BaseModel):
    label: str | None = Field(default=None, max_length=120)

    @model_validator(mode="after")
    def _normalize_label(self) -> "JobLabelRequest":
        if self.label is not None:
            stripped = self.label.strip()
            self.label = stripped or None
        return self

class MissedObjectRequest(BaseModel):
    lon: float
    lat: float
