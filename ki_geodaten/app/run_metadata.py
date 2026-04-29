"""Per-job run metadata snapshot for reproducibility (Spec §13 follow-up).

The webserver fills `jobs.run_metadata` at job creation time with a snapshot
of all settings and code state that influence the SAM 3 segmentation result.
This way each row in the database fully self-describes the experimental
condition that produced its polygons — no need to cross-reference git history
or .env files months later when writing up results.
"""
from __future__ import annotations

import os
import subprocess
from functools import lru_cache
from importlib import metadata
from pathlib import Path

from ki_geodaten.config import Settings
from ki_geodaten.models import TilePreset

# Settings fields that materially affect the produced polygons. Anything that
# only influences UX (poll intervals, retention) is intentionally excluded so
# that two jobs with otherwise identical metadata are guaranteed to be
# scientifically comparable.
_RELEVANT_SETTINGS_FIELDS = (
    "DOP_SOURCE",
    "WCS_URL",
    "WCS_VERSION",
    "WCS_COVERAGE_ID",
    "WCS_FORMAT",
    "WCS_CRS",
    "WCS_MAX_PIXELS",
    "WCS_GRID_ORIGIN_X",
    "WCS_GRID_ORIGIN_Y",
    "FILL_WCS_RGB_ZERO_WITH_WMS",
    "WMS_URL",
    "WMS_LAYER",
    "WMS_VERSION",
    "WMS_FORMAT",
    "SAM3_MODEL_ID",
    "SAM3_LOCAL_FILES_ONLY",
    "TILE_SIZE",
    "MIN_POLYGON_AREA_M2",
    "LOCAL_MASK_NMS_IOU",
    "LOCAL_MASK_CONTAINMENT_RATIO",
    "GLOBAL_POLYGON_NMS_IOU",
    "GLOBAL_POLYGON_CONTAINMENT_RATIO",
    "GLOBAL_POLYGON_FRAGMENT_COVERAGE_RATIO",
    "GLOBAL_POLYGON_FRAGMENT_MAX_AREA_RATIO",
    "GLOBAL_POLYGON_FRAGMENT_BUFFER_M",
    "SAM_IMAGE_PREPROCESS",
    "SAM_CLAHE_CLIP_LIMIT",
    "SAM_CLAHE_TILE_GRID_SIZE",
    "SAM_CLAHE_BLEND",
    "SAM_IMAGE_GAMMA",
    "SAM_IMAGE_BRIGHTNESS",
    "SAFE_CENTER_NODATA_THRESHOLD",
    "MAX_PROMPT_CHARS",
    "MAX_ENCODER_CONTEXT_TOKENS",
    "MODALITY_USE_NDVI",
    "MODALITY_USE_NDSM",
    "DGM_METALINK_URL",
    "DGM_TILE_CACHE_DIR",
    "DGM_NATIVE_STEP_M",
    "DOM_METALINK_URL",
    "DOM_TILE_CACHE_DIR",
    "DOM_NATIVE_STEP_M",
)


@lru_cache(maxsize=1)
def _git_commit_sha() -> str | None:
    """Best-effort git HEAD lookup; returns None outside a git checkout."""
    repo_root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return None
    if result.returncode != 0:
        return None
    sha = result.stdout.strip()
    return sha or None


@lru_cache(maxsize=1)
def _package_version() -> str | None:
    try:
        return metadata.version("ki-geodaten")
    except metadata.PackageNotFoundError:
        return None


def _settings_snapshot(settings: Settings) -> dict:
    snapshot: dict = {}
    for field in _RELEVANT_SETTINGS_FIELDS:
        value = getattr(settings, field, None)
        if isinstance(value, Path):
            value = str(value)
        snapshot[field] = value
    return snapshot


def build_run_metadata(
    settings: Settings,
    *,
    tile_preset: TilePreset,
    extra: dict | None = None,
) -> dict:
    """Capture the full reproducibility snapshot for one job.

    Includes git SHA, package version, environment-relevant CUDA/Python info
    (when cheaply available) and the subset of Settings that influences the
    pipeline output. Pass `extra` to merge in caller-specific fields like a
    runtime-resolved score threshold.
    """
    metadata_dict: dict = {
        "package_version": _package_version(),
        "git_commit_sha": _git_commit_sha(),
        "tile_preset": str(tile_preset),
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "settings": _settings_snapshot(settings),
    }
    if extra:
        metadata_dict.update(extra)
    return metadata_dict
