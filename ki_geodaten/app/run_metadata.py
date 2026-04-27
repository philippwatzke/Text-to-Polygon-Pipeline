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
    "WMS_URL",
    "WMS_LAYER",
    "WMS_VERSION",
    "WMS_FORMAT",
    "WMS_CRS",
    "WMS_MAX_PIXELS",
    "WMS_GRID_ORIGIN_X",
    "WMS_GRID_ORIGIN_Y",
    "SAM3_MODEL_ID",
    "TILE_SIZE",
    "MIN_POLYGON_AREA_M2",
    "LOCAL_MASK_NMS_IOU",
    "LOCAL_MASK_CONTAINMENT_RATIO",
    "SAFE_CENTER_NODATA_THRESHOLD",
    "MAX_PROMPT_CHARS",
    "MAX_ENCODER_CONTEXT_TOKENS",
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
