"""Post-segmentation modality filtering (NDVI, nDSM).

SAM 3 sees only RGB. To reduce false positives that can be ruled out by
auxiliary modalities, we compute per-mask aggregates and gate polygons by
configurable thresholds:

- **NDVI** (Normalized Difference Vegetation Index) from DOP20 band 4 (NIR)
  and band 1 (Red). Useful to require vegetation (e.g. ``ndvi_min=0.3`` for
  ``tree``) or non-vegetation (``ndvi_max=0.1`` for ``parking lot``).

- **nDSM** (height above ground in metres) derived locally from DOM-DGM tiles.
  Useful to require height (e.g. ``ndsm_min=2.0`` for ``building``) or
  flatness (``ndsm_max=0.5`` for ``solar panel on roof`` is wrong, but
  ``ndsm_max=0.5`` for ``parking lot`` works).

Filter semantics: a polygon is kept iff EVERY active bound passes. The
filter operates pixel-wise on the SAM mask shape and then takes the mean
over the mask area. Empty masks are dropped defensively (they should
already have been filtered upstream).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ki_geodaten.pipeline.segmenter import MaskResult


@dataclass(frozen=True)
class ModalityThresholds:
    """Job-scoped thresholds. Mirrors models.ModalityFilter on the pipeline side."""
    ndvi_min: float | None = None
    ndvi_max: float | None = None
    ndsm_min: float | None = None
    ndsm_max: float | None = None

    def is_active(self) -> bool:
        return any(
            v is not None
            for v in (self.ndvi_min, self.ndvi_max, self.ndsm_min, self.ndsm_max)
        )

    def needs_nir(self) -> bool:
        return self.ndvi_min is not None or self.ndvi_max is not None

    def needs_ndsm(self) -> bool:
        return self.ndsm_min is not None or self.ndsm_max is not None


def compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """NDVI = (NIR - R) / (NIR + R). Returns float32 in [-1, 1].

    Both inputs must be the same shape. Pixels where ``NIR + R == 0`` (a
    perfectly black sample, e.g. NoData) get NDVI = 0 — neutral, neither
    flagged as vegetation nor as non-vegetation. The caller is responsible
    for masking those pixels separately if they want stricter handling.

    The 8-bit precision of DOP20 means NDVI has effective steps of about
    1/256 ≈ 0.004 in the dynamic range, which is well below the typical
    decision threshold (NDVI ~0.2 distinguishes vegetation from soil).
    """
    if red.shape != nir.shape:
        raise ValueError(f"red {red.shape} and nir {nir.shape} must match")
    red_f = red.astype(np.float32, copy=False)
    nir_f = nir.astype(np.float32, copy=False)
    denominator = nir_f + red_f
    safe_denominator = np.where(denominator == 0.0, 1.0, denominator)
    ndvi = (nir_f - red_f) / safe_denominator
    ndvi = np.where(denominator == 0.0, 0.0, ndvi)
    return ndvi.astype(np.float32, copy=False)


def _mean_over_mask(channel: np.ndarray, mask: np.ndarray) -> float | None:
    """Return mean of ``channel`` over True-pixels of ``mask``, or None if empty."""
    if mask.shape != channel.shape:
        raise ValueError(
            f"channel {channel.shape} and mask {mask.shape} must match"
        )
    pixel_count = int(mask.sum())
    if pixel_count == 0:
        return None
    values = channel[mask]
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(finite.mean())


def _value_in_range(
    value: float | None,
    lower: float | None,
    upper: float | None,
) -> bool:
    """Return True iff value passes the (optional) bounds. None value -> reject."""
    if value is None:
        return False
    if lower is not None and value < lower:
        return False
    if upper is not None and value > upper:
        return False
    return True


def filter_masks(
    masks: list[MaskResult],
    *,
    red_channel: np.ndarray | None,
    nir_channel: np.ndarray | None,
    ndsm: np.ndarray | None,
    thresholds: ModalityThresholds,
) -> list[MaskResult]:
    """Apply the modality thresholds to a list of MaskResult.

    Channels are tile-shaped 2D arrays in the same coordinate system as the
    masks. Pass ``None`` for channels that are not available for this job
    (the corresponding thresholds will then be skipped, not failed — keeps
    the pipeline robust when a channel is missing).
    """
    if not thresholds.is_active():
        return list(masks)

    ndvi: np.ndarray | None = None
    if thresholds.needs_nir() and red_channel is not None and nir_channel is not None:
        ndvi = compute_ndvi(red_channel, nir_channel)

    kept: list[MaskResult] = []
    for mask_result in masks:
        mask = mask_result.mask
        if not mask.any():
            continue

        mean_ndvi: float | None = None
        mean_ndsm: float | None = None
        if thresholds.needs_nir() and ndvi is not None:
            mean_ndvi = _mean_over_mask(ndvi, mask)
            if not _value_in_range(mean_ndvi, thresholds.ndvi_min, thresholds.ndvi_max):
                continue

        if thresholds.needs_ndsm() and ndsm is not None:
            mean_ndsm = _mean_over_mask(ndsm, mask)
            if not _value_in_range(mean_ndsm, thresholds.ndsm_min, thresholds.ndsm_max):
                continue

        kept.append(
            MaskResult(
                mask=mask_result.mask,
                score=mask_result.score,
                box_pixel=mask_result.box_pixel,
                ndvi_mean=mean_ndvi,
                ndsm_mean=mean_ndsm,
            )
        )
    return kept
