import numpy as np
import pytest

from ki_geodaten.pipeline.modality_filter import (
    ModalityThresholds,
    compute_ndvi,
    filter_masks,
)
from ki_geodaten.pipeline.segmenter import MaskResult


def _mask_at(box, shape=(64, 64)):
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = box
    mask[r0:r1, c0:c1] = True
    return MaskResult(mask=mask, score=0.9, box_pixel=box)


def test_compute_ndvi_vegetation_signature_is_high():
    red = np.full((4, 4), 30, dtype=np.uint8)
    nir = np.full((4, 4), 200, dtype=np.uint8)
    ndvi = compute_ndvi(red, nir)
    assert ndvi.shape == (4, 4)
    assert ndvi.dtype == np.float32
    expected = (200 - 30) / (200 + 30)
    np.testing.assert_allclose(ndvi, expected, atol=1e-4)


def test_compute_ndvi_road_signature_is_low_or_negative():
    red = np.full((4, 4), 120, dtype=np.uint8)
    nir = np.full((4, 4), 100, dtype=np.uint8)
    ndvi = compute_ndvi(red, nir)
    assert ndvi.mean() < 0


def test_compute_ndvi_zero_pixels_yield_zero_no_nan():
    red = np.zeros((4, 4), dtype=np.uint8)
    nir = np.zeros((4, 4), dtype=np.uint8)
    ndvi = compute_ndvi(red, nir)
    assert np.all(ndvi == 0.0)
    assert not np.isnan(ndvi).any()


def test_compute_ndvi_shape_mismatch_raises():
    with pytest.raises(ValueError):
        compute_ndvi(np.zeros((4, 4)), np.zeros((4, 5)))


def test_thresholds_inactive_returns_inputs_unchanged():
    masks = [_mask_at((0, 0, 10, 10)), _mask_at((10, 10, 20, 20))]
    thresholds = ModalityThresholds()
    assert thresholds.is_active() is False
    out = filter_masks(masks, red_channel=None, nir_channel=None, ndsm=None, thresholds=thresholds)
    assert out == masks


def test_filter_keeps_only_high_ndvi_polygons():
    red = np.full((64, 64), 30, dtype=np.uint8)
    nir_low = np.full((64, 64), 30, dtype=np.uint8)  # NDVI = 0 -> non-veg
    nir = nir_low.copy()
    nir[20:30, 20:30] = 200  # vegetation patch
    masks = [
        _mask_at((20, 20, 30, 30)),  # over the vegetation -> keep
        _mask_at((40, 40, 50, 50)),  # outside the vegetation -> drop
    ]
    thresholds = ModalityThresholds(ndvi_min=0.3)
    out = filter_masks(masks, red_channel=red, nir_channel=nir, ndsm=None, thresholds=thresholds)
    assert len(out) == 1
    assert out[0].box_pixel == (20, 20, 30, 30)


def test_filter_keeps_only_tall_ndsm_polygons():
    ndsm = np.zeros((64, 64), dtype=np.float32)
    ndsm[20:30, 20:30] = 10.0  # 10 m tall structure
    masks = [_mask_at((20, 20, 30, 30)), _mask_at((40, 40, 50, 50))]
    thresholds = ModalityThresholds(ndsm_min=2.0)
    out = filter_masks(masks, red_channel=None, nir_channel=None, ndsm=ndsm, thresholds=thresholds)
    assert len(out) == 1
    assert out[0].box_pixel == (20, 20, 30, 30)


def test_filter_combined_ndvi_and_ndsm_must_both_pass():
    red = np.full((64, 64), 30, dtype=np.uint8)
    nir = np.full((64, 64), 30, dtype=np.uint8)
    nir[10:20, 10:20] = 200  # vegetated patch
    nir[40:50, 40:50] = 200  # second vegetated patch
    ndsm = np.zeros((64, 64), dtype=np.float32)
    ndsm[10:20, 10:20] = 5.0  # vegetated AND tall (tree-like)
    ndsm[40:50, 40:50] = 0.2  # vegetated but flat (grass)

    masks = [_mask_at((10, 10, 20, 20)), _mask_at((40, 40, 50, 50))]
    thresholds = ModalityThresholds(ndvi_min=0.3, ndsm_min=2.0)
    out = filter_masks(masks, red_channel=red, nir_channel=nir, ndsm=ndsm, thresholds=thresholds)
    assert len(out) == 1
    assert out[0].box_pixel == (10, 10, 20, 20)


def test_filter_skips_channel_when_data_missing():
    """If the user requests a filter but the channel isn't available, that
    bound is treated as 'pass' rather than 'reject everything'. This keeps
    the pipeline usable when nDSM service is briefly unreachable."""
    masks = [_mask_at((10, 10, 20, 20))]
    thresholds = ModalityThresholds(ndsm_min=2.0)
    out = filter_masks(masks, red_channel=None, nir_channel=None, ndsm=None, thresholds=thresholds)
    # nDSM channel not provided -> the bound is bypassed.
    assert out == masks


def test_filter_drops_empty_mask_defensively():
    empty = MaskResult(
        mask=np.zeros((64, 64), dtype=bool),
        score=0.5,
        box_pixel=(0, 0, 0, 0),
    )
    masks = [empty, _mask_at((10, 10, 20, 20))]
    ndsm = np.full((64, 64), 5.0, dtype=np.float32)
    thresholds = ModalityThresholds(ndsm_min=2.0)
    out = filter_masks(masks, red_channel=None, nir_channel=None, ndsm=ndsm, thresholds=thresholds)
    assert len(out) == 1
    assert out[0].box_pixel == (10, 10, 20, 20)


def test_thresholds_active_helpers():
    assert ModalityThresholds(ndvi_min=0.2).needs_nir() is True
    assert ModalityThresholds(ndsm_max=10.0).needs_ndsm() is True
    assert ModalityThresholds().needs_nir() is False
    assert ModalityThresholds().needs_ndsm() is False
