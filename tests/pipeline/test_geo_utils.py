import pytest
from ki_geodaten.pipeline.geo_utils import (
    snap_floor, snap_ceil, transform_bbox_wgs84_to_utm, pixel_count,
)

def test_snap_floor_fp_edge_case():
    # 0.6 / 0.2 == 2.9999... in IEEE-754; naive floor would give 0.4
    assert snap_floor(0.6, origin=0.0, step=0.2) == pytest.approx(0.6)
    assert snap_floor(1.3, origin=0.0, step=0.2) == pytest.approx(1.2)
    assert snap_floor(600000.15, origin=0.0, step=0.2) == pytest.approx(600000.0)

def test_snap_ceil_fp_edge_case():
    assert snap_ceil(0.55, origin=0.0, step=0.2) == pytest.approx(0.6)
    assert snap_ceil(600000.01, origin=0.0, step=0.2) == pytest.approx(600000.2)

def test_snap_with_nonzero_origin():
    assert snap_floor(0.45, origin=0.1, step=0.2) == pytest.approx(0.3)
    assert snap_ceil(0.45, origin=0.1, step=0.2) == pytest.approx(0.5)

def test_transform_bbox_uses_densify():
    minx, miny, maxx, maxy = transform_bbox_wgs84_to_utm(11.0, 48.0, 11.1, 48.1)
    assert 640_000 < minx < 700_000
    assert 5_300_000 < miny < 5_400_000
    assert maxx > minx
    assert maxy > miny

def test_pixel_count_round_not_int():
    assert pixel_count(minx=0.0, maxx=300.0, step=0.2) == 1500
    assert pixel_count(minx=0.0, maxx=204.8, step=0.2) == 1024
