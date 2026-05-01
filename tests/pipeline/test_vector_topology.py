import math

import pytest
from shapely.geometry import Polygon

from ki_geodaten.pipeline.vector_topology import (
    clean_polygon_topology,
    orthogonalize_polygon,
)


def _longest_edge_angle(polygon: Polygon) -> float:
    coords = list(polygon.exterior.coords)[:-1]
    best_length = 0.0
    best_angle = 0.0
    for idx, start in enumerate(coords):
        end = coords[(idx + 1) % len(coords)]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length > best_length:
            best_length = length
            best_angle = math.atan2(dy, dx)
    return best_angle


def _edge_delta_to_right_angle(angle: float, reference: float) -> float:
    return abs(math.remainder(angle - reference, math.pi / 2))


def test_clean_polygon_topology_simplifies_jagged_edges():
    polygon = Polygon(
        [
            (0, 0),
            (3, 0.08),
            (6, -0.06),
            (10, 0),
            (10.04, 4),
            (10, 8),
            (5, 8.06),
            (0, 8),
            (-0.03, 4),
        ]
    )

    cleaned = clean_polygon_topology(
        polygon,
        simplify_tolerance_m=0.3,
        orthogonalize=False,
        orthogonalize_angle_tolerance_deg=12.0,
        orthogonalize_max_area_delta_ratio=0.25,
        orthogonalize_max_shift_m=2.0,
    )

    assert len(cleaned) == 1
    assert cleaned[0].is_valid
    assert len(cleaned[0].exterior.coords) < len(polygon.exterior.coords)
    assert cleaned[0].area == pytest.approx(polygon.area, rel=0.05)


def test_orthogonalize_polygon_snaps_near_rectangular_building():
    polygon = Polygon([(0, 0), (10, 0.4), (10.2, 5.1), (0.2, 4.8)])

    orthogonalized = orthogonalize_polygon(
        polygon,
        angle_tolerance_deg=15.0,
        max_area_delta_ratio=0.5,
        max_shift_m=2.0,
    )

    assert orthogonalized.is_valid
    assert not orthogonalized.equals_exact(polygon, 1e-9)
    reference = _longest_edge_angle(orthogonalized)
    coords = list(orthogonalized.exterior.coords)[:-1]
    for idx, start in enumerate(coords):
        end = coords[(idx + 1) % len(coords)]
        angle = math.atan2(end[1] - start[1], end[0] - start[0])
        assert _edge_delta_to_right_angle(angle, reference) < math.radians(0.01)


def test_orthogonalize_polygon_prefers_guarded_rectangle_for_boxy_polygon():
    polygon = Polygon(
        [
            (0, 0),
            (10, 0.2),
            (10.1, 5.0),
            (6.2, 5.0),
            (6.2, 4.7),
            (4.0, 4.7),
            (4.0, 5.0),
            (0.1, 4.9),
        ]
    )

    orthogonalized = orthogonalize_polygon(
        polygon,
        angle_tolerance_deg=12.0,
        max_area_delta_ratio=0.25,
        max_shift_m=2.0,
    )

    assert len(orthogonalized.exterior.coords) == 5
    assert orthogonalized.area == pytest.approx(polygon.area, rel=0.05)


def test_orthogonalize_polygon_keeps_l_shape_when_rectangle_guard_fails():
    polygon = Polygon([(0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10)])

    orthogonalized = orthogonalize_polygon(
        polygon,
        angle_tolerance_deg=12.0,
        max_area_delta_ratio=0.25,
        max_shift_m=2.0,
    )

    assert orthogonalized.equals_exact(polygon, 0.0)
def test_orthogonalize_polygon_rejects_excessive_shift():
    polygon = Polygon([(0, 0), (10, 0.4), (10.2, 5.1), (0.2, 4.8)])

    unchanged = orthogonalize_polygon(
        polygon,
        angle_tolerance_deg=15.0,
        max_area_delta_ratio=0.5,
        max_shift_m=0.01,
    )

    assert unchanged.equals_exact(polygon, 0.0)

