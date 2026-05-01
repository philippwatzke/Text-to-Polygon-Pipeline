from __future__ import annotations

import math

from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.geometry.polygon import orient
from shapely.validation import make_valid

_EPS = 1e-9


def _extract_polygons(geom) -> list[Polygon]:
    if isinstance(geom, Polygon):
        return [geom] if not geom.is_empty else []
    if isinstance(geom, MultiPolygon):
        return [polygon for polygon in geom.geoms if not polygon.is_empty]
    if isinstance(geom, GeometryCollection):
        out: list[Polygon] = []
        for subgeom in geom.geoms:
            out.extend(_extract_polygons(subgeom))
        return out
    return []


def _dedupe_ring(coords: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for coord in coords:
        point = (float(coord[0]), float(coord[1]))
        if not out or math.hypot(point[0] - out[-1][0], point[1] - out[-1][1]) > _EPS:
            out.append(point)
    if len(out) > 1 and math.hypot(out[0][0] - out[-1][0], out[0][1] - out[-1][1]) <= _EPS:
        out.pop()
    return out


def _dominant_orientation_rad(polygon: Polygon) -> float | None:
    coords = _dedupe_ring(list(polygon.exterior.coords))
    if len(coords) < 3:
        return None

    best_length = 0.0
    best_angle = None
    for idx, start in enumerate(coords):
        end = coords[(idx + 1) % len(coords)]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length <= best_length:
            continue
        best_length = length
        best_angle = math.atan2(dy, dx)

    if best_angle is None:
        return None
    return math.remainder(best_angle, math.pi / 2)


def _rotate_point(
    point: tuple[float, float],
    origin: tuple[float, float],
    angle_rad: float,
) -> tuple[float, float]:
    x = point[0] - origin[0]
    y = point[1] - origin[1]
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    return (
        origin[0] + x * cos_a - y * sin_a,
        origin[1] + x * sin_a + y * cos_a,
    )


def _classify_segment(
    start: tuple[float, float],
    end: tuple[float, float],
    tan_tolerance: float,
) -> tuple[str, float] | None:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    if math.hypot(dx, dy) <= _EPS:
        return None
    if abs(dy) <= tan_tolerance * max(abs(dx), _EPS):
        return ("h", (start[1] + end[1]) / 2.0)
    if abs(dx) <= tan_tolerance * max(abs(dy), _EPS):
        return ("v", (start[0] + end[0]) / 2.0)
    return None


def _merge_axis_value(
    current: float,
    previous: tuple[str, float] | None,
    next_: tuple[str, float] | None,
    axis: str,
) -> float:
    values = []
    if previous is not None and previous[0] == axis:
        values.append(previous[1])
    if next_ is not None and next_[0] == axis:
        values.append(next_[1])
    if not values:
        return current
    return sum(values) / len(values)


def _orthogonalize_ring(
    coords: list[tuple[float, float]],
    origin: tuple[float, float],
    orientation_rad: float,
    tan_tolerance: float,
) -> list[tuple[float, float]] | None:
    points = _dedupe_ring(coords)
    if len(points) < 3:
        return None

    rotated = [_rotate_point(point, origin, -orientation_rad) for point in points]
    segment_axes = [
        _classify_segment(rotated[idx], rotated[(idx + 1) % len(rotated)], tan_tolerance)
        for idx in range(len(rotated))
    ]
    adjusted = []
    for idx, point in enumerate(rotated):
        previous_axis = segment_axes[idx - 1]
        next_axis = segment_axes[idx]
        x = _merge_axis_value(point[0], previous_axis, next_axis, "v")
        y = _merge_axis_value(point[1], previous_axis, next_axis, "h")
        adjusted.append(_rotate_point((x, y), origin, orientation_rad))

    adjusted = _dedupe_ring(adjusted)
    if len(adjusted) < 3:
        return None
    adjusted.append(adjusted[0])
    return adjusted


def _guarded_single_polygon(
    original: Polygon,
    candidate,
    *,
    max_area_delta_ratio: float,
    max_shift_m: float,
) -> Polygon | None:
    valid = make_valid(candidate)
    polygons = _extract_polygons(valid)
    if len(polygons) != 1:
        return None
    polygon = orient(polygons[0], sign=1.0)
    if polygon.is_empty or polygon.area <= 0.0:
        return None
    area_delta = abs(polygon.area - original.area) / original.area
    if area_delta > max_area_delta_ratio:
        return None
    if original.hausdorff_distance(polygon) > max_shift_m:
        return None
    return polygon


def _candidate_rank(original: Polygon, candidate: Polygon) -> tuple[int, float, float]:
    vertex_count = len(candidate.exterior.coords)
    shift = original.hausdorff_distance(candidate)
    area_delta = abs(candidate.area - original.area) / original.area
    return (vertex_count, shift, area_delta)


def orthogonalize_polygon(
    polygon: Polygon,
    *,
    angle_tolerance_deg: float,
    max_area_delta_ratio: float,
    max_shift_m: float,
) -> Polygon:
    """Inputs: Polygon and numeric limits. Logic: snap near-right-angle rings and optionally prefer a guarded oriented rectangle candidate. Returns: Polygon."""
    if polygon.is_empty or polygon.area <= 0.0:
        return polygon
    orientation_rad = _dominant_orientation_rad(polygon)
    if orientation_rad is None:
        return polygon

    origin = (polygon.centroid.x, polygon.centroid.y)
    tan_tolerance = math.tan(math.radians(angle_tolerance_deg))
    exterior = _orthogonalize_ring(
        list(polygon.exterior.coords),
        origin,
        orientation_rad,
        tan_tolerance,
    )
    if exterior is None:
        return polygon

    interiors = []
    for interior in polygon.interiors:
        ring = _orthogonalize_ring(list(interior.coords), origin, orientation_rad, tan_tolerance)
        if ring is not None:
            interiors.append(ring)

    candidates: list[Polygon] = []
    snapped = _guarded_single_polygon(
        polygon,
        Polygon(exterior, interiors),
        max_area_delta_ratio=max_area_delta_ratio,
        max_shift_m=max_shift_m,
    )
    if snapped is not None:
        candidates.append(snapped)

    if not polygon.interiors:
        rectangle = _guarded_single_polygon(
            polygon,
            polygon.minimum_rotated_rectangle,
            max_area_delta_ratio=max_area_delta_ratio,
            max_shift_m=max_shift_m,
        )
        if rectangle is not None:
            candidates.append(rectangle)

    if not candidates:
        return polygon
    return min(candidates, key=lambda candidate: _candidate_rank(polygon, candidate))


def clean_polygon_topology(
    polygon: Polygon,
    *,
    simplify_tolerance_m: float,
    orthogonalize: bool,
    orthogonalize_angle_tolerance_deg: float,
    orthogonalize_max_area_delta_ratio: float,
    orthogonalize_max_shift_m: float,
) -> list[Polygon]:
    """Inputs: Polygon and topology options. Logic: validate, simplify, and optionally orthogonalize polygon topology. Returns: list[Polygon]."""
    if polygon.is_empty or polygon.area <= 0.0:
        return []

    current = make_valid(polygon)
    polygons = _extract_polygons(current)
    if simplify_tolerance_m > 0.0:
        simplified: list[Polygon] = []
        for item in polygons:
            simplified_geom = make_valid(
                item.simplify(simplify_tolerance_m, preserve_topology=True)
            )
            simplified.extend(_extract_polygons(simplified_geom))
        polygons = simplified

    if orthogonalize:
        polygons = [
            orthogonalize_polygon(
                item,
                angle_tolerance_deg=orthogonalize_angle_tolerance_deg,
                max_area_delta_ratio=orthogonalize_max_area_delta_ratio,
                max_shift_m=orthogonalize_max_shift_m,
            )
            for item in polygons
        ]

    out: list[Polygon] = []
    for item in polygons:
        valid = make_valid(item)
        out.extend(_extract_polygons(valid))
    return [item for item in out if not item.is_empty and item.area > 0.0]
