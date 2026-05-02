from __future__ import annotations

import geopandas as gpd
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.validation import make_valid

from ki_geodaten.models import VectorOptions

ORTHOGONALIZE_MIN_FILL_RATIO = 0.72


def _extract_polygons(geom: BaseGeometry) -> list[Polygon]:
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    if isinstance(geom, GeometryCollection):
        out: list[Polygon] = []
        for part in geom.geoms:
            out.extend(_extract_polygons(part))
        return out
    return []


def _coerce_options(options: VectorOptions | dict | None) -> VectorOptions:
    if options is None:
        return VectorOptions()
    if isinstance(options, VectorOptions):
        return options
    return VectorOptions.model_validate(options)


def _is_active(options: VectorOptions) -> bool:
    return bool(options.orthogonalize or (options.simplification_tolerance_m or 0) > 0)


def _orthogonalized_candidate(polygon: Polygon) -> Polygon:
    rectangle = polygon.minimum_rotated_rectangle
    if rectangle.is_empty or rectangle.area <= 0:
        return polygon
    fill_ratio = polygon.area / rectangle.area
    if fill_ratio < ORTHOGONALIZE_MIN_FILL_RATIO:
        return polygon
    return rectangle


def _cleanup_geometry(geom: BaseGeometry, options: VectorOptions) -> list[Polygon]:
    if geom.is_empty:
        return []
    working = geom
    if options.simplification_tolerance_m and options.simplification_tolerance_m > 0:
        working = working.simplify(
            options.simplification_tolerance_m,
            preserve_topology=True,
        )
    working = make_valid(working)
    if working.is_empty:
        return []

    polygons: list[Polygon] = []
    for polygon in _extract_polygons(working):
        candidate = _orthogonalized_candidate(polygon) if options.orthogonalize else polygon
        candidate = make_valid(candidate)
        polygons.extend(_extract_polygons(candidate))
    return [polygon for polygon in polygons if not polygon.is_empty and polygon.area > 0]


def apply_vector_cleanup(
    gdf: gpd.GeoDataFrame,
    options: VectorOptions | dict | None,
) -> gpd.GeoDataFrame:
    options = _coerce_options(options)
    if len(gdf) == 0 or not _is_active(options):
        return gdf

    rows: list[dict] = []
    geometries: list[Polygon] = []
    for _, rec in gdf.iterrows():
        for polygon in _cleanup_geometry(rec.geometry, options):
            rows.append({key: rec[key] for key in rec.index if key != "geometry"})
            geometries.append(polygon)

    if not geometries:
        return gpd.GeoDataFrame(
            {column: [] for column in gdf.columns if column != "geometry"},
            geometry=[],
            crs=gdf.crs,
        )
    return gpd.GeoDataFrame(rows, geometry=geometries, crs=gdf.crs)
