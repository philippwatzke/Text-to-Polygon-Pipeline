from __future__ import annotations

from typing import Iterable

import geopandas as gpd
from rasterio.features import shapes
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.geometry import shape as shapely_shape
from shapely.ops import unary_union
from shapely.validation import make_valid

from ki_geodaten.pipeline.segmenter import MaskResult
from ki_geodaten.pipeline.tiler import Tile


def keep_center_only(masks: Iterable[MaskResult], tile: Tile) -> list[MaskResult]:
    margin = tile.center_margin
    lo = margin
    hi = tile.size - margin
    out: list[MaskResult] = []
    for mask in masks:
        r0, c0, r1, c1 = mask.box_pixel
        center_r = (r0 + r1) / 2
        center_c = (c0 + c1) / 2
        if lo <= center_r < hi and lo <= center_c < hi:
            out.append(mask)
    return out


def extract_polygons(geom) -> list[Polygon]:
    if isinstance(geom, Polygon):
        return [geom] if not geom.is_empty else []
    if isinstance(geom, MultiPolygon):
        return [polygon for polygon in geom.geoms if not polygon.is_empty]
    if isinstance(geom, GeometryCollection):
        out: list[Polygon] = []
        for subgeom in geom.geoms:
            out.extend(extract_polygons(subgeom))
        return out
    return []


def masks_to_polygons(
    masks: list[MaskResult],
    tile: Tile,
    *,
    min_area_m2: float,
) -> gpd.GeoDataFrame:
    records: list[dict] = []
    geometries: list[Polygon] = []

    for mask_result in masks:
        mask_u8 = mask_result.mask.astype("uint8")
        mask_polygons: list[Polygon] = []
        for geom_dict, value in shapes(
            mask_u8,
            mask=mask_result.mask,
            transform=tile.affine,
            connectivity=8,
        ):
            if value != 1:
                continue
            valid_geom = make_valid(shapely_shape(geom_dict))
            mask_polygons.extend(extract_polygons(valid_geom))

        if not mask_polygons:
            continue
        # connectivity=8 treats diagonally-touching pixels as connected, but
        # rasterio.features.shapes still emits two OGC-simple polygons that
        # share a single corner vertex (self-touch is forbidden in a single
        # polygon). unary_union + a tiny buffer/un-buffer collapses that
        # corner contact into one polygon, so a single SAM mask stays a
        # single detection record.
        merged_geom = unary_union(mask_polygons)
        if isinstance(merged_geom, MultiPolygon):
            merged_geom = merged_geom.buffer(1e-6).buffer(-1e-6)
        if not merged_geom.is_valid:
            merged_geom = make_valid(merged_geom)
        for polygon in extract_polygons(merged_geom):
            if polygon.area < min_area_m2:
                continue
            geometries.append(polygon)
            records.append(
                {
                    "score": float(mask_result.score),
                    "source_tile_row": tile.tile_row,
                    "source_tile_col": tile.tile_col,
                }
            )

    return gpd.GeoDataFrame(records, geometry=geometries, crs="EPSG:25832")
