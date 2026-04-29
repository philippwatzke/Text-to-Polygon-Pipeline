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

MASK_COMPONENT_MERGE_GAP_M = 0.8


def keep_center_only(masks: Iterable[MaskResult], tile: Tile) -> list[MaskResult]:
    margin = tile.center_margin
    lo = margin
    hi = tile.size - margin
    out: list[MaskResult] = []
    for mask in masks:
        r0, c0, r1, c1 = mask.box_pixel
        center_r = (r0 + r1) / 2
        center_c = (c0 + c1) / 2
        if tile.ownership_bounds_pixel is not None:
            owner_r0, owner_r1, owner_c0, owner_c1 = tile.ownership_bounds_pixel
            abs_r = tile.pixel_origin[0] + center_r
            abs_c = tile.pixel_origin[1] + center_c
            if owner_r0 <= abs_r < owner_r1 and owner_c0 <= abs_c < owner_c1:
                out.append(mask)
            continue
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


def _polygon_iou(a: Polygon, b: Polygon) -> float:
    inter_area = a.intersection(b).area
    if inter_area <= 0.0:
        return 0.0
    union_area = a.area + b.area - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def _polygon_containment(a: Polygon, b: Polygon) -> float:
    inter_area = a.intersection(b).area
    if inter_area <= 0.0:
        return 0.0
    min_area = min(a.area, b.area)
    if min_area <= 0.0:
        return 0.0
    return inter_area / min_area


def _buffered_coverage(fragment: Polygon, base: Polygon, buffer_m: float) -> float:
    if fragment.area <= 0.0:
        return 0.0
    base_geom = base.buffer(buffer_m) if buffer_m > 0.0 else base
    covered_area = fragment.intersection(base_geom).area
    if covered_area <= 0.0:
        return 0.0
    return covered_area / fragment.area


def _is_fragment_of(
    fragment: Polygon,
    base: Polygon,
    *,
    coverage_ratio: float,
    max_area_ratio: float,
    buffer_m: float,
) -> bool:
    if fragment.area <= 0.0 or base.area <= 0.0:
        return False
    if fragment.area / base.area > max_area_ratio:
        return False
    return _buffered_coverage(fragment, base, buffer_m) >= coverage_ratio


def global_polygon_nms(
    gdf: gpd.GeoDataFrame,
    *,
    iou_threshold: float,
    containment_ratio: float,
    fragment_coverage_ratio: float = 1.0,
    fragment_max_area_ratio: float = 0.0,
    fragment_buffer_m: float = 0.0,
) -> gpd.GeoDataFrame:
    """Suppress duplicate final polygons after tile merge.

    `local_mask_nms` removes duplicate masks inside one SAM tile. This second
    NMS stage operates in map coordinates across the whole job, after
    center-ownership clipping and multimodal filtering. It keeps the highest
    scoring polygon and suppresses lower-priority polygons when either their
    IoU is high or the smaller polygon is almost contained in the larger one.
    The optional fragment rule catches low-IoU artifacts that lie mostly on a
    larger survivor after a small metric buffer, which is common on roof edges.
    """
    if gdf is None or len(gdf) <= 1:
        return gdf
    if iou_threshold >= 1.0 and containment_ratio >= 1.0:
        return gdf

    work = gdf.copy().reset_index(drop=True)
    work["_nms_area"] = work.geometry.area
    work["_nms_original_pos"] = range(len(work))
    ordered = work.sort_values(
        by=[
            "score",
            "_nms_area",
            "source_tile_row",
            "source_tile_col",
            "_nms_original_pos",
        ],
        ascending=[False, False, True, True, True],
        kind="mergesort",
    )
    spatial_index = work.sindex

    kept_positions: list[int] = []
    kept_position_set: set[int] = set()
    for idx, row in ordered.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty or geom.area <= 0.0:
            continue

        drop = False
        remove_positions: list[int] = []
        query_geom = geom.buffer(fragment_buffer_m) if fragment_buffer_m > 0.0 else geom
        candidate_positions = spatial_index.query(query_geom, predicate="intersects")
        for existing_pos in candidate_positions:
            existing_pos = int(existing_pos)
            if existing_pos not in kept_position_set:
                continue

            existing = work.geometry.iloc[existing_pos]
            if _polygon_iou(geom, existing) >= iou_threshold:
                drop = True
                break
            containment = _polygon_containment(geom, existing)
            if containment >= containment_ratio and geom.area <= existing.area:
                drop = True
                break
            if containment >= containment_ratio:
                remove_positions.append(existing_pos)
                continue

            if _is_fragment_of(
                geom,
                existing,
                coverage_ratio=fragment_coverage_ratio,
                max_area_ratio=fragment_max_area_ratio,
                buffer_m=fragment_buffer_m,
            ):
                drop = True
                break
            if _is_fragment_of(
                existing,
                geom,
                coverage_ratio=fragment_coverage_ratio,
                max_area_ratio=fragment_max_area_ratio,
                buffer_m=fragment_buffer_m,
            ):
                remove_positions.append(existing_pos)

        if not drop:
            if remove_positions:
                remove_set = set(remove_positions)
                kept_positions = [
                    position for position in kept_positions if position not in remove_set
                ]
                kept_position_set.difference_update(remove_set)
            kept_positions.append(idx)
            kept_position_set.add(idx)

    return work.loc[kept_positions].drop(columns=["_nms_area", "_nms_original_pos"])


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
            merged_geom = merged_geom.buffer(MASK_COMPONENT_MERGE_GAP_M / 2).buffer(
                -MASK_COMPONENT_MERGE_GAP_M / 2
            )
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
