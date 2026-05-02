from __future__ import annotations

import json
from typing import Iterable

import shapely
from shapely.geometry import box, mapping
from shapely.ops import transform as shapely_transform
from shapely.validation import make_valid
from shapely.wkb import loads as wkb_loads

from ki_geodaten.pipeline.geo_utils import transformer_25832_to_4326
from ki_geodaten.pipeline.merger import extract_polygons

_PRECISION_DEG = 1e-6


def _transform_to_4326(geom):
    transformer = transformer_25832_to_4326()
    return shapely_transform(lambda x, y, z=None: transformer.transform(x, y), geom)


def _clip_transform_precision(geom, aoi_utm_polygon):
    clipped = geom.intersection(aoi_utm_polygon)
    if clipped.is_empty:
        return []
    clipped = make_valid(clipped)
    out = []
    for polygon in extract_polygons(clipped):
        transformed = _transform_to_4326(polygon)
        rounded = shapely.set_precision(transformed, grid_size=_PRECISION_DEG)
        rounded = make_valid(rounded)
        out.extend(extract_polygons(rounded))
    return out


def _feature_collection(features: list[dict]) -> dict:
    return {"type": "FeatureCollection", "features": features}


def build_polygons_feature_collection(
    rows: Iterable[dict],
    *,
    aoi_utm: tuple[float, float, float, float],
    clip_utm: tuple[float, float, float, float] | None = None,
) -> dict:
    aoi = box(*aoi_utm)
    clip = aoi if clip_utm is None else aoi.intersection(box(*clip_utm))
    features: list[dict] = []
    for row in rows:
        geom = wkb_loads(row["geometry_wkb"])
        if clip.is_empty or not geom.intersects(clip):
            continue
        for transformed in _clip_transform_precision(geom, clip):
            properties = {
                "id": row["id"],
                "score": row["score"],
                "validation": row["validation"],
            }
            if row.get("ndvi_mean") is not None:
                properties["ndvi_mean"] = row["ndvi_mean"]
            if row.get("ndsm_mean") is not None:
                properties["ndsm_mean"] = row["ndsm_mean"]
            if row.get("source_tile_row") is not None:
                properties["source_tile_row"] = row["source_tile_row"]
            if row.get("source_tile_col") is not None:
                properties["source_tile_col"] = row["source_tile_col"]
            features.append(
                {
                    "type": "Feature",
                    "geometry": mapping(transformed),
                    "properties": properties,
                }
            )
    return _feature_collection(features)


def build_polygons_geojson(
    rows: Iterable[dict],
    *,
    aoi_utm: tuple[float, float, float, float],
    clip_utm: tuple[float, float, float, float] | None = None,
) -> str:
    return json.dumps(
        build_polygons_feature_collection(rows, aoi_utm=aoi_utm, clip_utm=clip_utm)
    )


def build_nodata_feature_collection(
    rows: Iterable[dict],
    *,
    aoi_utm: tuple[float, float, float, float],
    clip_utm: tuple[float, float, float, float] | None = None,
) -> dict:
    aoi = box(*aoi_utm)
    clip = aoi if clip_utm is None else aoi.intersection(box(*clip_utm))
    features: list[dict] = []
    for row in rows:
        geom = wkb_loads(row["geometry_wkb"])
        if clip.is_empty or not geom.intersects(clip):
            continue
        for transformed in _clip_transform_precision(geom, clip):
            features.append(
                {
                    "type": "Feature",
                    "geometry": mapping(transformed),
                    "properties": {"reason": row["reason"]},
                }
            )
    return _feature_collection(features)


def build_nodata_geojson(
    rows: Iterable[dict],
    *,
    aoi_utm: tuple[float, float, float, float],
    clip_utm: tuple[float, float, float, float] | None = None,
) -> str:
    return json.dumps(build_nodata_feature_collection(rows, aoi_utm=aoi_utm, clip_utm=clip_utm))


def _transform_point_precision(geom, aoi_utm_polygon):
    if geom.is_empty or not aoi_utm_polygon.covers(geom):
        return None
    transformed = _transform_to_4326(geom)
    return shapely.set_precision(transformed, grid_size=_PRECISION_DEG)


def build_missed_objects_feature_collection(
    rows: Iterable[dict],
    *,
    aoi_utm: tuple[float, float, float, float],
    clip_utm: tuple[float, float, float, float] | None = None,
) -> dict:
    aoi = box(*aoi_utm)
    clip = aoi if clip_utm is None else aoi.intersection(box(*clip_utm))
    features: list[dict] = []
    for row in rows:
        geom = wkb_loads(row["geometry_wkb"])
        transformed = _transform_point_precision(geom, clip)
        if transformed is None:
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(transformed),
                "properties": {
                    "id": row["id"],
                    "created_at": row["created_at"],
                },
            }
        )
    return _feature_collection(features)


def build_missed_objects_geojson(
    rows: Iterable[dict],
    *,
    aoi_utm: tuple[float, float, float, float],
    clip_utm: tuple[float, float, float, float] | None = None,
) -> str:
    return json.dumps(
        build_missed_objects_feature_collection(rows, aoi_utm=aoi_utm, clip_utm=clip_utm)
    )
