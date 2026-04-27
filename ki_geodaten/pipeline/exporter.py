from __future__ import annotations

from pathlib import Path
from typing import Iterable

import geopandas as gpd
from shapely.geometry.base import BaseGeometry
from shapely.validation import make_valid

from ki_geodaten.pipeline.merger import extract_polygons

DETECTED_LAYER = "detected_objects"
NODATA_LAYER = "nodata_regions"
CRS = "EPSG:25832"

_EMPTY_DETECTED_COLS = ("score", "source_tile_row", "source_tile_col")
_EMPTY_NODATA_COLS = ("reason",)
_DETECTED_SCHEMA = {
    "geometry": "Polygon",
    "properties": {
        "score": "float",
        "source_tile_row": "int",
        "source_tile_col": "int",
    },
}
_NODATA_SCHEMA = {
    "geometry": "Polygon",
    "properties": {"reason": "str:32"},
}


def _clip_and_normalize(gdf: gpd.GeoDataFrame, aoi: BaseGeometry) -> gpd.GeoDataFrame:
    if len(gdf) == 0:
        return gpd.GeoDataFrame(
            {column: [] for column in gdf.columns if column != "geometry"},
            geometry=[],
            crs=CRS,
        )

    clipped = gdf.copy()
    clipped["geometry"] = clipped.geometry.intersection(aoi)
    clipped = clipped[~clipped.geometry.is_empty]

    rows: list[dict] = []
    geometries = []
    for _, rec in clipped.iterrows():
        geom = make_valid(rec.geometry)
        for polygon in extract_polygons(geom):
            rows.append({key: rec[key] for key in rec.index if key != "geometry"})
            geometries.append(polygon)

    if not geometries:
        return gpd.GeoDataFrame(
            {column: [] for column in gdf.columns if column != "geometry"},
            geometry=[],
            crs=CRS,
        )
    return gpd.GeoDataFrame(rows, geometry=geometries, crs=CRS)


def _empty_with_schema(cols: Iterable[str]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({column: [] for column in cols}, geometry=[], crs=CRS)


def _unlink_gpkg_family(out_path: Path) -> None:
    out_path.unlink(missing_ok=True)
    out_path.with_suffix(".gpkg-wal").unlink(missing_ok=True)
    out_path.with_suffix(".gpkg-shm").unlink(missing_ok=True)


def export_two_layer_gpkg(
    detected_gdf: gpd.GeoDataFrame,
    nodata_gdf: gpd.GeoDataFrame,
    requested_bbox: BaseGeometry,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _unlink_gpkg_family(out_path)

    detected = _clip_and_normalize(detected_gdf, requested_bbox)
    nodata = _clip_and_normalize(nodata_gdf, requested_bbox)
    if len(detected) == 0:
        detected = _empty_with_schema(_EMPTY_DETECTED_COLS)
    if len(nodata) == 0:
        nodata = _empty_with_schema(_EMPTY_NODATA_COLS)

    detected.to_file(
        out_path,
        layer=DETECTED_LAYER,
        driver="GPKG",
        schema=_DETECTED_SCHEMA,
        engine="fiona",
    )
    nodata.to_file(
        out_path,
        layer=NODATA_LAYER,
        driver="GPKG",
        schema=_NODATA_SCHEMA,
        engine="fiona",
    )
