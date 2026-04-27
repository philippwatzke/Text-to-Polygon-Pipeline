from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Polygon, box

fiona = pytest.importorskip("fiona")

from ki_geodaten.pipeline.exporter import export_two_layer_gpkg


def _aoi():
    return box(691000.0, 5335000.0, 692000.0, 5336000.0)


def _detected_gdf(geoms, scores=None):
    scores = scores or [0.9] * len(geoms)
    return gpd.GeoDataFrame(
        {
            "score": scores,
            "source_tile_row": [0] * len(geoms),
            "source_tile_col": [0] * len(geoms),
        },
        geometry=list(geoms),
        crs="EPSG:25832",
    )


def _nodata_gdf(geoms, reasons=None):
    reasons = reasons or ["NODATA_PIXELS"] * len(geoms)
    return gpd.GeoDataFrame({"reason": reasons}, geometry=list(geoms), crs="EPSG:25832")


def test_export_writes_two_layers(tmp_path: Path):
    out = tmp_path / "j.gpkg"
    inside = Polygon(
        [(691100, 5335100), (691200, 5335100), (691200, 5335200), (691100, 5335200)]
    )
    export_two_layer_gpkg(_detected_gdf([inside]), _nodata_gdf([]), _aoi(), out)
    assert out.exists()
    layers = fiona.listlayers(str(out))
    assert "detected_objects" in layers
    assert "nodata_regions" in layers


def test_export_clips_crossing_polygon_to_aoi(tmp_path: Path):
    out = tmp_path / "j.gpkg"
    crossing = Polygon(
        [(691900, 5335500), (692100, 5335500), (692100, 5335600), (691900, 5335600)]
    )
    export_two_layer_gpkg(_detected_gdf([crossing]), _nodata_gdf([]), _aoi(), out)
    gdf = gpd.read_file(out, layer="detected_objects")
    xs = [pt[0] for pt in gdf.geometry.iloc[0].exterior.coords]
    assert max(xs) <= 692000.0 + 1e-6


def test_export_overwrites_existing_file(tmp_path: Path):
    out = tmp_path / "j.gpkg"
    p1 = Polygon(
        [(691100, 5335100), (691200, 5335100), (691200, 5335200), (691100, 5335200)]
    )
    p2 = Polygon(
        [(691300, 5335300), (691400, 5335300), (691400, 5335400), (691300, 5335400)]
    )
    export_two_layer_gpkg(_detected_gdf([p1]), _nodata_gdf([]), _aoi(), out)
    export_two_layer_gpkg(_detected_gdf([p2]), _nodata_gdf([]), _aoi(), out)
    gdf = gpd.read_file(out, layer="detected_objects")
    assert len(gdf) == 1
