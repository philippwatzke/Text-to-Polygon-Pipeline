import numpy as np
import pytest
from affine import Affine
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Point, Polygon

from ki_geodaten.models import TilePreset
from ki_geodaten.pipeline.merger import extract_polygons, keep_center_only, masks_to_polygons
from ki_geodaten.pipeline.segmenter import MaskResult
from ki_geodaten.pipeline.tiler import Tile, TileConfig


def _tile(preset=TilePreset.MEDIUM):
    cfg = TileConfig.from_preset(preset)
    return Tile(
        array=np.zeros((cfg.size, cfg.size, 3), dtype=np.uint8),
        pixel_origin=(0, 0),
        size=cfg.size,
        center_margin=cfg.center_margin,
        affine=Affine(0.2, 0, 0, 0, -0.2, cfg.size * 0.2),
        tile_row=0,
        tile_col=0,
        nodata_mask=np.zeros((cfg.size, cfg.size), dtype=bool),
    )


def _mr(box, shape=(1024, 1024)):
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = box
    mask[r0:r1, c0:c1] = True
    return MaskResult(mask=mask, score=0.9, box_pixel=box)


def test_keep_when_center_in_safe_zone_medium():
    tile = _tile(TilePreset.MEDIUM)
    mask = _mr((500, 500, 524, 524))
    assert keep_center_only([mask], tile) == [mask]


def test_drop_when_center_in_margin():
    tile = _tile(TilePreset.MEDIUM)
    mask = _mr((50, 50, 100, 100))
    assert keep_center_only([mask], tile) == []


def test_halfopen_right_edge_excluded():
    tile = _tile(TilePreset.MEDIUM)
    mask = _mr((703, 320, 705, 322))
    assert keep_center_only([mask], tile) == []


def test_halfopen_left_edge_included():
    tile = _tile(TilePreset.MEDIUM)
    mask = _mr((319, 319, 321, 321))
    assert keep_center_only([mask], tile) == [mask]


def test_all_presets_accept_midpoint():
    for preset in TilePreset:
        tile = _tile(preset)
        mask = _mr((510, 510, 514, 514))
        assert keep_center_only([mask], tile) == [mask]


def test_uses_bbox_center_not_geometric_centroid():
    tile = _tile(TilePreset.MEDIUM)
    mask = np.zeros((1024, 1024), dtype=bool)
    mask[500:524, 500:510] = True
    mask[500:524, 514:524] = True
    mask[500:504, 500:524] = True
    mr = MaskResult(mask=mask, score=0.9, box_pixel=(500, 500, 524, 524))
    assert keep_center_only([mr], tile) == [mr]


def test_extract_polygons_from_polygon():
    polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    assert extract_polygons(polygon) == [polygon]


def test_extract_polygons_from_multipolygon():
    multipolygon = MultiPolygon(
        [
            Polygon([(0, 0), (5, 0), (5, 5), (0, 5)]),
            Polygon([(10, 10), (15, 10), (15, 15), (10, 15)]),
        ]
    )
    assert len(extract_polygons(multipolygon)) == 2


def test_extract_polygons_drops_non_polygon_from_collection():
    collection = GeometryCollection(
        [
            Polygon([(0, 0), (5, 0), (5, 5), (0, 5)]),
            LineString([(0, 0), (10, 10)]),
            Point(5, 5),
        ]
    )
    polygons = extract_polygons(collection)
    assert len(polygons) == 1
    assert isinstance(polygons[0], Polygon)


def test_extract_polygons_drops_empty():
    assert extract_polygons(Polygon()) == []


def test_masks_to_polygons_connectivity_8_joins_diagonal():
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:30, 10:30] = True
    mask[30:50, 30:50] = True
    mr = MaskResult(mask=mask, score=0.95, box_pixel=(10, 10, 50, 50))
    gdf = masks_to_polygons([mr], _tile(), min_area_m2=0.01)
    assert len(gdf) == 1


def test_masks_to_polygons_excludes_background():
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:30, 10:30] = True
    mr = MaskResult(mask=mask, score=0.9, box_pixel=(10, 10, 30, 30))
    gdf = masks_to_polygons([mr], _tile(), min_area_m2=0.01)
    assert len(gdf) == 1


def test_masks_to_polygons_area_filter():
    mask = np.zeros((100, 100), dtype=bool)
    mask[0:2, 0:2] = True
    mr = MaskResult(mask=mask, score=0.9, box_pixel=(0, 0, 2, 2))
    gdf = masks_to_polygons([mr], _tile(), min_area_m2=1.0)
    assert len(gdf) == 0


def test_masks_to_polygons_metadata():
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:30, 10:30] = True
    mr = MaskResult(mask=mask, score=0.77, box_pixel=(10, 10, 30, 30))
    gdf = masks_to_polygons([mr], _tile(), min_area_m2=0.01)
    assert gdf.iloc[0]["score"] == pytest.approx(0.77)
    assert gdf.iloc[0]["source_tile_row"] == 0
    assert gdf.iloc[0]["source_tile_col"] == 0
    assert str(gdf.crs) == "EPSG:25832"
