import numpy as np
import pytest
from affine import Affine
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Point, Polygon

from ki_geodaten.models import TilePreset
import geopandas as gpd

from ki_geodaten.pipeline.merger import (
    extract_polygons,
    global_polygon_nms,
    keep_center_only,
    masks_to_polygons,
)
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


def test_keep_center_uses_exclusive_global_ownership_bounds():
    cfg = TileConfig.from_preset(TilePreset.MEDIUM)
    tile = Tile(
        array=np.zeros((cfg.size, cfg.size, 3), dtype=np.uint8),
        pixel_origin=(0, 384),
        size=cfg.size,
        center_margin=cfg.center_margin,
        affine=Affine(0.2, 0, 0, 0, -0.2, cfg.size * 0.2),
        tile_row=0,
        tile_col=1,
        nodata_mask=np.zeros((cfg.size, cfg.size), dtype=bool),
        ownership_bounds_pixel=(320.0, 704.0, 704.0, 796.0),
    )
    owned = _mr((500, 350, 524, 374))
    overlap_owned_by_next_tile = _mr((500, 450, 524, 474))

    assert keep_center_only([owned, overlap_owned_by_next_tile], tile) == [owned]


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


def test_masks_to_polygons_closes_small_component_gap_for_one_detection():
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:30, 10:30] = True
    mask[10:30, 32:52] = True
    mr = MaskResult(mask=mask, score=0.95, box_pixel=(10, 10, 30, 52))
    gdf = masks_to_polygons([mr], _tile(), min_area_m2=0.01)
    assert len(gdf) == 1


def test_masks_to_polygons_keeps_distant_components_separate():
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:30, 10:30] = True
    mask[10:30, 42:62] = True
    mr = MaskResult(mask=mask, score=0.95, box_pixel=(10, 10, 30, 62))
    gdf = masks_to_polygons([mr], _tile(), min_area_m2=0.01)
    assert len(gdf) == 2


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


def test_global_polygon_nms_drops_lower_scoring_duplicate():
    correct = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    duplicate = Polygon([(1, 1), (11, 1), (11, 11), (1, 11)])
    separate = Polygon([(20, 0), (30, 0), (30, 10), (20, 10)])
    gdf = gpd.GeoDataFrame(
        [
            {"score": 0.6, "source_tile_row": 0, "source_tile_col": 0},
            {"score": 0.9, "source_tile_row": 0, "source_tile_col": 1},
            {"score": 0.5, "source_tile_row": 1, "source_tile_col": 0},
        ],
        geometry=[duplicate, correct, separate],
        crs="EPSG:25832",
    )

    kept = global_polygon_nms(gdf, iou_threshold=0.5, containment_ratio=0.85)

    assert len(kept) == 2
    assert list(kept["score"]) == [0.9, 0.5]


def test_global_polygon_nms_drops_mostly_contained_artifact_even_when_iou_is_low():
    building = Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])
    small_artifact = Polygon([(2, 2), (6, 2), (6, 6), (2, 6)])
    gdf = gpd.GeoDataFrame(
        [
            {"score": 0.9, "source_tile_row": 0, "source_tile_col": 0},
            {"score": 0.4, "source_tile_row": 0, "source_tile_col": 1},
        ],
        geometry=[building, small_artifact],
        crs="EPSG:25832",
    )

    kept = global_polygon_nms(gdf, iou_threshold=0.5, containment_ratio=0.85)

    assert len(kept) == 1
    assert kept.iloc[0].geometry.equals(building)


def test_global_polygon_nms_keeps_larger_polygon_for_containment_clash():
    building = Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])
    high_score_artifact = Polygon([(2, 2), (6, 2), (6, 6), (2, 6)])
    gdf = gpd.GeoDataFrame(
        [
            {"score": 0.4, "source_tile_row": 0, "source_tile_col": 0},
            {"score": 0.95, "source_tile_row": 0, "source_tile_col": 1},
        ],
        geometry=[building, high_score_artifact],
        crs="EPSG:25832",
    )

    kept = global_polygon_nms(gdf, iou_threshold=0.5, containment_ratio=0.85)

    assert len(kept) == 1
    assert kept.iloc[0].geometry.equals(building)


def test_global_polygon_nms_drops_low_iou_roof_fragment_with_buffered_coverage():
    base_roof = Polygon([(0, 0), (30, 0), (30, 10), (0, 10)])
    fragment = Polygon([(24, -0.6), (36, -0.6), (36, 9.4), (24, 9.4)])
    gdf = gpd.GeoDataFrame(
        [
            {"score": 0.9, "source_tile_row": 0, "source_tile_col": 0},
            {"score": 0.45, "source_tile_row": 0, "source_tile_col": 1},
        ],
        geometry=[base_roof, fragment],
        crs="EPSG:25832",
    )

    kept = global_polygon_nms(
        gdf,
        iou_threshold=0.5,
        containment_ratio=0.85,
        fragment_coverage_ratio=0.55,
        fragment_max_area_ratio=0.75,
        fragment_buffer_m=1.0,
    )

    assert len(kept) == 1
    assert kept.iloc[0].geometry.equals(base_roof)


def test_global_polygon_nms_keeps_partially_overlapping_distinct_objects():
    left = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    right = Polygon([(8, 0), (18, 0), (18, 10), (8, 10)])
    gdf = gpd.GeoDataFrame(
        [
            {"score": 0.9, "source_tile_row": 0, "source_tile_col": 0},
            {"score": 0.8, "source_tile_row": 0, "source_tile_col": 1},
        ],
        geometry=[left, right],
        crs="EPSG:25832",
    )

    kept = global_polygon_nms(gdf, iou_threshold=0.5, containment_ratio=0.85)

    assert len(kept) == 2


def test_global_polygon_nms_keeps_similarly_sized_adjacent_objects_under_fragment_rule():
    left = Polygon([(0, 0), (12, 0), (12, 10), (0, 10)])
    right = Polygon([(11.5, 0), (23.5, 0), (23.5, 10), (11.5, 10)])
    gdf = gpd.GeoDataFrame(
        [
            {"score": 0.9, "source_tile_row": 0, "source_tile_col": 0},
            {"score": 0.8, "source_tile_row": 0, "source_tile_col": 1},
        ],
        geometry=[left, right],
        crs="EPSG:25832",
    )

    kept = global_polygon_nms(
        gdf,
        iou_threshold=0.5,
        containment_ratio=0.85,
        fragment_coverage_ratio=0.55,
        fragment_max_area_ratio=0.75,
        fragment_buffer_m=1.0,
    )

    assert len(kept) == 2
