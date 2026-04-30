import json

from shapely.geometry import Point, Polygon
from shapely.wkb import dumps as wkb_dumps

from ki_geodaten.app.serialization import (
    build_nodata_feature_collection,
    build_missed_objects_feature_collection,
    build_polygons_feature_collection,
    build_polygons_geojson,
)


def _aoi_utm():
    return (691000.0, 5335000.0, 692000.0, 5336000.0)


def _poly_wkb(geom):
    return wkb_dumps(geom)


def test_polygons_geojson_has_4326_bounds_inside_bayern():
    poly = Polygon(
        [(691100, 5335100), (691200, 5335100), (691200, 5335200), (691100, 5335200)]
    )
    rows = [{"id": 1, "geometry_wkb": _poly_wkb(poly), "score": 0.88, "validation": "ACCEPTED"}]
    fc = build_polygons_feature_collection(rows, aoi_utm=_aoi_utm())
    feat = fc["features"][0]
    xs = [coord[0] for coord in feat["geometry"]["coordinates"][0]]
    assert 8.9 <= min(xs) <= max(xs) <= 13.9
    assert feat["properties"] == {"id": 1, "score": 0.88, "validation": "ACCEPTED"}


def test_polygons_precision_capped_at_1e_6():
    poly = Polygon(
        [
            (691123.456789, 5335111.222333),
            (691200, 5335100),
            (691200, 5335200),
            (691100, 5335200),
        ]
    )
    rows = [{"id": 1, "geometry_wkb": _poly_wkb(poly), "score": 0.5, "validation": "ACCEPTED"}]
    fc = build_polygons_feature_collection(rows, aoi_utm=_aoi_utm())
    for coord in fc["features"][0]["geometry"]["coordinates"][0]:
        for value in coord:
            assert abs(value - round(value, 6)) < 1e-9


def test_polygons_clip_by_aoi_and_drop_fully_outside():
    crossing = Polygon(
        [(691900, 5335500), (692200, 5335500), (692200, 5335600), (691900, 5335600)]
    )
    outside = Polygon(
        [(700000, 5400000), (700100, 5400000), (700100, 5400100), (700000, 5400100)]
    )
    rows = [
        {"id": 1, "geometry_wkb": _poly_wkb(crossing), "score": 0.8, "validation": "ACCEPTED"},
        {"id": 2, "geometry_wkb": _poly_wkb(outside), "score": 0.8, "validation": "ACCEPTED"},
    ]
    fc = build_polygons_feature_collection(rows, aoi_utm=_aoi_utm())
    assert [f["properties"]["id"] for f in fc["features"]] == [1]


def test_polygons_can_clip_to_viewport_bbox():
    left = Polygon(
        [(691100, 5335100), (691105, 5335100), (691105, 5335105), (691100, 5335105)]
    )
    right = Polygon(
        [(691500, 5335100), (691505, 5335100), (691505, 5335105), (691500, 5335105)]
    )
    rows = [
        {"id": 1, "geometry_wkb": _poly_wkb(left), "score": 0.8, "validation": "ACCEPTED"},
        {"id": 2, "geometry_wkb": _poly_wkb(right), "score": 0.8, "validation": "ACCEPTED"},
    ]

    fc = build_polygons_feature_collection(
        rows,
        aoi_utm=_aoi_utm(),
        clip_utm=(691050, 5335050, 691150, 5335150),
    )

    assert [f["properties"]["id"] for f in fc["features"]] == [1]


def test_nodata_geojson_carries_reason():
    poly = Polygon(
        [(691100, 5335100), (691200, 5335100), (691200, 5335200), (691100, 5335200)]
    )
    rows = [{"id": 9, "geometry_wkb": _poly_wkb(poly), "reason": "OOM"}]
    fc = build_nodata_feature_collection(rows, aoi_utm=_aoi_utm())
    assert fc["features"][0]["properties"] == {"reason": "OOM"}


def test_missed_objects_geojson_has_point_ids():
    rows = [
        {
            "id": 7,
            "geometry_wkb": _poly_wkb(Point(691100, 5335100)),
            "created_at": "2026-04-27T00:00:00+00:00",
        }
    ]
    fc = build_missed_objects_feature_collection(rows, aoi_utm=_aoi_utm())

    assert fc["features"][0]["geometry"]["type"] == "Point"
    assert fc["features"][0]["properties"] == {
        "id": 7,
        "created_at": "2026-04-27T00:00:00+00:00",
    }


def test_processpool_builder_returns_json_string():
    poly = Polygon(
        [(691100, 5335100), (691200, 5335100), (691200, 5335200), (691100, 5335200)]
    )
    rows = [{"id": 1, "geometry_wkb": _poly_wkb(poly), "score": 0.5, "validation": "ACCEPTED"}]
    payload = build_polygons_geojson(rows, aoi_utm=_aoi_utm())
    assert isinstance(payload, str)
    assert json.loads(payload)["type"] == "FeatureCollection"
