import geopandas as gpd
from shapely.geometry import Polygon

from ki_geodaten.pipeline.vector_cleanup import apply_vector_cleanup


def _gdf(geom):
    return gpd.GeoDataFrame(
        [{"score": 0.9, "source_tile_row": 0, "source_tile_col": 0}],
        geometry=[geom],
        crs="EPSG:25832",
    )


def test_simplification_reduces_vertices_and_keeps_valid_polygon():
    geom = Polygon(
        [
            (0, 0),
            (1, 0.02),
            (2, 0),
            (3, 0.01),
            (4, 0),
            (4, 2),
            (0, 2),
            (0, 0),
        ]
    )

    cleaned = apply_vector_cleanup(
        _gdf(geom),
        {"simplification_tolerance_m": 0.1, "orthogonalize": False},
    )

    assert len(cleaned) == 1
    assert cleaned.geometry.iloc[0].is_valid
    assert len(cleaned.geometry.iloc[0].exterior.coords) < len(geom.exterior.coords)


def test_orthogonalize_replaces_rectangular_candidate():
    geom = Polygon([(0, 0), (4, 0), (4, 2), (2, 2), (2, 1.9), (0, 2), (0, 0)])

    cleaned = apply_vector_cleanup(
        _gdf(geom),
        {"simplification_tolerance_m": None, "orthogonalize": True},
    )

    assert len(cleaned) == 1
    assert cleaned.geometry.iloc[0].area == 8
    assert len(cleaned.geometry.iloc[0].exterior.coords) == 5


def test_orthogonalize_leaves_irregular_polygon_unchanged():
    geom = Polygon([(0, 0), (4, 0), (0, 1), (0, 0)])

    cleaned = apply_vector_cleanup(
        _gdf(geom),
        {"simplification_tolerance_m": None, "orthogonalize": True},
    )

    assert len(cleaned) == 1
    assert cleaned.geometry.iloc[0].equals_exact(geom, tolerance=0)


def test_cleanup_drops_empty_results():
    geom = Polygon([(0, 0), (0, 0), (0, 0)])

    cleaned = apply_vector_cleanup(
        _gdf(geom),
        {"simplification_tolerance_m": 1.0, "orthogonalize": True},
    )

    assert len(cleaned) == 0
    assert "score" in cleaned.columns
