from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from ki_geodaten.pipeline.heuristic_baseline import (
    HeuristicBaselineConfig,
    build_heuristic_polygons,
)


def _write_single_band(path: Path, array: np.ndarray) -> None:
    profile = {
        "driver": "GTiff",
        "height": array.shape[0],
        "width": array.shape[1],
        "count": 1,
        "dtype": str(array.dtype),
        "crs": "EPSG:25832",
        "transform": from_origin(0, 10, 1, 1),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)


def _write_dop(path: Path, red: np.ndarray, nir: np.ndarray) -> None:
    bands = np.zeros((4, red.shape[0], red.shape[1]), dtype=np.uint8)
    bands[0] = red
    bands[1] = 20
    bands[2] = 20
    bands[3] = nir
    profile = {
        "driver": "GTiff",
        "height": red.shape[0],
        "width": red.shape[1],
        "count": 4,
        "dtype": "uint8",
        "crs": "EPSG:25832",
        "transform": from_origin(0, 10, 1, 1),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(bands)


def test_ndsm_threshold_vectorizes_connected_components(tmp_path):
    ndsm = np.zeros((10, 10), dtype=np.float32)
    ndsm[2:7, 2:7] = 4.0
    ndsm_path = tmp_path / "ndsm.tif"
    _write_single_band(ndsm_path, ndsm)

    gdf = build_heuristic_polygons(
        ndsm_path=ndsm_path,
        aoi_bbox_utm=(0.0, 0.0, 10.0, 10.0),
        config=HeuristicBaselineConfig(ndsm_min=3.0, min_area_m2=4.0, close_m=0.0),
    )

    assert len(gdf) == 1
    assert gdf.geometry.iloc[0].area == 25.0
    assert gdf.iloc[0]["score"] == 1.0


def test_min_area_drops_small_components(tmp_path):
    ndsm = np.zeros((10, 10), dtype=np.float32)
    ndsm[2:5, 2:5] = 4.0
    ndsm_path = tmp_path / "ndsm.tif"
    _write_single_band(ndsm_path, ndsm)

    gdf = build_heuristic_polygons(
        ndsm_path=ndsm_path,
        aoi_bbox_utm=(0.0, 0.0, 10.0, 10.0),
        config=HeuristicBaselineConfig(ndsm_min=3.0, min_area_m2=10.0, close_m=0.0),
    )

    assert len(gdf) == 0


def test_ndvi_threshold_requires_matching_dop_band_four(tmp_path):
    ndsm = np.full((10, 10), 4.0, dtype=np.float32)
    ndsm_path = tmp_path / "ndsm.tif"
    _write_single_band(ndsm_path, ndsm)

    red = np.full((10, 10), 80, dtype=np.uint8)
    nir = np.full((10, 10), 80, dtype=np.uint8)
    red[3:7, 3:7] = 30
    nir[3:7, 3:7] = 200
    dop_path = tmp_path / "dop.tif"
    _write_dop(dop_path, red, nir)

    gdf = build_heuristic_polygons(
        ndsm_path=ndsm_path,
        dop_path=dop_path,
        aoi_bbox_utm=(0.0, 0.0, 10.0, 10.0),
        config=HeuristicBaselineConfig(
            ndsm_min=3.0,
            ndvi_min=0.3,
            min_area_m2=4.0,
            close_m=0.0,
        ),
    )

    assert len(gdf) == 1
    assert gdf.geometry.iloc[0].area == 16.0
