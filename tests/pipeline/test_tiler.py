from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.enums import ColorInterp
from rasterio.transform import from_bounds
from affine import Affine
from shapely.geometry import Polygon

from ki_geodaten.models import TilePreset
from ki_geodaten.pipeline.tiler import (
    NodataTile,
    Tile,
    TileConfig,
    iter_grid,
    iter_tiles,
    safe_center_polygon,
)


ARTIFACT_DIR = Path("data/dop/test_tiler")


def _fresh_path(name: str) -> Path:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACT_DIR / name
    if path.exists():
        path.unlink()
    return path


def _make_rgb_tif(
    path: Path,
    width: int,
    height: int,
    bbox: tuple[float, float, float, float],
    nodata_mask: np.ndarray | None = None,
    bands: int = 3,
    *,
    set_nodata: bool = True,
) -> Path:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=bands,
        dtype="uint8",
        crs="EPSG:25832",
        transform=from_bounds(*bbox, width, height),
        nodata=0 if set_nodata else None,
    ) as dst:
        for band in range(1, bands + 1):
            arr = np.full((height, width), 128, dtype="uint8")
            if nodata_mask is not None:
                arr[nodata_mask] = 0
            dst.write(arr, band)
    return path


class FakeSrc:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height


def test_tile_config_small():
    cfg = TileConfig.from_preset(TilePreset.SMALL)
    assert cfg.size == 1024
    assert cfg.overlap == 320
    assert cfg.center_margin == 160
    assert cfg.tile_step == 704
    assert cfg.safe_center_size == 704


def test_tile_config_medium():
    cfg = TileConfig.from_preset(TilePreset.MEDIUM)
    assert cfg.overlap == 640
    assert cfg.center_margin == 320
    assert cfg.tile_step == 384
    assert cfg.safe_center_size == 384


def test_tile_config_large():
    cfg = TileConfig.from_preset(TilePreset.LARGE)
    assert cfg.overlap == 960
    assert cfg.center_margin == 480
    assert cfg.tile_step == 64
    assert cfg.safe_center_size == 64


def test_tile_config_invariants():
    for preset in TilePreset:
        cfg = TileConfig.from_preset(preset)
        assert cfg.tile_step == cfg.safe_center_size
        assert cfg.center_margin * 2 == cfg.overlap


def test_iter_grid_exact_fit_medium():
    grid = list(iter_grid(FakeSrc(1408, 1408), TileConfig.from_preset(TilePreset.MEDIUM)))
    assert len({(row, col, row_off, col_off) for row, col, row_off, col_off in grid}) == 4


def test_iter_grid_edge_shifts_last_tile_inward():
    grid = list(iter_grid(FakeSrc(1500, 1024), TileConfig.from_preset(TilePreset.MEDIUM)))
    col_offsets = sorted({col_off for _, _, _, col_off in grid})
    assert col_offsets[-1] == 1500 - 1024
    last_tile = max(grid, key=lambda item: item[3])
    assert last_tile[3] + 1024 == 1500


def test_iter_grid_unique_logical_indices():
    grid = list(iter_grid(FakeSrc(1500, 1500), TileConfig.from_preset(TilePreset.MEDIUM)))
    indices = [(row, col) for row, col, _, _ in grid]
    assert len(indices) == len(set(indices))


def test_iter_grid_smaller_than_tile_single_tile():
    grid = list(iter_grid(FakeSrc(1024, 1024), TileConfig.from_preset(TilePreset.MEDIUM)))
    assert grid == [(0, 0, 0, 0)]


def test_iter_tiles_per_tile_affine():
    path = _fresh_path("affine.tif")
    _make_rgb_tif(path, 2048, 2048, (691000.0, 5335000.0, 691409.6, 5335409.6))
    cfg = TileConfig.from_preset(TilePreset.MEDIUM)

    tiles = list(iter_tiles(path, cfg))

    assert len(tiles) >= 1
    tile = tiles[0]
    assert not isinstance(tile, NodataTile)
    assert tile.affine.a == pytest.approx(0.2)
    assert tile.affine.c == pytest.approx(691000.0)
    assert tile.array.shape == (1024, 1024, 3)
    assert tile.array.dtype == np.uint8


def test_iter_tiles_reads_only_rgb_from_4band():
    path = _fresh_path("four_band.tif")
    _make_rgb_tif(
        path,
        1024,
        1024,
        (0, 0, 204.8, 204.8),
        bands=4,
        set_nodata=False,
    )
    cfg = TileConfig.from_preset(TilePreset.MEDIUM)

    tiles = list(iter_tiles(path, cfg))

    assert not isinstance(tiles[0], NodataTile)
    assert tiles[0].array.shape == (1024, 1024, 3)
    assert tiles[0].nodata_mask.shape == (1024, 1024)


def test_iter_tiles_flags_nodata_in_safe_center():
    mask = np.zeros((1024, 1024), dtype=bool)
    mask[500:520, 500:520] = True
    path = _fresh_path("safe_center_nodata.tif")
    _make_rgb_tif(path, 1024, 1024, (0, 0, 204.8, 204.8), nodata_mask=mask)
    cfg = TileConfig.from_preset(TilePreset.MEDIUM)

    tiles = list(iter_tiles(path, cfg))

    assert isinstance(tiles[0], NodataTile)


def test_iter_tiles_black_rgb_without_nodata_mask_is_not_nodata():
    path = _fresh_path("black_rgb.tif")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=1024,
        height=1024,
        count=3,
        dtype="uint8",
        crs="EPSG:25832",
        transform=from_bounds(0, 0, 204.8, 204.8, 1024, 1024),
    ) as dst:
        dst.write(np.zeros((3, 1024, 1024), dtype="uint8"))
    cfg = TileConfig.from_preset(TilePreset.MEDIUM)

    tiles = list(iter_tiles(path, cfg))

    assert not isinstance(tiles[0], NodataTile)


def test_iter_tiles_alpha_mask_marks_nodata():
    path = _fresh_path("alpha_nodata.tif")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=1024,
        height=1024,
        count=4,
        dtype="uint8",
        crs="EPSG:25832",
        transform=from_bounds(0, 0, 204.8, 204.8, 1024, 1024),
    ) as dst:
        dst.write(np.full((3, 1024, 1024), 128, dtype="uint8"), indexes=[1, 2, 3])
        alpha = np.full((1024, 1024), 255, dtype="uint8")
        alpha[500:520, 500:520] = 0
        dst.write(alpha, 4)
        dst.colorinterp = (
            ColorInterp.red,
            ColorInterp.green,
            ColorInterp.blue,
            ColorInterp.alpha,
        )
    cfg = TileConfig.from_preset(TilePreset.MEDIUM)

    tiles = list(iter_tiles(path, cfg))

    assert isinstance(tiles[0], NodataTile)


def test_safe_center_polygon_medium():
    cfg = TileConfig.from_preset(TilePreset.MEDIUM)
    affine = Affine(0.2, 0, 691000.0, 0, -0.2, 5336204.8)
    tile = Tile(
        array=np.zeros((cfg.size, cfg.size, 3), dtype=np.uint8),
        pixel_origin=(0, 0),
        size=cfg.size,
        center_margin=cfg.center_margin,
        affine=affine,
        tile_row=0,
        tile_col=0,
        nodata_mask=np.zeros((cfg.size, cfg.size), dtype=bool),
    )
    poly = safe_center_polygon(tile)
    assert isinstance(poly, Polygon)
    minx, miny, maxx, maxy = poly.bounds
    assert maxx - minx == pytest.approx(76.8)
    assert maxy - miny == pytest.approx(76.8)
    assert minx == pytest.approx(691064.0)
    assert maxy == pytest.approx(5336204.8 - 64.0)
