from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

import numpy as np
import rasterio
from affine import Affine
from rasterio.windows import Window
from shapely.geometry import Polygon

from ki_geodaten.models import TilePreset

_PRESET_PARAMS: dict[TilePreset, tuple[int, int]] = {
    TilePreset.SMALL: (320, 160),
    TilePreset.MEDIUM: (640, 320),
    TilePreset.LARGE: (960, 480),
}


@dataclass(frozen=True)
class TileConfig:
    size: int
    overlap: int
    center_margin: int

    @property
    def tile_step(self) -> int:
        return self.size - self.overlap

    @property
    def safe_center_size(self) -> int:
        return self.size - 2 * self.center_margin

    @classmethod
    def from_preset(cls, preset: TilePreset | str) -> TileConfig:
        tile_preset = TilePreset(str(preset))
        overlap, center_margin = _PRESET_PARAMS[tile_preset]
        return cls(size=1024, overlap=overlap, center_margin=center_margin)


@dataclass(frozen=True)
class Tile:
    array: np.ndarray
    pixel_origin: tuple[int, int]
    size: int
    center_margin: int
    affine: Affine
    tile_row: int
    tile_col: int
    nodata_mask: np.ndarray


@dataclass(frozen=True)
class NodataTile:
    tile_row: int
    tile_col: int
    pixel_origin: tuple[int, int]
    size: int
    center_margin: int
    affine: Affine
    reason: Literal["NODATA_PIXELS"] = "NODATA_PIXELS"


def _axis_offsets(extent: int, size: int, step: int) -> list[int]:
    if extent <= size:
        return [0]

    offsets = list(range(0, extent - size + 1, step))
    last = extent - size
    if offsets[-1] != last:
        offsets.append(last)
    return offsets


def iter_grid(src, cfg: TileConfig) -> Iterator[tuple[int, int, int, int]]:
    col_offsets = _axis_offsets(src.width, cfg.size, cfg.tile_step)
    row_offsets = _axis_offsets(src.height, cfg.size, cfg.tile_step)
    for tile_row, row_off in enumerate(row_offsets):
        for tile_col, col_off in enumerate(col_offsets):
            yield (tile_row, tile_col, row_off, col_off)


def _safe_center_has_nodata(
    nodata_mask: np.ndarray,
    margin: int,
    size: int,
    threshold: float,
) -> bool:
    safe_center = nodata_mask[margin : size - margin, margin : size - margin]
    return bool(safe_center.mean() > threshold)


def iter_tiles(
    vrt_path: Path,
    cfg: TileConfig,
    *,
    safe_center_nodata_threshold: float = 0.0,
) -> Iterator[Tile | NodataTile]:
    with rasterio.Env(GDAL_MAX_DATASET_POOL_SIZE=256):
        with rasterio.open(vrt_path) as src:
            for tile_row, tile_col, row_off, col_off in iter_grid(src, cfg):
                window = Window(
                    col_off=col_off,
                    row_off=row_off,
                    width=cfg.size,
                    height=cfg.size,
                )
                tile_affine = src.window_transform(window)
                dataset_mask = src.dataset_mask(window=window)
                nodata_mask = dataset_mask == 0

                if _safe_center_has_nodata(
                    nodata_mask,
                    cfg.center_margin,
                    cfg.size,
                    safe_center_nodata_threshold,
                ):
                    yield NodataTile(
                        tile_row=tile_row,
                        tile_col=tile_col,
                        pixel_origin=(row_off, col_off),
                        size=cfg.size,
                        center_margin=cfg.center_margin,
                        affine=tile_affine,
                    )
                    continue

                array_chw = src.read(
                    indexes=[1, 2, 3],
                    window=window,
                    boundless=True,
                    fill_value=0,
                )
                yield Tile(
                    array=array_chw.transpose(1, 2, 0),
                    pixel_origin=(row_off, col_off),
                    size=cfg.size,
                    center_margin=cfg.center_margin,
                    affine=tile_affine,
                    tile_row=tile_row,
                    tile_col=tile_col,
                    nodata_mask=nodata_mask,
                )


def safe_center_polygon(tile: Tile | NodataTile) -> Polygon:
    margin = tile.center_margin
    size = tile.size
    corners_px = [
        (margin, margin),
        (size - margin, margin),
        (size - margin, size - margin),
        (margin, size - margin),
    ]
    corners_utm = [tile.affine * corner for corner in corners_px]
    return Polygon(corners_utm)
