from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import reproject
from rasterio.windows import Window, from_bounds as window_from_bounds
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
    ownership_bounds_pixel: tuple[float, float, float, float] | None = None
    # Optional auxiliary modalities, both already resampled onto the tile's
    # DOP-grid and therefore identically shaped (size × size). They are
    # populated by iter_tiles only when explicitly requested, so existing
    # callers see None and can ignore them.
    nir: np.ndarray | None = None  # uint8, DOP20 band 4
    ndsm: np.ndarray | None = None  # float32, height above ground in metres


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


def _ownership_intervals(offsets: list[int], size: int, margin: int) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    for idx, offset in enumerate(offsets):
        lo = offset + margin
        hi = offset + size - margin
        if idx + 1 < len(offsets):
            hi = min(hi, offsets[idx + 1] + margin)
        intervals.append((float(lo), float(hi)))
    return intervals


def _safe_center_has_nodata(
    nodata_mask: np.ndarray,
    margin: int,
    size: int,
    threshold: float,
) -> bool:
    safe_center = nodata_mask[margin : size - margin, margin : size - margin]
    return bool(safe_center.mean() > threshold)


def _read_ndsm_for_tile(
    ndsm_src: rasterio.io.DatasetReader,
    tile_affine: Affine,
    size: int,
) -> np.ndarray:
    """Resample the DEM/nDSM onto the tile's DOP grid (size × size).

    Uses rasterio's WarpedVRT with the *tile's* affine + (size × size) as
    explicit output geometry. Bilinear resampling because DEM is continuous
    data — nearest would create blocky thresholds when filtering by height.
    Pixels outside the source raster come back as the VRT's NoData value
    (typically 0 for elevation services), which the modality filter treats
    as a neutral failure rather than as "tall".
    """
    with WarpedVRT(
        ndsm_src,
        crs=ndsm_src.crs,
        transform=tile_affine,
        width=size,
        height=size,
        resampling=Resampling.bilinear,
    ) as vrt:
        ndsm = vrt.read(1)
    return ndsm.astype(np.float32, copy=False)


def _same_grid(src, other) -> bool:
    return (
        src.width == other.width
        and src.height == other.height
        and src.crs == other.crs
        and tuple(round(value, 9) for value in src.transform) == tuple(
            round(value, 9) for value in other.transform
        )
    )


def align_raster_to_reference_grid(
    source_path: Path,
    reference_path: Path,
    out_path: Path,
) -> Path:
    """Inputs: Path, Path, Path. Logic: resample a single-band raster onto a reference raster grid. Returns: Path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(source_path) as src, rasterio.open(reference_path) as ref:
        profile = {
            "driver": "GTiff",
            "width": ref.width,
            "height": ref.height,
            "count": 1,
            "dtype": "float32",
            "crs": ref.crs,
            "transform": ref.transform,
            "nodata": np.nan,
            "compress": "deflate",
            "predictor": 3,
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,
        }
        with rasterio.open(out_path, "w", **profile) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs or ref.crs,
                src_nodata=src.nodata,
                dst_transform=ref.transform,
                dst_crs=ref.crs,
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
    return out_path


def iter_tiles(
    vrt_path: Path,
    cfg: TileConfig,
    *,
    safe_center_nodata_threshold: float = 0.0,
    read_nir: bool = False,
    ndsm_path: Path | None = None,
) -> Iterator[Tile | NodataTile]:
    """Iterate tiles from the DOP VRT, optionally enriched with NIR/nDSM.

    - ``read_nir``: pull DOP band 4 (NIR) into ``Tile.nir``. Cheap because
      it's the same dataset; one extra read per tile.
    - ``ndsm_path``: a separate DEM/nDSM raster (any CRS, any resolution).
      The tiler opens it once outside the per-tile loop and resamples per
      tile via WarpedVRT + bilinear interpolation onto the DOP 0.2 m grid.
    """
    with rasterio.Env(GDAL_MAX_DATASET_POOL_SIZE=256):
        ndsm_src = rasterio.open(ndsm_path) if ndsm_path is not None else None
        try:
            with rasterio.open(vrt_path) as src:
                ndsm_is_on_dop_grid = ndsm_src is not None and _same_grid(ndsm_src, src)
                col_offsets = _axis_offsets(src.width, cfg.size, cfg.tile_step)
                row_offsets = _axis_offsets(src.height, cfg.size, cfg.tile_step)
                col_ownership = _ownership_intervals(col_offsets, cfg.size, cfg.center_margin)
                row_ownership = _ownership_intervals(row_offsets, cfg.size, cfg.center_margin)

                read_indexes: list[int] = [1, 2, 3]
                if read_nir and src.count >= 4:
                    read_indexes.append(4)

                for tile_row, row_off in enumerate(row_offsets):
                    owner_r0, owner_r1 = row_ownership[tile_row]
                    for tile_col, col_off in enumerate(col_offsets):
                        owner_c0, owner_c1 = col_ownership[tile_col]
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
                            indexes=read_indexes,
                            window=window,
                            boundless=True,
                            fill_value=0,
                        )
                        rgb = array_chw[:3].transpose(1, 2, 0)
                        nir_band: np.ndarray | None = None
                        if read_nir and array_chw.shape[0] >= 4:
                            nir_band = array_chw[3]

                        ndsm_band: np.ndarray | None = None
                        if ndsm_src is not None:
                            if ndsm_is_on_dop_grid:
                                ndsm_band = ndsm_src.read(
                                    1,
                                    window=window,
                                    boundless=True,
                                    fill_value=np.nan,
                                ).astype(np.float32, copy=False)
                            else:
                                ndsm_band = _read_ndsm_for_tile(ndsm_src, tile_affine, cfg.size)

                        yield Tile(
                            array=rgb,
                            pixel_origin=(row_off, col_off),
                            size=cfg.size,
                            center_margin=cfg.center_margin,
                            affine=tile_affine,
                            tile_row=tile_row,
                            tile_col=tile_col,
                            nodata_mask=nodata_mask,
                            ownership_bounds_pixel=(owner_r0, owner_r1, owner_c0, owner_c1),
                            nir=nir_band,
                            ndsm=ndsm_band,
                        )
        finally:
            if ndsm_src is not None:
                ndsm_src.close()


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
