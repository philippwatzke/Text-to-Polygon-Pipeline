from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import shapes
from rasterio.vrt import WarpedVRT
from shapely.geometry import box, shape
from shapely.validation import make_valid

from ki_geodaten.pipeline.merger import extract_polygons
from ki_geodaten.pipeline.modality_filter import compute_ndvi

CRS = "EPSG:25832"


@dataclass(frozen=True)
class HeuristicBaselineConfig:
    ndsm_min: float | None = 3.0
    ndsm_max: float | None = None
    ndvi_min: float | None = None
    ndvi_max: float | None = None
    min_area_m2: float = 10.0
    close_m: float = 1.0
    open_m: float = 0.0

    def needs_ndvi(self) -> bool:
        return self.ndvi_min is not None or self.ndvi_max is not None

    def as_metadata(self) -> dict:
        return {
            "ndsm_min": self.ndsm_min,
            "ndsm_max": self.ndsm_max,
            "ndvi_min": self.ndvi_min,
            "ndvi_max": self.ndvi_max,
            "min_area_m2": self.min_area_m2,
            "close_m": self.close_m,
            "open_m": self.open_m,
        }


def _pixel_size_m(transform) -> float:
    return float((abs(transform.a) + abs(transform.e)) / 2.0)


def _kernel(radius_m: float, pixel_size_m: float):
    if radius_m <= 0:
        return None
    import cv2

    radius_px = max(1, int(round(radius_m / pixel_size_m)))
    return cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (radius_px * 2 + 1, radius_px * 2 + 1),
    )


def _apply_morphology(mask: np.ndarray, *, transform, close_m: float, open_m: float) -> np.ndarray:
    close_kernel = _kernel(close_m, _pixel_size_m(transform))
    open_kernel = _kernel(open_m, _pixel_size_m(transform))
    if close_kernel is None and open_kernel is None:
        return mask

    import cv2

    work = mask.astype(np.uint8, copy=False)
    if close_kernel is not None:
        work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, close_kernel)
    if open_kernel is not None:
        work = cv2.morphologyEx(work, cv2.MORPH_OPEN, open_kernel)
    return work.astype(bool, copy=False)


def _read_ndsm_on_grid(
    ndsm_path: Path,
    *,
    transform,
    crs,
    width: int,
    height: int,
) -> np.ndarray:
    with rasterio.open(ndsm_path) as ndsm_src:
        with WarpedVRT(
            ndsm_src,
            crs=crs,
            transform=transform,
            width=width,
            height=height,
            resampling=Resampling.bilinear,
        ) as vrt:
            return vrt.read(1).astype(np.float32, copy=False)


def _read_analysis_grid(
    ndsm_path: Path,
    dop_path: Path | None,
    config: HeuristicBaselineConfig,
) -> tuple[np.ndarray, np.ndarray | None, object, object]:
    if config.needs_ndvi():
        if dop_path is None:
            raise ValueError("NDVI thresholds require dop_path with red and NIR bands")
        with rasterio.open(dop_path) as dop_src:
            if dop_src.count < 4:
                raise ValueError("NDVI thresholds require a DOP raster with band 4 (NIR)")
            red = dop_src.read(1)
            nir = dop_src.read(4)
            ndvi = compute_ndvi(red, nir)
            ndsm = _read_ndsm_on_grid(
                ndsm_path,
                transform=dop_src.transform,
                crs=dop_src.crs,
                width=dop_src.width,
                height=dop_src.height,
            )
            return ndsm, ndvi, dop_src.transform, dop_src.crs

    with rasterio.open(ndsm_path) as ndsm_src:
        return (
            ndsm_src.read(1).astype(np.float32, copy=False),
            None,
            ndsm_src.transform,
            ndsm_src.crs,
        )


def _threshold_mask(
    ndsm: np.ndarray,
    ndvi: np.ndarray | None,
    config: HeuristicBaselineConfig,
) -> np.ndarray:
    mask = np.isfinite(ndsm)
    if config.ndsm_min is not None:
        mask &= ndsm >= config.ndsm_min
    if config.ndsm_max is not None:
        mask &= ndsm <= config.ndsm_max
    if config.ndvi_min is not None:
        if ndvi is None:
            raise ValueError("ndvi_min requires NDVI data")
        mask &= np.isfinite(ndvi) & (ndvi >= config.ndvi_min)
    if config.ndvi_max is not None:
        if ndvi is None:
            raise ValueError("ndvi_max requires NDVI data")
        mask &= np.isfinite(ndvi) & (ndvi <= config.ndvi_max)
    return mask


def _vectorize_mask(
    mask: np.ndarray,
    *,
    transform,
    aoi_bbox_utm: tuple[float, float, float, float],
    min_area_m2: float,
) -> gpd.GeoDataFrame:
    aoi = box(*aoi_bbox_utm)
    rows: list[dict] = []
    geometries = []
    mask_u8 = mask.astype(np.uint8, copy=False)
    for geom_json, value in shapes(mask_u8, mask=mask, transform=transform):
        if int(value) != 1:
            continue
        geom = make_valid(shape(geom_json)).intersection(aoi)
        for polygon in extract_polygons(geom):
            if polygon.area < min_area_m2:
                continue
            rows.append({"score": 1.0, "source_tile_row": 0, "source_tile_col": 0})
            geometries.append(polygon)

    return gpd.GeoDataFrame(rows, geometry=geometries, crs=CRS)


def build_heuristic_polygons(
    *,
    ndsm_path: Path,
    aoi_bbox_utm: tuple[float, float, float, float],
    config: HeuristicBaselineConfig,
    dop_path: Path | None = None,
) -> gpd.GeoDataFrame:
    """Build a naive raster-threshold baseline as polygons.

    This is intentionally simple and explainable: threshold nDSM/NDVI pixels,
    optionally apply morphology, connected-component vectorization, then drop
    tiny polygons. It is suitable as the thesis null baseline against SAM.
    """
    ndsm, ndvi, transform, _crs = _read_analysis_grid(ndsm_path, dop_path, config)
    mask = _threshold_mask(ndsm, ndvi, config)
    mask = _apply_morphology(
        mask,
        transform=transform,
        close_m=config.close_m,
        open_m=config.open_m,
    )
    return _vectorize_mask(
        mask,
        transform=transform,
        aoi_bbox_utm=aoi_bbox_utm,
        min_area_m2=config.min_area_m2,
    )
