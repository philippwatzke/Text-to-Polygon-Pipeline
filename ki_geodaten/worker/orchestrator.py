from __future__ import annotations

import json
import logging
import shutil
import traceback
from pathlib import Path

import geopandas as gpd
import rasterio
from shapely.wkb import dumps as wkb_dumps

from ki_geodaten.jobs.store import (
    get_job,
    increment_tile_completed,
    increment_tile_failed,
    insert_nodata_region,
    insert_polygons,
    update_status,
)
from ki_geodaten.models import ErrorReason, JobStatus, NoDataReason, TilePreset
from ki_geodaten.pipeline.dop_client import DopDownloadError, download_dop20, prepare_download_bbox
from ki_geodaten.pipeline.merger import keep_center_only, masks_to_polygons
from ki_geodaten.pipeline.segmenter import SegmenterUnavailableError
from ki_geodaten.pipeline.tiler import (
    NodataTile,
    TileConfig,
    iter_grid,
    iter_tiles,
    safe_center_polygon,
)

logger = logging.getLogger(__name__)


def _is_cuda_oom(exc: BaseException) -> bool:
    try:
        import torch
    except Exception:
        return False
    return isinstance(exc, torch.cuda.OutOfMemoryError)


def _empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _clear_exception_context(exc: BaseException) -> None:
    exc.__traceback__ = None
    exc.__context__ = None
    exc.__cause__ = None


def _traceback_tail(exc: BaseException, n: int = 20) -> str:
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return "\n".join(tb.splitlines()[-n:])


def _is_system_segmenter_error(exc: BaseException) -> bool:
    return isinstance(
        exc,
        (
            SegmenterUnavailableError,
            NotImplementedError,
            ImportError,
            ModuleNotFoundError,
        ),
    )


def _persist_polygons_for_tile(db_path: Path, job_id: str, gdf: gpd.GeoDataFrame) -> None:
    if gdf is None or len(gdf) == 0:
        return
    rows = [
        {
            "geometry_wkb": wkb_dumps(row.geometry),
            "score": float(row["score"]),
            "source_tile_row": int(row["source_tile_row"]),
            "source_tile_col": int(row["source_tile_col"]),
        }
        for _, row in gdf.iterrows()
    ]
    insert_polygons(db_path, job_id, rows)


def _persist_safe_center_nodata(db_path: Path, job_id: str, tile, reason: NoDataReason) -> None:
    insert_nodata_region(
        db_path,
        job_id,
        geometry_wkb=wkb_dumps(safe_center_polygon(tile)),
        tile_row=tile.tile_row,
        tile_col=tile.tile_col,
        reason=str(reason),
    )


def _download_error_reason(exc: DopDownloadError) -> str:
    text = str(exc)
    if text in {ErrorReason.DOP_TIMEOUT, ErrorReason.DOP_HTTP_ERROR}:
        return text
    return str(ErrorReason.DOP_HTTP_ERROR)


def run_job(
    db_path: Path,
    *,
    job_id: str,
    segmenter,
    data_root: Path,
    wms_url: str,
    layer: str,
    max_pixels: int,
    wms_version: str,
    fmt: str,
    crs: str,
    origin_x: float,
    origin_y: float,
    min_polygon_area_m2: float,
    safe_center_nodata_threshold: float,
) -> None:
    job = get_job(db_path, job_id)
    if job is None:
        return

    out_dir = data_root / "dop" / job_id
    preset = TilePreset(job["tile_preset"])
    cfg = TileConfig.from_preset(preset)

    try:
        update_status(db_path, job_id, JobStatus.DOWNLOADING, set_started=True)
        aoi_utm = tuple(json.loads(job["bbox_utm_snapped"]))
        prepared = prepare_download_bbox(
            *aoi_utm,
            preset=preset,
            origin_x=origin_x,
            origin_y=origin_y,
        )
        vrt_path = download_dop20(
            prepared.download_bbox,
            out_dir=out_dir,
            wms_url=wms_url,
            layer=layer,
            wms_version=wms_version,
            fmt=fmt,
            crs=crs,
            max_pixels=max_pixels,
            origin_x=origin_x,
            origin_y=origin_y,
        )

        update_status(db_path, job_id, JobStatus.INFERRING, dop_vrt_path=str(vrt_path))
        with rasterio.open(vrt_path) as src:
            tile_total = sum(1 for _ in iter_grid(src, cfg))
        update_status(db_path, job_id, JobStatus.INFERRING, tile_total=tile_total)

        for tile in iter_tiles(
            vrt_path,
            cfg,
            safe_center_nodata_threshold=safe_center_nodata_threshold,
        ):
            if isinstance(tile, NodataTile):
                _persist_safe_center_nodata(db_path, job_id, tile, NoDataReason.NODATA_PIXELS)
                increment_tile_failed(db_path, job_id)
                continue

            try:
                masks = segmenter.predict(tile, job["prompt"])
            except Exception as exc:  # noqa: BLE001
                if _is_system_segmenter_error(exc):
                    raise
                reason = NoDataReason.OOM if _is_cuda_oom(exc) else NoDataReason.INFERENCE_ERROR
                _persist_safe_center_nodata(db_path, job_id, tile, reason)
                increment_tile_failed(db_path, job_id)
                _clear_exception_context(exc)
                del exc
                _empty_cuda_cache()
                continue

            try:
                kept = keep_center_only(masks, tile)
                gdf = masks_to_polygons(kept, tile, min_area_m2=min_polygon_area_m2)
                _persist_polygons_for_tile(db_path, job_id, gdf)
                increment_tile_completed(db_path, job_id)
            except Exception as exc:  # noqa: BLE001
                logger.exception("invalid geometry in job=%s tile=%s/%s", job_id, tile.tile_row, tile.tile_col)
                _persist_safe_center_nodata(db_path, job_id, tile, NoDataReason.INVALID_GEOMETRY)
                increment_tile_failed(db_path, job_id)
                _clear_exception_context(exc)
                del exc
            finally:
                _empty_cuda_cache()

        update_status(db_path, job_id, JobStatus.READY_FOR_REVIEW, set_finished=True)
    except DopDownloadError as exc:
        update_status(
            db_path,
            job_id,
            JobStatus.FAILED,
            error_reason=_download_error_reason(exc),
            error_message=_traceback_tail(exc),
            set_finished=True,
        )
        shutil.rmtree(out_dir, ignore_errors=True)
    except Exception as exc:  # noqa: BLE001
        update_status(
            db_path,
            job_id,
            JobStatus.FAILED,
            error_reason=str(ErrorReason.INFERENCE_ERROR),
            error_message=_traceback_tail(exc),
            set_finished=True,
        )
        shutil.rmtree(out_dir, ignore_errors=True)
