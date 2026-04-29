from __future__ import annotations

import json
import logging
import shutil
import traceback
from pathlib import Path

import geopandas as gpd
import rasterio
from shapely.wkb import dumps as wkb_dumps
from shapely.wkb import loads as wkb_loads

from ki_geodaten.jobs.store import (
    get_job,
    get_polygons_for_job,
    increment_tile_completed,
    increment_tile_failed,
    insert_nodata_region,
    insert_polygons,
    replace_polygons_for_job,
    update_status,
)
from ki_geodaten.models import ErrorReason, JobStatus, NoDataReason, TilePreset
from ki_geodaten.pipeline.dem_client import (
    DemDownloadError,
    derive_ndsm_from_dom_dgm,
    fetch_tiles_via_metalink,
)
from ki_geodaten.pipeline.dop_client import DopDownloadError, download_dop20, prepare_download_bbox
from ki_geodaten.pipeline.merger import global_polygon_nms, keep_center_only, masks_to_polygons
from ki_geodaten.pipeline.modality_filter import ModalityThresholds, filter_masks
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


def _apply_global_polygon_nms(
    db_path: Path,
    job_id: str,
    *,
    iou_threshold: float,
    containment_ratio: float,
    fragment_coverage_ratio: float,
    fragment_max_area_ratio: float,
    fragment_buffer_m: float,
) -> None:
    rows = get_polygons_for_job(db_path, job_id)
    if len(rows) <= 1:
        return

    records: list[dict] = []
    geometries = []
    for row in rows:
        geometries.append(wkb_loads(row["geometry_wkb"]))
        records.append(
            {
                "score": float(row["score"]),
                "source_tile_row": int(row["source_tile_row"]),
                "source_tile_col": int(row["source_tile_col"]),
            }
        )

    gdf = gpd.GeoDataFrame(records, geometry=geometries, crs="EPSG:25832")
    kept = global_polygon_nms(
        gdf,
        iou_threshold=iou_threshold,
        containment_ratio=containment_ratio,
        fragment_coverage_ratio=fragment_coverage_ratio,
        fragment_max_area_ratio=fragment_max_area_ratio,
        fragment_buffer_m=fragment_buffer_m,
    )
    if len(kept) == len(gdf):
        return

    replacement_rows = [
        {
            "geometry_wkb": wkb_dumps(row.geometry),
            "score": float(row["score"]),
            "source_tile_row": int(row["source_tile_row"]),
            "source_tile_col": int(row["source_tile_col"]),
        }
        for _, row in kept.iterrows()
    ]
    replace_polygons_for_job(db_path, job_id, replacement_rows)


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


def _load_modality_thresholds(job: dict) -> ModalityThresholds:
    raw = job.get("modality_filter")
    if not raw:
        return ModalityThresholds()
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        return ModalityThresholds()
    return ModalityThresholds(
        ndvi_min=payload.get("ndvi_min"),
        ndvi_max=payload.get("ndvi_max"),
        ndsm_min=payload.get("ndsm_min"),
        ndsm_max=payload.get("ndsm_max"),
    )


def run_job(
    db_path: Path,
    *,
    job_id: str,
    segmenter,
    data_root: Path,
    wcs_url: str,
    coverage_id: str,
    max_pixels: int,
    wcs_version: str,
    fmt: str,
    crs: str,
    origin_x: float,
    origin_y: float,
    min_polygon_area_m2: float,
    safe_center_nodata_threshold: float,
    wcs_username: str = "",
    wcs_password: str = "",
    fill_wcs_rgb_zero_with_wms: bool = False,
    wms_url: str = "",
    wms_layer: str = "",
    wms_version: str = "1.1.1",
    wms_format: str = "image/png",
    dop_source: str = "wcs",
    dgm_metalink_url: str = "",
    dgm_tile_cache_dir: Path | None = None,
    dgm_native_step_m: float = 1.0,
    dom_metalink_url: str = "",
    dom_tile_cache_dir: Path | None = None,
    dom_native_step_m: float = 1.0,
    global_polygon_nms_iou: float = 0.5,
    global_polygon_containment_ratio: float = 0.85,
    global_polygon_fragment_coverage_ratio: float = 0.65,
    global_polygon_fragment_max_area_ratio: float = 0.75,
    global_polygon_fragment_buffer_m: float = 1.0,
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
            wcs_url=wcs_url,
            coverage_id=coverage_id,
            wcs_version=wcs_version,
            fmt=fmt,
            crs=crs,
            max_pixels=max_pixels,
            origin_x=origin_x,
            origin_y=origin_y,
            username=wcs_username,
            password=wcs_password,
            fill_rgb_zero_with_wms=fill_wcs_rgb_zero_with_wms,
            wms_url=wms_url,
            wms_layer=wms_layer,
            wms_version=wms_version,
            wms_format=wms_format,
            source=dop_source,
        )

        update_status(db_path, job_id, JobStatus.INFERRING, dop_vrt_path=str(vrt_path))
        with rasterio.open(vrt_path) as src:
            tile_total = sum(1 for _ in iter_grid(src, cfg))
        update_status(db_path, job_id, JobStatus.INFERRING, tile_total=tile_total)

        thresholds = _load_modality_thresholds(job)
        ndsm_path: Path | None = None
        if (
            thresholds.needs_ndsm()
            and dgm_metalink_url
            and dom_metalink_url
        ):
            ndsm_path = out_dir / "ndsm.tif"
            try:
                # Use the *expanded* download bbox (with center-margin) so that
                # tiles at the AOI edge still have nDSM coverage instead of
                # falling into the DEM-NoData zone after WarpedVRT resampling.
                dom_result = fetch_tiles_via_metalink(
                    bbox_utm=prepared.download_bbox,
                    out_dir=out_dir,
                    metalink_url=dom_metalink_url,
                    cache_dir=dom_tile_cache_dir,
                    vrt_name="dom.vrt",
                )
                dgm_result = fetch_tiles_via_metalink(
                    bbox_utm=prepared.download_bbox,
                    out_dir=out_dir,
                    metalink_url=dgm_metalink_url,
                    cache_dir=dgm_tile_cache_dir,
                    vrt_name="dgm.vrt",
                )
                derive_ndsm_from_dom_dgm(
                    dom_path=dom_result.path,
                    dgm_path=dgm_result.path,
                    out_path=ndsm_path,
                )
            except DemDownloadError:
                logger.exception(
                    "DOM/DGM fetch failed for job=%s; continuing without nDSM filter",
                    job_id,
                )
                ndsm_path = None
            except Exception:  # noqa: BLE001
                logger.exception(
                    "nDSM derivation failed for job=%s; continuing without nDSM filter",
                    job_id,
                )
                ndsm_path = None

        for tile in iter_tiles(
            vrt_path,
            cfg,
            safe_center_nodata_threshold=safe_center_nodata_threshold,
            read_nir=thresholds.needs_nir(),
            ndsm_path=ndsm_path,
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
                if thresholds.is_active():
                    red = tile.array[..., 0] if tile.array.shape[-1] >= 1 else None
                    kept = filter_masks(
                        kept,
                        red_channel=red,
                        nir_channel=tile.nir,
                        ndsm=tile.ndsm,
                        thresholds=thresholds,
                    )
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

        _apply_global_polygon_nms(
            db_path,
            job_id,
            iou_threshold=global_polygon_nms_iou,
            containment_ratio=global_polygon_containment_ratio,
            fragment_coverage_ratio=global_polygon_fragment_coverage_ratio,
            fragment_max_area_ratio=global_polygon_fragment_max_area_ratio,
            fragment_buffer_m=global_polygon_fragment_buffer_m,
        )
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
