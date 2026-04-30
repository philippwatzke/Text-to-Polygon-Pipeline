from __future__ import annotations

import logging
import math
import contextlib
import os
import shutil
import ctypes
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ki_geodaten.jobs.retention import cleanup_old_jobs
from ki_geodaten.jobs.store import abort_incomplete_jobs_on_startup, claim_next_pending_job, connect
from ki_geodaten.worker.orchestrator import run_job

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkerLock:
    path: Path
    fd: int

    def release(self) -> None:
        os.close(self.fd)
        self.path.unlink(missing_ok=True)


def acquire_worker_lock(path: Path) -> WorkerLock | None:
    """Acquire an exclusive process lock for the single-GPU worker.

    SQLite job claiming is atomic, but startup cleanup deliberately marks
    DOWNLOADING/INFERRING jobs as failed after a worker restart. Running two
    workers concurrently would therefore make the second one abort the first
    one's live job. An atomic lock file prevents that class of failure.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    stale_checked = False
    while True:
        try:
            fd = os.open(path, flags)
        except FileExistsError:
            if not stale_checked and _remove_stale_worker_lock(path):
                stale_checked = True
                continue
            logger.error("worker lock exists at %s; another worker may be running", path)
            return None
        os.write(fd, str(os.getpid()).encode("ascii"))
        return WorkerLock(path=path, fd=fd)


def _remove_stale_worker_lock(path: Path) -> bool:
    try:
        pid = int(path.read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        return False
    if _pid_exists(pid):
        return False
    try:
        path.unlink()
    except OSError:
        return False
    logger.warning("removed stale worker lock at %s for dead pid %s", path, pid)
    return True


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        process_query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            process_query_limited_information,
            False,
            pid,
        )
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _active_job_ids(db_path: Path) -> set[str]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id FROM jobs WHERE status IN ('PENDING','DOWNLOADING','INFERRING')"
        ).fetchall()
    return {row["id"] for row in rows}


def startup_cleanup(
    db_path: Path,
    *,
    data_root: Path,
    results_dir: Path | None = None,
    retention_days: int | None = None,
) -> None:
    abort_incomplete_jobs_on_startup(db_path)

    if retention_days is not None and results_dir is not None:
        try:
            cleanup_old_jobs(
                db_path,
                results_dir=results_dir,
                retention_days=retention_days,
            )
        except Exception:  # noqa: BLE001
            logger.exception("retention cleanup failed at startup")

    active = _active_job_ids(db_path)
    dop_root = data_root / "dop"
    if not dop_root.exists():
        return
    for child in dop_root.iterdir():
        if child.is_dir() and child.name not in active:
            shutil.rmtree(child, ignore_errors=True)


def run_forever(
    *,
    db_path: Path,
    data_root: Path,
    segmenter_factory: Callable[[], object],
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
    max_jobs: int,
    poll_interval: float,
    dop_download_workers: int = 4,
    idle_exit_after: int | None = None,
    results_dir: Path | None = None,
    retention_days: int | None = None,
    max_runtime_seconds: float | None = None,
    clock: Callable[[], float] = time.monotonic,
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
    startup_cleanup(
        db_path,
        data_root=data_root,
        results_dir=results_dir,
        retention_days=retention_days,
    )
    segmenter = segmenter_factory()
    processed = 0
    idle_polls = 0
    deadline = math.inf if max_runtime_seconds is None else clock() + max_runtime_seconds

    while processed < max_jobs:
        if clock() >= deadline:
            logger.info("worker reached wall-clock budget; exiting for supervisor restart")
            return

        job = claim_next_pending_job(db_path)
        if job is None:
            idle_polls += 1
            if idle_exit_after is not None and idle_polls >= idle_exit_after:
                return
            time.sleep(poll_interval)
            continue

        idle_polls = 0
        try:
            run_job(
                db_path,
                job_id=job["id"],
                segmenter=segmenter,
                data_root=data_root,
                wcs_url=wcs_url,
                coverage_id=coverage_id,
                max_pixels=max_pixels,
                dop_download_workers=dop_download_workers,
                wcs_version=wcs_version,
                fmt=fmt,
                crs=crs,
                origin_x=origin_x,
                origin_y=origin_y,
                min_polygon_area_m2=min_polygon_area_m2,
                safe_center_nodata_threshold=safe_center_nodata_threshold,
                wcs_username=wcs_username,
                wcs_password=wcs_password,
                fill_wcs_rgb_zero_with_wms=fill_wcs_rgb_zero_with_wms,
                wms_url=wms_url,
                wms_layer=wms_layer,
                wms_version=wms_version,
                wms_format=wms_format,
                dop_source=dop_source,
                dgm_metalink_url=dgm_metalink_url,
                dgm_tile_cache_dir=dgm_tile_cache_dir,
                dgm_native_step_m=dgm_native_step_m,
                dom_metalink_url=dom_metalink_url,
                dom_tile_cache_dir=dom_tile_cache_dir,
                dom_native_step_m=dom_native_step_m,
                global_polygon_nms_iou=global_polygon_nms_iou,
                global_polygon_containment_ratio=global_polygon_containment_ratio,
                global_polygon_fragment_coverage_ratio=global_polygon_fragment_coverage_ratio,
                global_polygon_fragment_max_area_ratio=global_polygon_fragment_max_area_ratio,
                global_polygon_fragment_buffer_m=global_polygon_fragment_buffer_m,
            )
        except Exception:
            logger.exception("job crashed outside orchestrator try/except: %s", job["id"])
        processed += 1


def main() -> None:  # pragma: no cover
    from ki_geodaten.config import Settings
    from ki_geodaten.pipeline.segmenter import Sam3Segmenter

    settings = Settings()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    worker_lock = acquire_worker_lock(settings.DATA_DIR / "worker.lock")
    if worker_lock is None:
        return
    with contextlib.ExitStack() as stack:
        stack.callback(worker_lock.release)
        _run_main(settings, Sam3Segmenter)


def _run_main(settings, segmenter_cls) -> None:
    run_forever(
        db_path=settings.DB_PATH,
        data_root=settings.DATA_DIR,
        segmenter_factory=lambda: segmenter_cls(
            settings.SAM3_MODEL_ID,
            iou_threshold=settings.LOCAL_MASK_NMS_IOU,
            containment_ratio=settings.LOCAL_MASK_CONTAINMENT_RATIO,
            image_preprocess=settings.SAM_IMAGE_PREPROCESS,
            clahe_clip_limit=settings.SAM_CLAHE_CLIP_LIMIT,
            clahe_tile_grid_size=settings.SAM_CLAHE_TILE_GRID_SIZE,
            clahe_blend=settings.SAM_CLAHE_BLEND,
            image_gamma=settings.SAM_IMAGE_GAMMA,
            image_brightness=settings.SAM_IMAGE_BRIGHTNESS,
            local_files_only=settings.SAM3_LOCAL_FILES_ONLY,
        ),
        wcs_url=settings.WCS_URL,
        coverage_id=settings.WCS_COVERAGE_ID,
        max_pixels=settings.WCS_MAX_PIXELS,
        dop_download_workers=settings.DOP_DOWNLOAD_WORKERS,
        wcs_version=settings.WCS_VERSION,
        fmt=settings.WCS_FORMAT,
        crs=settings.WCS_CRS,
        origin_x=settings.WCS_GRID_ORIGIN_X,
        origin_y=settings.WCS_GRID_ORIGIN_Y,
        wcs_username=settings.WCS_USERNAME,
        wcs_password=settings.WCS_PASSWORD,
        fill_wcs_rgb_zero_with_wms=settings.FILL_WCS_RGB_ZERO_WITH_WMS,
        wms_url=settings.WMS_URL,
        wms_layer=settings.WMS_LAYER,
        wms_version=settings.WMS_VERSION,
        wms_format=settings.WMS_FORMAT,
        dop_source=settings.DOP_SOURCE,
        min_polygon_area_m2=settings.MIN_POLYGON_AREA_M2,
        safe_center_nodata_threshold=settings.SAFE_CENTER_NODATA_THRESHOLD,
        max_jobs=settings.MAX_JOBS_PER_WORKER,
        poll_interval=settings.WORKER_POLL_INTERVAL_SEC,
        results_dir=settings.RESULTS_DIR,
        retention_days=settings.RETENTION_DAYS,
        max_runtime_seconds=settings.MAX_WORKER_RUNTIME_SECONDS,
        dgm_metalink_url=settings.DGM_METALINK_URL,
        dgm_tile_cache_dir=settings.DGM_TILE_CACHE_DIR,
        dgm_native_step_m=settings.DGM_NATIVE_STEP_M,
        dom_metalink_url=settings.DOM_METALINK_URL,
        dom_tile_cache_dir=settings.DOM_TILE_CACHE_DIR,
        dom_native_step_m=settings.DOM_NATIVE_STEP_M,
        global_polygon_nms_iou=settings.GLOBAL_POLYGON_NMS_IOU,
        global_polygon_containment_ratio=settings.GLOBAL_POLYGON_CONTAINMENT_RATIO,
        global_polygon_fragment_coverage_ratio=settings.GLOBAL_POLYGON_FRAGMENT_COVERAGE_RATIO,
        global_polygon_fragment_max_area_ratio=settings.GLOBAL_POLYGON_FRAGMENT_MAX_AREA_RATIO,
        global_polygon_fragment_buffer_m=settings.GLOBAL_POLYGON_FRAGMENT_BUFFER_M,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
