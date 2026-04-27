from __future__ import annotations

import logging
import math
import os
import shutil
import time
from pathlib import Path
from typing import Callable

from ki_geodaten.jobs.retention import cleanup_old_jobs
from ki_geodaten.jobs.store import abort_incomplete_jobs_on_startup, claim_next_pending_job, connect
from ki_geodaten.worker.orchestrator import run_job

logger = logging.getLogger(__name__)


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
    max_jobs: int,
    poll_interval: float,
    idle_exit_after: int | None = None,
    results_dir: Path | None = None,
    retention_days: int | None = None,
    max_runtime_seconds: float | None = None,
    clock: Callable[[], float] = time.monotonic,
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
                wms_url=wms_url,
                layer=layer,
                max_pixels=max_pixels,
                wms_version=wms_version,
                fmt=fmt,
                crs=crs,
                origin_x=origin_x,
                origin_y=origin_y,
                min_polygon_area_m2=min_polygon_area_m2,
                safe_center_nodata_threshold=safe_center_nodata_threshold,
            )
        except Exception:
            logger.exception("job crashed outside orchestrator try/except: %s", job["id"])
        processed += 1


def main() -> None:  # pragma: no cover
    from ki_geodaten.config import Settings
    from ki_geodaten.pipeline.segmenter import Sam3Segmenter

    settings = Settings()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    run_forever(
        db_path=settings.DB_PATH,
        data_root=settings.DATA_DIR,
        segmenter_factory=lambda: Sam3Segmenter(
            settings.SAM3_MODEL_ID,
            iou_threshold=settings.LOCAL_MASK_NMS_IOU,
            containment_ratio=settings.LOCAL_MASK_CONTAINMENT_RATIO,
        ),
        wms_url=settings.WMS_URL,
        layer=settings.WMS_LAYER,
        max_pixels=settings.WMS_MAX_PIXELS,
        wms_version=settings.WMS_VERSION,
        fmt=settings.WMS_FORMAT,
        crs=settings.WMS_CRS,
        origin_x=settings.WMS_GRID_ORIGIN_X,
        origin_y=settings.WMS_GRID_ORIGIN_Y,
        min_polygon_area_m2=settings.MIN_POLYGON_AREA_M2,
        safe_center_nodata_threshold=settings.SAFE_CENTER_NODATA_THRESHOLD,
        max_jobs=settings.MAX_JOBS_PER_WORKER,
        poll_interval=settings.WORKER_POLL_INTERVAL_SEC,
        results_dir=settings.RESULTS_DIR,
        retention_days=settings.RETENTION_DAYS,
        max_runtime_seconds=settings.MAX_WORKER_RUNTIME_SECONDS,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
