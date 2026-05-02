from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import geopandas as gpd
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from shapely.geometry import Point, box
from shapely.wkb import loads as wkb_loads
from shapely.wkb import dumps as wkb_dumps

from ki_geodaten.config import Settings
from ki_geodaten.jobs.store import (
    get_job,
    get_job_summary,
    get_missed_objects_for_job,
    get_nodata_for_job,
    get_polygons_for_job,
    insert_job,
    insert_missed_object,
    list_jobs,
    delete_missed_object,
    delete_job,
    update_job_label,
    update_status,
    update_missed_estimate,
    validate_bulk,
)
from ki_geodaten.models import (
    CreateJobRequest,
    JobLabelRequest,
    JobStatus,
    MissedObjectRequest,
    MissedEstimateRequest,
    ValidateBulkRequest,
)
import logging

logger = logging.getLogger(__name__)
from ki_geodaten.app.run_metadata import build_run_metadata
from ki_geodaten.pipeline.dop_client import prepare_download_bbox
from ki_geodaten.pipeline.exporter import export_two_layer_gpkg
from ki_geodaten.pipeline.geo_utils import transform_bbox_wgs84_to_utm, transformer_4326_to_25832

router = APIRouter()


def _within_bayern(bbox: list[float], bayern: tuple[float, float, float, float]) -> bool:
    lon_min, lat_min, lon_max, lat_max = bayern
    return bbox[0] >= lon_min and bbox[2] <= lon_max and bbox[1] >= lat_min and bbox[3] <= lat_max


def _utm_area_km2(bbox_wgs84: list[float]) -> float:
    minx, miny, maxx, maxy = transform_bbox_wgs84_to_utm(*bbox_wgs84)
    return (maxx - minx) * (maxy - miny) / 1_000_000.0


def _job_view(job: dict) -> dict:
    fields = (
        "id",
        "label",
        "prompt",
        "tile_preset",
        "status",
        "error_reason",
        "error_message",
        "tile_completed",
        "tile_failed",
        "tile_total",
        "validation_revision",
        "exported_revision",
        "created_at",
        "started_at",
        "finished_at",
        "missed_estimate",
        "bbox_wgs84",
        "run_metadata",
        "modality_filter",
    )
    view = {field: job.get(field) for field in fields}
    if view.get("bbox_wgs84"):
        view["bbox_wgs84"] = json.loads(view["bbox_wgs84"])
    if view.get("run_metadata"):
        try:
            view["run_metadata"] = json.loads(view["run_metadata"])
        except (TypeError, ValueError):
            view["run_metadata"] = None
    if view.get("modality_filter"):
        try:
            view["modality_filter"] = json.loads(view["modality_filter"])
        except (TypeError, ValueError):
            view["modality_filter"] = None
    exported = job.get("exported_revision")
    validation = job.get("validation_revision") or 0
    view["export_stale"] = exported is None or exported < validation
    return view


@router.post("/jobs")
async def create_job(req: CreateJobRequest, request: Request):
    settings: Settings = request.app.state.settings
    bbox = req.bbox_wgs84
    if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        raise HTTPException(422, "bbox_wgs84 must have minx<maxx and miny<maxy")
    if not _within_bayern(bbox, settings.BAYERN_BBOX_WGS84):
        raise HTTPException(422, "bbox must be within Bayern")

    area_km2 = _utm_area_km2(bbox)
    if area_km2 > settings.MAX_BBOX_AREA_KM2:
        raise HTTPException(422, f"bbox area {area_km2:.3f} km2 exceeds limit")

    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(422, "prompt must be non-empty")
    if len(prompt) > settings.MAX_PROMPT_CHARS:
        raise HTTPException(422, "prompt too long")
    token_count = request.app.state.token_counter(prompt)
    if token_count > settings.MAX_ENCODER_CONTEXT_TOKENS:
        raise HTTPException(422, "prompt exceeds encoder token limit")

    utm_bounds = transform_bbox_wgs84_to_utm(*bbox)
    prepared = prepare_download_bbox(
        *utm_bounds,
        preset=req.tile_preset,
        origin_x=settings.WCS_GRID_ORIGIN_X,
        origin_y=settings.WCS_GRID_ORIGIN_Y,
    )
    if req.modality_filter.needs_nir() and not settings.MODALITY_USE_NDVI:
        raise HTTPException(422, "NDVI filter requested but MODALITY_USE_NDVI is disabled")
    if req.modality_filter.needs_ndsm() and not settings.MODALITY_USE_NDSM:
        raise HTTPException(422, "nDSM filter requested but MODALITY_USE_NDSM is disabled")
    if req.modality_filter.needs_ndsm() and not (
        settings.DGM_METALINK_URL
        and settings.DOM_METALINK_URL
    ):
        raise HTTPException(
            422,
            "nDSM filter requested but DGM_METALINK_URL and DOM_METALINK_URL "
            "are not configured",
        )

    job_id = str(uuid.uuid4())
    modality_filter_dict = req.modality_filter.model_dump()
    vector_options_dict = req.vector_options.model_dump()
    run_metadata = build_run_metadata(
        settings,
        tile_preset=req.tile_preset,
        extra={
            "modality_filter": modality_filter_dict,
            "vector_options": vector_options_dict,
        },
    )
    insert_job(
        request.app.state.db_path,
        job_id=job_id,
        prompt=prompt,
        bbox_wgs84=list(bbox),
        bbox_utm_snapped=list(prepared.aoi_bbox),
        tile_preset=req.tile_preset,
        run_metadata=run_metadata,
        modality_filter=modality_filter_dict if req.modality_filter.is_active() else None,
    )
    return {"id": job_id, "status": "PENDING"}


@router.get("/jobs")
async def list_jobs_endpoint(request: Request):
    return [_job_view(job) for job in list_jobs(request.app.state.db_path)]


@router.get("/jobs/{job_id}")
async def get_job_endpoint(job_id: str, request: Request):
    job = get_job(request.app.state.db_path, job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return _job_view(job)


@router.delete("/jobs/{job_id}")
def delete_job_endpoint(job_id: str, request: Request):
    job = get_job(request.app.state.db_path, job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    if job["status"] in {"DOWNLOADING", "INFERRING"}:
        raise HTTPException(409, "running jobs cannot be deleted")

    deleted = delete_job(request.app.state.db_path, job_id)
    if deleted:
        shutil.rmtree(request.app.state.data_root / "dop" / job_id, ignore_errors=True)
        gpkg_path = job.get("gpkg_path")
        if gpkg_path:
            Path(gpkg_path).unlink(missing_ok=True)
            Path(gpkg_path).with_suffix(".gpkg-wal").unlink(missing_ok=True)
            Path(gpkg_path).with_suffix(".gpkg-shm").unlink(missing_ok=True)
        for key in list(request.app.state.geojson_cache.keys()):
            if key[0] == job_id:
                del request.app.state.geojson_cache[key]
    return {"deleted": deleted}


@router.get("/jobs/{job_id}/summary")
async def get_job_summary_endpoint(job_id: str, request: Request):
    summary = get_job_summary(request.app.state.db_path, job_id)
    if summary is None:
        raise HTTPException(404, "job not found")
    return summary


@router.patch("/jobs/{job_id}/label")
def update_job_label_endpoint(job_id: str, req: JobLabelRequest, request: Request):
    job = get_job(request.app.state.db_path, job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    update_job_label(request.app.state.db_path, job_id, req.label)
    updated = get_job(request.app.state.db_path, job_id)
    return _job_view(updated)


@router.post("/jobs/{job_id}/missed_estimate")
def update_missed_estimate_endpoint(
    job_id: str,
    req: MissedEstimateRequest,
    request: Request,
):
    job = get_job(request.app.state.db_path, job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    update_missed_estimate(
        request.app.state.db_path,
        job_id,
        req.missed_estimate,
    )
    return {"missed_estimate": req.missed_estimate}


@router.post("/jobs/{job_id}/missed_objects")
def create_missed_object(job_id: str, req: MissedObjectRequest, request: Request):
    job = get_job(request.app.state.db_path, job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    if job["status"] not in {"READY_FOR_REVIEW", "EXPORTED"}:
        raise HTTPException(409, "job status does not allow missed object annotation")

    transformer = transformer_4326_to_25832()
    x, y = transformer.transform(req.lon, req.lat)
    point = Point(x, y)
    if not box(*json.loads(job["bbox_utm_snapped"])).covers(point):
        raise HTTPException(422, "missed object must be inside the job bbox")

    missed_id = insert_missed_object(
        request.app.state.db_path,
        job_id,
        geometry_wkb=wkb_dumps(point),
    )
    for key in list(request.app.state.geojson_cache.keys()):
        if key[0] == job_id and key[2] == "missed_objects":
            del request.app.state.geojson_cache[key]
    return {"id": missed_id}


@router.delete("/jobs/{job_id}/missed_objects/{missed_id}")
def remove_missed_object(job_id: str, missed_id: int, request: Request):
    job = get_job(request.app.state.db_path, job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    if job["status"] not in {"READY_FOR_REVIEW", "EXPORTED"}:
        raise HTTPException(409, "job status does not allow missed object annotation")
    deleted = delete_missed_object(request.app.state.db_path, job_id, missed_id)
    if not deleted:
        raise HTTPException(404, "missed object not found")
    for key in list(request.app.state.geojson_cache.keys()):
        if key[0] == job_id and key[2] == "missed_objects":
            del request.app.state.geojson_cache[key]
    return {"deleted": deleted}


@router.get("/jobs/{job_id}/export.gpkg")
async def download_gpkg(job_id: str, request: Request):
    job = get_job(request.app.state.db_path, job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    gpkg_path = job.get("gpkg_path")
    if not gpkg_path or not Path(gpkg_path).exists():
        raise HTTPException(404, "gpkg not generated")
    return FileResponse(
        gpkg_path,
        media_type="application/geopackage+sqlite3",
        filename=f"{job_id}.gpkg",
    )


@router.post("/jobs/{job_id}/polygons/validate_bulk")
def validate_bulk_endpoint(job_id: str, req: ValidateBulkRequest, request: Request):
    job = get_job(request.app.state.db_path, job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    if job["status"] not in {"READY_FOR_REVIEW", "EXPORTED"}:
        raise HTTPException(409, "job status does not allow validation")
    updated = validate_bulk(
        request.app.state.db_path,
        job_id,
        [update.model_dump() for update in req.updates],
    )
    for key in list(request.app.state.geojson_cache.keys()):
        if key[0] == job_id and key[2] == "polygons":
            del request.app.state.geojson_cache[key]
    return {"updated": updated}


def _rows_to_gdf(rows: list[dict], kind: str) -> gpd.GeoDataFrame:
    if kind == "polygons":
        attrs = [
            {
                "score": row["score"],
                "source_tile_row": row["source_tile_row"],
                "source_tile_col": row["source_tile_col"],
            }
            for row in rows
            if row.get("validation") == "ACCEPTED"
        ]
        geoms = [wkb_loads(row["geometry_wkb"]) for row in rows if row.get("validation") == "ACCEPTED"]
    else:
        attrs = [{"reason": row["reason"]} for row in rows]
        geoms = [wkb_loads(row["geometry_wkb"]) for row in rows]
    return gpd.GeoDataFrame(attrs, geometry=geoms, crs="EPSG:25832")


def _missed_rows_to_gdf(rows: list[dict]) -> gpd.GeoDataFrame:
    attrs = [{"created_at": row["created_at"]} for row in rows]
    geoms = [wkb_loads(row["geometry_wkb"]) for row in rows]
    return gpd.GeoDataFrame(attrs, geometry=geoms, crs="EPSG:25832")


@router.post("/jobs/{job_id}/export")
def export_job(job_id: str, request: Request):
    with request.app.state.export_lock:
        job = get_job(request.app.state.db_path, job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        if job["status"] not in {"READY_FOR_REVIEW", "EXPORTED"}:
            raise HTTPException(409, "job not ready for export")
        out_path = request.app.state.results_dir / f"{job_id}.gpkg"
        try:
            aoi = box(*json.loads(job["bbox_utm_snapped"]))
            export_two_layer_gpkg(
                _rows_to_gdf(get_polygons_for_job(request.app.state.db_path, job_id), "polygons"),
                _rows_to_gdf(get_nodata_for_job(request.app.state.db_path, job_id), "nodata"),
                aoi,
                out_path,
                missed_gdf=_missed_rows_to_gdf(
                    get_missed_objects_for_job(request.app.state.db_path, job_id)
                ),
            )
        except Exception as exc:  # noqa: BLE001
            # Don't downgrade the job to FAILED — the user must remain able to
            # retry the export (Spec §6 allows EXPORTED→EXPORTED, and a
            # transient Fiona/GDAL hiccup should not lose review state).
            logger.exception("export failed for job %s", job_id)
            raise HTTPException(500, f"export failed: {exc}") from exc

        shutil.rmtree(request.app.state.data_root / "dop" / job_id, ignore_errors=True)
        update_status(
            request.app.state.db_path,
            job_id,
            JobStatus.EXPORTED,
            gpkg_path=str(out_path),
            exported_revision=job["validation_revision"],
            set_finished=True,
        )
        return {"gpkg_path": str(out_path)}
