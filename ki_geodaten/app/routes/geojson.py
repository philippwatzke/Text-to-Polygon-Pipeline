from __future__ import annotations

import asyncio
import json
from functools import partial

from fastapi import APIRouter, HTTPException, Request, Response

from ki_geodaten.app.serialization import (
    build_missed_objects_geojson,
    build_nodata_geojson,
    build_polygons_geojson,
)
from ki_geodaten.jobs.store import (
    get_job,
    get_missed_objects_for_job,
    get_nodata_for_job,
    get_polygons_for_job,
)
from ki_geodaten.pipeline.geo_utils import transform_bbox_wgs84_to_utm

router = APIRouter()


def _viewport_bbox_utm(request: Request) -> tuple[float, float, float, float] | None:
    raw = request.query_params.get("bbox")
    if raw is None:
        return None
    try:
        parts = [float(part) for part in raw.split(",")]
    except ValueError as exc:
        raise HTTPException(422, "bbox must contain four comma-separated numbers") from exc
    if len(parts) != 4 or parts[0] >= parts[2] or parts[1] >= parts[3]:
        raise HTTPException(422, "bbox must be minLon,minLat,maxLon,maxLat")
    return transform_bbox_wgs84_to_utm(*parts)


async def _build_geojson(request: Request, job_id: str, target: str) -> Response:
    job = get_job(request.app.state.db_path, job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    if job["status"] not in {"READY_FOR_REVIEW", "EXPORTED"}:
        raise HTTPException(409, "job not ready")

    revision = job["validation_revision"] if target in {"polygons", "missed_objects"} else 0
    clip_utm = _viewport_bbox_utm(request)
    clip_key = None if clip_utm is None else tuple(round(value, 2) for value in clip_utm)
    cache_key = (job_id, revision, target, clip_key)
    cached = request.app.state.geojson_cache.get(cache_key)
    if cached is not None:
        return Response(content=cached, media_type="application/json")

    aoi_utm = tuple(json.loads(job["bbox_utm_snapped"]))
    if target == "polygons":
        rows = get_polygons_for_job(request.app.state.db_path, job_id)
        fn = partial(build_polygons_geojson, rows, aoi_utm=aoi_utm, clip_utm=clip_utm)
    elif target == "missed_objects":
        rows = get_missed_objects_for_job(request.app.state.db_path, job_id)
        fn = partial(build_missed_objects_geojson, rows, aoi_utm=aoi_utm, clip_utm=clip_utm)
    else:
        rows = get_nodata_for_job(request.app.state.db_path, job_id)
        fn = partial(build_nodata_geojson, rows, aoi_utm=aoi_utm, clip_utm=clip_utm)

    loop = asyncio.get_running_loop()
    payload = await loop.run_in_executor(request.app.state.geojson_executor, fn)
    request.app.state.geojson_cache[cache_key] = payload
    return Response(content=payload, media_type="application/json")


@router.get("/jobs/{job_id}/polygons")
async def polygons_geojson(job_id: str, request: Request):
    return await _build_geojson(request, job_id, "polygons")


@router.get("/jobs/{job_id}/nodata")
async def nodata_geojson(job_id: str, request: Request):
    return await _build_geojson(request, job_id, "nodata")


@router.get("/jobs/{job_id}/missed_objects")
async def missed_objects_geojson(job_id: str, request: Request):
    return await _build_geojson(request, job_id, "missed_objects")
