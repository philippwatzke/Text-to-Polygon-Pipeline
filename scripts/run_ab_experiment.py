from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from ki_geodaten.app.run_metadata import build_run_metadata
from ki_geodaten.config import Settings
from ki_geodaten.jobs.store import init_schema, insert_job
from ki_geodaten.models import TilePreset
from ki_geodaten.pipeline.dop_client import prepare_download_bbox
from ki_geodaten.pipeline.geo_utils import transform_bbox_wgs84_to_utm
from ki_geodaten.pipeline.segmenter import Sam3Segmenter
from ki_geodaten.worker.orchestrator import run_job


DEFAULT_BBOX = [
    9.932037591934206,
    49.79387118434604,
    9.93916153907776,
    49.79764543675126,
]
DEFAULT_PROMPT = "car"
DEFAULT_PRESET = TilePreset.SMALL


EXPERIMENTS = (
    {
        "label": "ab_wcs_clahe",
        "description": "WCS with current CLAHE preprocessing",
        "dop_source": "wcs",
        "image_preprocess": "clahe",
    },
    {
        "label": "ab_wcs_none",
        "description": "WCS without image preprocessing",
        "dop_source": "wcs",
        "image_preprocess": "none",
    },
    {
        "label": "ab_wms_none",
        "description": "WMS-rendered diagnostic baseline without local preprocessing",
        "dop_source": "wms",
        "image_preprocess": "none",
    },
)


def _experiment_settings(base: Settings, experiment: dict) -> Settings:
    return base.model_copy(
        update={
            "DOP_SOURCE": experiment["dop_source"],
            "SAM_IMAGE_PREPROCESS": experiment["image_preprocess"],
        }
    )


def _create_job(
    settings: Settings,
    *,
    label: str,
    description: str,
    prompt: str,
    bbox_wgs84: list[float],
    tile_preset: TilePreset,
) -> str:
    utm_bounds = transform_bbox_wgs84_to_utm(*bbox_wgs84)
    prepared = prepare_download_bbox(
        *utm_bounds,
        preset=tile_preset,
        origin_x=settings.WCS_GRID_ORIGIN_X,
        origin_y=settings.WCS_GRID_ORIGIN_Y,
    )
    job_id = str(uuid.uuid4())
    metadata = build_run_metadata(
        settings,
        tile_preset=tile_preset,
        extra={
            "experiment_label": label,
            "experiment_description": description,
            "experiment_created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    insert_job(
        settings.DB_PATH,
        job_id=job_id,
        prompt=prompt,
        bbox_wgs84=list(bbox_wgs84),
        bbox_utm_snapped=list(prepared.aoi_bbox),
        tile_preset=tile_preset,
        run_metadata=metadata,
    )
    return job_id


def _run_one(settings: Settings, job_id: str) -> None:
    segmenter = Sam3Segmenter(
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
    )
    run_job(
        settings.DB_PATH,
        job_id=job_id,
        segmenter=segmenter,
        data_root=settings.DATA_DIR,
        wcs_url=settings.WCS_URL,
        coverage_id=settings.WCS_COVERAGE_ID,
        max_pixels=settings.WCS_MAX_PIXELS,
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
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed WCS/WMS segmentation A/B experiment.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--bbox-wgs84",
        nargs=4,
        type=float,
        default=DEFAULT_BBOX,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
    )
    parser.add_argument(
        "--tile-preset",
        choices=[item.value for item in TilePreset],
        default=DEFAULT_PRESET.value,
    )
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    base_settings = Settings()
    init_schema(base_settings.DB_PATH)
    tile_preset = TilePreset(args.tile_preset)

    manifest: list[dict] = []
    for experiment in EXPERIMENTS:
        settings = _experiment_settings(base_settings, experiment)
        job_id = _create_job(
            settings,
            label=experiment["label"],
            description=experiment["description"],
            prompt=args.prompt,
            bbox_wgs84=args.bbox_wgs84,
            tile_preset=tile_preset,
        )
        print(f"running {experiment['label']} job={job_id}", flush=True)
        _run_one(settings, job_id)
        manifest.append({**experiment, "job_id": job_id})

    manifest_path = args.manifest
    if manifest_path is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        manifest_path = base_settings.RESULTS_DIR / f"ab_experiment_{stamp}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "jobs": manifest}, indent=2), flush=True)


if __name__ == "__main__":
    main()
