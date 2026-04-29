from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box
from shapely.wkb import dumps as wkb_dumps

from ki_geodaten.app.run_metadata import build_run_metadata
from ki_geodaten.config import Settings
from ki_geodaten.jobs.store import (
    increment_tile_completed,
    init_schema,
    insert_job,
    insert_polygons,
    update_status,
)
from ki_geodaten.models import JobStatus, TilePreset
from ki_geodaten.pipeline.dem_client import (
    derive_ndsm_from_dom_dgm,
    fetch_tiles_via_metalink,
)
from ki_geodaten.pipeline.dop_client import download_dop20, prepare_download_bbox
from ki_geodaten.pipeline.exporter import export_two_layer_gpkg
from ki_geodaten.pipeline.geo_utils import transform_bbox_wgs84_to_utm
from ki_geodaten.pipeline.heuristic_baseline import (
    HeuristicBaselineConfig,
    build_heuristic_polygons,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a reviewable heuristic baseline job from nDSM/NDVI raster "
            "thresholds, without running SAM."
        )
    )
    parser.add_argument("--prompt", default="building")
    parser.add_argument(
        "--bbox-wgs84",
        nargs=4,
        type=float,
        required=True,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
    )
    parser.add_argument(
        "--tile-preset",
        choices=[item.value for item in TilePreset],
        default=TilePreset.SMALL.value,
        help="Used only for the same WCS-grid AOI snapping as regular jobs.",
    )
    parser.add_argument("--label", default="heuristic_baseline")
    parser.add_argument("--ndsm-min", type=float, default=3.0)
    parser.add_argument("--ndsm-max", type=float, default=None)
    parser.add_argument("--ndvi-min", type=float, default=None)
    parser.add_argument("--ndvi-max", type=float, default=None)
    parser.add_argument("--min-area-m2", type=float, default=10.0)
    parser.add_argument("--close-m", type=float, default=1.0)
    parser.add_argument("--open-m", type=float, default=0.0)
    parser.add_argument(
        "--export-gpkg",
        action="store_true",
        help="Also write a GeoPackage immediately and mark the baseline job EXPORTED.",
    )
    parser.add_argument("--out-gpkg", type=Path, default=None)
    return parser.parse_args()


def _empty_nodata_gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame({"reason": []}, geometry=[], crs="EPSG:25832")


def _insert_polygons(settings: Settings, job_id: str, gdf: gpd.GeoDataFrame) -> None:
    rows = [
        {
            "geometry_wkb": wkb_dumps(row.geometry),
            "score": float(row["score"]),
            "source_tile_row": int(row["source_tile_row"]),
            "source_tile_col": int(row["source_tile_col"]),
        }
        for _, row in gdf.iterrows()
    ]
    insert_polygons(settings.DB_PATH, job_id, rows)


def _download_ndsm_inputs(
    settings: Settings,
    *,
    bbox_utm: tuple[float, float, float, float],
    out_dir: Path,
) -> Path:
    dom = fetch_tiles_via_metalink(
        bbox_utm=bbox_utm,
        out_dir=out_dir,
        metalink_url=settings.DOM_METALINK_URL,
        cache_dir=settings.DOM_TILE_CACHE_DIR,
        vrt_name="dom.vrt",
    )
    dgm = fetch_tiles_via_metalink(
        bbox_utm=bbox_utm,
        out_dir=out_dir,
        metalink_url=settings.DGM_METALINK_URL,
        cache_dir=settings.DGM_TILE_CACHE_DIR,
        vrt_name="dgm.vrt",
    )
    ndsm_path = out_dir / "ndsm.tif"
    derive_ndsm_from_dom_dgm(dom_path=dom.path, dgm_path=dgm.path, out_path=ndsm_path)
    return ndsm_path


def _download_dop_if_needed(
    settings: Settings,
    *,
    bbox_utm: tuple[float, float, float, float],
    out_dir: Path,
    config: HeuristicBaselineConfig,
) -> Path | None:
    if not config.needs_ndvi():
        return None
    return download_dop20(
        bbox_utm,
        out_dir=out_dir / "dop",
        wcs_url=settings.WCS_URL,
        coverage_id=settings.WCS_COVERAGE_ID,
        wcs_version=settings.WCS_VERSION,
        fmt=settings.WCS_FORMAT,
        crs=settings.WCS_CRS,
        max_pixels=settings.WCS_MAX_PIXELS,
        origin_x=settings.WCS_GRID_ORIGIN_X,
        origin_y=settings.WCS_GRID_ORIGIN_Y,
        username=settings.WCS_USERNAME,
        password=settings.WCS_PASSWORD,
        fill_rgb_zero_with_wms=settings.FILL_WCS_RGB_ZERO_WITH_WMS,
        wms_url=settings.WMS_URL,
        wms_layer=settings.WMS_LAYER,
        wms_version=settings.WMS_VERSION,
        wms_format=settings.WMS_FORMAT,
        source=settings.DOP_SOURCE,
    )


def main() -> None:
    args = _parse_args()
    settings = Settings()
    init_schema(settings.DB_PATH)

    tile_preset = TilePreset(args.tile_preset)
    config = HeuristicBaselineConfig(
        ndsm_min=args.ndsm_min,
        ndsm_max=args.ndsm_max,
        ndvi_min=args.ndvi_min,
        ndvi_max=args.ndvi_max,
        min_area_m2=args.min_area_m2,
        close_m=args.close_m,
        open_m=args.open_m,
    )
    utm_bounds = transform_bbox_wgs84_to_utm(*args.bbox_wgs84)
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
            "experiment_label": args.label,
            "experiment_type": "heuristic_baseline",
            "experiment_created_at": datetime.now(timezone.utc).isoformat(),
            "heuristic_baseline": config.as_metadata(),
        },
    )
    insert_job(
        settings.DB_PATH,
        job_id=job_id,
        prompt=f"{args.label}: {args.prompt}",
        bbox_wgs84=list(args.bbox_wgs84),
        bbox_utm_snapped=list(prepared.aoi_bbox),
        tile_preset=tile_preset,
        run_metadata=metadata,
    )

    out_dir = settings.DATA_DIR / "heuristic" / job_id
    try:
        update_status(settings.DB_PATH, job_id, JobStatus.DOWNLOADING, set_started=True)
        ndsm_path = _download_ndsm_inputs(
            settings,
            bbox_utm=prepared.aoi_bbox,
            out_dir=out_dir,
        )
        dop_path = _download_dop_if_needed(
            settings,
            bbox_utm=prepared.aoi_bbox,
            out_dir=out_dir,
            config=config,
        )
        update_status(settings.DB_PATH, job_id, JobStatus.INFERRING, tile_total=1)
        gdf = build_heuristic_polygons(
            ndsm_path=ndsm_path,
            dop_path=dop_path,
            aoi_bbox_utm=prepared.aoi_bbox,
            config=config,
        )
        _insert_polygons(settings, job_id, gdf)
        increment_tile_completed(settings.DB_PATH, job_id)
        update_status(
            settings.DB_PATH,
            job_id,
            JobStatus.READY_FOR_REVIEW,
            tile_total=1,
            set_finished=True,
        )

        gpkg_path = None
        if args.export_gpkg:
            gpkg_path = args.out_gpkg or (settings.RESULTS_DIR / f"{job_id}.gpkg")
            export_two_layer_gpkg(
                gdf,
                _empty_nodata_gdf(),
                box(*prepared.aoi_bbox),
                gpkg_path,
            )
            update_status(
                settings.DB_PATH,
                job_id,
                JobStatus.EXPORTED,
                gpkg_path=str(gpkg_path),
                exported_revision=0,
            )

        print(
            json.dumps(
                {
                    "job_id": job_id,
                    "status": "EXPORTED" if gpkg_path else "READY_FOR_REVIEW",
                    "polygon_count": int(len(gdf)),
                    "ndsm_path": str(ndsm_path),
                    "dop_path": str(dop_path) if dop_path else None,
                    "gpkg_path": str(gpkg_path) if gpkg_path else None,
                    "heuristic_baseline": config.as_metadata(),
                },
                indent=2,
            ),
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        update_status(
            settings.DB_PATH,
            job_id,
            JobStatus.FAILED,
            error_reason="INFERENCE_ERROR",
            error_message=str(exc),
            set_finished=True,
        )
        raise


if __name__ == "__main__":
    main()
