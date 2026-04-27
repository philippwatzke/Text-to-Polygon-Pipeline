from pathlib import Path

import numpy as np
import pytest

fiona = pytest.importorskip("fiona")

import rasterio
from fastapi.testclient import TestClient
from rasterio.transform import from_bounds

from ki_geodaten.app.main import create_app
from ki_geodaten.jobs.store import get_job
from ki_geodaten.pipeline.dop_client import _build_vrt
from ki_geodaten.pipeline.segmenter import MaskResult
from ki_geodaten.worker.loop import run_forever


def _make_vrt(path: Path, bbox):
    tif = path.parent / "chunk_0_0.tif"
    width = round((bbox[2] - bbox[0]) / 0.2)
    height = round((bbox[3] - bbox[1]) / 0.2)
    with rasterio.open(
        tif,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=4,
        dtype="uint8",
        crs="EPSG:25832",
        transform=from_bounds(*bbox, width, height),
    ) as dst:
        dst.write(np.full((3, height, width), 128, dtype=np.uint8), indexes=[1, 2, 3])
        dst.write(np.full((height, width), 255, dtype=np.uint8), 4)
    _build_vrt(path, [tif], crs="EPSG:25832")


class StubSegmenter:
    def predict(self, tile, prompt):
        mask = np.zeros((tile.size, tile.size), dtype=bool)
        mask[500:524, 500:524] = True
        return [MaskResult(mask=mask, score=0.95, box_pixel=(500, 500, 524, 524))]

    def encoder_token_count(self, value):
        return len(value.split())


def test_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def fake_download(bbox_utm, *, out_dir, **kwargs):
        out_dir.mkdir(parents=True, exist_ok=True)
        vrt = out_dir / "out.vrt"
        _make_vrt(vrt, bbox_utm)
        return vrt

    monkeypatch.setattr("ki_geodaten.worker.orchestrator.download_dop20", fake_download)

    from concurrent.futures import ThreadPoolExecutor

    app = create_app(
        executor_factory=lambda: ThreadPoolExecutor(max_workers=2),
        token_counter=lambda value: len(value.split()),
    )
    with TestClient(app) as client:
        response = client.post(
            "/jobs",
            json={"prompt": "building", "bbox_wgs84": [11.55, 48.13, 11.56, 48.14]},
        )
        assert response.status_code == 200
        job_id = response.json()["id"]

        run_forever(
            db_path=app.state.db_path,
            data_root=app.state.data_root,
            segmenter_factory=StubSegmenter,
            wms_url="",
            layer="by_dop20c",
            max_pixels=6000,
            wms_version="1.1.1",
            fmt="image/png",
            crs="EPSG:25832",
            origin_x=0.0,
            origin_y=0.0,
            min_polygon_area_m2=0.01,
            safe_center_nodata_threshold=0.0,
            max_jobs=1,
            poll_interval=0.01,
            idle_exit_after=1,
        )

        job = get_job(app.state.db_path, job_id)
        assert job["status"] == "READY_FOR_REVIEW"

        fc = client.get(f"/jobs/{job_id}/polygons").json()
        assert fc["type"] == "FeatureCollection"
        assert len(fc["features"]) >= 1

        export_response = client.post(f"/jobs/{job_id}/export")
        assert export_response.status_code == 200
        gpkg_path = Path(export_response.json()["gpkg_path"])
        assert gpkg_path.exists()
        assert {"detected_objects", "nodata_regions"}.issubset(
            set(fiona.listlayers(str(gpkg_path)))
        )

        download = client.get(f"/jobs/{job_id}/export.gpkg")
        assert download.status_code == 200
        assert download.content
