from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from shapely.geometry import Polygon
from shapely.wkb import dumps as wkb_dumps

from ki_geodaten.app.main import create_app
from ki_geodaten.jobs.store import get_job, insert_polygons, update_status
from ki_geodaten.models import JobStatus


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = create_app(
        executor_factory=lambda: ThreadPoolExecutor(max_workers=2),
        token_counter=lambda value: len(value.split()),
    )
    with TestClient(app) as test_client:
        yield test_client, app


def _munich_bbox():
    return [11.55, 48.13, 11.56, 48.14]


def test_post_jobs_accepts_valid_and_defaults_medium(client):
    test_client, app = client
    response = test_client.post("/jobs", json={"prompt": "building", "bbox_wgs84": _munich_bbox()})
    assert response.status_code == 200
    job = get_job(app.state.db_path, response.json()["id"])
    assert job is not None
    assert job["tile_preset"] == "medium"
    assert job["status"] == "PENDING"


def test_post_jobs_persists_run_metadata_snapshot(client):
    test_client, app = client
    response = test_client.post("/jobs", json={"prompt": "building", "bbox_wgs84": _munich_bbox()})
    assert response.status_code == 200
    job_id = response.json()["id"]

    detail = test_client.get(f"/jobs/{job_id}").json()
    metadata = detail["run_metadata"]
    assert metadata is not None
    assert metadata["tile_preset"] == "medium"
    assert "settings" in metadata
    assert metadata["settings"]["SAM3_MODEL_ID"] == app.state.settings.SAM3_MODEL_ID
    assert metadata["settings"]["WMS_LAYER"] == app.state.settings.WMS_LAYER
    assert "git_commit_sha" in metadata
    assert "package_version" in metadata


def test_post_jobs_rejects_outside_bayern(client):
    test_client, _ = client
    response = test_client.post(
        "/jobs",
        json={"prompt": "building", "bbox_wgs84": [13.30, 52.40, 13.31, 52.41]},
    )
    assert response.status_code == 422


def test_post_jobs_rejects_area_over_1sqkm(client):
    test_client, _ = client
    response = test_client.post(
        "/jobs",
        json={"prompt": "building", "bbox_wgs84": [11.55, 48.13, 11.60, 48.20]},
    )
    assert response.status_code == 422


def test_post_jobs_rejects_token_count_over_limit(client):
    test_client, app = client
    app.state.token_counter = lambda value: 999
    response = test_client.post(
        "/jobs",
        json={"prompt": "solar panel", "bbox_wgs84": _munich_bbox()},
    )
    assert response.status_code == 422


def _make_reviewable_job(test_client, app, n_polys=2, status=JobStatus.READY_FOR_REVIEW):
    job_id = test_client.post(
        "/jobs",
        json={"prompt": "building", "bbox_wgs84": _munich_bbox()},
    ).json()["id"]
    polys = []
    for index in range(n_polys):
        x0 = 691100 + index * 20
        poly = Polygon([(x0, 5335100), (x0 + 5, 5335100), (x0 + 5, 5335105), (x0, 5335105)])
        polys.append(
            {
                "geometry_wkb": wkb_dumps(poly),
                "score": 0.9,
                "source_tile_row": 0,
                "source_tile_col": index,
            }
        )
    insert_polygons(app.state.db_path, job_id, polys)
    update_status(app.state.db_path, job_id, status, set_finished=True)
    return job_id


def test_list_and_get_job_return_revision_fields(client):
    test_client, app = client
    job_id = _make_reviewable_job(test_client, app)
    listed = test_client.get("/jobs").json()
    entry = next(item for item in listed if item["id"] == job_id)
    assert entry["bbox_wgs84"] == _munich_bbox()
    assert entry["validation_revision"] == 0
    assert entry["exported_revision"] is None
    assert entry["export_stale"] is True
    detail = test_client.get(f"/jobs/{job_id}").json()
    assert detail["id"] == job_id


def test_job_summary_and_missed_estimate_endpoints(client):
    test_client, app = client
    job_id = _make_reviewable_job(test_client, app, n_polys=3)
    rejected = test_client.post(
        f"/jobs/{job_id}/polygons/validate_bulk",
        json={"updates": [{"pid": 1, "validation": "REJECTED"}]},
    )
    assert rejected.status_code == 200

    updated = test_client.post(
        f"/jobs/{job_id}/missed_estimate",
        json={"missed_estimate": 1},
    )
    assert updated.status_code == 200
    assert updated.json() == {"missed_estimate": 1}

    summary = test_client.get(f"/jobs/{job_id}/summary")
    assert summary.status_code == 200
    assert summary.json()["total"] == 3
    assert summary.json()["accepted"] == 2
    assert summary.json()["rejected"] == 1
    assert summary.json()["missed_estimate"] == 1
    assert summary.json()["precision_review"] == pytest.approx(2 / 3)
    assert summary.json()["recall_estimate"] == pytest.approx(2 / 3)


def test_missed_estimate_rejects_negative_values(client):
    test_client, app = client
    job_id = _make_reviewable_job(test_client, app)

    response = test_client.post(
        f"/jobs/{job_id}/missed_estimate",
        json={"missed_estimate": -1},
    )

    assert response.status_code == 422


def test_job_view_includes_error_details(client):
    test_client, app = client
    job_id = test_client.post(
        "/jobs",
        json={"prompt": "building", "bbox_wgs84": _munich_bbox()},
    ).json()["id"]
    update_status(
        app.state.db_path,
        job_id,
        JobStatus.FAILED,
        error_reason="DOP_HTTP_ERROR",
        error_message="ProxyError: Unable to connect to proxy\ntrace tail",
        set_finished=True,
    )

    detail = test_client.get(f"/jobs/{job_id}").json()

    assert detail["error_reason"] == "DOP_HTTP_ERROR"
    assert "ProxyError" in detail["error_message"]


def test_validate_bulk_status_gate_and_revision(client):
    test_client, app = client
    pending_id = test_client.post(
        "/jobs",
        json={"prompt": "building", "bbox_wgs84": _munich_bbox()},
    ).json()["id"]
    blocked = test_client.post(
        f"/jobs/{pending_id}/polygons/validate_bulk",
        json={"updates": [{"pid": 1, "validation": "REJECTED"}]},
    )
    assert blocked.status_code == 409

    job_id = _make_reviewable_job(test_client, app)
    ok = test_client.post(
        f"/jobs/{job_id}/polygons/validate_bulk",
        json={"updates": [{"pid": 1, "validation": "REJECTED"}]},
    )
    assert ok.status_code == 200
    assert test_client.get(f"/jobs/{job_id}").json()["validation_revision"] == 1


def test_geojson_routes_return_json_for_reviewable_job(client):
    test_client, app = client
    job_id = _make_reviewable_job(test_client, app)
    response = test_client.get(f"/jobs/{job_id}/polygons")
    assert response.status_code == 200
    assert response.json()["type"] == "FeatureCollection"
    nodata = test_client.get(f"/jobs/{job_id}/nodata")
    assert nodata.status_code == 200
    assert nodata.json()["features"] == []
