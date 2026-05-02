from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

from ki_geodaten.app.main import create_app
from ki_geodaten.config import Settings
from ki_geodaten.jobs.store import connect, insert_job, update_status, upsert_worker_heartbeat
from ki_geodaten.models import JobStatus, TilePreset


def _client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = create_app(
        settings=Settings(WORKER_STALE_AFTER_SEC=1.0),
        executor_factory=lambda: ThreadPoolExecutor(max_workers=1),
        token_counter=lambda value: len(value.split()),
    )
    return app


def test_system_health_reports_offline_without_heartbeat(tmp_path, monkeypatch):
    app = _client(tmp_path, monkeypatch)
    with TestClient(app) as client:
        response = client.get("/system/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["worker"]["state"] == "offline"
    assert payload["queue"]["pending"] == 0


def test_system_health_reports_online_worker_and_queue_counts(tmp_path, monkeypatch):
    app = _client(tmp_path, monkeypatch)
    with TestClient(app) as client:
        insert_job(
            app.state.db_path,
            job_id="j1",
            prompt="building",
            bbox_wgs84=[0, 0, 1, 1],
            bbox_utm_snapped=[0, 0, 1, 1],
            tile_preset=TilePreset.MEDIUM,
        )
        update_status(app.state.db_path, "j1", JobStatus.INFERRING)
        upsert_worker_heartbeat(
            app.state.db_path,
            worker_id="w1",
            pid=123,
            hostname="host",
            started_at="2026-05-02T00:00:00+00:00",
            state="running",
            current_job_id="j1",
            processed_jobs=2,
        )

        payload = client.get("/system/health").json()

    assert payload["worker"]["state"] == "online"
    assert payload["worker"]["heartbeat_state"] == "running"
    assert payload["worker"]["current_job_id"] == "j1"
    assert payload["queue"]["running"] == 1


def test_system_health_reports_stale_worker(tmp_path, monkeypatch):
    app = _client(tmp_path, monkeypatch)
    with TestClient(app) as client:
        upsert_worker_heartbeat(
            app.state.db_path,
            worker_id="w1",
            pid=123,
            hostname="host",
            started_at="2026-05-02T00:00:00+00:00",
            state="idle",
        )
        with connect(app.state.db_path) as conn:
            conn.execute(
                "UPDATE worker_heartbeats SET last_seen_at = ? WHERE worker_id = ?",
                ("2026-05-02T00:00:00+00:00", "w1"),
            )

        payload = client.get("/system/health").json()

    assert payload["worker"]["state"] == "stale"
    assert payload["worker"]["seconds_since_seen"] > 1
