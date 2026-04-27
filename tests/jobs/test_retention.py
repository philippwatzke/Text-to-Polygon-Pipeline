from datetime import datetime, timedelta, timezone
from pathlib import Path

from ki_geodaten.jobs.retention import cleanup_old_jobs
from ki_geodaten.jobs.store import (
    connect,
    get_job,
    init_schema,
    insert_job,
    insert_polygons,
    update_status,
)
from ki_geodaten.models import JobStatus, TilePreset


def _set_finished_at(db_path: Path, job_id: str, when: datetime) -> None:
    with connect(db_path) as conn:
        conn.execute("UPDATE jobs SET finished_at = ? WHERE id = ?", (when.isoformat(), job_id))


def test_cleanup_deletes_old_failed_and_exported(tmp_path):
    db = tmp_path / "j.db"
    results = tmp_path / "results"
    results.mkdir()
    init_schema(db)
    for jid in ("old_failed", "old_exported", "recent_exported", "active"):
        insert_job(
            db,
            job_id=jid,
            prompt="p",
            bbox_wgs84=[0, 0, 1, 1],
            bbox_utm_snapped=[0, 0, 1, 1],
            tile_preset=TilePreset.MEDIUM,
        )
    update_status(db, "old_failed", JobStatus.FAILED, error_reason="DOP_TIMEOUT", set_finished=True)
    old_gpkg = results / "old_exported.gpkg"
    recent_gpkg = results / "recent_exported.gpkg"
    update_status(db, "old_exported", JobStatus.EXPORTED, set_finished=True, gpkg_path=str(old_gpkg))
    old_gpkg.write_bytes(b"gpkg")
    update_status(db, "recent_exported", JobStatus.EXPORTED, set_finished=True, gpkg_path=str(recent_gpkg))
    recent_gpkg.write_bytes(b"gpkg")

    now = datetime.now(timezone.utc)
    _set_finished_at(db, "old_failed", now - timedelta(days=10))
    _set_finished_at(db, "old_exported", now - timedelta(days=10))
    _set_finished_at(db, "recent_exported", now - timedelta(days=1))

    deleted = cleanup_old_jobs(db, results_dir=results, retention_days=7)
    assert set(deleted) == {"old_failed", "old_exported"}
    assert get_job(db, "old_failed") is None
    assert get_job(db, "old_exported") is None
    assert get_job(db, "recent_exported") is not None
    assert get_job(db, "active") is not None
    assert not old_gpkg.exists()
    assert recent_gpkg.exists()


def test_cleanup_cascades_to_polygons(tmp_path):
    db = tmp_path / "j.db"
    results = tmp_path / "results"
    results.mkdir()
    init_schema(db)
    insert_job(
        db,
        job_id="old",
        prompt="p",
        bbox_wgs84=[0, 0, 1, 1],
        bbox_utm_snapped=[0, 0, 1, 1],
        tile_preset=TilePreset.MEDIUM,
    )
    insert_polygons(
        db,
        "old",
        [{"geometry_wkb": b"a", "score": 0.9, "source_tile_row": 0, "source_tile_col": 0}],
    )
    update_status(db, "old", JobStatus.FAILED, error_reason="OOM", set_finished=True)
    _set_finished_at(db, "old", datetime.now(timezone.utc) - timedelta(days=10))
    cleanup_old_jobs(db, results_dir=results, retention_days=7)
    with connect(db) as conn:
        count = conn.execute("SELECT COUNT(*) FROM polygons WHERE job_id='old'").fetchone()[0]
    assert count == 0


def test_cleanup_batches_large_delete_set(tmp_path):
    db = tmp_path / "j.db"
    results = tmp_path / "results"
    results.mkdir()
    init_schema(db)
    old_when = datetime.now(timezone.utc) - timedelta(days=10)
    for i in range(1050):
        jid = f"old_{i}"
        insert_job(
            db,
            job_id=jid,
            prompt="p",
            bbox_wgs84=[0, 0, 1, 1],
            bbox_utm_snapped=[0, 0, 1, 1],
            tile_preset=TilePreset.MEDIUM,
        )
        update_status(db, jid, JobStatus.FAILED, error_reason="DOP_TIMEOUT", set_finished=True)
        _set_finished_at(db, jid, old_when)
    assert len(cleanup_old_jobs(db, results_dir=results, retention_days=7)) == 1050
