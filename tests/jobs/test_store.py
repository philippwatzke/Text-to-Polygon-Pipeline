import pytest
import sqlite3
from pathlib import Path
from ki_geodaten.jobs.store import connect, init_schema, insert_job, get_job, update_status
from ki_geodaten.models import JobStatus, TilePreset

def test_init_schema_creates_tables(tmp_path: Path):
    db = tmp_path / "t.db"
    init_schema(db)
    with connect(db) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    names = [r[0] for r in rows]
    assert "jobs" in names
    assert "polygons" in names
    assert "nodata_regions" in names

def test_connect_sets_wal_mode(tmp_path):
    db = tmp_path / "t.db"
    init_schema(db)
    with connect(db) as conn:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
    assert mode.lower() == "wal"
    assert fk == 1

def test_insert_and_get_job(tmp_path):
    db = tmp_path / "t.db"
    init_schema(db)
    jid = "00000000-0000-0000-0000-000000000001"
    insert_job(
        db, job_id=jid, prompt="building",
        bbox_wgs84=[11.0, 48.0, 11.1, 48.1],
        bbox_utm_snapped=[691000.0, 5335000.0, 692000.0, 5336000.0],
        tile_preset=TilePreset.MEDIUM,
    )
    job = get_job(db, jid)
    assert job["status"] == JobStatus.PENDING
    assert job["prompt"] == "building"
    assert job["validation_revision"] == 0
    assert job["exported_revision"] is None

def test_status_check_constraint_rejects_unknown(tmp_path):
    db = tmp_path / "t.db"
    init_schema(db)
    with connect(db) as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO jobs(id,prompt,bbox_wgs84,bbox_utm_snapped,tile_preset,status,created_at)"
                " VALUES ('x','p','[]','[]','medium','WEIRD','2026-01-01')"
            )

def test_cascade_delete(tmp_path):
    db = tmp_path / "t.db"
    init_schema(db)
    jid = "j1"
    insert_job(db, job_id=jid, prompt="p", bbox_wgs84=[0,0,1,1],
               bbox_utm_snapped=[0,0,1,1], tile_preset=TilePreset.MEDIUM)
    with connect(db) as conn:
        conn.execute(
            "INSERT INTO polygons(job_id,geometry_wkb,score,source_tile_row,source_tile_col)"
            " VALUES (?,?,?,?,?)", (jid, b"\x00", 0.5, 0, 0))
        conn.commit()
    with connect(db) as conn:
        conn.execute("DELETE FROM jobs WHERE id=?", (jid,))
        conn.commit()
        n = conn.execute("SELECT COUNT(*) FROM polygons").fetchone()[0]
    assert n == 0
