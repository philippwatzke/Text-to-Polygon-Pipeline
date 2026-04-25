from __future__ import annotations
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from ki_geodaten.models import JobStatus, TilePreset

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id                TEXT PRIMARY KEY,
    prompt            TEXT NOT NULL,
    bbox_wgs84        TEXT NOT NULL,
    bbox_utm_snapped  TEXT NOT NULL,
    tile_preset       TEXT NOT NULL CHECK (tile_preset IN ('small','medium','large')),
    status            TEXT NOT NULL CHECK (status IN (
                          'PENDING','DOWNLOADING','INFERRING',
                          'READY_FOR_REVIEW','EXPORTED','FAILED')),
    error_reason      TEXT CHECK (error_reason IS NULL OR error_reason IN (
                          'DOP_TIMEOUT','DOP_HTTP_ERROR','OOM',
                          'INFERENCE_ERROR','WORKER_RESTARTED',
                          'EXPORT_ERROR','INVALID_GEOMETRY')),
    error_message     TEXT,
    dop_vrt_path      TEXT,
    gpkg_path         TEXT,
    tile_total        INTEGER,
    tile_completed    INTEGER NOT NULL DEFAULT 0,
    tile_failed       INTEGER NOT NULL DEFAULT 0,
    validation_revision INTEGER NOT NULL DEFAULT 0,
    exported_revision   INTEGER,
    created_at        TEXT NOT NULL,
    started_at        TEXT,
    finished_at       TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON jobs(status, created_at);

CREATE TABLE IF NOT EXISTS polygons (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id           TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    geometry_wkb     BLOB NOT NULL,
    score            REAL NOT NULL,
    source_tile_row  INTEGER NOT NULL,
    source_tile_col  INTEGER NOT NULL,
    validation       TEXT NOT NULL DEFAULT 'ACCEPTED' CHECK (validation IN ('ACCEPTED','REJECTED'))
);
CREATE INDEX IF NOT EXISTS idx_polygons_job ON polygons(job_id);

CREATE TABLE IF NOT EXISTS nodata_regions (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id           TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    geometry_wkb     BLOB NOT NULL,
    tile_row         INTEGER NOT NULL,
    tile_col         INTEGER NOT NULL,
    reason           TEXT NOT NULL CHECK (reason IN (
                          'OOM','INFERENCE_ERROR','INVALID_GEOMETRY','NODATA_PIXELS'))
);
CREATE INDEX IF NOT EXISTS idx_nodata_job ON nodata_regions(job_id);
"""

def _apply_pragmas(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA foreign_keys = ON")

@contextmanager
def connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(str(db_path), isolation_level=None, timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        _apply_pragmas(conn)
        yield conn
    finally:
        conn.close()

def init_schema(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)

def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def insert_job(
    db_path: Path, *, job_id: str, prompt: str,
    bbox_wgs84: list[float], bbox_utm_snapped: list[float],
    tile_preset: TilePreset,
) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO jobs(id,prompt,bbox_wgs84,bbox_utm_snapped,tile_preset,status,created_at)"
            " VALUES (?,?,?,?,?,?,?)",
            (job_id, prompt, json.dumps(bbox_wgs84), json.dumps(bbox_utm_snapped),
             str(tile_preset), JobStatus.PENDING, _utc_iso()),
        )

def get_job(db_path: Path, job_id: str) -> dict | None:
    with connect(db_path) as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return dict(row) if row else None

def update_status(
    db_path: Path, job_id: str, status: JobStatus,
    *, error_reason: str | None = None, error_message: str | None = None,
    dop_vrt_path: str | None = None, gpkg_path: str | None = None,
    tile_total: int | None = None, exported_revision: int | None = None,
    set_started: bool = False, set_finished: bool = False,
) -> None:
    fields = ["status = ?"]
    params: list = [str(status)]
    if error_reason is not None:
        fields.append("error_reason = ?"); params.append(error_reason)
    if error_message is not None:
        fields.append("error_message = ?"); params.append(error_message)
    if dop_vrt_path is not None:
        fields.append("dop_vrt_path = ?"); params.append(dop_vrt_path)
    if gpkg_path is not None:
        fields.append("gpkg_path = ?"); params.append(gpkg_path)
    if tile_total is not None:
        fields.append("tile_total = ?"); params.append(tile_total)
    if exported_revision is not None:
        fields.append("exported_revision = ?"); params.append(exported_revision)
    if set_started:
        fields.append("started_at = ?"); params.append(_utc_iso())
    if set_finished:
        fields.append("finished_at = ?"); params.append(_utc_iso())
    params.append(job_id)
    with connect(db_path) as conn:
        conn.execute(f"UPDATE jobs SET {', '.join(fields)} WHERE id = ?", params)
