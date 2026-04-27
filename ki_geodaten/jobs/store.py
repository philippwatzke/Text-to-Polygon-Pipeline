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
    missed_estimate   INTEGER,
    run_metadata      TEXT,
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
        cols = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(jobs)").fetchall()
        }
        if "missed_estimate" not in cols:
            conn.execute("ALTER TABLE jobs ADD COLUMN missed_estimate INTEGER")
        if "run_metadata" not in cols:
            conn.execute("ALTER TABLE jobs ADD COLUMN run_metadata TEXT")

def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def insert_job(
    db_path: Path, *, job_id: str, prompt: str,
    bbox_wgs84: list[float], bbox_utm_snapped: list[float],
    tile_preset: TilePreset,
    run_metadata: dict | None = None,
) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO jobs(id,prompt,bbox_wgs84,bbox_utm_snapped,tile_preset,"
            "status,run_metadata,created_at)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (
                job_id,
                prompt,
                json.dumps(bbox_wgs84),
                json.dumps(bbox_utm_snapped),
                str(tile_preset),
                JobStatus.PENDING,
                json.dumps(run_metadata) if run_metadata is not None else None,
                _utc_iso(),
            ),
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

def insert_polygons(db_path: Path, job_id: str, polys: list[dict]) -> None:
    if not polys:
        return
    with connect(db_path) as conn:
        conn.execute("BEGIN")
        conn.executemany(
            "INSERT INTO polygons(job_id,geometry_wkb,score,source_tile_row,source_tile_col)"
            " VALUES (?,?,?,?,?)",
            [(job_id, p["geometry_wkb"], p["score"],
              p["source_tile_row"], p["source_tile_col"]) for p in polys],
        )
        conn.execute("COMMIT")

def insert_nodata_region(
    db_path: Path, job_id: str, *, geometry_wkb: bytes,
    tile_row: int, tile_col: int, reason: str,
) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "INSERT INTO nodata_regions(job_id,geometry_wkb,tile_row,tile_col,reason)"
            " VALUES (?,?,?,?,?)",
            (job_id, geometry_wkb, tile_row, tile_col, reason),
        )

def increment_tile_completed(db_path: Path, job_id: str) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE jobs SET tile_completed = tile_completed + 1 WHERE id = ?",
            (job_id,),
        )

def increment_tile_failed(db_path: Path, job_id: str) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE jobs SET tile_failed = tile_failed + 1 WHERE id = ?",
            (job_id,),
        )

def get_polygons_for_job(db_path: Path, job_id: str) -> list[dict]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id,geometry_wkb,score,source_tile_row,source_tile_col,validation"
            " FROM polygons WHERE job_id = ?",
            (job_id,),
        ).fetchall()
    return [dict(r) for r in rows]

def get_nodata_for_job(db_path: Path, job_id: str) -> list[dict]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id,geometry_wkb,tile_row,tile_col,reason"
            " FROM nodata_regions WHERE job_id = ?",
            (job_id,),
        ).fetchall()
    return [dict(r) for r in rows]

def validate_bulk(db_path: Path, job_id: str, updates: list[dict]) -> int:
    """executemany-based bulk update per Spec §8. Increments validation_revision.

    Note: we rely on sqlite3.Cursor.rowcount being cumulative across an
    executemany() call — this is only guaranteed on Python 3.12+. The
    pyproject.toml requires-python setting enforces that; do NOT downgrade.
    """
    if not updates:
        return 0
    with connect(db_path) as conn:
        conn.execute("BEGIN")
        cur = conn.cursor()
        cur.executemany(
            "UPDATE polygons SET validation = ? WHERE id = ? AND job_id = ?",
            [(u["validation"], u["pid"], job_id) for u in updates],
        )
        updated = cur.rowcount
        conn.execute(
            "UPDATE jobs SET validation_revision = validation_revision + 1 WHERE id = ?",
            (job_id,),
        )
        conn.execute("COMMIT")
    return max(updated, 0)

def update_missed_estimate(db_path: Path, job_id: str, missed_estimate: int | None) -> None:
    with connect(db_path) as conn:
        conn.execute(
            "UPDATE jobs SET missed_estimate = ? WHERE id = ?",
            (missed_estimate, job_id),
        )

def get_job_summary(db_path: Path, job_id: str) -> dict | None:
    with connect(db_path) as conn:
        job = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if job is None:
            return None
        counts = conn.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN validation = 'ACCEPTED' THEN 1 ELSE 0 END) AS accepted,
                SUM(CASE WHEN validation = 'REJECTED' THEN 1 ELSE 0 END) AS rejected,
                MIN(score) AS min_score,
                AVG(score) AS avg_score,
                MAX(score) AS max_score
            FROM polygons
            WHERE job_id = ?
            """,
            (job_id,),
        ).fetchone()
        buckets = conn.execute(
            """
            SELECT
                SUM(CASE WHEN score < 0.35 THEN 1 ELSE 0 END) AS lt_035,
                SUM(CASE WHEN score >= 0.35 AND score < 0.5 THEN 1 ELSE 0 END) AS gte_035_lt_05,
                SUM(CASE WHEN score >= 0.5 AND score < 0.7 THEN 1 ELSE 0 END) AS gte_05_lt_07,
                SUM(CASE WHEN score >= 0.7 THEN 1 ELSE 0 END) AS gte_07
            FROM polygons
            WHERE job_id = ?
            """,
            (job_id,),
        ).fetchone()

    total = int(counts["total"] or 0)
    accepted = int(counts["accepted"] or 0)
    rejected = int(counts["rejected"] or 0)
    missed = job["missed_estimate"]
    recall_estimate = None
    if missed is not None and accepted + int(missed) > 0:
        recall_estimate = accepted / (accepted + int(missed))
    precision_review = accepted / total if total else None
    return {
        "id": job["id"],
        "prompt": job["prompt"],
        "tile_preset": job["tile_preset"],
        "status": job["status"],
        "tile_total": job["tile_total"],
        "tile_completed": job["tile_completed"],
        "tile_failed": job["tile_failed"],
        "validation_revision": job["validation_revision"],
        "exported_revision": job["exported_revision"],
        "export_stale": job["exported_revision"] is None
        or job["exported_revision"] < job["validation_revision"],
        "created_at": job["created_at"],
        "missed_estimate": missed,
        "total": total,
        "accepted": accepted,
        "rejected": rejected,
        "precision_review": precision_review,
        "recall_estimate": recall_estimate,
        "min_score": counts["min_score"],
        "avg_score": counts["avg_score"],
        "max_score": counts["max_score"],
        "score_buckets": {
            "lt_035": int(buckets["lt_035"] or 0),
            "gte_035_lt_05": int(buckets["gte_035_lt_05"] or 0),
            "gte_05_lt_07": int(buckets["gte_05_lt_07"] or 0),
            "gte_07": int(buckets["gte_07"] or 0),
        },
    }

def list_jobs(db_path: Path) -> list[dict]:
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]

def claim_next_pending_job(db_path: Path) -> dict | None:
    """Atomic PENDING→DOWNLOADING claim."""
    with connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM jobs WHERE status='PENDING' ORDER BY created_at ASC LIMIT 1"
        ).fetchone()
        if row is None:
            conn.execute("COMMIT")
            return None
        conn.execute(
            "UPDATE jobs SET status='DOWNLOADING', started_at=? WHERE id=?",
            (_utc_iso(), row["id"]),
        )
        conn.execute("COMMIT")
    return dict(row)

def abort_incomplete_jobs_on_startup(db_path: Path) -> list[str]:
    """Per Spec §10.3 — mark DOWNLOADING/INFERRING as FAILED(WORKER_RESTARTED)."""
    with connect(db_path) as conn:
        conn.execute("BEGIN")
        rows = conn.execute(
            "SELECT id FROM jobs WHERE status IN ('DOWNLOADING','INFERRING')"
        ).fetchall()
        ids = [r["id"] for r in rows]
        conn.execute(
            "UPDATE jobs SET status='FAILED', error_reason='WORKER_RESTARTED', finished_at=?"
            " WHERE status IN ('DOWNLOADING','INFERRING')",
            (_utc_iso(),),
        )
        conn.execute("COMMIT")
    return ids
