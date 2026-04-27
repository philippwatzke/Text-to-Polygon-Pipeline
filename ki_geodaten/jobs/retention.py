from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ki_geodaten.jobs.store import connect

logger = logging.getLogger(__name__)
DELETE_BATCH_SIZE = 900


def cleanup_old_jobs(
    db_path: Path,
    *,
    results_dir: Path,
    retention_days: int,
) -> list[str]:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat()
    with connect(db_path) as conn:
        conn.execute("BEGIN")
        rows = conn.execute(
            "SELECT id, gpkg_path FROM jobs"
            " WHERE status IN ('FAILED','EXPORTED')"
            " AND finished_at IS NOT NULL AND finished_at < ?",
            (cutoff,),
        ).fetchall()
        ids = [row["id"] for row in rows]
        gpkg_paths = [row["gpkg_path"] for row in rows if row["gpkg_path"]]
        for idx in range(0, len(ids), DELETE_BATCH_SIZE):
            batch = ids[idx : idx + DELETE_BATCH_SIZE]
            placeholders = ",".join("?" * len(batch))
            conn.execute(f"DELETE FROM jobs WHERE id IN ({placeholders})", batch)
        conn.execute("COMMIT")
        conn.execute("VACUUM")

    for path_str in gpkg_paths:
        path = Path(path_str)
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning("retention: failed to unlink %s", path)

    return ids
