from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Request

from ki_geodaten.config import Settings
from ki_geodaten.jobs.store import get_latest_worker_heartbeat, get_queue_counts

router = APIRouter()


def _parse_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


@router.get("/system/health")
async def system_health(request: Request):
    settings: Settings = request.app.state.settings
    heartbeat = get_latest_worker_heartbeat(request.app.state.db_path)
    now = datetime.now(timezone.utc)

    worker = {
        "state": "offline",
        "heartbeat_state": None,
        "worker_id": None,
        "pid": None,
        "hostname": None,
        "started_at": None,
        "last_seen_at": None,
        "seconds_since_seen": None,
        "current_job_id": None,
        "processed_jobs": 0,
        "last_error": None,
    }
    if heartbeat is not None:
        last_seen = _parse_utc(heartbeat.get("last_seen_at"))
        seconds_since_seen = None
        state = "offline"
        if last_seen is not None:
            seconds_since_seen = max((now - last_seen).total_seconds(), 0.0)
            if heartbeat.get("state") != "stopped":
                state = (
                    "online"
                    if seconds_since_seen <= settings.WORKER_STALE_AFTER_SEC
                    else "stale"
                )
        worker.update(
            {
                "state": state,
                "heartbeat_state": heartbeat.get("state"),
                "worker_id": heartbeat.get("worker_id"),
                "pid": heartbeat.get("pid"),
                "hostname": heartbeat.get("hostname"),
                "started_at": heartbeat.get("started_at"),
                "last_seen_at": heartbeat.get("last_seen_at"),
                "seconds_since_seen": seconds_since_seen,
                "current_job_id": heartbeat.get("current_job_id"),
                "processed_jobs": heartbeat.get("processed_jobs") or 0,
                "last_error": heartbeat.get("last_error"),
            }
        )

    return {
        "worker": worker,
        "queue": get_queue_counts(request.app.state.db_path),
    }
