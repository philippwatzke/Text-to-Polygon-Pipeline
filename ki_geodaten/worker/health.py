from __future__ import annotations

import logging
import os
import socket
import threading
from datetime import datetime, timezone
from pathlib import Path

from ki_geodaten.jobs.store import upsert_worker_heartbeat

logger = logging.getLogger(__name__)


class WorkerHealthReporter:
    def __init__(
        self,
        *,
        db_path: Path,
        worker_id: str,
        interval: float,
    ) -> None:
        self.db_path = db_path
        self.worker_id = worker_id
        self.interval = max(interval, 0.1)
        self.pid = os.getpid()
        self.hostname = socket.gethostname()
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.state = "starting"
        self.current_job_id: str | None = None
        self.processed_jobs = 0
        self.last_error: str | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run,
            name="worker-heartbeat",
            daemon=True,
        )

    def start(self) -> None:
        self.beat()
        self._thread.start()

    def stop(self) -> None:
        self.set_state("stopped", current_job_id=None)
        self._stop.set()
        self._thread.join(timeout=1.0)

    def set_state(
        self,
        state: str,
        *,
        current_job_id: str | None = None,
        last_error: str | None = None,
    ) -> None:
        with self._lock:
            self.state = state
            self.current_job_id = current_job_id
            if last_error is not None:
                self.last_error = last_error
        self.beat()

    def mark_processed(self) -> None:
        with self._lock:
            self.processed_jobs += 1
        self.beat()

    def beat(self) -> None:
        with self._lock:
            payload = {
                "state": self.state,
                "current_job_id": self.current_job_id,
                "processed_jobs": self.processed_jobs,
                "last_error": self.last_error,
            }
        try:
            upsert_worker_heartbeat(
                self.db_path,
                worker_id=self.worker_id,
                pid=self.pid,
                hostname=self.hostname,
                started_at=self.started_at,
                **payload,
            )
        except Exception:  # noqa: BLE001
            logger.exception("failed to write worker heartbeat")

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self.beat()
