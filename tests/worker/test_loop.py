from ki_geodaten.jobs.store import get_job, init_schema, insert_job, update_status
from ki_geodaten.models import JobStatus, TilePreset
import ki_geodaten.worker.loop as worker_loop
from ki_geodaten.worker.loop import acquire_worker_lock, run_forever, startup_cleanup


def _insert(db, jid):
    insert_job(
        db,
        job_id=jid,
        prompt="p",
        bbox_wgs84=[0, 0, 1, 1],
        bbox_utm_snapped=[0, 0, 1, 1],
        tile_preset=TilePreset.MEDIUM,
    )


def test_startup_marks_incomplete_jobs_failed(tmp_path):
    db = tmp_path / "j.db"
    init_schema(db)
    _insert(db, "j1")
    update_status(db, "j1", JobStatus.DOWNLOADING, set_started=True)
    startup_cleanup(db, data_root=tmp_path)
    job = get_job(db, "j1")
    assert job["status"] == JobStatus.FAILED
    assert job["error_reason"] == "WORKER_RESTARTED"


def test_startup_rmtree_removes_orphan_dirs(tmp_path):
    db = tmp_path / "j.db"
    init_schema(db)
    orphan = tmp_path / "dop" / "ghost-job"
    orphan.mkdir(parents=True)
    (orphan / "chunk_0_0.tif").write_bytes(b"x")
    startup_cleanup(db, data_root=tmp_path)
    assert not orphan.exists()


def test_worker_lock_prevents_second_worker(tmp_path):
    lock_path = tmp_path / "worker.lock"
    first = acquire_worker_lock(lock_path)
    assert first is not None
    try:
        assert acquire_worker_lock(lock_path) is None
    finally:
        first.release()
    assert not lock_path.exists()


def test_worker_lock_recovers_stale_dead_pid(tmp_path, monkeypatch):
    lock_path = tmp_path / "worker.lock"
    lock_path.write_text("123456", encoding="ascii")
    monkeypatch.setattr(worker_loop, "_pid_exists", lambda pid: False)

    lock = acquire_worker_lock(lock_path)

    assert lock is not None
    try:
        assert int(lock_path.read_text(encoding="ascii")) > 0
    finally:
        lock.release()


def test_run_forever_exits_after_wall_clock_budget(tmp_path, monkeypatch):
    db = tmp_path / "j.db"
    init_schema(db)
    _insert(db, "j1")
    _insert(db, "j2")
    calls: list[str] = []

    def fake_run_job(*args, **kwargs):
        calls.append(kwargs["job_id"])
        update_status(db, kwargs["job_id"], JobStatus.READY_FOR_REVIEW, set_finished=True)

    monkeypatch.setattr("ki_geodaten.worker.loop.run_job", fake_run_job)

    ticks = iter([0.0, 0.0, 5.0, 5.0])

    run_forever(
        db_path=db,
        data_root=tmp_path,
        segmenter_factory=lambda: object(),
        wcs_url="",
        coverage_id="by_dop20c",
        max_pixels=6000,
        wcs_version="2.0.1",
        fmt="image/png",
        crs="EPSG:25832",
        origin_x=0.0,
        origin_y=0.0,
        min_polygon_area_m2=1.0,
        safe_center_nodata_threshold=0.0,
        max_jobs=10,
        poll_interval=0.0,
        idle_exit_after=1,
        max_runtime_seconds=2.0,
        clock=lambda: next(ticks),
    )

    # First job claimed, processed; on the next loop iteration the clock has
    # advanced past the deadline, so j2 must remain PENDING.
    assert calls == ["j1"]
    j2 = get_job(db, "j2")
    assert j2["status"] == "PENDING"


def test_run_forever_exits_after_max_jobs(tmp_path, monkeypatch):
    db = tmp_path / "j.db"
    init_schema(db)
    _insert(db, "j1")
    _insert(db, "j2")
    calls: list[str] = []

    def fake_run_job(*args, **kwargs):
        calls.append(kwargs["job_id"])
        update_status(db, kwargs["job_id"], JobStatus.READY_FOR_REVIEW, set_finished=True)

    monkeypatch.setattr("ki_geodaten.worker.loop.run_job", fake_run_job)

    class StubSegmenter:
        pass

    run_forever(
        db_path=db,
        data_root=tmp_path,
        segmenter_factory=StubSegmenter,
        wcs_url="",
        coverage_id="by_dop20c",
        max_pixels=6000,
        wcs_version="2.0.1",
        fmt="image/png",
        crs="EPSG:25832",
        origin_x=0.0,
        origin_y=0.0,
        min_polygon_area_m2=1.0,
        safe_center_nodata_threshold=0.0,
        max_jobs=2,
        poll_interval=0.01,
        idle_exit_after=2,
    )
    assert calls == ["j1", "j2"]
