from ki_geodaten.jobs.store import get_job, init_schema, insert_job, update_status
from ki_geodaten.models import JobStatus, TilePreset
from ki_geodaten.worker.loop import run_forever, startup_cleanup


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
        wms_url="",
        layer="by_dop20c",
        max_pixels=6000,
        wms_version="1.1.1",
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
        wms_url="",
        layer="by_dop20c",
        max_pixels=6000,
        wms_version="1.1.1",
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
