import numpy as np
from affine import Affine
from pathlib import Path

from ki_geodaten.jobs.store import (
    get_job,
    get_nodata_for_job,
    get_polygons_for_job,
    init_schema,
    insert_job,
)
from ki_geodaten.models import JobStatus, TilePreset
from ki_geodaten.pipeline.segmenter import MaskResult, SegmenterUnavailableError
from ki_geodaten.pipeline.tiler import NodataTile, Tile, TileConfig
from ki_geodaten.worker.orchestrator import run_job


class StubSegmenter:
    def __init__(self, behaviours):
        self.behaviours = list(behaviours)

    def predict(self, tile, prompt):
        behaviour = self.behaviours.pop(0)
        if isinstance(behaviour, Exception):
            raise behaviour
        return behaviour


class FakeSrc:
    width = 1024
    height = 1024

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None


def _setup_job(tmp_path: Path) -> Path:
    db = tmp_path / "j.db"
    init_schema(db)
    insert_job(
        db,
        job_id="j1",
        prompt="building",
        bbox_wgs84=[11.0, 48.0, 11.01, 48.01],
        bbox_utm_snapped=[691000.0, 5335000.0, 692000.0, 5336000.0],
        tile_preset=TilePreset.MEDIUM,
    )
    return db


def _tile(row=0, col=0):
    cfg = TileConfig.from_preset(TilePreset.MEDIUM)
    affine = Affine(0.2, 0, 691000.0, 0, -0.2, 5336204.8)
    return Tile(
        array=np.zeros((cfg.size, cfg.size, 3), dtype=np.uint8),
        pixel_origin=(0, 0),
        size=cfg.size,
        center_margin=cfg.center_margin,
        affine=affine,
        tile_row=row,
        tile_col=col,
        nodata_mask=np.zeros((cfg.size, cfg.size), dtype=bool),
    )


def _nodata_tile(row=0, col=0):
    cfg = TileConfig.from_preset(TilePreset.MEDIUM)
    affine = Affine(0.2, 0, 691000.0, 0, -0.2, 5336204.8)
    return NodataTile(
        tile_row=row,
        tile_col=col,
        pixel_origin=(0, 0),
        size=cfg.size,
        center_margin=cfg.center_margin,
        affine=affine,
    )


def _patch_io(monkeypatch, tiles):
    monkeypatch.setattr(
        "ki_geodaten.worker.orchestrator.download_dop20",
        lambda *args, **kwargs: Path("fake.vrt"),
    )
    monkeypatch.setattr("ki_geodaten.worker.orchestrator.rasterio.open", lambda *args, **kwargs: FakeSrc())
    monkeypatch.setattr("ki_geodaten.worker.orchestrator.iter_tiles", lambda *args, **kwargs: iter(tiles))


def _run(db, tmp_path, segmenter):
    run_job(
        db,
        job_id="j1",
        segmenter=segmenter,
        data_root=tmp_path,
        wms_url="",
        layer="by_dop20c",
        max_pixels=6000,
        wms_version="1.1.1",
        fmt="image/png",
        crs="EPSG:25832",
        origin_x=0.0,
        origin_y=0.0,
        min_polygon_area_m2=0.01,
        safe_center_nodata_threshold=0.0,
    )


def test_orchestrator_happy_path_marks_ready(tmp_path, monkeypatch):
    db = _setup_job(tmp_path)
    mask = np.zeros((1024, 1024), dtype=bool)
    mask[500:524, 500:524] = True
    result = MaskResult(mask=mask, score=0.9, box_pixel=(500, 500, 524, 524))
    _patch_io(monkeypatch, [_tile()])

    _run(db, tmp_path, StubSegmenter([[result]]))

    job = get_job(db, "j1")
    assert job["status"] == JobStatus.READY_FOR_REVIEW
    assert job["tile_total"] == 1
    assert job["tile_completed"] == 1
    assert len(get_polygons_for_job(db, "j1")) == 1


def test_orchestrator_records_inference_error_as_nodata(tmp_path, monkeypatch):
    db = _setup_job(tmp_path)
    _patch_io(monkeypatch, [_tile()])

    _run(db, tmp_path, StubSegmenter([ValueError("boom")]))

    nodata = get_nodata_for_job(db, "j1")
    assert nodata[0]["reason"] == "INFERENCE_ERROR"
    assert get_job(db, "j1")["status"] == JobStatus.READY_FOR_REVIEW


def test_orchestrator_fails_job_for_unavailable_segmenter(tmp_path, monkeypatch):
    db = _setup_job(tmp_path)
    _patch_io(monkeypatch, [_tile()])

    _run(db, tmp_path, StubSegmenter([SegmenterUnavailableError("sam3 unavailable")]))

    job = get_job(db, "j1")
    assert job["status"] == JobStatus.FAILED
    assert job["error_reason"] == "INFERENCE_ERROR"
    assert "sam3 unavailable" in job["error_message"]
    assert get_nodata_for_job(db, "j1") == []


def test_orchestrator_fails_job_for_placeholder_segmenter(tmp_path, monkeypatch):
    db = _setup_job(tmp_path)
    _patch_io(monkeypatch, [_tile()])

    _run(db, tmp_path, StubSegmenter([NotImplementedError("Task 12A")]))

    job = get_job(db, "j1")
    assert job["status"] == JobStatus.FAILED
    assert job["error_reason"] == "INFERENCE_ERROR"
    assert "Task 12A" in job["error_message"]


def test_orchestrator_records_nodata_tile_without_invoking_segmenter(tmp_path, monkeypatch):
    db = _setup_job(tmp_path)
    _patch_io(monkeypatch, [_nodata_tile()])

    _run(db, tmp_path, StubSegmenter([]))

    nodata = get_nodata_for_job(db, "j1")
    assert nodata[0]["reason"] == "NODATA_PIXELS"
    assert get_job(db, "j1")["tile_failed"] == 1


def test_orchestrator_download_error_marks_failed(tmp_path, monkeypatch):
    from ki_geodaten.pipeline.dop_client import DopDownloadError

    db = _setup_job(tmp_path)

    def fail_download(*args, **kwargs):
        raise DopDownloadError("DOP_TIMEOUT")

    monkeypatch.setattr("ki_geodaten.worker.orchestrator.download_dop20", fail_download)

    _run(db, tmp_path, StubSegmenter([]))

    job = get_job(db, "j1")
    assert job["status"] == JobStatus.FAILED
    assert job["error_reason"] == "DOP_TIMEOUT"
