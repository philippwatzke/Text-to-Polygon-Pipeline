import numpy as np
import geopandas as gpd
from affine import Affine
from pathlib import Path
from shapely.geometry import Polygon
from shapely.wkb import loads as wkb_loads

from ki_geodaten.jobs.store import (
    get_job,
    get_nodata_for_job,
    get_polygons_for_job,
    init_schema,
    insert_job,
)
from ki_geodaten.models import JobStatus, TilePreset
from ki_geodaten.pipeline.segmenter import MaskResult, SegmenterUnavailableError
from ki_geodaten.pipeline.dem_client import TileFetchResult
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


def _setup_job(tmp_path: Path, *, vector_options=None) -> Path:
    db = tmp_path / "j.db"
    init_schema(db)
    insert_job(
        db,
        job_id="j1",
        prompt="building",
        bbox_wgs84=[11.0, 48.0, 11.01, 48.01],
        bbox_utm_snapped=[691000.0, 5335000.0, 692000.0, 5336000.0],
        tile_preset=TilePreset.MEDIUM,
        vector_options=vector_options,
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
        wcs_url="",
        coverage_id="by_dop20c",
        max_pixels=6000,
        wcs_version="2.0.1",
        fmt="image/png",
        crs="EPSG:25832",
        origin_x=0.0,
        origin_y=0.0,
        min_polygon_area_m2=0.01,
        safe_center_nodata_threshold=0.0,
    )


def test_orchestrator_applies_ndvi_filter_when_active(tmp_path, monkeypatch):
    """When the job has an active NDVI threshold, only masks whose NIR/Red
    aggregate passes survive into the polygon table."""
    db = tmp_path / "j.db"
    init_schema(db)
    insert_job(
        db,
        job_id="ndvi_job",
        prompt="tree",
        bbox_wgs84=[11.0, 48.0, 11.01, 48.01],
        bbox_utm_snapped=[691000.0, 5335000.0, 692000.0, 5336000.0],
        tile_preset=TilePreset.MEDIUM,
        modality_filter={"ndvi_min": 0.3},
    )

    cfg = TileConfig.from_preset(TilePreset.MEDIUM)
    affine = Affine(0.2, 0, 691000.0, 0, -0.2, 5336204.8)
    rgb = np.zeros((cfg.size, cfg.size, 3), dtype=np.uint8)
    rgb[..., 0] = 30  # uniform red
    nir = np.full((cfg.size, cfg.size), 30, dtype=np.uint8)  # NDVI = 0 globally
    nir[500:524, 500:524] = 200  # vegetation patch -> NDVI ~0.74

    tile = Tile(
        array=rgb,
        pixel_origin=(0, 0),
        size=cfg.size,
        center_margin=cfg.center_margin,
        affine=affine,
        tile_row=0,
        tile_col=0,
        nodata_mask=np.zeros((cfg.size, cfg.size), dtype=bool),
        nir=nir,
    )

    veg_mask = np.zeros((cfg.size, cfg.size), dtype=bool)
    veg_mask[500:524, 500:524] = True
    vegetation = MaskResult(mask=veg_mask, score=0.9, box_pixel=(500, 500, 524, 524))

    nonveg_mask = np.zeros((cfg.size, cfg.size), dtype=bool)
    nonveg_mask[400:420, 400:420] = True
    non_vegetation = MaskResult(mask=nonveg_mask, score=0.9, box_pixel=(400, 400, 420, 420))

    _patch_io(monkeypatch, [tile])
    run_job(
        db,
        job_id="ndvi_job",
        segmenter=StubSegmenter([[vegetation, non_vegetation]]),
        data_root=tmp_path,
        wcs_url="",
        coverage_id="by_dop20c",
        max_pixels=6000,
        wcs_version="2.0.1",
        fmt="image/tiff",
        crs="EPSG:25832",
        origin_x=0.0,
        origin_y=0.0,
        min_polygon_area_m2=0.01,
        safe_center_nodata_threshold=0.0,
    )

    polys = get_polygons_for_job(db, "ndvi_job")
    assert len(polys) == 1  # only the vegetation polygon survived


def test_orchestrator_derives_ndsm_from_dom_and_dgm_when_requested(tmp_path, monkeypatch):
    db = tmp_path / "j.db"
    init_schema(db)
    insert_job(
        db,
        job_id="ndsm_job",
        prompt="building",
        bbox_wgs84=[11.0, 48.0, 11.01, 48.01],
        bbox_utm_snapped=[691000.0, 5335000.0, 692000.0, 5336000.0],
        tile_preset=TilePreset.MEDIUM,
        modality_filter={"ndsm_min": 2.0},
    )

    calls = []
    captured = {}

    def fake_fetch_tiles_via_metalink(
        *,
        bbox_utm,
        out_dir,
        metalink_url,
        cache_dir=None,
        vrt_name="tiles.vrt",
    ):
        calls.append((vrt_name, metalink_url, cache_dir))
        vrt_path = out_dir / vrt_name
        vrt_path.parent.mkdir(parents=True, exist_ok=True)
        vrt_path.write_bytes(vrt_name.encode("ascii"))
        return TileFetchResult(path=vrt_path, tile_paths=(vrt_path,), bbox_utm=bbox_utm)

    def fake_derive(*, dom_path, dgm_path, out_path):
        captured["derive"] = (dom_path.name, dgm_path.name, out_path.name)
        out_path.write_bytes(b"ndsm")

    def fake_align(source_path, reference_path, out_path):
        captured["align"] = (source_path.name, reference_path.name, out_path.name)
        return out_path

    def fake_iter_tiles(vrt_path, cfg, **kwargs):
        captured["ndsm_path"] = kwargs.get("ndsm_path")
        return iter([_tile()])

    mask = np.zeros((1024, 1024), dtype=bool)
    mask[500:524, 500:524] = True
    result = MaskResult(mask=mask, score=0.9, box_pixel=(500, 500, 524, 524))
    monkeypatch.setattr("ki_geodaten.worker.orchestrator.download_dop20", lambda *args, **kwargs: Path("fake.vrt"))
    monkeypatch.setattr("ki_geodaten.worker.orchestrator.rasterio.open", lambda *args, **kwargs: FakeSrc())
    monkeypatch.setattr("ki_geodaten.worker.orchestrator.fetch_tiles_via_metalink", fake_fetch_tiles_via_metalink)
    monkeypatch.setattr("ki_geodaten.worker.orchestrator.derive_ndsm_from_dom_dgm", fake_derive)
    monkeypatch.setattr("ki_geodaten.worker.orchestrator.align_raster_to_reference_grid", fake_align)
    monkeypatch.setattr("ki_geodaten.worker.orchestrator.iter_tiles", fake_iter_tiles)

    run_job(
        db,
        job_id="ndsm_job",
        segmenter=StubSegmenter([[result]]),
        data_root=tmp_path,
        wcs_url="",
        coverage_id="by_dop20c",
        max_pixels=6000,
        wcs_version="2.0.1",
        fmt="image/tiff",
        crs="EPSG:25832",
        origin_x=0.0,
        origin_y=0.0,
        min_polygon_area_m2=0.01,
        safe_center_nodata_threshold=0.0,
        dom_metalink_url="http://example/metalink/dom20dom",
        dom_tile_cache_dir=tmp_path / "dom-cache",
        dgm_metalink_url="http://example/metalink/dgm1",
        dgm_tile_cache_dir=tmp_path / "dgm-cache",
    )

    assert calls == [
        ("dom.vrt", "http://example/metalink/dom20dom", tmp_path / "dom-cache"),
        ("dgm.vrt", "http://example/metalink/dgm1", tmp_path / "dgm-cache"),
    ]
    assert captured["derive"] == ("dom.vrt", "dgm.vrt", "ndsm.tif")
    assert captured["align"] == ("ndsm.tif", "fake.vrt", "ndsm_aligned_to_dop.tif")
    assert captured["ndsm_path"].name == "ndsm_aligned_to_dop.tif"


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


def test_orchestrator_applies_vector_simplification_before_persist(tmp_path, monkeypatch):
    db = _setup_job(
        tmp_path,
        vector_options={"simplification_tolerance_m": 0.1, "orthogonalize": False},
    )
    noisy = Polygon(
        [
            (0, 0),
            (1, 0.02),
            (2, 0),
            (3, 0.01),
            (4, 0),
            (4, 2),
            (0, 2),
            (0, 0),
        ]
    )
    gdf = gpd.GeoDataFrame(
        [{
            "score": 0.9,
            "source_tile_row": 0,
            "source_tile_col": 0,
            "ndvi_mean": None,
            "ndsm_mean": None,
        }],
        geometry=[noisy],
        crs="EPSG:25832",
    )
    mask = np.zeros((1024, 1024), dtype=bool)
    mask[500:524, 500:524] = True
    result = MaskResult(mask=mask, score=0.9, box_pixel=(500, 500, 524, 524))
    _patch_io(monkeypatch, [_tile()])
    monkeypatch.setattr("ki_geodaten.worker.orchestrator.masks_to_polygons", lambda *args, **kwargs: gdf)

    _run(db, tmp_path, StubSegmenter([[result]]))

    geom = wkb_loads(get_polygons_for_job(db, "j1")[0]["geometry_wkb"])
    assert len(geom.exterior.coords) < len(noisy.exterior.coords)


def test_orchestrator_applies_vector_orthogonalization_before_persist(tmp_path, monkeypatch):
    db = _setup_job(
        tmp_path,
        vector_options={"simplification_tolerance_m": None, "orthogonalize": True},
    )
    near_rect = Polygon([(0, 0), (4, 0), (4, 2), (2, 2), (2, 1.9), (0, 2), (0, 0)])
    gdf = gpd.GeoDataFrame(
        [{
            "score": 0.9,
            "source_tile_row": 0,
            "source_tile_col": 0,
            "ndvi_mean": None,
            "ndsm_mean": None,
        }],
        geometry=[near_rect],
        crs="EPSG:25832",
    )
    mask = np.zeros((1024, 1024), dtype=bool)
    mask[500:524, 500:524] = True
    result = MaskResult(mask=mask, score=0.9, box_pixel=(500, 500, 524, 524))
    _patch_io(monkeypatch, [_tile()])
    monkeypatch.setattr("ki_geodaten.worker.orchestrator.masks_to_polygons", lambda *args, **kwargs: gdf)

    _run(db, tmp_path, StubSegmenter([[result]]))

    geom = wkb_loads(get_polygons_for_job(db, "j1")[0]["geometry_wkb"])
    assert geom.area == 8
    assert len(geom.exterior.coords) == 5


def test_orchestrator_applies_global_polygon_nms_across_tiles(tmp_path, monkeypatch):
    db = _setup_job(tmp_path)
    low_mask = np.zeros((1024, 1024), dtype=bool)
    low_mask[500:524, 500:524] = True
    high_mask = np.zeros((1024, 1024), dtype=bool)
    high_mask[500:524, 500:524] = True
    low = MaskResult(mask=low_mask, score=0.4, box_pixel=(500, 500, 524, 524))
    high = MaskResult(mask=high_mask, score=0.9, box_pixel=(500, 500, 524, 524))
    _patch_io(monkeypatch, [_tile(row=0), _tile(row=1)])

    _run(db, tmp_path, StubSegmenter([[low], [high]]))

    polys = get_polygons_for_job(db, "j1")
    assert len(polys) == 1
    assert polys[0]["score"] == 0.9


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
