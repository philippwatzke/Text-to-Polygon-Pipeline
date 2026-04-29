# tests/test_config.py
from pathlib import Path

from ki_geodaten.config import Settings

# Settings that pydantic-settings will pick up from a developer's local .env
# and that we need to silence here so the defaults test reflects the source
# code, not the local override.
_ENV_OVERRIDES_TO_CLEAR = (
    "MODALITY_USE_NDSM",
    "MODALITY_USE_NDVI",
    "FILL_WCS_RGB_ZERO_WITH_WMS",
    "SAM3_LOCAL_FILES_ONLY",
    "DOP_SOURCE",
)


def test_settings_defaults(monkeypatch, tmp_path):
    for name in _ENV_OVERRIDES_TO_CLEAR:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)  # no .env in this dir -> defaults from code
    s = Settings()
    # WCS — primary pipeline data path
    assert s.DOP_SOURCE == "wcs"
    assert s.WCS_URL == "https://geoservices.bayern.de/pro/wcs/dop/v1/wcs_inspire_dop20"
    assert s.WCS_VERSION == "2.0.1"
    assert s.WCS_COVERAGE_ID == "OI.OrthoimageCoverage"
    assert s.WCS_FORMAT == "image/tiff"
    assert s.WCS_CRS == "EPSG:25832"
    assert s.WCS_MAX_PIXELS == 6000
    # LDBV-Grid-Origin (0.1, 0.1), DescribeCoverage 2026-04-28 verified
    assert s.WCS_GRID_ORIGIN_X == 0.1
    assert s.WCS_GRID_ORIGIN_Y == 0.1
    assert s.FILL_WCS_RGB_ZERO_WITH_WMS is True
    # WMS — UI-only basemap
    assert s.WMS_URL == "https://geoservices.bayern.de/od/wms/dop/v1/dop20"
    assert s.WMS_LAYER == "by_dop20c"
    assert s.WMS_VERSION == "1.1.1"
    assert s.WMS_FORMAT == "image/png"
    assert s.WMS_CRS == "EPSG:25832"
    assert s.SAM3_MODEL_ID == "facebook/sam3"
    assert s.SAM3_LOCAL_FILES_ONLY is True
    assert s.TILE_SIZE == 1024
    assert s.DEFAULT_TILE_PRESET == "medium"
    assert s.MIN_POLYGON_AREA_M2 == 1.0
    assert s.LOCAL_MASK_NMS_IOU == 0.6
    assert s.LOCAL_MASK_CONTAINMENT_RATIO == 0.9
    assert s.GLOBAL_POLYGON_NMS_IOU == 0.5
    assert s.GLOBAL_POLYGON_CONTAINMENT_RATIO == 0.85
    assert s.GLOBAL_POLYGON_FRAGMENT_COVERAGE_RATIO == 0.65
    assert s.GLOBAL_POLYGON_FRAGMENT_MAX_AREA_RATIO == 0.75
    assert s.GLOBAL_POLYGON_FRAGMENT_BUFFER_M == 1.0
    assert s.SAM_IMAGE_PREPROCESS == "clahe"
    assert s.SAM_CLAHE_CLIP_LIMIT == 2.0
    assert s.SAM_CLAHE_TILE_GRID_SIZE == 8
    assert s.SAM_CLAHE_BLEND == 0.65
    assert s.SAM_IMAGE_GAMMA == 1.0
    assert s.SAM_IMAGE_BRIGHTNESS == 1.0
    assert s.SAFE_CENTER_NODATA_THRESHOLD == 0.0
    assert s.MAX_JOBS_PER_WORKER == 50
    assert s.WORKER_POLL_INTERVAL_SEC == 2.0
    assert s.MAX_BBOX_AREA_KM2 == 1.0
    assert s.MAX_PROMPT_CHARS == 240
    assert s.MAX_ENCODER_CONTEXT_TOKENS == 77
    assert s.MAX_CLIENT_BUFFER_UPDATES == 100
    assert s.MODALITY_USE_NDVI is True
    assert s.MODALITY_USE_NDSM is False
    assert s.DGM_METALINK_URL == "https://geoservices.bayern.de/services/poly2metalink/metalink/dgm1"
    assert s.DGM_TILE_CACHE_DIR == Path("data/opendata/dgm1")
    assert s.DGM_NATIVE_STEP_M == 1.0
    assert s.DOM_METALINK_URL == "https://geoservices.bayern.de/services/poly2metalink/metalink/dom20dom"
    assert s.DOM_TILE_CACHE_DIR == Path("data/opendata/dom20")
    assert s.DOM_NATIVE_STEP_M == 0.2
    assert s.BAYERN_BBOX_WGS84 == (8.9, 47.2, 13.9, 50.6)
    assert s.RETENTION_DAYS == 7

def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("MAX_BBOX_AREA_KM2", "2.5")
    s = Settings()
    assert s.MAX_BBOX_AREA_KM2 == 2.5
