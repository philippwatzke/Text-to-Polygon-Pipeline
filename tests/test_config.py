# tests/test_config.py
from ki_geodaten.config import Settings

def test_settings_defaults():
    s = Settings()
    assert s.WMS_URL == "https://geoservices.bayern.de/od/wms/dop/v1/dop20"
    assert s.WMS_LAYER == "by_dop20c"
    assert s.WMS_VERSION == "1.1.1"
    assert s.WMS_FORMAT == "image/png"
    assert s.WMS_CRS == "EPSG:25832"
    assert s.WMS_MAX_PIXELS == 6000
    assert s.WMS_GRID_ORIGIN_X == 0.0
    assert s.WMS_GRID_ORIGIN_Y == 0.0
    assert s.SAM3_MODEL_ID == "facebook/sam3"
    assert s.TILE_SIZE == 1024
    assert s.DEFAULT_TILE_PRESET == "medium"
    assert s.MIN_POLYGON_AREA_M2 == 1.0
    assert s.LOCAL_MASK_NMS_IOU == 0.6
    assert s.LOCAL_MASK_CONTAINMENT_RATIO == 0.9
    assert s.SAFE_CENTER_NODATA_THRESHOLD == 0.0
    assert s.MAX_JOBS_PER_WORKER == 50
    assert s.WORKER_POLL_INTERVAL_SEC == 2.0
    assert s.MAX_BBOX_AREA_KM2 == 1.0
    assert s.MAX_PROMPT_CHARS == 240
    assert s.MAX_ENCODER_CONTEXT_TOKENS == 77
    assert s.MAX_CLIENT_BUFFER_UPDATES == 100
    assert s.BAYERN_BBOX_WGS84 == (8.9, 47.2, 13.9, 50.6)
    assert s.RETENTION_DAYS == 7

def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("MAX_BBOX_AREA_KM2", "2.5")
    s = Settings()
    assert s.MAX_BBOX_AREA_KM2 == 2.5
