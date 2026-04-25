# ki_geodaten/config.py
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # WMS (LDBV OpenData DOP20 — Task 0 verified)
    WMS_URL: str = "https://geoservices.bayern.de/od/wms/dop/v1/dop20"
    WMS_LAYER: str = "by_dop20c"
    WMS_VERSION: str = "1.1.1"
    WMS_FORMAT: str = "image/png"
    WMS_CRS: str = "EPSG:25832"
    WMS_MAX_PIXELS: int = 6000
    # WMS has no native server-side grid; we snap client-side to this origin.
    WMS_GRID_ORIGIN_X: float = 0.0
    WMS_GRID_ORIGIN_Y: float = 0.0

    # SAM 3.1
    SAM3_CHECKPOINT: Path = Path("models/sam3.1_hiera_large.pt")

    # Tiling
    TILE_SIZE: int = 1024
    DEFAULT_TILE_PRESET: str = "medium"

    # Filtering
    MIN_POLYGON_AREA_M2: float = 1.0
    LOCAL_MASK_NMS_IOU: float = 0.6
    LOCAL_MASK_CONTAINMENT_RATIO: float = 0.9
    SAFE_CENTER_NODATA_THRESHOLD: float = 0.0

    # Worker
    MAX_JOBS_PER_WORKER: int = 50
    WORKER_POLL_INTERVAL_SEC: float = 2.0

    # API limits
    MAX_BBOX_AREA_KM2: float = 1.0
    MAX_PROMPT_CHARS: int = 240
    MAX_ENCODER_CONTEXT_TOKENS: int = 77
    MAX_CLIENT_BUFFER_UPDATES: int = 100

    # Geographic Fence
    BAYERN_BBOX_WGS84: tuple[float, float, float, float] = (8.9, 47.2, 13.9, 50.6)

    # Retention
    RETENTION_DAYS: int = 7

    # Paths
    DATA_DIR: Path = Path("data")
    DOP_DIR: Path = Path("data/dop")
    RESULTS_DIR: Path = Path("data/results")
    DB_PATH: Path = Path("data/jobs.db")

settings = Settings()
