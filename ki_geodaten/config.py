# ki_geodaten/config.py
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # WCS — primary data acquisition path (Spec §5.1, post-2026-04-28 revision).
    # All pipeline input flows through WCS so we get raw uint8 RGB(A) coverage
    # without any server-side rendering pipeline. Authenticated via Basic Auth.
    DOP_SOURCE: str = "wcs"
    WCS_URL: str = "https://geoservices.bayern.de/pro/wcs/dop/v1/wcs_inspire_dop20"
    WCS_VERSION: str = "2.0.1"
    WCS_COVERAGE_ID: str = "OI.OrthoimageCoverage"  # verified via DescribeCoverage 2026-04-28
    WCS_FORMAT: str = "image/tiff"
    WCS_CRS: str = "EPSG:25832"
    WCS_MAX_PIXELS: int = 6000
    DOP_DOWNLOAD_WORKERS: int = 4
    WCS_USERNAME: str = ""
    WCS_PASSWORD: str = ""
    # DOP20 raster grid origin per LDBV DescribeCoverage (gml:origin =
    # 492346.1, 5618159.9 → both ≡ 0.1 mod 0.2). Pixel corners therefore land
    # on x ≡ 0.1 mod 0.2, y ≡ 0.1 mod 0.2, NOT the integer-metre grid.
    WCS_GRID_ORIGIN_X: float = 0.1
    WCS_GRID_ORIGIN_Y: float = 0.1
    FILL_WCS_RGB_ZERO_WITH_WMS: bool = True

    # WMS — UI-only basemap for the Leaflet review map. Not used for any
    # pipeline input. Kept here so the frontend can render a backdrop tile
    # layer without a separate config block.
    WMS_URL: str = "https://geoservices.bayern.de/od/wms/dop/v1/dop20"
    WMS_LAYER: str = "by_dop20c"
    WMS_VERSION: str = "1.1.1"
    WMS_FORMAT: str = "image/png"
    WMS_CRS: str = "EPSG:25832"

    # SAM 3.1. Use the Hugging Face model id by default; set this to a local
    # snapshot directory for offline runs.
    SAM3_MODEL_ID: str = "facebook/sam3"
    SAM3_CHECKPOINT: Path = Path("models/sam3.1_hiera_large.pt")
    SAM3_LOCAL_FILES_ONLY: bool = True

    # Tiling
    TILE_SIZE: int = 1024
    DEFAULT_TILE_PRESET: str = "medium"

    # Filtering
    MIN_POLYGON_AREA_M2: float = 1.0
    LOCAL_MASK_NMS_IOU: float = 0.6
    LOCAL_MASK_CONTAINMENT_RATIO: float = 0.9
    GLOBAL_POLYGON_NMS_IOU: float = 0.5
    GLOBAL_POLYGON_CONTAINMENT_RATIO: float = 0.85
    GLOBAL_POLYGON_FRAGMENT_COVERAGE_RATIO: float = 0.65
    GLOBAL_POLYGON_FRAGMENT_MAX_AREA_RATIO: float = 0.75
    GLOBAL_POLYGON_FRAGMENT_BUFFER_M: float = 1.0
    SAM_IMAGE_PREPROCESS: str = "clahe"
    SAM_CLAHE_CLIP_LIMIT: float = 2.0
    SAM_CLAHE_TILE_GRID_SIZE: int = 8
    SAM_CLAHE_BLEND: float = 0.65
    SAM_IMAGE_GAMMA: float = 1.0
    SAM_IMAGE_BRIGHTNESS: float = 1.0
    SAFE_CENTER_NODATA_THRESHOLD: float = 0.0

    # Worker
    MAX_JOBS_PER_WORKER: int = 50
    MAX_WORKER_RUNTIME_SECONDS: int = 24 * 3600  # Spec §10: wall-clock restart
    WORKER_POLL_INTERVAL_SEC: float = 2.0

    # API limits
    MAX_BBOX_AREA_KM2: float = 1.0
    MAX_PROMPT_CHARS: int = 240
    MAX_ENCODER_CONTEXT_TOKENS: int = 77
    MAX_CLIENT_BUFFER_UPDATES: int = 100

    # NDVI — derived from DOP20 band 4 (NIR). No extra service needed; the
    # WCS coverage `OI.OrthoimageCoverage` already ships R/G/B/IR per
    # DescribeCoverage. Tiler reads band 4 only when MODALITY_USE_NDVI is on.
    MODALITY_USE_NDVI: bool = True
    # SAM 3 sees the RGB; NDVI is computed on the raw uint8 (R, NIR) and used
    # only for post-segmentation polygon filtering.

    # nDSM — derived locally as DOM - DGM. Both inputs are selected through
    # Bayern's polygonal OpenData Metalink API and cached locally as GeoTIFF
    # tiles; no additional DGM WCS account is required.
    MODALITY_USE_NDSM: bool = False
    DGM_METALINK_URL: str = "https://geoservices.bayern.de/services/poly2metalink/metalink/dgm1"
    DGM_TILE_CACHE_DIR: Path = Path("data/opendata/dgm1")
    DGM_NATIVE_STEP_M: float = 1.0
    DOM_METALINK_URL: str = "https://geoservices.bayern.de/services/poly2metalink/metalink/dom20dom"
    DOM_TILE_CACHE_DIR: Path = Path("data/opendata/dom20")
    DOM_NATIVE_STEP_M: float = 0.2

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
