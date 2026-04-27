from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path

import pytest


def _pin_proj_data_dir() -> None:
    """Force PROJ + GDAL to use the bundled databases.

    Some Windows machines have a PostgreSQL/PostGIS install whose
    PROJ_LIB / GDAL_DATA env vars survive in the user shell and point at
    an older proj.db ("DATABASE.LAYOUT.VERSION.MINOR = 2 whereas a number
    >= 4 is expected"). We always override these to the venv-bundled
    pyproj / rasterio data directories so tests don't depend on shell
    state. Set KI_GEODATEN_RESPECT_PROJ=1 to opt out.
    """
    if os.environ.get("KI_GEODATEN_RESPECT_PROJ") == "1":
        return
    try:
        import pyproj
    except ImportError:
        pyproj = None  # type: ignore[assignment]
    if pyproj is not None:
        proj_dir = pyproj.datadir.get_data_dir()
        os.environ["PROJ_DATA"] = proj_dir
        os.environ["PROJ_LIB"] = proj_dir

    try:
        import rasterio
    except ImportError:
        return
    candidate = Path(rasterio.__file__).parent / "gdal_data"
    if candidate.is_dir():
        os.environ["GDAL_DATA"] = str(candidate)


_pin_proj_data_dir()


@pytest.fixture
def tmp_path():
    root = (Path(__file__).resolve().parents[1] / "data" / "test_tmp")
    root.mkdir(parents=True, exist_ok=True)
    path = root / uuid.uuid4().hex
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
