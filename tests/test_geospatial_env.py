from __future__ import annotations

import os
from pathlib import Path

from ki_geodaten.geospatial_env import configure_geospatial_runtime


def test_configure_geospatial_runtime_overrides_stale_postgresql_proj(monkeypatch):
    stale_proj = r"C:\Program Files\PostgreSQL\16\share\contrib\postgis-3.4\proj"
    monkeypatch.setenv("PROJ_LIB", stale_proj)
    monkeypatch.setenv("PROJ_DATA", stale_proj)
    monkeypatch.setenv("GDAL_DATA", r"C:\Program Files\PostgreSQL\16\gdal-data")

    configured = configure_geospatial_runtime()

    assert os.environ["PROJ_LIB"] != stale_proj
    assert os.environ["PROJ_DATA"] != stale_proj
    assert Path(os.environ["PROJ_LIB"], "proj.db").is_file()
    assert os.environ["PROJ_LIB"] == os.environ["PROJ_DATA"]
    assert configured["PROJ_LIB"] == os.environ["PROJ_LIB"]
    assert configured["GDAL_DATA"] == os.environ["GDAL_DATA"]


def test_configure_geospatial_runtime_respects_opt_out(monkeypatch):
    stale_proj = r"C:\Program Files\PostgreSQL\16\share\contrib\postgis-3.4\proj"
    monkeypatch.setenv("KI_GEODATEN_RESPECT_PROJ", "1")
    monkeypatch.setenv("PROJ_LIB", stale_proj)

    configured = configure_geospatial_runtime()

    assert configured == {}
    assert os.environ["PROJ_LIB"] == stale_proj


def test_configure_geospatial_runtime_updates_loaded_rasterio_config(monkeypatch):
    import rasterio.env

    monkeypatch.setenv("PROJ_LIB", r"C:\Program Files\PostgreSQL\16\proj")

    configure_geospatial_runtime()

    assert rasterio.env.get_gdal_config("PROJ_LIB") == os.environ["PROJ_LIB"]
    assert rasterio.env.get_gdal_config("PROJ_DATA") == os.environ["PROJ_DATA"]
    assert rasterio.env.get_gdal_config("GDAL_DATA") == os.environ["GDAL_DATA"]
