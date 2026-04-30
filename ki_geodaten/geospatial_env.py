from __future__ import annotations

import os
import sys
from importlib.util import find_spec
from pathlib import Path


def _package_dir(package: str) -> Path | None:
    spec = find_spec(package)
    if spec is None or spec.submodule_search_locations is None:
        return None
    try:
        return Path(next(iter(spec.submodule_search_locations)))
    except StopIteration:
        return None


def _bundled_pyproj_data_dir() -> Path | None:
    package_dir = _package_dir("pyproj")
    if package_dir is None:
        return None

    candidate = package_dir / "proj_dir" / "share" / "proj"
    if (candidate / "proj.db").is_file():
        return candidate
    return None


def _conda_share_dir(name: str, sentinel: str) -> Path | None:
    roots = [os.environ.get("CONDA_PREFIX"), sys.prefix]
    seen: set[Path] = set()
    for raw_root in roots:
        if not raw_root:
            continue
        root = Path(raw_root)
        if root in seen:
            continue
        seen.add(root)
        candidates = (
            root / "Library" / "share" / name,
            root / "share" / name,
        )
        for candidate in candidates:
            if (candidate / sentinel).is_file():
                return candidate
    return None


def _bundled_rasterio_gdal_data_dir() -> Path | None:
    package_dir = _package_dir("rasterio")
    if package_dir is None:
        return None

    candidate = package_dir / "gdal_data"
    if candidate.is_dir():
        return candidate
    return None


def _set_gdal_config_if_loaded(key: str, value: str) -> None:
    if "rasterio" not in sys.modules and "rasterio.env" not in sys.modules:
        return
    try:
        import rasterio.env

        rasterio.env.set_gdal_config(key, value)
    except Exception:
        return


def configure_geospatial_runtime() -> dict[str, str]:
    """Pin GDAL/PROJ data paths to the bundled package databases.

    Windows development shells often inherit PostgreSQL/PostGIS variables such
    as PROJ_LIB. Those can point at an older proj.db, which then breaks rasterio
    or pyproj with DATABASE.LAYOUT.VERSION errors during worker inference.
    """
    if os.environ.get("KI_GEODATEN_RESPECT_PROJ") == "1":
        return {}

    configured: dict[str, str] = {}

    proj_dir = _bundled_pyproj_data_dir() or _conda_share_dir("proj", "proj.db")
    if proj_dir is not None:
        proj_value = str(proj_dir)
        os.environ["PROJ_DATA"] = proj_value
        os.environ["PROJ_LIB"] = proj_value
        configured["PROJ_DATA"] = proj_value
        configured["PROJ_LIB"] = proj_value
        _set_gdal_config_if_loaded("PROJ_DATA", proj_value)
        _set_gdal_config_if_loaded("PROJ_LIB", proj_value)

        if "pyproj" in sys.modules or "pyproj.datadir" in sys.modules:
            import pyproj.datadir

            pyproj.datadir.set_data_dir(proj_value)

    gdal_dir = _bundled_rasterio_gdal_data_dir() or _conda_share_dir("gdal", "gdalvrt.xsd")
    if gdal_dir is not None:
        gdal_value = str(gdal_dir)
        os.environ["GDAL_DATA"] = gdal_value
        configured["GDAL_DATA"] = gdal_value
        _set_gdal_config_if_loaded("GDAL_DATA", gdal_value)

    return configured
