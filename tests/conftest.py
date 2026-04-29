from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import pytest

from ki_geodaten.geospatial_env import configure_geospatial_runtime


def _pin_proj_data_dir() -> None:
    configure_geospatial_runtime()


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
