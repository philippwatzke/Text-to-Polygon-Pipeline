import io
import shutil
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pytest
import rasterio
import responses
from PIL import Image
from rasterio.enums import ColorInterp
from rasterio.transform import from_bounds

from ki_geodaten.models import TilePreset
from ki_geodaten.pipeline.dop_client import (
    DopDownloadError,
    PreparedBBox,
    _build_session,
    _build_vrt,
    _wcs_get_coverage_params,
    download_dop20,
    plan_chunk_grid,
    prepare_download_bbox,
)
import ki_geodaten.pipeline.dop_client as dop_client


def test_prepare_expands_by_center_margin_medium():
    result = prepare_download_bbox(
        minx=691000.0,
        miny=5335000.0,
        maxx=692000.0,
        maxy=5336000.0,
        preset=TilePreset.MEDIUM,
        origin_x=0.0,
        origin_y=0.0,
        step=0.2,
    )
    assert isinstance(result, PreparedBBox)
    assert result.download_bbox == (690936.0, 5334936.0, 692064.0, 5336064.0)
    assert result.aoi_bbox == (691000.0, 5335000.0, 692000.0, 5336000.0)


def test_prepare_expands_by_center_margin_small():
    result = prepare_download_bbox(
        minx=691000.0,
        miny=5335000.0,
        maxx=692000.0,
        maxy=5336000.0,
        preset=TilePreset.SMALL,
        origin_x=0.0,
        origin_y=0.0,
        step=0.2,
    )
    assert result.download_bbox == (690968.0, 5334968.0, 692032.0, 5336032.0)


def test_prepare_expands_by_center_margin_large():
    result = prepare_download_bbox(
        minx=691000.0,
        miny=5335000.0,
        maxx=692000.0,
        maxy=5336000.0,
        preset=TilePreset.LARGE,
        origin_x=0.0,
        origin_y=0.0,
        step=0.2,
    )
    assert result.download_bbox == (690904.0, 5334904.0, 692096.0, 5336096.0)


def test_prepare_minimum_size_expands_to_204_8():
    result = prepare_download_bbox(
        minx=691000.0,
        miny=5335000.0,
        maxx=691050.0,
        maxy=5335050.0,
        preset=TilePreset.MEDIUM,
        origin_x=0.0,
        origin_y=0.0,
        step=0.2,
    )
    dx = result.download_bbox[2] - result.download_bbox[0]
    dy = result.download_bbox[3] - result.download_bbox[1]
    assert dx >= 204.8
    assert dy >= 204.8


def test_prepare_snaps_unaligned_input():
    result = prepare_download_bbox(
        minx=691000.15,
        miny=5335000.07,
        maxx=692000.05,
        maxy=5336000.11,
        preset=TilePreset.MEDIUM,
        origin_x=0.0,
        origin_y=0.0,
        step=0.2,
    )
    minx, miny, maxx, maxy = result.aoi_bbox
    assert minx == pytest.approx(691000.0)
    assert miny == pytest.approx(5335000.0)
    assert maxx == pytest.approx(692000.2)
    assert maxy == pytest.approx(5336000.2)


def test_prepare_with_nonzero_origin():
    result = prepare_download_bbox(
        minx=691000.1,
        miny=5335000.1,
        maxx=691500.1,
        maxy=5335500.1,
        preset=TilePreset.MEDIUM,
        origin_x=0.1,
        origin_y=0.1,
        step=0.2,
    )
    for coord in result.aoi_bbox:
        assert abs(((coord - 0.1) / 0.2) - round((coord - 0.1) / 0.2)) < 1e-9


def test_download_error_is_exception_type():
    assert issubclass(DopDownloadError, Exception)


def test_session_ignores_environment_proxies(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")

    session = _build_session()

    assert session.trust_env is False


def test_session_uses_basic_auth_when_credentials_provided():
    session = _build_session(username="user", password="secret")
    assert session.auth is not None


def test_session_skips_auth_when_no_credentials():
    session = _build_session(username="", password="")
    assert session.auth is None


def test_wcs_get_coverage_params_uses_201_subset_form():
    from ki_geodaten.pipeline.dop_client import ChunkPlan

    chunk = ChunkPlan(row=0, col=0, minx=0.0, miny=0.0, maxx=204.8, maxy=204.8)
    params = _wcs_get_coverage_params(
        chunk,
        coverage_id="by_dop20c",
        wcs_version="2.0.1",
        fmt="image/tiff",
        crs="EPSG:25832",
    )
    keys = [name for name, _ in params]
    values = dict((k, v) for k, v in params if k != "SUBSET")
    subsets = [v for k, v in params if k == "SUBSET"]

    assert keys.count("SUBSET") == 2
    assert values["SERVICE"] == "WCS"
    assert values["VERSION"] == "2.0.1"
    assert values["REQUEST"] == "GetCoverage"
    assert values["COVERAGEID"] == "by_dop20c"
    assert values["FORMAT"] == "image/tiff"
    assert values["SUBSETTINGCRS"].endswith("/25832")
    assert values["OUTPUTCRS"].endswith("/25832")
    assert subsets == ["X(0.0,204.8)", "Y(0.0,204.8)"]


def test_chunk_grid_seamless_edges():
    chunks = plan_chunk_grid(
        minx=0.0,
        miny=0.0,
        maxx=2000.0,
        maxy=2000.0,
        max_pixels=6000,
        step=0.2,
        origin_x=0.0,
        origin_y=0.0,
    )

    assert len(chunks) == 4
    for chunk in chunks:
        for value in (chunk.minx, chunk.miny, chunk.maxx, chunk.maxy):
            assert abs(round(value / 0.2) * 0.2 - value) < 1e-9

    by_row: dict = {}
    for chunk in chunks:
        by_row.setdefault(chunk.row, []).append(chunk)
    for row_chunks in by_row.values():
        row_chunks.sort(key=lambda c: c.col)
        for left, right in zip(row_chunks, row_chunks[1:]):
            assert left.maxx == right.minx

    by_col: dict = {}
    for chunk in chunks:
        by_col.setdefault(chunk.col, []).append(chunk)
    for col_chunks in by_col.values():
        col_chunks.sort(key=lambda c: c.row)
        for lower, upper in zip(col_chunks, col_chunks[1:]):
            assert lower.maxy == upper.miny


def test_chunk_grid_small_bbox_single_chunk():
    chunks = plan_chunk_grid(
        minx=0.0,
        miny=0.0,
        maxx=204.8,
        maxy=204.8,
        max_pixels=6000,
        step=0.2,
        origin_x=0.0,
        origin_y=0.0,
    )
    assert len(chunks) == 1
    assert chunks[0].width_px() == 1024
    assert chunks[0].height_px() == 1024


def _make_geotiff_bytes(
    bbox: tuple[float, float, float, float],
    *,
    width: int,
    height: int,
    band_count: int = 4,
    crs: str = "EPSG:25832",
) -> bytes:
    """Render an in-memory GeoTIFF that mimics a WCS GetCoverage response."""
    transform = from_bounds(*bbox, width, height)
    profile = dict(
        driver="GTiff",
        width=width,
        height=height,
        count=band_count,
        dtype="uint8",
        crs=crs,
        transform=transform,
    )
    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(**profile) as dst:
            for band_idx in range(1, band_count + 1):
                fill = (band_idx * 50) % 256
                dst.write(np.full((height, width), fill, dtype=np.uint8), band_idx)
        return memfile.read()


def test_download_dop20_http_success():
    def callback(request):
        query = parse_qs(urlparse(request.url).query)
        assert query["SERVICE"] == ["WCS"]
        assert query["VERSION"] == ["2.0.1"]
        assert query["REQUEST"] == ["GetCoverage"]
        assert query["COVERAGEID"] == ["by_dop20c"]
        assert query["FORMAT"] == ["image/tiff"]
        # WCS 2.0.1 emits SUBSET twice
        assert query["SUBSET"] == ["X(0.0,204.8)", "Y(0.0,204.8)"]
        assert query["SUBSETTINGCRS"][0].endswith("/25832")
        assert query["OUTPUTCRS"][0].endswith("/25832")
        body = _make_geotiff_bytes((0.0, 0.0, 204.8, 204.8), width=1024, height=1024)
        return (200, {"Content-Type": "image/tiff"}, body)

    out_dir = Path("data/dop/test_dop_client_http_success")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    with responses.RequestsMock() as rsps:
        rsps.add_callback(
            responses.GET,
            "http://example/wcs",
            callback=callback,
        )
        vrt_path = download_dop20(
            bbox_utm=(0.0, 0.0, 204.8, 204.8),
            out_dir=out_dir,
            wcs_url="http://example/wcs",
            coverage_id="by_dop20c",
            wcs_version="2.0.1",
            fmt="image/tiff",
            crs="EPSG:25832",
            max_pixels=6000,
            origin_x=0.0,
            origin_y=0.0,
            username="user",
            password="pw",
        )

    assert vrt_path.exists()
    assert vrt_path.suffix == ".vrt"

    tif = next(out_dir.glob("chunk_*.tif"))
    with rasterio.open(tif) as src:
        assert src.crs.to_string() == "EPSG:25832"
        assert src.count == 4
        assert src.width == 1024
        assert src.height == 1024
        assert src.bounds.left == pytest.approx(0.0)
        assert src.bounds.right == pytest.approx(204.8)

    with rasterio.open(vrt_path) as src:
        assert src.crs.to_string() == "EPSG:25832"
        assert src.count == 4
        assert src.bounds.right == pytest.approx(204.8)


def test_download_dop20_passes_basic_auth():
    captured = {}

    def callback(request):
        captured["auth_header"] = request.headers.get("Authorization")
        body = _make_geotiff_bytes((0.0, 0.0, 204.8, 204.8), width=1024, height=1024)
        return (200, {"Content-Type": "image/tiff"}, body)

    out_dir = Path("data/dop/test_dop_client_auth")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    with responses.RequestsMock() as rsps:
        rsps.add_callback(responses.GET, "http://example/wcs", callback=callback)
        download_dop20(
            bbox_utm=(0.0, 0.0, 204.8, 204.8),
            out_dir=out_dir,
            wcs_url="http://example/wcs",
            coverage_id="by_dop20c",
            wcs_version="2.0.1",
            fmt="image/tiff",
            crs="EPSG:25832",
            max_pixels=6000,
            origin_x=0.0,
            origin_y=0.0,
            username="user",
            password="pw",
        )

    assert captured["auth_header"] is not None
    assert captured["auth_header"].startswith("Basic ")


def test_download_dop20_fills_wcs_rgb_zero_pixels_from_wms(tmp_path):
    def make_wcs_with_black_patch() -> bytes:
        transform = from_bounds(0.0, 0.0, 204.8, 204.8, 1024, 1024)
        with rasterio.io.MemoryFile() as memfile:
            with memfile.open(
                driver="GTiff",
                width=1024,
                height=1024,
                count=4,
                dtype="uint8",
                crs="EPSG:25832",
                transform=transform,
            ) as dst:
                rgb = np.full((3, 1024, 1024), 100, dtype=np.uint8)
                rgb[:, 100:110, 200:210] = 0
                dst.write(rgb, indexes=[1, 2, 3])
                dst.write(np.full((1024, 1024), 255, dtype=np.uint8), 4)
            return memfile.read()

    def make_wms_png() -> bytes:
        arr = np.zeros((1024, 1024, 4), dtype=np.uint8)
        arr[..., 0] = 11
        arr[..., 1] = 22
        arr[..., 2] = 33
        arr[..., 3] = 255
        buf = io.BytesIO()
        Image.fromarray(arr, mode="RGBA").save(buf, format="PNG")
        return buf.getvalue()

    with responses.RequestsMock() as rsps:
        rsps.add(
            responses.GET,
            "http://example/wcs",
            body=make_wcs_with_black_patch(),
            status=200,
            content_type="image/tiff",
        )
        rsps.add(
            responses.GET,
            "http://example/wms",
            body=make_wms_png(),
            status=200,
            content_type="image/png",
        )
        download_dop20(
            bbox_utm=(0.0, 0.0, 204.8, 204.8),
            out_dir=tmp_path,
            wcs_url="http://example/wcs",
            coverage_id="by_dop20c",
            wcs_version="2.0.1",
            fmt="image/tiff",
            crs="EPSG:25832",
            max_pixels=6000,
            origin_x=0.0,
            origin_y=0.0,
            fill_rgb_zero_with_wms=True,
            wms_url="http://example/wms",
            wms_layer="by_dop20c",
        )

    with rasterio.open(tmp_path / "chunk_0_0.tif") as src:
        assert src.read(1)[105, 205] == 11
        assert src.read(2)[105, 205] == 22
        assert src.read(3)[105, 205] == 33
        assert src.read(1)[50, 50] == 100


def test_download_dop20_rejects_xml_exception_report():
    xml_body = b"""<?xml version='1.0'?>
<ows:ExceptionReport xmlns:ows='http://www.opengis.net/ows/2.0'>
  <ows:Exception exceptionCode='InvalidParameterValue'/>
</ows:ExceptionReport>"""

    out_dir = Path("data/dop/test_dop_client_xml_error")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    with responses.RequestsMock() as rsps:
        rsps.add(
            responses.GET,
            "http://example/wcs",
            body=xml_body,
            status=200,
            content_type="text/xml",
        )
        with pytest.raises(DopDownloadError, match="non-tiff"):
            download_dop20(
                bbox_utm=(0.0, 0.0, 204.8, 204.8),
                out_dir=out_dir,
                wcs_url="http://example/wcs",
                coverage_id="by_dop20c",
                wcs_version="2.0.1",
                fmt="image/tiff",
                crs="EPSG:25832",
                max_pixels=6000,
                origin_x=0.0,
                origin_y=0.0,
            )


def test_download_dop20_builds_vrt_without_osgeo(tmp_path, monkeypatch):
    import sys

    class BlockOsgeoImporter:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "osgeo" or fullname.startswith("osgeo."):
                raise ModuleNotFoundError("blocked osgeo for test")
            return None

    blocker = BlockOsgeoImporter()
    monkeypatch.delitem(sys.modules, "osgeo", raising=False)
    monkeypatch.syspath_prepend(str(tmp_path / "empty_import_path"))
    sys.meta_path.insert(0, blocker)

    def callback(request):
        body = _make_geotiff_bytes((0.0, 0.0, 204.8, 204.8), width=1024, height=1024)
        return (200, {"Content-Type": "image/tiff"}, body)

    try:
        with responses.RequestsMock() as rsps:
            rsps.add_callback(responses.GET, "http://example/wcs", callback=callback)
            vrt_path = download_dop20(
                bbox_utm=(0.0, 0.0, 204.8, 204.8),
                out_dir=tmp_path,
                wcs_url="http://example/wcs",
                coverage_id="by_dop20c",
                wcs_version="2.0.1",
                fmt="image/tiff",
                crs="EPSG:25832",
                max_pixels=6000,
                origin_x=0.0,
                origin_y=0.0,
            )

        with rasterio.open(vrt_path) as src:
            assert src.count == 4
            assert src.width == 1024
            assert src.height == 1024
    finally:
        sys.meta_path.remove(blocker)


def test_download_dop20_parallel_fetch_keeps_vrt_chunk_order(tmp_path, monkeypatch):
    calls = []
    built = {}

    def fake_fetch_chunk(session, wcs_url, coverage_id, chunk, out_path, **kwargs):
        time.sleep(0.02 if chunk.col == 0 else 0.0)
        out_path.write_bytes(b"chunk")
        calls.append((chunk.row, chunk.col))

    def fake_build_vrt(vrt_path, chunk_files, *, crs, step=0.2):
        built["files"] = [path.name for path in chunk_files]
        vrt_path.write_text("vrt", encoding="utf-8")

    monkeypatch.setattr(dop_client, "_fetch_chunk", fake_fetch_chunk)
    monkeypatch.setattr(dop_client, "_build_vrt", fake_build_vrt)

    vrt_path = download_dop20(
        bbox_utm=(0.0, 0.0, 409.6, 204.8),
        out_dir=tmp_path,
        wcs_url="http://example/wcs",
        coverage_id="by_dop20c",
        wcs_version="2.0.1",
        fmt="image/tiff",
        crs="EPSG:25832",
        max_pixels=1024,
        origin_x=0.0,
        origin_y=0.0,
        download_workers=2,
    )

    assert vrt_path == tmp_path / "out.vrt"
    assert sorted(calls) == [(0, 0), (0, 1)]
    assert built["files"] == ["chunk_0_0.tif", "chunk_0_1.tif"]


def test_build_vrt_mosaics_adjacent_chunks(tmp_path):
    left = tmp_path / "left.tif"
    right = tmp_path / "right.tif"
    for path, bbox, value in (
        (left, (0.0, 0.0, 204.8, 204.8), 10),
        (right, (204.8, 0.0, 409.6, 204.8), 20),
    ):
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=1024,
            height=1024,
            count=4,
            dtype="uint8",
            crs="EPSG:25832",
            transform=from_bounds(*bbox, 1024, 1024),
        ) as dst:
            dst.write(np.full((3, 1024, 1024), value, dtype=np.uint8), indexes=[1, 2, 3])
            dst.write(np.full((1024, 1024), 255, dtype=np.uint8), 4)

    vrt_path = tmp_path / "out.vrt"
    _build_vrt(vrt_path, [left, right], crs="EPSG:25832")

    with rasterio.open(vrt_path) as src:
        assert src.width == 2048
        assert src.height == 1024
        assert src.bounds.left == pytest.approx(0.0)
        assert src.bounds.right == pytest.approx(409.6)
        assert src.read(1, window=((0, 1), (1023, 1025))).tolist() == [[10, 20]]


def test_build_vrt_treats_fourth_wcs_band_as_data_not_alpha(tmp_path):
    path = tmp_path / "wcs_rgb_ir.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=1024,
        height=1024,
        count=4,
        dtype="uint8",
        crs="EPSG:25832",
        transform=from_bounds(0.0, 0.0, 204.8, 204.8, 1024, 1024),
    ) as dst:
        dst.write(np.full((3, 1024, 1024), 128, dtype=np.uint8), indexes=[1, 2, 3])
        dst.write(np.zeros((1024, 1024), dtype=np.uint8), 4)
        dst.colorinterp = (
            ColorInterp.red,
            ColorInterp.green,
            ColorInterp.blue,
            ColorInterp.alpha,
        )

    vrt_path = tmp_path / "out.vrt"
    _build_vrt(vrt_path, [path], crs="EPSG:25832")

    with rasterio.open(vrt_path) as src:
        assert src.colorinterp[3] == ColorInterp.undefined
        assert np.all(src.dataset_mask(window=((320, 704), (320, 704))) == 255)


def test_build_vrt_supports_three_band_output(tmp_path):
    left = tmp_path / "left.tif"
    with rasterio.open(
        left,
        "w",
        driver="GTiff",
        width=1024,
        height=1024,
        count=3,
        dtype="uint8",
        crs="EPSG:25832",
        transform=from_bounds(0.0, 0.0, 204.8, 204.8, 1024, 1024),
    ) as dst:
        dst.write(np.full((3, 1024, 1024), 50, dtype=np.uint8), indexes=[1, 2, 3])

    vrt_path = tmp_path / "out.vrt"
    _build_vrt(vrt_path, [left], crs="EPSG:25832")

    with rasterio.open(vrt_path) as src:
        assert src.count == 3
