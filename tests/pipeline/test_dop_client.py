import pytest
from pathlib import Path
import shutil

from ki_geodaten.models import TilePreset
from ki_geodaten.pipeline.dop_client import (
    DopDownloadError,
    PreparedBBox,
    _build_vrt,
    _build_session,
    _png_bytes_to_array,
    download_dop20,
    plan_chunk_grid,
    prepare_download_bbox,
)


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


def test_wms_session_ignores_environment_proxies(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")

    session = _build_session()

    assert session.trust_env is False


def test_png_decode_rejects_non_rgba():
    import io

    import numpy as np
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8), mode="RGB").save(buf, format="PNG")
    with pytest.raises(DopDownloadError, match="Expected RGBA"):
        _png_bytes_to_array(buf.getvalue())


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

    by_row = {}
    for chunk in chunks:
        by_row.setdefault(chunk.row, []).append(chunk)
    for row_chunks in by_row.values():
        row_chunks.sort(key=lambda c: c.col)
        for left, right in zip(row_chunks, row_chunks[1:]):
            assert left.maxx == right.minx

    by_col = {}
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


def test_download_dop20_http_success():
    import io
    from urllib.parse import parse_qs, urlparse

    import numpy as np
    import rasterio
    import responses
    from PIL import Image

    def make_png(width: int, height: int) -> bytes:
        arr = np.zeros((height, width, 4), dtype=np.uint8)
        arr[..., 0] = 10
        arr[..., 1] = 20
        arr[..., 2] = 30
        arr[..., 3] = 255
        buf = io.BytesIO()
        Image.fromarray(arr, mode="RGBA").save(buf, format="PNG")
        return buf.getvalue()

    def callback(request):
        query = parse_qs(urlparse(request.url).query)
        assert query["SERVICE"] == ["WMS"]
        assert query["VERSION"] == ["1.1.1"]
        assert query["REQUEST"] == ["GetMap"]
        assert query["LAYERS"] == ["by_dop20c"]
        assert query["SRS"] == ["EPSG:25832"]
        assert "CRS" not in query
        assert query["BBOX"] == ["0.0,0.0,204.8,204.8"]
        assert query["WIDTH"] == ["1024"]
        assert query["HEIGHT"] == ["1024"]
        assert query["FORMAT"] == ["image/png"]
        return (200, {"Content-Type": "image/png"}, make_png(1024, 1024))

    out_dir = Path("data/dop/test_dop_client_http_success")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    with responses.RequestsMock() as rsps:
        rsps.add_callback(
            responses.GET,
            "http://example/wms",
            callback=callback,
        )
        vrt_path = download_dop20(
            bbox_utm=(0.0, 0.0, 204.8, 204.8),
            out_dir=out_dir,
            wms_url="http://example/wms",
            layer="by_dop20c",
            wms_version="1.1.1",
            fmt="image/png",
            crs="EPSG:25832",
            max_pixels=6000,
            origin_x=0.0,
            origin_y=0.0,
        )

    assert vrt_path.exists()
    assert vrt_path.suffix == ".vrt"

    tif = next(out_dir.glob("chunk_*.tif"))
    with rasterio.open(tif) as src:
        assert src.crs.to_string() == "EPSG:25832"
        assert src.count == 4
        assert src.bounds.left == pytest.approx(0.0)
        assert src.bounds.right == pytest.approx(204.8)
        assert src.colorinterp[3].name == "alpha"
        mask = src.dataset_mask()
        assert mask.shape == (1024, 1024)
        assert mask.min() == 255

    with rasterio.open(vrt_path) as src:
        assert src.crs.to_string() == "EPSG:25832"
        assert src.count == 4
        assert src.bounds.right == pytest.approx(204.8)


def test_download_dop20_builds_vrt_without_osgeo(tmp_path, monkeypatch):
    import io
    import sys
    from urllib.parse import parse_qs, urlparse

    import numpy as np
    import rasterio
    import responses
    from PIL import Image

    class BlockOsgeoImporter:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "osgeo" or fullname.startswith("osgeo."):
                raise ModuleNotFoundError("blocked osgeo for test")
            return None

    blocker = BlockOsgeoImporter()
    monkeypatch.delitem(sys.modules, "osgeo", raising=False)
    monkeypatch.syspath_prepend(str(tmp_path / "empty_import_path"))
    sys.meta_path.insert(0, blocker)

    def make_png() -> bytes:
        arr = np.zeros((1024, 1024, 4), dtype=np.uint8)
        arr[..., 3] = 255
        buf = io.BytesIO()
        Image.fromarray(arr, mode="RGBA").save(buf, format="PNG")
        return buf.getvalue()

    def callback(request):
        query = parse_qs(urlparse(request.url).query)
        assert query["SRS"] == ["EPSG:25832"]
        return (200, {"Content-Type": "image/png"}, make_png())

    try:
        with responses.RequestsMock() as rsps:
            rsps.add_callback(responses.GET, "http://example/wms", callback=callback)
            vrt_path = download_dop20(
                bbox_utm=(0.0, 0.0, 204.8, 204.8),
                out_dir=tmp_path,
                wms_url="http://example/wms",
                layer="by_dop20c",
                wms_version="1.1.1",
                fmt="image/png",
                crs="EPSG:25832",
                max_pixels=6000,
                origin_x=0.0,
                origin_y=0.0,
            )

        with rasterio.open(vrt_path) as src:
            assert src.count == 4
            assert src.width == 1024
            assert src.height == 1024
            assert src.dataset_mask().min() == 255
    finally:
        sys.meta_path.remove(blocker)


def test_build_vrt_mosaics_adjacent_chunks(tmp_path):
    import numpy as np
    import rasterio
    from rasterio.transform import from_bounds

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
