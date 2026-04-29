import io
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pytest
import rasterio
import responses
from rasterio.transform import from_bounds

from ki_geodaten.pipeline.dem_client import (
    DemDownloadError,
    DemFetchResult,
    derive_ndsm_from_dom_dgm,
    fetch_dem,
    fetch_tiles_via_metalink,
)


def _make_single_band_geotiff_bytes(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
    *,
    fill: float = 5.0,
    dtype: str = "float32",
) -> bytes:
    transform = from_bounds(*bbox, width, height)
    profile = dict(
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype=dtype,
        crs="EPSG:25832",
        transform=transform,
    )
    arr = np.full((height, width), fill, dtype=dtype)
    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(**profile) as dst:
            dst.write(arr, 1)
        return memfile.read()


def test_fetch_dem_misconfigured_raises():
    with pytest.raises(DemDownloadError, match="misconfigured"):
        fetch_dem(
            (690000.0, 5334000.0, 690100.0, 5334100.0),
            Path("/tmp/dem_should_not_exist.tif"),
            wcs_url="",
            coverage_id="anything",
            wcs_version="2.0.1",
            fmt="image/tiff",
            crs="EPSG:25832",
            origin_x=0.0,
            origin_y=0.0,
            native_step_m=1.0,
        )


def test_fetch_dem_snaps_to_native_grid_and_calls_wcs(tmp_path):
    captured_params = {}

    def callback(request):
        query = parse_qs(urlparse(request.url).query)
        captured_params.update(query)
        # Server returns a 100x100 single-band TIFF for the snapped bbox.
        body = _make_single_band_geotiff_bytes(
            (690000.0, 5334000.0, 690100.0, 5334100.0),
            width=100,
            height=100,
            fill=3.5,
        )
        return (200, {"Content-Type": "image/tiff"}, body)

    out_path = tmp_path / "ndsm.tif"
    with responses.RequestsMock() as rsps:
        rsps.add_callback(responses.GET, "http://example/wcs/dem", callback=callback)
        result = fetch_dem(
            # Unaligned input — should snap outward to integer-metre grid.
            (690000.3, 5334000.6, 690099.7, 5334099.4),
            out_path,
            wcs_url="http://example/wcs/dem",
            coverage_id="dgm1",
            wcs_version="2.0.1",
            fmt="image/tiff",
            crs="EPSG:25832",
            origin_x=0.0,
            origin_y=0.0,
            native_step_m=1.0,
            username="user",
            password="pw",
        )

    assert isinstance(result, DemFetchResult)
    assert result.path == out_path
    # Snapped outward to 1 m grid.
    assert result.bbox_utm == (690000.0, 5334000.0, 690100.0, 5334100.0)
    assert captured_params["SERVICE"] == ["WCS"]
    assert captured_params["VERSION"] == ["2.0.1"]
    assert captured_params["REQUEST"] == ["GetCoverage"]
    assert captured_params["COVERAGEID"] == ["dgm1"]
    assert captured_params["FORMAT"] == ["image/tiff"]
    assert captured_params["SUBSET"] == ["X(690000.0,690100.0)", "Y(5334000.0,5334100.0)"]


def test_fetch_dem_writes_basic_auth(tmp_path):
    captured = {}

    def callback(request):
        captured["auth"] = request.headers.get("Authorization")
        return (
            200,
            {"Content-Type": "image/tiff"},
            _make_single_band_geotiff_bytes((690000.0, 5334000.0, 690100.0, 5334100.0), 100, 100),
        )

    with responses.RequestsMock() as rsps:
        rsps.add_callback(responses.GET, "http://example/wcs/dem", callback=callback)
        fetch_dem(
            (690000.0, 5334000.0, 690100.0, 5334100.0),
            tmp_path / "ndsm.tif",
            wcs_url="http://example/wcs/dem",
            coverage_id="dgm1",
            wcs_version="2.0.1",
            fmt="image/tiff",
            crs="EPSG:25832",
            origin_x=0.0,
            origin_y=0.0,
            native_step_m=1.0,
            username="alice",
            password="secret",
        )
    assert captured["auth"] is not None
    assert captured["auth"].startswith("Basic ")


def test_fetch_dem_rejects_xml_exception(tmp_path):
    with responses.RequestsMock() as rsps:
        rsps.add(
            responses.GET,
            "http://example/wcs/dem",
            body=b"<ows:ExceptionReport/>",
            status=200,
            content_type="text/xml",
        )
        with pytest.raises(DemDownloadError, match="non-tiff"):
            fetch_dem(
                (690000.0, 5334000.0, 690100.0, 5334100.0),
                tmp_path / "ndsm.tif",
                wcs_url="http://example/wcs/dem",
                coverage_id="dgm1",
                wcs_version="2.0.1",
                fmt="image/tiff",
                crs="EPSG:25832",
                origin_x=0.0,
                origin_y=0.0,
                native_step_m=1.0,
            )


def test_fetch_dem_persists_bytes_verbatim(tmp_path):
    payload = _make_single_band_geotiff_bytes(
        (690000.0, 5334000.0, 690100.0, 5334100.0), 100, 100, fill=7.25
    )

    with responses.RequestsMock() as rsps:
        rsps.add(
            responses.GET,
            "http://example/wcs/dem",
            body=payload,
            status=200,
            content_type="image/tiff",
        )
        out_path = tmp_path / "ndsm.tif"
        fetch_dem(
            (690000.0, 5334000.0, 690100.0, 5334100.0),
            out_path,
            wcs_url="http://example/wcs/dem",
            coverage_id="dgm1",
            wcs_version="2.0.1",
            fmt="image/tiff",
            crs="EPSG:25832",
            origin_x=0.0,
            origin_y=0.0,
            native_step_m=1.0,
        )
    with rasterio.open(out_path) as src:
        assert src.count == 1
        assert src.width == 100
        assert src.height == 100
        np.testing.assert_allclose(src.read(1).mean(), 7.25, atol=1e-3)


def test_derive_ndsm_from_dom_minus_dgm(tmp_path):
    bbox = (690000.0, 5334000.0, 690010.0, 5334010.0)
    dom_path = tmp_path / "dom.tif"
    dgm_path = tmp_path / "dgm.tif"
    out_path = tmp_path / "ndsm.tif"
    dom_path.write_bytes(_make_single_band_geotiff_bytes(bbox, 10, 10, fill=125.0))
    dgm_path.write_bytes(_make_single_band_geotiff_bytes(bbox, 10, 10, fill=120.0))

    result = derive_ndsm_from_dom_dgm(
        dom_path=dom_path,
        dgm_path=dgm_path,
        out_path=out_path,
    )

    assert result.path == out_path
    with rasterio.open(out_path) as src:
        assert src.count == 1
        assert src.dtypes[0] == "float32"
        np.testing.assert_allclose(src.read(1), 5.0, atol=1e-5)


def test_fetch_tiles_via_metalink_downloads_tiles_and_builds_vrt(tmp_path):
    metalink = """<?xml version="1.0" encoding="UTF-8"?>
<metalink xmlns="urn:ietf:params:xml:ns:metalink">
  <file name="32567_5516_20_DOM.tif">
    <url>http://example/32567_5516_20_DOM.tif</url>
  </file>
</metalink>
"""
    bbox = (567000.0, 5516000.0, 568000.0, 5517000.0)
    tile_bytes = _make_single_band_geotiff_bytes(bbox, 10, 10, fill=123.0)

    with responses.RequestsMock() as rsps:
        rsps.add(
            responses.POST,
            "http://example/metalink/dom20dom",
            body=metalink,
            status=200,
            content_type="application/metalink4+xml",
        )
        rsps.add(
            responses.GET,
            "http://example/32567_5516_20_DOM.tif",
            body=tile_bytes,
            status=200,
            content_type="image/tiff",
        )
        result = fetch_tiles_via_metalink(
            bbox,
            tmp_path / "job",
            metalink_url="http://example/metalink/dom20dom",
            cache_dir=tmp_path / "cache",
            vrt_name="dom.vrt",
        )

    assert result.path == tmp_path / "job" / "dom.vrt"
    assert result.tile_paths == (tmp_path / "cache" / "32567_5516_20_DOM.tif",)
    with rasterio.open(result.path) as src:
        assert src.count == 1
        np.testing.assert_allclose(src.read(1).mean(), 123.0, atol=1e-3)


def test_fetch_tiles_via_metalink_sets_text_plain_content_type(tmp_path):
    """The LDBV poly2metalink endpoint expects an explicit Content-Type
    on the EWKT body. Verify our client sets it instead of relying on
    requests' default (which is no Content-Type for str payloads)."""
    captured: dict[str, str | bytes] = {}

    bbox = (567000.0, 5516000.0, 568000.0, 5517000.0)
    tile_bytes = _make_single_band_geotiff_bytes(bbox, 10, 10, fill=99.0)

    def metalink_callback(request):
        captured["content_type"] = request.headers.get("Content-Type", "")
        captured["body"] = request.body
        body = """<?xml version="1.0" encoding="UTF-8"?>
<metalink xmlns="urn:ietf:params:xml:ns:metalink">
  <file name="ok.tif"><url>http://example/ok.tif</url></file>
</metalink>"""
        return (200, {"Content-Type": "application/metalink4+xml"}, body)

    with responses.RequestsMock() as rsps:
        rsps.add_callback(
            responses.POST,
            "http://example/metalink/dom20dom",
            callback=metalink_callback,
        )
        rsps.add(
            responses.GET,
            "http://example/ok.tif",
            body=tile_bytes,
            status=200,
            content_type="image/tiff",
        )
        fetch_tiles_via_metalink(
            bbox,
            tmp_path / "job",
            metalink_url="http://example/metalink/dom20dom",
            cache_dir=tmp_path / "cache",
        )

    assert captured["content_type"].startswith("text/plain")
    # Body is bytes (we encode) — make sure the EWKT survived round-trip.
    body_text = captured["body"].decode("utf-8") if isinstance(captured["body"], (bytes, bytearray)) else captured["body"]
    assert body_text.startswith("SRID=4326;POLYGON((")


def test_fetch_tiles_via_metalink_rejects_path_traversal_filename(tmp_path):
    """A bad <file name="..."> must not cause writes outside the cache dir."""
    metalink = """<?xml version="1.0" encoding="UTF-8"?>
<metalink xmlns="urn:ietf:params:xml:ns:metalink">
  <file name="../../../evil.tif">
    <url>http://example/whatever.tif</url>
  </file>
</metalink>
"""
    with responses.RequestsMock() as rsps:
        rsps.add(
            responses.POST,
            "http://example/metalink/dom20dom",
            body=metalink,
            status=200,
            content_type="application/metalink4+xml",
        )
        with pytest.raises(DemDownloadError, match="invalid tile filename"):
            fetch_tiles_via_metalink(
                (567000.0, 5516000.0, 568000.0, 5517000.0),
                tmp_path / "job",
                metalink_url="http://example/metalink/dom20dom",
                cache_dir=tmp_path / "cache",
            )


def test_fetch_tiles_via_metalink_atomic_write_recovers_from_partial_file(tmp_path):
    """If a previous worker died mid-write, a leftover .part file must not
    be confused with a valid cached tile, and a fresh write must succeed."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    # Simulate a leftover from a previous crash: a half-written .part file
    # next to where the real tile would land.
    leftover = cache_dir / "ok.tif.part"
    leftover.write_bytes(b"truncated bytes")

    bbox = (567000.0, 5516000.0, 568000.0, 5517000.0)
    tile_bytes = _make_single_band_geotiff_bytes(bbox, 10, 10, fill=42.0)
    metalink = """<?xml version="1.0" encoding="UTF-8"?>
<metalink xmlns="urn:ietf:params:xml:ns:metalink">
  <file name="ok.tif"><url>http://example/ok.tif</url></file>
</metalink>"""

    with responses.RequestsMock() as rsps:
        rsps.add(
            responses.POST,
            "http://example/metalink/dom20dom",
            body=metalink,
            status=200,
            content_type="application/metalink4+xml",
        )
        rsps.add(
            responses.GET,
            "http://example/ok.tif",
            body=tile_bytes,
            status=200,
            content_type="image/tiff",
        )
        result = fetch_tiles_via_metalink(
            bbox,
            tmp_path / "job",
            metalink_url="http://example/metalink/dom20dom",
            cache_dir=cache_dir,
        )

    final = cache_dir / "ok.tif"
    assert final.exists()
    assert not (cache_dir / "ok.tif.part").exists()  # cleaned up
    with rasterio.open(final) as src:
        np.testing.assert_allclose(src.read(1).mean(), 42.0, atol=1e-3)


def test_derive_ndsm_pins_output_crs_even_when_source_crs_invalid(tmp_path):
    """LDBV TIFFs sometimes ship a LOCAL_CS-style block that rasterio can't
    resolve. The nDSM output must still carry the canonical EPSG:25832."""
    bbox = (690000.0, 5334000.0, 690010.0, 5334010.0)
    dom_path = tmp_path / "dom.tif"
    dgm_path = tmp_path / "dgm.tif"
    out_path = tmp_path / "ndsm.tif"
    dom_path.write_bytes(_make_single_band_geotiff_bytes(bbox, 10, 10, fill=100.0))
    dgm_path.write_bytes(_make_single_band_geotiff_bytes(bbox, 10, 10, fill=98.0))

    derive_ndsm_from_dom_dgm(
        dom_path=dom_path,
        dgm_path=dgm_path,
        out_path=out_path,
    )

    with rasterio.open(out_path) as src:
        assert src.crs is not None
        assert src.crs.to_epsg() == 25832
        np.testing.assert_allclose(src.read(1), 2.0, atol=1e-5)


def test_derive_ndsm_resamples_dgm_onto_finer_dom_grid(tmp_path):
    """DOM 0.2 m + DGM 1 m: the output must adopt DOM's grid and DGM
    must be bilinearly upsampled before the subtraction."""
    bbox = (690000.0, 5334000.0, 690010.0, 5334010.0)
    dom_path = tmp_path / "dom.tif"
    dgm_path = tmp_path / "dgm.tif"
    out_path = tmp_path / "ndsm.tif"
    # DOM at 0.2 m: 50 × 50 pixels, all 105 m
    dom_path.write_bytes(
        _make_single_band_geotiff_bytes(bbox, 50, 50, fill=105.0)
    )
    # DGM at 1.0 m: 10 × 10 pixels, all 100 m
    dgm_path.write_bytes(
        _make_single_band_geotiff_bytes(bbox, 10, 10, fill=100.0)
    )

    derive_ndsm_from_dom_dgm(
        dom_path=dom_path,
        dgm_path=dgm_path,
        out_path=out_path,
    )

    with rasterio.open(out_path) as src:
        assert src.width == 50
        assert src.height == 50
        # Exact subtraction over the constant fields = 5 m everywhere.
        np.testing.assert_allclose(src.read(1), 5.0, atol=1e-4)


def test_derive_ndsm_writes_geotiff_when_dom_source_is_vrt(tmp_path):
    bbox = (690000.0, 5334000.0, 690010.0, 5334010.0)
    dom_tile = tmp_path / "dom_tile.tif"
    dgm_path = tmp_path / "dgm.tif"
    vrt_path = tmp_path / "dom.vrt"
    out_path = tmp_path / "ndsm.tif"
    dom_tile.write_bytes(_make_single_band_geotiff_bytes(bbox, 10, 10, fill=125.0))
    dgm_path.write_bytes(_make_single_band_geotiff_bytes(bbox, 10, 10, fill=120.0))
    vrt_path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<VRTDataset rasterXSize="10" rasterYSize="10">
  <SRS>EPSG:25832</SRS>
  <GeoTransform>{bbox[0]}, 1.0, 0.0, {bbox[3]}, 0.0, -1.0</GeoTransform>
  <VRTRasterBand dataType="Float32" band="1">
    <SimpleSource>
      <SourceFilename relativeToVRT="1">{dom_tile.name}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="10" ySize="10"/>
      <DstRect xOff="0" yOff="0" xSize="10" ySize="10"/>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>
""",
        encoding="utf-8",
    )

    derive_ndsm_from_dom_dgm(dom_path=vrt_path, dgm_path=dgm_path, out_path=out_path)

    with rasterio.open(out_path) as src:
        assert src.driver == "GTiff"
        np.testing.assert_allclose(src.read(1), 5.0, atol=1e-5)


def test_fetch_tiles_via_metalink_rejects_empty_response(tmp_path):
    with responses.RequestsMock() as rsps:
        rsps.add(
            responses.POST,
            "http://example/metalink/dom20dom",
            body='<metalink xmlns="urn:ietf:params:xml:ns:metalink"/>',
            status=200,
            content_type="application/metalink4+xml",
        )
        with pytest.raises(DemDownloadError, match="did not contain files"):
            fetch_tiles_via_metalink(
                (567000.0, 5516000.0, 568000.0, 5517000.0),
                tmp_path,
                metalink_url="http://example/metalink/dom20dom",
            )
