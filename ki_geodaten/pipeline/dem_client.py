"""Clients for auxiliary elevation data used to derive nDSM.

Separate from ``dop_client`` because:

- DOP20 is a 4-band uint8 raster requested as a tiled mosaic (~1 GB per
  km²); DEM/nDSM inputs are single-band OpenData GeoTIFF tiles.
- DOP20 sits on a 0.2 m grid with origin (0.1, 0.1); the DEM grid is
  service-specific (LDBV DGM1 is at integer-metre origin, 1 m step).
- Spec §5.x (NDVI/nDSM extension): the DEM raster is later resampled onto
  the DOP tile grid by the tiler. The client itself only needs to deliver
  a georeferenced, single-band float raster covering the requested AOI.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import rasterio
import requests
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import reproject
from requests.adapters import HTTPAdapter
from requests.auth import HTTPBasicAuth
from urllib3.util.retry import Retry

from ki_geodaten.pipeline.geo_utils import snap_ceil, snap_floor, transformer_25832_to_4326


class DemDownloadError(Exception):
    """Transport or content error from the DEM/nDSM clients."""


@dataclass(frozen=True)
class DemFetchResult:
    path: Path
    bbox_utm: tuple[float, float, float, float]
    native_step_m: float


@dataclass(frozen=True)
class NdsmDeriveResult:
    path: Path
    dom_path: Path
    dgm_path: Path


@dataclass(frozen=True)
class TileFetchResult:
    path: Path
    tile_paths: tuple[Path, ...]
    bbox_utm: tuple[float, float, float, float]


def _build_session(username: str = "", password: str = "") -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    if username:
        session.auth = HTTPBasicAuth(username, password)
    retry = Retry(
        total=3,
        backoff_factor=2.0,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _epsg_code(crs: str) -> str:
    if ":" in crs:
        return crs.split(":", 1)[1]
    return crs


def _snap_to_native_grid(
    bbox_utm: tuple[float, float, float, float],
    *,
    origin_x: float,
    origin_y: float,
    step: float,
) -> tuple[float, float, float, float]:
    """Snap an AOI outward to the DEM's native grid so the server doesn't resample."""
    minx, miny, maxx, maxy = bbox_utm
    return (
        snap_floor(minx, origin_x, step),
        snap_floor(miny, origin_y, step),
        snap_ceil(maxx, origin_x, step),
        snap_ceil(maxy, origin_y, step),
    )


def _wcs_get_coverage_params(
    bbox: tuple[float, float, float, float],
    *,
    coverage_id: str,
    wcs_version: str,
    fmt: str,
    crs: str,
) -> list[tuple[str, str]]:
    epsg = _epsg_code(crs)
    crs_uri = f"http://www.opengis.net/def/crs/EPSG/0/{epsg}"
    minx, miny, maxx, maxy = bbox
    return [
        ("SERVICE", "WCS"),
        ("VERSION", wcs_version),
        ("REQUEST", "GetCoverage"),
        ("COVERAGEID", coverage_id),
        ("FORMAT", fmt),
        ("SUBSET", f"X({minx},{maxx})"),
        ("SUBSET", f"Y({miny},{maxy})"),
        ("SUBSETTINGCRS", crs_uri),
        ("OUTPUTCRS", crs_uri),
    ]


def _bbox_utm_to_ewkt_4326(bbox_utm: tuple[float, float, float, float]) -> str:
    minx, miny, maxx, maxy = bbox_utm
    transformer = transformer_25832_to_4326()
    corners = [
        (minx, miny),
        (maxx, miny),
        (maxx, maxy),
        (minx, maxy),
        (minx, miny),
    ]
    coords = [transformer.transform(x, y) for x, y in corners]
    coord_text = ",".join(f"{lon:.12f} {lat:.12f}" for lon, lat in coords)
    return f"SRID=4326;POLYGON(({coord_text}))"


def _safe_tile_basename(name: str) -> str:
    """Validate a Metalink-supplied filename before using it as a path component.

    The Metalink API is operated by LDBV and not malicious, but the
    ``<file name="...">`` value lands in a filesystem path. Defense-in-depth:
    reject anything that looks like a path (separators, drive letters, dot
    components). Plain filenames pass through unchanged.
    """
    raw = name.strip()
    if not raw:
        raise DemDownloadError("empty tile filename in metalink response")
    if any(sep in raw for sep in ("/", "\\")):
        raise DemDownloadError(
            f"invalid tile filename in metalink response (path separator): {name!r}"
        )
    if raw in {".", ".."}:
        raise DemDownloadError(
            f"invalid tile filename in metalink response (relative path): {name!r}"
        )
    if Path(raw).name != raw:
        # catches NUL, drive letters etc. that Path normalizes away
        raise DemDownloadError(
            f"invalid tile filename in metalink response (normalization): {name!r}"
        )
    return raw


def _parse_metalink_urls(payload: str) -> list[tuple[str, str]]:
    ns = {"m": "urn:ietf:params:xml:ns:metalink"}
    root = ET.fromstring(payload)
    out: list[tuple[str, str]] = []
    for file_el in root.findall("m:file", ns):
        raw_name = file_el.attrib.get("name", "").strip()
        urls = [url_el.text.strip() for url_el in file_el.findall("m:url", ns) if url_el.text]
        if not raw_name and urls:
            raw_name = Path(urlparse(urls[0]).path).name
        if raw_name and urls:
            out.append((_safe_tile_basename(raw_name), urls[0]))
    return out


def _atomic_write_bytes(out_path: Path, payload: bytes) -> None:
    """Write payload to a sibling .part file, then rename.

    Survives SIGKILL between bytes and verify: if the worker dies, the
    target name is either fully present *and* verified, or absent —
    never half-written and tagged as cached.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    try:
        tmp_path.write_bytes(payload)
        _verify_raster_readable(tmp_path)
        tmp_path.replace(out_path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _download_file(session: requests.Session, url: str, out_path: Path) -> None:
    try:
        response = session.get(url, timeout=(10, 120))
        response.raise_for_status()
    except requests.exceptions.Timeout as exc:
        raise DemDownloadError("DEM_TIMEOUT") from exc
    except requests.exceptions.RequestException as exc:
        raise DemDownloadError("DEM_HTTP_ERROR") from exc
    _atomic_write_bytes(out_path, response.content)


def _verify_raster_readable(path: Path) -> None:
    try:
        src = rasterio.open(path)
    except rasterio.errors.RasterioIOError as exc:
        raise DemDownloadError(
            f"DEM_HTTP_ERROR: returned bytes are not a valid GeoTIFF: {exc}"
        ) from exc
    try:
        if src.count < 1:
            raise DemDownloadError(f"DEM raster has unexpected band count={src.count}")
    finally:
        src.close()


def fetch_tiles_via_metalink(
    bbox_utm: tuple[float, float, float, float],
    out_dir: Path,
    *,
    metalink_url: str,
    cache_dir: Path | None = None,
    vrt_name: str = "tiles.vrt",
) -> TileFetchResult:
    """Download OpenData GeoTIFF tiles selected by a polygonal Metalink API."""
    if not metalink_url:
        raise DemDownloadError("DEM client misconfigured: metalink_url must be set")

    session = _build_session()
    ewkt = _bbox_utm_to_ewkt_4326(bbox_utm)
    try:
        # Explicit Content-Type: requests defaults to no Content-Type for
        # str payloads, which makes some servers respond 415 Unsupported
        # Media Type. The LDBV poly2metalink endpoint accepts plain EWKT.
        response = session.post(
            metalink_url,
            data=ewkt.encode("utf-8"),
            headers={"Content-Type": "text/plain; charset=utf-8"},
            timeout=(10, 60),
        )
        response.raise_for_status()
    except requests.exceptions.Timeout as exc:
        raise DemDownloadError("DEM_TIMEOUT") from exc
    except requests.exceptions.RequestException as exc:
        raise DemDownloadError("DEM_HTTP_ERROR") from exc

    tile_specs = _parse_metalink_urls(response.text)
    if not tile_specs:
        raise DemDownloadError("DEM_HTTP_ERROR: metalink response did not contain files")

    target_dir = cache_dir or out_dir
    tile_paths: list[Path] = []
    for name, url in tile_specs:
        tile_path = target_dir / name
        if not tile_path.exists():
            _download_file(session, url, tile_path)
        else:
            _verify_raster_readable(tile_path)
        tile_paths.append(tile_path)

    out_dir.mkdir(parents=True, exist_ok=True)
    vrt_path = out_dir / vrt_name
    _build_single_band_vrt(vrt_path, tile_paths)
    return TileFetchResult(path=vrt_path, tile_paths=tuple(tile_paths), bbox_utm=bbox_utm)


def _vrt_filename(vrt_path: Path, source_path: Path) -> tuple[str, bool]:
    try:
        return (
            source_path.resolve().relative_to(vrt_path.parent.resolve()).as_posix(),
            True,
        )
    except ValueError:
        return source_path.resolve().as_posix(), False


def _add_text(parent: ET.Element, tag: str, text: str, **attrs) -> ET.Element:
    child = ET.SubElement(parent, tag, attrs)
    child.text = text
    return child


def _build_single_band_vrt(vrt_path: Path, tile_paths: list[Path]) -> None:
    if not tile_paths:
        raise DemDownloadError("VRT_BUILD_FAILED: no tiles")

    footprints = []
    for path in tile_paths:
        with rasterio.open(path) as src:
            footprints.append(
                (
                    float(src.bounds.left),
                    float(src.bounds.bottom),
                    float(src.bounds.right),
                    float(src.bounds.top),
                    int(src.width),
                    int(src.height),
                    src.dtypes[0],
                    src.crs,
                )
            )
    dtypes = {item[6] for item in footprints}
    crs_values = {item[7].to_string() if item[7] else "" for item in footprints}
    if len(dtypes) != 1:
        raise DemDownloadError(f"VRT_BUILD_FAILED: heterogeneous dtypes {dtypes}")
    if len(crs_values) != 1:
        raise DemDownloadError(f"VRT_BUILD_FAILED: heterogeneous CRS {crs_values}")

    minx = min(item[0] for item in footprints)
    miny = min(item[1] for item in footprints)
    maxx = max(item[2] for item in footprints)
    maxy = max(item[3] for item in footprints)
    sample = footprints[0]
    step_x = (sample[2] - sample[0]) / sample[4]
    step_y = (sample[3] - sample[1]) / sample[5]
    width = round((maxx - minx) / step_x)
    height = round((maxy - miny) / step_y)

    dtype = dtypes.pop()
    crs_string = crs_values.pop()
    root = ET.Element("VRTDataset", rasterXSize=str(width), rasterYSize=str(height))
    if crs_string:
        _add_text(root, "SRS", CRS.from_string(crs_string).to_wkt())
    _add_text(root, "GeoTransform", f"{minx}, {step_x}, 0.0, {maxy}, 0.0, {-step_y}")
    band = ET.SubElement(root, "VRTRasterBand", dataType=_vrt_dtype(dtype), band="1")
    _add_text(band, "ColorInterp", "Gray")

    for path, footprint in zip(tile_paths, footprints):
        tile_minx, _tile_miny, _tile_maxx, tile_maxy, tile_w, tile_h, _dtype, _crs = footprint
        dst_x = round((tile_minx - minx) / step_x)
        dst_y = round((maxy - tile_maxy) / step_y)
        source_filename, is_relative = _vrt_filename(vrt_path, path)
        simple = ET.SubElement(band, "SimpleSource")
        _add_text(
            simple,
            "SourceFilename",
            source_filename,
            relativeToVRT="1" if is_relative else "0",
        )
        _add_text(simple, "SourceBand", "1")
        ET.SubElement(simple, "SrcRect", xOff="0", yOff="0", xSize=str(tile_w), ySize=str(tile_h))
        ET.SubElement(simple, "DstRect", xOff=str(dst_x), yOff=str(dst_y), xSize=str(tile_w), ySize=str(tile_h))

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(vrt_path, encoding="UTF-8", xml_declaration=True)
    _verify_raster_readable(vrt_path)


def _vrt_dtype(dtype: str) -> str:
    mapping = {
        "uint8": "Byte",
        "uint16": "UInt16",
        "int16": "Int16",
        "uint32": "UInt32",
        "int32": "Int32",
        "float32": "Float32",
        "float64": "Float64",
    }
    return mapping.get(dtype, "Float32")


def fetch_dem(
    bbox_utm: tuple[float, float, float, float],
    out_path: Path,
    *,
    wcs_url: str,
    coverage_id: str,
    wcs_version: str,
    fmt: str,
    crs: str,
    origin_x: float,
    origin_y: float,
    native_step_m: float,
    username: str = "",
    password: str = "",
) -> DemFetchResult:
    """Download a single-band DEM/nDSM raster covering the AOI.

    The output file is the verbatim TIFF the server returned, with bytes
    flushed to disk. Bounds, single-band, and a sane ``transform`` are
    sanity-checked; the file's CRS is intentionally not strictly verified
    (LDBV emits a non-standard ``LOCAL_CS`` block analogous to DOP20).
    """
    if not wcs_url or not coverage_id:
        raise DemDownloadError(
            "DEM client misconfigured: WCS_URL and COVERAGE_ID must be set"
        )

    snapped = _snap_to_native_grid(
        bbox_utm,
        origin_x=origin_x,
        origin_y=origin_y,
        step=native_step_m,
    )
    params = _wcs_get_coverage_params(
        snapped,
        coverage_id=coverage_id,
        wcs_version=wcs_version,
        fmt=fmt,
        crs=crs,
    )
    session = _build_session(username=username, password=password)
    try:
        response = session.get(wcs_url, params=params, timeout=(10, 60))
        response.raise_for_status()
    except requests.exceptions.Timeout as exc:
        raise DemDownloadError("DEM_TIMEOUT") from exc
    except requests.exceptions.RequestException as exc:
        raise DemDownloadError("DEM_HTTP_ERROR") from exc

    content_type = response.headers.get("Content-Type", "")
    if "image/tiff" not in content_type and "image/geotiff" not in content_type:
        raise DemDownloadError(
            f"DEM_HTTP_ERROR: non-tiff response (Content-Type={content_type!r}): "
            f"{response.text[:200]}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(response.content)
    _verify_dem(out_path, snapped)
    return DemFetchResult(path=out_path, bbox_utm=snapped, native_step_m=native_step_m)


def derive_ndsm_from_dom_dgm(
    *,
    dom_path: Path,
    dgm_path: Path,
    out_path: Path,
    output_crs: str = "EPSG:25832",
) -> NdsmDeriveResult:
    """Create a local nDSM raster as ``DOM - DGM`` in metres.

    The DOM grid is used as the output grid. DGM is bilinearly resampled onto
    that grid before subtraction, so DOM and DGM may have different native
    resolutions or origins.

    ``output_crs`` is written into the GeoTIFF unconditionally. This guards
    against LDBV emitting a non-standard ``LOCAL_CS["ETRS89 / UTM zone 32N",
    ...]`` block on the source tiles (we observed that on the DOP20 WCS); we
    know all LDBV elevation services are EPSG:25832 by datasheet, so pinning
    the CRS at write time keeps the downstream tiler from inheriting an
    unresolvable CRS.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    target_crs = CRS.from_string(output_crs)
    with rasterio.open(dom_path) as dom_src, rasterio.open(dgm_path) as dgm_src:
        dom = dom_src.read(1, masked=True).astype(np.float32)
        dgm_on_dom = np.full(dom.shape, np.nan, dtype=np.float32)
        # For reproject(), prefer the CRS we actually know is right over
        # whatever non-standard string the source TIFF carries.
        src_crs_dgm = dgm_src.crs if dgm_src.crs else target_crs
        dst_crs_dom = dom_src.crs if dom_src.crs else target_crs
        reproject(
            source=rasterio.band(dgm_src, 1),
            destination=dgm_on_dom,
            src_transform=dgm_src.transform,
            src_crs=src_crs_dgm,
            src_nodata=dgm_src.nodata,
            dst_transform=dom_src.transform,
            dst_crs=dst_crs_dom,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
        ndsm = dom.filled(np.nan).astype(np.float32, copy=False) - dgm_on_dom
        ndsm = np.where(np.isfinite(ndsm), ndsm, np.nan).astype(np.float32)
        profile = dom_src.profile.copy()
        profile.update(
            driver="GTiff",
            count=1,
            dtype="float32",
            nodata=np.nan,
            crs=target_crs,
        )

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(ndsm, 1)
    _verify_dem(out_path, _bounds_tuple(out_path))
    return NdsmDeriveResult(path=out_path, dom_path=dom_path, dgm_path=dgm_path)


def _bounds_tuple(path: Path) -> tuple[float, float, float, float]:
    with rasterio.open(path) as src:
        return (
            float(src.bounds.left),
            float(src.bounds.bottom),
            float(src.bounds.right),
            float(src.bounds.top),
        )


def _verify_dem(path: Path, bbox: tuple[float, float, float, float]) -> None:
    """Open the GeoTIFF, sanity-check bounds and band count.

    DEM coverages should be single-band float (or int). We do not enforce
    the dtype here because some services ship int16 metres × 100 etc.; the
    consumer (tiler) handles dtype coercion explicitly.
    """
    try:
        src = rasterio.open(path)
    except rasterio.errors.RasterioIOError as exc:
        raise DemDownloadError(
            f"DEM_HTTP_ERROR: returned bytes are not a valid GeoTIFF: {exc}"
        ) from exc
    try:
        if src.count < 1:
            raise DemDownloadError(
                f"DEM coverage has unexpected band count={src.count}"
            )
        for actual, expected, label in (
            (float(src.bounds.left), bbox[0], "minx"),
            (float(src.bounds.bottom), bbox[1], "miny"),
            (float(src.bounds.right), bbox[2], "maxx"),
            (float(src.bounds.top), bbox[3], "maxy"),
        ):
            if abs(actual - expected) > 1.0:
                # 1 m tolerance — DEM grids are often 1 m, server may snap.
                raise DemDownloadError(
                    f"DEM bounds.{label}={actual} mismatch (expected {expected})"
                )
    finally:
        src.close()
