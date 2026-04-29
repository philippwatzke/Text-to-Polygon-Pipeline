"""DOP20 acquisition via WCS 2.0.1 (Spec §5.1, post-2026-04-28 revision).

Why WCS instead of WMS:
- WMS is a *rendering* protocol — server returns a paint, client receives PNG.
  The pixels go through a server-side rendering pipeline (potential resampling,
  histogram tweaks, anti-aliasing) before they ever land on disk. That's
  fine for human visualization but not for a pipeline whose entire premise
  is "feed the model the rawest possible coverage".
- WCS is a *coverage* protocol — server returns the underlying raster as
  GeoTIFF in its native datatype, no rendering. Same byte values you'd
  get if you opened the LDBV's source files directly.
- Future modalities (nDSM/DGM as float32, CIR-DOP as multi-band, hyperspectral
  as 16-bit) are structurally only available via WCS. Keeping a single
  acquisition protocol simplifies the multi-modality extension later.

The WMS-based implementation lived here previously and is preserved in git
history. See docs/wcs-verification.md for the empirical verification of the
LDBV WCS at native 0.2 m resolution.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from io import BytesIO
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import requests
from PIL import Image
from ki_geodaten.models import TilePreset
from rasterio.crs import CRS
from rasterio.enums import ColorInterp
from rasterio.transform import from_bounds
from requests.adapters import HTTPAdapter
from requests.auth import HTTPBasicAuth
from urllib3.util.retry import Retry

from ki_geodaten.pipeline.geo_utils import pixel_count, snap_ceil, snap_floor

STEP = 0.2
TILE_SIZE = 1024
MIN_BBOX_SIDE_M = TILE_SIZE * STEP

CENTER_MARGIN_PX: dict[TilePreset, int] = {
    TilePreset.SMALL: 160,
    TilePreset.MEDIUM: 320,
    TilePreset.LARGE: 480,
}


class DopDownloadError(Exception):
    """Transport or mosaic error from the DOP20 WCS client."""


@dataclass(frozen=True)
class PreparedBBox:
    aoi_bbox: tuple[float, float, float, float]
    download_bbox: tuple[float, float, float, float]


def _snap_bbox(
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    origin_x: float,
    origin_y: float,
    step: float,
) -> tuple[float, float, float, float]:
    return (
        snap_floor(minx, origin_x, step),
        snap_floor(miny, origin_y, step),
        snap_ceil(maxx, origin_x, step),
        snap_ceil(maxy, origin_y, step),
    )


def _expand_symmetric(
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    target_side: float,
    origin_x: float,
    origin_y: float,
    step: float,
) -> tuple[float, float, float, float]:
    dx = maxx - minx
    dy = maxy - miny
    if dx < target_side:
        extra = (target_side - dx) / 2
        minx = snap_floor(minx - extra, origin_x, step)
        maxx = snap_ceil(maxx + extra, origin_x, step)
    if dy < target_side:
        extra = (target_side - dy) / 2
        miny = snap_floor(miny - extra, origin_y, step)
        maxy = snap_ceil(maxy + extra, origin_y, step)
    return (minx, miny, maxx, maxy)


def prepare_download_bbox(
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    *,
    preset: TilePreset,
    origin_x: float,
    origin_y: float,
    step: float = STEP,
) -> PreparedBBox:
    aoi = _snap_bbox(minx, miny, maxx, maxy, origin_x, origin_y, step)
    margin_m = CENTER_MARGIN_PX[preset] * step

    dmin_x = snap_floor(aoi[0] - margin_m, origin_x, step)
    dmin_y = snap_floor(aoi[1] - margin_m, origin_y, step)
    dmax_x = snap_ceil(aoi[2] + margin_m, origin_x, step)
    dmax_y = snap_ceil(aoi[3] + margin_m, origin_y, step)
    download = _expand_symmetric(
        dmin_x,
        dmin_y,
        dmax_x,
        dmax_y,
        MIN_BBOX_SIDE_M,
        origin_x,
        origin_y,
        step,
    )

    return PreparedBBox(aoi_bbox=aoi, download_bbox=download)


@dataclass(frozen=True)
class ChunkPlan:
    row: int
    col: int
    minx: float
    miny: float
    maxx: float
    maxy: float

    def width_px(self, step: float = STEP) -> int:
        return pixel_count(self.minx, self.maxx, step)

    def height_px(self, step: float = STEP) -> int:
        return pixel_count(self.miny, self.maxy, step)

    def bbox(self) -> tuple[float, float, float, float]:
        return (self.minx, self.miny, self.maxx, self.maxy)


def plan_chunk_grid(
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    *,
    max_pixels: int,
    step: float,
    origin_x: float,
    origin_y: float,
) -> list[ChunkPlan]:
    chunk_side_m = max_pixels * step
    chunks: list[ChunkPlan] = []

    row = 0
    y0 = miny
    while y0 < maxy:
        y1 = min(snap_ceil(y0 + chunk_side_m, origin_y, step), maxy)
        col = 0
        x0 = minx
        while x0 < maxx:
            x1 = min(snap_ceil(x0 + chunk_side_m, origin_x, step), maxx)
            chunks.append(
                ChunkPlan(row=row, col=col, minx=x0, miny=y0, maxx=x1, maxy=y1)
            )
            x0 = x1
            col += 1
        y0 = y1
        row += 1

    return chunks


def _build_session(username: str = "", password: str = "") -> requests.Session:
    session = requests.Session()
    # Local development shells often carry proxy variables from other tools.
    # The LDBV endpoints are reachable directly; a stale proxy turns valid
    # live jobs into opaque DOP_HTTP_ERROR failures.
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
    """Extract the bare EPSG number from 'EPSG:25832' for WCS URI form."""
    if ":" in crs:
        return crs.split(":", 1)[1]
    return crs


def _wcs_get_coverage_params(
    chunk: ChunkPlan,
    *,
    coverage_id: str,
    wcs_version: str,
    fmt: str,
    crs: str,
) -> list[tuple[str, str]]:
    """Build WCS 2.0.1 GetCoverage params as ordered list (subset is repeated)."""
    epsg = _epsg_code(crs)
    crs_uri = f"http://www.opengis.net/def/crs/EPSG/0/{epsg}"
    return [
        ("SERVICE", "WCS"),
        ("VERSION", wcs_version),
        ("REQUEST", "GetCoverage"),
        ("COVERAGEID", coverage_id),
        ("FORMAT", fmt),
        ("SUBSET", f"X({chunk.minx},{chunk.maxx})"),
        ("SUBSET", f"Y({chunk.miny},{chunk.maxy})"),
        ("SUBSETTINGCRS", crs_uri),
        ("OUTPUTCRS", crs_uri),
    ]


def _fetch_chunk(
    session: requests.Session,
    wcs_url: str,
    coverage_id: str,
    chunk: ChunkPlan,
    out_path: Path,
    *,
    wcs_version: str,
    fmt: str,
    crs: str,
    fill_rgb_zero_with_wms: bool = False,
    wms_url: str = "",
    wms_layer: str = "",
    wms_version: str = "1.1.1",
    wms_format: str = "image/png",
    source: str = "wcs",
) -> None:
    source = source.lower().strip()
    if source == "wms":
        _fetch_wms_chunk(
            session,
            chunk,
            out_path,
            wms_url=wms_url,
            layer=wms_layer,
            wms_version=wms_version,
            fmt=wms_format,
            crs=crs,
        )
        return
    if source != "wcs":
        raise DopDownloadError(f"DOP_HTTP_ERROR: unsupported DOP source {source!r}")

    params = _wcs_get_coverage_params(
        chunk,
        coverage_id=coverage_id,
        wcs_version=wcs_version,
        fmt=fmt,
        crs=crs,
    )

    try:
        response = session.get(wcs_url, params=params, timeout=(10, 60))
        response.raise_for_status()
    except requests.exceptions.Timeout as exc:
        raise DopDownloadError("DOP_TIMEOUT") from exc
    except requests.exceptions.RequestException as exc:
        raise DopDownloadError("DOP_HTTP_ERROR") from exc

    content_type = response.headers.get("Content-Type", "")
    if "image/tiff" not in content_type and "image/geotiff" not in content_type:
        # WCS errors come back as text/xml ows:ExceptionReport; surface that.
        raise DopDownloadError(
            f"DOP_HTTP_ERROR: non-tiff response (Content-Type={content_type!r}): "
            f"{response.text[:200]}"
        )

    out_path.write_bytes(response.content)
    _verify_chunk(out_path, chunk, crs)
    if fill_rgb_zero_with_wms:
        _fill_rgb_zero_pixels_from_wms(
            session,
            chunk,
            out_path,
            wms_url=wms_url,
            layer=wms_layer,
            wms_version=wms_version,
            fmt=wms_format,
            crs=crs,
        )


def _verify_chunk(path: Path, chunk: ChunkPlan, crs: str) -> None:
    """Open the GeoTIFF the server returned and sanity-check georef + size.

    We deliberately do **not** verify the chunk's CRS string. The LDBV WCS
    emits a non-standard `LOCAL_CS["ETRS89 / UTM zone 32N", ...]` block
    that rasterio handles inconsistently — `linear_units` is sometimes
    `'metre'`, sometimes `'unknown'`, and `to_epsg()` is always `None`.
    The materially correct check is "does the affine transform place pixels
    where we asked for them?" — i.e. the bounds match the requested subset.
    The canonical `EPSG:25832` is set explicitly in the VRT we build on top.
    """
    try:
        src = rasterio.open(path)
    except rasterio.errors.RasterioIOError as exc:
        raise DopDownloadError(
            f"DOP_HTTP_ERROR: returned bytes are not a valid GeoTIFF: {exc}"
        ) from exc
    try:
        if src.count not in (3, 4):
            raise DopDownloadError(
                f"WCS chunk has unexpected band count={src.count} (expected 3 or 4)"
            )
        expected_w = chunk.width_px()
        expected_h = chunk.height_px()
        if src.width != expected_w or src.height != expected_h:
            raise DopDownloadError(
                f"WCS chunk size mismatch: server={src.width}x{src.height} "
                f"expected={expected_w}x{expected_h}"
            )
        for actual, expected, label in (
            (float(src.bounds.left), chunk.minx, "minx"),
            (float(src.bounds.bottom), chunk.miny, "miny"),
            (float(src.bounds.right), chunk.maxx, "maxx"),
            (float(src.bounds.top), chunk.maxy, "maxy"),
        ):
            if abs(actual - expected) > 1e-3:
                raise DopDownloadError(
                    f"WCS chunk bounds.{label}={actual} mismatch (expected {expected})"
                )
    finally:
        src.close()
    _ = crs  # explicit-CRS handling happens in the VRT layer


def _wms_get_map_params(
    chunk: ChunkPlan,
    *,
    layer: str,
    wms_version: str,
    fmt: str,
    crs: str,
) -> dict[str, str]:
    crs_param = "SRS" if wms_version.startswith("1.1") else "CRS"
    return {
        "SERVICE": "WMS",
        "VERSION": wms_version,
        "REQUEST": "GetMap",
        "LAYERS": layer,
        "STYLES": "",
        crs_param: crs,
        "BBOX": f"{chunk.minx},{chunk.miny},{chunk.maxx},{chunk.maxy}",
        "WIDTH": str(chunk.width_px()),
        "HEIGHT": str(chunk.height_px()),
        "FORMAT": fmt,
        "TRANSPARENT": "TRUE",
    }


def _fetch_wms_rgb(
    session: requests.Session,
    chunk: ChunkPlan,
    *,
    wms_url: str,
    layer: str,
    wms_version: str,
    fmt: str,
    crs: str,
) -> np.ndarray:
    if not wms_url or not layer:
        raise DopDownloadError("DOP_HTTP_ERROR: WMS fallback is enabled but not configured")
    params = _wms_get_map_params(
        chunk,
        layer=layer,
        wms_version=wms_version,
        fmt=fmt,
        crs=crs,
    )
    wms_session = _build_session()
    try:
        response = wms_session.get(wms_url, params=params, timeout=(10, 60))
        response.raise_for_status()
    except requests.exceptions.Timeout as exc:
        raise DopDownloadError("DOP_TIMEOUT") from exc
    except requests.exceptions.RequestException as exc:
        raise DopDownloadError("DOP_HTTP_ERROR") from exc

    if not response.headers.get("Content-Type", "").startswith("image/"):
        raise DopDownloadError(
            f"DOP_HTTP_ERROR: WMS fallback non-image response {response.text[:200]}"
        )
    with Image.open(BytesIO(response.content)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def _fetch_wms_chunk(
    session: requests.Session,
    chunk: ChunkPlan,
    out_path: Path,
    *,
    wms_url: str,
    layer: str,
    wms_version: str,
    fmt: str,
    crs: str,
) -> None:
    rgb = _fetch_wms_rgb(
        session,
        chunk,
        wms_url=wms_url,
        layer=layer,
        wms_version=wms_version,
        fmt=fmt,
        crs=crs,
    )
    expected_shape = (chunk.height_px(), chunk.width_px())
    if rgb.shape[:2] != expected_shape:
        raise DopDownloadError(
            f"DOP_HTTP_ERROR: WMS chunk size mismatch {rgb.shape[:2]} != {expected_shape}"
        )

    profile = {
        "driver": "GTiff",
        "height": rgb.shape[0],
        "width": rgb.shape[1],
        "count": 3,
        "dtype": "uint8",
        "crs": CRS.from_string(crs),
        "transform": from_bounds(
            chunk.minx,
            chunk.miny,
            chunk.maxx,
            chunk.maxy,
            rgb.shape[1],
            rgb.shape[0],
        ),
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(np.moveaxis(rgb, -1, 0))
        dst.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)


def _fill_rgb_zero_pixels_from_wms(
    session: requests.Session,
    chunk: ChunkPlan,
    path: Path,
    *,
    wms_url: str,
    layer: str,
    wms_version: str,
    fmt: str,
    crs: str,
) -> None:
    with rasterio.open(path) as src:
        rgb = src.read([1, 2, 3])
        zero_mask = np.all(rgb == 0, axis=0)
        if not bool(zero_mask.any()):
            return
        profile = src.profile.copy()
        bands = src.read()
        colorinterp = src.colorinterp
        tags = src.tags()

    wms_rgb = _fetch_wms_rgb(
        session,
        chunk,
        wms_url=wms_url,
        layer=layer,
        wms_version=wms_version,
        fmt=fmt,
        crs=crs,
    )
    if wms_rgb.shape[:2] != zero_mask.shape:
        raise DopDownloadError(
            "DOP_HTTP_ERROR: WMS fallback size mismatch "
            f"{wms_rgb.shape[:2]} != {zero_mask.shape}"
        )

    for band_idx in range(3):
        band = bands[band_idx]
        band[zero_mask] = wms_rgb[..., band_idx][zero_mask]
        bands[band_idx] = band

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with rasterio.open(tmp_path, "w", **profile) as dst:
        dst.write(bands)
        dst.colorinterp = colorinterp
        dst.update_tags(**tags)
    tmp_path.replace(path)


def _chunk_footprint(path: Path) -> tuple[float, float, float, float, int, int, int]:
    with rasterio.open(path) as src:
        if src.count not in (3, 4):
            raise DopDownloadError(f"Unexpected band count for VRT: {src.count}")
        return (
            float(src.bounds.left),
            float(src.bounds.bottom),
            float(src.bounds.right),
            float(src.bounds.top),
            int(src.width),
            int(src.height),
            int(src.count),
        )


def _relative_vrt_filename(vrt_path: Path, source_path: Path) -> str:
    try:
        return source_path.resolve().relative_to(vrt_path.parent.resolve()).as_posix()
    except ValueError:
        return source_path.resolve().as_posix()


def _add_text(parent: ET.Element, tag: str, text: str, **attrs) -> ET.Element:
    child = ET.SubElement(parent, tag, attrs)
    child.text = text
    return child


def _build_vrt(vrt_path: Path, chunk_files: list[Path], *, crs: str, step: float = STEP) -> None:
    if not chunk_files:
        raise DopDownloadError("VRT_BUILD_FAILED: no chunks")

    footprints = [_chunk_footprint(path) for path in chunk_files]
    band_counts = {fp[6] for fp in footprints}
    if len(band_counts) != 1:
        raise DopDownloadError(
            f"VRT_BUILD_FAILED: heterogeneous band counts across chunks: {band_counts}"
        )
    band_count = band_counts.pop()
    minx = min(item[0] for item in footprints)
    miny = min(item[1] for item in footprints)
    maxx = max(item[2] for item in footprints)
    maxy = max(item[3] for item in footprints)
    width = pixel_count(minx, maxx, step)
    height = pixel_count(miny, maxy, step)

    root = ET.Element("VRTDataset", rasterXSize=str(width), rasterYSize=str(height))
    _add_text(root, "SRS", CRS.from_string(crs).to_wkt())
    _add_text(root, "GeoTransform", f"{minx}, {step}, 0.0, {maxy}, 0.0, {-step}")

    color_names = (
        ("Red", "Green", "Blue")
        if band_count == 3
        else ("Red", "Green", "Blue", "Undefined")
    )
    for band_idx, color_name in enumerate(color_names, start=1):
        band = ET.SubElement(
            root,
            "VRTRasterBand",
            dataType="Byte",
            band=str(band_idx),
        )
        _add_text(band, "ColorInterp", color_name)
        for path, footprint in zip(chunk_files, footprints):
            chunk_minx, _chunk_miny, _chunk_maxx, chunk_maxy, chunk_w, chunk_h, _bands = footprint
            dst_x = pixel_count(minx, chunk_minx, step)
            dst_y = pixel_count(chunk_maxy, maxy, step)
            simple = ET.SubElement(band, "SimpleSource")
            _add_text(
                simple,
                "SourceFilename",
                _relative_vrt_filename(vrt_path, path),
                relativeToVRT="1",
            )
            _add_text(simple, "SourceBand", str(band_idx))
            ET.SubElement(
                simple,
                "SrcRect",
                xOff="0",
                yOff="0",
                xSize=str(chunk_w),
                ySize=str(chunk_h),
            )
            ET.SubElement(
                simple,
                "DstRect",
                xOff=str(dst_x),
                yOff=str(dst_y),
                xSize=str(chunk_w),
                ySize=str(chunk_h),
            )

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(vrt_path, encoding="UTF-8", xml_declaration=True)

    try:
        with rasterio.open(vrt_path):
            pass
    except Exception as exc:  # noqa: BLE001
        raise DopDownloadError("VRT_BUILD_FAILED") from exc


def download_dop20(
    bbox_utm: tuple[float, float, float, float],
    out_dir: Path,
    *,
    wcs_url: str,
    coverage_id: str,
    wcs_version: str,
    fmt: str,
    crs: str,
    max_pixels: int,
    origin_x: float,
    origin_y: float,
    username: str = "",
    password: str = "",
    fill_rgb_zero_with_wms: bool = False,
    wms_url: str = "",
    wms_layer: str = "",
    wms_version: str = "1.1.1",
    wms_format: str = "image/png",
    source: str = "wcs",
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks = plan_chunk_grid(
        *bbox_utm,
        max_pixels=max_pixels,
        step=STEP,
        origin_x=origin_x,
        origin_y=origin_y,
    )
    session = _build_session(username=username, password=password)
    chunk_files: list[Path] = []
    for chunk in chunks:
        path = out_dir / f"chunk_{chunk.row}_{chunk.col}.tif"
        _fetch_chunk(
            session,
            wcs_url,
            coverage_id,
            chunk,
            path,
            wcs_version=wcs_version,
            fmt=fmt,
            crs=crs,
            fill_rgb_zero_with_wms=fill_rgb_zero_with_wms,
            wms_url=wms_url,
            wms_layer=wms_layer,
            wms_version=wms_version,
            wms_format=wms_format,
            source=source,
        )
        chunk_files.append(path)

    vrt_path = out_dir / "out.vrt"
    _build_vrt(vrt_path, chunk_files, crs=crs)
    return vrt_path
