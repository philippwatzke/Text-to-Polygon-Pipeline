from __future__ import annotations

import io
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import requests
from PIL import Image
from ki_geodaten.models import TilePreset
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from requests.adapters import HTTPAdapter
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
    """Transport or mosaic error from the DOP20 WMS client."""


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


def _build_session() -> requests.Session:
    session = requests.Session()
    # Local development shells often carry proxy variables from other tools.
    # The LDBV WMS is public and should be reached directly; a stale proxy
    # turns otherwise valid live jobs into opaque DOP_HTTP_ERROR failures.
    session.trust_env = False
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


def _png_bytes_to_array(payload: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(payload)) as image:
        if image.mode != "RGBA":
            raise DopDownloadError(f"Expected RGBA PNG from WMS, got mode={image.mode!r}")
        return np.asarray(image, dtype=np.uint8).copy()


def _write_geotiff(
    out_path: Path,
    arr: np.ndarray,
    bbox: tuple[float, float, float, float],
    crs: str,
) -> None:
    height, width, bands = arr.shape
    if bands != 4:
        raise DopDownloadError(f"Expected RGBA from WMS, got {bands} bands")

    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=4,
        dtype="uint8",
        crs=crs,
        transform=from_bounds(*bbox, width, height),
        photometric="RGB",
    ) as dst:
        for band_idx in range(4):
            dst.write(arr[..., band_idx], band_idx + 1)
        dst.colorinterp = (
            rasterio.enums.ColorInterp.red,
            rasterio.enums.ColorInterp.green,
            rasterio.enums.ColorInterp.blue,
            rasterio.enums.ColorInterp.alpha,
        )


def _fetch_chunk(
    session: requests.Session,
    wms_url: str,
    layer: str,
    chunk: ChunkPlan,
    out_path: Path,
    *,
    wms_version: str,
    fmt: str,
    crs: str,
) -> None:
    crs_param = "SRS" if wms_version.startswith("1.1") else "CRS"
    params = {
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

    try:
        response = session.get(wms_url, params=params, timeout=(10, 60))
        response.raise_for_status()
    except requests.exceptions.Timeout as exc:
        raise DopDownloadError("DOP_TIMEOUT") from exc
    except requests.exceptions.RequestException as exc:
        raise DopDownloadError("DOP_HTTP_ERROR") from exc

    if not response.headers.get("Content-Type", "").startswith("image/"):
        raise DopDownloadError(f"DOP_HTTP_ERROR: non-image response {response.text[:200]}")

    arr = _png_bytes_to_array(response.content)
    _write_geotiff(out_path, arr, chunk.bbox(), crs)


def _chunk_footprint(path: Path) -> tuple[float, float, float, float, int, int]:
    with rasterio.open(path) as src:
        if src.count != 4:
            raise DopDownloadError(f"Expected 4-band chunk for VRT, got {src.count}")
        return (
            float(src.bounds.left),
            float(src.bounds.bottom),
            float(src.bounds.right),
            float(src.bounds.top),
            int(src.width),
            int(src.height),
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
    minx = min(item[0] for item in footprints)
    miny = min(item[1] for item in footprints)
    maxx = max(item[2] for item in footprints)
    maxy = max(item[3] for item in footprints)
    width = pixel_count(minx, maxx, step)
    height = pixel_count(miny, maxy, step)

    root = ET.Element("VRTDataset", rasterXSize=str(width), rasterYSize=str(height))
    _add_text(root, "SRS", CRS.from_string(crs).to_wkt())
    _add_text(root, "GeoTransform", f"{minx}, {step}, 0.0, {maxy}, 0.0, {-step}")

    color_names = ("Red", "Green", "Blue", "Alpha")
    for band_idx, color_name in enumerate(color_names, start=1):
        band = ET.SubElement(
            root,
            "VRTRasterBand",
            dataType="Byte",
            band=str(band_idx),
        )
        _add_text(band, "ColorInterp", color_name)
        for path, footprint in zip(chunk_files, footprints):
            chunk_minx, _chunk_miny, _chunk_maxx, chunk_maxy, chunk_w, chunk_h = footprint
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
    wms_url: str,
    layer: str,
    wms_version: str,
    fmt: str,
    crs: str,
    max_pixels: int,
    origin_x: float,
    origin_y: float,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks = plan_chunk_grid(
        *bbox_utm,
        max_pixels=max_pixels,
        step=STEP,
        origin_x=origin_x,
        origin_y=origin_y,
    )
    session = _build_session()
    chunk_files: list[Path] = []
    for chunk in chunks:
        path = out_dir / f"chunk_{chunk.row}_{chunk.col}.tif"
        _fetch_chunk(
            session,
            wms_url,
            layer,
            chunk,
            path,
            wms_version=wms_version,
            fmt=fmt,
            crs=crs,
        )
        chunk_files.append(path)

    vrt_path = out_dir / "out.vrt"
    _build_vrt(vrt_path, chunk_files, crs=crs)
    return vrt_path
