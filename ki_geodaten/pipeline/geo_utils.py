from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING
from functools import cache
import pyproj

@cache
def _transformer_4326_to_25832() -> pyproj.Transformer:
    return pyproj.Transformer.from_crs(4326, 25832, always_xy=True)

@cache
def _transformer_25832_to_4326() -> pyproj.Transformer:
    return pyproj.Transformer.from_crs(25832, 4326, always_xy=True)

def transformer_25832_to_4326() -> pyproj.Transformer:
    return _transformer_25832_to_4326()

def transformer_4326_to_25832() -> pyproj.Transformer:
    return _transformer_4326_to_25832()

def snap_floor(x: float, origin: float, step: float = 0.2) -> float:
    d_origin = Decimal(str(origin))
    d_step = Decimal(str(step))
    d_x = Decimal(str(x))
    units = ((d_x - d_origin) / d_step).to_integral_value(rounding=ROUND_FLOOR)
    return float(units * d_step + d_origin)

def snap_ceil(x: float, origin: float, step: float = 0.2) -> float:
    d_origin = Decimal(str(origin))
    d_step = Decimal(str(step))
    d_x = Decimal(str(x))
    units = ((d_x - d_origin) / d_step).to_integral_value(rounding=ROUND_CEILING)
    return float(units * d_step + d_origin)

def transform_bbox_wgs84_to_utm(
    lon_min: float, lat_min: float, lon_max: float, lat_max: float,
    densify_pts: int = 21,
) -> tuple[float, float, float, float]:
    t = _transformer_4326_to_25832()
    return t.transform_bounds(lon_min, lat_min, lon_max, lat_max, densify_pts=densify_pts)

def transform_bbox_utm_to_wgs84(
    minx: float, miny: float, maxx: float, maxy: float,
    densify_pts: int = 21,
) -> tuple[float, float, float, float]:
    t = _transformer_25832_to_4326()
    return t.transform_bounds(minx, miny, maxx, maxy, densify_pts=densify_pts)

def pixel_count(minx: float, maxx: float, step: float = 0.2) -> int:
    """Robust against IEEE-754 FP: uses round() per Spec Section 5.1 pt 6."""
    return round((maxx - minx) / step)
