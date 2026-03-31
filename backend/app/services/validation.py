"""
Validate an incoming AOI and date range before spending time fetching imagery
for it. Two independent checks: the geometry is a real, sane polygon, and the
date range is one Sentinel-2 could plausibly have covered.
"""

from datetime import date, timedelta
from math import cos, radians

from fastapi import HTTPException
from shapely.geometry import Polygon, shape
from shapely.geometry.base import BaseGeometry

MIN_AREA_HECTARES = 0.1
MAX_AREA_HECTARES = 500
MAX_DATE_RANGE_DAYS = 400


def validate_aoi(geojson: dict) -> Polygon:
    try:
        geometry: BaseGeometry = shape(geojson)
    except Exception as exc:
        raise HTTPException(400, f"Invalid AOI geometry: {exc}") from exc

    if not isinstance(geometry, Polygon):
        raise HTTPException(400, "AOI must be a single Polygon")
    if not geometry.is_valid:
        raise HTTPException(400, "AOI polygon is self-intersecting or otherwise invalid")

    area_hectares = _approx_area_hectares(geometry)
    if not (MIN_AREA_HECTARES <= area_hectares <= MAX_AREA_HECTARES):
        raise HTTPException(
            400,
            f"AOI area ({area_hectares:.2f} ha) is outside the supported range "
            f"({MIN_AREA_HECTARES}-{MAX_AREA_HECTARES} ha)",
        )
    return geometry


def validate_date_range(start_date: date, end_date: date) -> None:
    if end_date > date.today():
        raise HTTPException(400, "end_date cannot be in the future")
    if (end_date - start_date) > timedelta(days=MAX_DATE_RANGE_DAYS):
        raise HTTPException(400, f"date range cannot exceed {MAX_DATE_RANGE_DAYS} days")


def _approx_area_hectares(polygon: Polygon) -> float:
    """
    Rough WGS84 degrees -> hectares conversion, latitude-corrected. Good enough
    for a sanity bound, not precise enough for real area statistics.
    """
    lat = polygon.centroid.y
    meters_per_degree_lon = 111_320 * cos(radians(lat))
    meters_per_degree_lat = 110_540
    return polygon.area * meters_per_degree_lon * meters_per_degree_lat / 10_000
