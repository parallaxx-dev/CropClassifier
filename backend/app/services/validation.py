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

# Every checkpoint has only ever been trained on full Jan1-Dec31 sequences
# (see CLAUDE.md's known-gaps list, item 4). Measured directly, not assumed:
# training/eurocrops_pipeline/eval_partial_range.py replayed the held-out
# test set through the production preprocess/predict path with only the
# first 196 days of each parcel's real observations kept (Jan1-Jul16, the
# same span a user actually hit this gap with) -- ensembled accuracy fell
# from 73.5% (full year) to 57.7% (196-day window), a 15.8-point drop on a
# 1696-parcel sample, not noise on one field. A `--windowing` augmentation
# retrain was tried as a fix and made both numbers *worse* (see progress.md)
# -- there is currently no model-side fix, only this disclosure.
FULL_SEASON_DAYS = 365
PARTIAL_RANGE_WARNING_THRESHOLD_DAYS = 300


def validate_aoi(geojson: dict) -> Polygon:
    try:
        geometry: BaseGeometry = shape(geojson)
    except Exception as exc:
        raise HTTPException(400, f"Invalid AOI geometry: {exc}") from exc

    if not isinstance(geometry, Polygon):
        raise HTTPException(400, "AOI must be a single Polygon")
    if not geometry.is_valid:
        raise HTTPException(400, "AOI polygon is self-intersecting or otherwise invalid")

    area_hectares = approx_area_hectares(geometry)
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


def partial_range_warning(start_date: date, end_date: date) -> str | None:
    """None if the range is close enough to a full growing-season year to
    match training; otherwise a message quantifying the measured accuracy
    cost (see PARTIAL_RANGE_WARNING_THRESHOLD_DAYS above) -- a real trust
    signal instead of a silent, confidently-wrong prediction on an input
    shape the model has never been evaluated on."""
    days = (end_date - start_date).days
    if days >= PARTIAL_RANGE_WARNING_THRESHOLD_DAYS:
        return None
    return (
        f"This date range covers {days} days; every checkpoint has only ever been "
        f"trained on full {FULL_SEASON_DAYS}-day (Jan1-Dec31) sequences. Measured "
        f"impact on held-out data: accuracy drops from 73.5% (full year) to 57.7% "
        f"on a 196-day range -- treat this prediction as low-confidence and widen "
        f"the date range toward a full year if possible."
    )


def approx_area_hectares(polygon: Polygon) -> float:
    """
    Rough WGS84 degrees -> hectares conversion, latitude-corrected. Good enough
    for a sanity bound, not precise enough for real area statistics.
    """
    lat = polygon.centroid.y
    meters_per_degree_lon = 111_320 * cos(radians(lat))
    meters_per_degree_lat = 110_540
    return polygon.area * meters_per_degree_lon * meters_per_degree_lat / 10_000
