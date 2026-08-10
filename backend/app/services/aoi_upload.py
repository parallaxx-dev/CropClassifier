"""
aoi_upload.py — turn an arbitrary uploaded geospatial file into a single AOI
polygon for the existing /predict pipeline (see routers/upload.py).

Deliberately does NOT introduce a new inference pathway -- the deployed
model takes a field-boundary polygon + date range (see CLAUDE.md's Core
Pipeline), not an image, regardless of what format the boundary came from.
So this module's only job is "arbitrary file in, GeoJSON Polygon out"; the
frontend then feeds that polygon into the same date-range + Run Prediction
flow DrawAOI.jsx already has for a hand-drawn AOI.

Two source shapes, tried in order:
1. Vector (KML, GeoPackage, GeoJSON, zipped Shapefile, and anything else
   GDAL/OGR reads via geopandas) -- same geometry-cleaning logic
   custom_parcels.py's load_vector_file() already uses (repair invalid,
   collapse Multi* to the largest part, reproject last), reused via
   largest_polygon_part() rather than re-implemented. If the file has
   multiple features, the largest-area one is used -- a single AOI is
   the whole point of this endpoint, not a batch of them.
2. Raster (GeoTIFF and other GDAL raster formats) -- there's no "the crop
   boundary" inside an arbitrary raster, so the AOI is the raster's own
   georeferenced footprint (its bounds, reprojected to WGS84).
"""

import tempfile
from pathlib import Path

import geopandas as gpd
import rasterio
from fastapi import HTTPException
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from app.services.custom_parcels import largest_polygon_part
from app.services.validation import approx_area_hectares

MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25MB -- generous for a single field boundary or a small raster,
                                       # small enough that an accidental/malicious huge upload doesn't
                                       # tie up the process (no async job queue exists yet to offload this to)


def _vector_polygon(path: Path) -> BaseGeometry | None:
    gdf = gpd.read_file(path)
    if len(gdf) == 0:
        return None

    invalid = ~gdf.geometry.is_valid
    if invalid.any():
        gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].buffer(0)
    gdf["geometry"] = gdf.geometry.apply(largest_polygon_part)
    gdf = gdf[gdf.geometry.notna()]
    if len(gdf) == 0:
        return None

    gdf = gdf.to_crs(4326)
    # Multiple features (e.g. a multi-field export) -> the largest one is
    # "the AOI", not a batch -- this endpoint returns a single polygon.
    largest_row = gdf.loc[gdf.geometry.area.idxmax()]
    return largest_row.geometry


def _raster_footprint(path: Path) -> BaseGeometry | None:
    with rasterio.open(path) as src:
        if src.crs is None:
            return None
        bounds_geom = box(*src.bounds)
        return gpd.GeoSeries([bounds_geom], crs=src.crs).to_crs(4326).iloc[0]


def extract_aoi_from_file(tmp_path: Path, original_filename: str) -> dict:
    """Returns {"aoi": geojson dict, "source_filename": str, "area_hectares": float}.
    Tries vector first (the common case -- a hand-digitized field boundary),
    falls back to treating it as a raster footprint. Raises HTTPException
    with an actionable message if neither works, rather than a raw 500 --
    this is exactly the kind of "expected external input" failure CLAUDE.md's
    known-gaps list already flags predict() itself as still missing."""
    geom = None
    vector_error = None
    try:
        geom = _vector_polygon(tmp_path)
    except Exception as exc:
        vector_error = exc

    if geom is None:
        try:
            geom = _raster_footprint(tmp_path)
        except Exception as raster_error:
            # GDAL's own error text embeds the internal temp file path -- swap
            # it back out for the name the user actually recognizes before it
            # reaches the response.
            detail = f"(vector attempt: {vector_error}; raster attempt: {raster_error})".replace(
                str(tmp_path), original_filename
            )
            raise HTTPException(
                422,
                f"Could not read '{original_filename}' as a vector or raster geospatial file. "
                f"Supported: KML, GeoPackage, GeoJSON, zipped Shapefile, GeoTIFF, and most other "
                f"GDAL-readable formats. {detail}",
            ) from raster_error

    if geom is None or geom.is_empty:
        raise HTTPException(422, f"'{original_filename}' was read but contained no usable polygon geometry.")

    return {
        "aoi": geom.__geo_interface__,
        "source_filename": original_filename,
        "area_hectares": approx_area_hectares(geom),
    }


def save_upload_to_tempfile(contents: bytes, original_filename: str) -> tempfile.TemporaryDirectory:
    """Writes `contents` into a fresh temp directory under the ORIGINAL
    file's extension (GDAL driver detection leans on it for some formats)
    but never trusts the original filename as a path -- only its suffix is
    reused, inside a directory this function created itself. Caller is
    responsible for cleaning up the returned TemporaryDirectory (use it as
    a context manager)."""
    suffix = Path(original_filename).suffix
    tmp_dir = tempfile.TemporaryDirectory(prefix="aoi_upload_")
    tmp_path = Path(tmp_dir.name) / f"upload{suffix}"
    tmp_path.write_bytes(contents)
    return tmp_dir, tmp_path
