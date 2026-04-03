"""
Builds a per-parcel Sentinel-2 time series for an AOI polygon: search scenes,
clip each to the polygon, and average the in-polygon pixels per band. This is
the zonal-statistics step — the raster-to-spreadsheet-row conversion the model
actually consumes.

Known limitation (see app.config.SENTINEL2_L2A_ASSET_KEYS): sourced from
Planetary Computer's L2A collection because that's what's actually available
there, not the L1C data the model was trained on. Predictions from this path
are provisional until that's fixed.
"""

from datetime import date

import numpy as np
import planetary_computer
import pystac_client
import rasterio
import rasterio.warp
from rasterio.mask import mask as rasterio_mask
from shapely.geometry import Polygon, mapping

from app.config import SENTINEL2_L1C_BANDS, SENTINEL2_L2A_ASSET_KEYS

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"
MAX_CLOUD_COVER_PERCENT = 60
REFLECTANCE_SCALE = 1e-4  # Sentinel-2 stores digital numbers; x1e-4 -> 0-1 reflectance


def fetch_time_series(
    polygon: Polygon, start_date: date, end_date: date
) -> tuple[np.ndarray, list[date]]:
    """
    Search Sentinel-2 scenes covering `polygon` within [start_date, end_date]
    and compute one averaged row per usable scene.

    Returns (x_raw, dates): x_raw has shape (num_scenes, 13), one row per
    scene ordered by date, columns matching SENTINEL2_L1C_BANDS order. Scenes
    with no valid pixels inside the AOI (e.g. fully cloud-masked) are skipped.
    """
    catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    search = catalog.search(
        collections=[COLLECTION],
        intersects=mapping(polygon),
        datetime=f"{start_date.isoformat()}/{end_date.isoformat()}",
        query={"eo:cloud_cover": {"lt": MAX_CLOUD_COVER_PERCENT}},
    )
    items = sorted(search.item_collection(), key=lambda item: item.datetime)

    rows: list[list[float]] = []
    dates: list[date] = []
    for item in items:
        row = _zonal_mean_per_band(item, polygon)
        if row is None:
            continue
        rows.append(row)
        dates.append(item.datetime.date())

    return np.array(rows, dtype=np.float32), dates


def _zonal_mean_per_band(item, polygon: Polygon) -> list[float] | None:
    """One row for one scene: mean in-polygon reflectance, per band, in
    SENTINEL2_L1C_BANDS order. Returns None if the scene has no usable pixels
    inside the AOI at all (skip it rather than poison the sequence with junk)."""
    row: list[float] = []
    for band in SENTINEL2_L1C_BANDS:
        asset_key = SENTINEL2_L2A_ASSET_KEYS[band]
        if asset_key is None:
            row.append(0.0)  # B10 — not present in L2A, see module docstring
            continue

        try:
            href = item.assets[asset_key].href
            with rasterio.open(href) as src:
                aoi_in_raster_crs = [
                    rasterio.warp.transform_geom("EPSG:4326", src.crs, mapping(polygon))
                ]
                clipped, _ = rasterio_mask(src, aoi_in_raster_crs, crop=True, filled=False)
        except Exception:
            return None  # asset missing/unreadable for this scene — drop the scene

        pixels = clipped[0].compressed()
        if pixels.size == 0:
            return None  # AOI fell entirely outside this scene's data footprint
        row.append(float(pixels.mean()) * REFLECTANCE_SCALE)

    return row
