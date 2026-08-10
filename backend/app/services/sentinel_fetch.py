"""
Builds a per-parcel Sentinel-2 L1C time series for an AOI polygon using the
Copernicus Data Space Ecosystem's Sentinel Hub Statistical API: one HTTP
request computes the cloud-aware, per-band zonal mean for every available
scene in the date range server-side. This replaces the earlier approach of
searching a STAC catalog and clipping/averaging rasters locally with rasterio
— no local raster download, and no L2A/B10 workaround, since the Statistical
API serves true L1C directly (the product this model was actually trained on).
"""

import json
from datetime import date

import numpy as np
from sentinelhub import CRS, DataCollection, Geometry, SentinelHubStatistical, SHConfig
from shapely.geometry import Polygon, mapping

from app.config import (
    CDSE_SH_BASE_URL,
    CDSE_SH_CLIENT_ID,
    CDSE_SH_CLIENT_SECRET,
    CDSE_SH_TOKEN_URL,
    SENTINEL2_L1C_BANDS,
)

MAX_CLOUD_COVER_PERCENT = 60  # cheap whole-tile catalog pre-filter, kept as a first pass
MIN_AOI_CLEAR_FRACTION = 0.4  # below this, a day's AOI-specific cloud cover is too high
                               # to trust even though *some* clear pixels exist — skip the
                               # whole day rather than average over an unrepresentative sliver
REFLECTANCE_SCALE = 1e-4  # Sentinel-2 stores digital numbers; x1e-4 -> 0-1 reflectance
RESOLUTION_METERS = 10

# SENTINEL2_L1C_BANDS uses BreizhCrops' short band names ("B1", "B9") to match
# the model's training-time column order. Sentinel Hub's evalscript API needs
# its own zero-padded identifiers ("B01", "B09") — this translates between the
# two without disturbing SENTINEL2_L1C_BANDS, which is also our output column
# order and must keep matching the checkpoint exactly.
#
# Plain 1:1 mapping, no cross-wiring. The B2/B10 swap previously here existed
# only to match a quirk in BreizhCrops' own stored data (the original
# breizhcrops_france_9class.pth checkpoint). Every checkpoint trained since
# (see backend/*.pth) was trained on real, correctly-labeled bands fetched
# through this same pipeline, so reproducing that swap here would reintroduce
# the exact class of bug it existed to work around — feeding the model bands
# it never saw during training.
_SH_BAND_NAMES = {
    "B1": "B01", "B2": "B02", "B3": "B03", "B4": "B04", "B5": "B05",
    "B6": "B06", "B7": "B07", "B8": "B08", "B8A": "B8A", "B9": "B09",
    "B10": "B10", "B11": "B11", "B12": "B12",
}
_sh_bands = [_SH_BAND_NAMES[b] for b in SENTINEL2_L1C_BANDS]

# Cloud masking: "CLM" is Sentinel Hub's precomputed s2cloudless binary cloud mask
# (0 clear / 1 cloud / 255 no-data), available for L1C directly — no L2A/SCL needed.
# It's generated at 160m resolution and upsampled to match our 10m request, so cloud
# edges are blocky rather than pixel-precise, but it's real per-pixel cloud exclusion
# where previously there was none (only a whole-tile catalog filter). This masks
# CLOUDS only, not cloud SHADOWS — true SCL-based shadow classes don't exist for L1C;
# shadow masking is a still-open gap, not solved by this.
#
# Trick to get both a cloud-corrected band mean AND an AOI-specific cloud fraction out
# of a single Statistical API request (each extra request costs against the CDSE rate
# budget): "dataMask" stays footprint-only (unchanged), so its own stats give the total
# in-footprint pixel count. "clear_bands" zeroes out cloudy pixels; "clear_frac" reports
# 1/0 per pixel for clear/cloudy. Both are averaged over the SAME footprint-only-masked
# denominator, so True clear-sky band mean = clear_bands.mean / clear_frac.mean (as long
# as clear_frac.mean isn't ~0 — checked in Python below before dividing).
_EVALSCRIPT = f"""
//VERSION=3
function setup() {{
  return {{
    input: [{{
      bands: {json.dumps(_sh_bands + ["CLM", "dataMask"])},
      units: "DN"
    }}],
    output: [
      {{ id: "clear_bands", bands: {len(SENTINEL2_L1C_BANDS)}, sampleType: "INT16" }},
      {{ id: "clear_frac", bands: 1, sampleType: "FLOAT32" }},
      {{ id: "dataMask", bands: 1 }}
    ]
  }};
}}
function evaluatePixel(sample) {{
  var isClear = (sample.CLM === 0) ? 1 : 0;
  return {{
    clear_bands: [{", ".join(f"sample.{b} * isClear" for b in _sh_bands)}],
    clear_frac: [isClear],
    dataMask: [sample.dataMask]
  }};
}}
"""


def _sh_config() -> SHConfig:
    if not CDSE_SH_CLIENT_ID or not CDSE_SH_CLIENT_SECRET:
        raise RuntimeError(
            "CDSE_SH_CLIENT_ID / CDSE_SH_CLIENT_SECRET are not set. Register an "
            "OAuth client at the Sentinel Hub Dashboard (dataspace.copernicus.eu "
            "-> User Settings -> OAuth clients) and put them in backend/.env"
        )
    config = SHConfig()
    config.sh_client_id = CDSE_SH_CLIENT_ID
    config.sh_client_secret = CDSE_SH_CLIENT_SECRET
    config.sh_base_url = CDSE_SH_BASE_URL
    config.sh_token_url = CDSE_SH_TOKEN_URL
    return config


def _utm_epsg_for(lon: float, lat: float) -> int:
    """
    Nearest UTM zone EPSG code for a WGS84 point. The Statistical API requires
    resolution to be specified in the same units as the request geometry, so
    a plain WGS84 (degrees) geometry can't be paired with a 10m resolution —
    it has to be reprojected to a local metric CRS first.
    """
    zone = int((lon + 180) / 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def fetch_time_series(
    polygon: Polygon, start_date: date, end_date: date
) -> tuple[np.ndarray, list[date]]:
    """
    Query the Statistical API for `polygon` within [start_date, end_date].
    Daily aggregation naturally collapses to one row per actual acquisition —
    a given tile is only revisited every ~5 days, so empty days simply don't
    appear in the response. Returns (x_raw, dates): x_raw has shape
    (num_scenes, 13), columns matching SENTINEL2_L1C_BANDS order. Scenes with
    no valid pixels inside the AOI (e.g. fully outside the tile footprint) are
    skipped.
    """
    config = _sh_config()
    data_collection = DataCollection.SENTINEL2_L1C.define_from(
        "CDSE_SENTINEL2_L1C", service_url=config.sh_base_url
    )

    centroid = polygon.centroid
    geometry = Geometry(mapping(polygon), crs=CRS.WGS84).transform(
        CRS(_utm_epsg_for(centroid.x, centroid.y))
    )

    request = SentinelHubStatistical(
        aggregation=SentinelHubStatistical.aggregation(
            evalscript=_EVALSCRIPT,
            time_interval=(start_date, end_date),
            aggregation_interval="P1D",
            resolution=(RESOLUTION_METERS, RESOLUTION_METERS),
        ),
        input_data=[
            SentinelHubStatistical.input_data(
                data_collection, maxcc=MAX_CLOUD_COVER_PERCENT / 100
            )
        ],
        geometry=geometry,
        config=config,
    )
    result = request.get_data()[0]

    rows: list[list[float]] = []
    dates: list[date] = []
    for interval in result.get("data", []):
        row = _extract_band_means(interval)
        if row is None:
            continue
        rows.append(row)
        dates.append(date.fromisoformat(interval["interval"]["from"][:10]))

    return np.array(rows, dtype=np.float32), dates


def _extract_band_means(interval: dict) -> list[float] | None:
    """One row for one day's scene: mean in-AOI *clear-sky* reflectance per
    band, in SENTINEL2_L1C_BANDS order. Returns None if every pixel in the
    AOI was outside the scene's data footprint (dataMask all zero), or if
    fewer than MIN_AOI_CLEAR_FRACTION of the in-footprint pixels are cloud-free
    (the day is kept/dropped as a whole, not partially averaged over a small
    unrepresentative clear sliver)."""
    outputs = interval["outputs"]
    footprint_stat = outputs["clear_frac"]["bands"]["B0"]["stats"]
    if footprint_stat["sampleCount"] == 0 or footprint_stat["noDataCount"] >= footprint_stat["sampleCount"]:
        return None

    clear_fraction = footprint_stat["mean"]
    if clear_fraction < MIN_AOI_CLEAR_FRACTION:
        return None

    bands = outputs["clear_bands"]["bands"]
    return [
        (bands[f"B{i}"]["stats"]["mean"] / clear_fraction) * REFLECTANCE_SCALE
        for i in range(len(SENTINEL2_L1C_BANDS))
    ]
