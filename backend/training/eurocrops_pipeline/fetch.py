"""
fetch.py — cloud-masked Sentinel-2 time series fetch, concurrent + checkpointed.

The evalscript/cloud-masking logic is identical to backend/app/services/
sentinel_fetch.py (CLM-based per-pixel cloud exclusion, clear_bands/clear_frac
division trick — see that file's comments for the full explanation), extended
to also capture per-observation dates (needed for the windowing mechanism later;
the single-country script silently dropped these, which this pipeline's Phase 4/5
plan already flagged as a gap worth fixing now rather than retrofitting after a
multi-hour fetch).

Concurrency: ThreadPoolExecutor (sentinelhub's get_data() is a blocking sync
call — I/O-bound, so real concurrency is achieved despite the GIL), a shared
TokenBucket rate limiter, thread-safe checkpointing. Worker count and rate
target come from config.py, both set conservatively per Phase 0's empirical
finding that pushing concurrency too high collapses throughput rather than
scaling it.
"""

import json
import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import geopandas as gpd
import numpy as np
from sentinelhub import CRS, DataCollection, Geometry, SentinelHubStatistical, SHConfig
from shapely.geometry import Polygon, mapping

from config import (
    CDSE_SH_BASE_URL,
    CDSE_SH_CLIENT_ID,
    CDSE_SH_CLIENT_SECRET,
    CDSE_SH_TOKEN_URL,
    CHECKPOINT_EVERY_N_COMPLETIONS,
    CHECKPOINT_EVERY_N_SECONDS,
    FETCH_MAX_RETRIES,
    FETCH_TIMEOUT_SECONDS,
    MAX_CLOUD_COVER_PERCENT,
    MAX_WORKERS,
    MIN_AOI_CLEAR_FRACTION,
    RATE_LIMITER_BURST_CAPACITY,
    RATE_LIMITER_TARGET_PER_SEC,
    REFLECTANCE_SCALE,
    RESOLUTION_METERS,
    SENTINEL2_L1C_BANDS,
)
from rate_limiter import TokenBucket

_SH_BAND_NAMES = {
    "B1": "B01", "B2": "B02", "B3": "B03", "B4": "B04", "B5": "B05",
    "B6": "B06", "B7": "B07", "B8": "B08", "B8A": "B8A", "B9": "B09",
    "B10": "B10", "B11": "B11", "B12": "B12",
}
_sh_bands = [_SH_BAND_NAMES[b] for b in SENTINEL2_L1C_BANDS]

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


_shared_data_collection: DataCollection | None = None


def save_geoms_checkpoint(sampled: gpd.GeoDataFrame, geoms_ckpt_path: Path) -> None:
    """Writes the sampled-geometries checkpoint in BOTH formats: the .pkl
    the training pipeline's own resume logic reads back (fast, and fine --
    same venv writes and reads it), and a companion .geojson for
    app/services/parcels.py + demo_parcels.py, which run in a *different*
    environment (a Docker image) where pickle's binary coupling to the
    exact pandas/geopandas build that wrote it is a real, previously-hit
    failure mode (pandas 3.x's StringDtype array layout failing to unpickle
    against an older pandas resolved at Docker build time). GeoJSON has no
    such cross-version coupling."""
    with open(geoms_ckpt_path, "wb") as f:
        pickle.dump(sampled, f)
    sampled.to_file(geoms_ckpt_path.with_suffix(".geojson"), driver="GeoJSON")


def get_data_collection(config: SHConfig) -> DataCollection:
    """DataCollection.define_from() registers a permanent, process-wide named
    collection in the sentinelhub library — calling it more than once with a
    different name but the same underlying definition (same service_url) raises
    ValueError('...already taken...'). This crashed the real multi-country run
    right after the first country: fetch_all() used to call define_from() with a
    per-country name on every invocation. Fixed by creating it exactly once per
    process and reusing it for every country/source afterward."""
    global _shared_data_collection
    if _shared_data_collection is None:
        _shared_data_collection = DataCollection.SENTINEL2_L1C.define_from(
            "MULTICOUNTRY_L1C", service_url=config.sh_base_url
        )
    return _shared_data_collection


def sh_config() -> SHConfig:
    assert CDSE_SH_CLIENT_ID and CDSE_SH_CLIENT_SECRET, (
        "CDSE_SH_CLIENT_ID / CDSE_SH_CLIENT_SECRET not set — put them in backend/.env"
    )
    config = SHConfig()
    config.sh_client_id = CDSE_SH_CLIENT_ID
    config.sh_client_secret = CDSE_SH_CLIENT_SECRET
    config.sh_base_url = CDSE_SH_BASE_URL
    config.sh_token_url = CDSE_SH_TOKEN_URL
    return config


def _utm_epsg_for(lon: float, lat: float) -> int:
    zone = int((lon + 180) / 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def _extract_band_means(interval: dict) -> list[float] | None:
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


def fetch_time_series(
    polygon: Polygon,
    start_date: date,
    end_date: date,
    config: SHConfig,
    data_collection: DataCollection,
    rate_limiter: TokenBucket,
) -> tuple[np.ndarray, list[date]]:
    """Returns (x_raw, dates) — dates captured (unlike the single-country
    script), needed for the windowing mechanism later even though it's off by
    default for this retrain."""
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
            SentinelHubStatistical.input_data(data_collection, maxcc=MAX_CLOUD_COVER_PERCENT / 100)
        ],
        geometry=geometry,
        config=config,
    )
    rate_limiter.acquire()
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


def _fetch_one_with_retry(
    polygon: Polygon,
    start_date: date,
    end_date: date,
    config: SHConfig,
    data_collection: DataCollection,
    rate_limiter: TokenBucket,
) -> tuple[np.ndarray, list[date]]:
    last_exc = None
    for attempt in range(1, FETCH_MAX_RETRIES + 1):
        try:
            return fetch_time_series(polygon, start_date, end_date, config, data_collection, rate_limiter)
        except Exception as e:
            last_exc = e
            wait = min(30, 2**attempt)
            print(f"    retry {attempt}/{FETCH_MAX_RETRIES} after error ({type(e).__name__}: {e}), "
                  f"waiting {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"fetch failed after {FETCH_MAX_RETRIES} retries") from last_exc


def fetch_all(sampled, country_code: str, start_date: date, end_date: date, checkpoint_path) -> dict:
    """Concurrent, checkpointed fetch for one country's (or BreizhCrops') already-
    sampled GeoDataFrame. Resumes from checkpoint_path if it exists — keyed by
    the GeoDataFrame's row index, which sample_country() already resets to a
    clean 0..N-1 range."""
    if checkpoint_path.exists():
        with open(checkpoint_path, "rb") as f:
            results = pickle.load(f)
        print(f"[{country_code}] resuming, {len(results)} parcels already fetched")
    else:
        results = {}

    config = sh_config()
    data_collection = get_data_collection(config)
    rate_limiter = TokenBucket(RATE_LIMITER_TARGET_PER_SEC, RATE_LIMITER_BURST_CAPACITY)

    lock = threading.Lock()
    last_checkpoint_time = time.monotonic()
    completions_since_checkpoint = 0

    def maybe_checkpoint(force: bool = False):
        nonlocal last_checkpoint_time, completions_since_checkpoint
        with lock:
            elapsed = time.monotonic() - last_checkpoint_time
            if force or completions_since_checkpoint >= CHECKPOINT_EVERY_N_COMPLETIONS or elapsed >= CHECKPOINT_EVERY_N_SECONDS:
                with open(checkpoint_path, "wb") as f:
                    pickle.dump(results, f)
                completions_since_checkpoint = 0
                last_checkpoint_time = time.monotonic()

    todo = [(idx, row) for idx, row in sampled.iterrows() if str(idx) not in results]
    print(f"[{country_code}] {len(todo)} parcels to fetch ({len(results)} already done)")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_to_idx = {
            pool.submit(
                _fetch_one_with_retry, row.geometry, start_date, end_date, config, data_collection, rate_limiter
            ): (idx, row)
            for idx, row in todo
        }
        n_done = 0
        for fut in as_completed(future_to_idx):
            idx, row = future_to_idx[fut]
            key = str(idx)
            try:
                x_raw, dates = fut.result()
                if len(x_raw) > 0:
                    with lock:
                        results[key] = {
                            "x_raw": x_raw,
                            "dates": dates,
                            "classname": row["classname"],
                        }
            except Exception as e:
                print(f"    [{country_code}] permanent skip for parcel {key}: {type(e).__name__}: {e}")

            n_done += 1
            with lock:
                completions_since_checkpoint += 1
            if n_done % 25 == 0 or n_done == len(todo):
                print(f"[{country_code}] {n_done}/{len(todo)} processed, {len(results)} succeeded so far")
            maybe_checkpoint()

    maybe_checkpoint(force=True)
    print(f"[{country_code}] done: {len(results)} parcels fetched, checkpoint saved to {checkpoint_path}")
    return results
