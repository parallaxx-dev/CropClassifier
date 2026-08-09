"""
phase0_probe.py — one-off script, run manually, not part of the fetch pipeline.

Measures real CDSE Statistical API latency (serial baseline + a concurrent
sub-test) so MAX_WORKERS can be sized via Little's Law instead of guessed, and
checks for auth/rate-limit problems under concurrent load before committing to
a multi-hour fetch run. See plan doc, "Phase 0" and "Concurrency" sections.

Run: ../.venv/bin/python phase0_probe.py
"""

import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

from sentinelhub import CRS, DataCollection, Geometry, SentinelHubStatistical, SHConfig
from shapely.geometry import Polygon, mapping

from config import (
    CDSE_SH_BASE_URL,
    CDSE_SH_CLIENT_ID,
    CDSE_SH_CLIENT_SECRET,
    CDSE_SH_TOKEN_URL,
    RESOLUTION_METERS,
    SENTINEL2_L1C_BANDS,
)

# Real, already-verified-valid small parcel (from this session's earlier live
# testing) — a corn field in France with a full year of real CDSE fetch history.
TEST_POLYGON = Polygon([
    [2.0411045203813343, 44.317693430991774],
    [2.0404942867865192, 44.31747967281202],
    [2.040238168387811, 44.31800479956438],
    [2.040265280792046, 44.318166999785475],
    [2.0404148832691953, 44.318344558721634],
    [2.041088492734179, 44.31880616161624],
    [2.0418067135640428, 44.31841585004259],
    [2.0411045203813343, 44.317693430991774],
])

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


def _sh_config() -> SHConfig:
    assert CDSE_SH_CLIENT_ID and CDSE_SH_CLIENT_SECRET, "CDSE credentials not set in .env"
    config = SHConfig()
    config.sh_client_id = CDSE_SH_CLIENT_ID
    config.sh_client_secret = CDSE_SH_CLIENT_SECRET
    config.sh_base_url = CDSE_SH_BASE_URL
    config.sh_token_url = CDSE_SH_TOKEN_URL
    return config


def one_request(config: SHConfig) -> float:
    """Issue one real Statistical API request, return elapsed seconds."""
    data_collection = DataCollection.SENTINEL2_L1C.define_from(
        "PHASE0_PROBE_L1C", service_url=config.sh_base_url
    )
    geometry = Geometry(mapping(TEST_POLYGON), crs=CRS.WGS84).transform(CRS(32631))
    request = SentinelHubStatistical(
        aggregation=SentinelHubStatistical.aggregation(
            evalscript=_EVALSCRIPT,
            time_interval=(date(2018, 1, 1), date(2018, 12, 31)),
            aggregation_interval="P1D",
            resolution=(RESOLUTION_METERS, RESOLUTION_METERS),
        ),
        input_data=[SentinelHubStatistical.input_data(data_collection, maxcc=0.6)],
        geometry=geometry,
        config=config,
    )
    start = time.monotonic()
    request.get_data()
    return time.monotonic() - start


def serial_baseline(config: SHConfig, n: int = 15) -> list[float]:
    print(f"--- serial baseline: {n} requests ---")
    latencies = []
    for i in range(n):
        t = one_request(config)
        latencies.append(t)
        print(f"  request {i + 1}/{n}: {t:.2f}s")
    return latencies


def concurrent_smoke_test(config: SHConfig, n: int = 15, workers: int = 8) -> tuple[list[float], int]:
    print(f"--- concurrent smoke test: {n} requests, {workers} workers ---")
    latencies = []
    errors = 0
    start_wall = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one_request, config) for _ in range(n)]
        for i, fut in enumerate(as_completed(futures)):
            try:
                t = fut.result()
                latencies.append(t)
                print(f"  completed {i + 1}/{n}: {t:.2f}s")
            except Exception as e:
                errors += 1
                print(f"  ERROR on request {i + 1}/{n}: {type(e).__name__}: {e}")
    wall = time.monotonic() - start_wall
    print(f"  wall-clock for {n} concurrent requests: {wall:.2f}s "
          f"({n / wall * 60:.0f} req/min effective)")
    return latencies, errors


def main():
    config = _sh_config()

    serial = serial_baseline(config, n=15)
    avg_latency = statistics.mean(serial)
    print(f"\nserial avg latency: {avg_latency:.2f}s, "
          f"median: {statistics.median(serial):.2f}s, "
          f"max: {max(serial):.2f}s\n")

    concurrent, errors = concurrent_smoke_test(config, n=15, workers=8)
    if errors:
        print(f"\n*** {errors} error(s) during concurrent test — check for auth/rate-limit "
              f"issues before trusting concurrency in the real pipeline ***\n")
    if concurrent:
        print(f"concurrent avg latency: {statistics.mean(concurrent):.2f}s "
              f"(vs serial {avg_latency:.2f}s — should be similar if the server isn't "
              f"throttling individual concurrent requests)\n")

    print("NOTE: naive Little's Law (workers ~= target_rate * avg_latency) does NOT hold for")
    print("this API in practice — empirically, pushing worker count up past a fairly low")
    print("ceiling causes the sentinelhub SDK's own client-side rate limiter to kick in and")
    print("collapse throughput (latencies balloon 4s -> 30-70s, effective req/min DROPS as")
    print("workers increase past ~8, even with 0 hard errors). Don't use the formula output")
    print("here as a target — increase workers cautiously from a low baseline (5) and watch")
    print("for SHRateLimitWarning frequency increasing before trusting a higher number.")


if __name__ == "__main__":
    main()
