"""One-off builder script — writes eurocrops_model_training_kaggle_multicountry.ipynb.
Run once, not part of the pipeline. Not committed logic to import elsewhere."""
import json

MD = "markdown"
CODE = "code"

cells = []


def add(cell_type, source):
    cells.append({
        "cell_type": cell_type,
        "metadata": {},
        **({"execution_count": None, "outputs": []} if cell_type == CODE else {}),
        "source": source.splitlines(keepends=True),
    })


add(MD, """# EuroCrops multi-country retraining pipeline (Kaggle)

Kaggle-notebook variant of `backend/training/eurocrops_pipeline/` (the local/laptop
package) — same pipeline, adapted for Kaggle's environment. This is NOT a thin wrapper
that imports the local package (Kaggle notebooks don't have straightforward access to
this repo's file layout) — it's a self-contained inline copy, matching how
`eurocrops_model_training_kaggle.ipynb` (the single-country predecessor) already
duplicates rather than imports.

**What this does differently from the single-country notebook:** samples a small,
class-balanced set of parcels across 18 EuroCrops countries (not just France) plus
BreizhCrops, using EuroCrops' own harmonized HCAT taxonomy instead of France-specific
RPG codes, fixes a real bug where the single-country version's resampling was computed
once and reused for all 20 epochs instead of fresh per epoch, and adds a per-class
metrics harness (nothing in this project previously reported anything beyond overall
accuracy). Full design rationale: see
`/home/parallax/.claude/plans/see-for-re-training-the-cheerful-bee.md` in the repo, and
`progress.md`'s "Code audit" / multi-country entries.

**Storage**: raw per-country zip download + extraction routes through `/kaggle/tmp`
(non-persistent scratch, ~60GB budget — auto-cleaned by Kaggle, plus this notebook also
explicitly deletes each country's extracted files immediately after sampling it, so
disk doesn't accumulate across countries within one long session either). Only the small
sampled-parcel checkpoints and the final model go to `/kaggle/working/` (persistent,
~20GB — survives "Save Version").

**GPU usage**: fetching does NOT use the GPU at all (it's a network-bound API fetch
loop). **Run the fetch cells with the accelerator set to "None"** to avoid burning your
weekly GPU quota on a phase that can't use it — switch to GPU (Settings → Accelerator)
only before running the training cell near the end.

**Concurrency**: `MAX_WORKERS` below is set to 5, not derived fresh on Kaggle — this was
empirically measured against the real CDSE API from a different machine (same
credentials, same server-side rate limiting applies regardless of which machine issues
the requests, so the earlier measurement transfers). That measurement found something
non-obvious: pushing concurrency past ~8 workers doesn't get throttled at the API
gateway (no hard errors) — instead, `sentinelhub`'s own client-side rate limiter kicks
in and backs off hard, so *more concurrent workers made throughput worse*, not better
(latencies went from ~4s to 30-70s, throughput collapsed from ~100 req/min to ~20-35).
Don't raise `MAX_WORKERS` without re-testing that specifically — it's not a "faster
laptop or better network fixes this" situation.

**BreizhCrops section**: needs `frh04.shp` (+ .shx/.dbf/.prj/.cpg), which lives in this
repo at `backend/breizhcrops_dataset/2017/` and isn't otherwise available on Kaggle.
Upload those files as a private Kaggle Dataset (e.g. named `breizhcrops-frh04`) and
attach it to this notebook before running that section — it's skipped automatically
with a clear message if not found, so the rest of the notebook still runs fine without
it.""")

add(CODE, """# ---- Setup ----
!pip install -q geopandas sentinelhub shapely scikit-learn

import gc
import json as _json
import os
import pickle
import random
import shutil
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from sentinelhub import CRS, DataCollection, Geometry, SentinelHubStatistical, SHConfig
from shapely.geometry import mapping as shapely_mapping
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset

random.seed(42)
np.random.seed(42)

# ---- Credentials (Kaggle Secrets — Add-ons -> Secrets in the notebook editor,
# do NOT paste real values into this cell, notebooks are commonly made public/forked) ----
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
CDSE_SH_CLIENT_ID = user_secrets.get_secret("CDSE_SH_CLIENT_ID")
CDSE_SH_CLIENT_SECRET = user_secrets.get_secret("CDSE_SH_CLIENT_SECRET")
CDSE_SH_BASE_URL = "https://sh.dataspace.copernicus.eu"
CDSE_SH_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

# ---- Paths ----
WORKING_DIR = Path("/kaggle/working")
SCRATCH_DIR = Path("/kaggle/tmp/eurocrops_scratch")
CHECKPOINT_DIR = WORKING_DIR / "checkpoints"
SAVE_PATH = WORKING_DIR / "multicountry_india_18class.pth"
for d in (SCRATCH_DIR, CHECKPOINT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---- Sequence / band config — must match the deployed backend exactly ----
TARGET_SEQ_LEN = 45
INPUT_DIM = 13
SENTINEL2_L1C_BANDS = ["B1", "B10", "B11", "B12", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9"]
_SH_BAND_NAMES = {
    "B1": "B01", "B2": "B02", "B3": "B03", "B4": "B04", "B5": "B05",
    "B6": "B06", "B7": "B07", "B8": "B08", "B8A": "B8A", "B9": "B09",
    "B10": "B10", "B11": "B11", "B12": "B12",
}
_sh_bands = [_SH_BAND_NAMES[b] for b in SENTINEL2_L1C_BANDS]

# ---- Cloud masking (CLM = Sentinel Hub's s2cloudless binary cloud mask, real for
# L1C directly, no L2A/SCL needed — see backend/app/services/sentinel_fetch.py for
# the full explanation of the clear_bands/clear_frac division trick used below) ----
MAX_CLOUD_COVER_PERCENT = 60
MIN_AOI_CLEAR_FRACTION = 0.4
REFLECTANCE_SCALE = 1e-4
RESOLUTION_METERS = 10

# ---- Sampling / concurrency (see markdown above re: MAX_WORKERS) ----
N_PER_CLASS_PER_COUNTRY = 150
OVERSAMPLE_PER_HCAT_CODE = max(50, N_PER_CLASS_PER_COUNTRY)
MAX_WORKERS = 5
RATE_LIMITER_TARGET_PER_SEC = 4.5
RATE_LIMITER_BURST_CAPACITY = 8
CHECKPOINT_EVERY_N_COMPLETIONS = 50
CHECKPOINT_EVERY_N_SECONDS = 60
FETCH_MAX_RETRIES = 4

# ---- Country table — verified directly against Zenodo record 10118572 (v10, the
# CURRENT latest version — not the v9 record 8229128 the single-country notebook
# still points at; v10 adds Czech Republic + Germany-Brandenburg). Romania excluded:
# its zip is named "RO_ny" ("no year") — the parcel-declaration year satellite
# imagery needs to be matched against is unconfirmed for it. ----
@dataclass(frozen=True)
class CountryEntry:
    code: str
    zenodo_filename: str
    year: int

COUNTRIES = [
    CountryEntry("AT", "AT_2021.zip", 2021),
    CountryEntry("BE_VLG", "BE_VLG_2021.zip", 2021),
    CountryEntry("CZ", "CZ_2023.zip", 2023),
    CountryEntry("DE_BB", "DE_BB_2023.zip", 2023),
    CountryEntry("DE_LS", "DE_LS_2021.zip", 2021),
    CountryEntry("DE_NRW", "DE_NRW_2021.zip", 2021),
    CountryEntry("DK", "DK_2019.zip", 2019),
    CountryEntry("EE", "EE_2021.zip", 2021),
    CountryEntry("ES_NA", "ES_NA_2020.zip", 2020),
    CountryEntry("FR", "FR_2018.zip", 2018),
    CountryEntry("HR", "HR_2020.zip", 2020),
    CountryEntry("LT", "LT_2021.zip", 2021),
    CountryEntry("LV", "LV_2021.zip", 2021),
    CountryEntry("NL", "NL_2020.zip", 2020),
    CountryEntry("PT", "PT.zip", 2021),
    CountryEntry("SE", "SE_2021.zip", 2021),
    CountryEntry("SI", "SI_2021.zip", 2021),
    CountryEntry("SK", "SK_2021.zip", 2021),
]
ZENODO_RECORD_ID = "10118572"
def zenodo_url(filename):
    return f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}/files/{filename}/content"

print(f"{len(COUNTRIES)} countries configured, MAX_WORKERS={MAX_WORKERS}")""")

add(CODE, """# ---- Taxonomy: 11 target classes, built from EuroCrops' own harmonized HCAT3
# reference table (downloaded fresh, not bundled) + France's RPG crosswalk (used
# both for France's own EuroCrops contribution — actually not needed since France's
# shapefile ALSO carries EC_hcat_n/EC_hcat_c directly, verified — and for BreizhCrops,
# which does NOT carry HCAT columns and needs the raw-code path).
#
# Class scheme decisions (see plan doc for full discussion/corrections):
# - Expanded beyond the old 9-class scheme to match EuroCrops' actual harmonized
#   categories, validated against the "Top 10 Absolute Scale" counts on EuroCrops' site.
# - Permanent + temporary meadow merged into one `meadow` class — a DELIBERATE
#   simplicity tradeoff, not a data limitation: HCAT3 actually DOES distinguish them
#   (pasture_meadow_grassland_grass vs temporary_grass), verified directly against
#   real crosswalk data for France AND Portugal (an earlier notebook comment claiming
#   HCAT collapses this split was checked here and found to be wrong).
# - Winter/spring/summer variants of wheat, barley, triticale, rapeseed merged into
#   one class each, via HCAT3's own parent/child code hierarchy.

_taxonomy_ref_dir = SCRATCH_DIR / "_taxonomy_ref"
_taxonomy_ref_dir.mkdir(parents=True, exist_ok=True)
for fname in ("HCAT3.csv", "fr_2018.csv"):
    fpath = _taxonomy_ref_dir / fname
    if not fpath.exists():
        r = requests.get(zenodo_url(fname), timeout=30)
        r.raise_for_status()
        fpath.write_bytes(r.content)

CLASS_NAMES = {
    0: "meadow", 1: "wheat", 2: "barley", 3: "triticale", 4: "rapeseed",
    5: "maize", 6: "sunflower", 7: "vineyards", 8: "fruit", 9: "nuts", 10: "potatoes",
}
CLASS_NAME_TO_ID = {v: k for k, v in CLASS_NAMES.items()}
NUM_CLASSES = len(CLASS_NAMES)

_CLASS_HCAT3_ROOTS = {
    "meadow": ["pasture_meadow_grassland_grass", "temporary_grass"],
    "wheat": ["common_soft_wheat"],
    "barley": ["barley"],
    "triticale": ["triticale"],
    "rapeseed": ["rapeseed_rape"],
    "maize": ["grain_maize_corn_popcorn"],
    "sunflower": ["sunflower"],
    "vineyards": ["vineyards_wine_vine_rebland_grapes"],
    "fruit": ["orchards_fruits"],
    "nuts": ["nuts"],
    "potatoes": ["potatoes"],
}

def _descendants_or_self(hcat3_df, root_name):
    matches = hcat3_df.loc[hcat3_df["HCAT3_name"] == root_name, "HCAT3_code"]
    if matches.empty:
        raise ValueError(f"HCAT3 root name not found: {root_name!r}")
    root_code = matches.iloc[0]
    trailing_zeros = len(root_code) - len(root_code.rstrip("0"))
    prefix = root_code[: len(root_code) - trailing_zeros]
    return hcat3_df.loc[hcat3_df["HCAT3_code"].str.startswith(prefix), "HCAT3_name"].tolist()

_hcat3_df = pd.read_csv(_taxonomy_ref_dir / "HCAT3.csv")
_hcat3_df["HCAT3_code"] = _hcat3_df["HCAT3_code"].astype(str)

HCAT_TO_CLASS = {}
for _cls, _roots in _CLASS_HCAT3_ROOTS.items():
    for _root in _roots:
        for _member in _descendants_or_self(_hcat3_df, _root):
            HCAT_TO_CLASS[_member] = _cls

_fr_crosswalk = pd.read_csv(_taxonomy_ref_dir / "fr_2018.csv")
RPG_CODE_TO_CLASS = {}
for _, _row in _fr_crosswalk.iterrows():
    _code, _hcat3_name = _row.get("original_code"), _row.get("HCAT3_name")
    if pd.isna(_code) or pd.isna(_hcat3_name):
        continue
    _cls = HCAT_TO_CLASS.get(_hcat3_name)
    if _cls is not None:
        RPG_CODE_TO_CLASS[str(_code)] = _cls

_CLASS_TO_HCAT3_NAMES = {}
for _name, _cls in HCAT_TO_CLASS.items():
    _CLASS_TO_HCAT3_NAMES.setdefault(_cls, []).append(_name)
_CLASS_TO_RPG_CODES = {}
for _code, _cls in RPG_CODE_TO_CLASS.items():
    _CLASS_TO_RPG_CODES.setdefault(_cls, []).append(_code)

print(f"{NUM_CLASSES} classes, {len(HCAT_TO_CLASS)} HCAT3 names mapped, "
      f"{len(RPG_CODE_TO_CLASS)} RPG codes mapped (BreizhCrops path)")
for _cls in CLASS_NAMES.values():
    print(f"  {_cls:12s} <- {sorted(_CLASS_TO_HCAT3_NAMES.get(_cls, []))}")""")

add(CODE, """# ---- Shared token-bucket rate limiter (one instance for the whole run) ----
class TokenBucket:
    def __init__(self, rate_per_sec, capacity):
        self._rate = rate_per_sec
        self._capacity = capacity
        self._tokens = capacity
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self):
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_refill
                self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
                self._last_refill = now
                if self._tokens >= 1:
                    self._tokens -= 1
                    return
                wait = (1 - self._tokens) / self._rate
            time.sleep(wait)""")

add(CODE, """# ---- Download + extract + cleanup (raw data lives in /kaggle/tmp scratch, not
# the persistent /kaggle/working — see markdown intro) ----

def _download_with_resume(url, dest_path, max_retries=12):
    for attempt in range(1, max_retries + 1):
        existing_size = dest_path.stat().st_size if dest_path.exists() else 0
        headers = {"Range": f"bytes={existing_size}-"} if existing_size else {}
        try:
            with requests.get(url, stream=True, headers=headers, timeout=60) as resp:
                if existing_size and resp.status_code == 416:
                    return
                if existing_size and resp.status_code == 200:
                    existing_size = 0
                resp.raise_for_status()
                mode = "ab" if existing_size and resp.status_code == 206 else "wb"
                with open(dest_path, mode) as out:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        out.write(chunk)
            return
        except Exception as e:
            print(f"  download attempt {attempt}/{max_retries} failed: {e}")
            if attempt == max_retries:
                raise
            time.sleep(min(60, 2 ** attempt))


def download_and_extract(country):
    country_dir = SCRATCH_DIR / country.code
    zip_path = SCRATCH_DIR / f"{country.code}.zip"
    country_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{country.code}] downloading {country.zenodo_filename}...")
    _download_with_resume(zenodo_url(country.zenodo_filename), zip_path)
    print(f"[{country.code}] extracting...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(country_dir)
    for p in country_dir.rglob("*.shp"):
        return p
    raise FileNotFoundError(f"[{country.code}] no .shp found")


def cleanup(country):
    country_dir = SCRATCH_DIR / country.code
    zip_path = SCRATCH_DIR / f"{country.code}.zip"
    if country_dir.exists():
        shutil.rmtree(country_dir)
    if zip_path.exists():
        zip_path.unlink()
    print(f"[{country.code}] cleaned up scratch space")""")

add(CODE, """# ---- Per-country / BreizhCrops sampling (per-class capped reads, bounded memory) ----

def probe_schema(shp_path):
    probe = gpd.read_file(shp_path, rows=500)
    missing = {"EC_hcat_n", "EC_hcat_c"} - set(probe.columns)
    if missing:
        raise ValueError(f"{shp_path}: missing columns {missing} — got {probe.columns.tolist()}")
    if probe.crs is None:
        raise ValueError(f"{shp_path}: no CRS set")
    return {"crs": str(probe.crs), "null_rate": probe["EC_hcat_c"].isna().mean()}


def sample_country(shp_path, country):
    frames = []
    for cls in CLASS_NAMES.values():
        members = _CLASS_TO_HCAT3_NAMES.get(cls, [])
        if not members:
            continue
        where = "EC_hcat_n IN (" + ", ".join(f"'{m}'" for m in members) + ")"
        cls_gdf = gpd.read_file(shp_path, where=where, rows=OVERSAMPLE_PER_HCAT_CODE)
        if len(cls_gdf) == 0:
            continue
        cls_gdf["classname"] = cls
        frames.append(cls_gdf[["classname", "geometry"]])
        print(f"  [{country.code}] {cls}: read {len(cls_gdf)} parcels")
        del cls_gdf
    if not frames:
        return gpd.GeoDataFrame(columns=["classname", "geometry"])
    gdf_labeled = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)
    del frames; gc.collect()
    sample_parts = [
        group.sample(n=min(N_PER_CLASS_PER_COUNTRY, len(group)), random_state=42)
        for _, group in gdf_labeled.groupby("classname")
    ]
    sampled = gpd.GeoDataFrame(pd.concat(sample_parts, ignore_index=True), crs=gdf_labeled.crs).to_crs(epsg=4326)
    del gdf_labeled; gc.collect()
    print(f"  [{country.code}] sampled {len(sampled)}: {sampled['classname'].value_counts().to_dict()}")
    return sampled


def sample_breizhcrops(shp_path):
    frames = []
    for cls in CLASS_NAMES.values():
        codes = _CLASS_TO_RPG_CODES.get(cls, [])
        if not codes:
            continue
        where = "CODE_CULTU IN (" + ", ".join(f"'{c}'" for c in codes) + ")"
        cls_gdf = gpd.read_file(shp_path, where=where, rows=OVERSAMPLE_PER_HCAT_CODE)
        if len(cls_gdf) == 0:
            continue
        cls_gdf["classname"] = cls
        frames.append(cls_gdf[["classname", "geometry"]])
        print(f"  [BZH] {cls}: read {len(cls_gdf)} parcels")
        del cls_gdf
    if not frames:
        return gpd.GeoDataFrame(columns=["classname", "geometry"])
    gdf_labeled = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)
    del frames; gc.collect()
    sample_parts = [
        group.sample(n=min(N_PER_CLASS_PER_COUNTRY, len(group)), random_state=42)
        for _, group in gdf_labeled.groupby("classname")
    ]
    sampled = gpd.GeoDataFrame(pd.concat(sample_parts, ignore_index=True), crs=gdf_labeled.crs).to_crs(epsg=4326)
    del gdf_labeled; gc.collect()
    print(f"  [BZH] sampled {len(sampled)}: {sampled['classname'].value_counts().to_dict()}")
    return sampled""")

add(CODE, """# ---- Cloud-masked fetch (identical logic to backend/app/services/sentinel_fetch.py
# — CLM-based per-pixel cloud exclusion, clear_bands/clear_frac division trick — plus
# dates captured for the windowing hook, which single-country versions dropped) ----

_EVALSCRIPT = f'''
//VERSION=3
function setup() {{
  return {{
    input: [{{
      bands: {_json.dumps(_sh_bands + ["CLM", "dataMask"])},
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
'''


def sh_config():
    config = SHConfig()
    config.sh_client_id = CDSE_SH_CLIENT_ID
    config.sh_client_secret = CDSE_SH_CLIENT_SECRET
    config.sh_base_url = CDSE_SH_BASE_URL
    config.sh_token_url = CDSE_SH_TOKEN_URL
    return config


# DataCollection.define_from() registers a permanent, process-wide named collection —
# calling it more than once with a different name but the same underlying definition
# (same service_url) raises ValueError('...already taken...'). Caught this live: it
# crashed the real multi-country run right after the first country finished, since
# fetch_all() used to call define_from() with a per-country name on every invocation.
# Fixed by creating it exactly once per kernel session and reusing it for every
# country/source afterward.
_shared_data_collection = None


def get_data_collection(config):
    global _shared_data_collection
    if _shared_data_collection is None:
        _shared_data_collection = DataCollection.SENTINEL2_L1C.define_from("MULTICOUNTRY_L1C", service_url=config.sh_base_url)
    return _shared_data_collection


def _utm_epsg_for(lon, lat):
    zone = int((lon + 180) / 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def _extract_band_means(interval):
    outputs = interval["outputs"]
    footprint_stat = outputs["clear_frac"]["bands"]["B0"]["stats"]
    if footprint_stat["sampleCount"] == 0 or footprint_stat["noDataCount"] >= footprint_stat["sampleCount"]:
        return None
    clear_fraction = footprint_stat["mean"]
    if clear_fraction < MIN_AOI_CLEAR_FRACTION:
        return None
    bands = outputs["clear_bands"]["bands"]
    return [(bands[f"B{i}"]["stats"]["mean"] / clear_fraction) * REFLECTANCE_SCALE
            for i in range(len(SENTINEL2_L1C_BANDS))]


def fetch_time_series(polygon, start_date, end_date, config, data_collection, rate_limiter):
    centroid = polygon.centroid
    geometry = Geometry(shapely_mapping(polygon), crs=CRS.WGS84).transform(CRS(_utm_epsg_for(centroid.x, centroid.y)))
    request = SentinelHubStatistical(
        aggregation=SentinelHubStatistical.aggregation(
            evalscript=_EVALSCRIPT, time_interval=(start_date, end_date),
            aggregation_interval="P1D", resolution=(RESOLUTION_METERS, RESOLUTION_METERS),
        ),
        input_data=[SentinelHubStatistical.input_data(data_collection, maxcc=MAX_CLOUD_COVER_PERCENT / 100)],
        geometry=geometry, config=config,
    )
    rate_limiter.acquire()
    result = request.get_data()[0]
    rows, dates = [], []
    for interval in result.get("data", []):
        row = _extract_band_means(interval)
        if row is None:
            continue
        rows.append(row)
        dates.append(date.fromisoformat(interval["interval"]["from"][:10]))
    return np.array(rows, dtype=np.float32), dates


def _fetch_one_with_retry(polygon, start_date, end_date, config, data_collection, rate_limiter):
    last_exc = None
    for attempt in range(1, FETCH_MAX_RETRIES + 1):
        try:
            return fetch_time_series(polygon, start_date, end_date, config, data_collection, rate_limiter)
        except Exception as e:
            last_exc = e
            time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"fetch failed after {FETCH_MAX_RETRIES} retries") from last_exc


def fetch_all(sampled, source_code, start_date, end_date, checkpoint_path):
    if checkpoint_path.exists():
        with open(checkpoint_path, "rb") as f:
            results = pickle.load(f)
        print(f"[{source_code}] resuming, {len(results)} already fetched")
    else:
        results = {}

    config = sh_config()
    data_collection = get_data_collection(config)
    rate_limiter = TokenBucket(RATE_LIMITER_TARGET_PER_SEC, RATE_LIMITER_BURST_CAPACITY)
    lock = threading.Lock()
    state = {"last_checkpoint": time.monotonic(), "since_checkpoint": 0}

    def maybe_checkpoint(force=False):
        with lock:
            elapsed = time.monotonic() - state["last_checkpoint"]
            if force or state["since_checkpoint"] >= CHECKPOINT_EVERY_N_COMPLETIONS or elapsed >= CHECKPOINT_EVERY_N_SECONDS:
                with open(checkpoint_path, "wb") as f:
                    pickle.dump(results, f)
                state["since_checkpoint"] = 0
                state["last_checkpoint"] = time.monotonic()

    todo = [(idx, row) for idx, row in sampled.iterrows() if str(idx) not in results]
    print(f"[{source_code}] {len(todo)} parcels to fetch ({len(results)} already done)")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_to_idx = {
            pool.submit(_fetch_one_with_retry, row.geometry, start_date, end_date, config, data_collection, rate_limiter): (idx, row)
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
                        results[key] = {"x_raw": x_raw, "dates": dates, "classname": row["classname"]}
            except Exception as e:
                print(f"    [{source_code}] permanent skip for {key}: {type(e).__name__}: {e}")
            n_done += 1
            with lock:
                state["since_checkpoint"] += 1
            if n_done % 25 == 0 or n_done == len(todo):
                print(f"[{source_code}] {n_done}/{len(todo)} processed, {len(results)} succeeded so far")
            maybe_checkpoint()

    maybe_checkpoint(force=True)
    print(f"[{source_code}] done: {len(results)} parcels fetched")
    return results""")

add(CODE, """# ---- Main fetch loop: EuroCrops, 18 countries. Resumable — re-running this cell
# skips any country already fully sampled+fetched. RUN WITH GPU ACCELERATOR OFF. ----

for country in COUNTRIES:
    print(f"\\n{'=' * 60}\\n{country.code} ({country.year})\\n{'=' * 60}")
    geoms_ckpt = CHECKPOINT_DIR / f"{country.code}_sampled_geoms.pkl"
    fetch_ckpt = CHECKPOINT_DIR / f"{country.code}_fetched.pkl"

    if fetch_ckpt.exists() and geoms_ckpt.exists():
        with open(fetch_ckpt, "rb") as f:
            existing = pickle.load(f)
        with open(geoms_ckpt, "rb") as f:
            sampled_check = pickle.load(f)
        if len(existing) >= len(sampled_check):
            print(f"[{country.code}] already fully fetched ({len(existing)}) — skipping")
            continue

    if geoms_ckpt.exists():
        with open(geoms_ckpt, "rb") as f:
            sampled = pickle.load(f)
        print(f"[{country.code}] reusing sampled geometries ({len(sampled)})")
    else:
        shp_path = download_and_extract(country)
        schema = probe_schema(shp_path)
        print(f"[{country.code}] schema: {schema}")
        if schema["null_rate"] > 0.1:
            print(f"[{country.code}] *** WARNING: {schema['null_rate']:.1%} null EC_hcat_c rate ***")
        sampled = sample_country(shp_path, country)
        with open(geoms_ckpt, "wb") as f:
            pickle.dump(sampled, f)
        cleanup(country)

    fetch_all(sampled, country.code, date(country.year, 1, 1), date(country.year, 12, 31), fetch_ckpt)

print("\\nAll EuroCrops countries done.")""")

add(MD, """## BreizhCrops (optional)

Requires `frh04.shp` (+ `.shx`/`.dbf`/`.prj`/`.cpg`) from `backend/breizhcrops_dataset/2017/`
in this repo, uploaded as a private Kaggle Dataset and attached to this notebook. If not
attached, this cell prints a message and the notebook continues fine without it (training
just won't include the `--breizhcrops` fold-in data).""")

add(CODE, """# ---- BreizhCrops fetch — skips gracefully if the shapefile isn't attached ----
_BZH_CANDIDATES = list(Path("/kaggle/input").glob("**/frh04.shp"))

if not _BZH_CANDIDATES:
    print("frh04.shp not found under /kaggle/input — skipping BreizhCrops. "
          "Upload backend/breizhcrops_dataset/2017/ as a Kaggle Dataset and attach it "
          "to this notebook, then re-run this cell, if you want it included.")
else:
    bzh_shp_path = _BZH_CANDIDATES[0]
    print(f"found BreizhCrops shapefile: {bzh_shp_path}")
    bzh_geoms_ckpt = CHECKPOINT_DIR / "BZH_sampled_geoms.pkl"
    bzh_fetch_ckpt = CHECKPOINT_DIR / "BZH_fetched.pkl"

    if bzh_geoms_ckpt.exists():
        with open(bzh_geoms_ckpt, "rb") as f:
            bzh_sampled = pickle.load(f)
        print(f"[BZH] reusing sampled geometries ({len(bzh_sampled)})")
    else:
        bzh_sampled = sample_breizhcrops(bzh_shp_path)
        with open(bzh_geoms_ckpt, "wb") as f:
            pickle.dump(bzh_sampled, f)

    fetch_all(bzh_sampled, "BZH", date(2017, 1, 1), date(2017, 12, 31), bzh_fetch_ckpt)
    print("BreizhCrops done.")""")

add(MD, """## Training

**Switch the accelerator to GPU now** (Settings → Accelerator → T4 x2 or similar) before
running the cells below — everything above was fetch-only and didn't need it.""")

add(CODE, """# ---- Dataset: the actual per-epoch resampling fix ----
# Bug this replaces: earlier single-country notebooks called preprocess_sequence()
# ONCE per parcel to build a static array, reused for all 20 epochs via a plain
# TensorDataset — the model never actually saw fresh resampling despite that being
# the entire point of resample-with-replacement (mirrors what the ORIGINAL
# breizhcrops.BreizhCrops package's own __getitem__ did — genuinely fresh per access).
# Fix: store raw variable-length sequences, resample fresh inside __getitem__ each
# time — a shuffling DataLoader then gives real per-epoch augmentation for free.

MIN_WINDOW_DAYS, MAX_WINDOW_DAYS = 30, 60
ENABLE_WINDOWING = False  # off by default per plan — see markdown intro / plan doc


def preprocess_sequence(x_raw, target_seq_len=TARGET_SEQ_LEN):
    seq_len = x_raw.shape[0]
    idxs = np.random.choice(seq_len, target_seq_len, replace=seq_len < target_seq_len)
    idxs.sort()
    x_resampled = x_raw[idxs]
    mean = x_resampled.mean(axis=0)
    std = x_resampled.std(axis=0)
    std[std == 0] = 1.0
    return (x_resampled - mean) / std


def _random_window(x_raw, dates):
    if len(dates) < 5:
        return x_raw, dates
    day_span = (max(dates) - min(dates)).days
    if day_span < MIN_WINDOW_DAYS:
        return x_raw, dates
    window_len = random.randint(MIN_WINDOW_DAYS, min(MAX_WINDOW_DAYS, day_span))
    start_offset = random.randint(0, day_span - window_len)
    window_start = min(dates) + timedelta(days=start_offset)
    window_end = window_start + timedelta(days=window_len)
    mask = [window_start <= d <= window_end for d in dates]
    if sum(mask) < 5:
        return x_raw, dates
    return x_raw[mask], [d for d, m in zip(dates, mask) if m]


class ParcelSequenceDataset(Dataset):
    def __init__(self, raw_sequences, dates_list, labels, enable_windowing=False):
        self.raw_sequences, self.dates_list, self.labels = raw_sequences, dates_list, labels
        self.enable_windowing = enable_windowing

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x_raw, dates = self.raw_sequences[idx], self.dates_list[idx]
        if self.enable_windowing:
            x_raw, dates = _random_window(x_raw, dates)
        return torch.tensor(preprocess_sequence(x_raw), dtype=torch.float32), int(self.labels[idx])


class FixedDrawDataset(Dataset):
    \"\"\"Test/validation: resample-once-reuse-forever IS correct here (unlike train) —
    a stable set is what makes an epoch-over-epoch val_acc curve interpretable.\"\"\"
    def __init__(self, raw_sequences, labels, seed=42):
        rng_state = np.random.get_state()
        np.random.seed(seed)
        self.x = np.stack([preprocess_sequence(seq) for seq in raw_sequences])
        np.random.set_state(rng_state)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx], dtype=torch.float32), int(self.labels[idx])""")

add(CODE, """# ---- Model (copied from backend/app/models/architecture.py, not imported —
# importing the deployed app package would pull FastAPI into this notebook) ----

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1), :])


class ImprovedTimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=64, nhead=4, num_layers=3, dim_feedforward=64, dropout=0.2):
        super().__init__()
        self.input_projection = nn.Sequential(nn.Linear(input_dim, d_model), nn.LayerNorm(d_model))
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, num_classes)
        )
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        return self.classifier(x.mean(dim=1))""")

add(CODE, """# ---- Load all fetched parcels, train/test split, train ----
INCLUDE_BREIZHCROPS = True  # fold-in by default per plan; flip to False to exclude

raw_sequences, dates_list, labels = [], [], []
checkpoint_files = sorted(CHECKPOINT_DIR.glob("*_fetched.pkl"))
if not INCLUDE_BREIZHCROPS:
    checkpoint_files = [p for p in checkpoint_files if not p.name.startswith("BZH_")]

for ckpt_path in checkpoint_files:
    with open(ckpt_path, "rb") as f:
        results = pickle.load(f)
    n_before = len(raw_sequences)
    for v in results.values():
        if v["x_raw"].shape[0] < 1:
            continue
        raw_sequences.append(v["x_raw"])
        dates_list.append(v["dates"])
        labels.append(CLASS_NAME_TO_ID[v["classname"]])
    print(f"  loaded {len(raw_sequences) - n_before} parcels from {ckpt_path.name}")

labels = np.array(labels, dtype=np.int64)
print(f"Total: {len(raw_sequences)} parcels, {len(set(labels.tolist()))} classes present")

indices = np.arange(len(labels))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=labels)

train_raw = [raw_sequences[i] for i in train_idx]
train_dates = [dates_list[i] for i in train_idx]
train_labels = labels[train_idx]
test_raw = [raw_sequences[i] for i in test_idx]
test_labels = labels[test_idx]

train_dataset = ParcelSequenceDataset(train_raw, train_dates, train_labels, enable_windowing=ENABLE_WINDOWING)
test_dataset = FixedDrawDataset(test_raw, test_labels)
print(f"train: {len(train_dataset)}  test: {len(test_dataset)}  windowing={'ON' if ENABLE_WINDOWING else 'off'}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    print("\\n*** No GPU detected — Settings (right sidebar) -> Accelerator -> select a GPU, then re-run from here. ***\\n")
else:
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

EPOCHS = 20
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

model = ImprovedTimeSeriesTransformer(input_dim=INPUT_DIM, num_classes=NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = OneCycleLR(optimizer, max_lr=0.005, epochs=EPOCHS, steps_per_epoch=len(train_loader), pct_start=0.3)

best_acc = -1.0  # NOT 0.0 — if val_acc never exceeds 0 on any epoch, "> 0.0" would
# never trigger and no checkpoint would ever be saved, crashing the load below.
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(batch_x), batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        epoch_loss += loss.item()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            _, predicted = torch.max(model(batch_x), 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    val_acc = correct / total
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), SAVE_PATH)
    print(f"epoch {epoch + 1}/{EPOCHS}  loss={epoch_loss / len(train_loader):.4f}  val_acc={val_acc:.4f}")

print(f"best val acc: {best_acc:.4f}, saved to {SAVE_PATH}")
model.load_state_dict(torch.load(SAVE_PATH, map_location=device))""")

add(CODE, """# ---- Per-class metrics harness — nothing in this project previously reported
# anything beyond overall accuracy. Two evaluations: single-draw (matches the
# val_acc curve above) and ensembled (K-draw softmax average per parcel, mirroring
# the deployed predict()'s approach in backend/app/models/inference.py — the more
# honest "real" accuracy number since it matches production behavior). ----
INFERENCE_ENSEMBLE_SIZE = 16  # matches backend/app/config.py's deployed default


def _report(y_true, y_pred, label):
    present_ids = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    target_names = [CLASS_NAMES[i] for i in present_ids]
    print(f"\\n=== {label} evaluation ===")
    print(classification_report(y_true, y_pred, labels=present_ids, target_names=target_names, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=present_ids)
    print("confusion matrix (rows=true, cols=predicted):")
    print("            " + " ".join(f"{n[:8]:>8s}" for n in target_names))
    for i, row in enumerate(cm):
        print(f"{target_names[i]:12s}" + " ".join(f"{v:8d}" for v in row))
    print(f"\\noverall accuracy: {(y_true == y_pred).mean():.4f}")


model.eval()
x_fixed = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))]).to(device)
y_true_fixed = np.array([test_dataset[i][1] for i in range(len(test_dataset))])
with torch.no_grad():
    y_pred_fixed = torch.argmax(model(x_fixed), dim=1).cpu().numpy()
_report(y_true_fixed, y_pred_fixed, "single-draw")

y_pred_ens = []
with torch.no_grad():
    for x_raw in test_raw:
        draws = np.stack([preprocess_sequence(x_raw) for _ in range(INFERENCE_ENSEMBLE_SIZE)])
        probs = torch.softmax(model(torch.tensor(draws, dtype=torch.float32).to(device)), dim=1).mean(dim=0)
        y_pred_ens.append(int(torch.argmax(probs).item()))
_report(test_labels, np.array(y_pred_ens), "ensembled")""")

add(MD, """## Next steps after this finishes

- `multicountry_india_18class.pth` is NOT a drop-in replacement for the currently
  deployed checkpoint — `NUM_CLASSES` changed (11 vs. 9) and the class scheme itself
  changed (e.g. `orchards`/`nuts` -> `fruit`/`nuts` with different membership, new
  classes like `vineyards`/`potatoes`/`triticale`). Deploying this requires updating
  `backend/app/config.py`'s `CLASS_NAMES`/`NUM_CLASSES` and anything downstream that
  assumes the old 9-class scheme (frontend labels, `CLASS_COLORS`, etc.) — not just
  swapping the checkpoint filename.
- Compare this model's per-class metrics above against the currently deployed
  checkpoint's known failure modes (e.g. the `nuts` -> `wheat` misclassification found
  during live testing, documented in `progress.md`) to confirm measurable improvement
  before replacing the deployed checkpoint, not just a different set of failure modes.
- `ENABLE_WINDOWING` was left off for this run by design — evaluate it as a separate,
  fetch-free experiment (flip the flag, re-run training on the same already-fetched
  checkpoints) once you want to isolate its effect specifically.""")

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out_path = "eurocrops_model_training_kaggle_multicountry.ipynb"
with open(out_path, "w") as f:
    json.dump(nb, f, indent=1)
print(f"wrote {out_path}, {len(cells)} cells")
