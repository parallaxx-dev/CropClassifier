"""
config.py — Shared constants for the multi-country EuroCrops retraining pipeline.

Country table verified directly against the actual Zenodo API listing for record
10118572 (v10, published 2023-11-13 — the current latest version; NOT the same as
the v9 record 8229128 that train_eurocrops_local.py's single-country script still
points at. v10 adds Czech Republic and Germany-Brandenburg on top of v9's content;
every other file's size matches exactly between the two, confirmed by cross-checking
both records directly, not assumed). See plan doc for the verification trail:
/home/parallax/.claude/plans/see-for-re-training-the-cheerful-bee.md
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

# ============================================
# Paths
# ============================================
PIPELINE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PIPELINE_DIR.parent / "eurocrops_output" / "multi_country"
SCRATCH_DIR = OUTPUT_DIR / "scratch"  # raw zips/extracted shapefiles — deleted per-country, not precious
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"  # per-country fetched-parcel pickles — precious, keep
SAVE_PATH = OUTPUT_DIR / "best_transformer_multicountry.pth"

for d in (SCRATCH_DIR, CHECKPOINT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ============================================
# CDSE Sentinel Hub Statistical API credentials — same .env as the deployed backend
# and the single-country training script (backend/.env, gitignored).
# ============================================
CDSE_SH_CLIENT_ID = os.environ.get("CDSE_SH_CLIENT_ID")
CDSE_SH_CLIENT_SECRET = os.environ.get("CDSE_SH_CLIENT_SECRET")
CDSE_SH_BASE_URL = "https://sh.dataspace.copernicus.eu"
CDSE_SH_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

# ============================================
# Sequence preprocessing — must match the deployed backend exactly
# ============================================
TARGET_SEQ_LEN = 45
INPUT_DIM = 13
SENTINEL2_L1C_BANDS = [
    "B1", "B10", "B11", "B12", "B2", "B3", "B4",
    "B5", "B6", "B7", "B8", "B8A", "B9",
]

# ============================================
# Cloud masking — identical to backend/app/services/sentinel_fetch.py
# ============================================
MAX_CLOUD_COVER_PERCENT = 60
MIN_AOI_CLEAR_FRACTION = 0.4
REFLECTANCE_SCALE = 1e-4
RESOLUTION_METERS = 10

# ============================================
# Sampling
# ============================================
N_PER_CLASS_PER_COUNTRY = int(os.environ.get("N_PER_CLASS_PER_COUNTRY", 150))
OVERSAMPLE_PER_HCAT_CODE = max(50, N_PER_CLASS_PER_COUNTRY)  # per-code read cap, bounded-memory read pattern

# ============================================
# Fetch concurrency / rate limiting
# ============================================
# CDSE's documented cap (already established via the existing single-country script's
# own comment: "stay comfortably under CDSE's 300 req/min cap").
CDSE_RATE_LIMIT_PER_MIN = 300
RATE_LIMITER_TARGET_PER_SEC = 4.5  # deliberately under the 5 req/sec nominal ceiling
RATE_LIMITER_BURST_CAPACITY = 8

# MAX_WORKERS: NOT sized from the naive Little's Law formula (workers ~= target_rate *
# avg_latency) — that assumed clean linear scaling up to CDSE's stated 300/min cap, which
# empirically does not hold. Phase 0's actual measurement (phase0_probe.py, run against
# real CDSE traffic) found:
#   - serial baseline: ~4.3s avg latency, 15/15 clean
#   - 8 workers, first 15 requests: clean, 101 req/min, no rate-limit warnings
#   - 8-19 workers sustained over 25-40 requests: the sentinelhub SDK's own client-side
#     rate limiter (SHRateLimitWarning) kicks in and backs off hard — latencies balloon
#     from ~4s to 30-70s, effective throughput COLLAPSES to 20-35 req/min (worse than
#     fewer workers), even at 6 workers sustained over 30 requests (51 warnings observed).
#   - 0 hard errors throughout ~140 test requests — the SDK's backoff never fails, just
#     gets much slower under sustained concurrent load.
# Conclusion: more workers does not mean more throughput past a fairly low ceiling for
# this API, and pushing it costs latency instead of buying rate. Deliberately conservative:
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", 5))

CHECKPOINT_EVERY_N_COMPLETIONS = 50
CHECKPOINT_EVERY_N_SECONDS = 60
# Generous: real observed latencies under backoff reached 74s in testing. A tight timeout
# here would abort requests that were about to succeed, turning a slowdown into data loss.
FETCH_TIMEOUT_SECONDS = 180
FETCH_MAX_RETRIES = 4

# ============================================
# Country table — 18 usable EuroCrops countries/regions (Romania excluded: its zip
# is named RO_ny.zip / "no year", meaning the parcel-declaration year that satellite
# imagery needs to be matched against is unconfirmed — pairing without it risks
# silently mispairing imagery to the wrong season).
#
# zenodo_filename is the exact filename in record 10118572's file listing (case-
# sensitive, verified directly against the API response — some countries use
# lowercase-with-underscore region codes, not a uniform pattern, so don't guess).
# ============================================


@dataclass(frozen=True)
class CountryEntry:
    code: str  # short identifier used in checkpoint filenames, logs
    zenodo_filename: str
    year: int
    approx_size_mb: int


COUNTRIES: list[CountryEntry] = [
    CountryEntry("AT", "AT_2021.zip", 2021, 844),
    CountryEntry("BE_VLG", "BE_VLG_2021.zip", 2021, 140),
    CountryEntry("CZ", "CZ_2023.zip", 2023, 553),
    CountryEntry("DE_BB", "DE_BB_2023.zip", 2023, 103),
    CountryEntry("DE_LS", "DE_LS_2021.zip", 2021, 241),
    CountryEntry("DE_NRW", "DE_NRW_2021.zip", 2021, 277),
    CountryEntry("DK", "DK_2019.zip", 2019, 297),
    CountryEntry("EE", "EE_2021.zip", 2021, 287),
    CountryEntry("ES_NA", "ES_NA_2020.zip", 2020, 468),
    CountryEntry("FR", "FR_2018.zip", 2018, 2630),
    CountryEntry("HR", "HR_2020.zip", 2020, 351),
    CountryEntry("LT", "LT_2021.zip", 2021, 545),
    CountryEntry("LV", "LV_2021.zip", 2021, 547),
    CountryEntry("NL", "NL_2020.zip", 2020, 283),
    CountryEntry("PT", "PT.zip", 2021, 37),
    CountryEntry("SE", "SE_2021.zip", 2021, 440),
    CountryEntry("SI", "SI_2021.zip", 2021, 213),
    CountryEntry("SK", "SK_2021.zip", 2021, 311),
]
# Excluded pending a confirmed reference year: Romania (RO_ny.zip)

ZENODO_RECORD_ID = "10118572"


def zenodo_url(filename: str) -> str:
    return f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}/files/{filename}/content"


# ============================================
# BreizhCrops (separate, non-EuroCrops shapefile — no EC_hcat_* columns, uses raw
# French RPG codes directly; see taxonomy.py's RPG_CODE_TO_CLASS)
# ============================================
BREIZHCROPS_SHP_PATH = (
    PIPELINE_DIR.parent.parent / "breizhcrops_dataset" / "2017" / "frh04.shp"
)
BREIZHCROPS_YEAR = 2017
