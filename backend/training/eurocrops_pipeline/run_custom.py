"""
run_custom.py — fetch Sentinel-2 time series for the hand-labeled CUSTOM
region (backend/local_parcels/: wheat/mustard fields near Varanasi + one
meadow near Jabalpur) so it can be folded into the multi-country training
set via the normal *_fetched.pkl path, same as any other source.

Reuses fetch.py's fetch_all() (checkpointing, rate limiting, retries)
unchanged -- only the parcel source differs: custom_parcels.py's static
registry instead of a downloaded country shapefile, so no shapefile
download/sampling step is needed here.

Run: ../.venv/bin/python run_custom.py [--dry-run]
"""

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # backend/ on sys.path, for app.*
from app.services.custom_parcels import CUSTOM_REGIONS, load_custom_region  # noqa: E402

from config import CHECKPOINT_DIR  # noqa: E402
from fetch import fetch_all  # noqa: E402

# Most recent complete calendar year at write time. Every other region in
# this pipeline is fetched as a full Jan1-Dec31 span (see config.py's
# per-country `year`) -- these parcels aren't tied to a declaration year the
# way EuroCrops/AgriFieldNet are, so matching that convention avoids
# introducing yet another date-range shape (see CLAUDE.md's known gap #4).
CUSTOM_YEAR = 2025


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="load geometries but don't fetch")
    args = parser.parse_args()

    gdf = load_custom_region(CUSTOM_REGIONS["CUSTOM"])
    print(f"[CUSTOM] {len(gdf)} hand-labeled parcels: {gdf['classname'].value_counts().to_dict()}")

    if args.dry_run:
        print("[CUSTOM] dry-run: stopping before fetch")
        return

    fetch_ckpt = CHECKPOINT_DIR / "CUSTOM_fetched.pkl"
    fetch_all(gdf, "CUSTOM", date(CUSTOM_YEAR, 1, 1), date(CUSTOM_YEAR, 12, 31), fetch_ckpt)


if __name__ == "__main__":
    main()
