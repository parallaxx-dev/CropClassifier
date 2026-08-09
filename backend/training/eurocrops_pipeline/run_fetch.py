"""
run_fetch.py — CLI entry point for Phases 2-3: loop the verified country table,
download -> probe schema -> sample -> concurrent fetch -> cleanup, per country.

No GPU touched anywhere in this file — safe for a CPU-only Kaggle session, safe
to interrupt and resume across multiple sessions (per-country checkpoints for
both the sampled-geometry step and the fetch step; a country already fully
fetched is skipped entirely on a re-run).

Run: ../.venv/bin/python run_fetch.py [--country CODE ...] [--dry-run]
"""

import argparse
import pickle
from datetime import date

import download
import fetch
import sampling
from config import CHECKPOINT_DIR, COUNTRIES


def run_country(country, dry_run: bool = False) -> None:
    geoms_ckpt = CHECKPOINT_DIR / f"{country.code}_sampled_geoms.pkl"
    fetch_ckpt = CHECKPOINT_DIR / f"{country.code}_fetched.pkl"

    if fetch_ckpt.exists():
        with open(fetch_ckpt, "rb") as f:
            existing = pickle.load(f)
        # only truly done if every sampled parcel has a result — cheap check via
        # the sampled-geoms checkpoint, which is written before fetch starts
        if geoms_ckpt.exists():
            with open(geoms_ckpt, "rb") as f:
                sampled = pickle.load(f)
            if len(existing) >= len(sampled):
                print(f"[{country.code}] already fully fetched ({len(existing)} parcels) — skipping")
                return

    if geoms_ckpt.exists():
        with open(geoms_ckpt, "rb") as f:
            sampled = pickle.load(f)
        print(f"[{country.code}] reusing already-sampled geometries ({len(sampled)} parcels) "
              f"— skipping download/extract")
    else:
        shp_path = download.download_and_extract(country)
        schema = sampling.probe_schema(shp_path)
        print(f"[{country.code}] schema probe: {schema}")
        if schema["null_rate"] > 0.1:
            print(f"[{country.code}] *** WARNING: {schema['null_rate']:.1%} EC_hcat_c null rate — "
                  f"check this country's data quality before trusting its sample ***")

        sampled = sampling.sample_country(shp_path, country)
        with open(geoms_ckpt, "wb") as f:
            pickle.dump(sampled, f)
        download.cleanup(country)

    if dry_run:
        print(f"[{country.code}] dry-run: would fetch {len(sampled)} parcels, stopping here")
        return

    fetch.fetch_all(sampled, country.code, date(country.year, 1, 1), date(country.year, 12, 31), fetch_ckpt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", action="append", help="only run these country codes (repeatable)")
    parser.add_argument("--dry-run", action="store_true", help="sample but don't fetch")
    args = parser.parse_args()

    countries = COUNTRIES
    if args.country:
        wanted = set(args.country)
        countries = [c for c in COUNTRIES if c.code in wanted]
        missing = wanted - {c.code for c in countries}
        if missing:
            raise SystemExit(f"unknown country code(s): {missing}")

    for country in countries:
        print(f"\n{'=' * 60}\n{country.code} ({country.year})\n{'=' * 60}")
        try:
            run_country(country, dry_run=args.dry_run)
        except Exception as e:
            # One bad country's data (e.g. a missing .prj/CRS -- hit for real with
            # CZ) must not halt the whole multi-hour run. Free its scratch space
            # (extraction can be gigabytes) since the geoms checkpoint was never
            # written, so a later resume will cleanly retry it from a known state
            # rather than trip over a half-extracted directory.
            print(f"[{country.code}] *** SKIPPING after error: {type(e).__name__}: {e} ***")
            try:
                download.cleanup(country)
            except Exception:
                pass


if __name__ == "__main__":
    main()
