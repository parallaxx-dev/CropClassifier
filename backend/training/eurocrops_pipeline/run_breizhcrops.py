"""
run_breizhcrops.py — Phase 4: fetch BreizhCrops (Brittany, 2017) live via the
same CDSE pipeline used for the 18 EuroCrops countries, instead of the stale
prepackaged breizhcrops archive (see taxonomy.py's module docstring for why).

Own checkpoint file, kept deliberately separate from the EuroCrops fetch
outputs — whether to fold this into the combined training set or treat it as a
future fine-tuning pass stays a train-time choice (run_train.py), not something
baked into the fetch step.

Run: ../.venv/bin/python run_breizhcrops.py [--dry-run]
"""

import argparse
import pickle
from datetime import date

import fetch
import sampling
from config import BREIZHCROPS_SHP_PATH, BREIZHCROPS_YEAR, CHECKPOINT_DIR
from fetch import fetch_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not BREIZHCROPS_SHP_PATH.exists():
        raise FileNotFoundError(f"BreizhCrops shapefile not found: {BREIZHCROPS_SHP_PATH}")

    geoms_ckpt = CHECKPOINT_DIR / "BZH_sampled_geoms.pkl"
    fetch_ckpt = CHECKPOINT_DIR / "BZH_fetched.pkl"

    if geoms_ckpt.exists():
        with open(geoms_ckpt, "rb") as f:
            sampled = pickle.load(f)
        print(f"[BZH] reusing already-sampled geometries ({len(sampled)} parcels)")
    else:
        schema_probe = sampling.gpd.read_file(BREIZHCROPS_SHP_PATH, rows=500)
        if "CODE_CULTU" not in schema_probe.columns:
            raise ValueError(f"BreizhCrops shapefile missing CODE_CULTU column: {schema_probe.columns.tolist()}")
        sampled = sampling.sample_breizhcrops(BREIZHCROPS_SHP_PATH)
        fetch.save_geoms_checkpoint(sampled, geoms_ckpt)

    if args.dry_run:
        print(f"[BZH] dry-run: would fetch {len(sampled)} parcels, stopping here")
        return

    fetch_all(sampled, "BZH", date(BREIZHCROPS_YEAR, 1, 1), date(BREIZHCROPS_YEAR, 12, 31), fetch_ckpt)


if __name__ == "__main__":
    main()
