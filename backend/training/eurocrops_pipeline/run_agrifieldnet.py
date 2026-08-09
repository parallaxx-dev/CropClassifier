"""
run_agrifieldnet.py — fetch AgriFieldNet India (Bihar/UP/Rajasthan/Odisha)
live via the same CDSE pipeline used for the 18 EuroCrops countries and
BreizhCrops. See taxonomy.py's AGRIFIELDNET_CROP_ID_TO_CLASS and
agrifieldnet.py's module docstring for the class-mapping and data-source
rationale (why we fetch our own time series instead of using AgriFieldNet's
bundled single-composite imagery).

No GPU touched -- safe to interrupt and resume (per-field-source checkpoints,
same as every other source in this pipeline).

Run: ../.venv/bin/python run_agrifieldnet.py [--n-per-class 150] [--dry-run]
"""

import argparse
import pickle
from datetime import date

import agrifieldnet
from config import CHECKPOINT_DIR
from fetch import fetch_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-class", type=int, default=150)
    parser.add_argument("--dry-run", action="store_true", help="sample but don't fetch")
    args = parser.parse_args()

    geoms_ckpt = CHECKPOINT_DIR / "IN_sampled_geoms.pkl"
    fetch_ckpt = CHECKPOINT_DIR / "IN_fetched.pkl"

    if geoms_ckpt.exists():
        with open(geoms_ckpt, "rb") as f:
            sampled = pickle.load(f)
        print(f"[IN] reusing already-sampled geometries ({len(sampled)} parcels)")
    else:
        sampled = agrifieldnet.sample_fields(n_per_class=args.n_per_class)
        with open(geoms_ckpt, "wb") as f:
            pickle.dump(sampled, f)

    if args.dry_run:
        print(f"[IN] dry-run: would fetch {len(sampled)} parcels, stopping here")
        return

    fetch_all(
        sampled, "IN",
        date(agrifieldnet.AGRIFIELDNET_YEAR, 1, 1),
        date(agrifieldnet.AGRIFIELDNET_YEAR, 12, 31),
        fetch_ckpt,
    )


if __name__ == "__main__":
    main()
