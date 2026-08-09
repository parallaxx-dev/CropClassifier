"""
regions.py — country/region identity as an auxiliary model input, alongside
the existing per-class taxonomy.

Motivation (empirically demonstrated this session, not theoretical): the same
crop's Sentinel-2 growth curve differs meaningfully by region -- India's Rabi
wheat (sown ~Nov, harvested ~Mar-Apr) produced 0/9 correct wheat predictions
from a model trained only on European winter wheat. A single shared `wheat`
class forces the model to reconcile very different growth curves under one
label with no signal about which curve to expect. `REGION_NAMES` gives the
model an explicit region identity to condition on.

IDs are assigned from config.COUNTRIES' order (the full 18-country EuroCrops
plan) plus BZH (BreizhCrops) and IN (AgriFieldNet India), REGARDLESS of which
ones have actually been fetched yet -- sized generously up front so adding a
new EuroCrops country later doesn't force another embedding-table resize (the
same architecture-change pain NUM_CLASSES already went through twice this
session). Rows for not-yet-fetched countries simply stay near their random
initialization until real data trains them; nothing breaks by leaving them
unused.
"""

import pickle
from pathlib import Path

from config import COUNTRIES

REGION_NAMES: dict[int, str] = {i: c.code for i, c in enumerate(COUNTRIES)}
REGION_NAMES[len(COUNTRIES)] = "BZH"
REGION_NAMES[len(COUNTRIES) + 1] = "IN"
REGION_NAME_TO_ID: dict[str, int] = {v: k for k, v in REGION_NAMES.items()}
NUM_REGIONS = len(REGION_NAMES)


def region_id_for_checkpoint(checkpoint_filename: str) -> int:
    """'AT_fetched.pkl' -> REGION_NAME_TO_ID['AT']. Checkpoint filenames are the
    only place the fetch pipeline records which source a parcel came from."""
    code = checkpoint_filename.removesuffix("_fetched.pkl")
    if code not in REGION_NAME_TO_ID:
        raise ValueError(f"unknown region code {code!r} parsed from {checkpoint_filename!r}")
    return REGION_NAME_TO_ID[code]


def compute_real_bboxes(checkpoint_dir: Path) -> dict[str, tuple[float, float, float, float]]:
    """(min_lon, min_lat, max_lon, max_lat) per region, computed directly from
    that region's actual *_sampled_geoms.pkl -- only for regions that have
    been sampled/fetched. Used to build the deployed API's region lookup
    table; never guessed/hardcoded for a country we have no real geometry
    for."""
    bboxes = {}
    for geoms_path in Path(checkpoint_dir).glob("*_sampled_geoms.pkl"):
        code = geoms_path.name.removesuffix("_sampled_geoms.pkl")
        with open(geoms_path, "rb") as f:
            gdf = pickle.load(f)
        if len(gdf) == 0:
            continue
        min_lon, min_lat, max_lon, max_lat = gdf.total_bounds.tolist()
        bboxes[code] = (min_lon, min_lat, max_lon, max_lat)
    return bboxes


if __name__ == "__main__":
    from config import CHECKPOINT_DIR
    print(f"{NUM_REGIONS} region slots reserved: {REGION_NAMES}")
    print(f"Real bounding boxes (regions actually fetched so far):")
    for code, bbox in compute_real_bboxes(CHECKPOINT_DIR).items():
        print(f"  {code}: {bbox}")
