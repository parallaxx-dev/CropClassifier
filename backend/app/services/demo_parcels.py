"""
demo_parcels.py — a small, spatially-clustered ("neighbouring") sample of
parcels per trained region, each carrying both a true label and a predicted
label, for the "Live Validation" map tab (see routers/demo.py). Professors
can see real predictions next to ground truth on a map without running the
full pipeline over every parcel in every region -- slow, and pointless for a
spot-check demo; a representative cluster per region shows real behavior,
including real mistakes, just as well.

Two-stage design, matching parcels.py/model_stats.py's own split between
"static definition" and "state that changes":
- select_demo_parcels() deterministically (seeded) picks WHICH parcels are
  in the demo, so the same set survives a server restart.
- The predictions themselves are cached in app/data/demo_predictions.json
  (not Redis/a DB -- small, hand-curated, infrequently-updated data, same
  tier as app/data/model_stats.py; see CLAUDE.md's "no database yet,
  intentionally") and are only ever refreshed by an explicit live
  re-predict call (POST /api/v1/demo/repredict), not on every page load.
"""

import json
from datetime import date
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from app.config import get_color
from app.models.inference import predict as run_prediction
from app.services import sentinel_fetch
from app.services.custom_parcels import CUSTOM_REGIONS, load_custom_region
from app.services.parcels import CHECKPOINT_DIR, REGION_NAMES
from app.services.validation import MAX_AREA_HECTARES, MIN_AREA_HECTARES, approx_area_hectares

CLUSTER_SEED = 0

# Per-class quota for the demo, weighted by how well the deployed checkpoint
# actually handles each class (see app/data/model_stats.py's per_class F1) --
# not a flat N-per-region split. Direct request: the demo should mostly show
# high-confidence correct predictions, with a few weaker ones, but still
# cover every crop type -- a demo that's either all cherry-picked easy wins
# or all misses on data-starved classes would misrepresent what this model
# does either way. More slots for classes the model is actually good at
# (F1 >= ~0.6), one slot each for weak/zero-F1 classes so they're still
# visible -- those will mostly show as real misses, which is honest, not a
# bug to hide.
CLASS_QUOTA = {
    "rapeseed": 2, "maize": 2, "potatoes": 2, "barley": 2,
    "sunflower": 2, "vineyards": 2, "wheat": 2, "meadow": 2,
    "triticale": 1, "fruit": 1, "sugarcane": 1, "fallow": 1,
    "nuts": 1, "rice": 1, "mustard": 1,
    "lentil": 1, "gram": 1, "garlic": 1,
}

# Full Jan1-Dec31 span for each region's own real declared/fetched year --
# matches training exactly (a partial range measurably degrades accuracy,
# see CLAUDE.md's known-gaps item 4 -- the demo must not repeat that
# mistake). Kept in sync by hand with training/eurocrops_pipeline/config.py's
# COUNTRIES table + BREIZHCROPS_YEAR, agrifieldnet.py's AGRIFIELDNET_YEAR,
# and run_custom.py's CUSTOM_YEAR. Also doubles as "which regions are in the
# demo" -- exactly the regions the deployed checkpoint was actually trained
# on (see app/data/model_stats.py's training_data), not every region with
# parcel geometries on disk (DK/CZ have geometries but weren't trained on,
# so a mismatch there wouldn't demonstrate what this model does).
REGION_YEAR = {
    "AT": 2021,
    "BE_VLG": 2021,
    "DE_BB": 2023,
    "DE_LS": 2021,
    "DE_NRW": 2021,
    "IN": 2024,
    "CUSTOM": 2025,
    "BZH": 2017,
}

_CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "demo_predictions.json"


def _region_display_name(code: str) -> str:
    if code in CUSTOM_REGIONS:
        return CUSTOM_REGIONS[code]["display_name"]
    return REGION_NAMES.get(code, code)


def _load_region_gdf(code: str) -> gpd.GeoDataFrame:
    if code in CUSTOM_REGIONS:
        gdf = load_custom_region(CUSTOM_REGIONS[code])
    else:
        # GeoJSON, not the training pipeline's own *_sampled_geoms.pkl -- see
        # parcels.py's module docstring for why (pickle's cross-version
        # incompatibility is a real failure this project hit in production).
        geoms_path = CHECKPOINT_DIR / f"{code}_sampled_geoms.geojson"
        if not geoms_path.exists():
            return gpd.GeoDataFrame(columns=["classname", "geometry"], geometry="geometry", crs=4326)
        gdf = gpd.read_file(geoms_path)

    if len(gdf) == 0:
        return gdf
    # Same area bounds POST /predict enforces (services/validation.py) --
    # picking a parcel too small/large for the live endpoint to accept would
    # just show up as a dead "error" cell in the demo, not a real mistake.
    areas = gdf.geometry.apply(approx_area_hectares)
    return gdf[(areas >= MIN_AREA_HECTARES) & (areas <= MAX_AREA_HECTARES)]


def _load_all_regions_pooled() -> gpd.GeoDataFrame:
    """Every trained region's parcels concatenated into one pool, tagged
    with their source region -- selection is driven by CLASS_QUOTA now, not
    per-region clustering, since several classes (mustard/gram/garlic in
    India, rapeseed/barley across the EU regions) are concentrated in
    specific regions and wouldn't be reachable from a fixed N-per-region
    pick."""
    parts = []
    for code in REGION_YEAR:
        gdf = _load_region_gdf(code)
        if len(gdf) == 0:
            continue
        gdf = gdf.copy()
        gdf["region"] = code
        parts.append(gdf)
    if not parts:
        return gpd.GeoDataFrame(columns=["classname", "geometry", "region"], geometry="geometry", crs=4326)
    return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=parts[0].crs)


def select_demo_parcels() -> list[dict]:
    """Deterministic (seeded) list of {id, region, region_name, geometry,
    classname} dicts, drawn per CLASS_QUOTA -- the WHICH, independent of any
    prediction."""
    pooled = _load_all_regions_pooled()
    selected = []
    for classname, quota in CLASS_QUOTA.items():
        class_pool = pooled[pooled["classname"] == classname]
        if len(class_pool) == 0:
            continue
        sample = class_pool.sample(n=min(quota, len(class_pool)), random_state=CLUSTER_SEED)
        for idx, row in sample.iterrows():
            code = row["region"]
            selected.append(
                {
                    "id": f"{code}_{idx}",
                    "region": code,
                    "region_name": _region_display_name(code),
                    "geometry": row.geometry,
                    "classname": row["classname"],
                }
            )
    return selected


def _load_cache() -> dict:
    if not _CACHE_PATH.exists():
        return {}
    with open(_CACHE_PATH) as f:
        return json.load(f)


def _save_cache(cache: dict) -> None:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def run_predictions(parcels: list[dict], model, device=None) -> dict:
    """Live-fetches + predicts each parcel in turn (sequential -- this is a
    small, occasional, manually-triggered demo action, not a high-frequency
    endpoint, so a Celery/async queue isn't justified yet, same reasoning
    CLAUDE.md's Stage 3 plan already documents). Per-parcel failures (a CDSE
    hiccup, no usable scenes) are caught and recorded rather than aborting
    the whole batch -- one flaky parcel shouldn't blank out the other 20."""
    cache = _load_cache()
    for p in parcels:
        year = REGION_YEAR.get(p["region"])
        try:
            x_raw, dates = sentinel_fetch.fetch_time_series(p["geometry"], date(year, 1, 1), date(year, 12, 31))
            if len(dates) == 0:
                raise RuntimeError("no usable Sentinel-2 scenes for this parcel/year")
            result = run_prediction(model, x_raw)
            cache[p["id"]] = {
                "predicted_class": result["pred_label"],
                "confidence": result["confidence"],
                "observations_used": len(dates),
                "error": None,
            }
        except Exception as exc:
            cache[p["id"]] = {
                "predicted_class": None,
                "confidence": None,
                "observations_used": None,
                "error": str(exc),
            }
    _save_cache(cache)
    return cache


def get_demo_geojson() -> dict:
    parcels = select_demo_parcels()
    cache = _load_cache()
    features = []
    for p in parcels:
        pred = cache.get(p["id"], {})
        predicted_class = pred.get("predicted_class")
        features.append(
            {
                "type": "Feature",
                "geometry": p["geometry"].__geo_interface__,
                "properties": {
                    "id": p["id"],
                    "region": p["region"],
                    "region_name": p["region_name"],
                    "classname": p["classname"],
                    "color": get_color(p["classname"]),
                    "area_hectares": approx_area_hectares(p["geometry"]),
                    "predicted_class": predicted_class,
                    "predicted_color": get_color(predicted_class) if predicted_class else None,
                    "confidence": pred.get("confidence"),
                    "observations_used": pred.get("observations_used"),
                    "error": pred.get("error"),
                    "match": (predicted_class == p["classname"]) if predicted_class else None,
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}
