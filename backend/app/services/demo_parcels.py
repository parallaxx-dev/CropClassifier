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
import pickle
from datetime import date
from pathlib import Path

import geopandas as gpd
import numpy as np

from app.config import get_color
from app.models.inference import predict as run_prediction
from app.services import sentinel_fetch
from app.services.custom_parcels import CUSTOM_REGIONS, load_custom_region
from app.services.parcels import CHECKPOINT_DIR, REGION_NAMES
from app.services.validation import MAX_AREA_HECTARES, MIN_AREA_HECTARES, approx_area_hectares

N_PER_REGION = 3
CLUSTER_SEED = 0

# Full Jan1-Dec31 span for each region's own real declared/fetched year --
# matches training exactly (a partial range measurably degrades accuracy,
# see CLAUDE.md's known-gaps item 4 -- the demo must not repeat that
# mistake). Kept in sync by hand with training/eurocrops_pipeline/config.py's
# COUNTRIES table, agrifieldnet.py's AGRIFIELDNET_YEAR, and
# run_custom.py's CUSTOM_YEAR. Also doubles as "which regions are in the
# demo" -- exactly the 7 regions the deployed checkpoint was actually
# trained on (see app/data/model_stats.py's training_data), not every region
# with parcel geometries on disk (BZH/DK/CZ have geometries but weren't
# trained on, so a mismatch there wouldn't demonstrate what this model does).
REGION_YEAR = {
    "AT": 2021,
    "BE_VLG": 2021,
    "DE_BB": 2023,
    "DE_LS": 2021,
    "DE_NRW": 2021,
    "IN": 2024,
    "CUSTOM": 2025,
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
        geoms_path = CHECKPOINT_DIR / f"{code}_sampled_geoms.pkl"
        if not geoms_path.exists():
            return gpd.GeoDataFrame(columns=["classname", "geometry"], geometry="geometry", crs=4326)
        with open(geoms_path, "rb") as f:
            gdf = pickle.load(f)

    if len(gdf) == 0:
        return gdf
    # Same area bounds POST /predict enforces (services/validation.py) --
    # picking a parcel too small/large for the live endpoint to accept would
    # just show up as a dead "error" cell in the demo, not a real mistake.
    areas = gdf.geometry.apply(approx_area_hectares)
    return gdf[(areas >= MIN_AREA_HECTARES) & (areas <= MAX_AREA_HECTARES)]


def _nearest_cluster(gdf: gpd.GeoDataFrame, n: int, seed: int) -> gpd.GeoDataFrame:
    """Anchor + its n-1 nearest neighbors by centroid distance -- real,
    physically neighbouring fields, not a random scatter across the whole
    country. Degrades gracefully to "all of them" if the region has <= n."""
    if len(gdf) <= n:
        return gdf
    rng = np.random.default_rng(seed)
    anchor_idx = int(rng.integers(0, len(gdf)))
    centroids = gdf.geometry.centroid
    anchor = centroids.iloc[anchor_idx]
    dist = centroids.distance(anchor)
    nearest_index = dist.nsmallest(n).index
    return gdf.loc[nearest_index]


def select_demo_parcels(n_per_region: int = N_PER_REGION) -> list[dict]:
    """Deterministic list of {id, region, region_name, geometry, classname}
    dicts -- the WHICH, independent of any prediction."""
    selected = []
    for code in REGION_YEAR:
        gdf = _load_region_gdf(code)
        if len(gdf) == 0:
            continue
        cluster = _nearest_cluster(gdf, n_per_region, CLUSTER_SEED)
        for idx, row in cluster.iterrows():
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


def get_demo_geojson(n_per_region: int = N_PER_REGION) -> dict:
    parcels = select_demo_parcels(n_per_region)
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
