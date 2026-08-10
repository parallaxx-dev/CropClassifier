"""
parcels.py — serves real, already-fetched field parcel boundaries as GeoJSON,
for the frontend's parcel-browser page (pick a real, pre-outlined field
instead of drawing one).

Two independent sources, merged transparently:
1. The training pipeline's *_sampled_geoms.pkl checkpoints
   (backend/training/eurocrops_output/, gitignored -- local training-run
   output, not shipped with the repo). If that directory doesn't exist (a
   fresh clone that hasn't run any fetch yet), that source just contributes
   nothing rather than erroring -- the parcel browser degrades to "no
   parcels available yet" for those regions, not a crash.
2. custom_parcels.CUSTOM_REGIONS -- small, separately-registered
   user-provided datasets (e.g. hand-labeled test sets) that never went
   through the training pipeline at all. Each of these regions can itself be
   built from multiple source files (see custom_parcels.py).
"""

import pickle
from pathlib import Path

import geopandas as gpd

from app.config import get_color
from app.services.custom_parcels import CUSTOM_REGIONS, load_custom_region
from app.services.validation import approx_area_hectares

CHECKPOINT_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "training" / "eurocrops_output" / "multi_country" / "checkpoints"
)

# region code -> display name, kept in sync with training/eurocrops_pipeline/regions.py
REGION_NAMES = {
    "AT": "Austria",
    "BE_VLG": "Belgium-Flanders",
    "CZ": "Czech Republic",
    "DE_BB": "Germany-Brandenburg",
    "DE_LS": "Germany-Lower Saxony",
    "DE_NRW": "Germany-North Rhine-Westphalia",
    "DK": "Denmark",
    "BZH": "Brittany (France)",
    "IN": "India",
}


def list_regions() -> list[dict]:
    regions = []
    if CHECKPOINT_DIR.exists():
        for path in sorted(CHECKPOINT_DIR.glob("*_sampled_geoms.pkl")):
            code = path.name.removesuffix("_sampled_geoms.pkl")
            with open(path, "rb") as f:
                gdf = pickle.load(f)
            regions.append({"code": code, "name": REGION_NAMES.get(code, code), "count": len(gdf)})

    for code, entry in CUSTOM_REGIONS.items():
        gdf = load_custom_region(entry)
        if len(gdf) == 0:
            continue
        regions.append({"code": code, "name": entry["display_name"], "count": len(gdf)})

    return regions


def get_parcels_geojson(region: str | None = None, limit: int = 40) -> dict:
    """Real field boundaries (classname + geometry) as a GeoJSON
    FeatureCollection, capped at `limit` per region so the map stays
    responsive -- these are individual smallholder/farm field polygons, not
    simplified shapes, so a full region's worth (hundreds-1000+) would be a
    slow, cluttered map for no benefit."""
    features = []

    if CHECKPOINT_DIR.exists():
        paths = sorted(CHECKPOINT_DIR.glob("*_sampled_geoms.pkl"))
        if region:
            paths = [p for p in paths if p.name == f"{region}_sampled_geoms.pkl"]

        for path in paths:
            code = path.name.removesuffix("_sampled_geoms.pkl")
            with open(path, "rb") as f:
                gdf: gpd.GeoDataFrame = pickle.load(f)
            sample = gdf.sample(n=min(limit, len(gdf)), random_state=0) if len(gdf) > limit else gdf
            for idx, row in sample.iterrows():
                features.append({
                    "type": "Feature",
                    "geometry": row.geometry.__geo_interface__,
                    "properties": {
                        "id": f"{code}_{idx}",
                        "region": code,
                        "region_name": REGION_NAMES.get(code, code),
                        "classname": row["classname"],
                        "color": get_color(row["classname"]),
                        "area_hectares": approx_area_hectares(row.geometry),
                    },
                })

    for code, entry in CUSTOM_REGIONS.items():
        if region and region != code:
            continue
        gdf = load_custom_region(entry)
        if len(gdf) == 0:
            continue
        sample = gdf.sample(n=min(limit, len(gdf)), random_state=0) if len(gdf) > limit else gdf
        for idx, row in sample.iterrows():
            features.append({
                "type": "Feature",
                "geometry": row.geometry.__geo_interface__,
                "properties": {
                    "id": f"{code}_{idx}",
                    "region": code,
                    "region_name": entry["display_name"],
                    "classname": row["classname"],
                    "color": get_color(row["classname"]),
                    "area_hectares": approx_area_hectares(row.geometry),
                },
            })

    return {"type": "FeatureCollection", "features": features}
