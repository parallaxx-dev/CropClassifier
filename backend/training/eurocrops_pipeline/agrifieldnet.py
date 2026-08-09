"""
agrifieldnet.py — sample real field polygons + ground-truth crop labels from
the AgriFieldNet India Challenge dataset (Bihar/UP/Rajasthan/Odisha), publicly
mirrored (no auth needed) on Azure blob storage. Produces a GeoDataFrame with
the same ('classname', 'geometry') schema as sample_country()/
sample_breizhcrops(), so it plugs into the existing checkpointed fetch_all()
CDSE fetch unchanged.

We do NOT use AgriFieldNet's own Sentinel-2 imagery (a single cloud-free
composite per chip, not a time series -- our model needs TARGET_SEQ_LEN
growing-season observations). Instead we take only the field geometries +
ground-truth labels (from the label chips' field_ids.tif / raster_labels.tif)
and fetch our own live Sentinel-2 time series for those exact coordinates via
the same CDSE pipeline used everywhere else in this project.

Crop legend verified directly against the dataset's own Documentation.pdf
(not inferred): 1=Wheat, 2=Mustard, 3=Lentil, 4=No crop/Fallow, 5=Green pea,
6=Sugarcane, 8=Garlic, 9=Maize, 13=Gram, 14=Coriander, 15=Potato, 16=Bersem,
36=Rice. See taxonomy.py's AGRIFIELDNET_CROP_ID_TO_CLASS for which of these
map to a target class (Green pea/Coriander/Bersem are deliberately excluded
-- single-digit example counts across the ENTIRE public dataset, surveyed
directly, not a partial-fetch gap that improves later).
"""

import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import requests
from rasterio.features import shapes
from rasterio.warp import transform_geom
from shapely.geometry import mapping, shape

from config import SCRATCH_DIR
from taxonomy import AGRIFIELDNET_CROP_ID_TO_CLASS

_BLOB_BASE = "https://radiantearth.blob.core.windows.net/mlhub/ref_agrifieldnet_competition_v1/train_labels"
_LIST_URL = "https://radiantearth.blob.core.windows.net/mlhub?restype=container&comp=list&prefix=ref_agrifieldnet_competition_v1/train_labels/"
_CHIP_CACHE_DIR = SCRATCH_DIR / "agrifieldnet_chips"

MIN_AREA_HA, MAX_AREA_HA = 0.15, 400  # margin above the deployed API's 0.1ha floor

# AgriFieldNet's exact original survey year isn't published anywhere accessible
# without a Radiant MLHub account (checked: no stac.json/datetime tag is mirrored
# on the public Azure blob copy this module reads from) -- use a recent full
# calendar year instead, same as the earlier live-pipeline India test this
# session. Real caveat, not resolved: possible crop-rotation drift between
# whatever year AgriFieldNet actually surveyed and this one.
AGRIFIELDNET_YEAR = 2024


def list_chip_ids() -> list[str]:
    resp = requests.get(_LIST_URL, timeout=30)
    resp.raise_for_status()
    names = re.findall(r"<Name>(.*?)</Name>", resp.text)
    chip_ids = set()
    for n in names:
        base = n.split("/")[-1].replace(".tif", "")
        base = base.replace("ref_agrifieldnet_competition_v1_labels_train_", "")
        base = base.replace("_field_ids", "")
        chip_ids.add(base)
    return sorted(chip_ids)


def _fetch_chip_files(chip_id: str) -> tuple[Path, Path]:
    _CHIP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    labels_name = f"ref_agrifieldnet_competition_v1_labels_train_{chip_id}.tif"
    fields_name = f"ref_agrifieldnet_competition_v1_labels_train_{chip_id}_field_ids.tif"
    labels_path = _CHIP_CACHE_DIR / labels_name
    fields_path = _CHIP_CACHE_DIR / fields_name
    for name, dest in [(labels_name, labels_path), (fields_name, fields_path)]:
        if not dest.exists():
            r = requests.get(f"{_BLOB_BASE}/{name}", timeout=30)
            r.raise_for_status()
            dest.write_bytes(r.content)
    return labels_path, fields_path


def sample_fields(n_per_class: int = 150, chip_ids: list[str] | None = None) -> gpd.GeoDataFrame:
    """Download AgriFieldNet label chips (cached under SCRATCH_DIR), polygonize
    single-crop fields whose crop_id maps to a target class, and return up to
    n_per_class per class (fewer if the dataset doesn't have that many -- same
    "take everything available" fallback as sample_country())."""
    if chip_ids is None:
        chip_ids = list_chip_ids()

    by_class: dict[str, list[dict]] = {cls: [] for cls in set(AGRIFIELDNET_CROP_ID_TO_CLASS.values())}
    remaining = set(by_class.keys())

    for i, chip_id in enumerate(chip_ids):
        if not remaining:
            break
        labels_path, fields_path = _fetch_chip_files(chip_id)
        with rasterio.open(labels_path) as lsrc, rasterio.open(fields_path) as fsrc:
            labels = lsrc.read(1)
            field_ids = fsrc.read(1)
            transform = lsrc.transform
            crs = lsrc.crs

        for fid in np.unique(field_ids):
            if fid == 0:
                continue
            mask = field_ids == fid
            npix = int(mask.sum())
            crop_vals = np.unique(labels[mask])
            crop_vals = crop_vals[crop_vals != 0]
            if len(crop_vals) != 1:
                continue  # skip mixed/ambiguous fields
            crop_id = int(crop_vals[0])
            cls = AGRIFIELDNET_CROP_ID_TO_CLASS.get(crop_id)
            if cls is None or cls not in remaining:
                continue
            area_ha = npix * 100 / 10_000
            if not (MIN_AREA_HA <= area_ha <= MAX_AREA_HA):
                continue

            polys = [
                shape(geom) for geom, val in shapes(mask.astype(np.uint8), mask=mask, transform=transform)
                if val == 1
            ]
            if not polys:
                continue
            geom = max(polys, key=lambda p: p.area)  # largest connected piece
            geom_wgs84 = shape(transform_geom(crs, "EPSG:4326", mapping(geom)))
            if not geom_wgs84.is_valid or geom_wgs84.geom_type != "Polygon":
                continue

            by_class[cls].append({"classname": cls, "geometry": geom_wgs84})
            if len(by_class[cls]) >= n_per_class:
                remaining.discard(cls)

        if (i + 1) % 100 == 0 or not remaining:
            counts = {k: len(v) for k, v in by_class.items()}
            print(f"  [IN] scanned {i + 1}/{len(chip_ids)} chips: {counts}")

    rows = [r for lst in by_class.values() for r in lst]
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    print(f"  [IN] sampled {len(gdf)} parcels: {gdf['classname'].value_counts().to_dict()}")
    return gdf
