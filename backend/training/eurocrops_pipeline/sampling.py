"""
sampling.py — per-country schema probe + per-(country,class) capped stratified
sampling, generalizing train_eurocrops_local.py's download_and_sample_parcels().

Reads per TARGET CLASS (11 reads per country), not per individual HCAT3 name (48) —
each class's member HCAT3 names (from taxonomy.HCAT_TO_CLASS) are combined into one
`EC_hcat_n IN (...)` filter, keeping the same bounded-memory read pattern the
existing single-country script established (row-capped read, one query at a time)
without needing 48x as many round trips through the shapefile.
"""

import gc

import geopandas as gpd
import pandas as pd

from config import CountryEntry, N_PER_CLASS_PER_COUNTRY, OVERSAMPLE_PER_HCAT_CODE
from taxonomy import CLASS_NAMES, HCAT_TO_CLASS, RPG_CODE_TO_CLASS

# class -> its member raw RPG codes, built once (inverse of RPG_CODE_TO_CLASS) —
# used for BreizhCrops, the one shapefile in this pipeline without EC_hcat_* columns.
_CLASS_TO_RPG_CODES: dict[str, list[str]] = {}
for _code, _cls in RPG_CODE_TO_CLASS.items():
    _CLASS_TO_RPG_CODES.setdefault(_cls, []).append(_code)

# class -> its member HCAT3 names, built once (inverse of HCAT_TO_CLASS)
_CLASS_TO_HCAT3_NAMES: dict[str, list[str]] = {}
for _name, _cls in HCAT_TO_CLASS.items():
    _CLASS_TO_HCAT3_NAMES.setdefault(_cls, []).append(_name)


def probe_schema(shp_path) -> dict:
    """Read a small sample, confirm EC_hcat_n/EC_hcat_c + CRS are present, and
    report the HCAT null-rate. Raises if the shapefile doesn't look usable — one
    bad country's schema should be caught here, not halfway through a fetch."""
    probe = gpd.read_file(shp_path, rows=500)
    missing = {"EC_hcat_n", "EC_hcat_c"} - set(probe.columns)
    if missing:
        raise ValueError(f"{shp_path}: missing expected columns {missing} — got {probe.columns.tolist()}")
    if probe.crs is None:
        raise ValueError(f"{shp_path}: no CRS set")
    null_rate = probe["EC_hcat_c"].isna().mean()
    return {"crs": str(probe.crs), "null_rate": null_rate, "columns": probe.columns.tolist()}


def sample_country(shp_path, country: CountryEntry) -> gpd.GeoDataFrame:
    """Per-(class) capped read + stratified sample for one EuroCrops country,
    reprojected to WGS84. Mirrors download_and_sample_parcels()'s bounded-memory
    per-code read loop, but keyed on target class (via an EC_hcat_n IN (...)
    filter over that class's member HCAT3 names) instead of a single raw code."""
    frames = []
    for cls in CLASS_NAMES.values():
        members = _CLASS_TO_HCAT3_NAMES.get(cls, [])
        if not members:
            continue
        in_clause = ", ".join(f"'{m}'" for m in members)
        where = f"EC_hcat_n IN ({in_clause})"
        cls_gdf = gpd.read_file(shp_path, where=where, rows=OVERSAMPLE_PER_HCAT_CODE)
        if len(cls_gdf) == 0:
            continue
        cls_gdf["classname"] = cls
        frames.append(cls_gdf[["classname", "geometry"]])
        print(f"  [{country.code}] {cls}: read {len(cls_gdf)} parcels "
              f"(capped at {OVERSAMPLE_PER_HCAT_CODE})")
        del cls_gdf

    if not frames:
        print(f"  [{country.code}] no parcels matched any target class — skipping")
        return gpd.GeoDataFrame(columns=["classname", "geometry"])

    gdf_labeled = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)
    del frames
    gc.collect()

    sample_parts = []
    for cls_name, group in gdf_labeled.groupby("classname"):
        n = min(N_PER_CLASS_PER_COUNTRY, len(group))
        sample_parts.append(group.sample(n=n, random_state=42))
    sampled = gpd.GeoDataFrame(pd.concat(sample_parts, ignore_index=True), crs=gdf_labeled.crs)
    sampled = sampled.to_crs(epsg=4326)

    del gdf_labeled
    gc.collect()

    print(f"  [{country.code}] sampled {len(sampled)} parcels: "
          f"{sampled['classname'].value_counts().to_dict()}")
    return sampled


def sample_breizhcrops(shp_path) -> gpd.GeoDataFrame:
    """Per-class capped read + stratified sample for BreizhCrops (frh04.shp),
    which carries raw French RPG codes (CODE_CULTU) instead of EC_hcat_* — uses
    RPG_CODE_TO_CLASS (derived from EuroCrops' own fr_2018.csv crosswalk, see
    taxonomy.py) rather than the HCAT-based path used for the 18 EuroCrops
    countries. Same bounded-memory per-class read pattern as sample_country()."""
    frames = []
    for cls in CLASS_NAMES.values():
        codes = _CLASS_TO_RPG_CODES.get(cls, [])
        if not codes:
            continue
        in_clause = ", ".join(f"'{c}'" for c in codes)
        where = f"CODE_CULTU IN ({in_clause})"
        cls_gdf = gpd.read_file(shp_path, where=where, rows=OVERSAMPLE_PER_HCAT_CODE)
        if len(cls_gdf) == 0:
            continue
        cls_gdf["classname"] = cls
        frames.append(cls_gdf[["classname", "geometry"]])
        print(f"  [BZH] {cls}: read {len(cls_gdf)} parcels (capped at {OVERSAMPLE_PER_HCAT_CODE})")
        del cls_gdf

    if not frames:
        return gpd.GeoDataFrame(columns=["classname", "geometry"])

    gdf_labeled = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)
    del frames
    gc.collect()

    sample_parts = []
    for cls_name, group in gdf_labeled.groupby("classname"):
        n = min(N_PER_CLASS_PER_COUNTRY, len(group))
        sample_parts.append(group.sample(n=n, random_state=42))
    sampled = gpd.GeoDataFrame(pd.concat(sample_parts, ignore_index=True), crs=gdf_labeled.crs)
    sampled = sampled.to_crs(epsg=4326)

    del gdf_labeled
    gc.collect()

    print(f"  [BZH] sampled {len(sampled)} parcels: {sampled['classname'].value_counts().to_dict()}")
    return sampled
