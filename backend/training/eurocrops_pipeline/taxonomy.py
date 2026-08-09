"""
taxonomy.py — target class scheme + the lookup tables that map into it.

Three independent input vocabularies, one shared output class set:

- HCAT_TO_CLASS: for the 18 EuroCrops countries (including France's own EuroCrops
  contribution — its shapefile carries EC_hcat_n/EC_hcat_c directly, verified by
  reading FR_2018_EC21.shp's actual columns, same as every other country; there is
  no need to special-case France via its native RPG code anymore).
- RPG_CODE_TO_CLASS: for BreizhCrops (backend/breizhcrops_dataset/2017/frh04.shp),
  the one shapefile in this pipeline that does NOT carry EC_hcat_* columns — it only
  has the raw French RPG crop code (CODE_CULTU). Derived from EuroCrops' own
  fr_2018.csv crosswalk (original_code -> HCAT3_name) composed with HCAT_TO_CLASS,
  not hand-maintained separately — BreizhCrops and EuroCrops' FR_2018 share the
  exact same RPG code vocabulary (both are French RPG declarations, just different
  years/regions), confirmed directly, not assumed.
- AGRIFIELDNET_CROP_ID_TO_CLASS: for AgriFieldNet India (Bihar/UP/Rajasthan/Odisha,
  see agrifieldnet.py) — its own 13-class numeric crop_id scheme, entirely
  independent of HCAT/RPG. Only crop_ids with either a genuine semantic match to an
  existing EuroCrops class (wheat, maize, potatoes) or enough real ground-truth
  examples to support training a new class (>=40, surveyed directly against the
  full public dataset, not assumed) are mapped. Green pea (12 examples), Coriander
  (8), and Bersem (8) are the ENTIRE public dataset's supply for those crops — not
  a partial fetch that improves later, a real ceiling — and are deliberately left
  unmapped: too few for any meaningful train/test split.

Class scheme decisions (see plan doc for the full discussion/corrections):
- Expanded beyond the old 9-class scheme to match EuroCrops' actual harmonized
  categories, validated against the "Top 10 Absolute Scale" counts from the
  EuroCrops website.
- Permanent + temporary meadow merged into one `meadow` class, as a deliberate
  simplicity tradeoff — NOT because HCAT lacks the split (it doesn't: HCAT3 has
  distinct `pasture_meadow_grassland_grass` vs `temporary_grass` categories,
  verified directly against real crosswalk data for France AND Portugal, which
  contradicted an earlier, unverified notebook comment claiming otherwise).
- Winter/spring/summer variants of wheat, barley, triticale, and rapeseed merged
  into one class each (e.g. `wheat` covers winter_common_soft_wheat +
  spring_common_soft_wheat + ...), via HCAT3's own parent/child code hierarchy.
- `mustard` (AgriFieldNet) kept SEPARATE from `rapeseed` (EuroCrops), not merged,
  despite both being Brassica oilseed crops — they're different species
  (B. juncea vs. B. napus) with no verified spectral/phenological equivalence
  established here; merging on a taxonomic-family assumption alone would repeat
  the exact kind of unverified-claim mistake already caught once in this file's
  own history (see the meadow-split correction above).
"""

from pathlib import Path

import pandas as pd

_HCAT3_CSV = Path(__file__).resolve().parent / "_taxonomy_ref" / "HCAT3.csv"
_FR_CROSSWALK_CSV = Path(__file__).resolve().parent / "_taxonomy_ref" / "fr_2018.csv"

CLASS_NAMES: dict[int, str] = {
    0: "meadow",
    1: "wheat",
    2: "barley",
    3: "triticale",
    4: "rapeseed",
    5: "maize",
    6: "sunflower",
    7: "vineyards",
    8: "fruit",
    9: "nuts",
    10: "potatoes",
    # AgriFieldNet India additions (see agrifieldnet.py) -- new indices appended
    # rather than renumbering the block above, purely so existing indices stay
    # stable/easy to cross-reference; index order only matters insofar as it must
    # match the checkpoint's output layer, and this is a from-scratch retrain
    # regardless (NUM_CLASSES changed), so no warm-start compatibility to protect.
    11: "mustard",
    12: "sugarcane",
    13: "lentil",
    14: "rice",
    15: "gram",
    16: "garlic",
    17: "fallow",
}
CLASS_NAME_TO_ID = {v: k for k, v in CLASS_NAMES.items()}
NUM_CLASSES = len(CLASS_NAMES)

# AgriFieldNet's own numeric crop_id -> target class name. Source: the dataset's
# official Documentation.pdf legend (radiantearth.blob.core.windows.net/mlhub/
# ref_agrifieldnet_competition_v1/Documentation.pdf), not inferred. crop_ids 5
# (Green pea), 14 (Coriander), 16 (Bersem) are deliberately absent -- see module
# docstring above.
AGRIFIELDNET_CROP_ID_TO_CLASS: dict[int, str] = {
    1: "wheat",       # matches existing EuroCrops class
    2: "mustard",
    3: "lentil",
    4: "fallow",
    6: "sugarcane",
    8: "garlic",
    9: "maize",        # matches existing EuroCrops class
    13: "gram",
    15: "potatoes",    # matches existing EuroCrops class
    36: "rice",
}

# Each target class's HCAT3 "root" name(s) — descendants (via the code hierarchy)
# are included automatically, not hand-enumerated (see _descendants_or_self below,
# and the verification this was run through — printed, eyeballed, matches the real
# HCAT3.csv content exactly, not assumed).
_CLASS_HCAT3_ROOTS: dict[str, list[str]] = {
    "meadow": ["pasture_meadow_grassland_grass", "temporary_grass"],
    "wheat": ["common_soft_wheat"],
    "barley": ["barley"],
    "triticale": ["triticale"],
    "rapeseed": ["rapeseed_rape"],
    "maize": ["grain_maize_corn_popcorn"],
    "sunflower": ["sunflower"],
    "vineyards": ["vineyards_wine_vine_rebland_grapes"],
    "fruit": ["orchards_fruits"],
    "nuts": ["nuts"],
    "potatoes": ["potatoes"],
}


def _load_hcat3() -> pd.DataFrame:
    df = pd.read_csv(_HCAT3_CSV)
    df["HCAT3_code"] = df["HCAT3_code"].astype(str)
    return df


def _descendants_or_self(hcat3_df: pd.DataFrame, root_name: str) -> list[str]:
    """All HCAT3_name values whose code is root_name's code or a descendant of it,
    via trailing-zero-stripped prefix matching on the hierarchical code. Verified
    against real HCAT3.csv content during implementation (printed and eyeballed
    for every root used here) — see plan doc's Phase 1 notes."""
    matches = hcat3_df.loc[hcat3_df["HCAT3_name"] == root_name, "HCAT3_code"]
    if matches.empty:
        raise ValueError(f"HCAT3 root name not found: {root_name!r} — taxonomy reference may be stale")
    root_code = matches.iloc[0]
    trailing_zeros = len(root_code) - len(root_code.rstrip("0"))
    prefix = root_code[: len(root_code) - trailing_zeros]
    return hcat3_df.loc[hcat3_df["HCAT3_code"].str.startswith(prefix), "HCAT3_name"].tolist()


def build_hcat_to_class() -> dict[str, str]:
    """HCAT3_name -> target class name, for the 18 EuroCrops-shapefile countries."""
    hcat3_df = _load_hcat3()
    mapping: dict[str, str] = {}
    for cls, roots in _CLASS_HCAT3_ROOTS.items():
        for root in roots:
            for member in _descendants_or_self(hcat3_df, root):
                if member in mapping and mapping[member] != cls:
                    raise ValueError(
                        f"HCAT3 name {member!r} claimed by both {mapping[member]!r} "
                        f"and {cls!r} — overlapping class roots, fix _CLASS_HCAT3_ROOTS"
                    )
                mapping[member] = cls
    return mapping


def build_rpg_code_to_class() -> dict[str, str]:
    """Original French RPG code -> target class name, for BreizhCrops (frh04.shp).
    Derived from EuroCrops' own fr_2018.csv crosswalk composed with HCAT_TO_CLASS —
    not a separately hand-maintained table, so it can't silently drift out of sync
    with the HCAT-based mapping used for every other country."""
    hcat_to_class = build_hcat_to_class()
    crosswalk = pd.read_csv(_FR_CROSSWALK_CSV)
    mapping: dict[str, str] = {}
    for _, row in crosswalk.iterrows():
        code = row.get("original_code")
        hcat3_name = row.get("HCAT3_name")
        if pd.isna(code) or pd.isna(hcat3_name):
            continue
        cls = hcat_to_class.get(hcat3_name)
        if cls is not None:
            mapping[str(code)] = cls
    return mapping


HCAT_TO_CLASS: dict[str, str] = build_hcat_to_class()
RPG_CODE_TO_CLASS: dict[str, str] = build_rpg_code_to_class()


def validate_against_published_counts() -> None:
    """Cheap sanity check: print which HCAT3 names feed each class, so the mapping
    can be eyeballed against the EuroCrops 'Top 10 Absolute Scale' counts before a
    multi-hour fetch run starts. Does not hit the network or read any shapefile."""
    hcat3_to_class = HCAT_TO_CLASS
    by_class: dict[str, list[str]] = {}
    for name, cls in hcat3_to_class.items():
        by_class.setdefault(cls, []).append(name)
    print(f"{NUM_CLASSES} classes, {len(hcat3_to_class)} HCAT3 names mapped:")
    for cls in CLASS_NAMES.values():
        members = sorted(by_class.get(cls, []))
        print(f"  {cls:12s} <- {members}")
    print(f"\nRPG_CODE_TO_CLASS (BreizhCrops/France-native): {len(RPG_CODE_TO_CLASS)} codes mapped")
    for code, cls in sorted(RPG_CODE_TO_CLASS.items()):
        print(f"  {code} -> {cls}")


if __name__ == "__main__":
    validate_against_published_counts()
