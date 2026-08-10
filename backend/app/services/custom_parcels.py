"""
custom_parcels.py — load user-provided vector files (GeoPackage, KML, or
anything else GDAL/geopandas can read) as a parcel region, in the same
(classname, geometry) shape as the training pipeline's *_sampled_geoms.pkl
checkpoints (see parcels.py).

A region can be built from MULTIPLE source files, concatenated into one
combined dataset -- e.g. the "CUSTOM" region below combines a GeoPackage
(wheat/mustard fields near Varanasi) with a KML (a single meadow parcel near
Jabalpur, IIITDMJ campus): two different places, two different file formats,
one logical tab in the UI.

This is the "fast path now" half of a two-step plan: today, CUSTOM_REGIONS
is a small static registry. load_vector_file() is written generically enough
that a future POST /api/v1/parcels/upload endpoint can call it directly
against an uploaded temp file and append it to a region's source list --
no rewrite needed then, just a new caller.
"""

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry.base import BaseGeometry

logger = logging.getLogger(__name__)


def largest_polygon_part(geom: BaseGeometry):
    """Real AOIs must be a single Polygon (see app/schemas.py's
    AOIPolygon.type: Literal["Polygon"] and services/validation.py's
    isinstance check) -- the deployed API rejects MultiPolygon outright.
    Many real-world hand-digitized/merged files carry single-part
    MultiPolygons as an artifact of GIS tooling (confirmed: this is exactly
    what happened to all_crops_87.gpkg via a QGIS "Merge Vector Layers" step).
    Collapsing to the largest-area part is also what correctly handles a
    buffer(0) geometry repair that splits into a real lobe plus a tiny
    self-intersection sliver -- picking by area, not by part order, is what
    keeps the real field and drops the artifact.
    """
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "Polygon":
        return geom
    if geom.geom_type == "MultiPolygon":
        return max(geom.geoms, key=lambda p: p.area)
    return None  # line/point/GeometryCollection -- not usable as an AOI


def load_vector_file(
    path: Path,
    layer: str | None = None,
    label_column: str | None = None,
    label_value: str | None = None,
) -> gpd.GeoDataFrame:
    """Returns a 2-column GeoDataFrame (classname, geometry) in EPSG:4326,
    every row a single valid Polygon -- ready to feed straight into
    get_parcels_geojson()'s feature-building loop, same as a loaded
    *_sampled_geoms.pkl checkpoint. Format-agnostic: geopandas/GDAL picks the
    driver from the file itself, so this works unchanged for .gpkg, .kml, or
    anything else GDAL reads.

    Exactly one of label_column / label_value must be given:
    - label_column: the file has its own attribute column that already holds
      the real crop label for every row (e.g. all_crops_87.gpkg's `name`
      column: "wheat"/"mustard").
    - label_value: the file's own attributes do NOT carry a usable crop
      label (e.g. the IIITDMJ KML's only attribute is a Placemark name of
      "operating area" -- not a crop type), but the true label is known
      externally (confirmed directly with the user) and applies to every
      geometry from this source. Never inferred/guessed -- only used when a
      human has explicitly confirmed the label.

    Order of operations matters:
    1. Whitelist columns (not blocklist) -- keeps only the label + geometry,
       so any stray provenance/PII column in an arbitrary future upload
       (e.g. all_crops_87.gpkg's own `path` column, which leaks the original
       contributor's local filesystem path) can never reach the API by
       construction, not by name-matching a column blocklist.
    2. Repair invalid geometry BEFORE reprojecting, in the source CRS --
       collapsing to the largest part is an area comparison, and area
       comparisons are only undistorted in a projected (metric) CRS, not
       after reprojecting to degrees.
    3. Collapse Multi* to the largest single Polygon, still pre-reprojection.
    4. Reproject to EPSG:4326 last, once geometry is already clean.
    """
    if (label_column is None) == (label_value is None):
        raise ValueError("load_vector_file: give exactly one of label_column or label_value")

    gdf = gpd.read_file(path, layer=layer)
    if label_column is not None:
        gdf = gdf[[label_column, "geometry"]].rename(columns={label_column: "classname"})
    else:
        gdf = gdf[["geometry"]].copy()
        gdf["classname"] = label_value
    gdf["classname"] = gdf["classname"].astype(str).str.strip()

    invalid = ~gdf.geometry.is_valid
    if invalid.any():
        gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].buffer(0)

    gdf["geometry"] = gdf.geometry.apply(largest_polygon_part)
    n_before = len(gdf)
    gdf = gdf[gdf.geometry.notna()]
    if len(gdf) < n_before:
        logger.warning(
            "load_vector_file(%s): dropped %d/%d rows with unrecoverable geometry",
            path, n_before - len(gdf), n_before,
        )

    gdf = gdf.to_crs(4326).reset_index(drop=True)
    return gdf


def load_custom_region(entry: dict) -> gpd.GeoDataFrame:
    """Loads and concatenates every source in entry["sources"], skipping any
    whose file doesn't exist (same "fresh clone, no local data yet" degrade-
    gracefully philosophy as the rest of parcels.py). Sources can be
    different formats and different places -- that's the point, this is what
    lets one region combine a GeoPackage and a KML into a single tab."""
    parts = []
    for src in entry["sources"]:
        if not Path(src["path"]).exists():
            continue
        parts.append(load_vector_file(
            src["path"],
            layer=src.get("layer"),
            label_column=src.get("label_column"),
            label_value=src.get("label_value"),
        ))
    if not parts:
        return gpd.GeoDataFrame(columns=["classname", "geometry"], geometry="geometry", crs=4326)
    combined = pd.concat(parts, ignore_index=True)
    return gpd.GeoDataFrame(combined, crs=4326).reset_index(drop=True)


_LOCAL_PARCELS_DIR = Path(__file__).resolve().parent.parent.parent / "local_parcels"

# Small static registry of extra, non-pickle parcel regions -- deliberately
# separate from parcels.py's REGION_NAMES (which is kept in sync with the
# training pipeline's regions.py; nothing here ever went through training).
CUSTOM_REGIONS = {
    "CUSTOM": {
        "display_name": "Custom Hand-Labeled",
        "sources": [
            {
                # 87 fields near Varanasi, UP -- wheat/mustard, labeled in
                # the file's own `name` column.
                "path": _LOCAL_PARCELS_DIR / "all_crops_87.gpkg",
                "layer": "crops_merged",
                "label_column": "name",
            },
            {
                # 1 field near the IIITDMJ campus, Jabalpur -- the KML's own
                # Placemark name ("operating area") is not a crop label, so
                # the true label ("meadow") was confirmed directly with the
                # user instead of read from the file.
                "path": _LOCAL_PARCELS_DIR / "IIITDMJ (1).kml",
                "layer": "Untitled",
                "label_value": "meadow",
            },
        ],
    },
}
