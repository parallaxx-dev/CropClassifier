"""
visualize.py — 3-panel (RGB satellite / ground truth / prediction) figure
generation for a single BreizhCrops parcel.

Requires: geopandas, matplotlib, contextily (optional — degrades gracefully
if missing or if tile fetch fails).
"""

import random
import time

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from config import get_color

try:
    import contextily as cx

    HAS_CONTEXTILY = True
except ImportError:
    HAS_CONTEXTILY = False


def get_parcel_geometry(gdf_all, parcel_id):
    """Look up a parcel's polygon geometry by id. Raises if not found."""
    gdf_all = gdf_all.copy()
    gdf_all["id"] = gdf_all["id"].astype(str)
    parcel_geom = gdf_all.loc[gdf_all["id"] == str(parcel_id)]
    if len(parcel_geom) == 0:
        raise ValueError(
            f"Parcel {parcel_id} not found in geodataframe. "
            f"Sample ids: {gdf_all['id'].head().tolist()}"
        )
    return parcel_geom


def plot_parcel_result(
    parcel_id,
    true_label,
    pred_label,
    confidence,
    gdf_all,
    save_path=None,
    show=True,
    basemap_retries=3,
    basemap_zoom=16,
):
    """
    Generate the 3-panel figure: RGB satellite chip | ground truth | prediction.

    Returns the matplotlib Figure. Optionally saves to save_path if provided.
    """
    parcel_geom = get_parcel_geometry(gdf_all, parcel_id)

    true_color = get_color(true_label)
    pred_color = get_color(pred_label)

    parcel_3857 = parcel_geom.to_crs(epsg=3857)
    minx, miny, maxx, maxy = parcel_3857.total_bounds
    w_margin = max((maxx - minx) * 0.20, 15)
    h_margin = max((maxy - miny) * 0.20, 15)
    xlims = (minx - w_margin, maxx + w_margin)
    ylims = (miny - h_margin, maxy + h_margin)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor="white")

    # Panel 1: RGB satellite chip
    parcel_3857.plot(ax=axes[0], facecolor="none", edgecolor="cyan", linewidth=2.5)
    basemap_ok = False
    if HAS_CONTEXTILY:
        for attempt in range(basemap_retries):
            try:
                cx.add_basemap(
                    axes[0], source=cx.providers.Esri.WorldImagery, zoom=basemap_zoom
                )
                basemap_ok = True
                break
            except Exception as e:
                wait = 3 + random.uniform(0, 2)
                print(
                    f"⚠️ Basemap attempt {attempt + 1}/{basemap_retries} failed "
                    f"for parcel {parcel_id}: {e} — retrying in {wait:.1f}s"
                )
                time.sleep(wait)
    if not basemap_ok:
        print(f"❌ Basemap unavailable for parcel {parcel_id} — outline only")
        axes[0].text(
            0.5, 0.5, "Basemap unavailable",
            transform=axes[0].transAxes, ha="center", va="center",
            color="red", fontsize=10,
        )
    axes[0].set_xlim(xlims)
    axes[0].set_ylim(ylims)
    axes[0].set_title(f"Parcel #{parcel_id} — Sentinel-2 RGB", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: ground truth
    parcel_3857.plot(ax=axes[1], color=true_color, edgecolor="black", linewidth=1.5)
    axes[1].set_xlim(xlims)
    axes[1].set_ylim(ylims)
    axes[1].set_title(f"Ground Truth: {true_label}", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    # Panel 3: prediction
    parcel_3857.plot(ax=axes[2], color=pred_color, edgecolor="black", linewidth=1.5)
    axes[2].set_xlim(xlims)
    axes[2].set_ylim(ylims)
    axes[2].set_title(
        f"Prediction: {pred_label} ({confidence * 100:.1f}%)", fontsize=12, fontweight="bold"
    )
    axes[2].axis("off")

    handles = [
        Patch(color=true_color, label=true_label),
        Patch(color=pred_color, label=pred_label),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.05), frameon=True)
    fig.suptitle(f"BreizhCrops Parcel ID: {parcel_id}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ Saved {save_path} (basemap: {'OK' if basemap_ok else 'MISSING'})")

    if show:
        plt.show()

    return fig
