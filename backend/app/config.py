"""
config.py — Shared configuration for EuroCrops multi-country crop classification.

Everything here must match what was used during TRAINING (class index order,
sequence length, model dims). Do not hand-edit CLASS_NAMES without checking
backend/training/eurocrops_pipeline/taxonomy.py — a mismatch here silently
scrambles ground-truth/prediction labels.
"""

import os
from pathlib import Path

import torch
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ============================================
# Paths / device
# ============================================
# Resolved relative to this file, not the process's cwd, so it works the same
# whether uvicorn is launched from backend/ or anywhere else.
#
# Trained on EuroCrops multi-country + AgriFieldNet India + hand-labeled
# CUSTOM data (see backend/training/eurocrops_pipeline/) with imagery
# fetched live through the same CDSE pipeline used at inference time.
# As of this retrain (2026-08-10): Austria (1650), Belgium-Flanders (1464),
# Germany-Brandenburg (1411, now complete), Germany-Lower Saxony (1416),
# Germany-North Rhine-Westphalia (1313), India/AgriFieldNet (1009), and the
# 88-parcel hand-labeled CUSTOM region (wheat/mustard near Varanasi, one
# meadow near Jabalpur — see app/services/custom_parcels.py) all went into
# training. 12 of 18 EuroCrops countries still remain unfetched — retrain
# and repoint this path as more land (backend/training/eurocrops_pipeline/
# run_fetch.py) — see progress.md for the full validation breakdown.
CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "multicountry_india_custom_18class.pth"
REGION = "frh04"                                    # BreizhCrops region code used
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================
# Sequence preprocessing
# ============================================
TARGET_SEQ_LEN = 45   # must match training: sequences are resampled (with
                       # replacement if shorter, without if longer) to this
                       # length, then sorted back into chronological order —
                       # see breizhcrops.datasets.breizhcrops.get_default_transform.
                       # There is no zero-padding at any point in training.

INFERENCE_ENSEMBLE_SIZE = 16  # preprocess_sequence's resampling is random with no
                       # fixed seed (this is correct/intentional as training-time
                       # augmentation — see breizhcrops get_default_transform). At
                       # inference a single draw is a high-variance point estimate:
                       # the same parcel can land on different confident classes on
                       # different calls. predict() draws this many independent
                       # resamples and averages their softmax probabilities to get a
                       # stable prediction instead of trusting one draw.

# ============================================
# Input bands — verified against breizhcrops.datasets.breizhcrops.get_default_transform
# (installed package source), NOT natural/numeric order. This is a lexicographic
# ("B10" < "B2" as strings) artifact of BreizhCrops' own band list, but the
# checkpoint trained on exactly this column order, so it must be reproduced
# exactly here even though it looks wrong at a glance.
# ============================================
INPUT_DIM = 13
SENTINEL2_L1C_BANDS = [
    "B1", "B10", "B11", "B12", "B2", "B3", "B4",
    "B5", "B6", "B7", "B8", "B8A", "B9",
]

# ============================================
# Copernicus Data Space Ecosystem — Sentinel Hub Statistical API credentials.
# Register an OAuth client at the Sentinel Hub Dashboard (dataspace.copernicus.eu
# -> User Settings -> OAuth clients) and put the values in backend/.env
# (gitignored). This is what lets sentinel_fetch.py pull true Sentinel-2 L1C
# data server-side, rather than the L2A-with-zero-filled-B10 workaround this
# pipeline previously used against Planetary Computer.
# ============================================
CDSE_SH_CLIENT_ID = os.environ.get("CDSE_SH_CLIENT_ID")
CDSE_SH_CLIENT_SECRET = os.environ.get("CDSE_SH_CLIENT_SECRET")
CDSE_SH_BASE_URL = "https://sh.dataspace.copernicus.eu"
CDSE_SH_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

# ============================================
# Class index -> name mapping
# Source of truth: backend/training/eurocrops_pipeline/taxonomy.py's CLASS_NAMES
# (HCAT3-derived, harmonized across EuroCrops countries). Do NOT hand-edit
# without also checking that file — index order must match the checkpoint
# exactly (it is the model's output layer order).
# ============================================
CLASS_NAMES = {
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
    # AgriFieldNet India additions (backend/training/eurocrops_pipeline/taxonomy.py)
    11: "mustard",
    12: "sugarcane",
    13: "lentil",
    14: "rice",
    15: "gram",
    16: "garlic",
    17: "fallow",
}

NUM_CLASSES = len(CLASS_NAMES)

# ============================================
# Display colors (used for both ground-truth and prediction map fills —
# keep consistent so the same crop always renders the same color)
# ============================================
CLASS_COLORS = {
    "meadow": "#8dd3c7",
    "wheat": "#ffed6f",
    "barley": "#fdb462",
    "triticale": "#d9a066",
    "rapeseed": "#f4d442",
    "maize": "#fb8072",
    "sunflower": "#f4a300",
    "vineyards": "#80b1d3",
    "fruit": "#b3de69",
    "nuts": "#bebada",
    "potatoes": "#bc80bd",
    "mustard": "#ffdd55",
    "sugarcane": "#66c266",
    "lentil": "#c49a6c",
    "rice": "#9ecae1",
    "gram": "#d9b382",
    "garlic": "#f2f2d0",
    "fallow": "#a9a9a9",
}
DEFAULT_COLOR = "#7f8c8d"  # fallback grey for unmatched/unknown class names

# ============================================
# Model hyperparameters (must match training exactly)
# ============================================
MODEL_PARAMS = dict(
    d_model=64,
    nhead=4,
    num_layers=3,
    dim_feedforward=64,
    dropout=0.2,
)


def get_color(label: str) -> str:
    """Fuzzy-match a class name string to its display color."""
    key = label.lower().strip()
    for k, v in CLASS_COLORS.items():
        if k in key:
            return v
    return DEFAULT_COLOR
