"""
config.py — Shared configuration for BreizhCrops crop classification.

Everything here must match what was used during TRAINING (class index order,
sequence length, model dims). Do not hand-edit CLASS_NAMES without checking
classmapping.csv from the breizhcrops package — a mismatch here silently
scrambles ground-truth/prediction labels.
"""

from pathlib import Path

import torch

# ============================================
# Paths / device
# ============================================
# Resolved relative to this file, not the process's cwd, so it works the same
# whether uvicorn is launched from backend/ or anywhere else.
CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "best_transformer_breizh.pth"
REGION = "frh04"                                    # BreizhCrops region code used
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================
# Sequence preprocessing
# ============================================
TARGET_SEQ_LEN = 45   # must match training: shorter sequences are zero-padded,
                       # longer ones truncated to this length before normalization

# ============================================
# Input bands — verified against the checkpoint itself (input_projection weight
# shape is (64, 13)), which matches BreizhCrops' Sentinel-2 L1C band layout.
# Order matters: it must match the column order the model was trained on.
# ============================================
INPUT_DIM = 13
SENTINEL2_L1C_BANDS = [
    "B1", "B2", "B3", "B4", "B5", "B6", "B7",
    "B8", "B8A", "B9", "B10", "B11", "B12",
]

# Planetary Computer only serves L2A, not L1C — verified against its live STAC
# catalog (no "sentinel-2-l1c" collection exists there). L2A drops B10 entirely
# (it's the cirrus band, meaningless post-atmospheric-correction) and every
# other band is surface reflectance rather than the top-of-atmosphere values
# this checkpoint was trained on. B10 is zero-filled below; there is currently
# no correct source for it through this pipeline. Treat predictions from the
# AOI-fetch path as provisional until this points at a real L1C source.
SENTINEL2_L2A_ASSET_KEYS = {
    "B1": "B01", "B2": "B02", "B3": "B03", "B4": "B04", "B5": "B05",
    "B6": "B06", "B7": "B07", "B8": "B08", "B8A": "B8A", "B9": "B09",
    "B10": None,
    "B11": "B11", "B12": "B12",
}

# ============================================
# Class index -> name mapping
# Source of truth: classmapping.csv shipped with the breizhcrops package.
# Verify with:
#   import pandas as pd
#   pd.read_csv("breizhcrops_dataset/classmapping.csv")
# ============================================
CLASS_NAMES = {
    0: "barley",
    1: "wheat",
    2: "rapeseed",
    3: "corn",
    4: "sunflower",
    5: "orchards",
    6: "nuts",
    7: "permanent meadows",
    8: "temporary meadows",
}

NUM_CLASSES = len(CLASS_NAMES)

# ============================================
# Display colors (used for both ground-truth and prediction map fills —
# keep consistent so the same crop always renders the same color)
# ============================================
CLASS_COLORS = {
    "barley": "#e74c3c",
    "wheat": "#3498db",
    "rapeseed": "#2ecc71",
    "corn": "#9b59b6",
    "sunflower": "#f39c12",
    "orchards": "#16a085",
    "nuts": "#8e44ad",
    "permanent meadows": "#f1c40f",
    "temporary meadows": "#1abc9c",
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
