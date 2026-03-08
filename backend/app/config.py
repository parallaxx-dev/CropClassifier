"""
config.py — Shared configuration for BreizhCrops crop classification.

Everything here must match what was used during TRAINING (class index order,
sequence length, model dims). Do not hand-edit CLASS_NAMES without checking
classmapping.csv from the breizhcrops package — a mismatch here silently
scrambles ground-truth/prediction labels.
"""

import torch

# ============================================
# Paths / device
# ============================================
CHECKPOINT_PATH = "best_transformer_breizh.pth"   # update path for deployment
REGION = "frh04"                                    # BreizhCrops region code used
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================
# Sequence preprocessing
# ============================================
TARGET_SEQ_LEN = 45   # must match training: shorter sequences are zero-padded,
                       # longer ones truncated to this length before normalization

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
