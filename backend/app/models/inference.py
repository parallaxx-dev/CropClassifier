"""
inference.py — Preprocessing + prediction for a single BreizhCrops parcel.

This is the core logic your web backend should call. It does NOT handle
plotting/visualization (see visualize.py for that) — just: raw parcel time
series in, (predicted_class_name, confidence, all_class_probs) out.
"""

import numpy as np
import torch

from app.config import CLASS_NAMES, TARGET_SEQ_LEN, DEVICE


def preprocess_sequence(x_raw, target_seq_len=TARGET_SEQ_LEN):
    """
    Two steps, both verified against the actual training notebook (not just
    the breizhcrops package defaults):

    1. Resample to `target_seq_len` observations — with replacement if fewer
       are available, without replacement if more — then sort back into
       chronological order. This mirrors breizhcrops' own get_default_transform,
       which every dataset[i] call goes through at training time.
    2. Per-sample z-score normalize (mean/std across the time axis, per band),
       computed on the already-resampled array. The training notebook applies
       this on top of get_default_transform's output before feeding the model
       — it is a real, separate step, not something the model's internal
       LayerNorm substitutes for.

    x_raw: numpy array or torch tensor, shape (seq_len, num_features), seq_len >= 1,
    already in reflectance units (sentinel_fetch applies the 1e-4 DN scale) and
    already in SENTINEL2_L1C_BANDS column order.
    Returns: numpy array, shape (target_seq_len, num_features)
    """
    if torch.is_tensor(x_raw):
        x_raw = x_raw.numpy()

    seq_len = x_raw.shape[0]
    idxs = np.random.choice(seq_len, target_seq_len, replace=seq_len < target_seq_len)
    idxs.sort()
    x_resampled = x_raw[idxs]

    mean = x_resampled.mean(axis=0)
    std = x_resampled.std(axis=0)
    std[std == 0] = 1.0  # avoid divide-by-zero on constant features
    return (x_resampled - mean) / std


def predict(model, x_raw, device=DEVICE, class_names=CLASS_NAMES):
    """
    Run inference on one raw parcel time series.

    Returns dict:
        {
            "pred_class_idx": int,
            "pred_label": str,
            "confidence": float,
            "probs": {label: prob, ...}   # full distribution, for debugging/UI
        }
    """
    x_resampled = preprocess_sequence(x_raw)
    input_tensor = torch.tensor(x_resampled, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    pred_idx = int(torch.argmax(probs).item())
    confidence = float(probs[pred_idx].item())
    pred_label = class_names.get(pred_idx, f"UNKNOWN_CLASS_{pred_idx}")

    probs_by_label = {
        class_names.get(i, f"UNKNOWN_CLASS_{i}"): float(probs[i].item())
        for i in range(len(probs))
    }

    return {
        "pred_class_idx": pred_idx,
        "pred_label": pred_label,
        "confidence": confidence,
        "probs": probs_by_label,
    }
