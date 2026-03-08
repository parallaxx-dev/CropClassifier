"""
inference.py — Preprocessing + prediction for a single BreizhCrops parcel.

This is the core logic your web backend should call. It does NOT handle
plotting/visualization (see visualize.py for that) — just: raw parcel time
series in, (predicted_class_name, confidence, all_class_probs) out.
"""

import numpy as np
import torch

from config import CLASS_NAMES, TARGET_SEQ_LEN, DEVICE


def preprocess_sequence(x_raw, target_seq_len=TARGET_SEQ_LEN):
    """
    Pad/truncate a raw time series to a fixed length and per-feature
    z-normalize it. Must match the exact steps used during training.

    x_raw: numpy array or torch tensor, shape (seq_len, num_features)
    Returns: numpy array, shape (target_seq_len, num_features)
    """
    if torch.is_tensor(x_raw):
        x_raw = x_raw.numpy()

    seq_len = x_raw.shape[0]
    if seq_len < target_seq_len:
        pad_len = target_seq_len - seq_len
        x_padded = np.pad(
            x_raw, ((0, pad_len), (0, 0)), mode="constant", constant_values=0
        )
    else:
        x_padded = x_raw[:target_seq_len]

    mean = x_padded.mean(axis=0)
    std = x_padded.std(axis=0)
    std[std == 0] = 1.0  # avoid divide-by-zero on constant features
    return (x_padded - mean) / std


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
    x_norm = preprocess_sequence(x_raw)
    input_tensor = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0).to(device)

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
