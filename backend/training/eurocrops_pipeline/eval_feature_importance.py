"""
eval_feature_importance.py — permutation importance per input band, on the
same held-out test split run_train.py produces.

Method: for each of the 13 bands, shuffle that band's values ACROSS the test
set (breaks the band's real signal while preserving its marginal
distribution and every other band untouched), re-run the deployed model's
own ensembled predict() path, and measure the accuracy drop vs. the
unpermuted baseline. A bigger drop means the model leans on that band more.
Model-agnostic (no gradients needed), reuses the exact same eval harness as
eval_partial_range.py so results are on the same real held-out parcels the
rest of this project's numbers already come from -- not fabricated.

Run: ../.venv/bin/python eval_feature_importance.py --checkpoint PATH [--dump-json PATH]
"""

import argparse
import json
import pickle

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from config import CHECKPOINT_DIR, INPUT_DIM, SENTINEL2_L1C_BANDS
from dataset import preprocess_sequence
from model import ImprovedTimeSeriesTransformer
from taxonomy import CLASS_NAME_TO_ID, NUM_CLASSES

INFERENCE_ENSEMBLE_SIZE = 16
PERMUTATION_SEED = 0


def load_all_fetched(exclude_codes: set[str] | None = None):
    exclude_codes = exclude_codes or set()
    raw_sequences, labels = [], []
    for ckpt_path in sorted(CHECKPOINT_DIR.glob("*_fetched.pkl")):
        if ckpt_path.name.removesuffix("_fetched.pkl") in exclude_codes:
            continue
        with open(ckpt_path, "rb") as f:
            results = pickle.load(f)
        for v in results.values():
            if v["x_raw"].shape[0] < 1:
                continue
            raw_sequences.append(v["x_raw"])
            labels.append(CLASS_NAME_TO_ID[v["classname"]])
    return raw_sequences, np.array(labels, dtype=np.int64)


def predict_ensembled_batch(model, raw_sequences, device):
    y_pred = []
    with torch.no_grad():
        for x_raw in raw_sequences:
            draws = np.stack([preprocess_sequence(x_raw) for _ in range(INFERENCE_ENSEMBLE_SIZE)])
            batch = torch.tensor(draws, dtype=torch.float32).to(device)
            probs = torch.softmax(model(batch), dim=1).mean(dim=0)
            y_pred.append(int(torch.argmax(probs).item()))
    return np.array(y_pred)


def permute_band(raw_sequences: list[np.ndarray], band_idx: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Shuffle one column's values across the WHOLE test set, sample by
    sample (each parcel gets some other random parcel's value for that
    band, per-timestep-count-preserving since we permute at the parcel
    level using another parcel's full column, resampled/truncated to match
    this parcel's own sequence length)."""
    n = len(raw_sequences)
    donor_order = rng.permutation(n)
    permuted = []
    for i, x_raw in enumerate(raw_sequences):
        donor = raw_sequences[donor_order[i]]
        donor_col = donor[:, band_idx]
        # match this parcel's own sequence length by resampling the donor column
        idxs = rng.integers(0, len(donor_col), size=len(x_raw))
        x_new = x_raw.copy()
        x_new[:, band_idx] = donor_col[idxs]
        permuted.append(x_new)
    return permuted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--dump-json", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedTimeSeriesTransformer(input_dim=INPUT_DIM, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    raw_sequences, labels = load_all_fetched(exclude_codes=set(args.exclude))
    indices = np.arange(len(labels))
    _, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=labels)
    test_raw = [raw_sequences[i] for i in test_idx]
    test_labels = labels[test_idx]
    print(f"held-out test set: {len(test_idx)} parcels")

    baseline_pred = predict_ensembled_batch(model, test_raw, device)
    baseline_acc = accuracy_score(test_labels, baseline_pred)
    print(f"baseline accuracy (unpermuted): {baseline_acc:.4f}")

    rng = np.random.default_rng(PERMUTATION_SEED)
    importances = []
    for band_idx, band_name in enumerate(SENTINEL2_L1C_BANDS):
        permuted_raw = permute_band(test_raw, band_idx, rng)
        permuted_pred = predict_ensembled_batch(model, permuted_raw, device)
        permuted_acc = accuracy_score(test_labels, permuted_pred)
        drop = baseline_acc - permuted_acc
        importances.append({"band": band_name, "accuracy_drop": drop, "permuted_accuracy": permuted_acc})
        print(f"  {band_name:5s}  permuted_acc={permuted_acc:.4f}  drop={drop:+.4f}")

    importances.sort(key=lambda x: -x["accuracy_drop"])

    if args.dump_json:
        with open(args.dump_json, "w") as f:
            json.dump(
                {"baseline_accuracy": baseline_acc, "n_test": len(test_idx), "importances": importances},
                f,
                indent=2,
            )
        print(f"wrote {args.dump_json}")


if __name__ == "__main__":
    main()
