"""
eval_partial_range.py — quantify how much a partial-season date range (e.g.
the Jan1-Jul15 range a user actually queried, vs. the full Jan1-Dec31 range
every checkpoint has ever been trained on) degrades accuracy, using data
already on disk -- no new CDSE fetches needed.

Reuses the exact same held-out test split run_train.py produces (same
random_state=42 stratified split over the same load order), so this is a
genuine apples-to-apples comparison against the reported val accuracy, not a
new/different sample.

For each held-out test parcel, builds a second raw sequence by keeping only
observations whose date falls within the first WINDOW_DAYS days of that
parcel's own date range (mimicking "Jan1 through mid-way" -- parcels are
fetched Jan1-Dec31, so day-of-range 0-196 IS Jan1-Jul16), then runs the
existing production preprocess_sequence()/predict() path, unmodified,
against both the full sequence and the windowed one.

Run: ../.venv/bin/python eval_partial_range.py --checkpoint PATH [--window-days 196]
"""

import argparse
import pickle
from datetime import timedelta

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from config import CHECKPOINT_DIR, INPUT_DIM
from dataset import preprocess_sequence
from model import ImprovedTimeSeriesTransformer
from taxonomy import CLASS_NAME_TO_ID, NUM_CLASSES

INFERENCE_ENSEMBLE_SIZE = 16


def load_all_fetched(exclude_codes: set[str] | None = None):
    exclude_codes = exclude_codes or set()
    raw_sequences, dates_list, labels = [], [], []
    for ckpt_path in sorted(CHECKPOINT_DIR.glob("*_fetched.pkl")):
        if ckpt_path.name.removesuffix("_fetched.pkl") in exclude_codes:
            continue
        with open(ckpt_path, "rb") as f:
            results = pickle.load(f)
        for v in results.values():
            if v["x_raw"].shape[0] < 1:
                continue
            raw_sequences.append(v["x_raw"])
            dates_list.append(v["dates"])
            labels.append(CLASS_NAME_TO_ID[v["classname"]])
    return raw_sequences, dates_list, np.array(labels, dtype=np.int64)


def window_sequence(x_raw: np.ndarray, dates: list, window_days: int) -> np.ndarray:
    """Keep only observations within the first `window_days` days of this
    parcel's own date range -- mirrors what fetching Jan1-through-mid-season
    (instead of Jan1-Dec31) would actually return for the same field."""
    start = min(dates)
    cutoff = start + timedelta(days=window_days)
    mask = [d <= cutoff for d in dates]
    windowed = x_raw[mask]
    return windowed if len(windowed) > 0 else x_raw  # degrade to full seq rather than crash on empty


def predict_ensembled(model, x_raw, device):
    draws = np.stack([preprocess_sequence(x_raw) for _ in range(INFERENCE_ENSEMBLE_SIZE)])
    batch = torch.tensor(draws, dtype=torch.float32).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(batch), dim=1).mean(dim=0)
    return int(torch.argmax(probs).item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--window-days", type=int, default=196,
                         help="196 = Jan1 through Jul16, matching the reported mismatch")
    parser.add_argument("--exclude", action="append", default=[],
                         help="region code to skip entirely (repeatable)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedTimeSeriesTransformer(input_dim=INPUT_DIM, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    raw_sequences, dates_list, labels = load_all_fetched(exclude_codes=set(args.exclude))
    indices = np.arange(len(labels))
    _, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=labels)
    print(f"checkpoint: {args.checkpoint}")
    print(f"held-out test set: {len(test_idx)} parcels (same split run_train.py uses)")

    y_true, y_pred_full, y_pred_windowed = [], [], []
    n_dropped_to_zero = 0
    for i in test_idx:
        x_raw, dates, label = raw_sequences[i], dates_list[i], labels[i]
        windowed = window_sequence(x_raw, dates, args.window_days)
        if len(windowed) == len(x_raw):
            n_dropped_to_zero += 0  # no-op, just keeping the counter's intent explicit
        y_true.append(label)
        y_pred_full.append(predict_ensembled(model, x_raw, device))
        y_pred_windowed.append(predict_ensembled(model, windowed, device))

    y_true = np.array(y_true)
    acc_full = accuracy_score(y_true, y_pred_full)
    acc_windowed = accuracy_score(y_true, y_pred_windowed)
    print(f"\nfull Jan1-Dec31 sequence  accuracy: {acc_full:.4f}  (n={len(y_true)})")
    print(f"first {args.window_days}-day window accuracy: {acc_windowed:.4f}  (n={len(y_true)})")
    print(f"delta: {acc_windowed - acc_full:+.4f}")


if __name__ == "__main__":
    main()
