"""
dataset.py — ParcelSequenceDataset: the per-epoch resampling fix.

The bug this replaces: train_eurocrops_local.py's build_dataset() called
preprocess_sequence() exactly ONCE per parcel to build a static array, reused for
all 20 training epochs via a plain TensorDataset — the model never actually saw
fresh resampling, despite that being the whole point of the resample-with-
replacement design (mirrors what the original breizhcrops.BreizhCrops package's
own __getitem__ did, which DID resample fresh each access — see progress.md's
correction note on the ensembling work).

Fix: store raw (pre-resample) variable-length sequences, call preprocess_sequence
fresh inside __getitem__ each time. A shuffling DataLoader then gives genuine
per-epoch augmentation with zero training-loop restructuring.

Windowing hook: present, but gated by enable_windowing (default False per the
plan's decision — evaluate its effect separately, later, once the metrics
harness below can isolate it from every other change in this retrain).
"""

import random
from datetime import date, timedelta

import numpy as np
import torch
from torch.utils.data import Dataset

from config import TARGET_SEQ_LEN

MIN_WINDOW_DAYS = 30
MAX_WINDOW_DAYS = 60


def preprocess_sequence(x_raw: np.ndarray, target_seq_len: int = TARGET_SEQ_LEN) -> np.ndarray:
    """Unchanged from backend/app/models/inference.py / train_eurocrops_local.py —
    resample (with replacement if shorter, without if longer) to target_seq_len,
    sort chronologically, per-sample z-score normalize. Do not modify this
    function's logic; only when/how often it's called changes in this file."""
    seq_len = x_raw.shape[0]
    idxs = np.random.choice(seq_len, target_seq_len, replace=seq_len < target_seq_len)
    idxs.sort()
    x_resampled = x_raw[idxs]
    mean = x_resampled.mean(axis=0)
    std = x_resampled.std(axis=0)
    std[std == 0] = 1.0
    return (x_resampled - mean) / std


def _random_window(x_raw: np.ndarray, dates: list[date]) -> tuple[np.ndarray, list[date]]:
    """Pick a random date-based sub-window of the raw sequence, applied BEFORE
    preprocess_sequence's resample step. Falls back to the full sequence if the
    window would leave too few observations (preprocess_sequence needs at least
    1 row; fewer than ~5 makes per-sample z-score normalization meaningless)."""
    if len(dates) < 5:
        return x_raw, dates

    day_span = (max(dates) - min(dates)).days
    if day_span < MIN_WINDOW_DAYS:
        return x_raw, dates

    window_len = random.randint(MIN_WINDOW_DAYS, min(MAX_WINDOW_DAYS, day_span))
    latest_start_offset = day_span - window_len
    start_offset = random.randint(0, latest_start_offset)
    window_start = min(dates) + timedelta(days=start_offset)
    window_end = window_start + timedelta(days=window_len)

    mask = [window_start <= d <= window_end for d in dates]
    if sum(mask) < 5:
        return x_raw, dates
    return x_raw[mask], [d for d, m in zip(dates, mask) if m]


class ParcelSequenceDataset(Dataset):
    def __init__(
        self,
        raw_sequences: list[np.ndarray],
        dates_list: list[list[date]],
        labels: np.ndarray,
        region_ids: np.ndarray,
        enable_windowing: bool = False,
    ):
        assert len(raw_sequences) == len(dates_list) == len(labels) == len(region_ids)
        self.raw_sequences = raw_sequences
        self.dates_list = dates_list
        self.labels = labels
        self.region_ids = region_ids
        self.enable_windowing = enable_windowing

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        x_raw = self.raw_sequences[idx]
        dates = self.dates_list[idx]
        if self.enable_windowing:
            x_raw, dates = _random_window(x_raw, dates)
        x = preprocess_sequence(x_raw)
        return torch.tensor(x, dtype=torch.float32), int(self.region_ids[idx]), int(self.labels[idx])


class FixedDrawDataset(Dataset):
    """For test/validation: resample-once-reuse-forever is correct here (not the
    bug it was for train) — a stable set is what makes an epoch-over-epoch
    val_acc curve interpretable. Seeded so the fixed draw is reproducible."""

    def __init__(self, raw_sequences: list[np.ndarray], labels: np.ndarray, region_ids: np.ndarray, seed: int = 42):
        assert len(raw_sequences) == len(labels) == len(region_ids)
        rng_state = np.random.get_state()
        np.random.seed(seed)
        self.x = np.stack([preprocess_sequence(seq) for seq in raw_sequences])
        np.random.set_state(rng_state)
        self.labels = labels
        self.region_ids = region_ids

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        return torch.tensor(self.x[idx], dtype=torch.float32), int(self.region_ids[idx]), int(self.labels[idx])
