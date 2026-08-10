"""
run_train.py — Phase 6: load whatever fetch checkpoints exist, train, evaluate.

Brief GPU session (the model is tiny — this finishes in minutes regardless of
how many countries' data went into it). Loads every per-country *_fetched.pkl
in CHECKPOINT_DIR; BreizhCrops fold-in is a train-time flag here (--breizhcrops),
not baked into the fetch step, per the plan's decision.

Run: ../.venv/bin/python run_train.py [--breizhcrops] [--epochs 20] [--windowing]
"""

import argparse
import json
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from config import CHECKPOINT_DIR, INPUT_DIM, SAVE_PATH
from dataset import FixedDrawDataset, ParcelSequenceDataset
from metrics import evaluate_ensembled, evaluate_single_draw
from model import ImprovedTimeSeriesTransformer
from regions import NUM_REGIONS, region_id_for_checkpoint
from taxonomy import CLASS_NAME_TO_ID, NUM_CLASSES


def load_all_fetched(
    include_breizhcrops: bool, exclude_codes: set[str] | None = None
) -> tuple[list[np.ndarray], list, np.ndarray, np.ndarray]:
    """Combine every per-country *_fetched.pkl (and optionally BZH_fetched.pkl)
    into flat raw_sequences / dates_list / labels / region_ids arrays. Region id
    is derived from which checkpoint file a parcel came from -- the fetch
    pipeline already segregates by source, so no extra bookkeeping is needed.

    exclude_codes skips specific region codes outright -- for a country whose
    background fetch (run_fetch.py) is still mid-run when a retrain is kicked
    off, so a moving-target partial fetch doesn't silently get folded in (see
    the DE_BB "partial fetch" mistake this project already made once)."""
    raw_sequences, dates_list, labels, region_ids = [], [], [], []
    exclude_codes = exclude_codes or set()

    checkpoint_files = sorted(CHECKPOINT_DIR.glob("*_fetched.pkl"))
    if not include_breizhcrops:
        checkpoint_files = [p for p in checkpoint_files if not p.name.startswith("BZH_")]
    checkpoint_files = [
        p for p in checkpoint_files if p.name.removesuffix("_fetched.pkl") not in exclude_codes
    ]

    if not checkpoint_files:
        raise FileNotFoundError(
            f"No *_fetched.pkl files found in {CHECKPOINT_DIR} — run run_fetch.py "
            f"(and optionally run_breizhcrops.py / run_agrifieldnet.py) first"
        )

    for ckpt_path in checkpoint_files:
        region_id = region_id_for_checkpoint(ckpt_path.name)
        with open(ckpt_path, "rb") as f:
            results = pickle.load(f)
        n_before = len(raw_sequences)
        for v in results.values():
            if v["x_raw"].shape[0] < 1:
                continue
            raw_sequences.append(v["x_raw"])
            dates_list.append(v["dates"])
            labels.append(CLASS_NAME_TO_ID[v["classname"]])
            region_ids.append(region_id)
        print(f"  loaded {len(raw_sequences) - n_before} parcels from {ckpt_path.name}")

    return raw_sequences, dates_list, np.array(labels, dtype=np.int64), np.array(region_ids, dtype=np.int64)


def train(
    train_dataset: ParcelSequenceDataset,
    test_dataset: FixedDrawDataset,
    epochs: int,
    device: torch.device,
    num_regions: int,
) -> ImprovedTimeSeriesTransformer:
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    model = ImprovedTimeSeriesTransformer(
        input_dim=INPUT_DIM, num_classes=NUM_CLASSES, num_regions=num_regions,
        d_model=64, nhead=4, num_layers=3, dim_feedforward=64, dropout=0.2,
    ).to(device)
    use_regions = model.region_embedding is not None

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = OneCycleLR(optimizer, max_lr=0.005, epochs=epochs, steps_per_epoch=len(train_loader), pct_start=0.3)

    best_acc = -1.0  # not 0.0: if val_acc never exceeds 0 (possible on tiny/early
    # runs), "> 0.0" would never trigger and no checkpoint would ever be saved,
    # crashing the unconditional torch.load below. -1.0 guarantees epoch 1 always
    # saves at least one checkpoint.
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_region, batch_y in train_loader:
            batch_x, batch_region, batch_y = batch_x.to(device), batch_region.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x, batch_region) if use_regions else model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch_x, batch_region, batch_y in test_loader:
                batch_x, batch_region, batch_y = batch_x.to(device), batch_region.to(device), batch_y.to(device)
                outputs = model(batch_x, batch_region) if use_regions else model(batch_x)
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        val_acc = correct / total

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)

        print(f"epoch {epoch + 1}/{epochs}  loss={epoch_loss / len(train_loader):.4f}  val_acc={val_acc:.4f}")

    print(f"best val acc: {best_acc:.4f}, saved to {SAVE_PATH}")
    model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--breizhcrops", action="store_true", help="fold BreizhCrops into the combined training set")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--windowing", action="store_true", help="enable partial-season window augmentation (off by default per plan)")
    parser.add_argument("--regions", action="store_true", help="condition the model on region/country identity (see regions.py)")
    parser.add_argument("--dump-json", type=str, default=None,
                         help="write final single-draw + ensembled eval dicts to this path, "
                              "for hand-updating app/data/model_stats.py after a retrain")
    parser.add_argument("--exclude", action="append", default=[],
                         help="region code to skip entirely (repeatable) -- for a country whose "
                              "background fetch is still mid-run")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("*** No CUDA GPU detected — training will run on CPU. Model is tiny, should still be fast. ***")
    else:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    print("Loading fetched parcels...")
    raw_sequences, dates_list, labels, region_ids = load_all_fetched(
        include_breizhcrops=args.breizhcrops, exclude_codes=set(args.exclude)
    )
    print(f"Total: {len(raw_sequences)} parcels, {len(set(labels.tolist()))} classes present, "
          f"{len(set(region_ids.tolist()))} regions present, region-conditioning={'ON' if args.regions else 'off'}")

    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=labels)

    train_raw = [raw_sequences[i] for i in train_idx]
    train_dates = [dates_list[i] for i in train_idx]
    train_labels = labels[train_idx]
    train_regions = region_ids[train_idx]
    test_raw = [raw_sequences[i] for i in test_idx]
    test_labels = labels[test_idx]
    test_regions = region_ids[test_idx]

    train_dataset = ParcelSequenceDataset(train_raw, train_dates, train_labels, train_regions, enable_windowing=args.windowing)
    test_dataset = FixedDrawDataset(test_raw, test_labels, test_regions)

    print(f"train: {len(train_dataset)}  test: {len(test_dataset)}  windowing={'ON' if args.windowing else 'off'}")

    num_regions = NUM_REGIONS if args.regions else 1
    model = train(train_dataset, test_dataset, args.epochs, device, num_regions)

    print("\nFinal evaluation:")
    single_draw_report = evaluate_single_draw(model, test_dataset, device)
    ensembled_report = evaluate_ensembled(model, test_raw, test_labels, test_regions, device)

    if args.dump_json:
        with open(args.dump_json, "w") as f:
            json.dump(
                {
                    "n_train": len(train_dataset),
                    "n_test": len(test_dataset),
                    "single_draw": single_draw_report,
                    "ensembled": ensembled_report,
                },
                f,
                indent=2,
            )
        print(f"\nwrote eval report to {args.dump_json}")


if __name__ == "__main__":
    main()
