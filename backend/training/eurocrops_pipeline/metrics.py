"""
metrics.py — per-class evaluation. No per-class metrics exist anywhere in this
repo today (checked during the earlier code audit) — with this many simultaneous
methodology changes landing in one retrain (new countries, new taxonomy, the
resampling fix, possibly BreizhCrops fold-in), this is the only way to attribute
any accuracy change to a specific cause rather than guessing.

Two evaluations, both worth having:
- single-draw: one FixedDrawDataset pass — matches how a plain val_acc curve
  during training is computed.
- ensembled: INFERENCE_ENSEMBLE_SIZE-draw average per test parcel, mirroring the
  deployed predict()'s approach in backend/app/models/inference.py — the more
  honest "real" accuracy number since it matches production behavior, not a
  single lucky/unlucky resample.
"""

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from dataset import preprocess_sequence
from taxonomy import CLASS_NAMES

INFERENCE_ENSEMBLE_SIZE = 16  # matches backend/app/config.py's deployed default


def evaluate_single_draw(model, fixed_dataset, device) -> dict:
    model.eval()
    x = torch.stack([fixed_dataset[i][0] for i in range(len(fixed_dataset))]).to(device)
    region_id_list = [fixed_dataset[i][1] for i in range(len(fixed_dataset))]
    region_id = torch.tensor(region_id_list, dtype=torch.long).to(device)
    y_true = np.array([fixed_dataset[i][2] for i in range(len(fixed_dataset))])
    with torch.no_grad():
        logits = model(x, region_id) if model.region_embedding is not None else model(x)
        y_pred = torch.argmax(logits, dim=1).cpu().numpy()
    return _report(y_true, y_pred, "single-draw", region_ids=np.array(region_id_list))


def evaluate_ensembled(
    model, raw_sequences: list[np.ndarray], labels: np.ndarray, region_ids: np.ndarray, device
) -> dict:
    """Mirrors the deployed predict()'s ensembling exactly: K independent
    resampling draws per parcel, batched through the model in one forward pass,
    softmax-averaged before taking argmax."""
    model.eval()
    y_pred = []
    with torch.no_grad():
        for x_raw, region in zip(raw_sequences, region_ids):
            draws = np.stack([preprocess_sequence(x_raw) for _ in range(INFERENCE_ENSEMBLE_SIZE)])
            batch = torch.tensor(draws, dtype=torch.float32).to(device)
            if model.region_embedding is not None:
                region_batch = torch.full((batch.shape[0],), int(region), dtype=torch.long, device=device)
                logits = model(batch, region_batch)
            else:
                logits = model(batch)
            probs = torch.softmax(logits, dim=1).mean(dim=0)
            y_pred.append(int(torch.argmax(probs).item()))
    return _report(labels, np.array(y_pred), "ensembled", region_ids=np.asarray(region_ids))


def _report(y_true: np.ndarray, y_pred: np.ndarray, label: str, region_ids: np.ndarray | None = None) -> dict:
    present_ids = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    target_names = [CLASS_NAMES[i] for i in present_ids]

    print(f"\n=== {label} evaluation ===")
    print(classification_report(y_true, y_pred, labels=present_ids, target_names=target_names, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=present_ids)
    print("confusion matrix (rows=true, cols=predicted):")
    print("            " + " ".join(f"{n[:8]:>8s}" for n in target_names))
    for i, row in enumerate(cm):
        print(f"{target_names[i]:12s}" + " ".join(f"{v:8d}" for v in row))

    overall_acc = float((y_true == y_pred).mean())
    print(f"\noverall accuracy: {overall_acc:.4f}")

    per_region_acc = None
    if region_ids is not None:
        from regions import REGION_NAMES
        per_region_acc = {}
        print("accuracy by region:")
        for rid in sorted(set(region_ids.tolist())):
            mask = region_ids == rid
            acc = float((y_true[mask] == y_pred[mask]).mean())
            name = REGION_NAMES.get(rid, f"region_{rid}")
            per_region_acc[name] = {"accuracy": acc, "n": int(mask.sum())}
            print(f"  {name:10s} n={int(mask.sum()):4d}  acc={acc:.4f}")

    return {
        "overall_accuracy": overall_acc,
        "confusion_matrix": cm.tolist(),
        "class_names": target_names,
        "per_region_accuracy": per_region_acc,
        "report": classification_report(
            y_true, y_pred, labels=present_ids, target_names=target_names, zero_division=0, output_dict=True
        ),
    }
