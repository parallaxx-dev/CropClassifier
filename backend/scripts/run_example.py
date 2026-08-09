"""
run_example.py — End-to-end example: load dataset + model, run inference
on one parcel, generate the 3-panel figure.

This is the reference your friend should follow when wiring this into a
web backend (e.g. call load_dataset_and_model() once at startup, then
call run_single_parcel() per request).
"""

import breizhcrops as bzh

from config import CHECKPOINT_PATH, REGION, DEVICE, CLASS_NAMES, NUM_CLASSES, MODEL_PARAMS
from model import load_model
from inference import predict
from visualize import plot_parcel_result


def load_dataset_and_model():
    """Call once at app startup — downloads/loads dataset + model into memory."""
    print("🔄 Loading BreizhCrops dataset...")
    dataset = bzh.BreizhCrops(REGION)

    sample_x, _, _ = dataset[0]
    num_features = sample_x.shape[1]

    model = load_model(
        checkpoint_path=CHECKPOINT_PATH,
        input_dim=num_features,
        num_classes=NUM_CLASSES,
        device=DEVICE,
        model_params=MODEL_PARAMS,
    )
    print("✅ Model loaded")

    gdf_all = dataset.geodataframe()
    gdf_all["id"] = gdf_all["id"].astype(str)
    print("✅ Geodataframe loaded")

    return dataset, model, gdf_all


def run_single_parcel(dataset, model, gdf_all, parcel_id, save_path=None):
    """
    Given a parcel_id, find it in the dataset, run inference, and generate
    the 3-panel figure. Returns the prediction dict.
    """
    # Find the parcel's index in the dataset
    sample_idx = None
    for idx in range(len(dataset)):
        _, _, p_id = dataset[idx]
        if str(p_id) == str(parcel_id):
            sample_idx = idx
            break
    if sample_idx is None:
        raise ValueError(f"Parcel {parcel_id} not found in dataset")

    x_raw, y_true, p_id = dataset[sample_idx]
    y_true = int(y_true)

    # Ground truth: prefer the geodataframe's own classname (guaranteed to
    # match the geometry being plotted) over the raw label index
    row = gdf_all.loc[gdf_all["id"] == str(p_id)]
    true_label = row["classname"].iloc[0] if len(row) > 0 else CLASS_NAMES.get(y_true, "UNKNOWN")

    result = predict(model, x_raw)

    if save_path or True:  # always render for the demo
        plot_parcel_result(
            parcel_id=p_id,
            true_label=true_label,
            pred_label=result["pred_label"],
            confidence=result["confidence"],
            gdf_all=gdf_all,
            save_path=save_path,
        )

    return {
        "parcel_id": str(p_id),
        "true_label": true_label,
        **result,
    }


if __name__ == "__main__":
    dataset, model, gdf_all = load_dataset_and_model()

    # Example: run on a known parcel ID
    EXAMPLE_PARCEL_ID = 6079945
    output = run_single_parcel(
        dataset, model, gdf_all, EXAMPLE_PARCEL_ID,
        save_path=f"parcel_{EXAMPLE_PARCEL_ID}_result.png",
    )
    print(output)
