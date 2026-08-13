"""
model_stats.py — static, precomputed validation numbers for GET /api/v1/model-info.

These are real numbers from the actual multi-country + India retrain and its
live-pipeline validation (see progress.md for the full narrative and how each
number was produced) — not live-computed per request, per CLAUDE.md's Stage 4
plan ("this is a one-time computation, not per-request"). Update this file by
hand after a retrain; there is currently no automated pipeline that
regenerates it.
"""

MODEL_INFO = {
    "checkpoint": "multicountry_india_custom_18class.pth",
    "num_classes": 18,
    "training_data": {
        "AT": {"name": "Austria", "parcels": 1650, "status": "complete", "total_area_hectares": 1971.1},
        "BE_VLG": {"name": "Belgium-Flanders", "parcels": 1464, "status": "complete", "total_area_hectares": 2655.9},
        "DE_BB": {"name": "Germany-Brandenburg", "parcels": 1411, "status": "complete", "total_area_hectares": 11300.5},
        "DE_LS": {"name": "Germany-Lower Saxony", "parcels": 1416, "status": "complete", "total_area_hectares": 3874.2},
        "DE_NRW": {"name": "Germany-North Rhine-Westphalia", "parcels": 1440, "status": "complete", "total_area_hectares": 3411.8},
        "DK": {"name": "Denmark", "parcels": 1459, "status": "complete", "total_area_hectares": 6309.1},
        "BZH": {"name": "Brittany (France)", "parcels": 1212, "status": "complete", "total_area_hectares": 3078.7},
        "IN": {"name": "India (AgriFieldNet)", "parcels": 1009, "status": "complete", "total_area_hectares": 442.9},
        "CUSTOM": {"name": "Hand-labeled (Varanasi wheat/mustard + Jabalpur meadow)", "parcels": 88,
                   "status": "complete", "total_area_hectares": 14.9,
                   "note": "backend/local_parcels/ -- see app/services/custom_parcels.py"},
    },
    "overall": {
        "single_draw_accuracy": 0.6682,
        "ensembled_accuracy": 0.7211,
        "macro_f1": 0.53,
        "weighted_f1": 0.71,
        "n_test": 2230,
        "note": "held-out 20% split of training data (11149 parcels total), ensembled = 16-draw average matching production predict(). "
                "Estonia (EE) excluded -- still mid-fetch (972/1253) when this retrain ran.",
    },
    "per_class": [
        {"class": "rapeseed", "precision": 0.94, "recall": 0.96, "f1": 0.95, "support": 210},
        {"class": "maize", "precision": 0.88, "recall": 0.94, "f1": 0.91, "support": 240},
        {"class": "potatoes", "precision": 0.86, "recall": 0.77, "f1": 0.81, "support": 215},
        {"class": "barley", "precision": 0.81, "recall": 0.75, "f1": 0.78, "support": 210},
        {"class": "sunflower", "precision": 0.78, "recall": 0.77, "f1": 0.78, "support": 131},
        {"class": "vineyards", "precision": 0.79, "recall": 0.72, "f1": 0.75, "support": 104},
        {"class": "wheat", "precision": 0.63, "recall": 0.79, "f1": 0.70, "support": 252},
        {"class": "meadow", "precision": 0.61, "recall": 0.71, "f1": 0.66, "support": 210},
        {"class": "triticale", "precision": 0.66, "recall": 0.60, "f1": 0.63, "support": 210},
        {"class": "fruit", "precision": 0.57, "recall": 0.68, "f1": 0.62, "support": 210},
        {"class": "sugarcane", "precision": 0.44, "recall": 0.63, "f1": 0.52, "support": 27},
        {"class": "fallow", "precision": 0.52, "recall": 0.47, "f1": 0.49, "support": 30},
        {"class": "rice", "precision": 0.57, "recall": 0.29, "f1": 0.38, "support": 14},
        {"class": "nuts", "precision": 0.56, "recall": 0.26, "f1": 0.35, "support": 96},
        {"class": "mustard", "precision": 0.21, "recall": 0.19, "f1": 0.20, "support": 36},
        {"class": "lentil", "precision": 0.00, "recall": 0.00, "f1": 0.00, "support": 15},
        {"class": "gram", "precision": 0.00, "recall": 0.00, "f1": 0.00, "support": 11},
        {"class": "garlic", "precision": 0.00, "recall": 0.00, "f1": 0.00, "support": 9},
    ],
    "per_region_training_split": [
        {"region": "AT", "name": "Austria", "accuracy": 0.8283, "n": 332},
        {"region": "BE_VLG", "name": "Belgium-Flanders", "accuracy": 0.8125, "n": 272},
        {"region": "DE_LS", "name": "Germany-Lower Saxony", "accuracy": 0.7518, "n": 274},
        {"region": "DK", "name": "Denmark", "accuracy": 0.7533, "n": 304},
        {"region": "DE_BB", "name": "Germany-Brandenburg", "accuracy": 0.7119, "n": 295},
        {"region": "BZH", "name": "Brittany (France)", "accuracy": 0.6995, "n": 213},
        {"region": "DE_NRW", "name": "Germany-North Rhine-Westphalia", "accuracy": 0.6911, "n": 327},
        {"region": "CUSTOM", "name": "Hand-labeled (Varanasi/Jabalpur)", "accuracy": 0.6316, "n": 19,
         "note": "tiny held-out slice (~20% of 88 parcels) -- directionally useful only, not a solid estimate"},
        {"region": "IN", "name": "India", "accuracy": 0.4124, "n": 194},
    ],
    "live_pipeline": {
        "note": "real POST /api/v1/predict calls against the running server, not the training script's internal eval -- "
                "figures below predate this retrain (previous checkpoint) and have not yet been re-run against the "
                "current one; kept for historical context, not as a claim about current live behavior.",
        "by_region": [
            {"region": "AT", "name": "Austria", "correct": 15, "n": 20, "accuracy": 0.75, "held_out": False},
            {"region": "BE_VLG", "name": "Belgium-Flanders", "correct": 16, "n": 22, "accuracy": 0.7273, "held_out": False},
            {"region": "DE_BB", "name": "Germany-Brandenburg", "correct": 7, "n": 21, "accuracy": 0.3333, "held_out": False,
             "note": "measured against the prior checkpoint, when DE_BB's fetch was still partial (184 parcels) -- "
                     "DE_BB is now a complete, full-size training region (1411 parcels), see training_data above"},
            {"region": "IN", "name": "India", "correct": 31, "n": 58, "accuracy": 0.5345, "held_out": True,
             "note": "fields explicitly confirmed excluded from training (geometry-matched)"},
        ],
        "india_by_class": [
            {"class": "wheat", "correct": 12, "n": 19, "accuracy": 0.6316, "note": "was 0/9 = 0% before India training data existed"},
            {"class": "maize", "correct": 13, "n": 14, "accuracy": 0.9286},
            {"class": "fallow", "correct": 6, "n": 10, "accuracy": 0.60},
            {"class": "mustard", "correct": 0, "n": 15, "accuracy": 0.0,
             "note": "unexplained failure despite 150 training examples -- see progress.md"},
        ],
    },
    "known_gaps": [
        "No shadow masking (cloud masking exists; shadow does not).",
        "No minimum-observation-count quality gate before returning a prediction.",
        "No graceful handling of CDSE failures -- surfaces as a raw 500.",
        "Positional encoding uses ordinal step index, not real day-of-year.",
        "lentil, garlic, and 3 excluded India classes are permanently data-limited by the source dataset.",
        "mustard fails despite adequate training data in both EU and India+CUSTOM sources -- open, unexplained.",
        "Region-conditioning was built and A/B tested but showed no benefit -- not deployed.",
        "CUSTOM region held-out slice is only 19 parcels -- real signal, not a statistically solid estimate.",
        "Estonia (EE) excluded from this retrain -- fetch was still incomplete (972/1253) when it ran.",
        "live_pipeline numbers below predate this retrain -- not yet re-measured against the current checkpoint.",
    ],
    "confusion_matrix": {
        "note": "ensembled (16-draw) predictions on the held-out test set, n=2230 -- rows=true, cols=predicted, "
                "same run as 'overall'/'per_class' above (training/eurocrops_pipeline/run_train.py --breizhcrops --exclude EE --dump-json)",
        "class_names": [
            "meadow", "wheat", "barley", "triticale", "rapeseed", "maize", "sunflower", "vineyards",
            "fruit", "nuts", "potatoes", "mustard", "sugarcane", "lentil", "rice", "gram", "garlic", "fallow",
        ],
        "matrix": [
            [149, 1, 3, 5, 0, 2, 1, 1, 43, 5, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 198, 8, 27, 0, 2, 1, 0, 1, 1, 1, 7, 5, 0, 0, 0, 0, 0],
            [3, 17, 158, 17, 3, 1, 3, 2, 3, 0, 3, 0, 0, 0, 0, 0, 0, 0],
            [9, 49, 18, 125, 4, 0, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 4, 201, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [3, 2, 0, 1, 0, 226, 0, 0, 1, 0, 6, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 2, 1, 1, 7, 101, 3, 3, 1, 12, 0, 0, 0, 0, 0, 0, 0],
            [3, 0, 0, 2, 0, 3, 1, 75, 18, 2, 0, 0, 0, 0, 0, 0, 0, 0],
            [42, 0, 1, 2, 3, 0, 1, 7, 143, 10, 1, 0, 0, 0, 0, 0, 0, 0],
            [29, 0, 1, 1, 1, 3, 1, 5, 30, 25, 0, 0, 0, 0, 0, 0, 0, 0],
            [3, 2, 2, 3, 1, 9, 17, 1, 8, 1, 165, 0, 0, 0, 0, 0, 0, 3],
            [0, 19, 0, 0, 0, 3, 0, 0, 0, 0, 1, 7, 6, 0, 0, 0, 0, 0],
            [0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 0, 0, 0, 0, 1],
            [0, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0, 0, 9],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0],
            [0, 5, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 5, 0, 3, 0, 0, 14],
        ],
    },
    "band_correlation": {
        "note": "Pearson correlation across 238,955 real fetched Sentinel-2 observations (all 7 trained "
                "regions -- AT/BE_VLG/DE_BB/DE_LS/DE_NRW/IN/CUSTOM; DK excluded, still mid-fetch). Bands "
                "in SENTINEL2_L1C_BANDS order (the model's actual input column order, not numeric order).",
        "bands": ["B1", "B10", "B11", "B12", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9"],
        "matrix": [
            [1.00, 0.22, -0.27, -0.08, 0.99, 0.97, 0.94, 0.94, 0.73, 0.57, 0.56, 0.51, 0.70],
            [0.22, 1.00, -0.06, 0.00, 0.22, 0.21, 0.21, 0.21, 0.15, 0.11, 0.12, 0.10, 0.25],
            [-0.27, -0.06, 1.00, 0.92, -0.20, -0.11, -0.02, -0.03, -0.07, -0.06, -0.06, -0.01, -0.20],
            [-0.08, 0.00, 0.92, 1.00, -0.02, 0.06, 0.17, 0.13, -0.09, -0.15, -0.14, -0.12, -0.12],
            [0.99, 0.22, -0.20, -0.02, 1.00, 0.99, 0.97, 0.97, 0.75, 0.59, 0.58, 0.54, 0.70],
            [0.97, 0.21, -0.11, 0.06, 0.99, 1.00, 0.99, 0.99, 0.80, 0.65, 0.64, 0.60, 0.71],
            [0.94, 0.21, -0.02, 0.17, 0.97, 0.99, 1.00, 0.99, 0.73, 0.57, 0.56, 0.52, 0.67],
            [0.94, 0.21, -0.03, 0.13, 0.97, 0.99, 0.99, 1.00, 0.81, 0.67, 0.66, 0.62, 0.74],
            [0.73, 0.15, -0.07, -0.09, 0.75, 0.80, 0.73, 0.81, 1.00, 0.97, 0.97, 0.95, 0.80],
            [0.57, 0.11, -0.06, -0.15, 0.59, 0.65, 0.57, 0.67, 0.97, 1.00, 1.00, 1.00, 0.72],
            [0.56, 0.12, -0.06, -0.14, 0.58, 0.64, 0.56, 0.66, 0.97, 1.00, 1.00, 0.99, 0.75],
            [0.51, 0.10, -0.01, -0.12, 0.54, 0.60, 0.52, 0.62, 0.95, 1.00, 0.99, 1.00, 0.69],
            [0.70, 0.25, -0.20, -0.12, 0.70, 0.71, 0.67, 0.74, 0.80, 0.72, 0.75, 0.69, 1.00],
        ],
    },
    "feature_importance": {
        "note": "Permutation importance on the same held-out test set (n=2230, EE excluded) -- each band's "
                "values shuffled across the test set (breaking its real signal, preserving its marginal "
                "distribution and every other band untouched), re-run through the deployed model's own "
                "ensembled predict(), accuracy drop vs. the unpermuted baseline measures how much the model "
                "actually relies on that band. Model-agnostic, no gradients needed -- "
                "training/eurocrops_pipeline/eval_feature_importance.py.",
        "baseline_accuracy": 0.7260,
        "importances": [
            {"band": "B6", "accuracy_drop": 0.5296},
            {"band": "B8", "accuracy_drop": 0.4628},
            {"band": "B5", "accuracy_drop": 0.4296},
            {"band": "B11", "accuracy_drop": 0.3511},
            {"band": "B3", "accuracy_drop": 0.2937},
            {"band": "B12", "accuracy_drop": 0.2830},
            {"band": "B4", "accuracy_drop": 0.2794},
            {"band": "B1", "accuracy_drop": 0.2224},
            {"band": "B8A", "accuracy_drop": 0.1951},
            {"band": "B2", "accuracy_drop": 0.1924},
            {"band": "B7", "accuracy_drop": 0.1803},
            {"band": "B9", "accuracy_drop": 0.1601},
            {"band": "B10", "accuracy_drop": 0.0072},
        ],
    },
}
