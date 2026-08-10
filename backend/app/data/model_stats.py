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
        "IN": {"name": "India (AgriFieldNet)", "parcels": 1009, "status": "complete", "total_area_hectares": 442.9},
        "CUSTOM": {"name": "Hand-labeled (Varanasi wheat/mustard + Jabalpur meadow)", "parcels": 88,
                   "status": "complete", "total_area_hectares": 14.9,
                   "note": "backend/local_parcels/ -- see app/services/custom_parcels.py"},
    },
    "overall": {
        "single_draw_accuracy": 0.6691,
        "ensembled_accuracy": 0.7020,
        "macro_f1": 0.54,
        "weighted_f1": 0.69,
        "n_test": 1671,
        "note": "held-out 20% split of training data (8351 parcels total), ensembled = 16-draw average matching production predict()",
    },
    "per_class": [
        {"class": "rapeseed", "precision": 0.93, "recall": 0.95, "f1": 0.94, "support": 150},
        {"class": "maize", "precision": 0.89, "recall": 0.94, "f1": 0.91, "support": 180},
        {"class": "potatoes", "precision": 0.82, "recall": 0.75, "f1": 0.78, "support": 156},
        {"class": "barley", "precision": 0.78, "recall": 0.76, "f1": 0.77, "support": 150},
        {"class": "sunflower", "precision": 0.75, "recall": 0.74, "f1": 0.75, "support": 125},
        {"class": "vineyards", "precision": 0.74, "recall": 0.69, "f1": 0.71, "support": 74},
        {"class": "meadow", "precision": 0.64, "recall": 0.73, "f1": 0.68, "support": 150},
        {"class": "wheat", "precision": 0.56, "recall": 0.73, "f1": 0.64, "support": 166},
        {"class": "fruit", "precision": 0.54, "recall": 0.68, "f1": 0.60, "support": 150},
        {"class": "triticale", "precision": 0.67, "recall": 0.50, "f1": 0.57, "support": 150},
        {"class": "sugarcane", "precision": 0.42, "recall": 0.74, "f1": 0.53, "support": 27},
        {"class": "fallow", "precision": 0.50, "recall": 0.53, "f1": 0.52, "support": 30},
        {"class": "nuts", "precision": 0.76, "recall": 0.32, "f1": 0.45, "support": 78},
        {"class": "gram", "precision": 0.75, "recall": 0.27, "f1": 0.40, "support": 11},
        {"class": "mustard", "precision": 0.40, "recall": 0.39, "f1": 0.39, "support": 36},
        {"class": "lentil", "precision": 0.00, "recall": 0.00, "f1": 0.00, "support": 15},
        {"class": "rice", "precision": 0.00, "recall": 0.00, "f1": 0.00, "support": 14},
        {"class": "garlic", "precision": 0.00, "recall": 0.00, "f1": 0.00, "support": 9},
    ],
    "per_region_training_split": [
        {"region": "AT", "name": "Austria", "accuracy": 0.7908, "n": 325},
        {"region": "BE_VLG", "name": "Belgium-Flanders", "accuracy": 0.7885, "n": 279},
        {"region": "DE_LS", "name": "Germany-Lower Saxony", "accuracy": 0.7466, "n": 292},
        {"region": "DE_BB", "name": "Germany-Brandenburg", "accuracy": 0.6761, "n": 284},
        {"region": "DE_NRW", "name": "Germany-North Rhine-Westphalia", "accuracy": 0.6617, "n": 266},
        {"region": "CUSTOM", "name": "Hand-labeled (Varanasi/Jabalpur)", "accuracy": 0.6471, "n": 17,
         "note": "tiny held-out slice (20% of 88 parcels) -- directionally useful only, not a solid estimate"},
        {"region": "IN", "name": "India", "accuracy": 0.4760, "n": 208},
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
        "CUSTOM region held-out slice is only 17 parcels -- real signal, not a statistically solid estimate.",
        "live_pipeline numbers below predate this retrain -- not yet re-measured against the current checkpoint.",
    ],
}
