"""
train_eurocrops_local.py — local GPU training for the EuroCrops-based crop
classifier retraining pipeline.

Standalone companion to eurocrops_model_training.ipynb (the Colab version) —
same logic (same class scheme, same band order/resampling/normalization
matching the deployed backend exactly, same memory-safety fixes), adapted to
run as a plain script against your own GPU instead of a Colab runtime.

Why this exists: the deployed model was trained on BreizhCrops' frozen,
pre-extracted Sentinel-2 archive (old, fixed satellite calibration). The
deployed app fetches live data from CDSE at prediction time, under whatever
calibration is *currently* live — a confirmed mismatch. This script instead
pairs EuroCrops' labeled parcel geometries with imagery fetched live through
the same CDSE API the deployed app uses, so training and inference are
always measured by the same instrument.

---------------------------------------------------------------------------
Setup (once):
    # Install a CUDA-matched PyTorch build — plain `pip install torch` may
    # resolve to a CPU-only wheel. Pick the command for your CUDA version at
    # https://pytorch.org/get-started/locally/, e.g. for CUDA 12.1:
    pip install torch --index-url https://download.pytorch.org/whl/cu121

    pip install geopandas sentinelhub shapely pandas scikit-learn requests python-dotenv

Credentials:
    Reads CDSE_SH_CLIENT_ID / CDSE_SH_CLIENT_SECRET from environment
    variables. By default loads them from backend/.env (the same file the
    deployed FastAPI backend already uses — reuses the same credentials).
    Never hardcode real values in this file or commit a filled-in .env.

Run:
    python train_eurocrops_local.py
---------------------------------------------------------------------------
"""

import gc
import io
import json
import os
import pickle
import time
import zipfile
from datetime import date
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
from sentinelhub import CRS, DataCollection, Geometry, SentinelHubStatistical, SHConfig
from shapely.geometry import mapping as shapely_mapping
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, TensorDataset

# ============================================
# Config
# ============================================
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

CDSE_SH_CLIENT_ID = os.environ.get("CDSE_SH_CLIENT_ID")
CDSE_SH_CLIENT_SECRET = os.environ.get("CDSE_SH_CLIENT_SECRET")
CDSE_SH_BASE_URL = "https://sh.dataspace.copernicus.eu"
CDSE_SH_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

EUROCROPS_YEAR = 2018  # France's currently-available EuroCrops season
EUROCROPS_ZIP_URL = "https://zenodo.org/api/records/8229128/files/FR_2018.zip/content"

OUTPUT_DIR = Path(__file__).resolve().parent / "eurocrops_output"
EUROCROPS_LOCAL_DIR = OUTPUT_DIR / "eurocrops_fr_2018"
ZIP_LOCAL_PATH = OUTPUT_DIR / "FR_2018.zip"
CHECKPOINT_PATH = OUTPUT_DIR / "eurocrops_fetch_progress.pkl"
SAVE_PATH = OUTPUT_DIR / "best_transformer_eurocrops.pth"

N_PER_CLASS = int(os.environ.get("N_PER_CLASS", 300))  # stratified sample size per class -
                    # tune based on time/PU budget. Recommended: do a first pass with
                    # N_PER_CLASS=20-30 (env var override) to confirm the whole pipeline
                    # runs end to end before committing to a long full run.
TARGET_SEQ_LEN = 45  # must match the deployed model's expectation exactly
MAX_CLOUD_COVER_PERCENT = 60
REFLECTANCE_SCALE = 1e-4
RESOLUTION_METERS = 10
FETCH_SLEEP_SECONDS = 0.3  # stay comfortably under CDSE's 300 req/min cap

# Exact band order the deployed model expects (verified against breizhcrops' own
# get_default_transform - lexicographic, not natural numeric order). No B2/B10 swap
# workaround here: that swap only existed in BreizhCrops' own stored archive. We're
# fetching real, correctly-labeled bands for both training and inference this time.
SENTINEL2_L1C_BANDS = ["B1", "B10", "B11", "B12", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9"]
_SH_BAND_NAMES = {
    "B1": "B01", "B2": "B02", "B3": "B03", "B4": "B04", "B5": "B05",
    "B6": "B06", "B7": "B07", "B8": "B08", "B8A": "B8A", "B9": "B09",
    "B10": "B10", "B11": "B11", "B12": "B12",
}
_sh_bands = [_SH_BAND_NAMES[b] for b in SENTINEL2_L1C_BANDS]

# Exact same 9-class scheme as the deployed model, mapped from the original French RPG
# codes EuroCrops preserves (verified against EuroCrops' own fr_2018.csv mapping file,
# and identical to backend/breizhcrops_dataset/classmapping.csv already used in this repo).
CODE_TO_CLASS = {
    "ORH": "barley", "ORP": "barley",
    "BTH": "wheat", "BTP": "wheat",
    "CZH": "rapeseed", "CZP": "rapeseed",
    "MID": "corn", "MIE": "corn", "MIS": "corn",
    "TRN": "sunflower",
    "AGR": "orchards", "PFR": "orchards", "PWT": "orchards", "VRG": "orchards",
    "CAB": "nuts", "CTG": "nuts", "NOS": "nuts", "NOX": "nuts", "PIS": "nuts",
    "PPH": "permanent meadows", "PRL": "permanent meadows",
    "PTR": "temporary meadows", "RGA": "temporary meadows",
}
CLASS_NAMES = {
    0: "barley", 1: "wheat", 2: "rapeseed", 3: "corn", 4: "sunflower",
    5: "orchards", 6: "nuts", 7: "permanent meadows", 8: "temporary meadows",
}
CLASS_NAME_TO_ID = {v: k for k, v in CLASS_NAMES.items()}


# ============================================
# Model architecture (must match the deployed backend exactly)
# ============================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ImprovedTimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=64, nhead=4, num_layers=3, dim_feedforward=64, dropout=0.2):
        super().__init__()
        self.input_projection = nn.Sequential(nn.Linear(input_dim, d_model), nn.LayerNorm(d_model))
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)


# ============================================
# Step 1: download + load EuroCrops France, bounded memory throughout
# ============================================
def download_and_sample_parcels():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not EUROCROPS_LOCAL_DIR.exists():
        if not ZIP_LOCAL_PATH.exists():
            print("Downloading EuroCrops FR_2018.zip (~2.6GB)...")
            with requests.get(EUROCROPS_ZIP_URL, stream=True) as resp:
                resp.raise_for_status()
                with open(ZIP_LOCAL_PATH, "wb") as out:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        out.write(chunk)
        print("Extracting...")
        with zipfile.ZipFile(ZIP_LOCAL_PATH) as zf:
            zf.extractall(EUROCROPS_LOCAL_DIR)
        ZIP_LOCAL_PATH.unlink()  # free disk space now that it's extracted
        print("Extracted to", EUROCROPS_LOCAL_DIR)

    shp_path = None
    for root, _, files in os.walk(EUROCROPS_LOCAL_DIR):
        for f in files:
            if f.endswith(".shp"):
                shp_path = os.path.join(root, f)
                break
    assert shp_path is not None, "No .shp found - check EUROCROPS_LOCAL_DIR contents"
    print("Found shapefile:", shp_path)

    # Probe just the schema (1 row - cheap) to confirm the actual column name before
    # doing any real reads.
    probe = gpd.read_file(shp_path, rows=1)
    print("columns:", probe.columns.tolist())

    # NOTE: verify this against the columns printed above - EuroCrops preserves the
    # original per-country code under a country-specific field name; for France this
    # is expected to carry the same RPG codes as CODE_TO_CLASS above (shapefile field
    # names are truncated to 10 chars, commonly "CODE_CULTU" for French RPG-derived
    # data, matching BreizhCrops' own field name since it's the same source system).
    original_code_column = "CODE_CULTU"
    assert original_code_column in probe.columns, (
        f"{original_code_column} not found - check the columns above and update this constant"
    )

    # Read one RPG code at a time, each capped with rows= - not one big query for all
    # 23 codes at once. Common crops (wheat, corn, temporary meadows) can have hundreds
    # of thousands of parcels nationally; capping per code bounds peak memory to a
    # small, known multiple of N_PER_CLASS regardless of how common any crop is.
    oversample_per_code = max(50, N_PER_CLASS)
    frames = []
    for code, cls_name in CODE_TO_CLASS.items():
        code_gdf = gpd.read_file(shp_path, where=f"{original_code_column} = '{code}'", rows=oversample_per_code)
        if len(code_gdf) == 0:
            continue
        code_gdf["classname"] = cls_name
        frames.append(code_gdf[["classname", "geometry"]])
        print(f"{code} ({cls_name}): read {len(code_gdf)} parcels (capped at {oversample_per_code})")
        del code_gdf

    assert len(frames) > 0, (
        "No parcels matched any known code - check original_code_column and the codes "
        "in CODE_TO_CLASS against the actual data (see the columns printed above)"
    )
    gdf_labeled = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)
    del frames
    gc.collect()

    print()
    print("total parcels loaded (bounded):", len(gdf_labeled))
    print(gdf_labeled["classname"].value_counts())

    gdf_labeled = gdf_labeled.to_crs(epsg=4326)

    sampled = (
        gdf_labeled.groupby("classname", group_keys=False)
        .apply(lambda g: g.sample(n=min(N_PER_CLASS, len(g)), random_state=42))
        .reset_index(drop=True)
    )
    print("sampled parcels:", len(sampled))
    print(sampled["classname"].value_counts())

    del gdf_labeled
    gc.collect()
    return sampled


# ============================================
# Step 2: CDSE Statistical API fetch (mirrors backend/app/services/sentinel_fetch.py)
# ============================================
def _sh_config():
    assert CDSE_SH_CLIENT_ID and CDSE_SH_CLIENT_SECRET, (
        "CDSE_SH_CLIENT_ID / CDSE_SH_CLIENT_SECRET not set - put them in backend/.env "
        "(gitignored) or export them as environment variables before running"
    )
    config = SHConfig()
    config.sh_client_id = CDSE_SH_CLIENT_ID
    config.sh_client_secret = CDSE_SH_CLIENT_SECRET
    config.sh_base_url = CDSE_SH_BASE_URL
    config.sh_token_url = CDSE_SH_TOKEN_URL
    return config


def _utm_epsg_for(lon, lat):
    zone = int((lon + 180) / 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


_EVALSCRIPT = f"""
//VERSION=3
function setup() {{
  return {{
    input: [{{
      bands: {json.dumps(_sh_bands + ["dataMask"])},
      units: "DN"
    }}],
    output: [
      {{ id: "bands", bands: {len(SENTINEL2_L1C_BANDS)}, sampleType: "INT16" }},
      {{ id: "dataMask", bands: 1 }}
    ]
  }};
}}
function evaluatePixel(sample) {{
  return {{
    bands: [{", ".join(f"sample.{b}" for b in _sh_bands)}],
    dataMask: [sample.dataMask]
  }};
}}
"""


def fetch_time_series(polygon, start_date, end_date, config, data_collection):
    centroid = polygon.centroid
    geometry = Geometry(shapely_mapping(polygon), crs=CRS.WGS84).transform(
        CRS(_utm_epsg_for(centroid.x, centroid.y))
    )
    request = SentinelHubStatistical(
        aggregation=SentinelHubStatistical.aggregation(
            evalscript=_EVALSCRIPT,
            time_interval=(start_date, end_date),
            aggregation_interval="P1D",
            resolution=(RESOLUTION_METERS, RESOLUTION_METERS),
        ),
        input_data=[SentinelHubStatistical.input_data(data_collection, maxcc=MAX_CLOUD_COVER_PERCENT / 100)],
        geometry=geometry,
        config=config,
    )
    result = request.get_data()[0]

    rows = []
    for interval in result.get("data", []):
        bandsout = interval["outputs"]["bands"]["bands"]
        first = bandsout["B0"]["stats"]
        if first["sampleCount"] == 0 or first["noDataCount"] >= first["sampleCount"]:
            continue
        rows.append([bandsout[f"B{i}"]["stats"]["mean"] * REFLECTANCE_SCALE for i in range(len(SENTINEL2_L1C_BANDS))])
    return np.array(rows, dtype=np.float32)


def fetch_all(sampled):
    sh_config = _sh_config()
    data_collection = DataCollection.SENTINEL2_L1C.define_from("EUROCROPS_TRAIN_L1C", service_url=sh_config.sh_base_url)

    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, "rb") as f:
            results = pickle.load(f)
        print(f"resuming, {len(results)} parcels already fetched")
    else:
        results = {}

    start_date = date(EUROCROPS_YEAR, 1, 1)
    end_date = date(EUROCROPS_YEAR, 12, 31)

    for i, row in sampled.iterrows():
        key = f"{i}"
        if key in results:
            continue
        try:
            x_raw = fetch_time_series(row.geometry, start_date, end_date, sh_config, data_collection)
            if len(x_raw) == 0:
                continue
            results[key] = {"x_raw": x_raw, "classname": row["classname"]}
        except Exception as e:
            print(f"skip {key}: {e}")
        if i % 50 == 0:
            with open(CHECKPOINT_PATH, "wb") as f:
                pickle.dump(results, f)
            print(f"{i}/{len(sampled)} processed, {len(results)} succeeded")
        time.sleep(FETCH_SLEEP_SECONDS)

    with open(CHECKPOINT_PATH, "wb") as f:
        pickle.dump(results, f)
    print("done:", len(results), "parcels fetched")
    return results


# ============================================
# Step 3: preprocess — resample to 45 timesteps + per-sample z-score normalize
# (matches backend/app/models/inference.py exactly)
# ============================================
def preprocess_sequence(x_raw, target_seq_len=TARGET_SEQ_LEN):
    seq_len = x_raw.shape[0]
    idxs = np.random.choice(seq_len, target_seq_len, replace=seq_len < target_seq_len)
    idxs.sort()
    x_resampled = x_raw[idxs]
    mean = x_resampled.mean(axis=0)
    std = x_resampled.std(axis=0)
    std[std == 0] = 1.0
    return (x_resampled - mean) / std


def build_dataset(results):
    X, y = [], []
    for v in results.values():
        if v["x_raw"].shape[0] < 1:
            continue
        X.append(preprocess_sequence(v["x_raw"]))
        y.append(CLASS_NAME_TO_ID[v["classname"]])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    print("X shape:", X.shape, " y shape:", y.shape)
    return X, y


# ============================================
# Step 4: train
# ============================================
def train(X, y):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print(
            "\n*** No CUDA GPU detected - training will run on CPU and be much slower. ***\n"
            "Check: `nvidia-smi` shows your RTX 3050, and torch was installed with a CUDA\n"
            "build matching your driver (see the Setup note at the top of this file) -\n"
            "`python -c \"import torch; print(torch.__version__, torch.version.cuda)\"`\n"
            "should print a CUDA version, not None.\n"
        )
    else:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
        batch_size=128, shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)),
        batch_size=128, shuffle=False,
    )

    model = ImprovedTimeSeriesTransformer(
        input_dim=X.shape[2], num_classes=len(CLASS_NAMES),
        d_model=64, nhead=4, num_layers=3, dim_feedforward=64, dropout=0.2,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = OneCycleLR(optimizer, max_lr=0.005, epochs=20, steps_per_epoch=len(train_loader), pct_start=0.3)

    best_acc = 0.0
    for epoch in range(20):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        val_acc = correct / total

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)

        print(f"epoch {epoch + 1}/20  loss={epoch_loss / len(train_loader):.4f}  val_acc={val_acc:.4f}")

    print(f"best val acc: {best_acc:.4f}, saved to {SAVE_PATH}")


def main():
    sampled = download_and_sample_parcels()
    results = fetch_all(sampled)
    X, y = build_dataset(results)
    train(X, y)

    print()
    print("IMPORTANT: the deployed sentinel_fetch.py's _SH_BAND_NAMES currently cross-wires")
    print("B2/B10 on purpose, to match a quirk in BreizhCrops' own stored data. This model")
    print("trains on real, correctly-labeled bands (no swap) - deploying this checkpoint")
    print("requires reverting that cross-wire back to a normal 1:1 band mapping first, or")
    print("every prediction will silently feed the new model B2/B10 swapped.")


if __name__ == "__main__":
    main()
