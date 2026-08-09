"""
model.py — copied from backend/app/models/architecture.py, not imported.

Importing the deployed `app` package here would pull FastAPI and unrelated
backend dependencies into a notebook/training environment for no benefit — the
existing single-country script already made this same tradeoff (its own copy of
this class), this just continues the established pattern.
"""

import numpy as np
import torch
import torch.nn as nn


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
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class ImprovedTimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes,
        num_regions=1,
        d_model=64,
        nhead=4,
        num_layers=3,
        dim_feedforward=64,
        dropout=0.2,
    ):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model), nn.LayerNorm(d_model)
        )
        # Region conditioning: a learned per-region vector added to every time
        # step of the projected sequence (broadcast, same mechanism as
        # PositionalEncoding below) -- gives the model an explicit "which
        # region is this" signal so it can learn region-specific growth-curve
        # shapes for a shared class label instead of averaging them together.
        # num_regions=1 (the default) disables this entirely -- forward()
        # requires no region_id and behaves exactly as before.
        self.region_embedding = nn.Embedding(num_regions, d_model) if num_regions > 1 else None
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, region_id=None):
        x = self.input_projection(x)
        if self.region_embedding is not None:
            if region_id is None:
                raise ValueError("model was built with num_regions > 1 -- region_id is required")
            x = x + self.region_embedding(region_id).unsqueeze(1)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)
