"""Small MLP heads sitting on top of the shared feature extractor: the
fault classifier (G_y) and the domain discriminator (G_d). Structurally
identical (Linear -> ReLU -> Dropout -> Linear) -- they differ only in
output width (n_classes vs. n_domains) and in what feeds them (the
discriminator sits behind a GradientReversalLayer, wired up in dann.py).
"""
from __future__ import annotations

import torch
from torch import nn


class MLPHead(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, n_outputs: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_outputs),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)
