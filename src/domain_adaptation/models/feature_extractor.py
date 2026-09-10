"""Shared CNN feature extractor (G_f) -- the part of the network every
domain's spectrograms pass through, and the part the domain-adversarial
loss shapes to be domain-invariant.

Same 4-layer stride-2 conv shape as vibration_ad.models.autoencoder's
encoder (reused idea, not reused code -- this one feeds a classifier +
domain head, not a decoder). Deliberately NOT ResNet-18 as the original
plan proposed -- see DOMAIN_ADAPTATION_PLAN.md's "Revisions" section: a
CPU-only Kaggle kernel training across two combined datasets needs a model
small enough to iterate on quickly.
"""
from __future__ import annotations

import torch
from torch import nn


class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_shape: tuple[int, int], base_channels: int = 16, feature_dim: int = 256):
        super().__init__()
        freq_bins, time_frames = input_shape
        if freq_bins % 16 != 0 or time_frames % 16 != 0:
            raise ValueError(
                f"input_shape {input_shape} must be divisible by 16 (four stride-2 layers); "
                "adjust configs/domain_adaptation.yaml's common/stft params."
            )
        c = base_channels
        self.conv = nn.Sequential(
            nn.Conv2d(1, c, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c * 2, c * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(c * 4, c * 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c * 8),
            nn.ReLU(inplace=True),
        )
        flat_dim = c * 8 * (freq_bins // 16) * (time_frames // 16)
        self.fc = nn.Linear(flat_dim, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv(x)
        z = z.flatten(1)
        return self.fc(z)


def select_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
