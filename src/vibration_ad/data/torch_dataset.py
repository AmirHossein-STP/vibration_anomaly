"""Thin torch.utils.data.Dataset wrapper around a normalized spectrogram array.

Kept separate from dataset.py so the actual data-engineering code (windowing,
STFT, splitting, caching) has no torch dependency and can be unit-tested or
reused without a deep learning framework installed.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class SpectrogramDataset(Dataset):
    def __init__(self, spectrograms: np.ndarray):
        # (N, F, T) -> (N, 1, F, T): add the single-channel dim conv2d expects.
        self.data = torch.from_numpy(spectrograms).unsqueeze(1).float()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]
