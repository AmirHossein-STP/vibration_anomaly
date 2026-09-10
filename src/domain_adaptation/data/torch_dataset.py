"""Thin torch.utils.data.Dataset wrapper around a normalized spectrogram
array plus its per-window domain id / label. Kept separate from dataset.py
so the actual data engineering (windowing, STFT, resampling, role
splitting, caching) has no torch dependency -- same reasoning as
vibration_ad.data.torch_dataset.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class LabeledDomainDataset(Dataset):
    """Yields (spectrogram, domain_id, universal_label) triples. Used for
    every role (SOURCE_TRAIN/VAL, TARGET_TEST) except TARGET_ADAPT, whose
    labels are never used for training -- see UnlabeledDomainDataset.
    """

    def __init__(self, spectrograms: np.ndarray, domain_ids: np.ndarray, labels: np.ndarray):
        # (N, F, T) -> (N, 1, F, T): add the single-channel dim conv2d expects.
        self.data = torch.from_numpy(spectrograms).unsqueeze(1).float()
        self.domain_ids = torch.from_numpy(np.asarray(domain_ids)).long()
        self.labels = torch.from_numpy(np.asarray(labels)).long()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.data[idx], self.domain_ids[idx], self.labels[idx]


class UnlabeledDomainDataset(Dataset):
    """Yields (spectrogram, domain_id) pairs -- for TARGET_ADAPT, whose
    rows are all healthy by construction (data/dataset.py) but whose
    labels are deliberately never read during training, only the domain id
    (for the discriminator) and the features themselves (for the
    healthy-anchor loss).
    """

    def __init__(self, spectrograms: np.ndarray, domain_ids: np.ndarray):
        self.data = torch.from_numpy(spectrograms).unsqueeze(1).float()
        self.domain_ids = torch.from_numpy(np.asarray(domain_ids)).long()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.domain_ids[idx]
