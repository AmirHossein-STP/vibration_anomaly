"""Thin torch.utils.data.Dataset wrapper around normalized multi-branch arrays.

Kept separate from dataset.py so the actual data-engineering code (parsing,
splitting, caching) has no torch dependency, same division as vibration_ad.

__getitem__ returns a dict {"slow": tensor, "medium": tensor, "fast": tensor}
per cycle -- torch's default collate_fn handles a dict of tensors natively
(batches each key independently), so no custom collate function is needed.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class MultiBranchDataset(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray]):
        self.data = {group: torch.from_numpy(arr).float() for group, arr in arrays.items()}
        lengths = {group: len(arr) for group, arr in self.data.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"All branches must have the same number of cycles, got {lengths}")
        self._n = next(iter(lengths.values()))

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {group: tensor[idx] for group, tensor in self.data.items()}
