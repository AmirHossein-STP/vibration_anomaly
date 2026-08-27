"""Per-sensor z-score normalization, fit on TRAIN healthy cycles only.

Direct analogue of vibration_ad's SpectrogramNormalizer (per-frequency-bin
there, per-sensor here): each of the 17 sensors has a very different native
scale (bar vs. Watts vs. l/min vs. degrees C vs. %), so one global
mean/std per branch would leave the smaller-scale sensors nearly flat after
normalization. Per-sensor (per-channel) stats, computed separately within
each branch, fixes that. Fit only on TRAIN so nothing about val/test/faulty
cycles leaks into preprocessing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class MultiBranchNormalizer:
    # one (n_sensors, 1) mean/std pair per branch, e.g. {"slow": (mean, std), ...}
    stats: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    def fit(self, arrays: dict[str, np.ndarray]) -> "MultiBranchNormalizer":
        for group, arr in arrays.items():
            # arr: (N, n_sensors, samples) -> per-sensor mean/std over (N, samples)
            mean = arr.mean(axis=(0, 2), keepdims=True)[0]  # (n_sensors, 1)
            std = arr.std(axis=(0, 2), keepdims=True)[0] + 1e-8
            self.stats[group] = (mean, std)
        return self

    def transform(self, arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if not self.stats:
            raise RuntimeError("MultiBranchNormalizer.fit() must be called before transform().")
        out = {}
        for group, arr in arrays.items():
            mean, std = self.stats[group]
            out[group] = ((arr - mean) / std).astype(np.float32)
        return out

    def save(self, path: Path) -> None:
        payload = {}
        for group, (mean, std) in self.stats.items():
            payload[f"{group}_mean"] = mean
            payload[f"{group}_std"] = std
        np.savez(path, **payload)

    @classmethod
    def load(cls, path: Path, groups: list[str]) -> "MultiBranchNormalizer":
        data = np.load(path)
        stats = {group: (data[f"{group}_mean"], data[f"{group}_std"]) for group in groups}
        return cls(stats=stats)
