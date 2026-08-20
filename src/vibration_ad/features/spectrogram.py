"""Turn raw signal windows into normalized log-magnitude STFT spectrograms.

We use a plain STFT (linear frequency axis) rather than a Mel-spectrogram.
Mel scaling warps the frequency axis to match human pitch perception, which
is the right thing for speech/music but actively unhelpful here: bearing
fault energy shows up at specific mechanical frequencies (BPFO, BPFI, BSF,
shaft rate and their harmonics -- see notes/05_glossary.md), and Mel binning
would blur exactly the higher-frequency detail that separates one fault
type/severity from another. Linear STFT keeps that resolution intact and
also drops the librosa dependency entirely (scipy.signal.stft is enough).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import stft

from vibration_ad.config import STFTConfig, WindowingConfig


def expected_shape(windowing: WindowingConfig, stft_cfg: STFTConfig) -> tuple[int, int]:
    """(n_freq_bins, n_time_frames) that windows_to_log_spectrograms() will
    produce for the given config, computed without running any STFT.

    Freq bins = nperseg // 2 (the Nyquist bin is dropped, see below).
    Time frames follows scipy's no-padding, no-boundary STFT frame count:
    1 + (segment_length - nperseg) // hop_length.
    """
    n_freq = stft_cfg.nperseg // 2
    n_time = (windowing.segment_length - stft_cfg.nperseg) // stft_cfg.hop_length + 1
    return n_freq, n_time


def windows_to_log_spectrograms(
    windows: np.ndarray, fs_hz: int, stft_cfg: STFTConfig
) -> np.ndarray:
    """windows: (N, segment_length) -> log-magnitude spectrograms (N, F, T).

    boundary=None, padded=False so the output time-frame count is exactly
    predictable from (segment_length, nperseg, hop_length) -- see
    expected_shape(). The Nyquist bin (last freq bin) is dropped only to
    make F a round, pooling-friendly number for the conv autoencoder; for
    these signals it carries negligible energy anyway.
    """
    _, _, Zxx = stft(
        windows,
        fs=fs_hz,
        window="hann",
        nperseg=stft_cfg.nperseg,
        noverlap=stft_cfg.noverlap,
        boundary=None,
        padded=False,
        axis=-1,
    )
    magnitude = np.abs(Zxx)[:, :-1, :]  # drop Nyquist bin -> (N, nperseg/2, T)
    return np.log(magnitude + stft_cfg.log_epsilon).astype(np.float32)


@dataclass
class SpectrogramNormalizer:
    """Per-frequency-bin z-score normalization, fit on TRAIN healthy data only.

    Per-bin (not one global scalar) matters here: low frequencies carry far
    more energy than high frequencies in these signals, so a single global
    mean/std would leave high-frequency bins -- where subtle fault harmonics
    often show up -- almost flat after normalization. Fitting only on
    training-split healthy spectrograms (never val/test/faulty data) avoids
    leaking any information about the evaluation set into preprocessing.
    """

    mean: np.ndarray | None = None  # shape (F, 1)
    std: np.ndarray | None = None  # shape (F, 1)

    def fit(self, spectrograms: np.ndarray) -> "SpectrogramNormalizer":
        self.mean = spectrograms.mean(axis=(0, 2), keepdims=True)[0]
        self.std = spectrograms.std(axis=(0, 2), keepdims=True)[0] + 1e-8
        return self

    def transform(self, spectrograms: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("SpectrogramNormalizer.fit() must be called before transform().")
        return ((spectrograms - self.mean) / self.std).astype(np.float32)

    def save(self, path: Path) -> None:
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: Path) -> "SpectrogramNormalizer":
        data = np.load(path)
        return cls(mean=data["mean"], std=data["std"])
