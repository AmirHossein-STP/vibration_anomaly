"""Shaft speed from the vibration itself (MASTER_PLAN WP10, notes/20).

For machines with no tachometer. Every fault order is a multiple of shaft
rate, so an unknown speed must be estimated before any physics feature can be
read. The shaft harmonics (1X, 2X, ...) that imbalance and misalignment leave
on almost every rotating machine are used: the candidate rate whose first
eight harmonics carry the most (log) spectral magnitude wins. Harmonic k
resolves the rate k times more finely than 1X alone.
"""
from __future__ import annotations

import numpy as np

N_HARMONICS = 8
PAD = 16
GRID_HZ = 0.01


def estimate_shaft_hz(signal: np.ndarray, fs_hz: float, lo_hz: float, hi_hz: float,
                      n_harmonics: int = N_HARMONICS) -> float:
    x = signal - signal.mean()
    n = len(x) * PAD
    mag = np.abs(np.fft.rfft(x * np.hanning(len(x)), n=n))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs_hz)
    logmag = np.log(mag + 1e-12 * mag.max())
    cand = np.arange(lo_hz, hi_hz + GRID_HZ / 2, GRID_HZ)
    k = np.arange(1, n_harmonics + 1)
    pts = np.outer(cand, k)
    score = np.interp(pts, freqs, logmag, right=np.nan)
    score = np.where(np.isnan(score), logmag.min(), score).sum(1)
    return float(cand[np.argmax(score)])
