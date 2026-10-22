"""Geometry-free envelope-order features (notes/23 §2): for a bearing whose
geometry is unknown. Same CPW + squared-envelope order spectrum as method
v1.0, read in 135 bins of 0.1 order over 1.5-15 orders instead of at six
named fault lines. Each bin is its peak divided by the envelope sum over
0.5-12 orders (the v1 normalisation), so a z-score against the unit's own
healthy windows means the same thing it does for the named lines.
"""
from __future__ import annotations

import numpy as np

from domain_adaptation.features.cot import angle_resample, time_envelope

LO, HI, WIDTH = 1.5, 15.0, 0.1
EDGES = np.round(np.arange(LO, HI + WIDTH / 2, WIDTH), 4)
BIN_NAMES = [f"o{e:05.2f}" for e in EDGES[:-1]]
BIN_CENTRES = (EDGES[:-1] + EDGES[1:]) / 2


def read_bins(orders: np.ndarray, spec: np.ndarray) -> dict[str, float]:
    ref = (orders >= 0.5) & (orders <= 12.0)
    total = float(spec[ref].sum()) + 1e-12
    idx = np.searchsorted(EDGES, orders, side="right") - 1
    out = np.zeros(len(BIN_NAMES))
    ok = (idx >= 0) & (idx < len(BIN_NAMES))
    np.maximum.at(out, idx[ok], spec[ok])
    return dict(zip(BIN_NAMES, out / total))


def _spectrum(env: np.ndarray, d: float):
    env = env - env.mean()
    return np.fft.rfftfreq(len(env), d=d), np.abs(np.fft.rfft(env * np.hanning(len(env))))


def geofree_features(signal: np.ndarray, fs_hz: float, rpm: float, geometry=None) -> dict[str, float]:
    env = time_envelope(signal, fs_hz, band_hz=None, whiten=True, squared=True)
    freqs, spec = _spectrum(env, 1.0 / fs_hz)
    return read_bins(freqs / (rpm / 60.0), spec)


def geofree_features_cot(signal: np.ndarray, fs_hz: float, phase_revs: np.ndarray, geometry=None,
                         samples_per_rev: int = 128) -> dict[str, float]:
    env = time_envelope(signal, fs_hz, band_hz=None, whiten=True, squared=True)
    env_a, _ = angle_resample(env, phase_revs, samples_per_rev)
    orders, spec = _spectrum(env_a, 1.0 / samples_per_rev)
    return read_bins(orders, spec)
