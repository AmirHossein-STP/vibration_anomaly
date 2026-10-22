"""Envelope-order features under TIME-VARYING speed: computed order tracking
(notes/17 §2, MASTER_PLAN WP5; Shen et al. 2021).

envelope.py assumes one rpm per window. That holds for CWRU, Paderborn,
MFPT, Ottawa 2023 and XJTU-SY, but not for Ottawa 2018, where shaft speed
sweeps 13–28 Hz inside one 10 s file. There a fault order would smear across
several bins within a window. So here:

1. the envelope is computed in TIME exactly as envelope.envelope_spectrum
   does (same band / whitening / squaring), because the resonance carrier is
   a fixed-Hz structural property;
2. the envelope is resampled onto a uniform grid of SHAFT ANGLE using the
   encoder phase, so a fault at k orders is a fixed frequency of the
   resampled signal;
3. the spectrum is read in orders with the same features as
   envelope.envelope_order_features.

envelope.py is deliberately not modified: its source is part of every
cache fingerprint, and the frozen method must stay byte-identical. The
feature reading below therefore copies its v1 logic (n_harmonics, tol_orders,
norm="sum"). tests/test_cot.py checks that on a constant-speed signal the two
agree.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

from domain_adaptation.features.envelope import BearingGeometry, _band_energy


def time_envelope(signal: np.ndarray, fs_hz: float, band_hz=(1500.0, 5500.0),
                  whiten: bool = False, squared: bool = False) -> np.ndarray:
    """Steps 1 of envelope.envelope_spectrum, returned before the FFT."""
    if whiten:
        X = np.fft.rfft(signal)
        signal = np.fft.irfft(X / (np.abs(X) + 1e-12), n=len(signal))
    if band_hz is None:
        banded = signal
    else:
        nyq = fs_hz / 2.0
        sos = butter(4, [band_hz[0] / nyq, min(band_hz[1] / nyq, 0.99)], btype="bandpass", output="sos")
        banded = sosfiltfilt(sos, signal)
    env = np.abs(hilbert(banded))
    return env ** 2 if squared else env


def angle_resample(x: np.ndarray, phase_revs: np.ndarray, samples_per_rev: int) -> tuple[np.ndarray, float]:
    """x sampled at the (monotone) shaft phases `phase_revs` -> uniform in angle.
    Returns (resampled, n_revolutions)."""
    if np.any(np.diff(phase_revs) <= 0):
        raise ValueError("phase must increase strictly (shaft reversing or encoder fault)")
    n = int(np.floor((phase_revs[-1] - phase_revs[0]) * samples_per_rev))
    grid = phase_revs[0] + np.arange(n) / samples_per_rev
    return np.interp(grid, phase_revs, x), n / samples_per_rev


def order_features(env_angle: np.ndarray, samples_per_rev: int, geometry: BearingGeometry,
                   n_harmonics: int = 3, tol_orders: float = 0.08) -> dict[str, float]:
    """envelope.envelope_order_features' v1 reading, on an angle-domain envelope."""
    env = env_angle - env_angle.mean()
    spec = np.abs(np.fft.rfft(env * np.hanning(len(env))))
    orders = np.fft.rfftfreq(len(env), d=1.0 / samples_per_rev)
    ref = (orders >= 0.5) & (orders <= 12.0)
    total = float(spec[ref].sum()) + 1e-12
    feats: dict[str, float] = {}
    for label, base in (("bpfo", geometry.bpfo), ("bpfi", geometry.bpfi)):
        harmonics = []
        for h in range(1, n_harmonics + 1):
            e = _band_energy(orders, spec, base * h, tol_orders) / total
            feats[f"{label}_h{h}"] = e
            harmonics.append(e)
        feats[f"{label}_sum"] = float(np.sum(harmonics))
    feats["bpfi_sidebands"] = sum(
        _band_energy(orders, spec, geometry.bpfi + s, tol_orders) / total for s in (-1.0, 1.0))
    feats["ftf"] = _band_energy(orders, spec, geometry.ftf, tol_orders) / total
    band = spec[ref]
    feats["env_kurtosis"] = float(((band - band.mean()) ** 4).mean() / (band.std() ** 4 + 1e-12))
    feats["env_crest"] = float(band.max() / (np.sqrt((band ** 2).mean()) + 1e-12))
    feats["env_flatness"] = float(np.exp(np.log(band + 1e-12).mean()) / (band.mean() + 1e-12))
    return feats


def cot_envelope_order_features(signal: np.ndarray, fs_hz: float, phase_revs: np.ndarray,
                                geometry: BearingGeometry, samples_per_rev: int = 128,
                                band_hz=(1500.0, 5500.0), whiten: bool = False,
                                squared: bool = False) -> dict[str, float]:
    """One window: time envelope -> angle resample -> order features."""
    env = time_envelope(signal, fs_hz, band_hz, whiten=whiten, squared=squared)
    env_a, _ = angle_resample(env, phase_revs, samples_per_rev)
    return order_features(env_a, samples_per_rev, geometry)
