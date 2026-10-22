"""Computed order tracking (features/cot.py) and the WP5 encoder phase.

The check notes/17 §2 requires before Ottawa 2018 is scored: on a
constant-speed signal, angle-domain features must reproduce the frozen
time-domain ones.
"""
from __future__ import annotations

import numpy as np

from domain_adaptation.data.public_datasets import encoder_phase
from domain_adaptation.features.cot import angle_resample, cot_envelope_order_features
from domain_adaptation.features.envelope import FEATURE_NAMES, PU_6203, envelope_order_features

FS = 12000.0


def _faulty_signal(seconds: float, shaft_hz_fn, order: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Impulses at `order` x shaft angle exciting a 3 kHz resonance, plus noise.
    Returns (signal, phase in revolutions)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * FS)) / FS
    phase = np.cumsum(shaft_hz_fn(t)) / FS
    impacts = np.flatnonzero(np.diff(np.floor(phase * order)) > 0)
    x = rng.normal(0, 0.15, len(t))
    ring = np.exp(-np.arange(120) / 15.0) * np.sin(2 * np.pi * 3000 * np.arange(120) / FS)
    for i in impacts:
        seg = ring[: len(t) - i]
        x[i:i + len(seg)] += seg
    return x, phase


def test_constant_speed_cot_matches_time_domain():
    rpm = 900.0
    x, phase = _faulty_signal(1.0, lambda t: np.full_like(t, rpm / 60), PU_6203.bpfo)
    for kw in ({}, {"whiten": True, "band_hz": None, "squared": True}):
        f_time = envelope_order_features(x, FS, rpm, PU_6203, **kw)
        f_cot = cot_envelope_order_features(x, FS, phase, PU_6203, **kw)
        # the fault family must dominate in both, and the six peaks agree closely
        assert f_time["bpfo_h1"] > 2 * f_time["bpfi_h1"] and f_cot["bpfo_h1"] > 2 * f_cot["bpfi_h1"]
        peaks = ["bpfo_h1", "bpfo_h2", "bpfo_h3", "bpfi_h1", "bpfi_h2", "bpfi_h3"]
        a = np.log([f_time[k] for k in peaks]); b = np.log([f_cot[k] for k in peaks])
        assert np.corrcoef(a, b)[0, 1] > 0.9   # measured 0.94-0.99: interpolation + bin placement
        assert set(f_cot) == set(FEATURE_NAMES)


def test_cot_recovers_fault_order_under_a_speed_sweep():
    sweep = lambda t: 13.0 + 15.0 * t / 1.5          # 13 -> 28 Hz in 1.5 s, like Ottawa 2018
    x, phase = _faulty_signal(1.5, sweep, PU_6203.bpfi)
    f_cot = cot_envelope_order_features(x, FS, phase, PU_6203)
    f_naive = envelope_order_features(x, FS, 60 * float(np.mean(sweep(np.arange(len(x)) / FS))), PU_6203)
    # tracking keeps the inner-race peak sharp; a single mean rpm smears it
    assert f_cot["bpfi_h1"] > 2 * f_cot["bpfo_h1"]
    assert f_cot["bpfi_h1"] > 1.5 * f_naive["bpfi_h1"]


def test_angle_resample_length_and_monotonic_guard():
    ph = np.linspace(0, 20, 5000)
    y, revs = angle_resample(np.sin(2 * np.pi * 5 * ph), ph, 128)
    assert abs(revs - 20) < 1 / 128 and len(y) == int(20 * 128)
    try:
        angle_resample(np.zeros(3), np.array([0.0, 1.0, 0.5]), 128)
        raise AssertionError("non-monotone phase accepted")
    except ValueError:
        pass


def test_encoder_phase_counts_revolutions():
    fs, cpr, shaft_hz = 200000.0, 1024, 20.0
    t = np.arange(int(fs * 0.5)) / fs
    enc = 4.3 * ((np.floor(t * shaft_hz * cpr * 2) % 2) == 1)  # square wave, cpr cycles per rev
    ph = encoder_phase(enc, fs, cpr)
    ok = np.isfinite(ph)
    rate = (ph[ok][-1] - ph[ok][0]) / (ok.sum() / fs)
    assert abs(rate - shaft_hz) / shaft_hz < 0.01
