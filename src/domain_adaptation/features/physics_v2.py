"""Physics features v2 (MASTER_PLAN WP3, notes/19): every rolling-element
fault family, read from the CPW + squared-envelope order spectrum.

v1 (envelope.py, frozen) reads BPFO and BPFI × 1–3. v2 adds:
- harmonics to 5 for both races;
- shaft-rate sidebands around BPFI (an inner-race defect rotates through the load zone);
- the ball-spin family (BSF × 1–3; a ball defect strikes both races, so 2×BSF is often the strongest);
- the cage family (FTF × 1–2).

Tolerance grows with order (±max(0.08, 2% of the order)), because slip and
speed error shift harmonic k by k times as much.

Both paths are here: constant speed (time-domain windows) and
order-tracked (cot.py's angle-domain envelope). They share the reading.
"""
from __future__ import annotations

import numpy as np

from domain_adaptation.features.cot import angle_resample, time_envelope
from domain_adaptation.features.envelope import BearingGeometry, _band_energy

REF_BAND = (0.5, 30.0)
TOL_ABS, TOL_REL = 0.08, 0.02
FAMILIES = {
    "outer": [f"bpfo_h{h}" for h in range(1, 6)],
    "inner": [f"bpfi_h{h}" for h in range(1, 6)] + ["bpfi_sb_m1", "bpfi_sb_p1"],
    "ball": [f"bsf_h{h}" for h in range(1, 4)],
    "cage": ["ftf_h1", "ftf_h2"],
}
V2_NAMES = [n for fam in FAMILIES.values() for n in fam]
FAMILY_LABEL = {"inner": 1, "outer": 2, "ball": 3, "cage": 4}


def bsf(g: BearingGeometry) -> float:
    """Ball spin frequency in orders of shaft rate."""
    r = g.ball_diameter / g.pitch_diameter * np.cos(np.radians(g.contact_angle_deg))
    return g.pitch_diameter / (2 * g.ball_diameter) * (1 - r ** 2)


def read_families(orders: np.ndarray, spec: np.ndarray, g: BearingGeometry) -> dict[str, float]:
    ref = (orders >= REF_BAND[0]) & (orders <= REF_BAND[1])
    total = float(spec[ref].sum()) + 1e-12
    peak = lambda c: _band_energy(orders, spec, c, max(TOL_ABS, TOL_REL * c)) / total
    out = {}
    for h in range(1, 6):
        out[f"bpfo_h{h}"] = peak(g.bpfo * h)
        out[f"bpfi_h{h}"] = peak(g.bpfi * h)
    out["bpfi_sb_m1"], out["bpfi_sb_p1"] = peak(g.bpfi - 1.0), peak(g.bpfi + 1.0)
    b = bsf(g)
    for h in range(1, 4):
        out[f"bsf_h{h}"] = peak(b * h)
    out["ftf_h1"], out["ftf_h2"] = peak(g.ftf), peak(2 * g.ftf)
    return out


def _spectrum(env: np.ndarray, d: float) -> tuple[np.ndarray, np.ndarray]:
    env = env - env.mean()
    return np.fft.rfftfreq(len(env), d=d), np.abs(np.fft.rfft(env * np.hanning(len(env))))


def v2_features(signal: np.ndarray, fs_hz: float, rpm: float, g: BearingGeometry) -> dict[str, float]:
    """Constant speed: CPW + squared envelope in time, read in orders."""
    env = time_envelope(signal, fs_hz, band_hz=None, whiten=True, squared=True)
    freqs, spec = _spectrum(env, 1.0 / fs_hz)
    return read_families(freqs / (rpm / 60.0), spec, g)


def v2_features_cot(signal: np.ndarray, fs_hz: float, phase_revs: np.ndarray, g: BearingGeometry,
                    samples_per_rev: int = 128) -> dict[str, float]:
    """Time-varying speed: the same envelope, resampled to shaft angle."""
    env = time_envelope(signal, fs_hz, band_hz=None, whiten=True, squared=True)
    env_a, _ = angle_resample(env, phase_revs, samples_per_rev)
    orders, spec = _spectrum(env_a, 1.0 / samples_per_rev)
    return read_families(orders, spec, g)
