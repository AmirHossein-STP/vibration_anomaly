"""L0: ISO 20816-3 (formerly ISO 10816-3) vibration severity zones (MASTER_PLAN §3 L0, WP19).

Broadband velocity RMS, measured on bearing housings, against fixed zone
boundaries by machine group and foundation:
    A  newly commissioned
    B  unrestricted long-term operation
    C  restricted operation (plan maintenance)
    D  damage may occur
No training, no baseline. A coarse, standard first layer: it answers "is this
machine vibrating more than machines of its class should", not "which bearing
part is failing".

Boundaries in mm/s RMS (ISO 10816-3:2009 Table A.1/A.2, unchanged in 20816-3):
    group 1 (large, 300 kW-50 MW): rigid 2.3 / 4.5 / 7.1   flexible 3.5 / 7.1 / 11.0
    group 2 (medium, 15-300 kW):   rigid 1.4 / 2.8 / 4.5   flexible 2.3 / 4.5 / 7.1
Band: 10-1000 Hz for machines above 600 rpm, 2-1000 Hz from 120 to 600 rpm.
"""
from __future__ import annotations

import numpy as np

G = 9.80665
BOUNDARIES = {(1, "rigid"): (2.3, 4.5, 7.1), (1, "flexible"): (3.5, 7.1, 11.0),
              (2, "rigid"): (1.4, 2.8, 4.5), (2, "flexible"): (2.3, 4.5, 7.1)}
UNIT_TO_MS2 = {"g": G, "m/s2": 1.0, "mm/s2": 1e-3}


def velocity_rms_mm_s(acc: np.ndarray, fs_hz: float, unit: str = "g", rpm: float | None = None) -> float:
    """Acceleration -> velocity RMS in mm/s over the ISO band, by integration in
    the frequency domain (divide by j*2*pi*f inside the band, zero outside)."""
    if unit not in UNIT_TO_MS2:
        raise ValueError(f"acceleration unit must be one of {sorted(UNIT_TO_MS2)}")
    lo = 2.0 if (rpm is not None and rpm <= 600) else 10.0
    hi = min(1000.0, fs_hz / 2 * 0.95)
    a = (np.asarray(acc, dtype=np.float64) - np.mean(acc)) * UNIT_TO_MS2[unit]
    A = np.fft.rfft(a * np.hanning(len(a)))
    f = np.fft.rfftfreq(len(a), 1.0 / fs_hz)
    band = (f >= lo) & (f <= hi)
    V = np.zeros_like(A)
    V[band] = A[band] / (2j * np.pi * f[band])
    v = np.fft.irfft(V, n=len(a))
    # Hann window loses energy: rescale by its mean square
    return float(np.sqrt(np.mean(v ** 2) / np.mean(np.hanning(len(a)) ** 2)) * 1000.0)


def zone(v_mm_s: float, group: int, foundation: str = "rigid") -> str:
    ab, bc, cd = BOUNDARIES[(int(group), foundation)]
    return "A" if v_mm_s < ab else "B" if v_mm_s < bc else "C" if v_mm_s < cd else "D"
