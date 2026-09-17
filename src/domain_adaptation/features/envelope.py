"""Envelope-spectrum features in the ORDER domain -- a representation that
can in principle cross rigs, which log-spectrograms provably cannot.

Why this exists
---------------
Every cross-dataset experiment in this project failed the same way: a model
trained on one rig calls every window on the other rig faulty (see
notes/12_reproduction_investigation.md -- 11/11 strict runs collapsed, arm B
failed identically with a 4x richer source, and per-window normalization,
which demonstrably removes the scale mismatch, did not help).

The reason is physical, not numerical. A log-spectrogram is dominated by the
STRUCTURAL RESONANCES of the machine -- housing, shaft, mounting -- which sit
at fixed frequencies in Hz and are a property of the rig, not the bearing.
CWRU and Paderborn are different machines, so their resonances differ, and a
network trained on one has learned, in large part, what that one machine
sounds like.

A bearing fault does not produce a tone. Each time a rolling element strikes
the defect it excites those resonances, so the fault appears as AMPLITUDE
MODULATION of a resonance carrier. The modulation rate is what identifies
the fault, and it depends only on bearing geometry and shaft speed:

    BPFO = (n/2) * (1 - d/D * cos(alpha)) * shaft_rate    # outer race
    BPFI = (n/2) * (1 + d/D * cos(alpha)) * shaft_rate    # inner race

Expressed in ORDERS (multiples of shaft rate) these are pure numbers, so
"energy at BPFO" means the same physical thing on any machine. That is the
quantity this module extracts, and it is why it is worth trying after the
spectrogram approach has been ruled out.

The transform is: band-pass around the resonance band -> Hilbert envelope ->
spectrum of the envelope -> read off energy at the fault orders. The carrier
frequency (rig-specific) is discarded by the envelope step; only the
modulation (bearing-specific) survives. That discarding is the entire point.

Note the fault ORDERS differ between the two datasets because the bearings
differ -- CWRU's 6205 gives BPFO 3.585, Paderborn's 6203 gives 3.053. The
features below are therefore read at EACH dataset's own fault orders, which
is what makes them comparable: the feature is "how much energy sits at this
bearing's outer-race frequency", not "how much energy sits at 3.585x".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt


@dataclass(frozen=True)
class BearingGeometry:
    """Enough to compute fault orders. Orders are dimensionless multiples of
    shaft rate, so they carry across rigs and speeds without conversion."""

    name: str
    n_balls: int
    ball_diameter: float       # d, any unit
    pitch_diameter: float      # D, same unit as d
    contact_angle_deg: float = 0.0

    @property
    def _ratio(self) -> float:
        return (self.ball_diameter / self.pitch_diameter) * np.cos(np.radians(self.contact_angle_deg))

    @property
    def bpfo(self) -> float:
        """Ball-pass frequency, outer race, in orders of shaft rate."""
        return 0.5 * self.n_balls * (1.0 - self._ratio)

    @property
    def bpfi(self) -> float:
        """Ball-pass frequency, inner race, in orders of shaft rate."""
        return 0.5 * self.n_balls * (1.0 + self._ratio)

    @property
    def ftf(self) -> float:
        """Cage (fundamental train) frequency, in orders. Inner-race faults
        are modulated by shaft rate; outer-race ones sit near the cage rate,
        which is why sidebands help tell them apart."""
        return 0.5 * (1.0 - self._ratio)


# CWRU drive end: SKF 6205-2RS JEM. Geometry from the CWRU bearing data
# center's published table; the derived orders reproduce their quoted
# multipliers exactly (BPFO 3.5848, BPFI 5.4152), which is the check that
# these numbers are entered correctly.
CWRU_6205 = BearingGeometry("SKF 6205-2RS JEM", n_balls=9, ball_diameter=0.3126, pitch_diameter=1.537)

# Paderborn test bearings: 6203. Geometry per Lessmeier et al. 2016; the
# derived orders are BPFO 3.053 and BPFI 4.947, matching the paper.
PU_6203 = BearingGeometry("6203", n_balls=8, ball_diameter=6.75, pitch_diameter=28.5)


def envelope_spectrum(
    signal: np.ndarray,
    fs_hz: float,
    band_hz: tuple[float, float] = (1500.0, 5500.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Band-pass -> Hilbert envelope -> magnitude spectrum of the envelope.

    `band_hz` selects the resonance carrier. It is deliberately a FIXED
    wide band rather than one chosen per window: picking the band by which
    choice best separates the classes would leak label information, and
    picking it per window would make features incomparable between windows.
    Both datasets are resampled to 12 kHz here, so Nyquist is 6 kHz and
    1.5-5.5 kHz covers the usable high-frequency region where bearing
    resonances live and shaft/imbalance harmonics do not.

    Returns (freqs_hz, magnitude) for the envelope, DC removed.
    """
    nyq = fs_hz / 2.0
    lo, hi = band_hz[0] / nyq, min(band_hz[1] / nyq, 0.99)
    if not 0 < lo < hi < 1:
        raise ValueError(f"band {band_hz} invalid for fs={fs_hz}")
    sos = butter(4, [lo, hi], btype="bandpass", output="sos")
    banded = sosfiltfilt(sos, signal)

    env = np.abs(hilbert(banded))
    env = env - env.mean()          # the DC term dwarfs every fault peak
    window = np.hanning(len(env))
    spec = np.abs(np.fft.rfft(env * window))
    freqs = np.fft.rfftfreq(len(env), d=1.0 / fs_hz)
    return freqs, spec


def _band_energy(orders: np.ndarray, spec: np.ndarray, centre: float, tol: float) -> float:
    """Peak energy within +-tol orders of `centre`.

    A peak rather than a sum: shaft speed is only nominally known (CWRU
    encodes it in the filename, Paderborn quotes a setpoint), so the true
    fault order drifts by a percent or two and a narrow sum would miss it
    entirely. `tol` is what absorbs that.
    """
    m = np.abs(orders - centre) <= tol
    return float(spec[m].max()) if m.any() else 0.0


def envelope_order_features(
    signal: np.ndarray,
    fs_hz: float,
    rpm: float,
    geometry: BearingGeometry,
    band_hz: tuple[float, float] = (1500.0, 5500.0),
    n_harmonics: int = 3,
    tol_orders: float = 0.08,
) -> dict[str, float]:
    """A small, physically-named feature vector comparable ACROSS rigs.

    Every feature is a RATIO against the window's own broadband envelope
    energy, so overall gain -- sensor sensitivity, amplifier settings, how
    hard the rig is driven -- divides out. That matters more here than
    usual: the two datasets were recorded years apart on different
    equipment, and absolute amplitude carries no shared meaning.
    """
    shaft_hz = rpm / 60.0
    freqs, spec = envelope_spectrum(signal, fs_hz, band_hz)
    orders = freqs / shaft_hz

    # Reference band: everything the bearing could plausibly produce, used
    # to normalize. Starts above 0.5 to exclude the shaft-rate peak, which
    # is imbalance/misalignment and present on healthy machines too.
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

    # Inner-race defects rotate through the load zone, so their envelope
    # peaks carry shaft-rate sidebands; outer-race defects are stationary
    # and do not. This is the main geometric cue separating the two.
    feats["bpfi_sidebands"] = sum(
        _band_energy(orders, spec, geometry.bpfi + s, tol_orders) / total for s in (-1.0, 1.0)
    )
    feats["ftf"] = _band_energy(orders, spec, geometry.ftf, tol_orders) / total

    # Shape statistics of the envelope spectrum: impulsiveness rises with
    # localized damage regardless of which race it is on, so these help the
    # healthy-vs-faulty call even when the order peaks are ambiguous.
    band = spec[ref]
    feats["env_kurtosis"] = float(((band - band.mean()) ** 4).mean() / (band.std() ** 4 + 1e-12))
    feats["env_crest"] = float(band.max() / (np.sqrt((band ** 2).mean()) + 1e-12))
    feats["env_flatness"] = float(
        np.exp(np.log(band + 1e-12).mean()) / (band.mean() + 1e-12)
    )
    return feats


FEATURE_NAMES = [
    "bpfo_h1", "bpfo_h2", "bpfo_h3", "bpfo_sum",
    "bpfi_h1", "bpfi_h2", "bpfi_h3", "bpfi_sum",
    "bpfi_sidebands", "ftf",
    "env_kurtosis", "env_crest", "env_flatness",
]
