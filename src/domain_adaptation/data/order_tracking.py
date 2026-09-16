"""Angular resampling: turn a signal sampled uniformly in TIME into one
sampled uniformly in SHAFT ANGLE.

This is the fix for the one domain gap five kernel runs never addressed.
Bearing fault frequencies (BPFO, BPFI, BSF) are all proportional to shaft
rate, so dropping 1500rpm -> 900rpm moves every fault signature to 60% of
its original frequency. On a linear-frequency spectrogram that is a
*multiplicative rescaling of the frequency axis* -- not a translation. A
CNN has real translation tolerance and essentially none for scaling, which
is why a classifier trained at one speed transferred so poorly to another
(see notes/10_domain_adaptation_notes.md section 2).

Resample against angle instead and the axis becomes **orders** (events per
shaft revolution) rather than Hz. A fault at 5.4 orders sits in the same
bin at 900rpm as at 1500rpm, because the speed has been divided out of the
representation entirely. That is the standard trick in rotating-machinery
diagnostics, and it is the reason to expect this to work where
adversarial feature alignment did not: it removes the shift by
construction rather than asking a network to learn around it.

Constant-speed assumption: both datasets run at nominally fixed rpm per
recording (PU's N09/N15 codes, CWRU's per-file rpm), so a single scalar
rate per recording is enough and angular resampling collapses to a
rational resample. True order tracking under varying speed would integrate
an instantaneous speed signal to get shaft phase, then interpolate onto
uniform phase -- PU ships a `speed` channel that would support it. Not
needed while every recording is constant-speed; see `angular_resample`'s
note if that changes.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly

# Samples per shaft revolution after angular resampling. This sets the
# Nyquist ORDER (= samples_per_rev / 2), i.e. the highest fault order
# representable. Bearing tones of interest (BPFO/BPFI are typically ~3-12
# orders) plus a healthy stack of harmonics fit comfortably under 128.
DEFAULT_SAMPLES_PER_REV = 256


def angular_resample(
    signal: np.ndarray, fs_hz: float, rpm: float, samples_per_rev: int = DEFAULT_SAMPLES_PER_REV
) -> np.ndarray:
    """Resample `signal` from uniform-in-time to uniform-in-angle.

    Returns a signal whose sample index advances by exactly
    1/samples_per_rev of a revolution per step, so its spectrum is indexed
    in orders rather than Hz.

    For constant speed this is just a rational resample: one revolution
    occupies `fs_hz * 60 / rpm` samples in the time domain and must occupy
    `samples_per_rev` after, giving the ratio below. (Varying speed would
    need phase integration + interpolation instead -- see module docstring.)
    """
    if rpm <= 0:
        raise ValueError(f"rpm must be positive, got {rpm}")
    time_samples_per_rev = fs_hz * 60.0 / rpm
    ratio = Fraction(samples_per_rev / time_samples_per_rev).limit_denominator(1000)
    if ratio.numerator == ratio.denominator:
        return signal
    return resample_poly(signal, ratio.numerator, ratio.denominator).astype(signal.dtype)


def order_axis(nperseg: int, samples_per_rev: int = DEFAULT_SAMPLES_PER_REV) -> np.ndarray:
    """Order value of each STFT bin produced from an angular-resampled
    signal -- the analogue of a frequency axis, in events per revolution.

    An STFT bin k of nperseg sits at k/nperseg cycles per SAMPLE; each
    sample is 1/samples_per_rev of a revolution, so that is
    k/nperseg * samples_per_rev cycles per REVOLUTION.
    """
    return np.arange(nperseg // 2 + 1) / nperseg * samples_per_rev


def revolutions_per_window(segment_length: int, samples_per_rev: int = DEFAULT_SAMPLES_PER_REV) -> float:
    """How many shaft revolutions one window spans. Unlike a time-domain
    window this is speed-independent by construction -- the same window
    covers the same number of revolutions at any rpm, which is exactly the
    invariance the representation is bought for.
    """
    return segment_length / samples_per_rev
