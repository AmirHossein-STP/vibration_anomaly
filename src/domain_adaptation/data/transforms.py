"""Cross-domain sample-rate alignment.

CWRU's Drive-End signal is already 12kHz; Paderborn's vibration_1 channel
is 64kHz. A CNN trained on log-STFT-magnitude images needs pixels that
mean the same physical frequency range in every domain, so every signal is
resampled to one common rate (`config.common.fs_hz`, default 12000) BEFORE
windowing/STFT -- see DOMAIN_ADAPTATION_PLAN.md's "Revisions" section for
why 12kHz (CWRU's native rate) was chosen over up sampling everything to
48kHz as the original plan proposed.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
from scipy.signal import resample_poly


def resample_to_rate(signal: np.ndarray, native_fs_hz: int, target_fs_hz: int) -> np.ndarray:
    """Polyphase resample (proper anti-aliasing filter, unlike a naive
    FFT-bin resample) from native_fs_hz to target_fs_hz.

    No-ops (returns `signal` unchanged, not even copied) when the rates
    already match -- e.g. CWRU at 12kHz when target_fs_hz=12000 -- so data
    that doesn't need resampling never picks up filter artifacts from one.
    """
    if native_fs_hz == target_fs_hz:
        return signal
    ratio = Fraction(target_fs_hz, native_fs_hz).limit_denominator(1000)
    return resample_poly(signal, up=ratio.numerator, down=ratio.denominator).astype(signal.dtype)
