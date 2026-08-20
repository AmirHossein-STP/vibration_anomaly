"""Cut a long 1D vibration signal into fixed-length overlapping windows.

This is the step that turns "one long recording per bearing condition"
into "many independent examples" for the autoencoder to train/evaluate on.
"""
from __future__ import annotations

import numpy as np


def make_windows(signal: np.ndarray, segment_length: int, stride: int) -> np.ndarray:
    """Slice `signal` into overlapping windows of length `segment_length`.

    Returns an array of shape (n_windows, segment_length). Uses a strided
    view (no copy) via numpy's sliding_window_view for speed, then takes
    every `stride`-th window.

    Any samples that don't fill a final complete window are dropped.
    """
    if len(signal) < segment_length:
        return np.empty((0, segment_length), dtype=signal.dtype)

    all_windows = np.lib.stride_tricks.sliding_window_view(signal, segment_length)
    return np.ascontiguousarray(all_windows[::stride])


def n_windows(signal_length: int, segment_length: int, stride: int) -> int:
    """Number of windows make_windows() would produce, without building them."""
    if signal_length < segment_length:
        return 0
    return (signal_length - segment_length) // stride + 1
