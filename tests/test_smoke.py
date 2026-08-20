"""Fast sanity tests using synthetic signals -- no dataset download required.

These check the plumbing (shapes line up end to end), not model quality.
Run with: .venv/bin/python -m pytest
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from vibration_ad.config import STFTConfig, WindowingConfig
from vibration_ad.data.windows import make_windows, n_windows
from vibration_ad.features.spectrogram import SpectrogramNormalizer, expected_shape, windows_to_log_spectrograms
from vibration_ad.models.autoencoder import ConvAutoencoder, per_sample_mse

WINDOWING = WindowingConfig(segment_length=8384, train_stride=512, eval_stride=2096)
STFT = STFTConfig(nperseg=256, hop_length=64, log_epsilon=1e-10)
FS_HZ = 12000


def test_make_windows_shape_and_count():
    signal = np.arange(20000, dtype=np.float64)
    windows = make_windows(signal, WINDOWING.segment_length, WINDOWING.eval_stride)
    assert windows.shape[1] == WINDOWING.segment_length
    assert windows.shape[0] == n_windows(len(signal), WINDOWING.segment_length, WINDOWING.eval_stride)


def test_make_windows_too_short_returns_empty():
    signal = np.zeros(100)
    windows = make_windows(signal, WINDOWING.segment_length, WINDOWING.eval_stride)
    assert windows.shape == (0, WINDOWING.segment_length)


def test_spectrogram_shape_matches_expected_shape():
    rng = np.random.default_rng(0)
    windows = rng.standard_normal((3, WINDOWING.segment_length))
    spectrograms = windows_to_log_spectrograms(windows, FS_HZ, STFT)
    assert spectrograms.shape == (3,) + expected_shape(WINDOWING, STFT)
    assert expected_shape(WINDOWING, STFT) == (128, 128)


def test_normalizer_roundtrip_is_standardized():
    rng = np.random.default_rng(0)
    spectrograms = rng.normal(loc=5.0, scale=2.0, size=(20, 16, 16)).astype(np.float32)
    normalizer = SpectrogramNormalizer().fit(spectrograms)
    normed = normalizer.transform(spectrograms)
    assert abs(normed.mean()) < 0.1
    assert abs(normed.std() - 1.0) < 0.1


def test_normalizer_raises_before_fit():
    with pytest.raises(RuntimeError):
        SpectrogramNormalizer().transform(np.zeros((1, 4, 4), dtype=np.float32))


def test_autoencoder_forward_preserves_shape():
    model = ConvAutoencoder(input_shape=(128, 128), base_channels=4, latent_dim=8)
    x = torch.randn(2, 1, 128, 128)
    recon = model(x)
    assert recon.shape == x.shape


def test_autoencoder_rejects_non_multiple_of_16_shape():
    with pytest.raises(ValueError):
        ConvAutoencoder(input_shape=(100, 100))


def test_per_sample_mse_shape_and_zero_for_identical_input():
    x = torch.randn(4, 1, 8, 8)
    assert torch.allclose(per_sample_mse(x, x), torch.zeros(4))


def test_tiny_training_step_reduces_loss():
    """One gradient step on a trivial batch should not blow up / NaN."""
    torch.manual_seed(0)
    model = ConvAutoencoder(input_shape=(32, 32), base_channels=4, latent_dim=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    x = torch.randn(4, 1, 32, 32)

    losses = []
    for _ in range(5):
        recon = model(x)
        loss = per_sample_mse(x, recon).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert all(np.isfinite(losses))
    assert losses[-1] < losses[0]
