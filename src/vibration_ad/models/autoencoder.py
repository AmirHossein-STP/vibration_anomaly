"""Convolutional autoencoder: the "brain" of the detector.

Trained ONLY on spectrograms of healthy vibration signal (see
data/dataset.py's train split). The idea: if the network only ever sees a
healthy motor's spectral signature, its bottleneck learns a compact code
for "what healthy looks like" -- and reconstruction quality degrades on
anything that doesn't fit that pattern, i.e. a fault. The reconstruction
error (see per_sample_mse below) is the anomaly score; see evaluate.py for
how the detection threshold is chosen and models_notes in
notes/03_model_notes.md for the bottleneck-size tradeoff.

Architecture: 4 stride-2 conv layers down to an 8x8 feature map, a dense
bottleneck, then the mirror image in ConvTranspose2d layers back up. This
assumes a square, power-of-two input (128x128 by default -- see
configs/default.yaml and features/spectrogram.expected_shape) so each
conv/deconv layer exactly halves/doubles spatial size with no rounding.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    def __init__(self, input_shape: tuple[int, int], base_channels: int = 16, latent_dim: int = 128):
        super().__init__()
        freq_bins, time_frames = input_shape
        if freq_bins % 16 != 0 or time_frames % 16 != 0:
            raise ValueError(
                f"input_shape {input_shape} must be divisible by 16 (four stride-2 layers); "
                "adjust configs/default.yaml's stft/windowing params."
            )
        c = base_channels

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, c, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c * 2, c * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(c * 4, c * 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c * 8),
            nn.ReLU(inplace=True),
        )
        self.reduced_shape = (c * 8, freq_bins // 16, time_frames // 16)
        flat_dim = c * 8 * (freq_bins // 16) * (time_frames // 16)

        self.encoder_fc = nn.Linear(flat_dim, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, flat_dim)

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(c * 8, c * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c * 4, c * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c * 2, c, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            # No final activation: inputs are z-scored (roughly N(0,1)), so
            # the reconstruction target is any real number, not a [0,1] pixel.
            nn.ConvTranspose2d(c, 1, kernel_size=4, stride=2, padding=1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder_conv(x)
        z = z.flatten(1)
        return self.encoder_fc(z)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.decoder_fc(z)
        x = x.view(-1, *self.reduced_shape)
        return self.decoder_conv(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def per_sample_mse(x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    """Reconstruction error per example: mean squared error over (C,F,T),
    i.e. one scalar anomaly score per input spectrogram. Shape (N,) -> (N,).
    """
    return ((x - recon) ** 2).mean(dim=(1, 2, 3))


def select_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
