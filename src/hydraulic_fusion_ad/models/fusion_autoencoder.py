"""Multi-branch sensor-fusion autoencoder: the "brain" of the detector.

Trained ONLY on cycles where all four monitored components are healthy (see
data/dataset.py's split). Three independent encoder branches -- one per
native sampling rate (slow/1Hz, medium/10Hz, fast/100Hz) -- each compress
their own sensors at their own native resolution (no resampling), are
concatenated into one shared latent vector (the actual bottleneck), and
three independent decoder branches mirror them back out. Reconstruction
error (per_branch_mse / combined_mse below) is the anomaly score. See
notes/09_hydraulic_notes.md section 3 for the full architecture diagram and
reasoning, and models/autoencoder.py in vibration_ad/ for the 2D-conv
sibling this 1D design mirrors.

    slow   (N, n_slow, 60)   -> MLP encoder    -> branch_latent_slow
    medium (N, n_medium,600) -> Conv1d encoder -> branch_latent_medium
    fast   (N, n_fast, 6000) -> Conv1d encoder -> branch_latent_fast
                                            concat -> Linear -> fused_latent
                                            Linear -> split -> mirror decoders
"""
from __future__ import annotations

import torch
import torch.nn as nn

from hydraulic_fusion_ad.config import Config


def conv_layer_plan(samples_per_cycle: int, max_layers: int = 4, min_final_length: int = 8) -> tuple[int, int]:
    """How many stride-2 Conv1d(kernel=4, padding=1) layers to stack, and the
    resulting sequence length, for a given branch's samples-per-cycle.

    Each such layer exactly halves the sequence length (see the module
    docstring / notes/09_hydraulic_notes.md section 3), so we keep halving
    while the length stays exactly divisible by 2 and doesn't drop below
    `min_final_length`, up to `max_layers`. Returns (n_layers, final_length).
    """
    n_layers, length = 0, samples_per_cycle
    while n_layers < max_layers and length % 2 == 0 and length // 2 >= min_final_length:
        length //= 2
        n_layers += 1
    return n_layers, length


class ConvBranchEncoder(nn.Module):
    """Conv1d(kernel=4, stride=2, padding=1) stack: exactly halves sequence
    length each layer (input length must stay even through every layer --
    see conv_layer_plan), doubling channel width each layer from base_channels.
    """

    def __init__(self, n_sensors: int, samples_per_cycle: int, base_channels: int, branch_latent: int):
        super().__init__()
        n_layers, final_length = conv_layer_plan(samples_per_cycle)
        if n_layers == 0:
            raise ValueError(
                f"samples_per_cycle={samples_per_cycle} is too short/odd for any "
                "stride-2 Conv1d layer -- check configs/hydraulic.yaml's sensor_groups."
            )
        layers = []
        in_ch = n_sensors
        for i in range(n_layers):
            out_ch = base_channels * (2**i)
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.reduced_shape = (in_ch, final_length)
        self.fc = nn.Linear(in_ch * final_length, branch_latent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv(x).flatten(1)
        return self.fc(z)


class ConvBranchDecoder(nn.Module):
    """Mirror image of ConvBranchEncoder: ConvTranspose1d(kernel=4, stride=2,
    padding=1) exactly doubles sequence length each layer, regardless of
    whether the current length is odd or even (see notes/09_hydraulic_notes.md
    section 3), so this exactly inverts the encoder's shape with no
    cropping/padding needed.
    """

    def __init__(self, n_sensors: int, samples_per_cycle: int, base_channels: int, branch_latent: int):
        super().__init__()
        n_layers, final_length = conv_layer_plan(samples_per_cycle)
        channel_widths = [base_channels * (2**i) for i in range(n_layers)]  # encoder's out_ch per layer, in order
        self.reduced_shape = (channel_widths[-1], final_length)
        self.fc = nn.Linear(branch_latent, channel_widths[-1] * final_length)

        layers = []
        # Walk the encoder's channel widths in reverse to mirror it exactly.
        reversed_widths = list(reversed(channel_widths))  # e.g. [64, 32, 16] for base_channels=16, 3 layers
        for i in range(n_layers):
            in_ch = reversed_widths[i]
            out_ch = reversed_widths[i + 1] if i + 1 < n_layers else n_sensors
            layers.append(nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1))
            if i + 1 < n_layers:
                layers += [nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True)]
            # No final activation on the last layer: inputs are z-scored, so
            # the reconstruction target is any real number, not a bounded value.
        self.deconv = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(-1, *self.reduced_shape)
        return self.deconv(x)


class MLPBranchEncoder(nn.Module):
    """Plain MLP for the slow (1Hz) branch -- at 60 samples/60s these
    sensors are near-monotonic thermal/efficiency drift, not oscillatory
    signal, so there's little for a convolution to exploit that a dense
    layer over the flattened input can't also see (see
    notes/09_hydraulic_notes.md section 3).
    """

    def __init__(self, n_sensors: int, samples_per_cycle: int, hidden_dim: int, branch_latent: int):
        super().__init__()
        self.flat_dim = n_sensors * samples_per_cycle
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, branch_latent),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPBranchDecoder(nn.Module):
    def __init__(self, n_sensors: int, samples_per_cycle: int, hidden_dim: int, branch_latent: int):
        super().__init__()
        self.out_shape = (n_sensors, samples_per_cycle)
        self.net = nn.Sequential(
            nn.Linear(branch_latent, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_sensors * samples_per_cycle),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.net(z)
        return x.view(-1, *self.out_shape)


class FusionAutoencoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        groups = config.data.sensor_groups
        m = config.model

        self.slow_encoder = MLPBranchEncoder(
            len(groups["slow"].sensors), groups["slow"].samples_per_cycle, m.slow_hidden_dim, m.slow_branch_latent
        )
        self.slow_decoder = MLPBranchDecoder(
            len(groups["slow"].sensors), groups["slow"].samples_per_cycle, m.slow_hidden_dim, m.slow_branch_latent
        )
        self.medium_encoder = ConvBranchEncoder(
            len(groups["medium"].sensors), groups["medium"].samples_per_cycle, m.medium_base_channels, m.medium_branch_latent
        )
        self.medium_decoder = ConvBranchDecoder(
            len(groups["medium"].sensors), groups["medium"].samples_per_cycle, m.medium_base_channels, m.medium_branch_latent
        )
        self.fast_encoder = ConvBranchEncoder(
            len(groups["fast"].sensors), groups["fast"].samples_per_cycle, m.fast_base_channels, m.fast_branch_latent
        )
        self.fast_decoder = ConvBranchDecoder(
            len(groups["fast"].sensors), groups["fast"].samples_per_cycle, m.fast_base_channels, m.fast_branch_latent
        )

        concat_dim = m.slow_branch_latent + m.medium_branch_latent + m.fast_branch_latent
        self.fuse = nn.Linear(concat_dim, m.fused_latent_dim)
        self.unfuse = nn.Linear(m.fused_latent_dim, concat_dim)
        self._branch_latents = (m.slow_branch_latent, m.medium_branch_latent, m.fast_branch_latent)

    def encode(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        z_slow = self.slow_encoder(x["slow"])
        z_medium = self.medium_encoder(x["medium"])
        z_fast = self.fast_encoder(x["fast"])
        return self.fuse(torch.cat([z_slow, z_medium, z_fast], dim=1))

    def decode(self, fused: torch.Tensor) -> dict[str, torch.Tensor]:
        concat = self.unfuse(fused)
        n_slow, n_medium, n_fast = self._branch_latents
        z_slow, z_medium, z_fast = torch.split(concat, [n_slow, n_medium, n_fast], dim=1)
        return {
            "slow": self.slow_decoder(z_slow),
            "medium": self.medium_decoder(z_medium),
            "fast": self.fast_decoder(z_fast),
        }

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.decode(self.encode(x))


def per_branch_mse(x: dict[str, torch.Tensor], recon: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Per-example MSE for each branch separately, shape (N,) each -- each
    branch's error is its own mean over (channels, samples), so a branch
    isn't penalized/rewarded just for having more numbers than another.
    """
    return {group: ((x[group] - recon[group]) ** 2).mean(dim=(1, 2)) for group in x}


def combined_mse(
    branch_errors: dict[str, torch.Tensor], branch_weights: dict[str, float]
) -> torch.Tensor:
    """Weighted mean of per-branch errors (equal weight by default) -- see
    notes/09_hydraulic_notes.md section 3, "Loss", for why branches are
    combined this way instead of pooling raw elementwise error.
    """
    total_weight = sum(branch_weights.values())
    combined = sum(branch_errors[group] * branch_weights[group] for group in branch_errors)
    return combined / total_weight


def select_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
