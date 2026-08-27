"""Load configs/hydraulic.yaml into a small tree of dataclasses.

Mirrors vibration_ad/config.py's role and style: one Config object per run,
built from one YAML file, so every stage (data prep, training, evaluation)
is reproducible from that single file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class PathsConfig:
    raw_dir: str
    processed_dir: str
    models_dir: str
    figures_dir: str
    reports_dir: str


@dataclass
class SensorGroupConfig:
    rate_hz: int
    samples_per_cycle: int
    sensors: list[str]


@dataclass
class DataConfig:
    n_cycles_expected: int
    profile_columns: list[str]
    healthy_values: dict[str, int]
    valid_values: dict[str, list[int]]
    sensor_groups: dict[str, SensorGroupConfig]  # keys: slow, medium, fast
    # Which of healthy_values' four component columns this run's detector
    # targets -- "healthy" and the train/val/test split are defined by THIS
    # column alone, the other three are ignored. See
    # notes/09_hydraulic_notes.md's "Per-component redesign" section for why
    # single-detector-requires-all-four-healthy was abandoned (only 20 of
    # 2205 cycles qualified) in favor of four separate per-component runs.
    target_component: str

    def __post_init__(self) -> None:
        if self.target_component not in self.healthy_values:
            raise ValueError(
                f"data.target_component={self.target_component!r} must be one of "
                f"{list(self.healthy_values)} (data.healthy_values' keys)."
            )

    @property
    def all_sensors(self) -> list[str]:
        """Every sensor filename stem, in a fixed group-then-within-group order."""
        out = []
        for group in self.sensor_groups.values():
            out.extend(group.sensors)
        return out


@dataclass
class SplitConfig:
    train_fraction: float
    val_fraction: float
    # Contiguous cycles per block before block-shuffling (see
    # notes/09_hydraulic_notes.md's "Block-shuffled split" section) --
    # small enough to keep adjacent-cycle autocorrelation mostly within one
    # block, large enough that shuffling isn't effectively a per-cycle
    # random split.
    block_size: int


@dataclass
class ModelConfig:
    slow_hidden_dim: int
    slow_branch_latent: int
    medium_base_channels: int
    medium_branch_latent: int
    fast_base_channels: int
    fast_branch_latent: int
    fused_latent_dim: int
    branch_weights: dict[str, float]


@dataclass
class TrainConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    early_stopping_patience: int
    device: str


@dataclass
class EvaluationConfig:
    threshold_percentile: float


@dataclass
class Config:
    seed: int
    paths: PathsConfig
    data: DataConfig
    split: SplitConfig
    model: ModelConfig
    train: TrainConfig
    evaluation: EvaluationConfig
    # Directory the config file lives in; relative paths in `paths` are
    # resolved against the project root (parent of configs/), not against cwd.
    project_root: Path = field(default_factory=Path.cwd)

    def resolve(self, relative_path: str) -> Path:
        return (self.project_root / relative_path).resolve()


def load_config(config_path: str | Path) -> Config:
    config_path = Path(config_path).resolve()
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    project_root = config_path.parent.parent  # configs/hydraulic.yaml -> project root

    data_raw = dict(raw["data"])
    data_raw["sensor_groups"] = {
        name: SensorGroupConfig(**group) for name, group in data_raw["sensor_groups"].items()
    }

    return Config(
        seed=raw["seed"],
        paths=PathsConfig(**raw["paths"]),
        data=DataConfig(**data_raw),
        split=SplitConfig(**raw["split"]),
        model=ModelConfig(**raw["model"]),
        train=TrainConfig(**raw["train"]),
        evaluation=EvaluationConfig(**raw["evaluation"]),
        project_root=project_root,
    )
