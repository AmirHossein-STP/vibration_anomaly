"""Load the YAML config into a small tree of dataclasses.

Everything downstream (data prep, training, evaluation, inference) takes
one of these `Config` objects instead of scattered CLI flags, so a run is
fully described by a single YAML file (see configs/default.yaml).
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
class DataConfig:
    sensor: str
    fs_hz: int
    train_rpms: list[int]
    holdout_rpm: int
    fault_types: list[str]
    fault_rpms: list[int]


@dataclass
class WindowingConfig:
    segment_length: int
    train_stride: int
    eval_stride: int


@dataclass
class STFTConfig:
    nperseg: int
    hop_length: int
    log_epsilon: float

    @property
    def noverlap(self) -> int:
        return self.nperseg - self.hop_length


@dataclass
class ModelConfig:
    latent_dim: int
    base_channels: int


@dataclass
class TrainConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    early_stopping_patience: int
    val_fraction_of_holdout: float
    device: str


@dataclass
class EvaluationConfig:
    threshold_percentile: float


@dataclass
class Config:
    seed: int
    paths: PathsConfig
    data: DataConfig
    windowing: WindowingConfig
    stft: STFTConfig
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

    project_root = config_path.parent.parent  # configs/default.yaml -> project root

    return Config(
        seed=raw["seed"],
        paths=PathsConfig(**raw["paths"]),
        data=DataConfig(**raw["data"]),
        windowing=WindowingConfig(**raw["windowing"]),
        stft=STFTConfig(**raw["stft"]),
        model=ModelConfig(**raw["model"]),
        train=TrainConfig(**raw["train"]),
        evaluation=EvaluationConfig(**raw["evaluation"]),
        project_root=project_root,
    )
