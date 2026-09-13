"""Load configs/domain_adaptation.yaml into a small tree of dataclasses.

Same pattern as vibration_ad.config / hydraulic_fusion_ad.config: one YAML
file is the single source of truth for a run, and every stage (data prep,
baseline training, DANN training, evaluation) takes a `Config` object
instead of scattered CLI flags.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class PathsConfig:
    cwru_raw_dir: str
    processed_dir: str
    models_dir: str
    figures_dir: str
    reports_dir: str
    # Local-dev-only fallback: a directory laid out like the real PU dataset
    # (bearing-code subfolders of .mat files), used when /kaggle/input isn't
    # present and --pu-root wasn't passed on the CLI. Never required to
    # exist -- see data/pu_loader.find_pu_root().
    pu_root_local: str


@dataclass
class CommonConfig:
    # Every domain is resampled to this rate before windowing/STFT -- see
    # data/transforms.py and DOMAIN_ADAPTATION_PLAN.md's "Revisions" section
    # for why 12kHz (CWRU's native rate) rather than up sampling everything.
    fs_hz: int
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
class CWRUConfig:
    sensor: str
    train_rpms: list[int]
    val_rpm: int


@dataclass
class PUConfig:
    dataset_slug: str
    native_fs_hz: int
    vibration_channel: str
    healthy_codes: list[str]
    artificial_outer_codes: list[str]
    artificial_inner_codes: list[str]
    real_outer_codes: list[str]
    real_inner_codes: list[str]
    source_conditions: list[str]
    target_condition: str
    # Fraction of each source-condition bearing's runs (of ~20) used for
    # SOURCE TRAIN; the rest -> source val. Contiguous split by run index,
    # not random -- same anti-leakage spirit as CWRU's RPM holdout, though
    # here it's whole 4s runs, not overlapping windows, that get split.
    source_train_fraction: float
    # Fraction of the TARGET condition's healthy runs used for the
    # adaptation step (domain discriminator + healthy-anchor loss); the
    # rest are held out as healthy rows in the final zero-shot test set.
    target_normal_adapt_fraction: float

    @property
    def source_codes(self) -> list[str]:
        return self.healthy_codes + self.artificial_outer_codes + self.artificial_inner_codes

    @property
    def real_codes(self) -> list[str]:
        return self.real_outer_codes + self.real_inner_codes


@dataclass
class ModelConfig:
    base_channels: int
    feature_dim: int
    classifier_hidden: int
    domain_hidden: int
    dropout: float


@dataclass
class BaselineTrainConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    early_stopping_patience: int
    device: str


@dataclass
class DANNTrainConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    early_stopping_patience: int
    domain_loss_weight: float
    anchor_loss_weight: float
    grl_gamma: float  # steepness of the GRL lambda warm-up schedule (see models/grl.py)
    # Learning-rate multiplier for the domain discriminator's own parameter
    # group (1.0 = same as everything else). Lets the discriminator stay a
    # step ahead in the minimax game if it's losing the race -- see
    # train_dann.py's optimizer setup.
    discriminator_lr_multiplier: float
    device: str


@dataclass
class EvaluationConfig:
    tsne_perplexity: float


@dataclass
class Config:
    seed: int
    paths: PathsConfig
    common: CommonConfig
    stft: STFTConfig
    cwru: CWRUConfig
    pu: PUConfig
    model: ModelConfig
    train_baseline: BaselineTrainConfig
    train_dann: DANNTrainConfig
    evaluation: EvaluationConfig
    project_root: Path = field(default_factory=Path.cwd)

    def resolve(self, relative_path: str) -> Path:
        return (self.project_root / relative_path).resolve()


def load_config(config_path: str | Path) -> Config:
    config_path = Path(config_path).resolve()
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    project_root = config_path.parent.parent  # configs/domain_adaptation.yaml -> project root

    return Config(
        seed=raw["seed"],
        paths=PathsConfig(**raw["paths"]),
        common=CommonConfig(**raw["common"]),
        stft=STFTConfig(**raw["stft"]),
        cwru=CWRUConfig(**raw["cwru"]),
        pu=PUConfig(**raw["pu"]),
        model=ModelConfig(**raw["model"]),
        train_baseline=BaselineTrainConfig(**raw["train_baseline"]),
        train_dann=DANNTrainConfig(**raw["train_dann"]),
        evaluation=EvaluationConfig(**raw["evaluation"]),
        project_root=project_root,
    )
