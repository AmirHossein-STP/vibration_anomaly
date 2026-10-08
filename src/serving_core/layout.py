"""The folder contract. Same names for every service.

    <data_root>/<service>/in/        operator drops files to score
    <data_root>/<service>/train/     operator drops HEALTHY files for train/calibrate
    <data_root>/<service>/source/    (bearing only) labelled public data for `train`
    <data_root>/<service>/config/    service.yaml (machine geometry, thresholds, ...)
    <data_root>/<service>/out/<date>/<job_id>/   results written by the service
    <models_root>/<service>/versions/<version>/  model registry
    <models_root>/<service>/ACTIVE               pointer, written only by promote/rollback
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATA_ROOT = "/data"
DEFAULT_MODELS_ROOT = "/models"


@dataclass(frozen=True)
class Layout:
    service: str
    data_root: Path
    models_root: Path

    @classmethod
    def from_env(cls, service: str, data_root: str | None = None, models_root: str | None = None) -> "Layout":
        return cls(
            service,
            Path(data_root or os.environ.get("SERVING_DATA_ROOT", DEFAULT_DATA_ROOT)),
            Path(models_root or os.environ.get("SERVING_MODELS_ROOT", DEFAULT_MODELS_ROOT)),
        )

    @property
    def base(self) -> Path: return self.data_root / self.service
    @property
    def in_dir(self) -> Path: return self.base / "in"
    @property
    def train_dir(self) -> Path: return self.base / "train"
    @property
    def source_dir(self) -> Path: return self.base / "source"
    @property
    def config_dir(self) -> Path: return self.base / "config"
    @property
    def config_file(self) -> Path: return self.config_dir / "service.yaml"
    @property
    def out_dir(self) -> Path: return self.base / "out"
    @property
    def models_dir(self) -> Path: return self.models_root / self.service

    def ensure(self) -> None:
        for p in (self.in_dir, self.train_dir, self.config_dir, self.out_dir, self.models_dir / "versions"):
            p.mkdir(parents=True, exist_ok=True)
