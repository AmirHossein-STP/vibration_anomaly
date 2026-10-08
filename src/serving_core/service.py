"""What a service must provide. The shared CLI (cli.py) does everything else."""
from __future__ import annotations

import argparse
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from serving_core.layout import Layout


class NotSupported(RuntimeError):
    pass


class ConfigError(ValueError):
    pass


@dataclass
class Context:
    layout: Layout
    config: dict
    log: logging.Logger
    args: argparse.Namespace
    version_dir: Path | None = None       # set for train/calibrate
    active_version: str | None = None
    active_dir: Path | None = None
    job_id: str = ""


@dataclass
class Outcome:
    """What train/calibrate hands back to the shared code for the registry."""
    kind: str                                  # "train" | "calibrate"
    metrics: dict
    summary: list[str]
    inputs: list[Path]                         # files that fed the model (hashed into data_hash)
    warnings: list[str] = field(default_factory=list)
    parent: str | None = None
    extra_meta: dict = field(default_factory=dict)


class Service(ABC):
    name: str = ""
    description: str = ""
    # Columns every result row has; services add their own after these.
    base_columns = ["file", "status", "verdict", "score", "flagged_fraction", "n_windows", "message"]
    extra_columns: list[str] = []

    @property
    def columns(self) -> list[str]:
        return self.base_columns + self.extra_columns + ["model_version", "job_id"]

    # ---- configuration ---------------------------------------------------
    @abstractmethod
    def default_config(self) -> dict: ...

    def example_config_text(self) -> str:
        import yaml

        return yaml.safe_dump(self.default_config(), sort_keys=False, allow_unicode=True)

    def validate_config(self, cfg: dict) -> list[str]:
        """Problems that make the config unusable (empty list = fine)."""
        return []

    # ---- CLI additions ---------------------------------------------------
    def add_train_args(self, p: argparse.ArgumentParser) -> None: ...

    def add_calibrate_args(self, p: argparse.ArgumentParser) -> None: ...

    # ---- life cycle ------------------------------------------------------
    @abstractmethod
    def train(self, ctx: Context) -> Outcome: ...

    def calibrate(self, ctx: Context) -> Outcome:
        raise NotSupported(f"{self.name}: `calibrate` is not a separate step for this service; use `train`.")

    @abstractmethod
    def load_model(self, version_dir: Path, meta: dict) -> Any: ...

    def list_inputs(self, in_dir: Path) -> list[Path]:
        from serving_core.signals import list_signal_files

        return list_signal_files(in_dir)

    @abstractmethod
    def infer_one(self, model: Any, path: Path, ctx: Context) -> list[dict]:
        """Return one or more result rows for this input. Raise InputError for
        a bad file (it is reported and the batch continues)."""

    def promote_check(self, version_dir: Path, meta: dict) -> list[str]:
        """Service-specific reasons a version must not become active."""
        return []

    def doctor(self, layout: Layout, cfg: dict) -> list[tuple[str, str, str]]:
        """Extra checks: list of (status 'ok'|'warn'|'fail', name, detail)."""
        return []
