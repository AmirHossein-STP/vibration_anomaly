"""Phase 2: supervised source-only baseline -- NO domain adaptation.

Trains FeatureClassifier on SOURCE_TRAIN (CWRU + PU_Artificial, fully
labeled), early-stops on SOURCE_VAL. Never sees TARGET_ADAPT or
TARGET_TEST during training. evaluate.py then runs this exact model on
TARGET_TEST (the held-out condition's real faults) to measure the
domain-shift performance drop BEFORE adaptation -- the number Phase 3
(train_dann.py) is trying to improve on.

This script is also the canonical place the per-frequency-bin normalizer
is fit (on SOURCE_TRAIN only) and saved -- train_dann.py loads the SAME
saved normalizer rather than re-fitting, so baseline and DANN are
evaluated on identically-normalized inputs and their numbers are directly
comparable.

Usage:
    python -m domain_adaptation.train_baseline --config configs/domain_adaptation.yaml [--pu-root DIR]
"""
from __future__ import annotations

import argparse
import random
import shutil
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from vibration_ad.features.spectrogram import SpectrogramNormalizer, expected_shape

from domain_adaptation.config import Config, load_config
from domain_adaptation.data.dataset import build_domain_adaptation_datasets
from domain_adaptation.data.pu_loader import find_pu_root
from domain_adaptation.data.torch_dataset import LabeledDomainDataset
from domain_adaptation.models.dann import FeatureClassifier
from domain_adaptation.models.feature_extractor import select_device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, loader, optimizer, device, train: bool) -> tuple[float, float]:
    model.train(train)
    total_loss, total_correct, n = 0.0, 0, 0
    criterion = torch.nn.CrossEntropyLoss()
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for X, _domain_id, label in loader:
            X, label = X.to(device), label.to(device)
            _features, logits = model(X)
            loss = criterion(logits, label)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(X)
            total_correct += (logits.argmax(dim=1) == label).sum().item()
            n += len(X)
    return total_loss / n, total_correct / n


def train(config: Config, config_path: str, pu_root, force_data: bool = False) -> None:
    set_seed(config.seed)

    models_dir = config.resolve(config.paths.models_dir)
    reports_dir = config.resolve(config.paths.reports_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("Building datasets (cached after first run)...")
    datasets = build_domain_adaptation_datasets(config, pu_root, force=force_data)
    train_X, train_meta = datasets["source_train"]
    val_X, val_meta = datasets["source_val"]
    print(f"  source_train windows: {train_X.shape}  source_val windows: {val_X.shape}")

    expected = expected_shape(config.common, config.stft)
    if train_X.shape[1:] != expected:
        raise RuntimeError(f"Spectrogram shape {train_X.shape[1:]} != expected {expected}")

    normalizer = SpectrogramNormalizer().fit(train_X)
    train_X_norm = normalizer.transform(train_X)
    val_X_norm = normalizer.transform(val_X)

    device = select_device(config.train_baseline.device)
    print(f"Using device: {device}")

    train_loader = DataLoader(
        LabeledDomainDataset(train_X_norm, train_meta.domain_id.to_numpy(), train_meta.universal_label.to_numpy()),
        batch_size=config.train_baseline.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        LabeledDomainDataset(val_X_norm, val_meta.domain_id.to_numpy(), val_meta.universal_label.to_numpy()),
        batch_size=config.train_baseline.batch_size,
        shuffle=False,
    )

    model = FeatureClassifier(expected, config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.train_baseline.learning_rate, weight_decay=config.train_baseline.weight_decay
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = []
    best_ckpt_path = models_dir / "baseline_best.pt"

    for epoch in range(1, config.train_baseline.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, optimizer, device, train=False)
        dt = time.time() - t0

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_ckpt_path)
        else:
            epochs_without_improvement += 1

        marker = " *" if improved else ""
        print(
            f"epoch {epoch:3d}/{config.train_baseline.epochs}  "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f}  "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}  ({dt:.1f}s){marker}"
        )
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc}
        )

        if epochs_without_improvement >= config.train_baseline.early_stopping_patience:
            print(f"Early stopping: no val improvement in {config.train_baseline.early_stopping_patience} epochs.")
            break

    pd.DataFrame(history).to_csv(reports_dir / "baseline_training_log.csv", index=False)
    normalizer.save(models_dir / "normalizer.npz")
    shutil.copy(config_path, models_dir / "run_config.yaml")

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved to: {best_ckpt_path}")
    print("Next: python -m domain_adaptation.train_dann --config <same config>")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/domain_adaptation.yaml")
    parser.add_argument("--pu-root", default=None, help="Root dir of the PU dataset (default: auto-detect /kaggle/input, else config's pu_root_local).")
    parser.add_argument("--force-data", action="store_true", help="Recompute cached spectrograms even if present.")
    args = parser.parse_args()

    config = load_config(args.config)
    pu_root = args.pu_root or find_pu_root(config.resolve(config.paths.pu_root_local))
    train(config, config_path=args.config, pu_root=pu_root, force_data=args.force_data)


if __name__ == "__main__":
    main()
