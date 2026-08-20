"""Train the autoencoder on healthy-only spectrograms.

Usage:
    python -m vibration_ad.train --config configs/default.yaml

Produces (all under config.paths.models_dir, default outputs/models/):
    autoencoder_best.pt   - state_dict of the best (lowest val loss) checkpoint
    normalizer.npz         - per-frequency-bin mean/std fit on TRAIN spectrograms
    run_config.yaml        - copy of the config used, for reproducibility
And under config.paths.reports_dir (default outputs/reports/):
    training_log.csv       - per-epoch train/val reconstruction error
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

from vibration_ad.config import Config, load_config
from vibration_ad.data.dataset import build_raw_datasets
from vibration_ad.data.torch_dataset import SpectrogramDataset
from vibration_ad.features.spectrogram import SpectrogramNormalizer, expected_shape
from vibration_ad.models.autoencoder import ConvAutoencoder, per_sample_mse, select_device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, loader, optimizer, device, train: bool) -> float:
    model.train(train)
    total_loss, n_samples = 0.0, 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            batch = batch.to(device)
            recon = model(batch)
            per_sample = per_sample_mse(batch, recon)
            loss = per_sample.mean()

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += per_sample.sum().item()
            n_samples += len(batch)
    return total_loss / n_samples


def train(config: Config, config_path: str, force_data: bool = False) -> None:
    set_seed(config.seed)

    models_dir = config.resolve(config.paths.models_dir)
    reports_dir = config.resolve(config.paths.reports_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("Building datasets (cached after first run)...")
    train_X, val_X, _val_meta, _test_X, _test_meta = build_raw_datasets(config, force=force_data)
    print(f"  train windows: {train_X.shape}  val windows: {val_X.shape}")

    expected = expected_shape(config.windowing, config.stft)
    if train_X.shape[1:] != expected:
        raise RuntimeError(f"Spectrogram shape {train_X.shape[1:]} != expected {expected}")

    normalizer = SpectrogramNormalizer().fit(train_X)
    train_X_norm = normalizer.transform(train_X)
    val_X_norm = normalizer.transform(val_X)

    device = select_device(config.train.device)
    print(f"Using device: {device}")

    train_loader = DataLoader(
        SpectrogramDataset(train_X_norm), batch_size=config.train.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        SpectrogramDataset(val_X_norm), batch_size=config.train.batch_size, shuffle=False
    )

    model = ConvAutoencoder(
        input_shape=expected,
        base_channels=config.model.base_channels,
        latent_dim=config.model.latent_dim,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = []
    best_ckpt_path = models_dir / "autoencoder_best.pt"

    for epoch in range(1, config.train.epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, device, train=False)
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
            f"epoch {epoch:3d}/{config.train.epochs}  "
            f"train_mse={train_loss:.5f}  val_mse={val_loss:.5f}  ({dt:.1f}s){marker}"
        )
        history.append({"epoch": epoch, "train_mse": train_loss, "val_mse": val_loss})

        if epochs_without_improvement >= config.train.early_stopping_patience:
            print(f"Early stopping: no val improvement in {config.train.early_stopping_patience} epochs.")
            break

    pd.DataFrame(history).to_csv(reports_dir / "training_log.csv", index=False)
    normalizer.save(models_dir / "normalizer.npz")
    shutil.copy(config_path, models_dir / "run_config.yaml")

    print(f"\nBest val MSE: {best_val_loss:.5f}")
    print(f"Checkpoint saved to: {best_ckpt_path}")
    print(f"Normalizer saved to: {models_dir / 'normalizer.npz'}")
    print("Next: python -m vibration_ad.evaluate --config <same config>")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--force-data", action="store_true", help="Recompute cached spectrograms even if present."
    )
    args = parser.parse_args()

    config = load_config(args.config)
    train(config, config_path=args.config, force_data=args.force_data)


if __name__ == "__main__":
    main()
