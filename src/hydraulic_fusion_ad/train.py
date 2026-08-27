"""Train the multi-branch fusion autoencoder on one component's healthy
cycles only (config.data.target_component -- one config per component, see
configs/hydraulic_{cooler,valve,pump,accumulator}.yaml and
notes/09_hydraulic_notes.md's per-component redesign section for why).

Usage:
    python -m hydraulic_fusion_ad.train --config configs/hydraulic_cooler.yaml

Produces (all under config.paths.models_dir, e.g. outputs/hydraulic_models/cooler/):
    fusion_autoencoder_best.pt  - state_dict of the best (lowest val loss) checkpoint
    normalizer.npz               - per-sensor mean/std fit on TRAIN cycles
    run_config.yaml              - copy of the config used, for reproducibility
And under config.paths.reports_dir (e.g. outputs/hydraulic_reports/cooler/):
    training_log.csv             - per-epoch combined + per-branch train/val error
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

from hydraulic_fusion_ad.config import Config, load_config
from hydraulic_fusion_ad.data.dataset import build_raw_datasets
from hydraulic_fusion_ad.data.torch_dataset import MultiBranchDataset
from hydraulic_fusion_ad.features.normalize import MultiBranchNormalizer
from hydraulic_fusion_ad.models.fusion_autoencoder import (
    FusionAutoencoder,
    combined_mse,
    per_branch_mse,
    select_device,
)

BRANCH_NAMES = ("slow", "medium", "fast")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, loader, optimizer, device, branch_weights, train: bool) -> dict[str, float]:
    model.train(train)
    totals = {"combined": 0.0, **{b: 0.0 for b in BRANCH_NAMES}}
    n_samples = 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            batch = {group: t.to(device) for group, t in batch.items()}
            recon = model(batch)
            branch_errors = per_branch_mse(batch, recon)  # {group: (B,)}
            combined = combined_mse(branch_errors, branch_weights)  # (B,)
            loss = combined.mean()

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size = combined.shape[0]
            totals["combined"] += combined.sum().item()
            for b in BRANCH_NAMES:
                totals[b] += branch_errors[b].sum().item()
            n_samples += batch_size
    return {k: v / n_samples for k, v in totals.items()}


def train(config: Config, config_path: str, force_data: bool = False) -> None:
    set_seed(config.seed)

    models_dir = config.resolve(config.paths.models_dir)
    reports_dir = config.resolve(config.paths.reports_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("Building datasets (cached after first run)...")
    train_arrays, val_arrays, _val_meta, _test_arrays, _test_meta = build_raw_datasets(config, force=force_data)
    for group in BRANCH_NAMES:
        print(f"  {group}: train {train_arrays[group].shape}  val {val_arrays[group].shape}")

    normalizer = MultiBranchNormalizer().fit(train_arrays)
    train_norm = normalizer.transform(train_arrays)
    val_norm = normalizer.transform(val_arrays)

    device = select_device(config.train.device)
    print(f"Using device: {device}")

    # drop_last=True on TRAIN only: BatchNorm1d errors on a batch of size 1
    # in training mode ("expected more than 1 value per channel"), and
    # n_train isn't guaranteed to divide evenly by batch_size. VAL runs in
    # eval mode (uses running stats, not batch stats) so this isn't a risk
    # there, and dropping val/test data would bias the reported error.
    train_loader = DataLoader(
        MultiBranchDataset(train_norm), batch_size=config.train.batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(MultiBranchDataset(val_norm), batch_size=config.train.batch_size, shuffle=False)

    model = FusionAutoencoder(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = []
    best_ckpt_path = models_dir / "fusion_autoencoder_best.pt"

    for epoch in range(1, config.train.epochs + 1):
        t0 = time.time()
        train_stats = run_epoch(model, train_loader, optimizer, device, config.model.branch_weights, train=True)
        val_stats = run_epoch(model, val_loader, optimizer, device, config.model.branch_weights, train=False)
        dt = time.time() - t0

        improved = val_stats["combined"] < best_val_loss
        if improved:
            best_val_loss = val_stats["combined"]
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_ckpt_path)
        else:
            epochs_without_improvement += 1

        marker = " *" if improved else ""
        print(
            f"epoch {epoch:3d}/{config.train.epochs}  "
            f"train_mse={train_stats['combined']:.5f}  val_mse={val_stats['combined']:.5f}  "
            f"(slow/med/fast val: {val_stats['slow']:.4f}/{val_stats['medium']:.4f}/{val_stats['fast']:.4f})  "
            f"({dt:.1f}s){marker}"
        )
        history.append(
            {"epoch": epoch}
            | {f"train_{k}": v for k, v in train_stats.items()}
            | {f"val_{k}": v for k, v in val_stats.items()}
        )

        if epochs_without_improvement >= config.train.early_stopping_patience:
            print(f"Early stopping: no val improvement in {config.train.early_stopping_patience} epochs.")
            break

    pd.DataFrame(history).to_csv(reports_dir / "training_log.csv", index=False)
    normalizer.save(models_dir / "normalizer.npz")
    shutil.copy(config_path, models_dir / "run_config.yaml")

    print(f"\nBest val combined MSE: {best_val_loss:.5f}")
    print(f"Checkpoint saved to: {best_ckpt_path}")
    print(f"Normalizer saved to: {models_dir / 'normalizer.npz'}")
    print("Next: python -m hydraulic_fusion_ad.evaluate --config <same config>")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # No default: four separate configs exist (one per target_component),
    # a silent default risks training the wrong component without noticing.
    parser.add_argument(
        "--config", required=True,
        help="e.g. configs/hydraulic_cooler.yaml (also valve/pump/accumulator)",
    )
    parser.add_argument(
        "--force-data", action="store_true", help="Recompute cached split arrays even if present."
    )
    args = parser.parse_args()

    config = load_config(args.config)
    train(config, config_path=args.config, force_data=args.force_data)


if __name__ == "__main__":
    main()
