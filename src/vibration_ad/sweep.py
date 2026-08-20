"""Bottleneck-size sensitivity sweep.

notes/03_model_notes.md reasons through why the autoencoder's `latent_dim`
is the single most important hyperparameter (too small: can't reconstruct
even healthy signal well, uniformly high error, poor separation; too
large: starts reconstructing faults too, uniformly low error, poor
separation) -- but that was reasoning, not a measurement. This turns it
into one: trains + evaluates the full pipeline across a range of
latent_dim values and plots how detection performance actually responds.

Each value gets its own model/report directory under outputs/sweep/
latent_<N>/, so a normal single-config run's outputs/models,outputs/
reports are never touched. Uses fewer epochs / tighter early stopping
than the default config -- the sweep cares about relative comparison
across values, not squeezing the last bit of performance from any one.
The (expensive) data windowing/STFT step is NOT repeated per value: it's
cached (see data/dataset.build_raw_datasets), so only training itself
repeats.

Usage:
    python -m vibration_ad.sweep --config configs/default.yaml \\
        --values 4,8,16,32,64,128,256,512
"""
from __future__ import annotations

import argparse
import copy

import pandas as pd

from vibration_ad import viz
from vibration_ad.config import Config, load_config
from vibration_ad.evaluate import evaluate
from vibration_ad.train import train

SWEEP_EPOCHS = 25
SWEEP_PATIENCE = 5


def run_sweep(base_config: Config, base_config_path: str, values: list[int]) -> pd.DataFrame:
    sweep_root = base_config.resolve("outputs/sweep")
    sweep_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for latent_dim in values:
        print(f"\n{'=' * 60}\nlatent_dim = {latent_dim}\n{'=' * 60}")
        config = copy.deepcopy(base_config)
        config.model.latent_dim = latent_dim
        config.train.epochs = SWEEP_EPOCHS
        config.train.early_stopping_patience = SWEEP_PATIENCE
        config.paths.models_dir = f"outputs/sweep/latent_{latent_dim}/models"
        config.paths.reports_dir = f"outputs/sweep/latent_{latent_dim}/reports"
        config.paths.figures_dir = f"outputs/sweep/latent_{latent_dim}/figures"

        train(config, config_path=base_config_path)
        metrics = evaluate(config)
        rows.append({"latent_dim": latent_dim, **metrics})

    df = pd.json_normalize(rows)
    summary_path = sweep_root / "summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"\nSweep summary saved to {summary_path}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--values", default="4,8,16,32,64,128,256,512")
    args = parser.parse_args()

    base_config = load_config(args.config)
    values = [int(v) for v in args.values.split(",")]
    df = run_sweep(base_config, args.config, values)

    figures_dir = base_config.resolve(base_config.paths.figures_dir)
    viz.plot_latent_dim_sweep(df, figures_dir / "latent_dim_sweep.png", default_value=base_config.model.latent_dim)
    print(f"Sweep plot saved to {figures_dir / 'latent_dim_sweep.png'}")


if __name__ == "__main__":
    main()
