"""Deeper diagnostics beyond evaluate.py's core metrics.

Reuses evaluate.py's saved checkpoint, normalizer, cached test
spectrograms, and test_predictions.csv rather than recomputing them --
run this *after* evaluate.py, not instead of it.

Produces:
    outputs/figures/latent_space_pca.png            -- healthy/faulty separation in the bottleneck itself
    outputs/figures/latent_space_by_fault_type.png   -- do the visible sub-clusters correspond to fault type?
    outputs/figures/latent_space_by_rpm.png          -- ...or to load condition?
    outputs/figures/latent_space_by_severity.png     -- ...or to fault severity?
    outputs/figures/latent_space_by_fault_and_severity.png  -- ...or to the combination of both?
    outputs/figures/confusion_matrix.png             -- at the deployed threshold
    outputs/figures/detection_rate_by_rpm.png        -- generalization across load conditions
    outputs/figures/threshold_sensitivity.png        -- how sensitive the 99th-pct choice is
    outputs/reports/threshold_sensitivity.csv

Usage:
    python -m vibration_ad.analyze --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json

import pandas as pd
import torch

from vibration_ad import viz
from vibration_ad.analysis import extract_latents, pca_2d, threshold_sensitivity
from vibration_ad.config import Config, load_config
from vibration_ad.data.dataset import build_raw_datasets
from vibration_ad.evaluate import compute_errors_and_recons
from vibration_ad.features.spectrogram import SpectrogramNormalizer, expected_shape
from vibration_ad.models.autoencoder import ConvAutoencoder, select_device

FAULT_TYPE_ORDER = ["Normal", "B", "IR", "OR@3", "OR@6", "OR@12"]
RPM_ORDER = [1730, 1750, 1772, 1797]
SEVERITY_ORDER = [7, 14, 21, 28]
FAULT_SEVERITY_ORDER = [
    f"{ft} {sz}mil"
    for ft in ["B", "IR", "OR@12", "OR@3", "OR@6"]
    for sz in SEVERITY_ORDER
]  # not every combination exists in the data (e.g. OR@12 14mil) -- those just render as empty (n=0) panels


def analyze(config: Config) -> None:
    models_dir = config.resolve(config.paths.models_dir)
    reports_dir = config.resolve(config.paths.reports_dir)
    figures_dir = config.resolve(config.paths.figures_dir)

    test_predictions_path = reports_dir / "test_predictions.csv"
    metrics_path = reports_dir / "evaluation_metrics.json"
    if not test_predictions_path.exists() or not metrics_path.exists():
        raise FileNotFoundError(
            f"Expected {test_predictions_path} and {metrics_path} -- run "
            "`python -m vibration_ad.evaluate` first."
        )
    test_predictions = pd.read_csv(test_predictions_path)
    metrics = json.loads(metrics_path.read_text())

    print("Loading cached datasets and trained model...")
    _train_X, val_X, _val_meta, test_X, _test_meta = build_raw_datasets(config)
    normalizer = SpectrogramNormalizer.load(models_dir / "normalizer.npz")
    val_X_norm = normalizer.transform(val_X)
    test_X_norm = normalizer.transform(test_X)

    expected = expected_shape(config.windowing, config.stft)
    device = select_device(config.train.device)
    model = ConvAutoencoder(
        input_shape=expected, base_channels=config.model.base_channels, latent_dim=config.model.latent_dim
    ).to(device)
    model.load_state_dict(torch.load(models_dir / "autoencoder_best.pt", map_location=device))

    print("Extracting latent codes and computing threshold sensitivity...")
    val_errors, _ = compute_errors_and_recons(model, val_X_norm, device)
    test_latents = extract_latents(model, test_X_norm, device)

    y_true = test_predictions["label"].to_numpy()
    test_errors = test_predictions["recon_error"].to_numpy()

    coords, explained_var = pca_2d(test_latents)
    viz.plot_latent_space_pca(coords, explained_var, y_true, figures_dir / "latent_space_pca.png")

    print("Faceting the latent space by fault type / RPM / severity...")
    viz.plot_latent_space_faceted(
        coords, y_true, test_predictions["fault_type"].to_numpy(), "fault type",
        figures_dir / "latent_space_by_fault_type.png", category_order=FAULT_TYPE_ORDER,
    )
    viz.plot_latent_space_faceted(
        coords, y_true, test_predictions["rpm"].to_numpy(), "RPM (load condition)",
        figures_dir / "latent_space_by_rpm.png", category_order=RPM_ORDER, ncols=4,
    )
    viz.plot_latent_space_faceted(
        coords, y_true, test_predictions["fault_size_mils"].to_numpy(), "severity (mils)",
        figures_dir / "latent_space_by_severity.png", category_order=SEVERITY_ORDER, ncols=4,
    )
    fault_severity = test_predictions["fault_type"] + " " + test_predictions["fault_size_mils"].astype(str) + "mil"
    viz.plot_latent_space_faceted(
        coords, y_true, fault_severity.to_numpy(), "fault type + severity",
        figures_dir / "latent_space_by_fault_and_severity.png", category_order=FAULT_SEVERITY_ORDER, ncols=5,
    )

    viz.plot_confusion_matrix(metrics["confusion_matrix"], figures_dir / "confusion_matrix.png")
    viz.plot_detection_by_rpm(test_predictions, figures_dir / "detection_rate_by_rpm.png")

    sensitivity_df = threshold_sensitivity(val_errors, test_errors, y_true)
    sensitivity_df.to_csv(reports_dir / "threshold_sensitivity.csv", index=False)
    viz.plot_threshold_sensitivity(
        sensitivity_df, figures_dir / "threshold_sensitivity.png", metrics["threshold_percentile"]
    )

    print(f"Deeper analysis complete. Figures saved to {figures_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    analyze(load_config(args.config))


if __name__ == "__main__":
    main()
