"""Phase 3: turn reconstruction error into anomaly detection, and score it.

Threshold selection is UNSUPERVISED: it's a percentile of the reconstruction
error on the VALIDATION healthy set only (config.evaluation.threshold_percentile,
default 99th). No fault label is used to pick it -- that's the honest
simulation of "we don't have labels, we only know what healthy looks like".

Fault labels from the CWRU filenames ARE used here, but only downstream of
that choice, to grade the detector: ROC-AUC, PR-AUC, confusion matrix, and
detection rate broken down by fault type/severity. See
notes/04_evaluation_notes.md for why this is legitimate and what it doesn't
prove about a real deployment.

Usage:
    python -m vibration_ad.evaluate --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from vibration_ad import viz
from vibration_ad.config import Config, load_config
from vibration_ad.data.dataset import build_raw_datasets
from vibration_ad.data.torch_dataset import SpectrogramDataset
from vibration_ad.features.spectrogram import SpectrogramNormalizer, expected_shape
from vibration_ad.models.autoencoder import ConvAutoencoder, per_sample_mse, select_device


@torch.no_grad()
def compute_errors_and_recons(model, X_norm: np.ndarray, device, batch_size: int = 64):
    loader = DataLoader(SpectrogramDataset(X_norm), batch_size=batch_size, shuffle=False)
    errors, recons = [], []
    model.eval()
    for batch in loader:
        batch = batch.to(device)
        recon = model(batch)
        errors.append(per_sample_mse(batch, recon).cpu().numpy())
        recons.append(recon.cpu().numpy())
    return np.concatenate(errors), np.concatenate(recons)[:, 0]  # drop channel dim


def evaluate(config: Config) -> dict:
    models_dir = config.resolve(config.paths.models_dir)
    reports_dir = config.resolve(config.paths.reports_dir)
    figures_dir = config.resolve(config.paths.figures_dir)
    for d in (reports_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("Loading cached datasets...")
    _train_X, val_X, _val_meta, test_X, test_meta = build_raw_datasets(config)

    normalizer = SpectrogramNormalizer.load(models_dir / "normalizer.npz")
    val_X_norm = normalizer.transform(val_X)
    test_X_norm = normalizer.transform(test_X)

    expected = expected_shape(config.windowing, config.stft)
    device = select_device(config.train.device)
    model = ConvAutoencoder(
        input_shape=expected,
        base_channels=config.model.base_channels,
        latent_dim=config.model.latent_dim,
    ).to(device)
    model.load_state_dict(torch.load(models_dir / "autoencoder_best.pt", map_location=device))

    print("Computing reconstruction error on validation (healthy) set...")
    val_errors, _ = compute_errors_and_recons(model, val_X_norm, device)
    threshold = float(np.percentile(val_errors, config.evaluation.threshold_percentile))
    print(f"Threshold ({config.evaluation.threshold_percentile}th pct of val healthy error): {threshold:.5f}")

    print("Computing reconstruction error on test (healthy + faulty) set...")
    test_errors, test_recons = compute_errors_and_recons(model, test_X_norm, device)
    test_meta = test_meta.copy()
    test_meta["recon_error"] = test_errors
    test_meta["predicted_anomaly"] = (test_errors > threshold).astype(int)

    y_true = test_meta["label"].to_numpy()
    y_pred = test_meta["predicted_anomaly"].to_numpy()

    roc_auc = roc_auc_score(y_true, test_errors)
    pr_auc = average_precision_score(y_true, test_errors)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        "threshold": threshold,
        "threshold_percentile": config.evaluation.threshold_percentile,
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_val_healthy": int(len(val_errors)),
        "n_test_healthy": int((y_true == 0).sum()),
        "n_test_faulty": int((y_true == 1).sum()),
    }
    print(json.dumps(metrics, indent=2))

    # Detection rate broken down by fault type / severity (faulty rows only).
    faulty_rows = test_meta[test_meta.label == 1]
    severity_summary = (
        faulty_rows.groupby(["fault_type", "fault_size_mils"])
        .agg(detection_rate=("predicted_anomaly", "mean"), n=("predicted_anomaly", "size"))
        .reset_index()
        .sort_values(["fault_type", "fault_size_mils"])
    )

    # --- save artifacts ---
    with open(reports_dir / "evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    test_meta.to_csv(reports_dir / "test_predictions.csv", index=False)
    severity_summary.to_csv(reports_dir / "detection_rate_by_severity.csv", index=False)

    healthy_errors = test_errors[y_true == 0]
    faulty_errors = test_errors[y_true == 1]
    viz.plot_error_histogram(healthy_errors, faulty_errors, threshold, figures_dir / "error_histogram.png")
    viz.plot_roc_curve(y_true, test_errors, roc_auc, figures_dir / "roc_curve.png")
    viz.plot_detection_rate_by_severity(severity_summary, figures_dir / "detection_rate_by_severity.png")

    # A few qualitative examples: best-reconstructed healthy window, and the
    # faulty window with the highest error (clearest "what went wrong").
    examples = []
    idx_healthy_best = np.argmin(np.where(y_true == 0, test_errors, np.inf))
    idx_faulty_worst = np.argmax(np.where(y_true == 1, test_errors, -np.inf))
    for label, idx in [("healthy (low error)", idx_healthy_best), ("faulty (high error)", idx_faulty_worst)]:
        examples.append((label, test_X_norm[idx], test_recons[idx]))
    viz.plot_reconstruction_examples(examples, figures_dir / "reconstruction_examples.png")

    print(f"\nSaved metrics/plots to {reports_dir} and {figures_dir}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    evaluate(config)


if __name__ == "__main__":
    main()
