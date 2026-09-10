"""Deeper diagnostics beyond evaluate.py's core metrics -- built after the
first two real Kaggle runs raised three specific open questions (see
notes/09_hydraulic_notes.md sections 2 and 5):

  1. The deployed 99th-percentile threshold gave near-zero recall for
     valve/pump despite good ROC-AUC -- is a different percentile better?
  2. Does a detector's "healthy" score secretly react to some OTHER
     component being degraded (the "second-order confound")?
  3. Is this component's condition itself correlated with campaign time in
     the raw data (a property no split can fix, unlike the first,
     split-level time confound already fixed in data/dataset.py)?

Reuses evaluate.py's saved checkpoint, normalizer, cached split, and
test_predictions.csv rather than recomputing them -- run this *after*
evaluate.py, not instead of it.

Produces (all under config.paths.reports_dir / figures_dir):
    threshold_sensitivity.csv / .png   -- question 1
    confound_check.csv / .png          -- question 2
    time_correlation.json              -- question 3 (numbers only, no plot --
                                           it's one number per profile column)

Usage:
    python -m hydraulic_fusion_ad.analyze --config configs/hydraulic_cooler.yaml
"""
from __future__ import annotations

import argparse
import json

import pandas as pd
import torch

from hydraulic_fusion_ad import viz
from hydraulic_fusion_ad.analysis import (
    best_deployable_threshold,
    confound_check,
    f1_optimal_threshold,
    raw_time_correlation,
    threshold_sensitivity,
)
from hydraulic_fusion_ad.config import Config, load_config
from hydraulic_fusion_ad.data.dataset import build_raw_datasets
from hydraulic_fusion_ad.evaluate import compute_errors
from hydraulic_fusion_ad.features.normalize import MultiBranchNormalizer
from hydraulic_fusion_ad.models.fusion_autoencoder import FusionAutoencoder, select_device

BRANCH_NAMES = ("slow", "medium", "fast")


def analyze(config: Config) -> None:
    reports_dir = config.resolve(config.paths.reports_dir)
    figures_dir = config.resolve(config.paths.figures_dir)
    models_dir = config.resolve(config.paths.models_dir)
    processed_dir = config.resolve(config.paths.processed_dir)

    test_predictions_path = reports_dir / "test_predictions.csv"
    metrics_path = reports_dir / "evaluation_metrics.json"
    if not test_predictions_path.exists() or not metrics_path.exists():
        raise FileNotFoundError(
            f"Expected {test_predictions_path} and {metrics_path} -- run "
            "`python -m hydraulic_fusion_ad.evaluate` first."
        )
    test_predictions = pd.read_csv(test_predictions_path)
    metrics = json.loads(metrics_path.read_text())

    print("Loading cached val split and trained model (for threshold sensitivity)...")
    _train_arrays, val_arrays, _val_meta, _test_arrays, _test_meta = build_raw_datasets(config)
    normalizer = MultiBranchNormalizer.load(models_dir / "normalizer.npz", groups=list(BRANCH_NAMES))
    val_norm = normalizer.transform(val_arrays)

    device = select_device(config.train.device)
    model = FusionAutoencoder(config).to(device)
    model.load_state_dict(torch.load(models_dir / "fusion_autoencoder_best.pt", map_location=device))
    val_errors, _ = compute_errors(model, val_norm, device, config.model.branch_weights)

    y_true = test_predictions["label"].to_numpy()
    test_errors = test_predictions["recon_error"].to_numpy()

    print("1. Threshold sensitivity...")
    sensitivity_df = threshold_sensitivity(val_errors, test_errors, y_true)
    sensitivity_df.to_csv(reports_dir / "threshold_sensitivity.csv", index=False)
    viz.plot_threshold_sensitivity(
        sensitivity_df, figures_dir / "threshold_sensitivity.png", metrics["threshold_percentile"]
    )
    # Reported for comparison ONLY -- see f1_optimal_threshold's and
    # best_deployable_threshold's docstrings for why NEITHER of these
    # becomes the deployed threshold.
    cheat_comparison = {
        "deployed": {
            "percentile": metrics["threshold_percentile"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        },
        "raw_f1_optimal__MISLEADING_ALONE": f1_optimal_threshold(sensitivity_df),
        "f1_optimal_at_fpr_10pct_or_less": best_deployable_threshold(sensitivity_df, max_fpr=0.10),
        "warning": (
            "Neither comparison entry is a valid detector choice -- both are chosen by "
            "looking at test labels, which the deployed threshold never does. "
            "raw_f1_optimal can (and on this dataset's skewed test sets, often does) achieve "
            "high F1 by flagging most of the test set, including most genuinely healthy "
            "cycles -- always check its false_positive_rate before reading its f1 as good "
            "news. f1_optimal_at_fpr_10pct_or_less is null if no swept percentile keeps "
            "false_positive_rate at or below 10% at all -- that itself is a real, important "
            "result (no simple threshold works well for this detector at that FPR tolerance). "
            "See notes/09_hydraulic_notes.md section 4b."
        ),
    }
    (reports_dir / "threshold_cheat_comparison.json").write_text(json.dumps(cheat_comparison, indent=2))
    print(json.dumps(cheat_comparison, indent=2))

    print("2. Second-order confound check...")
    confound_df = confound_check(test_predictions)
    confound_df.to_csv(reports_dir / "confound_check.csv", index=False)
    viz.plot_confound_check(confound_df, figures_dir / "confound_check.png")

    print("3. Raw time-correlation (does this component's condition drift with campaign time?)...")
    catalog = pd.read_csv(processed_dir / "catalog.csv")
    time_corr = raw_time_correlation(catalog, config.data.profile_columns)
    (reports_dir / "time_correlation.json").write_text(json.dumps(time_corr, indent=2))
    print(json.dumps(time_corr, indent=2))

    print(f"\nAnalysis complete. Reports/figures saved to {reports_dir} / {figures_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True,
        help="e.g. configs/hydraulic_cooler.yaml (also valve/pump/accumulator)",
    )
    args = parser.parse_args()
    analyze(load_config(args.config))


if __name__ == "__main__":
    main()
