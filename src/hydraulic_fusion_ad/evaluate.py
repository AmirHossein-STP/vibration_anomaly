"""Turn combined reconstruction error into anomaly detection, and score it.

One config = one component's detector (config.data.target_component) --
run this once per configs/hydraulic_{cooler,valve,pump,accumulator}.yaml.

Threshold selection is UNSUPERVISED: a percentile of the combined
reconstruction error on the VALIDATION healthy set only
(config.evaluation.threshold_percentile, default 99th) -- no condition
label is used to pick it. The true profile.txt labels ARE used here, but
only downstream of that choice, to grade the detector. See
notes/09_hydraulic_notes.md sections 2 and 4 for the full reasoning.

Usage:
    python -m hydraulic_fusion_ad.evaluate --config configs/hydraulic_cooler.yaml
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

from hydraulic_fusion_ad import viz
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


@torch.no_grad()
def compute_errors(model, arrays_norm, device, branch_weights, batch_size: int = 64):
    """Returns (combined_errors (N,), {branch: errors (N,)})."""
    loader = DataLoader(MultiBranchDataset(arrays_norm), batch_size=batch_size, shuffle=False)
    combined_chunks = []
    branch_chunks = {b: [] for b in BRANCH_NAMES}
    model.eval()
    for batch in loader:
        batch = {group: t.to(device) for group, t in batch.items()}
        recon = model(batch)
        branch_errors = per_branch_mse(batch, recon)
        combined = combined_mse(branch_errors, branch_weights)
        combined_chunks.append(combined.cpu().numpy())
        for b in BRANCH_NAMES:
            branch_chunks[b].append(branch_errors[b].cpu().numpy())
    combined_errors = np.concatenate(combined_chunks)
    branch_errors_out = {b: np.concatenate(v) for b, v in branch_chunks.items()}
    return combined_errors, branch_errors_out


def _severity_summary(test_meta: pd.DataFrame, target_component: str) -> pd.DataFrame:
    """One row per off-healthy value of `target_component` among degraded
    (label==1) cycles -- the fusion-dataset analogue of CWRU's fault_type x
    fault_size_mils table, just for this one detector's own component
    rather than all four (see the per-component redesign in
    notes/09_hydraulic_notes.md -- each config now targets exactly one
    component, so there's exactly one column to break severity down by).
    """
    degraded = test_meta[test_meta["label"] == 1]
    rows = []
    for value, group in degraded.groupby(target_component):
        rows.append(
            {
                "component": target_component,
                "value": value,
                "detection_rate": group["predicted_anomaly"].mean(),
                "n": len(group),
            }
        )
    return pd.DataFrame(rows).sort_values("value").reset_index(drop=True)


def _branch_error_by_value(test_meta: pd.DataFrame, target_component: str) -> pd.DataFrame:
    """Mean per-branch error grouped by target_component's actual value
    (including the healthy value itself, as a baseline). Renamed from a
    multi-column "degraded_component" grouping in the earlier
    all-four-healthy design -- with one target column per detector now,
    grouping by its own value is the direct analogue.
    """
    summary = (
        test_meta.groupby(target_component)
        .agg(
            slow_error=("slow_error", "mean"),
            medium_error=("medium_error", "mean"),
            fast_error=("fast_error", "mean"),
            n=(target_component, "size"),
        )
        .reset_index()
        .rename(columns={target_component: "degraded_component"})
        .sort_values("n", ascending=False)
    )
    return summary


def evaluate(config: Config) -> dict:
    models_dir = config.resolve(config.paths.models_dir)
    reports_dir = config.resolve(config.paths.reports_dir)
    figures_dir = config.resolve(config.paths.figures_dir)
    for d in (reports_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("Loading cached datasets...")
    _train_arrays, val_arrays, _val_meta, test_arrays, test_meta = build_raw_datasets(config)

    normalizer = MultiBranchNormalizer.load(models_dir / "normalizer.npz", groups=list(BRANCH_NAMES))
    val_norm = normalizer.transform(val_arrays)
    test_norm = normalizer.transform(test_arrays)

    device = select_device(config.train.device)
    model = FusionAutoencoder(config).to(device)
    model.load_state_dict(torch.load(models_dir / "fusion_autoencoder_best.pt", map_location=device))

    print("Computing reconstruction error on validation (healthy) set...")
    val_errors, _ = compute_errors(model, val_norm, device, config.model.branch_weights)
    threshold = float(np.percentile(val_errors, config.evaluation.threshold_percentile))
    print(f"Threshold ({config.evaluation.threshold_percentile}th pct of val healthy error): {threshold:.5f}")

    print("Computing reconstruction error on test (healthy + degraded) set...")
    test_errors, test_branch_errors = compute_errors(model, test_norm, device, config.model.branch_weights)
    test_meta = test_meta.copy()
    test_meta["recon_error"] = test_errors
    for b in BRANCH_NAMES:
        test_meta[f"{b}_error"] = test_branch_errors[b]
    test_meta["predicted_anomaly"] = (test_errors > threshold).astype(int)
    test_meta["label"] = (~test_meta["is_healthy"]).astype(int)

    y_true = test_meta["label"].to_numpy()
    y_pred = test_meta["predicted_anomaly"].to_numpy()

    roc_auc = roc_auc_score(y_true, test_errors)
    pr_auc = average_precision_score(y_true, test_errors)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        "target_component": config.data.target_component,
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
        "n_test_degraded": int((y_true == 1).sum()),
    }
    print(json.dumps(metrics, indent=2))

    severity_summary = _severity_summary(test_meta, config.data.target_component)
    branch_by_component = _branch_error_by_value(test_meta, config.data.target_component)

    # --- save artifacts ---
    with open(reports_dir / "evaluation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    test_meta.to_csv(reports_dir / "test_predictions.csv", index=False)
    severity_summary.to_csv(reports_dir / "detection_rate_by_component_severity.csv", index=False)
    branch_by_component.to_csv(reports_dir / "branch_error_by_component.csv", index=False)

    healthy_errors = test_errors[y_true == 0]
    degraded_errors = test_errors[y_true == 1]
    viz.plot_error_histogram(healthy_errors, degraded_errors, threshold, figures_dir / "error_histogram.png")
    viz.plot_roc_curve(y_true, test_errors, roc_auc, figures_dir / "roc_curve.png")
    viz.plot_confusion_matrix(metrics["confusion_matrix"], figures_dir / "confusion_matrix.png")
    viz.plot_detection_rate_by_value(severity_summary, figures_dir / "detection_rate_by_component_severity.png")
    viz.plot_branch_error_by_component(branch_by_component, figures_dir / "branch_error_by_component.png")

    print(f"\nSaved metrics/plots to {reports_dir} and {figures_dir}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True,
        help="e.g. configs/hydraulic_cooler.yaml (also valve/pump/accumulator)",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    evaluate(config)


if __name__ == "__main__":
    main()
