"""Matplotlib plotting helpers shared by evaluate.py.

Every function saves a PNG to the given path and returns it; none call
plt.show() since this is meant to run headless (VS Code / CLI / Kaggle).
Colors reused from vibration_ad/viz.py for consistency across both
experiments: blue = healthy, red = degraded/anomaly, green = detection rate.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

BRANCH_COLORS = {"slow": "#f59e0b", "medium": "#8b5cf6", "fast": "#06b6d4"}


def plot_error_histogram(
    healthy_errors: np.ndarray, degraded_errors: np.ndarray, threshold: float, save_path: Path
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, max(healthy_errors.max(), degraded_errors.max()) * 1.05, 60)
    ax.hist(healthy_errors, bins=bins, alpha=0.6, label=f"healthy (n={len(healthy_errors)})", color="#3b82f6")
    ax.hist(degraded_errors, bins=bins, alpha=0.6, label=f"degraded (n={len(degraded_errors)})", color="#ef4444")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5, label=f"threshold={threshold:.4f}")
    ax.set_xlabel("Combined reconstruction error (weighted mean of branch MSEs)")
    ax.set_ylabel("Cycle count")
    ax.set_title("Reconstruction error: healthy vs. degraded")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_roc_curve(y_true: np.ndarray, scores: np.ndarray, auc: float, save_path: Path) -> Path:
    fpr, tpr, _ = roc_curve(y_true, scores)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="#3b82f6", label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve (unsupervised anomaly score vs. true condition labels)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_confusion_matrix(cm: dict, save_path: Path) -> Path:
    """cm: {'tn':..,'fp':..,'fn':..,'tp':..} as produced by evaluate.py."""
    mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(mat, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(mat[i, j]), ha="center", va="center", fontsize=16,
                color="white" if mat[i, j] > mat.max() / 2 else "black",
            )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["pred: healthy", "pred: degraded"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["true: healthy", "true: degraded"])
    ax.set_title("Confusion matrix (at deployed threshold)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_detection_rate_by_value(summary: "pd.DataFrame", save_path: Path) -> Path:
    """summary: columns component, value, detection_rate, n -- one bar per
    (component, off-healthy value), the fusion-dataset analogue of CWRU's
    detection-rate-by-fault-type-and-severity chart.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [f"{r.component}\n={r.value}" for r in summary.itertuples()]
    ax.bar(labels, summary["detection_rate"], color="#22c55e")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Detection rate at threshold")
    ax.set_title("Detection rate by degraded component and severity")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_branch_error_by_component(summary: "pd.DataFrame", save_path: Path) -> Path:
    """summary: columns degraded_component, slow_error, medium_error,
    fast_error, n -- grouped bars, one group per degraded_component
    (including "none" = fully healthy, as a baseline). Answers a question
    CWRU's single-stream model has no equivalent for: which sensor group
    (slow/medium/fast) reacts most to a given kind of degradation.
    """
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(summary))
    width = 0.25
    for i, (branch, color) in enumerate(BRANCH_COLORS.items()):
        ax.bar(x + (i - 1) * width, summary[f"{branch}_error"], width, label=f"{branch} branch", color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["degraded_component"], rotation=30, ha="right")
    ax.set_ylabel("Mean per-branch reconstruction error")
    ax.set_title("Which sensor group reacts to which degradation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_threshold_sensitivity(
    sensitivity_df: "pd.DataFrame", save_path: Path, deployed_percentile: float = 99.0
) -> Path:
    """Includes false_positive_rate alongside precision/recall/F1 --
    critical here, not just extra detail: this dataset's per-component
    test sets skew heavily toward the degraded class, so a percentile can
    post a great F1 purely by flagging most of the test set, including
    most genuinely healthy cycles (see notes/09_hydraulic_notes.md section
    4b) -- a good-looking F1 curve alone would be misleading without it.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sensitivity_df["percentile"], sensitivity_df["precision"], marker="o", markersize=3, label="precision")
    ax.plot(sensitivity_df["percentile"], sensitivity_df["recall"], marker="o", markersize=3, label="recall")
    ax.plot(sensitivity_df["percentile"], sensitivity_df["f1"], marker="o", markersize=3, label="F1")
    ax.plot(
        sensitivity_df["percentile"], sensitivity_df["false_positive_rate"],
        marker="o", markersize=3, linestyle="--", color="#ef4444", label="false positive rate",
    )
    ax.axvline(deployed_percentile, color="gray", linestyle="--", alpha=0.7, label=f"deployed ({deployed_percentile:.0f}th pct)")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Threshold percentile (of validation healthy error)")
    ax.set_ylabel("Score")
    ax.set_title("Threshold sensitivity (watch false positive rate, not just F1)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_confound_check(summary: "pd.DataFrame", save_path: Path) -> Path:
    """summary: columns n_other_degraded, mean_recon_error,
    false_positive_rate, n -- among TARGET-HEALTHY test cycles, does error
    or the false-positive rate climb with how many OTHER components are
    also degraded? A flat line across n_other_degraded is the reassuring
    result (see notes/09_hydraulic_notes.md's "second-order confound" note).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    x = summary["n_other_degraded"]
    ax1.bar(x, summary["mean_recon_error"], color="#3b82f6")
    ax1.set_xlabel("# other components also degraded")
    ax1.set_ylabel("Mean reconstruction error")
    ax1.set_title("Error vs. other-component confound")
    ax2.bar(x, summary["false_positive_rate"], color="#ef4444")
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel("# other components also degraded")
    ax2.set_ylabel("False positive rate")
    ax2.set_title("FP rate vs. other-component confound")
    for ax in (ax1, ax2):
        ax.set_xticks(sorted(x.unique()))
    fig.suptitle(f"n per group: {dict(zip(summary['n_other_degraded'], summary['n']))}", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path
