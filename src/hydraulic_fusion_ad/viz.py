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
