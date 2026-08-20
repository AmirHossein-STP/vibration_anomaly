"""Matplotlib plotting helpers shared by evaluate.py and infer.py.

Every function saves a PNG to the given path and returns it; none call
plt.show() since this is meant to run headless (VS Code / CLI / CI).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve


def plot_error_histogram(
    healthy_errors: np.ndarray, faulty_errors: np.ndarray, threshold: float, save_path: Path
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(
        0, max(healthy_errors.max(), faulty_errors.max()) * 1.05, 60
    )
    ax.hist(healthy_errors, bins=bins, alpha=0.6, label=f"healthy (n={len(healthy_errors)})", color="#3b82f6")
    ax.hist(faulty_errors, bins=bins, alpha=0.6, label=f"faulty (n={len(faulty_errors)})", color="#ef4444")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5, label=f"threshold={threshold:.4f}")
    ax.set_xlabel("Reconstruction error (MSE)")
    ax.set_ylabel("Window count")
    ax.set_title("Reconstruction error: healthy vs. faulty")
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
    ax.set_title("ROC curve (unsupervised anomaly score vs. true fault label)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_detection_rate_by_severity(summary: "pd.DataFrame", save_path: Path) -> Path:
    """summary: columns fault_type, fault_size_mils, detection_rate, n."""
    import pandas as pd  # local import: keep top-level module numpy/sklearn only

    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [f"{r.fault_type}\n{r.fault_size_mils}mil" for r in summary.itertuples()]
    ax.bar(labels, summary["detection_rate"], color="#22c55e")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Detection rate at threshold")
    ax.set_title("Detection rate by fault type and severity")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_reconstruction_examples(
    examples: list[tuple[str, np.ndarray, np.ndarray]], save_path: Path
) -> Path:
    """examples: list of (title, original_spectrogram, reconstructed_spectrogram),
    each (F, T). Renders original / reconstruction / |diff| columns per row.
    """
    n = len(examples)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[None, :]
    for i, (title, orig, recon) in enumerate(examples):
        diff = np.abs(orig - recon)
        for j, (arr, subtitle) in enumerate(
            [(orig, "input"), (recon, "reconstruction"), (diff, "|difference|")]
        ):
            ax = axes[i, j]
            im = ax.imshow(arr, aspect="auto", origin="lower", cmap="magma")
            ax.set_title(f"{title} - {subtitle}" if j == 0 else subtitle, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_error_timeline(
    errors: np.ndarray, threshold: float, save_path: Path, title: str = "Reconstruction error over time"
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(errors))
    flagged = errors > threshold
    ax.plot(x, errors, color="#3b82f6", linewidth=1, label="reconstruction error")
    ax.scatter(x[flagged], errors[flagged], color="#ef4444", s=20, zorder=3, label="flagged anomaly")
    ax.axhline(threshold, color="black", linestyle="--", linewidth=1.2, label=f"threshold={threshold:.4f}")
    ax.set_xlabel("Window index (time order)")
    ax.set_ylabel("Reconstruction error (MSE)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path
