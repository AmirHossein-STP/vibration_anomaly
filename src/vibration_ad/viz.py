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


def plot_latent_space_pca(coords: np.ndarray, explained_variance: np.ndarray, labels: np.ndarray, save_path: Path) -> Path:
    """2D PCA projection of the autoencoder's bottleneck vectors, colored by
    true healthy/faulty label. The model never saw these labels during
    training -- if healthy and faulty form visibly separate clusters here,
    that's the compression itself learning a "normal machine" concept, not
    just the reconstruction-error scalar downstream of it.

    coords/explained_variance come from analysis.pca_2d() -- computed once
    and shared with plot_latent_space_faceted() so every latent-space plot
    uses the identical projection, not a separately-refit one.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    for lbl, color, name in [(0, "#3b82f6", "healthy"), (1, "#ef4444", "faulty")]:
        mask = labels == lbl
        ax.scatter(coords[mask, 0], coords[mask, 1], s=10, alpha=0.6, color=color, label=f"{name} (n={mask.sum()})")
    ax.set_xlabel(f"PC1 ({explained_variance[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained_variance[1] * 100:.1f}% variance)")
    ax.set_title("Autoencoder latent space (PCA projection)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_latent_space_faceted(
    coords: np.ndarray,
    labels: np.ndarray,
    category_values: np.ndarray,
    category_name: str,
    save_path: Path,
    category_order: list | None = None,
    ncols: int = 3,
) -> Path:
    """Small-multiples version of plot_latent_space_pca: one panel per
    category value, every point shown faint gray as context, that panel's
    subset highlighted (still split healthy/faulty within the highlight,
    same two colors as everywhere else in this project).

    Why small multiples and not one scatter with 6+ hues: a single
    categorical palette only holds up to about 3 simultaneously
    distinguishable colors on a scatter plot (unlike bars/lines) before
    colorblind-safety and plain legibility break down -- see the dataviz
    skill's palette notes. Faceting one highlighted category per panel
    sidesteps that limit entirely rather than picking 6 colors that don't
    actually separate.
    """
    import pandas as pd  # local import: keep top-level module numpy/sklearn only

    categories = category_order if category_order is not None else sorted(pd.unique(category_values))
    n = len(categories)
    nrows = -(-n // ncols)  # ceil
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.4 * nrows), squeeze=False)
    axes = axes.reshape(-1)

    pad_x = (coords[:, 0].max() - coords[:, 0].min()) * 0.05
    pad_y = (coords[:, 1].max() - coords[:, 1].min()) * 0.05
    xlim = (coords[:, 0].min() - pad_x, coords[:, 0].max() + pad_x)
    ylim = (coords[:, 1].min() - pad_y, coords[:, 1].max() + pad_y)

    for i, cat in enumerate(categories):
        ax = axes[i]
        mask = category_values == cat
        ax.scatter(coords[~mask, 0], coords[~mask, 1], s=5, alpha=0.08, color="#9ca3af", linewidths=0)
        for lbl, color in [(0, "#3b82f6"), (1, "#ef4444")]:
            sub = mask & (labels == lbl)
            if sub.any():
                ax.scatter(coords[sub, 0], coords[sub, 1], s=9, alpha=0.75, color=color, linewidths=0)
        ax.set_title(f"{cat}  (n={int(mask.sum())})", fontsize=10)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xticks([])
        ax.set_yticks([])

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Latent space PCA, faceted by {category_name}", y=1.0)
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
    ax.set_xticklabels(["pred: healthy", "pred: faulty"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["true: healthy", "true: faulty"])
    ax.set_title("Confusion matrix (at deployed threshold)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_detection_by_rpm(test_predictions: "pd.DataFrame", save_path: Path) -> Path:
    """Detection rate among faulty windows, grouped by RPM/load condition --
    a generalization check: healthy TRAINING data came from 1730/1750/1772
    RPM only (see notes/01_dataset_notes.md), so faulty windows from 1797
    RPM are being detected under a load condition the model never even saw
    a healthy example of directly.
    """
    faulty = test_predictions[test_predictions.label == 1]
    summary = (
        faulty.groupby("rpm")
        .agg(detection_rate=("predicted_anomaly", "mean"), n=("predicted_anomaly", "size"))
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(summary["rpm"].astype(str), summary["detection_rate"], color="#22c55e")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("RPM (load condition)")
    ax.set_ylabel("Detection rate")
    ax.set_title("Detection rate by load condition")
    for i, row in summary.iterrows():
        ax.text(i, row["detection_rate"] + 0.02, f"n={row['n']}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_threshold_sensitivity(sensitivity_df: "pd.DataFrame", save_path: Path, deployed_percentile: float = 99.0) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sensitivity_df["percentile"], sensitivity_df["precision"], marker="o", markersize=3, label="precision")
    ax.plot(sensitivity_df["percentile"], sensitivity_df["recall"], marker="o", markersize=3, label="recall")
    ax.plot(sensitivity_df["percentile"], sensitivity_df["f1"], marker="o", markersize=3, label="F1")
    ax.axvline(deployed_percentile, color="gray", linestyle="--", alpha=0.7, label=f"deployed ({deployed_percentile:.0f}th pct)")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Threshold percentile (of validation healthy error)")
    ax.set_ylabel("Score")
    ax.set_title("Threshold sensitivity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_latent_dim_sweep(sweep_df: "pd.DataFrame", save_path: Path, default_value: int = 128) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sweep_df["latent_dim"], sweep_df["roc_auc"], marker="o", label="ROC-AUC")
    ax.plot(sweep_df["latent_dim"], sweep_df["pr_auc"], marker="o", label="PR-AUC")
    ax.plot(sweep_df["latent_dim"], sweep_df["f1"], marker="o", label="F1 (at 99th pct threshold)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sweep_df["latent_dim"])
    ax.set_xticklabels(sweep_df["latent_dim"])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Bottleneck size (latent_dim)")
    ax.set_ylabel("Score")
    ax.set_title("Detection performance vs. autoencoder bottleneck size")
    ax.axvline(default_value, color="gray", linestyle="--", alpha=0.6, label=f"default ({default_value})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path
