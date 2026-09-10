"""Matplotlib plotting helpers for evaluate.py. Same conventions as
vibration_ad.viz / hydraulic_fusion_ad.viz: every function saves a PNG and
returns the path, headless (Agg backend), no plt.show().
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CLASS_NAMES = ["healthy", "inner_race", "outer_race"]
# Same blue/red semantic pair validated (dataviz-skill palette check) and
# reused across every experiment in this repo for healthy/faulty; outer_race
# gets a third, distinguishable hue since this is a 3-class task, not binary.
CLASS_COLORS = {"healthy": "#3b82f6", "inner_race": "#ef4444", "outer_race": "#f59e0b"}
DOMAIN_NAMES = {0: "cwru", 1: "pu_artificial", 2: "pu_target"}
DOMAIN_COLORS = {"cwru": "#3b82f6", "pu_artificial": "#8b5cf6", "pu_target": "#ef4444"}


def plot_confusion_matrix_multiclass(cm: np.ndarray, save_path: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]), ha="center", va="center", fontsize=13,
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels([f"pred: {n}" for n in CLASS_NAMES], rotation=30, ha="right")
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_yticklabels([f"true: {n}" for n in CLASS_NAMES])
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_baseline_vs_dann_bars(metrics_baseline: dict, metrics_dann: dict | None, save_path: Path) -> Path:
    """Side-by-side accuracy/macro-F1 on the TARGET_TEST zero-shot set --
    the headline "did adaptation help" comparison."""
    metric_names = ["accuracy", "macro_f1"]
    models = [("baseline (no adaptation)", metrics_baseline, "#9ca3af")]
    if metrics_dann is not None:
        models.append(("DANN (adapted)", metrics_dann, "#3b82f6"))

    x = np.arange(len(metric_names))
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (name, metrics, color) in enumerate(models):
        values = [metrics[m] for m in metric_names]
        bars = ax.bar(x + i * width, values, width, label=name, color=color)
        ax.bar_label(bars, fmt="%.3f", fontsize=9, padding=2)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(["accuracy", "macro F1"])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score on TARGET_TEST (zero-shot, held-out condition)")
    ax.set_title("Domain-shift performance: before vs. after adaptation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_feature_space(
    coords: np.ndarray, domain_ids: np.ndarray, labels: np.ndarray, save_path: Path, method_name: str = "t-SNE"
) -> Path:
    """Two-panel 2D projection of feature-extractor embeddings: left
    colored by domain (does CWRU / PU_Artificial / PU_Target overlap?),
    right colored by fault class (do target real faults fall inside the
    source fault clusters?) -- the two hypotheses DOMAIN_ADAPTATION_PLAN.md
    Phase 4 asks to check visually.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    ax = axes[0]
    for domain_id, name in DOMAIN_NAMES.items():
        mask = domain_ids == domain_id
        if mask.any():
            ax.scatter(coords[mask, 0], coords[mask, 1], s=8, alpha=0.5, color=DOMAIN_COLORS[name], label=f"{name} (n={mask.sum()})")
    ax.set_title(f"{method_name}: colored by domain")
    ax.legend(fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axes[1]
    for label_id, name in enumerate(CLASS_NAMES):
        mask = labels == label_id
        if mask.any():
            ax.scatter(coords[mask, 0], coords[mask, 1], s=8, alpha=0.5, color=CLASS_COLORS[name], label=f"{name} (n={mask.sum()})")
    ax.set_title(f"{method_name}: colored by fault class")
    ax.legend(fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path
