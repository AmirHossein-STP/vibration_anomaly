"""Deeper diagnostics beyond evaluate.py's core metrics: latent-space
extraction and threshold sensitivity. Torch-dependent helpers used by
analyze.py; plotting itself lives in viz.py to keep this module focused on
computation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader

from vibration_ad.data.torch_dataset import SpectrogramDataset


@torch.no_grad()
def extract_latents(model, X_norm: np.ndarray, device, batch_size: int = 64) -> np.ndarray:
    """Run only the encoder half of the autoencoder, returning the
    bottleneck vector for every input spectrogram: (N, latent_dim).
    """
    loader = DataLoader(SpectrogramDataset(X_norm), batch_size=batch_size, shuffle=False)
    model.eval()
    latents = []
    for batch in loader:
        batch = batch.to(device)
        latents.append(model.encode(batch).cpu().numpy())
    return np.concatenate(latents, axis=0)


def threshold_sensitivity(
    val_errors: np.ndarray,
    test_errors: np.ndarray,
    y_true: np.ndarray,
    percentiles: np.ndarray | None = None,
) -> pd.DataFrame:
    """Precision/recall/F1 across a range of candidate thresholds, each
    still chosen unsupervised (a percentile of validation-healthy error --
    see notes/04_evaluation_notes.md) -- shows how sensitive the deployed
    99th-percentile choice is, without ever picking a threshold using the
    test labels themselves.
    """
    if percentiles is None:
        percentiles = np.arange(80.0, 100.0, 1.0)

    rows = []
    for p in percentiles:
        threshold = float(np.percentile(val_errors, p))
        y_pred = (test_errors > threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        rows.append(
            {"percentile": p, "threshold": threshold, "precision": precision, "recall": recall, "f1": f1}
        )
    return pd.DataFrame(rows)
