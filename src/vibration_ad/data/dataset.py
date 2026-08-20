"""Build the train/val/test spectrogram arrays used by every other stage.

Split strategy (see notes/01_dataset_notes.md and notes/04_evaluation_notes.md
for the full reasoning):

  TRAIN (healthy only, dense window overlap):
    Normal.npz windows from config.data.train_rpms (default: 1730/1750/1772).

  VAL (healthy only, sparse overlap):
    first half of Normal.npz windows from config.data.holdout_rpm (1797).
    Used for early stopping and for picking the anomaly threshold.

  TEST (healthy + faulty, sparse overlap):
    second half of Normal.npz windows from holdout_rpm (the "normal" class)
    PLUS every fault file from config.data.fault_rpms x fault_types (the
    "anomaly" class). No fault data, and no holdout-RPM healthy data, is
    ever seen during training.

The val/test split of the holdout healthy file is a contiguous split (first
N windows vs. last M windows), not a random shuffle: because windows overlap
by design, a random split would put near-duplicate windows on both sides of
the split and quietly leak information from val into test.

This module only does deterministic data engineering (no learned state --
normalization is fit later, in train.py). Results are cached to
data/processed/ as plain .npy/.csv so repeated runs skip the STFT recompute.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from vibration_ad.config import Config
from vibration_ad.data.catalog import build_catalog, save_catalog
from vibration_ad.data.windows import make_windows
from vibration_ad.features.spectrogram import windows_to_log_spectrograms


def load_signal(npz_path: str, sensor_key: str) -> np.ndarray:
    with np.load(npz_path) as data:
        if sensor_key not in data:
            raise KeyError(f"{npz_path} has no '{sensor_key}' array (has: {list(data.keys())})")
        return data[sensor_key].astype(np.float64).ravel()


def _windows_for_file(filepath: str, sensor: str, segment_length: int, stride: int) -> np.ndarray:
    signal = load_signal(filepath, sensor)
    return make_windows(signal, segment_length, stride)


def _spectrograms_for_rows(
    rows: pd.DataFrame, config: Config, stride: int
) -> tuple[np.ndarray, pd.DataFrame]:
    """Window + STFT every file in `rows`, concatenate, and return matching
    per-window metadata (one metadata row per output window, not per file).
    """
    all_windows = []
    meta_rows = []
    for _, row in rows.iterrows():
        w = _windows_for_file(row["filepath"], config.data.sensor, config.windowing.segment_length, stride)
        if len(w) == 0:
            continue
        all_windows.append(w)
        for _ in range(len(w)):
            meta_rows.append(
                {
                    "rpm": row["rpm"],
                    "fault_type": row["fault_type"],
                    "fault_size_mils": row["fault_size_mils"],
                    "label": row["label"],
                    "source_file": row["filename"],
                }
            )

    windows = np.concatenate(all_windows, axis=0)
    spectrograms = windows_to_log_spectrograms(windows, config.data.fs_hz, config.stft)
    return spectrograms, pd.DataFrame(meta_rows)


def _cache_paths(processed_dir: Path) -> dict[str, Path]:
    return {
        "train_X": processed_dir / "train_spectrograms.npy",
        "val_X": processed_dir / "val_spectrograms.npy",
        "val_meta": processed_dir / "val_meta.csv",
        "test_X": processed_dir / "test_spectrograms.npy",
        "test_meta": processed_dir / "test_meta.csv",
    }


def build_raw_datasets(
    config: Config, force: bool = False
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Returns (train_X, val_X, val_meta, test_X, test_meta).

    train_X: (N_train, F, T) float32, un-normalized log-magnitude spectrograms.
    val_X:   (N_val, F, T), healthy only.
    test_X:  (N_test, F, T), healthy (label=0) + faulty (label=1).
    *_meta:  one row per window, columns rpm/fault_type/fault_size_mils/label/source_file.
    """
    processed_dir = config.resolve(config.paths.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    paths = _cache_paths(processed_dir)

    if not force and all(p.exists() for p in paths.values()):
        train_X = np.load(paths["train_X"])
        val_X = np.load(paths["val_X"])
        val_meta = pd.read_csv(paths["val_meta"])
        test_X = np.load(paths["test_X"])
        test_meta = pd.read_csv(paths["test_meta"])
        return train_X, val_X, val_meta, test_X, test_meta

    raw_dir = config.resolve(config.paths.raw_dir)
    catalog = build_catalog(raw_dir)
    save_catalog(catalog, processed_dir)

    # --- TRAIN: Normal files from train_rpms, dense overlap ---
    train_rows = catalog[
        (catalog.fault_type == "Normal") & (catalog.rpm.isin(config.data.train_rpms))
    ]
    if len(train_rows) != len(config.data.train_rpms):
        raise RuntimeError(
            f"Expected one Normal.npz per train RPM {config.data.train_rpms}, "
            f"found {len(train_rows)} matching files."
        )
    train_X, _ = _spectrograms_for_rows(train_rows, config, config.windowing.train_stride)

    # --- HOLDOUT healthy: Normal file from holdout_rpm, sparse overlap, split in two ---
    holdout_rows = catalog[
        (catalog.fault_type == "Normal") & (catalog.rpm == config.data.holdout_rpm)
    ]
    if len(holdout_rows) != 1:
        raise RuntimeError(
            f"Expected exactly one Normal.npz for holdout RPM {config.data.holdout_rpm}, "
            f"found {len(holdout_rows)}."
        )
    holdout_X, holdout_meta = _spectrograms_for_rows(holdout_rows, config, config.windowing.eval_stride)
    n_val = int(round(len(holdout_X) * config.train.val_fraction_of_holdout))
    val_X, test_healthy_X = holdout_X[:n_val], holdout_X[n_val:]
    val_meta, test_healthy_meta = holdout_meta.iloc[:n_val].copy(), holdout_meta.iloc[n_val:].copy()

    # --- TEST faulty: every fault file from fault_rpms x fault_types ---
    fault_rows = catalog[
        catalog.fault_type.isin(config.data.fault_types) & catalog.rpm.isin(config.data.fault_rpms)
    ]
    if len(fault_rows) == 0:
        raise RuntimeError("No faulty files matched config.data.fault_types/fault_rpms.")
    fault_X, fault_meta = _spectrograms_for_rows(fault_rows, config, config.windowing.eval_stride)

    test_X = np.concatenate([test_healthy_X, fault_X], axis=0)
    test_meta = pd.concat([test_healthy_meta, fault_meta], axis=0, ignore_index=True)

    np.save(paths["train_X"], train_X)
    np.save(paths["val_X"], val_X)
    val_meta.to_csv(paths["val_meta"], index=False)
    np.save(paths["test_X"], test_X)
    test_meta.to_csv(paths["test_meta"], index=False)

    return train_X, val_X, val_meta, test_X, test_meta
