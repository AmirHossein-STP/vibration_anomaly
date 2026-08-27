"""Build the train/val/test split used by every other stage.

Split strategy (see notes/09_hydraulic_notes.md's "Block-shuffled split"
section for the full reasoning, including why "healthy" means
"config.data.target_component at its healthy value", NOT "all four
components healthy at once" -- that stricter definition left only 20 of
2205 cycles usable):

  TRAIN/VAL/TEST-healthy: cycles where target_component is healthy, split
    by BLOCK-SHUFFLING -- chop them into contiguous blocks of
    `split.block_size` consecutive cycles (in original file order), shuffle
    the BLOCKS (seeded by config.seed), then take the first
    `split.train_fraction` of blocks for train, the next
    `split.val_fraction` for val, the rest for test.
  TEST also includes every cycle where target_component is NOT healthy.

Why block-shuffled and not a straight chronological split (the first
design, reverted after a real run): the test rig runs each component's
conditions in long, non-interleaved stretches, so "first 70% of
target-healthy cycles chronologically" confines TRAIN/VAL to one narrow
slice of the whole campaign's calendar time -- and reconstruction error
turned out to correlate far more with TIME (thermal/sensor drift over the
campaign, |r| up to 0.83) than with the actual target condition, making
the first real run's numbers meaningless (one component's near-perfect
score was time-drift correlating with its particular healthy-window
placement, not real detection). Why not a pure per-cycle random shuffle
either: adjacent cycles are still autocorrelated (thermal/wear state
carries over cycle-to-cycle), so an unblocked shuffle risks leaking
near-duplicate neighbors across splits the same way CWRU's window overlap
does. Blocks are the middle ground: shuffled at a coarse enough grain to
spread train/val/test across the FULL campaign timespan (breaking the
time confound), while keeping short local structure intact within a block.

No windowing/feature-extraction step here (unlike vibration_ad's
dataset.py) -- each cycle is already one fixed-length example per sensor
group. This module only slices and normalizes-later (normalization is fit
in train.py, same division of responsibility as vibration_ad). Results are
cached to data/hydraulic_processed/ so repeated runs skip re-parsing the
(large) raw .txt files.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from hydraulic_fusion_ad.config import Config
from hydraulic_fusion_ad.data.catalog import build_catalog, save_catalog

SplitArrays = dict[str, np.ndarray]  # {"slow": ..., "medium": ..., "fast": ...}


def _cache_paths(processed_dir: Path) -> dict[str, Path]:
    return {
        "train": processed_dir / "train_arrays.npz",
        "val": processed_dir / "val_arrays.npz",
        "val_meta": processed_dir / "val_meta.csv",
        "test": processed_dir / "test_arrays.npz",
        "test_meta": processed_dir / "test_meta.csv",
    }


def _save_arrays(arrays: SplitArrays, path: Path) -> None:
    np.savez(path, **arrays)


def _load_arrays(path: Path) -> SplitArrays:
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def _slice(arrays: dict[str, np.ndarray], idx: np.ndarray) -> SplitArrays:
    return {group: arr[idx] for group, arr in arrays.items()}


def _block_shuffle_split(
    healthy_idx: np.ndarray, train_fraction: float, val_fraction: float, block_size: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Partition healthy_idx (cycle indices, in original chronological
    order) into (train_idx, val_idx, test_healthy_idx) by shuffling
    contiguous BLOCKS of `block_size` cycles, not individual cycles and not
    the raw chronological order -- see this module's docstring for why.
    """
    blocks = [healthy_idx[i : i + block_size] for i in range(0, len(healthy_idx), block_size)]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(blocks))
    blocks = [blocks[i] for i in order]

    n_blocks = len(blocks)
    n_train_blocks = int(round(n_blocks * train_fraction))
    n_val_blocks = int(round(n_blocks * val_fraction))

    def _concat(bs: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(bs) if bs else np.array([], dtype=healthy_idx.dtype)

    train_idx = _concat(blocks[:n_train_blocks])
    val_idx = _concat(blocks[n_train_blocks : n_train_blocks + n_val_blocks])
    test_healthy_idx = _concat(blocks[n_train_blocks + n_val_blocks :])
    return train_idx, val_idx, test_healthy_idx


def build_raw_datasets(
    config: Config, force: bool = False
) -> tuple[SplitArrays, SplitArrays, pd.DataFrame, SplitArrays, pd.DataFrame]:
    """Returns (train_arrays, val_arrays, val_meta, test_arrays, test_meta).

    Each *_arrays is {"slow": (N,8,60), "medium": (N,2,600), "fast": (N,7,6000)},
    un-normalized float32. *_meta is one row per cycle (see catalog.build_catalog).
    """
    processed_dir = config.resolve(config.paths.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    paths = _cache_paths(processed_dir)

    if not force and all(p.exists() for p in paths.values()):
        train_arrays = _load_arrays(paths["train"])
        val_arrays = _load_arrays(paths["val"])
        val_meta = pd.read_csv(paths["val_meta"])
        test_arrays = _load_arrays(paths["test"])
        test_meta = pd.read_csv(paths["test_meta"])
        return train_arrays, val_arrays, val_meta, test_arrays, test_meta

    raw_dir = config.resolve(config.paths.raw_dir)
    group_arrays, meta = build_catalog(raw_dir, config)
    save_catalog(meta, processed_dir)

    healthy_idx = meta.loc[meta.is_healthy, "cycle_idx"].to_numpy()  # already in original order
    non_healthy_idx = meta.loc[~meta.is_healthy, "cycle_idx"].to_numpy()

    n_healthy = len(healthy_idx)
    train_idx, val_idx, test_healthy_idx = _block_shuffle_split(
        healthy_idx, config.split.train_fraction, config.split.val_fraction, config.split.block_size, config.seed
    )
    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_healthy_idx) == 0:
        raise RuntimeError(
            f"Split fractions/block_size leave an empty split: {n_healthy} target-healthy "
            f"cycles total, train_fraction={config.split.train_fraction}, "
            f"val_fraction={config.split.val_fraction}, block_size={config.split.block_size} "
            f"-> train={len(train_idx)}, val={len(val_idx)}, test_healthy={len(test_healthy_idx)}."
        )
    test_idx = np.sort(np.concatenate([test_healthy_idx, non_healthy_idx]))

    train_arrays = _slice(group_arrays, train_idx)
    val_arrays = _slice(group_arrays, val_idx)
    test_arrays = _slice(group_arrays, test_idx)

    val_meta = meta.iloc[val_idx].reset_index(drop=True)
    test_meta = meta.iloc[test_idx].reset_index(drop=True)

    _save_arrays(train_arrays, paths["train"])
    _save_arrays(val_arrays, paths["val"])
    val_meta.to_csv(paths["val_meta"], index=False)
    _save_arrays(test_arrays, paths["test"])
    test_meta.to_csv(paths["test_meta"], index=False)

    print(
        f"Split (block-shuffled, block_size={config.split.block_size}): "
        f"{len(train_idx)} train (healthy) / {len(val_idx)} val (healthy) / "
        f"{len(test_idx)} test ({len(test_healthy_idx)} healthy + "
        f"{len(non_healthy_idx)} degraded)."
    )
    return train_arrays, val_arrays, val_meta, test_arrays, test_meta
