"""Load the ZeMA hydraulic dataset's raw sensor .txt files + profile.txt.

Each sensor file is tab-separated, one row per 60s cycle, one column per
sample within that cycle (rate-dependent column count -- 60/600/6000, see
configs/hydraulic_*.yaml's sensor_groups). `profile.txt` is one row per
cycle, five columns: the ground-truth condition of all four components
plus a "stable" flag for that cycle.

One config = one component's detector: `data.target_component` says which
of the four columns defines "healthy" for this run (the other three are
read into metadata but not used to decide health) -- see
notes/09_hydraulic_notes.md's per-component redesign section for why a
single all-four-healthy detector didn't work.

Every value this module reads is validated against configs/hydraulic_*.yaml's
`data.valid_values` and checked for consistent row counts across files,
raising a clear error naming the actual vs. expected values on any
mismatch -- this dataset's public documentation wasn't independently
re-verified this session (see notes/09_hydraulic_notes.md section 1 for
why), so silently trusting an assumed column order/value mapping would risk
mislabeling data without any visible failure.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from hydraulic_fusion_ad.config import Config, DataConfig


def _load_sensor_file(raw_dir: Path, sensor: str, expected_samples: int) -> np.ndarray:
    path = raw_dir / f"{sensor}.txt"
    if not path.exists():
        found = sorted(p.name for p in raw_dir.glob("*.txt"))
        raise FileNotFoundError(
            f"Expected sensor file {path} not found. Files present in {raw_dir}: {found}"
        )
    arr = np.loadtxt(path, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != expected_samples:
        raise ValueError(
            f"{path}: expected shape (n_cycles, {expected_samples}), got {arr.shape}. "
            "Check configs/hydraulic.yaml's data.sensor_groups against the actual file."
        )
    return arr


def _load_profile(raw_dir: Path, data_cfg: DataConfig) -> pd.DataFrame:
    path = raw_dir / "profile.txt"
    if not path.exists():
        raise FileNotFoundError(f"Expected {path} (ground-truth condition labels) not found.")
    arr = np.loadtxt(path, dtype=int)
    n_cols = len(data_cfg.profile_columns)
    if arr.ndim != 2 or arr.shape[1] != n_cols:
        raise ValueError(
            f"{path}: expected shape (n_cycles, {n_cols}) matching "
            f"data.profile_columns={data_cfg.profile_columns}, got {arr.shape}."
        )
    df = pd.DataFrame(arr, columns=data_cfg.profile_columns)

    for col, allowed in data_cfg.valid_values.items():
        bad = sorted(set(df[col].unique()) - set(allowed))
        if bad:
            raise ValueError(
                f"profile.txt column '{col}' has value(s) {bad} not in the expected "
                f"set {allowed} (configs/hydraulic.yaml's data.valid_values). This "
                "usually means the assumed column order/mapping is wrong for this "
                "copy of the dataset -- see notes/09_hydraulic_notes.md section 1."
            )
    return df


def build_catalog(raw_dir: Path, config: Config) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Load every sensor file + profile.txt, validate, and return:

    - raw arrays, one per rate group: {"slow": (N, 8, 60), "medium": (N, 2, 600), "fast": (N, 7, 6000)}
      (channel order within a group matches configs/hydraulic.yaml's sensor list order)
    - metadata: one row per cycle, columns = profile columns + derived
      `is_healthy` (bool) and `degraded_component` (comma-joined names of
      whichever of the four condition columns is off its healthy value, or
      "none").
    """
    data_cfg = config.data
    profile = _load_profile(raw_dir, data_cfg)
    n_cycles = len(profile)
    if n_cycles != data_cfg.n_cycles_expected:
        print(
            f"NOTE: profile.txt has {n_cycles} cycles, expected "
            f"{data_cfg.n_cycles_expected} per configs/hydraulic.yaml -- continuing "
            "with the actual count, but double check this is the dataset you expect."
        )

    group_arrays: dict[str, np.ndarray] = {}
    for group_name, group_cfg in data_cfg.sensor_groups.items():
        per_sensor = []
        for sensor in group_cfg.sensors:
            arr = _load_sensor_file(raw_dir, sensor, group_cfg.samples_per_cycle)
            if len(arr) != n_cycles:
                raise ValueError(
                    f"{sensor}.txt has {len(arr)} rows but profile.txt has {n_cycles} -- "
                    "every sensor file and profile.txt must have one row per cycle, in "
                    "the same order."
                )
            per_sensor.append(arr)
        # (n_sensors, N, samples) -> (N, n_sensors, samples)
        group_arrays[group_name] = np.stack(per_sensor, axis=0).transpose(1, 0, 2)

    # `degraded_component`: diagnostic only -- comma-joined names of EVERY
    # column off its healthy value, across all four (not just the target).
    # Kept so you can see, e.g., how many of this detector's "healthy"
    # training cycles are quietly degraded on some OTHER component too --
    # see notes/09_hydraulic_notes.md's per-component redesign section.
    degraded_component = np.full(n_cycles, "", dtype=object)
    for col, healthy_value in data_cfg.healthy_values.items():
        off = (profile[col] != healthy_value).to_numpy()
        degraded_component = np.where(
            off,
            np.where(degraded_component == "", col, degraded_component + "," + col),
            degraded_component,
        )
    degraded_component = np.where(degraded_component == "", "none", degraded_component)

    # `is_healthy`: THE column that actually drives the train/val/test split
    # (see data/dataset.py) -- True iff data.target_component alone is at
    # its healthy value. Requiring all four simultaneously (the original
    # design) left only 20 of 2205 cycles qualifying, nowhere near enough to
    # train on -- see notes/09_hydraulic_notes.md.
    target_healthy_value = data_cfg.healthy_values[data_cfg.target_component]
    is_healthy = (profile[data_cfg.target_component] == target_healthy_value).to_numpy()

    meta = profile.copy()
    meta.insert(0, "cycle_idx", np.arange(n_cycles))
    meta["is_healthy"] = is_healthy
    meta["degraded_component"] = degraded_component

    return group_arrays, meta


def save_catalog(meta: pd.DataFrame, processed_dir: Path) -> Path:
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "catalog.csv"
    meta.to_csv(out_path, index=False)
    return out_path
