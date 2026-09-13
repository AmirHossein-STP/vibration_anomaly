"""Build the four data splits every downstream stage reads: SOURCE_TRAIN,
SOURCE_VAL, TARGET_ADAPT, TARGET_TEST.

Domain id (used by the DANN discriminator; source_val/target_test rows
still get one for plotting, just never see the discriminator loss):
    0 = cwru            1 = pu_artificial            2 = pu_target
(pu_target covers BOTH the held-out condition's healthy rows and its real
fault rows -- physically the same rig/condition, just different bearing
health, so they're one domain either way.)

Role -> what it's used for and where its rows come from:
    SOURCE_TRAIN  supervised classification loss + domain loss (DANN).
                  CWRU rows from cwru.train_rpms (ALL fault types, labels
                  used -- this is supervised, unlike vibration_ad's
                  healthy-only autoencoder) + PU rows from pu.source_codes
                  under pu.source_conditions, first
                  pu.source_train_fraction of each bearing's ~20 runs.
    SOURCE_VAL    supervised loss only, for early stopping / baseline
                  metric -- NEVER seen by the domain/anchor loss. CWRU's
                  held-out RPM (cwru.val_rpm) + PU's remaining source runs.
    TARGET_ADAPT  domain loss + healthy-anchor loss ONLY -- classification
                  labels are never used here even though they're known
                  (all healthy). PU healthy-code rows under
                  pu.target_condition, first
                  pu.target_normal_adapt_fraction of runs. This is the
                  ENTIRE adaptation signal: no fault example, labeled or
                  not, from the target ever appears during training.
    TARGET_TEST   evaluate.py ONLY -- the zero-shot check. PU healthy-code
                  rows under pu.target_condition NOT used in TARGET_ADAPT
                  (label=0), plus ALL of pu.real_codes under
                  pu.target_condition (label=1 or 2, per bearing code).

Split-by-run (not by window) for PU, same anti-leakage spirit as CWRU's
RPM holdout and vibration_ad's contiguous val/test split of overlapping
windows: a run is one continuous ~4s recording, so splitting whole runs
(rather than shuffling windows across the split) keeps near-duplicate
overlapping windows from a single run on only one side.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from vibration_ad.data.dataset import load_signal as load_cwru_signal
from vibration_ad.data.windows import make_windows
from vibration_ad.features.spectrogram import windows_to_log_spectrograms

from domain_adaptation.config import Config
from domain_adaptation.data.cwru_loader import build_cwru_catalog
from domain_adaptation.data.pu_loader import UnreadableMatFile, build_pu_catalog, load_vibration_signal
from domain_adaptation.data.transforms import resample_to_rate

DOMAIN_IDS = {"cwru": 0, "pu_artificial": 1, "pu_target": 2}
ROLES = ("source_train", "source_val", "target_adapt", "target_test")


def _split_pu_runs_by_fraction(df: pd.DataFrame, fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split each bearing code's runs into a contiguous (first_part,
    second_part) pair by run index, independently per code (so e.g. one
    code having 18 runs and another 20 both split proportionally).
    """
    n_runs = df.groupby("code")["run"].transform("nunique")
    cutoff = (n_runs * fraction).round()
    first = df[df["run"] <= cutoff]
    second = df[df["run"] > cutoff]
    return first.copy(), second.copy()


def _assign_pu_roles(pu_catalog: pd.DataFrame, config: Config) -> pd.DataFrame:
    pu_cfg = config.pu
    source_mask = pu_catalog.code.isin(pu_cfg.source_codes) & pu_catalog.condition.isin(pu_cfg.source_conditions)
    target_healthy_mask = pu_catalog.code.isin(pu_cfg.healthy_codes) & (pu_catalog.condition == pu_cfg.target_condition)
    target_real_mask = pu_catalog.code.isin(pu_cfg.real_codes) & (pu_catalog.condition == pu_cfg.target_condition)

    parts = []

    source_train, source_val = _split_pu_runs_by_fraction(pu_catalog[source_mask], pu_cfg.source_train_fraction)
    source_train["role"], source_train["domain"] = "source_train", "pu_artificial"
    source_val["role"], source_val["domain"] = "source_val", "pu_artificial"
    parts += [source_train, source_val]

    target_adapt, target_healthy_test = _split_pu_runs_by_fraction(
        pu_catalog[target_healthy_mask], pu_cfg.target_normal_adapt_fraction
    )
    target_adapt["role"], target_adapt["domain"] = "target_adapt", "pu_target"
    target_healthy_test["role"], target_healthy_test["domain"] = "target_test", "pu_target"
    parts += [target_adapt, target_healthy_test]

    target_real_test = pu_catalog[target_real_mask].copy()
    target_real_test["role"], target_real_test["domain"] = "target_test", "pu_target"
    parts.append(target_real_test)

    assigned = pd.concat(parts, ignore_index=True)
    for role in ROLES:
        if assigned[assigned.role == role].empty:
            raise RuntimeError(f"PU role assignment produced zero rows for {role!r} -- check config.pu's code/condition lists.")
    return assigned


def _windows_and_meta_for_cwru_rows(rows: pd.DataFrame, config: Config, stride: int) -> tuple[np.ndarray, pd.DataFrame]:
    all_windows, meta_rows = [], []
    for _, row in rows.iterrows():
        signal = load_cwru_signal(row["filepath"], config.cwru.sensor)  # already 12kHz == common.fs_hz
        w = make_windows(signal, config.common.segment_length, stride)
        if len(w) == 0:
            continue
        all_windows.append(w)
        for _ in range(len(w)):
            meta_rows.append(
                {
                    "domain": row["domain"],
                    "domain_id": DOMAIN_IDS["cwru"],
                    "role": row["role"],
                    "universal_label": row["universal_label"],
                    "condition": row["condition"],
                    "code": "CWRU",
                    "source_file": Path(row["filepath"]).name,
                }
            )
    if not all_windows:
        return np.empty((0, config.common.segment_length), dtype=np.float64), pd.DataFrame(meta_rows)
    return np.concatenate(all_windows, axis=0), pd.DataFrame(meta_rows)


# A few unreadable files out of ~1400 is a flaw in the Kaggle mirror, not a
# reason to throw away a 100-minute run; a LOT of them means something
# systemic (wrong dataset attached, truncated mount) that must not be
# silently trained through. Same posture the CWRU downloader landed on --
# see vibration_ad's notes/07_dvc_and_kaggle.md, where the opposite bug
# (swallowing a total failure) hid a broken run for 35 minutes.
# Calibration matters here. At ~1000 files per split, one entirely-corrupt
# bearing code is 3 conditions x 20 runs = 60 files = ~6%, which is NOT
# systemic -- it costs 1 of 18 codes' diversity and should proceed loudly,
# not abort. A broken mount or wrong dataset is far above that. 10% sits
# between the two.
MAX_UNREADABLE_FRACTION = 0.10


def _windows_and_meta_for_pu_rows(rows: pd.DataFrame, config: Config, stride: int) -> tuple[np.ndarray, pd.DataFrame]:
    all_windows, meta_rows = [], []
    unreadable: list[str] = []
    for _, row in rows.iterrows():
        try:
            signal = load_vibration_signal(row["filepath"], config.pu.vibration_channel)
        except UnreadableMatFile as exc:
            unreadable.append(str(row["filepath"]))
            print(f"  SKIPPING unreadable file: {exc}")
            continue
        signal = resample_to_rate(signal, row["fs_native_hz"], config.common.fs_hz)
        w = make_windows(signal, config.common.segment_length, stride)
        if len(w) == 0:
            continue
        all_windows.append(w)
        for _ in range(len(w)):
            meta_rows.append(
                {
                    "domain": row["domain"],
                    "domain_id": DOMAIN_IDS[row["domain"]],
                    "role": row["role"],
                    "universal_label": row["universal_label"],
                    "condition": row["condition"],
                    "code": row["code"],
                    "source_file": Path(row["filepath"]).name,
                }
            )
    if unreadable:
        fraction = len(unreadable) / max(len(rows), 1)
        # Per-code breakdown: "60 files failed" is ambiguous between one dead
        # bearing code and scattered rot, and those need different responses.
        by_code: dict[str, int] = {}
        for path in unreadable:
            by_code[Path(path).parent.name] = by_code.get(Path(path).parent.name, 0) + 1
        per_code = ", ".join(f"{code}={n}" for code, n in sorted(by_code.items()))
        print(f"  {len(unreadable)}/{len(rows)} PU files unreadable ({fraction:.1%}) in this split -- by code: {per_code}")
        if fraction > MAX_UNREADABLE_FRACTION:
            raise RuntimeError(
                f"{len(unreadable)} of {len(rows)} PU .mat files unreadable ({fraction:.1%}, "
                f"limit {MAX_UNREADABLE_FRACTION:.0%}) -- that is systemic, not a few bad files. "
                f"First few: {unreadable[:5]}"
            )
    if not all_windows:
        return np.empty((0, config.common.segment_length), dtype=np.float64), pd.DataFrame(meta_rows)
    return np.concatenate(all_windows, axis=0), pd.DataFrame(meta_rows)


def _cache_paths(processed_dir: Path) -> dict[str, Path]:
    paths = {}
    for role in ROLES:
        paths[f"{role}_X"] = processed_dir / f"{role}_spectrograms.npy"
        paths[f"{role}_meta"] = processed_dir / f"{role}_meta.csv"
    return paths


def build_domain_adaptation_datasets(
    config: Config, pu_root: str | Path, force: bool = False
) -> dict[str, tuple[np.ndarray, pd.DataFrame]]:
    """Returns {role: (X, meta)} for role in ROLES.

    X: (N, F, T) float32 un-normalized log-magnitude spectrograms (see
    features/spectrogram.expected_shape via config.common/config.stft).
    meta: one row per window -- domain, domain_id, role, universal_label,
    condition, code, source_file.

    `pu_root` is passed explicitly (not read from config) because it's a
    runtime fact -- the Kaggle-mounted dataset path or a local dev
    directory -- see data/pu_loader.find_pu_root().
    """
    processed_dir = config.resolve(config.paths.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    paths = _cache_paths(processed_dir)

    if not force and all(p.exists() for p in paths.values()):
        return {role: (np.load(paths[f"{role}_X"]), pd.read_csv(paths[f"{role}_meta"])) for role in ROLES}

    include_cwru = config.cwru.include_in_source
    cwru_catalog = (
        build_cwru_catalog(config.resolve(config.paths.cwru_raw_dir), config.cwru.train_rpms, config.cwru.val_rpm)
        if include_cwru
        else None
    )
    if not include_cwru:
        print("CWRU excluded from the source pool (config.cwru.include_in_source=false) -- PU-only run.")
    pu_catalog = _assign_pu_roles(
        build_pu_catalog(pu_root, config.pu.source_codes + config.pu.real_codes, config.pu.native_fs_hz), config
    )

    # CWRU only ever contributes to source_train/source_val -- it isn't the
    # target domain, so target_adapt/target_test are PU-only below.
    cwru_split_for_role = {"source_train": "train", "source_val": "val"}
    empty_windows = np.empty((0, config.common.segment_length), dtype=np.float64)

    results: dict[str, tuple[np.ndarray, pd.DataFrame]] = {}
    for role in ROLES:
        stride = config.common.train_stride if role in ("source_train", "target_adapt") else config.common.eval_stride

        if include_cwru and role in cwru_split_for_role:
            cwru_rows = cwru_catalog[cwru_catalog.split == cwru_split_for_role[role]].copy()
            cwru_rows["role"] = role
            cwru_X, cwru_meta = _windows_and_meta_for_cwru_rows(cwru_rows, config, stride)
        else:
            cwru_X, cwru_meta = empty_windows, pd.DataFrame()

        pu_X, pu_meta = _windows_and_meta_for_pu_rows(pu_catalog[pu_catalog.role == role], config, stride)

        windows = np.concatenate([cwru_X, pu_X], axis=0)
        meta = pd.concat([cwru_meta, pu_meta], axis=0, ignore_index=True)
        if len(windows) == 0:
            raise RuntimeError(f"Role {role!r} produced zero windows -- check config.")

        spectrograms = windows_to_log_spectrograms(windows, config.common.fs_hz, config.stft)
        results[role] = (spectrograms, meta)

        np.save(paths[f"{role}_X"], spectrograms)
        meta.to_csv(paths[f"{role}_meta"], index=False)

    return results
