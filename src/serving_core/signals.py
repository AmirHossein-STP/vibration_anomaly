"""Reading and validating 1-D vibration files (shared by the two vibration services)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

SIGNAL_SUFFIXES = (".npy", ".npz", ".csv")


class InputError(ValueError):
    """A problem with ONE input file. The batch continues; the file is reported."""


def load_signal(path: Path, key: str | None = None) -> np.ndarray:
    """.npy / .npz (named array, or the only array) / .csv (one column or one row)."""
    suffix = path.suffix.lower()
    try:
        if suffix == ".npy":
            arr = np.load(path, allow_pickle=False)
        elif suffix == ".npz":
            with np.load(path, allow_pickle=False) as z:
                names = list(z.keys())
                if key is None:
                    if len(names) != 1:
                        raise InputError(
                            f"holds arrays {names}; set `signal_key` in config/service.yaml or in a sidecar file"
                        )
                    key = names[0]
                if key not in z:
                    raise InputError(f"no array named {key!r} (has {names})")
                arr = z[key]
        elif suffix == ".csv":
            import pandas as pd

            arr = pd.read_csv(path, header=None).to_numpy(dtype=np.float64)
        else:
            raise InputError(f"unsupported file type {suffix!r} (use .npy, .npz or .csv)")
    except InputError:
        raise
    except Exception as e:  # corrupt file, parse error ...
        raise InputError(f"cannot read file: {e}") from e
    arr = np.asarray(arr).squeeze()
    if arr.ndim != 1:
        raise InputError(f"expected a single-channel 1-D signal, got shape {arr.shape}")
    arr = arr.astype(np.float64)
    if not np.isfinite(arr).all():
        bad = int((~np.isfinite(arr)).sum())
        raise InputError(f"{bad} NaN/inf samples -- fix the recording or cut the bad part out")
    if np.ptp(arr) == 0:
        raise InputError("signal is constant (flat line) -- sensor disconnected?")
    return arr


def read_sidecar(path: Path) -> dict:
    """Optional per-file overrides next to the data: `<file>.meta.json|yaml`
    (e.g. {"rpm": 1480, "fs_hz": 25600}). Missing = {}."""
    for ext in (".meta.json", ".meta.yaml", ".meta.yml"):
        side = path.with_name(path.name + ext)
        if side.is_file():
            try:
                text = side.read_text(encoding="utf8")
                data = json.loads(text) if ext.endswith("json") else yaml.safe_load(text)
            except Exception as e:
                raise InputError(f"cannot read sidecar {side.name}: {e}") from e
            if not isinstance(data, dict):
                raise InputError(f"sidecar {side.name} must hold a mapping")
            return data
    return {}


def list_signal_files(folder: Path) -> list[Path]:
    """Data files only: sidecars (*.meta.*) and hidden files are skipped."""
    if not folder.is_dir():
        return []
    out = []
    for p in sorted(folder.iterdir()):
        if (
            p.is_file()
            and p.suffix.lower() in SIGNAL_SUFFIXES
            and not p.name.startswith(".")
            and ".meta." not in p.name
        ):
            out.append(p)
    return out
