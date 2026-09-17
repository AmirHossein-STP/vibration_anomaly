"""Parse the Paderborn University (PU / KAt-DataCenter) bearing dataset.

Layout (confirmed 2026-09-10 by downloading 3 real files off the attached
Kaggle mirror, `dippatel03/paderborn-db` -- see DOMAIN_ADAPTATION_PLAN.md's
"Revisions" section):

    <root>/<bearing_code>/<condition>_<bearing_code>_<run>.mat

e.g. `K001/N09_M07_F10_K001_1.mat`. `condition` encodes rotational speed
(N09=900rpm, N15=1500rpm), load torque (M01=0.1Nm, M07=0.7Nm) and radial
force (F04=400N, F10=1000N) -- see pu_bearing_codes.py for what each
`bearing_code` means (healthy/artificial/real, inner/outer race).

Each `.mat` file is a MATLAB struct (top-level key = the filename with the
extension stripped) with a `Y` field: a length-7 list of per-channel
dicts, each `{'Name': str, 'Data': 1D array}`. Channels and their
approximate native rates (from the same inspection): force/speed/torque
@ ~4kHz, phase_current_1/2 @ ~64kHz, temp_2_bearing_module @ ~1.25Hz (5
samples per 4s run), vibration_1 @ ~64kHz -- only `vibration_1` is used
here, matching CWRU's single-accelerometer-channel scope. Raw run lengths
vary by a few hundred samples around the nominal 4s/256000 samples (a DAQ
quirk, not a bug) -- nothing downstream assumes an exact length.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io

from domain_adaptation.data.pu_bearing_codes import PU_BEARING_CODES

FILENAME_RE = re.compile(r"^(?P<condition>N\d{2}_M\d{2}_F\d{2})_(?P<code>[A-Z]+\d+)_(?P<run>\d+)\.mat$")


def find_pu_root(local_hint: str | Path | None = None) -> Path:
    """Locate the PU dataset's root directory (the folder that directly
    contains K001/, K002/, ... subfolders).

    On Kaggle, searches /kaggle/input for a folder containing a K001/
    subfolder -- by content, not a fixed path, because attached-dataset
    mount paths have proven inconsistent in this project before (see
    vibration_ad's notes/07_dvc_and_kaggle.md). Falls back to
    `local_hint` (configs/domain_adaptation.yaml's paths.pu_root_local) for
    local development against a small hand-built sample.
    """
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        candidates = sorted(p.parent for p in kaggle_input.rglob("K001") if p.is_dir())
        if candidates:
            return candidates[0]
        raise FileNotFoundError(
            f"No K001/ folder found under {kaggle_input} -- is the PU dataset "
            "attached? (see notebooks/domain_adaptation/kernel-metadata.json's dataset_sources)"
        )
    if local_hint is not None and Path(local_hint).is_dir():
        return Path(local_hint)
    raise FileNotFoundError(
        "Not running on Kaggle (/kaggle/input missing) and no usable local "
        f"pu_root ({local_hint!r}) -- pass --pu-root explicitly."
    )


class UnreadableMatFile(RuntimeError):
    """A PU .mat file scipy cannot parse. Raised instead of letting scipy's
    bare `TypeError: Expecting matrix here` escape with no indication of
    WHICH of ~1400 files was at fault -- which is exactly how kernel v5
    died after 3 minutes with an unattributable traceback.
    """


def load_vibration_signal(mat_path: str | Path, channel: str = "vibration_1") -> np.ndarray:
    """Extract one channel's 1D array from a PU .mat file."""
    try:
        data = scipy.io.loadmat(str(mat_path), simplify_cells=True)
    except Exception as exc:  # scipy raises bare TypeError/ValueError with no file context
        raise UnreadableMatFile(f"{mat_path}: scipy could not parse this .mat ({type(exc).__name__}: {exc})") from exc
    key = next(k for k in data if not k.startswith("__"))
    channels = data[key]["Y"]
    for ch in channels:
        if ch["Name"] == channel:
            return np.asarray(ch["Data"], dtype=np.float64).ravel()
    available = [ch["Name"] for ch in channels]
    raise KeyError(f"{mat_path}: channel {channel!r} not found (has: {available})")


def build_pu_catalog(root_dir: str | Path, codes: list[str], native_fs_hz: int = 64000) -> pd.DataFrame:
    """One row per `<code>/<condition>_<code>_<run>.mat` file found under
    `root_dir`, for every `code` in `codes` (skips the per-code .pdf docs
    and anything that doesn't match the naming convention or has no
    universal-label mapping, e.g. combined-damage codes -- see
    pu_bearing_codes.py). Row order is deterministic (sorted glob), not
    disk order.

    `native_fs_hz` is a parameter (not hardcoded) so tests can exercise
    this against small synthetic signals at a lower rate -- production
    callers pass config.pu.native_fs_hz (64000 for the real dataset).

    Returns columns: filepath, code, condition, run, fault_type,
    damage_mode, universal_label, fs_native_hz, domain.
    """
    root_dir = Path(root_dir)
    rows = []
    for code in codes:
        code_dir = root_dir / code
        if not code_dir.is_dir():
            raise FileNotFoundError(f"PU bearing code folder not found: {code_dir}")
        meta = PU_BEARING_CODES.get(code)
        if meta is None:
            raise KeyError(f"Unknown PU bearing code {code!r} -- not in pu_bearing_codes.PU_BEARING_CODES")
        if meta["universal_label"] is None:
            continue  # combined-damage code, excluded from this task (see module docstring)

        for mat_path in sorted(code_dir.glob("*.mat")):
            m = FILENAME_RE.match(mat_path.name)
            if not m or m.group("code") != code:
                continue
            rows.append(
                {
                    "filepath": str(mat_path),
                    "code": code,
                    "condition": m.group("condition"),
                    "run": int(m.group("run")),
                    "fault_type": meta["fault_type"],
                    "damage_mode": meta["damage_mode"],
                    "universal_label": meta["universal_label"],
                    "fs_native_hz": native_fs_hz,
                    "domain": "pu",
                }
            )
    if not rows:
        raise FileNotFoundError(f"No matching PU .mat files found under {root_dir} for codes {codes}")
    return pd.DataFrame(rows)

# --- condensed ("prepared") store ------------------------------------------
# prepare_pu.py turns the 9.7 GB raw dataset into ~1 GB of per-code .npz
# archives. Everything below reads that form. Detection is automatic, so the
# same --pu-root works whether it points at raw .mat folders or a condensed
# store -- no config flag to forget to flip, and no second code path through
# the rest of the pipeline.


def is_prepared_store(root: str | Path) -> bool:
    """A condensed store is identified by its manifest; raw PU has none."""
    return (Path(root) / "manifest.csv").is_file()


class PreparedPUStore:
    """Reads the per-code archives written by prepare_pu.py.

    numpy loads .npz members lazily, so holding one open handle per bearing
    code and indexing into it avoids re-reading a ~33 MB archive for every
    one of its 80 runs.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        manifest_path = self.root / "manifest.csv"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"{self.root} is not a prepared store (no manifest.csv)")
        self.manifest = pd.read_csv(manifest_path)
        self._handles: dict[str, object] = {}

    def _handle(self, code: str):
        if code not in self._handles:
            path = self.root / f"{code}.npz"
            if not path.is_file():
                raise FileNotFoundError(f"{path} missing -- manifest lists {code} but the archive is absent")
            self._handles[code] = np.load(path)
        return self._handles[code]

    def load(self, code: str, condition: str, run: int) -> np.ndarray:
        key = f"{condition}_{run}"
        handle = self._handle(code)
        if key not in handle:
            raise KeyError(f"{code}.npz has no run {key!r}")
        return np.asarray(handle[key], dtype=np.float64)

    def catalog(self, codes: list[str]) -> pd.DataFrame:
        """Same columns build_pu_catalog returns, so downstream code is
        identical for either storage form."""
        wanted = self.manifest[self.manifest.code.isin(codes)].copy()
        missing = sorted(set(codes) - set(wanted.code.unique()))
        if missing:
            raise FileNotFoundError(f"Prepared store {self.root} has no data for codes {missing}")
        wanted = wanted[wanted.universal_label.notna()]
        wanted["universal_label"] = wanted.universal_label.astype(int)
        wanted["filepath"] = ""          # signals live inside the archives, not as files
        wanted["fs_native_hz"] = wanted["fs_hz"]
        wanted["domain"] = "pu"
        cols = ["filepath", "code", "condition", "run", "fault_type",
                "damage_mode", "universal_label", "fs_native_hz", "domain"]
        return wanted[cols].reset_index(drop=True)
