"""Parse CWRU .npz filenames into structured metadata.

CWRU_Bearing_NumPy naming convention (see data/README.md):
    {rpm}_Normal.npz
    {rpm}_{fault}_{size_mils}_{end}.npz

`fault` is one of IR (inner race), B (ball), OR@3 / OR@6 / OR@12 (outer
race, clock position relative to the load zone). `size_mils` is the fault
diameter in thousandths of an inch. `end` is DE12/DE48/FE; we only use
DE12 (12 kHz Drive-End) so it's constant across our raw files.

Note: these fault labels are ONLY ever used for evaluation in this project
(computing ROC-AUC, confusion matrices, per-severity detection rate). The
autoencoder itself is trained exclusively on windows from Normal.npz files
and never sees a label. See notes/04_evaluation_notes.md.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

NORMAL_RE = re.compile(r"^(?P<rpm>\d+)_Normal\.npz$")
FAULT_RE = re.compile(
    r"^(?P<rpm>\d+)_(?P<fault>IR|B|OR@3|OR@6|OR@12)_(?P<size>\d+)_(?P<end>DE\d+|FE)\.npz$"
)


def parse_filename(filename: str) -> dict:
    """Return metadata for one CWRU npz filename, or raise ValueError."""
    m = NORMAL_RE.match(filename)
    if m:
        return {
            "rpm": int(m.group("rpm")),
            "fault_type": "Normal",
            "fault_size_mils": 0,
            "end": None,
            "label": 0,
        }

    m = FAULT_RE.match(filename)
    if m:
        return {
            "rpm": int(m.group("rpm")),
            "fault_type": m.group("fault"),
            "fault_size_mils": int(m.group("size")),
            "end": m.group("end"),
            "label": 1,
        }

    raise ValueError(f"Filename doesn't match CWRU naming convention: {filename}")


def build_catalog(raw_dir: Path) -> pd.DataFrame:
    """Scan raw_dir for .npz files and return one metadata row per file."""
    rows = []
    for path in sorted(raw_dir.glob("*.npz")):
        meta = parse_filename(path.name)
        meta["filename"] = path.name
        meta["filepath"] = str(path)
        rows.append(meta)

    if not rows:
        raise FileNotFoundError(
            f"No .npz files found in {raw_dir}. Run "
            "`python -m vibration_ad.data.download` first."
        )

    df = pd.DataFrame(rows)
    cols = ["filename", "filepath", "rpm", "fault_type", "fault_size_mils", "end", "label"]
    return df[cols].sort_values(["rpm", "fault_type", "fault_size_mils"]).reset_index(drop=True)


def save_catalog(df: pd.DataFrame, processed_dir: Path) -> Path:
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "catalog.csv"
    df.to_csv(out_path, index=False)
    return out_path
