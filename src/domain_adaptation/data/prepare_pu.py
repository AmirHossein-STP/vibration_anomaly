"""Condense the Paderborn dataset into a compact, transfer-friendly form.

Runs ON KAGGLE, where the 9.7 GB dataset is already mounted and compute is
free. Writes one `.npz` per bearing code holding every run's vibration
signal, resampled and stored as float32. The result is ~1 GB for the entire
dataset, which is what actually gets moved to the GPU server.

Why this exists rather than copying the raw dataset:

* **Seven channels in, one channel out.** Each `.mat` is ~8.7 MB carrying
  force, two phase currents, speed, temperature, torque and vibration. This
  project uses `vibration_1` alone -- everything else is bytes we would
  pay to transfer, store, and then never read.
* **64 kHz in, 25.6 kHz out.** The pipeline resamples PU down before doing
  anything with it, so shipping the original rate means paying to move
  samples that get discarded on arrival.
* **`scipy.io.loadmat` is slow.** Building the dataset from 1420 raw `.mat`
  files cost ~72 s of every Kaggle run. Reading pre-extracted arrays is a
  fast `np.load`, which matters once a hyperparameter sweep runs it dozens
  of times.
* **Rented compute is billed by the minute.** A 1 GB transfer instead of a
  9.7 GB one is the difference per server rebuild.

Deliberately converts the WHOLE dataset, not just the subset the current
config reads. The marginal cost is small (~1 GB vs ~0.7 GB) and it removes
any chance of a future config change -- a fourth source condition, more
bearing codes, the combined-damage KB* units -- forcing another download at
server rates.

25.6 kHz is a clean 2/5 step down from 64 kHz and sits above the 12 kHz the
cross-dataset pipeline currently uses (CWRU's DE12 files cap the *common*
rate at 12 kHz regardless), leaving headroom for PU-only and order-domain
experiments without a second trip.

Usage (on Kaggle):
    python -m domain_adaptation.data.prepare_pu --pu-root <mount> --out prepared_pu
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from domain_adaptation.data.pu_bearing_codes import PU_BEARING_CODES
from domain_adaptation.data.pu_loader import FILENAME_RE, UnreadableMatFile, load_vibration_signal
from domain_adaptation.data.transforms import resample_to_rate

NATIVE_FS_HZ = 64000
PREPARED_FS_HZ = 25600  # 64000 * 2/5 -- exact rational step, no resampling error
# Nominal shaft speed per PU operating-condition code, needed for order
# tracking (data/order_tracking.py). N15 = 1500 rpm, N09 = 900 rpm.
CONDITION_RPM = {"N15": 1500, "N09": 900}


def rpm_for_condition(condition: str) -> int:
    """`N09_M07_F10` -> 900. Speed is the first field; torque and radial
    force (M/F) do not change shaft rate."""
    prefix = condition.split("_")[0]
    if prefix not in CONDITION_RPM:
        raise KeyError(f"Unknown speed code {prefix!r} in condition {condition!r}")
    return CONDITION_RPM[prefix]


def prepare_code(code_dir: Path, out_dir: Path, fs_out: int = PREPARED_FS_HZ) -> dict:
    """Condense every run of one bearing code into a single .npz.

    Arrays are keyed `<condition>_<run>`; an unreadable source file is
    skipped and counted rather than aborting, the same posture
    data/dataset.py takes (one corrupt file in this Kaggle mirror,
    KA08/N15_M01_F10_KA08_2.mat, once killed an entire run).
    """
    code = code_dir.name
    arrays, rows, unreadable = {}, [], []
    for mat_path in sorted(code_dir.glob("*.mat")):
        m = FILENAME_RE.match(mat_path.name)
        if not m or m.group("code") != code:
            continue
        condition, run = m.group("condition"), int(m.group("run"))
        try:
            signal = load_vibration_signal(mat_path)
        except UnreadableMatFile as exc:
            unreadable.append(mat_path.name)
            print(f"    SKIP unreadable: {exc}")
            continue
        resampled = resample_to_rate(signal, NATIVE_FS_HZ, fs_out).astype(np.float32)
        arrays[f"{condition}_{run}"] = resampled
        rows.append(
            {
                "code": code, "condition": condition, "run": run,
                "rpm": rpm_for_condition(condition), "fs_hz": fs_out,
                "n_samples": len(resampled),
                "fault_type": PU_BEARING_CODES[code]["fault_type"],
                "damage_mode": PU_BEARING_CODES[code]["damage_mode"],
                "universal_label": PU_BEARING_CODES[code]["universal_label"],
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{code}.npz"
    np.savez_compressed(out_path, **arrays)
    mb = out_path.stat().st_size / 1024**2
    print(f"  {code}: {len(arrays)} runs -> {out_path.name} ({mb:.1f} MB)"
          + (f"  [{len(unreadable)} unreadable]" if unreadable else ""))
    return {"rows": rows, "unreadable": unreadable, "mb": mb}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pu-root", required=True, help="Mounted Paderborn dataset root.")
    parser.add_argument("--out", required=True, help="Directory for the condensed .npz files.")
    parser.add_argument("--fs", type=int, default=PREPARED_FS_HZ)
    parser.add_argument("--codes", nargs="*", default=None,
                        help="Bearing codes to convert (default: every code found).")
    args = parser.parse_args()

    root, out_dir = Path(args.pu_root), Path(args.out)
    codes = args.codes or sorted(
        d.name for d in root.iterdir() if d.is_dir() and re.fullmatch(r"K[ABI]?\d+", d.name)
    )
    print(f"Condensing {len(codes)} bearing codes from {root} at {args.fs} Hz\n")

    all_rows, all_unreadable, total_mb = [], [], 0.0
    for code in codes:
        result = prepare_code(root / code, out_dir, args.fs)
        all_rows += result["rows"]
        all_unreadable += result["unreadable"]
        total_mb += result["mb"]

    manifest = pd.DataFrame(all_rows)
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    print(f"\n{len(manifest)} runs across {manifest.code.nunique()} codes, {total_mb:.0f} MB total")
    print(f"manifest -> {out_dir / 'manifest.csv'}")
    if all_unreadable:
        print(f"{len(all_unreadable)} unreadable source files skipped: {all_unreadable[:5]}")


if __name__ == "__main__":
    main()
