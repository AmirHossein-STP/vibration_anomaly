"""Convert the public ZeMA/UCI hydraulic files into the `hydraulic` service's format.

    python -m serving_core.tools.zema_to_cycles <raw_dir> <out_dir>

raw_dir holds the 17 sensor .txt files (PS1.txt ... SE.txt) and profile.txt.
out_dir receives cycle_00001.npz ... (one per 60 s cycle) and status.csv
(file,cooler,valve,pump,accumulator in time order) -- copy both into
<data>/hydraulic/train/ to train.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from serving_core.services.hydraulic_spec import GROUPS, SENSORS


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw_dir"); ap.add_argument("out_dir")
    args = ap.parse_args()
    raw, out = Path(args.raw_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    prof = np.loadtxt(raw / "profile.txt", dtype=int)
    if prof.ndim != 2 or prof.shape[1] < 4:
        raise SystemExit(f"{raw / 'profile.txt'}: expected one row per cycle with >= 4 columns")
    n = len(prof)
    data = {}
    for g, (samples, sensors) in GROUPS.items():
        for s in sensors:
            a = np.loadtxt(raw / f"{s}.txt", dtype=np.float32)
            if a.shape != (n, samples):
                raise SystemExit(f"{s}.txt: expected {(n, samples)}, got {a.shape}")
            data[s] = a
    with open(out / "status.csv", "w", newline="", encoding="utf8") as f:
        w = csv.writer(f); w.writerow(["file", "cooler", "valve", "pump", "accumulator"])
        for i in range(n):
            name = f"cycle_{i + 1:05d}.npz"
            np.savez(out / name, **{s: data[s][i] for s in SENSORS})
            w.writerow([name, *prof[i, :4]])
    print(f"wrote {n} cycles and status.csv to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
