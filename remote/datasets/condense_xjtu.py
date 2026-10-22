"""Condense XJTU-SY (6-part .rar of ~9,200 CSV files) into one float32 .npy
per bearing. Called by fetch.py on the server; the 4.4 GB raw archive stays
on the ephemeral SSD.

Raw layout: <condition>/<BearingX_Y>/<minute>.csv, where the condition
directory is e.g. 35Hz12kN, and each CSV holds 32768 rows (1.28 s at
25.6 kHz) of horizontal and vertical acceleration, one file per minute of
the run-to-failure test.

Output:
    <condition>/<bearing>.npy   shape (n_minutes, 32768, 2) float32, ordered by
                                minute; [..., 0] horizontal, [..., 1] vertical.
                                Load with np.load(path, mmap_mode="r").
    manifest.csv                one row per bearing: condition, rpm, load_kN,
                                n_minutes, samples_per_file, fs_hz, file

Usage (fetch.py does this):
    python remote/datasets/condense_xjtu.py --raw-dir <dir of .partNN.rar> --out <dir> --work <scratch dir>
"""
from __future__ import annotations

import argparse
import re
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

COND_RE = re.compile(r"^(?P<hz>\d+(?:\.\d+)?)Hz(?P<kn>\d+)kN$")
FS_HZ = 25600


def condense_bearing(bearing_dir: str, out_path: str) -> dict:
    import numpy as np
    import pandas as pd
    bdir, out = Path(bearing_dir), Path(out_path)
    files = sorted(bdir.glob("*.csv"), key=lambda p: int(p.stem))
    if [int(p.stem) for p in files] != list(range(1, len(files) + 1)):
        raise ValueError(f"{bdir}: minute files are not 1..{len(files)}")
    arrays = []
    for p in files:
        df = pd.read_csv(p)
        if df.shape[1] != 2:
            raise ValueError(f"{p}: expected 2 columns, got {list(df.columns)}")
        arrays.append(df.to_numpy(dtype=np.float32))
    lengths = {a.shape[0] for a in arrays}
    if len(lengths) != 1:
        raise ValueError(f"{bdir}: files have different lengths {sorted(lengths)}")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, np.stack(arrays))
    cond = COND_RE.match(bdir.parent.name)
    print(f"  {bdir.parent.name}/{bdir.name}: {len(files)} minutes", flush=True)
    return {"condition": bdir.parent.name, "rpm": round(float(cond["hz"]) * 60), "load_kN": int(cond["kn"]),
            "bearing": bdir.name, "n_minutes": len(files), "samples_per_file": lengths.pop(),
            "fs_hz": FS_HZ, "columns": "horizontal,vertical",
            "file": f"{bdir.parent.name}/{bdir.name}.npy"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--src", help="unused; accepted for a uniform condenser interface")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()

    import pandas as pd
    from safe_extract import extract_rar

    first = sorted(args.raw_dir.glob("*.part01.rar")) or sorted(args.raw_dir.glob("*.part1.rar"))
    if len(first) != 1:
        raise SystemExit(f"expected one multi-part set under {args.raw_dir}, found {first}")
    extracted = args.work / "extract"
    if extracted.exists():
        shutil.rmtree(extracted)
    print(f"extracting {first[0].name} (+ parts)", flush=True)
    extract_rar(first[0], extracted)

    bearings = sorted(d for d in extracted.rglob("Bearing*") if d.is_dir() and COND_RE.match(d.parent.name))
    if len(bearings) != 15:
        print(f"WARNING: expected 15 bearings, found {len(bearings)}", flush=True)
    outs = [str(args.out / b.parent.name / f"{b.name}.npy") for b in bearings]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        rows = list(pool.map(condense_bearing, [str(b) for b in bearings], outs))
    pd.DataFrame(rows).to_csv(args.out / "manifest.csv", index=False)
    shutil.rmtree(extracted, ignore_errors=True)
    print(f"{len(rows)} bearings, {sum(r['n_minutes'] for r in rows)} minutes -> {args.out}")


if __name__ == "__main__":
    main()
