"""Phase 3 A0: geometry-free features for IMS, now development data (IMS is spent: notes/27).

Reads the per-record .npy files that WP18 converted for the service (test 1: x channel of each
bearing pair), in time order, and runs the frozen `v2_cache.rows_of` on them. A separate script
so that v2_cache.py, and with it every existing cache fingerprint, stays untouched.

Usage (box, after wp18_ims.py has built ~/work/scratch/ims_service):
    python3 scripts/analysis/p3_ims_cache.py --jobs 8
Output: outputs/p3/cache/ims_geofree.csv.gz
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))
from domain_adaptation.bearings import lookup
from domain_adaptation.data import public_datasets as pdx
from domain_adaptation.features.geofree import BIN_NAMES
from v2_cache import rows_of
from wp18_ims import FAILED, SERVICE

DATA = SERVICE / "geometry_free" / "data" / "bearing"
OUT = Path("outputs/p3/cache/ims_geofree.csv.gz")


def stamp_time(name: str) -> datetime:
    return datetime.strptime(name.removesuffix(".npy"), "%Y.%m.%d.%H.%M.%S")


def _job(unit: str) -> list[dict]:
    t, b = (int(x[1:]) for x in unit.split("_"))
    files = sorted((DATA / "train" / unit).glob("*.npy")) + sorted((DATA / "in" / unit).glob("*.npy"))
    t_start = stamp_time(files[0].name)
    g = lookup("Rexnord ZA-2115")
    rows = []
    for k, f in enumerate(files):
        hours = (stamp_time(f.name) - t_start).total_seconds() / 3600
        rec = pdx.Recording("ims", unit, f"test{t}", f.stem, -1, "run", 20000.0, np.load(f), rpm=2000.0,
                            meta={"minute": k + 1, "extra": {"record": k, "hours": round(hours, 4),
                                                              "n_records": len(files),
                                                              "failed": FAILED.get((t, b), "")}})
        rows += rows_of(rec, g, "", "geofree")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    units = sorted(p.name for p in (DATA / "train").iterdir())
    if len(units) != 12:
        raise SystemExit(f"expected 12 IMS units in {DATA / 'train'}, found {units}")
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(a.jobs) as ex:
        for part in ex.map(_job, units):
            rows += part
    df = pd.DataFrame(rows)
    df = df[[c for c in df.columns if c not in BIN_NAMES] + BIN_NAMES]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, float_format="%.7g")
    print(f"[ims] {len(df)} windows, {df.unit.nunique()} units in {time.time() - t0:.0f} s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
