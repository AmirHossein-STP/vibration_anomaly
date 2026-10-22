"""Time-domain window statistics (RMS, kurtosis, crest) for CWRU and
Paderborn, on exactly envelope_cache.py's windows (12 kHz, 1 s, hop 0.5 s),
so they can be joined to its feature rows (notes/18 §6.2).

Usage: python scripts/analysis/td_cache.py [--jobs 8]
Output: outputs/wp5/cache/td_cwru_pu_<fingerprint>.csv.gz
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))
from domain_adaptation.data.cwru_loader import build_cwru_catalog
from domain_adaptation.data.pu_loader import PreparedPUStore
from envelope_cache import CWRU_RAW, FS, HOP, PU_CODES, PU_STORE, WIN, fingerprint as env_fp
from wp5_cache import time_domain

KEYS = ["dataset", "unit", "condition", "run", "win"]


def path(out: Path) -> Path:
    cat = build_cwru_catalog(CWRU_RAW, [1730, 1750, 1772], 1797)
    fp = hashlib.sha256((env_fp([Path(p) for p in cat.filepath]) + Path(__file__).read_text()
                         + Path("scripts/analysis/wp5_cache.py").read_text()).encode()).hexdigest()[:12]
    return out / f"td_cwru_pu_{fp}.csv.gz"


def load_td(out: Path = Path("outputs/wp5/cache")) -> pd.DataFrame:
    p = path(out)
    if not p.is_file():
        raise FileNotFoundError(f"{p} missing: run scripts/analysis/td_cache.py on the box")
    return pd.read_csv(p)


def _windows(sig):
    for i, s in enumerate(range(0, len(sig) - WIN + 1, HOP)):
        yield i, sig[s:s + WIN]


def _cwru(args):
    p, rpm = args
    sig = np.asarray(np.load(p)["DE"]).squeeze().astype(np.float64)
    return [{"dataset": "cwru", "unit": Path(p).stem, "condition": str(rpm), "run": 0, "win": i, **time_domain(w)}
            for i, w in _windows(sig)]


def _pu(args):
    code, cond, run = args
    sig = np.asarray(PreparedPUStore(PU_STORE).load(code, cond, run)).squeeze().astype(np.float64)
    return [{"dataset": "pu", "unit": code, "condition": cond, "run": int(run), "win": i, **time_domain(w)}
            for i, w in _windows(sig)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("outputs/wp5/cache"))
    a = ap.parse_args()
    p = path(a.out)
    if p.is_file():
        print(f"current: {p}"); return
    cat = build_cwru_catalog(CWRU_RAW, [1730, 1750, 1772], 1797)
    pcat = PreparedPUStore(PU_STORE).catalog(PU_CODES)
    rows = []
    with ProcessPoolExecutor(a.jobs) as ex:
        for part in ex.map(_cwru, [(r.filepath, int(r.rpm)) for r in cat.itertuples()]):
            rows += part
        for part in ex.map(_pu, [(r.code, r.condition, int(r.run)) for r in pcat.itertuples()], chunksize=16):
            rows += part
    a.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(p, index=False, float_format="%.7g")
    print(f"wrote {len(rows)} windows -> {p}")


if __name__ == "__main__":
    main()
