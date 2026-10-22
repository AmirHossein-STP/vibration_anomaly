"""Physics features v2 (notes/19) for every development dataset, on the same
windows as envelope_cache.py (CWRU, Paderborn) and wp5_cache.py (the rest).

Usage (box): python scripts/analysis/v2_cache.py --jobs 8
Output: outputs/wp3/cache/<dataset>_v2_<fingerprint>.csv.gz
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))
from domain_adaptation.data import public_datasets as pdx
from domain_adaptation.data.cwru_loader import build_cwru_catalog
from domain_adaptation.data.pu_loader import PreparedPUStore
from domain_adaptation.features.envelope import CWRU_6205, PU_6203
from domain_adaptation.features.physics_v2 import V2_NAMES, v2_features, v2_features_cot
from envelope_cache import CWRU_RAW, PU_CODES, PU_STORE
from wp5_cache import COT_HOP_REVS, COT_REVS, COT_SPR, GEOMETRY, HOP, ROOTS, WIN, fingerprint as wp5_fp, to_12k

DATASETS = ["cwru", "pu", "ottawa_2018", "ottawa_2023", "mfpt", "xjtu_sy"]
META = ["dataset", "unit", "condition", "recording", "label", "state", "damage_mode", "win", "t0", "rpm",
        "channel", "minute"]
SOURCES = [Path(p) for p in ("src/domain_adaptation/features/physics_v2.py", "src/domain_adaptation/features/cot.py",
                             "scripts/analysis/v2_cache.py")]


def fingerprint(dataset: str) -> str:
    h = hashlib.sha256(b"".join(p.read_bytes() for p in SOURCES))
    if dataset == "pu":
        h.update((PU_STORE / "manifest.csv").read_bytes())
    elif dataset == "cwru":
        h.update(b"".join(sorted(p.name.encode() for p in CWRU_RAW.glob("*.npz"))))
    else:
        h.update(wp5_fp(dataset, "v1").encode())
    return h.hexdigest()[:12]


def cache_path(out: Path, dataset: str) -> Path:
    return out / f"{dataset}_v2_{fingerprint(dataset)}.csv.gz"


def load_v2(dataset: str, out: Path = Path("outputs/wp3/cache")) -> pd.DataFrame:
    p = cache_path(out, dataset)
    if not p.is_file():
        raise FileNotFoundError(f"{p} missing or stale: run scripts/analysis/v2_cache.py on the box")
    return pd.read_csv(p)


def windows_const(sig, fs, rpm, g):
    for i, s in enumerate(range(0, len(sig) - WIN + 1, HOP)):
        yield i, s / fs, rpm, v2_features(sig[s:s + WIN], fs, rpm, g)


def rows_of(rec: pdx.Recording, g, damage_mode: str = "") -> list[dict]:
    sig, fs = to_12k(rec.signal, rec.fs)
    base = {"dataset": rec.dataset, "unit": rec.unit, "condition": rec.condition, "recording": rec.recording,
            "label": rec.label, "state": rec.state, "damage_mode": damage_mode,
            "channel": rec.meta.get("channel", ""), "minute": rec.meta.get("minute", -1)}
    out = []
    if rec.phase_revs is None:
        for i, t0, rpm, f in windows_const(sig, fs, rec.rpm, g):
            out.append({**base, "win": i, "t0": round(t0, 4), "rpm": rpm, **f})
        return out
    phase = np.interp(np.arange(len(sig)) / fs, np.arange(len(rec.phase_revs)) / rec.fs, rec.phase_revs)
    starts = np.searchsorted(phase, phase[0] + np.arange(0, phase[-1] - phase[0], COT_HOP_REVS))
    for i, a in enumerate(starts):
        b = np.searchsorted(phase, phase[a] + COT_REVS)
        if b >= len(sig):
            break
        out.append({**base, "win": i, "t0": round(a / fs, 4), "rpm": 60.0 * COT_REVS / ((b - a) / fs),
                    **v2_features_cot(sig[a:b], fs, phase[a:b], g, COT_SPR)})
    return out


def _job(args) -> list[dict]:
    dataset, key = args
    if dataset == "cwru":
        path, rpm, label = key
        sig = np.asarray(np.load(path)["DE"]).squeeze().astype(np.float64)
        rec = pdx.Recording("cwru", Path(path).stem, str(rpm), Path(path).stem + "_0", label,
                            "healthy" if label == 0 else "faulty", 12000.0, sig, rpm=rpm)
        return rows_of(rec, CWRU_6205, "none" if label == 0 else "artificial")
    if dataset == "pu":
        code, cond, run, rpm, label, mode = key
        sig = np.asarray(PreparedPUStore(PU_STORE).load(code, cond, run)).squeeze().astype(np.float64)
        rec = pdx.Recording("pu", code, cond, f"{code}_{run}", label, "healthy" if label == 0 else "faulty",
                            12000.0, sig, rpm=rpm)
        return rows_of(rec, PU_6203, mode)
    g, root = GEOMETRY[dataset], ROOTS[dataset]
    if dataset == "xjtu_sy":
        bearing, ch = key
        return [r for rec in pdx.xjtu_records(root, bearing, ch) for r in rows_of(rec, g)]
    loader = {"ottawa_2018": pdx.ottawa2018, "ottawa_2023": pdx.ottawa2023, "mfpt": pdx.mfpt}[dataset]
    return [r for rec in loader(root, only=key) for r in rows_of(rec, g)]


def keys(dataset: str) -> list:
    if dataset == "cwru":
        cat = build_cwru_catalog(CWRU_RAW, [1730, 1750, 1772], 1797)
        return [(r.filepath, int(r.rpm), int(r.universal_label)) for r in cat.itertuples()]
    if dataset == "pu":
        pcat = PreparedPUStore(PU_STORE).catalog(PU_CODES)
        return [(r.code, r.condition, int(r.run), int(r.rpm), int(r.universal_label), r.damage_mode)
                for r in pcat.itertuples()]
    if dataset == "xjtu_sy":
        return [(b, ch) for b in pdx.xjtu_bearings(ROOTS[dataset]).bearing for ch in (0, 1)]
    root = ROOTS[dataset]
    return sorted(p.stem for p in (root.rglob("*.mat") if dataset == "mfpt" else root.glob("*.mat")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("outputs/wp3/cache"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for d in a.datasets:
        p = cache_path(a.out, d)
        if p.is_file():
            print(f"[{d}] current: {p}"); continue
        t0 = time.time()
        rows = []
        with ProcessPoolExecutor(a.jobs) as ex:
            for part in ex.map(_job, [(d, k) for k in keys(d)], chunksize=4):
                rows += part
        pd.DataFrame(rows)[META + V2_NAMES].to_csv(p, index=False, float_format="%.7g")
        print(f"[{d}] {len(rows)} windows in {time.time() - t0:.0f} s -> {p}", flush=True)


if __name__ == "__main__":
    main()
