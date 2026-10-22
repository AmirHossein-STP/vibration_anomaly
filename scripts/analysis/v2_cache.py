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
from domain_adaptation.bearings import lookup
from domain_adaptation.features.cot import cot_envelope_order_features
from domain_adaptation.features.envelope import FEATURE_NAMES, envelope_order_features
from domain_adaptation.features.geofree import BIN_NAMES, geofree_features, geofree_features_cot
from domain_adaptation.method import CPW_SES
from domain_adaptation.features.physics_v2 import V2_NAMES, v2_features, v2_features_cot
from envelope_cache import CWRU_RAW, PU_CODES, PU_STORE
from wp5_cache import COT_HOP_REVS, COT_REVS, COT_SPR, GEOMETRY, HOP, ROOTS, WIN, fingerprint as wp5_fp, time_domain, to_12k

DATASETS = ["cwru", "pu", "ottawa_2018", "ottawa_2023", "mfpt", "xjtu_sy", "hust", "paderborn_ldm"]
META = ["dataset", "unit", "condition", "recording", "label", "state", "damage_mode", "win", "t0", "rpm",
        "channel", "minute"]
SOURCES = [Path(p) for p in ("src/domain_adaptation/features/physics_v2.py", "src/domain_adaptation/features/cot.py",
                             "src/domain_adaptation/features/geofree.py", "scripts/analysis/v2_cache.py")]
# feature set -> (constant-speed fn, order-tracked fn, column names, default output dir)
FEATURE_SETS = {"v2": (v2_features, v2_features_cot, V2_NAMES, Path("outputs/wp3/cache")),
                "geofree": (geofree_features, geofree_features_cot, BIN_NAMES, Path("outputs/wp13/cache")),
                # method v1.0's own features (six lines among them), for the Phase 2 datasets
                "cpw": (lambda x, fs, rpm, g: envelope_order_features(x, fs, rpm, g, **CPW_SES),
                        lambda x, fs, ph, g, spr: cot_envelope_order_features(x, fs, ph, g, spr, **CPW_SES),
                        FEATURE_NAMES, Path("outputs/phase2/cache"))}


def fingerprint(dataset: str, fset: str = "v2") -> str:
    h = hashlib.sha256(b"".join(p.read_bytes() for p in SOURCES) + fset.encode())
    if dataset == "pu":
        h.update((PU_STORE / "manifest.csv").read_bytes())
    elif dataset == "cwru":
        h.update(b"".join(sorted(p.name.encode() for p in CWRU_RAW.glob("*.npz"))))
    else:
        h.update(wp5_fp(dataset, "v1").encode())
    return h.hexdigest()[:12]


def cache_path(out: Path, dataset: str, fset: str = "v2") -> Path:
    return out / f"{dataset}_{fset}_{fingerprint(dataset, fset)}.csv.gz"


def load_v2(dataset: str, out: Path | None = None, fset: str = "v2") -> pd.DataFrame:
    p = cache_path(out or FEATURE_SETS[fset][3], dataset, fset)
    if not p.is_file():
        raise FileNotFoundError(f"{p} missing or stale: run scripts/analysis/v2_cache.py on the box")
    return pd.read_csv(p)


def rows_of(rec: pdx.Recording, g, damage_mode: str = "", fset: str = "v2") -> list[dict]:
    const_fn, cot_fn, _, _ = FEATURE_SETS[fset]
    sig, fs = to_12k(rec.signal, rec.fs)
    base = {"dataset": rec.dataset, "unit": rec.unit, "condition": rec.condition, "recording": rec.recording,
            "label": rec.label, "state": rec.state, "damage_mode": damage_mode,
            "channel": rec.meta.get("channel", ""), "minute": rec.meta.get("minute", -1),
            **rec.meta.get("extra", {})}
    td = (lambda w: time_domain(w)) if fset == "cpw" else (lambda w: {})   # level stats ride with method features
    out = []
    if rec.phase_revs is None:
        for i, s in enumerate(range(0, len(sig) - WIN + 1, HOP)):
            out.append({**base, "win": i, "t0": round(s / fs, 4), "rpm": rec.rpm,
                        **const_fn(sig[s:s + WIN], fs, rec.rpm, g), **td(sig[s:s + WIN])})
        return out
    phase = np.interp(np.arange(len(sig)) / fs, np.arange(len(rec.phase_revs)) / rec.fs, rec.phase_revs)
    starts = np.searchsorted(phase, phase[0] + np.arange(0, phase[-1] - phase[0], COT_HOP_REVS))
    for i, a in enumerate(starts):
        b = np.searchsorted(phase, phase[a] + COT_REVS)
        if b >= len(sig):
            break
        out.append({**base, "win": i, "t0": round(a / fs, 4), "rpm": 60.0 * COT_REVS / ((b - a) / fs),
                    **cot_fn(sig[a:b], fs, phase[a:b], g, COT_SPR)})
    return out


def _job(args) -> list[dict]:
    dataset, key, fset = args
    if dataset == "cwru":
        path, rpm, label = key
        sig = np.asarray(np.load(path)["DE"]).squeeze().astype(np.float64)
        rec = pdx.Recording("cwru", Path(path).stem, str(rpm), Path(path).stem + "_0", label,
                            "healthy" if label == 0 else "faulty", 12000.0, sig, rpm=rpm)
        return rows_of(rec, CWRU_6205, "none" if label == 0 else "artificial", fset)
    if dataset == "pu":
        code, cond, run, rpm, label, mode = key
        sig = np.asarray(PreparedPUStore(PU_STORE).load(code, cond, run)).squeeze().astype(np.float64)
        rec = pdx.Recording("pu", code, cond, f"{code}_{run}", label, "healthy" if label == 0 else "faulty",
                            12000.0, sig, rpm=rpm)
        return rows_of(rec, PU_6203, mode, fset)
    if dataset == "paderborn_ldm":
        bearing, ch = key
        g = lookup("61806-2RS")
        # notes/25 §2a: records whose MEASURED speed is < 1000 rpm or not finite (rig stopped,
        # warm-up, interruptions) carry no valid order axis and are skipped
        return [r for rec in pdx.ldm_records(Path("data/paderborn_ldm"), bearing, ch)
                if np.isfinite(rec.rpm) and rec.rpm >= 1000 for r in rows_of(rec, g, "", fset)]
    if dataset == "hust":   # one geometry per bearing type, from the catalog
        return [r for rec in pdx.hust(Path("data/hust"), only=key)
                for r in rows_of(rec, lookup(rec.meta["bearing"]), "", fset)]
    g, root = GEOMETRY[dataset], ROOTS[dataset]
    if dataset == "xjtu_sy":
        bearing, ch = key
        return [r for rec in pdx.xjtu_records(root, bearing, ch) for r in rows_of(rec, g, "", fset)]
    loader = {"ottawa_2018": pdx.ottawa2018, "ottawa_2023": pdx.ottawa2023, "mfpt": pdx.mfpt}[dataset]
    return [r for rec in loader(root, only=key) for r in rows_of(rec, g, "", fset)]


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
    if dataset == "hust":
        return sorted(p.stem for p in Path("data/hust").glob("*.mat"))
    if dataset == "paderborn_ldm":     # channel A (primary) and C
        return [(b, ch) for b in pdx.ldm_bearings(Path("data/paderborn_ldm")) for ch in (0, 1)]
    root = ROOTS[dataset]
    return sorted(p.stem for p in (root.rglob("*.mat") if dataset == "mfpt" else root.glob("*.mat")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--set", default="v2", choices=list(FEATURE_SETS))
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    a.out = a.out or FEATURE_SETS[a.set][3]
    names = FEATURE_SETS[a.set][2]
    a.out.mkdir(parents=True, exist_ok=True)
    for d in a.datasets:
        p = cache_path(a.out, d, a.set)
        if p.is_file():
            print(f"[{d}] current: {p}"); continue
        t0 = time.time()
        rows = []
        with ProcessPoolExecutor(a.jobs) as ex:
            for part in ex.map(_job, [(d, k, a.set) for k in keys(d)], chunksize=4):
                rows += part
        df = pd.DataFrame(rows)
        df = df[[c for c in df.columns if c not in names] + names]     # meta (+ dataset extras) first
        df.to_csv(p, index=False, float_format="%.7g")
        print(f"[{d}] {len(rows)} windows in {time.time() - t0:.0f} s -> {p}", flush=True)


if __name__ == "__main__":
    main()
