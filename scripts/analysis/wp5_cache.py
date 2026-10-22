"""Window features for the WP5 datasets (notes/17), to fingerprinted caches.

Exactly the frozen pipeline of envelope_cache.py, so a new dataset is
scored with the same features as Paderborn and CWRU:
- resample to 12 kHz (effective rate kept where the ratio is not exact);
- 1 s windows, hop 0.5 s;
- `v1` and `hd_cpw_ses` feature variants;
- features read at each dataset's own bearing geometry.

Ottawa 2018 (time-varying speed) uses computed order tracking
(features/cot.py) over 20-revolution windows, hop 10 (notes/17 §2).

This is the heavy step. Run it on the box:
    remote/box run wp5cache -- python3 scripts/analysis/wp5_cache.py --jobs 8
Output: outputs/wp5/cache/<dataset>_<variant>_<fingerprint>.csv.gz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from domain_adaptation.data import public_datasets as pdx
from domain_adaptation.data.transforms import resample_to_rate
from domain_adaptation.features.cot import cot_envelope_order_features
from domain_adaptation.features.envelope import FEATURE_NAMES, envelope_order_features

FS = 12000
WIN, HOP = 12000, 6000
COT_REVS, COT_HOP_REVS, COT_SPR = 20, 10, 128
VARIANTS = {"v1": {}, "hd_cpw_ses": {"whiten": True, "band_hz": None, "squared": True}}
ROOTS = {"ottawa_2018": Path("data/ottawa_2018"), "ottawa_2023": Path("data/ottawa_2023"),
         "mfpt": Path("data/mfpt"), "xjtu_sy": Path("data/xjtu_sy")}
GEOMETRY = {"ottawa_2018": pdx.OTTAWA18_ER16K, "ottawa_2023": pdx.OTTAWA23_6203,
            "mfpt": pdx.MFPT_BEARING, "xjtu_sy": pdx.XJTU_UER204}
META = ["dataset", "unit", "condition", "recording", "label", "state", "win", "t0", "rpm", "channel", "minute"]
# time-domain level/impulsiveness of the raw 12 kHz window (notes/18 §6; WP3's time-domain statistics)
TD = ["td_rms", "td_kurtosis", "td_crest"]
SOURCES = [Path(p) for p in ("src/domain_adaptation/features/envelope.py", "src/domain_adaptation/features/cot.py",
                             "src/domain_adaptation/data/public_datasets.py", "scripts/analysis/wp5_cache.py")]


def fingerprint(dataset: str, variant: str) -> str:
    h = hashlib.sha256()
    for p in SOURCES:
        h.update(p.read_bytes())
    h.update(json.dumps({"variant": VARIANTS[variant], "fs": FS, "win": WIN, "hop": HOP,
                         "cot": [COT_REVS, COT_HOP_REVS, COT_SPR]}, sort_keys=True).encode())
    manifest = Path("remote/datasets/manifests") / f"{dataset}.json"
    m = json.loads(manifest.read_text())
    h.update(json.dumps({k: v["sha256"] for k, v in {**m["raw"], **m["condensed"]}.items()}, sort_keys=True).encode())
    return h.hexdigest()[:12]


def cache_path(out: Path, dataset: str, variant: str) -> Path:
    return out / f"{dataset}_{variant}_{fingerprint(dataset, variant)}.csv.gz"


def load_wp5(dataset: str, variant: str, out: Path = Path("outputs/wp5/cache")) -> pd.DataFrame:
    p = cache_path(out, dataset, variant)
    if not p.is_file():
        raise FileNotFoundError(f"{p} missing or stale: run scripts/analysis/wp5_cache.py on the box")
    return pd.read_csv(p)


def to_12k(sig: np.ndarray, fs: float) -> tuple[np.ndarray, float]:
    ratio = Fraction(FS, int(round(fs))).limit_denominator(1000)
    return resample_to_rate(sig, int(round(fs)), FS), fs * ratio.numerator / ratio.denominator


def time_domain(w: np.ndarray) -> dict:
    x = w - w.mean()
    rms = float(np.sqrt((x ** 2).mean())) + 1e-12
    return {"td_rms": rms, "td_kurtosis": float((x ** 4).mean() / rms ** 4), "td_crest": float(np.abs(x).max() / rms)}


def _row(rec: pdx.Recording, win: int, t0: float, rpm: float, feats: dict) -> dict:
    return {"dataset": rec.dataset, "unit": rec.unit, "condition": rec.condition, "recording": rec.recording,
            "label": rec.label, "state": rec.state, "win": win, "t0": round(t0, 4), "rpm": rpm,
            "channel": rec.meta.get("channel", ""), "minute": rec.meta.get("minute", -1), **feats}


def features_of(rec: pdx.Recording, geometry, kw: dict) -> list[dict]:
    sig, fs = to_12k(rec.signal, rec.fs)
    rows = []
    if rec.phase_revs is None:
        for i, s in enumerate(range(0, len(sig) - WIN + 1, HOP)):
            rows.append(_row(rec, i, s / fs, rec.rpm,
                             {**envelope_order_features(sig[s:s + WIN], fs, rec.rpm, geometry, **kw),
                              **time_domain(sig[s:s + WIN])}))
        return rows
    # time-varying speed: phase at the 12 kHz sample times, windows of fixed revolutions
    t_old = np.arange(len(rec.phase_revs)) / rec.fs
    phase = np.interp(np.arange(len(sig)) / fs, t_old, rec.phase_revs)
    starts = np.searchsorted(phase, phase[0] + np.arange(0, phase[-1] - phase[0], COT_HOP_REVS))
    for i in range(len(starts)):
        a = starts[i]
        b = np.searchsorted(phase, phase[a] + COT_REVS)
        if b >= len(sig):
            break
        rpm = 60.0 * COT_REVS / ((b - a) / fs)
        rows.append(_row(rec, i, a / fs, rpm,
                         {**cot_envelope_order_features(sig[a:b], fs, phase[a:b], geometry, COT_SPR, **kw),
                          **time_domain(sig[a:b])}))
    return rows


def _job(args) -> list[dict]:
    dataset, key, variant = args
    kw, geom, root = VARIANTS[variant], GEOMETRY[dataset], ROOTS[dataset]
    rows = []
    if dataset == "xjtu_sy":
        bearing, channel = key
        for rec in pdx.xjtu_records(root, bearing, channel):
            rows += features_of(rec, geom, kw)
        return rows
    loader = {"ottawa_2018": pdx.ottawa2018, "ottawa_2023": pdx.ottawa2023, "mfpt": pdx.mfpt}[dataset]
    for rec in loader(root, only=key):
        rows += features_of(rec, geom, kw)
    return rows


def keys_of(dataset: str) -> list:
    root = ROOTS[dataset]
    if dataset == "xjtu_sy":
        return [(b, ch) for b in pdx.xjtu_bearings(root).bearing for ch in (0, 1)]
    return sorted(p.stem for p in (root.rglob("*.mat") if dataset == "mfpt" else root.glob("*.mat")))


def build(dataset: str, variant: str, out: Path, jobs: int) -> None:
    p = cache_path(out, dataset, variant)
    if p.is_file():
        print(f"[{dataset}/{variant}] current: {p}")
        return
    keys = keys_of(dataset)
    t0 = time.time()
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for part in ex.map(_job, [(dataset, k, variant) for k in keys]):
            rows += part
    df = pd.DataFrame(rows)[META + FEATURE_NAMES + TD]
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False, float_format="%.7g")
    p.with_suffix("").with_suffix(".json").write_text(json.dumps({
        "dataset": dataset, "variant": variant, "n_windows": len(df), "n_recordings": int(df.recording.nunique()),
        "seconds": round(time.time() - t0, 1), "by_label": df.groupby("label").size().to_dict().__repr__()}, indent=1))
    print(f"[{dataset}/{variant}] {len(df)} windows from {df[['unit', 'recording']].drop_duplicates().shape[0]} "
          f"recordings in {time.time() - t0:.0f} s -> {p}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(ROOTS))
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("outputs/wp5/cache"))
    args = ap.parse_args()
    for d in args.datasets:
        for v in args.variants:
            build(d, v, args.out, args.jobs)


if __name__ == "__main__":
    main()
