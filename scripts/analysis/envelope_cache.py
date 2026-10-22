"""Extract envelope-order features once, to a fingerprinted cache.

MASTER_PLAN.md WP1 and WP2 both need envelope-order window features for
CWRU (the source) and for Paderborn under all four operating conditions.
Extracting them is the only heavy step; everything after it (thresholds,
recording verdicts, per-unit tables) reads this cache in seconds. So this
script is run once, wherever the owner decides heavy work runs, and the
scorers (`wp1_rescore.py`) run anywhere.

Every window is kept with its provenance (dataset, unit, condition, run,
window index), so subsets reproduce earlier numbers exactly: the published
CWRU->PU N09 result used the first 8 windows per CWRU file and the first 4
per Paderborn run, i.e. `win < 8` / `win < 4` here.

The cache name carries a fingerprint of everything that changes its
contents -- envelope.py's source, the window settings, the CWRU file list,
the Paderborn manifest (MASTER_PLAN §4 rule 9). A stale cache is never
silently reused.

Feature variants (MASTER_PLAN WP2) are named in VARIANTS, each fixed before
it is run; `v1` is the published configuration. Every variant gets its own
cache file.

Usage (from the repo root):
    python scripts/analysis/envelope_cache.py [--variant v1 ...] [--jobs N] [--out outputs/envelope_cache]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from domain_adaptation.data import pu_bearing_codes as pbc
from domain_adaptation.data.cwru_loader import build_cwru_catalog
from domain_adaptation.data.pu_loader import PreparedPUStore
from domain_adaptation.features.envelope import (
    CWRU_6205, FEATURE_NAMES, PU_6203, envelope_order_features,
)

FS = 12000
WIN = 12000          # 1 s, identical to envelope_transfer.py and the bearing service
HOP = 6000
CWRU_RAW = Path("data/raw")
PU_STORE = Path("data/pu/prepared_pu")
ENVELOPE_SRC = Path("src/domain_adaptation/features/envelope.py")
PU_CODES = pbc.HEALTHY + pbc.REAL_OUTER + pbc.REAL_INNER + pbc.ARTIFICIAL_OUTER + pbc.ARTIFICIAL_INNER
META_COLUMNS = ["dataset", "unit", "condition", "run", "win", "t0", "rpm", "label", "damage_mode"]

# win_revs: window length in shaft revolutions instead of WIN samples (hop
# is half a window either way). kw: passed to envelope_order_features.
VARIANTS: dict[str, dict] = {
    "v1": {"win_revs": None, "kw": {}},
    # H-b: 15 revolutions is what the 1 s window holds at 900 rpm, where the
    # method works; fixing it makes the order resolution identical at every
    # speed, so ratio and shape features see the same number of bins
    "hb_revs15": {"win_revs": 15, "kw": {}},
    # H-b, normalisation only: peak / median of the reference band
    "hb_median": {"win_revs": None, "kw": {"norm": "median"}},
    # H-c: +-5% relative tolerance, as Shen et al. 2021 use for slip and speed error
    "hc_reltol5": {"win_revs": None, "kw": {"tol_rel": 0.05}},
    # H-d: no band choice at all -- cepstrum pre-whitening + squared envelope
    "hd_cpw_ses": {"win_revs": None, "kw": {"whiten": True, "band_hz": None, "squared": True}},
}


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fingerprint(cwru_files: list[Path], variant: str = "v1") -> str:
    h = hashlib.sha256()
    if variant != "v1":
        h.update(json.dumps({variant: VARIANTS[variant]}, sort_keys=True).encode())
    h.update(ENVELOPE_SRC.read_bytes())
    h.update(json.dumps({"fs": FS, "win": WIN, "hop": HOP, "features": FEATURE_NAMES,
                         "pu_codes": PU_CODES}).encode())
    for f in sorted(cwru_files):
        h.update(f"{f.name}:{f.stat().st_size}".encode())
    h.update((PU_STORE / "manifest.csv").read_bytes())
    return h.hexdigest()[:12]


def _windows(sig: np.ndarray, rpm: float, win_revs: float | None):
    win = WIN if win_revs is None else int(round(win_revs * 60.0 / rpm * FS))
    hop = win // 2
    for i, s in enumerate(range(0, len(sig) - win + 1, hop)):
        yield i, s / FS, sig[s:s + win]


def _cwru_job(args) -> list[dict]:
    path, rpm, label, variant = args
    v = VARIANTS[variant]
    sig = np.asarray(np.load(path)["DE"]).squeeze().astype(np.float64)
    unit = Path(path).stem
    out = []
    for i, t0, w in _windows(sig, rpm, v["win_revs"]):
        f = envelope_order_features(w, FS, rpm, CWRU_6205, **v["kw"])
        out.append({"dataset": "cwru", "unit": unit, "condition": str(rpm), "run": 0, "win": i, "t0": t0,
                    "rpm": rpm, "label": int(label), "damage_mode": "artificial" if label else "none", **f})
    return out


def _pu_job(args) -> list[dict]:
    code, condition, run, rpm, label, mode, variant = args
    v = VARIANTS[variant]
    store = PreparedPUStore(PU_STORE)
    sig = np.asarray(store.load(code, condition, run)).squeeze().astype(np.float64)
    out = []
    for i, t0, w in _windows(sig, rpm, v["win_revs"]):
        f = envelope_order_features(w, FS, rpm, PU_6203, **v["kw"])
        out.append({"dataset": "pu", "unit": code, "condition": condition, "run": int(run), "win": i, "t0": t0,
                    "rpm": rpm, "label": int(label), "damage_mode": mode, **f})
    return out


def cache_path(out_dir: Path, variant: str = "v1") -> Path:
    cat = build_cwru_catalog(CWRU_RAW, [1730, 1750, 1772], 1797)
    fp = fingerprint([Path(p) for p in cat.filepath], variant)
    return out_dir / (f"envelope_windows_{fp}.csv.gz" if variant == "v1"
                      else f"envelope_windows_{variant}_{fp}.csv.gz")


def load_cache(out_dir: Path = Path("outputs/envelope_cache"), variant: str = "v1") -> pd.DataFrame:
    """The scorers' entry point. Fails loudly on a missing or stale cache."""
    p = cache_path(out_dir, variant)
    if not p.is_file():
        have = sorted(x.name for x in out_dir.glob("envelope_windows_*.csv.gz")) if out_dir.is_dir() else []
        raise FileNotFoundError(
            f"{p} not found (stale or never built; present: {have or 'none'}). "
            "Run scripts/analysis/envelope_cache.py first -- it is the heavy step, see MASTER_PLAN §5.")
    return pd.read_csv(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("outputs/envelope_cache"))
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--variant", nargs="*", default=["v1"], choices=list(VARIANTS))
    args = ap.parse_args()
    for variant in args.variant:
        extract(variant, args.out, args.jobs)


def extract(variant: str, out_dir: Path, n_jobs: int) -> None:
    cat = build_cwru_catalog(CWRU_RAW, [1730, 1750, 1772], 1797)
    out_path = cache_path(out_dir, variant)
    if out_path.is_file():
        print(f"[{variant}] cache is current: {out_path}")
        return
    jobs = [(r.filepath, int(r.rpm), int(r.universal_label), variant) for r in cat.itertuples()]
    store = PreparedPUStore(PU_STORE)
    pcat = store.catalog(PU_CODES)
    pjobs = [(r.code, r.condition, int(r.run), int(r.rpm), int(r.universal_label), r.damage_mode, variant)
             for r in pcat.itertuples()]
    print(f"[{variant}] {len(jobs)} CWRU files, {len(pjobs)} Paderborn runs; {n_jobs} worker(s)")

    t0 = time.time()
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        for part in ex.map(_cwru_job, jobs):
            rows += part
        for k, part in enumerate(ex.map(_pu_job, pjobs, chunksize=16)):
            rows += part
            if (k + 1) % 250 == 0:
                print(f"  {k + 1}/{len(pjobs)} runs, {time.time() - t0:.0f} s", flush=True)

    df = pd.DataFrame(rows)[META_COLUMNS + FEATURE_NAMES]
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, float_format="%.7g")
    (out_path.with_suffix("").with_suffix(".json")).write_text(json.dumps({
        "fingerprint": out_path.stem.split(".")[0].split("_")[-1], "variant": variant,
        "variant_config": VARIANTS[variant], "n_windows": len(df),
        "fs": FS, "win": WIN, "hop": HOP, "envelope_sha": _sha(ENVELOPE_SRC.read_bytes()),
        "seconds": round(time.time() - t0, 1),
        "windows_by": df.groupby(["dataset", "condition"]).size().to_dict().__repr__(),
    }, indent=2))
    print(f"wrote {len(df)} windows to {out_path} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
