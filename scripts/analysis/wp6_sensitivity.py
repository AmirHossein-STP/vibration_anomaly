"""WP6 (part): per-machine sensitivity curve from synthetic faults injected into
the machine's own healthy windows (notes/21; generator fixed there).

Usage (box): python scripts/analysis/wp6_sensitivity.py --jobs 8
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import spearmanr

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))
from domain_adaptation.data import public_datasets as pdx
from domain_adaptation.data import pu_bearing_codes as pbc
from domain_adaptation.data.pu_loader import PreparedPUStore
from domain_adaptation.features.envelope import PU_6203, envelope_order_features
from envelope_cache import PU_STORE
from wp1_rescore import conformal_threshold
from wp5_cache import FS, HOP, WIN, to_12k
from wp5_score import PEAKS, physics_xf

CPW = {"whiten": True, "band_hz": None, "squared": True}
AMPS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
ZETA, JITTER, SEEDS, ALPHA = 0.03, 0.015, 3, 0.10


def resonance_hz(windows: list[np.ndarray]) -> float:
    f, p = welch(np.concatenate(windows), fs=FS, nperseg=2048)
    band = (f >= 1000) & (f <= 5500)
    return float(f[band][np.argmax(p[band])])


def synthetic(n: int, fault_hz: float, shaft_hz: float, res_hz: float, inner: bool, rng) -> np.ndarray:
    """Unit-peak impulse train (notes/21 §1)."""
    w = 2 * np.pi * res_hz
    t_h = np.arange(int(5 / (ZETA * w) * FS) + 1) / FS
    h = np.exp(-ZETA * w * t_h) * np.sin(w * np.sqrt(1 - ZETA ** 2) * t_h)
    h /= np.abs(h).max()
    out = np.zeros(n + len(h))
    t, period, phi = rng.uniform(0, 1 / fault_hz), 1 / fault_hz, rng.uniform(0, 2 * np.pi)
    while t < n / FS:
        amp = 0.5 + 0.5 * np.cos(2 * np.pi * shaft_hz * t + phi) if inner else 1.0
        i = int(round(t * FS))
        out[i:i + len(h)] += amp * h
        t += period * (1 + JITTER * rng.standard_normal())
    return out[:n]


def feats(w: np.ndarray, rpm: float, g) -> dict:
    return envelope_order_features(w, FS, rpm, g, **CPW)


def frame(rows: list[dict], recording: str) -> pd.DataFrame:
    f = pd.DataFrame(rows)
    f["recording"] = recording if isinstance(recording, str) else recording
    return f


def curve(cal_rows: pd.DataFrame, held: list[np.ndarray], rpm: float, g, res_hz: float, seed: int) -> dict:
    score, _ = physics_xf(cal_rows)
    thr = conformal_threshold(score.calibration, ALPHA)
    shaft = rpm / 60.0
    out = {}
    for name, order, inner in (("outer", g.bpfo, False), ("inner", g.bpfi, True)):
        probs = []
        for a in AMPS:
            hits = []
            for s in range(SEEDS):
                rng = np.random.default_rng([seed, s, int(a * 1000), int(inner)])
                rows = []
                for x in held:
                    syn = synthetic(len(x), order * shaft, shaft, res_hz, inner, rng)
                    rows.append(feats(x + a * np.sqrt((x ** 2).mean()) * syn, rpm, g))
                hits += list(score(pd.DataFrame(rows)) > thr)
            probs.append(float(np.mean(hits)))
        out[name] = probs
    return out


def a50(probs: list[float]) -> float:
    if probs[0] >= 0.5:
        return AMPS[0]
    for i in range(1, len(AMPS)):
        if probs[i] >= 0.5:
            p0, p1 = probs[i - 1], probs[i]
            return float(np.exp(np.log(AMPS[i - 1]) + (0.5 - p0) / max(p1 - p0, 1e-9)
                                * (np.log(AMPS[i]) - np.log(AMPS[i - 1]))))
    return 4.0   # never reached 50% within the sweep (> 2)


def _ottawa(bearing: int) -> dict:
    rec = next(pdx.ottawa2023(Path("data/ottawa_2023"), only=f"H_{bearing}_0"))
    sig, _ = to_12k(rec.signal, rec.fs)
    wins = [(s / FS, sig[s:s + WIN]) for s in range(0, len(sig) - WIN + 1, HOP)]
    cal = [(t, w) for t, w in wins if t + 1.0 <= 5.0 + 1e-9]
    held = [w for t, w in wins if t >= 5.0 - 1e-9]
    cal_rows = pd.DataFrame([{**feats(w, rec.rpm, pdx.OTTAWA23_6203), "t0": t} for t, w in cal])
    cal_rows["recording"] = rec.recording
    res = resonance_hz([w for _, w in cal])
    c = curve(cal_rows, held, rec.rpm, pdx.OTTAWA23_6203, res, bearing)
    return {"unit": f"ott23_b{bearing:02d}", "resonance_hz": res, **{f"p_{k}": v for k, v in c.items()},
            "a50_outer": a50(c["outer"]), "a50_inner": a50(c["inner"])}


def _pu(args) -> dict:
    cond, held_unit = args
    store = PreparedPUStore(PU_STORE)
    rpm = 900 if cond.startswith("N09") else 1500

    def wins(code, runs, per_run):
        out = []
        for r in runs:
            sig = np.asarray(store.load(code, cond, r), dtype=np.float64)
            out += [(f"{code}_{r}", s / FS, sig[s:s + WIN]) for s in list(range(0, len(sig) - WIN + 1, HOP))[:per_run]]
        return out
    cal = [w for u in pbc.HEALTHY if u != held_unit for w in wins(u, range(1, 21), 3)]
    held = [w for _, _, w in wins(held_unit, range(1, 21), 1)]
    cal_rows = pd.DataFrame([{**feats(w, rpm, PU_6203), "t0": t, "recording": rid} for rid, t, w in cal])
    res = resonance_hz([w for _, _, w in cal[:60]])
    c = curve(cal_rows, held, rpm, PU_6203, res, hash((cond, held_unit)) % 10**6)
    return {"condition": cond, "held_unit": held_unit, "resonance_hz": res, **{f"p_{k}": v for k, v in c.items()},
            "a50_outer": a50(c["outer"]), "a50_inner": a50(c["inner"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("outputs/wp6"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    conds = ["N09_M07_F10", "N15_M01_F10", "N15_M07_F04", "N15_M07_F10"]
    with ProcessPoolExecutor(a.jobs) as ex:
        ott = pd.DataFrame(list(ex.map(_ottawa, range(1, 21))))
        pu = pd.DataFrame(list(ex.map(_pu, [(c, u) for c in conds for u in pbc.HEALTHY])))

    wp5 = json.loads(Path("outputs/wp5/wp5_results.json").read_text())["results"]
    real = {f["fold"]: f.get("recall_race") for f in wp5["ottawa_2023:own"]["d_xf"]["folds"]}
    ott["real_recall_race"] = ott.unit.map(real)
    ott["bearing"] = ott.unit.str[-2:].astype(int)
    race = ott[ott.bearing <= 10].copy()
    race["a50_own_type"] = np.where(race.bearing <= 5, race.a50_inner, race.a50_outer)
    rho, pval = spearmanr(race.a50_own_type, race.real_recall_race)
    predicts = bool(rho <= -0.5)
    pu_real = {c: wp5[f"pu:{c}"]["d_xf"]["summary"]["recall_race"] for c in conds}
    pu_sum = pu.groupby("condition").agg(a50_outer=("a50_outer", "median"), a50_inner=("a50_inner", "median"),
                                         resonance_hz=("resonance_hz", "median")).reset_index()
    pu_sum["real_recall_race"] = pu_sum.condition.map(pu_real)

    L = ["# WP6: per-machine sensitivity from synthetic faults", "",
         "Generated by `scripts/analysis/wp6_sensitivity.py`; pre-registered in notes/21. a50 = injected impulse peak "
         "(x window RMS) at 50% detection; 4.0 = not reached by 2.0.", "",
         f"**Reading: Spearman rho(a50 for the bearing's own fault type, real race recall) over Ottawa 2023's 10 race "
         f"bearings = {rho:.2f} (p = {pval:.2f}) -> {'PREDICTS (ship as sensitivity estimate)' if predicts else 'does NOT predict (not shipped as a sensitivity estimate)'}.**", "",
         "| bearing | fault | resonance Hz | a50 outer | a50 inner | a50 own type | real race recall |", "|---|---|---|---|---|---|---|"]
    for r in race.itertuples():
        L.append(f"| {r.unit} | {'inner' if r.bearing <= 5 else 'outer'} | {r.resonance_hz:.0f} | {r.a50_outer:.2f} | "
                 f"{r.a50_inner:.2f} | {r.a50_own_type:.2f} | {r.real_recall_race:.2f} |")
    L += ["", "Paderborn (machine baseline, median over 6 held-out healthy units):", "",
          "| condition | a50 outer | a50 inner | resonance Hz | real race recall (d_xf) |", "|---|---|---|---|---|"]
    for r in pu_sum.itertuples():
        L.append(f"| {r.condition} | {r.a50_outer:.2f} | {r.a50_inner:.2f} | {r.resonance_hz:.0f} | {r.real_recall_race:.3f} |")
    L += ["", "Detection probability vs amplitude, Ottawa 2023 median over bearings:", "",
          "| amplitude | " + " | ".join(str(x) for x in AMPS) + " |", "|---|" + "---|" * len(AMPS)]
    for k in ("outer", "inner"):
        med = np.median(np.stack(ott[f"p_{k}"].to_list()), 0)
        L.append(f"| {k} | " + " | ".join(f"{v:.2f}" for v in med) + " |")
    md = "\n".join(L) + "\n"
    (a.out / "wp6_report.md").write_text(md)
    ott.to_json(a.out / "wp6_ottawa2023.json", orient="records", indent=1)
    pu.to_json(a.out / "wp6_pu.json", orient="records", indent=1)
    (a.out / "wp6_results.json").write_text(json.dumps({"rho": rho, "p": pval, "predicts": predicts}, indent=1))
    print(md)


if __name__ == "__main__":
    main()
