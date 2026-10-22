"""WP10: validate shaft speed from vibration (notes/20) on Ottawa 2018 (encoder)
and Paderborn (measured speed channel), and test it downstream on Ottawa 2018.

Usage (box): python scripts/analysis/wp10_speed.py --jobs 8
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))
from domain_adaptation.data import public_datasets as pdx
from domain_adaptation.data.pu_loader import PreparedPUStore
from domain_adaptation.features.envelope import envelope_order_features
from domain_adaptation.features.speed import estimate_shaft_hz
from envelope_cache import PU_CODES, PU_STORE
from wp5_cache import FS, HOP, WIN, to_12k
from wp5_score import evaluate, physics_xf

CPW = {"whiten": True, "band_hz": None, "squared": True}
PU_V2 = Path("data/pu/prepared_pu_v2")


def _ottawa(stem: str) -> list[dict]:
    rec = next(pdx.ottawa2018(Path("data/ottawa_2018"), only=stem))
    sig, fs = to_12k(rec.signal, rec.fs)
    phase = np.interp(np.arange(len(sig)) / fs, np.arange(len(rec.phase_revs)) / rec.fs, rec.phase_revs)
    rows = []
    for i, s in enumerate(range(0, len(sig) - WIN + 1, HOP)):
        w = sig[s:s + WIN]
        true_hz = (phase[s + WIN - 1] - phase[s]) / ((WIN - 1) / fs)
        est_hz = estimate_shaft_hz(w, fs, 10.0, 35.0)
        row = {"dataset": "ottawa_2018", "unit": rec.unit, "recording": rec.recording, "label": rec.label,
               "state": rec.state, "win": i, "t0": s / fs, "true_hz": true_hz, "est_hz": est_hz}
        for tag, hz in (("true", true_hz), ("est", est_hz)):
            f = envelope_order_features(w, fs, 60.0 * hz, pdx.OTTAWA18_ER16K, **CPW)
            row.update({f"{tag}__{k}": v for k, v in f.items()})
        rows.append(row)
    return rows


def _pu(args) -> list[dict]:
    code, cond, run, rpm = args
    if not (PU_V2 / f"{code}.npz").is_file():
        return []          # KA08: not in the Zenodo raw set, so no v2 speed channel
    sig = np.asarray(PreparedPUStore(PU_STORE).load(code, cond, run), dtype=np.float64)
    with np.load(PU_V2 / f"{code}.npz") as z:
        key = f"{cond}_{run}__speed"
        if key not in z.files:
            return []
        spd = np.asarray(z[key], dtype=np.float64)
    spd_hz = spd / 60.0 if np.median(spd) > 200 else spd     # channel in rpm (checked) -> Hz
    nom = rpm / 60.0
    rows = []
    for i, s in enumerate(range(0, len(sig) - WIN + 1, HOP)):
        a, b = int(s / len(sig) * len(spd_hz)), int((s + WIN) / len(sig) * len(spd_hz))
        rows.append({"dataset": "pu", "unit": code, "condition": cond, "run": run, "win": i,
                     "true_hz": float(spd_hz[a:b].mean()), "nominal_hz": nom,
                     "est_hz": estimate_shaft_hz(sig[s:s + WIN], FS, 0.8 * nom, 1.2 * nom)})
    return rows


def errors(df: pd.DataFrame, col: str) -> dict:
    r = df[col] / df.true_hz
    rel = (r - 1).abs()
    octave = ((r - 2).abs() < 0.05) | ((r - 0.5).abs() < 0.05) | ((r - 1 / 3).abs() < 0.05) | ((r - 3).abs() < 0.05)
    return {"n": int(len(df)), "median_abs_pct": float(100 * rel.median()), "within_1pct": float((rel <= 0.01).mean()),
            "within_2pct": float((rel <= 0.02).mean()), "octave_errors": float(octave.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("outputs/wp10"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    stems = sorted(p.stem for p in Path("data/ottawa_2018").glob("*.mat"))
    pcat = PreparedPUStore(PU_STORE).catalog(PU_CODES)
    with ProcessPoolExecutor(a.jobs) as ex:
        ott = pd.DataFrame([r for part in ex.map(_ottawa, stems) for r in part])
        pu = pd.DataFrame([r for part in ex.map(_pu, [(r.code, r.condition, int(r.run), int(r.rpm))
                                                      for r in pcat.itertuples()], chunksize=16) for r in part])
    ott.to_csv(a.out / "ottawa_2018_speed_windows.csv.gz", index=False)
    pu.to_csv(a.out / "pu_speed_windows.csv.gz", index=False)

    res = {"ottawa_2018": {"estimate": errors(ott, "est_hz")},
           "pu": {"estimate": errors(pu, "est_hz"), "nominal": errors(pu, "nominal_hz"),
                  "by_condition": {c: {"estimate": errors(g, "est_hz"), "nominal": errors(g, "nominal_hz")}
                                   for c, g in pu.groupby("condition")}}}
    # downstream: physics score at per-window constant rpm, true vs estimated
    down = {}
    for tag in ("true", "est"):
        f = ott[["dataset", "unit", "recording", "label", "state", "win", "t0"]].copy()
        for c in [c for c in ott.columns if c.startswith(f"{tag}__")]:
            f[c.split("__", 1)[1]] = ott[c]
        down[tag] = evaluate("ottawa_2018", f, physics_xf)["summary"]
    res["downstream_ottawa_2018"] = {k: {m: down[k][m] for m in ("auc_race", "recall_race", "fa_mean")} for k in down}
    res["downstream_ottawa_2018"]["cot_encoder_notes17"] = {"auc_race": 0.922}
    adequate = down["est"]["auc_race"] >= down["true"]["auc_race"] - 0.05
    cot_worth = min(down["true"]["auc_race"], down["est"]["auc_race"]) <= 0.922 - 0.05
    res["readings"] = {"estimator_adequate": adequate, "cot_worth_encoder": cot_worth}

    e, p = res["ottawa_2018"]["estimate"], res["pu"]
    L = ["# WP10: shaft speed from vibration", "", "Generated by `scripts/analysis/wp10_speed.py`; pre-registered in notes/20.", "",
         "| data | estimator | windows | median abs error | within 1% | within 2% | octave errors |", "|---|---|---|---|---|---|---|",
         f"| Ottawa 2018 (13-28 Hz sweeps, encoder truth) | harmonic sum | {e['n']} | {e['median_abs_pct']:.2f}% | "
         f"{e['within_1pct']:.2f} | {e['within_2pct']:.2f} | {e['octave_errors']:.3f} |"]
    for c, d in p["by_condition"].items():
        for k in ("estimate", "nominal"):
            x = d[k]
            L.append(f"| Paderborn {c} (speed channel truth) | {'harmonic sum' if k == 'estimate' else 'nominal setpoint'} | "
                     f"{x['n']} | {x['median_abs_pct']:.2f}% | {x['within_1pct']:.2f} | {x['within_2pct']:.2f} | {x['octave_errors']:.3f} |")
    L += ["", "Downstream, Ottawa 2018 physics score (arm d, cross-fitted), 1 s windows at one rpm per window:", "",
          "| speed source | race AUC | race recall | FA |", "|---|---|---|---|"]
    for k, lbl in (("true", "encoder mean per window"), ("est", "vibration estimate per window")):
        d = res["downstream_ottawa_2018"][k]
        L.append(f"| {lbl} | {d['auc_race']:.3f} | {d['recall_race']:.3f} | {d['fa_mean']:.3f} |")
    L += ["| encoder + computed order tracking (notes/17) | 0.922 | 0.757 | 0.097 |", "",
          f"**Readings:** estimator adequate as fallback: {adequate}; order tracking worth its encoder: {cot_worth}."]
    md = "\n".join(L) + "\n"
    (a.out / "wp10_report.md").write_text(md)
    (a.out / "wp10_results.json").write_text(json.dumps(res, indent=1, default=float))
    print(md)


if __name__ == "__main__":
    main()
