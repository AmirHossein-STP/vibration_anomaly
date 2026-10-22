"""WP15: Paderborn LDM run-to-failure under time-varying load and speed (notes/25, pre-registered).

Usage (box): python scripts/analysis/v2_cache.py --set cpw --datasets paderborn_ldm --jobs 8
             python scripts/analysis/v2_cache.py --set geofree --datasets paderborn_ldm --jobs 8
             python scripts/analysis/wp15_ldm.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))
from domain_adaptation.features.geofree import BIN_NAMES
from domain_adaptation.method import LINES, conformal_threshold, cusum, fit_baseline, standardise
from v2_cache import load_v2

BASE_N, HEALTHY_END, LONG_MIN = 150, 450, 1000
STATS = ["phys6", "geofree", "L_pooled", "L_cond"]
EXPECT = {"inner": ["B02", "B03", "B04", "B06", "B07", "B08", "B12", "B13", "B15"], "outer": ["B09", "B10"]}


def per_bearing(cpw: pd.DataFrame, geo: pd.DataFrame) -> dict:
    cpw = cpw[cpw.set_rpm >= 1000].sort_values(["record", "win"])
    geo = geo[geo.set_rpm >= 1000].sort_values(["record", "win"])
    recs = np.sort(cpw.record.unique())
    base_recs = recs[:BASE_N]
    out = {"records": recs, "time": pd.to_datetime(cpw.groupby("record").time.first().loc[recs].to_numpy())}

    def record_series(frame, cols):
        bmask = frame.record.isin(base_recs).to_numpy()
        b = fit_baseline(frame.loc[bmask, cols].to_numpy(), frame.loc[bmask, "record"].to_numpy())
        s = pd.Series(b.score(frame[cols].to_numpy()), index=frame.index).groupby(frame.record).mean().loc[recs]
        cal = pd.Series(b.cal_scores, index=frame.index[bmask]).groupby(frame.record[bmask]).mean()
        return standardise(s.to_numpy(), cal.to_numpy()), standardise(cal.to_numpy(), cal.to_numpy()), b

    out["phys6"], out["phys6_cal"], b6 = record_series(cpw, LINES)
    out["geofree"], out["geofree_cal"], _ = record_series(geo, BIN_NAMES)
    rec = cpw.groupby("record").agg(rms=("td_rms", "mean"), speed=("speed_rpm", "first"), stat=("stat_N", "first"),
                                    dyn=("peak_dyn_N", "first")).loc[recs]
    lr = np.log(rec.rms.to_numpy())
    isb = np.isin(recs, base_recs)
    out["L_pooled"] = standardise(lr, lr[isb])
    out["L_pooled_cal"] = out["L_pooled"][isb]
    X = np.c_[np.ones(len(rec)), np.log(rec.speed.to_numpy()), rec.stat.to_numpy(), rec.dyn.to_numpy()]
    coef, *_ = np.linalg.lstsq(X[isb], lr[isb], rcond=None)
    resid = lr - X @ coef
    out["L_cond"] = standardise(resid, resid[isb])
    out["L_cond_cal"] = out["L_cond"][isb]
    # typing at end of life (last 10% of records), majority over windows
    tail = cpw[cpw.record.isin(recs[-max(3, int(np.ceil(0.1 * len(recs)))):])]
    pred = b6.typing(tail[LINES].to_numpy())
    out["typed"] = "inner" if (pred == 1).mean() > 0.5 else "outer"
    out["inner_share"] = float((pred == 1).mean())
    return out


def analyse(cpw_all: pd.DataFrame, geo_all: pd.DataFrame, channel: str) -> dict:
    S = {}
    for bn in sorted(cpw_all.unit.unique()):
        c = cpw_all[(cpw_all.unit == bn) & (cpw_all.channel == channel)]
        g = geo_all[(geo_all.unit == bn) & (geo_all.channel == channel)]
        S[bn] = per_bearing(c, g)
    longs = [b for b in S if len(S[b]["records"]) >= LONG_MIN]
    seg = slice(BASE_N, HEALTHY_END)
    res = {"fa_alpha10": {}, "cusum": {}, "per_bearing": {}}
    # (a)/(b): record-level FA in the assumed-healthy segment at conformal alpha 0.10
    for st in STATS:
        res["fa_alpha10"][st] = float(np.mean([np.mean(S[b][st][seg] > conformal_threshold(S[b][f"{st}_cal"]))
                                               for b in longs]))
    # CUSUM at FA budget 0, leave one bearing out
    C = {b: {st: cusum(S[b][st], start=BASE_N) for st in STATS} for b in S}
    for st in STATS:
        det = {}
        for b in S:
            h = max(np.nanmax(C[o][st][seg]) for o in longs if o != b) + 1e-9
            start = HEALTHY_END if b in longs else BASE_N
            idx = np.flatnonzero(C[b][st][start:] > h)
            if len(idx):
                i = idx[0] + start
                det[b] = {"record": int(S[b]["records"][i]),
                          "lead_h": float((S[b]["time"][-1] - S[b]["time"][i]).total_seconds() / 3600)}
            else:
                det[b] = None
            if b in longs:
                det.setdefault("_fa_heldout", 0)
        leads = [d["lead_h"] for k, d in det.items() if not k.startswith("_") and d]
        res["cusum"][st] = {"detected": len(leads), "median_lead_h": float(np.median(leads)) if leads else 0.0,
                            "per_bearing": {k: v for k, v in det.items() if not k.startswith("_")}}
    for b in S:
        res["per_bearing"][b] = {"n_records": int(len(S[b]["records"])), "typed": S[b]["typed"],
                                 "inner_share": S[b]["inner_share"],
                                 "duration_h": float((S[b]["time"][-1] - S[b]["time"][0]).total_seconds() / 3600)}
    exp = {b: k for k, bs in EXPECT.items() for b in bs}
    correct = [res["per_bearing"][b]["typed"] == exp[b] for b in exp if b in res["per_bearing"]]
    res["typing"] = {"correct": int(sum(correct)), "of": len(correct)}
    best_level = max(("L_pooled", "L_cond"), key=lambda s: (res["cusum"][s]["detected"], res["cusum"][s]["median_lead_h"]))
    res["readings"] = {
        "a_confounding": res["fa_alpha10"]["L_pooled"] > 0.25 and res["fa_alpha10"]["phys6"] <= 0.15,
        "b_conditioning_helps": res["fa_alpha10"]["L_cond"] < res["fa_alpha10"]["L_pooled"],
        "best_level": best_level,
        "Q1_physics_takes_over": res["cusum"]["phys6"]["detected"] >= res["cusum"][best_level]["detected"],
        "c_geofree_wins_ldm": (res["cusum"]["geofree"]["detected"] >= res["cusum"]["phys6"]["detected"]
                               and res["cusum"]["geofree"]["median_lead_h"] >= res["cusum"]["phys6"]["median_lead_h"]),
        "d_typing_pass": sum(correct) >= 2 / 3 * len(correct)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("outputs/wp15"))
    a = ap.parse_args()
    cpw = load_v2("paderborn_ldm", fset="cpw")
    geo = load_v2("paderborn_ldm", fset="geofree")
    allres = {}
    L = ["# WP15: Paderborn LDM run-to-failure under varying load and speed (unseen)", "",
         "Generated by `scripts/analysis/wp15_ldm.py`; pre-registered in notes/25. Own baseline = first 150 records "
         "after warm-up; assumed-healthy = records 151-450; CUSUM k=0.5 at FA budget 0 (leave one bearing out).", ""]
    for ch in ("accHorizRear_A", "accHorizFrontal_C"):
        r = analyse(cpw, geo, ch)
        allres[ch] = r
        rd = r["readings"]
        L += [f"## channel {ch}{' (primary)' if ch.endswith('_A') else ''}", "",
              f"- (a) record FA at alpha 0.10 in assumed-healthy records: " +
              ", ".join(f"{s} {r['fa_alpha10'][s]:.3f}" for s in STATS) +
              f" -> confounding as predicted: **{rd['a_confounding']}**",
              f"- (b) conditioning lowers level FA: **{rd['b_conditioning_helps']}**",
              f"- **Q1: physics takes over as trigger: {rd['Q1_physics_takes_over']}** (best level: {rd['best_level']})",
              f"- (c) geometry-free wins LDM: **{rd['c_geofree_wins_ldm']}**",
              f"- (d) typing at end of life: {r['typing']['correct']}/{r['typing']['of']} -> **{'PASS' if rd['d_typing_pass'] else 'FAIL'}**", "",
              "| statistic | bearings detected (of 17) | median lead (h) |", "|---|---|---|"]
        for s in STATS:
            L.append(f"| {s} | {r['cusum'][s]['detected']} | {r['cusum'][s]['median_lead_h']:.1f} |")
        L += ["", "| bearing | records | duration h | defect (Table 3) | typed (inner share) | phys6 lead h | geofree lead h | L_pooled lead h | L_cond lead h |",
              "|---|---|---|---|---|---|---|---|---|"]
        from domain_adaptation.data.public_datasets import LDM_DEFECTS
        for b, pb in r["per_bearing"].items():
            f = lambda s: (f"{r['cusum'][s]['per_bearing'][b]['lead_h']:.1f}" if r['cusum'][s]['per_bearing'][b] else "—")
            L.append(f"| {b} | {pb['n_records']} | {pb['duration_h']:.1f} | {LDM_DEFECTS[b]} | {pb['typed']} ({pb['inner_share']:.2f}) | "
                     f"{f('phys6')} | {f('geofree')} | {f('L_pooled')} | {f('L_cond')} |")
        L.append("")
    md = "\n".join(L) + "\n"
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "wp15_report.md").write_text(md)
    (a.out / "wp15_results.json").write_text(json.dumps(allres, indent=1, default=float))
    print(md)


if __name__ == "__main__":
    main()
