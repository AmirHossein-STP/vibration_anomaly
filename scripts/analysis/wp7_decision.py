"""MASTER_PLAN WP7: decision layer on XJTU-SY, lead time at a false-alarm
budget. Protocol and readings are pre-registered in notes/18 §2–§3.

Usage: python scripts/analysis/wp7_decision.py   (box: needs data/xjtu_sy for RMS)
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
from domain_adaptation.data.public_datasets import xjtu_bearings
from wp5_cache import load_wp5
from domain_adaptation.method import LINES, cusum, ewma, fit_baseline, standardise
from wp7_xjtu import CAL_MIN, ROOT, rms_per_minute

HEALTHY = (21, 60)
LONG_LIFE = 120
LAMBDA, K = 0.2, 0.5
SOURCES = ("phys", "rms", "kurt", "comb")
KINDS = ("raw", "ewma", "cusum")


def standardised(df: pd.DataFrame, b, channel: int) -> dict[str, np.ndarray]:
    name = "horizontal" if channel == 0 else "vertical"
    f = df[(df.unit == b.bearing) & (df.channel == name)].sort_values("minute")
    cal = f[f.minute <= CAL_MIN]
    base = fit_baseline(cal[LINES].to_numpy(), cal.minute.to_numpy())    # method v1.0: leave one minute out
    phys = standardise(base.score(f[LINES].to_numpy()), base.cal_scores)
    rms = rms_per_minute(b.path, channel)
    r = standardise(rms, rms[:CAL_MIN])
    # notes/18 §6.2: kurtosis of the record's 1 s window (log, own baseline)
    lk = np.log(f.td_kurtosis.to_numpy())
    kurt = standardise(lk, lk[:CAL_MIN])
    return {"phys": phys, "rms": r, "kurt": kurt}


def statistic(u: np.ndarray, kind: str) -> np.ndarray:
    """Over minutes 21.. (index 20..); minutes 1-20 are NaN."""
    if kind == "raw":
        out = np.asarray(u, dtype=np.float64).copy()
        out[:CAL_MIN] = np.nan
        return out
    return ewma(u, LAMBDA, CAL_MIN) if kind == "ewma" else cusum(u, K, CAL_MIN)


def series(df, bearings, channel_mode: str) -> dict:
    """{bearing: {detector: statistic per minute}}."""
    out = {}
    for b in bearings.itertuples():
        chans = [0] if channel_mode == "horizontal" else [1] if channel_mode == "vertical" else [0, 1]
        us = [standardised(df, b, ch) for ch in chans]
        u = {k: np.maximum.reduce([x[k] for x in us]) for k in ("phys", "rms", "kurt")}
        u["comb"] = np.maximum(u["phys"], u["rms"])
        out[b.bearing] = {f"{src}_{kind}": statistic(u[src], kind) for src in SOURCES for kind in KINDS}
    return out


def detect(stat: np.ndarray, h: float, long_life: bool) -> int | None:
    start = HEALTHY[1] if long_life else CAL_MIN
    idx = np.flatnonzero(stat[start:] > h)
    return int(idx[0] + start + 1) if len(idx) else None       # minute number (1-based)


def healthy_max(stat: np.ndarray) -> float:
    return float(np.nanmax(stat[HEALTHY[0] - 1:HEALTHY[1]]))


def evaluate(S: dict, bearings: pd.DataFrame) -> dict:
    life = dict(zip(bearings.bearing, bearings.n_minutes))
    longs = [b for b in life if life[b] >= LONG_LIFE]
    res = {}
    for det in next(iter(S.values())):
        per = []
        for b in life:
            others = [o for o in longs if o != b]
            h = max(healthy_max(S[o][det]) for o in others) + 1e-9      # FA budget 0, leave one bearing out
            st = S[b][det]
            d = detect(st, h, life[b] >= LONG_LIFE)
            fa_min = int((st[HEALTHY[0] - 1:HEALTHY[1]] > h).sum()) if life[b] >= LONG_LIFE else None
            per.append({"bearing": b, "life": int(life[b]), "h": h, "alarm": d,
                        "lead": (int(life[b]) - d) if d else None,
                        "lead_frac": (life[b] - d) / life[b] if d else None, "fa_minutes": fa_min})
        p = pd.DataFrame(per)
        lp = p[p.life >= LONG_LIFE]
        fa_total = int(lp.fa_minutes.sum())
        res[det] = {"detected": int(p.alarm.notna().sum()), "detected_long": int(lp.alarm.notna().sum()),
                    "median_lead": float(p.lead.median()) if p.lead.notna().any() else 0.0,
                    "median_lead_frac": float(p.lead_frac.median()) if p.lead_frac.notna().any() else 0.0,
                    "fa_minutes_heldout": fa_total,
                    "fa_per_day": fa_total / (len(lp) * (HEALTHY[1] - HEALTHY[0] + 1)) * 1440,
                    "per_bearing": per}
    return res


def curve(S: dict, bearings: pd.DataFrame, det: str) -> pd.DataFrame:
    life = dict(zip(bearings.bearing, bearings.n_minutes))
    longs = [b for b in life if life[b] >= LONG_LIFE]
    hs = np.unique(np.concatenate([S[b][det][HEALTHY[0] - 1:HEALTHY[1]] for b in longs]))
    hs = np.quantile(hs, np.linspace(0.5, 1, 60)).tolist() + [max(hs) + 1e-9]
    rows = []
    for h in hs:
        fa = np.mean(np.concatenate([S[b][det][HEALTHY[0] - 1:HEALTHY[1]] > h for b in longs]))
        leads = [life[b] - d for b in life if (d := detect(S[b][det], h, life[b] >= LONG_LIFE))]
        rows.append({"h": h, "healthy_alarm_frac": fa, "detected": len(leads),
                     "median_lead": float(np.median(leads)) if leads else 0.0})
    return pd.DataFrame(rows)


def figure(curves: dict, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"phys_ewma": "#2a78d6", "phys_cusum": "#1baf7a", "rms_ewma": "#eb6834", "rms_cusum": "#eda100",
              "kurt_cusum": "#e87ba4", "comb_cusum": "#6250d6"}
    ink, muted = "#0b0b0b", "#52514e"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for det, c in curves.items():
        if det not in colors:
            continue
        x = np.maximum(c.healthy_alarm_frac, 1e-3)
        axes[0].plot(x, c.detected, color=colors[det], lw=2, label=det)
        axes[1].plot(x, c.median_lead, color=colors[det], lw=2, label=det)
    for ax, yl in zip(axes, ("bearings detected (of 15)", "median lead time, minutes")):
        ax.set_xscale("log")
        ax.set_xlabel("share of assumed-healthy minutes in alarm (minutes 21-60; 0 plotted at 0.001)", fontsize=8, color=muted)
        ax.set_ylabel(yl, fontsize=9, color=muted)
        ax.tick_params(labelsize=8, colors=muted)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=11, color=ink, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("outputs/wp7"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    bearings = xjtu_bearings(ROOT)
    df = load_wp5("xjtu_sy", "hd_cpw_ses")
    allres = {}
    L = ["# WP7: decision layer on XJTU-SY (lead time at FA budget 0)", "",
         "Generated by `scripts/analysis/wp7_decision.py`; pre-registered in notes/18. Threshold per held-out bearing = "
         "just above the max statistic in the OTHER long-lived bearings' minutes 21-60 (assumed healthy).", ""]
    for mode in ("horizontal", "vertical", "both"):
        S = series(df, bearings, mode)
        res = evaluate(S, bearings)
        allres[mode] = res
        L += [f"## {mode}", "", "| detector | detected (of 15) | long-lived detected (of 12) | median lead min | "
              "median lead / life | held-out healthy FA minutes | FA per day |", "|---|---|---|---|---|---|---|"]
        for det, r in res.items():
            L.append(f"| {det} | {r['detected']} | {r['detected_long']} | {r['median_lead']:.0f} | "
                     f"{r['median_lead_frac']:.2f} | {r['fa_minutes_heldout']} | {r['fa_per_day']:.1f} |")
        if mode == "horizontal":
            curves = {d: curve(S, bearings, d) for d in res}
            figure(curves, args.out / "wp7_decision_horizontal.png",
                   "XJTU-SY horizontal: detection vs false alarms in assumed-healthy minutes, own-baseline detectors")
            ph = max(("phys_ewma", "phys_cusum"), key=lambda d: (res[d]["detected"], res[d]["median_lead"]))
            rb = max(("rms_ewma", "rms_cusum", "rms_raw"), key=lambda d: (res[d]["detected"], res[d]["median_lead"]))
            comp = (res[ph]["detected"] >= res[rb]["detected"]
                    and res[ph]["median_lead"] >= 0.8 * res[rb]["median_lead"])
            comb_better = res["comb_cusum"]["detected"] >= max(res[ph]["detected"], res[rb]["detected"]) and \
                res["comb_cusum"]["median_lead"] > max(res[ph]["median_lead"], res[rb]["median_lead"])
            smooth = res[ph]["fa_minutes_heldout"] <= res["phys_raw"]["fa_minutes_heldout"]
            kj = res["kurt_cusum"]["detected"] >= res["rms_cusum"]["detected"]
            L.append(f"**§6.2:** kurt_cusum detects {res['kurt_cusum']['detected']}/15 vs rms_cusum "
                     f"{res['rms_cusum']['detected']}/15 -> {'joins the trigger' if kj else 'does not join the trigger'}.")
            allres["readings"] = {"kurt_joins_trigger": kj, "best_phys": ph, "best_rms": rb, "physics_competitive": comp,
                                  "comb_better": comb_better, "smoothing_helps_fa": smooth}
            L += ["", f"**Readings (notes/18 §3):** best physics `{ph}` vs best RMS `{rb}` -> "
                      f"**{'physics competitive' if comp else 'RMS stays the trigger'}**; comb better: {comb_better}; "
                      f"smoothing reduces held-out FA vs raw: {smooth}.", ""]
        L += [""]
    md = "\n".join(L) + "\n"
    (args.out / "wp7_decision_report.md").write_text(md)
    (args.out / "wp7_decision_results.json").write_text(json.dumps(allres, indent=1, default=float))
    print(md)


if __name__ == "__main__":
    main()
