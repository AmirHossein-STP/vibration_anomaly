"""MASTER_PLAN.md WP1: re-score the envelope model at recording and unit level.

No new science: same features, same CWRU-only logistic model, same
healthy-only calibration as envelope_transfer.py. What changes is how the
result is counted, to answer the questions a deployment actually asks:

1. Window FA was calibrated on random healthy windows, so calibration and
   test shared bearing units. A deployed machine is a NEW unit. Here the
   threshold is set on five healthy units and false alarms are counted on
   the sixth (leave-one-healthy-unit-out, LOHUO), with a finite-sample
   (conformal) quantile. Exchangeability across units does not hold, so the
   nominal rate is a target, not a guarantee -- the held-out FA shows by
   how much it is missed.
2. A recording (one 4 s Paderborn run, 7 windows) gets a verdict from the
   fraction of its windows above threshold, as the bearing service does.
   The service's levels (watch 25%, faulty 50%) were defaults; here they
   are measured. Levels are recommended from HEALTHY data only (recording
   FA), never from fault recall (MASTER_PLAN §4 rule 4).
3. A unit is "caught" by the fraction of its 20 recordings judged faulty.

Reads the cache written by envelope_cache.py; runs in seconds.

Usage: python scripts/analysis/wp1_rescore.py [--conditions N09_M07_F10 ...]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from domain_adaptation.features.envelope import FEATURE_NAMES
from envelope_cache import load_cache

CONDITIONS = ["N09_M07_F10", "N15_M01_F10", "N15_M07_F04", "N15_M07_F10"]
ALPHAS = (0.05, 0.10, 0.20)
SERVICE_LEVELS = {"watch": 0.25, "faulty": 0.50}
# Pre-registered rule for recommending levels (healthy data only): the
# smallest window-count level whose mean held-out recording FA is at most
# this rate, at the 10% window operating point.
LEVEL_RULE = {"watch": 0.20, "faulty": 0.05}
CWRU_SECONDS = 4.0   # source windows starting in a file's first 4 s: the published 8 x 1 s windows


def conformal_threshold(cal: np.ndarray, alpha: float) -> float:
    """The ceil((n+1)(1-alpha))-th smallest calibration score: for a new
    exchangeable score s, P(s > thr) <= alpha. +inf when n is too small."""
    s = np.sort(np.asarray(cal))
    k = math.ceil((len(s) + 1) * (1 - alpha))
    return float("inf") if k > len(s) else float(s[k - 1])


def mean_sd(v) -> str:
    v = np.asarray(v, dtype=float)
    return f"{v.mean():.3f} ± {v.std(ddof=1) if len(v) > 1 else 0.0:.3f}"


def fit_source(df: pd.DataFrame):
    src = df[(df.dataset == "cwru") & (df.t0 < CWRU_SECONDS - 1e-9)]
    scaler = StandardScaler().fit(src[FEATURE_NAMES])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
        scaler.transform(src[FEATURE_NAMES]), (src.label > 0).astype(int))
    return scaler, clf, src


def drift_report(src: pd.DataFrame, tgt: pd.DataFrame) -> pd.Series:
    """Standardised mean difference per feature, healthy windows only (§4 rule 7)."""
    a, b = src[src.label == 0][FEATURE_NAMES], tgt[tgt.label == 0][FEATURE_NAMES]
    pooled = np.sqrt((a.var() + b.var()) / 2).replace(0, np.nan)
    return ((b.mean() - a.mean()) / pooled).round(2)


def reproduce_published(t: pd.DataFrame) -> dict:
    """envelope_transfer.py's protocol exactly: first 4 windows per run,
    real damage only, 10 random 60/40 healthy-window splits, np.quantile."""
    t = t[(t.win < 4) & (t.damage_mode != "artificial")]
    y, s = t.label.to_numpy(), t.score.to_numpy()
    recs, fas = [], []
    for seed in range(10):
        hi = np.where(y == 0)[0].copy(); np.random.default_rng(seed).shuffle(hi)
        n = int(0.6 * len(hi)); cal, th = hi[:n], hi[n:]
        thr = np.quantile(s[cal], 0.9)
        recs.append(float((s[y > 0] > thr).mean())); fas.append(float((s[th] > thr).mean()))
    bal = [(r + 1 - f) / 2 for r, f in zip(recs, fas)]
    return {"recall": recs, "fa": fas, "balanced": bal}


def analyse_condition(t: pd.DataFrame) -> dict:
    """t: one condition's Paderborn windows with a `score` column."""
    healthy_units = sorted(t[t.label == 0].unit.unique())
    real = t[(t.label > 0) & (t.damage_mode == "real")]
    art = t[(t.label > 0) & (t.damage_mode == "artificial")]
    n_win = int(t.groupby(["unit", "run"]).size().mode()[0])
    levels = [k / n_win for k in range(1, n_win + 1)]

    def rec_frac(frame, thr):
        return frame.assign(f=frame.score > thr).groupby(["unit", "run"]).f.mean()

    out = {"n_windows_per_recording": n_win, "healthy_units": healthy_units,
           "window": {}, "recording": {}, "recording_median": {}, "units": {}}

    for alpha in ALPHAS:
        w_fa, w_rec, w_rec_art, insample_fa = [], [], [], {}
        lvl_fa = {k: [] for k in levels}; lvl_rec = {k: [] for k in levels}
        lvl_rec_art = {k: [] for k in levels}
        med_fa, med_rec = [], []
        unit_faulty = {u: [] for u in sorted(real.unit.unique())}
        unit_healthy = {}
        for h in healthy_units:
            cal = t[(t.label == 0) & (t.unit != h)]
            held = t[(t.label == 0) & (t.unit == h)]
            thr = conformal_threshold(cal.score, alpha)
            w_fa.append(float((held.score > thr).mean()))
            w_rec.append(float((real.score > thr).mean()))
            w_rec_art.append(float((art.score > thr).mean()) if len(art) else float("nan"))
            fh, fr, fa_ = rec_frac(held, thr), rec_frac(real, thr), rec_frac(art, thr)
            for k in levels:
                lvl_fa[k].append(float((fh >= k - 1e-9).mean()))
                lvl_rec[k].append(float((fr >= k - 1e-9).mean()))
                lvl_rec_art[k].append(float((fa_ >= k - 1e-9).mean()) if len(fa_) else float("nan"))
            # verdict per unit at the service's faulty level
            for u, g in fr.groupby(level="unit"):
                unit_faulty[u].append(float((g >= SERVICE_LEVELS["faulty"] - 1e-9).mean()))
            unit_healthy[h] = {"window_fa": w_fa[-1],
                               "recording_faulty_rate": float((fh >= SERVICE_LEVELS["faulty"] - 1e-9).mean()),
                               "recording_watch_rate": float((fh >= SERVICE_LEVELS["watch"] - 1e-9).mean())}
            # alternative: recording score = median window score, threshold
            # calibrated on the other units' RECORDINGS directly
            rmed = lambda fr_: fr_.groupby(["unit", "run"]).score.median()
            rthr = conformal_threshold(rmed(cal), alpha)
            med_fa.append(float((rmed(held) > rthr).mean()))
            med_rec.append(float((rmed(real) > rthr).mean()))
        # the service's own way: threshold in-sample on every healthy window
        thr_all = float(np.quantile(t[t.label == 0].score, 1 - alpha))
        for h in healthy_units:
            insample_fa[h] = float((t[(t.label == 0) & (t.unit == h)].score > thr_all).mean())

        key = f"{alpha:.2f}"
        out["window"][key] = {"lohuo_fa": w_fa, "lohuo_recall_real": w_rec, "lohuo_recall_artificial": w_rec_art,
                              "insample_fa_by_unit": insample_fa}
        out["recording"][key] = {f"{k:.3f}": {"fa": lvl_fa[k], "recall_real": lvl_rec[k],
                                              "recall_artificial": lvl_rec_art[k]} for k in levels}
        out["recording_median"][key] = {"fa": med_fa, "recall_real": med_rec}
        out["units"][key] = {"faulty_units_caught_fraction": {u: float(np.mean(v)) for u, v in unit_faulty.items()},
                             "healthy_units": unit_healthy}
    # recommended levels (healthy-only rule, at the 10% window point)
    rec10 = out["recording"]["0.10"]
    out["recommended_levels"] = {}
    for name, cap in LEVEL_RULE.items():
        ok = [float(k) for k, v in rec10.items() if np.mean(v["fa"]) <= cap]
        out["recommended_levels"][name] = min(ok) if ok else None
    return out


def service_level_index(levels: list[str], frac: float) -> str:
    """The window-count level the service's fraction resolves to (>= frac)."""
    return min((k for k in levels if float(k) >= frac - 1e-9), key=float)


def render(results: dict, drift: dict, repro: dict) -> str:
    L = ["# WP1 results: recording- and unit-level re-scoring", "",
         "Generated by `scripts/analysis/wp1_rescore.py` from the envelope cache. "
         "Model: CWRU-only logistic regression on 13 envelope-order features; no Paderborn label "
         "enters any fit or threshold. Faulty = real damage unless stated.", ""]
    L += ["## 0. Reproduction of the published N09 number", "",
          "Protocol of `envelope_transfer.py` (first 4 windows/run, 10 random healthy-window splits, 10% FA):", "",
          f"- recall {mean_sd(repro['recall'])}, FA {mean_sd(repro['fa'])}, balanced {mean_sd(repro['balanced'])}",
          "- published (notes/13): recall 0.746 ± 0.012, balanced 0.823 ± 0.007", ""]
    L += ["## 1. Drift report (healthy windows, standardised mean difference PU − CWRU)", "",
          "| feature | " + " | ".join(drift) + " |", "|---|" + "---|" * len(drift)]
    for f in FEATURE_NAMES:
        L.append(f"| {f} | " + " | ".join(f"{drift[c][f]:+.2f}" for c in drift) + " |")
    L.append("")
    L += ["## 2. Window level: random-window vs held-out-unit calibration", "",
          "LOHUO = threshold from 5 healthy units (conformal quantile), FA counted on the 6th; "
          "mean ± sd over the 6 folds. `worst` = highest held-out unit FA.", "",
          "| condition | target FA | held-out FA | worst unit FA | recall real | recall artificial |",
          "|---|---|---|---|---|---|"]
    for c, r in results.items():
        for a, w in r["window"].items():
            L.append(f"| {c} | {float(a):.0%} | {mean_sd(w['lohuo_fa'])} | {max(w['lohuo_fa']):.3f} | "
                     f"{mean_sd(w['lohuo_recall_real'])} | {mean_sd(w['lohuo_recall_artificial'])} |")
    L.append("")
    L += ["## 3. Recording level at the 10% window point", "",
          "A recording is flagged when at least the given fraction of its windows exceed the threshold. "
          "FA = held-out healthy recordings flagged (mean ± sd over folds). "
          f"Service levels: watch {SERVICE_LEVELS['watch']:.0%}, faulty {SERVICE_LEVELS['faulty']:.0%}.", ""]
    for c, r in results.items():
        rec = r["recording"]["0.10"]
        lv = list(rec)
        sw, sf = service_level_index(lv, SERVICE_LEVELS["watch"]), service_level_index(lv, SERVICE_LEVELS["faulty"])
        L += [f"### {c} ({r['n_windows_per_recording']} windows/recording)", "",
              "| level (≥ windows) | recording FA | worst unit | recall real | recall artificial | note |",
              "|---|---|---|---|---|---|"]
        n = r["n_windows_per_recording"]
        for k, v in rec.items():
            note = []
            if k == sw: note.append("service watch")
            if k == sf: note.append("service faulty")
            for name, lvl in r["recommended_levels"].items():
                if lvl is not None and abs(float(k) - lvl) < 1e-9: note.append(f"recommended {name}")
            L.append(f"| {round(float(k) * n)}/{n} | {mean_sd(v['fa'])} | {max(v['fa']):.2f} | "
                     f"{mean_sd(v['recall_real'])} | {mean_sd(v['recall_artificial'])} | {', '.join(note)} |")
        m = r["recording_median"]["0.10"]
        L += ["", f"Alternative statistic, median window score with a recording-level conformal threshold at 10%: "
                  f"FA {mean_sd(m['fa'])}, recall real {mean_sd(m['recall_real'])}.", ""]
    L += ["## 4. Units at the service's faulty level (10% window point)", "",
          "Faulty units: fraction of their 20 recordings judged faulty (mean over the 6 calibration folds). "
          "Healthy units: when held out, fraction of their recordings judged watch / faulty.", ""]
    for c, r in results.items():
        u = r["units"]["0.10"]
        fu = ", ".join(f"{k} {v:.2f}" for k, v in u["faulty_units_caught_fraction"].items())
        hu = ", ".join(f"{k} {v['recording_watch_rate']:.2f}/{v['recording_faulty_rate']:.2f}"
                       for k, v in u["healthy_units"].items())
        caught = sum(v >= 0.5 for v in u["faulty_units_caught_fraction"].values())
        L += [f"- **{c}** — faulty units caught on ≥ half their recordings: "
              f"{caught}/{len(u['faulty_units_caught_fraction'])}", f"  - faulty: {fu}",
              f"  - healthy (watch/faulty): {hu}"]
    L += ["", "## 5. Recommended verdict levels (healthy data only)", "",
          f"Rule (fixed before looking): smallest level with mean held-out recording FA ≤ "
          f"{LEVEL_RULE['watch']:.0%} (watch) and ≤ {LEVEL_RULE['faulty']:.0%} (faulty), at the 10% window point.", ""]
    for c, r in results.items():
        L.append(f"- {c}: " + ", ".join(f"{k} {'none meets the cap' if v is None else f'{v:.2f}'}"
                                        for k, v in r["recommended_levels"].items()))
    return "\n".join(L) + "\n"


def score_variant(df: pd.DataFrame, conditions=CONDITIONS):
    """Fit on CWRU, score every Paderborn window, analyse each condition."""
    scaler, clf, src = fit_source(df)
    pu = df[df.dataset == "pu"].copy()
    pu["score"] = clf.predict_proba(scaler.transform(pu[FEATURE_NAMES]))[:, 1]
    results, drift = {}, {}
    for c in conditions:
        t = pu[pu.condition == c]
        drift[c] = drift_report(src, t)
        results[c] = analyse_condition(t)
    return results, drift, pu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", nargs="*", default=CONDITIONS)
    ap.add_argument("--cache", type=Path, default=Path("outputs/envelope_cache"))
    ap.add_argument("--out", type=Path, default=Path("outputs/wp1"))
    args = ap.parse_args()

    results, drift, pu = score_variant(load_cache(args.cache), args.conditions)
    repro = reproduce_published(pu[pu.condition == "N09_M07_F10"])

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "wp1_results.json").write_text(json.dumps(
        {"results": results, "drift": {c: d.to_dict() for c, d in drift.items()}, "reproduction": repro},
        indent=1, default=float))
    md = render(results, drift, repro)
    (args.out / "wp1_report.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
