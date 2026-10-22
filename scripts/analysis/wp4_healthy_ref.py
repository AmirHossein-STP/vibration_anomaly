"""MASTER_PLAN.md WP4: healthy-referenced normalisation and physics typing.

Paderborn adaptation. A faulty Paderborn bearing was never recorded healthy,
so a baseline of "this unit's own healthy data" does not exist there. The
deployable analogue is the MACHINE's healthy baseline in that operating
condition. In each leave-one-healthy-unit-out fold it comes from the 5
calibration healthy units only; the held-out unit is never seen by
normalisation or threshold.

Arms (same features, same counting as WP1: conformal threshold at 10% from
the 5 calibration units, FA on the 6th, recall on real damage):
  a  global   CWRU StandardScaler + CWRU logistic regression (the WP1 model)
  b  hz       healthy z-scoring: source standardised by CWRU healthy stats,
              target by the calibration units' healthy stats; same classifier
  c  hcoral   healthy-to-healthy CORAL: each domain whitened by its own healthy
              covariance (Ledoit-Wolf shrinkage), then the classifier
  d  physics  no classifier, no source: score = max over the 6 fault-order
              peaks (bpfo_h1..3, bpfi_h1..3) of the one-sided z-score of
              log(peak) against the healthy baseline (notes/15 §3)

Physics typing (no learning): on every true-faulty window, the race whose
peaks rose most (max z of bpfi_h* vs bpfo_h*). Compared with notes/13's
60.8% correct race (CWRU 3-class classifier, N09).

Run on the v1 features and on CPW+SES (notes/15: the information is there,
the CWRU boundary is inverted).

Pre-registered readings (written before running):
- WP4 gate (MASTER_PLAN): b or c PASSES if worst-unit held-out FA drops
  below arm a's in that condition while mean recall falls by at most 0.02.
- d is the healthy-only physics detector of MASTER_PLAN §3 (L1+L2). It is
  promising if, on CPW+SES, it reaches arm a's v1 recall at N09 (0.740) and
  beats it at N15. If it is worse everywhere, the source model is doing
  real work and stays.
- Typing passes if correct-race accuracy beats 0.608 at N09 and is above
  0.5 in every condition; otherwise typing stays out of the claims.
- Nothing here is chosen by Paderborn fault labels; arms and features are
  fixed above.

Usage: python scripts/analysis/wp4_healthy_ref.py
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
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from domain_adaptation.features.envelope import FEATURE_NAMES
from envelope_cache import load_cache
from wp1_rescore import CONDITIONS, CWRU_SECONDS, conformal_threshold

ALPHA = 0.10
PEAKS = ["bpfo_h1", "bpfo_h2", "bpfo_h3", "bpfi_h1", "bpfi_h2", "bpfi_h3"]
ARMS = ["a_global", "b_hz", "c_hcoral", "d_physics"]


def whitener(X: np.ndarray):
    mu = X.mean(0)
    cov = LedoitWolf().fit(X).covariance_
    w, V = np.linalg.eigh(cov)
    W = V @ np.diag(1 / np.sqrt(np.maximum(w, 1e-12))) @ V.T
    return lambda Z: (Z - mu) @ W


def log_peaks(frame: pd.DataFrame) -> np.ndarray:
    return np.log(frame[PEAKS].to_numpy() + 1e-9)


def run(df: pd.DataFrame) -> dict:
    src = df[(df.dataset == "cwru") & (df.t0 < CWRU_SECONDS - 1e-9)]
    Xs, ys = src[FEATURE_NAMES].to_numpy(), (src.label > 0).astype(int).to_numpy()
    Xs_h = Xs[ys == 0]
    lr = lambda X: LogisticRegression(max_iter=2000, class_weight="balanced").fit(X, ys)
    sc = StandardScaler().fit(Xs); clf_a = lr(sc.transform(Xs))
    mu_h, sd_h = Xs_h.mean(0), Xs_h.std(0) + 1e-12; clf_b = lr((Xs - mu_h) / sd_h)
    ws = whitener(Xs_h); clf_c = lr(ws(Xs))

    pu = df[df.dataset == "pu"]
    out = {}
    for c in CONDITIONS:
        t = pu[pu.condition == c]
        healthy_units = sorted(t[t.label == 0].unit.unique())
        real = t[(t.label > 0) & (t.damage_mode == "real")]
        res = {a: {"fa": [], "recall": [], "auc": []} for a in ARMS}
        typing = []
        for h in healthy_units:
            cal = t[(t.label == 0) & (t.unit != h)]
            held = t[(t.label == 0) & (t.unit == h)]
            Xc = cal[FEATURE_NAMES].to_numpy()
            mu_t, sd_t = Xc.mean(0), Xc.std(0) + 1e-12
            wt = whitener(Xc)
            lp_mu, lp_sd = log_peaks(cal).mean(0), log_peaks(cal).std(0) + 1e-12
            scorers = {
                "a_global": lambda f: clf_a.predict_proba(sc.transform(f[FEATURE_NAMES].to_numpy()))[:, 1],
                "b_hz": lambda f: clf_b.predict_proba((f[FEATURE_NAMES].to_numpy() - mu_t) / sd_t)[:, 1],
                "c_hcoral": lambda f: clf_c.predict_proba(wt(f[FEATURE_NAMES].to_numpy()))[:, 1],
                "d_physics": lambda f: ((log_peaks(f) - lp_mu) / lp_sd).max(1),
            }
            for a, fn in scorers.items():
                thr = conformal_threshold(fn(cal), ALPHA)
                res[a]["fa"].append(float((fn(held) > thr).mean()))
                res[a]["recall"].append(float((fn(real) > thr).mean()))
                # threshold-free: held-out healthy unit vs real damage
                sh, sr = fn(held), fn(real)
                res[a]["auc"].append(float(roc_auc_score(np.r_[np.zeros(len(sh)), np.ones(len(sr))], np.r_[sh, sr])))
            z = (log_peaks(real) - lp_mu) / lp_sd
            pred = np.where(z[:, 3:].max(1) > z[:, :3].max(1), 1, 2)    # 1 inner, 2 outer
            typing.append(float((pred == real.label.to_numpy()).mean()))
            if h == healthy_units[0]:
                z_in = z[real.label.to_numpy() == 1]; z_out = z[real.label.to_numpy() == 2]
                inner_ok = float((z_in[:, 3:].max(1) > z_in[:, :3].max(1)).mean())
                outer_ok = float((z_out[:, :3].max(1) >= z_out[:, 3:].max(1)).mean())
        out[c] = {a: {"fa_mean": float(np.mean(v["fa"])), "fa_worst": float(max(v["fa"])),
                      "recall": float(np.mean(v["recall"])), "recall_sd": float(np.std(v["recall"], ddof=1)),
                      "auc": float(np.mean(v["auc"])), "auc_min": float(min(v["auc"]))}
                  for a, v in res.items()}
        out[c]["typing"] = {"race_correct": float(np.mean(typing)), "inner_ok_fold0": inner_ok,
                            "outer_ok_fold0": outer_ok}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=["v1", "hd_cpw_ses"])
    ap.add_argument("--out", type=Path, default=Path("outputs/wp4"))
    args = ap.parse_args()
    allres = {v: run(load_cache(Path("outputs/envelope_cache"), v)) for v in args.variants}

    L = ["# WP4 results: healthy-referenced scoring and physics typing", "",
         "Generated by `scripts/analysis/wp4_healthy_ref.py`. Real damage; conformal threshold at 10% from 5 healthy "
         "units, FA on the 6th; mean over the 6 folds. Gate per arm vs `a_global` of the same features.", "",
         "| features | condition | arm | recall | held-out FA | worst-unit FA | AUC (worst fold) | gate (b, c) |", "|---|---|---|---|---|---|---|---|"]
    for v, r in allres.items():
        for c in CONDITIONS:
            base = r[c]["a_global"]
            for a in ARMS:
                m = r[c][a]
                g = ""
                if a in ("b_hz", "c_hcoral"):
                    g = "pass" if (m["fa_worst"] < base["fa_worst"] and m["recall"] >= base["recall"] - 0.02) else "fail"
                L.append(f"| {v} | {c} | {a} | {m['recall']:.3f} ± {m['recall_sd']:.3f} | {m['fa_mean']:.3f} | "
                         f"{m['fa_worst']:.2f} | {m['auc']:.3f} ({m['auc_min']:.2f}) | {g} |")
    L += ["", "## Physics typing (no learning): correct race on true-faulty windows", "",
          "| features | condition | correct race | inner read as inner | outer read as outer |", "|---|---|---|---|---|"]
    for v, r in allres.items():
        for c in CONDITIONS:
            ty = r[c]["typing"]
            L.append(f"| {v} | {c} | {ty['race_correct']:.3f} | {ty['inner_ok_fold0']:.3f} | {ty['outer_ok_fold0']:.3f} |")
    L.append("\nReference: notes/13, CWRU 3-class classifier at N09, 0.608 correct race (chance 0.5).")
    md = "\n".join(L) + "\n"
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "wp4_report.md").write_text(md)
    (args.out / "wp4_results.json").write_text(json.dumps(allres, indent=1))
    print(md)


if __name__ == "__main__":
    main()
