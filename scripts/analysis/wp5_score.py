"""MASTER_PLAN.md WP5 scoring: the frozen physics score (arm d) and the
CWRU-trained classifier (arm a) on CWRU, Ottawa 2018, Ottawa 2023 and MFPT.
The protocol and gate are pre-registered in notes/17 §2–§4 and nothing here
is chosen by these datasets' labels.

Arms:
  a     v1 features, StandardScaler + logistic regression fit on CWRU
        (wp1_rescore.fit_source), threshold calibrated on target healthy windows
  d_v1  physics score on v1 features
  d     physics score on CPW+SES features (the primary candidate, notes/16)
  d_xf  POST-HOC (notes/17 §5): d with cross-fitted calibration scores

Physics score: max over BPFO/BPFI h1–h3 of the one-sided z-score of
log(peak) against the calibration (healthy) windows. Threshold: conformal
quantile at alpha = 0.10. Folds per dataset: notes/17 §3.

Usage: python scripts/analysis/wp5_score.py   (reads outputs/wp5/cache and outputs/envelope_cache)
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
from sklearn.metrics import roc_auc_score

from domain_adaptation.data.public_datasets import LABEL_NAMES
from domain_adaptation.method import LINES, default_groups, fit_baseline
from domain_adaptation.features.envelope import FEATURE_NAMES
from envelope_cache import load_cache
from wp1_rescore import conformal_threshold, drift_report, fit_source
from wp5_cache import load_wp5

ALPHA = 0.10
PEAKS = LINES
RACE = (1, 2)
FAULTY_LEVEL = 0.50          # bearing service `faulty` recording level (notes/14)


def log_peaks(f: pd.DataFrame) -> np.ndarray:
    return np.log(f[PEAKS].to_numpy() + 1e-9)


def physics(cal: pd.DataFrame):
    lp = log_peaks(cal)
    mu, sd = lp.mean(0), lp.std(0) + 1e-12
    return (lambda f: ((log_peaks(f) - mu) / sd).max(1)), (mu, sd)


def crossfit_scores(cal: pd.DataFrame) -> np.ndarray:
    """POST-HOC (notes/17 §5): calibration scores computed out of sample.

    The pre-registered scorer fits mean/sd on the calibration windows and
    then scores those same windows, so with few windows the threshold is
    biased low and held-out FA overshoots alpha. Here each calibration
    recording (or, inside a single recording, each contiguous third) is
    scored by a baseline fit without it. Overlapping windows of one
    recording therefore never score each other.
    """
    groups = cal.recording.to_numpy() if cal.recording.nunique() >= 2 else \
        np.minimum((np.argsort(np.argsort(cal.t0.to_numpy())) * 3) // len(cal), 2)
    lp, out = log_peaks(cal), np.empty(len(cal))
    for g in np.unique(groups):
        m = groups == g
        mu, sd = lp[~m].mean(0), lp[~m].std(0) + 1e-12
        out[m] = ((lp[m] - mu) / sd).max(1)
    return out


def physics_xf(cal: pd.DataFrame):
    """Method v1.0 (src/domain_adaptation/method.py, notes/22): own baseline,
    cross-fitted calibration. Numerically identical to physics() + crossfit_scores()."""
    b = fit_baseline(cal[PEAKS].to_numpy(), default_groups(cal.recording.to_numpy(), cal.t0.to_numpy()))
    score = lambda f: b.score(f[PEAKS].to_numpy())
    score.calibration = b.cal_scores
    return score, (b.mu, b.sd)


def zmax_xf(cols: list[str]):
    """One-sided max z of log(cols) vs the calibration windows, cross-fitted
    calibration (as d_xf). Used for the level arms of notes/18 §6."""
    lg = lambda f: np.log(f[cols].to_numpy() + 1e-12)

    def factory(cal: pd.DataFrame):
        lc = lg(cal)
        mu, sd = lc.mean(0), lc.std(0) + 1e-12
        score = lambda f: ((lg(f) - mu) / sd).max(1)
        groups = cal.recording.to_numpy() if cal.recording.nunique() >= 2 else \
            np.minimum((np.argsort(np.argsort(cal.t0.to_numpy())) * 3) // len(cal), 2)
        out = np.empty(len(cal))
        for g in np.unique(groups):
            m = groups == g
            out[m] = ((lc[m] - lc[~m].mean(0)) / (lc[~m].std(0) + 1e-12)).max(1)
        score.calibration = out
        return score, None
    return factory


def combined_dK(cal: pd.DataFrame):
    """notes/18 §6.4: max of the physics score and the kurtosis z, each
    standardised by its own cross-fitted calibration scores."""
    sd_, _ = physics_xf(cal)
    sk, _ = zmax_xf(["td_kurtosis"])(cal)
    cd, ck = sd_.calibration, sk.calibration
    md, vd, mk, vk = cd.mean(), cd.std() + 1e-12, ck.mean(), ck.std() + 1e-12
    score = lambda f: np.maximum((sd_(f) - md) / vd, (sk(f) - mk) / vk)
    score.calibration = np.maximum((cd - md) / vd, (ck - mk) / vk)
    return score, None


def type_correct(f: pd.DataFrame, mu, sd) -> float:
    f = f[f.label.isin(RACE)]
    if f.empty:
        return float("nan")
    z = (log_peaks(f) - mu) / sd
    pred = np.where(z[:, 3:].max(1) > z[:, :3].max(1), 1, 2)
    return float((pred == f.label.to_numpy()).mean())


# --- folds: (name, calibration healthy, held-out healthy, faulty) -------------

def folds(dataset: str, df: pd.DataFrame, baseline: str = "machine"):
    h, f = df[df.label == 0], df[df.label > 0]
    if dataset == "cwru":
        for c in sorted(h.condition.unique()):
            yield c, h[h.condition != c], h[h.condition == c], f[f.condition == c]
    elif dataset.startswith("pu:"):          # reference: notes/16's protocol, one condition
        real = f[f.damage_mode == "real"]
        for u in sorted(h.unit.unique()):
            yield u, h[h.unit != u], h[h.unit == u], real
    elif dataset in ("ottawa_2018", "mfpt"):
        for r in sorted(h.recording.unique()):
            yield r, h[h.recording != r], h[h.recording == r], f
    elif dataset == "ottawa_2023":
        for u in sorted(df.unit.unique()):
            mine = df[df.unit == u]
            fu = mine[mine.label > 0]
            if baseline == "machine":
                yield u, h[h.unit != u], h[h.unit == u], fu
            else:   # own unit: first 5 s calibrate, last 5 s held out; the straddling window is dropped
                hu = mine[mine.label == 0]
                yield u, hu[hu.t0 + 1.0 <= 5.0 + 1e-9], hu[hu.t0 >= 5.0 - 1e-9], fu
    else:
        raise ValueError(dataset)


def evaluate(dataset: str, df: pd.DataFrame, scorer_factory, baseline: str = "machine", typing: bool = False) -> dict:
    per = []
    for name, cal, held, fault in folds(dataset, df, baseline):
        score, stats = scorer_factory(cal)
        thr = conformal_threshold(getattr(score, "calibration", None) if hasattr(score, "calibration") else score(cal), ALPHA)
        sh, sf = score(held), score(fault)
        race = fault.label.isin(RACE).to_numpy()
        row = {"fold": name, "n_cal": len(cal), "thr": float(thr), "fa": float((sh > thr).mean()),
               "recall_race": float((sf[race] > thr).mean()) if race.any() else np.nan}
        for lab in sorted(fault.label.unique()):
            m = fault.label.to_numpy() == lab
            row[f"recall_{LABEL_NAMES[lab]}"] = float((sf[m] > thr).mean())
        for st in sorted(fault.state.unique()):
            m = (fault.state.to_numpy() == st) & race
            if m.any():
                row[f"recall_race_{st}"] = float((sf[m] > thr).mean())
        if race.any():
            row["auc_race"] = float(roc_auc_score(np.r_[np.zeros(len(sh)), np.ones(race.sum())], np.r_[sh, sf[race]]))
        bc = fault.label.isin((3, 4)).to_numpy()          # ball + cage (notes/19)
        if bc.any():
            row["auc_bc"] = float(roc_auc_score(np.r_[np.zeros(len(sh)), np.ones(bc.sum())], np.r_[sh, sf[bc]]))
        # recording level: share of a recording's windows over threshold
        fr = fault.assign(over=sf > thr).groupby("recording").agg(over=("over", "mean"), label=("label", "first"))
        frr = fr[fr.label.isin(RACE)]
        row["rec_detect_race"] = float((frr.over >= FAULTY_LEVEL).mean()) if len(frr) else np.nan
        row["rec_fa"] = float(((sh > thr).mean() >= FAULTY_LEVEL))
        if typing and stats is not None:
            if callable(stats):
                row.update(stats(fault))
            else:
                row["typing"] = type_correct(fault, *stats)
        per.append(row)
    p = pd.DataFrame(per)
    summary = {"folds": len(p), "fa_mean": p.fa.mean(), "fa_worst": p.fa.max(),
               "recall_race": p.recall_race.mean(), "recall_race_sd": p.recall_race.std(ddof=1) if len(p) > 1 else 0.0,
               "auc_race": p.auc_race.mean() if "auc_race" in p else np.nan,
               "auc_race_min": p.auc_race.min() if "auc_race" in p else np.nan,
               "rec_detect_race": p.rec_detect_race.mean(), "rec_fa": p.rec_fa.mean()}
    for c in p.columns:
        if c.startswith("recall_") and c not in summary:
            summary[c] = p[c].mean()   # per-type: mean over folds that contain the type
    for c in p.columns:
        if c.startswith("typing"):
            summary[c] = p[c].mean()
    if "auc_bc" in p:
        summary["auc_bc"], summary["auc_bc_min"] = p.auc_bc.mean(), p.auc_bc.min()
    return {"summary": {k: float(v) for k, v in summary.items()}, "folds": p.to_dict(orient="records")}


def load_frames():
    """{name: (v1 frame, CPW+SES frame)} for CWRU, Paderborn x4 and the WP5
    datasets, plus the CWRU source windows and arm a's scorer factory."""
    cwru_v1 = load_cache(Path("outputs/envelope_cache"), "v1")
    cwru_cpw = load_cache(Path("outputs/envelope_cache"), "hd_cpw_ses")
    scaler, clf, src = fit_source(cwru_v1)
    arm_a = lambda cal: ((lambda f: clf.predict_proba(scaler.transform(f[FEATURE_NAMES]))[:, 1]), None)

    data = {}
    for name, v1, cpw in [("cwru", cwru_v1[cwru_v1.dataset == "cwru"].copy(), cwru_cpw[cwru_cpw.dataset == "cwru"].copy())] + \
            [(d, load_wp5(d, "v1"), load_wp5(d, "hd_cpw_ses")) for d in ("ottawa_2018", "ottawa_2023", "mfpt")]:
        for frame in (v1, cpw):
            if "recording" not in frame:
                frame["recording"] = frame["unit"] + "_" + frame["run"].astype(str)
            if "state" not in frame:
                frame["state"] = np.where(frame.label > 0, "faulty", "healthy")
        data[name] = (v1, cpw)

    from td_cache import KEYS as TD_KEYS, load_td   # notes/18 §6.2: time-domain stats for CWRU/PU
    td = load_td()
    cwru_cpw = cwru_cpw.merge(td, on=TD_KEYS, how="left", validate="one_to_one")
    data["cwru"] = (data["cwru"][0], cwru_cpw[cwru_cpw.dataset == "cwru"].copy().assign(
        recording=lambda f: f.unit + "_" + f.run.astype(str),
        state=lambda f: np.where(f.label > 0, "faulty", "healthy")))
    pu_v1, pu_cpw = cwru_v1[cwru_v1.dataset == "pu"].copy(), cwru_cpw[cwru_cpw.dataset == "pu"].copy()
    for frame in (pu_v1, pu_cpw):
        frame["recording"] = frame["unit"] + "_" + frame["run"].astype(str)
        frame["state"] = np.where(frame.label > 0, "faulty", "healthy")
    for c in sorted(pu_v1.condition.unique()):
        data[f"pu:{c}"] = (pu_v1[pu_v1.condition == c], pu_cpw[pu_cpw.condition == c])

    return data, src, arm_a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("outputs/wp5"))
    args = ap.parse_args()

    data, src, arm_a = load_frames()
    results, drift = {}, {}
    for name, (v1, cpw) in data.items():
        drift[name] = drift_report(src, v1).to_dict() if name in ("ottawa_2018", "ottawa_2023", "mfpt") else {}
        baselines = ["machine", "own"] if name == "ottawa_2023" else ["machine"]
        for b in baselines:
            key = name if b == "machine" else f"{name}:own"
            results[key] = {}
            if name != "cwru":   # arm a was fit on CWRU
                results[key]["a"] = evaluate(name, v1, arm_a, b)
            results[key]["d_v1"] = evaluate(name, v1, physics, b, typing=True)
            results[key]["d"] = evaluate(name, cpw, physics, b, typing=True)
            results[key]["d_xf"] = evaluate(name, cpw, physics_xf, b, typing=True)   # post-hoc, notes/17 §5
            if "td_rms" in cpw:                                                    # notes/18 §6
                results[key]["L_rms"] = evaluate(name, cpw, zmax_xf(["td_rms"]), b)
                results[key]["K_kurt"] = evaluate(name, cpw, zmax_xf(["td_kurtosis"]), b)
                results[key]["dK"] = evaluate(name, cpw, combined_dK, b)

    # pre-registered gate (notes/17 §4)
    gate_sets = ["ottawa_2018", "ottawa_2023", "mfpt"]
    ok = [results[s]["d"]["summary"]["auc_race"] >= 0.80 and results[s]["d"]["summary"]["fa_mean"] <= 0.15
          for s in gate_sets]
    beats_a = [results[s]["d"]["summary"]["auc_race"] >= results[s]["a"]["summary"]["auc_race"] for s in gate_sets]
    bad = [results[s]["d"]["summary"]["auc_race"] < 0.70 for s in gate_sets]
    verdict = ("CONFIRMED" if sum(ok) >= 2 and sum(beats_a) >= 2 else
               "NOT CONFIRMED" if sum(bad) >= 2 else "PARTIAL")
    ok_xf = [results[s]["d_xf"]["summary"]["auc_race"] >= 0.80 and results[s]["d_xf"]["summary"]["fa_mean"] <= 0.15
             for s in gate_sets]
    own, mach = results["ottawa_2023:own"]["d"]["summary"], results["ottawa_2023"]["d"]["summary"]
    own_reading = own["fa_worst"] < mach["fa_worst"] and own["recall_race"] >= mach["recall_race"] - 0.05

    dk_rows = {k: (r["dK"]["summary"]["auc_race"], r["d_xf"]["summary"]["auc_race"], r["K_kurt"]["summary"]["auc_race"])
               for k, r in results.items() if "dK" in r}
    dk_ok = {k: (dk >= max(d, kk) - 0.05) or (dk >= d) for k, (dk, d, kk) in dk_rows.items()}
    L = ["# WP5 results: frozen physics score on new datasets", "",
         f"**notes/18 §6.4 combined score dK: {'ADOPTED' if all(dk_ok.values()) else 'not adopted'}** "
         f"(fails on: {[k for k, v in dk_ok.items() if not v] or 'none'}).", "",
         "Generated by `scripts/analysis/wp5_score.py`; protocol and gate pre-registered in notes/17. "
         f"Conformal threshold at alpha = {ALPHA}; race = inner + outer faults. "
         "Recall/FA/AUC are means over folds (worst fold in brackets).", "",
         f"**Gate (notes/17 §4): {verdict}.** Datasets passing AUC >= 0.80 & FA <= 0.15: "
         f"{[s for s, o in zip(gate_sets, ok) if o]}; arm d AUC >= arm a: {[s for s, o in zip(gate_sets, beats_a) if o]}.",
         f"Post-hoc, cross-fitted calibration (`d_xf`, notes/17 §5, not part of the gate): passing "
         f"{[s for s, o in zip(gate_sets, ok_xf) if o]}.",
         f"**Own-unit vs machine baseline (Ottawa 2023): {'supported' if own_reading else 'not supported'}** "
         f"(worst-unit FA {own['fa_worst']:.2f} vs {mach['fa_worst']:.2f}; race recall {own['recall_race']:.3f} vs {mach['recall_race']:.3f}).", "",
         "| dataset | arm | folds | race recall | FA (worst) | race AUC (worst) | recording detect / FA | typing | per type recall |",
         "|---|---|---|---|---|---|---|---|---|"]
    for key, arms in results.items():
        for arm, r in arms.items():
            s = r["summary"]
            types = ", ".join(f"{k[7:]} {v:.2f}" for k, v in s.items()
                              if k.startswith("recall_") and k[7:] in LABEL_NAMES.values())
            states = ", ".join(f"{k[12:]} {v:.2f}" for k, v in s.items() if k.startswith("recall_race_") and k != "recall_race_sd")
            ty = f"{s['typing']:.3f}" if "typing" in s and not np.isnan(s["typing"]) else ""
            L.append(f"| {key} | {arm} | {int(s['folds'])} | {s['recall_race']:.3f} ± {s['recall_race_sd']:.3f} | "
                     f"{s['fa_mean']:.3f} ({s['fa_worst']:.2f}) | {s['auc_race']:.3f} ({s['auc_race_min']:.2f}) | "
                     f"{s['rec_detect_race']:.2f} / {s['rec_fa']:.2f} | {ty} | {types}{'; ' + states if states else ''} |")
    L += ["", "## Drift report (§4 rule 7): standardised mean difference, CWRU healthy vs target healthy, v1 features", "",
          "| feature | " + " | ".join(k for k in drift if drift[k]) + " |", "|---|" + "---|" * sum(1 for k in drift if drift[k])]
    for feat in FEATURE_NAMES:
        L.append(f"| {feat} | " + " | ".join(f"{drift[k][feat]:+.2f}" for k in drift if drift[k]) + " |")
    md = "\n".join(L) + "\n"
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "wp5_report.md").write_text(md)
    (args.out / "wp5_results.json").write_text(json.dumps({"verdict": verdict, "own_reading": own_reading,
                                                            "dK": {"rows": dk_rows, "ok": dk_ok},
                                                            "results": results, "drift": drift}, indent=1, default=float))
    print(md)


if __name__ == "__main__":
    main()
