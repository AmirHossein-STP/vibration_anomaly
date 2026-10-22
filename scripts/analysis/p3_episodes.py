"""Phase 3 A0: the episode bank for Track A (notes/29 §2), and method v1.1 scored on it.

One episode = one calibration fold of one development dataset:
    support   healthy calibration windows (their cross-fitted z, exactly as fit_baseline)
    neg       held-out healthy windows
    pos       race-involving fault windows (run-to-failure: end of life)
    bc        ball/cage fault windows (reported, never trained on)
Every array holds z-scores of the 135 geometry-free bins against the support, so a
learned scorer and the v1.1 max-z see the same numbers.

Usage (box): python3 scripts/analysis/p3_episodes.py      # builds the bank + parity report
Output: outputs/p3/bank.pkl, outputs/p3/a0_reference.json, outputs/p3/a0_report.md
"""
from __future__ import annotations

import json
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))
from domain_adaptation.data.public_datasets import RACE_INVOLVING
from domain_adaptation.features.geofree import BIN_NAMES
from domain_adaptation.method import conformal_threshold, default_groups, fit_baseline, log_lines
from v2_cache import load_v2
from wp5_score import folds as wp5_folds

DATASETS = ["cwru", "pu", "ottawa_2018", "ottawa_2023", "mfpt", "hust", "xjtu_sy", "paderborn_ldm", "ims"]
OUT = Path("outputs/p3")
IMS_CACHE = OUT / "cache" / "ims_geofree.csv.gz"
CAP = 3000                  # run-to-failure parts only (no committed reference to match there)
BC = (3, 4)
# run-to-failure folds (notes/29 §2): support records, healthy segment end, "long" unit, end-of-life share
RTF = {"xjtu_sy": dict(base=20, healthy_end=60, long=120, tail=0.10),          # notes/18
       "paderborn_ldm": dict(base=150, healthy_end=450, long=1000, tail=0.10),  # notes/25
       "ims": dict(base=200, healthy_end=None, long=0, tail=0.05)}              # notes/27; healthy to the 50% point


@dataclass
class Fold:
    dataset: str
    name: str
    mu: np.ndarray            # (135,) support mean of log bins
    logsd: np.ndarray         # (135,) log support sd
    sup: np.ndarray           # (n, 135) cross-fitted z of the support windows
    neg: np.ndarray
    pos: np.ndarray
    bc: np.ndarray
    info: dict = field(default_factory=dict)


def crossfit_z(lc: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """z of each support window against the support without its group (fit_baseline's rule)."""
    out = np.empty_like(lc)
    for g in np.unique(groups):
        m = groups == g
        mu, sd = lc[~m].mean(0), lc[~m].std(0) + 1e-12
        out[m] = (lc[m] - mu) / sd
    return out


def make_fold(dataset, name, cal, held, pos, bc, info=None) -> Fold:
    X = lambda f: f[BIN_NAMES].to_numpy()
    groups = default_groups(cal.recording.astype(str).to_numpy(), cal.t0.to_numpy())
    b = fit_baseline(X(cal), groups)
    sup = crossfit_z(log_lines(X(cal)), groups)
    assert np.allclose(sup.max(1), b.cal_scores), "cross-fitted z disagrees with fit_baseline"
    z = lambda f: (b.z(X(f)) if len(f) else np.zeros((0, len(BIN_NAMES)))).astype(np.float32)
    return Fold(dataset, name, b.mu.astype(np.float32), np.log(b.sd).astype(np.float32), sup.astype(np.float32),
                z(held), z(pos), z(bc), info or {})


def load(dataset: str) -> pd.DataFrame:
    df = pd.read_csv(IMS_CACHE) if dataset == "ims" else load_v2(dataset, fset="geofree")
    if "recording" not in df:
        df["recording"] = df["unit"] + "_" + df["run"].astype(str)
    return df


def cap(f: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    return f if len(f) <= CAP else f.iloc[np.sort(rng.choice(len(f), CAP, replace=False))]


def static_folds(dataset: str, df: pd.DataFrame) -> list[Fold]:
    out = []
    if dataset == "hust":                                   # notes/24: per type x load
        for t in sorted(df.unit.unique()):
            d = df[df.unit == t]
            h = d[d.label == 0]
            for load_ in sorted(d.condition.unique()):
                f = d[(d.label > 0) & (d.condition == load_)]
                out.append(make_fold(dataset, f"{t}:{load_}", h[h.condition != load_], h[h.condition == load_],
                                     f[f.label.isin(RACE_INVOLVING)], f[f.label.isin(BC)], {"type": str(t)}))
        return out
    parts = {f"pu:{c}": df[df.condition == c] for c in sorted(df.condition.unique())} if dataset == "pu" else {dataset: df}
    modes = ["own", "machine"] if dataset == "ottawa_2023" else ["machine"]
    for pname, frame in parts.items():
        if "state" not in frame:
            frame = frame.assign(state=np.where(frame.label > 0, "faulty", "healthy"))
        for mode in modes:
            for name, cal, held, fault in wp5_folds(pname, frame, mode):
                out.append(make_fold(dataset, f"{pname}:{mode}:{name}", cal, held, fault[fault.label.isin((1, 2))],
                                     fault[fault.label.isin(BC)], {"part": pname, "baseline": mode}))
    return out


def rtf_folds(dataset: str, df: pd.DataFrame) -> list[Fold]:
    r, rng, out = RTF[dataset], np.random.default_rng(0), []
    if dataset == "paderborn_ldm":
        df = df[(df.channel == "accHorizRear_A") & (df.set_rpm >= 1000)]
        key = "record"
    elif dataset == "xjtu_sy":
        df = df[df.channel == "horizontal"]
        key = "minute"
    else:
        key = "record"
    for u in sorted(df.unit.unique()):
        d = df[df.unit == u].sort_values([key, "win"])
        recs = np.sort(d[key].unique())
        n = len(recs)
        base = recs[:r["base"]]
        end = (n // 2) if r["healthy_end"] is None else r["healthy_end"]
        healthy = recs[r["base"]:end] if n >= r["long"] and end > r["base"] else recs[:0]
        failed = dataset != "ims" or bool(d.failed.fillna("").astype(str).iloc[0])
        tail = recs[-max(1, int(np.ceil(r["tail"] * n))):] if failed else recs[:0]
        none = d.iloc[:0]
        out.append(make_fold(dataset, str(u), d[d[key].isin(base)], cap(d[d[key].isin(healthy)], rng),
                             cap(d[d[key].isin(tail)], rng), none,
                             {"n_records": int(n), "failed": failed, "long": bool(len(healthy))}))
    return out


def build() -> dict[str, list[Fold]]:
    bank = {}
    for d in DATASETS:
        df = load(d)
        bank[d] = rtf_folds(d, df) if d in RTF else static_folds(d, df)
        print(f"[{d}] {len(bank[d])} folds", flush=True)
    return bank


def fold_metrics(score, fold: Fold) -> dict:
    """score: (n, 135) z -> (n,) scores. AUC neg vs pos, FA at the conformal threshold."""
    thr = conformal_threshold(score(fold.sup))
    sn, sp = score(fold.neg), score(fold.pos)
    r = {"fa": float((sn > thr).mean()) if len(sn) else np.nan, "recall": float((sp > thr).mean()) if len(sp) else np.nan}
    r["auc"] = float(roc_auc_score(np.r_[np.zeros(len(sn)), np.ones(len(sp))], np.r_[sn, sp])) if len(sn) and len(sp) else np.nan
    if len(fold.bc) and len(sn):
        sb = score(fold.bc)
        r["auc_bc"] = float(roc_auc_score(np.r_[np.zeros(len(sn)), np.ones(len(sb))], np.r_[sn, sb]))
    return r


def fold_row(f: Fold, score) -> dict:
    return {"name": f.name, "part": f.info.get("part", f.info.get("type", f.dataset)),
            "baseline": f.info.get("baseline", "own"), **fold_metrics(score, f)}


def dataset_summary(bank_d: list[Fold], score) -> dict:
    return summarise_rows([fold_row(f, score) for f in bank_d])


def summarise_rows(rows: list[dict]) -> dict:
    """Primary rows only (ottawa_2023: own baseline). PU: mean of the 4 condition means."""
    rows = pd.DataFrame(rows)
    prim = rows[rows.baseline != "machine"] if (rows.baseline == "own").any() else rows
    by_part = prim.groupby("part").agg(auc=("auc", "mean"), fa=("fa", "mean"), fa_worst=("fa", "max"))
    s = {"auc": float(by_part.auc.mean()), "fa": float(by_part.fa.mean()), "fa_worst": float(by_part.fa_worst.mean()),
         "parts": by_part.to_dict(orient="index")}
    if "auc_bc" in rows:
        s["auc_bc"] = float(prim.auc_bc.mean())
    if (rows.baseline == "machine").any() and (rows.baseline == "own").any():
        s["machine_baseline_auc"] = float(rows[rows.baseline == "machine"].auc.mean())
    return s


def zmax(z: np.ndarray) -> np.ndarray:
    return z.max(1)


def main():
    bank = build()
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "bank.pkl", "wb") as f:
        pickle.dump(bank, f, protocol=pickle.HIGHEST_PROTOCOL)
    ref = {d: dataset_summary(bank[d], zmax) for d in DATASETS}
    (OUT / "a0_reference.json").write_text(json.dumps(ref, indent=1, default=float))
    # parity against the committed WP13 / WP14 geometry-free numbers (notes/29 §5)
    wp13 = json.loads(Path("outputs/wp13/wp13_results.json").read_text())["results"]
    wp14 = json.loads(Path("outputs/wp14/wp14_results.json").read_text())
    wp14 = wp14.get("results", wp14)["geometry_free"]
    checks = []
    for key, v in wp13.items():
        d = key.split(":")[0]
        if key == "ottawa_2023":
            mine = ref[d]["machine_baseline_auc"]
        elif key.startswith("pu:"):
            mine = ref["pu"]["parts"][key]["auc"]
        else:
            mine = ref[d]["auc"]
        checks.append((key, v["geofree"]["auc_race"], mine))
    for t, v in wp14.items():
        checks.append((f"hust:{t}", v["auc_race"], ref["hust"]["parts"][t]["auc"]))
    ok = all(abs(a - b) < 1e-9 for _, a, b in checks)
    L = ["# Phase 3 A0: episode bank and the v1.1 reference", "",
         "Generated by `scripts/analysis/p3_episodes.py` (notes/29 §2, §5).", "",
         f"**Parity with committed WP13/WP14 geometry-free AUCs: {'PASS' if ok else 'FAIL'}** ({len(checks)} numbers)", "",
         "| row | committed | episode bank |", "|---|---|---|"]
    L += [f"| {k} | {a:.6f} | {b:.6f} |" for k, a, b in checks]
    L += ["", "## v1.1 (max z) per dataset, primary folds", "",
          "| dataset | folds | AUC | FA mean | FA worst (mean over parts) | ball/cage AUC |", "|---|---|---|---|---|---|"]
    for d in DATASETS:
        s = ref[d]
        L.append(f"| {d} | {len(bank[d])} | {s['auc']:.3f} | {s['fa']:.3f} | {s['fa_worst']:.3f} | {s.get('auc_bc', float('nan')):.3f} |")
    md = "\n".join(L) + "\n"
    (OUT / "a0_report.md").write_text(md)
    print(md)
    if not ok:
        raise SystemExit("A0 parity FAILED")


if __name__ == "__main__":
    main()
