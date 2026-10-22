"""Phase 3 Track B: MOMENT-1 embeddings + own-baseline kNN (notes/31, pre-registered).

Embeds the geometry-free windows of the six static development sets with MOMENT-1-base (chunks of
512 samples, averaged per window), then scores Track A's exact folds by the mean cosine distance
to the 5 nearest support embeddings, cross-fitted. B-ctrl: the same kNN on the geometry-free z.

Usage (box; venv ~/work/venv_fm, remote/requirements-p3b.txt):
    ~/work/venv_fm/bin/python scripts/analysis/p3_fm.py
Output: outputs/p3/fm/{<dataset>_moment.npz, results.json, report.md}
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))
import v2_cache
from domain_adaptation.features.geofree import BIN_NAMES
from domain_adaptation.method import conformal_threshold, default_groups, log_lines
from p3_episodes import OUT, static_parts, summarise_rows
from v2_cache import load_v2

DATASETS = ["cwru", "pu", "ottawa_2018", "ottawa_2023", "mfpt", "hust"]
FM_OUT = OUT / "fm"
CHUNK, K = 512, 5
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODEL = None


def model():
    global _MODEL
    if _MODEL is None:
        from momentfm import MOMENTPipeline
        _MODEL = MOMENTPipeline.from_pretrained("AutonLab/MOMENT-1-base", model_kwargs={"task_name": "embedding"})
        _MODEL.init()
        _MODEL.to(DEV).eval()
    return _MODEL


@torch.no_grad()
def embed(x: np.ndarray) -> np.ndarray:
    n = len(x) // CHUNK
    chunks = torch.from_numpy(np.ascontiguousarray(x[:n * CHUNK].reshape(n, 1, CHUNK), dtype=np.float32)).to(DEV)
    return model()(x_enc=chunks).embeddings.mean(0).double().cpu().numpy()


EMB = [f"e{i:03d}" for i in range(768)]


def window_fn(x, fs, rpm, g):
    return dict(zip(EMB, embed(x)))


def window_fn_cot(x, fs, phase, g, spr):
    return dict(zip(EMB, embed(x)))


def embed_dataset(d: str) -> pd.DataFrame:
    """Same windows as the geometry-free cache (v2_cache._job), embedded in-process on the GPU."""
    p = FM_OUT / f"{d}_moment.npz"
    if p.is_file():
        z = np.load(p, allow_pickle=True)
        return pd.concat([pd.DataFrame(z["meta"], columns=list(z["meta_cols"])), pd.DataFrame(z["emb"], columns=EMB)], axis=1)
    v2_cache.FEATURE_SETS["moment"] = (window_fn, window_fn_cot, EMB, FM_OUT)
    t0, rows = time.time(), []
    for k in v2_cache.keys(d):
        rows += v2_cache._job((d, k, "moment"))
    df = pd.DataFrame(rows)
    meta_cols = [c for c in df.columns if c not in EMB]
    FM_OUT.mkdir(parents=True, exist_ok=True)
    np.savez(p, meta=df[meta_cols].to_numpy(dtype=object), meta_cols=np.array(meta_cols), emb=df[EMB].to_numpy(np.float32))
    print(f"[{d}] {len(df)} windows embedded in {time.time() - t0:.0f} s", flush=True)
    return df


def knn_dist(q: np.ndarray, ref: np.ndarray) -> np.ndarray:
    qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    rn = ref / (np.linalg.norm(ref, axis=1, keepdims=True) + 1e-12)
    d = 1 - qn @ rn.T
    return np.sort(d, axis=1)[:, :K].mean(1)


def fold_row(name, cal, held, pos, info, feats) -> dict:
    """feats(frame) -> (n, d) array; for B-ctrl the log bins, z-scored by the support."""
    groups = default_groups(cal.recording.astype(str).to_numpy(), cal.t0.to_numpy())
    C = feats(cal)
    cal_scores = np.empty(len(C))
    for g in np.unique(groups):
        m = groups == g
        cal_scores[m] = knn_dist(C[m], C[~m])
    thr = conformal_threshold(cal_scores)
    sn, sp = knn_dist(feats(held), C), knn_dist(feats(pos), C) if len(pos) else np.zeros(0)
    r = {"name": name, "part": info.get("part", info.get("type")), "baseline": info.get("baseline", "own"),
         "fa": float((sn > thr).mean()), "recall": float((sp > thr).mean()) if len(sp) else np.nan,
         "auc": float(roc_auc_score(np.r_[np.zeros(len(sn)), np.ones(len(sp))], np.r_[sn, sp])) if len(sp) else np.nan}
    return r


def evaluate(d: str, df: pd.DataFrame, cols: list[str], zscore: bool) -> dict:
    rows = []
    for name, cal, held, pos, bc, info in static_parts(d, df):
        if zscore:      # B-ctrl: geometry-free log bins z-scored by the support (the v1.1 input)
            lc = log_lines(cal[cols].to_numpy())
            mu, sd = lc.mean(0), lc.std(0) + 1e-12
            feats = lambda f, mu=mu, sd=sd: (log_lines(f[cols].to_numpy()) - mu) / sd
        else:
            feats = lambda f: f[cols].to_numpy(np.float64)
        rows.append(fold_row(name, cal, held, pos, info, feats))
    return summarise_rows(rows)


def main():
    ref = json.loads((OUT / "a0_reference.json").read_text())
    res = {}
    for d in DATASETS:
        emb = embed_dataset(d)
        geo = load_v2(d, fset="geofree")
        for frame in (emb, geo):
            if "recording" not in frame:
                frame["recording"] = frame["unit"] + "_" + frame["run"].astype(str)
            frame["label"] = frame["label"].astype(int)
            frame["t0"] = frame["t0"].astype(float)
        res[d] = {"moment": evaluate(d, emb, EMB, False), "bctrl": evaluate(d, geo, BIN_NAMES, True)}
        print(f"[{d}] v1.1 {ref[d]['auc']:.3f}  MOMENT {res[d]['moment']['auc']:.3f}  B-ctrl {res[d]['bctrl']['auc']:.3f}", flush=True)
    m = {k: float(np.mean([res[d][k]["auc"] for d in DATASETS])) for k in ("moment", "bctrl")}
    v = float(np.mean([ref[d]["auc"] for d in DATASETS]))
    worst = min(res[d]["moment"]["auc"] - ref[d]["auc"] for d in DATASETS)
    reading = ("referencing carried the transfer" if m["moment"] >= v - 0.02 and worst >= -0.10 else
               "the representation matters" if m["moment"] < v - 0.05 else "in between")
    (FM_OUT / "results.json").write_text(json.dumps({"results": res, "mean": m, "v11_mean": v, "worst_delta": worst,
                                                     "reading": reading}, indent=1, default=float))
    L = ["# Phase 3 Track B: MOMENT-1 embeddings + own-baseline kNN (notes/31)", "",
         f"Generated by `scripts/analysis/p3_fm.py`. **Reading: {reading}** "
         f"(mean AUC v1.1 {v:.3f}, MOMENT {m['moment']:.3f}, B-ctrl {m['bctrl']:.3f}; worst MOMENT - v1.1 {worst:+.3f}).", "",
         "| dataset | v1.1 AUC | MOMENT kNN AUC | B-ctrl kNN on geometry-free z | v1.1 FA worst | MOMENT FA worst |", "|---|---|---|---|---|---|"]
    for d in DATASETS:
        L.append(f"| {d} | {ref[d]['auc']:.3f} | {res[d]['moment']['auc']:.3f} | {res[d]['bctrl']['auc']:.3f} | "
                 f"{ref[d]['fa_worst']:.3f} | {res[d]['moment']['fa_worst']:.3f} |")
    md = "\n".join(L) + "\n"
    (FM_OUT / "report.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
