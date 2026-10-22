"""Phase 3 Track A: meta-learned comparison with a healthy reference (notes/29, pre-registered).

Leave one dataset out over the 9 development sets; for each held-out set, train on episodes of
the other 8 and score the held-out set's folds. Both models are max-z plus a learned head whose
last layer starts at zero, so at step 0 they ARE method v1.1. Training moves them away only
where the source datasets reward it.

    A1  rank pool    MLP on the 32 largest z (sorted) + mean, sd of z     (bin-agnostic)
    A2  bin transformer over 135 tokens [z, mu, log sd, order code]       (set-conditioned)

Usage (box, after p3_episodes.py): python3 scripts/analysis/p3_meta.py [--models a1 a2] [--seeds 0 1 2 3 4]
(one worker process per model x seed, --jobs at a time; the GPU is idle at these batch sizes)
Output: outputs/p3/meta/results.json, outputs/p3/meta/report.md
"""
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))
from domain_adaptation.features.geofree import BIN_CENTRES
from p3_episodes import DATASETS, OUT, Fold, fold_row, summarise_rows

STEPS, EPISODES, PER_SIDE, LR, WD = 3000, 8, 64, 1e-3, 1e-4      # notes/29 §3, fixed
TOPK, ZCLIP = 32, 20.0
GATE_MEAN, GATE_LOSS, GATE_FA = 0.02, 0.02, 0.02                  # notes/29 §6
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def order_code(n: int = 8) -> torch.Tensor:
    """Sinusoidal code of each bin's order value (1.5-15 orders), (135, n)."""
    o = torch.tensor(BIN_CENTRES, dtype=torch.float32)[:, None]
    freqs = torch.tensor([0.25, 0.5, 1.0, 2.0])[None, :]
    return torch.cat([torch.sin(o * freqs * np.pi), torch.cos(o * freqs * np.pi)], 1)[:, :n]


class A1(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(TOPK + 2, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, z, mu, logsd):
        zc = z.clamp(-ZCLIP, ZCLIP)
        top = zc.topk(TOPK, dim=1).values
        x = torch.cat([top, zc.mean(1, keepdim=True), zc.std(1, keepdim=True)], 1)
        return z.max(1).values + self.mlp(x).squeeze(1)


class A2(nn.Module):
    def __init__(self, d: int = 32):
        super().__init__()
        self.register_buffer("code", order_code())
        self.inp = nn.Linear(3 + self.code.shape[1], d)
        layer = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.att = nn.Linear(d, 1)
        self.head = nn.Linear(d, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, z, mu, logsd):
        n = z.shape[0]
        tok = torch.cat([z.clamp(-ZCLIP, ZCLIP)[..., None], mu.expand(n, -1)[..., None] / 10,
                         logsd.expand(n, -1)[..., None], self.code.expand(n, -1, -1)], 2)
        h = self.enc(self.inp(tok))
        w = torch.softmax(self.att(h), 1)
        return z.max(1).values + self.head((w * h).sum(1)).squeeze(1)


MODELS = {"a1": A1, "a2": A2}


def tensors(f: Fold) -> dict:
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(DEV)
    return {"mu": t(f.mu)[None], "logsd": t(f.logsd)[None], "neg": t(np.r_[f.neg, f.sup]), "pos": t(f.pos)}


def train(model: nn.Module, train_sets: dict[str, list[dict]], seed: int) -> nn.Module:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    names = [d for d, fs in train_sets.items() if fs]
    model.train()
    for step in range(STEPS):
        loss = 0.0
        for _ in range(EPISODES):
            fs = train_sets[names[rng.integers(len(names))]]
            e = fs[rng.integers(len(fs))]
            neg = e["neg"][torch.from_numpy(rng.integers(len(e["neg"]), size=PER_SIDE)).to(DEV)]
            pos = e["pos"][torch.from_numpy(rng.integers(len(e["pos"]), size=PER_SIDE)).to(DEV)]
            s = model(torch.cat([neg, pos]), e["mu"], e["logsd"])
            loss = loss + F.softplus(-(s[PER_SIDE:, None] - s[None, :PER_SIDE])).mean()
        opt.zero_grad()
        (loss / EPISODES).backward()
        opt.step()
    model.eval()
    return model


def scorer(model: nn.Module, f: Fold):
    mu = torch.from_numpy(f.mu).to(DEV)[None]
    logsd = torch.from_numpy(f.logsd).to(DEV)[None]

    @torch.no_grad()
    def score(z: np.ndarray) -> np.ndarray:
        if len(z) == 0:
            return np.zeros(0)
        out = [model(torch.from_numpy(np.ascontiguousarray(z[i:i + 4096], dtype=np.float32)).to(DEV), mu, logsd)
               for i in range(0, len(z), 4096)]
        return torch.cat(out).double().cpu().numpy()
    return score


def summary_per_fold(folds: list[Fold], per: dict) -> dict:
    """dataset_summary with one scorer per fold (each reads its own fold's support)."""
    return summarise_rows([fold_row(f, per[id(f)]) for f in folds])


def gate(ref: dict, res: dict) -> dict:
    d_auc = {d: res[d]["auc"] - ref[d]["auc"] for d in DATASETS}
    d_fa = np.mean([res[d]["fa_worst"] - ref[d]["fa_worst"] for d in DATASETS])
    c1 = float(np.mean(list(d_auc.values())))
    worst = min(d_auc, key=d_auc.get)
    return {"mean_delta_auc": c1, "worst_dataset": worst, "worst_delta": d_auc[worst],
            "delta_fa_worst": float(d_fa), "per_dataset_delta": d_auc,
            "pass": bool(c1 >= GATE_MEAN and d_auc[worst] >= -GATE_LOSS and d_fa <= GATE_FA)}


def run_one(m: str, seed: int, out_dir: Path) -> None:
    """One model x seed over all 9 held-out datasets -> out_dir/runs/<m>_<seed>.json"""
    torch.set_num_threads(1)
    with open(OUT / "bank.pkl", "rb") as fh:
        bank: dict[str, list[Fold]] = pickle.load(fh)
    ref = json.loads((OUT / "a0_reference.json").read_text())
    trainable = {d: [tensors(f) for f in fs if len(f.pos) and len(f.neg) + len(f.sup)] for d, fs in bank.items()}
    res, t0 = {}, time.time()
    for held in DATASETS:
        train_sets = {d: v for d, v in trainable.items() if d != held}
        assert held not in train_sets                       # leakage guard (notes/29 §2)
        model = train(MODELS[m]().to(DEV), train_sets, seed * 100 + DATASETS.index(held))
        res[held] = summary_per_fold(bank[held], {id(f): scorer(model, f) for f in bank[held]})
    run = {"datasets": res, "gate": gate(ref, res), "seconds": time.time() - t0}
    (out_dir / "runs").mkdir(parents=True, exist_ok=True)
    (out_dir / "runs" / f"{m}_{seed}.json").write_text(json.dumps(run, indent=1, default=float))
    g = run["gate"]
    print(f"{m} seed {seed}: mean dAUC {g['mean_delta_auc']:+.3f}, worst {g['worst_dataset']} "
          f"{g['worst_delta']:+.3f}, dFA_worst {g['delta_fa_worst']:+.3f} ({run['seconds']:.0f} s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--steps", type=int, default=STEPS, help="smoke tests only; the result uses the default")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--worker", nargs=2, metavar=("MODEL", "SEED"))
    a = ap.parse_args()
    globals()["STEPS"] = a.steps
    out_dir = OUT / "meta"
    if a.worker:
        run_one(a.worker[0], int(a.worker[1]), out_dir)
        return
    todo = [(m, s) for m in a.models for s in a.seeds]
    procs, failed = [], []
    while todo or procs:
        while todo and len(procs) < a.jobs:
            m, s = todo.pop(0)
            procs.append(((m, s), subprocess.Popen([sys.executable, __file__, "--steps", str(a.steps), "--worker", m, str(s)])))
        time.sleep(5)
        for p in [p for p in procs if p[1].poll() is not None]:
            procs.remove(p)
            if p[1].returncode:
                failed.append(p[0])
    if failed:
        raise SystemExit(f"workers failed: {failed}")
    ref = json.loads((OUT / "a0_reference.json").read_text())
    results = {"steps": STEPS, "device": str(DEV),
               "runs": {f"{m}:{s}": json.loads((out_dir / "runs" / f"{m}_{s}.json").read_text())
                        for m in a.models for s in a.seeds}}
    (out_dir / "results.json").write_text(json.dumps(results, indent=1, default=float))
    report(ref, results, a.models, a.seeds, out_dir)


def report(ref: dict, results: dict, models: list[str], seeds: list[int], out_dir: Path) -> None:
    L = ["# Phase 3 Track A: meta-learned comparison (notes/29)", "",
         f"Generated by `scripts/analysis/p3_meta.py`; {results['steps']} steps; seeds {seeds}; leave one dataset out.", ""]
    for m in models:
        runs = [results["runs"][f"{m}:{s}"] for s in seeds]
        d_mean = np.array([r["gate"]["mean_delta_auc"] for r in runs])
        mean_res = {d: {k: float(np.mean([r["datasets"][d][k] for r in runs])) for k in ("auc", "fa", "fa_worst")}
                    for d in DATASETS}
        g = gate(ref, mean_res)
        L += [f"## {m.upper()}: **gate {'PASS' if g['pass'] else 'FAIL'}** "
              f"(5-seed means: mean dAUC {g['mean_delta_auc']:+.3f}, need >= +{GATE_MEAN}; "
              f"worst {g['worst_dataset']} {g['worst_delta']:+.3f}, need >= -{GATE_LOSS}; "
              f"d FA worst {g['delta_fa_worst']:+.3f}, need <= +{GATE_FA})", "",
              f"Per-seed mean dAUC: {', '.join(f'{x:+.3f}' for x in d_mean)} (sd {d_mean.std(ddof=1) if len(d_mean) > 1 else 0:.3f})", "",
              "| held-out dataset | v1.1 AUC | model AUC (sd) | dAUC | v1.1 FA worst | model FA worst |", "|---|---|---|---|---|---|"]
        for d in DATASETS:
            aucs = np.array([r["datasets"][d]["auc"] for r in runs])
            L.append(f"| {d} | {ref[d]['auc']:.3f} | {aucs.mean():.3f} ({aucs.std(ddof=1) if len(aucs) > 1 else 0:.3f}) | "
                     f"{aucs.mean() - ref[d]['auc']:+.3f} | {ref[d]['fa_worst']:.3f} | {mean_res[d]['fa_worst']:.3f} |")
        L.append("")
    md = "\n".join(L) + "\n"
    (out_dir / "report.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
