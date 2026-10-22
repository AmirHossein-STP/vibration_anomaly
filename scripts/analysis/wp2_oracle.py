"""MASTER_PLAN WP2, post-hoc diagnostic (not pre-registered): is the
1500 rpm information missing from the features, or is it there and the
CWRU-learned boundary misses it?

Window ROC-AUC (real damage vs healthy) for v1 features, three fits:
- CWRU -> PU (the deployed model; label-free on PU)
- PU N09 -> PU condition c (same rig, other speed), units disjoint
- PU c -> PU c (in-condition, grouped by unit: an upper bound on what a
  linear model can get from these features at that condition)
The two PU fits use Paderborn labels and are diagnostics only: nothing
here selects a feature, band or threshold for the method.

Usage: python scripts/analysis/wp2_oracle.py [--variant v1]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, str(Path(__file__).parent))
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from domain_adaptation.features.envelope import FEATURE_NAMES
from envelope_cache import load_cache
from wp1_rescore import CONDITIONS, CWRU_SECONDS


def model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))


def unit_halves(units, seed):
    u = np.array(sorted(units)); np.random.default_rng(seed).shuffle(u)
    return set(u[: len(u) // 2]), set(u[len(u) // 2:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="v1")
    args = ap.parse_args()
    df = load_cache(Path("outputs/envelope_cache"), args.variant)
    src = df[(df.dataset == "cwru") & (df.t0 < CWRU_SECONDS - 1e-9)]
    pu = df[(df.dataset == "pu") & (df.damage_mode != "artificial")]
    pu = pu.assign(y=(pu.label > 0).astype(int))
    m = model().fit(src[FEATURE_NAMES], (src.label > 0).astype(int))
    print(f"variant {args.variant}: window ROC-AUC, real damage vs healthy (mean ± sd over 20 unit splits)\n")
    print(f"{'condition':<13} {'CWRU->PU':>9} {'PU N09->c':>16} {'PU c->c':>16}")
    for c in CONDITIONS:
        t = pu[pu.condition == c]
        a_src = roc_auc_score(t.y, m.predict_proba(t[FEATURE_NAMES])[:, 1])
        cross, within = [], []
        for seed in range(20):
            # disjoint units within each class, so no bearing is in both fit and test
            hA, hB = unit_halves(t[t.y == 0].unit.unique(), seed)
            fA, fB = unit_halves(t[t.y == 1].unit.unique(), seed)
            fit_u, test_u = hA | fA, hB | fB
            n09 = pu[(pu.condition == "N09_M07_F10") & pu.unit.isin(fit_u)]
            test = t[t.unit.isin(test_u)]
            cross.append(roc_auc_score(test.y, model().fit(n09[FEATURE_NAMES], n09.y).predict_proba(test[FEATURE_NAMES])[:, 1]))
            fit = t[t.unit.isin(fit_u)]
            within.append(roc_auc_score(test.y, model().fit(fit[FEATURE_NAMES], fit.y).predict_proba(test[FEATURE_NAMES])[:, 1]))
        print(f"{c:<13} {a_src:>9.3f} {np.mean(cross):>8.3f} ± {np.std(cross):.3f} {np.mean(within):>8.3f} ± {np.std(within):.3f}")


if __name__ == "__main__":
    main()
