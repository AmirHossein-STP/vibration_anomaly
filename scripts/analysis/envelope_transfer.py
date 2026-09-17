"""Does an envelope-order representation cross rigs where spectrograms cannot?

Background: every spectrogram-based cross-dataset run in this project
failed the same way -- the model calls every target window faulty (11/11
strict runs collapsed; arm B failed identically with a 4x richer source;
per-window normalization, which provably removes the scale mismatch, did
not help). See notes/12_reproduction_investigation.md.

The diagnosis was physical: a log-spectrogram is dominated by the machine's
structural resonances, which live at fixed Hz and belong to the RIG. This
script tests the alternative -- envelope demodulation read at each
bearing's own fault ORDERS, which belong to the BEARING (see
src/domain_adaptation/features/envelope.py).

The test is the strict one the project cares about: fit on CWRU only,
never look at a Paderborn label, then detect faults on Paderborn.

Usage: python scripts/analysis/envelope_transfer.py [--max-files N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from domain_adaptation.data.cwru_loader import build_cwru_catalog
from domain_adaptation.data.pu_loader import PreparedPUStore
from domain_adaptation.features.envelope import (
    CWRU_6205, FEATURE_NAMES, PU_6203, envelope_order_features,
)

FS = 12000
WIN = 12000          # 1 second -- long enough to resolve ~0.1 order at 900rpm
HOP = 6000


def windows(sig: np.ndarray, win: int = WIN, hop: int = HOP, limit: int | None = None):
    n = 0
    for s in range(0, len(sig) - win + 1, hop):
        yield sig[s:s + win]
        n += 1
        if limit and n >= limit:
            return


def featurize(sig, rpm, geom, limit=None) -> list[dict]:
    return [envelope_order_features(w, FS, rpm, geom) for w in windows(sig, limit=limit)]


def build_cwru(max_files=None, per_file=8):
    cat = build_cwru_catalog(Path("data/raw"), [1730, 1750, 1772], 1797)
    if max_files:
        cat = cat.groupby("universal_label", group_keys=False).head(max_files)
    rows, labels = [], []
    for r in cat.itertuples():
        z = np.load(r.filepath)
        sig = np.asarray(z["DE"]).squeeze().astype(np.float64)
        for f in featurize(sig, r.rpm, CWRU_6205, limit=per_file):
            rows.append(f); labels.append(r.universal_label)
    return pd.DataFrame(rows)[FEATURE_NAMES], np.array(labels)


def build_pu(per_run=4):
    from domain_adaptation.data import pu_bearing_codes as pbc
    store = PreparedPUStore(Path("data/pu/prepared_pu"))
    # healthy + REAL damage only: artificial (filed/drilled) defects are not
    # what a deployed machine produces, and the project's target has always
    # been the real-damage bearings.
    codes = pbc.HEALTHY + pbc.REAL_OUTER + pbc.REAL_INNER
    cat = store.catalog(codes)
    cat = cat[cat.condition == "N09_M07_F10"]
    rows, labels, codes = [], [], []
    for r in cat.itertuples():
        sig = np.asarray(store.load(r.code, r.condition, r.run)).squeeze().astype(np.float64)
        for f in featurize(sig, r.rpm, PU_6203, limit=per_run):
            rows.append(f); labels.append(int(r.universal_label)); codes.append(r.code)
    return pd.DataFrame(rows)[FEATURE_NAMES], np.array(labels), np.array(codes)


def report(name, y_true, y_pred):
    ft, fp_ = y_true > 0, y_pred > 0
    tp = int((ft & fp_).sum()); fn = int((ft & ~fp_).sum())
    fp = int((~ft & fp_).sum()); tn = int((~ft & ~fp_).sum())
    rec = tp / max(tp + fn, 1); fa = fp / max(fp + tn, 1); prec = tp / max(tp + fp, 1)
    bal = (rec + (1 - fa)) / 2
    print(f"  {name:>22}: recall {rec:.3f}  precision {prec:.3f}  FA {fa:6.1%}  "
          f"balanced {bal:.3f}   (healthy found {tn}/{tn + fp})")
    return bal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-files", type=int, default=None)
    args = ap.parse_args()

    print("Extracting CWRU (source) envelope-order features...")
    Xs, ys = build_cwru(max_files=args.max_files)
    print(f"  {Xs.shape[0]} windows, labels {np.bincount(ys).tolist()}")

    print("Extracting Paderborn N09 (target) envelope-order features...")
    Xt, yt, codes = build_pu()
    print(f"  {Xt.shape[0]} windows, labels {np.bincount(yt).tolist()}")

    scaler = StandardScaler().fit(Xs)
    Xs_s, Xt_s = scaler.transform(Xs), scaler.transform(Xt)

    print("\nSTRICT TRANSFER -- fit on CWRU only, never a Paderborn label:")
    for name, clf in (
        ("logistic (binary)", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ("random forest (binary)", RandomForestClassifier(n_estimators=300, random_state=0, class_weight="balanced")),
    ):
        clf.fit(Xs_s, (ys > 0).astype(int))
        report(name, yt, clf.predict(Xt_s))

    # The comparison that matters: the spectrogram pipeline scored balanced
    # accuracy 0.499-0.505 here, i.e. exactly chance, in every configuration
    # tried. Anything meaningfully above 0.5 is new information.
    print("\n  reference: spectrogram pipeline reached balanced 0.499-0.505 (chance)")

    # --- calibrated on target HEALTHY data only -------------------------
    # This is not cheating, it is the project's actual premise: you have
    # healthy recordings from the machine you want to monitor, and no
    # faulty ones. A raw classifier threshold is meaningless across rigs
    # because the score distribution shifts wholesale; calibrating the
    # threshold on target healthy data fixes the false-alarm rate BY
    # CONSTRUCTION without ever seeing a target fault. infer.py --calibrate
    # is the deployment form of this.
    print("\nCALIBRATED on target healthy windows only (no target fault labels):")
    healthy_idx = np.where(yt == 0)[0]
    rng = np.random.default_rng(0)
    rng.shuffle(healthy_idx)
    n_cal = int(0.6 * len(healthy_idx))
    cal_idx, test_healthy = healthy_idx[:n_cal], healthy_idx[n_cal:]
    eval_idx = np.concatenate([test_healthy, np.where(yt > 0)[0]])

    # Repeated over calibration splits: which healthy windows land in the
    # calibration set is arbitrary, and a single split could flatter the
    # result. This project has already been burned once by a number that
    # was never replicated.
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xs_s, (ys > 0).astype(int))
    scores = clf.predict_proba(Xt_s)[:, 1]
    import statistics as st
    for target_fa in (0.05, 0.10, 0.20):
        recs, fas, bals = [], [], []
        for seed in range(10):
            r = np.random.default_rng(seed)
            hi = np.where(yt == 0)[0].copy(); r.shuffle(hi)
            n_cal = int(0.6 * len(hi))
            cal, th = hi[:n_cal], hi[n_cal:]
            ev = np.concatenate([th, np.where(yt > 0)[0]])
            thr = np.quantile(scores[cal], 1 - target_fa)
            pred = (scores[ev] > thr).astype(int); truth = yt[ev] > 0
            tp = int((truth & (pred > 0)).sum()); fn = int((truth & (pred == 0)).sum())
            fp = int((~truth & (pred > 0)).sum()); tn = int((~truth & (pred == 0)).sum())
            rec = tp / max(tp + fn, 1); fa = fp / max(fp + tn, 1)
            recs.append(rec); fas.append(fa); bals.append((rec + (1 - fa)) / 2)
        sd = lambda v: st.stdev(v) if len(v) > 1 else 0.0
        print(f"  logistic @ {target_fa:.0%} target FA (10 splits): "
              f"recall {st.mean(recs):.3f}+-{sd(recs):.3f}  "
              f"FA {st.mean(fas):.3f}+-{sd(fas):.3f}  "
              f"balanced {st.mean(bals):.3f}+-{sd(bals):.3f}")

    print("\nPer-bearing-unit detection at the 10% operating point:")
    thr = np.quantile(scores[yt == 0], 0.90)
    flagged = scores > thr
    for code in sorted(set(codes)):
        m = codes == code
        truth = "healthy" if yt[m][0] == 0 else "FAULTY"
        print(f"  {code} {truth:>7}: flagged {flagged[m].mean():.2f}  (n={m.sum()})")

    print("\nSanity check -- does the representation work WITHIN CWRU?")
    from sklearn.model_selection import cross_val_score
    s = cross_val_score(RandomForestClassifier(n_estimators=200, random_state=0),
                        Xs_s, (ys > 0).astype(int), cv=4)
    print(f"  CWRU 4-fold binary accuracy: {s.mean():.3f} +- {s.std():.3f}")


if __name__ == "__main__":
    main()
