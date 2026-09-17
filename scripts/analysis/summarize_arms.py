"""Arm A: F1 and balanced accuracy per representation, over seeds.

Why these two metrics and not recall alone: when order tracking showed BOTH
lower recall and lower false alarms than the time representation, that is
the signature of a more conservative operating point, which would be a
threshold difference rather than worse discrimination. F1 and balanced
accuracy weigh both directions, so they separate the two explanations.

Usage:  python scripts/analysis/summarize_arms.py [outputs_dir]
"""
import glob
import os
import statistics as st
import sys
import json

root = sys.argv[1] if len(sys.argv) > 1 else "outputs"

print(f"{'rep':>6}{'n':>4}{'recall':>18}{'precision':>18}{'F1':>18}{'balanced acc':>18}")
for rep in ("time", "order"):
    # glob rather than an exact path: the run dirs come out as
    # da_reports_armA_<rep>_armA_<rep>_s<seed> (the config already names the
    # arm, and --tag appends it again), so an exact-path version of this
    # script silently reports n=0.
    paths = sorted(glob.glob(os.path.join(root, f"*armA_{rep}*", "evaluation_metrics.json")))
    recs, precs, f1s, bals = [], [], [], []
    for p in paths:
        b = json.load(open(p))["baseline"]["binary_detection"]
        recs.append(b["recall"]); precs.append(b["precision"]); f1s.append(b["f1"])
        bals.append((b["recall"] + (1 - b["false_alarm_rate"])) / 2)
    if not f1s:
        print(f"{rep:>6}{0:>4}    (no runs found)")
        continue
    sd = lambda v: st.stdev(v) if len(v) > 1 else 0.0
    print(f"{rep:>6}{len(f1s):>4}"
          f"   {st.mean(recs):.3f} +- {sd(recs):.3f}"
          f"   {st.mean(precs):.3f} +- {sd(precs):.3f}"
          f"   {st.mean(f1s):.3f} +- {sd(f1s):.3f}"
          f"   {st.mean(bals):.3f} +- {sd(bals):.3f}")
