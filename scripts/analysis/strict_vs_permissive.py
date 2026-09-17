"""The control that decides what the strict-arm collapse MEANS.

Strict arm A holds Paderborn entirely out of training and then diagnoses
Paderborn faults. It collapses: every target window predicted class 2.

The permissive config differs in exactly one premise -- Paderborn's
artificial faults ARE in training -- and reported recall ~0.76 on Kaggle.
Running both on the same box, same code, same torch separates two readings:

  * permissive works, strict collapses -> the pipeline is sound and the
    strict cross-dataset premise fails. Every earlier "success" depended on
    having target-rig data in training.
  * both collapse -> the rebuild is broken and nothing has been shown about
    the premise either way.

Usage: python scripts/analysis/strict_vs_permissive.py [outputs_dir]
"""
import glob
import json
import os
import statistics as st
import sys

root = sys.argv[1] if len(sys.argv) > 1 else "outputs/server_rescue"

GROUPS = [
    ("strict armA (PU held out)", "*armA_time_armA_time*"),
    ("strict, torch 2.6 rerun", "*t26_s*"),
    ("PERMISSIVE (PU faults in training)", "*perm_s*"),
]

print(f"{'configuration':>36}{'n':>4}{'recall':>16}{'FA':>16}{'balanced':>16}  collapsed?")
for label, pat in GROUPS:
    paths = sorted(glob.glob(os.path.join(root, pat, "evaluation_metrics.json")))
    recs, fas, bals, collapsed = [], [], [], 0
    for p in paths:
        d = json.load(open(p))
        b = d["baseline"]["binary_detection"]
        cm = d["baseline"]["confusion_matrix"]
        recs.append(b["recall"]); fas.append(b["false_alarm_rate"])
        bals.append((b["recall"] + (1 - b["false_alarm_rate"])) / 2)
        # "collapsed" = every window landed in a single predicted class
        cols = [sum(r[c] for r in cm) for c in range(3)]
        collapsed += sum(1 for c in cols if c > 0) == 1
    if not recs:
        print(f"{label:>36}{0:>4}    (not run)")
        continue
    sd = lambda v: st.stdev(v) if len(v) > 1 else 0.0
    print(f"{label:>36}{len(recs):>4}"
          f"   {st.mean(recs):.3f} +- {sd(recs):.3f}"
          f"   {st.mean(fas):.3f} +- {sd(fas):.3f}"
          f"   {st.mean(bals):.3f} +- {sd(bals):.3f}"
          f"   {collapsed}/{len(recs)} runs")

print("\nReference: Kaggle v8 permissive reported recall 0.761, false alarm 16.7%.")
print("Balanced accuracy 0.500 with a 100% false-alarm rate is the degenerate")
print("'call everything faulty' point -- it is chance, not detection.")
