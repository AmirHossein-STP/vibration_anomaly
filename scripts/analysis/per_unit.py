"""Per-BEARING-UNIT breakdown of arm A, pooled over seeds.

Why this and not the seed spread we already have: a deployment sees ONE
machine. The question it poses is not "what is the mean recall over 8
seeds" but "if I install this on a bearing, what is the chance it stays
silent while that bearing is failing". That is a per-unit quantity, and a
headline mean hides it -- kernel v5 already showed false alarms were
per-unit, not random (one healthy unit produced 100% of them).

Also reports per-RECORDING detection: a real system integrates many
windows before it alarms, so window-level recall understates it.
"""
import sys, glob, os
import numpy as np, pandas as pd

pat = sys.argv[1] if len(sys.argv) > 1 else "outputs/da_reports_armA_time_seed*/target_test_predictions.csv"
paths = sorted(glob.glob(pat))
if not paths:
    sys.exit(f"no prediction files matched {pat}")

frames = []
for p in paths:
    df = pd.read_csv(p)
    df["seed"] = os.path.basename(os.path.dirname(p))
    frames.append(df)
df = pd.concat(frames, ignore_index=True)
df["faulty_true"] = df.universal_label > 0
df["faulty_pred"] = df.baseline_pred > 0
print(f"{len(paths)} runs, {len(df)} window-predictions, {df.code.nunique()} bearing units\n")

rows = []
for code, g in df.groupby("code"):
    faulty = bool(g.faulty_true.iloc[0])
    # per seed, so the spread reported is over runs for THIS unit
    per_seed = g.groupby("seed").faulty_pred.mean()
    rows.append({
        "code": code,
        "truth": "FAULTY" if faulty else "healthy",
        "n_win": len(g) // len(paths),
        # for a faulty unit this is recall; for a healthy one, false-alarm rate
        "flagged": per_seed.mean(),
        "sd": per_seed.std(ddof=1),
        "worst_seed": per_seed.min() if faulty else per_seed.max(),
    })
out = pd.DataFrame(rows).sort_values(["truth", "flagged"])
print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

print("\n--- per-RECORDING (majority of windows in a file, pooled seeds) ---")
rec = df.groupby(["seed", "code", "source_file"]).agg(
    faulty_true=("faulty_true", "first"), frac=("faulty_pred", "mean")).reset_index()
rec["alarm"] = rec.frac > 0.5
for truth, g in rec.groupby("faulty_true"):
    name = "FAULTY recordings detected" if truth else "healthy recordings false-alarmed"
    print(f"  {name}: {g.alarm.mean():.3f}  (n={len(g)//len(paths)} recordings x {len(paths)} runs)")

f = out[out.truth == "FAULTY"]
print(f"\nAcross the {len(f)} faulty units, per-unit recall ranges "
      f"{f.flagged.min():.3f} to {f.flagged.max():.3f} (unit-level sd {f.flagged.std(ddof=1):.3f}).")
print(f"Units missed more often than caught: "
      f"{sorted(f[f.flagged < 0.5].code.tolist()) or 'none'}")
