# 14. WP1: recording- and unit-level re-scoring (and WP2 H-a)

MASTER_PLAN.md WP1, run 2026-10-15. No new model: the same 13 envelope-order
features, the same CWRU-only logistic regression and the same healthy-only
calibration as notes/13. What changed is the counting.

- Code: `scripts/analysis/envelope_cache.py` (extraction, the only heavy step:
  17,346 windows, 79 s on one core of the local Mac, run with the owner's
  approval) and `scripts/analysis/wp1_rescore.py` (scoring, seconds).
- Full tables: `outputs/wp1/wp1_report.md`, raw numbers in
  `outputs/wp1/wp1_results.json`. Cache fingerprint `d343970bb7b3`.
- Every number below is for **real damage**, unless it says "artificial". No
  Paderborn label enters any fit, threshold or level choice.

## 0. The cache reproduces notes/13 exactly

Using the old protocol (first 4 windows per run, 10 random healthy-window
splits): recall **0.746 ± 0.012**, balanced **0.823 ± 0.007**. These are the
published numbers to the third decimal, so the counting below is the same
model.

## 1. Calibrating on other units: average FA holds, per-unit FA does not

The old protocol calibrated and tested on windows from the *same* healthy
bearings. A deployed machine is a new unit. Here the threshold comes from 5
healthy units (finite-sample conformal quantile). False alarms are counted on
the 6th unit, over all six folds (LOHUO).

| condition | target FA | held-out FA (mean ± sd) | worst unit | recall |
|---|---|---|---|---|
| N09_M07_F10 | 10% | 0.115 ± 0.149 | 0.371 | 0.740 ± 0.026 |
| N15_M01_F10 | 10% | 0.107 ± 0.067 | 0.220 | 0.480 ± 0.012 |
| N15_M07_F04 | 10% | 0.114 ± 0.140 | 0.379 | 0.289 ± 0.020 |
| N15_M07_F10 | 10% | 0.108 ± 0.110 | 0.307 | 0.506 ± 0.024 |

- Recall barely moves (0.740 vs 0.746), so the headline was not inflated by
  shared units.
- The **false-alarm rate is only right on average over units**. One healthy
  bearing in six alarms at 2–4× the target.
  - This is the exchangeability failure MASTER_PLAN §3 predicted. The
    conformal guarantee is marginal over units, never per unit.
  - Practical consequence: a one-machine deployment cannot know in advance
    whether it is a K001 (0% FA) or a K005 (37%). The calibrate step must
    report each healthy recording's flagged fraction, which the service
    already does, and should prefer healthy data from several days.

## 2. Recording verdicts: watch at 25% was too eager

A Paderborn run is 4 s, which gives 7 windows. A recording's verdict is the
fraction of its windows above the 10% window threshold, as in
`serving_core/services/bearing.py`. FA below means held-out healthy recordings
flagged.

N09_M07_F10:

| ≥ windows | recording FA | worst unit | recall | |
|---|---|---|---|---|
| 2/7 | 0.233 ± 0.339 | 0.80 | 0.942 | old `watch` (25%) |
| 3/7 | 0.125 ± 0.216 | 0.55 | 0.875 | new `watch` (40%) |
| 4/7 | 0.050 ± 0.100 | 0.25 | 0.764 | `faulty` (50%) |
| 5/7 | 0.008 ± 0.020 | 0.05 | 0.668 | |

**Rule for choosing levels.** It was fixed before reading the table and uses
healthy data only: the smallest level whose mean held-out recording FA is at
most 20% (watch) and at most 5% (faulty).

| condition | watch | faulty |
|---|---|---|
| N09 | 3/7 | 4/7 |
| N15_M01_F10 | 2/7 | 4/7 |
| N15_M07_F04 | 3/7 | 5/7 |
| N15_M07_F10 | 3/7 | 4/7 |

**Decision (WP1 gate):**
- `watch_fraction` default **0.25 → 0.40**. The old level put a quarter of
  healthy recordings on watch at 900 rpm.
- `faulty_fraction` **stays at 0.50**. It meets the 5% cap in 3 of 4
  conditions and misses at F04 by 0.8 points (5.8%).
- Changed in `src/serving_core/services/bearing.py` and
  `deploy/runbooks/bearing.md`.

Caveat: the levels are fractions of a 4 s recording. On longer recordings the
window noise averages out, and the per-unit offset (§1) dominates instead.
That is why the service's calibrate-time warning, "a healthy recording already
reaches the watch level", matters more than the exact fraction.

**Alternative tried and rejected.** Score a recording by its median window
score, with a conformal threshold over recordings. This gives 0.167 FA at a
nominal 10% (recall 0.853), so it is no better. With 100 calibration
recordings from only 5 units, the threshold is driven by unit offsets.

## 3. Units

At the service's faulty level and the 10% window point, a faulty unit counts
as caught when at least half of its 20 recordings are judged faulty. Each
figure is averaged over the 6 calibration folds.

- **N09: 9/11 real-damage units caught.**
  - Missed: KA15 (0.03 of recordings) and KA22 (0.30).
  - Weak: KI14 (0.56) and KA30 (0.72).
  - The rest are at 0.88–1.00.
- Healthy units held out, as watch/faulty rates:
  - K005 0.80 / 0.25;
  - K002 0.50 / 0.05;
  - K001, K003 and K004 0 / 0.
- **The offending healthy unit changes with the operating condition.**

  | condition | worst healthy unit |
  |---|---|
  | N09 | K005 |
  | all three N15 | K002 |
  | F04 | K006 |

  So a per-unit offset is not a fixed property of the bearing. Healthy
  referencing in WP4 has to be per unit *and* per operating condition. That
  argues for a baseline indexed by speed/load, not one number per machine.

## 4. WP2 H-a, answered by the same cache: it is speed

All three 1500 rpm conditions are bad, whatever the load torque (M01 vs M07)
or radial force (F04 vs F10):

| condition | recall @10% (LOHUO) | units caught | artificial recall |
|---|---|---|---|
| N09_M07_F10 | 0.740 | 9/11 | 0.801 |
| N15_M01_F10 | 0.480 | 4/11 | 0.604 |
| N15_M07_F10 | 0.506 | 4/11 | 0.585 |
| N15_M07_F04 | **0.289** | 3/11 | 0.582 |

- **Speed is the primary factor.** Pre-registered reading: "if all N15
  conditions are bad, it is speed".
- **Radial force is a second factor**, on top of speed. Lower force (F04)
  roughly halves recall on real damage but leaves artificial damage
  unchanged. A plausible reading is weaker impacts from real pitting under
  less load. It is untested.
- **Clue for H-b/H-d.** In the drift report, `env_flatness` flips sign between
  speeds: SMD −1.84 at N09, +0.36…+0.55 at N15. `bpfi_h2` and `bpfo_h3` also
  flip.
  - Flatness and the ratio features are computed over a fixed-Hz/fixed-order
    span whose bin count changes with speed. That is H-b's mechanism.
  - Rig resonance relative to the fixed 1.5–5.5 kHz band is H-d's.
  - H-c (absolute tolerance vs speed error) is not distinguished by this.
- **Next (WP2, heavy each):** recompute with peak-to-median normalisation and
  revolution-based windows (H-b), with relative tolerance (H-c), and with a
  label-free band (H-d). Each variant is another ~80 s extraction.

## What changed in the ledger

- "Verdict levels are engineering defaults" becomes **measured**. Watch is
  lowered to 40%, faulty is kept at 50%.
- "N15 unexplained" becomes **speed-driven, with radial force as a second
  factor; mechanism open (H-b/c/d)**.
- New: **window FA calibrated on other units is right on average and off by
  2–4× on one unit in six**. The offending unit depends on the operating
  condition.
