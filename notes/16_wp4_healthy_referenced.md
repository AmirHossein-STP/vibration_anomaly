# 16. WP4: healthy-referenced scoring and physics typing

MASTER_PLAN.md WP4, run 2026-10-15.

- Code: `scripts/analysis/wp4_healthy_ref.py`. It reads the WP2 caches and
  runs in seconds.
- Tables: `outputs/wp4/wp4_report.md`.
- Arms, features and readings are fixed in the script's docstring before it
  was run.

**The Paderborn adaptation.** A faulty Paderborn bearing was never recorded
healthy, so "this unit's own healthy baseline" does not exist there. The
stand-in is the *machine's* healthy baseline in that operating condition,
taken from the 5 calibration healthy bearings in each
leave-one-healthy-unit-out fold. The held-out bearing is never seen by
normalisation or threshold.

On a real machine, the baseline would be the same bearing earlier in its
life. That removes the unit-to-unit spread that drives Paderborn's false
alarms. XJTU-SY (WP5/WP7) can test that setting.

## 1. Results (real damage, conformal threshold at 10%, mean over 6 folds)

| features | condition | arm | recall | held-out FA | worst unit FA | held-out AUC (worst fold) |
|---|---|---|---|---|---|---|
| v1 | N09 | a: CWRU model (WP1) | 0.740 | 0.115 | 0.37 | 0.890 (0.80) |
| v1 | N09 | d: physics score | 0.764 | 0.196 | 0.86 | 0.892 (0.67) |
| CPW+SES | N09 | d: physics score | 0.746 | 0.112 | 0.23 | **0.901 (0.85)** |
| v1 | N15 ×3 | a: CWRU model | 0.29–0.51 | 0.11 | 0.22–0.38 | 0.61–0.76 |
| v1 | N15 ×3 | d: physics score | 0.74–0.77 | 0.14–0.28 | 0.33–0.83 | **0.83–0.89** |
| CPW+SES | N15 ×3 | d: physics score | 0.66–0.68 | 0.12–0.14 | 0.31–0.44 | **0.84–0.86** (0.75–0.78) |

Arms b and c (healthy z-scoring and healthy-to-healthy CORAL in front of the
CWRU classifier) are in the full table.

**What arm d is.** It uses no source data and no classifier. The score is
the largest one-sided z-score of log(peak) over the six fault-order peaks
(BPFO and BPFI, harmonics 1–3), measured against the healthy baseline. It is
the "healthy-referenced physics features" layer of MASTER_PLAN §3 (L1+L2),
in its simplest form.

## 2. Against the pre-registered readings

**WP4 gate (b or c lower worst-unit FA, recall within 0.02): fail on v1
features.**
- Healthy z-scoring (b) raises recall slightly, but worst-unit FA becomes
  0.97 at N09.
- CORAL (c) is worse on both counts.
- Both pass only on CPW+SES, and only because that baseline is broken (the
  inverted CWRU boundary, notes/15).
- Conclusion: normalising the *input* of the source classifier by healthy
  statistics does not fix per-unit false alarms.

**Arm d: promising, by the stated criterion.**
- On CPW+SES it reaches the WP1 model's recall at N09 (0.746 against 0.740,
  at 0.112 against 0.115 FA, with a lower worst unit, 0.23 against 0.37).
- It beats it at all three 1500 rpm conditions.
- Threshold-free, its held-out AUC at 1500 rpm is 0.83–0.89 against 0.61–0.76.
- On v1 features its AUC at N15 even exceeds what a linear model fit *on
  Paderborn itself* reached (0.70–0.79, notes/15 §2).
- So the 1500 rpm drop of notes/13–15 belongs mainly to the CWRU-trained
  classifier, not to the order peaks.

**Physics typing: passes.** Correct race on true-faulty windows is
0.74–0.81 in every condition, with either feature set. The CWRU 3-class
classifier reached 0.608 (notes/13).
- Outer race is no longer read as inner: 0.66–0.82 of outer windows are read
  as outer.
- That supports notes/13's masking hypothesis. A classifier trained on CWRU's
  large EDM faults underweights the BPFO peaks, while a race-wise rise over
  the baseline does not.

## 3. Caveats (why this is "promising", not "established")

1. **Paderborn is now spent for this choice.**
   - Arm d on v1 was pre-registered in notes/15 §3.
   - CPW+SES had been chosen a priori in WP2, but the idea of pairing it with
     a healthy reference came from notes/15's diagnostic, which looked at
     Paderborn labels.
   - Confirmation must come from datasets that played no part: XJTU-SY and
     Ottawa (WP5), then the IMS vault (WP11).
2. **False alarms overshoot the nominal rate.**
   - Mean held-out FA is 0.11–0.14 on CPW+SES and up to 0.28 on v1.
   - The worst healthy bearing reaches 0.31–0.44 (CPW+SES) and 0.83–0.86 (v1).
   - A max-of-z score is sensitive to a single bearing with one unusually
     strong order line.
   - The per-unit FA problem of notes/14 is reduced on CPW+SES, not solved.
3. **The baseline here is other bearings**, not the same bearing earlier.
   Whether the deployment setting is easier is a WP5/WP7 question.
4. **Healthy-referenced scoring of a single unit cannot be checked on
   Paderborn.** Each bearing appears in only one health state there.

## 4. Consequences for the plan

- **The physics score (arm d, on CPW+SES) becomes the primary detector
  candidate.** The CWRU-trained classifier becomes the comparison arm. This
  reverses the roles §3 of MASTER_PLAN assumed. It needs no source dataset,
  which suits the turbomachine target.
- **WP3 (feature set v2):** extends the physics score rather than the
  classifier: harmonics to 5, sidebands, FTF/BSF families, unknown-geometry
  broad bands. The peaks-only classifier arm of notes/15 §3 stays as the
  comparison.
- **WP5** is now the critical path. Nothing more can be learned from
  Paderborn without overfitting to it.
- **WP9** (productising) waits until WP5 confirms. The bearing service keeps
  v1 for now.
