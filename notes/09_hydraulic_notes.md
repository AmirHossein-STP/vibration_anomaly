# Hydraulic sensor-fusion experiment: dataset, labels, architecture, evaluation

Companion to `HYDRAULIC_PLAN.md` (the what-and-why summary) — this is the
detail doc, same role `notes/01`–`04` play for the CWRU pipeline.

## 1. Dataset structure (as documented publicly — see caveat below)

Source: [Condition monitoring of hydraulic systems](https://www.kaggle.com/datasets/jjacostupa/condition-monitoring-of-hydraulic-systems),
a Kaggle mirror of the ZeMA/UCI dataset (Helwig, Pignanelli, Schütze). One
`.txt` file per sensor, tab-separated, **rows = cycles (2205 total), columns
= samples within that 60-second cycle**. Plus `profile.txt`: one row per
cycle, five tab-separated columns, the ground-truth condition of all four
components for that cycle.

| File | Sensor | Rate | Cols (samples/cycle) |
|---|---|---|---|
| `PS1.txt`...`PS6.txt` | Pressure (bar) | 100 Hz | 6000 |
| `EPS1.txt` | Motor power (W) | 100 Hz | 6000 |
| `FS1.txt`, `FS2.txt` | Volume flow (l/min) | 10 Hz | 600 |
| `TS1.txt`...`TS4.txt` | Temperature (°C) | 1 Hz | 60 |
| `VS1.txt` | Vibration (mm/s) | 1 Hz | 60 |
| `CE.txt` | Cooling efficiency (%) | 1 Hz | 60 |
| `CP.txt` | Cooling power (kW) | 1 Hz | 60 |
| `SE.txt` | Efficiency factor (%) | 1 Hz | 60 |

`profile.txt` columns, in order, and the value → condition mapping:

| Column | Meaning | Values | Healthy value |
|---|---|---|---|
| 1 | Cooler condition | 3 (close to failure), 20 (reduced efficiency), 100 (full efficiency) | 100 |
| 2 | Valve condition | 73 (close to failure), 80 (severe lag), 90 (small lag), 100 (optimal) | 100 |
| 3 | Internal pump leakage | 0 (none), 1 (weak), 2 (severe) | 0 |
| 4 | Hydraulic accumulator | 90 (close to failure), 100 (severely reduced), 115 (slightly reduced), 130 (optimal) | 130 |
| 5 | Stable flag | 0 (conditions were stable), 1 (static conditions possibly not yet reached) | 0 |

**Caveat resolved (2026-08-27, first real Kaggle run)**: the table above
was originally transcribed unverified (Kaggle CLI auth was broken that
session). Now confirmed directly two ways: `kaggle datasets files` lists
exactly these 17 sensor files + `profile.txt` + `documentation.txt`/
`description.txt`, flat, no subfolder nesting; and the notebook's diagnostic
cell printed the real per-column value counts from the actual file, which
matched every value in the table above exactly:
`cooler_condition: 3=732,20=732,100=741` / `valve_condition: 73=360,80=360,
90=360,100=1125` / `pump_leakage: 0=1221,1=492,2=492` /
`accumulator_pressure: 90=808,100=399,115=399,130=599` /
`stable_flag: 0=1449,1=756`. `data/catalog.py` still validates every value
it reads against these tables and raises on any mismatch, in case a future
dataset republish changes something.

## 2. Split strategy — per-component redesign + block-shuffled split (2026-08-27)

No windowing step: each cycle is already one fixed-length example (unlike
CWRU's long continuous recordings).

**The original design required all four components healthy at once**
("fully healthy" = all four profile columns at their healthy value
simultaneously) and it doesn't work: the first real Kaggle run crashed
(`ZeroDivisionError` — a train loader with 15 examples and
`batch_size=32`+`drop_last=True` produces zero batches), and even without
that crash, 15 training examples is nowhere near enough to train a
6.7M-parameter autoencoder. The value counts above explain why: the test
rig sweeps these four conditions in a *coupled* way over the run (long
stretches with one component varied while the others sit at some other
setting), not independently per cycle, so requiring all four optimal
simultaneously is rare by construction — only 20 of 2205 cycles qualified,
even though each *individual* component's healthy population is large
(599–1225 cycles, see the value counts above).

**Redesign: one detector per component**, each with its own config
(`configs/hydraulic_{cooler,valve,pump,accumulator}.yaml`,
`data.target_component`) and its own "healthy" definition — that ONE
component at its healthy value, full stop, with the other three components'
state ignored (not required healthy, not filtered on). This matches how
this dataset's four sub-tasks are conventionally used in the literature
(each treated as its own independent detection/classification problem) and
gives each detector 599–1225 healthy cycles to work with instead of 20.
`degraded_component` (comma-joined names of every column off its healthy
value, computed across all four regardless of which one is the current
target) is still recorded in metadata as a diagnostic — e.g. to check how
many of a detector's "healthy" training cycles are quietly degraded on
some *other* component too, a possible confound worth checking once real
results exist.

### Block-shuffled split (revised after the SECOND real run)

The first version of this redesign kept the original CHRONOLOGICAL split
logic (first 70% of target-healthy cycles by row order → train, next 15% →
val, last 15% + all degraded → test) — it fixed the crash, but the second
real run's numbers were still wrong: ROC-AUC came back 1.0 for cooler but
*below random* for valve (0.077), pump (0.292), and accumulator (0.107).
Below-random is itself informative — it means degraded cycles got *lower*
error than healthy ones, a systematic inversion, not noise. Checking
`test_predictions.csv`'s `cycle_idx` column against `recon_error` found
the cause: reconstruction error correlates strongly with **cycle_idx
(campaign time)** — up to |r|=0.83 — because the test rig runs each
component's conditions in long, non-interleaved stretches. A chronological
split confines TRAIN/VAL to one narrow slice of calendar time (e.g.
accumulator's healthy TEST cycles were *all* within cycle_idx 1707–1796, a
90-cycle window out of 2205), so the model mostly learned "what does the
signal look like during this time slice," not "is this component healthy."
Cooler's apparent 1.0 was the same confound working in its favor by
coincidence (its healthy-test window happened to correlate with low error
for unrelated reasons) — not real detection either.

**Fix: block-shuffled split.** Chop each detector's target-healthy cycles
into contiguous blocks of `split.block_size` cycles (default 20, in
original row order), shuffle the BLOCKS (seeded by `config.seed`), then
`train_fraction`/`val_fraction` of blocks go to train/val, the rest to
test — see `data/dataset.py`'s `_block_shuffle_split`. This spreads
train/val/test across the FULL campaign timespan, breaking the time
confound, while still keeping short-range structure (a block) intact —
unlike a pure per-cycle shuffle, which would risk leaking near-duplicate
adjacent cycles across splits the same way CWRU's overlapping windows do
(the concern that motivated a chronological split in the first place, and
still valid at the individual-cycle grain, just not at the whole-dataset
grain). TEST still gets every cycle where the target component isn't
healthy, same as before. `stable_flag` is kept as a metadata column (not
filtered on) — it's reported per-cycle in the output CSVs so you can see
whether detection performance differs on cycles where the rig hadn't fully
settled, without baking an assumption about its effect into the split.

This is still not a complete fix for every possible confound in this
dataset — see section 5's "second-order confound" note for what's still
an open, harder-to-eliminate risk (the OTHER three components' state
correlating with the target component's state, independent of time).

## 3. Model: multi-branch fusion autoencoder

Three independent encoder branches (one per native rate), fused at a single
bottleneck, three independent decoder branches mirroring them back out.

```
slow  (N, 8, 60)   -> MLP encoder      -> branch_latent_slow   (default 16)
medium(N, 2, 600)  -> Conv1d encoder   -> branch_latent_medium (default 32)
fast  (N, 7, 6000) -> Conv1d encoder   -> branch_latent_fast   (default 64)
                                    concat -> Linear -> fused_latent (default 32)
                                    Linear -> split -> three mirror decoders
```

- **Slow branch**: an MLP (Linear → ReLU → Linear), not a conv, because at
  1Hz/60s these are near-monotonic thermal/efficiency drift, not
  oscillatory signal — there's little for a convolution to exploit that a
  dense layer over the flattened (8×60=480) input can't also see.
- **Medium/fast branches**: `Conv1d(kernel=4, stride=2, padding=1)` layers,
  the exact 1D analogue of the CWRU spectrogram encoder's 2D conv layers —
  each layer exactly halves the sequence length (600→300→150→75 for
  medium's 3 layers; 6000→3000→1500→750→375 for fast's 4 layers), and
  `ConvTranspose1d` with the same kernel/stride/padding exactly doubles it
  back on the decode side, so the reconstruction shape matches the input
  shape exactly with no cropping/padding hacks — same trick
  `models/autoencoder.py` uses, just 1D.
- **Loss**: each branch's own per-sample MSE is computed first (mean over
  that branch's own channels × length), *then* the three branch losses are
  averaged with equal weight. Averaging raw elementwise error across all
  branches together would let the fast branch (6000×7 = 42000 numbers per
  example) drown out the slow branch (8×60 = 480 numbers per example)
  purely because it has ~90x more numbers, not because it's more
  diagnostic — this is the same normalization concern that motivated
  per-frequency-bin z-scoring in the CWRU spectrogram pipeline
  (`notes/02_signal_processing_notes.md`), just at the branch level instead
  of the frequency-bin level. `evaluate.py` reports each branch's error
  separately too, not just the combined score — see below.
- **Normalization**: per-sensor z-score (mean/std per one of the 17
  sensors, shape (17,1) conceptually, computed separately within each
  branch), fit on TRAIN healthy cycles only — direct analogue of
  `SpectrogramNormalizer`.

## 4. Evaluation

Same shape as CWRU's `evaluate.py`, run once per component: threshold = a
percentile of the combined reconstruction error on the VALIDATION healthy
set for that detector (no fault label used to pick it), then graded
against the real profile labels for that one target column (ROC-AUC,
PR-AUC, confusion matrix). Detection rate is broken down by the target
component's own off-healthy value (`_severity_summary` — the direct
analogue of CWRU's `fault_type` × `fault_size_mils` table, just for one
component instead of a multi-column combination, now that each config
targets exactly one).

**Fusion-specific report**: mean per-branch (slow/medium/fast)
reconstruction error, broken down by the target component's value
(`_branch_error_by_value`) — answers "does a pump leak show up more in the
pressure sensors (fast) or the temperature sensors (slow)?", which CWRU's
single-stream architecture has no equivalent question for.

## 4b. Threshold-calibration finding, and the two open questions it resolved (2026-08-27)

After block-shuffling fixed the time confound (section 2), a real run's
numbers were finally coherent enough to investigate the two things left
open from the redesign. `analyze.py` (`hydraulic_fusion_ad/analysis.py`)
was built specifically to answer them with saved, reproducible reports
instead of one-off scripts:

**1. Second-order confound (`confound_check`) — checked, found to be
minor.** Among each detector's target-healthy TEST cycles, correlating
"how many OTHER components are also degraded" against reconstruction error
gave |r| ≤ 0.29 for all four detectors, and false-positive rate among
those cycles stayed at 0–4% regardless of how many other things were
wrong. A detector reacting substantially to some other component's fault
would show a much stronger effect than this — the concern raised in the
previous version of this section is resolved, not just deferred.

**2. Threshold miscalibration (`threshold_sensitivity`) — real, but NOT a
simple "use a lower percentile" fix.** First pass: sweeping percentiles
and taking whichever maximizes test F1 (`f1_optimal_threshold`) looked
dramatic —

| Component | F1 @ deployed (99th pct) | Raw F1-optimal | at percentile |
|---|---|---|---|
| cooler | 0.999 | 0.998 | 96th (no real gain — already saturated) |
| valve | 0.000 | 0.966 | 68th |
| pump | 0.004 | 0.905 | 50th |
| accumulator | 0.636 | 0.932 | 50th |

**...until checking the false-positive rate those "best" percentiles
actually carry.** This dataset's per-component TEST sets are heavily
skewed toward the degraded class (e.g. pump: 984 degraded vs. only 200
healthy) — the *opposite* skew from a real deployment, where healthy
operation dominates. A percentile can score a great F1 on a
degraded-heavy test set simply by flagging almost everything. Recomputing
with `false_positive_rate` tracked explicitly (fraction of TARGET-HEALTHY
test cycles wrongly flagged):

| Component | Raw F1-optimal pct | F1 | **false_positive_rate** |
|---|---|---|---|
| cooler | 96th | 0.998 | 5% |
| valve | 68th | 0.966 | **47%** |
| pump | 50th | 0.905 | **75%** |
| accumulator | 50th | 0.932 | **40%** |

A 40-75% false-positive rate on healthy operation is useless in any real
deployment, no matter how good the F1 looks on this test set's skewed
class balance. So `best_deployable_threshold(sensitivity_df, max_fpr=0.10)`
was added: same F1-maximization, but constrained to percentiles that keep
false_positive_rate at or below a stated tolerance (10% by default) —
returns `None` if nothing in the sweep qualifies, which is itself a real,
informative result (see the actual numbers once kernel v6 pulls, referenced
from `HYDRAULIC_PLAN.md`'s Status section — this note was written to
explain the corrected methodology, not to re-embed numbers that change
per run).

**What this actually means**: cooler is the only component with a
genuinely good, deployment-realistic operating point (high recall at ~5%
FPR). For the other three, the honest picture is more sobering than "just
pick a better percentile" — their reconstruction-error distributions for
healthy vs. degraded cycles overlap enough that *no* single percentile
threshold delivers both good recall and an acceptable false-positive rate
simultaneously. Good ROC-AUC (0.66–0.89) still means the score ranks
degraded cycles higher on average — the model isn't producing noise — but
turning that ranking into a usable point detector may need real
improvements (better features for the fast/pressure branch, a larger
distinguishing signal, or accepting a different anomaly-scoring approach
entirely), not just a threshold tweak. Section 5 reflects this.

**Methodological line held throughout**: nothing here EVER uses
test-set-informed percentiles to set `configs/hydraulic_*.yaml`'s deployed
`threshold_percentile: 99.0` — both `f1_optimal_threshold` and
`best_deployable_threshold` are saved to `threshold_cheat_comparison.json`,
explicitly labeled as comparisons, never as a valid selection procedure
(see `notes/04_evaluation_notes.md`'s original reasoning for CWRU, held to
the same standard here). This is the CWRU-extension-ideas.md "F1-optimal
threshold as a reported, clearly-labeled second number" idea, never built
there (every percentile scored about the same, so it wasn't informative)
— genuinely informative here, in both its raw and FPR-corrected forms.

## 5. Scope, what's resolved, and what's still open

Mirrors the original MVP's phased history (see `notes/06_extension_ideas.md`
in the CWRU pipeline). Built across four Kaggle runs: data loader, model,
`train.py`, `evaluate.py`, the per-component redesign, the block-shuffled
split, and (this pass) `analyze.py` with the three diagnostics in section
4b. **Resolved**: the second-order confound (checked, minor). **Root-caused
but NOT fixed**: threshold miscalibration turned out to be a symptom of a
deeper issue (real distribution overlap for 3 of 4 components, not a bad
percentile choice — see section 4b's FPR correction). **Still open, in
rough priority order**:

- **Pick a legitimate (non-test-informed) lower default `threshold_percentile`
  for valve and accumulator.** With the FPR constraint applied honestly,
  these two are NOT stuck — a percentile around 88–96 gets F1 to
  0.68–0.70 at only 6–9% false-positive rate (see `HYDRAULIC_PLAN.md`'s
  Status section for the exact numbers), a real, usable improvement over
  the 99th-percentile default. The remaining problem is procedural, not
  architectural: today's number was found by looking at test performance,
  which can't become the deployed default as-is. Needs either (a) a
  nested threshold-selection split (hold out a labeled slice purely for
  calibration, distinct from train/val/final-test), (b) a criterion based
  only on the validation-healthy distribution's own shape, or (c) an
  explicit business decision about acceptable false-alarm rate, made
  independently of having seen this result.
- **Pump needs real model/feature improvement, not threshold tuning** —
  even at its best FPR-constrained operating point it only reaches
  F1≈0.39 (recall 24%). Concrete candidates, roughly in order of effort:
  (a) spectrogram features for the fast pressure branch (already listed
  below, now with a stronger motivation specific to pump), (b) per-branch
  loss reweighting if one branch is drowning out a more diagnostic one
  (`branch_error_by_component.csv` already has the data to check this),
  (c) a larger/differently-shaped model if `training_log.csv` shows
  underfitting rather than overfitting.
- **`cooler_condition`'s raw time-correlation (r=+0.88, see section 2)
  remains unresolved** — `analyze.py`'s `raw_time_correlation` now reports
  this number every run, but nothing yet distinguishes "genuine wear
  progressing over the campaign" from "an artifact of how conditions were
  scheduled." Would need either domain knowledge about this specific test
  rig's protocol, or a dataset with less time/condition entanglement, to
  resolve further.
- **Latent-space PCA / fused-latent-dim sweep** (CWRU's `analyze.py`
  section this one doesn't yet cover) — worth adding once the threshold
  question above is settled, so the "what is the model learning" story
  isn't built on top of a known-bad operating point.
- **Overfitting check**: each detector still has a 6.7M-parameter model
  against a few hundred to ~1200 training cycles. `training_log.csv` (not
  yet inspected for train/val divergence) is the place to check; shrinking
  `model.*_branch_latent`/`*_base_channels` is the fix if it shows up.
- **Spectrogram features for the fast pressure branch** — the fast sensors
  do have oscillatory structure a frequency-domain view could exploit;
  left out of this first pass to avoid conflating two architecture changes
  (fusion + feature representation) in one untested result.
- **A presentation report** (`build_presentation.py` equivalent) — worth
  building once the threshold question is settled and the numbers it would
  report are the real, deployed-default numbers.
