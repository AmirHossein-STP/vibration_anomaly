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

## 5. Scope of this first pass, and what's deliberately deferred

Mirrors the original MVP's phased history (see `notes/06_extension_ideas.md`
in the CWRU pipeline) — built now: data loader, model, `train.py`,
`evaluate.py`, smoke tests, Kaggle notebook + push/pull scripts, and (after
two real runs forced them) the per-component redesign and block-shuffled
split in section 2. Deferred, in rough priority order for a next session:

- **Second-order confound, not yet addressed**: block-shuffling fixes the
  TIME confound, but each detector's "healthy" population is still
  selected on ONE column (e.g. `valve_condition==100`) while the other
  three components vary freely underneath it — if, say, cycles with a
  healthy valve also happen to have a healthier-than-average cooler more
  often than chance (a property of how the test campaign was actually run,
  not something the split controls), the model could partly be learning
  "cooler is fine" and mislabeling it as "valve is fine." `degraded_component`
  is already recorded per cycle specifically to let you check this (e.g.
  cross-tabulate it against predictions), but the check itself hasn't been
  done — worth doing before trusting any one detector's number in
  isolation.
- **`analyze.py`/`sweep.py` equivalents** (latent-space PCA per component,
  fused-latent-dim sweep) — CWRU added these as a second pass after the
  first real run showed the baseline worked; same plan here, once these
  four detectors have real, trustworthy results.
- **Overfitting check**: each detector still has a 6.7M-parameter model
  against a few hundred to ~1200 training cycles — meaningfully more data
  than the 20-cycle crash, but still modest relative to CWRU's ~2800
  training windows for a similarly-sized model. Watch `training_log.csv`
  (train vs. val error diverging) on the first real run per component;
  shrinking `model.*_branch_latent`/`*_base_channels` is the fix if so, not
  yet applied pre-emptively since it wasn't yet confirmed as a problem.
- **Spectrogram features for the fast pressure branch** — the fast sensors
  do have oscillatory structure a frequency-domain view could exploit;
  left out of this first pass to avoid conflating two architecture changes
  (fusion + feature representation) in one untested result.
- **A presentation report** (`build_presentation.py` equivalent) — after a
  real run exists to report on, same as CWRU.
