# Plan: sensor-fusion extension (ZeMA hydraulic dataset)

This is a second, independent experiment added alongside the original CWRU
vibration MVP — not a rewrite of it. `vibration_ad/` (single accelerometer
channel, spectrogram autoencoder) is untouched; this adds
`hydraulic_fusion_ad/` (17 sensors at 3 different native sampling rates,
multi-branch fusion autoencoder). Both read the same architectural idea
(train only on healthy data, use reconstruction error as the anomaly score)
but the data shape is different enough that the actual data/model code
doesn't share much beyond that idea — see "What's genuinely new" below.

## The dataset

[Condition monitoring of hydraulic systems](https://www.kaggle.com/datasets/jjacostupa/condition-monitoring-of-hydraulic-systems)
(ZeMA / UCI, via Kaggle mirror). A hydraulic test rig run for 2205 cycles
(60s each), with four components independently degraded by the test rig's
control system across the run: a cooler, a directional valve, an internal
pump (leakage), and a hydraulic accumulator. 17 sensors, three native rates:

| Group | Sensors | Rate | Samples/cycle |
|---|---|---|---|
| fast | PS1–PS6 (pressure), EPS1 (motor power) — 7 sensors | 100 Hz | 6000 |
| medium | FS1, FS2 (volume flow) — 2 sensors | 10 Hz | 600 |
| slow | TS1–TS4 (temperature), VS1 (vibration), CE, CP, SE (cooling efficiency/power, overall efficiency factor) — 8 sensors | 1 Hz | 60 |

`profile.txt` has one row per cycle: `[cooler_condition, valve_condition,
pump_leakage, accumulator_pressure, stable_flag]`. Healthy = cooler 100,
valve 100, pump_leakage 0, accumulator 130 — **confirmed against the real
files** on the first Kaggle run (2026-08-27); see `notes/09_hydraulic_notes.md`
section 1 for the actual value counts.

## Decisions made and why

1. **Revised after the first real run: four separate per-component
   detectors, not one combined detector.** The original plan (below, kept
   for the record) was "one detector, healthy = all four components
   healthy at once." That crashed on the very first Kaggle run
   (`ZeroDivisionError` — an empty train loader) because only **20 of 2205
   cycles** turn out to be healthy on all four components simultaneously;
   the test rig varies these four conditions in a coupled way over the
   run, not independently per cycle. Each *individual* component's healthy
   population is large (599–1225 cycles) — so the fix is one detector per
   component, each trained only on that component's own healthy cycles,
   ignoring the other three. See `notes/09_hydraulic_notes.md` section 2,
   "Split strategy — per-component redesign," for the full story and
   `configs/hydraulic_{cooler,valve,pump,accumulator}.yaml` for the four
   resulting configs.

   *Original plan (superseded)*: one multi-branch autoencoder trained on
   cycles healthy on all four components; anomaly = any degradation
   anywhere; `evaluate.py` breaks detection down by which component
   degraded and by severity. Picked over four separate detectors at the
   time specifically to avoid "four independent runs" as unnecessary
   overhead — turned out to be necessary, not optional, once the real
   healthy-cycle counts were known.

2. **Multi-branch fusion**: one small encoder per rate group (slow/medium/
   fast), each seeing its sensors at native resolution (no resampling), concatenated
   into one fused latent vector, decoded back out per branch. This is
   *intermediate fusion* — a genuine architectural step beyond the
   single-stream CWRU autoencoder. Loss is the mean of each branch's own
   per-sample MSE (not summed) — a branch with a 6000-length signal
   shouldn't dominate the loss just because it has more numbers, the same
   reasoning that motivated per-frequency-bin normalization in the CWRU
   pipeline (`notes/02_signal_processing_notes.md`), just applied across
   branches instead of across frequency bins.

## What's genuinely new (not reused from `vibration_ad/`)

- **No windowing.** CWRU cut one long recording into overlapping windows;
  here each 60s cycle *is already* one fixed-length example — one row in
  `profile.txt` = one training/eval example. `data/windows.py` has no
  equivalent here.
- **No STFT/spectrogram.** The slow sensors (temperature, efficiency) are
  near-monotonic drift over 60 1Hz samples, not oscillatory signal — a
  spectrogram would be meaningless for them. Sensors go into the model as
  raw (normalized) time series, split by native rate into three branches,
  not converted to frequency domain at all. (The fast pressure sensors
  *do* have oscillatory structure a spectrogram could exploit — noted as a
  future extension, not in this first pass, to keep this experiment's
  first result comparable and not conflate two changes at once.)
- **Split strategy**: CWRU held out one whole RPM family. There's no
  equivalent grouping variable here, so: per detector, sort its
  target-component-healthy cycles by their original (chronological) row
  order, first ~70% → train, next ~15% → val, last ~15% → test, and *all*
  cycles where that target component isn't healthy → test. Contiguous
  chronological blocks, not a random shuffle — same anti-leakage reasoning
  as CWRU's non-random split (`notes/01_dataset_notes.md`), adapted from
  "don't split overlapping windows randomly" to "don't split
  temporally-autocorrelated cycles randomly."
- **Per-branch error breakdown** falls out of the architecture almost for
  free (each branch has its own reconstruction error before they're
  combined into one score) — evaluate.py reports it per degraded
  component, e.g. "does a valve fault show up more in the fast (pressure)
  branch or the slow (temperature) branch." This is the fusion-specific
  question CWRU's single-stream model couldn't ask at all.

## Resolved: raw file layout (was an open item)

Kaggle auth was broken as of the first planning session; you fixed it. File
layout confirmed both via `kaggle datasets files` (flat directory, exactly
the 17 sensor files + `profile.txt` + `documentation.txt`/`description.txt`,
matching `configs/hydraulic_*.yaml`'s assumed filenames exactly) and via
the notebook's own diagnostic cell against the real `profile.txt` — see
`notes/09_hydraulic_notes.md` section 1.

## Status

**Kernel v1 (2026-08-27): failed**, `ZeroDivisionError` during `train.py` —
root cause was the "all four components healthy at once" split leaving
only 20 usable cycles (see decision 1 above). Fixed by redesigning to four
separate per-component detectors.

**Kernel v2 (2026-08-27): ran to completion, but the numbers were wrong**
— ROC-AUC 1.0 for cooler, but *below random* for valve (0.077), pump
(0.292), accumulator (0.107). Traced to reconstruction error correlating
strongly with `cycle_idx`/campaign time (|r| up to 0.83), because the
chronological train/val split (kept from decision 1's fix) confined each
detector's healthy TRAIN/VAL data to one narrow calendar-time slice —
cooler's 1.0 was very likely the same time confound working in its favor
by coincidence, not real detection. See `notes/09_hydraulic_notes.md`
section 2's "Block-shuffled split" subsection for the full diagnostic.
Fixed by block-shuffling the split (shuffle 20-cycle blocks before
partitioning, instead of taking chronological slices) — spreads
train/val/test across the full campaign, still keeps short-range
adjacent-cycle structure intact within a block.

**Kernel v3 (2026-08-27): ran to completion, results are now coherent.**
(One operational gotcha along the way: the first pull of v3's output
silently returned v2's stale reports — `kaggle kernels output` skips any
file it thinks has a newer local copy, which is every source file we'd
just edited. `scripts/16_hydraulic_kaggle_pull.sh` now always passes
`--force`; re-pulling with it gave the real numbers below.)

| Component | ROC-AUC | PR-AUC | Recall @ 99th-pct threshold |
|---|---|---|---|
| cooler | 1.000 | 1.000 | 1.000 |
| valve | 0.889 | 0.976 | 0.000 |
| pump | 0.656 | 0.895 | 0.002 |
| accumulator | 0.867 | 0.990 | 0.375 |

All four ROC-AUCs are now above random (previously 3 of 4 were *below*
random — the time confound was actively inverting the signal), and PR-AUC
is high across the board (0.89–1.0) — genuine evidence the reconstruction
error DOES rank degraded cycles higher than healthy ones for every
component, cooler most cleanly. But recall at the actual deployed
threshold (99th percentile of validation-healthy error) is near zero for
valve and pump: the score separates the classes well, the fixed 99th-pct
cutoff just doesn't transfer from CWRU's much cleaner separation to these
noisier distributions. That's a threshold-calibration problem, not a
broken detector — worth a threshold-sensitivity sweep (CWRU has one in
`analyze.py`; hydraulic doesn't yet, see `notes/09_hydraulic_notes.md`
section 5) before concluding valve/pump "don't work."

**Cooler's 1.0 still deserves a grain of salt.** Re-checking
`corr(cycle_idx, recon_error)` after block-shuffling: it dropped from
0.78–0.84 to 0.13–0.43 for valve/pump/accumulator (block-shuffling clearly
helped), but stayed at -0.77 for cooler (barely moved from -0.82). That
suggests `cooler_condition` may be genuinely correlated with campaign time
in the raw data itself — plausible if the physical cooler's wear
progressed roughly monotonically over the test period — rather than being
a split artifact block-shuffling can fix. Unresolved; worth investigating
before treating cooler's near-perfect score as clean evidence the fusion
architecture "solved" that component.

Not yet investigated: section 5's "second-order confound" note (each
detector's healthy population is selected on one column while the other
three vary freely) — worth checking, especially for pump and valve, before
trusting these numbers as clean single-component detection.
