# Vibration Anomaly Detection (MVP)

Unsupervised, vibration-based anomaly detection: an autoencoder is trained
only on healthy vibration signal and flags anything it reconstructs poorly
as a potential fault. Built as a learning project and architecture proof
for turbo pump predictive maintenance, using the CWRU bearing dataset as a
labeled stand-in (see `notes/00_project_overview.md` for the full framing,
and `PLAN.md` for how this plan differs from the original draft and why).

**The forward plan for the whole repository is [`MASTER_PLAN.md`](MASTER_PLAN.md)**
(2026-10-15). It unifies the earlier per-experiment plans, which are now kept
as the historical record.

**Three experiments live in this repo now.** Everything below this point is
the original single-sensor CWRU vibration MVP (`src/vibration_ad/`). A
second, independent experiment -- multi-sensor **fusion** anomaly
detection on the ZeMA hydraulic systems dataset (17 sensors at 3 different
sampling rates) -- lives in `src/hydraulic_fusion_ad/`; see
`HYDRAULIC_PLAN.md` and `notes/09_hydraulic_notes.md` for its own plan and
design notes, and the "Hydraulic sensor-fusion experiment" section near the
bottom of this file for how to run it. A third, independent experiment --
**zero-shot fault diagnosis via domain adaptation** (DANN), combining CWRU
with the Paderborn/KAt-DataCenter bearing dataset -- lives in
`src/domain_adaptation/`; see `DOMAIN_ADAPTATION_PLAN.md` and the "Domain
adaptation experiment" section near the bottom of this file.

## Pipeline

```
raw signal (.npz)
  -> windowing (fixed-length overlapping chunks)
  -> STFT -> log-magnitude spectrogram (128 x 128 image)
  -> per-frequency-bin normalization
  -> conv autoencoder, trained on HEALTHY windows only
  -> reconstruction error = anomaly score
  -> threshold (percentile of validation healthy error)
  -> flagged anomaly
```

## Setup

This project needs its own venv with PyTorch — **you're setting this up
via pyenv (Python 3.12)**; once `.venv/` exists at the project root:

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e .   # makes `vibration_ad` importable
.venv/bin/python -m pytest             # smoke tests, no dataset needed
```

Every script in `scripts/` calls `.venv/bin/python` explicitly (no `source
activate` needed) — that also means they won't work until `.venv/` exists
at the project root with the above installed.

## Running the pipeline

The raw dataset is **already downloaded** (`data/raw/`, 64 files, ~200MB —
see `data/README.md`). Once the venv is ready:

```bash
bash scripts/02_train.sh      # train the autoencoder on healthy-only data
bash scripts/03_evaluate.sh   # ROC-AUC/PR-AUC/confusion matrix + plots
bash scripts/04_infer_demo.sh # reconstruction-error timeline for one healthy + one faulty file
bash scripts/09_analyze.sh    # latent-space PCA, threshold sensitivity, detection by RPM
bash scripts/10_sweep.sh      # optional, slow: empirical bottleneck-size (latent_dim) sweep
```

Or all at once (re-downloads data too, harmless if already present):
`bash scripts/run_all.sh`

Every stage reads `configs/default.yaml` — that one file is the single
source of truth for paths and every hyperparameter. Copy it and pass
`--config path/to/your.yaml` to any stage to run a variant without editing
the default.

### Outputs

- `outputs/models/` — `autoencoder_best.pt` (checkpoint), `normalizer.npz`
  (fit-on-train z-score stats), `run_config.yaml` (exact config used).
- `outputs/reports/` — `training_log.csv`, `evaluation_metrics.json`,
  `test_predictions.csv` (per-window error + prediction + true label),
  `detection_rate_by_severity.csv`.
- `outputs/figures/` — `error_histogram.png`, `roc_curve.png`,
  `detection_rate_by_severity.png`, `reconstruction_examples.png`, and
  per-file `infer_*_timeline.png` from `infer.py`; plus, from
  `analyze.py`/`sweep.py`: `latent_space_pca.png`, `confusion_matrix.png`,
  `detection_rate_by_rpm.png`, `threshold_sensitivity.png`,
  `latent_dim_sweep.png`.
- `outputs/sweep/latent_<N>/` — one full model/report/figure set per
  bottleneck size tested by `sweep.py`, plus `outputs/sweep/summary.csv`.

### Presentation report

`bash scripts/11_build_presentation.sh` renders
`reports/presentation_template.html` + the real results in
`outputs/kaggle_run/` (or `--run-dir outputs` for a local run) into one
self-contained `outputs/reports/presentation.html` — architecture
explanation, methodology, every figure above, and honest limitations, no
external dependencies. See `notes/08_deeper_analysis_and_presentation.md`.

### Inference on other files

```bash
.venv/bin/python -m vibration_ad.infer --config configs/default.yaml \
    --input data/raw/1772_OR@6_14_DE12.npz
```

Works on any `data/raw/*.npz` file (or any signal with the right sensor
key); `evaluate.py` must be run at least once first, since `infer.py`
reuses the threshold it calibrates.

## Project layout

See `notes/00_project_overview.md` for the full directory map and
module-by-module description. Short version: `src/vibration_ad/` is the
package (`data/`, `features/`, `models/`, plus `train.py`/`evaluate.py`/
`infer.py`/`viz.py`), `configs/default.yaml` drives everything, `scripts/`
are thin CLI wrappers, `notes/` is the project's design documentation.

## Notes / documentation

Read in order for the full picture, or jump to whichever's relevant:

1. `notes/00_project_overview.md` — framing, what changed from the original
   plan and why, directory map.
2. `notes/01_dataset_notes.md` — CWRU specifics, why this file subset, the
   train/val/test split strategy and why it's split the way it is.
3. `notes/02_signal_processing_notes.md` — STFT vs. Mel, window-length
   derivation, frequency resolution tradeoffs, normalization.
4. `notes/03_model_notes.md` — autoencoder architecture, why the bottleneck
   size is the key knob, training details.
5. `notes/04_evaluation_notes.md` — threshold selection, metric meanings,
   honest limitations.
6. `notes/05_glossary.md` — every technical term used above, defined.
7. `notes/06_extension_ideas.md` — menu of follow-up work, roughly ordered
   by effort.
8. `notes/07_dvc_and_kaggle.md` — data version control setup, and how the
   pipeline actually runs on Kaggle (CPU, offline, no local install).
9. `notes/08_deeper_analysis_and_presentation.md` — the analyze/sweep
   stages, what the bottleneck sweep actually found, and the presentation
   report.
10. `notes/09_hydraulic_notes.md` — the hydraulic sensor-fusion experiment's
    design notes.
11. `notes/10_domain_adaptation_notes.md` — domain adaptation concepts:
    what zero-shot means here, how DANN/GRL work, the three ways the
    adversarial approach failed, and why the metric you choose decides
    whether the result looks like a failure or a usable detector.
12. `notes/13_envelope_order_features.md` — the representation that
    actually crosses rigs, why spectrograms could not, and the bearing
    geometry that makes CWRU and Paderborn features comparable.
13. `notes/11_gpu_server_runbook.md` — **read before renting a GPU box
    again.** What the provider's machine already has (a working CUDA torch
    — do not build a venv), how slow the uplink is, and the six traps that
    cost real time on 2026-09-17. The box is DELETED at subscription end,
    so results must be pulled continuously.
14. `notes/12_reproduction_investigation.md` — why the arm A numbers did
    not reproduce on a second machine, the four hypotheses that were tested
    and refuted, and what is still open.

## Data versioning (DVC) and testing on Kaggle

`data/raw/` is DVC-tracked (`.venv/bin/dvc status`) rather than git-tracked
directly — see `notes/07_dvc_and_kaggle.md` for what that means and the one
thing still missing (a configured remote for `dvc push`/`pull`).

Local `.venv` has no PyTorch installed yet. Until it does, this project
runs `notebooks/kaggle_train.ipynb` on Kaggle via their API (already
authenticated on this machine) as the way to actually execute the
train/evaluate/infer pipeline:

```bash
bash scripts/05_kaggle_push.sh    # regenerate notebook from current source + push
bash scripts/06_kaggle_status.sh  # check run status (repeat until COMPLETE/ERROR)
bash scripts/07_kaggle_pull.sh    # download logs + output files to outputs/kaggle_run/
```

Data comes from the public `esraakhaled299/cwru-data` Kaggle dataset
(converted from `.mat` to our `.npz` convention in-notebook), not a
runtime download — this account isn't phone-verified, which disables both
internet access and GPU in kernels, so the pipeline runs CPU-only and
needs no internet at all. Live view:
https://www.kaggle.com/code/aghzuzu/vibration-anomaly-detection-mvp.

**First successful run (2026-08-20)**: ROC-AUC 1.0, 100% detection across
every fault type/severity and every load condition, full results in
`outputs/kaggle_run/`. Full narrative — three real bugs found and fixed,
two dead-end workarounds, and why the numbers are this clean — in
`notes/07_dvc_and_kaggle.md`. Deeper evidence (latent-space separation,
threshold sensitivity, an empirical bottleneck-size sweep) in
`notes/08_deeper_analysis_and_presentation.md`, or as one shareable report:
`bash scripts/11_build_presentation.sh`.

## Hydraulic sensor-fusion experiment

A second, independent experiment (`src/hydraulic_fusion_ad/`): multi-branch
sensor fusion on the [ZeMA hydraulic systems dataset](https://www.kaggle.com/datasets/jjacostupa/condition-monitoring-of-hydraulic-systems)
(17 sensors at 1Hz/10Hz/100Hz). **Four separate healthy-only fusion
autoencoders**, one per monitored component (cooler/valve/pump/
accumulator) -- not one combined detector; see `HYDRAULIC_PLAN.md` decision
1 for why (the first real run found only 20 of 2205 cycles are healthy on
all four components at once, nowhere near enough to train on). Full design
in `HYDRAULIC_PLAN.md` and `notes/09_hydraulic_notes.md`. Same
config-driven, Kaggle-first pattern as the CWRU pipeline above, entirely
separate paths per component (`configs/hydraulic_{cooler,valve,pump,
accumulator}.yaml`, `outputs/hydraulic_*/<component>/`) so the four
detectors and the CWRU pipeline never collide:

```bash
.venv/bin/python -m pytest tests/test_hydraulic_smoke.py   # plumbing only, synthetic data

bash scripts/14_hydraulic_kaggle_push.sh    # regenerate notebook + push (trains + evaluates all 4)
bash scripts/15_hydraulic_kaggle_status.sh  # poll until COMPLETE/ERROR
bash scripts/16_hydraulic_kaggle_pull.sh    # download results to outputs/hydraulic_kaggle_run/
```

Once real data exists locally (`data/hydraulic_raw/`, see its `README.md`),
`scripts/12_hydraulic_train.sh <component>` / `13_hydraulic_evaluate.sh
<component>` (component = `cooler`/`valve`/`pump`/`accumulator`) mirror
`02_train.sh` / `03_evaluate.sh` above, one component at a time.

**Status**: first Kaggle run (2026-08-27) failed on the all-four-healthy
design above; fixed by the per-component redesign -- see
`HYDRAULIC_PLAN.md`'s "Status" section (now several kernel versions in)
before trusting any numbers from this experiment.

## Domain adaptation experiment (zero-shot fault diagnosis)

A third, independent experiment (`src/domain_adaptation/`): trains a fault
classifier on CWRU + Paderborn University's KAt-DataCenter bearing dataset
(a Domain-Adversarial Neural Network -- feature extractor + classifier +
domain discriminator behind a Gradient Reversal Layer, plus a
healthy-anchor alignment loss), adapts it to a held-out Paderborn operating
condition using ONLY that condition's healthy vibration data, then checks
whether it can diagnose that condition's real bearing faults it never
trained on -- the "zero-shot" claim. Full plan, dataset verification, and
every architecture/scope simplification (and why) in
`DOMAIN_ADAPTATION_PLAN.md`. Same config-driven, Kaggle-first pattern as
the other two experiments; the Paderborn dataset (9.7GB) is attached as a
Kaggle input dataset, never downloaded in full:

```bash
.venv/bin/python -m pytest tests/test_domain_adaptation_smoke.py   # plumbing only, synthetic data

bash scripts/19_domain_adaptation_kaggle_push.sh    # regenerate notebook + push (baseline -> DANN -> evaluate)
bash scripts/20_domain_adaptation_kaggle_status.sh  # poll until COMPLETE/ERROR
bash scripts/21_domain_adaptation_kaggle_pull.sh    # download results to outputs/domain_adaptation_kaggle_run/
```

### What it found

Five Kaggle runs in, the answer is not the one the plan set out to prove,
and it is worth stating plainly:

| | faulty recall | false-alarm rate |
|---|---|---|
| baseline, 7 source bearing codes | 0.403 | 31.1% |
| **baseline, 18 source bearing codes** | **0.761** | **16.7%** |
| DANN (adapted), 18 codes | 0.085 | 9.7% |

1. **Zero-shot fault DETECTION transfers; fault TYPING does not.** No
   version could tell inner-race from outer-race damage on an unseen
   machine/speed. Healthy-vs-faulty -- the question a real deployment asks
   -- is answerable: **95.4% precision, 76.1% recall** on a rig speed the
   model never trained on, using zero target fault data.
2. **Source diversity beat every other lever.** Widening the source from 7
   to 18 bearing codes nearly doubled detection *while halving* false
   alarms. That single config change outperformed three rounds of
   adversarial hyperparameter tuning combined.
3. **The domain adaptation itself did not work.** Across five runs the DANN
   model never beat the plain baseline and usually destroyed it -- the
   adapted v5 model calls 91% of everything healthy. That is a real
   negative result, established with a diagnostic (a post-hoc separability
   probe) that distinguishes "features aligned" from "the discriminator was
   fooled". **The deployable model is the baseline; adaptation is kept as
   the documented negative result.**

Full reasoning, every failed hypothesis, and the two infrastructure bugs
found along the way are in `DOMAIN_ADAPTATION_PLAN.md`'s Status section.

**Current phase (2026-10-15):** the forward plan is
[`MASTER_PLAN.md`](MASTER_PLAN.md). (`GPU_SERVER_PLAN.md`, the 2026-09-17 plan
for the strict cross-dataset question, is now history.)

> **WARNING (2026-09-17): the numbers in the table above are single runs,
> and the run-to-run spread is larger than the differences being compared.**
> Six identical-config runs differing only in random seed produced target
> recall from **0.42 to 0.97** (mean 0.763 -- essentially the 0.761 quoted
> below). Source validation accuracy was 0.9997 in every one and predicts
> nothing about target performance. Treat every comparison here as
> indicative until replicated; see `GPU_SERVER_PLAN.md` section 1b.

**Read the table above with a second caveat: it also answers an easier
question than the project's actual goal.** Those runs trained on CWRU *plus Paderborn's
artificial faults* and tested on Paderborn's real faults — the same rig,
sensors and DAQ as the target, differing only in bearing unit. The real
question is harder: train on one dataset, adapt using nothing but healthy
data from a completely different machine, and diagnose its faults. That is
now enforced by `pu.include_in_source: false`, and the strict numbers
should be expected to come out lower.

Two further things to know:

- **"DANN is harmful" is under-powered.** It rests on three hyperparameter
  points at one seed each. A real sweep either confirms it or overturns it;
  if overturned, the table above gets corrected.
- **Order tracking was implemented and then measured to be worse.**
  Angular resampling (`data/order_tracking.py`, `common.representation:
  order`) puts a fault at the same *order* regardless of shaft speed,
  removing the cross-speed gap by construction. It validated beautifully on
  synthetic signals (spectrogram correlation between 1500 and 900 rpm:
  0.799 -> 0.978) and lost on real data at 8 seeds per arm -- see the arm A
  table below. The synthetic test used a pure order-locked tone, i.e.
  exactly what the transform is built to align, so it could not have
  failed. Real bearings also carry structural resonances that are fixed in
  **Hz**, and angular resampling smears those.

### Arm A: the strict cross-dataset result (2026-09-17, n=8 per arm)

Train on **CWRU only** -- no Paderborn data of any kind in training --
adapt using nothing but Paderborn healthy windows, then diagnose
Paderborn's real bearing damage. Unseen machine, unseen sensors, unseen
DAQ, unseen lab.

| representation | recall | precision | false alarm | F1 | balanced acc |
|---|---|---|---|---|---|
| **time** | **0.812 +- 0.159** | **0.943 +- 0.022** | 22.3% +- 9.4% | **0.865 +- 0.098** | **0.795 +- 0.078** |
| order | 0.606 +- 0.218 | 0.948 +- 0.026 | 15.9% +- 10.1% | 0.719 +- 0.155 | 0.723 +- 0.078 |

> **RETRACTED (2026-09-17). This table did not reproduce and is now
> believed to be a caching artifact.** A clean-room rebuild on a second
> machine collapsed completely -- every target window assigned to one class,
> identically across 8 seeds. The permissive control (Paderborn faults IN
> training) reproduced its historical result on the same box to within a
> percentage point, so the pipeline is sound; only the strict setup fails.
>
> The tell was visible in the original numbers: 0.812 +- 0.159 strict
> against 0.763 +- 0.19 permissive is no difference at all, which was
> written up at the time as "holding PU out cost nothing measurable". The
> simpler reading is that PU was never held out -- the window cache was
> keyed on file existence with no config hash, so arm A reused windows
> built while `include_in_source: true`. Full account in
> `notes/12_reproduction_investigation.md`; the cache is now guarded by a
> config fingerprint.
>
> **Where this leaves the project's claim:** zero-shot detection on an
> unseen machine, sensors, DAQ and lab is **NOT supported**. With this
> architecture, on these two datasets, detection requires target-rig fault
> data in training. What transfers across bearing units within one rig does
> not transfer across datasets.
>
> Three independent lines of evidence, all pointing the same way:
>
> | test | result |
> |---|---|
> | strict CWRU -> Paderborn (11 runs, 2 torch versions) | collapsed to one class **11/11** |
> | permissive, same pipeline (3 runs) | collapsed **0/3** — it works, if noisily |
> | strict Paderborn -> CWRU, a 4x richer source (5 runs) | 100% false alarms, balanced acc **0.499** |
> | per-window normalization, which provably removes the scale mismatch | balanced acc **0.499–0.505** |
>
> The reverse direction matters most: Paderborn offers 18 bearing units
> across 4 operating conditions against CWRU's 4 rpms, and still not one
> healthy CWRU window out of ~2730 is identified. So the ceiling is not
> source diversity, and it is not preprocessing.
>
> **But the conclusion is about the REPRESENTATION, not the premise.** A
> log-spectrogram is dominated by structural resonances, which sit at fixed
> Hz and belong to the rig; a CNN trained on one machine largely learns what
> that machine sounds like. Replacing it with envelope-order features --
> which discard the rig-specific carrier and keep only the bearing's own
> fault orders -- makes the strict transfer work:
>
> | fit on CWRU only, calibrated on target HEALTHY windows alone | recall | FA | balanced acc |
> |---|---|---|---|
> | envelope-order features @10% target FA | **0.746 ± 0.012** | 10.1% | **0.823 ± 0.007** |
> | every spectrogram variant tried | — | — | 0.499–0.505 (chance) |
>
> So zero-shot detection across rigs IS feasible; it just is not feasible by
> learning a spectrogram end-to-end. Details, per-unit breakdown and the
> honest limits are in `notes/13_envelope_order_features.md`.

Fault *typing* still does not transfer (3-class accuracy 0.378). The
deployable claim remains detection, not diagnosis.

Paderborn reaches the server as a **437 MB condensed bundle** rather than
the 9.7 GB raw dataset: same 2,559 runs and every bearing code, but keeping
only the `vibration_1` channel at 12 kHz float32 -- which is what the
pipeline resamples to anyway, so it is lossless for our use. Built by a
free Kaggle kernel (`scripts/build_kaggle_notebook_prepare_pu.py`); read
automatically, since `pu_loader.is_prepared_store` detects the condensed
form by its manifest.

### Deploying it on your own machine (Phase 5)

`infer.py` runs the detector on a new machine's vibration data at whatever
sampling rate it records, and reports healthy-vs-faulty per recording:

```bash
# 1. Calibrate against a recording you KNOW is healthy.
.venv/bin/python -m domain_adaptation.infer --input healthy_baseline.npy \
    --fs 25600 --calibrate

# 2. Then monitor.
.venv/bin/python -m domain_adaptation.infer --input todays_recording.npy --fs 25600
```

Calibration is not optional ceremony. The dominant error mode is
**per-unit, not random**: on the target condition every false alarm came
from ONE of six healthy bearing units (100% of its windows), while the
other five were essentially clean. A machine that reads as faulty when
healthy does so consistently -- so you can detect that with healthy data
alone, before any fault exists, which is exactly what `--calibrate` checks
and what the zero-shot premise gives you.

## Tests

`tests/test_smoke.py` uses synthetic data (no dataset download required) to
check the plumbing end to end: window shapes, STFT output shape,
normalizer round-trip, model forward pass, one real gradient step. Run with
`.venv/bin/python -m pytest`.

`tests/test_hydraulic_smoke.py` does the same for the hydraulic fusion
experiment: catalog parsing/validation, chronological split correctness,
per-sensor normalizer round-trip, multi-branch model forward pass, one real
gradient step -- all against small synthetic data, no download needed.
