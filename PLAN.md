# Plan: original vs. revised, and why

> **Status 2026-10-15: historical record. The forward plan is [MASTER_PLAN.md](MASTER_PLAN.md).**

## Your original plan

1. Download CWRU (via CWRU_Bearing_NumPy).
2. Time-frequency transform: STFT or Mel-spectrogram (via `librosa`).
3. Autoencoder trained only on healthy CWRU data.
4. Detection via reconstruction error vs. a threshold; visualize it.

The overall shape was right — this is a legitimate, standard approach to
unsupervised vibration anomaly detection, and a good one to learn on. What
follows is what I changed and, more importantly, what wasn't specified yet
that a *working* MVP actually needs.

## What changed

| Your plan | Revised | Why |
|---|---|---|
| Mel-spectrogram via `librosa` | Plain linear STFT via `scipy.signal.stft` | Mel scaling optimizes for human pitch perception; it blurs exactly the frequency resolution that separates bearing fault signatures from each other, for no benefit here. Also removes a dependency. Full reasoning: `notes/02_signal_processing_notes.md`. |
| "Train on healthy data" (unspecified split) | Explicit train/val/test split **by file/RPM**, not by random window | Windows overlap heavily by design (for training-data volume) — a random split would leak near-duplicate windows across train/val/test and inflate reported performance. Full reasoning: `notes/01_dataset_notes.md`. |
| "If error spikes past a certain threshold" (unspecified how) | Threshold = a percentile of reconstruction error on a **held-out healthy validation set**, chosen *before* looking at any fault label | This is the one place it'd be easy to accidentally cheat (tune the threshold against known fault labels) and get numbers that don't reflect a real blind deployment. Full reasoning: `notes/04_evaluation_notes.md`. |
| "We don't have labels" | CWRU labels used, deliberately, **only to grade the finished detector** in `evaluate.py` — never seen by training or thresholding | CWRU is actually labeled (that's what makes it possible to know whether any of this works); the exercise is building as if blind, then checking against ground truth. Stated explicitly so it's never accidentally load-bearing elsewhere. `notes/00_project_overview.md`. |
| (not specified) | Healthy training data **pooled across all 4 RPM/load conditions**; faulty data from all 4 used for evaluation | More realistic (a machine doesn't run at one fixed load) and meaningfully more training volume, at no extra cost — the RPM-specific fault files were never going into training anyway. |
| (not specified) | Exact windowing/STFT parameters chosen together to produce a fixed 128×128 spectrogram, derived and explained, not just asserted | The conv autoencoder architecture depends on a fixed, evenly-divisible input shape; `notes/02_signal_processing_notes.md` shows the derivation so it's not a mystery why these numbers. |
| "Visualizing Autoencoder Reconstruction Error" | `evaluate.py` (histogram, ROC curve, per-severity detection rate, example reconstructions) **and** `infer.py` (reconstruction-error-over-time for one file, the demo you described) | Splits "how good is this detector overall" (evaluate) from "watch it run on one signal" (infer) — both were implied by your plan but are different questions with different outputs. |
| (not specified) | Config file (`configs/default.yaml`) as single source of truth; caching of processed data; CLI entrypoints per stage; smoke tests | Baseline engineering an MVP needs to actually be reproducible and re-runnable, not just a one-off script. |

## Phases (as actually implemented)

**Phase 0 — Data.** Download the 12kHz Drive-End + Normal-baseline `.npz`
files (64 files, ~200MB) from CWRU_Bearing_NumPy. Already done — see
`data/raw/` and `data/README.md`.

**Phase 1 — Windowing + time-frequency transform.** Cut each long recording
into overlapping ~0.7s windows, STFT each into a 128×128 log-magnitude
spectrogram, normalize per frequency bin using training-set statistics.
(`src/vibration_ad/data/windows.py`, `features/spectrogram.py`)

**Phase 2 — Autoencoder.** Conv encoder → 128-dim bottleneck → conv
decoder, trained with MSE loss on healthy-only spectrograms, early-stopped
on held-out healthy validation loss. (`models/autoencoder.py`, `train.py`)

**Phase 3 — Detection + evaluation.** Reconstruction error as anomaly
score; threshold from validation-healthy error percentile; graded against
CWRU's true fault labels (ROC-AUC, PR-AUC, confusion matrix, detection rate
by fault type/severity); visualized (histograms, ROC curve, example
reconstructions, per-file error timelines). (`evaluate.py`, `infer.py`,
`viz.py`)

## What's still open

- The venv (PyTorch via pyenv Python 3.12) — you're setting this up; I
  couldn't run/test the training or evaluation code myself as a result.
  Once it's ready: `pip install -r requirements.txt && pip install -e .`,
  then `python -m pytest` first (fast, no dataset needed) before the full
  `scripts/run_all.sh`-style pipeline (which takes longer — real training).
- Hyperparameters in `configs/default.yaml` (`latent_dim`, `base_channels`,
  `threshold_percentile`, etc.) are reasoned-through starting points, not
  tuned — `notes/03_model_notes.md` explains how to read the diagnostic
  plots to adjust them once you can actually train.
- See `notes/06_extension_ideas.md` for what's deliberately out of scope
  for an MVP but worth developing further, per your stated learning goal.
