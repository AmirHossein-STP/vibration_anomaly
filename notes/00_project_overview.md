# Project overview

## The goal, and the honest framing

You asked for vibration-based predictive maintenance for turbo pumps,
without labels, so: anomaly detection. There's no turbo pump dataset here
(CWRU is bearing data from a lab motor test rig) — this project is a
**learning vehicle and architecture proof**: the same pipeline (windowing →
time-frequency transform → healthy-only autoencoder → reconstruction-error
threshold) is what you'd point at real turbo pump accelerometer data later.
Swapping in real data means revisiting almost every number in
`configs/default.yaml` (sampling rate, window length, RPM/frequency ranges)
but not the architecture.

One thing worth being precise about: CWRU is *not* actually unlabeled — the
fault type/size is encoded right in the filename. We deliberately don't use
that during training (the autoencoder only ever sees `*_Normal.npz`
windows, see `notes/01_dataset_notes.md`), to simulate the real situation —
a running pump you believe is healthy, no fault examples to learn from. The
labels come back into play *only* in `evaluate.py`, to grade the detector,
because otherwise there's no way to know if any of this actually works. A
real deployment wouldn't have that luxury; treat every evaluation number in
this project as "how well would this have worked, in a case where we
happen to know the answer," not a guarantee about a real machine.

## Why an autoencoder + reconstruction error (and what it's for)

Your original plan already had the right shape: train a network to
reconstruct *only* healthy signal, and use reconstruction error as the
anomaly score. That's a standard, well-understood approach for exactly this
situation (no fault examples, only a "this is normal" baseline) and it's a
good one to learn on, since it forces you to think about compression,
bottlenecks, and what "normal" even means numerically — all
transferable to other anomaly-detection problems (not just vibration).

What changed from your draft, and why:

1. **Mel-spectrogram → plain STFT.** Mel scaling exists to match human pitch
   perception; it compresses high-frequency resolution in a way that's
   actively counterproductive for mechanical fault frequencies (see
   `notes/02_signal_processing_notes.md`). Also drops the `librosa`
   dependency entirely — `scipy.signal.stft` is all we need.
2. **Explicit train/val/test split by file/RPM, not by random window.**
   Overlapping windows are highly correlated with their neighbors; a random
   split would leak near-duplicate windows across train/val/test and make
   the results look better than they are. See `notes/01_dataset_notes.md`.
3. **Threshold is chosen without looking at fault labels** (a percentile of
   validation-set healthy error), and *then* graded against labels — not
   the other way around. See `notes/04_evaluation_notes.md`.
4. **Healthy data pooled across all four load conditions (1730–1797 RPM)**
   for training, rather than one fixed operating point — closer to how a
   real machine actually runs, and meaningfully increases training data
   volume. Faulty data from all four load conditions is used for
   evaluation, since none of it was ever seen in training regardless of RPM.
5. Added the pieces your plan didn't cover but any working MVP needs: a
   concrete windowing/feature-shape design, a config file as the single
   source of truth, caching, a reproducible train/eval/infer CLI split, and
   an evaluation stage with real metrics (ROC-AUC, PR-AUC, per-severity
   detection rate) instead of just "look at the reconstruction error."

## Directory map

```
configs/default.yaml       single source of truth for every hyperparameter/path
data/raw/                  downloaded CWRU .npz files + MANIFEST.txt (tracked)
data/processed/            cached windowed spectrograms (gitignored, regenerable)
src/vibration_ad/          the actual package (see below)
scripts/                   thin bash wrappers around the package's CLI entrypoints
notebooks/                 exploratory notebook (optional, not part of the pipeline)
outputs/models/            trained checkpoint + normalizer stats + config snapshot
outputs/reports/           metrics json/csv, per-window predictions
outputs/figures/           all generated plots
notes/                     you are here
tests/                     fast synthetic-data smoke tests (no dataset needed)
```

`src/vibration_ad/` submodules:

- `data/` — download, filename parsing (`catalog.py`), windowing, the
  train/val/test split (`dataset.py`), and the torch `Dataset` wrapper.
- `features/spectrogram.py` — STFT + per-frequency-bin normalization.
- `models/autoencoder.py` — the conv autoencoder + reconstruction-error function.
- `train.py`, `evaluate.py`, `infer.py` — the three pipeline stages, each a
  runnable CLI module (`python -m vibration_ad.<stage>`).
- `viz.py` — every plotting function, shared by `evaluate.py`/`infer.py`.

## Definition of "MVP done"

1. `scripts/01_download_data.sh` — data downloaded (already done as part of
   building this project — see the raw file count in `data/raw/`).
2. `scripts/02_train.sh` — autoencoder trained on healthy-only windows,
   checkpoint + normalizer saved to `outputs/models/`.
3. `scripts/03_evaluate.sh` — ROC-AUC / PR-AUC / confusion matrix / per-fault-
   severity detection rate computed and saved; histogram + ROC + example
   reconstruction plots generated.
4. `scripts/04_infer_demo.sh` — reconstruction-error-over-time plot for one
   healthy and one faulty file, the "MVP demo" artifact.

Steps 2–4 need the venv you're setting up (PyTorch); step 1 is already done.
