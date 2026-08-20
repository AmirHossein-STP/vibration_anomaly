# Deeper analysis, and the presentation report

## What's here

Beyond `train.py` → `evaluate.py`'s core metrics (ROC-AUC, PR-AUC, confusion
matrix, detection-by-severity — see `notes/04_evaluation_notes.md`), two
more analysis stages, added when the natural next question after "does it
work" became "show me more evidence, and let me present it":

- **`src/vibration_ad/analyze.py`** (`scripts/09_analyze.sh`) — reuses
  evaluate.py's saved checkpoint/normalizer/test set, adds:
  - `latent_space_pca.png` — 2D PCA of the autoencoder's 128-number
    bottleneck vector itself, colored by true label (never shown to the
    model). Answers "is the *compression* learning something about
    healthy-vs-faulty, or is separation only happening in the final error
    scalar downstream of it?"
  - `confusion_matrix.png`, `detection_rate_by_rpm.png` — generalization
    check across load conditions (the RPM the model's training data never
    included, see `notes/01_dataset_notes.md`).
  - `threshold_sensitivity.png` / `.csv` — precision/recall/F1 recomputed
    across percentiles 80–99 (all still unsupervised choices), to see if
    the deployed 99th-percentile pick is a fragile cliff-edge or a stable
    plateau.
- **`src/vibration_ad/sweep.py`** (`scripts/10_sweep.sh`) — empirically
  tests the bottleneck-size reasoning in `notes/03_model_notes.md` instead
  of just asserting it: trains + evaluates 8 models across
  `latent_dim` ∈ {4,...,512}. Each variant gets its own
  `outputs/sweep/latent_<N>/` so a normal run's `outputs/models`,
  `outputs/reports` are untouched; the expensive windowing/STFT step isn't
  repeated per variant (cached, see `data/dataset.build_raw_datasets`).

Both are wired into `notebooks/kaggle_train.ipynb` (regenerate via
`scripts/build_kaggle_notebook.py` after any source change) as steps 7–8,
after the core train/evaluate/infer steps.

## What the sweep actually found

Not the inverted-U curve the reasoning predicted. See
`notes/03_model_notes.md`'s updated bottleneck section for the honest
result: aggregate metrics saturate at ROC-AUC 1.000 across the *entire*
range tested, and the predicted pattern only shows up as a faint signal in
raw false-positive counts. Worth reading if you're tempted to skip the
sweep on a new dataset because "it didn't matter last time" — it mattered
this little specifically *because* CWRU's healthy/faulty gap is unusually
wide (see `notes/04_evaluation_notes.md`'s limitations); that's not
guaranteed to hold elsewhere.

## The presentation report

`reports/presentation_template.html` (HTML with `{{IMG:name}}` /
`{{TXT:name}}` placeholders) + `scripts/build_presentation.py` (fills them
in from a real run's `outputs/`, base64-embedding every figure so the
result is one self-contained file with no external dependencies) →
`outputs/reports/presentation.html`, or `bash scripts/11_build_presentation.sh`.

Deliberately *not* just the raw matplotlib PNGs dropped into a page: three
of them (`confusion_matrix.png`, `threshold_sensitivity.png`,
`latent_dim_sweep.png`) render as visually uninformative once the
underlying numbers saturate near 1.0 on a 0–1 axis (a real result, not a
bug — see above) — the report replaces those three with native
HTML/CSS components (a colored confusion-matrix grid, a stat callout with
the actual precision/recall range, and a small bar chart of the raw
false-positive counts) that show the real signal instead of a flat line.
Everything else (error histogram, ROC curve, detection-rate bars, latent
space scatter, reconstruction examples, inference timelines) embeds the
actual generated PNGs as-is.

Regenerate any time `outputs/kaggle_run/` (or a local run's `outputs/`,
via `--run-dir`) has fresher results — the script reads real numbers out
of `evaluation_metrics.json` / `threshold_sensitivity.csv` /
`sweep/summary.csv` rather than hardcoding anything from this run, so a
different result (a different dataset, a code change, a re-run) produces
an accurate report, not a stale one.
