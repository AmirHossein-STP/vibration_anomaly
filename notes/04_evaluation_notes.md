# Evaluation notes

## Threshold selection: unsupervised on purpose

`evaluate.py` sets the anomaly threshold as the **99th percentile
(`evaluation.threshold_percentile`) of reconstruction error on the
validation healthy set** — no fault label involved in picking that number.
This matters: it's the one choice in this whole pipeline that has to stay
honest about "we don't have fault examples," because it's the one choice a
real deployment would actually have to make blind. It's easy to
accidentally cheat here (e.g. sweep the threshold against the test set's
true labels and report whichever gives the best F1) — that number would be
real, but it would be answering a different, less useful question ("what's
the best possible threshold, given the answers") than the one that matters
("what threshold can we actually pick without the answers").

Concretely, 99th percentile means: on healthy validation data, ~1% of
windows are expected to false-alarm by construction. That's a deliberate,
interpretable operating point, not a magic number — raise the percentile
for fewer false alarms (at the cost of missing smaller faults), lower it
for more sensitivity. `evaluate.py`'s `evaluation_metrics.json` records
whatever percentile was configured, so this choice is always visible
alongside the results it produced.

## What the metrics mean here

- **ROC-AUC / PR-AUC** (`sklearn.metrics.roc_auc_score` /
  `average_precision_score`) are computed over *all* candidate thresholds,
  independent of which one we actually pick — they answer "how well does
  reconstruction error rank healthy vs. faulty windows overall," which is a
  measure of the *representation* the autoencoder learned, not of the
  specific threshold choice above.
- **Precision / recall / F1 / confusion matrix** ARE threshold-dependent —
  computed at the one threshold chosen above, so these numbers do reflect a
  real, deployable operating point, not a best-case sweep.
- **Detection rate by fault type & severity**
  (`outputs/reports/detection_rate_by_severity.csv`, and the matching bar
  chart) is the most informative single output for actually understanding
  the model, more than the aggregate metrics above: it should show larger
  faults (28 mil) detected more reliably than smaller ones (7 mil), and
  probably some fault *types* detected more easily than others depending on
  how strongly they perturb the frequency bands the model resolves well
  (see `notes/02_signal_processing_notes.md`'s frequency-resolution table).
  If a fault type/severity you'd expect to be easy (e.g. 28-mil outer race)
  is detected worse than a smaller one, that's a real finding worth digging
  into, not noise to ignore.

## Why we're allowed to use CWRU's labels for grading at all

Restated from `notes/00_project_overview.md` because it's worth repeating
here specifically: CWRU filenames encode ground truth. We don't use it in
`train.py` (which only ever loads `*_Normal.npz` windows) or in the
threshold choice above — but `evaluate.py` uses it, deliberately, because
otherwise there is no way to know whether any of this actually detects
anything. This is standard practice for validating an unsupervised method:
build and calibrate it blind, then grade it on held-back ground truth you
didn't let it see. The output of `evaluate.py` tells you how well *this
approach* would likely have worked on a genuinely unlabeled deployment —
it is not itself a claim that a real, unlabeled turbo pump deployment would
reproduce these exact numbers.

## Known limitations, stated plainly

- **Single machine, lab conditions.** CWRU is one motor/bearing rig, not a
  turbo pump, and lab vibration is cleaner than a real industrial
  environment (less sensor noise, no other nearby rotating equipment
  cross-talk, no piping-induced vibration, etc). Expect a real deployment
  to need re-tuning (window length, threshold percentile, possibly
  architecture) against real baseline data before it's trustworthy.
- **Fault severities are synthetic and instantaneous.** EDM-seeded faults
  don't develop gradually the way real wear does; this pipeline has never
  been shown a *slowly worsening* fault, only "healthy" vs. "already
  faulty at a fixed severity." A real predictive-maintenance system cares
  a lot about trend over time (see `notes/06_extension_ideas.md`).
- **Class imbalance in the test set is by construction, not realistic.**
  The test set has far more faulty windows than a real deployment would
  ever see in a healthy-running pump (we intentionally included every
  fault file for evaluation power). PR-AUC is reported alongside ROC-AUC
  partly because PR-AUC is more sensitive to this kind of imbalance and a
  more honest single number if you had to pick one.
- **One sensor, one axis.** Real predictive maintenance systems often use
  multiple accelerometer axes/locations and combine them; this pipeline is
  intentionally single-sensor for simplicity.
