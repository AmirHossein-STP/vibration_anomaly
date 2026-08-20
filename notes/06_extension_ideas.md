# Extension ideas

Roughly ordered by effort. None of these are required for the MVP; this is
a menu for "I've read the code and want to develop it further" (per your
stated learning goal), not a to-do list.

## Small, self-contained
- **Sweep `latent_dim`/`base_channels`** and compare error histograms
  (`notes/03_model_notes.md`) — the fastest way to build real intuition for
  the compression tradeoff, no new code needed, just re-runs.
- **Try the 48kHz Drive-End files** (`*_DE48.npz`) instead of 12kHz — better
  frequency resolution, at the cost of updating `data.fs_hz` and
  re-deriving `windowing`/`stft` params for a new target spectrogram shape.
- **F1-optimal threshold as a *reported*, clearly-labeled second number**
  (never used for the actual detector) — a good exercise in seeing exactly
  how much "cheating" by peeking at test labels would have inflated the
  results, which is a useful thing to have felt once.
- **Per-frequency-bin error contribution plot** — instead of one scalar
  reconstruction error, plot `(input - recon)**2` averaged over time only
  (not frequency), to see *which* frequency bands the model finds
  surprising for a given faulty window. Cheap addition to `viz.py`, and a
  first step toward explainability (see below).

## Medium
- **Envelope-spectrum features as a second, classical detector**, run
  alongside the autoencoder (see `notes/02_signal_processing_notes.md`).
  Since CWRU bearing geometry is published, you can compute exact expected
  BPFO/BPFI/BSF and check energy there directly — then compare how often it
  agrees/disagrees with the learned model. Great for building trust in (or
  catching failure modes of) the learned approach.
- **VAE (variational autoencoder) instead of a plain autoencoder** — gives
  a probabilistic reconstruction error (log-likelihood) instead of raw MSE,
  which can be a more principled anomaly score. Meaningful architecture
  change to `models/autoencoder.py`, not just a hyperparameter.
- **Domain generalization check**: train on 3 of the 4 RPM families,
  evaluate healthy/faulty detection on the 4th (partially already done via
  the holdout split — extend it to systematically rotate which RPM is held
  out and compare).

## Larger
- **Trend-over-time / gradual degradation**: CWRU faults are instantaneous
  and fixed-severity (see `notes/04_evaluation_notes.md`'s limitations) —
  real predictive maintenance cares about a *worsening* trend, not a single
  reconstruction-error spike. Would need either synthetic degradation
  (blend healthy/faulty signal in increasing proportion) or a dataset that
  actually records run-to-failure data (e.g. NASA/IMS bearing dataset) to
  practice on.
- **Real turbo pump data.** Once available: everything in `configs/` needs
  re-deriving from that machine's actual sampling rate, shaft speed, and
  bearing/component geometry — the code structure (windowing → spectrogram
  → healthy-only autoencoder → calibrated threshold) is meant to carry over
  directly, but every *number* in the config is CWRU-specific right now and
  should not be assumed to transfer.
- **Streaming/online inference**: `infer.py` currently processes one whole
  file at a time; a real deployment would process a continuous stream and
  need decisions about buffering, latency, and possibly online threshold
  adaptation (e.g. as the machine ages or gets serviced).
