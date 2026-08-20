# Model notes: the autoencoder

## Architecture (`models/autoencoder.py`)

A standard convolutional autoencoder for 128×128 single-channel images
(the log-magnitude spectrograms):

```
input (1,128,128)
  -> Conv2d stride2 -> BN -> ReLU   x4   (128 -> 64 -> 32 -> 16 -> 8, channels 16 -> 32 -> 64 -> 128)
  -> flatten -> Linear -> latent (128-dim)
  -> Linear -> reshape (128, 8, 8)
  -> ConvTranspose2d stride2 -> BN -> ReLU   x3
  -> ConvTranspose2d stride2 (no activation)  -> output (1,128,128)
```

Conv layers use kernel 3 / stride 2 / padding 1, which exactly halves an
even spatial dimension (`out = in/2`, no rounding); the mirrored
`ConvTranspose2d` layers use kernel 4 / stride 2 / padding 1, which exactly
doubles it back (`out = 2*in`). That's *why* `ConvAutoencoder.__init__`
asserts the input shape is divisible by 16 (four halvings) — anything else
would need explicit padding/cropping to round-trip cleanly, which is extra
complexity this project doesn't need given the spectrogram shape is fixed
by config anyway.

No activation on the final layer: the target (a normalized spectrogram) is
roughly standard-normal-distributed, not bounded to `[0,1]` like a typical
image, so `sigmoid`/`tanh` on the output would actively hurt — the network
needs to be able to output any real value.

## The bottleneck size is the single most important knob

`model.latent_dim: 128` in `configs/default.yaml` compresses each 128×128
(=16,384 values) spectrogram down to 128 numbers — a 128x compression
ratio. This is the whole mechanism the anomaly detector depends on, so it's
worth being deliberate about, not just accepting the default:

- **Too small** a bottleneck: the network can't even reconstruct *healthy*
  signal well. Reconstruction error is uniformly high regardless of input,
  and the healthy/faulty error distributions overlap — poor separation, bad
  ROC-AUC, because there's no "quiet, healthy" baseline error to compare
  against.
- **Too large**: the network has enough capacity to reconstruct
  *anything* reasonably well, including faulty spectrograms it never
  trained on, because a large bottleneck stops forcing real compression —
  it starts approaching an identity function. Reconstruction error becomes
  uniformly low, and again poor separation, but for the opposite reason.

The useful regime is in between: small enough that the network is forced to
learn "what healthy vibration structure looks like" rather than memorizing
individual windows, large enough that healthy reconstruction is actually
good. There's no formula for the right number — `evaluate.py`'s error
histogram (`outputs/figures/error_histogram.png`) is the diagnostic: if the
healthy and faulty error distributions overlap heavily, try adjusting
`latent_dim` (and/or `base_channels`) before touching anything else.

## Training details worth knowing

- **Loss = reconstruction MSE**, averaged per-sample then over the batch
  (`models/autoencoder.per_sample_mse`) — deliberately the *same function*
  used later as the anomaly score in `evaluate.py`/`infer.py`, so "training
  loss" and "anomaly score" are conceptually the same quantity throughout,
  not two different things that happen to correlate.
- **Early stopping** (`train.early_stopping_patience: 10`) on validation
  (held-out healthy) loss, not training loss — stops training once the
  model stops improving at reconstructing *unseen* healthy data, which is
  the thing that actually matters for a good threshold later.
- **Device selection is automatic** (`models.autoencoder.select_device`):
  CUDA if available, else Apple Silicon MPS, else CPU. On a Mac this model
  is small enough (order of a few hundred thousand parameters) to train on
  CPU in a reasonable time regardless, but MPS will be noticeably faster if
  available.
- **Batch norm** is used throughout, which is why `model.eval()` /
  `model.train()` mode matters when running inference vs. training — both
  `evaluate.py` and `infer.py` call `.eval()` before computing reconstruction
  error (see `evaluate.compute_errors_and_recons`), otherwise BatchNorm
  would use per-batch statistics instead of the learned running statistics
  and reconstruction error would be noisy/inconsistent between runs.
