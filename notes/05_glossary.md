# Glossary

**STFT (Short-Time Fourier Transform)** — slides a small window along a
signal and takes an FFT of each chunk, producing a 2D image (spectrogram):
one axis is time, the other frequency, and each pixel's value is how much
energy that frequency had at that moment. Lets you see how the frequency
content of a signal changes over time, unlike a single FFT of the whole
signal which only tells you the average frequency content.

**Spectrogram** — the 2D output of an STFT (or similar transform). In this
project, always log-magnitude: `log(|STFT| + epsilon)`, because raw
magnitude is heavily right-skewed (a few huge peaks, lots of small values)
and log compresses that into a range a neural net trains on much better.

**Nyquist bin/frequency** — the highest frequency an STFT can represent
given its sample rate: `fs / 2`. We drop this single bin (see
`features/spectrogram.py`) purely to make the frequency axis a round
power-of-two-friendly number for the conv autoencoder; it's a negligible-
energy bin for this signal anyway.

**Autoencoder** — a neural net trained to reconstruct its own input,
through a bottleneck narrower than the input (see below). Forces it to
learn a compressed representation of whatever data it's trained on.

**Bottleneck / latent space / latent dim** — the narrowest layer of an
autoencoder, where the input has been compressed to a small vector (128
numbers here, from a 16,384-number input image). See
`notes/03_model_notes.md` for why its size is the key hyperparameter.

**Reconstruction error** — the (mean squared) difference between an
autoencoder's input and its output. The core anomaly signal in this
project: low for inputs similar to what the model trained on (healthy
spectrograms), higher for inputs unlike anything it trained on (faults).

**Overlap / stride / hop** — when cutting a signal into windows (or an STFT
frame into the next), *stride* (or *hop*) is how far to move forward each
time; *overlap* is how much consecutive windows share. Smaller stride =
more overlap = more (correlated) windows from the same data. We use dense
overlap for training data (volume) and sparse overlap for eval data (less
redundant/correlated evaluation examples). See `notes/01_dataset_notes.md`.

**Z-score / standardization** — rescaling data to zero mean, unit standard
deviation: `(x - mean) / std`. Used here per-frequency-bin
(`SpectrogramNormalizer`), fit only on training data.

**BPFO / BPFI / BSF / FTF** — characteristic bearing fault frequencies,
each a function of shaft rotation speed and bearing geometry (ball count,
diameter, contact angle):
- **BPFO** (Ball Pass Frequency, Outer race) — rate at which rolling
  elements pass a fixed point on the *outer* race. Elevated energy here (and
  its harmonics) indicates an outer-race fault.
- **BPFI** (Ball Pass Frequency, Inner race) — same idea, for the *inner*
  race (which rotates with the shaft, making its signature amplitude-
  modulated at the shaft rate).
- **BSF** (Ball Spin Frequency) — rate at which a single rolling element
  spins on its own axis; elevated energy here indicates a ball/rolling-
  element fault.
- **FTF** (Fundamental Train Frequency, aka cage frequency) — rate at which
  the entire cage (which holds the rolling elements in place) rotates;
  relevant for cage faults, which CWRU doesn't include (see
  `notes/02_signal_processing_notes.md`).

**Drive-End (DE) / Fan-End (FE) / Base (BA)** — accelerometer placement on
the CWRU test rig, relative to the motor. We use Drive-End throughout (the
sensor closest to where faults were seeded).

**Mils** — thousandths of an inch (0.001"), the unit CWRU uses for seeded
fault diameter (7/14/21/28 mils = 0.007"–0.028"). Larger = more severe.

**ROC-AUC** — Area under the Receiver Operating Characteristic curve
(true-positive rate vs. false-positive rate across all thresholds); 1.0 is
perfect ranking, 0.5 is no better than random.

**PR-AUC** (aka average precision) — area under the Precision-Recall curve;
more sensitive than ROC-AUC to class imbalance (see
`notes/04_evaluation_notes.md`), which is relevant here since our test set
has far more faulty windows than a real healthy-running deployment would.
