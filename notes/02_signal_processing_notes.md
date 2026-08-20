# Signal processing notes

## STFT, not Mel-spectrogram

Your draft proposed a Mel-spectrogram via `librosa`. Mel scaling
re-bins frequency to match *human pitch perception* (roughly logarithmic,
denser resolution at low frequencies, coarser at high) — a good idea for
speech/music, where the downstream consumer is a human ear or a model
trained to mimic one. Bearing faults don't care about human hearing: their
signature lives at specific mechanical frequencies (ball-pass frequencies,
shaft-rate harmonics — see the table below), which can be anywhere across
the spectrum depending on bearing geometry and speed. Mel binning would
blur exactly the frequency detail that separates fault types/severities
from each other, for no benefit. A plain linear-frequency STFT
(`scipy.signal.stft`, see `features/spectrogram.py`) keeps full, uniform
resolution across the band and drops the `librosa` dependency entirely.

## Window length: what 8384 samples actually buys you

`configs/default.yaml`'s `windowing.segment_length: 8384` (at 12kHz =
0.699s) and `stft.nperseg: 256` / `hop_length: 64` were picked together to
land on an exact, round `(128, 128)` spectrogram image (see
`features/spectrogram.expected_shape` — the derivation is a comment right
above it) so the conv autoencoder's four stride-2 layers halve the input
cleanly with no padding/cropping logic needed anywhere in the pipeline.

That constraint still leaves real tradeoffs, worth understanding rather
than just trusting the defaults:

- **Frequency resolution** = `fs / nperseg` = 12000/256 ≈ **46.9 Hz/bin**.
  Finer resolution (bigger `nperseg`) means better separation between
  close-together fault frequencies, at the cost of *time* resolution
  (longer window needed per STFT frame) and a bigger, slower model input.
- **Time coverage** = 0.699s ≈ **~21 shaft revolutions** at 1797 RPM (29.95
  Hz shaft rate) — enough for periodic fault impacts (which repeat once or
  a few times per revolution) to show up as a stable pattern rather than
  noise, without the window getting so long that transients from very
  different operating moments get smeared into one image.

Approximate bearing fault frequencies for this rig's bearings at 1797 RPM
(shaft rate ≈ 29.95 Hz), rounded from the standard SKF-6205 multipliers:

| Frequency | ~Hz | Fits in one 46.9Hz bin? |
|---|---|---|
| FTF (cage) | ~11.9 | blurs into the DC/shaft-rate bin |
| BSF (ball spin) | ~70.6 | yes, own bin |
| BPFO (outer race) | ~107.3 | yes, own bin |
| BPFI (inner race) | ~162.2 | yes, own bin |

FTF being under-resolved is a real limitation of this config, but low-
stakes for this specific dataset: CWRU only has IR/ball/outer-race faults,
not cage faults, so nothing we're trying to detect actually lives at FTF.
If you extend this to a machine/dataset where cage faults matter, that's a
sign to lower `nperseg` or otherwise revisit this tradeoff.

## Normalization: per-frequency-bin, fit on train only

`SpectrogramNormalizer` (in `features/spectrogram.py`) z-scores each
frequency bin independently (subtract that bin's mean, divide by that
bin's std), rather than one global mean/std for the whole spectrogram.
Vibration energy is heavily concentrated at low frequencies; a single
global scale would leave high-frequency bins — often where subtle fault
harmonics live — numerically flat and nearly invisible to the loss
function. Per-bin normalization gives every frequency comparable dynamic
range going into the network.

The mean/std are fit **only on the training split** (healthy windows from
1730/1750/1772 RPM) and then applied as-is to val/test — never refit on
data the model shouldn't have "seen." This mirrors the same leakage
discipline as the train/val/test split itself (`notes/01_dataset_notes.md`).

## What this project doesn't do (and a well-known classic alternative)

The classic non-deep-learning approach to bearing fault detection is
**envelope analysis**: band-pass filter around a high-frequency resonance,
take the Hilbert transform envelope, then FFT that envelope to look for
energy exactly at BPFO/BPFI/BSF and their harmonics. It's precise,
physically interpretable, and doesn't need training data at all — but it
requires knowing the bearing geometry (to compute expected fault
frequencies) and degrades if the operating speed varies. We use a generic
spectrogram + autoencoder instead specifically because it *doesn't* require
per-machine geometry knowledge — closer to what you'd want for "point this
at a new turbo pump without hand-deriving its fault frequencies first." See
`notes/06_extension_ideas.md` for combining the two.
