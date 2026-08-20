# Dataset notes: CWRU bearing data

## What it actually is

Case Western Reserve University Bearing Data Center dataset: a 2HP electric
motor test rig with accelerometers near the motor bearings. Bearings were
seeded with faults via electro-discharge machining (a precisely-sized
defect cut into the inner race, outer race, or a ball), then reinstalled
and run under four load conditions (0–3 HP, which correspondingly changes
the RPM slightly: 1797/1772/1750/1730). It is **not** turbo pump data — it's
the closest widely-available, well-labeled, real (not simulated) rolling-
element-bearing vibration dataset, and bearings are a component turbo pumps
also depend on, so the signal-processing/ML pipeline transfers even though
the specific machine doesn't.

We use the [CWRU_Bearing_NumPy](https://github.com/srigas/CWRU_Bearing_NumPy)
mirror (original data is MATLAB `.mat`; this repo converts to `.npz` with
cleaned-up metadata) rather than the original site — same data, easier to
load, publication-cited.

## Why DE12 + Normal, and not DE48/FE

The dataset has three variants of accelerometer position/rate: Drive-End at
12kHz (DE12), Drive-End at 48kHz (DE48), Fan-End at 12kHz (FE). We use
**DE12 for faults, paired with the Normal baseline files** (which were
themselves recorded at 12kHz), because:

- It's the one combination present for *every* fault type/size/RPM combo in
  the dataset (DE48 and FE both have gaps — e.g. some 28-mil fault files
  only exist as DE12).
- One consistent sampling rate means one consistent `configs/default.yaml`
  windowing/STFT setup for the whole dataset — no per-file special-casing.
- It's the historically standard pairing in CWRU-based papers, so if you
  read other work on this dataset later, DE12+Normal is what "the CWRU
  dataset" usually means by default.

48kHz DE data exists and would give better frequency resolution — worth
trying as an extension (see `notes/06_extension_ideas.md`) once the 12kHz
pipeline works end to end.

## Fault taxonomy

| Code | Location | Notes |
|---|---|---|
| `Normal` | — | Healthy baseline, no seeded fault |
| `IR` | Inner race | Fault rotates with the shaft |
| `B` | Ball (rolling element) | |
| `OR@6` | Outer race, 6 o'clock (in the load zone) | Most commonly cited OR variant |
| `OR@3` | Outer race, 3 o'clock (orthogonal to load) | |
| `OR@12` | Outer race, 12 o'clock (opposite the load zone) | |

Each fault type comes in 1–4 severities (7/14/21/28 mils diameter — bigger
means more severe). This is what `evaluate.py`'s per-severity detection
rate breaks down by: a well-behaved detector should catch 28-mil faults
more reliably than 7-mil ones, since a larger seeded defect produces a
stronger vibration signature. If it doesn't, that's a real signal something
in the pipeline (window length, frequency resolution, threshold) needs
tuning — not just a number to report.

## Split strategy: why by file/RPM, not by random window

`windowing.py` cuts each long recording into overlapping fixed-length
windows (see `notes/02_signal_processing_notes.md` for why 8384 samples).
With `train_stride=512` on an ~8384-sample window, consecutive windows
share >90% of their samples — they are *not* independent examples. If you
randomly shuffled all windows into train/val/test, near-duplicates of a
training window would end up in val/test, and the reported metrics would be
measuring memorization, not generalization.

`dataset.py` avoids this at the file/condition level instead:

- **Train** (healthy only): every window from `1730_Normal.npz`,
  `1750_Normal.npz`, `1772_Normal.npz` — dense overlap for volume (~2,800
  windows total).
- **Val** (healthy only): the *first half* of windows from
  `1797_Normal.npz`, held out entirely from training.
- **Test** (healthy + faulty): the *second half* of `1797_Normal.npz`'s
  windows, plus every window from every `*_DE12.npz` fault file across all
  four RPM families (faulty data is never touched during training,
  regardless of which RPM it came from).

The val/test split of `1797_Normal.npz` is a contiguous split (first N
windows vs. last M), not a shuffle — a shuffled split would put windows
adjacent in time (and therefore highly overlapping) on both sides of the
val/test boundary, which is the same leakage problem one level down.

This also means the detector is evaluated partly on a load condition (1797
RPM / 0 HP) it never saw a single healthy training example from directly —
a mild but real generalization test, closer to how a deployed model would
actually need to behave (load varies; you can't retrain per operating
point).
