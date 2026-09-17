# Envelope-order features: the strict transfer that actually works

The spectrogram pipeline cannot cross rigs. Eleven strict runs collapsed to
one class, arm B failed identically with a source four times richer, and
per-window normalization -- which provably removes the scale mismatch --
left balanced accuracy at 0.499. See `notes/12_reproduction_investigation.md`.

This note records what replaced it, and why it was the obvious thing to try
once the failure was understood physically rather than numerically.

## Why spectrograms could not have worked

A log-spectrogram of a bearing signal is dominated by the machine's
STRUCTURAL RESONANCES -- housing, shaft, mounting -- which sit at fixed
frequencies in Hz and are a property of the RIG. CWRU and Paderborn are
different machines, so a network trained on one has learned, in large part,
what that particular machine sounds like. Nothing about that description
transfers.

A bearing fault does not produce a tone. Each time a rolling element strikes
the defect it excites those resonances, so the fault shows up as AMPLITUDE
MODULATION of a resonance carrier. The modulation rate identifies the fault
and depends only on bearing geometry and shaft speed:

    BPFO = (n/2)(1 - d/D cos a) x shaft_rate      # outer race
    BPFI = (n/2)(1 + d/D cos a) x shaft_rate      # inner race

Envelope demodulation -- band-pass, Hilbert envelope, spectrum of the
envelope -- throws away the carrier (rig-specific) and keeps the modulation
(bearing-specific). **Discarding the carrier is the entire point**, and it is
exactly the information a spectrogram CNN was keying on.

## The one subtlety that makes it comparable

The fault ORDERS differ between the datasets, because the bearings differ:
CWRU runs a 6205 (BPFO 3.5848, BPFI 5.4152), Paderborn a 6203 (BPFO 3.0526,
BPFI 4.9474). Both derived from geometry in
`src/domain_adaptation/features/envelope.py` and matching each source's
published values exactly.

So the features are read at EACH dataset's OWN fault orders. The feature is
"how much envelope energy sits at this bearing's outer-race frequency", not
"how much sits at 3.585x". That is what makes a CWRU-trained coefficient
mean the same thing on Paderborn. Every feature is also a ratio against the
window's own broadband envelope energy, so sensor sensitivity and drive
level divide out.

The resonance band is fixed at 1.5-5.5 kHz a priori, never tuned against
target labels -- picking it by what separates the classes best would leak.

## Result

Fit on CWRU only. Never a Paderborn fault label. Threshold calibrated on
Paderborn HEALTHY windows alone, which is precisely the project's premise:
you have healthy recordings of the machine you want to monitor and no
faulty ones.

| operating point (10 calibration splits) | recall | false alarm | balanced acc |
|---|---|---|---|
| @5% target FA | 0.671 +- 0.013 | 5.3% +- 1.5% | 0.809 +- 0.002 |
| **@10% target FA** | **0.746 +- 0.012** | **10.1% +- 2.7%** | **0.823 +- 0.007** |
| @20% target FA | 0.812 +- 0.011 | 20.3% +- 4.3% | 0.804 +- 0.016 |
| *spectrogram pipeline, every variant* | -- | -- | *0.499-0.505 (chance)* |

Uncalibrated it still clears chance on its own (balanced 0.654), so the
calibration is sharpening a real signal rather than manufacturing one.

Within CWRU the same features give 0.971 +- 0.032 binary accuracy, so the
representation is sound in-domain too.

### Per bearing unit, at the 10% operating point

| healthy | flagged | | faulty | flagged |
|---|---|---|---|---|
| K001 | 0.01 | | KA04 | 0.93 |
| K002 | 0.14 | | KA15 | 0.23 |
| K003 | 0.03 | | KA16 | 1.00 |
| K004 | 0.03 | | KA22 | 0.39 |
| K005 | **0.30** | | KA30 | 0.69 |
| K006 | 0.10 | | KI04 | 0.99 |
| | | | KI14 | 0.59 |
| | | | KI16 | 0.93 |
| | | | KI17 | 0.69 |
| | | | KI18 | 1.00 |
| | | | KI21 | 0.80 |

**No faulty unit is missed entirely** -- the weakest, KA15, still flags 23%
of its windows, and a deployed system integrating over a recording would
catch it. False alarms remain per-UNIT rather than random (K005 accounts for
most of them), which is the same pattern kernel v5 saw with K002 and is
what makes healthy-only calibration work as a deployment step.

## Honest limits

- **Binary detection only.** Fault TYPING is not attempted here, and every
  earlier experiment says it does not transfer.
- **Sub-sampled**: 8 windows per CWRU file, 4 per Paderborn run (1360 target
  windows). Scaling up is the obvious next step and costs nothing but time.
- **One target condition** (N09, 900 rpm). N15 is untested.
- **A linear model on 13 features.** No deep model, no adaptation, no DANN.
  That it beats every neural configuration tried is itself the finding.
- Artificial-damage Paderborn codes are excluded; only real damage is tested.
