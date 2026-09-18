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

## Scaling up, a second speed, and fault typing

**Scale is not doing the work.** Re-running with 2.5x the windows (956
source / 2380 target) leaves the headline unchanged: balanced accuracy
0.821 +- 0.009 against 0.823 +- 0.007 before. The result is not a
small-sample artifact.

**The second operating condition is markedly worse, and this is unexplained.**
N15_M07_F10 is the same rig, same bearings, same load, at 1500 rpm instead
of 900:

| target condition | recall @10% FA | balanced acc |
|---|---|---|
| N09 (900 rpm) | 0.755 +- 0.010 | **0.825 +- 0.007** |
| N15 (1500 rpm) | 0.469 +- 0.004 | **0.685 +- 0.005** |

Still well above chance, but a large drop -- and it runs against the
obvious prediction. CWRU is recorded at 1730-1797 rpm, so N15 is the
*closer* speed to the training data and should have been the easier target.
It is not. The cause is not established; the fixed 1.5-5.5 kHz resonance
band interacting differently with the higher shaft rate is the first thing
to check, and the band was deliberately never tuned. **Do not quote the N09
number as if it were the method's general performance.**

**Fault typing transfers for the first time, weakly.** Every earlier
experiment found typing did not transfer at all (3-class accuracy ~0.37,
i.e. chance). With geometry-keyed features:

- 3-class accuracy **0.593**, against a 0.353 majority-class baseline
- among windows that really are faulty, the correct race is chosen **60.8%**
  of the time (chance 50%)

That is real but not usable, and it is lopsided: inner-race faults are
mostly caught (590/840 correct), outer-race ones are mostly misread as
inner (347/700 correct, 299 called inner). Since `bpfo_*` and `bpfi_*` are
literally the outer/inner evidence, the asymmetry suggests the outer-race
peak is being masked -- plausibly because an outer-race defect sits in a
fixed load zone and its envelope peak does not carry the shaft-rate
sidebands the classifier leans on. Worth pursuing, not worth claiming.

## Honest limits

- **Detection is the usable result.** Typing clears chance for the first
  time (60.8% race accuracy) but is not reliable enough to act on.
- **Sub-sampled**: 8 windows per CWRU file, 4 per Paderborn run (1360 target
  windows). Scaling up is the obvious next step and costs nothing but time.
- **Speed-dependent**: strong at 900 rpm, much weaker at 1500 rpm, for
  reasons not yet understood (see above).
- **A linear model on 13 features.** No deep model, no adaptation, no DANN.
  That it beats every neural configuration tried is itself the finding.
- Artificial-damage Paderborn codes are excluded; only real damage is tested.
