# GPU Server Plan — strict cross-dataset feasibility

Written 2026-09-17, rewritten the same day once the objective was stated
precisely. Companion to `DOMAIN_ADAPTATION_PLAN.md` (the record of kernels
v1-v8) and `notes/10_domain_adaptation_notes.md` (the concepts).

**The server bills by the minute.** Estimates are in GPU-minutes, and
anything that can run free elsewhere does.

---

## 1. The actual question

> We have labelled data — healthy and every fault type — on **many**
> bearings. We have nothing from the **target** bearing except perhaps some
> healthy recordings. It sits in a different environment, on a different
> rig, with different sensors and a different DAQ. Can we diagnose its
> faults?

So the unseen thing is **the machine**, not the failure mode. Fault types
are known and labelled; the model must carry that knowledge onto hardware
it has never encountered. That makes this a *classification* problem
(we want "inner race", not merely "something is wrong"), and it makes the
test a **cross-dataset** one.

### Two arms

| | arm A | arm B (mirror) |
|---|---|---|
| **train** (labelled, all fault types) | CWRU only | Paderborn only |
| **adapt** (healthy, unlabelled) | PU healthy | CWRU healthy |
| **test** (never seen) | PU real damage | CWRU faults |

Running both distinguishes "this *direction* is hard" from "cross-dataset
transfer is hard". They are not symmetric: PU is the larger, more varied
source (18 bearing units × 4 conditions), CWRU the smaller (4 rpms).

### What this invalidates

**Every result through kernel v8 answers an easier question.** Those runs
trained on CWRU **+ PU artificial faults** and tested on PU real faults —
same rig, same sensors, same DAQ as the target, differing only in bearing
unit. The headline 95.4% precision / 76.1% recall was measured under that
leakage with respect to the strict question. It is not wrong, it simply
answers "unseen *unit*" rather than "unseen *machine*". Expect the strict
numbers to be lower, possibly much lower.

`pu.include_in_source: false` now enforces the separation, with a test
asserting it mechanically.

---

## 1b. MEASURED (2026-09-17): the variance swamps the effects

Phase 0 was meant to be a formality. It failed, and the investigation
produced the most consequential result in the project so far.

Six runs of the **identical** config on identical data, differing only in
random seed:

| seed | recall | precision | false alarm |
|---|---|---|---|
| 42 (Phase 0) | 0.424 | 0.921 | 16.6% |
| 1 | 0.902 | 0.922 | 34.9% |
| 2 | 0.859 | 0.960 | 16.6% |
| 3 | 0.710 | 0.947 | 18.2% |
| 4 | 0.974 | 0.964 | 16.7% |
| 5 | 0.711 | 0.956 | 14.8% |

**Target recall spans 0.42 to 0.97. Six-run mean 0.763 -- almost exactly
the 0.761 kernel v8 reported as a result.** That number was a typical draw
from a very wide distribution, presented as a measurement.

Source validation accuracy was 0.9997 in every run. It does not move, and
it does not predict target performance at all.

Ruled out as causes, by measurement rather than argument:
- **float32 storage**: worst-case 0.006 sigma on any pixel, 0.0000% of
  pixels shifted more than 0.01 sigma.
- **resampling**: bit-identical between the raw and condensed paths
  (`np.array_equal` true, max signal difference 7.4e-09).
- **a broken port**: source-side behaviour reproduces exactly.

### What this invalidates

The run-to-run spread (recall std ~0.12, 3-class accuracy std ~0.058) is
the same size as, or larger than, the effects the project concluded from:

| claim | effect size | vs noise | verdict |
|---|---|---|---|
| DANN is harmful | 0.761 -> 0.085 recall | >3 sigma | probably real |
| diversity helps (7 -> 18 codes) | 0.403 -> 0.761 | ~2-3 sigma | plausible, not established |
| CWRU helps (v4) | 0.282 -> 0.171 accuracy | ~2 sigma | not established |
| v3's collapse was the learning rate | single run | within noise | unsupported |

**Every single-seed comparison in `DOMAIN_ADAPTATION_PLAN.md` and
`README.md` must be read as indicative at best.** Nothing below 2 sigma
survives, and nothing was replicated.

### And a constraint that cannot be engineered away

We cannot select or tune for recall. Measuring it needs target fault
labels, which the problem definition denies us. A criterion built on target
*healthy* data alone actively rewards the failure mode observed here -- a
model that calls everything healthy scores perfectly on false alarms. So
"just pick the good run" is not available at deployment.

Ensembling is the one lever that needs no labels. Majority vote over the
five seeds: recall 0.805, precision 0.958, false alarms 16.4% -- near the
*mean* rather than the best, with better precision and fewer false alarms.
Its value is predictability, not peak performance. (Measured once; the
variance-reduction claim itself is not yet measured.)

The five models agree on only **33% of windows**, and disagree on **43%**
of the binary fault/no-fault question. Identical data, identical config.

### Consequence for everything downstream

**Every experiment from here reports mean +- std over >= 3 seeds.** A
single run is not a result. Arm A in particular would have produced a
number indistinguishable from noise had the gate not caught this.

## 2. What is established, and what is not

**Established** (multiple runs, large effects):
- Fault *detection* transfers better than fault *typing*; no configuration
  ever separated inner from outer race on the target.
- Source **diversity** dominates every other lever tried: 7 → 18 bearing
  codes took detection 0.403 → 0.761 while false alarms *fell* 31.1% →
  16.7%. Removing CWRU (v4) cost ~11 accuracy points.
- Errors are **per-unit, not random**: every false alarm came from one of
  six healthy bearings (K002, 100% of its windows), the rest near-clean.
  Under the strict framing this is the central risk, not a footnote.

**Not established:**
- "DANN is harmful" — three hyperparameter points, one seed each.
- Any single number's precision; everything is n=1.
- Whether v3's collapse was its learning rate or its seed.

---

## 3. Infrastructure findings

- **The server cannot reach Kaggle usefully**: 10.8 KB/s to kaggle.com
  against 9.0 MB/s to Cloudflare and full speed from PyPI — roughly 830×
  slower, Kaggle-specific. The 437 MB bundle would have taken ~12 hours.
  Data therefore routes **Kaggle → laptop → server**, and the server cannot
  be relied on for any future Kaggle interaction.
- **Paderborn is condensed to 437 MB** from 9.7 GB: one channel
  (`vibration_1`) of seven, at 12 kHz — the rate the pipeline resamples to
  anyway, so lossless for our use. Covers every code and condition.
- **torch 2.6.0+cu124 on an RTX 3090**, and the full suite runs: **69
  passed**, versus 35 passed / 12 skipped on the laptop. The GRL gradient
  reversal, DANN shapes, healthy-anchor loss and a training step have now
  been verified against real torch rather than reasoned about.

---

## 4. Phases

### Phase 0 — Validate the port (~20 GPU-min)
Suite green on GPU (done), then reproduce kernel v8's *permissive* config
(recall 0.761, false alarms 16.7%) within seed noise.

*Gate:* if v8 does not reproduce, stop. Everything downstream would be
built on a port that silently differs.

### Phase 1 — Arm A, the strict question (~60 GPU-min) — **primary**
Train CWRU only → adapt on PU healthy → diagnose PU real damage.
Run `{time, order} × 3 seeds`, so the speed gap is a measured variable
rather than an uncontrolled confound.

**Order tracking is necessary here, not optional.** CWRU runs at 1730-1797
rpm, PU at 900/1500 — a ~2× stretch of the frequency axis, which a CNN has
no tolerance for. Without it a poor result cannot be attributed.

*Pre-registered readings:*
- **Detection (binary) at ≤20% false alarms** is the headline. Typing is
  secondary; it has never transferred.
- *order beats time* → the speed gap was a real barrier and the
  representation fixes it.
- *both poor* → cross-dataset transfer from CWRU alone does not work, and
  the honest next question is whether it is a diversity problem (arm B has
  far more source data) or a domain-gap problem.

### Phase 2 — Arm B, the mirror (~60 GPU-min)
Train PU → adapt on CWRU healthy → diagnose CWRU faults. Needs a small
change: CWRU currently only ever plays a source role, so its healthy files
must be splittable into adaptation and test.

Arm B is the better-powered direction (more source units and conditions).
If A fails and B succeeds, the limit is source diversity, not the domain
gap — which is actionable, because more source data is obtainable.

### Phase 3 — Seed replication and variance (~40 GPU-min)
5 seeds on whichever arm looks live, plus the permissive v8 config for
comparability. Turns every single-seed claim in the existing write-up into
mean±std, and settles whether v3's collapse was seed-specific.

*Pre-registered:* if baseline recall's std exceeds ~0.05, single-seed
comparisons in `DOMAIN_ADAPTATION_PLAN.md` get re-labelled as indicative.

### Phase 4 — Leave-one-unit-out (~90 GPU-min)
Feasibility needs an error bar across *machines*, not one number. Rotate
each target bearing unit through the held-out role and report the
distribution. The K002 finding says this spread may be large, and "works on
9 of 11 units, fails on these 2, and here is what distinguishes them" is a
far better basis for deployment than a single average.

### Phase 5 — Diversity curve (~90 GPU-min)
Diversity is the one proven lever and its shape is unknown. 4-6 points from
narrow to full source. Saturating vs still-climbing changes what to do next,
and directly informs whether arm A's likely weakness is fixable with data.

### Phase 6 — Settle DANN, demoted (~3-4 GPU-hr)
Random search over `domain_loss_weight` × `grl_gamma` ×
`discriminator_lr_multiplier` × `anchor_loss_weight`, 3 seeds each.

*Pre-registered success:* any configuration beating its baseline's recall
at ≤ its false-alarm rate. None → the negative result becomes defensible
rather than under-powered. One → the current write-up is wrong and
`README.md` gets corrected prominently.

Demoted because adaptation has never helped in eight runs while
representation and diversity both have.

### Phase 7 — Deployment (~30 GPU-min)
Re-verify `infer.py` against the best model. Confirm `--calibrate` detects
a K002-like unit from healthy data alone — the safeguard that matters most
under the strict framing, since it is the only check available before a
real fault ever occurs.

**Total ≈ 7-9 GPU-hours.**

---

## 5. Deliberately not doing

- Re-proving what already replicates (detection-over-typing, diversity).
- Chasing 3-class accuracy; the deployment question is binary.
- Bigger models on a hunch. Diversity beat every optimisation knob tried,
  which points at data rather than capacity. ResNet-18 stays available if
  Phase 5 shows the source is saturated.
- Kaggle in the training loop — the route is unusable from the server.

---

## 6. How we would know we are wrong

- **Phase 0 fails to reproduce v8** → broken port; stop and fix.
- **Phase 3 shows large seed variance** → much of the narrative is noise
  and gets downgraded in writing.
- **Phase 6 finds a winning DANN config** → the headline negative result is
  wrong; correct `README.md` and `DOMAIN_ADAPTATION_PLAN.md` prominently.
- **Both arms fail even with order tracking** → cross-*machine* transfer
  without target fault data does not work at this data scale. That is a
  legitimate finding and the honest deliverable becomes the healthy-baseline
  detector plus per-machine calibration, not a cross-dataset classifier.
  Saying so clearly is more useful than a number that will not survive
  contact with a real pump.
