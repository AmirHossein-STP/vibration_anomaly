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

## 1c. ARM A RESULT (2026-09-17): detection transfers across datasets

Train CWRU only. Adapt on Paderborn HEALTHY only. Diagnose Paderborn's real
damage. No PU fault -- artificial or real -- ever seen in training.
3 seeds per representation.

> **REPRODUCTION FAILED (2026-09-17, second server).** After the first
> server was deleted, a clean-room rebuild of arm A on an identical RTX
> 3090 did NOT reproduce the table below. All 8 time-representation seeds
> collapsed to a single class (every one of 5117 target windows predicted
> outer-race), giving recall 1.000, false-alarm 100%, balanced accuracy
> 0.500 -- and **identical to 3 decimal places across all 8 seeds**, where
> the original runs spanned 0.42-0.97. Order-rep runs varied by seed but
> were also far worse (balanced accuracy 0.493).
>
> What was ruled out: source training is perfect (val_acc 1.000 on CUDA);
> the repacked PU store is byte-identical to the original for every array
> used; the window cache is per-config so no cross-config contamination;
> all cached spectrograms are 100% finite. What was found: target windows
> have std 0.857 against the source's 2.145, so normalizing by source
> statistics squeezes the target into a narrow blob -- constant features,
> seed-independent argmax.
>
> What differs from the original runs: torch 2.8.0+cu128 (was 2.6.0+cu124),
> and a PU store trimmed to the N09 target condition (arm A never trains on
> PU, and the arrays are identical, so this *should* be inert -- but it is
> untested against the full store).
>
> **The numbers below are therefore unconfirmed.** The original artifacts
> were never pulled off the first server, so they cannot be re-examined.
> Until a rebuild reproduces them, treat this table as provisional.

**Updated to n=8 per representation** (the n=3 figures first recorded here
left the error bars too wide to act on):

| representation | recall | precision | false alarm | F1 | balanced acc |
|---|---|---|---|---|---|
| **time** | **0.812 +- 0.159** | **0.943 +- 0.022** | 22.3% +- 9.4% | **0.865 +- 0.098** | **0.795 +- 0.078** |
| order | 0.606 +- 0.218 | 0.948 +- 0.026 | 15.9% +- 10.1% | 0.719 +- 0.155 | 0.723 +- 0.078 |
| *permissive reference (PU faults IN training, 6 seeds)* | 0.763 +- 0.19 | | | | |

**1. Detection transfers.** ~94% precision on an unseen rig, sensor set and
DAQ. When the model says "faulty" it is right roughly 19 times in 20. For
the project's actual question -- diagnosing a machine we have no fault data
for -- this is the affirmative result.

**2. Holding Paderborn out of training cost essentially nothing.** Strict
0.710 +- 0.197 against permissive 0.763 +- 0.19: indistinguishable. The PU
artificial faults, recorded on the very rig under test, bought almost
nothing. This is the strongest practical finding here, because it means a
deployment does **not** need fault data from the target machine's rig.

**3. Order tracking is WORSE, and at n=8 that is now measurable.** This
plan previously called it "necessary rather than optional" for arm A. It is
the opposite: recall 0.812 -> 0.606 (2.2 SE), F1 0.865 -> 0.719 (2.3 SE),
balanced accuracy 0.795 -> 0.723 (1.9 SE), with precision unchanged
(0.4 SE).

The obvious confound was checked and rejected: order shows *both* lower
recall and lower false alarms, which is the signature of a more
conservative operating point rather than weaker discrimination. But F1 and
balanced accuracy weigh both directions, and order loses on both -- so it
genuinely discriminates worse, not merely differently.

**Why the synthetic validation misled.** The transform was accepted on a
synthetic test where spectrogram correlation between 1500 and 900 rpm rose
0.799 -> 0.978. That test used a pure order-locked tone, which is exactly
what angular resampling is built to align -- it could not have failed.

A physical reason was missed when that claim was made: bearing fault energy
rides on **structural resonances, which are fixed in Hz, not in orders**.
Angular resampling aligns the fault harmonics and simultaneously smears
those resonances, so it trades one alignment for another. The classical
remedy is envelope analysis -- demodulate around the resonance band, then
order-track the envelope -- which aligns both. That is the version worth
testing before order tracking is written off.

**4. Typing still fails** (0.33-0.37 accuracy), as in every configuration
ever run.

### Caveats
n=3, and recall's spread remains wide. The recall figures should be read as
"somewhere in 0.4-0.9" rather than as point estimates; precision is far more
stable (+-0.029) and is the number to lean on.

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
