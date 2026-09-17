# GPU Server Plan

Written 2026-09-17, when a rented 24 GB-class GPU box replaced CPU-only
Kaggle kernels. Companion to `DOMAIN_ADAPTATION_PLAN.md` (the chronological
record of kernels v1-v8) and `notes/10_domain_adaptation_notes.md` (the
concepts). This file is the forward plan.

**The server bills by the minute.** Every estimate below is in GPU-minutes,
and anything that can run free elsewhere (Kaggle, this laptop) does.

---

## 1. What compute actually changes

Not "bigger model". The important change is **statistical power for claims
already made**.

The project's strongest conclusion -- *domain-adversarial training is
harmful here* -- rests on **3 distinct hyperparameter points, one seed
each**:

| kernel | domain_w | grl_γ | disc_lr | seed |
|---|---|---|---|---|
| v1 | 1.0 | 10 | 1× | 42 |
| v2 | 3.0 | 25 | 1× | 42 |
| v3 | 3.0 | 25 | 5× | 42 |
| v4, v5/v8 | 3.0 | 25 | 1× | 42 (data varied, not hyperparameters) |

Three points in a four-dimensional space. And the diagnosed failure modes
-- "the classifier wins the race", "the discriminator loses the race" --
are precisely what a sweep resolves. v3 collapsed at epoch 4 and that was
attributed to its 5× learning rate, but **it was never checked whether the
collapse was seed-specific**; with one seed it could not have been.

The positive results are equally thin: baseline faulty-recall 0.761 is
n=1, and v4's "CWRU helps" is n=1 versus n=1.

Every compute-driven compromise is now revisitable:

| compromise | why it was made | now |
|---|---|---|
| small CNN, not ResNet-18 | CPU cost; documented deviation from the original plan | testable as specified |
| 12 kHz common rate | partly "keep it CPU-feasible" | revisit (though CWRU's DE12 caps it) |
| 3 of 4 source conditions, `source_train_fraction` 0.7 | budget | widen |
| single seeds, no sweeps | 142 min/run | the core of Phases 1 and 3 |
| torch untestable locally | Intel Mac caps torch at 2.2.2 | **12 skipped tests finally run** |

---

## 2. Established vs. under-powered

**Carry forward (multiple runs agree, large effects):**
- Zero-shot fault *detection* transfers; fault *typing* does not. Every
  version failed to separate inner from outer race on the target.
- Source *diversity* dominates every other lever: 7 → 18 bearing codes took
  detection 0.403 → 0.761 while false alarms *fell* 31.1% → 16.7%.
  Removing CWRU (v4) cost ~11 accuracy points.
- False alarms are per-UNIT, not random: all of them came from one of six
  healthy bearings (K002, 100% of its windows).

**Treat as unproven until Phase 1/3:**
- "DANN is harmful." Under-powered, as above.
- Any single headline number's precision (all n=1).
- v3's collapse being caused by the learning rate rather than the seed.

---

## 3. Data logistics — DONE

| | route | size |
|---|---|---|
| CWRU | `scp` from the laptop, already `.npz` | 179 MB |
| Paderborn | server pulls a condensed bundle from Kaggle | **437 MB** |

Raw Paderborn is 9.7 GB of which this project reads one channel
(`vibration_1`) out of seven, at a rate it immediately downsamples.
`data/prepare_pu.py` runs on Kaggle (free, data already mounted) and keeps
only that channel at 12 kHz float32 — **22× smaller, and lossless for our
use**, since the pipeline resamples to 12 kHz anyway and CWRU's DE12 files
cap the cross-dataset common rate there regardless. Covers every bearing
code and condition (2559 of 2560 runs; the miss is the known-corrupt
`KA08/N15_M01_F10_KA08_2.mat`), so no config change triggers another
download at server rates.

int16 would halve it again to ~218 MB but perturbs 2.29% of the model's
input pixels for ~15 s of transfer — measured and rejected. Regenerating
at any other rate/precision is a free 6-minute Kaggle run.

Reading it is automatic: `pu_loader.is_prepared_store` detects the
condensed form by its manifest, so the same `--pu-root` serves either.

---

## 4. Phases

Estimates assume a 24 GB GPU and that the current model (2.2 M params,
128×128 inputs, ~69 k windows) trains ~20-40× faster than the Kaggle CPU
baseline of 8 min, with data loading becoming the floor.

### Phase 0 — Provision and validate (~20 GPU-min)
`scripts/23_server_setup.sh`: CUDA torch, deps, pull the bundle, `scp` CWRU,
run the suite. **The 12 torch-gated tests execute for the first time** —
GRL gradient reversal, DANN forward shapes, the healthy-anchor loss, a
training step. Every one of those has only ever been checked by inference.

*Exit criterion:* full suite green on GPU, and one baseline run reproduces
kernel v8's numbers within seed noise (recall 0.761, false alarms 16.7%).
If it does not, stop — something about the port is wrong and every later
number would be built on it.

### Phase 1 — Seed replication (~40 GPU-min)
5 seeds × {baseline, DANN} at the v8 configuration. Produces mean±std on
every headline number and answers whether v3's collapse was the learning
rate or the seed.

*Pre-registered:* if baseline recall's std exceeds ~0.05, every
single-seed comparison in `DOMAIN_ADAPTATION_PLAN.md` — including v4's
"CWRU helps" — gets re-labelled as indicative rather than established.

### Phase 2 — Order-domain transfer (~60 GPU-min) — *highest upside*
The untried physics fix. Bearing fault frequencies scale with shaft speed,
so 1500 → 900 rpm multiplies the frequency axis by 0.6 — a *stretch*, which
a CNN has no tolerance for. `common.representation: order` resamples
against shaft angle instead, so a defect sits at the same order at any
speed. Already implemented, wired behind a config switch, and validated on
synthetic signals: spectrogram correlation between the two speeds goes
**0.799 → 0.978**.

Run: {time, order} × {baseline, DANN} × 3 seeds.

*Pre-registered success:* order-domain baseline beats time-domain baseline
on target faulty recall at equal-or-lower false-alarm rate. If it does,
this becomes the project's main result and Phase 3 drops in priority — a
better *representation* beating better *adversarial training* is the more
useful finding either way.

### Phase 3 — Settle the DANN question (~4-6 GPU-hr)
Random search over `domain_loss_weight` × `grl_gamma` ×
`discriminator_lr_multiplier` × `anchor_loss_weight`, 3 seeds per config,
~16-24 configs.

*Pre-registered success:* **any** configuration beating the baseline's
faulty recall at ≤ its false-alarm rate. If none does, "DANN does not work
on this problem" becomes a defensible negative result backed by a real
search rather than three guesses. If one does, the current write-up is
wrong and gets corrected.

Track `domain_acc` (balanced, chance = 1/3) and the post-hoc separability
probe per run — together they distinguish "aligned" from "discriminator was
fooled", the ambiguity that made v1 unreadable.

### Phase 4 — Undo the compute compromises (~2-3 GPU-hr)
- **ResNet-18**, as the original plan specified, vs the small CNN.
- **All 4 source conditions** and a higher `source_train_fraction`.
- **Source-diversity curve**: 4-6 points, not the current two. Diversity is
  the proven lever and its shape is unknown — saturating or still climbing
  changes what to do next.

### Phase 5 — Deployment hardening (~30 GPU-min)
Re-verify `infer.py` against the best model, and settle the per-unit
calibration story (K002 was misread in 100% of its windows while five other
healthy units were clean). Confirm `--calibrate` detects that case from
healthy data alone, which is what a real machine can offer.

**Total: roughly 8-12 GPU-hours** for Phases 0-5.

---

## 5. Deliberately not doing

- **Re-running things that already agree across runs.** Detection-transfers
  and diversity-dominates are established; re-proving them is paid time.
- **Chasing accuracy on 3-class typing.** It failed in every configuration;
  the deployment question is binary and already answerable.
- **Bigger models for their own sake.** Diversity beat every optimisation
  knob tried, which is evidence that data, not capacity, is the constraint.
  ResNet-18 is in Phase 4 to test a documented deviation, not on a hunch.
- **Keeping Kaggle in the loop for training.** It stays useful for free
  preprocessing; the 142-minute cycle does not.

---

## 6. How we would know we are wrong

- **Phase 0 fails to reproduce v8.** Then the port is broken and everything
  downstream is suspect — stop and fix.
- **Phase 1 shows large seed variance.** Then most of the existing
  narrative is noise, and the conclusions get downgraded in writing.
- **Phase 3 finds a winning DANN config.** Then the headline negative
  result is wrong and `DOMAIN_ADAPTATION_PLAN.md` and `README.md` both get
  corrected, prominently.
- **Phase 2 shows no order-domain gain.** Then the speed shift was not the
  bottleneck either, and the honest conclusion is that this cross-rig,
  cross-speed transfer is harder than any single fix addresses — at which
  point the deployable result is the existing detector plus per-machine
  calibration, and further compute is better spent elsewhere.
