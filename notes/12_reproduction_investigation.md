# The arm A reproduction failure (2026-09-17)

A record of an investigation that is still open at the time of writing, kept
because the dead ends are most of its value and because the artifacts that
would have made it easy no longer exist.

## What happened, in order

**First rental.** Arm A -- train on CWRU only, adapt on Paderborn healthy
windows alone, then diagnose Paderborn's real bearing damage -- was run to
n=8 per representation and gave:

| representation | recall | precision | F1 | balanced acc |
|---|---|---|---|---|
| time | 0.812 +- 0.159 | 0.943 +- 0.022 | 0.865 +- 0.098 | 0.795 +- 0.078 |
| order | 0.606 +- 0.218 | 0.948 +- 0.026 | 0.719 +- 0.155 | 0.723 +- 0.078 |

Those numbers were committed to `GPU_SERVER_PLAN.md` and the README. The
per-run artifacts behind them were **never pulled off the server**. The
server's subscription then ended and the machine was deleted.

**Second rental.** A clean-room rebuild on an identical RTX 3090 produced,
for all 8 time seeds:

```
recall 1.000   precision 0.821   false alarm 100%   balanced accuracy 0.500
predicted class counts: [0, 0, 5117]      # every window -> class 2
```

Identical to three decimal places across all 8 seeds. Eight different weight
initialisations cannot agree exactly unless the features are constant, so
this is a mechanism rather than bad luck. Order-rep runs varied by seed but
were also degenerate (balanced accuracy 0.493).

## What was ruled out, and how

Each of these was a real hypothesis, tested rather than argued away. All
four failed, which is why the investigation is still open.

**1. My repack of the Paderborn store.** To fit a ~20x slower uplink, the
prepared store was trimmed to the N09 target condition (437 MB -> 63 MB).
Ruled out twice over: every retained array is `np.array_equal` to the
original, and the manifest is structurally identical (same columns,
`fs_hz` 12000, detected as a prepared store). Crucially the *split* is
identical too -- runs 1-20 per healthy code, cutoff 12, the same 60/40
adapt/test division as the full store produces.

**2. The torch version.** The first server ran torch 2.6.0+cu124; the second
shipped 2.8.0+cu128. Installing 2.6.0+cu124 on the second box and rerunning
3 seeds reproduced the collapse *to three decimals*. Torch is not involved.

**3. Normalization / outlier saturation.** Source-normalized target windows
reach p99 = 7.78 where the source itself reaches 2.15 -- a 3.6x excursion
beyond anything seen in training, which looked like a saturation mechanism.
Re-running the already-trained model over the same windows with clips of
+-5, +-3, +-2.5 and +-2.0 (the last *tighter* than the source's own p99)
gave byte-identical collapse at every setting. Not an outlier effect.

**4. Cross-config cache contamination on the second box.** Ruled out: the
processed_dir paths are per-config (`data/da_processed_armA_time` vs
`..._armA_order`), and this box was new, so every cache was built fresh.

## What is left

**A stale window cache on the FIRST server** is the leading explanation.
`build_domain_adaptation_datasets` returned a cached `processed_dir`
whenever the files merely EXISTED:

```python
if not force and all(p.exists() for p in paths.values()):
    return {role: (np.load(...), pd.read_csv(...)) for role in ROLES}
```

No config hash, no validation. The first server accumulated caches across
many config edits over several days; the second built everything fresh from
the config as written. That ordering makes the clean-room collapse the
honest behaviour of this config, and the committed numbers the artifact --
rather than the reverse.

**This cannot be proven.** The first server is gone, so its caches cannot be
inspected. The claim is "leading explanation", not "established". It is now
guarded so it cannot recur: `_cache_fingerprint()` hashes common/stft/cwru/pu
and target_dataset (deliberately excluding the seed, which does not change
the window set), is written last so an interrupted build is rebuilt rather
than trusted, and has a regression test asserting that `include_in_source`
invalidates the cache while the seed does not.

## What the collapse looks like mechanically

Source training reaches `val_acc 1.000` in a SINGLE epoch. The source's
majority class is 2 (4674 windows vs 2794 healthy and 2664 inner-race), and
every out-of-distribution target window lands on exactly that class. The
domain separability probe reports 1.000 with proxy A-distance 2.00 -- source
and target features are perfectly separable.

Read together: the network learns a CWRU-specific shortcut that is trivially
sufficient on the source and absent in Paderborn, then falls back to the
majority class on everything it cannot place. A source a model can solve
perfectly in one epoch is a warning, not a success.

## The open question

Whether the strict cross-dataset premise genuinely fails, or whether
something about the rebuild is broken, is decided by the **permissive
control**: the same pipeline with Paderborn's artificial faults IN training,
which reported recall ~0.76 on Kaggle.

- If permissive reproduces ~0.76 here, the pipeline is sound and the strict
  premise fails -- meaning every earlier "success" depended on having
  target-rig data in training, and the project's headline claim does not
  hold in the form it was stated.
- If permissive also collapses, the rebuild is broken and nothing about the
  premise has been shown either way.

That control was queued behind the full 437 MB transfer and had not
completed when this note was written.

## RESOLVED (18:18): the strict premise fails; the old number was the artifact

The permissive control ran on the same box, same code, same torch
(2.6.0+cu124), differing from strict arm A in exactly one premise --
Paderborn's artificial faults are in training:

| | n | recall | false alarm | balanced acc | collapsed runs |
|---|---|---|---|---|---|
| strict (2 torch versions) | 11 | 1.000 +- 0.000 | 100% | 0.500 +- 0.000 | **11/11** |
| **permissive** | 3 | 0.571 +- 0.313 | 17.5% +- 7.0% | 0.698 +- 0.135 | **0/3** |
| Kaggle v8 reference | 1 | 0.761 | 16.7% | -- | -- |

Read the **collapsed** column, not the recall. Permissive recall is wildly
seed-dependent (its best seed hit 0.772 at 15.7% false alarms, squarely on
the Kaggle reference; its worst dragged the mean to 0.571) -- which is the
same run-to-run variance this project measured earlier, spanning 0.42-0.97.
The categorical fact is what matters and it has no overlap: **strict
collapses in 11 runs out of 11, permissive in 0 out of 3.** One is a
degenerate model, the other is a noisy working one.

**The pipeline is sound.** The collapse is specific to holding Paderborn
out of training.

**And this identifies the artifact precisely.** The committed arm A figure
was recall 0.812 +- 0.159 against a permissive reference of 0.763 +- 0.19 --
statistically indistinguishable. That was recorded at the time as the
finding "holding PU entirely out of training cost nothing measurable". The
simpler reading, now that strict-from-a-clean-cache collapses completely,
is that **PU was never held out**: the arm A `processed_dir` on the first
server still held windows built while `include_in_source: true`, so
"strict" arm A was silently measuring the permissive setup. That is exactly
what an unvalidated cache produces, and it explains why the two agreed so
closely.

A result that matches the condition you thought you had excluded is
evidence that you did not exclude it.

### Can it be rescued by fixing the normalization? No.

The obvious suspect was the normalizer: it fits per-frequency-bin mean/std
on the SOURCE and applies them to the target, baking CWRU's spectral
profile into the representation (target p99 = 7.78 against the source's
2.15). Two alternatives were tested, each retraining from the same cached
windows, since the cache stores un-normalized spectrograms:

| normalization | train p99 | target p99 | recall | FA | balanced | predicted classes |
|---|---|---|---|---|---|---|
| source-fit per-bin (current) | 2.15 | **7.78** | 1.000 | 100% | 0.500 | 0 / 0 / 5117 |
| per-window | 1.89 | **2.24** | 0.998 | 100% | 0.499 | 8 / 3308 / 1801 |
| per-window per-bin | 1.74 | **1.79** | 0.969 | 96% | 0.505 | 166 / 1549 / 3402 |

Both alternatives **fix the scale mismatch completely** -- target p99 drops
from 7.78 to 2.24 and then 1.79, matching the source. And the model still
cannot find a healthy Paderborn window: balanced accuracy stays at chance
in all three.

This is more informative than another failure. Per-window normalization
removes the rig-specific reference entirely -- each window is scored against
itself, so a constant offset or gain difference between rigs cannot
survive. It still does not transfer. The per-window run is not even a
single-class collapse any more (it uses classes 1 and 2, and 3-class
accuracy rises 0.372 -> 0.481), yet it calls essentially everything faulty.

So the failure is not a scaling artifact that preprocessing can fix. The
CWRU-trained representation does not carry a healthy-vs-faulty boundary
that means anything on Paderborn.

### Arm B: the reverse direction fails too

Arm B had never run to completion (the first server died at 2/6). It makes
Paderborn the source -- 18 bearing units across 4 operating conditions,
against CWRU's 4 rpms -- and CWRU the unseen target. It is the
better-powered direction by a wide margin.

| | n | recall | false alarm | balanced acc |
|---|---|---|---|---|
| arm B (Paderborn -> CWRU) | 4 | 0.998 +- 0.005 | **100%** | **0.499 +- 0.002** |

It fails the same way: essentially every CWRU window is called faulty, and
not one healthy window out of ~2730 is identified. (It is not a
single-class collapse -- predictions spread over classes 1 and 2 -- but the
healthy/faulty boundary is gone entirely, which is what matters.)

**This closes the diversity hypothesis.** The open question from the first
server was whether arm A's ceiling was source diversity or the domain gap
itself. It is neither: transfer fails in BOTH directions, and it fails just
as completely when the source is four times richer. The method does not do
cross-dataset transfer, and no amount of source breadth within these two
datasets changes that.

### What this means for the project's claim

The headline -- zero-shot fault detection transfers to an unseen machine,
sensors, DAQ and lab -- is **not supported**. With this architecture, on
these two datasets, detection requires fault data from the target rig in
training. What transfers across *bearing units within one rig* (the
permissive setup, 0.772 recall at 15.7% false alarms) does not transfer
across *datasets*.

That is a negative result, but it is a clean one, and it is the answer to
the question the project actually asked. It is now supported from three
independent directions: the strict arm collapses (11/11 runs), the reverse
direction fails identically despite a 4x richer source (4/4 runs), and
fixing the normalization -- which demonstrably removes the scale mismatch --
does not help. The one configuration that works is the one with target-rig
fault data in training.

## Consequences already established

Independent of how the open question resolves:

- **The committed arm A table is unconfirmed** and is marked as such in
  `GPU_SERVER_PLAN.md`.
- **The "91.8% detection when aggregated per recording" figure cannot be
  recomputed** from the project's own artifacts: PU rows from the condensed
  store carry an empty `source_file`, so per-recording identity is absent
  from the window metadata. `scripts/analysis/per_unit.py` now says so
  instead of silently printing nothing.
- **Never again commit a number whose artifacts are not on the laptop.**
  This entire investigation would have been a ten-minute diff.
