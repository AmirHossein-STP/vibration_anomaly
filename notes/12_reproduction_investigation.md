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
