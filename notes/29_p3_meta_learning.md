# 29. Phase 3 Track A: a meta-learned detector that learns to compare with a healthy reference

**Pre-registration.** Written 2026-10-22, before any model in this note was trained or any
learned score was computed. The parity reference (§5) reproduces numbers that are already
committed, so it reveals nothing new.

## 1. The idea
Method v1.1 detects with one fixed, hand-made comparison between a window and the unit's own
healthy reference:

    score(x | support) = max_j (log x_j − μ_j) / σ_j,   j = 135 geometry-free order bins,
    μ, σ = mean and sd over the support (the unit's healthy windows)

**Meta-learning here = learning that comparison function** `f(x | support)` across many source
machines. The aim is that on a machine it has never seen, with only that machine's healthy
reference, it separates faults better than the max-z.
- Every training "episode" is one calibration fold of one source dataset: support = its healthy
  calibration windows, query = held-out healthy windows (negatives) and fault windows (positives).
- Fault labels are used **only from source datasets**. A held-out dataset contributes nothing to
  training, so this respects constraint #1 (MASTER_PLAN §0).
- The usual few-shot meta-learning for fault diagnosis (MAML, ProtoNets) needs labelled target
  faults in the support set and is not applicable (MASTER_PLAN §2). The nearest published idea
  is image-domain class-generalisable anomaly detection (ResAD, InCTRL): score against a few
  normal references of a class never seen in training.

## 2. Data: the episode bank (A0)
Features: the frozen geometry-free cache, `v2_cache.py --set geofree`, unchanged fingerprints.
IMS is new: `scripts/analysis/p3_ims_cache.py`, the same `rows_of`.
Nine datasets, each held out as a whole in turn (leave one dataset out):

| dataset | folds (support / negatives / positives) | source of the fold rule |
|---|---|---|
| cwru | `wp5_score.folds` (other conditions' healthy / this condition's healthy / its faults) | notes/17 |
| pu (4 conditions, held out together) | `wp5_score.folds` per condition, real damage | notes/16 |
| ottawa_2018, mfpt | `wp5_score.folds` (leave one healthy recording out) | notes/17 |
| ottawa_2023 | `wp5_score.folds`, **own** baseline (primary); machine baseline reported | notes/17 |
| hust | per type × load: other loads' healthy / this load's healthy / its faults | notes/24 |
| xjtu_sy | per bearing, horizontal: minutes 1–20 / 21–60 / last 10% of records | notes/18 |
| paderborn_ldm | per bearing, channel A, set speed ≥ 1000 rpm: records 1–150 / 151–450 / last 10% of records | notes/25 |
| ims | per unit (test 1: x): records 1–200 / 201 to the 50% point of the test / last 5% of records for the 4 failed units; the 8 others give negatives only | notes/27 |

- Positives: race-involving faults where labels exist (labels 1, 2, 5, 6, 7). On run-to-failure
  sets, all end-of-life windows (the failure mode is not used). Ball/cage (3, 4) are reported
  separately and never trained on as positives.
- Episodes are sub-sampled to at most 3000 windows per part with a fixed seed (0). The
  reference and every model see **identical windows**.

## 3. Models
Both read the support only through statistics computed from it (μ, σ, and for calibration the
cross-fitted z of each support window, exactly as `method.fit_baseline` does).
- **A1, rank pool (bin-agnostic).** Input: the 32 largest z of the window (sorted), plus the mean
  and sd of all 135 z. MLP 34 → 64 → 64 → 1 (about 6.5k parameters).
  - Rationale: fault orders differ between bearings, so *which* bin rose does not transfer. *How*
    the rise is shaped might.
  - Monotone in the top z by construction: its first coordinate is added as a skip, `s = z_(1) + MLP(...)`.
- **A2, bin transformer (set-conditioned).** One token per bin: [z_j, μ_j, log σ_j, 8-dim
  sinusoidal code of the bin's order]. 2 encoder layers, width 32, 4 heads, attention pooling → 1
  (about 30k parameters). Because the order value is known, harmonic families (f, 2f, 3f) are
  learnable without any geometry.
- **Training (fixed now, not tuned):**
  - each step samples a training dataset uniformly, then one of its folds, then 64 negatives
    (held-out healthy + cross-fitted support windows) and 64 positives;
  - pairwise logistic ranking loss;
  - Adam, lr 1e-3, weight decay 1e-4, 3000 steps, batch of 8 episodes;
  - seeds 0–4.
  - No early stopping, no validation on the held-out dataset.
- **Threshold:** conformal quantile at α = 0.10 of the model's scores on the cross-fitted support
  windows, the same rule as v1.1.

## 4. Metrics
Per held-out dataset:
- mean over its folds of window AUC (negatives vs positives);
- FA mean and FA worst fold at the conformal threshold;
- ball/cage AUC where labels exist;
- the 5-seed mean ± sd.

## 5. Parity (A0 gate, before any training)
Scoring the bank with v1.1 (max z) must reproduce the committed geometry-free AUCs exactly:
- WP13 rows: cwru 0.951, pu ×4, ottawa_2018 0.998, ottawa_2023 own 0.971 and machine 0.825, mfpt 0.996;
- WP14 per type.

LDM, XJTU-SY and IMS have no committed window-AUC to match. Their reference numbers are produced
by the same code and recorded here before any training.

## 6. Gate and readings
**The gate changes from the plan's draft** ("+0.03 on ≥ 6/9 sets"). v1.1 is already at 0.95–1.00
on cwru, ottawa_2018, mfpt and hust 6204, so no model could gain 0.03 on six sets. A model
**passes** if, on 5-seed means:
1. the mean over the 9 held-out datasets of (AUC_model − AUC_v1.1) ≥ **+0.02**;
2. no held-out dataset loses more than **0.02**;
3. the mean worst-fold FA is no more than **0.02** above v1.1's.

Readings:

| outcome | meaning |
|---|---|
| A1 or A2 passes | Learning the comparison transfers across machines. The simpler passing model becomes the v2.0 detection candidate, and A3 (synthetic-fault episodes) is run on it |
| both fail, and the learned models win on the hard rows (PU N15, ottawa_2023 machine, hust 6205, LDM, IMS) but lose elsewhere | A specialist, not a replacement. Not shipped; reported |
| both fail | With these features, max-z against the unit's own baseline is already as good a comparator as 8 source datasets can teach. Track A closes. **This is a result, not a failure of the track** |
| A2 passes and A1 does not | The order-aware structure (harmonics) is what transfers. That is checked by the post-hoc attention map, which is descriptive only |

## 7. Results (2026-10-22): **both fail; max-z is already as good a comparator as 8 sources can teach**
Run: `remote/box run p3meta` (second rental; one process per model × seed; 5622 s). Report:
`outputs/p3/meta/report.md`. A0 parity held first (14/14, `outputs/p3/a0_report.md`).

| held-out dataset | v1.1 AUC | A1 rank pool (5 seeds, sd) | A2 bin transformer (5 seeds, sd) |
|---|---|---|---|
| cwru | 0.951 | 0.958 (0.001) | 0.943 (0.011) |
| pu | 0.899 | **0.839 (0.041)** | 0.823 (0.036) |
| ottawa_2018 | 0.998 | 0.999 (0.000) | 0.936 (0.042) |
| ottawa_2023 (own) | 0.971 | 0.987 (0.001) | 0.976 (0.012) |
| mfpt | 0.996 | 0.985 (0.005) | **0.873 (0.113)** |
| hust | 0.894 | **0.936 (0.002)** | 0.915 (0.014) |
| xjtu_sy | 0.830 | 0.844 (0.001) | 0.805 (0.013) |
| paderborn_ldm | 0.678 | 0.692 (0.005) | 0.622 (0.025) |
| ims | 0.833 | 0.843 (0.006) | 0.778 (0.048) |
| **gate** | | mean dAUC **+0.004** (need +0.02); worst pu −0.060; dFA worst −0.008 → **FAIL** | mean dAUC **−0.042**; worst mfpt −0.123; dFA worst +0.136 → **FAIL** |

Reading (§6): **both fail**. With these features, max-z against the unit's own baseline is as good
a comparator as eight source datasets can teach. **Track A closes.** This is a result, not a defect
of the track.
- **A1** stays close to v1.1, because it starts *as* v1.1 and moves only a little.
  - It gains a little on 7 of 9 held-out sets: HUST +0.042, Ottawa 2023 +0.016, XJTU-SY / LDM +0.014,
    IMS +0.010.
  - It loses where it matters most: Paderborn −0.060, with a large seed spread (0.041). That is
    the one set with real damage on several physical bearings.
  - The "specialist" reading does not apply either: of the hard rows it helps HUST, LDM and IMS a
    little and hurts Paderborn a lot.
- **A2** overfits the sources. The order-aware transformer learns which bins matter on the training
  bearings, and that does not carry to another bearing's orders. It loses on 7 of 9 sets and raises
  worst-unit FA by 0.14.
  - This is the same lesson as notes/12, one level up: *what* rose does not transfer, only *that
    something rose against the unit's own baseline*.
- Together with Track B (notes/31), the conclusion is that the gain is not in learning a better
  comparison. The physics representation plus its nuisance invariance, plus a simple max-rise
  rule against the unit's own baseline, is close to what this data supports.
- A3 (synthetic-fault episodes) is **not run**. It was conditional on A1 or A2 showing a signal.
