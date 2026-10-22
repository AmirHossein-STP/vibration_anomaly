# 31. Phase 3 Track B: generic foundation-model embeddings, referenced to the unit's own healthy data

**Pre-registration.** Written 2026-10-22, before any embedding was computed.

## 1. Question
Spectrogram networks trained on one machine failed on another (notes/12). The fix that worked
had two parts: a physics representation (envelope orders) **and** referencing to the target's own
healthy data (notes/16–17). Which part carried the transfer?
- Track B swaps the physics representation for a **generic pretrained time-series model**, with
  no vibration-specific design and no training here.
- It keeps the referencing: each window is scored by its distance to the unit's own healthy
  embeddings.

## 2. Setup (fixed now)
- **Model:** MOMENT-1-base (AutonLab, MIT, Hugging Face `AutonLab/MOMENT-1-base`, ICML 2024),
  `task_name="embedding"`.
  - It was pretrained on the "Time-Series Pile". Whether that includes any of our bearing sets
    was **not verified**, so an overlap would flatter Track B.
  - VibFM (PHME) was the first choice. No public weights were found.
- **Windows:** exactly the geometry-free windows (`v2_cache` windowing, 12 kHz, 1 s, hop 0.5 s;
  Ottawa 2018 order-tracked spans).
  - Each window is cut into non-overlapping 512-sample chunks (43 ms each). Each chunk is embedded
    (MOMENT normalises every input itself, so level is removed), and the chunk embeddings are
    averaged per window (768-d).
  - Environment: separate venv `~/work/venv_fm` (`remote/requirements-p3b.txt`).
- **Score:** mean cosine distance to the k = 5 nearest support embeddings.
  - Cross-fitted on the support by the same groups as v1.1 (`default_groups`).
  - Conformal threshold at α = 0.10.
- **Control (B-ctrl):** the same kNN score on the geometry-free features (log bins z-scored by the
  support). This separates "kNN vs max-z" from "MOMENT vs physics".
- **Datasets:** the six static development sets with Track A's exact folds
  (`p3_episodes.static_parts`): cwru, pu (4 conditions), ottawa_2018, ottawa_2023 (own), mfpt, hust.
  The run-to-failure sets are left out, because their raw volume makes per-window embedding costly.
  Detection on the static sets is the question here.

## 3. Readings
`v1.1` = the A0 reference (`outputs/p3/a0_reference.json`).

| outcome | meaning |
|---|---|
| MOMENT mean AUC ≥ v1.1 − 0.02, and every dataset ≥ v1.1 − 0.10 | **Referencing carried the transfer.** A generic representation is enough once each unit is compared with itself. Embeddings become a candidate for components that have no physics model (L1 for non-bearing parts) |
| MOMENT mean AUC < v1.1 − 0.05 | **The representation matters even with own-baseline referencing.** Physics stays necessary for bearings, which confirms notes/12–13 from a second angle |
| in between | Reported as is. Where it loses is read per dataset |
| B-ctrl ≈ v1.1 while MOMENT loses | The gap is the representation, not the kNN rule |

## 4. Results (2026-10-22): **in between**: it ranks faults superbly but cannot be thresholded
Run: `remote/box run p3fm` (resumed on a second rental; the embeddings are cached per dataset).
Report: `outputs/p3/fm/report.md`.

| dataset | v1.1 AUC | MOMENT kNN AUC | B-ctrl (cosine kNN on geometry-free z) | v1.1 FA worst | MOMENT FA worst |
|---|---|---|---|---|---|
| cwru | 0.951 | **1.000** | 0.368 | 0.103 | **1.000** |
| pu (4 conditions) | 0.899 | **0.732** | 0.385 | 0.467 | **1.000** |
| ottawa_2018 | 0.998 | 1.000 | 0.520 | 0.278 | 0.200 |
| ottawa_2023 (own) | 0.971 | 1.000 | 0.449 | 0.333 | 0.778 |
| mfpt | 0.996 | 1.000 | 0.677 | 0.000 | 0.000 |
| hust | 0.894 | 0.993 | 0.347 | 0.126 | 0.832 |
| **mean** | 0.952 | 0.954 | 0.458 | | |

Reading (§3): **in between**. The mean equals v1.1 (≥ v1.1 − 0.02 holds), but Paderborn loses 0.17,
beyond the 0.10 allowance.

- **Ranking vs calibration.** On 5 of 6 sets a generic, untrained time-series model, compared with
  the unit's own healthy data, ranks faults above healthy windows almost perfectly. But the
  threshold from the cross-fitted support does not hold: worst-unit FA is 0.78–1.00 on four sets.
  - The embedding is extremely sensitive to *any* change: another load (CWRU, HUST folds hold out a
    load), another physical bearing (Paderborn), or another stretch of the same recording (Ottawa
    2023 own).
  - Faults are simply further away than those nuisance changes are.
- **Where it fails is the realistic case.** Paderborn is the only set whose held-out healthy windows
  come from a *different physical bearing* than the support. There the nuisance distance competes
  with the fault distance, and ranking drops to 0.73.
- **So the answer to §1 is "both".** Own-baseline referencing makes even a generic representation
  rank well. But only the physics representation is *invariant enough to nuisance changes* for a
  calibrated threshold to hold. notes/12's spectrogram failure was a representation failure in this
  sense: not blindness to faults, but sensitivity to everything else.
- **B-ctrl is uninformative.** Cosine distance on z-scored bins is a poor rule (AUC 0.35–0.68,
  below v1.1's max-z on every set). It shows the max-z *rule* matters for the physics features,
  and it does not separate representation from rule as intended. Reported as pre-registered.
- **Not shipped.** A possible later use, untested: change detection *on the same unit at the same
  operating point* (the L1 layer for components with no physics model), where nuisance changes
  are controlled by design. That would need its own pre-registration and a real-machine check.
- Caveat: MOMENT's pretraining corpus was not checked for overlap with these sets.
