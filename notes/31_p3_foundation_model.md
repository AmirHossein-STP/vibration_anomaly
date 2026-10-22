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

## 4. Results
(to be filled by `scripts/analysis/p3_fm.py`)
