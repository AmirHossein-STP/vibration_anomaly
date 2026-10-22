# 23. WP13: geometry in, and a geometry-free fallback

MASTER_PLAN Phase 2 WP13. Goal: "the system works on many bearings" (owner,
2026-10-22). Written 2026-10-22. **§1–§3 were written before any geometry-free number was computed.**

## 1. Supplying geometry (`src/domain_adaptation/bearings.py`)
A unit's bearing is given in one of three ways:
- a **catalog designation**;
- explicit (n, d, D, α);
- the manufacturer's **fault factors** (BPFO/BPFI/FTF/BSF in orders), which is what most catalogs publish.

Catalog entries are only bearings verified against a cited source, each with a test that it
reproduces its published fault frequencies:

| designation | source | check |
|---|---|---|
| SKF 6205-2RS JEM (CWRU) | CWRU bearing data center | BPFO 3.5848, BPFI 5.4152 |
| 6203 (Paderborn KAt) | Lessmeier et al. 2016 | BPFO 3.053, BPFI 4.947 |
| ER16K (Ottawa 2018) | Huang & Baddour 2018 | BPFO 3.57, BPFI 5.43 |
| NSK 6203ZZ / FAFNIR 203KD (Ottawa 2023) | Sehri et al. 2023, Table 2 | geometry only |
| LDK UER204 (XJTU-SY) | Lei et al. tutorial, Table 1 | BPFO 107.91 Hz at 35 Hz |
| MFPT test bearing | factors shipped in every file | BPFO 81.125, BPFI 118.875, FTF 14.8375, BSF 63.91 Hz at 25 Hz |
| Rexnord ZA-2115 (IMS) | Qiu et al. 2006 | BPFO 236.4, BPFI 296.9 Hz at 2000 rpm |
| 61806-2RS (Paderborn LDM) | LDM description, Table 1 | z 19, Dw 3.18, DT 35.5 mm |
| KG 6204–6208 (HUST) | Thuan & Hong 2023 | balls and Dw from the paper; pitch = (bore + OD) / 2, an approximation, stated |

## 2. Geometry-free score (pre-registered)
For a bearing whose geometry is unknown. It uses the same CPW + squared-envelope order spectrum
as method v1.0, but instead of six named lines:
- 135 bins of 0.1 order over 1.5–15 orders, each the peak in the bin divided by the
  envelope sum over 0.5–12 orders;
- score = max over bins of the z-score of log(bin) against the unit's healthy windows;
- cross-fitted conformal threshold, α = 0.10 (`method.fit_baseline` on the bin columns);
- no typing. The **dominant anomalous order** (the arg-max bin) is reported instead.

Order tracking on Ottawa 2018 is as in method v1.0. Folds and baselines are exactly those of
notes/17 (`wp5_score.folds`).

## 3. Gate
The geometry-free score **ships as the unknown-geometry fallback** if its race-fault AUC
≥ arm d (`d_xf`, notes/17) − 0.05 on ≥ 5 of these 6 rows:
- CWRU;
- Paderborn N09;
- Paderborn N15 (the worst of its three conditions);
- Ottawa 2018;
- Ottawa 2023 own-unit;
- MFPT.

Otherwise the fallback for unknown geometry is the own-baseline level/kurtosis statistics, with
a "geometry unknown, diagnosis unavailable" warning.

Also reported: how often the dominant anomalous order, on faulty windows over threshold, falls
within ±0.15 orders of a true BPFO/BPFI harmonic 1–3. That measures whether the fallback can
point a technician at the fault line.

All 6 datasets are already spent for development, so this is a development comparison.
Confirmation is on HUST (WP14) and LDM (WP15).

---

## 4. Results (after computing, 2026-10-22)

Runs: `remote/box run geofreecache` (all five datasets, ~25 s) and `remote/box run wp13`.
Table: `outputs/wp13/wp13_report.md`. Catalog tests: `tests/test_bearings.py`. All 9 published
fault-frequency checks pass, IMS ZA-2115 included.

**Gate: SHIPS as the unknown-geometry fallback (6/6 rows within 0.05 of arm d).**

| dataset | arm d AUC | geometry-free AUC | geometry-free recall / FA (worst) | dominant order on a true fault line |
|---|---|---|---|---|
| CWRU | 0.929 | 0.951 | 0.92 / 0.09 (0.10) | 0.85 |
| Paderborn N09 | 0.901 | 0.919 | 0.87 / **0.20** (0.48) | 0.33 |
| Paderborn N15 ×3 | 0.84–0.86 | 0.88–0.90 | 0.75–0.80 / **0.18–0.21** | 0.59–0.64 |
| Ottawa 2018 | 0.922 | **0.998** | 1.00 / 0.10 (0.28) | 0.82 |
| Ottawa 2023, fleet baseline | 0.712 | 0.825 | 0.50 / 0.10 (1.00) | 0.56 |
| Ottawa 2023, own-unit | 0.930 | 0.971 | **0.27** / 0.02 (0.33) | 0.66 |
| MFPT | 1.000 | 0.996 | 0.74 / 0.00 | 0.95 |

What this means:
- **Unexpected: geometry adds nothing to detection on these data.** The geometry-free score
  *ranks* faulty above healthy as well as or better than the six named lines everywhere.
  Faults evidently light orders other than BPFO/BPFI × 1–3 too: higher harmonics, sidebands, or
  the other race. This is consistent with notes/19, where `v2max`, the max over more lines, won
  on three datasets.
- **But its threshold is worse calibrated.** Its max over 135 bins is much more sensitive to how
  unit-to-unit and window-to-window spread is sampled:
  - with a fleet baseline (Paderborn, 5 calibration units) held-out FA is 0.18–0.21 against α = 0.10;
  - with a tiny own baseline (Ottawa 2023, 9 windows) the cross-fitted threshold is very
    conservative (FA 0.02, recall 0.27).

  AUC is threshold-free. A deployed detector is not.
- The dominant anomalous order lands on a true fault line on 33–95% of detected windows. So the
  fallback can usually point at the right line. Telling which race it is still needs the geometry.

**Consequences:**
1. Geometry-free ships as the **unknown-geometry fallback** (notes/22 §6), with a calibration caveat:
   - it needs a **longer own baseline** than the six-line score;
   - its FA must be checked per unit at commissioning.
2. Because geometry-free beat the named lines on *every* development set, it becomes a
   **co-primary candidate**. This was not pre-registered, so it is not a decision. **HUST (WP14)
   and LDM (WP15), both unseen, will compare the two head to head**, on AUC *and* on held-out
   FA. That pre-registration decides which one detects, and the geometry-aware lines keep the
   typing either way.
