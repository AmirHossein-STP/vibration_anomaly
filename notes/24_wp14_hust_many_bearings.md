# 24. WP14: many bearings: HUST, five bearing types (unseen)

MASTER_PLAN Phase 2 WP14. Written 2026-10-22 **before any HUST signal or score was looked at.**
What was read: the paper (Thuan & Hong 2023), the file list, and per-file metadata only
(variable names and shapes, and the measured shaft speed `fs`, which is 22.5–25 Hz).

## 1. Data
- 6204, 6205, 6206, 6207, 6208 (KG Bearing India).
- States: healthy (N), inner (I), outer (O), ball (B), inner+outer (IO), inner+ball (IB),
  outer+ball (OB). B and IB are missing for 6204.
- Loads 0 / 200 / 400 W (last digit 0/2/4).
- 10 s at 51.2 kHz per file, accelerometer vertical on the bearing. Defects are 0.2 mm wire-cut cracks.
- Geometry from the catalog (notes/23 §1): balls and ball diameter from the paper, pitch
  diameter (bore + OD) / 2.
- Speed: each file's measured `fs` (shaft Hz).

## 2. Protocol (method v1.0, notes/22; resampled to 12 kHz, 19 windows per file)
- Per bearing type and per held-out load L (3 folds per type):
  - **baseline** = the same type's healthy files at the other two loads, cross-fitted by
    recording;
  - **held-out healthy** = its healthy file at L;
  - **faults** = all of that type's defect files at L.

  The baseline spans load states, but the faults are scored at the same load as the held-out
  healthy file.
- Two detectors, both cross-fitted, conformal α = 0.10:
  - **six-line** (arm d, catalog geometry);
  - **geometry-free** (notes/23).
- Race-involving faults are I, O, IO, IB and OB. Ball-only (B) is reported separately.
- Typing (six-line) is scored on I and O only.

## 3. Pre-registered readings
- **Many-bearings gate (from the plan):** six-line race AUC ≥ 0.80 on ≥ 4 of the 5 bearing
  types, and typing > 0.6 over all I/O windows. Pass means the method works across bearing
  sizes at this rig. Failures are named per type.
- **Head to head (notes/23 §4):** geometry-free "wins HUST" if its race AUC ≥ six-line's on
  ≥ 4 of 5 types **and** its mean held-out FA ≤ 0.15. Geometry-free becomes the primary
  *detector* only if it also wins LDM (WP15). Typing always uses the six lines.
- Ball-only: no expectation (the documented gap); reported.

---

## 4. Results (after computing, 2026-10-22)

Runs: `remote/box run hustcache` (1807 windows, seconds) and `remote/box run wp14`.
Table: `outputs/wp14/wp14_report.md`.

**Many-bearings gate: PASS.**
- Six-line race AUC ≥ 0.80 on **4 of 5 bearing types**:

  | type | six-line race AUC |
  |---|---|
  | 6204 | 0.98 |
  | 6206 | 0.88 |
  | 6207 | 0.95 |
  | 6208 | 0.84 |

- Typing is right on **0.83** of 570 inner/outer windows. By type: 6204 1.00, 6205 0.81, 6206 0.89, 6207 0.82, 6208 0.62.
- **Failure, named: 6205** (AUC 0.45, worst load 0.21, held-out FA up to 1.00).
  - Its healthy file at one load scores as anomalous against the other two loads, so the
    baseline does not span 6205's load states.
  - It is the same per-unit failure as Paderborn K005 and Ottawa 2023 b02: one unit whose
    "healthy" data is not like its own other healthy data.
  - The geometry-free score is less hurt (0.77).

**Head to head: geometry-free wins HUST.**
- Its AUC ≥ six-line on 4/5 types (all but 6207, 0.93 vs 0.95).
- Mean held-out FA is 0.04, within the 0.15 limit.
- At the conformal threshold it is **more conservative**: race recall 0.28–0.89 vs 0.17–0.93.
  This is the calibration pattern of notes/23 again: a max over 135 bins needs a longer baseline
  to set a tight threshold.

Per pre-registration, geometry-free becomes the primary detector only if it also wins LDM (WP15).

What HUST adds to "works on many bearings":
- Five sizes of deep-groove ball bearing (8 or 9 balls, 7.6–12 mm balls) on one rig. The same
  method, fed only the catalog geometry and the measured speed, detects and types race damage on
  four of them without any fault data from them.
- The failure mode remains a unit whose healthy baseline is not representative, not a bearing
  size the method cannot read.
- Ball-only faults: 0.00–0.39 (six-line) and 0.26–0.58 (geometry-free). Still the documented gap,
  though geometry-free sees part of it.
