# 26. WP16: long-baseline false alarms: FEMTO/PRONOSTIA (unseen; Q2)

MASTER_PLAN Phase 2 WP16, question **Q2**: what do false alarms per day look like with
realistically long baselines, and how long must a unit's healthy baseline be? Written
2026-10-22, **before any FEMTO signal was read.** What was read: the archive's member list
(Learning_set: 6 bearings, Full_Test_Set: 11, Test_set truncated and unused) and the dataset's
documentation (Nectoux et al. 2012).

## 1. Data
- 17 complete run-to-failure bearings under 3 constant conditions (1800 rpm / 4000 N,
  1650 / 4200, 1500 / 5000).
- One record (`acc_*.csv`) every 10 s: 0.1 s at 25.6 kHz, horizontal and vertical.
- Each test was stopped at 20 g.
- 0.1 s is 2.5–3 shaft revolutions, **too short for order features**. So this tests only the
  **trigger**: level and impulsiveness against the unit's own baseline (notes/18 §5).
  The physics score is not applicable, and that is reported, not hidden.

## 2. Protocol
- Per record, horizontal channel (primary; vertical reported): log RMS and log kurtosis of the
  raw 0.1 s record, demeaned.
- **Own baseline** = the first B records, for **B ∈ {60, 180, 360}** (10, 30, 60 min).
  Statistic = z against the baseline; CUSUM with k = 0.5 from record B + 1.
- **Assumed healthy** (an assumption): records B+1 … B+720, the next 2 h. Used on bearings with
  ≥ B + 720 + 100 records.
- **Operating point:** FA budget 0, leave one bearing out (notes/18). The threshold for a held-out
  bearing is just above the largest CUSUM in the other bearings' assumed-healthy segments.
- **Reported per B:**
  - bearings detected;
  - median lead time (hours) to the end of the test;
  - **held-out false alarms per day**: alarm *events* (rising edges) in held-out assumed-healthy
    segments, per day of monitoring at 8640 records/day.

## 3. Pre-registered readings
- **Baseline-duration rule (for notes/22 and the data spec).** The shortest B whose held-out
  FA/day ≤ 1 *and* whose detected count is within 1 of the best B. If no B reaches ≤ 1/day, the
  achieved FA/day is reported as the rule's input instead. A pilot then needs a longer baseline
  than 60 min, by extrapolation, and says so.
- **RMS vs kurtosis.** As on XJTU-SY (notes/18), RMS CUSUM is expected to detect at least as many
  bearings as kurtosis CUSUM at each B.

---

## 4. Results (after computing, 2026-10-22)

Run: `remote/box run wp16` (≈ 2 min, archives read in place). Table: `outputs/wp16/wp16_report.md`.

Horizontal (primary), own-baseline CUSUM at FA budget 0 (leave one bearing out):

| baseline | RMS detected | RMS median lead | RMS held-out FA/day | kurtosis detected |
|---|---|---|---|---|
| 10 min | 12/17 | 0.04 h | 1.2 | 5/17 |
| 30 min | **15/17** | 0.11 h | 1.3 | 9/17 |
| 60 min | 15/17 | 0.26 h | 8.0 | 9/17 |

Readings:
- **RMS ≥ kurtosis at every baseline length: yes**, as predicted and as on XJTU-SY.
- **Baseline-duration rule: unresolved.** No baseline length reaches ≤ 1 FA/day.
  - The 1.2–1.3/day values are *one* false event in all held-out healthy segments: 9–10
    bearings × 2 h ≈ 0.8 day. So the data cannot resolve rates below about 1.2/day.
  - At 60 min the assumed-healthy window (records 361–1080) runs into the early degradation of
    the shorter-lived bearings, and "FA" rises to 8/day. That is the limit of assuming health by
    position in life.
- **Warning time is short on FEMTO:** a median of minutes. PRONOSTIA's accelerated tests degrade
  abruptly at the very end (a known property of this set). RMS fires once damage is
  large, and kurtosis earlier on fewer bearings (lead 1–1.5 h).

What this means for the spec (notes/22 §4 and the data spec):
- At one record per 10 s, an own-baseline RMS CUSUM with an FA-budget-0 threshold gives about
  **one false event per day** of healthy operation. That is a rate an operator can accept for an
  advisory "watch" level, not for an automatic stop.
- **The baseline must be at least 30 min of steady healthy operation per operating state**
  (10 min loses 3 of 15 detections). Longer is not automatically better if the "healthy" period
  quietly includes early damage. Commissioning must confirm health independently (a clean
  calibration verdict, notes/22 §1).
- The physics score cannot run on 0.1 s snapshots. **Records must be ≥ 1 s long** (≥ 15
  shaft revolutions). This goes into the data spec as a hard requirement.
