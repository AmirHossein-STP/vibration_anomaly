# 27. WP18: the IMS final exam (sealed until the freeze)

MASTER_PLAN Phase 2 WP18 (formerly WP11). Written 2026-10-22, **while IMS is still sealed**:
`fetch.py` has never extracted it, and nobody has opened its files or its readme. The protocol
below comes from the published description of the set (Qiu et al. 2006; the NASA PCoE catalog
entry) and is fixed before unsealing. The scored system is the **frozen `bearing` service**
(method v1.0 at the commit recorded in notes/22 §8), run through its CLI. It is not the analysis
scripts.

## 1. What IMS is (from the literature)
- Four Rexnord ZA-2115 double-row bearings on one shaft at **2000 rpm**, with 6000 lb radial
  load on bearings 2 and 3. Accelerometers on each housing.
- 1 s records (20480 samples, 20 kHz), mostly every 10 min. Each test runs until failure:

  | test | channels | duration | documented failure |
  |---|---|---|---|
  | 1 | 8 (x, y per bearing) | about 34 days | bearing 3 inner race, bearing 4 roller |
  | 2 | 4 | about 7 days | bearing 1 outer race |
  | 3 | 4 | about 44 days | bearing 3 outer race |

  The readme files are known to name test 3's folder "4th_test"; it is treated as test 3.

## 2. Protocol (fixed now)
- **Units:** test × bearing, 12 in total. In test 1 the first channel of each bearing pair
  (x) is used, and y is reported as secondary.
- **Bearing:** catalog "Rexnord ZA-2115" (BPFO 7.09, BPFI 8.91 orders). Six-line detector.
  The geometry-free detector is reported alongside only if WP15 makes it primary; then the roles swap.
- **Speed:** rpm 2000 for every record (the logged setpoint of a speed-controlled rig, notes/22 §1).
- **Baseline:** the first **200 records** of each unit (the service's default minimum: 200 windows
  of 1 s; about 33 h at the 10 min cadence). Each record is one `train/<unit>/` file.
- **Monitoring:** every later record, in time order, is one `in/<unit>/` file, scored in a single
  `infer` job per unit. That gives verdicts, typing, and the per-unit CUSUM trend statistics (level
  and physics) with the thresholds stored at calibration.
- **Reported:**
  - **failing units** (T1-B3, T1-B4, T2-B1, T3-B3): time of the first `trend_alarm` after
    the baseline, lead to the end of the test (hours), the first `faulty` verdict, and typing
    (majority `fault_type` over the last 10% of records). Expected: inner for T1-B3, outer for
    T2-B1 and T3-B3; T1-B4 (roller) is not counted;
  - **units that did not fail** (the other 8): trend-alarm events per day, and the share of records
    `faulty`. They share a shaft with failing bearings, so a late alarm there may be real
    cross-coupling; this is stated, not removed.

## 3. Pre-registered readings (no gate: the result is reported whatever it is)
The IMS result is **consistent with development** if:
- a trend alarm precedes the end of the test on ≥ 3 of the 4 failing units;
- typing is right on ≥ 2 of the 3 race failures;
- the non-failing units average ≤ 1 trend event per day before the last 10% of their test.

Otherwise the method was overfit to the development sets, or IMS differs in a way not
anticipated (MASTER_PLAN §7). That is reported prominently, before any claim, in the README and
in MASTER_PLAN §1.

## 4. Unsealing record
- Frozen system: commit `e7c01a3eee93d00c2b0c7b1b25a915ca663f5629` (notes/22 §8). Detection = geometry-free (primary), six-line reported alongside.
- Unsealed on the box with `fetch.py unseal ims --commit e7c01a3eee93 --confirm WP11` right after this note was committed.

---

## 5. Results (2026-10-22), frozen system `e7c01a3`

Run: `remote/box run wp18`. Report: `outputs/wp18/wp18_report.md`.

**Three runner bugs** were found and fixed during the exam. All were in `scripts/analysis/wp18_ims.py`;
the frozen method and service were untouched.
1. The archive's `__MACOSX` metadata files broke the extraction.
2. The six-line service root was a hard-linked copy, so its config overwrote the geometry-free
   config. **Run 1 therefore scored six-line twice** (kept as `wp18_report_run1_both_six_line.md`).
3. The typing check compared "inner race (hint)" with "inner race".

All numbers below come from the corrected run.

### 5.1 Pre-registered readings (§3): **NOT consistent with development** (both detectors)

| reading | geometry-free (primary) | six-line |
|---|---|---|
| trend alarm before end, failing units (need ≥ 3/4) | 4/4 | 4/4 |
| typing right, race failures (need ≥ 2/3) | 2/3 | **3/3** |
| non-failing units, trend events/day (need ≤ 1) | **1.02** | **1.03** |

The failure is on false alarms, and narrow. The "4/4 before end" pass is **not informative**: the trend
alarm fires 2–84 h after the baseline on *every* unit, healthy ones included, years before any failure
by the standards of these tests. With a 200-record (≈ 33 h) baseline at a 10-min cadence, the level
drifts away from its baseline on all units, and the CUSUM at h ≥ 5 crosses early.

### 5.2 Post hoc, descriptive only (not part of the verdict)
Share of a record's windows flagged, first half of the test vs its last 2%:

| unit | documented | geometry-free | six-line |
|---|---|---|---|
| T1-B3 | inner race | 0.10 → **0.88** | 0.06 → 0.30 |
| T1-B4 | roller | 0.07 → **0.90** | 0.03 → 0.03 |
| T2-B1 | outer race | 0.13 → **0.94** | 0.08 → 0.25 |
| T3-B3 | outer race | 0.16 → **0.97** | 0.14 → 0.22 |
| the 8 others | none | 0.09–0.21 → 0.17–0.94 | 0.07–0.23 → 0.05–0.38 |

- **The geometry-free detection score works on IMS.** Every failing bearing ends near 0.9+, and it is the
  highest unit in its test, or tied (T2-B4, also 0.94, shares the shaft with failing T2-B1).
  Neighbours rise too as the whole rig degrades.
- **The six-line score does not**, on this double-row roller bearing. The v1.1 switch to
  geometry-free, decided on HUST and LDM *before* IMS was opened, is borne out.
- **The trend layer is what fails.** Level and CUSUM drift on all units long before failure. The window-level
  verdict carries the signal; the trend trigger as specified does not separate units.

### 5.3 What this changes (MASTER_PLAN §7)
- The method is **not** "overfit to development" in its detection score: geometry-free transfers.
- The **trend trigger is not ready**. Under a 10-min cadence with slow drift it gives about 1 false
  event/day and no specificity. Before any pilot, the trigger needs a baseline that spans days
  of operation (notes/26), or a slower statistic (e.g. a weekly re-baselined EWMA). It also needs a new
  confirmation set. **IMS is now spent**, so a fresh unseen run-to-failure set is required.
- Typing as a hint: 3/3 with six lines, 2/3 when typing runs on geometry-free-flagged windows.
