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
(written at unsealing: the commit and the time)
