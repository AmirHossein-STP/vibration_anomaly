# 25. WP15: varying load and speed: Paderborn LDM run-to-failure (unseen)

MASTER_PLAN Phase 2 WP15, question **Q1**: under time-varying load and speed, is the level
trigger confounded, and does the physics score take over? Written 2026-10-22, **before any
LDM signal or score was looked at.** What was read:
- the description PDF (Table 1 bearing, Table 2 operating ranges, Table 3 durations and defects);
- the per-bearing logs (warm-up and stop reasons);
- the operating-condition table format.

## 1. Data
- 17 run-to-failure tests on 61806-2RS bearings: 19 balls, BPFO ≈ 8.65, BPFI ≈ 10.35 orders.
- Speed and load are re-drawn at random before every record (Table 2):
  - speed 1500–3600 rpm;
  - static load 2000–4000 N;
  - dynamic load 250–1700 N (sinusoidal for B01–B06, noise for B07–B17).
- One 1.6 s record roughly every 12 s. Each record has its measured mean speed.
- Condensed on the box to 12 kHz (two channels; A = accHorizRear_A is primary).
- Defects found after dismantling: Table 3 (IR / OR / B).

## 2. Protocol (method v1.0, notes/22; frozen as of commit 27a2b51 plus the WP13 catalog)
- **Per record.** Two 1 s windows (hop 0.5 s) at the record's measured speed (`meanAbs_speed`).
  The record score is the mean of its two windows.
- **Warm-up records** (set speed < 1000 rpm: 10 min at 750 rpm, no dynamic load) are excluded.
- **Own baseline:** the first **150 records** after warm-up (≈ 30 min). Cross-fitted
  calibration groups are the records.
- **Statistics per record**, each standardised against the baseline records:
  - **phys6**: six-line physics score (catalog 61806-2RS), standardised by its cross-fitted calibration scores;
  - **geofree**: the geometry-free score, the same way;
  - **L_pooled**: log RMS, z against the baseline records;
  - **L_cond**: log RMS **conditioned on the operating point**. Residual of a linear regression,
    fit on the baseline records, of log RMS on [log speed, static load, dynamic load];
    z of the residual.
- **"Assumed healthy" segment**, stated as an assumption: records 151–450 (the next ≈ 60 min).
  Used on the 13 bearings with ≥ 1000 records (all but B01, B03, B05, B12).
- **Trigger:** CUSUM, k = 0.5, from record 151. The threshold per held-out bearing is set leave
  one bearing out, just above the largest statistic in the other long bearings' assumed-healthy
  segments (FA budget 0, notes/18).
- **Detection** = first record over threshold after record 450 (after 150 for the four short
  bearings). Lead = time from detection to the last record, in hours.

### 2a. Amendment (2026-10-22, before any score: the first cache build crashed)
Some records have a **measured** speed near zero: the rig stopped, or the test was interrupted
(the logs mention both). They have no valid order axis. **Records with measured speed < 1000
rpm or not finite are excluded**, by the same 1000 rpm threshold as the warm-up rule. This is a
data-validity filter decided from the operating-condition table alone, not from any score.

## 3. Pre-registered readings
- **(a) Confounding.** Record-level FA in the assumed-healthy segment at conformal α = 0.10
  (the per-record raw z, not the CUSUM). Predicted: **L_pooled FA > 0.25**, confounded by
  speed and load, while **phys6 FA ≤ 0.15**.
- **(b) Conditioning.** L_cond's FA is lower than L_pooled's.
- **(Q1 verdict) The physics score takes over as trigger under varying conditions** if phys6
  CUSUM detects at least as many bearings as the better of L_pooled and L_cond CUSUM at FA
  budget 0. Otherwise the level trigger stays, with conditioning if L_cond is the better level.
  Either way the decision layer in notes/22 §4 gets the corresponding rule.
- **(c) Head to head.** Geometry-free "wins LDM" if its CUSUM detects at least as many bearings as
  phys6, with median lead ≥ phys6's. **If it also won HUST (it did, notes/24), geometry-free
  becomes the primary detector, and the six lines stay for typing.**
- **(d) Typing at end of life** (last 10% of records, majority vote, phys6 lines).
  - Expected inner on IR without OR: B02, B03, B04, B06, B07, B08, B12, B13, B15.
  - Expected outer on OR without IR: B09, B10.
  - Not counted: mixed IR+OR (B01, B11, B16) and ball only (B05, B14, B17).

  It passes if ≥ 2/3 of the 11 are right.

---

## 4. Results (after computing, 2026-10-22)

Runs: `remote/box run fetch-paderborn_ldm` (50 GB streamed per bearing, 7 GB condensed),
`ldmcache` (379k windows per feature set, ~8 min each), `wp15`. Table: `outputs/wp15/wp15_report.md`.

Channel A (primary), 17 bearings, 1.6 s records about every 12 s, speed and load re-drawn every record:

| statistic | record FA at α 0.10 (assumed-healthy) | CUSUM detected (FA budget 0) | median lead |
|---|---|---|---|
| phys6 (six lines) | **0.106** | 9/17 | 1.3 h |
| geofree | 0.122 | 10/17 | 2.5 h |
| L_pooled (log RMS) | 0.191 | 13/17 | 1.6 h |
| **L_cond** (log RMS given speed and load) | 0.151 | **14/17** | **2.8 h** |

Readings, as pre-registered:
- **(a) Confounding: FALSE as stated.** The level's FA (0.19) is higher than the physics score's (0.11),
  but not above the predicted 0.25. Varying speed and load confound the raw level only moderately on this rig.
- **(b) Conditioning helps: TRUE.** Regressing log RMS on log speed, static load and dynamic load
  over the baseline lowers FA from 0.19 to 0.15, and gives the most detections (14/17) with the longest lead.
- **Q1 verdict: the physics score does NOT take over as trigger.** The level trigger stays, **conditioned
  on the operating point**. This is the "varying conditions" rule for notes/22.
- **(c) Geometry-free wins LDM: TRUE** (10 vs 9 detected, 2.5 vs 1.3 h). It also won HUST (notes/24).
  So **per pre-registration, geometry-free becomes the primary detection score, and the six lines stay
  for typing.**
- **(d) Typing at end of life: 7/11 = 0.64 → FAIL** (the bar was 2/3). Channel C gives 8/11.
  The misses are IR+B bearings typed outer (B04, B06, B08, B12). With a co-existing ball defect,
  "which race rose most" is not reliable on this rig.

What else the table shows:
- Lead times are short against test durations (hours on 2–176 h tests). Most damage here grows
  fast at the end.
- Ball-only bearings (B05, B14, B17) are found only by the level (B14, B17), and late.
  That is the documented ball gap again.
- B06's very early level detection (60 h before the end of a 66 h test) is likely a false
  early alarm: its log records a broken shaker stinger.
