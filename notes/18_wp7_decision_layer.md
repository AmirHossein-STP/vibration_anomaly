# 18. WP7: decision layer on XJTU-SY: lead time at a false-alarm budget

MASTER_PLAN.md WP7. Written 2026-10-22, **§1–§3 before any number below was
computed.** It follows directly from notes/17 §5.4. There, a 3-in-a-row rule
on the raw per-minute physics score fired late or spuriously, and "earliest
alarm" was compared without matching false alarms.

## 1. Question

At a false-alarm budget chosen **without looking at failure timing**, how much
warning does each detector give before end of life? Does smoothing the
physics score over minutes make it competitive with the overall level (RMS)?

## 2. Protocol (fixed here)

- **Data.** XJTU-SY, horizontal channel (primary as in notes/17). Vertical and the
  max of both channels are reported. 15 bearings, one record per minute.
- **Own baseline.** Minutes 1–20 of each bearing. Each statistic is standardised
  against them:
  - physics: arm d CPW+SES score, standardised by the mean and sd of its
    *cross-fitted* calibration scores (leave one minute out; notes/17 §5.2);
  - RMS: (rms − mean) / sd over minutes 1–20.
- **Detectors.** Each is computed on the standardised series u_t from minute 21 on,
  with state starting at 0:
  - `raw`: u_t;
  - `ewma`: e_t = 0.2 u_t + 0.8 e_{t−1};
  - `cusum`: C_t = max(0, C_{t−1} + u_t − 0.5).

  Each detector is applied to physics, to RMS, and to `comb` = max(u_phys, u_rms).
  That gives 9 detectors. An alarm is the statistic above a threshold h.
- **"Healthy" segment (an assumption, stated as such).** Minutes 21–60 of the
  12 bearings whose life is ≥ 120 minutes (all but 1_5, 2_4, 3_5).
- **Detection time.** The first alarm after minute 60, or after minute 20 for the 3 short-lived
  bearings. Lead = life − detection minute. On the long-lived bearings, alarms in
  minutes 21–60 count as false alarms.
- **Operating point, FA budget = 0, leave one bearing out.** For each held-out bearing,
  h is set just above the largest statistic in the *other* long-lived bearings' minutes 21–60.
  That point comes from healthy data only, never from failure timing. Reported per detector:
  - bearings detected;
  - median lead time (minutes and as a fraction of life);
  - false alarms on held-out healthy segments: minutes in the alarm state, converted to
    alarms per day of monitoring at one record per minute.
- **Curve.** Sweeping h gives the pooled alarm fraction in healthy minutes against bearings
  detected and median lead.

## 3. Pre-registered readings

- **Physics competitive.** Smoothed physics (`ewma` or `cusum`) at the operating
  point detects at least as many bearings as the better RMS detector, with
  median lead within 20% of it. Then the physics score can be the trigger as
  well as the diagnosis.
- **RMS stays the trigger.** RMS detects more bearings, or has median lead
  greater by more than 20%. Then the architecture is: level (L0/RMS-type,
  own-baseline CUSUM) triggers, physics features diagnose (typing). The
  physics score is kept for detection only where the level is uninformative
  (variable load or speed), which XJTU-SY cannot test.
- **Comb.** If `comb` beats both at the same FA, the two signals are complementary,
  and the decision layer uses both.
- **Smoothing.** `ewma` or `cusum` on physics should reduce healthy-segment false alarms
  against `raw` at equal threshold. If it does not, the per-minute noise is not the issue.
- Short-lived bearings (≤ 114 min, calibration over 18–48% of life) are reported
  separately, because their "healthy" baseline may already contain damage.

---

## 4. Results (written after the run, 2026-10-22)

Run: `remote/box run wp7decision -- python3 scripts/analysis/wp7_decision.py`.
Output: `outputs/wp7/wp7_decision_report.md` and `outputs/wp7/wp7_decision_horizontal.png`.

Horizontal channel, at FA budget 0 (leave one bearing out):

| detector | bearings detected (of 15) | median lead | lead / life | held-out healthy alarm-minutes (of 480) |
|---|---|---|---|---|
| physics, raw | 2 | 72 min | 0.49 | 5 |
| physics, EWMA | 3 | 100 min | 0.37 | 3 |
| **physics, CUSUM** | **9** | 93 min | **0.34** | 5 |
| RMS, raw | 12 | 60 min | 0.21 | 3 |
| RMS, EWMA | 13 | 52 min | 0.21 | 1 |
| **RMS, CUSUM** | **13** | 84 min | 0.28 | 2 |
| comb (max), CUSUM | 13 | 84 min | 0.28 | 2 |

**Readings, as pre-registered:**
- **RMS stays the trigger.** Its CUSUM detects 13/15 bearings, against 9/15 for the best
  physics detector.
- `comb` is no better than RMS. After standardisation the RMS term dominates the max:
  its calibration sd is far smaller than the physics score's.
- Smoothing does help the physics score. CUSUM detects 9 bearings where raw detects 2,
  at the same held-out false alarms. So per-minute noise *was* the issue for the raw score,
  but removing it is not enough to beat the level.

What this does and does not show:
- On XJTU-SY's accelerated, constant-speed, constant-load tests, damage raises the overall
  vibration level early and cleanly. An own-baseline RMS CUSUM is the better trigger there.
  When the physics CUSUM fires, it fires earlier in relative life (0.34 vs 0.28), but on fewer bearings.
- It cannot show what happens under varying speed and load. There the level moves with the
  operating point, which is where the order-normalised physics score exists for.
  XJTU-SY cannot test that, but Ottawa 2018 (speed sweeps) and MFPT (load 0–300 lbs) can (§6).
- **False alarms remain the unsolved part.** Even at "budget 0", held-out
  healthy segments see 1–5 alarm-minutes out of 480. At one record per minute that is
  3–15 per day. With only 20 baseline minutes, nothing tighter can be calibrated or measured.
  A pilot needs days of healthy baseline per unit, not minutes (WP10 data spec).

## 5. Consequence (MASTER_PLAN §3 decision layer, revised)

- **Two roles, two signals.**
  - The **trigger** is a per-unit, own-baseline CUSUM on a level feature: RMS here, or ISO
    velocity RMS in L0.
  - The **diagnosis** is the physics score and its typing. That is the race (10/11 right on
    XJTU-SY, notes/17) and the rise against baseline per fault family.
  - Where the operating point varies, the physics score's CUSUM is also run as a trigger,
    because the level is confounded there.
- Calibration duration becomes a pilot requirement. The baseline must span enough healthy
  operation to bound false alarms per day at the rate the operator accepts.

## 6. Addendum, pre-registered before computing: the level detector on the WP5 datasets

The trigger/diagnosis split of §5 rests on the claim that a level feature is
confounded where the operating point varies. Test it with the notes/17
protocol (same folds and windows, conformal α = 0.10, cross-fitted where the
physics arm is):
- `L`: a one-sided z-score of log(window RMS) against the healthy calibration windows;
- `K`: the same on time-domain kurtosis.

Datasets:
- **Ottawa 2018**: speed 13–28 Hz within a file. Prediction: `L` is weaker than arm d, AUC lower by ≥ 0.1.
- **MFPT**: healthy at 270 lbs only, faults at 0–300 lbs. No prediction, because load
  confounds the level in both directions.
- **Ottawa 2023 own-unit**: constant speed and load. Prediction: `L` is competitive with arm d.

If `L` matches arm d on Ottawa 2018, the "physics where the level is confounded" argument
is weakened, and §5's split rests on typing alone.

### 6.1 Result (after computing)

Race faults, cross-fitted calibration, α = 0.10 (`outputs/wp5/wp5_report.md`):

| dataset | arm d AUC | `L` (log RMS) AUC | `K` (kurtosis) AUC | `K` recall / FA |
|---|---|---|---|---|
| Ottawa 2018 (speed sweeps) | 0.922 | **0.809** | **0.993** | 0.98 / 0.10 |
| Ottawa 2023, machine baseline | 0.712 | 0.906 | **1.000** | 0.92 / 0.10 (worst unit 0.95) |
| Ottawa 2023, own-unit baseline | 0.930 | 0.902 | **1.000** | 0.83 / 0.07 |
| MFPT (load 0–300 lbs) | 1.000 | 0.893 | **1.000** | 1.00 / 0.03 |

- Ottawa 2018: `L` is weaker than arm d by 0.113. The prediction (≥ 0.1) **held**:
  the level is confounded by speed.
- Ottawa 2023 own-unit: `L` is competitive (0.90 vs 0.93), as predicted.
- **`K`, healthy-referenced time-domain kurtosis, was included without a
  prediction. It is the best single detector on all three new datasets.** Kurtosis is
  scale-free, so speed and gain changes largely divide out. Their faults (seeded
  Ottawa 2018 and MFPT, naturally developed but advanced Ottawa 2023) are strongly
  impulsive.
- On Ottawa 2023 the ball faults were recorded at 0 N and their healthy files at 400 N.
  So `L`'s ball/cage "recall" (0.90–1.00) is partly load, not damage.

### 6.2 Addendum b, pre-registered before computing: `K` where faults are subtle

Kurtosis is the textbook impulsiveness indicator. It is also known to be insensitive
to small or distributed damage. Paderborn's real damage is the subtle case, and
CWRU's is the seeded one. Same protocols as notes/16 and §3 of notes/17, `K`
cross-fitted against arm d (`d_xf`):
- If `K`'s AUC ≥ arm d's on Paderborn in ≥ 3 of 4 conditions, `K` becomes the
  primary **detection** score (layer L1) and physics stays the **diagnosis** (typing).
- If `K` < arm d on Paderborn in ≥ 3 of 4, `K` is a strong-fault detector only. Then the
  decision layer uses both (max of standardised z), and arm d stays primary on
  subtle damage.
- XJTU-SY (decision layer, §2 protocol): add `kurt_cusum`. It **joins the trigger**
  if it detects ≥ RMS CUSUM's 13/15 at the same FA budget.

### 6.3 Result of addendum b (after computing)

| dataset | arm d (`d_xf`) AUC | `K` AUC | `L` AUC |
|---|---|---|---|
| CWRU | 0.929 | 0.986 | 1.000 (FA worst 1.00) |
| Paderborn N09 | **0.901** | **0.333** | 0.492 |
| Paderborn N15 ×3 | **0.84–0.86** | **0.44–0.45** | 0.79 |
| XJTU-SY trigger (of 15, FA budget 0) | 9 (CUSUM) | 11 (CUSUM) | **13** (CUSUM) |

**Reading: `K` < arm d on Paderborn in 4/4 conditions** (AUC 0.33–0.45, *below
chance*: Paderborn's real-damage windows are, if anything, less impulsive than its
healthy ones). So **`K` is a strong-fault detector only**. Arm d stays primary,
and it is the only score with AUC ≥ 0.84 on every dataset given a unit's own or same-model
baseline. `kurt_cusum` does not join the XJTU-SY trigger (11/15 vs RMS 13/15).

The picture across six datasets:
- No single statistic wins everywhere.
- Impulsiveness (kurtosis) and level (RMS) are excellent when damage is large and
  impulsive (CWRU, MFPT, Ottawa 2023, XJTU-SY end of life). They fail on subtle real damage
  (Paderborn), and the level also fails under speed sweeps (Ottawa 2018).
- The physics score is never the best by much, but it is never bad.

### 6.4 Addendum c, pre-registered before computing: one combined score

`dK` = max(u_d, u_K). Each u is the score standardised by the mean and sd of its
own cross-fitted calibration scores. The threshold is the conformal quantile of the
cross-fitted calibration `dK` scores. It is **adopted as the L1+L2 detection score**
if, on every dataset (CWRU, Paderborn ×4, Ottawa 2018, Ottawa 2023 machine and
own, MFPT), its AUC ≥ max(arm d, `K`) − 0.05 *or* ≥ arm d. That is: it must
never lose arm d's robustness, while picking up `K`'s strong-fault cases.
Otherwise arm d stays alone.

### 6.5 Result of addendum c (after computing)

**`dK` is not adopted.**
- It keeps `K`'s wins on the strongly impulsive sets:
  - Ottawa 2018: 0.995;
  - Ottawa 2023: 0.986 (machine) and 0.998 (own);
  - MFPT: 1.000;
  - CWRU: 0.978.
- But it loses 0.08–0.11 AUC against arm d on all four Paderborn conditions: 0.73–0.82 vs 0.84–0.90.
  Healthy Paderborn units differ in kurtosis, so the max lets the `K` term raise false alarms
  (held-out FA 0.17–0.22, worst unit up to 1.00).
- The adoption rule required never losing arm d's robustness. It fails there.

**Net (MASTER_PLAN §3, L1+L2):**
- The detection score stays **arm d alone** (CPW+SES physics score, own-unit baseline,
  cross-fitted calibration).
- Kurtosis and RMS are reported alongside as context ("level / impulsiveness vs own
  baseline") but are not fused into the alarm.
- The trigger for slow trends stays the RMS CUSUM where speed and load are constant (§5).
