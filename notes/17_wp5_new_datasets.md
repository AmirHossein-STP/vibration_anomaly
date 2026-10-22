# 17. WP5: the frozen physics score on datasets that played no part

MASTER_PLAN.md WP5 (and the XJTU-SY half of WP7). Started 2026-10-22, on the
rented box (`remote/box`; owner instruction of that day: run the plan without
asking). **§1–§4 were written before any score on these datasets was computed.**
§5 onward was written after.

## 1. Why this run exists

notes/16 left one primary detector candidate, **arm d**: a healthy-referenced
physics score on CPW+SES features. It needs no source data and no classifier.
Paderborn is spent for that choice. This run is its confirmation on four
datasets nobody has looked at in this project: **Ottawa 2018, Ottawa 2023,
MFPT, XJTU-SY**. CWRU is reported as well. Its labels were used for the arm-a
classifier, never for arm d. IMS stays sealed (WP11).

## 2. The frozen method (no free parameter is chosen in this note)

Exactly notes/16 arm d, and exactly `envelope_cache.py` variant `hd_cpw_ses`:
- resample to 12 kHz (`resample_to_rate`; where the ratio is not exact, the
  effective rate is used);
- 1 s windows (12000 samples), hop 0.5 s;
- cepstrum pre-whitening, no band, squared envelope, Hann window, |rFFT|;
- features read at **each dataset's own bearing geometry**: six peaks,
  BPFO/BPFI × harmonics 1–3, ±0.08 orders, each divided by the envelope sum
  over 0.5–12 orders;
- score = max over the six of the one-sided z-score of log(peak + 1e-9)
  against the healthy calibration windows' mean and sd;
- threshold = conformal quantile at α = 0.10 of the calibration scores
  (`wp1_rescore.conformal_threshold`);
- typing = the race whose three peaks hold the largest z.

Also reported, for comparison:
- **arm d on v1** features;
- **arm a**: the WP1 model. That is v1 features, a StandardScaler plus logistic regression fit on
  CWRU (first 4 s of every file), with its threshold calibrated on the target's healthy windows.
  For the new datasets this is a genuine leave-one-dataset-out test of the CWRU-trained classifier.

**The one necessary extension.** Ottawa 2018 runs at time-varying speed (13–28 Hz
within one 10 s file, measured from its 1024-CPR encoder). There the envelope
is computed in time exactly as above. It is then **resampled to uniform shaft
angle**, using the encoder phase (computed order tracking, Shen et al. 2021; MASTER_PLAN
WP5 names Ottawa as the test bed for this). The spectrum is taken over windows
of a fixed **20 revolutions**, hop 10. That is ≈1 s at the median speed. Nothing else
changes. Correctness check before use: on a constant-speed signal the
angle-domain features must reproduce the time-domain ones.

## 3. Datasets and the protocol on each

| dataset | geometry (orders) | healthy baseline (calibration → held-out) | faults scored |
|---|---|---|---|
| CWRU (12k DE) | 6205: BPFO 3.585, BPFI 5.415 | Normal at 3 loads → Normal at the 4th (4 folds); faults of the held-out load | inner, outer, ball (EDM) |
| Ottawa 2018 | ER16K: 9 balls, 0.3125/1.516 in → BPFO 3.572, BPFI 5.428 | 11 healthy files → the 12th (12 folds) | inner, outer, ball, combined |
| Ottawa 2023 | 6203-size: 8 balls, 6.77/28.5 mm → BPFO 3.050, BPFI 4.950 | (i) **machine**: the other 19 bearings' healthy files → this bearing's healthy file (20 folds); (ii) **own unit**: first 5 s of this bearing's healthy file → its last 5 s | this bearing's developing (`_1`) and faulty (`_2`) files: inner, outer, ball, cage; naturally developed |
| MFPT | 8 rollers, 0.235/1.245 in, 25 Hz → BPFO 3.245, BPFI 4.755 | 2 baseline files → the 3rd (3 folds) | outer (270 lbs and 25–300 lbs), inner (0–300 lbs) |
| XJTU-SY | LDK UER204: 8 balls, 7.92/34.55 mm → BPFO 3.083, BPFI 4.917 | own early life: first 20 minutes (one 1 s window per minute; horizontal channel primary) | run-to-failure, §4 |

Geometry and speed come from each dataset's own documentation:
- Ottawa 2023: Sehri et al. 2023, Table 2. Speed is the per-file value in column 3.
- MFPT: the shipped BPFO/BPFI values (81.125/118.875 Hz at 25 Hz shaft speed).
- XJTU-SY: the dataset tutorial (Lei et al.), Table 1. Its BPFO at 35 Hz is 107.91 Hz.

Ball, cage and combined faults are **not covered by the six race peaks**. They are reported per type,
but they are **not part of the gate**: they are the coverage gap WP3 addresses (BSF/FTF families).

Metrics per dataset: recall at α = 0.10; mean and worst held-out FA; held-out AUC
(held-out healthy vs faulty windows); recording-level detection at the
service's `faulty` level (≥ 50% of windows over threshold); typing accuracy on
inner and outer faults.

## 4. Pre-registered readings

**Gate (WP5 confirmation of arm d on CPW+SES).** Over the three new labelled
datasets with race faults, Ottawa 2018, Ottawa 2023 (machine baseline) and MFPT,
counting inner and outer race faults only:
- **confirmed** if AUC ≥ 0.80 and mean held-out FA ≤ 0.15 on at least 2 of the
  3, and arm d's AUC ≥ arm a's on at least 2 of the 3;
- **not confirmed** if AUC < 0.70 on 2 or more of them. Arm d then stays a
  candidate, and the multi-arm comparison (WP6) decides;
- otherwise **partial**. The dataset that fails is named, and the failure is
  explained or left open.

Other readings:
- **Own unit vs machine baseline (Ottawa 2023).** This is MASTER_PLAN §3's claim that a
  unit's own healthy data removes the unit-to-unit spread. It is supported if worst-unit FA
  drops and recall falls by at most 0.05 at α = 0.10.
- **Typing** on the new datasets passes if correct-race accuracy is > 0.6 on each.
- **XJTU-SY (WP7 form).** The detector is arm d, own baseline: the first 20 minutes, conformal at
  α = 0.05 (the maximum calibration score). An alarm is **3 consecutive minutes above the
  threshold**. Reported per bearing:
  - first-alarm time and lead time to end of life;
  - the alarm rate in minutes 21–40, assumed healthy and stated as an assumption, never as a measurement;
  - the same quantities for the classical baseline, RMS above mean + 3 sd of the same 20 minutes
    for 3 consecutive minutes;
  - typing at the end of life against the documented failed element.

  It **supports** the method if, on the 13 bearings whose failed element includes a race:
  - an alarm precedes end of life on ≥ 10;
  - the alarm comes no later than the RMS baseline's on the majority;
  - typing in the last 10% of life is correct on ≥ 2/3 of the 11 single-race failures.
- The CWRU-trained classifier (arm a) is expected to be **worse** than arm d
  wherever the target rig differs strongly (notes/12–13). If it is better on a
  dataset, that is reported prominently.

Nothing in this note is tuned on these datasets' labels. If a bug is found
after scoring, the fix and the before/after numbers are both recorded here.

---

## 5. Results (written after scoring, 2026-10-22)

Runs: `remote/box run wp5cache | wp5score | wp7xjtu`. Run records (git
sha, diff, environment, log) are in `outputs/remote_runs/`. Tables:
`outputs/wp5/wp5_report.md` and `outputs/wp7/wp7_xjtu_report.md`. Figures:
`outputs/wp7/xjtu_{horizontal,vertical}.png`. Feature extraction for all four
datasets took under a minute on the box.

**Scorer check.** The Paderborn rows of `wp5_score.py` reproduce notes/16
exactly, for example N09 CPW+SES arm d: recall 0.746, FA 0.112, AUC 0.901.

### 5.1 Race faults, conformal α = 0.10 (mean over folds, worst fold in brackets)

| dataset | arm a (CWRU classifier) AUC | arm d AUC | arm d recall | arm d FA | arm d typing |
|---|---|---|---|---|---|
| CWRU (seen) | — | 0.929 (0.90) | 0.874 | 0.111 (0.14) | 0.872 |
| Ottawa 2018 (order-tracked) | **0.481** | **0.922** (0.88) | 0.767 | 0.106 (0.30) | 0.630 |
| Ottawa 2023, machine baseline | 0.716 | 0.712 (0.12) | 0.579 | 0.134 (**1.00**) | 0.747 |
| Ottawa 2023, own-unit baseline | 0.721 | **0.930** (0.69) | 0.876 | 0.300 (0.78) | 0.774 |
| MFPT | 0.835 | **1.000** (1.00) | 1.000 | 0.182 (0.27) | 1.000 |
| Paderborn N09 / N15 ×3 (spent) | 0.61–0.89 | 0.84–0.90 | 0.66–0.75 | 0.11–0.14 | 0.77–0.81 |

**Gate as pre-registered: PARTIAL.**
- Ottawa 2018 passes outright: AUC 0.92, FA 0.106.
- MFPT separates perfectly (AUC 1.00, recall 1.00), but FA is 0.182 > 0.15.
- Ottawa 2023 against the machine baseline does not pass: AUC 0.71, and one healthy bearing alarms on every window.
- Arm d beats arm a on Ottawa 2018 and MFPT. On Ottawa 2023 machine they tie (0.712 vs 0.716).
- The CWRU-trained classifier is at chance on Ottawa 2018 (AUC 0.48), the third rig on which it fails.

### 5.2 Two things the gate exposed, and what they mean

**(1) Small calibration sets: a bias in the method, found post hoc and fixed.**
The frozen scorer fits the baseline mean and sd on the calibration windows and
then scores those same windows. In-sample scores run low, so the conformal
threshold is too low and held-out FA overshoots α. The bias shrinks with calibration size:
- negligible on Paderborn (thousands of windows): every number moves by ≤ 0.005;
- large with MFPT's 22 windows and Ottawa 2023's 9 own-unit windows.

The fix, `d_xf`, is not tuned on labels. Each calibration recording (or contiguous third of a single
recording) is scored by a baseline fit without it, and the threshold is set on those
out-of-sample scores. AUC is unchanged by construction. With it:

| | FA before → after | recall before → after |
|---|---|---|
| MFPT | 0.182 → **0.000** | 1.000 → 0.997 |
| Ottawa 2023 own-unit | 0.300 → **0.056** (worst 0.78 → 0.22) | 0.876 → 0.634 |
| Ottawa 2018 | 0.106 → 0.097 | 0.767 → 0.757 |
| CWRU, Paderborn | unchanged ± 0.004 | unchanged ± 0.005 |

With `d_xf`, Ottawa 2018 and MFPT both pass the gate's AUC/FA criteria. This is
reported as post hoc, not as the gate result. **Consequence for WP9:** the
service's `calibrate` must use cross-fitted calibration scores.

**(2) A fleet baseline is not enough; the unit's own baseline is (MASTER_PLAN §3).**
Ottawa 2023 is the only dataset with each bearing recorded healthy and then faulty.
Pooling the other 19 bearings' healthy data as the baseline gives AUC 0.71, and
one bearing alarms on all of its own healthy data. That is the same per-unit failure as
Paderborn's K005 (notes/14) at a larger scale: Ottawa 2023 used two bearing
models and re-mounted bearings between tests. Calibrating each bearing on the first
5 s of its *own* healthy file raises AUC to **0.93** (worst bearing 0.69).
With `d_xf` it holds FA at 0.056 (worst 0.22) and keeps recall at 0.63. The pre-registered own-vs-machine reading is
**supported** (worst-unit FA 0.78 vs 1.00, recall +0.30). This is the strongest
evidence yet for the deployment design: physics features, each machine
referenced to its own healthy history.

### 5.3 Coverage gaps (outside the gate, as pre-registered)

Ball and cage faults are mostly missed by the six race peaks:
- Ottawa 2018 ball: 0.13;
- Ottawa 2023 ball: 0.02–0.18, cage: 0.07 against the machine baseline.

One exception: cage 0.71 against the own-unit baseline, which is probably broadband change rather than a cage line.
This is WP3's job (BSF/FTF families).

On Ottawa 2018 the outer race reads at 0.54 while the inner reads at 1.00. The
order-tracked outer-race peak is weak on that rig. That is worth a WP3 look at higher
harmonics and sidebands.

### 5.4 XJTU-SY run to failure (WP7 form)

Horizontal channel (primary), own baseline minutes 1–20:
- An alarm precedes end of life on **11/13** race-involving bearings.
- **Typing at end of life is right on 10/11** single-race failures. The one miss is Bearing2_1, inner read as outer, with an inner share of 0.40.
- The alarm comes **no later than the RMS 3σ baseline on only 2/13**.

**Pre-registered reading: does not support** (the timing criterion fails). The
vertical channel meets all three criteria (12/13, 7/13, 9/11). The primary
channel was fixed in advance, so the verdict stays "does not support".

Why the timing loses (`outputs/wp7/xjtu_horizontal.png`):
- The per-minute physics score is one 1 s window, a max of six z-scores, against a 20-minute
  baseline. It is far noisier minute to minute than RMS.
- Spurious 3-in-a-row runs happen in the "healthy" minutes 21–40 (rates up to 0.35).
- Its real rises mostly coincide with the RMS rise. These are accelerated tests with large
  outer-race damage, where the overall level is a strong detector.
- The RMS rule is not clean either. Its 21–40 alarm rate reaches 0.45–0.55 on the short-lived
  bearings, and it alarms on Bearing3_1 at minute 26 of 2538.

"Earliest alarm" without a matched false-alarm rate does not decide between the
two detectors. That is exactly WP7's point: the operating point must be chosen
by an FA budget (lead time vs false alarms per day). So the next step is the
decision layer (§6), not a different feature.

What XJTU-SY does establish:
- **physics typing works on real, naturally developed run-to-failure damage**: 10/11;
- the score does rise with damage on the bearings that fail by a race;
- cage failures (1_4, 2_3) and the mixed 1_5 are not read as race faults
  until late, if at all.

## 6. Consequences (MASTER_PLAN revised the same day)

1. **Primary detector stays arm d on CPW+SES, now with cross-fitted calibration
   and a per-unit own baseline.** It transferred to three rigs nobody tuned on:
   - Ottawa 2018 (with order tracking): AUC 0.92;
   - MFPT: AUC 1.00;
   - Ottawa 2023 own-unit: AUC 0.93.

   The CWRU-trained classifier does not transfer (AUC 0.48 / 0.72 / 0.84).
2. **WP7 next (decision layer).** Score smoothing or CUSUM over minutes, and lead time vs
   false alarms per day on XJTU-SY with an FA-budgeted operating point.
   Pre-registered in notes/18 before it is run.
3. **WP3 (feature v2)**: BSF/FTF families for ball/cage, and harmonics/sidebands for
   the weak outer-race line on Ottawa 2018.
4. **WP9**: cross-fitted calibration; per-unit baselines as the default.
