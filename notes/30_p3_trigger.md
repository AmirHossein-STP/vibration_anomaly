# 30. Phase 3 Track C: the trend trigger as a sequential test

**Pre-registration.** Written 2026-10-22, before any trigger in this note was computed. Track A
(notes/29) was running at the time; nothing from it is used here.

## 1. The problem (notes/27 §5.3)
On IMS the v1.1 trigger gave 1.02 false events/day on the 8 bearings that did not fail. It fired
2–84 h after the baseline on *every* unit, because the level drifts away from a fixed 33 h baseline.
The window-level detection score carried the signal; the trigger did not.

Track C keeps the **statistic** fixed (v1.1's level: log RMS per record against the unit's own
baseline; on LDM the operating-point-conditioned residual, notes/22 §7a) and changes only the
**trigger**. That separates "what is measured" from "when to alarm".

## 2. Data and conventions (all development; IMS is development since notes/28)

| dataset | units | baseline B (records) | false-alarm region | precedent |
|---|---|---|---|---|
| ims | 12 (4 failed) | 200 | failed units: after B to the 50% point; the 8 others: after B to the 90% point | notes/27 |
| xjtu_sy (horizontal) | 15 | 20 (min) | minutes 21–60, units with life ≥ 120 min | notes/18 |
| paderborn_ldm (channel A, L_cond) | 17 | 150 | records 151–450, units with ≥ 1000 records | notes/25 |
| femto (horizontal) | 17 | 180 (30 min) | the 720 records after B, units with ≥ B + 820 records | notes/26 |

- **False events per day:** alarms inside the false-alarm regions divided by those regions' total
  duration. The trigger restarts after every alarm.
- **Detection:** the first alarm after the false-alarm region (after B where the unit has none)
  and before the end. Counted on failed units only. The lead is the end time minus the alarm
  time, in hours.

## 3. Triggers (fixed now)
`z_t` = the record's statistic standardised by the baseline mean and sd.
- **T0, v1.1:** CUSUM k = 0.5 on z_t; h from FA budget 0, leaving one unit out within the
  dataset (the notes/18, notes/25 rule).
- **T1, conformal CUSUM martingale, fixed reference.**
  - p-value: smoothed conformal p-value of s_t against the B baseline records (high = anomalous).
  - Betting function: the mixture of power functions ε p^(ε−1), ε ∈ {0.05, 0.1, 0.2, 0.3, 0.5}.
  - Statistic: log form L_t = max(0, L_{t−1}) + log b(p_t).
  - Alarm at L_t ≥ log 100. The threshold comes from **Ville's inequality**: a bound that holds
    under exchangeability, so it is not tuned on any data.
- **T2, the same with a sliding, lagged reference.**
  - Reference at t: the B records ending B records before t (the baseline itself until t > 2B).
  - Drift slower than about B records is absorbed; a change faster than that is not.
- **T3, forecast residual.**
  - r_t = z_t − EWMA_{t−1}(z), λ = 2/(B+1), standardised by the sd of the baseline's own residuals.
  - CUSUM k = 0.5, with h set as in T0.
- **T4, fleet-relative (IMS only, descriptive).**
  - z_t of the unit minus the median z_t of the other bearings in the same test at the same record.
  - CUSUM k = 0.5, with h set as in T0 over IMS units.
  - It cannot be confirmed on the exam sets, which have no simultaneous siblings, so it cannot
    pass the gate.

## 4. Gate and readings
A trigger among T1–T3 **passes** if:
1. **IMS, the 8 bearings that did not fail:** ≤ **0.2** false events/day;
2. **detections:** failed units detected across all four sets (4 + 15 + 17 + 17 = 53) ≥ T0's count.

| outcome | meaning |
|---|---|
| T1 passes | The fixed-baseline drift was a threshold problem, and Ville's bound fixes it. It becomes the v2.0 trigger candidate |
| T2 or T3 passes and T1 does not | The cure is to forget slowly: a reference that moves with the unit. The candidate is the simpler of the two that pass. **Caveat stated now:** it also absorbs degradation slower than its lag, and a real machine's degradation timescale is not known from accelerated tests |
| none passes | There is no trigger on the level that is both quiet and sensitive on these sets. The trend layer stays out of the product: alarms come from the recording verdict (detection score) only, and the trade-off curve is reported |

Confirmation of any candidate: the sealed Ferrara and KAIST sets (notes/28), once.

## 5. Results
(to be filled by `scripts/analysis/p3_trigger.py`)
