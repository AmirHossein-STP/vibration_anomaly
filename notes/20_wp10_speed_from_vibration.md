# 20. WP10: shaft speed from the vibration itself

MASTER_PLAN.md WP10 ("speed-from-vibration estimator, validated on Ottawa and
PU against measured speed") and §0 constraint 4. Written 2026-10-22. **§1–§2 were written
before any number was computed.**

## 1. Estimator (fixed here, label-free)

Per 1 s window (12 kHz), given only a plausible speed range for the machine (its
operating envelope, which an operator always knows):
- take the Hann-windowed spectrum of the raw vibration, zero-padded 16×;
- score each candidate shaft rate f, on a 0.01 Hz grid over the range, by the
  **harmonic sum** Σ_{k=1..8} log|X(k·f)|, read by linear interpolation;
- the estimate is the arg max.

Shaft harmonics (imbalance, misalignment, looseness) are present on almost any
rotating machine. Their k-th harmonic pins f with k-fold resolution. The arg max can
lock onto a sub-multiple or a multiple ("octave error"), so these are counted separately.

## 2. Validation and the downstream test

- **Ottawa 2018.** Truth is the encoder's mean shaft rate over each window. The speed
  sweeps 13–28 Hz within a file. Range given: 10–35 Hz.
- **Paderborn.** Truth is the mean of the measured `speed` channel (v2 store) over the same
  4 s run, per window. Range given: nominal ± 20%. Nominal is also reported as an estimator,
  i.e. what the pipeline uses now.

Metrics: median |error| in %, the share of windows within 1% and within 2%, and the octave-error share.

**Downstream (Ottawa 2018).** The physics score (arm d, cross-fitted, notes/17 folds) on
1 s windows, at a constant rpm per window, taken from (i) the encoder mean and (ii) the
vibration estimate. These are compared with the order-tracked result of notes/17 (AUC 0.922).

Readings:
- The estimator is **adequate as the pilot fallback** if (ii) is within 0.05 AUC of (i).
- Computed order tracking is **worth its encoder** if (i) or (ii) is ≥ 0.05 below 0.922.

---

## 3. Results (after computing, 2026-10-22)

Run: `remote/box run wp10speed`. Table: `outputs/wp10/wp10_report.md`.

| data | estimator | median abs error | within 1% | within 2% | octave errors |
|---|---|---|---|---|---|
| Ottawa 2018 (13–28 Hz sweeps; encoder truth) | harmonic sum | **0.42%** | 0.79 | 0.87 | 0.075 |
| Paderborn N09 (speed-channel truth) | harmonic sum | 8.35% | 0.08 | 0.20 | 0 |
| Paderborn N15 ×3 | harmonic sum | 2.5–2.9% | 0.14–0.17 | 0.34–0.39 | 0 |
| Paderborn, all | nominal setpoint | 0.03–0.04% | 1.00 | 1.00 | 0 |

Downstream on Ottawa 2018 (physics score, 1 s windows, one rpm per window):

| speed source | race AUC | recall | FA |
|---|---|---|---|
| encoder, mean per window | 0.804 | 0.655 | 0.088 |
| vibration estimate per window | 0.661 | 0.479 | 0.092 |
| encoder + computed order tracking (notes/17) | **0.922** | 0.757 | 0.097 |

**Readings:**
- **The estimator is not adequate as a fallback** (−0.14 AUC against encoder speed).
- **Order tracking is worth its encoder**: +0.12 over a constant rpm per window, even with the true mean speed.

What this means:
- On a machine whose vibration carries clear shaft harmonics (Ottawa 2018's rig), the
  speed can be read to well under 1% most of the time. But the 7.5% octave locks and the
  tail are enough to cost 0.14 AUC, because a wrong speed puts every fault line in the wrong place.
- On Paderborn's drive the shaft harmonics are not dominant, and the estimate is poor.
  There the nominal setpoint is exact anyway (a speed-controlled drive).
- Under varying speed, even a perfect *mean* speed per window loses 0.12 AUC
  against order tracking.
- **Pilot requirement, strengthened (MASTER_PLAN WP10 data spec):** a synchronous speed signal
  (keyphasor or encoder) is required, not optional. For speed-controlled machines, the setpoint
  logged with every record is an acceptable substitute. Speed from vibration stays a
  last-resort fallback. It needs continuity tracking (no octave jumps between adjacent windows)
  and a fault-line-free harmonic source before it can be trusted. Open.
