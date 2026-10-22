# 22. Method v1.0: the bearing monitor, specified

The one specification of the method (MASTER_PLAN Phase 2, WP12). Implementation:
`src/domain_adaptation/method.py` (`METHOD_VERSION = "1.0"`). The analysis scripts
(`wp5_score.py`, `wp7_decision.py`) and, from WP17, the `bearing` service call it.
Each line below is backed by notes/16–21.

**Status:** specified 2026-10-22. **Not frozen yet**: freezing happens at WP18, with the commit recorded here.

## 1. Inputs per unit
- Vibration waveforms from one accelerometer near the bearing, at any rate ≥ 12 kHz. Everything is
  resampled to **12 kHz** (`to_12k`; the effective rate is used where the ratio is not exact).
- **Shaft speed is required**, in one of these forms:
  - an rpm per recording (sidecar, or the logged setpoint of a speed-controlled drive), used
    for constant speed within a recording;
  - **a tachometer / encoder channel**, giving the shaft phase, for speed that varies within a
    recording. The envelope is then order-tracked (`features/cot.py`; 20-revolution windows, hop 10).

  Without either, the unit is refused. Speed estimated from vibration is not adequate (notes/20).
- **Bearing geometry**: a designation from the catalog, or explicit (n, d, D, α), or the
  manufacturer's fault factors. Without geometry, see §6.
- **Healthy data from the same unit** (its own baseline; notes/17 §5.2: fleet baselines fail).

## 2. Features (per window)
- Windows of 1 s (12000 samples), hop 0.5 s.
- Cepstrum pre-whitening of the window, then the squared Hilbert envelope (no band
  choice), DC removed, Hann window, |rFFT|. Read in orders of shaft speed.
- Six lines: **BPFO and BPFI × 1, 2, 3**, each the peak within ±0.08 orders, divided by the
  envelope-spectrum sum over 0.5–12 orders.
- (`envelope.envelope_order_features` with `whiten=True, band_hz=None, squared=True`; frozen source.)

## 3. Baseline and score
- `fit_baseline`: mean and sd of log(line + 1e-9) over the unit's healthy windows.
- Window score = max over the six lines of (log line − mean) / sd. Only a rise counts.
- **Cross-fitted calibration scores:** each healthy recording is scored by a baseline fit
  without it. Inside a single recording, its three contiguous thirds are used instead.
- Threshold = conformal quantile at **α = 0.10** of those scores (`conformal_threshold`).
  The minimum is 9 calibration windows, else the threshold is +∞ and the unit is refused.
- **Typing:** the race whose lines have the larger max z, inner vs outer.

## 4. Verdicts
- Recording verdict, from the share of windows over threshold (notes/14):
  - `watch` ≥ 0.40;
  - `faulty` ≥ 0.50.
- **Trend trigger:** one-sided CUSUM, k = 0.5, on the per-recording standardised level
  (RMS, z against the baseline recordings), and on the physics score standardised by its
  cross-fitted calibration scores. The threshold is set by an FA budget on healthy data only
  (notes/18). At constant speed and load the level CUSUM is the primary trigger. The rule for
  varying conditions comes from WP15.
- The explanation names the dominant family (inner/outer), its z, and which lines rose.

## 5. Not covered (documented gaps)
- Ball and cage faults (notes/17 §5.3, notes/19).
- Machines without a speed signal.
- A brand-new unit with no healthy history: a fleet baseline is not adequate (AUC 0.71 on Ottawa 2023).

## 6. Unknown geometry
To be decided by WP13 (notes/23): the geometry-free fallback, or level/kurtosis with a warning.

## 7. Parity (WP12 gate, 2026-10-22): PASS
`scripts/analysis/wp12_parity.py` re-ran `wp5_score.py` and `wp7_decision.py` through
`method.py` on the box. It compared **10,749** numbers with the committed results: **0 differ.**

| headline | value |
|---|---|
| Ottawa 2018 AUC (cross-fitted) | 0.922 |
| Ottawa 2023, own-unit baseline | 0.930 |
| MFPT | 1.000 |
| Paderborn N09 | 0.901 |
| XJTU-SY RMS CUSUM / physics CUSUM (detected of 15) | 13 / 9 |

## 8. Freeze record
(empty until WP18)
