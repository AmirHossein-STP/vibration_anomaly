# 21. WP6 (part): a per-machine sensitivity estimate from synthetic faults

MASTER_PLAN.md §3 ("L3's second payoff") and WP6's "also produce the per-machine
sensitivity curve, and check whether it predicts which real faults were caught."
Written 2026-10-22. **§1–§2 were written before any number was computed.** The L3
*training* arms (A–E) are not run here. This is the label-free sensitivity estimate only.

## 1. Generator (fixed; never tuned on real faults)

Added to a real healthy 1 s window x of the unit, at 12 kHz:
- an **impulse train** at the fault order × shaft rate, with inter-impulse intervals
  jittered by 1.5% (sd);
- each impulse is a **decaying sinusoid at the unit's own dominant resonance**: the
  frequency is the arg max of the Welch PSD of its calibration healthy windows over 1–5.5 kHz,
  with damping ratio 0.03;
- **inner race** only: impulse amplitudes modulated by the load zone, 0.5 + 0.5·cos(2π f_shaft t);
- peak amplitude = a × RMS(x), for a ∈ {0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2}.

## 2. Protocol and readings

Detector: arm d exactly as deployed per notes/17 (CPW+SES, own-unit baseline,
cross-fitted, conformal α = 0.10).

- **Ottawa 2023 (primary).** Per bearing: calibrate on the first 5 s of its healthy file.
  Inject into its held-out last 5 s windows. That gives the detection probability per amplitude
  per race, and **a50**, the amplitude at 50% detection (log-interpolated; > 2 if never reached).
- **Paderborn** (descriptive). Per condition, machine baseline (5 calibration units, the 6th held out),
  same injection.

Readings:
- **The curve predicts real detection** if, across Ottawa 2023's 10 race-fault
  bearings (inner 1–5, outer 6–10), Spearman ρ between the bearing's a50 for its
  own fault type and its real race recall (notes/17, `d_xf`, own baseline) is ≤ −0.5.
  Then it can ship as a per-machine sensitivity estimate (WP9).
- If ρ > −0.5, it is **not shipped as a sensitivity estimate** (MASTER_PLAN §7). It can still be
  shown as a relative comparison of machines or sensor placements, labelled as generator-dependent.
- Descriptive: whether the conditions ordered by a50 on Paderborn match their
  real-recall order (N09 best in notes/16).

---

## 3. Results (after computing, 2026-10-22)

Run: `remote/box run wp6sens` (~2 min). Table: `outputs/wp6/wp6_report.md`.

**Reading: does NOT predict.**
- Spearman ρ between a bearing's a50 for its own fault type and its real race recall is
  **+0.20** (p = 0.58, n = 10). The criterion was ≤ −0.5.
- So the curve is **not shipped as a per-machine sensitivity estimate** (MASTER_PLAN §7).

What the curves do show:
- On Ottawa 2023 the median detection probability rises from ≈ 0.05 to ≈ 1 between impulse
  peaks of 0.5× and 1.5× the window RMS. Outer reaches 50% at about 0.6×, inner at about 1.1×:
  the load-zone modulation halves the mean impulse energy.
- On Paderborn's machine baseline the synthetic inner fault is never detected within the sweep,
  and the outer fault needs 1.7–3× RMS. Paderborn's real recall (0.66–0.74) is nonetheless
  similar to Ottawa 2023's. The generator's impulses are evidently not what real damage on
  either rig looks like in this feature space.
- One bearing shows why the curve can mislead. Ottawa 2023 b02 "detects" at 0.05× RMS, because
  its held-out healthy half already crosses the threshold: the floor of the curve is that
  unit's false-alarm rate, not its sensitivity. Yet its real recall is 0.00 (its later files
  score *lower* than its own baseline).
- A per-unit curve therefore needs at least:
  - the no-injection rate subtracted (detection lift, not raw detection);
  - a generator validated against real damage signatures, which is WP6's main task (Wang, Taal, Fink 2022).

Neither fix is applied after the fact here.

**Consequence:** the L3 sensitivity estimate stays out of WP9. The L3 training arms (WP6 A–E) are
still open. This result lowers the prior that a fixed, untuned impulse generator stands in for
real damage.
