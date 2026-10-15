# Master plan: fault detection on machines we have only healthy data for

Written 2026-10-15. **This is the single forward plan for the repository.** It
replaces, as forward plans, `PLAN.md`, `HYDRAULIC_PLAN.md`,
`DOMAIN_ADAPTATION_PLAN.md`, `GPU_SERVER_PLAN.md` and `DEPLOYMENT_PLAN.md`.
Those files stay as the historical record of how we got here. See §9 for which
file holds what.

It folds in:
- every result from the three experiments;
- the turbomachinery roadmap (`reports/turbomachinery_roadmap.html`, layers
  L0–L5);
- the deployment work (`src/serving_core/`);
- three papers supplied for this revision (`papers/`), plus three more found
  while writing it (§2).

Decisions recorded with the owner on 2026-10-15:

| Question | Decision |
|---|---|
| Real turbomachine data available? | **None yet**, neither recordings nor inventory. The pilot is gated on data collection (WP10). |
| More public datasets? | **Yes**: XJTU-SY and Ottawa for development. **IMS is sealed** as the final exam (WP11). |
| Hydraulic (ZeMA) subproject | **Frozen** as the reference template for process-sensor fusion. No new research until real pump process data exists. |
| Where does computation run? | **Not on the local Mac.** Every heavy step asks the owner first (§5). |

---

## 0. The question

> We have turbomachines with raw vibration, speed and load signals, geometry
> specifications, and several identical units. We have **no recording of any
> of them faulty**, and today we have no data from them at all. Public lab
> datasets have faults, but on other machines and other sensors. Can we detect,
> and if possible localise, faults on our machines?

Four constraints shape every choice below:

1. **No target fault labels, ever.** So recall can never be measured on the
   real machine, and nothing may be tuned against target faults. A selection
   criterion built on target healthy data alone rewards the "call everything
   healthy" failure.
2. **Fault knowledge must come from physics and from public data.** It cannot
   come from the target.
3. **Bearing types are mixed or unknown.** The working method (§1) applies only
   to rolling-element bearings with known geometry, so it needs an
   unknown-geometry fallback and companions for other components.
4. **The speed signal is necessary.** Every fault frequency is a multiple of
   shaft speed. Where there is no tachometer, speed must be estimated from the
   vibration itself.

---

## 1. Evidence ledger

What the three experiments actually established. Status is one of
**established** (replicated, large effect), **indicative** (single run or
under-powered), or **retracted**. Sources are in the last column.

### Established

| Claim | Evidence | Source |
|---|---|---|
| Healthy-only autoencoder detects faults **in-domain** on CWRU | ROC-AUC 1.0. CWRU is an easy lab benchmark, so this sets no expectation for real machines | notes/07, notes/08 |
| Spectrogram CNNs do **not** transfer across datasets | Strict CWRU→PU: collapsed to one class in **11/11** runs (2 torch versions). PU→CWRU: balanced accuracy **0.499 ± 0.002** (n=5). Per-window normalisation: 0.499–0.505 | notes/12 |
| The cause is the **representation**: log-spectrograms are dominated by rig-specific structural resonances | The physics argument, plus the next row | notes/13 |
| **Envelope-order features + logistic regression, fit on CWRU only, calibrated on target healthy windows only, transfer** | PU N09 @10% target FA: recall **0.746 ± 0.012**, balanced **0.823 ± 0.007** (10 calibration splits). Unchanged at 2.5× the windows (0.821 ± 0.009). Uncalibrated: 0.654 | notes/13 |
| Errors are **per unit**, not random | False alarms concentrate on one healthy bearing: K002 under the spectrogram model, K005 under the envelope model | notes/13, GPU_SERVER_PLAN §2 |
| **Seed variance swamps single-run claims** | 6 identical-config runs: target recall 0.42–0.97 | GPU_SERVER_PLAN §1b |
| Thresholds must be set by the **false-alarm rate** you accept | Hydraulic: "best F1" percentiles carried 40–75% FA. At FPR ≤ 10%: valve F1 0.705, accumulator 0.683, pump 0.385 | HYDRAULIC_PLAN v6 |
| Deployment plumbing works | serving_core `bearing` service on real PU data (900 rpm): `faulty` on 4% of healthy and 76% of real-damage recordings. Calibrate flagged K005. Same numbers on host and in Docker | DEPLOYMENT_PLAN §8 |

### Indicative or open

| Claim | Why it is not established | Source |
|---|---|---|
| The method's performance at 1500 rpm | PU N15: balanced **0.685 ± 0.005**, recall@10% 0.469. **Unexplained**, and it runs against the obvious prediction (N15 is closer to CWRU's speed) | notes/13 |
| Fault typing transfers | 3-class 0.593 against a 0.353 majority baseline. Correct race on faulty windows: **60.8%** (chance 50%). Outer race is mostly read as inner | notes/13 |
| The bearing service's verdict levels (`watch` 25%, `faulty` 50% of windows) | Engineering defaults. The study only measured window-level rates | DEPLOYMENT_PLAN §8 |
| Hydraulic cooler 1.0 | `cooler_condition` correlates with campaign time (r = 0.88) | HYDRAULIC_PLAN |
| Hydraulic pump detector | ROC-AUC 0.656. Shipped opt-in only | HYDRAULIC_PLAN |
| Real-data behaviour of the `vibration` and `hydraulic` services | Verified on synthetic data only | DEPLOYMENT_PLAN §8 |

### Retracted

| Claim | Why |
|---|---|
| Arm A table (strict CWRU→PU spectrogram, recall 0.812) | Stale window cache. Did not reproduce. notes/12 |
| "Holding Paderborn out of training cost nothing" | Same artifact |
| CWRU in-domain envelope accuracy 0.971 ± 0.032 | Windows of one file appeared in both train and test folds. Split by file: **0.88 ± 0.20** |
| "DANN is harmful / helps" as a tuned verdict | Moot: alignment over a representation that does not transfer. Closed, not settled (§6) |

**The one-line summary:** what crosses machines is *physics computed from
bearing geometry and speed, calibrated on the target's own healthy data*.
What does not cross machines is anything learned end to end from how a
machine sounds.

---

## 2. What the literature adds, and what does not fit our setting

### The general point
Published cross-domain fault-diagnosis papers almost always assume an
**unlabeled target pool that contains faults**, often class-balanced, and
sometimes a few labeled target samples. Our target is **healthy-only**.

Every method that aligns the *whole* target distribution to the source is
therefore not applicable as published: CORAL, MMD, adversarial discriminators,
pseudo-labelling. With healthy-only target data they align healthy target
windows to a source mixture of healthy and faulty windows. That is exactly
wrong.

We take these papers' **representations, diagnostics and protocols**, not
their adaptation step.

### Xie et al., *Physics-informed cross-domain fault diagnosis for rolling bearings…* (Eksploatacja i Niezawodność 29(1), 2027)
Method: CWRU → in-service high-speed-train axlebox (and a fan rig, 600 → 1000
rpm). 187 physics features (time, spectral, envelope; BPFO/BPFI/BSF/FTF with 5
harmonics and 3 shaft-rate sidebands; ±0.1-order bands). Sparse-group lasso
with stability selection, a random-forest benchmark, order normalisation, Deep
CORAL plus pseudo-label self-training, then SHAP and a drift report.

- **Adopt:**
  - The richer feature families: harmonics to M=5, sidebands to Q=3, BSF and
    FTF families, alongside our 13 features (WP3).
  - **Analysis windows defined in shaft revolutions** (they use 20), so order
    resolution is identical at every speed (WP2, H-b).
  - **Unknown-geometry mode:** when the target bearing's geometry is
    unavailable, read broad fault-order bands instead of imposing another
    bearing's exact orders. Directly relevant to our mixed or unknown fleet
    (WP3).
  - **Stability selection** for feature robustness, but run across *source
    datasets*, never against the target (WP3).
  - An **ex-ante drift report**: standardised mean difference per feature,
    source vs target. We compute it on healthy data only, which makes it
    label-free and a calibration-time diagnostic (§4, WP9).
  - **AWGN robustness test**: inject noise at 20 … −5 dB into target signals,
    re-score with no retraining (WP3).
  - Permutation importance and SHAP for operator-facing explanations (WP9).
- **Reject:**
  - Deep CORAL and pseudo-labelling. Both need faulty target windows, and they
    standardise each domain with its own statistics, which includes target
    faults.
  - The 98% headline as a benchmark. The target is 16 files (256 windows,
    balanced, used transductively), so it is effectively n=16 recordings.

### Shen et al., *A physics-informed deep learning approach for bearing fault detection* (Eng. Appl. of AI, 2021)
Method: field data from 18 roller bearings on an agricultural machine at
time-varying speed (21–54 Hz). Computed order tracking, envelope order
spectrum, **sub-bands of ±5% around each fault order and its first two
harmonics** as CNN input, and a loss penalty when the CNN contradicts a simple
physics threshold model.

- **Adopt:**
  - **Relative tolerance** (±n% of the order) instead of our absolute ±0.08
    orders. Slip and speed error scale with the harmonic number (WP2, H-c).
  - The **sub-band envelope order spectrum** as a richer input vector, for the
    roadmap's arm D and for WP8.
  - **Computed order tracking** for varying speed, needed for real
    turbomachines and for the Ottawa dataset (WP5).
  - A **physics-consistency rate** as a QA metric for any learned model: how
    often it contradicts a plain physics rule (§4).
  - Test splits by bearing unit, which is already our rule.
- **Reject:** thresholds learned from labeled faults on the same machine.

### Zhu, Chen, Shen, *A new multiple source domain adaptation fault diagnosis method between different rotating machines* (IEEE TII 17(7), 2021)
Method: four bearing datasets (CWRU, IMS, two lab rigs), leave-one-out,
raw-signal-as-image CNN, one adversarial discriminator per source with dynamic
weights. Assumes 80 labeled target samples plus unlabeled target data.

- **Adopt:**
  - The **leave-one-dataset-out** protocol (§4).
  - Their finding that **naively pooling sources hurt**: "source combine" was
    worse than no transfer at all. Sources are compared per source as well as
    pooled (WP5).
  - Their **hardest target was IMS**, the only run-to-failure dataset with real
    faults. That matches our own lesson that artificial faults flatter results,
    and it is why IMS is our sealed final exam.
  - Their few-labeled-target curve. It is the model for the step taken when
    the first maintenance-confirmed fault on a real unit arrives (§3, the
    "first real label" path).
- **Reject:** the raw-signal-image CNN plus adversarial alignment. We have
  already shown this representation/method family fails across rigs (§1).

### Found while writing this plan (precedents for the roadmap's L3 idea)
- **Wang, Taal, Fink**, *Integrating expert knowledge with domain adaptation
  for unsupervised fault diagnosis*, IEEE Trans. Instrum. Meas. 71, 2022
  ([arXiv 2107.01849](https://arxiv.org/abs/2107.01849)). Builds a synthetic
  fault dataset **by augmenting real healthy vibration samples** with expert
  knowledge of fault signatures, then transfers to real faults. Tested on CWRU
  and a wind-turbine dataset. The closest precedent for L3. Read in full
  before WP6.
- **Sobie, Freitas, Nicolai**, *Bearing race faults classification using
  simulation-generated training data and feature-free methods*, PHM-AP 2017.
  Classifiers trained on simulated signals beat ones trained on data from
  other machines.
- **Kim et al.**, *Simulation-driven bearing fault diagnosis for condition
  monitoring without faulty data*, PHM-AP 2023. Domain generalisation from a
  bearing simulation model with adversarial augmentation.

### Classical, label-free signal processing to use as baselines
- **Fast kurtogram** (Antoni, MSSP 2007): picks the demodulation band per
  signal from its own impulsiveness.
- **Cepstrum pre-whitening + squared envelope spectrum** (Borghesani et al.,
  MSSP 2013): removes discrete components and resonance colouring. Smith &
  Randall's CWRU benchmark study (MSSP 64, 2015) is the standard reference for
  this family.

Neither uses labels, so both are legitimate under the zero-shot constraint.
Both attack our fixed 1.5–5.5 kHz band (WP2, H-d).

---

## 3. Target architecture: a layered detector

No single method is enough. Kept from the roadmap: L0–L5 stack from simple and
safe to clever and new, and every layer needs only healthy data, geometry and
speed. Revised here in three ways (marked **new**).

| Layer | What it does | Needs | Status |
|---|---|---|---|
| **L0** Standard severity limits | ISO 20816 velocity-RMS zones A–D by machine class | Machine class, velocity RMS | Ready to build (WP10) |
| **L1+L2** Healthy-referenced physics features (**new: merged**) | Physics features per component (envelope-order families for bearings; 1X/2X, blade pass, sub-synchronous for others), each expressed as a **deviation from that unit's own healthy baseline** at the same operating state | Geometry, speed, healthy data across operating states | Bearings proven across rigs (§1). Healthy-referencing is WP4 |
| **L3** Synthetic faults in real healthy data | Train on "real healthy" vs "real healthy + simulated fault". The machine's own sound is in both classes, so only the fault can be learned. **New: also a per-machine sensitivity estimate** (below) | Healthy data, geometry | WP6 |
| **L4** Fleet comparison | Identical units at the same operating point. One that drifts from its siblings is flagged | Several identical units | Designed. Needs real units |
| **L5** Trend and change-point | Slow rise over weeks beats one bad reading | Months of L1–L3 history | WP7 on run-to-failure data |
| **Decision layer** (**new: explicit**) | Window → recording aggregation; sequential test (CUSUM) across recordings; per-unit calibrated threshold; an explanation naming the dominant fault family | Calibration recordings | WP1, WP7, WP9 |

### Why merge L1 and L2
The current model standardises features with *source* statistics, then shifts
only the decision threshold on target healthy data. A per-feature offset on the
target (a different rig, a different unit) is therefore only partly absorbed.
K005 is the visible symptom.

Expressing each physics feature as a z-score against the unit's own healthy
baseline (first order), or whitening by its healthy covariance (second order:
"healthy-to-healthy CORAL") does two things:
- It is the healthy-only analogue of what CORAL tries to do.
- It is the same baseline L4 and L5 need.

It remains physics (L2) in what it measures and becomes "learn normal" (L1) in
how it is scaled.

### L3's second payoff: a sensitivity estimate we can actually compute on a real machine
Recall cannot be measured on our machines (§0). L3 can do the next best thing:
1. inject simulated faults of increasing amplitude into that machine's own
   healthy recordings;
2. run the deployed detector on them;
3. read off **detection probability vs fault amplitude, on that machine's own
   noise floor**.

That curve depends on the generator, so it is an estimate, not a measurement.
But it is label-free, per machine, and lets an operator compare machines and
sensor placements. WP6 checks it against real faults on public data before it
is trusted.

### The "first real label" path
When maintenance confirms the first real fault on a unit, it becomes a labeled
target example. Zhu et al.'s few-labeled-target curve shows how much a handful
of such examples are worth. The service logs every alarm with the maintenance
outcome (roadmap step 5) so this path exists from day one.

### Component coverage
Rolling-element bearings are the only component validated so far. The other
rows below are designed now and validated only when turbomachine data exists.

| Part | Fault | Where to look | Sensor |
|---|---|---|---|
| Rolling-element bearing | Outer/inner race, ball, cage | Envelope order spectrum at BPFO/BPFI/BSF/FTF; broad order bands if geometry unknown | Accelerometer, ≥ 10 kHz (25 kHz preferred) |
| Fluid-film bearing | Oil whirl, rub, wear | Sub-synchronous 0.38–0.48X; orbit shape; 1X phase | Proximity probes |
| Rotor | Imbalance | 1X amplitude and phase | Any, plus once-per-rev mark |
| Coupling, shaft line | Misalignment | 2X; high axial vibration | Accelerometer, radial + axial |
| Impeller, blades, vanes | Damage, fouling, flow | Blade-pass frequency and sidebands | Accelerometer |
| Pump | Cavitation | Broadband high-frequency rise | Accelerometer, high fs |
| Mounting, fits | Looseness | Many 1X harmonics; 0.5X, 1.5X | Accelerometer |

---

## 4. Uniform evaluation rules (apply to every experiment)

These rules exist because each one was paid for (notes/12, GPU_SERVER_PLAN
§1b, HYDRAULIC_PLAN v4–v6).

1. **Split by unit** (bearing ID or recording), never by window. Report the
   leave-one-unit-out spread, not only the mean.
2. **Leave-one-dataset-out** across the development datasets (CWRU, PU,
   XJTU-SY, Ottawa). A method is judged by how it does on a dataset it never
   saw.
3. **IMS is the vault.** It is downloaded and sealed. Nobody looks at its
   signals or scores until WP11, where the frozen method is scored once.
4. **No target fault labels in any choice:** band, tolerance, features,
   generator parameters, threshold. Development targets and their roles are
   named in advance in each experiment's write-up.
5. **Replication:** at least 5 seeds for anything stochastic, at least 10
   calibration splits, reported as mean ± sd. A single run is not a result.
6. **Metrics:**
   - recall at 5 / 10 / 20% target false-alarm rate, and balanced accuracy;
   - each at **window, recording and unit** level;
   - **FA per healthy unit**, because per-unit failure is the dominant risk;
   - on run-to-failure data: **detection lead time vs false alarms per day**;
   - for learned models: **physics-consistency rate** (Shen et al.).
   - Accuracy and F1 alone are never reported as a verdict.
7. **Drift report first:** source vs target standardised mean differences,
   healthy windows only, printed before any transfer result.
8. **Pre-registered readings:** every experiment writes down what each outcome
   would mean *before* it runs (the GPU plan's style is kept).
9. **Caches are fingerprinted** by config and data manifest (the
   `_cache_fingerprint` lesson). Results are pulled off any rented machine
   continuously (notes/11).

---

## 5. Work packages

In order. Each has a gate. **The compute line on each WP is binding.**

> **Compute rule (owner, 2026-10-15).** The local Mac is not for heavy
> computation.
> - **Light:** seconds; reading cached results, plotting, editing code or docs.
>   May run locally.
> - **Heavy:** feature extraction over a dataset, training, sweeps, dataset
>   downloads or condensing.
>
> Before any heavy step, **stop and ask the owner where to run it**: a rented
> server (`ephemeral_server_manual.md`, notes/11), a Kaggle kernel, or
> elsewhere. Never assume the laptop.

### WP1. Re-score what already exists (no new science)
- Recording-level and unit-level detection on PU from the envelope model's
  window scores. PU has 20 runs per bearing per condition.
- Validate or replace the `bearing` service's `watch`/`faulty` levels (25% /
  50% of windows).
- Calibrated threshold with a finite-sample (conformal-style) quantile, and an
  honest note that exchangeability across units does not hold.
- Reuse: `scripts/analysis/envelope_transfer.py`,
  `scripts/analysis/per_unit.py`, `src/serving_core/services/bearing.py`.
- **Gate:** for every faulty unit, whether it is caught at recording level, and
  FA per healthy unit, are known. The service's verdict levels are justified
  by data or changed.
- **Compute:** light only if cached window scores exist; regenerating features
  is heavy → ask.

### WP2. Explain the 1500 rpm drop
Four hypotheses, tested cheapest and most discriminating first. No band or
parameter is chosen by target labels. Each fix is applied identically to every
dataset and judged on all PU conditions.

| | Hypothesis | Test | Fix if confirmed |
|---|---|---|---|
| H-a | Speed vs load/force | Run all four PU conditions. N15_M01_F10 and N15_M07_F04 share 1500 rpm but differ in load/force. If all N15 conditions are bad, it is speed | Determines which of H-b…H-d to pursue |
| H-b | **Resolution artifact.** Ratio features divide a peak by a sum over the 0.5–12-order band. With 1 s windows that band is ~172 bins at 900 rpm and ~288 at 1500 rpm, so ratios shift with speed for identical physics | Recompute with peak-to-median normalisation and with windows of fixed revolutions | Revolution-based windows (Xie et al.) + resolution-invariant normalisation |
| H-c | **Speed error vs absolute tolerance.** Nominal rpm with ±0.08 orders: a 1% speed error moves BPFI's 3rd harmonic by ~0.15 orders, outside the window | Relative tolerance; measured speed (re-extract PU's `speed` channel, which the condensed bundle dropped) or speed estimated from the vibration | Relative tolerance + measured or estimated speed |
| H-d | **Fixed 1.5–5.5 kHz band** misses the resonance at N15 | Fast kurtogram per window; CPW + squared envelope spectrum | The label-free band selection that wins |

- **Gate:** the cause is identified and fixed, or N15 is reported as a
  documented limit of the method. Either way, the README stops quoting N09
  alone.
- **Compute:** heavy (PU feature extraction ×4 conditions × variants) → ask.
  Re-extracting the speed channel uses `scripts/build_kaggle_notebook_prepare_pu.py`
  (Kaggle).

### WP3. Feature set v2
- Extend `src/domain_adaptation/features/envelope.py` (keep v1 reproducible
  behind a version flag):
  - harmonics to 5, shaft-rate sidebands to 3, BSF and FTF families;
  - the sub-band order spectrum vector;
  - a few time-domain impulsiveness statistics (kurtosis, crest, impulse
    factor).
- **Unknown-geometry mode:** broad order bands covering the plausible BPFO and
  BPFI ranges for common bearing families, for units whose geometry is unknown.
- **Stability selection across source datasets.** Keep the features whose
  selection is stable when the source dataset changes. The target is never
  used.
- AWGN robustness curve.
- **Gate:** v2 beats v1 on leave-one-dataset-out at equal or lower FA, or v1
  stays.
- **Compute:** heavy → ask.

### WP4. Healthy-referenced normalisation and physics typing
- Compare three scalings:
  - (a) the current global source scaler + threshold calibration;
  - (b) per-unit healthy z-scoring;
  - (c) healthy-to-healthy CORAL whitening.
  - Same features, same classifier.
- **Physics typing:** which fault-order family rose most relative to the unit's
  own baseline. No learning. It directly tests notes/13's hypothesis that the
  outer-race peak is being masked.
- **Gate:** (b) or (c) reduces FA on the worst healthy unit (K005-type) without
  losing recall. Typing beats 60.8% race accuracy, or is dropped from the
  claims.
- **Compute:** heavy if features must be regenerated → ask.

### WP5. Multi-source with new public datasets
- **XJTU-SY:** 15 run-to-failure bearings, 3 operating conditions, known
  geometry, real faults. A development source and target, and the run-to-failure
  test bed for WP7.
- **Ottawa:** time-varying speed. Exercises computed order tracking and the
  speed estimator.
- **IMS:** downloaded and sealed (§4.3).
- Each dataset goes in its own new directory, is treated as untrusted, and is
  read by scripts kept outside it. Large data is condensed to the channel and
  rate we use (the PU 437 MB pattern) on Kaggle or the rented box, never on the
  laptop.
- Compare per-source and pooled training (Zhu et al.: pooling can hurt).
- **Gate:** leave-one-dataset-out results for all four development datasets
  exist, with the same rules as §4.
- **Compute:** heavy (downloads, condensing, features) → **ask, including the
  download route**.

### WP6. L3: synthetic faults in real healthy data
Read Wang/Taal/Fink 2022 in full first. Ask the owner for any paper that cannot
be downloaded.

Arms, all on identical test data and all with 5 seeds:

| Arm | Method | Why it is here |
|---|---|---|
| A | One-class on target healthy (no fault knowledge) | The bar to beat |
| B | Physics features trained on public sources (current best, v2 if WP3 won) | Current method |
| C | Synthetic faults + physics features | L3, simple |
| D | Synthetic faults + sub-band envelope order spectrum | L3, richer |
| E | C or D + healthy-referenced scaling (WP4) | The combined architecture |

- **Generator, pre-registered and never tuned on real faults:**
  - an impulse train at the fault order with 1–2% slip jitter;
  - an impulse response taken from the healthy signal's own dominant
    resonance;
  - inner-race amplitude modulation at shaft rate;
  - an amplitude sweep.
- **Success:** C, D or E clearly beats A on held-out real faults, **across
  datasets** (PU N09, PU N15, XJTU-SY), including a held-out healthy unit.
  Beating B alone is not enough.
- **Also produce** the per-machine sensitivity curve (§3), and check whether
  it predicts which real faults were caught.
- **Compute:** heavy → ask.

### WP7. Decision layer and trend (L5)
- Window → recording aggregation; CUSUM/sequential test over recordings.
- On XJTU-SY run-to-failure:
  1. calibrate on each bearing's early life (assumed healthy);
  2. report **detection lead time vs false alarms per day**. This is the
     deployment question in its natural form.
- **Gate:** a lead-time/FA trade-off curve exists and its operating point is
  chosen by an FA budget, not by lead time.
- **Compute:** heavy → ask.

### WP8. Learned models, only inside the physics representation
- A small 1D CNN on the sub-band envelope order spectrum, optionally with
  Shen et al.'s physics-consistency penalty. Trained on public sources plus L3
  synthetic faults.
- **Kept only if** it beats the linear model on leave-one-dataset-out at equal
  FA, with 5 seeds.
- **Closed:** DANN or any alignment over spectrograms. GPU_SERVER_PLAN phase 6
  (the DANN sweep) is dropped. The barrier was the representation, so tuning
  the alignment answers a question nobody needs answered.
- **Compute:** heavy, possibly GPU → ask.

### WP9. Productise into serving_core
In `src/serving_core/services/bearing.py`, using the registry in
`src/serving_core/registry.py`:
- Feature version v2 (or v1) recorded in model metadata. Inference refuses a
  mismatch.
- `calibrate` stores **per-unit healthy baseline statistics** and prints:
  - the drift report (§4.7);
  - the L3 sensitivity curve (if WP6 succeeded);
  - the existing "healthy baseline already reads as faulty" verdict.
- Recording-level verdict levels come from WP1.
- Per-file explanation: the dominant fault family and its rise against
  baseline.
- `vibration` remains the generic L1 autoencoder baseline. `hydraulic` is
  frozen as is.
- Tests in the style of `tests/test_serving_core.py`.
- **Gate:** contract tests pass. One end-to-end train/calibrate → promote →
  infer → rollback run.
- **Compute:** code is light. Running the torch suite or Docker builds is
  heavy → ask.

### WP10. Turbomachine readiness (no data yet)
- **Inventory template** for each machine type: bearing type and part numbers,
  blade/vane counts, speed range, sensors and mounting, unit IDs.
- **Data-collection spec** (from the roadmap §7):
  - raw waveforms at ≥ 25 kHz;
  - a synchronous speed signal or keyphasor;
  - the operating state (load, flow, pressure, temperature);
  - sensor positions;
  - the maintenance log;
  - healthy data across every operating state actually used.
- **L0 module:** ISO 20816 velocity-RMS zones (no training).
- **Speed-from-vibration estimator** (1X tracking), validated on Ottawa and PU
  against measured speed.
- **Pilot acceptance test:** what must hold on the first healthy real
  recordings before monitoring starts:
  - the calibration verdict is clean on every unit;
  - the drift report has no extreme features;
  - speed is known;
  - the sensitivity curve is acceptable.
- **Compute:** light (code, docs). The speed-estimator validation is heavy →
  ask.

### WP11. Final exam and write-up
- Freeze the method: features, scaling, threshold rule, decision layer. Record
  the git commit.
- Score it **once** on IMS.
- Update the README verdict, the turbomachinery roadmap and the reports with
  development vs vault numbers side by side.
- **Gate:** none. The result is reported whatever it is (§7).
- **Compute:** heavy → ask.

---

## 6. What we stop doing

- Spectrogram (or raw-signal-image) networks trained on one machine and
  expected to work on another.
- DANN, CORAL, MMD or pseudo-labelling over the target marginal. They do not
  apply to healthy-only targets as published.
- Single-seed claims, and any number without its spread.
- Tuning anything on the faults we test on.
- Kaggle as a runtime (it stays a data relay only).
- Assuming where computation runs. The owner decides for every heavy step.
- Research on the hydraulic rig, until real pump process data exists.

---

## 7. How we would know we are wrong

| If… | then… |
|---|---|
| WP2 finds N15 cannot be fixed with label-free choices | The method is speed-sensitive in a way we do not understand. The pilot needs per-operating-state calibration, and the README says so |
| WP3/WP4 variants never beat v1 + global scaling | v1 stays. Simpler is kept |
| WP6: no L3 arm beats one-class (A) across datasets | L3 is dropped from the pilot. Physics features + calibration + fleet + trend is the product |
| WP6: the sensitivity curve does not predict which real faults were caught | It is not shipped as a sensitivity estimate |
| WP8: no learned model beats linear | No learned model ships |
| WP11: the vault result is far below development | The method was overfit to the development datasets. That is reported prominently, before any pilot claim |
| The first real turbomachine data has no speed signal or no usable geometry | Bearing layers fall back to the unknown-geometry mode and L0/L1/L4/L5 carry the pilot |

---

## 8. Logistics

- **Compute:** see the §5 rule. Rented GPU/CPU boxes follow
  `ephemeral_server_manual.md` and `notes/11_gpu_server_runbook.md`. Results
  are pulled continuously because the box is deleted at subscription end.
- **Data routes:**
  - Kaggle → laptop → server was needed before (the server reached Kaggle
    ~830× slower than other hosts). Condensing happens on Kaggle or the server.
  - Each dataset sits in its own directory and is treated as untrusted. Scripts
    live elsewhere and are run with paths as arguments.
- **Papers:** `papers/` holds the three supplied PDFs (untracked). Anything
  that cannot be downloaded is requested from the owner.

---

## 9. Document map

| File | Role now |
|---|---|
| `MASTER_PLAN.md` | **The forward plan** (this file) |
| `README.md` | How to run things; headline results |
| `PLAN.md` | History: the CWRU autoencoder MVP's design decisions |
| `HYDRAULIC_PLAN.md` | History: hydraulic experiment (frozen) |
| `DOMAIN_ADAPTATION_PLAN.md` | History: DANN kernels v1–v8 |
| `GPU_SERVER_PLAN.md` | History: strict cross-dataset arms (arm A retracted) |
| `DEPLOYMENT_PLAN.md` | History, **except §8**, which still describes what `src/serving_core/` implements |
| `notes/12_reproduction_investigation.md` | Evidence: why spectrogram transfer fails |
| `notes/13_envelope_order_features.md` | Evidence: the envelope-order result |
| `notes/11_gpu_server_runbook.md`, `ephemeral_server_manual.md` | How to use a rented server |
| `reports/turbomachinery_roadmap.html` | Non-technical summary (L0–L5). To be updated at WP11 |
| `deploy/README.md`, `deploy/runbooks/` | Operator documentation |
