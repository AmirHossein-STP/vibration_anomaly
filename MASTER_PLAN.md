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

**Status 2026-10-22 (one autonomous session on the rented box; owner instruction:
"run the plan, don't ask").** Done: WP5, WP7 (on XJTU-SY), WP3 (v1 stays), the WP10
speed-estimator validation, and the WP6 sensitivity curve (rejected). The current method:
- **detect** with the healthy-referenced CPW+SES physics score (arm d), against each unit's
  *own* healthy baseline, with cross-fitted conformal thresholds;
- **trigger slow trends** with an own-baseline CUSUM on a level feature where speed and load
  are constant;
- **diagnose** with physics typing.

**Phase 2 executed 2026-10-22** (owner: "works on many bearings", heavy steps without asking).
The pre-registered results are in notes/22–27:

| WP | result |
|---|---|
| 12 | method v1.0 = one module (`method.py`), parity exact (10,749 numbers) |
| 13 | bearing catalog (11 bearings, published frequencies checked); **geometry-free score ships** (AUC ≥ six-line on 6/6 dev rows) |
| 14 | **HUST, 5 bearing types (unseen): passes** (AUC ≥ 0.80 on 4/5 types, typing 0.83); 6205 fails (inconsistent healthy baseline) |
| 15 | **LDM, varying load and speed (unseen):** level trigger conditioned on the operating point is best (14/17); physics does not take over as trigger; geometry-free beats six-line → **method v1.1** |
| 16 | FEMTO: own-baseline RMS CUSUM detects 15/17 with about 1 false event/day; records must be ≥ 1 s; baseline ≥ 30 min |
| 17 | `bearing` service rebuilt on the method: fleet of units, catalog/geometry/factors/geometry-free, cross-fitted thresholds, typing hint, per-unit trend; replay on real data exact; Docker cycle passes |
| 18 | **IMS final exam (frozen `e7c01a3`): NOT consistent with development** on the pre-registered readings (1.02 false trend events/day vs ≤ 1; the trend alarm is non-specific). Post hoc: the geometry-free *detection score* rises to 0.88–0.97 on all 4 failing bearings, and the six-line score does not |
| 19 | deployment kit: data spec, inventory template, acceptance checklist, ISO 20816 L0 |

**Phase 3 executed 2026-10-22: different angles** (owner: "totally different approaches, e.g.
meta-learning"). Each track was pre-registered in notes/28–31 before it ran.

| track | result |
|---|---|
| Step 0 | **Ferrara** (6 run-to-failure tests) and **KAIST** (1) sealed *by checksum* (`stage: pin`; nothing downloaded or seen). They are the next exam. IMS is now development data |
| A, meta-learned comparison | Learning `f(window \| own healthy support)` across 8 datasets and testing on the 9th (5 seeds). **Both fail**: the rank pool +0.004 mean AUC (Paderborn −0.06), the bin transformer −0.042. Max-z against the own baseline is as good as the sources can teach (notes/29) |
| B, foundation model (MOMENT-1) + own-baseline kNN | **Ranks faults at AUC ≈ 1.0 on 5/6 sets but cannot be thresholded** (worst-unit FA 0.78–1.00; Paderborn 0.73). The physics features' value is invariance to nuisance changes (notes/31) |
| C, trigger as sequential test | **No trigger passes.** Conformal martingales (Ville threshold) give 3.7–7.1 false events/day on healthy IMS units: real healthy data is not exchangeable. Post hoc, the v1.1 CUSUM with h from *long healthy history* gives 1 event in 236 days and detects 4/4 IMS failures, so **the WP18 failure was the threshold, not the CUSUM** (notes/30) |

**Open next:**
- **The trend trigger's h, learned from accumulated healthy history** (notes/30 §5.1). Confirm on
  Ferrara with h from tests E1..E(k−1) applied to E(k), with no peeking. This is the one Phase 3 lead.
- Write the Ferrara/KAIST exam protocol from the published articles (notes/28 §2) before unsealing.
- Ball/cage coverage.
- Real machine data, still the only real test.

Decisions recorded with the owner on 2026-10-15:

| Question | Decision |
|---|---|
| Real turbomachine data available? | **None yet**, neither recordings nor inventory. The pilot is gated on data collection (WP10). |
| More public datasets? | **Yes**: XJTU-SY and Ottawa for development. **IMS is sealed** as the final exam (WP11). |
| Hydraulic (ZeMA) subproject | **Frozen** as the reference template for process-sensor fusion. No new research until real pump process data exists. |
| Where does computation run? | **Not on the local Mac.** Every heavy step asks the owner first (§5). |
| (2026-10-22) Autonomy | For the 2026-10-22 session only: heavy steps ran on the rented box without asking, each pre-registered in notes/17–21 before it was run. |

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
| Healthy-only calibration holds FA **on average over units, not per unit** | Threshold from 5 healthy PU units, FA on the 6th (conformal quantile, 10% target): mean 0.107–0.115 in all 4 conditions, worst unit 0.22–0.38. The offending unit changes with condition (K005 at N09, K002 at N15, K006 at F04) | notes/14 |
| Recording verdict levels (bearing service) | Measured on held-out healthy units: `watch` at 25% flagged 23% of healthy recordings at N09, so the default is now **40%** (12.5%); `faulty` at 50% flags ~5% (kept). At N09, 9/11 real-damage units are judged faulty on ≥ half their recordings | notes/14 |
| Deployment plumbing works | serving_core `bearing` service on real PU data (900 rpm): `faulty` on 4% of healthy and 76% of real-damage recordings. Calibrate flagged K005. Same numbers on host and in Docker | DEPLOYMENT_PLAN §8 |

### Indicative or open

| Claim | Why it is not established | Source |
|---|---|---|
| The method's performance at 1500 rpm | PU N15: balanced **0.685 ± 0.005**, recall@10% 0.469 (notes/13). WP2 H-a: **all three N15 conditions are bad** (held-out-unit recall 0.48 / 0.51 / 0.29 vs 0.74 at N09), so it is speed, with low radial force (F04) as a second factor. H-b, H-c (nominal speed) and H-d as a drop-in **rejected** (notes/15). Post-hoc: v1 features hold less information at 1500 rpm (in-condition ceiling AUC 0.70–0.79 vs 0.94); CPW+SES features hold it (0.88–0.90) but the CWRU boundary inverts on them; order-peak features keep their direction everywhere, shape statistics and sidebands do not | notes/13, notes/14, notes/15 |
| A **healthy-referenced physics score** (no source data, no classifier) detects across speeds | Max one-sided z of log fault-order peaks vs the machine's healthy baseline, on CPW+SES features: PU recall 0.746 at N09 (FA 0.112) and 0.66–0.68 at N15; held-out AUC 0.84–0.90 in all 4 conditions vs 0.61–0.89 for the CWRU model. Paderborn is spent for this choice; needs WP5 + vault | notes/16 |
| Healthy-referenced physics score transfers to rigs never tuned on (WP5) | Frozen arm d, CPW+SES: Ottawa 2018 AUC **0.92** (CWRU classifier 0.48), MFPT **1.00** (0.84), Ottawa 2023 own-unit baseline **0.93** (0.72). Gate as pre-registered PARTIAL: FA overshoot with tiny calibration sets (fixed post hoc by cross-fitting) and a fleet baseline on Ottawa 2023 (0.71) | notes/17 |
| No single statistic wins everywhere; the physics score is the robust one | Own-baseline kurtosis: AUC 0.99–1.00 on Ottawa 2018/2023, MFPT, CWRU, but **0.33–0.45** on Paderborn real damage. RMS: confounded by speed (Ottawa 2018 0.81) and fails on Paderborn. Physics score ≥ 0.84 everywhere with an own/same-model baseline. Max-fusion (dK) loses 0.08–0.11 on Paderborn: rejected | notes/18 §6 |
| A unit's own healthy baseline beats a fleet baseline | Ottawa 2023 (each bearing healthy → faulty): AUC 0.71 → 0.93, worst-unit FA 1.00 → 0.78 (0.22 cross-fitted). One dataset | notes/17 |
| Fault typing transfers | CWRU 3-class classifier: correct race on faulty windows **60.8%** (chance 50%), outer read as inner (notes/13). **Physics typing** (which race's peaks rose most vs healthy baseline, no learning): **0.74–0.81** in all 4 PU conditions, outer no longer masked. Same Paderborn caveat | notes/13, notes/16 |
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
| **Decision layer** (**new: explicit**) | Window → recording aggregation; sequential test (CUSUM) across recordings; per-unit calibrated threshold; an explanation naming the dominant fault family. **Revised 2026-10-22 (notes/18):** trigger = own-baseline CUSUM on a level feature (RMS / L0); physics score = diagnosis, and also a trigger where speed or load varies | Calibration recordings | WP1, WP7 done on XJTU-SY, WP9 |

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
>
> **Update 2026-10-22.** The rented box is driven by `remote/box` (`remote/README.md`). Kaggle
> is unreachable from it. The "ask first" rule applies again unless the owner restates the
> 2026-10-22 autonomy instruction.

### Phase 2 (written 2026-10-22): from candidate method to frozen method to pilot

**Where Phase 1 left us** (notes/17–21). One method works on every public rig tried, and
nothing richer beat it:
- **detect**: CPW+SES physics score against the unit's **own** healthy baseline, cross-fitted
  conformal threshold;
- **trend trigger**: own-baseline CUSUM on a level feature;
- **diagnose**: physics typing.

Two questions decide the architecture and are still open:
- **(Q1)** Under **time-varying load and speed**, which is what turbomachines actually do, is the level trigger
  confounded, and does the physics score take over as trigger? Every run-to-failure set
  so far ran at constant conditions.
- **(Q2)** What do **false alarms per day** look like with realistically long baselines?

Neither can be answered on data already seen. Two **unseen** datasets were found reachable from the box:
- **Paderborn LDM run-to-failure**: 17 bearings under time-varying load and speed, measured speed per
  record, 128 kHz, ~50 GB. Zenodo 10805042, with a lossless HF repack.
- **FEMTO/PRONOSTIA**: 17 bearings, constant conditions. NASA S3, 1.2 GB.

**Rule for both:** nobody looks at their signals or labels before the WP's pre-registration
is written.

Order, with gates:

#### WP12. Method v1.0, written down and built once (light)
- `notes/22_method_v1.md` is the exact spec:
  - features, windows, speed source (tach / logged setpoint / computed order tracking);
  - own-unit baseline and minimum calibration duration;
  - cross-fitted conformal threshold, recording and unit verdicts, CUSUM trigger, typing output;
  - what is explicitly *not* covered (ball/cage, no-speed machines).
- One implementation, `src/domain_adaptation/method.py`, which the analysis scripts and WP13 both call.
- **Gate:** parity. Run through the new module, notes/17–18's headline numbers reproduce
  exactly (Ottawa 2018 0.922, Ottawa 2023 own 0.930, MFPT 1.000, Paderborn N09 0.901,
  XJTU-SY 13/15 and 9/15).
- **Compute:** light (code). The parity run is seconds on the box.

#### WP13. Productise (replaces WP9)
- `serving_core` `bearing` service on top of `method.py`:
  - per-unit baselines (`calibrate --unit`);
  - cross-fitted thresholds;
  - the level-CUSUM trigger plus physics score;
  - typing in the explanation;
  - speed input required (tach channel, or a setpoint per record; refuse without one);
  - feature/method version in model metadata, with inference refusing a mismatch.
- Replay tests: Ottawa 2023 and XJTU-SY files fed through `ad calibrate/infer` give the WP12
  numbers.
- **Gate:** contract and replay tests pass on the box. One end-to-end calibrate → promote →
  infer → rollback cycle in Docker.
- **Compute:** code is light; the test suite and Docker build run on the box.

#### WP14. Confirmation under time-varying conditions (Q1): Paderborn LDM run-to-failure
- **Pre-register first** (`notes/23`), with v1.0 frozen as of WP12. Questions:
  - (a) false alarms of the level CUSUM vs the physics CUSUM over the early-life period,
    as load and speed change;
  - (b) detection and lead time at an FA budget chosen on early life only;
  - (c) end-of-life typing against the test log;
  - (d) whether the baseline must span the operating states: per-state baselines vs one pooled baseline.
- **Data handling:**
  - one bearing at a time on the SSD (raw ~3 GB each);
  - features and level statistics go to the vault, raw is deleted;
  - the manifest records the checksums. 50 GB never fits the vault and does not need to.
- **Gate (to be fixed in notes/23):** a pre-registered verdict on Q1. Then the decision layer in §3
  says which trigger runs when operating conditions vary.
- **Compute:** heavy (download ~50 GB through the box's downlink, about 1.5–2 h; extraction minutes).

#### WP15. Long-baseline false alarms (Q2): FEMTO/PRONOSTIA
- Its 0.1 s snapshots every 10 s are too short for order features (3 revolutions). So this tests
  the **trigger** (level/kurtosis CUSUM) and its false alarms per day with hours of
  baseline and fine time resolution. The physics score is not applicable, and that is reported.
- **Gate:** an FA-per-day vs lead-time curve whose operating point is chosen from early life only.
  It is compared with XJTU-SY (notes/18).
- **Compute:** heavy but small (1.2 GB, minutes).

#### WP16. Freeze v1.0, then the IMS final exam (replaces WP11)
- After WP14 (and WP15), freeze: record the git commit in `notes/22`. A WP14 failure means
  revising, re-freezing and saying so. It does not mean tuning on IMS.
- Score IMS **once** with the frozen system. Unseal it with `fetch.py unseal ims --commit <sha> --confirm WP11`.
  IMS has 3 run-to-failure tests at 20 kHz, 1 s every 10 min, with documented inner, roller and outer failures.
- Report development vs IMS numbers side by side in the README, the roadmap and the reports.
- **Gate:** none. The result is reported whatever it is (§7).
- **Compute:** heavy (minutes on the box).

#### WP17. Pilot readiness (replaces the WP10 documents; light; **needs the owner**)
- **Inventory template:** per machine, the bearing type and geometry, blade/vane counts, speed range,
  speed signal available, sensors and mounting, unit IDs.
- **Data-collection spec**, revised by Phase 1:
  - a **keyphasor/encoder, or a logged speed setpoint per record, is mandatory** (notes/20);
  - ≥ 25 kHz raw waveforms;
  - **healthy baseline: days, spanning every operating state** (notes/18);
  - the operating state logged per record;
  - a maintenance log.
- **L0 module:** ISO 20816 velocity-RMS zones.
- **Pilot acceptance test:**
  - clean calibration verdict per unit;
  - drift report;
  - speed known;
  - baseline duration met.
- **Ingestion format** for the service, plus `box data push` for private recordings (already built).
- **Gate:** the owner can hand the spec to the site, and the service accepts a recording in the spec's
  format end to end.

#### WP18. Research backlog (only after WP16; in this order)
1. Ball/cage coverage. notes/19: lines alone failed. Try a non-race "anomalous but untyped" output
   from the own-baseline score.
2. Fleet baseline for new units with no history. v2max helped Ottawa 2023; test it on LDM.
3. Speed from vibration with continuity tracking, for machines without a tach.
4. WP6 L3 arms A–E with a generator validated against real signatures (Wang/Taal/Fink 2022).
5. WP3 leftovers: unknown-geometry bands, AWGN robustness.
6. WP8 learned models stay closed unless 1–5 leave a measurable gap.

**What would change this order:**
- If real turbomachine data arrives, WP17 jumps to the front, and WP14 is run on it, on its
  healthy part, before anything else.
- If WP14 shows the physics score cannot trigger under varying conditions either, the pilot
  must log the operating state and use per-state baselines. WP17's spec already demands that.

### Phase 1 (WP1–WP11, record; statuses as of 2026-10-22)

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
- **Status 2026-10-15: done** (notes/14). Feature cache built locally with the
  owner's approval (79 s). Gate met: per-unit table known; `watch` 0.25 → 0.40,
  `faulty` 0.50 kept. Conformal quantile in the service itself is left to WP9.

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
- **Status 2026-10-15: H-a done** from the WP1 cache (notes/14 §4): speed is
  the primary factor, radial force a second one.
- **Status 2026-10-15: H-b, H-c (nominal speed), H-d (CPW+SES) done, none
  explains the drop** (notes/15). Gate outcome: N15 is a documented limit for
  now. Open: H-c with measured speed (Kaggle re-extraction). Two follow-ups
  were pre-registered in notes/15 §3: a peaks-only arm in WP3, and
  healthy-referenced CPW+SES scoring in WP4.

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
- **Status 2026-10-22: done, v1 stays** (notes/19). v2 adds harmonics to 5, BPFI sidebands,
  and the BSF and FTF families, with family-mean aggregation.
  - It improves ball/cage AUC on Ottawa 2023 (+0.13 to +0.14).
  - It loses race AUC on Paderborn 1500 rpm (−0.10 to −0.19), Ottawa 2018 and MFPT, and it
    degrades typing.
  - The naive max over all lines wins on 3 datasets and loses on Paderborn.
  - Ball and cage stay a documented gap. Unknown-geometry mode, stability selection and the AWGN
    curve were not run (time). Open.

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
- **Status 2026-10-15: done on Paderborn** (notes/16), from the WP2 caches.
  Gate: (b)/(c) **fail** on v1 features (worst-unit FA rises). Typing
  **passes** (0.74–0.81). Unplanned but pre-registered arm (d), the
  healthy-referenced physics score, is the new primary candidate. It needs
  confirmation on WP5 datasets because Paderborn is spent for this choice.

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
- **Status 2026-10-22: done** (notes/17). The box downloaded every dataset onto the vault
  (`remote/README.md`): XJTU-SY, Ottawa 2018 and 2023, and MFPT added, IMS sealed.
  - Frozen arm d (CPW+SES physics score) gate as pre-registered: **PARTIAL**.
    - Ottawa 2018 passes (AUC 0.92, with computed order tracking).
    - MFPT gives AUC 1.00 but FA 0.18.
    - Ottawa 2023 against a fleet baseline gives AUC 0.71.
  - Two causes, both now fixed in the method:
    - small-sample calibration bias → cross-fitted calibration scores (FA back to ≤ 0.10);
    - unit-to-unit spread → each unit's own healthy baseline (Ottawa 2023 AUC 0.71 → 0.93).
  - The CWRU classifier fails on every new rig (AUC 0.48–0.84).
  - Physics typing transfers: 0.63–1.00, and 10/11 on XJTU-SY run-to-failure.
  - The XJTU-SY timing criterion fails against a plain RMS 3σ rule. The per-minute score is too
    noisy for a 3-in-a-row alarm, which is WP7's decision-layer job.

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
- **Status 2026-10-22: sensitivity curve done; it does NOT predict** (notes/21).
  - Spearman ρ = +0.20 between synthetic a50 and real recall across Ottawa 2023's 10 race
    bearings (criterion ≤ −0.5). It is not shipped as a sensitivity estimate.
  - The curve's floor is the unit's own false-alarm rate (Ottawa 2023 b02), and the untuned
    impulse generator does not match real damage in this feature space.
  - Arms A–E (training on synthetic faults) are not run yet. They should start from a generator
    checked against real signatures (Wang/Taal/Fink 2022).

### WP7. Decision layer and trend (L5)
- Window → recording aggregation; CUSUM/sequential test over recordings.
- On XJTU-SY run-to-failure:
  1. calibrate on each bearing's early life (assumed healthy);
  2. report **detection lead time vs false alarms per day**. This is the
     deployment question in its natural form.
- **Gate:** a lead-time/FA trade-off curve exists and its operating point is
  chosen by an FA budget, not by lead time.
- **Compute:** heavy → ask.
- **Status 2026-10-22: done on XJTU-SY** (notes/18). The curve exists, and the operating point
  is FA budget 0, set leave one bearing out on healthy minutes only.
  - An own-baseline **RMS CUSUM detects 13/15** bearings (median lead 84 min).
  - The best physics detector (**CUSUM**) detects **9/15**, though earlier in relative life
    (0.34 vs 0.28).
  - Pre-registered reading: **RMS stays the trigger and physics is the diagnosis** at constant
    speed and load. The decision layer in §3 is revised accordingly.
  - False alarms per day cannot be bounded tightly with 20-minute baselines. The pilot must
    record days of healthy data per unit.

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
- **Status 2026-10-22: speed-estimator validation done** (notes/20).
  - Harmonic-sum estimate from vibration: 0.42% median error on Ottawa 2018 (7.5% octave
    locks), but 2.5–8% on Paderborn.
  - Downstream it costs 0.14 AUC against encoder speed, and constant speed per window costs
    0.12 against order tracking.
  - **A synchronous speed signal (keyphasor/encoder), or for speed-controlled drives the
    logged setpoint, is now a hard pilot requirement.**
  - Inventory template, data-collection spec document and the L0 module are not written yet. Open.

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
- (Phase 3) Learning the detection comparison across datasets (notes/29), and generic
  time-series embeddings as a detector across loads or units (notes/31). Both are closed.
- (Phase 3) Conformal-martingale triggers with a theoretical threshold on vibration levels
  (notes/30). Healthy levels are not exchangeable.

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

- **Compute:** see the §5 rule. Rented boxes are driven by `remote/box`
  (runbook: `remote/README.md`): `box up <ip>` makes a fresh rental ready in
  ~30 s, results sync to the persistent vault every 60 s, `box down` before the
  rental ends. Every rental is Ubuntu 22.04 with CUDA 12.8, PyTorch, TensorFlow
  and Docker preinstalled.
- **Data routes (2026-10-22):** the box downloads every public dataset itself,
  straight onto the vault (`box data fetch`), from GitHub, Zenodo, MediaFire,
  Mendeley and NASA. Kaggle/Google are **unreachable from the box** ("not
  available in your location"), so the old Kaggle → laptop → server route is
  retired. Large raw sets (Paderborn, XJTU-SY) are condensed on the box's SSD.
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
| `notes/14_wp1_recording_level.md` | Evidence: WP1 recording/unit-level scoring, verdict levels, WP2 H-a |
| `notes/15_wp2_n15_variants.md` | Evidence: WP2 feature variants, the 1500 rpm diagnostic |
| `notes/16_wp4_healthy_referenced.md` | Evidence: WP4 healthy-referenced scoring, physics typing |
| `notes/17_wp5_new_datasets.md` | Evidence: WP5, the frozen physics score on Ottawa 2018/2023, MFPT, XJTU-SY; cross-fitted calibration; own vs fleet baseline |
| `notes/18_wp7_decision_layer.md` | Evidence: WP7 decision layer (lead time at an FA budget); RMS/kurtosis vs physics; the rejected fusion |
| `notes/19_wp3_feature_v2.md` | Evidence: WP3 feature v2 (rejected; v1 stays) |
| `notes/20_wp10_speed_from_vibration.md` | Evidence: WP10 speed estimator validation; the speed-signal requirement |
| `notes/21_wp6_sensitivity_curve.md` | Evidence: WP6 synthetic-fault sensitivity curve (does not predict real detection) |
| `notes/28_phase3_exam_sealed.md` | Phase 3: Ferrara + KAIST sealed by checksum (the next exam) |
| `notes/29_p3_meta_learning.md`, `notes/30_p3_trigger.md`, `notes/31_p3_foundation_model.md` | Phase 3 evidence: meta-learning, trigger as sequential test, foundation-model embeddings |
| `remote/README.md` | **How to use a rented server** (`remote/box`), the vault, the dataset registry |
| `notes/11_gpu_server_runbook.md`, `ephemeral_server_manual.md` | History: the hand-run server setup |
| `reports/turbomachinery_roadmap.html` | Non-technical summary (L0–L5). To be updated at WP11 |
| `deploy/README.md`, `deploy/runbooks/` | Operator documentation |
