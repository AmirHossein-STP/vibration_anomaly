# Domain Adaptation on Bearing Conditions: MVP & Feasibility Study

## 1. Executive Summary & Objective
This project addresses **Zero-Shot Fault Diagnosis** via Unsupervised Domain Adaptation (UDA). 
Your physical target bearing currently generates **only normal (healthy) data**. Our objective is to train a diagnostic model on external datasets (where fault data is abundant), adapt it to your target bearing using *only its normal baseline*, and accurately diagnose future faults when they occur.

To prove this works before deploying on your physical system, we will use the **CWRU** and **Paderborn University (PU)** datasets to run a "Leave-One-Domain-Out" feasibility study.

## 2. Dataset Strategy (CWRU + Paderborn)
We will combine two of the most heavily benchmarked datasets in predictive maintenance to create our source domains and our proxy target domain.

*   **Dataset 1: CWRU (Source Domain A)**
    *   **Data:** Deep groove ball bearings, artificial faults (EDM).
    *   **Conditions:** 4 motor loads (0-3 HP), ~12kHz / 48kHz sampling rates.
*   **Dataset 2: Paderborn University - PU (Source Domain B & Proxy Target)**
    *   **Data:** 6 healthy bearings, 12 artificial damages, 14 real damages (run-to-failure).
    *   **Conditions:** 4 combinations of rotational speed, load torque, and radial force. 64kHz sampling rate.

**The "Simulated Target" Approach:**
To simulate your physical bearing (which has no fault data), we will treat the **PU Real Damage** bearings under a specific operating condition as our "Target Domain". We will hide their fault labels during training, use their healthy data to adapt the model, and then test if the model can successfully diagnose their real faults.

## 3. Data Engineering & Preprocessing Pipeline
Since CWRU and PU were recorded on entirely different test rigs with different sensors, aligning their data is the first major hurdle.

1.  **Data Acquisition & Workflow:** 
    *   **CWRU:** Data and metadata are already available locally.
    *   **PU:** Since training will execute on Kaggle, we do not need to download the full PU dataset locally. We will attach the PU dataset directly to our Kaggle environment (`pouriyaamini/vibration-based-fault-diagnosis-of-machines` or similar). For local development and testing of the dataloaders, we will only download the PU metadata and perhaps a tiny dummy sample.
2.  **Resampling:** PU is sampled at 64kHz, CWRU at 12kHz/48kHz. We will downsample PU or upsample CWRU to a common frequency (e.g., 48kHz) to align the Nyquist limits.
3.  **Label Harmonization:** Map both datasets to a universal label space: `[0: Healthy, 1: Inner Race Fault, 2: Outer Race Fault]`. (Note: Ball faults can be added if prevalent in both).
4.  **Feature Extraction:** 
    *   Segment raw 1D vibration signals into overlapping windows (e.g., 2048 or 4096 points).
    *   Apply Continuous Wavelet Transform (CWT) or STFT to generate 2D time-frequency representations (spectrograms/scalograms), which capture transient fault impulses better than pure frequency (FFT).
    *   Z-score normalization applied per domain.

## 4. The Architecture (Deep Domain Adaptation)

We will implement a **Domain Adversarial Neural Network (DANN)** with a conditional alignment strategy.

*   **Feature Extractor ($G_f$):** A 2D CNN (e.g., ResNet-18 backbone) that ingests the spectrograms and outputs a dense feature embedding.
*   **Fault Classifier ($G_y$):** A fully connected head that takes the feature embedding and outputs the fault class probabilities (Healthy, Inner, Outer).
*   **Domain Discriminator ($G_d$):** A secondary classification head that tries to predict which dataset/condition the data came from (CWRU vs. PU_Artificial vs. PU_Target_Normal).

**The Magic (Gradient Reversal Layer - GRL):** 
During training, the gradient from the Domain Discriminator is multiplied by a negative constant before passing to the Feature Extractor. This forces the Feature Extractor to learn features that maximize fault classification accuracy *while simultaneously maximizing the domain discriminator's confusion*. The resulting features become "domain-invariant".

## 5. Execution Phases

### Phase 1: Data Infrastructure (`src/domain_adaptation/data/`)
*   Implement `cwru_loader.py` and `pu_loader.py`.
*   Implement `transforms.py` for resampling, windowing, and STFT/CWT.
*   Create a unified PyTorch `ConcatDataset` that batches data from both domains simultaneously (crucial for DANN training).

### Phase 2: Supervised Baseline (`src/domain_adaptation/models/`)
*   Build the CNN Feature Extractor and Classifier.
*   Train purely on CWRU + PU_Artificial. 
*   Test on PU_Real (without adaptation) to establish the baseline performance drop caused by the domain shift.

### Phase 3: Zero-Shot Target Adaptation (`scripts/train_dann.py`)
*   Introduce the Domain Discriminator and GRL.
*   **The Adaptation Step:** Feed the Target Normal data (PU healthy bearings) into the network. Apply a **Conditional Maximum Mean Discrepancy (C-MMD)** or Adversarial loss *specifically* to align the Target Normal cluster with the Source Normal cluster in the feature space.
*   By aligning the "Healthy" baselines, we hypothesize the relative geometry of the fault clusters will transfer.

### Phase 4: Validation & Visualization (`scripts/evaluate.py`)
*   Run inference on the hidden PU_Real fault data. 
*   Calculate Accuracy, F1-Score, and Confusion Matrices.
*   Generate **t-SNE / UMAP plots** of the feature space. We should visually see:
    1.  CWRU and PU normal data overlapping perfectly.
    2.  Target PU_Real faults falling directly into the boundaries of the CWRU/PU_Artificial fault clusters.

### Phase 5: Deployment Readiness
*   Wrap the trained Feature Extractor and Classifier into a lightweight inference script.
*   Document how to feed your physical bearing's normal data into Phase 3 to create a custom-adapted model for your specific machine.

## Revisions & Decisions (implementation pass, 2026-09-10)

Read alongside the original plan above -- same spirit as `PLAN.md`
(original vs. revised, and why) for the first experiment. Third
independent experiment in this repo: `src/domain_adaptation/`, alongside
untouched `vibration_ad/` and `hydraulic_fusion_ad/`.

**Paderborn dataset source, confirmed by direct inspection, not assumed.**
Searched Kaggle (`kaggle datasets list -s paderborn`), found 11 candidates;
picked `dippatel03/paderborn-db` (most downloaded/voted, 9.7GB) and
verified its structure directly rather than trusting the listing: paged
through the *full* file listing via the API (`kaggle datasets files`,
14 pages) and confirmed all 32 canonical KAt-DataCenter bearing codes are
present (`K001-K006` healthy, `KA01/03/05/06/07/08/09` + `KI01/03/05/07/08`
artificial damage, `KA04/15/16/22/30` + `KI04/14/16/17/18/21` real damage,
`KB23/24/27` real combined-damage), each with exactly 20 runs under all 4
operating conditions (`N15_M07_F10`, `N09_M07_F10`, `N15_M01_F10`,
`N15_M07_F04`) -- full factorial, no gaps to design around. Downloaded 3
real `.mat` files (K001, KA04, KI04) to a scratch dir and loaded them with
`scipy.io.loadmat(..., simplify_cells=True)` to confirm the internal
struct layout empirically: top-level key = filename (minus `.mat`), field
`Y` = a length-7 list of `{'Name', 'Data'}` channel dicts (`force`,
`phase_current_1/2`, `speed`, `temp_2_bearing_module`, `torque`,
`vibration_1`), matching the dataset's own documentation. Only
`vibration_1` (~64kHz, ~256000 samples/4s run) is used, matching this
project's single-accelerometer-channel scope everywhere else. Attached via
`dataset_sources` in the new kernel's `kernel-metadata.json`, exactly like
CWRU's `esraakhaled299/cwru-data` -- **nothing downloaded into the repo,
nothing uploaded anywhere**, per your instruction; the 3 inspection files
were deleted from the scratch dir, never committed.

**Common sample rate: 12kHz (CWRU's native rate), not 48kHz as originally
proposed.** CWRU's Drive-End signal is already 12kHz; PU's `vibration_1`
is 64kHz. Downsampling PU to 12kHz (`scipy.signal.resample_poly(up=3,
down=16)`, exact ratio, proper anti-aliasing) instead of upsampling
everything to 48kHz: (1) reuses `vibration_ad`'s entire, already-verified
windowing/STFT/normalization pipeline (`configs/default.yaml`'s
`segment_length=8384`, `nperseg=256`, `hop_length=64` -> 128x128 spectrogram)
completely unchanged -- literally imported, not reimplemented; (2) avoids
inflating PU's data volume 4x for zero informational gain (bearing fault
harmonics at these rig speeds sit well under 12kHz's 6kHz Nyquist, so
nothing is lost); (3) keeps training CPU-feasible on Kaggle's unverified
(no-GPU) account, the same constraint that shaped `hydraulic_fusion_ad`'s
design. `scipy.signal.stft`'s `fs` argument only labels the returned
frequency axis, never affects the magnitude values from a fixed
`nperseg`/`hop_length` -- so resampling both domains to one common rate
*before* an identical fixed-sample STFT is what actually makes "bin 40"
mean the same physical frequency in every domain; without that step the
whole cross-domain premise (a CNN generalizing across datasets) wouldn't
hold.

**Architecture simplified: a small custom CNN, not ResNet-18.** Same
4-layer stride-2 conv stack as `vibration_ad.models.autoencoder`'s encoder
(reused idea, not reused code -- this one ends in a classification/domain
head, not a decoder), `base_channels=16`. ResNet-18 (~11M params) trained
across two combined datasets on a CPU-only Kaggle kernel would make even
one epoch slow enough to hurt iteration speed, for a benefit (ImageNet-style
residual depth) these 128x128 single-channel spectrograms don't obviously
need. Revisit if the small CNN turns out to underfit both domains at once.

**C-MMD simplified to a "healthy-anchor" alignment loss.** Full Conditional
Maximum Mean Discrepancy needs class-conditional kernel estimates across
domains -- meaningfully more machinery for a first pass. What the plan
actually wants ("align the healthy baselines") is captured much more
cheaply by penalizing the squared distance between the mean feature vector
of labeled healthy examples in the source batch and the mean feature
vector of the target-normal batch (`models/dann.py::healthy_anchor_loss`).
This is a real simplification, not a drop-in equivalent -- documented so
it isn't mistaken for the literature's C-MMD if this gets compared to a
paper later.

**Ball ("B") faults excluded from the universal label space.** PU's real
damage bearings are only ever inner- or outer-race (per KAt-DataCenter
docs, confirmed against the bearing-code table above) -- there's no PU
counterpart to align a "ball fault" class against, so CWRU's `B` files are
dropped from this task entirely (universal space: `0=healthy,
1=inner_race, 2=outer_race`). They remain fully used, unchanged, in the
original `vibration_ad` experiment.

**One target operating condition held out end-to-end: `N09_M07_F10`.**
The other three source conditions all keep rotational speed at N15
(1500rpm) and vary only load torque or radial force; `N09` (900rpm) is the
one axis that shifts bearing characteristic frequencies (BPFO/BPFI/BSF, all
proportional to shaft speed) multiplicatively across the whole spectrum --
the most physically distinct domain gap available among the 4, so the
most meaningful test of whether adaptation actually transfers. Under this
condition: `K001-K003`'s healthy runs split into `pu_target_normal`
(adaptation) vs. held-out healthy test rows; every real-damage bearing's
runs (`KA04/15/16/22/30`, `KI04/14/16/17/18/21`) are 100% test-only,
labels never touched before `evaluate.py` -- the actual zero-shot check.

**Bearing-code subset trimmed for a first Kaggle run, not the full
dataset.** Using every artificial code x all 3 source conditions x all 20
runs would mean ~700+ `.mat` loads (~9MB each) and tens of thousands of
128x128 windows -- workable but slow to iterate on for a first pass.
Defaults in `configs/domain_adaptation.yaml`: 3 healthy codes
(`K001-K003`), 2 artificial-outer (`KA01, KA05`), 2 artificial-inner (`KI01,
KI05`) for source training; all 11 real-damage codes for the target test
set (that's the actual result being measured, not a place to cut corners).
Trivially widened later by editing those lists -- every downstream module
reads them from config, nothing is hardcoded.

## Status

**Implementation complete (2026-09-10)**: `src/domain_adaptation/` (data
loaders for both datasets, DANN model + GRL + healthy-anchor loss,
train_baseline.py / train_dann.py / evaluate.py), `configs/domain_adaptation.yaml`,
`tests/test_domain_adaptation_smoke.py` (9 data-layer tests passing locally
against synthetic data; 7 model/training tests written but only run on
Kaggle -- this Mac's PyPI wheel index caps torch at 2.2.2, which predates
numpy 2.x support and crashes on `torch.from_numpy` with the numpy 2.5
already in this venv; not worth downgrading the whole scientific stack just
for local unit tests), `notebooks/domain_adaptation/` (Kaggle notebook +
kernel-metadata.json, both input datasets attached, nothing
downloaded/uploaded).

**Kernel v1 (2026-09-10): ran end to end (~50 min CPU). Real numbers, and
a real problem.**

*(Correction, worth recording because it contradicts a note in
`HYDRAULIC_PLAN.md`: the kernel log appears to print the whole run three
times over, which looked exactly like that experiment's documented
"each stage runs twice, byte-identical" Kaggle quirk. It isn't. The
per-epoch timings add up to the wall clock with nothing left over --
baseline's 7 epochs sum to 395.4s and that stage reports done at 485.7s
(the ~72s difference being the one-time data build); DANN's 18 epochs sum
to 2438.9s and 485.9 + 2438.9 = 2924.8s against a reported 2931.6s, a 6.8s
residual for process startup. If any stage had genuinely run twice the
second block's timestamps would continue upward, not restart at 6.1s with
identical absolute values. So the repeated blocks are the log VIEWER
re-emitting its buffer; every stage ran exactly once, and there is no
wasted compute to reclaim. The hydraulic experiment's "Kaggle oddity"
note has the same signature and is probably the same artifact -- worth
re-checking there before treating it as a real infra behavior.)*

| Model | Accuracy | Macro F1 | n |
|---|---|---|---|
| baseline (no adaptation) | 0.282 | 0.269 | 4656 |
| DANN (adapted) | 0.206 | 0.171 | 4656 |

DANN is *worse* than the no-adaptation baseline, not better. Two things in
the confusion matrices (rows=true, cols=pred, order
[healthy, inner_race, outer_race]):

```
baseline:                      dann:
[314,  142,    0]              [394,   62,    0]
[901,  837,  556]              [1731, 563,    0]
[1605, 137,  164]              [984,  922,    0]
```

- **Baseline already shows a severe, genuine domain-shift problem** on
  outer_race specifically: 1605 of 1906 true outer_race target windows get
  called "healthy" (8.6% recall) -- this is exactly the "domain shift hurts
  without adaptation" baseline Phase 2 is supposed to establish, and it's a
  real, large effect, not a rounding error.
- **DANN doesn't fix that -- it collapses entirely**: column 3
  (predicted outer_race) is **zero across every true class**. The model
  never predicts outer_race at all on TARGET_TEST once adapted, despite
  reaching 99% accuracy on SOURCE_VAL (epoch 12, the checkpoint actually
  saved) -- so the classifier head can clearly represent all 3 classes on
  source data; something about adaptation specifically destroys that for
  the target domain.

**Confirmed by the actual t-SNE plot (pulled and inspected, not guessed
at): domain alignment did not happen at all -- this is bigger than an
anchor-loss side effect.** `outputs/da_figures/feature_space_tsne_dann.png`,
colored by domain: CWRU and PU_Artificial (source) sit in a handful of
small, tight clusters; PU_Target (the held-out condition, `pu_target`,
n=4656) sprawls across most of the rest of the plot, including large
regions where **no source point exists anywhere nearby**. The classifier
was only ever shaped by source-region data -- in the uncharted regions
where most of the target lands, its behavior is essentially undefined,
and it happened to collapse toward never emitting the outer_race logit
there. My first-pass hypothesis (healthy-anchor loss dragging target
features toward "healthy" specifically) doesn't survive contact with this
plot -- it's not that target features got pulled to the wrong specific
place, it's that they mostly never got pulled anywhere near source at all.

**Likely real cause**: `domain_loss` hovered near chance (ln(3)=1.099) from
epoch 1 onward rather than starting low and rising under adversarial
pressure -- the textbook DANN curve shape. Two plausible, non-exclusive
reasons: (1) the classifier converges on source very fast (`cls_acc=0.812`
already in epoch 1, while `lambda`=0.165 is still near its floor per the
warm-up schedule) -- by the time adversarial pressure ramps up, the feature
extractor may have already committed to source-specific shortcuts the GRL
can't undo; (2) marginal domain-invariance (what the discriminator/GRL
actually optimizes) doesn't imply class-conditional alignment -- a
well-known DANN limitation -- and the single-class, mean-only
healthy-anchor loss is too weak a constraint to compensate on its own for
a domain gap this large (different rig, different sensor, resampled 64kHz
-> 12kHz vs. CWRU's native 12kHz).

**Kernel v2 (2026-09-13): adversarial pressure strengthened, applied
earlier, and -- most importantly -- made falsifiable.** Changes, all one
coherent fix (make the adversarial signal win a race it was losing) plus
one new diagnostic:

1. `domain_loss_weight` 1.0 -> 3.0: a louder adversarial signal relative
   to the classification gradient.
2. `grl_gamma` 10.0 -> 25.0: lambda reaches ~0.46 by the end of epoch 1
   instead of ~0.17, so the reversal arrives while the feature extractor
   is still malleable rather than after it has already committed.
3. **The GRL lambda is now scheduled per ITERATION, not per epoch** --
   which is what the original DANN paper does, and which v1 got wrong.
   This is what makes (2) safe rather than reckless: with a per-epoch
   lambda, a high gamma would hit the discriminator with strong reversed
   gradients from the very first batch, before it has learned anything
   worth reversing, scrambling it into emitting noise. Per-iteration,
   lambda still starts at ~0 during epoch 1's first batches, so the
   discriminator gets a clean head start and the reversal ramps in
   smoothly.
4. `discriminator_lr_multiplier` (default 1.0, inert for now): wired up so
   the discriminator can be given a higher learning rate to stay a step
   ahead in the minimax game, without another code change, if v2's
   evidence says it's losing the race.
5. `anchor_loss_weight` deliberately UNCHANGED at 0.5 -- v1's evidence
   never actually implicated it, and changing two things at once would
   make the result unreadable.

**The new diagnostic is the part that matters most**
(`analysis.domain_separability_probe`, reported per model in
`evaluation_metrics.json`): after training, a FRESH logistic-regression
probe is fit on each model's frozen features to separate source from
target, with classes balanced so chance is exactly 0.5. This closes v1's
blind spot. A domain loss sitting at chance during training has two
opposite meanings -- features genuinely aligned, or a discriminator that
simply never learned anything and is being fooled -- and v1 had no way to
tell them apart, which is why the failure was only caught by eyeballing a
t-SNE plot at the very end. The probe can't be fooled that way: it sees
frozen features with no gradient reversal fighting it. Reported for
baseline AND DANN, so the question "did adaptation move the needle at
all?" has a number, not a vibe -- and it needs no target labels, so it
stays legitimate under the zero-shot constraint.

Read the v2 result as: `separability_accuracy` near 0.5 means alignment
worked; near 1.0 means it didn't. If DANN's probe stays near 1.0 while
`domain_acc` in the training log sits at chance (~0.33), that is exactly
the "discriminator is losing the race" signature -- turn up
`discriminator_lr_multiplier` for v3.
