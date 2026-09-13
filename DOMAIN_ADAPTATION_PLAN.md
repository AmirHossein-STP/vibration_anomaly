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
worked; near 1.0 means it didn't.

**Kernel v2 result (2026-09-13): the diagnostic paid for itself
immediately -- it named the failure instead of leaving us to guess.**

| Model | Accuracy | Macro F1 | Probe separability |
|---|---|---|---|
| baseline (no adaptation) | 0.282 | 0.269 | **0.999** |
| DANN v1 | 0.206 | 0.171 | (not measured) |
| DANN v2 | 0.278 | 0.237 | **0.969** |

- **Adaptation still barely moved the needle.** The probe went 0.999 ->
  0.969: source and target features remain essentially trivially
  separable. Tripling `domain_loss_weight` and ramping lambda 2.5x faster
  bought almost nothing in actual alignment.
- **The class collapse rotated rather than resolved.** v1 never predicted
  outer_race (recall 0.086 -> 0.000 on that column); v2 never gets
  inner_race right instead (recall 0.000, with all 2294 true inner_race
  windows split between healthy and outer_race), while outer_race recall
  *improved* 0.086 -> 0.522. Which class the model abandons is essentially
  arbitrary -- the signature of a classifier extrapolating into feature
  regions it has no training signal for.
- **And the diagnosis is now unambiguous**: `domain_acc` sat at 0.487-0.524
  for all 13 epochs. That is *exactly* the majority-class rate, because
  every batch is 64 source + 64 target, so answering "target" for
  everything scores 0.50. The in-training discriminator learned **nothing**
  -- while a fresh probe on the very same frozen features separates the
  domains at 0.969. Features separable + discriminator unable to exploit
  them = the discriminator is losing the minimax race, being scrambled by
  the reversal faster than it can learn, and therefore feeding the feature
  extractor meaningless gradients. (Credit where due: this was predicted in
  advance as the risk of raising `grl_gamma`, before the numbers came in.)

**Sobering context that belongs next to these numbers**: TARGET_TEST is
456 healthy / 2294 inner_race / 1906 outer_race, so a model that blindly
answers "inner_race" every time scores **0.493 accuracy / 0.220 macro F1**.
Both our models are *below* that on accuracy and only marginally above it
on macro F1. This is not a system that nearly works and needs tuning -- as
set up, source-trained features do not transfer to the held-out condition
at all. Worth stating plainly rather than reporting 0.282 as though it
were a result.

**Kernel v3 (2026-09-13, running)**: single change ->
`discriminator_lr_multiplier` 1.0 -> 5.0, the knob wired up in v2 for
exactly this evidence. Plus one measurement fix that doesn't affect
training: `domain_acc` is now macro-averaged over the three domains, so
chance is a true 1/3 and the number can't be inflated by the batch's 50%
target share (v2's "0.49" needed a footnote to interpret; v3's won't).

**Kernel v3 result (2026-09-13): the 5x discriminator LR destabilized
training outright -- none of the three predicted outcomes, a fourth one.**

```
 ep  lambda  cls_acc  dom_loss  dom_acc   anchor  val_acc
  3   0.774    0.857    1.0010    0.333   0.5374    0.903
  4   0.895    0.546   15.3907    0.335  67.7713    0.367   <- blow-up
  5   0.953    0.382    1.0009    0.333   4.8997    0.367
  ...  (frozen at these exact values through epoch 9, then early stop)
```

At epoch 4 the discriminator's larger updates drove `domain_loss` to 15.4
and `anchor_loss` to 67.8; the feature extractor was blown out and never
recovered. From epoch 5 on, `cls_acc` is pinned at 0.382 and `domain_acc`
at exactly 0.333 -- both heads emitting a constant. The saved checkpoint
is therefore an epoch-3 snapshot from *before* the collapse, i.e. an
undertrained model, not an adapted one. 5x was too much; the knob was the
right idea, the setting was not, and there is no gradient clipping in the
loop to contain a spike like this.

**Three rounds of tuning the adversarial game have not produced
alignment** (probe: 0.999 baseline -> 0.969 v2 -> 0.937 v3). Before
spending a fourth run guessing, the cached spectrograms were pulled and
interrogated locally (minutes, not a 50-minute kernel):

- **The domains are separable from ~10 trivial statistics.** A logistic
  probe on per-window mean, std, and 8 coarse band means separates source
  from target at **0.989** on the raw log-spectrograms, and **0.987**
  after the existing normalizer -- i.e. the current fit-on-pooled-source
  per-bin normalization does not remove the giveaway.
- **Per-window standardization helps but doesn't close it** (0.987 ->
  0.872 on those stats, 0.997 -> 0.946 on a 16x16 downsample). Worth doing,
  not sufficient.
- **Every domain pair is perfectly separable at the input level** --
  including PU_artificial vs PU_target_healthy, which is the *same rig and
  same dataset*, differing only in shaft speed (N15=1500rpm vs N09=900rpm).
  A crude order-alignment proxy (rescaling the frequency axis by 900/1500)
  did not reduce it either.
- **Caveat on that last set, stated because it limits the conclusion**:
  a probe returning exactly 1.000 for *every* comparison is saturated and
  therefore has no resolution -- with 256 features and ~1700 training rows
  it can separate almost any systematic difference. It is consistent with a
  large gap; it does not prove one is unbridgeable. The probe on *learned*
  256-d features (0.999/0.969/0.937) does have resolution, and that is the
  number to track.

**The finding that actually matters, and it is a positive one.** Collapsing
the 3-class predictions to healthy-vs-faulty -- the question this whole
project exists to answer, since the real machine only ever supplies healthy
data -- the adapted model is clearly *better* than the unadapted one:

| Model | Precision | Recall | F1 | False-alarm rate on healthy |
|---|---|---|---|---|
| baseline (no adaptation) | 0.923 | 0.403 | 0.561 | 142/456 = **31.1%** |
| DANN v3 | **0.994** | 0.501 | **0.666** | 12/456 = **2.6%** |

When the adapted model says "faulty" it is right 99.4% of the time, it
catches half of all faulty windows, and it false-alarms on 2.6% of healthy
ones -- against 31.1% for the unadapted baseline. For condition monitoring,
where a developing fault produces many consecutive windows and the cost of
a false alarm is high, that is a genuinely useful operating point, and it
came from a checkpoint taken *before* v3's collapse. (Accuracy is a trap
here: TARGET_TEST is 90% faulty, so "always say faulty" scores 0.902.)

**So the honest summary of Phases 2-4**: zero-shot *fault-type diagnosis*
(3-class) does not transfer -- both models sit below the trivial
majority-class predictor. Zero-shot *fault detection* (binary) partially
does, and adaptation improves it substantially, mainly by cutting false
alarms 31.1% -> 2.6%. That split is the result, and it is a more faithful
answer to the original objective than either number alone.

**Kernel v4 (2026-09-13, running): drop CWRU from the source pool.**
Chosen over two alternatives (retreating the LR knob to 2.0; or simply
reframing the headline as binary detection) because it is the one
remaining experiment that could still change the story rather than just
tune it.

*Hypothesis*: CWRU is not extra data, it is actively harmful. v1-v3 all
asked the feature extractor to be invariant across two entirely different
test rigs AND to transfer fault structure across a speed change. Forcing
CWRU<->PU invariance may be destroying the very structure the classifier
needs. With `cwru.include_in_source: false`, the only shift left between
source and target is PU's own N15 -> N09 speed change plus artificial ->
real damage -- one rig, one sensor, one dataset.

*Kept clean as a single-variable test*: `discriminator_lr_multiplier` goes
back to **1.0** (v2's value, which ran stably) rather than staying at
v3's 5.0, so the only training variable that differs from the v2 reference
run is CWRU in vs. out. `grad_clip_norm: 5.0` is added as a guardrail but
is inert on a healthy run -- it exists so a v3-style spike cannot silently
waste another 50 minutes.

*Checked before spending the run*: dropping CWRU does not skew the label
balance (healthy/inner/outer goes 38.2/27.9/33.9% -> 42.8/28.6/28.6%), so
a change in result can't be attributed to that. Source shrinks 33132 ->
23003 windows, so epochs should also run ~31% faster.

*How to read it*: if target metrics jump, CWRU was hurting and the
multi-rig framing was the mistake. If they stay flat, the cross-dataset
gap was never the bottleneck and the speed shift / artificial-vs-real
damage gap is -- which would make order-domain features (resampling
against shaft angle so fault frequencies land in the same bins regardless
of rpm) the principled next move rather than more adversarial tuning.

**Kernel v4 result (2026-09-13): hypothesis REFUTED, and informatively so.
CWRU was helping, not hurting -- removing it made everything worse.**

| Metric (TARGET_TEST) | with CWRU (v2/v3) | PU-only (v4) |
|---|---|---|
| baseline 3-class accuracy | 0.282 | **0.171** |
| DANN 3-class accuracy | 0.278 / 0.225 | **0.150** |
| DANN binary F1 | 0.666 | **0.371** |
| DANN false alarms on healthy | 2.6% | **35.3%** |
| probe separability (DANN) | 0.937-0.969 | 0.964 |

The baseline comparison is the clean one, since both baselines see only
labeled source data with no adaptation involved: **0.282 -> 0.171 purely
from deleting CWRU**. So the cross-rig data was contributing roughly 11
accuracy points, not sabotaging the features.

**What this actually teaches**: source *diversity* beats source
*purity*. The PU-only source is "closer" to the target in every obvious
respect -- same rig, same sensor, same dataset, one speed step away -- and
it still generalizes worse, because 23k windows from 7 bearing codes on
one rig is a narrower world to learn from than 33k windows spanning two
rigs. The intuition that removing the alien domain would help was exactly
backwards. Note also that outer_race recall went to **0.000 for both
models** under PU-only: with just KA01/KA05 as artificial outer-race
examples, nothing transfers to the 5 real outer-race bearings at all.

Training itself was stable this time (no v3-style blow-up; `grad_clip_norm`
never needed to bite, and `val_acc` peaked at 0.971), so the result is
about the data, not an optimization failure. `domain_acc` hovered at
0.516-0.566 against a chance of **0.50** -- note chance is 1/2 here, not
1/3, because only two domains remain once CWRU is gone, and the balanced
metric correctly averages over just the domains present.

**Config has been restored to `include_in_source: true`** -- leaving the
repo in the worse configuration would be a trap for the next run.

**Where this leaves the experiment.** Three adversarial-tuning rounds
(v1-v3) failed to produce alignment, and the one data-side hypothesis
(v4) is refuted in the opposite direction. The evidence now points at two
things, in priority order:

1. **Add MORE source diversity, since that is the lever v4 just proved
   matters.** v1-v4 used a deliberately trimmed subset: 3 of 6 healthy
   codes, 2 of 7 artificial-outer, 2 of 5 artificial-inner. Widening to
   all of them is a config-only change (`pu.*_codes` lists) and directly
   exploits the finding. Cheapest high-value next run.
2. **Order-domain features** (angular resampling against shaft speed, so
   BPFO/BPFI land in the same bins at 900rpm as at 1500rpm). This is the
   principled fix for the speed shift, which remains the untreated gap.
   PU ships a `speed` channel in every `.mat`; CWRU encodes rpm in the
   filename -- so the data supports it. Bigger implementation than (1).

And independently of both: the binary-detection reframing still stands as
the honest headline, and still needs no new training.

**Kernel v5 (2026-09-13, running): use every available source bearing
code, 7 -> 18.** The direct application of v4's finding. v1-v4 all ran on
a subset chosen in the very first planning pass purely to keep the first
run cheap (3 of 6 healthy, 2 of 7 artificial-outer, 2 of 5
artificial-inner); that choice was never revisited once it stopped being
justified. With diversity now the one lever shown to move target
performance, spending compute on breadth rather than on more adversarial
tuning is the best-evidenced move available.

Single variable again: every other setting matches the v2 reference run
(CWRU back in, `discriminator_lr_multiplier` 1.0, `domain_loss_weight` 3.0,
`grl_gamma` 25.0), with `grad_clip_norm` retained as the inert guardrail.

**CORRECTION (written after the run): the claim originally made here -- that
TARGET_TEST was untouched and the numbers stayed directly comparable to
v1-v4 -- was wrong.** `healthy_codes` feeds BOTH the source pool and the
target condition's healthy holdout, so widening it 3 -> 6 doubled
TARGET_TEST's healthy rows, 456 -> 917 (total 4,656 -> 5,117). The faulty
half is genuinely untouched (2,294 inner + 1,906 outer = 4,200 identical
windows), but accuracy, precision and macro-F1 all depend on the class mix
and are therefore NOT comparable across the v4/v5 boundary. Only two
metrics survive it: **faulty recall** (computed on the identical 4,200
windows) and **false-alarm rate** (a rate measured on healthy windows
only). Those are the ones used in the verdict table below. Fixing this
properly would mean a separate `pu.target_healthy_codes` so source breadth
and test composition can move independently -- noted, not yet done.

Cost: 2.09x the source windows (33,132 -> ~69,280), 1,420 PU `.mat` reads
instead of 760, ~103 min projected against v2's ~50. Class balance stays
sane (6 healthy / 7 outer / 5 inner codes).

*How to read it*: if target metrics climb, diversity was the bottleneck
all along and the adversarial machinery was never the limiting factor --
which would also mean v1-v4's conclusions about DANN were drawn on an
under-powered source. If they stay flat, breadth is exhausted as a lever
and the untreated speed shift is the remaining suspect, making
order-domain features the next real engineering effort.

**v5 attempt 1 (kernel version 5) died after 3 minutes -- a latent bug the
narrow code subset had been hiding.** `scipy.io.loadmat` raised
`TypeError: Expecting matrix here` on one of the newly-added bearing
codes' `.mat` files. The 7 codes used in v1-v4 all parse cleanly; at least
one of the 11 added for v5 does not (a flaw in the Kaggle mirror, not in
this code). **A single unreadable file out of ~1,420 killed the entire
run**, and scipy's exception names no path, so the traceback did not even
say which file.

Fixed properly rather than by dropping the suspect codes blindly:

- `pu_loader.UnreadableMatFile` wraps scipy's bare exception **with the
  file path**, so a bad file identifies itself.
- `dataset._windows_and_meta_for_pu_rows` skips unreadable files, prints
  each one, and reports a per-split count **broken down by bearing code**
  -- "60 files failed" is ambiguous between one dead code and scattered
  rot, and those warrant different responses.
- It aborts if more than `MAX_UNREADABLE_FRACTION` (10%) of a split is
  unreadable. Calibration is deliberate: one entirely-corrupt code is
  3 conditions x 20 runs = ~6% of a split, which costs 1 of 18 codes and
  should proceed loudly rather than abort; a broken mount is far above
  10%. This is the same posture the CWRU downloader arrived at
  (vibration_ad's `notes/07_dvc_and_kaggle.md`), where the opposite bug --
  swallowing a *total* failure -- hid a dead run for 35 minutes.
- Three tests cover it: the error names the file, an isolated bad file is
  survived, and wholesale corruption aborts.

Re-pushed as kernel version 6. Note the run is now also self-diagnosing:
whatever it skips will be named in the log, so the next question ("how
much of the new breadth actually loaded?") answers itself.

**v5 attempt 2 (kernel version 6) died at the same 3-minute mark, but from
a different and more fundamental cause: SIGKILL 9, the OOM killer.** The
loader fix worked -- zero files were skipped, so the corrupt file was never
even reached. The build simply ran out of memory first. Doing the arithmetic
rather than guessing:

| source_train memory, one-shot STFT | |
|---|---|
| raw windows (float64) | 4.3 GB |
| + `np.concatenate` copy | 4.3 GB |
| + scipy's complex128 STFT intermediate | **17.0 GB** |
| + magnitude (float64) | 8.5 GB |
| **peak** | **~34 GB** vs Kaggle's ~30 GB |

**The one-shot STFT was always the scaling wall** -- v2's 33,132 windows
peaked near 16 GB and survived on luck as much as design; 2.09x does not
fit. Nothing about the breadth experiment was wrong; the data pipeline
simply never scaled.

Fixed structurally, so it now scales to any breadth:

- **STFT is applied per FILE** (~19-78 windows) and raw windows are dropped
  immediately, instead of accumulating the entire split and transforming it
  in one call. Peak drops to ~8.5 GB -- just the float32 output plus one
  concatenate copy -- and the complex intermediates become negligible.
- `train_baseline.py` / `train_dann.py` `del` the un-normalized arrays once
  normalized copies exist (~4.2 GB each at this scale); they remain cached
  on disk, so nothing is lost.
- Each role now prints its array size and peak RSS, so headroom is visible
  in the log instead of being inferred from a corpse.
- The RSS helper handles `ru_maxrss` being KB on Linux but BYTES on macOS
  -- caught locally when it cheerfully reported 150 GB. A misleading gauge
  during a memory investigation is worse than no gauge.

Re-pushed as kernel version 7.

**Both fixes confirmed working in kernel v7's log**: exactly ONE unreadable
file (`KA08/N15_M01_F10_KA08_2.mat`, 0.1% of its split) -- named, skipped,
run continued, where that same file had killed the previous attempt outright
-- and **peak RSS 11.6 GB** against the ~34 GB that triggered the OOM
killer. source_train built out at 69,250 windows (4.2 GB), within 30 windows
of the 69,280 predicted, the difference being the skipped file.

## Kernel v5 result (2026-09-14): breadth works, adaptation does not

| Model | faulty recall (identical 4,200 windows) | false-alarm rate |
|---|---|---|
| v2/v3 baseline (7 source codes) | 0.403 | 31.1% |
| v3 DANN (7 source codes) | 0.501 | 2.6% |
| **v5 baseline (18 source codes)** | **0.761** | **16.7%** |
| v5 DANN (18 source codes) | 0.085 | 9.7% |

**Source breadth was a real bottleneck, and fixing it was the single
biggest win of the entire experiment.** The baseline's detection recall
went 0.403 -> 0.761 on byte-identical faulty windows *while false alarms
simultaneously fell* 31.1% -> 16.7%. Both moved the right way at once, so
this is a genuine capability gain rather than a threshold being traded
along a fixed ROC curve. Its 3-class accuracy also rose 0.282 -> 0.456 and
outer_race recall 0.086 -> 0.513, though those particular figures straddle
the test-set change above, so treat them as indicative.

**The adversarial machinery is not merely useless here -- it is harmful,
and more data makes it more harmful.** v5's DANN collapsed to calling 91%
of everything healthy (recall 0.085) while scoring `cls_acc` 0.99 /
`val_acc` 0.98 on source data. `domain_acc` sat at 0.340 against a chance
of 0.333, and the post-hoc probe still separated the domains at 0.949. So
across five kernels the adversarial component never once achieved
alignment, and its cost to target performance grew with source size.

### Verdict

**The best model produced by this study is the v5 BASELINE -- no domain
adaptation at all.** As a zero-shot fault *detector* on a rig speed it
never trained on, using zero target fault data: **95.4% precision, 76.1%
recall, 16.7% false-alarm rate.** It is also the simplest thing to deploy,
having no discriminator, no GRL and no adaptation step.

Three things this study actually establishes, none of which were the
hypothesis it set out to test:

1. **Zero-shot fault DETECTION transfers across rigs and speeds; fault
   TYPING does not.** Every version, adapted or not, failed to tell inner
   from outer race on the target. The binary question -- the one the
   deployment actually asks -- is answerable.
2. **Source diversity dominates every other lever tried.** Breadth (7 -> 18
   codes) beat three rounds of adversarial hyperparameter tuning combined,
   and v4 showed the converse: removing CWRU cost ~11 accuracy points even
   though it made the source "closer" to the target.
3. **DANN, as applied here, does not work on this problem.** That is a real
   negative result, established across five runs with a diagnostic (the
   separability probe) that can distinguish "aligned" from "discriminator
   was fooled" -- the ambiguity that made v1's failure so hard to read.

### Phase 5 verified (kernel v8, 2026-09-14)

The deployment path is no longer just written, it is exercised. `infer.py`
ran on real data inside the kernel, since it cannot run on the dev laptop
(no torch there):

```
--- calibrate on a known-HEALTHY recording ---
1797_Normal.npz      113 windows   0.0% flagged   healthy
  VERDICT: this machine reads clean. Monitor with the default threshold.
--- detect on a known-FAULTY recording (inner race, 21 mil) ---
1797_IR_21_DE12.npz   55 windows   100% flagged   FAULTY
```

Both verdicts correct, and pointed the right way round -- the cell is
written so a backwards result would be loud rather than silent. **What this
does and does not show**: it verifies the deployment plumbing end to end
(file loading, resampling from an arbitrary rate, windowing, per-recording
aggregation, calibration logic, verdict thresholds). It is NOT a zero-shot
performance claim -- those are CWRU files, in-domain for this model. The
zero-shot numbers are the Paderborn ones above.

v8 also re-ran the whole pipeline with `binary_detection` computed inside
`evaluate.py` rather than by hand afterwards. It reproduced v5's figures
exactly (0.954 / 0.761 / 16.7% for the baseline; 0.800 / 0.085 / 9.7% for
DANN), which doubles as a determinism check across two independent runs of
the same seed and config.

### What would be worth trying next, in order

1. **Order-domain features** -- angular resampling against shaft speed, so
   BPFO/BPFI land in the same bins at 900rpm as at 1500rpm. The speed shift
   is the one gap never addressed; PU ships a `speed` channel and CWRU
   encodes rpm in the filename, so the data supports it. This is the
   principled fix and the only remaining idea with a strong physical
   argument behind it.
2. **More breadth still** -- the source is now every artificial code, but
   `source_conditions` is still 3 of 4, and `source_train_fraction` 0.7.
   Breadth is the proven lever; this is the cheap way to push it.
3. **Drop the adversarial component entirely** for deployment, and keep
   `train_baseline.py` as the product. Adaptation can stay in the repo as
   the documented negative result.

**Still available regardless of v4's outcome**: reframing the headline
result as binary detection. It needs no new training -- the numbers are
already in hand (0.994 precision / 2.6% false-alarm rate, above) -- and it
matches both this repo's other two experiments and the actual deployment
goal.
