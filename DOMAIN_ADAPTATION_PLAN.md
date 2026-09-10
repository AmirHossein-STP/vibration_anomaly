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
