"""Zero-shot fault diagnosis via unsupervised domain adaptation (DANN).

Third, independent experiment in this repo (see DOMAIN_ADAPTATION_PLAN.md):
`vibration_ad/` (single-sensor CWRU autoencoder) and `hydraulic_fusion_ad/`
(multi-sensor ZeMA fusion autoencoder) are both untouched. This package
combines CWRU (already local, `data/raw/`) and Paderborn University's KAt
bearing dataset (attached as a Kaggle input dataset, never downloaded in
full) to train a classifier that transfers across both datasets and a
held-out operating condition, using only that condition's HEALTHY data to
adapt -- never a single labeled (or even unlabeled) fault example from the
target -- then checks whether it can diagnose real target faults it never
trained on.
"""
