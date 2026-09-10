"""Pure-pandas/numpy analysis helpers for analyze.py (kept separate from
plotting/torch code, same division as vibration_ad/analysis.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


def threshold_sensitivity(
    val_errors: np.ndarray,
    test_errors: np.ndarray,
    y_true: np.ndarray,
    percentiles: np.ndarray | None = None,
) -> pd.DataFrame:
    """Precision/recall/F1/false-positive-rate across a range of candidate
    thresholds, each still chosen unsupervised (a percentile of
    validation-healthy error) -- see notes/09_hydraulic_notes.md section 4.
    Sweeps down to the 50th percentile, not just 80-100 like
    vibration_ad's version: the first real hydraulic run found near-zero
    recall at the deployed 99th percentile for 2 of 4 components despite
    good ROC-AUC, so the useful range here is wider.

    `false_positive_rate` matters as much as F1 here, maybe more: this
    dataset's per-component TEST sets are heavily skewed toward the
    degraded class (e.g. pump: 984 degraded vs. 200 healthy), so a low
    percentile can score a great F1 by flagging almost everything --
    including the majority of genuinely healthy cycles -- which would be
    useless in real deployment (where healthy operation dominates, the
    opposite skew). FPR = fraction of TARGET-HEALTHY test cycles wrongly
    flagged; low F1-optimal percentiles turned out to carry FPR of
    40-75% for 3 of 4 components on the first real run this was computed
    for -- see notes/09_hydraulic_notes.md section 4b's correction.
    """
    if percentiles is None:
        percentiles = np.arange(50.0, 100.0, 2.0)

    rows = []
    for p in percentiles:
        threshold = float(np.percentile(val_errors, p))
        y_pred = (test_errors > threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        healthy = y_true == 0
        n_healthy = int(healthy.sum())
        fpr = float((y_pred[healthy] == 1).sum() / n_healthy) if n_healthy else float("nan")
        rows.append(
            {
                "percentile": p,
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "false_positive_rate": fpr,
            }
        )
    return pd.DataFrame(rows)


def _row_to_dict(row: pd.Series) -> dict:
    return {k: float(row[k]) for k in ["percentile", "threshold", "precision", "recall", "f1", "false_positive_rate"]}


def f1_optimal_threshold(sensitivity_df: pd.DataFrame) -> dict:
    """The sweep row that maximizes raw F1, with NO constraint on
    false-positive rate.

    MISLEADING ON ITS OWN, and NEVER use this to pick the deployed
    threshold: this dataset's per-component TEST sets are heavily skewed
    toward the degraded class (e.g. pump: 984 degraded vs. 200 healthy),
    so the F1-maximizing percentile can simply be "flag almost everything"
    -- on the first real run this was computed for, the F1-optimal
    percentiles for 3 of 4 components carried a false_positive_rate of
    40-75%, i.e. they'd misflag most genuinely healthy operation in a real
    deployment (where healthy cycles dominate, the OPPOSITE skew from this
    test set). Always read alongside `best_deployable_threshold()` below,
    and always alongside its own false_positive_rate field -- see
    notes/09_hydraulic_notes.md section 4b's correction for the full story.
    Also: picking WHICHEVER percentile maximizes TEST F1 at all means
    looking at test labels to choose a hyperparameter, the thing this
    project's threshold selection has avoided everywhere else (see
    notes/04_evaluation_notes.md) -- reported only as a labeled comparison,
    per notes/06_extension_ideas.md's original (unbuilt, for CWRU) idea.
    """
    return _row_to_dict(sensitivity_df.loc[sensitivity_df["f1"].idxmax()])


def best_deployable_threshold(sensitivity_df: pd.DataFrame, max_fpr: float = 0.10) -> dict | None:
    """The sweep row that maximizes F1 SUBJECT TO false_positive_rate <=
    max_fpr -- a much more honest "best achievable" number than raw
    f1_optimal_threshold(), since it rules out operating points that would
    be unusable in real deployment regardless of their test-set F1. Still
    carries the same "chosen by looking at test labels" caveat as
    f1_optimal_threshold() -- never deploy this choice directly either,
    it's a ceiling estimate, not a valid selection procedure. Returns None
    if no swept percentile achieves false_positive_rate <= max_fpr at all
    (a real, informative result in itself -- it means no simple percentile
    threshold works for this detector at that FPR tolerance).
    """
    candidates = sensitivity_df[sensitivity_df["false_positive_rate"] <= max_fpr]
    if candidates.empty:
        return None
    return _row_to_dict(candidates.loc[candidates["f1"].idxmax()])


def confound_check(test_predictions: pd.DataFrame) -> pd.DataFrame:
    """Among TARGET-HEALTHY test cycles (label==0), does reconstruction
    error or the false-positive rate rise with how many OTHER components
    are also degraded on that cycle? A real effect here would mean a
    detector is partly reacting to some other component's fault, not just
    its own target -- see notes/09_hydraulic_notes.md section 5's
    "second-order confound" note. Grouped by count, not identity, of other
    degraded components, since which specific ones vary is a much larger
    (and sparser) breakdown.
    """
    healthy = test_predictions[test_predictions["label"] == 0].copy()
    healthy["n_other_degraded"] = healthy["degraded_component"].apply(
        lambda s: 0 if s == "none" else len(s.split(","))
    )
    summary = (
        healthy.groupby("n_other_degraded")
        .agg(
            mean_recon_error=("recon_error", "mean"),
            false_positive_rate=("predicted_anomaly", "mean"),
            n=("n_other_degraded", "size"),
        )
        .reset_index()
    )
    return summary


def raw_time_correlation(catalog: pd.DataFrame, profile_columns: list[str]) -> dict[str, float]:
    """corr(cycle_idx, raw column value) across EVERY cycle in the dataset
    (not just this detector's split) for each of the four condition
    columns -- diagnoses whether a component's condition is inherently
    correlated with campaign time in the raw data itself (a property no
    split strategy can fix, see notes/09_hydraulic_notes.md section 2's
    "cooler's 1.0 still deserves a grain of salt" finding: cooler_condition
    correlates with cycle_idx at r=+0.88 across the full dataset, an order
    of magnitude stronger than the other three).
    """
    condition_cols = [c for c in profile_columns if c != "stable_flag"]
    return {col: float(catalog["cycle_idx"].corr(catalog[col])) for col in condition_cols}
