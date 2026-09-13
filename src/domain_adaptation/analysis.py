"""Pure-numpy/sklearn analysis helpers (no torch), so they can be unit
tested locally -- same split as vibration_ad.analysis / hydraulic_fusion_ad.analysis,
where analysis.py holds the computation and evaluate.py/analyze.py just
call it.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def domain_separability_probe(
    source_features: np.ndarray, target_features: np.ndarray, seed: int = 42, test_size: float = 0.3
) -> dict:
    """How easily can a FRESH classifier tell source features from target
    features? This is the standard proxy-A-distance measurement, and it's
    the diagnostic that disambiguates the one thing the training log cannot.

    During DANN training the domain discriminator's loss sitting at chance
    (ln(3) = 1.099 for 3 domains) has two completely different possible
    meanings:
      1. The features really are domain-invariant -- adaptation worked.
      2. The discriminator is just losing the minimax race and has learned
         nothing -- adaptation did nothing, and the flat loss is an illusion.
    A fresh probe trained on FROZEN features, with no gradient reversal
    fighting it, can't be fooled that way: if it separates the domains
    easily, the features were never aligned, whatever the training log said.

    Returns accuracy on a held-out split plus the chance baseline. Classes
    are balanced by subsampling the larger side first, so chance is always
    0.5 and the number is directly readable:
        ~0.5 -> domains genuinely indistinguishable (aligned)
        ~1.0 -> trivially separable (not aligned at all)
    """
    rng = np.random.default_rng(seed)
    n = min(len(source_features), len(target_features))
    if n < 10:
        raise ValueError(f"Need >=10 samples per domain for a meaningful probe, got {n}")

    source_sample = source_features[rng.choice(len(source_features), size=n, replace=False)]
    target_sample = target_features[rng.choice(len(target_features), size=n, replace=False)]

    X = np.concatenate([source_sample, target_sample], axis=0)
    y = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    probe = LogisticRegression(max_iter=2000)
    probe.fit(X_train, y_train)
    accuracy = float(probe.score(X_test, y_test))

    return {
        "separability_accuracy": accuracy,
        "chance": 0.5,
        # Proxy A-distance, the usual summary of this measurement:
        # 2 * (1 - 2*error) -- 0 means indistinguishable, 2 means trivially separable.
        "proxy_a_distance": float(2.0 * (1.0 - 2.0 * (1.0 - accuracy))),
        "n_per_domain": int(n),
    }


def binary_detection_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Collapse the 3-class prediction to healthy (0) vs faulty (1 or 2) and
    score it as a detector.

    Why this belongs alongside the 3-class numbers rather than in a
    throwaway script: the deployment question this whole project exists to
    answer is "the machine only ever gave us healthy data -- will we notice
    when it breaks?", which is binary. Fault *typing* is a bonus. Across
    kernels v1-v5 the 3-class task never transferred while detection partly
    did, so reporting only the 3-class figure understates the result and
    reporting only accuracy overstates it.

    `false_alarm_rate` is the number to lead with, not accuracy: TARGET_TEST
    is ~90% faulty, so "always say faulty" scores 0.90 accuracy while being
    useless. False-alarm rate is measured on healthy windows only, so the
    test set's skew cannot flatter it, and in condition monitoring it is the
    figure that decides whether anyone keeps listening to the alarm.
    """
    y_true_bin = (np.asarray(y_true) != 0).astype(int)
    y_pred_bin = (np.asarray(y_pred) != 0).astype(int)

    tp = int(((y_pred_bin == 1) & (y_true_bin == 1)).sum())
    fp = int(((y_pred_bin == 1) & (y_true_bin == 0)).sum())
    tn = int(((y_pred_bin == 0) & (y_true_bin == 0)).sum())
    fn = int(((y_pred_bin == 0) & (y_true_bin == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    n_healthy, n_faulty = tn + fp, tp + fn

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float((tp + tn) / len(y_true_bin)) if len(y_true_bin) else 0.0,
        # Measured on healthy windows only -- immune to the test set's skew.
        "false_alarm_rate": float(fp / n_healthy) if n_healthy else 0.0,
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "n_healthy": n_healthy,
        "n_faulty": n_faulty,
        # What a model that always shouts "faulty" would score, so the
        # headline number can never be read without its null comparison.
        "always_faulty_accuracy": float(n_faulty / len(y_true_bin)) if len(y_true_bin) else 0.0,
    }
