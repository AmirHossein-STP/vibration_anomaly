"""Method v1.0 (src/domain_adaptation/method.py, notes/22) invariants."""
from __future__ import annotations

import numpy as np
import pytest

from domain_adaptation.method import (
    INNER, LINES, METHOD_VERSION, OUTER, Baseline, conformal_threshold, cusum, default_groups, fit_baseline,
)


def _healthy(n=60, seed=0):
    rng = np.random.default_rng(seed)
    return np.exp(rng.normal(np.log(0.01), 0.2, size=(n, len(LINES))))


def test_conformal_threshold_matches_definition():
    s = np.arange(1, 20, dtype=float)
    assert conformal_threshold(s, 0.10) == 18.0           # ceil(20 * 0.9) = 18th smallest
    assert conformal_threshold(np.arange(5.0), 0.10) == float("inf")


def test_crossfit_scores_are_out_of_sample():
    lines = _healthy()
    groups = np.repeat(np.arange(6), 10)
    b = fit_baseline(lines, groups)
    # recompute group 0's scores by hand without it
    lc = np.log(lines + 1e-9)
    m = groups == 0
    mu, sd = lc[~m].mean(0), lc[~m].std(0) + 1e-12
    assert np.allclose(b.cal_scores[m], ((lc[m] - mu) / sd).max(1))
    # the deployed scorer is fit on all windows
    assert np.allclose(b.score(lines[:3]), ((lc[:3] - lc.mean(0)) / (lc.std(0) + 1e-12)).max(1))


def test_crossfit_threshold_controls_false_alarms_with_small_baselines():
    """notes/17 §5.2: with ~9-20 windows, in-sample thresholds overshoot alpha; cross-fitted ones do not."""
    fa_xf, fa_in = [], []
    for seed in range(200):
        lines = _healthy(n=18, seed=seed)
        new = _healthy(n=200, seed=10_000 + seed)
        b = fit_baseline(lines, default_groups(np.zeros(18), np.arange(18)))
        fa_xf.append((b.score(new) > b.threshold()).mean())
        fa_in.append((b.score(new) > conformal_threshold(b.score(lines))).mean())
    assert np.mean(fa_in) > 0.10 >= np.mean(fa_xf)          # measured: 0.129 vs 0.039


def test_typing_reads_the_risen_race():
    b = fit_baseline(_healthy(), np.repeat(np.arange(6), 10))
    inner = _healthy(5, seed=1); inner[:, 3] *= 30          # bpfi_h1 rises
    outer = _healthy(5, seed=2); outer[:, 0] *= 30          # bpfo_h1 rises
    assert (b.typing(inner) == INNER).all() and (b.typing(outer) == OUTER).all()
    assert (b.score(inner) > b.threshold()).all()


def test_default_groups_single_recording_uses_thirds():
    g = default_groups(np.array(["r"] * 9), np.arange(9) * 0.5)
    assert list(g) == [0, 0, 0, 1, 1, 1, 2, 2, 2]


def test_baseline_roundtrip_and_version_guard():
    b = fit_baseline(_healthy(), np.repeat(np.arange(6), 10))
    d = b.to_dict()
    assert Baseline.from_dict(d).threshold() == b.threshold()
    d["method_version"] = "0.9"
    with pytest.raises(ValueError):
        Baseline.from_dict(d)
    assert METHOD_VERSION == "1.0"


def test_cusum_accumulates_only_above_k():
    u = np.array([0, 0, 0.4, 2, 2, -5, 1.5])
    c = cusum(u, k=0.5, start=1)
    assert np.isnan(c[0]) and c[2] == 0.0 and c[4] == pytest.approx(3.0) and c[5] == 0.0
