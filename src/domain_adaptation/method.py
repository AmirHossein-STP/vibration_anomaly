"""Method v1.0 (notes/22): the one implementation of the bearing method,
used by the analysis scripts and by the serving_core `bearing` service.

    detect    healthy-referenced physics score: max over the six fault lines
              (BPFO, BPFI x 1-3) of the z-score of log(line) against the
              UNIT'S OWN healthy windows (CPW + squared envelope spectrum)
    threshold conformal quantile of CROSS-FITTED calibration scores
    trigger   one-sided CUSUM (k = 0.5) on a standardised series per unit
    diagnose  typing: the race whose lines rose most against baseline

Evidence and every number behind each choice: notes/16-21. Nothing here is
fitted to fault data; the only data any function sees is healthy baseline.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from domain_adaptation.features.envelope import BearingGeometry, envelope_order_features

METHOD_VERSION = "1.1"   # 1.1: geometry-free detection + six-line typing; conditioned level trigger (notes/22 §7a)
LINES = ["bpfo_h1", "bpfo_h2", "bpfo_h3", "bpfi_h1", "bpfi_h2", "bpfi_h3"]
CPW_SES = {"whiten": True, "band_hz": None, "squared": True}   # envelope_cache variant hd_cpw_ses
FS = 12000
WIN, HOP = 12000, 6000            # 1 s windows, hop 0.5 s, at 12 kHz
ALPHA = 0.10
CUSUM_K = 0.5
EPS_LINE = 1e-9
INNER, OUTER = 1, 2


def conformal_threshold(cal: np.ndarray, alpha: float = ALPHA) -> float:
    """ceil((n+1)(1-alpha))-th smallest calibration score; +inf when n is too small."""
    s = np.sort(np.asarray(cal, dtype=np.float64))
    k = math.ceil((len(s) + 1) * (1 - alpha))
    return float("inf") if k > len(s) else float(s[k - 1])


def log_lines(lines: np.ndarray) -> np.ndarray:
    """(n, 6) line ratios in LINES order -> log."""
    return np.log(np.asarray(lines, dtype=np.float64) + EPS_LINE)


def default_groups(recording: np.ndarray, t0: np.ndarray) -> np.ndarray:
    """Cross-fitting groups: one per recording; inside a single recording,
    three contiguous thirds (overlapping windows never score each other)."""
    recording = np.asarray(recording)
    if len(np.unique(recording)) >= 2:
        return recording
    rank = np.argsort(np.argsort(np.asarray(t0)))
    return np.minimum((rank * 3) // len(rank), 2)


@dataclass
class Baseline:
    mu: np.ndarray            # mean log line, (6,)
    sd: np.ndarray            # sd of log line, (6,)
    cal_scores: np.ndarray    # cross-fitted scores of the calibration windows
    n_windows: int
    meta: dict = field(default_factory=dict)

    def threshold(self, alpha: float = ALPHA) -> float:
        return conformal_threshold(self.cal_scores, alpha)

    def z(self, lines: np.ndarray) -> np.ndarray:
        return (log_lines(lines) - self.mu) / self.sd

    def score(self, lines: np.ndarray) -> np.ndarray:
        return self.z(lines).max(1)

    def typing(self, lines: np.ndarray) -> np.ndarray:
        """1 = inner race, 2 = outer race, per window."""
        z = self.z(lines)
        return np.where(z[:, 3:].max(1) > z[:, :3].max(1), INNER, OUTER)

    def to_dict(self) -> dict:
        return {"method_version": METHOD_VERSION, "mu": self.mu.tolist(), "sd": self.sd.tolist(),
                "cal_scores": self.cal_scores.tolist(), "n_windows": self.n_windows, "meta": self.meta}

    @classmethod
    def from_dict(cls, d: dict) -> "Baseline":
        if d.get("method_version") != METHOD_VERSION:
            raise ValueError(f"baseline is method {d.get('method_version')!r}, this code is {METHOD_VERSION}")
        return cls(np.asarray(d["mu"]), np.asarray(d["sd"]), np.asarray(d["cal_scores"]), int(d["n_windows"]),
                   d.get("meta", {}))


def fit_baseline(lines: np.ndarray, groups: np.ndarray, meta: dict | None = None) -> Baseline:
    """Healthy windows of ONE unit (or of the reference set) -> Baseline."""
    lc = log_lines(lines)
    groups = np.asarray(groups)
    if len(np.unique(groups)) < 2:
        raise ValueError("cross-fitting needs at least 2 groups (recordings or time blocks)")
    cal = np.empty(len(lc))
    for g in np.unique(groups):
        m = groups == g
        mu, sd = lc[~m].mean(0), lc[~m].std(0) + 1e-12
        cal[m] = ((lc[m] - mu) / sd).max(1)
    return Baseline(lc.mean(0), lc.std(0) + 1e-12, cal, len(lc), meta or {})


def window_lines(signal: np.ndarray, fs_hz: float, rpm: float, geometry: BearingGeometry,
                 win: int = WIN, hop: int = HOP) -> tuple[np.ndarray, np.ndarray]:
    """Constant-speed path on a 12 kHz signal -> ((n, 6) lines, window start times s)."""
    rows, t0 = [], []
    for s in range(0, len(signal) - win + 1, hop):
        f = envelope_order_features(signal[s:s + win], fs_hz, rpm, geometry, **CPW_SES)
        rows.append([f[k] for k in LINES])
        t0.append(s / fs_hz)
    return np.asarray(rows).reshape(-1, len(LINES)), np.asarray(t0)


# --- trend trigger ------------------------------------------------------------

def standardise(series: np.ndarray, cal: np.ndarray) -> np.ndarray:
    cal = np.asarray(cal, dtype=np.float64)
    return (np.asarray(series, dtype=np.float64) - cal.mean()) / (cal.std(ddof=1) + 1e-12)


def cusum(u: np.ndarray, k: float = CUSUM_K, start: int = 0) -> np.ndarray:
    """One-sided CUSUM C_t = max(0, C_{t-1} + u_t - k) from index `start`; NaN before."""
    out = np.full(len(u), np.nan)
    c = 0.0
    for i in range(start, len(u)):
        c = max(0.0, c + float(u[i]) - k)
        out[i] = c
    return out


def ewma(u: np.ndarray, lam: float = 0.2, start: int = 0) -> np.ndarray:
    out = np.full(len(u), np.nan)
    e = 0.0
    for i in range(start, len(u)):
        e = lam * float(u[i]) + (1 - lam) * e
        out[i] = e
    return out


# --- level conditioned on the operating point (notes/25, v1.1) ----------------

MIN_RECORDINGS_CONDITIONED = 10


def fit_level_model(levels: np.ndarray, state: np.ndarray) -> dict:
    """log-level per recording ~ [1, log rpm, loads...] on the baseline recordings.
    Falls back to the pooled level (intercept only) with < 10 recordings or no spread."""
    levels = np.asarray(levels, dtype=np.float64)
    X = np.c_[np.ones(len(levels)), np.asarray(state, dtype=np.float64).reshape(len(levels), -1)]
    keep = [0] + [j for j in range(1, X.shape[1]) if np.std(X[:, j]) > 1e-9]
    if len(levels) < MIN_RECORDINGS_CONDITIONED or len(keep) == 1:
        keep = [0]
    coef = np.zeros(X.shape[1])
    coef[keep] = np.linalg.lstsq(X[:, keep], levels, rcond=None)[0]
    resid = levels - X @ coef
    sd = float(resid.std(ddof=1)) if len(levels) > 1 else 0.0
    return {"coef": coef.tolist(), "resid_sd": sd + 1e-12, "conditioned": len(keep) > 1}


def level_z(model: dict, level: float, state: np.ndarray) -> float:
    x = np.r_[1.0, np.asarray(state, dtype=np.float64).ravel()]
    return float((level - x @ np.asarray(model["coef"])) / model["resid_sd"])
