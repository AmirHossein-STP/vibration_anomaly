"""Service `bearing` -- subproject 3: zero-shot bearing fault DETECTION.

What is deployed is the method that was measured to cross machines:
envelope-order features (src/domain_adaptation/features/envelope.py) fed to a
logistic regression fitted on public labelled data (CWRU), with the decision
threshold set from the operator's own HEALTHY recordings. It is NOT the
spectrogram CNN / DANN of the original experiment -- that did not transfer
(see notes/12 and notes/13, and report R03).

Two ideas to keep in mind when operating it:

* `train` is rare. It refits the source classifier from a folder of labelled
  public CWRU files (`source/`). A fitted copy ships inside the image
  (assets/source_cwru.json), so most operators never run it.
* `calibrate` is the step an operator runs per machine. It reads healthy
  recordings from `train/`, writes the machine's bearing geometry and a
  threshold into a new candidate version, and reports whether the machine's
  healthy data already looks faulty (the "K002/K005" failure mode).

Scope (do not oversell): detection only (healthy vs not healthy), rolling-element
bearings only, geometry and shaft speed must be known. Validated at 900 rpm on
Paderborn (balanced accuracy ~0.82) and weaker at 1500 rpm (~0.69). Fault TYPE
is deliberately not reported -- it does not transfer reliably.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from domain_adaptation.data.transforms import resample_to_rate
from domain_adaptation.features.envelope import (
    CWRU_6205, FEATURE_NAMES, BearingGeometry, envelope_order_features,
)
from serving_core.service import ConfigError, Context, Outcome, Service
from serving_core.signals import InputError, list_signal_files, load_signal, read_sidecar
from vibration_ad.data.windows import make_windows

FS_MODEL = 12000          # every signal is resampled to this before windowing
WIN = 12000               # 1 s: resolves ~0.07 order at 900 rpm
HOP = 6000
ASSET = Path(__file__).parent / "assets" / "source_cwru.json"


# ───────────────────────── features ─────────────────────────
def geometry_from(d: dict) -> BearingGeometry:
    return BearingGeometry(
        name=str(d.get("name", "bearing")), n_balls=int(d["n_balls"]),
        ball_diameter=float(d["ball_diameter"]), pitch_diameter=float(d["pitch_diameter"]),
        contact_angle_deg=float(d.get("contact_angle_deg", 0.0)),
    )


def check_speed(rpm: float, geom: BearingGeometry, band_hz) -> None:
    """The 1 s window and 12 kHz rate limit which speeds are measurable."""
    shaft = rpm / 60.0
    if shaft < 5:
        raise InputError(f"rpm={rpm:g} is too low for 1 s analysis windows (need >= 300 rpm)")
    top = 3 * geom.bpfi * shaft + 2 * shaft
    if top > FS_MODEL / 2 * 0.9:
        raise InputError(f"rpm={rpm:g} is too high: BPFI harmonics ({top:.0f} Hz) would pass the analysis range")


def window_features(signal: np.ndarray, fs_hz: float, rpm: float, geom: BearingGeometry,
                    band_hz, max_windows: int | None) -> tuple[np.ndarray, int]:
    """(features [n, 13], total windows available). Evenly subsamples long files."""
    if fs_hz < 2 * band_hz[1]:
        raise InputError(f"fs={fs_hz:g} Hz is too low for the {band_hz[0]:g}-{band_hz[1]:g} Hz analysis band")
    sig = resample_to_rate(signal, int(round(fs_hz)), FS_MODEL) if int(round(fs_hz)) != FS_MODEL else signal
    wins = make_windows(sig, WIN, HOP)
    if len(wins) == 0:
        raise InputError(f"too short: {len(sig) / FS_MODEL:.2f} s at {FS_MODEL} Hz after resampling, need >= {WIN / FS_MODEL:.0f} s")
    total = len(wins)
    if max_windows and total > max_windows:
        wins = wins[np.linspace(0, total - 1, max_windows).round().astype(int)]
    rows = [envelope_order_features(w, FS_MODEL, rpm, geom, tuple(band_hz)) for w in wins]
    return np.array([[r[n] for n in FEATURE_NAMES] for r in rows], dtype=np.float64), total


class SourceModel:
    """Standardiser + logistic regression, stored as plain JSON (no pickle)."""

    def __init__(self, d: dict):
        self.d = d
        self.mean = np.array(d["mean"]); self.scale = np.array(d["scale"])
        self.coef = np.array(d["coef"]); self.intercept = float(d["intercept"])
        if d["feature_names"] != FEATURE_NAMES:
            raise ValueError("source model was fitted with different feature names than this code version")

    def score(self, X: np.ndarray) -> np.ndarray:
        z = (X - self.mean) / self.scale
        return 1.0 / (1.0 + np.exp(-(z @ self.coef + self.intercept)))


def fit_source(source_dir: Path, per_file: int | None, log=None) -> tuple[dict, dict, list[Path]]:
    """Fit on the labelled CWRU folder. Returns (model dict, metrics, files used)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler

    from domain_adaptation.data.cwru_loader import build_cwru_catalog

    cat = build_cwru_catalog(source_dir, [1730, 1750, 1772], 1797)
    X, y, groups, used = [], [], [], []
    for i, r in enumerate(cat.itertuples()):
        z = np.load(r.filepath)
        sig = np.asarray(z["DE"]).squeeze().astype(np.float64)
        feats, _ = window_features(sig, 12000, float(r.rpm), CWRU_6205, (1500.0, 5500.0), per_file)
        X.append(feats); y += [int(r.universal_label > 0)] * len(feats); groups += [i] * len(feats)
        used.append(Path(r.filepath))
        if log and i % 10 == 0:
            log.info("source features: %d/%d files", i + 1, len(cat))
    X = np.vstack(X); y = np.array(y); groups = np.array(groups)
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(X), y)
    # Honest in-domain check: folds split by FILE, not by window (windows of one
    # recording are near-duplicates and would leak).
    cv = cross_val_score(LogisticRegression(max_iter=2000, class_weight="balanced"),
                         sc.transform(X), y, cv=GroupKFold(n_splits=4), groups=groups, scoring="balanced_accuracy")
    model = {
        "feature_names": FEATURE_NAMES, "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
        "coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0]),
        "source": "CWRU DE12 (SKF 6205), healthy vs inner/outer race faults; ball faults excluded",
        "n_windows": int(len(y)), "n_files": len(used), "n_faulty_windows": int(y.sum()),
    }
    metrics = {"n_windows": int(len(y)), "n_files": len(used), "n_faulty_windows": int(y.sum()),
               "cwru_group_cv_balanced_accuracy_mean": float(cv.mean()), "cwru_group_cv_balanced_accuracy_std": float(cv.std())}
    return model, metrics, used


# ───────────────────────── the service ─────────────────────────
class BearingService(Service):
    name = "bearing"
    description = "Rolling-element bearing fault detection (envelope-order features, healthy-only calibration)."
    extra_columns = ["rpm", "fs_hz", "score_p95", "threshold", "n_windows_flagged"]

    def default_config(self) -> dict:
        return {
            "bearing": {"name": "my-bearing", "n_balls": None, "ball_diameter": None,
                        "pitch_diameter": None, "contact_angle_deg": 0.0},
            "fs_hz": None,          # sampling rate of your files (Hz); a sidecar can override per file
            "rpm": None,            # shaft speed; a sidecar `rpm` overrides per file
            "signal_key": None,     # array name inside .npz files, if they hold several
            "band_hz": [1500.0, 5500.0],   # resonance band for the envelope; fixed a priori in the study
            "calibration": {"target_false_alarm": 0.10, "min_windows": 200,
                            "min_recordings_warn": 3, "max_windows_per_file": 600},
            # MASTER_PLAN WP1 (notes/14): at 0.25, a quarter of held-out healthy recordings
            # read as watch; 0.40 keeps that near 10%. 0.5 kept ~5% healthy recordings faulty.
            "verdict": {"watch_fraction": 0.40, "faulty_fraction": 0.5},
        }

    def validate_config(self, cfg: dict) -> list[str]:
        p = []
        b = cfg["bearing"]
        given = [b.get(k) for k in ("n_balls", "ball_diameter", "pitch_diameter")]
        if any(v is not None for v in given) and any(v is None for v in given):
            p.append("bearing: set n_balls, ball_diameter and pitch_diameter together")
        for k in ("n_balls", "ball_diameter", "pitch_diameter"):
            v = b.get(k)
            if v is not None and (not isinstance(v, (int, float)) or v <= 0):
                p.append(f"bearing.{k} must be a positive number")
        if b.get("n_balls") is not None and b["n_balls"] != int(b["n_balls"]):
            p.append("bearing.n_balls must be a whole number")
        if all(v is not None for v in given) and b["ball_diameter"] >= b["pitch_diameter"]:
            p.append("bearing: ball_diameter must be smaller than pitch_diameter (same unit)")
        band = cfg["band_hz"]
        if not (isinstance(band, list) and len(band) == 2 and 0 < band[0] < band[1] < FS_MODEL / 2):
            p.append(f"band_hz must be [low, high] with 0 < low < high < {FS_MODEL / 2:g}")
        c = cfg["calibration"]
        if not 0 < c["target_false_alarm"] < 0.5:
            p.append("calibration.target_false_alarm must be between 0 and 0.5")
        v = cfg["verdict"]
        if not 0 < v["watch_fraction"] < v["faulty_fraction"] <= 1:
            p.append("verdict: need 0 < watch_fraction < faulty_fraction <= 1")
        return p

    def add_train_args(self, p) -> None:
        p.add_argument("--source-dir", default=None,
                       help="folder of labelled CWRU .npz files (default: <data>/bearing/source)")
        p.add_argument("--windows-per-file", type=int, default=None, help="cap windows per source file")

    def add_calibrate_args(self, p) -> None:
        p.add_argument("--base", default=None,
                       help="version whose source classifier to use (default: bundled CWRU model)")

    # ---- helpers -------------------------------------------------------
    def _file_params(self, path: Path, cfg: dict, fallback: dict | None = None) -> tuple[float, float, str | None]:
        side = read_sidecar(path)
        fs = side.get("fs_hz", cfg["fs_hz"] if cfg["fs_hz"] is not None else (fallback or {}).get("fs_hz"))
        rpm = side.get("rpm", cfg["rpm"] if cfg["rpm"] is not None else (fallback or {}).get("rpm"))
        key = side.get("signal_key", cfg["signal_key"])
        if fs is None:
            raise InputError("sampling rate unknown: set `fs_hz` in config/service.yaml or `fs_hz` in a sidecar file")
        if rpm is None:
            raise InputError("shaft speed unknown: set `rpm` in config/service.yaml or `rpm` in a sidecar file")
        try:
            fs, rpm = float(fs), float(rpm)
        except (TypeError, ValueError):
            raise InputError(f"fs_hz/rpm must be numbers (got {fs!r}, {rpm!r})")
        if fs <= 0 or rpm <= 0:
            raise InputError("fs_hz and rpm must be positive")
        return fs, rpm, key

    # ---- train: refit the source classifier ------------------------------
    def train(self, ctx: Context) -> Outcome:
        src = Path(ctx.args.source_dir) if ctx.args.source_dir else ctx.layout.source_dir
        if not src.is_dir() or not any(src.glob("*.npz")):
            raise InputError(f"no labelled CWRU .npz files in {src}; put them there or pass --source-dir")
        ctx.log.info("fitting source classifier from %s", src)
        model, metrics, files = fit_source(src, ctx.args.windows_per_file, ctx.log)
        (ctx.version_dir / "source_model.json").write_text(json.dumps(model), encoding="utf8")
        cv = metrics["cwru_group_cv_balanced_accuracy_mean"]
        warns = ["This candidate has NO machine calibration yet: run `calibrate --base <this version>` before promoting."]
        if cv < 0.75:
            warns.append(f"in-domain CWRU balanced accuracy is only {cv:.2f} (the bundled model scores ~0.88 with folds split by file)")
        return Outcome(
            "train", metrics,
            [f"source classifier refitted on {metrics['n_files']} CWRU files / {metrics['n_windows']} windows",
             f"CWRU balanced accuracy, folds split by file: {cv:.3f} +- {metrics['cwru_group_cv_balanced_accuracy_std']:.3f}"],
            files, warns, extra_meta={"needs_calibration": True},
        )

    # ---- calibrate: threshold from the machine's healthy recordings ---------
    def _source_json(self, ctx: Context) -> tuple[dict, str | None, list[Path]]:
        base = getattr(ctx.args, "base", None)
        if base:
            from serving_core.registry import Registry
            p = Registry(ctx.layout.models_dir).path(base) / "source_model.json"
            if not p.is_file():
                raise InputError(f"version {base} has no source_model.json")
            return json.loads(p.read_text(encoding="utf8")), base, []
        return json.loads(ASSET.read_text(encoding="utf8")), None, [ASSET]

    def calibrate(self, ctx: Context) -> Outcome:
        cfg = ctx.config
        b = cfg["bearing"]
        if any(b.get(k) is None for k in ("n_balls", "ball_diameter", "pitch_diameter")):
            raise ConfigError("bearing geometry is required: fill bearing.n_balls, ball_diameter and pitch_diameter "
                              f"in {ctx.layout.config_file} (see the bearing's datasheet)")
        geom = geometry_from(b)
        src_json, parent, src_inputs = self._source_json(ctx)
        src = SourceModel(src_json)
        c = cfg["calibration"]
        files = list_signal_files(ctx.layout.train_dir)
        if not files:
            raise InputError(f"no healthy recordings in {ctx.layout.train_dir}")
        scores_by_file: dict[str, np.ndarray] = {}
        feats_all, used, skipped = [], [], []
        for f in files:
            try:
                fs, rpm, key = self._file_params(f, cfg)
                check_speed(rpm, geom, cfg["band_hz"])
                X, _ = window_features(load_signal(f, key), fs, rpm, geom, cfg["band_hz"], c["max_windows_per_file"])
            except InputError as e:
                skipped.append(f"{f.name}: {e}"); ctx.log.warning("skipped %s: %s", f.name, e); continue
            scores_by_file[f.name] = src.score(X); feats_all.append(X); used.append(f)
            ctx.log.info("%s: %d windows", f.name, len(X))
        n_win = sum(len(s) for s in scores_by_file.values())
        if n_win < c["min_windows"]:
            raise InputError(f"only {n_win} usable healthy windows (need >= {c['min_windows']}; 1 window = 1 s of signal). "
                             + (f"Skipped: {skipped}" if skipped else "Add more healthy recordings."))
        allscores = np.concatenate(list(scores_by_file.values()))
        thr = float(np.quantile(allscores, 1 - c["target_false_alarm"]))
        fracs = {n: float((s > thr).mean()) for n, s in scores_by_file.items()}
        watch, faulty = cfg["verdict"]["watch_fraction"], cfg["verdict"]["faulty_fraction"]
        worst = max(fracs.values())

        warns = []
        if skipped:
            warns.append(f"{len(skipped)} file(s) skipped: " + "; ".join(skipped[:3]) + (" ..." if len(skipped) > 3 else ""))
        if len(used) < c["min_recordings_warn"]:
            warns.append(f"only {len(used)} healthy recording(s): false alarms were per-unit in the study, "
                         "so use several healthy recordings (different days / conditions) if you can")
        if worst >= watch:
            bad = [n for n, f in fracs.items() if f >= watch]
            warns.append(f"healthy recording(s) {bad} already reach {worst:.0%} flagged windows (WATCH starts at {watch:.0%}); "
                         "this machine's healthy baseline reads as abnormal. Do NOT rely on the verdicts: add healthy data "
                         "covering all normal operating conditions, or raise verdict.watch_fraction knowingly.")
        warns.append("threshold was set on the same healthy windows it is reported on (in-sample); "
                     "confirm with a LATER healthy recording via `infer` before trusting it")

        cal = {
            "geometry": {"name": geom.name, "n_balls": geom.n_balls, "ball_diameter": geom.ball_diameter,
                         "pitch_diameter": geom.pitch_diameter, "contact_angle_deg": geom.contact_angle_deg},
            "fault_orders": {"bpfo": geom.bpfo, "bpfi": geom.bpfi, "ftf": geom.ftf},
            "band_hz": cfg["band_hz"], "threshold": thr, "target_false_alarm": c["target_false_alarm"],
            "verdict": cfg["verdict"], "max_windows_per_file": c["max_windows_per_file"],
            "default_fs_hz": cfg["fs_hz"], "default_rpm": cfg["rpm"], "signal_key": cfg["signal_key"],
            "healthy_score_quantiles": {str(q): float(np.quantile(allscores, q)) for q in (0.5, 0.9, 0.95, 0.99)},
        }
        (ctx.version_dir / "source_model.json").write_text(json.dumps(src_json), encoding="utf8")
        (ctx.version_dir / "calibration.json").write_text(json.dumps(cal, indent=2), encoding="utf8")
        metrics = {"n_recordings": len(used), "n_windows": n_win, "threshold": thr,
                   "healthy_flagged_fraction_by_file": fracs, "healthy_flagged_fraction_max": worst,
                   "healthy_flagged_fraction_mean": float(np.mean(list(fracs.values()))),
                   "reads_clean": bool(worst < watch), "skipped": skipped}
        summary = [f"calibrated on {len(used)} healthy recording(s), {n_win} windows (1 s each)",
                   f"threshold {thr:.4f} for {c['target_false_alarm']:.0%} window false alarms",
                   f"healthy recordings flagged: mean {metrics['healthy_flagged_fraction_mean']:.0%}, worst {worst:.0%}"
                   f" -> {'reads clean' if worst < watch else 'READS ABNORMAL'}"]
        return Outcome("calibrate", metrics, summary, used + src_inputs, warns, parent=parent,
                       extra_meta={"needs_calibration": False})

    # ---- inference -----------------------------------------------------
    def promote_check(self, version_dir: Path, meta: dict) -> list[str]:
        if not (version_dir / "calibration.json").is_file():
            return ["no calibration.json: this version cannot score files yet -- run `calibrate` (use --base to build on it)"]
        return []

    def load_model(self, version_dir: Path, meta: dict):
        if not (version_dir / "calibration.json").is_file():
            raise InputError("this version has no calibration (run `calibrate`)")
        src = SourceModel(json.loads((version_dir / "source_model.json").read_text(encoding="utf8")))
        cal = json.loads((version_dir / "calibration.json").read_text(encoding="utf8"))
        return {"src": src, "cal": cal, "geom": geometry_from(cal["geometry"])}

    def infer_one(self, model, path: Path, ctx: Context) -> list[dict]:
        cal, geom, src = model["cal"], model["geom"], model["src"]
        fb = {"fs_hz": cal.get("default_fs_hz"), "rpm": cal.get("default_rpm"), "signal_key": cal.get("signal_key")}
        cfg = {"fs_hz": ctx.config["fs_hz"], "rpm": ctx.config["rpm"], "signal_key": ctx.config["signal_key"]}
        fs, rpm, key = self._file_params(path, cfg, fb)
        check_speed(rpm, geom, cal["band_hz"])
        X, total = window_features(load_signal(path, key or fb["signal_key"]), fs, rpm, geom, cal["band_hz"], cal["max_windows_per_file"])
        s = src.score(X)
        flagged = s > cal["threshold"]
        frac = float(flagged.mean())
        v = cal["verdict"]
        verdict = "faulty" if frac >= v["faulty_fraction"] else "watch" if frac >= v["watch_fraction"] else "healthy"
        msg = ""
        if len(s) < total:
            msg = f"scored {len(s)} of {total} windows (evenly spaced)"
        return [{"verdict": verdict, "score": float(np.median(s)), "flagged_fraction": round(frac, 4),
                 "n_windows": int(len(s)), "message": msg, "rpm": rpm, "fs_hz": fs,
                 "score_p95": float(np.percentile(s, 95)), "threshold": cal["threshold"],
                 "n_windows_flagged": int(flagged.sum())}]

    def doctor(self, layout, cfg):
        out = []
        out.append(("ok" if ASSET.is_file() else "fail", "bundled source model", str(ASSET)))
        b = cfg["bearing"]
        if any(b.get(k) is None for k in ("n_balls", "ball_diameter", "pitch_diameter")):
            out.append(("warn", "bearing geometry", "not filled in config/service.yaml yet (needed for `calibrate`)"))
        else:
            g = geometry_from(b)
            out.append(("ok", "bearing geometry", f"BPFO {g.bpfo:.3f}, BPFI {g.bpfi:.3f}, FTF {g.ftf:.3f} (orders of shaft speed)"))
        if cfg["fs_hz"] is None:
            out.append(("warn", "fs_hz", "not set (each file then needs a sidecar with fs_hz)"))
        return out
