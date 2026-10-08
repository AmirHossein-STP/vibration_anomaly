"""Service `vibration` -- subproject 1: healthy-only autoencoder anomaly detection.

The operator drops HEALTHY recordings of one machine into train/; `train` fits
the convolutional autoencoder of `vibration_ad` on them, sets the threshold
from held-out healthy windows (99th percentile of reconstruction error), and
reports a candidate. `calibrate` re-sets only the threshold from newer healthy
files, keeping the weights.

Hard facts from the study that shape this wrapper:
* The model learned *what the training machine sounds like*; it does not move
  between machines. One deployment = one machine = its own healthy data.
* The threshold in the study came from only 56 validation windows and gave
  3.5% false alarms instead of the nominal 1%. Hence the minimum amounts of
  healthy data below, and the several-seed spread report.
* CWRU numbers (AUC 1.0) say nothing about a real machine until it has been
  checked against that machine's own data.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from domain_adaptation.data.transforms import resample_to_rate
from serving_core.registry import Registry
from serving_core.service import ConfigError, Context, Outcome, Service
from serving_core.signals import InputError, list_signal_files, load_signal, read_sidecar
from vibration_ad.config import STFTConfig, WindowingConfig
from vibration_ad.data.torch_dataset import SpectrogramDataset
from vibration_ad.data.windows import make_windows
from vibration_ad.features.spectrogram import SpectrogramNormalizer, expected_shape, windows_to_log_spectrograms
from vibration_ad.models.autoencoder import ConvAutoencoder, per_sample_mse, select_device
from vibration_ad.train import run_epoch, set_seed

FS_MODEL = 12000
WINDOWING = WindowingConfig(segment_length=8384, train_stride=512, eval_stride=2096)
STFT = STFTConfig(nperseg=256, hop_length=64, log_epsilon=1e-10)
SHAPE = expected_shape(WINDOWING, STFT)


def _windows(signal: np.ndarray, fs_hz: float, stride: int, max_windows: int | None = None) -> np.ndarray:
    fs = int(round(fs_hz))
    sig = resample_to_rate(signal, fs, FS_MODEL) if fs != FS_MODEL else signal
    w = make_windows(sig, WINDOWING.segment_length, stride)
    if len(w) == 0:
        raise InputError(f"too short: {len(sig) / FS_MODEL:.2f} s at {FS_MODEL} Hz after resampling, "
                         f"need >= {WINDOWING.segment_length / FS_MODEL:.2f} s")
    if max_windows and len(w) > max_windows:
        w = w[np.linspace(0, len(w) - 1, max_windows).round().astype(int)]
    return w


def _spectrograms(windows: np.ndarray) -> np.ndarray:
    return windows_to_log_spectrograms(windows, FS_MODEL, STFT)


@torch.no_grad()
def reconstruction_errors(model, norm: SpectrogramNormalizer, spectrograms: np.ndarray, device) -> np.ndarray:
    model.eval()
    X = torch.from_numpy(norm.transform(spectrograms)).unsqueeze(1).float()
    out = []
    for i in range(0, len(X), 64):
        xb = X[i:i + 64].to(device)
        out.append(per_sample_mse(xb, model(xb)).cpu().numpy())
    return np.concatenate(out)


class VibrationService(Service):
    name = "vibration"
    description = "Healthy-only autoencoder anomaly detection for one machine's vibration signal."
    extra_columns = ["fs_hz", "score_p95", "threshold", "n_windows_flagged"]

    def default_config(self) -> dict:
        return {
            "fs_hz": None,            # sampling rate of your files (Hz); sidecar `fs_hz` overrides per file
            "signal_key": None,       # array name inside .npz files, if they hold several
            "model": {"latent_dim": 128, "base_channels": 16},
            "train": {"epochs": 60, "batch_size": 32, "learning_rate": 1.0e-3, "weight_decay": 1.0e-5,
                      "early_stopping_patience": 10, "validation_fraction": 0.2,
                      "max_train_windows": 20000, "device": "cpu"},
            "calibration": {"threshold_percentile": 99.0, "min_train_windows": 500,
                            "min_validation_windows": 50, "max_windows_per_file": 400},
            "verdict": {"watch_fraction": 0.10, "faulty_fraction": 0.30},
        }

    def validate_config(self, cfg: dict) -> list[str]:
        p = []
        if cfg["fs_hz"] is not None and (not isinstance(cfg["fs_hz"], (int, float)) or cfg["fs_hz"] < 2 * 1000):
            p.append("fs_hz must be a number >= 2000")
        t = cfg["train"]
        if not 0.05 <= t["validation_fraction"] <= 0.5:
            p.append("train.validation_fraction must be between 0.05 and 0.5")
        if t["epochs"] < 1 or t["batch_size"] < 1:
            p.append("train.epochs and train.batch_size must be >= 1")
        if not 50 <= cfg["calibration"]["threshold_percentile"] < 100:
            p.append("calibration.threshold_percentile must be in [50, 100)")
        v = cfg["verdict"]
        if not 0 < v["watch_fraction"] < v["faulty_fraction"] <= 1:
            p.append("verdict: need 0 < watch_fraction < faulty_fraction <= 1")
        return p

    def add_train_args(self, p) -> None:
        p.add_argument("--seeds", type=int, default=3, help="train this many seeds; keep the best (default 3)")
        p.add_argument("--epochs", type=int, default=None, help="override train.epochs (testing)")

    def add_calibrate_args(self, p) -> None:
        p.add_argument("--base", default=None, help="version whose weights to keep (default: the active one)")

    def _fs(self, path: Path, cfg: dict, fallback_fs=None):
        side = read_sidecar(path)
        fs = side.get("fs_hz", cfg["fs_hz"] if cfg["fs_hz"] is not None else fallback_fs)
        if fs is None:
            raise InputError("sampling rate unknown: set `fs_hz` in config/service.yaml or in a sidecar file")
        try:
            fs = float(fs)
        except (TypeError, ValueError):
            raise InputError(f"fs_hz must be a number (got {fs!r})")
        if fs < 2000:
            raise InputError(f"fs_hz={fs:g} is too low (need >= 2000 Hz)")
        return fs, side.get("signal_key", cfg["signal_key"])

    # ---- train ---------------------------------------------------------
    def _split_files(self, ctx: Context):
        cfg = ctx.config; t = cfg["train"]; c = cfg["calibration"]
        files = list_signal_files(ctx.layout.train_dir)
        if not files:
            raise InputError(f"no healthy recordings in {ctx.layout.train_dir}")
        tr, va, used, skipped = [], [], [], []
        gap = WINDOWING.segment_length
        for f in files:
            try:
                fs, key = self._fs(f, cfg)
                sig = load_signal(f, key)
                fs_i = int(round(fs))
                sig = resample_to_rate(sig, fs_i, FS_MODEL) if fs_i != FS_MODEL else sig
                n = len(sig); cut = int(n * (1 - t["validation_fraction"]))
                # time-ordered split with a one-window gap: windows overlap, so a
                # random split would leak near-duplicates into validation.
                a, b = sig[:cut], sig[cut + gap:]
                if len(a) < WINDOWING.segment_length or len(b) < WINDOWING.segment_length:
                    raise InputError(f"too short to split ({n / FS_MODEL:.1f} s); need roughly >= 3 s")
                tr.append(make_windows(a, WINDOWING.segment_length, WINDOWING.train_stride))
                va.append(make_windows(b, WINDOWING.segment_length, WINDOWING.eval_stride))
                used.append(f)
            except InputError as e:
                skipped.append(f"{f.name}: {e}"); ctx.log.warning("skipped %s: %s", f.name, e)
        if not used:
            raise InputError("no usable healthy recordings. " + "; ".join(skipped[:3]))
        Wtr, Wva = np.concatenate(tr), np.concatenate(va)
        if len(Wtr) < c["min_train_windows"] or len(Wva) < c["min_validation_windows"]:
            raise InputError(
                f"not enough healthy data: {len(Wtr)} training windows (need {c['min_train_windows']}) and "
                f"{len(Wva)} validation windows (need {c['min_validation_windows']}). One window = {WINDOWING.segment_length / FS_MODEL:.2f} s; "
                "add longer / more healthy recordings covering all normal operating conditions.")
        if len(Wtr) > t["max_train_windows"]:
            Wtr = Wtr[np.linspace(0, len(Wtr) - 1, t["max_train_windows"]).round().astype(int)]
        return Wtr, Wva, used, skipped

    def train(self, ctx: Context) -> Outcome:
        cfg = ctx.config; t = cfg["train"]; m = cfg["model"]; c = cfg["calibration"]
        epochs = ctx.args.epochs or t["epochs"]
        Wtr, Wva, used, skipped = self._split_files(ctx)
        ctx.log.info("windows: train %d, validation %d (from %d files)", len(Wtr), len(Wva), len(used))
        Str, Sva = _spectrograms(Wtr), _spectrograms(Wva)
        norm = SpectrogramNormalizer().fit(Str)
        Xtr, Xva = norm.transform(Str), norm.transform(Sva)
        device = select_device(t["device"])
        per_seed = []
        for seed in range(ctx.args.seeds):
            set_seed(seed)
            model = ConvAutoencoder(SHAPE, m["base_channels"], m["latent_dim"]).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
            tl = DataLoader(SpectrogramDataset(Xtr), batch_size=t["batch_size"], shuffle=True)
            vl = DataLoader(SpectrogramDataset(Xva), batch_size=t["batch_size"], shuffle=False)
            best, best_state, bad = float("inf"), None, 0
            for ep in range(1, epochs + 1):
                trl = run_epoch(model, tl, opt, device, train=True)
                val = run_epoch(model, vl, opt, device, train=False)
                if val < best:
                    best, bad = val, 0
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else:
                    bad += 1
                ctx.log.info("seed %d epoch %3d train=%.4f val=%.4f%s", seed, ep, trl, val, " *" if bad == 0 else "")
                if bad >= t["early_stopping_patience"]:
                    break
            model.load_state_dict(best_state)
            errs = reconstruction_errors(model, norm, Sva, device)
            thr = float(np.percentile(errs, c["threshold_percentile"]))
            per_seed.append({"seed": seed, "best_val_mse": best, "epochs_run": ep, "threshold": thr, "state": best_state, "val_errors": errs})
        winner = min(per_seed, key=lambda r: r["best_val_mse"])
        thr_all = np.array([r["threshold"] for r in per_seed])
        spread = float(thr_all.std() / thr_all.mean()) if len(per_seed) > 1 else None

        torch.save(winner["state"], ctx.version_dir / "model.pt")
        norm.save(ctx.version_dir / "normalizer.npz")
        cal = self._calibration_json(cfg, winner["threshold"], winner["val_errors"], len(Wtr), len(Wva), "validation (time-ordered tail of each file)")
        (ctx.version_dir / "calibration.json").write_text(json.dumps(cal, indent=2), encoding="utf8")

        warns = []
        if len(per_seed) == 1:
            warns.append("trained with a single seed: the study found seed-to-seed spread can exceed the effects being compared; use --seeds 3 or more")
        elif spread is not None and spread > 0.25:
            warns.append(f"threshold differs a lot between seeds (relative spread {spread:.0%}); the model is unstable on this much data")
        if skipped:
            warns.append(f"{len(skipped)} file(s) skipped: " + "; ".join(skipped[:3]))
        warns.append("the threshold comes from the same healthy machine's later signal, not from any fault data; "
                     "recall on real faults is UNMEASURED until a fault (or an injected fault) is seen")
        metrics = {"n_files": len(used), "n_train_windows": int(len(Wtr)), "n_validation_windows": int(len(Wva)),
                   "winner_seed": winner["seed"], "threshold": winner["threshold"], "threshold_percentile": c["threshold_percentile"],
                   "threshold_relative_spread_across_seeds": spread,
                   "per_seed": [{k: v for k, v in r.items() if k in ("seed", "best_val_mse", "epochs_run", "threshold")} for r in per_seed],
                   "validation_error_quantiles": cal["healthy_score_quantiles"], "skipped": skipped}
        summary = [f"autoencoder trained on {len(used)} healthy file(s): {len(Wtr)} train / {len(Wva)} validation windows",
                   f"{len(per_seed)} seed(s); best validation MSE {winner['best_val_mse']:.4f} (seed {winner['seed']})",
                   f"threshold {winner['threshold']:.4f} = P{c['threshold_percentile']:g} of validation error"
                   + (f"; spread across seeds {spread:.0%}" if spread is not None else "")]
        return Outcome("train", metrics, summary, used, warns)

    def _calibration_json(self, cfg, thr, errs, n_tr, n_va, how) -> dict:
        return {"threshold": float(thr), "threshold_percentile": cfg["calibration"]["threshold_percentile"],
                "threshold_source": how, "n_train_windows": int(n_tr), "n_threshold_windows": int(n_va),
                "verdict": cfg["verdict"], "max_windows_per_file": cfg["calibration"]["max_windows_per_file"],
                "default_fs_hz": cfg["fs_hz"], "signal_key": cfg["signal_key"],
                "model": cfg["model"],
                "healthy_score_quantiles": {str(q): float(np.quantile(errs, q)) for q in (0.5, 0.9, 0.95, 0.99)}}

    # ---- calibrate: new threshold, same weights -----------------------------
    def calibrate(self, ctx: Context) -> Outcome:
        cfg = ctx.config; c = cfg["calibration"]
        base = ctx.args.base or ctx.active_version
        if not base:
            raise InputError("no base model: pass --base <version> or promote one first")
        bdir = Registry(ctx.layout.models_dir).path(base)
        base_cal = json.loads((bdir / "calibration.json").read_text(encoding="utf8"))
        norm = SpectrogramNormalizer.load(bdir / "normalizer.npz")
        device = select_device(cfg["train"]["device"])
        model = ConvAutoencoder(SHAPE, base_cal["model"]["base_channels"], base_cal["model"]["latent_dim"]).to(device)
        model.load_state_dict(torch.load(bdir / "model.pt", map_location=device, weights_only=True))

        files = list_signal_files(ctx.layout.train_dir)
        if not files:
            raise InputError(f"no healthy recordings in {ctx.layout.train_dir}")
        errs_by_file, skipped, used = {}, [], []
        for f in files:
            try:
                fs, key = self._fs(f, cfg, base_cal.get("default_fs_hz"))
                W = _windows(load_signal(f, key or base_cal.get("signal_key")), fs, WINDOWING.eval_stride, c["max_windows_per_file"])
                errs_by_file[f.name] = reconstruction_errors(model, norm, _spectrograms(W), device); used.append(f)
            except InputError as e:
                skipped.append(f"{f.name}: {e}"); ctx.log.warning("skipped %s: %s", f.name, e)
        allerr = np.concatenate(list(errs_by_file.values())) if errs_by_file else np.array([])
        if len(allerr) < c["min_validation_windows"]:
            raise InputError(f"only {len(allerr)} usable healthy windows (need >= {c['min_validation_windows']})")
        thr = float(np.percentile(allerr, c["threshold_percentile"]))
        fracs = {n: float((e > thr).mean()) for n, e in errs_by_file.items()}
        worst = max(fracs.values()); watch = cfg["verdict"]["watch_fraction"]
        shutil.copy(bdir / "model.pt", ctx.version_dir / "model.pt")
        shutil.copy(bdir / "normalizer.npz", ctx.version_dir / "normalizer.npz")
        cal = self._calibration_json(cfg, thr, allerr, base_cal["n_train_windows"], len(allerr), f"calibrate on {len(used)} healthy file(s)")
        cal["model"] = base_cal["model"]
        (ctx.version_dir / "calibration.json").write_text(json.dumps(cal, indent=2), encoding="utf8")
        warns = []
        if worst >= watch:
            warns.append(f"healthy recording(s) already reach {worst:.0%} flagged windows (WATCH starts at {watch:.0%}): "
                         "the weights do not describe this healthy data well -- consider `train` on it instead")
        if skipped:
            warns.append(f"{len(skipped)} file(s) skipped: " + "; ".join(skipped[:3]))
        warns.append("threshold was set on the same healthy windows it is reported on (in-sample)")
        metrics = {"n_recordings": len(used), "n_windows": int(len(allerr)), "threshold": thr, "base": base,
                   "old_threshold": base_cal["threshold"], "healthy_flagged_fraction_by_file": fracs,
                   "healthy_flagged_fraction_max": worst, "reads_clean": bool(worst < watch), "skipped": skipped}
        return Outcome("calibrate", metrics,
                       [f"threshold re-set on {len(used)} healthy file(s) / {len(allerr)} windows, weights from {base}",
                        f"threshold {base_cal['threshold']:.4f} -> {thr:.4f}",
                        f"healthy recordings flagged: worst {worst:.0%} -> {'reads clean' if worst < watch else 'READS ABNORMAL'}"],
                       used, warns, parent=base)

    # ---- inference -----------------------------------------------------
    def load_model(self, version_dir: Path, meta: dict):
        cal = json.loads((version_dir / "calibration.json").read_text(encoding="utf8"))
        device = torch.device("cpu")
        model = ConvAutoencoder(SHAPE, cal["model"]["base_channels"], cal["model"]["latent_dim"]).to(device)
        model.load_state_dict(torch.load(version_dir / "model.pt", map_location=device, weights_only=True))
        model.eval()
        return {"model": model, "norm": SpectrogramNormalizer.load(version_dir / "normalizer.npz"), "cal": cal, "device": device}

    def infer_one(self, m, path: Path, ctx: Context) -> list[dict]:
        cal = m["cal"]
        fs, key = self._fs(path, ctx.config, cal.get("default_fs_hz"))
        W = _windows(load_signal(path, key or cal.get("signal_key")), fs, WINDOWING.eval_stride, cal["max_windows_per_file"])
        errs = reconstruction_errors(m["model"], m["norm"], _spectrograms(W), m["device"])
        flagged = errs > cal["threshold"]
        frac = float(flagged.mean()); v = cal["verdict"]
        verdict = "faulty" if frac >= v["faulty_fraction"] else "watch" if frac >= v["watch_fraction"] else "healthy"
        return [{"verdict": verdict, "score": float(np.median(errs)), "flagged_fraction": round(frac, 4),
                 "n_windows": int(len(errs)), "message": "", "fs_hz": fs, "score_p95": float(np.percentile(errs, 95)),
                 "threshold": cal["threshold"], "n_windows_flagged": int(flagged.sum())}]

    def doctor(self, layout, cfg):
        n = len(list_signal_files(layout.train_dir))
        return [("ok" if n else "warn", "healthy files in train/", f"{n} file(s)")]
