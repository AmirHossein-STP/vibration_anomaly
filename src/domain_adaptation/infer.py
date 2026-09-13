"""Phase 5: run a trained detector on a NEW machine's vibration data.

This is the deployment path the original plan asks for, and it is
deliberately built around what kernels v1-v5 actually established rather
than what they set out to prove:

* It loads the **baseline** model (`train_baseline.py`), not the adapted
  one. Across five runs the DANN model was never better and was usually
  far worse -- v5's adapted model called 91% of everything healthy. There
  is no adaptation step to perform here, which also makes deployment
  simpler: no discriminator, no GRL, no target-domain training.
* It reports **healthy vs. faulty**, not the fault type. Fault typing
  never transferred to an unseen machine/speed in any version; detection
  did. See DOMAIN_ADAPTATION_PLAN.md's verdict section.
* It has a **calibration mode**, because the dominant error mode is
  per-unit, not random: on the target condition, every false alarm came
  from ONE of six healthy bearing units (K002: 8/8 recordings, 100% of
  its windows), while the other five were essentially clean. A machine
  whose healthy baseline reads as faulty will do so consistently -- which
  means you can detect that with healthy data alone, before ever seeing a
  fault, and correct for it.

Typical use on a real machine:

    # 1. Calibrate on a recording you KNOW is healthy.
    python -m domain_adaptation.infer --input baseline_healthy.npy --fs 25600 --calibrate

    # 2. Monitor. Compare each new recording's flagged fraction against
    #    the calibration figure the previous step printed.
    python -m domain_adaptation.infer --input today.npy --fs 25600

Accepts `.npy`, `.npz` (via --signal-key, else the first array) and `.csv`
(single column). `--fs` is REQUIRED and must be your true sampling rate:
the signal is resampled to the model's common rate, and getting it wrong
shifts every fault frequency and invalidates the result.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from vibration_ad.data.windows import make_windows
from vibration_ad.features.spectrogram import SpectrogramNormalizer, expected_shape, windows_to_log_spectrograms

from domain_adaptation.config import Config, load_config
from domain_adaptation.data.transforms import resample_to_rate
from domain_adaptation.models.dann import FeatureClassifier
from domain_adaptation.models.feature_extractor import select_device

# Fraction of a recording's windows that must be flagged before the whole
# recording is called faulty. 0.10 is where per-recording detection peaked on
# the target condition (91.8%, vs 76.1% per-window) with no increase in false
# alarms -- aggregating helps because a real fault marks EVERY window of a
# recording, while the model's mistakes are correlated within a unit anyway.
DEFAULT_RECORDING_THRESHOLD = 0.10


def load_signal_file(path: Path, signal_key: str | None = None) -> np.ndarray:
    """Read a 1D vibration signal from .npy / .npz / .csv."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path).astype(np.float64).ravel()
    if suffix == ".npz":
        with np.load(path) as data:
            keys = list(data.keys())
            if signal_key is None:
                if len(keys) != 1:
                    raise ValueError(f"{path} holds {keys}; pass --signal-key to choose one")
                signal_key = keys[0]
            if signal_key not in data:
                raise KeyError(f"{path}: no array {signal_key!r} (has {keys})")
            return data[signal_key].astype(np.float64).ravel()
    if suffix == ".csv":
        return pd.read_csv(path, header=None).to_numpy(dtype=np.float64).ravel()
    raise ValueError(f"{path}: unsupported format {suffix!r} (use .npy, .npz or .csv)")


def score_signal(signal: np.ndarray, fs_hz: int, config: Config, model, normalizer, device) -> pd.DataFrame:
    """One row per window: predicted class and whether it counts as faulty."""
    signal = resample_to_rate(signal, fs_hz, config.common.fs_hz)
    windows = make_windows(signal, config.common.segment_length, config.common.eval_stride)
    if len(windows) == 0:
        raise ValueError(
            f"Signal too short: {len(signal)} samples at {config.common.fs_hz}Hz after resampling, "
            f"but one window needs {config.common.segment_length} "
            f"(~{config.common.segment_length / config.common.fs_hz:.2f}s). Record a longer sample."
        )
    spectrograms = normalizer.transform(windows_to_log_spectrograms(windows, config.common.fs_hz, config.stft))

    X = torch.from_numpy(spectrograms).unsqueeze(1).float().to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(X), 64):
            _features, logits = model(X[start : start + 64])
            preds.append(logits.argmax(dim=1).cpu().numpy())
    preds = np.concatenate(preds)
    return pd.DataFrame(
        {
            "window": np.arange(len(preds)),
            "predicted_class": preds,
            # Fault TYPE is deliberately not surfaced as a verdict: it did not
            # transfer to unseen machines in any run. Kept only for inspection.
            "flagged_faulty": (preds != 0).astype(int),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/domain_adaptation.yaml")
    parser.add_argument("--input", required=True, help="A signal file, or a directory of them.")
    parser.add_argument("--fs", type=int, required=True, help="Sampling rate of YOUR data, in Hz.")
    parser.add_argument("--signal-key", default=None, help="Array name inside a multi-array .npz.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_RECORDING_THRESHOLD,
                        help=f"Flagged-window fraction above which a recording is called faulty (default {DEFAULT_RECORDING_THRESHOLD}).")
    parser.add_argument("--calibrate", action="store_true",
                        help="Treat every input as KNOWN-HEALTHY and report the false-alarm rate on it.")
    parser.add_argument("--out", default=None, help="Optional path to write per-window predictions as CSV.")
    args = parser.parse_args()

    config = load_config(args.config)
    models_dir = config.resolve(config.paths.models_dir)
    ckpt = models_dir / "baseline_best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"{ckpt} not found -- train it first (see README's domain-adaptation section).")

    device = select_device(config.train_baseline.device)
    model = FeatureClassifier(expected_shape(config.common, config.stft), config).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    normalizer = SpectrogramNormalizer.load(models_dir / "normalizer.npz")

    path = Path(args.input)
    files = sorted(p for p in path.iterdir() if p.suffix.lower() in {".npy", ".npz", ".csv"}) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"No .npy/.npz/.csv files found in {path}")

    all_rows, summary = [], []
    for f in files:
        signal = load_signal_file(f, args.signal_key)
        per_window = score_signal(signal, args.fs, config, model, normalizer, device)
        per_window.insert(0, "source_file", f.name)
        all_rows.append(per_window)
        frac = float(per_window.flagged_faulty.mean())
        summary.append({"source_file": f.name, "n_windows": len(per_window),
                        "flagged_fraction": frac, "verdict": "FAULTY" if frac > args.threshold else "healthy"})

    summary_df = pd.DataFrame(summary)
    print(f"\nScored {len(files)} recording(s) at {args.fs}Hz -> {config.common.fs_hz}Hz, "
          f"{config.common.segment_length / config.common.fs_hz:.2f}s windows\n")
    print(summary_df.to_string(index=False))

    if args.calibrate:
        false_alarm = float((summary_df.flagged_fraction > args.threshold).mean())
        mean_frac = float(summary_df.flagged_fraction.mean())
        print("\n--- CALIBRATION (inputs assumed healthy) ---")
        print(f"  mean flagged fraction on healthy data: {mean_frac:.1%}")
        print(f"  recordings that would false-alarm:     {false_alarm:.1%}")
        if mean_frac <= 0.10:
            print("  VERDICT: this machine reads clean. Monitor with the default threshold.")
        elif mean_frac >= 0.50:
            print("  VERDICT: this machine's HEALTHY baseline already reads as faulty.\n"
                  "  This is the K002 failure mode (see module docstring): a per-unit offset,\n"
                  "  not noise. Do NOT deploy as-is -- the detector would alarm constantly.\n"
                  f"  Either raise --threshold above {mean_frac:.2f}, or (better) retrain with\n"
                  "  recordings from this machine added to the source pool: breadth was the\n"
                  "  single most effective lever found in this study.")
        else:
            print(f"  VERDICT: borderline. Set --threshold comfortably above {mean_frac:.2f}\n"
                  "  and confirm against a second known-healthy recording before trusting it.")
    else:
        n_faulty = int((summary_df.verdict == "FAULTY").sum())
        print(f"\n{n_faulty} of {len(summary_df)} recording(s) flagged as FAULTY "
              f"(threshold: >{args.threshold:.0%} of windows).")
        print("Calibrate on known-healthy data from THIS machine first if you have not: "
              "python -m domain_adaptation.infer --input <healthy> --fs <rate> --calibrate")

    if args.out:
        out = Path(args.out)
        pd.concat(all_rows, ignore_index=True).to_csv(out, index=False)
        summary_df.to_csv(out.with_name(out.stem + "_summary.csv"), index=False)
        print(f"\nWrote per-window predictions to {out}")
    print()


if __name__ == "__main__":
    main()
