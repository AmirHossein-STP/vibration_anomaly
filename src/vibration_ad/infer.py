"""Run the trained detector on one signal file and visualize the result.

This is the "deployment" entrypoint: point it at any CWRU .npz file and it
reproduces the whole pipeline (window -> STFT -> normalize -> reconstruct ->
error) end to end on that one file, then plots reconstruction error over
time with the calibrated threshold -- the "Visualizing Autoencoder
Reconstruction Error" view from the original project plan.

Requires `evaluate.py` to have been run first (it reuses the threshold
calibrated there, rather than recomputing it, so the flagging behavior here
always matches the reported evaluation metrics).

Usage:
    python -m vibration_ad.infer --config configs/default.yaml \\
        --input data/raw/1797_IR_21_DE12.npz
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from vibration_ad.config import Config, load_config
from vibration_ad.data.catalog import parse_filename
from vibration_ad.data.dataset import load_signal
from vibration_ad.data.windows import make_windows
from vibration_ad.evaluate import compute_errors_and_recons
from vibration_ad.features.spectrogram import (
    SpectrogramNormalizer,
    expected_shape,
    windows_to_log_spectrograms,
)
from vibration_ad.models.autoencoder import ConvAutoencoder, select_device
from vibration_ad import viz


def infer(config: Config, input_path: str, stride: int | None = None) -> pd.DataFrame:
    models_dir = config.resolve(config.paths.models_dir)
    reports_dir = config.resolve(config.paths.reports_dir)
    figures_dir = config.resolve(config.paths.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = reports_dir / "evaluation_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"{metrics_path} not found. Run `python -m vibration_ad.evaluate` first "
            "so infer.py can reuse the calibrated threshold."
        )
    threshold = json.loads(metrics_path.read_text())["threshold"]

    stride = stride or config.windowing.eval_stride
    signal = load_signal(input_path, config.data.sensor)
    windows = make_windows(signal, config.windowing.segment_length, stride)
    if len(windows) == 0:
        raise ValueError(
            f"Signal has {len(signal)} samples, shorter than segment_length="
            f"{config.windowing.segment_length}; can't form even one window."
        )

    spectrograms = windows_to_log_spectrograms(windows, config.data.fs_hz, config.stft)
    normalizer = SpectrogramNormalizer.load(models_dir / "normalizer.npz")
    spectrograms_norm = normalizer.transform(spectrograms)

    expected = expected_shape(config.windowing, config.stft)
    device = select_device(config.train.device)
    model = ConvAutoencoder(
        input_shape=expected,
        base_channels=config.model.base_channels,
        latent_dim=config.model.latent_dim,
    ).to(device)
    model.load_state_dict(torch.load(models_dir / "autoencoder_best.pt", map_location=device))

    errors, _recons = compute_errors_and_recons(model, spectrograms_norm, device)
    flagged = errors > threshold

    result = pd.DataFrame(
        {
            "window_index": np.arange(len(errors)),
            "start_sample": np.arange(len(errors)) * stride,
            "recon_error": errors,
            "predicted_anomaly": flagged.astype(int),
        }
    )

    fname = input_path.rsplit("/", 1)[-1]
    try:
        meta = parse_filename(fname)
        result["true_fault_type"] = meta["fault_type"]
        result["true_label"] = meta["label"]
        ground_truth_note = f" (true label: {meta['fault_type']}, size={meta['fault_size_mils']}mil)"
    except ValueError:
        ground_truth_note = " (filename not in CWRU naming convention -- no ground truth to compare)"

    out_csv = reports_dir / f"infer_{fname.replace('.npz', '')}.csv"
    result.to_csv(out_csv, index=False)

    out_png = figures_dir / f"infer_{fname.replace('.npz', '')}_timeline.png"
    viz.plot_error_timeline(errors, threshold, out_png, title=f"{fname}{ground_truth_note}")

    n_flagged = int(flagged.sum())
    print(f"{fname}: {n_flagged}/{len(errors)} windows flagged as anomalous ({100 * n_flagged / len(errors):.1f}%)")
    print(f"Timeline plot: {out_png}")
    print(f"Per-window CSV: {out_csv}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input", required=True, help="Path to a CWRU .npz signal file")
    parser.add_argument(
        "--stride", type=int, default=None, help="Window stride in samples (default: config eval_stride)"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    infer(config, args.input, stride=args.stride)


if __name__ == "__main__":
    main()
