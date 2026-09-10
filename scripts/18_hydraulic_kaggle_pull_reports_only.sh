#!/usr/bin/env bash
# Fast pull: only the small JSON/CSV reports + kernel log, skipping raw
# data, split .npz arrays, model checkpoints, and figures -- the full
# `kaggle kernels output` pull can be very slow/prone to stalling on this
# kernel's larger files (hundreds of MB of raw hydraulic sensor data +
# per-component .npz/.pt files). Use this when you just need to check
# metrics/analysis output, e.g. after a small code change to analyze.py --
# use 16_hydraulic_kaggle_pull.sh instead when you actually need the model
# checkpoints, split arrays, or figures.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$DIR/outputs/hydraulic_kaggle_run"
mkdir -p "$OUT"
"$DIR/.venv/bin/kaggle" kernels output aghzuzu/hydraulic-sensor-fusion-anomaly-detection \
    -p "$OUT" --force \
    --file-pattern ".*outputs/hydraulic_reports/.*\.(json|csv)$"
echo "Saved reports-only files to $OUT"
