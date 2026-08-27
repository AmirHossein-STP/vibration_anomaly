#!/usr/bin/env bash
# Download the finished hydraulic fusion kernel's log + output files into
# outputs/hydraulic_kaggle_run/. Only useful after 15_hydraulic_kaggle_status.sh
# shows a terminal status (KernelWorkerStatus.COMPLETE or .ERROR).
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$DIR/outputs/hydraulic_kaggle_run"
mkdir -p "$OUT"
"$DIR/.venv/bin/kaggle" kernels output aghzuzu/hydraulic-sensor-fusion-anomaly-detection -p "$OUT"
echo "Saved to $OUT"
