#!/usr/bin/env bash
# Download the finished kernel's log + any output files (figures, reports,
# whatever it wrote under /kaggle/working) into outputs/kaggle_run/.
# Only useful after 06_kaggle_status.sh shows a terminal status
# (KernelWorkerStatus.COMPLETE or .ERROR).
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$DIR/outputs/kaggle_run"
mkdir -p "$OUT"
"$DIR/.venv/bin/kaggle" kernels output aghzuzu/vibration-anomaly-detection-mvp -p "$OUT"
echo "Saved to $OUT"
