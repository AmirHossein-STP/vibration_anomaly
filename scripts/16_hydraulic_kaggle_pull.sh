#!/usr/bin/env bash
# Download the finished hydraulic fusion kernel's log + output files into
# outputs/hydraulic_kaggle_run/. Only useful after 15_hydraulic_kaggle_status.sh
# shows a terminal status (KernelWorkerStatus.COMPLETE or .ERROR).
#
# --force is required, not optional: `kaggle kernels output` silently SKIPS
# any file it thinks has a "more recently modified local copy" -- which is
# every source file we just edited locally before pushing, AND every report
# left over from a previous pull of this same kernel. Without --force, a
# second pull after a re-push can silently keep serving STALE report/model
# files while claiming success (this actually happened once -- see
# HYDRAULIC_PLAN.md's Status section, kernel v3's first pull).
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$DIR/outputs/hydraulic_kaggle_run"
mkdir -p "$OUT"
"$DIR/.venv/bin/kaggle" kernels output aghzuzu/hydraulic-sensor-fusion-anomaly-detection -p "$OUT" --force
echo "Saved to $OUT"
