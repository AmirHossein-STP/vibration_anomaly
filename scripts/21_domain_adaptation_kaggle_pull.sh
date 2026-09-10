#!/usr/bin/env bash
# Download the finished domain-adaptation kernel's log + output files into
# outputs/domain_adaptation_kaggle_run/. Only useful after
# 20_domain_adaptation_kaggle_status.sh shows a terminal status
# (KernelWorkerStatus.COMPLETE or .ERROR).
#
# --force is required, not optional: `kaggle kernels output` silently SKIPS
# any file it thinks has a "more recently modified local copy" -- every
# source file just edited locally before pushing, plus any report left over
# from a previous pull -- see HYDRAULIC_PLAN.md's Status section for how
# this bit us once on the hydraulic experiment.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$DIR/outputs/domain_adaptation_kaggle_run"
mkdir -p "$OUT"
"$DIR/.venv/bin/kaggle" kernels output aghzuzu/domain-adaptation-zero-shot-fault-diagnosis -p "$OUT" --force
echo "Saved to $OUT"
