#!/usr/bin/env bash
# Fast pull: only the small JSON/CSV reports + kernel log, skipping raw
# data, cached spectrogram arrays, model checkpoints, and figures -- useful
# for a quick metrics check after a small code change, same rationale as
# 18_hydraulic_kaggle_pull_reports_only.sh. Use
# 21_domain_adaptation_kaggle_pull.sh instead when you need the model
# checkpoints, cached arrays, or figures too.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$DIR/outputs/domain_adaptation_kaggle_run"
mkdir -p "$OUT"
"$DIR/.venv/bin/kaggle" kernels output aghzuzu/domain-adaptation-zero-shot-fault-diagnosis \
    -p "$OUT" --force \
    --file-pattern ".*outputs/da_reports/.*\.(json|csv)$"
echo "Saved reports-only files to $OUT"
