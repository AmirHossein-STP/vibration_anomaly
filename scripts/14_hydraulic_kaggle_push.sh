#!/usr/bin/env bash
# Regenerate notebooks/hydraulic/kaggle_train_hydraulic.ipynb from current
# source, then push it as a new version of the
# aghzuzu/hydraulic-sensor-fusion-anomaly-detection kernel. Runs remotely on
# Kaggle's infrastructure -- this does not block; use
# 15_hydraulic_kaggle_status.sh to poll, 16_hydraulic_kaggle_pull.sh to
# fetch results once done.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KAGGLE="$DIR/.venv/bin/kaggle"
cd "$DIR"

python3 scripts/build_kaggle_notebook_hydraulic.py
"$KAGGLE" kernels push -p notebooks/hydraulic
