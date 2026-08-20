#!/usr/bin/env bash
# Regenerate notebooks/kaggle_train.ipynb from current source, then push it
# as a new version of the aghzuzu/vibration-anomaly-detection-mvp kernel.
# Runs remotely on Kaggle's infrastructure -- this does not block; use
# 06_kaggle_status.sh to poll, 07_kaggle_pull.sh to fetch results once done.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KAGGLE="$DIR/.venv/bin/kaggle"
cd "$DIR"

python3 scripts/build_kaggle_notebook.py
"$KAGGLE" kernels push -p notebooks
