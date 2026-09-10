#!/usr/bin/env bash
# Regenerate notebooks/domain_adaptation/kaggle_train_domain_adaptation.ipynb
# from current source, then push it as a new version of the
# aghzuzu/domain-adaptation-zero-shot-fault-diagnosis kernel. Runs remotely
# on Kaggle's infrastructure -- this does not block; use
# 20_domain_adaptation_kaggle_status.sh to poll,
# 21_domain_adaptation_kaggle_pull.sh to fetch results once done.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KAGGLE="$DIR/.venv/bin/kaggle"
cd "$DIR"

python3 scripts/build_kaggle_notebook_domain_adaptation.py
"$KAGGLE" kernels push -p notebooks/domain_adaptation
