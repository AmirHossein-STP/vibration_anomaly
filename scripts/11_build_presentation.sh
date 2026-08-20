#!/usr/bin/env bash
# Render reports/presentation_template.html + the latest pulled Kaggle run
# (outputs/kaggle_run/) into a single self-contained HTML report at
# outputs/reports/presentation.html. Run 07_kaggle_pull.sh first if
# outputs/kaggle_run/ is stale.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$DIR/.venv/bin/python"
[ -x "$PY" ] || PY="python3"
cd "$DIR"
"$PY" scripts/build_presentation.py "$@"
