#!/usr/bin/env bash
# Deeper diagnostics beyond evaluate.py: latent-space PCA, confusion matrix,
# detection rate by RPM, threshold sensitivity. Run after 03_evaluate.sh.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$DIR/.venv/bin/python"
cd "$DIR"
"$PY" -m vibration_ad.analyze --config configs/default.yaml
