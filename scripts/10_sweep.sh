#!/usr/bin/env bash
# Bottleneck-size (latent_dim) sensitivity sweep -- see
# notes/03_model_notes.md and src/vibration_ad/sweep.py. Trains+evaluates
# 8 model variants; takes a while.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$DIR/.venv/bin/python"
cd "$DIR"
"$PY" -m vibration_ad.sweep --config configs/default.yaml
