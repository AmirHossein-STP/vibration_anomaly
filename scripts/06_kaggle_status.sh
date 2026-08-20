#!/usr/bin/env bash
# One-shot status check of the Kaggle kernel run. For continuous polling,
# wrap this in a loop (see notes/07_dvc_and_kaggle.md for the pattern) --
# not built into this script since a fixed poll interval doesn't suit
# every run length.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$DIR/.venv/bin/kaggle" kernels status aghzuzu/vibration-anomaly-detection-mvp
