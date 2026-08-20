#!/usr/bin/env bash
# Run the detector on one example faulty file and one healthy file, so you
# get two reconstruction-error timelines to compare (outputs/figures/).
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$DIR/.venv/bin/python"
cd "$DIR"

"$PY" -m vibration_ad.infer --config configs/default.yaml \
    --input data/raw/1797_IR_21_DE12.npz

"$PY" -m vibration_ad.infer --config configs/default.yaml \
    --input data/raw/1797_Normal.npz
