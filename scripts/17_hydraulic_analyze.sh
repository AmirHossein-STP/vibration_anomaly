#!/usr/bin/env bash
# Deeper diagnostics for one component's detector -- threshold sensitivity,
# second-order confound check, raw time-correlation. Run AFTER
# 12_hydraulic_train.sh and 13_hydraulic_evaluate.sh for the same component.
#
# Usage: bash scripts/17_hydraulic_analyze.sh cooler|valve|pump|accumulator
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$DIR/.venv/bin/python"
cd "$DIR"

COMPONENT="${1:?Usage: $0 cooler|valve|pump|accumulator}"
"$PY" -m hydraulic_fusion_ad.analyze --config "configs/hydraulic_${COMPONENT}.yaml"
