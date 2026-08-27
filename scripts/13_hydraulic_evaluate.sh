#!/usr/bin/env bash
# Score one component's trained fusion detector against its held-out
# healthy + degraded cycles. See scripts/12_hydraulic_train.sh's comment --
# same four-separate-detectors reasoning.
#
# Usage: bash scripts/13_hydraulic_evaluate.sh cooler|valve|pump|accumulator
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$DIR/.venv/bin/python"
cd "$DIR"

COMPONENT="${1:?Usage: $0 cooler|valve|pump|accumulator}"
"$PY" -m hydraulic_fusion_ad.evaluate --config "configs/hydraulic_${COMPONENT}.yaml"
