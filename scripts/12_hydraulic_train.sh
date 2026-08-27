#!/usr/bin/env bash
# Train one component's fusion autoencoder on its own healthy hydraulic
# cycles. Requires a component name -- there are four separate detectors
# now (see notes/09_hydraulic_notes.md's "Per-component redesign" section),
# not one shared config.
#
# Usage: bash scripts/12_hydraulic_train.sh cooler|valve|pump|accumulator [extra train.py args]
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$DIR/.venv/bin/python"
cd "$DIR"

COMPONENT="${1:?Usage: $0 cooler|valve|pump|accumulator [extra train.py args]}"
shift
"$PY" -m hydraulic_fusion_ad.train --config "configs/hydraulic_${COMPONENT}.yaml" "$@"
