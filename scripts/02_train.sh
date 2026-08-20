#!/usr/bin/env bash
# Train the autoencoder on healthy-only spectrograms.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$DIR/.venv/bin/python"
cd "$DIR"
"$PY" -m vibration_ad.train --config configs/default.yaml "$@"
