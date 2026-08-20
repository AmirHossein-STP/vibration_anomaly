#!/usr/bin/env bash
# Download the CWRU .npz files listed in data/raw/MANIFEST.txt.
# Stdlib-only, so this works even before the project venv exists --
# but we still call it through the venv's python for consistency once it does.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$DIR/.venv/bin/python"
[ -x "$PY" ] || PY="python3"
cd "$DIR"
"$PY" -m vibration_ad.data.download --config configs/default.yaml
