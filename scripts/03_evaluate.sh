#!/usr/bin/env bash
# Score the trained detector against held-out healthy + all faulty windows.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$DIR/.venv/bin/python"
cd "$DIR"
"$PY" -m vibration_ad.evaluate --config configs/default.yaml
