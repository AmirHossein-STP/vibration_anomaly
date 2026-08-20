#!/usr/bin/env bash
# Full pipeline from scratch: download -> train -> evaluate -> demo inference.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bash "$DIR/scripts/01_download_data.sh"
bash "$DIR/scripts/02_train.sh"
bash "$DIR/scripts/03_evaluate.sh"
bash "$DIR/scripts/04_infer_demo.sh"
