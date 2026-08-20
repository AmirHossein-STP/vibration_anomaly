#!/usr/bin/env bash
# Full pipeline from scratch: download -> train -> evaluate -> demo inference
# -> deeper analysis. (Not included: 10_sweep.sh -- trains 8 extra model
# variants, takes a while, run it separately when you want that evidence.)
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bash "$DIR/scripts/01_download_data.sh"
bash "$DIR/scripts/02_train.sh"
bash "$DIR/scripts/03_evaluate.sh"
bash "$DIR/scripts/04_infer_demo.sh"
bash "$DIR/scripts/09_analyze.sh"
