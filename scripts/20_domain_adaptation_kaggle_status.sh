#!/usr/bin/env bash
# One-shot status check of the domain-adaptation Kaggle kernel run.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$DIR/.venv/bin/kaggle" kernels status aghzuzu/domain-adaptation-zero-shot-fault-diagnosis
