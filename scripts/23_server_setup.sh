#!/usr/bin/env bash
# One-time provisioning for the GPU server, to be run ON the server.
#
# Everything this project has done so far ran on Kaggle CPU kernels with a
# ~142-minute turnaround, which forced several compromises (small CNN
# instead of ResNet-18, 12kHz common rate, trimmed source conditions,
# single seeds, no sweeps -- see DOMAIN_ADAPTATION_PLAN.md). This script
# sets up the box that lets those be revisited.
#
# Assumes: Linux, NVIDIA driver already installed (check with nvidia-smi),
# python3 >= 3.10, and a Kaggle API token for the dataset downloads.
# Re-runnable: every step skips work that is already done.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/vibration_anomaly_detection}"
DATA_DIR="${DATA_DIR:-$PROJECT_DIR/data}"
VENV="$PROJECT_DIR/.venv"

echo "=== 1. GPU check ==="
if ! command -v nvidia-smi >/dev/null; then
  echo "nvidia-smi not found -- install the NVIDIA driver first. Aborting." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo "=== 2. venv ==="
[ -d "$VENV" ] || python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install -q --upgrade pip

echo "=== 3. PyTorch with CUDA ==="
# Unlike the dev laptop (an Intel Mac, where PyPI caps torch at 2.2.2 and it
# cannot even import against numpy 2.x), a Linux box gets current CUDA
# wheels. This is also what finally makes the torch-dependent tests runnable
# outside a Kaggle kernel.
if ! "$VENV/bin/python" -c "import torch" 2>/dev/null; then
  "$VENV/bin/pip" install -q torch --index-url https://download.pytorch.org/whl/cu124
fi
"$VENV/bin/python" - <<'EOF'
import torch
print(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device: {torch.cuda.get_device_name(0)}  "
          f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.0f} GB")
else:
    raise SystemExit("ERROR: torch cannot see the GPU -- check driver/CUDA match")
EOF

echo "=== 4. project deps ==="
"$VENV/bin/pip" install -q -r "$PROJECT_DIR/requirements.txt"
"$VENV/bin/pip" install -q -e "$PROJECT_DIR"

echo "=== 5. datasets ==="
# Paderborn arrives CONDENSED, not raw. scripts/build_kaggle_notebook_prepare_pu.py
# runs on Kaggle (free, data already mounted there) and reduces the 9.7 GB
# dataset to ~437 MB by keeping only the vibration_1 channel at 12 kHz --
# the rest is six unused channels at a rate the pipeline downsamples anyway.
# That is ~22x less paid transfer, and it covers EVERY bearing code and
# operating condition, so no config change sends us back for more.
mkdir -p "$DATA_DIR"
PREPARED="$DATA_DIR/paderborn_prepared"
if [ ! -f "$PREPARED/manifest.csv" ]; then
  echo "  pulling condensed Paderborn (~437 MB, 32 archives)..."
  "$VENV/bin/kaggle" kernels output aghzuzu/prepare-paderborn-subset -p "$PREPARED" --force
fi
if [ -f "$PREPARED/manifest.csv" ]; then
  echo "  Paderborn: $(wc -l < "$PREPARED/manifest.csv") manifest rows, $(du -sh "$PREPARED" | cut -f1)"
else
  echo "  WARNING: condensed Paderborn missing -- rerun the prepare-paderborn-subset kernel" >&2
fi

# CWRU is NOT downloaded: the laptop already holds it converted to .npz
# (179 MB, 64 files), which is ~10x smaller than the .mat form on Kaggle and
# skips the conversion step. Copy it up before running this:
#   scp -r data/raw <server>:$PROJECT_DIR/data/raw
if [ -d "$DATA_DIR/raw" ] && [ "$(ls -1 "$DATA_DIR/raw"/*.npz 2>/dev/null | wc -l)" -ge 64 ]; then
  echo "  CWRU: $(ls -1 "$DATA_DIR/raw"/*.npz | wc -l) .npz files present"
else
  echo "  WARNING: CWRU not found at $DATA_DIR/raw -- scp it from the laptop:" >&2
  echo "    scp -r data/raw <server>:$DATA_DIR/raw" >&2
fi

echo "=== 6. smoke test ==="
cd "$PROJECT_DIR" && "$VENV/bin/python" -m pytest -q

cat <<EOF

Setup complete.
  project : $PROJECT_DIR
  venv    : $VENV
  data    : $DATA_DIR

Unlike the laptop, torch works here -- so the ~10 tests that have been
skipping since day one now actually run, and model changes no longer need a
142-minute Kaggle round trip to verify.
EOF
