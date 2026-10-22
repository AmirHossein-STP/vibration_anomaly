# Shared paths for the server scripts. Sourced, never run. The laptop's
# remote/box passes PROJECT, VAULT_MNT, ... from remote/box.conf.
PROJECT="${PROJECT:-vibration_anomaly_detection}"
VAULT_LABEL="${VAULT_LABEL:-vault}"
VAULT_MNT="${VAULT_MNT:-/mnt/vault}"
SYNC_INTERVAL="${SYNC_INTERVAL:-60}"
UV_VERSION="${UV_VERSION:-0.5.11}"

WORK="$HOME/work"                       # SSD: everything here dies with the rental
REPO="$WORK/$PROJECT"                   # code checkout (pushed by `box push`)
VENV="$WORK/venv"
RUNS="$WORK/runs"                       # live run records, mirrored to the vault
SCRATCH="$WORK/scratch"                 # raw downloads awaiting condensing
STAGED="$WORK/data"                     # vault -> SSD dataset copies

VP="$VAULT_MNT/projects/$PROJECT"       # vault: this project's outputs + runs
UV="$VAULT_MNT/env/bin/uv"
export UV_CACHE_DIR="$VAULT_MNT/env/uv-cache"
export UV_LINK_MODE=copy                # cache (HDD) and venv (SSD) are different filesystems

vault_mounted() { mountpoint -q "$VAULT_MNT"; }
