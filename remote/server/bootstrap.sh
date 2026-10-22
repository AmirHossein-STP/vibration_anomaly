#!/usr/bin/env bash
# Make a fresh rental ready: mount vault -> tools -> venv -> restore outputs
# -> stage default datasets -> start the sync daemon. Run by `remote/box up`
# after the code push; idempotent (every step skips work already done), so
# re-running it is always safe.
#
#   bootstrap.sh [--init-vault]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_env.sh"
INIT=""
[ "${1:-}" = "--init-vault" ] && INIT="--init"

T0=$(date +%s)
step() { printf '\n[%3ss] == %s\n' "$(( $(date +%s) - T0 ))" "$*"; }

step "machine"
echo "$(hostname)  $(nproc) cores  $(free -g | awk '/Mem/{print $2}') GiB RAM  SSD free $(df -h --output=avail "$HOME" | tail -1 | tr -d ' ')"
if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
  echo "WARNING: no usable GPU (fine for CPU-only work)"
fi

step "vault"
VAULT_LABEL="$VAULT_LABEL" VAULT_MNT="$VAULT_MNT" bash "$HERE/mount_vault.sh" $INIT
mkdir -p "$VAULT_MNT/env/bin" "$UV_CACHE_DIR" "$VAULT_MNT/env/apt-debs" "$VAULT_MNT/datasets" "$VP/outputs" "$VP/runs"
cp "$HERE/VAULT.md" "$VAULT_MNT/VAULT.md"

step "system tools (unrar, 7z)"
if ! command -v unrar >/dev/null || ! command -v 7z >/dev/null; then
  # .debs are kept on the vault, so later rentals install without downloading
  sudo mkdir -p "$VAULT_MNT/env/apt-debs/partial"
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    -o Dir::Cache::Archives="$VAULT_MNT/env/apt-debs" -o APT::Sandbox::User=root \
    unrar p7zip-full >/dev/null \
  || { sudo apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
         -o Dir::Cache::Archives="$VAULT_MNT/env/apt-debs" -o APT::Sandbox::User=root unrar p7zip-full >/dev/null; }
fi
echo "unrar $(command -v unrar)  7z $(command -v 7z)"

step "uv $UV_VERSION"
if [ ! -x "$UV" ] || ! "$UV" --version | grep -q "$UV_VERSION"; then
  tmp=$(mktemp -d)
  curl -fsSL "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-x86_64-unknown-linux-gnu.tar.gz" | tar xz -C "$tmp"
  install -m 755 "$tmp"/uv-x86_64-unknown-linux-gnu/uv "$UV"
  rm -rf "$tmp"
fi
"$UV" --version

step "venv (system torch + remote/requirements-server.txt)"
# --system-site-packages: the image's torch 2.8+cu128 is used as is. Never
# reinstall torch (notes/11: a fresh CUDA torch download cost a rental 10 min).
[ -x "$VENV/bin/python" ] || "$UV" venv -q --system-site-packages --python /usr/bin/python3 "$VENV"
"$UV" pip install -q --python "$VENV/bin/python" -r "$REPO/remote/requirements-server.txt"
"$VENV/bin/python" - <<'EOF'
import importlib, sys
mods = ["numpy", "scipy", "sklearn", "pandas", "yaml", "matplotlib"]
v = {m: importlib.import_module(m).__version__ for m in mods}
try:
    import torch
    v["torch"] = f"{torch.__version__} cuda={torch.cuda.is_available()}"
except ImportError:
    v["torch"] = "MISSING"
print(f"python {sys.version.split()[0]}  " + "  ".join(f"{k} {x}" for k, x in v.items()))
EOF

step "restore outputs/ from the vault"
mkdir -p "$REPO/outputs" "$RUNS" "$SCRATCH" "$STAGED"
rsync -a "$VP/outputs/" "$REPO/outputs/"
echo "outputs: $(du -sh "$REPO/outputs" | cut -f1)"

step "stage: ${STAGE_DEFAULT:-none}"
if [ -n "${STAGE_DEFAULT:-}" ]; then
  # shellcheck disable=SC2086  # word splitting intended
  bash "$HERE/stage.sh" $STAGE_DEFAULT || echo "(staging incomplete -- fine before the first data fetch)"
fi

step "sync daemon (SSD -> vault every ${SYNC_INTERVAL}s)"
bash "$HERE/syncd.sh" --start

step "ready in $(( $(date +%s) - T0 ))s"
