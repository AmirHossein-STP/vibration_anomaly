#!/usr/bin/env bash
# One-screen state of the box: GPU, vault, datasets, sync daemon, runs.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_env.sh"

echo "== $(hostname)  up $(uptime -p | sed 's/up //')"
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null \
  | sed 's/^/gpu: /' || echo "gpu: none"
if vault_mounted; then
  echo "vault: $(df -h --output=used,size,avail "$VAULT_MNT" | tail -1 | awk '{print $1" used of "$2", "$3" free"}')  $(grep uuid "$VAULT_MNT/.vault-id" 2>/dev/null)"
  echo "datasets:"
  for m in "$VAULT_MNT"/datasets/*/*/MANIFEST.json; do
    [ -f "$m" ] || { echo "  (none yet)"; break; }
    d=$(dirname "$m")
    st=$(grep -o '"status": "[a-z-]*"' "$m" | cut -d'"' -f4)
    printf '  %-13s %-28s %-13s %s%s\n' "$(basename "$(dirname "$d")")" "$(basename "$d")" "$st" \
      "$(du -sh "$d" 2>/dev/null | cut -f1)" "$([ -f "$d/SEALED" ] && echo '  SEALED')"
  done
else
  echo "vault: NOT MOUNTED"
fi
echo "syncd: $(bash "$HERE/syncd.sh" --status)"
echo "runs (latest 10):"
ls -1d "$RUNS"/*/ 2>/dev/null | tail -10 | while read -r rd; do
  rd=${rd%/}
  if [ -f "$rd/exit_code" ]; then st="exit $(cat "$rd/exit_code")  $(cat "$rd/duration_s" 2>/dev/null)s"
  elif kill -0 "$(cat "$rd/pid" 2>/dev/null)" 2>/dev/null; then st="RUNNING $(( $(date +%s) - $(stat -c %Y "$rd/cmd.txt") ))s"
  else st="DIED (no exit code)"; fi
  printf '  %-40s %s\n' "$(basename "$rd")" "$st"
done
[ -n "$(ls -A "$RUNS" 2>/dev/null)" ] || echo "  (none)"
