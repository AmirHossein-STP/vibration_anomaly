#!/usr/bin/env bash
# Copy datasets from the vault (HDD) to the SSD and link them where the code
# already looks, so no analysis script needs a path change.
#
#   stage.sh name[:part] ...
#
#   cwru:de12     the 64 files in remote/datasets/cwru_de12.txt -> data/raw
#                 (catalog code globs *.npz, so the 48 kHz / fan-end files
#                 must NOT be in data/raw)
#   cwru          all 161 CWRU files               -> data/cwru_full
#   paderborn:v1  12 kHz vibration store           -> data/pu/prepared_pu
#   paderborn:v2  64 kHz + speed/torque/force      -> data/pu/prepared_pu_v2
#   <other>       raw/ or condensed/               -> data/<name>
#   (ims is sealed and cannot be staged)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_env.sh"
vault_mounted || { echo "stage: vault not mounted" >&2; exit 1; }
[ $# -gt 0 ] || { echo "usage: stage.sh name[:part] ..." >&2; exit 2; }

dataset_dir() {  # the one complete version of dataset $1 on the vault
  local d
  for d in "$VAULT_MNT/datasets/$1"/*/; do
    [ -f "$d/MANIFEST.json" ] && grep -q '"status": "complete"' "$d/MANIFEST.json" && { echo "${d%/}"; return; }
  done
  return 1
}

link() {  # link repo path $2 -> staged dir $1
  local target="$REPO/$2"
  mkdir -p "$(dirname "$target")"
  if [ -e "$target" ] && [ ! -L "$target" ]; then
    echo "stage: $target exists and is not a link; leaving it" >&2; return 1
  fi
  ln -sfn "$1" "$target"
}

rc=0
for spec in "$@"; do
  name=${spec%%:*}; part=""
  [[ "$spec" == *:* ]] && part=${spec#*:}
  if [ "$name" = ims ]; then echo "stage: ims is SEALED (MASTER_PLAN §4.3)" >&2; rc=1; continue; fi
  if ! d=$(dataset_dir "$name"); then echo "stage: $name not on the vault yet (box data fetch $name)" >&2; rc=1; continue; fi
  t0=$(date +%s)
  case "$spec" in
    cwru:de12)
      dest="$STAGED/cwru_de12"; mkdir -p "$dest"
      grep -v -e '^#' -e '^$' "$REPO/remote/datasets/cwru_de12.txt" \
        | rsync -a --files-from=- "$d/raw/" "$dest/"
      link "$dest" data/raw ;;
    cwru)
      dest="$STAGED/cwru_full"; rsync -a "$d/raw/" "$dest/"; link "$dest" data/cwru_full ;;
    paderborn:v1|paderborn:v2)
      dest="$STAGED/paderborn_$part"; rsync -a "$d/condensed/$part/" "$dest/"
      if [ "$part" = v1 ]; then link "$dest" data/pu/prepared_pu; else link "$dest" data/pu/prepared_pu_v2; fi ;;
    *)
      sub=raw; [ -d "$d/condensed" ] && sub=condensed
      [ -n "$part" ] && sub="$sub/$part"
      dest="$STAGED/$name${part:+_$part}"; rsync -a "$d/$sub/" "$dest/"
      link "$dest" "data/$name${part:+_$part}" ;;
  esac
  echo "staged $spec -> $dest ($(du -sh "$dest" | cut -f1), $(( $(date +%s) - t0 ))s)"
done
exit $rc
