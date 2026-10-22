#!/usr/bin/env bash
# Mount the persistent vault volume at $VAULT_MNT, found by its ext4 LABEL
# (the device has been vdb on one rental and vdx in the provider's docs, so
# the name is never trusted).
#
#   mount_vault.sh           mount, or fail with a list of candidate disks
#   mount_vault.sh --init    FIRST RENTAL ONLY: partition + format a blank
#                            volume. Refuses unless exactly one disk qualifies:
#                            not the root disk, no partitions, no filesystem or
#                            partition-table signature, 20-100 GiB.
#   VAULT_DISK=/dev/xyz mount_vault.sh --init
#                            name the disk instead of auto-picking it (the same
#                            safety checks still apply; used by the loop-device test)
set -euo pipefail
VAULT_LABEL="${VAULT_LABEL:-vault}"
VAULT_MNT="${VAULT_MNT:-/mnt/vault}"
INIT=0
[ "${1:-}" = "--init" ] && INIT=1

say() { echo "vault: $*"; }

if mountpoint -q "$VAULT_MNT"; then
  say "already mounted at $VAULT_MNT"
  exit 0
fi

blank_reason() {  # prints why $1 is NOT safe to format; prints nothing if it is
  local d=$1 size
  size=$(lsblk -dbno SIZE "$d")
  if [ "$(lsblk -no NAME "$d" | wc -l)" -gt 1 ]; then echo "has partitions"; return; fi
  if [ -n "$(sudo wipefs -n "$d" 2>/dev/null | tail -n +2)" ]; then echo "has a filesystem/partition signature"; return; fi
  if sudo blkid -p "$d" >/dev/null 2>&1; then echo "blkid finds a signature"; return; fi
  if [ "$size" -lt $((20 * 1024**3)) ] || [ "$size" -gt $((100 * 1024**3)) ]; then echo "size $((size / 1024**3)) GiB outside 20-100"; return; fi
}

dev=$(sudo blkid -L "$VAULT_LABEL" 2>/dev/null || true)
if [ -z "$dev" ]; then
  root_disk=$(lsblk -no PKNAME "$(findmnt -no SOURCE /)")
  if [ -n "${VAULT_DISK:-}" ]; then
    candidates=("$VAULT_DISK")
  else
    candidates=()
    while read -r name type; do
      [ "$type" = disk ] && [ "$name" != "$root_disk" ] && candidates+=("/dev/$name")
    done < <(lsblk -dno NAME,TYPE)
  fi
  if [ "$INIT" -eq 0 ]; then
    say "no filesystem labelled '$VAULT_LABEL' found."
    for d in "${candidates[@]}"; do
      r=$(blank_reason "$d"); echo "  $d $(lsblk -dno SIZE "$d")  ${r:-BLANK (formattable with --init-vault)}"
    done
    say "is the storage volume attached to this server in the provider's panel?"
    say "if this is the first rental with a NEW, blank volume: remote/box up <ip> --init-vault"
    exit 1
  fi
  blank=()
  for d in "${candidates[@]}"; do
    r=$(blank_reason "$d")
    if [ -z "$r" ]; then blank+=("$d"); else say "not formatting $d: $r"; fi
  done
  if [ "${#blank[@]}" -ne 1 ]; then
    say "refusing to format: need exactly one blank candidate disk, found ${#blank[@]} (${blank[*]:-none})"
    exit 1
  fi
  disk=${blank[0]}
  say "formatting BLANK disk $disk ($(lsblk -dno SIZE "$disk")) as ext4 label=$VAULT_LABEL"
  echo 'type=83' | sudo sfdisk --quiet "$disk"
  sudo partprobe "$disk" 2>/dev/null || true
  sudo udevadm settle
  part=$(lsblk -lnpo NAME,TYPE "$disk" | awk '$2=="part"{print $1; exit}')
  [ -n "$part" ] || { say "partition did not appear on $disk"; exit 1; }
  # -m 0: no root-reserved blocks on a data-only disk (+5% usable space)
  sudo mkfs.ext4 -q -m 0 -L "$VAULT_LABEL" "$part"
  dev=$part
fi

sudo mkdir -p "$VAULT_MNT"
sudo mount -o noatime "$dev" "$VAULT_MNT"
me="$(id -u):$(id -g)"
if [ "$(stat -c %u:%g "$VAULT_MNT")" != "$me" ]; then
  # Only the top level on a fresh volume; the whole tree only if a rental ever
  # comes with a different uid (recursive chown on an HDD is slow).
  if [ -e "$VAULT_MNT/.vault-id" ] && [ "$(stat -c %u "$VAULT_MNT/.vault-id")" != "$(id -u)" ]; then
    say "uid changed since the last rental; chown -R (one-off, may take a minute)"
    sudo chown -R "$me" "$VAULT_MNT"
  else
    sudo chown "$me" "$VAULT_MNT"
  fi
fi

if [ ! -f "$VAULT_MNT/.vault-id" ]; then
  printf 'uuid=%s\ncreated=%s\ncreated_on=%s\nlayout=1\n' \
    "$(cat /proc/sys/kernel/random/uuid)" "$(date -u +%FT%TZ)" "$(hostname)" > "$VAULT_MNT/.vault-id"
fi
echo "$(date -u +%FT%TZ) mounted on $(hostname) ($(hostname -I | awk '{print $1}')) dev=$dev" >> "$VAULT_MNT/.mount-history"
say "mounted $dev at $VAULT_MNT  ($(df -h --output=avail "$VAULT_MNT" | tail -1 | tr -d ' ') free)  $(grep uuid "$VAULT_MNT/.vault-id")"
