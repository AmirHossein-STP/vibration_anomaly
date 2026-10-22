#!/usr/bin/env bash
# Mirror results from the SSD to the vault: the worktree's outputs/ and the
# run records. The box is deleted when the rental ends, so at most one
# interval of results is ever at risk.
#
#   syncd.sh --start   start the background loop (no-op if it is running)
#   syncd.sh --stop    stop it
#   syncd.sh --once    one sync now (runs, box down); serialised with the loop
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_env.sh"
PIDFILE="$WORK/.syncd.pid"
LOG="$WORK/.syncd.log"
LOCK="$WORK/.syncd.lock"

sync_once() {
  vault_mounted || { echo "$(date -u +%T) vault not mounted, skipped"; return 1; }
  (
    flock 9
    mkdir -p "$VP/outputs" "$VP/runs"
    rsync -a "$REPO/outputs/" "$VP/outputs/"
    rsync -a "$RUNS/" "$VP/runs/"
    date -u +%FT%TZ > "$WORK/.last_sync"
  ) 9>"$LOCK"
}

running() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }

case "${1:-}" in
  --once) sync_once ;;
  --start)
    if running; then echo "syncd already running (pid $(cat "$PIDFILE"))"; exit 0; fi
    setsid nohup bash -c "
      while true; do
        bash '$HERE/syncd.sh' --once >> '$LOG' 2>&1 || true
        sleep '$SYNC_INTERVAL'
      done" < /dev/null >> "$LOG" 2>&1 &
    echo $! > "$PIDFILE"
    echo "syncd started (pid $!, log $LOG)"
    ;;
  --stop)
    if running; then kill -- -"$(ps -o pgid= "$(cat "$PIDFILE")" | tr -d ' ')" 2>/dev/null || kill "$(cat "$PIDFILE")"; fi
    rm -f "$PIDFILE"; echo "syncd stopped"
    ;;
  --status)
    if running; then echo "running (pid $(cat "$PIDFILE")), last sync $(cat "$WORK/.last_sync" 2>/dev/null || echo never)"
    else echo "NOT running, last sync $(cat "$WORK/.last_sync" 2>/dev/null || echo never)"; fi
    ;;
  *) echo "usage: syncd.sh --start|--stop|--once|--status" >&2; exit 2 ;;
esac
