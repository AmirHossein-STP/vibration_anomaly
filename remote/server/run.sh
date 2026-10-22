#!/usr/bin/env bash
# Start a detached run with a full record. Behind `remote/box run`.
#
#   run.sh NAME -- CMD [ARGS...]
#
# The run survives ssh drops (setsid + nohup). It runs in the repo, in the
# venv, with PYTHONPATH=src. It leaves ~/work/runs/<id>/ holding cmd.txt,
# git.json + diff.patch (exactly what code ran), env.txt, log, exit_code and
# duration_s, and syncs to the vault the moment it ends. Prints the run id.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/_env.sh"
NAME=${1:?usage: run.sh NAME -- CMD...}
shift
[ "${1:-}" = "--" ] && shift
[ $# -gt 0 ] || { echo "run.sh: no command given" >&2; exit 2; }
[[ "$NAME" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "run.sh: name may only hold letters, digits, . _ -" >&2; exit 2; }

ID="$(date -u +%Y%m%d-%H%M%S)_$NAME"
RD="$RUNS/$ID"
mkdir -p "$RD"
printf '%q ' "$@" > "$RD/cmd.txt"; echo >> "$RD/cmd.txt"
cp "$REPO/.box_provenance.json" "$RD/git.json" 2>/dev/null || echo '{"error": "no provenance; code not pushed by box?"}' > "$RD/git.json"
cp "$REPO/.box_diff.patch" "$RD/diff.patch" 2>/dev/null || true
{
  echo "host: $(hostname)  started: $(date -u +%FT%TZ)"
  nvidia-smi -L 2>/dev/null || echo "no GPU"
  "$VENV/bin/python" -V
  "$UV" pip freeze --python "$VENV/bin/python" 2>/dev/null
} > "$RD/env.txt" 2>&1

cd "$REPO"
setsid nohup bash -c '
  rd=$1; syncd=$2; venv=$3; shift 3
  source "$venv/bin/activate"
  export PYTHONPATH=src PYTHONUNBUFFERED=1
  t0=$(date +%s)
  "$@" > "$rd/log" 2>&1
  rc=$?
  echo $rc > "$rd/exit_code"
  echo $(( $(date +%s) - t0 )) > "$rd/duration_s"
  bash "$syncd" --once > /dev/null 2>&1 || true
' _ "$RD" "$HERE/syncd.sh" "$VENV" "$@" < /dev/null > /dev/null 2>&1 &
echo $! > "$RD/pid"
echo "$ID"
