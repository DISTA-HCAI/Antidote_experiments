#!/usr/bin/env bash
# Lancia UN trial in background, in modo silenzioso, e ritorna subito.
#   selfsrc/hpo/run_trial.sh t01 "nota breve" --set training.k_steps=8 --set ...
# Output: stdout del processo -> selfsrc/hpo/logs/<nome>.out (SOLO la riga JSON finale)
#         stderr             -> selfsrc/hpo/logs/<nome>.err (warning, traceback)
#         pid                -> selfsrc/hpo/logs/<nome>.pid
# Rifiuta di partire se un altro trial e' ancora in esecuzione (mai due run insieme: RAM).
set -euo pipefail
cd "$(dirname "$0")/../.."
NAME="$1"; NOTE="${2:-}"; shift 2 || shift $#
LOGS=selfsrc/hpo/logs; mkdir -p "$LOGS"
for pidf in "$LOGS"/*.pid; do
  [ -f "$pidf" ] || continue
  if kill -0 "$(cat "$pidf")" 2>/dev/null; then
    echo "ERRORE: trial $(basename "$pidf" .pid) ancora in esecuzione (pid $(cat "$pidf")). Aspettalo con wait_trial.sh." >&2
    exit 1
  fi
done
CFG="${HPO_CONFIG:-selfsrc/hpo/base_hpo.json}"
PY="${PYTHON:-.venv/bin/python}"
: > "$LOGS/$NAME.out"
nohup "$PY" -m selfsrc.run --config "$CFG" --name "$NAME" --note "$NOTE" --quiet "$@" \
  > "$LOGS/$NAME.out" 2> "$LOGS/$NAME.err" &
echo $! > "$LOGS/$NAME.pid"
echo "avviato $NAME (pid $!) con: $*"
