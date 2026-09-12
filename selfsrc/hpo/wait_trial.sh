#!/usr/bin/env bash
# Aspetta la fine di un trial lanciato con run_trial.sh e stampa SOLO il risultato:
#   selfsrc/hpo/wait_trial.sh t01 [intervallo_secondi=60]
# Se il trial e' finito bene stampa la riga JSON compatta; se e' morto stampa le ultime
# righe dell'errore. Non stampa mai i log per intero.
set -uo pipefail
cd "$(dirname "$0")/../.."
NAME="$1"; EVERY="${2:-60}"
LOGS=selfsrc/hpo/logs
PID=$(cat "$LOGS/$NAME.pid" 2>/dev/null || echo "")
[ -z "$PID" ] && { echo "nessun pid per $NAME"; exit 1; }
while kill -0 "$PID" 2>/dev/null; do sleep "$EVERY"; done
if [ -s "$LOGS/$NAME.out" ]; then
  tail -n 1 "$LOGS/$NAME.out"
else
  echo "TRIAL $NAME TERMINATO SENZA RISULTATO. Ultime righe di stderr:"
  grep -v -i "warn\|Loading weights\|deprecated\|use_cache" "$LOGS/$NAME.err" | tail -n 12
  exit 2
fi
