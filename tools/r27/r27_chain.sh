#!/bin/bash
# TASK_R27_001 fixed chain.  usage: r27_chain.sh <round> <release-name>
# Stages already closed PASS under the same round are skipped (re-entry after an engineering fix keeps their
# evidence; a stage directory left by a failure must be preserved under another name before re-entry).
# The formal stages only run if gate/GATE.json says promoted_formal.
set -u
R=$1
REL=$2
PY=/home/lyy/.cache/r0-seed0-e120-20261010-001/environment/bin/python
RUN="$R/$REL/source/tools/r27/r27_run.py"
export R27_ROUND="$R"
cd "$R" || exit 2
log() { echo "$(date -Is) $*" >> "$R/CHAIN.log"; }
stage() {
  local name=$1; shift
  if [ -f "$R/$name/CLOSED.json" ]; then log "SKIP $name (closed)"; return 0; fi
  log "START $name"
  "$@" > "$R/$name.log" 2>&1
  local rc=$?
  log "END $name rc=$rc"
  if [ $rc -ne 0 ]; then log "CHAIN STOP at $name"; exit $rc; fi
}
log "CHAIN START release=$REL pid=$$"
stage cpu env CUDA_VISIBLE_DEVICES= R27_CPU_THREADS=16 "$PY" "$RUN" cpu --out "$R/cpu"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for a in N C P; do stage "accept-$a" "$PY" "$RUN" accept --arm "$a" --out "$R/accept-$a"; done
for a in C P; do stage "diag-$a" "$PY" "$RUN" diag --arm "$a" --out "$R/diag-$a"; done
stage cost "$PY" "$RUN" cost --out "$R/cost"
stage gate "$PY" "$RUN" gate --out "$R/gate"
D=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['decision'])" "$R/gate/GATE.json")
log "GATE decision=$D"
if [ "$D" != "promoted_formal" ]; then log "CHAIN DONE (formal not started: $D)"; exit 0; fi
for a in C P; do stage "formal-$a" "$PY" "$RUN" formal --arm "$a" --out "$R/formal-$a"; done
for a in C P; do stage "eval-$a" "$PY" "$RUN" evaluate --arm "$a" --out "$R/eval-$a"; done
stage final-cost "$PY" "$RUN" final-cost --out "$R/final-cost"
stage summary "$PY" "$RUN" summarize --out "$R/summary"
log "CHAIN DONE"
