#!/usr/bin/env bash
# TH2 C0 (TASK_TH2_C0_001) chain: storage estimate -> release -> contract -> wiring -> pure endpoints (wiring, scored)
#   -> hit-level tables -> routed scorings -> tables.
# usage: TH2C0_TASK=<path of TASK_TH2_C0_001.md> th2c0_chain.sh <home> [dev-round-N]
#   Run from a work tree of claude/th2-c0.  <home> is the formal round; a dev round is <home>/dev-round-N and runs the
#   wiring checks only (task section 8).
# The first step writes <home>/STORAGE_ESTIMATE.json as the first file of <home>; every later step refuses to run
# without it.  CPU only: CUDA_VISIBLE_DEVICES is empty for every process.  At most 3 scoring processes at a time.
# Stages that are already PASS are skipped; a directory without CLOSED.json stops the chain (no automatic retry).
set -u
WT=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
HOME_ARG=$1
DEVR=${2:-}
REL=release-001
PY=/home/lyy/.cache/r0-seed0-e120-20261010-001/environment/bin/python
TASK_SRC=${TH2C0_TASK:?TH2C0_TASK must name the task file}
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

if [ ! -e "$HOME_ARG" ]; then
  mkdir "$HOME_ARG" || exit 1
  "$PY" -I "$WT/tools/r28/th2_c0.py" storage --home "$HOME_ARG" || { echo "storage estimate failed"; exit 1; }
fi
[ -f "$HOME_ARG/STORAGE_ESTIMATE.json" ] || { echo "no STORAGE_ESTIMATE.json in $HOME_ARG: refusing to run"; exit 1; }
H=$(readlink -f "$HOME_ARG")
[ -f "$H/TASK_TH2_C0_001.md" ] || cp "$TASK_SRC" "$H/TASK_TH2_C0_001.md" || exit 1
if [ -n "$DEVR" ]; then
  case "$DEVR" in dev-round*) ;; *) echo "the second argument must be dev-round*"; exit 1 ;; esac
  R="$H/$DEVR"
  mkdir -p "$R" || exit 1
else
  R="$H"
fi
TAG=${DEVR:-formal}
LOG="$R/CHAIN.log"
RUN="$R/$REL/source/tools/r28/th2_c0.py"

log() { echo "$(date '+%F %T %Z') $*" >> "$LOG"; }

exec 9> "$R/.chain.lock"
flock -n 9 || { echo "another chain holds $R/.chain.lock"; exit 1; }
echo "$TAG start $(date +%s)" >> "$H/chain-runs.log"
trap 'echo "$TAG end $(date +%s)" >> "$H/chain-runs.log"' EXIT

log "CHAIN START round=$TAG work_tree=$WT head=$(git -C "$WT" rev-parse HEAD) load=$(cut -d' ' -f1-3 /proc/loadavg) cpus=$(nproc)"
if [ ! -d "$R/$REL" ]; then
  "$PY" -I "$WT/tools/r27_l0/l0_release.py" "$WT" "$R" "$REL" > "$R/release.log" 2>&1 || { log "STOP: release failed (see release.log)"; exit 1; }
  log "release $REL written"
fi
if [ ! -f "$R/CONTRACT.json" ]; then
  (cd "$R/$REL/source" && "$PY" -I "$RUN" prepare --round "$R" --release "$REL") > "$R/prepare.log" 2>&1 || { log "STOP: prepare failed (see prepare.log)"; exit 1; }
  log "contract written"
fi

stage() {            # stage <directory name> <action> [--unit u]
  local name=$1; shift
  if [ -f "$R/$name/CLOSED.json" ]; then log "skip $name (closed)"; return 0; fi
  if [ -e "$R/$name" ]; then log "STOP: $R/$name exists without CLOSED.json"; return 1; fi
  log "start $name"
  if "$PY" -I "$RUN" "$@" --round "$R" > "$R/$name.log" 2>&1; then log "done $name"; else log "STOP: $name failed (see $name.log)"; return 1; fi
}

seeds() {            # seeds <action>: the three seeds side by side (3 processes)
  local action=$1 rc=0 s pids=""
  for s in 0 1 2; do stage "$action-s$s" "$action" --unit "s$s" & pids="$pids $!"; done
  for s in $pids; do wait "$s" || rc=1; done
  return $rc
}

stage wire wire || exit 1
seeds ends || exit 1
if [ -n "$DEVR" ]; then log "CHAIN DONE (dev round: wiring checks only)"; exit 0; fi
stage hits hits || exit 1
seeds routed || exit 1
stage tables tables || exit 1
log "CHAIN DONE"
