#!/usr/bin/env bash
# TH2 phase 1 (TASK_TH2_PHASE1_001) chain: wiring -> demand -> geometry (amplitude) -> E0 -> E1 -> R1-R4 -> E2 -> scoring -> E3 -> tables.
# usage: th2_chain.sh <round> <release>
# Stages that are already PASS are skipped; a directory without CLOSED.json stops the chain (no automatic retry).
# The chain never stops or pauses another process on the GPU; it logs what it sees.
set -u
R=$(readlink -f "$1")
REL=$2
PY=/home/lyy/.cache/r0-seed0-e120-20261010-001/environment/bin/python
RUN="$R/$REL/source/tools/r28/th2_phase1.py"
LOG="$R/CHAIN.log"
GPU_UUID=GPU-1faee6f0-1da7-4ede-2475-67a5a00274a8

log() { echo "$(date '+%F %T %Z') $*" >> "$LOG"; }

exec 9> "$R/.chain.lock"
flock -n 9 || { echo "another chain holds $R/.chain.lock"; exit 1; }
[ -f "$R/CONTRACT.json" ] || { echo "no CONTRACT.json in $R"; exit 1; }
[ "$(nvidia-smi --query-gpu=uuid --format=csv,noheader)" = "$GPU_UUID" ] || { echo "GPU differs"; exit 1; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cpu() { env CUDA_VISIBLE_DEVICES= "$PY" -I "$RUN" "$@" --round "$R"; }
gpu() { "$PY" -I "$RUN" "$@" --round "$R"; }

stage() {            # stage <directory name> <cpu|gpu> <action> [--unit u]
  local name=$1 kind=$2; shift 2
  if [ -f "$R/$name/CLOSED.json" ]; then log "skip $name (closed)"; return 0; fi
  if [ -e "$R/$name" ]; then log "STOP: $R/$name exists without CLOSED.json"; exit 1; fi
  log "start $name; GPU compute processes: $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';')"
  if "$kind" "$@" > "$R/$name.log" 2>&1; then log "done $name"; else log "STOP: $name failed (see $name.log)"; exit 1; fi
}

log "CHAIN START release=$REL"
for s in 0 1 2; do stage "wire-FG-s$s" gpu wire --unit "FG-s$s"; done
stage demand gpu demand
stage geom cpu geom
stage e0 cpu e0
for s in 0 1 2; do stage "e1-s$s" gpu e1 --unit "s$s"; done
stage r14 cpu r14
for u in b-s1 b-s2 c-s0 c-s1 c-s2 d-s0 d-s1 d-s2; do stage "e2-$u" gpu e2 --unit "$u"; done
for u in b-s1 b-s2 c-s0 c-s1 c-s2 d-s0 d-s1 d-s2; do stage "score-$u" cpu score --unit "$u"; done
stage e3 cpu e3
stage tables cpu tables
log "CHAIN DONE"
