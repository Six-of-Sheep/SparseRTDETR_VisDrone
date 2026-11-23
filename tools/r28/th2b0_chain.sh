#!/usr/bin/env bash
# TH2 B0 (TASK_TH2_B0_001) chain: wiring -> demand on train_core -> T1 -> width selection -> development -> figures -> tables.
# usage: th2b0_chain.sh <round> <release>
# Stages that are already PASS are skipped; a directory without CLOSED.json stops the chain (no automatic retry).
# The chain never stops or pauses another process on the GPU; it logs what it sees.
set -u
R=$(readlink -f "$1")
REL=$2
PY=/home/lyy/.cache/r0-seed0-e120-20261010-001/environment/bin/python
RUN="$R/$REL/source/tools/r28/th2_b0.py"
LOG="$R/CHAIN.log"
GPU_UUID=GPU-1faee6f0-1da7-4ede-2475-67a5a00274a8

log() { echo "$(date '+%F %T %Z') $*" >> "$LOG"; }

exec 9> "$R/.chain.lock"
flock -n 9 || { echo "another chain holds $R/.chain.lock"; exit 1; }
[ -f "$R/CONTRACT.json" ] || { echo "no CONTRACT.json in $R"; exit 1; }
[ "$(nvidia-smi --query-gpu=uuid --format=csv,noheader)" = "$GPU_UUID" ] || { echo "GPU differs"; exit 1; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

cpu() { env CUDA_VISIBLE_DEVICES= "$PY" -I "$RUN" "$@" --round "$R"; }
gpu() { "$PY" -I "$RUN" "$@" --round "$R"; }

stage() {            # stage <name> <cpu|gpu>
  local name=$1 kind=$2
  if [ -f "$R/$name/CLOSED.json" ]; then log "skip $name (closed)"; return 0; fi
  if [ -e "$R/$name" ]; then log "STOP: $R/$name exists without CLOSED.json"; exit 1; fi
  log "start $name; GPU compute processes: $(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';')"
  if "$kind" "$name" > "$R/$name.log" 2>&1; then log "done $name"; else log "STOP: $name failed (see $name.log)"; exit 1; fi
}

log "CHAIN START release=$REL"
stage wire cpu
stage demand_tc gpu
stage tc cpu
stage select cpu
stage dev cpu
stage figs cpu
stage tables cpu
log "CHAIN DONE"
