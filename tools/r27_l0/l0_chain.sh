#!/bin/bash
# TASK_R27_L0_CONTROL_002 fixed chain (arm L0 only).
#
# usage: l0_chain.sh <round> <release-name>
#        l0_chain.sh <round> <release-name> --resume-formal <state.pth> <N>      (N >= 2: writes <round>/formal-L0-r<N>)
#
# Order: cpu -> zero -> accept -> formal-L0 -> eval-L0 -> cost4 -> summary.  Log: <round>/CHAIN.log; every stage writes
# its stdout/stderr to <round>/<stage>.log (an earlier log of the same stage is kept as <stage>.log.<k>).
#
# Start: the GPU UUID must equal the contract's gpu_uuid, and the GPU must have no compute process.  A busy GPU
# ends the chain with "CHAIN STOP (GPU busy)"; nothing is pre-empted and no other process is touched.
#
# Re-entry.  A stage directory with CLOSED.json (status PASS) is skipped.  A stage directory that exists without
# CLOSED.json stops the chain: nothing is overwritten or deleted, nothing is retried automatically.  Disposal:
#   * cpu / zero / accept / eval-L0 / cost4 / summary: keep the evidence by renaming the directory to
#     <stage>-fail-NNN, fix the cause, assemble what the fix requires, re-enter with the same command.
#   * formal-L0 that failed BEFORE the e060 save (the directory holds STOP.json and no usable checkpoint): rename it
#     to formal-L0-fail-NNN and re-enter with the same command.  A renamed directory is never taken for a formal
#     directory (only formal-L0 and formal-L0-rN, N >= 2, are candidates) and cannot be a resume source.
#   * formal-L0 (or formal-L0-rN) that failed AFTER a save: do NOT rename it (a resume source must be a state in a
#     formal-L0 / formal-L0-rN directory of this round).  Re-enter with
#         --resume-formal <round>/formal-L0[-rK]/checkpoint-epoch-XXX.pth <N>       (N: the next unused number >= 2)
#     The interrupted directory stays as it is; the summary lists it in formal_chain.
# formal: if exactly one formal-L0 / formal-L0-rN holds a PASS CLOSED.json it is skipped (also when --resume-formal
# is given again on a later re-entry); more than one is an error.  There is no automatic resume.
#
# Dev only: L0_DEV_RUNNER=<script> replaces l0_run.py for the stages named in L0_DEV_RUNNER_STAGES (default
# "formal-L0 eval-L0 cost4 summary"); L0_DEV_GPU_APPS replaces the compute-process query.  Either variable is refused
# unless the round directory name starts with "dev-round".
set -u
PY=/home/lyy/.cache/r0-seed0-e120-20261010-001/environment/bin/python
FORMAL_RE='^formal-L0(-r([2-9]|[1-9][0-9]+))?$'

usage() { echo "usage: l0_chain.sh <round> <release-name> [--resume-formal <state.pth> <N>]" >&2; exit 2; }
[ $# -eq 2 ] || [ $# -eq 5 ] || usage
R=${1%/}
REL=$2
RESUME_STATE=""
RESUME_N=""
if [ $# -eq 5 ]; then
  [ "$3" = "--resume-formal" ] || usage
  RESUME_STATE=$4
  RESUME_N=$5
  [[ "formal-L0-r$RESUME_N" =~ $FORMAL_RE ]] || { echo "--resume-formal: N must be an integer >= 2 without leading zero" >&2; exit 2; }
fi
[ -d "$R" ] || { echo "round directory missing: $R" >&2; exit 2; }
RUN="$R/$REL/source/tools/r27_l0/l0_run.py"
export L0_ROUND="$R"
cd "$R" || exit 2

log() { echo "$(date -Is) $*" >> "$R/CHAIN.log"; }
stop() { log "$1"; exit "${2:-3}"; }

is_dev_round() { case "$(basename "$R")" in dev-round*) return 0 ;; *) return 1 ;; esac; }

gpu_apps() {  # one line per compute process on the GPU; empty when the GPU is free
  if [ -n "${L0_DEV_GPU_APPS+x}" ]; then printf '%s' "$L0_DEV_GPU_APPS"; return 0; fi
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
}

closed_pass() { [ -f "$1/CLOSED.json" ] && grep -q '"status": "PASS"' "$1/CLOSED.json"; }

runner_for() {  # $1 = stage name
  local s
  if [ -n "${L0_DEV_RUNNER:-}" ]; then
    for s in ${L0_DEV_RUNNER_STAGES:-formal-L0 eval-L0 cost4 summary}; do
      [ "$s" = "$1" ] && { echo "$L0_DEV_RUNNER"; return 0; }
    done
  fi
  echo "$RUN"
}

keep_log() {  # never truncate the log of an earlier attempt
  local f="$R/$1.log" k=1
  [ -e "$f" ] || return 0
  while [ -e "$f.$k" ]; do k=$((k + 1)); done
  mv "$f" "$f.$k"
}

stage() {  # stage <directory name> <command...>
  local name=$1 rc
  shift
  if [ -e "$R/$name/CLOSED.json" ]; then
    closed_pass "$R/$name" || stop "CHAIN STOP at $name (CLOSED.json is not PASS)"
    log "SKIP $name (closed)"
    return 0
  fi
  [ -e "$R/$name" ] && stop "CHAIN STOP at $name (unclosed directory present; preserve it, then resume or rename)"
  keep_log "$name"
  log "START $name"
  "$@" > "$R/$name.log" 2>&1
  rc=$?
  log "END $name rc=$rc"
  [ $rc -eq 0 ] || stop "CHAIN STOP at $name" $rc
  closed_pass "$R/$name" || stop "CHAIN STOP at $name (rc=0 without a PASS CLOSED.json)"
}

formal_stage() {
  local d b closed="" n=0 target
  for d in "$R"/formal-L0 "$R"/formal-L0-r*; do
    [ -d "$d" ] || continue
    b=$(basename "$d")
    [[ "$b" =~ $FORMAL_RE ]] || continue
    if closed_pass "$d"; then closed="$closed $b"; n=$((n + 1)); fi
  done
  [ $n -le 1 ] || stop "CHAIN STOP at formal-L0 (more than one closed formal directory:$closed)"
  if [ $n -eq 1 ]; then log "SKIP formal-L0 (closed in$closed)"; return 0; fi
  if [ -n "$RESUME_STATE" ]; then
    target="formal-L0-r$RESUME_N"
    [ -f "$RESUME_STATE" ] || stop "CHAIN STOP at $target (resume state missing: $RESUME_STATE)"
    log "RESUME formal from $RESUME_STATE into $target"
    stage "$target" "$PY" "$(runner_for formal-L0)" formal --resume "$RESUME_STATE" --out "$R/$target"
    return 0
  fi
  if [ -e "$R/formal-L0" ]; then
    stop "CHAIN STOP at formal-L0 (unclosed directory present; preserve it, then resume or rename: after a saved checkpoint re-enter with --resume-formal <state> <N>; if it failed before the e060 save rename it to formal-L0-fail-NNN and re-enter)"
  fi
  stage formal-L0 "$PY" "$(runner_for formal-L0)" formal --out "$R/formal-L0"
}

exec 9> "$R/.chain.lock"
flock -n 9 || { log "CHAIN STOP (another chain holds $R/.chain.lock)"; exit 5; }
log "CHAIN START release=$REL pid=$$ resume_formal=${RESUME_STATE:-none}${RESUME_N:+ N=$RESUME_N}"
if [ -n "${L0_DEV_RUNNER:-}" ] || [ -n "${L0_DEV_GPU_APPS+x}" ]; then
  is_dev_round || stop "CHAIN STOP (L0_DEV_RUNNER / L0_DEV_GPU_APPS are refused outside a dev round)"
  log "DEV runner=${L0_DEV_RUNNER:-none} stages=${L0_DEV_RUNNER_STAGES:-formal-L0 eval-L0 cost4 summary} gpu_apps_stub=${L0_DEV_GPU_APPS+set}"
fi
[ -f "$RUN" ] || stop "CHAIN STOP (release runner missing: $RUN)"
[ -f "$R/CONTRACT.json" ] || stop "CHAIN STOP (CONTRACT.json missing: run prepare first)"
WANT=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['gpu_uuid'])" "$R/CONTRACT.json") || stop "CHAIN STOP (cannot read gpu_uuid from the contract)"
HAVE=$(nvidia-smi --query-gpu=uuid --format=csv,noheader) || stop "CHAIN STOP (nvidia-smi failed)"
log "GPU uuid=$(echo "$HAVE" | tr '\n' ' ')contract=$WANT"
[ "$HAVE" = "$WANT" ] || stop "CHAIN STOP (GPU UUID differs from the contract)"
APPS=$(gpu_apps) || stop "CHAIN STOP (GPU compute-process query failed)"
log "GPU compute processes: ${APPS:-none}"
[ -z "$APPS" ] || stop "CHAIN STOP (GPU busy)"

stage cpu env CUDA_VISIBLE_DEVICES= L0_CPU_THREADS=16 "$PY" "$(runner_for cpu)" cpu --out "$R/cpu"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
stage zero "$PY" "$(runner_for zero)" zero --out "$R/zero"
stage accept "$PY" "$(runner_for accept)" accept --out "$R/accept"
formal_stage
stage eval-L0 "$PY" "$(runner_for eval-L0)" evaluate --out "$R/eval-L0"
stage cost4 "$PY" "$(runner_for cost4)" cost4 --out "$R/cost4"
stage summary "$PY" "$(runner_for summary)" summarize --out "$R/summary"
log "CHAIN DONE"
