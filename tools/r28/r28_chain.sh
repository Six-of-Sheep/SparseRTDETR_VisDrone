#!/bin/bash
# TASK_R28_001 + TASK_R28_CODEX_AMENDMENT_002 fixed chain.
#
# usage: r28_chain.sh <round> <release-name>
#        r28_chain.sh <round> <release-name> --resume-formal <W|K> <state.pth> <N>   (N >= 2: writes formal-<arm>-r<N>)
#
# Order:
#   cpu -> demand -> prior -> g0            [stop here when G0 is not passed: the W run is not opened]
#   zero -> accept-W -> evalpath -> precost [stop here when the pre-launch cost ratio is above the limit]
#   formal-W -> eval-W -> cost -> summary-W [stop here when the registered rule does not trigger K]
#   accept-K -> formal-K -> eval-K -> cost-K -> summary-K
# Log: <round>/CHAIN.log; every stage writes <round>/<stage>.log (an earlier log is kept as <stage>.log.<k>).
#
# Start: the GPU UUID must equal the contract's gpu_uuid and the GPU must have no compute process; a busy GPU ends
# the chain, nothing is pre-empted.  Re-entry: a stage directory with a PASS CLOSED.json is skipped; a stage
# directory without CLOSED.json stops the chain (rename it to <stage>-fail-NNN to keep the evidence, fix the cause,
# re-enter).  A formal directory that failed after a save is not renamed: re-enter with --resume-formal and an
# epoch-boundary state of that directory.  Nothing is retried, overwritten or deleted automatically.
#
# Dev rounds (directory name dev-round*) may set R28_DEV_* variables; they are refused elsewhere by the runner.
set -u
PY=/home/lyy/.cache/r0-seed0-e120-20261010-001/environment/bin/python

usage() { echo "usage: r28_chain.sh <round> <release-name> [--resume-formal <W|K> <state.pth> <N>]" >&2; exit 2; }
[ $# -eq 2 ] || [ $# -eq 6 ] || usage
R=${1%/}
REL=$2
RESUME_ARM=""; RESUME_STATE=""; RESUME_N=""
if [ $# -eq 6 ]; then
  [ "$3" = "--resume-formal" ] || usage
  RESUME_ARM=$4; RESUME_STATE=$5; RESUME_N=$6
  [ "$RESUME_ARM" = "W" ] || [ "$RESUME_ARM" = "K" ] || usage
  [[ "$RESUME_N" =~ ^([2-9]|[1-9][0-9]+)$ ]] || { echo "--resume-formal: N must be an integer >= 2" >&2; exit 2; }
fi
[ -d "$R" ] || { echo "round directory missing: $R" >&2; exit 2; }
[ -z "${L0_ROUND+x}" ] || { echo "L0_ROUND must not be set" >&2; exit 2; }
RUN="$R/$REL/source/tools/r28/r28_run.py"
export R28_ROUND="$R"
cd "$R" || exit 2

log() { echo "$(date -Is) $*" >> "$R/CHAIN.log"; }
stop() { log "$1"; exit "${2:-3}"; }
closed_pass() { [ -f "$1/CLOSED.json" ] && grep -q '"status": "PASS"' "$1/CLOSED.json"; }
flag() { "$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))[sys.argv[2]])" "$R/$1/CLOSED.json" "$2"; }

keep_log() {
  local f="$R/$1.log" k=1
  [ -e "$f" ] || return 0
  while [ -e "$f.$k" ]; do k=$((k + 1)); done
  mv "$f" "$f.$k"
}

stage() {  # stage <arm> <directory name> <env...> -- <runner arguments...>
  local arm=$1 name=$2 rc
  shift 2
  if [ -e "$R/$name/CLOSED.json" ]; then
    closed_pass "$R/$name" || stop "CHAIN STOP at $name (CLOSED.json is not PASS)"
    log "SKIP $name (closed)"
    return 0
  fi
  [ -e "$R/$name" ] && stop "CHAIN STOP at $name (unclosed directory present; preserve it, then rename or resume)"
  keep_log "$name"
  log "START $name"
  env R28_ARM="$arm" "$@" > "$R/$name.log" 2>&1 9>&-
  rc=$?
  log "END $name rc=$rc"
  [ $rc -eq 0 ] || stop "CHAIN STOP at $name" $rc
  closed_pass "$R/$name" || stop "CHAIN STOP at $name (rc=0 without a PASS CLOSED.json)"
}

formal_stage() {  # formal_stage <arm>
  local arm=$1 d b closed="" n=0 target re
  re="^formal-$arm(-r([2-9]|[1-9][0-9]+))?$"
  for d in "$R"/formal-"$arm" "$R"/formal-"$arm"-r*; do
    [ -d "$d" ] || continue
    b=$(basename "$d")
    [[ "$b" =~ $re ]] || continue
    if closed_pass "$d"; then closed="$closed $b"; n=$((n + 1)); fi
  done
  [ $n -le 1 ] || stop "CHAIN STOP at formal-$arm (more than one closed formal directory:$closed)"
  if [ $n -eq 1 ]; then log "SKIP formal-$arm (closed in$closed)"; return 0; fi
  if [ -n "$RESUME_STATE" ] && [ "$RESUME_ARM" = "$arm" ]; then
    target="formal-$arm-r$RESUME_N"
    [ -f "$RESUME_STATE" ] || stop "CHAIN STOP at $target (resume state missing: $RESUME_STATE)"
    log "RESUME formal-$arm from $RESUME_STATE into $target"
    stage "$arm" "$target" "$PY" "$RUN" formal --resume "$RESUME_STATE" --out "$R/$target"
    return 0
  fi
  if [ -e "$R/formal-$arm" ]; then
    stop "CHAIN STOP at formal-$arm (unclosed directory present: after a saved epoch state re-enter with --resume-formal $arm <state> <N>; if it failed before the e060 save rename it to formal-$arm-fail-NNN)"
  fi
  stage "$arm" "formal-$arm" "$PY" "$RUN" formal --out "$R/formal-$arm"
}

exec 9> "$R/.chain.lock"
flock -n 9 || { log "CHAIN STOP (another chain holds $R/.chain.lock)"; exit 5; }
log "CHAIN START release=$REL pid=$$ resume=${RESUME_ARM:-none} ${RESUME_STATE:-} ${RESUME_N:-}"
case "$(basename "$R")" in dev-round*) log "DEV round: $(env | grep '^R28_DEV_' | tr '\n' ' ')" ;; esac
[ -f "$RUN" ] || stop "CHAIN STOP (release runner missing: $RUN)"
[ -f "$R/CONTRACT.json" ] || stop "CHAIN STOP (CONTRACT.json missing: run prepare first)"
WANT=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['gpu_uuid'])" "$R/CONTRACT.json") || stop "CHAIN STOP (cannot read gpu_uuid from the contract)"
HAVE=$(nvidia-smi --query-gpu=uuid --format=csv,noheader) || stop "CHAIN STOP (nvidia-smi failed)"
log "GPU uuid=$(echo "$HAVE" | tr '\n' ' ')contract=$WANT"
[ "$HAVE" = "$WANT" ] || stop "CHAIN STOP (GPU UUID differs from the contract)"
APPS=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader) || stop "CHAIN STOP (GPU compute-process query failed)"
log "GPU compute processes: ${APPS:-none}"
[ -z "$APPS" ] || stop "CHAIN STOP (GPU busy)"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
stage W cpu env CUDA_VISIBLE_DEVICES= R28_CPU_THREADS=16 "$PY" "$RUN" cpu --out "$R/cpu"
stage W demand "$PY" "$RUN" demand --out "$R/demand"
stage W prior env CUDA_VISIBLE_DEVICES= "$PY" "$RUN" prior --out "$R/prior"
stage W g0 "$PY" "$RUN" g0 --out "$R/g0"
G0=$(flag g0 G0_passed) || stop "CHAIN STOP (cannot read G0)"
log "G0 passed=$G0 small_geomean=$(flag g0 G0_small_geomean_predicted)"
if [ "$G0" != "True" ] && [ "${R28_DEV_IGNORE_G0:-}" != "1" ]; then stop "CHAIN END (G0 not passed: the W run is not opened)" 0; fi
stage W zero "$PY" "$RUN" zero --out "$R/zero"
stage W accept-W "$PY" "$RUN" accept --out "$R/accept-W"
stage W evalpath "$PY" "$RUN" evalpath --out "$R/evalpath"
stage W precost "$PY" "$RUN" precost --out "$R/precost"
CP=$(flag precost C_pre_passed) || stop "CHAIN STOP (cannot read the pre-launch cost)"
log "precost passed=$CP ratio=$(flag precost C_pre_model_gpu_W_over_L0)"
if [ "$CP" != "True" ] && [ "${R28_DEV_IGNORE_COST:-}" != "1" ]; then stop "CHAIN END (pre-launch cost above the limit: the W run is not opened)" 0; fi
formal_stage W
stage W eval-W "$PY" "$RUN" evaluate --out "$R/eval-W"
stage W cost "$PY" "$RUN" cost --out "$R/cost"
stage W summary-W env CUDA_VISIBLE_DEVICES= "$PY" "$RUN" summarize --out "$R/summary-W"
KT=$(flag summary-W K_trigger) || stop "CHAIN STOP (cannot read the K trigger)"
log "summary-W label=$(flag summary-W label_key) K_trigger=$KT"
if [ "$KT" != "True" ] && [ "${R28_DEV_FORCE_K:-}" != "1" ]; then stop "CHAIN END (K not triggered)" 0; fi
stage K accept-K "$PY" "$RUN" accept --out "$R/accept-K"
formal_stage K
stage K eval-K "$PY" "$RUN" evaluate --out "$R/eval-K"
stage K cost-K "$PY" "$RUN" cost --out "$R/cost-K"
stage K summary-K env CUDA_VISIBLE_DEVICES= "$PY" "$RUN" summarize --out "$R/summary-K"
log "CHAIN DONE"
