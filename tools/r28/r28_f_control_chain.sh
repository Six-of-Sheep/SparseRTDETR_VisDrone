#!/bin/bash
# TASK_R28_FOVEA_MAPPING_CONTROL_004: one arm F (arm W's frozen demand network with the FOVEA-style map), added after
# the W and K results were seen.
#
# usage: r28_f_control_chain.sh <round> <release-name>
#        r28_f_control_chain.sh <round> <release-name> --resume-formal <state.pth> <N>   (N >= 2: writes formal-F-rN)
#
# Order:  fovea-check (CPU) -> wire-F -> accept-F -> formal-F -> eval-F -> cost-F (L0, W, F) -> summary-F (CPU)
# The round must carry the mapping-control contract (prepare-fovea).  Nothing of the closed W round or of the K round
# is rerun or written: no demand training, no G0, no W / K training, no rescoring of stored predictions.  The closed
# W decision is neither a trigger nor changed.  Every R28_DEV_* variable is refused.  A dev round (dev-round* inside
# the round directory) is accepted only with --until: it shakes out the checks and never trains, evaluates or scores.
# Log: <round>/CHAIN.log; every stage writes <round>/<stage>.log (an earlier log is kept as <stage>.log.<k>).
#
# Start: the GPU UUID must equal the contract's gpu_uuid and the GPU must have no compute process; a busy GPU ends
# the chain, nothing is pre-empted.  Re-entry: a stage directory with a PASS CLOSED.json is skipped; a stage directory
# without CLOSED.json stops the chain (rename it to <stage>-fail-NNN to keep the evidence, fix the cause, re-enter).
# A formal directory that failed after a save is not renamed: re-enter with --resume-formal and an epoch-boundary
# state of that directory.  Nothing is retried, overwritten or deleted automatically.
#
# --until <stage> (optional, last two arguments): leave the chain after that stage closed.  Used to run the checks
# first; a later call without it continues from the closed stages.
set -u
PY=/home/lyy/.cache/r0-seed0-e120-20261010-001/environment/bin/python
ARM=F

usage() { echo "usage: r28_f_control_chain.sh <round> <release-name> [--resume-formal <state.pth> <N>] [--until <stage>]" >&2; exit 2; }
UNTIL=""
if [ $# -ge 4 ] && [ "${@: -2:1}" = "--until" ]; then
  UNTIL="${@: -1}"
  set -- "${@:1:$(($# - 2))}"
  case "$UNTIL" in fovea-check|wire-F|accept-F) ;; *) echo "--until takes fovea-check, wire-F or accept-F" >&2; exit 2 ;; esac
fi
[ $# -eq 2 ] || [ $# -eq 5 ] || usage
R=${1%/}
REL=$2
RESUME_STATE=""; RESUME_N=""
if [ $# -eq 5 ]; then
  [ "$3" = "--resume-formal" ] || usage
  RESUME_STATE=$4; RESUME_N=$5
  [[ "$RESUME_N" =~ ^([2-9]|[1-9][0-9]+)$ ]] || { echo "--resume-formal: N must be an integer >= 2" >&2; exit 2; }
fi
[ -d "$R" ] || { echo "round directory missing: $R" >&2; exit 2; }
[ -z "${L0_ROUND+x}" ] || { echo "L0_ROUND must not be set" >&2; exit 2; }
if env | grep -q '^R28_DEV_'; then echo "R28_DEV_* variables are refused by the mapping-control chain" >&2; exit 2; fi
# a dev round (directory name dev-round*, inside the round) only shakes out the checks: it must stop before formal-F
case "$(basename "$R")" in dev-round*) [ -n "$UNTIL" ] || { echo "a dev round of the mapping control needs --until (no formal training, evaluation or scoring)" >&2; exit 2; } ;; esac
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

stage() {  # stage <directory name> <env...> -- <runner arguments...>
  local name=$1 rc
  shift 1
  if [ -e "$R/$name/CLOSED.json" ]; then
    closed_pass "$R/$name" || stop "CHAIN STOP at $name (CLOSED.json is not PASS)"
    log "SKIP $name (closed)"
  else
    [ -e "$R/$name" ] && stop "CHAIN STOP at $name (unclosed directory present; preserve it, then rename or resume)"
    keep_log "$name"
    log "START $name"
    env R28_ARM="$ARM" "$@" > "$R/$name.log" 2>&1 9>&-
    rc=$?
    log "END $name rc=$rc"
    [ $rc -eq 0 ] || stop "CHAIN STOP at $name" $rc
    closed_pass "$R/$name" || stop "CHAIN STOP at $name (rc=0 without a PASS CLOSED.json)"
  fi
  if [ "$name" = "$UNTIL" ]; then log "CHAIN END (--until $UNTIL reached; later stages not started)"; exit 0; fi
  return 0
}

formal_stage() {
  local d b closed="" n=0 target re
  re="^formal-$ARM(-r([2-9]|[1-9][0-9]+))?$"
  for d in "$R"/formal-"$ARM" "$R"/formal-"$ARM"-r*; do
    [ -d "$d" ] || continue
    b=$(basename "$d")
    [[ "$b" =~ $re ]] || continue
    if closed_pass "$d"; then closed="$closed $b"; n=$((n + 1)); fi
  done
  [ $n -le 1 ] || stop "CHAIN STOP at formal-$ARM (more than one closed formal directory:$closed)"
  if [ $n -eq 1 ]; then log "SKIP formal-$ARM (closed in$closed)"; return 0; fi
  if [ -n "$RESUME_STATE" ]; then
    target="formal-$ARM-r$RESUME_N"
    [ -f "$RESUME_STATE" ] || stop "CHAIN STOP at $target (resume state missing: $RESUME_STATE)"
    log "RESUME formal-$ARM from $RESUME_STATE into $target"
    stage "$target" "$PY" "$RUN" formal --resume "$RESUME_STATE" --out "$R/$target"
    return 0
  fi
  if [ -e "$R/formal-$ARM" ]; then
    stop "CHAIN STOP at formal-$ARM (unclosed directory present: after a saved epoch state re-enter with --resume-formal <state> <N>; if it failed before the e060 save rename it to formal-$ARM-fail-NNN)"
  fi
  stage "formal-$ARM" "$PY" "$RUN" formal --out "$R/formal-$ARM"
}

exec 9> "$R/.chain.lock"
flock -n 9 || { log "CHAIN STOP (another chain holds $R/.chain.lock)"; exit 5; }
log "CHAIN START (F mapping control) release=$REL pid=$$ resume=${RESUME_STATE:-none} ${RESUME_N:-} until=${UNTIL:-end}"
[ -f "$RUN" ] || stop "CHAIN STOP (release runner missing: $RUN)"
[ -f "$R/CONTRACT.json" ] || stop "CHAIN STOP (CONTRACT.json missing: run prepare-fovea first)"
BASIS=$("$PY" -c "import json,sys;c=json.load(open(sys.argv[1]));print(c.get('authorization_basis'), c['followup']['authorization_basis'], c['followup']['arm'], c['followup']['prior_W_decision']['K_trigger'])" "$R/CONTRACT.json") || stop "CHAIN STOP (not a mapping-control contract)"
log "contract: authorization_basis / follow-up basis / arm / prior K_trigger = $BASIS"
[ "$BASIS" = "post_K_user_forwarded_mapping_control post_K_user_forwarded_mapping_control F False" ] || stop "CHAIN STOP (the contract is not the registered F mapping control)"
WANT=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['gpu_uuid'])" "$R/CONTRACT.json") || stop "CHAIN STOP (cannot read gpu_uuid from the contract)"
HAVE=$(nvidia-smi --query-gpu=uuid --format=csv,noheader) || stop "CHAIN STOP (nvidia-smi failed)"
log "GPU uuid=$(echo "$HAVE" | tr '\n' ' ')contract=$WANT"
[ "$HAVE" = "$WANT" ] || stop "CHAIN STOP (GPU UUID differs from the contract)"
APPS=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader) || stop "CHAIN STOP (GPU compute-process query failed)"
log "GPU compute processes: ${APPS:-none}"
[ -z "$APPS" ] || stop "CHAIN STOP (GPU busy)"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
stage fovea-check env CUDA_VISIBLE_DEVICES= R28_CPU_THREADS=16 "$PY" "$RUN" fovea-check --out "$R/fovea-check"
stage wire-F "$PY" "$RUN" wire --out "$R/wire-F"
stage accept-F "$PY" "$RUN" accept --out "$R/accept-F"
formal_stage
stage eval-F "$PY" "$RUN" evaluate --out "$R/eval-F"
stage cost-F "$PY" "$RUN" cost --out "$R/cost-F"
stage summary-F env CUDA_VISIBLE_DEVICES= "$PY" "$RUN" summarize --out "$R/summary-F"
log "summary-F label=$(flag summary-F label) W_minus_F_APS600_pp=$(flag summary-F W_minus_F_APS600_pp) W_minus_F_AP75_small600_pp=$(flag summary-F W_minus_F_AP75_small600_pp)"
log "CHAIN DONE"
