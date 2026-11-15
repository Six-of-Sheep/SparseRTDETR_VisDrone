#!/bin/bash
# TASK_R28_CALIBRATED_F_CONTROL_006: one arm FS ("F*": arm F with one calibrated global kernel width), added after the
# W, K and F results were seen, together with the input-pipeline acceptance of ENGINEERING_INPUT_OVERLAP_NEXT_005.
#
# usage: r28_cal_chain.sh <round> <release-name>
#        r28_cal_chain.sh <round> <release-name> --resume-formal <state.pth> <N>   (N >= 2: writes formal-FS-rN)
#
# Order:  cal-check (CPU) -> calibrate (demand network only) -> cal-fixture (CPU) -> wire-FS -> pipeline
#         -> formal-FS -> eval-FS -> cost-FS (L0, K, F, FS, W) -> summary-FS (CPU)
# The round must carry the calibrated-control contract (prepare-cal).  Nothing of the closed W round, of the K round
# or of the F round is rerun or written: no demand training, no W / K / F / L0 training, no rescoring.
#
# calibrate: a status other than "matched" (calibration_invalid, calibration_unmatched, confirmation_invalid) ends
# the chain after that stage.  No formal training, no other width, range, target or parameter is tried.
# pipeline: the formal run uses the overlapped input path only if the closed acceptance enabled it; R28_PIPELINE=serial
# in the environment keeps the serial reference path (nothing else is accepted in that variable).
#
# Dev flags: every R28_DEV_* variable is refused, except in a dev round (dev-round* inside the round directory), where
# R28_DEV_CAL_WINDOWS, R28_DEV_PIPE and R28_DEV_SPAN shorten the calibration set, the acceptance and the formal span.
# A dev round needs --until and never evaluates or scores.
# Log: <round>/CHAIN.log; every stage writes <round>/<stage>.log (an earlier log is kept as <stage>.log.<k>).
#
# Start: the GPU UUID must equal the contract's gpu_uuid and the GPU must have no compute process; a busy GPU ends
# the chain, nothing is pre-empted.  Re-entry: a stage directory with a PASS CLOSED.json is skipped; a stage directory
# without CLOSED.json stops the chain (rename it to <stage>-fail-NNN to keep the evidence, fix the cause, re-enter).
# A formal directory that failed after a save is not renamed: re-enter with --resume-formal and an epoch-boundary
# state of that directory.  Nothing is retried, overwritten or deleted automatically.
#
# --until <stage> (optional, last two arguments): leave the chain after that stage closed.
set -u
PY=/home/lyy/.cache/r0-seed0-e120-20261010-001/environment/bin/python
ARM=FS

usage() { echo "usage: r28_cal_chain.sh <round> <release-name> [--resume-formal <state.pth> <N>] [--until <stage>]" >&2; exit 2; }
UNTIL=""
if [ $# -ge 4 ] && [ "${@: -2:1}" = "--until" ]; then
  UNTIL="${@: -1}"
  set -- "${@:1:$(($# - 2))}"
  case "$UNTIL" in cal-check|calibrate|cal-fixture|wire-FS|pipeline|formal-FS) ;;
    *) echo "--until takes cal-check, calibrate, cal-fixture, wire-FS, pipeline or formal-FS" >&2; exit 2 ;; esac
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
DEVROUND=0
case "$(basename "$R")" in dev-round*) DEVROUND=1 ;; esac
if [ $DEVROUND -eq 1 ]; then
  [ -n "$UNTIL" ] || { echo "a dev round of the calibrated control needs --until (it never evaluates or scores)" >&2; exit 2; }
  if env | grep '^R28_DEV_' | grep -qvE '^R28_DEV_(CAL_WINDOWS|PIPE|SPAN)='; then
    echo "a dev round of the calibrated control only accepts R28_DEV_CAL_WINDOWS, R28_DEV_PIPE and R28_DEV_SPAN" >&2; exit 2
  fi
elif env | grep -q '^R28_DEV_'; then
  echo "R28_DEV_* variables are refused outside a dev round" >&2; exit 2
fi
if [ -n "${R28_PIPELINE+x}" ] && [ "$R28_PIPELINE" != "serial" ]; then
  echo "R28_PIPELINE can only be 'serial' (the reference path)" >&2; exit 2
fi
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

stage() {  # stage <directory name> <command...>
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
  return 0
}

until_reached() { if [ "$1" = "$UNTIL" ]; then log "CHAIN END (--until $UNTIL reached; later stages not started)"; exit 0; fi; }

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
log "CHAIN START (F* calibrated mapping control) release=$REL pid=$$ resume=${RESUME_STATE:-none} ${RESUME_N:-} until=${UNTIL:-end} pipeline_override=${R28_PIPELINE:-none} dev=$(env | grep '^R28_DEV_' | tr '\n' ' ')"
[ -f "$RUN" ] || stop "CHAIN STOP (release runner missing: $RUN)"
[ -f "$R/CONTRACT.json" ] || stop "CHAIN STOP (CONTRACT.json missing: run prepare-cal first)"
BASIS=$("$PY" -c "import json,sys;c=json.load(open(sys.argv[1]));print(c.get('authorization_basis'), c['followup']['authorization_basis'], c['followup']['arm'], c['followup']['prior_W_decision']['K_trigger'])" "$R/CONTRACT.json") || stop "CHAIN STOP (not a calibrated-control contract)"
log "contract: authorization_basis / follow-up basis / arm / prior K_trigger = $BASIS"
[ "$BASIS" = "post_F_user_forwarded_calibrated_mapping_control post_F_user_forwarded_calibrated_mapping_control FS False" ] || stop "CHAIN STOP (the contract is not the registered F* calibrated mapping control)"
WANT=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['gpu_uuid'])" "$R/CONTRACT.json") || stop "CHAIN STOP (cannot read gpu_uuid from the contract)"
HAVE=$(nvidia-smi --query-gpu=uuid --format=csv,noheader) || stop "CHAIN STOP (nvidia-smi failed)"
log "GPU uuid=$(echo "$HAVE" | tr '\n' ' ')contract=$WANT"
[ "$HAVE" = "$WANT" ] || stop "CHAIN STOP (GPU UUID differs from the contract)"
APPS=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader) || stop "CHAIN STOP (GPU compute-process query failed)"
log "GPU compute processes: ${APPS:-none}"
[ -z "$APPS" ] || stop "CHAIN STOP (GPU busy)"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
stage cal-check env CUDA_VISIBLE_DEVICES= R28_CPU_THREADS=16 "$PY" "$RUN" cal-check --out "$R/cal-check"
until_reached cal-check
stage calibrate "$PY" "$RUN" calibrate --out "$R/calibrate"
STATUS=$(flag calibrate calibration_status) || stop "CHAIN STOP (cannot read the calibration status)"
log "calibrate status=$STATUS h_star=$(flag calibrate h_star) h_star_hex=$(flag calibrate h_star_hex) L_W=$(flag calibrate L_W) relative_error=$(flag calibrate relative_error)"
if [ "$STATUS" != "matched" ]; then
  log "CHAIN END (calibration status $STATUS: the control cannot be built; no formal training, nothing else is tried)"
  exit 0
fi
until_reached calibrate
stage cal-fixture env CUDA_VISIBLE_DEVICES= R28_CPU_THREADS=16 "$PY" "$RUN" cal-fixture --out "$R/cal-fixture"
until_reached cal-fixture
stage wire-FS "$PY" "$RUN" cal-wire --out "$R/wire-FS"
until_reached wire-FS
stage pipeline "$PY" "$RUN" pipeline --out "$R/pipeline"
log "pipeline overlap_enabled=$(flag pipeline overlap_enabled) (formal input path: $([ "${R28_PIPELINE:-}" = "serial" ] && echo "serial, forced by R28_PIPELINE" || echo "by the acceptance decision"))"
until_reached pipeline
formal_stage
until_reached formal-FS
stage eval-FS "$PY" "$RUN" evaluate --out "$R/eval-FS"
stage cost-FS "$PY" "$RUN" cost --out "$R/cost-FS"
stage summary-FS env CUDA_VISIBLE_DEVICES= "$PY" "$RUN" summarize --out "$R/summary-FS"
log "summary-FS label=$(flag summary-FS label) h_star=$(flag summary-FS h_star) W_minus_FS_APS600_pp=$(flag summary-FS W_minus_FS_APS600_pp) W_minus_FS_AP75_small600_pp=$(flag summary-FS W_minus_FS_AP75_small600_pp)"
log "CHAIN DONE"
