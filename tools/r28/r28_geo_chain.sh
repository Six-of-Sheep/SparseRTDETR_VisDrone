#!/bin/bash
# TASK_R28_STRONG_F_CONTROL_009: one arm FG ("F_geo": arm F with the fixed kernel width 19.03125, the closest reading
# of the unmatched 006 calibration), added after the W, K and F results and the 006 geometry readings were seen.
#
# usage: r28_geo_chain.sh <round> <release-name>
#        r28_geo_chain.sh <round> <release-name> --resume-formal <state.pth> <N>   (N >= 2: writes formal-FG-rN)
#
# Order:  geo-check (CPU) -> geo-confirm (demand network only) -> wire-FG -> accept-FG (4 + 4 + 2 updates)
#         -> formal-FG -> eval-FG -> cost-FG (L0, K, F, FG, W) -> summary-FG (CPU) -> ledger-means (CPU)
# The round must carry the F_geo contract (prepare-geo).  Nothing of the closed W round, of the K round, of the F round
# or of the 006 round is rerun or written: no demand training, no W / K / F / L0 training, no rescoring, no calibration.
#
# geo-confirm: a geometry that is not valid on the confirmation windows ends the chain after that stage.  The width is
# never changed; no other width, range, gain or parameter is tried.
# accept-FG: the formal run uses the overlapped input path only if the closed acceptance enabled it; R28_PIPELINE=serial
# in the environment keeps the serial reference path (nothing else is accepted in that variable).
#
# Dev flags: every R28_DEV_* variable is refused, except in a dev round (dev-round* inside the round directory), where
# R28_DEV_CONFIRM_WINDOWS and R28_DEV_SPAN shorten the confirmation set and the formal span.
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
ARM=FG
BASIS_WANT=post_006_user_forwarded_fixed_geometry_selected_width_control

usage() { echo "usage: r28_geo_chain.sh <round> <release-name> [--resume-formal <state.pth> <N>] [--until <stage>]" >&2; exit 2; }
UNTIL=""
if [ $# -ge 4 ] && [ "${@: -2:1}" = "--until" ]; then
  UNTIL="${@: -1}"
  set -- "${@:1:$(($# - 2))}"
  case "$UNTIL" in geo-check|geo-confirm|wire-FG|accept-FG|formal-FG) ;;
    *) echo "--until takes geo-check, geo-confirm, wire-FG, accept-FG or formal-FG" >&2; exit 2 ;; esac
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
  [ -n "$UNTIL" ] || { echo "a dev round of the F_geo control needs --until (it never evaluates or scores)" >&2; exit 2; }
  if env | grep '^R28_DEV_' | grep -qvE '^R28_DEV_(CONFIRM_WINDOWS|SPAN)='; then
    echo "a dev round of the F_geo control only accepts R28_DEV_CONFIRM_WINDOWS and R28_DEV_SPAN" >&2; exit 2
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
log "CHAIN START (F_geo fixed-width strong mapping control) release=$REL pid=$$ resume=${RESUME_STATE:-none} ${RESUME_N:-} until=${UNTIL:-end} pipeline_override=${R28_PIPELINE:-none} dev=$(env | grep '^R28_DEV_' | tr '\n' ' ')"
[ -f "$RUN" ] || stop "CHAIN STOP (release runner missing: $RUN)"
[ -f "$R/CONTRACT.json" ] || stop "CHAIN STOP (CONTRACT.json missing: run prepare-geo first)"
BASIS=$("$PY" -c "import json,sys;c=json.load(open(sys.argv[1]));f=c['followup'];print(c.get('authorization_basis'), f['authorization_basis'], f['arm'], f['prior_W_decision']['K_trigger'], f['width_selection']['fwhm_hex'], f['cal_status'], f['cal_h_star'])" "$R/CONTRACT.json") || stop "CHAIN STOP (not an F_geo contract)"
log "contract: authorization_basis / follow-up basis / arm / prior K_trigger / width / 006 status / 006 h_star = $BASIS"
[ "$BASIS" = "$BASIS_WANT $BASIS_WANT FG False 0x1.3080000000000p+4 calibration_unmatched None" ] || stop "CHAIN STOP (the contract is not the registered F_geo strong mapping control)"
WANT=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['gpu_uuid'])" "$R/CONTRACT.json") || stop "CHAIN STOP (cannot read gpu_uuid from the contract)"
HAVE=$(nvidia-smi --query-gpu=uuid --format=csv,noheader) || stop "CHAIN STOP (nvidia-smi failed)"
log "GPU uuid=$(echo "$HAVE" | tr '\n' ' ')contract=$WANT"
[ "$HAVE" = "$WANT" ] || stop "CHAIN STOP (GPU UUID differs from the contract)"
APPS=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader) || stop "CHAIN STOP (GPU compute-process query failed)"
log "GPU compute processes: ${APPS:-none}"
[ -z "$APPS" ] || stop "CHAIN STOP (GPU busy)"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
stage geo-check env CUDA_VISIBLE_DEVICES= R28_CPU_THREADS=16 "$PY" "$RUN" geo-check --out "$R/geo-check"
until_reached geo-check
stage geo-confirm "$PY" "$RUN" geo-confirm --out "$R/geo-confirm"
VALID=$(flag geo-confirm geometry_valid) || stop "CHAIN STOP (cannot read the confirmation status)"
log "geo-confirm geometry_valid=$VALID fwhm_hex=$(flag geo-confirm fwhm_hex) windows=$(flag geo-confirm windows) FG_over_W_minus_1=$(flag geo-confirm FG_over_W_minus_1)"
if [ "$VALID" != "True" ]; then
  log "CHAIN END (the fixed width is not valid on the confirmation windows: returned to the controller; no formal training, nothing else is tried)"
  exit 0
fi
until_reached geo-confirm
stage wire-FG "$PY" "$RUN" geo-wire --out "$R/wire-FG"
until_reached wire-FG
stage accept-FG "$PY" "$RUN" geo-accept --out "$R/accept-FG"
log "accept-FG overlap_enabled=$(flag accept-FG overlap_enabled) (formal input path: $([ "${R28_PIPELINE:-}" = "serial" ] && echo "serial, forced by R28_PIPELINE" || echo "by the acceptance decision"))"
until_reached accept-FG
formal_stage
until_reached formal-FG
stage eval-FG "$PY" "$RUN" evaluate --out "$R/eval-FG"
stage cost-FG "$PY" "$RUN" cost --out "$R/cost-FG"
stage summary-FG env CUDA_VISIBLE_DEVICES= "$PY" "$RUN" summarize --out "$R/summary-FG"
log "summary-FG label=$(flag summary-FG label) fwhm=$(flag summary-FG fwhm) W_minus_FG_APS600_pp=$(flag summary-FG W_minus_FG_APS600_pp) W_minus_FG_AP75_small600_pp=$(flag summary-FG W_minus_FG_AP75_small600_pp)"
stage ledger-means env CUDA_VISIBLE_DEVICES= "$PY" "$RUN" ledger-means --out "$R/ledger-means"
log "ledger-means g_small_mean=$(flag ledger-means g_small_mean)"
log "CHAIN DONE"
