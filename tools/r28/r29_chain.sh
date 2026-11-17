#!/bin/bash
# TASK_R29_001: query-wise un-zoomed decoder read on the fixed strong F_geo input.  Two arms are trained here, each
# from the seed-0 e060 parent: QP ("P", direct read of the warped memory) and then QC ("C", dense un-zoom control).
# N is the certified F_geo endpoint of task 009: it is read, never trained or rescored.
#
# usage: r29_chain.sh <round> <release-name>
#        r29_chain.sh <round> <release-name> --resume-formal <QP|QC> <state.pth> <N>   (N >= 2: writes formal-<arm>-rN)
#
# Order:  r29-check (CPU) -> wire-R29 -> ncheck-FG (2 batches of the stored N endpoint) -> accept-R29 (N 2, C 4 + 2,
#         P 4 + 2 updates) -> formal-QP -> formal-QC -> eval-QP -> eval-QC -> cost-R29 (N, C, P; exclusive GPU)
#         -> summary-R29 (CPU)
# The round must carry the R29 contract (prepare-r29).  Nothing of the W, K, F, 006 or F_geo rounds is rerun or
# written: no demand training, no N training, no rescoring of N, no calibration, no width or point-count scan.
#
# accept-R29: a read that does not act, an inexact save / restore replay or an N path that differs from the stored
# accept-FG receipts closes nothing and ends the chain; no formal run starts.  The formal runs use the overlapped input
# path only if the closed acceptance enabled it for that arm; R28_PIPELINE=serial in the environment keeps the serial
# reference path (nothing else is accepted in that variable).  Window delivery (r29_input): the formal runs hand each
# audited window to the unchanged engine as its device copy only if the acceptance found every update of both arms
# bit-for-bit equal to the stored receipts of release-001 (host delivery); R29_DELIVERY=host keeps the host delivery.
#
# GPU: the UUID must equal the contract's gpu_uuid.  Unless the chain is limited to the CPU check (--until r29-check),
# the GPU must hold no compute process at the start; a busy GPU ends the chain, nothing is pre-empted.  The compute
# processes are logged before every CUDA stage.  cost-R29 needs an exclusive GPU: if another compute process is present
# when the chain reaches it, the chain ends with rc 3 before the stage directory exists and is re-entered later with
# the same command.
#
# Dev flags: every R28_DEV_* variable is refused, except in a dev round (dev-round* inside the round directory), where
# R28_DEV_SPAN shortens the formal spans.  A dev round needs --until (at most formal-QC) and never evaluates or scores.
# Log: <round>/CHAIN.log; every stage writes <round>/<stage>.log (an earlier log is kept as <stage>.log.<k>).
#
# Re-entry: a stage directory with a PASS CLOSED.json is skipped; a stage directory without CLOSED.json stops the chain
# (rename it to <stage>-fail-NNN to keep the evidence, fix the cause, re-enter).  A formal directory that failed after
# a save is not renamed: re-enter with --resume-formal and an epoch-boundary state of that directory.  Nothing is
# retried, overwritten or deleted automatically.
#
# --until <stage> (optional, last two arguments): leave the chain after that stage closed.
set -u
PY=/home/lyy/.cache/r0-seed0-e120-20261010-001/environment/bin/python
BASIS_WANT=post_009_user_forwarded_query_unzoom_read
FG_CONTRACT=0ae1eca4b3e7231bfad70ab6e7aebb93622cec2143975e41bae273869ed85553

usage() { echo "usage: r29_chain.sh <round> <release-name> [--resume-formal <QP|QC> <state.pth> <N>] [--until <stage>]" >&2; exit 2; }
UNTIL=""
if [ $# -ge 4 ] && [ "${@: -2:1}" = "--until" ]; then
  UNTIL="${@: -1}"
  set -- "${@:1:$(($# - 2))}"
  case "$UNTIL" in r29-check|wire-R29|ncheck-FG|accept-R29|formal-QP|formal-QC|eval-QC) ;;
    *) echo "--until takes r29-check, wire-R29, ncheck-FG, accept-R29, formal-QP, formal-QC or eval-QC" >&2; exit 2 ;; esac
fi
[ $# -eq 2 ] || [ $# -eq 6 ] || usage
R=${1%/}
REL=$2
RESUME_ARM=""; RESUME_STATE=""; RESUME_N=""
if [ $# -eq 6 ]; then
  [ "$3" = "--resume-formal" ] || usage
  RESUME_ARM=$4; RESUME_STATE=$5; RESUME_N=$6
  case "$RESUME_ARM" in QP|QC) ;; *) echo "--resume-formal: the arm is QP or QC" >&2; exit 2 ;; esac
  [[ "$RESUME_N" =~ ^([2-9]|[1-9][0-9]+)$ ]] || { echo "--resume-formal: N must be an integer >= 2" >&2; exit 2; }
fi
[ -d "$R" ] || { echo "round directory missing: $R" >&2; exit 2; }
[ -z "${L0_ROUND+x}" ] || { echo "L0_ROUND must not be set" >&2; exit 2; }
DEVROUND=0
case "$(basename "$R")" in dev-round*) DEVROUND=1 ;; esac
if [ $DEVROUND -eq 1 ]; then
  case "$UNTIL" in r29-check|wire-R29|ncheck-FG|accept-R29|formal-QP|formal-QC) ;;
    *) echo "a dev round of R29 needs --until (at most formal-QC): it never evaluates or scores" >&2; exit 2 ;; esac
  if env | grep '^R28_DEV_' | grep -qvE '^R28_DEV_SPAN='; then
    echo "a dev round of R29 only accepts R28_DEV_SPAN" >&2; exit 2
  fi
elif env | grep -q '^R28_DEV_'; then
  echo "R28_DEV_* variables are refused outside a dev round" >&2; exit 2
fi
if [ -n "${R28_PIPELINE+x}" ] && [ "$R28_PIPELINE" != "serial" ]; then
  echo "R28_PIPELINE can only be 'serial' (the reference path)" >&2; exit 2
fi
if [ -n "${R29_DELIVERY+x}" ] && [ "$R29_DELIVERY" != "host" ]; then
  echo "R29_DELIVERY can only be 'host' (the reference window delivery)" >&2; exit 2
fi
RUN="$R/$REL/source/tools/r28/r28_run.py"
export R28_ROUND="$R"
cd "$R" || exit 2

log() { echo "$(date -Is) $*" >> "$R/CHAIN.log"; }
stop() { log "$1"; exit "${2:-3}"; }
closed_pass() { [ -f "$1/CLOSED.json" ] && grep -q '"status": "PASS"' "$1/CLOSED.json"; }
flag() { "$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))[sys.argv[2]])" "$R/$1/CLOSED.json" "$2"; }
apps() { nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader; }

keep_log() {
  local f="$R/$1.log" k=1
  [ -e "$f" ] || return 0
  while [ -e "$f.$k" ]; do k=$((k + 1)); done
  mv "$f" "$f.$k"
}

stage() {  # stage <directory name> <arm> <command...>
  local name=$1 arm=$2 rc
  shift 2
  if [ -e "$R/$name/CLOSED.json" ]; then
    closed_pass "$R/$name" || stop "CHAIN STOP at $name (CLOSED.json is not PASS)"
    log "SKIP $name (closed)"
  else
    [ -e "$R/$name" ] && stop "CHAIN STOP at $name (unclosed directory present; preserve it, then rename or resume)"
    keep_log "$name"
    log "START $name (R28_ARM=$arm; GPU compute processes: $(apps | tr '\n' ';'))"
    env R28_ARM="$arm" "$@" > "$R/$name.log" 2>&1 9>&-
    rc=$?
    log "END $name rc=$rc"
    [ $rc -eq 0 ] || stop "CHAIN STOP at $name" $rc
    closed_pass "$R/$name" || stop "CHAIN STOP at $name (rc=0 without a PASS CLOSED.json)"
  fi
  return 0
}

until_reached() { if [ "$1" = "$UNTIL" ]; then log "CHAIN END (--until $UNTIL reached; later stages not started)"; exit 0; fi; }

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
    stage "$target" "$arm" "$PY" "$RUN" formal --resume "$RESUME_STATE" --out "$R/$target"
    return 0
  fi
  if [ -e "$R/formal-$arm" ]; then
    stop "CHAIN STOP at formal-$arm (unclosed directory present: after a saved epoch state re-enter with --resume-formal $arm <state> <N>; if it failed before the e060 save rename it to formal-$arm-fail-NNN)"
  fi
  stage "formal-$arm" "$arm" "$PY" "$RUN" formal --out "$R/formal-$arm"
}

exec 9> "$R/.chain.lock"
flock -n 9 || { log "CHAIN STOP (another chain holds $R/.chain.lock)"; exit 5; }
log "CHAIN START (R29 query-wise un-zoomed read: P then C) release=$REL pid=$$ resume=${RESUME_ARM:-none} ${RESUME_STATE:-} ${RESUME_N:-} until=${UNTIL:-end} pipeline_override=${R28_PIPELINE:-none} delivery_override=${R29_DELIVERY:-none} dev=$(env | grep '^R28_DEV_' | tr '\n' ' ')"
[ -f "$RUN" ] || stop "CHAIN STOP (release runner missing: $RUN)"
[ -f "$R/CONTRACT.json" ] || stop "CHAIN STOP (CONTRACT.json missing: run prepare-r29 first)"
BASIS=$("$PY" -c "import json,sys;c=json.load(open(sys.argv[1]));f=c['followup'];print(c.get('authorization_basis'), f['authorization_basis'], f['arm'], ','.join(f['arms']), ','.join(f['training_order']), f['width_selection']['fwhm_hex'], f['FG_contract_sha256'])" "$R/CONTRACT.json") || stop "CHAIN STOP (not an R29 contract)"
log "contract: authorization_basis / follow-up basis / arm / arms / training order / width / F_geo contract = $BASIS"
[ "$BASIS" = "$BASIS_WANT $BASIS_WANT None QP,QC QP,QC 0x1.3080000000000p+4 $FG_CONTRACT" ] || stop "CHAIN STOP (the contract is not the registered R29 round)"
WANT=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['gpu_uuid'])" "$R/CONTRACT.json") || stop "CHAIN STOP (cannot read gpu_uuid from the contract)"
HAVE=$(nvidia-smi --query-gpu=uuid --format=csv,noheader) || stop "CHAIN STOP (nvidia-smi failed)"
log "GPU uuid=$(echo "$HAVE" | tr '\n' ' ')contract=$WANT"
[ "$HAVE" = "$WANT" ] || stop "CHAIN STOP (GPU UUID differs from the contract)"
APPS=$(apps) || stop "CHAIN STOP (GPU compute-process query failed)"
log "GPU compute processes: ${APPS:-none}"
if [ "$UNTIL" != "r29-check" ]; then
  [ -z "$APPS" ] || stop "CHAIN STOP (GPU busy)"
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
stage r29-check QP env CUDA_VISIBLE_DEVICES= R28_CPU_THREADS=16 "$PY" "$RUN" r29-check --out "$R/r29-check"
until_reached r29-check
stage wire-R29 QP "$PY" "$RUN" r29-wire --out "$R/wire-R29"
until_reached wire-R29
stage ncheck-FG QP "$PY" "$RUN" r29-ncheck --out "$R/ncheck-FG"
log "ncheck-FG field_by_field_equal=$(flag ncheck-FG field_by_field_equal) images=$(flag ncheck-FG images) scored=$(flag ncheck-FG scored)"
until_reached ncheck-FG
stage accept-R29 QP "$PY" "$RUN" r29-accept --out "$R/accept-R29"
log "accept-R29 N_equal_to_the_FG_receipts=$(flag accept-R29 N_equal_to_the_FG_receipts) restore_replay_equal=$(flag accept-R29 restore_replay_equal) read_active=$(flag accept-R29 read_active) overlap_enabled=$(flag accept-R29 overlap_enabled) window_delivery=$(flag accept-R29 window_delivery) (formal input path: $([ "${R28_PIPELINE:-}" = "serial" ] && echo "serial, forced by R28_PIPELINE" || echo "by the acceptance decision"))"
until_reached accept-R29
formal_stage QP
until_reached formal-QP
formal_stage QC
until_reached formal-QC
stage eval-QP QP "$PY" "$RUN" evaluate --out "$R/eval-QP"
stage eval-QC QC "$PY" "$RUN" evaluate --out "$R/eval-QC"
until_reached eval-QC
if [ ! -e "$R/cost-R29/CLOSED.json" ] && [ ! -e "$R/cost-R29" ]; then
  APPS=$(apps) || stop "CHAIN STOP before cost-R29 (GPU compute-process query failed)"
  [ -z "$APPS" ] || stop "CHAIN STOP before cost-R29 (the timing needs an exclusive GPU and another compute process is present: $(echo "$APPS" | tr '\n' ';') nothing was started; re-enter with the same command when the GPU is free)"
fi
stage cost-R29 QP "$PY" "$RUN" cost --out "$R/cost-R29"
stage summary-R29 QP env CUDA_VISIBLE_DEVICES= "$PY" "$RUN" summarize --out "$R/summary-R29"
log "summary-R29 branch=$(flag summary-R29 branch) label=$(flag summary-R29 label) N_gate_passed=$(flag summary-R29 N_gate_passed) M_gate_passed=$(flag summary-R29 M_gate_passed) P_minus_N_APS600_pp=$(flag summary-R29 P_minus_N_APS600_pp) P_minus_N_AP75_small600_pp=$(flag summary-R29 P_minus_N_AP75_small600_pp) P_minus_C_APS600_pp=$(flag summary-R29 P_minus_C_APS600_pp) C_minus_N_APS600_pp=$(flag summary-R29 C_minus_N_APS600_pp)"
log "CHAIN DONE"
