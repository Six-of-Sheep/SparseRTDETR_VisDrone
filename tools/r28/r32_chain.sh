#!/bin/bash
# TASK_R32_W_SEEDS_001: seed-1 / seed-2 replication of arm W (the budgeted separable resampling of R28-001).  Two units
# are trained here, each from the e060 parent of its own seed, in the order of task section 4: W-s1 -> W-s2.
# Seed 0 of W is quoted from R28-001, L0 and F_geo from R30-002, S15 from its metric files: read, never retrained or
# rescored as a result.
#
# usage: r32_chain.sh <round> <release-name>
#        r32_chain.sh <round> <release-name> --resume-formal <unit> <state.pth> <N>   (N >= 2: writes formal-<unit>-rN)
#
# Order:  r32-check (CPU) -> wire-s0 (seed 0, 4 updates, not a result) -> endpoint-s0 (seed 0, 141 forwards, the stored
#         prediction lists rescored, not a result) -> accept-s1 -> accept-s2 (10 updates each)
#         -> formal-W-s1 -> formal-W-s2 -> eval-W-s1 -> eval-W-s2 -> summary-R32 (CPU)
# The round must carry the R32-001 contract (r32_run.py prepare).  Nothing of the earlier rounds is rerun or written:
# no demand training, no K / F / P / C unit, no L0 / F_geo / S15 unit, no intermediate endpoint, no timing stage.
#
# wire-s0 / endpoint-s0 / accept-sN: a seed-0 path that differs from the stored R28-001 receipts, a prediction list or
# a scored cell that differs from the stored ones, a formal input path that differs from the serial host reference, an
# inexact save / restore replay or a window that differs from the reference ledger of the seed closes nothing and ends
# the chain; no formal run starts.  Window delivery (r29_input): the formal runs hand each audited window to the
# unchanged engine as its device copy, which the acceptance of that seed must have found bit-for-bit equal to the host
# delivery; R32_DELIVERY=host selects the serial host reference path instead (the acceptance must then have been run with
# the same setting).
#
# GPU: the UUID must equal the contract's gpu_uuid.  The compute processes on the GPU are logged at the start and
# before every stage; nothing is pre-empted, killed or paused, and a shared GPU does not end the chain (the launcher
# decides whether to start; every formal stage records the compute processes at its start and end).
#
# Dev flags: every R28_DEV_* variable is refused, except in a dev round (dev-round* inside the round directory), where
# R28_DEV_SPAN is required and shortens the formal spans; a dev round evaluates two batches of each new unit and scores
# none of them.
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
BASIS_WANT=post_R30_user_forwarded_W_seed_replication
DEMAND=c9ffc7505e47adbcc45d70949c3d87371c6f3a7795f1aaaab886d98b15ee9e5a
TASK_SHA=fb2bb39802a44c30d219ead4be72d562a910aed93bf4aabb3000ac5745a3b157
UNITS="W-s1 W-s2"

usage() { echo "usage: r32_chain.sh <round> <release-name> [--resume-formal <unit> <state.pth> <N>] [--until <stage>]" >&2; exit 2; }
UNTIL=""
if [ $# -ge 4 ] && [ "${@: -2:1}" = "--until" ]; then
  UNTIL="${@: -1}"
  set -- "${@:1:$(($# - 2))}"
  case "$UNTIL" in r32-check|wire-s0|endpoint-s0|accept-s1|accept-s2|formal-W-s1|formal-W-s2|eval-W-s2) ;;
    *) echo "--until takes r32-check, wire-s0, endpoint-s0, accept-s1, accept-s2, formal-<unit> or eval-W-s2" >&2; exit 2 ;; esac
fi
[ $# -eq 2 ] || [ $# -eq 6 ] || usage
R=${1%/}
REL=$2
RESUME_UNIT=""; RESUME_STATE=""; RESUME_N=""
if [ $# -eq 6 ]; then
  [ "$3" = "--resume-formal" ] || usage
  RESUME_UNIT=$4; RESUME_STATE=$5; RESUME_N=$6
  case "$RESUME_UNIT" in W-s1|W-s2) ;; *) echo "--resume-formal: the unit is one of $UNITS" >&2; exit 2 ;; esac
  [[ "$RESUME_N" =~ ^([2-9]|[1-9][0-9]+)$ ]] || { echo "--resume-formal: N must be an integer >= 2" >&2; exit 2; }
fi
[ -d "$R" ] || { echo "round directory missing: $R" >&2; exit 2; }
[ -z "${L0_ROUND+x}" ] || { echo "L0_ROUND must not be set" >&2; exit 2; }
for v in R28_ARM R28_PIPELINE R29_DELIVERY R30_UNIT R30_SEED R30_DELIVERY R32_UNIT R32_SEED; do
  [ -z "${!v+x}" ] || { echo "$v must not be set in the environment of the chain" >&2; exit 2; }
done
DEVROUND=0
case "$(basename "$R")" in dev-round*) DEVROUND=1 ;; esac
if [ $DEVROUND -eq 1 ]; then
  [ -n "${R28_DEV_SPAN+x}" ] || { echo "a dev round of R32 needs R28_DEV_SPAN" >&2; exit 2; }
  if env | grep '^R28_DEV_' | grep -qvE '^R28_DEV_SPAN='; then
    echo "a dev round of R32 only accepts R28_DEV_SPAN" >&2; exit 2
  fi
elif env | grep -q '^R28_DEV_'; then
  echo "R28_DEV_* variables are refused outside a dev round" >&2; exit 2
fi
if [ -n "${R32_DELIVERY+x}" ] && [ "$R32_DELIVERY" != "host" ]; then
  echo "R32_DELIVERY can only be 'host' (the serial host reference path)" >&2; exit 2
fi
RUN="$R/$REL/source/tools/r28/r32_run.py"
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

stage() {  # stage <directory name> <VAR=value or -> <command...>
  local name=$1 assign=$2 rc
  shift 2
  if [ -e "$R/$name/CLOSED.json" ]; then
    closed_pass "$R/$name" || stop "CHAIN STOP at $name (CLOSED.json is not PASS)"
    log "SKIP $name (closed)"
  else
    [ -e "$R/$name" ] && stop "CHAIN STOP at $name (unclosed directory present; preserve it, then rename or resume)"
    keep_log "$name"
    log "START $name ($assign; GPU compute processes: $(apps | tr '\n' ';'))"
    if [ "$assign" = "-" ]; then "$@" > "$R/$name.log" 2>&1 9>&-; else env "$assign" "$@" > "$R/$name.log" 2>&1 9>&-; fi
    rc=$?
    log "END $name rc=$rc (GPU compute processes: $(apps | tr '\n' ';'))"
    [ $rc -eq 0 ] || stop "CHAIN STOP at $name" $rc
    closed_pass "$R/$name" || stop "CHAIN STOP at $name (rc=0 without a PASS CLOSED.json)"
  fi
  return 0
}

until_reached() { if [ "$1" = "$UNTIL" ]; then log "CHAIN END (--until $UNTIL reached; later stages not started)"; exit 0; fi; }

formal_stage() {  # formal_stage <unit>
  local unit=$1 d b closed="" n=0 target re
  re="^formal-$unit(-r([2-9]|[1-9][0-9]+))?$"
  for d in "$R"/formal-"$unit" "$R"/formal-"$unit"-r*; do
    [ -d "$d" ] || continue
    b=$(basename "$d")
    [[ "$b" =~ $re ]] || continue
    if closed_pass "$d"; then closed="$closed $b"; n=$((n + 1)); fi
  done
  [ $n -le 1 ] || stop "CHAIN STOP at formal-$unit (more than one closed formal directory:$closed)"
  if [ $n -eq 1 ]; then log "SKIP formal-$unit (closed in$closed)"; return 0; fi
  if [ -n "$RESUME_STATE" ] && [ "$RESUME_UNIT" = "$unit" ]; then
    target="formal-$unit-r$RESUME_N"
    [ -f "$RESUME_STATE" ] || stop "CHAIN STOP at $target (resume state missing: $RESUME_STATE)"
    log "RESUME formal-$unit from $RESUME_STATE into $target"
    stage "$target" "R32_UNIT=$unit" "$PY" "$RUN" formal --resume "$RESUME_STATE" --out "$R/$target"
    return 0
  fi
  if [ -e "$R/formal-$unit" ]; then
    stop "CHAIN STOP at formal-$unit (unclosed directory present: after a saved epoch state re-enter with --resume-formal $unit <state> <N>; if it failed before the e060 save rename it to formal-$unit-fail-NNN)"
  fi
  stage "formal-$unit" "R32_UNIT=$unit" "$PY" "$RUN" formal --out "$R/formal-$unit"
}

exec 9> "$R/.chain.lock"
flock -n 9 || { log "CHAIN STOP (another chain holds $R/.chain.lock)"; exit 5; }
log "CHAIN START (R32-001 W seed replication: $UNITS) release=$REL pid=$$ resume=${RESUME_UNIT:-none} ${RESUME_STATE:-} ${RESUME_N:-} until=${UNTIL:-end} delivery_override=${R32_DELIVERY:-none} dev=$(env | grep '^R28_DEV_' | tr '\n' ' ')"
[ -f "$RUN" ] || stop "CHAIN STOP (release runner missing: $RUN)"
[ -f "$R/CONTRACT.json" ] || stop "CHAIN STOP (CONTRACT.json missing: run r32_run.py prepare first)"
BASIS=$("$PY" -c "import json,sys;c=json.load(open(sys.argv[1]));print(c.get('authorization_basis'), ','.join(c['training_order']), c['task']['sha256'], c['demand']['weights']['sha256'], c['demand']['retrained'])" "$R/CONTRACT.json") || stop "CHAIN STOP (not an R32-001 contract)"
log "contract: authorization_basis / training order / task file / demand network / demand retrained = $BASIS"
[ "$BASIS" = "$BASIS_WANT W-s1,W-s2 $TASK_SHA $DEMAND False" ] || stop "CHAIN STOP (the contract is not the registered R32-001 round)"
WANT=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['gpu_uuid'])" "$R/CONTRACT.json") || stop "CHAIN STOP (cannot read gpu_uuid from the contract)"
HAVE=$(nvidia-smi --query-gpu=uuid --format=csv,noheader) || stop "CHAIN STOP (nvidia-smi failed)"
log "GPU uuid=$(echo "$HAVE" | tr '\n' ' ')contract=$WANT"
[ "$HAVE" = "$WANT" ] || stop "CHAIN STOP (GPU UUID differs from the contract)"
APPS=$(apps) || stop "CHAIN STOP (GPU compute-process query failed)"
log "GPU compute processes at chain start: ${APPS:-none} (nothing is pre-empted; a shared GPU is recorded, not a stop)"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
stage r32-check - env CUDA_VISIBLE_DEVICES= "$PY" "$RUN" r32-check --out "$R/r32-check"
until_reached r32-check
stage wire-s0 R32_SEED=0 "$PY" "$RUN" r32-wire --out "$R/wire-s0"
log "wire-s0 W_equal_to_the_R28_001_receipts=$(flag wire-s0 W_equal_to_the_R28_001_receipts) formal_path_equal_to_the_reference_path=$(flag wire-s0 formal_path_equal_to_the_reference_path) state_after_update_1_compared=$(flag wire-s0 state_after_update_1_compared)"
until_reached wire-s0
stage endpoint-s0 R32_SEED=0 "$PY" "$RUN" r32-endpoint --out "$R/endpoint-s0"
log "endpoint-s0 predictions_identical=$(flag endpoint-s0 predictions_identical) prediction_records_compared=$(flag endpoint-s0 prediction_records_compared) scoring_reproduced=$(flag endpoint-s0 scoring_reproduced) per_class_APL_max_abs_difference=$(flag endpoint-s0 per_class_APL_max_abs_difference)"
until_reached endpoint-s0
for s in 1 2; do
  stage "accept-s$s" "R32_SEED=$s" "$PY" "$RUN" r32-accept --out "$R/accept-s$s"
  log "accept-s$s restore_replay_equal=$(flag "accept-s$s" restore_replay_equal) window_delivery=$(flag "accept-s$s" window_delivery) same_windows_as_the_reference_ledger=$(flag "accept-s$s" same_windows_as_the_reference_ledger)"
  until_reached "accept-s$s"
done
for u in $UNITS; do
  formal_stage "$u"
  until_reached "formal-$u"
done
for u in $UNITS; do
  stage "eval-$u" "R32_UNIT=$u" "$PY" "$RUN" evaluate --out "$R/eval-$u"
done
until_reached eval-W-s2
stage summary-R32 - env CUDA_VISIBLE_DEVICES= "$PY" "$RUN" summarize --out "$R/summary-R32"
log "summary-R32 reading_A=$(flag summary-R32 reading_A) reading_B=$(flag summary-R32 reading_B) reading_C=$(flag summary-R32 reading_C) result=$(flag summary-R32 result)"
log "CHAIN DONE"
