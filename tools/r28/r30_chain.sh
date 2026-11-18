#!/bin/bash
# TASK_R30_SEEDS_002: paired seed-1 / seed-2 replication of F_geo against L0.  Four units are trained here, each from
# the e060 parent of its own seed, in the order of task section 3: L0-s1 -> FG-s1 -> L0-s2 -> FG-s2.
# Seed 0 is quoted from task 009 and R27-002: it is read, never trained, evaluated or rescored.
#
# usage: r30_chain.sh <round> <release-name>
#        r30_chain.sh <round> <release-name> --resume-formal <unit> <state.pth> <N>   (N >= 2: writes formal-<unit>-rN)
#
# Order:  r30-check (CPU) -> wire-s0 (seed 0, 8 updates, not a result) -> accept-s1 -> accept-s2 (20 updates each)
#         -> formal-L0-s1 -> formal-FG-s1 -> formal-L0-s2 -> formal-FG-s2
#         -> eval-L0-s1 -> eval-FG-s1 -> eval-L0-s2 -> eval-FG-s2 -> summary-R30 (CPU)
# The round must carry the R30-002 contract (r30_run.py prepare).  Nothing of the earlier rounds is rerun or written:
# no demand training, no W / K / F / P / C unit, no intermediate endpoint, no timing stage.
#
# wire-s0 / accept-sN: a seed-0 path that differs from the stored receipts, a formal input path that differs from the
# serial host reference, an inexact save / restore replay or a window that differs from the S15 ledger of the seed
# closes nothing and ends the chain; no formal run starts.  Window delivery (r29_input): the formal runs hand each
# audited window to the unchanged engine as its device copy, which the acceptance of that seed must have found
# bit-for-bit equal to the host delivery for that unit; R30_DELIVERY=host keeps the host delivery (the acceptance must
# then have been run with the same setting).
#
# GPU: the UUID must equal the contract's gpu_uuid.  The compute processes on the GPU are logged at the start and
# before every stage; nothing is pre-empted, killed or paused, and a shared GPU does not end the chain (the launcher
# decides whether to start; every formal stage records the compute processes at its start and end).
#
# Dev flags: every R28_DEV_* variable is refused, except in a dev round (dev-round* inside the round directory), where
# R28_DEV_SPAN is required and shortens the formal spans; a dev round evaluates two batches and scores nothing.
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
BASIS_WANT=post_R29_user_forwarded_fgeo_l0_seed_replication
DEMAND=c9ffc7505e47adbcc45d70949c3d87371c6f3a7795f1aaaab886d98b15ee9e5a
UNITS="L0-s1 FG-s1 L0-s2 FG-s2"

usage() { echo "usage: r30_chain.sh <round> <release-name> [--resume-formal <unit> <state.pth> <N>] [--until <stage>]" >&2; exit 2; }
UNTIL=""
if [ $# -ge 4 ] && [ "${@: -2:1}" = "--until" ]; then
  UNTIL="${@: -1}"
  set -- "${@:1:$(($# - 2))}"
  case "$UNTIL" in r30-check|wire-s0|accept-s1|accept-s2|formal-L0-s1|formal-FG-s1|formal-L0-s2|formal-FG-s2|eval-FG-s2) ;;
    *) echo "--until takes r30-check, wire-s0, accept-s1, accept-s2, formal-<unit> or eval-FG-s2" >&2; exit 2 ;; esac
fi
[ $# -eq 2 ] || [ $# -eq 6 ] || usage
R=${1%/}
REL=$2
RESUME_UNIT=""; RESUME_STATE=""; RESUME_N=""
if [ $# -eq 6 ]; then
  [ "$3" = "--resume-formal" ] || usage
  RESUME_UNIT=$4; RESUME_STATE=$5; RESUME_N=$6
  case "$RESUME_UNIT" in L0-s1|FG-s1|L0-s2|FG-s2) ;; *) echo "--resume-formal: the unit is one of $UNITS" >&2; exit 2 ;; esac
  [[ "$RESUME_N" =~ ^([2-9]|[1-9][0-9]+)$ ]] || { echo "--resume-formal: N must be an integer >= 2" >&2; exit 2; }
fi
[ -d "$R" ] || { echo "round directory missing: $R" >&2; exit 2; }
[ -z "${L0_ROUND+x}" ] || { echo "L0_ROUND must not be set" >&2; exit 2; }
for v in R28_ARM R28_PIPELINE R29_DELIVERY R30_UNIT R30_SEED; do
  [ -z "${!v+x}" ] || { echo "$v must not be set in the environment of the chain" >&2; exit 2; }
done
DEVROUND=0
case "$(basename "$R")" in dev-round*) DEVROUND=1 ;; esac
if [ $DEVROUND -eq 1 ]; then
  [ -n "${R28_DEV_SPAN+x}" ] || { echo "a dev round of R30 needs R28_DEV_SPAN" >&2; exit 2; }
  if env | grep '^R28_DEV_' | grep -qvE '^R28_DEV_SPAN='; then
    echo "a dev round of R30 only accepts R28_DEV_SPAN" >&2; exit 2
  fi
elif env | grep -q '^R28_DEV_'; then
  echo "R28_DEV_* variables are refused outside a dev round" >&2; exit 2
fi
if [ -n "${R30_DELIVERY+x}" ] && [ "$R30_DELIVERY" != "host" ]; then
  echo "R30_DELIVERY can only be 'host' (the reference window delivery)" >&2; exit 2
fi
RUN="$R/$REL/source/tools/r28/r30_run.py"
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
    stage "$target" "R30_UNIT=$unit" "$PY" "$RUN" formal --resume "$RESUME_STATE" --out "$R/$target"
    return 0
  fi
  if [ -e "$R/formal-$unit" ]; then
    stop "CHAIN STOP at formal-$unit (unclosed directory present: after a saved epoch state re-enter with --resume-formal $unit <state> <N>; if it failed before the e060 save rename it to formal-$unit-fail-NNN)"
  fi
  stage "formal-$unit" "R30_UNIT=$unit" "$PY" "$RUN" formal --out "$R/formal-$unit"
}

exec 9> "$R/.chain.lock"
flock -n 9 || { log "CHAIN STOP (another chain holds $R/.chain.lock)"; exit 5; }
log "CHAIN START (R30-002 F_geo - L0 seed replication: $UNITS) release=$REL pid=$$ resume=${RESUME_UNIT:-none} ${RESUME_STATE:-} ${RESUME_N:-} until=${UNTIL:-end} delivery_override=${R30_DELIVERY:-none} dev=$(env | grep '^R28_DEV_' | tr '\n' ' ')"
[ -f "$RUN" ] || stop "CHAIN STOP (release runner missing: $RUN)"
[ -f "$R/CONTRACT.json" ] || stop "CHAIN STOP (CONTRACT.json missing: run r30_run.py prepare first)"
BASIS=$("$PY" -c "import json,sys;c=json.load(open(sys.argv[1]));print(c.get('authorization_basis'), ','.join(c['training_order']), c['width_selection']['fwhm_hex'], c['demand']['weights']['sha256'], c['demand']['retrained'])" "$R/CONTRACT.json") || stop "CHAIN STOP (not an R30-002 contract)"
log "contract: authorization_basis / training order / width / demand network / demand retrained = $BASIS"
[ "$BASIS" = "$BASIS_WANT L0-s1,FG-s1,L0-s2,FG-s2 0x1.3080000000000p+4 $DEMAND False" ] || stop "CHAIN STOP (the contract is not the registered R30-002 round)"
WANT=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['gpu_uuid'])" "$R/CONTRACT.json") || stop "CHAIN STOP (cannot read gpu_uuid from the contract)"
HAVE=$(nvidia-smi --query-gpu=uuid --format=csv,noheader) || stop "CHAIN STOP (nvidia-smi failed)"
log "GPU uuid=$(echo "$HAVE" | tr '\n' ' ')contract=$WANT"
[ "$HAVE" = "$WANT" ] || stop "CHAIN STOP (GPU UUID differs from the contract)"
APPS=$(apps) || stop "CHAIN STOP (GPU compute-process query failed)"
log "GPU compute processes at chain start: ${APPS:-none} (nothing is pre-empted; a shared GPU is recorded, not a stop)"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
stage r30-check - env CUDA_VISIBLE_DEVICES= "$PY" "$RUN" r30-check --out "$R/r30-check"
until_reached r30-check
stage wire-s0 R30_SEED=0 "$PY" "$RUN" r30-wire --out "$R/wire-s0"
log "wire-s0 FG_equal_to_the_accept_FG_receipts=$(flag wire-s0 FG_equal_to_the_accept_FG_receipts) L0_equal_to_the_R27_002_receipts=$(flag wire-s0 L0_equal_to_the_R27_002_receipts) formal_path_equal_to_the_reference_path=$(flag wire-s0 formal_path_equal_to_the_reference_path)"
until_reached wire-s0
for s in 1 2; do
  stage "accept-s$s" "R30_SEED=$s" "$PY" "$RUN" r30-accept --out "$R/accept-s$s"
  log "accept-s$s restore_replay_equal=$(flag "accept-s$s" restore_replay_equal) window_delivery=$(flag "accept-s$s" window_delivery) same_windows_as_the_S15_ledger=$(flag "accept-s$s" same_windows_as_the_S15_ledger)"
  until_reached "accept-s$s"
done
for u in $UNITS; do
  formal_stage "$u"
  until_reached "formal-$u"
done
for u in $UNITS; do
  stage "eval-$u" "R30_UNIT=$u" "$PY" "$RUN" evaluate --out "$R/eval-$u"
done
until_reached eval-FG-s2
stage summary-R30 - env CUDA_VISIBLE_DEVICES= "$PY" "$RUN" summarize --out "$R/summary-R30"
log "summary-R30 label=$(flag summary-R30 label) result=$(flag summary-R30 result)"
log "CHAIN DONE"
