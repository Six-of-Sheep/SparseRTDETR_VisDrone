#!/bin/bash
# TASK_R28_K_FOLLOWUP_003: independent fixed-prior control K, added after the W closure.
#
# usage: r28_k_followup_chain.sh <round> <release-name>
#        r28_k_followup_chain.sh <round> <release-name> --resume-formal <state.pth> <N>   (N >= 2: writes formal-K-rN)
#
# Order:  followup-check (CPU) -> accept-K -> formal-K -> eval-K -> cost-K (L0, W, K) -> medium (CPU) -> summary-K (CPU)
# The round must carry a follow-up contract (prepare-followup).  Nothing of the closed R28 round is rerun or written:
# no demand training, no G0, no W training, no W cost gate.  The closed W decision (K_trigger=false) is not a trigger
# here and is not changed; R28_DEV_FORCE_K / R28_DEV_IGNORE_* are refused.
# Log: <round>/CHAIN.log; every stage writes <round>/<stage>.log (an earlier log is kept as <stage>.log.<k>).
#
# Start: the GPU UUID must equal the contract's gpu_uuid and the GPU must have no compute process; a busy GPU ends
# the chain, nothing is pre-empted.  Re-entry: a stage directory with a PASS CLOSED.json is skipped; a stage directory
# without CLOSED.json stops the chain (rename it to <stage>-fail-NNN to keep the evidence, fix the cause, re-enter).
# A formal directory that failed after a save is not renamed: re-enter with --resume-formal and an epoch-boundary
# state of that directory.  Nothing is retried, overwritten or deleted automatically.
#
# Dev rounds (directory name dev-round*) may set R28_DEV_SPAN only.
set -u
PY=/home/lyy/.cache/r0-seed0-e120-20261010-001/environment/bin/python
ARM=K

usage() { echo "usage: r28_k_followup_chain.sh <round> <release-name> [--resume-formal <state.pth> <N>]" >&2; exit 2; }
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
for v in R28_DEV_FORCE_K R28_DEV_IGNORE_G0 R28_DEV_IGNORE_COST R28_DEV_CPU R28_DEV_DEMAND_EPOCHS R28_DEV_DEMAND_WINDOWS; do
  [ -z "${!v+x}" ] || { echo "$v is refused by the follow-up chain" >&2; exit 2; }
done
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
    return 0
  fi
  [ -e "$R/$name" ] && stop "CHAIN STOP at $name (unclosed directory present; preserve it, then rename or resume)"
  keep_log "$name"
  log "START $name"
  env R28_ARM="$ARM" "$@" > "$R/$name.log" 2>&1 9>&-
  rc=$?
  log "END $name rc=$rc"
  [ $rc -eq 0 ] || stop "CHAIN STOP at $name" $rc
  closed_pass "$R/$name" || stop "CHAIN STOP at $name (rc=0 without a PASS CLOSED.json)"
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
log "CHAIN START (K follow-up) release=$REL pid=$$ resume=${RESUME_STATE:-none} ${RESUME_N:-}"
case "$(basename "$R")" in dev-round*) log "DEV round: $(env | grep '^R28_DEV_' | tr '\n' ' ')" ;; esac
[ -f "$RUN" ] || stop "CHAIN STOP (release runner missing: $RUN)"
[ -f "$R/CONTRACT.json" ] || stop "CHAIN STOP (CONTRACT.json missing: run prepare-followup first)"
BASIS=$("$PY" -c "import json,sys;c=json.load(open(sys.argv[1]));print(c.get('authorization_basis'), c['followup']['authorization_basis'], c['followup']['prior_W_decision']['K_trigger'])" "$R/CONTRACT.json") || stop "CHAIN STOP (not a follow-up contract)"
log "contract: authorization_basis / follow-up basis / prior K_trigger = $BASIS"
[ "$BASIS" = "post_W_user_approved_K_followup post_W_user_approved_K_followup False" ] || stop "CHAIN STOP (the contract is not the registered K follow-up)"
WANT=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['gpu_uuid'])" "$R/CONTRACT.json") || stop "CHAIN STOP (cannot read gpu_uuid from the contract)"
HAVE=$(nvidia-smi --query-gpu=uuid --format=csv,noheader) || stop "CHAIN STOP (nvidia-smi failed)"
log "GPU uuid=$(echo "$HAVE" | tr '\n' ' ')contract=$WANT"
[ "$HAVE" = "$WANT" ] || stop "CHAIN STOP (GPU UUID differs from the contract)"
APPS=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader) || stop "CHAIN STOP (GPU compute-process query failed)"
log "GPU compute processes: ${APPS:-none}"
[ -z "$APPS" ] || stop "CHAIN STOP (GPU busy)"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
stage followup-check env CUDA_VISIBLE_DEVICES= R28_CPU_THREADS=16 "$PY" "$RUN" followup-check --out "$R/followup-check"
stage accept-K "$PY" "$RUN" accept --out "$R/accept-K"
formal_stage
stage eval-K "$PY" "$RUN" evaluate --out "$R/eval-K"
stage cost-K "$PY" "$RUN" cost --out "$R/cost-K"
stage medium env CUDA_VISIBLE_DEVICES= "$PY" "$RUN" medium --out "$R/medium"
stage summary-K env CUDA_VISIBLE_DEVICES= "$PY" "$RUN" summarize --out "$R/summary-K"
log "summary-K label=$(flag summary-K label) A_difference_pp=$(flag summary-K A_difference_pp)"
log "CHAIN DONE"
