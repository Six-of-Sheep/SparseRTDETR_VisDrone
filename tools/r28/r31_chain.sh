#!/bin/bash
# TASK_R31_EXTERNAL_UAVDT_001: the frozen L0 / F_geo e120 EMA endpoints of seeds 0, 1, 2 on the UAVDT-M test set
# (inference and scoring only; no update of any kind; no gate), then the three S15 endpoints as the reference arm of
# task section 6 (recorded only, not part of the label).
#
# usage: r31_chain.sh <round> <release-name> [--without-reference]
#
# Order:  data (CPU: packages, unpacking, census, ignore rule)
#         -> gtcheck, dhash (CPU, in the background)
#         -> wire-<unit> for every unit (CUDA, 2 forwards each on the first 8 development images: the stored top600
#            records must be reproduced field by field before any UAVDT frame is inferred)
#         -> infer-<unit> (CUDA): L0-s0 FG-s0 L0-s1 FG-s1 L0-s2 FG-s2, then S15-s0 S15-s1 S15-s2;
#            score-<unit> (CPU) starts in the background as soon as its inference closed
#         -> geometry (CPU) -> tables (CPU)
# The round must carry the R31-001 contract (r31_external.py prepare).  Nothing of an earlier round is rerun or
# written: no training, no fine-tuning, no demand training, no other arm, no intermediate endpoint, no timing stage.
# --without-reference leaves the S15 units out (task section 6 struck).
#
# GPU: the UUID must equal the contract's gpu_uuid.  The compute processes on the GPU are logged at the start and
# around every stage; nothing is pre-empted, killed or paused, and a shared GPU does not end the chain (every CUDA
# stage records the other compute processes at its start and end).
#
# Every stage runs as "python -I" (isolated mode) from the release.  Log: <round>/CHAIN.log; every stage writes
# <round>/<stage>.log (an earlier log is kept as <stage>.log.<k>).
#
# Re-entry: a stage directory with a PASS CLOSED.json is skipped; a stage directory without CLOSED.json stops the chain
# (rename it to <stage>-fail-NNN to keep the evidence, fix the cause, re-enter).  Nothing is retried, overwritten or
# deleted automatically.
set -u
PY=/home/lyy/.cache/r0-seed0-e120-20261010-001/environment/bin/python
VERSION_WANT=R31-001-external-uavdt-v1
TASK_SHA=b9216d72d07e0640ec9fcebeb7e3aa8ebf25c4884031593bc5f2c6a686673e74
DEMAND=c9ffc7505e47adbcc45d70949c3d87371c6f3a7795f1aaaab886d98b15ee9e5a
MAIN="L0-s0 FG-s0 L0-s1 FG-s1 L0-s2 FG-s2"
REFERENCE="S15-s0 S15-s1 S15-s2"

usage() { echo "usage: r31_chain.sh <round> <release-name> [--without-reference]" >&2; exit 2; }
[ $# -eq 2 ] || [ $# -eq 3 ] || usage
R=${1%/}
REL=$2
UNITS="$MAIN $REFERENCE"
if [ $# -eq 3 ]; then
  [ "$3" = "--without-reference" ] || usage
  UNITS="$MAIN"
fi
[ -d "$R" ] || { echo "round directory missing: $R" >&2; exit 2; }
DEVROUND=False
case "$(basename "$R")" in dev-round*) DEVROUND=True ;; esac
RUN="$R/$REL/source/tools/r28/r31_external.py"
cd "$R" || exit 2

log() { echo "$(date -Is) $*" >> "$R/CHAIN.log"; }
stop() { log "$1"; wait; exit "${2:-3}"; }
closed_pass() { [ -f "$1/CLOSED.json" ] && grep -q '"status": "PASS"' "$1/CLOSED.json"; }
flag() { "$PY" -I -c "import json,sys;print(json.load(open(sys.argv[1]))[sys.argv[2]])" "$R/$1/CLOSED.json" "$2"; }
apps() { nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader; }

keep_log() {
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
  else
    [ -e "$R/$name" ] && stop "CHAIN STOP at $name (unclosed directory present; preserve it, then rename it to $name-fail-NNN)"
    keep_log "$name"
    log "START $name (GPU compute processes: $(apps | tr '\n' ';'))"
    "$@" > "$R/$name.log" 2>&1 9>&-
    rc=$?
    log "END $name rc=$rc (GPU compute processes: $(apps | tr '\n' ';'))"
    [ $rc -eq 0 ] || stop "CHAIN STOP at $name" $rc
    closed_pass "$R/$name" || stop "CHAIN STOP at $name (rc=0 without a PASS CLOSED.json)"
  fi
  return 0
}

BG_PIDS=""
background() {  # background <directory name> <command...>: the same stage in a subshell; joined by join_background
  ( stage "$@" ) &
  BG_PIDS="$BG_PIDS $!"
}
join_background() {
  local rc=0 pid
  for pid in $BG_PIDS; do wait "$pid" || rc=$?; done
  BG_PIDS=""
  [ $rc -eq 0 ] || stop "CHAIN STOP (a background stage failed: see its STOP line above)" $rc
}

cpu() { env CUDA_VISIBLE_DEVICES= "$PY" -I "$RUN" "$@" --round "$R"; }
gpu() { "$PY" -I "$RUN" "$@" --round "$R"; }

exec 9> "$R/.chain.lock"
flock -n 9 || { log "CHAIN STOP (another chain holds $R/.chain.lock)"; exit 5; }
log "CHAIN START (R31-001 external UAVDT-M: $UNITS) release=$REL pid=$$ dev=$DEVROUND"
[ -f "$RUN" ] || stop "CHAIN STOP (release runner missing: $RUN)"
[ -f "$R/CONTRACT.json" ] || stop "CHAIN STOP (CONTRACT.json missing: run r31_external.py prepare first)"
BASIS=$("$PY" -I -c "import json,sys;c=json.load(open(sys.argv[1]));print(c.get('version'), c['task']['sha256'], c['mapping']['fwhm_hex'], c['demand']['weights']['sha256'], c['demand']['retrained'], c['dev'], c['dev_limit'])" "$R/CONTRACT.json") || stop "CHAIN STOP (not an R31-001 contract)"
log "contract: version / task / width / demand network / demand retrained / dev / dev frames per sequence = $BASIS"
case "$BASIS" in
  "$VERSION_WANT $TASK_SHA 0x1.3080000000000p+4 $DEMAND False $DEVROUND "*) ;;
  *) stop "CHAIN STOP (the contract is not the registered R31-001 round)" ;;
esac
WANT=$("$PY" -I -c "import json,sys;print(json.load(open(sys.argv[1]))['gpu_uuid'])" "$R/CONTRACT.json") || stop "CHAIN STOP (cannot read gpu_uuid from the contract)"
HAVE=$(nvidia-smi --query-gpu=uuid --format=csv,noheader) || stop "CHAIN STOP (nvidia-smi failed)"
log "GPU uuid=$(echo "$HAVE" | tr '\n' ' ')contract=$WANT"
[ "$HAVE" = "$WANT" ] || stop "CHAIN STOP (GPU UUID differs from the contract)"
APPS=$(apps) || stop "CHAIN STOP (GPU compute-process query failed)"
log "GPU compute processes at chain start: ${APPS:-none} (nothing is pre-empted; a shared GPU is recorded, not a stop)"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
stage data cpu data
log "data census_equal_to_selection_evidence=$(flag data census_equal_to_selection_evidence)"
background gtcheck cpu gtcheck
background dhash cpu dhash
for u in $UNITS; do
  stage "wire-$u" gpu wire --unit "$u"
  log "wire-$u equal_to_stored_top600=$(flag "wire-$u" equal_to_stored_top600)"
done
for u in $UNITS; do
  stage "infer-$u" gpu infer --unit "$u"
  background "score-$u" cpu score --unit "$u"
done
stage geometry cpu geometry
join_background
stage tables cpu tables
log "tables label=$(flag tables label) reference_arm=$(flag tables reference_arm)"
log "CHAIN DONE"
