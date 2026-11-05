#!/usr/bin/env bash
# Task B of TASK_CACHED_FOLLOWUP_002, one batch at lowest priority.
# usage: run_followup_b.sh <worktree root> <new output dir> <commit>
set -euo pipefail
SOURCE=$1
OUT=$2
COMMIT=$3
PY=/home/lyy/.cache/r0-seed0-e120-20261010-001/environment/bin/python
TOOLS=$SOURCE/tools/cached_error_channels
CONFIG=$SOURCE/configs/r25_cached_followup_002.json
EXP=$(dirname "$(dirname "$OUT")")
TIDE=$EXP/r25-cached-error-channels-20261105-001/third_party/tide-49a5d2a
R25=$EXP/r25-query-local-regression-20261104-001/formal-001
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
run() { nice -n 19 ionice -c3 "$PY" -B -I "$@"; }
mkdir "$OUT"
date +%s > "$OUT/START_STAMP"
run "$TOOLS/test_followup.py" > "$OUT/test.log" 2>&1
tail -n 1 "$OUT/test.log"
run "$TOOLS/coord_check.py" --contract "$EXP/sahi-softnms-fixed-20261017-001/revision-007/execution-contract.json"     --contract-sha b2ac33e7a99cfa4e8ea3ccce7297f5088272d839f98fae7b7e13c4e2a7f46c28 --output "$OUT/COORDS_TRANSFORMS.json" --commit "$COMMIT" > "$OUT/coord.log" 2>&1
run "$TOOLS/followup_shift.py" --config "$CONFIG" --out "$OUT" --commit "$COMMIT" --phases coords lists r23 aggregate > "$OUT/run.log" 2>&1
find "$EXP/rtdetrv2-r18-common-protocol-seed0-e120-20261013-001/e088-io-20261014-001/evaluation" \
     "$EXP/r18-sahi-fixed-20261016-001" "$EXP/r18-sahi-seed12-20261016-001" "$EXP/r23-flipview-20261103-001" \
     "$EXP/r25-cached-error-channels-20261105-001/run-001" -newer "$OUT/START_STAMP" -type f > "$OUT/INPUT_DIRS_NEW_FILES.txt" 2>/dev/null || true
sha256sum "$OUT"/*.json "$OUT"/*.npz > "$OUT/OUTPUT_SHA256.txt"
du -sb "$OUT" > "$OUT/OUTPUT_BYTES.txt"
echo COMPLETE
