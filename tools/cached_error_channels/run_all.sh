#!/usr/bin/env bash
# One batch at lowest priority.  usage: run_all.sh <release source root> <new output dir> <commit>
set -euo pipefail
SOURCE=$1
OUT=$2
COMMIT=$3
PY=/home/lyy/.cache/r0-seed0-e120-20261010-001/environment/bin/python
TOOLS=$SOURCE/tools/cached_error_channels
CONFIG=$SOURCE/configs/r25_cached_error_channels.json
TIDE=$(dirname "$(dirname "$OUT")")/r25-cached-error-channels-20261105-001/third_party/tide-49a5d2a
R25=$(dirname "$(dirname "$OUT")")/r25-query-local-regression-20261104-001/formal-001
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
run() { nice -n 19 ionice -c3 "$PY" -B -I "$@"; }
mkdir "$OUT"
sample() {
    # read-only pacing of the running experiment: update line counts and load, never scores or endpoints
    { date +%s; cat /proc/loadavg; wc -l "$R25"/train-A1/UPDATES.jsonl "$R25"/train-A2/UPDATES.jsonl 2>/dev/null || true; } >> "$OUT/R25_PACING.txt"
}
sample
run "$TOOLS/test_core.py" --tide-root "$TIDE" \
    --frozen-src "$(dirname "$(dirname "$OUT")")/sahi-softnms-fixed-20261017-001/revision-007/release/source/src" \
    --zoom-refine "$(dirname "$(dirname "$OUT")")/r20-zoom-refine-20261102-001/execution-release-r20-005/source/tools/zoom_refine/zoom_refine.py" \
    > "$OUT/test.log" 2>&1
tail -n 1 "$OUT/test.log"
run "$TOOLS/run_channels.py" --config "$CONFIG" --out "$OUT" --commit "$COMMIT" --phases admit lists raw r20 > "$OUT/run.log" 2>&1
sample
run "$TOOLS/render_check.py" --config "$CONFIG" --run-dir "$OUT" --commit "$COMMIT" > "$OUT/render.log" 2>&1
run "$TOOLS/tide_native.py" --tide-root "$TIDE" --run-dir "$OUT" --tags A-seed0 C-seed0 A-seed1 C-seed1 A-seed2 C-seed2 > "$OUT/native.log" 2>&1
sample
run "$TOOLS/aggregate.py" --config "$CONFIG" --run-dir "$OUT" --commit "$COMMIT" > "$OUT/aggregate.log" 2>&1
sample
sha256sum "$OUT"/*.json "$OUT"/*.npz > "$OUT/OUTPUT_SHA256.txt"
du -sb "$OUT" > "$OUT/OUTPUT_BYTES.txt"
echo COMPLETE
