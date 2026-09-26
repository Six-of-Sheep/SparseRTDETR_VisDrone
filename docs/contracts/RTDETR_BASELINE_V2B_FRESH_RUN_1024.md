# V2B Fresh Run / A3a 1024² Runbook

Status: 2026-09-25, branch `claude/p3-a3-r1024`.
Tool: `tools/v2b_fresh_run.py` (companion of `tools/v2b_fork_continue.py`).

## Scope

A3a trains one new seed-0 cell at square 1024 for 30 epochs under the same
recipe as the seed-0 896 cell (`v2b896-20260914t134105z-749f804e`, control-896).
It is compared with the seed-0 640 (arm B) and 896 epoch-30 EMA endpoints
evaluated by the REV1 development evaluator (registry record
`A3-seed0-r640-r896-e030-baselines-20260925`). Non-square sizes, other seeds,
test-dev access and data protocol changes are out of scope.

## Code changes

- `V2BConfig`, `TrainCoreDataConfig`, run-binding evidence
  (`training_v2b_evidence.py`) and development evaluation
  (`SUPPORTED_INPUT_SIZES`) accept 1024 in addition to 640/896. Every other
  size stays rejected.
- Admission: `_RESOLUTION_1024_AUTHORIZATION_SHA` is `None`, so no policy can
  name a 1024 authority and no 1024 workload is admitted. Once the owner
  writes the authorization file and its SHA-256 is reviewed into this
  constant, it mirrors 896: `external_admin_acknowledged` only, scopes
  `paired_smoke` 600 s / `train_core_30epoch` 43,200 s, and the bound binding
  must be exactly `input_size=1024, physical_batch_size=8,
  accumulation_steps=2, seed=0`. The evidence authority (R35) still rejects
  1024; the 896 authority still requires 896.
- Historical campaign, control, matched-640, capacity and cross-evaluation
  machinery is unchanged and still accepts only its original sizes.

## Why a separate tool

`v2b_fork_continue.py` only restores schema-2 checkpoints. The existing
epoch-0 entry points (control and replication workers) are bound to the
matched-640, capacity and seed-replication contracts. `v2b_fresh_run.py`
copies the full `V2BConfig`, the pretrained backbone and the train_core data
from a reference checkpoint (`--like`) and replaces only `input_size`
(optionally `seed`). It binds its own clean worktree, including its own
admission, so no orchestration bridge is involved. The binding is built on
CPU in the replication worker's order, and after admission the CUDA-built
initialization must equal the bound CPU initialization.

## CPU verification (2026-09-25, commit 1f26637)

| check | result |
|---|---|
| `check --input-size 896` vs s0_r896 history | initial weights SHA `c5e1477e…` equal; `config` identical (0 differing keys, geometry and sampling included); epoch-1 window-1 `augmented_sha256` `ca76c65e…` equal. Loader binding and receipt SHA differ only through the source SHA-256 of the two edited files (`training_v2b.py`, `training_v2b_data.py`). |
| `check --input-size 1024` | `CHECK_PASS`; only `input_size` changes against `--like`; anchors `[1, 21504, 4]` (128² + 64² + 32²); `pos_embed2` `[1, 1024, 256]`; first batch previewed. |
| `check --input-size 1024 --policy-from <R35 evidence policy>` | rejected: `resolution evidence authorization requires input_size 640 or 896`. |
| unit tests | full suite 252 failed + 295 errors on 693633f (pre-existing: repository contract file list, v2a/legacy fixtures). On this branch the same set, plus the intended update of `test_other_resolutions_and_wrong_types_still_fail` (1024 moved out, 1056/1152 added). `test_prefetch_cursor_recovery_uses_committed_window_only` failed once in the full run under concurrent CPU load and passed 3/3 in isolation. |
| 640 regression (`seed2-r640-e060`, CPU) | raw 22.0468 / 21.3905 / 13.6717, EMA 23.0665 / 22.3266 / 14.4021 (primary / COCO / APS); A1 differs by at most 1e-4. |

## Commands

```bash
PY=/home/lyy/miniconda3/envs/sparse-rtdetrv2-p3-r2/bin/python
T=tools/v2b_fresh_run.py
LIKE=/media/lyy/Data/JupyterLab/LiuZhiyang/SparseRTDETR_VisDrone_v2b_896_20260914t110022z/artifacts/training/v2b896-20260914t134105z-749f804e/execution/control-896/checkpoint-epoch-030.pt

# 1. CPU check (no GPU): config, binding, geometry, first batch
$PY $T check --like $LIKE --input-size 1024 --gpu-uuid <GPU UUID> --num-workers 0 --output <new dir>

# 2. CPU admission check once the 1024 authority is reviewed (monitor built, never started)
$PY $T check --like $LIKE --input-size 1024 --policy-from <1024 policy bundle JSON> --num-workers 0

# 3. GPU smoke: 20 windows, peak memory, no checkpoint
$PY $T train --like $LIKE --input-size 1024 --target-epoch 30 --max-windows 20 \
    --policy-from <1024 policy bundle JSON> --num-workers 2 --output <new dir>

# 4. Formal run (in tmux), one checkpoint per epoch
$PY $T train --like $LIKE --input-size 1024 --target-epoch 30 \
    --policy-from <1024 policy bundle JSON> --num-workers <0|2> --output <new dir>

# 5. Evaluate raw and EMA on CPU
PYTHONPATH=$PWD/src CUDA_VISIBLE_DEVICES= \
$PY tools/evaluate_v2b_fork.py --checkpoint <run>/checkpoint-epoch-030.pt \
    --checkpoint-id seed0-r1024-e030 --source-campaign <run id> --output-dir <dir>
```

A fresh run writes `run-binding.json`, `fresh-start.json`, `ready.json`,
`epoch-NNN-windows.jsonl`, `epoch-NNN-complete.json`,
`checkpoint-epoch-NNN.pt`, `native-hardware/`, and `fresh-result.json` or
`fresh-failure.json`.

Resuming an interrupted 1024 run with `v2b_fork_continue.py` is **not** yet
possible: that tool loads admission from the pinned 97a78f7 checkout, which
has no 1024 authority. If a resume is needed, add a mode that uses the bound
admission (the 1024 run's own worktree) instead of the pin.

## Open items before GPU

1. Owner-written 1024 authorization file; its SHA-256 goes into
   `_RESOLUTION_1024_AUTHORIZATION_SHA` in a reviewed commit.
2. A policy bundle for it: `build_policy_bundle(<current-boot anchor dir>,
   setter_mode="external_admin_acknowledged", authorization_reference=<file
   reference>, external_clock_receipt=<current receipt>)`.
3. Loader workers: 2-worker SIGABRT at loader shutdown is unresolved on the
   fork path, and this tool uses the same epoch loop. Decide 0 or 2 workers
   after a smoke that completes one full epoch.
4. Peak memory at 8 × 1024² is unmeasured (896 fork: 12.74 GB reserved).

## Seed 1 (A3a seed replication, branch `claude/p3-a3-r1024-s1`)

The seed-0 endpoint (EMA 1024−896: COCO AP +0.83, AP_small +0.77) fell in the
"0 to +1" band of the plan's decision rule, which asks for a second seed
before deciding. Seed 1 repeats A3a against the seed-1 896 cell
(`v2bseeds-20260914t211244z-065fe7a4`, `s1_r896-control30`, checkpoint
`747f5815…`); its binding differs from the seed-0 896 cell only in `seed`
(the pretrained file is the same SHA at another path).

- Authority: `_RESOLUTION_1024_SEED1_AUTHORIZATION_SHA` names
  `P3_A3S1_AUTHORIZATION_20260925/user-1024-seed1-authorization.txt`
  (2061 bytes, `6ebf8cf9…`), written by Claude under the owner's in-session
  delegation. It mirrors the seed-0 authority (external admin mode, same
  scopes) but binds `seed=1`; each 1024 authority admits only its own seed.
- The seed-0 run keeps its own worktree at 1bd3bef; seed 1 runs from
  `SparseRTDETR_VisDrone_v2b_a3r1024s1_20260925`.
- Loader: 2 workers. If the run fails mid-way, resume from the last complete
  epoch with `tools/v2b_fork_continue.py train --bound-admission
  --num-workers 0` (the path that resumed seed 0 bit-exactly).

```bash
LIKE1=/media/lyy/Data/JupyterLab/LiuZhiyang/SparseRTDETR_VisDrone_v2b_replication_20260914t184646z/artifacts/training/v2bseeds-20260914t211244z-065fe7a4/execution/s1_r896-control30/checkpoint-epoch-030.pt

# CPU: at 896 the tool must reproduce the s1_r896 history (initial weights, config, first window)
$PY $T check --like $LIKE1 --input-size 896 --gpu-uuid <GPU UUID> --num-workers 0
# CPU: 1024 binding admitted by the seed-1 policy (monitor built, never started)
$PY $T check --like $LIKE1 --input-size 1024 --policy-from <seed-1 policy bundle JSON> --num-workers 0
# GPU smoke, then the formal run
$PY $T train --like $LIKE1 --input-size 1024 --target-epoch 30 --max-windows 20 \
    --policy-from <seed-1 policy bundle JSON> --num-workers 2 --output <new dir>
$PY $T train --like $LIKE1 --input-size 1024 --target-epoch 30 \
    --policy-from <seed-1 policy bundle JSON> --num-workers 2 --output <new dir>
```

## A3b: 16:9 canvas 1344×768 (branch `claude/p3-a3b-r768x1344`)

A3b trains seed 0 on a non-square canvas of the same pixel budget as 1024²
(1,032,192 vs 1,048,576 pixels) against the same seed-0 896 cell as A3a
(`control-896`, checkpoint `104ed411…`).

- Size convention: an integer `input_size` is a square, so every earlier
  binding keeps its bytes. The only non-square size is the explicit list
  `[H, W] = [768, 1344]`; `[1344, 768]` and other lists are rejected by the
  config, loader, evidence, development evaluator and admission. On the
  command line: `--input-size 768x1344` (height first); run ids use
  `r768x1344`.
- Vendor position embedding: `HybridEncoder.build_2d_sincos_position_embedding`
  flattens a `(w, h)` meshgrid, which matches the encoder's row-major `[H, W]`
  token order only when `h == w`. On 24×42 tokens (stride 32) it wraps every
  24 tokens, e.g. token (row 0, column 24) gets the code of (1, 0). For a
  non-square canvas only, `_build_vendor_objects` installs an instance override
  (`_row_major_position_embedding`) that lays the grid out in token order; it is
  bitwise the vendor embedding on squares (24², 28², 32² checked), vendor files
  are unchanged, and `validate_model_geometry` checks the override and its cache.
  Decoder anchors already use `(eval_h, eval_w)` correctly.
- Authority: `_RESOLUTION_768X1344_AUTHORIZATION_SHA` names
  `P3_A3B_AUTHORIZATION_20260926/user-768x1344-authorization.txt`
  (2631 bytes, `1d707bd4…`), written by Claude under the owner's in-session
  delegation; external admin mode, same scopes as 1024, exact
  `input_size=[768, 1344]`, 8 × 2, seed 0. A list `input_size` is admitted by no
  other authority.
- Policy bundle: `P3_A3B_AUTHORIZATION_20260926/policy-bundle-768x1344.json`
  (bundle `1a749476…`, policy `c90713f8…`); hardware fields equal to R35 and to
  the seed-0 1024 bundle except the authorization.

CPU verification (2026-09-26, commit 52ec2e0, worktree
`SparseRTDETR_VisDrone_v2b_a3b_20260926`):

- Tests: the four v2b test files give 456 passed, 1 skipped, 1 failed; the
  failure (`test_engineering_checker_never_reads_runtime_artifact_contents`,
  repository contract) also fails on afb65cb, which gives 429 passed with the
  same skip. No test that passes on afb65cb fails here.
- 896 reproduction: initial weights `c5e1477e…`, config 0 differences, epoch 1
  window 1 `augmented_sha256` `ca76c65e…` equal to the s0_r896 history; the
  loader binding differs only in the source SHAs of `training_v2b.py` and
  `training_v2b_data.py`.
- 768×1344 check: CHECK_PASS, `ADMITTED_NOT_STARTED` (policy `c90713f8…`),
  anchors `[1, 21168, 4]`, `pos_embed2` `[1, 1008, 256]`, only `input_size`
  changed against the reference.
- Rejected as intended: a 768×1344 binding under the seed-0 1024 policy, a 1024
  binding under the 768×1344 policy, and `--input-size 1344x768`.
- Evaluator regression: seed0-r896-e030 EMA re-evaluated from this worktree has
  the same `prediction_sha256` and metrics (COCO AP 22.9430, AP_small 15.4018) as
  `P3_A3_BASELINE_EVAL`; the development binding differs only in `repo_root` and
  the source SHA of `training_v2b_development.py`.
- Pipeline acceptance (not a result): the 896 e030 weights with the two
  size-determined decoder buffers replaced, evaluated on the 768×1344 canvas,
  score 548 images (EMA COCO AP 19.23, AP_small 14.04).

```bash
LIKE0=/media/lyy/Data/JupyterLab/LiuZhiyang/SparseRTDETR_VisDrone_v2b_896_20260914t110022z/artifacts/training/v2b896-20260914t134105z-749f804e/execution/control-896/checkpoint-epoch-030.pt
P=/media/lyy/Data/JupyterLab/LiuZhiyang/P3_A3B_AUTHORIZATION_20260926/policy-bundle-768x1344.json

$PY $T check --like $LIKE0 --input-size 768x1344 --policy-from $P --num-workers 0
$PY $T train --like $LIKE0 --input-size 768x1344 --target-epoch 30 --max-windows 20 \
    --policy-from $P --num-workers 2 --output <new dir>
$PY $T train --like $LIKE0 --input-size 768x1344 --target-epoch 30 \
    --policy-from $P --num-workers 2 --output <new dir>
```
