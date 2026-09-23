# V2B Fork / Continuation Runbook

Status: 2026-09-23. Tools: `tools/v2b_fork_continue.py`, `tools/evaluate_v2b_fork.py`.

## Scope freeze (REV1 execution layer)

The seed2/r896 epoch 53→60 continuation is complete
(`formal-v2b-s2-r896-e053-e060-20260923-003`, epoch-060 SHA `95141834…`).
The REV1 bridge / seal / verify / smoke-launcher chain is frozen: no new
`verify_v2b_smoke_launcher_r*`, `execution_contract_r*`, bridge or rebase
revisions. Its artifacts stay as permanent evidence. New training work uses
the fork tool below.

## Why no rebase is needed

Every schema-2 checkpoint embeds its run binding; every bound source path
points into the worktree that produced the run (e.g. R35 epoch 31–60
checkpoints of seed 1 bind to `SparseRTDETR_VisDrone_v2b_replication_20260914t184646z`,
not to the epoch60 worktree). The fork tool reads that binding, re-executes
itself with `PYTHONPATH=<bound worktree>/src`, and restores with the normal
`V2BTrainingSession.restore`. The binding therefore verifies unchanged. The
only requirement is that the bound worktree stays clean at its commit.

| cell | bound worktree (commit) |
|---|---|
| s0_r640 (30ep, arm B) | `_v2b_engineering_20260913` (858cf2b) |
| s0_r896 (30ep) | `_v2b_896_20260914t110022z` (bda8c42) |
| s1_r640 / s1_r896 / s2_r640 (30ep and R35 31–60) | `_v2b_replication_20260914t184646z` (7475654) |
| s2_r896 (30ep completion and R35 31–53) | `_v2b_s2completion_20260915t225518z` (6f0b891) |
| s2_r896 epoch 54–60 (formal-003) | `_v2b_epoch60_20260916` (217da68 at the time of writing) |

## Learning-rate decay by forking

The bound schedule is constant (`MultiStepLR(milestones=[1000])`). With
`--lr-scale 0.1`, forking at the epoch-E checkpoint reproduces a run whose
LR decays by 0.1 after epoch E exactly, because both runs share every update
through E. `_check_optimizer` excludes `lr` from the saved-state comparison
and the vendor warmup is inert after 2000 updates, so the scaled LR is kept
by the scheduler and stored in every later checkpoint.

## Loader workers

`--num-workers 2` (default) is the topology of all ten 30-epoch cells.
Sampling and augmentation seeds are position-derived, so the worker count
does not change batches: on 2026-09-23 a CPU preview of s1_r896 epoch 49
windows 1–3 with 0 and 2 workers matched the R35 receipts
(`augmented_sha256` and `receipt_sha256`) exactly.

Open item: R12 (2026-09-17) lost a 2-worker DataLoader worker to SIGABRT
(`terminate called without an active exception`) about 11 s after restore
in the GPU worker process. It did not reproduce on CPU. Run a GPU smoke
(`--max-windows 20`) before any long fork; if it recurs, use
`--num-workers 0` and record it.

## Commands

Use the P3 interpreter `PY=/home/lyy/miniconda3/envs/sparse-rtdetrv2-p3-r2/bin/python`
from this worktree. The launcher sets the clean environment itself.

```bash
T=tools/v2b_fork_continue.py
R35=/media/lyy/Data/JupyterLab/LiuZhiyang/SparseRTDETR_VisDrone_v2b_epoch60_campaign_r35_20260918/artifacts/training/v2bepoch60-r35-20260918t200000z-97a78f7
CK=$R35/execution/s1_r896-formal60/checkpoint-epoch-048.pt
POLICY=$R35/contracts/s1_r896-formal60.json   # reviewed policy_bundle (current boot)

# 1. CPU check (no GPU): binding, bound files, clean worktree, next-epoch batch vs history
$PY $T check --checkpoint $CK --lr-scale 0.1 --reference $R35/execution/s1_r896-formal60/receipts

# 2. GPU smoke: 20 windows, no checkpoint
$PY $T train --checkpoint $CK --lr-scale 0.1 --target-epoch 60 --max-windows 20 \
    --policy-from $POLICY --output /media/lyy/Data/JupyterLab/fork-smoke-s1r896-e048-$(date +%Y%m%d-%H%M)

# 3. Fork run (in tmux)
$PY $T train --checkpoint $CK --lr-scale 0.1 --target-epoch 60 \
    --policy-from $POLICY --output /media/lyy/Data/JupyterLab/fork-s1r896-e048-lrd01-e060

# 4. Evaluate raw and EMA on CPU (same evaluator as the REV1 read-only evaluation)
$PY tools/evaluate_v2b_fork.py --checkpoint <fork>/checkpoint-epoch-060.pt \
    --checkpoint-id seed1-r896-e060-lrd48 --source-campaign fork-s1r896-e048-lrd01 \
    --output-dir /media/lyy/Data/JupyterLab/LiuZhiyang/P3_FORK_EVAL
```

Resuming an interrupted fork: `train --checkpoint <fork>/checkpoint-epoch-NNN.pt --lr-scale 1`
(the stored LR is already scaled).

The policy bundle binds the current boot (`boot_id`). After a reboot a new
reviewed hardware authority is required before `train`; `check` is unaffected.

## Outputs (`--output`, created exclusively)

`fork-start.json` (identity: checkpoint SHA, binding, bound worktree HEAD, tool
SHA, environment, argv), `ready.json` (restored counters, effective LR),
`epoch-NNN-windows.jsonl` (per window: loss, grad norm, LR, input receipt,
input wait, window time), `epoch-NNN-complete.json`, `checkpoint-epoch-NNN.pt`,
`native-hardware/` (monitor evidence), and `fork-result.json` or `fork-failure.json`.
