# RT-DETRv2 R18 VisDrone Formal Training Contract V1

This document records owner decision `T1_RANDOM_INITIALIZATION` for
`rtdetrv2_r18_visdrone_baseline_training_v1`. It preserves baseline identity
`rtdetrv2_r18_visdrone_baseline_v1`: PResNet-18 is randomly initialized with
seed 0, `pretrained=false`, and `checkpoint=null`. Network downloads,
pretrained caches, external checkpoints, seed selection, retry, resume, and
overwrite are forbidden. A pretrained route requires a new baseline and
training identity.

The portable authority is
`configs/baseline/rtdetrv2_r18_visdrone_training_v1.json`. It binds the raw
and canonical baseline config, frozen vendor commit and recipe files,
Conversion R3, category mapping, model identity, and allowed data roles. It
contains no runtime data root, host path, username, or AutoDL path.

## Frozen recipe

The run is one non-distributed `cuda:0` process. Train micro-batch and
effective batch are both 16 with no gradient accumulation; train workers are
4. Development batch/workers are 32/4. Train drops its incomplete last batch;
development does not. Runtime batch adaptation is forbidden: an unsuccessful
resource probe requires a new contract revision.

Training lasts 120 epochs. Development evaluation and `last` checkpoint occur
every epoch, periodic checkpoints every 10 epochs, and a final checkpoint is
required. The upstream R18 augmentation order is frozen. Photometric distort,
zoom-out, and IoU crop stop at epoch 117. Multiscale collation is disabled.

AdamW uses default LR `1e-4`, backbone non-norm LR `1e-5`, betas
`[0.9,0.999]`, default weight decay `1e-4`, zero norm/BN weight decay, and
max gradient norm `0.1`. Parameter groups must be mutually exclusive and
cover every trainable parameter by identity. Regex match counts are not proof.

Linear warmup lasts 2000 optimizer steps. MultiStepLR steps once per epoch,
has milestone 1000 and gamma 0.1, and therefore has zero decay events during
120 epochs. Milestone 1000 must never be interpreted as an iteration.

AMP and GradScaler are enabled, but init scale, growth factor, backoff factor,
and growth interval remain `IMPLEMENTATION_MUST_FREEZE_EXPLICITLY`. No current
PyTorch default is asserted. The implementation contract must freeze these
before launch. Non-finite loss/gradient, overflow, and skipped optimizer steps
are forbidden. EMA uses decay 0.9999 with a 2000-optimizer-update warmup.
Development evaluation and model selection use EMA; raw and EMA states are
both retained.

## Evaluation and checkpoints

Primary evaluation is `visdrone_official_style_v1`, development-only, every
epoch. Selection maximizes unrounded float64
`AP@[0.50:0.95,maxDets=500]`, then AP50, AR500, and earlier epoch. The vendor
COCO evaluator is diagnostic only and cannot certify or select. The primary
evaluator is not yet independently certified, so training launch remains
blocked. No post-hoc accuracy threshold is defined for this first baseline.

Each checkpoint must atomically bind raw model, EMA, optimizer, scheduler,
warmup, GradScaler, epoch, global optimizer step, RNG states, and config,
source, and environment identities. File and directory fsync, readback,
loadability, SHA-256, and inventory are mandatory. Interrupted runs are
`PERMANENT_FAIL`; exactly-once run identity forbids retry and overwrite.

Runtime integrity, evidence integrity, checkpoint integrity, development
accuracy reporting, model-selection certification, confirmatory access, test
access, and speed measurement are separate states. Training certification
requires all 120 epochs and expected optimizer steps, finite loss/gradients,
no AMP skip/overflow, exit code zero, bound terminal evidence, and loadable
checkpoints. Confirmatory and test remain inaccessible.

## Public API

`sparse_rtdetr.baseline.training_contract` exposes:

- `load_training_contract(repo_root, config_path)`
- `validate_training_contract(config, baseline_config)`
- `canonical_training_contract_bytes(config)`
- `training_contract_binding(repo_root, config_path)`

The module is standard-library-only and has no import-time filesystem access.
It does not import torch, construct a model or dataset, access runtime data, or
create output. This phase does not implement a trainer, evaluator, launcher,
checkpoint writer, or training evidence writer.

Every object role in the portable contract is closed by one recursive static
schema descriptor. Root objects, nested objects, objects inside lists, lists,
and scalar roles are validated before path/SHA semantics, cross-field rules,
and the frozen canonical digest. Missing, extra, and renamed keys fail with
their JSON pointer. Builtin dictionary insertion order is irrelevant, while
list length and order remain part of the frozen contract. Integer and boolean
roles are distinct, floats must be finite, and the four GradScaler fields only
accept `IMPLEMENTATION_MUST_FREEZE_EXPLICITLY` until a later implementation
contract freezes executable values.

All 50 builtin numeric leaves are also covered by a static JSON-pointer
constraint registry before path, cross-field, and digest validation. Each role
declares its exact builtin type, finite requirement, basic legal range, and T1
frozen literal. Errors distinguish non-finite values, values outside the legal
domain, and in-range values that differ from the frozen decision. In
particular, the initialization seed is exactly integer zero, not an arbitrary
non-negative seed. Cross-field checks separately bind effective batch,
single-run policy, epoch completion, AMP/EMA requirements, evaluator launch
blocking, and exactly-once checkpoint ownership. The final canonical digest
remains a separate identity layer rather than a substitute for these semantic
checks.

The four GradScaler placeholders remain non-numeric
`IMPLEMENTATION_MUST_FREEZE_EXPLICITLY` values. Numeric validation does not
make them executable, does not certify the primary evaluator, and does not
make this contract frozen or training-launch ready.
