# RT-DETRv2 R18 VisDrone Formal Training Runtime Plan V1

## Status and scope

This document defines the project-owned, deterministic runtime plan for the
formal T1 training contract. T1-T4 and the certified primary evaluator are
prerequisites. The checked-in plan is a pre-CUDA declaration and mapper. It is
not a trainer, launcher, filesystem provisioner, evaluator invocation, or
training authorization.

T5A validates only deterministic mapping. T5B, T5C, T5D, T5E, and T5F remain
required before implementation or launch readiness can change. The plan does
not change `training_implementation_ready=false` or `training_ready=false`.

No data, Dataset/DataLoader, production evaluator, model, GPU/CUDA, checkpoint,
training, confirmatory, test, speed, or model-selection operation occurred in
T5A.

## Frozen identity

The plan has schema version `1`, ID
`rtdetrv2_r18_visdrone_baseline_training_runtime_v1`, and stage
`pre_cuda_plan_only`. It binds the raw and canonical T1 training-contract
identity, the training-contract module identity, the certified primary
evaluator, the observed vendor runtime, and Conversion R3. Those identities
are checked independently and are not inferred from the plan digest alone.

The runtime mapper reads the plan through the training contract's verified
directory-file-descriptor boundary. It rejects path escape, symlink, hardlink,
directory, special-file, byte drift, metadata drift, and replacement during
read. It performs no filesystem writes, environment mutation, Git, network,
subprocess, torch, numpy, vendor Python, dataset, or evaluator operation.

## Invocation policy

The formal policy is one prepared run (`formal_run_count=1`) on the later
formal device declaration `cuda:0`, with visible device `0`, world size `1`,
no distribution, no sync batch normalization, no resume, retry, overwrite,
tuning, test-only mode, arbitrary update, network download, external
checkpoint, or pretrained initialization. The plan records random
initialization and seed `0` declaratively; T5A does not set CUDA variables,
construct a device, or launch a child process.

## Model and data roles

The mapping is RTDETR with PResNet-18, HybridEncoder, and
RTDETRTransformerv2; it has 10 classes, 20,094,584 parameters, input size
`[640,640]`, 300 queries, and 3 decoder layers. Logical data roles are
`train_core`, `development`, `sealed_and_forbidden`, and `forbidden` for train,
development, confirmatory, and test respectively. Batch sizes are 16 and 32,
with four workers for each role; train drops the last batch and development
does not. No host-absolute data root is recorded or read.

## Optimizer and schedule

The plan freezes AdamW and three ordered parameter groups:

1. `backbone_non_norm`: backbone parameters excluding literal `norm` and `bn`,
   learning rate `0.00001`, weight decay `0.0001`.
2. `norm_or_bn`: remaining trainable parameters containing literal `norm` or
   `bn`, learning rate `0.0001`, weight decay `0.0`.
3. `default`: every remaining trainable parameter, learning rate `0.0001`,
   weight decay `0.0001`.

The groups are mutually exclusive and must cover every trainable parameter.
The later prepared-trainer stage must observe names, counts, and exact
coverage before CUDA. AdamW betas are `[0.9,0.999]` and gradient clipping is
`0.1`.

The schedule is 120 epochs, LinearWarmup for 2000 optimizer steps, MultiStepLR
per epoch with milestone `[1000]`, gamma `0.1`, and zero expected decay events
within 120 epochs. Checkpoints are declaratively scheduled every epoch,
periodically every 10 epochs, and development evaluation occurs every epoch.
Augmentation stops at epoch 117 and multiscale is disabled.

## Augmentation, AMP, and EMA

The exact transform order is RandomPhotometricDistort, RandomZoomOut,
RandomIoUCrop, SanitizeBoundingBoxes, RandomHorizontalFlip, Resize,
SanitizeBoundingBoxes, ConvertPILImage, and ConvertBoxes. The first three are
the stopped transforms at epoch 117. Conversion uses float32 with scaling and
cxcywh normalized boxes.

AMP is enabled with GradScaler, init scale `65536.0`, growth factor `2.0`,
backoff factor `0.5`, and growth interval `2000`. Nonfinite loss, nonfinite
gradient, skipped optimizer steps, and overflow events are all disallowed.
EMA is enabled with decay `0.9999`, 2000 optimizer-update warmups, EMA weights
for development evaluation and model selection, and preservation of both raw
and EMA weights.

## Evaluation and checkpoint policy

Development-only selection uses the certified
`visdrone_official_primary_evaluator_v1` and protocol
`visdrone_official_style_v1`. The secondary COCO evaluator is diagnostic only
and cannot certify or select. Selection maximizes unrounded
`AP@[0.50:0.95,maxDets=500]`, with ordered tie-breakers `AP50`, `AR500`, and
earlier epoch. Checkpoint roles are last every epoch, best on deterministic
development improvement, periodic every 10 epochs, and final at epoch 120.
Resume, retry, and overwrite remain false; interruption is
`PERMANENT_FAIL`.

Required checkpoint state is exactly raw model, EMA, optimizer, scheduler,
warmup, grad scaler, epoch, global optimizer step, RNG states, config identity,
source identity, and environment identity. Atomic write, file fsync, atomic
rename, directory fsync, readback, loadability, SHA-256, and inventory are
required. T5A does not write or simulate checkpoint bytes.

## Evidence and readiness

Runtime integrity, training evidence, checkpoint integrity, primary evaluator
result binding, source identity, environment identity, data identity, and
exactly-once execution are required later. Completion requires exit code zero
and 120 completed epochs. Confirmatory access, test access, speed measurement,
model selection certification, training implementation readiness, and training
readiness remain false or sealed.

## Vendor conflicts and explicit overrides

The plan records each observed vendor conflict and the formal override:

- CLI `--resume`, `--tuning`, `--update`, and `--test-only` are forbidden.
- The solver's `torch.cuda.is_available()` fallback is replaced later by exact
  `cuda:0` binding.
- Direct vendor output-directory creation is replaced by prepared runtime
  ownership.
- Vendor network/local checkpoint loading is replaced by random initialization
  with external checkpoints forbidden.
- Ordinary `save_on_master` and append-only vendor logs are replaced by the
  atomic identity-bound checkpoint protocol.
- Vendor YAML's CocoEvaluator-only builder is replaced by the certified
  project primary evaluator.
- Vendor `best_stat` is replaced by the frozen unrounded metric and tie-break
  policy.
- Upstream `pretrained: True` is overridden by random initialization.
- Upstream `sync_bn: True` is overridden by `sync_bn=false`.
- Missing vendor GradScaler YAML values are frozen explicitly in the plan.
- Upstream multiscale and stop-epoch-71 defaults are overridden by disabled
  multiscale and stop epoch 117.

These observations are declarative evidence of the mapping decision. T5A does
not import or execute the vendor transform registry or runtime.

## Remaining gates

T5B-T5F must implement and independently verify prepared runtime observation,
AMP/scaler behavior, checkpoint schema and writing, optimizer parameter
identity, EMA, entrypoint/evidence binding, and sealed data enforcement. Until
those stages and their independent audits pass, no formal trainer, launcher,
production evaluator, model selection, speed measurement, confirmatory access,
test access, or training may begin.
