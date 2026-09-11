# RT-DETRv2 R18 VisDrone Baseline-v2a Frozen Contract

## Status and boundary

This document records T7B's implementation of a new baseline-v2a contract.
It coexists with baseline-v1 and does not alter the v1 validator, config,
authority, engine, launcher, authorization, or training state. The module is
portable and standard-library-only. It does not import torch, construct a
model or dataset, access data, invoke an evaluator, probe a GPU, or train.

The contract is declarative. Its implementation and its local authority
binding are complete, but the five runtime closure requirements remain
required and unimplemented. Consequently `training_implementation_ready=false`
and `training_ready=false`.

## Frozen identity

The contract ID is `rtdetrv2_r18_visdrone_baseline_training_v2a` and the
baseline ID is `rtdetrv2_r18_visdrone_baseline_v2a`. The checked-in config is
`configs/baseline/rtdetrv2_r18_visdrone_baseline_v2a.json`, with raw identity
12751 bytes / SHA-256
`40800725fa0409ac440f463d1df08ce647714aa7fa715a05f7e98878403ae407` and
canonical identity 10198 bytes / SHA-256
`96db5c69286775fb2b2b6a1a6997e58fe9d6019040eeca7328c02c7c2426ab0e`.

The parent baseline-v1 training contract remains bound by raw identity
10907 / `0f9ddb2e6d8ec6d2b21e42f6c419511f5bd70df287457c2c2b6109eaf8a93297`
and canonical identity 9117 /
`a20c71ef90cb4ccc091a717286a1c49be8a1ebffb178156942b77798d3d7f868`.
Its development AP `22.9469` is recorded only as a comparison value and is
not an acceptance threshold.

## Initialization and model

Initialization is `PResNet-18-vd` ImageNet pretrained, using only the local
T7A authority at
`artifacts/pretrained_rtdetrv2_presnet18_imagenet_v1`. Network downloads,
random initialization, and external checkpoints are forbidden. The weight is
44878642 bytes with SHA-256
`911a745b62e173c8f4b9af513c2ea295428cf23f1bbfe9048381500f140fd720`.
The authority manifest is 1608 bytes with SHA-256
`f290281d7c9f1589e2f6da3aca2251e351df48dbad82317ee59e4381ecb01dd0`.
Its certified state inventory is 115 keys, 11209824 tensor elements, and
`0dca09c370c8ae6ee4f82db357c3a3e1c4c2fdb398577e205dd5a318ac1ef6fc`, with
strict load passing and no missing or unexpected keys.

The model is RTDETR with PResNet-18, HybridEncoder, and RTDETRTransformerv2,
10 classes, 20094584 parameters, input size `[640,640]`, 300 queries, three
decoder layers, and NMS disabled.

## Input, data, and schedule

The input is fixed at `640x640`. Training uses micro-batch 16, development
batch 32, gradient accumulation 1, effective batch 16, and no runtime batch
adaptation. The logical roles remain `train_core` and `development`; their
annotations are `train_core_coco.json` and `development_coco.json` with the
v1 allowed-role identity. Confirmatory is sealed and forbidden, and test is
forbidden.

The schedule is 120 epochs, seed 0, EMA enabled, and the v1 augmentation
order and selection semantics. Augmentation stops at epoch 117 and
multiscale is disabled. Development evaluation and checkpointing occur every
epoch, with periodic checkpoints every 10 epochs and a final epoch-120
checkpoint. Development selection maximizes unrounded
`AP@[0.50:0.95,maxDets=500]`, then AP50, AR500, and earlier epoch. The parent
AP value is not reused as a threshold.

## AMP and optimizer

CUDA autocast dtype is `bfloat16`. GradScaler is disabled and absent from the
contract's executable parameters. No fp16 scaler initialization, growth,
backoff, or interval pseudo-parameters are retained. Nonfinite loss,
nonfinite gradients, skipped optimizer steps, and overflow events are all
allowed zero times.

The vendor R18 recipe, PResNet source, and optimizer include are bound by
their exact path, size, and SHA-256 identities. The effective optimizer is
AdamW with default LR `1e-4`, betas `[0.9,0.999]`, and default weight decay
`1e-4`. The sole explicit selector is the case-sensitive literal `norm|bn`
selector, which uses weight decay `0`; every other trainable parameter goes
to the default group. The vendor backbone `1e-5` override is explicitly
forbidden. The public parameter audit API returns a name-keyed, complete,
mutually exclusive partition without importing a model framework.

## Hardware launch gate

Before any later training launch, the host must be proven exclusive and free
of other training load. The required device is NVIDIA GeForce RTX 4090 D with
CUDA 12.4 and a graphics-clock upper limit of 1500 MHz. T7B records and
validates these fields only; it does not execute a hardware probe.

## Runtime closure requirements

The following requirements are recorded as `required=true`,
`implemented=false`, and `independently_certified=false`:

1. Preserve raw process stdout bytes and parse the last valid canonical JSON
   terminal-result line after any vendor log lines.
2. Require the existing `artifacts/training` parent identity while requiring
   the concrete v2a run target to be absent.
3. Bind owner authorization to independent current `train_core` and
   `development` data-root layouts.
4. Bind environment identity to the 1500 MHz clock cap and exclusive-host
   observation.
5. Append, flush, and fsync one canonical JSONL progress-evidence line per
   epoch, and bind completion to its final inventory.

These are not implemented by T7B. No authorization, production target, engine,
launcher, or runtime evidence is created here.

## Public API

The independent module
`src/sparse_rtdetr/baseline/training_v2a_contract.py` exposes:

- `load_training_v2a_contract(repo_root, config_path)` for portable contract
  loading without requiring the ignored T7A artifact;
- `validate_training_v2a_contract(config, parent_config)` for detached closed
  schema, numeric, semantic, relation, and digest validation;
- `canonical_training_v2a_contract_bytes(config)` for deterministic identity;
- `training_v2a_contract_binding(repo_root, config_path)` for local vendor and
  T7A authority binding;
- `audit_optimizer_parameter_groups(parameter_names)` and
  `resolve_optimizer_parameter_group(parameter_name)` for identity-based
  optimizer partition auditing.

The local authority boundary rejects missing, changed, symlinked, hardlinked,
directory, FIFO, and other special-file authority objects. It reads no
authority object until its descriptor identity is checked and compares the
authority directory before and after binding.
