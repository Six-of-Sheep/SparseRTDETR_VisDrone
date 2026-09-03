# RT-DETRv2 R18 VisDrone Prepared Trainer Adapter V1

## Scope and status

This contract defines the CPU-only T5C prepared-trainer boundary for the
formal T1 training contract. It is an injected-port adapter and a synthetic
one-batch observation. It is not a trainer, launcher, filesystem provisioner,
production evaluator, model constructor, Dataset/DataLoader, checkpoint
resume implementation, or training authorization.

T5C must not import torch, initialize CUDA, read a dataset, construct a real
model or optimizer, access test or confirmatory data, create the production
evidence root, spawn a process, use the network, or start tmux. Completion of
this contract does not make implementation or training ready.

## Identity and public API

The adapter has schema version `1`, ID
`rtdetrv2_r18_visdrone_prepared_trainer_adapter_t5c_v1`, mode `synthetic`, and
component protocol `t5c.synthetic_component_port.v1`. Its only public API is:

```text
PreparedTrainerError
prepare_training_adapter(repo_root, *, run_id, nonce, mode="synthetic")
validate_prepared_training_adapter(prepared)
run_synthetic_prepared_batch(prepared, components, evidence_root)
```

The adapter consumes detached results from the T4 training contract, T5A
runtime plan, T5B evidence contract, and certified primary-evaluator binding.
Each binding is validated through its public API, including raw and canonical
identity, source/implementation identity, vendor inventory, Conversion R3,
evaluator certification, and checkpoint/evidence authority. Caller-side
repacking or coordinated mutation is rejected before a descriptor is returned.

## Prepared descriptor schema

The returned builtin dictionary has exactly these top-level keys:

```text
schema_version, adapter_id, mode, run_id, nonce, authorities,
component_roles, state_machine, data_roles, optimizer_observation,
amp_observation, ema_observation, evaluator_integration, selection_policy,
evidence_checkpoint_handoff, execution_policy, aggregate_sha256
```

`authorities` has exactly `identity` and `payloads`. `identity` records the
T4 contract, T5A runtime plan, T5B evidence contract, primary evaluator,
vendor runtime, and Conversion R3 identities. `payloads` retains detached
validated binding objects for `training_contract_binding`,
`runtime_plan_binding`, `evidence_contract`, and
`primary_evaluator_binding`.

`component_roles` declares exactly ten roles:

```text
model, criterion, optimizer, lr_scheduler, amp_scaler, ema,
primary_evaluator, train_batch_source, development_batch_source,
checkpoint_serializer
```

Each port has the exact protocol, method parameter list, one-call expectation,
and identity keys `role`, `protocol`, `implementation`, and `source_sha256`.
Extra public callables, missing methods, optional parameters, async methods,
generators, and identity drift are failures.

`state_machine.states` and `state_machine.transitions` use this exact order:

```text
CREATED
AUTHORITIES_BOUND
COMPONENTS_PREPARED
TRAIN_BATCH_ACQUIRED
FORWARD_COMPLETED
LOSS_VALIDATED
BACKWARD_COMPLETED
OPTIMIZER_STEP_COMPLETED
SCHEDULER_OBSERVED
EMA_UPDATED
DEVELOPMENT_EVALUATION_COMPLETED
CHECKPOINT_SERIALIZED
EVIDENCE_TERMINALIZED
COMPLETED
```

Every transition is exactly once and records the SHA-256 of its predecessor.
There is no retry, resume, fallback, overwrite, polling, second batch,
second optimizer step, second evaluator call, or second checkpoint.

## Runtime observations

Only `train_core` and `development` data roles are allowed. The synthetic
batch schemas are exactly `role`, `batch_id`, `batch_size`, and
`payload_sha256`; batch sizes are 16 and 32. Host-absolute paths and tokens
for test, confirmatory, tuning, speed, benchmark, validation, dataset, or
checkpoint data are rejected.

The optimizer observation is AdamW with three ordered, mutually exclusive
groups. All trainable parameters must occur once, non-trainable parameters
must occur zero times, and declared counts and numel must reconcile. The
groups are `backbone_non_norm` at learning rate `0.00001` and weight decay
`0.0001`, `norm_or_bn` at `0.0001` and `0.0`, and `default` at `0.0001` and
`0.0001`. Numeric values are finite builtin numbers and bool is not an int.

AMP is observed once with `init_scale=65536.0`, `growth_factor=2.0`,
`backoff_factor=0.5`, and `growth_interval=2000`. Nonfinite loss or gradient,
skipped steps, overflow, scale drift, and counter drift fail closed. EMA is
updated once after the optimizer step with decay `0.9999` and 2000
`optimizer_updates` warmups. Both raw and EMA state bytes must be preserved
distinctly.

The evaluator is called exactly once after EMA update, only for the synthetic
`development` role, with evaluator ID
`visdrone_official_primary_evaluator_v1`, protocol
`visdrone_official_style_v1`, and EMA weights. It is bound to run, nonce,
batch, epoch, global optimizer step, EMA identity, and predecessor state.
Selection maximizes unrounded `AP@[0.50:0.95,maxDets=500]`, then uses `AP50`,
`AR500`, and earlier epoch as ordered tie-breakers. Secondary evaluators cannot
certify or select.

## T5B evidence and checkpoint handoff

The caller supplies an absent temporary evidence root. T5C uses the public T5B
writer and validator to emit exactly one epoch record and one atomic `last`
checkpoint, then completes the writer exactly once. The production evidence
root is never selected or created. The classifier must return
`SYNTHETIC_TERMINAL_COMPLETE`.

The checkpoint contains exactly these twelve states, in order:

```text
raw_model, ema, optimizer, scheduler, warmup, grad_scaler, epoch,
global_optimizer_step, rng_states, config_identity, source_identity,
environment_identity
```

The checkpoint and evidence bind T4/T5A IDs and canonical SHA-256 values,
mode, run, nonce, epoch, step, evaluator, and predecessor evidence. Detached,
published-path, and forged-expected authority mutations must all fail. Raw and
EMA payloads must be nonempty and different, and the checkpoint must pass the
public T5B validator and loadability probe.

## Failure semantics and readiness

The adapter is fail-closed. A failed transition produces no success result and
permanently consumes that prepared descriptor, so later reuse or retry fails.
All returned descriptors, observations, identities, and results are detached
builtin values with deterministic canonical aggregate SHA-256 identities.

The execution policy remains `synthetic`, one batch, one epoch record, one
checkpoint, no resume/retry/overwrite, and
`production_training_authorized=false`. T5C implementation and CPU-test
completion do not certify an independent audit, open the production trainer,
authorize test or confirmatory access, certify model selection, measure speed,
or set `training_implementation_ready` or `training_ready` to true.
