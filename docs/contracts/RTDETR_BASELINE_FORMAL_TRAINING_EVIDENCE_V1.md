# RT-DETRv2 R18 VisDrone Formal Training Evidence V1

## Contract identity

This document defines the portable evidence and synthetic checkpoint contract for
the formal T1 training implementation. It is a persistence contract only. T5B
does not construct a model, optimizer, scheduler, scaler, EMA object, dataset,
evaluator, or process, and it does not authorize training.

```text
training_evidence_contract_id=rtdetrv2_r18_visdrone_baseline_training_evidence_v1
schema_version=1
raw_size_bytes=7139
raw_sha256=4d6bad4afbede236f1169796aa640f9d82bdcf91d06b799d39c183b20c230c9e
canonical_size_bytes=6069
canonical_sha256=3184707c6cfa115477d007ac0be5eb142a054f78e6309a079ad1e9fab0db9bd6
```

The checked-in JSON is UTF-8, has no BOM, NUL, CR, duplicate key, or non-finite
number, and has exactly one trailing LF. Canonical bytes use ASCII JSON,
sorted keys, compact separators, `allow_nan=false`, and no trailing LF.

## Authority bindings

The evidence contract binds the certified T4 formal training contract and its
primary evaluator, the certified T5A pre-CUDA runtime plan, the observed vendor
runtime inventory, Conversion R3, and the primary evaluator runtime identity.
The following identities are part of the frozen configuration:

```text
training_contract_id=rtdetrv2_r18_visdrone_baseline_training_v1
training_contract_canonical=9117/a20c71ef90cb4ccc091a717286a1c49be8a1ebffb178156942b77798d3d7f868
training_contract_module_sha256=21599fae8312a32ddf833bf2f53557ed5b4d4122f0be167f2071f7f82008deef
runtime_plan_id=rtdetrv2_r18_visdrone_baseline_training_runtime_v1
runtime_plan_raw=12957/cb6af1abae9351b4a268681587db82ad059745f1d7e198d7d8c829b4cee41aef
runtime_plan_canonical=10888/3812b04d957e1cd7c3990c8458a540651c95c6bd551c631fc51717f2f7b386ac
runtime_plan_module_sha256=735d75f612f59e6cec22dc53002980471c34c8b18601cd58704440f85741c7c9
vendor_files=124
vendor_directories_excluding_root=25
vendor_total_size_bytes=373735
vendor_compact_inventory_sha256=0fc6803665bc4b5720e983345b2cacb0147eceead9f882588880b6f8a0e68051
vendor_manifest_size_bytes=33264
vendor_manifest_raw_sha256=f65a2d475365346a5dd5ce4f46b022a135b187e21421e922cc41eee4d28d20ae
vendor_manifest_inventory_sha256=2312c80d5b0fba88d43ffc6807c3fc150ae74b77740f2ab65072f044e033d6d7
conversion_r3_files=25
conversion_r3_total_size_bytes=304418794
conversion_r3_completion_sha256=46734010937168ac65bf27c3a6f4bf3f234554d5d4a99407903dec4b3ae9e817
conversion_r3_artifact_inventory_sha256=aaee01ce4e00749b9db8ac5e6e49cf35875a8b266e9c3efc237bdf0b8b109ae7
conversion_r3_entry_inventory_sha256=ddf32745302d6095e1c3dde89b1a85f05db53e6b4cc8ee85638e0d45bfa8d982
```

The public contract binding API is detached and read-only. Its validation order
is raw JSON type, closed schema, numeric and finite literals, path and SHA syntax,
semantic relations, whole-document digest, and detached T4/T5A authority
bindings. It returns deep copies and does not use environment or working
directory fallbacks.

## Evidence root

The production target names are repository-relative future targets. T5B never
creates them:

```text
artifacts/training/rtdetrv2_r18_visdrone_training_evidence_v1
artifacts/training/rtdetrv2_r18_visdrone_training_evidence_v1/checkpoints
```

A writer test fixture creates an absent evidence root with mode `0700` and
current owner. It creates no parent directories. The root is created
exclusively and contains exactly:

```text
prepared.json
epoch_records.jsonl
checkpoint_references.jsonl
checkpoints/
```

The terminal success pair is `artifact_inventory.json` and `completion.json`.
The permanent failure pair is `partial_inventory.json` and `failure.json`.
These pairs are mutually exclusive. Every regular object is mode `0600`, owned
by the current user and group, has link count one, and is read through an
`O_NOFOLLOW` descriptor-relative boundary. Symlink, hardlink, directory, FIFO,
socket, device, path traversal, replacement, metadata drift, partial read, and
unexpected object cases fail closed.

## Record binding

Every prepared, epoch, checkpoint-reference, completion, and failure record
binds the evidence contract, T4 training contract, T5A runtime plan, mode, run
ID, nonce, training-contract SHA, runtime-plan SHA, source identity SHA, data
identity SHA, environment identity SHA, and primary evaluator ID. Ordered
records carry a predecessor SHA. Epoch and checkpoint-reference streams form a
single hash chain by predecessor links, so independently appended or reordered
records cannot be accepted.

`prepared.json` is immutable and has status `PREPARED`. Epoch records are
strictly increasing in epoch, global optimizer step, and nondecreasing EMA
updates. Loss, gradient norm, and AMP scale are finite builtin floats. AMP
skips, overflow events, and nonfinite counters must remain zero. Evaluator,
source, data, and environment identities cannot drift. No observation is a
model-selection metric.

## Checkpoint schema

Checkpoint files use the exact deterministic names below, with one of the
frozen roles `last`, `best`, `periodic`, or `final`:

```text
checkpoints/checkpoint-{role}-{epoch:04d}.ckpt
```

The required state order is immutable:

```text
raw_model
ema
optimizer
scheduler
warmup
grad_scaler
epoch
global_optimizer_step
rng_states
config_identity
source_identity
environment_identity
```

T5B accepts only caller-supplied detached bytes. Each state records its exact
name, type, format, byte size, SHA-256, loadability affirmation, and synthetic
payload representation. Raw model and EMA are separate states. The aggregate
state inventory SHA, predecessor evidence SHA, source/data/environment
identity, and publication status are bound in the checkpoint envelope.

Publication uses an exclusive same-device temporary regular file, complete
write with EINTR recovery, file fsync, no-overwrite hard-link publication,
temporary cleanup, directory fsync, readback, exact size/SHA verification, and
a caller-supplied immutable loadability probe. Missing parents, existing
targets, loader failure, input mutation, invalid state order, or any inventory
drift fail closed. No real serialization or pickle loading is used.

## Terminal policy

The allowed classifier states are:

```text
ABSENT
IN_PROGRESS
SYNTHETIC_TERMINAL_COMPLETE
TERMINAL_COMPLETE
TERMINAL_FAILED
UNKNOWN
```

Synthetic completion is deliberately distinct from real completion and can
never certify training. A real success requires exactly 120 ordered epochs and
exit code zero. T5B tests create no real terminal success. A permanent failure
is immutable and cannot be converted into success; an invalid, partial,
repacked, or contradictory root is `UNKNOWN`.

The final readiness values remain closed:

```text
runtime_observation_certified=false
training_implementation_ready=false
training_ready=false
model_selection_certified=false
confirmatory_metrics_accessed=false
test_access=false
speed_measurement=false
production_training_authorized=false
```

The implementation is exposed through the standard-library boundary in
`sparse_rtdetr.baseline.training_evidence`:

```text
load_training_evidence_contract
validate_training_evidence_contract
canonical_training_evidence_contract_bytes
training_evidence_contract_binding
TrainingEvidenceWriter
write_atomic_checkpoint
validate_training_evidence
validate_training_checkpoint
classify_training_evidence
TrainingEvidenceError
```
