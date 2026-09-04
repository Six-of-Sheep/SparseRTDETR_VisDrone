# RT-DETRv2 R18 VisDrone Formal Training Process Launch V1

## Scope and status

This contract defines the T5D process and entry boundary for the frozen formal
training chain.  The checked-in implementation is intentionally synthetic and
CPU-only.  It accepts injected component ports and a synchronous fake runner so
that process protocol behavior can be tested without starting a real process.

It is not a trainer, model constructor, evaluator invocation, Dataset,
DataLoader, filesystem provisioner for production, terminal multiplexer, or
training authorization.  The production training evidence, process-evidence,
and launch-evidence roots are future targets and must remain absent in T5D.

The readiness values remain:

```text
production_training_authorized=false
real_entry_execution_authorized=false
real_process_launch_authorized=false
training_implementation_ready=false
training_ready=false
model_selection_certified=false
confirmatory_access=false
test_access=false
speed_measurement=false
```

## Contract identity

The process contract has schema version `1` and ID
`rtdetrv2_r18_visdrone_baseline_training_process_v1`.  Its synthetic entry ID
is `rtdetrv2_r18_visdrone_training_entry_t5d_v1`; its launcher ID is
`rtdetrv2_r18_visdrone_training_process_launcher_t5d_v1`.

The process contract closes over the T1 training contract, T5A runtime plan,
T5B evidence/checkpoint contract, T5C prepared adapter, certified primary
evaluator, vendor compact and manifest identities, and Conversion R3 entry
inventory.  It also records every repository Python module imported by the
entry or launcher closure.  Public T1-T5C bindings are revalidated before a
descriptor is returned; detached authority payloads are never trusted merely
because a caller labels them certified.

## Invocation descriptor

The inner entry consumes a detached descriptor with these top-level fields:

```text
schema_version, process_contract_id, entry_id, mode, run_id, nonce,
repo_root, working_directory, python_executable, module_identity, argv,
environment, authorities, future_targets, handoff, source_bindings,
aggregate_sha256
```

`mode` is exactly `synthetic`.  The descriptor binds the canonical repository
root, working directory, executable, module path and SHA-256, an exact argv
list, and a closed environment containing exactly:

```text
CUDA_VISIBLE_DEVICES
PYTHONNOUSERSITE
PYTHONDONTWRITEBYTECODE
PYTHONPATH
```

The argv is a sequence, never a shell command.  Shell metacharacters are data;
no shell evaluator exists in this boundary.  Run IDs and high-entropy lowercase
hexadecimal nonces are bound into the aggregate digest.  Relative paths,
traversal, aliases, symlinks, hardlinks, special objects, missing or extra
keys, reordered argv/environment values, and caller-side mutation fail closed.

The descriptor carries the detached T5C prepared adapter and its complete
authority identity.  It also binds one normalized synthetic evidence root and
an adjacent durable receipt path.  The receipt is created exclusively before
the injected entry can be called; replay after interpreter reload, or replay
with a different root, fails closed.  Future target paths are normalized
repository-relative paths under `artifacts/`; descriptor construction does not
create them.

## Inner entry

`training_entry.py` is import-safe.  It does not import torch, CUDA, vendor
Python, model constructors, Dataset/DataLoader, networking, or a subprocess
module.  Its public API is limited to:

```text
TrainingEntryError
build_synthetic_entry_descriptor
validate_entry_descriptor
run_synthetic_entry
validate_entry_result
```

`build_synthetic_entry_descriptor` performs only public authority binding and
returns detached builtin data.  `run_synthetic_entry` is the only executable
path and invokes the certified T5C synthetic API once with injected ports.  A
real mode, production evidence target, second descriptor use, component
mutation, or T5C result drift is rejected before success can be returned.

The synthetic result binds the invocation digest, prepared-adapter digest,
entry result, bound root and receipt identity, production authorization flag,
and its own canonical aggregate SHA-256.  The entry root is created through a
held parent directory descriptor and revalidated before the result is
returned.  T5B remains responsible for the atomic synthetic checkpoint and
evidence chain created in the entry's dedicated child directory.

## Outer launcher

`training_process_launcher.py` is also import-safe and contains no production
subprocess path.  The launcher descriptor binds the exact entry descriptor,
argv sequence, cwd, closed environment, config identity, source closure,
future targets, and the following state protocol:

The process-config identity includes its relative path, raw and canonical byte
sizes and SHA-256 values, and the observed filesystem mode.  A shared checkout
may expose a tracked regular config blob as either `0644` or `0664`; both modes
are accepted only when the observed mode is recorded and remains identical in
the entry descriptor, launcher descriptor, result, and later source
revalidation.  World-writable, executable, and mode-drifted config objects are
rejected.

```text
ABSENT
PREPARED
ACCEPTED
SYNTHETIC_TERMINAL_COMPLETE
TERMINAL_COMPLETE
TERMINAL_FAILED
UNKNOWN
```

The accepted fake runner has exactly this synchronous signature:

```text
runner(argv, cwd, environment, entry_descriptor)
```

It is called at most once.  Async functions, generators, awaitable results,
extra parameters, defaults, exceptions, timeouts, nonzero return codes,
malformed result objects, mutated invocation inputs, and invalid entry results
are distinct fail-closed terminal outcomes.  The failure class is selected
from a frozen allowlist and return codes are builtin integers in the inclusive
range 0 through 255.  A successful synthetic result requires return code zero,
exact stdout/stderr byte identities, a validated T5D entry result, and all
predecessor bindings.  A real terminal success is never accepted by this
implementation.

Terminal runner failures use the frozen return-code mapping `2` for runner
exception, `3` for timeout, `4` for an async/generator result, `5` for schema
or entry failure, and `6` for invocation-input mutation.  A runner-provided
nonzero return code remains the original builtin integer in the inclusive
range `1` through `255`; persistence failure is fail-closed and is never
published as a complete terminal record.

No retry, resume, overwrite, fallback, polling, sleep, second launch, or
second entry execution is available.  The result state trace stores the SHA-256
of every predecessor state, and the descriptor/result aggregate digests cover
the complete detached values.

## Synthetic launch evidence

For CPU tests only, `run_synthetic_launch` accepts a fresh absolute temporary
root whose parent already exists and must equal the descriptor-bound root.  It
claims an adjacent mode-0600 durable receipt, then creates one mode-0700
directory and publishes exactly:

```text
invocation.json
result.json
completion.json
artifact_inventory.json
stdout.bin
stderr.bin
```

Each JSON file is canonical compact UTF-8 without a trailing newline.  Every
record and byte anchor is opened relative to a held root directory descriptor,
written with exclusive creation, fsynced, published without overwriting an
existing name, and read back through a stable descriptor.  JSON publication
uses a temporary file and an exclusive hard-link into the final name before
the temporary name is removed.  The inventory includes invocation, result,
stdout, and stderr and excludes itself and the completion record to avoid a
circular digest.  Result and completion records repeat and reconcile the
descriptor, root, receipt, state, return-code, and stdout/stderr identities.
Existing roots, terminal roots, symlinks, hardlinks, directories in place of
files, partial records, byte changes, metadata changes, inventory repacking,
and descriptor/result disagreement classify conservatively as `UNKNOWN` or
fail closed.  A valid accepted-only root is `IN_PROGRESS`; a valid synthetic
terminal root is `SYNTHETIC_TERMINAL_COMPLETE`.

The production targets remain absent:

```text
artifacts/training/rtdetrv2_r18_visdrone_training_evidence_v1
artifacts/process_evidence/rtdetrv2_r18_visdrone_training_process_v1
artifacts/outer_launch_evidence/rtdetrv2_r18_visdrone_training_process_v1
```

## Verification and handoff

T5D verification uses only CPU and synthetic injected ports.  It covers strict
config and descriptor schemas, authority/source identity, argv and environment
closure, caller detachment, runner signatures and failure classes, exactly-once
launch and entry calls, durable evidence publication, inventory and terminal
classifier truth tables, import isolation and parent-process sentinel ordering,
cache-free operation, and clean archive behavior.

T5D implementation does not certify an independent T5D audit, does not make
training implementation ready, and does not make training ready.  The next
stage may independently audit the published implementation; no real launch is
permitted before that audit and all later gates pass.
