# RT-DETR Baseline Smoke V2 Outer Launch Contract

Smoke V2 uses an explicit configuration and a separate outer evidence identity.
The V1 configuration, output, process evidence, and tmux identity remain
frozen and are never selected by the V2 production launcher.

## Ownership

The production inner CLI exposes the data-free contract check as:

```text
python -m sparse_rtdetr.baseline.smoke_launcher contract-check \
  --repo-root <canonical-repository-root> [--config <registered-config>]
```

Without `--config`, the historical V1 configuration remains the default. An
explicit configuration is resolved and validated against the registry identity;
its `smoke_id` must bind to the registered config and runtime paths. The
`--config` option belongs only to the `contract-check` subparser. It is not
accepted by `_child`, and it is never copied into the canonical child argv.
Contract-check remains data-free and creates no output, process, or outer
evidence directory. Invalid, unregistered, symlinked, or path-drifted configs
fail closed rather than falling back to V1.
`contract-check --config` is a single-occurrence argument. Repeating it creates
an ambiguous command identity and is rejected by the argument parser with exit
code `2`, before contract validation, torch, data, or artifact behavior. No
first-value or last-value selection is permitted, and the parser never falls
back to V1 for a repeated or invalid configuration.
The `contract-check` subparser also disables GNU-style long-option abbreviation:
only the complete `--repo-root` and `--config` tokens are valid. Prefixes,
look-alike options, and `--name=value` forms using an incomplete option name
are rejected with exit code `2` before contract, torch, data, or artifact
behavior. Single-occurrence and no-abbreviation are independent parser gates.

The outer launcher owns `artifacts/outer_launch_evidence/.../launcher`.
The pane wrapper owns `artifacts/outer_launch_evidence/.../pane`.
The inner launcher owns process evidence, and the Smoke entry owns output.
The outer launcher never creates either child directory.

For outer-enabled Smoke V2, V3, and V4, the outer nonce is the sole
cross-layer nonce. The pane wrapper exports it as `P3_PANE_NONCE`, and the
inner launcher must strictly validate and pass that value to the process
launcher. The process launcher must not generate a replacement nonce; process
and entry evidence inherit the pane nonce. V1 is not outer-enabled and keeps
its existing nonce-generation behavior. The nonce is intentionally not added
to the inner argv, so the deterministic argv hash remains unchanged.

## Launch protocol

The outer evidence root is created only after its existing parent has passed the
regular-directory, owner, and mode checks. The initial invocation is persisted
before data, Python, tmux, or CUDA preflight. Every later failure writes a
terminal outer completion record.

The only tmux call is one `new-session` invocation whose command is the short,
single-use pane wrapper path. The session is exactly
`p3_rtdetrv2_r18_visdrone_baseline_smoke_r2` and the client timeout is exactly
10 seconds from the V2 configuration. The wrapper first obtains an atomic
exclusive `pane_consume.lock` containing the plan SHA, wrapper SHA, nonce, and
session. A replay therefore cannot truncate or rewrite any first-run evidence.
The wrapper persists its start marker, receipt, console, exit code, timing, and
completion before or around the inner Python call. Receipt and inventory JSON
are produced with the shared canonical JSON implementation, and all config,
runtime path, PID, nonce, and argv bindings are revalidated before completion.

Outer statuses are limited to `PREFLIGHT_FAILED`, `TMUX_REJECTED`,
`TMUX_ACCEPTED`, and `INTERRUPTED_WITH_EVIDENCE`. A successful tmux client
return never certifies a pane, inner launcher, entry, or scientific result.

Missing or malformed `P3_PANE_NONCE` is rejected before process or output
directories are created. The prepared handoff remains immutable with
`state=PREPARED` and `consumed=false`; the durable exclusive receipt is the
authoritative consumption claim. Validators and the classifier retain strict
nonce equality and are not relaxed for compatibility.

## Entry pre-CUDA binding

The versioned V2, V3, and V4 entry writes `config.json` and then
`invocation.json` before frozen-image, CUDA, dataset, or model access. The
entry config is the canonical JSON serialization of the loaded config
(`ensure_ascii=true`, sorted keys, compact separators, UTF-8). Its byte size
and SHA-256 are the values bound by `invocation.json` and completion evidence.

This entry `config_size_bytes` is intentionally distinct from the process and
handoff `config_size_bytes`, which remains the original source config file
size. The entry SHA is the canonical source-config SHA, while the handoff and
process layers also retain the source file SHA. A failure after invocation has
been persisted must preserve the config and invocation, finalize `FAILED`
entry evidence, and leave CUDA and data execution unstarted.

Process failure status has separate layers: `process_exit_code.txt` records
the child return code, `process_completion.json.child_returncode` records that
same value, and `launcher_exit_code` is the launcher's external contract
result. For a child return code of `1`, the marker is exactly `1\n` and the
launcher result is `2`.

## Process Handoff Durability

`handoff_prepared.json` is an immutable `PREPARED` event. It remains
`single_use=true` and `consumed=false` for its entire lifetime. The child-owned
`handoff_receipt.json` is the authoritative consumption claim; it is schema
version 1 with `consumed=true` and binds the complete prepared relative path,
size, and SHA-256 together with the child PID/PPID, argv, executable identity,
nonce, configuration, and runtime paths.

The receipt has a dedicated filesystem claim writer. It constructs the complete
canonical JSON bytes before opening the final path with `O_WRONLY|O_CREAT|O_EXCL`
and `O_NOFOLLOW` when available, mode `0600`. The writer handles partial writes,
fsyncs the receipt file, closes it, fsyncs the process-evidence directory, then
reopens the receipt read-only and verifies its type, owner, mode, link count,
size, bytes, SHA-256, and JSON binding before entry invocation. It never uses a
temporary receipt file or `os.replace()` and never changes the prepared event.

An `EEXIST` result or any failure after exclusive creation is fail-closed. The
receipt is never deleted, repaired, overwritten, or retried by that consumer, and
entry is not called. This deliberately preserves a possible zero-byte or partial
receipt after a crash between claim and durability. A crash after receipt
durability but before entry, or after entry but before process completion, also
permanently blocks retry; the receipt cannot be used to infer that entry is safe
to repeat. Only the filesystem `O_EXCL` operation arbitrates concurrent
consumers.

## Read-only classification

`classify_outer_evidence` only reads existing evidence and independently
validates outer, pane, process, and entry layers. It can report
`OUTER_PREPARED`, `PREFLIGHT_FAILED`, `TMUX_REJECTED`,
`TMUX_ACCEPTED_PANE_NOT_STARTED`, `PANE_STARTED_NOT_TERMINAL`,
`PANE_FINALIZATION_FAILED`,
`INNER_STARTED_PROCESS_EVIDENCE_ABSENT`, `INNER_PROCESS_EVIDENCE_PRESENT`,
`ENTRY_PRESENT`, `TERMINAL_COMPLETE`, `TERMINAL_FAILED`, or `UNKNOWN`.
Contradictory or incomplete evidence is `UNKNOWN`; the classifier never fills
missing files or infers an execution stage from an exit code alone.

Pane completion is a closed state machine: only `PANE_COMPLETED` and
`PANE_FAILED` are valid. `PANE_COMPLETED` requires a started inner command,
zero integer exit bytes, no error reference, and consistent timing and
inventory evidence. `PANE_FAILED` requires non-success evidence.

Outer signal events have a fixed schema with monitored signal number/name,
UTC timestamp, forwarding boolean, and constrained forwarding result. Events
are timestamp ordered and their count is bound to the strict integer counter.
The outer invocation is also a fixed schema. Its repository, data, config,
output, process, outer, Python, tmux, environment, plan, and wrapper
identities are cross-checked against the V2 configuration and all available
outer/pane evidence before any terminal classification.

The outer client records installed signal handlers for HUP, TERM, INT, and
QUIT, observed events, forwarding results, timeout, TERM/KILL escalation,
return code, and the original exception. A timeout never retries tmux or
claims that a pane or child started. Outer and pane finalization failures keep
the original exception in a separate secondary record and never synthesize a
successful terminal state.

## Pane integrity

The pane plan and wrapper are deterministic products of the frozen V2 config
bytes and canonical SHA, repository/data/output/process/outer paths, the fixed
Smoke ID and tmux session, nonce, creation timestamp, inner argv, child Python
identity, and pane evidence Python identity. Validation rebuilds both products
and compares their canonical bytes; inventory and self-reported hashes are not
trusted expected values.

The V2, V3, and V4 outer paths use the same configuration evidence binding. The
source config's relative path, absolute path, source size, source file SHA-256,
canonical JSON SHA-256, and `smoke_id` are copied into the prepared handoff,
receipt, process invocation, and process completion records and are checked
against the selected runtime spec. The entry's portable `config.json` has its
own actual size and file SHA-256; its SHA-256 is required to equal the source
canonical JSON SHA-256, not the source file SHA-256. The classifier passes the
validated smoke ID, relative path, and canonical SHA to entry validation, so a
valid V2, V3, or V4 entry follows one real-entry contract and any cross-version or
cross-layer drift is terminally rejected.

Smoke V4 is the frozen R4 runtime identity. Its scientific configuration is
field-for-field identical to V3; only the V4 config/smoke identity, R4 output,
process-evidence, outer-evidence, and tmux session paths differ. V4 must use
the same handoff, receipt, entry, process, pane, and outer contracts and must
not reuse any R1, R2, or R3 runtime identity.

Smoke V5 is the frozen R5 contract identity. Its exact values are:

- smoke ID: `rtdetrv2_r18_visdrone_baseline_smoke_v5`
- config: `configs/baseline/rtdetrv2_r18_visdrone_smoke_v5.json`
- output: `artifacts/runs/rtdetrv2_r18_visdrone_baseline_smoke_r5`
- process evidence: `artifacts/process_evidence/rtdetrv2_r18_visdrone_baseline_smoke_r5`
- outer evidence: `artifacts/outer_launch_evidence/rtdetrv2_r18_visdrone_baseline_smoke_r5`
- tmux session: `p3_rtdetrv2_r18_visdrone_baseline_smoke_r5`
- tmux client timeout: `10` seconds

R4 is permanently failed with failure class
`ENTRY_CONFIG_CANONICAL_SIZE_HELPER_NAMEERROR`; retry, resume, repair,
replacement, or reuse of the R4 launch is forbidden. V5 is a new frozen
runtime identity and does not alter that conclusion. V5 scientific fields are
field-for-field identical to V4. The only differences are the six explicit
identity pointers: `smoke_id`, config relative path, output relative path,
process-evidence relative path, outer-evidence relative path, and tmux session
name. V5 therefore changes the successor runtime identity only; it does not
change image selection, data bindings, model fields, seeds, input or batch
sizes, query or postprocessor settings, or any reportability rule.

The V5 source config is `3631` bytes with SHA-256
`18c425805f33726d0bbe1834e295366a4aa39e5ad830f1f1f3cf92b3d1509716`.
The canonical entry serialization is `2922` bytes with SHA-256
`ed39e40602b716fcc7e94bb521dbcf66ce6f57436b9cbc99ed7ed778462f672b`.
The source size and source-file SHA remain the handoff and process bindings;
the canonical size and canonical SHA remain the portable entry bindings.
This distinction is part of the V5 contract and is not a size-helper
substitution.

V5 retains the existing nonce equality and exclusive durable receipt
semantics. The child-owned receipt remains the one-time filesystem claim, and
process child return code, process launcher result, pane/finalizer status, and
outer classification retain their existing independent meanings. No V5
production output, process-evidence, or outer-evidence directory is created by
this contract. The V5 registry and CPU/synthetic checks make the contract
available for independent audit only; they do not authorize filesystem
provisioning, tmux, Smoke, GPU, CUDA, data, model, training, evaluation, or
metrics execution.

## Entry scientific evidence schema

Scientific evidence schema v2 is the failed historical implementation from the
independent audit; no v2 entry is certified. Future successful entries use
schema v3 and are rejected when any v2 scientific file is present. The v3
entry contains the four scientific evidence files
`input_batch_audit.json`, `cuda_runtime_identity.json`, `model_identity.json`,
and `source_identity.json`, plus the v3 `data_binding_audit.json`. The entry validator uses
exact-key objects and rejects missing, extra, boolean-as-integer, non-finite,
device, shape, dtype, range, and SHA-256 drift. Rebuilding
`artifact_inventory.json` and `completion.json` after changing a scientific
field does not make the entry valid.

`input_batch_audit.json` binds the one batch, two frozen stable image IDs,
loader CPU tensors, model tensors, original target sizes, labels, boxes, and
the actual `Resize([640, 640])` plus `ConvertPILImage(dtype="float32",
scale=True)` pipeline. The descriptor is introspected from the production
instances: torchvision 0.19.1 reports `interpolation=bilinear`,
`antialias=true`, and `max_size=null`. Tensor hashes are
`sha256(detached.cpu().contiguous().numpy().tobytes())` and are recorded only
when the tensor is finite. Synthetic tests use CPU tensors and
`cuda_runtime_identity.json` records an explicit CPU fallback; a real entry
must record the initialized single `cuda:0` runtime from PyTorch APIs.

`model_identity.json` uses scientific schema v4 for model state. It binds the
frozen RT-DETRv2 R18/VisDrone contract, 20,094,584 parameters, eval state,
postprocessor settings, output tensor audits, and deterministic parameter/state
schema and value hashes. In synthetic mode the observed state is a strict empty
state: all parameter and buffer counts are zero, both inventories are empty,
and both state hashes are the canonical empty-inventory hash. In real mode the
observed parameter count must equal `contract_parameters`, the baseline contract
count, the smoke config count, and 20,094,584. Parameters and buffers have
separate finite flags and non-finite name lists. All parameters must be finite;
the only allowed non-finite state is the frozen structural `decoder.anchors`
buffer, whose positive-infinity coordinates are bound position-by-position to
`decoder.valid_mask == false`. All state rows bind finite counts, NaN/positive-
infinity/negative-infinity counts, and a SHA-256 of complete contiguous CPU raw
bytes. The validator compares the complete ordered parameter and buffer
inventory with two seed-zero CPU R18 constructions; it does not accept an
inventory merely because its internal sums and hashes are self-consistent.
The production real-model state helper applies the same frozen count, finite-state,
anchor/mask, and aggregate schema/value checks before returning an identity; these
checks are v4 producer preconditions and do not alter the persisted evidence schema.
`source_identity.json` uses
an explicit production-source allowlist, the baseline config and upstream
manifest hashes, the vendor inventory and R18 config/include hashes, and the
frozen Conversion R3 binding. It does not discover the repository recursively
and does not use Git or the network.

The explicit source dependency allowlist covers the AST-resolved local
production closure: the package initializers, baseline artifacts/categories/
config/contract/dataset/postprocessor/smoke modules, smoke launchers, and the
data-protocol categories/converter/evaluation/lineage/parser/protocol/schema/
split modules. The checker rejects a production local import that is missing
from this allowlist. Vendor code remains bound by the upstream canonical
inventory.

R3 binding is mode-specific. Real mode re-reads and verifies the five frozen
R3 metadata files plus `train_core_manifest.json`, including regular-file,
non-symlink, link-count, size, SHA, selected-record, and frozen-contract
checks. Synthetic mode reads selection only from tracked smoke config, records
`real_data_accessed=false`, `runtime_artifacts_accessed=false`, and
`artifact_validation_mode=synthetic_tracked_contract_only`, and does not claim
manifest verification. A clean tracked archive can therefore write and
validate synthetic evidence without ignored R3 runtime files; real mode still
fails closed when those files are absent.

These CPU/synthetic checks establish only that the future entry evidence
writer and validator are internally consistent. They do not certify R5, make
V6 launch-ready, authorize R6, or access real data, CUDA, models, tmux, or
confirmatory metrics.

The child handoff has one canonical argv schema shared by the parent and child:
the bound executable path, `-m`, `sparse_rtdetr.baseline.smoke_launcher`,
`_child`, `--repo-root`, and the canonical repository root. It never contains
`--config`. The executable identity is recorded beside the argv and the
canonical argv SHA. On Linux the child cross-checks bounded, strict UTF-8
`/proc/self/cmdline` bytes against `sys.orig_argv` and the module/subcommand
view in `sys.argv`; it then verifies the executable identity and compares the
complete argv and SHA with the prepared handoff. Missing, extra, reordered,
script-style, forged, or interpreter-drifted argv evidence is rejected before
the receipt is created.

An executable identity has exactly `canonical_path`, `size_bytes`, `sha256`,
`mode`, `regular_file`, and `executable`. The canonical target is a regular,
executable file. A final Conda Python symlink may resolve to that target, while
symlinked parent components and canonical escapes are rejected. The bound
`PANE_EVIDENCE_PYTHON` value is exactly the finalizer identity's canonical path.

Pane start, receipt, completion, and failure evidence bind the positive pane
shell PID (`$$`). `PANE_FAILED` requires an exact `pane_error.json` schema with
`failure_class` equal to `PREINNER_BINDING_FAILURE`, `INNER_NONZERO_EXIT`, or
`PANE_FINALIZATION_FAILURE`; its console reference is checked against the
current console bytes. A completed pane must have no error file.

`pane_timing.json` has exactly `schema_version`, `started_at_utc`,
`finished_at_utc`, `elapsed_seconds`, `inner_started`, `inner_returncode`, and
`signal_name`. Timestamps require an explicit `Z` or `+00:00` UTC offset,
finished time cannot precede start/consume, and elapsed time must equal the UTC
delta within `1e-6` seconds. Booleans, NaN, infinity, negative values, and
unknown fields are rejected.

The shell wrapper captures the Python finalizer's actual process return code
immediately after that process exits and exclusively creates
`pane_finalizer_exit_code.txt`. Its bytes are strict ASCII canonical decimal
process status in the range `0..255`, followed by exactly one LF. The marker is
owned by the wrapper and cannot overwrite pre-existing evidence. The marker is
created after Python finalizer inventory generation, so it is intentionally
excluded from `pane_inventory.json`; this is a post-inventory ownership
exception, not a direct pane-inventory binding. Pane validation independently
reads and validates the marker, requires zero for both `PANE_COMPLETED` and
ordinary `PANE_FAILED`, and requires a nonzero marker for
`PANE_FINALIZATION_FAILED`. A finalization secondary record is accepted only
when its `finalizer_returncode` exactly equals the raw marker.

The inner return code remains independently stored in `pane_exit_code.txt`.
The finalizer return code is never reconstructed from JSON and never replaces
the inner return code. Missing, malformed, inconsistent, symlinked, or
non-regular marker evidence is incomplete/unknown and cannot produce a terminal
classification.

For `PANE_FAILED`, `pane_error.created_at_utc` must be explicit UTC and satisfy
`pane_start.started_at_utc <= pane_error.created_at_utc <=
pane_timing.finished_at_utc`, as well as not preceding the plan creation time.
The validator applies this check using the already validated start and timing
records. Shell writers obtain a fresh UTC timestamp when pre-inner binding or
inner execution failure is established; the Python finalizer likewise obtains
the error timestamp before it records timing completion.

Before the exclusive consume lock, the wrapper checks every runtime evidence
path. A pre-existing receipt, console, start, exit, timing, error, inventory,
completion, secondary, or finalizer-exit file causes status 73 without any
write, truncation, deletion, or inner invocation.

If finalization cannot complete, the original start marker, console, exit bytes,
and inner state remain available and an exclusive
`pane_secondary_finalization_failure.json` is written. Its
`finalizer_executable_identity` records the executable bound by the plan and
its `original_inner_returncode` is never replaced by the finalizer return code.
No `PANE_COMPLETED` record is created; classification reports
`PANE_FINALIZATION_FAILED`.

Outer and process evidence are nonportable and may contain absolute runtime
paths. Entry evidence remains portable and contains no host data root, repo
root, Python path, or credentials.

## V7 dual-gate certification policy

V7 freezes overall certification as the conjunction of two independent
gates: durable terminal evidence must classify as `TERMINAL_COMPLETE`, and
the immediate snapshot protocol must validate as `PASS`. Neither gate can
override the other. `TMUX_ACCEPTED`, directory presence, and snapshot `PASS`
are not durable execution certification.

The outer launcher owns the one-shot observation. Immediately after the only
`tmux new-session` call returns, and before the outer CLI returns, it records
the tmux-finished monotonic/UTC times, calls the session observer exactly
once, and records snapshot-started/snapshot-finished monotonic/UTC times.
There is no sleep, polling, retry, background writer, or second capture. A
successfully observed absent session is valid because the pane may already
have terminated. An observer exception produces durable `FAIL` evidence and
is never retried. Return code 0 means present, 1 means absent, and every other
return code is a lookup failure. The subprocess timeout is 0.25 seconds, the
maximum observer elapsed time is 0.5 seconds, the maximum start delay is 1.0
second, and UTC/monotonic deltas must agree within 0.01 second.

V7 defines these exact launcher-owned post-inventory files:

- `outer_cli_response.jsonl`: canonical JSON response bytes followed by one
  LF, binding status, outer evidence path, completion SHA-256, nonce, and the
  snapshot result filename, size, SHA-256, and status.
- `immediate_snapshot.json`: the unique successful snapshot result.
- `immediate_snapshot_failure.json`: the unique failed snapshot result.

Exactly one snapshot result file must exist. These three names are V7-only
post-inventory ownership exceptions; they are not retroactively excluded
from V2-V6 inventories. Creation order is outer inventory, outer completion,
snapshot result, CLI response, then actual stdout. Completion references
neither snapshot nor response; snapshot references completion; response
references completion and snapshot. This is acyclic. The successful CLI writes
the exact persisted response bytes to stdout and successful stderr is empty.
Snapshot validation and certification require those externally captured stdout
bytes. The local response must match them byte-for-byte, making stdout the
root-of-trust against simultaneous snapshot/local-response repacking.

The snapshot binds its exact schema/policy versions, smoke identity, nonce,
session, canonical runtime paths, raw/canonical config identity, tmux argv and
stream references, one observed session state, pane/process/output presence,
timings, policy limits, capture count, and completion reference. The response
then anchors the exact snapshot bytes and its own single trailing LF. The
validator rejects missing, extra, pre-existing, symlinked, hardlinked,
non-regular, non-canonical, mistimed, non-UTC, non-finite, cross-boundary, or
mutated evidence. Builtin numeric and boolean types are checked separately.

The certification truth table is:

| Durable terminal | Immediate snapshot | Overall certified |
| --- | --- | --- |
| PASS | PASS | true |
| PASS | FAIL | false |
| FAIL | PASS | false |
| FAIL | FAIL | false |
