# RT-DETR Baseline Smoke V2 Outer Launch Contract

Smoke V2 uses an explicit configuration and a separate outer evidence identity.
The V1 configuration, output, process evidence, and tmux identity remain
frozen and are never selected by the V2 production launcher.

## Ownership

The outer launcher owns `artifacts/outer_launch_evidence/.../launcher`.
The pane wrapper owns `artifacts/outer_launch_evidence/.../pane`.
The inner launcher owns process evidence, and the Smoke entry owns output.
The outer launcher never creates either child directory.

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
