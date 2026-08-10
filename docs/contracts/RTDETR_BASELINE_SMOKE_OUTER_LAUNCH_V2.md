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
`INNER_STARTED_PROCESS_EVIDENCE_ABSENT`, `INNER_PROCESS_EVIDENCE_PRESENT`,
`ENTRY_PRESENT`, `TERMINAL_COMPLETE`, `TERMINAL_FAILED`, or `UNKNOWN`.
Contradictory or incomplete evidence is `UNKNOWN`; the classifier never fills
missing files or infers an execution stage from an exit code alone.

The outer client records installed signal handlers for HUP, TERM, INT, and
QUIT, observed events, forwarding results, timeout, TERM/KILL escalation,
return code, and the original exception. A timeout never retries tmux or
claims that a pane or child started. Outer and pane finalization failures keep
the original exception in a separate secondary record and never synthesize a
successful terminal state.

Outer and process evidence are nonportable and may contain absolute runtime
paths. Entry evidence remains portable and contains no host data root, repo
root, Python path, or credentials.
