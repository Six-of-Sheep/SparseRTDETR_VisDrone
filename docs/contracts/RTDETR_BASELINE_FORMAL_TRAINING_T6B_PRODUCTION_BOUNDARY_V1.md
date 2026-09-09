# RT-DETRv2 VisDrone T6B Production Execution Boundary V1

## Scope

T6B implements the production execution boundary only.  The checked-in
configuration is declarative and cannot authorize a run.  A later launch
stage must supply one detached T6A owner authorization and must re-observe the
repository, source, environment, data-role and target-absence bindings at
launch time.

T6A authorization is consumed through the public
`owner_authorization_binding` API in
`training_t6_authorization.py`.  The authorization is external to Git, has a
single raw-byte identity, and receives one adjacent exclusive mode-0600
consumption receipt.  Reuse, replacement, symlink, hardlink, directory, FIFO,
metadata drift and a second receipt are rejected.

Each claimed entry, process and outer root also receives an adjacent immutable
mode-0600 receipt containing a canonical descriptor snapshot.  The process and
outer layers additionally publish adjacent stream receipts after their single
child or tmux call.  Classifiers verify those external snapshots and raw-byte
anchors before accepting any root-local result, so coordinated rewrites of
result, inventory and completion records cannot manufacture a different run.

## Frozen bindings

The four T6B modules are `training_t6_entry.py`,
`training_t6_process_launcher.py`, `training_t6_outer_launcher.py` and
`training_t6_engine.py`.  Their current byte identities are checked by the
repository contract and are re-observed by the entry descriptor.  The
descriptor also binds the T6A contract binding, the detached
owner-authorized launch-time Git branch/HEAD/tree/parent/upstream, and the
live checkout re-observation of those exact values. The historical T5D/
pre-T6 Git object is only a frozen reference and is never a branch allowlist.
The descriptor also binds the config raw and canonical identities, Python and
tmux identity, GPU UUID/driver/CUDA/power limit, filesystem device and mount,
and the detached `train_core` and `development` manifests.

The production call chain has distinct, content-addressed command ownership.
The outer layer publishes `process_descriptor.json` in its outer evidence root
and invokes `python -B -m sparse_rtdetr.baseline.training_t6_process_launcher
--descriptor <outer-root>/process_descriptor.json` through the one tmux request.
The process layer publishes `entry_descriptor.json` in its process evidence
root and invokes `python -B -m sparse_rtdetr.baseline.training_t6_entry
--descriptor <process-root>/entry_descriptor.json` exactly once. The two
descriptor paths, canonical/raw sizes and SHA-256 identities, mode `0600`,
nlink `1`, embedded descriptor digests, Python identity and module names are
recorded in descriptor, result, receipt, invocation and snapshot evidence.
Neither layer may construct its child command from the other layer's argv.

The only permitted data roles are `train_core` and `development`.  The
runtime resolver constructs the role-specific datasets and loaders from those
exact owner-bound roots and manifest identities; it does not use the vendor
YAML default dataset paths. `test` and `confirmatory` are permanently sealed
for this stage. No path resolver in the T6B modules accepts those roles, raw
annotations, or speed measurement.

The training engine accepts no private identity token for production mode.
Instead, the checked entry constructs a closed, content-addressed production
execution context binding the complete descriptor, detached authorization and
consumption receipt, entry evidence bytes, live repository/source/config/data
identities, policy and evidence roots.  The engine independently revalidates
that context and exclusively publishes one durable
`engine_execution_claim.json` under the claimed training evidence root before
resolving or calling any runtime factory.  A copied object, module attribute,
boolean, callback, forged mapping, repacked JSON or replayed claim cannot
authorize production.  The public CPU fake path accepts only non-production
ports, never creates a production claim and never certifies training.  The
actual runtime libraries and vendor/model/data/evaluator ports are resolved
lazily only after the detached authorization, descriptor, evidence and claim
gates.

The final claim pathname is the irreversible execution reservation.  Once
exclusive creation has created that pathname, every later publication or
verification failure, including directory fsync, metadata, readback, close or
final validation failure, retains an object at the exact final pathname and
permanently blocks replay.  No post-creation failure path may unlink, replace,
truncate or rewrite that pathname; a retained invalid or partial object is a
permanent failure rather than reusable authorization.

The claim's duplicated top-level identity is also bound to the fully
revalidated inner execution context.  `descriptor_sha256`, `training_run_id`,
`nonce`, `evidence_root` and `context_sha256` must each equal the corresponding
value from the checked context, with `context_sha256` recomputed from that
checked context.  A canonical rewrite that changes either copy, even when
local derived evidence summaries are refreshed, is rejected by the production
claim validator before any runtime factory is reached.

The training policy is random initialization with seed 0, one GPU and one
world process, batch sizes 16 and 32, and 120 epochs.  AMP GradScaler values,
EMA, AdamW/warmup/scheduler policy, development-only primary evaluator and
`last`/`best`/`periodic`/`final` checkpoint roles are frozen by the T6B JSON
config. The vendor-shaped batch/loss path, 2000 optimizer-update warmup,
epoch scheduler, AMP scaler, and EMA are applied by the runtime engine.
Development evaluation uses EMA weights and converts vendor outputs and
targets into the certified `PrimaryEvaluatorInputV2` before invoking the
primary evaluator. Checkpoint writers must provide root-contained exclusive,
atomic, loadable, append-only references with durable inventory/readback
identities. Non-finite values, AMP overflows, skipped optimizer steps,
checkpoint failures and evaluator or identity drift are permanent failures.

## Boundaries and evidence

The entry publishes its invocation and consumed-authorization evidence before
calling the engine.  The process launcher performs exactly one child call,
constructs argv as a sequence, passes `shell=False`, and preserves child
stdout/stderr as bytes.  The outer launcher performs exactly one tmux
`new-session` request and immediately snapshots its return code, PID, UTC and
monotonic timestamps, and raw stream identities.  There is no polling, sleep,
retry, resume, overwrite or fallback.

`TMUX_ACCEPTED` means only that the outer request was accepted.  It does not
mean that the child is running, that training reached a terminal state, or
that training is scientifically certified.  A terminal success requires
durable process and training evidence and a later independent terminal audit.

All JSON evidence is canonical, closed and append-only.  Each root uses
exclusive creation, durable writes, directory fsync and byte-for-byte
readback.  Inventory, completion and aggregate digests bind the actual files;
raw stdout/stderr anchors are checked independently of JSON repacking.

Descriptor publication is part of the irreversible transaction. After the
owning evidence root is claimed, the descriptor is exclusively created,
fsynced, read back through a stable descriptor read, and identity-revalidated
before the invocation record and the sole child call. A persistence or
validation failure after a root or descriptor pathname is claimed is a
permanent failure; the target is never unlinked, retried, resumed or reused.

## State machine

```text
DESIGN_ONLY
  -> OWNER_AUTHORIZED
  -> PREFLIGHT_PASS
  -> LAUNCH_ACCEPTED
  -> RUNNING
  -> TERMINAL_COMPLETE | PERMANENT_FAIL
  -> INDEPENDENT_TERMINAL_AUDIT_PASS
  -> TRAINING_CERTIFIED
```

After `LAUNCH_ACCEPTED`, resume, retry, overwrite and fallback are forbidden.
Signals, timeouts, nonzero exit, partial writes, host reboot, disk-full,
incomplete evidence, non-loadable checkpoints and any identity drift are
permanent failures.  `TRAINING_CERTIFIED` is never inferred by this
implementation and remains false until the independent audit stage.

## Launch ordering

The later launch authorization transaction is:

```text
fresh immutable gate
-> fresh GPU/environment gate
-> target/session/process absence
-> detached owner authorization binding and one consumption receipt
-> exactly-once outer invocation
-> immediate outer snapshot
-> stop
```

This implementation stage performs none of those real production operations.
Its CPU verification uses only temporary roots and injected fake ports.  It
does not create the owner authorization, production targets or tmux session,
does not access data, and does not start training.
