# RT-DETRv2 R18 VisDrone Baseline-v2a T7E Exactly-Once Launcher

## Scope

T7E is the release boundary between the audited T7D production CLI and a
later, single tmux launch. It adds no training implementation and does not
make a tmux acceptance equivalent to training readiness. Importing the
launcher is inert: it does not import PyTorch, read data, create files, call
Git, call tmux, initialize CUDA, or consume authorization.

The CPU/fake entry is the only entry used by this implementation transaction.
It uses explicit fake launcher and snapshot ports, never consumes a disk
authorization, and never creates a repository production target.

## Four closed records

The owner authorization, launch plan, consumption receipt, and immediate
launch result each use a closed builtin-JSON schema. Every record has a
canonical SHA-256 that excludes only its own digest field. The authorization
binds the T7D descriptor and context, the T7D policy/source/data/environment
identities, the T7A pretrained authority binding, current Git identity,
host/GPU policy, all T7E paths, and the unique session name.

The plan binds two independently persisted parameter layers to the fixed
production argv:

```text
PYTHON -m sparse_rtdetr.baseline.training_v2a_production
  --descriptor DESCRIPTOR_PATH
  --authorization-context AUTHORIZATION_CONTEXT_PATH
```

The descriptor, context, and plan are written with `O_EXCL`, mode `0600`, and
file plus parent-directory fsync. They are read back and checked before the
authorization can be consumed. A consumption receipt is created exactly once;
existing objects, symlinks, hardlinks, metadata drift, and replay are rejected.

## Ordering

The owner path has one preflight, one persistence phase, one authorization
consumption, one `tmux new-session -d -s SESSION -- ARGV` call, and one
immediate `tmux has-session -t SESSION` snapshot. It does not attach, sleep,
poll, retry, resume, overwrite, or use `shell=True`. Post-consumption failure
publishes a permanent T7E result and retains the receipt.

The tmux result is classified as `TMUX_ACCEPTED` only when the request and the
single immediate snapshot both succeed. The result always records
`training_ready=false`, `formal_training_executed=false`, and
`training_certified=false`. `TMUX_ACCEPTED` is therefore an outer launch
observation, not scientific or training evidence.

## Protected history and release

T7E artifact names use the independent `t7e_` namespace. T7B, T7C, T7D,
V1, and T6 bytes and production targets are not modified or reused. Before a
real launch, the current Git/worktree/cache, T7D targets, historical targets,
receipt, lock, and session absence are all asserted. This transaction only
tests CPU/fake ports; it does not create owner authorization, consume a real
authorization, start tmux, launch training, or enter T7F.
