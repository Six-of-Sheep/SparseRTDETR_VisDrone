# RT-DETRv2 VisDrone T6A Detached Owner Authorization Contract

This document defines the checked-in T6A launch contract. It is a design and
validation boundary only. It does not authorize a production launch and it
does not create a nonce or an owner authorization artifact.

## Scope

The T6A config freezes the identities and policies that a later launch stage
must bind. The config is UTF-8 JSON with one final LF and a closed schema. Its
`readiness.static_config_authorizes_production` value is false. The string
`production authorization requires detached owner artifact` is a requirement,
not an authorization.

The pure Python module exports only these validation APIs:

- `TrainingLaunchContractError`
- `canonical_training_launch_contract_bytes`
- `load_training_launch_contract`
- `validate_training_launch_contract`
- `training_launch_contract_binding`
- `canonical_owner_authorization_bytes`
- `validate_owner_authorization`
- `owner_authorization_binding`

The module is standard-library-only. It does not import torch, write files,
call subprocess, call tmux, start a process, access a Dataset or DataLoader,
read annotations, read test or confirmatory data, generate a production
target, or sign an authorization. A caller supplies a detached object or its
canonical raw bytes and supplies the external raw size and SHA-256.

## Detached Authorization

The owner authorization is external to Git. It must be a regular non-symlink
file with mode 0600, uid 1000, gid 1000, and nlink 1. Its raw bytes are UTF-8
canonical JSON followed by exactly one LF. The canonical form has no LF. The
validator rejects duplicate keys, BOM, NUL, CR, non-finite numbers,
bool-as-int values, open schemas, unsafe paths, invalid Git IDs, weak or
placeholder nonces, and any identity mismatch.

T6A does not generate the formal nonce. A later owner-authenticated launch
transaction must provide a fresh lowercase hexadecimal nonce of at least 128
bits, bind it to the run, and provide its external raw size and SHA-256. The
authorization must contain `authorized=true`; that key is intentionally absent
from the static config.

The authorization binds all of the following to the launch-time observation:

- canonical repository path, branch, HEAD, tree, parent, and upstream;
- contract canonical and raw identities and all declared source bindings;
- the run ID, tmux session, three evidence targets, and six absence facts;
- Python, tmux, GPU UUID, driver, CUDA, power limit, filesystem device, and mount;
- train_core and development roots and manifest identities;
- sealed test and confirmatory policies;
- initialization, seed, topology, schedule, AMP, EMA, evaluator, checkpoint,
  and no-overwrite/no-resume/no-retry/no-fallback policy;
- state machine and permanent-failure policy.

The T6A validator receives these facts; it does not inspect or alter the
future production targets. A later stage owns the actual target absence gate.
Conversion R3 identity is declared in the source binding but T6A does not
read the conversion data or any test/confirmatory directory.

## Contract Binding Closure

`training_launch_contract_binding()` returns one closed object. Its exact
top-level fields are `schema_version`, `contract_id`, `relative_path`,
`raw_size_bytes`, `raw_sha256`, `canonical_size_bytes`,
`canonical_sha256`, `contract_identity`, `source_bindings`, `targets`,
`run_identity`, `repository_reference`, and `contract`. The authorization
validator rejects missing, extra, renamed, flattened, reordered-container, or
non-builtin binding structures.

The validator checks the complete frozen contract and requires every binding
copy to agree: the flattened config identity equals `contract_identity`, all
declared `source_bindings` equal both the frozen source inventory and the
contract copy, and `targets`, `run_identity`, and `repository_reference`
equal their corresponding contract fields. Each source identity is therefore
validated before authorization bytes are parsed or accepted; a whole-object
digest is not the sole semantic gate. `owner_authorization_binding()` uses the
same validation path and cannot bypass it.

## State Separation

These are distinct states and must not be inferred from one another:

`contract present` != `owner authorized` != `preflight pass` != `launch accepted`
!= `running` != `terminal complete` != `independently certified` !=
`training certified`.

The frozen state sequence is:

`DESIGN_ONLY` -> `OWNER_AUTHORIZED` -> `PREFLIGHT_PASS` -> `LAUNCH_ACCEPTED`
-> `RUNNING` -> `TERMINAL_COMPLETE` or `PERMANENT_FAIL`
-> `INDEPENDENT_TERMINAL_AUDIT_PASS` -> `TRAINING_CERTIFIED`.

After `LAUNCH_ACCEPTED`, resume, retry, overwrite, and fallback are forbidden.
Signals, timeout, host reboot, nonzero exit, partial write, identity drift,
disk full, incomplete evidence, unloadable checkpoints, and checkpoint
inventory drift are permanent failures. A `tmux accepted` result only proves
that an outer request was accepted by tmux; it does not prove that the child
started, that the child used the bound identity, that training ran, or that
terminal evidence is durable. Training certification therefore cannot be
derived from tmux acceptance or an outer return code alone.

## Later Launch Boundary

T6A ends before production authorization. The later transaction must use a
fresh immutable gate, a fresh GPU gate, and a target-absence gate. It must then
verify owner authorization against the launch-time repository, environment,
data-role and filesystem observations. Only after those gates may it make one
outer invocation and one tmux session request. It must immediately snapshot
outer stdout/stderr as bytes and then stop the authorization phase. Terminal
evidence, checkpoint atomicity/loadability/inventory, child identity, and the
independent terminal audit remain separate later responsibilities.

No T6A file changes `P3_TRAINING_READY` or starts T6B/T6C. At the end of this
implementation stage the required states remain:

```
P3_T6A_OWNER_DECISION_RECORDED=true
P3_T6A_DETACHED_AUTHORIZATION_CONTRACT_IMPLEMENTED=true
P3_T6A_INDEPENDENT_AUDIT_PASS=false
P3_T6_IMPLEMENTATION_READY=false
P3_T6_INDEPENDENT_AUDIT_PASS=false
P3_TRAINING_READY=false
P3_TRAINING_PRODUCTION_LAUNCH_EXECUTED=false
MODEL_SELECTION_CERTIFIED=false
TEST_ACCESS_READY=false
SPEED_MEASUREMENT_READY=false
P3_CONFIRMATORY_METRICS_ACCESSED=false
P3_DATASET_TEST_SPLIT_ACCESSED_BY_THIS_STAGE=false
```
