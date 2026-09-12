# RT-DETRv2 R18 VisDrone Baseline-v2a T7D Production Boundary

## Scope

T7D adds the production execution boundary for the frozen T7B baseline-v2a
contract and the independently audited T7C runtime policy. It is a new,
independent surface: T7B and T7C bytes remain unchanged, and no V1 or T6
production target, lock, receipt, process evidence, outer evidence, or
checkpoint name is reused.

Importing the module is inert. It does not import PyTorch, initialize CUDA,
read a dataset, create a target, consume owner authorization, or launch a
process. The public production entry accepts only a detached launch descriptor
and its authorization context. Injectable ports remain confined to the
separate CPU-fake API used by tests.

## Descriptor and authorization closure

`build_v2a_detached_launch_descriptor` validates the owner context while it is
still unconsumed, binds its run, nonce, contract, data-root, environment, and
source identities, and writes the final descriptor-binding digest back to the
caller-owned context. The descriptor and context therefore form a single
cross-bound pair. Closed schemas, builtin scalar types, canonical digests, and
cross-field equality are revalidated before the pair can reach the production
entry.

The production entry validates the pair before rebuilding the T7C policy. The
only accepted data roles are `train_core` and `development`; test and
confirmatory roles are forbidden. Authorization consumption is deliberately
outside this module and is reserved for the later exactly-once launcher.

## Ordering and capability boundary

The enforced order is:

1. validate descriptor and authorization context;
2. bind the frozen T7B contract and T7C runtime policy;
3. create-exclusively claim the independent T7D target;
4. strict-load the T7A PResNet-18-vd pretrained backbone;
5. create the optimizer;
6. create the data and primary-evaluator ports;
7. execute the epoch loop and close durable evidence.

Real model, data, evaluator, and execution objects are carried by an internal
capability that public callers cannot construct or inject. The CPU-fake entry
rejects production capability, authorization-context, real-data, and
real-model keys before making any durable target claim. For all other valid
requests, create-exclusive target semantics remain the first irreversible
operation and reject overwrite, resume, and duplicate execution.

## Training semantics and durable evidence

All training values come from the T7C policy. The real model factory uses the
T7C strict pretrained loader before optimizer creation. The real loaders are
bound to the owner-authorized `train_core` and `development` raw-image roots.
Their annotation file paths are the canonical Conversion R3 paths carried by
the T7C identity binding, never paths synthesized under the raw-image roots.
The isolated production process selects PyTorch's `file_system` tensor-sharing
strategy before constructing four-worker loaders, avoiding file-descriptor
exhaustion under the host's fixed descriptor limit.
Samples and targets are transferred to the
model device; the vendor criterion applies its `weight_dict` before returning
the base, auxiliary, DN, and encoder loss values, and the production layer
sums those returned values exactly once without reapplying the dictionary;
gradients are clipped at the frozen maximum norm. The vendor 2,000-update
linear warmup precedes the inert 1,000-epoch milestone scheduler. AMP is CUDA
bfloat16 autocast with no GradScaler object, state, calls, or checkpoint
fields. Every valid micro-batch produces one direct optimizer step, followed
by EMA update; development evaluation uses the independently certified
VisDrone primary evaluator over EMA state and returns unrounded `AP`, `AP50`,
and `AR500` for deterministic selection.

Epoch progress is canonical JSONL flushed and fsynced per record. Production
loss tensors are detached and converted to a finite scalar for every batch;
the recorded `mean_loss` is the arithmetic mean of those observed values.
last, best, periodic, and final checkpoints are durable `torch.save` artifacts
that bind raw and EMA model state, optimizer, scheduler, warmup, RNG,
no-scaler semantics, source, contract, environment, data, and run identities.
Every checkpoint is fsynced, read back for loadability, and recorded in an
append-only inventory; CPU-fake checkpoints remain canonical JSON fixtures.
Terminal result and stdout carry recomputable byte and digest identities.
Success, failure, exception, and signal outcomes are classified without
accepting a second terminal object.

## Readiness boundary

T7D CPU/fake tests demonstrate the call graph and durable schema only. They do
not create owner authorization, access production data, probe a GPU, start
tmux, launch production training, certify model selection, or make training
ready. An independent terminal audit of the committed T7D bytes is required
before any later exactly-once launch phase.
