# RT-DETRv2 R18 VisDrone Baseline-v2a T7C Runtime Integration

## Scope

T7C adds the CPU/fake runtime integration for the frozen T7B baseline-v2a
contract. It is an independent runtime surface. T7B's JSON, contract module,
tests, and document remain unchanged, as do all V1 and T6B modules.

The runtime module is inert at import time. Importing it does not import
PyTorch, initialize CUDA, read a dataset, create a runtime target, or launch a
process. The fake runner is the only executable runner in this phase. It is a
CPU evidence exercise and is not formal training.

## Contract binding and model load

`build_v2a_runtime_policy` first calls
`training_v2a_contract_binding`. Topology, optimizer, AMP, EMA, selection,
checkpoint, model, and data-role values are copied from that detached binding;
the runtime does not maintain a second set of v2a policy constants.

`load_v2a_pretrained_backbone` is the sole real model factory. It validates the
local T7A weight identity, uses the vendor PResNet-18-vd constructor with
`pretrained=false`, loads the bytes with `map_location="cpu"`, and performs a
strict state-dict load. Missing keys, unexpected keys, inventory drift, byte
drift, special files, and network or random-initialization fallback are
rejected. The optimizer is created only after this load returns successfully.

The optimizer audit uses the case-sensitive `norm`/`bn` selector from the
binding. The two groups are mutually exclusive and cover every trainable
parameter. Both groups replay the bound `1e-4` learning rate; the V1/T6B
backbone learning-rate override is not available.

AMP is represented by CUDA bfloat16 autocast only. No scaler object or scaler
state is constructed, called, or serialized. The fake runner performs one
direct optimizer step per valid micro-batch and rejects batch adaptation.

## Runtime closures

The five closures are independently represented in the policy and terminal
result:

1. Raw process stdout bytes are retained and the last canonical terminal JSON
   line is selected after preceding vendor log lines.
2. The existing `artifacts/training` parent identity is captured while the
   independent T7C target is required to be absent before the run.
3. Only the independent `train_core` and `development` raw-image roots are
   bound. Their COCO annotations are resolved through the frozen Conversion
   R3 artifact binding, never by appending annotation names to image roots.
   The canonical annotation path and object identity are carried into the
   runtime policy. Confirmatory and test roots are never read.
4. The supplied fake environment observation must match the frozen GPU/CUDA
   identity, remain at or below the 1500 MHz clock cap, and report an
   exclusive host with no other training load. No hardware probe is run.
5. `EpochProgressWriter` appends one canonical JSONL record per epoch, flushes
   and fsyncs each record, and returns a final byte/hash inventory bound into
   the terminal result.

Fake checkpoints carry contract, T7A authority, parameter-group, environment,
and source identities, plus raw and EMA state. Checkpoint validation rejects
scaler state, raw/EMA aliasing, identity drift, and digest drift. Development
selection uses EMA metrics and the bound unrounded AP, AP50, AR500, and earlier
epoch tie-breakers.

## Boundary

T7C does not consume owner authorization, probe a GPU, access confirmatory or
test data, launch a process, train, certify model selection, or open training
readiness. The runtime reports these states as false. A later independent
terminal audit is required before any readiness claim.
