# RT-DETRv2 baseline-v2b epoch 30→60 continuation contract

## Scope

This contract closes the engineering gap identified by the T8A continuation-eligibility audit. It does not authorize GPU execution. A later owner authorization must bind the frozen continuation campaign and native hardware policies.

The four prospective epoch-30 checkpoints (`s1_r640`, `s1_r896`, `s2_r640`, and `s2_r896`) retain their original run bindings. The continuation campaign has a distinct outer execution identity. It must never claim that a historical checkpoint or historical run was created by the continuation campaign.

## Frozen scientific boundary

- Source boundary: complete epoch 30, 9,120 optimizer/EMA updates and 18,240 physical microsteps.
- Continuation: epochs 31 through 60 inclusive.
- Target boundary: complete epoch 60, 18,240 optimizer/EMA updates and 36,480 physical microsteps.
- The original resolved training horizon remains 120 epochs.
- The optimizer, scheduler, warmup, augmentation, BN, DN, loss, EMA, data, and development-evaluation semantics remain those in the source run binding.
- The epoch-order digests for epochs 31–60 are frozen before launch for each seed.
- Confirmatory and test data remain forbidden.

## Identity model

There are two deliberately separate identities:

1. **Source training identity.** The exact historical run binding embedded in the epoch-30 checkpoint. Model construction, CUDA admission, checkpoint restore, data binding, and every subsequent checkpoint continue to use this identity.
2. **Continuation execution identity.** A new campaign/cell/stage identifier plus a frozen inventory of the continuation controller and worker. This identity proves who resumed the source state and where the new evidence was written; it does not replace the source training identity.

All source checkpoints, source results, source contracts, and source run bindings are immutable references outside the new output tree.

## Required stage machine

Each cell executes three one-shot stages:

1. `smoke`: restore the certified epoch-30 boundary, enter epoch 31, execute four logical windows, and save a midpoint after window 2.
2. `smoke_replay`: in a new process, restore that midpoint and reproduce windows 3–4 under the original replay tolerances.
3. `formal60`: in a new process, restore the original epoch-30 checkpoint (not a smoke checkpoint), execute exactly 9,120 logical updates, evaluate development at epochs 45 and 60, and close epoch 60.

For each seed, the 640 arm precedes the 896 arm. The 896 worker authenticates and compares the corresponding 640 logical-window receipts. Any worker failure produces `STOP_NO_RETRY`; the campaign stops and preserves the scene.

## Prohibited behavior

- changing training-core or vendor scientific bytes;
- starting from fresh update zero while claiming continuation;
- relabelling a historical run or checkpoint;
- overwriting a historical or new output;
- continuing a failed formal stage, retrying it, or adapting batch size/precision;
- accessing confirmatory/test data;
- running without a fresh hardware gate and separately approved owner launch authorization.

## CPU engineering acceptance

Before any GPU authority is requested:

- contract mutation tests must reject altered lineage, schedules, identities, paths, and prerequisites;
- the worker bootstrap must authenticate the contract before project/torch/CUDA imports;
- the selected historical checkout must be first on `sys.path`;
- source training-core identities must remain byte-equal to T8A;
- fake stage-ledger tests must enforce 4/2/9,120 receipts and the exact target clocks;
- repository checker, tracked Python compile, diff check, cache-zero, targeted tests, and the applicable CPU suite must pass.

CPU acceptance is not GPU readiness, owner authorization, or a training result.
