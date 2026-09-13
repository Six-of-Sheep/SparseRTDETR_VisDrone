# RT-DETRv2 baseline-v2b engineering foundation

This stage supplies a CPU-verifiable model factory, accumulation engine,
checkpoint implementation and evidence writer. It does not modify the frozen
v1/v2a implementation, authorize production training, or certify scientific AP.
Only generated RGB tensors and normalized boxes are used by its verification tool.

The recorded foundation results below belong to commit74201c1. The subsequent
[runtime extension](RTDETR_BASELINE_V2B_RUNTIME.md) adds train_core loading,
multiworker recovery, schema2 CPU/CUDA RNG checkpoints and a native hardware
gate. Historical artifact hashes below must not be relabeled as tests of the
new source. tools/verify_training_v2b_cpu.py remains synthetic CPU-only; the
separate pre-run tool reads bounded train_core batches without model updates.

## Model and initialization

The factory builds the actual vendored RT-DETRv2 R18: 20,094,584 parameters,
300 detection queries, 10 classes and three decoder layers. Python, NumPy and
the torch CPU generator are seeded immediately before the first model
construction. Pretrained backbone bytes are checked against an explicit SHA-256,
loaded with weights_only=True on CPU, and applied strictly to the constructed
backbone. No second random backbone is constructed. The source authority is the
existing ResNet18_vd_pretrained_from_paddle.pth artifact; there is no download.

Run identity includes parameter bytes, persistent model buffers, actual imported
package/vendor source identities and resolved configuration. Foreign checkout
modules already cached under the vendor namespace are rejected. Shape and byte
identities cover resolution-dependent anchors and positional caches. Unregistered
positional tensors are also bound by checkpoint layout validation.

The default engineering configuration describes the 640, physical batch 16,
accumulation 1 target. The CPU tool records any smaller synthetic verification
batch and size explicitly. It accepts only 128 and 640. Dataset paths are cleared;
the factory constructs no Dataset, DataLoader or evaluator.

## Precision and objective

FP32 and BF16 are explicit choices. BF16 autocast covers model and criterion only.
Loss summation, reweighting, backward, clipping and optimizer update occur outside
autocast. There is no GradScaler, FP16 path or automatic batch/precision fallback.

For microbatch i with T_i augmented ground-truth boxes, let
D_i=max(T_i,1) and D=max(sum_i T_i,1). The logical objective is

    L_window = sum_i (D_i / D) * sum(all preweighted vendor losses_i).

There is no additional division by accumulation_steps. Empty microbatches retain
negative/background classification contributions. The engine takes every returned
loss key exactly once; family membership is never inferred from a fixed 21-key
schema. The actual positive-GT R18 case has base/aux/DN/encoder counts 3/6/9/3;
an all-empty microbatch omits DN outputs.

The common logical image/target batch is fixed before physical splitting.
logical_batch_indices supplies a deterministic, epoch-specific order independent
of model/DN draws; incomplete logical tails are dropped at the sampler boundary.
The engine rejects incomplete windows. This helper does not certify restored
prefetch or augmentation RNG in a real multiworker loader.

## BN, DN and clocks

All 69 BatchNorm modules are inventoried. The train policy updates their statistics
per physical forward. The frozen policy freezes statistics for every BN while
keeping affine parameters trainable and leaving the rest of the model in training
mode, including auxiliary and denoising branches.

DN remains the native per-microbatch construction. Its group count, padding/query
count, random corruption and attention context can differ between 16x1 and 8x2.
Equal effective batch therefore does not imply equal BN/DN behavior. BF16 and
different kernel/reduction shapes also preclude a bitwise batch-equivalence claim.

One completed logical window performs one clip, AdamW step, EMA update and warmup
step. Vendor EMA averages floating buffers and leaves integer counters unchanged.
A scheduler step occurs once at a completed epoch boundary when warmup is finished.
The warmup constructor starts last_step at zero; after N updates it is N.
Each window records GT counts, denominator, coefficients, individual losses,
gradient norm, DN metadata and update counters.

Nonfinite losses, gradients or optimized parameters, skipped updates, and partial
windows permanently invalidate the current engine. Buffers already changed by a
failed forward are not rolled back. Recovery requires a fresh component set and a
verified checkpoint.

CPU compatibility is explicit: bn_backward_layout=cpu_contiguous_cuda_native.
The tested torch 2.4.1 CPU BatchNorm backward is incorrect for certain batch-one
gradient strides produced by flatten/permute/concatenate. A minimal FP64 BN test
violates the analytic bias/input gradient formulas for stride (256,1,2048,256)
at shape (1,256,8,8). Normalizing only BN backward gradient layout to contiguous
fixes the analytic test and restores actual-model FP64 full/split gradient
agreement from about 3.15% relative error to 6.94e-16. The hook applies only inside
CPU engine windows, is removed on success or failure, changes no forward values,
and leaves the GPU path native. The FP32 model comparison retains its original
strict tolerances; no initializer or test input is changed.

## Checkpoint and evidence

Only phase-zero complete logical windows can be saved. Checkpoints include raw
model, EMA, optimizer, engine, scheduler, warmup, CPU Python/NumPy/torch RNG and
explicit named torch generators. Layout checks include parameter ordering,
tensor shapes/dtypes, persistent and nonpersistent caches, configuration and
update clocks. Loading requires an external complete-file SHA-256 and matching
code/config/initial-state/input bindings. Standard malformed states are rejected
before live components are mutated. Arbitrary third-party component load methods
are not a transactional recovery guarantee.

Publication is exclusive: an existing checkpoint or JSON output is never
overwritten. The evidence validator rereads the checkpoint and checks its internal
counters against a complete contiguous fresh-run window ledger. It verifies source,
input and artifact hashes; a self-reported counter or file name is insufficient.

CPU environment evidence is collected from the current process, host, boot and
installed environment. It always states cpu_synthetic, scientific_certified=false
and gpu_ready=false. Persisted hashes provide integrity, not a cryptographic proof
that an untrusted person ran a collector.

A COCO AP extraction helper supports only an explicitly supplied CPU precision
tensor and evaluator axes, with a real prediction-file reference. It rereads
precision/parameters/predictions and recomputes the chosen area, IoU and maxDets
slice. No evaluator or real annotation file is executed by this stage. It rejects
a maxDets=100 value relabeled as 500. A production primary-VisDrone metric adapter
is still required.

At the recorded foundation commit, the GPU collector had only mocked parser
tests. The runtime extension adds a passive native collector. Neither a GPU query
nor a CUDA context is run by the synthetic CPU verifier. Current graphics clock
is not evidence of a locked upper clock; hardware admission remains false.

## Verification artifacts and remaining gates

tools/verify_training_v2b_cpu.py accepts an explicit pretrained authority path and
a unique run ID. It writes a fresh artifacts/v2b_cpu_<run_id> directory containing
binding, initialization identity, complete windows, checkpoint and CPU report.
Two-window verification also rebuilds the actual model, restores after window one,
and compares the next window, raw model, optimizer, EMA, clocks and all recorded CPU
RNG states with uninterrupted execution. The one-window 640 mode verifies real
resolution geometry and BF16 forward/backward without making a resume claim.

The automated tests use both analytic small models and the actual R18 model.
They cover unequal/empty GT counts, DN changes, all-model BN behavior, encoder-head
optimizer participation, rejected partial/nonfinite updates, real replay,
malformed checkpoints and conflicting evidence. Numerical observations from
default-initialized model batch comparisons must be reported separately from the
logical-denominator proof.

The runtime extension closes the train_core sampler/augmentation and complete-
window recovery wiring. Actual CUDA execution still requires separate
authorization and native hardware admission; a trustworthy development metric
adapter remains necessary for scientific AP. Paired640 16x1 versus8x2 short
smoke precedes the scientific control: epoch10 is an early checkpoint and epoch30
the main comparison endpoint. Neither it nor896/960 training is authorized by
these files. Confirmatory/test access remains excluded.

Repository source validation has an explicit source-only mode for this isolated
worktree. It retains frozen file, source, configuration and vendor identities but
does not validate historical runtime/data artifacts. The default historical
checker behavior is unchanged. Old tests that require the completed historical
training directories or the complete data conversion runtime must be reported as
outside this CPU engineering scope, not as passed. Selected metadata-only and
pretrained-authority regressions use byte-verified copies; no images, annotation
datasets or historical training targets are copied.

## Recorded CPU verification on 2026-09-13

Python 3.10.16 and torch 2.4.1 were used with two CPU threads and no CUDA
initialization. All model inputs were generated tensors.

| Verification | Result | Scope |
| --- | --- | --- |
| Four v2b test modules | 258 passed | Actual R18 and analytic engine/checkpoint/evidence tests |
| Selected legacy suites | 142 passed, 19 deselected | Metadata, authority and source regressions; no real annotations |
| Source repository checker | SOURCE_ONLY_PASS | Runtime artifacts were not validated |
| R18 BF16, 128, physical 1 x accumulation 2 | 2 updates, 4 microsteps | Next-window checkpoint replay is byte exact, including optimizer/EMA/RNG/caches |
| R18 BF16, 640, physical 1 x accumulation 2 | 1 update, 2 microsteps | Actual geometry, forward, backward, finite update and evidence; no replay claim |

The 19 deselected legacy cases require complete historical runtime targets or
the original train_core/development annotation artifacts. They remain outside
this stage's validation. The frozen vendor emits one existing deprecation warning.

The append-only reports are under the worktree's ignored artifact directory:

- v2b_cpu_r18-bf16-128-replay-20260913-r1/verification.json:
  SHA256 aa6de987f4ffced31994e60176323a52b357079c6dafef1b0106d7d71815f328.
- v2b_cpu_r18-bf16-640-window-20260913-r1/verification.json:
  SHA256 aef720a58aa92d411e6d27595c5659eb8758538657bfbf0c7861976963049bf9.

Each report references its actual checkpoint, full run binding and window ledger.
These observations establish CPU engineering behavior, not GPU batch capacity,
GPU stability, real-data augmentation recovery, AP, or scientific significance.
