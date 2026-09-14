# baseline-v2b: monitored 640 paired control

This extension implements the user's 2026-09-14 authorization for one sequential
640 comparison. Source files and CPU tests are engineering evidence; neither a
prepared campaign nor a successful smoke is a scientific result. The original
CPU engineering/runtime configurations retain their original scope. This entry
does not change the v1/v2a contracts or the data protocol.

## Scope and frozen transitions

`tools/run_training_v2b_campaign.py` prepares an immutable, uniquely named
campaign under `artifacts/training`, then runs separate worker processes in this
order: A smoke, A checkpoint replay, B smoke, B checkpoint replay, A 30 epochs,
B 30 epochs. A is physical 16 with accumulation 1; B is physical 8 with
accumulation 2. Both use logical batch 16, seed 0, the same pretrained SHA,
initialized tensors, source tree, sample order and actual augmented input bytes.

Each smoke executes four logical windows, saves at window two, and stops after
window four. A fresh process restores window two and replays the final two
windows. It also exercises the fixed development evaluator on four images;
these preview metrics are not scientific endpoints. All four workers and their
post-exit native monitors must pass before the controller writes formal freeze.

The formal A and B scientific templates are fixed before A starts. Only B's
reference to A's completed evidence is filled later. Each worker verifies the
source inventory, and the controller verifies the clean commit/tree and every
tracked file between stages. A failure stops this invocation; the controller
never retries, changes batching, resumes a failed formal run or splices runs.
Any code/configuration change after formal A starts invalidates continuation
into B. Ordinary pre-formal harness repairs require a new campaign ID and a
complete new smoke. Hard stop events may not be retried automatically.

## Training and evaluation semantics

- 640, BF16 model and criterion, no GradScaler; AdamW 1e-4, weight decay 1e-4,
  gradient clipping 0.1, warmup 2000 updates, EMA 0.9999 with warmup 2000.
- Live BatchNorm statistics and native denoising per physical microbatch.
  Equal logical batches do not make BN, DN, matching or optimization trajectories
  identical. Their between-arm differences are recorded rather than rejected.
- Loss normalization uses the common logical target total; optimizer, warmup and
  EMA advance once per complete logical window. Scheduler and epoch clocks follow
  the original nominal 120-epoch recipe, including internal augmentation stop
  117. This 30-epoch experiment does not rescale that recipe.
- Exactly 4869 train_core images, 304 complete logical batches per epoch, five
  dropped samples. All 30 epoch-order hashes are fixed before execution. Workers
  use spawn, two workers, prefetch two and stateless per-sample augmentation.
- Each completed epoch has its own checkpoint. EMA/FP32/640 development
  evaluation uses batch four and zero workers at epochs 10, 20 and 30. Epochs
  10 and 20 are health checks; epoch 30 is the primary comparison endpoint.
- The primary metric preserves the existing converted formal-GT/legacy matcher
  semantics. Spatial-ignore sidecars are not introduced. COCO's 12 secondary
  metrics, including AP_small, are saved separately with evaluator axes and
  tensors. This does not certify full official-ignore semantics or model
  selection. A single seed does not support statistical significance claims.

Only train_core and development are data inputs. Confirmatory/test, data
protocol changes, 896/960, sparse methods and model-selection certification are
outside this campaign.

## Checkpoint and replay acceptance

The repaired campaign explicitly uses `sampling_backend=deterministic_gather`.
The model factory retains `native` for CPU and isolated diagnostics, but the
paired campaign entry point and every formal worker reject native sampling.
This selection is present in the engineering configuration, CPU verifier,
train_core prerun, smoke, checkpoint, runtime and formal A/B contracts.

The repair replaces only each model instance's v2 bilinear sampler core. It
preserves bilinear interpolation, zero padding, `align_corners=False`, gradients
for values and locations, and the vendor's layout, weighting and summation
order. Vendor files, attention geometry, DN/BN policies and the experiment
design are unchanged. CUDA uses strict deterministic algorithms, not warn-only,
with `CUBLAS_WORKSPACE_CONFIG=:4096:8` established before CUDA initialization.
Raw and EMA callable identities and partial arguments are checked against the
bound implementation; a state dictionary alone cannot identify this wiring.

The native operator diagnostic showed different value gradients for identical
inputs without any checkpoint restoration. A separate native model diagnostic
on commit `c5a178b3114b98c9720dc3b757ad20fd5e6efb6c` restored window two exactly,
then produced parameter differences at window three and BN plus out-of-tolerance
loss differences at window four. Its tensor layouts, inputs, RNG and clocks
matched. The candidate operator passed 64 repeated CUDA forwards/backwards in
each of FP32 and BF16-value/FP32-grid autocast, with exact output and both input
gradients. These observations motivate the repair; successful full-model
smoke and fresh-process replay remain required before any formal experiment.

Smoke/replay always save actual receipts and complete BN snapshots before
comparison, including failures. Native-only core proxies record AMP input
metadata; candidate workers preserve their strictly verified callable and keep
the existing model/criterion AMP, BN and DN observations. No replay threshold
is relaxed by this repair.

The restore boundary must exactly reproduce model/EMA/optimizer state, Python,
NumPy, CPU and CUDA RNGs, loader cursor, module modes and every update clock.
Replayed windows must preserve exact input receipts, loss denominators, BN/DN,
RNG and clocks. Logical loss relative error is bounded by 1e-4; raw parameter
relative L2 and maximum absolute error are each bounded by 1e-5.

Continued EMA and AdamW moments have separate aggregate relative-L2 and
maximum-absolute bounds of 1e-5. Keys, shapes, dtypes, optimizer scalar settings,
steps, BN buffers and geometry caches are exact. All ordinary floating tensors
must be finite. The existing model's invalid-anchor +inf sentinel is allowed
only in its verified geometry field with exact positions and complete bytes;
this is not an exception for nonfinite losses, gradients or optimizer values.
Preview predictions and metrics are diagnostics, not evidence of replay
equivalence. Hardware clearance is independently required for every worker.

## Native hardware admission

`training_v2b_admission.py` grants a nonserializable capability only to a live
worker with an independent native guardian. JSON cannot grant admission. The
older passive gate remains BLOCKED because this driver does not expose a getter
for the locked graphics upper bound. The admission binds either an agent-captured setter return or an explicitly
reviewed external administrator acknowledgement, plus current native identity
and bounded observations. It does not claim an unavailable getter or prove an
unobservable continuous-time clock bound.

The policy is bound to the complete SHA256 of the user authorization and the
reviewed evidence manifest, boot, host, GPU UUID/PCI, driver, NVML library and
nvidia-smi executable. Only the exact reviewed Xorg process is allowed: PID,
start time, executable hash/ownership/mode, UID, command line, cgroup, parent and
local active logind session all match; its graphics memory is at most 16 MiB.
No other graphics or compute process is allowed. During execution only the
exact owned worker may have a compute context.

Every worker first requires an idle GPU, unchanged identity, acceptable
temperatures/power limits, and a complete kernel journal anchor. In the original
`direct` or `sudo_n` modes, exactly one 1500/1500 MHz setter targets the bound
UUID, without fallback or retry; its actual return code and diagnostics are
retained and identity is checked again afterward.

The current campaign explicitly uses `external_admin_acknowledged`. This mode
never invokes a setter. It binds the reviewed receipt (complete-file SHA256
`aed11ff98f0b8e5478cc757e6f5929e22405972e535c97f1c586d9ec5c51847c`),
original user-supplied terminal bytes and native sudo journal. The native
command record identifies this boot, host, sudo PID, target UUID and exact
1500/1500 request; its same-PID PAM records identify the root session. The
terminal transcript must contain one matching success acknowledgement followed
by one `All done.`; its SSH connection-close line is separate transport text.
Both complete driver formats, `(1500, 1500)` and
`(gpuClkMin 1500, gpuClkMax 1500)`, are accepted; mixed, duplicate, incomplete,
wrong-target or diagnostic output is rejected.

The externally observed acknowledgement does not supply a native setter exit
code, separate stderr capture, loaded setter library or command monotonic
start/end. Those unavailable fields remain null or explicitly unavailable.
Known later clock/reset commands invalidate the receipt. Native journal checks
cannot establish the absence of every unlogged direct-root operation; current
identity, bounded sampling and reset/reboot detection remain mandatory.
Settings are left at 1500/1500 after every worker exits.

The two exact historical boot BERT records are retained with full raw hashes.
They are unresolved historical firmware evidence, not proof of healthy RAM.
New BERT/MCE/Xid/driver reset/load/unload observations stop the worker. Journal
continuity is required. EDAC unavailability is explicitly reported as
unavailable, never as zero errors.

Graphics and SM clock observations run together every 0.2 seconds, both with
zero tolerance above 1500 MHz and a maximum age/gap of one second;
health observations run every second with a maximum of two seconds. At least
three preload observations and loaded observations are required. A separate
controller watches the guardian's main-loop heartbeat every 50 ms (heartbeat
target 0.1 seconds, maximum gap one second), including native sample ages.
Samples form a hash chain. Admission and final reports are published atomically
without overwriting existing evidence. The guardian also watches the evidence
writer and the worker's pidfd, and observes context release after worker exit.

Any foreign compute process, clock proof failure, observation gap, new hardware
error, identity/reset/reboot event, OOM, nonfinite value, input mismatch,
checkpoint/replay mismatch or batch change stops the campaign without automatic
retry. Termination is confined to this controller's owned worker and, after
confirmed worker exit, its bound failed guardian. CPU fault
injection uses only explicitly spawned dummy processes, including an unaffected
sentinel that verifies process isolation.

## Durable evidence

The campaign binds every tracked source file, commit/tree, full input/configuration
hashes, initialization, all epoch orders, and its authorization/policy. Workers
save actual per-window sample/augmentation receipts, target denominators, loss
families, BN/DN, clocks, CUDA memory and elapsed time. They save initialization,
checkpoints, exact restore state, fixed-epoch predictions/metrics and terminal
reports. Native startup/setter/heartbeat/sampling/final records and controller
launch/exit records are linked by complete file hashes.

Budget approximately 22–26 GB for the two arms, smoke, per-epoch checkpoints and
evidence; verify free space and reserve before launch and retain every failed
scene. CPU validation uses the explicit engineering scope, reporting historical
data-dependent skips rather than manufacturing a completed historical result.
The clean archive contains tracked source plus the already allowlisted minimal
CPU fixtures; it never copies whole data-conversion or held-out partitions.
