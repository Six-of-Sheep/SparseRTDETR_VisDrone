# Baseline-v2b: 896 matched control contract

The intervention changes training and development input size from 640 to 896.
The comparison reference is the completed, sealed 640 × 8 × 2 B experiment,
authenticated by its complete-file completion SHA-256
`8bf192eb292ff6cc2687a09879ff670632b360dca45c4e2ab6003807db53aaf9`.
The old checkout and evidence remain read-only. A new isolated checkout and a
new, exclusive campaign directory hold all 896 work.

## Scientific definition

The 896 worker uses physical batch 8, accumulation 2, logical batch 16, seed 0,
the same pretrained authority, BF16 model and criterion without GradScaler,
native DN per physical microbatch, training BatchNorm, vendor EMA, optimizer,
warmup and scheduler clocks as completed B. Sampling remains deterministic
gather. Workers remain spawn/2 with prefetch 2. Data remains the frozen
train_core (4,869 images; 304 complete logical windows, five dropped images per
epoch) and development (548 images), with the same 30 ordered epoch plans.

Only input size changes in the scientific configuration. The development policy
changes its input size and Resize size together; all other policy fields remain
fixed. Vendor code, accumulation engine, checkpoint, runtime, device placement,
sampling implementation, evaluator and data-protocol mathematics retain their
matched source SHA-256 values. The reviewed support and orchestration additions
have a new source commit and complete source inventory.

Fresh learned parameters must match B before training. Complete model-state and
geometry hashes are resolution-specific: 640 has 8,400 decoder anchors and 400
positions at stride 32; 896 has 16,464 anchors and 784 positions. Each raw and EMA
cache is observed independently. EMA floating geometry buffers retain vendor
averaging semantics and must not be reconstructed to force equality with raw
buffers.

Source image identity, logical order and sample augmentation seed must match B.
Resized tensors, retained GT counts, loss denominators, DN group counts and
padded DN query counts need not match across resolutions. The existing
post-Resize SanitizeBoundingBoxes(min_size=1) can retain a narrow box at 896 that
is removed at 640; this mechanism is recorded, not silently changed. An exact
same-resolution recovery check still requires the original input receipts and
the established checkpoint/state comparisons.

This comparison estimates the effect of increasing both training and inference
resolution under this recipe. It does not isolate training size from evaluation
size, establish single-seed statistical significance, or equate overall AP with
AP_small.

## Fixed sequence and admission

The controller executes exactly one sequence:

1. Four natural 896 smoke windows, with checkpoint after window two.
2. A new-process replay of the remaining two windows, then independent
   same-resolution smoke/replay comparison.
3. A separate 896 capacity worker, followed by final monitor clearance and an
   independent native memory audit.
4. An immutable formal freeze record, then a fresh 896 30-epoch worker.

Epochs 10 and 20 are intermediate evaluation and health checkpoints. Epoch 30 is
the primary comparison endpoint. The controller never pauses for an additional
authorization at epoch 10, reuses a pilot checkpoint as a formal prefix, selects
a different epoch as the endpoint, resumes an interrupted formal experiment, or
automatically retries a failed stage.

Capacity uses four natural warmup windows; the three reviewed B stress
positions (14,82), (25,21), (27,128), expressed as logged epoch and zero-based
logical-window index; the raw metadata maxima for physical and logical GT
counts; and two synthetic envelope windows covering maximum per-image DN
padding and maximum physical-microbatch GT cost in both microbatch positions.
It also executes full development coverage in FP32 with EMA and batch four.
Capacity evaluation is labeled development_capacity_only and is not a fixed
epoch scientific result.

Both worker-side conservative free memory and independently audited native
memory.free must remain at least 3,072 MiB. The native audit consumes the
completed capacity worker and its final hardware monitor; a worker PASS alone
cannot authorize formal training. Emptying the CUDA cache or offloading a model
to satisfy the margin is prohibited. Capacity success is evidence for this
bounded pilot, not a permanent guarantee against memory or hardware failures.

## Hardware policy

Preparation is CPU only. The CLI authenticates the authorization text, matched
B completion, policy evidence and administrator clock receipt by complete-file
SHA-256. Only external_admin_acknowledged is accepted. There is no controller
clock-setting or clock-reset branch.

Each actual worker reuses the existing native admission and guardian supervisor:
fresh boot/GPU/driver/Xorg identity, exact Xorg allowlist, exclusive compute
ownership, known-boot BERT separation, explicit EDAC unavailability, incremental
hardware-log monitoring, immutable clock receipt and continuous runtime
sampling. The declared lock remains 1500/1500 MHz and every observed GPU clock
sample must be at most 1500 MHz. Existing sampling-gap limits remain unchanged.
The lock stays in place after exit.

A foreign compute process, unproven frequency bound, sampling gap, new
BERT/MCE/Xid, reset/reboot, OOM, nonfinite value, source/configuration drift,
logical-input mismatch or failed recovery ends the invocation. There is no
batch/resolution fallback. The inherited supervisor may signal only its
authenticated owned worker/guardian; unrelated project processes are untouched.

## Persistent evidence and failure behavior

The campaign records the exact new source commit/tree/file inventory, old B
completion and result references, unchanged pretrained/data references, explicit
896 configuration, development binding, ordered logical epochs and full inline
capacity plan. Worker contracts carry the same plan and matched completion.
All directories and JSON reports are created exclusively.

The smoke comparison, capacity completion and independent capacity-admission
report are bound into formal-freeze.json, together with the exact formal worker
template. Campaign bytes and source identity are checked before and after every
stage and again at the formal boundary. Each worker persists its native
monitoring, source/model/data bindings, initialization, receipts, update clocks,
checkpoints, restoration evidence, fixed epoch metrics and terminal result.

Any failure persists STOP_NO_RETRY and prevents all later stages. A repeated
invocation cannot overwrite or resume the same execution directory. Engineering
changes require a new reviewed source/campaign; they cannot be applied in place
after formal launch or used to splice a comparison.

The controller's CPU fake tests cover stage order, immutable identities,
failure transitions, receipt/hash checks, capacity admission and forbidden
setter modes. They prove orchestration behavior, not GPU capacity, CUDA
numerics, model accuracy or hardware stability.

No 960/1024 or other resolution experiment, confirmatory/test access, protocol
change, sparse experiment or model-selection certification is authorized by
this contract.
