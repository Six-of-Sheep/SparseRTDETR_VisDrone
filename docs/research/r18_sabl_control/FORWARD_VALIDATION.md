# S/A raw-model forward validation

Scope: engineering convergence of the dedicated PR27 entry. The numerical repair
base is `992b6b13f0b866700a9bd631ec1a557da96e025e`. No source under `src/`,
`configs/` or `vendor/` changes here. Historical STOPs, releases, N/S endpoints,
the separate A-u938 state and all incurred costs remain immutable. This CPU
preparation does not authorize a GPU invocation.

## Fixed computation and state bridge

A reads the independent raw model at update 938 (SHA b9e7a005...); S reads the
original raw model at e24/update 7296 (SHA deb67cf4...). These are not EMA weights.
The registered full-state v1 schema, original contract/manifest, parent N, runtime,
clock/config, criterion identity, full model layout and training modes are checked.
Each arm has 558 state tensors (346 parameters and 212 buffers). Raw tensors are
strictly copied and compared byte for byte; only its own saved RNG is restored.
Optimizer, EMA and engine are neither restored nor constructed by this entry.
This explicit v1-to-v2 raw bridge is not a complete training-state recovery.

The frozen input remains e4/window26, first microbatch of the original logical16.
The original loader/input_record identity, all 16 IDs, pixels, GT and ignore SHA
must match before using its first 8 images. A's flat RoI352 (b1/q52, zero-based)
retains exact comparisons of roi_before_A_mean, roi_after_A_mean, pre.3, att_x and
att_y against the saved FP32 tensors. The first mismatch stops execution.

Keep training mode, native BN-layout context, DN/aux and original guards, enabled
autograd, BF16 native detector and FP32 local head, deterministic algorithms and
CUDA compilation. The repaired operator remains logsigmoid plus softmax along
the original height/width axes 2/3. We retain the original finite-output and <=1e-6
axis-normalization checks and the specified A zero old denominator check.
No warm-up, extra detector forward, loss evaluation, objective backward or update
exists in the dedicated call path.

## Update prevention and observation

No global torch.autograd.grad/backward/Tensor.backward patch and no broad
function-name blacklist is used. hashlib.update and compiler-internal autodiff
remain available. The entry builds the raw model, computes its two forwards and
releases it. It never instantiates an optimizer/EMA updater or computes a training
objective. Parameter hashes and each parameter's gradient state are checked before
and after the call; all gradients must remain None. Partial failure observations
also attempt the same audit. BN running means/variances/counters may change in
memory; their before/after SHA, dtype, changed elements and maximum absolute
change are recorded. Other buffer changes reject the run. Nothing is written back
to a checkpoint.

Saved logits/features are reused for local old/new normalization and pooling
comparisons under no_grad, without an additional detector call. These observations
cannot establish long-term stability or detection benefit.

## One registered caller and closure

`tools/sabl_forward_check.py run` is the actual caller. It validates the immutable
source manifest, physical material/protection indexes, exact interpreter, release
cwd, module origin, original runtime metadata and inherited cost. An exact typed
JSON authorization binds commit, manifest SHA, definition SHA and attempt. Only
then is the exclusive FORWARD_ONCE marker consumed. Existing targets reject a
second invocation. A CPU `run --check-only` traverses the same admission code,
never creates authorization/attempt/once markers, and never calls the detector.

`prepare` performs real hidden-CUDA checkpoint loading and strict raw bridges on
CPU and writes a fresh definition. It accepts only registered bytes/size_bytes
physical reference forms, rejects missing/invalid/conflicting sizes and SHA errors,
and explicitly extracts nested protected_files. Empty files are allowed only for
source-manifest members; checkpoint and protection records require positive sizes.
Source paths, duplicates and release/source identity are checked.

Success writes RESULT and SHA-bound CLOSED and exits. Failure samples allocator
peak and storage/cache best-effort, retains the primary traceback and any secondary
receipt error, writes STOP and exits. Failed allocator sampling remains UNKNOWN
with a separately labelled last-known peak. A failed or reserved forward is charged
and is never refunded. No path resumes training, scoring or speed measurements.

The same stop_receipt and whitelist exporter are exercised by the labelled synthetic
CPU CLI. `export` requires exactly one terminal marker, verifies CLOSED/result SHA,
and checks prior return bytes plus its JSON wire size and decoded attachments
against 1MiB. It never includes checkpoints, images or the full cache index.

## Resources and caches

The limits remain 2 batch8 detector forwards, zero training-objective backward,
zero optimizer/EMA updates, 300 GPU-stage seconds, 600 wall seconds, 18GiB allocated,
128MiB added storage, and 1MiB total SSH/attachment return. Wall timing starts before
admission and GPU timing before any CUDA resource call. The original 1 failed
forward and conservative 9.326126370113343 seconds remain charged: inherited total
31741 forwards / 36528.27560652013 seconds. No experiment cost is spent by this CPU
preparation; CPU operator autodiff is recorded separately from detector updates.

Inductor, Triton, extensions, CUDA and XDG cache paths are bound inside the task
before imports and must start fresh for GPU execution. They are included once in
task storage; positive growth in explicit default cache roots is additionally
charged. Full cache indexes stay server-side, while compact summaries and their
SHA enter the return. Task bytes plus 32MiB development/Git reserve plus external
cache growth must stay within 128MiB. This is the declared cache scope, not a
claim of whole-filesystem allocation measurement. Peak allocation is cumulative
across A/S and is never reset between arms.

## Acceptance and remaining unknowns

| Check | CPU evidence | Still requires the two authorized GPU forwards |
|---|---|---|
| Historical materials / raw state | Physical SHA, native v1 full schema, 558-tensor bridges, exact interpreter/cwd/origin | Actual CUDA state use and saved input reproduction |
| Dedicated entry / no updates | Actual CLI, parameter/gradient/BN audits, rejection tests; no global API patches | Full-model post-forward state and BN observations |
| RoIAlign / compilation | Native CPU RoIAlign and CPU aot_eager tracing only | Deterministic CUDA RoIAlign / Inductor compilation |
| Local numerical repair | Unchanged operator tests and constructed FP32 cases | A saved local exactness, repaired weights and downstream outputs; S full outputs |
| Failure / export | Same failure handler, simulated allocator failure => UNKNOWN, bounded SHA whitelist | Real exception peak, real cache growth and actual GPU-stage cost |
| Science | No detector learning, images or scientific evaluation | Long training stability and detection quality remain outside this validation |

Server-only CPU acceptance uses the final interpreter and actual caller. CI on
Python3.10/3.12 supplements it. The inherited optimizer-containing jobs remain
explicitly skipped, never counted as passed. Development code/test/checker errors
may be corrected and rerun under the current authorization while old failures are
preserved. Scientific changes, a real GPU invocation, expanded resources or edits
to historical materials require separate authorization. PR27 remains draft and
unmerged.
