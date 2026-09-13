# baseline-v2b runtime engineering closure

This extension connects the seeded model factory, actual train_core loader,
logical-window engine, runtime identity and checkpoint recovery. It provides no
automatic training/evaluation loop. CPU tests may update real R18 models using
synthetic JPEG/COCO fixtures. The real-data pre-run tool only previews bounded
train_core inputs; it never calls model forward or optimizer step.

GPU smoke, formal training, 896/960 experiments and confirmatory/test access are
not authorized by any configuration or report produced here. Development image
or annotation loading is absent. Two existing train_core/development manifest
metadata files support old CPU regressions without loading their images.

## Data and reproducibility

The loader requires explicitly named train_core_coco.json and
train_core_manifest.json, complete-file SHA256 values and an approved image
root. The COCO IDs, stable IDs, dimensions, class mapping and manifest membership
are checked before constructing a dataset. It never discovers raw images by
directory traversal. Each selected JPEG must match its manifest size and SHA256
before decoding. Other data roles and path redirections to them are rejected.

The epoch shuffle uses a dedicated CPU generator. Each sample's augmentation
seed depends on the run seed, logged epoch, position in the common sample order
and stable image ID. The actual vendor transforms run while Python, NumPy and
CPU torch RNG are saved/restored. Model/DN RNG is therefore independent of
worker scheduling and prefetch. The vendor receives logged_epoch-1; internal
augmentation stop 117 means logged epoch 118.

A full logical batch is collated before physical splitting. Worker count and
prefetch settings are recorded as runtime observations, while the semantic
binding covers sources, library versions, data hashes, transforms, seed, size
and logical batch. This allows a verified worker0-to-worker2 restore without
changing sample order or augmented tensors. Worker processes use spawn, with
persistent_workers=false and pin_memory=false in this engineering version.

The frozen NumPy MKL build and PyTorch libgomp conflict under spawn with the
INTEL threading layer. Process entry points set MKL_THREADING_LAYER=GNU before
imports; the loader rejects an incompatible process instead of silently
changing a loaded library. OMP/MKL thread settings are recorded as well.

The saved sampler cursor counts only acknowledged successful optimizer updates,
not yielded or prefetched items. Pending batches are validated before the model
changes, then acknowledged after the complete engine update succeeds. Prefetched
items are regenerated from their sample keys on restore. A partially failed
update poisons the session: BN or optimizer changes are not rolled back. Restore
requires fresh components and a closed destination loader iterator.

This is a new v2b augmentation RNG protocol. It reproduces paired v2b inputs and
recovery; it does not reconstruct the historical v2a worker RNG trajectory.
Comparing v2b with v2a therefore cannot isolate accumulation alone.

## Device, precision and checkpoint

The model and pretrained backbone are initialized and byte-identified on CPU.
All registered state and unregistered positional caches move to a prepared
runtime before AdamW construction. Our CPU preparation/checkpoint/augmentation
helpers avoid CUDA API calls. Stock torch optimizer/worker internals may query
CUDA availability or queue lazy generator seeding;
the whole CPU verification claim is no CUDA context or kernels, with devices
hidden and lazy initialization blocked.

CUDA preparation requires a fresh native capability for the exact run binding,
CUDA_VISIBLE_DEVICES equal to the full GPU UUID, one visible device and explicit
cuda:0. The actual runtime records device topology, torch/CUDA versions, backend
flags and RNG shape. CPU and CUDA default generators are seeded explicitly.
The prepared device's seed must match the model/data seed. UUID/device remapping
on restore is rejected. CUDA backend control-flow tests use CPU mocks and do not
certify real CUDA execution.

BF16 uses model/criterion autocast without GradScaler. Loss normalization remains
sum_i max(GT_i,1)/max(sum_i GT_i,1) times the already-weighted vendor losses.
One complete logical window advances AdamW, clipping, EMA and warmup once;
the scheduler advances at a completed epoch after warmup. Live BN updates once
per physical forward, and DN remains native per physical microbatch. Equal
logical batch size does not make BN, DN or floating-point reductions equivalent.
The existing CPU-only contiguous BN backward workaround remains in place.

Checkpoint schema2 stores model, raw/EMA buffers and caches, AdamW, engine,
scheduler, warmup, CPU/CUDA RNG, explicit generators, runtime identity and the
acknowledged loader cursor. Validation precedes live mutation; RNG is restored
after state loading. Unexpected mutation/load failures invalidate the engine.
Schema1 CPU checkpoints remain readable without a sampler. Unbound old CUDA
checkpoints are rejected.

Two real-model integration findings are covered by tests: vendor EMA averages
floating geometry buffers, so their saved rounding must be preserved rather
than treated as immutable raw-model geometry; MultiStepLR.milestones must be
restored as a Counter, including repeated milestones and closed-form behavior.

V2BTrainingSession binds these components together. It rejects source/config,
initial-state, input geometry, seed, logical batch, augmentation-stop, device or
cursor mismatches. Saving cannot hide an engine update that the loader did not
acknowledge. A fresh native GPU probe is required before each CUDA window.

## Native hardware gate

The new collector reads native NVSMI XML, actual loaded NVML library evidence,
driver versions, GPU UUID, memory, utilization, temperatures, power/clock fields,
process identities, CPU sensors, RAPL and kernel journal entries/cursors. Raw
command output, executable/library/source hashes, PID/start time, boot ID, UTC
and monotonic interval bind the observation to the same source and process.
Supplied JSON, stale probes, source drift, boot/PID changes or cursor gaps cannot
grant admission. Native evidence provides integrity, not cryptographic proof
against a malicious operator.

Current graphics clock, maximum hardware clock and application clock are not a
readback of a locked upper clock. The installed backend has no verified getter;
the collector records GRAPHICS_CLOCK_CAP_UNVERIFIED and blocks admission. It
does not use a successful historical run as a substitute. No clock/power setter
or reset operation is executed.

The passive observation found the expected RTX4090D and driver580.178.04,
PL1=125W/PL2=150W and no compute occupant. It also found the desktop Xorg graphics
process and two BERT log messages about one skipped firmware record at boot.
The graphics process is a recorded allowlist condition, not evidence of hardware
failure. Boot BERT is unresolved history, not a newly observed training crash.
Together with missing cap readback, these findings leave the GPU gate blocked.
An existing observation cannot be reused as a later launch capability.

Immediate stop conditions include new Xid, hardware/MCE/BERT/fatal errors,
boot change, driver/library disagreement, wrong UUID, unknown or excessive
caps/RAPL, unexpected compute occupants, sensor loss, GPU>=80C or CPU>=90C,
nonfinite model/loss/gradient state, a failed update, or inconsistent clocks and
checkpoint/data evidence. These conservative engineering thresholds are
explicit policy, not a proof of permanent hardware stability.

## Disposition of the original nineteen old tests

Sixteen of the original nineteen cases are applicable: fifteen fake-port
tests and one direct source/pretrained-authority identity check. Three
historical runtime checks remain explicit engineering-scope exclusions.

The fifteen fake-port tests run in a temporary source checkout containing exact
frozen source, the existing pretrained authority and explicitly synthetic empty
annotations. The production binder is not patched; canonical paths and regular
file checks still run. No historical successful training target is fabricated.
Their original test source bytes remain unchanged.

| Old module | Exact test function | Disposition |
| --- | --- | --- |
| repository_contract | RepositoryContractTests.test_contract_passes | Historical runtime artifacts required; explicit engineering-scope skip |
| rtdetr_baseline_adapter | test_repository_contract_checker_passes_after_allowlist_update | Historical runtime artifacts required; explicit engineering-scope skip |
| training_v2a_contract | test_v1_and_t7a_identities_are_not_changed | Source config and existing pretrained weight/manifest identities verified; passed |
| training_v2a_contract | test_repository_checker_accepts_checkout_with_completed_v1_targets | Completed historical training targets required; explicit engineering-scope skip |
| training_v2a_runtime | test_policy_reads_binding_and_binds_all_input_closures | Synthetic fixture restored; passed |
| training_v2a_runtime | test_checkpoint_records_contract_authority_groups_and_no_scaler | Synthetic fixture restored; passed |
| training_v2a_runtime | test_fake_runtime_closes_process_parent_data_environment_and_progress | Synthetic fixture restored; passed |
| training_v2a_runtime | test_fake_runtime_rejects_secondary_evaluator_and_batch_adaptation | Synthetic fixture restored; passed |
| training_v2a_production | test_policy_binds_t7c_and_uses_independent_target_names | Synthetic fixture restored; passed |
| training_v2a_production | test_cpu_fake_call_order_step_ema_primary_and_terminal | Synthetic fixture restored; passed |
| training_v2a_production | test_tensor_loss_is_recorded_as_epoch_mean | Synthetic fixture restored; passed |
| training_v2a_production | test_cpu_fake_duplicate_and_production_port_injection_are_rejected | Synthetic fixture restored; passed |
| training_v2a_production | test_detached_descriptor_context_digest_and_cross_field_mutations | Synthetic fixture restored; passed |
| training_v2a_production | test_production_entry_rejects_unconsumed_artifact_drift_and_existing_target | Synthetic fixture restored; passed |
| training_v2a_production | test_bf16_context_has_no_scaler_construction_or_source_surface | Synthetic fixture restored; passed |
| training_v2a_pilot | test_cpu_fake_records_ten_epochs_loss_families_and_encoder_state | Synthetic fixture restored; passed |
| training_v2a_pilot | test_cpu_fake_delegates_corrected_weighted_loss_and_rejects_injection | Synthetic fixture restored; passed |
| training_v2a_pilot | test_cpu_fake_rejects_replay_and_missing_encoder_evidence | Synthetic fixture restored; passed |
| training_v2a_pilot | test_fake_result_is_canonical_and_round_trips | Synthetic fixture restored; passed |

An independent audit reviewed all 76 broader data-closure exclusions and
restored one existing source-only branch:
test_training_runtime_plan_binding_returns_complete_detached_observed_identity.
In engineering scope, this unchanged frozen test runs against a temporary
source tree with no artifacts subtree. It proves binder refusal without
complete R3, unchanged
filesystem paths and readable canonical pre-CUDA plan configuration. It does
not exercise or certify the complete binder's successful identity path.

The remaining 75 explicitly enumerated function families require the complete
25-file historical R3 closure: 11 contract, 21 evidence, 20 adapter, 7 entry and
16 process-launcher families. The production binder enumerates and hashes all
R3 members, including sealed partitions; its limited metadata capture does not
limit those reads. Success cases need a valid full baseline, and the existing
mutation cases either first construct that baseline or can otherwise pass on
an unrelated missing-file rejection. Such a rejection is never counted as
their intended negative-test proof. tests/conftest.py lists every family.
These families and the three original historical checks are skipped only with
the explicit v2b engineering scope option. The default historical checker
remains strict. Source-only validation never asserts historical runtime
admission.

The scope option controls the explicitly listed skip markers and the one
source-only ROOT redirection above. Without it, the frozen runtime test keeps
its original choice between source-only refusal and successful full binding
based on the actual checkout. Selected synthetic fake-port fixtures remain
isolated by autouse fixtures in both default and engineering-scope runs.
Neither fixture isolation certifies historical runtime outputs.

Additional legacy fake launcher and file-identity tests use the same isolated
fixture. One frozen smoke argv-routing test replaces both directory creators;
its empty process directory is explicitly pre-existing synthetic fixture state.
That test does not prove actual launcher directory creation. No receipt,
completion or historical successful output is fabricated for it.

## Pre-run and next experiment contract

The bounded pre-run tool constructs both actual640 CPU model configurations,
verifies the same seeded/pretrained raw initial state, and previews the same
logical batches with workers0,2,4. It records full source/data/initialization
bindings, augmented input receipts, actual worker environment, zero-update
engine/loader state and the common planned sample-order hashes through epoch30.
An optional native hardware snapshot is passive; CPU_PRERUN_PASS always leaves
gpu_ready=false. Artifacts are written exclusively to a new run directory and
read back; previous reports are not overwritten.

After separate GPU authorization and successful hardware admission, first run
paired640x16x1 and640x8x2 short smoke: four identical logical windows per arm,
save after two and replay the remaining two with a fresh runtime. Bind the same
source tree, logical inputs, seed, pretrained SHA, sample order, worker settings
and numerical policies. Compare peak allocated/reserved/native device memory,
loss families/denominators, BN/DN behavior, gradients, update clocks, raw/EMA
state, checkpoint integrity and restored RNG/cursor. The runtime JSON freezes
same-arm replay tolerances and requires diagnosis rather than widening them.
Between-arm loss or bitwise equality is not a smoke acceptance criterion under
live BN and native DN. OOM does not trigger automatic batch changes or retries.

If smoke passes and the scientific run is separately authorized, epoch10 is an
early checkpoint; epoch30 is the primary paired comparison endpoint. Both arms
retain the same nominal120-epoch schedule and augmentation-stop117 recipe.
Epoch10 alone cannot establish batch equivalence or performance. A bound
development metric adapter/protocol remains required before AP comparison.
Single-seed results are exploratory; no AP or AP_small result is produced here.
