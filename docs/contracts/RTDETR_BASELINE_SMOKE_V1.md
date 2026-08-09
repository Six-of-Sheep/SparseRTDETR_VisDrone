# RT-DETRv2 R18 VisDrone Baseline Smoke V1

Smoke V1 is a bounded inference harness for the certified `train_core` conversion
and the frozen ten-class RT-DETRv2 R18 baseline. It is a wiring check, not a
training, evaluation, model-selection, speed, or accuracy experiment.

## Scientific boundary

The harness verifies that the two explicitly identified records in the certified
`train_core_manifest.json` can enter the adapter, that category IDs `1..10` map
to model labels `0..9`, that a ten-class R18 model can execute one eval/no-grad
forward on exactly one visible CUDA device, and that vendor sigmoid/global-top-k
postprocessing returns categories `1..10` with finite boxes and scores. NMS is
disabled. The harness does not access the confirmatory or test split, calculate
metrics, construct a criterion or optimizer, load weights, resume, train, or
measure speed.

## Frozen execution

The portable JSON contract in
`configs/baseline/rtdetrv2_r18_visdrone_smoke_v1.json` is strict. It freezes two
images, batch size `2`, workers `0`, one `next()` call, one model forward, one
postprocessor call, input size `[640, 640]`, seed `0`, ten classes, `300`
queries, no checkpoint, and no CPU fallback. The future real entrypoint accepts
an explicit data root, rejects a path with an independent `test` component, and
reads only the two manifest-selected image paths after verifying their SHA-256
values.

The committed config and child entry artifact are portable: they contain only
relative repository identities, certified R3 hashes, and the two frozen stable
image IDs/relative paths. Runtime absolute repository, output, process, and
data-root paths are restricted to the nonportable launcher process evidence.

`contract-check` runs with `CUDA_VISIBLE_DEVICES=''` and
`PYTHONNOUSERSITE=1`. It validates source/config/R3 metadata without importing
the model framework, creating CUDA objects, constructing a dataset or model,
reading images, creating output directories, or requesting a network.

`smoke` requires explicit authorization via
`P3_RTDETR_BASELINE_SMOKE_AUTHORIZED=1`, exactly `CUDA_VISIBLE_DEVICES=0`, one
available CUDA device, and `PYTHONNOUSERSITE=1`. The launcher owns the
process-evidence directory. Before spawning the child it atomically writes a
single-use `handoff_prepared.json`. The child must consume that handoff before
importing torch or creating entry output, then atomically writes the only
child-owned process file, `handoff_receipt.json`. The launcher cross-checks the
receipt against the child PID/PPID, argv SHA, nonce, paths, and prepared-handoff
SHA. The launcher first resolves and validates the explicit `train_core` data
root as an absolute, existing, non-symlink directory with no independent
`test` component. That canonical path is bound into the prepared handoff,
receipt, process invocation, and process completion evidence. The child must
resolve `P3_SMOKE_DATA_ROOT` before receipt or output creation and match it to
the prepared binding; after consumption, the real entry receives the validated
receipt binding rather than rereading the mutable environment. The child owns
the entry output directory and creates it only after the launcher has
established the process evidence.

The real entrypoint resolves each certified logical image path through the
baseline runtime image adapter before CUDA validation or torch import, then
performs the frozen image SHA check on the returned official raw path. Dataset
image loading uses the same resolver. `train_core` maps to
`VisDrone2019-DET-train/images` and `development` maps to
`VisDrone2019-DET-val/images`; no split discovery, fallback, symlink, or
portable-artifact absolute path is permitted.

## Evidence

The child writes `config.json` atomically before any model, dataset, dataloader,
image, or CUDA object. It then records invocation, source and portable data binding,
runtime identity, model and tensor audits, call counters, input selection,
postprocessing, and RNG restoration. Successful entry evidence must contain a
complete artifact inventory that binds every other entry file. A child exit code
of zero is insufficient: the launcher also requires `completion.json` with all
frozen smoke counters and flags, a complete inventory, and the non-reportable
qualification fields, including strict integer counters and complete
inference-only flags. Failure evidence preserves the original exception and
partial inventories.

Tensor logical SHA-256 values describe the particular smoke invocation. They are
diagnostic evidence for that run, not cross-GPU acceptance thresholds.

## CPU validation

The repository tests use injected synthetic tensors, model, loader, and
postprocessor components under the R2 Python environment with CUDA hidden. They
exercise schema drift, manifest drift, one-batch limits, failure ordering,
finite-value gates, RNG restoration, evidence binding, launcher ownership,
nonce handoff, process-group signal handling, and repository source policy.
These tests never launch the real smoke path. Runtime data-root tests also
cover canonical-path validation, pre-receipt environment drift, prepared and
receipt binding drift, process-invocation drift, and post-consumption
environment changes. Absolute runtime roots are present only in nonportable
process evidence; entry JSON remains portable.
