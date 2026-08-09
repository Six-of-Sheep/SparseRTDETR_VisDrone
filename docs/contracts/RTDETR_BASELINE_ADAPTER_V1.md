# RT-DETRv2 R18 VisDrone Baseline Adapter V1

This document freezes the interface boundary for
`rtdetrv2_r18_visdrone_baseline_v1`. It does not freeze a formal training
configuration and it does not make an accuracy, sparsity, or speed claim.

## Frozen identity

The implementation is the vendored RT-DETRv2 PyTorch implementation at
upstream commit `1c8ac3f7ba84f14bd5651ab7b1b70d69a5f55f47`. The model is
PResNet-18 plus HybridEncoder and RTDETRTransformerv2 with three decoder
layers, 300 queries, 256 hidden channels, three feature levels at strides
8/16/32, and 4/4/4 deformable sampling points. The engineering input-size
candidate is 640x640. PResNet pretrained loading is disabled and checkpoint
is null.

The standard dense postprocess is sigmoid followed by one global top-k of 300
query/class scores. It has no threshold and no NMS. Global top-k is a result
selection rule, not sparse computation: it does not prove that decoder query
work, cross-attention, FFN work, or sampling work was skipped.

The parameter identity is explicitly separated from the upstream default:

```text
upstream R18 default (80 classes): 20,184,464 parameters
VisDrone baseline (10 classes):    20,094,584 parameters
class-dependent delta:                 89,880 parameters
```

These counts are for random initialization with no checkpoint. The delta is
exactly `70 * 1,284`: four class-related linear heads contribute `4 * 257`
per extra class and the denoising class embedding contributes 256. The
reduction is only the change from the 80-class output space to the 10-class
VisDrone output space. It is not pruning, sparsification, speedup, or a
research contribution. Future papers must report `20,094,584` for this
10-class baseline and must not count the `89,880` reduction as sparse gain.

## Category boundary

The certified COCO artifacts use VisDrone scored category IDs 1..10. The
model criterion uses labels 0..9. The adapter applies this exact bijection:

```text
COCO/VisDrone category 1..10 -> model label 0..9
model label 0..9 -> COCO/VisDrone category 1..10
```

The mapping is strict. Missing values, null, strings, bools, negative values,
0 on the COCO side, 11, and out-of-range values fail closed. An empty target
is valid. Vendor `mscoco_category2label` and `mscoco_label2category` are not
used because their sparse COCO IDs do not define the VisDrone contract.

The dataset adapter lets vendor `CocoDetection` load the certified COCO
artifact, maps labels immediately after vendor `load_item`, and only then
invokes transforms. Thus labels reaching a future criterion are 0..9. The
postprocessor delegates vendor sigmoid, box conversion, and top-k math with
`remap_mscoco_category=false`; it maps only final labels back to 1..10. Boxes,
scores, order, and count are not changed. Final rows bind to stable image IDs
and the existing `data_protocol.evaluation.Detection` schema.

## Data roles and evaluators

Only explicit `train_core` and `development` roles can be resolved. The
confirmatory role is sealed and forbidden. The test role is forbidden. Runtime
data roots are explicit parameters, are never written into the portable JSON,
and are rejected when they contain an independent `test` path segment. The
single runtime image resolver maps the certified logical COCO paths
`train/images/<official-name>.jpg` and `val/images/<official-name>.jpg` to the
fixed official raw directories `VisDrone2019-DET-train/images` and
`VisDrone2019-DET-val/images`, respectively. It requires an existing regular
file, rejects symlinks and path escape, and does not enumerate a parent
directory or discover sibling splits. The dataset adapter and Smoke pre-hash
check use this same resolver; the COCO `file_name` remains unchanged and raw
absolute paths never enter portable entry evidence.

The primary evaluator is `visdrone_official_style_v1`. It is not implemented
by this adapter. `coco_secondary_vendor_v1` is diagnostic only and is not
equivalent to the primary evaluator. No evaluator metrics are accessed here.

The repository has historical metadata enumeration of the project test split,
so it is not independent unseen test data. This stage does not access test
files, predictions, a test server, or confirmatory predictions/metrics.

## Scope and next proof obligation

This is a dense baseline adapter only. There is currently no precision,
sparsity, or speed conclusion. Formal training configuration, model selection,
and speed measurement remain unfrozen.

Any later real sparse work must prove that computation is actually skipped at
the claimed boundary, including decoder Query work, cross-attention, FFN, or
the relevant sampling operator. A top-k output count alone is insufficient.
