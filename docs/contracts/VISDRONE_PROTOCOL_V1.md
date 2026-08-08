# VisDrone Protocol V1

This is a P3-native, data-free protocol contract. It freezes the audited
train/val identity, parser semantics, category mapping, split planner and the
two deliberately separate evaluator schemas. It does not generate real COCO
JSON, a production split manifest, predictions, or metrics.

## Source and lineage

The official train inventory has 6471 image/annotation pairs and 208 sequence
keys. The official val inventory has 548 pairs and 76 sequence keys. Twenty-
four sequence keys occur on both sides and exact image-content overlap is zero.
The four inventory hashes and the audit counts are recorded in
`configs/visdrone_protocol_v1.json`.

Every future annotation record must retain the split, normalized relative
source path, physical line number, all eight normalized source fields, raw
line SHA-256, stable image ID, stable annotation ID, converted bbox/area,
category mapping, disposition, and one reason code. Stable IDs use
`sha256(namespace + NUL + canonical_json(payload))`; Python `hash()`, mtime,
filesystem order and absolute machine paths are forbidden.

## Parser and geometry

The semantic fields are `left, top, width, height, score, category,
truncation, occlusion`. Exactly eight fields are required. One final empty
field caused by a single trailing comma is accepted; missing fields, extra
non-empty fields, non-finite numbers, and non-integer categories are rejected.
The raw physical line bytes are hashed before parsing. Geometry is
`xyxy=[left,top,left+width,top+height]` and `area=width*height`; no clipping is
performed. Non-positive formal boxes remain in lineage and are filtered from
matching/evaluation. Exact normalized duplicate rows retain the first row and
mark later rows `FILTERED_DUPLICATE_EXACT_GT`; equal bboxes with other
attributes are not duplicates.

## Categories and ignore semantics

Raw categories 1..10 map explicitly to training IDs 0..9 and COCO IDs 1..10.
There are ten model classes and no explicit background category. Category 0 is
a spatial ignore region. A ground-truth score of zero is ignored. Category 11
(`others`) is not one of the scored official categories and does not enter
matching. Formal category records with positive area are the only records
entering normal matching. Official ignore matching uses
intersection-over-detection area, allows repeated ignore matches, and is
considered after normal GT matching. A future converter must preserve these
semantics instead of treating a plain COCO conversion as equivalent.

## Evaluator separation

The primary schema carries explicit ignore-region and ignored-state fields and
is reserved for official-style evaluation. The secondary schema carries
ordinary COCO `iscrowd` records and is diagnostic only. The secondary vendor
evaluator is not mathematically equivalent to the primary evaluator because
it cannot recover category-0 spatial ignore behavior from ordinary COCO
records. NMS is false in both contracts; changing thresholds, maxDets or NMS
to force agreement is forbidden.

## Split planner

Development is official val. Confirmatory candidates come only from official
train. An atomic group starts with a filename sequence key, then merges any
groups connected by equal image SHA-256. An official sequence metadata table,
if supplied later, must agree exactly. No atomic group is split. Groups with
an identity gap or abnormal flag are excluded.

The fixed planner uses seed `20260808`, salt `P3-confirmatory-v1`, target
`round(6471*0.10)=647` images, lexicographically normalized group IDs, and
ordering by `SHA256(salt + NUL + seed + NUL + group_id)`. It chooses the prefix
closest to the target, then prefers not-over-target, fewer images, shorter
prefix, and smaller canonical group-list SHA. Group/content disjointness,
development-sequence exclusion, class proportions, and COCO-small proportion
must pass within five percentage points; otherwise planning fails closed. No
alternate seed or manual selection is allowed.

The resulting membership may be persisted only with
`selection_allowed=false`, `metrics_access_allowed=false`, and
`single_final_access_only=true`. Confirmatory metrics cannot be accessed
before the formal freeze. The test split remains disabled.

## Phase boundary

Synthetic CPU tests exercise all contracts. This phase does not access test,
construct a model, create a Dataset/DataLoader, download weights, run
forward/backward, train, evaluate, measure speed, or create CUDA objects.
