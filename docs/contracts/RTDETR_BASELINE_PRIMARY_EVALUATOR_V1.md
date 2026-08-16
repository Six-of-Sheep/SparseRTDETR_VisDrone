# RT-DETRv2 VisDrone Primary Evaluator V1

`visdrone_official_style_v1` is the development-only primary evaluator for
the VisDrone DET protocol.  Its implementation is a pure NumPy and standard
library port of the frozen offline authority snapshot identified by commit
`005445782213e20cb91bc50a597db3dd949e749a` and tree
`038b9e68c6e9a93a64662a4d7a39be2cd2c0654e`.

The public evaluator consumes `PrimaryEvaluatorInputV2` and never opens a
dataset, image, model, checkpoint, or result file.  Image dimensions are part
of the input because the official ignored-region procedure rasterizes and
clips spatial regions.  Detections use repository `xyxy` coordinates and are
converted to official `xywh` with `w=x2-x1` and `h=y2-y1`, without a `+1`
term.

## Frozen semantics

The evaluator uses exact float64 IoU thresholds `0.50:0.05:0.95`,
`maxDets=(1,10,100,500)`, no NMS, stable per-image descending-score order,
and applies each maxDets prefix before category filtering.  Matlab
half-away-from-zero rounding, one-based clipping, inclusive ignored-region
rasterization, standard IoU, ignored-GT intersection-over-detection-area,
inclusive threshold comparison, standard-GT preference, and the official
`max(1,numel(gtMatch))` recall denominator are preserved.

The seven emitted metrics are `AP`, `AP50`, `AP75`, `AR1`, `AR10`, `AR100`,
and `AR500` in percentage units.  AP comes from the maxDets=500 pass and uses
the monotonic precision envelope and continuous recall integration from
`VOCap.m`.  No AP-small metric is emitted.  The official `evalClass`
aggregation is intentionally preserved: a category contributes once per
image in which it has ground truth, so repeated image/category occurrences
weight the aggregate rather than a unique-category mean.

Category 0 is a spatial ignored region, category 11 is the unscored
`others` state, and scored categories are 1 through 10.  Ordinary GT and
detections with ignored-region area fraction greater than or equal to `0.5`
are removed.  Ignored GT may absorb multiple detections; standard GT matches
remain one-to-one.

## Evidence and gates

`primary_evaluator_contract_binding()` verifies the committed strict JSON
contract and source-identity manifest through one descriptor-rooted boundary.
The repository root must be an absolute, non-symlink directory.  Every path
component is opened with `O_DIRECTORY|O_NOFOLLOW`, and the final object is
checked before opening as a same-device, same-owner, single-link regular file.
Reads use a complete EINTR-safe descriptor loop and compare before/after
device, inode, mode, link-count, owner, size, mtime, and ctime identities.  A
single stable root descriptor covers both the config and manifest reads.

The evaluator accepts only the exact builtin-dict authoritative binding with
these ten keys: `config`, `authority_manifest`,
`config_raw_size_bytes`, `config_raw_sha256`,
`config_canonical_size_bytes`, `config_canonical_sha256`,
`authority_manifest_raw_size_bytes`, `authority_manifest_raw_sha256`,
`authority_manifest_canonical_size_bytes`, and
`authority_manifest_canonical_sha256`.  Bare config dictionaries and partial
bindings are rejected.  Contract, manifest, binding, input, and result
structures reject container and scalar subclasses, non-finite floats, invalid
Git blob OIDs, and invalid SHA-256 strings before semantic comparison.

`evaluate_primary_v1()` returns an immutable detached result containing the
canonical input identity, both contract identities, evaluated counts, repeated
evaluation-class occurrences, per-class by IoU AP, per-class by IoU and maxDets
AR, match counts, seven metrics, and a recomputed canonical result SHA-256.
The result validator first checks the exact result schema, table dimensions,
metric bindings, ranges, and count cross-fields, then recomputes the full
result and rejects stale, repacked, or hash-only mutations.  A structurally
separate deterministic CPU oracle covers 100 valid multi-image cases and
compares metrics, AP/AR tables, and match-count tables exactly under the
frozen algorithm.

This evaluator is development-only.  Confirmatory and test access are
forbidden, the secondary COCO evaluator cannot certify or select a model, and
the training gate remains closed until a later independent audit certifies this
implementation.
