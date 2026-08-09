# VisDrone Production Converter V1

This phase implements the production converter and its evidence writer but
does not execute `run`. Only the explicit future `train` and `val` source
directories are addressable. There is no generic dataset walk and no test
split argument.

## CLI gate

The data-free command is:

```text
CUDA_VISIBLE_DEVICES='' visdrone-converter contract-check
```

It reads only the committed Protocol V1 configuration, imports no torch or
torchvision, creates no output directory, and makes no network request.

The future conversion command must explicitly provide `--data-root`,
`--output-dir`, `--protocol-config`, and `--split train,val`, and must set
`P3_VISDRONE_PRODUCTION_CONVERSION_AUTHORIZED=1`. The output directory must
not exist. `overwrite`, `resume`, automatic output selection, test paths, and
relative `..` escapes are rejected. The absolute data root is never written
to a portable artifact or error manifest.

## IDs and source lineage

The frozen SHA-256 `stable_image_id` and `stable_annotation_id` remain the
scientific lineage identity. COCO integer IDs are a separate namespace:
images are globally numbered from 1 after sorting train then val by normalized
relative path; annotations are numbered from 1 separately within each COCO
JSON after sorting `(coco_image_id, physical_line_number, stable_annotation_id)`.
Same-split duplicate image content is allowed and remains connected for split
planning; cross-split duplicate content is rejected.

## Outputs

The writer produces deterministic canonical JSON/JSONL artifacts for source
identity, raw manifests, connected sequence groups, split plan, train-core,
development and sealed confirmatory lineage, category contract, the two COCO
inputs, ignore-region sidecars, conversion audit and round-trip audit. Raw
JSONL manifests are chunked so each file is below 1 MiB. Category-0 spatial
ignore records are never placed in ordinary COCO GT; score-zero, category-11,
duplicate and non-positive records remain only in lineage with a reason code.

Confirmatory membership is written for train exclusion only, with
`selection_allowed=false`, `metrics_access_allowed=false`,
`single_final_access_only=true`, and no metrics. Development is exactly
official val. Test access is recorded as zero.

All successful files are written atomically. `artifact_inventory.json` binds
the successful artifacts but excludes itself and `completion.json` to avoid a
hash cycle. Completion binds the config and inventory SHA but does not contain
its own SHA. A failed run retains its directory and writes `error.json`,
`partial_inventory.json`, and a failed `completion.json`; it never writes a
fake successful completion and cannot resume.

## Process evidence

The independent process_launcher owns a separate, preconditioned evidence
directory and never creates the child output directory. It records the exact
child argv, child Python, source module path, PYTHONPATH, hidden CUDA state,
flushed console SHA, timing, an ASCII exit-code file, entry completion and
artifact inventory references, and a process inventory. Child contract failure
and generic failure retain their actual return codes and still produce process
completion plus partial evidence. Missing entry completion is fail-closed and
cannot be reported as success. Process completion excludes itself from its
inventory to avoid a hash cycle.
