# Evaluation Contract

The primary contract is VisDrone official-style evaluation: IoU thresholds
`0.50:0.05:0.95`, `maxDets=(1,10,100,500)`, spatial ignore regions, and the
official VOC-style AP integration. It does not use NMS and does not claim an
AP-small metric.

The secondary contract is an explicitly diagnostic, unmodified vendor COCO
evaluator: `maxDets=(1,10,100)`, COCO area ranges, and AP/AP50/AP75 plus
AP-small/medium/large. Ordinary COCO `iscrowd` records do not reproduce
VisDrone category-0 intersection-over-detection ignore matching, so the two
schemas and metric names must never be mixed. `P3_VENDOR_EVALUATION_EQUIVALENCE_STATUS=NOT_EQUIVALENT`.

This phase defines input/output schemas only. It does not run an evaluator,
generate predictions, access confirmatory metrics, or select a model.
`MODEL_SELECTION_READY=false` and `ACCURACY_ACCEPTANCE_READY=false`.
