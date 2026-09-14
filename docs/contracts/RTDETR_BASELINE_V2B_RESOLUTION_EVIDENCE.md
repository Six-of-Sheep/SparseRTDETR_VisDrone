# Baseline-v2b development resolution evidence

This phase evaluates the existing seed-0 epoch-30 EMA checkpoints in four fixed
cells, in this order: train640/eval640, train640/eval896, train896/eval640,
train896/eval896. It does not select checkpoints, change training, access held-out
splits, set/reset GPU clocks, or launch a trainable sparse model.

The existing monitored worker admission is extended under the explicit new user
authorization only for development evaluation (1,800 seconds). All native
identity, Xorg, hardware error, temperature, clock and observation-gap limits stay
unchanged. Every independent process needs fresh admission and final guardian
clearance after process exit. Any failure stops the sequence, preserving evidence.

## Model and inference identity

The fixed checkpoint files contain 9,120 optimizer/EMA updates, 18,240 physical
microsteps, completed epoch 30, seed 0, physical batch 8 and accumulation 2.
The inference worker loads EMA state strictly on CPU, authenticates the complete
checkpoint and its internal binding, and preserves all learned parameters and
non-geometry buffers exactly.

The decoder's persistent floating anchors were themselves updated by EMA.
Canonical anchors are replayed through the identical two-operation EMA formula
for all 9,120 updates; their bytes must equal the saved anchors at the source
training size before an alternate evaluation geometry is admitted. The common
rule is applied to all four cells. Native diagonal geometry must remain exactly
unchanged. Plain encoder position caches are explicitly rebuilt and inspected.

Inference is FP32 without autocast, batch 4, all 548 development images in ascending
COCO image-ID order. It preserves the original 300-query, 10-class sigmoid
flattened top-300 postprocessor without NMS or score thresholds. The sidecar stores
raw logits, normalized boxes, actual runtime sigmoid scores and actual selected
flat indices, in addition to the unmodified final predictions. A per-batch exact
check binds query provenance to the postprocessor output.

This inference load does not restore training RNG or claim checkpoint
continuation. CPU/CUDA RNG and model state must remain unchanged by evaluation.

## Primary evaluation and annotation provenance

Use the existing full primary_evaluator.evaluate_primary_v1 entry and its exact
authoritative binding for point results. The committed evaluator JSON intentionally
retains independent_audit_pass=false and training_gate_open=false. A later,
separate external certification record in the training contract binds its audited
source and configuration. Those original flags must not be silently changed or
mistaken for absence of that subsequent certification. Each campaign also binds
the explicit certification-review evidence and its stated provenance limits.

The historical development adapter used converted formal COCO GT only. The
development lineage contains 1,378 raw category-0 regions with score zero; an
earlier score-zero disposition caused the converted ignore sidecar to be empty.
The new adapter reconstructs complete primary GT from the authenticated raw
fields, preserving every physical row, category-0 spatial ignore regardless of
score, category-11 unscored others, and class-specific ignored labels. It verifies
all 548 manifest memberships, raw row counts, physical line sequence, stable
annotation identities and exact formal-GT joins. Optional raw-byte verification
is reported separately; authenticated lineage is not falsely described as
independent re-reading of original annotation bytes.

Historical converted data and scores remain untouched. Each cell emits both the
uniform complete-GT primary result and the historical formal-GT primary result.
COCO AP, AP-small and AR-small retain their separate converted-formal-GT,
original-pixel area and maxDets=(1,10,100) definitions. Primary AP has no AP-small
output and cannot be substituted for COCO AP-small.

## Paired uncertainty and errors

The method is frozen before the new cross-evaluation and resampling analysis;
historical diagonal scores were already known. The primary bootstrap uses 2,000
paired resamples of 75 development sequence groups, merging groups connected by
identical image content. The same draws are applied to all four cells. A
200-resample image bootstrap is sensitivity analysis only.

Each repeated image is an independent copied instance. Stable equal-score order
is original integer image ID, copy index, then original prediction row. AP is
re-accumulated on the complete sampled dataset, never averaged over per-image AP.
Cached primary and COCO matching must reproduce the full point evaluators before
bootstrap, and CPU adversarial fixtures compare them to literal repeated datasets.

The diagonal total, evaluation-size effects at each fixed training size,
training-size effects at each fixed evaluation size, averaged effects and
interaction are all reported. Only the predeclared diagonal is a qualification
gate; other intervals are descriptive. These intervals condition on fixed trained
weights and development groups. They do not estimate training-seed variation or
certify held-out generalization. Training-size effects also include the existing
size-dependent augmentation sanitation and resulting target/DN changes.

The small-GT ledger separates true positives, maxDets losses, matching competition,
classification, localization, their combination, and absence among saved
candidates at IoU 0.5 and 0.75. A separate prediction ledger accounts for duplicates,
ignored results and background. Occlusion, truncation, neighbors and image-border
contact are orthogonal strata. Raw-query geometry coverage and actual flat-index
membership distinguish saved-output truncation from absence of geometric coverage.
They are upper-bound diagnostics, not counterfactual AP gains. Query rank ties use
actual saved membership; a descriptive stable rank is not a replacement for
the runtime top-k decision.

## Fixed-budget routing and replication qualification

The routing diagnostic uses a 4-by-4 image grid and four selected cells. It compares
a fixed random route, coarse-query uncertainty on small predicted boxes, and a
non-deployable oracle for fine-detected/coarse-missed small GT. Fine windows have
224-pixel cores on an 896 canvas plus 16-pixel halos (256-by-256 crops). The
coarse-plus-crop pixel proxy is reported explicitly. No measured sparse FLOPs,
latency, stitched AP or trainable method is claimed.

The predeclared additional paired seeds are 1 and 2. Qualification requires a
positive uniform-primary diagonal AP difference, positive COCO AP-small difference
whose paired cluster 95% interval has a positive lower endpoint, fewer unmatched
small GT at IoU 0.5, and passed engineering/hardware evidence. Qualification is not
model certification. The evaluation and CPU-analysis tools do not launch training;
the separately frozen replication controller must authenticate this evidence
before any seed workload.

Any authorized replication preserves the scientific training implementation and
30-epoch endpoint, physical batch 8, accumulation 2, pretrained authority and
within-pair initialization/sample-order identities. Epochs 10 and 20 are health
observations. Report all three paired seeds at epoch 30, their differences, mean,
sample standard deviation and range. No best-seed selection or automatic 60/120
epoch, 960, held-out-data or trainable-sparse transition is allowed.
