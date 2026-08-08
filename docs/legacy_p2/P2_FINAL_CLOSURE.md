# P2 Legacy Research Closure

## Scope

This document freezes the P2 semantic-probe source provenance and scientific
conclusions. It is an archive record, not an authorization for training,
validation, test access, speed measurement, promotion, or P3 implementation.
The legacy source, runs, checkpoints, dumps, logs, dataset, and failed-run
evidence remain in place and are not rewritten by this closure.

## Frozen validation evidence

The common-contract Base reference is AP_small `0.17575682354003425`.

- Confidence-v1 selected epoch 12: Merged AP_small `0.1753013370094127`.
- Confidence-v2 selected epoch 39: Merged AP_small `0.17568721377525598`.
- Confidence-v3 epoch 50: 164400 Query candidates, Merged AP_small
  `0.1708990459622116`, and 626 suppressed Base true positives.
- Semantic probe epoch 15: 100513 Query candidates, Query-only AP_small
  `0.06483205700365657`, Merged AP_small `0.12688295744539288`, and 5709
  suppressed Base true positives.

None of the four generations produced a net-positive Merged AP_small result.
The semantic argmax failure count was 3396, greater than 3100, so that
hypothesis was falsified.

The results are highly consistent with independently trained confidence
spaces competing destructively inside shared class-aware NMS. This is an
evidence-consistent explanation, not an isolated proof of a unique causal
mechanism. The P2 route therefore stops further development of the YOLO
Sidecar Query approach.

## Efficiency and future boundary

The tile16+halo2 route has only an approximately 11.09% theoretical compute
reduction upper bound before dynamic overhead. Python sparse tile and
Multi-Slot Query speed routes are NO-GO. P3 is a separate repository effort
for RT-DETR-style unified set prediction and real operator skipping; no
upstream implementation has been selected in this archive.

## Test protocol provenance

The historical protocol evidence is recorded in:

- `coarse_sparse_det/runs/p1_t6a_failure_diagnosis/protocol_history.md`
- `coarse_sparse_det/runs/p1_t6a_failure_diagnosis/phase2_candidate_protocol.json`

P1 observed test before proposing the exploratory 5% top-k budget. That budget
is post-hoc/exploratory evidence, not confirmation on an unseen test set.
Later P2-T11/P2-T12 records with `test_used=false` only show that those later
runs did not use test; they do not restore test's status as an independent
unseen confirmatory set. Future test access must not tune thresholds, budgets,
checkpoints, or model selection.

## Environment limitation

The legacy visdrone environment is represented only by the safe package
inventories in this archive. The original installation included nonportable
origins, so exact replay of the original installation is not claimed. The
P3 project must establish an independent environment and must not treat the
legacy visdrone environment as its formal RT-DETR environment.

## Status

P2 legacy artifacts are preserved. P2 source provenance is frozen by the
containing Git commit. P3 initialization remains a separate, future task.
