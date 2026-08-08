# SparseRTDETR VisDrone

Repository: Six-of-Sheep/SparseRTDETR_VisDrone

This repository is the independent P3 research workspace for real coarse-to-
fine sparse real-time detection of small VisDrone objects. The P2 YOLO
Sidecar Query route is scientifically closed and is retained only as a
byte-bound legacy reference.

P3 will study sparse Query, attention, or decoder execution inside a unified
set-prediction framework. No RT-DETR upstream implementation has been chosen,
and no baseline has been reproduced. There are currently no accuracy, true
sparsity, or speed conclusions.

The historical test split was observed during P1 and influenced the post-hoc
5% top-k exploratory proposal. It is not an independent unseen confirmatory
set and must not be used for development, tuning, or model selection.

Data, weights, runs, checkpoints, predictions, logs, and process evidence are
outside Git. This initial repository contains only research contracts, a
minimal code skeleton, and legacy P2 references.
