# SparseRTDETR VisDrone

Repository: Six-of-Sheep/SparseRTDETR_VisDrone

This repository is the independent P3 research workspace for real coarse-to-
fine sparse real-time detection of small VisDrone objects. The P2 YOLO
Sidecar Query route is scientifically closed and is retained only as a
byte-bound legacy reference.

P3 will study sparse Query, attention, or decoder execution inside a unified
set-prediction framework. The official RT-DETRv2 PyTorch implementation is
fixed and vendored from the upstream commit recorded in
`manifests/rtdetrv2_upstream.json`. The vendored source is an unmodified
reference snapshot; project changes belong in wrapper and patch layers.

No environment has been installed and no model has been imported or
executed. There are currently no baseline, accuracy, true sparsity, or speed
conclusions.

The historical test split was observed during P1 and influenced the post-hoc
5% top-k exploratory proposal. It is not an independent unseen confirmatory
set and must not be used for development, tuning, or model selection.

Data, weights, runs, checkpoints, predictions, logs, and process evidence are
outside Git. The repository contains research contracts, the fixed upstream
vendor snapshot, a minimal code skeleton, and legacy P2 references.
