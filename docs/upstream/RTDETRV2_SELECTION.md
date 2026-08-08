# RT-DETRv2 Selection

## Fixed source

P3 uses the original author's official `lyuwenyu/RT-DETR` repository and the
fixed `rtdetrv2_pytorch` snapshot at commit
`1c8ac3f7ba84f14bd5651ab7b1b70d69a5f55f47`. The source is Apache-2.0 and is
recorded in `manifests/rtdetrv2_upstream.json` and the vendor `UPSTREAM.md`.

`rtdetrv2_pytorch` was selected because it preserves a clear set-prediction
decoder boundary: 300 Queries by default, stride 8/16/32 multi-scale
features, self-attention, deformable cross-attention, and FFN. Hungarian
matching and auxiliary/denoising losses remain in the same reference path.
The standard DETR postprocessor uses sigmoid/top-k rather than NMS; the
repository also contains an explicit NMS postprocessor, so the exact path
must be frozen before any experiment.

## Initial candidate

RT-DETRv2-S/R18 is the first smoke candidate because it is the smallest
official model that retains the complete multi-scale decoder. This is an
engineering candidate, not a formal model selection or training configuration.
The official table reports 20M parameters and 60G FLOPs at 640 input, with
T4 TensorRT FP16 measurements; neither 1024 input nor RTX 4090 D performance
has been validated here.

## Integration boundary

The fixed source is vendored so AutoDL migration, review, and paper release
have one repository identity. Future changes must use a wrapper or explicit
patch layer and must not edit vendor files. No environment, baseline, data
adapter, accuracy, true sparsity, or speed conclusion is ready.
