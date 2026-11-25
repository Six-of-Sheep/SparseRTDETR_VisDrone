# SparseRTDETR · Sampling density for small-object detection

How should a detector spend a fixed number of pixels on a UAV image? This repository contains the RT-DETRv2-R18 implementations used to compare uniform resizing, a fixed spatial prior, image-dependent sampling, and a larger uniform detection canvas. VisDrone provides the development comparison. UAVDT provides a separate evaluation of frozen models without fine-tuning.

**[Paper code guide](docs/paper-code.md)** · [Browse the sampling code](https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone/tree/3c491df1a4765d9826f69e247cb295ecb00dc04c/src/sparse_rtdetr/research) · [Browse the experiment drivers](https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone/tree/3c491df1a4765d9826f69e247cb295ecb00dc04c/tools/r28)

| To inspect | Start here |
| --- | --- |
| How W distributes pixels and maps boxes back | [Allocation and geometry](https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone/blob/59e367de86cb55dbe1fe6606962d4f0bf65458e3/src/sparse_rtdetr/research/r28_warp.py) and [detector integration](https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone/blob/59e367de86cb55dbe1fe6606962d4f0bf65458e3/src/sparse_rtdetr/research/r28_resample.py) |
| How the FOVEA-style controls differ | [F mapping](https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone/blob/e78e8cbe310084da5523466833086796c91ea554/src/sparse_rtdetr/research/r28_fovea.py) and [geometry-selected control (F_geo)](https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone/blob/e1c904f21ad08607871db401be90e2eff47c0465/src/sparse_rtdetr/research/r28_fovea_geo.py) |
| Which code produced each comparison | [Arms, seeds, diagnostics, and evaluation](docs/paper-code.md#experiments-and-source-revisions) |

The [paper/resampling branch](https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone/tree/paper/resampling) opens the main research source snapshot. For citations and reproduction, use the commit links in the guide. It also identifies analyses implemented on separate commits and lists the inputs needed to reproduce the reported scores.

中文导航见[论文代码索引](docs/paper-code.md)。各对照按论文中的名称排列，可从对应行打开当次使用的源码。

## Earlier experiments and repository maintenance

The earlier whole-image and four-tile comparison has its own [results and protocol](docs/research/abc_results_20261019/README.md), [frozen manifest](manifests/frozen_abc_results_20261019.json), and [archived project overview](https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone/blob/458d645f70839cda3d1ff35d67bf3ff812e68bd1/README.md). The [experiment index](docs/EXPERIMENTS.md), [source import record](docs/SOURCE_IMPORT.md), and [development guide](docs/DEVELOPMENT.md) document the original repository and its checks.
