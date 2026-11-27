# SparseRTDETR · Sampling density for small-object detection

How should a detector spend a fixed number of pixels on a UAV image? This study compares how RT-DETRv2-R18 samples a common 1152 × 2016 input onto a 768 × 1344 detection canvas (height × width). Each image goes through the detector once. A larger uniform canvas provides a reference for spending more pixels.

**[Paper code guide](docs/paper-code.md)** · [Sampling code](https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone/tree/3c491df1a4765d9826f69e247cb295ecb00dc04c/src/sparse_rtdetr/research) · [Experiment drivers](https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone/tree/3c491df1a4765d9826f69e247cb295ecb00dc04c/tools/r28) · [CPU quick start](#cpu-quick-start)

The default branch holds this guide and the shared evaluation utilities. Open the [paper source snapshot](https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone/tree/paper/resampling) to browse the detector and samplers together. The implementation links below point to the commits used for each comparison.

中文读者可按论文中的组名、训练种子和评价数据集定位到每次实验使用的提交。[论文代码索引](docs/paper-code.md#experiments-and-source-revisions)

## Sampling methods and controls

| Arm | How the image is sampled | Code at the recorded revision |
| --- | --- | --- |
| L0 | Uniformly downsample to 768 × 1344. | [Uniform baseline](https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone/blob/fac5f79dcec7f649a18392f29d3109a3db23f998/tools/r27_l0/l0_run.py) |
| K | Use the same bounded allocation as W with one spatial prior averaged from training annotations. | [Fixed-prior control](https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone/blob/1b4f2a24f7d5a6fbbc953d6ad7e742e3fd2f63ca/tools/r28/r28_followup.py) |
| W | Predict demand for each image and distribute samples along each axis within fixed magnification bounds. | [Allocation](https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone/blob/59e367de86cb55dbe1fe6606962d4f0bf65458e3/src/sparse_rtdetr/research/r28_warp.py) · [Detector and box mapping](https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone/blob/59e367de86cb55dbe1fe6606962d4f0bf65458e3/src/sparse_rtdetr/research/r28_resample.py) |
| F and F_geo | Apply a FOVEA-style mapping to the same per-image demand as W. F uses a kernel width of 4. F_geo uses 19.03125, selected from training-set geometry. | [F](https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone/blob/e78e8cbe310084da5523466833086796c91ea554/src/sparse_rtdetr/research/r28_fovea.py) · [F_geo](https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone/blob/e1c904f21ad08607871db401be90e2eff47c0465/src/sparse_rtdetr/research/r28_fovea_geo.py) · [Upstream FOVEA](https://github.com/tchittesh/fovea/tree/c8707bd2e4f10678f6636d24c2915a43b44a59ed) |
| S15 | Keep the uniform 1152 × 2016 canvas for continued training and evaluation. | [Large-canvas reference](https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone/blob/61646b37377b178e045c36f8149c9dffecfacf94/tools/scale_baseline/s15_formal_eval.py) |

## Results

**VisDrone.** All three detector training seeds were evaluated on the same 548 development images. The image-dependent samplers share one frozen demand network trained with seed 0. Each cell gives the paired difference in average precision (AP), in percentage points, for seeds **0 / 1 / 2** at 600 candidates per image.

| Comparison | Small-object AP | Small-object AP at IoU 0.75 |
| --- | --- | --- |
| F_geo − L0 | +2.03 / +1.75 / +1.63 | +2.33 / +1.90 / +2.06 |
| W − L0 | +1.77 / +1.60 / +1.54 | +2.55 / +2.09 / +2.46 |
| W − F_geo | −0.27 / −0.15 / −0.09 | +0.22 / +0.19 / +0.40 |

Both mappings improved small-object AP over uniform sampling. Their ordering depended on the metric, and F_geo and W delivered different magnifications. W also reduced medium-object AP relative to L0 by 0.69, 1.08 and 0.52 points. [F_geo replication](https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone/pull/42) · [W replication](https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone/pull/48)

**UAVDT.** We evaluated the frozen VisDrone checkpoints without fine-tuning, using a single vehicle category and 300 candidates per frame. The small-object AP differences for F_geo versus L0 were +0.148, +0.164 and −0.169 points. The larger S15 canvas also produced both gains and losses in small-object AP across seeds. [External evaluation and per-seed results](https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone/pull/43)

## CPU quick start

With PyTorch, NumPy and pytest installed, these commands check W's sampling allocation against an independent numerical reference on synthetic inputs. This example runs on CPU without a dataset or checkpoint.

```bash
git clone --depth 1 --branch journal-mva-baseline-20261122 https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone.git paper-sampler
cd paper-sampler
python -m pytest -q tests/test_r28_warp.py::test_waterfill_random_marginals
```

The source tag resolves to `3c491df1a4765d9826f69e247cb295ecb00dc04c`. For a reported score, select the commit for the **specific arm and seed**. The W replication and supplementary analyses have separate revisions. Reproducing scores also requires the matching model states, data splits, ignore rules and run inputs, which are managed separately. [Experiment revisions](docs/paper-code.md#experiments-and-source-revisions) · [Reproduction requirements](docs/paper-code.md#reproducing-a-reported-score)

## Earlier experiments and repository maintenance

The [legacy SAHI drivers](tools/legacy_sahi/) and [CPU tests](tests/) have their own directories. The evaluation code imports [image loading and score conversion](infer_seed.py) and [slice geometry](slice_core.py) from the repository root.

The earlier whole-image and four-tile comparison has its own [results and protocol](docs/research/abc_results_20261019/README.md), [frozen manifest](manifests/frozen_abc_results_20261019.json), and [archived project overview](https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone/blob/458d645f70839cda3d1ff35d67bf3ff812e68bd1/README.md). The [experiment index](docs/EXPERIMENTS.md), [source import record](docs/SOURCE_IMPORT.md), and [development guide](docs/DEVELOPMENT.md) document the original repository and its checks.
