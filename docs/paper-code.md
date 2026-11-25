# Paper code guide

The study compares ways of mapping a common 1152 × 2016 input onto a 768 × 1344 detection canvas. L0, K, W, F and F_geo use the smaller detection canvas. S15 uses the larger one for both training and evaluation. All dimensions here are height × width. The common input is obtained by resizing the image and can itself be larger than the original.

## Inspect the sampler

The [main source tag](https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone/tree/journal-mva-baseline-20261122) points to `3c491df1a4765d9826f69e247cb295ecb00dc04c`. It contains the main sampler, paired-seed experiment driver and UAVDT evaluator. The [diagnostics tag](https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone/tree/journal-mva-baseline-20261122-r30diag) points to `7c72638658cd374c811d51dba2c2b7ac8447312b` and contains the separate train/evaluation geometry diagnostic.

```bash
git clone --depth 1 --branch journal-mva-baseline-20261122 https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone.git
cd SparseRTDETR_VisDrone
git rev-parse HEAD
```

One small CPU check compares the bounded allocation with an independent numerical reference using synthetic demand vectors. With PyTorch, NumPy and pytest installed, run the following at the source tag. It needs neither images nor trained weights.

```bash
python -m pytest -q tests/test_r28_warp.py::test_waterfill_random_marginals
```

## Experiments and source revisions

Each link opens a file at the revision for that row. The scripts use `FG` for F_geo. “Seed” identifies a paired training run.

For the supplementary three-seed analysis and the W seed replication, use a full clone. These two experiments use separate commits outside the main source tag. In the clone, select the revision for the experiment you want to reproduce.

```bash
git clone https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone.git paper-experiments
cd paper-experiments
git switch --detach b24dec791b72812c8d9a1e6342ff637497bef770  # supplementary analysis
# For the W replication, select f564e2589ef8732f522852d371c3ff026288187a instead.
```

| Paper arm or comparison | What changes | Source at the recorded revision |
| --- | --- | --- |
| L0, seed 0 | Uniform resizing to the smaller detection canvas | [Baseline driver](https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone/blob/fac5f79dcec7f649a18392f29d3109a3db23f998/tools/r27_l0/l0_run.py) |
| W, seed 0 | A per-image demand map drives a bounded allocation along each axis | [Allocation and resampling](https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone/blob/59e367de86cb55dbe1fe6606962d4f0bf65458e3/src/sparse_rtdetr/research/r28_warp.py) · [demand training](https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone/blob/59e367de86cb55dbe1fe6606962d4f0bf65458e3/tools/r28/r28_demand.py) · [box mapping](https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone/blob/59e367de86cb55dbe1fe6606962d4f0bf65458e3/src/sparse_rtdetr/research/r28_resample.py) |
| K, seed 0 | The W allocator receives one fixed spatial prior averaged from training-label demand maps | [Fixed-prior control](https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone/blob/1b4f2a24f7d5a6fbbc953d6ad7e742e3fd2f63ca/tools/r28/r28_followup.py) |
| F and F_geo, seed 0 | Both use the same image-dependent demand with a FOVEA-style mapping. F uses kernel width 4. F_geo uses width 19.03125 selected from training geometry | [F](https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone/blob/e78e8cbe310084da5523466833086796c91ea554/src/sparse_rtdetr/research/r28_fovea.py) · [width calibration](https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone/blob/f376dc973d8333252d4cf50c9ed37d4fd8b9d244/tools/r28/r28_cal_control.py) · [F_geo](https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone/blob/e1c904f21ad08607871db401be90e2eff47c0465/src/sparse_rtdetr/research/r28_fovea_geo.py) |
| S15, seeds 0–2 | Larger uniform canvas throughout training and evaluation | [Seed 0](https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone/blob/61646b37377b178e045c36f8149c9dffecfacf94/tools/scale_baseline/s15_formal_eval.py) · [seeds 1 and 2](https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone/blob/8b338937d129f6d6398b53a9c48ef6cd33544803/tools/scale_baseline/s15_formal_eval.py) |
| L0 and F_geo, seeds 1 and 2 | Within-seed comparisons from the matched starting states | [Replication driver](https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone/blob/fa0a381ba9e106a0381590225c48277f796b9c26/tools/r28/r30_run.py) |
| P and C | Alternative ways to read features in the resampled coordinate system | [Feature readout](https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone/blob/6c656518da9c0ea97121bc15d0a147c8bdef3fef/src/sparse_rtdetr/research/r29_unzoom.py) |
| Training and evaluation geometry, demand replacement, coverage | Diagnostics with frozen detectors | [Diagnostic implementation](https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone/blob/7c72638658cd374c811d51dba2c2b7ac8447312b/tools/r28/r30_diag.py) · [supplementary three-seed analysis](https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone/blob/b24dec791b72812c8d9a1e6342ff637497bef770/tools/r28/th2_phase1.py) |
| External vehicle evaluation | Evaluate frozen VisDrone-trained L0, F_geo and S15 endpoints on UAVDT | [Evaluation driver](https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone/blob/3c491df1a4765d9826f69e247cb295ecb00dc04c/tools/r28/r31_external.py) · [data and scoring routines](https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone/blob/3c491df1a4765d9826f69e247cb295ecb00dc04c/tools/r28/r31_core.py) |
| W follow-up, seeds 1 and 2 | Additional paired comparisons using the W configuration | [Follow-up driver](https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone/blob/f564e2589ef8732f522852d371c3ff026288187a/tools/r28/r32_run.py) · [comparison rules](https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone/blob/f564e2589ef8732f522852d371c3ff026288187a/tools/r28/r32_reading.py) |

F and F_geo adapt the sampling rule from [FOVEA](https://github.com/tchittesh/fovea/tree/c8707bd2e4f10678f6636d24c2915a43b44a59ed) to this detector and demand map. The geometry calibration evaluated 66 widths. The closest tested width still fell outside the tolerance for matching W's small-object magnification. F_geo identifies the stronger control at that width.

## Reproducing a reported score

The formal comparisons use the matching epoch-60 training state for each seed and evaluate the epoch-120 exponential moving average (EMA). Exact runs additionally require the corresponding data split and ignore rules, the frozen demand-network weights or fixed prior where applicable, and each run's configuration and input records. These large or run-specific artifacts are managed separately from this source repository. The recorded training environment used Python 3.10.16, PyTorch 2.4.1, torchvision 0.19.1 and NumPy 1.26.4 with CUDA 12.4.

For VisDrone, the [evaluation code](https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone/blob/e1c904f21ad08607871db401be90e2eff47c0465/tools/r28/r28_eval.py) handles the development comparison with budgets of 600 and 300 detections per image. The [UAVDT protocol](https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone/blob/3c491df1a4765d9826f69e247cb295ecb00dc04c/tools/r28/r31_core.py) uses a single vehicle category and a budget of 300. Compare two arms within the same dataset, budget and seed. For timing, the [five-arm comparison](https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone/blob/e1c904f21ad08607871db401be90e2eff47c0465/tools/r28/r28_geo_control.py) and the [comparison including S15](https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone/blob/59e367de86cb55dbe1fe6606962d4f0bf65458e3/tools/r28/r28_cost.py) use separate timing sessions.
