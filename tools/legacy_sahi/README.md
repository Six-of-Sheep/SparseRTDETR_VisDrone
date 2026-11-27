# Whole-image and four-tile experiment drivers

These scripts belong to the earlier RT-DETRv2 comparison of whole-image detection and four overlapping crops. The sampling-density paper has a separate [code and revision guide](../../docs/paper-code.md).

| Purpose | Entry |
| --- | --- |
| Run a paired seed protocol | [Supervisor](new_chain.py) and [per-seed entry](new_entry.py) |
| Aggregate the three seeds | [Paired readout](aggregate_seeds.py) |
| Prepare confirmation inputs | [Preparation](confirmation_prepare.py) |
| Inspect the supplemental A/B/C workflow | [Supervisor](supplemental_job.py) and [shell entry](supplemental_launch.sh) |

On `main`, [inference helpers](../../infer_seed.py) and [slice geometry](../../slice_core.py) keep the module names used by later evaluations. The [tests](../../tests/) include the synthetic geometry and readout checks.

### Reproducing an earlier release

The archived contracts record paths relative to the release that produced each result. Use that release's original paths and file hashes. The [seed replication import](https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone/tree/a4cdcb1bbaec967c2b62200d0b0eca51fb2a3fa7) contains the 317 files listed in the [source manifest](../../manifests/imports/r18-seed12-release-002.json). The [pre-reorganization tree](https://github.com/Six-of-Sheep/SparseRTDETR_VisDrone/tree/b9fe222a93f715d7a39a0041f28fe5420fcea0c6) also shows the later confirmation and supplemental scripts at their former locations. Actual run identities are recorded in the [experiment index](../../manifests/experiments.json).

From a full clone of `main`, these commands verify the historical Git objects and run the current synthetic checks.

```bash
python -B tools/repo_manage.py verify-import
python -B -m unittest tests.test_slice_core tests.test_seed_readout tests.test_legacy_sahi_layout
```
