# R27 post-L0 design check (TASK_CONDITIONAL_COMPUTE_DESIGN_001)

Pure standard-library CPU scripts. No torch, no CUDA, no model, no dataset.
They do not import anything from this repository.

- `check_backbone_shapes.py --json` : PResNet18(d) C3-C5 block inventory at
  1152x2016, 4x4 macro-region mapping, exact executed residual MACs per mask
  (halo included), saving ceilings from measured stage timings.
- `check_set_risk.py --json` : synthetic set-risk counterexamples, selector
  enumeration, pairwise-predictor limit, teacher-cost model. Reads
  `check_backbone_shapes.result.json`, so run the first script before it.
- `make_identity.py` : hashes the task inputs and the source files read. It
  expects to sit in the task's delivery directory
  (`post-l0-decision-001/`), next to the review receipt; it is kept here for
  the record and is not runnable from this path without that layout.

Every risk value in `check_set_risk.py` is synthetic. Nothing here is
evidence about VisDrone, AP, or a trained detector. Outcome of the task:
`not_justified` (see DESIGN_VERDICT_001.md in the delivery directory).
