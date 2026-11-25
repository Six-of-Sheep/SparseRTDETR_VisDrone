# r18-matcher-scale round delta (r12 screening segment, 2026-10-31)

Candidate-1 prototype of the r12 method screen: a **size-relative matching cost**
for the frozen common-protocol R18 one-to-one assignment.  Science plan:
`r18_matcher_scale_v1.json` (this directory).  Single exploration record:
`delivery/research/r18-explore-20261031-001/EXPLORATION_R1.md` section 15.

This round is CPU-only: prototype, control, targeted checks, and the carrier-side
decision diagnostic.  No GPU, no detection scoring, no training, no new data.
The three-arm chain wiring, the release freeze and the CPU gate belong to the
next authorization (the r10 -> section 11 precedent).

## Base bytes

Base: `execution-release-003/source` of `r18-dual-ema-20261031-001`
(source manifest SHA `59b3ea0eca5e3fb6e5a8333e5b9b6c8ae6854b624df3d7077d40e3b6141beae8`,
derived head `1f14e404f5281660bcdc05402ce774c39cab696d`, branch
`claude/r18-explore-20261031`), which is itself the frozen detail-arms
`execution-release-002` source plus the r11 dual-EMA overlay.  Every base file
stays byte-identical; the delta below is the complete change set.

## Delta map

| file | staged path in the release tree | kind | reason |
|---|---|---|---|
| `matcher_scale.py` | `src/sparse_rtdetr/protocol_r18/matcher_scale.py` | new | the mechanism: `ScaledHungarianMatcher` (modes `rel`/`ctrl`), `install_matcher_scale`, `assert_installed`, `declaration`, `cost_terms`, `assignment`, `bind_vendor_names` |
| `patch_frozen.py` | `tools/matcher_scale/patch_frozen.py` | new | applies the two minimal frozen-file patches with asserted anchors (see below) |
| `test_matcher_scale.py` | `tools/matcher_scale/test_matcher_scale.py` | new | targeted CPU unit tests (identity, assignment equivalence with the real vendor matcher, constructed ownership flip, control invariant, gradients, RNG neutrality, geometry-consumer patch landing, declaration stability) |
| `matcher_scale_smoke.py` | `tools/matcher_scale/matcher_scale_smoke.py` | new | real-`Engine` synthetic smoke per mode: two windows, save/restore/replay exactness, cross-mode and native restore refusal, closed-RNG equality with a native engine, matcher compute cost |
| `matcher_probe.py` | `tools/matcher_scale/matcher_probe.py` | new | carrier-side decision diagnostic on the frozen 16 train-side images with the frozen e120 endpoint; evaluates the pre-registered rules F1/F2 |
| `frozen_r18.py` | `tools/matcher_scale/frozen_r18.py` | copy | the r10 read-only frozen-endpoint CPU loader (byte-identical to the branch copy), imported by the probe |
| `r18_matcher_scale_v1.json` | `configs/research/r18_matcher_scale_v1.json` | new | the three-arm plan (arms + install map, dual EMA records, primary endpoint EMA-C, pre-registered gates, budget and storage model) |

## Two minimal patches to frozen files (applied by `patch_frozen.py`)

| file | patch |
|---|---|
| `src/sparse_rtdetr/protocol_r18/criterion.py` | the recipe gains `'matching': matching_declaration(self)`; the native declaration is the default, so an uninstalled criterion's recipe is unchanged |
| `src/sparse_rtdetr/protocol_r18/checkpoint.py` | `behavior_identity` gains `"matching": declared_matching(engine)`, derived from the engine's declared install mode; it refuses a criterion whose declaration differs from the engine's mode, and with no declared mode the expected dict is byte-identical to the frozen one |

## Mechanism (declared as implemented)

Inside the frozen criterion the one-to-one assignment solves
`C = w_bbox * L1(cxcywh) + w_class * focal_class + w_giou * (-GIoU)` in absolute
normalized-canvas units.  For a small instance a fixed absolute error is a large
fraction of the object but contributes the same absolute cost as for a large
instance, so in dense competition the objective allocates tokens by absolute
error magnitude.  The prototype replaces the localization term only:

- `rel` (method): the L1 term is evaluated in the GT's own size units,
  `|dcx|/s_g + |dcy|/s_g + |dw|/w_g + |dh|/h_g`, `s_g = sqrt(w_g h_g)`;
- `ctrl` (control): the same arithmetic with `s_g, w_g, h_g` replaced by their
  batch means, i.e. the operator without the per-instance differentiation;
- `off` (identity): the native expression, byte-for-byte (used by the tests).

A matcher call whose target list carries no GT box at all is passed through with
the native expression (the normalization has no columns to act on); empty windows
are a legitimate input of this chain.

Nothing else changes: the class and GIoU terms, the one-to-one structure, the
loss path, the model, the optimizer, inference and the checkpoint schema.  The
matcher still runs under `torch.no_grad()` and consumes no RNG draw.

Nearest precedents and the declared differences: scale-normalized localization
metrics exist in anchor-assignment work (NWD family) and scale-aware token
matching exists in recent DETR variants; the differences declared here are that
the normalization is applied to the *assignment cost* only (the closed rsmall
item was a *loss* reweighting), that the decision that changes is token
ownership, and that a control with the same operator but no per-instance
differentiation exists.  No originality is claimed for the operator.

## Pre-registered carrier-side screening rules (evaluated by `matcher_probe.py`)

- **F1**: the fraction of small GTs (raw area <= 1024) whose owning token changes
  under `rel` must be >= 5%; otherwise the mechanism has no decision difference
  on this carrier and the round stops before any GPU work.
- **F2**: the median change of the paired IoU among those changed small GTs must
  be strictly positive; otherwise the change is not in the declared direction and
  the round stops.

## Checks run this round (frozen interpreter, CUDA hidden, staged source tree)

`test_matcher_scale.py` (unit tests), `matcher_scale_smoke.py` per mode,
`matcher_probe.py` on the frozen 16 images, plus the staging manifest that pins
every file SHA.  Evidence is returned under
`delivery/research/r18-matcher-scale-20261031-001/`.

## Not done this round (declared scope boundary)

No change to `slice_finetune.py` / `slice_ft_job.py` / `slice_ft_execution.py` /
`slice_ft_progress.py` / `slice_ft_cpu_accept.py` (the 13-phase three-arm chain),
no release freeze, no CPU gate, no GPU admission, no detection scoring, no
training, no new data, no holdout access.
