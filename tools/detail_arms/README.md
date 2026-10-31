# r18-detail-arms execution-round delta (r10 execution segment, 2026-10-31)

Three-arm fixed fine-tune (G0 native / G1 capacity control / G2 sub-stride
detail) executed as a new single-use execution release over the frozen
whole-slice chain.  Science plan: `r18_detail_arms_v1.json` (this directory).
Single exploration record: `delivery/research/r18-explore-20261031-001/EXPLORATION_R1.md`.

## Base bytes

Base: `execution-release-003/source` of `r18-slice-finetune-20261018-001`
(source manifest SHA `b989e9f45e50b26a30c36f34132d960b60932c93da8cad7fccaea961d81d4090`,
commit `57ae4d1990ef059818825e6d8813435b04046505`, branch
`codex/r18-slice-finetune`).  Every base file stays byte-identical; the delta
below is the complete change set (each file SHA recorded in the release
manifest built from this directory plus the base tree).

## Delta map

| file | release path | kind | reason |
|---|---|---|---|
| `r18_detail_arms_v1.json` | `configs/research/r18_detail_arms_v1.json` | new | three-arm plan: arms + install map, unchanged training math (12 passes x 304, warmup200/cosine, EMA decay .9999, sampler epoch109..120), A-only inference, pre-registered `A_G2_minus_G1` gates, budget limits |
| `slice_finetune.py` | `src/sparse_rtdetr/protocol_r18/slice_finetune.py` | patch | `ARMS=('G0','G1','G2')`, `ARM_INSTALL`, whole-image-only `view_id` (no draws), versioned plan validation |
| `slice_ft_job.py` | `src/sparse_rtdetr/protocol_r18/slice_ft_job.py` | patch | install hook (after strict parent load, before optimizer/EMA; `assert_installed` on live+EMA); acceptance `arm_probe`; evaluation generalized to 3 arms x A-only with pre-registered vs descriptive comparisons; per-stage cost/budget accounting |
| `slice_ft_execution.py` | `src/sparse_rtdetr/protocol_r18/slice_ft_execution.py` | patch | TASK, PLAN_SHA (plan file SHA `c699eeececd347f480f044d21113f9bd4142948aef5c3f8edf19cbdd184ca938`), 10-phase DAG (accept/formal x G0-G2, infer-A x G0-G2, evaluation), arm parsing, launch counters (9 acceptance updates, 10944 formal updates) |
| `slice_ft_progress.py` | `src/sparse_rtdetr/protocol_r18/slice_ft_progress.py` | patch | protocol id, arm set G0/G1/G2, A-only evaluation order (6 records), terminal completeness at 6 |
| `slice_ft_cpu_accept.py` | `tools/slice_ft_cpu_accept.py` | patch | three-arm fixture gate; worker-exception fault moves to `accept-G1`; evaluation asserts 6 score calls |
| `detail_path.py` | `src/sparse_rtdetr/protocol_r18/detail_path.py` | new | the mechanism (below) |
| `test_detail_path.py` | `tools/explore/test_detail_path.py` | new | 8 targeted CPU unit tests |
| `detail_smoke.py` | `tools/explore/detail_smoke.py` | new | real-Engine synthetic smoke incl. nonzero-gate save/restore/replay |
| `probe_layouts.py` | `tools/explore/probe_layouts.py` | new | frozen checkpoint layout-invisibility probe |
| `ledger_build.py` | delivery copy (run server-side post-chain) | new | accurate execution ledger from published stage artifacts |

## Method description (corrected this round)

- Exact insertion point: inside the decoder's `_get_encoder_input`, level-0
  branch, AFTER the native 1x1 `input_proj[0]` projection and BEFORE the
  per-level flatten/concat into the encoder token sequence:
  `proj_feats[0] += gate * conv1x1(pixel_unshuffle(stride4))`.
  Not inside the backbone (tap-only, key-preserving wrapper), not a new
  pyramid level/head/anchor, queries untouched.
- `pixel_unshuffle(2)` + 1x1 linear on the stride-4 tensor is algebraically a
  single 2x2 stride-2 convolution (the space-to-depth + linear family already
  used by the cited precedents).  NO originality is claimed for the operator
  and no new literature sweep was performed.  Declared differences: the tap
  point (stride-4 stage output the native model discards), the insertion
  interface, the zero-initialized gate (bitwise identity at step 0), and the
  capacity-matched control arm.

## Gradient definition (fixed mechanism of this round)

Branch input stop-gradient in BOTH modes: `detail` uses the stride-4 capture
(detached at capture), `control` uses `proj_feats[0].detach()` (same tensor,
stop-gradient copy).  The main path trains normally; only `gate`+`conv` train;
a forward-time guard enforces the detach and `LAST_SOURCE_INFO` records the
evidence.  This is a declared scientific definition, NOT a save-state
workaround.  The acceptance probe verifies, against a parent-loaded native
twin on the real configuration: zero-gate outputs bitwise identical, zero-gate
loss bitwise identical, every shared-parameter gradient bitwise identical
(installation must not change the main-path gradient definition), gate
gradient nonzero, conv gradient exactly zero at gate 0, and the detach guard
evidence.

## RNG policy

Installation consumes NO global RNG draws (explicit seeded generator, xavier
bounds); the chain resets Python/NumPy/torchCPU/CUDA seed0 after each
independent construction; view selection draws nothing, so G0/G1/G2 share the
identical data order and augmentation stream.  The acceptance probe pins
`torch.manual_seed` per call; the direct-vs-replay exactness stays anchored on
checkpoint-carried RNG + full state, so probe draws cannot perturb it.

## Checks

`test_detail_path.py` (8 tests), `probe_layouts.py` (6 layout diffs),
`detail_smoke.py` per mode, and the release's own
`tools/slice_ft_cpu_accept.py` end-to-end fixture gate (real supervisors,
duplicate-launch refusal, fault injection, mutated-wire rejection, 6 dual-
evaluator score calls).
