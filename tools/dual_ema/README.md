# tools/dual_ema — round delta for the single-trajectory dual-EMA recording experiment

This directory is the **only** delta of the round `r18-dual-ema-20261031-001`
over the frozen slice execution chain.  The release assembly copies
`r18-detail-arms-20261031-001/execution-release-002/source` (which is the frozen
`r18-slice-finetune-20261018-001/execution-release-003/source` plus the r10
detail-arms overlay), then overlays the files below, each byte-verified against
its git HEAD blob.

## Overlay map

| this file | release path |
|---|---|
| `slice_finetune.py` | `src/sparse_rtdetr/protocol_r18/slice_finetune.py` |
| `slice_ft_job.py` | `src/sparse_rtdetr/protocol_r18/slice_ft_job.py` |
| `slice_ft_execution.py` | `src/sparse_rtdetr/protocol_r18/slice_ft_execution.py` |
| `slice_ft_progress.py` | `src/sparse_rtdetr/protocol_r18/slice_ft_progress.py` |
| `slice_ft_cpu_accept.py` | `tools/slice_ft_cpu_accept.py` |
| `dual_ema_check.py` | `tools/dual_ema_check.py` |
| `dual_ema_verify.py` | `tools/dual_ema_verify.py` |
| `ledger_build.py` | `tools/ledger_build.py` |
| `r18_dual_ema_v1.json` | `configs/research/r18_dual_ema_v1.json` |

`r18_detail_arms_v1.json` stays in the tree untouched; the new plan file is the
frozen science plan of this round (`PLAN_SHA` in `slice_ft_execution.py`).

## Scientific delta (nothing else changes)

* one arm `D0`: the original native G0 trajectory (no installed module, original
  parent-EMA load, fresh AdamW, the original 12-pass / 3,648-update LR clock,
  the original data order and augmentation stream);
* two EMA records built from the same model state by the same `R0EMA` class:
  `ema` (counter 0, the original G0 record) and `ema_alt` (counter 36,480);
  both are stepped once per completed update, immediately after the original
  `Engine.train_window` EMA step, on the same post-optimizer-step model state;
* the second record consumes no training RNG draw: every update compares the
  CPU and CUDA generator states around the extra step and the engine refuses to
  continue if either changed (`FineTuneEngine._step_alt_ema`);
* neither record feeds back into training; the online model, optimizer,
  criterion, LR clock and RNG stream are bitwise those of the original G0 path
  (proved by `dual_ema_check.py` against a single-EMA reference engine);
* complete checkpoints carry `ema` and `ema_alt` modules **and** their counters,
  with a separate layout identity for the second record, and restore validates
  `ema_alt.updates == completed_updates + 36480`;
* declared checkpoint cadence `[0, 6, 12]` (endpoint always saved), fixed before
  writing to stay inside the 8 GiB server-new budget; every pass still emits its
  wire events and nothing written is deleted afterwards;
* endpoint readout: both records evaluated once each (`infer-D0-A-emaR`,
  `infer-D0-A-emaC`), original A protocol; `evaluation` reports
  `A_C_minus_R`, `A_R_minus_originalG0`, `A_C_minus_originalG0`,
  `A_R_minus_parent`, `A_C_minus_parent` with `quality = RECORD_ONLY`.

## Checks before the release freeze

* `tools/dual_ema_check.py` — targeted CPU checks (both records' math and
  counters, no-RNG-draw proof, unchanged online trajectory, exact save/restore);
* `tools/slice_ft_cpu_accept.py` — the real CLI end to end under real
  supervisors and both evaluators on the CPU fixture (success chain, duplicate
  launch refusal, worker-exception stop, four mutated wire records rejected),
  now also asserting the declared checkpoint cadence and the dual-EMA payload;
* `tools/dual_ema_verify.py` — after the GPU run, the read-only reproduction
  comparison of the new retained passes against the original G0 passes
  (`accept-D0/C2-direct`, `FT000`, `FT006`, `FT012`): model, optimizer,
  per-parameter step ledger, EMA-R module and counter, engine clock and
  behaviour digest.  A mismatch is preserved as `REPRODUCTION_DIFF.json` and
  forbids any attribution of the observed differences to the EMA counter.

## Budget (declared)

2.0 GPU-hours, 24 h wall, 8 GiB server-new bytes, 1 MiB return; 3,648 training
updates + 3 acceptance updates + 7,612 batch forwards planned; VRAM recorded,
not quota-limited.  Engineering exception or overrun after launch = STOP, no
automatic retry, no deletion of material to free space.
