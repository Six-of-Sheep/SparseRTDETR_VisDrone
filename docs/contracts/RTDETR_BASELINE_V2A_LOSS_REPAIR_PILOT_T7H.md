# RT-DETRv2 baseline-v2a T7H loss-repair diagnostic pilot

## Boundary

T7H is a detached contract and CPU/fake verification layer for a future,
development-only ten-epoch diagnostic pilot.  Its only scientific question is
whether the corrected aggregation of all already-preweighted vendor losses
restores the early training/development trajectory lost by the defective v2a
run.  It is not a model-selection run and it cannot certify the final
baseline.

The contract is identified by
`rtdetrv2_r18_visdrone_baseline_v2a_t7h_loss_repair_pilot_r1`.  It inherits the
T7B baseline-v2a policy and binds the published T7G commit, its production
source identity, the certified `_weighted_loss` source, and the existing T7A
PResNet-18 authority.  The parent policy is validated in full before the
detached ten-epoch overlay is applied.  The allowlist contains only the
effective epoch horizon, required complete epochs, and terminal checkpoint
label.  All other parent values are byte-for-value inherited.

## Pilot overlay

- The run is fresh, exactly once, and uses the unique `t7h_loss_repair_pilot_r1`
  run, stage, artifact, and session namespaces.
- Epochs, required complete epochs, checkpoint epochs, and development
  evaluation epochs are exactly `1..10`.
- The original augmentation stop value remains `117`, outside the pilot
  horizon.  The original optimizer, warmup, LR milestone, bf16/no-GradScaler,
  EMA, batch, topology, and evaluator policies remain bound to baseline-v2a.
- Development AP is recorded only as a comparison observation.  There is no
  numerical acceptance threshold, checkpoint selection, model selection, or
  readiness claim.

## Loss and evidence contract

The single loss owner is the already-certified T7G
`training_v2a_production._weighted_loss` function.  Its 21 already-preweighted
terms must be aggregated exactly once without reading `weight_dict` or
filtering by key name: 3 base terms, 6 auxiliary terms, 6 DN terms, and 6
encoder terms.  Every encoder auxiliary parameter has to show a finite,
nonzero gradient and optimizer state after the first update.

Every fake epoch records the total loss and base, auxiliary, DN, and encoder
family subtotals so the total can be recomputed.  Checkpoints and EMA-backed
development evaluations are recorded for every epoch.  The CPU/fake adapter
delegates the epoch, checkpoint, evaluator, and already-certified loss
mechanics to the T7G production implementation; it does not duplicate them.

## Forbidden operations

This stage never creates or consumes owner authorization, reads confirmatory or
test data, accesses real data, initializes CUDA, starts GPU or tmux work,
launches training, invokes a real evaluator, resumes or retries a run, selects
a model/seed/hyperparameter, downloads a network artifact, or enters the T7H
independent-audit phase.  Importing the module is inert and performs no file,
Git, subprocess, network, data, CUDA, or Torch operation.

The next action after a successful implementation, CPU/fake verification,
archive verification, and publication is a separately supplied T7H independent
read-only audit attachment.  This contract authorizes no launch.
