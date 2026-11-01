# r16: opportunity statistics, chain entry and the CUDA adaptation

Round scope (record section 20): keep every r15 production-path evidence item, then in one
batch (1) fix device placement, per-image selected sets, illegal-owner checks and the
evidence wording without touching eligibility thresholds, loss weights or the quality
formula; (2) run a PRE-FROZEN read-only scan over 32 consecutive train_core images for
pairing and control opportunity; (3) register "三臂机制比较依据不足" instead of starting
the three arms mechanically if no choice space is observed, and never judge the method
invalid from that; (4) deliver the CUDA production-entry adaptation, the A1 state restore
evidence on the real checkpoint chain and the update budget, then request one bounded GPU
acceptance and short training.  CPU only; no GPU is authorized by this round.

## Item 1: what was fixed (nothing scientific changed)

    device         every mask, index, reduction and audit tensor is created on the pool
                   logits' device, so the term runs unchanged on CUDA; its only mutable
                   state is a CPU torch.Generator (no device RNG is consumed)
    selected set   each image records its selected top-k token indices
    owner checks   owner tokens must be in range, unique per instance, and (production
                   path) among the selected tokens; a partner is never a selected token
                   and never an owner; a violation raises
    wording        the record states what the r15 run did and did not verify (per-image
                   selected sets were not recorded then; the CPU-only placement; the
                   opportunity numbers are evidence of chance, not of benefit)

Thresholds (B/q_floor/q_margin/tau/alpha), the loss weight (λ = 1.0), the position, the
1/N normalization and the quality formula are unchanged.

## Item 2: the pre-frozen 32-image opportunity scan

    set        chain order, source epoch 109, windows 0 and 1, positions 0..31 (32 distinct
               stable image ids, frozen in scan_opportunity_32.py before the scan ran)
    seeds      derived inside the chain from seed 0, the epoch, the position and the stable
               image id, so freezing the rows freezes the augmentation
    parent     the same certified e120 EMA, loaded with the chain's own
               initialize_from_parent; the state digest is checked before and after
    read-only  eval forwards only: no backward, no optimizer, no EMA, no checkpoint, no
               parameter/buffer/running-statistic change, training RNG unchanged
    reports    total pairs, instances, multi-partner (choice space) instances, images
               involved, GT size-band distribution, eligible-count histogram, and the
               A1 versus A2 partner choice on the SAME forward
    decision   pre-registered: choice_space_instances > 0 -> the A1/A2 comparison has a
               basis (magnitude reported, no post-hoc threshold); == 0 -> register
               三臂机制比较依据不足 and do not start the three arms mechanically

## Item 4: CUDA production entry, checkpoint state and budget

    adapter    selection_rank_arm.install/declaration/assert_installed/state/load; the arm
               declaration is non-vacuous, so a state saved under one arm cannot be
               restored under another
    patches    criterion (recipe declaration, matched indices, loss_rank hook), worker and
               slice_ft_job loss counts (13/22 vs 12/21), slice_ft_job construct install
               before the move to CUDA, slice_finetune RANKING_INSTALL/capture/restore,
               checkpoint behaviour identity
    evidence   verify_chain_entry.py: chain save/restore at zero updates, term state exact
               (generator state and draw counter), draws continue identically, refusals in
               both directions; one checkpoint file, declared in the record
    budget     the request restates the pre-registered three-arm envelope (3,648 updates
               per arm, ≈5.03 GPU-h, ≤12 h, storage item by item, return ≤1 MiB) plus a
               bounded acceptance (3 updates) before it

## Files

    selection_rank.py           the term (device placement, per-image selected sets, owner
                                checks, gradient audit, r14-suite adapters)
    selection_rank_arm.py       the arm adapter used by the chain patches
    patch_frozen.py             the full anchored patch set (r15 + r16) for release-003
    scan_opportunity_32.py      the pre-frozen 32-image read-only opportunity scan
    verify_chain_entry.py       the A1 state check through the real checkpoint chain
    verify_production_path.py   the r15 production-path harness (unchanged, kept as evidence)
    test_selection_rank.py      the r14 synthetic suite (unchanged; runs against v3)

## Declared limits

The scan measures opportunity, not benefit: no AP, coverage or selection-rate claim, and a
zero result is registered as an insufficient comparison basis, not as a verdict on the
method.  The chain-entry check runs at zero updates on CPU; any CUDA execution belongs to
the separately authorized GPU acceptance.
