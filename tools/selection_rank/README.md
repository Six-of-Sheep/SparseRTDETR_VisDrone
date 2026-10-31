# r15: the selection-ranking term wired into the production path

Round scope (record section 19): wire the r14 quality-ordered near-tie selection-ranking
term into the real encoder outputs and the criterion, freeze the total-loss weight,
application site, normalization and stop-gradient boundary **before any run**, complete
the owner-to-pool index mapping, the production invariants, the A1 independent RNG state
and the empty-pair path, and verify all of that on a few real training-side inputs with
the real production code.  CPU only; no GPU (still requires explicit authorization), no
training round, no scoring, no new module and no parameter sweep.

## What is wired where

    statistic      the term acts on the full encoder pool logits handed to
                   ``decoder._select_topk`` (the same tensor the selection rule uses),
                   not on the decoder queries
    capture        ``SelectionRankTerm.attach`` wraps ``decoder._select_topk``; the
                   cache lives on the term (never on a module) and is consumed by the
                   criterion call of the same micro-batch
    owner          the frozen matcher's matched query, mapped to its pool index through
                   the cached top-k indices: owner_pool = topk_indices[query]; the
                   harness checks bitwise that the cached indices reproduce the
                   decoder's gathered enc_topk logits and boxes
    matched pairs  the criterion's ``get_loss`` diagnostics record the final family's
                   ``matched_indices`` (query, ground truth); aux/enc/dn families never
                   feed the term
    quality        q(t,g) = IoU(sigmoid(cxcywh pool box of t), GT box of g), under no_grad
    gradient       only the pool logits' max-class entries of the two chosen tokens per
                   pair; a backward hook records -coef/N on each better (unselected)
                   token and nothing else on any unselected pool token

## Frozen before launch (declared, never tuned on results)

    loss key       loss_rank
    weight         lambda = 1.0 inside the criterion, once per micro-batch
    position       final family only, added after the native losses
    normalization  L = (1/N) * sum relu(margin - gap), N = instances with a non-empty
                   eligible set in that micro-batch; N = 0 -> exactly 0.0 while the key
                   stays present (native loss counts become 13 without DN, 22 with DN)
    stop gradient  q, size band, coverage, neighbour protection, score band, exclusivity,
                   partner choice and margins are all no-grad

## A1 control

Same eligibility, dedup, counts, margins and loss.  The partner is drawn uniformly from
the eligible set with the term's own ``torch.Generator``; its state (and the draw
counter) is part of ``state_dict`` and must be saved/restored with the checkpoint.  The
training RNG stream is never read or written (checked around a real criterion call).

## Test-criteria correction (same batch)

``worker.py`` ``full_clock`` and ``slice_ft_job.py`` ``update`` become mode-aware
(13/22 with an installed term, 12/21 otherwise); ``slice_finetune_cpu.py`` keeps its
native 21/12 assertion and is not reused by a term-installed arm (declared, not patched).

## Files

    selection_rank.py           the term (module plus the r14-suite compatibility adapters)
    patch_frozen.py             the four anchored frozen-file patches of this round
    verify_production_path.py   the CPU harness on the real chain (contract, Dataset,
                                FineTuneEngine): real loss-key counts, pair count, zero-pair
                                fraction, choice-space fraction, the backward audit, the A1
                                state round-trip and the index mapping
    test_selection_rank.py      the r14 synthetic suite (unchanged; runs against v3)

## Declared limits

The verification configuration is not the production recipe (real canvas 768x1344,
physical batch 1, fp32; production stays batch 8 / bf16).  GPU cost is not measured.
The reported numbers are mechanism-opportunity evidence on a few real inputs, not a
benefit result and not a threshold.
