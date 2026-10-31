# r14 corrected prototype: quality-ordered near-tie selection ranking term

Round scope (user instruction 2026-11-01, second pass): record section 16 stays an
**unclosed hypothesis**; no formal chain is assembled; this round delivers the
corrected formulas, the corresponding source and a small set of synthetic checks.
No GPU, no training, no detection scoring, and no claim of a research result.

## What it raises and what it lowers, and why (corrected)

The v1 rule decided the direction from *matched* versus *unselected* status.  That is
a bookkeeping proxy: the frozen Hungarian matcher minimises a global cost, so an
instance's matched token can be a worse localizer pairwise than an unselected
competitor, and v1 would then have pushed the worse token up.  The corrected rule
decides the direction from an explicit quality measure and fires only where the
frozen selection is measurably mis-ordered:

    q(t, g)  = IoU(predicted box of t, ground-truth box of g)      (higher is better)
    owner    = the frozen matcher's matched token (bookkeeping only)
    raise    = the BETTER of the pair;   lower = the WORSE one       (decided by q)

## Eligible competitor (spatial relation, coverage, neighbour protection)

    spatial    |log sqrt(area_u / area_g)| <= B = 0.7            (same size regime)
    coverage   q(u,g) >= q_floor = 0.25                          (u really covers g)
               and q(u,g) > q(owner,g) + q_margin = 0.10         (u is measurably better)
    neighbour  for every other GT g' != g: IoU(pred_box_u, gt_box_g') <= tau = 0.10
    score band rank_s(u) <= ceil(alpha * k), alpha = 4, and u is not selected
    exclusivity u is not another instance's matched token; a token partners at most one
               instance -- smallest |log size error| wins, ties to the smallest instance index

One pair per instance at most.  **No padding**: an instance without an eligible
competitor contributes zero pairs, exactly zero loss and no gradient.

    partner    A2: the highest-q eligible token (ties -> lowest index)
               A1: uniform draw from the same eligible set with a declared seed
    margin(g)  = m0 * clamp(sqrt(w_g h_g) / s_ref, 0.5, 2.0),  m0 = 1.0, s_ref = 0.05
    L          = (1/N) * sum_g relu(margin(g) - (s(better) - s(worse))),  N = 0 -> L = 0
    gradients  dL/ds(better) = -1/N (raised), dL/ds(worse) = +1/N (lowered), zero elsewhere

## Added direct supervision, registered from the frozen source (corrected)

The earlier statement that VFL supervises the whole encoder pool was wrong and is
retracted.  In the frozen release:

- `vendor/rtdetrv2_pytorch/src/zoo/rtdetr/rtdetrv2_decoder.py:504-511` builds the
  encoder auxiliary outputs from `self._select_topk(...)`, and line 593 sets
  `out['enc_aux_outputs']` from those **top-k selected** logits/boxes;
- `vendor/rtdetrv2_pytorch/src/zoo/rtdetr/rtdetrv2_criterion.py:194-207` applies the
  losses to exactly those `enc_aux_outputs`, with one Hungarian match per family;
- the carrier names those families `enc{i}` (`src/sparse_rtdetr/protocol_r18/criterion.py:23`).

So the encoder-level direct supervision covers the selected tokens; an **unselected
pool token takes part only through the hard top-k selection**, which carries no
gradient.  The partner token in this term therefore receives its first direct
supervision at that level, and the term is registered as an added direct supervision
of that token (at most 2N tokens per update, plus one new weak label: the matched
token must be outranked by a measurably better eligible competitor).

## What the checks verify (and what they do not)

`test_selection_rank.py` (synthetic logits/boxes, frozen interpreter, CUDA hidden):
direction follows quality in a scene where the selected token is worse and an
unselected token is better; no padding (owner already better, no coverage, wrong size
band); neighbour protection excluded/admitted across two variants of one scene;
exclusivity when a token is eligible for two instances; a control scene with **two
legal competitors** where A1 and A2 demonstrably choose differently with identical
eligibility, counts and margins; an already-satisfied pair with exactly zero loss and
zero gradient; margin monotonicity with the clamp; the added gradient touching at most
2N tokens; and the quality measure being the predicted-box/GT IoU.

It claims nothing about AP, coverage or selection rates.  Whether aligning the score
order with the box branch's own quality helps or hurts can only be answered by
training; q is the model's own opinion, so the target moves with the box branch (risk
declared).
