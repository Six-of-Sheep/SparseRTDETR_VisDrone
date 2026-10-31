# r13 minimal prototype: train-time near-tie selection ranking term

Scope of this round (user instruction 2026-11-01): record section 16 is kept as an
**unclosed hypothesis**, the formal chain is NOT assembled on the current formula,
and this round delivers one minimal prototype plus synthetic verification only.
No GPU, no training, no detection scoring, no frozen-model gain is required.

## What it raises, what it suppresses, and on what basis

- Carries on the selection statistic the rule already uses: `s(t) = max_class(logits[t])`,
  top-k selected (k=300 in production). The rule itself is untouched at inference.
- **Raises** the matched token of each ground-truth instance: the token the frozen
  Hungarian matcher already chose, i.e. the best localizer for that instance under the
  matcher's own localization cost.
- **Suppresses** its partner: the best-localizing *unselected* near-tie token of the
  same instance, restricted to a declared relative-size band. Basis is again the
  matcher's localization cost, evaluated on tokens the selection rule discarded.
- Why those two: the r10 D1 diagnostic measured that lost small objects sit in a narrow
  near-tie band (58.2% of losses within the top 3% by score) and that the score ordering
  there does not track localization quality (small paired IoU median 0.7126 vs 0.8823
  for medium; score-size Spearman -0.51).

## Pairing and loss formulas (as implemented)

    eligible(g) = { t : t not selected, |log sqrt(area_t / area_g)| <= B = 0.7,
                        rank(t) <= ceil(alpha * k), alpha = 4,
                        t is not the matched token of another instance }
    dedup       : a token partners at most one instance -- smallest |log size error|
                  wins, ties to the smallest instance index
    partner(g)  = argmin_{t in eligible(g)} [ w_bbox * L1 + w_giou * (1 - IoU) ]   (A2)
                  uniform draw from eligible(g) with a declared generator seed     (A1)
    margin(g)   = m0 * clamp(sqrt(w_g h_g) / s_ref, 0.5, 2.0)   (m0 = 1.0, s_ref = 0.05)
    L           = (1/N) * sum_g relu(margin(g) - (s(t*(g)) - s(t-(g)))),  N = instances
                  with a non-empty eligible set;  N = 0  ->  L = 0 exactly

Gradients: `dL/ds(t*) = -1/N`, `dL/ds(t-) = +1/N` on active pairs, exactly zero
elsewhere and exactly zero for pairs whose margin is already satisfied.

## A1/A2 shared rules

Identical eligibility code, identical dedup, identical pair count (one pair per
eligible instance), identical margins and identical loss form. The only difference is
which eligible token is chosen. A1's draw is seeded and reproducible.

## Registered added direct supervision (honest scope)

No token becomes supervised that was not already supervised by VFL. What is new is a
pairwise, geometry-conditioned constraint on two already-supervised tokens per pair:
at most 2N tokens per update receive the added ranking gradient, and the term
introduces one new weak label -- "the matched token must outrank the best-localizing
eligible near-tie token of the same instance". It is therefore an added direct
supervision of the partner token, and the inference path, parameters and selection
rule are unchanged.

## What the synthetic checks verify (and what they do not)

`test_selection_rank.py` verifies with small synthetic logits/boxes only: gradient
direction and magnitude, exact zero for a satisfied pair and for an empty partner,
cross-instance conflict resolution by the declared dedup rule, A1/A2 eligibility,
count and margin parity plus A1 reproducibility, margin monotonicity with the declared
clamp, and that the added gradient touches at most 2N tokens. It claims nothing about
AP, coverage or selection rates, which only training can answer.
