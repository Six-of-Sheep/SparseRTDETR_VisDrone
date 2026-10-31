"""Train-time near-tie selection ranking term -- r13 minimal prototype (record section 16/17).

WHAT IT RAISES AND WHAT IT SUPPRESSES (declared before implementation)
---------------------------------------------------------------------
The selection rule of the frozen carrier is unchanged at inference: it takes the
global top-k of the max-class encoder logit ``s(t) = max_c logits[t, c]``.  D1
measured that the tokens lost for small objects sit inside a narrow near-tie band
(58.2% of lost small GTs have their best token within the top 3% by score) and
that the score ordering does not track localization quality there (small paired
IoU median 0.7126 against 0.8823 for medium, score-size Spearman -0.51).

This term therefore
  * RAISES the matched token ``t*(g)`` of each ground-truth instance ``g``: the
    token the frozen Hungarian matcher already chose, i.e. the token that is the
    best localizer for ``g`` under the matcher's own localization cost.  The
    basis is the matcher's decision, not a new quality estimate; and
  * SUPPRESSES its partner ``t-(g)``: the best-localizing UNSELECTED near-tie
    token of the same ``g``, restricted to a declared relative-size band.  The
    basis is again the matcher's localization cost, evaluated on tokens the
    selection rule discarded.

It is a ranking constraint, not a new score, and it changes nothing at inference:
if the term works, the same top-k rule selects different tokens.

PAIRING AND LOSS (complete formulas)
------------------------------------
Per image, with ``S`` the selected set (top-k by ``s``), ``M`` the matched pairs
from the frozen matcher and ``rank(t)`` the descending-score rank with a
deterministic tie-break (higher token index gets the worse rank):

    eligible(g) = { t : t not in S,
                        |log(sqrt(area_t / area_g))| <= B,
                        rank(t) <= R_max = ceil(alpha * k),
                        t is not the matched token of any other GT }
    dedup: a token may partner at most ONE g -- it is assigned to the g with the
           smallest |log size error| (ties -> smallest GT index) and dropped from
           all others; so the eligible sets are disjoint after dedup.
    partner(g) = argmin_{t in eligible(g)} loc(t, g)        (A2, the method)
               = uniform draw from eligible(g)              (A1, the control)
    loc(t, g)  = w_bbox * L1(pred_box_t, gt_box_g) + w_giou * (1 - IoU)   (same
                 terms as the matcher's localization cost; ties -> lowest index)
    margin(g)  = m0 * clamp(sqrt(w_g * h_g) / s_ref, lo, hi)   (shrinks with the
                 object's scale)
    L          = (1 / N) * sum_g relu(margin(g) - (s(t*(g)) - s(t-(g)))) ,  N =
                 number of GTs with a non-empty eligible set after dedup
    if N == 0:  L == 0 exactly and no gradient anywhere.

Gradients on the max-class logit only:
    dL/ds(t*(g)) = -1/N on active pairs, dL/ds(t-(g)) = +1/N on active pairs,
    exactly 0 for every other token and exactly 0 for pairs whose margin is
    already satisfied.  The term is class-agnostic in the sense that it acts on
    whatever class is each token's argmax.

CONTROL AND COUNTING RULES SHARED BY A1/A2
------------------------------------------
A1 and A2 use the SAME eligibility code, the SAME dedup rule, the SAME pair count
(one eligible GT -> one pair), the SAME margin and the SAME loss form; they differ
only in how the partner is picked inside the already-fixed eligible set (uniform
random under a declared generator vs. the best localizer).  A1's draw is seeded
and reproducible; a GT whose eligible set is empty contributes nothing in both.

HONEST REGISTRATION OF THE ADDED DIRECT SUPERVISION
--------------------------------------------------
No token becomes supervised that was not already supervised (the VFL term already
supervises every token, as positive or negative).  What is NEW is a pairwise,
geometry-conditioned constraint placed on two already-supervised tokens per pair:
at most 2*N tokens per update receive the added ranking gradient, and the term
introduces one new weak label -- "the matched token must outrank the
best-localizing eligible near-tie token of the same instance".  The term is
therefore registered as an added DIRECT supervision of the partner token, not as a
re-weighting inside an unchanged objective.

Boundary: this prototype only shows that the formulas implement the mechanism
described above (gradient direction, empty partner, cross-GT conflict, A1/A2
eligibility and counting parity).  It claims nothing about AP or coverage, is not
wired into the formal chain, and does not run on any GPU.
"""
from __future__ import annotations

import math

import torch

DECLARATION = {
    "objective": "selection_rank_v1",
    "statistic": "max_class(encoder score logits) per token",
    "selected_set": "global top-k by that statistic (k=300 in production)",
    "raises": "the matched token of each ground-truth instance",
    "suppresses": "its partner: the best-localizing unselected near-tie token of the same instance",
    "basis": "the frozen matcher's own localization cost (w_bbox*L1 + w_giou*(1-IoU)), not a new quality estimate",
    "pairs_per_instance": 1,
    "size_band_B": 0.7,
    "rank_band_alpha": 4.0,
    "margin_m0": 1.0,
    "margin_s_ref": 0.05,
    "margin_clamp": [0.5, 2.0],
    "tie_break": "lowest token index for partners, smallest GT index for the dedup owner",
    "control_A1": "identical eligibility/dedup/count/margin/loss; partner drawn uniformly from the eligible set with a declared generator seed",
    "method_A2": "identical eligibility/dedup/count/margin/loss; partner chosen as the best localizer inside the eligible set",
    "gradient": "dL/ds(t*) = -1/N, dL/ds(t-) = +1/N on active pairs; zero elsewhere",
    "new_direct_supervision": ("no new supervised tokens; a pairwise geometry-conditioned constraint on two "
                              "already-supervised tokens per pair; at most 2N tokens per update; one new weak "
                              "label: the matched token must outrank the best-localizing eligible near-tie token"),
    "inference": "unchanged: same top-k rule, no module, no head, no parameter",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def max_class_statistic(logits):
    """s(t) = max over classes of the encoder score logit (the statistic the rule uses).

    Accepts (tokens, classes) for a single image and (images, tokens, classes).
    """
    _require(logits.dim() in (2, 3), "logits must be (images, tokens, classes) or (tokens, classes)")
    return logits.max(dim=-1).values


def ranks_of(scores):
    """Descending-score rank with a deterministic tie-break (higher index ranks worse)."""
    order = sorted(range(scores.shape[0]), key=lambda i: (-float(scores[i]), i))
    rank = torch.empty(scores.shape[0], dtype=torch.int64)
    for position, index in enumerate(order):
        rank[index] = position
    return rank


def _iou_xyxy(a, b):
    low = torch.maximum(a[:, :2], b[None, :2])
    high = torch.minimum(a[:, 2:], b[None, 2:])
    inter = (high - low).clamp_min(0).prod(-1)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return (inter / (area_a + area_b - inter).clamp_min(1e-12))


def _xyxy(boxes):
    return torch.cat([boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2], dim=-1)


def pairing(logits, pred_boxes, gt_boxes, matched, *, k, mode, weights, generator=None):
    """Eligibility, dedup and partner choice for one image.

    ``matched`` is a list of (token_index, gt_index) from the frozen matcher.
    Returns per-GT records so the caller can compute the loss and the counts.
    """
    _require(mode in ("A1", "A2"), "unknown ranking mode: " + str(mode))
    _require(k >= 1 and k <= logits.shape[0], "k outside the token count")
    scores = max_class_statistic(logits)
    rank = ranks_of(scores)
    order = torch.argsort(scores, descending=True, stable=True)
    selected = set(int(i) for i in order[:k])
    matched_tokens = {int(i) for i, _ in matched}
    r_max = int(math.ceil(DECLARATION["rank_band_alpha"] * k))
    boxes_xyxy = _xyxy(pred_boxes)
    gt_xyxy = _xyxy(gt_boxes)

    candidates = {}
    for _, g in matched:
        g = int(g)
        area_g = float(gt_boxes[g, 2] * gt_boxes[g, 3])
        _require(area_g > 0, "ground-truth box has no area")
        pool = []
        for t in range(logits.shape[0]):
            if t in selected or t in matched_tokens:
                continue
            if int(rank[t]) > r_max:
                continue
            area_t = float(pred_boxes[t, 2] * pred_boxes[t, 3])
            if area_t <= 0:
                continue
            error = abs(math.log(math.sqrt(area_t / area_g)))
            if error > DECLARATION["size_band_B"]:
                continue
            pool.append((t, error))
        candidates[g] = pool

    # dedup: a token partners at most one instance -- smallest |log size error| wins
    owner = {}
    for g, pool in sorted(candidates.items()):
        for t, error in pool:
            best = owner.get(t)
            if best is None or (error, g) < (best[0], best[1]):
                owner[t] = (error, g)
    eligible = {g: [t for t, error in pool if owner[t][1] == g] for g, pool in candidates.items()}

    records = []
    for token, g in sorted((int(i), int(j)) for i, j in matched):
        pool = eligible.get(g, [])
        if not pool:
            records.append({"gt": g, "matched_token": token, "partner": None,
                            "eligible_count": 0, "margin": None, "active": False})
            continue
        margin = float(DECLARATION["margin_m0"] * min(max(
            float(math.sqrt(float(gt_boxes[g, 2] * gt_boxes[g, 3]))) / DECLARATION["margin_s_ref"],
            DECLARATION["margin_clamp"][0]), DECLARATION["margin_clamp"][1]))
        if mode == "A2":
            costs = []
            for t in pool:
                l1 = float((pred_boxes[t] - gt_boxes[g]).abs().sum())
                iou = float(_iou_xyxy(boxes_xyxy[t:t + 1], gt_xyxy[g].unsqueeze(0))[0])
                costs.append((weights["cost_bbox"] * l1 + weights["cost_giou"] * (1.0 - iou), t))
            partner = min(costs, key=lambda row: (row[0], row[1]))[1]
        else:
            _require(generator is not None, "A1 needs the declared generator")
            draw = torch.randint(len(pool), (1,), generator=generator).item()
            partner = pool[int(draw)]
        gap = float(scores[token] - scores[partner])
        records.append({"gt": g, "matched_token": token, "partner": int(partner),
                        "eligible_count": len(pool), "margin": margin, "active": bool(gap < margin),
                        "score_gap": gap})
    return {"records": records, "selected": sorted(selected), "r_max": r_max,
            "eligible": {g: list(pool) for g, pool in eligible.items()},
            "eligible_counts": {g: len(pool) for g, pool in eligible.items()},
            "candidate_counts": {g: len(pool) for g, pool in candidates.items()}}


def loss(images_logits, pred_boxes, gt_boxes, matched, *, k, mode, weights, generator=None):
    """The ranking term; returns (loss tensor, evidence dict).

    A1 and A2 share eligibility, dedup, pair count, margins and the loss form.
    """
    _require(isinstance(gt_boxes, (list, tuple)) or (hasattr(gt_boxes, "dim") and gt_boxes.dim() == 3),
             "gt_boxes must be a list of per-image (G,4) tensors or a (B,G,4) tensor")
    _require(len(matched) == images_logits.shape[0] == len(gt_boxes), "batch members differ")
    total_pairs = 0
    active_pairs = 0
    terms = []
    per_image = []
    touched = set()
    for batch in range(images_logits.shape[0]):
        pairs = [(int(i), int(j)) for i, j in matched[batch]]
        info = pairing(images_logits[batch], pred_boxes[batch], gt_boxes[batch], pairs,
                       k=k, mode=mode, weights=weights, generator=generator)
        per_image.append(info)
        for record in info["records"]:
            if record["partner"] is None:
                continue
            total_pairs += 1
            scores = max_class_statistic(images_logits[batch])
            gap = scores[record["matched_token"]] - scores[record["partner"]]
            term = torch.relu(torch.as_tensor(record["margin"], dtype=scores.dtype, device=scores.device) - gap)
            terms.append(term)
            if bool(term > 0):
                active_pairs += 1
                touched.add((batch, record["matched_token"]))
                touched.add((batch, record["partner"]))
    if not terms:
        value = images_logits.sum() * 0.0
    else:
        value = torch.stack(terms).sum() / float(len(terms))
    evidence = {
        "mode": mode, "pairs": total_pairs, "active_pairs": active_pairs,
        "tokens_with_added_gradient": len(touched), "k": k, "declaration": DECLARATION,
        "per_image": per_image,
    }
    return value, evidence


def check_gradient_direction(logits, pred_boxes, gt_boxes, matched, *, k, mode="A2", weights, generator=None):
    """Reference implementation of the expected gradient, for the synthetic checks."""
    _require(isinstance(gt_boxes, (list, tuple)) or (hasattr(gt_boxes, "dim") and gt_boxes.dim() == 3),
             "gt_boxes must be a list of per-image (G,4) tensors or a (B,G,4) tensor")
    scores = max_class_statistic(logits)
    total = 0
    pairs = []
    for batch in range(logits.shape[0]):
        info = pairing(logits[batch], pred_boxes[batch], gt_boxes[batch],
                       [(int(i), int(j)) for i, j in matched[batch]], k=k, mode=mode,
                       weights=weights, generator=generator)
        for record in info["records"]:
            if record["partner"] is None:
                continue
            total += 1
            pairs.append((batch, record))
    expected = torch.zeros_like(logits)
    if total == 0:
        return 0.0, expected, pairs
    value = 0.0
    for batch, record in pairs:
        gap = float(scores[batch, record["matched_token"]] - scores[batch, record["partner"]])
        term = max(0.0, record["margin"] - gap)
        value += term
        if term > 0:
            best = int(logits[batch, record["matched_token"]].argmax())
            other = int(logits[batch, record["partner"]].argmax())
            expected[batch, record["matched_token"], best] -= 1.0 / total
            expected[batch, record["partner"], other] += 1.0 / total
    return value / total, expected, pairs
