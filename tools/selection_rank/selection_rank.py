"""Quality-ordered near-tie selection ranking term -- r14 corrected definition (record section 18).

WHAT IT RAISES AND WHAT IT LOWERS, AND WHY (corrected)
-----------------------------------------------------
The v1 rule of r13 decided the direction from ``matched`` versus ``unselected``
status, which is a bookkeeping proxy: the frozen Hungarian matcher minimises a
global cost, so an instance's matched token can easily be a WORSE localizer
pairwise than an unselected competitor (that is exactly the dense-competition
case the mechanism targets).  v1 would then have pushed the worse token up.  This
definition fixes the direction by an explicit quality measure and only ever fires
where the frozen selection is measurably mis-ordered:

    quality   q(t, g) = IoU(pred_box_t, gt_box_g)                  (higher is better)

    owner     t*(g) = the frozen matcher's matched token for g      (bookkeeping only)

    eligible competitor u for g  (all conditions required):
      spatial   |log sqrt(area_u / area_g)| <= B                     (same size regime)
      coverage  q(u, g) >= q_floor            (u really covers this instance)
                and q(u, g) > q(t*(g), g) + q_margin   (u is measurably BETTER)
      neighbour protection: for every other GT g' != g, IoU(pred_box_u, gt_box_g') <= tau
      score band: rank_s(u) <= ceil(alpha * k) and u is not selected
      exclusivity: u is not another instance's matched token, and a token partners at
                  most one instance (smallest |log size error| wins, ties to the
                  smallest instance index)

    pairing   one pair per instance at most.  If the eligible set is empty the
              instance contributes NOTHING: zero pairs, exactly zero loss, no
              padding.  The competitor is by construction the BETTER token and the
              owner the WORSE one, so the term raises the better and lowers the
              worse -- decided by q, never by selected/unselected status.

    partner   A2 (method): the eligible token with the highest q(u, g)
                           (ties -> lowest token index)
              A1 (control): uniform draw from the same eligible set with a
                           declared generator seed

    margin(g) = m0 * clamp(sqrt(w_g * h_g) / s_ref, 0.5, 2.0)      (shrinks with scale)

    L         = (1/N) * sum_g relu(margin(g) - (s(better) - s(worse))),   N = number of
                instances with a non-empty eligible set;   N = 0  ->  L == 0 exactly

    gradients dL/ds(better) = -1/N (raised), dL/ds(worse) = +1/N (lowered) on active
              pairs; exactly zero elsewhere and zero for pairs whose margin already holds.

CONTROL AND SHARED RULES
------------------------
A1 and A2 use the SAME eligibility code, the SAME dedup, the SAME pair count, the
SAME margin and the SAME loss form; they differ only in which eligible token is
chosen.  A1's draw is seeded and reproducible; a scene with two legal competitors
makes the two arms choose differently (checked).

WHAT ONLY TRAINING CAN ANSWER
-----------------------------
Whether aligning the score order with the box branch's current quality helps or
hurts; the risk is declared: q is the model's own opinion, so the target moves with
the box branch, and the term never rewards lowering a token's box quality.  This
prototype claims nothing about AP, coverage or selection rates.

Boundary: corrected formulas only.  No formal chain is assembled, nothing runs on a
GPU, and the checks below verify the mechanism as implemented, not a research result.
"""
from __future__ import annotations

import math

import torch

DECLARATION = {
    "objective": "selection_rank_v2_quality_ordered",
    "statistic": "max_class(encoder score logits) per token; the selection rule itself is unchanged",
    "quality": "q(t,g) = IoU(predicted box of t, ground-truth box of g); higher is better; no matcher involvement",
    "owner": "the frozen matcher's matched token t*(g) -- bookkeeping only, it does not set the direction",
    "direction": ("raise the BETTER of the pair and lower the WORSE one, where better/worse are decided by q; "
                  "the pairing only fires when the owner is measurably worse than an eligible competitor"),
    "eligible": {
        "spatial": "|log sqrt(area_u / area_g)| <= B",
        "coverage": "q(u,g) >= q_floor and q(u,g) > q(owner,g) + q_margin",
        "neighbour_protection": "for every other GT g' != g: IoU(pred_box_u, gt_box_g') <= tau",
        "score_band": "rank_s(u) <= ceil(alpha * k) and u not selected",
        "exclusivity": "u is not another instance's matched token; a token partners at most one instance",
    },
    "pairs_per_instance": 1,
    "empty_allowed": "an instance without an eligible competitor contributes nothing: zero pairs, exactly zero loss, no padding",
    "constants": {"B": 0.7, "q_floor": 0.25, "q_margin": 0.10, "tau": 0.10, "alpha": 4.0,
                  "m0": 1.0, "s_ref": 0.05, "margin_clamp": [0.5, 2.0]},
    "loss": "L = (1/N) * sum_g relu(margin(g) - (s(better) - s(worse)))",
    "gradient": "dL/ds(better) = -1/N, dL/ds(worse) = +1/N on active pairs; zero elsewhere",
    "control_A1": ("identical eligibility/dedup/count/margin/loss; the competitor is drawn uniformly from the "
                   "eligible set with a declared generator seed"),
    "method_A2": "identical eligibility/dedup/count/margin/loss; the competitor is the highest-q eligible token",
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


def _xyxy(boxes):
    return torch.cat([boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2], dim=-1)


def iou_matrix(boxes_a, boxes_b):
    """Pairwise IoU between (N,4) and (M,4) xyxy tensors -> (N, M).  This IS the quality."""
    low = torch.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    high = torch.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    inter = (high - low).clamp_min(0).prod(-1)
    area_a = ((boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1]))[:, None]
    area_b = ((boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1]))[None, :]
    return inter / (area_a + area_b - inter).clamp_min(1e-12)


def pairing(logits, pred_boxes, gt_boxes, matched, *, k, mode, generator=None):
    """Eligibility, exclusivity and competitor choice for one image (corrected rule).

    ``matched`` is a list of (token_index, gt_index) from the frozen matcher.
    """
    _require(mode in ("A1", "A2"), "unknown ranking mode: " + str(mode))
    _require(k >= 1 and k <= logits.shape[0], "k outside the token count")
    _require(gt_boxes.dim() == 2 and gt_boxes.shape[1] == 4, "gt_boxes must be (G,4) for one image")
    constants = DECLARATION["constants"]
    scores = max_class_statistic(logits)
    rank = ranks_of(scores)
    order = torch.argsort(scores, descending=True, stable=True)
    selected = set(int(i) for i in order[:k])
    matched_tokens = {int(i) for i, _ in matched}
    r_max = int(math.ceil(constants["alpha"] * k))
    boxes_xyxy = _xyxy(pred_boxes)
    gt_xyxy = _xyxy(gt_boxes)
    quality = iou_matrix(boxes_xyxy, gt_xyxy)          # (T, G)

    candidates = {}
    for _token, gt in matched:
        g = int(gt)
        area_g = float(gt_boxes[g, 2] * gt_boxes[g, 3])
        _require(area_g > 0, "ground-truth box has no area")
        owner = int(_token)
        pool = []
        for u in range(logits.shape[0]):
            if u == owner or u in selected or u in matched_tokens:
                continue
            if int(rank[u]) > r_max:
                continue
            area_u = float(pred_boxes[u, 2] * pred_boxes[u, 3])
            if area_u <= 0:
                continue
            error = abs(math.log(math.sqrt(area_u / area_g)))
            if error > constants["B"]:
                continue
            q_u = float(quality[u, g])
            if q_u < constants["q_floor"] or q_u <= float(quality[owner, g]) + constants["q_margin"]:
                continue
            if any(float(quality[u, other]) > constants["tau"] for other in range(len(gt_boxes)) if other != g):
                continue
            pool.append((u, q_u, error))
        candidates[g] = pool

    # exclusivity: a token partners at most one instance -- smallest relative size error wins
    owner_of = {}
    for g, pool in sorted(candidates.items()):
        for u, _q, error in pool:
            best = owner_of.get(u)
            if best is None or (error, g) < (best[0], best[1]):
                owner_of[u] = (error, g)
    eligible = {g: [(u, q) for u, q, error in pool if owner_of[u][1] == g] for g, pool in candidates.items()}

    records = []
    for token, gt in sorted((int(i), int(j)) for i, j in matched):
        g = int(gt)
        pool = eligible.get(g, [])
        if not pool:
            records.append({"gt": g, "owner": int(token), "partner": None, "better": None, "worse": None,
                            "eligible_count": 0, "margin": None, "active": False, "score_gap": None})
            continue
        margin = float(constants["m0"] * min(max(
            float(math.sqrt(float(gt_boxes[g, 2] * gt_boxes[g, 3]))) / constants["s_ref"],
            constants["margin_clamp"][0]), constants["margin_clamp"][1]))
        if mode == "A2":
            partner = min(pool, key=lambda row: (-row[1], row[0]))[0]
        else:
            _require(generator is not None, "A1 needs the declared generator")
            partner = pool[int(torch.randint(len(pool), (1,), generator=generator).item())][0]
        # the competitor is eligible only if it is measurably better than the owner
        better, worse = int(partner), int(token)
        _require(float(quality[better, g]) > float(quality[worse, g]) + constants["q_margin"],
                 "pairing produced a token that is not measurably better than the owner")
        gap = float(scores[better] - scores[worse])
        records.append({"gt": g, "owner": int(token), "partner": int(partner), "better": better, "worse": worse,
                        "eligible_count": len(pool), "margin": margin, "active": bool(gap < margin),
                        "score_gap": gap,
                        "quality_better": float(quality[better, g]), "quality_worse": float(quality[worse, g])})
    return {"records": records, "selected": sorted(selected), "r_max": r_max,
            "eligible": {g: [u for u, _q in pool] for g, pool in eligible.items()},
            "eligible_counts": {g: len(pool) for g, pool in eligible.items()},
            "candidate_counts": {g: len(pool) for g, pool in candidates.items()}}


def loss(images_logits, pred_boxes, gt_boxes, matched, *, k, mode, generator=None):
    """The ranking term; returns (loss tensor, evidence dict).

    A1 and A2 share eligibility, exclusivity, pair count, margins and the loss form.
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
                       k=k, mode=mode, generator=generator)
        per_image.append(info)
        scores = max_class_statistic(images_logits[batch])
        for record in info["records"]:
            if record["partner"] is None:
                continue
            total_pairs += 1
            gap = scores[record["better"]] - scores[record["worse"]]
            term = torch.relu(torch.as_tensor(record["margin"], dtype=scores.dtype, device=scores.device) - gap)
            terms.append(term)
            if bool(term > 0):
                active_pairs += 1
                touched.add((batch, record["better"]))
                touched.add((batch, record["worse"]))
    value = images_logits.sum() * 0.0 if not terms else torch.stack(terms).sum() / float(len(terms))
    evidence = {"mode": mode, "pairs": total_pairs, "active_pairs": active_pairs,
                "tokens_with_added_gradient": len(touched), "k": k,
                "declaration": DECLARATION, "per_image": per_image}
    return value, evidence


def check_gradient_direction(logits, pred_boxes, gt_boxes, matched, *, k, mode="A2", generator=None):
    """Reference implementation of the expected gradient, for the synthetic checks."""
    _require(isinstance(gt_boxes, (list, tuple)) or (hasattr(gt_boxes, "dim") and gt_boxes.dim() == 3),
             "gt_boxes must be a list of per-image (G,4) tensors or a (B,G,4) tensor")
    scores = max_class_statistic(logits)
    total = 0
    pairs = []
    for batch in range(logits.shape[0]):
        info = pairing(logits[batch], pred_boxes[batch], gt_boxes[batch],
                       [(int(i), int(j)) for i, j in matched[batch]], k=k, mode=mode, generator=generator)
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
        gap = float(scores[batch, record["better"]] - scores[batch, record["worse"]])
        term = max(0.0, record["margin"] - gap)
        value += term
        if term > 0:
            expected[batch, record["better"], int(logits[batch, record["better"]].argmax())] -= 1.0 / total
            expected[batch, record["worse"], int(logits[batch, record["worse"]].argmax())] += 1.0 / total
    return value / total, expected, pairs
