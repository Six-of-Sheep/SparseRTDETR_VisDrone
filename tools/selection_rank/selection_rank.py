"""Quality-ordered near-tie selection ranking term, production wiring (r15, record section 19).

WHAT IS WIRED WHERE
-------------------
The statistic the selection rule uses is the **full encoder pool** ``enc_outputs_logits``
(the tensor handed to ``decoder._select_topk``), so the ranking term must act on that
tensor, not on the decoder queries.  The owner of an instance is the *query* index the
frozen Hungarian matcher assigned; the term maps it to the pool index through the cached
top-k indices of the same forward: ``owner_pool = topk_indices[query]``.

    pool logits/boxes  captured by wrapping ``decoder._select_topk`` (same interception
                       the r10 D1 probe used); the cache lives on this object -- never on
                       a module -- so the frozen model layout stays untouched, and it is
                       consumed by the criterion call of the same micro-batch.
    matched pairs      recorded by the criterion's ``get_loss`` diagnostics for the
                       FINAL family (query index, ground-truth index); the term asserts
                       the head is ``final`` and refuses anything else.
    quality            q(t,g) = IoU(box(t), GT box of g), computed under
                       ``torch.no_grad()`` together with every eligibility mask, the
                       exclusivity rule and the partner choice.  In production the pool
                       coordinates are the unactivated cxcywh tensor, so the boxes are
                       ``sigmoid(coords)`` -- the harness checks that this reproduces the
                       gathered ``enc_topk_bboxes`` the decoder itself uses.
    gradient           flows ONLY through the pool logits' max-class entries of the two
                       chosen tokens per pair; nothing else in the graph is touched.
    gradient audit     optional (``audit_gradients``): a backward hook records what the
                       real pass put on the pool logits; only unselected tokens are
                       inspected, because no native loss reaches them
    geometry consumer  optional (``attach(..., consumer=geo)``): the term's pool logits,
                       quality matrix and loss join the frozen consumer-geometry audit

FROZEN BEFORE LAUNCH (declared, never tuned on results)
------------------------------------------------------
    loss key        "loss_rank"
    weight          lambda = 1.0, applied inside the criterion, once per micro-batch
    position        added to the final family's loss dict AFTER the native losses
                    (never on aux / enc / dn families)
    normalization   divided by N = instances with a non-empty eligible set in that
                    micro-batch; N = 0 -> exactly 0.0 (and the key is still present, so
                    the declared loss-key count becomes 13 without DN, 22 with DN)
    stop gradient   q, the size band, coverage, neighbour protection, the score band,
                    exclusivity, the partner choice and the margins are all no-grad
    A1 control      draws its partner from its own ``torch.Generator``; that generator's
                    state (and the draw counter) is part of ``state_dict`` and must be
                    saved/restored with the checkpoint; the training RNG stream is
                    never read or written

Everything about eligibility, coverage, neighbour protection, exclusivity and the
direction (raise the BETTER, lower the WORSE, decided by q) is unchanged from the r14
definition.  The r14 synthetic suite keeps running against this file: ``pairing`` and
``loss`` at the bottom are thin adapters over the same pairing core.
"""
from __future__ import annotations

import functools
import math
import time

import torch

MODES = ("A2", "A1")

DECLARATION = {
    "objective": "selection_rank_v3_quality_ordered",
    "statistic": "max_class(encoder pool logits); the selection rule itself is unchanged",
    "quality": "q(t,g) = IoU(box(t), GT box of g); computed under no_grad",
    "pool_boxes": "sigmoid(unactivated cxcywh pool coordinates) -> xyxy; identical to the gathered enc_topk_bboxes the decoder uses",
    "owner": "the frozen matcher's matched query, mapped to its pool index via the cached top-k indices",
    "direction": "raise the BETTER of the pair, lower the WORSE one; decided by q, never by selected/unselected status",
    "eligible": {
        "spatial": "|log sqrt(area_u / area_g)| <= B",
        "coverage": "q(u,g) >= q_floor and q(u,g) > q(owner,g) + q_margin",
        "neighbour_protection": "for every other GT g' != g: IoU(pred_box_u, gt_box_g') <= tau",
        "score_band": "rank_s(u) <= ceil(alpha * k) and u not selected",
        "exclusivity": "u is not any instance's matched token; a token partners at most one instance (smallest relative size error wins, ties to the smallest instance index)",
    },
    "instances": "matched ground truths of the final family (a GT without an owner token cannot form a pair)",
    "pairs_per_instance": 1,
    "empty_allowed": "an instance without an eligible competitor contributes nothing: zero pairs, exactly zero loss, no padding",
    "constants": {"B": 0.7, "q_floor": 0.25, "q_margin": 0.10, "tau": 0.10, "alpha": 4.0,
                  "m0": 1.0, "s_ref": 0.05, "margin_clamp": [0.5, 2.0]},
    "loss": "L = (1/N) * sum_g relu(margin(g) - (s(better) - s(worse)))",
    "gradient": "dL/ds(better) = -1/N, dL/ds(worse) = +1/N on contributing pairs; zero elsewhere",
    "total_loss": {
        "key": "loss_rank", "weight": 1.0,
        "position": "final family only, added to the loss dict after the native losses, once per micro-batch",
        "normalization": "divided by N (instances with a non-empty eligible set in that micro-batch); N = 0 -> exactly 0.0",
        "loss_key_count": "13 without DN, 22 with DN when the term is installed",
        "stop_gradient": "q, all eligibility masks, exclusivity, partner choice and margins are no-grad; gradients only on the pool logits' max-class entries of the pairs",
        "frozen_before_launch": True,
    },
    "control_A1": "identical eligibility/dedup/count/margin/loss; the partner is drawn uniformly from the eligible set with a per-term torch.Generator whose state is saved and restored with the checkpoint",
    "method_A2": "identical eligibility/dedup/count/margin/loss; the partner is the highest-q eligible token",
    "inference": "unchanged: same top-k rule, no module, no head, no parameter",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def max_class_statistic(logits):
    """s(t) = max over classes; accepts (tokens, classes) or (images, tokens, classes)."""
    _require(logits.dim() in (2, 3), "logits must be (images, tokens, classes) or (tokens, classes)")
    return logits.max(dim=-1).values


def ranks_of(scores):
    """Descending-score rank with a deterministic tie-break (lower index ranks better)."""
    _require(scores.dim() == 1, "ranks_of takes one token dimension")
    order = torch.argsort(scores, descending=True, stable=True)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(order.shape[0], dtype=order.dtype)
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


def _margin(side, constants):
    return (constants["m0"] * min(max(side / constants["s_ref"], constants["margin_clamp"][0]),
                                  constants["margin_clamp"][1]))


class SelectionRankTerm:
    """The production term: capture, pair, loss, state.  One instance per arm."""

    def __init__(self, mode, *, seed=0, k=300, audit_gradients=False):
        _require(mode in MODES, "unknown ranking mode: " + str(mode))
        self.mode = mode
        self.k = int(k)
        self.generator = torch.Generator().manual_seed(int(seed))
        self.seed = int(seed)
        self.draws = 0
        self.audit_gradients = bool(audit_gradients)
        self._cache = None
        self._original_select = None
        self._installed = False
        self._consumer = None
        self._pending_grad_audit = None
        self.last_grad_audit = None
        self.grad_audits = []
        self.history = []
        self.last_stats = {}

    # ---- installation ---------------------------------------------------------------
    def attach(self, model, criterion, *, consumer=None):
        """Wrap the decoder's selection call and register the term on the criterion.

        ``consumer`` is the frozen geometry consumer (``consumer_geometry``); when given,
        the term's own tensors join the same audit family as the native losses.
        """
        _require(not self._installed, "term already installed")
        self._consumer = consumer
        decoder = model.decoder
        _require(hasattr(decoder, "_select_topk"), "decoder has no _select_topk")
        self._original_select = decoder._select_topk
        decoder._select_topk = functools.partial(_cached_select, decoder, self)
        criterion.selection_rank = self
        self._installed = True
        return self

    def assert_installed(self, model, criterion):
        decoder = model.decoder
        bound = getattr(decoder, "_select_topk", None)
        _require(isinstance(bound, functools.partial), "selection call is not bound to a term partial")
        _require(bound.args[0] is decoder, "selection binding does not reference this decoder copy")
        _require(bound.args[1] is self, "selection binding does not reference this term")
        _require(getattr(criterion, "selection_rank", None) is self, "criterion is not bound to this term")
        return True

    # ---- state (A1's own generator) --------------------------------------------------
    def state_dict(self):
        return {"mode": self.mode, "k": self.k, "seed": self.seed, "draws": self.draws,
                "a1_generator": self.generator.get_state().clone()}

    def load_state_dict(self, state):
        _require(state["mode"] == self.mode and int(state["k"]) == self.k and int(state["seed"]) == self.seed,
                 "term state identity differs")
        self.generator.set_state(state["a1_generator"].clone())
        self.draws = int(state["draws"])
        return True

    # ---- capture (called by the wrapped decoder selection) ---------------------------
    def _capture(self, pool_logits, pool_coord_unact, topk_indices):
        self._cache = {"pool_logits": pool_logits, "pool_coord_unact": pool_coord_unact,
                       "topk_indices": topk_indices}

    def bind_manual(self, cache, *, consumer=None):
        """Bind one captured pool batch for a manual call (no decoder wrapper needed)."""
        self._consumer = consumer
        self._cache = {key: cache[key] for key in ("pool_logits", "pool_coord_unact", "topk_indices")}
        return self

    def consume_cache(self):
        _require(self._cache is not None, "no captured pool tensors for this micro-batch")
        cache, self._cache = self._cache, None
        return cache

    # ---- the term --------------------------------------------------------------------
    def criterion_loss(self, targets, diagnostics):
        """Called by the criterion once per micro-batch, final family only."""
        started = time.perf_counter()
        cache = self.consume_cache()
        pool_logits = cache["pool_logits"]
        pool_coord = cache["pool_coord_unact"]
        topk = cache["topk_indices"]
        pairs = self._pairs_from_diagnostics(diagnostics)
        _require(len(targets) == pool_logits.shape[0] == len(pairs), "term batch members differ")
        owners = [[(int(topk[image][query]), gt) for query, gt in pairs[image]]
                  for image in range(len(pairs))]
        self._pending_grad_audit = None
        if self._consumer is not None:
            self._consumer.tensor("rank/pool_logits", pool_logits)
        if self.audit_gradients:
            _require(pool_logits.shape[0] == 1, "the gradient audit is declared for a physical batch of one")
            pool_logits.register_hook(self._observe_pool_gradients)
        terms = []
        total = 0
        instances = 0
        with_pair = 0
        choice_space = 0
        per_image = []
        for image in range(len(targets)):
            gt_boxes = targets[image]["boxes"]
            if gt_boxes.numel() == 0 or not pairs[image]:
                per_image.append({"gt": int(gt_boxes.shape[0]), "instances": 0, "pairs": 0,
                                  "choice_space": 0, "records": []})
                continue
            records, info = self._pair_image(pool_logits[image], pool_coord[image], topk[image],
                                             owners[image], gt_boxes, activate=True)
            info.pop("eligible", None)
            info["records"] = records
            per_image.append(info)
            instances += info["instances"]
            with_pair += info["pairs"]
            choice_space += info["choice_space"]
            scores = max_class_statistic(pool_logits[image])
            for record in records:
                if record["partner"] is None:
                    continue
                total += 1
                if not record["active"]:
                    continue
                margin = torch.as_tensor(record["margin"], dtype=scores.dtype, device=scores.device)
                gap = scores[record["better"]] - scores[record["worse"]]
                terms.append(torch.relu(margin - gap))
        _require(total == with_pair, "pair count and instance-with-pair count differ")
        self._arm_grad_audit(pool_logits, topk, per_image, total)
        if terms:
            value = torch.stack(terms).sum() / float(total)
        else:
            value = pool_logits.sum() * 0.0
        if self._consumer is not None:
            self._consumer.tensor("rank/loss", value)
        self.last_stats = {"instances": instances, "pairs": total,
                           "active_pairs": sum(1 for row in per_image
                                               for record in row["records"] if record["active"]),
                           "zero_pair_instances": instances - with_pair,
                           "choice_space_instances": choice_space,
                           "loss": float(value), "per_image": per_image,
                           "milliseconds": round((time.perf_counter() - started) * 1000.0, 3)}
        self.history.append({key: self.last_stats[key] for key in
                             ("instances", "pairs", "active_pairs", "zero_pair_instances",
                              "choice_space_instances", "loss", "milliseconds")})
        return value

    def _arm_grad_audit(self, pool_logits, topk, per_image, total):
        """Declare, before the backward pass, exactly which pool entries the term may touch."""
        if not self.audit_gradients:
            return
        argmax = pool_logits.detach().max(dim=-1).indices
        selected = {int(token) for token in topk.detach().reshape(-1).tolist()}
        pairs = []
        for image, row in enumerate(per_image):
            for record in row.get("records", ()):
                if record["partner"] is None:
                    continue
                pairs.append({"image": int(image), "better": int(record["better"]),
                              "worse": int(record["worse"]), "active": bool(record["active"]),
                              "better_class": int(argmax[image, record["better"]]),
                              "worse_class": int(argmax[image, record["worse"]])})
        self._pending_grad_audit = {"pairs": pairs, "selected_tokens": selected, "total": int(total)}

    def _observe_pool_gradients(self, grad):
        """Observe the real backward pass; facts only, the harness asserts the values.

        Only unselected pool tokens are inspected: no native loss reaches them, so every
        nonzero entry there is the term's own and must be one of the declared
        (token, argmax-class) entries with value -/+1/N before the engine's .5 factor.
        """
        pending = self._pending_grad_audit
        audit = {"finite": bool(torch.isfinite(grad).all()), "shape": list(grad.shape),
                 "unselected_nonzero": [], "unselected_nonzero_truncated": False,
                 "expected_entries": []}
        if pending is not None and audit["finite"]:
            count = max(1, pending["total"])
            for pair in pending["pairs"]:
                if not pair["active"]:
                    continue
                audit["expected_entries"].append([pair["image"], pair["better"],
                                                 pair["better_class"], -1.0 / count])
                audit["expected_entries"].append([pair["image"], pair["worse"],
                                                 pair["worse_class"], +1.0 / count])
            view = grad.detach()
            for image in range(view.shape[0]):
                row = view[image]
                tokens = [token for token in range(row.shape[0])
                          if token not in pending["selected_tokens"]]
                if not tokens:
                    continue
                sub_rows = row.index_select(0, torch.tensor(tokens, dtype=torch.int64))
                nonzero = torch.nonzero(sub_rows.abs() > 0.0)
                audit["unselected_nonzero_truncated"] = bool(nonzero.shape[0] > 64)
                for entry in nonzero[:64].tolist():
                    token, cls = tokens[entry[0]], entry[1]
                    audit["unselected_nonzero"].append([int(image), int(token), int(cls),
                                                        float(sub_rows[entry[0], entry[1]])])
        self.last_grad_audit = audit
        self.grad_audits.append(audit)
        return None

    @staticmethod
    def _pairs_from_diagnostics(diagnostics):
        _require(diagnostics, "no matching diagnostics were recorded")
        final = [row for row in diagnostics if row.get("head") == "final"]
        _require(len(final) == 1 and "matched_indices" in final[0],
                 "the final family's matched indices were not recorded")
        return [[(int(q), int(g)) for q, g in zip(row[0], row[1])]
                for row in final[0]["matched_indices"]]

    def _pair_image(self, pool_logits, pool_coord, topk, pairs, gt_boxes, *, activate):
        """Eligibility, exclusivity and partner choice for one image (masks under no_grad).

        Returns (records, info): one record per matched ground truth, partner None when the
        eligible set is empty.  ``activate`` converts the unactivated production pool
        coordinates with sigmoid; the synthetic suite feeds boxes that are already in
        normalized cxcywh space and passes False.  ``pairs`` are owner pool tokens:
        the caller maps the matcher's matched queries through the cached top-k indices.
        """
        constants = DECLARATION["constants"]
        scores = max_class_statistic(pool_logits)
        token_count, gt_count = pool_logits.shape[0], gt_boxes.shape[0]
        with torch.no_grad():
            rank = ranks_of(scores.detach())
            selected = torch.zeros(token_count, dtype=torch.bool)
            selected[topk[:self.k]] = True
            matched_tokens = torch.zeros(token_count, dtype=torch.bool)
            owner_of_gt = {}
            for token, gt in pairs:
                token = int(token)
                matched_tokens[token] = True
                owner_of_gt[gt] = token
            coords = pool_coord.detach()
            if activate:
                coords = torch.sigmoid(coords)
            pool_xyxy = _xyxy(coords.detach())
            gt_xyxy = _xyxy(gt_boxes.detach())
            quality = iou_matrix(pool_xyxy, gt_xyxy)                    # (T, G)
            if self._consumer is not None:
                self._consumer.tensor("rank/quality", quality)
            areas_pool = (coords[:, 2] * coords[:, 3]).clamp_min(1e-12).detach()
            size_error = torch.empty(token_count, gt_count)
            for g in range(gt_count):
                area_g = float(gt_boxes[g, 2] * gt_boxes[g, 3])
                _require(area_g > 0, "ground-truth box has no area")
                size_error[:, g] = (areas_pool / area_g).sqrt().log().abs()
            base = (~selected) & (~matched_tokens) & (rank <= int(math.ceil(constants["alpha"] * self.k)))
            eligible = torch.zeros(token_count, gt_count, dtype=torch.bool)
            for g in range(gt_count):
                if g not in owner_of_gt:
                    continue
                others = torch.cat([quality[:, :g], quality[:, g + 1:]], dim=1)
                protected = ((others.max(dim=1).values <= constants["tau"]) if others.numel()
                             else torch.ones(token_count, dtype=torch.bool))
                owner_quality = float(quality[owner_of_gt[g], g])
                eligible[:, g] = (base & protected
                                  & (size_error[:, g] <= constants["B"])
                                  & (quality[:, g] >= constants["q_floor"])
                                  & (quality[:, g] > owner_quality + constants["q_margin"]))
            # exclusivity: a token partners at most one instance (smallest size error, then smallest GT index)
            masked_error = torch.where(eligible, size_error, torch.full_like(size_error, float("inf")))
            best = masked_error.min(dim=1)
            keeps = torch.zeros_like(eligible)
            for g in range(gt_count):
                keeps[:, g] = (eligible[:, g] & (best.values < float("inf"))
                               & (best.indices == g) & (masked_error[:, g] == best.values))
            eligible_lists = {int(g): torch.nonzero(keeps[:, g], as_tuple=False).flatten().tolist()
                              for g in sorted(owner_of_gt)}
            eligible_counts = {g: len(tokens) for g, tokens in eligible_lists.items()}
            choice_space = sum(1 for tokens in eligible_lists.values() if len(tokens) >= 2)
        records = []
        for g in sorted(owner_of_gt):
            owner_token = int(owner_of_gt[g])
            tokens = eligible_lists[int(g)]
            margin = _margin(float(math.sqrt(float(gt_boxes[g, 2] * gt_boxes[g, 3]))), constants)
            record = {"gt": int(g), "owner": owner_token, "partner": None, "better": None,
                      "worse": owner_token, "eligible_count": len(tokens), "margin": margin,
                      "score_gap": None, "quality_better": None,
                      "quality_worse": float(quality[owner_token, g]), "active": False}
            if tokens:
                if self.mode == "A2":
                    partner = min(tokens, key=lambda t: (-float(quality[t, g]), t))
                else:
                    draw = int(torch.randint(len(tokens), (1,), generator=self.generator).item())
                    self.draws += 1
                    partner = tokens[draw]
                _require(float(quality[partner, g]) > float(quality[owner_token, g])
                         + constants["q_margin"], "partner is not measurably better than the owner")
                gap = float(scores[partner] - scores[owner_token])
                record.update({"partner": int(partner), "better": int(partner),
                               "score_gap": gap, "quality_better": float(quality[partner, g]),
                               "active": bool(gap < margin)})
            records.append(record)
        info = {"gt": int(gt_count), "instances": len(owner_of_gt),
                "pairs": sum(1 for record in records if record["partner"] is not None),
                "choice_space": choice_space, "eligible_counts": eligible_counts,
                "eligible": eligible_lists,
                "owners": {int(g): int(token) for g, token in owner_of_gt.items()}}
        return records, info

    # ---- reporting -------------------------------------------------------------------
    def summary(self):
        stats = dict(self.last_stats)
        instances = max(1, stats.get("instances", 0))
        return {"mode": self.mode, "instances": stats.get("instances", 0),
                "pairs": stats.get("pairs", 0), "active_pairs": stats.get("active_pairs", 0),
                "zero_pair_fraction": round(stats.get("zero_pair_instances", 0) / instances, 6),
                "choice_space_fraction": round(stats.get("choice_space_instances", 0) / instances, 6),
                "loss": stats.get("loss", 0.0), "milliseconds": stats.get("milliseconds", 0.0),
                "draws": self.draws}


def _cached_select(decoder, term, memory, outputs_logits, outputs_coords_unact, topk):
    """Decoder selection wrapper: capture the pool tensors, then run the native selection."""
    original = term._original_select
    _require(original is not None, "term holds no original selection callable")
    indices = torch.topk(outputs_logits.max(dim=-1).values, topk, dim=1).indices
    gathered = original(memory, outputs_logits, outputs_coords_unact, topk)
    term._capture(outputs_logits, outputs_coords_unact, indices)
    return gathered


# ---------------------------------------------------------------------------------
# r14 synthetic-suite compatibility layer.  Both adapters delegate to
# ``SelectionRankTerm._pair_image``, so the 10 r14 checks exercise the production
# pairing core instead of a copy.  They pass ``activate=False`` because the synthetic
# scenes supply normalized cxcywh boxes directly.

def _synthetic_term(mode, k, generator):
    term = SelectionRankTerm(mode, seed=0, k=k)
    if generator is not None:
        term.generator = generator
    return term


def pairing(logits, pred_boxes, gt_boxes, matched, *, k, mode, generator=None):
    """r14 entry point: one image's (tokens, classes) logits; returns records + eligible sets."""
    _require(logits.dim() == 2, "pairing takes a single image's (tokens, classes) logits")
    term = _synthetic_term(mode, k, generator)
    scores = max_class_statistic(logits)
    topk = torch.topk(scores.detach(), k, dim=0).indices
    records, info = term._pair_image(logits, pred_boxes, topk,
                                     [(int(q), int(g)) for q, g in matched], gt_boxes, activate=False)
    return {"mode": mode, "k": k, "records": records, "eligible": info["eligible"],
            "eligible_counts": info["eligible_counts"], "pairs": info["pairs"],
            "active_pairs": sum(1 for record in records if record["active"])}


def loss(images_logits, pred_boxes, gt_boxes, matched, *, k, mode, generator=None):
    """r14 entry point: the (value, evidence) pair with production normalization."""
    _require(images_logits.dim() == 3, "loss takes (images, tokens, classes) logits")
    term = _synthetic_term(mode, k, generator)
    terms = []
    total = 0
    per_image = []
    for image in range(images_logits.shape[0]):
        scores = max_class_statistic(images_logits[image])
        topk = torch.topk(scores.detach(), k, dim=0).indices
        pairs = [(int(q), int(g)) for q, g in matched[image]]
        if gt_boxes[image].numel() == 0 or not pairs:
            per_image.append({"records": [], "pairs": 0, "active_pairs": 0})
            continue
        records, info = term._pair_image(images_logits[image], pred_boxes[image], topk, pairs,
                                         gt_boxes[image], activate=False)
        info.pop("eligible", None)
        info["records"] = records
        info["active_pairs"] = 0
        for record in records:
            if record["partner"] is None:
                continue
            total += 1
            if record["active"]:
                info["active_pairs"] += 1
                margin = torch.as_tensor(record["margin"], dtype=scores.dtype, device=scores.device)
                terms.append(torch.relu(margin - (scores[record["better"]] - scores[record["worse"]])))
        per_image.append(info)
    value = torch.stack(terms).sum() / float(total) if terms else images_logits.sum() * 0.0
    touched = set()
    for info in per_image:
        for record in info["records"]:
            if record["partner"] is not None:
                touched.add(record["better"])
                touched.add(record["worse"])
    evidence = {"mode": mode, "pairs": total, "k": k,
                "active_pairs": sum(info["active_pairs"] for info in per_image),
                "tokens_with_added_gradient": len(touched), "declaration": DECLARATION,
                "per_image": per_image}
    return value, evidence
