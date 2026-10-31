"""Size-relative Hungarian matching cost for the common-protocol R18 (r12 round prototype).

Mechanism (candidate 1 of the r12 screening; see the single exploration record
``delivery/research/r18-explore-20261031-001/EXPLORATION_R1.md`` section 15):
the one-to-one assignment solved inside the frozen criterion is dominated by the
large instances, because the localization cost is computed in absolute
normalized-canvas units.  For a small ground-truth instance a fixed absolute
error of e.g. 0.01 (normalized) is a whole object width, yet it costs the
Hungarian objective only 0.01; for a large instance the same 0.01 is a small
fraction of the object and equally cheap.  When several instances compete for
nearby tokens (the frozen carrier's D1 diagnostic: 64% of selected slots have no
unique match and matched instances carry 1.82 tokens on average), the objective
therefore allocates tokens by absolute-error magnitude rather than by relative
alignment.

This module replaces the localization term of the matching cost only:

    native :  C[q,g] = w_bbox * L1(cxcywh) + w_class * focal_class + w_giou * (-GIoU)
    mode 'rel' :  L1 is evaluated in the GT's own size units, per GT
                  |dcx|/s_g + |dcy|/s_g + |dw|/w_g + |dh|/h_g ,  s_g = sqrt(w_g h_g)
    mode 'ctrl':  identical arithmetic with the per-GT sizes replaced by the
                  batch means (s_bar, w_bar, h_bar) -- same operator, same compute,
                  no per-instance differentiation (this is the control arm)

The class term and the GIoU term are untouched, the one-to-one structure is
untouched, no parameter is added, nothing is added to the model or to inference,
and the matching runs under ``torch.no_grad()`` exactly as the native one does.
The mechanism acts only through WHICH token each instance owns.

Relation to the nearest precedents (declared honestly; no originality is claimed
for the operator): scale-normalized localization metrics for tiny objects exist
in the anchor-assignment family (e.g. NWD-style normalized distances) and
scale-aware token matching exists in recent DETR variants.  The declared
differences of this round are: (a) the normalization is applied to the
*assignment cost* only, not to the regression loss (the closed rsmall item was a
loss reweighting), (b) the decision that changes is token ownership among
competing instances, keeping the one-to-one structure, and (c) the control arm
applies the same operator with the per-instance differentiation removed, so a
gain that survives the control cannot be attributed to scale relativity.

Installation happens after the frozen parent state is strict-loaded and before
the optimizer/EMA are constructed.  Installing consumes no RNG draw (no
parameter is created), so all arms share one identical data/DN random stream.
``assert_installed`` verifies the wrapper, the declared matching declaration and
the module-level vendor names the criterion's geometry consumer patches.

CPU checks: ``test_matcher_scale.py``, ``matcher_scale_smoke.py`` and the
carrier-side decision diagnostic ``matcher_probe.py``.
"""
from __future__ import annotations

import importlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MODES = ("rel", "ctrl")
NATIVE_DECLARATION = {"matching": "native_own_head_Hungarian", "mode": "off",
                      "normalizer": "none", "operator": "native_l1_cxcywh"}
SCALE_EPS = 1e-9

# Module-level names the criterion's geometry consumer patches/records.  They are
# bound at install time from the vendor matcher module that owns the wrapped
# instance, so ``geo.matching(matcher)`` -- which monkeypatches the globals of
# ``type(matcher).__module__`` -- lands on THIS module and this module's calls are
# the ones it observes.
VENDOR_MODULE = None
generalized_box_iou = None
linear_sum_assignment = None
box_cxcywh_to_xyxy = None

LAST_COST_INFO = {}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def declaration(mode):
    """The declared matching identity stored in the criterion recipe and checkpoint."""
    if mode is None:
        return dict(NATIVE_DECLARATION)
    require(mode in MODES, "unknown matching mode: " + str(mode))
    return {"matching": "scale_relative_localization_v1", "mode": mode,
            "normalizer": "gt_scale" if mode == "rel" else "batch_mean_scale",
            "operator": "l1_relative_per_gt" if mode == "rel" else "l1_relative_batch_mean"}


def _class_cost(out_prob, tgt_ids, alpha, gamma):
    """Native focal-style class cost (verbatim arithmetic of the frozen matcher)."""
    out_prob = out_prob[:, tgt_ids]
    neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
    pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
    return pos_cost_class - neg_cost_class


def cost_terms(outputs, targets, mode, weights):
    """The three native cost matrices plus the final cost, for the declared mode.

    Mirrors the frozen ``HungarianMatcher.forward`` arithmetic line by line; the
    only substitution is the localization term in modes 'rel'/'ctrl'.  Returns
    (cost_class, cost_bbox_term, cost_giou, C, info).  ``mode='off'`` reproduces
    the native expression byte-for-byte on purpose (it is the identity control).
    """
    require(mode in ("off",) + MODES, "unknown matching mode: " + str(mode))
    out_prob = F.sigmoid(outputs["pred_logits"].flatten(0, 1))
    out_bbox = outputs["pred_boxes"].flatten(0, 1)
    tgt_ids = torch.cat([v["labels"] for v in targets])
    tgt_bbox = torch.cat([v["boxes"] for v in targets])
    cost_class = _class_cost(out_prob, tgt_ids, weights["alpha"], weights["gamma"])
    if mode == "off":
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
    else:
        require(tgt_bbox.numel() > 0, "a normalized cost needs at least one GT box")
        delta = (out_bbox[:, None, :] - tgt_bbox[None, :, :]).abs()
        width, height = tgt_bbox[:, 2], tgt_bbox[:, 3]
        if mode == "rel":
            side = (width * height).clamp_min(SCALE_EPS).sqrt()
            divisors = torch.stack([side, side, width, height], dim=-1)
        else:
            side = (width * height).clamp_min(SCALE_EPS).sqrt().mean()
            divisors = torch.stack([side, side, width.mean(), height.mean()]).expand(
                tgt_bbox.shape[0], 4)
        cost_bbox = (delta / divisors.clamp_min(SCALE_EPS)).sum(-1)
    cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
    C = (weights["cost_bbox"] * cost_bbox + weights["cost_class"] * cost_class
         + weights["cost_giou"] * cost_giou)
    info = {"mode": mode, "queries": int(out_bbox.shape[0]), "targets": int(tgt_bbox.shape[0]),
            "cost_bbox_mean": float(cost_bbox.mean()) if cost_bbox.numel() else None}
    if mode != "off" and tgt_bbox.numel():
        info["scale_norm_mean"] = float(divisors[:, 0].mean())
        info["scale_norm_min"] = float(divisors[:, 0].min())
        info["scale_norm_max"] = float(divisors[:, 0].max())
    return cost_class, cost_bbox, cost_giou, C, info


class ScaledHungarianMatcher(nn.Module):
    """Wrap a frozen ``HungarianMatcher`` instance; same call contract, same weights.

    The native cost attributes are re-exposed unchanged, so the criterion recipe's
    matcher fields are identical to the frozen ones for every mode.
    """

    def __init__(self, base, mode):
        super().__init__()
        require(mode in MODES, "unknown matching mode: " + str(mode))
        require(type(base).__name__ == "HungarianMatcher", "unexpected base matcher type")
        self.mode = mode
        self.base_type = type(base).__module__ + "." + type(base).__name__
        self.cost_class = base.cost_class
        self.cost_bbox = base.cost_bbox
        self.cost_giou = base.cost_giou
        self.alpha = base.alpha
        self.gamma = base.gamma
        self.use_focal_loss = base.use_focal_loss
        require(self.use_focal_loss is True, "this round declares the focal-cost matcher only")

    def weights(self):
        return {"cost_bbox": self.cost_bbox, "cost_class": self.cost_class,
                "cost_giou": self.cost_giou, "alpha": self.alpha, "gamma": self.gamma}

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        cost_class, cost_bbox, cost_giou, C, info = cost_terms(
            {"pred_logits": outputs["pred_logits"], "pred_boxes": outputs["pred_boxes"]},
            targets, self.mode, self.weights())
        info["batch"] = int(bs)
        LAST_COST_INFO.clear()
        LAST_COST_INFO.update(info)
        C = C.view(bs, num_queries, -1).cpu()
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
                   for i, j in indices]
        return {"indices": indices}


def bind_vendor_names(module):
    """Bind the vendor names this module calls as globals (see the module docstring)."""
    global VENDOR_MODULE, generalized_box_iou, linear_sum_assignment, box_cxcywh_to_xyxy
    require(hasattr(module, "generalized_box_iou") and hasattr(module, "linear_sum_assignment")
            and hasattr(module, "box_cxcywh_to_xyxy"),
            "vendor matcher module lacks the patched names: " + str(module))
    VENDOR_MODULE = module
    generalized_box_iou = module.generalized_box_iou
    linear_sum_assignment = module.linear_sum_assignment
    box_cxcywh_to_xyxy = module.box_cxcywh_to_xyxy
    return module


def matching_declaration(criterion):
    """The criterion's declared matching identity (native declaration when not installed).

    Used by the criterion recipe, so the declaration travels inside every saved
    checkpoint's behaviour identity.
    """
    return dict(getattr(criterion, "matching_declaration", NATIVE_DECLARATION))


def install_matcher_scale(criterion, mode):
    """Install the size-relative cost on a real criterion; returns the wrapper.

    Must run after the frozen parent state is loaded and before the optimizer/EMA
    are built.  Consumes no RNG draw and adds no parameter.
    """
    global VENDOR_MODULE, generalized_box_iou, linear_sum_assignment, box_cxcywh_to_xyxy
    require(mode in MODES, "unknown matching mode: " + str(mode))
    require(hasattr(criterion, "matcher"), "criterion has no matcher")
    base = criterion.matcher
    require(type(base).__name__ == "HungarianMatcher", "unexpected matcher type: " + str(type(base)))
    require(not isinstance(base, ScaledHungarianMatcher), "matcher scale already installed")
    bind_vendor_names(importlib.import_module(type(base).__module__))
    wrapper = ScaledHungarianMatcher(base, mode)
    criterion.matcher = wrapper
    criterion.matching_declaration = declaration(mode)
    return wrapper


def assert_installed(criterion, mode):
    """Verify the installed wrapper, its declaration and the patched module names."""
    require(getattr(criterion, "matcher", None) is not None, "criterion has no matcher")
    wrapper = criterion.matcher
    require(isinstance(wrapper, ScaledHungarianMatcher), "matcher scale is not installed")
    require(wrapper.mode == mode, "installed matching mode differs")
    require(getattr(criterion, "matching_declaration", None) == declaration(mode),
            "criterion does not declare the installed matching mode")
    require(VENDOR_MODULE is not None and generalized_box_iou is VENDOR_MODULE.generalized_box_iou
            and linear_sum_assignment is VENDOR_MODULE.linear_sum_assignment
            and box_cxcywh_to_xyxy is VENDOR_MODULE.box_cxcywh_to_xyxy,
            "the module-level vendor names the geometry consumer patches were rebound")
    return True


def assignment(outputs, targets, weights, mode):
    """Standalone assignment helper (probe/tests): no module state, no side effects."""
    require(mode in ("off",) + MODES, "unknown matching mode: " + str(mode))
    bs, num_queries = outputs["pred_logits"].shape[:2]
    _, _, _, C, info = cost_terms(outputs, targets, mode, weights)
    C = C.view(bs, num_queries, -1).cpu()
    sizes = [len(v["boxes"]) for v in targets]
    solved = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
    return ([(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
             for i, j in solved], [c[i] for i, c in enumerate(C.split(sizes, -1))], info)


def normalized_scale(gt_boxes):
    """The per-GT scale used by mode 'rel' (normalized units); exposed for reporting."""
    width, height = gt_boxes[:, 2], gt_boxes[:, 3]
    return {"side": (width * height).clamp_min(SCALE_EPS).sqrt(),
            "width": width, "height": height}


def declared_digest():
    """Stable digest of the declared matching identity (used by the checks)."""
    import hashlib
    import json
    return hashlib.sha256(json.dumps(declaration("rel"), sort_keys=True).encode()).hexdigest()
