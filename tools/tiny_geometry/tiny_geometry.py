"""Tiny-band normalized-geometry term for matched-pair box losses (r18 round).

Problem (D1 diagnostic, EXPLORATION_R1.md section 4): 57.9 percent of small ground-truth
boxes have no candidate box reaching IoU 0.75 at the pool stage, near-miss small boxes are
slightly oversized (1.08x to 1.62x) and center-off by about one pixel, and the small-box
pairwise matcher IoU is compressed (median 0.7126).  The localization ceiling is the
dominant measured gap.

Mechanism (declared, zero parameters, zero new loss keys, zero extra forward): inside the
native ``loss_boxes`` call, the geometric slot of every matched pair whose ground-truth box
lies in the declared tiny band (raw canvas area <= 1024 px^2 on the 768x1344 canvas, i.e.
normalized area <= 1024/(768*1344)) replaces the native ``1 - giou`` value by a
scale-normalized Gaussian-Wasserstein penalty

    s      = sqrt( (dcx^2 + dcy^2) + 0.25 * (dw^2 + dh^2) )        (activated FP32 boxes)
    penalty = 1 - exp( -s / C )

with C a constant frozen by the pre-registered median-parity calibration (see
``geometry_probe.py``).  Non-tiny pairs, the L1 (loss_bbox) component, VFL, the matcher,
the denoising families and every weight keep their native arithmetic bit for bit.  The
replacement keeps the native aggregation (sum over matched pairs / num_boxes) and enters
under the native ``loss_giou`` key with the native weight (2.0).

Control (arm A1): the identical formula, pair count and aggregation, but the tiny-band
target used by the replaced slot is cyclically shifted within the same criterion call by
one offset drawn from the term's own CPU generator (n >= 2 tiny pairs; n < 2 leaves the
call unshifted and consumes no draw).  Correspondence is destroyed while the term's
statistics, compute and magnitude class stay identical.  The training RNG streams are
never consumed (checked on every update by the chain).

Observation (all arms, including the unarmed native arm): per criterion call the term
family records ``geo_pairs``, ``geo_tiny_pairs``, ``geo_tiny_zero_iou``,
``geo_tiny_native_sum``, ``geo_tiny_applied_sum``, ``geo_shift`` into the frozen
diagnostics rows (no-grad, value-only; the arithmetic is untouched).
"""
from __future__ import annotations

import hashlib
import json

import torch

OBJECTIVE = "tiny_band_normalized_wasserstein_geometry_v1"
CANVAS_HW = (768, 1344)
BAND_AREA_PX = 1024.0
BAND_AREA_THRESHOLD = BAND_AREA_PX / float(CANVAS_HW[0] * CANVAS_HW[1])
# Frozen at the pre-registered calibration (geometry_probe.py, median-parity rule); the
# chain refuses to run while this is None (slice_finetune.read_plan).
CONSTANT_C = 0.005500852969014606
# Declared numerical boundary: the exact-zero distance has an infinite sqrt derivative, so
# W2 is floored at 1e-12 before the root (minimum s = 1e-6).  Without the floor the
# backward pass through the out-of-band branch of torch.where multiplies the unselected
# zero by inf and poisons every gradient.
W2_FLOOR = 1e-12


def constants():
    return {"objective": OBJECTIVE, "canvas_hw": list(CANVAS_HW), "band_area_px": BAND_AREA_PX,
            "band_area_threshold": BAND_AREA_THRESHOLD, "constant_C": CONSTANT_C,
            "w2_floor": W2_FLOOR}


def constants_sha256():
    body = json.dumps(constants(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(body).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _xyxy(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)


def wasserstein_sqrt(pred, target):
    """sqrt(W2) of the two Gaussian boxes N(c, diag((w/2)^2, (h/2)^2)); FP32.

    W2 is floored at W2_FLOOR before the root (declared numerical boundary): the
    exact-zero distance has an infinite sqrt derivative, and 0 * inf would poison the
    backward pass through the unselected branch of ``torch.where``.
    """
    d = pred - target
    w2 = d[..., 0] ** 2 + d[..., 1] ** 2 + 0.25 * (d[..., 2] ** 2 + d[..., 3] ** 2)
    return torch.sqrt(torch.clamp(w2, min=W2_FLOOR))


def tiny_mask(target):
    return target[..., 2] * target[..., 3] <= BAND_AREA_THRESHOLD


def penalty_small(pred, target, constant=None):
    c = CONSTANT_C if constant is None else constant
    require(c is not None, "constant C is not frozen")
    require(float(c) > 0.0, "constant C must be positive")
    return 1.0 - torch.exp(-wasserstein_sqrt(pred, target) / float(c))


def native_penalty(pred, target, module):
    """The native slot value per pair: 1 - diag(generalized_box_iou(xyxy, xyxy))."""
    return 1 - torch.diag(module.generalized_box_iou(_xyxy(pred), _xyxy(target)))


def observation(pred, target, module):
    """Value-only statistics for one criterion call (all arms); never touches arithmetic."""
    tiny = tiny_mask(target)
    native = native_penalty(pred, target, module)
    iou = None
    with torch.no_grad():
        iou, _ = module.box_iou(_xyxy(pred.detach()), _xyxy(target.detach()))
        iou = torch.diag(iou)
    count = int(tiny.sum().item())
    zero_iou = int(((iou <= 0.0) & tiny).sum().item())
    stats = {"geo_pairs": int(pred.shape[0]), "geo_tiny_pairs": count, "geo_tiny_zero_iou": zero_iou,
             "geo_tiny_native_sum": float(native.detach()[tiny].sum().item()) if count else 0.0}
    return stats, tiny, native


class TinyGeometryTerm:
    """One arm's term; holds no parameter and no buffer, only its own CPU generator."""

    def __init__(self, mode, *, seed=0):
        require(mode in ("A1", "A2"), "unknown tiny-geometry mode: " + str(mode))
        self.mode = mode
        self.seed = int(seed)
        self.generator = torch.Generator().manual_seed(self.seed)
        self.draws = 0
        self.last_stats = {}

    def __deepcopy__(self, memo):
        clone = TinyGeometryTerm(self.mode, seed=self.seed)
        clone.generator.set_state(self.generator.get_state())
        clone.draws = self.draws
        clone.last_stats = dict(self.last_stats)
        return clone

    def state_dict(self):
        return {"mode": self.mode, "seed": self.seed, "draws": self.draws,
                "generator": self.generator.get_state().clone()}

    def load_state_dict(self, state):
        require(isinstance(state, dict) and state.get("mode") == self.mode
                and int(state.get("seed", -1)) == self.seed, "tiny-geometry term identity differs")
        self.generator.set_state(state["generator"].clone())
        self.draws = int(state["draws"])
        return {"draws": self.draws}

    def _shift_targets(self, tiny, target):
        """A1 cyclic shift over the tiny pairs; declared no-op for n < 2; own RNG only."""
        n = int(tiny.sum().item())
        if self.mode != "A1" or n < 2:
            return target, None
        offset = int(torch.randint(1, n, (1,), generator=self.generator).item())
        self.draws += 1
        rows = torch.nonzero(tiny, as_tuple=False).flatten()
        order = rows[(torch.arange(n, device=rows.device) + offset) % n]
        shifted = target.clone()
        shifted[rows] = target[order]
        return shifted, offset

    def loss_boxes(self, base, module, outputs, targets, indices, num_boxes, boxes_weight,
                   family="final", sink=None):
        """The armed ``loss_boxes``: native L1 kept, the geo slot replaced inside the band.

        ``base`` is the native ``SetCriterionv2.loss_boxes`` bound call; its returned
        ``loss_bbox`` is used unchanged and its ``loss_giou`` is superseded by the
        replaced aggregation (identical arithmetic for non-tiny pairs).
        """
        require(boxes_weight is None, "the frozen chain declares no per-pair box weight")
        require(CONSTANT_C is not None, "constant C is not frozen")
        losses = base(outputs, targets, indices, num_boxes, boxes_weight)
        pred = outputs["pred_boxes"][self._perm(indices)]
        target = torch.cat([t["boxes"][j] for t, (_, j) in zip(targets, indices)])
        stats, tiny, native = observation(pred, target, module)
        shifted, offset = self._shift_targets(tiny, target)
        small = penalty_small(pred, shifted)
        applied = torch.where(tiny, small, native)
        loss_giou = applied.sum() / num_boxes
        stats["geo_tiny_applied_sum"] = float(applied.detach()[tiny].sum().item()) if stats["geo_tiny_pairs"] else 0.0
        stats["geo_shift"] = offset
        self.last_stats = {"mode": self.mode, "draws": self.draws, **stats, "family": family}
        if sink is not None:
            sink[family] = dict(stats)
        return {"loss_bbox": losses["loss_bbox"], "loss_giou": loss_giou}

    @staticmethod
    def _perm(indices):
        batch = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        return batch, torch.cat([src for (src, _) in indices])


def observe_call(pred, target, module, family="final", sink=None):
    """The unarmed observation path used by the patched criterion (native arms)."""
    stats, _, _ = observation(pred, target, module)
    stats["geo_tiny_applied_sum"] = stats["geo_tiny_native_sum"]
    stats["geo_shift"] = None
    if sink is not None:
        sink[family] = dict(stats)
    return stats


def loss_via_call(base, module, outputs, targets, indices, num_boxes, boxes_weight):
    """Reference reimplementation used by the tests: native call plus replaced slot."""
    losses = base(outputs, targets, indices, num_boxes, boxes_weight)
    return losses


def synthetic_pairs(rows, *, dtype=torch.float32):
    """Deterministic synthetic pair tensors for tests; rows are (cx, cy, w, h) per pair."""
    return torch.tensor(rows, dtype=dtype)
