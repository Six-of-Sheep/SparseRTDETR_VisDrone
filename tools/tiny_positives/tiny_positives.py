"""Tiny-band extra positive regression supervision (r19 round).

Problem.  The D1 diagnostic (EXPLORATION_R1.md section 4) measured the single-view tiny-object
gap in this carrier: 57.9 percent of small ground-truth boxes (raw canvas area <= 1024 px^2)
have no candidate box reaching IoU 0.75 anywhere in the pool, the small-band matcher
matched-pair overlap is the most compressed of the three bands at the group median (0.7126
against 0.8823 medium and 0.9346 large; this is a between-group median comparison, not a
per-instance claim), and among the 300 selected tokens 64 percent carry no unique match while
a matched instance has on average 1.82 selected tokens at IoU >= 0.5 near it.  The one-to-one
Hungarian matching therefore gives every tiny ground-truth box exactly ONE regression positive.  Two consecutive rounds reshaped what that single pair's penalty is (r18 geometry
replacement) or ranked the near-cut partners (r17 ranking term); both left the number of
regression positives unchanged and both returned no net benefit.  The unaddressed quantity is
the NUMBER of regression positives per tiny ground-truth box.

Mechanism (declared, zero parameters, zero new loss keys, zero extra forward, inference
untouched).  Inside every non-denoising ``loss_boxes`` call, the positive set handed to the
native loss is augmented with ONE extra pair per tiny-band ground-truth box:

    extra predicted box = the UNMATCHED query of that image whose predicted box has the
                          highest IoU with that ground-truth box, admitted only when
                          IoU >= IOU_TAU (0.5) and the ground-truth box is that query's own
                          best-overlapping ground-truth box;
    extra target        = that ground-truth box;
    no admissible query -> no extra pair (recorded).

The augmented set is passed to the frozen native ``loss_boxes`` unchanged, so the extra pair
contributes the native L1 and native 1 - GIoU at the native per-pair scale (sum / num_boxes)
under the native keys with the native weights.  ``num_boxes``, the VFL classification term,
the matcher, the denoising families, the anchors, the selection rule and the inference path are
untouched bit for bit.  The extra pair is REGRESSION-ONLY: its classification target stays
native, so the one-to-one score semantics (and therefore the NMS-free output) is unchanged, and
the direct beneficiary (the sibling box) is normally discarded at inference.  The mechanism's
testable in-training prediction is therefore that the band's MATCHED pair (the token that does
carry the detection) improves; the audit rows record the band's matched-pair IoU sum, the
extra-pair IoU sum and both extra terms' magnitudes on every criterion call.

Control (arm A1, the native matched-pair duplicated-regression control).  A1 shares A2's
admission gate exactly: for every tiny-band ground-truth box that admits a legitimate extra
box, A1 appends that box's OWN native matched pair instead, so on the same outputs the
appended instance set, its count, the normalisation (num_boxes) and the loss coefficients are
identical to A2, and neither arm appends when the gate finds no candidate.  The contrast
therefore compares supervising an additional real, geometrically supported box against
duplicating the supervision of the original matched box; it does NOT claim equal loss values,
gradient magnitudes or regression quality between the arms.  Registered fact: the appended box
under A2 is an unmatched query that keeps its native classification objective (background);
the mechanism adds and changes no classification supervision, and the appended instance itself
is not trained to become a detection.

Observation (all arms, including the unarmed native arm): per criterion call the term family
records ``pos_pairs``, ``pos_band_gt``, ``pos_band_pairs``, ``pos_band_zero_iou``,
``pos_band_native_iou_sum``, ``pos_extra_candidates`` (admissible instances before the
one-per-box pick), ``pos_extra_pairs``, ``pos_band_gt_no_sibling``, ``pos_extra_no_native``,
``pos_extra_iou_sum``, ``pos_extra_l1_sum``, ``pos_extra_giou_sum`` and
``pos_extra_set_digest`` into the frozen diagnostics rows (value-only, no-grad; the arithmetic
is untouched).  Count semantics: every ``pos_*`` count is a matched-pair/instance OBSERVATION
count per family and per criterion call (one ground-truth box contributes once per family per
call), not an independent ground-truth count; after the training fork each arm's counts come
from its own run and are reported separately.
"""
from __future__ import annotations

import hashlib
import json

import torch

OBJECTIVE = "tiny_band_positive_multiplicity_v1"
CANVAS_HW = (768, 1344)
BAND_AREA_PX = 1024.0
BAND_AREA_THRESHOLD = BAND_AREA_PX / float(CANVAS_HW[0] * CANVAS_HW[1])
# Pre-registered admission bar for an extra positive: the standard 0.5 overlap bar, applied to
# the UNMATCHED query's predicted box against the tiny ground-truth box, with the additional
# requirement that this ground-truth box is that query's own best-overlapping partner.
IOU_TAU = 0.5
EXTRAS_PER_GT = 1
REQUIRE_BEST_GT = True


def constants():
    return {"objective": OBJECTIVE, "canvas_hw": list(CANVAS_HW), "band_area_px": BAND_AREA_PX,
            "band_area_threshold": BAND_AREA_THRESHOLD, "iou_tau": IOU_TAU,
            "extras_per_gt": EXTRAS_PER_GT, "require_best_gt": REQUIRE_BEST_GT}


def constants_sha256():
    body = json.dumps(constants(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(body).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _xyxy(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)


def band_mask(target):
    return target[..., 2] * target[..., 3] <= BAND_AREA_THRESHOLD


def band_gt(targets):
    """Number of ground-truth boxes inside the declared tiny band for one batch."""
    return int(sum(int(band_mask(t["boxes"]).sum().item()) for t in targets))


def _safe_iou(module, pred, target):
    if pred.shape[0] == 0 or target.shape[0] == 0:
        return torch.zeros((pred.shape[0], target.shape[0]), dtype=pred.dtype, device=pred.device)
    iou, _ = module.box_iou(_xyxy(pred.detach()), _xyxy(target.detach()))
    return iou


def _pair_terms(pred, target, module):
    """Native per-pair regression terms for aligned pairs (value only, detached)."""
    with torch.no_grad():
        l1 = (pred.detach() - target.detach()).abs().sum(-1)
        giou = 1.0 - torch.diag(module.generalized_box_iou(_xyxy(pred.detach()), _xyxy(target.detach())))
    return l1, giou


def observation(pred_pairs, target_pairs, module, band_gt_count):
    """Value-only band statistics for one family call; never touches arithmetic."""
    band = band_mask(target_pairs)
    stats = {"pos_pairs": int(pred_pairs.shape[0]), "pos_band_gt": int(band_gt_count),
             "pos_band_pairs": int(band.sum().item()), "pos_band_zero_iou": 0,
             "pos_band_native_iou_sum": 0.0, "pos_band_native_l1_sum": 0.0,
             "pos_band_native_giou_sum": 0.0}
    if pred_pairs.shape[0] == 0 or not bool(band.any()):
        return stats
    with torch.no_grad():
        iou, _ = module.box_iou(_xyxy(pred_pairs.detach()), _xyxy(target_pairs.detach()))
        pair_iou = torch.diag(iou)
    l1, giou = _pair_terms(pred_pairs, target_pairs, module)
    stats["pos_band_zero_iou"] = int(((pair_iou <= 0.0) & band).sum().item())
    stats["pos_band_native_iou_sum"] = float(pair_iou[band].sum().item())
    stats["pos_band_native_l1_sum"] = float(l1[band].sum().item())
    stats["pos_band_native_giou_sum"] = float(giou[band].sum().item())
    return stats


def extra_pairs(mode, pred, target, matched_src, matched_tgt, module):
    """The declared extra positive pairs for one image; deterministic, no RNG.

    Both armed modes share the SAME admission gate, computed once here: a tiny-band
    ground-truth box qualifies when at least one unmatched query reaches IOU_TAU with it and
    this ground-truth box is that query's own best-overlapping partner.

    A2: append the qualifying query with the highest overlap with that ground-truth box.
    A1 (the native matched-pair duplicated-regression control): append that ground-truth box's
    OWN native matched pair every time the gate qualifies it; the appended set, its count, the
    normalisation and the loss coefficients are therefore identical to A2 on the same outputs,
    and neither arm appends anything when the gate finds no candidate.  A1's appended instances
    carry the native pair's own (possibly low) regression quality; this control does not claim
    equal loss values, gradient magnitudes or regression quality against A2.

    Returns (src_idx, tgt_idx, ledger) with int64 index tensors into this image's queries.
    """
    require(mode in ("A1", "A2"), "unknown tiny-positives mode: " + str(mode))
    device = pred.device
    empty = torch.zeros(0, dtype=torch.int64, device=device)
    band = band_mask(target)
    ledger = {"pos_extra_pairs": 0, "pos_extra_candidates": 0, "pos_band_gt_no_sibling": 0,
              "pos_extra_no_native": 0, "pos_extra_iou_sum": 0.0, "pos_extra_l1_sum": 0.0,
              "pos_extra_giou_sum": 0.0,
              "pos_extra_set_digest": hashlib.sha256(b"").hexdigest()[:16],
              "pos_band_gt": int(band.sum().item())}
    src, tgt = empty, empty
    if pred.shape[0] and target.shape[0] and bool(band.any()):
        iou = _safe_iou(module, pred, target)
        best_gt = iou.argmax(dim=1)
        unmatched = torch.ones(pred.shape[0], dtype=torch.bool, device=device)
        unmatched[matched_src] = False
        native_of = {int(t): int(s) for s, t in zip(matched_src.tolist(), matched_tgt.tolist())} \
            if mode == "A1" else {}
        picks, targets, appended = [], [], []
        for j in torch.nonzero(band, as_tuple=False).flatten().tolist():
            eligible = unmatched & (iou[:, j] >= IOU_TAU)
            if REQUIRE_BEST_GT:
                eligible = eligible & (best_gt == j)
            ledger["pos_extra_candidates"] += int(eligible.sum().item())
            if not bool(eligible.any()):
                ledger["pos_band_gt_no_sibling"] += 1
                continue
            if mode == "A1":
                native = native_of.get(j)
                if native is None:
                    # structurally unreachable under the frozen one-to-one matcher (every
                    # ground truth carries exactly one native pair); recorded, never silently
                    # dropped
                    ledger["pos_extra_no_native"] += 1
                    continue
                picks.append(native)
            else:
                rows = torch.nonzero(eligible, as_tuple=False).flatten()
                picks.append(int(rows[int(iou[rows, j].argmax().item())]))
            targets.append(j)
            appended.append(j)
        if picks:
            src = torch.tensor(picks, dtype=torch.int64, device=device)
            tgt = torch.tensor(targets, dtype=torch.int64, device=device)
        ledger["pos_extra_set_digest"] = hashlib.sha256(
            ",".join(str(j) for j in appended).encode()).hexdigest()[:16]
    if src.numel():
        iou_e = iou[src, tgt]
        l1, giou = _pair_terms(pred[src], target[tgt], module)
        ledger.update({"pos_extra_pairs": int(src.numel()),
                       "pos_extra_iou_sum": float(iou_e.sum().item()),
                       "pos_extra_l1_sum": float(l1.sum().item()),
                       "pos_extra_giou_sum": float(giou.sum().item())})
    return src, tgt, ledger


def _combine_digest(left, right):
    if left is None:
        return right
    if right is None:
        return left
    return hashlib.sha256(("%s|%s" % (left, right)).encode()).hexdigest()[:16]


def accumulate(ledger, row):
    """Sum one image's ledger into an accumulated ledger; the set digest combines pair-wise."""
    if ledger is None:
        return dict(row)
    out = {}
    for key in ledger:
        if key == "pos_extra_set_digest":
            out[key] = _combine_digest(ledger[key], row[key])
        else:
            out[key] = ledger[key] + row[key]
    return out


def pair_tensors(outputs, targets, indices):
    """The native matched pairs as (pred, target) tensors gathered image by image."""
    pred = torch.cat([outputs["pred_boxes"][i][src] for i, (src, _) in enumerate(indices)]) \
        if indices else outputs["pred_boxes"].reshape(-1, 4)[:0]
    target = torch.cat([targets[i]["boxes"][tgt] for i, (_, tgt) in enumerate(indices)]) \
        if indices else outputs["pred_boxes"].reshape(-1, 4)[:0]
    return pred, target


class TinyPositivesTerm:
    """One arm's term; holds no parameter and no buffer, only a deterministic pair counter."""

    def __init__(self, mode):
        require(mode in ("A1", "A2"), "unknown tiny-positives mode: " + str(mode))
        self.mode = mode
        self.extras = 0
        self.last_stats = {}

    def __deepcopy__(self, memo):
        clone = TinyPositivesTerm(self.mode)
        clone.extras = self.extras
        clone.last_stats = dict(self.last_stats)
        return clone

    def state_dict(self):
        return {"mode": self.mode, "extras": int(self.extras)}

    def load_state_dict(self, state):
        require(isinstance(state, dict) and state.get("mode") == self.mode,
                "tiny-positives term identity differs")
        self.extras = int(state["extras"])
        return {"extras": self.extras}

    def loss_boxes(self, base, module, outputs, targets, indices, num_boxes, boxes_weight,
                   family="final", sink=None):
        """The armed ``loss_boxes``: the native loss over the augmented positive set.

        The extra pairs are appended to the frozen matching's index lists and the frozen native
        ``loss_boxes`` is called once on the augmented lists, so every extra pair carries the
        native terms at the native per-pair scale and nothing else changes.
        """
        require(boxes_weight is None, "the frozen chain declares no per-pair box weight")
        augmented = []
        ledger = None
        extras_total = 0
        for i, (src, tgt) in enumerate(indices):
            esrc, etgt, row = extra_pairs(self.mode, outputs["pred_boxes"][i], targets[i]["boxes"],
                                          matched_src=src, matched_tgt=tgt, module=module)
            extras_total += int(esrc.numel())
            ledger = accumulate(ledger, row)
            if esrc.numel():
                # Hungarian indices are normally CPU int64; extras follow pred.device.
                # Preserve the native index representation and all selected values/order.
                esrc = esrc.to(device=src.device, dtype=src.dtype)
                etgt = etgt.to(device=tgt.device, dtype=tgt.dtype)
                augmented.append((torch.cat([src, esrc]), torch.cat([tgt, etgt])))
            else:
                augmented.append((src, tgt))
        losses = base(outputs, targets, augmented, num_boxes, boxes_weight)
        pred_pairs, target_pairs = pair_tensors(outputs, targets, indices)
        stats = observation(pred_pairs, target_pairs, module, band_gt(targets))
        stats.update(ledger or {})
        stats["pos_extra_mode"] = self.mode
        self.extras += extras_total
        self.last_stats = {"mode": self.mode, "family": family, "extras_total": extras_total, **stats}
        if sink is not None:
            sink[family] = dict(stats)
        return losses


def observe_call(pred_pairs, target_pairs, module, band_gt_count, family="final", sink=None):
    """The unarmed observation path used by the patched criterion (native arms)."""
    stats = observation(pred_pairs, target_pairs, module, band_gt_count)
    stats.update({"pos_extra_pairs": 0, "pos_extra_candidates": 0, "pos_band_gt_no_sibling": 0,
                  "pos_extra_no_native": 0, "pos_extra_iou_sum": 0.0, "pos_extra_l1_sum": 0.0,
                  "pos_extra_giou_sum": 0.0, "pos_extra_set_digest": None, "pos_extra_mode": None})
    if sink is not None:
        sink[family] = dict(stats)
    return stats
