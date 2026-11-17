"""Band-restricted ROI-resolution box distillation (regression-only, identity-anchored).

Supervision object.  For every tiny-band ground-truth instance of the CURRENT image the
target is the box the SAME frozen weights produce when they re-encode a ROI of that image
at TRUE RAW PIXEL density, mapped back to original coordinates and replayed through the
student's own augmentation.  That is the object the closed r20 round isolated at inference
(A2-A1: +0.2809 APS / +0.3439 AP75_small).  Nothing here consumes the teacher's prediction
SET, its scores, its class logits, or any NMS fusion.

Correspondence.  The extra pair is anchored on the student's OWN native ground-truth
assignment: the query the frozen matcher already assigned to that ground-truth row receives
one additional regression-only pair whose target is that instance's ROI box.  There is no
second matching, no auxiliary query group and no new query.

Learning path.  The pair is appended to the frozen matching index lists inside ``loss_boxes``
only, so it carries the native L1 and native 1-GIoU at the native per-pair scale under the
native keys (``loss_bbox`` / ``loss_giou``) and the native normaliser.  Classification, VFL,
the matcher, the DN families, anchors, selection and inference are untouched; zero
parameters, zero new loss keys.

Control.  Mode R1 shares the identical pipeline, window plan, counts, normaliser and keys and
differs only in the rendered pixels: its window is enlarged from the already-seen canvas, so
it cannot carry new information.  R2 is the method.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import torch

OBJECTIVE = "band_roi_resolution_box_distill_v1"
CANVAS_HW = (768, 1344)
BAND_AREA_PX = 1024.0
BAND_AREA_THRESHOLD = BAND_AREA_PX / float(CANVAS_HW[0] * CANVAS_HW[1])
# Pre-registered admission bar for an instance ROI target: the ROI box must carry the SAME
# class as the ground-truth row and reach this overlap with it.  One target at most per row.
TARGET_IOU = 0.3
TARGETS_PER_GT = 1
INPUT_HW = (768, 1344)
# The declared view group of each armed mode: 0 = the whole-image view (no new information),
# 1 = the four raw-pixel tiles (the method).  Both arms attach both groups and read the same
# tags; only this constant differs, so the target-box provenance is the single variable.
GROUP = {"R1": 0, "R2": 1}


def constants():
    return {"objective": OBJECTIVE, "canvas_hw": list(CANVAS_HW), "band_area_px": BAND_AREA_PX,
            "band_area_threshold": BAND_AREA_THRESHOLD, "target_iou": TARGET_IOU,
            "targets_per_gt": TARGETS_PER_GT, "input_hw": list(INPUT_HW)}


def constants_sha256():
    body = json.dumps(constants(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(body).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _xyxy(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)


def band_mask(boxes):
    """Canvas-normalised cxcywh boxes inside the tiny band (area <= 1024 px on 768x1344)."""
    return boxes[..., 2] * boxes[..., 3] <= BAND_AREA_THRESHOLD


def band_gt(targets):
    """Number of ground-truth rows inside the declared tiny band for one batch."""
    return int(sum(int(band_mask(t["boxes"]).sum().item()) for t in targets))


def _safe_iou(module, first, second):
    if first.shape[0] == 0 or second.shape[0] == 0:
        return torch.zeros((first.shape[0], second.shape[0]), dtype=first.dtype, device=first.device)
    iou, _ = module.box_iou(_xyxy(first.detach()), _xyxy(second.detach()))
    return iou


def _pair_terms(pred, target, module):
    """Native per-pair regression terms (value only, detached)."""
    with torch.no_grad():
        l1 = (pred.detach() - target.detach()).abs().sum(-1)
        giou = 1.0 - torch.diag(module.generalized_box_iou(_xyxy(pred.detach()), _xyxy(target.detach())))
    return l1, giou


def instance_targets(target, matched_src, matched_tgt, module, group):
    """Choose the ROI box target for every band ground-truth row of one image.

    ``target["roi_boxes"]`` carries every aligned ROI box (canvas-normalised cxcywh),
    ``target["roi_rows"]`` its source ground-truth row slot and ``target["roi_group"]`` the
    view group it came from: 0 for the whole-image view, 1 for the raw-pixel tiles.  Both
    armed arms attach the same boxes and read the same group tags, and differ only in
    ``group`` - which group's box is used as the target.

    The admitted set is defined by COMMON SUPPORT so the two arms supervise the same pairs and
    the same number of pairs: a row qualifies only when the declared group's best same-class
    box and the other group's best same-class box BOTH reach TARGET_IOU.  The target box is the
    declared group's best; ties break on higher overlap then lower ROI box index.

    Returns (src_idx, box_index, boxes_tensor, ledger) with ``box_index`` into ``roi_boxes``.
    """
    boxes = target["boxes"]
    labels = target["labels"]
    band = band_mask(boxes)
    ledger = {"roi_band_gt": int(band.sum().item()), "roi_eligible": 0, "roi_selected": 0,
              "roi_no_candidate": 0, "roi_no_class": 0, "roi_below_iou": 0,
              "roi_one_sided": 0, "roi_unmatched_band_gt": 0, "roi_target_iou_sum": 0.0,
              "roi_other_group_iou_sum": 0.0,
              "roi_set_digest": hashlib.sha256(b"").hexdigest()[:16]}
    device = boxes.device
    empty = torch.zeros(0, dtype=torch.int64, device=device)
    src, picks = empty, empty
    roi_boxes = target.get("roi_boxes")
    if roi_boxes is None or roi_boxes.numel() == 0 or not bool(band.any()):
        ledger["roi_no_candidate"] = int(band.sum().item())
        return src, picks, None, ledger
    roi_boxes = roi_boxes.to(device=device, dtype=boxes.dtype)
    roi_rows = target["roi_rows"].to(device=device)
    roi_labels = target["roi_labels"].to(device=device)
    roi_group = target["roi_group"].to(device=device)
    require(group in (0, 1), "roi group must be 0 or 1")
    owner = {}
    for s, t in zip(matched_src.tolist(), matched_tgt.tolist()):
        owner[int(t)] = int(s)
    rows, chosen, overlaps, others = [], [], [], []
    for row in torch.nonzero(band, as_tuple=False).flatten().tolist():
        query = owner.get(row)
        if query is None:
            ledger["roi_unmatched_band_gt"] += 1
            continue
        same = torch.nonzero(roi_rows == row, as_tuple=False).flatten()
        if not bool(same.numel()):
            ledger["roi_no_candidate"] += 1
            continue
        same = same[roi_labels[same] == labels[row]]
        if not bool(same.numel()):
            ledger["roi_no_class"] += 1
            continue
        ledger["roi_eligible"] += 1
        mine = same[roi_group[same] == group]
        theirs = same[roi_group[same] != group]
        if not bool(mine.numel()) or not bool(theirs.numel()):
            ledger["roi_no_candidate"] += 1
            continue
        iou_mine = _safe_iou(module, boxes[row].reshape(1, 4), roi_boxes[mine]).reshape(-1)
        iou_theirs = _safe_iou(module, boxes[row].reshape(1, 4), roi_boxes[theirs]).reshape(-1)
        best_mine = int(torch.argsort(iou_mine, descending=True, stable=True)[0].item())
        best_theirs = int(torch.argsort(iou_theirs, descending=True, stable=True)[0].item())
        if float(iou_theirs[best_theirs].item()) < TARGET_IOU:
            ledger["roi_one_sided"] += 1
            continue
        if float(iou_mine[best_mine].item()) < TARGET_IOU:
            ledger["roi_below_iou"] += 1
            continue
        rows.append(query)
        chosen.append(int(mine[best_mine].item()))
        overlaps.append(float(iou_mine[best_mine].item()))
        others.append(float(iou_theirs[best_theirs].item()))
    if rows:
        src = torch.tensor(rows, dtype=torch.int64, device=device)
        picks = torch.tensor(chosen, dtype=torch.int64, device=device)
        ledger["roi_selected"] = int(src.numel())
        ledger["roi_target_iou_sum"] = float(sum(overlaps))
        ledger["roi_other_group_iou_sum"] = float(sum(others))
        ledger["roi_set_digest"] = hashlib.sha256(
            ",".join(str(p) for p in chosen).encode()).hexdigest()[:16]
    return src, picks, roi_boxes, ledger


def _augmented_native_loss(base, outputs, targets, augmented, extra_by_image, roi_boxes,
                           num_boxes, boxes_weight):
    """Native loss over the augmented index lists with explicit ROI target boxes.

    The frozen native ``loss_boxes`` gathers target boxes as ``t["boxes"][j]``.  The extra
    pairs therefore index into a temporary per-image target whose ``boxes`` tensor is the real
    rows followed by this image's ROI boxes; real rows stay bit-identical and the temporary
    list is discarded immediately.  The temporary ``labels`` tensor pads with zeros only to
    satisfy shape checks and is never read for the appended pairs (regression-only).
    """
    if not any(n for _, n in extra_by_image):
        return base(outputs, targets, augmented, num_boxes, boxes_weight)
    patched = []
    for i, target in enumerate(targets):
        entry = dict(target)
        extra = [p for (j, n) in extra_by_image if j == i for p in n]
        if extra:
            boxes = target["boxes"]
            appended = roi_boxes[i][torch.tensor([int(p[1]) for p in extra],
                                                 dtype=torch.int64, device=boxes.device)]
            entry["boxes"] = torch.cat([boxes, appended.to(boxes.dtype)], dim=0)
            labels = target["labels"]
            entry["labels"] = torch.cat([labels, labels.new_zeros(appended.shape[0])], dim=0)
        patched.append(entry)
    return base(outputs, patched, augmented, num_boxes, boxes_weight)


class RoiTerm:
    """One arm's term; holds no parameter and no buffer, only deterministic counters."""

    def __init__(self, mode):
        require(mode in ("R1", "R2"), "unknown roi-ld mode: " + str(mode))
        self.mode = mode
        self.targets = 0
        self.last_stats = {}

    def __deepcopy__(self, memo):
        clone = RoiTerm(self.mode)
        clone.targets = self.targets
        clone.last_stats = dict(self.last_stats)
        return clone

    def state_dict(self):
        return {"mode": self.mode, "targets": int(self.targets)}

    def load_state_dict(self, state):
        require(isinstance(state, dict) and state.get("mode") == self.mode,
                "roi-ld term identity differs")
        self.targets = int(state["targets"])
        return {"targets": self.targets}

    def loss_boxes(self, base, module, outputs, targets, indices, num_boxes, boxes_weight,
                   family="final", sink=None):
        augmented = []
        ledger = None
        total = 0
        extra_by_image = []
        roi_boxes = []
        for i, (src, tgt) in enumerate(indices):
            esrc, epick, boxes, row = instance_targets(
                targets[i], matched_src=src, matched_tgt=tgt, module=module,
                group=GROUP[self.mode])
            ledger = _accumulate(ledger, row)
            roi_boxes.append(torch.zeros(0, dtype=torch.float32, device=src.device)
                             if boxes is None else boxes)
            if esrc.numel():
                esrc = esrc.to(device=src.device, dtype=src.dtype)
                etgt = torch.arange(len(esrc), dtype=tgt.dtype, device=tgt.device) \
                    + len(targets[i]["boxes"])
                augmented.append((torch.cat([src, esrc]), torch.cat([tgt, etgt])))
                extra_by_image.append((i, [(int(a), int(b)) for a, b in zip(esrc.tolist(), epick.tolist())]))
                total += int(esrc.numel())
            else:
                augmented.append((src, tgt))
                extra_by_image.append((i, []))
        losses = _augmented_native_loss(base, outputs, targets, augmented, extra_by_image,
                                        roi_boxes, num_boxes, boxes_weight)
        stats = _observation(outputs, targets, indices, module)
        stats.update(ledger or {})
        stats["roi_mode"] = self.mode
        self.targets += total
        self.last_stats = {"mode": self.mode, "family": family, "roi_targets_total": total, **stats}
        if sink is not None:
            sink[family] = dict(stats)
        return losses


def _observation(outputs, targets, indices, module):
    """Value-only band statistics for one family call; never touches arithmetic."""
    pred, target = _pair_tensors(outputs, targets, indices)
    band = band_mask(target) if target.numel() else torch.zeros(0, dtype=torch.bool)
    stats = {"roi_pos_pairs": int(pred.shape[0]), "roi_band_pairs": int(band.sum().item()),
             "roi_band_native_iou_sum": 0.0, "roi_band_zero_iou": 0}
    if pred.shape[0] == 0 or not bool(band.numel()) or not bool(band.any()):
        return stats
    with torch.no_grad():
        iou, _ = module.box_iou(_xyxy(pred.detach()), _xyxy(target.detach()))
        pair_iou = torch.diag(iou)
    stats["roi_band_native_iou_sum"] = float(pair_iou[band].sum().item())
    stats["roi_band_zero_iou"] = int(((pair_iou <= 0.0) & band).sum().item())
    return stats


def _pair_tensors(outputs, targets, indices):
    if indices:
        pred = torch.cat([outputs["pred_boxes"][i][src] for i, (src, _) in enumerate(indices)])
        target = torch.cat([targets[i]["boxes"][tgt] for i, (_, tgt) in enumerate(indices)])
    else:
        pred = outputs["pred_boxes"].reshape(-1, 4)[:0]
        target = pred
    return pred, target


def _accumulate(ledger, row):
    if ledger is None:
        return dict(row)
    out = {}
    for key in ledger:
        if key == "roi_set_digest":
            left, right = ledger[key], row[key]
            out[key] = hashlib.sha256(("%s|%s" % (left, right)).encode()).hexdigest()[:16]
        else:
            out[key] = ledger[key] + row[key]
    return out


def observe_call(pred_pairs, target_pairs, module, band_gt_count, family="final", sink=None):
    """Unarmed observation path used by the native arm of the patched criterion."""
    band = band_mask(target_pairs) if target_pairs.numel() else torch.zeros(0, dtype=torch.bool)
    stats = {"roi_pos_pairs": int(pred_pairs.shape[0]), "roi_band_gt": int(band_gt_count),
             "roi_band_pairs": int(band.sum().item()), "roi_band_zero_iou": 0,
             "roi_band_native_iou_sum": 0.0, "roi_eligible": 0, "roi_selected": 0,
             "roi_no_candidate": 0, "roi_no_class": 0, "roi_below_iou": 0, "roi_one_sided": 0,
             "roi_unmatched_band_gt": 0, "roi_target_iou_sum": 0.0,
             "roi_other_group_iou_sum": 0.0, "roi_set_digest": None, "roi_mode": None}
    if pred_pairs.shape[0] and target_pairs.numel() and bool(band.any()):
        with torch.no_grad():
            iou, _ = module.box_iou(_xyxy(pred_pairs.detach()), _xyxy(target_pairs.detach()))
            pair_iou = torch.diag(iou)
        stats["roi_band_native_iou_sum"] = float(pair_iou[band].sum().item())
        stats["roi_band_zero_iou"] = int(((pair_iou <= 0.0) & band).sum().item())
    if sink is not None:
        sink[family] = dict(stats)
    return stats


def replay_boxes(candidates, original_hw, trace, target, input_hw=INPUT_HW):
    """Replay the recorded augmentation geometry on original-coordinate boxes.

    Same arithmetic as the closed distillation round's align_teacher: the recorded geometry
    steps (zoom padding, strict-centre crop admission, horizontal flip, resize) are applied
    to each box, boxes that cannot survive are dropped, and the survivors are returned in
    normalised xyxy on the student canvas together with their ground-truth row index.  No
    randomness, and no teacher influence on the augmentation draws.

    The replay is driven by the operations the augmentation actually recorded, so it is valid
    in both the strong (zoom/crop/flip) and the weak regime; the caller is expected to prove
    exactness against the augmented targets rather than to assume one regime.
    """
    if not isinstance(trace.get("geometry_steps"), list):
        raise ValueError("augmentation trace carries no geometry step list")
    if trace.get("image_id") != target["image_id"]:
        raise ValueError("roi teacher/student augmentation image identity differs")
    if tuple(target["original_hw"].tolist()) != tuple(original_hw):
        raise ValueError("roi teacher/student original dimensions differ")
    h, w = original_hw
    rows = []
    for row_index, box in candidates:
        b = np.asarray(box, dtype=np.float64).copy()
        b[[0, 2]] = np.clip(b[[0, 2]], 0, w)
        b[[1, 3]] = np.clip(b[[1, 3]], 0, h)
        if np.any(b[2:] <= b[:2]):
            continue
        hh, ww = h, w
        keep = True
        for step in trace["geometry_steps"]:
            op = step["operation"]
            if op == "zoom":
                left, top, right, bottom = step["padding"]
                b += np.array([left, top] * 2)
                hh += top + bottom
                ww += left + right
            elif op == "crop":
                top, left, height, width = (step[k] for k in ("top", "left", "height", "width"))
                cx, cy = (b[:2] + b[2:]) / 2
                if not (left < cx < left + width and top < cy < top + height):
                    keep = False
                    break
                b -= np.array([left, top] * 2)
                b[[0, 2]] = np.clip(b[[0, 2]], 0, width)
                b[[1, 3]] = np.clip(b[[1, 3]], 0, height)
                hh, ww = height, width
            elif op == "horizontal_flip":
                b[[0, 2]] = ww - b[[2, 0]]
            else:
                raise ValueError("unrecognized actual augmentation step")
        if not keep or np.any(b[2:] - b[:2] < 1):
            continue
        b *= np.array([input_hw[1] / ww, input_hw[0] / hh] * 2)
        if np.any(b[2:] - b[:2] < 1):
            continue
        b /= np.array(input_hw[::-1] * 2)
        rows.append((row_index, b))
    return rows
