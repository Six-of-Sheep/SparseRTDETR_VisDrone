"""r23 baseline mechanism: a mirrored-time-augmentation view with geometric pairing and box averaging.

Positioning.  This module is deliberately a pragmatic baseline, not an original mechanism: mirrored
test-time augmentation and box fusion are established practice, and this line having not tried them
before is not novelty.  What it does is fixed and small: one extra forward pass runs the SAME frozen
weights on the horizontally mirrored, resize-only canvas; the mirrored boxes are mapped back by
cx -> 1 - cx; each query is paired with the mirrored query of highest box overlap; and when the
ORIGINAL box is small (pixel area < 1024) and that overlap reaches 0.5, the query's box becomes the
plain arithmetic mean of the two.  Everything else of the certified endpoint is untouched: the names,
their scores and the certified ranking are unchanged by construction (the box never enters the certified
ordering), and no ground truth is read anywhere in this module.

Vocabulary.  The pairing is a geometric operation defined by box overlap only; it makes no claim that
the two boxes belong to the same object, and the two views share one set of weights, so no claim of
statistical independence is made either.  The mode "second" pairs with the second-highest-overlap
partner on the same treated set: it is the second-highest-overlap partner sensitivity control.
"""
from __future__ import annotations

import numpy as np
import torch

NUM_QUERIES = 300
NUM_CLASSES = 10
SMALL_AREA = 1024.0
PARTNER_IOU = 0.5
PAIRING_MODES = ("best", "second")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def mirror_boxes(boxes):
    """Map normalized cxcywh boxes of the mirrored canvas back: cx' = 1 - cx, all else equal."""
    require(boxes.shape[-1] == 4, "expected four coordinates")
    out = boxes.clone()
    out[..., 0] = 1.0 - out[..., 0]
    return out


def cxcywh_to_xyxy(boxes):
    return torch.cat((boxes[..., :2] - boxes[..., 2:] / 2,
                      boxes[..., :2] + boxes[..., 2:] / 2), -1)


def xyxy_to_cxcywh(boxes):
    return torch.cat(((boxes[..., :2] + boxes[..., 2:]) / 2,
                      boxes[..., 2:] - boxes[..., :2]), -1)


def pixel_boxes(boxes, width, height):
    """The certified mapping: normalized corners times [width, height] (rank_candidates convention)."""
    scale = torch.tensor([float(width), float(height)] * 2, dtype=boxes.dtype)
    return cxcywh_to_xyxy(boxes) * scale


def iou_matrix(boxes_a, boxes_b):
    """(N,4) against (M,4) pixel xyxy boxes, one full overlap matrix."""
    lt = torch.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    rb = torch.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    wh = (rb - lt).clamp(min=0.0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = ((boxes_a[:, 2] - boxes_a[:, 0]).clamp(min=0.0)
              * (boxes_a[:, 3] - boxes_a[:, 1]).clamp(min=0.0))[:, None]
    area_b = ((boxes_b[:, 2] - boxes_b[:, 0]).clamp(min=0.0)
              * (boxes_b[:, 3] - boxes_b[:, 1]).clamp(min=0.0))[None, :]
    return inter / (area_a + area_b - inter).clamp(min=1e-12)


def fuse_view_boxes(boxes_original, boxes_mirrored, width, height, mode="best",
                    partner_iou=PARTNER_IOU, small_area=SMALL_AREA):
    """Fuse the small-band queries of one image with their highest-overlap mirrored partner.

    `boxes_original` are the certified normalized cxcywh boxes; `boxes_mirrored` are the mirrored
    pass's normalized boxes ALREADY mapped back by `mirror_boxes`.  Returns the fused normalized
    boxes (every untreated row bitwise identical to the input, by plain assignment), the treated
    boolean mask, and JSON-friendly counters.  The treated set depends only on the original box area
    and the highest overlap; mode "second" shares it exactly and only swaps the partner box.
    """
    require(boxes_original.shape[-1] == 4 and boxes_mirrored.shape[-1] == 4,
            "expected four coordinates")
    require(boxes_original.shape == boxes_mirrored.shape, "the two views have different shapes")
    require(mode in PAIRING_MODES, "unknown pairing mode %r" % (mode,))
    with torch.no_grad():
        original = pixel_boxes(boxes_original, width, height)
        mirrored = pixel_boxes(boxes_mirrored, width, height)
        agreement = iou_matrix(original, mirrored)
        ranked = torch.topk(agreement, 2, dim=1)
        best_iou, best_index = ranked.values[:, 0], ranked.indices[:, 0]
        second_iou = ranked.values[:, 1]
        partner_index = best_index if mode == "best" else ranked.indices[:, 1]
        area = ((original[:, 2] - original[:, 0]).clamp(min=0.0)
                * (original[:, 3] - original[:, 1]).clamp(min=0.0))
        small = area < float(small_area)
        treated = small & (best_iou >= float(partner_iou))
        fused = boxes_original.clone()
        if bool(treated.any()):
            mean_pixel = (original[treated] + mirrored[partner_index[treated]]) / 2.0
            scale = torch.tensor([float(width), float(height)] * 2, dtype=boxes_original.dtype)
            fused[treated] = xyxy_to_cxcywh(mean_pixel / scale)
        counters = {
            "queries": int(boxes_original.shape[0]),
            "small_band_queries": int(small.sum()),
            "treated_queries": int(treated.sum()),
            "partner_iou_of_treated": [float(v) for v in best_iou[treated]],
            "second_iou_of_treated": [float(v) for v in second_iou[treated]],
            "queries_with_overlap_tie": int(((agreement == best_iou[:, None]).sum(dim=1) > 1).sum()),
        }
    return fused, treated, counters


def per_class_macro_metrics(eval_result):
    """Class-equal-weight aggregation, the form of the certified published block.

    COCO's own `stats` average over (threshold, recall, class) cells, which weights classes with more
    valid recall bins more heavily; the certified endpoint's published block averages per class
    instead.  The definition is the same one the r22 module carries, restated here so this release is
    self-contained (the r22 overlay is not part of the r23 base).
    """
    precision = eval_result["precision"]
    recall = eval_result["recall"]
    num_class = precision.shape[2]

    def cls_precision(threshold, area, max_det):
        values = []
        for klass in range(num_class):
            cell = precision[threshold, :, klass, area, max_det]
            cell = cell[cell > -1]
            if len(cell):
                values.append(float(cell.mean()))
        return float(np.mean(values)) if values else -1.0

    def cls_recall(area, max_det):
        values = []
        for klass in range(recall.shape[1]):
            cell = recall[:, klass, area, max_det]
            cell = cell[cell > -1]
            if len(cell):
                values.append(float(cell.mean()))
        return float(np.mean(values)) if values else -1.0

    whole = slice(None)
    return {
        "AP": cls_precision(whole, 0, 2), "AP50": cls_precision(0, 0, 2),
        "AP75": cls_precision(5, 0, 2), "APS": cls_precision(whole, 1, 2),
        "APM": cls_precision(whole, 2, 2), "APL": cls_precision(whole, 3, 2),
        "AP75_small": cls_precision(5, 1, 2),
        "AR1": cls_recall(0, 0), "AR10": cls_recall(0, 1), "AR100": cls_recall(0, 2),
        "AR_small": cls_recall(1, 2), "AR_medium": cls_recall(2, 2), "AR_large": cls_recall(3, 2),
    }


def per_class_values(eval_result, area=0, max_det=2):
    """Per-class AP at (all thresholds, given area/maxDets), for the decomposition report."""
    precision = eval_result["precision"]
    out = []
    for klass in range(precision.shape[2]):
        cell = precision[:, :, klass, area, max_det]
        cell = cell[cell > -1]
        out.append(float(cell.mean()) if len(cell) else float("nan"))
    return out
