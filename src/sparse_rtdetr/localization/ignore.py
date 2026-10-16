"""Continuous union/IoA masking of unmatched negative classification only."""
from __future__ import annotations

import numpy as np
import torch

from .distribution import box_to_corners, require_finite
from .consumer_geometry import corners,tensor


def disjoint_union(rectangles):
    """Exact axis-aligned rectangle union partition; no rasterization for training."""
    rects = np.asarray(rectangles, dtype=np.float64).reshape(-1, 4)
    if not np.isfinite(rects).all() or np.any(rects[:, 2:] <= rects[:, :2]):
        raise ValueError("invalid ignore rectangles")
    if not len(rects):
        return np.empty((0, 4), dtype=np.float64)
    xs = np.unique(rects[:, [0, 2]])
    output = []
    for left, right in zip(xs[:-1], xs[1:]):
        candidates = rects[(rects[:, 0] < right) & (rects[:, 2] > left)]
        intervals = sorted((r[1], r[3]) for r in candidates)
        merged = []
        for low, high in intervals:
            if merged and low <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], high)
            else:
                merged.append([low, high])
        output.extend([left, low, right, high] for low, high in merged)
    return np.asarray(output, dtype=np.float64).reshape(-1, 4)


def intersections(boxes, rectangles):
    low = torch.maximum(boxes[:, None, :2], rectangles[None, :, :2])
    high = torch.minimum(boxes[:, None, 2:], rectangles[None, :, 2:])
    return (high-low).clamp_min(0).prod(-1)


@torch.no_grad()
def negative_keep_mask(pred_boxes, indices, targets, num_classes=10):
    """All coordinates normalized to canvas; prediction denominator is NOT clipped."""
    boxes = box_to_corners(pred_boxes.detach().double())
    require_finite(boxes, "classification mask boxes")
    area = (boxes[..., 2:]-boxes[..., :2]).prod(-1)
    corners("ignore/predicted",boxes)
    if bool(((boxes[..., 2:]-boxes[..., :2]) <= 0).any()):
        raise ValueError("negative mask requires full positive-area predicted boxes")
    keep = torch.ones((*boxes.shape[:2], num_classes), dtype=torch.bool, device=boxes.device)
    for batch, target in enumerate(targets):
        union = target.get("ignore_union", boxes.new_empty((0, 4))).to(boxes.device, torch.float64)
        require_finite(union, "ignore union")
        if union.numel() and bool(((union[:, 2:] <= union[:, :2]) | (union[:, :2] < 0) | (union[:, 2:] > 1)).any()):
            raise ValueError("ignore union outside positive clipped canvas rectangles")
        if union.numel():
            inter = intersections(boxes[batch], union)
            ratio = inter.sum(-1) / area[batch]
            tensor("ignore/union_intersection",inter,False);tensor("ignore/union_ioa",ratio,False)
            mask = ratio >= .5
            keep[batch, mask] = False
        ignored = target.get("ignore_class_boxes", boxes.new_empty((0, 4))).to(boxes.device, torch.float64)
        labels = target.get("ignore_class_labels", torch.empty(0, dtype=torch.long, device=boxes.device)).to(boxes.device)
        require_finite(ignored, "class ignore rectangles")
        if ignored.shape != (len(labels),4) or bool(((labels < 0) | (labels >= num_classes)).any()):
            raise ValueError("class ignore boxes/labels mismatch")
        if ignored.numel() and bool(((ignored[:, 2:] <= ignored[:, :2]) | (ignored[:, :2] < 0) | (ignored[:, 2:] > 1)).any()):
            raise ValueError("class ignore outside positive clipped canvas rectangles")
        if ignored.numel():
            inter = intersections(boxes[batch], ignored)
            ratio = inter / area[batch, :, None]
            tensor("ignore/class_intersection",inter,False);tensor("ignore/class_ioa",ratio,False)
            hit = ratio >= .5
            for label in labels.unique():
                keep[batch, hit[:, labels == label].any(-1), label] = False
        keep[batch, indices[batch][0]] = True
    return keep
