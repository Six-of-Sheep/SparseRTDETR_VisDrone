"""Geometry availability, not AP, recall, TIDE attribution or a deployable oracle."""
import numpy as np

STAGES = ('raw_geometry', 'view_class_pool', 'hard_nms', 'top_budget',
          'raw_ignore', 'class100')
BUCKETS = ('no_raw_geometry', 'view_class_selection', 'hard_nms_loss',
           'budget_loss', 'ignore_loss', 'class100_loss', 'available')


def ious(gt, boxes):
    gt = np.asarray(gt, dtype=np.float64)
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if gt.shape != (4,) or not np.isfinite(gt).all() or np.any(gt[2:] <= gt[:2]):
        raise ValueError('invalid GT')
    if not np.isfinite(boxes).all() or np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError('invalid candidate')
    ext = np.maximum(0., np.minimum(gt[2:], boxes[:, 2:]) - np.maximum(gt[:2], boxes[:, :2]))
    inter = ext[:, 0] * ext[:, 1]
    area = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
    union = np.prod(gt[2:] - gt[:2]) + area - inter
    if not np.isfinite(union).all() or np.any(union <= 0):
        raise ValueError('invalid union')
    return inter / union


def availability(gt, stage_boxes, thresholds=(.5, .75)):
    if tuple(stage_boxes) != STAGES:
        raise ValueError('stage order differs')
    maxima = [float(ious(gt, stage_boxes[s]).max(initial=0.)) for s in STAGES]
    # Every stage is a subset of the preceding geometry. No tolerance hides a bug.
    if any(b > a for a, b in zip(maxima, maxima[1:])):
        raise ValueError('non-nested stage maxima')
    buckets = {}
    for t in thresholds:
        if not 0 < t <= 1:
            raise ValueError('invalid IoU threshold')
        lost = next((i for i, v in enumerate(maxima) if v < t), len(STAGES))
        buckets[str(t)] = BUCKETS[lost]
    return {'max_iou': dict(zip(STAGES, maxima)), 'first_unavailable': buckets}


def class_cap(candidates, limit=100):
    """Input must already have the certified deterministic score order."""
    seen, kept = {}, []
    for c in candidates:
        cat = c.category_id
        seen[cat] = seen.get(cat, 0) + 1
        if seen[cat] <= limit:
            kept.append(c)
    return kept
