"""Diagnostic geometry and fixed-threshold matching, never AP accumulation."""
from __future__ import annotations
import numpy as np


def iou_matrix(boxes, gt_boxes, crowd=None):
    a = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
    for v in (a, b):
        if not np.isfinite(v).all() or np.any(v[:, 2:] <= v[:, :2]):
            raise ValueError('nonfinite/nonpositive geometry')
    wh = np.maximum(0., np.minimum(a[:, None, 2:], b[None, :, 2:])
                    - np.maximum(a[:, None, :2], b[None, :, :2]))
    intersection = wh[:, :, 0] * wh[:, :, 1]
    aa = np.prod(a[:, 2:] - a[:, :2], axis=1)
    bb = np.prod(b[:, 2:] - b[:, :2], axis=1)
    union = aa[:, None] + bb[None, :] - intersection
    if crowd is not None:
        mask = np.asarray(crowd, dtype=bool)
        if mask.shape != (len(b),): raise ValueError('crowd shape')
        union[:, mask] = aa[:, None]
    if not np.isfinite(union).all() or np.any(union <= 0): raise ValueError('invalid union')
    return intersection / union


def pair_match(ious, ignored, crowd, threshold):
    """COCO score-greedy GT matches; row order is fixed prediction score order.

    Return last matched detection per GT. Only nonignored noncrowd columns are
    scientific denominators. GT stable input order and exact IoU tie behavior
    follow evaluateImg. Removing a detection can change subsequent assignments.
    """
    v = np.asarray(ious, dtype=np.float64)
    ignore = np.asarray(ignored, dtype=bool)
    crowd = np.asarray(crowd, dtype=bool)
    if (v.ndim != 2 or v.shape[1] != len(ignore) or crowd.shape != ignore.shape
            or not np.isfinite(v).all() or np.any((v < 0) | (v > 1))
            or threshold not in (.5, .75)):
        raise ValueError('invalid matching inputs')
    matched = np.full(len(ignore), -1, dtype=int)
    for d in range(len(v)):
        eligible = np.flatnonzero((v[d] >= threshold) & ((matched < 0) | crowd))
        if not len(eligible): continue
        regular = eligible[~ignore[eligible]]
        if len(regular): eligible = regular
        # Last equal-IoU GT in stable original order, matching COCO's >= update.
        best_value = v[d, eligible].max()
        best = eligible[np.flatnonzero(v[d, eligible] == best_value)[-1]]
        matched[best] = d
    return matched


def maximum_match(edges):
    """Deterministic maximum cardinality, not a deployable ranking/oracle AP."""
    e = np.asarray(edges, dtype=bool)
    if e.ndim != 2: raise ValueError('edge shape')
    assigned = np.full(e.shape[0], -1, dtype=int)
    adjacency = [np.flatnonzero(e[:, g]).tolist() for g in range(e.shape[1])]
    def visit(g, seen):
        for d in adjacency[g]:
            if d in seen: continue
            seen.add(d)
            if assigned[d] < 0 or visit(int(assigned[d]), seen):
                assigned[d] = g
                return True
        return False
    for g in range(e.shape[1]): visit(g, set())
    return int(np.count_nonzero(assigned >= 0))


def transition(before, after):
    a, b = np.asarray(before, dtype=bool), np.asarray(after, dtype=bool)
    if a.shape != b.shape: raise ValueError('transition shape')
    return {'retained': int(np.count_nonzero(a & b)), 'lost': int(np.count_nonzero(a & ~b)),
            'gained': int(np.count_nonzero(~a & b)), 'neither': int(np.count_nonzero(~a & ~b))}


def cap(candidates, enabled=True):
    seen, out = {}, []
    for c in candidates:
        seen[c.category_id] = seen.get(c.category_id, 0) + 1
        if not enabled or seen[c.category_id] <= 100: out.append(c)
    return out
