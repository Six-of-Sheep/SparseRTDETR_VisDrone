"""Suppressor attribution for the frozen classwise hard NMS.

Pure numpy binary64 helpers. The mirror loop repeats the frozen
``slice_core.classwise_nms`` arithmetic step by step and only adds bookkeeping:
for every removed candidate it records the pivot (a kept box) that removed it.
No torch, no model, no scoring; GT enters only through explicit arguments.
Counts are descriptive. They are not AP, recall, recoverable TP or a method claim.
"""
from __future__ import annotations
from collections import Counter
import numpy as np

MODULE_VERSION = 'nms-suppressor-attribution-v1'
BOUNDARY_EPS = 1e-9
CATEGORIES = ('K1_mid', 'K1_low', 'K2', 'K3')
FATES = ('in_final', 'class100_removed', 'ignore_removed', 'budget_removed')


def sort_key(candidate):
    """Frozen order: score descending, then view, query, category ascending."""
    return (-candidate.score, candidate.view_id, candidate.query_id, candidate.category_id)


def _iou_rows(boxes, areas, first, rest):
    left = np.maximum(boxes[first, :2], boxes[rest, :2])
    right = np.minimum(boxes[first, 2:], boxes[rest, 2:])
    extent = np.maximum(0.0, right - left)
    intersection = extent[:, 0] * extent[:, 1]
    union = areas[first] + areas[rest] - intersection
    if not (np.isfinite(intersection).all() and np.isfinite(union).all() and (union > 0).all()):
        raise ValueError('NMS intersection/union is not finite and valid')
    iou = intersection / union
    if not (np.isfinite(iou).all() and (iou >= 0).all() and (iou <= 1).all()):
        raise ValueError('NMS IoU outside [0,1]')
    return iou


def classwise_nms_with_suppressor(candidates, nms_iou):
    """Return (kept, suppressed, stats).

    kept: candidates in the frozen output order.
    suppressed: removed identity -> {'suppressor': pivot identity, 'iou': float, 'step': int}.
    The suppressor is the first pivot in frozen order whose IoU with the candidate is
    strictly above nms_iou. A removed box never becomes a pivot, so no chain exists.
    Exact score ties follow (view, query, category) ascending, as in the frozen code.
    """
    ordered = sorted(candidates, key=sort_key)
    identities = [c.identity for c in ordered]
    if len(set(identities)) != len(identities):
        raise ValueError('duplicate candidate identity')
    by_class = {}
    for index, candidate in enumerate(ordered):
        by_class.setdefault(candidate.category_id, []).append(index)
    kept_indices, suppressed = set(), {}
    comparisons = boundary = 0
    for indices in by_class.values():
        boxes = np.asarray([ordered[i].box_xyxy for i in indices], dtype=np.float64)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        remaining = np.arange(len(indices), dtype=np.int64)
        step = 0
        while remaining.size:
            first = int(remaining[0])
            kept_indices.add(indices[first])
            rest = remaining[1:]
            if not rest.size:
                break
            iou = _iou_rows(boxes, areas, first, rest)
            comparisons += int(rest.size)
            boundary += int(np.sum(np.abs(iou - nms_iou) <= BOUNDARY_EPS))
            removed_mask = iou > nms_iou
            for local, value in zip(rest[removed_mask].tolist(), iou[removed_mask].tolist()):
                suppressed[identities[indices[local]]] = {
                    'suppressor': identities[indices[first]], 'iou': value, 'step': step}
            remaining = rest[iou <= nms_iou]
            step += 1
    kept = [c for i, c in enumerate(ordered) if i in kept_indices]
    if len(kept) + len(suppressed) != len(ordered):
        raise ValueError('kept/suppressed partition incomplete')
    kept_set = {c.identity for c in kept}
    if any(v['suppressor'] not in kept_set or k in kept_set for k, v in suppressed.items()):
        raise ValueError('suppressor is not a kept box, or a removed box was kept')
    return kept, suppressed, {'comparisons': comparisons, 'boundary_within_eps': boundary,
                              'boundary_eps': BOUNDARY_EPS, 'nms_iou': nms_iou}


def fate(identity, membership, budget):
    """Where a kept box ends for one budget. membership: stage name -> set(identity)."""
    b = str(budget)
    if identity in membership['cap' + b]:
        return 'in_final'
    if identity in membership['filtered' + b]:
        return 'class100_removed'
    if identity in membership['top' + b]:
        return 'ignore_removed'
    return 'budget_removed'


def argmax_ties(values):
    """Indices attaining the exact maximum (binary64 equality); ([], None) when empty."""
    if not len(values):
        return [], None
    best = float(max(values))
    return [i for i, v in enumerate(values) if float(v) == best], best


def category(gt_in_ties, best):
    """K1: the suppressor's best-matched GT is this GT (mid: >= .5, low: < .5).
    K2: best-matched is another GT with IoU >= .5. K3: no GT reaches .5 or no overlap."""
    if best is None or best <= 0:
        return 'K3'
    if gt_in_ties:
        return 'K1_mid' if best >= .5 else 'K1_low'
    return 'K2' if best >= .5 else 'K3'


def shared_counts(lost_by_gt):
    """Candidate identity -> number of GTs (same image, same tau) it supports at >= tau."""
    counter = Counter()
    for identities in lost_by_gt.values():
        counter.update(identities)
    return counter


def attribute_gt(gt, lost, suppressed, same_class_gts, tau, membership, ious):
    """One GT record.

    gt: {'id','box'}; lost: candidates of the GT class with IoU >= tau at the pool stage,
    all of which must have been removed by NMS (bucket precondition, enforced).
    suppressed: output of classwise_nms_with_suppressor with 'suppressor_box' and
    'suppressor_score' added by the caller. same_class_gts: [{'id','box'}] non-crowd GT of
    the same class, any size. membership: stage -> set(identity) for both budgets.
    ious: c_stage_core.ious (binary64, raises on invalid geometry).
    """
    gt_boxes = [g['box'] for g in same_class_gts]
    gt_ids = [g['id'] for g in same_class_gts]
    if gt['id'] not in gt_ids:
        raise ValueError('GT missing from its own class list')
    if not lost:
        raise ValueError('no lost candidate; bucket precondition violated')
    records, suppressors = [], Counter()
    for c in lost:
        entry = suppressed.get(c.identity)
        if entry is None:
            raise ValueError('lost candidate survived NMS; bucket precondition violated')
        values = ious(entry['suppressor_box'], gt_boxes)
        ties, best = argmax_ties(values)
        tie_ids = [gt_ids[i] for i in ties] if best is not None and best > 0 else []
        gt_iou = float(values[gt_ids.index(gt['id'])])
        if gt_iou >= tau:
            raise ValueError('suppressor supports the GT at tau; bucket precondition violated')
        kind = category(gt['id'] in tie_ids, best)
        suppressors[entry['suppressor']] += 1
        records.append({
            'candidate': list(c.identity), 'candidate_rank': c.view_candidate_rank,
            'candidate_score': c.score,
            'candidate_gt_iou': float(ious(gt['box'], [c.box_xyxy])[0]),
            'suppressor': list(entry['suppressor']), 'suppressor_score': entry['suppressor_score'],
            'suppressor_candidate_iou': entry['iou'], 'suppressor_gt_iou': gt_iou,
            'same_view': c.view_id == entry['suppressor'][0],
            'suppressor_argmax_gt': tie_ids, 'suppressor_argmax_iou': best,
            'suppressor_argmax_tie': len(tie_ids) > 1,
            'suppressor_gts_ge_half': int(np.sum(np.asarray(values) >= .5)),
            'suppressor_gts_ge_tau': int(np.sum(np.asarray(values) >= tau)),
            'category': kind,
            'suppressor_fate': {str(b): fate(entry['suppressor'], membership, b) for b in (600, 300)},
        })
    kinds = sorted({r['category'] for r in records})
    return {'lost_count': len(records),
            'lost_by_view': {str(k): v for k, v in sorted(Counter(r['candidate'][0] for r in records).items())},
            'distinct_suppressors': len(suppressors), 'single_suppressor': len(suppressors) == 1,
            'categories': kinds, 'mixed': len(kinds) > 1,
            'same_view_count': int(sum(r['same_view'] for r in records)), 'candidates': records}
