"""Fixed Gaussian Soft-NMS. No model, data, GT, CLI or mutable global state."""
from dataclasses import replace
import math
import numpy as np
from slice_core import Candidate, _validated_candidates

SIGMA = 0.5
VERSION = 'gaussian-softnms-continuous-fp64-v1'
METRICS = ('AP', 'APS', 'AP75_small', 'APM', 'APL', 'AP50', 'AP75')
LIMITS = {
    'D-C': {'600': {'AP': .20, 'APS': .30, 'AP75_small': 0., 'APM': -.20, 'APL': -.20},
            '300': {'AP': -.20, 'APS': -.20}},
    'D-A': {'600': {'AP': .30, 'APS': .50, 'AP75_small': -.20, 'APM': -.50, 'APL': -.50},
            '300': {'AP': -.20, 'APS': -.20}},
}


def gaussian_soft_nms(candidates):
    items = sorted(_validated_candidates(candidates), key=lambda c: c.identity)
    output = []
    zero_input = sum(c.score == 0 for c in items)
    underflow = 0
    by_class = {}
    for c in items:
        by_class.setdefault(c.category_id, []).append(c)
    for values in by_class.values():
        boxes = np.asarray([c.box_xyxy for c in values], dtype=np.float64)
        scores = np.asarray([c.score for c in values], dtype=np.float64)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        remaining = np.arange(len(values))
        while remaining.size:
            # remaining stays in identity order; argmax implements the tie rule.
            pos = int(np.argmax(scores[remaining]))
            selected = int(remaining[pos])
            output.append(replace(values[selected], score=float(scores[selected])))
            remaining = np.delete(remaining, pos)
            if not remaining.size:
                break
            left = np.maximum(boxes[selected, :2], boxes[remaining, :2])
            right = np.minimum(boxes[selected, 2:], boxes[remaining, 2:])
            extent = np.maximum(0., right - left)
            intersection = extent[:, 0] * extent[:, 1]
            union = areas[selected] + areas[remaining] - intersection
            if not (np.isfinite(union).all() and (union > 0).all() and np.isfinite(intersection).all()):
                raise ValueError('Soft-NMS invalid IoU denominator/intersection')
            iou = intersection / union
            if not (np.isfinite(iou).all() and (iou >= 0).all() and (iou <= 1).all()):
                raise ValueError('Soft-NMS invalid IoU')
            before = scores[remaining].copy()
            # Underflow is explicitly allowed as zero; NaN/Inf are never repaired.
            with np.errstate(under='ignore', over='raise', invalid='raise', divide='raise'):
                scores[remaining] *= np.exp(-(iou * iou) / SIGMA)
            after = scores[remaining]
            if not (np.isfinite(after).all() and (after >= 0).all() and (after <= before).all()):
                raise ValueError('Soft-NMS invalid cumulative score')
            underflow += int(np.count_nonzero((before > 0) & (after == 0)))
    output.sort(key=lambda c: c.sort_key)
    if len(output) != len(items):
        raise ValueError('Soft-NMS must retain all candidate identities')
    # Counts cross the strict JSON boundary; NumPy comparisons can yield np.int64.
    stats = {'candidate_count': int(len(items)), 'input_zero_scores': int(zero_input),
             'underflow_to_zero': int(underflow), 'output_zero_scores': int(sum(c.score == 0 for c in output)),
             'subnormal_scores': int(sum(0 < c.score < np.finfo(np.float64).tiny for c in output)),
             'sigma': SIGMA, 'version': VERSION, 'score_cutoff': None}
    return {'top600': output[:600], 'top300': output[:300], 'all': output, 'stats': stats}


def compare_scores(scores):
    for arm in ('A', 'B', 'C', 'D', 'E'):
        for budget in ('600', '300'):
            for metric in METRICS:
                value = scores[arm][budget][metric]
                if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError('metric missing/nonfinite/outside original 0..1 units')
    differences = {str(b): {pair: {m: (scores[pair[0]][str(b)][m] - scores[pair[2]][str(b)][m]) * 100.
                   for m in METRICS} for pair in ('D-C', 'D-A', 'E-B', 'D-E')} for b in (600, 300)}
    gates = {pair: {b: {m: {'difference_pp': differences[b][pair][m], 'minimum_pp': limit,
                          'passed': differences[b][pair][m] >= limit}
                        for m, limit in limits.items()} for b, limits in budgets.items()}
             for pair, budgets in LIMITS.items()}
    passed = all(v['passed'] for p in gates.values() for b in p.values() for v in b.values())
    return {'differences_pp': differences, 'gates': gates, 'quality_passed': passed,
            'end_to_end_cost': 'NOT_MEASURED', 'units': 'percentage points; converted once'}
