"""Synthetic-only checks for an evidence plan; no detector/data/evaluator imports.

The CLI accepts no production input. Graphs describe possible coverage, NOT AP.
"""
import argparse
import json
import math


def iou(a, b):
    for box in (a, b):
        if len(box) != 4 or not all(math.isfinite(x) for x in box):
            raise ValueError('invalid box')
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError('nonpositive box')
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + bb - inter)


def max_cover(edges):
    """Maximum bipartite cardinality: GT index -> distinct query IDs.

    A physical query can cover only one GT, even across categories. This is a
    diagnostic capacity convention, not either frozen evaluator's matching.
    """
    owner = {}

    def augment(g, seen):
        for q in sorted(set(edges[g])):
            if q in seen:
                continue
            seen.add(q)
            if q not in owner or augment(owner[q], seen):
                owner[q] = g
                return True
        return False

    for g in range(len(edges)):
        augment(g, set())
    return len(owner)


def coverage(boxes, gt_boxes, gt_classes, scores, budget, keep=None, threshold=.75):
    """Constructed scores only. Production must reuse native FP32 ranking/ignore.

    keep is an explicitly supplied post-budget query mask, not a new ignore rule.
    Full geometry does not depend on class scores. topK does. All rates use the
    same input GT population, including uncovered GT. No filtering/refill.
    """
    if len(gt_boxes) != len(gt_classes) or len(boxes) != len(scores):
        raise ValueError('shape mismatch')
    nclass = len(scores[0]) if scores else 0
    if any(len(s) != nclass for s in scores) or any(c < 0 or c >= nclass for c in gt_classes):
        raise ValueError('class shape mismatch')
    if any(not math.isfinite(s) or not 0 <= s <= 1 for row in scores for s in row):
        raise ValueError('invalid score')
    if not isinstance(budget, int) or not 0 <= budget <= len(boxes) * nclass:
        raise ValueError('invalid budget')
    if not 0 < threshold <= 1:
        raise ValueError('invalid threshold')
    for b in boxes + gt_boxes:
        iou(b, b)
    keep = [True] * len(boxes) if keep is None else keep
    if len(keep) != len(boxes) or any(type(x) is not bool for x in keep):
        raise ValueError('invalid explicit keep mask')
    order = sorted(((q, c) for q in range(len(boxes)) for c in range(nclass)),
                   key=lambda qc: (-scores[qc[0]][qc[1]], qc[0], qc[1]))
    selected = set(order[:budget])
    geom = [[q for q, b in enumerate(boxes) if iou(b, g) >= threshold] for g in gt_boxes]
    retained = [[q for q in e if (q, c) in selected] for e, c in zip(geom, gt_classes)]
    filtered = [[q for q in e if keep[q]] for e in retained]
    n = len(gt_boxes)
    counts = dict(geometry=max_cover(geom), budget=max_cover(retained), post_ignore=max_cover(filtered))
    independent = dict(geometry=sum(bool(e) for e in geom), budget=sum(bool(e) for e in retained),
                       post_ignore=sum(bool(e) for e in filtered))
    assert 0 <= counts['post_ignore'] <= counts['budget'] <= counts['geometry'] <= n
    return dict(denominator=n, independent=independent, capacity=counts,
                capacity_rates={k: v / n if n else None for k, v in counts.items()},
                geometry_absent=sum(not e for e in geom),
                budget_lost_gt=sum(bool(e) and not r for e, r in zip(geom, retained)),
                ignore_lost_gt=sum(bool(r) and not f for r, f in zip(retained, filtered)))


def boundary_pairs(quality, logits, classes, matched, ranks, budget):
    """Hypothesis only: matched positives straddling a frozen budget boundary.

    Inputs are detached metadata except logits; pure Python cannot train.
    Returns pair IDs and the fixed unit-weight hinge value for algebra checks.
    """
    n = len(quality)
    if any(len(x) != n for x in (logits, classes, matched, ranks)):
        raise ValueError('pair shape mismatch')
    if any(not math.isfinite(t) or not 0 <= t <= 1 for t in quality):
        raise ValueError('quality invalid')
    if any(not math.isfinite(z) for z in logits):
        raise ValueError('logit invalid')
    pairs = [(i, j) for i in range(n) for j in range(n)
             if matched[i] and matched[j] and classes[i] == classes[j]
             and quality[i] >= .75 > quality[j] and ranks[i] > budget >= ranks[j]]
    value = sum(max(0., quality[i] - quality[j] - (logits[i] - logits[j])) for i, j in pairs)
    return pairs, value / len(pairs) if pairs else 0.


def synthetic_checks():
    g = [0., 0., 8., 8.]
    cases = {}
    cases['one_pixel_shift_iou'] = iou(g, [1., 0., 9., 8.])
    assert math.isclose(cases['one_pixel_shift_iou'], 7 / 9)
    assert iou(g, [2., 0., 10., 8.]) == .6
    assert iou(g, [8., 0., 16., 8.]) == 0
    cases['perfect_box_lost_to_budget'] = coverage([g, [20., 20., 28., 28.]], [g], [0], [[.2], [.9]], 1)
    assert cases['perfect_box_lost_to_budget']['capacity'] == dict(geometry=1, budget=0, post_ignore=0)
    cases['geometry_missing'] = coverage([[2., 0., 10., 8.]], [g], [0], [[.99]], 1)
    assert cases['geometry_missing']['geometry_absent'] == 1
    cases['two_GT_share_one_query'] = coverage([g], [g, g], [0, 0], [[.9]], 1)
    assert cases['two_GT_share_one_query']['independent']['geometry'] == 2
    assert cases['two_GT_share_one_query']['capacity']['geometry'] == 1
    cases['no_refill'] = coverage([g, g], [g], [0], [[.9], [.8]], 1, [False, True])
    assert cases['no_refill']['capacity']['post_ignore'] == 0
    cases['wrong_class_budget'] = coverage([g], [g], [0], [[.1, .9]], 1)
    assert cases['wrong_class_budget']['budget_lost_gt'] == 1
    cases['stable_tie'] = coverage([[20., 20., 28., 28.], g], [g], [0], [[.5], [.5]], 1)
    assert cases['stable_tie']['budget_lost_gt'] == 1
    cases['empty_GT'] = coverage([g], [], [], [[.9]], 1)
    assert cases['empty_GT']['capacity_rates']['geometry'] is None
    cases['all_retained_but_ranking_unresolved'] = coverage([g, [20., 20., 28., 28.]], [g], [0], [[.2], [.9]], 2)
    assert cases['all_retained_but_ranking_unresolved']['capacity']['budget'] == 1
    # Capacity unchanged by score inversion when both candidates survive: not AP.
    assert coverage([g, [20., 20., 28., 28.]], [g], [0], [[.9], [.2]], 2)['capacity'] == cases['all_retained_but_ranking_unresolved']['capacity']
    assert max_cover([[0, 1], [0]]) == 2  # Greedy count would miss the augmenting path.
    assert max_cover([[0], [0, 1]]) == 2
    pairs, value = boundary_pairs([.9, .6], [-1., 1.], [0, 0], [True, True], [601, 1], 600)
    assert pairs == [(0, 1)] and math.isclose(value, 2.3)
    assert boundary_pairs([.9, .6], [-1., 1.], [0, 0], [False, True], [601, 1], 600) == ([], 0.)
    assert boundary_pairs([.9, .6], [-1., 1.], [0, 1], [True, True], [601, 1], 600) == ([], 0.)
    cases['boundary_hinge'] = dict(pairs=pairs, loss=value, positive_gradient=-1., negative_gradient=1.)
    for eps in (1e-6,):
        p = boundary_pairs([.9, .6], [-1. + eps, 1.], [0, 0], [True, True], [601, 1], 600)[1]
        assert math.isclose((p - value) / eps, -1., abs_tol=1e-8)
    # The boundary term cannot correct an inversion already inside topK.
    assert boundary_pairs([.9, .6], [-1., 1.], [0, 0], [True, True], [2, 1], 600) == ([], 0.)
    rejected = 0
    for call in (lambda: iou([0, 0, 0, 1], g),
                 lambda: coverage([g], [g], [0], [[float('nan')]], 1),
                 lambda: coverage([g], [g], [1], [[.9]], 1),
                 lambda: coverage([g], [g], [0], [[.9]], 2)):
        try:
            call()
        except ValueError:
            rejected += 1
    assert rejected == 4
    return dict(status='SYNTHETIC_CHECKS_PASS', scientific_status='NOT_SCIENTIFIC',
                production_input_supported=False, GPU_forward=0, model_forward=0,
                cases=cases, invalid_cases_rejected=rejected)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--synthetic-check', action='store_true', required=True)
    parser.parse_args()
    print(json.dumps(synthetic_checks(), indent=2, allow_nan=False))
