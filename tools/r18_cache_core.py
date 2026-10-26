"""CPU diagnostic graph calculations; no AP/AR or model code."""
from collections import Counter
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

STAGES = ('geometry', 'argmax_correct', 'budget_correct', 'post_ignore', 'score_order_match')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def overlap(boxes, truth):
    a = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(truth, dtype=np.float64).reshape(-1, 4)
    require(np.isfinite(a).all() and np.isfinite(b).all(), 'nonfinite geometry')
    require((a[:, 2:] > a[:, :2]).all() and (b[:, 2:] > b[:, :2]).all(), 'nonpositive geometry')
    inter = np.maximum(0, np.minimum(a[:, None, 2:], b[None, :, 2:]) -
                       np.maximum(a[:, None, :2], b[None, :, :2])).prod(-1)
    aa = (a[:, 2:] - a[:, :2]).prod(-1)
    bb = (b[:, 2:] - b[:, :2]).prod(-1)
    return inter / (aa[:, None] + bb[None, :] - inter)


def cardinality(edges):
    # Shape GT x physical-query. One query can cover at most one GT across classes.
    if not edges.size:
        return 0
    return int((maximum_bipartite_matching(csr_matrix(edges), perm_type='column') >= 0).sum())


def score_match(edges, quality, ranks, categories):
    """Diagnostic score-ordered one-query/one-GT matching, NOT an evaluator.

    Same native score/q/class ordering, then largest IoU, then input GT order.
    No maxDets/score0 crowd matching; therefore not either official TP vector.
    """
    used_q, used_g = set(), set()
    items = [(int(ranks[q, categories[g]]), q, int(categories[g]), g)
             for g, q in zip(*np.where(edges))]
    for _, q, category, _ in sorted(items):
        if q in used_q:
            continue
        options = [g for g in np.flatnonzero(edges[:, q]) if g not in used_g and categories[g] == category]
        if not options:
            continue
        g = min(options, key=lambda x: (-quality[x, q], x))
        used_q.add(q)
        used_g.add(g)
    out = np.zeros(len(edges), dtype=bool)
    if used_g:
        out[list(used_g)] = True
    return out


def stages(quality, categories, argmax, ranks, keep, threshold, budget):
    n = len(categories)
    geom = quality >= threshold
    cls = geom & (np.asarray(argmax)[None, :] == np.asarray(categories)[:, None])
    selected = ranks[:, categories].T <= budget if n else np.empty((0, len(argmax)), dtype=bool)
    ranked = geom & selected
    filtered = ranked & np.asarray(keep)[None, :]
    final = score_match(filtered, quality, ranks, categories)
    flags = np.stack((geom.any(1), cls.any(1), ranked.any(1), filtered.any(1), final), axis=1)
    caps = [cardinality(x) for x in (geom, cls, ranked, filtered)]
    require(0 <= caps[3] <= caps[2] <= caps[0] <= n and caps[1] <= caps[0], 'graph capacity order')
    require(int(final.sum()) <= caps[3], 'score-order match exceeds maximum capacity')
    # Argmax is a parallel description, not a mandatory filter of multi-class topK.
    return flags, caps


def summarize(flags, caps):
    n = len(flags)
    c = dict(zip(STAGES, (int(x) for x in flags.sum(0))))
    c.update(denominator=n, geometry_absent=int((~flags[:, 0]).sum()),
             budget_lost=int((flags[:, 0] & ~flags[:, 2]).sum()),
             ignore_lost=int((flags[:, 2] & ~flags[:, 3]).sum()),
             retained_unmatched=int((flags[:, 3] & ~flags[:, 4]).sum()),
             non_argmax_but_retained=int((~flags[:, 1] & flags[:, 2]).sum()),
             max_geometry=caps[0], max_argmax_correct=caps[1],
             max_budget_correct=caps[2], max_post_ignore=caps[3])
    require(sum(c[k] for k in ('geometry_absent', 'budget_lost', 'ignore_lost',
                              'retained_unmatched', 'score_order_match')) == n, 'stage partition differs')
    return c


def add_counts(target, key, value):
    target.setdefault(key, Counter()).update(value)


def size_group(area):
    require(area > 0 and np.isfinite(area), 'invalid raw GT area')
    return 'small' if area <= 1024 else 'medium' if area <= 9216 else 'large'
