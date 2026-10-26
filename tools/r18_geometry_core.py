"""Fixed-pair geometric sensitivity only. No detector, ranking or evaluator."""
import numpy as np
from r18_cache_core import overlap, require

GROUPS = ('center_only', 'size_only', 'both_individually', 'neither_individually')
QUANTILES = (0, .05, .25, .5, .75, .95, 1)


def analyze(boxes, gt):
    """Select ONCE by original float64 IoU; exact ties use smallest query ID."""
    boxes = np.asarray(boxes, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    quality = overlap(boxes, [gt])[:, 0]
    q = int(np.argmax(quality))
    b = boxes[q].copy()
    pc, gc = (b[:2] + b[2:]) / 2, (gt[:2] + gt[2:]) / 2
    pw, gw = b[2:] - b[:2], gt[2:] - gt[:2]
    center = np.r_[gc - pw / 2, gc + pw / 2]
    size = np.r_[pc - gw / 2, pc + gw / 2]
    # Both replacements reconstruct the GT; do not pick a new query.
    both = np.r_[gc - gw / 2, gc + gw / 2]
    values = overlap([b, center, size, both], [gt])[:, 0]
    require(abs(values[3] - 1.) < 1e-12, 'both-replaced IoU is not one')
    c, s = values[1] >= .75, values[2] >= .75
    group = 'both_individually' if c and s else 'center_only' if c else 'size_only' if s else 'neither_individually'
    dx, dy = pc - gc
    rw, rh = pw / gw
    metrics = dict(original_IoU=values[0], center_IoU=values[1], size_IoU=values[2], both_IoU=values[3],
                   delta_center=values[1]-values[0], delta_size=values[2]-values[0],
                   interaction=values[3]-values[1]-values[2]+values[0],
                   dx_pixels=dx, dy_pixels=dy, dx_over_GT_width=dx/gw[0], dy_over_GT_height=dy/gw[1],
                   abs_dx_over_GT_width=abs(dx/gw[0]), abs_dy_over_GT_height=abs(dy/gw[1]),
                   width_ratio=rw, height_ratio=rh, log_width_ratio=np.log(rw), log_height_ratio=np.log(rh),
                   abs_log_width_ratio=abs(np.log(rw)), abs_log_height_ratio=abs(np.log(rh)))
    return dict(query=q, exact_ties=int(np.sum(quality == quality[q])), group=group,
                prediction=b.tolist(), truth=gt.tolist(), metrics={k: float(v) for k, v in metrics.items()})


def describe(records):
    n = len(records)
    if not n:
        return dict(denominator=0, groups={g: 0 for g in GROUPS}, distributions={})
    counts = {g: sum(r['group'] == g for r in records) for g in GROUPS}
    require(sum(counts.values()) == n, 'four-group partition differs')
    distributions = {}
    for name in records[0]['metrics']:
        a = np.asarray([r['metrics'][name] for r in records])
        require(np.isfinite(a).all(), 'nonfinite metric')
        distributions[name] = dict(n=n, mean=float(a.mean()), quantiles=dict(zip(
            (str(q) for q in QUANTILES), (float(x) for x in np.quantile(a, QUANTILES, method='linear')))))
    return dict(denominator=n, images=len({r['image_id'] for r in records}), groups=counts,
                proportions={g: counts[g]/n for g in GROUPS},
                exact_ties=sum(r['exact_ties'] > 1 for r in records), distributions=distributions)


def aggregate(records, categories, sequences):
    result = dict(overall=describe(records), groups={})
    dimensions = dict(category=[str(x) for x in categories], sequence=sequences,
                      area_bin=['(0,64]', '(64,256]', '(256,576]', '(576,1024]'])
    for dimension, keys in dimensions.items():
        result['groups'][dimension] = {str(key): describe([r for r in records if str(r[dimension]) == str(key)]) for key in keys}
    # Show sensitivity to domination without interpreting leave-one-group-out as uncertainty.
    result['concentration'] = {}
    for dimension in ('category', 'sequence'):
        nonempty = {k: v for k, v in result['groups'][dimension].items() if v['denominator']}
        n = len(records)
        macro = {g: float(np.mean([v['proportions'][g] for v in nonempty.values()])) for g in GROUPS} if nonempty else {}
        excluded = {k: dict(denominator=n-v['denominator'], groups={
            g: result['overall']['groups'][g]-v['groups'][g] for g in GROUPS}) for k, v in nonempty.items()}
        result['concentration'][dimension] = dict(nonempty_groups=len(nonempty), macro_group_proportions=macro,
            maximum_share=max((v['denominator']/n for v in nonempty.values()), default=0), leave_one_group_out=excluded)
    return result
