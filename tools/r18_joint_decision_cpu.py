"""Finite-box MBR witnesses only: no model, training, cache or AP evaluation."""
from fractions import Fraction as F
from itertools import product
import json
import math

from r18_boundary_design_cpu import softmax, numeric_gradient, close


def boxes_from_reference(cx, cy, w, h):
    if not all(math.isfinite(float(v)) for v in (cx, cy, w, h)) or w <= 0 or h <= 0:
        raise ValueError('invalid reference')
    levels = ((cx-3*w/4, cx-w/4), (cy-3*h/4, cy-h/4),
              (cx+w/4, cx+3*w/4), (cy+h/4, cy+3*h/4))
    return list(product(*levels))  # (left, top, right, bottom), lexicographic


def iou(a, b):
    x = max(0, min(a[2], b[2])-max(a[0], b[0]))
    y = max(0, min(a[3], b[3])-max(a[1], b[1]))
    intersection = x*y
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union


def validate(boxes, p):
    if len(boxes) != len(p) or not boxes:
        raise ValueError('shape')
    if any(not math.isfinite(float(v)) or v < 0 for v in p) or abs(sum(p)-1) > 1e-12:
        raise ValueError('probabilities')
    if len(set(boxes)) != len(boxes):
        raise ValueError('duplicate box')
    if any(not all(math.isfinite(float(v)) for v in b) or b[0] >= b[2] or b[1] >= b[3] for b in boxes):
        raise ValueError('invalid box')


def decision(boxes, p):
    validate(boxes, p)
    utility = [sum(prob*iou(action, state) for state, prob in zip(boxes, p)) for action in boxes]
    best = min(range(len(boxes)), key=lambda k: (-utility[k], boxes[k]))
    return best, utility


def marginals(boxes, p):
    return [{value: sum(prob for b, prob in zip(boxes, p) if b[axis] == value)
             for value in sorted({b[axis] for b in boxes})} for axis in range(4)]


def independent(boxes, p):
    marginal = marginals(boxes, p)
    return [math.prod(marginal[axis][b[axis]] for axis in range(4)) for b in boxes]


def target_index(boxes, gt, w, h):
    # Detached reference-relative L1 quantization; never select by learned P.
    return min(range(len(boxes)), key=lambda k: (
        sum(abs(boxes[k][a]-gt[a])/(w if a in (0, 2) else h) for a in range(4)), boxes[k]))


def run():
    boxes = boxes_from_reference(F(3), F(2), F(4), F(4))
    def distribution(weights, epsilon=F(1, 10000)):
        # Nonzero mass on all 16 states: realizable by finite softmax logits.
        active = {(F(0), F(-1), F(4), F(3)): weights[0],
                  (F(0), F(-1), F(6), F(3)): weights[1],
                  (F(2), F(-1), F(4), F(3)): weights[2],
                  (F(2), F(-1), F(6), F(3)): weights[3]}
        return [(1-epsilon)*active.get(b, 0)+epsilon/16 for b in boxes]
    p = distribution([F(1,100), F(9,100), F(49,100), F(41,100)])
    q = distribution([F(9,100), F(1,100), F(41,100), F(49,100)])
    assert marginals(boxes,p) == marginals(boxes,q)
    ip, up = decision(boxes,p)
    iq, uq = decision(boxes,q)
    assert boxes[ip] == (2,-1,4,3) and boxes[iq] == (2,-1,6,3)
    assert sum(v == up[ip] for v in up) == sum(v == uq[iq] for v in uq) == 1
    c = independent(boxes,p)
    assert c == independent(boxes,q)
    ic, _ = decision(boxes,c)
    checks = ['same_four_marginals_distinct_unique_decisions', 'product_marginal_control_same_for_both']
    # Either dependence assumption can hurt if it is miscalibrated.
    assert iou(boxes[ip],boxes[ip]) > iou(boxes[iq],boxes[ip])
    assert iou(boxes[iq],boxes[iq]) > iou(boxes[ip],boxes[iq])
    checks.append('correlation_misleads_under_opposite_true_state')
    multi = distribution([F(4,10),F(1,10),F(1,10),F(4,10)],epsilon=F(0))
    im, _ = decision(boxes,multi)
    map_index = min(range(len(boxes)), key=lambda k: (-multi[k], boxes[k]))
    assert boxes[im] == (0,-1,6,3) and im != map_index
    assert iou(boxes[im],boxes[map_index]) < 1
    checks.append('multimodal_mbr_not_map_and_not_always_more_accurate')
    tied = distribution([F(1,2),0,0,F(1,2)],epsilon=F(0))
    it, ut = decision(boxes,tied)
    assert sum(v == ut[it] for v in ut) > 1
    assert boxes[it] == min(boxes[k] for k,v in enumerate(ut) if v == max(ut))
    near_a = distribution([F(1,2)+F(1,100000),0,0,F(1,2)-F(1,100000)],epsilon=F(0))
    near_b = distribution([F(1,2)-F(1,100000),0,0,F(1,2)+F(1,100000)],epsilon=F(0))
    assert decision(boxes,near_a)[0] != decision(boxes,near_b)[0]
    checks.append('ambiguous_boundary_tie_rule_and_small_change_instability')
    z = [.03*i for i in range(16)]
    target = target_index(boxes,(F(2),F(-1),F(4),F(3)),F(4),F(4))
    prob = softmax(z)
    analytic = [x-(i==target) for i,x in enumerate(prob)]
    numerical = numeric_gradient(lambda logits: -math.log(softmax(logits)[target]),z)
    close(analytic,numerical)
    checks.append('joint_gt_nll_gradient_no_argmax_backprop')
    # Single GT joint supervision distinguishes fixed-marginal P and Q.
    assert -math.log(float(p[target])) != -math.log(float(q[target]))
    checks.append('joint_nll_distinguishes_equal_marginals')
    # Closest conventional finite-candidate MBR: pairwise loss-matrix product.
    costs = [sum(probability*(1-iou(action,state)) for state,probability in zip(boxes,p)) for action in boxes]
    conventional = min(range(len(boxes)),key=lambda k:(costs[k],boxes[k]))
    assert conventional == ip
    assert all(1-up[k] == costs[k] for k in range(len(boxes)))
    checks.append('exact_equivalence_to_conventional_finite_candidate_mbr')
    far_gt = (F(-3),F(-1),F(9),F(3))
    assert max(iou(b,far_gt) for b in boxes) < F(3,4)
    checks.append('support_limit_cannot_reach_high_iou')
    for invalid in ([F(0)]*16, [float('nan')]*16):
        try:
            decision(boxes,invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid probabilities accepted')
    checks.append('invalid_probability_rejection')
    return {'schema':'r18-joint-decision-cpu-v1','evidence':'SYNTHETIC_NOT_SCIENTIFIC',
            'checks':checks,'status':'PASS','p_decision':list(boxes[ip]),'q_decision':list(boxes[iq]),
            'independent_decision':list(boxes[ic]),'p':p,'q':q,'p_utilities':up,'q_utilities':uq,
            'multimodal_decision':list(boxes[im]),'multimodal_map':list(boxes[map_index]),
            'nll_gradient_max_error':max(abs(a-b) for a,b in zip(analytic,numerical)),
            'retained_candidate':None,'learning_budget':0,'gpu_calls':0,'model_calls':0,'training_updates':0}


if __name__ == '__main__':
    print(json.dumps(run(),indent=2,default=lambda x: str(x) if isinstance(x,F) else x,allow_nan=False))
