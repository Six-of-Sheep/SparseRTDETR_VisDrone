"""Constructed-only boundary readout falsification. No data/model dependencies."""
import json
import math


def softmax(z):
    m = max(z)
    e = [math.exp(v - m) for v in z]
    return [v / sum(e) for v in e]


def joint_edges(p, left, right):
    k = len(left)
    assert len(right) == k and len(p) == k * k
    assert all(math.isfinite(v) and v >= 0 for v in p)
    assert abs(sum(p) - 1) < 1e-12
    assert max(left) < min(right)
    return (sum(p[i*k+j] * left[i] for i in range(k) for j in range(k)),
            sum(p[i*k+j] * right[j] for i in range(k) for j in range(k)))


def marginals(p, k):
    return ([sum(p[i*k+j] for j in range(k)) for i in range(k)],
            [sum(p[i*k+j] for i in range(k)) for j in range(k)])


def independent_edges(p, left, right):
    a, b = marginals(p, len(left))
    return sum(x*y for x, y in zip(a, left)), sum(x*y for x, y in zip(b, right))


def loss(edges, target=(0.25, 4.25)):
    # Smooth witness, not a new proposed detector loss.
    return sum((x-y)**2 for x, y in zip(edges, target)) / 2


def numeric_gradient(fn, z, eps=1e-5):
    result = []
    for i in range(len(z)):
        hi, lo = list(z), list(z)
        hi[i] += eps
        lo[i] -= eps
        result.append((fn(hi)-fn(lo))/(2*eps))
    return result


def close(a, b, tolerance=1e-9):
    assert len(a) == len(b)
    assert max(abs(x-y) for x, y in zip(a, b)) < tolerance, (a, b)


def interval_iou(a, b):
    inter = max(0.0, min(a[1], b[1])-max(a[0], b[0]))
    return inter / (a[1]-a[0]+b[1]-b[0]-inter)


def run():
    left, right = [0., 2.], [4., 6.]
    checks = []
    p, q = [.4, .1, .1, .4], [.1, .4, .4, .1]
    for probabilities in (p, q, [.7, .1, .05, .15]):
        close(joint_edges(probabilities, left, right), independent_edges(probabilities, left, right))
    checks.append('joint_expectation_equals_marginal_readout')
    covariance = []
    for probabilities in (p, q):
        l, r = joint_edges(probabilities, left, right)
        covariance.append(sum(probabilities[i*2+j]*(left[i]-l)*(right[j]-r)
                              for i in range(2) for j in range(2)))
    close(covariance, [.6, -.6])
    close(joint_edges(p, left, right), joint_edges(q, left, right))
    assert loss(joint_edges(p, left, right)) == loss(joint_edges(q, left, right))
    checks.append('different_covariance_same_box_and_coordinate_loss')
    # Zero row/column sums: coupling changes, readout and loss do not.
    direction = [1., -1., -1., 1.]
    for epsilon in (-.05, 0., .05):
        shifted = [x+epsilon*d for x, d in zip(p, direction)]
        close(joint_edges(shifted, left, right), [1., 5.])
    checks.append('fixed_marginal_coupling_direction_is_null')
    z = [.2, -.4, .8, .1]
    probabilities = softmax(z)
    l, r = joint_edges(probabilities, left, right)
    analytic = [probabilities[i*2+j]*((l-.25)*(left[i]-l)+(r-4.25)*(right[j]-r))
                for i in range(2) for j in range(2)]
    direct = numeric_gradient(lambda a: loss(joint_edges(softmax(a), left, right)), z)
    control = numeric_gradient(lambda a: loss(independent_edges(softmax(a), left, right)), z)
    close(analytic, direct)
    close(direct, control)
    checks.append('analytic_logit_gradient_and_control_finite_difference_agree')
    a, b, u, v = [.1, -.2], [.4, .7], [.3, -.5], [.2, .6]
    rank_z = [a[i]+b[j]+u[i]*v[j] for i in range(2) for j in range(2)]
    rank_p = softmax(rank_z)
    rl, rr = joint_edges(rank_p, left, right)
    dz = [rank_p[i*2+j]*((rl-.25)*(left[i]-rl)+(rr-4.25)*(right[j]-rr))
          for i in range(2) for j in range(2)]
    du = [sum(dz[i*2+j]*v[j] for j in range(2)) for i in range(2)]
    numerical_u = numeric_gradient(lambda uv: loss(joint_edges(softmax(
        [a[i]+b[j]+uv[i]*v[j] for i in range(2) for j in range(2)]), left, right)), u)
    close(du, numerical_u)
    checks.append('rank_one_interaction_chain_rule')
    independent = softmax([a[i]+b[j] for i in range(2) for j in range(2)])
    close(independent, [x*y for x in softmax(a) for y in softmax(b)])
    checks.append('zero_interaction_recovers_independent_distribution_control')
    # A generic coupling-logit perturbation CAN change marginals; do not claim
    # the entire joint head has zero gradient or is untrainable.
    assert max(abs(v) for v in analytic) > .01
    checks.append('generic_joint_parameter_gradient_nonzero_counterexample')
    # Moving one boundary adjusts center AND width; separate center/size heads
    # are not necessary merely because oracle sensitivities were mixed.
    old, corrected = (1., 4.), (0., 4.)
    assert interval_iou(corrected, (0., 4.)) > interval_iou(old, (0., 4.))
    assert sum(old)/2 != sum(corrected)/2 and old[1]-old[0] != corrected[1]-corrected[0]
    checks.append('one_edge_correction_changes_center_and_width_positive_example')
    # Conflicting modes: the expected box need not be either supported mode.
    modes = [.5, 0., 0., .5]
    expected = joint_edges(modes, left, right)
    assert expected == (1., 5.) and interval_iou(expected, (0., 4.)) == .6
    checks.append('multimodal_average_can_be_worse_than_one_mode')
    # An unobserved GT outside the support cannot be recovered by expectation.
    assert expected[0] >= min(left) > -1.
    checks.append('finite_support_cannot_recover_out_of_support_boundary')
    # A different, nonlinear objective CAN see correlation. It is not included
    # in the reviewed candidate and is not dismissed by the expectation proof.
    nll_p, nll_q = -math.log(p[0]), -math.log(q[0])
    assert nll_p != nll_q
    checks.append('joint_nll_is_not_covered_by_coordinate_loss_equivalence')
    return {'schema': 'r18-boundary-design-cpu-v1', 'status': 'PASS',
            'evidence': 'SYNTHETIC_NOT_SCIENTIFIC', 'checks': checks,
            'covariance_pair': covariance, 'same_expected_edges': list(expected),
            'mode_counterexample_iou': .6, 'gradient_max_error': max(abs(a-b) for a,b in zip(analytic,direct)),
            'retained_candidate': None, 'gpu_calls': 0, 'model_forward_calls': 0,
            'training_updates': 0, 'real_cache_reads': 0}


if __name__ == '__main__':
    print(json.dumps(run(), ensure_ascii=False, indent=2, allow_nan=False))
