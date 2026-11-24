"""TASK_TH2_U1_001 reference functions (controller; numpy only).  The definitions of record of the brief.

A uniform intermediate canvas U896 (896 x 1568) is compared with the two stored uniform end points of the same seed,
A0 (768 x 1344) and S15 (1152 x 2016), and with the per-image routed system of C0.  Nothing here reads a file; the
self-test uses synthetic numbers only.

Per-image arrays: a, u, b = small-GT hit75 counts of A0, U896, S15; n = small-GT counts; all of length 548.
"""
import math

import numpy as np

L_HW, U_HW, H_HW = (768, 1344), (896, 1568), (1152, 2016)
STRIDES = (8, 16, 32)


def tokens(hw):
    """Multi-scale tokens of a canvas (strides 8, 16, 32)."""
    assert all(v % 32 == 0 for v in hw)
    return sum((hw[0] // s) * (hw[1] // s) for s in STRIDES)


B_U = tokens(U_HW) / tokens(L_HW)                       # 49/36 = 1.361111
B_H = tokens(H_HW) / tokens(L_HW)                       # 2.25
M_U, M_H = U_HW[0] / L_HW[0], H_HW[0] / L_HW[0]         # linear magnification against L: 7/6 and 1.5
LOGLINEAR = math.log(B_U) / math.log(B_H)               # 0.380182: share of the lead if hits were linear in log pixels
CHORD = (B_U - 1.0) / (B_H - 1.0)                       # 0.288889: share of the lead if hits were linear in pixels
M_IMAGES, K_GATE = 548, 158                             # the routed system of C0: the first 158 images of order P
LINE_HIT = 0.50                                         # THESIS_PLAN_UPDATE_003 section 4, unchanged
BOOT, BOOT_SEED = 1000, 20261122                        # the phase-1 / C0 image bootstrap (c0_ref.boot_weights)
SHIFT_WIDTH = math.log(M_H) / 4.0                       # bin width in log size: the H shift is exactly 4 bins
SHIFT_TAU = 20.0                                        # shrinkage of a class curve towards the pooled curve


# ---------------------------------------------------------------- share of the lead
def fraction(u, a, b):
    """(U896 - A0) / (S15 - A0) for per-image hit counts (or three scalars of one metric)."""
    u, a, b = (float(np.sum(x)) for x in (u, a, b))
    assert b != a
    return (u - a) / (b - a)


def fraction_bootstrap(u, a, b, w):
    """Image bootstrap of the share (ratio of resampled sums); w = c0_ref.boot_weights(548)."""
    u, a, b = (np.asarray(x, dtype=np.float64) for x in (u, a, b))
    f = (w @ (u - a)) / (w @ (b - a))
    return [float(np.quantile(f, 0.025)), float(np.quantile(f, 0.975))]


# ---------------------------------------------------------------- routed system against the uniform canvas
def routed_hits(a, b, routed):
    """Per-image hits when the images of the boolean mask `routed` are read from S15 and the others from A0."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    routed = np.asarray(routed)
    assert routed.dtype == bool and routed.shape == a.shape == b.shape
    return np.where(routed, b, a)


def delta_hit(a, b, u, n, routed):
    """Routed system minus U896, small hit75, pp."""
    n = np.asarray(n, dtype=np.float64)
    return 100.0 * float((routed_hits(a, b, routed) - np.asarray(u, dtype=np.float64)).sum()) / float(n.sum())


def delta_hit_bootstrap(a, b, u, n, routed, w):
    """Image bootstrap with the routed set held fixed."""
    d = routed_hits(a, b, routed) - np.asarray(u, dtype=np.float64)
    q = 100.0 * (w @ d) / (w @ np.asarray(n, dtype=np.float64))
    return [float(np.quantile(q, 0.025)), float(np.quantile(q, 0.975))]


def stage1_reading(delta):
    """THESIS_PLAN_UPDATE_003 section 4, after seed 0.  A reading for the controller, never an authorisation."""
    if delta >= LINE_HIT:
        return "ROUTING_AHEAD_ON_SEED0"          # the controller may propose seeds 1 and 2
    if delta <= 0.0:
        return "UNIFORM_NOT_BEHIND_ON_SEED0"     # no second stage on behalf of routing
    return "IN_BETWEEN"                          # the controller rules, with reasons


# ---------------------------------------------------------------- how many uniform pixels a gain is worth
def equivalent_budget(gain, gain_u, gain_h):
    """Budget (in units of the L canvas) at which the uniform family reaches `gain`, by log-linear interpolation
    through (1, 0), (B_U, gain_u), (B_H, gain_h).  All gains against the L-canvas end point of the own family."""
    if not (0.0 < gain_u < gain_h):
        return {"status": "uniform_gains_not_increasing", "budget": None}
    if gain <= 0.0:
        return {"status": "at_or_below_L", "budget": None}
    if gain <= gain_u:
        return {"status": "between_L_and_U", "budget": math.exp(math.log(B_U) * gain / gain_u)}
    if gain <= gain_h:
        t = (gain - gain_u) / (gain_h - gain_u)
        return {"status": "between_U_and_H", "budget": math.exp(math.log(B_U) + t * (math.log(B_H) - math.log(B_U)))}
    return {"status": "above_H", "budget": None}


# ---------------------------------------------------------------- prediction 3: concave response over native-size groups
def group_concave(lead, count, scale_l):
    """Groups of images by native size.  lead[g]: S15 minus the L end point per small GT of the group, pp;
    count[g]: small GT of the group; scale_l[g]: linear scale of the L canvas against the native image,
    sqrt((1344 / W) * (768 / H)).

    Model: the hit rate of a group is a_g + b1 * x + b2 * x^2 in x = log(linear scale against the native image), so the
    lead between the canvas factors 1 and m is log(m) * (b1 + b2 * (2 * log(scale_l) + log(m))).  b1, b2 by least squares
    on the leads at m = 1.5, weighted by the GT counts.  Returns the predicted lead of U896 and its share of the lead."""
    lead, count, scale_l = (np.asarray(x, dtype=np.float64) for x in (lead, count, scale_l))
    assert lead.shape == count.shape == scale_l.shape and len(lead) >= 2
    x0 = 2.0 * np.log(scale_l)
    design = np.stack([np.ones_like(x0), x0 + math.log(M_H)], axis=1)
    sw = np.sqrt(count)
    beta, *_ = np.linalg.lstsq(design * sw[:, None], (lead / math.log(M_H)) * sw, rcond=None)
    at = lambda m: math.log(m) * (beta[0] + beta[1] * (x0 + math.log(m)))
    share = count / count.sum()
    observed = float(share @ lead)
    predicted_u = float(share @ at(M_U))
    return {"b1": float(beta[0]), "b2": float(beta[1]), "fitted_lead_H": at(M_H).tolist(), "predicted_lead_U": at(M_U).tolist(),
            "observed_lead_H_pp": observed, "predicted_lead_U_pp": predicted_u, "f": predicted_u / observed}


# ---------------------------------------------------------------- prediction 4: size shift
def _curves(x, cls, hit):
    """Hit rate by class and by bin of log size (bin k covers [k, k + 1) * SHIFT_WIDTH), class curves shrunk towards
    the pooled curve; empty pooled bins take the value of the nearest non-empty bin (the lower one on a tie)."""
    k = np.floor(x / SHIFT_WIDTH).astype(np.int64)
    lo, hi = int(k.min()), int(k.max())
    nb = hi - lo + 1
    tot, cnt = np.bincount(k - lo, weights=hit, minlength=nb), np.bincount(k - lo, minlength=nb).astype(np.float64)
    filled = np.flatnonzero(cnt > 0)
    pooled = np.empty(nb)
    for i in range(nb):
        j = filled[np.argmin(np.abs(filled - i))]          # argmin returns the first, i.e. the lower bin on a tie
        pooled[i] = tot[j] / cnt[j]
    curves = {}
    for c in np.unique(cls):
        m = cls == c
        t = np.bincount(k[m] - lo, weights=hit[m], minlength=nb)
        n = np.bincount(k[m] - lo, minlength=nb)
        curves[int(c)] = (t + SHIFT_TAU * pooled) / (n + SHIFT_TAU)
    return lo, curves


def _evaluate(curve, lo, x):
    """Linear interpolation between bin centres, constant beyond the first / last centre."""
    t = x / SHIFT_WIDTH - 0.5 - lo
    t = np.clip(t, 0.0, len(curve) - 1.0)
    i = np.minimum(np.floor(t).astype(np.int64), len(curve) - 2) if len(curve) > 1 else np.zeros(len(t), dtype=np.int64)
    frac = t - i
    nxt = np.minimum(i + 1, len(curve) - 1)
    return (1.0 - frac) * curve[i] + frac * curve[nxt]


def shift_prediction(s, cls, hit, target, factors=(1.0, M_U, M_H)):
    """'A small object magnified by m is found as often as an object of the same class that is natively m times larger.'

    s: linear size sqrt(w * h) on canvas L of every scored GT (all sizes); cls: class id; hit: hit75 (0 / 1) of the
    L-canvas arm; target: boolean mask of the GT whose rate is predicted (the small GT).  The curve is the hit rate of
    the L-canvas arm against log size; the prediction for the factor m reads the curve of the GT's class at
    log(s) + log(m).  Returns the predicted rate (pp) per factor; the gain of m is rate(m) - rate(1)."""
    s, cls, hit = np.asarray(s, dtype=np.float64), np.asarray(cls), np.asarray(hit, dtype=np.float64)
    target = np.asarray(target)
    assert s.ndim == 1 and s.shape == cls.shape == hit.shape == target.shape and target.dtype == bool
    assert bool((s > 0).all()) and bool(np.isin(hit, (0.0, 1.0)).all()) and target.any()
    x = np.log(s)
    lo, curves = _curves(x, cls, hit)
    out = {}
    for m in factors:
        p = np.empty(len(x))
        for c, curve in curves.items():
            sel = cls == c
            p[sel] = _evaluate(curve, lo, x[sel] + math.log(m))
        out[float(m)] = 100.0 * float(p[target].mean())
    return out


def shift_summary(pred, observed_l, observed_h):
    """pred: shift_prediction(...) for the factors (1, M_U, M_H); observed_*: small hit75 of A0 and S15, pp."""
    gain_u, gain_h = pred[float(M_U)] - pred[1.0], pred[float(M_H)] - pred[1.0]
    lead = observed_h - observed_l
    return {"rate_at_1_pp": pred[1.0], "reproduction_error_pp": pred[1.0] - observed_l,
            "law_gain_U_pp": gain_u, "law_gain_H_pp": gain_h, "observed_lead_H_pp": lead,
            "kappa": lead / gain_h if gain_h else None,                # realised share of the law's gain at H
            "f_law": gain_u / gain_h if gain_h else None,             # the law rescaled to the observed lead at H
            "f_pure": gain_u / lead if lead else None}                # the law taken at face value


# ---------------------------------------------------------------- self-test
def selftest():
    rng = np.random.default_rng(11)
    assert tokens(L_HW) == 21168 and tokens(U_HW) == 28812 and tokens(H_HW) == 47628
    assert abs(B_U - 49.0 / 36.0) < 1e-15 and B_H == 2.25 and abs(M_U - 7.0 / 6.0) < 1e-15 and M_H == 1.5
    assert abs(LOGLINEAR - 0.380182) < 5e-7 and abs(CHORD - 0.288889) < 5e-7
    assert 1.0 + (B_H - 1.0) * K_GATE / M_IMAGES <= B_U          # the routed system does not spend more than U896
    # 1. share of the lead and the routed difference by brute force
    m = M_IMAGES
    n = rng.poisson(rng.gamma(1.0, 48.0, m)).astype(np.float64)
    a = rng.binomial(n.astype(int), 0.44).astype(np.float64)
    u = rng.binomial(n.astype(int), 0.46).astype(np.float64)
    b = rng.binomial(n.astype(int), 0.49).astype(np.float64)
    assert fraction(a, a, b) == 0.0 and fraction(b, a, b) == 1.0
    assert abs(fraction(u, a, b) - (u.sum() - a.sum()) / (b.sum() - a.sum())) < 1e-15
    assert abs(fraction(0.31, 0.30, 0.32) - 0.5) < 1e-12
    routed = np.zeros(m, dtype=bool)
    routed[np.argsort(-n, kind="mergesort")[:K_GATE]] = True
    brute = a.copy()
    brute[routed] = b[routed]
    assert abs(delta_hit(a, b, u, n, routed) - 100 * (brute.sum() - u.sum()) / n.sum()) < 1e-12
    assert abs(delta_hit(a, b, b, n, np.ones(m, dtype=bool))) < 1e-12 and abs(delta_hit(a, b, a, n, np.zeros(m, dtype=bool))) < 1e-12
    w1 = np.ones((1, m))
    assert max(abs(v - delta_hit(a, b, u, n, routed)) for v in delta_hit_bootstrap(a, b, u, n, routed, w1)) < 1e-9
    assert max(abs(v - fraction(u, a, b)) for v in fraction_bootstrap(u, a, b, w1)) < 1e-12
    assert stage1_reading(0.50) == "ROUTING_AHEAD_ON_SEED0" and stage1_reading(0.0) == "UNIFORM_NOT_BEHIND_ON_SEED0"
    assert stage1_reading(-0.3) == "UNIFORM_NOT_BEHIND_ON_SEED0" and stage1_reading(0.2) == "IN_BETWEEN"
    # 2. equivalent budget: the three nodes, both segments, the refusals
    e = equivalent_budget
    assert abs(e(2.0, 2.0, 5.0)["budget"] - B_U) < 1e-12 and abs(e(5.0, 2.0, 5.0)["budget"] - B_H) < 1e-12
    assert abs(e(1.0, 2.0, 5.0)["budget"] - math.sqrt(B_U)) < 1e-12 and e(1.0, 2.0, 5.0)["status"] == "between_L_and_U"
    assert abs(e(3.5, 2.0, 5.0)["budget"] - math.sqrt(B_U * B_H)) < 1e-12 and e(3.5, 2.0, 5.0)["status"] == "between_U_and_H"
    assert e(6.0, 2.0, 5.0)["status"] == "above_H" and e(-0.1, 2.0, 5.0)["status"] == "at_or_below_L"
    assert e(1.0, -0.2, 5.0)["status"] == e(1.0, 5.0, 5.0)["status"] == "uniform_gains_not_increasing"
    # 3. concave group model: exact recovery on noise-free leads, and a log-linear response gives the log-linear share
    scale = np.array([0.7, 0.988, 1.4])
    count = np.array([829.0, 19380.0, 6232.0])
    b1, b2 = 20.0, -13.5
    x0 = 2 * np.log(scale)
    lead = math.log(M_H) * (b1 + b2 * (x0 + math.log(M_H)))
    g = group_concave(lead, count, scale)
    truth_u = float((count / count.sum()) @ (math.log(M_U) * (b1 + b2 * (x0 + math.log(M_U)))))
    assert abs(g["b1"] - b1) < 1e-9 and abs(g["b2"] - b2) < 1e-9 and abs(g["predicted_lead_U_pp"] - truth_u) < 1e-9
    assert max(abs(x - y) for x, y in zip(g["fitted_lead_H"], lead)) < 1e-9
    flat = group_concave(np.full(3, 5.0), count, scale)
    assert abs(flat["b2"]) < 1e-9 and abs(flat["f"] - math.log(M_U) / math.log(M_H)) < 1e-9 and abs(flat["f"] - LOGLINEAR) < 1e-9
    # 4. size shift: a world in which the hit rate depends on class and on-canvas size only
    ngt = 400000
    cls = rng.integers(1, 4, ngt)
    s = np.exp(rng.uniform(math.log(4.0), math.log(220.0), ngt))
    offset = np.array([0.0, -9.0, -7.5, -10.5])[cls]
    prob = lambda size: 1.0 / (1.0 + np.exp(-(offset + 3.0 * np.log(size))))
    hit = (rng.random(ngt) < prob(s)).astype(np.float64)
    small = s <= 32.0
    pred = shift_prediction(s, cls, hit, small)
    exact = {mm: 100.0 * float(prob(mm * s)[small].mean()) for mm in (1.0, M_U, M_H)}
    assert all(abs(pred[float(mm)] - exact[mm]) < 0.35 for mm in exact), (pred, exact)
    summary = shift_summary(pred, exact[1.0], exact[M_H])
    f_exact = (exact[M_U] - exact[1.0]) / (exact[M_H] - exact[1.0])
    assert abs(summary["f_law"] - f_exact) < 0.01 and abs(summary["kappa"] - 1.0) < 0.03 and abs(summary["reproduction_error_pp"]) < 0.35
    # a world in which magnification does nothing: the law over-predicts, kappa reports it
    still = shift_summary(pred, exact[1.0], exact[1.0] + 0.1 * (exact[M_H] - exact[1.0]))
    assert abs(still["kappa"] - 0.1) < 0.01
    # sparse classes and empty bins do not break the curve
    few = rng.random(ngt) < 0.002
    p2 = shift_prediction(s[few], cls[few], hit[few], small[few])
    assert all(np.isfinite(v) and 0.0 <= v <= 100.0 for v in p2.values())
    one = shift_prediction(np.array([10.0, 10.2]), np.array([1, 1]), np.array([1.0, 0.0]), np.array([True, True]))   # one bin
    assert all(abs(v - 50.0) < 1e-9 for v in one.values())
    two = shift_prediction(np.array([10.0, 20.0]), np.array([1, 1]), np.array([0.0, 1.0]), np.array([True, False]))  # empty bins between
    assert two[1.0] == 0.0 and two[1.0] <= two[float(M_U)] <= two[float(M_H)] and abs(two[float(M_H)] - 100.0) < 1e-9
    print("budgets: U896 %.6f, H %.6f; routed k=%d %.6f; loglinear %.6f, chord %.6f" % (
        B_U, B_H, K_GATE, 1.0 + (B_H - 1.0) * K_GATE / M_IMAGES, LOGLINEAR, CHORD))
    print("size shift on the synthetic world: predicted %s, exact %s, f_law %.4f (exact %.4f)" % (
        {round(k, 4): round(v, 3) for k, v in pred.items()}, {round(k, 4): round(v, 3) for k, v in exact.items()}, summary["f_law"], f_exact))
    print("selftest PASS")


if __name__ == "__main__":
    selftest()
