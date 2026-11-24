"""TASK_TH2_C0_001 reference functions (controller; numpy only).  The definitions of record of the brief.

Per-image routing between two stored arms of the same seed: every image is evaluated by exactly one arm, the cheap one
(canvas L, cost 1) or the expensive one (canvas H, cost 2.25).  Hits are per GT and do not depend on other images, so
the routed number of hits is an exact sum.  Nothing here reads a file; the self-test uses synthetic numbers only.
"""
import math

import numpy as np

H_COST = (1152 * 2016) / (768 * 1344)        # 2.25
LOG_H = math.log(H_COST)
M_IMAGES = 548
K_GATE = 158                                 # budget 1.3604: just under the uniform canvas 896 x 1568 (49/36 = 1.3611)
K_CURVE = (55, 110, 158, 219, 274, 341, 438)   # 219: budget 1.4995 (geometric midpoint); 341: 1.7778 (canvas 1024 x 1792)
RESIDUES = 7                                 # random-routing rows: the 7 subsets {i : i mod 7 in {r, (r + 3) mod 7}}
BOOT, BOOT_SEED = 1000, 20261122                               # the phase-1 image bootstrap


def per_image(values, image, n_images):
    """Sum of per-GT values per image."""
    out = np.zeros(n_images, dtype=np.float64)
    np.add.at(out, np.asarray(image), np.asarray(values, dtype=np.float64))
    return out


def order_desc(stat):
    """Image order by a statistic, largest first; ties by image index, ascending."""
    s = np.asarray(stat, dtype=np.float64)
    assert s.ndim == 1 and bool(np.isfinite(s).all())
    return np.lexsort((np.arange(len(s)), -s))


def budget(p):
    """Mean pixels per image in units of the L canvas when the share p of the images goes to H."""
    return 1.0 + (H_COST - 1.0) * p


def loglinear(p):
    """Share of the lead a uniform arm of the same budget would have if hits were linear in log pixels."""
    return np.log(budget(np.asarray(p, dtype=np.float64))) / LOG_H


def cumulative(order, d):
    """c[k] = sum of d over the first k images of the order, k = 0..n."""
    return np.concatenate([[0.0], np.cumsum(np.asarray(d, dtype=np.float64)[order])])


def readings(a, b, n, order, k):
    """a, b: per-image hits of the cheap / expensive arm; n: per-image GT count; the first k images of `order` go to H."""
    a, b, n = (np.asarray(x, dtype=np.float64) for x in (a, b, n))
    m = len(a)
    d = b - a
    tot_n, tot_a, tot_d = n.sum(), a.sum(), d.sum()
    got = cumulative(order, d)[k]
    p = k / m
    ell = float(loglinear(p))
    return {"k": int(k), "p": p, "budget": budget(p), "loglinear": ell,
            "cheap_pp": 100 * tot_a / tot_n, "expensive_pp": 100 * (tot_a + tot_d) / tot_n, "lead_pp": 100 * tot_d / tot_n,
            "routed_pp": 100 * (tot_a + got) / tot_n, "chord_pp": 100 * (tot_a + p * tot_d) / tot_n,
            "reference_pp": 100 * (tot_a + ell * tot_d) / tot_n,
            "capture": got / tot_d, "gt_share": n[order[:k]].sum() / tot_n,
            "routed_minus_chord_pp": 100 * (got - p * tot_d) / tot_n, "Q_pp": 100 * (got - ell * tot_d) / tot_n}


def boot_weights(n_images, n=BOOT, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_images, np.full(n_images, 1.0 / n_images), size=n).astype(np.float64)


def q_bootstrap(a, b, n, order, k, w):
    """Image bootstrap of Q with the routed set held fixed (the first k images of the order): in a replicate the share
    of routed images, the budget and the reference move with the resampled images."""
    a, b, n = (np.asarray(x, dtype=np.float64) for x in (a, b, n))
    m = len(a)
    d = b - a
    r = np.zeros(m)
    r[order[:k]] = 1.0
    p = (w @ r) / m
    q = 100 * (w @ (r * d) - loglinear(p) * (w @ d)) / (w @ n)
    return [float(np.quantile(q, 0.025)), float(np.quantile(q, 0.975))]


def balanced_subsets(n_images):
    """The 7 subsets {i : i mod 7 in {r, (r + 3) mod 7}}, r = 0..6; every image is in exactly 2 of them, so the mean
    share of routed images is 2/7 and, for a quantity that is a sum over images, the mean routed value is exactly
    cheap + (2/7) * lead."""
    res = np.arange(n_images) % RESIDUES
    return [np.flatnonzero((res == r) | (res == (r + 3) % RESIDUES)) for r in range(RESIDUES)]


def q_metric(routed, cheap, expensive, k, n_images=M_IMAGES):
    """Q of a scorer metric (0..1 units in, pp out): routed minus the log-linear reference at the budget of k."""
    return 100.0 * (routed - (cheap + float(loglinear(k / n_images)) * (expensive - cheap)))


def mixing(random_values, cheap, expensive):
    """Random-routing rows (the 7 balanced subsets): mean value minus the chord at their mean share 2/7, pp."""
    v = np.asarray(random_values, dtype=np.float64)
    assert v.shape == (RESIDUES,)
    return 100.0 * (float(v.mean()) - (cheap + (2.0 / RESIDUES) * (expensive - cheap)))


def groups5(order):
    """Five groups of images along an order (sizes 110, 110, 110, 109, 109 for 548 images)."""
    return np.array_split(np.asarray(order), 5)


def verdict(q_hit, lead_hit, q_aps, lead_aps, q_ap, q_hit_count):
    """Three seeds each, cheap arm L0, expensive arm S15, k = K_GATE.  All arguments in pp (AP metrics times 100).
    q_hit, q_aps, q_ap: routed minus the log-linear reference under the label-free order (small hit75, APS, AP);
    q_hit_count: the same small-hit75 quantity under the label-count order (annotation only)."""
    mean = lambda v: float(np.mean(v))
    t1, t2 = 0.50 + 0.10 * mean(lead_hit), 0.20 + 0.10 * mean(lead_aps)
    go = {"G1_small_hit75": mean(q_hit) >= t1 and all(x > 0 for x in q_hit),
          "G2_APS": mean(q_aps) >= t2 and min(q_aps) >= -0.10,
          "G3_AP": mean(q_ap) >= 0.0}
    stop = {"S1_small_hit75": mean(q_hit) <= 0.0 or len({x > 0 for x in q_hit}) > 1}
    label = "STOP" if any(stop.values()) else "GO-CANDIDATE" if all(go.values()) else "UNDECIDED"
    return {"label": label, "GO": go, "STOP": stop, "thresholds_pp": {"G1": t1, "G2": t2},
            "count_order_clears_G1": bool(mean(q_hit_count) >= t1 and all(x > 0 for x in q_hit_count))}


def selftest():
    rng = np.random.default_rng(7)
    m = 548
    n = rng.poisson(rng.gamma(1.0, 48.0, m)).astype(np.float64)
    a = rng.binomial(n.astype(int), 0.44).astype(np.float64)
    b = rng.binomial(n.astype(int), 0.49).astype(np.float64)
    d = b - a
    o = order_desc(n)
    # 1. end points
    r0, r1 = readings(a, b, n, o, 0), readings(a, b, n, o, m)
    assert r0["routed_pp"] == r0["cheap_pp"] and r0["capture"] == 0 and abs(r0["Q_pp"]) < 1e-12
    assert abs(r1["routed_pp"] - r1["expensive_pp"]) < 1e-12 and abs(r1["capture"] - 1) < 1e-12 and abs(r1["Q_pp"]) < 1e-12
    assert budget(0) == 1 and abs(budget(1) - 2.25) < 1e-15 and loglinear(0) == 0 and abs(loglinear(1) - 1) < 1e-15
    # 2. routed hits by brute force
    k = K_GATE
    routed = a.copy()
    routed[o[:k]] = b[o[:k]]
    r = readings(a, b, n, o, k)
    assert abs(r["routed_pp"] - 100 * routed.sum() / n.sum()) < 1e-12
    assert abs(r["Q_pp"] - (r["routed_pp"] - r["reference_pp"])) < 1e-12
    assert abs(r["routed_minus_chord_pp"] - (r["routed_pp"] - r["chord_pp"])) < 1e-12
    # 3. the in-sample oracle is the largest gain of any k images
    oo = order_desc(d)
    for kk in (1, 55, 158, 400):
        best = np.sort(d)[::-1][:kk].sum()
        assert abs(cumulative(oo, d)[kk] - best) < 1e-9
        assert cumulative(oo, d)[kk] >= cumulative(o, d)[kk] - 1e-9
    # 4. balanced subsets: every image twice; the mean gain is exactly 2/7 of the lead
    subs = balanced_subsets(m)
    count = np.zeros(m)
    for s in subs:
        count[s] += 1
    sizes = sorted(len(s) for s in subs)
    assert len(subs) == 7 and bool((count == 2).all()) and sum(sizes) == 2 * m and sizes[0] == 156 and sizes[-1] == 157
    assert abs(np.mean([d[s].sum() for s in subs]) - (2.0 / 7.0) * d.sum()) < 1e-9
    # 5. bootstrap: unit weights give the point value; ties are ordered by index
    w1 = np.ones((1, m))
    lo, hi = q_bootstrap(a, b, n, o, k, w1)
    assert abs(lo - r["Q_pp"]) < 1e-9 and abs(hi - r["Q_pp"]) < 1e-9
    assert order_desc([1.0, 3.0, 3.0, 0.0]).tolist() == [1, 2, 0, 3]
    # 6. verdict table
    L, A = [5.0, 5.2, 5.4], [2.9, 2.8, 3.0]
    v = verdict([1.2, 1.1, 1.3], L, [0.6, 0.5, 0.7], A, [0.1, 0.2, 0.0], [1.5, 1.4, 1.6])
    assert v["label"] == "GO-CANDIDATE" and abs(v["thresholds_pp"]["G1"] - 1.02) < 1e-12 and abs(v["thresholds_pp"]["G2"] - 0.49) < 1e-12
    assert v["count_order_clears_G1"] is True
    assert verdict([0.6, 0.5, 0.7], L, [0.6, 0.5, 0.7], A, [0.1, 0.2, 0.0], [1.5, 1.4, 1.6])["label"] == "UNDECIDED"      # G1 short
    assert verdict([1.2, 1.1, 1.3], L, [0.3, 0.2, 0.4], A, [0.1, 0.2, 0.0], [1.5, 1.4, 1.6])["label"] == "UNDECIDED"      # G2 short
    assert verdict([1.2, 1.1, 1.3], L, [-0.3, -0.2, -0.4], A, [0.1, 0.2, 0.0], [1.5, 1.4, 1.6])["label"] == "UNDECIDED"   # APS below the reference
    assert verdict([1.2, 1.1, 1.3], L, [0.6, 0.5, 0.7], A, [-0.1, -0.2, 0.0], [1.5, 1.4, 1.6])["label"] == "UNDECIDED"    # G3 short
    s = verdict([-0.1, 0.1, -0.2], L, [0.6, 0.5, 0.7], A, [0.1, 0.2, 0.0], [1.5, 1.4, 1.6])
    assert s["label"] == "STOP" and s["count_order_clears_G1"] is True
    s = verdict([-0.3, -0.1, -0.2], L, [0.6, 0.5, 0.7], A, [0.1, 0.2, 0.0], [0.2, 0.1, 0.3])
    assert s["label"] == "STOP" and s["count_order_clears_G1"] is False
    assert verdict([0.3, -0.1, 0.4], L, [0.6, 0.5, 0.7], A, [0.1, 0.2, 0.0], [1.5, 1.4, 1.6])["label"] == "STOP"          # mixed signs
    # 7. scorer-metric helpers
    assert abs(q_metric(0.31, 0.30, 0.32, 0) - 1.0) < 1e-12 and abs(q_metric(0.32, 0.30, 0.32, M_IMAGES)) < 1e-12
    assert abs(q_metric(0.31, 0.30, 0.32, K_GATE) - 100 * (0.31 - 0.30 - float(loglinear(K_GATE / M_IMAGES)) * 0.02)) < 1e-12
    assert abs(mixing([0.30 + 0.02 * 2 / 7] * 7, 0.30, 0.32)) < 1e-12 and abs(mixing([0.30] * 7, 0.30, 0.32) + 100 * 0.02 * 2 / 7) < 1e-12
    assert [len(g) for g in groups5(o)] == [110, 110, 110, 109, 109] and np.concatenate(groups5(o)).tolist() == o.tolist()
    for kk in (K_GATE, 219, 341):
        r = readings(a, b, n, o, kk)
        print("k=%d: p=%.6f budget=%.6f loglinear=%.6f" % (kk, r["p"], r["budget"], r["loglinear"]))
    print("selftest PASS")


if __name__ == "__main__":
    selftest()
