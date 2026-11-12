#!/usr/bin/env python3
"""R27 post-L0 design check -- script 2 of 2.

Set-conditional detection-risk allocation: non-degeneracy counterexamples,
selector comparison by exhaustive enumeration, the pairwise-order limit, the
teacher-cost model and the deploy-time information budget.

Pure standard library. No torch, no CUDA, no model, no dataset, no prediction
file. Every "risk" value here is a SYNTHETIC number constructed to exercise the
mathematics in section 4 of the task book. Nothing below is evidence about
VisDrone, AP, or the trained detector.

Notation (task book section 4):
    S        = set of executed macro-actions
    J(S)     = L_native(f_S(x), y), with the native matching RE-DERIVED per S
    delta_i(S) = J(S) - J(S + {i})        (may be negative, depends on S)

Run:  python check_set_risk.py [--json]
"""

import argparse
import itertools
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# ==========================================================================
# helpers
# ==========================================================================


def subsets_of_size(n, k):
    return itertools.combinations(range(n), k)


def all_subsets(n):
    for mask in range(1 << n):
        yield mask


def bits(mask, n):
    return [i for i in range(n) if mask >> i & 1]


def popcount(m):
    return bin(m).count("1")


def solve_lstsq(A, b, ridge=1e-9):
    """Least squares by normal equations + Gauss-Jordan (small systems only)."""
    m, d = len(A), len(A[0])
    G = [[0.0] * d for _ in range(d)]
    r = [0.0] * d
    for row, y in zip(A, b):
        nz = [(j, v) for j, v in enumerate(row) if v != 0.0]
        for j, vj in nz:
            r[j] += vj * y
            Gj = G[j]
            for k, vk in nz:
                Gj[k] += vj * vk
    for j in range(d):
        G[j][j] += ridge
    M = [G[j] + [r[j]] for j in range(d)]
    for c in range(d):
        piv = max(range(c, d), key=lambda q: abs(M[q][c]))
        if abs(M[piv][c]) < 1e-14:
            continue
        M[c], M[piv] = M[piv], M[c]
        pv = M[c][c]
        M[c] = [v / pv for v in M[c]]
        for q in range(d):
            if q != c and M[q][c] != 0.0:
                f = M[q][c]
                Mc = M[c]
                M[q] = [a - f * bb for a, bb in zip(M[q], Mc)]
    return [M[j][d] for j in range(d)]


# ==========================================================================
# PART 1 -- four explicit counterexamples on a 3-action ground set
# ==========================================================================


def part1(out):
    p = out.append
    p("=" * 78)
    p("PART 1 -- explicit counterexamples (synthetic J, 3 actions a,b,c)")
    p("=" * 78)
    p("J is a RISK (lower is better). delta_i(S) = J(S) - J(S+{i}).")
    p("'independent rank' = sort by the leave-one-out drop from the full set,")
    p("LOO_i = J(All-{i}) - J(All), the proxy the task book forbids calling")
    p("an optimal budget teacher.")
    p("")
    names = "abc"
    cases = {}

    # (1) complementarity: a and b are useless alone, strong together.
    cases["complementary"] = {
        (): 10.0, (0,): 9.9, (1,): 9.9, (2,): 9.0,
        (0, 1): 6.0, (0, 2): 8.9, (1, 2): 8.9, (0, 1, 2): 5.9,
    }
    # (2) redundancy: a and b each fix the same error.
    cases["redundant"] = {
        (): 10.0, (0,): 7.0, (1,): 7.0, (2,): 8.5,
        (0, 1): 6.9, (0, 2): 5.5, (1, 2): 5.5, (0, 1, 2): 5.4,
    }
    # (3) negative increment: adding c on top of {a} makes the risk worse.
    cases["negative increment"] = {
        (): 10.0, (0,): 6.0, (1,): 9.0, (2,): 8.0,
        (0, 1): 5.5, (0, 2): 7.2, (1, 2): 7.5, (0, 1, 2): 6.4,
    }

    results = {}
    for title, J in cases.items():
        p(f"--- {title} ---")
        full = (0, 1, 2)
        for S in sorted(J, key=lambda s: (len(s), s)):
            p(f"   J({{{','.join(names[i] for i in S)}}}) = {J[S]:.1f}")
        # independent (leave-one-out) ranking
        loo = {}
        for i in range(3):
            rest = tuple(j for j in full if j != i)
            loo[i] = J[rest] - J[full]
        # empty-set marginals
        d0 = {i: J[()] - J[(i,)] for i in range(3)}
        rank_loo = sorted(range(3), key=lambda i: -loo[i])
        rank_d0 = sorted(range(3), key=lambda i: -d0[i])
        p("   delta_i(empty): " + ", ".join(
            f"{names[i]}={d0[i]:+.1f}" for i in range(3)))
        p("   LOO_i         : " + ", ".join(
            f"{names[i]}={loo[i]:+.1f}" for i in range(3)))
        res = {}
        for k in (1, 2):
            opt = min(subsets_of_size(3, k), key=lambda s: J[s])
            ind0 = tuple(sorted(rank_d0[:k]))
            indl = tuple(sorted(rank_loo[:k]))
            # greedy with the true set-conditional delta
            S = ()
            for _ in range(k):
                cand = [i for i in range(3) if i not in S]
                best = max(cand, key=lambda i: J[S] - J[tuple(sorted(S + (i,)))])
                S = tuple(sorted(S + (best,)))
            fmt = lambda s: "{" + ",".join(names[i] for i in s) + "}"
            p(f"   budget {k}: OPT {fmt(opt)} J={J[opt]:.1f} | "
              f"rank-by-delta(empty) {fmt(ind0)} J={J[ind0]:.1f} | "
              f"rank-by-LOO {fmt(indl)} J={J[indl]:.1f} | "
              f"greedy-true-delta {fmt(S)} J={J[S]:.1f}")
            res[k] = dict(opt=J[opt], ind_empty=J[ind0], ind_loo=J[indl],
                          greedy=J[S])
        # any negative increment?
        neg = []
        for S in J:
            for i in range(3):
                if i not in S:
                    d = J[S] - J[tuple(sorted(S + (i,)))]
                    if d < 0:
                        neg.append((S, i, d))
        if neg:
            for S, i, d in neg:
                p(f"   NEGATIVE: delta_{names[i]}({{"
                  f"{','.join(names[j] for j in S)}}}) = {d:+.1f}")
        results[title] = res
        p("")
    p("reading, per case:")
    p("  complementary: both independent rankings fail at SOME budget. rank-by-")
    p("    d(empty) cannot see {a,b} at all (delta_a(empty)=+0.1) and returns")
    p("    {a,c}; rank-by-LOO sees {a,b} at budget 2 but gets budget 1 wrong")
    p("    ({a}, J=9.9 vs OPT {c}, J=9.0). greedy-true-delta also returns {a,c}")
    p("    at budget 2: greedy is an approximation with no monotone/submodular")
    p("    guarantee, and this J is neither.")
    p("  redundant: rank-by-d(empty) spends its second pick on the redundant b")
    p("    ({a,b}, J=6.9 vs OPT {a,c}, J=5.5); rank-by-LOO gets budget 1 wrong.")
    p("  negative increment: delta_c({a}) = -1.2 and delta_c({a,b}) = -0.9, so")
    p("    a selector that can only add 'beneficial' actions is mis-specified.")
    p("no single independent ranking is right in all three; the LOO-from-All")
    p("proxy is right in one of them and wrong in the other two -- which is the")
    p("concrete reason the task book forbids calling it an optimal teacher.")
    p("")
    return results


# ==========================================================================
# PART 2 -- native-matching flip
# ==========================================================================


def hungarian_bruteforce(cost):
    """Min-cost assignment of rows (GT) to distinct columns (queries)."""
    n_gt = len(cost)
    n_q = len(cost[0])
    best = None
    for perm in itertools.permutations(range(n_q), n_gt):
        c = sum(cost[g][perm[g]] for g in range(n_gt))
        if best is None or c < best[0]:
            best = (c, perm)
    return best


def part2(out):
    p = out.append
    p("=" * 78)
    p("PART 2 -- native matching flips when one action is added")
    p("=" * 78)
    p("toy: 2 GT, 3 queries. cost[g][q] = matching cost of GT g to query q.")
    p("J = (sum of matched costs) + (unmatched-query 'no-object' cost).")
    p("")
    # Without action i: query q0 is the best for GT0, q1 for GT1.
    cost_S = [[1.0, 4.0, 5.0],
              [4.5, 2.0, 2.6]]
    # Adding action i sharpens query q2 a lot and slightly degrades q1
    # (its features shift because the region's residual is now present).
    cost_Si = [[1.0, 4.0, 5.0],
               [4.5, 2.4, 1.1]]
    noobj = 0.3
    JS, permS = hungarian_bruteforce(cost_S)
    JSi, permSi = hungarian_bruteforce(cost_Si)
    JS += noobj * (3 - 2)
    JSi += noobj * (3 - 2)
    p(f"   matching under S    : GT0->q{permS[0]}, GT1->q{permS[1]}  J(S)   = {JS:.2f}")
    p(f"   matching under S+i  : GT0->q{permSi[0]}, GT1->q{permSi[1]}  J(S+i) = {JSi:.2f}")
    true_delta = JS - JSi
    # row-level attribution with the OLD matching frozen:
    frozen = sum(cost_Si[g][permS[g]] for g in range(2)) + noobj
    frozen_delta = JS - frozen
    p(f"   true delta_i(S) (matching re-derived)      = {true_delta:+.2f}")
    p(f"   'same-GT local replacement' with the matching frozen at S")
    p(f"       J_frozen(S+i) = {frozen:.2f}  -> delta = {frozen_delta:+.2f}")
    p("")
    p("   the frozen-matching (row-level) label says the action HURTS;")
    p("   the set-level risk says it HELPS. A per-GT / per-row target and a")
    p("   Hamming distance over rows are therefore not substitutes for the")
    p("   set loss -- the same defect that withdrew the pairwise swap candidate")
    p("   (AUDIT_RESPONSE_CANDIDATE_002.md section 3).")
    p("")
    return dict(J_S=JS, J_Si=JSi, true_delta=true_delta,
                frozen_delta=frozen_delta,
                matching_S=list(permS), matching_Si=list(permSi))


# ==========================================================================
# PART 3 -- 16 actions (the 4x4 grid): exhaustive optimum vs selectors
# ==========================================================================


TIE_EPS = 1e-7


def argmax_low_index(cands, score):
    """The ONE selection rule used by every chain in this file.

    argmax of `score`; candidates within TIE_EPS of the maximum are tied and
    the tie goes to the LOWEST action index. Without the tolerance a fitted
    predictor's ~1e-10 numerical noise breaks exact ties arbitrarily and the
    comparison between selectors measures noise (observed and fixed during
    this task; see the harness self-check in PART 4).
    """
    vals = {i: score(i) for i in cands}
    top = max(vals.values())
    return min(i for i in cands if vals[i] >= top - TIE_EPS)


def make_value(n, seed, order3=True):
    """Synthetic VALUE v(S) = J(empty) - J(S) as a degree<=3 polynomial of the
    indicator vector, with planted complementary / redundant / harmful groups.

    Every planted magnitude is jittered so that no two marginals are exactly
    equal: outcomes then do not depend on the tie-break rule.
    """
    rng = random.Random(seed)
    u = [rng.uniform(0.0, 1.0) for _ in range(n)]
    c = {}
    t = {}
    idx = list(range(n))
    rng.shuffle(idx)
    small = lambda: rng.uniform(0.02, 0.08)
    # planted complementary pairs: worthless alone, valuable together
    for a, b in ((idx[0], idx[1]), (idx[2], idx[3])):
        u[a] = small()
        u[b] = small()
        c[(min(a, b), max(a, b))] = rng.uniform(2.2, 2.6)
    # planted redundant pairs: each strong alone, almost nothing extra together
    for a, b in ((idx[4], idx[5]), (idx[6], idx[7])):
        u[a] = rng.uniform(1.5, 1.7)
        u[b] = rng.uniform(1.5, 1.7)
        c[(min(a, b), max(a, b))] = -min(u[a], u[b]) + small()
    # planted harmful pair: adding the second on top of the first hurts
    a, b = idx[8], idx[9]
    u[a] = rng.uniform(1.1, 1.3)
    u[b] = rng.uniform(0.3, 0.5)
    c[(min(a, b), max(a, b))] = -rng.uniform(1.0, 1.2)
    # background weak pairwise terms
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in c and rng.random() < 0.25:
                c[(i, j)] = rng.uniform(-0.25, 0.25)
    if order3:
        # planted third-order group: only all three together pay off
        a, b, d = sorted((idx[10], idx[11], idx[12]))
        u[a], u[b], u[d] = small(), small(), small()
        t[(a, b, d)] = rng.uniform(2.8, 3.2)
    return u, c, t


def make_value_small(n, seed):
    """n = 8 instance for PART 4: one complementary pair, one redundant pair,
    one third-order group, one free action. Jittered, tie-free."""
    assert n == 8
    rng = random.Random(1000 + seed)
    u = [rng.uniform(0.0, 1.0) for _ in range(n)]
    c, t = {}, {}
    ids = list(range(n))
    rng.shuffle(ids)
    small = lambda: rng.uniform(0.02, 0.08)
    a, b = sorted(ids[0:2])
    u[a], u[b] = small(), small()
    c[(a, b)] = rng.uniform(2.2, 2.6)
    a, b = sorted(ids[2:4])
    u[a], u[b] = rng.uniform(1.5, 1.7), rng.uniform(1.5, 1.7)
    c[(a, b)] = -min(u[a], u[b]) + small()
    a, b, d = sorted(ids[4:7])
    u[a], u[b], u[d] = small(), small(), small()
    t[(a, b, d)] = rng.uniform(2.8, 3.2)
    return u, c, t


def value_table(n, u, c, t):
    """v[mask] for all 2^n masks."""
    N = 1 << n
    v = [0.0] * N
    pairs = [(1 << i | 1 << j, w) for (i, j), w in c.items()]
    trips = [(1 << i | 1 << j | 1 << k, w) for (i, j, k), w in t.items()]
    for m in range(N):
        s = 0.0
        mm = m
        i = 0
        while mm:
            if mm & 1:
                s += u[i]
            mm >>= 1
            i += 1
        for pm, w in pairs:
            if m & pm == pm:
                s += w
        for tm, w in trips:
            if m & tm == tm:
                s += w
        v[m] = s
    return v


def greedy_true(n, v, k):
    """Forward chain driven by the TRUE set-conditional increments."""
    S = 0
    for _ in range(k):
        cand = [i for i in range(n) if not S >> i & 1]
        bi = argmax_low_index(cand, lambda i: v[S | 1 << i] - v[S])
        S |= 1 << bi
    return S


def greedy_backward(n, v, k):
    """Start from All and drop the action whose removal costs least (true
    set-conditional value), until k remain. The natural direction for a
    method that starts from the native full network and skips compute."""
    S = (1 << n) - 1
    while popcount(S) > k:
        cand = [i for i in range(n) if S >> i & 1]
        bi = argmax_low_index(cand, lambda i: v[S ^ (1 << i)])
        S ^= 1 << bi
    return S


def part3(out):
    p = out.append
    p("=" * 78)
    p("PART 3 -- 16 actions (4x4 grid), exhaustive optimum vs selectors")
    p("=" * 78)
    p("synthetic value v(S) = J(empty) - J(S); higher is better. 20 seeds,")
    p("each with planted complementary, redundant, harmful and one 3-way group.")
    p("OPT is the exhaustive best subset of the given size (all C(16,k)).")
    p("All five selectors below use the TRUE values (a perfect teacher).")
    p("  rank d(empty) : independent, marginal at the empty set")
    p("  rank LOO(All) : independent, leave-one-out from the full set")
    p("  rank avg-marg : independent, marginal AVERAGED over all 2^15 sets S")
    p("                  not containing i (Banzhaf value) -- the target a")
    p("                  set-blind predictor converges to under uniform S")
    p("  fwd / bwd     : set-conditional greedy chains, add from empty /")
    p("                  remove from All")
    p("")
    n = 16
    budgets = (2, 4, 6, 8, 12)
    cols = ("ind0", "loo", "banz", "fwd", "bwd")
    INDEP = ("ind0", "loo", "banz")
    agg = {k: dict(opt=[], **{c: [] for c in cols},
                   **{c + "_hit": 0 for c in cols}) for k in budgets}
    neg_count = 0
    nonmono = 0
    full = (1 << n) - 1
    seeds = range(20)
    for seed in seeds:
        u, c, t = make_value(n, seed)
        v = value_table(n, u, c, t)
        d0 = [v[1 << i] - v[0] for i in range(n)]
        loo = [v[full] - v[full ^ (1 << i)] for i in range(n)]
        banz = []
        for i in range(n):
            bit = 1 << i
            s = 0.0
            for m in range(1 << n):
                if not m & bit:
                    s += v[m | bit] - v[m]
            banz.append(s / (1 << (n - 1)))
        r0 = sorted(range(n), key=lambda i: -d0[i])
        rl = sorted(range(n), key=lambda i: -loo[i])
        rb = sorted(range(n), key=lambda i: -banz[i])
        has_neg = any(v[m | 1 << i] - v[m] < -1e-12
                      for m in range(0, 1 << n, 37) for i in range(n)
                      if not m >> i & 1)
        neg_count += has_neg
        nonmono += (v[full] < max(v))
        for k in budgets:
            best = max((sum(1 << i for i in comb)
                        for comb in itertools.combinations(range(n), k)),
                       key=lambda m: v[m])
            sel = dict(ind0=sum(1 << i for i in r0[:k]),
                       loo=sum(1 << i for i in rl[:k]),
                       banz=sum(1 << i for i in rb[:k]),
                       fwd=greedy_true(n, v, k),
                       bwd=greedy_backward(n, v, k))
            a = agg[k]
            a["opt"].append(v[best])
            for cname, m in sel.items():
                a[cname].append(v[m])
                a[cname + "_hit"] += (abs(v[m] - v[best]) < 1e-9)
    mean = lambda xs: sum(xs) / len(xs)
    label = dict(ind0="d(empty)", loo="LOO(All)", banz="avg-marg",
                 fwd="fwd greedy", bwd="bwd greedy")
    p(f"{'k':>3} | {'OPT':>6} | " + " | ".join(
        f"{label[c]:>10} {'=OPT':>5}" for c in cols))
    res = {}
    winners = {}
    best_indep = {}
    best_chain = {}
    for k in budgets:
        a = agg[k]
        o = mean(a["opt"])
        cells = []
        for cname in cols:
            m = mean(a[cname])
            cells.append(f"{100*m/o:>9.0f}% {a[cname+'_hit']:>2}/20")
        p(f"{k:>3} | {o:>6.3f} | " + " | ".join(cells))
        res[k] = dict(opt=o, **{c: mean(a[c]) for c in cols},
                      **{c + "_hits": a[c + "_hit"] for c in cols})
        winners[k] = max(cols, key=lambda c: mean(a[c]))
        best_indep[k] = max(INDEP, key=lambda c: mean(a[c]))
        best_chain[k] = max(("fwd", "bwd"), key=lambda c: mean(a[c]))
    p("")
    p(f"seeds with at least one negative increment    : {neg_count}/20")
    p(f"seeds where v(All) < max_S v(S) (non-monotone): {nonmono}/20")
    p("")
    p("best independent ranking vs best set-conditional chain, per budget")
    p("(computed, not asserted):")
    indep_wins = []
    for k in budgets:
        a = agg[k]
        bi, bc = best_indep[k], best_chain[k]
        vi, vc, o = mean(a[bi]), mean(a[bc]), mean(a["opt"])
        tag = "INDEPENDENT wins" if vi > vc + 1e-12 else "chain wins"
        if vi > vc + 1e-12:
            indep_wins.append(k)
        p(f"   k={k:>2}: {label[bi]:<9} {100*vi/o:>4.0f}%  vs  "
          f"{label[bc]:<10} {100*vc/o:>4.0f}%   -> {tag} "
          f"by {100*abs(vi-vc)/o:.0f} pts")
    chain_wins = [k for k in budgets if k not in indep_wins]
    p("")
    p("reading (holds for this synthetic generator only):")
    p("  1. the set-conditional structure is REPRESENTABLE: a chain beats")
    p(f"     every independent ranking at budgets {chain_wins}.")
    if indep_wins:
        p(f"  2. it is NOT uniformly better: at budgets {indep_wins} an")
        p("     independent ranking beats the better of two chains that are")
        p("     each driven by a PERFECT set-conditional teacher.")
    else:
        p("  2. no budget here has an independent ranking beating both chains.")
    p("  3. forward and backward chains disagree and neither dominates, so")
    p("     the 'one and only one' selection order the task book requires")
    p("     cannot be fixed on principle; it would have to be chosen on real")
    p("     J(S), which this task forbids computing.")
    p("  4. even with a perfect teacher the chains miss the exact optimum on")
    p("     most seeds; a learned predictor can only be worse.")
    p("Nothing here says the real detector has such interactions.")
    p("")
    res["winners"] = {str(k): winners[k] for k in budgets}
    res["best_independent"] = {str(k): best_indep[k] for k in budgets}
    res["best_chain"] = {str(k): best_chain[k] for k in budgets}
    res["budgets_where_independent_beats_both_chains"] = indep_wins
    return res


# ==========================================================================
# PART 4 -- the proposed predictor and the pairwise-order limit (n = 8)
# ==========================================================================


def feats(S_mask, i, n, order):
    """Features of (S, i) for a predictor of delta_i(S).

    order 0: one-hot of i only                      -> independent score
    order 1: + [i, j in S]                          -> pairwise in v
    order 2: + [i, j in S, k in S]                  -> third order in v
    """
    f = [0.0] * n
    f[i] = 1.0
    if order >= 1:
        g = [0.0] * (n * n)
        for j in range(n):
            if S_mask >> j & 1:
                g[i * n + j] = 1.0
        f += g
    if order >= 2:
        h = [0.0] * (n * n * (n - 1) // 2)
        col = 0
        for j in range(n):
            for k in range(j + 1, n):
                if S_mask >> j & 1 and S_mask >> k & 1:
                    h[i * (n * (n - 1) // 2) + col] = 1.0
                col += 1
        f += h
    return f


def fit_and_greedy(n, v, order, k):
    """Exact LS fit of d_hat on ALL (S, i) pairs, then forward greedy."""
    rows, ys = [], []
    for S in range(1 << n):
        for i in range(n):
            if not S >> i & 1:
                rows.append((S, i))
                ys.append(v[S | 1 << i] - v[S])
    A = [feats(S, i, n, order) for S, i in rows]
    w = solve_lstsq(A, ys)
    pred = [sum(a_ * w_ for a_, w_ in zip(r, w) if a_) for r in A]
    rmse = (sum((y - q) ** 2 for y, q in zip(ys, pred)) / len(ys)) ** 0.5
    lut = {rc: q for rc, q in zip(rows, pred)}
    S, trace = 0, []
    for _ in range(k):
        cand = [i for i in range(n) if not S >> i & 1]
        bi = argmax_low_index(cand, lambda i: lut[(S, i)])
        trace.append((bi, lut[(S, bi)], v[S | 1 << bi] - v[S]))
        S |= 1 << bi
    return S, rmse, trace


def part4(out):
    p = out.append
    p("=" * 78)
    p("PART 4 -- one proposed predictor, three orders, exact fit")
    p("=" * 78)
    p("predictor: d_hat_i(S) = w_i + sum_{j in S} A_ij + sum_{j<k in S} B_ijk")
    p("   order 0 keeps only w_i            (= independent risk ranking)")
    p("   order 1 adds A_ij                 (the proposed structure)")
    p("   order 2 adds B_ijk                (reference only, not proposed)")
    p("fitted by least squares on ALL (S, i) pairs, noise-free -- the best case")
    p("any learner could reach -- then used in a forward greedy chain with the")
    p(f"shared rule: argmax, ties within {TIE_EPS:g} go to the lowest index.")
    p("")
    # ---- (a) aggregate, n = 8 -------------------------------------------
    n = 8
    orders = (0, 1, 2)
    stats = {o: dict(rmse=[], val=[], hit=0) for o in orders}
    optv, truev = [], []
    true_hit = 0
    harness_ok = 0
    k = 4
    N_SEEDS = 12
    for seed in range(N_SEEDS):
        u, c, t = make_value_small(n, seed)
        v = value_table(n, u, c, t)
        best = max((sum(1 << i for i in comb)
                    for comb in itertools.combinations(range(n), k)),
                   key=lambda m: v[m])
        optv.append(v[best])
        gt = greedy_true(n, v, k)
        truev.append(v[gt])
        true_hit += (abs(v[gt] - v[best]) < 1e-9)
        for o in orders:
            S, rmse, _ = fit_and_greedy(n, v, o, k)
            stats[o]["rmse"].append(rmse)
            stats[o]["val"].append(v[S])
            stats[o]["hit"] += (abs(v[S] - v[best]) < 1e-9)
            if o == 2:
                harness_ok += (S == gt)
    mean = lambda xs: sum(xs) / len(xs)
    o_mean = mean(optv)
    p(f"harness self-check: the order-2 fit is exact (v is cubic), so its")
    p(f"chain must equal the true-delta chain on every seed: "
      f"{harness_ok}/{N_SEEDS} "
      f"{'PASS' if harness_ok == N_SEEDS else 'FAIL'}")
    if harness_ok != N_SEEDS:
        raise SystemExit("harness self-check failed: exact fit != true chain")
    p("")
    p(f"(a) n = 8, budget k = {k}, {N_SEEDS} seeds; mean exhaustive optimum "
      f"= {o_mean:.3f}")
    p(f"{'selector':>22} {'#params':>8} {'fit RMSE':>9} {'value':>8} "
      f"{'% of OPT':>9} {'=OPT':>6}")
    res = {}
    for o in orders:
        npar = n + (n * n if o >= 1 else 0) + (n * n * (n - 1) // 2 if o >= 2 else 0)
        s = stats[o]
        p(f"{'fitted order ' + str(o):>22} {npar:>8} {mean(s['rmse']):>9.4f} "
          f"{mean(s['val']):>8.3f} {100*mean(s['val'])/o_mean:>8.1f}% "
          f"{s['hit']:>3}/{N_SEEDS}")
        res[o] = dict(params=npar, rmse=mean(s["rmse"]), value=mean(s["val"]),
                      frac_opt=mean(s["val"]) / o_mean, hits=s["hit"])
    p(f"{'true delta (teacher)':>22} {'-':>8} {0.0:>9.4f} {mean(truev):>8.3f} "
      f"{100*mean(truev)/o_mean:>8.1f}% {true_hit:>3}/{N_SEEDS}")
    gain01 = mean(stats[1]["val"]) - mean(stats[0]["val"])
    gain12 = mean(stats[2]["val"]) - mean(stats[1]["val"])
    p("")
    p(f"    order 0 -> 1 changes the greedy value by {gain01:+.3f} "
      f"({100*gain01/o_mean:+.1f}% of OPT)")
    p(f"    order 1 -> 2 changes the greedy value by {gain12:+.3f} "
      f"({100*gain12/o_mean:+.1f}% of OPT); order-1 fit RMSE "
      f"{mean(stats[1]['rmse']):.3f},")
    p(f"    order-2 fit RMSE {mean(stats[2]['rmse']):.3f}")
    if gain01 < -1e-9:
        p("    -> on these seeds the pairwise predictor is WORSE than the")
        p("       independent (set-averaged) ranking.")
    if gain12 < -1e-9:
        p("    -> and the exact order-2 predictor, which reproduces the perfect")
        p("       teacher, is worse again. A better fit of delta_i(S) gave a")
        p("       WORSE forward chain: the chain is myopic about groups whose")
        p("       members pay off only together, while the order-0 weight is")
        p("       the marginal averaged over all S and so already carries the")
        p("       group's payoff.")
    if abs(gain12) < 1e-9:
        p("    -> the third-order residual is real in the FIT but changed no")
        p("       greedy outcome on average.")
    p("")

    # ---- (b) searched explicit counterexample, n = 5 --------------------
    p("(b) explicit instance where the pairwise (order-1) predictor fails")
    p("    although a perfect teacher and the order-2 predictor succeed.")
    p("    Found by search over seeded random instances (n = 5, k = 3) whose")
    p("    true chain has no tie at any step; reported as found, or as not")
    p("    found.")

    def chain_has_tie(n_, v_, k_):
        S = 0
        for _ in range(k_):
            cand = [i for i in range(n_) if not S >> i & 1]
            ds = sorted((v_[S | 1 << i] - v_[S] for i in cand), reverse=True)
            if len(ds) > 1 and ds[0] - ds[1] < 1e-6:
                return True
            S |= 1 << argmax_low_index(cand, lambda i: v_[S | 1 << i] - v_[S])
        return False

    found = None
    tried = 0
    n5, k5 = 5, 3
    for seed in range(4000):
        rng = random.Random(50000 + seed)
        u = [round(rng.uniform(0.0, 1.2), 2) for _ in range(n5)]
        c, t = {}, {}
        tri = tuple(sorted(rng.sample(range(n5), 3)))
        for q in tri:
            u[q] = round(rng.uniform(0.0, 0.3), 2)
        t[tri] = round(rng.uniform(1.5, 3.5), 2)
        v = value_table(n5, u, c, t)
        if chain_has_tie(n5, v, k5):
            continue
        tried += 1
        best = max((sum(1 << i for i in comb)
                    for comb in itertools.combinations(range(n5), k5)),
                   key=lambda m: v[m])
        gt = greedy_true(n5, v, k5)
        if abs(v[gt] - v[best]) > 1e-9:
            continue
        S1, r1, tr1 = fit_and_greedy(n5, v, 1, k5)
        S2, r2, tr2 = fit_and_greedy(n5, v, 2, k5)
        if S2 == gt and v[S1] < v[best] - 1e-6:
            found = dict(seed=50000 + seed, u=u, tri=list(tri), t=t[tri],
                         opt=bits(best, n5), opt_value=v[best],
                         order1=bits(S1, n5), order1_value=v[S1],
                         order1_rmse=r1, order2=bits(S2, n5),
                         order2_value=v[S2], order2_rmse=r2,
                         order1_trace=[(i, round(ph, 3), round(tr, 3))
                                       for i, ph, tr in tr1])
            break
    if found:
        p(f"    FOUND after {tried} instances (seed {found['seed']}):")
        p(f"      v(S) = sum_i u_i [i in S] + {found['t']} * "
          f"[{found['tri']} all in S],  u = {found['u']}")
        p(f"      OPT (k=3)          = {found['opt']}  v = {found['opt_value']:.2f}")
        p(f"      true-delta greedy  = {found['opt']}  v = {found['opt_value']:.2f}")
        p(f"      order-2 greedy     = {found['order2']}  v = "
          f"{found['order2_value']:.2f}  (fit RMSE {found['order2_rmse']:.4f})")
        p(f"      order-1 greedy     = {found['order1']}  v = "
          f"{found['order1_value']:.2f}  (fit RMSE {found['order1_rmse']:.4f})")
        p("      order-1 chain (pick, predicted delta, TRUE delta):")
        for i, ph, tr in found["order1_trace"]:
            p(f"         pick {i}: predicted {ph:+.3f}, true {tr:+.3f}")
    else:
        p(f"    NOT FOUND in {tried} instances: no instance separated order 1")
        p("    from order 2 under these generator settings.")
    p("")
    p("pairwise limit, stated exactly: if v(S) contains t*[a,b,d all in S],")
    p("then delta_d(S) contains t*[a in S][b in S], a PRODUCT of two")
    p("memberships. An order-1 predictor is additive over the members of S,")
    p("so for every choice of A it satisfies")
    p("   d_hat_d({a,b}) - d_hat_d({a}) - d_hat_d({b}) + d_hat_d({}) = 0,")
    p("while the truth gives t. That residual does not shrink with data.")
    p("The proposed structure therefore represents pairwise complementarity")
    p("and redundancy and NOT third-order groups: a declared limit.")
    p("")
    res["true_delta"] = dict(value=mean(truev), frac_opt=mean(truev) / o_mean,
                             hits=true_hit)
    res["gain_order0_to_1"] = gain01
    res["gain_order1_to_2"] = gain12
    res["counterexample"] = found
    res["counterexample_instances_tried"] = tried
    return res


# ==========================================================================
# PART 5 -- teacher cost: forwards and matchings per training example
# ==========================================================================


def part5(out):
    p = out.append
    p("=" * 78)
    p("PART 5 -- teacher cost model (counts exact; seconds from measured anchors)")
    p("=" * 78)
    # measured anchors
    UPDATES = 18240                 # e061..e120, L0 / R27 / S15 same schedule
    IMGS_PER_UPDATE = 16            # physical 8 x accumulation 2
    FWD_MS_IMG = 1634.6 / 64        # S15 e120 EMA, FP32 batch4, model GPU only
    L0_UPDATE_S = 1.157             # L0 median s/update (L canvas, fwd+bwd)
    S15_UPDATE_S = 30506 / 18240    # S15 formal-002 sum of update time / updates
    S15_TRAIN_H = 30506 / 3600
    p(f"anchors: {UPDATES} updates x {IMGS_PER_UPDATE} images; teacher forward "
      f"(S15 EMA, eval, H) = {FWD_MS_IMG:.2f} ms/image model-GPU (measured,")
    p(f"         FP32 batch4); S15 training update = {S15_UPDATE_S:.3f} s "
      f"(measured, {S15_TRAIN_H:.1f} h of update time per run)")
    p("the teacher is a DENSE masked forward (multiply-by-mask is allowed for")
    p("the teacher: it need not save compute), one full forward per distinct S,")
    p("plus one native Hungarian matching per S. Matching time is NOT included")
    p("below (unmeasured for H); it only adds.")
    p("")

    def chain(n, k):
        """Forwards to run a true-delta greedy chain of k picks over n actions:
        J(S_t) once, then J(S_t + i) for every remaining i."""
        return 1 + sum(n - t for t in range(k))

    rows = []
    p(f"{'action space':<28} {'target':<34} {'fwd/img':>8} {'s/update':>9} "
      f"{'x S15 upd':>9} {'GPU-h/run':>10}")
    specs = [
        ("16 macro (4x4)", "LOO from All (forbidden proxy)", 16 + 1),
        ("16 macro (4x4)", "greedy chain k=4, true delta", chain(16, 4)),
        ("16 macro (4x4)", "greedy chain k=8, true delta", chain(16, 8)),
        ("16 macro (4x4)", "greedy chain k=16 (full order)", chain(16, 16)),
        ("16 macro (4x4)", "all 2^16 subsets (exact J)", 1 << 16),
        ("4 row-bands", "greedy chain k=2, true delta", chain(4, 2)),
        ("4 row-bands", "all 2^4 subsets (exact J)", 1 << 4),
        ("16 macro (4x4)", "1 sampled S + 1 sampled S+i", 2),
    ]
    for space, target, f in specs:
        s_upd = f * IMGS_PER_UPDATE * FWD_MS_IMG / 1000.0
        gh = s_upd * UPDATES / 3600.0
        p(f"{space:<28} {target:<34} {f:>8d} {s_upd:>9.1f} "
          f"{s_upd / S15_UPDATE_S:>8.1f}x {gh:>10.1f}")
        rows.append(dict(space=space, target=target, forwards_per_image=f,
                         seconds_per_update=s_upd,
                         ratio_to_S15_update=s_upd / S15_UPDATE_S,
                         gpu_hours_per_run=gh))
    p("")
    # images consumed per epoch, DERIVED from measured records (not a dataset
    # constant quoted from memory): 18,240 updates over e061..e120 = 304
    # updates/epoch, x 16 images/update, drop_last_train = true.
    EPOCHS = 60
    N_TRAIN = UPDATES // EPOCHS * IMGS_PER_UPDATE      # 4,864
    triples = UPDATES * IMGS_PER_UPDATE
    cells = 16 * (1 << 15)
    p("the last row is the only one cheaper than the training it supervises.")
    p(f"It yields ONE (S, i, delta) triple per image per update: {triples:,}")
    p(f"triples over the run = {triples // N_TRAIN} visits per training image "
      f"(one per epoch; {N_TRAIN:,} images/epoch,")
    p(f"derived from {UPDATES} updates / {EPOCHS} epochs x {IMGS_PER_UPDATE}), "
      f"against {cells:,} distinct (S, i) cells")
    p("per image at 16 actions. It is a stochastic estimate of the same")
    p("target, not a cheaper exact teacher: each visit is a difference of two")
    p("300-query matched losses whose matching may flip (PART 2), and that")
    p("variance is unmeasured.")
    p("")
    p("why it cannot be cached: section 4 fixes x as the completed augmented")
    p("microbatch. Augmentation is redrawn each epoch, so a per-image cache")
    p("supervises a different input than the one the selector sees. A fixed-")
    p("view cache (no augmentation) costs one row of the table ONCE per image")
    p(f"({N_TRAIN:,} images):")
    for space, target, f in specs[1:4]:
        h = f * N_TRAIN * FWD_MS_IMG / 1000.0 / 3600.0
        p(f"   {target:<34} {f * N_TRAIN:>9,} forwards = {h:>6.1f} GPU-h, "
          f"but the label then refers to an un-augmented view")
    p("")
    return rows


# ==========================================================================
# PART 6 -- what is available before C3, and what routing costs
# ==========================================================================


def part6(out):
    p = out.append
    p("=" * 78)
    p("PART 6 -- deploy-time information budget")
    p("=" * 78)
    # Cost anchors are READ from script 1's result file, not retyped, so the
    # two scripts cannot drift apart (the first pass of this task retyped
    # them and carried script 1's MAC undercount into this part).
    src = os.path.join(HERE, "check_backbone_shapes.result.json")
    if not os.path.isfile(src):
        raise SystemExit("run check_backbone_shapes.py --json first")
    with open(src, encoding="utf-8") as f:
        s1 = json.load(f)
    MODEL_H = s1["measured"]["S15_model_ms_batch4"]   # S15 at H (L0 cost4)
    MODEL_L = s1["measured"]["A0_model_ms_batch4"]    # A0 at L (L0 cost4)
    BEST_SAVING = s1["best_case_ms_batch4"]           # ceiling: 100% skip, 0 overhead
    BACKBONE_H = s1["measured"]["backbone_ms"]        # R26 arm N measured
    # stem + C2 MAC share of the H backbone (script 1) and the L/H pixel ratio
    STEM_C2_SHARE = s1["stem_and_C2_macs_G"] / s1["backbone_macs_G"]
    PIX_L_OVER_H = (768 * 1344) / (1152 * 2016)
    PARTIAL_L = BACKBONE_H * STEM_C2_SHARE * PIX_L_OVER_H   # MAC-scaled ESTIMATE
    # receptive field of C2, computed from the layer list in presnet.py:
    # conv1_1 (k3,s2), conv1_2 (k3,s1), conv1_3 (k3,s1), max-pool (k3,s2),
    # then res_layers[0] = 2 BasicBlocks x 2 convs (k3,s1).
    rf, jump = 1, 1
    for kk, ss in [(3, 2), (3, 1), (3, 1), (3, 2)] + [(3, 1)] * 4:
        rf += (kk - 1) * jump
        jump *= ss
    p("computed before the first maskable block (C3 block0): conv1 (3 convs),")
    p("max-pool, res_layers[0] (2 BasicBlocks) -> C2: 64 ch, stride 4, 288x504.")
    p("nothing from C3/C4/C5, the HybridEncoder or the decoder exists yet.")
    p("")
    p("option A  selector reads pooled C2 per macro-region + the set indicator")
    p("          cost ~ k tiny MLP calls: negligible; information = 2 residual")
    p(f"          blocks deep, receptive field {rf} input px (computed), "
      f"stride {jump}.")
    p("option B  selector reads a full L-resolution pass of the same detector")
    p(f"          cost = {MODEL_L:.1f} ms/batch4 (measured A0) = "
      f"{100*MODEL_L/MODEL_H:.0f}% of the H model,")
    p(f"          against a best-case saving of {BEST_SAVING:.1f} ms "
      f"({100*BEST_SAVING/MODEL_H:.0f}%).")
    p(f"          net = {BEST_SAVING - MODEL_L:+.1f} ms/batch4 -> ALWAYS a loss, "
      f"even at 100% skip and zero overhead.")
    p("          this is also R27-P's routing source (L encoder top-300 scores).")
    p("option C  selector reads the detector's own decoder output")
    p("          not available: it is downstream of the blocks being gated.")
    p("option D  selector reads an extra stem+C2 pass on the L-downsampled input")
    p(f"          cost ~ {PARTIAL_L:.1f} ms/batch4 (ESTIMATE: measured backbone "
      f"{BACKBONE_H:.1f} ms x")
    p(f"          stem+C2 MAC share {100*STEM_C2_SHARE:.0f}% x pixel ratio "
      f"{PIX_L_OVER_H:.3f}) = {100*PARTIAL_L/BEST_SAVING:.0f}% of the best-case")
    p("          saving, and still only stride-4 features with no encoder or")
    p("          decoder: it buys a wider field of view, not detection semantics.")
    p("")
    p("so A is the only free option and D the only paid one that can net")
    p("positive. Both must predict a property of the 300-query matched loss")
    p("AFTER encoder and decoder from stride-4 features that have seen")
    p("neither. Whether those features carry that signal is an empirical")
    p("question this task may not measure.")
    p("")
    return dict(model_H_ms=MODEL_H, model_L_ms=MODEL_L,
                best_saving_ms=BEST_SAVING,
                optionB_net_ms=BEST_SAVING - MODEL_L,
                optionD_cost_ms_estimate=PARTIAL_L,
                c2_receptive_field_px=rf, c2_stride=jump)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(newline="\n")
    except Exception:
        pass
    out = []
    r1 = part1(out)
    r2 = part2(out)
    r3 = part3(out)
    r4 = part4(out)
    r5 = part5(out)
    r6 = part6(out)
    print("\n".join(out))
    if args.json:
        with open(os.path.join(HERE, "check_set_risk.result.json"), "w",
                  encoding="utf-8", newline="\n") as f:
            json.dump(dict(part1=r1, part2=r2, part3=r3, part4=r4, part5=r5,
                           part6=r6), f, indent=2, ensure_ascii=False)
            f.write("\n")
        print("[written] check_set_risk.result.json")


if __name__ == "__main__":
    sys.exit(main())
