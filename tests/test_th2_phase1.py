"""TH2 phase 1 (TASK_TH2_PHASE1_001): the pure helpers.  No model, no data file."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "r28"))
import th2_core as T  # noqa: E402


def test_folds_alternate_in_sorted_order():
    ids = ["c", "a", "d", "b", "e"]
    assert T.folds(ids).tolist() == [0, 0, 1, 1, 0]           # sorted: a b c d e -> A B A B A


def test_draws_depend_on_id_and_index_only_and_balance_directions():
    a = [T.draw("img", k) for k in range(1, 49)]
    assert a == [T.draw("img", k) for k in range(1, 49)] and a != [T.draw("other", k) for k in range(1, 49)]
    assert all(0 <= r < 12 and 0 <= c < 21 for r, c, _ in a)
    up = np.array([u for _, _, u in a])
    assert up.sum() == 24 and up[0::2].sum() == 12 and up[1::2].sum() == 12
    with pytest.raises(ValueError):
        T.draw("img", 0)


def test_block_is_clipped_and_perturbation_touches_only_the_block():
    assert T.block_mask(0, 0, (3, 5)).sum() == 2 * 3 and T.block_mask(5, 10, (3, 5)).sum() == 15
    assert T.block_mask(11, 20, (5, 9)).sum() == 3 * 5
    d = np.arange(252, dtype=np.float64).reshape(12, 21) + 1
    out, m = T.perturb(d, "img", 1, 4.0, (3, 5))
    assert T.draw("img", 1)[2] and np.array_equal(out[~m], d[~m]) and np.array_equal(out[m], d[m] * 4)
    out, m = T.perturb(d, "img", 2, 4.0, (3, 5))
    assert np.array_equal(out[m], d[m] / 4)


def test_amplitude_choice():
    assert T.choose_amplitude({2.0: 0.05, 3.0: 0.09, 4.0: 0.13, 6.0: 0.18, 8.0: 0.22}) == (4.0, True)
    assert T.choose_amplitude({2.0: 0.01, 3.0: 0.02, 4.0: 0.03, 6.0: 0.05, 8.0: 0.07}) == (None, False)
    assert T.choose_amplitude({2.0: 0.2, 3.0: 0.3, 4.0: 0.4, 6.0: 0.5, 8.0: 0.6}) == (2.0, True)


def test_top600_follows_the_frozen_ranking_rule():
    rng = np.random.default_rng(0)
    s = rng.random((300, 10)).astype(np.float32)
    s[5, 3] = s[2, 7] = np.float32(2.0)                        # a tie: query id decides
    b = np.concatenate([rng.random((300, 2)) * 0.5 + 0.25, rng.random((300, 2)) * 0.2 + 0.01], 1).astype(np.float32)
    cat, score, box, q = T.top600(s, b, 1000, 500)
    assert len(cat) == 600 and (q[0], cat[0], q[1], cat[1]) == (2, 8, 5, 4)
    assert np.all(np.diff(score) <= 0) and box.dtype == np.float64
    half = b[2, 2:] / np.float32(2)
    want = np.concatenate([b[2, :2] - half, b[2, :2] + half]) * np.array([1000, 500, 1000, 500], dtype=np.float32)
    assert box[0].tolist() == [float(v) for v in want]


def test_outcomes_same_class_best_iou_and_q50():
    gt = np.array([[0, 0, 10, 10], [100, 100, 110, 110]], dtype=np.float64)
    cand = np.array([[0, 0, 10, 10], [0, 0, 10, 20], [0, 0, 10, 10], [200, 200, 210, 210]], dtype=np.float64)
    iou, q = T.outcomes(gt, np.array([1, 2]), np.array([2, 1, 1, 2]), np.array([0.9, 0.8, 0.3, 0.7]), cand)
    assert iou.tolist() == [1.0, 0.0] and q.tolist() == [0.8, 0.0]      # the class-2 box on GT 1 is ignored; IoU 0.5 counts
    iou, q = T.outcomes(gt, np.array([1, 2]), np.array([], dtype=int), np.array([]), np.zeros((0, 4)))
    assert iou.tolist() == [0.0, 0.0]


def test_bins():
    assert T.size_bin([255, 256, 1024, 1025, 4095, 4096, 9215, 9216]).tolist() == [0, 1, 1, 2, 2, 3, 3, 4]
    assert T.g_bin([0.99, 1.0, 1.19, 1.2, 1.4]).tolist() == [0, 1, 1, 2, 3]
    assert T.crowd_bin([0, 1, 3, 4, 9, 10]).tolist() == [0, 1, 1, 2, 2, 3]
    assert T.omega([16, 32, 48, 64, 80]).tolist() == [1.0, 1.0, 0.5, 0.0, 0.0]


def test_crowd_counts_within_an_image_only():
    image = np.array([0, 0, 0, 1])
    n = T.crowd_counts(image, [0, 60, 200, 0], [0, 0, 0, 0])
    assert n.tolist() == [1, 1, 0, 0]


def test_ratio_and_bootstrap_interval():
    w = T.boot_weights(4, n=200, seed=1)
    assert w.shape == (200, 4) and np.all(w.sum(1) == 4)
    r = T.ratio(np.array([1, 0, 1, 1.0]), np.ones(4), np.array([0, 1, 2, 3]), 4, w, scale=100.0)
    assert r["value"] == 75.0 and r["ci95"][0] <= 75.0 <= r["ci95"][1]


def test_slope_recovers_a_known_coefficient():
    rng = np.random.default_rng(0)
    dx = rng.normal(0, 0.1, (50, 8))
    dy = 0.3 * dx
    image = np.arange(50) % 5
    w = T.boot_weights(5, n=50)
    s = T.slope(dx, dy, image, 5, w, np.ones_like(dx, dtype=bool))
    assert abs(s["value"] - 3.0) < 1e-9 and s["pairs"] == 400      # 0.3 per log g = 3 pp per 0.1
    b = T.per_gt_slope(dx, dy)
    assert np.allclose(b[np.isfinite(b)], 0.3)


def test_fit_slope_model_recovers_size_dependence():
    rng = np.random.default_rng(1)
    s = rng.uniform(4, 150, 400)
    phi = T.m1_basis(s)
    theta = np.array([0.5, -0.2, 0.1, 0.05, 0.0, 0.02])
    dx = rng.normal(0, 0.1, (400, 6))
    dy = (phi @ theta)[:, None] * dx
    assert np.allclose(T.fit_slope_model(phi, dx, dy), theta, atol=1e-8)
    f = {"s": s, "category": rng.integers(1, 11, 400), "occlusion": rng.integers(0, 3, 400), "truncation": rng.integers(0, 2, 400),
         "log_aspect": rng.normal(0, 0.3, 400), "crowd": rng.integers(0, 20, 400), "g0": rng.uniform(0.8, 1.6, 400)}
    assert T.m2_basis(f).shape == (400, 6 + 2 * 16)


def test_auc_and_correlation():
    assert T.auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0 and T.auc([1, 1, 1, 1], [0, 1, 0, 1]) == 0.5
    assert T.auc([1, 2], [1, 1]) is None
    x = np.arange(20.0)
    image = np.arange(20) % 4
    w = T.boot_weights(4, n=20)
    c = T.corr(T.corr_sums(x, 2 * x + 1, image, 4), w)
    assert abs(c["value"] - 1.0) < 1e-12
    d = T.corr_difference(T.corr_sums(x, x, image, 4), T.corr_sums(x, x, image, 4), w)
    assert d["value"] == 0.0 and d["ci_contains_zero"] is True


def test_rank_remap_keeps_the_values_and_follows_the_ranks():
    rng = np.random.default_rng(2)
    t = rng.random((12, 21))
    v = rng.random((12, 21))
    out = T.rank_remap(v, t)
    assert np.array_equal(np.sort(out.ravel()), np.sort(t.ravel()))
    assert np.array_equal(np.argsort(-out.ravel(), kind="stable"), np.argsort(-v.ravel(), kind="stable"))
    assert np.array_equal(T.rank_remap(np.zeros((12, 21)), t), t)           # no information in the value map: unchanged
    assert np.array_equal(T.rank_remap(t, t), t)


def test_weighted_demand_and_blur():
    b = np.array([[0.0, 0.0, 0.01, 0.01], [0.99, 0.99, 1.0, 1.0]])
    d = T.weighted_demand(b, [1.0, 2.0])
    assert d[0, 0] == 1.0 and d[11, 20] == 2.0 and d.sum() == 3.0
    assert abs(T.blur3(d)[0, 0] - 4.0 / 9.0) < 1e-12 and abs(T.blur3(np.ones((12, 21))).max() - 1.0) < 1e-12


def test_waterfill_area_respects_the_budget_and_the_bounds():
    count = np.zeros(252)
    count[:10] = 5
    a = T.waterfill_area(count, 4 / 9, 9 / 4)
    assert abs(a.mean() - 1.0) < 1e-6 and a.min() >= 4 / 9 - 1e-12 and a.max() <= 9 / 4 + 1e-12
    assert np.allclose(a[:10], 9 / 4) and np.allclose(T.waterfill_area(np.zeros(252), 4 / 9, 9 / 4), 1.0)


def test_verdict_rule():
    base = {"small_hit75_pp": [0.6, 0.7, 0.5], "medium_hit75_pp": [-0.1, 0.0, -0.2], "aps_pp": [0.3, 0.2, 0.25],
            "ap75s_pp": [0.3, 0.2, 0.4], "m2_increment_ci_contains_zero": False, "m1_shape_agrees": False, "divergent_share": 0.3}
    assert T.verdict(base)["label"] == "GO"
    assert T.verdict({**base, "small_hit75_pp": [0.3, 0.3, 0.3]})["label"] == "UNDECIDED"
    assert T.verdict({**base, "small_hit75_pp": [0.1, 0.2, 0.1]})["label"] == "STOP"
    assert T.verdict({**base, "small_hit75_pp": [0.9, 0.9, -0.1]})["label"] == "STOP"
    assert T.verdict({**base, "aps_pp": [0.5, 0.5, -0.2]})["label"] == "UNDECIDED"
    assert T.verdict({**base, "small_hit75_pp": [0.3, 0.3, 0.3], "divergent_share": 0.05})["label"] == "STOP"
    assert T.verdict({**base, "small_hit75_pp": [0.3, 0.3, 0.3], "m2_increment_ci_contains_zero": True, "m1_shape_agrees": True})["label"] == "STOP"
