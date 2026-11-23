"""TH2 B0 (TASK_TH2_B0_001): the pure helpers on synthetic cases.  No data file, no model."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "r28"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import th2b0_core as C  # noqa: E402


def demand(seed, n=3):
    rng = np.random.default_rng(seed)
    d = rng.gamma(0.4, 1.5, size=(n, 12, 21))
    d[rng.random(d.shape) < 0.5] = 0.0
    return d


def official_nonseparable(saliency, fwhm):
    """RecasensSaliencyToGridMixin.nonseparable_saliency_to_grid of tchittesh/fovea @ c8707bd2 up to the node grid,
    transcribed statement by statement (FP64 on the CPU instead of FP32 on CUDA; grid_shape (31, 51), anti_crop)."""
    grid_shape = (31, 51)
    padding_size = min(grid_shape) - 1
    total_shape = tuple(dim + 2 * padding_size for dim in grid_shape)
    size = 2 * padding_size + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    filt = torch.from_numpy(np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm ** 2)).unsqueeze(0).unsqueeze(0)
    P_basis = torch.zeros(2, *total_shape, dtype=torch.float64)
    for k in range(2):
        for i in range(total_shape[0]):
            for j in range(total_shape[1]):
                P_basis[k, i, j] = k * (i - padding_size) / (grid_shape[0] - 1.0) + (1.0 - k) * (j - padding_size) / (grid_shape[1] - 1.0)
    p = padding_size
    saliency = F.pad(saliency, (p, p, p, p), mode="reflect")
    N = saliency.shape[0]
    P = torch.zeros(1, 2, *total_shape, dtype=torch.float64)
    P[0, :, :, :] = P_basis
    P = P.expand(N, 2, *total_shape)
    saliency_cat = torch.cat((saliency, saliency), 1)
    weights = F.conv2d(saliency, filt)
    weighted_offsets = torch.mul(P, saliency_cat).view(-1, 1, *total_shape)
    weighted_offsets = F.conv2d(weighted_offsets, filt).view(-1, 2, *grid_shape)
    weighted_offsets_x = weighted_offsets[:, 0, :, :].contiguous().view(-1, 1, *grid_shape)
    xgrid = weighted_offsets_x / weights
    xgrid = torch.clamp(xgrid * 2 - 1, min=-1, max=1)
    weighted_offsets_y = weighted_offsets[:, 1, :, :].contiguous().view(-1, 1, *grid_shape)
    ygrid = weighted_offsets_y / weights
    ygrid = torch.clamp(ygrid * 2 - 1, min=-1, max=1)
    return torch.cat((xgrid, ygrid), 1)          # (N, 2, 31, 51) in [-1, 1]


@pytest.mark.parametrize("w", [4.0, 13.0, C.FG_WIDTH, 30.0])
def test_ns_nodes_equal_the_official_function(w):
    s = C.node_saliency_2d(demand(1) + C.EPS)
    _, G = C.ns_nodes(s, w)
    ref = official_nonseparable(torch.from_numpy(s).unsqueeze(1), w)
    mine = np.stack([G[..., 0], G[..., 1]], 1) * 2 - 1
    assert np.abs(mine - ref.numpy()).max() < 1e-12


def test_official_interpolation_is_bilinear_between_nodes():
    s = C.node_saliency_2d(demand(2, 1) + C.EPS)
    _, G = C.ns_nodes(s, 8.0)
    grid = torch.from_numpy(np.stack([G[..., 0], G[..., 1]], 1))
    hw = (61, 101)                               # two output samples per node interval, align_corners=True
    fine = F.interpolate(grid, size=hw, mode="bilinear", align_corners=True)[0].numpy()
    v, u = np.meshgrid(np.arange(hw[0]) / (hw[0] - 1), np.arange(hw[1]) / (hw[1] - 1), indexing="ij")
    q = C.ev(G, np.zeros(u.size, dtype=np.int64), u.reshape(-1), v.reshape(-1))
    assert np.abs(q[0].reshape(hw) - fine[0]).max() < 1e-12 and np.abs(q[1].reshape(hw) - fine[1]).max() < 1e-12


def sep_nodes(d, w):
    from sparse_rtdetr.research import r28_fovea_cal as FS
    nodes, _ = FS.nodes_from_demand(torch.from_numpy(d), w)
    return nodes["x"][1].numpy(), nodes["y"][1].numpy()


def test_rank_one_demand_gives_the_separable_map():
    rng = np.random.default_rng(3)
    s = np.outer(rng.random(12) + 0.3, rng.random(21) + 0.3)[None]
    gx, gy = sep_nodes(s - C.EPS, C.FG_WIDTH)
    _, G = C.ns_nodes(C.node_saliency_2d(s), C.FG_WIDTH)
    assert np.abs(G - C.sep_grid(gx, gy)).max() < 1e-12
    assert C.nonseparability(s)[0] < 1e-12 and C.nonseparability(demand(4) + C.EPS).min() > 1e-3


def test_node_saliency_twins():
    from sparse_rtdetr.research import r28_fovea as FV
    m = np.random.default_rng(5).random((4, 21)) + 0.1
    assert np.abs(C.node_saliency_1d(m, 51) - FV.node_saliency(torch.from_numpy(m), 51).numpy()).max() < 1e-15
    assert np.array_equal(C.reflect_index(31), FV.reflect_index(31).numpy())
    s = C.node_saliency_2d(np.ones((1, 12, 21)))
    assert np.abs(s - 1 / (31 * 51)).max() < 1e-15


def boxes(n, seed):
    rng = np.random.default_rng(seed)
    c = rng.random((n, 2)) * 0.9 + 0.05
    wh = rng.random((n, 2)) * 0.06 + 0.004
    b = np.concatenate([c - wh / 2, c + wh / 2], 1)
    return np.clip(b, 0, 1)


def test_uniform_map_is_the_identity():
    _, G = C.ns_nodes(C.node_saliency_2d(np.ones((1, 12, 21))), 13.0)
    v, u = np.meshgrid(np.arange(31) / 30, np.arange(51) / 50, indexing="ij")
    assert np.abs(G[0, ..., 0] - u).max() < 1e-12 and np.abs(G[0, ..., 1] - v).max() < 1e-12 and not C.folded(G)[0]
    b = boxes(50, 6)
    m = C.box_metrics(G, np.zeros(50, dtype=np.int64), b)
    assert np.abs(m["g"] - 1).max() < 1e-9 and m["stretch"].max() < 1e-6 and np.abs(m["expand"] - 1).max() < 1e-9 and m["iou"].min() > 1 - 1e-9


def test_separable_map_closed_form_and_round_trip():
    from sparse_rtdetr.research import r28_fovea as FV
    d = demand(7)
    gx, gy = sep_nodes(d, C.FG_WIDTH)
    G = C.sep_grid(gx, gy)
    assert not C.folded(G).any()
    b = boxes(300, 8)
    im = np.arange(300) % 3
    m = C.box_metrics(G, im, b)
    plan = FV.Plan(torch.from_numpy(gx[im]), torch.from_numpy(gy[im]), None, {})
    ox, oy = FV.magnification(plan, torch.from_numpy(b)[:, None, :])
    ox, oy = ox[:, 0].numpy(), oy[:, 0].numpy()
    assert np.abs(m["g"] - np.sqrt(ox * oy)).max() < 1e-10
    assert m["iou"].min() > 1 - 1e-9 and np.abs(m["expand"] - 1).max() < 1e-9 and m["residual"].max() < 1e-12
    cx, cy = torch.from_numpy((b[:, 0] + b[:, 2]) / 2)[:, None], torch.from_numpy((b[:, 1] + b[:, 3]) / 2)[:, None]
    e = 1e-7
    lx = ((FV.warp(plan.gx, cx + e) - FV.warp(plan.gx, cx - e)) / (2 * e))[:, 0].numpy()
    ly = ((FV.warp(plan.gy, cy + e) - FV.warp(plan.gy, cy - e)) / (2 * e))[:, 0].numpy()
    ok = np.abs(np.abs(np.log(lx / ly)) - m["stretch"]) < 1e-6          # equal away from node lines
    assert ok.mean() > 0.95


@pytest.mark.parametrize("w", [4.0, 10.0, 30.0])
def test_non_separable_inverse_and_metrics(w):
    d = demand(9)
    _, G = C.ns_nodes(C.node_saliency_2d(d + C.EPS), w)
    keep = np.flatnonzero(~C.folded(G))
    if not len(keep):
        pytest.skip("every synthetic map folds at this width")
    b = boxes(400, 10)
    im = keep[np.arange(400) % len(keep)]
    m = C.box_metrics(G, im, b)
    assert m["residual"].max() <= 1e-9
    assert (m["g"] > 0).all() and (m["expand"] >= 1 - 1e-9).all() and (m["iou"] <= 1 + 1e-12).all() and (m["stretch"] >= 0).all()
    # g against a finer polygon and against the local Jacobian for tiny boxes
    tiny = np.concatenate([b[:, :2], b[:, :2] + 1e-4], 1)
    mt = C.box_metrics(G, im, tiny)
    u, v, _, _ = C.invert(G, im, tiny[:, 0] + 5e-5, tiny[:, 1] + 5e-5)
    q = C.ev(G, im, u, v)
    local = 1 / np.sqrt(np.abs(q[2] * q[5] - q[3] * q[4]))
    assert np.median(np.abs(mt["g"] / local - 1)) < 1e-3


def test_fold_detection():
    v, u = np.meshgrid(np.arange(31) / 30, np.arange(51) / 50, indexing="ij")
    G = np.stack([u, v], -1)[None].repeat(3, 0)
    G[1, 10, 20, 0] = G[1, 10, 22, 0]            # a node pushed past its right neighbour
    G[2, 5, 5] = G[2, 5, 6]                      # two nodes collapsed: a zero determinant
    assert C.folded(G).tolist() == [False, True, True]
    assert (C.corner_dets(G[:1]) > 0).all()


def test_exhaustive_search_finds_what_newton_misses():
    d = demand(11, 1)
    _, G = C.ns_nodes(C.node_saliency_2d(d + C.EPS), 6.0)
    if C.folded(G)[0]:
        pytest.skip("folded")
    hit = C._brute(G[0], *[float(x) for x in C.ev(G, np.array([0]), np.array([0.3123]), np.array([0.7411]))[:2]])
    assert hit is not None and abs(hit[0] - 0.3123) < 1e-9 and abs(hit[1] - 0.7411) < 1e-9


def test_cover_mask_is_the_rectangle_rule_for_rectangles():
    rng = np.random.default_rng(12)
    hw = (96, 160)
    b = boxes(12, 13)
    mask = C.cover_mask(C.box_points(b), hw, dilate=8.0)
    ref = np.zeros(hw, dtype=bool)
    for x1, y1, x2, y2 in b * np.array([hw[1], hw[0], hw[1], hw[0]]):
        ref[max(0, int(np.floor(y1 - 8))):int(np.ceil(y2 + 8)), max(0, int(np.floor(x1 - 8))):int(np.ceil(x2 + 8))] = True
    assert np.array_equal(mask, ref)
    assert not C.cover_mask(np.zeros((0, 32, 2)), hw).any() and rng is not None


def test_local_magnification_of_a_separable_map():
    d = demand(14, 1)
    gx, gy = sep_nodes(d, C.FG_WIDTH)
    m = C.local_magnification(C.sep_grid(gx, gy)[0], (60, 100))
    ox = 1.0 / (np.diff(gx[0]) * 50)[np.minimum((np.arange(100) + 0.5) / 100 * 50, 49).astype(int)]
    oy = 1.0 / (np.diff(gy[0]) * 30)[np.minimum((np.arange(60) + 0.5) / 60 * 30, 29).astype(int)]
    assert np.abs(m - np.sqrt(oy[:, None] * ox[None, :])).max() < 1e-9


def test_bins_rows_and_bootstrap():
    area = np.array([10.0, 16.0 ** 2, 32.0 ** 2, 32.0 ** 2 + 1, 64.0 ** 2, 96.0 ** 2 - 1, 96.0 ** 2])
    assert C.size_bin(area).tolist() == [0, 1, 1, 2, 3, 3, 4]
    assert C.g_bin([0.99, 1.0, 1.19, 1.2, 1.4]).tolist() == [0, 1, 1, 2, 3]
    size = np.array([0, 1, 2, 3, 4, 0])
    m = {"g": np.array([1.0, 1.44, 0.9, 1.1, 0.8, np.nan]), "stretch": np.array([0.0, 0.2, 0.1, 0.1, 0.0, np.nan]),
         "expand": np.ones(6), "iou": np.array([1.0, 0.8, 1.0, 1.0, 1.0, np.nan])}
    ok = np.array([True] * 5 + [False])
    r = C.row(size, m, ok)
    assert r["n_small"] == 2 and r["small_g_geomean"] == pytest.approx(1.2) and r["small_share_g_lt_1.2"] == 0.5
    assert r["small_share_iou_ge_0.90"] == 0.5 and r["small_share_iou_ge_0.75"] == 1.0 and r["medium_share_g_lt_1"] == 0.5
    assert r["large96_g_geomean"] == pytest.approx(0.8)
    image = np.array([0, 0, 1, 1, 2, 2])
    rc = C.row_ci(size, m, ok, image, 3, C.boot_weights(3, 200))
    for k in ("small_g_geomean", "small_share_g_lt_1.2", "medium_share_g_lt_1", "small_share_iou_ge_0.90", "small_stretch_median"):
        assert rc[k]["value"] == pytest.approx(r[k]), k
    w = C.boot_weights(5)
    assert w.shape == (1000, 5) and (w.sum(1) == 5).all()


def test_selection_rule():
    base = {"folded_images": 0, "medium_mean_logg": 0.05, "small_share_iou_ge_0.90": 0.99, "small_mean_logg": 0.2}
    rows = {(4.0).hex(): {**base, "folded_images": 3, "small_mean_logg": 0.9},
            (8.0).hex(): {**base, "medium_mean_logg": 0.0399, "small_mean_logg": 0.8},
            (13.0).hex(): {**base, "small_share_iou_ge_0.90": 0.9499, "small_mean_logg": 0.7},
            (16.0).hex(): {**base, "small_mean_logg": 0.30}, (24.0).hex(): {**base, "small_mean_logg": 0.31},
            (30.0).hex(): {**base, "medium_mean_logg": 0.04, "small_mean_logg": 0.1}}
    s = C.select(rows, 0.05)
    assert s["w_star"] == 24.0 and sorted(s["left"]) == sorted([(16.0).hex(), (24.0).hex(), (30.0).hex()])
    assert s["removed"][(4.0).hex()]["first_reason"] == "folded" and "medium" in s["removed"][(8.0).hex()]["first_reason"]
    assert "iou" in s["removed"][(13.0).hex()]["first_reason"]
    none = C.select({(4.0).hex(): rows[(4.0).hex()]}, 0.05)
    assert none["w_star"] is None and none["left"] == []


def test_verdict_rule():
    ok = {"selected": True, "G1_pp": 15.0, "G2_pp": 2.0, "G3_share": 0.95, "folds": {"TC_P": 0, "DEV_P": 0}}
    assert C.verdict(ok)["label"] == "GO"
    assert C.verdict({**ok, "G1_pp": 7.99})["label"] == "STOP" and C.verdict({**ok, "G3_share": 0.849})["label"] == "STOP"
    assert C.verdict({**ok, "selected": False})["label"] == "STOP"
    assert C.verdict({**ok, "G1_pp": 12.0})["label"] == "GREY" and C.verdict({**ok, "G2_pp": 2.5})["label"] == "GREY"
    assert C.verdict({**ok, "G3_share": 0.90})["label"] == "GREY" and C.verdict({**ok, "folds": {"TC_P": 0, "DEV_P": 1}})["label"] == "GREY"
    assert math.isclose(C.FG_WIDTH, 19.03125) and len(C.WIDTHS) == 9
