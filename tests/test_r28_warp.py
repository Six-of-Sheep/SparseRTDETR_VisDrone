"""R28 warp operator checks (CPU, synthetic): TASK_R28_001 section 5 item 1 and AMENDMENT_002 section 2.

The numpy functions below are the reference of evidence/warp_definition_check.py (bisection water-filling, clipped
maps, tap table); the production module must agree with them inside [0, 1] and extend linearly outside.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sparse_rtdetr.research import r28_warp as W  # noqa: E402

HH, HW = W.H_HW
LH, LW = W.L_HW
GH, GW = W.GRID


def ref_waterfill(d, lo=W.LO, hi=W.HI):
    d = np.asarray(d, dtype=np.float64)
    if d.max() - d.min() <= 1e-12 * max(d.max(), 1e-30):
        return np.ones_like(d)
    a, b = 1e-12, 1e12
    for _ in range(200):
        lam = math.sqrt(a * b)
        if np.clip(d / lam, lo, hi).mean() > 1.0:
            a = lam
        else:
            b = lam
    return np.clip(d / math.sqrt(a * b), lo, hi)


def ref_slopes(marg, fine):
    n = len(marg)
    return ref_waterfill(np.interp((np.arange(fine) + 0.5) / fine, (np.arange(n) + 0.5) / n, marg))


def ref_knots(m):
    k = np.concatenate([[0.0], np.cumsum(m) / len(m)])
    k[-1] = 1.0
    return k


def ref_fwd(k, u):
    n = len(k) - 1
    t = np.clip(np.asarray(u, dtype=np.float64), 0.0, 1.0) * n
    i = np.minimum(t.astype(np.int64), n - 1)
    return k[i] + (t - i) * (k[i + 1] - k[i])


def ref_inv(k, v):
    n = len(k) - 1
    v = np.clip(np.asarray(v, dtype=np.float64), 0.0, 1.0)
    i = np.clip(np.searchsorted(k, v, side="right") - 1, 0, n - 1)
    return (i + (v - k[i]) / (k[i + 1] - k[i])) / n


def ref_taps(k, n_out, n_src):
    j = np.arange(n_out)
    p = n_src * ref_inv(k, (j + 0.5) / n_out) - 0.5
    d = n_src * (ref_inv(k, (j + 1.0) / n_out) - ref_inv(k, j / n_out))
    h = np.maximum(1.0, d / 1.5)
    base = np.floor(p - h).astype(np.int64) + 1
    idx = base[:, None] + np.arange(4)[None, :]
    w = np.maximum(0.0, 1.0 - np.abs(idx - p[:, None]) / h[:, None])
    over = int((np.floor(p + h).astype(np.int64) - base + 1).max())
    return np.clip(idx, 0, n_src - 1), w / w.sum(1, keepdims=True), over


def ref_resample(img, kx, ky):
    ix, wx, ox = ref_taps(kx, LW, HW)
    iy, wy, oy = ref_taps(ky, LH, HH)
    t = (img[:, :, torch.from_numpy(ix)] * torch.from_numpy(wx)[None, None]).sum(-1)
    t = (t[:, torch.from_numpy(iy), :] * torch.from_numpy(wy)[None, :, :, None]).sum(2)
    return t, max(ox, oy)


def ref_demand(b):
    sL = np.sqrt((b[:, 2] - b[:, 0]) * LW * (b[:, 3] - b[:, 1]) * LH)
    om = np.clip((W.S_HI - sL) / (W.S_HI - W.S_LO), 0.0, 1.0)
    ci = np.minimum(((b[:, 1] + b[:, 3]) / 2 * GH).astype(np.int64), GH - 1)
    cj = np.minimum(((b[:, 0] + b[:, 2]) / 2 * GW).astype(np.int64), GW - 1)
    D = np.zeros((GH, GW))
    np.add.at(D, (ci, cj), om)
    P = np.pad(D, 1, mode="edge")
    return sum(P[i:i + GH, j:j + GW] for i in range(3) for j in range(3)) / 9.0


def layout(rng):
    boxes = []
    for _ in range(rng.integers(1, 4)):
        cx, cy = rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)
        for _ in range(rng.integers(5, 60)):
            w, h = rng.uniform(8, 30) / LW, rng.uniform(8, 30) / LH
            x, y = cx + rng.normal(0, 0.06), cy + rng.normal(0, 0.06)
            boxes.append([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
    for _ in range(rng.integers(0, 4)):
        w, h = rng.uniform(100, 300) / LW, rng.uniform(100, 300) / LH
        x, y = rng.uniform(0.15, 0.85), rng.uniform(0.15, 0.85)
        boxes.append([x - w / 2, y - h / 2, x + w / 2, y + h / 2])
    b = np.clip(np.array(boxes), 0.0, 1.0)
    return b[(b[:, 2] - b[:, 0] > 4 / LW) & (b[:, 3] - b[:, 1] > 4 / LH)]


@pytest.fixture(scope="module")
def cases():
    rng = np.random.default_rng(0)
    out = []
    for _ in range(40):
        b = layout(rng)
        out.append((b, ref_demand(b)))
    return out


@pytest.fixture(scope="module")
def plan(cases):
    return W.plan_from_demand(torch.stack([W.demand_from_boxes(torch.from_numpy(b)) for b, _ in cases]))


def test_demand_matches_reference(cases):
    for b, d in cases:
        assert float((W.demand_from_boxes(torch.from_numpy(b)) - torch.from_numpy(d)).abs().max()) <= 1e-12
    assert float(W.demand_from_boxes(torch.zeros(0, 4)).abs().max()) == 0.0


def test_allocation_matches_reference_and_bounds(cases, plan):
    for n, (_, d) in enumerate(cases):
        de = d + W.EPS
        mx, my = ref_slopes(de.sum(0), LW), ref_slopes(de.sum(1), LH)
        assert float((plan.mx[n] - torch.from_numpy(mx)).abs().max()) <= 1e-9
        assert float((plan.my[n] - torch.from_numpy(my)).abs().max()) <= 1e-9
        assert float((plan.kx[n] - torch.from_numpy(ref_knots(mx))).abs().max()) <= 1e-10
    for m in (plan.mx, plan.my):
        assert float(m.min()) >= W.LO - 1e-12 and float(m.max()) <= W.HI + 1e-12
        assert float((m.mean(1) - 1).abs().max()) <= 1e-12
    for k in (plan.kx, plan.ky):
        assert float(k[:, 0].abs().max()) == 0.0 and float((k[:, -1] - 1).abs().max()) == 0.0
        assert float((k[:, 1:] - k[:, :-1]).min()) > 0
    assert not bool(plan.identity.any())
    assert float(plan.mx.max()) > 1.2 and float(plan.mx.min()) < 0.9


def test_waterfill_random_marginals():
    g = torch.Generator().manual_seed(1)
    d = torch.rand(64, 301, generator=g, dtype=torch.float64) ** 4 * 50 + 1e-3
    d[0] = 3.0
    d[1, :150] = 1e-6
    m, const = W.waterfill(d)
    assert const.tolist() == [True] + [False] * 63 and bool((m[0] == 1).all())
    for i in range(64):
        assert float((m[i] - torch.from_numpy(ref_waterfill(d[i].numpy()))).abs().max()) <= 1e-9
    assert float((m.mean(1) - 1).abs().max()) <= 1e-12


def test_identity_plan_equals_to_low():
    for value in (0.0, 3.5):
        p = W.plan_from_demand(torch.full((2, GH, GW), value))
        assert bool(p.identity.all()) and bool((p.mx == 1).all()) and bool((p.my == 1).all())
    img = torch.rand(2, 3, HH, HW, generator=torch.Generator().manual_seed(2))
    ref = F.interpolate(img, size=(LH, LW), mode="bilinear", align_corners=False, antialias=False)
    out = W.resample(img, p)
    assert out.dtype == torch.float32 and out.shape == ref.shape
    assert float((out - ref).abs().max()) <= 1e-6
    _, wx, span, d = W.taps(p.kx, LW, HW)
    assert float(span) == 2 and int((wx > 0).sum(-1).max()) == 2 and float((d - 1.5).abs().max()) <= 1e-9


def test_maps_round_trip_and_linear_extension(plan):
    g = torch.Generator().manual_seed(3)
    u = torch.rand(plan.kx.shape[0], 500, generator=g, dtype=torch.float64) * 1.6 - 0.3
    for k, m in ((plan.kx, plan.mx), (plan.ky, plan.my)):
        assert float((W.inv(k, W.fwd(k, u)) - u).abs().max()) <= 1e-12
        assert float((W.fwd(k, W.inv(k, u)) - u).abs().max()) <= 1e-12
        lo, hi = torch.full((k.shape[0], 1), -0.25, dtype=torch.float64), torch.full((k.shape[0], 1), 1.25, dtype=torch.float64)
        assert float((W.fwd(k, lo)[:, 0] - (-0.25) * m[:, 0]).abs().max()) <= 1e-9
        assert float((W.fwd(k, hi)[:, 0] - (1 + 0.25 * m[:, -1])).abs().max()) <= 1e-9
        assert float((W.inv(k, lo)[:, 0] - (-0.25) / m[:, 0]).abs().max()) <= 1e-9
        assert float((W.inv(k, hi)[:, 0] - (1 + 0.25 / m[:, -1])).abs().max()) <= 1e-9
        inside = u.clamp(0, 1)
        for n in range(0, k.shape[0], 7):
            assert float((W.fwd(k, inside)[n] - torch.from_numpy(ref_fwd(k[n].numpy(), inside[n].numpy()))).abs().max()) <= 1e-12
            assert float((W.inv(k, inside)[n] - torch.from_numpy(ref_inv(k[n].numpy(), inside[n].numpy()))).abs().max()) <= 1e-12


def test_boxes_inside_outside_crossing_tiny_and_gradient(plan):
    B = plan.kx.shape[0]
    base = torch.tensor([[0.50, 0.50, 0.20, 0.30],      # inside
                         [0.01, 0.02, 0.10, 0.12],      # crosses the left / top border
                         [0.99, 0.97, 0.08, 0.20],      # crosses the right / bottom border
                         [1.10, -0.08, 0.05, 0.04],     # completely outside
                         [0.37, 0.61, 1e-6, 1e-6],      # tiny
                         [0.50, 0.50, 1.30, 1.40]],     # larger than the image
                        dtype=torch.float32)
    boxes = base[None].repeat(B, 1, 1)
    warped = W.map_boxes(plan, boxes, inverse=False)
    back = W.map_boxes(plan, warped, inverse=True)
    assert warped.dtype == torch.float32 and back.dtype == torch.float32
    assert float((back - boxes).abs().max()) <= 1e-6
    assert bool((warped[..., 2:] > 0).all()) and bool((back[..., 2:] > 0).all())
    # no edge is clipped: an edge outside [0, 1] stays outside after either map
    xy = W.cxcywh_to_xyxy(boxes.double())
    wxy = W.cxcywh_to_xyxy(warped.double())
    assert bool(((xy < 0) == (wxy < 0)).all()) and bool(((xy > 1) == (wxy > 1)).all())
    # coordinate gradient of the inverse map of an edge is 1 / slope of the fine cell that holds the edge
    v = torch.tensor([[0.3137, 0.9051, -0.04, 1.07]], dtype=torch.float64).repeat(B, 1).requires_grad_(True)
    out = W.inv(plan.kx, v)
    out.sum().backward()
    cell = (torch.searchsorted(plan.kx, v.detach().contiguous(), right=True) - 1).clamp(0, LW - 1)
    slope = (plan.kx.gather(1, cell + 1) - plan.kx.gather(1, cell)) * LW
    assert float((v.grad - 1 / slope).abs().max()) <= 1e-9
    b = boxes.clone().requires_grad_(True)
    W.map_boxes(plan, b, inverse=True).sum().backward()
    assert bool(torch.isfinite(b.grad).all()) and float(b.grad.abs().min()) > 0


def test_resampler_matches_reference_and_tap_count(plan):
    img = torch.rand(1, 3, HH, HW, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
    worst = 0
    for n in (0, 9, 23):
        ref, over = ref_resample(img[0], plan.kx[n].numpy(), plan.ky[n].numpy())
        out = W.resample(img.float(), plan.select(slice(n, n + 1)))
        assert float((out[0].double() - ref).abs().max()) <= 2e-6
        worst = max(worst, over)
    for k, n_out, n_src in ((plan.kx, LW, HW), (plan.ky, LH, HH)):
        idx, w, span, d = W.taps(k, n_out, n_src)
        assert float(span) <= W.TAPS and float((w.sum(-1) - 1).abs().max()) <= 1e-6
        assert float(d.min()) >= 1.0 - 1e-9 and float(d.max()) <= 2.25 + 1e-9
        assert int(idx.min()) >= 0 and int(idx.max()) <= n_src - 1
    assert worst <= W.TAPS


def crossing(profile, lo_idx, hi_idx, level, rising):
    for i in range(lo_idx, hi_idx):
        a, b = profile[i], profile[i + 1]
        if (rising and a < level <= b) or ((not rising) and a >= level > b):
            return i + 0.5 + (level - a) / (b - a)
    return float("nan")


def test_image_and_box_geometry_agree(cases, plan):
    """Rectangles drawn in the H image: the half-intensity edges of the warped image against the mapped box edges."""
    rng = np.random.default_rng(5)
    errors = []
    for n in range(6):
        b = cases[n][0]
        for bi in rng.permutation(len(b))[:6]:
            x1, y1, x2, y2 = b[bi]
            px1, px2, py1, py2 = int(round(x1 * HW)), int(round(x2 * HW)), int(round(y1 * HH)), int(round(y2 * HH))
            if px2 - px1 < 12 or py2 - py1 < 12:
                continue
            im = torch.zeros(1, 1, HH, HW)
            im[0, 0, py1:py2, px1:px2] = 1.0
            p = plan.select(slice(n, n + 1))
            w = W.resample(im, p)[0, 0].double().numpy()
            edges = torch.tensor([[px1 / HW, px2 / HW]], dtype=torch.float64), torch.tensor([[py1 / HH, py2 / HH]], dtype=torch.float64)
            ex1, ex2 = (W.fwd(p.kx, edges[0])[0] * LW).tolist()
            ey1, ey2 = (W.fwd(p.ky, edges[1])[0] * LH).tolist()
            c0, c1 = int((ex1 + ex2) / 2), int((ey1 + ey2) / 2)
            row, col = w[c1], w[:, c0]
            errors.append([crossing(row, max(int(ex1) - 6, 0), c0, 0.5, True) - ex1,
                           crossing(row, c0, min(int(ex2) + 6, LW - 1), 0.5, False) - ex2,
                           crossing(col, max(int(ey1) - 6, 0), c1, 0.5, True) - ey1,
                           crossing(col, c1, min(int(ey2) + 6, LH - 1), 0.5, False) - ey2])
    e = np.array(errors)
    assert len(e) >= 12 and not np.isnan(e).any()
    assert float(np.abs(e).max()) <= 0.5


def test_magnification_is_the_edge_ratio(cases, plan):
    b = torch.from_numpy(cases[0][0])[None]
    ox, oy = W.magnification(plan.select(slice(0, 1)), b)
    assert float(ox.min()) >= W.LO - 1e-9 and float(ox.max()) <= W.HI + 1e-9
    assert float(oy.min()) >= W.LO - 1e-9 and float(oy.max()) <= W.HI + 1e-9
    small = (b[0, :, 2] - b[0, :, 0]) * LW * (b[0, :, 3] - b[0, :, 1]) * LH < 1024
    assert float(torch.sqrt(ox * oy)[0][small].log().mean().exp()) > 1.05


def test_demand_net_shape_parameters_and_loss():
    torch.manual_seed(0)
    net = W.DemandNet()
    assert sum(p.numel() for p in net.parameters()) == 337745
    x = W.demand_input(torch.rand(2, 3, HH, HW))
    assert tuple(x.shape) == (2, 3) + W.DEMAND_HW
    y = net.eval()(x)
    assert tuple(y.shape) == (2, GH, GW) and float(y.min()) >= 0
    assert float(W.demand_loss(y, y.double())) == 0.0
    assert float(W.demand_loss(y, y + 1)) > 0
