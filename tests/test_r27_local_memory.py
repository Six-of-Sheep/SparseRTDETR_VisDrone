"""TASK_R27_001 synthetic checks: geometry, boundaries, routing, ramp read, masks, zero init (CPU, no data)."""
import hashlib
import math
import os
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sparse_rtdetr.research import r27_local_memory as R  # noqa: E402

torch.set_default_dtype(torch.float32)


# ---------------------------------------------------------------- cores
def test_core_boundaries_left_closed_right_open():
    W, H = 2016., 1152.
    xs = torch.tensor([0., 671.999 / W, 672. / W, 1343.9 / W, 1344. / W, 2015.9 / W, 1.0])
    ys = torch.tensor([0., 383.99 / H, 384. / H, 767.9 / H, 768. / H, 1151.9 / H, 1.0])
    cols = R.core_of(xs, torch.zeros_like(xs)).tolist()
    rows = (R.core_of(torch.zeros_like(ys), ys) // 3).tolist()
    assert cols == [0, 0, 1, 1, 2, 2, 2]
    assert rows == [0, 0, 1, 1, 2, 2, 2]
    # row-major ids
    assert R.core_of(torch.tensor([0.9]), torch.tensor([0.5])).item() == 5
    assert R.core_of(torch.tensor([0.1]), torch.tensor([0.9])).item() == 6


def test_extract_patches_equal_direct_zero_padded_crop():
    g = torch.Generator().manual_seed(0)
    x = torch.rand(2, 3, 1152, 2016, generator=g)
    tiles = torch.tensor([[0, 2, 4, 8], [1, 3, 5, 6]])
    p = R.extract_patches(x, tiles).view(2, 4, 3, 448, 736)
    for b in range(2):
        for k in range(4):
            t = int(tiles[b, k])
            y0, x0 = (t // 3) * 384 - 32, (t % 3) * 672 - 32
            ref = torch.zeros(3, 448, 736)
            ys, xs = max(y0, 0), max(x0, 0)
            ye, xe = min(y0 + 448, 1152), min(x0 + 736, 2016)
            ref[:, ys - y0:ye - y0, xs - x0:xe - x0] = x[b, :, ys:ye, xs:xe]
            assert torch.equal(p[b, k], ref), (b, t)
    # corner tile 0: first 32 rows / cols are image-outside zeros; centre tile 4 has no padding
    assert p[0, 0, :, :32].abs().sum() == 0 and p[0, 0, :, :, :32].abs().sum() == 0
    assert torch.equal(p[0, 2], x[0, :, 384 - 32:384 - 32 + 448, 672 - 32:672 - 32 + 736])


# ---------------------------------------------------------------- C routing
def test_route_c_rule_and_refusals():
    s = R.route_string("train", "train/images/a.jpg", 61, 0, 3)
    assert s == "r27-v1|seed0|train|train/images/a.jpg|61|0|3"
    tiles = R.route_c_one("train", "train/images/a.jpg", 61, 0)
    expect = sorted(range(9), key=lambda j: (hashlib.sha256(
        f"r27-v1|seed0|train|train/images/a.jpg|61|0|{j}".encode()).hexdigest(), j))[:4]
    assert tiles == sorted(expect) and len(set(tiles)) == 4
    assert R.route_c_one("train", "train/images/a.jpg", 61, 0) == tiles          # stateless / repeatable
    assert R.route_c_one("eval", "train/images/a.jpg", 0, 0) == R.route_c_one("eval", "train/images/a.jpg", 0, 0)
    for bad in (("train", "a|b.jpg", 61, 0), ("train", "a\nb", 61, 0), ("dev", "a", 0, 0), ("train", "", 0, 0),
                ("train", "a", -1, 0), ("train", "a", 1.0, 0)):
        with pytest.raises(ValueError):
            R.route_c_one(*bad)
    # roughly uniform over many keys (descriptive sanity, not a statistical gate)
    counts = [0] * 9
    for i in range(900):
        for t in R.route_c_one("train", f"k{i}", 61, 0):
            counts[t] += 1
    assert min(counts) > 300 and max(counts) < 500


def test_context_c_batch_grouping_invariant():
    keys = [(f"id{i}", "train", f"train/images/{i}.jpg", 61, 0) for i in range(6)]
    def run(groups):
        ctx = R.Context("C")
        out = []
        for g in groups:
            ctx.route_keys.extend(keys[g[0]:g[1]])
            ctx.begin(torch.zeros(g[1] - g[0], 3, 1, 1), None)
            out.extend(ctx.tiles.tolist())
            ctx.clear()
        return out
    assert run([(0, 6)]) == run([(0, 2), (2, 3), (3, 6)])


# ---------------------------------------------------------------- P routing
def test_route_p_mass_ties_and_dn_free():
    g = torch.Generator().manual_seed(1)
    logits = torch.randn(2, 1000, 10, generator=g)
    _, ind = torch.topk(logits.max(-1).values, 300, dim=-1)
    boxes = torch.rand(2, 300, 4, generator=g) * 0.98 + 0.01
    tiles, mass, p = R.route_p(logits, boxes)
    for b in range(2):
        ref = [0.] * 9
        for i in range(300):
            pi = float(torch.sigmoid(logits[b, ind[b, i]]).max())
            assert abs(pi - float(p[b, i])) < 1e-6
            ref[int(R.core_of(boxes[b, i, 0:1], boxes[b, i, 1:2]))] += pi
        assert max(abs(a - float(m)) for a, m in zip(ref, mass[b])) < 1e-4
        order = sorted(range(9), key=lambda j: (-float(mass[b, j]), j))[:4]
        assert tiles[b].tolist() == sorted(order)
    # ties -> smaller id.  p = sigmoid(0) = 0.5 exactly, so every partial sum is exact.
    lg = torch.zeros(1, 300, 10)
    bx = torch.full((1, 300, 4), 0.1)
    centres = {0: (0.1, 0.1), 1: (0.5, 0.1), 2: (0.9, 0.1), 3: (0.1, 0.5), 4: (0.5, 0.5), 5: (0.9, 0.5), 6: (0.1, 0.9),
               7: (0.5, 0.9), 8: (0.9, 0.9)}
    for i in range(300):
        c = [8, 7, 6, 5, 4, 3, 2, 1, 0][i % 9]
        bx[0, i, 0], bx[0, i, 1] = centres[c]
    t, m, _ = R.route_p(lg, bx)
    # 300 = 33*9 + 3 -> cores 8,7,6 hold 34 boxes, the rest 33 (equal masses) -> smallest id 0 wins the tie
    assert m[0].tolist() == [16.5] * 6 + [17.0] * 3
    assert t[0].tolist() == [0, 6, 7, 8]


# ---------------------------------------------------------------- local read
class FakeCA(nn.Module):
    def __init__(self, g):
        super().__init__()
        self.num_heads, self.num_points_list, self.embed_dim, self.offset_scale, self.method = 8, [4, 4, 4], 256, 0.5, "default"
        self.sampling_offsets = nn.Linear(256, 192)
        self.attention_weights = nn.Linear(256, 96)
        self.value_proj = nn.Linear(256, 256)
        self.output_proj = nn.Linear(256, 256)
        with torch.no_grad():
            for m in (self.sampling_offsets, self.attention_weights, self.value_proj, self.output_proj):
                m.weight.copy_(torch.randn(m.weight.shape, generator=g) * 0.05)
                m.bias.copy_(torch.randn(m.bias.shape, generator=g) * 0.05)


def test_from_native_rows_are_level0_points_and_output_zero():
    g = torch.Generator().manual_seed(2)
    ca = FakeCA(g)
    rng = torch.get_rng_state()
    r = R.LocalRead.from_native(ca)
    assert torch.equal(rng, torch.get_rng_state()), "construction consumed the global RNG"
    q = torch.randn(2, 7, 256, generator=g)
    native_off = ca.sampling_offsets(q).view(2, 7, 8, 12, 2)[:, :, :, :4]
    native_w = ca.attention_weights(q).view(2, 7, 8, 12)[..., :4]
    # row subsets of a GEMM are not bitwise the full GEMM's rows; the copied parameters are checked exactly below
    assert torch.allclose(r.sampling_offsets(q).view(2, 7, 8, 4, 2), native_off, atol=1e-6)
    assert torch.allclose(r.attention_weights(q).view(2, 7, 8, 4), native_w, atol=1e-6)
    off, w = R._level0_rows()
    assert torch.equal(r.sampling_offsets.weight, ca.sampling_offsets.weight[off])
    assert torch.equal(r.attention_weights.bias, ca.attention_weights.bias[w])
    assert torch.equal(r.value_proj.weight, ca.value_proj.weight) and r.value_proj.weight.data_ptr() != ca.value_proj.weight.data_ptr()
    assert r.output_proj.weight.abs().sum() == 0 and r.output_proj.bias.abs().sum() == 0


def _ramp_read(points_px, ref_cxcy, tiles, *, bias=0.):
    """LocalRead with identity value/output projections reading a global-pixel-centre ramp.

    head0 channels hold the global x pixel centre of each local feature cell, head1 the y centre.
    All four points of every head share one offset (the requested pixel); returns (out, ctx)."""
    b = 1
    r = R.LocalRead()
    with torch.no_grad():
        r.value_proj.weight.copy_(torch.eye(256)); r.value_proj.bias.zero_()
        r.output_proj.weight.copy_(torch.eye(256)); r.output_proj.bias.fill_(bias)
        r.attention_weights.weight.zero_(); r.attention_weights.bias.zero_()
        r.sampling_offsets.weight.zero_(); r.sampling_offsets.bias.zero_()
    q = len(ref_cxcy)
    mem = torch.zeros(b * 4, 256, 56, 92)
    for k, t in enumerate(tiles):
        oy, ox = (t // 3) * 384 - 32, (t % 3) * 672 - 32
        xs = ox + 8 * torch.arange(92).float() + 4
        ys = oy + 8 * torch.arange(56).float() + 4
        mem[k, 0:32] = xs[None, None, :].expand(32, 56, 92)
        mem[k, 32:64] = ys[None, :, None].expand(32, 56, 92)
    ctx = R.Context("P")
    ctx.memory = mem.flatten(2).transpose(1, 2)
    ctx.tiles = torch.tensor([tiles])
    ref = torch.zeros(b, q, 1, 4)
    w, h = 0.1, 0.1
    for i, (cx, cy) in enumerate(ref_cxcy):
        ref[0, i, 0] = torch.tensor([cx, cy, w, h])
    # choose the query vector so that offsets give exactly the requested global pixel for every head/point
    # offset = (p_global - c) / (0.25*0.5*wh)  -> put it in the bias via a per-query query vector trick:
    out = []
    for i, (px, py) in enumerate(points_px):
        cx, cy = ref_cxcy[i]
        ox = (px / 2016. - cx) / (0.25 * w * 0.5)
        oy = (py / 1152. - cy) / (0.25 * h * 0.5)
        with torch.no_grad():
            r.sampling_offsets.bias.copy_(torch.tensor([ox, oy]).repeat(32))
        o = r(torch.zeros(1, q, 256), ref, ctx, 0)
        out.append(o[0, i])
    return torch.stack(out), r


def test_ramp_read_pixel_centres_xy_order_halo_and_bounds():
    tiles = [0, 4, 5, 8]
    # interior of core 4, halo of core 4 (inside the patch, outside the core), patch exterior, image exterior of core 8
    cases = [((1000.5, 600.25), (0.5, 0.5), True),
             ((672 - 20., 400.), (0.5, 0.5), True),        # 20 px into core 3's area, still inside core 4's halo
             ((672 - 40., 400.), (0.5, 0.5), False),       # beyond the 32 px halo
             ((1500., 1100.), (0.9, 0.9), True),           # core 8 interior
             ((2020., 1100.), (0.9, 0.9), False),          # right of the image (inside the zero-padded patch)
             ((1500., 1160.), (0.9, 0.9), False)]          # below the image
    pts = [c[0] for c in cases]
    refs = [c[1] for c in cases]
    out, _ = _ramp_read(pts, refs, tiles)
    for (p, _, valid), o in zip(cases, out):
        if valid:
            # bilinear on a linear ramp of cell centres is exact away from the patch edge
            assert abs(float(o[0]) - p[0]) < 2e-3 and abs(float(o[32]) - p[1]) < 2e-3, (p, o[0], o[32])
            assert abs(float(o[31]) - p[0]) < 2e-3 and abs(float(o[63]) - p[1]) < 2e-3
        else:
            assert torch.equal(o, torch.zeros_like(o)), p


def test_masks_exact_zero_after_output_bias_and_no_nan_gradients():
    tiles = [0, 1, 2, 3]
    # query in unselected core 8; query in core 0 whose points all leave the patch
    out, r = _ramp_read([(1500., 1100.), (5000., 100.)], [(0.9, 0.9), (0.1, 0.1)], tiles, bias=0.7)
    assert torch.equal(out, torch.zeros_like(out))
    # gradient path with mixed valid/invalid points stays finite
    g = torch.Generator().manual_seed(3)
    ctx = R.Context("P")
    ctx.memory = torch.randn(4, 5152, 256, generator=g)
    ctx.tiles = torch.tensor([[0, 4, 5, 8]])
    rr = R.LocalRead()
    with torch.no_grad():
        rr.sampling_offsets.weight.copy_(torch.randn(rr.sampling_offsets.weight.shape, generator=g) * 3)
    q = torch.randn(1, 50, 256, generator=g, requires_grad=True)
    ref = torch.rand(1, 50, 1, 4, generator=g) * 0.9 + 0.05
    o = rr(q, ref, ctx, 0)
    o.sum().backward()
    assert torch.isfinite(q.grad).all() and all(torch.isfinite(p.grad).all() for p in rr.parameters())


def test_zero_output_proj_gives_exact_zero_and_only_output_proj_gradient():
    g = torch.Generator().manual_seed(4)
    r = R.LocalRead.from_native(FakeCA(g))
    ctx = R.Context("P")
    mem = torch.randn(4, 5152, 256, generator=g, requires_grad=True)
    ctx.memory = mem
    ctx.tiles = torch.tensor([[0, 4, 5, 8]])
    q = torch.randn(1, 40, 256, generator=g, requires_grad=True)
    ref = torch.rand(1, 40, 1, 4, generator=g) * 0.9 + 0.05
    o = r(q, ref, ctx, 0)
    assert torch.equal(o, torch.zeros_like(o))
    (o * torch.randn(o.shape, generator=g)).sum().backward()
    assert r.output_proj.weight.grad.abs().sum() > 0
    for m in (r.value_proj, r.sampling_offsets, r.attention_weights):
        assert m.weight.grad.abs().sum() == 0 and m.bias.grad.abs().sum() == 0
    assert q.grad.abs().sum() == 0 and mem.grad.abs().sum() == 0


def test_read_deterministic_and_dn_queries_use_same_rule():
    g = torch.Generator().manual_seed(5)
    r = R.LocalRead.from_native(FakeCA(g))
    with torch.no_grad():
        r.output_proj.weight.copy_(torch.randn(256, 256, generator=g) * 0.05)
    ctx = R.Context("P")
    ctx.memory = torch.randn(8, 5152, 256, generator=g)
    ctx.tiles = torch.tensor([[0, 4, 5, 8], [1, 2, 3, 7]])
    q = torch.randn(2, 320, 256, generator=g)
    ref = torch.rand(2, 320, 1, 4, generator=g) * 0.9 + 0.05
    a, b = r(q, ref, ctx, 0), r(q, ref, ctx, 0)
    assert torch.equal(a, b)
    # the first 20 rows (DN) read with the same rule as the same rows placed among the normal queries
    c = r(q[:, 20:], ref[:, 20:], ctx, 0)
    assert torch.allclose(a[:, 20:], c, atol=1e-4)  # different GEMM row count only
