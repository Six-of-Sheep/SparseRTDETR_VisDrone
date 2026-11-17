"""R29 read operator checks (CPU, synthetic): TASK_R29_001 sections 2 and 3.

Every forward here is a synthetic unit-test forward and belongs to no formal ledger.  Covered: the identities and the
source rules; the geometry and both reads against an independent scalar implementation (constant, linear and
high-frequency fields; interior, cross-node, border and outside points); the identity map and an affine map (P equal
to N); finite differences of the offset and value gradients away from the kinks; identity images of a mixed batch;
the class swap, the identity bypass, the copied layer prefix, the read context and the audit on the synthetic detector.
"""
import copy
import functools
import hashlib
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sparse_rtdetr.baseline import training_v2b_deterministic_sampling as DS  # noqa: E402
from sparse_rtdetr.research import r27_l0_control as L0  # noqa: E402
from sparse_rtdetr.research import r28_fovea as FV  # noqa: E402
from sparse_rtdetr.research import r28_fovea_cal as FS  # noqa: E402
from sparse_rtdetr.research import r28_fovea_geo as FG  # noqa: E402
from sparse_rtdetr.research import r28_resample as R  # noqa: E402
from sparse_rtdetr.research import r29_unzoom as UZ  # noqa: E402
from sparse_rtdetr.research.r27_local_memory import H_HW, L_HW  # noqa: E402
from sparse_rtdetr.protocol_r18.model import Config, build  # noqa: E402

torch.set_default_dtype(torch.float32)
torch.set_num_threads(min(torch.get_num_threads(), 16))
F64, F32 = torch.float64, torch.float32
SHAPES = [(6, 10), (3, 5), (2, 3)]
BIG = [(24, 40), (12, 20), (6, 10)]
POINTS = [4, 4, 4]
HEADS, CH = 2, 3
SEL = {"kind": "fixed_training_geometry_selected_width", "fwhm": 19.03125, "fwhm_hex": "0x1.3080000000000p+4",
       "calibration": {"path": "/x/CALIBRATION.json", "bytes": 10, "sha256": "8dcfcd5cf2cbd7ef96a54b1832d9eabd6923a5b7da860a9e634784ed0fa05c59"},
       "calibration_status": "calibration_unmatched", "h_star": None}
NET = {"kind": "network", "file": {"path": "w"}}


# ---------------------------------------------------------------- definition
def test_identities_and_source_rules():
    assert UZ.ARMS == ("QP", "QC") and UZ.SHORT == {"QP": "P", "QC": "C"} and UZ.READ == {"QP": "direct", "QC": "dense"}
    assert UZ.BASE_ARM == "FG" and (UZ.LAYERS, UZ.LEVELS, UZ.POINTS, UZ.OFFSET_SCALE) == (3, 3, [4, 4, 4], 0.5)
    for arm in UZ.ARMS:
        canonical = json.dumps(UZ.IDENTITY[arm], sort_keys=True, separators=(",", ":")).encode()
        i = UZ.IDENTITY[arm]
        assert UZ.IDENTITY_SHA256[arm] == hashlib.sha256(canonical).hexdigest()
        assert i["arm"] == arm and i["read_mode"] == UZ.READ[arm] and i["task"] == "TASK_R29_001" and i["base_arm"] == "FG"
        assert i["base_identity_sha256"] == FG.IDENTITY_SHA256["FG"] and i["detector_new_learnable_parameters"] == 0
        assert i["read_definition"]["read"] == UZ.DEFINITION["read"][arm]
        assert {"FG", "the_other_arm", "R27_P_or_C"} <= set(i["is_not"])
        same = [k for k in FG.IDENTITY["FG"] if k not in ("version", "arm", "task", "is_not")]
        assert same and all(i[k] == FG.IDENTITY["FG"][k] for k in same)             # the map, demand source and renderer of arm FG
    assert len({UZ.IDENTITY_SHA256["QP"], UZ.IDENTITY_SHA256["QC"], FG.IDENTITY_SHA256["FG"], FS.IDENTITY_SHA256["FS"],
                FV.IDENTITY_SHA256["F"], R.IDENTITY_SHA256["W"], R.IDENTITY_SHA256["K"], L0.IDENTITY_SHA256}) == 8
    d = UZ.DEFINITION
    assert d["new_learnable_parameters"] == 0 and d["sampler"]["geometry"] == {"mode": "bilinear", "padding_mode": "zeros", "align_corners": False}
    assert d["sampler"]["clip"] is None and d["precision"]["geometry"] == "float64" and d["precision"]["bilinear"] == "float32"
    assert "not G(c_w)" in d["source_reference"] and d["gradient"]["map_nodes"] == "none"
    good = {**NET, "fwhm": FG.WIDTH, "selection": SEL}
    for arm in UZ.ARMS:
        assert UZ.check_source(arm, good, False) == FG.check_source("FG", good, False) == UZ.mapping(good)
    for arm, source, check in (("FG", good, False), ("F", good, False), ("W", good, False), ("QP", {**good, "fwhm": 4.0}, False),
                               ("QC", {**good, "fwhm": 13.0}, False), ("QP", {**NET, "fwhm": FG.WIDTH}, False),
                               ("QC", {**good, "kind": "fixed"}, False), ("QP", {**good, "calibration": SEL["calibration"]}, False)):
        with pytest.raises(ValueError):
            UZ.check_source(arm, source, check)
    with pytest.raises(ValueError):
        FG.check_source("QP", good, False)                                          # the arm FG module does not build an R29 arm
    with pytest.raises(ValueError):
        UZ.Reader("bilinear", None, 300)


# ---------------------------------------------------------------- synthetic operator inputs
def attn_stub():
    scale = torch.tensor([1 / n for n in POINTS for _ in range(n)], dtype=F32)
    return types.SimpleNamespace(ms_deformable_attn_core=functools.partial(DS.deformable_attention_core_func_v2, method="default"),
                                 num_points_list=list(POINTS), num_points_scale=scale, offset_scale=0.5)


def nodes(n, seed):
    """n + 1 strictly increasing warped -> source nodes from 0 to 1 with uneven steps (not affine)."""
    g = torch.Generator().manual_seed(seed)
    step = 0.4 + 1.6 * torch.rand(n, generator=g, dtype=F64)
    c = torch.cat([torch.zeros(1, dtype=F64), step.cumsum(0)])
    return c / c[-1]


def uniform(n):
    return torch.arange(n + 1, dtype=F64) / n


def make_plan(batch, seed, identity=()):
    gx = torch.stack([uniform(50) if b in identity else nodes(50, seed + 10 * b) for b in range(batch)])
    gy = torch.stack([uniform(30) if b in identity else nodes(30, seed + 10 * b + 1) for b in range(batch)])
    flag = torch.tensor([b in identity for b in range(batch)])
    return FV.Plan(gx, gy, flag, {})


def field(kind, shapes, batch, seed):
    """(B, tokens, heads, C) FP32 value tensor in the layout of the attention layer."""
    g = torch.Generator().manual_seed(seed)
    parts = []
    for h, w in shapes:
        if kind == "constant":
            v = torch.rand(batch, 1, 1, HEADS, CH, generator=g).expand(batch, h, w, HEADS, CH)
        elif kind == "linear":
            a, bx, by = (torch.rand(batch, 1, 1, HEADS, CH, generator=g) for _ in range(3))
            x = ((torch.arange(w) + 0.5) / w)[None, None, :, None, None]
            y = ((torch.arange(h) + 0.5) / h)[None, :, None, None, None]
            v = a + bx * x + by * y
        else:
            v = torch.randn(batch, h, w, HEADS, CH, generator=g)
        parts.append(v.reshape(batch, h * w, HEADS, CH))
    return torch.cat(parts, 1).contiguous()


def queries(batch, q, seed, spread=2.5):
    g = torch.Generator().manual_seed(seed)
    size = 0.02 + 0.45 * torch.rand(batch, q, 2, generator=g)
    centre = 0.05 + 0.9 * torch.rand(batch, q, 2, generator=g)
    centre[:, 0] = torch.tensor([0.02, 0.03])                                       # one box across the left / top border
    size[:, 0] = torch.tensor([0.30, 0.20])
    centre[:, 1] = torch.tensor([0.97, 0.985])                                      # one across the right / bottom border
    size[:, 1] = torch.tensor([0.20, 0.10])
    ref = torch.cat([centre, size], -1)[:, :, None, :]
    off = (torch.rand(batch, q, HEADS, sum(POINTS), 2, generator=g) * 2 - 1) * spread
    aw = torch.softmax(torch.randn(batch, q, HEADS, sum(POINTS), generator=g), -1)
    return ref, off, aw


# ---------------------------------------------------------------- independent scalar reference (python floats, numpy)
def ref_g(g, v):
    n = len(g) - 1
    t = v * n
    i = min(max(int(math.floor(t)), 0), n - 1)
    return g[i] + (t - i) * (g[i + 1] - g[i])


def ref_f(g, u):
    n = len(g) - 1
    i = 0
    while i < n - 1 and u >= g[i + 1]:
        i += 1
    return (i + (u - g[i]) / (g[i + 1] - g[i])) / n


def ref_bilinear(v, x, y):
    """v (C, h, w); normalised (x, y); bilinear, zeros padding, align_corners=False."""
    c, h, w = v.shape
    fx, fy = x * w - 0.5, y * h - 0.5
    x0, y0 = math.floor(fx), math.floor(fy)
    out = np.zeros(c)
    for ix, iy, wt in ((x0, y0, (x0 + 1 - fx) * (y0 + 1 - fy)), (x0 + 1, y0, (fx - x0) * (y0 + 1 - fy)),
                       (x0, y0 + 1, (x0 + 1 - fx) * (fy - y0)), (x0 + 1, y0 + 1, (fx - x0) * (fy - y0))):
        if 0 <= ix < w and 0 <= iy < h:
            out += wt * v[:, iy, ix]
    return out


def ref_levels(value, shapes, b, head):
    v = value[b, :, head, :].double().numpy()
    out, start = [], 0
    for h, w in shapes:
        out.append(v[start:start + h * w].reshape(h, w, -1).transpose(2, 0, 1))
        start += h * w
    return out


def ref_source_points(plan, ref, off, b, q, head):
    gx, gy = plan.gx[b].tolist(), plan.gy[b].tolist()
    cx, cy, sw, sh = (float(v) for v in ref[b, q, 0].double())
    ax, bx = ref_g(gx, cx - sw / 2), ref_g(gx, cx + sw / 2)
    ay, by = ref_g(gy, cy - sh / 2), ref_g(gy, cy + sh / 2)
    co, so = ((ax + bx) / 2, (ay + by) / 2), (bx - ax, by - ay)
    pts, k = [], 0
    for level, n in enumerate(POINTS):
        for _ in range(n):
            ox, oy = (float(v) for v in off[b, q, head, k].double())
            pts.append((level, co[0] + ox * (1 / n) * so[0] * 0.5, co[1] + oy * (1 / n) * so[1] * 0.5))
            k += 1
    return pts


def ref_read(mode, plan, value, shapes, ref, off, aw):
    batch, q = ref.shape[:2]
    out = np.zeros((batch, q, HEADS * CH))
    points = []
    for b in range(batch):
        gx, gy = plan.gx[b].tolist(), plan.gy[b].tolist()
        for head in range(HEADS):
            levels = ref_levels(value, shapes, b, head)
            if mode == "dense":
                rebuilt = []
                for (h, w), v in zip(shapes, levels):
                    u = np.zeros_like(v)
                    for i in range(h):
                        for j in range(w):
                            u[:, i, j] = ref_bilinear(v, ref_f(gx, (j + 0.5) / w), ref_f(gy, (i + 0.5) / h))
                    rebuilt.append(u)
            for qi in range(q):
                acc = np.zeros(CH)
                for k, (level, px, py) in enumerate(ref_source_points(plan, ref, off, b, qi, head)):
                    points.append((px, py))
                    if mode == "direct":
                        s = ref_bilinear(levels[level], ref_f(gx, px), ref_f(gy, py))
                    else:
                        s = ref_bilinear(rebuilt[level], px, py)
                    acc += float(aw[b, qi, head, k]) * s
                out[b, qi, head * CH:(head + 1) * CH] = acc
    return out, np.array(points)


def run_read(mode, plan, value, shapes, ref, off, aw, audit=None):
    reader = UZ.Reader(mode, plan, ref.shape[1], audit)
    return UZ.read(attn_stub(), reader, 0, F32, value, shapes, ref, off, aw)


# ---------------------------------------------------------------- geometry and reads against the scalar reference
def test_geometry_against_the_scalar_reference():
    plan = make_plan(2, 3)
    ref, off, _ = queries(2, 6, 5)
    stub = attn_stub()
    c_o, s_o = UZ.source_reference(plan, ref)
    p_o = UZ.source_points(stub, c_o, s_o, off)
    p_w = UZ.to_warped(plan, p_o)
    assert c_o.dtype == s_o.dtype == p_o.dtype == p_w.dtype == F64 and p_o.shape == (2, 6, HEADS, 12, 2)
    worst = 0.0
    for b in range(2):
        gx, gy = plan.gx[b].tolist(), plan.gy[b].tolist()
        for q in range(6):
            for head in range(HEADS):
                for k, (_, px, py) in enumerate(ref_source_points(plan, ref, off, b, q, head)):
                    got_o, got_w = p_o[b, q, head, k].tolist(), p_w[b, q, head, k].tolist()
                    worst = max(worst, abs(got_o[0] - px), abs(got_o[1] - py), abs(got_w[0] - ref_f(gx, px)), abs(got_w[1] - ref_f(gy, py)))
    assert worst <= 1e-12, worst
    # the centre rule is "both edges through G, then the centre", which is not G(c_w) on a non-affine map
    g_c = torch.stack([FV.unwarp(plan.gx, ref[:, :, 0, 0]), FV.unwarp(plan.gy, ref[:, :, 0, 1])], -1)
    assert float((c_o - g_c).abs().max()) > 1e-3 and bool((s_o > 0).all())
    # the test set contains points outside the domain on both sides and references that span several map segments
    assert bool((p_o < 0).any()) and bool((p_o > 1).any()) and bool(((p_o > 0) & (p_o < 1)).all(-1).any())
    # the dense grid of the C reconstruction: F at the source pixel centres
    grid = UZ.source_grid(plan, 3, 5)
    assert grid.dtype == F32 and grid.shape == (2, 3, 5, 2)
    for b in range(2):
        for i in range(3):
            for j in range(5):
                want = (2 * ref_f(plan.gx[b].tolist(), (j + 0.5) / 5) - 1, 2 * ref_f(plan.gy[b].tolist(), (i + 0.5) / 3) - 1)
                assert abs(float(grid[b, i, j, 0]) - want[0]) <= 2e-7 and abs(float(grid[b, i, j, 1]) - want[1]) <= 2e-7


@pytest.mark.parametrize("kind", ["constant", "linear", "random"])
@pytest.mark.parametrize("mode", ["direct", "dense"])
def test_read_against_the_scalar_reference(mode, kind):
    plan = make_plan(2, 7)
    value = field(kind, SHAPES, 2, 11)
    ref, off, aw = queries(2, 6, 13)
    out = run_read(mode, plan, value, SHAPES, ref, off, aw)
    want, points = ref_read(mode, plan, value, SHAPES, ref, off, aw)
    assert out.dtype == F32 and out.shape == (2, 6, HEADS * CH)
    err = float(np.abs(out.double().numpy() - want).max())
    assert err <= 2e-5, (mode, kind, err)
    inside = ((points > 0) & (points < 1)).all(1)
    assert inside.any() and (~inside).any()                                         # interior and outside points both occur


def test_direct_and_dense_differ_on_a_non_affine_map_and_agree_on_constant_interior():
    plan = make_plan(2, 7)
    ref, off, aw = queries(2, 6, 13)
    value = field("random", SHAPES, 2, 11)
    a, b = run_read("direct", plan, value, SHAPES, ref, off, aw), run_read("dense", plan, value, SHAPES, ref, off, aw)
    assert float((a - b).abs().max()) > 1e-2
    # a constant field read well inside the domain is the constant under both reads
    const = field("constant", BIG, 2, 3)
    ref2 = torch.tensor([0.5, 0.5, 0.2, 0.2]).expand(2, 4, 1, 4).contiguous()
    off2 = off[:, :4] * 0.2
    aw2 = aw[:, :4]
    # every level holds its own constant: the read is the attention-weighted mean of the three constants
    starts = [0, BIG[0][0] * BIG[0][1], BIG[0][0] * BIG[0][1] + BIG[1][0] * BIG[1][1]]
    level_const = torch.stack([const[:, s0] for s0 in starts], 1)                   # (B, levels, heads, C)
    level_weight = torch.stack([w.sum(-1) for w in aw2.split(POINTS, dim=-1)], -1)  # (B, Q, heads, levels)
    want = torch.einsum("bqhl,blhc->bqhc", level_weight, level_const).reshape(2, 4, HEADS * CH)
    for mode in ("direct", "dense"):
        got = run_read(mode, plan, const, BIG, ref2, off2, aw2)
        assert float((got - want).abs().max()) <= 2e-6, mode


# ---------------------------------------------------------------- identity map, affine map, single-segment boxes
def native_read(value, shapes, ref, off, aw):
    stub = attn_stub()
    loc = UZ.native_locations(stub, F32, ref, off)
    return loc, stub.ms_deformable_attn_core(value, shapes, loc, aw, stub.num_points_list)


def test_identity_map_reads_like_the_native_layer():
    plan = FV.Plan(uniform(50).expand(2, -1).contiguous(), uniform(30).expand(2, -1).contiguous(), torch.zeros(2, dtype=torch.bool), {})
    value = field("random", SHAPES, 2, 11)
    ref, off, aw = queries(2, 6, 13)                                                # includes border and outside points
    loc, want = native_read(value, SHAPES, ref, off, aw)
    stub = attn_stub()
    c_o, s_o = UZ.source_reference(plan, ref)
    p_w = UZ.to_warped(plan, UZ.source_points(stub, c_o, s_o, off))
    assert float((p_w - loc.double()).abs().max()) <= 1e-6
    for mode in ("direct", "dense"):
        got = run_read(mode, plan, value, SHAPES, ref, off, aw)
        assert float((got - want).abs().max()) <= 5e-5, mode


def test_affine_map_direct_read_equals_the_native_read():
    t50, t30 = uniform(50), uniform(30)
    plan = FV.Plan((0.1 + 0.8 * t50).expand(2, -1).contiguous(), (-0.05 + 1.1 * t30).expand(2, -1).contiguous(),
                   torch.zeros(2, dtype=torch.bool), {})
    value = field("random", SHAPES, 2, 11)
    ref, off, aw = queries(2, 6, 13)
    loc, want = native_read(value, SHAPES, ref, off, aw)
    stub = attn_stub()
    c_o, s_o = UZ.source_reference(plan, ref)
    p_w = UZ.to_warped(plan, UZ.source_points(stub, c_o, s_o, off))
    assert float((p_w - loc.double()).abs().max()) <= 1e-6                          # box edges and every point in one affine region
    got = run_read("direct", plan, value, SHAPES, ref, off, aw)
    assert float((got - want).abs().max()) <= 5e-5
    # the source frame itself is not the warped frame here
    assert float((c_o - ref[:, :, 0, :2].double()).abs().max()) > 1e-2


def test_boxes_inside_one_map_segment_keep_the_native_positions():
    plan = make_plan(2, 7)
    g = torch.Generator().manual_seed(2)
    ix, iy = torch.randint(0, 50, (2, 6), generator=g), torch.randint(0, 30, (2, 6), generator=g)
    centre = torch.stack([(ix + 0.5) / 50, (iy + 0.5) / 30], -1)
    ref = torch.cat([centre, torch.full((2, 6, 2), 0.004)], -1)[:, :, None, :]
    _, off, aw = queries(2, 6, 13, spread=1.0)
    stub = attn_stub()
    loc = UZ.native_locations(stub, F32, ref, off)
    c_o, s_o = UZ.source_reference(plan, ref)
    p_w = UZ.to_warped(plan, UZ.source_points(stub, c_o, s_o, off))
    assert float((p_w - loc.double()).abs().max()) <= 1e-7


# ---------------------------------------------------------------- gradients
def _cells(mode, plan, shapes, p_o):
    """Integer cell (and, for the direct read, map segment) of every sampling position: the kinks of the read."""
    level_of = [k for k, n in enumerate(POINTS) for _ in range(n)]
    wh = torch.tensor([[shapes[k][1], shapes[k][0]] for k in level_of], dtype=F64)
    if mode == "dense":
        return torch.floor(p_o * wh - 0.5)
    p_w = UZ.to_warped(plan, p_o)
    seg = torch.stack([torch.searchsorted(plan.gx, p_o[..., 0].reshape(p_o.shape[0], -1).contiguous(), right=True).reshape(p_o.shape[:-1]),
                       torch.searchsorted(plan.gy, p_o[..., 1].reshape(p_o.shape[0], -1).contiguous(), right=True).reshape(p_o.shape[:-1])], -1)
    return torch.cat([torch.floor(p_w * wh - 0.5), seg.to(F64)], -1)


@pytest.mark.parametrize("mode", ["direct", "dense"])
def test_offset_and_value_gradients_against_finite_differences(mode):
    plan = make_plan(2, 7)
    value = field("random", BIG, 2, 21)
    g = torch.Generator().manual_seed(4)
    centre = 0.25 + 0.5 * torch.rand(2, 5, 2, generator=g)
    ref = torch.cat([centre, 0.1 + 0.2 * torch.rand(2, 5, 2, generator=g)], -1)[:, :, None, :]
    off = (torch.rand(2, 5, HEADS, 12, 2, generator=g) * 2 - 1) * 1.5
    aw = torch.softmax(torch.randn(2, 5, HEADS, 12, generator=g), -1)
    weight = torch.randn(HEADS * CH, generator=g)
    stub = attn_stub()

    def scalar(v, o, b, q):
        return (run_read(mode, plan, v, BIG, ref, o, aw)[b, q] * weight).sum()

    def positions(o):
        c_o, s_o = UZ.source_reference(plan, ref)
        return UZ.source_points(stub, c_o, s_o, o)

    v0, o0 = value.clone().requires_grad_(True), off.clone().requires_grad_(True)
    assert not plan.gx.requires_grad and not plan.gy.requires_grad                  # the map nodes take no gradient
    eps, checked, worst = 2e-2, 0, 0.0
    base_cells = _cells(mode, plan, BIG, positions(off))
    for trial in range(60):
        b, q, head, k, axis = (int(torch.randint(0, n, (1,), generator=g)) for n in (2, 5, HEADS, 12, 2))
        lo, hi = off.clone(), off.clone()
        lo[b, q, head, k, axis] -= eps
        hi[b, q, head, k, axis] += eps
        if not (torch.equal(_cells(mode, plan, BIG, positions(lo)), base_cells) and torch.equal(_cells(mode, plan, BIG, positions(hi)), base_cells)):
            continue                                                                # the step crosses a pixel or a map node: skipped
        grad_o, = torch.autograd.grad(scalar(v0, o0, b, q), o0)
        ad = float(grad_o[b, q, head, k, axis])
        with torch.no_grad():
            fd = float(scalar(value, hi, b, q).double() - scalar(value, lo, b, q).double()) / (2 * eps)
        worst = max(worst, abs(fd - ad) / max(abs(ad), 1e-3))
        checked += 1
    assert checked >= 15 and worst <= 2e-2, (mode, checked, worst)
    # the read is linear in the value tensor: one-sided differences of single entries
    grad_v, = torch.autograd.grad(scalar(v0, o0, 0, 0), v0)
    assert bool(torch.isfinite(grad_v).all()) and float(grad_v.abs().max()) > 0
    top = grad_v.abs().flatten().topk(8).indices.tolist()
    with torch.no_grad():
        base = float(scalar(value, off, 0, 0).double())
        for flat in top:
            idx = np.unravel_index(flat, tuple(value.shape))
            moved = value.clone()
            moved[idx] += 0.5
            fd = (float(scalar(moved, off, 0, 0).double()) - base) / 0.5
            assert abs(fd - float(grad_v[idx])) <= 2e-3 * max(abs(float(grad_v[idx])), 1e-3), (mode, idx, fd, float(grad_v[idx]))


def test_offset_gradient_of_the_direct_read_passes_through_the_map():
    """d position / d offset of the direct read is the native factor times the local slope of F, never the native factor alone."""
    plan = make_plan(1, 7)
    ref = torch.tensor([[[[0.5, 0.5, 0.2, 0.2]]]])
    off = torch.zeros(1, 1, HEADS, 12, 2, requires_grad=True)
    stub = attn_stub()
    c_o, s_o = UZ.source_reference(plan, ref)
    p_w = UZ.to_warped(plan, UZ.source_points(stub, c_o, s_o, off))
    grad, = torch.autograd.grad(p_w[..., 0].sum(), off)
    gx = plan.gx[0]
    i = int(torch.searchsorted(gx, c_o[0, 0, 0:1].contiguous(), right=True)) - 1
    slope_f = 1.0 / (50 * float(gx[i + 1] - gx[i]))
    want = 0.25 * 0.5 * float(s_o[0, 0, 0]) * slope_f
    assert abs(float(grad[0, 0, 0, 0, 0]) - want) <= 1e-6 * abs(want) and abs(slope_f - 1.0) > 1e-3     # FP32 offsets
    assert float(grad[..., 1].abs().max()) == 0.0


# ---------------------------------------------------------------- identity images of a mixed batch
@pytest.mark.parametrize("mode", ["direct", "dense"])
def test_identity_image_of_a_mixed_batch_reads_natively(mode):
    plan = make_plan(2, 7, identity=(1,))
    value = field("random", SHAPES, 2, 11)
    ref, off, aw = queries(2, 6, 13)
    _, native = native_read(value, SHAPES, ref, off, aw)
    got = run_read(mode, plan, value, SHAPES, ref, off, aw)
    assert torch.equal(got[1], native[1])                                           # bit for bit: no identity resampling is inserted
    assert float((got[0] - native[0]).abs().max()) > 1e-3
    alone = run_read(mode, plan.select(slice(0, 1)), value[:1], SHAPES, ref[:1], off[:1], aw[:1])
    assert float((got[0] - alone[0]).abs().max()) <= 1e-6                           # the other image is read as it is alone
    sink = []
    again = run_read(mode, plan, value, SHAPES, ref, off, aw, audit=sink)
    assert torch.equal(again, got) and sink[0]["identity_images"] == 1 and sink[0]["normal"]["queries"] == 6


# ---------------------------------------------------------------- synthetic detector
@pytest.fixture(scope="module")
def built():
    model, criterion, _ = build(Config(seed=0), synthetic=True)
    return model.eval(), criterion


def wrapped(built, source, arm):
    source = {**source, "fwhm": FG.WIDTH}
    model = copy.deepcopy(built[0])
    L0.wrap(model)
    UZ.swap(model, UZ._context(source, UZ.check_source(arm, source, True), arm), arm)
    return model


def wrapped_fg(built, source):
    source = {**source, "fwhm": FG.WIDTH}
    model = copy.deepcopy(built[0])
    L0.wrap(model)
    FG.swap(model, FG._context(source, FG.check_source("FG", source, True)))
    return model


def images(seed, batch=2):
    g = torch.Generator().manual_seed(seed)
    low = torch.rand(batch, 3, 36, 63, generator=g)
    return torch.nn.functional.interpolate(low, size=H_HW, mode="bilinear", align_corners=False).clamp(0, 1)


def targets(seed, batch=2):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(batch):
        n = 5 + i
        wh = torch.rand(n, 2, generator=g) * 0.08 + 0.004
        c = wh / 2 + torch.rand(n, 2, generator=g) * (1 - wh)
        out.append({"boxes": torch.cat([c, wh], 1), "labels": torch.randint(0, 10, (n,), generator=g),
                    "gt_ids": [f"t{i}-{k}" for k in range(n)]})
    return out


def tensors(out, prefix=""):
    """Every tensor of a (nested) detector output, with its path."""
    if isinstance(out, torch.Tensor):
        return [(prefix, out)]
    if isinstance(out, dict):
        return [x for k in sorted(out) for x in tensors(out[k], f"{prefix}/{k}")]
    if isinstance(out, (list, tuple)):
        return [x for i, v in enumerate(out) for x in tensors(v, f"{prefix}[{i}]")]
    return []


def installed(model):
    return any("r29_reader" in vars(layer.cross_attn) for layer in model.decoder.decoder.layers)


def test_swap_keeps_the_native_layout_and_the_arms_do_not_mix(built):
    src = {"kind": "synthetic", "seed": 0}
    qp, qc, fg = wrapped(built, src, "QP"), wrapped(built, src, "QC"), wrapped_fg(built, src)
    for model, arm, other in ((qp, "QP", "QC"), (qc, "QC", "QP")):
        assert list(model.state_dict()) == list(built[0].state_dict())
        assert [n for n, _ in model.named_modules()] == [n for n, _ in built[0].named_modules()]
        assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in built[0].parameters())
        assert UZ.is_wrapped(model) and UZ.is_wrapped(model, arm) and not UZ.is_wrapped(model, other)
        assert not FG.is_wrapped(model) and not FS.is_wrapped(model) and not FV.is_wrapped(model) and not R.is_wrapped(model) and not L0.is_wrapped(model)
        assert vars(model.r28_ctx)["r28_fwhm"] == FG.WIDTH and vars(model.r28_ctx)["r29_read"] == UZ.READ[arm]
    assert not UZ.is_wrapped(fg) and len({type(qp), type(qc), type(fg)}) == 3
    attn = [type(layer.cross_attn) for m in (qp, qc) for layer in m.decoder.decoder.layers]
    assert len(set(attn)) == 1 and attn[0].__name__ == "R29ReadAttention" and type(fg.decoder.decoder.layers[0].cross_attn) is attn[0]._r29_native
    for other in (qp, qc, fg):
        with pytest.raises(ValueError):
            UZ.swap(other, qp.r28_ctx, "QP")                                        # a swapped model is not wrapped again
    plain = copy.deepcopy(built[0])
    L0.wrap(plain)
    with pytest.raises(ValueError):
        UZ.swap(plain, qc.r28_ctx, "QP")                                            # the context of the other arm
    with pytest.raises(ValueError):
        UZ.swap(plain, fg.r28_ctx, "QP")                                            # an arm FG context carries no read mode
    with pytest.raises(ValueError):
        UZ.swap(plain, qp.r28_ctx, "FG")
    assert L0.is_wrapped(plain)                                                     # a refused swap changed nothing
    vars(qp.r28_ctx)["r29_read"] = "dense"                                          # a read mode changed after the swap is refused
    assert not UZ.is_wrapped(qp, "QP")
    with pytest.raises(ValueError):
        qp(images(0, 1))
    vars(qp.r28_ctx)["r29_read"] = "direct"
    with pytest.raises(ValueError):
        qp(torch.rand(1, 3, *L_HW))
    layer = qp.decoder.decoder.layers[0].cross_attn
    with pytest.raises(ValueError):                                                 # no read context outside an R29 model forward
        layer(torch.zeros(1, 4, 256), torch.rand(1, 4, 1, 4), torch.zeros(1, 126, 256), [[8, 12], [4, 6], [2, 3]])
    assert not installed(qp)


@pytest.mark.parametrize("arm", ["QP", "QC"])
def test_constant_demand_is_the_l0_forward_bit_exact(built, arm):
    x, t = images(2), targets(2)
    plain = copy.deepcopy(built[0])
    L0.wrap(plain)
    model = wrapped(built, {"kind": "constant"}, arm)
    sink = vars(model.r28_ctx)["r29_audit"] = []
    with torch.no_grad():
        a, b = model(x), plain(x)
    assert set(a) == set(b) == {"pred_logits", "pred_boxes"}
    assert torch.equal(a["pred_logits"], b["pred_logits"]) and torch.equal(a["pred_boxes"], b["pred_boxes"])
    assert sink == [{"identity_batch": True, "images": 2, "layers": []}] and not installed(model)
    model.train(), plain.train()
    torch.manual_seed(11)
    a = model(x, t)
    torch.manual_seed(11)
    b = plain(x, t)
    ta, tb = tensors(a), tensors(b)
    assert [n for n, _ in ta] == [n for n, _ in tb] and len(ta) > 10 and all(torch.equal(u, v) for (_, u), (_, v) in zip(ta, tb))


def test_copied_layer_prefix_is_the_native_layer(built):
    """The statements copied in front of the read: with the native sampling positions the layer is the native layer."""
    model = wrapped(built, {"kind": "synthetic", "seed": 6}, "QP")
    layer = model.decoder.decoder.layers[1].cross_attn
    g = torch.Generator().manual_seed(8)
    shapes = [[8, 12], [4, 6], [2, 3]]
    query, value = torch.randn(2, 7, 256, generator=g), torch.randn(2, 126, 256, generator=g)
    ref = torch.cat([0.2 + 0.6 * torch.rand(2, 7, 2, generator=g), 0.05 + 0.3 * torch.rand(2, 7, 2, generator=g)], -1)[:, :, None, :]
    plan = make_plan(2, 7)
    native = type(layer)._r29_native
    with torch.no_grad():
        want = native.forward(layer, query, ref, value, shapes)
        outs = {}
        for mode in ("native", "direct", "dense"):
            reader = UZ.Reader(mode, plan, 7)
            with UZ.reading(model, reader):
                outs[mode] = layer(query, ref, value, shapes)
            assert reader.calls == 1 and not installed(model)
        bypass = UZ.Reader("direct", None, 7)
        with UZ.reading(model, bypass):
            same = layer(query, ref, value, shapes)
            with pytest.raises(ValueError):
                with UZ.reading(model, UZ.Reader("direct", None, 7)):               # no nested read context
                    pass
        with UZ.reading(model, UZ.Reader("direct", plan, 7)):
            with pytest.raises(ValueError):
                layer(query, ref[..., :2], value, shapes)                           # the two-dimensional reference branch is refused
    assert torch.equal(outs["native"], want) and torch.equal(same, want) and not installed(model)
    assert float((outs["direct"] - want).abs().max()) > 1e-4 and float((outs["dense"] - want).abs().max()) > 1e-4
    assert float((outs["direct"] - outs["dense"]).abs().max()) > 1e-4


@pytest.mark.parametrize("mode", ["direct", "dense"])
def test_sampler_calls_and_dtypes_with_and_without_cpu_autocast(built, mode, monkeypatch):
    model = wrapped(built, {"kind": "synthetic", "seed": 6}, "QP")
    layer = model.decoder.decoder.layers[0].cross_attn
    g = torch.Generator().manual_seed(8)
    shapes = [[8, 12], [4, 6], [2, 3]]
    query, value = torch.randn(2, 7, 256, generator=g), torch.randn(2, 126, 256, generator=g)
    ref = torch.cat([0.2 + 0.6 * torch.rand(2, 7, 2, generator=g), 0.05 + 0.3 * torch.rand(2, 7, 2, generator=g)], -1)[:, :, None, :]
    plan = make_plan(2, 7)
    calls, original = [], DS.bilinear_grid_sample

    def recorder(v, grid):
        out = original(v, grid)
        calls.append((v.dtype, grid.dtype, out.dtype, tuple(grid.shape[1:3])))
        return out

    monkeypatch.setattr(DS, "bilinear_grid_sample", recorder)
    outs = []
    for autocast in (False, True):
        calls.clear()
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast), UZ.reading(model, UZ.Reader(mode, plan, 7)):
            outs.append(layer(query, ref, value, shapes))
        # every bilinear of the read is the frozen sampler: three levels read, plus three levels rebuilt in the dense arm
        assert len(calls) == (3 if mode == "direct" else 6), calls
        assert all(g_ == F32 and o == F32 for _, g_, o, _ in calls), calls
        if mode == "dense":
            assert [c[3] for c in calls[:3]] == [(8, 12), (4, 6), (2, 3)] and all(c[0] == F32 for c in calls[3:])
        assert bool(torch.isfinite(outs[-1].float()).all())
    assert float((outs[0].float() - outs[1].float()).abs().max()) <= 0.05 * float(outs[0].abs().max())


@pytest.mark.parametrize("arm", ["QP", "QC"])
def test_evaluation_forward_reads_without_targets_and_leaves_nothing_installed(built, arm):
    x, t = images(1), targets(1)
    src = {"kind": "synthetic", "seed": 6}
    model, fg = wrapped(built, src, arm), wrapped_fg(built, src)
    marks = []
    vars(model.r28_ctx)["probe"] = marks.append
    with torch.no_grad():
        a, again, with_targets, n = model(x), model(x), model(x, t), fg(x)
    vars(model.r28_ctx).pop("probe")
    assert marks[:5] == ["demand", "map", "render", "detector", "unwarp"] and len(marks) == 15
    assert set(a) == {"pred_logits", "pred_boxes"} and not installed(model)
    for k in a:
        assert torch.equal(a[k], again[k]) and torch.equal(a[k], with_targets[k]) and bool(torch.isfinite(a[k]).all())
    assert a["pred_boxes"].dtype == F32 and bool((a["pred_boxes"][..., 2:] > 0).all())
    assert float((a["pred_logits"] - n["pred_logits"]).abs().max()) > 1e-4          # the read acts: not arm FG's output
    twin = copy.deepcopy(model)                                                     # an EMA copy reads alike with its own context
    with torch.no_grad():
        c = twin(x)
    assert UZ.is_wrapped(twin, arm) and twin.r28_ctx is not model.r28_ctx and all(torch.equal(a[k], c[k]) for k in a)
    # the forward is arm FG's forward with the read context around the detector call
    plan = FG.plan_from_demand(model.r28_ctx.demand(x))
    manual = wrapped(built, src, arm)
    with torch.no_grad():
        with UZ.reading(manual, UZ.Reader(UZ.READ[arm], plan, manual.decoder.num_queries)) as reader:
            raw = type(manual)._r29_native.forward(manual, FV.resample(x, plan)[0], None)
        want = FV.unwarp_outputs(plan, raw)
    assert reader.calls == 3 and all(torch.equal(a[k], want[k]) for k in a)
    assert float((a["pred_boxes"] - raw["pred_boxes"]).abs().max()) > 1e-4          # the output is not left in the warped frame
    # an error inside the detector call removes the read context
    handle = model.backbone.register_forward_hook(lambda *_: (_ for _ in ()).throw(RuntimeError("stop")))
    with pytest.raises(RuntimeError):
        model(x)
    handle.remove()
    assert not installed(model)
    with torch.no_grad():
        assert torch.equal(model(x)["pred_logits"], a["pred_logits"])


def test_the_two_arms_read_differently_on_the_same_plan(built):
    x = images(1)
    src = {"kind": "synthetic", "seed": 6}
    with torch.no_grad():
        a, b = wrapped(built, src, "QP")(x), wrapped(built, src, "QC")(x)
    assert float((a["pred_logits"] - b["pred_logits"]).abs().max()) > 1e-4


@pytest.mark.parametrize("arm", ["QP", "QC"])
def test_training_forward_audit_and_gradients(built, arm):
    x, t = images(4), targets(4)
    original = copy.deepcopy(t)
    src = {"kind": "synthetic", "seed": 6}
    model, fg = wrapped(built, src, arm).train(), wrapped_fg(built, src).train()
    model.r28_ctx.observe = fg.r28_ctx.observe = True
    sink = vars(model.r28_ctx)["r29_audit"] = []
    torch.manual_seed(11)
    out = model(x, t)
    torch.manual_seed(11)
    base = fg(x, t)
    assert not installed(model) and all(torch.equal(a["boxes"], b["boxes"]) for a, b in zip(t, original))
    assert {"pred_logits", "pred_boxes", "aux_outputs", "dn_aux_outputs", "dn_meta"} <= set(out)
    assert [n for n, _ in tensors(out)] == [n for n, _ in tensors(base)]
    assert out["dn_meta"]["dn_num_split"] == base["dn_meta"]["dn_num_split"]       # the denoising groups of arm FG
    # the plan, the warped targets and the observation record are arm FG's: the same record from both models
    (mine,), (theirs,) = model.r28_ctx.records, fg.r28_ctx.records
    assert mine["n"] == theirs["n"] == [5, 6] and set(mine) == set(theirs)
    for k in ("g", "ox", "oy", "demand", "xyxy"):
        a, b = mine[k], theirs[k]
        assert (torch.equal(a, b) if isinstance(a, torch.Tensor) else all(torch.equal(u, v) for u, v in zip(a, b))), k
    assert float((out["pred_logits"] - base["pred_logits"]).abs().max()) > 1e-4
    # audit: one record of this forward, three layers, ordinary and denoising queries apart
    (rec,) = sink
    assert rec["identity_batch"] is False and rec["images"] == 2 and len(rec["layers"]) == 3
    for i, layer in enumerate(rec["layers"]):
        assert layer["layer"] == i and layer["mode"] == UZ.READ[arm] and layer["identity_images"] == 0
        assert layer["dn_queries"] == out["dn_meta"]["dn_num_split"][0] and layer["queries"] == sum(out["dn_meta"]["dn_num_split"])
        for group in ("normal", "dn"):
            g = layer[group]
            assert g["displacement_feature_px"]["max"] > 0 and g["displacement_normalised"]["mean"] > 0
            assert 0 <= g["reference_edges_in_different_segments"]["any"] <= 1 and len(g["levels"]) == 3
            for which in ("N", "P", "C"):
                assert abs(sum(g["border"][which].values()) - 1) <= 1e-9
            d = g["read_difference_P_vs_C"]
            assert d["samples_relative_l2"] > 0 and d["attention_output_relative_l2"] > 0
            assert sum(v["points"] for v in d["by_border_class_of_the_source_point"].values()) == g["queries"] * 8 * 12
    assert out["dn_meta"]["dn_num_split"][1] == model.decoder.num_queries          # dn_num_split holds [dn queries, ordinary queries]
    loss = sum(v.float().sum() for n, v in tensors(out) if n.endswith("pred_boxes") or n.endswith("pred_logits"))
    model.zero_grad()
    loss.backward()                                                                 # no layer is called again: no read context is needed
    assert not installed(model)
    for layer in rec["layers"]:
        assert layer["location_gradient"]["finite"] is True and layer["location_gradient"]["abs_max"] > 0
    for layer in model.decoder.decoder.layers:
        ca = layer.cross_attn
        for p in (ca.sampling_offsets.bias, ca.value_proj.weight, ca.attention_weights.weight, ca.output_proj.weight):
            assert p.grad is not None and bool(torch.isfinite(p.grad).all()) and float(p.grad.abs().max()) > 0
    json.dumps(sink, allow_nan=False)
