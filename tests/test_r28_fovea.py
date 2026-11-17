"""R28 arm F checks (CPU, synthetic): TASK_R28_FOVEA_MAPPING_CONTROL_004 section 4 items 1-2.

Every forward here is a synthetic unit-test forward and belongs to no formal ledger.  The references are written apart
from the production code: numpy loops for the attraction rule, np.interp for the saliency sampling and for both map
directions, dense per-pixel loops for the tent renderer.
"""
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sparse_rtdetr.research import r27_l0_control as L0  # noqa: E402
from sparse_rtdetr.research import r28_fovea as FV  # noqa: E402
from sparse_rtdetr.research import r28_resample as R  # noqa: E402
from sparse_rtdetr.research import r28_warp as W  # noqa: E402
from sparse_rtdetr.research.r27_local_memory import H_HW, L_HW  # noqa: E402
from sparse_rtdetr.protocol_r18.engine import tensor_digest  # noqa: E402
from sparse_rtdetr.protocol_r18.model import Config, build, fp32_outputs  # noqa: E402

torch.set_default_dtype(torch.float32)
torch.set_num_threads(min(torch.get_num_threads(), 16))
FAMILIES = ("aux_outputs", "enc_aux_outputs", "dn_aux_outputs")
F64 = torch.float64


# ---------------------------------------------------------------- references
def ref_node_saliency(marg, nodes):
    n = len(marg)
    s = np.interp(np.arange(nodes) / (nodes - 1), (np.arange(n) + 0.5) / n, marg)   # constant beyond the outer centres
    return s / s.sum()


def ref_attraction(s, fwhm=4.0, pad=30):
    """The official rule written as plain loops in FP64: reflect padding, Gaussian window, weighted mean of p_j."""
    n = len(s)
    out = np.empty(n)
    for i in range(n):
        num = den = 0.0
        for j in range(i - pad, i + pad + 1):
            src = -j if j < 0 else (2 * (n - 1) - j if j > n - 1 else j)
            k = math.exp(-4.0 * math.log(2.0) * (i - j) ** 2 / fwhm ** 2)
            num += k * s[src] * (j / (n - 1))
            den += k * s[src]
        out[i] = num / den
    return out


def ref_unwarp(g, v):
    """warped -> source, numpy: np.interp inside [0, 1], the end node intervals outside."""
    n = len(g) - 1
    nodes = np.arange(n + 1) / n
    s0, s1 = (g[1] - g[0]) * n, (g[-1] - g[-2]) * n
    return np.where(v < 0, g[0] + v * s0, np.where(v > 1, g[-1] + (v - 1) * s1, np.interp(v, nodes, g)))


def ref_warp(g, u):
    """source -> warped, numpy."""
    n = len(g) - 1
    nodes = np.arange(n + 1) / n
    s0, s1 = (g[1] - g[0]) * n, (g[-1] - g[-2]) * n
    return np.where(u < g[0], (u - g[0]) / s0, np.where(u > g[-1], 1 + (u - g[-1]) / s1, np.interp(u, g, nodes)))


def ref_boxes(plan, boxes, f):
    """cxcywh (B, N, 4) through f edge by edge, FP64 numpy."""
    b = boxes.detach().double().numpy()
    gx, gy = plan.gx.numpy(), plan.gy.numpy()
    out = np.empty_like(b)
    for i in range(b.shape[0]):
        x1, x2 = (f(gx[i], b[i, :, 0] + s * b[i, :, 2] / 2) for s in (-1, 1))
        y1, y2 = (f(gy[i], b[i, :, 1] + s * b[i, :, 3] / 2) for s in (-1, 1))
        out[i] = np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], -1)
    return out


def ref_axis_weights(g, n_out, n_src):
    """Dense (n_out, n_src) FP64 tent weights of one axis, replicate border; returns the widest support too."""
    a = np.zeros((n_out, n_src))
    widest = 0
    for i in range(n_out):
        left, right, mid = (float(ref_unwarp(g, np.array(v))) for v in (i / n_out, (i + 1) / n_out, (i + 0.5) / n_out))
        p, d = n_src * mid - 0.5, n_src * (right - left)
        h = max(1.0, d / 1.5)
        ws = {}
        for k in range(int(math.floor(p - h)) - 1, int(math.ceil(p + h)) + 2):
            w = max(0.0, 1.0 - abs(k - p) / h)
            if w > 0:
                ws[k] = w
        widest = max(widest, len(ws))
        total = sum(ws.values())
        for k, w in ws.items():
            a[i, min(max(k, 0), n_src - 1)] += w / total
    return a, widest


def demand_case(name):
    gh, gw = W.GRID
    d = torch.zeros(gh, gw, dtype=F64)
    if name == "constant":
        return d + 0.7
    if name == "centre":
        d[5:7, 9:12] = 6.0
    elif name == "edge":
        d[0:2, 0:2] = 8.0
        d[10:12, 19:21] = 3.0
    elif name == "bimodal":
        d[2, 3] = 9.0
        d[9, 17] = 5.0
    elif name == "peak":
        d[6, 10] = 60.0
    elif name == "stripes":
        d[:, ::4] = 20.0                                      # strong alternation: node slopes of G beyond 1.5
    elif name == "random":
        d = torch.rand(gh, gw, generator=torch.Generator().manual_seed(7), dtype=F64) * 3
    else:
        raise KeyError(name)
    return W.blur3(d) if name in ("centre", "edge", "bimodal") else d


CASES = ("constant", "centre", "edge", "bimodal", "peak", "stripes", "random")


# ---------------------------------------------------------------- definition
def test_definition_identity_and_fixed_values():
    canonical = json.dumps(FV.IDENTITY["F"], sort_keys=True, separators=(",", ":")).encode()
    assert FV.IDENTITY_SHA256["F"] == hashlib.sha256(canonical).hexdigest()
    assert len({FV.IDENTITY_SHA256["F"], R.IDENTITY_SHA256["W"], R.IDENTITY_SHA256["K"], L0.IDENTITY_SHA256}) == 4
    d = FV.DEFINITION
    assert (FV.GRID_SHAPE, FV.FWHM, FV.PAD) == ((31, 51), 4.0, 30) and d["grid_shape"] == [31, 51]
    assert d["attraction_fwhm"] == 4.0 and d["anti_crop"] is True and d["padding_size"] == 30 and d["padding_mode"] == "reflect"
    assert d["official"]["commit"] == "c8707bd2e4f10678f6636d24c2915a43b44a59ed"
    assert d["official"]["files"]["fovea/models/grid_generator.py"].startswith("4d36cec0")
    assert d["shared_with_W"] == {k: W.DEFINITION[k] for k in ("grid", "demand_input_hw", "eps", "demand_net")}
    assert FV.IDENTITY["F"]["detector_new_learnable_parameters"] == 0 and FV.IDENTITY["F"]["mapping_learnable_parameters"] == 0
    assert FV.IDENTITY["F"]["demand_source"] == R.IDENTITY["W"]["demand_source"]


# ---------------------------------------------------------------- saliency and the attraction rule
def test_node_saliency_is_marginal_first_then_sampling():
    g = torch.Generator().manual_seed(3)
    for n, nodes in ((21, 51), (12, 31)):
        marg = torch.rand(4, n, generator=g, dtype=F64) + 0.2
        mine = FV.node_saliency(marg, nodes).numpy()
        for b in range(4):
            assert np.abs(mine[b] - ref_node_saliency(marg[b].numpy(), nodes)).max() <= 1e-14
        assert np.abs(mine.sum(1) - 1).max() <= 1e-14
    flat = FV.node_saliency(torch.full((1, 21), 2.4, dtype=F64), 51)
    assert float(flat.max() - flat.min()) == 0.0


def test_reflect_index_is_torch_reflect_padding():
    for n in FV.GRID_SHAPE:
        s = torch.rand(2, n, generator=torch.Generator().manual_seed(n), dtype=F64)
        pad = torch.nn.functional.pad(s[:, None], (FV.PAD, FV.PAD), mode="reflect")[:, 0]
        assert torch.equal(s[:, FV.reflect_index(n)], pad)
    k = FV.kernel_matrix(31)
    assert k.shape == (31, 91) and float(k[5, 5 + 30]) == 1.0 and abs(float(k[5, 5 + 30 + 2]) - 0.5) <= 1e-15   # FWHM 4: half at r = 2
    assert float(k[0, 61]) == 0.0 and float(k[0, 60]) > 0.0


@pytest.mark.parametrize("name", CASES)
def test_attraction_nodes_match_the_loop_reference(name):
    d = demand_case(name)[None] + W.EPS
    for marg, nodes in ((d.sum(1), 51), (d.sum(2), 31)):
        s = FV.node_saliency(marg, nodes)
        raw, g = FV.attraction_nodes(s)
        ref = ref_attraction(ref_node_saliency(marg[0].numpy(), nodes))
        assert np.abs(raw[0].numpy() - ref).max() <= 1e-10
        assert np.abs(g[0].numpy() - np.clip(ref, 0, 1)).max() <= 1e-10
        assert abs(float(raw[0, 0])) <= 1e-12 and abs(float(raw[0, -1]) - 1) <= 1e-12        # anti-crop end points
        assert bool((g[0, 1:] > g[0, :-1]).all())
        if name == "constant":
            assert np.abs(raw[0].numpy() - np.arange(nodes) / (nodes - 1)).max() <= 1e-12


def test_plan_reads_only_the_demand_and_follows_w_identity_test():
    const = demand_case("constant")
    board = torch.zeros(W.GRID, dtype=F64)
    board[::2, ::3] = 1.0                                     # not constant, both marginals not constant
    near = const.clone()
    near[3, 4] += 1e-14                                       # inside arm W's relative tolerance
    cases = torch.stack([const, demand_case("centre"), near, board, demand_case("peak")])
    plan = FV.plan_from_demand(cases)
    assert plan.identity.tolist() == W.plan_from_demand(cases).identity.tolist() == [True, False, True, False, False]
    for b in (0, 2):                                          # exact uniform nodes for a constant demand
        assert torch.equal(plan.gx[b], torch.arange(51, dtype=F64) / 50) and torch.equal(plan.gy[b], torch.arange(31, dtype=F64) / 30)
    assert plan.gx.shape == (5, 51) and plan.gy.shape == (5, 31) and plan.gx.dtype == F64
    for axis in ("x", "y"):
        st = plan.stats[axis]
        assert st["node_slope_min"] > 0 and st["endpoint_error_max"] <= 1e-12 and st["clamp_max"] <= 1e-12
        assert st["clamped_interior_nodes"] == 0
        assert st["first_node_max"] <= 1e-12 and st["last_node_min"] >= 1 - 1e-12
    assert plan.stats["x"]["node_slope_min"] < 0.9 and plan.stats["x"]["node_slope_max"] > 1.1
    one = FV.plan_from_demand(cases[1:2])                    # per image: the batch composition does not enter
    assert torch.equal(one.gx[0], plan.gx[1]) and torch.equal(one.gy[0], plan.gy[1])


def test_non_monotone_nodes_are_refused_not_repaired(monkeypatch):
    def broken(s):
        raw, g = real(s)
        g = g.clone()
        g[:, 7] = g[:, 6]                                     # a plateau
        return raw, g
    real = FV.attraction_nodes
    monkeypatch.setattr(FV, "attraction_nodes", broken)
    with pytest.raises(RuntimeError):
        FV.plan_from_demand(demand_case("centre")[None])


# ---------------------------------------------------------------- map directions
@pytest.mark.parametrize("name", ("centre", "edge", "bimodal", "peak", "stripes", "random"))
def test_maps_round_trip_inside_and_outside_and_match_numpy(name):
    plan = FV.plan_from_demand(demand_case(name)[None])
    v = torch.linspace(-0.3, 1.3, 4001, dtype=F64)[None]
    for g in (plan.gx, plan.gy):
        src = FV.unwarp(g, v)
        back = FV.warp(g, src)
        assert float((back - v).abs().max()) <= 1e-9
        fwd = FV.warp(g, v)
        assert float((FV.unwarp(g, fwd) - v).abs().max()) <= 1e-9
        gn = g[0].numpy()
        assert np.abs(src[0].numpy() - ref_unwarp(gn, v[0].numpy())).max() <= 1e-12
        assert np.abs(fwd[0].numpy() - ref_warp(gn, v[0].numpy())).max() <= 1e-9
        assert bool((src[0, 1:] > src[0, :-1]).all()) and bool((fwd[0, 1:] > fwd[0, :-1]).all())
        n = g.shape[1] - 1
        s0, s1 = float(g[0, 1] - g[0, 0]) * n, float(g[0, -1] - g[0, -2]) * n
        lo, hi = FV.unwarp(g, torch.tensor([[-0.2, 1.25]], dtype=F64))[0]
        assert abs(float(lo) - (float(g[0, 0]) - 0.2 * s0)) <= 1e-12 and abs(float(hi) - (float(g[0, -1]) + 0.25 * s1)) <= 1e-12


def test_unwarp_gradient_is_the_node_interval_slope():
    plan = FV.plan_from_demand(demand_case("bimodal")[None])
    v = torch.tensor([[-0.05, 0.013, 0.37, 0.731, 0.999, 1.08]], dtype=F64, requires_grad=True)
    out = FV.unwarp(plan.gx, v)
    out.sum().backward()
    n = plan.gx.shape[1] - 1
    i = np.clip(np.floor(v.detach().numpy()[0] * n), 0, n - 1).astype(int)
    g = plan.gx[0].numpy()
    assert np.abs(v.grad[0].numpy() - (g[i + 1] - g[i]) * n).max() <= 1e-12 and bool((v.grad > 0).all())


def test_map_boxes_uses_one_map_for_both_directions_without_clipping():
    plan = FV.plan_from_demand(torch.stack([demand_case("edge"), demand_case("peak")]))
    boxes = torch.tensor([[0.02, 0.5, 0.10, 0.2], [0.97, 0.5, 0.12, 0.2], [0.5, 0.01, 0.2, 0.06], [0.5, 0.99, 0.2, 0.08],
                          [0.5, 0.5, 1.2, 1.3], [0.31, 0.62, 1e-4, 2e-4]], dtype=torch.float32).repeat(2, 1, 1)
    src = FV.map_boxes(plan, boxes, inverse=True)
    assert src.dtype == torch.float32 and bool((src[..., 2:] > 0).all())
    assert np.abs(src.double().numpy() - ref_boxes(plan, boxes, ref_unwarp)).max() <= 1e-6
    edges = W.cxcywh_to_xyxy(src.double())
    assert float(edges[0, 0, 0]) < 0 and float(edges[0, 1, 2]) > 1 and float(edges[0, 2, 1]) < 0 and float(edges[0, 3, 3]) > 1
    wide = FV.map_boxes(plan, boxes.double(), inverse=True)
    back = FV.map_boxes(plan, wide, inverse=False)
    assert float((W.cxcywh_to_xyxy(back) - W.cxcywh_to_xyxy(boxes.double())).abs().max()) <= 1e-9
    assert np.abs(FV.map_boxes(plan, boxes.double(), inverse=False).numpy() - ref_boxes(plan, boxes, ref_warp)).max() <= 1e-9


# ---------------------------------------------------------------- renderer
def small_image(seed, hw=(72, 126), batch=1):
    return torch.rand(batch, 3, *hw, generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("name", ("centre", "peak", "edge", "stripes"))
def test_full_support_tent_matches_the_dense_reference(name):
    plan = FV.plan_from_demand(demand_case(name)[None])
    x = small_image(1)
    out, info = FV.resample(x, plan, out_hw=(48, 84))
    ax, wide_x = ref_axis_weights(plan.gx[0].numpy(), 84, 126)
    ay, wide_y = ref_axis_weights(plan.gy[0].numpy(), 48, 72)
    ref = np.einsum("ih,chw,jw->cij", ay, x[0].double().numpy(), ax)
    assert out.shape == (1, 3, 48, 84) and out.dtype == torch.float32
    assert np.abs(out[0].double().numpy() - ref).max() <= 1e-5
    assert np.abs(ax.sum(1) - 1).max() <= 1e-12 and np.abs(ay.sum(1) - 1).max() <= 1e-12
    assert info["taps_x"] >= wide_x and info["taps_y"] >= wide_y
    if name == "stripes":                                     # output pixels that need more than arm W's 4 taps
        assert wide_x > W.TAPS and info["taps_x"] > W.TAPS and info["footprint_max_px"] > 2.25
        ix, wx, _, _ = FV.tent_taps(*FV.pixel_positions(plan.gx, 84), 126)
        iy, wy, _, _ = FV.tent_taps(*FV.pixel_positions(plan.gy, 48), 72)
        cut = wx[..., :W.TAPS] / wx[..., :W.TAPS].sum(-1, keepdim=True)
        short = FV.apply_taps(x, ix[..., :W.TAPS], cut, iy, wy)
        assert float((short - out).abs().max()) > 1e-3      # truncating to 4 taps is a different image


def test_tap_count_does_not_change_an_image():
    """A wider batch support only adds zero-weight taps: an image renders to the same bits alone and in a batch."""
    d = torch.stack([demand_case("centre"), demand_case("stripes")])
    x = small_image(2, batch=2)
    both, info = FV.resample(x, FV.plan_from_demand(d), out_hw=(48, 84))
    alone, info1 = FV.resample(x[:1], FV.plan_from_demand(d[:1]), out_hw=(48, 84))
    assert info["taps_x"] > info1["taps_x"] and torch.equal(both[:1], alone)


def test_w_plan_through_the_full_support_code_is_w_resample_bit_exact():
    plan = W.plan_from_demand(torch.stack([R.synthetic_demand(0), R.synthetic_demand(4)]))
    x = torch.nn.functional.interpolate(small_image(5, (36, 63), 2), size=H_HW, mode="bilinear", align_corners=False)
    old = W.resample(x, plan)

    def positions(k, n_out):
        j = torch.arange(n_out + 1, dtype=F64)
        return W.inv(k, (j / n_out).expand(2, -1)), W.inv(k, ((j[:-1] + 0.5) / n_out).expand(2, -1))

    ix, wx, sx, _ = FV.tent_taps(*positions(plan.kx, L_HW[1]), H_HW[1])
    iy, wy, sy, _ = FV.tent_taps(*positions(plan.ky, L_HW[0]), H_HW[0])
    assert max(sx, sy) <= W.TAPS and torch.equal(FV.apply_taps(x, ix, wx, iy, wy), old)


def test_uniform_nodes_render_like_the_uniform_tent():
    """Exactly uniform nodes give footprint 1.5 and a 2-tap tent at every pixel (the mathematical identity map)."""
    n = torch.arange(51, dtype=F64)[None] / 50
    m = torch.arange(31, dtype=F64)[None] / 30
    plan = FV.Plan(n, m, torch.tensor([True]), {})
    x = small_image(6)
    out, info = FV.resample(x, plan, out_hw=(48, 84))
    assert abs(info["footprint_max_px"] - 1.5) <= 1e-9 and abs(info["footprint_min_px"] - 1.5) <= 1e-9
    ax, _ = ref_axis_weights(n[0].numpy(), 84, 126)
    ay, _ = ref_axis_weights(m[0].numpy(), 48, 72)
    assert np.abs(out[0].double().numpy() - np.einsum("ih,chw,jw->cij", ay, x[0].double().numpy(), ax)).max() <= 1e-5


# ---------------------------------------------------------------- wrapper
@pytest.fixture(scope="module")
def built():
    model, criterion, _ = build(Config(seed=0), synthetic=True)
    return model.eval(), criterion


def wrapped(built, source):
    model = copy.deepcopy(built[0])
    L0.wrap(model)
    FV.swap(model, R.Context(source))
    return model


def images(seed, batch=2):
    g = torch.Generator().manual_seed(seed)
    low = torch.rand(batch, 3, 36, 63, generator=g)
    return torch.nn.functional.interpolate(low, size=H_HW, mode="bilinear", align_corners=False).clamp(0, 1)


def targets(seed, batch=2, empty=()):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(batch):
        n = 0 if i in empty else 5 + i
        wh = torch.rand(n, 2, generator=g) * 0.08 + 0.004
        c = wh / 2 + torch.rand(n, 2, generator=g) * (1 - wh)
        out.append({"boxes": torch.cat([c, wh], 1), "labels": torch.randint(0, 10, (n,), generator=g),
                    "gt_ids": [f"t{i}-{k}" for k in range(n)]})
    return out


def all_boxes(out):
    return [("final", out["pred_boxes"])] + [(f"{k}[{i}]", d["pred_boxes"]) for k in FAMILIES for i, d in enumerate(out.get(k, ()))]


def test_swap_keeps_the_native_layout_and_source_rules(built):
    plain = copy.deepcopy(built[0])
    L0.wrap(plain)
    model = wrapped(built, {"kind": "synthetic", "seed": 0})
    assert list(model.state_dict()) == list(plain.state_dict()) == list(built[0].state_dict())
    assert [n for n, _ in model.named_modules()] == [n for n, _ in built[0].named_modules()]
    assert tensor_digest(model.state_dict().items()) == tensor_digest(built[0].state_dict().items())
    assert FV.is_wrapped(model) and not R.is_wrapped(model) and not L0.is_wrapped(model) and not FV.is_wrapped(plain)
    w_model = copy.deepcopy(built[0])
    L0.wrap(w_model)
    R.swap(w_model, R.Context({"kind": "constant"}))
    assert R.is_wrapped(w_model) and not FV.is_wrapped(w_model)
    with pytest.raises(ValueError):
        FV.swap(w_model, R.Context({"kind": "constant"}))                    # an arm W model is not re-wrapped as F
    with pytest.raises(ValueError):
        FV.swap(model, R.Context({"kind": "constant"}))                      # already swapped
    FV.check_source("F", {"kind": "network"}, False)
    FV.check_source("F", {"kind": "synthetic"}, True)
    for arm, kind, check in (("F", "fixed", False), ("F", "synthetic", False), ("W", "network", False), ("K", "fixed", False),
                             ("F", "fixed", True)):
        with pytest.raises(ValueError):
            FV.check_source(arm, {"kind": kind}, check)
    with pytest.raises(ValueError):
        model(torch.rand(1, 3, *L_HW))
    with pytest.raises(ValueError):
        model(torch.rand(1, 3, *H_HW, dtype=torch.float64))


def test_constant_demand_is_the_l0_forward_bit_exact(built):
    x = images(1)
    plain = copy.deepcopy(built[0])
    L0.wrap(plain)
    model = wrapped(built, {"kind": "constant"})
    with torch.no_grad():
        a, b = model(x), plain(x)
    assert set(a) == set(b) == {"pred_logits", "pred_boxes"}
    assert torch.equal(a["pred_logits"], b["pred_logits"]) and torch.equal(a["pred_boxes"], b["pred_boxes"])


def test_non_identity_eval_maps_boxes_back_without_clipping(built):
    x = images(2)
    model = wrapped(built, {"kind": "synthetic", "seed": 3})
    native = type(model)._r28f_native
    marks = []
    model.r28_ctx.probe = marks.append
    with torch.no_grad():
        out = model(x)
        plan = FV.plan_from_demand(model.r28_ctx.demand(x))
        raw = native.forward(model, FV.resample(x, plan)[0])
    assert marks == ["demand", "map", "render", "detector", "unwarp"]
    assert not bool(plan.identity.any()) and torch.equal(out["pred_logits"], raw["pred_logits"])
    assert np.abs(out["pred_boxes"].double().numpy() - ref_boxes(plan, raw["pred_boxes"], ref_unwarp)).max() <= 1e-6
    assert float((out["pred_boxes"] - raw["pred_boxes"]).abs().max()) > 1e-3
    assert bool((out["pred_boxes"][..., 2:] > 0).all()) and out["pred_boxes"].dtype == torch.float32
    with pytest.raises(ValueError):
        FV.unwarp_outputs(plan, {"pred_logits": out["pred_logits"], "pred_boxes": out["pred_boxes"], "other_boxes": out["pred_boxes"]})


@pytest.fixture(scope="module")
def train_case(built):
    x, t = images(4), targets(4)
    original = copy.deepcopy(t)
    model = wrapped(built, {"kind": "synthetic", "seed": 6}).train()
    manual = wrapped(built, {"kind": "synthetic", "seed": 6}).train()
    native = type(model)._r28f_native
    model.r28_ctx.observe = True
    torch.manual_seed(11)
    out = model(x, t)
    plan = FV.plan_from_demand(manual.r28_ctx.demand(x))
    tw = FV.warp_targets(plan, t)
    torch.manual_seed(11)
    raw = native.forward(manual, FV.resample(x, plan)[0], tw)
    return {"x": x, "t": t, "original": original, "out": out, "raw": raw, "plan": plan, "tw": tw, "model": model, "manual": manual}


def test_training_forward_maps_every_box_family(train_case):
    out, raw, plan = train_case["out"], train_case["raw"], train_case["plan"]
    assert set(out) == set(raw) == {"pred_logits", "pred_boxes", "aux_outputs", "enc_aux_outputs", "enc_meta", "dn_aux_outputs", "dn_meta"}
    mine, theirs = all_boxes(out), all_boxes(raw)
    assert [n for n, _ in mine] == [n for n, _ in theirs] and len(mine) == 1 + 2 + len(raw["enc_aux_outputs"]) + 3
    for (name, a), (_, b) in zip(mine, theirs):
        assert np.abs(a.detach().double().numpy() - ref_boxes(plan, b, ref_unwarp)).max() <= 1e-6, name
        assert float((a - b).abs().max()) > 1e-3, name                      # no family is left in the warped frame
        assert a.dtype == torch.float32 and bool((a[..., 2:] > 0).all()), name
    for k in FAMILIES:
        for d, e in zip(out[k], raw[k]):
            assert torch.equal(d["pred_logits"], e["pred_logits"])
    assert torch.equal(out["pred_logits"], raw["pred_logits"])
    assert out["dn_meta"]["dn_num_split"] == raw["dn_meta"]["dn_num_split"] and out["enc_meta"] == raw["enc_meta"]


def test_targets_are_not_modified_and_denoising_reads_the_inverse_of_g(train_case):
    t, original, tw, plan = train_case["t"], train_case["original"], train_case["tw"], train_case["plan"]
    for i, (a, b, w) in enumerate(zip(t, original, tw)):
        assert torch.equal(a["boxes"], b["boxes"]) and torch.equal(a["labels"], b["labels"]) and a["gt_ids"] == b["gt_ids"]
        assert w["labels"] is a["labels"] and w["boxes"] is not a["boxes"] and float((w["boxes"] - a["boxes"]).abs().max()) > 1e-4
        assert w["boxes"].dtype == a["boxes"].dtype and bool((w["boxes"][:, 2:] > 0).all())
        ref = ref_boxes(plan.select(slice(i, i + 1)), a["boxes"][None], ref_warp)[0]
        assert np.abs(w["boxes"].double().numpy() - ref).max() <= 1e-6
        back = FV.map_boxes(plan.select(slice(i, i + 1)), w["boxes"][None].double(), inverse=True)[0]
        assert float((back - a["boxes"].double()).abs().max()) <= 1e-6      # FP32 targets: the map pair is one map
    rec = train_case["model"].r28_ctx.records
    assert len(rec) == 1 and rec[0]["n"] == [5, 6] and rec[0]["identity_images"] == 0 and rec[0]["slope_min"] < 1 < rec[0]["slope_max"]
    f = rec[0]["fovea"]
    assert f["taps_x"] >= 2 and f["taps_y"] >= 2 and f["endpoint_error_max"] <= 1e-12 and f["clamp_max"] <= 1e-12
    assert abs(rec[0]["slope_min"] * f["node_slope_max"] - 1) <= 1e-12
    ox, oy = FV.magnification(plan.select(slice(0, 1)), W.cxcywh_to_xyxy(t[0]["boxes"].double())[None])
    assert float((rec[0]["g"][0] - torch.sqrt(ox * oy)[0]).abs().max()) <= 1e-12


def test_plan_and_image_do_not_depend_on_the_targets(built, train_case):
    x, t = train_case["x"], train_case["t"]
    model = wrapped(built, {"kind": "synthetic", "seed": 6}).train()
    seen = []
    real = FV.resample

    def spy(images_, plan, out_hw=L_HW):
        res = real(images_, plan, out_hw)
        seen.append((plan.gx.clone(), plan.gy.clone(), res[0].clone()))
        return res

    FV.resample = spy
    try:
        other = targets(99)
        for tt in (t, other, targets(4, empty=(0,)), targets(4, empty=(0, 1))):
            torch.manual_seed(11)
            out = model(x, tt)
            assert set(out) >= {"pred_logits", "pred_boxes", "aux_outputs", "enc_aux_outputs"}
            assert all(bool(torch.isfinite(b).all()) for _, b in all_boxes(out))
        with torch.no_grad():
            model.eval()(x)                                   # no targets at all (the evaluation call)
    finally:
        FV.resample = real
    assert len(seen) == 5
    for gx, gy, img in seen[1:]:
        assert torch.equal(gx, seen[0][0]) and torch.equal(gy, seen[0][1]) and torch.equal(img, seen[0][2])
    assert torch.equal(seen[0][0], train_case["plan"].gx)


def test_empty_targets_keep_the_native_keys(built, train_case):
    x = train_case["x"]
    model = wrapped(built, {"kind": "synthetic", "seed": 6}).train()
    model.r28_ctx.observe = True
    t = targets(4, empty=(0, 1))
    plan = FV.plan_from_demand(model.r28_ctx.demand(x))
    tw = FV.warp_targets(plan, t)
    assert all(len(w["boxes"]) == 0 and w["boxes"] is a["boxes"] for w, a in zip(tw, t))
    torch.manual_seed(3)
    out = model(x, t)
    assert "pred_boxes" in out and bool((out["pred_boxes"][..., 2:] > 0).all())
    rec = model.r28_ctx.records[-1]
    assert rec["n"] == [0, 0] and all(len(v) == 0 for v in rec["g"])


def test_criterion_sees_source_frame_boxes_and_a_leak_changes_the_loss(built, train_case):
    criterion = copy.deepcopy(built[1]).train()
    out, raw, plan, t = train_case["out"], train_case["raw"], train_case["plan"], train_case["t"]

    def total(o):
        return {k: float(v) for k, v in criterion(fp32_outputs(o), t).items()}

    mine = total(out)
    ref = copy.copy(raw)
    ref["pred_boxes"] = torch.from_numpy(ref_boxes(plan, raw["pred_boxes"], ref_unwarp)).float()
    for k in FAMILIES:
        ref[k] = [{"pred_logits": d["pred_logits"], "pred_boxes": torch.from_numpy(ref_boxes(plan, d["pred_boxes"], ref_unwarp)).float()}
                  for d in raw[k]]
    theirs = total(ref)
    assert set(mine) == set(theirs) and len(mine) == 21
    assert max(abs(mine[k] - theirs[k]) for k in mine) <= 1e-4 * max(1.0, max(abs(v) for v in theirs.values()))
    base = sum(theirs.values())
    for name in ("final",) + FAMILIES:
        leak = copy.copy(ref)
        if name == "final":
            leak["pred_boxes"] = raw["pred_boxes"].detach().float()
        else:
            leak[name] = [{"pred_logits": d["pred_logits"], "pred_boxes": d["pred_boxes"].detach().float()} for d in raw[name]]
        assert abs(sum(total(leak).values()) - base) > 1e-3, name


def test_gradients_reach_the_detector_through_g(built, train_case):
    criterion = copy.deepcopy(built[1]).train()
    model, out, t = train_case["model"], train_case["out"], train_case["t"]
    loss = sum(criterion(fp32_outputs(out), t).values())
    assert bool(torch.isfinite(loss))
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(bool(torch.isfinite(g).all()) for g in grads)
    head = model.decoder.dec_bbox_head[-1].layers[-1].weight.grad
    assert head is not None and float(head.abs().max()) > 0
