"""R28 arm FS checks (CPU, synthetic): TASK_R28_CALIBRATED_F_CONTROL_006 section 2.

Every forward here is a synthetic unit-test forward and belongs to no formal ledger.  The references are written apart
from the production code: plain FP64 loops for the attraction rule at a given width, closed-form responses for the
selection rule.
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
from sparse_rtdetr.research import r28_fovea_cal as FS  # noqa: E402
from sparse_rtdetr.research import r28_resample as R  # noqa: E402
from sparse_rtdetr.research import r28_warp as W  # noqa: E402
from sparse_rtdetr.research.r27_local_memory import H_HW, L_HW  # noqa: E402
from sparse_rtdetr.protocol_r18.model import Config, build  # noqa: E402

torch.set_default_dtype(torch.float32)
torch.set_num_threads(min(torch.get_num_threads(), 16))
F64 = torch.float64
WIDTHS = (4.0, 13.0, 21.0625, 30.0)


# ---------------------------------------------------------------- references
def ref_attraction(s, fwhm, pad=30):
    """The official rule as plain FP64 loops: reflect padding, Gaussian window of the given FWHM, weighted mean."""
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
    elif name == "random":
        d = torch.rand(gh, gw, generator=torch.Generator().manual_seed(7), dtype=F64) * 3
    else:
        raise KeyError(name)
    return W.blur3(d) if name in ("centre", "edge", "bimodal") else d


CASES = ("centre", "edge", "bimodal", "peak", "random")
SMOOTH = ("centre", "edge", "bimodal", "random")          # cases used where the test itself needs a valid wide map


# ---------------------------------------------------------------- definition
def test_identity_definition_and_registered_rule():
    canonical = json.dumps(FS.IDENTITY["FS"], sort_keys=True, separators=(",", ":")).encode()
    assert FS.IDENTITY_SHA256["FS"] == hashlib.sha256(canonical).hexdigest()
    assert len({FS.IDENTITY_SHA256["FS"], FV.IDENTITY_SHA256["F"], R.IDENTITY_SHA256["W"], R.IDENTITY_SHA256["K"], L0.IDENTITY_SHA256}) == 5
    d = FS.DEFINITION
    assert (FS.ARM, FS.GRID_SHAPE, FS.PAD) == ("FS", FV.GRID_SHAPE, FV.PAD) and d["only_difference_from_arm_F"] == "attraction_fwhm 4 -> h*"
    assert d["base"]["identity_sha256"] == FV.IDENTITY_SHA256["F"] and d["attraction_fwhm"] != FV.DEFINITION["attraction_fwhm"]
    same = [k for k in FV.DEFINITION if k not in ("name", "attraction_fwhm", "kernel", "differences_from_official")]
    assert same and all(d[k] == FV.DEFINITION[k] for k in same)             # grid, padding, eps, official source, render ...
    assert d["grid_shape"] == [31, 51] and d["padding_size"] == 30 and d["padding_mode"] == "reflect" and d["anti_crop"] is True
    c = FS.CALIBRATION
    assert (FS.H_MIN, FS.H_MAX, FS.GRID_POINTS, FS.EXTRA_WIDTHS, FS.TOLERANCE, FS.BISECTIONS) == (4.0, 30.0, 65, (13.0,), 0.005, 16)
    assert c["data"] == {**c["data"], "epoch": 61, "windows": 304, "occurrences": 4864}
    assert c["confirmation"]["epoch"] == 118 and c["confirmation"]["windows"] == 32 and c["confirmation"]["occurrences"] == 512
    assert c["tolerance"] == 0.005 and c["bisections"] == 16 and c["statuses"] == list(FS.STATUSES)
    assert FS.STATUSES == ("matched", "calibration_invalid", "calibration_unmatched", "confirmation_invalid")
    assert FS.IDENTITY["FS"]["mapping_learnable_parameters"] == 0 and FS.IDENTITY["FS"]["mapping_calibrated_scalars"] == 1
    assert FS.IDENTITY["FS"]["demand_source"] == FV.IDENTITY["F"]["demand_source"] == R.IDENTITY["W"]["demand_source"]


def test_width_table_is_65_equally_spaced_points_and_the_official_default():
    t = FS.candidates()
    assert len(t) == 66 and t == sorted(set(t)) and t[0] == 4.0 and t[-1] == 30.0 and 13.0 in t
    grid = [h for h in t if h != 13.0]
    assert len(grid) == 65 and all(abs((b - a) - 26.0 / 64) <= 1e-12 for a, b in zip(grid, grid[1:]))
    assert all(type(h) is float for h in t)
    for bad in (4, 13, 3.999, 30.001, float("nan"), float("inf"), "13.0", None):
        with pytest.raises(ValueError):
            FS.check_width(bad)


# ---------------------------------------------------------------- kernel and nodes
def test_kernel_cache_is_keyed_by_the_width():
    n = FS.GRID_SHAPE[0]
    assert FS.kernel_key(n, 4.0) != FS.kernel_key(n, 13.0) and FS.kernel_key(n, 13.0) != FS.kernel_key(51, 13.0)
    assert FS.kernel_key(n, 13.0) == (31, 30, (13.0).hex(), "float64")
    k4 = FS.kernel_matrix(n, 4.0)
    k13 = FS.kernel_matrix(n, 13.0)
    assert FS.kernel_matrix(n, 4.0) is k4 and FS.kernel_matrix(n, 13.0) is k13 and not torch.equal(k4, k13)
    assert torch.equal(k4, FV.kernel_matrix(n))                                # width 4 is arm F's kernel bit for bit
    h = 13.0 + 2.0 ** -40                                                      # a width that differs in the last bits only
    assert FS.kernel_key(n, h) != FS.kernel_key(n, 13.0) and FS.kernel_matrix(n, h) is not k13
    assert tuple(k13.shape) == (n, n + 2 * FS.PAD) and k13.dtype == F64
    for i, j in ((0, 0), (5, 30), (30, 90), (12, 42)):                          # padded column j is the position j - PAD
        r = i - (j - FS.PAD)
        want = math.exp(-4.0 * math.log(2.0) * r * r / (13.0 * 13.0)) if abs(r) <= FS.PAD else 0.0
        assert abs(float(k13[i, j]) - want) <= 1e-15
    assert float(k13[0, 0]) > 0 and float(k13[0, 61]) == 0.0 and float(k13[30, 29]) == 0.0      # support r = -30..30 only


@pytest.mark.parametrize("h", WIDTHS)
@pytest.mark.parametrize("name", CASES)
def test_attraction_nodes_match_the_loop_reference_at_every_width(name, h):
    d = demand_case(name) + W.EPS
    for marg, nodes in ((d.sum(0), 51), (d.sum(1), 31)):
        s = FV.node_saliency(marg[None], nodes)
        raw, g = FS.attraction_nodes(s, h)
        ref = ref_attraction(s[0].numpy(), h)
        assert np.abs(raw[0].numpy() - ref).max() <= 1e-12
        assert abs(float(raw[0, 0])) <= 1e-12 and abs(float(raw[0, -1]) - 1) <= 1e-12        # anti-crop end points at any width
        assert bool(torch.isfinite(raw).all())               # monotonicity is not assumed here: the calibration reads it
        assert np.abs(g[0].numpy() - np.clip(ref, 0, 1)).max() <= 1e-12                      # the official clamp only


def test_width_4_is_arm_f_bit_for_bit():
    every = torch.stack([demand_case(n) for n in ("constant",) + CASES])
    a, b = FV.plan_from_demand(every), FS.plan_from_demand(every, 4.0)
    assert torch.equal(a.gx, b.gx) and torch.equal(a.gy, b.gy) and torch.equal(a.identity, b.identity) and a.stats == b.stats
    assert b.identity.tolist() == [True] + [False] * len(CASES)
    cases = torch.stack([demand_case(n) for n in ("constant",) + SMOOTH])
    a = FV.plan_from_demand(cases)
    wide = FS.plan_from_demand(cases, 13.0)
    assert torch.equal(wide.identity, a.identity) and torch.equal(wide.gx[0], a.gx[0])       # a constant map stays uniform
    assert not torch.equal(wide.gx[1:], a.gx[1:]) and not torch.equal(wide.gy[1:], a.gy[1:])
    one = FS.plan_from_demand(cases[2:3], 13.0)                                             # per image: no batch dependence
    assert torch.equal(one.gx[0], wide.gx[2]) and torch.equal(one.gy[0], wide.gy[2])
    for st in wide.stats.values():
        assert st["node_slope_min"] > 0 and st["endpoint_error_max"] <= 1e-12 and st["clamp_max"] <= 1e-12
        assert st["clamped_interior_nodes"] == 0


def test_a_map_that_is_not_strictly_increasing_is_refused_and_flagged_never_repaired(monkeypatch):
    real = FS.attraction_nodes

    def plateau(s, h):
        raw, g = real(s, h)
        g = g.clone()
        g[:, 7] = g[:, 6]
        return raw, g

    def not_finite(s, h):
        raw, g = real(s, h)
        g = g.clone()
        g[0, 3] = float("nan")
        return raw, g

    def shifted_end(s, h):
        raw, g = real(s, h)
        raw = raw.clone()
        raw[:, 0] = 1e-9
        return raw, g

    def interior_clamp(s, h):
        raw, g = real(s, h)
        raw = raw.clone()
        raw[:, 4] = -1e-15
        return raw, g

    d = torch.stack([demand_case("centre"), demand_case("edge")])
    nodes, identity = FS.nodes_from_demand(d, 13.0)
    ok, flags = FS.valid_images(nodes, identity)
    assert ok.tolist() == [True, True] and not any(bool(v.any()) for v in flags.values())
    for broken, flag in ((plateau, "not_strictly_increasing"), (not_finite, "not_finite"), (shifted_end, "end_nodes"),
                         (interior_clamp, "clamp")):
        monkeypatch.setattr(FS, "attraction_nodes", broken)
        nodes, identity = FS.nodes_from_demand(d, 13.0)
        ok, flags = FS.valid_images(nodes, identity)
        assert not bool(ok.all()) and bool(flags[flag].any()), flag
        if flag in ("not_strictly_increasing", "not_finite"):
            with pytest.raises(RuntimeError):
                FS.plan_from_demand(d, 13.0)                    # the forward raises; nothing sorts or bounds the nodes
    monkeypatch.setattr(FS, "attraction_nodes", real)
    const = torch.stack([demand_case("constant"), demand_case("centre")])
    nodes, identity = FS.nodes_from_demand(const, 13.0)
    ok, _ = FS.valid_images(nodes, identity)
    assert identity.tolist() == [True, False] and ok.tolist() == [True, True]
    assert torch.equal(nodes["x"][1][0], torch.arange(51, dtype=F64) / 50)


# ---------------------------------------------------------------- selection rule (pure)
def response(error, valid=lambda h: True, l_w=0.25):
    """evaluate(h) of a synthetic geometry whose relative error exp(L_h - L_W) - 1 is error(h)."""
    calls = []

    def evaluate(h):
        calls.append(h)
        if not valid(h):
            return False, None, {"invalid_images": 1}
        return True, l_w + math.log1p(error(h)), {"invalid_images": 0}

    return evaluate, calls, l_w


TABLE = FS.candidates()


def test_select_takes_the_smallest_table_width_inside_the_tolerance():
    evaluate, calls, l_w = response(lambda h: (h - 10.0) * 0.01)
    out = FS.select(evaluate, l_w)
    inside = [h for h in TABLE if abs((h - 10.0) * 0.01) <= 0.005]
    assert len(inside) >= 2 and out["status"] == "matched" and out["h_star"] == min(inside) == 9.6875
    assert out["h_star_hex"] == (9.6875).hex() and out["interval"] is None and calls == TABLE
    assert abs(out["relative_error"] - (9.6875 - 10.0) * 0.01) <= 1e-12 and abs(out["relative_error"]) <= FS.TOLERANCE
    assert len(out["readings"]) == 66 and all(r["stage"] == "table" and r["valid"] for r in out["readings"])
    assert out["L_h_star"] == out["readings"][TABLE.index(9.6875)]["L"]


def test_select_bisects_the_first_valid_neighbours_with_a_sign_change():
    evaluate, calls, l_w = response(lambda h: (h - 10.3) * 0.1)
    out = FS.select(evaluate, l_w)
    lo, hi = max(h for h in TABLE if h < 10.3), min(h for h in TABLE if h > 10.3)
    assert out["status"] == "matched" and out["interval"]["lo"] == lo and out["interval"]["hi"] == hi
    mids = [r for r in out["readings"] if r["stage"].startswith("bisection")]
    assert len(mids) == 16 and calls[:66] == TABLE and all(lo < r["h"] < hi for r in mids)
    best = min(mids, key=lambda r: (abs(r["relative_error"]), r["h"]))
    assert out["h_star"] == best["h"] and abs(out["h_star"] - 10.3) <= (hi - lo) / 2 ** 15 and abs(out["relative_error"]) <= 1e-5
    assert out["rule_applied"] == "bisection midpoint with the smallest absolute error"
    # two sign changes: only the first pair of the sorted table is used
    evaluate, _, l_w = response(lambda h: 0.05 * (h - 8.2) if h < 12 else 0.05 * (15.9 - h))
    out = FS.select(evaluate, l_w)
    assert out["status"] == "matched" and out["interval"]["lo"] < 8.2 < out["interval"]["hi"] and abs(out["h_star"] - 8.2) <= 1e-4


def test_select_never_bridges_an_invalid_width():
    lo, hi = max(h for h in TABLE if h < 8.2), min(h for h in TABLE if h > 8.2)
    after = TABLE[TABLE.index(hi) + 1]
    # the width just above the crossing is invalid: its two valid neighbours are not adjacent in the table
    evaluate, calls, l_w = response(lambda h: 0.05 * (h - 8.2), valid=lambda h: h != hi)
    out = FS.select(evaluate, l_w)
    assert out["status"] == "calibration_unmatched" and out["h_star"] is None and out["interval"] is None and calls == TABLE
    assert out["closest"]["h"] == lo and [r["h"] for r in out["readings"] if not r["valid"]] == [hi]
    assert (lo - 8.2) * (after - 8.2) < 0                                      # a sign change exists across the invalid width
    # with a later sign change between valid neighbours, that later pair is the first admissible one
    evaluate, _, l_w = response(lambda h: 0.05 * (h - 8.2) if h < 12 else 0.05 * (15.9 - h), valid=lambda h: h != hi)
    out = FS.select(evaluate, l_w)
    assert out["status"] == "matched" and out["interval"]["lo"] < 15.9 < out["interval"]["hi"]


def test_select_statuses_without_a_match():
    evaluate, calls, l_w = response(lambda h: 0.0, valid=lambda h: False)
    out = FS.select(evaluate, l_w)                                             # nothing valid: no optimum over an empty set
    assert out["status"] == "calibration_invalid" and out["h_star"] is None and calls == TABLE and "closest" not in out
    evaluate, calls, l_w = response(lambda h: -0.2 + 0.005 * (h - 4.0))         # always below the target, closest at 30
    out = FS.select(evaluate, l_w)
    assert out["status"] == "calibration_unmatched" and out["h_star"] is None and out["closest"]["h"] == 30.0 and calls == TABLE
    assert abs(out["closest"]["relative_error"] - (-0.2 + 0.005 * 26)) <= 1e-12
    evaluate, calls, l_w = response(lambda h: (h - 10.3) * 0.1, valid=lambda h: h in TABLE)
    out = FS.select(evaluate, l_w)                                             # an invalid map inside the bisection interval
    assert out["status"] == "calibration_invalid" and out["h_star"] is None and len(calls) == 67
    assert out["interval"] is not None and out["readings"][-1]["valid"] is False
    evaluate, calls, l_w = response(lambda h: -0.02 if h < 10.3 else 0.02)      # a jump: bisection cannot reach the tolerance
    out = FS.select(evaluate, l_w)
    assert out["status"] == "calibration_unmatched" and out["rule_applied"] == "bisection did not reach the tolerance"
    assert len(calls) == 66 + 16 and out["h_star"] is None and abs(abs(out["closest"]["relative_error"]) - 0.02) <= 1e-12
    with pytest.raises(ValueError):                                            # a reading has a value exactly when it is valid
        FS.select(lambda h: (True, None, {}), 0.25)
    assert all(s in FS.STATUSES for s in ("matched", "calibration_invalid", "calibration_unmatched"))


def test_select_prefers_the_smaller_width_on_equal_errors():
    evaluate, _, l_w = response(lambda h: 0.004 if h in (TABLE[20], TABLE[40]) else 0.3)
    out = FS.select(evaluate, l_w)
    assert out["status"] == "matched" and out["h_star"] == TABLE[20]
    assert FS.relative_error(0.25 + math.log(1.004), 0.25) == pytest.approx(0.004, abs=1e-15)


# ---------------------------------------------------------------- source and binding
def test_mapping_binding_and_source_rules():
    cal = {"path": "/x/CALIBRATION.json", "bytes": 10, "sha256": "0" * 64}
    net = {"kind": "network", "file": {"path": "w"}}
    m = FS.check_source("FS", {**net, "fwhm": 13.5, "calibration": cal}, False)
    assert m == {"fwhm": 13.5, "fwhm_hex": (13.5).hex(), "calibration": cal} == FS.mapping({"fwhm": 13.5, "calibration": cal})
    assert FS.check_source("FS", {"kind": "synthetic", "seed": 0, "fwhm": 13.5}, True)["calibration"] is None
    for arm, source, check in (("F", {**net, "fwhm": 4.0, "calibration": cal}, False),        # arm F is not built here
                               ("W", {**net, "fwhm": 13.5, "calibration": cal}, False),
                               ("FS", {**net, "calibration": cal}, False),                    # no width
                               ("FS", {**net, "fwhm": 13, "calibration": cal}, False),        # an integer width
                               ("FS", {**net, "fwhm": 31.0, "calibration": cal}, False),      # outside the calibration range
                               ("FS", {**net, "fwhm": 13.5}, False),                          # formal source without the lock
                               ("FS", {"kind": "fixed", "fwhm": 13.5, "calibration": cal}, False),
                               ("FS", {"kind": "synthetic", "seed": 0, "fwhm": 13.5, "calibration": cal}, False),
                               ("FS", {"kind": "fixed", "fwhm": 13.5}, True)):
        with pytest.raises(ValueError):
            FS.check_source(arm, source, check)


# ---------------------------------------------------------------- wrapper
@pytest.fixture(scope="module")
def built():
    model, criterion, _ = build(Config(seed=0), synthetic=True)
    return model.eval(), criterion


def wrapped(built, source, h):
    source = {**source, "fwhm": h}
    model = copy.deepcopy(built[0])
    L0.wrap(model)
    FS.swap(model, FS._context(source, FS.check_source("FS", source, True)))
    return model


def wrapped_f(built, source):
    model = copy.deepcopy(built[0])
    L0.wrap(model)
    FV.swap(model, R.Context(source))
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


def test_swap_keeps_the_native_layout_and_the_arms_do_not_mix(built):
    model = wrapped(built, {"kind": "synthetic", "seed": 0}, 13.0)
    f_model = wrapped_f(built, {"kind": "synthetic", "seed": 0})
    assert list(model.state_dict()) == list(built[0].state_dict())
    assert [n for n, _ in model.named_modules()] == [n for n, _ in built[0].named_modules()]
    assert FS.is_wrapped(model) and not FV.is_wrapped(model) and not R.is_wrapped(model) and not L0.is_wrapped(model)
    assert FV.is_wrapped(f_model) and not FS.is_wrapped(f_model)
    assert type(model) is not type(f_model) and vars(model.r28_ctx)["r28_fwhm"] == 13.0
    with pytest.raises(ValueError):
        FS.swap(f_model, model.r28_ctx)                                         # an arm F model is not re-wrapped as FS
    with pytest.raises(ValueError):
        FS.swap(model, model.r28_ctx)                                           # already swapped
    plain = copy.deepcopy(built[0])
    L0.wrap(plain)
    no_width = R.Context({"kind": "synthetic", "seed": 0})
    with pytest.raises(ValueError):
        FS.swap(plain, no_width)                                                # a context without a width is not arm FS
    with pytest.raises(ValueError):
        model(torch.rand(1, 3, *L_HW))


def test_width_4_forward_is_the_arm_f_forward_and_a_wider_kernel_changes_it(built):
    x = images(1)
    f_model = wrapped_f(built, {"kind": "synthetic", "seed": 6})
    four = wrapped(built, {"kind": "synthetic", "seed": 6}, 4.0)
    wide = wrapped(built, {"kind": "synthetic", "seed": 6}, 13.0)
    with torch.no_grad():
        a, b, c = f_model(x), four(x), wide(x)
    assert set(a) == set(b) == set(c) == {"pred_logits", "pred_boxes"}
    assert torch.equal(a["pred_logits"], b["pred_logits"]) and torch.equal(a["pred_boxes"], b["pred_boxes"])
    assert not torch.equal(a["pred_boxes"], c["pred_boxes"])
    # the wide forward is: render with the width-13 plan, native forward, boxes mapped back with the same plan
    plan = FS.plan_from_demand(wide.r28_ctx.demand(x), 13.0)
    native = type(wide)._r28fs_native
    manual = wrapped(built, {"kind": "synthetic", "seed": 6}, 13.0)
    with torch.no_grad():
        raw = native.forward(manual, FV.resample(x, plan)[0], None)
        ref = FV.unwarp_outputs(plan, raw)
    assert torch.equal(c["pred_logits"], ref["pred_logits"]) and torch.equal(c["pred_boxes"], ref["pred_boxes"])
    assert float((c["pred_boxes"] - raw["pred_boxes"]).abs().max()) > 1e-4      # the output is not left in the warped frame


def test_constant_demand_is_the_l0_forward_bit_exact(built):
    x = images(2)
    plain = copy.deepcopy(built[0])
    L0.wrap(plain)
    model = wrapped(built, {"kind": "constant"}, 13.0)
    with torch.no_grad():
        a, b = model(x), plain(x)
    assert torch.equal(a["pred_logits"], b["pred_logits"]) and torch.equal(a["pred_boxes"], b["pred_boxes"])


def test_training_record_carries_the_demand_and_the_per_axis_magnification(built):
    x, t = images(4), targets(4)
    original = copy.deepcopy(t)
    model = wrapped(built, {"kind": "synthetic", "seed": 6}, 13.0).train()
    model.r28_ctx.observe = True
    torch.manual_seed(11)
    out = model(x, t)
    assert {"pred_logits", "pred_boxes", "aux_outputs", "dn_aux_outputs"} <= set(out)
    assert all(torch.equal(a["boxes"], b["boxes"]) for a, b in zip(t, original))            # the caller's targets are untouched
    (rec,) = model.r28_ctx.records
    assert rec["n"] == [5, 6] and {"g", "size", "short", "fovea", "demand", "xyxy", "ox", "oy"} <= set(rec)
    d = model.r28_ctx.demand(x)
    assert torch.equal(rec["demand"], d) and rec["demand"].dtype == F64 and rec["demand"].device.type == "cpu"
    plan = FS.plan_from_demand(d, 13.0)
    pad, n = R._padded_xyxy(t)
    ox, oy = FV.magnification(plan, pad)
    assert torch.equal(rec["ox"], ox) and torch.equal(rec["oy"], oy) and torch.equal(rec["xyxy"], pad) and n == rec["n"]
    for i, k in enumerate(n):
        assert float((rec["g"][i] - torch.sqrt(ox * oy)[i, :k]).abs().max()) <= 1e-12
    plan4 = FV.plan_from_demand(d)
    ox4, oy4 = FV.magnification(plan4, pad)
    assert float((ox - ox4).abs().max()) > 1e-4                                 # the record is the width-13 geometry, not arm F's
    model.r28_ctx.observe = False
    model.r28_ctx.records.clear()
    model.eval()
    with torch.no_grad():
        model(x)
    assert model.r28_ctx.records == []
