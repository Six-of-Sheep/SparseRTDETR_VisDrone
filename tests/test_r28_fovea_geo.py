"""R28 arm FG checks (CPU, synthetic): TASK_R28_STRONG_F_CONTROL_009 section 1.

Every forward here is a synthetic unit-test forward and belongs to no formal ledger.  Covered: the fixed width and its
registered origin, the identity, the source rules (no other width, no calibration lock, the selection reference), the
class swap (arms F, FS and FG do not mix), the forward being arm FS's forward at that width bit for bit, and the
training record.
"""
import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sparse_rtdetr.research import r27_l0_control as L0  # noqa: E402
from sparse_rtdetr.research import r28_fovea as FV  # noqa: E402
from sparse_rtdetr.research import r28_fovea_cal as FS  # noqa: E402
from sparse_rtdetr.research import r28_fovea_geo as FG  # noqa: E402
from sparse_rtdetr.research import r28_resample as R  # noqa: E402
from sparse_rtdetr.research import r28_warp as W  # noqa: E402
from sparse_rtdetr.research.r27_local_memory import H_HW, L_HW  # noqa: E402
from sparse_rtdetr.protocol_r18.model import Config, build  # noqa: E402

torch.set_default_dtype(torch.float32)
torch.set_num_threads(min(torch.get_num_threads(), 16))
F64 = torch.float64
CAL = {"path": "/x/CALIBRATION.json", "bytes": 10, "sha256": "8dcfcd5cf2cbd7ef96a54b1832d9eabd6923a5b7da860a9e634784ed0fa05c59"}
SEL = {"kind": "fixed_training_geometry_selected_width", "fwhm": 19.03125, "fwhm_hex": "0x1.3080000000000p+4", "calibration": CAL,
       "calibration_status": "calibration_unmatched", "h_star": None}
NET = {"kind": "network", "file": {"path": "w"}}


# ---------------------------------------------------------------- definition
def test_fixed_width_identity_and_registered_selection():
    assert FG.WIDTH == 19.03125 and FG.WIDTH.hex() == FG.WIDTH_HEX == "0x1.3080000000000p+4" and float.fromhex(FG.WIDTH_HEX) == FG.WIDTH
    assert FG.WIDTH in FS.candidates() and FS.H_MIN < FG.WIDTH < FS.H_MAX          # a width of the 006 table, not a new one
    assert (FG.ARM, FG.ARMS, FG.GRID_SHAPE, FG.PAD) == ("FG", ("FG",), FV.GRID_SHAPE, FV.PAD)
    canonical = json.dumps(FG.IDENTITY["FG"], sort_keys=True, separators=(",", ":")).encode()
    assert FG.IDENTITY_SHA256["FG"] == hashlib.sha256(canonical).hexdigest()
    assert len({FG.IDENTITY_SHA256["FG"], FS.IDENTITY_SHA256["FS"], FV.IDENTITY_SHA256["F"], R.IDENTITY_SHA256["W"], R.IDENTITY_SHA256["K"],
                L0.IDENTITY_SHA256}) == 6
    d = FG.DEFINITION
    assert d["only_difference_from_arm_F"] == "attraction_fwhm 4 -> 19.03125" and d["base"]["identity_sha256"] == FV.IDENTITY_SHA256["F"]
    same = [k for k in FV.DEFINITION if k not in ("name", "attraction_fwhm", "kernel", "differences_from_official")]
    assert same and all(d[k] == FV.DEFINITION[k] for k in same)                    # grid, padding, eps, official source, render ...
    assert d["grid_shape"] == [31, 51] and d["padding_size"] == 30 and d["padding_mode"] == "reflect" and d["anti_crop"] is True
    s = FG.SELECTION
    assert s["kind"] == FG.SELECTION_KIND == "fixed_training_geometry_selected_width" and s["fwhm_hex"] == FG.WIDTH_HEX
    assert s["source"]["commit"] == "f376dc973d8333252d4cf50c9ed37d4fd8b9d244"
    assert s["source"]["release_manifest_sha256"] == "07247d2d2a19fca7daa14273617f178051f0c9e4a94ccd1effcb2000d8540cd3"
    assert s["source"]["calibration_sha256"] == CAL["sha256"]
    assert s["source"]["calibration_status"] == "calibration_unmatched" and s["source"]["h_star"] is None and s["reading"]["matched"] is False
    assert abs(s["reading"]["FG_pooled_small_magnification"] / s["reading"]["W_pooled_small_magnification"] - 1
               - s["reading"]["FG_over_W_minus_1"]) <= 1e-12 and abs(s["reading"]["FG_over_W_minus_1"]) > s["reading"]["tolerance_of_006"]
    i = FG.IDENTITY["FG"]
    assert (i["mapping_learnable_parameters"], i["mapping_calibrated_scalars"], i["mapping_fixed_scalars"]) == (0, 0, 1)
    assert i["demand_source"] == FV.IDENTITY["F"]["demand_source"] == R.IDENTITY["W"]["demand_source"]
    assert {"F", "FS", "equal_magnitude_control", "AP_selected", "preregistered"} <= set(i["is_not"])
    assert i["task"] == "TASK_R28_STRONG_F_CONTROL_009" and i["definition"] is d


def test_map_is_the_arm_fs_function_at_the_fixed_width():
    g = torch.Generator().manual_seed(3)
    d = W.blur3(torch.rand(*W.GRID, generator=g, dtype=F64) * 3)[None].repeat(2, 1, 1)
    d[1] = 0.7                                                                      # a constant map: the exact uniform nodes
    a, b, f = FG.plan_from_demand(d), FS.plan_from_demand(d, 19.03125), FV.plan_from_demand(d)
    assert torch.equal(a.gx, b.gx) and torch.equal(a.gy, b.gy) and torch.equal(a.identity, b.identity) and a.stats == b.stats
    assert a.identity.tolist() == [False, True] and not torch.equal(a.gx[0], f.gx[0]) and torch.equal(a.gx[1], f.gx[1])
    assert bool((a.gx[:, 1:] > a.gx[:, :-1]).all()) and bool((a.gy[:, 1:] > a.gy[:, :-1]).all())


def test_mapping_binding_and_source_rules():
    good = {**NET, "fwhm": FG.WIDTH, "selection": SEL}
    m = FG.check_source("FG", good, False)
    assert m == {"fwhm": 19.03125, "fwhm_hex": "0x1.3080000000000p+4", "selection": SEL} == FG.mapping(good)
    assert "calibration" not in m                                                   # never the shape of an arm FS lock
    assert FG.check_source("FG", {"kind": "synthetic", "seed": 0, "fwhm": FG.WIDTH}, True)["selection"] is None
    for arm, source, check in (("F", good, False), ("FS", good, False), ("W", good, False),
                               ("FG", {**NET, "selection": SEL}, False),                       # no width
                               ("FG", {**good, "fwhm": 4.0}, False),                           # arm F's width
                               ("FG", {**good, "fwhm": 13.0}, False),                          # the official default
                               ("FG", {**good, "fwhm": 19.4375}, False),                       # the next table width
                               ("FG", {**good, "fwhm": 19}, False),                            # an integer
                               ("FG", {**good, "fwhm": 19.03125 + 2.0 ** -40}, False),         # one bit pattern only
                               ("FG", {**NET, "fwhm": FG.WIDTH}, False),                       # formal source without the selection
                               ("FG", {**good, "calibration": CAL}, False),                    # posing as a matched lock
                               ("FG", {**good, "selection": {**SEL, "calibration_status": "matched"}}, False),
                               ("FG", {**good, "selection": {**SEL, "h_star": 19.03125}}, False),
                               ("FG", {**good, "selection": {**SEL, "fwhm_hex": (13.0).hex()}}, False),
                               ("FG", {**good, "selection": {**SEL, "kind": "calibrated"}}, False),
                               ("FG", {**good, "selection": {**SEL, "calibration": {**CAL, "sha256": "0" * 64}}}, False),
                               ("FG", {**good, "selection": {**SEL, "calibration_status": "matched"}}, True),    # also inside a check
                               ("FG", {**good, "kind": "fixed"}, False),
                               ("FG", {"kind": "synthetic", "seed": 0, "fwhm": FG.WIDTH, "selection": SEL}, False),
                               ("FG", {"kind": "fixed", "fwhm": FG.WIDTH}, True)):
        with pytest.raises(ValueError):
            FG.check_source(arm, source, check)
    # arm FS keeps its own guard: a formal FS source still needs a calibration file, whatever the width
    with pytest.raises(ValueError):
        FS.check_source("FS", {**NET, "fwhm": FG.WIDTH}, False)
    with pytest.raises(ValueError):
        FS.check_source("FG", {**NET, "fwhm": FG.WIDTH, "calibration": CAL}, False)


# ---------------------------------------------------------------- wrapper
@pytest.fixture(scope="module")
def built():
    model, criterion, _ = build(Config(seed=0), synthetic=True)
    return model.eval(), criterion


def wrapped(built, source):
    source = {**source, "fwhm": FG.WIDTH}
    model = copy.deepcopy(built[0])
    L0.wrap(model)
    FG.swap(model, FG._context(source, FG.check_source("FG", source, True)))
    return model


def wrapped_fs(built, source, h):
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


def test_swap_keeps_the_native_layout_and_f_fs_fg_do_not_mix(built):
    src = {"kind": "synthetic", "seed": 0}
    model, fs_model, f_model = wrapped(built, src), wrapped_fs(built, src, FG.WIDTH), wrapped_f(built, src)
    assert list(model.state_dict()) == list(built[0].state_dict())
    assert [n for n, _ in model.named_modules()] == [n for n, _ in built[0].named_modules()]
    assert FG.is_wrapped(model) and not FS.is_wrapped(model) and not FV.is_wrapped(model) and not R.is_wrapped(model) and not L0.is_wrapped(model)
    assert FS.is_wrapped(fs_model) and not FG.is_wrapped(fs_model)                  # the same width does not make an FS model FG
    assert FV.is_wrapped(f_model) and not FG.is_wrapped(f_model)
    assert len({type(model), type(fs_model), type(f_model)}) == 3 and vars(model.r28_ctx)["r28_fwhm"] == FG.WIDTH
    for other in (fs_model, f_model, model):
        with pytest.raises(ValueError):
            FG.swap(other, model.r28_ctx)                                           # F / FS / an already swapped model are not re-wrapped
    with pytest.raises(ValueError):
        FS.swap(model, model.r28_ctx)                                               # and an FG model is not re-wrapped as FS
    plain = copy.deepcopy(built[0])
    L0.wrap(plain)
    other_width = R.Context(src)
    vars(other_width)["r28_fwhm"] = 13.0
    with pytest.raises(ValueError):
        FG.swap(plain, other_width)                                                 # a context with another width is not arm FG
    plain2 = copy.deepcopy(built[0])
    L0.wrap(plain2)
    with pytest.raises(ValueError):
        FG.swap(plain2, R.Context(src))                                             # nor one without a width
    vars(model.r28_ctx)["r28_fwhm"] = 13.0                                          # a width changed after the swap is refused at the forward
    assert not FG.is_wrapped(model)
    with pytest.raises(ValueError):
        model(images(0, 1))
    vars(model.r28_ctx)["r28_fwhm"] = FG.WIDTH
    with pytest.raises(ValueError):
        model(torch.rand(1, 3, *L_HW))


def test_forward_is_the_arm_fs_forward_at_the_fixed_width_bit_for_bit(built):
    x = images(1)
    src = {"kind": "synthetic", "seed": 6}
    fg, fs, f = wrapped(built, src), wrapped_fs(built, src, FG.WIDTH), wrapped_f(built, src)
    with torch.no_grad():
        a, b, c = fg(x), fs(x), f(x)
    assert set(a) == set(b) == {"pred_logits", "pred_boxes"}
    assert torch.equal(a["pred_logits"], b["pred_logits"]) and torch.equal(a["pred_boxes"], b["pred_boxes"])
    assert not torch.equal(a["pred_boxes"], c["pred_boxes"])                        # not arm F's width-4 forward
    plan = FG.plan_from_demand(fg.r28_ctx.demand(x))
    native = type(fg)._r28fg_native
    with torch.no_grad():
        raw = native.forward(wrapped(built, src), FV.resample(x, plan)[0], None)
        ref = FV.unwarp_outputs(plan, raw)
    assert torch.equal(a["pred_logits"], ref["pred_logits"]) and torch.equal(a["pred_boxes"], ref["pred_boxes"])
    assert float((a["pred_boxes"] - raw["pred_boxes"]).abs().max()) > 1e-4          # the output is not left in the warped frame


def test_constant_demand_is_the_l0_forward_bit_exact(built):
    x = images(2)
    plain = copy.deepcopy(built[0])
    L0.wrap(plain)
    model = wrapped(built, {"kind": "constant"})
    with torch.no_grad():
        a, b = model(x), plain(x)
    assert torch.equal(a["pred_logits"], b["pred_logits"]) and torch.equal(a["pred_boxes"], b["pred_boxes"])


def test_training_record_is_the_fixed_width_geometry_with_the_demand(built):
    x, t = images(4), targets(4)
    original = copy.deepcopy(t)
    model = wrapped(built, {"kind": "synthetic", "seed": 6}).train()
    model.r28_ctx.observe = True
    torch.manual_seed(11)
    out = model(x, t)
    assert {"pred_logits", "pred_boxes", "aux_outputs", "dn_aux_outputs"} <= set(out)
    assert all(torch.equal(a["boxes"], b["boxes"]) for a, b in zip(t, original))    # the caller's targets are untouched
    (rec,) = model.r28_ctx.records
    assert rec["n"] == [5, 6] and {"g", "size", "short", "fovea", "demand", "xyxy", "ox", "oy"} <= set(rec)
    d = model.r28_ctx.demand(x)
    plan = FG.plan_from_demand(d)
    pad, n = R._padded_xyxy(t)
    ox, oy = FV.magnification(plan, pad)
    assert torch.equal(rec["demand"], d) and torch.equal(rec["ox"], ox) and torch.equal(rec["oy"], oy) and n == rec["n"]
    ox4, _ = FV.magnification(FV.plan_from_demand(d), pad)
    assert float((ox - ox4).abs().max()) > 1e-4                                     # not arm F's geometry
    assert rec["fovea"]["taps_x"] >= 2 and rec["fovea"]["clamped_interior_nodes"] == 0
