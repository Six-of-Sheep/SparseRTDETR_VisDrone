"""R28 wrapper checks (CPU, randomly initialised detector, synthetic images): TASK_R28_001 section 5 item 2 and
AMENDMENT_002 section 2.

Every forward here is a synthetic unit-test forward and belongs to no formal ledger.  The inverse map used as the
reference is written apart from the production one (numpy interpolation inside [0, 1], end slopes outside).
"""
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sparse_rtdetr.research import r27_l0_control as L0  # noqa: E402
from sparse_rtdetr.research import r28_resample as R  # noqa: E402
from sparse_rtdetr.research import r28_warp as W  # noqa: E402
from sparse_rtdetr.research.r27_local_memory import H_HW, L_HW  # noqa: E402
from sparse_rtdetr.protocol_r18.engine import tensor_digest  # noqa: E402
from sparse_rtdetr.protocol_r18.model import Config, build, fp32_outputs  # noqa: E402

torch.set_default_dtype(torch.float32)
torch.set_num_threads(min(torch.get_num_threads(), 16))
FAMILIES = ("aux_outputs", "enc_aux_outputs", "dn_aux_outputs")


@pytest.fixture(scope="module")
def built():
    model, criterion, _ = build(Config(seed=0), synthetic=True)
    return model.eval(), criterion


def wrapped(built, source):
    model = copy.deepcopy(built[0])
    L0.wrap(model)
    R.swap(model, R.Context(source))
    return model


def images(seed, batch=2):
    """Smooth random images (a detector with random weights still gives varied boxes)."""
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


def ref_inverse(knots, v):
    n = len(knots) - 1
    grid = np.arange(n + 1, dtype=np.float64) / n
    m0, m1 = (knots[1] - knots[0]) * n, (knots[-1] - knots[-2]) * n
    return np.where(v < 0, v / m0, np.where(v > 1, 1 + (v - 1) / m1, np.interp(v, knots, grid)))


def ref_unwarp(plan, boxes):
    """cxcywh (B, N, 4) in the warped canvas -> cxcywh in the source frame, FP64 numpy."""
    b = boxes.detach().double().numpy()
    kx, ky = plan.kx.numpy(), plan.ky.numpy()
    out = np.empty_like(b)
    for i in range(b.shape[0]):
        x1, x2 = (ref_inverse(kx[i], b[i, :, 0] + s * b[i, :, 2] / 2) for s in (-1, 1))
        y1, y2 = (ref_inverse(ky[i], b[i, :, 1] + s * b[i, :, 3] / 2) for s in (-1, 1))
        out[i] = np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], -1)
    return out


def all_boxes(out):
    return [("final", out["pred_boxes"])] + [(f"{k}[{i}]", d["pred_boxes"]) for k in FAMILIES for i, d in enumerate(out.get(k, ()))]


# ---------------------------------------------------------------- identity, layout, refusals
def test_identities_are_stable_and_distinct():
    for arm in R.ARMS:
        canonical = json.dumps(R.IDENTITY[arm], sort_keys=True, separators=(",", ":")).encode()
        assert R.IDENTITY_SHA256[arm] == hashlib.sha256(canonical).hexdigest()
        assert R.IDENTITY[arm]["detector_new_learnable_parameters"] == 0 and R.IDENTITY[arm]["definition"] == W.DEFINITION
    assert len({R.IDENTITY_SHA256["W"], R.IDENTITY_SHA256["K"], L0.IDENTITY_SHA256}) == 3
    assert W.DEFINITION["eps"] == 0.2 and W.DEFINITION["slope_bounds"] == [2 / 3, 1.5] and W.DEFINITION["grid"] == [12, 21]


def test_swap_keeps_the_native_layout_and_adds_nothing(built):
    plain = copy.deepcopy(built[0])
    L0.wrap(plain)
    model = wrapped(built, {"kind": "synthetic", "seed": 0})
    assert list(model.state_dict()) == list(plain.state_dict()) == list(built[0].state_dict())
    assert [n for n, _ in model.named_modules()] == [n for n, _ in built[0].named_modules()]
    assert tensor_digest(model.state_dict().items()) == tensor_digest(built[0].state_dict().items())
    assert R.is_wrapped(model) and not L0.is_wrapped(model) and L0.is_wrapped(plain) and not R.is_wrapped(plain)
    assert not any("r28" in k or "demand" in k for k in model.state_dict())
    with pytest.raises(ValueError):
        R.swap(copy.deepcopy(built[0]), R.Context({"kind": "constant"}))     # not L0-wrapped
    with pytest.raises(ValueError):
        R.swap(model, R.Context({"kind": "constant"}))                       # already swapped


def test_demand_source_rules():
    for arm, kind in (("W", "fixed"), ("K", "network"), ("W", "synthetic"), ("K", "constant")):
        with pytest.raises(ValueError):
            R.check_source(arm, {"kind": kind}, False)
    R.check_source("W", {"kind": "network"}, False)
    R.check_source("K", {"kind": "fixed"}, False)
    R.check_source("W", {"kind": "constant"}, True)
    with pytest.raises(ValueError):
        R.check_source("K", {"kind": "network"}, True)
    a, b = R.synthetic_demand(0), R.synthetic_demand(1)
    assert a.shape == W.GRID and float((a - b).abs().max()) > 0.1 and not bool(W.plan_from_demand(a[None]).identity.any())
    assert bool(W.plan_from_demand(R.Context({"kind": "constant"}).fixed[None]).identity.all())


def test_non_h_or_non_fp32_input_is_refused(built):
    model = wrapped(built, {"kind": "constant"})
    with pytest.raises(ValueError):
        model(torch.rand(1, 3, *L_HW))
    with pytest.raises(ValueError):
        model(torch.rand(1, 3, *H_HW, dtype=torch.float64))


# ---------------------------------------------------------------- identity path
def test_constant_demand_is_the_l0_forward_bit_exact(built):
    x = images(1)
    plain = copy.deepcopy(built[0])
    L0.wrap(plain)
    model = wrapped(built, {"kind": "constant"})
    with torch.no_grad():
        a, b = model(x), plain(x)
    assert set(a) == set(b) == {"pred_logits", "pred_boxes"}
    assert torch.equal(a["pred_logits"], b["pred_logits"]) and torch.equal(a["pred_boxes"], b["pred_boxes"])


# ---------------------------------------------------------------- non-identity path, evaluation
def test_non_identity_eval_maps_boxes_back_without_clipping(built):
    x = images(2)
    model = wrapped(built, {"kind": "synthetic", "seed": 3})
    native = type(model)._r28_native
    with torch.no_grad():
        out = model(x)
        plan = model.r28_ctx.plan(x)
        raw = native.forward(model, W.resample(x, plan))
    assert not bool(plan.identity.any())
    assert torch.equal(out["pred_logits"], raw["pred_logits"])
    ref = ref_unwarp(plan, raw["pred_boxes"])
    assert float(np.abs(out["pred_boxes"].double().numpy() - ref).max()) <= 1e-6
    assert float((out["pred_boxes"] - raw["pred_boxes"]).abs().max()) > 1e-3     # the map really moves boxes
    assert bool((out["pred_boxes"][..., 2:] > 0).all()) and out["pred_boxes"].dtype == torch.float32


def test_unwarp_extends_linearly_outside_the_unit_interval():
    plan = W.plan_from_demand(R.synthetic_demand(5)[None])
    boxes = torch.tensor([[[0.02, 0.5, 0.10, 0.2],       # left edge at -0.03
                           [0.97, 0.5, 0.12, 0.2],       # right edge at 1.03
                           [0.5, 0.01, 0.2, 0.06],       # top edge at -0.02
                           [0.5, 0.99, 0.2, 0.08],       # bottom edge at 1.03
                           [0.5, 0.5, 1.2, 1.3],         # larger than the canvas on both axes
                           [0.31, 0.62, 1e-4, 2e-4]]], dtype=torch.float32)
    out = R.unwarp_outputs(plan, {"pred_logits": torch.zeros(1, 6, 10), "pred_boxes": boxes})["pred_boxes"]
    ref = ref_unwarp(plan, boxes)
    assert float(np.abs(out.double().numpy() - ref).max()) <= 1e-6
    edges = W.cxcywh_to_xyxy(out.double())
    assert float(edges[0, 0, 0]) < 0 and float(edges[0, 1, 2]) > 1 and float(edges[0, 2, 1]) < 0 and float(edges[0, 3, 3]) > 1
    assert float(edges[0, 4, 0]) < 0 and float(edges[0, 4, 2]) > 1 and bool((out[..., 2:] > 0).all())
    mx0 = float(plan.mx[0, 0])
    assert abs(float(edges[0, 0, 0]) - (0.02 - 0.05) / mx0) <= 1e-6      # the end slope, not a clip at 0
    with pytest.raises(ValueError):
        R.unwarp_outputs(plan, {"pred_logits": torch.zeros(1, 6, 10), "pred_boxes": boxes, "other_boxes": boxes})


# ---------------------------------------------------------------- non-identity path, training
@pytest.fixture(scope="module")
def train_case(built):
    x, t = images(4), targets(4)
    original = copy.deepcopy(t)
    model = wrapped(built, {"kind": "synthetic", "seed": 6}).train()
    manual = wrapped(built, {"kind": "synthetic", "seed": 6}).train()
    native = type(model)._r28_native
    model.r28_ctx.observe = True
    torch.manual_seed(11)
    out = model(x, t)
    plan = manual.r28_ctx.plan(x)
    tw = R.warp_targets(plan, t)
    torch.manual_seed(11)
    raw = native.forward(manual, W.resample(x, plan), tw)
    return {"x": x, "t": t, "original": original, "out": out, "raw": raw, "plan": plan, "tw": tw, "model": model, "manual": manual}


def test_training_forward_maps_every_box_family(train_case):
    out, raw, plan = train_case["out"], train_case["raw"], train_case["plan"]
    assert set(out) == set(raw) == {"pred_logits", "pred_boxes", "aux_outputs", "enc_aux_outputs", "enc_meta", "dn_aux_outputs", "dn_meta"}
    mine, theirs = all_boxes(out), all_boxes(raw)
    assert [n for n, _ in mine] == [n for n, _ in theirs] and len(mine) == 1 + 2 + len(raw["enc_aux_outputs"]) + 3
    for (name, a), (_, b) in zip(mine, theirs):
        ref = ref_unwarp(plan, b)
        assert float(np.abs(a.detach().double().numpy() - ref).max()) <= 1e-6, name
        assert float((a - b).abs().max()) > 1e-3, name                      # no family is left in the warped frame
        assert a.dtype == torch.float32 and bool((a[..., 2:] > 0).all()), name
    for k in FAMILIES:
        for d, e in zip(out[k], raw[k]):
            assert torch.equal(d["pred_logits"], e["pred_logits"])
    assert torch.equal(out["pred_logits"], raw["pred_logits"])
    assert out["dn_meta"]["dn_num_split"] == raw["dn_meta"]["dn_num_split"] and out["enc_meta"] == raw["enc_meta"]


def test_targets_are_not_modified_and_denoising_reads_warped_boxes(train_case):
    t, original, tw, plan = train_case["t"], train_case["original"], train_case["tw"], train_case["plan"]
    for a, b, w in zip(t, original, tw):
        assert torch.equal(a["boxes"], b["boxes"]) and torch.equal(a["labels"], b["labels"]) and a["gt_ids"] == b["gt_ids"]
        assert w["labels"] is a["labels"] and w["boxes"] is not a["boxes"] and float((w["boxes"] - a["boxes"]).abs().max()) > 1e-4
        assert w["boxes"].dtype == a["boxes"].dtype and bool((w["boxes"][:, 2:] > 0).all())
    # forward map of the GT edges, checked against numpy interpolation of the knots
    for i, (a, w) in enumerate(zip(t, tw)):
        e = W.cxcywh_to_xyxy(a["boxes"].double()).numpy()
        grid_x, grid_y = (np.arange(plan.kx.shape[1]) / (plan.kx.shape[1] - 1), np.arange(plan.ky.shape[1]) / (plan.ky.shape[1] - 1))
        f = np.stack([np.interp(e[:, 0], grid_x, plan.kx[i].numpy()), np.interp(e[:, 1], grid_y, plan.ky[i].numpy()),
                      np.interp(e[:, 2], grid_x, plan.kx[i].numpy()), np.interp(e[:, 3], grid_y, plan.ky[i].numpy())], -1)
        assert float(np.abs(W.cxcywh_to_xyxy(w["boxes"].double()).numpy() - f).max()) <= 1e-6
    rec = train_case["model"].r28_ctx.records
    assert len(rec) == 1 and rec[0]["n"] == [5, 6] and rec[0]["identity_images"] == 0 and rec[0]["slope_min"] < 1 < rec[0]["slope_max"]
    ox, oy = W.magnification(plan.select(slice(0, 1)), W.cxcywh_to_xyxy(t[0]["boxes"].double())[None])
    assert float((rec[0]["g"][0] - torch.sqrt(ox * oy)[0]).abs().max()) <= 1e-12


def test_criterion_sees_source_frame_boxes_and_a_leak_changes_the_loss(built, train_case):
    criterion = copy.deepcopy(built[1]).train()
    out, raw, plan, t = train_case["out"], train_case["raw"], train_case["plan"], train_case["t"]

    def total(o):
        return {k: float(v) for k, v in criterion(fp32_outputs(o), t).items()}

    mine = total(out)
    # independent: every family of the raw output mapped by the numpy reference, then the same criterion
    ref = copy.copy(raw)
    ref["pred_boxes"] = torch.from_numpy(ref_unwarp(plan, raw["pred_boxes"])).float()
    for k in FAMILIES:
        ref[k] = [{"pred_logits": d["pred_logits"], "pred_boxes": torch.from_numpy(ref_unwarp(plan, d["pred_boxes"])).float()} for d in raw[k]]
    theirs = total(ref)
    assert set(mine) == set(theirs) and len(mine) == 21
    assert max(abs(mine[k] - theirs[k]) for k in mine) <= 1e-4 * max(1.0, max(abs(v) for v in theirs.values()))
    base = sum(theirs.values())
    # a single family left in the warped frame must change the box losses
    for name in ("final",) + FAMILIES:
        leak = copy.copy(ref)
        if name == "final":
            leak["pred_boxes"] = raw["pred_boxes"].detach().float()
        else:
            leak[name] = [{"pred_logits": d["pred_logits"], "pred_boxes": d["pred_boxes"].detach().float()} for d in raw[name]]
        assert abs(sum(total(leak).values()) - base) > 1e-3, name


def test_gradients_reach_the_detector_through_the_inverse_map(built, train_case):
    criterion = copy.deepcopy(built[1]).train()
    model, out, t = train_case["model"], train_case["out"], train_case["t"]
    loss = sum(criterion(fp32_outputs(out), t).values())
    assert bool(torch.isfinite(loss))
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(bool(torch.isfinite(g).all()) for g in grads)
    head = model.decoder.dec_bbox_head[-1].layers[-1].weight.grad
    assert head is not None and float(head.abs().max()) > 0
