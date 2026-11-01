"""Synthetic checks for the tiny-band geometry term (CPU only; CUDA must be hidden).

Every check is value-level or bitwise on small synthetic tensors; the installed-criterion
integration is covered separately by ``tiny_smoke.py`` and ``geometry_check.py`` on the
real engine path.  The tests pin a synthetic constant C for the logic checks only
(``tg.CONSTANT_C``/``arm.CONSTANT_C``); the frozen chain value is the plan-verified module
constant and is exercised by ``geometry_probe.py``.

usage: <frozen python> -I tools/test_tiny_geometry.py
"""
from __future__ import annotations

import copy
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sparse_rtdetr.protocol_r18 import tiny_geometry as tg          # noqa: E402
from sparse_rtdetr.protocol_r18 import tiny_geometry_arm as arm    # noqa: E402

TEST_C = 0.1
tg.CONSTANT_C = TEST_C
arm.CONSTANT_C = TEST_C

RESULTS = []


def check(name, condition, detail=""):
    RESULTS.append((name, bool(condition)))
    print(("PASS " if condition else "FAIL ") + name + (" | " + str(detail) if detail else ""))
    if not condition:
        raise SystemExit("FAILED: " + name)


class Ops:
    @staticmethod
    def generalized_box_iou(a, b):
        import torchvision
        return torchvision.ops.generalized_box_iou(a, b)

    @staticmethod
    def box_iou(a, b):
        import torchvision
        return torchvision.ops.box_iou(a, b)


def make_case(pred_rows, tgt_rows):
    pred = torch.tensor(pred_rows, dtype=torch.float32)
    tgt = torch.tensor(tgt_rows, dtype=torch.float32)
    outputs = {"pred_boxes": pred}
    targets = [{"boxes": tgt, "labels": torch.zeros(len(tgt_rows), dtype=torch.long)}]
    indices = [(torch.arange(len(tgt_rows)), torch.arange(len(tgt_rows)))]

    def base(outputs, targets, indices, num_boxes, boxes_weight=None):
        p = outputs["pred_boxes"]
        l1 = torch.nn.functional.l1_loss(p, tgt, reduction="none").sum() / num_boxes
        giou = torch.diag(Ops.generalized_box_iou(tg._xyxy(p), tg._xyxy(tgt)))
        return {"loss_bbox": l1, "loss_giou": (1 - giou).sum() / num_boxes}

    return pred, tgt, outputs, targets, indices, float(len(tgt_rows)), base


def manual_native(pred, tgt):
    giou = torch.diag(Ops.generalized_box_iou(tg._xyxy(pred), tg._xyxy(tgt)))
    return 1 - giou


def manual_small(pred, tgt, constant=TEST_C):
    return 1 - torch.exp(-tg.wasserstein_sqrt(pred, tgt) / constant)


TINY = [0.40, 0.40, 0.020, 0.020]          # area 4.0e-4 <= threshold
TINY2 = [0.60, 0.40, 0.025, 0.016]         # area 4.0e-4
BIG = [0.50, 0.50, 0.400, 0.300]           # area 1.2e-1


def test_formula():
    torch.manual_seed(0)
    pred = torch.rand(64, 4) * 0.8 + 0.1
    tgt = torch.rand(64, 4) * 0.8 + 0.1
    s = tg.wasserstein_sqrt(pred, tgt)
    manual = torch.sqrt((pred[:, 0] - tgt[:, 0]) ** 2 + (pred[:, 1] - tgt[:, 1]) ** 2
                        + 0.25 * ((pred[:, 2] - tgt[:, 2]) ** 2 + (pred[:, 3] - tgt[:, 3]) ** 2))
    check("formula.w2", bool(torch.allclose(s, manual, atol=1e-6, rtol=0)), float((s - manual).abs().max()))
    p = tg.penalty_small(pred, tgt)
    check("formula.penalty", bool(torch.allclose(p, 1 - torch.exp(-manual / TEST_C), atol=1e-6, rtol=0)))
    check("formula.range", bool(((p >= 0) & (p < 1)).all()))
    base_box = torch.tensor([[0.5, 0.5, 0.1, 0.1]])
    near = manual_small(base_box, torch.tensor([[0.51, 0.5, 0.1, 0.1]]))
    far = manual_small(base_box, torch.tensor([[0.55, 0.5, 0.1, 0.1]]))
    grow = manual_small(base_box, torch.tensor([[0.5, 0.5, 0.12, 0.1]]))
    check("formula.monotone", float(near) < float(far) and float(near) < float(grow))


def test_band_boundary():
    w = math.sqrt(tg.BAND_AREA_THRESHOLD)
    at = torch.tensor([[0.5, 0.5, w, w]])
    above = torch.tensor([[0.5, 0.5, w * (1 + 1e-3), w]])
    check("band.inclusive", bool(tg.tiny_mask(at).all()))
    check("band.above", not bool(tg.tiny_mask(above).any()))


def test_out_of_band_is_native():
    pred, tgt, outputs, targets, indices, num, base = make_case(
        [[0.50, 0.50, 0.40, 0.30]], [BIG])
    native = base(outputs, targets, indices, num)
    term = tg.TinyGeometryTerm("A2")
    out = term.loss_boxes(base, Ops, outputs, targets, indices, num, None, family="final")
    check("native.bbox_identical", torch.equal(out["loss_bbox"], native["loss_bbox"]))
    check("native.giou_identical", torch.equal(out["loss_giou"], native["loss_giou"]))
    check("native.no_draw", term.draws == 0)


def test_in_band_replaces():
    pred, tgt, outputs, targets, indices, num, base = make_case(
        [[0.42, 0.40, 0.024, 0.018]], [TINY])
    term = tg.TinyGeometryTerm("A2")
    out = term.loss_boxes(base, Ops, outputs, targets, indices, num, None, family="final")
    expected = manual_small(outputs["pred_boxes"], tgt).sum() / num
    check("band.replaced", abs(float(out["loss_giou"]) - float(expected)) < 1e-6,
          (float(out["loss_giou"]), float(expected)))
    native_sum = manual_native(outputs["pred_boxes"], tgt).sum() / num
    check("band.differs_from_native", abs(float(out["loss_giou"]) - float(native_sum)) > 1e-9)
    check("band.stats", term.last_stats["geo_tiny_pairs"] == 1 and term.last_stats["geo_shift"] is None)


def test_mixed_pairs():
    pred_rows = [[0.42, 0.40, 0.024, 0.018], [0.50, 0.50, 0.40, 0.30]]
    tgt_rows = [TINY, BIG]
    pred, tgt, outputs, targets, indices, num, base = make_case(pred_rows, tgt_rows)
    term = tg.TinyGeometryTerm("A2")
    out = term.loss_boxes(base, Ops, outputs, targets, indices, num, None, family="final")
    small = manual_small(outputs["pred_boxes"][:1], tgt[:1])
    native = manual_native(outputs["pred_boxes"], tgt)
    expected = (small.sum() + native[1]).item() / num
    check("mixed.aggregate", abs(float(out["loss_giou"]) - expected) < 1e-6)


def test_a1_shift():
    pred_rows = [[0.30, 0.30, 0.022, 0.020], [0.60, 0.30, 0.022, 0.020], [0.60, 0.60, 0.022, 0.020]]
    tgt_rows = [[0.31, 0.31, 0.021, 0.020], [0.59, 0.31, 0.021, 0.020], [0.61, 0.61, 0.021, 0.020]]
    term = tg.TinyGeometryTerm("A1")
    pred, tgt, outputs, targets, indices, num, base = make_case(pred_rows, tgt_rows)
    out = term.loss_boxes(base, Ops, outputs, targets, indices, num, None, family="final")
    check("a1.draw_once", term.draws == 1, term.draws)
    offset = term.last_stats["geo_shift"]
    check("a1.shift_recorded", offset in (1, 2), offset)
    rotated = tgt[torch.tensor([(i + offset) % 3 for i in range(3)])]
    expected = manual_small(outputs["pred_boxes"], rotated).sum() / num
    check("a1.rotated_targets", abs(float(out["loss_giou"]) - float(expected)) < 1e-6)
    native = manual_native(outputs["pred_boxes"], tgt).sum() / num
    check("a1.differs_from_native", abs(float(out["loss_giou"]) - float(native)) > 1e-9)


def test_a1_singleton_and_empty():
    pred, tgt, outputs, targets, indices, num, base = make_case([[0.42, 0.40, 0.024, 0.018]], [TINY])
    t1 = tg.TinyGeometryTerm("A1")
    out1 = t1.loss_boxes(base, Ops, outputs, targets, indices, num, None, family="final")
    t2 = tg.TinyGeometryTerm("A2")
    out2 = t2.loss_boxes(base, Ops, outputs, targets, indices, num, None, family="final")
    check("a1.singleton_no_draw", t1.draws == 0 and t1.last_stats["geo_shift"] is None)
    check("a1.singleton_equals_a2", torch.equal(out1["loss_giou"], out2["loss_giou"]))
    empty = make_case([[0.5, 0.5, 0.1, 0.1]], [[0.5, 0.5, 0.1, 0.1]])
    _, _, o, t, i, n, b = empty
    t["boxes"] = t["boxes"][:0]
    i = [(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))]
    term = tg.TinyGeometryTerm("A1")
    out = term.loss_boxes(b, Ops, o, [{"boxes": t["boxes"][:0], "labels": torch.empty(0, dtype=torch.long)}],
                          i, 1.0, None, family="final")
    check("empty.zero", float(out["loss_giou"]) == 0.0 and term.draws == 0)


def test_gradients():
    pred_rows = [[0.42, 0.40, 0.024, 0.018], [0.50, 0.50, 0.40, 0.30]]
    tgt_rows = [TINY, BIG]
    pred, tgt, outputs, targets, indices, num, base = make_case(pred_rows, tgt_rows)
    outputs["pred_boxes"] = pred.clone().requires_grad_(True)
    term = tg.TinyGeometryTerm("A2")
    out = term.loss_boxes(base, Ops, outputs, targets, indices, num, None, family="final")
    total = out["loss_bbox"] + out["loss_giou"]
    total.backward()
    g = outputs["pred_boxes"].grad
    check("grad.finite", bool(torch.isfinite(g).all()) and float(g.abs().sum()) > 0)
    # outside the band the armed path must reproduce the native gradient bitwise
    p2, t2, o2, tg2, i2, n2, b2 = make_case([[0.50, 0.50, 0.40, 0.30]], [BIG])
    o2["pred_boxes"] = p2.clone().requires_grad_(True)
    nat = b2(o2, tg2, i2, n2)
    (nat["loss_bbox"] + nat["loss_giou"]).backward()
    g_native = o2["pred_boxes"].grad.clone()
    p3, t3, o3, tg3, i3, n3, b3 = make_case([[0.50, 0.50, 0.40, 0.30]], [BIG])
    o3["pred_boxes"] = p3.clone().requires_grad_(True)
    term3 = tg.TinyGeometryTerm("A2")
    out3 = term3.loss_boxes(b3, Ops, o3, tg3, i3, n3, None, family="final")
    (out3["loss_bbox"] + out3["loss_giou"]).backward()
    check("grad.native_bitwise", torch.equal(g_native, o3["pred_boxes"].grad))


def test_state_and_deepcopy():
    term = tg.TinyGeometryTerm("A1")
    pred_rows = [[0.30, 0.30, 0.022, 0.020], [0.60, 0.30, 0.022, 0.020]]
    tgt_rows = [[0.31, 0.31, 0.021, 0.020], [0.59, 0.31, 0.021, 0.020]]
    _, _, o, t, i, n, b = make_case(pred_rows, tgt_rows)
    term.loss_boxes(b, Ops, o, t, i, n, None, family="final")
    state = copy.deepcopy(term.state_dict())
    clone = tg.TinyGeometryTerm("A1")
    clone.load_state_dict(state)
    check("state.roundtrip", torch.equal(clone.state_dict()["generator"], term.state_dict()["generator"])
          and clone.draws == term.draws == 1)
    twin = copy.deepcopy(term)
    a = int(torch.randint(2 ** 31, (1,), generator=term.generator).item())
    c = int(torch.randint(2 ** 31, (1,), generator=twin.generator).item())
    check("state.deepcopy_stream", a == c)


def test_arm_adapter():
    criterion = SimpleNamespace()
    model = SimpleNamespace()
    term = arm.install(model, criterion, "A2")
    check("arm.installed", arm.installed(SimpleNamespace(criterion=criterion)))
    check("arm.assert", arm.assert_installed(model, criterion, "A2", term=term) is True)
    try:
        arm.install(model, criterion, "A2")
        check("arm.double_install_refused", False)
    except ValueError:
        check("arm.double_install_refused", True)
    engine = SimpleNamespace(criterion=criterion, geo_mode="A2")
    check("arm.declared", arm.declared(engine) == arm.declaration("A2"))
    saved = arm.state(engine)
    other = SimpleNamespace(criterion=SimpleNamespace(), geo_mode="A2")
    arm.install(SimpleNamespace(), other.criterion, "A2")
    arm.load(other, saved)
    check("arm.state_load", other.criterion.tiny_geometry.draws == term.draws)
    bad = SimpleNamespace(criterion=SimpleNamespace(), geo_mode="A1")
    arm.install(SimpleNamespace(), bad.criterion, "A1")
    try:
        arm.load(bad, saved)
        check("arm.cross_arm_refused", False)
    except ValueError:
        check("arm.cross_arm_refused", True)
    native = SimpleNamespace(criterion=SimpleNamespace(), geo_mode=None)
    native.criterion.tiny_geometry_declaration = arm.NATIVE_DECLARATION
    check("arm.native_declaration", arm.state(native) is None and arm.declared(native) == arm.NATIVE_DECLARATION)


def main():
    for name, fn in (("formula", test_formula), ("band_boundary", test_band_boundary),
                     ("out_of_band_is_native", test_out_of_band_is_native),
                     ("in_band_replaces", test_in_band_replaces),
                     ("mixed_pairs", test_mixed_pairs), ("a1_shift", test_a1_shift),
                     ("a1_singleton_and_empty", test_a1_singleton_and_empty),
                     ("gradients", test_gradients), ("state_and_deepcopy", test_state_and_deepcopy),
                     ("arm_adapter", test_arm_adapter)):
        fn()
    passed = sum(1 for _, ok in RESULTS if ok)
    print("TESTS %d/%d PASS" % (passed, len(RESULTS)))
    assert not torch.cuda.is_initialized(), "the synthetic suite must run with CUDA hidden"
    assert passed == len(RESULTS)


if __name__ == "__main__":
    main()
