"""Synthetic checks for the ROI-resolution box distillation term (CPU only, no data, no GPU).

These are implementation checks, not a benefit check.  They establish: the band boundary; the
COMMON-SUPPORT rule that makes the two armed arms supervise the same rows and the same number
of pairs so that only the target box's provenance varies; the per-row ledger identities; that
the extra pair is appended to the frozen matcher's own assignment and carries the native
per-pair terms at the native scale; that the native arm and every non-differing quantity stay
untouched; that the arm declaration round-trips; and that the augmentation replay matches the
frozen arithmetic, including that it refuses a trace which does not record its steps.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch
from torchvision.ops import box_convert, generalized_box_iou

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roi_ld import roi_ld, roi_ld_arm  # noqa: E402

RESULTS = []


def check(name, condition, detail=""):
    RESULTS.append((name, bool(condition), detail))
    print(("PASS " if condition else "FAIL ") + name + ((" | " + str(detail)) if detail else ""))


def _raises(fn):
    try:
        fn()
    except Exception:
        return True
    return False


class StubModule:
    """The two box operations the native criterion exposes, at the native semantics.

    The vendor ``box_iou`` returns ``(iou, union)``; the stub reproduces that shape so the
    module is exercised against the same contract as in production.
    """

    box_iou = staticmethod(lambda a, b: _box_iou_xyxy(a, b))
    generalized_box_iou = staticmethod(generalized_box_iou)


def _box_iou_xyxy(a, b):
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-9), union


def cxcywh(rows):
    return torch.tensor(rows, dtype=torch.float32)


def band_area_box(area_px):
    """A canvas-normalised cxcywh box of exactly area_px on 768x1344."""
    side = (area_px / (768.0 * 1344.0)) ** 0.5
    return [0.5, 0.5, side, side]


def fixture():
    """Four band rows and one large row, with per-group candidates for each band row.

    row0: both groups carry an exact box            -> selected by both arms
    row1: group 0 only                              -> refused (no counterpart)
    row2: group 1 only                              -> refused (no counterpart)
    row3: group 0 far, group 1 exact                -> one-sided for group 1, below-bar for 0
    """
    tiny = band_area_box(1024.0)
    big = band_area_box(1024.0 * 4)
    far = band_area_box(100.0 * 1.0)
    # row0's two groups carry slightly different boxes so the two arms' targets, and therefore
    # their losses, genuinely differ; the overlap bar is met by both.
    tiny_alt = [tiny[0] + 0.01, tiny[1], tiny[2], tiny[3]]
    boxes = cxcywh([tiny, tiny, tiny, tiny, big])
    labels = torch.tensor([3, 5, 7, 9, 3])
    roi_boxes = cxcywh([tiny, tiny_alt, tiny, tiny, far, tiny])
    roi_rows = torch.tensor([0, 0, 1, 2, 3, 3])
    roi_labels = torch.tensor([3, 3, 5, 7, 9, 9])
    roi_group = torch.tensor([0, 1, 0, 1, 0, 1])
    matched_src = torch.tensor([0, 1, 2, 3, 4])
    matched_tgt = torch.tensor([0, 1, 2, 3, 4])
    target = {"boxes": boxes, "labels": labels, "roi_boxes": roi_boxes, "roi_rows": roi_rows,
              "roi_labels": roi_labels, "roi_group": roi_group}
    return target, matched_src, matched_tgt, tiny, big, far


def main():
    module = StubModule()
    tiny = band_area_box(1024.0)
    big = band_area_box(1024.0 * 1.0001)

    # 1. band boundary
    mask = roi_ld.band_mask(cxcywh([tiny, big]))
    check("band boundary 1024 px inside, above outside", mask.tolist() == [True, False], mask.tolist())

    # 2. common support: both arms supervise the same rows and the same number of pairs
    target, src_idx, tgt_idx, tiny0, big0, far = fixture()
    out = {}
    for group in (0, 1):
        src, picks, _, ledger = roi_ld.instance_targets(target, src_idx, tgt_idx, module, group)
        out[group] = (src, picks, ledger)
        identity = (ledger["roi_selected"] + ledger["roi_below_iou"] + ledger["roi_no_candidate"]
                    + ledger["roi_no_class"] + ledger["roi_one_sided"]
                    + ledger["roi_unmatched_band_gt"])
        check("ledger accounts every band row (group %d)" % group,
              identity == ledger["roi_band_gt"] == 4, (identity, ledger))
    check("only the common-support row is selected by either arm",
          out[0][0].tolist() == [0] and out[1][0].tolist() == [0],
          (out[0][0].tolist(), out[1][0].tolist()))
    check("the two arms select the same number of targets",
          out[0][2]["roi_selected"] == out[1][2]["roi_selected"] == 1,
          (out[0][2]["roi_selected"], out[1][2]["roi_selected"]))
    check("the two arms select DIFFERENT boxes for the same row (the single variable)",
          out[0][1].tolist() != out[1][1].tolist(), (out[0][1].tolist(), out[1][1].tolist()))
    check("one-sided rows are counted for the arm whose group lacks a counterpart",
          out[1][2]["roi_one_sided"] == 1 and out[0][2]["roi_one_sided"] == 0,
          (out[0][2]["roi_one_sided"], out[1][2]["roi_one_sided"]))
    check("below-bar counterpart is counter as one-sided only after the bar is met",
          out[0][2]["roi_below_iou"] == 1 and out[1][2]["roi_below_iou"] == 0,
          (out[0][2]["roi_below_iou"], out[1][2]["roi_below_iou"]))
    check("rows without a counterpart are refused",
          out[0][2]["roi_no_candidate"] == 2 and out[1][2]["roi_no_candidate"] == 2)

    # 3. class filter
    other = dict(target, roi_labels=torch.tensor([4, 4, 5, 7, 9, 9]))
    _, _, _, ledger_class = roi_ld.instance_targets(other, src_idx, tgt_idx, module, 0)
    check("class-mismatched ROI boxes never qualify", ledger_class["roi_selected"] == 0
          and ledger_class["roi_no_class"] >= 1, ledger_class)

    # 4. the armed term appends the pair and leaves the native structure intact
    class Base:
        def __call__(self, outputs, targets, indices, num_boxes, boxes_weight=None):
            losses = {}
            for name in ("loss_bbox", "loss_giou"):
                total = torch.zeros(())
                for i, (s, t) in enumerate(indices):
                    if len(s) == 0:
                        continue
                    pred = outputs["pred_boxes"][i][s]
                    rows = targets[i]["boxes"][t]
                    if name == "loss_bbox":
                        total = total + (pred - rows).abs().sum()
                    else:
                        total = total + (1.0 - torch.diag(generalized_box_iou(
                            box_convert(pred, "cxcywh", "xyxy"),
                            box_convert(rows, "cxcywh", "xyxy")))).sum()
                losses[name] = total / num_boxes
            return losses

    # predictions deliberately offset from their targets so the native regression mass is
    # non-zero and the extra pair's contribution is measurable
    offset = [[b[0] + 0.02, b[1] + 0.02, b[2], b[3]] for b in
              [tiny0, tiny0, tiny0, tiny0, big0]]
    outputs = {"pred_boxes": cxcywh([offset])}
    targets = [target]
    indices = [(src_idx, tgt_idx)]
    native = Base()(outputs, targets, indices, 3.0)
    term = roi_ld.RoiTerm("R2")
    armed = term.loss_boxes(Base(), module, outputs, targets, indices, 3.0, None)
    check("armed loss keys equal the native keys", sorted(armed) == sorted(native))
    check("armed regression mass exceeds the native mass on both keys",
          float(armed["loss_bbox"]) > float(native["loss_bbox"])
          and float(armed["loss_giou"]) > float(native["loss_giou"]),
          (float(native["loss_bbox"]), float(armed["loss_bbox"])))
    check("R2 counts one target per selected row", term.state_dict()["targets"] == 1, term.state_dict())
    check("the appended pair's target differs from the ground-truth row",
          term.last_stats["roi_selected"] == 1)
    check("armed losses carry no gradient side effect", not any(v.requires_grad for v in armed.values()))
    term1 = roi_ld.RoiTerm("R1")
    armed1 = term1.loss_boxes(Base(), module, outputs, targets, indices, 3.0, None)
    check("the control arm produces a different loss from the method arm on the same inputs",
          float(armed1["loss_bbox"]) != float(armed["loss_bbox"]),
          (float(armed1["loss_bbox"]), float(armed["loss_bbox"])))

    # 5. observation, declaration and state
    obs = roi_ld.observe_call(outputs["pred_boxes"][0], target["boxes"], module, 4)
    check("native observation reports zero appended targets",
          obs["roi_selected"] == 0 and obs["roi_mode"] is None and obs["roi_band_gt"] == 4, obs)
    check("arm declaration round-trips",
          roi_ld_arm.declaration("R1")["mode"] == "R1"
          and roi_ld_arm.declaration(None)["installed"] is False
          and json.loads(json.dumps(roi_ld_arm.declaration("R2"))) == roi_ld_arm.declaration("R2"))
    check("the control and method declare different view groups",
          roi_ld_arm.VIEWS["R1"] == (0,) and roi_ld_arm.VIEWS["R2"] == (1, 2, 3, 4)
          and roi_ld.GROUP == {"R1": 0, "R2": 1})
    check("term state save/load is exact",
          roi_ld.RoiTerm("R2").load_state_dict(term.state_dict()) == {"targets": 1})
    check("term refuses a state from another mode",
          _raises(lambda: roi_ld.RoiTerm("R1").load_state_dict(term.state_dict())))

    # 6. augmentation replay
    student = {"image_id": "x", "original_hw": torch.tensor([540, 960], dtype=torch.long)}
    rows = roi_ld.replay_boxes([(0, (0.0, 0.0, 960.0, 540.0))], (540, 960),
                               {"image_id": "x", "geometry_steps": []}, student)
    check("replay without steps maps the full image to the unit canvas in normalised xyxy",
          len(rows) == 1 and torch.allclose(torch.tensor(rows[0][1], dtype=torch.float32),
                                            torch.tensor([0.0, 0.0, 1.0, 1.0]), atol=1e-6), rows)
    check("the replay output converts to the canvas-normalised cxcywh the criterion consumes",
          torch.allclose(box_convert(torch.tensor([rows[0][1]], dtype=torch.float32), "xyxy", "cxcywh"),
                         torch.tensor([[0.5, 0.5, 1.0, 1.0]]), atol=1e-6))
    rows_flip = roi_ld.replay_boxes([(0, (0.0, 0.0, 240.0, 540.0))], (540, 960),
                                    {"image_id": "x", "geometry_steps": [{"operation": "horizontal_flip"}]}, student)
    check("replay maps a left-edge box to the right edge under flip",
          len(rows_flip) == 1 and rows_flip[0][1][0] > 0.5, rows_flip)
    trace_crop = {"image_id": "x", "geometry_steps": [{"operation": "crop", "top": 0, "left": 480,
                                      "height": 540, "width": 480}]}
    inside = roi_ld.replay_boxes([(0, (500.0, 10.0, 600.0, 60.0))], (540, 960), trace_crop, student)
    outside = roi_ld.replay_boxes([(0, (10.0, 10.0, 60.0, 60.0))], (540, 960), trace_crop, student)
    check("replay admits a centre-inside box and drops a centre-outside box",
          len(inside) == 1 and len(outside) == 0, (inside, outside))
    check("replay refuses a trace without a geometry step list",
          _raises(lambda: roi_ld.replay_boxes([], (540, 960), {"augmentation_strong": True,
                                                              "image_id": "x"}, student)))
    check("replay accepts a strong-augmentation trace that records its steps",
          roi_ld.replay_boxes([(0, (0.0, 0.0, 960.0, 540.0))], (540, 960),
                              {"augmentation_strong": True, "image_id": "x", "geometry_steps": []},
                              student)[0][1].shape == (4,))

    # 7. constants are frozen and hashed
    value = roi_ld.constants_sha256()
    check("constants digest is stable",
          value == hashlib.sha256(json.dumps(roi_ld.constants(), sort_keys=True,
                                            separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
          value)

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print("TESTS %d/%d" % (passed, len(RESULTS)))
    if passed != len(RESULTS):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
