"""Read the frozen whole-image teacher cache and attach one ROI box per qualifying row.

The cache is the closed distillation round's own teacher store: per train_core member it holds
the frozen parent's five views (view 0 whole image, views 1-4 the four raw-pixel tiles) as
cross-class top-600 candidates in ORIGINAL image coordinates, bound by an index that this
module re-verifies byte by byte before any read.

R1 takes view 0 only; R2 takes views 1-4.  Everything else - the window geometry, the
selection rule, the bars, the counts, the keys and the normaliser - is identical, so the two
armed arms differ only in whether the pixels the teacher re-encoded can carry information the
student's own canvas does not already have.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torchvision.ops import box_convert

from . import roi_ld
from .roi_ld_arm import VIEWS

# Every view of the frozen cache.  Both armed arms attach boxes from all of them and tag each
# box with its group (0 = whole-image view, 1 = raw-pixel tiles), so the two arms read the same
# inputs and differ only in the declared group the criterion targets.
ALL_VIEWS = (0, 1, 2, 3, 4)

# Pre-filter in original coordinates only, to bound the replay work; the admission bar that
# decides the target set is TARGET_IOU, applied in the student's augmented frame.
PREFILTER_IOU = 0.1
EXPECTED_SHAPE = (300, 10)

# The frozen whole-image teacher store of the closed distillation round.  The index is
# re-verified byte by byte on every construction; the binding must equal this dict exactly.
FROZEN_INDEX = {
    "path": ("/media/lyy/Data/JupyterLab/LiuZhiyang/SparseRTDETR_Workspace/experiments/"
             "localization-distribution/r18-multiview-distill-20261020-001/attempt-gpu-002/"
             "teacher/index.json"),
    "sha256": "fbb24c2ac4fd382eeb1d902aad3bf18fb088aeccfb4f812aa1cb8b11d5d0f2e3",
    "bytes": 2386443,
}
FROZEN_BINDING = {
    "protocol": "r18-multiview-positive-kd-v1", "seed": 0, "role": "train_core",
    "manifest_sha256": "df87e6a377a5546d3d3fe0cd0e860d15c2fe5e8003ea87d887153579999a4e9d",
    "checkpoint_sha256": "13a35283a02d2b728e3449917586ded5cdf2d578ebe6847510748e0dc4d21757",
    "source_manifest_sha256": "1667699cdfb1814dad4ee44b3e9db7440cf4c9dbed98b71c26f1fc264ba75d38",
    "contract_sha256": "53529ccb4899b689457a38e00066defa3527f58e333d58c3b9abf0e452c164b0",
    "EMA": True, "epoch": 120, "input_hw": [768, 1344], "inference_dtype": "FP32",
}


def frozen_cache():
    """The authenticated frozen teacher store; raises if any member byte has changed."""
    return TeacherCache(FROZEN_INDEX, FROZEN_BINDING)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_bound(path, expected_sha, expected_bytes):
    raw = Path(path).read_bytes()
    require(len(raw) == expected_bytes and hashlib.sha256(raw).hexdigest() == expected_sha,
            "bound reference changed: " + str(path))
    return raw


def box_iou(a, b):
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1])
    x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area = lambda q: max(0.0, q[2] - q[0]) * max(0.0, q[3] - q[1])
    union = area(a) + area(b) - inter
    return 0.0 if union <= 0 else inter / union


class TeacherCache:
    """One authenticated teacher index; member records are read on demand and re-verified."""

    def __init__(self, index_reference, binding):
        raw = read_bound(index_reference["path"], index_reference["sha256"], index_reference["bytes"])
        index = json.loads(raw)
        require(index.get("closed") is True, "teacher index is not closed")
        require(index.get("binding") == binding, "teacher index binding differs")
        records = index["records"]
        require(len(records) == len({r["image_id"] for r in records}), "teacher index repeats a member")
        self.binding = binding
        self.records = {r["image_id"]: r["record"] for r in records}
        self.index_sha256 = index_reference["sha256"]
        self.index_bytes = index_reference["bytes"]

    def __len__(self):
        return len(self.records)

    def lookup(self, image_id):
        reference = self.records.get(image_id)
        require(reference is not None, "teacher cache has no member " + str(image_id))
        raw = read_bound(reference["path"], reference["sha256"], reference["bytes"])
        record = json.loads(raw)
        require(record.get("binding") == self.binding, "teacher member binding differs")
        return record


def candidates_for(record, mode="ALL"):
    """Per-instance candidate boxes, in original coordinates, of one view selection.

    ``mode`` selects an armed arm's declared view set; ``"ALL"`` selects every view, which is
    what the attachment uses so that both groups are present for every row.
    """
    allowed = set(ALL_VIEWS) if mode == "ALL" else set(VIEWS[mode])
    rows = []
    for view in record["views"]:
        if view["view_id"] not in allowed:
            continue
        for candidate in view["candidates"]:
            rows.append((int(view["view_id"]), int(candidate["category_id"]),
                         tuple(float(v) for v in candidate["box_xyxy"]),
                         int(candidate["query_id"]), float(candidate["score"])))
    return rows


def attach(target, trace, raw, row, mode, cache):
    """Attach both view groups' ROI boxes for one image; no-op for the native arm.

    For every ground-truth row, per view group, the best same-class candidate by original-
    coordinate overlap that clears PREFILTER_IOU is kept; group 0 is the whole-image view and
    group 1 is the four raw-pixel tiles.  Both armed modes attach exactly these boxes, so the
    two arms read identical inputs and differ only in the declared group the criterion uses.
    Each kept box is replayed through the student's own recorded augmentation geometry and
    attached to the retained row slot carrying the same stable ground-truth id; rows the
    augmentation dropped are dropped here too.
    """
    if mode is None:
        return target
    require(mode in VIEWS, "unknown roi-ld mode: " + str(mode))
    record = cache.lookup(row["stable_image_id"])
    require(record["member"]["stable_image_id"] == row["stable_image_id"], "teacher member differs")
    height, width = int(row["height"]), int(row["width"])
    canvas_area = roi_ld.BAND_AREA_PX
    scale = (roi_ld.CANVAS_HW[0] / height) * (roi_ld.CANVAS_HW[1] / width)
    candidates = candidates_for(record, "ALL")  # every view; the group tag splits them
    by_group = {}
    for view_id, category_id, box, query_id, score in candidates:
        by_group.setdefault(0 if view_id == 0 else 1, []).append(
            (category_id, box, query_id, view_id))
    raw_rows = []
    for gt in raw["gt"]:
        box = tuple(float(v) for v in gt["box"])
        if max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]) * scale > canvas_area:
            continue
        # gt["id"] is the stable string row identity used by target["gt_ids"]; never cast it.
        raw_rows.append((gt["id"], int(gt["category"]) - 1, box))

    # End-to-end proof that the replay reproduces the recorded augmentation: replay the
    # ground-truth rows themselves and compare with the augmented target rows that carry the
    # same stable id.  A non-zero residual or a missing row means the replay is not faithful,
    # and the caller is expected to refuse to train on it.
    all_rows = []
    for gt in raw["gt"]:
        box = tuple(float(v) for v in gt["box"])
        if max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]) <= 0:
            continue
        all_rows.append((gt["id"], int(gt["category"]) - 1, box))
    slot_of = {gt_id: j for j, gt_id in enumerate(target["gt_ids"])}
    max_error, compared = 0.0, 0
    seen = set()
    for index, box in roi_ld.replay_boxes([(i, b) for i, (_, _, b) in enumerate(all_rows)],
                                          (height, width), trace, target):
        gt_id = all_rows[index][0]
        slot = slot_of.get(gt_id)
        if slot is None:
            continue
        got = box_convert(torch.tensor([box], dtype=torch.float32), "xyxy", "cxcywh")[0]
        max_error = max(max_error, float((got - target["boxes"][slot]).abs().max().item()))
        compared += 1
        seen.add(gt_id)
    target["roi_replay_max_error"] = torch.tensor(max_error, dtype=torch.float64)
    target["roi_replay_compared"] = torch.tensor(compared, dtype=torch.long)
    target["roi_replay_missing"] = torch.tensor(len(target["gt_ids"]) - len(seen), dtype=torch.long)

    picked = []
    for gt_id, label, box in raw_rows:
        for group, pool in by_group.items():
            best = None
            for category_id, candidate, query_id, view_id in pool:
                if category_id - 1 != label:
                    continue
                value = box_iou(box, candidate)
                if value < PREFILTER_IOU:
                    continue
                key = (-value, view_id, query_id)
                if best is None or key < best[0]:
                    best = (key, candidate)
            if best is not None:
                picked.append((gt_id, label, group, best[1]))
    if not picked:
        target["roi_boxes"] = torch.zeros((0, 4), dtype=torch.float32)
        target["roi_rows"] = torch.zeros(0, dtype=torch.long)
        target["roi_labels"] = torch.zeros(0, dtype=torch.long)
        target["roi_group"] = torch.zeros(0, dtype=torch.long)
        target["roi_ids"] = []
        target["roi_mode"] = mode
        target["roi_group_declared"] = roi_ld.GROUP[mode]
        return target
    aligned = roi_ld.replay_boxes([(index, box) for index, (_, _, _, box) in enumerate(picked)],
                                  (height, width), trace, target)
    boxes, rows, labels, groups, ids = [], [], [], [], []
    for index, box in aligned:
        gt_id, label, group, _ = picked[index]
        slot = slot_of.get(gt_id)
        if slot is None:
            continue
        boxes.append(box)
        rows.append(slot)
        labels.append(label)
        groups.append(group)
        ids.append(gt_id)
    xyxy = torch.tensor(np.asarray(boxes, dtype=np.float64).reshape(-1, 4), dtype=torch.float32)
    # replay_boxes returns canvas-normalised xyxy (the frozen replay arithmetic); the criterion
    # consumes cxcywh, so convert once here and nowhere else.
    target["roi_boxes"] = box_convert(xyxy, "xyxy", "cxcywh")
    target["roi_rows"] = torch.tensor(rows, dtype=torch.long)
    target["roi_labels"] = torch.tensor(labels, dtype=torch.long)
    target["roi_group"] = torch.tensor(groups, dtype=torch.long)
    target["roi_ids"] = ids
    target["roi_mode"] = mode
    target["roi_group_declared"] = roi_ld.GROUP[mode]
    return target


def coverage(targets, mode):
    """Value-only coverage readout: attached boxes, band rows, rows carrying a box."""
    attached = sum(int(t["roi_boxes"].shape[0]) for t in targets if t.get("roi_boxes") is not None)
    band = roi_ld.band_gt(targets)
    carried = 0
    for t in targets:
        boxes = t.get("roi_boxes")
        if boxes is None or boxes.numel() == 0:
            continue
        band_rows = set(torch.nonzero(roi_ld.band_mask(t["boxes"]), as_tuple=False).flatten().tolist())
        carried += len(band_rows & set(int(v) for v in t["roi_rows"].tolist()))
    return {"mode": mode, "attached_boxes": attached, "band_gt": band, "band_rows_carrying": carried}
