"""CPU prototype: does a mirrored second view supply the independent evidence the endpoint lacks?

Where this sits.  Two offline refutations came out of the r22 follow-on.  Warm-starting the decoder on
its own output is a wash (stability mean 0.910 with pass two within -1.5e-4 mean IoU of pass one) and the
stability signal does not separate the improvements from the degradations (improved 0.9189 vs worsened
0.9264).  Fusing the boxes of mutually-agreeing same-class candidates is uniformly harmful and grows
worse as the clusters grow (AP75_small -0.58 pp for the mildest rule at IoU 0.7, and the fused box is on
average 0.0086 IoU further from the object than the cluster's seed box).  Both say the same thing: the
endpoint's own evidence, at the anchor's own resolution, is exhausted, because its box errors are
per-object and correlated, not zero-mean noise to be averaged away.

The mechanism under test.  Run the same frozen weights once more on the horizontally mirrored canvas and
map the boxes back (cx -> 1 - cx; the evaluation transform is resize-only, which frozen_r18 itself
asserts, so the mirror is exact in normalised canvas coordinates).  For each certified query take its
best same-class partner in the mirrored pass by box agreement, and replace the box by the plain mean of
the two when they agree at IoU >= 0.5.  Nothing else changes: no new names, no re-ranking, no training,
one fixed threshold.  The mirrored pass samples a different pixel grid, so if the two views' errors are
partly independent, averaging should move the box toward the object; if they are as correlated as
within-view candidates were, it will not.

Readings, all offline, on the eight-image CPU sample: the partner-agreement distribution (which also
validates the mirror mapping: a wrong mapping would put it at zero), and, for every object the original
box already overlaps at 0.50, the change in IoU against the certified region-filtered annotations.

The control that can refute the explanation: keep the same coverage (fuse exactly the queries whose best
partner agrees at 0.50) but fuse with the SECOND-best partner, i.e. a deliberately wrong object.  If the
payoff survives that, then what helps is smoothing and not a second independent view.

usage: probe_flip_view.py <release-source> <contract> <checkpoint> <source-manifest> <out-json> [images]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
for _path in (str(HERE), str(HERE.parent / "explore"), str(HERE.parents[1])):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import frozen_r18                                                       # noqa: E402
from tools.expand600.expand600 import (NUM_QUERIES, capture_selection, find_decoder_module,
                                       select_indices, xyxy_iou_matrix)  # noqa: E402

SMALL_AREA = 1024.0
PARTNER_IOU = 0.5
GEOMETRIC_HITS_FLOOR = 20


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def cxcywh_to_xyxy(boxes):
    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2), -1)


def to_pixels(boxes, width, height):
    return cxcywh_to_xyxy(boxes) * torch.tensor([width, height] * 2)


def iou_pair(boxes_a, boxes_b):
    lt = torch.maximum(boxes_a[:, :2], boxes_b[:, :2])
    rb = torch.minimum(boxes_a[:, 2:], boxes_b[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]).clamp(min=0) * (boxes_a[:, 3] - boxes_a[:, 1]).clamp(min=0)
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]).clamp(min=0) * (boxes_b[:, 3] - boxes_b[:, 1]).clamp(min=0)
    return inter / (area_a + area_b - inter).clamp(min=1e-12)


def summary(values):
    values = [float(v) for v in values if v == v]
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {"count": len(values), "mean": float(array.mean()), "median": float(np.median(array)),
            "p10": float(np.percentile(array, 10)), "p90": float(np.percentile(array, 90))}


def main():
    source, contract_path, checkpoint, manifest, out_path = sys.argv[1:6]
    limit = int(sys.argv[6]) if len(sys.argv) > 6 else 8
    loaded = frozen_r18.load_frozen_cpu(source, contract_path, checkpoint, manifest)
    model = loaded["model"]
    contract = loaded["contract"]
    dev = frozen_r18.development548(contract)
    rows = dev.records[:limit]
    require(len(rows) >= 1, "no images selected")
    decoder = find_decoder_module(model)
    torch.set_num_threads(8)

    partner_iou, treated, delta_treat, delta_control, mirror_only = [], [], [], [], []
    better, worse = 0, 0
    control_better, control_worse = 0, 0
    with torch.no_grad():
        for position, row in enumerate(rows):
            pixels, _target = frozen_r18.image_pixels_and_target(dev, row, position)
            require(pixels.dim() == 3, "expected a single (C, H, W) canvas")
            width, height = row["width"], row["height"]
            with capture_selection(decoder) as captured:
                output = model(pixels.unsqueeze(0))
            require(captured.get("recomputed_matches_production") is True,
                    "capture does not reproduce the production gather")
            mirrored = model(pixels.flip(-1).unsqueeze(0))
            boxes1 = output["pred_boxes"][0].float()
            scores1 = output["pred_logits"][0].float().sigmoid()
            boxes2 = mirrored["pred_boxes"][0].float().clone()
            boxes2[:, 0] = 1.0 - boxes2[:, 0]                      # mirror back in normalised canvas
            xyxy1 = to_pixels(boxes1, width, height)
            xyxy2 = to_pixels(boxes2, width, height)
            require(bool(torch.isfinite(xyxy2).all()), "the mirrored pass produced a non-finite box")
            # query-by-query agreement: a full (300, 300) matrix, not the element-wise IoU
            agreement = torch.as_tensor(xyxy_iou_matrix(xyxy1, xyxy2), dtype=torch.float32)

            raw_rows = dev.raw(row)["rows"]
            for query in range(NUM_QUERIES):
                category = int(scores1[query].argmax()) + 1
                same = [[float(v) for v in g["box"]] for g in raw_rows
                        if int(g["category"]) == category]
                if not same:
                    continue
                box1 = xyxy1[query]
                area1 = float(max(box1[2] - box1[0], 0.0) * max(box1[3] - box1[1], 0.0))
                if area1 >= SMALL_AREA:
                    continue
                values = agreement[query]
                order = torch.argsort(values, descending=True)
                best, second = int(order[0]), int(order[1])
                best_iou = float(values[best])
                if best_iou < PARTNER_IOU:
                    continue
                partner_iou.append(best_iou)
                fused = (box1 + xyxy2[best]) / 2.0
                swapped = (box1 + xyxy2[second]) / 2.0
                ref = torch.tensor(same, dtype=torch.float32)
                gt_iou = float(iou_pair(box1.unsqueeze(0), ref).max())
                treat_iou = float(iou_pair(fused.unsqueeze(0), ref).max())
                control_iou = float(iou_pair(swapped.unsqueeze(0), ref).max())
                mirror_iou = float(iou_pair(xyxy2[best].unsqueeze(0), ref).max())
                treated.append(gt_iou)
                delta_treat.append(treat_iou - gt_iou)
                delta_control.append(control_iou - gt_iou)
                mirror_only.append(mirror_iou - gt_iou)
                better += treat_iou > gt_iou + 1e-6
                worse += treat_iou < gt_iou - 1e-6
                control_better += control_iou > gt_iou + 1e-6
                control_worse += control_iou < gt_iou - 1e-6
            print("image %d/%d %s treated %d" % (position + 1, len(rows),
                                                 row["stable_image_id"][:12], len(treated)), flush=True)

    hits = sum(1 for value in treated if value >= 0.5)
    require(hits >= GEOMETRIC_HITS_FLOOR,
            "only %d treated boxes overlap their annotation at 0.50: box space or identity is wrong"
            % hits)
    report = {
        "images": len(rows), "treated_names": len(treated), "geometric_hits_at_050": hits,
        "protocol": {
            "second_view": "same frozen weights on the horizontally mirrored resize-only canvas; "
                           "boxes mirrored back by cx -> 1 - cx",
            "pairing": "for each certified query, the mirrored query with the highest box IoU, fused "
                       "only when that IoU reaches %.2f" % PARTNER_IOU,
            "fusion": "plain mean of the original and the partner box on the original query",
            "control": "identical coverage, partner replaced by the second-best (a different object)",
            "matching": "per query, best IoU against the frozen raw annotations of the argmax class",
            "ground_truth_use": "geometric-overlap diagnostic only; not certified matching; never used "
                                "to pair, to fuse or to select",
            "small_band": "original box pixel area < %g (original image pixels)" % SMALL_AREA},
        "partner_iou": summary(partner_iou),
        "original_iou": summary(treated),
        "delta_iou_treatment": summary(delta_treat),
        "delta_iou_control": summary(delta_control),
        "delta_iou_mirror_view_alone": summary(mirror_only),
        "treatment_better": int(better), "treatment_worse": int(worse),
        "control_better": int(control_better), "control_worse": int(control_worse)}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False)[:4000])


if __name__ == "__main__":
    main()
