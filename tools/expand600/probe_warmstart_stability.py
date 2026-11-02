"""CPU prototype: does a warm-started second decode give a usable per-candidate quality signal?

Why this mechanism.  The offline decomposition shows the certified endpoint's small band carries about
ten thousand matches that hold at IoU 0.50 but not 0.75, and the r20 plan already measured the box-only
headroom at about 22 pp of AP75_small, so tight localization is the binding constraint.  The same
decomposition also shows the loose boxes are spread on both sides of the ground truth (area ratio p10
0.73, p90 1.61), so the error is per object, not a uniform scale bias, and no GT-free transform of the
existing boxes can repair it.  What is missing is a GT-free signal that says which of several candidate
boxes for the same object is the trustworthy one.

The mechanism under test.  The certified pass refines each query from a reference built out of the
encoder anchors.  A second pass uses the SAME decoder and heads but warm-starts the reference from the
first pass's own output boxes, so the model refines its own answer instead of the anchor prior.  Two
readings follow, and both are measurable with existing code, no GT in the method and no GPU:

  (a) stability   the IoU between the two passes' boxes for the same query, a per-candidate signal;
  (b) payoff      whether the second pass is, per candidate, a better box against the ground truth.

Ground truth is used ONLY to evaluate the readings offline; the method itself never sees it.

The control that can refute the explanation: replace the warm-start reference by a shuffled reference
from another query of the same image.  If the payoff and the stability ordering survive the shuffle,
then the effect is extra computation and not self-refinement, and the explanation is refuted.

usage: probe_warmstart_stability.py <release-source> <contract> <checkpoint> <source-manifest> <out-json> [images]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import frozen_r18                                                       # noqa: E402
from tools.expand600.expand600 import (NUM_QUERIES, capture_selection, find_decoder_module,
                                       gather_rows, select_indices)     # noqa: E402

SMALL_AREA = 1024.0


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def cxcywh_to_xyxy(boxes):
    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2), -1)


def iou_pair(boxes_a, boxes_b):
    """IoU between two (N,4) pixel xyxy tensors."""
    lt = torch.maximum(boxes_a[:, :2], boxes_b[:, :2])
    rb = torch.minimum(boxes_a[:, 2:], boxes_b[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]).clamp(min=0) * (boxes_a[:, 3] - boxes_a[:, 1]).clamp(min=0)
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]).clamp(min=0) * (boxes_b[:, 3] - boxes_b[:, 1]).clamp(min=0)
    return inter / (area_a + area_b - inter).clamp(min=1e-12)


def second_pass(decoder, captured, indices, reference_unact):
    stack = decoder.decoder
    content = gather_rows(captured["output_memory"], indices).detach()
    reference = gather_rows(reference_unact, indices).detach()
    boxes, logits = stack(content, reference, captured["memory"], captured["spatial_shapes"],
                          decoder.dec_bbox_head, decoder.dec_score_head, decoder.query_pos_head)
    return boxes[-1].float(), logits[-1].float()


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

    stability, iou_pass1, iou_pass2, shuffled_iou, areas = [], [], [], [], []
    per_image = []
    with torch.no_grad():
        for position, row in enumerate(rows):
            pixels, _target = frozen_r18.image_pixels_and_target(dev, row, position)
            with capture_selection(decoder) as captured:
                output = model(pixels.unsqueeze(0))
            require(captured.get("recomputed_matches_production") is True,
                    "capture does not reproduce the production gather")
            ind300 = select_indices(captured["enc_logits"], NUM_QUERIES)
            # pass one output, expressed as the reference of pass two
            boxes1 = output["pred_boxes"][0].float()
            scores1 = output["pred_logits"][0].float().sigmoid()
            xyxy1 = cxcywh_to_xyxy(boxes1) * torch.tensor([row["width"], row["height"]] * 2)
            ref_unact = boxes1.clone()
            boxes2, logits2 = second_pass(decoder, captured, ind300, ref_unact)
            # the shuffled control keeps the same compute but takes the reference from another query
            permutation = torch.roll(torch.arange(NUM_QUERIES), shifts=NUM_QUERIES // 2, dims=0)
            shuffled = boxes1.clone()
            shuffled[permutation] = boxes1.clone()
            boxes_shuffled, _ = second_pass(decoder, captured, ind300, shuffled)
            xyxy2 = cxcywh_to_xyxy(boxes2[0]) * torch.tensor([row["width"], row["height"]] * 2)
            xyxy_shuffled = cxcywh_to_xyxy(boxes_shuffled[0]) * torch.tensor(
                [row["width"], row["height"]] * 2)

            raw_rows = dev.raw(row)["rows"]
            gt_boxes = [g["box"] for g in raw_rows if int(g["category"]) in range(1, 11)]
            gt_areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in gt_boxes]
            for query in range(NUM_QUERIES):
                best_class = int(scores1[query].argmax())
                category = best_class + 1
                same = [gt_boxes[i] for i, g in enumerate(raw_rows)
                        if int(g["category"]) in range(1, 11) and int(g["category"]) == category]
                if not same:
                    continue
                box1 = xyxy1[query]
                box2 = xyxy2[query]
                box_s = xyxy_shuffled[query]
                area1 = float(max(box1[2] - box1[0], 0) * max(box1[3] - box1[1], 0))
                if area1 >= SMALL_AREA:
                    continue
                ref = torch.tensor(same, dtype=torch.float32)
                i1 = float(iou_pair(box1.unsqueeze(0), ref).max())
                i2 = float(iou_pair(box2.unsqueeze(0), ref).max())
                is_ = float(iou_pair(box_s.unsqueeze(0), ref).max())
                stability.append(float(iou_pair(box1.unsqueeze(0), box2.unsqueeze(0))[0]))
                iou_pass1.append(i1)
                iou_pass2.append(i2)
                shuffled_iou.append(is_)
                areas.append(area1)
            per_image.append({"image_id": row["stable_image_id"], "queries": NUM_QUERIES})
            print("image %d/%d %s stability %d samples" % (position + 1, len(rows),
                                                           row["stable_image_id"][:12], len(stability)),
                  flush=True)

    stability = np.asarray(stability, dtype=np.float64)
    i1 = np.asarray(iou_pass1, dtype=np.float64)
    i2 = np.asarray(iou_pass2, dtype=np.float64)
    is_ = np.asarray(shuffled_iou, dtype=np.float64)
    report = {"images": len(rows), "small_detections": int(len(stability)),
              "protocol": {"matching": "per query, best IoU against same-class raw annotations",
                           "ground_truth_use": "offline analysis only; the method never sees it"},
              "stability": summary(stability.tolist()),
              "iou_pass1": summary(i1.tolist()), "iou_pass2": summary(i2.tolist()),
              "iou_shuffled": summary(is_.tolist()),
              "pass2_minus_pass1": summary((i2 - i1).tolist()),
              "shuffled_minus_pass1": summary((is_ - i1).tolist())}
    for name, low, high in (("unstable_third", np.percentile(stability, 33),
                             np.percentile(stability, 67)),):
        mask_low = stability <= low
        mask_high = stability >= high
        report["by_stability"] = {
            name: {"low_stability_mean_iou_pass1": float(i1[mask_low].mean()) if mask_low.any() else None,
                   "low_stability_mean_iou_pass2": float(i2[mask_low].mean()) if mask_low.any() else None,
                   "high_stability_mean_iou_pass1": float(i1[mask_high].mean()) if mask_high.any() else None,
                   "high_stability_mean_iou_pass2": float(i2[mask_high].mean()) if mask_high.any() else None,
                   "low_count": int(mask_low.sum()), "high_count": int(mask_high.sum())}}
    report["improved_by_pass2"] = int((i2 > i1 + 1e-6).sum())
    report["worsened_by_pass2"] = int((i2 < i1 - 1e-6).sum())
    report["improved_by_shuffled"] = int((is_ > i1 + 1e-6).sum())
    report["worsened_by_shuffled"] = int((is_ < i1 - 1e-6).sum())
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False)[:4000])


if __name__ == "__main__":
    main()
