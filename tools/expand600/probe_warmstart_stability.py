"""CPU prototype: does a warm-started second decode give a usable per-candidate quality signal?

Why this mechanism.  The offline decomposition shows the certified endpoint's small band carries about
ten thousand matches that hold at IoU 0.50 but not 0.75, and the r20 plan already measured the box-only
headroom at about 22 pp of AP75_small, so tight localization is the binding constraint.  The same
decomposition also shows the loose boxes are spread on both sides of the ground truth (area ratio p10
0.73, p90 1.61), so the error is per object, not a uniform scale bias, and no GT-free transform of the
existing boxes can repair it.  What is missing is a GT-free signal that says which of several candidate
boxes for the same object is the trustworthy one.

The mechanism under test, and the space it lives in (read off the vendor decoder, not guessed).
`TransformerDecoder.forward` receives `ref_points_unact`, sets `ref_points_detach = sigmoid(
ref_points_unact)` as the attention reference, and each layer returns `sigmoid(bbox_head[i](output) +
inverse_sigmoid(ref_points)))`, so the model's own `pred_boxes` are normalised cxcywh in sigmoid space
and the decoder's argument is their pre-sigmoid form.  A second pass therefore runs the SAME decoder and
heads with the SAME certified content, but warm-starts the reference at the first pass's own answer,
`ref_points_unact = inverse_sigmoid(pred_boxes_pass1)`: layer 0 begins exactly where pass one finished
and the model refines itself instead of the encoder-anchor prior.  Two readings follow, both measurable
offline on CPU with no ground truth inside the method:

  (a) stability   the IoU between the two passes' boxes for the same query, a per-candidate signal;
  (b) payoff      whether the second pass is, per candidate, a better box against the annotations.

Ground truth enters ONLY the offline reading.  The reference boxes here are the frozen raw annotations
of the same class, so every IoU in this report is a geometric-overlap diagnostic, not certified
matching: no category-0 region-ignore filter, no per-class ranking, no maxDets.

The control that can refute the explanation: replace the warm-start reference by a reference taken from
another query of the same image (a roll by half the query count), keeping the exact same compute.  If
the payoff and the stability ordering survive the shuffle, then what helps is extra computation and not
self-refinement, and the explanation is refuted.

Defects of attempt 001, both mine and both fixed here (scene kept in WARMSTART_STABILITY.failure-001.log):
the reference carried no batch axis, so `gather_rows` refused it (the decoder always works batched); and
the reference was gathered by anchor index although it is already one row per selected query.

usage: probe_warmstart_stability.py <release-source> <contract> <checkpoint> <source-manifest> <out-json> [images]
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
                                       gather_rows, select_indices)     # noqa: E402

SMALL_AREA = 1024.0
GEOMETRIC_HITS_FLOOR = 20
CHANGE_TOLERANCE = 1e-6


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def load_inverse_sigmoid():
    """The vendor's own `inverse_sigmoid` when the release source exposes it, else the same formula."""
    for name in ("zoo.rtdetr.utils", "vendor.rtdetrv2_pytorch.src.zoo.rtdetr.utils"):
        try:
            module = __import__(name, fromlist=["inverse_sigmoid"])
            return module.inverse_sigmoid, name + ".inverse_sigmoid"
        except Exception:
            continue

    def fallback(x):
        x = x.clamp(min=0.0, max=1.0)
        return torch.log(x.clamp(min=1e-5) / (1 - x).clamp(min=1e-5))

    return fallback, "inline copy of vendor inverse_sigmoid (clip 0..1, eps 1e-5)"


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
    """One extra decoder pass: certified content for `indices`, caller-supplied warm reference.

    The content comes from the encoder pool gathered by anchor index; the reference does NOT, because it
    is already one row per selected query.  Both keep the leading batch axis, the shape the vendor
    `TransformerDecoder.forward` consumes.
    """
    stack = decoder.decoder
    content = gather_rows(captured["output_memory"], indices).detach()
    reference = reference_unact.detach()
    require(tuple(reference.shape) == (int(content.shape[0]), int(content.shape[1]), 4),
            "reference %s does not match content %s" % (tuple(reference.shape), tuple(content.shape)))
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
    inverse_sigmoid, inverse_source = load_inverse_sigmoid()
    loaded = frozen_r18.load_frozen_cpu(source, contract_path, checkpoint, manifest)
    model = loaded["model"]
    contract = loaded["contract"]
    dev = frozen_r18.development548(contract)
    rows = dev.records[:limit]
    require(len(rows) >= 1, "no images selected")
    decoder = find_decoder_module(model)
    torch.set_num_threads(8)

    stability, iou_pass1, iou_pass2, shuffled_iou = [], [], [], []
    with torch.no_grad():
        for position, row in enumerate(rows):
            pixels, _target = frozen_r18.image_pixels_and_target(dev, row, position)
            with capture_selection(decoder) as captured:
                output = model(pixels.unsqueeze(0))
            require(captured.get("recomputed_matches_production") is True,
                    "capture does not reproduce the production gather")
            ind300 = select_indices(captured["enc_logits"], NUM_QUERIES)
            boxes1 = output["pred_boxes"][0].float()            # normalised cxcywh, top-k query order
            scores1 = output["pred_logits"][0].float().sigmoid()
            scale = torch.tensor([row["width"], row["height"]] * 2)
            xyxy1 = cxcywh_to_xyxy(boxes1) * scale
            # warm start: layer 0 of pass two starts exactly where pass one finished
            boxes2, _logits2 = second_pass(decoder, captured, ind300,
                                           inverse_sigmoid(boxes1).unsqueeze(0))
            # shuffled control: identical compute, reference taken from another query of the same image
            permutation = torch.roll(torch.arange(NUM_QUERIES), shifts=NUM_QUERIES // 2, dims=0)
            boxes_shuffled, _ = second_pass(decoder, captured, ind300,
                                            inverse_sigmoid(boxes1[permutation]).unsqueeze(0))
            require(bool(torch.isfinite(boxes2).all()) and bool(torch.isfinite(boxes_shuffled).all()),
                    "second pass produced a non-finite box")
            xyxy2 = cxcywh_to_xyxy(boxes2[0]) * scale
            xyxy_shuffled = cxcywh_to_xyxy(boxes_shuffled[0]) * scale

            raw_rows = dev.raw(row)["rows"]
            for query in range(NUM_QUERIES):
                category = int(scores1[query].argmax()) + 1
                same = [[float(v) for v in g["box"]] for g in raw_rows
                        if int(g["category"]) == category]
                if not same:
                    continue
                box1, box2, box_s = xyxy1[query], xyxy2[query], xyxy_shuffled[query]
                area1 = float(max(box1[2] - box1[0], 0.0) * max(box1[3] - box1[1], 0.0))
                if area1 >= SMALL_AREA:
                    continue
                ref = torch.tensor(same, dtype=torch.float32)
                iou_pass1.append(float(iou_pair(box1.unsqueeze(0), ref).max()))
                iou_pass2.append(float(iou_pair(box2.unsqueeze(0), ref).max()))
                shuffled_iou.append(float(iou_pair(box_s.unsqueeze(0), ref).max()))
                stability.append(float(iou_pair(box1.unsqueeze(0), box2.unsqueeze(0))[0]))
            print("image %d/%d %s samples %d" % (position + 1, len(rows), row["stable_image_id"][:12],
                                                 len(stability)), flush=True)

    i1 = np.asarray(iou_pass1, dtype=np.float64)
    i2 = np.asarray(iou_pass2, dtype=np.float64)
    is_ = np.asarray(shuffled_iou, dtype=np.float64)
    stability = np.asarray(stability, dtype=np.float64)
    hits = int((i1 >= 0.5).sum())
    require(hits >= GEOMETRIC_HITS_FLOOR,
            "only %d first-pass boxes overlap the raw annotations at 0.50: box space or image "
            "identity is wrong, the reading would be meaningless" % hits)

    report = {
        "images": len(rows), "small_detections": int(len(stability)),
        "inverse_sigmoid_source": inverse_source,
        "protocol": {
            "first_pass": "certified forward, untouched; predicate capture asserts the production gather",
            "second_pass": "same decoder, heads and content; reference = inverse_sigmoid(pass1 boxes)",
            "control": "same compute; reference rolled by NUM_QUERIES//2 across queries of the image",
            "box_space": "pred_boxes are normalised cxcywh; pixel boxes use [width,height]*2, the "
                         "certified rank_candidates convention",
            "matching": "per query, best IoU against the frozen raw annotations of the argmax class",
            "ground_truth_use": "geometric-overlap diagnostic only; not certified matching (no region "
                                "ignore filter, no per-class ranking, no maxDets); never used to select",
            "small_band": "detection pixel area < %g (original image pixels)" % SMALL_AREA},
        "geometric_hits_pass1_at_050": hits,
        "stability": summary(stability.tolist()),
        "iou_pass1": summary(i1.tolist()), "iou_pass2": summary(i2.tolist()),
        "iou_shuffled": summary(is_.tolist()),
        "pass2_minus_pass1": summary((i2 - i1).tolist()),
        "shuffled_minus_pass1": summary((is_ - i1).tolist())}

    improved, worsened = i2 > i1 + CHANGE_TOLERANCE, i2 < i1 - CHANGE_TOLERANCE
    s_improved, s_worsened = is_ > i1 + CHANGE_TOLERANCE, is_ < i1 - CHANGE_TOLERANCE
    report["change_counts"] = {
        "improved_by_pass2": int(improved.sum()), "worsened_by_pass2": int(worsened.sum()),
        "unchanged_by_pass2": int((~(improved | worsened)).sum()),
        "improved_by_shuffled": int(s_improved.sum()), "worsened_by_shuffled": int(s_worsened.sum())}
    report["crossing"] = {}
    for threshold in (0.5, 0.75):
        report["crossing"][str(threshold)] = {
            "pass2_rescues": int(((i1 < threshold) & (i2 >= threshold)).sum()),
            "pass2_breaks": int(((i1 >= threshold) & (i2 < threshold)).sum()),
            "shuffled_rescues": int(((i1 < threshold) & (is_ >= threshold)).sum()),
            "shuffled_breaks": int(((i1 >= threshold) & (is_ < threshold)).sum())}
    report["stability_of_change"] = {
        "improved": summary(stability[improved].tolist()),
        "worsened": summary(stability[worsened].tolist()),
        "unchanged": summary(stability[~(improved | worsened)].tolist()),
        "shuffled_improved": summary(stability[s_improved].tolist()),
        "shuffled_worsened": summary(stability[s_worsened].tolist())}
    median = float(np.median(stability))
    high, low = stability > median, stability <= median
    report["stability_split_at_median"] = {
        "median": median, "high_count": int(high.sum()), "low_count": int(low.sum()),
        "high_mean_delta": float((i2 - i1)[high].mean()) if high.any() else None,
        "low_mean_delta": float((i2 - i1)[low].mean()) if low.any() else None,
        "high_mean_iou_pass1": float(i1[high].mean()) if high.any() else None,
        "high_mean_iou_pass2": float(i2[high].mean()) if high.any() else None,
        "low_mean_iou_pass1": float(i1[low].mean()) if low.any() else None,
        "low_mean_iou_pass2": float(i2[low].mean()) if low.any() else None,
        "high_shuffled_mean_delta": float((is_ - i1)[high].mean()) if high.any() else None,
        "low_shuffled_mean_delta": float((is_ - i1)[low].mean()) if low.any() else None}

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False)[:6000])


if __name__ == "__main__":
    main()
