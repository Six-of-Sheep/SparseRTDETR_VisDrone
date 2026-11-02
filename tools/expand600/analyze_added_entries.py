"""Offline decomposition of the r22 and r20 arms under the certified matching rule. No model, no GPU.

Protocol.  The certified bridge builds the dataset and filters BOTH sides with the category-0 raster
ignore filter, so what is matched here is exactly what the certified scorer matches.  Matching is the
certified style: per image and class, detections in descending score order, each taking the
highest-IoU still-unmatched ground truth of its class, at IoU 0.50 and 0.75, with the top
MAX_DETS_PER_CLASS detections per class and image kept.  Ground truth is used ONLY here, offline; no
prediction was ever chosen with it.

Identity.  bridge_inputs keeps the prediction file order inside an image and drops only the detections
its raster ignore filter rejects, so a two-pointer walk over the file order recovers the (query_id,
category_id) of every surviving detection exactly; the walk is asserted to be complete.  That is what
lets every added and displaced entry be classified.

Reported per comparison
  GT instances         common TP, new TP, lost TP, each also restricted to small GT (area < 1024 px)
  added detections      true positive, duplicate of a true positive, wrong class, weak localization
                       (same class, IoU in [0.1, 0.5)), background; the independent share, i.e. added
                       true positives whose ground truth the earlier list had not matched; the IoU
                       distribution of the added true positives; their ground-truth areas; and, for the
                       subset matched at 0.50 but not 0.75, the detection-to-ground-truth area ratio
  displaced detections  whether their ground truth is still matched in the later list
  r20                  box-change statistics, since its arms share the detection set and order
  gain sets            the r22 and r20 gain sets compared on the ground-truth instance level

usage: analyze_added_entries.py <release-source> <contract> <r22-round> <r20-round> <out-json> [images]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

IOU_THRESHOLDS = (0.5, 0.75)
MAX_DETS_PER_CLASS = 100
SMALL_AREA = 1024.0
WEAK_IOU = 0.1
MIN_TPS_PER_IMAGE = 20


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def iou_matrix(boxes_a, boxes_b):
    """IoU between two stacks of xywh boxes, the convention the certified bridge uses."""
    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), dtype=np.float64)
    aw, ah = np.clip(a[:, 2], 0.0, None), np.clip(a[:, 3], 0.0, None)
    bw, bh = np.clip(b[:, 2], 0.0, None), np.clip(b[:, 3], 0.0, None)
    inter = (np.clip(np.minimum(a[:, 0][:, None] + aw[:, None], b[:, 0][None, :] + bw[None, :])
                     - np.maximum(a[:, 0][:, None], b[:, 0][None, :]), 0.0, None)
             * np.clip(np.minimum(a[:, 1][:, None] + ah[:, None], b[:, 1][None, :] + bh[None, :])
                       - np.maximum(a[:, 1][:, None], b[:, 1][None, :]), 0.0, None))
    return inter / np.clip(aw[:, None] * ah[:, None] + bw[None, :] * bh[None, :] - inter, 1e-12, None)


def iou_xyxy(box_a, box_b):
    """IoU between two single pixel xyxy boxes, used only for the r20 box-change audit."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def match_group(det_boxes, det_scores, gt_boxes, other_boxes, threshold):
    """Certified-style greedy matching over one image and class.

    A detection takes the highest-IoU unmatched ground truth of its class when that IoU reaches the
    threshold; otherwise it is a duplicate when a ground truth of its class is already matched at the
    threshold, a wrong-class detection when another class overlaps it at the threshold, a
    weak-localization detection when its best same-class overlap is at least WEAK_IOU, and background
    otherwise.
    """
    same = iou_matrix(det_boxes, gt_boxes)
    other = iou_matrix(det_boxes, other_boxes)
    matched, status = {}, [None] * len(det_boxes)
    order = sorted(range(len(det_boxes)), key=lambda index: -det_scores[index])
    for index in order:
        best_free, best_free_j = 0.0, None
        for j in range(len(gt_boxes)):
            if j in matched:
                continue
            if same[index, j] > best_free:
                best_free, best_free_j = float(same[index, j]), j
        if best_free_j is not None and best_free >= threshold:
            matched[best_free_j] = index
            status[index] = ("tp", best_free_j, best_free)
            continue
        best_any, best_any_j = 0.0, None
        for j in range(len(gt_boxes)):
            if same[index, j] > best_any:
                best_any, best_any_j = float(same[index, j]), j
        if best_any >= threshold:
            status[index] = ("duplicate", best_any_j, best_any)
        elif len(other_boxes) and float(other[index].max(initial=0.0)) >= threshold:
            status[index] = ("wrong_class", int(other[index].argmax()), float(other[index].max()))
        elif best_any >= WEAK_IOU:
            status[index] = ("weak_localization", best_any_j, best_any)
        else:
            status[index] = ("background", None, best_any)
    return status, matched


def group_predictions(predictions):
    by_image = defaultdict(list)
    for entry in predictions:
        by_image[entry["image_id"]].append(entry)
    return by_image


def align_queries(rows, predictions, det):
    """Recover (query_id, category_id) for every bridge detection by a two-pointer walk."""
    ids = {row["stable_image_id"]: index + 1 for index, row in enumerate(rows)}
    pred_by_image = group_predictions(predictions)
    query_of, index = {}, 0
    for row in rows:
        for entry in pred_by_image.get(row["stable_image_id"], []):
            if index >= len(det):
                break
            target = det[index]
            box = [float(v) for v in entry["bbox_xyxy"]]
            expected = [box[0], box[1], box[2] - box[0], box[3] - box[1]]
            if (int(target["category_id"]) == int(entry["category_id"])
                    and float(target["score"]) == float(entry["score"])
                    and [float(v) for v in target["bbox"]] == expected):
                query_of[index] = (int(entry["query_id"]), int(entry["category_id"]))
                index += 1
    require(index == len(det),
            "alignment lost detections: %d of %d" % (index, len(det)))
    return query_of, ids


def analyze_list(rows, parsed, predictions, ev):
    """Match one prediction list: ground-truth keys and per-detection statuses keyed by query id."""
    dataset, det, counts = ev.bridge_inputs(rows, parsed, predictions)
    query_of, ids = align_queries(rows, predictions, det)
    gt_group, det_group = defaultdict(list), defaultdict(list)
    for annotation in dataset["annotations"]:
        gt_group[(annotation["image_id"], annotation["category_id"])].append(annotation)
    for index, entry in enumerate(det):
        det_group[(entry["image_id"], entry["category_id"])].append((index, entry))
    other_by_image = defaultdict(list)
    for (image_id, category_id), group in gt_group.items():
        other_by_image[image_id].append((category_id, group))
    out = {"counts": counts, "detections": len(det), "ground_truth": len(dataset["annotations"]),
           "ids": ids, "thresholds": {}}
    for threshold in IOU_THRESHOLDS:
        tp_gt, statuses = set(), {}
        for key in sorted(set(gt_group) | set(det_group)):
            gts = gt_group.get(key, [])
            indexed = sorted(det_group.get(key, []), key=lambda pair: -pair[1]["score"])
            indexed = indexed[:MAX_DETS_PER_CLASS]
            if not indexed:
                continue
            other_class = [g["bbox"] for category_id, group in other_by_image[key[0]]
                           if category_id != key[1] for g in group]
            status, matched = match_group([e["bbox"] for _i, e in indexed],
                                          [e["score"] for _i, e in indexed],
                                          [g["bbox"] for g in gts], other_class, threshold)
            for gt_index, det_index in matched.items():
                tp_gt.add((key[0], gts[gt_index]["raw_annotation_id"]))
            for det_index, (kind, gt_index, value) in enumerate(status):
                bridge_index, entry = indexed[det_index]
                query, category = query_of[bridge_index]
                statuses[(key[0], query, category)] = {
                    "kind": kind, "iou": value, "category_id": category, "score": entry["score"],
                    "gt_raw_id": (gts[gt_index]["raw_annotation_id"]
                                  if kind == "tp" and gt_index is not None else None)}
        out["thresholds"][str(threshold)] = {"tp_gt": tp_gt, "statuses": statuses}
    return out, dataset


def summary(values):
    values = [float(v) for v in values if v == v]          # drop NaN
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {"count": len(values), "mean": float(array.mean()), "median": float(np.median(array)),
            "min": float(array.min()), "max": float(array.max())}


def main():
    source, contract_path, r22_dir, r20_dir, out_path = sys.argv[1:6]
    limit = int(sys.argv[6]) if len(sys.argv) > 6 else 548
    sys.path.insert(0, source)              # root level modules such as slice_core
    sys.path.insert(0, source + "/src")     # the frozen package
    from sparse_rtdetr.r0 import evaluation as ev
    from sparse_rtdetr.protocol_r18 import slice_ft_job as job
    contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    data = job.development(contract)
    rows = data.records[:limit]
    require(len(rows) >= 8, "expected at least eight images")
    parsed = [data.raw(row) for row in rows]

    paths = {"r22_A0": Path(r22_dir) / "run-001/A0/pred-top600.json",
             "r22_E600": Path(r22_dir) / "run-001/E600/pred-top600.json",
             "r20_baseline": Path(r20_dir) / "run-003/baseline/pred-top600.json",
             "r20_method": Path(r20_dir) / "run-003/method/pred-top600.json"}
    for name, path in paths.items():
        require(path.is_file(), "missing %s at %s" % (name, path))
    loaded = {name: json.loads(path.read_text()) for name, path in paths.items()}
    if limit < 548:
        wanted = {row["stable_image_id"] for row in rows}
        loaded = {name: [e for e in entries if e["image_id"] in wanted]
                  for name, entries in loaded.items()}

    results = {}
    for name in ("r22_A0", "r22_E600", "r20_baseline", "r20_method"):
        result, _dataset = analyze_list(rows, parsed, loaded[name], ev)
        results[name] = result
        count = len(result["thresholds"]["0.5"]["tp_gt"])
        require(count >= MIN_TPS_PER_IMAGE * len(rows),
                "sanity floor: %s matched only %d ground truths at IoU 0.50" % (name, count))
        print("%-13s dets=%6d gt=%5d tp@0.50=%5d tp@0.75=%5d"
              % (name, result["detections"], result["ground_truth"], count,
                 len(result["thresholds"]["0.75"]["tp_gt"])), flush=True)

    dataset, _det, _counts = ev.bridge_inputs(rows, parsed, [])
    class_of, area_of, bbox_of = {}, {}, {}
    for annotation in dataset["annotations"]:
        key = (annotation["image_id"], annotation["raw_annotation_id"])
        class_of[key] = annotation["category_id"]
        area_of[key] = float(annotation["area"])
        bbox_of[key] = [float(v) for v in annotation["bbox"]]

    def tp(result, threshold):
        return result["thresholds"][str(threshold)]["tp_gt"]

    def subset(keys):
        return {key for key in keys if area_of.get(key, 1e18) < SMALL_AREA}

    report = {"protocol": {
        "matching": "certified bridge dataset and det filtering; greedy per class by descending score; "
                    "IoU 0.50 and 0.75; top %d detections per class and image" % MAX_DETS_PER_CLASS,
        "ground_truth_use": "offline analysis only; no prediction was chosen with it",
        "images": len(rows)}, "lists": {}, "comparisons": {}, "gain_sets": {}}
    for name, result in results.items():
        report["lists"][name] = {"detections": result["detections"],
                                 "ground_truth": result["ground_truth"],
                                 "tp_at_0.5": len(tp(result, 0.5)),
                                 "tp_at_0.75": len(tp(result, 0.75)),
                                 "filter_counts": result["counts"]}
    gain_at_05, gain_at_075 = {}, {}
    for label, early, late in (("r22", "r22_A0", "r22_E600"), ("r20", "r20_baseline", "r20_method")):
        comparison = {}
        for threshold in IOU_THRESHOLDS:
            a, b = tp(results[early], threshold), tp(results[late], threshold)
            comparison[str(threshold)] = {
                "common": len(a & b), "new": len(b - a), "lost": len(a - b),
                "new_small": len(subset(b - a)), "lost_small": len(subset(a - b)),
                "common_small": len(subset(a & b))}
        report["comparisons"][label] = comparison
        gain_at_05[label] = tp(results[late], 0.5) - tp(results[early], 0.5)
        gain_at_075[label] = tp(results[late], 0.75) - tp(results[early], 0.75)
        loose = gain_at_05[label] - gain_at_075[label]
        report["gain_sets"][label] = {
            "count_at_0.5": len(gain_at_05[label]), "count_at_0.75": len(gain_at_075[label]),
            "loose_only_at_0.5": len(loose),
            "loose_gt_area": summary([area_of.get(key, float("nan")) for key in loose])}
    shared = gain_at_05["r22"] & gain_at_05["r20"]
    report["gain_overlap"] = {
        "r22_new_gt": len(gain_at_05["r22"]), "r20_new_gt": len(gain_at_05["r20"]),
        "shared_gt": len(shared), "union_gt": len(gain_at_05["r22"] | gain_at_05["r20"]),
        "shared_small": len(subset(shared)),
        "r22_gain_small": len(subset(gain_at_05["r22"])),
        "r20_gain_small": len(subset(gain_at_05["r20"]))}
    for label in ("r22", "r20"):
        per_class, small = defaultdict(int), 0
        for key in gain_at_05[label]:
            per_class[class_of.get(key)] += 1
            if area_of.get(key, 1e18) < SMALL_AREA:
                small += 1
        report["gain_sets"][label]["per_class"] = {str(k): v for k, v in sorted(per_class.items())}
        report["gain_sets"][label]["small_share"] = small

    a0, e600 = loaded["r22_A0"], loaded["r22_E600"]
    e600_keys = {(e["image_id"], int(e["query_id"]), int(e["category_id"])) for e in e600}
    added = [e for e in e600 if e.get("source") == "new_query" or int(e["query_id"]) >= 300]
    displaced = [e for e in a0 if (e["image_id"], int(e["query_id"]), int(e["category_id"]))
                 not in e600_keys]
    ids = results["r22_E600"]["ids"]
    status_e600 = results["r22_E600"]["thresholds"]["0.5"]["statuses"]
    status_a0 = results["r22_A0"]["thresholds"]["0.5"]["statuses"]
    a0_tp_keys, e600_tp_keys = tp(results["r22_A0"], 0.5), tp(results["r22_E600"], 0.5)
    added_kinds, added_tp, added_tp_new = defaultdict(int), 0, 0
    added_iou, added_gt_area, loose_ratio = [], [], []
    for entry in added:
        info = status_e600.get((ids[entry["image_id"]], int(entry["query_id"]),
                                int(entry["category_id"])))
        if info is None:
            added_kinds["absent"] += 1
            continue
        added_kinds[info["kind"]] += 1
        if info["kind"] != "tp":
            continue
        added_tp += 1
        added_iou.append(info["iou"])
        gt_key = (ids[entry["image_id"]], info["gt_raw_id"])
        added_gt_area.append(area_of.get(gt_key, float("nan")))
        if gt_key not in a0_tp_keys:
            added_tp_new += 1
        if gt_key in gain_at_075["r22"]:
            continue                           # tight: also matched at 0.75
        det_box = [float(v) for v in entry["bbox_xyxy"]]
        det_area = max(0.0, det_box[2] - det_box[0]) * max(0.0, det_box[3] - det_box[1])
        gt_box = bbox_of.get(gt_key, [0.0, 0.0, 0.0, 0.0])
        gt_area = max(gt_box[2] * gt_box[3], 1e-9)
        loose_ratio.append(det_area / gt_area)
    report["added_entries"] = {
        "count": len(added), "kinds": dict(added_kinds), "true_positives": added_tp,
        "true_positives_new_gt": added_tp_new,
        "true_positives_already_matched_gt": added_tp - added_tp_new,
        "tp_iou": summary(added_iou), "tp_gt_area": summary(added_gt_area),
        "tp_loose_only_det_over_gt_area": summary(loose_ratio)}
    displaced_kinds = defaultdict(int)
    recovered = lost = 0
    lost_iou = []
    for entry in displaced:
        info = status_a0.get((ids[entry["image_id"]], int(entry["query_id"]),
                              int(entry["category_id"])))
        if info is None:
            displaced_kinds["absent"] += 1
            continue
        displaced_kinds[info["kind"]] += 1
        if info["kind"] == "tp" and info["gt_raw_id"] is not None:
            if (ids[entry["image_id"]], info["gt_raw_id"]) in e600_tp_keys:
                recovered += 1
            else:
                lost += 1
                lost_iou.append(info["iou"])
    report["displaced_entries"] = {"count": len(displaced), "kinds": dict(displaced_kinds),
                                   "true_positives_recovered": recovered,
                                   "true_positives_lost": lost, "lost_tp_iou": summary(lost_iou)}

    index20 = {(e["image_id"], int(e["query_id"]), int(e["category_id"])): e
               for e in loaded["r20_baseline"]}
    changed, changed_small, ious, ratios = 0, 0, [], []
    for entry in loaded["r20_method"]:
        key = (entry["image_id"], int(entry["query_id"]), int(entry["category_id"]))
        base = index20.get(key)
        if base is None:
            continue
        base_box = [float(v) for v in base["bbox_xyxy"]]
        new_box = [float(v) for v in entry["bbox_xyxy"]]
        if base_box == new_box:
            continue
        changed += 1
        ious.append(iou_xyxy(base_box, new_box))
        base_area = max(0.0, base_box[2] - base_box[0]) * max(0.0, base_box[3] - base_box[1])
        new_area = max(0.0, new_box[2] - new_box[0]) * max(0.0, new_box[3] - new_box[1])
        if base_area < SMALL_AREA:
            changed_small += 1
        ratios.append(new_area / max(base_area, 1e-9))
    report["r20_box_changes"] = {"changed": changed, "changed_small": changed_small,
                                 "box_iou": summary(ious), "new_over_old_area": summary(ratios)}

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("comparisons", "gain_overlap", "added_entries", "displaced_entries",
                       "r20_box_changes")}, indent=1, ensure_ascii=False))
    print(json.dumps({"gain_sets": report["gain_sets"]}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
