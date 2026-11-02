"""Offline decomposition of the r22 and r20 arms under the certified matching rule. No model, no GPU.

Protocol.  The certified bridge builds the dataset and filters BOTH sides with the category-0 raster
ignore filter, so what is matched here is exactly what the certified scorer matches.  Matching is the
certified style: per image and class, detections in descending score order, each taking the
highest-IoU still-unmatched ground truth of its class, evaluated at IoU 0.50 and 0.75 with the top
MAX_DETS_PER_CLASS detections per class and image kept.  Ground truth is used ONLY here, offline; no
prediction was ever chosen with it.

Reported per comparison
  GT instances        common TP, new TP (matched in the later list only), lost TP (earlier list only),
                      each also restricted to small GT (area < 1024 px)
  added detections     true positive, duplicate of a true positive, wrong class, weak localization
                      (same class, IoU in [0.1, 0.5)), background; plus the independent share, i.e.
                      added true positives whose ground truth the earlier list had not matched
  displaced detections whether their ground truth is still matched in the later list
  r20                box-change statistics, since its two arms share the same detection set and order
  gain sets          the r22 and r20 gain sets compared on the ground-truth instance level

usage: analyze_added_entries.py <release-source> <contract> <r22-round> <r20-round> <out-json>
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


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def iou_matrix(boxes_a, boxes_b):
    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), dtype=np.float64)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.clip(a[:, 2], 0.0, None) * np.clip(a[:, 3], 0.0, None)
    area_b = np.clip(b[:, 2], 0.0, None) * np.clip(b[:, 3], 0.0, None)
    return inter / np.clip(area_a[:, None] + area_b[None, :] - inter, 1e-12, None)


def match_group(det_boxes, det_scores, gt_boxes, other_boxes, threshold):
    """Certified-style greedy matching over one image and class.

    Returns per-detection status tuples and the ground-truth assignment.  A detection takes the
    highest-IoU unmatched ground truth if that IoU reaches the threshold; otherwise it is a duplicate
    when some ground truth of its class is already matched at the threshold, a wrong-class detection
    when another class overlaps it at the threshold, a weak-localization detection when its best
    same-class overlap is at least WEAK_IOU, and background otherwise.
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


def analyze_list(rows, parsed, predictions, ev):
    """Match one prediction list and return ground-truth keys, per-detection statuses and counts."""
    dataset, det, counts = ev.bridge_inputs(rows, parsed, predictions)
    gt_group, det_group = defaultdict(list), defaultdict(list)
    for annotation in dataset["annotations"]:
        gt_group[(annotation["image_id"], annotation["category_id"])].append(annotation)
    for entry in det:
        det_group[(entry["image_id"], entry["category_id"])].append(entry)
    other_by_image = defaultdict(list)
    for (image_id, category_id), group in gt_group.items():
        other_by_image[image_id].append((category_id, group))
    out = {"counts": counts, "detections": len(det), "ground_truth": len(dataset["annotations"]),
           "thresholds": {}}
    for threshold in IOU_THRESHOLDS:
        tp_gt, statuses = set(), {}
        for key in sorted(set(gt_group) | set(det_group)):
            gts = gt_group.get(key, [])
            dets = sorted(det_group.get(key, []), key=lambda e: -e["score"])[:MAX_DETS_PER_CLASS]
            if not dets:
                continue
            other_class = [g["bbox"] for (image_id, category_id), group in other_by_image[key[0]]
                           if category_id != key[1] for g in group]
            status, matched = match_group([d["bbox"] for d in dets], [d["score"] for d in dets],
                                          [g["bbox"] for g in gts], other_class, threshold)
            for gt_index, det_index in matched.items():
                tp_gt.add((key[0], gts[gt_index]["raw_annotation_id"]))
            for det_index, (kind, gt_index, value) in enumerate(status):
                entry = dets[det_index]
                statuses[(key[0], det_index)] = {
                    "kind": kind, "iou": value, "category_id": key[1], "score": entry["score"],
                    "gt_raw_id": (gts[gt_index]["raw_annotation_id"]
                                  if kind == "tp" and gt_index is not None else None)}
        out["thresholds"][str(threshold)] = {"tp_gt": tp_gt, "statuses": statuses}
    return out, dataset


def statuses_by_key(result, threshold, predictions):
    """Re-key one list's detection statuses to (image_id, query_id, category_id)."""
    per_class = defaultdict(list)
    for entry in predictions:
        per_class[(entry["image_id"], int(entry["category_id"]))].append(entry)
    key_of = {}
    for (image_id, category_id), group in per_class.items():
        group = sorted(group, key=lambda e: -e["score"])[:MAX_DETS_PER_CLASS]
        for index, entry in enumerate(group):
            key_of[(image_id, index)] = (image_id, int(entry["query_id"]), int(entry["category_id"]))
    out = {}
    for index_key, info in result["thresholds"][str(threshold)]["statuses"].items():
        target = key_of.get(index_key)
        if target is not None:
            out[target] = info
    return out


def group_predictions(predictions):
    by_image = defaultdict(list)
    for entry in predictions:
        by_image[entry["image_id"]].append(entry)
    return by_image


def main():
    source, contract_path, r22_dir, r20_dir, out_path = sys.argv[1:6]
    sys.path.insert(0, source + "/src")
    from sparse_rtdetr.r0 import evaluation as ev
    from sparse_rtdetr.protocol_r18 import slice_ft_job as job
    contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    data = job.development(contract)
    rows = data.records
    require(len(rows) == 548, "expected the 548-image development set")
    parsed = [data.raw(row) for row in rows]

    paths = {"r22_A0": Path(r22_dir) / "run-001/A0/pred-top600.json",
             "r22_E600": Path(r22_dir) / "run-001/E600/pred-top600.json",
             "r20_baseline": Path(r20_dir) / "run-003/baseline/pred-top600.json",
             "r20_method": Path(r20_dir) / "run-003/method/pred-top600.json"}
    for name, path in paths.items():
        require(path.is_file(), "missing %s at %s" % (name, path))
    loaded = {name: json.loads(path.read_text()) for name, path in paths.items()}

    results = {}
    for name in ("r22_A0", "r22_E600", "r20_baseline", "r20_method"):
        result, _dataset = analyze_list(rows, parsed, loaded[name], ev)
        result["statuses_by_key"] = statuses_by_key(result, 0.5, loaded[name])
        results[name] = result
        print("%-13s dets=%6d gt=%5d tp@0.50=%5d tp@0.75=%5d"
              % (name, result["detections"], result["ground_truth"],
                 len(result["thresholds"]["0.5"]["tp_gt"]),
                 len(result["thresholds"]["0.75"]["tp_gt"])), flush=True)

    dataset, _det, _counts = ev.bridge_inputs(rows, parsed, [])
    class_of, area_of = {}, {}
    for annotation in dataset["annotations"]:
        key = (annotation["image_id"], annotation["raw_annotation_id"])
        class_of[key] = annotation["category_id"]
        area_of[key] = float(annotation["area"])

    def tp(result, threshold):
        return result["thresholds"][str(threshold)]["tp_gt"]

    def subset(keys):
        return {key for key in keys if area_of.get(key, 1e18) < SMALL_AREA}

    report = {"protocol": {
        "matching": "certified bridge dataset and det filtering; greedy per class by descending score; "
                    "IoU 0.50 and 0.75; top %d detections per class and image" % MAX_DETS_PER_CLASS,
        "ground_truth_use": "offline analysis only; no prediction was chosen with it"},
        "lists": {}, "comparisons": {}, "gain_sets": {}}
    for name, result in results.items():
        report["lists"][name] = {"detections": result["detections"],
                                 "ground_truth": result["ground_truth"],
                                 "tp_at_0.5": len(tp(result, 0.5)), "tp_at_0.75": len(tp(result, 0.75)),
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
        report["gain_sets"][label] = {"count_at_0.5": len(gain_at_05[label]),
                                      "count_at_0.75": len(gain_at_075[label])}
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
    status_e600 = results["r22_E600"]["statuses_by_key"]
    status_a0 = results["r22_A0"]["statuses_by_key"]
    added_kinds = defaultdict(int)
    added_tp = added_tp_new = 0
    a0_tp_keys = tp(results["r22_A0"], 0.5)
    e600_tp_keys = tp(results["r22_E600"], 0.5)
    for entry in added:
        info = status_e600.get((entry["image_id"], int(entry["query_id"]), int(entry["category_id"])))
        if info is None:
            added_kinds["beyond_maxdets"] += 1
            continue
        added_kinds[info["kind"]] += 1
        if info["kind"] == "tp":
            added_tp += 1
            if (entry["image_id"], info["gt_raw_id"]) not in a0_tp_keys:
                added_tp_new += 1
    report["added_entries"] = {"count": len(added), "kinds": dict(added_kinds),
                               "true_positives": added_tp, "true_positives_new_gt": added_tp_new,
                               "true_positives_already_matched_gt": added_tp - added_tp_new}
    displaced_kinds = defaultdict(int)
    recovered = lost = 0
    for entry in displaced:
        info = status_a0.get((entry["image_id"], int(entry["query_id"]), int(entry["category_id"])))
        if info is None:
            displaced_kinds["beyond_maxdets"] += 1
            continue
        displaced_kinds[info["kind"]] += 1
        if info["kind"] == "tp" and info["gt_raw_id"] is not None:
            if (entry["image_id"], info["gt_raw_id"]) in e600_tp_keys:
                recovered += 1
            else:
                lost += 1
    report["displaced_entries"] = {"count": len(displaced), "kinds": dict(displaced_kinds),
                                   "true_positives_recovered": recovered,
                                   "true_positives_lost": lost}

    # r20 shares its detection set between arms, so only boxes differ
    index20 = {(e["image_id"], int(e["query_id"]), int(e["category_id"])): e for e in loaded["r20_baseline"]}
    changed, changed_small, ious = 0, 0, []
    for entry in loaded["r20_method"]:
        key = (entry["image_id"], int(entry["query_id"]), int(entry["category_id"]))
        base = index20.get(key)
        if base is None:
            continue
        if [float(v) for v in base["bbox_xyxy"]] != [float(v) for v in entry["bbox_xyxy"]]:
            changed += 1
            value = float(iou_matrix([base["bbox_xyxy"]], [entry["bbox_xyxy"]])[0, 0])
            ious.append(value)
            width = base["bbox_xyxy"][2] - base["bbox_xyxy"][0]
            height = base["bbox_xyxy"][3] - base["bbox_xyxy"][1]
            if width * height < SMALL_AREA:
                changed_small += 1
    report["r20_box_changes"] = {
        "changed": changed, "changed_small": changed_small,
        "box_iou_min": min(ious) if ious else None, "box_iou_median":
            float(np.median(ious)) if ious else None, "box_iou_max": max(ious) if ious else None}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("comparisons", "gain_overlap", "added_entries", "displaced_entries",
                       "r20_box_changes")}, indent=1, ensure_ascii=False))
    print(json.dumps({"gain_per_class": {label: report["gain_sets"][label]["per_class"]
                                         for label in ("r22", "r20")},
                      "gain_small_share": {label: report["gain_sets"][label]["small_share"]
                                           for label in ("r22", "r20")}}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
