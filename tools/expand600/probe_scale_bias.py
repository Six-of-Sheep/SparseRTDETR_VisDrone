"""Measure the detection-to-ground-truth area ratio of the true positives, per arm and per band.

Question this answers, with no model run at all: is the roughly 12 percent area inflation seen on the
added entries whose match holds at IoU 0.50 but not 0.75 a property of the expansion only, or a
systematic scale bias of the model's small-object boxes that the certified list carries too?  The
answer decides whether a scale mechanism belongs on the expanded candidates or on the endpoint itself.

Reuses analyze_added_entries for the certified dataset, the greedy certified matching and the identity
alignment.  Ground truth enters only this offline analysis; no prediction was ever chosen with it.

usage: probe_scale_bias.py <release-source> <contract> <r22-round> <out-json> [images]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_added_entries as base                                   # noqa: E402

SMALL_AREA = base.SMALL_AREA


def tp(result, threshold):
    return result["thresholds"][str(threshold)]["tp_gt"]


def summary(values):
    values = [float(v) for v in values if v == v]
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {"count": len(values), "mean": float(array.mean()), "median": float(np.median(array)),
            "p10": float(np.percentile(array, 10)), "p90": float(np.percentile(array, 90)),
            "min": float(array.min()), "max": float(array.max())}


def boxes_of(predictions):
    table = {}
    for entry in predictions:
        box = [float(v) for v in entry["bbox_xyxy"]]
        table[(entry["image_id"], int(entry["query_id"]), int(entry["category_id"]))] = box
    return table


def main():
    source, contract_path, r22_dir, out_path = sys.argv[1:5]
    limit = int(sys.argv[5]) if len(sys.argv) > 5 else 548
    sys.path.insert(0, source)
    sys.path.insert(0, source + "/src")
    from sparse_rtdetr.r0 import evaluation as ev
    from sparse_rtdetr.protocol_r18 import slice_ft_job as job
    contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    data = job.development(contract)
    rows = data.records[:limit]
    parsed = [data.raw(row) for row in rows]

    predictions = {}
    for name, relative in (("A0", "run-001/A0/pred-top600.json"),
                           ("E600", "run-001/E600/pred-top600.json")):
        path = Path(r22_dir) / relative
        predictions[name] = json.loads(path.read_text())
    if limit < 548:
        wanted = {row["stable_image_id"] for row in rows}
        predictions = {name: [e for e in entries if e["image_id"] in wanted]
                       for name, entries in predictions.items()}

    results, ids = {}, None
    for name in ("A0", "E600"):
        result, dataset = base.analyze_list(rows, parsed, predictions[name], ev)
        results[name] = result
        ids = result["ids"]
        print("%s matched %d ground truths at 0.50 and %d at 0.75"
              % (name, len(tp(result, 0.5)), len(tp(result, 0.75))), flush=True)

    bbox_of, area_of, class_of = {}, {}, {}
    for annotation in dataset["annotations"]:
        key = (annotation["image_id"], annotation["raw_annotation_id"])
        bbox_of[key] = [float(v) for v in annotation["bbox"]]
        area_of[key] = float(annotation["area"])
        class_of[key] = annotation["category_id"]

    report = {"images": len(rows), "protocol": {
        "matching": "certified bridge dataset and det filtering, greedy per class at IoU 0.50 and 0.75",
        "ground_truth_use": "offline analysis only; no prediction was chosen with it"}, "arms": {}}
    for name in ("A0", "E600"):
        table = boxes_of(predictions[name])
        statuses = results[name]["thresholds"]["0.5"]["statuses"]
        tight_gt = tp(results[name], 0.75)
        buckets = defaultdict(list)
        aspect = defaultdict(list)
        for (image_id, query, category), info in statuses.items():
            if info["kind"] != "tp" or info["gt_raw_id"] is None:
                continue
            gt_key = (image_id, info["gt_raw_id"])
            gt_box, gt_area = bbox_of.get(gt_key), area_of.get(gt_key, 0.0)
            det_box = table.get((image_id, query, category))
            if gt_box is None or det_box is None:
                continue
            det_area = max(0.0, det_box[2] - det_box[0]) * max(0.0, det_box[3] - det_box[1])
            if gt_area <= 0:
                continue
            tight = "tight" if gt_key in tight_gt else "loose_only"
            band = "small" if gt_area < SMALL_AREA else "larger"
            buckets["%s/%s" % (tight, band)].append(det_area / gt_area)
            gt_ratio = (gt_box[2] / gt_box[3]) if gt_box[3] > 0 else float("nan")
            det_ratio = ((det_box[2] - det_box[0]) / (det_box[3] - det_box[1])
                         if det_box[3] > det_box[1] else float("nan"))
            if gt_ratio == gt_ratio and det_ratio == det_ratio:
                aspect["%s/%s" % (tight, band)].append(det_ratio / gt_ratio)
        report["arms"][name] = {
            "tp_total": len(tp(results[name], 0.5)),
            "det_over_gt_area": {key: summary(value) for key, value in sorted(buckets.items())},
            "det_over_gt_aspect": {key: summary(value) for key, value in sorted(aspect.items())}}
        print(name, json.dumps(report["arms"][name]["det_over_gt_area"], indent=1), flush=True)

    # the same split for the added entries only, to compare against the earlier figure
    entries_added = [e for e in predictions["E600"]
                     if e.get("source") == "new_query" or int(e["query_id"]) >= 300]
    table = boxes_of(entries_added)
    statuses = results["E600"]["thresholds"]["0.5"]["statuses"]
    tight_gt = tp(results["E600"], 0.75)
    buckets = defaultdict(list)
    for (image_id, query, category), info in statuses.items():
        if info["kind"] != "tp" or info["gt_raw_id"] is None:
            continue
        det_box = table.get((image_id, query, category))
        if det_box is None:
            continue
        gt_key = (image_id, info["gt_raw_id"])
        gt_area = area_of.get(gt_key, 0.0)
        if gt_area <= 0:
            continue
        det_area = max(0.0, det_box[2] - det_box[0]) * max(0.0, det_box[3] - det_box[1])
        tight = "tight" if gt_key in tight_gt else "loose_only"
        band = "small" if gt_area < SMALL_AREA else "larger"
        buckets["%s/%s" % (tight, band)].append(det_area / gt_area)
    report["added_entries_det_over_gt_area"] = {key: summary(value)
                                                for key, value in sorted(buckets.items())}

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"added": report["added_entries_det_over_gt_area"]}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
