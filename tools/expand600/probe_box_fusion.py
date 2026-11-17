"""Offline pilot: fuse the boxes of mutually-agreeing same-class candidates. No model, no GPU.

Why this mechanism, and how it differs from the nearest precedents.  The offline decomposition says the
binding constraint in the small band is per-object box precision, not candidate supply: r22's 630 newly
matched ground truths at IoU 0.50 brought only 251 at 0.75, and the loose boxes are spread on both sides
of the truth (area ratio p10 0.73, p90 1.61).  The certified list also carries several names for the same
object (different queries, same class).  r22 dropped such duplicates (dedup 0.7), r20 applied one fixed
transform to every box (an enlarged window), r21 merged the encoder pool's class scores into the list.
None of them used the *agreement between independent candidates* as a signal.  This pilot asks whether
that agreement is usable GT-free evidence for localization: replace each name's box by its cluster's
fused box, leave the score, the names and their order alone, and score with the certified protocol.  The
decision-relevant cell is AP75_small: a mechanism that only trades duplicates for budget would move AP
and APS (as r22 did) and leave AP75_small where it was.

Ground truth enters only the offline measurement (the certified bridge and the per-object diagnostic).
The method never sees it: clustering is greedy, seed-anchored, score-ordered; the fusion is a plain mean,
a score-weighted mean, a coordinate median or the seed's own box; and the only fitted quantity is the
clustering threshold, scanned over a three-point grid that is disclosed in full.

Controls that can refute the explanation
  own boxes     rebuild every name with its own box: must reproduce the certified reading exactly, so the
                clustering and rebuild path is shown not to perturb anything (plus the published anchor);
  seed box      every member takes its cluster's seed box: separates "one box per cluster" from averaging;
  shuffled      the same fused boxes are dealt to the next cluster of the same image and class, so the
                multiset of boxes, the cluster sizes and the list are unchanged: if a gain survives the
                deal, then what helps is not "this box belongs to this object".

usage: probe_box_fusion.py <release-source> <contract> <r22-round> <release-root> <out-json> [images]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import analyze_added_entries as base                                   # noqa: E402

FUSE_GRID = (0.7, 0.6, 0.5)
MODES = ("mean", "wmean", "median")
SMALL_AREA = base.SMALL_AREA
PUBLISHED = {"AP": 0.2978928000787488, "APS": 0.2217618575337924, "AP75_small": 0.1996810452844532}
ANCHOR_TOLERANCE = 1e-9


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def macro_from_det(dataset, det, macro):
    """Certified bridge metrics under the class-equal-weight aggregation the published block uses."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    coco = COCO()
    coco.dataset = dataset
    coco.createIndex()
    if det:
        loaded = coco.loadRes(det)
    else:
        loaded = COCO()
        loaded.dataset = {"info": dataset["info"], "images": dataset["images"],
                          "categories": dataset["categories"], "annotations": []}
        loaded.createIndex()
    evaluator = COCOeval(coco, loaded, "bbox")
    evaluator.params.imgIds = sorted(row["id"] for row in dataset["images"])
    evaluator.params.catIds = list(range(1, 11))
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.evaluate()
    evaluator.accumulate()
    return macro(evaluator.eval), evaluator.eval


def clusters_of(entries, threshold, iou_matrix):
    """Greedy seed-anchored same-class clusters inside one (image, class) group. GT-free and stable."""
    order = sorted(range(len(entries)),
                   key=lambda i: (-float(entries[i]["score"]), int(entries[i]["query_id"]),
                                  int(entries[i]["category_id"])))
    assigned = [False] * len(entries)
    groups = []
    for seed in order:
        if assigned[seed]:
            continue
        assigned[seed] = True
        members = [seed]                                   # group[0] is always the cluster seed
        rest = [i for i in order if not assigned[i]]
        if rest:
            values = iou_matrix([entries[seed]["bbox_xyxy"]],
                                [entries[i]["bbox_xyxy"] for i in rest]).reshape(-1)
            for i, value in zip(rest, values):
                if float(value) >= threshold:
                    assigned[i] = True
                    members.append(i)
        groups.append(members)
    require(sum(len(group) for group in groups) == len(entries), "clustering is not a partition")
    return groups


def fuse_boxes(boxes, weights, mode):
    array = np.asarray(boxes, dtype=np.float64)
    if mode == "mean":
        return array.mean(axis=0)
    if mode == "wmean":
        weight = np.asarray(weights, dtype=np.float64)
        weight = weight / max(weight.sum(), 1e-12)
        return (array * weight[:, None]).sum(axis=0)
    if mode == "median":
        return np.median(array, axis=0)
    if mode == "seed":
        return array[0]
    raise SystemExit("unknown fusion mode %s" % mode)


def variant_list(predictions, threshold, mode, control, iou_matrix):
    """Rewrite every name's box with its cluster's fused box; same names, same scores, same order.

    The output keeps the certified file order exactly, so the certified bridge and the identity walk
    downstream see the same sequence they see for the untouched list.
    """
    positions_of = defaultdict(list)
    for position, entry in enumerate(predictions):
        positions_of[(entry["image_id"], int(entry["category_id"]))].append(position)
    boxes = [None] * len(predictions)
    sizes = [0] * len(predictions)
    for key in positions_of:
        positions = positions_of[key]
        entries = [predictions[position] for position in positions]
        groups = clusters_of(entries, threshold, iou_matrix)
        if mode == "own":
            fused = [None] * len(groups)                       # plumbing check: keep every own box
        else:
            fused = [fuse_boxes([entries[i]["bbox_xyxy"] for i in group],
                                [float(entries[i]["score"]) for i in group], mode)
                     for group in groups]
        if control and len(groups) > 1:
            fused = fused[1:] + fused[:1]                       # deal the boxes to the next cluster
        for group, box in zip(groups, fused):
            for i in group:
                target = positions[i]
                boxes[target] = ([float(v) for v in entries[i]["bbox_xyxy"]] if mode == "own"
                                 else [float(v) for v in box])
                sizes[target] = len(group)
    out = [{**entry, "bbox_xyxy": boxes[position], "fused_from": sizes[position]}
           for position, entry in enumerate(predictions)]
    require(len(out) == len(predictions), "fusion changed the number of names")
    return out


def cluster_stats(predictions, threshold, iou_matrix):
    by_group = defaultdict(list)
    for entry in predictions:
        by_group[(entry["image_id"], int(entry["category_id"]))].append(entry)
    stats = {"clusters": 0, "multi_clusters": 0, "names": 0, "names_in_multi": 0, "pairwise": []}
    for key in by_group:
        entries = by_group[key]
        for group in clusters_of(entries, threshold, iou_matrix):
            stats["clusters"] += 1
            stats["names"] += len(group)
            if len(group) > 1:
                stats["multi_clusters"] += 1
                stats["names_in_multi"] += len(group)
                boxes = [entries[i]["bbox_xyxy"] for i in group]
                matrix = iou_matrix(boxes, boxes)
                stats["pairwise"].extend(float(matrix[a][b])
                                         for a in range(len(boxes)) for b in range(a + 1, len(boxes)))
    return stats


def per_object_delta(predictions, threshold, mode, annotations, ids, iou_matrix):
    """GT-offline: for each object a cluster's seed already overlaps at 0.50, does the fused box sit closer?

    The overlap is a geometric diagnostic against the certified dataset's region-filtered annotations,
    used only to bucket objects; nothing here feeds the method.
    """
    by_group = defaultdict(list)
    for entry in predictions:
        by_group[(ids[entry["image_id"]], int(entry["category_id"]))].append(entry)
    truth = defaultdict(list)
    for annotation in annotations:
        box = annotation["bbox"]
        truth[(annotation["image_id"], int(annotation["category_id"]))].append(
            [box[0], box[1], box[0] + box[2], box[1] + box[3]])
    delta, small_delta, touched = [], [], 0
    for key in by_group:
        entries = by_group[key]
        boxes = truth.get(key, [])
        if not boxes:
            continue
        for group in clusters_of(entries, threshold, iou_matrix):
            if len(group) < 2:
                continue
            seed_box = entries[group[0]]["bbox_xyxy"]
            fused = fuse_boxes([entries[i]["bbox_xyxy"] for i in group],
                               [float(entries[i]["score"]) for i in group], mode)
            overlaps = iou_matrix([seed_box], boxes).reshape(-1)
            for index, value in enumerate(overlaps):
                if float(value) < 0.5:
                    continue
                touched += 1
                fused_iou = float(iou_matrix([fused.tolist()], [boxes[index]]).reshape(-1)[0])
                delta.append(fused_iou - float(value))
                area = (boxes[index][2] - boxes[index][0]) * (boxes[index][3] - boxes[index][1])
                if area < SMALL_AREA:
                    small_delta.append(fused_iou - float(value))
    return {"multi_cluster_objects_overlapped": touched, "delta_iou": base.summary(delta),
            "delta_iou_small": base.summary(small_delta)}


def main():
    source, contract_path, r22_dir, release_root, out_path = sys.argv[1:6]
    limit = int(sys.argv[6]) if len(sys.argv) > 6 else 548
    release_source = Path(release_root) / "source"
    for extra in (str(release_source / "src"), str(release_source), str(Path(release_root)),
                  source, source + "/src"):
        if extra not in sys.path:
            sys.path.insert(0, extra)
    # The frozen r22 release source supplies the endpoint, the scorer, the development list and the
    # fusion module, i.e. exactly the code the r22 run itself was bound to (release r22-004); its
    # slice_ft_job.development is the build that takes the production contract.
    import hashlib
    import tools.expand600.expand600 as expand600_module
    module_path = Path(expand600_module.__file__).resolve()
    require(str(module_path).startswith(str(release_source.resolve())),
            "the fusion module came from %s, outside the frozen release source" % module_path)
    per_class_macro_metrics = expand600_module.per_class_macro_metrics
    xyxy_iou_matrix = expand600_module.xyxy_iou_matrix
    module_info = {"path": str(module_path), "root": "frozen r22 release",
                   "sha256": hashlib.sha256(module_path.read_bytes()).hexdigest()}
    from sparse_rtdetr.r0 import evaluation as ev
    from sparse_rtdetr.protocol_r18 import slice_ft_job as job
    evaluation_path = Path(ev.__file__).resolve()
    require(str(evaluation_path).startswith(str(release_source.resolve())),
            "the certified scorer came from %s, outside the frozen release source" % evaluation_path)
    evaluation_info = {"path": str(evaluation_path), "root": "frozen r22 release",
                       "sha256": hashlib.sha256(evaluation_path.read_bytes()).hexdigest()}

    contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    data = job.development(contract)
    rows = data.records[:limit]
    require(len(rows) >= 8, "expected at least eight images")
    parsed = [data.raw(row) for row in rows]
    ids = {row["stable_image_id"]: index + 1 for index, row in enumerate(rows)}

    path = Path(r22_dir) / "run-001/A0/pred-top600.json"
    require(path.is_file(), "missing %s" % path)
    certified = json.loads(path.read_text())
    if limit < 548:
        certified = [entry for entry in certified if entry["image_id"] in ids]
    require(len(certified) == limit * 600, "expected 600 names per image, found %d" % len(certified))

    report = {"images": len(rows), "names": len(certified), "fusion_module": module_info,
        "evaluation_module": evaluation_info,
        "protocol": {
        "arm": "A0 certified top-600 name list; names, scores and order unchanged, boxes rewritten",
        "clustering": "per image and class, greedy seed-anchored by descending certified score",
        "fusion": "plain mean / score-weighted mean / coordinate median / seed box of the cluster",
        "controls": "own boxes (must equal certified); seed box only; fused boxes dealt to the next "
                    "cluster of the same image and class",
        "ground_truth_use": "offline measurement only, never used to build a cluster or a box",
        "scoring": "certified bridge (category-0 raster ignore filter after truncation) with the "
                   "class-equal-weight aggregation of the published block"},
        "published_anchor": PUBLISHED, "anchor_tolerance": ANCHOR_TOLERANCE,
        "grid": list(FUSE_GRID), "variants": {}, "tables": {}}

    def dump():
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")

    def delta_pp(metrics):
        return {key: (metrics[key] - anchor[key]) * 100.0
                for key in ("AP", "AP50", "AP75", "APS", "APM", "APL", "AP75_small")}

    dataset, det, _counts = ev.bridge_inputs(rows, parsed, certified)
    anchor, _eval = macro_from_det(dataset, det, per_class_macro_metrics)
    report["anchor_check"] = {key: {"measured": anchor[key], "published": PUBLISHED[key],
                                    "abs_diff": abs(anchor[key] - PUBLISHED[key])}
                              for key in PUBLISHED}
    for key in PUBLISHED:
        require(abs(anchor[key] - PUBLISHED[key]) <= ANCHOR_TOLERANCE,
                "certified anchor not reproduced for %s: %.17g vs %.17g"
                % (key, anchor[key], PUBLISHED[key]))
    report["variants"]["certified"] = {key: anchor[key] for key in anchor}
    print("certified anchor reproduced: AP %.16f APS %.16f AP75_small %.16f"
          % (anchor["AP"], anchor["APS"], anchor["AP75_small"]), flush=True)
    dump()

    annotations = dataset["annotations"]
    area_of = {(annotation["image_id"], annotation["raw_annotation_id"]): float(annotation["area"])
               for annotation in annotations}

    def small_only(keys):
        return {key for key in keys if area_of.get(key, 1e18) < SMALL_AREA}

    for threshold in FUSE_GRID:
        stats = cluster_stats(certified, threshold, xyxy_iou_matrix)
        pairwise = stats.pop("pairwise")
        stats["pairwise_iou"] = base.summary(pairwise)
        report["tables"][str(threshold)] = stats
        print("thr %.1f: %d clusters, %d of size>=2, %d names (%d inside size>=2)"
              % (threshold, stats["clusters"], stats["multi_clusters"], stats["names"],
                 stats["names_in_multi"]), flush=True)
        for mode in ("own",) + MODES:
            name = "%s@%.1f" % (mode, threshold)
            fused = variant_list(certified, threshold, mode, False, xyxy_iou_matrix)
            if mode == "own":
                identical = all(fused[i]["bbox_xyxy"] == certified[i]["bbox_xyxy"]
                                for i in range(len(certified)))
                report["variants"][name] = {"boxes_identical_to_certified": bool(identical)}
                require(identical, "the rebuild path does not preserve boxes under mode 'own'")
                print("  %-12s boxes identical to certified: %s" % (name, identical), flush=True)
                continue
            dataset, det, _counts = ev.bridge_inputs(rows, parsed, fused)
            metrics, _eval = macro_from_det(dataset, det, per_class_macro_metrics)
            report["variants"][name] = {key: metrics[key] for key in metrics}
            report["variants"][name]["delta_pp"] = delta_pp(metrics)
            print("  %-12s dAP %+.4f dAPS %+.4f dAP75 %+.4f dAP75_small %+.4f"
                  % (name, delta_pp(metrics)["AP"], delta_pp(metrics)["APS"],
                     delta_pp(metrics)["AP75"], delta_pp(metrics)["AP75_small"]), flush=True)
            dump()
        for mode, control in (("mean", True), ("seed", False)):
            name = "%s@%.1f" % ("shuffled" if control else "seedbox", threshold)
            fused = variant_list(certified, threshold, mode, control, xyxy_iou_matrix)
            dataset, det, _counts = ev.bridge_inputs(rows, parsed, fused)
            metrics, _eval = macro_from_det(dataset, det, per_class_macro_metrics)
            report["variants"][name] = {key: metrics[key] for key in metrics}
            report["variants"][name]["delta_pp"] = delta_pp(metrics)
            print("  %-12s dAP %+.4f dAPS %+.4f dAP75 %+.4f dAP75_small %+.4f"
                  % (name, delta_pp(metrics)["AP"], delta_pp(metrics)["APS"],
                     delta_pp(metrics)["AP75"], delta_pp(metrics)["AP75_small"]), flush=True)
            dump()

    for threshold in FUSE_GRID:
        report["tables"][str(threshold)]["per_object_seed_vs_fused_mean"] = per_object_delta(
            certified, threshold, "mean", annotations, ids, xyxy_iou_matrix)
        print("thr %.1f per-object %s" % (threshold, json.dumps(
            report["tables"][str(threshold)]["per_object_seed_vs_fused_mean"],
            ensure_ascii=False)), flush=True)
        dump()

    scored = [name for name in report["variants"] if "delta_pp" in report["variants"][name]]
    best = max(scored, key=lambda name: report["variants"][name]["delta_pp"]["AP75_small"])
    report["best_by_AP75_small"] = best
    print("best by AP75_small: %s" % best, flush=True)
    if best.startswith(("mean@", "wmean@", "median@", "seedbox@")):
        mode, threshold = best.split("@")
        mode = {"seedbox": "seed"}.get(mode, mode)             # the variant name is not the fusion mode
        fused = variant_list(certified, float(threshold), mode, False, xyxy_iou_matrix)
        early, _dataset = base.analyze_list(rows, parsed, certified, ev)
        late, _dataset = base.analyze_list(rows, parsed, fused, ev)
        decomposition = {"variant": best, "counts": {}}
        for level in ("0.5", "0.75"):
            early_tp = early["thresholds"][level]["tp_gt"]
            late_tp = late["thresholds"][level]["tp_gt"]
            decomposition["counts"][level] = {
                "early": len(early_tp), "late": len(late_tp), "common": len(early_tp & late_tp),
                "gained": len(late_tp - early_tp), "lost": len(early_tp - late_tp),
                "gained_small": len(small_only(late_tp - early_tp)),
                "lost_small": len(small_only(early_tp - late_tp))}
        report["decomposition"] = decomposition
        print(json.dumps(decomposition, ensure_ascii=False), flush=True)
    dump()
    print("FUSION_PILOT_COMPLETE")


if __name__ == "__main__":
    main()
