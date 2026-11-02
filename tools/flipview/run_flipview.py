"""r23 production entry: plan-driven, identity-bound, inference only.

Arms
  A0    the pristine certified path: one whole-image forward per batch, no instrumentation at all;
        predictions are the certified ranking of the 300 queries (rank_candidates) at both budgets.
  FLIP  the same forward plus ONE extra forward of the same frozen weights on the horizontally mirrored
        canvas.  Two lists come out of the two views: R23 replaces the box of every small-band query
        (original box pixel area < 1024, the only scope condition, ground-truth free) by the plain mean
        of its box and the mirrored pass's box of highest overlap, when that overlap reaches 0.5; R23C
        is the second-highest-overlap partner sensitivity control, sharing the exact same treated set,
        the same forwards and the same thresholds.  Names, scores and the certified ordering are
        unchanged by construction, because the box never enters the certified ranking key
        (descending score, query id, class id).

Positioning: R23 is a pragmatic mirrored-TTA baseline, not a claimed original mechanism.  The pairing is
geometric (box overlap only); the two views share one set of weights, so no independence is claimed.

The certified evaluation protocol is used unchanged for the comparison: the category-0 raster ignore
filter is applied AFTER the budget truncation and nothing is refilled, so a treated box that the filter
rejects simply shrinks the list, and the per-image records count that.

Reports written by --score
  anchor_reproduction      score the certified prediction file here and compare with the published
                           top600-metrics.json block (class-equal-weight aggregation)
  first_stage_vs_native    A0's raw queries vs FLIP's original-view raw queries, same device, same
                           session; with no instrumentation anywhere this is a same-session determinism
                           check of the certified path
  cache_reconstruction     A0's raw queries and boxes, and its derived predictions, vs the certified
                           2026-10-14 cache and prediction file (last-bit differences recorded, never
                           repaired)
  cross_device             the cpu-check dumps vs the GPU dumps, for A0 and for R23
  gates / cost / per_class / deltas, plus the R23C sensitivity readings

usage: run_flipview.py --plan <plan.json> --arm A0|FLIP --out <dir> [--limit N] [--cpu-check]
       run_flipview.py --plan <plan.json> --score --out <round_dir>/run-001
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.flipview.flipview import (NUM_CLASSES, NUM_QUERIES, PAIRING_MODES, SMALL_AREA,  # noqa: E402
                                     fuse_view_boxes, mirror_boxes)

METRIC_KEYS = ("AP", "AP50", "AP75", "APS", "APM", "APL", "AP75_small")


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def state_digest(model):
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def load_plan(path):
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("runner", "mechanism", "arms", "gates", "evaluation", "endpoint", "decision_rule"):
        require(key in plan, "plan is missing %s" % key)
    return plan


def import_chain(plan):
    runner = plan["runner"]
    root = Path(runner["worktree_root"])
    src = Path(runner["release_source"])
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(src / "src"))
    return root, src


def admit_cuda(contract, out, plan):
    from sparse_rtdetr.localization import guard, worker as base
    from sparse_rtdetr.protocol_r18 import slice_ft_job as job
    from sparse_rtdetr.protocol_r18.control import install_resources
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == contract["gpu_uuid"],
            "CUDA_VISIBLE_DEVICES must equal the frozen GPU UUID at process startup")
    require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8",
            "CUBLAS_WORKSPACE_CONFIG must be :4096:8 at process startup")
    install_resources()
    sample = guard.snapshot(str(out))
    (out / "resource-entry.json").write_text(json.dumps(sample, ensure_ascii=False) + "\n")
    admission = {k: v for k, v in contract.items() if k != "runtime_root"}
    runtime = base.prepare_cuda(admission, out)
    model, _criterion = job.native_twin(contract, out)
    model = model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, "cuda:0", runtime


def cpu_parent_model(contract):
    """Real parent EMA weights on CPU: the same reconstruction path as native_twin, minus CUDA."""
    from sparse_rtdetr.protocol_r18 import slice_finetune as ft
    from sparse_rtdetr.protocol_r18 import slice_ft_execution as execution
    from sparse_rtdetr.protocol_r18 import checkpoint as native_state
    from sparse_rtdetr.protocol_r18.model import Config, build
    require(not torch.cuda.is_initialized(), "CUDA was initialized before the CPU check")
    model, _criterion, _info = build(Config(), synthetic=True)
    binding = execution.verified(contract["parent_binding"])
    reference = contract["parent_checkpoint"]
    payload = native_state.read(reference["path"], reference["sha256"], binding)
    ft.initialize_from_parent(model, payload, expected_binding=binding)
    del payload
    model = model.to("cpu").eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, "cpu", {"scope": "CPU_CHECK", "torch": str(torch.__version__)}


def native_identity(model):
    """Structural native-endpoint checks (the certified geometry validator is attempted as well)."""
    decoder = model.decoder
    anchors, valid_mask = decoder._generate_anchors()
    checks = {
        "decoder_has_selection": bool(hasattr(decoder, "_select_topk")
                                      and hasattr(decoder, "_get_decoder_input")),
        "no_r0_adaptation": not any(hasattr(decoder, name) for name in
                                    ("pre_bbox_head", "r0_geometry_parameters", "reg_scale",
                                     "box_history")),
        "dec_bbox_head_layers": len(list(getattr(decoder, "dec_bbox_head", []))),
        "dec_score_head_layers": len(list(getattr(decoder, "dec_score_head", []))),
        "num_queries": int(getattr(decoder, "num_queries", -1)),
        "num_classes": int(getattr(decoder, "num_classes", -1)),
        "canvas": [int(v) for v in getattr(decoder, "eval_spatial_size", [])],
        "anchors_bitwise": bool(torch.equal(decoder.anchors.cpu(), anchors)),
        "valid_mask_bitwise": bool(torch.equal(decoder.valid_mask.cpu(), valid_mask)),
    }
    try:
        from sparse_rtdetr.protocol_r18.geometry import validate_geometry
        validate_geometry(model)
        checks["validate_geometry"] = "PASS"
    except Exception as error:                                   # noqa: BLE001 - recorded, not raised
        checks["validate_geometry"] = "ERROR: %s: %s" % (type(error).__name__, str(error)[:200])
    return checks


def trim(per_image_lists, budget):
    """The first `budget` entries of each image's ranked list (the certified truncation rule)."""
    return [entry for image_list in per_image_lists for entry in image_list[:budget]]


def identity_of(entries):
    return [(int(e["query_id"]), int(e["category_id"]), float(e["score"])) for e in entries]


def boxes_of(entries):
    return [[float(v) for v in e["bbox_xyxy"]] for e in entries]


def summarize(values):
    if not values:
        return {"count": 0}
    array = np.asarray([float(v) for v in values], dtype=np.float64)
    return {"count": len(values), "min": float(array.min()), "mean": float(array.mean()),
            "max": float(array.max())}


def run_arm(args, plan, contract, model, device, batch_size):
    from sparse_rtdetr.r0 import evaluation as ev
    from sparse_rtdetr.protocol_r18 import slice_ft_job as job
    from sparse_rtdetr.protocol_r18.geometry import forward_guards
    out = Path(args.out)
    two_view = args.arm == "FLIP"
    data = job.development(contract)
    rows = data.records if not args.limit else data.records[: args.limit]
    require(len(rows) > 0, "no images selected")
    identity = native_identity(model)
    digest_before = state_digest(model)
    budget_main, budget_secondary = plan["runner"]["budgets"]
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    if two_view:
        for name in ("R23", "R23C"):
            (out / name).mkdir(parents=True, exist_ok=True)

    lists = {name: {"main": [], "secondary": []}
             for name in (("R23", "R23C") if two_view else ("A0",))}
    raw_original, raw_original_boxes, raw_mirrored, raw_mirrored_boxes = [], [], [], []
    timings, image_records = [], []
    stage = {"original_forward": 0.0, "mirrored_forward": 0.0, "pair_best": 0.0, "pair_second": 0.0,
             "reference_build": 0.0, "r23_build": 0.0, "r23c_build": 0.0, "a0_build": 0.0}
    counters = {"name_score_identity_violations": 0, "untreated_box_changes": 0,
                "treated_set_mismatches": 0, "changed_above_treated": 0,
                "treated_total": 0, "small_band_total": 0, "changed_at_600_total": 0,
                "changed_at_300_total": 0, "control_changed_at_600_total": 0}
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start + batch_size]
            begin = time.perf_counter()
            pixels = [data.item(row, position)[0] for position, row in enumerate(chunk, start)]
            x = torch.stack(pixels).to(device)
            with torch.autocast(device_type, enabled=False), forward_guards(model):
                mark = time.perf_counter()
                z = model(x)
                if device_type == "cuda":
                    torch.cuda.synchronize()
                stage["original_forward"] += time.perf_counter() - mark
                if two_view:
                    mark = time.perf_counter()
                    z_mirror = model(x.flip(-1))
                    if device_type == "cuda":
                        torch.cuda.synchronize()
                    stage["mirrored_forward"] += time.perf_counter() - mark
            logits = z["pred_logits"].detach().float().cpu()
            boxes = z["pred_boxes"].detach().float().cpu()
            mirror_logits = mirror_boxes_raw = None
            if two_view:
                mirror_logits = z_mirror["pred_logits"].detach().float().cpu()
                mirror_boxes_raw = z_mirror["pred_boxes"].detach().float().cpu()

            for j, row in enumerate(chunk):
                require(logits[j].dtype == torch.float32 and boxes[j].dtype == torch.float32,
                        "prediction dtype is not FP32")
                raw_original.append(logits[j].clone())
                raw_original_boxes.append(boxes[j].clone())
                mark = time.perf_counter()
                a0_main = ev.rank_candidates(logits[j].contiguous(), boxes[j].contiguous(), row,
                                             budget_main)
                a0_secondary = ev.rank_candidates(logits[j].contiguous(), boxes[j].contiguous(), row,
                                                  budget_secondary)
                if not two_view:
                    stage["a0_build"] += time.perf_counter() - mark      # the A0 arm's own work
                    lists["A0"]["main"].append(a0_main)
                    lists["A0"]["secondary"].append(a0_secondary)
                    continue
                stage["reference_build"] += time.perf_counter() - mark
                raw_mirrored.append(mirror_logits[j].clone())
                raw_mirrored_boxes.append(mirror_boxes_raw[j].clone())
                mapped = mirror_boxes(mirror_boxes_raw[j])
                require(tuple(boxes[j].shape) == (NUM_QUERIES, 4)
                        and tuple(mapped.shape) == (NUM_QUERIES, 4),
                        "the two views must each carry exactly %d queries" % NUM_QUERIES)
                mark = time.perf_counter()
                fused_best, treated_best, counters_best = fuse_view_boxes(
                    boxes[j], mapped, row["width"], row["height"], mode="best")
                stage["pair_best"] += time.perf_counter() - mark
                mark = time.perf_counter()
                fused_second, treated_second, counters_second = fuse_view_boxes(
                    boxes[j], mapped, row["width"], row["height"], mode="second")
                stage["pair_second"] += time.perf_counter() - mark
                require(bool(torch.equal(treated_best, treated_second)),
                        "the two pairing modes disagree about the treated set")
                require(bool(torch.equal(fused_best[~treated_best], boxes[j][~treated_best]))
                        and bool(torch.equal(fused_second[~treated_second],
                                             boxes[j][~treated_second])),
                        "an untreated box changed")
                counters["treated_total"] += int(counters_best["treated_queries"])
                counters["small_band_total"] += int(counters_best["small_band_queries"])
                record = {"image_id": row["stable_image_id"],
                          "small_band_queries": counters_best["small_band_queries"],
                          "treated_queries": counters_best["treated_queries"],
                          "partner_iou_of_treated": summarize(counters_best["partner_iou_of_treated"]),
                          "second_iou_of_treated": summarize(counters_second["second_iou_of_treated"]),
                          "queries_with_overlap_tie": counters_best["queries_with_overlap_tie"]}
                for name, fused, mark_key in (("R23", fused_best, "r23_build"),
                                              ("R23C", fused_second, "r23c_build")):
                    mark = time.perf_counter()
                    variant_main = ev.rank_candidates(logits[j].contiguous(), fused.contiguous(), row,
                                                      budget_main)
                    variant_secondary = ev.rank_candidates(logits[j].contiguous(), fused.contiguous(),
                                                           row, budget_secondary)
                    stage[mark_key] += time.perf_counter() - mark
                    if identity_of(variant_main) != identity_of(a0_main) \
                            or identity_of(variant_secondary) != identity_of(a0_secondary):
                        counters["name_score_identity_violations"] += 1
                        require(False, "the fused list changed names or scores")
                    changed_main = sum(1 for left, right in zip(boxes_of(variant_main), boxes_of(a0_main))
                                       if left != right)
                    changed_secondary = sum(
                        1 for left, right in zip(boxes_of(variant_secondary), boxes_of(a0_secondary))
                        if left != right)
                    if changed_main > counters_best["treated_queries"]:
                        counters["changed_above_treated"] += 1
                        require(False, "more boxes changed than queries were treated")
                    if name == "R23":
                        counters["changed_at_600_total"] += changed_main
                        counters["changed_at_300_total"] += changed_secondary
                        record["changed_at_600"] = changed_main
                        record["changed_at_300"] = changed_secondary
                    else:
                        counters["control_changed_at_600_total"] += changed_main
                        record["control_changed_at_600"] = changed_main
                        record["control_changed_at_300"] = changed_secondary
                    lists[name]["main"].append(variant_main)
                    lists[name]["secondary"].append(variant_secondary)
                image_records.append(record)
            if device_type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - begin
            timings.append({"batch": len(timings) + 1, "images": len(chunk), "seconds": elapsed})
            print("batch %d/%d images=%d %.3fs" % (len(timings),
                                                   (len(rows) + batch_size - 1) // batch_size,
                                                   len(chunk), elapsed), flush=True)

    digest_after = state_digest(model)
    report = {
        "arm": args.arm,
        "images": len(rows),
        "budgets": [budget_main, budget_secondary],
        "batch_size": batch_size,
        "device": device,
        "inference_dtype": "float32",
        "cuda_initialized": torch.cuda.is_initialized(),
        "state_exact_equal": digest_before == digest_after,
        "params_have_grad": any(p.grad is not None for p in model.parameters()),
        "timing_total_seconds": sum(t["seconds"] for t in timings),
        "timing_batches": timings,
        "stage_seconds": stage,
        "native_identity": identity,
        "checks": counters,
        "mechanism_scope": {"small_area_px": SMALL_AREA, "partner_iou": 0.5,
                            "pairing_modes": list(PAIRING_MODES), "ground_truth_used": False},
        "resource_entry": sha(out / "resource-entry.json")
                          if (out / "resource-entry.json").is_file() else None,
    }
    for name in lists:
        directory = out if name == "A0" else out / name
        for budget, key in ((budget_main, "main"), (budget_secondary, "secondary")):
            path = directory / ("pred-top%d.json" % budget)
            path.write_text(json.dumps(trim(lists[name][key], budget), separators=(",", ":")))
            report["pred-top%d_sha256" % budget] = sha(path)
            report["pred-top%d_bytes" % budget] = path.stat().st_size
    torch.save({"schema": "r23_flipview_q300_v1", "arm": args.arm, "view": "original",
                "image_ids": [row["stable_image_id"] for row in rows],
                "pred_logits": torch.stack(raw_original),
                "pred_boxes": torch.stack(raw_original_boxes),
                "query_ids": torch.arange(NUM_QUERIES)}, out / "raw-queries-q300.pth")
    if two_view:
        torch.save({"schema": "r23_flipview_q300_v1", "arm": args.arm, "view": "mirrored",
                    "image_ids": [row["stable_image_id"] for row in rows],
                    "pred_logits": torch.stack(raw_mirrored),
                    "pred_boxes": torch.stack(raw_mirrored_boxes),
                    "query_ids": torch.arange(NUM_QUERIES)}, out / "raw-queries-mirrored-q300.pth")
        (out / "image-records.json").write_text(json.dumps(image_records, indent=1, ensure_ascii=False))
        report["image_records_sha256"] = sha(out / "image-records.json")
    (out / ("RESULT-%s.json" % args.arm)).write_text(json.dumps(report, indent=1,
                                                                ensure_ascii=False) + "\n")
    print(json.dumps({k: report[k] for k in ("arm", "images", "state_exact_equal", "params_have_grad",
                                             "timing_total_seconds", "stage_seconds", "checks")}))
    print("WROTE", out / ("RESULT-%s.json" % args.arm))


def score_only(plan, contract, out):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from tools.flipview.flipview import per_class_macro_metrics, per_class_values
    from sparse_rtdetr.r0 import evaluation as ev
    from sparse_rtdetr.protocol_r18 import slice_ft_job as job
    data = job.development(contract)
    rows = data.records
    parsed = [data.raw(row) for row in rows]
    budgets = plan["evaluation"]["budgets"]

    def score_predictions(predictions):
        dataset, det, counts = ev.bridge_inputs(rows, parsed, predictions)
        coco = COCO()
        coco.dataset = dataset
        coco.createIndex()
        evaluator = COCOeval(coco, coco.loadRes(det), "bbox")
        evaluator.params.imgIds = sorted(r["id"] for r in dataset["images"])
        evaluator.params.catIds = list(range(1, NUM_CLASSES + 1))
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate()
        evaluator.accumulate()
        names = ("AP", "AP50", "AP75", "APS", "APM", "APL", "AR1", "AR10", "AR100",
                 "AR_small", "AR_medium", "AR_large")
        stats = {k: float(v) for k, v in zip(names, evaluator.stats)}
        small = evaluator.eval["precision"][5, :, :, 1, 2]
        stats["AP75_small"] = float(np.nanmean(small))
        return per_class_macro_metrics(evaluator.eval), stats, per_class_values(evaluator.eval), counts

    layout = {"A0": out / "A0", "R23": out / "FLIP" / "R23", "R23C": out / "FLIP" / "R23C"}
    scores, stats_form, per_class, filter_counts = {}, {}, {}, {}
    for arm, directory in layout.items():
        for budget in budgets:
            path = directory / ("pred-top%d.json" % budget)
            require(path.is_file(), "missing prediction file %s" % path)
            macro, stats, classes, counts = score_predictions(json.loads(path.read_text()))
            scores.setdefault(arm, {})[str(budget)] = macro
            stats_form.setdefault(arm, {})[str(budget)] = stats
            per_class.setdefault(arm, {})[str(budget)] = classes
            filter_counts.setdefault(arm, {})[str(budget)] = counts
            (directory / ("metrics-top%d.json" % budget)).write_text(json.dumps(macro, indent=1))

    parent = Path(plan["runner"]["parent_job"]) / "evaluation"
    certified_pred = parent / ("top%d-predictions.json" % budgets[0])
    published = json.loads((parent / ("top%d-metrics.json" % budgets[0])).read_text())["COCO_bridge"]["metrics"]
    macro, _stats, _classes, _counts = score_predictions(json.loads(certified_pred.read_text()))
    anchor = {"certified_predictions": {"path": str(certified_pred), "sha256": sha(certified_pred)},
              "published": {k: round(v, 15) for k, v in published.items()},
              "rescored": {k: round(macro[k], 15) for k in macro if k in published},
              "max_abs_diff": max(abs(macro[k] - published[k]) for k in published if k in macro),
              "published_form": "class-equal-weight (per class mean); COCO stats average cells instead"}

    def load_raw(path):
        return torch.load(path, map_location="cpu", weights_only=False) if path.is_file() else None

    a0_raw = load_raw(out / "A0" / "raw-queries-q300.pth")
    flip_raw = load_raw(out / "FLIP" / "raw-queries-q300.pth")
    first_stage = {"available": False}
    if a0_raw is not None and flip_raw is not None:
        logits_diff = (a0_raw["pred_logits"] - flip_raw["pred_logits"]).abs()
        boxes_diff = (a0_raw["pred_boxes"] - flip_raw["pred_boxes"]).abs()
        first_stage = {"available": True,
                       "logits_bitwise_equal": bool(torch.equal(a0_raw["pred_logits"],
                                                                flip_raw["pred_logits"])),
                       "logits_max_abs_diff": float(logits_diff.max()),
                       "boxes_bitwise_equal": bool(torch.equal(a0_raw["pred_boxes"],
                                                               flip_raw["pred_boxes"])),
                       "boxes_max_abs_diff": float(boxes_diff.max()),
                       "entries": int(logits_diff.numel()),
                       "note": ("A0 is the pristine certified forward, FLIP's original view is the same "
                                "forward inside the two-view pass; neither is instrumented, so this is a "
                                "same-session determinism check of the certified path")}
    cache_report = {"available": False}
    cache_path = Path(plan["runner"]["certified_cache"])
    if a0_raw is not None and cache_path.is_file():
        theirs = torch.load(cache_path, map_location="cpu", weights_only=False)
        same_ids = [r["stable_image_id"] for r in rows][: int(a0_raw["pred_logits"].shape[0])]
        index = [theirs["image_ids"].index(image_id) for image_id in same_ids]
        index_tensor = torch.tensor(index, dtype=torch.long)
        other_logits = theirs["pred_logits"][index_tensor]
        other_boxes = theirs["pred_boxes"][index_tensor]
        logits_diff = (a0_raw["pred_logits"] - other_logits).abs()
        boxes_diff = (a0_raw["pred_boxes"] - other_boxes).abs()
        predictions = json.loads((out / "A0" / ("pred-top%d.json" % budgets[0])).read_text())
        certified = json.loads(certified_pred.read_text())
        exact = sum(1 for a, b in zip(predictions, certified)
                    if a["image_id"] == b["image_id"] and a["query_id"] == b["query_id"]
                    and a["category_id"] == b["category_id"] and a["score"] == b["score"]
                    and [float(v) for v in a["bbox_xyxy"]] == [float(v) for v in b["bbox_xyxy"]])
        cache_report = {"available": True,
                        "cache": {"path": str(cache_path), "sha256": sha(cache_path)},
                        "raw_queries_bitwise_equal": bool(torch.equal(a0_raw["pred_logits"],
                                                                      other_logits)),
                        "raw_queries_max_abs_diff": float(logits_diff.max()),
                        "raw_boxes_bitwise_equal": bool(torch.equal(a0_raw["pred_boxes"], other_boxes)),
                        "raw_boxes_max_abs_diff": float(boxes_diff.max()),
                        "entries": int(logits_diff.numel()),
                        "prediction_entries_fieldwise_identical": exact,
                        "prediction_entries": len(predictions),
                        "note": ("cross-run comparison against the 2026-10-14 certified cache and "
                                 "prediction file; last-bit differences are recorded, not repaired")}

    cross_device = {"available": False}
    candidates = [out / "cpu-check", out.parent / "cpu-check"]
    cpu_dir = next((path for path in candidates if path.is_dir()), None)
    if cpu_dir is not None:
        pairs = {}
        for arm, directory in (("A0", out / "A0"), ("R23", out / "FLIP" / "R23")):
            path = cpu_dir / arm / ("pred-top%d.json" % budgets[0])
            if not path.is_file() or not directory.is_dir():
                continue
            cpu_rows = json.loads(path.read_text())
            gpu_rows = json.loads((directory / ("pred-top%d.json" % budgets[0])).read_text())
            images = sorted({e["image_id"] for e in cpu_rows})
            gpu = {}
            for entry in gpu_rows:
                if entry["image_id"] in images:
                    gpu.setdefault(entry["image_id"], []).append(entry)
            matched = only_cpu = only_gpu = 0
            max_score = max_box = 0.0
            for image_id in images:
                left = sorted([e for e in cpu_rows if e["image_id"] == image_id],
                              key=lambda e: -e["score"])
                right = sorted(gpu.get(image_id, []), key=lambda e: -e["score"])
                i = j = 0
                while i < len(left) and j < len(right):
                    score_ok = abs(left[i]["score"] - right[j]["score"]) <= 1e-5
                    box_ok = max(abs(a - b) for a, b in
                                 zip(left[i]["bbox_xyxy"], right[j]["bbox_xyxy"])) <= 0.05
                    if score_ok and box_ok and left[i]["category_id"] == right[j]["category_id"]:
                        matched += 1
                        max_score = max(max_score, abs(left[i]["score"] - right[j]["score"]))
                        max_box = max(max_box, max(abs(a - b) for a, b in
                                                   zip(left[i]["bbox_xyxy"], right[j]["bbox_xyxy"])))
                        i += 1
                        j += 1
                    elif left[i]["score"] > right[j]["score"]:
                        only_cpu += 1
                        i += 1
                    else:
                        only_gpu += 1
                        j += 1
                only_cpu += len(left) - i
                only_gpu += len(right) - j
            pairs[arm] = {"images": len(images), "matched": matched, "only_cpu": only_cpu,
                          "only_gpu": only_gpu, "max_abs_score_diff": max_score,
                          "max_abs_box_diff_px": max_box}
        cross_device = {"available": True, "directory": str(cpu_dir), "pairs": pairs,
                        "note": ("CPU with real parent EMA weights vs GPU, content matching per class, "
                                 "score tolerance 1e-5, box tolerance 0.05 px")}

    def deltas_of(arm):
        return {str(budget): {k: (scores[arm][str(budget)][k] - scores["A0"][str(budget)][k]) * 100.0
                              for k in METRIC_KEYS} for budget in budgets}

    deltas = deltas_of("R23")
    deltas_control = deltas_of("R23C")
    spec = plan["gates"]["R23_minus_A0"]
    conditions = [(spec["metric"], spec["budget"], spec["op"], spec["threshold_pp"]),
                  (spec["and"]["metric"], spec["and"]["budget"], spec["and"]["op"],
                   spec["and"]["threshold_pp"])]
    conditions += [(c["metric"], c["budget"], c["op"], c["threshold_pp"]) for c in spec["protection"]]
    gate_rows, gate_ok = [], True
    for metric, budget, op, threshold in conditions:
        value = deltas[str(budget)][metric]
        passed = value >= threshold if op == ">=" else value <= threshold
        gate_ok = gate_ok and passed
        gate_rows.append({"metric": metric, "budget": budget, "op": op, "threshold": threshold,
                          "value": value, "passed": passed})
    timing, stages = {}, {}
    for arm in ("A0", "FLIP"):
        result = out / arm / ("RESULT-%s.json" % arm)
        if result.is_file():
            payload = json.loads(result.read_text())
            timing[arm] = payload["timing_total_seconds"]
            stages[arm] = payload["stage_seconds"]
    ratio = (timing["FLIP"] / timing["A0"]) if timing.get("A0") else None
    mechanism_seconds = None
    if stages.get("FLIP") and timing.get("A0"):
        flip = stages["FLIP"]
        mechanism_seconds = flip["original_forward"] + flip["mirrored_forward"] + flip["pair_best"] \
            + flip["r23_build"]
    cost = {"seconds": timing, "ratio_FLIP_over_A0": ratio,
            "stage_seconds": stages,
            "mechanism_only_seconds_estimate": mechanism_seconds,
            "mechanism_only_ratio_estimate": (mechanism_seconds / timing["A0"])
            if mechanism_seconds is not None and timing.get("A0") else None,
            "maximum_ratio": plan["gates"]["cost"]["maximum"],
            "passed": bool(ratio is not None and ratio <= plan["gates"]["cost"]["maximum"]),
            "convention": ("end to end, from the original image byte read to predictions materialized, "
                           "including decode/resize/H2D/forwards/D2H and, in the FLIP pass, both "
                           "pairings and all list builds; the strict FLIP/A0 ratio is the gate, the "
                           "mechanism-only estimate excludes the identity reference build, the "
                           "control pairing and the control list build")}
    quality = (plan["decision_rule"]["net_and_protection"] if gate_ok else
               plan["decision_rule"]["net_only"] if gate_rows[0]["passed"] else
               plan["decision_rule"]["failed"])
    comparison = {
        "task_id": plan["task_id"],
        "scores": scores,
        "scores_coco_stats_form": stats_form,
        "differences_pp": deltas,
        "differences_pp_control_R23C": deltas_control,
        "units": "pp = 100 x (arm - A0); aggregation = class-equal-weight",
        "gates": {"R23_minus_A0": {"passed": gate_ok, "detail": gate_rows}, "cost": cost},
        "per_class_AP": per_class,
        "spatial_filter_counts": filter_counts,
        "anchor_reproduction": anchor,
        "first_stage_vs_native": first_stage,
        "cache_reconstruction": cache_report,
        "cross_device": cross_device,
        "quality": quality,
    }
    (out / "COMPARISON.json").write_text(json.dumps(comparison, indent=1, ensure_ascii=False) + "\n")
    lines = ["comparison,budget,metric,delta_pp"]
    for label, table in (("R23_minus_A0", deltas), ("R23C_minus_A0", deltas_control)):
        for budget, values in table.items():
            for metric, value in values.items():
                lines.append("%s,%s,%s,%s" % (label, budget, metric, value))
    for gate in gate_rows:
        lines.append("gate,%s,%s,%s" % (gate["budget"], gate["metric"],
                                        "passed" if gate["passed"] else "failed"))
    lines.append("cost,-,ratio_FLIP_over_A0,%s" % ratio)
    (out / "TABLE.csv").write_text("\n".join(lines) + "\n")
    print(json.dumps({"quality": quality, "gate_passed": gate_ok, "cost": cost, "deltas_pp": deltas,
                      "deltas_pp_control": deltas_control, "anchor_max_abs_diff": anchor["max_abs_diff"],
                      "first_stage": first_stage, "cache_reconstruction": cache_report,
                      "cross_device": cross_device}, indent=1, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--arm", choices=("A0", "FLIP"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--cpu-check", action="store_true")
    parser.add_argument("--score", action="store_true")
    args = parser.parse_args()
    plan = load_plan(args.plan)
    import_chain(plan)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    contract = json.loads(Path(plan["runner"]["contract"]).read_text(encoding="utf-8"))
    if args.score:
        score_only(plan, contract, out)
        return
    require(args.arm in ("A0", "FLIP"), "--arm is required for an inference run")
    if args.cpu_check:
        model, device, runtime = cpu_parent_model(contract)
        batch_size = 1
        torch.set_num_threads(8)
    else:
        model, device, runtime = admit_cuda(contract, out, plan)
        batch_size = int(plan["runner"]["batch_size"])
    print(json.dumps({"arm": args.arm, "device": device, "batch_size": batch_size,
                      "runtime_scope": runtime.get("scope") if isinstance(runtime, dict) else None,
                      "images": args.limit or plan["evaluation"]["images"]}, indent=1), flush=True)
    run_arm(args, plan, contract, model, device, batch_size)


if __name__ == "__main__":
    main()
