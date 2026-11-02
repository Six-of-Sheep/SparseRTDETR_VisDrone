"""r22 production entry: plan-driven, identity-bound, inference only.

Arms
  A0    the pristine certified path: one whole-image forward per batch, no instrumentation at all;
        predictions are the certified ranking of the 300 queries (rank_candidates) at both budgets.
  E600  the same forward with a read-only capture, plus ONE extra decoder pass over the encoder
        top-600 anchors.  Only the NEW 300 queries are admitted, after a same-class IoU 0.7 dedup
        against the certified 600-name prefix of the same image.  Every certified entry keeps its
        score and box; no score, box, category or ordering rule of the first stage is modified.

The certified evaluation protocol is used unchanged for the main comparison: the category-0 raster
ignore filter is applied AFTER the budget truncation and nothing is refilled.  The 300-budget list of
each arm is materialised on its own; for E600 it is NOT the prefix of the 600-budget list, because the
dedup reference is the certified 600-name prefix while the competing set is the certified 300-name
prefix.

Reports written by --score
  anchor_reproduction      score the certified prediction file here and compare with the published
                           top600-metrics.json block (class-equal-weight aggregation)
  first_stage_vs_native    A0's raw queries vs E600's captured first stage, same device, same session
  cache_reconstruction     A0's raw queries and derived predictions vs the certified 2026-10-14 cache
  cross_device             the cpu-check dumps vs the GPU dumps for the same leading images
  gates / cost / per_class / deltas

usage: run_expand600.py --plan <plan.json> --arm A0|E600 --out <dir> [--limit N] [--cpu-check]
       run_expand600.py --plan <plan.json> --score --out <round_dir>/run-001
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

from tools.expand600.expand600 import (DEDUP_IOU, NEW_QUERY_BASE, NUM_CLASSES, NUM_NEW, NUM_QUERIES,
                                       capture_selection, dedup_new_names, find_decoder_module,
                                       gt_potential, merge_arm, new_query_selection,
                                       per_class_macro_metrics, per_class_values, query_names,
                                       select_indices, second_stage)

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


def run_arm(args, plan, contract, model, device, batch_size):
    from sparse_rtdetr.r0 import evaluation as ev
    from sparse_rtdetr.protocol_r18 import slice_ft_job as job
    from sparse_rtdetr.protocol_r18.geometry import forward_guards
    out = Path(args.out)
    data = job.development(contract)
    records_all = data.records
    rows = records_all if not args.limit else records_all[: args.limit]
    require(len(rows) > 0, "no images selected")
    identity = native_identity(model)
    digest_before = state_digest(model)
    decoder = find_decoder_module(model)
    budget_main, budget_secondary = plan["runner"]["budgets"]
    device_type = "cuda" if device.startswith("cuda") else "cpu"

    main_lists, secondary_lists, raw_q300, raw_new, timings, image_records = [], [], [], [], [], []
    new_anchor_indices = []
    counters = {"A0_300_equals_600_prefix": 0, "A0_300_prefix_violations": 0,
                "E600_mutated_main": 0, "E600_mutated_secondary": 0,
                "captured_topk_mismatch_batches": 0}
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start + batch_size]
            begin = time.perf_counter()
            pixels = [data.item(row, position)[0] for position, row in enumerate(chunk, start)]
            x = torch.stack(pixels).to(device)
            captured = None
            second_seconds = 0.0
            with torch.autocast(device_type, enabled=False), forward_guards(model):
                if args.arm == "A0":
                    z = model(x)
                else:
                    with capture_selection(decoder) as live:
                        z = model(x)
                    captured = live
                    if captured.get("recomputed_matches_production") is not True:
                        counters["captured_topk_mismatch_batches"] += 1
                        require(False, "captured top-k gather differs from the production gather")
                    second_begin = time.perf_counter()
                    ind300 = select_indices(captured["enc_logits"], NUM_QUERIES)
                    ind600 = select_indices(captured["enc_logits"], NUM_QUERIES + NUM_NEW)
                    selection = new_query_selection(ind300, ind600, captured["enc_logits"])
                    boxes600, logits600 = second_stage(decoder, captured, ind600)
                    if device_type == "cuda":
                        torch.cuda.synchronize()
                    second_seconds = time.perf_counter() - second_begin
            logits = z["pred_logits"].detach().float().cpu()
            boxes = z["pred_boxes"].detach().float().cpu()
            if captured is not None:
                valid = decoder.valid_mask.detach().cpu().reshape(-1)
                boxes600 = boxes600.detach().float().cpu()
                logits600 = logits600.detach().float().cpu()
                legal = []
                for j in range(len(chunk)):
                    index = torch.tensor(selection["anchors"][j], dtype=torch.long)
                    legal.append(bool(valid[index].all()))
            else:
                boxes600 = logits600 = None
                legal = [None] * len(chunk)

            for j, row in enumerate(chunk):
                require(logits[j].dtype == torch.float32 and boxes[j].dtype == torch.float32,
                        "prediction dtype is not FP32")
                a0_main = ev.rank_candidates(logits[j].contiguous(), boxes[j].contiguous(), row,
                                             budget_main)
                a0_secondary = ev.rank_candidates(logits[j].contiguous(), boxes[j].contiguous(), row,
                                                  budget_secondary)
                raw_q300.append(logits[j].clone())
                if a0_main[:len(a0_secondary)] == a0_secondary:
                    counters["A0_300_equals_600_prefix"] += 1
                else:
                    counters["A0_300_prefix_violations"] += 1
                if args.arm == "A0":
                    main_lists.append(a0_main)
                    secondary_lists.append(a0_secondary)
                    continue

                names = query_names(boxes600[j], logits600[j], selection["positions"][j],
                                    row["width"], row["height"])
                kept, dropped = dedup_new_names(names, a0_main, DEDUP_IOU)
                main_list, displaced_main, _key = merge_arm(a0_main, kept, budget_main)
                secondary_list, displaced_secondary, _key = merge_arm(a0_secondary, kept,
                                                                      budget_secondary)
                raw_new.append([logits600[j].clone(), boxes600[j].clone()])
                new_anchor_indices.append(list(selection["anchors"][j]))
                added_main = [e for e in main_list if e["source"] == "new_query"]
                added_secondary = [e for e in secondary_list if e["source"] == "new_query"]
                mutated = 0
                for reference, target in ((a0_main, main_list), (a0_secondary, secondary_list)):
                    lookup = {(int(e["query_id"]), int(e["category_id"])):
                              (e["score"], tuple(e["bbox_xyxy"])) for e in reference}
                    for entry in target:
                        if entry["source"] != "first_stage":
                            continue
                        want = lookup.get((entry["query_id"], entry["category_id"]))
                        if want is None or want[0] != entry["score"] \
                                or want[1] != tuple(entry["bbox_xyxy"]):
                            mutated += 1
                counters["E600_mutated_main"] += 0
                counters["E600_mutated_secondary"] += mutated
                require(mutated == 0, "a certified entry changed score or box in an E600 list")
                lowest_a0 = min(float(e["score"]) for e in a0_main)
                main_lists.append(main_list)
                secondary_lists.append(secondary_list)
                raw_rows = data.raw(row)["rows"]
                image_records.append({
                    "image_id": row["stable_image_id"],
                    "new_anchors": int(len(selection["anchors"][j])),
                    "new_anchors_all_legal": legal[j],
                    "selection_counters": selection["counters"][j],
                    "new_names": len(names),
                    "new_names_above_lowest_a0": int(sum(1 for item in names if item[2] > lowest_a0)),
                    "dedup_kept": len(kept), "dedup_dropped": len(dropped),
                    "dedup_dropped_iou_max": max((d["best_iou"] for d in dropped), default=0.0),
                    "added_at_600": len(added_main), "displaced_at_600": len(displaced_main),
                    "added_at_300": len(added_secondary),
                    "displaced_at_300": len(displaced_secondary),
                    "added_at_600_gt_reach_0.5": gt_potential(added_main, raw_rows, 0.5),
                    "added_at_600_gt_reach_0.75": gt_potential(added_main, raw_rows, 0.75),
                    "new_names_gt_reach_0.5": gt_potential(
                        [{"category_id": item[1], "bbox_xyxy": item[3]} for item in names],
                        raw_rows, 0.5),
                })
            if device_type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - begin
            timings.append({"batch": len(timings) + 1, "images": len(chunk), "seconds": elapsed,
                            "second_stage_seconds": second_seconds})
            print("batch %d/%d images=%d %.3fs" % (len(timings),
                                                   (len(rows) + batch_size - 1) // batch_size,
                                                   len(chunk), elapsed), flush=True)

    digest_after = state_digest(model)
    flat_main = trim(main_lists, budget_main)
    flat_secondary = trim(secondary_lists, budget_secondary)
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
        "second_stage_seconds": sum(t["second_stage_seconds"] for t in timings),
        "native_identity": identity,
        "checks": counters,
        "resource_entry": sha(out / "resource-entry.json")
                          if (out / "resource-entry.json").is_file() else None,
    }
    for budget, payload in ((budget_main, flat_main), (budget_secondary, flat_secondary)):
        path = out / ("pred-top%d.json" % budget)
        path.write_text(json.dumps(payload, separators=(",", ":")))
        report["pred-top%d_sha256" % budget] = sha(path)
        report["pred-top%d_bytes" % budget] = path.stat().st_size
    if args.arm == "E600":
        (out / "image-records.json").write_text(json.dumps(image_records, indent=1,
                                                           ensure_ascii=False))
        report["image_records_sha256"] = sha(out / "image-records.json")
        torch.save({"schema": "r22_expand600_new_queries_v1", "query_base": NEW_QUERY_BASE,
                    "image_ids": [row["stable_image_id"] for row in rows],
                    "pred_logits": [item[0] for item in raw_new],
                    "pred_boxes": [item[1] for item in raw_new],
                    "anchor_indices": new_anchor_indices}, out / "new-queries.pth")
    torch.save({"schema": "r22_expand600_q300_v1", "arm": args.arm,
                "image_ids": [row["stable_image_id"] for row in rows],
                "pred_logits": torch.stack(raw_q300), "query_ids": torch.arange(NUM_QUERIES)},
               out / "raw-queries-q300.pth")
    (out / ("RESULT-%s.json" % args.arm)).write_text(json.dumps(report, indent=1,
                                                               ensure_ascii=False) + "\n")
    print(json.dumps({k: report[k] for k in ("arm", "images", "state_exact_equal", "params_have_grad",
                                             "timing_total_seconds", "second_stage_seconds",
                                             "checks")}))
    print("WROTE", out / ("RESULT-%s.json" % args.arm))


def score_only(plan, contract, out):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
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

    scores, stats_form, per_class, filter_counts = {}, {}, {}, {}
    for arm in ("A0", "E600"):
        directory = out / arm
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

    first_stage = {"available": False}
    a0_raw, e600_raw = out / "A0" / "raw-queries-q300.pth", out / "E600" / "raw-queries-q300.pth"
    if a0_raw.is_file() and e600_raw.is_file():
        left = torch.load(a0_raw, map_location="cpu", weights_only=False)
        right = torch.load(e600_raw, map_location="cpu", weights_only=False)
        diff = (left["pred_logits"] - right["pred_logits"]).abs()
        first_stage = {"available": True,
                       "bitwise_equal": bool(torch.equal(left["pred_logits"], right["pred_logits"])),
                       "max_abs_diff": float(diff.max()),
                       "nonidentical_entries": int((diff > 0).sum()),
                       "entries": int(diff.numel()),
                       "A0_sha256": sha(a0_raw), "E600_sha256": sha(e600_raw),
                       "note": ("A0 is the pristine native forward, E600 the instrumented one, same "
                                "device and session; any difference is an instrumentation artefact")}

    cache_report = {"available": False}
    cache_path = Path(plan["runner"]["certified_cache"])
    if a0_raw.is_file() and cache_path.is_file():
        mine = torch.load(a0_raw, map_location="cpu", weights_only=False)
        theirs = torch.load(cache_path, map_location="cpu", weights_only=False)
        same_ids = [r["stable_image_id"] for r in rows][: int(mine["pred_logits"].shape[0])]
        index = [theirs["image_ids"].index(image_id) for image_id in same_ids]
        other = theirs["pred_logits"][torch.tensor(index, dtype=torch.long)]
        diff = (mine["pred_logits"] - other).abs()
        predictions = json.loads((out / "A0" / ("pred-top%d.json" % budgets[0])).read_text())
        certified = json.loads(certified_pred.read_text())
        exact = sum(1 for a, b in zip(predictions, certified)
                    if a["image_id"] == b["image_id"] and a["query_id"] == b["query_id"]
                    and a["category_id"] == b["category_id"] and a["score"] == b["score"]
                    and [float(v) for v in a["bbox_xyxy"]] == [float(v) for v in b["bbox_xyxy"]])
        cache_report = {"available": True,
                        "cache": {"path": str(cache_path), "sha256": sha(cache_path)},
                        "raw_queries_bitwise_equal": bool(torch.equal(mine["pred_logits"], other)),
                        "raw_queries_max_abs_diff": float(diff.max()),
                        "raw_queries_nonidentical_entries": int((diff > 0).sum()),
                        "raw_queries_entries": int(diff.numel()),
                        "prediction_entries_fieldwise_identical": exact,
                        "prediction_entries": len(predictions),
                        "note": ("cross-run comparison against the 2026-10-14 certified cache and "
                                 "prediction file; last-bit differences are recorded, not repaired")}

    cross_device = {"available": False}
    candidates = [out / "cpu-check", out.parent / "cpu-check"]
    cpu_dir = next((path for path in candidates if path.is_dir()), None)
    if cpu_dir is not None:
        pairs = {}
        for arm in ("A0", "E600"):
            path = cpu_dir / arm / ("pred-top%d.json" % budgets[0])
            if not path.is_file() or not (out / arm).is_dir():
                continue
            cpu_rows = json.loads(path.read_text())
            gpu_rows = json.loads((out / arm / ("pred-top%d.json" % budgets[0])).read_text())
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

    deltas = {str(budget): {k: (scores["E600"][str(budget)][k] - scores["A0"][str(budget)][k]) * 100.0
                            for k in METRIC_KEYS} for budget in budgets}
    spec = plan["gates"]["E600_minus_A0"]
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
    timing = {}
    for arm in ("A0", "E600"):
        result = out / arm / ("RESULT-%s.json" % arm)
        if result.is_file():
            timing[arm] = json.loads(result.read_text())["timing_total_seconds"]
    ratio = (timing["E600"] / timing["A0"]) if timing.get("A0") else None
    cost = {"seconds": timing, "ratio_E600_over_A0": ratio,
            "maximum_ratio": plan["gates"]["cost"]["maximum"],
            "passed": bool(ratio is not None and ratio <= plan["gates"]["cost"]["maximum"])}
    quality = (plan["decision_rule"]["net_and_protection"] if gate_ok else
               plan["decision_rule"]["net_only"] if gate_rows[0]["passed"] else
               plan["decision_rule"]["failed"])
    comparison = {
        "task_id": plan["task_id"],
        "scores": scores,
        "scores_coco_stats_form": stats_form,
        "differences_pp": deltas,
        "units": "pp = 100 x (E600 - A0); aggregation = class-equal-weight",
        "gates": {"E600_minus_A0": {"passed": gate_ok, "detail": gate_rows}, "cost": cost},
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
    for budget, values in deltas.items():
        for metric, value in values.items():
            lines.append("E600_minus_A0,%s,%s,%s" % (budget, metric, value))
    for gate in gate_rows:
        lines.append("gate,%s,%s,%s" % (gate["budget"], gate["metric"],
                                        "passed" if gate["passed"] else "failed"))
    lines.append("cost,-,ratio_E600_over_A0,%s" % ratio)
    (out / "TABLE.csv").write_text("\n".join(lines) + "\n")
    print(json.dumps({"quality": quality, "gate_passed": gate_ok, "cost": cost, "deltas_pp": deltas,
                      "anchor_max_abs_diff": anchor["max_abs_diff"], "first_stage": first_stage,
                      "cache_reconstruction": cache_report, "cross_device": cross_device},
                     indent=1, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--arm", choices=("A0", "E600"))
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
    require(args.arm in ("A0", "E600"), "--arm is required for an inference run")
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
