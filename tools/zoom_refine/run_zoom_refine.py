"""r20 production entry: plan-driven windowed zoom-refine inference and fixed scoring.

Arms (one window per image, 2x, substitution only):
  baseline  whole-image pass only; predictions are the certified ranking of the batch outputs
  control   whole pass, then the planned window rendered from the base canvas region enlarged
            to the full canvas (no original-resolution information), mapped back and substituted
  method    whole pass, then the planned window rendered from true raw pixels at 2x, mapped
            back and substituted

Both window arms share one geometry: plan -> window rect -> tensor of the fixed canvas ->
inverse map raw = origin + canvas_fraction * window_size.  Only the pixel source differs.

usage: run_zoom_refine.py --plan <plan.json> --arm control|method|baseline --out <dir> [--limit N] [--dry-run]
       run_zoom_refine.py --plan <plan.json> --score --out <round_out_dir>
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch


def load_plan(path):
    plan = json.loads(Path(path).read_text())
    for key in ("runner", "windows", "arms", "gates", "band_area_px", "match_iou_min", "evaluation"):
        if key not in plan:
            raise SystemExit(f"plan is missing {key}")
    return plan


def import_chain(plan, worktree):
    runner = plan["runner"]
    root = Path(runner["worktree_root"]) if worktree is None else Path(worktree)
    src = Path(runner["release_source"])
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(src / "src"))
    return root, src


def map_window_boxes(boxes_canvas, window):
    """Single inverse map for both arms: canvas fraction -> raw pixels inside the window rect."""
    out = []
    for x0, y0, x1, y1 in boxes_canvas:
        out.append((window.x0 + x0 / 1344.0 * window.width, window.y0 + y0 / 768.0 * window.height,
                    window.x0 + x1 / 1344.0 * window.width, window.y0 + y1 / 768.0 * window.height))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", choices=("baseline", "control", "method"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--cpu-parent", action="store_true",
                    help="validation mode: real parent EMA weights on CPU, no CUDA, receipt still written")
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()
    plan = load_plan(args.plan)
    root, src = import_chain(plan, None if not args.dry_run else plan["runner"]["worktree_root"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from sparse_rtdetr.r0 import evaluation as ev
    from sparse_rtdetr.protocol_r18 import slice_ft_job as job
    from sparse_rtdetr.protocol_r18.geometry import forward_guards
    from tools.zoom_refine.zoom_refine import (WINDOW_RAW_HW, invariants, plan_windows,
                                               substitute, window_pixels, control_pixels)

    contract = json.loads(Path(plan["runner"]["contract"]).read_text())
    if args.score:
        score_only(plan, contract, out, ev)
        return

    data = job.development(contract)
    rows = data.records
    if args.limit:
        rows = rows[: args.limit]
    device = "cpu" if (args.dry_run or args.cpu_parent) else "cuda:0"
    if args.dry_run:
        from sparse_rtdetr.protocol_r18.model import Config, build
        model, _criterion, _info = build(Config(), synthetic=True)
        model = model.to(device).eval()
    elif args.cpu_parent:
        from sparse_rtdetr.protocol_r18.model import Config, build
        from sparse_rtdetr.protocol_r18 import slice_finetune as ft
        from sparse_rtdetr.protocol_r18 import slice_ft_execution as execution
        from sparse_rtdetr.protocol_r18 import checkpoint as native_state
        model, _criterion, _info = build(Config(), synthetic=True)
        binding = execution.verified(contract["parent_binding"])
        ref = contract["parent_checkpoint"]
        payload = native_state.read(ref["path"], ref["sha256"], binding)
        loaded = ft.initialize_from_parent(model, payload, expected_binding=binding)
        del payload
        model = model.to("cpu").eval()
        if loaded != contract["initial_model_sha256"]:
            raise SystemExit("cpu-parent initial model digest differs from the contract")
    else:
        # certified CUDA admission: resource sample, boot/storage identity, then the
        # deterministic backend policy is fixed before any CUDA initialization
        from sparse_rtdetr.protocol_r18.control import install_resources
        from sparse_rtdetr.localization import guard, worker as base
        want = contract["gpu_uuid"]
        if os.environ.get("CUDA_VISIBLE_DEVICES") != want:
            raise SystemExit("CUDA_VISIBLE_DEVICES must equal the frozen GPU UUID at process startup")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise SystemExit("CUBLAS_WORKSPACE_CONFIG must be :4096:8 at process startup")
        install_resources()
        sample = guard.snapshot(str(out))
        (out / "resource-entry.json").write_text(json.dumps(sample, ensure_ascii=False) + "\n")
        # the certified worker reads the fresh on-disk sample when no runtime channel is bound
        admission = {k: v for k, v in contract.items() if k != "runtime_root"}
        runtime = base.prepare_cuda(admission, out)
        model, _criterion = job.native_twin(contract, out)
        model = model.eval()
    digest_before = torch_state_digest(model)

    budget = int(plan["evaluation"]["budgets"][0])
    per_image_windows = int(plan["windows"]["per_image"])
    window_candidates = int(plan["runner"]["window_candidates"])
    batch_size = int(plan["runner"]["batch_size"])
    records, predictions, plans, substitutions, timings = [], [], [], [], []
    whole_sets, repaired_sets = [], []

    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start + batch_size]
            t0 = time.perf_counter()
            canvases, whole = [], []
            for position, row in enumerate(chunk, start):
                pixels, _raw = data.item(row, position)
                canvases.append(pixels)
                whole.append(None)
            x = torch.stack(canvases).to(device)
            with torch.autocast(device, enabled=False), forward_guards(model):
                z = model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            logits = z["pred_logits"].detach().float().cpu()
            boxes = z["pred_boxes"].detach().float().cpu()
            per_image = []
            for j, row in enumerate(chunk):
                preds = ev.rank_candidates(logits[j], boxes[j], row, budget)
                per_image.append(preds)
            whole_image = [list(image_preds) for image_preds in per_image]   # pre-substitution snapshot

            window_tensors, window_index = [], []
            if args.arm in ("control", "method"):
                for j, row in enumerate(chunk):
                    windows, stats = plan_windows(int(row["width"]), int(row["height"]), per_image[j],
                                                  windows=per_image_windows, window_hw=WINDOW_RAW_HW)
                    plans.append({"image_id": row["stable_image_id"], "stats": stats,
                                  "windows": [w.as_dict() for w in windows]})
                    for window in windows:
                        if args.arm == "control":
                            window_tensors.append(control_pixels(canvases[j], window))
                        else:
                            from infer_seed import load_original
                            image = load_original(data, row)
                            try:
                                window_tensors.append(window_pixels(image, window))
                            finally:
                                image.close()
                        window_index.append((j, window))
                if window_tensors:
                    win_logits, win_boxes = [], []
                    for k in range(0, len(window_tensors), batch_size):
                        wx = torch.stack(window_tensors[k:k + batch_size]).to(device)
                        with torch.autocast(device, enabled=False), forward_guards(model):
                            wz = model(wx)
                        if device == "cuda":
                            torch.cuda.synchronize()
                        win_logits.append(wz["pred_logits"].detach().float().cpu())
                        win_boxes.append(wz["pred_boxes"].detach().float().cpu())
                    win_logits = torch.cat(win_logits)
                    win_boxes = torch.cat(win_boxes)
                else:
                    win_logits = torch.zeros(0, 300, 10)
                    win_boxes = torch.zeros(0, 300, 4)

                per_image_window_dets = {}
                for index, (j, window) in enumerate(window_index):
                    row = chunk[j]
                    boxes_canvas = []
                    scores, queries, categories = [], [], []
                    lg, bx = win_logits[index], win_boxes[index]
                    xy = torch.cat((bx[:, :2] - bx[:, 2:] / 2, bx[:, :2] + bx[:, 2:] / 2), -1)
                    # model boxes are normalised canvas cxcywh; map_window_boxes takes canvas pixels
                    xy = xy * torch.tensor([1344.0, 768.0, 1344.0, 768.0])
                    scores_np = lg.sigmoid().reshape(-1).numpy()
                    order = torch.argsort(-torch.tensor(scores_np), stable=True)[:window_candidates]
                    q = torch.div(order, 10, rounding_mode="floor")
                    c = order % 10
                    for rank, flat in enumerate(order.tolist()):
                        boxes_canvas.append([float(v) for v in xy[q[rank]].tolist()])
                        scores.append(float(scores_np[flat]))
                        queries.append(int(q[rank]))
                        categories.append(int(c[rank]) + 1)
                    mapped = map_window_boxes(boxes_canvas, window)
                    per_image_window_dets.setdefault(j, []).extend(
                        [{"category_id": categories[i], "query_id": queries[i], "score": scores[i],
                          "window_id": window.view_id, "bbox_xyxy": list(mapped[i])}
                         for i in range(len(mapped))])

                for j, row in enumerate(chunk):
                    fixed, stats = substitute(per_image[j], per_image_window_dets.get(j, []))
                    stats["image_id"] = row["stable_image_id"]
                    stats["window_dets"] = len(per_image_window_dets.get(j, []))
                    substitutions.append(stats)
                    per_image[j] = fixed
                whole_sets.extend(whole_image)
                repaired_sets.extend(per_image)

            elapsed = time.perf_counter() - t0
            timings.append({"batch": len(timings) + 1, "images": len(chunk), "seconds": elapsed,
                            "ids": [r["stable_image_id"] for r in chunk]})
            for j, row in enumerate(chunk):
                predictions.extend(per_image[j])
                records.append(row["stable_image_id"])

    digest_after = torch_state_digest(model)
    report = {"arm": args.arm, "images": len(rows), "predictions": len(predictions),
              "budget_ranked": budget, "window_candidates": window_candidates,
              "state_exact_equal": digest_before == digest_after,
              "params_have_grad": any(p.grad is not None for p in model.parameters()),
              "cuda_initialized": torch.cuda.is_initialized(),
              "dry_run": bool(args.dry_run),
              "timing_total_seconds": sum(t["seconds"] for t in timings),
              "timing_batches": timings,
              "plan_stats": summarize(plans),
              "substitution_stats": summarize(substitutions)}
    if args.arm == "baseline" and not args.dry_run:
        ref = Path(plan["runner"]["parent_job"]) / "evaluation/top600-predictions.json"
        if ref.is_file():
            certified = json.loads(ref.read_text())
            same = len(certified) == len(predictions) and all(
                c["image_id"] == p["image_id"] and c["query_id"] == p["query_id"]
                and c["category_id"] == p["category_id"] and c["score"] == p["score"]
                and [float(v) for v in c["bbox_xyxy"]] == [float(v) for v in p["bbox_xyxy"]]
                for c, p in zip(certified, predictions))
            report["certified_identity"] = {"path": str(ref), "sha256": sha(ref),
                                            "exact_match": bool(same), "certified_count": len(certified),
                                            "rerun_count": len(predictions)}
    if args.arm in ("control", "method"):
        before = [p for image_preds in whole_sets for p in image_preds]
        after = [p for image_preds in repaired_sets for p in image_preds]
        report["invariants"] = invariants(before, after)
        if not report["invariants"]["ok"]:
            raise SystemExit(f"substitution invariants violated: {report['invariants']}")
    for name, payload in (("pred-top600", predictions), ("pred-top300", trim(predictions, 300))):
        path = out / f"{name}.json"
        path.write_text(json.dumps(payload, separators=(",", ":")))
        report[f"{name}_sha256"] = sha(path)
        report[f"{name}_bytes"] = path.stat().st_size
    if not args.dry_run:
        report["plan_audit"] = plans
        report["substitution_audit"] = substitutions
    (out / f"RESULT-{args.arm}.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: report[k] for k in ("arm", "images", "predictions", "state_exact_equal",
                                             "params_have_grad", "timing_total_seconds")}))
    print("plan_stats", report["plan_stats"])
    print("substitution_stats", report["substitution_stats"])
    print("WROTE", out / f"RESULT-{args.arm}.json")


def trim(predictions, budget):
    out, counts = [], {}
    for p in predictions:
        image = p["image_id"]
        if counts.get(image, 0) < budget:
            out.append(p)
            counts[image] = counts.get(image, 0) + 1
    return out


def summarize(rows):
    """Per-arm aggregates; coverage and repair_share are reported as means, never sums."""
    if not rows:
        return {}
    keys = ("windows", "band_points", "covered", "coverage")
    if "repaired" in rows[0]:
        keys = ("band_considered", "repaired", "repair_share")
    totals = {k: 0.0 for k in keys}
    for row in rows:
        source = row.get("stats", row)
        for k in keys:
            if k in source:
                totals[k] += float(source[k])
    totals["n"] = len(rows)
    for k in ("coverage", "repair_share"):
        if k in totals:
            totals[k] = totals[k] / len(rows)
    if "coverage" in totals or "repair_share" in totals:
        totals["aggregation"] = "mean_per_image"
    if "windows" in keys:
        windows = sum(len(r.get("windows", [])) for r in rows)
        totals["windows"] = windows
        totals["images_with_window"] = sum(1 for r in rows if r.get("windows"))
    return totals


def sha(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def torch_state_digest(model):
    import hashlib
    h = hashlib.sha256()
    for key, value in model.state_dict().items():
        h.update(key.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def score_only(plan, contract, out, ev):
    """Certified scoring of the arm prediction files plus A0 reuse, deltas, gates and cost."""
    from sparse_rtdetr.protocol_r18 import slice_ft_job as job
    root = Path(plan["runner"]["release_source"])
    data = job.development(contract)
    rows, parsed = data.records, [data.raw(r) for r in data.records]
    KEYS = ("AP", "AP50", "AP75", "APS", "APM", "APL", "AP75_small")
    scores, metrics_refs = {}, {}
    for arm in ("A0", "A0r", "A1", "A2"):
        scores[arm] = {}
        for budget in plan["evaluation"]["budgets"]:
            if arm == "A0":
                ref = Path(plan["runner"]["parent_job"]) / f"evaluation/top{budget}-metrics.json"
                metrics = json.loads(ref.read_text())["COCO_bridge"]["metrics"]
                metrics_refs[arm] = {str(budget): {"path": str(ref), "sha256": sha(ref)}}
            else:
                name = {"A0r": "baseline", "A1": "control", "A2": "method"}[arm]
                directory = out / name
                preds = json.loads((directory / f"pred-top{budget}.json").read_text())
                metrics = ev.score_budget(rows, parsed, preds, root)["COCO_bridge"]["metrics"]
                (directory / f"metrics-top{budget}.json").write_text(json.dumps(metrics, indent=1))
            scores[arm][str(budget)] = {k: metrics[k] for k in KEYS}
    invariants_from_artifacts = {}
    for budget in plan["evaluation"]["budgets"]:
        base_path = out / "baseline" / f"pred-top{budget}.json"
        base_preds = json.loads(base_path.read_text()) if base_path.is_file() else None
        per_arm = {}
        for name in ("control", "method"):
            path = out / name / f"pred-top{budget}.json"
            if base_preds is None or not path.is_file():
                per_arm[name] = {"checked": False, "reason": "missing prediction file"}
                continue
            preds = json.loads(path.read_text())
            problem, changed = None, 0
            if len(preds) != len(base_preds):
                problem = f"prediction count differs: {len(preds)} vs {len(base_preds)}"
            else:
                for base_row, arm_row in zip(base_preds, preds):
                    if any(base_row[key] != arm_row[key]
                           for key in ("image_id", "query_id", "category_id", "score")):
                        problem = f"non-geometry field changed at {base_row['image_id']} / {base_row['query_id']}"
                        break
                    if [float(v) for v in base_row["bbox_xyxy"]] != [float(v) for v in arm_row["bbox_xyxy"]]:
                        changed += 1
            per_arm[name] = {"checked": True, "ok": problem is None, "reason": problem,
                             "changed_boxes": changed, "total": len(preds),
                             "baseline": {"path": str(base_path), "sha256": sha(base_path)},
                             "arm": {"path": str(path), "sha256": sha(path)}}
        invariants_from_artifacts[str(budget)] = per_arm
    deltas = {}
    for lhs, rhs in (("A2", "A0"), ("A2", "A1"), ("A1", "A0"), ("A0r", "A0")):
        deltas[f"{lhs}_minus_{rhs}"] = {b: {k: (scores[lhs][b][k] - scores[rhs][b][k]) * 100 for k in KEYS}
                                        for b in scores[lhs]}
    gates = plan["gates"]
    def evaluate(spec, name):
        rows_out, ok = [], True
        conditions = [{"metric": spec["metric"], "budget": spec["budget"], "op": spec["op"], "threshold": spec["threshold_pp"]}] + \
                     [spec["and"]] + [spec["protection"][i] for i in range(len(spec["protection"]))]
        for cond in conditions:
            value = deltas[name][str(cond["budget"])][cond["metric"]]
            passed = value >= cond["threshold"] if cond["op"] == ">=" else value <= cond["threshold"]
            ok = ok and passed
            rows_out.append({"metric": cond["metric"], "budget": cond["budget"], "op": cond["op"],
                             "threshold": cond["threshold"], "value": value, "passed": passed})
        return {"passed": ok, "detail": rows_out}
    costs = {"A0_seconds": None, "A1_seconds": None, "A2_seconds": None}
    for arm, name in (("A0", "baseline"), ("A1", "control"), ("A2", "method")):
        result = out / name / f"RESULT-{name}.json"
        if result.is_file():
            costs[f"{arm}_seconds"] = json.loads(result.read_text())["timing_total_seconds"]
    comparison = {
        "scores": scores, "differences_pp": deltas,
        "gates": {"A2_minus_A0": evaluate(gates["A2_minus_A0"], "A2_minus_A0"),
                  "A2_minus_A1": evaluate(gates["A2_minus_A1"], "A2_minus_A1")},
        "cost": {"seconds": costs,
                 "ratio_A2_over_A0": (costs["A2_seconds"] / costs["A0_seconds"]) if costs["A0_seconds"] else None,
                 "ratio_A1_over_A0": (costs["A1_seconds"] / costs["A0_seconds"]) if costs["A0_seconds"] else None,
                 "maximum_ratio": gates["cost"]["maximum"]},
        "quality": None, "units": "pp = 100 x (lhs - rhs)", "metrics_refs": metrics_refs,
        "substitution_invariants": invariants_from_artifacts,
    }
    main = comparison["gates"]["A2_minus_A0"]["passed"]
    mech = comparison["gates"]["A2_minus_A1"]["passed"]
    comparison["quality"] = ("initial_positive_net_and_increment" if main and mech else
                             "single_seed_preliminary_net_benefit" if main else "closed_fixed_config")
    (out / "COMPARISON.json").write_text(json.dumps(comparison, indent=1, ensure_ascii=False) + "\n")
    lines = ["comparison,budget,metric,delta_pp"]
    for name, per_budget in deltas.items():
        for budget, values in per_budget.items():
            for metric, value in values.items():
                lines.append(f"{name},{budget},{metric},{value}")
    for gate, payload in comparison["gates"].items():
        lines.append(f"gate_{gate},-,{'passed' if payload['passed'] else 'failed'},")
    lines.append(f"cost_A2_over_A0,-,ratio,{comparison['cost']['ratio_A2_over_A0']}")
    (out / "TABLE.csv").write_text("\n".join(lines) + "\n")
    print(json.dumps({"quality": comparison["quality"],
                      "A2_minus_A0": comparison["gates"]["A2_minus_A0"]["passed"],
                      "A2_minus_A1": comparison["gates"]["A2_minus_A1"]["passed"],
                      "cost": comparison["cost"]}, indent=1))
    print(json.dumps({"substitution_invariants": {b: {n: v.get("ok") for n, v in arms.items()}
                                                  for b, arms in invariants_from_artifacts.items()}}, indent=1))
    print(json.dumps(scores, indent=1))


if __name__ == "__main__":
    main()
