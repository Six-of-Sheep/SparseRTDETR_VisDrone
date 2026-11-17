"""TASK_MIX_SCALE_001 fixed e120 EMA readout.

  infer  --canvas H|L   one fresh process per endpoint: 4 warm-up + 137 batches, FP32 batch 4, 548 images
  score                 4 score_budget calls (MIX@H, MIX@L x top600/top300) and one per-class APL COCOeval
                        per endpoint top600 (2 in total); pre-registered decision; COMPARISON.json
  cpu-check             CUDA hidden: references, retention thresholds, endpoint identity/refusals, H/L pixels and
                        evaluation caches, one 4-image CPU forward per canvas, decision branches

Primary endpoint MIX@H (1152x2016); MIX@L (768x1344) is a readout only and never replaces MIX@H.
References are pinned and never re-inferred or re-scored: native A0@L (R25 formal-001), S15@H (S15 formal-002),
S15@L (S15 cross-scale diag-001).  No online endpoint, mid-run score, selection, fusion or threshold change.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools" / "query_local"), str(ROOT / "tools" / "scale_baseline"),
                str(ROOT)]

import torch

from infer_seed import BUDGETS, METRICS, _group, _prefix, bridge_values, load_original, pixels
from sparse_rtdetr.localization.runio import event, publish, read, reference, require
from sparse_rtdetr.protocol_r18 import checkpoint as native
from sparse_rtdetr.protocol_r18.engine import tensor_digest
from sparse_rtdetr.protocol_r18.geometry import forward_guards
from sparse_rtdetr.r0.evaluation import POLICY, rank_candidates, score_budget
from sparse_rtdetr.research import mix_scale as M
from sparse_rtdetr.research import scale_baseline as S

ARM = "MIX"
KIND = "EMA"
BATCH = 4
WARMUPS = 4
ENDPOINT = M.END_UPDATE
PRIMARY = ((600, "APS", 0.50), (600, "AP75_small", 0.30))
PROTECTIONS = ((600, "AP", -0.10), (300, "AP", -0.20), (300, "APS", -0.20), (600, "APM", -0.50), (600, "APL", -0.50))
RETENTION_FRACTION = 0.80
RETENTION = ((600, "APS"), (600, "AP75_small"))
PASS, FAIL = "single_seed_preliminary_scale_tradeoff_reduction", "closed_fixed_config"
EVALUATION = {
    "epoch": 120, "weights": ["ema"], "online": "kept in the complete e120 checkpoint; not inferred or scored",
    "primary_endpoint": "MIX@H 1152x2016", "secondary_endpoint": "MIX@L 768x1344 (readout only)",
    "canvases": {"H": list(M.H_HW), "L": list(M.L_HW)}, "view": "A", "images": 548, "batch": BATCH,
    "precision": "float32", "warmup_batches_per_endpoint": WARMUPS, "steady_batches_per_endpoint": 137,
    "budgets": list(BUDGETS), "score_calls": 4, "per_class_APL_budget": 600,
    "per_class_APL_extra_COCOeval_calls_max_per_endpoint": 1,
}


def _validate_policy(c):
    require(c["workload"]["evaluation"] == EVALUATION, "MIX endpoint workload differs")
    policy = c["evaluation_policy"]
    require(reference(policy["path"]) == policy and read(policy["path"]) == POLICY, "MIX view-A evaluation policy differs")


def development(c, canvas):
    data = S.development(c["data"]["development"], M.CANVAS[canvas])
    require([r["stable_image_id"] for r in data.records] == c["development_ids"], "MIX development members or order")
    return data


def _weights(payload, c, flag):
    require((payload["version"], payload["arm"], payload["parent_sha256"], payload["config"], payload["schedule"],
             payload["activation_checkpoint"]) ==
            (M.VERSION, ARM, c["parent"]["sha256"], M.base_config().binding(), c["schedule"]["identity"], flag),
            "MIX evaluation state identity (version, parent, recipe, frozen table or memory mode)")
    require(payload["engine"]["completed_updates"] == payload["ema"]["updates"] == ENDPOINT,
            "MIX evaluation requires e120 (online and EMA clocks)")
    state = payload["ema"]["module"]
    require(all(not v.is_floating_point() or v.dtype == torch.float32 for v in state.values()),
            "MIX EMA floating state is not FP32")
    return state


def forward_batch(model, data, rows, device):
    device = torch.device(device)
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None)
    start = time.perf_counter()
    images = []
    for row in rows:
        with load_original(data, row) as image:
            images.append(pixels(data, image))
    x = torch.stack(images).to(device)
    sync()
    input_done = time.perf_counter()
    with torch.autocast(device.type, enabled=False), forward_guards(model):
        result = model(x)
    sync()
    forward_done = time.perf_counter()
    require(set(result) == {"pred_boxes", "pred_logits"} and result["pred_boxes"].shape == (len(rows), 300, 4)
            and result["pred_logits"].shape == (len(rows), 300, 10)
            and result["pred_boxes"].dtype == result["pred_logits"].dtype == torch.float32, "MIX endpoint output schema")
    boxes, logits = result["pred_boxes"].cpu(), result["pred_logits"].cpu()
    output_done = time.perf_counter()
    six = [rank_candidates(lg, bx, row, 600) for lg, bx, row in zip(logits, boxes, rows)]
    predictions = {600: [p for g in six for p in g], 300: [p for g in six for p in g[:300]]}
    rank_done = time.perf_counter()
    return predictions, {"data_and_h2d": input_done - start, "forward": forward_done - input_done,
                         "cpu_materialization": output_done - forward_done, "ranking": rank_done - output_done,
                         "total": rank_done - start}


# ---------------------------------------------------------------- inference (one canvas per process)
def infer(c, train_root, canvas, out):
    from mix_formal_run import load_contract
    from mix_formal_train import read_endpoint
    from r25_run import setup
    counts = {"attempted_forwards": 0, "completed_forwards": 0}
    started = time.perf_counter()
    try:
        _validate_policy(c)
        source = load_contract(train_root["contract"])
        require(source["formal_identity"] == c["formal_identity"]
                and source["schedule"]["identity"] == c["schedule"]["identity"], "MIX endpoint formal identity/table")
        path = Path(train_root["path"]) / "checkpoint-epoch-120.pth"
        side = read(path.with_suffix(".pth.json"))
        require((side["status"], side["arm"], side["contract_sha256"], side["formal_identity"], side["global_update"],
                 side["scope"], side["schedule"]) ==
                ("COMPLETE", ARM, source["contract_sha256"], source["formal_identity"], ENDPOINT, "formal",
                 c["schedule"]["identity"]) and Path(side["file"]["path"]).resolve() == path.resolve(),
                "MIX evaluator endpoint sidecar identity")
        setup("cuda:0")
        device = torch.device("cuda:0")
        torch.cuda.reset_peak_memory_stats(device)
        payload = read_endpoint(source, path)
        state = _weights(payload, c, c["workload"]["activation_checkpoint"])
        model, record = M.evaluation_model(state, canvas)
        del payload, state
        gc.collect()
        model.to(device)
        weight_sha = tensor_digest(model.state_dict().items())
        data = development(c, canvas)
        rng = native.capture_rng(device)
        torch.cuda.synchronize(device)
        load_seconds = time.perf_counter() - started

        def batch(rows):
            counts["attempted_forwards"] += 1
            predictions, timing = forward_batch(model, data, rows, device)
            counts["completed_forwards"] += 1
            return predictions, timing

        warmup, measured = [], []
        all_predictions = {b: [] for b in BUDGETS}
        with torch.inference_mode():
            for i in range(WARMUPS):
                _, timing = batch(data.records[i * BATCH:(i + 1) * BATCH])
                warmup.append(timing)
                event(out, "endpoint_warmup", arm=ARM, canvas=canvas, batch=i + 1)
            for offset in range(0, len(data.records), BATCH):
                predictions, timing = batch(data.records[offset:offset + BATCH])
                measured.append(timing)
                for b in BUDGETS:
                    all_predictions[b].extend(predictions[b])
                event(out, "endpoint_batch", arm=ARM, canvas=canvas, batch=offset // BATCH + 1)
        require(len(warmup) == WARMUPS and len(measured) == 137
                and counts["attempted_forwards"] == counts["completed_forwards"] == WARMUPS + 137, "MIX batch count")
        require(tensor_digest(model.state_dict().items()) == weight_sha and native.exact_equal(native.capture_rng(device), rng)
                and all(not m.training for m in model.modules()) and all(p.grad is None for p in model.parameters()),
                "MIX evaluation modified model state or RNG")
        refs = {}
        for b in BUDGETS:
            p = out / f"top{b}-predictions.json"
            publish(p, all_predictions[b])
            refs[str(b)] = reference(p)
        result = {"status": "PASS", "arm": ARM, "kind": KIND, "canvas": canvas, "input_hw": list(M.CANVAS[canvas]),
                  "primary": canvas == "H", "epoch": 120, "update": ENDPOINT, "view": "A", "device": str(device),
                  "FP32": True, "batch_size": BATCH, "contract_sha256": c["contract_sha256"],
                  "formal_identity": c["formal_identity"], "checkpoint": side["file"], "weights_sha256": weight_sha,
                  "evaluation_model": record, "predictions": refs, "images": len(data.records),
                  "steady_forwards": len(measured), "warmup_forwards": len(warmup), "load_seconds": load_seconds,
                  "warmup_seconds": sum(r["total"] for r in warmup), "warmup_timing": warmup,
                  "cold_first_forward_seconds": warmup[0]["forward"],
                  "steady_timing": {k: sum(r[k] for r in measured) for k in measured[0]},
                  "elapsed_seconds": time.perf_counter() - started,
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                  "model_and_RNG_unchanged": True, **counts}
        publish(out / "RESULT.json", result)
        return result
    except BaseException as exc:
        peak = {}
        try:
            peak = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
        except Exception:
            pass
        publish(out / "STOP.json", {"status": "STOP_CUDA_OOM" if isinstance(exc, torch.cuda.OutOfMemoryError) else "STOP",
                                    "phase": f"MIX@{canvas} inference", **counts, **peak,
                                    "elapsed_seconds": time.perf_counter() - started,
                                    "error": str(exc)[:4000], "traceback": traceback.format_exc()})
        raise


# ---------------------------------------------------------------- pinned references
def _pinned(ref):
    require(reference(ref["path"]) == ref, f"pinned reference differs: {ref['path']}")
    return read(ref["path"])


def references(c):
    """A0@L, S15@H, S15@L values (0..1), GT counts and per-class APL; cross-bound to their certified records."""
    r = c["references"]
    values, counts = {}, {}
    for name in ("A0@L", "S15@H", "S15@L"):
        values[name], counts[name] = {}, {}
        for b in BUDGETS:
            m = _pinned(r[name]["metrics"][str(b)])
            values[name][str(b)] = bridge_values(m)
            counts[name][str(b)] = {k: m["spatial_filter_counts"][k] for k in ("GT_before", "GT_after")}
    s15 = _pinned(r["S15_comparison"])
    require(s15["scores"]["A0"]["EMA"] == values["A0@L"] and s15["scores"]["S15"]["EMA"] == values["S15@H"]
            and s15["A0_reference"]["A0_metrics"]["EMA"] == r["A0@L"]["metrics"]
            and s15["metrics"]["EMA"] == r["S15@H"]["metrics"], "S15 formal-002 comparison binding of A0@L / S15@H")
    four = _pinned(r["four_cell"])
    require(four["metrics"]["LL"] == r["A0@L"]["metrics"] and four["metrics"]["HL"] == r["S15@L"]["metrics"]
            and four["metrics"]["HH"] == r["S15@H"]["metrics"], "cross-scale four-cell binding of S15@L")
    require(counts["A0@L"] == counts["S15@H"] == counts["S15@L"], "reference GT sets differ")
    per_class = {"A0@L": {row["category_id"]: row for row in s15["per_class_APL_top600"]["EMA"]["A0"]},
                 "S15@H": {row["category_id"]: row for row in s15["per_class_APL_top600"]["EMA"]["S15"]},
                 "S15@L": {int(k): {"category_id": int(k), "GT": v["GT_large"], "APL": v["HL"] / 100.}
                           for k, v in four["per_class_APL_top600_pp"].items()}}
    for k, v in four["per_class_APL_top600_pp"].items():
        require(abs(v["LL"] / 100. - per_class["A0@L"][int(k)]["APL"]) < 1e-12
                and abs(v["HH"] / 100. - per_class["S15@H"][int(k)]["APL"]) < 1e-12
                and v["GT_large"] == per_class["A0@L"][int(k)]["GT"], "per-class APL reference cross-binding")
    gains = {f"top{b}_{m}": 100. * (values["S15@H"][str(b)][m] - values["A0@L"][str(b)][m]) for b, m in RETENTION}
    return {"values": values, "GT_counts": counts["A0@L"], "per_class_APL_top600": per_class,
            "S15_gain_pp": gains, "retention_required_pp": {k: RETENTION_FRACTION * v for k, v in gains.items()}}


def decide(mix_h, refs):
    """Seven original cells (MIX@H - A0@L) and two retention conditions; unrounded values."""
    a0 = refs["values"]["A0@L"]
    diff = {str(b): {m: 100. * (mix_h[str(b)][m] - a0[str(b)][m]) for m in METRICS} for b in BUDGETS}
    cells = {}
    for b, m, minimum in (*PRIMARY, *PROTECTIONS):
        observed = diff[str(b)][m]
        cells[f"top{b}_{m}"] = {"difference_pp": observed, "minimum_pp": minimum, "passed": observed >= minimum,
                                "role": "primary" if (b, m, minimum) in PRIMARY else "protection"}
    retention = {}
    for b, m in RETENTION:
        key = f"top{b}_{m}"
        observed, required = diff[str(b)][m], refs["retention_required_pp"][key]
        retention[key] = {"MIX_H_minus_A0_L_pp": observed, "S15_H_minus_A0_L_pp": refs["S15_gain_pp"][key],
                          "fraction": RETENTION_FRACTION, "required_pp": required,
                          "retained_fraction": observed / refs["S15_gain_pp"][key], "passed": observed >= required}
    seven = all(v["passed"] for v in cells.values())
    kept = all(v["passed"] for v in retention.values())
    decision = PASS if seven and kept else FAIL
    branch = ("all_seven_and_retention_passed" if seven and kept else
              "seven_cells_passed_retention_insufficient" if seven else
              "retention_passed_original_cell_failed" if kept else "original_cell_and_retention_failed")
    return diff, {"seven_cells": cells, "seven_cells_passed": seven,
                  "failed_seven_cells": [k for k, v in cells.items() if not v["passed"]],
                  "retention": retention, "retention_passed": kept,
                  "failed_retention": [k for k, v in retention.items() if not v["passed"]],
                  "branch": branch}, decision


def _difference(a, b):
    return {str(x): {m: 100. * (a[str(x)][m] - b[str(x)][m]) for m in METRICS} for x in BUDGETS}


# ---------------------------------------------------------------- scoring
def score(c, infer_dirs, out):
    from s15_formal_eval import per_class_large
    started = time.perf_counter()
    _validate_policy(c)
    refs = references(c)
    data = development(c, "L")  # GT/scoring is canvas-free (original-image coordinates)
    rows, ids = data.records, c["development_ids"]
    parsed = [data.raw(row) for row in rows]
    values, metric_refs, calls, per_class, results = {}, {}, [], {}, {}
    for canvas in ("H", "L"):
        result_ref = reference(Path(infer_dirs[canvas]) / "RESULT.json")
        result = read(result_ref["path"])
        require((result["status"], result["canvas"], result["contract_sha256"], result["images"],
                 result["steady_forwards"], result["warmup_forwards"]) ==
                ("PASS", canvas, c["contract_sha256"], 548, 137, WARMUPS), f"MIX@{canvas} inference identity")
        results[canvas] = {"result": result_ref, **{k: result[k] for k in ("weights_sha256", "steady_timing",
                           "load_seconds", "warmup_seconds", "cold_first_forward_seconds", "elapsed_seconds",
                           "peak_allocated_bytes", "peak_reserved_bytes", "evaluation_model", "checkpoint")}}
        preds, grouped = {}, {}
        for b in BUDGETS:
            r = result["predictions"][str(b)]
            require(reference(r["path"]) == r, f"MIX@{canvas} prediction file")
            preds[b] = read(r["path"])
            grouped[b] = _group(preds[b], ids, b, whole=True, exact=True)
        _prefix(grouped[600], grouped[300], ids, whole=True)
        del grouped
        name = f"MIX@{canvas}"
        values[name], metric_refs[name] = {}, {}
        for b in BUDGETS:
            t = time.perf_counter()
            event(out, "score_started", ordinal=len(calls) + 1, endpoint=name, budget=b)
            m = score_budget(rows, parsed, preds[b], c["source_root"])
            m["MIX_binding"] = {"contract_sha256": c["contract_sha256"], "formal_identity": c["formal_identity"],
                                "endpoint": name, "canvas": canvas, "input_hw": list(M.CANVAS[canvas]),
                                "budget": b, "epoch": 120, "kind": KIND, "inference_result": result_ref,
                                "predictions": result["predictions"][str(b)], "weights_sha256": result["weights_sha256"],
                                "table_sha256": c["schedule"]["table_sha256"]}
            gt = {k: m["spatial_filter_counts"][k] for k in ("GT_before", "GT_after")}
            require(gt == refs["GT_counts"][str(b)], f"{name} scored GT set differs from A0: {gt}")
            p = out / f"MIX-{canvas}-EMA-A-top{b}-metrics.json"
            publish(p, m)
            metric_refs[name][str(b)] = reference(p)
            values[name][str(b)] = bridge_values(m)
            calls.append({"endpoint": name, "budget": b, "score_budget_seconds": time.perf_counter() - t,
                          "spatial_filter_counts": m["spatial_filter_counts"]})
            event(out, "score_complete", ordinal=len(calls), endpoint=name, budget=b)
            if b == EVALUATION["per_class_APL_budget"]:
                t = time.perf_counter()
                mine = per_class_large(rows, parsed, preds[b], m["COCO_bridge"])
                a0 = refs["per_class_APL_top600"]["A0@L"]
                require([r["category_id"] for r in mine] == sorted(a0) and all(r["GT"] == a0[r["category_id"]]["GT"]
                                                                              for r in mine),
                        f"{name} per-class GT denominators differ")
                per_class[name] = {"rows": mine, "COCOeval_seconds": time.perf_counter() - t}
        del preds
        gc.collect()
    require(len(calls) == EVALUATION["score_calls"] and len(per_class) == 2, "MIX fixed score/COCOeval count")
    diff, gates, decision = decide(values["MIX@H"], refs)
    allv = {**values, **refs["values"]}
    descriptive = {"MIX@H-A0@L": diff, "MIX@H-S15@H": _difference(allv["MIX@H"], allv["S15@H"]),
                   "MIX@L-A0@L": _difference(allv["MIX@L"], allv["A0@L"]),
                   "MIX@L-S15@L": _difference(allv["MIX@L"], allv["S15@L"]),
                   "MIX@H-MIX@L": _difference(allv["MIX@H"], allv["MIX@L"]),
                   "S15@H-A0@L (reference)": _difference(allv["S15@H"], allv["A0@L"])}
    classes = []
    for row in per_class["MIX@H"]["rows"]:
        k = row["category_id"]
        apl = {"MIX@H": row["APL"],
               "MIX@L": next(r["APL"] for r in per_class["MIX@L"]["rows"] if r["category_id"] == k),
               **{n: refs["per_class_APL_top600"][n][k]["APL"] for n in ("A0@L", "S15@H", "S15@L")}}
        pp = lambda a, b: None if apl[a] is None or apl[b] is None else 100. * (apl[a] - apl[b])
        classes.append({"category_id": k, "GT_large": row["GT"], "APL": apl,
                        "MIX@H-A0@L_pp": pp("MIX@H", "A0@L"), "MIX@H-S15@H_pp": pp("MIX@H", "S15@H"),
                        "MIX@L-A0@L_pp": pp("MIX@L", "A0@L"), "MIX@L-S15@L_pp": pp("MIX@L", "S15@L"),
                        "macro_contribution_MIX@H-A0@L_pp": None if pp("MIX@H", "A0@L") is None
                        else pp("MIX@H", "A0@L") / 10.})
    comparison = {"status": "PASS", "scientific_decision": decision, "decision_rule": c["decision_rule"],
                  "primary": "MIX@H e120 EMA minus native A0@L e120 EMA (seed0)", "gates": gates,
                  "secondary_readout_never_changes_decision": "MIX@L",
                  "scores": allv, "differences_pp": descriptive, "per_class_APL_top600": classes,
                  "per_class_COCOeval_seconds": {k: v["COCOeval_seconds"] for k, v in per_class.items()},
                  "GT_counts": refs["GT_counts"], "S15_gain_pp": refs["S15_gain_pp"],
                  "retention_required_pp": refs["retention_required_pp"],
                  "score_units": "0..1 COCO bridge", "delta_units": "percentage points", "metrics": metric_refs,
                  "references": c["references"], "inference": results, "scoring": calls,
                  "score_calls": len(calls), "extra_COCOeval_calls": len(per_class),
                  "scoring_wall_seconds": time.perf_counter() - started,
                  "contract_sha256": c["contract_sha256"], "formal_identity": c["formal_identity"]}
    publish(out / "COMPARISON.json", comparison)
    return comparison


# ---------------------------------------------------------------- CPU gate
def cpu_checks(c, out):
    from torchvision.transforms.v2 import functional as TF
    from sparse_rtdetr.baseline.primary_evaluator import primary_evaluator_contract_binding
    require(not torch.cuda.is_initialized(), "MIX CPU endpoint checks initialized CUDA")
    started, cpu0 = time.perf_counter(), time.process_time()
    _validate_policy(c)
    primary_evaluator_contract_binding(c["source_root"])
    refs = references(c)
    pixels_equal = {}
    for canvas in ("H", "L"):
        data = development(c, canvas)
        row = data.records[0]
        with load_original(data, row) as image:
            x = pixels(data, image)
            direct = TF.pil_to_tensor(TF.resize(image, list(M.CANVAS[canvas]))).float() / 255.
        require(tuple(x.shape) == (3, *M.CANVAS[canvas]) and torch.equal(x, direct), f"MIX {canvas} pixels")
        pixels_equal[canvas] = row["stable_image_id"]
    flag = c["workload"]["activation_checkpoint"]
    from mix_formal_train import formal_schedule
    schedule = formal_schedule(c)
    e, _ = M.create(c["parent"]["path"], "cpu", schedule, activation_checkpoint=flag)
    payload = M.capture(e)
    del e
    gc.collect()
    payload["engine"]["completed_updates"] = ENDPOINT
    payload["ema"]["updates"] = ENDPOINT
    key = next(k for k, v in payload["model"].items() if v.is_floating_point() and not S._is_size_cache(k))
    payload["ema"]["module"][key] = payload["model"][key] + 0.001
    state = _weights(payload, c, flag)
    require(torch.equal(state[key], payload["ema"]["module"][key]), "MIX EMA endpoint selection")
    refused = []
    bad = {"wrong clock": {**payload, "engine": {**payload["engine"], "completed_updates": ENDPOINT - 1}},
           "acceptance schedule": {**payload, "schedule": M.test_schedule("acceptance-LHL").identity},
           "constant-H test schedule": {**payload, "schedule": M.test_schedule("test-constant-H").identity},
           "other table SHA": {**payload, "schedule": {**payload["schedule"], "table_sha256": "0" * 64}},
           "S15 state version": {**payload, "version": S.VERSION, "arm": "S15"},
           "native 768 recipe": {**payload, "config": M.canvas_config(M.base_config(), "L").binding()},
           "memory mode": {**payload, "activation_checkpoint": not flag},
           "seed1 parent": {**payload, "parent_sha256": S.PARENT_SHAS[1]}}
    for label, broken in bad.items():
        try:
            _weights(broken, c, flag)
        except (ValueError, RuntimeError):
            refused.append(label)
        else:
            raise AssertionError("MIX evaluation accepted " + label)
    forwards = {}
    for canvas in ("H", "L"):
        model, record = M.evaluation_model(state, canvas)
        require(torch.equal(model.state_dict()[key], state[key]), "evaluation model EMA tensor")
        data = development(c, canvas)
        with torch.inference_mode():
            preds, timing = forward_batch(model, data, data.records[:BATCH], "cpu")
        ids = [r["stable_image_id"] for r in data.records[:BATCH]]
        grouped = {b: _group(preds[b], ids, b, whole=True, exact=True) for b in BUDGETS}
        _prefix(grouped[600], grouped[300], ids, whole=True)
        forwards[canvas] = {"images": BATCH, "timing": timing, "record": record}
        del model
    try:  # caches of a non-H engine are refused
        broken = dict(state)
        broken["decoder.anchors"] = broken["decoder.anchors"].clone()
        idx = tuple(torch.isfinite(broken["decoder.anchors"]).nonzero()[0].tolist())
        broken["decoder.anchors"][idx] += 1e-3
        M.evaluation_model(broken, "L")
    except ValueError:
        refused.append("endpoint size cache not the native 1152 regeneration")
    else:
        raise AssertionError("MIX evaluation accepted an altered size cache")
    del payload, state
    # decision branches on synthetic values
    base = {str(b): {m: 0.4 for m in METRICS} for b in BUDGETS}
    synthetic = {"values": {"A0@L": base}, "S15_gain_pp": {"top600_APS": 3.0, "top600_AP75_small": 4.0},
                 "retention_required_pp": {"top600_APS": 2.4, "top600_AP75_small": 3.2}}
    mk = lambda **kw: {b: {m: v + kw.get(f"{m}{b}", 0.) for m, v in base[b].items()} for b in base}
    branches = {
        "all": decide(mk(APS600=.025, AP75_small600=.033), synthetic)[2] == PASS,
        "retention short": decide(mk(APS600=.023, AP75_small600=.033), synthetic)[1]["branch"]
        == "seven_cells_passed_retention_insufficient",
        "APL fails": decide(mk(APS600=.025, AP75_small600=.033, APL600=-.006), synthetic)[1]["branch"]
        == "retention_passed_original_cell_failed",
        "zero gain": decide(mk(), synthetic)[2] == FAIL,
        "AP75-small retention short": decide(mk(APS600=.025, AP75_small600=.031), synthetic)[1]["failed_retention"]
        == ["top600_AP75_small"],
    }
    require(all(branches.values()), f"MIX decision branches {branches}")
    require(not torch.cuda.is_initialized(), "MIX CPU endpoint checks initialized CUDA")
    result = {"status": "PASS", "pixels_equal_direct_resize": pixels_equal, "refused": refused,
              "cpu_endpoint_forwards": forwards, "detector_forwards": 2, "score_calls": 0,
              "references": {"values": refs["values"], "GT_counts": refs["GT_counts"],
                             "S15_gain_pp": refs["S15_gain_pp"], "retention_required_pp": refs["retention_required_pp"]},
              "decision_branches": branches, "evaluation_policy": c["evaluation_policy"],
              "cost": {"wall_seconds": time.perf_counter() - started, "cpu_seconds": time.process_time() - cpu0}}
    publish(out / "EVALUATION_CPU_CHECKS.json", result)
    return result


def main():
    from mix_formal_run import load_contract
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("infer", "score", "cpu-check"))
    parser.add_argument("--contract", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--canvas", choices=("H", "L"))
    parser.add_argument("--train-root")
    parser.add_argument("--infer-H")
    parser.add_argument("--infer-L")
    args = parser.parse_args()
    c = load_contract(args.contract)
    out = Path(args.out).absolute()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        require(Path(c["source_root"]).resolve() == ROOT.resolve(), "MIX evaluator release directory")
        if args.action == "cpu-check":
            require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "MIX CPU evaluation gate requires hidden CUDA")
            torch.set_num_threads(int(os.environ.get("MIX_CPU_THREADS", "8")))
            result, path = cpu_checks(c, out), out / "EVALUATION_CPU_CHECKS.json"
        elif args.action == "infer":
            require(args.canvas and args.train_root, "MIX infer requires canvas and train root")
            result, path = infer(c, read(args.train_root), args.canvas, out), out / "RESULT.json"
        else:
            require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "MIX scoring is CPU only")
            result = score(c, {"H": args.infer_H, "L": args.infer_L}, out)
            path = out / "COMPARISON.json"
        require(result["status"] == "PASS", "MIX evaluator result incomplete")
        publish(out / "CLOSED.json", {"status": "PASS", "action": args.action, "canvas": args.canvas,
                                      "contract_sha256": c["contract_sha256"], "result": reference(path),
                                      "wall_seconds": time.perf_counter() - started})
    except BaseException as exc:
        if not (out / "STOP.json").exists():
            publish(out / "STOP.json", {"status": "STOP", "action": args.action, "canvas": args.canvas,
                                        "contract_sha256": c["contract_sha256"], "error": str(exc)[:4000],
                                        "traceback": traceback.format_exc(), "wall_seconds": time.perf_counter() - started})
        raise


if __name__ == "__main__":
    main()
