"""Fixed S15 seed1/seed2 e120 readout (TASK_S15_SEED12_REPLICATION_001; seed0 version: commit 61646b3).

Per seed: EMA endpoint only (online stays in the complete checkpoint, not inferred or scored),
top600/top300, 2 score calls, one per-class APL COCOeval of the scored top600 bridge input.
Compared with the seed's own certified native A0 e120 EMA (pinned historical files, never
re-inferred or re-scored).  ``a0-per-class`` is the single CPU COCOeval of each pinned A0 top600
prediction that the historical metric files lack (2 calls in total), run once before the contract.
No training, tuning, mid-run scoring or endpoint selection routes.
"""
from __future__ import annotations

import argparse
import gc
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
from sparse_rtdetr.protocol_r18.geometry import forward_guards, validate_geometry
from sparse_rtdetr.protocol_r18.model import build
from sparse_rtdetr.r0.evaluation import POLICY, rank_candidates, score_budget
from sparse_rtdetr.research import scale_baseline as S

ARM = "S15"
KIND = "EMA"
BATCH = 4
WARMUPS = 4
ENDPOINT = 36480
PRIMARY = ((600, "APS", 0.50), (600, "AP75_small", 0.30))
PROTECTIONS = ((600, "AP", -0.10), (300, "AP", -0.20), (300, "APS", -0.20),
               (600, "APM", -0.50), (600, "APL", -0.50))
PASS, FAIL = "seed_passes_original_rule", "closed_fixed_config"
EVALUATION = {
    "epoch": 120, "weights": ["ema"], "primary_weights": "ema",
    "online": "kept in the complete e120 checkpoint; not inferred or scored (pre-declared readout scope)",
    "view": "A", "images": 548, "batch": BATCH, "precision": "float32", "input_hw": list(S.S15_HW),
    "warmup_batches_per_endpoint": WARMUPS, "budgets": list(BUDGETS), "score_calls_per_seed": 2,
    "per_class_APL_budget": 600, "S15_per_class_APL_extra_COCOeval_calls_per_seed": 1,
    "A0_per_class_APL_extra_COCOeval_calls_total": 2,
}
LARGE_NAMES = ("AP", "AP50", "AP75", "APS", "APM", "APL", "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large")


def _validate_policy(contract):
    require(contract["workload"]["evaluation"] == EVALUATION, "S15 endpoint workload differs")
    policy = contract["evaluation_policy"]
    require(reference(policy["path"]) == policy and read(policy["path"]) == POLICY,
            "S15 certified view-A evaluation policy differs")


def development(contract):
    data = S.development(contract["data"]["development"], S.S15_HW)
    require([r["stable_image_id"] for r in data.records] == contract["development_ids"],
            "S15 development members or order changed")
    return data


def _weights(payload, c, kind, flag):
    require(kind in ("EMA", "online"), "S15 evaluation endpoint kind")
    require((payload["version"], payload["arm"], payload["parent_sha256"], payload["config"],
             payload["activation_checkpoint"]) ==
            (S.VERSION, ARM, c["parent"]["sha256"], S.production_config(c["seed"]).binding(), flag),
            "S15 evaluation state identity (seed, parent, recipe or memory mode)")
    require(payload["engine"]["completed_updates"] == payload["ema"]["updates"] == ENDPOINT,
            "S15 evaluation requires e120 (online and EMA clocks)")
    state = payload["ema"]["module"] if kind == "EMA" else payload["model"]
    require(all(not v.is_floating_point() or v.dtype == torch.float32 for v in state.values()),
            "S15 evaluation floating state is not FP32")
    return state


def _load_model(payload, c, kind, device, flag):
    state = _weights(payload, c, kind, flag)
    model, _, _ = build(S.production_config(c["seed"]), synthetic=True)
    model.load_state_dict(state, strict=True)
    validate_geometry(model)  # 1152x2016 anchors/valid_mask/pos caches are the native regeneration
    model.eval().requires_grad_(False).to(device)
    expected = tensor_digest(state.items())
    require(tensor_digest(model.state_dict().items()) == expected, "S15 evaluation strict load differs")
    require(not any(getattr(m, "_s15_checkpointed", False) for m in model.modules()),
            "activation checkpoint must not be installed for evaluation")
    return model, expected


def _image(data, row):
    with load_original(data, row) as image:
        return pixels(data, image)


def _rank_batch(logits, boxes, rows):
    six = [rank_candidates(logit, box, row, 600) for logit, box, row in zip(logits, boxes, rows)]
    return {600: [p for group in six for p in group], 300: [p for group in six for p in group[:300]]}


def forward_batch(model, data, rows, device):
    """One FP32 full-model forward on the actual 1152x2016 view-A pixels; returns ranked predictions."""
    device = torch.device(device)
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None)
    start = time.perf_counter()
    x = torch.stack([_image(data, row) for row in rows]).to(device)
    sync()
    input_done = time.perf_counter()
    with torch.autocast(device.type, enabled=False), forward_guards(model):
        result = model(x)
    sync()
    forward_done = time.perf_counter()
    require(set(result) == {"pred_boxes", "pred_logits"}
            and result["pred_boxes"].shape == (len(rows), 300, 4) and result["pred_logits"].shape == (len(rows), 300, 10)
            and result["pred_boxes"].dtype == result["pred_logits"].dtype == torch.float32, "S15 endpoint output schema")
    boxes, logits = result["pred_boxes"].cpu(), result["pred_logits"].cpu()
    output_done = time.perf_counter()
    predictions = _rank_batch(logits, boxes, rows)
    rank_done = time.perf_counter()
    return predictions, {"data_and_h2d": input_done - start, "forward": forward_done - input_done,
                         "cpu_materialization": output_done - forward_done, "ranking": rank_done - output_done,
                         "total": rank_done - start}


def infer(contract, endpoint_ref, out, read_endpoint):
    counts = {"attempted_forwards": 0, "completed_forwards": 0}
    started = time.perf_counter()
    try:
        return _infer(contract, endpoint_ref, out, read_endpoint, counts)
    except BaseException as exc:
        peak = {"peak_allocated_bytes": None, "peak_reserved_bytes": None}
        try:
            peak = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
        except Exception:
            pass
        publish(out / "STOP.json", {"status": "STOP_CUDA_OOM" if isinstance(exc, torch.cuda.OutOfMemoryError) else "STOP",
                                    "phase": "EMA endpoint inference", "seed": contract["seed"], "kind": KIND,
                                    "checkpoint": endpoint_ref, **counts, **peak,
                                    "elapsed_seconds": time.perf_counter() - started,
                                    "error": str(exc)[:4000], "traceback": traceback.format_exc()})
        raise


def _infer(contract, endpoint_ref, out, read_endpoint, counts):
    device = torch.device("cuda:0")
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    payload = read_endpoint(endpoint_ref["path"])
    model, weight_sha = _load_model(payload, contract, KIND, device, contract["workload"]["activation_checkpoint"])
    del payload
    data = development(contract)
    rng = native.capture_rng(device)
    torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - started

    def batch(rows):
        counts["attempted_forwards"] += 1
        predictions, timing = forward_batch(model, data, rows, device)
        counts["completed_forwards"] += 1
        return predictions, timing

    warmup, measured = [], []
    all_predictions = {budget: [] for budget in BUDGETS}
    with torch.inference_mode():
        for batch_index in range(WARMUPS):
            _, timing = batch(data.records[batch_index * BATCH:(batch_index + 1) * BATCH])
            warmup.append(timing)
            event(out, "endpoint_warmup", arm=ARM, seed=contract["seed"], endpoint_kind=KIND, batch=batch_index + 1)
        for offset in range(0, len(data.records), BATCH):
            predictions, timing = batch(data.records[offset:offset + BATCH])
            measured.append(timing)
            for budget in BUDGETS:
                all_predictions[budget].extend(predictions[budget])
            event(out, "endpoint_batch", arm=ARM, seed=contract["seed"], endpoint_kind=KIND, batch=offset // BATCH + 1)
    require(len(warmup) == WARMUPS and len(measured) == 137
            and counts["attempted_forwards"] == counts["completed_forwards"] == WARMUPS + 137, "S15 endpoint batch count")
    require(tensor_digest(model.state_dict().items()) == weight_sha
            and native.exact_equal(native.capture_rng(device), rng)
            and all(not m.training for m in model.modules()) and all(p.grad is None for p in model.parameters()),
            "S15 evaluation modified model state or RNG")
    summaries = {name: sum(row[name] for row in measured) for name in measured[0]}
    refs = {}
    for budget in BUDGETS:
        path = out / f"top{budget}-predictions.json"
        publish(path, all_predictions[budget])
        refs[str(budget)] = reference(path)
    result = {"status": "PASS", "arm": ARM, "seed": contract["seed"], "kind": KIND, "epoch": 120, "update": ENDPOINT,
              "view": "A", "input_hw": list(S.S15_HW), "device": str(device), "FP32": True, "batch_size": BATCH,
              "contract_sha256": contract["contract_sha256"], "formal_identity": contract["formal_identity"],
              "checkpoint": endpoint_ref, "weights_sha256": weight_sha, "predictions": refs,
              "images": len(data.records), "steady_forwards": len(measured), "warmup_forwards": len(warmup),
              "load_seconds": load_seconds, "warmup_seconds": sum(r["total"] for r in warmup), "warmup_timing": warmup,
              "cold_first_forward_seconds": warmup[0]["forward"], "steady_timing": summaries,
              "elapsed_seconds": time.perf_counter() - started,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
              "forward_timing_includes_native_geometry_guards": True, "model_and_RNG_unchanged": True, **counts}
    publish(out / "RESULT.json", result)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


# ---------------------------------------------------------------- per-class APL
def per_class_large(rows, parsed, predictions, bridge):
    """Per-class APL from an identical COCOeval of the same bridge inputs; aggregates must match exactly."""
    import contextlib
    import io
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from sparse_rtdetr.r0.evaluation import bridge_inputs
    dataset, det, _ = bridge_inputs(rows, parsed, predictions)
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(); coco.dataset = dataset; coco.createIndex()
        dt = coco.loadRes(det)
        e = COCOeval(coco, dt, "bbox")
        e.params.imgIds = sorted(r["id"] for r in dataset["images"]); e.params.catIds = list(range(1, 11))
        e.params.maxDets = [1, 10, 100]
        e.evaluate(); e.accumulate(); e.summarize()
    require(all(float(v) == bridge["metrics"][k] for k, v in zip(LARGE_NAMES, e.stats)),
            "per-class recomputation differs from the scored COCO bridge")
    ix = e.params.areaRngLbl.index("large")
    lo, hi = e.params.areaRng[ix]
    values = []
    for k, cat in enumerate(e.params.catIds):
        p = e.eval["precision"][:, :, k, ix, -1]
        v = p[p > -1]
        n = sum(1 for a in dataset["annotations"] if a["category_id"] == cat and not a.get("iscrowd")
                and lo <= a["area"] <= hi)
        values.append({"category_id": int(cat), "GT": n, "APL": float(v.mean()) if len(v) else None})
    mean = float(np.mean([v["APL"] for v in values if v["APL"] is not None]))
    require(abs(mean - e.stats[5]) < 1e-14, "per-class APL macro differs")
    return values


def a0_per_class(out_path):
    """Single CPU COCOeval of each seed's pinned historical A0 top600 prediction (2 calls in total)."""
    import os
    from s15_formal_run import DATA_REF, DATA_REF_SHA, SEEDS, a0_reference_refs, pinned
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized(), "A0 per-class is CPU only")
    started = time.perf_counter()
    receipt = pinned(DATA_REF, DATA_REF_SHA)
    old = read(receipt["path"])
    data = S.development(old["data"]["development"], S.S15_HW)
    require([r["stable_image_id"] for r in data.records] == old["development_ids"], "development members/order")
    rows = data.records
    parsed = [data.raw(row) for row in rows]
    result = {"status": "PASS", "data_receipt": receipt, "source_root": str(ROOT), "seeds": {},
              "COCOeval_calls": 0, "score_budget_calls": 0, "detector_forwards": 0}
    for seed in SEEDS:
        refs = a0_reference_refs(seed)
        metrics = read(refs["A0_metrics"]["600"]["path"])
        require(reference(refs["A0_metrics"]["600"]["path"]) == refs["A0_metrics"]["600"], "A0 metric pin")
        prediction = refs["A0_predictions"]["600"]
        require(reference(prediction["path"]) == prediction, "A0 prediction pin")
        t = time.perf_counter()
        values = per_class_large(rows, parsed, read(prediction["path"]), metrics["COCO_bridge"])
        result["COCOeval_calls"] += 1
        result["seeds"][str(seed)] = {"metrics": refs["A0_metrics"]["600"], "prediction": prediction,
                                      "A0_e120_sha256": refs["A0_e120_sha256"],
                                      "APL_macro": metrics["COCO_bridge"]["metrics"]["APL"],
                                      "twelve_aggregates_equal_historical_bridge": True,
                                      "large_by_class": values, "seconds": time.perf_counter() - t}
    result["wall_seconds"] = time.perf_counter() - started
    require(result["COCOeval_calls"] == EVALUATION["A0_per_class_APL_extra_COCOeval_calls_total"], "A0 per-class count")
    publish(out_path, result)
    return result


# ---------------------------------------------------------------- A0 reference (pinned, not recomputed)
def _pinned(ref):
    require(reference(ref["path"]) == ref, f"pinned A0 reference differs: {ref['path']}")
    return read(ref["path"])


def a0_reference(c):
    refs = c["reference"]
    values, counts = {}, {}
    for budget in BUDGETS:
        metrics = _pinned(refs["A0_metrics"][str(budget)])
        binding = metrics["sahi_binding"]
        require((binding["seed"], binding["arm"], binding["budget"], binding["checkpoint_sha256"],
                 binding["own_seed_A"]) == (c["seed"], "A", budget, refs["A0_e120_sha256"], True)
                and binding["prediction"] == refs["A0_predictions"][str(budget)],
                "historical A0 metric binding differs")
        values[str(budget)] = bridge_values(metrics)
        counts[str(budget)] = {k: metrics["spatial_filter_counts"][k] for k in ("GT_before", "GT_after")}
    cached = _pinned(refs["A0_per_class"])
    require(cached["status"] == "PASS", "A0 per-class status")
    entry = cached["seeds"][str(c["seed"])]
    require(entry["prediction"] == refs["A0_predictions"]["600"] and entry["metrics"] == refs["A0_metrics"]["600"]
            and entry["A0_e120_sha256"] == refs["A0_e120_sha256"]
            and entry["APL_macro"] == values["600"]["APL"], "A0 per-class APL binding differs")
    return {"values": {KIND: values}, "GT_counts": counts, "per_class_APL_top600": {KIND: entry["large_by_class"]}}


def compare(s15, a0):
    differences = {KIND: {str(b): {m: 100. * (s15[KIND][str(b)][m] - a0[KIND][str(b)][m]) for m in METRICS}
                          for b in BUDGETS}}
    cells = {}
    for budget, metric, minimum in (*PRIMARY, *PROTECTIONS):
        observed = differences[KIND][str(budget)][metric]
        cells[f"top{budget}_{metric}"] = {"difference_pp": observed, "minimum_pp": minimum,
                                          "role": "primary" if (budget, metric, minimum) in PRIMARY else "protection",
                                          "passed": observed >= minimum}
    passed = all(v["passed"] for v in cells.values())
    primary = all(v["passed"] for v in cells.values() if v["role"] == "primary")
    return differences, {"passed": passed, "primary_passed": primary,
                         "failed_cells": [k for k, v in cells.items() if not v["passed"]], "cells": cells}, \
        PASS if passed else FAIL


def score(contract, result, out):
    started = time.perf_counter()
    a0 = a0_reference(contract)
    data = development(contract)
    rows = data.records
    ids = contract["development_ids"]
    parsed = [data.raw(row) for row in rows]
    values, metric_refs, scoring, counts = {KIND: {}}, {}, [], {}
    require((result["status"], result["arm"], result["seed"], result["kind"], result["contract_sha256"], result["images"],
             result["steady_forwards"], result["warmup_forwards"]) ==
            ("PASS", ARM, contract["seed"], KIND, contract["contract_sha256"], 548, 137, WARMUPS),
            "S15 scoring inference identity")
    predictions, grouped = {}, {}
    for budget in BUDGETS:
        ref = result["predictions"][str(budget)]
        require(reference(ref["path"]) == ref, "S15 prediction reference differs")
        predictions[budget] = read(ref["path"])
        grouped[budget] = _group(predictions[budget], ids, budget, whole=True, exact=True)
    _prefix(grouped[600], grouped[300], ids, whole=True)
    per_class = None
    for budget in BUDGETS:
        t = time.perf_counter()
        event(out, "score_started", ordinal=len(scoring) + 1, arm=ARM, seed=contract["seed"], endpoint_kind=KIND,
              budget=budget)
        metrics = score_budget(rows, parsed, predictions[budget], contract["source_root"])
        metrics["S15_binding"] = {"contract_sha256": contract["contract_sha256"], "formal_identity": contract["formal_identity"],
                                  "arm": ARM, "seed": contract["seed"], "kind": KIND, "view": "A",
                                  "input_hw": list(S.S15_HW), "epoch": 120, "budget": budget,
                                  "checkpoint": result["checkpoint"], "weights_sha256": result["weights_sha256"],
                                  "predictions": result["predictions"][str(budget)]}
        values[KIND][str(budget)] = bridge_values(metrics)
        counts[str(budget)] = {k: metrics["spatial_filter_counts"][k] for k in ("GT_before", "GT_after")}
        path = out / f"S15-seed{contract['seed']}-{KIND}-A-top{budget}-metrics.json"
        publish(path, metrics)
        metric_refs[str(budget)] = reference(path)
        scoring.append({"kind": KIND, "budget": budget, "score_budget_seconds": time.perf_counter() - t})
        event(out, "score_complete", ordinal=len(scoring), arm=ARM, seed=contract["seed"], endpoint_kind=KIND,
              budget=budget, elapsed_seconds=scoring[-1]["score_budget_seconds"])
        if budget == EVALUATION["per_class_APL_budget"]:
            t = time.perf_counter()
            mine = per_class_large(rows, parsed, predictions[budget], metrics["COCO_bridge"])
            reference_rows = a0["per_class_APL_top600"][KIND]
            require([r["category_id"] for r in mine] == [r["category_id"] for r in reference_rows]
                    and [r["GT"] for r in mine] == [r["GT"] for r in reference_rows], "per-class GT denominators differ")
            per_class = {"S15": mine, "A0": reference_rows, "extra_COCOeval_seconds": time.perf_counter() - t,
                         "difference_pp": [
                             {"category_id": s["category_id"], "GT": s["GT"],
                              "APL_difference_pp": None if s["APL"] is None or r["APL"] is None else 100. * (s["APL"] - r["APL"]),
                              "macro_contribution_pp": None if s["APL"] is None or r["APL"] is None
                              else 10. * (s["APL"] - r["APL"])}
                             for s, r in zip(mine, reference_rows)]}
    del predictions, grouped
    gc.collect()
    require(len(scoring) == EVALUATION["score_calls_per_seed"], "S15 fixed score count differs")
    require(counts == a0["GT_counts"], f"scored GT set differs from A0: {counts} vs {a0['GT_counts']}")
    differences, gates, decision = compare(values, a0["values"])
    comparison = {"status": "PASS", "seed": contract["seed"], "scientific_decision": decision,
                  "scores": {"S15": values, "A0": a0["values"]}, "differences_pp_S15_minus_A0": differences,
                  "EMA_gates": gates, "GT_counts": counts, "per_class_APL_top600": {KIND: per_class},
                  "score_units": "0..1 COCO bridge", "delta_units": "percentage points",
                  "metrics": metric_refs, "A0_reference": contract["reference"],
                  "steady_EMA_inference_timing": result["steady_timing"],
                  "scoring": scoring, "score_calls": len(scoring), "extra_COCOeval_calls": 1,
                  "scoring_wall_seconds": time.perf_counter() - started,
                  "inference": {KIND: result}, "contract_sha256": contract["contract_sha256"],
                  "formal_identity": contract["formal_identity"]}
    publish(out / "COMPARISON.json", comparison)
    return comparison


# ---------------------------------------------------------------- CPU gate (per seed)
def cpu_checks(contract, other, out):
    from sparse_rtdetr.baseline.primary_evaluator import primary_evaluator_contract_binding
    from torchvision.transforms.v2 import functional as TF
    require(not torch.cuda.is_initialized(), "S15 CPU endpoint checks initialized CUDA")
    started, cpu0 = time.perf_counter(), time.process_time()
    _validate_policy(contract)
    data = development(contract)
    row = data.records[0]
    x = _image(data, row)
    with load_original(data, row) as image:
        direct = TF.pil_to_tensor(TF.resize(image, list(S.S15_HW))).float() / 255.
    require(x.dtype == torch.float32 and tuple(x.shape) == (3, *S.S15_HW) and torch.equal(x, direct),
            "S15 GT-free pixels differ from one direct 1152x2016 resize")
    primary_evaluator_contract_binding(contract["source_root"])
    a0 = a0_reference(contract)
    flag = contract["workload"]["activation_checkpoint"]
    e, _ = S.create(contract["parent"]["path"], "cpu", seed=contract["seed"], activation_checkpoint=flag)
    payload = S.capture(e)
    del e
    gc.collect()
    payload["engine"]["completed_updates"] = ENDPOINT
    payload["ema"]["updates"] = ENDPOINT
    key = next(k for k, v in payload["model"].items() if v.is_floating_point() and not S._is_size_cache(k))
    payload["ema"]["module"][key] = payload["model"][key] + 0.001
    model, sha = _load_model(payload, contract, KIND, "cpu", flag)
    require(torch.equal(model.state_dict()[key], payload["ema"]["module"][key])
            and not torch.equal(model.state_dict()[key], payload["model"][key]), "S15 EMA endpoint selected wrong state")
    refused = []
    for label, broken in (("wrong clock", {**payload, "engine": {**payload["engine"], "completed_updates": 36479}}),
                          ("768 recipe", {**payload, "config": S.Config(seed=contract["seed"]).binding()}),
                          ("memory mode", {**payload, "activation_checkpoint": not flag}),
                          (f"seed{other['seed']} recipe", {**payload, "config": S.production_config(other["seed"]).binding()}),
                          (f"seed{other['seed']} parent", {**payload, "parent_sha256": other["parent"]["sha256"]})):
        try:
            _load_model(broken, contract, KIND, "cpu", flag)
        except (ValueError, RuntimeError):  # runio.require raises RuntimeError
            refused.append(label)
        else:
            raise AssertionError("S15 evaluation accepted " + label)
    # One actual GT-free FP32 forward through the endpoint path (4 development images, no file, no score).
    with torch.inference_mode():
        preds, timing = forward_batch(model, data, data.records[:BATCH], "cpu")
    ids = [r["stable_image_id"] for r in data.records[:BATCH]]
    grouped = {b: _group(preds[b], ids, b, whole=True, exact=True) for b in BUDGETS}
    _prefix(grouped[600], grouped[300], ids, whole=True)
    del model, payload
    values = {KIND: {str(b): {m: 0.4 for m in METRICS} for b in BUDGETS}}
    base = {KIND: {str(b): dict(v) for b, v in values[KIND].items()}}
    require(compare(values, base)[2] == FAIL, "zero gain gate")
    values[KIND]["600"].update(APS=0.4051, AP75_small=0.4031)
    require(compare(values, base)[2] == PASS, "controlled gain gate")
    values[KIND]["600"]["AP75_small"] = 0.4029
    require(compare(values, base)[2] == FAIL, "AP75-small gate")
    values[KIND]["600"]["AP75_small"] = 0.4031
    values[KIND]["600"]["APL"] = 0.3949
    require(compare(values, base)[2] == FAIL and compare(values, base)[1]["failed_cells"] == ["top600_APL"]
            and compare(values, base)[1]["primary_passed"], "APL protection gate")
    values[KIND]["600"]["APL"] = 0.4
    values[KIND]["300"]["APS"] = 0.3979
    require(compare(values, base)[2] == FAIL, "APS300 protection gate")
    require(not torch.cuda.is_initialized(), "S15 CPU endpoint checks initialized CUDA")
    result = {"status": "PASS", "seed": contract["seed"], "development_images": len(data.records),
              "one_actual_development_image_pixels_equal_direct_resize": ids[0],
              "EMA_weight_selection_sha256": sha, "refused": refused,
              "one_cpu_endpoint_forward": {"images": BATCH, "timing": timing,
                                           "ranked_per_image": len(preds[600]) // BATCH},
              "ranking_prefix_exact": True, "gate_branches": 5,
              "A0_reference_values": a0["values"], "A0_GT_counts": a0["GT_counts"],
              "A0_per_class_APL_top600": a0["per_class_APL_top600"],
              "detector_forwards": 1, "score_calls": 0, "evaluation_policy": contract["evaluation_policy"],
              "cost": {"wall_seconds": time.perf_counter() - started, "cpu_seconds": time.process_time() - cpu0}}
    publish(out / "EVALUATION_CPU_CHECKS.json", result)
    return result


def evaluate(contract, train_root, out):
    from s15_formal_run import load_contract, seed_view
    from s15_formal_train import bind_seed, read_endpoint
    from r25_run import setup
    _validate_policy(contract)
    bind_seed(contract["seed"])
    source = seed_view(load_contract(train_root["contract"]), contract["seed"])
    require(source["formal_identity"] == contract["formal_identity"], "S15 endpoint formal identity differs")
    path = Path(train_root["path"]) / "checkpoint-epoch-120.pth"
    sidecar = read(path.with_suffix(".pth.json"))
    require((sidecar["status"], sidecar["arm"], sidecar["seed"], sidecar["contract_sha256"], sidecar["formal_identity"],
             sidecar["global_update"], sidecar["scope"]) ==
            ("COMPLETE", ARM, contract["seed"], source["contract_sha256"], source["formal_identity"], ENDPOINT, "formal")
            and Path(sidecar["file"]["path"]).resolve() == path.resolve(), "S15 evaluator endpoint sidecar identity")
    setup("cuda:0")
    stage = out / f"S15-{KIND}"
    stage.mkdir()
    result = infer(contract, sidecar["file"], stage, lambda p: read_endpoint(source, p))
    return score(contract, result, out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("evaluate", "cpu-check", "a0-per-class"))
    parser.add_argument("--contract")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--train-root")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.action == "a0-per-class":
        out = Path(args.out).absolute()
        require(not out.exists(), "A0 per-class output exists")
        a0_per_class(out)
        return
    from s15_formal_run import load_contract, seed_view
    contract = load_contract(args.contract)
    c = seed_view(contract, args.seed)
    out = Path(args.out).absolute()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        require(Path(c["source_root"]).resolve() == ROOT.resolve(), "S15 evaluator release directory")
        if args.action == "cpu-check":
            import os
            require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "S15 CPU evaluation gate requires hidden CUDA")
            torch.set_num_threads(int(os.environ.get("S15_CPU_THREADS", "8")))
            other = seed_view(contract, next(s for s in contract["seed_order"] if s != c["seed"]))
            result = cpu_checks(c, other, out)
            path = out / "EVALUATION_CPU_CHECKS.json"
        else:
            require(bool(args.train_root), "S15 evaluator requires the train mapping")
            result = evaluate(c, read(args.train_root), out)
            path = out / "COMPARISON.json"
        require(result["status"] == "PASS", "S15 evaluator result incomplete")
        publish(out / "CLOSED.json", {"status": "PASS", "mode": "eval", "seed": c["seed"],
                                      "contract_sha256": c["contract_sha256"],
                                      "result": reference(path), "wall_seconds": time.perf_counter() - started})
    except BaseException as exc:
        publish(out / "STOP.json", {"status": "STOP", "seed": c["seed"], "contract_sha256": c["contract_sha256"],
                                    "error": str(exc)[:4000], "traceback": traceback.format_exc(),
                                    "wall_seconds": time.perf_counter() - started})
        raise


if __name__ == "__main__":
    main()
