"""Fixed S15 e120 readout: EMA (primary) and online endpoints, top600/top300, 4 score calls.

Compared with the certified R25 A0 e120 values (pinned files, not re-inferred or re-scored).
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
KINDS = ("EMA", "online")
BATCH = 4
WARMUPS = 4
ENDPOINT = 36480
PRIMARY = ((600, "APS", 0.50), (600, "AP75_small", 0.30))
PROTECTIONS = ((600, "AP", -0.10), (300, "AP", -0.20), (300, "APS", -0.20),
               (600, "APM", -0.50), (600, "APL", -0.50))
EVALUATION = {
    "epoch": 120, "weights": ["ema", "online"], "primary_weights": "ema", "view": "A", "images": 548,
    "batch": BATCH, "precision": "float32", "input_hw": list(S.S15_HW), "warmup_batches_per_endpoint": WARMUPS,
    "budgets": list(BUDGETS), "score_calls": 4, "per_class_APL_budget": 600,
    "per_class_APL_extra_COCOeval_calls": 2,
}


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


def _weights(payload, kind, flag):
    require(kind in KINDS, "S15 evaluation endpoint kind")
    require((payload["version"], payload["arm"], payload["parent_sha256"], payload["config"],
             payload["activation_checkpoint"]) ==
            (S.VERSION, ARM, S.PARENT_SHA, S.production_config().binding(), flag), "S15 evaluation state identity")
    require(payload["engine"]["completed_updates"] == payload["ema"]["updates"] == ENDPOINT,
            "S15 evaluation requires both online and EMA e120")
    state = payload["ema"]["module"] if kind == "EMA" else payload["model"]
    require(all(not v.is_floating_point() or v.dtype == torch.float32 for v in state.values()),
            "S15 evaluation floating state is not FP32")
    return state


def _load_model(payload, kind, device, flag):
    state = _weights(payload, kind, flag)
    model, _, _ = build(S.production_config(), synthetic=True)
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


def infer(contract, endpoint_ref, kind, out, read_endpoint):
    counts = {"attempted_forwards": 0, "completed_forwards": 0}
    started = time.perf_counter()
    try:
        return _infer(contract, endpoint_ref, kind, out, read_endpoint, counts)
    except BaseException as exc:
        peak = {"peak_allocated_bytes": None, "peak_reserved_bytes": None}
        try:
            peak = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
        except Exception:
            pass
        publish(out / "STOP.json", {"status": "STOP_CUDA_OOM" if isinstance(exc, torch.cuda.OutOfMemoryError) else "STOP",
                                    "phase": "EMA/online endpoint inference", "kind": kind,
                                    "checkpoint": endpoint_ref, **counts, **peak,
                                    "elapsed_seconds": time.perf_counter() - started,
                                    "error": str(exc)[:4000], "traceback": traceback.format_exc()})
        raise


def _infer(contract, endpoint_ref, kind, out, read_endpoint, counts):
    device = torch.device("cuda:0")
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    payload = read_endpoint(endpoint_ref["path"])
    model, weight_sha = _load_model(payload, kind, device, contract["workload"]["activation_checkpoint"])
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
            event(out, "endpoint_warmup", arm=ARM, endpoint_kind=kind, batch=batch_index + 1)
        for offset in range(0, len(data.records), BATCH):
            predictions, timing = batch(data.records[offset:offset + BATCH])
            measured.append(timing)
            for budget in BUDGETS:
                all_predictions[budget].extend(predictions[budget])
            event(out, "endpoint_batch", arm=ARM, endpoint_kind=kind, batch=offset // BATCH + 1)
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
    result = {"status": "PASS", "arm": ARM, "kind": kind, "epoch": 120, "update": ENDPOINT, "view": "A",
              "input_hw": list(S.S15_HW), "device": str(device), "FP32": True, "batch_size": BATCH,
              "contract_sha256": contract["contract_sha256"], "checkpoint": endpoint_ref,
              "weights_sha256": weight_sha, "predictions": refs, "images": len(data.records),
              "steady_forwards": len(measured), "warmup_forwards": len(warmup), "load_seconds": load_seconds,
              "warmup_seconds": sum(r["total"] for r in warmup), "warmup_timing": warmup,
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


# ---------------------------------------------------------------- A0 reference (pinned, not recomputed)
def _pinned(ref):
    require(reference(ref["path"]) == ref, f"pinned A0 reference differs: {ref['path']}")
    return read(ref["path"])


def a0_reference(contract):
    refs = contract["reference"]
    comparison = _pinned(refs["A0_comparison"])
    require(comparison["status"] == "PASS" and comparison["contract_sha256"] == refs["R25_contract_sha256"],
            "R25 comparison identity")
    values, per_class = {}, {}
    endpoint = refs["A0_endpoint"]
    for kind in KINDS:
        inference = comparison["inference"][kind]["A0"]
        require((inference["status"], inference["arm"], inference["kind"], inference["epoch"], inference["images"],
                 inference["steady_forwards"]) == ("PASS", "A0", kind, 120, 548, 137)
                and inference["checkpoint"] == endpoint, "R25 A0 inference/endpoint binding")
        values[kind] = {}
        for budget in BUDGETS:
            metrics = _pinned(refs["A0_metrics"][kind][str(budget)])
            binding = metrics["R25_binding"]
            require((binding["arm"], binding["kind"], binding["view"], binding["epoch"], binding["budget"],
                     binding["contract_sha256"]) == ("A0", kind, "A", 120, budget, refs["R25_contract_sha256"])
                    and binding["checkpoint"] == endpoint and binding["weights_sha256"] == inference["weights_sha256"]
                    and binding["predictions"] == inference["predictions"][str(budget)]
                    and comparison["metrics"][kind]["A0"][str(budget)] == refs["A0_metrics"][kind][str(budget)],
                    "R25 A0 metric binding differs")
            values[kind][str(budget)] = bridge_values(metrics)
            require(values[kind][str(budget)] == comparison["scores"][kind]["A0"][str(budget)],
                    "R25 A0 metric file differs from its comparison")
        cached = _pinned(refs["A0_per_class"])["endpoints"]["A0-" + kind]
        require(cached["binding"]["prediction"] == inference["predictions"]["600"]
                and cached["binding"]["checkpoint"] == endpoint
                and cached["metrics"]["APL"] == values[kind]["600"]["APL"], "R25 A0 per-class APL binding differs")
        per_class[kind] = cached["large_by_class"]
    return {"values": values, "per_class_APL_top600": per_class, "comparison": comparison}


def compare(s15, a0):
    differences = {kind: {str(b): {m: 100. * (s15[kind][str(b)][m] - a0[kind][str(b)][m]) for m in METRICS}
                          for b in BUDGETS} for kind in KINDS}
    cells = {}
    for budget, metric, minimum in (*PRIMARY, *PROTECTIONS):
        observed = differences["EMA"][str(budget)][metric]
        cells[f"top{budget}_{metric}"] = {"difference_pp": observed, "minimum_pp": minimum,
                                          "role": "primary" if (budget, metric, minimum) in PRIMARY else "protection",
                                          "passed": observed >= minimum}
    passed = all(v["passed"] for v in cells.values())
    return differences, {"passed": passed, "cells": cells}, \
        "single_seed_preliminary_net_benefit" if passed else "closed_fixed_config"


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
    names = ("AP", "AP50", "AP75", "APS", "APM", "APL", "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large")
    require(all(float(v) == bridge["metrics"][k] for k, v in zip(names, e.stats)),
            "S15 per-class recomputation differs from the scored COCO bridge")
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
    require(abs(mean - e.stats[5]) < 1e-14, "S15 per-class APL macro differs")
    return values


def score(contract, inference_results, out):
    started = time.perf_counter()
    a0 = a0_reference(contract)
    data = development(contract)
    rows = data.records
    ids = contract["development_ids"]
    parsed = [data.raw(row) for row in rows]
    values = {kind: {} for kind in KINDS}
    metric_refs = {kind: {} for kind in KINDS}
    per_class, scoring = {}, []
    for kind in KINDS:
        result = inference_results[kind]
        require((result["status"], result["arm"], result["kind"], result["contract_sha256"], result["images"],
                 result["steady_forwards"], result["warmup_forwards"]) ==
                ("PASS", ARM, kind, contract["contract_sha256"], 548, 137, WARMUPS), "S15 scoring inference identity")
        predictions, grouped = {}, {}
        for budget in BUDGETS:
            ref = result["predictions"][str(budget)]
            require(reference(ref["path"]) == ref, "S15 prediction reference differs")
            predictions[budget] = read(ref["path"])
            grouped[budget] = _group(predictions[budget], ids, budget, whole=True, exact=True)
        _prefix(grouped[600], grouped[300], ids, whole=True)
        for budget in BUDGETS:
            t = time.perf_counter()
            event(out, "score_started", ordinal=len(scoring) + 1, arm=ARM, endpoint_kind=kind, budget=budget)
            metrics = score_budget(rows, parsed, predictions[budget], contract["source_root"])
            metrics["S15_binding"] = {"contract_sha256": contract["contract_sha256"], "arm": ARM, "kind": kind,
                                      "view": "A", "input_hw": list(S.S15_HW), "epoch": 120, "budget": budget,
                                      "checkpoint": result["checkpoint"], "weights_sha256": result["weights_sha256"],
                                      "predictions": result["predictions"][str(budget)]}
            values[kind][str(budget)] = bridge_values(metrics)
            path = out / f"S15-{kind}-A-top{budget}-metrics.json"
            publish(path, metrics)
            metric_refs[kind][str(budget)] = reference(path)
            scoring.append({"kind": kind, "budget": budget, "elapsed_seconds": time.perf_counter() - t})
            event(out, "score_complete", ordinal=len(scoring), arm=ARM, endpoint_kind=kind, budget=budget,
                  elapsed_seconds=scoring[-1]["elapsed_seconds"])
            if budget == EVALUATION["per_class_APL_budget"]:
                t = time.perf_counter()
                per_class[kind] = {"S15": per_class_large(rows, parsed, predictions[budget], metrics["COCO_bridge"]),
                                   "A0": a0["per_class_APL_top600"][kind]}
                per_class[kind]["difference_pp"] = [
                    {"category_id": s["category_id"], "GT": s["GT"],
                     "APL_difference_pp": None if s["APL"] is None or r["APL"] is None else 100. * (s["APL"] - r["APL"]),
                     "macro_contribution_pp": None if s["APL"] is None or r["APL"] is None else 10. * (s["APL"] - r["APL"])}
                    for s, r in zip(per_class[kind]["S15"], per_class[kind]["A0"])]
                per_class[kind]["extra_COCOeval_seconds"] = time.perf_counter() - t
        del predictions, grouped
        gc.collect()
    require(len(scoring) == EVALUATION["score_calls"], "S15 fixed score count differs")
    differences, gates, decision = compare(values, a0["values"])
    base = a0["comparison"]["inference"]["EMA"]["A0"]["steady_timing"]
    efficiency = {key + "_ratio_S15_over_A0": inference_results["EMA"]["steady_timing"][key] / base[key]
                  for key in ("forward", "total")}
    comparison = {"status": "PASS", "scientific_decision": decision, "scores": {"S15": values, "A0": a0["values"]},
                  "differences_pp_S15_minus_A0": differences, "EMA_gates": gates,
                  "online_role": "reported in full, does not decide",
                  "per_class_APL_top600": per_class, "score_units": "0..1 COCO bridge", "delta_units": "percentage points",
                  "metrics": metric_refs, "A0_reference": contract["reference"],
                  "steady_EMA_inference_cost": {**efficiency,
                                                "caveat": "A0 timing is from a different session (R25 formal-001); "
                                                          "ordering/cache state differ, not an exact speed ratio"},
                  "scoring": scoring, "score_calls": len(scoring),
                  "scoring_wall_seconds": time.perf_counter() - started,
                  "inference": inference_results, "contract_sha256": contract["contract_sha256"]}
    publish(out / "COMPARISON.json", comparison)
    return comparison


# ---------------------------------------------------------------- CPU gate
def cpu_checks(contract, out):
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
    e, _ = S.create(contract["parent"]["path"], "cpu", activation_checkpoint=flag)
    payload = S.capture(e)
    del e
    gc.collect()
    payload["engine"]["completed_updates"] = ENDPOINT
    payload["ema"]["updates"] = ENDPOINT
    key = next(k for k, v in payload["model"].items() if v.is_floating_point() and not S._is_size_cache(k))
    payload["ema"]["module"][key] = payload["model"][key] + 0.001
    selected = {}
    for kind, source in (("EMA", payload["ema"]["module"]), ("online", payload["model"])):
        model, sha = _load_model(payload, kind, "cpu", flag)
        require(torch.equal(model.state_dict()[key], source[key]), "S15 endpoint selected wrong state")
        selected[kind] = sha
    require(selected["EMA"] != selected["online"], "S15 endpoint selection fixture degenerate")
    refused = []
    for label, broken in (("wrong clock", {**payload, "engine": {**payload["engine"], "completed_updates": 36479}}),
                          ("768 recipe", {**payload, "config": S.Config(seed=0).binding()}),
                          ("memory mode", {**payload, "activation_checkpoint": not flag})):
        try:
            _load_model(broken, "EMA", "cpu", flag)
        except ValueError:
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
    values = {k: {str(b): {m: 0.4 for m in METRICS} for b in BUDGETS} for k in KINDS}
    base = {k: {str(b): dict(v) for b, v in values[k].items()} for k in KINDS}
    require(compare(values, base)[2] == "closed_fixed_config", "S15 zero gain gate")
    values["EMA"]["600"].update(APS=0.4051, AP75_small=0.4031)
    require(compare(values, base)[2] == "single_seed_preliminary_net_benefit", "S15 controlled gain gate")
    values["EMA"]["600"]["AP75_small"] = 0.4029
    require(compare(values, base)[2] == "closed_fixed_config", "S15 AP75-small gate")
    values["EMA"]["600"]["AP75_small"] = 0.4031
    values["EMA"]["600"]["APL"] = 0.3949
    require(compare(values, base)[2] == "closed_fixed_config", "S15 APL protection gate")
    values["EMA"]["600"]["APL"] = 0.4
    values["online"]["600"]["APS"] = 0.30
    require(compare(values, base)[2] == "single_seed_preliminary_net_benefit", "online must not decide")
    require(not torch.cuda.is_initialized(), "S15 CPU endpoint checks initialized CUDA")
    result = {"status": "PASS", "development_images": len(data.records),
              "one_actual_development_image_pixels_equal_direct_resize": ids[0],
              "strict_disjoint_weight_selections": selected, "refused": refused,
              "one_cpu_endpoint_forward": {"images": BATCH, "timing": timing,
                                           "ranked_per_image": len(preds[600]) // BATCH},
              "ranking_prefix_exact": True, "gate_branches": 5,
              "A0_reference_values": a0["values"], "A0_per_class_APL_top600": a0["per_class_APL_top600"],
              "detector_forwards": 1, "score_calls": 0, "evaluation_policy": contract["evaluation_policy"],
              "cost": {"wall_seconds": time.perf_counter() - started, "cpu_seconds": time.process_time() - cpu0}}
    publish(out / "EVALUATION_CPU_CHECKS.json", result)
    return result


def evaluate(contract, train_root, out):
    from s15_formal_run import load_contract
    from s15_formal_train import read_endpoint
    from r25_run import setup
    _validate_policy(contract)
    source = load_contract(train_root["contract"])
    require(source["formal_identity"] == contract["formal_identity"], "S15 endpoint formal identity differs")
    path = Path(train_root["path"]) / "checkpoint-epoch-120.pth"
    sidecar = read(path.with_suffix(".pth.json"))
    require((sidecar["status"], sidecar["arm"], sidecar["contract_sha256"], sidecar["formal_identity"],
             sidecar["global_update"], sidecar["scope"]) ==
            ("COMPLETE", ARM, source["contract_sha256"], source["formal_identity"], ENDPOINT, "formal")
            and Path(sidecar["file"]["path"]).resolve() == path.resolve(), "S15 evaluator endpoint sidecar identity")
    setup("cuda:0")
    results = {}
    for kind in KINDS:
        stage = out / f"S15-{kind}"
        stage.mkdir()
        results[kind] = infer(contract, sidecar["file"], kind, stage, lambda p: read_endpoint(source, p))
    return score(contract, results, out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("evaluate", "cpu-check"))
    parser.add_argument("--contract", required=True)
    parser.add_argument("--train-root")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from s15_formal_run import load_contract
    c = load_contract(args.contract)
    out = Path(args.out).absolute()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        require(Path(c["source_root"]).resolve() == ROOT.resolve(), "S15 evaluator release directory")
        if args.action == "cpu-check":
            import os
            require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "S15 CPU evaluation gate requires hidden CUDA")
            torch.set_num_threads(int(os.environ.get("S15_CPU_THREADS", "8")))
            result = cpu_checks(c, out)
            path = out / "EVALUATION_CPU_CHECKS.json"
        else:
            require(bool(args.train_root), "S15 evaluator requires the train mapping")
            result = evaluate(c, read(args.train_root), out)
            path = out / "COMPARISON.json"
        require(result["status"] == "PASS", "S15 evaluator result incomplete")
        publish(out / "CLOSED.json", {"status": "PASS", "mode": "eval", "contract_sha256": c["contract_sha256"],
                                      "result": reference(path), "wall_seconds": time.perf_counter() - started})
    except BaseException as exc:
        publish(out / "STOP.json", {"status": "STOP", "contract_sha256": c["contract_sha256"],
                                    "error": str(exc)[:4000], "traceback": traceback.format_exc(),
                                    "wall_seconds": time.perf_counter() - started})
        raise


if __name__ == "__main__":
    main()
