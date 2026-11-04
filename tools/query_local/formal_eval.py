"""Fixed R25 e120 readout and comparisons. No training or tuning routes."""
from __future__ import annotations

import argparse
import gc
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

import torch

from infer_seed import BUDGETS, METRICS, _group, _prefix, bridge_values, load_original, pixels
from sparse_rtdetr.localization.runio import event, publish, read, reference, require
from sparse_rtdetr.protocol_r18 import checkpoint as native
from sparse_rtdetr.protocol_r18.engine import tensor_digest
from sparse_rtdetr.protocol_r18.geometry import forward_guards
from sparse_rtdetr.protocol_r18.model import Config, build
from sparse_rtdetr.r0.evaluation import POLICY, rank_candidates, score_budget
from sparse_rtdetr.r0.evaluation_data import Development548
from sparse_rtdetr.research.query_local import VERSION, install
from sparse_rtdetr.research.query_local_runtime import PARENT_SHA

ARMS = ("A0", "A1", "A2")
KINDS = ("EMA", "online")
BATCH = 4
WARMUPS = 4
ENDPOINT = 36480
PROTECTIONS = ((600, "AP", -0.10), (300, "AP", -0.20), (300, "APS", -0.20),
               (600, "APM", -0.50), (600, "APL", -0.50))
PRIMARY = {
    "A2-A0": ((600, "APS", 0.50), (600, "AP75_small", 0.30)),
    "A2-A1": ((600, "APS", 0.30), (600, "AP75_small", 0.20)),
}
EVALUATION = {
    "epoch": 120, "weights": ["ema", "online"], "primary_weights": "ema",
    "view": "A", "images": 548, "batch": BATCH, "precision": "float32",
    "warmup_batches_per_endpoint": WARMUPS, "budgets": list(BUDGETS), "score_calls": 12,
}


def _validate_policy(contract):
    require(contract["workload"]["evaluation"] == EVALUATION,
            "R25 endpoint workload differs")
    policy = contract["evaluation_policy"]
    require(reference(policy["path"]) == policy and read(policy["path"]) == POLICY,
            "R25 certified view-A evaluation policy differs")



def development(contract):
    data = Development548(contract["data"]["development"])
    require([r["stable_image_id"] for r in data.records] == contract["development_ids"],
            "R25 development members or order changed")
    return data


def _weights(payload, arm, kind, cfg):
    require(arm in ARMS and kind in KINDS, "R25 evaluation arm or endpoint kind")
    require((payload["version"], payload["arm"], payload["parent_sha256"], payload["config"]) ==
            (VERSION, arm, PARENT_SHA, cfg.binding()), "R25 evaluation state identity")
    require(payload["engine"]["completed_updates"] == payload["ema"]["updates"] == ENDPOINT,
            "R25 evaluation requires both online and EMA e120")
    state = payload["ema"]["module"] if kind == "EMA" else payload["model"]
    require(any(k.startswith("local_head.") for k in state) == (arm != "A0"),
            "R25 evaluation local parameter presence differs")
    require(all(not v.is_floating_point() or v.dtype == torch.float32 for v in state.values()),
            "R25 evaluation floating state is not FP32")
    return state


def _load_model(payload, arm, kind, device, cfg=Config()):
    state = _weights(payload, arm, kind, cfg)
    model, _, _ = build(cfg, synthetic=True)
    install(model, arm)
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False).to(device)
    require((arm == "A0" and not hasattr(model, "local_head")) or
            (arm != "A0" and model.local_head.mode == arm), "R25 evaluation install differs")
    expected = tensor_digest(state.items())
    require(tensor_digest(model.state_dict().items()) == expected, "R25 evaluation strict load differs")
    return model, expected


def _image(data, row):
    with load_original(data, row) as image:
        return pixels(data, image)


def _rank_batch(logits, boxes, rows):
    six = [rank_candidates(logit, box, row, 600) for logit, box, row in zip(logits, boxes, rows)]
    return {600: [p for group in six for p in group],
            300: [p for group in six for p in group[:300]]}


def infer(contract, endpoint_ref, arm, kind, out, read_endpoint):
    """Read one validated e120 checkpoint and evaluate one weight selection."""
    counts = {"attempted_forwards": 0, "completed_forwards": 0}
    started = time.perf_counter()
    try:
        return _infer(contract, endpoint_ref, arm, kind, out, read_endpoint, counts)
    except Exception as exc:
        peak = {"peak_allocated_bytes": None, "peak_reserved_bytes": None}
        try:
            peak = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
        except Exception:
            pass
        publish(out / "STOP.json", {"status": "STOP", "arm": arm, "kind": kind,
                                    "checkpoint": endpoint_ref, **counts, **peak,
                                    "elapsed_seconds": time.perf_counter()-started,
                                    "error": str(exc), "traceback": traceback.format_exc()})
        raise


def _infer(contract, endpoint_ref, arm, kind, out, read_endpoint, counts):
    device = torch.device("cuda:0")
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    payload = read_endpoint(endpoint_ref["path"], arm)
    model, weight_sha = _load_model(payload, arm, kind, device)
    del payload
    data = development(contract)
    rng = native.capture_rng(device)
    torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - started

    def batch(rows):
        start = time.perf_counter()
        x = torch.stack([_image(data, row) for row in rows]).to(device)
        torch.cuda.synchronize(device)
        input_done = time.perf_counter()
        counts["attempted_forwards"] += 1
        with torch.autocast("cuda", enabled=False), forward_guards(model):
            result = model(x)
        torch.cuda.synchronize(device)
        counts["completed_forwards"] += 1
        forward_done = time.perf_counter()
        require(set(result) == {"pred_boxes", "pred_logits"}
                and result["pred_boxes"].shape == (BATCH, 300, 4)
                and result["pred_logits"].shape == (BATCH, 300, 10)
                and result["pred_boxes"].dtype == result["pred_logits"].dtype == torch.float32,
                "R25 endpoint output schema")
        boxes = result["pred_boxes"].cpu()
        logits = result["pred_logits"].cpu()
        output_done = time.perf_counter()
        predictions = _rank_batch(logits, boxes, rows)
        rank_done = time.perf_counter()
        return predictions, {"data_and_h2d": input_done-start,
                             "forward": forward_done-input_done,
                             "cpu_materialization": output_done-forward_done,
                             "ranking": rank_done-output_done,
                             "total": rank_done-start}

    warmup = []
    measured = []
    all_predictions = {budget: [] for budget in BUDGETS}
    with torch.inference_mode():
        for batch_index in range(WARMUPS):
            rows = data.records[batch_index*BATCH:(batch_index+1)*BATCH]
            _, timing = batch(rows)
            warmup.append(timing)
            event(out, "endpoint_warmup", arm=arm, endpoint_kind=kind, batch=batch_index+1)
        for offset in range(0, len(data.records), BATCH):
            predictions, timing = batch(data.records[offset:offset+BATCH])
            measured.append(timing)
            for budget in BUDGETS:
                all_predictions[budget].extend(predictions[budget])
            event(out, "endpoint_batch", arm=arm, endpoint_kind=kind, batch=offset//BATCH+1)
    require(len(warmup) == WARMUPS and len(measured) == 137
            and counts["attempted_forwards"] == counts["completed_forwards"] == WARMUPS + 137,
            "R25 endpoint batch count")
    require(tensor_digest(model.state_dict().items()) == weight_sha
            and native.exact_equal(native.capture_rng(device), rng)
            and all(not m.training for m in model.modules())
            and all(p.grad is None for p in model.parameters()),
            "R25 evaluation modified model state or RNG")
    summaries = {name: sum(row[name] for row in measured) for name in measured[0]}
    prediction_refs = {}
    for budget in BUDGETS:
        path = out / f"top{budget}-predictions.json"
        publish(path, all_predictions[budget])
        prediction_refs[str(budget)] = reference(path)
    result = {"status": "PASS", "arm": arm, "kind": kind, "epoch": 120, "update": ENDPOINT,
              "view": "A", "device": str(device), "FP32": True, "batch_size": BATCH,
              "contract_sha256": contract["contract_sha256"], "checkpoint": endpoint_ref,
              "weights_sha256": weight_sha, "predictions": prediction_refs,
              "images": len(data.records), "steady_forwards": len(measured), "warmup_forwards": len(warmup),
              "load_seconds": load_seconds, "warmup_seconds": sum(r["total"] for r in warmup),
              "warmup_timing": warmup, "cold_first_forward_seconds": warmup[0]["forward"],
              "steady_timing": summaries, "elapsed_seconds": time.perf_counter()-started,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
              "forward_timing_includes_native_geometry_guards": True,
              "model_and_RNG_unchanged": True, **counts}
    publish(out / "RESULT.json", result)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def compare(values):
    differences = {}
    for kind in KINDS:
        differences[kind] = {}
        for lhs, rhs in (("A2", "A0"), ("A2", "A1"), ("A1", "A0")):
            differences[kind][lhs + "-" + rhs] = {
                str(budget): {
                    metric: 100. * (values[kind][lhs][str(budget)][metric] -
                                    values[kind][rhs][str(budget)][metric])
                    for metric in METRICS
                } for budget in BUDGETS
            }
    gates = {}
    for pair, limits in PRIMARY.items():
        cells = {}
        for budget, metric, minimum in (*limits, *PROTECTIONS):
            observed = differences["EMA"][pair][str(budget)][metric]
            cells[f"top{budget}_{metric}"] = {"difference_pp": observed, "minimum_pp": minimum,
                                              "passed": observed >= minimum}
        gates[pair] = {"passed": all(v["passed"] for v in cells.values()), "cells": cells}
    if not gates["A2-A0"]["passed"]:
        decision = "closed_fixed_config"
    elif not gates["A2-A1"]["passed"]:
        decision = "single_seed_net_benefit_instance_conditioning_unproven"
    else:
        decision = "single_seed_preliminary_net_and_incremental_benefit"
    return differences, gates, decision


def score(contract, inference_results, out):
    started = time.perf_counter()
    data = development(contract)
    rows = data.records
    ids = contract["development_ids"]
    parsed = [data.raw(row) for row in rows]
    values = {kind: {} for kind in KINDS}
    metric_refs = {kind: {} for kind in KINDS}
    scoring = []
    for kind in KINDS:
        for arm in ARMS:
            result = inference_results[kind][arm]
            require((result["status"], result["arm"], result["kind"], result["contract_sha256"],
                     result["images"], result["steady_forwards"], result["warmup_forwards"]) ==
                    ("PASS", arm, kind, contract["contract_sha256"], 548, 137, WARMUPS),
                    "R25 scoring inference identity")
            predictions = {}
            grouped = {}
            values[kind][arm] = {}
            metric_refs[kind][arm] = {}
            for budget in BUDGETS:
                ref = result["predictions"][str(budget)]
                require(reference(ref["path"]) == ref, "R25 prediction reference differs")
                predictions[budget] = read(ref["path"])
                grouped[budget] = _group(predictions[budget], ids, budget, whole=True, exact=True)
            _prefix(grouped[600], grouped[300], ids, whole=True)
            for budget in BUDGETS:
                score_started = time.perf_counter()
                event(out, "score_started", ordinal=len(scoring)+1, arm=arm, endpoint_kind=kind,
                      budget=budget)
                metrics = score_budget(rows, parsed, predictions[budget], contract["source_root"])
                metrics["R25_binding"] = {"contract_sha256": contract["contract_sha256"],
                                          "arm": arm, "kind": kind, "view": "A", "epoch": 120,
                                          "budget": budget, "checkpoint": result["checkpoint"],
                                          "weights_sha256": result["weights_sha256"],
                                          "predictions": result["predictions"][str(budget)]}
                values[kind][arm][str(budget)] = bridge_values(metrics)
                path = out / f"{arm}-{kind}-A-top{budget}-metrics.json"
                publish(path, metrics)
                metric_refs[kind][arm][str(budget)] = reference(path)
                timing = {"arm": arm, "kind": kind, "budget": budget,
                          "elapsed_seconds": time.perf_counter()-score_started}
                scoring.append(timing)
                event(out, "score_complete", ordinal=len(scoring), arm=arm, endpoint_kind=kind,
                      budget=budget, elapsed_seconds=timing["elapsed_seconds"])
    require(len(scoring) == EVALUATION["score_calls"], "R25 fixed score count differs")
    differences, gates, decision = compare(values)
    base = inference_results["EMA"]["A0"]["steady_timing"]
    efficiency = {arm+"/A0": {key+"_ratio": inference_results["EMA"][arm]["steady_timing"][key]/base[key]
                              for key in ("forward", "total")} for arm in ("A1", "A2")}
    comparison = {"status": "PASS", "scientific_decision": decision,
                  "scores": values, "differences_pp": differences, "EMA_gates": gates,
                  "score_units": "0..1 COCO bridge", "delta_units": "percentage points",
                  "metrics": metric_refs, "steady_EMA_efficiency": efficiency,
                  "scoring": scoring, "score_calls": len(scoring),
                  "scoring_wall_seconds": time.perf_counter()-started,
                  "inference": inference_results, "contract_sha256": contract["contract_sha256"]}
    publish(out / "COMPARISON.json", comparison)
    return comparison


def cpu_checks(contract, out):
    """Strict e120 endpoint selection, actual view-A pixels and small ranking checks."""
    from sparse_rtdetr.baseline.primary_evaluator import primary_evaluator_contract_binding
    require(not torch.cuda.is_initialized(), "R25 CPU endpoint checks initialized CUDA")
    _validate_policy(contract)
    data = development(contract)
    row = data.records[0]
    x = _image(data, row)
    certified, _ = data.item(row, 0)
    require(x.dtype == torch.float32 and tuple(x.shape) == (3, 768, 1344)
            and torch.equal(x, certified), "R25 GT-free pixels differ from certified view A")
    primary_evaluator_contract_binding(contract["source_root"])
    cfg = Config(input_hw=(128, 128), physical_batch_size=1, amp_dtype="float32")
    checks = {}
    for arm in ARMS:
        model, _, _ = build(cfg, synthetic=True)
        install(model, arm)
        online = model.state_dict()
        key = "local_head.out.bias" if arm != "A0" else next(n for n, _ in model.named_parameters())
        ema = {**online, key: online[key] + 0.001}
        fixture = {"version": VERSION, "arm": arm, "parent_sha256": PARENT_SHA,
                   "config": cfg.binding(), "engine": {"completed_updates": ENDPOINT},
                   "model": online, "ema": {"module": ema, "updates": ENDPOINT}}
        checks[arm] = {}
        for kind, expected in (("EMA", ema), ("online", online)):
            selected, sha = _load_model(fixture, arm, kind, "cpu", cfg)
            require(torch.equal(selected.state_dict()[key], expected[key]),
                    "R25 endpoint selected wrong state")
            checks[arm][kind] = sha
        require(checks[arm]["EMA"] != checks[arm]["online"], "R25 endpoint selection fixture degenerate")
    logits = torch.zeros((1, 300, 10))
    boxes = torch.full((1, 300, 4), 0.5)
    preds = _rank_batch(logits, boxes, [row])
    ids = [row["stable_image_id"]]
    grouped = {b: _group(preds[b], ids, b, whole=True, exact=True) for b in BUDGETS}
    _prefix(grouped[600], grouped[300], ids, whole=True)
    require(preds[300] == rank_candidates(logits[0], boxes[0], row, 300),
            "R25 ranked budget prefix differs from native top300")
    values = {kind: {arm: {str(b): {m: 0.4 for m in METRICS} for b in BUDGETS}
                     for arm in ARMS} for kind in KINDS}
    require(compare(values)[2] == "closed_fixed_config", "R25 zero gain gate")
    values["EMA"]["A2"]["600"].update(APS=0.41, AP75_small=0.41)
    require(compare(values)[2] == "single_seed_preliminary_net_and_incremental_benefit",
            "R25 controlled gain gate")
    values["EMA"]["A1"]["600"].update(APS=0.409, AP75_small=0.409)
    require(compare(values)[2] == "single_seed_net_benefit_instance_conditioning_unproven",
            "R25 mechanism protection gate")
    values["EMA"]["A2"]["300"]["AP"] = 0.39
    require(compare(values)[2] == "closed_fixed_config", "R25 main protection gate")
    require(not torch.cuda.is_initialized(), "R25 CPU endpoint checks initialized CUDA")
    result = {"status": "PASS", "development_images": len(data.records),
              "one_actual_development_image_pixels_bitwise_equal": ids[0],
              "strict_disjoint_weight_selections": checks, "ranking_ties_and_prefix_exact": True,
              "gate_branches": 4, "detector_forwards": 0, "score_calls": 0,
              "evaluation_policy": contract["evaluation_policy"],
              "development_manifest": reference(contract["data"]["development"]["manifest"])}
    publish(out / "EVALUATION_CPU_CHECKS.json", result)
    return result


def evaluate(contract, train_roots, out):
    from formal_run import load_contract
    from formal_train import read_endpoint
    from r25_run import setup

    _validate_policy(contract)
    require(set(train_roots) == set(ARMS), "R25 evaluator needs exactly three formal arms")
    refs = {}
    contracts = {}
    for arm in ARMS:
        entry = train_roots[arm]
        source = load_contract(entry["contract"])
        require(source["formal_identity"] == contract["formal_identity"],
                "R25 prior endpoint formal identity differs")
        path = Path(entry["path"]) / "checkpoint-epoch-120.pth"
        sidecar = read(path.with_suffix(".pth.json"))
        require((sidecar["status"], sidecar["arm"], sidecar["contract_sha256"],
                 sidecar["formal_identity"], sidecar["global_update"]) ==
                ("COMPLETE", arm, source["contract_sha256"], source["formal_identity"], ENDPOINT)
                and Path(sidecar["file"]["path"]).resolve() == path.resolve(),
                "R25 evaluator formal endpoint sidecar identity")
        refs[arm] = sidecar["file"]
        contracts[arm] = source

    setup("cuda:0")
    results = {kind: {} for kind in KINDS}
    for kind in KINDS:
        for arm in ARMS:
            stage = out / f"{arm}-{kind}"
            stage.mkdir()
            results[kind][arm] = infer(contract, refs[arm], arm, kind, stage,
                                       lambda path, a: read_endpoint(contracts[a], path, a))
    return score(contract, results, out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("evaluate", "cpu-check"))
    parser.add_argument("--contract", required=True)
    parser.add_argument("--train-root")
    parser.add_argument("--train-roots")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from formal_run import load_contract

    c = load_contract(args.contract)
    out = Path(args.out).absolute()
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        require(Path(c["source_root"]).resolve() == ROOT.resolve(), "R25 evaluator release directory")
        if args.action == "cpu-check":
            torch.set_num_threads(2)
            result = cpu_checks(c, out)
            path = out / "EVALUATION_CPU_CHECKS.json"
        else:
            require(bool(args.train_root) != bool(args.train_roots), "R25 evaluator requires one train mapping")
            train_roots = read(args.train_roots) if args.train_roots else {
                arm: {"path": str(Path(args.train_root) / f"train-{arm}"), "contract": args.contract}
                for arm in ARMS
            }
            result = evaluate(c, train_roots, out)
            path = out / "COMPARISON.json"
        require(result["status"] == "PASS", "R25 evaluator result incomplete")
        publish(out / "CLOSED.json", {"status": "PASS", "mode": "eval", "contract_sha256": c["contract_sha256"],
                                      "result": reference(path), "wall_seconds": time.perf_counter()-started})
    except Exception as exc:
        publish(out / "STOP.json", {"status": "STOP", "contract_sha256": c["contract_sha256"],
                                    "error": str(exc), "traceback": traceback.format_exc(),
                                    "wall_seconds": time.perf_counter()-started})
        raise


if __name__ == "__main__":
    main()
