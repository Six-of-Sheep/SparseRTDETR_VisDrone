"""R27-002 L0 control: fixed endpoint evaluation and the summary (actions `evaluate` and `summarize` of l0_run.py).

evaluate: only this round's L0 e120 EMA (K.endpoint_state), view A, the frozen 548 development images at the same H
canvas as P (the model reduces H to L itself), FP32, batch 4, 4 warm-up + 137 batches = 141 forwards, top600 with
top300 as its ordered prefix, two score_budget calls and one per-class APL COCOeval (top600 only).  There is no
argument for another arm, epoch, weight kind, view or budget.
summarize: reads CLOSED.json files of this round and the contract's reference block; no model forward, no scoring.
"""
import gc
import statistics
import time
from pathlib import Path

import l0_common as K
import l0_summary as M

EVAL_STAGE, COST_STAGE = "eval-L0", "cost4"
CHECK_STAGES = {"cpu_fixture": "cpu", "cuda_zero_step": "zero", "cuda_accept": "accept"}
EPOCH = K.ENDPOINT // K.WINDOWS_PER_EPOCH


def formal_stage(c):
    """Name of the closed formal directory: K.formal_dir(c) when the common layer provides it, else formal-L0."""
    name = K.formal_dir(c).name if hasattr(K, "formal_dir") else "formal-L0"
    K.stage_closed(c, name)
    return name


def round_stage(out, name):
    out = Path(out).resolve()
    K.require(out.parent == K.ROUND.resolve() and out.name == name, f"this action writes <round>/{name}")
    return out


def evaluate(c, out, counts):
    from infer_seed import bridge_values, load_original, pixels, _group, _prefix
    from sparse_rtdetr.research import r27_l0_control as L
    from sparse_rtdetr.research import scale_baseline as S
    from sparse_rtdetr.protocol_r18 import checkpoint as ck
    from sparse_rtdetr.protocol_r18.engine import tensor_digest
    from sparse_rtdetr.protocol_r18.geometry import forward_guards
    from sparse_rtdetr.r0.evaluation import rank_candidates, score_budget
    from s15_formal_eval import per_class_large
    torch = K.torch_mod()
    K.PHASE["phase"] = "evaluate/guards"
    out = round_stage(out, EVAL_STAGE)
    formal = formal_stage(c)
    side, state = K.endpoint_state(c)  # e120 EMA module only; nothing else is reachable from here
    K.require(side["arm"] == K.ARM and side["scope"] == "formal" and side["global_update"] == K.ENDPOINT, "endpoint side-car")
    K.require(all(not v.is_floating_point() or v.dtype == torch.float32 for v in state.values()), "EMA not FP32")
    device = torch.device("cuda:0")
    model = L.build_eval_model(state, device)
    K.require(getattr(model, "l0_arm", None) == K.ARM and not model.training, "evaluation model is not an L0 eval model")
    weights = tensor_digest(state.items())
    model.register_forward_pre_hook(lambda *_: counts.update(forwards=1))
    data = S.development(c["data"]["development"], S.S15_HW)
    ids = c["development_ids"]
    K.require([r["stable_image_id"] for r in data.records] == ids and len(ids) == 548, "development members/order")
    K.require(len(ids) == K.EVAL_BATCH * K.EVAL_BATCHES, "548 images are 137 full batches of 4")
    rng = ck.capture_rng(device)
    preds = {b: [] for b in K.BUDGETS}
    timings = []
    K.PHASE["phase"] = "evaluate/inference"
    started = time.perf_counter()

    def batch(rows):
        t0 = time.perf_counter()
        imgs = []
        for row in rows:
            with load_original(data, row) as image:
                imgs.append(pixels(data, image))
        x = torch.stack(imgs).to(device)
        K.require(tuple(x.shape[-2:]) == tuple(S.S15_HW) and x.dtype == torch.float32, "evaluation input is not the FP32 H canvas")
        with torch.autocast("cuda", enabled=False), forward_guards(model):
            o = model(x)
        K.require(o["pred_boxes"].shape == (len(rows), 300, 4) and o["pred_logits"].dtype == torch.float32
                  and o["pred_boxes"].dtype == torch.float32, "eval output")
        boxes, logits = o["pred_boxes"].cpu(), o["pred_logits"].cpu()
        six = [rank_candidates(lg, bx, row, 600) for lg, bx, row in zip(logits, boxes, rows)]
        timings.append(time.perf_counter() - t0)
        return {600: [p for g in six for p in g], 300: [p for g in six for p in g[:300]]}

    with torch.inference_mode():
        for i in range(K.EVAL_WARMUP):
            batch(data.records[i * K.EVAL_BATCH:(i + 1) * K.EVAL_BATCH])
        for off in range(0, len(data.records), K.EVAL_BATCH):
            r = batch(data.records[off:off + K.EVAL_BATCH])
            for b in K.BUDGETS:
                preds[b].extend(r[b])
    K.require(counts["forwards"] == K.EVAL_WARMUP + K.EVAL_BATCHES == 141, "evaluation forward count")
    K.require(tensor_digest(model.state_dict().items()) == weights and ck.exact_equal(ck.capture_rng(device), rng),
              "evaluation changed model/RNG")
    infer_seconds = time.perf_counter() - started
    del model
    gc.collect()
    torch.cuda.empty_cache()
    K.PHASE["phase"] = "evaluate/predictions"
    K.require(all(len(preds[b]) == len(ids) * b for b in K.BUDGETS), "prediction record counts")
    refs = {}
    for b in K.BUDGETS:
        p = out / f"top{b}-predictions.json"
        K.write_new(p, preds[b])
        refs[str(b)] = K.ref(p)
    grouped = {b: _group(preds[b], ids, b, whole=True, exact=True) for b in K.BUDGETS}
    _prefix(grouped[600], grouped[300], ids, whole=True)
    rows = data.records
    parsed = [data.raw(row) for row in rows]
    values, metrics_refs, counts_gt, scoring = {}, {}, {}, []
    per_class = None
    K.PHASE["phase"] = "evaluate/scoring"
    for b in K.BUDGETS:
        t = time.perf_counter()
        metrics = score_budget(rows, parsed, preds[b], c["source_root"])
        counts.update(score_budget_calls=1)
        metrics["L0_binding"] = {"contract_sha256": c["contract_sha256"], "arm": K.ARM, "kind": "EMA", "view": "A",
                                 "epoch": EPOCH, "budget": b, "checkpoint": side["file"], "weights_sha256": weights,
                                 "predictions": refs[str(b)]}
        values[str(b)] = bridge_values(metrics)
        counts_gt[str(b)] = {k: metrics["spatial_filter_counts"][k] for k in ("GT_before", "GT_after")}
        p = out / f"L0-EMA-A-top{b}-metrics.json"
        K.write_new(p, metrics)
        metrics_refs[str(b)] = K.ref(p)
        scoring.append({"budget": b, "seconds": time.perf_counter() - t})
        if b == 600:
            per_class = per_class_large(rows, parsed, preds[b], metrics["COCO_bridge"])
            counts.update(per_class_COCOeval_calls=1)
    K.require(counts["score_budget_calls"] == 2 and counts["per_class_COCOeval_calls"] == 1, "scorer call counts")
    s15_counts = {str(b): {k: K.read_json(K.pinned(c["S15_metrics"][str(b)]))["spatial_filter_counts"][k]
                           for k in ("GT_before", "GT_after")} for b in K.BUDGETS}
    K.require(counts_gt == s15_counts, f"scored GT set differs from S15: {counts_gt} vs {s15_counts}")
    return {"endpoint": side["file"], "formal_stage": formal, "weights_sha256": weights, "kind": "EMA", "epoch": EPOCH,
            "view": "A", "values": values, "metrics": metrics_refs, "predictions": refs, "GT_counts": counts_gt,
            "per_class_APL_top600": per_class, "scoring": scoring, "inference_seconds": infer_seconds,
            "batch_seconds_median": statistics.median(timings)}


def summarize(c, out, counts):
    K.PHASE["phase"] = "summarize/read"
    out = round_stage(out, "summary")
    formal = formal_stage(c)
    stages = {**CHECK_STAGES, "formal": formal, "evaluation": EVAL_STAGE, "cost4": COST_STAGE}
    closed = {k: K.stage_closed(c, name) for k, name in stages.items()}
    ev, cost_closed = closed["evaluation"], closed["cost4"]
    K.require(ev["action"] == "evaluate" and cost_closed["action"] == "cost4", "eval-L0 / cost4 actions")
    side = K.read_side(c, K.ROUND / formal / f"checkpoint-epoch-{EPOCH:03d}.pth", "formal")[1]  # side-car only, no tensors
    K.require(ev["endpoint"] == side["file"] and ev.get("formal_stage") == formal, "eval-L0 scored another endpoint file")
    for b in K.BUDGETS:
        m = K.read_json(K.pinned(ev["metrics"][str(b)]))
        K.require(m["L0_binding"]["weights_sha256"] == ev["weights_sha256"] and m["L0_binding"]["budget"] == b
                  and m["L0_binding"]["contract_sha256"] == c["contract_sha256"], "L0 metrics binding")
        K.pinned(ev["predictions"][str(b)])
    K.require(K.ref(K.R27_RESULT) == c["R27_result"], "R27 RESULT.json identity")
    old = K.read_json(K.R27_RESULT)
    r27 = {"file": c["R27_result"], "label": old["label"], "E_passed": old["E_P_minus_S15"]["passed"],
           "I_passed": old["I_P_minus_C"]["passed"], "G_passed": old["G_P_minus_A0"]["passed"]}
    f = closed["formal"]
    others = [{"directory": d.name, "STOP": K.read_json(d / "STOP.json") if (d / "STOP.json").is_file() else None}
              for d in sorted(K.ROUND.iterdir()) if d.is_dir() and d.name != formal and d.name.startswith("formal-L0")]
    chain = {"closed_directory": formal, "resumed": f["resumed"], "start_update": f["plan"]["start_update"],
             "stop_update": f["plan"]["stop_update"], "updates_in_closed_process": f["updates"],
             "updates_before_the_resume_state": f["plan"]["start_update"] - K.PARENT_UPDATE, "ledger_rows": f["ledger_rows"],
             "other_formal_directories": [{"directory": o["directory"], "status": (o["STOP"] or {}).get("status"),
                                           "counts": (o["STOP"] or {}).get("counts")} for o in others]}
    K.require(f["plan"]["stop_update"] == K.ENDPOINT and f["ledger_rows"] == K.ENDPOINT - K.PARENT_UPDATE, "formal coverage")
    K.PHASE["phase"] = "summarize/compute"
    res = M.build(c, ev, cost_closed, {k: v["counts"] for k, v in closed.items()}, r27, chain)
    res["stages"] = {k: {"directory": name, "CLOSED": K.ref(K.ROUND / name / "CLOSED.json"),
                         "elapsed_seconds": closed[k]["elapsed_seconds"],
                         "process_cpu_seconds": closed[k].get("process_cpu_seconds")} for k, name in stages.items()}
    K.require(counts["forwards"] == 0 and counts["actual_updates"] == 0 and not counts.get("score_budget_calls"),
              "summary must not run a model or a scorer")
    K.write_new(out / "RESULT.json", res)
    d = res["D_P_minus_L0"]
    return {"result": K.ref(out / "RESULT.json"), "D_passed": d["passed"], "label": d["label"],
            "D_difference_pp": {k: v["difference_pp"] for k, v in d["cells"].items()},
            "workload_all_equal": res["workload"]["all_equal"]}
