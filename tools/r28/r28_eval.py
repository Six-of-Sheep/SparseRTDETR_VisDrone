"""R28 fixed endpoint evaluation and summaries (actions `evaluate` and `summarize`).

evaluate: the arm's e120 EMA (K.endpoint_state), view A, the frozen 548 development images on the H canvas (the model
resamples to L itself), FP32, batch 4, 4 warm-up + 137 batches, top600 with top300 as its ordered prefix, two
score_budget calls and one per-class APL COCOeval.  `infer` is shared with the evaluation-path check.
summarize: reads CLOSED.json files and the contract's reference block only (no model, no scorer).
  arm W -> <round>/summary-W : M (W - L0, seven cells), APL6, C, the registered label and the K trigger.
  arm K -> <round>/summary-K : everything of summary-W (copied) plus K - L0, W - K and A.
"""
import gc
import math
import statistics
import time
from pathlib import Path

import r28_common as K

METRICS = ("AP", "APS", "AP75_small", "APM", "APL", "AP50", "AP75")
STATEMENTS = {
    "single_seed": "单 seed（seed0）、重复使用的 548 图 development 集上的初步结果，不提供跨 seed 或独立泛化证明。",
    "G0": "G0 是 development 上的启发式投入门，参与了是否启动 W 的选择；它不是 AP 收益判据。",
    "design_exposure": "ε 与斜率界在任务下达前已用 development 标注做过纯标注探索（已登记的设计过程）。",
    "W_minus_L0": "W−L0 检验完整的实用净收益（需求网络＋分配＋重采样＋逆映的整体），不单独证明分配公式相对已有采样器的贡献。",
    "W_minus_K": "W−K 是逐图自适应与固定先验的对照；不达门限只登记 adaptive_increment_not_established，不称等效；"
                 "K 不能排除新增需求网络的作用。",
    "APL6": "APL6 是事前登记的探索性投入规则读数，不替代原 APL600 格；M_failed_APL 不豁免原质量 FAIL。",
    "closed": "固定配置未达条件只关闭本固定配置，不扩大为非均匀重采样整族无效。",
    "cost": "C 以固定端点同进程完整模型 GPU 时间为准（含需求网络、分配、重采样、逆映）；端到端时间另列。",
}


def round_stage(out, name):
    out = Path(out).resolve()
    K.require(out.parent == K.ROUND.resolve() and out.name == name, f"this action writes <round>/{name}")
    return out


def infer(c, model, counts, *, extra=None):
    """4 warm-up + 137 batches over the frozen 548 images; returns predictions per budget and timings.
    extra(x, rows, index) is called after each scored batch (used by the evaluation-path check)."""
    from infer_seed import load_original, pixels
    from sparse_rtdetr.research import scale_baseline as S
    from sparse_rtdetr.protocol_r18.geometry import forward_guards
    from sparse_rtdetr.r0.evaluation import rank_candidates
    torch = K.torch_mod()
    device = torch.device("cuda:0")
    data = S.development(c["data"]["development"], S.S15_HW)
    ids = c["development_ids"]
    K.require([r["stable_image_id"] for r in data.records] == ids and len(ids) == 548, "development members/order")
    K.require(len(ids) == K.EVAL_BATCH * K.EVAL_BATCHES, "548 images are 137 full batches of 4")
    preds = {b: [] for b in K.BUDGETS}
    timings = []

    def batch(rows, index=None):
        t0 = time.perf_counter()
        imgs = []
        for row in rows:
            with load_original(data, row) as image:
                imgs.append(pixels(data, image))
        x = torch.stack(imgs).to(device)
        K.require(tuple(x.shape[-2:]) == tuple(S.S15_HW) and x.dtype == torch.float32, "evaluation input is not the FP32 H canvas")
        with torch.autocast("cuda", enabled=False), forward_guards(model):
            o = model(x)
        counts.update(forwards=1)
        K.require(o["pred_boxes"].shape == (len(rows), 300, 4) and o["pred_logits"].dtype == torch.float32
                  and o["pred_boxes"].dtype == torch.float32 and set(o) == {"pred_logits", "pred_boxes"}, "eval output")
        boxes, logits = o["pred_boxes"].cpu(), o["pred_logits"].cpu()
        six = [rank_candidates(lg, bx, row, 600) for lg, bx, row in zip(logits, boxes, rows)]
        timings.append(time.perf_counter() - t0)
        if extra is not None and index is not None:
            extra(x, rows, index)
        return {600: [p for g in six for p in g], 300: [p for g in six for p in g[:300]]}

    started = time.perf_counter()
    with torch.inference_mode():
        for i in range(K.EVAL_WARMUP):
            batch(data.records[i * K.EVAL_BATCH:(i + 1) * K.EVAL_BATCH])
        for n, off in enumerate(range(0, len(data.records), K.EVAL_BATCH)):
            r = batch(data.records[off:off + K.EVAL_BATCH], n)
            for b in K.BUDGETS:
                preds[b].extend(r[b])
    K.require(all(len(preds[b]) == len(ids) * b for b in K.BUDGETS), "prediction record counts")
    return data, preds, {"inference_seconds": time.perf_counter() - started, "batch_seconds_median": statistics.median(timings)}


def evaluate(c, out, counts):
    from infer_seed import bridge_values, _group, _prefix
    from sparse_rtdetr.research import r28_resample as R
    from sparse_rtdetr.protocol_r18 import checkpoint as ck
    from sparse_rtdetr.protocol_r18.engine import tensor_digest
    from sparse_rtdetr.r0.evaluation import score_budget
    from s15_formal_eval import per_class_large
    torch = K.torch_mod()
    arm = K.ARM
    K.PHASE["phase"] = "evaluate/guards"
    out = round_stage(out, f"eval-{arm}")
    formal = K.formal_dir(c, arm).name
    side, state = K.endpoint_state(c, arm)
    K.require(side["arm"] == arm and side["scope"] == "formal", "endpoint side-car")
    K.require(all(not v.is_floating_point() or v.dtype == torch.float32 for v in state.values()), "EMA not FP32")
    source = K.demand_source(c, arm)
    K.require(side["demand"]["file"] == source["file"], "the endpoint was trained with another demand source")
    device = torch.device("cuda:0")
    model = R.build_eval_model(state, device, arm, source)
    K.require(R.is_wrapped(model) and not model.training and model.r28_ctx.binding() == side["demand"], "evaluation model")
    weights = tensor_digest(state.items())
    rng = ck.capture_rng(device)
    K.PHASE["phase"] = "evaluate/inference"
    data, preds, timing = infer(c, model, counts)
    K.require(counts["forwards"] == K.EVAL_WARMUP + K.EVAL_BATCHES == 141, "evaluation forward count")
    K.require(tensor_digest(model.state_dict().items()) == weights and ck.exact_equal(ck.capture_rng(device), rng)
              and model.r28_ctx.binding() == side["demand"], "evaluation changed model/RNG/demand source")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    K.PHASE["phase"] = "evaluate/predictions"
    ids = c["development_ids"]
    refs = {}
    for b in K.BUDGETS:
        p = out / f"top{b}-predictions.json"
        K.write_new(p, preds[b])
        refs[str(b)] = K.ref(p)
    grouped = {b: _group(preds[b], ids, b, whole=True, exact=True) for b in K.BUDGETS}
    _prefix(grouped[600], grouped[300], ids, whole=True)
    rows = data.records
    parsed = [data.raw(row) for row in rows]
    values, metrics_refs, counts_gt, scoring, per_class = {}, {}, {}, [], None
    K.PHASE["phase"] = "evaluate/scoring"
    epoch = side["global_update"] // K.WINDOWS_PER_EPOCH
    for b in K.BUDGETS:
        t = time.perf_counter()
        metrics = score_budget(rows, parsed, preds[b], c["source_root"])
        counts.update(score_budget_calls=1)
        metrics["R28_binding"] = {"contract_sha256": c["contract_sha256"], "arm": arm, "kind": "EMA", "view": "A", "epoch": epoch,
                                  "budget": b, "checkpoint": side["file"], "weights_sha256": weights, "demand": side["demand"],
                                  "predictions": refs[str(b)]}
        values[str(b)] = bridge_values(metrics)
        counts_gt[str(b)] = {k: metrics["spatial_filter_counts"][k] for k in ("GT_before", "GT_after")}
        p = out / f"{arm}-EMA-A-top{b}-metrics.json"
        K.write_new(p, metrics)
        metrics_refs[str(b)] = K.ref(p)
        scoring.append({"budget": b, "seconds": time.perf_counter() - t})
        if b == 600:
            per_class = per_class_large(rows, parsed, preds[b], metrics["COCO_bridge"])
            counts.update(per_class_COCOeval_calls=1)
    K.require(counts["score_budget_calls"] == 2 and counts["per_class_COCOeval_calls"] == 1, "scorer call counts")
    K.require(counts_gt == c["reference"]["GT_counts"], f"scored GT set differs from the L0 evaluation: {counts_gt}")
    return {"endpoint": side["file"], "formal_stage": formal, "weights_sha256": weights, "demand": side["demand"], "kind": "EMA",
            "epoch": epoch, "global_update": side["global_update"], "view": "A", "values": values, "metrics": metrics_refs,
            "predictions": refs, "GT_counts": counts_gt, "per_class_APL_top600": per_class, "scoring": scoring, **timing}


# ================================================================ summary arithmetic (pure)
def difference_pp(x, y):
    return {b: {m: 100. * (x[b][m] - y[b][m]) for m in METRICS} for b in ("600", "300")}


def apl6(mine, theirs):
    """Macro mean over the six registered classes of the per-class large-AP difference, in pp."""
    a = {r["category_id"]: r for r in mine}
    b = {r["category_id"]: r for r in theirs}
    K.require(sorted(a) == sorted(b) == list(range(1, 11)) and all(a[k]["GT"] == b[k]["GT"] for k in a), "per-class APL tables")
    rows = [{"category_id": k, "large_GT": a[k]["GT"], "minuend": a[k]["APL"], "subtrahend": b[k]["APL"],
             "difference_pp": 100. * (a[k]["APL"] - b[k]["APL"])} for k in K.APL6_CLASSES]
    K.require(all(type(r["minuend"]) is float and type(r["subtrahend"]) is float for r in rows), "APL6 class without a large AP")
    return {"classes": rows, "difference_pp": math.fsum(r["difference_pp"] for r in rows) / len(rows),
            "large_GT_covered": sum(r["large_GT"] for r in rows), "large_GT_total": sum(a[k]["GT"] for k in a)}


def per_class_table(tables):
    arms = list(tables)
    return [{"category_id": k, "large_GT": tables[arms[0]][k - 1]["GT"], **{a: tables[a][k - 1]["APL"] for a in arms}}
            for k in range(1, 11)]


def decide_m(m_cells, apl6_pp, c_ratio):
    failed = [k for k, v in m_cells.items() if not v["passed"]]
    c_passed = c_ratio <= K.COST_LIMIT
    if not failed and c_passed:
        return {"label_key": "M_passed", "K_trigger": True, "exploratory": False}
    if failed == ["top600_APL"] and c_passed and apl6_pp >= K.APL6_MINIMUM:
        return {"label_key": "M_failed_APL", "K_trigger": True, "exploratory": True}
    return {"label_key": "closed", "K_trigger": False, "exploratory": False}


def summary_w(c, ev, cost):
    ref = c["reference"]
    w, l0 = ev["values"], ref["values"]["L0"]
    m = K.cells(w, l0, [tuple(x) for x in c["gates"]["M_W_minus_L0"]])
    K.require(len(m) == 7, "M must have seven cells")
    a6 = apl6(ev["per_class_APL_top600"], ref["per_class_APL_top600"]["L0"])
    ratio = cost["ratios"]["model_gpu"]["W/L0"]
    d = decide_m(m, a6["difference_pp"], ratio)
    s15_gain = 100. * (ref["values"]["S15"]["600"]["APS"] - l0["600"]["APS"])
    return {"arm": "W", "values": {**ref["values"], "W": w},
            "value_sources": {**{a: "reused certified reading" for a in ref["values"]}, "W": "R28 eval-W (new)"},
            "M_W_minus_L0": {"cells": m, "failed_cells": [k for k, v in m.items() if not v["passed"]],
                             "passed": all(v["passed"] for v in m.values())},
            "APL6": {**a6, "minimum_pp": K.APL6_MINIMUM, "passed": a6["difference_pp"] >= K.APL6_MINIMUM},
            "C": {"model_gpu_W_over_L0": ratio, "maximum": K.COST_LIMIT, "passed": ratio <= K.COST_LIMIT,
                  "end_to_end_W_over_L0": cost["ratios"]["end_to_end"]["W/L0"], "source": "cost/CLOSED.json (fixed endpoints)"},
            "decision": {**d, "label": c["labels"][d["label_key"]]},
            "differences_pp": {f"W-{a}": difference_pp(w, ref["values"][a]) for a in ("L0", "A0", "S15", "P")},
            "S15_small_gain_retained": {"S15_minus_L0_APS600_pp": s15_gain, "W_minus_L0_APS600_pp": m["top600_APS"]["difference_pp"],
                                        "share": m["top600_APS"]["difference_pp"] / s15_gain, "note": "description, not a gate"},
            "per_class_APL_top600": per_class_table({**{a: ref["per_class_APL_top600"][a] for a in ("A0", "S15", "L0")},
                                                     "W": ev["per_class_APL_top600"]})}


def summary_k(c, w_result, ev_w, ev_k, cost_k):
    ref = c["reference"]
    w, k, l0 = ev_w["values"], ev_k["values"], ref["values"]["L0"]
    b, m, minimum = c["gates"]["A_W_minus_K"]
    a = 100. * (w[b][m] - k[b][m])
    passed = a >= minimum
    return {"arm": "K", "values": {**w_result["values"], "K": k},
            "A_W_minus_K": {"cell": f"top{b}_{m}", "difference_pp": a, "minimum_pp": minimum, "passed": passed,
                            "label_key": "A_passed" if passed else "A_failed",
                            "label": c["labels"]["A_passed" if passed else "A_failed"]},
            "differences_pp": {"W-K": difference_pp(w, k), "K-L0": difference_pp(k, l0)},
            "K_minus_L0_cells": K.cells(k, l0, [tuple(x) for x in c["gates"]["M_W_minus_L0"]]),
            "APL6_K_minus_L0": apl6(ev_k["per_class_APL_top600"], ref["per_class_APL_top600"]["L0"]),
            "APL6_W_minus_K": apl6(ev_w["per_class_APL_top600"], ev_k["per_class_APL_top600"]),
            "cost_K": {"model_gpu": cost_k["ratios"]["model_gpu"], "end_to_end": cost_k["ratios"]["end_to_end"],
                       "source": "cost-K/CLOSED.json (L0, W, K in one process)"},
            "per_class_APL_top600": per_class_table({"L0": ref["per_class_APL_top600"]["L0"], "W": ev_w["per_class_APL_top600"],
                                                     "K": ev_k["per_class_APL_top600"]}),
            "M_W_minus_L0_decision_unchanged": w_result["decision"]}


def _stage_account(c, names):
    out = {}
    for name in names:
        closed = K.stage_closed(c, name)
        out[name] = {"CLOSED": K.ref(K.ROUND / name / "CLOSED.json"), "counts": closed["counts"],
                     "elapsed_seconds": closed["elapsed_seconds"], "process_cpu_seconds": closed.get("process_cpu_seconds")}
    return out


def _other_directories():
    """Every directory of the round that is not a PASS stage (failed attempts kept as evidence)."""
    rows = []
    for d in sorted(K.ROUND.iterdir()):
        if d.is_dir() and not (d / "CLOSED.json").is_file() and not d.name.startswith(("release-", "dev-round")):
            stop = K.read_json(d / "STOP.json") if (d / "STOP.json").is_file() else None
            rows.append({"directory": d.name, "STOP": None if stop is None else {k: stop.get(k) for k in
                                                                               ("phase", "action", "arm", "error", "counts", "elapsed_seconds")}})
    return rows


def summarize(c, out, counts):
    arm = K.ARM
    K.PHASE["phase"] = "summarize/read"
    out = round_stage(out, f"summary-{arm}")
    ev_w, cost = K.stage_closed(c, "eval-W", "W"), K.stage_closed(c, "cost")
    K.require(cost["weights_sha256"]["W"] == ev_w["weights_sha256"], "the cost stage timed another W model than the evaluated one")
    for b in K.BUDGETS:
        K.pinned(ev_w["metrics"][str(b)]), K.pinned(ev_w["predictions"][str(b)])
    stages = ["cpu", "demand", "prior", "g0", "zero", "accept-W", "evalpath", "precost", K.formal_dir(c, "W").name, "eval-W", "cost"]
    K.PHASE["phase"] = "summarize/compute"
    if arm == "W":
        res = summary_w(c, ev_w, cost)
    else:
        w_closed = K.stage_closed(c, "summary-W", "W")
        w_result = K.read_json(K.pinned(w_closed["result"]))
        K.require(w_result["decision"]["K_trigger"] is True or K.DEV, "arm K was not triggered")
        ev_k, cost_k = K.stage_closed(c, "eval-K", "K"), K.stage_closed(c, "cost-K")
        K.require(cost_k["weights_sha256"]["K"] == ev_k["weights_sha256"] and cost_k["weights_sha256"]["W"] == ev_w["weights_sha256"],
                  "the cost-K stage timed other models than the evaluated ones")
        res = summary_k(c, w_result, ev_w, ev_k, cost_k)
        stages += ["summary-W", "accept-K", K.formal_dir(c, "K").name, "eval-K", "cost-K"]
    g0 = K.stage_closed(c, "g0")
    res.update(status="PASS", contract_sha256=c["contract_sha256"], source_commit=c["source_commit"],
               G0={k: g0[k] for k in ("G0_passed", "G0_small_geomean_predicted", "G0_minimum", "set", "development",
                                      "train_core_predicted")},
               cost_medians=cost["medians"], cost_ratios=cost["ratios"], precost=K.stage_closed(c, "precost")["ratios"],
               stages=_stage_account(c, stages), other_directories=_other_directories(), statements=STATEMENTS)
    K.require(counts["forwards"] == 0 and counts["actual_updates"] == 0 and not counts.get("score_budget_calls"),
              "summary must not run a model or a scorer")
    K.write_new(out / "RESULT.json", res)
    if arm == "W":
        d = res["decision"]
        return {"result": K.ref(out / "RESULT.json"), "label": d["label"], "label_key": d["label_key"], "K_trigger": d["K_trigger"],
                "M_passed": res["M_W_minus_L0"]["passed"], "M_failed_cells": res["M_W_minus_L0"]["failed_cells"],
                "M_difference_pp": {k: v["difference_pp"] for k, v in res["M_W_minus_L0"]["cells"].items()},
                "APL6_pp": res["APL6"]["difference_pp"], "C_model_gpu_W_over_L0": res["C"]["model_gpu_W_over_L0"]}
    a = res["A_W_minus_K"]
    return {"result": K.ref(out / "RESULT.json"), "label": a["label"], "A_passed": a["passed"], "A_difference_pp": a["difference_pp"]}
