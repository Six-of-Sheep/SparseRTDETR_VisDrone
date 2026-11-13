"""R28 same-process timing (L0 task section 5 method: FP32, batch 4, monitor64, 2 warm-up batches per endpoint, three
passes of 16 batches in rotated order, median of the three pass totals, then the ratio).

precost(c, out, counts)  before the formal W run: L0 (its e120 EMA) against Wpre = the same L0 e120 EMA weights behind
                         the R28 forward with the final frozen demand network.  The detector weights are identical, so
                         the ratio measures the added demand network, allocation, resampling and inverse mapping.
cost(c, out, counts)     fixed endpoints: A0, L0, W, S15 (arm W) or L0, W, K (arm K, written to <round>/cost-K).
The timed interval of every endpoint is the whole model call; for W / K it contains the demand source, the
allocation, the resampling and the inverse mapping.  No annotation is parsed while timing.
"""
import gc
import statistics
import time
from collections import Counter
from pathlib import Path

import r28_common as K
import l0_cost as LC

IMAGES = K.COST_BATCH * K.COST_BATCHES_PER_PASS
CANVAS = {"A0": "L", "L0": "H", "S15": "H", "W": "H", "K": "H", "Wpre": "H"}
SUMS = LC.SUMS


def _digest(model):
    from sparse_rtdetr.protocol_r18.engine import tensor_digest
    return tensor_digest(model.state_dict().items())


def load_reference(c, arm, torch):
    """A0 / L0 / S15 exactly as the L0 cost stage loaded them, under the verified L0 contract."""
    c0 = K.l0_contract(c)
    model, source = LC.LOADERS[arm](c0, torch)
    if arm == "L0":
        K.require(_digest(model) == c["L0_weights_sha256"], "L0 e120 EMA differs from the contract")
    return model, source


def load_arm(c, arm, torch):
    from sparse_rtdetr.research import r28_resample as R
    closed = K.stage_closed(c, f"eval-{arm}", arm)
    side, state = K.endpoint_state(c, arm)
    model = R.build_eval_model(state, "cuda:0", arm, K.demand_source(c, arm))
    K.require(closed["weights_sha256"] == _digest(model) and model.r28_ctx.binding() == side["demand"],
              f"cost model of arm {arm} differs from the evaluated endpoint")
    return model, {"endpoint": side["file"], "weights": f"{arm} e120 EMA", "demand": side["demand"],
                   "input": "H 1152x2016; demand source, allocation, resampling and inverse mapping inside the model"}


def load_wpre(c, torch):
    from sparse_rtdetr.research import r28_resample as R
    import l0_common as B
    c0 = K.l0_contract(c)
    side, state = B.endpoint_state(c0)
    K.require(side["file"] == c["L0_endpoint"], "L0 endpoint file")
    model = R.build_eval_model(state, "cuda:0", "W", K.demand_source(c, "W"))
    K.require(_digest(model) == c["L0_weights_sha256"], "Wpre detector weights are not the L0 e120 EMA")
    return model, {"endpoint": side["file"], "weights": "L0 e120 EMA (detector not adapted to the warp; timing only)",
                   "demand": model.r28_ctx.binding(),
                   "input": "H 1152x2016; final demand network, allocation, resampling and inverse mapping inside the model"}


def aggregate(passes, endpoints, rotations):
    order = [(p, a) for p, rotation in enumerate(rotations) for a in rotation]
    K.require([(p["pass"], p["endpoint"]) for p in passes] == order, "cost pass order differs from the fixed rotations")
    by = {a: [p for p in passes if p["endpoint"] == a] for a in endpoints}
    med = {}
    for a, ps in by.items():
        m = {f"{name}_per_pass": [p[f"{name}_total"] for p in ps] for name in ("model_gpu_ms", "end_to_end_s")}
        m.update({f"{name}_median_pass": statistics.median(p[f"{name}_total"] for p in ps) for name in SUMS})
        m["per_image_model_gpu_ms"] = m["model_gpu_ms_median_pass"] / IMAGES
        m["peak_allocated"] = max(p["peak_allocated"] for p in ps)
        med[a] = m
    pairs = [(a, b) for a in endpoints for b in endpoints if a != b]

    def ratios(ms, es):
        return {"model_gpu": {f"{a}/{b}": ms[a] / ms[b] for a, b in pairs}, "end_to_end": {f"{a}/{b}": es[a] / es[b] for a, b in pairs}}

    total = ratios({a: med[a]["model_gpu_ms_median_pass"] for a in by}, {a: med[a]["end_to_end_s_median_pass"] for a in by})
    per_pass = [ratios({a: by[a][i]["model_gpu_ms_total"] for a in by}, {a: by[a][i]["end_to_end_s_total"] for a in by})
                for i in range(K.COST_PASSES)]
    return med, total, per_pass


def timing(c, counts, models, endpoints, rotations):
    from sparse_rtdetr.research import scale_baseline as S
    from sparse_rtdetr.research import shared_query_cls as Q
    import sparse_rtdetr.localization.data as LD
    from sparse_rtdetr.r0.evaluation import rank_candidates
    import sqc_run as SQ
    torch = K.torch_mod()
    dev = torch.device("cuda:0")
    data = S.train_dataset(c["data"]["train_core"], hw=S.S15_HW, seed=0)
    hw = {"L": Q.L_HW, "H": Q.H_HW}
    rs = {k: {k: LD.RawAugmentation(v).resize} for k, v in hw.items()}
    rows = [data.by_id[i] for i in c["monitor_ids"]]
    batches = [list(range(k, k + K.COST_BATCH)) for k in range(0, IMAGES, K.COST_BATCH)]
    K.require(len(rows) == IMAGES and len(batches) == K.COST_BATCHES_PER_PASS, "cost monitor64 batches")
    K.require(all(sorted(r) == sorted(endpoints) for r in rotations) and len(rotations) == K.COST_PASSES, "cost rotations")
    done = Counter()

    def run(arm, idx):
        m, cv = models[arm], CANVAS[arm]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        x = torch.stack([SQ.load_pixels(data, rows[i], rs[cv])[cv] for i in idx])
        t1 = time.perf_counter()
        h0 = torch.cuda.Event(enable_timing=True); h1 = torch.cuda.Event(enable_timing=True)
        g0 = torch.cuda.Event(enable_timing=True); g1 = torch.cuda.Event(enable_timing=True)
        h0.record()
        xg = x.to(dev)
        h1.record()
        g0.record()
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            o = m(xg)
        g1.record()
        counts.update(forwards=1)
        done.update([arm])
        logits, boxes = o["pred_logits"].float().cpu(), o["pred_boxes"].float().cpu()
        t2 = time.perf_counter()
        ranked = [rank_candidates(logits[j], boxes[j], rows[i], 600) for j, i in enumerate(idx)]
        t3 = time.perf_counter()
        K.require(len(ranked) == len(idx) and tuple(x.shape) == (K.COST_BATCH, 3, *hw[cv]) and x.dtype == torch.float32,
                  f"cost timing batch: {arm}")
        return {"load_decode_resize_cpu_s": t1 - t0, "H2D_ms": h0.elapsed_time(h1), "model_gpu_ms": g0.elapsed_time(g1),
                "D2H_after_model_s": t2 - t1, "rank_top600_cpu_s": t3 - t2, "end_to_end_s": t3 - t0}

    def refused(*_a, **_k):
        raise RuntimeError("ground truth parse during cost timing")

    parse_raw, LD.parse_raw = LD.parse_raw, refused
    try:
        warmup = {a: [run(a, batches[i]) for i in range(K.COST_WARMUP)] for a in endpoints}
        passes = []
        for p, order in enumerate(rotations):
            for arm in order:
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                recs = [run(arm, b) for b in batches]
                passes.append({"pass": p, "endpoint": arm, "canvas": CANVAS[arm], "batches": recs, **LC.pass_totals(recs),
                               "peak_allocated": torch.cuda.max_memory_allocated()})
    finally:
        LD.parse_raw = parse_raw
    med, ratios, per_pass = aggregate(passes, endpoints, rotations)
    for a in endpoints:
        med[a]["parameters"] = sum(p.numel() for p in models[a].parameters())
        ctx = getattr(models[a], "r28_ctx", None)
        med[a]["demand_network_parameters"] = 0 if ctx is None or ctx.net is None else sum(p.numel() for p in ctx.net.parameters())
    K.require(dict(done) == {a: K.COST_PER_ENDPOINT for a in endpoints}
              and counts["forwards"] == len(endpoints) * K.COST_PER_ENDPOINT, f"cost forwards {dict(done)}")
    return {"warmup": warmup, "passes": passes, "medians": med, "ratios": ratios, "ratios_per_pass": per_pass,
            "order_executed": [list(r) for r in rotations], "forwards_per_endpoint": dict(done),
            "annotation_parse_during_timing": "refused (LD.parse_raw replaced); none occurred"}


def _run(c, counts, loaders, endpoints, rotations):
    torch = K.torch_mod()
    K.PHASE["phase"] = "cost:load"
    models, sources = {}, {}
    for arm in endpoints:
        models[arm], sources[arm] = loaders[arm]()
        K.require(not models[arm].training, f"cost endpoint {arm} is not in eval mode")
        gc.collect()
    torch.cuda.empty_cache()
    before = {a: _digest(m) for a, m in models.items()}
    K.PHASE["phase"] = "cost:timing"
    res = timing(c, counts, models, endpoints, rotations)
    K.require(before == {a: _digest(m) for a, m in models.items()}, "cost timing changed a model state")
    return {"sources": sources, "weights_sha256": before, "states_unchanged_by_timing": True, **res,
            "protocol": {"batch": K.COST_BATCH, "warmup_per_endpoint": K.COST_WARMUP, "passes": K.COST_PASSES,
                         "batches_per_pass": K.COST_BATCHES_PER_PASS, "images": IMAGES, "precision": "float32, autocast off",
                         "statistic": "median over the three pass totals per endpoint, then the ratio",
                         "device": torch.cuda.get_device_name(0), "torch": str(torch.__version__)}}


def precost(c, out, counts):
    K.require(Path(out).resolve() == (K.ROUND / "precost").resolve(), "this action writes <round>/precost")
    K.g0_passed(c)
    torch = K.torch_mod()
    loaders = {"L0": lambda: load_reference(c, "L0", torch), "Wpre": lambda: load_wpre(c, torch)}
    res = _run(c, counts, loaders, K.PRECOST_ENDPOINTS, K.PRECOST_ROTATIONS)
    ratio = res["ratios"]["model_gpu"]["Wpre/L0"]
    return {"label": "pre-launch cost check: identical detector weights, L0 forward against the complete W forward",
            "C_pre_model_gpu_W_over_L0": ratio, "C_pre_end_to_end_W_over_L0": res["ratios"]["end_to_end"]["Wpre/L0"],
            "maximum": K.COST_LIMIT, "C_pre_passed": bool(ratio <= K.COST_LIMIT), **res}


def cost(c, out, counts):
    arm = K.ARM
    name = "cost" if arm == "W" else "cost-K"
    K.require(Path(out).resolve() == (K.ROUND / name).resolve(), f"this action writes <round>/{name}")
    torch = K.torch_mod()
    endpoints = K.COST_ENDPOINTS[name]
    loaders = {a: (lambda a=a: load_arm(c, a, torch) if a in K.ARMS else load_reference(c, a, torch)) for a in endpoints}
    res = _run(c, counts, loaders, endpoints, K.COST_ROTATIONS[name])
    return {"label": f"fixed-endpoint joint timing ({', '.join(endpoints)})", **res}
