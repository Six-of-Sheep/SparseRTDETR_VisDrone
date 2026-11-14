"""R28 same-process timing (L0 task section 5 method: FP32, batch 4, monitor64, 2 warm-up batches per endpoint, three
passes of 16 batches in rotated order, median of the three pass totals, then the ratio).

precost(c, out, counts)  before the formal W run: L0 (its e120 EMA) against Wpre = the same L0 e120 EMA weights behind
                         the R28 forward with the final frozen demand network.  The detector weights are identical, so
                         the ratio measures the added demand network, allocation, resampling and inverse mapping.
cost(c, out, counts)     fixed endpoints: A0, L0, W, S15 (arm W), L0, W, K (arm K, <round>/cost-K) or L0, W, F (arm F,
                         <round>/cost-F).
The timed interval of every endpoint is the whole model call; for W / K / F it contains the demand source, the
allocation or map, the resampling and the inverse mapping.  No annotation is parsed while timing.

Stage readings (L0, W, K, F): CUDA events recorded at the stage boundaries inside the same timed model call split the
model GPU time into demand / map / render / detector / unwarp.  Arm F marks its own boundaries; for L0 and W / K the
boundaries are recorded by wrappers placed around to_low, Context.demand, Context.plan, r28_warp.resample and
r28_resample.unwarp_outputs for the duration of the timing and removed afterwards.  A wrapper calls the original
function with the original arguments and returns its result; it adds one event record per boundary.
"""
import gc
import statistics
import time
from collections import Counter
from pathlib import Path

import r28_common as K
import l0_cost as LC

IMAGES = K.COST_BATCH * K.COST_BATCHES_PER_PASS
CANVAS = {"A0": "L", "L0": "H", "S15": "H", "W": "H", "K": "H", "Wpre": "H", "F": "H"}
SUMS = LC.SUMS
STAGES = ("demand", "map", "render", "detector", "unwarp", "other")


class Probe:
    """CUDA events at the stage boundaries of one model call.  mark(name) closes the stage `name`."""

    def __init__(self, torch):
        self.torch, self.marks = torch, []

    def mark(self, name):
        ev = self.torch.cuda.Event(enable_timing=True)
        ev.record()
        self.marks.append((name, ev))

    def start(self):
        self.marks = []
        self.mark("start")

    def stages(self, total_ms):
        """Milliseconds per stage of the last call (read after the device has finished); `other` is the remainder of
        the timed model interval."""
        out = dict.fromkeys(STAGES, 0.0)
        for (_, a), (name, b) in zip(self.marks, self.marks[1:]):
            out[name] += a.elapsed_time(b)
        out["other"] = total_ms - sum(v for k, v in out.items() if k != "other")
        return out


def instrument(models, probe):
    """Place the stage boundaries; returns the function that removes every wrapper again."""
    from sparse_rtdetr.research import r27_l0_control as L0
    from sparse_rtdetr.research import r28_fovea as FV
    from sparse_rtdetr.research import r28_resample as R
    from sparse_rtdetr.research import r28_warp as W
    undo = []

    def after(owner, name, stage, before=None):
        original = getattr(owner, name)

        def wrapper(*a, **k):
            if before is not None:
                probe.mark(before)
            res = original(*a, **k)
            probe.mark(stage)
            return res

        had = name in vars(owner)
        setattr(owner, name, wrapper)
        undo.append((owner, name, original, had))

    kinds = {}
    for arm, m in models.items():
        if FV.is_wrapped(m):
            vars(m.r28_ctx)["probe"] = probe.mark
            undo.append((m.r28_ctx, "probe", None, False))
            kinds[arm] = "own marks of r28_fovea._forward"
        elif R.is_wrapped(m):
            after(m.r28_ctx, "demand", "demand")
            after(m.r28_ctx, "plan", "map")
            kinds[arm] = "wrappers: Context.demand, Context.plan, r28_warp.resample, r28_resample.unwarp_outputs"
        elif L0.is_wrapped(m):
            kinds[arm] = "wrapper: r27_l0_control.to_low (render); detector = the rest of the call"
        else:
            kinds[arm] = None
    if any(R.is_wrapped(m) for m in models.values()):
        after(W, "resample", "render")
        after(R, "unwarp_outputs", "unwarp", before="detector")
    if any(L0.is_wrapped(m) for m in models.values()):
        after(L0, "to_low", "render")

    def remove():
        for owner, name, original, had in reversed(undo):
            if had:
                setattr(owner, name, original)
            elif isinstance(owner, type(W)):
                setattr(owner, name, original)
            else:
                vars(owner).pop(name, None)
        K.require(W.resample.__module__ == W.__name__ and R.unwarp_outputs.__module__ == R.__name__
                  and L0.to_low.__module__ != __name__, "a timing wrapper was left in place")

    return kinds, remove


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
    R = K.arm_module(arm)
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


def stage_table(passes, endpoints, images):
    """Per endpoint: the median over the passes of each stage's pass total (ms), its share of the model GPU time of
    the same pass set, and the per-image value.  Endpoints without stage readings are left out."""
    out = {}
    for a in endpoints:
        ps = [p for p in passes if p["endpoint"] == a and p.get("stage_ms_total") is not None]
        if not ps:
            continue
        med = {s: statistics.median(p["stage_ms_total"][s] for p in ps) for s in STAGES}
        total = statistics.median(p["model_gpu_ms_total"] for p in ps)
        out[a] = {"stage_ms_median_pass": med, "per_image_ms": {s: v / images for s, v in med.items()},
                  "share_of_model_gpu": {s: v / total for s, v in med.items()},
                  "stage_ms_per_pass": [p["stage_ms_total"] for p in ps], "model_gpu_ms_median_pass": total}
    return out


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
                for i in range(len(rotations))]
    return med, total, per_pass


def timing(c, counts, models, endpoints, rotations, *, warmup=K.COST_WARMUP, batches_per_pass=K.COST_BATCHES_PER_PASS, stages=False):
    """warmup / batches_per_pass / the number of rotations are the fixed protocol unless a wiring check shortens them
    (its readings are not a cost result).  stages=True adds the stage readings to every timed call."""
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
    batches = [list(range(k, k + K.COST_BATCH)) for k in range(0, IMAGES, K.COST_BATCH)][:batches_per_pass]
    K.require(len(rows) == IMAGES and len(batches) == batches_per_pass, "cost monitor64 batches")
    K.require(all(sorted(r) == sorted(endpoints) for r in rotations), "cost rotations")
    fixed = (warmup, batches_per_pass, len(rotations)) == (K.COST_WARMUP, K.COST_BATCHES_PER_PASS, K.COST_PASSES)
    per_endpoint = warmup + len(rotations) * batches_per_pass
    K.require(not fixed or per_endpoint == K.COST_PER_ENDPOINT, "fixed cost protocol count")
    done = Counter()
    probe = Probe(torch) if stages else None
    kinds, remove = instrument(models, probe) if stages else ({}, lambda: None)

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
        if probe is not None:
            probe.start()
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            o = m(xg)
        g1.record()
        if probe is not None:
            probe.mark("detector" if (kinds.get(arm) or "").startswith("wrapper: r27_l0") else "other")
        counts.update(forwards=1)
        done.update([arm])
        logits, boxes = o["pred_logits"].float().cpu(), o["pred_boxes"].float().cpu()
        t2 = time.perf_counter()
        ranked = [rank_candidates(logits[j], boxes[j], rows[i], 600) for j, i in enumerate(idx)]
        t3 = time.perf_counter()
        K.require(len(ranked) == len(idx) and tuple(x.shape) == (K.COST_BATCH, 3, *hw[cv]) and x.dtype == torch.float32,
                  f"cost timing batch: {arm}")
        rec = {"load_decode_resize_cpu_s": t1 - t0, "H2D_ms": h0.elapsed_time(h1), "model_gpu_ms": g0.elapsed_time(g1),
               "D2H_after_model_s": t2 - t1, "rank_top600_cpu_s": t3 - t2, "end_to_end_s": t3 - t0}
        if probe is not None and kinds.get(arm):
            rec["stage_ms"] = probe.stages(rec["model_gpu_ms"])
        return rec

    def refused(*_a, **_k):
        raise RuntimeError("ground truth parse during cost timing")

    parse_raw, LD.parse_raw = LD.parse_raw, refused
    try:
        warm = {a: [run(a, batches[i]) for i in range(warmup)] for a in endpoints}
        passes = []
        for p, order in enumerate(rotations):
            for arm in order:
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                recs = [run(arm, b) for b in batches]
                passes.append({"pass": p, "endpoint": arm, "canvas": CANVAS[arm], "batches": recs, **LC.pass_totals(recs),
                               "stage_ms_total": ({s: sum(r["stage_ms"][s] for r in recs) for s in STAGES}
                                                  if all("stage_ms" in r for r in recs) else None),
                               "peak_allocated": torch.cuda.max_memory_allocated()})
    finally:
        LD.parse_raw = parse_raw
        remove()
    med, ratios, per_pass = aggregate(passes, endpoints, rotations)
    images = K.COST_BATCH * batches_per_pass
    for a in endpoints:
        med[a]["parameters"] = sum(p.numel() for p in models[a].parameters())
        ctx = getattr(models[a], "r28_ctx", None)
        med[a]["demand_network_parameters"] = 0 if ctx is None or ctx.net is None else sum(p.numel() for p in ctx.net.parameters())
    K.require(dict(done) == {a: per_endpoint for a in endpoints}, f"cost forwards {dict(done)}")
    res = {"warmup": warm, "passes": passes, "medians": med, "ratios": ratios, "ratios_per_pass": per_pass,
           "order_executed": [list(r) for r in rotations], "forwards_per_endpoint": dict(done),
           "annotation_parse_during_timing": "refused (LD.parse_raw replaced); none occurred"}
    if stages:
        res["stages"] = {"names": list(STAGES), "boundaries": kinds, "table": stage_table(passes, endpoints, images),
                         "note": "CUDA event intervals inside the timed model call; `other` is the remainder of the model GPU "
                                 "time (for arm W the detector interval also contains nothing else; for L0 the detector is the "
                                 "call after to_low); demand of W / F includes the 384x672 reduction and the device-to-host copy "
                                 "of the 12x21 map; map is the CPU FP64 allocation (W) or node computation (F) and its transfer"}
    return res


def _run(c, counts, loaders, endpoints, rotations, **protocol):
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
    res = timing(c, counts, models, endpoints, rotations, **protocol)
    K.require(before == {a: _digest(m) for a, m in models.items()}, "cost timing changed a model state")
    K.require(counts["forwards"] == sum(res["forwards_per_endpoint"].values()), f"cost forward count {dict(counts)}")
    return {"sources": sources, "weights_sha256": before, "states_unchanged_by_timing": True, **res,
            "protocol": {"batch": K.COST_BATCH, "warmup_per_endpoint": protocol.get("warmup", K.COST_WARMUP), "passes": len(rotations),
                         "batches_per_pass": protocol.get("batches_per_pass", K.COST_BATCHES_PER_PASS),
                         "images": K.COST_BATCH * protocol.get("batches_per_pass", K.COST_BATCHES_PER_PASS),
                         "precision": "float32, autocast off",
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
    name = {"W": "cost", "K": "cost-K", "F": "cost-F"}[arm]
    K.require(Path(out).resolve() == (K.ROUND / name).resolve(), f"this action writes <round>/{name}")
    torch = K.torch_mod()
    endpoints = K.COST_ENDPOINTS[name]
    loaders = {a: (lambda a=a: load_arm(c, a, torch) if a in K.ARMS else load_reference(c, a, torch)) for a in endpoints}
    res = _run(c, counts, loaders, endpoints, K.COST_ROTATIONS[name], **({"stages": True} if arm == "F" else {}))
    K.require(counts["forwards"] == len(endpoints) * K.COST_PER_ENDPOINT, f"cost forwards {dict(counts)}")
    return {"label": f"fixed-endpoint joint timing ({', '.join(endpoints)})", **res}
