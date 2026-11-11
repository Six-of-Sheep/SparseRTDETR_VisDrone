"""R27-002 L0 control: one joint short timing of four deployed endpoints in one process (task section 5).

A0 (R25 A0 e120 EMA, original image resized directly to L), L0 (this round's e120 EMA, H input, to_low inside the model),
P (R27 P e120 EMA, H input, local branch), S15 (S15 e120 EMA, H input).  Same GPU, FP32, autocast off, batch 4, the
fixed monitor64, the timing primitives of R27 `timing.run`.  No stage instrumentation, no MACs, no hook on the timed path.
"""
import gc
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import l0_common as K  # noqa: E402

PAIRS = (("L0", "A0"), ("P", "A0"), ("S15", "A0"), ("P", "L0"), ("P", "S15"), ("L0", "S15"))
IMAGES = K.COST_BATCH * K.COST_BATCHES_PER_PASS  # the fixed 64 monitor images, once per pass
CANVAS = {"A0": "L", "L0": "H", "P": "H", "S15": "H"}  # what the loader renders; L0 and P go H->L inside the model
ORDER = tuple((p, a) for p, rotation in enumerate(K.ROTATIONS) for a in rotation)
SUMS = {"model_gpu_ms": "model_gpu_ms", "end_to_end_s": "end_to_end_s", "load_decode_resize_s": "load_decode_resize_cpu_s",
        "H2D_ms": "H2D_ms", "rank_s": "rank_top600_cpu_s"}  # pass total name -> per-batch record key
# EMA module digests of the two old native endpoints as certified by s15_cross_scale.ENDPOINTS (ema_weights_sha256)
CERTIFIED_EMA = {"A0": "a322373726cc3c666c8ba0ea79a5dadcc80c149d87f8c4094e9c9c807652d676",
                 "S15": "072616ff442dec05693f8f4f98974213f59a35b72edfbf4a1fb87bb7fd49ac68"}
# R25 formal-001 contract digest and formal identity of the A0 endpoint (s15_cross_scale.ENDPOINTS["A0"])
A0_R25_FORMAL = ("178eeb821a2049e12c0ba0c21ebd2f4c628715f8c1358218aa60c9b240e32ece",
                 "e9b6ff34e9ec1e42a75fd6cb46b4e2faad938512dcd4327049faee3e149bbe8a")
NOTE = ("本测速是新环境下的补测，不覆盖原 R27 的终态成本与 E 判据；不从 P 的内部耗时推算 A0。"
        "四个端点各自按真实部署的取图与缩放路径计入端到端。")


# ================================================================ pure aggregation (per-pass totals -> median -> ratio)
def pass_totals(batches):
    return {f"{name}_total": sum(b[key] for b in batches) for name, key in SUMS.items()}


def ratios_of(model_ms, end_s):
    return {"model_gpu": {f"{a}/{b}": model_ms[a] / model_ms[b] for a, b in PAIRS},
            "end_to_end": {f"{a}/{b}": end_s[a] / end_s[b] for a, b in PAIRS}}


def aggregate(passes):
    """passes: executed pass records in execution order -> (medians, ratios of the medians, per-pass ratios)."""
    K.require([(p["pass"], p["endpoint"]) for p in passes] == list(ORDER), "cost4 pass order differs from the fixed rotations")
    by = {a: [p for p in passes if p["endpoint"] == a] for a in K.COST_ENDPOINTS}
    med = {}
    for a, ps in by.items():
        m = {f"{name}_per_pass": [p[f"{name}_total"] for p in ps] for name in ("model_gpu_ms", "end_to_end_s")}
        m.update({f"{name}_median_pass": statistics.median(p[f"{name}_total"] for p in ps) for name in SUMS})
        m["per_image_model_gpu_ms"] = m["model_gpu_ms_median_pass"] / IMAGES
        m["peak_allocated"] = max(p["peak_allocated"] for p in ps)
        med[a] = m
    ratios = ratios_of({a: med[a]["model_gpu_ms_median_pass"] for a in by}, {a: med[a]["end_to_end_s_median_pass"] for a in by})
    per_pass = [ratios_of({a: by[a][i]["model_gpu_ms_total"] for a in by}, {a: by[a][i]["end_to_end_s_total"] for a in by})
                for i in range(K.COST_PASSES)]
    return med, ratios, per_pass


# ================================================================ endpoints (each strictly by its own identity)
def _digest(model):
    from sparse_rtdetr.protocol_r18.engine import tensor_digest
    return tensor_digest(model.state_dict().items())


def load_a0(c, torch):
    """R25 A0 seed0 e120 EMA, selected and loaded by the R25 evaluator's own `_load_model` (native key set, strict)."""
    import formal_eval as E25
    from sparse_rtdetr.protocol_r18 import checkpoint as ck
    from sparse_rtdetr.protocol_r18.geometry import validate_geometry
    path = K.pinned(c["A0_endpoint"])
    side = K.read_json(K.side_of(path))
    K.require((side["status"], side["schema"], side["scope"], side["arm"], side["global_update"], side["parent_sha256"]) ==
              ("COMPLETE", "r25-complete-state-v1", "formal", "A0", K.ENDPOINT, c["parent"]["sha256"])
              and side["file"] == c["A0_endpoint"] and side["cursor"]["completed_epochs"] == 120
              and (side["contract_sha256"], side["formal_identity"]) == A0_R25_FORMAL, "A0 endpoint side-car")
    s = torch.load(path, map_location="cpu", weights_only=True)
    binding = {k: v for k, v in side.items() if k not in ("status", "file")}
    K.require(s.pop("formal_binding_sha256") == K.sha_bytes(ck.canonical(binding)), "A0 endpoint binding digest")
    model, digest = E25._load_model(s, "A0", "EMA", "cuda:0")  # version/arm/parent/config, clocks 36480, FP32, strict
    validate_geometry(model)
    K.require(digest == _digest(model) == CERTIFIED_EMA["A0"] and not hasattr(model, "local_head"), "A0 EMA weights")
    return model, {"endpoint": c["A0_endpoint"], "weights": "R25 A0 seed0 e120 EMA", "loader": "formal_eval._load_model",
                   "input": "original image resized directly to L 768x1344; native model at L"}


def load_l0(c, torch):
    from sparse_rtdetr.research import r27_l0_control as L
    closed = K.stage_closed(c, "eval-L0")
    side, state = K.endpoint_state(c)
    model = L.build_eval_model(state, "cuda:0")
    K.require(closed["weights_sha256"] == _digest(model), "cost4 L0 model differs from the evaluated e120 EMA (eval-L0)")
    return model, {"endpoint": side["file"], "weights": "L0 e120 EMA", "loader": "r27_l0_control.build_eval_model",
                   "equal_to_eval_L0_weights_sha256": True, "input": "H 1152x2016; FP32 to_low inside the model"}


def load_p(c, torch):
    """R27 P e120 EMA under the original R27 contract (r27_run.read_state: side-car, file identity, binding digest)."""
    sys.path.insert(0, str(K.ROOT / "tools" / "r27"))
    import r27_run as R27
    from sparse_rtdetr.research import r27_local_memory as R
    r = c["R27_contract"]
    old = K.read_json(K.pinned({k: r[k] for k in ("path", "bytes", "sha256")}))
    body = {k: v for k, v in old.items() if k != "contract_sha256"}
    K.require(old["contract_sha256"] == K.sha_bytes(K.canonical(body)) == r["contract_sha256"]
              and old["identity_sha256"] == r["identity_sha256"] == R.IDENTITY_SHA256, "original R27 contract / identity")
    path = K.pinned(c["P_endpoint"])
    K.require(K.pinned(c["P_endpoint_side"]) == K.side_of(path), "P endpoint side-car path")
    side, s = R27.read_state(old, path, "formal", "P")
    K.require(side["global_update"] == s["engine"]["completed_updates"] == s["ema"]["updates"] == K.ENDPOINT
              and side["file"] == c["P_endpoint"], "P e120 clocks / file")
    model = R.build_eval_model("P", s["ema"]["module"], "cuda:0")
    evaluated = K.read_json(K.pinned(c["R27_eval_closed"]["P"]))
    K.require(evaluated["status"] == "PASS" and evaluated["weights_sha256"] == _digest(model), "P weights differ from R27 eval-P")
    return model, {"endpoint": c["P_endpoint"], "side": c["P_endpoint_side"], "weights": "R27 P e120 EMA",
                   "loader": "r27_run.read_state + r27_local_memory.build_eval_model", "equal_to_R27_eval_P_weights_sha256": True,
                   "input": "H 1152x2016; H->L and local branch inside the model; route keys fed per batch"}


def load_s15(c, torch):
    """S15 seed0 e120 EMA as R27 `final_cost` loads its N endpoint."""
    from sparse_rtdetr.research import scale_baseline as S
    from sparse_rtdetr.protocol_r18.geometry import validate_geometry
    from sparse_rtdetr.protocol_r18.model import build
    s = torch.load(K.pinned(c["S15_endpoint"]), map_location="cpu", weights_only=True)
    K.require(s["engine"]["completed_updates"] == s["ema"]["updates"] == K.ENDPOINT
              and s["config"] == S.production_config(0).binding(), "S15 e120 endpoint identity")
    model, _, _ = build(S.production_config(0), synthetic=True)
    model.load_state_dict(s["ema"]["module"], strict=True)
    validate_geometry(model)
    model = model.eval().requires_grad_(False).to("cuda:0")
    K.require(_digest(model) == CERTIFIED_EMA["S15"], "S15 EMA weights")
    return model, {"endpoint": c["S15_endpoint"], "weights": "S15 seed0 e120 EMA", "loader": "R27 final_cost recipe",
                   "input": "H 1152x2016; native model at H"}


LOADERS = {"A0": load_a0, "L0": load_l0, "P": load_p, "S15": load_s15}


# ================================================================ timing
def timing(c, counts, models):
    from sparse_rtdetr.research import scale_baseline as S
    from sparse_rtdetr.research import shared_query_cls as Q
    import sparse_rtdetr.localization.data as LD
    from sparse_rtdetr.r0.evaluation import rank_candidates
    import sqc_run as B
    torch = K.torch_mod()
    dev = torch.device("cuda:0")
    data = S.train_dataset(c["data"]["train_core"], hw=S.S15_HW, seed=0)
    hw = {"L": Q.L_HW, "H": Q.H_HW}
    rs = {k: {k: LD.RawAugmentation(v).resize} for k, v in hw.items()}  # one direct resize of the original image
    rows = [data.by_id[i] for i in c["monitor_ids"]]
    batches = [list(range(k, k + K.COST_BATCH)) for k in range(0, IMAGES, K.COST_BATCH)]
    K.require(len(rows) == IMAGES and len(batches) == K.COST_BATCHES_PER_PASS, "cost4 monitor64 batches")
    done = Counter()

    def run(arm, idx):
        m, cv = models[arm], CANVAS[arm]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        x = torch.stack([B.load_pixels(data, rows[i], rs[cv])[cv] for i in idx])
        t1 = time.perf_counter()
        h0 = torch.cuda.Event(enable_timing=True); h1 = torch.cuda.Event(enable_timing=True)
        g0 = torch.cuda.Event(enable_timing=True); g1 = torch.cuda.Event(enable_timing=True)
        h0.record()
        xg = x.to(dev)
        h1.record()
        if arm == "P":
            m.r27_ctx.route_keys.extend((rows[i]["stable_image_id"], "eval", rows[i]["relative_path"], 0, 0) for i in idx)
        g0.record()
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            o = m(xg)
        g1.record()
        counts.update(forwards=1)  # plain counters outside the model (no hook on the timed path)
        done.update([arm])
        logits, boxes = o["pred_logits"].float().cpu(), o["pred_boxes"].float().cpu()
        t2 = time.perf_counter()
        ranked = [rank_candidates(logits[j], boxes[j], rows[i], 600) for j, i in enumerate(idx)]
        t3 = time.perf_counter()
        K.require(len(ranked) == len(idx) and tuple(x.shape) == (K.COST_BATCH, 3, *hw[cv]) and x.dtype == torch.float32
                  and (arm != "P" or not m.r27_ctx.route_keys), f"cost4 timing batch: {arm}")
        return {"load_decode_resize_cpu_s": t1 - t0, "H2D_ms": h0.elapsed_time(h1), "model_gpu_ms": g0.elapsed_time(g1),
                "D2H_after_model_s": t2 - t1, "rank_top600_cpu_s": t3 - t2, "end_to_end_s": t3 - t0}

    def refused(*_a, **_k):
        raise RuntimeError("ground truth parse during cost4 timing")

    parse_raw, LD.parse_raw = LD.parse_raw, refused  # pixels only: any annotation parse aborts the stage
    try:
        warmup = {a: [run(a, batches[i]) for i in range(K.COST_WARMUP)] for a in K.COST_ENDPOINTS}
        passes = []
        for p, order in enumerate(K.ROTATIONS):
            for arm in order:
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                recs = [run(arm, b) for b in batches]
                passes.append({"pass": p, "endpoint": arm, "canvas": CANVAS[arm], "batches": recs, **pass_totals(recs),
                               "peak_allocated": torch.cuda.max_memory_allocated()})
    finally:
        LD.parse_raw = parse_raw
    med, ratios, per_pass = aggregate(passes)
    for a in K.COST_ENDPOINTS:
        med[a]["parameters"] = sum(p.numel() for p in models[a].parameters())
    K.require(dict(done) == {a: K.COST_PER_ENDPOINT for a in K.COST_ENDPOINTS}
              and counts["forwards"] == len(K.COST_ENDPOINTS) * K.COST_PER_ENDPOINT == 200, f"cost4 forwards {dict(done)}")
    return {"warmup": warmup, "passes": passes, "medians": med, "ratios": ratios, "ratios_per_pass": per_pass,
            "order_executed": [[a for p, a in ORDER if p == i] for i in range(K.COST_PASSES)],
            "forwards_per_endpoint": dict(done), "annotation_parse_during_timing": "refused (LD.parse_raw replaced); none occurred"}


def cost4(c, out, counts):
    torch = K.torch_mod()
    K.PHASE["phase"] = "cost4:load"
    K.stage_closed(c, "eval-L0")  # refuse before any model is loaded; load_l0 compares its weights_sha256
    models, sources = {}, {}
    for arm in K.COST_ENDPOINTS:
        models[arm], sources[arm] = LOADERS[arm](c, torch)
        gc.collect()
    torch.cuda.empty_cache()
    before = {a: _digest(m) for a, m in models.items()}
    K.PHASE["phase"] = "cost4:timing"
    res = timing(c, counts, models)
    K.PHASE["phase"] = "cost4:close"
    K.require(before == {a: _digest(m) for a, m in models.items()}, "cost4 changed a model state")
    old = K.read_json(K.pinned(c["R27_result"]))["final_cost"]["ratios"]["P"]
    for a in K.COST_ENDPOINTS:
        sources[a]["state_sha256"] = before[a]
    return {"label": "four-endpoint joint timing (task R27-002 section 5)", "note": NOTE, "L0_weights_sha256": before["L0"],
            "sources": sources, "states_unchanged_by_timing": True, **res,
            "informational_P_over_S15": {
                "this_run": {k: res["ratios"][k]["P/S15"] for k in ("model_gpu", "end_to_end")},
                "R27_final_cost": {"model_gpu": old["model_gpu_ratio_vs_N"], "end_to_end": old["end_to_end_ratio_vs_N"]},
                "gate": None, "note": "信息性并列，不设门；原 R27 终态成本与 E 判据保持不变"},
            "protocol": {"batch": K.COST_BATCH, "warmup_per_endpoint": K.COST_WARMUP, "passes": K.COST_PASSES,
                         "batches_per_pass": K.COST_BATCHES_PER_PASS, "images": IMAGES, "precision": "float32, autocast off",
                         "statistic": "median over the three pass totals per endpoint, then the ratio",
                         "device": torch.cuda.get_device_name(0), "torch": str(torch.__version__)}}
