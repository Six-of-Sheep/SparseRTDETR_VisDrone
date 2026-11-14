"""R28 checks (TASK_R28_001 section 5 items 2-5, AMENDMENT_002 section 2).

cpu(c, out, counts)       synthetic unit tests (subprocess), declaration, construction from the parent, input identity,
                          and the small CPU fixture with a NON-identity synthetic demand: u1, u2, save u1, restore,
                          replay u2 (3 updates, 6 forwards), refusals.
zero(c, out, counts)      CUDA zero-step identity: W engine with a constant demand (identity plan, bypass) and the
                          training observer installed, against the certified L0 engine, on the two microbatches of
                          the formal first window (4 forwards, no optimizer step).
accept(c, out, counts)    CUDA, the arm's production path with its final demand source: u1, u2, save u1, restore,
                          replay u2 (3 updates, 6 forwards); shared audit fields against R27 formal-P / formal-C.
evalpath(c, out, counts)  the L0 e120 EMA through the R28 evaluation path with a constant demand must reproduce the
                          stored L0 predictions (141 forwards, no scoring); then a non-identity deployment check on
                          two batches: wrapped output against the native forward of the resampled image with an
                          independently written inverse map.
All engines of these stages are discarded; none of their states can enter formal training.
"""
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import r28_common as K
import l0_checks_state as H
import r28_loop as LP

TESTS = ("tests/test_r28_warp.py", "tests/test_r28_resample.py", "tests/test_r28_summary.py")
SYNTHETIC = {"kind": "synthetic", "seed": 0}
CONSTANT = {"kind": "constant"}


def _first_windows(c, n):
    from formal_data import windows
    from sparse_rtdetr.research import scale_baseline as S
    data = S.train_dataset(c["data"]["train_core"], S.S15_HW, seed=0)
    return list(windows(data, K.PARENT_UPDATE, limit=n))


def _stage(out, name):
    out = Path(out)
    K.require(out.resolve() == (K.ROUND / name).resolve(), f"this action writes <round>/{name}")
    return out


def cpu(c, out, counts):
    torch = K.torch_mod()
    from r25_run import setup
    from s15_formal_train import _bn_counts, _differences, _result_body
    from sparse_rtdetr.protocol_r18 import checkpoint as ck
    from sparse_rtdetr.research import r28_resample as R
    from sparse_rtdetr.research import r28_warp as W
    from sparse_rtdetr.research import scale_baseline as S
    K.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_available(), "CPU checks hide CUDA")
    out = _stage(out, "cpu")
    setup("cpu")
    threads = int(os.environ.get("R28_CPU_THREADS", "16"))
    torch.set_num_threads(threads)
    t0 = time.perf_counter()
    res = {"stage": "cpu", "workload": c["workload"]["cpu"], "threads": torch.get_num_threads()}
    K.PHASE["phase"] = "cpu:pytest"
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *[str(K.ROOT / t) for t in TESTS]],
                       capture_output=True, text=True, cwd=str(K.ROOT), env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    res["pytest"] = {"returncode": p.returncode, "tail": p.stdout.strip().splitlines()[-1:], "tests": list(TESTS),
                     "note": "synthetic unit tests in a subprocess; their forwards are not in this stage's counts"}
    K.require(p.returncode == 0, "synthetic pytest failed:\n" + p.stdout[-6000:])
    K.PHASE["phase"] = "cpu:declaration"
    decl = {a: {"identity_sha256": R.IDENTITY_SHA256[a], "equal_to_contract": c["identity"][a] == R.IDENTITY[a]
                and c["identity_sha256"][a] == R.IDENTITY_SHA256[a], "detector_new_learnable_parameters":
                R.IDENTITY[a]["detector_new_learnable_parameters"]} for a in R.ARMS}
    K.require(all(v["equal_to_contract"] and v["detector_new_learnable_parameters"] == 0 for v in decl.values())
              and R.IDENTITY_SHA256["W"] != R.IDENTITY_SHA256["K"] and c["definition"] == W.DEFINITION
              and (W.EPS, W.LO, W.HI, W.GRID) == (0.2, 2.0 / 3.0, 1.5, (12, 21)), ("R28 declaration", decl))
    res["declaration"] = decl
    K.PHASE["phase"] = "cpu:construct"
    parent = K.pinned(c["parent"])
    e, checks = R.create(parent, "cpu", "W", SYNTHETIC, fixture=True, check=True)
    ps = torch.load(parent, map_location="cpu", weights_only=True)
    sd, esd = e.model.state_dict(), e.ema.module.state_dict()
    keys = {"online_equal_parent": list(sd) == list(ps["model"]), "EMA_equal_parent": list(esd) == list(ps["ema"]["module"]),
            "state_keys": len(sd), "demand_or_r28_keys": sum("r28" in k or "demand" in k for k in sd)}
    K.require(keys["online_equal_parent"] and keys["EMA_equal_parent"] and keys["demand_or_r28_keys"] == 0, ("native key set", keys))
    del ps
    res.update(native_keys=keys, create_checks={k: v for k, v in checks.items() if k != "new_parameter_names"})
    K.PHASE["phase"] = "cpu:input"
    (x, t, inp), = _first_windows(c, 1)
    theirs = H.r27_rows(c, "P", 1)[0]["input"]
    K.require(set(inp) == set(theirs) == set(K.INPUT_FIELDS), "input record keys")
    for k in K.INPUT_FIELDS:
        K.require(type(inp[k]) is type(theirs[k]) and inp[k] == theirs[k], f"input identity differs from R27 formal-P row 1: {k}")
    res["input_identity"] = {"equal_fields": list(K.INPUT_FIELDS), "record": inp, "window_shape": list(x.shape)}
    x, t = x[:2], t[:2]
    before = [tt["boxes"].clone() for tt in t]
    res["fixture"] = fx = {"images": [tt["image_id"] for tt in t], "physical_batch": 1, "accumulation": 2, "precision": "float32",
                           "demand": e.r28_ctx.binding(), "non_identity": True}
    K.register_counts(e, counts)
    obs = LP.install_observer(e)
    bn0 = _bn_counts(e.model)
    rows, readings = [], []
    for i in (1, 2):
        K.PHASE["phase"] = f"cpu:fixture u{i}"
        r = e.train_window(x, t, capture=True)
        K.require(r["update"] == K.PARENT_UPDATE + i and r["finite"] and e.ema.updates == e.completed_updates, "fixture clock")
        rows.append(r)
        readings.append(obs.update_row(t))
        if i == 1:
            side = K.save(c, e, out / "cpu-fixture-u1.pth", "cpu-fixture")
    K.require(all(torch.equal(a, tt["boxes"]) for a, tt in zip(before, t)), "the forward modified the caller's targets")
    K.require(all(r["identity_images"] == 0 and r["slope_min"] < 0.99 and r["slope_max"] > 1.01 for r in readings),
              f"fixture plans are not non-identity: {readings}")
    expected = S.capture(e)
    erow = obs.epoch_row(61)
    fx["BN_delta_two_updates"] = sorted(set(v - bn0[k] for k, v in _bn_counts(e.model).items()))
    K.require(fx["BN_delta_two_updates"] == [4], f"fixture BN delta {fx['BN_delta_two_updates']}")
    del e
    gc.collect()
    K.PHASE["phase"] = "cpu:fixture restore+replay"
    clone, _ = R.create(parent, "cpu", "W", SYNTHETIC, fixture=True, check=True)
    K.register_counts(clone, counts)
    obs2 = LP.install_observer(clone)
    K.load(c, clone, out / "cpu-fixture-u1.pth", "cpu-fixture")
    replay = clone.train_window(x, t, capture=True)
    reading2 = obs2.update_row(t)
    got = S.capture(clone)
    diff = _differences(expected, got) + _differences(_result_body(rows[1]), _result_body(replay))
    K.require(not diff and ck.exact_equal(expected, got) and ck.exact_equal(expected["rng"], got["rng"]) and reading2 == readings[1],
              f"fixture replay differs {diff[:5]}")
    K.require(counts["actual_updates"] == 3 and counts["forwards"] == counts["forward_attempts"] == 6, f"fixture counts {dict(counts)}")
    fx.update(saved_u1=side, full_state_exact=True, rng_exact=True, replay_result_exact=True, observer_replay_exact=True,
              loss=[r["loss"] for r in rows], replay_loss=replay["loss"], observer_updates=readings, observer_epoch=erow,
              elapsed_seconds=[r["elapsed_seconds"] for r in rows + [replay]])
    K.PHASE["phase"] = "cpu:refusals"
    st = out / "cpu-fixture-u1.pth"
    c0 = K.l0_contract(c)
    cases = [("fixture state read in formal scope", lambda: K.read_side(c, st, "formal", "W")),
             ("fixture state loaded as formal", lambda: K.load(c, clone, st, "formal")),
             ("W fixture state read as arm K", lambda: K.read_side(c, st, "cpu-fixture", "K")),
             ("L0 formal e120 state loaded as W", lambda: K.load(c, clone, c["L0_endpoint"]["path"], "formal")),
             ("check engine writing a formal state", lambda: K.save(c, clone, out / "refused.pth", "formal")),
             ("formal engine with a synthetic demand", lambda: R.check_source("W", SYNTHETIC, False)),
             ("arm K with a network demand", lambda: R.check_source("K", {"kind": "network"}, True)),
             ("endpoint_state without an endpoint", lambda: K.endpoint_state(c, "W"))]
    res["refused"] = [H.refused(label, fn) for label, fn in cases]
    K.require(ck.exact_equal(S.capture(clone), got) and not (out / "refused.pth").exists(), "a refused action changed the engine or wrote a file")
    res.update(fixture_updates=3, full_forwards=6, formal_state_usable=False, L0_contract_sha256=c0["contract_sha256"],
               cpu_seconds_wall=time.perf_counter() - t0)
    return res


def zero(c, out, counts):
    import r27_run as R27
    from sparse_rtdetr.protocol_r18 import checkpoint as ck
    from sparse_rtdetr.protocol_r18.engine import tensor_digest
    from sparse_rtdetr.research import r27_l0_control as L0
    from sparse_rtdetr.research import r28_resample as R
    from sparse_rtdetr.research.r27_local_memory import H_HW
    torch = K.torch_mod()
    out = _stage(out, "zero")
    dev = "cuda:0"
    (x, t, inp), = _first_windows(c, 1)
    K.require(inp["update"] == K.PARENT_UPDATE + 1 and tuple(x.shape) == (16, 3, *H_HW), "zero-step window")
    K.PHASE["phase"] = "zero:reference (L0 engine)"
    ref_e, _ = L0.create(c["parent"]["path"], dev)
    ref_e.model.register_forward_hook(lambda *_: counts.update(reference_forwards=1))
    ref_e.optimizer.register_step_post_hook(lambda *_: counts.update(actual_updates=1))
    old_names = [n for n, _ in ref_e.model.named_parameters()]
    rng0 = ck.capture_rng(ref_e.device)
    p_ref0, bn_ref0 = tensor_digest(ref_e.model.named_parameters()), H.bn_digest(ref_e.model)
    nl = R27.micro_steps(ref_e.model, ref_e.criterion, x, t, dev, "bfloat16", old_names=old_names)
    nl_rng = ck.capture_rng(ref_e.device)
    p_ref1, bn_ref1 = tensor_digest(ref_e.model.named_parameters()), H.bn_digest(ref_e.model)
    del ref_e
    gc.collect()
    torch.cuda.empty_cache()
    K.PHASE["phase"] = "zero:W with a constant demand"
    e, checks = R.create(c["parent"]["path"], dev, "W", CONSTANT, check=True)
    K.register_counts(e, counts)
    LP.install_observer(e)
    names = [n for n, _ in e.model.named_parameters()]
    start_rng_equal = ck.exact_equal(ck.capture_rng(e.device), rng0)
    ck.restore_rng(rng0, e.device)
    p0, bn0 = tensor_digest(e.model.named_parameters()), H.bn_digest(e.model)
    mine = R27.micro_steps(e.model, e.criterion, x, t, dev, "bfloat16", old_names=old_names)
    mine_rng = ck.capture_rng(e.device)
    p1, bn1 = tensor_digest(e.model.named_parameters()), H.bn_digest(e.model)
    records = e.r28_ctx.records
    eq = {"parameter_names_equal": names == old_names,
          "start_parameters_equal": p0 == p_ref0 == checks["old_parameters_sha256"],
          "start_BN_buffers_equal": bn0 == bn_ref0,
          "start_RNG_equal_after_construction": start_rng_equal,
          "losses_equal": mine["losses"] == nl["losses"] and mine["loss"] == nl["loss"],
          "outputs_equal": mine["outputs"] == nl["outputs"],
          "DN_equal": mine["dn"] == nl["dn"],
          "native_preclip_gradients_equal": mine["old_gradient_sha256"] == nl["old_gradient_sha256"],
          "RNG_after_equal": ck.exact_equal(mine_rng, nl_rng),
          "BN_buffers_after_equal": bn1 == bn_ref1,
          "W_non_native_gradients_empty": mine["new_gradient_norms"] == {},
          "parameters_unchanged_W": p1 == p0, "parameters_unchanged_reference": p_ref1 == p_ref0,
          "identity_bypass_taken": len(records) == 2 and all(r["identity_images"] == 8 for r in records),
          "observer_installed": hasattr(e, "r28_observer")}
    del e
    gc.collect()
    torch.cuda.empty_cache()
    res = {"stage": "zero", "workload": c["workload"]["zero"], "input": inp, "precision": "bfloat16", "microbatches": [8, 8],
           "reference": "r27_l0_control.create (the certified L0 engine) consuming I_H",
           "production": "r28_resample.create arm W, constant demand (identity plan), observer installed",
           "comparisons": eq, "loss": {"W": mine["loss"], "reference": nl["loss"]},
           "microbatch_losses": {"W": mine["losses"], "reference": nl["losses"]},
           "outputs_sha256": {"W": mine["outputs"], "reference": nl["outputs"]},
           "forwards": {"reference": counts["reference_forwards"], "W": counts["forwards"]},
           "optimizer_steps": counts["actual_updates"], "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
           "formal_state_usable": False,
           "note": "the identity plan takes the bypass; the non-identity path is checked by the unit tests, the CPU "
                   "fixture, accept and the second part of evalpath"}
    K.require(all(eq.values()), ("CUDA zero-step identity failed", {k: v for k, v in eq.items() if not v}))
    K.require(counts["reference_forwards"] == 2 and counts["forwards"] == counts["forward_attempts"] == 2
              and counts["actual_updates"] == 0, f"zero-step counts {dict(counts)}")
    return res


def accept(c, out, counts):
    from s15_formal_train import _bn_counts, _differences, _result_body
    from sparse_rtdetr.protocol_r18 import checkpoint as ck
    from sparse_rtdetr.research import scale_baseline as S
    torch = K.torch_mod()
    arm = K.ARM
    R = K.arm_module(arm)
    out = _stage(out, f"accept-{arm}")
    if arm == "F":      # G0 is arm W's investment gate, not a gate of the mapping control
        K.require(K.followup_arm(c) == "F", "arm F is accepted only in the mapping-control round")
        K.stage_closed(c, "fovea-check", "F")
    else:
        K.g0_passed(c)
    source = K.demand_source(c, arm)
    wins = _first_windows(c, 2)
    K.PHASE["phase"] = "accept:construct"
    e, checks = R.create(c["parent"]["path"], "cuda:0", arm, source)
    K.register_counts(e, counts)
    obs = LP.install_observer(e)
    bn0 = _bn_counts(e.model)
    rows, results, readings = [], [], []
    for i, (x, t, inp) in enumerate(wins, 1):
        K.PHASE["phase"] = f"accept:u{i}"
        r = e.train_window(x, t, capture=True)
        K.require(r["update"] == inp["update"] == K.PARENT_UPDATE + i and r["finite"] and e.ema.updates == e.completed_updates,
                  "acceptance update clock/input")
        results.append(r)
        readings.append(obs.update_row(t))
        rows.append({"input": inp, "gt_count": int(sum(len(v["labels"]) for v in t)), "loss": r["loss"], "lr": r["lr"],
                     "gradient_norm_before_clip": r["gradient_norm_before_clip"], "ema_updates": e.ema.updates})
        if i == 1:
            saved = K.save(c, e, out / "u1.pth", "acceptance")
    K.require(all(r["identity_images"] == 0 for r in readings), "the production demand produced an exactly uniform plan")
    expected = S.capture(e)
    erow = obs.epoch_row(61)
    bn = sorted(set(v - bn0[k] for k, v in _bn_counts(e.model).items()))
    K.require(bn == [4], f"BN delta {bn}")
    demand_digest = e.r28_ctx.binding()
    del e
    gc.collect()
    torch.cuda.empty_cache()
    K.PHASE["phase"] = "accept:restore+replay u2"
    clone, _ = R.create(c["parent"]["path"], "cuda:0", arm, source)
    K.register_counts(clone, counts)
    obs2 = LP.install_observer(clone)
    K.load(c, clone, out / "u1.pth", "acceptance")
    x, t, inp = wins[1]
    replay = clone.train_window(x, t, capture=True)
    reading2 = obs2.update_row(t)
    got = S.capture(clone)
    diff = _differences(expected, got) + _differences(_result_body(results[1]), _result_body(replay))
    K.require(not diff and ck.exact_equal(expected, got) and ck.exact_equal(expected["rng"], got["rng"]) and reading2 == readings[1],
              f"replay differs {diff[:5]}")
    K.require(counts["actual_updates"] == 3 and counts["forwards"] == counts["forward_attempts"] == 6, f"acceptance counts {dict(counts)}")
    K.require(clone.r28_ctx.binding() == demand_digest, "demand source changed during acceptance")
    extra = {}
    if arm == "F":
        import r28_fovea_control as FC
        K.PHASE["phase"] = "accept:cross-state refusals"
        extra["refused"] = FC.state_refusals(c, clone, out / "u1.pth", "acceptance")
        K.require(ck.exact_equal(S.capture(clone), got), "a refused action changed the engine")
        extra["fovea_readings"] = [r.get("fovea") for r in readings]
    K.PHASE["phase"] = "accept:shared fields"
    shared = {}
    for a in ("P", "C"):
        shared[a] = [K.compare_shared(row, ref) for row, ref in zip(rows, H.r27_rows(c, a, 2))]
    K.require(shared["P"] == shared["C"], "shared fields of R27 formal-P and formal-C disagree")
    return {"stage": f"accept-{arm}", "workload": c["workload"]["accept"], "arm": arm, "demand": demand_digest,
            "create_checks": {k: v for k, v in checks.items() if k != "new_parameter_names"},
            "first_updates": [{**row, "r28": rd, "output_sha256": r["output_sha256"], "dn_noise_sha256": r["dn_noise_sha256"],
                               "gradient_sha256_before_clip": r["gradient_sha256_before_clip"], "elapsed_seconds": r["elapsed_seconds"]}
                              for row, rd, r in zip(rows, readings, results)],
            "observer_epoch": erow, "saved_u1": saved, "replay_exact": True, "full_state_exact": True, "rng_exact": True,
            "observer_replay_exact": True, "replay_elapsed_seconds": replay["elapsed_seconds"], "BN_delta_two_updates": bn,
            "shared_fields": {"fields": list(K.SHARED_FIELDS), "rows": [1, 2], "R27_formal_P": True, "R27_formal_C": True},
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "formal_state_usable": False, **extra}


def _reference_inverse(knots, v):
    """Inverse map written apart from r28_warp.inv: numpy interpolation inside [0, 1], the end slopes outside."""
    import numpy as np
    n = len(knots) - 1
    grid = np.arange(n + 1, dtype=np.float64) / n
    m0, m1 = (knots[1] - knots[0]) * n, (knots[-1] - knots[-2]) * n
    inside = np.interp(v, knots, grid)
    return np.where(v < 0, v / m0, np.where(v > 1, 1 + (v - 1) / m1, inside))


def _compare_predictions(mine, stored, budget):
    import numpy as np
    K.require(len(mine) == len(stored), f"top{budget}: record count differs")
    keys = all(a["image_id"] == b["image_id"] and a["query_id"] == b["query_id"] and a["category_id"] == b["category_id"]
               for a, b in zip(mine, stored))
    s = np.abs(np.array([a["score"] for a in mine]) - np.array([b["score"] for b in stored]))
    bm, bs = np.array([a["bbox_xyxy"] for a in mine]), np.array([b["bbox_xyxy"] for b in stored])
    rel = np.abs(bm - bs) / np.maximum(1.0, np.abs(bs))
    return {"records": len(mine), "identity_and_order_equal": bool(keys), "exactly_equal": mine == stored,
            "score_max_abs_difference": float(s.max()), "bbox_max_abs_difference_px": float(np.abs(bm - bs).max()),
            "bbox_max_relative_difference": float(rel.max()),
            "within_tolerance": bool(keys and s.max() <= 1e-5 and rel.max() <= 1e-5)}


def evalpath(c, out, counts):
    import numpy as np
    import l0_common as B
    import r28_eval as EV
    from sparse_rtdetr.protocol_r18.engine import tensor_digest
    from sparse_rtdetr.research import r28_resample as R
    from sparse_rtdetr.research import r28_warp as W
    torch = K.torch_mod()
    out = _stage(out, "evalpath")
    K.PHASE["phase"] = "evalpath:load"
    side, state = B.endpoint_state(K.l0_contract(c))
    K.require(side["file"] == c["L0_endpoint"] and tensor_digest(state.items()) == c["L0_weights_sha256"], "L0 e120 EMA identity")
    model = R.build_eval_model(state, "cuda:0", "W", CONSTANT, check=True)
    syn = R.build_eval_model(state, "cuda:0", "W", {"kind": "synthetic", "seed": 1}, check=True)
    native = type(syn)._r28_native
    extra = []

    def deployment(x, rows, index):
        if index >= K.EVALPATH_EXTRA:
            return
        with torch.autocast("cuda", enabled=False):
            o = syn(x)
            plan = syn.r28_ctx.plan(x)
            raw = native.forward(syn, W.resample(x, plan))
        counts.update(forwards=2)
        K.require(torch.equal(o["pred_logits"], raw["pred_logits"]), "non-identity deployment: logits differ")
        b = W.cxcywh_to_xyxy(raw["pred_boxes"].double()).cpu().numpy()
        kx, ky = plan.kx.cpu().numpy(), plan.ky.cpu().numpy()
        back = np.stack([np.stack([_reference_inverse(kx[i], b[i, :, 0]), _reference_inverse(ky[i], b[i, :, 1]),
                                   _reference_inverse(kx[i], b[i, :, 2]), _reference_inverse(ky[i], b[i, :, 3])], -1)
                         for i in range(len(rows))])
        got = W.cxcywh_to_xyxy(o["pred_boxes"].double()).cpu().numpy()
        moved = np.abs(b - back).max()
        extra.append({"batch": index, "images": [r["stable_image_id"] for r in rows],
                      "max_abs_difference_to_reference_inverse": float(np.abs(got - back).max()),
                      "max_abs_shift_by_the_inverse_map": float(moved),
                      "edges_outside_unit_interval": int(((b < 0) | (b > 1)).sum()),
                      "positive_width_height": bool((o["pred_boxes"][..., 2:] > 0).all()),
                      "slopes": [float(plan.mx.min()), float(plan.mx.max()), float(plan.my.min()), float(plan.my.max())],
                      "identity_images": int(plan.identity.sum())})

    K.PHASE["phase"] = "evalpath:inference"
    data, preds, timing = EV.infer(c, model, counts, extra=deployment)
    K.require(counts["forwards"] == K.EVAL_WARMUP + K.EVAL_BATCHES + 2 * K.EVALPATH_EXTRA, f"evalpath forward count {dict(counts)}")
    K.require(tensor_digest(model.state_dict().items()) == c["L0_weights_sha256"] == tensor_digest(syn.state_dict().items()),
              "evalpath changed a model state")
    del model, syn
    gc.collect()
    torch.cuda.empty_cache()
    K.PHASE["phase"] = "evalpath:compare"
    cmp = {}
    for b in K.BUDGETS:
        with K.pinned(c["L0_predictions"][str(b)]).open() as f:
            stored = json.load(f)
        cmp[str(b)] = _compare_predictions(preds[b], stored, b)
        del stored
    K.require(all(v["within_tolerance"] for v in cmp.values()), ("evaluation path differs from the stored L0 predictions", cmp))
    K.require(len(extra) == K.EVALPATH_EXTRA and all(r["max_abs_difference_to_reference_inverse"] <= 1e-6 and r["positive_width_height"]
                                                     and r["identity_images"] == 0 and r["max_abs_shift_by_the_inverse_map"] > 1e-3
                                                     for r in extra), ("non-identity deployment check failed", extra))
    return {"stage": "evalpath", "workload": c["workload"]["evalpath"], "weights": "L0 e120 EMA", "weights_sha256": c["L0_weights_sha256"],
            "identity_path": {"demand": "constant", "stored_predictions": c["L0_predictions"], "comparison": cmp,
                              "tolerance": "identity and order equal; score abs <= 1e-5; bbox relative <= 1e-5"},
            "non_identity_deployment": {"demand": "synthetic seed 1", "batches": extra, "tolerance": 1e-6,
                                        "reference": "native forward of the resampled image, boxes mapped back by a separately "
                                                     "written numpy inverse (np.interp inside, end slopes outside)"},
            "scorer_calls": 0, **timing}
