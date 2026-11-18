"""R30-002 formal training, evaluation and summary of the four units L0-s1, FG-s1, L0-s2, FG-s2 (TASK_R30_SEEDS_002).

  formal      one unit per process (R30_UNIT), 18,240 updates from the e060 parent of its seed.  The loop is the loop of
              task 009 (r28_loop.run_loop) with the seed as a parameter and the S15 replication ledger of that seed as
              the reference ledger; arm L0 writes the ledger rows of R27-002 (no mapping readings), arm FG the rows of
              task 009.  Every window is compared with the reference ledger before and after the update.
  evaluate    one unit per process: the e120 EMA module, view A, 548 images, top600 / top300 scored once each, one
              per-class COCOeval whose tensors are kept.  In a dev round: two batches of the short dev endpoint, no scoring.
  scorecheck  dev rounds only, CPU, run by hand: the scoring half of `evaluate` applied to the stored seed-0 F_geo
              predictions of task 009 must reproduce the stored values and per-class rows (no model, not a result).
  summarize   CPU: the four ledgers against the S15 ledgers and against each other, the 14 cells per seed and the
              reading of task section 4.
"""
import gc
import json
import os
import time
from pathlib import Path

import r28_common as K
import l0_loop as LL
import r28_loop as LP
import r29_input as RI
import r30_control as R30

LEDGER, OBS, TIMING = LL.LEDGER, LP.OBS, LP.TIMING
L0_ROW_KEYS = ("input", "gt_count", "loss", "lr", "gradient_norm_before_clip", "microbatch_losses", "update_seconds", "ema_updates")
SIZE_STATS = {"small": ("APS", 3), "medium": ("APM", 4), "large": ("APL", 5)}


def formal_dir(c, unit):
    found = []
    pattern = K.formal_re(unit)
    for d in sorted(K.ROUND.iterdir()) if K.ROUND.is_dir() else ():
        if not (d.is_dir() and pattern.fullmatch(d.name) and (d / "CLOSED.json").is_file()):
            continue
        closed = K.read_json(d / "CLOSED.json")
        if closed.get("status") == "PASS" and closed.get("unit") == unit and closed.get("contract_sha256") == c["contract_sha256"]:
            found.append(d)
    K.require(len(found) == 1, f"exactly one closed PASS formal-{unit} directory is required under {K.ROUND}: {[d.name for d in found]}")
    return found[0]


def guard(c, out, resume, unit):
    K.require(unit in R30.ORDER and unit in c["units"], f"the R30 round trains {R30.ORDER}: {unit!r}")
    arm, seed = R30.parse(unit)
    R30.stage_closed(c, "r30-check")
    wire = R30.stage_closed(c, "wire-s0")
    accept = R30.stage_closed(c, f"accept-s{seed}")
    K.require(all(wire["FG_equal_to_the_accept_FG_receipts"].values()) and all(wire["L0_equal_to_the_R27_002_receipts"].values())
              and accept.get("formal_training_allowed") is True and accept["restore_replay_equal"][unit] is True
              and accept["same_windows_as_the_S15_ledger"] is True, f"the checks do not open the formal run of unit {unit}")
    K.require(not any(k.startswith("R28_DEV_") and k != "R28_DEV_SPAN" for k in os.environ) and "R28_PIPELINE" not in os.environ,
              "dev override flags are refused in the R30 round")
    for earlier in R30.ORDER[:R30.ORDER.index(unit)]:
        formal_dir(c, earlier)                   # task section 3: L0-s1 -> FG-s1 -> L0-s2 -> FG-s2
    out = Path(out).resolve()
    K.require(out.parent == K.ROUND.resolve(), "formal output must be directly under the round")
    pattern = K.formal_re(unit)
    closed = sorted(d.name for d in K.ROUND.iterdir() if d.is_dir() and pattern.fullmatch(d.name) and (d / "CLOSED.json").is_file())
    K.require(not closed, f"this round already has a closed formal directory {closed}: another formal run is refused")
    if resume is None:
        K.require(out.name == f"formal-{unit}", f"a formal run without --resume writes <round>/formal-{unit}")
        return None
    path, side = R30.read_side(c, resume, "formal", unit)
    m = pattern.fullmatch(out.name)
    K.require(m is not None and m.group(1) is not None, f"a resumed formal run writes <round>/formal-{unit}-rN (N>=2)")
    K.require(path.parent.parent == K.ROUND.resolve() and pattern.fullmatch(path.parent.name) is not None and path.parent != out,
              "formal resume source must be a state in this round's formal directory of the same unit")
    K.require(side["global_update"] % K.WINDOWS_PER_EPOCH == 0 or K.DEV, "formal resume source must be an epoch boundary state")
    return path, side


def restore_files(side, src_dir, out, refs, arm):
    """Ledger prefix (L0 rule) and, for arm FG, the observation file prefix recorded by the resume state."""
    audit = side["audit"]
    copied, prior = LL.restore_ledger({**side, "audit": None if audit is None else audit["ledger"]}, src_dir, out, refs)
    if arm != "FG":
        return copied, prior
    head = b""
    if audit is not None:
        o = audit["observations"]
        K.require(Path(o["path"]) == src_dir / OBS, "state observation binding is not the file next to the state")
        with (src_dir / OBS).open("rb") as f:
            head = f.read(o["bytes"])
        K.require(len(head) == o["bytes"] and K.sha_bytes(head) == o["sha256"], "prior observation prefix differs from the state binding")
    with (out / OBS).open("xb") as dst:
        dst.write(head)
        dst.flush()
        os.fsync(dst.fileno())
    return copied, prior


def run_loop(c, e, unit, out, stop, counts, *, save_points, refs, label, prior_rows, first, pipeline, delivery):
    from formal_data import WINDOWS_PER_EPOCH
    from sparse_rtdetr.research import scale_baseline as S
    arm, seed = R30.parse(unit)
    obs = e.r28_observer if arm == "FG" else None
    history = out / LEDGER
    data = S.train_dataset(c["data"]["train_core"], S.S15_HW, seed=seed)
    start = e.completed_updates
    base = {"start": start, "recompute": S.RECOMPUTE["calls"], "bn": LL.bn_clocks(e.model), "hooks": dict(counts)}
    bn0 = base["bn"]
    K.require(start < stop and all(start < p <= stop for p in save_points), "R30 loop span / save points")
    prior_rows = dict(prior_rows)
    saves, seen, replayed, compared, seconds, checked, epochs = [], {}, 0, 0, 0.0, [], []
    check = R30.reference_check(refs, label)
    wall = time.monotonic()
    mark = time.perf_counter()
    with delivery.stream(pipeline, data, start, stop, check) as stream:
        with history.open("a", buffering=1) as ledger, (out / TIMING).open("a", buffering=1) as timing_file:
            obs_file = (out / OBS).open("a", buffering=1) if obs is not None else None
            try:
                for item in stream:
                    x, t, inp = item.x, item.t, item.inp
                    ids = [tt["image_id"] for tt in t]
                    s_ = seen.setdefault(inp["epoch"], set())
                    K.require(len(set(ids)) == len(ids) and not (s_ & set(ids)), "R30 occurrence: image repeated within an epoch")
                    s_.update(ids)
                    K.PHASE["phase"] = f"{unit}:input identity {e.completed_updates + 1}"
                    K.require(inp["update"] == e.completed_updates + 1 and inp["update"] in refs,
                              f"window {inp['update']} is not the next update of the schedule")
                    ref_row = refs[inp["update"]]
                    bad = LL.input_differences(inp, ref_row["input"])
                    K.require(not bad, f"input record differs from the {label} ledger before update {inp['update']} "
                                       f"(the model was not updated with this window): {bad}")
                    K.PHASE["phase"] = f"{unit}:update {e.completed_updates + 1}"
                    called = time.perf_counter()
                    xd, moved = delivery.take(item)
                    res = e.train_window(xd, t)
                    train_call = time.perf_counter() - called
                    del xd
                    K.require(res["update"] == inp["update"] and res["finite"] and e.ema.updates == e.completed_updates,
                              "R30 update clock/input")
                    row = {"input": inp, "gt_count": int(sum(len(v["labels"]) for v in t)), "loss": res["loss"],
                           "lr": res["lr"], "gradient_norm_before_clip": res["gradient_norm_before_clip"],
                           "microbatch_losses": res["microbatch_losses"], "update_seconds": res["elapsed_seconds"],
                           "ema_updates": e.ema.updates}
                    if obs is not None:
                        row["r28"] = obs.update_row(t)
                    bad = LL.shared_differences(row, ref_row)
                    K.require(not bad, f"shared fields differ from the {label} ledger at update {res['update']}: {bad}")
                    K.compare_shared(row, ref_row)
                    compared += 1
                    if res["update"] in prior_rows:
                        K.require(prior_rows.pop(res["update"]) == {k: v for k, v in row.items() if k not in LL.REPLAY_SKIP},
                                  f"recovery replay differs from the recorded trajectory at update {res['update']}")
                        replayed += 1
                    ledger.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                    seconds += res["elapsed_seconds"]
                    if res["update"] == first["update"]:
                        print(json.dumps({"first_update_loss": {"update": res["update"], unit: res["loss"], first["reference"]: first["loss"],
                                                                "note": "informational reading, not a gate"}}), flush=True)
                    if obs is not None and (e.completed_updates % WINDOWS_PER_EPOCH == 0 or e.completed_updates == stop):
                        erow = obs.epoch_row(inp["epoch"])
                        obs_file.write(json.dumps(erow, allow_nan=False) + "\n")
                        epochs.append(erow["epoch"])
                    if e.completed_updates in save_points:
                        ledger.flush()
                        os.fsync(ledger.fileno())
                        if obs_file is not None:
                            obs_file.flush()
                            os.fsync(obs_file.fileno())
                        K.PHASE["phase"] = f"{unit}:accounting before the save at {e.completed_updates}"
                        LL.account(e, counts, base)
                        checked.append(e.completed_updates)
                        LL._available(out)
                        ep, rest = divmod(e.completed_updates, WINDOWS_PER_EPOCH)
                        role = "formal_save" if not rest and ep in K.FORMAL_SAVE_EPOCHS else "recovery"
                        saving = time.perf_counter()
                        audit = {"ledger": K.ref(history), "observations": None if obs is None else K.ref(out / OBS)}
                        saves.append(R30.save(c, e, unit, out / K.endpoint_name(e.completed_updates), "formal", audit, role))
                        moved = {**moved, "save_s": time.perf_counter() - saving}
                    now = time.perf_counter()
                    timing_file.write(json.dumps({"update": res["update"], "cycle_s": now - mark, "wait_s": item.timing["wait_s"],
                                                  "train_call_s": train_call, "train_s": res["elapsed_seconds"],
                                                  "input": {k: item.timing.get(k) for k in ("next_s", "audit_s", "fetch_s", "reference_check_s")},
                                                  "prepared": stream.ready(), **moved}, separators=(",", ":")) + "\n")
                    if e.completed_updates % 16 == 0:
                        print(json.dumps({"unit": unit, "update": e.completed_updates, "loss": res["loss"],
                                          "seconds": res["elapsed_seconds"], "cycle_seconds": now - mark,
                                          **({} if obs is None else {"g_small": row["r28"]["g_small"]})}), flush=True)
                    mark = now
                    del item, x, t
                ledger.flush()
                os.fsync(ledger.fileno())
            finally:
                if obs_file is not None:
                    obs_file.close()
        pipe = stream.account()
    wall = time.monotonic() - wall
    K.PHASE["phase"] = f"{unit}:loop accounting"
    done, hooks, rec = LL.account(e, counts, base)
    K.require(e.completed_updates == stop and done == stop - start and not prior_rows, "R30 loop count")
    return {"updates": done, "hook_updates": hooks["actual_updates"], "hook_forwards": hooks["forwards"],
            "block_recomputations": rec, "block_recomputations_per_update": 16,
            "BN": {"tracked_modules": len(bn0), "training_modules": sum(m for _, m in bn0.values()),
                   "increment_per_training_module": 2 * done, "one_update_per_training_forward": True},
            "saves": saves, "accounting_checked_before_saves_at": checked, "prior_rows_replayed_exact": replayed,
            "shared_rows_compared_live": compared, "observation_epochs_written": epochs,
            "input_pipeline": pipe, "timing_file": TIMING, "window_delivery": delivery.account(),
            "loop_wall_seconds": wall, "update_seconds_sum": seconds,
            "peak_cuda_bytes": K.torch_mod().cuda.max_memory_allocated() if e.device.type == "cuda" else None}


def formal(c, out, counts, resume):
    unit = R30.process_unit()
    arm, seed = R30.parse(unit)
    stop, dev_points = K.formal_stop()
    save_points = dev_points or tuple(ep * K.WINDOWS_PER_EPOCH for ep in K.SAVE_EPOCHS if ep * K.WINDOWS_PER_EPOCH > K.PARENT_UPDATE)
    K.PHASE["phase"] = f"{unit}:formal guard"
    out = Path(out)
    source = guard(c, out, resume, unit)
    K.require(R30.bound_seed() == seed, f"the process is not bound to seed {seed}")
    gpu = R30.gpu_compute_processes()
    label = f"S15 seed {seed}"
    refs, first = R30.reference_rows(c, seed)
    pipeline, mode = R30.formal_mode(c, unit)
    K.PHASE["phase"] = f"{unit}:construct"
    e, info = R30.create(c, arm, seed)
    K.register_counts(e, counts)
    if arm == "FG":
        LP.install_observer(e)
    copied, prior_rows, saves = 0, {}, []
    if source is not None:
        side = R30.load(c, e, unit, source[0], "formal")
        K.require(side == source[1], "resume side-car changed between guard and load")
        copied, prior_rows = restore_files(side, source[0].parent, out, refs, arm)
        K.write_new(out / "RESTORE.json", {"resumed_from": side, "prior_rows_after_state": len(prior_rows), "ledger_rows_copied": copied})
    else:
        K.require(e.completed_updates == K.PARENT_UPDATE, "a fresh formal run starts at the parent update")
        LL._available(out)
        saves.append(R30.save(c, e, unit, out / "checkpoint-epoch-060.pth", "formal", None, "formal_save"))
    start = e.completed_updates
    used = LL.points_after(save_points, start)
    K.require(stop in used, "the stop update must be a save point after the start (the endpoint state is always written)")
    delivery = RI.Delivery(mode, e.device)
    plan = {"unit": unit, "arm": arm, "seed": seed, "parent": c["units"][unit]["parent"], "start_update": start, "stop_update": stop,
            "updates": stop - start, "schedule_first_update": K.PARENT_UPDATE + 1,
            "save_points": {"planned": sorted(set(save_points)), "used_in_this_process": list(used),
                            "at_or_before_the_start": sorted(p for p in set(save_points) if p <= start)},
            "formal_save_epochs": list(K.FORMAL_SAVE_EPOCHS), "scope": "formal", "input_pipeline": pipeline,
            "window_delivery": delivery.binding(), "reference_ledger": c["units"][unit]["reference_ledger"],
            **({"demand": e.r28_ctx.binding(), "mapping": e.r28_mapping} if arm == "FG" else {})}
    loop = run_loop(c, e, unit, out, stop, counts, save_points=set(used), refs=refs, label=label, prior_rows=prior_rows, first=first,
                    pipeline=pipeline, delivery=delivery)
    delivery.close()
    with (out / LEDGER).open() as f:
        head = json.loads(f.readline())
        rows = 1 + sum(1 for _ in f)
    K.require(rows == stop - K.PARENT_UPDATE == copied + loop["shared_rows_compared_live"] and head["input"]["update"] == first["update"]
              and set(head) == set(L0_ROW_KEYS) | ({"r28"} if arm == "FG" else set()), "R30 ledger row count / row keys")
    loop["saves"] = saves + loop["saves"]
    return {"unit": unit, "arm": arm, "seed": seed, "plan": plan, "install": {k: v for k, v in info.items() if k != "new_parameter_names"},
            "resumed": source is not None, "ledger_rows": rows, "ledger_rows_copied_on_resume": copied,
            "shared_rows_compared": copied + loop["shared_rows_compared_live"], "shared_rows_different": 0,
            "shared_fields": list(K.SHARED_FIELDS), "reference_ledger": c["units"][unit]["reference_ledger"],
            "first_update_loss": {"update": first["update"], unit: head["loss"], first["reference"]: first["loss"]},
            "gpu_at_start": gpu, "gpu_at_end": R30.gpu_compute_processes(), "dev_span": os.environ.get("R28_DEV_SPAN"), **loop}


# ================================================================ evaluation
def per_class(rows, parsed, predictions, bridge, keep):
    """The single per-class COCOeval of r28_followup.per_class_sizes (same construction, same check against the scored
    bridge), read for small, medium and large; its accumulated tensors are kept in `keep`."""
    import contextlib
    import io
    import numpy as np
    import r28_followup as FU
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from s15_formal_eval import LARGE_NAMES
    from sparse_rtdetr.r0.evaluation import bridge_inputs
    dataset, det, _ = bridge_inputs(rows, parsed, predictions)
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(); coco.dataset = dataset; coco.createIndex()
        dt = coco.loadRes(det)
        e = COCOeval(coco, dt, "bbox")
        e.params.imgIds = sorted(r["id"] for r in dataset["images"]); e.params.catIds = list(range(1, 11))
        e.params.maxDets = [1, 10, 100]
        e.evaluate(); e.accumulate(); e.summarize()
    K.require(all(float(v) == bridge["metrics"][k] for k, v in zip(LARGE_NAMES, e.stats)),
              "per-class recomputation differs from the scored COCO bridge")
    out = {}
    for label, (key, stat) in SIZE_STATS.items():
        ix = e.params.areaRngLbl.index(label)
        lo, hi = e.params.areaRng[ix]
        values = []
        for k, cat in enumerate(e.params.catIds):
            p = e.eval["precision"][:, :, k, ix, -1]
            v = p[p > -1]
            n = sum(1 for a in dataset["annotations"] if a["category_id"] == cat and not a.get("iscrowd") and lo <= a["area"] <= hi)
            values.append({"category_id": int(cat), "GT": n, key: float(v.mean()) if len(v) else None})
        mean = float(np.mean([v[key] for v in values if v[key] is not None]))
        K.require(abs(mean - e.stats[stat]) < 1e-14, f"per-class {key} macro differs")
        out[key] = values
    FU.keep_tensors(e, dataset, keep)
    return out


def score(c, out, counts, rows, parsed, preds, refs, binding, prefix):
    """top600 / top300 scored once each with the certified scorer, one per-class COCOeval on top600 (tensors kept)."""
    from infer_seed import bridge_values
    from sparse_rtdetr.r0.evaluation import score_budget
    values, metrics_refs, counts_gt, scoring, classes = {}, {}, {}, [], None
    for b in K.BUDGETS:
        t = time.perf_counter()
        metrics = score_budget(rows, parsed, preds[b], c["source_root"])
        counts.update(score_budget_calls=1)
        metrics["R30_binding"] = {"contract_sha256": c["contract_sha256"], "kind": "EMA", "view": "A", "budget": b,
                                  "predictions": refs[str(b)], **binding}
        values[str(b)] = bridge_values(metrics)
        counts_gt[str(b)] = {k: metrics["spatial_filter_counts"][k] for k in ("GT_before", "GT_after")}
        p = out / f"{prefix}-EMA-A-top{b}-metrics.json"
        K.write_new(p, metrics)
        metrics_refs[str(b)] = K.ref(p)
        if b == 600:
            classes = per_class(rows, parsed, preds[b], metrics["COCO_bridge"], out / "cocoeval-top600.npz")
            counts.update(per_class_COCOeval_calls=1)
        scoring.append({"budget": b, "seconds": time.perf_counter() - t})
    K.require(counts["score_budget_calls"] == 2 and counts["per_class_COCOeval_calls"] == 1, "scorer call counts")
    K.require(counts_gt == c["reference"]["GT_counts"], f"scored GT set differs from the seed-0 evaluations: {counts_gt}")
    K.require(all(sorted(values[b]) == sorted(R30.METRICS) for b in R30.BUDGET_KEYS), "metric cells of the scorer")
    return {"values": values, "metrics": metrics_refs, "GT_counts": counts_gt, "per_class_APS_top600": classes["APS"],
            "per_class_APM_top600": classes["APM"], "per_class_APL_top600": classes["APL"],
            "cocoeval_top600_tensors": K.ref(out / "cocoeval-top600.npz"), "scoring": scoring}


def endpoint_state(c, unit):
    stop = K.formal_stop()[0]
    side, s = R30.read_state(c, formal_dir(c, unit) / K.endpoint_name(stop), "formal", unit)
    K.require(side["global_update"] == stop and s["engine"]["completed_updates"] == s["ema"]["updates"] == stop
              and (K.DEV or stop == K.ENDPOINT), "endpoint clocks")
    return side, s["ema"]["module"]


def evaluate(c, out, counts):
    import r28_eval as EV
    from infer_seed import _group, _prefix
    from sparse_rtdetr.protocol_r18 import checkpoint as ck
    from sparse_rtdetr.protocol_r18.engine import tensor_digest
    from sparse_rtdetr.research import r27_l0_control as L0
    from sparse_rtdetr.research import r28_fovea_geo as FG
    torch = K.torch_mod()
    unit = R30.process_unit()
    arm, seed = R30.parse(unit)
    K.PHASE["phase"] = "evaluate/guards"
    out = EV.round_stage(out, f"eval-{unit}")
    formal_name = formal_dir(c, unit).name
    side, state = endpoint_state(c, unit)
    K.require(side["unit"] == unit and side["scope"] == "formal", "endpoint side-car")
    K.require(all(not v.is_floating_point() or v.dtype == torch.float32 for v in state.values()), "EMA not FP32")
    device = torch.device("cuda:0")
    if arm == "L0":
        model = L0.build_eval_model(state, device, seed)
        K.require(getattr(model, "l0_arm", None) == "L0" and L0.is_wrapped(model) and not model.training, "evaluation model is not an L0 eval model")
        extra = {}
    else:
        source = R30.demand_source(c)
        K.require(side["demand"]["file"] == source["file"] and side["mapping"] == FG.mapping(source), "the endpoint was trained with another demand source / width")
        model = FG.build_eval_model(state, device, "FG", source, seed=seed)
        K.require(FG.is_wrapped(model) and not model.training and model.r28_ctx.binding() == side["demand"], "evaluation model is not an F_geo eval model")
        extra = {"demand": side["demand"], "mapping": side["mapping"]}
    weights = tensor_digest(state.items())
    rng = ck.capture_rng(device)
    K.PHASE["phase"] = "evaluate/inference"
    limit = (1, 2) if K.DEV else None
    data, preds, timing = EV.infer(c, model, counts, limit=limit)
    forwards = K.EVAL_WARMUP + K.EVAL_BATCHES if limit is None else sum(limit)
    K.require(counts["forwards"] == forwards and (limit is not None or forwards == 141), "evaluation forward count")
    K.require(tensor_digest(model.state_dict().items()) == weights and ck.exact_equal(ck.capture_rng(device), rng)
              and (arm == "L0" or model.r28_ctx.binding() == side["demand"]), "evaluation changed model / RNG / demand source")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    head = {"unit": unit, "arm": arm, "seed": seed, "endpoint": side["file"], "formal_stage": formal_name, "weights_sha256": weights, "kind": "EMA",
            "epoch": side["global_update"] // K.WINDOWS_PER_EPOCH, "global_update": side["global_update"], "view": "A", **extra, **timing}
    if limit is not None:
        K.require(all(len(preds[b]) == limit[1] * K.EVAL_BATCH * b for b in K.BUDGETS), "dev prediction record counts")
        return {**head, "dev_wiring": True, "scored": False, "batches": list(limit),
                "note": "dev round: two batches of a short dev endpoint through the evaluation model; nothing is scored or kept"}
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
    K.PHASE["phase"] = "evaluate/scoring"
    binding = {"unit": unit, "arm": arm, "seed": seed, "epoch": head["epoch"], "checkpoint": side["file"], "weights_sha256": weights, **extra}
    return {**head, "predictions": refs, **score(c, out, counts, rows, parsed, preds, refs, binding, unit)}


def scorecheck(c, out, counts):
    """Dev rounds only: the scoring half of `evaluate` on the stored seed-0 F_geo predictions of task 009."""
    from sparse_rtdetr.research import scale_baseline as S
    K.require(K.DEV, "the scoring check is a dev-round action")
    out = Path(out)
    fg = c["seed0"]["FG"]
    ev = K.read_json(K.pinned(fg["eval_closed"]))
    preds = {b: K.read_json(K.pinned(fg["FG_predictions"][str(b)])) for b in K.BUDGETS}
    data = S.development(c["data"]["development"], S.S15_HW)
    K.require([r["stable_image_id"] for r in data.records] == c["development_ids"], "development members/order")
    rows = data.records
    parsed = [data.raw(row) for row in rows]
    got = score(c, out, counts, rows, parsed, preds, fg["FG_predictions"], {"unit": "FG-s0 (stored predictions of task 009)"}, "FG-s0-stored")
    K.require(got["values"] == ev["values"] == fg["values"] and got["per_class_APL_top600"] == ev["per_class_APL_top600"]
              and got["per_class_APM_top600"] == ev["per_class_APM_top600"] and got["GT_counts"] == ev["GT_counts"],
              "the scoring path does not reproduce the stored seed-0 F_geo evaluation")
    return {"stage": "scorecheck", "reproduced": ["values (14 cells)", "per_class_APL_top600", "per_class_APM_top600", "GT_counts"],
            "per_class_APS_top600": got["per_class_APS_top600"], "values": got["values"],
            "note": "wiring check of the scoring code on stored predictions; no model was run; seed 0 is quoted, not re-evaluated"}


# ================================================================ summary
def ledger_rows(path, n=None):
    rows = []
    with Path(path).open() as f:
        for line in f:
            rows.append(json.loads(line))
            if n is not None and len(rows) == n:
                break
    return rows


def ledger_check(c, dirs):
    """The four formal ledgers: row count and update sequence; every row against the S15 ledger of its seed on the
    shared fields; the two ledgers of a seed against each other on the same fields."""
    out, shared = {}, {}
    for unit in R30.ORDER:
        arm, seed = R30.parse(unit)
        refs, _ = R30.reference_rows(c, seed)
        rows = ledger_rows(dirs[unit] / LEDGER)
        K.require([r["input"]["update"] for r in rows] == list(range(K.PARENT_UPDATE + 1, K.PARENT_UPDATE + 1 + len(rows)))
                  and (K.DEV or len(rows) == K.ENDPOINT - K.PARENT_UPDATE), f"{unit}: ledger update sequence")
        shared[unit] = [K.compare_shared(r, refs[r["input"]["update"]]) for r in rows]
        out[unit] = {"ledger": K.ref(dirs[unit] / LEDGER), "rows": len(rows), "rows_equal_to_the_S15_ledger": len(rows), "rows_different": 0,
                     "S15_ledger": c["units"][unit]["reference_ledger"]}
    pairs = {}
    for seed in R30.SEEDS:
        a, b = (R30.unit_name(arm, seed) for arm in R30.ARMS)
        K.require(shared[a] == shared[b], f"the ledgers of {a} and {b} differ on the shared fields")
        pairs[str(seed)] = {"units": [a, b], "rows": len(shared[a]), "rows_different": 0}
    return {"fields": list(K.SHARED_FIELDS), "units": out, "same_seed_pairs": pairs}


def class_table(tables, key):
    return [{"category_id": rows[0]["category_id"], "GT": rows[0]["GT"], **{name: r[key] for name, r in zip(tables, rows)}}
            for rows in zip(*tables.values())]


def summarize(c, out, counts):
    import r28_eval as EV
    out = EV.round_stage(out, "summary-R30")
    dirs = {u: formal_dir(c, u) for u in R30.ORDER}
    formal = {u: K.read_json(dirs[u] / "CLOSED.json") for u in R30.ORDER}
    evals = {u: R30.stage_closed(c, f"eval-{u}") for u in R30.ORDER}
    ledgers = ledger_check(c, dirs)
    for u in R30.ORDER:
        K.require(evals[u]["unit"] == u and evals[u]["formal_stage"] == dirs[u].name
                  and evals[u]["endpoint"] == K.read_json(K.side_of(evals[u]["endpoint"]["path"]))["file"], f"eval-{u} scored another endpoint")
    training = {u: {"formal_stage": dirs[u].name, "updates": formal[u]["updates"], "ledger_rows": formal[u]["ledger_rows"], "resumed": formal[u]["resumed"],
                    "input_pipeline": formal[u]["plan"]["input_pipeline"], "window_delivery": formal[u]["plan"]["window_delivery"],
                    "parent": formal[u]["plan"]["parent"], "gpu_at_start": formal[u]["gpu_at_start"], "gpu_at_end": formal[u]["gpu_at_end"],
                    "loop_wall_seconds": formal[u]["loop_wall_seconds"], "peak_cuda_bytes": formal[u]["peak_cuda_bytes"],
                    "other_formal_directories": sorted(d.name for d in K.ROUND.iterdir() if d.is_dir() and d.name.startswith(f"formal-{u}") and d != dirs[u])}
                for u in R30.ORDER}
    stages = {}
    for d in sorted(K.ROUND.iterdir()):
        if d.is_dir() and (d / "CLOSED.json").is_file():
            closed = K.read_json(d / "CLOSED.json")
            if closed.get("contract_sha256") == c["contract_sha256"]:
                stages[d.name] = {"action": closed.get("action"), "counts": closed.get("counts"), "elapsed_seconds": closed.get("elapsed_seconds")}
    base = {"ledgers": ledgers, "training": training, "stages": stages, "statements": c["statements"], "reading_rule": c["reading"],
            "demand": c["demand"], "workload_planned": c["workload"]}
    if K.DEV:
        K.require(all(e.get("dev_wiring") is True and e.get("scored") is False for e in evals.values()), "a dev round scores nothing")
        s0 = c["reference"]["values"]
        wiring = R30.summary({s: {"FG": s0["FG-s0"], "L0": s0["L0-s0"]} for s in ("0", "1", "2")})
        K.write_new(out / "RESULT.json", {**base, "dev_wiring": True, "arithmetic_on_the_quoted_seed0_values_only": wiring,
                                          "note": "dev round: nothing was scored; the arithmetic ran on the quoted seed-0 values in place of "
                                                  "every seed; not a result"})
        return {"stage": "summary-R30", "dev_wiring": True, "label": None, "result": K.ref(out / "RESULT.json")}
    K.require(all(formal[u]["updates"] + formal[u]["ledger_rows_copied_on_resume"] == K.ENDPOINT - K.PARENT_UPDATE == formal[u]["ledger_rows"]
                  and evals[u]["global_update"] == K.ENDPOINT and evals[u]["GT_counts"] == c["reference"]["GT_counts"] for u in R30.ORDER),
              "formal / evaluation receipts of the four units")
    values = {"0": {"FG": c["reference"]["values"]["FG-s0"], "L0": c["reference"]["values"]["L0-s0"]},
              **{str(s): {a: evals[R30.unit_name(a, s)]["values"] for a in R30.ARMS} for s in R30.SEEDS}}
    s = R30.summary(values)
    K.require(s["F_geo_minus_L0_pp"]["0"] == c["reference"]["seed0_F_geo_minus_L0_pp"], "seed-0 differences changed")
    classes = {}
    for seed in R30.SEEDS:
        tabs = {a: evals[R30.unit_name(a, seed)] for a in R30.ARMS}
        classes[str(seed)] = {k: class_table({a: tabs[a][f"per_class_{k}_top600"] for a in R30.ARMS}, k) for k in ("APS", "APM", "APL")}
    s0 = c["seed0"]
    classes["0"] = {"APL": class_table({"L0": s0["L0"]["per_class_APL_top600"], "FG": s0["FG"]["per_class_APL_top600"]}, "APL"),
                    "APM": {"FG": s0["FG"]["per_class_APM_top600"], "L0": None}, "APS": None, "note": s0["L0"]["per_class_note"]}
    result = {**base, "values": values, "value_sources": {"0": "quoted: task 009 eval-FG and R27-002 eval-L0 (not rerun)",
                                                           "1": "R30-002 eval-FG-s1 / eval-L0-s1", "2": "R30-002 eval-FG-s2 / eval-L0-s2"},
              **s, "per_class_top600": classes,
              "evaluations": {u: {k: evals[u][k] for k in ("endpoint", "weights_sha256", "predictions", "metrics", "cocoeval_top600_tensors")}
                              for u in R30.ORDER},
              "actual": {"training_updates": sum(formal[u]["updates"] for u in R30.ORDER),
                         "training_forwards": sum(formal[u]["hook_forwards"] for u in R30.ORDER),
                         "evaluation_forwards": sum(evals[u]["counts"]["forwards"] for u in R30.ORDER),
                         "score_budget_calls": sum(evals[u]["counts"]["score_budget_calls"] for u in R30.ORDER),
                         "per_class_COCOeval_calls": sum(evals[u]["counts"]["per_class_COCOeval_calls"] for u in R30.ORDER)}}
    K.write_new(out / "RESULT.json", result)
    d = s["F_geo_minus_L0_pp"]
    return {"stage": "summary-R30", "label": s["label"], "scale_tradeoff": s["scale_tradeoff"], "scale_tradeoff_label": s["scale_tradeoff_label"],
            "seeds_with_small_gain_repeated": s["seeds_with_small_gain_repeated"],
            **{f"seed{k}_{m}600_pp": d[k]["600"][m] for k in ("0", "1", "2") for m in ("APS", "AP75_small", "AP", "APM", "APL")},
            "result": K.ref(out / "RESULT.json")}
