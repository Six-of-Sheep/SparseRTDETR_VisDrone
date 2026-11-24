"""R32-001 endpoint reproduction, formal training, evaluation and summary of the two units W-s1, W-s2 (TASK_R32_W_SEEDS_001).

  r32_endpoint  CUDA, seed 0, not a result (task section 3, check 2): the stored e120 EMA state of W seed 0 through the
                evaluation entry of this release, 548 images; the top600 / top300 lists must be identical item by item
                to the stored lists of R28-001.  Then the scoring entry of this release on the stored lists: 14 cells
                and GT counts bitwise, per-class large-AP rows within 1e-15.  A difference stops the stage and leaves
                the largest differences on disk for the controller.  Dev round: 2 batches against the stored prefix.
  formal        one unit per process (R32_UNIT), 18,240 updates from the e060 parent of its seed.  The loop is the loop
                of R30-002 for an observer arm (r30_loop.run_loop) with the R30-002 L0 ledger of the seed as the live
                reference: every window is compared with its row before and after the update.
  evaluate      one unit per process: the e120 EMA module, view A, 548 images, top600 / top300 scored once each, one
                per-class COCOeval whose tensors are kept.  In a dev round: two batches of the short dev endpoint, no scoring.
  summarize     CPU: the two ledgers against the three reference ledgers of their seed, the 14 cells per seed and the
                reading of task section 5, the per-class tables and the training geometry.
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
import r32_control as R32
import r32_reading as RD

LEDGER, OBS, TIMING = LL.LEDGER, LP.OBS, LP.TIMING
ROW_KEYS = ("input", "gt_count", "loss", "lr", "gradient_norm_before_clip", "microbatch_losses", "update_seconds", "ema_updates", "r28")
PER_CLASS_TOLERANCE = 1e-15


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
    K.require(unit in R32.ORDER and unit in c["units"], f"the R32 round trains {R32.ORDER}: {unit!r}")
    _, seed = R32.parse(unit)
    R32.stage_closed(c, "r32-check")
    wire = R32.stage_closed(c, "wire-s0")
    endpoint = R32.stage_closed(c, "endpoint-s0")
    accept = R32.stage_closed(c, f"accept-s{seed}")
    K.require(all(wire["W_equal_to_the_R28_001_receipts"].values()) and wire["formal_path_equal_to_the_reference_path"] is True
              and endpoint.get("predictions_identical") is True and endpoint.get("scoring_reproduced") is True
              and accept.get("formal_training_allowed") is True and accept["restore_replay_equal"] is True and accept["unit"] == unit
              and accept["same_windows_as_the_reference_ledger"] is True, f"the checks do not open the formal run of unit {unit}")
    K.require(not any(k.startswith("R28_DEV_") and k != "R28_DEV_SPAN" for k in os.environ) and "R28_PIPELINE" not in os.environ,
              "dev override flags are refused in the R32 round")
    for earlier in R32.ORDER[:R32.ORDER.index(unit)]:
        formal_dir(c, earlier)                   # task section 4: W-s1 -> W-s2
    out = Path(out).resolve()
    K.require(out.parent == K.ROUND.resolve(), "formal output must be directly under the round")
    pattern = K.formal_re(unit)
    closed = sorted(d.name for d in K.ROUND.iterdir() if d.is_dir() and pattern.fullmatch(d.name) and (d / "CLOSED.json").is_file())
    K.require(not closed, f"this round already has a closed formal directory {closed}: another formal run is refused")
    if resume is None:
        K.require(out.name == f"formal-{unit}", f"a formal run without --resume writes <round>/formal-{unit}")
        return None
    path, side = R32.read_side(c, resume, "formal", unit)
    m = pattern.fullmatch(out.name)
    K.require(m is not None and m.group(1) is not None, f"a resumed formal run writes <round>/formal-{unit}-rN (N>=2)")
    K.require(path.parent.parent == K.ROUND.resolve() and pattern.fullmatch(path.parent.name) is not None and path.parent != out,
              "formal resume source must be a state in this round's formal directory of the same unit")
    K.require(side["global_update"] % K.WINDOWS_PER_EPOCH == 0 or K.DEV, "formal resume source must be an epoch boundary state")
    return path, side


def restore_files(side, src_dir, out, refs):
    """Ledger prefix (L0 rule) and the observation file prefix recorded by the resume state."""
    audit = side["audit"]
    copied, prior = LL.restore_ledger({**side, "audit": None if audit is None else audit["ledger"]}, src_dir, out, refs)
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
    _, seed = R32.parse(unit)
    obs = e.r28_observer
    history = out / LEDGER
    data = S.train_dataset(c["data"]["train_core"], S.S15_HW, seed=seed)
    start = e.completed_updates
    base = {"start": start, "recompute": S.RECOMPUTE["calls"], "bn": LL.bn_clocks(e.model), "hooks": dict(counts)}
    bn0 = base["bn"]
    K.require(start < stop and all(start < p <= stop for p in save_points), "R32 loop span / save points")
    prior_rows = dict(prior_rows)
    saves, seen, replayed, compared, seconds, checked, epochs = [], {}, 0, 0, 0.0, [], []
    check = R32.reference_check(refs, label)
    wall = time.monotonic()
    mark = time.perf_counter()
    with delivery.stream(pipeline, data, start, stop, check) as stream:
        with history.open("a", buffering=1) as ledger, (out / TIMING).open("a", buffering=1) as timing_file, \
                (out / OBS).open("a", buffering=1) as obs_file:
            for item in stream:
                x, t, inp = item.x, item.t, item.inp
                ids = [tt["image_id"] for tt in t]
                s_ = seen.setdefault(inp["epoch"], set())
                K.require(len(set(ids)) == len(ids) and not (s_ & set(ids)), "R32 occurrence: image repeated within an epoch")
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
                          "R32 update clock/input")
                row = {"input": inp, "gt_count": int(sum(len(v["labels"]) for v in t)), "loss": res["loss"],
                       "lr": res["lr"], "gradient_norm_before_clip": res["gradient_norm_before_clip"],
                       "microbatch_losses": res["microbatch_losses"], "update_seconds": res["elapsed_seconds"],
                       "ema_updates": e.ema.updates, "r28": obs.update_row(t)}
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
                if e.completed_updates % WINDOWS_PER_EPOCH == 0 or e.completed_updates == stop:
                    erow = obs.epoch_row(inp["epoch"])
                    obs_file.write(json.dumps(erow, allow_nan=False) + "\n")
                    epochs.append(erow["epoch"])
                if e.completed_updates in save_points:
                    ledger.flush()
                    os.fsync(ledger.fileno())
                    obs_file.flush()
                    os.fsync(obs_file.fileno())
                    K.PHASE["phase"] = f"{unit}:accounting before the save at {e.completed_updates}"
                    LL.account(e, counts, base)
                    checked.append(e.completed_updates)
                    LL._available(out)
                    ep, rest = divmod(e.completed_updates, WINDOWS_PER_EPOCH)
                    role = "formal_save" if not rest and ep in K.FORMAL_SAVE_EPOCHS else "recovery"
                    saving = time.perf_counter()
                    audit = {"ledger": K.ref(history), "observations": K.ref(out / OBS)}
                    saves.append(R32.save(c, e, unit, out / K.endpoint_name(e.completed_updates), "formal", audit, role))
                    moved = {**moved, "save_s": time.perf_counter() - saving}
                now = time.perf_counter()
                timing_file.write(json.dumps({"update": res["update"], "cycle_s": now - mark, "wait_s": item.timing["wait_s"],
                                              "train_call_s": train_call, "train_s": res["elapsed_seconds"],
                                              "input": {k: item.timing.get(k) for k in ("next_s", "audit_s", "fetch_s", "reference_check_s")},
                                              "prepared": stream.ready(), **moved}, separators=(",", ":")) + "\n")
                if e.completed_updates % 16 == 0:
                    print(json.dumps({"unit": unit, "update": e.completed_updates, "loss": res["loss"],
                                      "seconds": res["elapsed_seconds"], "cycle_seconds": now - mark,
                                      "g_small": row["r28"]["g_small"]}), flush=True)
                mark = now
                del item, x, t
            ledger.flush()
            os.fsync(ledger.fileno())
        pipe = stream.account()
    wall = time.monotonic() - wall
    K.PHASE["phase"] = f"{unit}:loop accounting"
    done, hooks, rec = LL.account(e, counts, base)
    K.require(e.completed_updates == stop and done == stop - start and not prior_rows, "R32 loop count")
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
    unit = R32.process_unit()
    _, seed = R32.parse(unit)
    stop, dev_points = K.formal_stop()
    save_points = dev_points or tuple(ep * K.WINDOWS_PER_EPOCH for ep in K.SAVE_EPOCHS if ep * K.WINDOWS_PER_EPOCH > K.PARENT_UPDATE)
    K.PHASE["phase"] = f"{unit}:formal guard"
    out = Path(out)
    source = guard(c, out, resume, unit)
    K.require(R32.bound_seed() == seed, f"the process is not bound to seed {seed}")
    gpu = R32.gpu_compute_processes()
    label = R32.reference_label(seed)
    refs, first = R32.reference_rows(c, seed)
    pipeline, mode = R32.formal_mode(c, unit)
    K.PHASE["phase"] = f"{unit}:construct"
    e, info = R32.create(c, seed)
    K.register_counts(e, counts)
    LP.install_observer(e)
    copied, prior_rows, saves = 0, {}, []
    if source is not None:
        side = R32.load(c, e, unit, source[0], "formal")
        K.require(side == source[1], "resume side-car changed between guard and load")
        copied, prior_rows = restore_files(side, source[0].parent, out, refs)
        K.write_new(out / "RESTORE.json", {"resumed_from": side, "prior_rows_after_state": len(prior_rows), "ledger_rows_copied": copied})
    else:
        K.require(e.completed_updates == K.PARENT_UPDATE, "a fresh formal run starts at the parent update")
        LL._available(out)
        saves.append(R32.save(c, e, unit, out / "checkpoint-epoch-060.pth", "formal", None, "formal_save"))
    start = e.completed_updates
    used = LL.points_after(save_points, start)
    K.require(stop in used, "the stop update must be a save point after the start (the endpoint state is always written)")
    delivery = RI.Delivery(mode, e.device)
    plan = {"unit": unit, "arm": R32.ARM, "seed": seed, "parent": c["units"][unit]["parent"], "start_update": start, "stop_update": stop,
            "updates": stop - start, "schedule_first_update": K.PARENT_UPDATE + 1,
            "save_points": {"planned": sorted(set(save_points)), "used_in_this_process": list(used),
                            "at_or_before_the_start": sorted(p for p in set(save_points) if p <= start)},
            "formal_save_epochs": list(K.FORMAL_SAVE_EPOCHS), "scope": "formal", "input_pipeline": pipeline,
            "window_delivery": delivery.binding(), "reference_ledger": c["units"][unit]["reference_ledger"],
            "demand": e.r28_ctx.binding()}
    loop = run_loop(c, e, unit, out, stop, counts, save_points=set(used), refs=refs, label=label, prior_rows=prior_rows, first=first,
                    pipeline=pipeline, delivery=delivery)
    delivery.close()
    K.require(e.r28_ctx.binding() == c["demand"]["binding"], "the demand source changed during the formal run")
    with (out / LEDGER).open() as f:
        head = json.loads(f.readline())
        rows = 1 + sum(1 for _ in f)
    K.require(rows == stop - K.PARENT_UPDATE == copied + loop["shared_rows_compared_live"] and head["input"]["update"] == first["update"]
              and set(head) == set(ROW_KEYS), "R32 ledger row count / row keys")
    loop["saves"] = saves + loop["saves"]
    return {"unit": unit, "arm": R32.ARM, "seed": seed, "plan": plan, "install": {k: v for k, v in info.items() if k != "new_parameter_names"},
            "resumed": source is not None, "ledger_rows": rows, "ledger_rows_copied_on_resume": copied,
            "shared_rows_compared": copied + loop["shared_rows_compared_live"], "shared_rows_different": 0,
            "shared_fields": list(K.SHARED_FIELDS), "reference_ledger": c["units"][unit]["reference_ledger"],
            "first_update_loss": {"update": first["update"], unit: head["loss"], first["reference"]: first["loss"]},
            "gpu_at_start": gpu, "gpu_at_end": R32.gpu_compute_processes(), "dev_span": os.environ.get("R28_DEV_SPAN"), **loop}


# ================================================================ evaluation
def score(c, out, counts, rows, parsed, preds, refs, binding, prefix):
    """top600 / top300 scored once each with the certified scorer, one per-class COCOeval on top600 (tensors kept)."""
    import r30_loop as L30
    from infer_seed import bridge_values
    from sparse_rtdetr.r0.evaluation import score_budget
    values, metrics_refs, counts_gt, scoring, classes = {}, {}, {}, [], None
    for b in K.BUDGETS:
        t = time.perf_counter()
        metrics = score_budget(rows, parsed, preds[b], c["source_root"])
        counts.update(score_budget_calls=1)
        metrics["R32_binding"] = {"contract_sha256": c["contract_sha256"], "kind": "EMA", "view": "A", "budget": b,
                                  "predictions": refs[str(b)], **binding}
        values[str(b)] = bridge_values(metrics)
        counts_gt[str(b)] = {k: metrics["spatial_filter_counts"][k] for k in ("GT_before", "GT_after")}
        p = out / f"{prefix}-EMA-A-top{b}-metrics.json"
        K.write_new(p, metrics)
        metrics_refs[str(b)] = K.ref(p)
        if b == 600:
            classes = L30.per_class(rows, parsed, preds[b], metrics["COCO_bridge"], out / "cocoeval-top600.npz")
            counts.update(per_class_COCOeval_calls=1)
        scoring.append({"budget": b, "seconds": time.perf_counter() - t})
    K.require(counts["score_budget_calls"] == 2 and counts["per_class_COCOeval_calls"] == 1, "scorer call counts")
    K.require(all(sorted(values[b]) == sorted(RD.METRICS) for b in RD.BUDGET_KEYS), "metric cells of the scorer")
    return {"values": values, "metrics": metrics_refs, "GT_counts": counts_gt, "per_class_APS_top600": classes["APS"],
            "per_class_APM_top600": classes["APM"], "per_class_APL_top600": classes["APL"],
            "cocoeval_top600_tensors": K.ref(out / "cocoeval-top600.npz"), "scoring": scoring}


def prediction_differences(mine, stored):
    """Two prediction lists of equal length: where they differ and by how much."""
    import numpy as np
    K.require(len(mine) == len(stored), f"prediction record count differs: {len(mine)} / {len(stored)}")
    keys = [all(a[k] == b[k] for k in ("image_id", "query_id", "category_id")) for a, b in zip(mine, stored)]
    s = np.abs(np.array([a["score"] for a in mine], dtype=np.float64) - np.array([b["score"] for b in stored], dtype=np.float64))
    bm = np.array([a["bbox_xyxy"] for a in mine], dtype=np.float64)
    bs = np.array([b["bbox_xyxy"] for b in stored], dtype=np.float64)
    box = np.abs(bm - bs)
    different = [i for i, (a, b) in enumerate(zip(mine, stored)) if a != b]
    return {"records": len(mine), "records_different": len(different), "first_different_record": different[0] if different else None,
            "identity_and_order_equal": all(keys), "records_with_another_identity": len(keys) - sum(keys),
            "record_keys_equal": all(sorted(a) == sorted(b) for a, b in zip(mine[:1], stored[:1])),
            "score_max_abs_difference": float(s.max()), "records_with_another_score": int((s > 0).sum()),
            "bbox_max_abs_difference_px": float(box.max()), "records_with_another_box": int((box.max(1) > 0).sum()),
            "bbox_max_relative_difference": float((box / np.maximum(1.0, np.abs(bs))).max())}


def cell_differences(mine, stored):
    """Two 14-cell tables: bitwise equality and the largest absolute difference (fractions, not pp)."""
    gaps = {f"top{b}_{m}": abs(mine[b][m] - stored[b][m]) for b in RD.BUDGET_KEYS for m in RD.METRICS}
    worst = max(gaps, key=gaps.get)
    return {"bitwise_equal": mine == stored, "cells_different": sorted(k for k, v in gaps.items() if v != 0),
            "max_abs_difference": gaps[worst], "max_abs_difference_cell": worst}


def class_differences(mine, stored, key):
    K.require([(r["category_id"], r["GT"], r[key] is None) for r in mine] == [(r["category_id"], r["GT"], r[key] is None) for r in stored],
              f"per-class {key} rows: categories / GT counts")
    gap = max(abs(a[key] - b[key]) for a, b in zip(mine, stored) if a[key] is not None)
    return {"bitwise_equal": mine == stored, "max_abs_difference": gap, "within_tolerance": gap <= PER_CLASS_TOLERANCE}


def r32_endpoint(c, out, counts):
    """Seed 0, not a result (task section 3, check 2)."""
    import r28_eval as EV
    from infer_seed import _group, _prefix
    from sparse_rtdetr.protocol_r18 import checkpoint as ck
    from sparse_rtdetr.protocol_r18.engine import tensor_digest
    from sparse_rtdetr.research import r28_resample as R
    torch = K.torch_mod()
    K.PHASE["phase"] = "endpoint-s0/guards"
    out = EV.round_stage(out, "endpoint-s0")
    R32.stage_closed(c, "r32-check")
    R32.stage_closed(c, "wire-s0")
    K.require(R32.bound_seed() == 0, "endpoint-s0 runs in a process bound to seed 0")
    s0 = c["seed0"]
    c1 = R30._digest_checked(K.pinned(s0["contract"]), s0["contract_sha256"])
    side, s = K.read_state(c1, K.pinned(s0["endpoint"]), "formal", R32.ARM)
    state = s["ema"]["module"]
    del s
    weights = tensor_digest(state.items())
    K.require(side["global_update"] == K.ENDPOINT and side["file"] == s0["endpoint"] and side["demand"] == c["demand"]["binding"]
              and weights == s0["weights_sha256"], "the W seed-0 endpoint is not the state R28-001 evaluated")
    K.require(all(not v.is_floating_point() or v.dtype == torch.float32 for v in state.values()), "EMA not FP32")
    device = torch.device("cuda:0")
    source = R32.demand_source(c)
    model = R.build_eval_model(state, device, R32.ARM, source, seed=0)
    K.require(R.is_wrapped(model) and not model.training and model.r28_arm == R32.ARM and model.r28_ctx.binding() == side["demand"],
              "evaluation model is not an arm W eval model")
    rng = ck.capture_rng(device)
    K.PHASE["phase"] = "endpoint-s0/inference"
    limit = (1, 2) if K.DEV else None
    data, preds, timing = EV.infer(c, model, counts, limit=limit)
    forwards = K.EVAL_WARMUP + K.EVAL_BATCHES if limit is None else sum(limit)
    K.require(counts["forwards"] == forwards and (limit is not None or forwards == 141), "endpoint reproduction forward count")
    K.require(tensor_digest(model.state_dict().items()) == weights and ck.exact_equal(ck.capture_rng(device), rng)
              and model.r28_ctx.binding() == side["demand"], "evaluation changed model / RNG / demand source")
    del model
    gc.collect()
    torch.cuda.empty_cache()

    K.PHASE["phase"] = "endpoint-s0/predictions"
    stored = {b: K.read_json(K.pinned(s0[f"predictions_{b}"])) for b in K.BUDGETS}
    K.require(all(len(stored[b]) == 548 * b for b in K.BUDGETS), "stored prediction record counts")
    compared, new_refs = {}, {}
    for b in K.BUDGETS:
        part = stored[b][:len(preds[b])]
        same = preds[b] == part
        compared[str(b)] = {"records": len(preds[b]), "stored_records": len(stored[b]), "identical_item_by_item": same,
                            "stored": s0[f"predictions_{b}"], "differences": None if same else prediction_differences(preds[b], part)}
        if limit is None:
            p = out / f"top{b}-predictions.json"
            K.write_new(p, preds[b])
            new_refs[str(b)] = K.ref(p)
            compared[str(b)]["file"] = new_refs[str(b)]
            compared[str(b)]["file_sha256_equal_to_the_stored_file"] = new_refs[str(b)]["sha256"] == s0[f"predictions_{b}"]["sha256"]
    identical = all(v["identical_item_by_item"] for v in compared.values())
    head = {"schema": "r32-seed0-endpoint-v1", "contract_sha256": c["contract_sha256"], "seed": 0, "endpoint": side["file"],
            "weights_sha256": weights, "demand": side["demand"], "kind": "EMA", "view": "A", "global_update": side["global_update"],
            "batches": [K.EVAL_WARMUP, K.EVAL_BATCHES] if limit is None else list(limit), "forwards": forwards, **timing,
            "predictions": compared, "predictions_identical": identical,
            "result_status": "not a result: the W seed-0 values quoted in the reading are the stored ones of R28-001"}
    if not identical:
        K.write_new(out / "ENDPOINT_DIFFERENCE.json", {**head, "scoring": None,
                                                        "hand_back": "task section 3, check 2: the lists differ; stop and hand back to the controller"})
        K.require(False, "the evaluation entry does not reproduce the stored W seed-0 prediction lists: "
                         + json.dumps({b: v["differences"] for b, v in compared.items()}))
    del preds

    K.PHASE["phase"] = "endpoint-s0/scoring of the stored lists"
    ids = c["development_ids"]
    grouped = {b: _group(stored[b], ids, b, whole=True, exact=True) for b in K.BUDGETS}
    _prefix(grouped[600], grouped[300], ids, whole=True)
    del grouped
    rows = data.records
    parsed = [data.raw(row) for row in rows]
    refs = {str(b): s0[f"predictions_{b}"] for b in K.BUDGETS}
    got = score(c, out, counts, rows, parsed, stored, refs,
                {"unit": "W-s0 (stored prediction lists of R28-001)", "arm": R32.ARM, "seed": 0, "checkpoint": side["file"], "weights_sha256": weights},
                "W-s0-stored")
    cells = cell_differences(got["values"], s0["values"])
    gt_equal = got["GT_counts"] == s0["GT_counts"] == c["reference"]["GT_counts"]
    apl = class_differences(got["per_class_APL_top600"], s0["per_class_APL_top600"], "APL")
    bridges = {}
    for b in RD.BUDGET_KEYS:
        mine, theirs = K.read_json(got["metrics"][b]["path"]), K.read_json(K.pinned(s0[f"metrics_{b}"]))
        bridges[b] = {"COCO_bridge_equal": mine["COCO_bridge"] == theirs["COCO_bridge"],
                      "spatial_filter_counts_equal": mine["spatial_filter_counts"] == theirs["spatial_filter_counts"]}
    scoring = {"scored": "the stored prediction lists of R28-001 (identical to the lists of this stage)", "values": got["values"],
               "cells_against_eval_W": cells, "GT_counts": got["GT_counts"], "GT_counts_equal": gt_equal,
               "per_class_APL_against_eval_W": apl, "per_class_tolerance": PER_CLASS_TOLERANCE, "metric_files": got["metrics"],
               "metric_files_against_the_stored_ones": bridges,
               "not_comparable": ["per-class APS and APM rows of W seed 0 (R28-001 kept the per-class APL rows only and no COCOeval tensors)"],
               "by_product_not_a_result": {"per_class_APS_top600": got["per_class_APS_top600"], "per_class_APM_top600": got["per_class_APM_top600"],
                                           "cocoeval_top600_tensors": got["cocoeval_top600_tensors"]},
               "seconds": got["scoring"]}
    reproduced = cells["bitwise_equal"] and gt_equal and apl["within_tolerance"]
    body = {**head, "scoring": scoring, "scoring_reproduced": reproduced}
    if not reproduced:
        K.write_new(out / "ENDPOINT_DIFFERENCE.json", {**body, "hand_back": "task section 3, check 2: the scoring entry does not reproduce the stored "
                                                                           "cells; stop and hand back to the controller"})
        K.require(False, "the scoring entry does not reproduce summary-W: " + json.dumps({"cells": cells, "GT_counts_equal": gt_equal, "per_class_APL": apl}))
    K.write_new(out / "ENDPOINT_REPRODUCTION.json", body)
    return {"stage": "endpoint-s0", "seed": 0, "workload": c["workload"]["endpoint-s0"], "dev_prefix_only": limit is not None,
            "predictions_identical": True, "prediction_records_compared": {b: v["records"] for b, v in compared.items()},
            "prediction_files_sha256_equal": {b: v.get("file_sha256_equal_to_the_stored_file") for b, v in compared.items()},
            "scoring_reproduced": True, "cells_bitwise_equal": True, "GT_counts": got["GT_counts"],
            "per_class_APL_bitwise_equal": apl["bitwise_equal"], "per_class_APL_max_abs_difference": apl["max_abs_difference"],
            "metric_files_against_the_stored_ones": bridges, "not_comparable": scoring["not_comparable"],
            "reproduction": K.ref(out / "ENDPOINT_REPRODUCTION.json"), "endpoint": side["file"], "weights_sha256": weights, **timing,
            "result_status": head["result_status"]}


def endpoint_state(c, unit):
    stop = K.formal_stop()[0]
    side, s = R32.read_state(c, formal_dir(c, unit) / K.endpoint_name(stop), "formal", unit)
    K.require(side["global_update"] == stop and s["engine"]["completed_updates"] == s["ema"]["updates"] == stop
              and (K.DEV or stop == K.ENDPOINT), "endpoint clocks")
    return side, s["ema"]["module"]


def evaluate(c, out, counts):
    import r28_eval as EV
    from infer_seed import _group, _prefix
    from sparse_rtdetr.protocol_r18 import checkpoint as ck
    from sparse_rtdetr.protocol_r18.engine import tensor_digest
    from sparse_rtdetr.research import r28_resample as R
    torch = K.torch_mod()
    unit = R32.process_unit()
    _, seed = R32.parse(unit)
    K.PHASE["phase"] = "evaluate/guards"
    out = EV.round_stage(out, f"eval-{unit}")
    formal_name = formal_dir(c, unit).name
    side, state = endpoint_state(c, unit)
    K.require(side["unit"] == unit and side["scope"] == "formal" and side["seed"] == seed, "endpoint side-car")
    K.require(all(not v.is_floating_point() or v.dtype == torch.float32 for v in state.values()), "EMA not FP32")
    device = torch.device("cuda:0")
    source = R32.demand_source(c)
    K.require(side["demand"] == c["demand"]["binding"] and side["demand"]["file"] == source["file"], "the endpoint was trained with another demand source")
    model = R.build_eval_model(state, device, R32.ARM, source, seed=seed)
    K.require(R.is_wrapped(model) and not model.training and model.r28_arm == R32.ARM and model.r28_ctx.binding() == side["demand"],
              "evaluation model is not an arm W eval model")
    weights = tensor_digest(state.items())
    rng = ck.capture_rng(device)
    K.PHASE["phase"] = "evaluate/inference"
    limit = (1, 2) if K.DEV else None
    data, preds, timing = EV.infer(c, model, counts, limit=limit)
    forwards = K.EVAL_WARMUP + K.EVAL_BATCHES if limit is None else sum(limit)
    K.require(counts["forwards"] == forwards and (limit is not None or forwards == 141), "evaluation forward count")
    K.require(tensor_digest(model.state_dict().items()) == weights and ck.exact_equal(ck.capture_rng(device), rng)
              and model.r28_ctx.binding() == side["demand"], "evaluation changed model / RNG / demand source")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    head = {"unit": unit, "arm": R32.ARM, "seed": seed, "endpoint": side["file"], "formal_stage": formal_name, "weights_sha256": weights, "kind": "EMA",
            "epoch": side["global_update"] // K.WINDOWS_PER_EPOCH, "global_update": side["global_update"], "view": "A", "demand": side["demand"], **timing}
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
    binding = {"unit": unit, "arm": R32.ARM, "seed": seed, "epoch": head["epoch"], "checkpoint": side["file"], "weights_sha256": weights,
               "demand": side["demand"]}
    got = score(c, out, counts, rows, parsed, preds, refs, binding, unit)
    K.require(got["GT_counts"] == c["reference"]["GT_counts"], f"scored GT set differs from the seed-0 evaluations: {got['GT_counts']}")
    return {**head, "predictions": refs, **got}


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
    """The two formal ledgers: row count and update sequence; every row against each of the three reference ledgers
    of its seed (R30-002 L0, R30-002 F_geo, S15 replication) on the shared fields.  Returns the comparison and the
    ledger rows (for the training geometry)."""
    out, kept = {}, {}
    for unit in R32.ORDER:
        _, seed = R32.parse(unit)
        rows = ledger_rows(dirs[unit] / LEDGER)
        K.require([r["input"]["update"] for r in rows] == list(range(K.PARENT_UPDATE + 1, K.PARENT_UPDATE + 1 + len(rows)))
                  and (K.DEV or len(rows) == K.ENDPOINT - K.PARENT_UPDATE) and all(set(r) == set(ROW_KEYS) for r in rows), f"{unit}: ledger update sequence")
        refs = {"R30-002 L0": c["units"][unit]["reference_ledger"], **c["units"][unit]["equal_ledgers"]}
        against = {}
        for name, ref in refs.items():
            other = {r["input"]["update"]: r for r in R32._ledger(K.pinned(ref))}
            for r in rows:
                K.compare_shared(r, other[r["input"]["update"]])
                bad = LL.input_differences(r["input"], other[r["input"]["update"]]["input"])
                K.require(not bad, f"{unit}: input record differs from the {name} ledger at update {r['input']['update']}: {bad}")
            against[name] = {"file": ref, "rows_compared": len(rows), "rows_equal_on_the_shared_fields": len(rows), "rows_different": 0}
            del other
        out[unit] = {"ledger": K.ref(dirs[unit] / LEDGER), "rows": len(rows), "against": against}
        kept[unit] = rows
    return {"fields": list(K.SHARED_FIELDS), "units": out}, kept


def geometry_table(c, kept):
    """Small-GT magnification during training: the two new units against the W seed-0 ledger over the same updates."""
    n = min(len(r) for r in kept.values())
    _, seed0 = R32.w0_geometry(c, None if n == K.ENDPOINT - K.PARENT_UPDATE else n)
    table = {"W-s0": seed0, **{u: RD.geometry((r["input"]["epoch"], r["r28"]) for r in rows) for u, rows in kept.items()}}
    against = {}
    for u in kept:
        against[u] = {}
        for span in table[u]:
            a, b = table[u][span], seed0[span]
            against[u][span] = {k: None if a[k] is None or b[k] is None else a[k] - b[k] for k in ("g_small_geometric_mean", "small_share_with_g_below_1")}
    return {"units": table, "difference_from_W_s0": against,
            "source": {"W-s0": c["seed0"]["ledger"], "note": "ledger key r28: per update the geometric-mean magnification of the small GT of the "
                                                             "window, their number and the share with g < 1; pooled over a span with the "
                                                             "number of small GT as weight"}}


def class_tables(c, evals):
    """Per-class top600 rows of the three arms.  Seeds 1 / 2: L0 and F_geo quoted from R30-002, W from this round.
    Seed 0: the large-AP rows only (R28-001 kept no other per-class reading of W)."""
    tables, apl = {}, {}
    quoted = c["R30"]["per_class_APL_top600"]
    for s in RD.SEED_KEYS:
        w = c["seed0"]["per_class_APL_top600"] if s == "0" else evals[R32.unit_name(int(s))]["per_class_APL_top600"]
        K.require([(r["category_id"], r["GT"]) for r in w] == [(r["category_id"], r["GT"]) for r in quoted[s]], f"seed {s}: per-class large GT counts")
        apl[s] = {"W": w, **{a: [{"category_id": r["category_id"], "GT": r["GT"], "APL": r[a]} for r in quoted[s]] for a in ("L0", "FG")}}
        tables[s] = {"APL": [{**q, "W": r["APL"]} for q, r in zip(quoted[s], w)]}
    for s in ("1", "2"):
        ev = evals[R32.unit_name(int(s))]
        for key in ("APS", "APM"):
            q, w = c["R30"]["per_class_APS_APM_top600"][s][key], ev[f"per_class_{key}_top600"]
            K.require([(r["category_id"], r["GT"]) for r in w] == [(r["category_id"], r["GT"]) for r in q], f"seed {s}: per-class {key} GT counts")
            tables[s][key] = [{**a, "W": b[key]} for a, b in zip(q, w)]
    tables["0"].update(APS=None, APM=None, note=c["seed0"]["per_class_note"])
    return tables, apl


def summarize(c, out, counts):
    import r28_eval as EV
    out = EV.round_stage(out, "summary-R32")
    dirs = {u: formal_dir(c, u) for u in R32.ORDER}
    formal_ = {u: K.read_json(dirs[u] / "CLOSED.json") for u in R32.ORDER}
    evals = {u: R32.stage_closed(c, f"eval-{u}") for u in R32.ORDER}
    checks = {n: R32.stage_closed(c, n) for n in ("r32-check", "wire-s0", "endpoint-s0", "accept-s1", "accept-s2")}
    ledgers, kept = ledger_check(c, dirs)
    for u in R32.ORDER:
        K.require(evals[u]["unit"] == u and evals[u]["formal_stage"] == dirs[u].name
                  and evals[u]["endpoint"] == K.read_json(K.side_of(evals[u]["endpoint"]["path"]))["file"], f"eval-{u} scored another endpoint")
    training = {u: {"formal_stage": dirs[u].name, "updates": formal_[u]["updates"], "ledger_rows": formal_[u]["ledger_rows"], "resumed": formal_[u]["resumed"],
                    "ledger_rows_copied_on_resume": formal_[u]["ledger_rows_copied_on_resume"],
                    "input_pipeline": formal_[u]["plan"]["input_pipeline"], "window_delivery": formal_[u]["plan"]["window_delivery"],
                    "parent": formal_[u]["plan"]["parent"], "demand": formal_[u]["plan"]["demand"],
                    "gpu_at_start": formal_[u]["gpu_at_start"], "gpu_at_end": formal_[u]["gpu_at_end"],
                    "loop_wall_seconds": formal_[u]["loop_wall_seconds"], "peak_cuda_bytes": formal_[u]["peak_cuda_bytes"],
                    "saves": len(formal_[u]["saves"]),
                    "other_formal_directories": sorted(d.name for d in K.ROUND.iterdir() if d.is_dir() and d.name.startswith(f"formal-{u}") and d != dirs[u])}
                for u in R32.ORDER}
    stages = {}
    for d in sorted(K.ROUND.iterdir()):
        if d.is_dir() and (d / "CLOSED.json").is_file():
            closed = K.read_json(d / "CLOSED.json")
            if closed.get("contract_sha256") == c["contract_sha256"]:
                stages[d.name] = {"action": closed.get("action"), "counts": closed.get("counts"), "elapsed_seconds": closed.get("elapsed_seconds")}
    geometry = geometry_table(c, kept)
    del kept
    base = {"ledgers": ledgers, "training": training, "training_geometry": geometry, "stages": stages, "statements": c["statements"],
            "reading_rule": c["reading"], "demand": c["demand"], "workload_planned": c["workload"],
            "checks": {"wire-s0": {k: checks["wire-s0"][k] for k in ("W_equal_to_the_R28_001_receipts", "formal_path_equal_to_the_reference_path",
                                                                    "W_not_comparable", "state_after_update_1_compared", "equivalence")},
                       "endpoint-s0": {k: checks["endpoint-s0"][k] for k in ("predictions_identical", "prediction_records_compared",
                                                                            "prediction_files_sha256_equal", "scoring_reproduced",
                                                                            "per_class_APL_bitwise_equal", "per_class_APL_max_abs_difference",
                                                                            "not_comparable", "dev_prefix_only", "reproduction")},
                       **{n: {k: checks[n][k] for k in ("restore_replay_equal", "window_delivery", "same_windows_as_the_reference_ledger",
                                                        "formal_training_allowed", "equivalence")} for n in ("accept-s1", "accept-s2")}},
            "storage": {"plan": c["resource_plan"], "round_bytes_at_summary": R32._tree_bytes(K.ROUND)}}
    if K.DEV:
        K.require(all(e.get("dev_wiring") is True and e.get("scored") is False for e in evals.values()), "a dev round scores nothing")
        K.write_new(out / "RESULT.json", {**base, "dev_wiring": True, "arithmetic_on_the_quoted_seed0_values_only": R32.reading_self_check(c),
                                          "note": "dev round: nothing of the new units was scored; the arithmetic ran on the quoted seed-0 "
                                                  "values in place of every seed; not a result"})
        return {"stage": "summary-R32", "dev_wiring": True, "reading_A": None, "reading_B": None, "reading_C": None,
                "result": K.ref(out / "RESULT.json")}
    K.require(all(formal_[u]["updates"] + formal_[u]["ledger_rows_copied_on_resume"] == K.ENDPOINT - K.PARENT_UPDATE == formal_[u]["ledger_rows"]
                  and evals[u]["global_update"] == K.ENDPOINT and evals[u]["GT_counts"] == c["reference"]["GT_counts"]
                  and evals[u]["counts"]["forwards"] == K.EVAL_WARMUP + K.EVAL_BATCHES and evals[u]["counts"]["score_budget_calls"] == 2
                  and evals[u]["counts"]["per_class_COCOeval_calls"] == 1 for u in R32.ORDER), "formal / evaluation receipts of the two units")
    quoted = c["R30"]["values"]
    values = {"0": {"W": c["seed0"]["values"], "L0": quoted["0"]["L0"], "FG": quoted["0"]["FG"]},
              **{str(s): {"W": evals[R32.unit_name(s)]["values"], "L0": quoted[str(s)]["L0"], "FG": quoted[str(s)]["FG"]} for s in R32.SEEDS}}
    tables, apl = class_tables(c, evals)
    reading = RD.reading(values, c["S15"]["values_top600"], apl)
    K.require(reading["W_minus_L0_pp"]["0"] == c["reference"]["seed0_W_minus_L0_pp"]
              and reading["W_minus_F_geo_pp"]["0"] == c["reference"]["seed0_W_minus_F_geo_pp"], "seed-0 differences changed")
    result = {**base, "values": values,
              "value_sources": {"W": {"0": "quoted: R28-001 eval-W (not rerun)", "1": "R32-001 eval-W-s1", "2": "R32-001 eval-W-s2"},
                                "L0 and FG": "quoted: R30-002 summary-R30 (not rerun)", "S15": "quoted: the top600 metric files of its three seeds"},
              "S15_top600": c["S15"]["values_top600"], "endpoints": RD.endpoint_summary(values), **reading, "per_class_top600": tables,
              "evaluations": {u: {k: evals[u][k] for k in ("endpoint", "weights_sha256", "predictions", "metrics", "cocoeval_top600_tensors")}
                              for u in R32.ORDER},
              "actual": {"training_updates": sum(formal_[u]["updates"] for u in R32.ORDER),
                         "training_forwards": sum(formal_[u]["hook_forwards"] for u in R32.ORDER),
                         "evaluation_forwards": sum(evals[u]["counts"]["forwards"] for u in R32.ORDER),
                         "score_budget_calls": sum(evals[u]["counts"]["score_budget_calls"] for u in R32.ORDER),
                         "per_class_COCOeval_calls": sum(evals[u]["counts"]["per_class_COCOeval_calls"] for u in R32.ORDER),
                         "check_updates": sum(checks[n]["counts"]["actual_updates"] for n in checks),
                         "check_forwards": sum(checks[n]["counts"]["forwards"] for n in checks),
                         "check_score_budget_calls": checks["endpoint-s0"]["counts"].get("score_budget_calls", 0),
                         "check_per_class_COCOeval_calls": checks["endpoint-s0"]["counts"].get("per_class_COCOeval_calls", 0)}}
    K.write_new(out / "RESULT.json", result)
    d, f = reading["W_minus_L0_pp"], reading["W_minus_F_geo_pp"]
    return {"stage": "summary-R32", "reading_A": reading["reading_A"]["label"], "reading_B": reading["reading_B"]["label"],
            "reading_C": reading["reading_C"]["label"], "seeds_satisfying_A": reading["reading_A"]["seeds_satisfied"],
            "readings_B": reading["reading_B"]["readings"], "W_unique_advantage": reading["W_unique_advantage"]["status"],
            **{f"seed{k}_W_minus_L0_{m}600_pp": d[k]["600"][m] for k in RD.SEED_KEYS for m in ("APS", "AP75_small", "AP", "APM", "APL")},
            **{f"seed{k}_W_minus_FG_{m}600_pp": f[k]["600"][m] for k in RD.SEED_KEYS for m in ("APS", "AP75_small", "AP")},
            "result": K.ref(out / "RESULT.json")}
