"""R27-002 L0 control: the formal training loop (e061-e120 from the native e060 parent, no local branch).

The loop is R27 ``run_loop`` without the r27 context, route keys, local audit and monitor.  The ledger row keys are
the R27 ones.  Every update is compared, on the declared shared fields, with the same row of the pinned R27
formal-P ledger (P and C are first required to agree with each other); a difference raises immediately.
``formal`` is the only entry and always runs the fixed schedule; ``train`` takes the stop update and the save
points as arguments so that a dev shim can drive the same code over a short span.
"""
import json
import os
import shutil
import time
from pathlib import Path

import l0_common as K

FORMAL_POINTS = tuple(ep * K.WINDOWS_PER_EPOCH for ep in K.SAVE_EPOCHS if ep * K.WINDOWS_PER_EPOCH > K.PARENT_UPDATE)
LEDGER = "UPDATES.jsonl"
REPLAY_SKIP = ("update_seconds",)


def _available(path, need=2 * 1024 ** 3):
    free = shutil.disk_usage(path).free
    K.require(free > need, f"insufficient disk for the next save: {free} bytes")


def reference_rows(c):
    """Pinned R27 P/C ledgers: 18,240 rows each, equal to each other on the shared fields.  Returns P rows by update."""
    rows = {}
    for arm in ("P", "C"):
        with K.pinned(c["R27_updates"][arm]).open() as f:
            rows[arm] = [{k: v for k, v in json.loads(line).items() if k in ("input", "loss") + K.TOP_FIELDS} for line in f]
        K.require(len(rows[arm]) == K.ENDPOINT - K.PARENT_UPDATE, f"R27 {arm} ledger rows: {len(rows[arm])}")
    for i, (p, q) in enumerate(zip(rows["P"], rows["C"])):
        K.require(p["input"]["update"] == K.PARENT_UPDATE + 1 + i, "R27 P ledger update sequence")
        K.compare_shared(p, q)
    first = K.PARENT_UPDATE + 1
    return {r["input"]["update"]: r for r in rows["P"]}, {"update": first, "P": rows["P"][0]["loss"], "C": rows["C"][0]["loss"]}


def guard(c, out, resume):
    for stage in ("cpu", "zero", "accept"):
        K.stage_closed(c, stage)
    out = Path(out).resolve()
    K.require(out.parent == K.ROUND.resolve(), "formal output must be directly under the round")
    if resume is None:
        K.require(out.name == "formal-L0", "a formal run without --resume writes <round>/formal-L0")
        return None
    path, side = K.read_side(c, resume, "formal")
    m = K.FORMAL_DIR.fullmatch(out.name)
    K.require(m is not None and m.group(1) is not None, "a resumed formal run writes <round>/formal-L0-rN (N>=2)")
    K.require(path.parent.parent == K.ROUND.resolve() and K.FORMAL_DIR.fullmatch(path.parent.name) is not None
              and path.parent != out, "formal resume source must be a state in this round's formal-L0* directory")
    return path, side


def restore_ledger(side, src_dir, out, refs):
    """Copy the prior ledger up to the byte count recorded by the state; later complete rows become prior_rows."""
    audit = side["audit"]
    src = src_dir / LEDGER
    if audit is None:
        K.require(side["global_update"] == K.PARENT_UPDATE, "only the e060 state has no ledger binding")
        size, digest = 0, K.sha_bytes(b"")
    else:
        K.require(Path(audit["path"]) == src, "state ledger binding is not the ledger next to the state")
        size, digest = audit["bytes"], audit["sha256"]
    head = b""
    if size:
        with src.open("rb") as f:
            head = f.read(size)
    K.require(len(head) == size and K.sha_bytes(head) == digest, "prior ledger prefix differs from the state binding")
    with (out / LEDGER).open("xb") as dst:
        dst.write(head)
        dst.flush()
        os.fsync(dst.fileno())
    copied = [json.loads(line) for line in head.splitlines()]
    K.require([r["input"]["update"] for r in copied] == list(range(K.PARENT_UPDATE + 1, side["global_update"] + 1)),
              "prior ledger prefix is not the update sequence up to the state")
    for r in copied:
        K.compare_shared(r, refs[r["input"]["update"]])
    prior = {}
    if src.is_file():
        with src.open("rb") as f:
            f.seek(size)
            for line in f:
                if not line.endswith(b"\n"):
                    break
                row = json.loads(line)
                prior[row["input"]["update"]] = {k: v for k, v in row.items() if k not in REPLAY_SKIP}
    return len(copied), prior


def bn_clocks(model):
    torch = K.torch_mod()
    return {n: (int(m.num_batches_tracked), bool(m.training)) for n, m in model.named_modules()
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm) and m.track_running_stats}


def run_loop(c, e, out, stop, counts, *, save_points, refs, prior_rows):
    from formal_data import windows, WINDOWS_PER_EPOCH
    from sparse_rtdetr.research import scale_baseline as S
    history = out / LEDGER
    data = S.train_dataset(c["data"]["train_core"], S.S15_HW, seed=0)
    start, rec0, bn0, hook0 = e.completed_updates, S.RECOMPUTE["calls"], bn_clocks(e.model), dict(counts)
    K.require(start < stop and all(start < p <= stop for p in save_points), "L0 loop span / save points")
    prior_rows = dict(prior_rows)
    saves, seen, replayed, compared, seconds = [], {}, 0, 0, 0.0
    wall = time.monotonic()
    with history.open("a", buffering=1) as ledger:
        while e.completed_updates < stop:
            begin = e.completed_updates
            for x, t, inp in windows(data, begin, limit=min(stop - begin, WINDOWS_PER_EPOCH - begin % WINDOWS_PER_EPOCH)):
                ids = [tt["image_id"] for tt in t]
                s_ = seen.setdefault(inp["epoch"], set())
                K.require(len(set(ids)) == len(ids) and not (s_ & set(ids)), "L0 occurrence: image repeated within an epoch")
                s_.update(ids)
                K.PHASE["phase"] = f"L0:update {e.completed_updates + 1}"
                res = e.train_window(x, t)
                K.require(res["update"] == inp["update"] and res["finite"] and e.ema.updates == e.completed_updates,
                          "L0 update clock/input")
                row = {"input": inp, "gt_count": int(sum(len(v["labels"]) for v in t)), "loss": res["loss"],
                       "lr": res["lr"], "gradient_norm_before_clip": res["gradient_norm_before_clip"],
                       "microbatch_losses": res["microbatch_losses"], "update_seconds": res["elapsed_seconds"],
                       "ema_updates": e.ema.updates}
                K.compare_shared(row, refs[res["update"]])  # scientific identity guard: STOP on the first difference
                compared += 1
                if res["update"] in prior_rows:
                    K.require(prior_rows.pop(res["update"]) == {k: v for k, v in row.items() if k not in REPLAY_SKIP},
                              f"L0 recovery replay differs from the recorded trajectory at update {res['update']}")
                    replayed += 1
                ledger.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                seconds += res["elapsed_seconds"]
                if e.completed_updates in save_points:
                    ledger.flush()
                    os.fsync(ledger.fileno())
                    _available(out)
                    ep, rest = divmod(e.completed_updates, WINDOWS_PER_EPOCH)
                    name = f"checkpoint-update-{e.completed_updates}.pth" if rest else f"checkpoint-epoch-{ep:03d}.pth"
                    role = "formal_save" if not rest and ep in K.FORMAL_SAVE_EPOCHS else "recovery"
                    saves.append(K.save(c, e, out / name, "formal", K.ref(history), role))
                if e.completed_updates % 16 == 0:
                    print(json.dumps({"arm": K.ARM, "update": e.completed_updates, "loss": res["loss"],
                                      "seconds": res["elapsed_seconds"]}), flush=True)
            K.require(e.completed_updates != begin, "L0 loader made no progress")
        ledger.flush()
        os.fsync(ledger.fileno())
    wall = time.monotonic() - wall
    K.PHASE["phase"] = "L0:loop accounting"
    done = e.completed_updates - start
    hooks = {k: counts[k] - hook0.get(k, 0) for k in ("actual_updates", "forwards", "forward_attempts")}
    rec = S.RECOMPUTE["calls"] - rec0
    bn1 = bn_clocks(e.model)
    K.require(e.completed_updates == stop and done == stop - start and not prior_rows, "L0 loop count")
    K.require(hooks == {"actual_updates": done, "forwards": 2 * done, "forward_attempts": 2 * done}, f"L0 hook counts {hooks}")
    K.require(rec == 16 * done, f"L0 block recomputation count {rec}")
    K.require(set(bn1) == set(bn0) and all(bn1[n][1] == bn0[n][1] for n in bn0), "L0 BN module set / modes changed")
    bad = [n for n in bn0 if bn1[n][0] - bn0[n][0] != (2 * done if bn0[n][1] else 0)]
    K.require(not bad, f"L0 BN update count is not one per training forward: {bad[:4]}")
    return {"updates": done, "hook_updates": hooks["actual_updates"], "hook_forwards": hooks["forwards"],
            "block_recomputations": rec, "block_recomputations_per_update": 16,
            "block_recomputations_note": "activation recomputation of checkpointed segments, not extra complete forwards",
            "BN": {"tracked_modules": len(bn0), "training_modules": sum(m for _, m in bn0.values()),
                   "increment_per_training_module": 2 * done, "one_update_per_training_forward": True},
            "saves": saves, "prior_rows_replayed_exact": replayed, "shared_rows_compared_live": compared,
            "loop_wall_seconds": wall, "update_seconds_sum": seconds,
            "peak_cuda_bytes": K.torch_mod().cuda.max_memory_allocated() if e.device.type == "cuda" else None}


def train(c, out, counts, resume, *, stop, save_points):
    """Returns (engine, result).  ``formal`` calls this with the fixed schedule; nothing else in the release does."""
    from sparse_rtdetr.research import r27_l0_control as L
    K.PHASE["phase"] = "L0:formal guard"
    out = Path(out)
    source = guard(c, out, resume)
    refs, first = reference_rows(c)
    K.PHASE["phase"] = "L0:construct"
    e, info = L.create(c["parent"]["path"], "cuda:0")
    K.register_counts(e, counts)
    copied, prior_rows, saves = 0, {}, []
    if source is not None:
        side = K.load(c, e, source[0], "formal")
        K.require(side == source[1], "resume side-car changed between guard and load")
        copied, prior_rows = restore_ledger(side, source[0].parent, out, refs)
        K.write_new(out / "RESTORE.json", {"resumed_from": side, "prior_rows_after_state": len(prior_rows),
                                           "ledger_rows_copied": copied})
    else:
        K.require(e.completed_updates == K.PARENT_UPDATE, "a fresh formal run starts at the parent update")
        _available(out)
        saves.append(K.save(c, e, out / "checkpoint-epoch-060.pth", "formal", None, "formal_save"))
    start = e.completed_updates
    plan = {"arm": K.ARM, "parent": c["parent"], "start_update": start, "stop_update": stop, "updates": stop - start,
            "schedule_first_update": K.PARENT_UPDATE + 1, "save_points": sorted(save_points),
            "formal_save_epochs": list(K.FORMAL_SAVE_EPOCHS), "scope": "formal", "evaluation": c["workload"]["evaluation"]}
    loop = run_loop(c, e, out, stop, counts, save_points=set(save_points), refs=refs, prior_rows=prior_rows)
    with (out / LEDGER).open() as f:
        head = json.loads(f.readline())
        rows = 1 + sum(1 for _ in f)
    K.require(rows == stop - K.PARENT_UPDATE == copied + loop["shared_rows_compared_live"] and head["input"]["update"] == first["update"],
              "L0 ledger row count")
    loop["saves"] = saves + loop["saves"]
    reading = {"update": first["update"], "L0": head["loss"], "R27_P": first["P"], "R27_C": first["C"],
               "equal_to_P": head["loss"] == first["P"], "equal_to_C": head["loss"] == first["C"],
               "note": "informational reading, not a gate"}
    return e, {"arm": K.ARM, "plan": plan, "install": {k: v for k, v in info.items() if k != "new_parameter_names"},
               "resumed": source is not None, "ledger_rows": rows, "ledger_rows_copied_on_resume": copied,
               "shared_rows_compared": copied + loop["shared_rows_compared_live"], "shared_rows_different": 0,
               "shared_fields": list(K.SHARED_FIELDS), "first_update_loss": reading, **loop}


def formal(c, out, counts, resume):
    return train(c, out, counts, resume, stop=K.ENDPOINT, save_points=FORMAL_POINTS)[1]
