"""R28 formal training loop (arms W and K): e061-e120 from the native e060 parent, same windows as L0 / R27.

The loop is the L0 loop (tools/r27_l0/l0_loop.py) with the R28 engine.  Every window is compared with the pinned R27
formal-P ledger on the declared shared fields before and after the update, exactly as L0 did, so W, K and L0 consume
the same samples, augmentation draws and schedule.  Added without any extra forward:
  * per update, ledger key "r28": magnification of the batch GT by raw-area class, share of small GT with g < 1,
    slope range;
  * per epoch, OBS_EPOCH.jsonl: matched pairs of the final head (positive VFL term, L1, GIoU loss) averaged in four
    bins of the small-GT magnification, and the number of GT whose warped short side is below 2 L pixels.
The readings come from tensors the criterion already holds; they run without gradient and draw no random number.
"""
import json
import math
import os
import time
from pathlib import Path

import r28_common as K
import l0_loop as LL

LEDGER = LL.LEDGER
OBS = "OBS_EPOCH.jsonl"
G_BINS = ((None, 1.0), (1.0, 1.2), (1.2, 1.4), (1.4, None))
BIN_NAMES = ("g<1.0", "1.0<=g<1.2", "1.2<=g<1.4", "g>=1.4")


def _bin(g):
    return 0 if g < 1.0 else 1 if g < 1.2 else 2 if g < 1.4 else 3


class Observer:
    """Training-time readings.  install() wraps two bound methods of the criterion instance; the wrappers call the
    original methods with the original arguments and return their results unchanged."""

    def __init__(self, e):
        self.e, self.ctx = e, e.r28_ctx
        self.ctx.observe, self.ctx.records = True, []
        self.pairs = []          # (bin, vfl_pos, l1, giou_loss) of small matched GT of the final head
        self.pending = {}
        self.epoch = self._zero()
        crit = e.criterion
        self._boxes, self._vfl = crit.loss_boxes, crit.loss_labels_vfl
        crit.loss_boxes, crit.loss_labels_vfl = self.loss_boxes, self.loss_labels_vfl

    @staticmethod
    def _zero():
        return {"pairs": [0] * 4, "vfl_pos": [0.0] * 4, "l1": [0.0] * 4, "giou": [0.0] * 4, "gt": 0, "short_lt_2px": 0,
                "small": 0, "small_short_lt_2px": 0, "updates": 0}

    def _final(self, outputs):
        return self.e.criterion._families.get(id(outputs), "final") == "final"  # same default as the criterion

    def _matched(self, outputs, targets, indices):
        torch = K.torch_mod()
        rec = self.ctx.records[-1]
        g, small, qi, bi = [], [], [], []
        for i, (t, (src, tgt)) in enumerate(zip(targets, indices)):
            tgt = tgt.cpu()
            g.append(rec["g"][i][tgt])
            small.append((t["raw_area"].cpu()[tgt] <= K.SMALL_AREA))
            qi.append(src)
            bi.append(torch.full_like(src, i))
        return torch.cat(g), torch.cat(small), (torch.cat(bi), torch.cat(qi))

    def loss_boxes(self, outputs, targets, indices, num_boxes, boxes_weight=None):
        if self._final(outputs) and sum(len(s) for s, _ in indices):
            torch = K.torch_mod()
            with torch.no_grad():
                import importlib
                m = importlib.import_module(type(self.e.criterion).__mro__[1].__module__)
                g, small, idx = self._matched(outputs, targets, indices)
                src = outputs["pred_boxes"][idx].detach().float()
                gt = torch.cat([t["boxes"][j] for t, (_, j) in zip(targets, indices)]).float()
                l1 = (src - gt).abs().sum(-1).cpu()
                giou = (1 - torch.diag(m.generalized_box_iou(m.box_cxcywh_to_xyxy(src), m.box_cxcywh_to_xyxy(gt)))).cpu()
                self.pending["boxes"] = (g, small, l1, giou)
        return self._boxes(outputs, targets, indices, num_boxes, boxes_weight)

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes, values=None):
        if self._final(outputs) and sum(len(s) for s, _ in indices):
            torch = K.torch_mod()
            F = torch.nn.functional
            with torch.no_grad():
                import importlib
                m = importlib.import_module(type(self.e.criterion).__mro__[1].__module__)
                _, _, idx = self._matched(outputs, targets, indices)
                src = outputs["pred_boxes"][idx].detach().float()
                gt = torch.cat([t["boxes"][j] for t, (_, j) in zip(targets, indices)]).float()
                iou = torch.diag(m.box_iou(m.box_cxcywh_to_xyxy(src), m.box_cxcywh_to_xyxy(gt))[0]).clamp(0, 1)
                labels = torch.cat([t["labels"][j] for t, (_, j) in zip(targets, indices)])
                logit = outputs["pred_logits"].detach().float()[idx[0], idx[1], labels]
                self.pending["vfl"] = (F.binary_cross_entropy_with_logits(logit, iou, reduction="none") * iou).cpu()
        return self._vfl(outputs, targets, indices, num_boxes, values)

    def microbatch_done(self):
        """Called by the forward hook of the next forward and at the end of the update: fold the pending pair readings."""
        if "boxes" in self.pending and "vfl" in self.pending:
            g, small, l1, giou = self.pending["boxes"]
            vfl = self.pending["vfl"]
            K.require(len(vfl) == len(g), "observer: final-head pair counts differ between the box and the VFL readings")
            for k in small.nonzero().flatten().tolist():
                self.pairs.append((_bin(float(g[k])), float(vfl[k]), float(l1[k]), float(giou[k])))
        self.pending = {}

    def update_row(self, targets):
        """Per-update ledger entry from the two forward records of this update; also folds the epoch accumulators."""
        torch = K.torch_mod()
        self.microbatch_done()
        recs, self.ctx.records = self.ctx.records, []
        K.require(len(recs) == 2 and sum(len(r["n"]) for r in recs) == len(targets), "observer: forward records of the update")
        g = torch.cat([v for r in recs for v in r["g"]])
        short = torch.cat([v for r in recs for v in r["short"]])
        area = torch.cat([t["raw_area"] for t in targets]).to(g.dtype)
        K.require(len(g) == len(area), "observer: GT count")
        small, large = area <= K.SMALL_AREA, area >= K.LARGE_AREA
        medium = (area >= K.SMALL_AREA) & (area <= K.LARGE_AREA)

        def geo(sel):
            return float(torch.exp(torch.log(g[sel]).mean())) if bool(sel.any()) else None

        ep = self.epoch
        ep["updates"] += 1
        ep["gt"] += len(g)
        ep["short_lt_2px"] += int((short < 2.0).sum())
        ep["small"] += int(small.sum())
        ep["small_short_lt_2px"] += int(((short < 2.0) & small).sum())
        for b, v, l1, gi in self.pairs:
            ep["pairs"][b] += 1
            ep["vfl_pos"][b] += v
            ep["l1"][b] += l1
            ep["giou"][b] += gi
        self.pairs = []
        return {"g_small": geo(small), "g_medium": geo(medium), "g_large": geo(large), "n_small": int(small.sum()),
                "small_g_lt_1": float((g[small] < 1.0).float().mean()) if bool(small.any()) else None,
                "slope_min": min(r["slope_min"] for r in recs), "slope_max": max(r["slope_max"] for r in recs),
                "identity_images": sum(r["identity_images"] for r in recs)}

    def epoch_row(self, epoch):
        ep, self.epoch = self.epoch, self._zero()

        def mean(key):
            return [ep[key][b] / ep["pairs"][b] if ep["pairs"][b] else None for b in range(4)]

        return {"epoch": epoch, "updates": ep["updates"], "bins": list(BIN_NAMES), "small_matched_pairs_final_head": ep["pairs"],
                "vfl_positive_term_mean": mean("vfl_pos"), "l1_mean": mean("l1"), "giou_loss_mean": mean("giou"),
                "gt": ep["gt"], "gt_warped_short_side_lt_2_L_pixels": ep["short_lt_2px"], "small_gt": ep["small"],
                "small_gt_warped_short_side_lt_2_L_pixels": ep["small_short_lt_2px"],
                "note": "pairs are the Hungarian matches of the final decoder head on GT with raw area <= 32^2; "
                        "the VFL reading is the matched-class term BCE(logit, IoU) * IoU"}


def install_observer(e):
    obs = Observer(e)
    e.model.register_forward_pre_hook(lambda *_: obs.microbatch_done())
    e.r28_observer = obs
    return obs


def guard(c, out, resume, arm):
    K.g0_passed(c)
    for stage in ("cpu", "zero", "evalpath", f"accept-{arm}"):
        K.stage_closed(c, stage)
    if arm == "W":
        K.require(K.stage_closed(c, "precost")["C_pre_passed"] is True or (K.DEV and os.environ.get("R28_DEV_IGNORE_COST") == "1"),
                  "the pre-launch cost check did not pass")
    else:
        K.require(K.stage_closed(c, "summary-W")["K_trigger"] is True or (K.DEV and os.environ.get("R28_DEV_FORCE_K") == "1"),
                  "arm K is not triggered by the W result")
    out = Path(out).resolve()
    K.require(out.parent == K.ROUND.resolve(), "formal output must be directly under the round")
    pattern = K.formal_re(arm)
    closed = sorted(d.name for d in K.ROUND.iterdir() if d.is_dir() and pattern.fullmatch(d.name) and (d / "CLOSED.json").is_file())
    K.require(not closed, f"this round already has a closed formal directory {closed}: another formal run is refused")
    if resume is None:
        K.require(out.name == f"formal-{arm}", f"a formal run without --resume writes <round>/formal-{arm}")
        return None
    path, side = K.read_side(c, resume, "formal", arm)
    m = pattern.fullmatch(out.name)
    K.require(m is not None and m.group(1) is not None, f"a resumed formal run writes <round>/formal-{arm}-rN (N>=2)")
    K.require(path.parent.parent == K.ROUND.resolve() and pattern.fullmatch(path.parent.name) is not None and path.parent != out,
              "formal resume source must be a state in this round's formal directory of the same arm")
    K.require(side["global_update"] % K.WINDOWS_PER_EPOCH == 0 or K.DEV, "formal resume source must be an epoch boundary state")
    return path, side


def run_loop(c, e, out, stop, counts, *, save_points, refs, prior_rows, first=None):
    from formal_data import windows, WINDOWS_PER_EPOCH
    from sparse_rtdetr.research import scale_baseline as S
    arm, obs = e.r28_arm, e.r28_observer
    history = out / LEDGER
    data = S.train_dataset(c["data"]["train_core"], S.S15_HW, seed=0)
    start = e.completed_updates
    base = {"start": start, "recompute": S.RECOMPUTE["calls"], "bn": LL.bn_clocks(e.model), "hooks": dict(counts)}
    bn0 = base["bn"]
    K.require(start < stop and all(start < p <= stop for p in save_points), "R28 loop span / save points")
    prior_rows = dict(prior_rows)
    saves, seen, replayed, compared, seconds, checked, epochs = [], {}, 0, 0, 0.0, [], []
    wall = time.monotonic()
    with history.open("a", buffering=1) as ledger, (out / OBS).open("a", buffering=1) as obs_file:
        while e.completed_updates < stop:
            begin = e.completed_updates
            for x, t, inp in windows(data, begin, limit=min(stop - begin, WINDOWS_PER_EPOCH - begin % WINDOWS_PER_EPOCH)):
                ids = [tt["image_id"] for tt in t]
                s_ = seen.setdefault(inp["epoch"], set())
                K.require(len(set(ids)) == len(ids) and not (s_ & set(ids)), "R28 occurrence: image repeated within an epoch")
                s_.update(ids)
                K.PHASE["phase"] = f"{arm}:input identity {e.completed_updates + 1}"
                K.require(inp["update"] == e.completed_updates + 1 and inp["update"] in refs,
                          f"window {inp['update']} is not the next update of the schedule")
                ref_row = refs[inp["update"]]
                bad = LL.input_differences(inp, ref_row["input"])
                K.require(not bad, f"input record differs from R27 formal-P before update {inp['update']} "
                                   f"(the model was not updated with this window): {bad}")
                K.PHASE["phase"] = f"{arm}:update {e.completed_updates + 1}"
                res = e.train_window(x, t)
                K.require(res["update"] == inp["update"] and res["finite"] and e.ema.updates == e.completed_updates,
                          "R28 update clock/input")
                row = {"input": inp, "gt_count": int(sum(len(v["labels"]) for v in t)), "loss": res["loss"],
                       "lr": res["lr"], "gradient_norm_before_clip": res["gradient_norm_before_clip"],
                       "microbatch_losses": res["microbatch_losses"], "update_seconds": res["elapsed_seconds"],
                       "ema_updates": e.ema.updates, "r28": obs.update_row(t)}
                bad = LL.shared_differences(row, ref_row)
                K.require(not bad, f"shared fields differ from R27 formal-P at update {res['update']}: {bad}")
                K.compare_shared(row, ref_row)
                compared += 1
                if res["update"] in prior_rows:
                    K.require(prior_rows.pop(res["update"]) == {k: v for k, v in row.items() if k not in LL.REPLAY_SKIP},
                              f"recovery replay differs from the recorded trajectory at update {res['update']}")
                    replayed += 1
                ledger.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                seconds += res["elapsed_seconds"]
                if first is not None and res["update"] == first["update"]:
                    print(json.dumps({"first_update_loss": {"update": res["update"], arm: res["loss"], "R27_P": first["P"],
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
                    K.PHASE["phase"] = f"{arm}:accounting before the save at {e.completed_updates}"
                    LL.account(e, counts, base)
                    checked.append(e.completed_updates)
                    LL._available(out)
                    ep, rest = divmod(e.completed_updates, WINDOWS_PER_EPOCH)
                    name = K.endpoint_name(e.completed_updates)
                    role = "formal_save" if not rest and ep in K.FORMAL_SAVE_EPOCHS else "recovery"
                    saves.append(K.save(c, e, out / name, "formal", {"ledger": K.ref(history), "observations": K.ref(out / OBS)}, role))
                if e.completed_updates % 16 == 0:
                    print(json.dumps({"arm": arm, "update": e.completed_updates, "loss": res["loss"],
                                      "seconds": res["elapsed_seconds"], "g_small": row["r28"]["g_small"]}), flush=True)
            K.require(e.completed_updates != begin, "R28 loader made no progress")
        ledger.flush()
        os.fsync(ledger.fileno())
    wall = time.monotonic() - wall
    K.PHASE["phase"] = f"{arm}:loop accounting"
    done, hooks, rec = LL.account(e, counts, base)
    K.require(e.completed_updates == stop and done == stop - start and not prior_rows, "R28 loop count")
    return {"updates": done, "hook_updates": hooks["actual_updates"], "hook_forwards": hooks["forwards"],
            "block_recomputations": rec, "block_recomputations_per_update": 16,
            "BN": {"tracked_modules": len(bn0), "training_modules": sum(m for _, m in bn0.values()),
                   "increment_per_training_module": 2 * done, "one_update_per_training_forward": True},
            "saves": saves, "accounting_checked_before_saves_at": checked, "prior_rows_replayed_exact": replayed,
            "shared_rows_compared_live": compared, "observation_epochs_written": epochs,
            "loop_wall_seconds": wall, "update_seconds_sum": seconds,
            "peak_cuda_bytes": K.torch_mod().cuda.max_memory_allocated() if e.device.type == "cuda" else None}


def restore_files(side, src_dir, out, refs):
    """Ledger prefix (L0 rule) and the observation file prefix recorded by the resume state."""
    audit = side["audit"]
    if audit is None:
        copied, prior = LL.restore_ledger({**side, "audit": None}, src_dir, out, refs)
        (out / OBS).open("xb").close()
        return copied, prior
    copied, prior = LL.restore_ledger({**side, "audit": audit["ledger"]}, src_dir, out, refs)
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


def train(c, out, counts, resume, *, stop, save_points):
    from sparse_rtdetr.research import r28_resample as R
    arm = K.ARM
    K.PHASE["phase"] = f"{arm}:formal guard"
    out = Path(out)
    source = guard(c, out, resume, arm)
    refs, first = LL.reference_rows(c)
    K.PHASE["phase"] = f"{arm}:construct"
    e, info = R.create(c["parent"]["path"], "cuda:0", arm, K.demand_source(c, arm))
    K.register_counts(e, counts)
    install_observer(e)
    copied, prior_rows, saves = 0, {}, []
    if source is not None:
        side = K.load(c, e, source[0], "formal")
        K.require(side == source[1], "resume side-car changed between guard and load")
        copied, prior_rows = restore_files(side, source[0].parent, out, refs)
        K.write_new(out / "RESTORE.json", {"resumed_from": side, "prior_rows_after_state": len(prior_rows),
                                           "ledger_rows_copied": copied})
    else:
        K.require(e.completed_updates == K.PARENT_UPDATE, "a fresh formal run starts at the parent update")
        LL._available(out)
        saves.append(K.save(c, e, out / "checkpoint-epoch-060.pth", "formal", None, "formal_save"))
    start = e.completed_updates
    used = LL.points_after(save_points, start)
    K.require(stop in used, "the stop update must be a save point after the start (the endpoint state is always written)")
    plan = {"arm": arm, "parent": c["parent"], "start_update": start, "stop_update": stop, "updates": stop - start,
            "schedule_first_update": K.PARENT_UPDATE + 1, "demand": e.r28_ctx.binding(),
            "save_points": {"planned": sorted(set(save_points)), "used_in_this_process": list(used),
                            "at_or_before_the_start": sorted(p for p in set(save_points) if p <= start)},
            "formal_save_epochs": list(K.FORMAL_SAVE_EPOCHS), "scope": "formal"}
    loop = run_loop(c, e, out, stop, counts, save_points=set(used), refs=refs, prior_rows=prior_rows, first=first)
    with (out / LEDGER).open() as f:
        head = json.loads(f.readline())
        rows = 1 + sum(1 for _ in f)
    K.require(rows == stop - K.PARENT_UPDATE == copied + loop["shared_rows_compared_live"] and head["input"]["update"] == first["update"],
              "R28 ledger row count")
    loop["saves"] = saves + loop["saves"]
    return e, {"arm": arm, "plan": plan, "install": {k: v for k, v in info.items() if k != "new_parameter_names"},
               "resumed": source is not None, "ledger_rows": rows, "ledger_rows_copied_on_resume": copied,
               "shared_rows_compared": copied + loop["shared_rows_compared_live"], "shared_rows_different": 0,
               "shared_fields": list(K.SHARED_FIELDS),
               "first_update_loss": {"update": first["update"], arm: head["loss"], "R27_P": first["P"], "R27_C": first["C"]},
               "dev_span": os.environ.get("R28_DEV_SPAN"), **loop}


def formal(c, out, counts, resume):
    stop, dev_points = K.formal_stop()
    points = dev_points or tuple(ep * K.WINDOWS_PER_EPOCH for ep in K.SAVE_EPOCHS if ep * K.WINDOWS_PER_EPOCH > K.PARENT_UPDATE)
    return train(c, out, counts, resume, stop=stop, save_points=points)[1]
