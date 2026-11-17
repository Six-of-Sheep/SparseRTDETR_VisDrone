"""R29 window delivery: how an audited window reaches Engine.train_window (engineering only; user instruction of
2026-11-17 "pipeline optimisation before the formal R29 runs").

What was measured on the accepted path (release-001: workers 2, one prepared window, audit before consumption): the
loader and the audit already run beside the update and the consumer never waits for a window; the consumer itself
spends about 0.6 s per update in front of the engine's timer, in the engine's own input checks (finite, range [0, 1])
on the 446 MB host tensor.  The same checks on the device tensor take milliseconds.

Modes (the engine, its checks and every audit are unchanged in all of them):

  host      the reference delivery of release-001: the host tensor is handed to Engine.train_window, which checks it on
            the host and copies the two microbatches to the device.
  device    the consumer copies the whole audited window to the device (pageable, blocking) after the seal check and
            hands the device tensor to the unchanged Engine.train_window: the engine's checks run on the device tensor
            and its two microbatch `.to(device)` calls are views.
  prepared  the producer side pins the audited window and starts a non-blocking copy to the device right after the
            audit; the consumer receives the device tensor.  At most one prepared window exists (the bound of the
            overlapped path), now also on the device.

Not changed by any mode: the loader (spawn, workers 2, prefetch_factor 2, its generator), the augmentation draws, the
per-image and whole-window digests and their comparison with the reference ledger before consumption, the seal of the
audited host tensor, the order, the window bytes, the physical batch (8 x 2), the precision, the engine and its timer.
No mode draws a random number or touches a model.

`r29_pipe` is the dev-round measurement of the modes (and of another worker count) on the real QP / QC loop from the
e060 parent; it trains nothing that is kept.
"""
import contextlib
import gc
import json
import os
import statistics
import threading
import time
from pathlib import Path

import r28_common as K
import r28_pipeline as PL

VERSION = "r29-window-delivery-v1"
MODES = ("host", "device", "prepared")
FORMAL = "device"                       # the delivery of the formal QP / QC runs of this release
SCOPE = "r29-pipeline-measurement"      # state scope of the measurement saves: refused by every formal read
RULE = {
    "version": VERSION, "mode": FORMAL, "reference_mode": "host", "workers": PL.WORKERS, "ready_limit": PL.READY_LIMIT,
    "input_pipeline_version": PL.VERSION,
    "what": {"host": "the host tensor is handed to Engine.train_window (release-001)",
             "device": "the consumer copies the whole audited, sealed window to the device and hands the device tensor to the "
                       "unchanged Engine.train_window; the engine's finite / range checks run on the device tensor",
             "prepared": "the producer side pins the audited window and starts a non-blocking device copy; the consumer "
                         "receives the device tensor"}[FORMAL],
    "not_changed": ["workers=2, spawn, prefetch_factor=2, the loader generator", "augmentation inside the worker processes",
                    "per-image SHA, whole-window digests, comparison with the reference ledger before consumption",
                    "the seal of the audited host tensor, verified before the window is used",
                    "the engine (byte-identical file): its input checks, the 8 x 2 microbatches, BF16, the update, its timer",
                    "one prepared window at most; restart from the committed cursor"],
    "accepted_by": "accept-R29: every update of both arms (outputs, gradients, model / optimizer / EMA / RNG state, restore "
                   "replay) is compared bit for bit with the stored acceptance receipts of release-001 (host delivery)",
    "escape": "R29_DELIVERY=host selects the reference delivery",
}


class Delivery:
    """One per stream.  stream() opens the audited window stream; take(item) returns the tensor for the engine."""

    def __init__(self, mode, device):
        torch = K.torch_mod()
        K.require(mode in MODES, f"window delivery mode: {mode!r}")
        self.mode, self.device = mode, torch.device(device)
        K.require(mode != "prepared" or self.device.type == "cuda", "the prepared delivery needs a CUDA device")
        self.slots, self.lock = {}, threading.Lock()
        self.taken, self.deliver_s, self.prepare_s = 0, 0.0, 0.0

    # ---- producer side (prepared mode only)
    def _prepared_windows(self, data, completed, *, limit=None):
        import formal_data
        inner = formal_data.windows(data, completed, limit=limit)
        try:
            for x, t, inp in inner:             # audited by the generator itself before it yields
                started = time.perf_counter()
                pinned = x.pin_memory()
                moved = pinned.to(self.device, non_blocking=True)
                with self.lock:
                    K.require(inp["update"] not in self.slots, f"window {inp['update']} was prepared twice")
                    self.slots[inp["update"]] = (x, pinned, moved, time.perf_counter() - started)
                yield x, t, inp
        finally:
            inner.close()

    def stream(self, pipeline, data, start, stop, check, *, fine=False):
        import formal_data
        windows = self._prepared_windows if self.mode == "prepared" else formal_data.windows
        return PL.Stream(pipeline, windows, data, start, stop, formal_data.WINDOWS_PER_EPOCH, check,
                         formal_data=formal_data, fine=fine)

    # ---- consumer side
    def take(self, item):
        """The tensor handed to Engine.train_window for a delivered window (its seal was verified by the stream)."""
        x, prepared = item.x, None
        started = time.perf_counter()
        if self.mode == "host":
            out = x
        elif self.mode == "device":
            out = x.to(self.device)
        else:
            with self.lock:
                slot = self.slots.pop(item.inp["update"], None)
            K.require(slot is not None and slot[0] is x, f"window {item.inp['update']} has no prepared device copy of the audited tensor")
            out, prepared = slot[2], slot[3]
            self.prepare_s += prepared
            with self.lock:                      # a window dropped by a closed epoch generator is never consumed
                for u in [u for u in self.slots if u <= item.inp["update"]]:
                    del self.slots[u]
        K.require(tuple(out.shape) == tuple(x.shape) and out.dtype == x.dtype and (self.mode == "host" or out.device == self.device),
                  f"delivered window differs in shape / dtype / device from the audited one ({self.mode})")
        spent = time.perf_counter() - started
        self.taken += 1
        self.deliver_s += spent
        return out, {"deliver_s": spent, **({} if prepared is None else {"prepare_s": prepared})}

    def close(self):
        with self.lock:
            self.slots.clear()

    def binding(self):
        return {"version": VERSION, "mode": self.mode, "device": str(self.device)}

    def account(self):
        return {**self.binding(), "taken": self.taken, "deliver_seconds": self.deliver_s,
                **({"prepare_seconds": self.prepare_s} if self.mode == "prepared" else {})}


def formal_mode(c, arm):
    """Delivery of a formal R29 run: the release's mode only if the closed acceptance found it bit-for-bit equal to the
    stored host receipts for that arm; R29_DELIVERY=host keeps the reference delivery."""
    forced = os.environ.get("R29_DELIVERY")
    K.require(forced in (None, "host"), "R29_DELIVERY can only select the host reference delivery")
    K.require(K.r29(c) and arm in K.R29_ARMS, "window delivery is defined for arms QP and QC of the R29 round")
    K.require(c["input_pipeline"].get("r29_window_delivery") == RULE, "the window delivery rule of the contract differs from this release")
    if forced or FORMAL == "host":
        return "host"
    accepted = K.stage_closed(c, "accept-R29").get("window_delivery", {}).get(arm)
    K.require(accepted == {"mode": FORMAL, "equal_to_the_host_receipts": True},
              f"the acceptance did not accept the {FORMAL} delivery for arm {arm}: {accepted}")
    return FORMAL


@contextlib.contextmanager
def loader_workers(n):
    """Measurement only (dev rounds): the frozen loader with another worker count.  The formal path never enters this
    with n != 2: the committed cursor of every formal state names workers 2."""
    import formal_data
    K.require(type(n) is int and n in (2, 4, 8), f"loader workers: {n!r}")
    if n == PL.WORKERS:
        yield
        return
    K.require(K.DEV, "another loader worker count is a dev-round measurement only")
    original = formal_data.training_loader

    def loader(dataset, *, epoch, completed_windows=0, workers=PL.WORKERS, window_limit=None):
        return original(dataset, epoch=epoch, completed_windows=completed_windows, workers=n, window_limit=window_limit)

    formal_data.training_loader = loader
    try:
        yield
    finally:
        formal_data.training_loader = original


# ================================================================ dev-round measurement
def stats(values):
    v = sorted(x for x in values if x is not None)
    if not v:
        return None
    q = lambda p: v[min(len(v) - 1, int(p * len(v)))]
    return {"n": len(v), "median": statistics.median(v), "mean": statistics.fmean(v), "p10": q(0.10), "p90": q(0.90),
            "min": v[0], "max": v[-1]}


def parse_plan(text):
    plan = []
    for part in text.replace(",", " ").split():
        arm, mode, workers = part.split(":")
        K.require(arm in K.R29_ARMS and mode in MODES and workers in ("2", "4", "8"), f"measurement plan entry: {part!r}")
        plan.append((arm, mode, int(workers)))
    K.require(1 <= len(plan) <= 12, "the measurement plan holds 1..12 runs")
    return plan


def segments(rows, warm):
    """Steady rows: after the warm-up updates and without the last update (audited read, save).  The producer segments
    (next / audit / fetch) run beside the consumer segments: they do not add up to the cycle."""
    first, last = rows[0]["update"], rows[-1]["update"]
    sel = [r for r in rows if r["update"] - first >= warm and r["update"] != last]
    d = lambda r: r.get("deliver_s", 0.0)
    cols = {
        "cycle_s": lambda r: r["cycle_s"], "consumer_wait_for_window_s": lambda r: r["wait_s"],
        "deliver_s": d, "producer_prepare_s": lambda r: r.get("prepare_s"),
        "engine_checks_before_its_timer_s": lambda r: r["train_call_s"] - r["train_s"] - d(r),
        "gpu_update_s": lambda r: r["train_s"], "consumer_other_s": lambda r: r["cycle_s"] - r["wait_s"] - r["train_call_s"],
        "producer_next_s": lambda r: r["input"]["next_s"], "producer_audit_s": lambda r: r["input"]["audit_s"],
        "producer_fetch_s": lambda r: r["input"]["fetch_s"], "producer_reference_check_s": lambda r: r["input"]["reference_check_s"],
    }
    return {"steady_updates": len(sel), "steady_first_update": sel[0]["update"], "steady_last_update": sel[-1]["update"],
            "steady": {k: stats([f(r) for r in sel]) for k, f in cols.items()},
            "prepared_windows_seen": sorted({r["prepared"] for r in sel}),
            "first_update": {"cycle_s": rows[0]["cycle_s"], "wait_s": rows[0]["wait_s"], "train_call_s": rows[0]["train_call_s"],
                             "note": "includes the spawn of the loader workers and the first CUDA calls"},
            "last_update": {"cycle_s": rows[-1]["cycle_s"], "save_s": rows[-1].get("save_s"),
                            "note": "the audited read and the state save of the measurement"}}


def r29_pipe(c, out, counts):
    """Dev round, CUDA, exclusive GPU.  R29_PIPE_PLAN = "QP:host:2 QP:device:2 ..." (arm:mode:workers, run in that
    order in this process); every run builds a fresh engine from the e060 parent and runs the real loop on the same
    fixed windows (updates 18241 .. 18240 + R29_PIPE_STEPS), then saves a state (scope: measurement)."""
    import l0_loop as LL
    import r28_cal_control as CC
    import r28_loop as LP
    import r29_control as RC
    from sparse_rtdetr.research import r29_unzoom as UZ
    from sparse_rtdetr.research import scale_baseline as S
    torch = K.torch_mod()
    out = Path(out)
    K.require(K.DEV and K.r29(c) and out.resolve().parent == K.ROUND.resolve() and out.name.startswith("pipe-"),
              "the pipeline measurement writes <dev round>/pipe-<name> under an R29 dev contract")
    plan = parse_plan(os.environ.get("R29_PIPE_PLAN", ""))
    steps, warm = int(os.environ.get("R29_PIPE_STEPS", "48")), int(os.environ.get("R29_PIPE_WARM", "8"))
    K.require(16 <= steps <= 304 and 4 <= warm <= steps - 8, "measurement span / warm-up")
    before = RC.exclusive_gpu("before the pipeline measurement")
    refs, _ = LL.reference_rows(c)
    base = RC.base_ledger(c)
    p0 = K.PARENT_UPDATE
    runs = []
    for i, (arm, mode, workers) in enumerate(plan):
        name = f"run-{i:02d}-{arm}-{mode}-w{workers}"
        K.PHASE["phase"] = f"pipe:{name} construct"
        d = out / name
        d.mkdir()
        e, _info = UZ.create(c["parent"]["path"], "cuda:0", arm, K.demand_source(c, arm))
        K.register_counts(e, counts)
        LP.install_observer(e)
        K.require(e.completed_updates == p0 and UZ.is_wrapped(e.model, arm) and UZ.is_wrapped(e.ema.module, arm), f"{name}: start")
        start_state = CC.state_digests(S.capture(e))
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        delivery = Delivery(mode, e.device)
        with loader_workers(workers), PL.MemorySampler(0.5) as memory:
            opened = time.perf_counter()
            loop = LP.run_loop(c, e, d, p0 + steps, counts, save_points={p0 + steps}, refs=refs, prior_rows={}, pipeline="overlap",
                               base=base, audit=RC.audit_updates(p0, p0 + steps), delivery=delivery, scope=SCOPE)
            closed = time.perf_counter()
        peak = {"allocated_bytes": torch.cuda.max_memory_allocated(), "reserved_bytes": torch.cuda.max_memory_reserved()}
        end_state = CC.state_digests(S.capture(e))
        with (d / LP.LEDGER).open() as f:
            ledger = [{k: v for k, v in json.loads(line).items() if k not in LL.REPLAY_SKIP} for line in f]
        with (d / LP.TIMING).open() as f:
            timing = [json.loads(line) for line in f]
        K.require(len(ledger) == len(timing) == steps and loop["updates"] == steps, f"{name}: rows")
        runs.append({"name": name, "arm": arm, "mode": mode, "workers": workers, "updates": steps, "warm_up_updates": warm,
                     "wall_seconds_loop": closed - opened, "loop_wall_seconds": loop["loop_wall_seconds"],
                     **segments(timing, warm), "input_pipeline": loop["input_pipeline"], "window_delivery": loop["window_delivery"],
                     "peak_cuda": peak, "host_memory": memory.summary(),
                     "start_state": start_state, "end_state": end_state, "ledger_rows_sha256": K.sha_bytes(K.canonical(ledger)),
                     "losses": [r["loss"] for r in ledger], "saved": loop["saves"][-1]["file"], "timing_file": K.ref(d / LP.TIMING)})
        print(json.dumps({"run": name, "cycle_median_s": runs[-1]["steady"]["cycle_s"]["median"],
                          "gpu_update_median_s": runs[-1]["steady"]["gpu_update_s"]["median"],
                          "wait_median_s": runs[-1]["steady"]["consumer_wait_for_window_s"]["median"]}), flush=True)
        delivery.close()
        del e, loop, delivery
        gc.collect()
        torch.cuda.empty_cache()
    after = RC.exclusive_gpu("after the pipeline measurement")
    equal = {}
    for arm in sorted({r["arm"] for r in runs}):
        mine = [r for r in runs if r["arm"] == arm]
        equal[arm] = {"runs": [r["name"] for r in mine],
                      "start_states_equal": all(r["start_state"] == mine[0]["start_state"] for r in mine),
                      "ledger_rows_equal": all(r["ledger_rows_sha256"] == mine[0]["ledger_rows_sha256"] for r in mine),
                      "end_states_equal": all(r["end_state"] == mine[0]["end_state"] for r in mine)}
    K.write_new(out / "RESULT.json", {"schema": "r29-pipeline-measurement-v1", "contract_sha256": c["contract_sha256"],
                                      "plan": [list(p) for p in plan], "steps": steps, "warm_up": warm, "parent": c["parent"],
                                      "first_update": p0 + 1, "last_update": p0 + steps, "gpu_before": before, "gpu_after": after,
                                      "threads": {"intra_op": torch.get_num_threads()}, "runs": runs, "equal_within_this_stage": equal,
                                      "compared": "ledger rows without update_seconds (input record, GT count, loss, lr, gradient norm, "
                                                  "microbatch losses, EMA clock, map / read readings) and the digests of the complete "
                                                  "end state (model, BN buffers, optimizer, EMA, engine clocks, CPU and CUDA RNG)",
                                      "note": "an engineering measurement on windows of the formal schedule; the states are measurement "
                                              "states and are never read by a formal run"})
    return {"stage": out.name, "result": K.ref(out / "RESULT.json"), "runs": [r["name"] for r in runs], "equal_within_this_stage": equal,
            "cycle_median_s": {r["name"]: r["steady"]["cycle_s"]["median"] for r in runs}, "formal_state_usable": False}
