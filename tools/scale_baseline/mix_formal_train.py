"""TASK_MIX_SCALE_001 trainer: CPU gate, CUDA restore acceptance and formal MIX e061-e120.

actions (always launched by mix_formal_run.py from a frozen release with a bound contract):
  cpu     CUDA hidden.  (1) the frozen table: rule regeneration, coverage, per-epoch quotas, totals;
          (2) the real 2-worker prefetching loader on the formal table: e061 w0-w5 (H L H L L L) and
          e117 w300-w303 + e118 w0-w1 (L->H at the epoch/augmentation boundary), each window equal to the
          native same-canvas loader (formal_data.windows with the native 768 / S15 1152 dataset) and to the
          historical R25-A0 (L) / S15 (H) ledger row, plus an in-epoch restart;
          (3) constant-L / constant-H test modes vs the original entries (R25 A0 create at 768, S15 create
          at 1152), same parent and same input, FP32 bs1x2 fixture, 2 updates each: results and complete state;
          (4) the real formal loop on the frozen table (fixture: H, L, H), then exact resume from u1 replaying 2;
          (5) production e060 bridge on CPU and formal-binding save/load/exact restore;
          (6) refusals: parent, table SHA/rule, scope/schedule mode, cursor, actual canvas, microbatch.
  accept  CUDA.  Production 8x2/BF16 activation-checkpoint engine from the e060 parent with the
          acceptance-only L->H->L sequence on data windows 0,1,2: 3 direct updates (save u1), CPU capture,
          release, fresh engine restores u1 and replays u2/u3; per-step results and complete state exact.
          5 actual updates, 10 training forwards.  Acceptance states are never formal.
  train   CUDA.  Formal MIX from the certified e060 parent (or an exact restart of this formal identity).
"""
from __future__ import annotations

import os
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
import argparse
import gc
import hashlib
import json
import math
import shutil
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools" / "query_local"), str(ROOT / "tools" / "scale_baseline"),
                str(ROOT)]

import torch

from r25_run import setup
import formal_data
from formal_data import cursor
from sparse_rtdetr.protocol_r18 import checkpoint as ck
from sparse_rtdetr.protocol_r18.engine import assert_healthy, learning_rate
from sparse_rtdetr.protocol_r18.model import Config
from sparse_rtdetr.protocol_r18.runio import fsync_directory
from sparse_rtdetr.research import mix_scale as M
from sparse_rtdetr.research import scale_baseline as S

ARM = "MIX"
SCHEMA = "mix-complete-state-v1"
SAVE_EPOCHS = (60, 75, 90, 105, 120)
WPE = M.WINDOWS_PER_EPOCH
PARENT, END = M.PARENT_UPDATE, M.END_UPDATE
BLOCKS = 8
RECOMPUTES_PER_UPDATE = 2 * BLOCKS
FIXTURE_UPDATES = 3                      # formal table e061 w0..w2 = H, L, H
CONSTANT_UPDATES = 2
LOADER_RANGES = ((PARENT, 6), (117 * WPE - 4, 4), (117 * WPE, 2))  # e061 w0-5; e117 w300-303; e118 w0-1
RESTART_RANGE = (PARENT + 1, 2)
EXCLUDED_ROW_FIELDS = ("update_seconds", "peak_allocated_bytes")
PHASE = {"phase": "initialize"}


# ---------------------------------------------------------------- files
def _atomic(path, writer):
    path = Path(path)
    partial = path.with_name(path.name + ".partial")
    if path.exists():
        raise FileExistsError(path)
    with partial.open("xb") as stream:
        writer(stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(partial, path)
    fsync_directory(path.parent)
    partial.unlink()
    fsync_directory(path.parent)


def _json(path, data):
    _atomic(path, lambda stream: stream.write(json.dumps(data, indent=2, allow_nan=False).encode() + b"\n"))


def _file(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": ck.file_sha(path)}


def _pinned(reference):
    S.require(_file(reference["path"]) == {**reference, "path": str(Path(reference["path"]).resolve())},
              f"pinned reference differs: {reference['path']}")
    return Path(reference["path"])


def _sha_prefix(path, count):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while count:
            block = stream.read(min(count, 1024 * 1024))
            if not block:
                raise ValueError("incomplete MIX audit prefix")
            h.update(block)
            count -= len(block)
    return h.hexdigest()


def _memory():
    if not torch.cuda.is_initialized():
        return {"peak_allocated_bytes": None, "peak_reserved_bytes": None}
    return {"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved()}


def _reset_peak():
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def _ledger(reference):
    """Historical ledger rows keyed by update (R25 A0 at 768 or S15 at 1152; pinned bytes)."""
    rows = {}
    with _pinned(reference).open() as stream:
        for line in stream:
            row = json.loads(line)
            rows[row["input"]["update"]] = row
    S.require(sorted(rows) == list(range(PARENT + 1, END + 1)), "historical ledger coverage")
    return rows


# ---------------------------------------------------------------- schedules / state binding
def formal_schedule(c):
    table = c["schedule"]["table"]
    _pinned(table)
    s = M.formal_schedule(table["path"], c["schedule"]["table_sha256"])
    S.require(s.identity == c["schedule"]["identity"], "MIX formal schedule identity differs from the contract")
    return s


def _next_canvas(schedule, update):
    return schedule.canvas(update + 1) if update < END else None


def _state_binding(c, update, scope, audit, schedule, fixture=False):
    S.require(scope in ("formal", "acceptance", "cpu-fixture"), "MIX state scope")
    if scope == "formal":
        S.require(schedule.mode == M.FORMAL and schedule.identity == c["schedule"]["identity"] and not fixture,
                  "formal MIX state requires the frozen contract table and production recipe")
    return {"schema": SCHEMA, "scope": scope, "arm": ARM, "seed": M.SEED, "formal_identity": c["formal_identity"],
            "contract_sha256": c["contract_sha256"], "global_update": update,
            "parent_sha256": c["parent"]["sha256"],
            "cursor": {**cursor(update), "next_canvas": _next_canvas(schedule, update)},
            "schedule": schedule.identity, "audit": audit, "config": M.base_config(fixture).binding(),
            "activation_checkpoint": c["workload"]["activation_checkpoint"], "torch": str(torch.__version__)}


def _save(e, c, path, scope, schedule, audit=None, fixture=False):
    path = Path(path)
    S.require(e.ema.updates == e.completed_updates, "MIX EMA clock disagrees at save")
    S.require(e.mix_schedule is schedule, "MIX save schedule differs from the engine schedule")
    assert_healthy(e.model, e.optimizer, e.ema)
    state = M.capture(e)
    binding = _state_binding(c, e.completed_updates, scope, audit, schedule, fixture)
    state["formal_binding_sha256"] = hashlib.sha256(ck.canonical(binding)).hexdigest()
    _atomic(path, lambda stream: torch.save(state, stream))
    result = {"status": "COMPLETE", **binding, "file": _file(path)}
    _json(path.with_suffix(".pth.json"), result)
    return result


def _verify_audit(audit, path):
    S.require(audit["path"] == str(Path(path).resolve()) and type(audit["bytes"]) is int and audit["bytes"] >= 0
              and audit["sha256"] == _sha_prefix(path, audit["bytes"]), "MIX formal audit prefix differs")


def _load(c, path, scope, schedule, *, endpoint=False, fixture=False):
    path = Path(path).resolve()
    side = json.loads(path.with_suffix(".pth.json").read_text())
    update = side["global_update"]
    binding = _state_binding(c, update, scope, side.get("audit"), schedule, fixture)
    S.require(side == {"status": "COMPLETE", **binding, "file": _file(path)}, "MIX state receipt or file identity differs")
    if scope == "formal":
        S.require(update in tuple(ep * WPE for ep in SAVE_EPOCHS), "MIX checkpoint not at a registered boundary")
        S.require(not endpoint or update == END, "MIX evaluation requires e120")
    if side.get("audit") is not None:
        _verify_audit(side["audit"], path.parent / "UPDATES.jsonl")
    s = torch.load(path, map_location="cpu", weights_only=True)
    S.require(s.pop("formal_binding_sha256") == hashlib.sha256(ck.canonical(binding)).hexdigest()
              and s["version"] == M.VERSION and s["arm"] == ARM and s["parent_sha256"] == c["parent"]["sha256"]
              and s["config"] == M.base_config(fixture).binding() and s["schedule"] == schedule.identity
              and s["activation_checkpoint"] == c["workload"]["activation_checkpoint"]
              and s["engine"]["completed_updates"] == update and s["ema"]["updates"] == update
              and s["engine"]["next_epoch"] == side["cursor"]["next_epoch"]
              and s["engine"]["next_window"] == side["cursor"]["next_window"],
              "MIX checkpoint payload identity/clock differs")
    return side, s


def read_endpoint(contract, path):
    """Verified formal e120 MIX payload (model, optimizer, EMA, engine, RNG)."""
    return _load(contract, path, "formal", formal_schedule(contract), endpoint=True)[1]


def _restore_engine(e, path, side, state):
    restored = M.restore(e, state)
    return {**restored, "path": str(path), "global_update": side["global_update"], "sha256": side["file"]["sha256"],
            "cursor": side["cursor"]}


# ---------------------------------------------------------------- observers
def _register_counts(e, counts, label):
    def pre(*_):
        counts.update(forward_attempts=1)
        PHASE["phase"] = f"{label}:training-forward"

    def post(*_):
        counts.update(forwards=1)
        PHASE["phase"] = f"{label}:loss-backward"

    def step_pre(*_):
        PHASE["phase"] = f"{label}:optimizer-step-EMA"

    e.model.register_forward_pre_hook(pre)
    e.model.register_forward_hook(post)
    e.optimizer.register_step_pre_hook(step_pre)
    e.optimizer.register_step_post_hook(lambda *_: counts.update(actual_updates=1))


def _precision_observers(e):
    seen, criterion = [], []

    def model_hook(_m, args, result):
        seen.append({"input": str(args[0].dtype), "shape": list(args[0].shape),
                     "autocast": torch.is_autocast_enabled("cuda"),
                     "autocast_dtype": str(torch.get_autocast_dtype("cuda")),
                     "logits": str(result["pred_logits"].dtype), "boxes": str(result["pred_boxes"].dtype)})

    e.model.register_forward_hook(model_hook)
    e.criterion.register_forward_hook(lambda _c, _a, r: criterion.append(sorted({str(v.dtype) for v in r.values()})))
    return seen, criterion


def _bn_counts(model):
    return {n: int(m.num_batches_tracked) for n, m in model.named_modules()
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)}


def _differences(a, b, path="", out=None, limit=200):
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
            out.append({"path": path, "kind": "type"})
        elif a.dtype != b.dtype or a.shape != b.shape:
            out.append({"path": path, "kind": "dtype/shape", "a": [str(a.dtype), list(a.shape)],
                        "b": [str(b.dtype), list(b.shape)]})
        elif not torch.equal(a, b):
            if a.is_floating_point():
                d = (a.double() - b.double()).abs()
                out.append({"path": path, "kind": "value", "elements_differ": int((a != b).sum()),
                            "max_abs": float(d[torch.isfinite(d)].max()) if torch.isfinite(d).any() else None})
            else:
                out.append({"path": path, "kind": "value", "elements_differ": int((a != b).sum())})
    elif isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            out.append({"path": path, "kind": "keys", "only_a": sorted(map(str, set(a) - set(b)))[:20],
                        "only_b": sorted(map(str, set(b) - set(a)))[:20]})
        for k in sorted(set(a) & set(b), key=str):
            _differences(a[k], b[k], f"{path}/{k}", out, limit)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            out.append({"path": path, "kind": "length", "a": len(a), "b": len(b)})
        for i, (x, y) in enumerate(zip(a, b)):
            _differences(x, y, f"{path}/{i}", out, limit)
    elif a != b:
        out.append({"path": path, "kind": "scalar", "a": repr(a)[:200], "b": repr(b)[:200]})
    return out


def _result_body(result, drop=("elapsed_seconds",)):
    return {k: v for k, v in result.items() if k not in drop}


# ---------------------------------------------------------------- engines/data
def _make(c, device, schedule, *, fixture=False):
    PHASE["phase"] = "MIX:construct"
    return M.create(c["parent"]["path"], device, schedule, fixture=fixture,
                    activation_checkpoint=c["workload"]["activation_checkpoint"])


def _data(c):
    return M.MixDataset(c["data"]["train_core"])


def _update(e, images, targets, receipt, *, exact=False):
    result = M.train_window(e, images, targets, receipt, capture=exact)
    S.require(result["update"] == receipt["update"] and result["epoch"] == receipt["epoch"]
              and result["window"] == receipt["window"] and result["finite"] and e.ema.updates == e.completed_updates,
              "MIX update/input/EMA clock differs")
    return result


def _steps(e):
    result = {}
    for group in e.optimizer.param_groups:
        counts = [int(e.optimizer.state.get(p, {}).get("step", 0)) for p in group["params"]]
        S.require(all(n == e.parameter_steps[name] for n, name in zip(counts, group["param_names"])),
                  "MIX Adam step ledger differs at boundary")
        result[group["name"]] = {"steps_histogram": dict(sorted(Counter(counts).items())), "lr": group["lr"],
                                 "parameters": len(counts)}
    return result


def _relative_change(e, initial):
    num = den = 0.
    for name, p in e.model.named_parameters():
        v = initial[name].float()
        num += float((p.detach().cpu().float() - v).square().sum())
        den += float(v.square().sum())
    return {"shared": {"relative_l2_change": math.sqrt(num / max(den, 1e-30))}}


def _gradient_groups(e):
    norms = {}
    for group in e.optimizer.param_groups:
        values = [p.grad.detach().float().norm() for p in group["params"] if p.grad is not None]
        norms[group["name"]] = float(torch.stack(values).norm()) if values else 0.
    return {"clipped_group_gradient_norm": norms}


def _available(path, next_state_bytes):
    needed = next_state_bytes + 64 * 1024**2
    free = shutil.disk_usage(path).free
    if free < needed:
        raise RuntimeError(f"insufficient disk for next MIX save: {free} bytes, need {needed}")


def _quota(epoch):
    return ({"L": M.L_PER_MIX_EPOCH, "H": WPE - M.L_PER_MIX_EPOCH} if epoch <= M.LAST_MIX_EPOCH
            else {"L": 0, "H": WPE})


# ---------------------------------------------------------------- formal loop (also the CPU fixture)
def _formal(c, out, counts, resume, prior, *, fixture=False):
    """Formal loop on the frozen table.  ``fixture``: identical loop on CPU, FP32 bs1x2 (first two images
    of every real window), FIXTURE_UPDATES updates, every update saved."""
    out = Path(out)
    if (resume is None) != (prior is None):
        raise ValueError("resume requires both checkpoint and prior contract")
    device = "cpu" if fixture else "cuda:0"
    scope = "cpu-fixture" if fixture else "formal"
    stop = PARENT + FIXTURE_UPDATES if fixture else END
    save_points = set(range(PARENT, stop + 1)) if fixture else {ep * WPE for ep in SAVE_EPOCHS}
    schedule = formal_schedule(c)
    e, bridge = _make(c, device, schedule, fixture=fixture)
    _register_counts(e, counts, "MIX-cpu-fixture" if fixture else "MIX-formal")
    _json(out / "BRIDGE.json", bridge)
    initial = {n: p.detach().cpu().clone() for n, p in e.model.named_parameters()}
    history = out / "UPDATES.jsonl"
    restored = None
    previous_rows = {}
    epoch_counts = {}
    if resume is not None:
        if fixture:
            S.require(prior["contract_sha256"] == c["contract_sha256"], "fixture resume uses the same contract")
        else:
            S.require((c.get("prior") or {}).get("contract") == _file(prior["contract_path"])
                      and Path(resume).resolve().parent == Path(prior["outputs"]["train"])
                      and prior["formal_identity"] == c["formal_identity"]
                      and prior["schedule"]["identity"] == c["schedule"]["identity"]
                      and prior["contract_sha256"] != c["contract_sha256"],
                      "MIX restart must preserve scientific identity and table with a distinct attempt")
        side, state = _load(prior, resume, scope, schedule, fixture=fixture)
        PHASE["phase"] = "MIX:restore"
        restored = _restore_engine(e, resume, side, state)
        del state
        remaining = side["audit"]["bytes"]
        with Path(side["audit"]["path"]).open("rb") as source, history.open("xb") as dest:
            while remaining:
                block = source.read(min(remaining, 1024**2))
                S.require(bool(block), "prior audit prefix truncated")
                dest.write(block)
                remaining -= len(block)
            dest.flush()
            os.fsync(dest.fileno())
        with history.open() as stream:  # canvas counts of the committed prefix
            for line in stream:
                row = json.loads(line)
                epoch_counts.setdefault(row["input"]["epoch"], Counter())[row["canvas"]] += 1
        S.require(sum(sum(v.values()) for v in epoch_counts.values()) == e.completed_updates - PARENT,
                  "MIX restored ledger prefix length differs from the state clock")
        with Path(side["audit"]["path"]).open("rb") as source:
            source.seek(side["audit"]["bytes"])
            for line in source:
                if not line.endswith(b"\n"):
                    break
                row = json.loads(line)
                update = row["input"]["update"]
                S.require(update == e.completed_updates + len(previous_rows) + 1, "MIX previous replay evidence order")
                previous_rows[update] = {k: v for k, v in row.items() if k not in EXCLUDED_ROW_FIELDS}
        _json(out / "RESTORE.json", {**restored, "rows_available_for_replay": len(previous_rows)})
    else:
        history.touch(exist_ok=False)
    started_at = e.completed_updates
    recompute_before = S.RECOMPUTE["calls"]
    save_bytes = c["parent"]["bytes"]
    summaries, replayed, monitor = [], 0, {}
    peaks = {"L": 0, "H": 0}
    window_seconds = {"L": [], "H": []}

    def measure(*_):
        if e.completed_updates + 1 in save_points:
            monitor.update(_gradient_groups(e))

    e.optimizer.register_step_pre_hook(measure)

    def checkpoint(grad):
        nonlocal save_bytes
        _available(out, save_bytes)
        name = (f"fixture-update-{e.completed_updates}.pth" if fixture
                else f"checkpoint-epoch-{e.completed_updates // WPE:03d}.pth")
        p = out / name
        receipt = _save(e, c, p, scope, schedule, _file(history), fixture)
        save_bytes = receipt["file"]["bytes"]
        summaries.append(receipt)
        _json(p.with_suffix(".observations.json"), {
            "parameter_steps": _steps(e), "parameter_change_from_e060": _relative_change(e, initial),
            "gradient_last_update": grad, "EMA_updates": e.ema.updates,
            "canvas_counts_by_epoch": {str(k): dict(v) for k, v in sorted(epoch_counts.items())},
            "block_recomputations_this_process": S.RECOMPUTE["calls"] - recompute_before})

    if resume is None:
        checkpoint(None)
    dataset = _data(c)
    with history.open("a", buffering=1) as ledger:
        while e.completed_updates < stop:
            _available(out, save_bytes)
            begin = e.completed_updates
            limit = (stop - begin) if fixture else None
            for x, t, inp in M.windows(dataset, begin, schedule, limit=limit):
                if fixture:
                    x, t = x[:2], t[:2]
                monitor.clear()
                _reset_peak()
                result = _update(e, x, t, inp)
                peak = _memory()["peak_allocated_bytes"]
                canvas = result["canvas"]
                if peak is not None:
                    peaks[canvas] = max(peaks[canvas], peak)
                window_seconds[canvas].append(result["elapsed_seconds"])
                epoch_counts.setdefault(inp["epoch"], Counter())[canvas] += 1
                row = {"input": inp, "canvas": canvas, "gt_count": int(sum(len(v["labels"]) for v in t)),
                       "loss": result["loss"], "lr": result["lr"],
                       "gradient_norm_before_clip": result["gradient_norm_before_clip"],
                       "update_seconds": result["elapsed_seconds"], "peak_allocated_bytes": peak,
                       "ema_updates": e.ema.updates}
                if result["update"] in previous_rows:
                    S.require(previous_rows.pop(result["update"]) ==
                              {k: v for k, v in row.items() if k not in EXCLUDED_ROW_FIELDS},
                              "MIX recovery replay differs from the recorded trajectory")
                    replayed += 1
                ledger.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                if e.completed_updates % WPE == 0 and not fixture:
                    epoch = e.completed_updates // WPE
                    S.require(dict(epoch_counts.get(epoch, {})) == {k: v for k, v in _quota(epoch).items() if v},
                              f"MIX canvas quota differs at epoch {epoch}: {dict(epoch_counts.get(epoch, {}))}")
                if e.completed_updates in save_points:
                    ledger.flush()
                    os.fsync(ledger.fileno())
                    grad = {"global_before_clip": result["gradient_norm_before_clip"], **monitor}
                    factor = min(1., .1 / (result["gradient_norm_before_clip"] + 1e-6))
                    grad["group_before_clip_from_native_clip_coefficient"] = {
                        k: v / factor for k, v in grad.get("clipped_group_gradient_norm", {}).items()}
                    checkpoint(grad)
                if e.completed_updates % 20 == 0 or fixture:
                    print(json.dumps({"update": e.completed_updates, "canvas": canvas, "loss": result["loss"],
                                      "seconds": result["elapsed_seconds"], **_memory()}), flush=True)
            S.require(e.completed_updates != begin and (fixture or e.completed_updates % WPE == 0),
                      "MIX loader did not close epoch exactly")
            print(f"MIX completed epoch {e.completed_updates // WPE}, updates={e.completed_updates}", flush=True)
    done = e.completed_updates - started_at
    recomputes = S.RECOMPUTE["calls"] - recompute_before
    totals = {k: sum(v.get(k, 0) for v in epoch_counts.values()) for k in ("L", "H")}
    S.require(not previous_rows and e.completed_updates == stop and counts["actual_updates"] == done
              and counts["forwards"] == 2 * done
              and all(g["lr"] == learning_rate(stop) for g in e.optimizer.param_groups),
              "MIX terminal update, forward or LR count differs")
    if not fixture:
        S.require(totals == M.TOTALS and all(dict(epoch_counts[ep]) == {k: v for k, v in _quota(ep).items() if v}
                                             for ep in range(M.FIRST_EPOCH, M.LAST_EPOCH + 1)),
                  f"MIX final canvas totals differ: {totals}")
    S.require(recomputes == (RECOMPUTES_PER_UPDATE * done if c["workload"]["activation_checkpoint"] else 0),
              f"MIX block recomputation count differs: {recomputes}")

    def stats(values):
        if not values:
            return None
        v = sorted(values)
        return {"n": len(v), "sum": sum(v), "mean": sum(v) / len(v), "median": v[len(v) // 2],
                "p95": v[min(len(v) - 1, int(.95 * len(v)))], "max": v[-1]}

    return {"arm": ARM, "global_update": stop, "started_at_update": started_at,
            "effective_updates": stop - PARENT, "actual_updates": counts["actual_updates"],
            "canvas_totals": totals, "canvas_counts_by_epoch": {str(k): dict(v) for k, v in sorted(epoch_counts.items())},
            "peak_allocated_by_canvas_this_process": peaks,
            "train_window_seconds_by_canvas_this_process": {k: stats(v) for k, v in window_seconds.items()},
            "parameter_steps": _steps(e), "EMA_updates": e.ema.updates, "checkpoints": summaries,
            "endpoint": summaries[-1], "restored": restored, "prior_recorded_updates_replayed_exact": replayed,
            "block_recomputations": recomputes, "fixture": fixture, "schedule": schedule.identity}


# ---------------------------------------------------------------- CPU gate
def _native_loader_receipts(c, canvas, completed, limit):
    data = (formal_data.train_dataset(c["data"]["train_core"]) if canvas == "L"
            else S.train_dataset(c["data"]["train_core"], S.S15_HW, seed=0))
    return [r for _, _, r in formal_data.windows(data, completed, limit=limit)]


def _loader_check(c, schedule, a0_rows, s15_rows):
    """Real prefetching MIX loader vs the native same-canvas loader and the historical ledgers."""
    S.require(not torch.cuda.is_initialized(), "loader check before CUDA")
    import random
    import numpy as np
    py, nps, tor = random.getstate(), np.random.get_state(), torch.get_rng_state()
    mix_data = _data(c)
    result = []
    for completed, limit in LOADER_RANGES:
        mixed = [r for _, _, r in M.windows(mix_data, completed, schedule, limit=limit)]
        native = {k: _native_loader_receipts(c, k, completed, limit) for k in ("L", "H")}
        rows = []
        for i, r in enumerate(mixed):
            canvas = r["canvas"]
            other = "H" if canvas == "L" else "L"
            S.require(r["canvas"] == schedule.canvas(r["update"]) and r["actual_hw"] == list(M.CANVAS[canvas]),
                      "MIX loader canvas differs from the table")
            same = M.native_fields(r) == native[canvas][i]
            hist = (a0_rows if canvas == "L" else s15_rows)[r["update"]]["input"]
            historical = M.native_fields(r) == hist
            order_same_other = all(r[k] == native[other][i][k] for k in ("update", "epoch", "window", "image_ids_sha256",
                                                                         "augmentation_seeds_sha256", "augmentation_strong"))
            canvas_fields_differ_other = all(r[k] != native[other][i][k] for k in
                                             ("augmented_samples_sha256", "input_tensor_sha256", "target_sha256"))
            S.require(same and historical and order_same_other and canvas_fields_differ_other,
                      f"MIX loader window {r['update']} ({canvas}) differs from the native same-canvas entry")
            rows.append({"update": r["update"], "epoch": r["epoch"], "window": r["window"], "canvas": canvas,
                         "actual_hw": r["actual_hw"], "worker_canvas": r["worker_canvas"],
                         "equal_native_same_canvas_loader": same, "equal_historical_ledger_row": historical,
                         "order_draws_equal_other_canvas_loader": order_same_other,
                         "canvas_dependent_fields_differ_from_other_canvas": canvas_fields_differ_other,
                         "receipt": r})
        seq = "".join(r["canvas"] for r in rows)
        result.append({"completed_updates": completed, "windows": limit, "canvases": seq, "rows": rows})
    transitions = set()
    for block in result:
        transitions |= {block["canvases"][i:i + 2] for i in range(len(block["canvases"]) - 1)}
    boundary = result[1]["rows"][-1]["canvas"] + result[2]["rows"][0]["canvas"]
    S.require({"LH", "HL"} <= transitions and boundary == "LH", f"MIX loader check transitions {transitions} {boundary}")
    restart = [r for _, _, r in M.windows(mix_data, RESTART_RANGE[0], schedule, limit=RESTART_RANGE[1])]
    S.require(restart == [row["receipt"] for row in result[0]["rows"][1:1 + RESTART_RANGE[1]]],
              "MIX restarted worker input differs from its uninterrupted window")
    final = np.random.get_state()
    S.require(random.getstate() == py and torch.equal(tor, torch.get_rng_state()) and nps[0] == final[0]
              and np.array_equal(nps[1], final[1]) and nps[2:] == final[2:] and not torch.cuda.is_initialized(),
              "MIX loader mutated the parent RNG or initialized CUDA")
    return {"blocks": result, "transitions_covered": sorted(transitions), "epoch117_to_118": boundary,
            "restart_equal": True, "parent_rng_unchanged": True}


def _strip(state, *, size_caches=True, engine_config=True):
    """Complete state minus the declared canvas caches (model and EMA) and engine recipe binding."""
    out = dict(state)
    if size_caches:
        out["model"] = {k: v for k, v in state["model"].items() if not S._is_size_cache(k)}
        ema = dict(state["ema"])
        ema["module"] = {k: v for k, v in state["ema"]["module"].items() if not S._is_size_cache(k)}
        out["ema"] = ema
    if engine_config:
        out["engine"] = {k: v for k, v in state["engine"].items() if k != "config"}
    for k in ("version", "arm", "parent_sha256", "config", "schedule", "activation_checkpoint"):
        out.pop(k, None)
    return out


def _constant_mode(c, canvas):
    """constant-L vs R25 A0 create (768) / constant-H vs S15 create (1152); same parent, same input."""
    from sparse_rtdetr.research.query_local_runtime import capture as r25_capture, create as r25_create
    t0, c0 = time.perf_counter(), time.process_time()
    schedule = M.test_schedule(f"test-constant-{canvas}")
    fixture_cfg = M.canvas_config(M.base_config(True), canvas)
    # MIX wrapper
    mix, _ = _make(c, "cpu", schedule, fixture=True)
    mix_counts = Counter(actual_updates=0, forwards=0, forward_attempts=0)
    _register_counts(mix, mix_counts, f"MIX-constant-{canvas}")
    mix_rows = []
    for x, t, r in M.windows(_data(c), PARENT, schedule, limit=CONSTANT_UPDATES):
        mix_rows.append((r, _update(mix, x[:2], t[:2], r, exact=True)))
    mix_state = M.capture(mix)
    del mix
    gc.collect()
    # original entry at the same canvas
    if canvas == "L":
        ref, _ = r25_create(c["parent"]["path"], "A0", "cpu", config=fixture_cfg, synthetic=True)
        S.require(ref.r25_parent == c["parent"]["sha256"], "R25 entry parent")
        if c["workload"]["activation_checkpoint"]:
            S.install_activation_checkpoint(ref.model)
        snap = r25_capture
    else:
        ref, _ = S.create(c["parent"]["path"], "cpu", fixture_cfg, seed=0, fixture=True,
                          activation_checkpoint=c["workload"]["activation_checkpoint"])
        snap = S.capture
    S.require(ref.config == fixture_cfg, "original entry recipe")
    ref_counts = Counter(actual_updates=0, forwards=0, forward_attempts=0)
    _register_counts(ref, ref_counts, f"original-{canvas}")
    ref_rows = []
    for x, t, r in formal_data.windows(formal_data.train_dataset(c["data"]["train_core"]) if canvas == "L"
                                       else S.train_dataset(c["data"]["train_core"], S.S15_HW, seed=0),
                                       PARENT, limit=CONSTANT_UPDATES):
        ref_rows.append((r, ref.train_window(x[:2], t[:2], capture=True)))
    ref_state = snap(ref)
    del ref
    gc.collect()
    inputs_equal = [M.native_fields(a) == b for (a, _), (b, _) in zip(mix_rows, ref_rows)]
    result_diff = []
    for (_, a), (_, b) in zip(mix_rows, ref_rows):
        result_diff += _differences(_result_body(a, ("elapsed_seconds", "canvas", "input_hw")), _result_body(b))
    strip_caches = canvas == "L"
    state_diff = _differences(_strip(mix_state, size_caches=strip_caches, engine_config=strip_caches),
                              _strip(ref_state, size_caches=strip_caches, engine_config=strip_caches))
    S.require(all(inputs_equal) and len(inputs_equal) == CONSTANT_UPDATES and not result_diff and not state_diff,
              f"MIX constant-{canvas} differs from the original entry: {inputs_equal} {result_diff[:5]} {state_diff[:5]}")
    S.require(mix_counts["actual_updates"] == ref_counts["actual_updates"] == CONSTANT_UPDATES
              and mix_counts["forwards"] == ref_counts["forwards"] == 2 * CONSTANT_UPDATES, "constant-mode counts")
    return {"canvas": canvas, "schedule": schedule.identity, "fixture_config": fixture_cfg.binding(),
            "original_entry": ("R25 query_local_runtime.create(A0) at 768x1344" if canvas == "L"
                               else "S15 scale_baseline.create at 1152x2016"),
            "same_memory_mode_as_MIX": c["workload"]["activation_checkpoint"],
            "updates_each": CONSTANT_UPDATES, "inputs_equal": inputs_equal,
            "results_equal_excluding": ["elapsed_seconds"],
            "results": [_result_body(a) for _, a in mix_rows],
            "complete_state_equal": True,
            "state_comparison_excludes": (["decoder.anchors/valid_mask (model, EMA): MIX engine holds native 1152 caches",
                                           "engine.config (MIX base recipe 1152 vs native 768 recipe)",
                                           "version/arm/schedule labels"] if strip_caches
                                          else ["version/arm/schedule labels"]),
            "counts": {"MIX": dict(mix_counts), "original": dict(ref_counts)},
            "cost": {"wall_seconds": time.perf_counter() - t0, "cpu_seconds": time.process_time() - c0}}


def _expect_refusal(refused, label, fn, kinds=(ValueError, KeyError, RuntimeError)):
    try:
        fn()
    except kinds as exc:
        refused.append({"case": label, "reason": repr(exc)[:300]})
    else:
        raise AssertionError("MIX accepted: " + label)


def _refusals(c, out, schedule, fixture_dir):
    """Wrong parent, table SHA/rule, scope/schedule mode, cursor, actual canvas and microbatch are refused."""
    refused = []
    other_parent = c["refusal_materials"]["other_seed_parent"]["path"]
    s15_endpoint = c["refusal_materials"]["S15_e120"]["path"]
    _expect_refusal(refused, "seed1 e060 parent", lambda: M.create(other_parent, "cpu", schedule, fixture=True))
    _expect_refusal(refused, "S15 e120 checkpoint as parent", lambda: M.create(s15_endpoint, "cpu", schedule, fixture=True))
    table = c["schedule"]["table"]["path"]
    _expect_refusal(refused, "table with wrong SHA", lambda: M.formal_schedule(table, "0" * 64))
    rows = M.generate_rows()
    u, ep, w, cv, d = rows[0]
    flipped = [(u, ep, w, "L" if cv == "H" else "H", d)] + rows[1:]
    swapped = list(rows)
    j = next(i for i in range(1, 304) if rows[i][3] != rows[0][3])
    swapped[0] = rows[0][:3] + (rows[j][3], rows[0][4])
    swapped[j] = rows[j][:3] + (rows[0][3], rows[j][4])  # quota kept, rule violated
    for label, bad in (("table with one flipped canvas (quota broken)", flipped),
                       ("table with two swapped canvases (quota kept, rule violated)", swapped)):
        path = out / ("bad-" + str(len(refused)) + ".tsv")
        raw = M.table_bytes(bad)
        path.write_bytes(raw)
        _expect_refusal(refused, label, lambda p=path, r=raw: M.formal_schedule(p, hashlib.sha256(r).hexdigest()))
    first = fixture_dir / f"fixture-update-{PARENT + 1}.pth"
    _expect_refusal(refused, "cpu-fixture state as formal", lambda: _load(c, first, "formal", schedule))
    _expect_refusal(refused, "formal-table fixture state under the constant-L test schedule",
                    lambda: _load(c, first, "cpu-fixture", M.test_schedule("test-constant-L"), fixture=True))
    _expect_refusal(refused, "test schedule bound as formal state",
                    lambda: _state_binding(c, PARENT, "formal", None, M.test_schedule("test-constant-H")))
    # cursor: sidecar of u18241 relabelled as u18242
    side = json.loads(first.with_suffix(".pth.json").read_text())
    moved = out / "cursor-relabel.pth"
    shutil.copyfile(first, moved)
    forged = {**side, "global_update": PARENT + 2, "cursor": {**cursor(PARENT + 2), "next_canvas": schedule.canvas(PARENT + 3)},
              "audit": None, "file": _file(moved)}
    moved.with_suffix(".pth.json").write_text(json.dumps(forged))
    _expect_refusal(refused, "state with a relabelled cursor", lambda: _load(c, moved, "cpu-fixture", schedule, fixture=True))
    # actual canvas / cursor at the engine: window rendered at the wrong canvas, receipt of another update
    e, _ = _make(c, "cpu", schedule, fixture=True)
    data = _data(c)
    (xh, th, rh), = list(M.windows(data, PARENT, schedule, limit=1))            # u18241 = H
    before = M.capture(e)
    wrong = M.test_schedule("test-constant-L")
    (xl, tl, rl), = list(M.windows(data, PARENT, wrong, limit=1))              # same window rendered at L
    _expect_refusal(refused, "L-rendered window at a scheduled H update", lambda: M.train_window(e, xl[:2], tl[:2], rl))
    _expect_refusal(refused, "H tensor with an L receipt",
                    lambda: M.train_window(e, xh[:2], th[:2], {**rh, "canvas": "L", "expected_hw": list(M.L_HW)}))
    _expect_refusal(refused, "receipt of another update (cursor)",
                    lambda: M.train_window(e, xh[:2], th[:2], {**rh, "update": PARENT + 2}))
    _expect_refusal(refused, "single-image window (microbatch differs)", lambda: M.train_window(e, xh[:1], th[:1], rh))
    S.require(not e.failed and e.completed_updates == PARENT and ck.exact_equal(M.capture(e), before),
              "a refused MIX window changed the engine")
    del e, xh, th, xl, tl
    gc.collect()
    _expect_refusal(refused, "4x4 microbatch recipe",
                    lambda: S.validate_production(Config(seed=0, input_hw=S.S15_HW, physical_batch_size=4)))
    _expect_refusal(refused, "MIX engine without schedule", lambda: M.create(c["parent"]["path"], "cpu", None, fixture=True))
    _expect_refusal(refused, "acceptance schedule outside its three updates",
                    lambda: M.test_schedule("acceptance-LHL").canvas(PARENT + 4))
    return refused


def _cpu(c, out):
    out = Path(out)
    S.require(not torch.cuda.is_initialized(), "MIX CPU gate initialized CUDA")
    t0, c0 = time.perf_counter(), time.process_time()
    S.require(c["workload"]["checkpoint_segments"] == list(S.CHECKPOINT_SEGMENTS)
              and c["workload"]["canvases"] == {k: list(v) for k, v in M.CANVAS.items()}, "MIX contract workload")
    schedule = formal_schedule(c)
    table = {"identity": schedule.identity, "first_e061": "".join(schedule.canvas(PARENT + 1 + w) for w in range(20)),
             "last_e117": "".join(schedule.canvas(117 * WPE - 8 + 1 + i) for i in range(8))}
    a0_rows, s15_rows = _ledger(c["history"]["A0_updates"]), _ledger(c["history"]["S15_updates"])
    loader = _loader_check(c, schedule, a0_rows, s15_rows)
    loader_cost = {"wall_seconds": time.perf_counter() - t0, "cpu_seconds": time.process_time() - c0}
    del a0_rows, s15_rows
    gc.collect()
    constant = {k: _constant_mode(c, k) for k in ("L", "H")}
    # Real formal loop on the frozen table (fixture), then exact resume from u1 replaying the rest.
    t1, c1 = time.perf_counter(), time.process_time()
    first, second = out / "fixture-run", out / "fixture-resume"
    first.mkdir(); second.mkdir()
    counts1 = Counter(actual_updates=0, forwards=0, forward_attempts=0)
    run1 = _formal(c, first, counts1, None, None, fixture=True)
    gc.collect()
    counts2 = Counter(actual_updates=0, forwards=0, forward_attempts=0)
    run2 = _formal(c, second, counts2, first / f"fixture-update-{PARENT + 1}.pth", c, fixture=True)
    stop = PARENT + FIXTURE_UPDATES
    end1 = torch.load(first / f"fixture-update-{stop}.pth", map_location="cpu", weights_only=True)
    end2 = torch.load(second / f"fixture-update-{stop}.pth", map_location="cpu", weights_only=True)
    end1.pop("formal_binding_sha256"); end2.pop("formal_binding_sha256")
    S.require(ck.exact_equal(end1, end2), "MIX fixture resumed endpoint state differs")
    del end1, end2
    S.require(run2["prior_recorded_updates_replayed_exact"] == FIXTURE_UPDATES - 1
              and counts2["actual_updates"] == FIXTURE_UPDATES - 1, "MIX fixture resume count")
    rows1 = [json.loads(x) for x in (first / "UPDATES.jsonl").read_text().splitlines()]
    rows2 = [json.loads(x) for x in (second / "UPDATES.jsonl").read_text().splitlines()]
    strip = lambda r: {k: v for k, v in r.items() if k not in EXCLUDED_ROW_FIELDS}
    S.require(len(rows1) == len(rows2) == FIXTURE_UPDATES and [strip(r) for r in rows1] == [strip(r) for r in rows2],
              "MIX fixture ledgers differ")
    fixture_canvases = "".join(r["canvas"] for r in rows1)
    S.require(fixture_canvases == "".join(schedule.canvas(PARENT + 1 + i) for i in range(FIXTURE_UPDATES))
              and {"HL", "LH"} <= {fixture_canvases[i:i + 2] for i in range(len(fixture_canvases) - 1)}
              and [r["input"] for r in rows1] == [row["receipt"] for row in loader["blocks"][0]["rows"][:FIXTURE_UPDATES]],
              "MIX fixture loop canvases/inputs differ from the table and loader check")
    loop_cost = {"wall_seconds": time.perf_counter() - t1, "cpu_seconds": time.process_time() - c1}
    # Production e060 bridge on CPU; formal-binding save, load, exact restore (state never formal).
    t2 = time.perf_counter()
    e, bridge = _make(c, "cpu", schedule)
    S.require(e.config == M.base_config() and e.s15_activation_checkpoint is True
              and len(bridge["activation_checkpoint_segments"]) == BLOCKS, "MIX production engine")
    path = out / "cpu-production-state.pth"
    _save(e, c, path, "cpu-fixture", schedule)
    side, state = _load(c, path, "cpu-fixture", schedule)
    restored = _restore_engine(e, path, side, state)
    del state, e
    gc.collect()
    refused = _refusals(c, out, schedule, first)
    refused_more = []
    _expect_refusal(refused_more, "production cpu state as formal", lambda: _load(c, path, "formal", schedule))
    _expect_refusal(refused_more, "production cpu state as acceptance", lambda: _load(c, path, "acceptance", schedule))
    refused += refused_more
    S.require(not torch.cuda.is_initialized(), "MIX CPU gate initialized CUDA")
    fixture_updates = (counts1["actual_updates"] + counts2["actual_updates"]
                       + sum(v["counts"]["MIX"]["actual_updates"] + v["counts"]["original"]["actual_updates"]
                             for v in constant.values()))
    fixture_forwards = (counts1["forwards"] + counts2["forwards"]
                        + sum(v["counts"]["MIX"]["forwards"] + v["counts"]["original"]["forwards"] for v in constant.values()))
    return {"table": table, "loader_check": loader, "constant_mode": constant,
            "fixture_formal_loop": {"config": M.base_config(True).binding(), "canvases": fixture_canvases,
                                    "first": run1, "first_counts": dict(counts1), "resume": run2,
                                    "resume_counts": dict(counts2), "resumed_endpoint_complete_state_exact": True},
            "production_bridge": bridge, "production_restore": restored, "refused": refused,
            "cpu_fixture_updates": fixture_updates, "cpu_fixture_training_forwards": fixture_forwards,
            "cost": {"loader_and_table": loader_cost, "fixture_loop": loop_cost,
                     "bridge_and_refusals_wall_seconds": time.perf_counter() - t2},
            "CUDA_initialized": torch.cuda.is_initialized()}


# ---------------------------------------------------------------- CUDA acceptance
def _accept(c, out, counts):
    """Acceptance-only L->H->L on data windows 0,1,2 (production 8x2/BF16, activation checkpoint)."""
    out = Path(out)
    t1 = time.perf_counter()
    schedule = M.test_schedule("acceptance-LHL")
    a0_rows, s15_rows = _ledger(c["history"]["A0_updates"]), _ledger(c["history"]["S15_updates"])
    data = _data(c)
    _reset_peak()
    e, bridge = _make(c, "cuda:0", schedule)
    _register_counts(e, counts, "MIX-accept")
    seen, criterion = _precision_observers(e)
    parent_bn = _bn_counts(e.model)
    rec0, sec0 = S.RECOMPUTE["calls"], S.RECOMPUTE["seconds"]
    S.RECOMPUTE["timed"] = True
    direct = []
    for i, (x, t, inp) in enumerate(M.windows(data, PARENT, schedule, limit=3), 1):
        _reset_peak()
        r = _update(e, x, t, inp, exact=True)
        direct.append({"input": inp, "update": r, **_memory()})
        print(json.dumps({"phase": "MIX-direct", "update": r["update"], "canvas": r["canvas"], "loss": r["loss"],
                          **_memory()}), flush=True)
        if i == 1:
            saved = _save(e, c, out / "u1.pth", "acceptance", schedule)
    S.RECOMPUTE["timed"] = False
    expected = M.capture(e)
    bn_delta = sorted(set(v - parent_bn[k] for k, v in _bn_counts(e.model).items()))
    S.require(bn_delta == [6], f"MIX BN updates differ from the native forward count: {bn_delta}")
    S.require(len(seen) == 6 and all(v["autocast"] and v["autocast_dtype"] == "torch.bfloat16"
                                     and v["input"] == "torch.float32" for v in seen)
              and [v["shape"][-2:] for v in seen] == [list(M.L_HW)] * 2 + [list(M.H_HW)] * 2 + [list(M.L_HW)] * 2
              and criterion == [["torch.float32"]] * 6, "MIX CUDA precision/canvas differs")
    first_recompute = {"calls": S.RECOMPUTE["calls"] - rec0, "seconds_synchronized": S.RECOMPUTE["seconds"] - sec0}
    # Engineering regression: inputs equal the historical same-canvas ledgers; u1 (L from the parent) equals R25 A0 u18241.
    history = []
    for row in direct:
        inp, r = row["input"], row["update"]
        hist = (a0_rows if inp["canvas"] == "L" else s15_rows)[inp["update"]]
        history.append({"update": inp["update"], "canvas": inp["canvas"],
                        "input_equal_historical_same_canvas_row": M.native_fields(inp) == hist["input"]})
    a0_first = a0_rows[PARENT + 1]
    u1_equal_a0 = (direct[0]["update"]["loss"], direct[0]["update"]["lr"], direct[0]["update"]["gradient_norm_before_clip"]) \
        == (a0_first["loss"], a0_first["lr"], a0_first["gradient_norm_before_clip"])
    S.require(all(h["input_equal_historical_same_canvas_row"] for h in history) and u1_equal_a0,
              f"MIX acceptance inputs/u1 differ from history: {history} u1={u1_equal_a0}")
    del e
    gc.collect()
    torch.cuda.empty_cache()
    _reset_peak()
    clone, _ = _make(c, "cuda:0", schedule)
    _register_counts(clone, counts, "MIX-replay")
    PHASE["phase"] = "MIX:restore"
    side, state = _load(c, out / "u1.pth", "acceptance", schedule)
    restored = _restore_engine(clone, out / "u1.pth", side, state)
    del state
    replay, diffs = [], []
    for k, (x, t, inp) in enumerate(M.windows(data, PARENT + 1, schedule, limit=2), 1):
        _reset_peak()
        r = _update(clone, x, t, inp, exact=True)
        replay.append({"input": inp, "update": _result_body(r), **_memory()})
        diffs += _differences(_result_body(direct[k]["update"]), _result_body(r))
        S.require(inp == direct[k]["input"], "MIX replay input differs")
    replay_state = M.capture(clone)
    diffs += _differences(expected, replay_state)
    S.require(not diffs and ck.exact_equal(expected, replay_state), f"MIX CUDA replay differs: {diffs[:5]}")
    del expected, replay_state, clone
    recomputes = S.RECOMPUTE["calls"] - rec0
    S.require(recomputes == RECOMPUTES_PER_UPDATE * 5, f"MIX recomputation count {recomputes}")
    S.require(counts["actual_updates"] == 5 and counts["forwards"] == 10, "MIX acceptance count")
    peaks = {"L": max(r["peak_allocated_bytes"] for r in direct + replay if r["input"]["canvas"] == "L"),
             "H": max(r["peak_allocated_bytes"] for r in direct + replay if r["input"]["canvas"] == "H")}
    acc = c["workload"]["acceptance"]
    result = {"schedule": schedule.identity, "bridge": bridge, "direct_updates": direct, "saved_u1": saved,
              "restore": restored, "replay_updates": replay, "replay_results_and_complete_state_exact": True,
              "inputs_vs_historical_ledgers": history, "u1_L_equals_R25_A0_u18241_loss_lr_gradnorm": u1_equal_a0,
              "peak_allocated_by_canvas": peaks, "block_recomputations": recomputes,
              "block_recompute_direct_three_updates": first_recompute,
              "precision": {"model": seen, "criterion": criterion},
              "BN_num_batches_tracked_delta_direct_three_updates": bn_delta,
              "totals": {"optimizer_updates": counts["actual_updates"], "training_forwards": counts["forwards"],
                         "full_model_forwards": counts["forwards"], "block_recomputations": recomputes},
              "device_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
              "allocator_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"), "stage_seconds": time.perf_counter() - t1,
              "formal_state_usable": False}
    S.require(result["totals"]["optimizer_updates"] == acc["updates"]
              and result["totals"]["full_model_forwards"] == acc["training_forwards"], "MIX acceptance totals differ")
    return result


# ---------------------------------------------------------------- CLI
def main():
    from mix_formal_run import load_contract
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("cpu", "accept", "train"))
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--prior-contract", type=Path)
    args = parser.parse_args()
    if args.action != "train" and (args.resume or args.prior_contract):
        parser.error("resume is available only for train")
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    cpu0 = time.process_time()
    counts = Counter(actual_updates=0, forwards=0, forward_attempts=0)
    device = "cpu" if args.action == "cpu" else "cuda:0"
    c = None
    try:
        c = load_contract(args.contract)
        S.require(Path(c["source_root"]).resolve() == ROOT.resolve(), "MIX trainer is not running from the bound release")
        S.require(formal_data.SEED == 0 and cursor(PARENT)["seed"] == 0, "MIX loader seed binding")
        if device == "cpu":
            S.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "MIX CPU gate requires hidden CUDA")
        else:
            S.require(os.environ.get("PYTORCH_CUDA_ALLOC_CONF") == c["workload"]["allocator_conf"],
                      "MIX allocator configuration differs from contract")
        # Deterministic backend flags; every engine replaces the global RNGs by the parent (or restart) RNG.
        setup(device)
        if device == "cpu":
            torch.set_num_threads(int(os.environ.get("MIX_CPU_THREADS", "8")))
        else:
            torch.cuda.reset_peak_memory_stats()
        if args.action == "cpu":
            result = _cpu(c, args.out)
        elif args.action == "accept":
            result = _accept(c, args.out, counts)
        else:
            prior = load_contract(args.prior_contract) if args.prior_contract else None
            result = _formal(c, args.out, counts, args.resume, prior)
        _json(args.out / "CLOSED.json", {"status": "PASS", "action": args.action, **result,
                                        "contract_sha256": c["contract_sha256"], "counts": dict(counts),
                                        "elapsed_seconds": time.monotonic() - start,
                                        "process_cpu_seconds": time.process_time() - cpu0, **_memory()})
    except BaseException as exc:
        oom = isinstance(exc, torch.cuda.OutOfMemoryError)
        _json(args.out / "STOP.json", {"status": "STOP_CUDA_OOM" if oom else "STOP", "phase": PHASE["phase"],
                                      "error": repr(exc)[:4000], "traceback": traceback.format_exc(),
                                      "action": args.action, "counts": dict(counts),
                                      "attempted_forwards": counts["forward_attempts"],
                                      "completed_forwards": counts["forwards"],
                                      "actual_optimizer_updates": counts["actual_updates"],
                                      "block_recomputations": S.RECOMPUTE["calls"],
                                      "contract_sha256": c["contract_sha256"] if c else None,
                                      "elapsed_seconds": time.monotonic() - start,
                                      "process_cpu_seconds": time.process_time() - cpu0,
                                      "allocator_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"), **_memory()})
        raise


if __name__ == "__main__":
    main()
