"""S15 formal training, CUDA engineering acceptance and complete restart state (TASK_S15_EXECUTE_001).

actions (always launched by s15_formal_run.py from a frozen release with a bound contract):
  cpu     CUDA hidden.  Production e060 bridge on CPU, formal-binding save/load/exact restore and
          refusals, audit prefix, and the real formal loop on a 1152x2016 FP32 bs1x2 fixture:
          2 updates with checkpoints, then a resumed run that replays the second update exactly.
  accept  CUDA.  (4.1) 768x1344 A0 plain vs activation-checkpoint, 2 real production updates each,
          complete per-tensor comparison; (4.2) S15 checkpoint mode 2 updates, save u1, fresh engine
          restore + replay u2 (complete state exact), 6 timing updates, 4 EMA FP32 inference forwards.
          Acceptance states are never formal.
  train   CUDA.  Formal S15 e061-e120 from the certified e060 parent (or exact restart).
"""
from __future__ import annotations

import os
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
import argparse
import contextlib
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
from formal_data import END_UPDATE, PARENT_UPDATE, WINDOWS_PER_EPOCH, cursor, windows
from formal_data import train_dataset as native_train_dataset
from sparse_rtdetr.protocol_r18 import checkpoint as ck
from sparse_rtdetr.protocol_r18.engine import assert_healthy, learning_rate
from sparse_rtdetr.protocol_r18.model import Config
from sparse_rtdetr.protocol_r18.runio import fsync_directory
from sparse_rtdetr.research import scale_baseline as S

ARM = "S15"
SAVE_EPOCHS = (60, 75, 90, 105, 120)
SCHEMA = "s15-complete-state-v1"
FIXTURE = Config(seed=0, input_hw=S.S15_HW, physical_batch_size=1, amp_dtype="float32")
FIXTURE_STOP = PARENT_UPDATE + 2
BLOCKS = 8                       # 4 backbone stages + 2 FPN + 2 PAN blocks
RECOMPUTES_PER_UPDATE = 2 * BLOCKS
A0_UPDATES, S15_FIRST, S15_REPLAY, S15_TIMING = 2, 2, 1, 6
EVAL_IMAGES, EVAL_BATCH = 8, 4
# Declared non-scientific fields excluded from equality: timing and the implementation flag.
EXCLUDED_RESULT_FIELDS = ("elapsed_seconds",)
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
                raise ValueError("incomplete S15 audit prefix")
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


# ---------------------------------------------------------------- state binding
def _cfg(fixture):
    return FIXTURE if fixture else S.production_config()


def _state_binding(c, update, scope, audit, fixture=False):
    return {"schema": SCHEMA, "scope": scope, "arm": ARM, "formal_identity": c["formal_identity"],
            "contract_sha256": c["contract_sha256"], "global_update": update,
            "parent_sha256": c["parent"]["sha256"], "cursor": cursor(update), "audit": audit,
            "config": _cfg(fixture).binding(), "activation_checkpoint": c["workload"]["activation_checkpoint"],
            "torch": str(torch.__version__)}


def _save(e, c, path, scope, audit=None, fixture=False):
    path = Path(path)
    S.require(e.ema.updates == e.completed_updates, "S15 EMA clock disagrees at save")
    assert_healthy(e.model, e.optimizer, e.ema)
    state = S.capture(e)
    binding = _state_binding(c, e.completed_updates, scope, audit, fixture)
    state["formal_binding_sha256"] = hashlib.sha256(ck.canonical(binding)).hexdigest()
    _atomic(path, lambda stream: torch.save(state, stream))
    result = {"status": "COMPLETE", **binding, "file": _file(path)}
    _json(path.with_suffix(".pth.json"), result)
    return result


def _verify_audit(audit, path):
    S.require(audit["path"] == str(Path(path).resolve()) and type(audit["bytes"]) is int and audit["bytes"] >= 0
              and audit["sha256"] == _sha_prefix(path, audit["bytes"]), "S15 formal audit prefix differs")


def _load(c, path, scope, *, endpoint=False, fixture=False):
    path = Path(path).resolve()
    side = json.loads(path.with_suffix(".pth.json").read_text())
    update = side["global_update"]
    binding = _state_binding(c, update, scope, side.get("audit"), fixture)
    S.require(side == {"status": "COMPLETE", **binding, "file": _file(path)}, "S15 state receipt or file identity differs")
    if scope == "formal":
        S.require(update in tuple(ep * WINDOWS_PER_EPOCH for ep in SAVE_EPOCHS), "S15 checkpoint not at a registered boundary")
        S.require(not endpoint or update == END_UPDATE, "S15 evaluation requires e120")
    if side.get("audit") is not None:
        _verify_audit(side["audit"], path.parent / "UPDATES.jsonl")
    s = torch.load(path, map_location="cpu", weights_only=True)
    S.require(s.pop("formal_binding_sha256") == hashlib.sha256(ck.canonical(binding)).hexdigest()
              and s["version"] == S.VERSION and s["arm"] == ARM and s["parent_sha256"] == c["parent"]["sha256"]
              and s["config"] == _cfg(fixture).binding()
              and s["activation_checkpoint"] == c["workload"]["activation_checkpoint"]
              and s["engine"]["completed_updates"] == update and s["ema"]["updates"] == update
              and s["engine"]["next_epoch"] == side["cursor"]["next_epoch"]
              and s["engine"]["next_window"] == side["cursor"]["next_window"],
              "S15 checkpoint payload identity/clock differs")
    return side, s


def read_endpoint(contract, path):
    """Verified e120 S15 payload (model, optimizer, EMA, engine, RNG)."""
    return _load(contract, path, "formal", endpoint=True)[1]


def _restore_engine(e, path, side, state):
    S.require(state["activation_checkpoint"] == e.s15_activation_checkpoint, "S15 memory mode differs at restore")
    S.require(state["optimizer_names"] == [g["param_names"] for g in e.optimizer.param_groups], "S15 optimizer names")
    e.model.load_state_dict(state["model"], strict=True)
    e.optimizer.load_state_dict(state["optimizer"])
    e.ema.load_state_dict(state["ema"])
    modes = dict(e.model.named_modules())
    S.require(set(modes) == set(state["model_modes"]), "S15 training mode keys differ")
    for name, mode in state["model_modes"].items():
        modes[name].training = mode
    e.load_state_dict(state["engine"])
    ck.restore_rng(state["rng"], e.device)
    S.require(ck.exact_equal(S.capture(e), state), "S15 full state did not restore exactly")
    return {"full_state_exact": True, "path": str(path), "global_update": side["global_update"],
            "sha256": side["file"]["sha256"], "cursor": side["cursor"]}


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
        seen.append({"input": str(args[0].dtype), "autocast": torch.is_autocast_enabled("cuda"),
                     "autocast_dtype": str(torch.get_autocast_dtype("cuda")),
                     "logits": str(result["pred_logits"].dtype), "boxes": str(result["pred_boxes"].dtype)})

    e.model.register_forward_hook(model_hook)
    e.criterion.register_forward_hook(lambda _c, _a, r: criterion.append(sorted({str(v.dtype) for v in r.values()})))
    return seen, criterion


def _bn(model):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]


def _bn_counts(model):
    return {n: int(m.num_batches_tracked) for n, m in _bn(model)}


@contextlib.contextmanager
def _pre_clip_gradients(e):
    """Record every parameter gradient exactly as handed to the native clip (CPU copies)."""
    names = {id(p): n for n, p in e.model.named_parameters()}
    original = torch.nn.utils.clip_grad_norm_
    seen = {}

    def wrapper(parameters, *args, **kwargs):
        params = list(parameters)
        for p in params:
            seen[names[id(p)]] = None if p.grad is None else p.grad.detach().cpu().clone()
        return original(params, *args, **kwargs)

    torch.nn.utils.clip_grad_norm_ = wrapper
    try:
        yield seen
    finally:
        torch.nn.utils.clip_grad_norm_ = original


def _differences(a, b, path="", out=None, limit=200):
    """Every differing leaf (tensor/scalar/structure) between two nested values."""
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


def _result_body(result):
    return {k: v for k, v in result.items() if k not in EXCLUDED_RESULT_FIELDS}


# ---------------------------------------------------------------- engines/data
def _make(c, device, *, fixture=False):
    PHASE["phase"] = "S15:construct"
    return S.create(c["parent"]["path"], device, _cfg(fixture) if fixture else None, fixture=fixture,
                    activation_checkpoint=c["workload"]["activation_checkpoint"])


def _data(c):
    return S.train_dataset(c["data"]["train_core"], S.S15_HW)


def _update(e, images, targets, audit, *, exact=False):
    result = e.train_window(images, targets, capture=exact)
    S.require(result["update"] == audit["update"] and result["epoch"] == audit["epoch"]
              and result["window"] == audit["window"] and result["finite"] and e.ema.updates == e.completed_updates,
              "S15 update/input/EMA clock differs")
    return result


def _steps(e):
    result = {}
    for group in e.optimizer.param_groups:
        counts = [int(e.optimizer.state.get(p, {}).get("step", 0)) for p in group["params"]]
        S.require(all(n == e.parameter_steps[name] for n, name in zip(counts, group["param_names"])),
                  "S15 Adam step ledger differs at boundary")
        result[group["name"]] = {"steps_histogram": dict(sorted(Counter(counts).items())), "lr": group["lr"],
                                 "parameters": len(counts)}
    return result


def _initial_snapshot(e):
    return {n: p.detach().cpu().clone() for n, p in e.model.named_parameters()}


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
        raise RuntimeError(f"insufficient disk for next S15 save: {free} bytes, need {needed}")


# ---------------------------------------------------------------- formal loop (also the CPU fixture)
def _formal(c, out, counts, resume, prior, *, fixture=False):
    """Formal loop.  ``fixture`` runs the identical loop on CPU with the FP32 bs1x2 canvas fixture
    (first two images of each real window), stopping at u18242 and saving every update."""
    out = Path(out)
    if (resume is None) != (prior is None):
        raise ValueError("resume requires both checkpoint and prior contract")
    device = "cpu" if fixture else "cuda:0"
    scope = "cpu-fixture" if fixture else "formal"
    stop = FIXTURE_STOP if fixture else END_UPDATE
    save_points = set(range(PARENT_UPDATE, FIXTURE_STOP + 1)) if fixture else {ep * WINDOWS_PER_EPOCH for ep in SAVE_EPOCHS}
    e, bridge = _make(c, device, fixture=fixture)
    _register_counts(e, counts, "S15-formal" if not fixture else "S15-cpu-fixture")
    _json(out / "BRIDGE.json", bridge)
    initial = _initial_snapshot(e)
    history = out / "UPDATES.jsonl"
    restored = None
    previous_rows = {}
    if resume is not None:
        prior_ref = (c.get("prior") or {}).get("contract")
        if fixture:
            S.require(prior["contract_sha256"] == c["contract_sha256"], "fixture resume uses the same contract")
        else:
            S.require(prior_ref == _file(prior["contract_path"])
                      and Path(resume).resolve().parent == Path(prior["attempt_root"]) / "train-S15"
                      and prior["formal_identity"] == c["formal_identity"]
                      and prior["contract_sha256"] != c["contract_sha256"],
                      "S15 restart must preserve scientific identity with a distinct attempt")
        side, state = _load(prior, resume, scope, fixture=fixture)
        PHASE["phase"] = "S15:restore"
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
        with Path(side["audit"]["path"]).open("rb") as source:
            source.seek(side["audit"]["bytes"])
            for line in source:
                if not line.endswith(b"\n"):
                    break  # an interrupted final append is not a committed row
                row = json.loads(line)
                update = row["input"]["update"]
                S.require(update == e.completed_updates + len(previous_rows) + 1, "S15 previous replay evidence out of order")
                previous_rows[update] = {k: v for k, v in row.items() if k != "update_seconds"}
        _json(out / "RESTORE.json", {**restored, "rows_available_for_replay": len(previous_rows)})
    else:
        history.touch(exist_ok=False)
    started_at = e.completed_updates
    recompute_before = S.RECOMPUTE["calls"]
    save_bytes = c["parent"]["bytes"]
    summaries = []
    replayed = 0
    monitor = {}

    def measure(*_):
        if e.completed_updates + 1 in save_points:
            monitor.update(_gradient_groups(e))

    e.optimizer.register_step_pre_hook(measure)

    def checkpoint(grad):
        nonlocal save_bytes
        _available(out, save_bytes)
        name = (f"fixture-update-{e.completed_updates}.pth" if fixture
                else f"checkpoint-epoch-{e.completed_updates // WINDOWS_PER_EPOCH:03d}.pth")
        p = out / name
        receipt = _save(e, c, p, scope, _file(history), fixture)
        save_bytes = receipt["file"]["bytes"]
        summaries.append(receipt)
        _json(p.with_suffix(".observations.json"), {
            "parameter_steps": _steps(e), "parameter_change_from_e060": _relative_change(e, initial),
            "gradient_last_update": grad, "EMA_updates": e.ema.updates,
            "block_recomputations_this_process": S.RECOMPUTE["calls"] - recompute_before})

    if resume is None:
        checkpoint(None)
    dataset = _data(c)
    with history.open("a", buffering=1) as ledger:
        while e.completed_updates < stop:
            _available(out, save_bytes)
            begin = e.completed_updates
            limit = (stop - begin) if fixture else None
            for x, t, inp in windows(dataset, begin, limit=limit):
                if fixture:
                    x, t = x[:2], t[:2]
                monitor.clear()
                result = _update(e, x, t, inp)
                row = {"input": inp, "gt_count": int(sum(len(v["labels"]) for v in t)),
                       "loss": result["loss"], "lr": result["lr"],
                       "gradient_norm_before_clip": result["gradient_norm_before_clip"],
                       "update_seconds": result["elapsed_seconds"], "ema_updates": e.ema.updates}
                if result["update"] in previous_rows:
                    S.require(previous_rows.pop(result["update"]) == {k: v for k, v in row.items() if k != "update_seconds"},
                              "S15 recovery replay differs from the recorded trajectory")
                    replayed += 1
                ledger.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                if e.completed_updates in save_points:
                    ledger.flush()
                    os.fsync(ledger.fileno())
                    grad = {"global_before_clip": result["gradient_norm_before_clip"], **monitor}
                    factor = min(1., .1 / (result["gradient_norm_before_clip"] + 1e-6))
                    grad["group_before_clip_from_native_clip_coefficient"] = {
                        k: v / factor for k, v in grad.get("clipped_group_gradient_norm", {}).items()}
                    checkpoint(grad)
                if e.completed_updates % 20 == 0 or fixture:
                    print(json.dumps({"update": e.completed_updates, "loss": result["loss"],
                                      "seconds": result["elapsed_seconds"], **_memory()}), flush=True)
            S.require(e.completed_updates != begin and (fixture or e.completed_updates % WINDOWS_PER_EPOCH == 0),
                      "S15 loader did not close epoch exactly")
            print(f"S15 completed epoch {e.completed_updates // WINDOWS_PER_EPOCH}, updates={e.completed_updates}", flush=True)
    done = e.completed_updates - started_at
    recomputes = S.RECOMPUTE["calls"] - recompute_before
    S.require(not previous_rows and e.completed_updates == stop and counts["actual_updates"] == done
              and counts["forwards"] == 2 * done
              and all(g["lr"] == learning_rate(stop) for g in e.optimizer.param_groups),
              "S15 terminal update, forward or LR count differs")
    S.require(recomputes == (RECOMPUTES_PER_UPDATE * done if c["workload"]["activation_checkpoint"] else 0),
              f"S15 block recomputation count differs: {recomputes}")
    return {"arm": ARM, "global_update": stop, "started_at_update": started_at,
            "effective_updates": stop - PARENT_UPDATE, "actual_updates": counts["actual_updates"],
            "parameter_steps": _steps(e), "EMA_updates": e.ema.updates, "checkpoints": summaries,
            "endpoint": summaries[-1], "restored": restored, "prior_recorded_updates_replayed_exact": replayed,
            "block_recomputations": recomputes, "fixture": fixture}


# ---------------------------------------------------------------- CPU gate
def _cpu(c, out):
    out = Path(out)
    S.require(not torch.cuda.is_initialized(), "S15 CPU gate initialized CUDA")
    t0, c0 = time.perf_counter(), time.process_time()
    S.require(c["workload"]["checkpoint_segments"] == list(S.CHECKPOINT_SEGMENTS)
              and c["workload"]["input_hw"] == list(S.S15_HW), "S15 contract memory segments or canvas differ")
    # Production bridge and formal-binding save/load/exact restore on CPU.
    e, bridge = _make(c, "cpu")
    S.require(e.config == S.production_config() and e.s15_activation_checkpoint is True
              and bridge["activation_checkpoint_segments"] and len(bridge["activation_checkpoint_segments"]) == BLOCKS,
              "S15 production engine memory mode")
    path = out / "cpu-production-state.pth"
    _save(e, c, path, "cpu-fixture")
    side, state = _load(c, path, "cpu-fixture")
    restored = _restore_engine(e, path, side, state)
    del state, e
    gc.collect()
    refused = []
    for scope, fixture in (("formal", False), ("acceptance", False), ("cpu-fixture", True)):
        try:
            _load(c, path, scope, fixture=fixture)
        except (ValueError, KeyError):
            refused.append({"scope": scope, "fixture_config": fixture})
        else:
            raise AssertionError("S15 cross-scope or cross-config state accepted")
    history = out / "fixture-UPDATES.jsonl"
    with history.open("xb") as stream:
        stream.write(b'{"input":{"update":18241}}\n')
    audit = _file(history)
    with history.open("ab") as stream:
        stream.write(b'{"input":{"update":18242}}\n')
    _verify_audit(audit, history)
    try:
        _verify_audit({**audit, "sha256": "0" * 64}, history)
    except ValueError:
        pass
    else:
        raise AssertionError("S15 audit prefix accepted wrong SHA")
    bridge_cost = {"wall_seconds": time.perf_counter() - t0, "cpu_seconds": time.process_time() - c0}
    # Real formal loop on the fixture: 2 updates, then exact resume replaying the second.
    t1, c1 = time.perf_counter(), time.process_time()
    first, second = out / "fixture-run", out / "fixture-resume"
    first.mkdir(); second.mkdir()
    counts1 = Counter(actual_updates=0, forwards=0, forward_attempts=0)
    run1 = _formal(c, first, counts1, None, None, fixture=True)
    gc.collect()
    counts2 = Counter(actual_updates=0, forwards=0, forward_attempts=0)
    run2 = _formal(c, second, counts2, first / f"fixture-update-{PARENT_UPDATE + 1}.pth", c, fixture=True)
    end1 = torch.load(first / f"fixture-update-{FIXTURE_STOP}.pth", map_location="cpu", weights_only=True)
    end2 = torch.load(second / f"fixture-update-{FIXTURE_STOP}.pth", map_location="cpu", weights_only=True)
    end1.pop("formal_binding_sha256"); end2.pop("formal_binding_sha256")
    S.require(ck.exact_equal(end1, end2), "S15 fixture resumed endpoint state differs")
    S.require(run2["prior_recorded_updates_replayed_exact"] == 1 and counts2["actual_updates"] == 1, "S15 fixture resume count")
    rows1 = (first / "UPDATES.jsonl").read_text().splitlines()
    rows2 = (second / "UPDATES.jsonl").read_text().splitlines()
    S.require(len(rows1) == 2 and rows2[0] == rows1[0] and json.loads(rows2[1])["loss"] == json.loads(rows1[1])["loss"],
              "S15 fixture ledgers differ")
    del end1, end2
    loop_cost = {"wall_seconds": time.perf_counter() - t1, "cpu_seconds": time.process_time() - c1}
    S.require(not torch.cuda.is_initialized(), "S15 CPU gate initialized CUDA")
    return {"production_bridge": bridge, "production_restore": restored, "refused": refused,
            "audit_prefix_after_append_exact": True, "bad_audit_SHA_refused": True,
            "fixture_formal_loop": {"config": FIXTURE.binding(), "first": run1, "first_counts": dict(counts1),
                                    "resume": run2, "resume_counts": dict(counts2),
                                    "resumed_endpoint_complete_state_exact": True},
            "cpu_fixture_updates": counts1["actual_updates"] + counts2["actual_updates"],
            "cpu_fixture_training_forwards": counts1["forwards"] + counts2["forwards"],
            "cost": {"bridge_and_state": bridge_cost, "fixture_loop": loop_cost},
            "CUDA_initialized": torch.cuda.is_initialized()}


# ---------------------------------------------------------------- CUDA acceptance
def _a0_reference(c):
    ref = json.loads(_pinned(c["reference"]["A0_acceptance"]).read_text())
    rows = []
    with _pinned(c["reference"]["A0_updates"]).open() as f:
        for _ in range(A0_UPDATES):
            rows.append(json.loads(f.readline()))
    return ref, rows


def _a0_equivalence(c, out, counts):
    """4.1: same parent, same two production windows; plain then checkpoint, sequentially resident."""
    from sparse_rtdetr.research.query_local_runtime import capture as r25_capture, create as r25_create
    ref, ref_rows = _a0_reference(c)
    PHASE["phase"] = "A0-768:data"
    started = time.perf_counter()
    batches = list(windows(native_train_dataset(c["data"]["train_core"]), PARENT_UPDATE, limit=A0_UPDATES))
    data_seconds = time.perf_counter() - started
    arms, retained = {}, {}
    for mode in ("plain", "checkpoint"):
        _reset_peak()
        PHASE["phase"] = f"A0-768-{mode}:construct"
        t0 = time.perf_counter()
        e, bridge = r25_create(c["parent"]["path"], "A0", "cuda:0")
        wrapped = S.install_activation_checkpoint(e.model) if mode == "checkpoint" else []
        _register_counts(e, counts, f"A0-768-{mode}")
        parent_bn = _bn_counts(e.model)
        rec0, sec0 = S.RECOMPUTE["calls"], S.RECOMPUTE["seconds"]
        S.RECOMPUTE["timed"] = True
        rows, keep = [], []
        for x, t, inp in batches:
            with _pre_clip_gradients(e) as grads:
                r = _update(e, x, t, inp, exact=True)
            S.require(len(grads) == sum(1 for _ in e.model.parameters()), "pre-clip gradient capture incomplete")
            keep.append({"input": inp, "result": _result_body(r), "gradients": dict(grads), "state": r25_capture(e)})
            rows.append({"update": r["update"], "loss": r["loss"], "lr": r["lr"],
                         "gradient_norm_before_clip": r["gradient_norm_before_clip"],
                         "gradient_sha256_before_clip": r["gradient_sha256_before_clip"],
                         "update_seconds": r["elapsed_seconds"], **_memory()})
        S.RECOMPUTE["timed"] = False
        bn_delta = {n: v - parent_bn[n] for n, v in _bn_counts(e.model).items()}
        recomputes = S.RECOMPUTE["calls"] - rec0
        arms[mode] = {"bridge_shared_parameters_sha256": bridge["shared_parameters_sha256"],
                      "wrapped_blocks": wrapped, "updates": rows, "block_recomputations": recomputes,
                      "block_recompute_seconds_synchronized": S.RECOMPUTE["seconds"] - sec0,
                      "BN_modules": len(bn_delta), "BN_num_batches_tracked_delta": sorted(set(bn_delta.values())),
                      "stage_seconds": time.perf_counter() - t0, **_memory()}
        S.require(set(bn_delta.values()) == {2 * A0_UPDATES}, f"A0 {mode} BN updates differ from native forward count")
        S.require(recomputes == (RECOMPUTES_PER_UPDATE * A0_UPDATES if mode == "checkpoint" else 0),
                  f"A0 {mode} recomputation count {recomputes}")
        retained[mode] = keep
        del e
        gc.collect()
        torch.cuda.empty_cache()
    # Complete comparisons.
    plain, ckpt = retained["plain"], retained["checkpoint"]
    report = {"declared_excluded": {"result_fields": list(EXCLUDED_RESULT_FIELDS),
                                    "other": "memory statistics, wall timings, recomputation counters"},
              "compared": ["model outputs incl. aux/encoder/DN outputs (output_sha256 per micro-batch)",
                           "DN noise (dn_noise_sha256)", "every per-term micro-batch loss", "normalizers",
                           "each parameter's pre-clip gradient tensor", "clip norm", "LR",
                           "post-update model state_dict incl. BN running_mean/var/num_batches_tracked",
                           "AdamW exp_avg/exp_avg_sq/step", "EMA module and updates clock",
                           "engine cursor/step ledger", "CPU/Python/NumPy/CUDA RNG"],
              "updates": []}
    failures = []
    for i, (p, k) in enumerate(zip(plain, ckpt)):
        row = {"update": p["result"]["update"], "input_equal": p["input"] == k["input"]}
        row["result_differences"] = _differences(p["result"], k["result"])
        grad_diff = _differences(p["gradients"], k["gradients"])
        row["gradient_parameters_compared"] = len(p["gradients"])
        row["gradient_differences"] = grad_diff
        row["state_differences"] = _differences(p["state"], k["state"])
        row["state_exact_equal"] = ck.exact_equal(p["state"], k["state"])
        # Interface regression: plain equals the R25 A0 acceptance/formal records on the same input.
        r25 = ref["updates"][i]
        as_json = json.loads(json.dumps(p["result"], allow_nan=False))
        row["plain_equal_R25_A0_acceptance_record"] = (
            r25["input"] == p["input"] and _result_body(r25["update"]) == as_json)
        if not row["plain_equal_R25_A0_acceptance_record"]:
            row["R25_record_differences"] = _differences(_result_body(r25["update"]), as_json)
        ledger = ref_rows[i]
        row["plain_equal_R25_A0_formal_ledger"] = (
            ledger["input"] == p["input"] and ledger["loss"] == p["result"]["loss"]
            and ledger["lr"] == p["result"]["lr"]
            and ledger["gradient_norm_before_clip"] == p["result"]["gradient_norm_before_clip"])
        ok = (row["input_equal"] and not row["result_differences"] and not grad_diff
              and not row["state_differences"] and row["state_exact_equal"]
              and row["plain_equal_R25_A0_acceptance_record"] and row["plain_equal_R25_A0_formal_ledger"])
        row["equivalent"] = ok
        if not ok:
            failures.append(row["update"])
        report["updates"].append(row)
    report.update(arms=arms, window_load_seconds=data_seconds, status="PASS" if not failures else "STOP")
    _json(out / "A0_768_EQUIVALENCE.json", report)
    del plain, ckpt, retained, batches
    gc.collect()
    S.require(not failures, f"A0 768x1344 plain/checkpoint complete state not equivalent at {failures}")
    return {"status": "PASS", "file": _file(out / "A0_768_EQUIVALENCE.json"),
            "updates": A0_UPDATES * 2, "training_forwards": 4 * A0_UPDATES,
            "block_recomputations": arms["checkpoint"]["block_recomputations"],
            "peak": {m: {"allocated": arms[m]["peak_allocated_bytes"], "reserved": arms[m]["peak_reserved_bytes"]}
                     for m in arms},
            "update_seconds": {m: [r["update_seconds"] for r in arms[m]["updates"]] for m in arms}}


def _s15_inference(e, c):
    from s15_formal_eval import development, forward_batch
    data = development(c)
    modes = {n: m.training for n, m in e.ema.module.named_modules()}
    model = e.ema.module.eval()
    rng = ck.capture_rng(e.device)
    _reset_peak()
    PHASE["phase"] = "S15:EMA-inference"
    timings, forwards = [], 0
    with torch.inference_mode():
        for i in range(0, EVAL_IMAGES, EVAL_BATCH):
            rows = data.records[i:i + EVAL_BATCH]
            for repeat in range(2):  # first pass warm-up, second timed (2 warm-up + 2 timed batches)
                _, timing = forward_batch(model, data, rows, e.device)
                forwards += 1
                timings.append({"batch": i // EVAL_BATCH, "warmup": repeat == 0, **timing})
    S.require(ck.exact_equal(ck.capture_rng(e.device), rng), "S15 inference changed RNG")
    for n, m in model.named_modules():
        m.training = modes[n]
    return {"images": EVAL_IMAGES, "batch": EVAL_BATCH, "fp32": True, "forwards": forwards, "timings": timings,
            "predictions_written": False, "scored": False, **_memory()}


def _accept(c, out, counts):
    out = Path(out)
    t0 = time.perf_counter()
    equivalence = _a0_equivalence(c, out, counts)
    S.require(counts["actual_updates"] == 2 * A0_UPDATES and counts["forwards"] == 4 * A0_UPDATES, "A0 acceptance count")
    a0_seconds = time.perf_counter() - t0
    # 4.2 S15 production checkpoint mode
    t1 = time.perf_counter()
    folder = out / "S15"
    folder.mkdir()
    _reset_peak()
    data = _data(c)
    e, bridge = _make(c, "cuda:0")
    _register_counts(e, counts, "S15-accept")
    seen, criterion = _precision_observers(e)
    parent_bn = _bn_counts(e.model)
    rec0, sec0 = S.RECOMPUTE["calls"], S.RECOMPUTE["seconds"]
    S.RECOMPUTE["timed"] = True
    rows = []
    for i, (x, t, inp) in enumerate(windows(data, PARENT_UPDATE, limit=S15_FIRST), 1):
        r = _update(e, x, t, inp, exact=True)
        rows.append({"input": inp, "update": r, **_memory()})
        print(json.dumps({"phase": "S15-first", "update": r["update"], "loss": r["loss"], **_memory()}), flush=True)
        if i == 1:
            one = folder / "u1.pth"
            saved = _save(e, c, one, "acceptance")
    S.RECOMPUTE["timed"] = False
    expected = S.capture(e)
    bn_delta = sorted(set(v - parent_bn[k] for k, v in _bn_counts(e.model).items()))
    S.require(bn_delta == [2 * S15_FIRST], "S15 BN updates differ from native forward count")
    S.require(len(seen) == 2 * S15_FIRST and all(v["autocast"] and v["autocast_dtype"] == "torch.bfloat16"
                                                 and v["input"] == "torch.float32" for v in seen)
              and criterion == [["torch.float32"]] * (2 * S15_FIRST), "S15 CUDA precision differs")
    first_peak = _memory()
    del e
    gc.collect()
    torch.cuda.empty_cache()
    _reset_peak()
    clone, _ = _make(c, "cuda:0")
    _register_counts(clone, counts, "S15-replay")
    PHASE["phase"] = "S15:restore"
    side, state = _load(c, one, "acceptance")
    restored = _restore_engine(clone, one, side, state)
    del state
    ((x, t, inp),) = list(windows(data, PARENT_UPDATE + 1, limit=1))
    replay = _update(clone, x, t, inp, exact=True)
    replay_state = S.capture(clone)
    replay_diff = _differences(expected, replay_state) + _differences(_result_body(rows[1]["update"]), _result_body(replay))
    S.require(inp == rows[1]["input"] and not replay_diff and ck.exact_equal(expected, replay_state),
              f"S15 CUDA replay differs: {replay_diff[:5]}")
    del expected, replay_state, x, t
    timing = []
    started = time.perf_counter()
    for x, t, inp in windows(data, clone.completed_updates, limit=S15_TIMING):
        r = _update(clone, x, t, inp)
        timing.append({"update": r["update"], "update_seconds": r["elapsed_seconds"], "loss": r["loss"],
                       "gradient_norm_before_clip": r["gradient_norm_before_clip"], **_memory()})
        print(json.dumps({"phase": "S15-timing", **timing[-1]}), flush=True)
    loader_inclusive = time.perf_counter() - started
    train_peak = _memory()
    recomputes = S.RECOMPUTE["calls"] - rec0
    S.require(recomputes == RECOMPUTES_PER_UPDATE * (S15_FIRST + S15_REPLAY + S15_TIMING), f"S15 recomputation count {recomputes}")
    infer = _s15_inference(clone, c)
    s15_updates = S15_FIRST + S15_REPLAY + S15_TIMING
    S.require(counts["actual_updates"] == 2 * A0_UPDATES + s15_updates
              and counts["forwards"] == 2 * (2 * A0_UPDATES + s15_updates) and infer["forwards"] == 4,
              "S15 acceptance count")
    result = {"A0_768_equivalence": equivalence, "A0_stage_seconds": a0_seconds,
              "S15": {"bridge": bridge, "first_updates": rows, "saved_u1": saved, "restore": restored,
                      "replay_complete_state_and_record_exact": True,
                      "replay_update_seconds": replay["elapsed_seconds"],
                      "timing_updates": timing, "timing_loader_inclusive_seconds": loader_inclusive,
                      "steady_update_seconds_mean": sum(r["update_seconds"] for r in timing) / len(timing),
                      "first_phase_peak": first_peak, "replay_and_timing_peak": train_peak,
                      "block_recomputations": recomputes,
                      "block_recompute_seconds_synchronized_first_two_updates": S.RECOMPUTE["seconds"] - sec0,
                      "precision": {"model": seen, "criterion": criterion},
                      "BN_num_batches_tracked_delta_first_two_updates": bn_delta,
                      "stage_seconds": time.perf_counter() - t1},
              "S15_EMA_inference": infer,
              "totals": {"optimizer_updates": counts["actual_updates"], "training_forwards": counts["forwards"],
                         "EMA_inference_forwards": infer["forwards"],
                         "full_model_forwards": counts["forwards"] + infer["forwards"],
                         "block_recomputations": equivalence["block_recomputations"] + recomputes},
              "device_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
              "allocator_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
              "formal_state_usable": False}
    S.require(result["totals"]["optimizer_updates"] == 13 and result["totals"]["full_model_forwards"] == 30,
              "S15 acceptance success-path totals differ")
    return result


# ---------------------------------------------------------------- CLI
def main():
    from s15_formal_run import load_contract
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
        S.require(Path(c["source_root"]).resolve() == ROOT.resolve(), "S15 trainer is not running from the bound release")
        if device == "cpu":
            S.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "S15 CPU gate requires hidden CUDA")
        else:
            S.require(os.environ.get("PYTORCH_CUDA_ALLOC_CONF") == c["workload"]["allocator_conf"],
                      "S15 allocator configuration differs from contract")
        setup(device)
        if device == "cpu":
            # Fixture-only CPU work; thread count is recorded, never used for formal numerics.
            torch.set_num_threads(int(os.environ.get("S15_CPU_THREADS", "8")))
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
