"""S15 seed1/seed2 replication: training, CUDA restore acceptance and complete restart state
(TASK_S15_SEED12_REPLICATION_001; seed0 version: commit 61646b3 / S15 release-002).

Every action is bound to one seed of the contract (``--seed``): its own native e060 parent, recipe
seed, sampler/augmentation seed and A0 reference.  The formal loop is the seed0 loop unchanged.

actions (always launched by s15_formal_run.py from a frozen release with a bound contract):
  cpu     CUDA hidden.  Production e060 bridge on CPU, formal-binding save/load/exact restore,
          scope and cross-seed refusals, science projection, the bound loader against the seed's
          own historical 768 audit at two pre-registered windows, audit prefix, and the real formal
          loop on a 1152x2016 FP32 bs1x2 fixture (2 updates, then a resume replaying the second).
  accept  CUDA.  1152 production 8x2/BF16 checkpoint mode: u1, u2 from the seed's e060, save u1,
          fresh engine restore + replay u2; result and complete state/RNG exact.  3 updates,
          6 training forwards.  The 768 plain/checkpoint equivalence of seed0 is reused, not redone.
          Acceptance states are never formal.
  train   CUDA.  Formal S15 e061-e120 from the seed's certified e060 parent (or exact restart).
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
from formal_data import END_UPDATE, PARENT_UPDATE, WINDOWS_PER_EPOCH, cursor, windows
from sparse_rtdetr.protocol_r18 import checkpoint as ck
from sparse_rtdetr.protocol_r18.engine import assert_healthy, learning_rate
from sparse_rtdetr.protocol_r18.model import Config
from sparse_rtdetr.protocol_r18.runio import fsync_directory
from sparse_rtdetr.research import scale_baseline as S

ARM = "S15"
SAVE_EPOCHS = (60, 75, 90, 105, 120)
SCHEMA = "s15-seed-complete-state-v1"
FIXTURE_STOP = PARENT_UPDATE + 2
BLOCKS = 8                       # 4 backbone stages + 2 FPN + 2 PAN blocks
RECOMPUTES_PER_UPDATE = 2 * BLOCKS
S15_FIRST, S15_REPLAY = 2, 1
# Pre-registered loader windows compared with the seed's own historical 768 audit (e061 w0, e118 w0).
LOADER_CHECK_UPDATES = (PARENT_UPDATE, 117 * WINDOWS_PER_EPOCH)
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


# ---------------------------------------------------------------- seed binding
def bind_seed(seed):
    """Bind the unchanged formal_data loader/cursor (byte-identical to R25/seed0) to one seed.

    formal_data reads its module constant SEED at call time in train_dataset/cursor/windows; the
    binding is set once per process and can never be switched to another seed afterwards.
    """
    S.require(seed in S.PARENT_SHAS, "S15 seed has no certified parent")
    bound = getattr(formal_data, "_s15_bound_seed", None)
    S.require(bound in (None, seed), f"process already bound to seed {bound}")
    formal_data.SEED = seed
    formal_data._s15_bound_seed = seed
    S.require(cursor(PARENT_UPDATE)["seed"] == seed, "formal cursor seed binding failed")


def fixture_config(seed):
    return Config(seed=seed, input_hw=S.S15_HW, physical_batch_size=1, amp_dtype="float32")


# ---------------------------------------------------------------- state binding
def _cfg(c, fixture):
    return fixture_config(c["seed"]) if fixture else S.production_config(c["seed"])


def _state_binding(c, update, scope, audit, fixture=False):
    S.require(formal_data.SEED == c["seed"], "loader seed is not bound to the contract seed")
    return {"schema": SCHEMA, "scope": scope, "arm": ARM, "seed": c["seed"], "formal_identity": c["formal_identity"],
            "contract_sha256": c["contract_sha256"], "global_update": update,
            "parent_sha256": c["parent"]["sha256"], "cursor": cursor(update), "audit": audit,
            "config": _cfg(c, fixture).binding(), "activation_checkpoint": c["workload"]["activation_checkpoint"],
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
              and s["config"] == _cfg(c, fixture).binding()
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
    return S.create(c["parent"]["path"], device, _cfg(c, fixture) if fixture else None, seed=c["seed"],
                    fixture=fixture, activation_checkpoint=c["workload"]["activation_checkpoint"])


def _data(c):
    return S.train_dataset(c["data"]["train_core"], S.S15_HW, seed=c["seed"])


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
                      and Path(resume).resolve().parent == Path(prior["outputs"]["train"])
                      and prior["formal_identity"] == c["formal_identity"] and prior["seed"] == c["seed"]
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
def _historical_receipt(c, update):
    """Receipt fields of the seed's own historical 768 audit row, in the formal_data schema."""
    from sparse_rtdetr.localization.runio import digest
    ref = c["history"]["loader_check_audits"][str(update)]
    row = json.loads(_pinned(ref).read_text())
    S.require(row["seed"] == c["seed"] and row["update"] == update and row["result"]["update"] == update,
              "historical audit row identity")
    h = row["result"]["input"]
    return {"update": update, "epoch": row["result"]["epoch"], "window": row["result"]["window"],
            "image_ids_sha256": digest(h["ids"]), "gt_ids_sha256": digest(h["GT_ids"]),
            "augmentation_trace_sha256": h["metadata_sha256"], "augmented_samples_sha256": digest(h["augmented_sha256"]),
            "input_tensor_sha256": h["input_tensor_sha256"], "target_sha256": h["target_sha256"]}


def _loader_check(c):
    """The bound loader reproduces the seed's own native trajectory: at 768 every receipt field equals
    the historical audit; at 1152 order/draw fields equal and canvas-dependent fields are counted."""
    S.require(not torch.cuda.is_initialized(), "loader check before CUDA")
    native = formal_data.train_dataset(c["data"]["train_core"])  # native 768x1344, bound seed
    S.require(native.seed == c["seed"], "native dataset seed")
    scaled = _data(c)
    same = ("update", "epoch", "window", "image_ids_sha256", "augmentation_seeds_sha256", "augmentation_strong")
    differs = ("gt_ids_sha256", "augmentation_trace_sha256", "augmented_samples_sha256", "input_tensor_sha256",
               "target_sha256")
    rows = []
    for completed in LOADER_CHECK_UPDATES:
        (_, _, a768), = list(windows(native, completed, limit=1))
        (_, _, a1152), = list(windows(scaled, completed, limit=1))
        hist = _historical_receipt(c, completed + 1)
        exact768 = {k: a768[k] == v for k, v in hist.items()}
        rows.append({"update": completed + 1, "native_768_equal_historical_audit": exact768,
                     "S15_1152_order_draw_equal_768": {k: a1152[k] == a768[k] for k in same},
                     "S15_1152_canvas_dependent_differs": {k: a1152[k] != a768[k] for k in differs},
                     "receipt_768": a768, "receipt_1152": a1152})
        S.require(all(exact768.values()), f"bound 768 loader differs from the seed's historical audit at {completed + 1}")
        S.require(all(a1152[k] == a768[k] for k in same), "S15 order/draws differ from the native trajectory")
    S.require(not torch.cuda.is_initialized(), "loader check initialized CUDA")
    return rows


def _science_projection(c):
    """S15 seed config differs from seed0 S15 only in seed strings, and from the seed's native recipe
    only in the canvas; the historical training binding carries the same native recipe."""
    mine = S.production_config(c["seed"]).binding()
    seed0 = S.production_config(0).binding()
    native = Config(seed=c["seed"]).binding()
    historical = json.loads(_pinned(c["history"]["formal_binding"]).read_text())["config"]
    vs_seed0 = sorted(k for k in mine if mine[k] != seed0[k])
    vs_native = sorted(k for k in mine if mine[k] != native[k])
    S.require(vs_seed0 == (["initializer", "seed", "training_rng"] if c["seed"] else []), f"S15 seed recipe {vs_seed0}")
    S.require(vs_native == ["input_hw"] and historical == native, "S15 recipe differs from the seed's native recipe")
    return {"differs_from_seed0_S15": vs_seed0, "differs_from_native_seed_recipe": vs_native,
            "historical_training_binding_equals_native_recipe": True, "config": mine}


def _cpu(c, out, other):
    out = Path(out)
    S.require(not torch.cuda.is_initialized(), "S15 CPU gate initialized CUDA")
    t0, c0 = time.perf_counter(), time.process_time()
    S.require(c["workload"]["checkpoint_segments"] == list(S.CHECKPOINT_SEGMENTS)
              and c["workload"]["input_hw"] == list(S.S15_HW), "S15 contract memory segments or canvas differ")
    S.require(other["seed"] != c["seed"] and other["contract_sha256"] == c["contract_sha256"], "cross-seed fixture")
    projection = _science_projection(c)
    loader = _loader_check(c)
    loader_cost = {"wall_seconds": time.perf_counter() - t0, "cpu_seconds": time.process_time() - c0}
    # Production bridge and formal-binding save/load/exact restore on CPU.
    e, bridge = _make(c, "cpu")
    S.require(e.config == S.production_config(c["seed"]) and e.s15_activation_checkpoint is True
              and e.r25_parent == c["parent"]["sha256"]
              and bridge["activation_checkpoint_segments"] and len(bridge["activation_checkpoint_segments"]) == BLOCKS,
              "S15 production engine seed/parent/memory mode")
    path = out / "cpu-production-state.pth"
    _save(e, c, path, "cpu-fixture")
    side, state = _load(c, path, "cpu-fixture")
    restored = _restore_engine(e, path, side, state)
    del state, e
    gc.collect()
    refused = []
    for label, view, scope, fixture in (("formal scope", c, "formal", False), ("acceptance scope", c, "acceptance", False),
                                        ("fixture config", c, "cpu-fixture", True),
                                        (f"seed{other['seed']} contract view", other, "cpu-fixture", False)):
        try:
            _load(view, path, scope, fixture=fixture)
        except (ValueError, KeyError) as exc:
            refused.append({"case": label, "reason": repr(exc)[:300]})
        else:
            raise AssertionError("S15 cross-scope/config/seed state accepted: " + label)
    for label, parent, seed in ((f"seed{other['seed']} parent bound as seed{c['seed']}", other["parent"]["path"], c["seed"]),
                                (f"seed{c['seed']} parent bound as seed{other['seed']}", c["parent"]["path"], other["seed"])):
        try:
            S.create(parent, "cpu", seed=seed, activation_checkpoint=True)
        except ValueError as exc:
            refused.append({"case": label, "reason": repr(exc)[:300]})
        else:
            raise AssertionError("S15 accepted a parent of another seed: " + label)
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
    # The fixture's first window is the same seed-bound 1152 window as the loader check (images sliced later).
    S.require(json.loads(rows1[0])["input"] == loader[0]["receipt_1152"], "fixture loop window differs from loader check")
    del end1, end2
    loop_cost = {"wall_seconds": time.perf_counter() - t1, "cpu_seconds": time.process_time() - c1}
    S.require(not torch.cuda.is_initialized(), "S15 CPU gate initialized CUDA")
    return {"seed": c["seed"], "science_projection": projection, "loader_check": loader,
            "production_bridge": bridge, "production_restore": restored, "refused": refused,
            "audit_prefix_after_append_exact": True, "bad_audit_SHA_refused": True,
            "fixture_formal_loop": {"config": fixture_config(c["seed"]).binding(), "first": run1,
                                    "first_counts": dict(counts1), "resume": run2, "resume_counts": dict(counts2),
                                    "resumed_endpoint_complete_state_exact": True},
            "cpu_fixture_updates": counts1["actual_updates"] + counts2["actual_updates"],
            "cpu_fixture_training_forwards": counts1["forwards"] + counts2["forwards"],
            "cost": {"loader_and_projection": loader_cost, "bridge_and_state_cumulative": bridge_cost,
                     "fixture_loop": loop_cost},
            "CUDA_initialized": torch.cuda.is_initialized()}


# ---------------------------------------------------------------- CUDA acceptance (per seed)
def _accept(c, out, counts):
    """1152 production 8x2/BF16 checkpoint mode from the seed's e060: u1, u2, save u1, fresh engine
    restore and replay u2; result record and complete state/RNG must be exact."""
    out = Path(out)
    t1 = time.perf_counter()
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
        print(json.dumps({"phase": "S15-first", "seed": c["seed"], "update": r["update"], "loss": r["loss"],
                          **_memory()}), flush=True)
        if i == 1:
            one = out / "u1.pth"
            saved = _save(e, c, one, "acceptance")
    S.RECOMPUTE["timed"] = False
    expected = S.capture(e)
    bn_delta = sorted(set(v - parent_bn[k] for k, v in _bn_counts(e.model).items()))
    S.require(bn_delta == [2 * S15_FIRST], "S15 BN updates differ from native forward count")
    S.require(len(seen) == 2 * S15_FIRST and all(v["autocast"] and v["autocast_dtype"] == "torch.bfloat16"
                                                 and v["input"] == "torch.float32" for v in seen)
              and criterion == [["torch.float32"]] * (2 * S15_FIRST), "S15 CUDA precision differs")
    first_peak = _memory()
    first_recompute = {"calls": S.RECOMPUTE["calls"] - rec0, "seconds_synchronized": S.RECOMPUTE["seconds"] - sec0}
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
    replay_peak = _memory()
    recomputes = S.RECOMPUTE["calls"] - rec0
    updates = S15_FIRST + S15_REPLAY
    S.require(recomputes == RECOMPUTES_PER_UPDATE * updates, f"S15 recomputation count {recomputes}")
    S.require(counts["actual_updates"] == updates and counts["forwards"] == 2 * updates, "S15 acceptance count")
    result = {"seed": c["seed"],
              "S15": {"bridge": bridge, "first_updates": rows, "saved_u1": saved, "restore": restored,
                      "replay_update": _result_body(replay),
                      "replay_complete_state_and_record_exact": True,
                      "replay_update_seconds": replay["elapsed_seconds"],
                      "first_phase_peak": first_peak, "replay_peak": replay_peak,
                      "block_recomputations": recomputes, "block_recompute_first_two_updates": first_recompute,
                      "precision": {"model": seen, "criterion": criterion},
                      "BN_num_batches_tracked_delta_first_two_updates": bn_delta,
                      "stage_seconds": time.perf_counter() - t1},
              "reused_evidence": c["reused_evidence"],
              "totals": {"optimizer_updates": counts["actual_updates"], "training_forwards": counts["forwards"],
                         "EMA_inference_forwards": 0, "full_model_forwards": counts["forwards"],
                         "block_recomputations": recomputes},
              "device_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
              "allocator_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
              "formal_state_usable": False}
    acc = c["workload"]["acceptance"]
    S.require(result["totals"]["optimizer_updates"] == acc["S15_updates_per_seed"]
              and result["totals"]["full_model_forwards"] == acc["S15_training_forwards_per_seed"],
              "S15 acceptance success-path totals differ")
    return result


# ---------------------------------------------------------------- CLI
def main():
    from s15_formal_run import load_contract, seed_view
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("cpu", "accept", "train"))
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
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
        contract = load_contract(args.contract)
        c = seed_view(contract, args.seed)
        S.require(Path(c["source_root"]).resolve() == ROOT.resolve(), "S15 trainer is not running from the bound release")
        bind_seed(c["seed"])
        if device == "cpu":
            S.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "S15 CPU gate requires hidden CUDA")
        else:
            S.require(os.environ.get("PYTORCH_CUDA_ALLOC_CONF") == c["workload"]["allocator_conf"],
                      "S15 allocator configuration differs from contract")
        # Deterministic backend flags.  setup() also seeds the global RNGs, but every engine built
        # afterwards replaces them by the parent (or exact restart) RNG; nothing re-seeds after that.
        setup(device)
        if device == "cpu":
            # Fixture-only CPU work; thread count is recorded, never used for formal numerics.
            torch.set_num_threads(int(os.environ.get("S15_CPU_THREADS", "8")))
        else:
            torch.cuda.reset_peak_memory_stats()
        if args.action == "cpu":
            other = seed_view(contract, next(s for s in contract["seed_order"] if s != c["seed"]))
            result = _cpu(c, args.out, other)
        elif args.action == "accept":
            result = _accept(c, args.out, counts)
        else:
            prior = seed_view(load_contract(args.prior_contract), c["seed"]) if args.prior_contract else None
            result = _formal(c, args.out, counts, args.resume, prior)
        _json(args.out / "CLOSED.json", {"status": "PASS", "action": args.action, **result, "seed": c["seed"],
                                        "contract_sha256": c["contract_sha256"], "counts": dict(counts),
                                        "elapsed_seconds": time.monotonic() - start,
                                        "process_cpu_seconds": time.process_time() - cpu0, **_memory()})
    except BaseException as exc:
        oom = isinstance(exc, torch.cuda.OutOfMemoryError)
        _json(args.out / "STOP.json", {"status": "STOP_CUDA_OOM" if oom else "STOP", "phase": PHASE["phase"],
                                      "seed": args.seed, "error": repr(exc)[:4000], "traceback": traceback.format_exc(),
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
