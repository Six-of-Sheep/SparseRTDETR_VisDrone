"""Formal R25 training, real loader acceptance and arm-bound restart state."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import time
import traceback
from collections import Counter
from pathlib import Path

from r25_run import setup
import torch
from formal_data import END_UPDATE, PARENT_UPDATE, check_boundaries, cursor, train_dataset, windows
from sparse_rtdetr.protocol_r18 import checkpoint as ck
from sparse_rtdetr.protocol_r18.engine import assert_healthy, learning_rate
from sparse_rtdetr.protocol_r18.model import Config
from sparse_rtdetr.protocol_r18.runio import fsync_directory
from sparse_rtdetr.research.query_local import VERSION
from sparse_rtdetr.research.query_local_runtime import capture, create


ARMS = ("A0", "A1", "A2")
SAVE_EPOCHS = (60, 75, 90, 105, 120)
SCHEMA = "r25-complete-state-v1"


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


def _sha_prefix(path, count):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while count:
            block = stream.read(min(count, 1024 * 1024))
            if not block:
                raise ValueError("incomplete R25 audit prefix")
            h.update(block)
            count -= len(block)
    return h.hexdigest()


def _state_binding(c, arm, update, scope, audit):
    return {
        "schema": SCHEMA, "scope": scope, "arm": arm, "formal_identity": c["formal_identity"],
        "contract_sha256": c["contract_sha256"], "global_update": update,
        "parent_sha256": c["parent"]["sha256"], "cursor": cursor(update), "audit": audit,
        "torch": str(torch.__version__),
    }


def _save(e, c, path, scope, audit=None):
    path = Path(path)
    if e.ema.updates != e.completed_updates:
        raise ValueError("R25 EMA clock disagrees at save")
    assert_healthy(e.model, e.optimizer, e.ema)
    state = capture(e)
    binding = _state_binding(c, e.r25_arm, e.completed_updates, scope, audit)
    state["formal_binding_sha256"] = hashlib.sha256(ck.canonical(binding)).hexdigest()
    _atomic(path, lambda stream: torch.save(state, stream))
    result = {"status": "COMPLETE", **binding, "file": _file(path)}
    _json(path.with_suffix(".pth.json"), result)
    return result


def _verify_audit(audit, path):
    if (audit["path"] != str(path) or type(audit["bytes"]) is not int or audit["bytes"] < 0
            or audit["sha256"] != _sha_prefix(path, audit["bytes"])):
        raise ValueError("R25 formal audit prefix differs")


def _load(c, path, arm, scope, *, endpoint=False):
    path = Path(path).resolve()
    side = json.loads(path.with_suffix(".pth.json").read_text())
    update = side["global_update"]
    binding = _state_binding(c, arm, update, scope, side.get("audit"))
    if side != {"status": "COMPLETE", **binding, "file": _file(path)}:
        raise ValueError("R25 state receipt or file identity differs")
    if scope == "formal":
        if update not in tuple(epoch * 304 for epoch in SAVE_EPOCHS):
            raise ValueError("R25 checkpoint not at a registered boundary")
        if endpoint and update != END_UPDATE:
            raise ValueError("R25 evaluation requires e120")
        _verify_audit(side["audit"], path.parent / "UPDATES.jsonl")
    s = torch.load(path, map_location="cpu", weights_only=True)
    if (s.pop("formal_binding_sha256") != hashlib.sha256(ck.canonical(binding)).hexdigest()
        or s["version"] != VERSION or s["arm"] != arm or s["parent_sha256"] != c["parent"]["sha256"]
        or s["config"] != Config().binding() or s["engine"]["completed_updates"] != update
        or s["ema"]["updates"] != update or s["engine"]["next_epoch"] != side["cursor"]["next_epoch"]
        or s["engine"]["next_window"] != side["cursor"]["next_window"]):
        raise ValueError("R25 checkpoint payload identity/clock differs")
    return side, s


def read_endpoint(contract, path, arm):
    """Return the verified e120 R25 model/optimizer/EMA payload."""
    return _load(contract, path, arm, "formal", endpoint=True)[1]


def _restore_engine(e, path, side, state):
    # Reuse the accepted exact post-restore comparison with the same payload.
    # The formal binding is external to the original engine capture, so load the
    # already verified payload through its matching state adapters.
    if state["optimizer_names"] != [g["param_names"] for g in e.optimizer.param_groups]:
        raise ValueError("R25 optimizer names differ")
    e.model.load_state_dict(state["model"], strict=True)
    e.optimizer.load_state_dict(state["optimizer"])
    e.ema.load_state_dict(state["ema"])
    modes = dict(e.model.named_modules())
    if set(modes) != set(state["model_modes"]):
        raise ValueError("R25 training mode keys differ")
    for name, mode in state["model_modes"].items():
        modes[name].training = mode
    e.load_state_dict(state["engine"])
    ck.restore_rng(state["rng"], e.device)
    if not ck.exact_equal(capture(e), state):
        raise ValueError("R25 full formal state did not restore exactly")
    return {"full_state_exact": True, "path": str(path), "global_update": side["global_update"],
            "sha256": side["file"]["sha256"], "cursor": side["cursor"]}


def _register_counts(e, counts):
    e.model.register_forward_pre_hook(lambda *_: counts.update(forward_attempts=1))
    e.model.register_forward_hook(lambda *_: counts.update(forwards=1))
    e.optimizer.register_step_post_hook(lambda *_: counts.update(actual_updates=1))


def _gradient_groups(e, *, residual=True):
    norms = {}
    for group in e.optimizer.param_groups:
        values = [p.grad.detach().float().norm() for p in group["params"] if p.grad is not None]
        norms[group["name"]] = float(torch.stack(values).norm()) if values else 0.
    result = {"clipped_group_gradient_norm": norms}
    if residual and e.r25_arm != "A0":
        obs = e.model.local_observation
        box_delta = (obs["refined"].float() - obs["coarse"].float()).abs()
        result["last_microbatch_mean_abs_canvas_cxcywh_delta"] = float(
            (box_delta * box_delta.new_tensor([1344, 768, 1344, 768])).mean())
    return result


def _steps(e):
    result = {}
    for group in e.optimizer.param_groups:
        counts = [int(e.optimizer.state.get(p, {}).get("step", 0)) for p in group["params"]]
        if any(count != e.parameter_steps[n] for count, n in zip(counts, group["param_names"])):
            raise ValueError("R25 Adam step ledger differs at boundary")
        result[group["name"]] = {"steps_histogram": dict(sorted(Counter(counts).items())),
                                 "lr": group["lr"], "parameters": len(counts)}
    return result


def _initial_snapshot(e):
    return {n: p.detach().cpu().clone() for n, p in e.model.named_parameters()}


def _relative_change(e, initial):
    squares = {}
    for name, p in e.model.named_parameters():
        group = "local" if name.startswith("local_head.") else "shared"
        sums = squares.setdefault(group, [0., 0.])
        value = initial[name].float()
        sums[0] += float((p.detach().cpu().float() - value).square().sum())
        sums[1] += float(value.square().sum())
    return {g: {"relative_l2_change": math.sqrt(n / max(d, 1e-30))}
            for g, (n, d) in squares.items()}


def _data(c):
    return train_dataset(c["data"]["train_core"])


def _make(c, arm, device):
    return create(c["parent"]["path"], arm, device)


def _update(e, images, targets, audit, *, exact=False):
    result = e.train_window(images, targets, capture=exact)
    if (result["update"] != audit["update"] or result["epoch"] != audit["epoch"]
            or result["window"] != audit["window"] or not result["finite"]
            or e.ema.updates != e.completed_updates):
        raise ValueError("R25 update/input/EMA clock differs")
    return result


def _accept(c, out, counts):
    data = _data(c)
    receipts = {}
    first_bridge = None
    first_new = None
    first_inputs = None
    precision = {}
    for arm in ARMS:
        folder = out / arm
        folder.mkdir()
        e, bridge = _make(c, arm, "cuda:0")
        _register_counts(e, counts)
        first_bridge = first_bridge or bridge["shared_parameters_sha256"]
        if first_bridge != bridge["shared_parameters_sha256"]:
            raise ValueError("R25 CUDA acceptance shared initialization differs")
        if arm != "A0":
            first_new = first_new or bridge["new_parameters_sha256"]
            if first_new != bridge["new_parameters_sha256"]:
                raise ValueError("R25 CUDA acceptance new initialization differs")
        before = _initial_snapshot(e)
        seen = []
        criterion_types = []

        def observe(_model, args, result):
            seen.append({"input": str(args[0].dtype),
                         "autocast": torch.is_autocast_enabled("cuda"),
                         "autocast_dtype": str(torch.get_autocast_dtype("cuda")),
                         "logits": str(result["pred_logits"].dtype),
                         "boxes": str(result["pred_boxes"].dtype)})

        def observe_criterion(_criterion, _args, result):
            criterion_types.append(sorted({str(loss.dtype) for loss in result.values()}))

        e.model.register_forward_hook(observe)
        e.criterion.register_forward_hook(observe_criterion)
        grad = {}
        e.optimizer.register_step_pre_hook(lambda *_: grad.update(_gradient_groups(e)))
        rows = []
        for i, (x, t, inp) in enumerate(windows(data, PARENT_UPDATE, limit=2), 1):
            r = _update(e, x, t, inp, exact=True)
            rows.append({"input": inp, "update": r, "gradient": dict(grad)})
            if i == 1:
                one = folder / "u1.pth"
                _save(e, c, one, "acceptance")
        expected = capture(e)
        changes = _relative_change(e, before)
        if len(seen) != 4 or any(not v["autocast"] or v["autocast_dtype"] != "torch.bfloat16"
                                 or v["input"] != "torch.float32" for v in seen):
            raise ValueError("R25 CUDA training precision differs")
        if criterion_types != [["torch.float32"]] * 4:
            raise ValueError("R25 CUDA criterion precision differs")
        precision[arm] = {"model": seen, "criterion": criterion_types}
        if arm != "A0" and any(row["gradient"]["clipped_group_gradient_norm"]["r25_new"] <= 0
                              for row in rows):
            raise ValueError("R25 local head did not receive a finite gradient")
        if changes["shared"]["relative_l2_change"] <= 0 or (
                arm != "A0" and changes["local"]["relative_l2_change"] <= 0):
            raise ValueError("R25 expected parameter groups did not update")
        inputs = [row["input"] for row in rows]
        first_inputs = first_inputs or inputs
        if inputs != first_inputs:
            raise ValueError("R25 CUDA acceptance arms used different inputs")
        del e, before
        gc.collect()
        torch.cuda.empty_cache()
        clone, _ = _make(c, arm, "cuda:0")
        _register_counts(clone, counts)
        side, state = _load(c, one, arm, "acceptance")
        restored = _restore_engine(clone, one, side, state)
        del state
        ((x, t, inp),) = list(windows(data, PARENT_UPDATE + 1, limit=1))
        replay = _update(clone, x, t, inp, exact=True)
        if inp != rows[1]["input"] or not ck.exact_equal(expected, capture(clone)) or (
            {k: v for k, v in replay.items() if k != "elapsed_seconds"} !=
            {k: v for k, v in rows[1]["update"].items() if k != "elapsed_seconds"}):
            raise ValueError("R25 CUDA replay update, input or complete state differs")
        receipts[arm] = {"bridge": bridge, "updates": rows, "restore": restored, "parameter_change": changes,
                         "replay_update_seconds": replay["elapsed_seconds"], "full_replay_exact": True}
        _json(folder / "ACCEPTANCE.json", receipts[arm])
        del clone, expected, x, t
        gc.collect()
        torch.cuda.empty_cache()
    if counts["actual_updates"] != 9 or counts["forwards"] != 18:
        raise ValueError("R25 acceptance count differs")
    return {"arms": receipts, "precision": precision, "actual_updates": 9, "forward_calls": 18,
            "formal_state_usable": False}


def _available(path, next_state_bytes):
    # A save needs room for one complete state, and the following stage checks again.
    needed = next_state_bytes + 64 * 1024**2
    free = shutil.disk_usage(path).free
    if free < needed:
        raise RuntimeError(f"insufficient disk for next R25 save: {free} bytes, need {needed}")


def _formal(c, out, arm, counts, resume, prior):
    if (resume is None) != (prior is None):
        raise ValueError("resume requires both checkpoint and prior contract")
    e, bridge = _make(c, arm, "cuda:0")
    _register_counts(e, counts)
    _json(out / "BRIDGE.json", bridge)
    initial = _initial_snapshot(e)
    history = out / "UPDATES.jsonl"
    restored = None
    previous_rows = {}
    if resume is not None:
        prior_ref = c["prior"]["contract"] if c.get("prior") else None
        if (prior_ref != _file(prior["contract_path"]) or
                resume.resolve().parent != Path(prior["attempt_root"]) / f"train-{arm}" or
                prior["formal_identity"] != c["formal_identity"] or prior["contract_sha256"] == c["contract_sha256"]):
            raise ValueError("R25 restart must preserve scientific identity with distinct attempt")
        side, state = _load(prior, resume, arm, "formal")
        restored = _restore_engine(e, resume, side, state)
        del state
        remaining = side["audit"]["bytes"]
        with Path(side["audit"]["path"]).open("rb") as source, history.open("xb") as dest:
            while remaining:
                block = source.read(min(remaining, 1024**2))
                if not block:
                    raise ValueError("prior audit prefix truncated")
                dest.write(block)
                remaining -= len(block)
            dest.flush()
            os.fsync(dest.fileno())
        with Path(side["audit"]["path"]).open("rb") as source:
            source.seek(side["audit"]["bytes"])
            for line in source:
                if not line.endswith(b"\n"):
                    # An interrupted final append is not a committed update row.
                    break
                row = json.loads(line)
                update = row["input"]["update"]
                if update != e.completed_updates + len(previous_rows) + 1:
                    raise ValueError("R25 previous replay evidence is out of order")
                previous_rows[update] = {k: v for k, v in row.items() if k != "update_seconds"}
        _json(out / "RESTORE.json", {**restored, "rows_available_for_replay": len(previous_rows)})
    else:
        history.touch(exist_ok=False)
    started_at = e.completed_updates
    save_bytes = c["parent"]["bytes"]
    save_points = {ep * 304 for ep in SAVE_EPOCHS}
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
        p = out / f"checkpoint-epoch-{e.completed_updates//304:03d}.pth"
        audit = _file(history)
        receipt = _save(e, c, p, "formal", audit)
        save_bytes = receipt["file"]["bytes"]
        summaries.append(receipt)
        _json(p.with_suffix(".observations.json"), {
            "parameter_steps": _steps(e), "parameter_change_from_arm_e060": _relative_change(e, initial),
            "gradient_last_update": grad, "EMA_updates": e.ema.updates,
        })

    checkpoint(None)
    dataset = _data(c)
    with history.open("a", buffering=1) as ledger:
        while e.completed_updates < END_UPDATE:
            _available(out, save_bytes)
            begin_epoch_at = e.completed_updates
            for x, t, inp in windows(dataset, begin_epoch_at):
                monitor.clear()
                result = _update(e, x, t, inp)
                row = {"input": inp, "loss": result["loss"], "lr": result["lr"],
                       "gradient_norm_before_clip": result["gradient_norm_before_clip"],
                       "update_seconds": result["elapsed_seconds"], "ema_updates": e.ema.updates}
                if result["update"] in previous_rows:
                    if previous_rows.pop(result["update"]) != {k: v for k, v in row.items()
                                                              if k != "update_seconds"}:
                        raise ValueError("R25 recovery replay differs from the recorded trajectory")
                    replayed += 1
                ledger.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
                if e.completed_updates in save_points:
                    ledger.flush()
                    os.fsync(ledger.fileno())
                    grad = {"global_before_clip": result["gradient_norm_before_clip"], **monitor}
                    factor = min(1., .1 / (result["gradient_norm_before_clip"] + 1e-6))
                    grad["group_before_clip_from_native_clip_coefficient"] = {
                        k: v / factor for k, v in grad["clipped_group_gradient_norm"].items()}
                    checkpoint(grad)
            if e.completed_updates == begin_epoch_at or e.completed_updates % 304:
                raise ValueError("R25 loader did not close epoch exactly")
            print(f"{arm} completed epoch {e.completed_updates//304}, updates={e.completed_updates}", flush=True)
    final = summaries[-1]
    if (previous_rows or e.completed_updates != END_UPDATE or counts["actual_updates"] != END_UPDATE - started_at
            or counts["forwards"] != 2 * counts["actual_updates"]
            or any(g["lr"] != learning_rate(END_UPDATE) for g in e.optimizer.param_groups)):
        raise ValueError("R25 terminal update, forward or LR count differs")
    return {"arm": arm, "global_update": END_UPDATE, "started_at_update": started_at,
            "effective_updates": END_UPDATE - PARENT_UPDATE, "actual_updates": counts["actual_updates"],
            "parameter_steps": _steps(e), "EMA_updates": e.ema.updates, "checkpoints": summaries,
            "endpoint": final, "restored": restored, "prior_recorded_updates_replayed_exact": replayed}


def _cpu(c, out):
    data_check = check_boundaries(_data(c))
    _json(out / "DATA_BOUNDARIES.json", data_check)
    e, bridge = _make(c, "A2", "cpu")
    path = out / "cpu-state.pth"
    _save(e, c, path, "cpu-fixture")
    side, state = _load(c, path, "A2", "cpu-fixture")
    restored = _restore_engine(e, path, side, state)
    del state
    refused = []
    for arm, scope in (("A1", "cpu-fixture"), ("A2", "formal"), ("A2", "acceptance")):
        try:
            _load(c, path, arm, scope)
        except ValueError:
            refused.append({"arm": arm, "scope": scope})
        else:
            raise AssertionError("R25 cross-arm or cross-scope state accepted")
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
        raise AssertionError("R25 audit prefix accepted wrong SHA")
    return {"data_boundaries": str(out / "DATA_BOUNDARIES.json"), "fixture_restore": restored,
            "audit_prefix_after_append_exact": True, "bad_audit_SHA_refused": True,
            "fixture_refused": refused, "bridge": bridge, "CUDA_initialized": torch.cuda.is_initialized()}


def _resources(device):
    if device == "cpu" or not torch.cuda.is_initialized():
        return {"peak_allocated": None, "peak_reserved": None}
    return {"peak_allocated": torch.cuda.max_memory_allocated(),
            "peak_reserved": torch.cuda.max_memory_reserved()}


def main():
    from formal_run import load_contract
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("cpu", "accept", "train"))
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--prior-contract", type=Path)
    args = parser.parse_args()
    if (args.action == "train") != (args.arm is not None):
        parser.error("--arm is required only for train")
    if args.action != "train" and (args.resume or args.prior_contract):
        parser.error("resume is available only for train")
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    counts = Counter(actual_updates=0, forwards=0, forward_attempts=0)
    device = "cpu" if args.action == "cpu" else "cuda:0"
    c = None
    try:
        c = load_contract(args.contract)
        if Path(c["source_root"]).resolve() != Path(__file__).resolve().parents[2]:
            raise ValueError("R25 trainer is not running from the bound release")
        setup(device)
        if device != "cpu":
            torch.cuda.reset_peak_memory_stats()
        if args.action == "cpu":
            result = _cpu(c, args.out)
        elif args.action == "accept":
            result = _accept(c, args.out, counts)
        else:
            prior = load_contract(args.prior_contract) if args.prior_contract else None
            result = _formal(c, args.out, args.arm, counts, args.resume, prior)
        _json(args.out / "CLOSED.json", {"status": "PASS", "action": args.action, **result,
                                        "contract_sha256": c["contract_sha256"], "counts": dict(counts),
                                        "elapsed_seconds": time.monotonic() - start, **_resources(device)})
    except BaseException as exc:
        _json(args.out / "STOP.json", {"status": "STOP", "error": repr(exc), "traceback": traceback.format_exc(),
                                      "action": args.action, "arm": args.arm, "counts": dict(counts),
                                      "elapsed_seconds": time.monotonic() - start, **_resources(device)})
        raise


if __name__ == "__main__":
    main()
