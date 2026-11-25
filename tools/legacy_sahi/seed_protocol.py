"""Seed-bound R18 training and fixed SAHI events on the original SSD channel.

Only this new namespace is extended. The authenticated stream envelope, 1 MiB
transaction limit, persistent observer and archive implementation are unchanged.
"""
import math
from pathlib import Path

from sparse_rtdetr.localization import channel
from sparse_rtdetr.localization.runio import digest, publish, read, reference, require

TASK = "r18-sahi-seed12-20261016-001"
PROTOCOL = "r18-sahi-paired-seed-progress-v1"
ARM = "RT-DETRv2-R18-common-protocol-v1"
BINDINGS = ("source_sha256", "contract_sha256", "plan_sha256", "data_sha256", "material_binding_sha256")
SAHI = {"accept-A": (16, 1), "accept-B": (16, 1), "accept-C": (16, 5), "full-C": (137, 5)}
INFERENCE = ("A548", "infer-A")
TRAINING = ("cpu-state", "gpu-acceptance", "formal")
MODES = ("cpu-spawn", "cpu-state", "cpu-protocol", "gpu-acceptance", "formal", *INFERENCE, *SAHI, "evaluation")


def scope(b):
    seed = b.get("seed")
    require(type(seed) is int and seed in (1, 2), "paired seed identity differs")
    require(b.get("task_id") == TASK + "-seed" + str(seed) and b.get("arm") == ARM, "paired task/arm identity differs")
    require(all(channel._hex64(b.get(k)) for k in BINDINGS), "paired frozen binding missing")
    require(b.get("mode") in MODES, "paired mode differs")
    mode = b.get("protocol_mode", b["mode"])
    require(mode in MODES, "paired protocol mode differs")
    start = b.get("starting_cursor")
    require(type(start) is int and 0 <= start <= 36480, "paired starting cursor absent")
    if b.get("CPU_protocol_only"):
        require(b["mode"] == "cpu-protocol", "CPU protocol simulation escaped CPU namespace")
    else:
        require("protocol_mode" not in b and start == (0 if mode in TRAINING else 36480 if mode in (*INFERENCE, *SAHI, "evaluation") else 0), "fresh run cursor/scope differs")
    if mode in (*INFERENCE, *SAHI, "evaluation"):
        auth = b.get("evaluation_authorization")
        require(isinstance(auth, dict) and channel._hex64(auth.get("checkpoint_sha256")), "paired endpoint authority missing")
        ids = auth.get("image_ids")
        require(isinstance(ids, list) and len(ids) == len(set(ids)) == 548 and all(channel._hex64(i) for i in ids), "paired development image scope differs")
    return mode, start


def validate_audit(p, b):
    ref = p.get("audit")
    require(isinstance(ref, dict), "paired scientific audit reference absent")
    path = Path(ref["path"])
    root = Path(b["output_root"]) / "audits"
    require(path.is_absolute() and path.is_relative_to(root) and not path.is_symlink()
            and path.resolve().is_relative_to(root.resolve()), "paired audit path escaped bound output")
    require(reference(path) == ref, "paired scientific audit missing/corrupt")
    value = read(path)
    for key in ("task_id", "seed", *BINDINGS):
        require(value.get(key) == b[key], "paired audit identity differs: " + key)
    result = value.get("result")
    require(isinstance(result, dict) and value.get("update") == p.get("update") == result.get("update")
            and value.get("role") == p.get("audit_role") and result.get("input_sha256") == p.get("input_sha256")
            and channel._hex64(p.get("input_sha256")), "paired audit update/input differs")
    require(result.get("finite") is True and p.get("finite") is True
            and type(result.get("loss")) in (int, float) and math.isfinite(result["loss"])
            and result["loss"] == p.get("loss"), "paired scientific audit loss/finite differs")
    return value


def save_audit(out, b, result, role):
    """Preserve every scientific field, atomically publish, send a small reference."""
    require(b.get("progress_protocol") == PROTOCOL, "new audit writer requires paired namespace")
    scope(b)
    value = {**{k: b[k] for k in ("task_id", "seed", *BINDINGS)}, "update": result["update"], "role": role, "result": result}
    path = Path(out) / "audits" / f"update-{result['update']:05d}-{role}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    publish(path, value)
    payload = {"update": result["update"], "audit_role": role, "input_sha256": result["input_sha256"],
               "audit": reference(path), "loss": result["loss"], "finite": result["finite"],
               "completed_epochs": result["update"] // 304}
    validate_audit(payload, b)
    from sparse_rtdetr.localization.runio import event
    event(out, "update_complete", **payload)
    return payload


def _validate(p, b, stream):
    mode, start = scope(b)
    require(isinstance(p, dict) and type(p.get("monotonic")) in (int, float)
            and math.isfinite(p["monotonic"]), "paired finite clock absent")
    if stream != "progress":
        return None
    for key in ("task_id", "seed", *BINDINGS):
        if key in p:
            require(p[key] == b[key], "paired payload binding differs: " + key)
    if mode in (*INFERENCE, *SAHI, "evaluation"):
        require(not {"update", "completed_updates", "completed_epochs", "epoch", "optimizer_updates", "ema_updates"}.intersection(p),
                "paired inference/evaluation cannot publish training progress")
        for key in ("detector_updates", "backward_calls"):
            if key in p:
                require(type(p[key]) is int and p[key] == 0, "paired inference/evaluation mutation count differs")
    kind = p.get("event")
    if kind == "cuda_initialized":
        require(mode in ("gpu-acceptance", "formal", *INFERENCE, *SAHI) and not b.get("CPU_protocol_only"), "paired CPU CUDA event refused")
    elif kind == "cpu_input_window_complete":
        n = p.get("cpu_window_number")
        require(mode == "cpu-spawn" and type(n) is int and n in (1, 2)
                and p.get("input", {}).get("ids") == b["fixed_ids"][n - 1]
                and p.get("CUDA_initialized") is False, "paired CPU input identity/CUDA differs")
    elif kind == "epoch_begin":
        e = p.get("epoch"); u = p.get("completed_updates")
        require(mode == "formal" and type(e) is int and 1 <= e <= 120
                and type(u) is int and u == (e - 1) * 304 and p.get("completed_epochs") == e - 1,
                "paired epoch/cursor inconsistent")
        require(isinstance(p.get("dropped_ids"), list) and len(p["dropped_ids"]) == len(set(p["dropped_ids"])) == 5,
                "paired epoch dropped IDs differ")
    elif kind == "update_complete":
        u = p.get("update"); role = p.get("audit_role")
        require(mode in TRAINING and type(u) is int and start < u <= (36480 if mode == "formal" else 2), "paired update scope differs")
        require(role == "direct" if mode == "formal" else role in ("direct", "replay"), "paired update role differs")
        validate_audit(p, b)
    elif kind in ("checkpoint_begin", "checkpoint_complete"):
        u = p.get("update"); name = p.get("name")
        require(mode in TRAINING and type(u) is int and start <= u <= (36480 if mode == "formal" else 2)
                and isinstance(name, str) and name == Path(name).name and name.endswith(".pth"), "paired checkpoint scope differs")
        if kind == "checkpoint_complete":
            cp = p.get("checkpoint")
            require(isinstance(cp, dict) and cp.get("path") == str(Path(b["output_root"]) / name)
                    and cp.get("completed_updates") == u and channel._hex64(cp.get("sha256")), "paired checkpoint completion binding differs")
            require(reference(cp["path"]) == {"path": cp["path"], "sha256": cp["sha256"], "bytes": cp["size_bytes"]}, "paired checkpoint bytes changed")
    elif kind == "evaluation_image_saved":
        auth = b["evaluation_authorization"]; n = p.get("evaluation_image_number")
        require(mode in INFERENCE and type(n) is int and 1 <= n <= 548
                and p.get("image_id") == auth["image_ids"][n - 1]
                and p.get("checkpoint_sha256") == auth["checkpoint_sha256"], "paired inference image/endpoint differs")
    elif kind in ("sahi_warmup_complete", "sahi_batch_complete", "sahi_complete"):
        require(mode in SAHI, "paired SAHI event phase differs")
        batches, factor = SAHI[mode]
        if kind == "sahi_warmup_complete":
            require(p.get("warmup_original_batches") == 2 and p.get("forward_calls") == factor * 2, "paired warmup workload differs")
        elif kind == "sahi_batch_complete":
            n = p.get("batch_number")
            require(type(n) is int and 1 <= n <= batches and p.get("original_images") == n * 4
                    and p.get("forward_calls") == (n + 2) * factor, "paired SAHI batch workload differs")
            ids = b["evaluation_authorization"]["image_ids"] if mode == "full-C" else b["fixed64"]
            if "image_ids" in p:
                require(p["image_ids"] == ids[(n - 1) * 4:n * 4], "paired SAHI image order differs")
        else:
            require(p.get("detector_updates") == p.get("backward_calls") == 0 and p.get("state_unchanged") is True
                    and p.get("forward_calls") == factor * (batches + 2), "paired SAHI fixed work/state differs")
    elif kind == "evaluation_budget_complete":
        n = p.get("evaluation_budget_number", p.get("evaluation_number", p.get("check_number")))
        require(mode == "evaluation" and type(n) is int and 1 <= n <= 6
                and p.get("arm") == ("A", "B", "C")[(n - 1) // 2]
                and p.get("budget") == (600 if n % 2 else 300) and p.get("images") == 548
                and p.get("checkpoint_sha256") == b["evaluation_authorization"]["checkpoint_sha256"], "paired evaluation budget/order differs")
    elif kind == "seed_phase_complete":
        require(p.get("phase") == b["mode"] and type(p.get("actual_updates")) is int
                and p["actual_updates"] == (36480 if mode == "formal" else 3 if mode in ("cpu-state", "gpu-acceptance") else 0), "paired terminal workload differs")
        if "CUDA_initialized" in p and (mode.startswith("cpu") or mode == "evaluation"):
            require(p["CUDA_initialized"] is False, "paired CPU phase initialized CUDA")
        if "result" in p:
            ref = p["result"]
            require(isinstance(ref, dict) and ref.get("path") == str(Path(b["output_root"]) / "RESULT.json")
                    and reference(ref["path"]) == ref, "paired terminal result missing/corrupt")
            value = read(ref["path"])
            require(value.get("status") in ("PASS", "R18_COMMON_E120_EMA_LOCKED"), "paired terminal engineering result failed")
    else:
        raise RuntimeError("unapproved paired event: " + str(kind))
    return digest([PROTOCOL, b["task_id"], b["plan_sha256"], mode, kind,
                   p.get("update", p.get("completed_updates")), p.get("audit_role"), p.get("name"),
                   p.get("epoch"), p.get("cpu_window_number"), p.get("evaluation_image_number"),
                   p.get("batch_number"), p.get("evaluation_budget_number", p.get("evaluation_number", p.get("check_number")))])


def _advance(reader, p):
    b = reader.bound_identity; mode, start = scope(b); kind = p["event"]; clock = reader.clock
    cursor = clock.get("_paired_cursor", start)
    if kind == "cuda_initialized":
        require(not clock.get("_paired_cuda"), "paired repeated CUDA initialization")
        clock["_paired_cuda"] = True
    elif kind == "epoch_begin":
        require(p["completed_updates"] == cursor and clock.get("_paired_saved") == cursor, "paired epoch before checkpoint/missing updates")
        previous = clock.get("_paired_epoch", (start + 303) // 304 if start else 0)
        require(p["epoch"] == previous + 1, "paired epoch skipped/regressed")
        clock["_paired_epoch"] = p["epoch"]
    elif kind == "update_complete":
        if mode in ("cpu-state", "gpu-acceptance"):
            n = clock.get("_paired_steps", 0); sequence = [(1, "direct"), (2, "direct"), (2, "replay")]
            require(n < 3 and (p["update"], p["audit_role"]) == sequence[n], "paired fixed3 update sequence differs")
            if n == 2:
                require(clock.get("_paired_saved") == 2 and clock.get("_paired_saved_one"), "paired replay before checkpoint closure")
        else:
            require(p["update"] == cursor + 1 and clock.get("_paired_epoch", (start // 304 + 1) if start % 304 else 0) == (p["update"] - 1) // 304 + 1,
                    "paired formal update skipped/epoch context differs")
        if mode != "cpu-state" and not b.get("CPU_protocol_only"):
            require(clock.get("_paired_cuda"), "paired update before CUDA admission")
        require(not clock.get("_paired_open"), "paired update while checkpoint open")
        clock["_paired_steps"] = clock.get("_paired_steps", 0) + 1; clock["_paired_cursor"] = p["update"]
    elif kind == "checkpoint_begin":
        require(p["update"] == cursor and not clock.get("_paired_open"), "paired checkpoint wrong cursor/already open")
        clock["_paired_open"] = [p["name"], p["update"]]
    elif kind == "checkpoint_complete":
        require(clock.get("_paired_open") == [p["name"], p["update"]] and p["update"] == cursor, "paired checkpoint without matching begin")
        clock["_paired_open"] = None; clock["_paired_saved"] = cursor
        if cursor == 1:
            clock["_paired_saved_one"] = True
    elif kind == "evaluation_image_saved":
        require(p["evaluation_image_number"] == clock.get("_paired_images", 0) + 1, "paired inference image skipped")
        clock["_paired_images"] = p["evaluation_image_number"]
    elif kind == "sahi_warmup_complete":
        require(clock.get("_paired_cuda") and not clock.get("_paired_warmup") and not clock.get("_paired_batches"), "paired warmup phase/order differs")
        clock["_paired_warmup"] = True
    elif kind == "sahi_batch_complete":
        require(clock.get("_paired_cuda") and clock.get("_paired_warmup") and not clock.get("_paired_sahi_done")
                and p["batch_number"] == clock.get("_paired_batches", 0) + 1, "paired SAHI batch skipped/regressed")
        clock["_paired_batches"] = p["batch_number"]
    elif kind == "sahi_complete":
        require(clock.get("_paired_batches") == SAHI[mode][0], "paired SAHI incomplete")
        clock["_paired_sahi_done"] = True
    elif kind == "evaluation_budget_complete":
        n = p.get("evaluation_budget_number", p.get("evaluation_number", p.get("check_number")))
        require(n == clock.get("_paired_budgets", 0) + 1, "paired evaluation budget skipped/regressed")
        clock["_paired_budgets"] = n
    elif kind == "seed_phase_complete":
        require(not clock.get("_paired_open"), "paired phase closed with open checkpoint")
        if mode == "formal":
            require(cursor == clock.get("_paired_saved") == 36480 and clock.get("_paired_epoch", 120 if start % 304 and start // 304 == 119 else 0) == 120
                    and clock.get("_paired_steps", 0) == 36480 - start, "paired formal endpoint incomplete")
        elif mode in ("cpu-state", "gpu-acceptance"):
            require(clock.get("_paired_steps") == 3 and clock.get("_paired_saved") == 2, "paired acceptance incomplete")
        elif mode == "cpu-spawn":
            require(clock.get("_cpu_window") == 2, "paired real spawn incomplete")
        elif mode in INFERENCE:
            require(clock.get("_paired_images") == 548, "paired548 inference incomplete")
        elif mode in SAHI:
            require(clock.get("_paired_sahi_done"), "paired SAHI terminal before closure")
        elif mode == "evaluation":
            require(clock.get("_paired_budgets") == 6, "paired six evaluations incomplete")
        clock["_paired_closed"] = True


def install():
    if getattr(channel, "_r18_paired_seed_installed", False):
        return
    original = channel.validate_event; Parent = channel.Reader

    def validate(p, b, stream):
        return _validate(p, b, stream) if b.get("progress_protocol") == PROTOCOL else original(p, b, stream)

    class Reader(Parent):
        def _accept(self, record):
            paired = self.bound_identity.get("progress_protocol") == PROTOCOL and self.stream == "progress"
            if paired and self.clock.get("_paired_closed") and record.get("seq") not in self.seen:
                raise RuntimeError("paired event after phase closure")
            useful = super()._accept(record)
            if useful and paired:
                _advance(self, record["payload"])
            return useful

    channel.validate_event = validate; channel.Reader = Reader
    channel._r18_paired_seed_installed = True
