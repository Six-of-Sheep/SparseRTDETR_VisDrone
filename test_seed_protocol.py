"""Real temporary channel/observer/archive tests; no detector or CUDA calls.

All synthetic scientific/checkpoint payloads here are explicitly protocol-only;
these tests never substitute for CPU state or the three actual GPU updates.
"""
import copy
import os
import sys
import time
from pathlib import Path

from sparse_rtdetr.localization import channel
from sparse_rtdetr.localization.runio import canonical, digest, publish, read, reference, require, write
from seed_protocol import ARM, BINDINGS, PROTOCOL, TASK, SAHI, install


def run(root, c):
    install()
    root = Path(root); root.mkdir(exist_ok=False)
    checks = []
    seed = c["seed"]
    full = [digest(["CPU_PROTOCOL_ONLY", "development", n]) for n in range(548)]

    def ident(mode="cpu-state", start=0, *, simulated=False):
        result = {"task_id": TASK + "-seed" + str(seed), "seed": seed, "arm": ARM,
                  **{k: c.get(k, digest(["CPU_PROTOCOL_ONLY", k])) for k in BINDINGS},
                  "progress_protocol": PROTOCOL, "mode": "cpu-protocol" if simulated else mode,
                  "starting_cursor": start, "supervisor_pid": os.getpid(),
                  "fixed_ids": c["fixed_ids"], "fixed64": full[:64]}
        result["plan_sha256"] = c["plan"]["sha256"]
        if simulated:
            result.update(CPU_protocol_only=True, protocol_mode=mode)
        if mode in ("infer-A", "A548", *SAHI, "evaluation"):
            result["starting_cursor"] = 36480
            result["evaluation_authorization"] = {"checkpoint_sha256": digest("CPU_PROTOCOL_ONLY_CHECKPOINT"), "image_ids": full}
        return result

    def fixture(name, mode="cpu-state", start=0, simulated=False, mutate=None):
        b = ident(mode, start, simulated=simulated)
        if mutate:
            mutate(b)
        out = root / (name + "-out"); out.mkdir()
        b["output_root"] = str(out)
        bound = channel.create(root / name, b)
        return bound, channel.Writer(bound, "progress")

    def event(kind, **kw):
        return {"event": kind, "monotonic": time.monotonic(), **kw}

    def state(w):
        return (w.path.read_bytes(), w.reader.offset, copy.deepcopy(w.reader.clock), w.reader.semantic,
                w.sidecar.read_bytes() if w.sidecar.exists() else None)

    def refuse(w, p, name):
        before = state(w)
        try:
            w.append(p)
        except (RuntimeError, ValueError, KeyError, TypeError, FileNotFoundError):
            pass
        else:
            raise AssertionError("protocol accepted negative: " + name)
        require(before == state(w), "protocol refusal mutated stream/cursor/sidecar: " + name)
        checks.append(name)

    def audit(b, u, role="direct"):
        r = {"TEST_ONLY_PROTOCOL_FIXTURE": True, "update": u, "input_sha256": digest(["input", u]), "loss": 1., "finite": True,
             "all_original_scientific_fields": {"gradient": "retained", "DN": "retained", "normalizers": [1, 2]}}
        path = Path(b["output_root"]) / "audits" / f"update-{u:05d}-{role}.json"
        publish(path, {**{k: b[k] for k in ("task_id", "seed", *BINDINGS)}, "update": u, "role": role, "result": r})
        return event("update_complete", update=u, completed_epochs=u // 304, input_sha256=r["input_sha256"],
                     audit_role=role, audit=reference(path), loss=1., finite=True)

    def checkpoint(b, w, u, name):
        w.append(event("checkpoint_begin", name=name, update=u, completed_epochs=u // 304))
        path = Path(b["output_root"]) / name
        write(path, {"TEST_ONLY_PROTOCOL_CHECKPOINT_NOT_TORCH": True, "u": u, "name": name})
        ref = reference(path)
        p = event("checkpoint_complete", name=name, update=u, completed_epochs=u // 304,
                  checkpoint={"path": ref["path"], "sha256": ref["sha256"], "size_bytes": ref["bytes"], "completed_updates": u})
        w.append(p)

    def close(b, mode=None):
        mode = mode or b["mode"]
        effective = b.get("protocol_mode", mode)
        return event("seed_phase_complete", phase=mode, seed=seed,
                     actual_updates=36480 if effective == "formal" else 3 if effective in ("cpu-state", "gpu-acceptance") else 0)

    # Exact accepted records can be observed again, but a repeated operation with
    # different scientific content must fail atomically and cannot count progress.
    b, w = fixture("duplicate")
    p = audit(b, 1)
    w.append(p); w.append({**p, "monotonic": time.monotonic()})
    require(w.reader.semantic == 1, "identical body advanced semantic progress")
    checks.append("identical_duplicate_does_not_feed_progress")
    refuse(w, {**p, "monotonic": time.monotonic(), "contradictory": True}, "conflicting_duplicate_atomic_refusal")
    refuse(w, audit(b, 2, "replay"), "replay_before_direct_refusal")
    refuse(w, close(b), "early_terminal_refusal")

    # Complete protocol of update1/save, update2/save, restore/replay2/save.
    b, w = fixture("fixed3")
    w.append(audit(b, 1)); checkpoint(b, w, 1, "C1.pth")
    w.append(audit(b, 2)); checkpoint(b, w, 2, "C2-direct.pth")
    w.append(audit(b, 2, "replay")); checkpoint(b, w, 2, "C2-replay.pth")
    w.append(close(b))
    require(w.reader.clock["_paired_closed"], "valid fixed3 did not close")
    refuse(w, close(b), "postclosure_refusal")
    fresh = channel.Reader(w.path, b, "progress"); fresh.poll()
    require(fresh.clock == w.reader.clock, "fresh independent Reader disagrees")
    checks.append("fixed3_complete_external_audit_readback")

    for name, mutate in (("wrong-task", lambda b: b.update(task_id="foreign")),
                         ("wrong-seed", lambda b: b.update(seed=3)),
                         ("wrong-arm", lambda b: b.update(arm="R0")),
                         ("missing-data-binding", lambda b: b.pop("data_sha256"))):
        b, w = fixture(name, "cpu-spawn", mutate=mutate)
        refuse(w, event("cpu_input_window_complete", cpu_window_number=1,
                        input={"ids": c["fixed_ids"][0]}, CUDA_initialized=False), name)
    b, w = fixture("spawn", "cpu-spawn")
    refuse(w, event("cuda_initialized"), "CPU_CUDA_refusal")
    refuse(w, event("cpu_input_window_complete", cpu_window_number=2,
                    input={"ids": c["fixed_ids"][1]}, CUDA_initialized=False), "spawn_gap_refusal")
    for n in (1, 2):
        w.append(event("cpu_input_window_complete", cpu_window_number=n, input={"ids": c["fixed_ids"][n - 1]}, CUDA_initialized=False))
    w.append(close(b)); checks.append("two_window_spawn_protocol")

    for name, change in (("audit-corrupt", "corrupt"), ("audit-missing", "missing"), ("audit-wrong-binding", "binding")):
        b, w = fixture(name); p = audit(b, 1); path = Path(p["audit"]["path"])
        if change == "corrupt":
            path.write_bytes(path.read_bytes() + b"corrupt")
        elif change == "missing":
            path.rename(path.with_suffix(".preserved"))
        else:
            value = read(path); value["seed"] = 2 if seed == 1 else 1
            from sparse_rtdetr.localization.runio import atomic
            atomic(path, value); p["audit"] = reference(path)
        refuse(w, p, name)

    # The production fresh formal sequence must save the independently created
    # untrained state before epoch1. No GPU/model operations occur in this
    # explicitly CPU-only synthetic channel.
    b, w = fixture("fresh-formal", "formal", start=0, simulated=True)
    refuse(w, event("epoch_begin", epoch=1, completed_updates=0, completed_epochs=0,
                    dropped_ids=list("abcde")), "fresh_epoch_before_e000_refusal")
    checkpoint(b, w, 0, "checkpoint-epoch-000.pth")
    first_update = audit(b, 1)
    refuse(w, first_update, "fresh_update_before_epoch1_refusal")
    w.append(event("epoch_begin", epoch=1, completed_updates=0, completed_epochs=0, dropped_ids=list("abcde")))
    w.append({**first_update, "monotonic": time.monotonic()})
    require(w.reader.clock["_paired_cursor"] == 1 and w.reader.clock["_paired_epoch"] == 1,
            "fresh formal cp0/epoch1/update1 cursor differs")
    checks.append("fresh_formal_e000_epoch1_update1")

    # Explicit protocol-only simulation starts at303, so only two synthetic
    # updates are needed to check the actual epoch identity304 ->305 transition.
    b, w = fixture("epoch304-305", "formal", start=303, simulated=True)
    w.append(audit(b, 304)); checkpoint(b, w, 304, "e001.pth")
    refuse(w, event("epoch_begin", epoch=2, completed_updates=303, completed_epochs=1,
                    dropped_ids=list("abcde")), "epoch_cursor_mismatch_refusal")
    w.append(event("epoch_begin", epoch=2, completed_updates=304, completed_epochs=1, dropped_ids=list("abcde")))
    refuse(w, audit(b, 306), "formal_missing305_refusal")
    w.append(audit(b, 305)); refuse(w, close(b), "e002_cannot_claim_e120")
    checks.append("cursor304_to305_cross_epoch")

    b, w = fixture("endpoint", "formal", start=36479, simulated=True)
    w.append(audit(b, 36480)); checkpoint(b, w, 36480, "e120.pth"); w.append(close(b))
    require(w.reader.clock["_paired_closed"], "synthetic terminal did not close")
    b, w = fixture("evaluation", "evaluation")
    cp = b["evaluation_authorization"]["checkpoint_sha256"]
    refuse(w, event("update_complete", update=1), "evaluation_cannot_train")
    for n in range(1, 7):
        w.append(event("evaluation_budget_complete", evaluation_number=n, arm=("A", "B", "C")[(n - 1) // 2],
                       budget=600 if n % 2 else 300, images=548, checkpoint_sha256=cp))
    w.append(close(b)); checks.append("e120_to_six_ABC_evaluations")

    for mode, (batches, factor) in SAHI.items():
        b, w = fixture("sahi-" + mode, mode)
        w.append(event("cuda_initialized"))
        refuse(w, event("sahi_batch_complete", batch_number=1, original_images=4, forward_calls=3 * factor), mode + "_batch_before_warmup")
        w.append(event("sahi_warmup_complete", warmup_original_batches=2, forward_calls=2 * factor))
        for n in range(1, batches + 1):
            w.append(event("sahi_batch_complete", batch_number=n, original_images=n * 4, forward_calls=(n + 2) * factor))
        w.append(event("sahi_complete", detector_updates=0, backward_calls=0, state_unchanged=True, forward_calls=(batches + 2) * factor))
        w.append(close(b)); checks.append(mode + "_fixed_workload_protocol_only")

    # Final-envelope boundary: one byte over1MiB must change no stream, sidecar,
    # cursor or semantic progress. The historical ceiling is not increased.
    for excess in (0, 1):
        b, w = fixture("boundary-" + str(excess), "cpu-protocol")
        p = {**close(b), "padding": ""}
        record = {"identity": digest(b), "stream": "progress", "writer_pid": os.getpid(), "seq": 1,
                  "previous": None, "payload": p}; record["sha256"] = digest(record)
        p["padding"] = "x" * (1048576 + excess - len(canonical(record) + b"\n"))
        if excess:
            refuse(w, p, "serialized1MiB_plus1_atomic_refusal")
        else:
            w.append(p); require(w.path.stat().st_size == 1048576, "1MiB exact boundary differed")
            checks.append("serialized1MiB_exact_acceptance")

    # Real event/resource readers and immutable archive. Resource content below
    # is labelled TEST_ONLY and is never passed to resource admission.
    b, w = fixture("archive", "cpu-spawn")
    rw = channel.Writer(b, "resources"); rw.append({"monotonic": time.monotonic(), "TEST_ONLY_SYNTHETIC_RESOURCE_NOT_ADMISSION": True})
    for n in (1, 2):
        w.append(event("cpu_input_window_complete", cpu_window_number=n, input={"ids": c["fixed_ids"][n - 1]}, CUDA_initialized=False))
    w.append(close(b))
    if os.name == "posix":
        latest = channel.latest(b["runtime_root"], "progress")
        require(latest["event"] == "seed_phase_complete" and channel.latest(b["runtime_root"], "progress") == latest, "persistent observer differs")
        checks.append("persistent_observer_latest")
    for name in ("worker.log", "activity.jsonl", "supervisor-start.json", "worker-exit.json", "supervisor-exit.json"):
        write(Path(b["runtime_root"]) / name, {"TEST_ONLY_PROTOCOL_LIFECYCLE_RECEIPT": True})
    out = root / "verified-archive"; out.mkdir()
    channel.archive(b["runtime_root"], out); channel.verify_archive(b["runtime_root"], out, b)
    try:
        channel.verify_archive(b["runtime_root"], out, {**b, "seed": 9})
    except RuntimeError:
        checks.append("archive_wrong_identity_refusal")
    else:
        raise AssertionError("archive accepted wrong identity")
    path = out / "runtime-archive/worker.log"; path.write_bytes(path.read_bytes() + b"damage")
    try:
        channel.verify_archive(b["runtime_root"], out, b)
    except RuntimeError:
        checks.append("archive_SHA_damage_refusal")
    else:
        raise AssertionError("archive accepted SHA damage")
    checks.append("real_SSD_channel_archive_roundtrip")
    initialized = "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized()
    require(not initialized, "protocol tests initialized CUDA")
    return {"status": "PASS", "seed": seed, "checks": checks, "count": len(checks), "CUDA_initialized": False,
            "model_constructed": False, "actual_training_updates": 0, "synthetic_protocol_events_are_not_GPU_evidence": True,
            "persistent_observer_platform": os.name}
