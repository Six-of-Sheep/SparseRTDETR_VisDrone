"""CPU-only production-entry regression for the paired SAHI lifecycle.

Run with the certified laboratory Python and a new SSD --root. Each phase uses
independent supervisor and worker processes, calling new_entry.main. Only the
source/material loader, hardware admission, tensor device adapter, tiny synthetic
engine/checkpoint, image source and expensive metric backend are injected. The
production inference/evaluation loops, emitter, Writer, Reader, terminal checks,
supervise readback and archive/verification code are never replaced. Synthetic
metrics/checkpoint/outputs here are explicitly NOT scientific/GPU evidence.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from sparse_rtdetr.localization import channel
from sparse_rtdetr.localization.runio import canonical, digest, publish, read, reference, require, write
from seed_protocol import ARM, PROTOCOL, TASK, BINDINGS, install

PHASES = ("infer-A", "accept-A", "accept-B", "accept-C", "full-C", "evaluation")
MARKER = "TEST_ONLY_CPU_SYNTHETIC_NOT_SCIENTIFIC_EVIDENCE"


def fixture(root):
    root.mkdir(parents=True, exist_ok=False)
    job = root / "job" / "seed1"
    job.mkdir(parents=True)
    ids = [digest([MARKER, "image", i]) for i in range(548)]
    c = {"task_id": TASK + "-seed1", "seed": 1, "arm": ARM,
         "scope": "formal_R18_seed1_e120", "source_root": str(ROOT),
         "job_root": str(job), "runtime_root": str(root / "ssd" / "seed1"),
         "python": sys.executable, "gpu_uuid": "SYNTHETIC-NO-GPU", "boot_id": MARKER,
         "source_sha256": digest([MARKER, "source"]),
         "contract_sha256": digest([MARKER, "contract"]),
         "data_sha256": digest([MARKER, "data"]), "data": {}, "pretrained": {},
         "GPU_execution_authorized": True, "formal_training_authorized": True,
         "config": {}, "environment": {"CPU_fixture": True}, "TEST_ONLY": MARKER,
         "development_ids": ids, "fixed64": ids[:64],
         "fixed_ids": [ids[:16], ids[16:32]], "contract_path": str(root / "contract.json")}
    write(root / "plan.json", {"TEST_ONLY": MARKER})
    c["plan"] = reference(root / "plan.json")
    write(root / "cpu-gate.json", {"status": "PASS", "TEST_ONLY": MARKER,
                                  "seed": 1, "source_sha256": c["source_sha256"]})
    c["CPU_gate"] = reference(root / "cpu-gate.json")
    acc = job / "gpu-acceptance"
    acc.mkdir()
    write(acc / "RESULT.json", {"status": "PASS", "TEST_ONLY": MARKER,
          "actual_updates": 3, "sequence": [1, 2, 2], "same_next_result_exact": True,
          "complete_state_RNG_exact": True, "state_usable_for_formal": False})
    write(acc / "runtime-archive-index.json", {"TEST_ONLY": MARKER})
    write(acc / "execution-binding.json", {**{k: c[k] for k in
          ("task_id", "seed", "source_sha256", "contract_sha256")}, "mode": "gpu-acceptance"})
    write(acc / "CLOSED.json", {"status": "PASS", "seed": 1, "mode": "gpu-acceptance",
          "result": reference(acc / "RESULT.json"),
          "runtime_index": reference(acc / "runtime-archive-index.json"), "TEST_ONLY": MARKER})
    formal = job / "formal"
    formal.mkdir()
    write(formal / "binding.json", {"TEST_ONLY": MARKER})
    write(formal / "checkpoint-fixture.json", {"TEST_ONLY": MARKER})
    cp = {**reference(formal / "checkpoint-fixture.json"), "completed_updates": 36480}
    endpoint = {"status": "R18_COMMON_E120_EMA_LOCKED", "seed": 1,
                "epoch": 120, "updates": 36480, "actual_updates": 36480,
                "used_acceptance_state": False, "checkpoint": cp,
                "binding": reference(formal / "binding.json"), "TEST_ONLY": MARKER,
                **{k: c[k] for k in ("source_sha256", "contract_sha256")}}
    write(formal / "endpoint-lock.json", endpoint)
    write(formal / "CLOSED.json", {"status": "PASS", "TEST_ONLY": MARKER})
    write(root / "contract.json", c)
    return c


def adapters(root, mode, worker):
    """Install CPU fixture backends before entering the unmodified dispatcher."""
    import numpy as np
    import torch
    import new_entry
    import new_runtime
    import infer_seed
    from sparse_rtdetr.localization import worker as base, guard
    from sparse_rtdetr.protocol_r18 import checkpoint, geometry, worker as native
    from sparse_rtdetr.r0 import evaluation

    c = read(root / "contract.json")
    require(c.get("TEST_ONLY") == MARKER, "CPU adapter refuses a production contract")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_initialized(),
            "CPU entry adapter must have CUDA hidden")
    torch.set_num_threads(2)
    rows = [{"stable_image_id": i, "width": 120, "height": 80}
            for i in c["development_ids"]]
    data = SimpleNamespace(records=rows, by_id={r["stable_image_id"]: r for r in rows},
                           raw=lambda row: {"rows": [], "TEST_ONLY": MARKER})

    class TinyCPU(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.marker = torch.nn.Parameter(torch.ones(()), requires_grad=True)
            self.decoder = SimpleNamespace(decoder=SimpleNamespace(layers=[]))
            self.register_buffer("logits", torch.zeros(300, 10))
            self.register_buffer("boxes", torch.tensor([.5, .5, .25, .25]).repeat(300, 1))

        def forward(self, images):
            return {"pred_logits": self.logits.expand(len(images), -1, -1).clone(),
                    "pred_boxes": self.boxes.expand(len(images), -1, -1).clone()}

    def make_engine(contract, runtime, scope):
        model = TinyCPU()
        engine = SimpleNamespace(model=model, optimizer=SimpleNamespace(step=lambda: None),
                 ema=SimpleNamespace(module=copy.deepcopy(model), update=lambda: None, updates=36480),
                 completed_updates=36480)
        return engine, read(Path(c["job_root"]) / "formal/binding.json"), MARKER

    def capture(e, binding):
        return {"TEST_ONLY": MARKER, "model": copy.deepcopy(e.model.state_dict()),
                "ema": copy.deepcopy(e.ema.module.state_dict()),
                "training": [e.model.training, e.ema.module.training],
                "updates": [e.completed_updates, e.ema.updates],
                "rng": checkpoint.capture_rng(torch.device("cpu"))}

    def restore(e, binding, path, sha):
        require(reference(path)["sha256"] == sha and read(path)["TEST_ONLY"] == MARKER,
                "CPU fixture checkpoint bytes changed")
        return {"full_state_exact": True, "TEST_ONLY": MARKER}

    def prepare(contract, out):
        # Synthetic admission replaces hardware only. Production code owns every
        # workload/completion event; no test code invents either terminal event.
        from sparse_rtdetr.localization.runio import event
        event(out, "cuda_initialized", TEST_ONLY=MARKER, actual_CUDA_initialized=False)
        return {"TEST_ONLY": MARKER}

    def view_pixels(dataset, batch_rows, sliced):
        views = [[torch.zeros(3, 1, 1) for row in batch_rows]
                 for _ in range(5 if sliced else 1)]
        return views, [[t.as_dict() for t in infer_seed.tile_windows(r["height"], r["width"])]
                       for r in batch_rows]

    def score(rows, parsed, predictions, repo_root):
        require(len(rows) == len(parsed) == 548 and predictions, "CPU six-budget input incomplete")
        return {"official": {"TEST_ONLY": MARKER}, "COCO_bridge": {
                "units": "0..1 AP, -1 means no GT in an area/class aggregation",
                "metrics": {k: .2 for k in infer_seed.METRICS}}}

    original_to = torch.Tensor.to
    def to_cpu(self, *args, **kwargs):
        if args and str(args[0]).startswith("cuda"):
            require(args[0] == "cuda:0" and len(args) == 1 and not kwargs,
                    "unexpected synthetic device transfer")
            return self
        return original_to(self, *args, **kwargs)

    def cpu_supervisor(cmd, out, env, **kw):
        out = Path(out)
        runtime = Path(kw["runtime"])
        require(not runtime.exists(), "CPU runtime already consumed")
        bound = channel.create(runtime, {**kw["run_identity"], "TEST_ONLY": MARKER})
        writer = channel.Writer(bound, "resources")
        writer.append({"monotonic": time.monotonic(), "TEST_ONLY": MARKER,
                       "not_resource_admission_evidence": True})
        for name in ("activity.jsonl", "supervisor-start.json"):
            write(runtime / name, {"TEST_ONLY": MARKER})
        child_env = {**env, **channel.environment(bound), "CUDA_VISIBLE_DEVICES": ""}
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "--root", str(root),
                   "--child", "worker", "--mode", mode]
        with (runtime / "worker.log").open("xb") as log:
            completed = subprocess.run(command, env=child_env, stdout=log, stderr=subprocess.STDOUT,
                                       check=False)
        write(runtime / "supervisor-exit.json", {"status": "PASS" if completed.returncode == 0 else "STOP",
              "remaining_live": [], "unknown_pids": [], "cleanup_errors": [],
              "returncode": completed.returncode, "TEST_ONLY": MARKER})
        channel.archive(runtime, out)
        return completed.returncode

    stack = contextlib.ExitStack()
    replacements = [(new_runtime, "load", lambda path: read(path)),
                    (infer_seed, "_dataset", lambda contract: data),
                    (infer_seed, "make_view_pixels", view_pixels),
                    (base, "prepare_cuda", prepare), (base, "live_admission", lambda *a: None),
                    (native, "make_engine", make_engine), (checkpoint, "capture", capture),
                    (checkpoint, "restore", restore), (evaluation, "score_budget", score),
                    (torch.Tensor, "to", to_cpu),
                    (torch, "autocast", lambda *a, **k: contextlib.nullcontext()),
                    (torch.cuda, "synchronize", lambda: None),
                    (torch.cuda, "reset_peak_memory_stats", lambda: None),
                    (torch.cuda, "max_memory_allocated", lambda: 0),
                    (torch.cuda, "max_memory_reserved", lambda: 0)]
    if not worker:
        replacements.append((guard, "supervise", cpu_supervisor))
    for obj, attr, value in replacements:
        stack.enter_context(patch.object(obj, attr, value))
    return stack, c


def run_child(root, action, mode):
    import torch
    import new_entry
    stack, c = adapters(root, mode, action == "worker")
    try:
        code = new_entry.main([action, "--contract", c["contract_path"], "--mode", mode])
        require(not torch.cuda.is_initialized(), "real CPU entry initialized CUDA")
        return code
    finally:
        stack.close()


def protocol_faults(root, c):
    """Corrupt production-generated records; do not generate the positive path."""
    install()
    checks = []
    for mode in PHASES:
        phase = Path(c["job_root"]) / mode
        original = [json.loads(x)["payload"] for x in
                    (phase / "runtime-archive/progress.jsonl").read_text().splitlines()]
        original_binding = read(phase / "execution-binding.json")
        closure_index = next((i for i, p in enumerate(original) if p["event"] == "sahi_complete"), None)
        terminal_index = len(original) - 1
        require(original[terminal_index]["event"] == "seed_phase_complete", "worker terminal absent")
        mutations = {"missing-phase-terminal": original[:-1],
                     "duplicate-phase-terminal": original + [original[-1]]}
        if closure_index is not None:
            require(closure_index == terminal_index - 1, "production completion order differs")
            mutations.update({"missing-sahi-terminal": original[:closure_index] + original[closure_index + 1:],
               "early-sahi-terminal": original[:2] + [original[closure_index]] + original[2:closure_index] + original[closure_index + 1:],
               "reversed-terminals": original[:closure_index] + [original[-1], original[closure_index]],
               "duplicate-sahi-terminal": original[:terminal_index] + [original[closure_index]] + original[terminal_index:]})
        if mode == "evaluation":
            mutations["missing-budget"] = original[:2] + original[3:]
            mutations["reordered-budget"] = [original[1], original[0]] + original[2:]
        for label, records in mutations.items():
            location = root / "negative" / (mode + "-" + label)
            bound = channel.create(location, {k: v for k, v in original_binding.items()
                                             if k not in ("schema", "runtime_root")})
            writer = channel.Writer(bound, "progress")
            rejected = False
            for record in records:
                before = (writer.path.read_bytes(), writer.reader.offset, copy.deepcopy(writer.reader.clock),
                          writer.sidecar.read_bytes() if writer.sidecar.exists() else None)
                try:
                    writer.append({**record, "monotonic": time.monotonic()})
                except RuntimeError:
                    after = (writer.path.read_bytes(), writer.reader.offset, copy.deepcopy(writer.reader.clock),
                             writer.sidecar.read_bytes() if writer.sidecar.exists() else None)
                    require(before == after, "Writer refusal changed stream/cursor/sidecar")
                    rejected = True
                    break
            if label == "missing-phase-terminal":
                reader = channel.Reader(writer.path, bound, "progress")
                last, _ = reader.poll()
                require(last is None or last["payload"]["event"] != "seed_phase_complete",
                        "missing terminal was treated as phase completion")
                require(not reader.clock.get("_paired_closed"), "missing terminal closed Reader")
                reject_missing_phase(root, c, mode, bound, records)
                rejected = True
            require(rejected, "negative production sequence accepted: " + mode + "/" + label)
            if label != "missing-phase-terminal":
                # Independent Reader and archive get an authentically enveloped
                # but semantically faulty stream, bypassing the Writer on purpose.
                rawroot = root / "negative-raw" / (mode + "-" + label)
                rawbound = channel.create(rawroot, {k: v for k, v in original_binding.items()
                                                   if k not in ("schema", "runtime_root")})
                previous = None
                blobs = []
                for seq, payload in enumerate(records, 1):
                    record = {"identity": digest(rawbound), "stream": "progress", "writer_pid": os.getpid(),
                              "seq": seq, "previous": previous,
                              "payload": {**payload, "monotonic": time.monotonic()}}
                    record["sha256"] = digest(record)
                    previous = record["sha256"]
                    blobs.append(canonical(record) + b"\n")
                (rawroot / "progress.jsonl").write_bytes(b"".join(blobs))
                independent = channel.Reader(rawroot / "progress.jsonl", rawbound, "progress")
                try:
                    independent.poll()
                except RuntimeError:
                    require(independent.offset == 0 and not independent.clock,
                            "failed Reader committed a partial faulty sequence")
                else:
                    raise AssertionError("independent Reader accepted " + mode + "/" + label)
                archiveout = root / "negative-archives" / (mode + "-" + label)
                archiveout.mkdir(parents=True)
                try:
                    channel.archive(rawroot, archiveout)
                except RuntimeError:
                    pass
                else:
                    raise AssertionError("archive accepted " + mode + "/" + label)
            checks.append(mode + "/" + label)
    return checks


def reject_missing_phase(root, c, mode, bound, records):
    """The real supervise postcondition must refuse a parsed but unclosed stream."""
    import new_entry
    import new_runtime
    from sparse_rtdetr.localization import guard
    job = root / "missing-phase-lifecycle" / mode / "job"
    runtimebase = root / "missing-phase-lifecycle" / mode / "ssd"
    out = job / mode
    out.mkdir(parents=True)
    runtime = runtimebase / mode
    copybound = channel.create(runtime, {k: v for k, v in bound.items()
                                        if k not in ("schema", "runtime_root")})
    writer = channel.Writer(copybound, "progress")
    for payload in records:
        writer.append({**payload, "monotonic": time.monotonic()})
    channel.Writer(copybound, "resources").append({"monotonic": time.monotonic(), "TEST_ONLY": MARKER})
    for name in ("worker.log", "activity.jsonl", "supervisor-start.json", "worker-exit.json"):
        write(runtime / name, {"TEST_ONLY": MARKER})
    write(runtime / "supervisor-exit.json", {"status": "PASS", "remaining_live": [],
          "unknown_pids": [], "cleanup_errors": [], "TEST_ONLY": MARKER})
    channel.archive(runtime, out)
    trial_c = {**c, "job_root": str(job), "runtime_root": str(runtimebase)}
    with patch.object(guard, "supervise", lambda *a, **k: 0), patch.object(new_runtime, "identity", lambda *a: copybound):
        try:
            new_entry.supervise(trial_c, mode)
        except RuntimeError as exc:
            require("phase terminal event absent" in str(exc), "missing phase failed at an unrelated check: " + str(exc))
        else:
            raise AssertionError("real supervise accepted a missing phase terminal")
    require(not (out / "CLOSED.json").exists(), "missing phase terminal produced CLOSED")


def run(root):
    require(os.name == "posix", "integration requires real Linux parent/process identities")
    c = fixture(root)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1",
           "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
           "MKL_THREADING_LAYER": "GNU", "PYTHONPATH": str(ROOT / "src")}
    phases = []
    for mode in PHASES:
        out = Path(c["job_root"]) / mode
        out.mkdir()
        write(Path(c["job_root"]) / (mode + "-consumed.json"),
              {"mode": mode, "contract_sha256": c["contract_sha256"], "TEST_ONLY": MARKER})
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "--root", str(root),
                   "--child", "supervise", "--mode", mode]
        with (root / (mode + ".log")).open("xb") as log:
            completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        require(completed.returncode == 0, "actual entry phase failed: " + mode + "; inspect " + str(root / (mode + ".log")))
        closed = read(out / "CLOSED.json")
        require(closed["status"] == "PASS" and closed["result"] == reference(out / "RESULT.json"),
                "production supervise did not close exact RESULT")
        result = read(out / "RESULT.json")
        phases.append({"phase": mode, "closed": reference(out / "CLOSED.json"),
                       "forward_calls": result.get("model_batch_forwards", 0),
                       "budget_calls": result.get("new_score_budget_calls", 0)})
    checks = protocol_faults(root, c)
    comparison = read(Path(c["job_root"]) / "evaluation/COMPARISON.json")
    require(comparison["new_score_budget_calls"] == 6 and not comparison["benefit_and_protection_passed"],
            "synthetic scientific FAIL must still finish fixed comparison")
    result = {"status": "PASS", "TEST_ONLY": MARKER, "phases": phases,
              "negative_checks": checks, "negative_count": len(checks),
              "actual_entry": "new_entry.main -> supervise -> independent worker -> infer/evaluate",
              "production_event_emitter_Writer_Reader_archive_unpatched": True,
              "scientific_failure_completed": True, "CUDA_initialized": False,
              "real_detector_constructions": 0, "detector_training_updates": 0,
              "substituted_backends": ["frozen material loader fixture", "CPU hardware admission",
                "CPU Tensor.to device adapter", "tiny fixed tensor module", "synthetic checkpoint/RNG capture",
                "synthetic image pixels", "constant metric backend", "CPU process supervisor adapter"],
              "not_evidence_for_actual_resource_admission_or_AP": True}
    publish(root / "RESULT.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", "--output", dest="root", required=True)
    parser.add_argument("--child", choices=("supervise", "worker"))
    parser.add_argument("--mode", choices=PHASES)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if args.child:
        return run_child(root, args.child, args.mode)
    result = run(root)
    print(json.dumps({k: v for k, v in result.items() if k != "phases"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
