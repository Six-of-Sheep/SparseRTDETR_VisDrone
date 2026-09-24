#!/usr/bin/env python3
"""Start a v2b train_core run from the pretrained initialization (epoch 0).

Companion of ``tools/v2b_fork_continue.py`` for runs that have no parent
checkpoint (A3: a new input size). Every training hyperparameter, the
pretrained backbone and the train_core data are copied from a reference
schema-2 checkpoint binding (``--like``); only ``input_size`` (and optionally
``seed``) are replaced. The run binds this tool's own clean worktree: model,
data, optimizer, checkpoint and hardware admission all import from it, so no
orchestration bridge is involved.

The binding is built on CPU exactly as the replication worker does
(``build_train_core_run_binding`` with a requested ``cuda:0``); after native
admission the components are rebuilt on the admitted runtime and must have
the same initialized identity before a session is created.

Subcommands (run with the P3 environment interpreter):

  check  CPU only. Builds the configuration, CPU components, loader and run
         binding; records geometry and the first logical batch of epoch 1.
         Optionally validates a policy bundle against the binding without
         starting the monitor (``--policy-from``).
  train  GPU. Trains epochs 1..``--target-epoch`` under the native hardware
         monitor, one checkpoint per epoch. ``--max-windows N`` stops after N
         windows without writing a checkpoint (GPU smoke).

The launcher re-executes itself with the clean environment of the 30-epoch
campaigns and PYTHONPATH pointing at this worktree.
"""
from __future__ import annotations

import argparse
from dataclasses import fields, replace
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

CHILD_MARKER = "P3_FRESH_CHILD"
BASE_ENVIRONMENT = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "MKL_NUM_THREADS": "2",
    "MKL_THREADING_LAYER": "GNU",
    "OMP_NUM_THREADS": "2",
    "PYTHONNOUSERSITE": "1",
}
MONITOR_SCOPE = "train_core_30epoch"
MONITOR_SCOPE_LIMIT_SECONDS = 43200
REPO = Path(__file__).resolve().parents[1]
TOOL_RELATIVE = "tools/v2b_fresh_run.py"


def canonical(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                       allow_nan=False) + "\n").encode()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def publish(path: Path, value) -> None:
    with path.open("xb") as stream:
        stream.write(canonical(value))
        stream.flush()
        os.fsync(stream.fileno())


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def policy_bundle(path: Path) -> dict:
    document = json.loads(path.read_text())
    return document["policy_bundle"] if "policy_bundle" in document else document


def launch(args: argparse.Namespace) -> None:
    """Re-exec this file with a clean environment bound to this worktree."""
    env = {"HOME": os.environ.get("HOME", "/tmp"), "PATH": "/usr/bin:/bin",
           "PYTHONPATH": str(REPO / "src"), CHILD_MARKER: "1", **BASE_ENVIRONMENT}
    if args.command == "check":
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        env["CUDA_VISIBLE_DEVICES"] = policy_bundle(Path(args.policy_from))["gpu_uuid"]
    print(f"[fresh] bound worktree: {REPO}", flush=True)
    os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], env)


def reference_binding(path: Path) -> dict:
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if payload.get("schema_version") != 2:
        raise SystemExit("--like must be a schema-2 checkpoint")
    return payload["binding"]


def make_config(like: dict, input_size: int, seed: int | None):
    from sparse_rtdetr.baseline.training_v2b import V2BConfig
    source = like["config"]
    config = V2BConfig(**{field.name: source[field.name] for field in fields(V2BConfig)})
    return replace(config, input_size=input_size, seed=config.seed if seed is None else seed)


def build_loader(like: dict, config, num_workers: int, prefetch_factor: int):
    from sparse_rtdetr.baseline.training_v2b_data import TrainCoreDataConfig, build_train_core_loader
    data = like["data"]["loader"]
    reference = data["config"]
    loader = build_train_core_loader(
        TrainCoreDataConfig(
            seed=config.seed, input_size=config.input_size,
            logical_batch_size=config.logical_batch_size, num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            augmentation_stop_internal_epoch=reference["augmentation_stop_internal_epoch"],
        ), annotation_file=data["annotation"]["path"], annotation_sha256=data["annotation"]["sha256"],
        manifest_file=data["manifest"]["path"], manifest_sha256=data["manifest"]["sha256"],
        image_root=data["image_root"], repo_root=REPO,
    )
    if len(loader.dataset) != 4869 or loader.batches_per_epoch != 304:
        raise SystemExit("train_core loader differs from the frozen full epoch (4869 images, 304 windows)")
    return loader


def prepare(args: argparse.Namespace, gpu_uuid: str) -> dict:
    """CPU components, loader and run binding (the replication worker's order)."""
    from sparse_rtdetr.baseline.training_v2b import build_v2b_components
    from sparse_rtdetr.baseline.training_v2b_runtime import build_train_core_run_binding
    if git(REPO, "status", "--porcelain", "--untracked-files=no"):
        raise SystemExit(f"worktree has tracked modifications: {REPO}")
    like_path = Path(args.like).resolve(strict=True)
    like = reference_binding(like_path)
    config = make_config(like, args.input_size, args.seed)
    pretrained = like["config"]["pretrained"]
    arguments = {"repo_root": REPO, "pretrained_path": Path(pretrained["path"]),
                 "pretrained_sha256": pretrained["sha256"]}
    loader = build_loader(like, config, args.num_workers, args.prefetch_factor)
    cpu = build_v2b_components(config, **arguments)
    run_id = args.run_id or (f"v2bfresh-r{config.input_size}-s{config.seed}-"
                             + time.strftime("%Y%m%dt%H%M%Sz", time.gmtime()))
    binding = build_train_core_run_binding(
        cpu, loader, run_id=run_id, repo_root=REPO,
        additional_code_paths={TOOL_RELATIVE: REPO / TOOL_RELATIVE},
        requested_device="cuda:0", cuda_gpu_uuid=gpu_uuid)
    changed = {key: [like["config"].get(key), binding["config"].get(key)]
               for key in sorted(set(like["config"]) | set(binding["config"]))
               if key not in ("geometry", "resolved_model_optimizer_config", "sampling", "pretrained")
               and like["config"].get(key) != binding["config"].get(key)}
    return {
        "like": {"path": str(like_path), "sha256": sha_file(like_path), "run_id": like["run_id"]},
        "config": config, "arguments": arguments, "loader": loader, "cpu": cpu, "binding": binding,
        "identity": {
            "run_id": run_id, "binding_sha256": binding["binding_sha256"],
            "worktree": {"path": str(REPO), "head": git(REPO, "rev-parse", "HEAD")},
            "tool_sha256": sha_file(REPO / TOOL_RELATIVE), "interpreter": sys.executable,
            "environment": {key: os.environ.get(key) for key in (*BASE_ENVIRONMENT, "CUDA_VISIBLE_DEVICES", "PYTHONPATH")},
            "argv": sys.argv[1:],
            "config_changes_vs_like": changed,
            "geometry": binding["config"]["geometry"],
        },
    }


def run_check(args: argparse.Namespace) -> int:
    gpu_uuid = policy_bundle(Path(args.policy_from))["gpu_uuid"] if args.policy_from else args.gpu_uuid
    if not gpu_uuid:
        raise SystemExit("check needs --policy-from or --gpu-uuid to bind the requested CUDA device")
    state = prepare(args, gpu_uuid)
    report = {"status": "CHECK_PASS", **state["identity"], "like": state["like"]}
    started = time.time()
    batch = next(iter(state["loader"].preview_batches(1, epoch=1)))
    report["preview"] = {"epoch": 1, "num_workers": args.num_workers,
                         "seconds": round(time.time() - started, 2),
                         "augmented_sha256": batch.augmented_sha256, "receipt_sha256": batch.receipt_sha256}
    if args.policy_from:
        # Admission validation only: builds the monitor object, never starts it.
        from sparse_rtdetr.baseline.training_v2b_admission import MonitoredHardwareSession
        policy = policy_bundle(Path(args.policy_from))
        evidence = Path("/tmp") / f"p3-fresh-check-never-started-{os.getpid()}"
        session = MonitoredHardwareSession(state["binding"], policy, evidence, MONITOR_SCOPE,
                                           MONITOR_SCOPE_LIMIT_SECONDS)
        if session._state != "new" or evidence.exists():
            raise SystemExit("admission check must not start the monitor")
        report["admission"] = {"status": "ADMITTED_NOT_STARTED",
                               "policy_sha256": session.policy["policy_sha256"],
                               "authorization_sha256": policy["authorization_reference"]["sha256"]}
    if args.output:
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=False)
        publish(out / "run-binding.json", state["binding"])
        publish(out / "check.json", report)
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0


def run_train(args: argparse.Namespace) -> int:
    import torch
    from sparse_rtdetr.baseline.training_v2b import build_v2b_components
    from sparse_rtdetr.baseline.training_v2b_admission import MonitoredHardwareSession
    from sparse_rtdetr.baseline.training_v2b_device import prepare_runtime
    from sparse_rtdetr.baseline.training_v2b_runtime import V2BTrainingSession

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    policy = policy_bundle(Path(args.policy_from))
    gpu_uuid = policy["gpu_uuid"]
    state = prepare(args, gpu_uuid)
    config, binding, loader = state["config"], state["binding"], state["loader"]
    deadline = min(args.deadline_seconds, MONITOR_SCOPE_LIMIT_SECONDS)
    start = {**state["identity"], "like": state["like"], "target_epoch": args.target_epoch,
             "num_workers": args.num_workers, "prefetch_factor": args.prefetch_factor,
             "max_windows": args.max_windows, "save_every": args.save_every,
             "policy_from": str(Path(args.policy_from).resolve()), "gpu_uuid": gpu_uuid,
             "monitor_deadline_seconds": deadline, "started_unix": time.time(), "pid": os.getpid()}
    publish(output / "run-binding.json", binding)
    publish(output / "fresh-start.json", start)

    def peak_memory() -> dict:
        return {"peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0))}

    monitor = None
    windows_done = 0
    try:
        monitor = MonitoredHardwareSession(binding, policy, output / "native-hardware", MONITOR_SCOPE, deadline)
        monitor.start()
        runtime = prepare_runtime(device="cuda:0", seed=config.seed, binding=binding,
                                  gpu_probe=monitor.admission(), expected_gpu_uuid=gpu_uuid)
        components = build_v2b_components(config, runtime=runtime, **state["arguments"])
        if components.initialization != state["cpu"].initialization:
            raise RuntimeError("CUDA-built initialization differs from the bound CPU initialization")
        del state["cpu"]
        session = V2BTrainingSession(components, loader, binding)
        engine = components.engine
        if engine.epoch != 0 or engine.optimizer_updates != 0:
            raise RuntimeError("fresh session is not at update zero")
        publish(output / "ready.json", {
            "optimizer_updates": engine.optimizer_updates, "ema_updates": int(components.ema.updates),
            "learning_rates": [float(g["lr"]) for g in components.optimizer.param_groups],
            "runtime": runtime.identity, "pid": os.getpid()})

        for epoch in range(1, args.target_epoch + 1):
            session.begin_epoch(epoch)
            epoch_started = time.time()
            input_wait = 0.0
            with (output / f"epoch-{epoch:03d}-windows.jsonl").open("xb", buffering=0) as ledger:
                ready = time.time()
                for batch in loader:
                    fetched = time.time()
                    monitor.check(stage="before_window")
                    record = session.train_batch(batch, hardware_probe=monitor.admission())
                    torch.cuda.synchronize(0)
                    monitor.check(stage="after_window")
                    window = record["window"]
                    values = [float(window["loss"]), float(window["gradient_norm_before_clip"])]
                    values += [float(v) for row in window["microbatch_losses"] for v in row.values()]
                    if not all(math.isfinite(v) for v in values):
                        raise RuntimeError("nonfinite loss or gradient")
                    input_wait += fetched - ready
                    ledger.write(canonical({
                        "epoch": epoch, "window": batch.logical_batch_index + 1,
                        "optimizer_updates": window["optimizer_updates"], "microsteps": window["microsteps"],
                        "ema_updates": int(components.ema.updates), "loss": window["loss"],
                        "gradient_norm_before_clip": window["gradient_norm_before_clip"],
                        "microbatch_losses": window["microbatch_losses"],
                        "learning_rates": [float(g["lr"]) for g in components.optimizer.param_groups],
                        "input": {key: value for key, value in record["input"].items() if key != "samples"},
                        "input_wait_seconds": round(fetched - ready, 4),
                        "window_seconds": round(time.time() - fetched, 4), **peak_memory(),
                        "time": time.time()}))
                    windows_done += 1
                    if args.max_windows and windows_done >= args.max_windows:
                        break
                    ready = time.time()
            if args.max_windows and windows_done >= args.max_windows:
                reference = monitor.finish()
                publish(output / "fresh-result.json", {
                    "status": "SMOKE_PASS", "windows": windows_done, "epoch": epoch,
                    "checkpoint_written": False, "monitor_reference": reference, **peak_memory(),
                    "elapsed_seconds": round(time.time() - start["started_unix"], 1)})
                return 0
            completed = session.finish_epoch()
            saved = None
            if epoch == args.target_epoch or epoch % args.save_every == 0:
                saved = session.save(output / f"checkpoint-epoch-{epoch:03d}.pt")
            publish(output / f"epoch-{epoch:03d}-complete.json", {
                "epoch": epoch, "record": completed, "checkpoint": saved,
                "ema_updates": int(components.ema.updates),
                "learning_rates": [float(g["lr"]) for g in components.optimizer.param_groups],
                "epoch_seconds": round(time.time() - epoch_started, 1),
                "input_wait_seconds": round(input_wait, 1), **peak_memory()})
            print(f"[fresh] epoch {epoch} done in {time.time() - epoch_started:.0f}s "
                  f"(input wait {input_wait:.0f}s)", flush=True)
        reference = monitor.finish()
        publish(output / "fresh-result.json", {
            "status": "TRAINING_COMPLETE", "last_epoch": args.target_epoch,
            "monitor_reference": reference, **peak_memory(),
            "elapsed_seconds": round(time.time() - start["started_unix"], 1)})
        return 0
    except BaseException as exc:
        publish(output / "fresh-failure.json", {"type": type(exc).__name__, "message": str(exc),
                                                "windows_done": windows_done, "time": time.time()})
        if getattr(monitor, "_state", None) == "running":
            monitor.abort(f"fresh failure: {type(exc).__name__}: {exc}")
        raise


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("check", "train"):
        sub = commands.add_parser(name)
        sub.add_argument("--like", required=True,
                         help="schema-2 checkpoint whose config, pretrained and data are copied")
        sub.add_argument("--input-size", type=int, required=True)
        sub.add_argument("--seed", type=int, default=None, help="default: the --like seed")
        sub.add_argument("--run-id", default=None)
        sub.add_argument("--num-workers", type=int, default=2)
        sub.add_argument("--prefetch-factor", type=int, default=2)
    check = commands.choices["check"]
    check.add_argument("--policy-from", help="also validate admission (monitor built, never started)")
    check.add_argument("--gpu-uuid", help="requested CUDA device when no policy is given")
    check.add_argument("--output", help="new directory for run-binding.json and check.json")
    train = commands.choices["train"]
    train.add_argument("--target-epoch", type=int, required=True)
    train.add_argument("--output", required=True, help="new directory; must not exist")
    train.add_argument("--policy-from", required=True, help="JSON with a reviewed policy_bundle")
    train.add_argument("--max-windows", type=int, default=0, help="GPU smoke: stop after N windows")
    train.add_argument("--save-every", type=int, default=1)
    train.add_argument("--deadline-seconds", type=int, default=MONITOR_SCOPE_LIMIT_SECONDS)
    return root


def main() -> int:
    args = parser().parse_args()
    if args.num_workers < 0:
        raise SystemExit("--num-workers must be >= 0")
    if os.environ.get(CHILD_MARKER) != "1":
        launch(args)
    return run_check(args) if args.command == "check" else run_train(args)


if __name__ == "__main__":
    raise SystemExit(main())
