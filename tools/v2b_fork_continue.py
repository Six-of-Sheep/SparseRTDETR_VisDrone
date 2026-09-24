#!/usr/bin/env python3
"""Fork or resume a v2b train_core run from any schema-2 epoch checkpoint.

The checkpoint's own run binding decides which code runs: every bound source
path points into the worktree that produced the run, so the worker imports
``sparse_rtdetr`` from exactly that worktree and the binding verifies without
any rebase or bridge. This file itself is not part of any run binding.

Exception, ``train`` only: every bound worktree predates the 2026-09-17 reboot
and its admission accepts only the pre-reboot hardware authority. As in the
R35 epoch-60 worker, the current-boot admission (evidence, hardware, admission
and device modules) loads from a pinned orchestration checkout under a private
alias, and only its ``require_native_hardware_admission`` and the admitted
device runtime cross into the bound package (``orchestration_imports``,
``reseal_runtime``). Model, data, optimizer and checkpoint code stay bound.

Two optional runtime changes are supported and recorded; neither changes the
bound model, data order, augmentation or checkpoint semantics:

* ``--lr-scale``: multiply every optimizer group's learning rate once, right
  after restore. Forking the constant-LR run at epoch E with scale 0.1 is
  bit-for-bit the same trajectory as a from-scratch run whose schedule decays
  by 0.1 at the end of epoch E, because both share every update up to E.
  A checkpoint written by a fork stores the scaled LR, so resuming a fork
  later uses ``--lr-scale 1``.
* ``--num-workers``: DataLoader worker processes. Sampling and augmentation
  seeds are position-derived, so worker count does not change any batch
  (verified with ``check`` against historical receipts).

Subcommands (run with the P3 environment interpreter):

  check  CPU only. Verifies checkpoint SHA, run binding and bound source files,
         bound worktree cleanliness, checkpoint contents, and previews the
         first logical batch of the next epoch; optionally compares it with a
         historical receipt directory or window ledger.
  train  GPU. Restores, optionally rescales the LR, trains whole epochs up to
         ``--target-epoch`` under the native hardware monitor, and writes one
         checkpoint per epoch. ``--max-windows N`` stops after N windows
         without writing a checkpoint (GPU smoke).

The launcher re-executes itself with a clean environment (the one the 30-epoch
campaigns used) and PYTHONPATH pointing at the bound worktree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

CHILD_MARKER = "P3_FORK_CHILD"
# Environment of the 30-epoch campaign workers that ran ten cells with two
# loader workers (historical_startup_environment in worker results).
BASE_ENVIRONMENT = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "MKL_NUM_THREADS": "2",
    "MKL_THREADING_LAYER": "GNU",
    "OMP_NUM_THREADS": "2",
    "PYTHONNOUSERSITE": "1",
}
MONITOR_SCOPE = "train_core_30epoch"
MONITOR_SCOPE_LIMIT_SECONDS = 43200
# Current-boot hardware admission of the R35 epoch-60 worker: the commit and
# file SHA-256 values equal ``orchestration_sources`` in its cell contracts.
ORCHESTRATION_COMMIT = "97a78f70c2fd2112238976258075f725ed10cd81"
ORCHESTRATION_MODULES = {
    "training_v2b_evidence": "f54627021add76c4896def8f7a464fd26fdb4307a7c4d5339d28bc948f4eabbd",
    "training_v2b_hardware": "a25c7c5668b431f9f644f6e31dec6402da651f2ca52941368e5beebbc8d7f98c",
    "training_v2b_admission": "904a0f29ea8bf47aedeb9e9b07d6b7790cc1d872af714cfa53ee60d7e971a4e8",
    "training_v2b_device": "514308f70a19314a05596a82f71eddea9652de8bde4316a1fae7d8bab9f42bf9",
}
ORCHESTRATION_ALIAS = "_p3_fork_orchestration_runtime"


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


def read_binding(checkpoint: Path) -> dict:
    import torch
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if payload.get("schema_version") != 2:
        raise SystemExit("only schema-2 checkpoints (with loader cursor) can be continued")
    return payload["binding"]


def bound_repo(binding: dict) -> Path:
    reference = binding["code"]["src/sparse_rtdetr/baseline/training_v2b_data.py"]["path"]
    return Path(reference).parents[3]


def launch(args: argparse.Namespace) -> None:
    """Re-exec this file with a clean environment bound to the checkpoint's code."""
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    repo = bound_repo(read_binding(checkpoint))
    env = {"HOME": os.environ.get("HOME", "/tmp"), "PATH": "/usr/bin:/bin",
           "PYTHONPATH": str(repo / "src"), CHILD_MARKER: "1", **BASE_ENVIRONMENT}
    if args.command == "check":
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        env["CUDA_VISIBLE_DEVICES"] = policy_bundle(Path(args.policy_from))["gpu_uuid"]
    print(f"[fork] bound worktree: {repo}", flush=True)
    os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], env)


def policy_bundle(path: Path) -> dict:
    document = json.loads(path.read_text())
    return document["policy_bundle"] if "policy_bundle" in document else document


def orchestration_imports(root: Path, repo: Path) -> tuple[dict, dict]:
    """Load the current-boot admission beside the bound training package.

    Mirrors ``_source_imports`` and ``_bridge_hardware_admission`` of
    tools/run_training_v2b_epoch60_continuation_worker.py at 97a78f7: the
    pinned modules load under a private alias, ``sparse_rtdetr`` stays the
    bound package, and the current ``require_native_hardware_admission`` is
    bound into the bound hardware and runtime modules (the runtime imported
    it by name, so its own global must be replaced as well).
    """
    import importlib
    import importlib.util
    import types
    root = root.resolve(strict=True)
    if git(root, "rev-parse", "HEAD") != ORCHESTRATION_COMMIT:
        raise SystemExit(f"orchestration checkout is not at {ORCHESTRATION_COMMIT}: {root}")
    if git(root, "status", "--porcelain", "--untracked-files=no"):
        raise SystemExit(f"orchestration checkout has tracked modifications: {root}")
    baseline = root / "src/sparse_rtdetr/baseline"
    sources = {}
    for name, expected in ORCHESTRATION_MODULES.items():
        path = baseline / f"{name}.py"
        if sha_file(path) != expected:
            raise SystemExit(f"orchestration module identity mismatch: {name}")
        sources[name] = {"path": str(path), "sha256": expected}
    bound_baseline = (repo / "src/sparse_rtdetr/baseline").resolve(strict=True)
    if sha_file(bound_baseline / "training_v2b_device.py") != ORCHESTRATION_MODULES["training_v2b_device"]:
        raise SystemExit("bound and orchestration device sources differ")
    if ORCHESTRATION_ALIAS in sys.modules:
        raise SystemExit("orchestration alias already loaded")
    sys.dont_write_bytecode = True
    package = types.ModuleType(ORCHESTRATION_ALIAS)
    package.__package__ = ORCHESTRATION_ALIAS
    package.__path__ = [str(baseline)]
    sys.modules[ORCHESTRATION_ALIAS] = package
    current = {}
    for name in ORCHESTRATION_MODULES:
        spec = importlib.util.spec_from_file_location(f"{ORCHESTRATION_ALIAS}.{name}", baseline / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        current[name] = module
    bound = {name: importlib.import_module(f"sparse_rtdetr.baseline.{name}") for name in (
        "training_v2b", "training_v2b_hardware", "training_v2b_device", "training_v2b_runtime")}
    for name, module in bound.items():
        if Path(module.__file__).resolve() != bound_baseline / f"{name}.py":
            raise SystemExit(f"{name} did not load from the bound worktree")
    hardware = current["training_v2b_hardware"]
    validator = hardware.require_native_hardware_admission
    if getattr(validator, "__module__", None) != hardware.__name__:
        raise SystemExit("current hardware validator owner drift")
    bound["training_v2b_hardware"].require_native_hardware_admission = validator
    bound["training_v2b_runtime"].require_native_hardware_admission = validator
    train_batch = bound["training_v2b_runtime"].V2BTrainingSession.train_batch
    if train_batch.__globals__.get("require_native_hardware_admission") is not validator:
        raise SystemExit("bound session validator bridge failed")
    return current, {
        "commit": ORCHESTRATION_COMMIT, "root": str(root), "modules": sources,
        "bridged_validator": f"{validator.__module__}.{validator.__name__}",
        "bound_package": str(bound_baseline.parents[1]),
    }


def reseal_runtime(runtime, producer, consumer) -> tuple[object, dict]:
    """Re-seal the admitted device runtime for the bound device module.

    Mirrors ``_bridge_prepared_runtime`` of the R35 worker: the byte-identical
    device source executed under two module names has two private seals, so
    validate with the producer, rebuild the frozen dataclass with the bound
    module's seal and validate again.
    """
    import copy
    producer_sha, consumer_sha = sha_file(Path(producer.__file__)), sha_file(Path(consumer.__file__))
    if producer.__file__ == consumer.__file__ or producer_sha != consumer_sha:
        raise RuntimeError("runtime bridge needs byte-identical device sources in two checkouts")
    if type(runtime) is not producer.PreparedRuntime:
        raise RuntimeError("runtime bridge producer type differs")
    identity = producer.validate_prepared_runtime(runtime)
    if runtime._identity_sha256 != producer._digest(identity):
        raise RuntimeError("runtime bridge producer seal digest differs")
    bridged = consumer.PreparedRuntime(
        device=runtime.device, seed=runtime.seed,
        admitted_binding_sha256=runtime.admitted_binding_sha256,
        _identity=copy.deepcopy(identity), _identity_sha256=consumer._digest(identity),
        _seal=consumer._SEAL)
    if consumer.validate_prepared_runtime(bridged) != identity:
        raise RuntimeError("runtime bridge identity differs")
    return bridged, {"producer": producer.__name__, "consumer": consumer.__name__,
                     "device_sha256": consumer_sha,
                     "runtime_identity_sha256": consumer._digest(identity)}


def build_loader(binding: dict, repo: Path, num_workers: int, prefetch_factor: int):
    from sparse_rtdetr.baseline.training_v2b_data import TrainCoreDataConfig, build_train_core_loader
    data = binding["data"]["loader"]
    cfg = data["config"]
    loader = build_train_core_loader(
        TrainCoreDataConfig(
            seed=cfg["seed"], input_size=cfg["input_size"],
            logical_batch_size=cfg["logical_batch_size"], num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            augmentation_stop_internal_epoch=cfg["augmentation_stop_internal_epoch"],
        ), annotation_file=data["annotation"]["path"], annotation_sha256=data["annotation"]["sha256"],
        manifest_file=data["manifest"]["path"], manifest_sha256=data["manifest"]["sha256"],
        image_root=data["image_root"], repo_root=repo,
    )
    if loader.binding != data:
        raise SystemExit("live loader binding differs from the checkpoint's bound train_core input")
    return loader


def make_config(binding: dict):
    from dataclasses import fields
    from sparse_rtdetr.baseline.training_v2b import V2BConfig
    source = binding["config"]
    return V2BConfig(**{field.name: source[field.name] for field in fields(V2BConfig)})


def identity(args: argparse.Namespace, checkpoint: Path, checkpoint_sha: str, binding: dict,
             repo: Path) -> dict:
    """Common preflight: bound worktree clean and bound files unchanged."""
    from sparse_rtdetr.baseline.training_v2b_evidence import validate_run_binding
    if git(repo, "status", "--porcelain", "--untracked-files=no"):
        raise SystemExit(f"bound worktree has tracked modifications: {repo}")
    validate_run_binding(binding, verify_files=True)
    tool = Path(__file__).resolve()
    tool_repo = tool.parents[1]
    return {
        "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha},
        "binding_sha256": binding["binding_sha256"], "run_id": binding["run_id"],
        "bound_worktree": {"path": str(repo), "head": git(repo, "rev-parse", "HEAD")},
        "tool": {"path": str(tool), "sha256": sha_file(tool),
                 "repo_head": git(tool_repo, "rev-parse", "HEAD"),
                 "repo_dirty": bool(git(tool_repo, "status", "--porcelain", "--untracked-files=no"))},
        "interpreter": sys.executable,
        "environment": {key: os.environ.get(key) for key in (*BASE_ENVIRONMENT, "CUDA_VISIBLE_DEVICES", "PYTHONPATH")},
        "argv": sys.argv[1:],
    }


def reference_input(reference: Path, epoch: int) -> dict | None:
    """First-window input receipt from a campaign receipt dir or a window ledger dir."""
    receipt = reference / f"epoch-{epoch:03d}" / "window-0001.json"
    if receipt.is_file():
        return json.loads(receipt.read_text())["input"]
    ledger = reference / f"epoch-{epoch:03d}-windows.jsonl"
    if ledger.is_file():
        with ledger.open() as stream:
            return json.loads(stream.readline())["input"]
    return None


def run_check(args: argparse.Namespace) -> int:
    import torch
    from sparse_rtdetr.baseline.training_v2b_checkpoint import inspect_checkpoint
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    checkpoint_sha = sha_file(checkpoint)
    binding = read_binding(checkpoint)
    repo = bound_repo(binding)
    report = identity(args, checkpoint, checkpoint_sha, binding, repo)
    inspected = inspect_checkpoint(checkpoint, expected_sha256=checkpoint_sha, expected_binding=binding)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    engine = payload["engine"]
    if engine["epoch_active"]:
        raise SystemExit("checkpoint is not at an epoch boundary")
    next_epoch = engine["epoch"] + 1
    lrs = [float(group["lr"]) for group in payload["optimizer"]["param_groups"]]
    loader = build_loader(binding, repo, args.num_workers, args.prefetch_factor)
    started = time.time()
    batch = next(iter(loader.preview_batches(1, epoch=next_epoch)))
    report.update({
        "status": "CHECK_PASS",
        "inspect_keys": sorted(inspected) if isinstance(inspected, dict) else None,
        "engine": {key: engine[key] for key in ("epoch", "optimizer_updates", "microsteps")},
        "saved_learning_rates": lrs,
        "planned_learning_rates": [lr * args.lr_scale for lr in lrs],
        "config": {key: binding["config"].get(key) for key in (
            "seed", "input_size", "physical_batch_size", "accumulation_steps", "learning_rate")},
        "preview": {"epoch": next_epoch, "num_workers": args.num_workers,
                    "seconds": round(time.time() - started, 2),
                    "augmented_sha256": batch.augmented_sha256, "receipt_sha256": batch.receipt_sha256},
    })
    if args.reference:
        expected = reference_input(Path(args.reference), next_epoch)
        if expected is None:
            report["preview"]["reference"] = "NOT_FOUND"
        else:
            match = (expected["augmented_sha256"] == batch.augmented_sha256
                     and expected["receipt_sha256"] == batch.receipt_sha256)
            report["preview"]["reference"] = "MATCH" if match else "MISMATCH"
            if not match:
                report["status"] = "CHECK_FAIL"
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0 if report["status"] == "CHECK_PASS" else 1


def run_train(args: argparse.Namespace) -> int:
    import torch

    checkpoint = Path(args.checkpoint).resolve(strict=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    checkpoint_sha = sha_file(checkpoint)
    binding = read_binding(checkpoint)
    repo = bound_repo(binding)
    start = identity(args, checkpoint, checkpoint_sha, binding, repo)
    current, orchestration = orchestration_imports(Path(args.orchestration_root), repo)
    from sparse_rtdetr.baseline import training_v2b_device as bound_device
    from sparse_rtdetr.baseline.training_v2b import build_v2b_components
    from sparse_rtdetr.baseline.training_v2b_runtime import V2BTrainingSession
    MonitoredHardwareSession = current["training_v2b_admission"].MonitoredHardwareSession
    prepare_runtime = current["training_v2b_device"].prepare_runtime

    policy = policy_bundle(Path(args.policy_from))
    gpu_uuid = policy["gpu_uuid"]
    config = make_config(binding)
    deadline = min(args.deadline_seconds, MONITOR_SCOPE_LIMIT_SECONDS)
    start.update(target_epoch=args.target_epoch, lr_scale=args.lr_scale,
                 num_workers=args.num_workers, prefetch_factor=args.prefetch_factor,
                 max_windows=args.max_windows, save_every=args.save_every,
                 policy_from=str(Path(args.policy_from).resolve()), gpu_uuid=gpu_uuid,
                 orchestration=orchestration,
                 monitor_deadline_seconds=deadline, started_unix=time.time(), pid=os.getpid())
    publish(output / "fork-start.json", start)

    def peak_memory() -> dict:
        return {"peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0))}

    monitor = None
    windows_done = 0
    try:
        monitor = MonitoredHardwareSession(binding, policy, output / "native-hardware", MONITOR_SCOPE, deadline)
        monitor.start()
        admitted = prepare_runtime(device="cuda:0", seed=config.seed, binding=binding,
                                   gpu_probe=monitor.admission(), expected_gpu_uuid=gpu_uuid)
        runtime, runtime_bridge = reseal_runtime(admitted, current["training_v2b_device"], bound_device)
        pretrained = binding["config"]["pretrained"]
        components = build_v2b_components(config, repo_root=repo, runtime=runtime,
                                          pretrained_path=Path(pretrained["path"]),
                                          pretrained_sha256=pretrained["sha256"])
        loader = build_loader(binding, repo, args.num_workers, args.prefetch_factor)
        session = V2BTrainingSession(components, loader, binding)
        monitor.check(stage="before_restore")
        session.restore(checkpoint, checkpoint_sha)
        torch.cuda.synchronize(0)
        monitor.check(stage="after_restore")
        engine = components.engine
        first_epoch = engine.epoch + 1
        if not first_epoch <= args.target_epoch:
            raise ValueError(f"target epoch {args.target_epoch} is not after restored epoch {engine.epoch}")
        # MultiStepLR counts epochs completed after warmup, not engine epochs.
        scheduler_epoch = components.scheduler.last_epoch
        remaining = args.target_epoch - engine.epoch
        milestones = getattr(components.scheduler, "milestones", {})
        if any(scheduler_epoch < int(m) <= scheduler_epoch + remaining for m in milestones):
            raise ValueError("a bound scheduler milestone falls inside the fork range")
        if args.lr_scale != 1.0:
            if components.warmup is not None and not components.warmup.finished():
                raise ValueError("refusing to rescale the LR before the vendor warmup has finished")
            for group in components.optimizer.param_groups:
                group["lr"] = float(group["lr"]) * args.lr_scale
        learning_rates = [float(group["lr"]) for group in components.optimizer.param_groups]
        publish(output / "ready.json", {
            "restored_epoch": engine.epoch, "optimizer_updates": engine.optimizer_updates,
            "microsteps": engine.microsteps, "ema_updates": int(components.ema.updates),
            "learning_rates": learning_rates, "first_epoch": first_epoch,
            "runtime_bridge": runtime_bridge, "pid": os.getpid()})

        for epoch in range(first_epoch, args.target_epoch + 1):
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
                publish(output / "fork-result.json", {
                    "status": "SMOKE_PASS", "windows": windows_done, "epoch": epoch,
                    "checkpoint_written": False, "monitor_reference": reference, **peak_memory(),
                    "elapsed_seconds": round(time.time() - start["started_unix"], 1)})
                return 0
            completed = session.finish_epoch()
            saved = None
            if epoch == args.target_epoch or (epoch - first_epoch + 1) % args.save_every == 0:
                saved = session.save(output / f"checkpoint-epoch-{epoch:03d}.pt")
            publish(output / f"epoch-{epoch:03d}-complete.json", {
                "epoch": epoch, "record": completed, "checkpoint": saved,
                "ema_updates": int(components.ema.updates), "learning_rates": learning_rates,
                "epoch_seconds": round(time.time() - epoch_started, 1),
                "input_wait_seconds": round(input_wait, 1), **peak_memory()})
            print(f"[fork] epoch {epoch} done in {time.time() - epoch_started:.0f}s "
                  f"(input wait {input_wait:.0f}s)", flush=True)
        reference = monitor.finish()
        publish(output / "fork-result.json", {
            "status": "TRAINING_COMPLETE", "last_epoch": args.target_epoch,
            "monitor_reference": reference, **peak_memory(),
            "elapsed_seconds": round(time.time() - start["started_unix"], 1)})
        return 0
    except BaseException as exc:
        publish(output / "fork-failure.json", {"type": type(exc).__name__, "message": str(exc),
                                               "windows_done": windows_done, "time": time.time()})
        if getattr(monitor, "_state", None) == "running":
            monitor.abort(f"fork failure: {type(exc).__name__}: {exc}")
        raise


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = root.add_subparsers(dest="command", required=True)
    for name in ("check", "train"):
        sub = commands.add_parser(name)
        sub.add_argument("--checkpoint", required=True)
        sub.add_argument("--num-workers", type=int, default=2)
        sub.add_argument("--prefetch-factor", type=int, default=2)
        sub.add_argument("--lr-scale", type=float, default=1.0)
    check = commands.choices["check"]
    check.add_argument("--reference", help="campaign receipts/ dir or a fork/formal evidence dir")
    train = commands.choices["train"]
    train.add_argument("--target-epoch", type=int, required=True)
    train.add_argument("--output", required=True, help="new directory; must not exist")
    train.add_argument("--policy-from", required=True,
                       help="JSON with a reviewed policy_bundle (e.g. an R35 cell contract)")
    train.add_argument("--orchestration-root", required=True,
                       help=f"clean checkout at {ORCHESTRATION_COMMIT[:7]} (current-boot hardware admission)")
    train.add_argument("--max-windows", type=int, default=0, help="GPU smoke: stop after N windows")
    train.add_argument("--save-every", type=int, default=1)
    train.add_argument("--deadline-seconds", type=int, default=MONITOR_SCOPE_LIMIT_SECONDS)
    return root


def main() -> int:
    args = parser().parse_args()
    if not 0 < args.lr_scale <= 1.0:
        raise SystemExit("--lr-scale must be in (0, 1]")
    if args.num_workers < 0:
        raise SystemExit("--num-workers must be >= 0")
    if os.environ.get(CHILD_MARKER) != "1":
        launch(args)
    return run_check(args) if args.command == "check" else run_train(args)


if __name__ == "__main__":
    raise SystemExit(main())
