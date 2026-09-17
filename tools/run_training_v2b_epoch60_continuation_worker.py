"""Execute one frozen epoch-30 to epoch-60 continuation stage.

The contract is authenticated before any project, torch, data, CUDA, model, or
evaluator import.  The historical source checkout is then placed first on
``sys.path`` so checkpoint restore uses the exact source package that created
the source run binding.  This worker never launches another worker and never
retries a failed stage.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import sys
import time
import types


_DIGEST = re.compile(r"[0-9a-f]{64}")


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate contract JSON key: " + key)
        value[key] = item
    return value


def _invalid_constant(value):
    raise ValueError("nonfinite JSON constant is forbidden: " + value)


def _same(left, right, label):
    if type(left) is not type(right) or left != right:
        raise RuntimeError(label + " drift")


def _file_reference(path):
    source = Path(path).resolve()
    fd = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError("authority must be a single-link regular file")
        digest = hashlib.sha256()
        total = 0
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            total += len(block)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError("authority changed while it was read")
    if total != before.st_size:
        raise ValueError("authority short read")
    return {"path": str(source), "sha256": digest.hexdigest(), "size_bytes": total,
            "sha256_scope": "complete_file_bytes"}


def _load_contract(path_text, expected_sha256):
    if type(expected_sha256) is not str or _DIGEST.fullmatch(expected_sha256) is None:
        raise ValueError("contract SHA-256 must be the complete lowercase digest")
    path = Path(path_text)
    if not path.is_absolute() or path != path.resolve():
        raise ValueError("contract must be an explicit canonical absolute path")
    if any(part.casefold() == "test" or part.casefold().startswith("confirmatory") for part in path.parts):
        raise ValueError("contract path crosses a forbidden data role")
    reference = _file_reference(path)
    if reference["sha256"] != expected_sha256:
        raise ValueError("contract complete-file SHA-256 mismatch before imports")
    raw = path.read_bytes()
    if len(raw) != reference["size_bytes"] or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("contract identity changed between authenticated reads")
    contract = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    if type(contract) is not dict:
        raise ValueError("worker contract must be one JSON object")
    return contract, reference


def _contract_module(contract):
    root = Path(contract.get("repo_root", ""))
    if not root.is_absolute() or root != root.resolve():
        raise ValueError("contract orchestration checkout is not canonical")
    relative = "src/sparse_rtdetr/baseline/training_v2b_epoch60_continuation.py"
    expected = contract.get("orchestration_sources", {}).get(relative)
    path = root / relative
    if type(expected) is not dict or _file_reference(path) != expected:
        raise ValueError("continuation contract validator identity mismatch before import")
    spec = importlib.util.spec_from_file_location("p3_epoch60_contract", path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load continuation contract validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_imports(contract):
    if any(name == "sparse_rtdetr" or name.startswith("sparse_rtdetr.") for name in sys.modules):
        raise RuntimeError("repository package imported before source-checkout selection")
    source_root = Path(contract["source_repo_root"]).resolve()
    package_root = source_root / "src"
    if not package_root.is_dir():
        raise RuntimeError("historical source package root missing")
    sys.path[:] = [str(package_root)] + [entry for entry in sys.path if Path(entry or ".").resolve() != package_root]
    sys.dont_write_bytecode = True
    from sparse_rtdetr.baseline import training_v2b_control as control
    from sparse_rtdetr.baseline.training_v2b import V2BConfig, build_v2b_components
    from sparse_rtdetr.baseline.training_v2b_data import TrainCoreDataConfig, build_train_core_loader
    from sparse_rtdetr.baseline.training_v2b_development import evaluate_development
    from sparse_rtdetr.baseline.training_v2b_evidence import (
        validate_run_binding, write_exclusive_json,
    )
    from sparse_rtdetr.baseline.training_v2b_runtime import V2BTrainingSession
    imported = Path(sys.modules["sparse_rtdetr.baseline.training_v2b"].__file__).resolve()
    if not imported.is_relative_to(source_root) or imported != source_root / "src/sparse_rtdetr/baseline/training_v2b.py":
        raise RuntimeError("training package did not load from the frozen source checkout")

    orchestration_root = Path(contract["repo_root"]).resolve()
    baseline_root = orchestration_root / "src/sparse_rtdetr/baseline"
    sources = contract.get("orchestration_sources", {})
    names = ("training_v2b_evidence", "training_v2b_hardware",
             "training_v2b_admission", "training_v2b_device")
    paths = {}
    for name in names:
        relative = f"src/sparse_rtdetr/baseline/{name}.py"
        path = baseline_root / f"{name}.py"
        expected = sources.get(relative)
        if type(expected) is not dict or _file_reference(path) != expected:
            raise RuntimeError("orchestration runtime identity mismatch: " + name)
        paths[name] = path
    historical_device = _file_reference(
        source_root / "src/sparse_rtdetr/baseline/training_v2b_device.py")
    current_device = sources["src/sparse_rtdetr/baseline/training_v2b_device.py"]
    if ((historical_device["sha256"], historical_device["size_bytes"])
            != (current_device["sha256"], current_device["size_bytes"])):
        raise RuntimeError("orchestration and historical device semantics differ")

    alias = "_p3_epoch60_orchestration_runtime"
    if alias in sys.modules or any(name.startswith(alias + ".") for name in sys.modules):
        raise RuntimeError("orchestration runtime alias already exists")
    package = types.ModuleType(alias)
    package.__package__ = alias
    package.__path__ = [str(baseline_root)]
    sys.modules[alias] = package
    orchestration = {}
    for name in names:
        qualified = alias + "." + name
        spec = importlib.util.spec_from_file_location(qualified, paths[name])
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot load orchestration runtime module: " + name)
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        spec.loader.exec_module(module)
        orchestration[name] = module
    MonitoredHardwareSession = orchestration["training_v2b_admission"].MonitoredHardwareSession
    prepare_runtime = orchestration["training_v2b_device"].prepare_runtime
    return {
        "control": control, "V2BConfig": V2BConfig, "build_v2b_components": build_v2b_components,
        "MonitoredHardwareSession": MonitoredHardwareSession,
        "TrainCoreDataConfig": TrainCoreDataConfig,
        "build_train_core_loader": build_train_core_loader,
        "evaluate_development": evaluate_development,
        "validate_run_binding": validate_run_binding,
        "write_exclusive_json": write_exclusive_json,
        "V2BTrainingSession": V2BTrainingSession, "prepare_runtime": prepare_runtime,
    }


def _read_reference(contract_module, reference, label):
    return contract_module.read_json_reference(reference, label, verify=True)


def _load_prerequisites(contract_module, contract, control):
    smoke = replay = matched = None
    if contract["smoke_reference"] is not None:
        smoke = _read_reference(contract_module, contract["smoke_reference"], "original continuation smoke")
    if contract["replay_reference"] is not None:
        replay = _read_reference(contract_module, contract["replay_reference"], "continuation replay")
    if contract["matched_reference"] is not None:
        matched = _read_reference(contract_module, contract["matched_reference"], "matched 640 continuation")
    replay_index = control._receipt_index(smoke) if contract["stage"] == "smoke_replay" else {}
    matched_index = control._receipt_index(matched) if matched is not None else None
    if contract["stage"] == "smoke_replay" and set(replay_index) != {(31, i) for i in range(4)}:
        raise RuntimeError("original continuation smoke does not cover four epoch31 windows")
    if matched_index is not None:
        expected = ({(epoch, index) for epoch in range(31, 61) for index in range(304)}
                    if contract["stage"] == "formal60" else
                    {(31, 2), (31, 3)} if contract["stage"] == "smoke_replay" else
                    {(31, i) for i in range(4)})
        if set(matched_index) != expected:
            raise RuntimeError("matched 640 continuation result lacks the complete stage ledger")
    return smoke, replay, matched, replay_index, matched_index


def _component_map(components):
    return {name: getattr(components, name) for name in
            ("model", "ema", "postprocessor", "engine", "runtime", "optimizer", "scheduler", "warmup")}


def _execute_stage(contract, session, monitor, output, report, *, control,
                   evaluate_development, write_exclusive_json, replay_index, matched_index):
    components, loader = session.components, session.loader
    checkpoints, receipts = report["checkpoints"], report["receipts"]
    replay_checks, evaluations = report["replay_checks"], report["evaluations"]

    def evaluation_gate():
        import torch
        torch.cuda.synchronize(0)
        return monitor.check(stage="development_batch")

    def begin_epoch(epoch):
        session.begin_epoch(epoch)
        if loader.state_dict()["order_sha256"] != contract["epoch_order_sha256"][str(epoch)]:
            raise RuntimeError("actual future sampler order differs from the frozen continuation plan")

    if contract["stage"] == "smoke":
        begin_epoch(31)
        control._run_active_windows(
            session, monitor, output, limit=4, paired_index={}, replay_index={},
            save_midpoint=True, snapshots=True, checkpoints=checkpoints, receipts=receipts,
            replay_checks=replay_checks, matched_index=matched_index)
        checkpoints["window_4"] = control._write_checkpoint(
            session, output, "checkpoint-window-04", monitor)
    elif contract["stage"] == "smoke_replay":
        control._run_active_windows(
            session, monitor, output, limit=2, paired_index={}, replay_index=replay_index,
            save_midpoint=False, snapshots=True, checkpoints=checkpoints, receipts=receipts,
            replay_checks=replay_checks, matched_index=matched_index)
        checkpoints["window_4"] = control._write_checkpoint(
            session, output, "checkpoint-window-04-replay", monitor)
    elif contract["stage"] == "formal60":
        for epoch in range(31, 61):
            begin_epoch(epoch)
            control._run_active_windows(
                session, monitor, output, limit=304, paired_index={}, replay_index={},
                save_midpoint=False, snapshots=False, checkpoints=checkpoints, receipts=receipts,
                replay_checks=replay_checks, matched_index=matched_index)
            epoch_record = session.finish_epoch()
            checkpoints[f"epoch_{epoch}"] = control._write_checkpoint(
                session, output, f"checkpoint-epoch-{epoch:03d}", monitor)
            write_exclusive_json(output / f"epoch-{epoch:03d}.json", epoch_record)
            if epoch in (45, 60):
                evaluation_dir = output / f"development-epoch-{epoch:03d}"
                evaluation_dir.mkdir()
                before = control.capture_control_state(components, loader, full=True)
                evaluations.append(evaluate_development(
                    _component_map(components), data_binding=contract["development_binding"],
                    output_dir=evaluation_dir, logged_epoch=epoch, hardware_gate=evaluation_gate))
                _same(before, control.capture_control_state(components, loader, full=True),
                      "development evaluation preserved continuation state")
    else:
        raise RuntimeError("unknown continuation stage")


def _final_clocks(stage, components, receipts, replay_checks, evaluations, checkpoints):
    state = components.engine.state_dict()
    clocks = {name: state[name] for name in ("epoch", "epoch_active", "optimizer_updates", "microsteps")}
    if stage == "formal60":
        expected = {"epoch": 60, "epoch_active": False, "optimizer_updates": 18240,
                    "microsteps": 36480}
        if (len(receipts), len(replay_checks), len(evaluations), len(checkpoints)) != (9120, 0, 2, 30):
            raise RuntimeError("formal60 execution ledger coverage drift")
    elif stage == "smoke":
        expected = {"epoch": 31, "epoch_active": True, "optimizer_updates": 9124,
                    "microsteps": 18248}
        if (len(receipts), len(replay_checks), len(evaluations)) != (4, 0, 0):
            raise RuntimeError("continuation smoke ledger coverage drift")
    else:
        expected = {"epoch": 31, "epoch_active": True, "optimizer_updates": 9124,
                    "microsteps": 18248}
        if (len(receipts), len(replay_checks), len(evaluations)) != (2, 2, 0):
            raise RuntimeError("continuation replay ledger coverage drift")
    _same(clocks, expected, "continuation final clocks")
    return clocks


def run_continuation_worker(contract, contract_reference, contract_module):
    freeze = _read_reference(contract_module, contract["freeze_reference"], "continuation freeze")
    checked = contract_module.validate_continuation_worker_contract(
        contract, freeze, verify_files=True)
    _same(_read_reference(contract_module, contract_reference, "worker contract"), checked,
          "invoked continuation contract bytes")
    startup_environment = contract_module.validate_worker_process_environment(checked)
    modules = _source_imports(checked)
    control = modules["control"]
    historical_startup_environment = control._environment(checked)
    _same(
        historical_startup_environment,
        {name: startup_environment[name] for name in (
            "CUDA_VISIBLE_DEVICES", "MKL_THREADING_LAYER", "PYTHONNOUSERSITE",
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG",
        )},
        "continuation/historical startup environment authority",
    )
    output = Path(checked["output_dir"])
    output.mkdir(parents=False, exist_ok=False)
    report = {
        "schema_version": 1, "kind": contract_module.RESULT_KIND, "status": "RUNNING",
        "campaign_id": checked["campaign_id"], "execution_id": checked["execution_id"],
        "invocation_id": checked["invocation_id"], "cell_id": checked["cell_id"],
        "stage": checked["stage"], "source_campaign_id": checked["source_campaign_id"],
        "source_run_id": checked["source_run_id"],
        "source_binding_sha256": checked["source_binding_sha256"],
        "contract_reference": copy.deepcopy(contract_reference),
        "freeze_reference": copy.deepcopy(checked["freeze_reference"]),
        "worker_pid": os.getpid(), "receipts": [], "checkpoints": {},
        "evaluations": [], "replay_checks": [], "scientific_certified": False,
        "startup_environment": startup_environment,
        "historical_startup_environment": historical_startup_environment,
        "automatic_retry_or_batch_fallback": False,
        "historical_run_relabelled": False, "source_artifacts_modified": False,
    }
    monitor = None
    started = time.time()
    try:
        modules["write_exclusive_json"](output / "worker-start.json", report)
        binding = modules["validate_run_binding"](
            _read_reference(contract_module, checked["source_binding_reference"], "source run binding"),
            verify_files=True)
        _same(binding["binding_sha256"], checked["source_binding_sha256"], "source binding SHA")
        _same(binding["run_id"], checked["source_run_id"], "source binding run identity")
        config = modules["V2BConfig"](**checked["config"])
        data = checked["train_core"]
        loader = modules["build_train_core_loader"](
            modules["TrainCoreDataConfig"](
                seed=config.seed, input_size=config.input_size, logical_batch_size=16,
                num_workers=checked["num_workers"], prefetch_factor=checked["prefetch_factor"]),
            annotation_file=data["annotation"]["path"], annotation_sha256=data["annotation"]["sha256"],
            manifest_file=data["manifest"]["path"], manifest_sha256=data["manifest"]["sha256"],
            image_root=data["image_root"], repo_root=checked["source_repo_root"])
        if len(loader.dataset) != 4869 or loader.batches_per_epoch != 304:
            raise RuntimeError("continuation loader differs from the frozen source input")
        arguments = {"repo_root": checked["source_repo_root"],
                     "pretrained_path": checked["pretrained"]["path"],
                     "pretrained_sha256": checked["pretrained"]["sha256"]}
        cpu = modules["build_v2b_components"](config, **arguments)
        # Construction source hashes and initialization are checked again by
        # V2BTrainingSession against the historical run binding.
        monitor = modules["MonitoredHardwareSession"](
            binding, checked["policy_bundle"], output / "native-hardware",
            "train_core_30epoch" if checked["stage"] == "formal60" else "paired_smoke",
            43200 if checked["stage"] == "formal60" else 600)
        monitor.start()
        runtime = modules["prepare_runtime"](
            device="cuda:0", seed=config.seed, binding=binding,
            gpu_probe=monitor.admission(), expected_gpu_uuid=checked["expected_gpu_uuid"])
        components = modules["build_v2b_components"](config, runtime=runtime, **arguments)
        _same(cpu.initialization, components.initialization, "CPU/CUDA source initialization")
        del cpu
        session = modules["V2BTrainingSession"](components, loader, binding)
        smoke, replay, matched, replay_index, matched_index = _load_prerequisites(
            contract_module, checked, control)
        if checked["stage"] == "smoke_replay":
            midpoint = smoke["checkpoints"]["window_2"]
            restore_reference = midpoint["checkpoint"]
            expected_boundary = midpoint["state_reference"]
        else:
            restore_reference = checked["source_checkpoint_reference"]
            expected_boundary = checked["source_boundary_state_reference"]
        monitor.check(stage="before_restore")
        report["restore"] = session.restore(restore_reference["path"], restore_reference["sha256"])
        import torch
        torch.cuda.synchronize(0)
        monitor.check(stage="after_restore")
        restored = control.capture_control_state(components, loader, full=True)
        _same(_read_reference(contract_module, expected_boundary, "expected restore boundary"),
              restored, "complete restored continuation boundary")
        report["restored_boundary_reference"] = modules["write_exclusive_json"](
            output / "restored-boundary-state.json", restored)
        report["restored_from"] = copy.deepcopy(restore_reference)
        _execute_stage(
            checked, session, monitor, output, report, control=control,
            evaluate_development=modules["evaluate_development"],
            write_exclusive_json=modules["write_exclusive_json"],
            replay_index=replay_index, matched_index=matched_index)
        final = control.capture_control_state(components, loader, full=True)
        report["final_clocks"] = _final_clocks(
            checked["stage"], components, report["receipts"], report["replay_checks"],
            report["evaluations"], report["checkpoints"])
        report["final_state_reference"] = modules["write_exclusive_json"](
            output / "final-state.json", final)
        report["receipt_count"] = len(report["receipts"])
        report["logical_samples_executed_this_invocation"] = len(report["receipts"]) * 16
        report["source_optimizer_updates_before"] = 9120
        report["monitor_reference"] = monitor.finish()
        report["elapsed_seconds"] = time.time() - started
        report["status"] = "PASS"
        modules["write_exclusive_json"](output / "worker-result.json", report)
        return contract_module.validate_continuation_result(report, checked, verify_files=False)
    except BaseException as exc:
        report["status"] = "STOP_NO_RETRY"
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        report["elapsed_seconds"] = time.time() - started
        try:
            modules["write_exclusive_json"](output / "worker-result.json", report)
        finally:
            if monitor is not None:
                monitor.abort("epoch60 continuation worker failure: " + type(exc).__name__ + ": " + str(exc))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--contract-sha256", required=True)
    arguments = parser.parse_args(argv)
    try:
        contract, reference = _load_contract(arguments.contract, arguments.contract_sha256)
        contract_module = _contract_module(contract)
        result = run_continuation_worker(contract, reference, contract_module)
    except BaseException as exc:
        print(json.dumps({"status": "STOP_NO_RETRY", "failure_type": type(exc).__name__,
                          "failure": str(exc)}, ensure_ascii=False, allow_nan=False), flush=True)
        return 2
    print(json.dumps({
        "status": result["status"], "campaign_id": result["campaign_id"],
        "execution_id": result["execution_id"], "cell_id": result["cell_id"],
        "stage": result["stage"], "source_run_id": result["source_run_id"],
        "output_dir": contract["output_dir"], "historical_run_relabelled": False,
    }, ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
