#!/usr/bin/env python3
"""One-shot REV1 GPU pre-formal smoke with versioned runtime bindings.

This file is uploaded to /tmp only.  It does not belong to the scientific or
execution source identity and is never used for formal continuation.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import torch

from sparse_rtdetr.baseline.training_v2b import V2BConfig, build_v2b_components
from sparse_rtdetr.baseline.training_v2b_admission import (
    MonitoredHardwareSession, wait_for_monitored_finish,
)
from sparse_rtdetr.baseline.training_v2b_control import capture_control_state
from sparse_rtdetr.baseline.training_v2b_data import (
    TrainCoreDataConfig, augmented_tensor_sha256, build_train_core_loader,
)
from sparse_rtdetr.baseline.training_v2b_device import prepare_runtime
from sparse_rtdetr.baseline.training_v2b_runtime import (
    V2BTrainingSession,
    build_train_core_run_binding,
)
from sparse_rtdetr.baseline.training_v2b_evidence import canonical_json_bytes, validate_run_binding
from sparse_rtdetr.baseline.training_v2b_checkpoint import inspect_checkpoint
from sparse_rtdetr.baseline.training_v2b_runtime_locator import (
    RuntimeLocatorError,
    load_canonical_json,
    resolve_bridge,
    resolve_policy_authority,
)
from sparse_rtdetr.baseline.training_v2b_smoke_controller import verify_startup_environment


# All physical values below are supplied by the runtime launch contract.  They
# are deliberately not part of the scientific or execution identity.
REPO: Path | None = None
DERIVED: Path | None = None
DERIVED_SHA = ""
BRIDGE_MANIFEST_PATH: Path | None = None
POLICY_PATH: Path | None = None  # resolved runtime-only path, never a caller input
POLICY_SHA = ""
POLICY_SIZE: int | None = None
POLICY_AUTHORITY_PATH: Path | None = None
POLICY_AUTHORITY_ID = ""
POLICY_AUTHORITY_IDENTITY_SHA = ""
GPU_UUID = ""
AUTH_ID = ""
AUTH_SHA = ""
AUTH_PATH: Path | None = None
EXEC_SOURCE_SHA = ""
EXEC_CONTRACT_SHA = ""
EXEC_CONTRACT_PATH: Path | None = None
EXECUTION_SOURCE_PATH: Path | None = None
REVISION_VERIFIER_PATH: Path | None = None
STRUCTURED_INVOCATION_PATH: Path | None = None
FROZEN_PLAN_SHA = ""
EXPECTED_EXECUTION_CONTRACT_ID = "v2b-execution-contract-011"
TRAINING_CHILD_MODE = False
BRIDGE_BINDING_SHA = ""
BRIDGE_MANIFEST_SHA = ""
TRAINING_CONTRACT_SHA = ""
SCHEMA_VERSION = 1


def _configured() -> None:
    if not all((REPO, DERIVED, DERIVED_SHA, BRIDGE_MANIFEST_PATH,
                POLICY_AUTHORITY_PATH, POLICY_AUTHORITY_ID, POLICY_AUTHORITY_IDENTITY_SHA,
                AUTH_ID, AUTH_SHA, AUTH_PATH, EXEC_SOURCE_SHA, EXEC_CONTRACT_SHA,
                EXEC_CONTRACT_PATH, BRIDGE_BINDING_SHA, BRIDGE_MANIFEST_SHA, TRAINING_CONTRACT_SHA,
                STRUCTURED_INVOCATION_PATH, FROZEN_PLAN_SHA)):
        raise RuntimeLocatorError("runtime launch arguments are incomplete")


def _configuration_argv() -> list[str]:
    _configured()
    argv = [
        "--repo", str(REPO), "--derived", str(DERIVED), "--derived-sha", DERIVED_SHA,
        "--manifest", str(BRIDGE_MANIFEST_PATH),
        "--policy-authority", str(POLICY_AUTHORITY_PATH),
        "--policy-authority-id", POLICY_AUTHORITY_ID,
        "--policy-authority-identity-sha", POLICY_AUTHORITY_IDENTITY_SHA,
        "--auth", str(AUTH_PATH), "--auth-id", AUTH_ID, "--auth-sha", AUTH_SHA,
        "--exec-source-sha", EXEC_SOURCE_SHA, "--exec-contract-sha", EXEC_CONTRACT_SHA,
        "--execution-contract", str(EXEC_CONTRACT_PATH),
        "--bridge-binding-sha", BRIDGE_BINDING_SHA, "--bridge-manifest-sha", BRIDGE_MANIFEST_SHA,
        "--training-contract-sha", TRAINING_CONTRACT_SHA, "--gpu-uuid", GPU_UUID,
        "--expected-execution-contract-id", EXPECTED_EXECUTION_CONTRACT_ID,
    ]
    if EXECUTION_SOURCE_PATH is not None:
        argv += ["--execution-source", str(EXECUTION_SOURCE_PATH)]
    if REVISION_VERIFIER_PATH is not None:
        argv += ["--revision-verifier", str(REVISION_VERIFIER_PATH)]
    if STRUCTURED_INVOCATION_PATH is not None:
        argv += ["--invocation", str(STRUCTURED_INVOCATION_PATH), "--plan-sha", FROZEN_PLAN_SHA]
    return argv


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


RUNTIME_KEYS = {"runtime_locator", "runtime_file_count", "runtime_locators", "state_runtime_locator"}


def without_runtime(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: without_runtime(item) for key, item in value.items() if key not in RUNTIME_KEYS}
    if isinstance(value, list):
        return [without_runtime(item) for item in value]
    return value


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_sha(value: torch.Tensor) -> str:
    plain = value.detach().cpu().contiguous().as_subclass(torch.Tensor)
    h = hashlib.sha256()
    h.update(canonical([str(plain.dtype), list(plain.shape)]))
    h.update(plain.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def target_structure_sha(targets: list[dict]) -> str:
    rows = []
    for index, target in enumerate(targets):
        fields = []
        for key in sorted(target):
            value = target[key]
            if isinstance(value, torch.Tensor):
                fields.append([key, "tensor", str(value.dtype), list(value.shape), tensor_sha(value)])
            else:
                fields.append([key, type(value).__name__, repr(value)])
        rows.append([index, fields])
    return sha_bytes(canonical(rows))


def owned_input_sha(images: torch.Tensor, targets: list[dict]) -> dict[str, str]:
    return {
        "images_sha256": tensor_sha(images),
        "targets_sha256": target_structure_sha(targets),
        "augmented_sha256": augmented_tensor_sha256(images, targets),
    }


def load_policy() -> dict:
    global POLICY_PATH, POLICY_SHA, POLICY_SIZE
    _configured()
    resolved = resolve_policy_authority(
        POLICY_AUTHORITY_PATH,
        authority_id=POLICY_AUTHORITY_ID,
        expected_identity_sha256=POLICY_AUTHORITY_IDENTITY_SHA,
    )
    POLICY_PATH = Path(resolved["file"]["path"])
    POLICY_SHA = str(resolved["file"]["sha256"])
    POLICY_SIZE = int(resolved["file"]["size_bytes"])
    policy = dict(resolved["policy"])
    for runtime_field in ("authorized_scope", "workload_deadline_seconds", "minimum_loaded_clock_samples", "policy_sha256"):
        policy.pop(runtime_field, None)
    return policy


def _execution_contract() -> dict[str, Any]:
    _configured()
    assert EXEC_CONTRACT_PATH is not None
    raw = EXEC_CONTRACT_PATH.read_bytes()
    contract = load_canonical_json(EXEC_CONTRACT_PATH)
    if contract.get("execution_contract_sha256") != EXEC_CONTRACT_SHA:
        raise RuntimeLocatorError("EXECUTION_CONTRACT_IDENTITY_MISMATCH")
    if contract.get("execution_contract_id") != EXPECTED_EXECUTION_CONTRACT_ID:
        raise RuntimeLocatorError("EXECUTION_CONTRACT_REVISION_MISMATCH")
    body = {key: value for key, value in contract.items() if key != "execution_contract_sha256"}
    if contract.get("execution_contract_sha256") != sha_bytes(canonical(without_runtime(body))):
        raise RuntimeLocatorError("EXECUTION_CONTRACT_DIGEST_MISMATCH")
    if contract.get("execution_source_sha256") != EXEC_SOURCE_SHA:
        raise RuntimeLocatorError("EXECUTION_SOURCE_BINDING_MISMATCH")
    if contract.get("training_contract_sha256") != TRAINING_CONTRACT_SHA:
        raise RuntimeLocatorError("TRAINING_CONTRACT_BINDING_MISMATCH")
    return contract


def _verify_startup_environment(root: Path, *, cpu_rehearsal: bool) -> dict[str, Any]:
    contract = _execution_contract()
    authority = contract.get("startup_environment")
    if not isinstance(authority, dict):
        raise RuntimeLocatorError("STARTUP_ENVIRONMENT_AUTHORITY_MISSING")
    static = authority.get("static")
    mode_name = "cpu_rehearsal" if cpu_rehearsal else "gpu"
    mode = authority.get(mode_name)
    if not isinstance(static, dict) or not isinstance(mode, dict):
        raise RuntimeLocatorError("STARTUP_ENVIRONMENT_AUTHORITY_INVALID")
    expected = {str(key): str(value) for key, value in static.items()}
    expected.update({str(key): str(value) for key, value in mode.items()})
    if not cpu_rehearsal and expected.get("CUDA_VISIBLE_DEVICES") != GPU_UUID:
        raise RuntimeLocatorError("STARTUP_ENVIRONMENT_GPU_BINDING_MISMATCH")
    if cpu_rehearsal and expected.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeLocatorError("STARTUP_ENVIRONMENT_CPU_BINDING_MISMATCH")
    try:
        observed = verify_startup_environment(expected)
    except Exception as exc:
        raise RuntimeLocatorError(str(exc)) from exc
    evidence = {
        "status": "PASS",
        "mode": mode_name,
        "execution_contract_sha256": EXEC_CONTRACT_SHA,
        "expected": expected,
        "observed": observed,
        "python_dont_write_bytecode": bool(sys.dont_write_bytecode),
    }
    _write(root / "startup-environment.json", evidence)
    if not sys.dont_write_bytecode:
        raise RuntimeLocatorError("STARTUP_ENVIRONMENT_MISMATCH: bytecode generation enabled")
    return evidence


def make_config(binding: dict) -> V2BConfig:
    old = binding["config"]
    fields = ("seed", "input_size", "physical_batch_size", "accumulation_steps", "amp_dtype",
              "bn_statistics", "pretrained_required", "num_denoising", "learning_rate",
              "weight_decay", "clip_max_norm", "warmup_steps", "ema_decay", "ema_warmups",
              "sampling_backend")
    return V2BConfig(**{name: old[name] for name in fields})


def build_loader(binding: dict, *, repo: Path):
    data = binding["data"]["loader"]
    cfg = data["config"]
    return build_train_core_loader(
        TrainCoreDataConfig(
            seed=cfg["seed"], input_size=cfg["input_size"],
            logical_batch_size=cfg["logical_batch_size"], num_workers=0,
            prefetch_factor=2, augmentation_stop_internal_epoch=cfg["augmentation_stop_internal_epoch"],
        ), annotation_file=data["annotation"]["path"], annotation_sha256=data["annotation"]["sha256"],
        manifest_file=data["manifest"]["path"], manifest_sha256=data["manifest"]["sha256"],
        image_root=data["image_root"], repo_root=repo,
    )


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value))


def _gpu_admission_evidence(root: Path, monitor: MonitoredHardwareSession,
                            policy: dict[str, Any]) -> dict[str, Any]:
    """Return JSON evidence without serializing the live capability object.

    MonitoredHardwareAdmission is deliberately an in-process capability
    handle and must never cross an evidence/JSON boundary. Calling
    monitor.admission() still proves that the live capability exists; the
    published record contains only stable, serializable identity fields.
    """
    admission = monitor.admission()
    evidence = {
        "status": "PASS",
        "monitor_root": str(getattr(monitor, "root", root / "quiescence-native-hardware")),
        "policy_sha256": policy.get("policy_sha256"),
        "gpu_uuid": policy.get("gpu_uuid"),
        "capability_type": type(admission).__name__,
    }
    if hasattr(monitor, "monitor_phase"):
        evidence["monitor_phase"] = monitor.monitor_phase
    if hasattr(monitor, "transaction_id"):
        evidence["transaction_id"] = monitor.transaction_id
    return evidence


def _finish_monitor(monitor: MonitoredHardwareSession) -> tuple[dict[str, Any], dict[str, Any]]:
    """Publish and then verify the independent monitor's terminal report."""
    reference = monitor.finish()
    final = wait_for_monitored_finish(reference)
    return reference, final


def _restore_evidence(session: V2BTrainingSession, loader: Any, restored: dict[str, Any],
                      checkpoint: Path, checkpoint_sha: str, next_window: dict[str, Any]) -> dict[str, Any]:
    state = capture_control_state(session.components, loader, full=True)
    clocks = state["clocks"]
    return {
        "validated": True,
        "validation_method": "V2BTrainingSession.restore strict state/layout/RNG/sampler checks",
        "restore_input_checkpoint_path": str(checkpoint),
        "restore_input_checkpoint_sha256": checkpoint_sha,
        "restored": dict(restored),
        "counters": {
            "epoch": restored["epoch"],
            "optimizer_updates": restored["optimizer_updates"],
            "microsteps": restored["microsteps"],
            "ema_updates": int(session.components.ema.updates),
        },
        "next_boundary": {
            "epoch": next_window["epoch"],
            "logical_batch_index": next_window["logical_batch_index"],
        },
        "state_digests": {
            "model": state["raw_state_sha256"],
            "optimizer": state["optimizer_state_sha256"],
            "ema": state["ema_state_sha256"],
            "rng": sha_bytes(canonical_json_bytes(state["rng"])),
            "loader_sampler": sha_bytes(canonical_json_bytes(clocks["loader"])),
            "scheduler_warmup": sha_bytes(canonical_json_bytes({
                "scheduler": clocks["scheduler"], "warmup": clocks["warmup"],
            })),
            "engine": sha_bytes(canonical_json_bytes(clocks["engine"])),
        },
        "clock_state": clocks,
    }


def _run_revision_verifier(root: Path) -> dict[str, Any]:
    """Run the tracked REV1 identity verifier before declaring READY.

    This is deliberately a subprocess: the rehearsal must exercise the same
    verifier entry point that a real smoke controller uses, while keeping the
    verifier's imports and any CUDA-related environment isolated from the
    runner process.
    """
    _configured()
    contract_dir = EXEC_CONTRACT_PATH.parent if EXEC_CONTRACT_PATH is not None else REPO / "contracts" / "v2b" / "rev001"
    policy_authority = contract_dir / "runtime_policy_authority_r1.json"
    execution_source = EXECUTION_SOURCE_PATH or contract_dir / "execution_source_r11.json"
    verifier = REVISION_VERIFIER_PATH or REPO / "tools" / "verify_v2b_smoke_launcher_r11.py"
    command = [
        sys.executable,
        str(verifier),
        "--repo-root", str(REPO),
        "--contract-dir", str(contract_dir),
        "--bridge", str(DERIVED),
        "--bridge-manifest", str(BRIDGE_MANIFEST_PATH),
        "--policy-authority", str(policy_authority),
        "--execution-source", str(execution_source),
        "--execution-source-sha", EXEC_SOURCE_SHA,
        "--execution-contract", str(EXEC_CONTRACT_PATH),
        "--execution-contract-sha", EXEC_CONTRACT_SHA,
        "--authorization", str(AUTH_PATH),
        "--authorization-id", AUTH_ID,
        "--authorization-sha", AUTH_SHA,
        "--invocation", str(STRUCTURED_INVOCATION_PATH), "--plan-sha", FROZEN_PLAN_SHA,
    ]
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        command, cwd=str(REPO), env=environment,
        capture_output=True, text=True, check=False,
    )
    (root / "revision-verifier-stdout.log").write_text(completed.stdout, encoding="utf-8")
    (root / "revision-verifier-stderr.log").write_text(completed.stderr, encoding="utf-8")
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeLocatorError(f"REV1 verifier returned non-JSON output: {exc}") from exc
    if completed.returncode != 0 or result.get("status") != "PASS":
        raise RuntimeLocatorError(f"REV1/bridge verifier failed: {result}")
    return result


def _ready(root: Path, binding: dict, restored: dict) -> None:
    _write(root / "handshake" / "ready.json", {
        "schema_version": SCHEMA_VERSION, "event": "READY", "pid": os.getpid(),
        "cwd": os.getcwd(), "python_executable": sys.executable,
        "cuda_initialized": bool(torch.cuda.is_initialized()),
        "authorization_id": AUTH_ID, "authorization_sha256": AUTH_SHA,
        "bridge_checkpoint_sha256": DERIVED_SHA,
        "binding_sha256": binding["binding_sha256"], "restore": restored,
        "next_boundary": {"epoch": 54, "logical_batch_index": 0},
    })



def _quiescence_probe(root: Path) -> dict[str, Any]:
    """Validate runtime identity and hardware quiescence without CUDA init."""
    _configured()
    startup = _verify_startup_environment(root, cpu_rehearsal=False)
    verification = _run_revision_verifier(root)
    bridge = resolve_bridge(
        DERIVED, checkpoint_sha256=DERIVED_SHA, manifest=BRIDGE_MANIFEST_PATH,
        manifest_sha256=BRIDGE_MANIFEST_SHA,
    )
    payload = torch.load(DERIVED, map_location="cpu", weights_only=True)
    binding = payload["binding"]
    if binding.get("binding_sha256") != BRIDGE_BINDING_SHA:
        raise RuntimeLocatorError("bridge binding SHA mismatch")
    policy = load_policy()
    monitor = MonitoredHardwareSession(
        binding, policy, root / "quiescence-native-hardware", "paired_smoke",
        monitor_phase="quiescence", transaction_id=AUTH_ID,
    )
    try:
        monitor.start()
        report = {
            "status": "PASS",
            "startup_environment": startup,
            "revision_verifier": verification,
            "bridge_checkpoint_sha256": DERIVED_SHA,
            "bridge_manifest_sha256": BRIDGE_MANIFEST_SHA,
            "binding_sha256": binding["binding_sha256"],
            "policy_sha256": policy.get("policy_sha256"),
            "owner_none_admission": True,
            "cuda_initialized": bool(torch.cuda.is_initialized()),
            "training_started": False,
            "gpu_admission": _gpu_admission_evidence(root, monitor, policy),
        }
        if report["cuda_initialized"]:
            raise RuntimeError("quiescence probe initialized CUDA")
        monitor_reference, monitor_final = _finish_monitor(monitor)
        monitor = None
        report["monitor_finish"] = monitor_reference
        report["monitor_final"] = monitor_final
        _write(root / "quiescence-report.json", report)
        return report
    finally:
        if monitor is not None and monitor._state == "running":
            monitor.abort("quiescence logical path did not reach verified monitor final")


def _roundtrip(checkpoint: Path, checkpoint_sha: str, root: Path) -> dict:
    _configured()
    _verify_startup_environment(root, cpu_rehearsal=False)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    binding = payload["binding"]
    config = make_config(binding)
    policy = load_policy()
    hw_root = root / "roundtrip-native-hardware"
    monitor = MonitoredHardwareSession(
        binding, policy, hw_root, "paired_smoke", monitor_phase="restore", transaction_id=AUTH_ID,
    )
    try:
        monitor.start()
        runtime = prepare_runtime(device="cuda:0", seed=config.seed, binding=binding,
                                  gpu_probe=monitor.admission(), expected_gpu_uuid=GPU_UUID)
        weights = Path(binding["config"]["pretrained"]["path"])
        components = build_v2b_components(
            config, repo_root=REPO, pretrained_path=weights,
            pretrained_sha256=binding["config"]["pretrained"]["sha256"], runtime=runtime,
        )
        loader = build_loader(binding, repo=REPO)
        session = V2BTrainingSession(components, loader, binding)
        restored = session.restore(checkpoint, checkpoint_sha)
        if (restored["epoch"], restored["optimizer_updates"], restored["microsteps"]) != (54, 16113, 32226):
            raise RuntimeError("smoke checkpoint restore counters differ")
        first = next(loader.preview_batches(1)).evidence()
        second = next(loader.preview_batches(1)).evidence()
        if first != second:
            raise RuntimeError("epoch54/window2 deterministic preview differs")
        monitor_reference, monitor_final = _finish_monitor(monitor)
        monitor = None
        restore_evidence = _restore_evidence(session, loader, restored, checkpoint, checkpoint_sha, first)
        return {
            "status": "PASS", "restored": restored, "next_window_first": first,
            "next_window_second": second, "deterministic": True,
            "restore_evidence": restore_evidence,
            "monitor_finish": monitor_reference, "monitor_final": monitor_final,
            "monitor_root": str(hw_root),
        }
    finally:
        if monitor is not None and monitor._state == "running":
            monitor.abort("roundtrip logical path did not reach verified monitor final")


def cpu_rehearsal(root: Path) -> dict[str, Any]:
    """Run the full controller/runner initialization without CUDA or updates."""
    _configured()
    if not root.is_dir():
        raise RuntimeError("launch root missing")
    runner_workspace = root / "runner"
    runner_workspace.mkdir(parents=False, exist_ok=False)
    startup_environment = _verify_startup_environment(root, cpu_rehearsal=True)
    verification = _run_revision_verifier(root)
    bridge = resolve_bridge(
        DERIVED,
        checkpoint_sha256=DERIVED_SHA,
        manifest=BRIDGE_MANIFEST_PATH,
        manifest_sha256=BRIDGE_MANIFEST_SHA,
    )
    policy = load_policy()
    payload = torch.load(DERIVED, map_location="cpu", weights_only=True)
    binding = payload["binding"]
    if binding.get("binding_sha256") != BRIDGE_BINDING_SHA:
        raise RuntimeLocatorError("bridge binding SHA mismatch")
    config = make_config(binding)
    checkpoint_inspection = inspect_checkpoint(
        DERIVED, expected_sha256=DERIVED_SHA, expected_binding=binding,
    )
    # The CPU rehearsal is intentionally limited to restore/preview; it never
    # calls model forward, criterion, backward, optimizer, EMA, or CUDA APIs.
    runtime = prepare_runtime(device="cpu", seed=config.seed, binding=binding)
    weights = Path(binding["config"]["pretrained"]["path"])
    components = build_v2b_components(
        config,
        repo_root=REPO,
        pretrained_path=weights,
        pretrained_sha256=binding["config"]["pretrained"]["sha256"],
        runtime=runtime,
    )
    loader = build_loader(binding, repo=REPO)
    cpu_binding = build_train_core_run_binding(
        components, loader, run_id=f"{AUTH_ID}-cpu",
        repo_root=REPO, requested_device="cpu", cuda_gpu_uuid=None,
    )
    cpu_binding["provenance"] = copy.deepcopy(binding["provenance"])
    body = {key: value for key, value in cpu_binding.items() if key != "binding_sha256"}
    cpu_binding["binding_sha256"] = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    # The checkpoint binding carries the historical execution-source closure.
    # Scientific/loader references remain shape- and SHA-validated above; the
    # current execution layer is independently sealed by the REV1 verifier.
    validate_run_binding(binding, verify_files=False)
    binding_validation = {
        "status": "PASS",
        "historical_execution_files_verified": False,
        "scientific_loader_references_verified": True,
        "execution_source_verifier": "REV1 independent execution-source identity",
    }
    session = V2BTrainingSession(components, loader, cpu_binding)
    # A CUDA-bound checkpoint cannot be live-restored into a CPU runtime without
    # changing its sealed binding.  The strict, weights-only checkpoint
    # inspector verifies the complete restore identity without initializing
    # CUDA; the CPU session then exercises the same construction/preview path.
    if checkpoint_inspection.get("engine", {}).get("epoch") != 53:
        raise RuntimeError("bridge-v2 strict checkpoint inspection boundary mismatch")
    first = next(loader.preview_batches(1, epoch=54)).evidence()
    second = next(loader.preview_batches(1, epoch=54)).evidence()
    if first != second:
        raise RuntimeError("epoch54/window1 deterministic preview differs")
    ready = {
        "schema_version": SCHEMA_VERSION,
        "event": "READY",
        "pid": os.getpid(),
        "cwd": os.getcwd(),
        "python_executable": sys.executable,
        "cuda_initialized": bool(torch.cuda.is_initialized()),
        "training_started": False,
        "startup_environment": startup_environment,
        "authorization_id": AUTH_ID,
        "authorization_sha256": AUTH_SHA,
        "bridge_checkpoint_sha256": DERIVED_SHA,
        "bridge_manifest_sha256": BRIDGE_MANIFEST_SHA,
        "binding_sha256": binding["binding_sha256"],
        "runtime_locator_policy_sha256": POLICY_SHA,
        "revision_verifier": verification,
        "restore_inspection": checkpoint_inspection,
        "preview": first,
    }
    _write(root / "handshake" / "ready.json", ready)
    report = {
        "status": "READY",
        "cuda_initialized": False,
        "training_started": False,
        "startup_environment": startup_environment,
        "runtime_locator": {
            "policy": policy.get("policy_sha256"),
            "bridge": bridge["checkpoint"],
            "bridge_manifest": bridge["manifest"],
        },
        "revision_verifier": verification,
        "restore_inspection": checkpoint_inspection,
        "window1_preview": first,
        "window1_preview_repeat": second,
        "deterministic": first == second,
        "session_constructed": True,
        "binding_validation": binding_validation,
    }
    _write(root / "cpu-rehearsal-report.json", report)
    return report


def run(root: Path) -> dict:
    _configured()
    if not root.is_dir():
        raise RuntimeError("launch root missing")
    runner_workspace = root / "runner"
    runner_workspace.mkdir(parents=False, exist_ok=False)
    started_at = time.time()
    monitor = None
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "status": "RUNNING", "stage": "REV1_GPU_PREFORMAL_SMOKE",
        "authorization": {"id": AUTH_ID, "sha256": AUTH_SHA, "consumed": False,
                           "formal_launch_permitted": False},
        "execution_source_sha256": EXEC_SOURCE_SHA, "execution_contract_sha256": EXEC_CONTRACT_SHA,
        "bridge_binding_sha256": BRIDGE_BINDING_SHA, "bridge_manifest_sha256": BRIDGE_MANIFEST_SHA,
        "parent_checkpoint_sha256": "29bc58ebb5118a1bdd88da741783de3270876b8cce95e47683dedc2c37f01060",
        "derived_checkpoint_sha256": DERIVED_SHA,
        "runner_pid": os.getpid(), "started_at": started_at,
        "gpu_operations": 0, "training_windows": 0, "optimizer_updates": 0,
    }
    _write(root / "smoke-start.json", report)
    try:
        startup_environment = _verify_startup_environment(root, cpu_rehearsal=False)
        bridge = resolve_bridge(
            DERIVED,
            checkpoint_sha256=DERIVED_SHA,
            manifest=BRIDGE_MANIFEST_PATH,
            manifest_sha256=BRIDGE_MANIFEST_SHA,
        )
        if bridge["manifest_document"].get("binding_sha256") not in (None, BRIDGE_BINDING_SHA):
            raise RuntimeLocatorError("bridge manifest binding identity mismatch")
        if sha_file(DERIVED) != DERIVED_SHA:
            raise RuntimeLocatorError("derived checkpoint SHA mismatch")
        payload = torch.load(DERIVED, map_location="cpu", weights_only=True)
        binding = payload["binding"]
        if binding.get("binding_sha256") != BRIDGE_BINDING_SHA:
            raise RuntimeError("bridge binding SHA mismatch")
        config = make_config(binding)
        policy = load_policy()
        monitor = MonitoredHardwareSession(
            binding, policy, root / "native-hardware", "paired_smoke",
            monitor_phase="training", transaction_id=AUTH_ID,
        )
        monitor.start()
        report["gpu_operations"] = 1
        report["gpu_admission"] = {"status": "PASS", "monitor_root": str(root / "native-hardware"),
                                    "policy_sha256": policy.get("policy_sha256"), "gpu_uuid": GPU_UUID}
        report["startup_environment"] = startup_environment
        runtime = prepare_runtime(device="cuda:0", seed=config.seed, binding=binding,
                                  gpu_probe=monitor.admission(), expected_gpu_uuid=GPU_UUID)
        weights = Path(binding["config"]["pretrained"]["path"])
        components = build_v2b_components(
            config, repo_root=REPO, pretrained_path=weights,
            pretrained_sha256=binding["config"]["pretrained"]["sha256"], runtime=runtime,
        )
        loader = build_loader(binding, repo=REPO)
        session = V2BTrainingSession(components, loader, binding)
        restored = session.restore(DERIVED, DERIVED_SHA)
        if (restored["epoch"], restored["optimizer_updates"], restored["microsteps"]) != (53, 16112, 32224):
            raise RuntimeError("bridge-v2 restore boundary mismatch")
        session.begin_epoch(54)
        _ready(root, binding, restored)
        monitor.check(stage="before_window")
        torch.cuda.synchronize(0)
        torch.cuda.reset_peak_memory_stats(0)
        t0 = time.perf_counter()
        iterator = iter(loader)
        batch = next(iterator)
        input_before = owned_input_sha(batch.images, batch.targets)
        preview = batch.evidence()
        if preview.get("logical_batch_index") != 0 or preview.get("epoch") != 54:
            raise RuntimeError("smoke did not start at epoch54/window1")
        if preview.get("augmented_sha256") != "d436e899db90efa6d9a56847254dd2465b20e69ef1b0c37937fcff1dd47b67a3":
            raise RuntimeError("epoch54/window1 augmented SHA differs from bridge preview")
        if preview.get("receipt_sha256") != "7f15c554398a9f9086f1ad4282ad38d2d5a17c7ee0c91956eb423d7363649132":
            raise RuntimeError("epoch54/window1 receipt SHA differs from bridge preview")
        monitor.check(stage="before_forward")
        record = session.train_batch(batch, hardware_probe=monitor.admission())
        torch.cuda.synchronize(0)
        elapsed = time.perf_counter() - t0
        input_after = owned_input_sha(batch.images, batch.targets)
        if input_after != input_before:
            raise RuntimeError("loader-owned input SHA drift after forward/backward/optimizer")
        monitor.check(stage="after_window")
        try:
            iterator.close()
        except AttributeError:
            pass
        report["training_windows"] = 1
        report["optimizer_updates"] = 1
        window = record["window"]
        loss_terms = sorted({key for row in window["microbatch_losses"] for key in row})
        if len(loss_terms) != 21:
            raise RuntimeError(f"expected 21 loss terms, observed {len(loss_terms)}")
        finite_terms = all(torch.isfinite(torch.tensor(float(row[key]))) for row in window["microbatch_losses"] for key in loss_terms)
        if not finite_terms or not torch.isfinite(torch.tensor(float(window["loss"]))) or not torch.isfinite(torch.tensor(float(window["gradient_norm_before_clip"]))):
            raise RuntimeError("non-finite loss or gradient")
        smoke_checkpoint = root / "smoke-checkpoint-epoch54-window01.pt"
        checkpoint = session.save(smoke_checkpoint)
        smoke_sha = checkpoint["sha256"]
        if not smoke_sha or sha_file(smoke_checkpoint) != smoke_sha:
            raise RuntimeError("smoke checkpoint SHA publication mismatch")
        gpu_memory = {
            "allocated": int(torch.cuda.memory_allocated(0)),
            "reserved": int(torch.cuda.memory_reserved(0)),
            "peak_allocated": int(torch.cuda.max_memory_allocated(0)),
            "peak_reserved": int(torch.cuda.max_memory_reserved(0)),
        }
        ema_updates = int(components.ema.updates)
        monitor_report, monitor_final = _finish_monitor(monitor)
        monitor = None
        del session, components, loader, runtime
        torch.cuda.empty_cache()
        if TRAINING_CHILD_MODE:
            report.update({
                "status": "TRAINING_CHILD_PASS",
                "restore": restored,
                "epoch54_window1": preview,
                "ownership_before": input_before,
                "ownership_after": input_after,
                "ownership_pass": True,
                "window_record": record,
                "loss_terms": loss_terms,
                "loss_term_count": len(loss_terms),
                "finite_loss_and_gradient": True,
                "optimizer_updates_after": 16113,
                "microsteps_after": 32226,
                "ema_updates_after": ema_updates,
                "timing_seconds": elapsed,
                "gpu_memory": gpu_memory,
                "smoke_checkpoint": checkpoint,
                "smoke_checkpoint_path": str(smoke_checkpoint),
                "smoke_checkpoint_sha256": smoke_sha,
                "monitor_finish": monitor_report,
                "monitor_final": monitor_final,
                "training_child_terminated": True,
                "formal_authorization_consumed": False,
                "formal_launch_permitted": False,
            })
            _write(root / "training-report.json", report)
            return report
        roundtrip_cmd = [sys.executable, __file__, "--roundtrip", "--checkpoint", str(smoke_checkpoint),
                         "--checkpoint-sha", smoke_sha, "--root", str(root)] + _configuration_argv()
        roundtrip_proc = subprocess.run(roundtrip_cmd, cwd=str(REPO), capture_output=True, text=True, check=False)
        (root / "roundtrip-stdout.log").write_text(roundtrip_proc.stdout, encoding="utf-8")
        (root / "roundtrip-stderr.log").write_text(roundtrip_proc.stderr, encoding="utf-8")
        if roundtrip_proc.returncode != 0:
            raise RuntimeError(f"independent roundtrip failed rc={roundtrip_proc.returncode}")
        roundtrip = json.loads(roundtrip_proc.stdout)
        if roundtrip.get("status") != "PASS":
            raise RuntimeError("independent roundtrip did not pass")
        report.update({
            "status": "REV1_GPU_PREFORMAL_SMOKE_PASS",
            "restore": restored, "epoch54_window1": preview,
            "ownership_before": input_before, "ownership_after": input_after,
            "ownership_pass": True, "window_record": record,
            "loss_terms": loss_terms, "loss_term_count": len(loss_terms),
            "finite_loss_and_gradient": True,
            "optimizer_updates_after": 16113, "microsteps_after": 32226,
            "ema_updates_after": ema_updates,
            "timing_seconds": elapsed,
            "gpu_memory": gpu_memory,
            "smoke_checkpoint": checkpoint, "smoke_checkpoint_sha256": smoke_sha,
            "monitor_finish": monitor_report, "monitor_final": monitor_final, "roundtrip": roundtrip,
            "formal_authorization_consumed": False, "formal_launch_permitted": False,
        })
        report["evidence_sha256"] = sha_bytes(canonical({k: v for k, v in report.items() if k != "evidence_sha256"}))
        _write(root / "smoke-report.json", report)
        return report
    except BaseException as exc:
        report.update({"status": "REV1_GPU_PREFORMAL_SMOKE_FAIL", "failure": {
            "type": type(exc).__name__, "message": str(exc),
        }, "formal_authorization_consumed": False, "formal_launch_permitted": False})
        _write(root / "smoke-report.json", report)
        if monitor is not None and monitor._state == "running":
            try:
                monitor.abort(f"REV1 GPU smoke failure: {type(exc).__name__}: {exc}")
            except BaseException:
                pass
        raise


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--roundtrip", action="store_true")
    parser.add_argument("--cpu-rehearsal", action="store_true")
    parser.add_argument("--training-child", action="store_true")
    parser.add_argument("--quiescence-probe", action="store_true")
    parser.add_argument("--expected-execution-contract-id", default="v2b-execution-contract-025")
    parser.add_argument("--execution-source")
    parser.add_argument("--revision-verifier")
    parser.add_argument("--invocation", required=True)
    parser.add_argument("--plan-sha", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint-sha")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--derived", required=True)
    parser.add_argument("--derived-sha", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--policy-authority", required=True)
    parser.add_argument("--policy-authority-id", required=True)
    parser.add_argument("--policy-authority-identity-sha", required=True)
    parser.add_argument("--auth", required=True)
    parser.add_argument("--auth-id", required=True)
    parser.add_argument("--auth-sha", required=True)
    parser.add_argument("--exec-source-sha", required=True)
    parser.add_argument("--exec-contract-sha", required=True)
    parser.add_argument("--execution-contract", required=True)
    parser.add_argument("--bridge-binding-sha", required=True)
    parser.add_argument("--bridge-manifest-sha", required=True)
    parser.add_argument("--training-contract-sha", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    args = parser.parse_args()
    global REPO, DERIVED, DERIVED_SHA, BRIDGE_MANIFEST_PATH, POLICY_PATH, POLICY_SHA, POLICY_SIZE
    global POLICY_AUTHORITY_PATH, POLICY_AUTHORITY_ID, POLICY_AUTHORITY_IDENTITY_SHA
    global AUTH_PATH, AUTH_ID, AUTH_SHA, EXEC_SOURCE_SHA, EXEC_CONTRACT_SHA, EXEC_CONTRACT_PATH
    global EXECUTION_SOURCE_PATH, REVISION_VERIFIER_PATH, STRUCTURED_INVOCATION_PATH, FROZEN_PLAN_SHA, EXPECTED_EXECUTION_CONTRACT_ID
    global TRAINING_CHILD_MODE, BRIDGE_BINDING_SHA, BRIDGE_MANIFEST_SHA, TRAINING_CONTRACT_SHA, GPU_UUID
    REPO, DERIVED, BRIDGE_MANIFEST_PATH = map(Path, (args.repo, args.derived, args.manifest))
    POLICY_PATH = None
    POLICY_SHA = ""
    POLICY_SIZE = None
    POLICY_AUTHORITY_PATH = Path(args.policy_authority)
    POLICY_AUTHORITY_ID = args.policy_authority_id
    POLICY_AUTHORITY_IDENTITY_SHA = args.policy_authority_identity_sha
    DERIVED_SHA = args.derived_sha
    AUTH_PATH, AUTH_ID, AUTH_SHA = Path(args.auth), args.auth_id, args.auth_sha
    EXEC_SOURCE_SHA, EXEC_CONTRACT_SHA = args.exec_source_sha, args.exec_contract_sha
    EXEC_CONTRACT_PATH = Path(args.execution_contract)
    EXECUTION_SOURCE_PATH = Path(args.execution_source) if args.execution_source else None
    REVISION_VERIFIER_PATH = Path(args.revision_verifier) if args.revision_verifier else None
    STRUCTURED_INVOCATION_PATH = Path(args.invocation).resolve(strict=True)
    FROZEN_PLAN_SHA = str(args.plan_sha)
    if len(FROZEN_PLAN_SHA) != 64 or any(ch not in "0123456789abcdef" for ch in FROZEN_PLAN_SHA):
        raise RuntimeLocatorError("STRUCTURED_INVOCATION_PLAN_SHA_INVALID")
    EXPECTED_EXECUTION_CONTRACT_ID = str(args.expected_execution_contract_id)
    TRAINING_CHILD_MODE = bool(args.training_child)
    BRIDGE_BINDING_SHA, BRIDGE_MANIFEST_SHA = args.bridge_binding_sha, args.bridge_manifest_sha
    TRAINING_CONTRACT_SHA, GPU_UUID = args.training_contract_sha, args.gpu_uuid
    auth_doc = load_canonical_json(AUTH_PATH)
    if auth_doc.get("smoke_authorization_id") != AUTH_ID or auth_doc.get("authorization_sha256") != AUTH_SHA:
        raise RuntimeLocatorError("smoke authorization identity mismatch")
    if auth_doc.get("consumed") is not False or auth_doc.get("formal_launch_permitted") is not False:
        raise RuntimeLocatorError("smoke authorization is not pending and fail-closed")
    if args.quiescence_probe:
        value = _quiescence_probe(Path(args.root))
        print(json.dumps(value, sort_keys=True, indent=2))
        return 0
    if args.training_child:
        value = run(Path(args.root))
        print(json.dumps(value, sort_keys=True, indent=2))
        return 0
    if args.roundtrip:
        value = _roundtrip(Path(args.checkpoint), str(args.checkpoint_sha), Path(args.root))
        print(json.dumps(value, sort_keys=True, indent=2))
        return 0
    if args.cpu_rehearsal:
        value = cpu_rehearsal(Path(args.root))
        print(json.dumps(value, sort_keys=True, indent=2))
        return 0
    value = run(Path(args.root))
    print(json.dumps(value, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
