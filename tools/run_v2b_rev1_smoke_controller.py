#!/usr/bin/env python3
"""REV1 smoke controller entry point with explicit runtime locators.

No physical locator is baked into this execution identity; all locators are
provided by the launch contract and are verified by the child before READY.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from sparse_rtdetr.baseline.training_v2b_smoke_controller import (
    build_plan, launch, validate_versioned_contract_path, probe_visible_gpu_identity,
)
from sparse_rtdetr.baseline.training_v2b_runtime_locator import (
    RuntimeLocatorError, resolve_bridge, resolve_policy_authority, load_canonical_json,
)


RUNTIME_KEYS = {"runtime_locator", "runtime_file_count", "runtime_locators", "state_runtime_locator"}
EXPECTED_EXECUTION_CONTRACT_ID = "v2b-execution-contract-009"


def canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def without_runtime(value: object) -> object:
    if isinstance(value, dict):
        return {key: without_runtime(item) for key, item in value.items() if key not in RUNTIME_KEYS}
    if isinstance(value, list):
        return [without_runtime(item) for item in value]
    return value


def load_execution_contract(path: Path, expected_sha: str, expected_source_sha: str) -> dict:
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if raw != canonical(value):
        raise RuntimeError("EXECUTION_CONTRACT_NON_CANONICAL")
    if value.get("execution_contract_sha256") != expected_sha:
        raise RuntimeError("EXECUTION_CONTRACT_IDENTITY_MISMATCH")
    if value.get("execution_contract_id") != EXPECTED_EXECUTION_CONTRACT_ID:
        raise RuntimeError("EXECUTION_CONTRACT_REVISION_MISMATCH")
    body = {key: item for key, item in value.items() if key != "execution_contract_sha256"}
    if value.get("execution_contract_sha256") != hashlib.sha256(canonical(without_runtime(body))).hexdigest():
        raise RuntimeError("EXECUTION_CONTRACT_DIGEST_MISMATCH")
    if value.get("execution_source_sha256") != expected_source_sha:
        raise RuntimeError("EXECUTION_SOURCE_BINDING_MISMATCH")
    return value


def startup_environment(contract: dict, *, gpu_uuid: str, cpu_rehearsal: bool) -> dict[str, str]:
    policy = contract.get("startup_environment")
    if not isinstance(policy, dict):
        raise RuntimeError("STARTUP_ENVIRONMENT_AUTHORITY_MISSING")
    static = policy.get("static")
    mode = policy.get("cpu_rehearsal" if cpu_rehearsal else "gpu")
    if not isinstance(static, dict) or not isinstance(mode, dict):
        raise RuntimeError("STARTUP_ENVIRONMENT_AUTHORITY_INVALID")
    result = {str(key): str(value) for key, value in static.items()}
    result.update({str(key): str(value) for key, value in mode.items()})
    if not cpu_rehearsal and result.get("CUDA_VISIBLE_DEVICES") != gpu_uuid:
        raise RuntimeError("STARTUP_ENVIRONMENT_GPU_BINDING_MISMATCH")
    if cpu_rehearsal and result.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("STARTUP_ENVIRONMENT_CPU_BINDING_MISMATCH")
    return result


def _arg_value(argv: list[str], name: str) -> str:
    try:
        index = argv.index(name)
    except ValueError as exc:
        raise RuntimeLocatorError(f"FINAL_LAUNCH_PLAN_MISMATCH: missing {name}") from exc
    if index + 1 >= len(argv):
        raise RuntimeLocatorError(f"FINAL_LAUNCH_PLAN_MISMATCH: missing value for {name}")
    return str(argv[index + 1])


def verify_final_launch_plan(plan: dict) -> str:
    """Verify the exact argv that will reach tmux before it is created."""
    objects = plan.get("runtime_objects")
    if not isinstance(objects, dict):
        raise RuntimeLocatorError("FINAL_LAUNCH_PLAN_MISMATCH: runtime objects missing")
    argv = [str(item) for item in plan.get("runner_argv", [])]
    forbidden = {"--policy", "--policy-sha", "--policy-size"}
    if forbidden.intersection(argv):
        raise RuntimeLocatorError("FINAL_LAUNCH_PLAN_MISMATCH: physical policy argv is forbidden")
    authority = objects.get("policy_authority") or {}
    policy = objects.get("policy") or {}
    expected = {
        "--policy-authority": authority.get("path"),
        "--policy-authority-id": authority.get("authority_id"),
        "--policy-authority-identity-sha": authority.get("identity_sha256"),
        "--root": plan.get("evidence_root"),
        "--derived": (objects.get("bridge_checkpoint") or {}).get("path"),
        "--manifest": (objects.get("bridge_manifest") or {}).get("path"),
        "--auth": (objects.get("authorization") or {}).get("path"),
        "--execution-contract": (objects.get("execution_contract") or {}).get("path"),
    }
    for name, expected_value in expected.items():
        if not expected_value or _arg_value(argv, name) != str(expected_value):
            raise RuntimeLocatorError(
                f"FINAL_LAUNCH_PLAN_MISMATCH: {name} does not match sealed plan"
            )
    resolved_policy = policy.get("resolved_path")
    if not resolved_policy or resolved_policy in argv:
        raise RuntimeLocatorError("FINAL_LAUNCH_PLAN_MISMATCH: resolved policy leaked as caller argv")
    digest_body = {key: value for key, value in plan.items() if key != "launch_plan_sha256"}
    digest = hashlib.sha256(canonical(digest_body)).hexdigest()
    if plan.get("launch_plan_sha256") != digest:
        raise RuntimeLocatorError("FINAL_LAUNCH_PLAN_MISMATCH: launch plan digest drift")
    return digest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--python", dest="python_executable", required=True)
    parser.add_argument("--runner", required=True)
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
    parser.add_argument("--exec-contract", required=True)
    parser.add_argument("--bridge-binding-sha", required=True)
    parser.add_argument("--bridge-manifest-sha", required=True)
    parser.add_argument("--training-contract-sha", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--cpu-rehearsal", action="store_true")
    args = parser.parse_args()
    contract = load_execution_contract(Path(args.exec_contract), args.exec_contract_sha, args.exec_source_sha)
    repo = Path(args.repo).resolve(strict=True)
    authority_path = Path(args.policy_authority).resolve(strict=True)
    expected_authority = (repo / "contracts" / "v2b" / "rev001" / "runtime_policy_authority_r1.json").resolve(strict=True)
    if authority_path != expected_authority:
        raise RuntimeLocatorError("POLICY_AUTHORITY_PATH_MISMATCH")
    resolved_policy = resolve_policy_authority(
        authority_path,
        authority_id=args.policy_authority_id,
        expected_identity_sha256=args.policy_authority_identity_sha,
    )
    resolve_bridge(
        Path(args.derived), checkpoint_sha256=args.derived_sha,
        manifest=Path(args.manifest), manifest_sha256=args.bridge_manifest_sha,
    )
    auth_path = validate_versioned_contract_path(repo, Path(args.auth), "gpu_smoke_authorization_")
    execution_path = validate_versioned_contract_path(repo, Path(args.exec_contract), "execution_contract_")
    expected_environment = startup_environment(contract, gpu_uuid=args.gpu_uuid, cpu_rehearsal=args.cpu_rehearsal)
    gpu_identity_preflight = None
    if not args.cpu_rehearsal:
        policy_doc = resolved_policy.get("policy") or {}
        hardware = policy_doc.get("hardware_policy") or {}
        expected_nvidia_smi = (
            (policy_doc.get("expected_nvidia_smi_executable") or {}).get("path")
            or "/usr/bin/nvidia-smi"
        )
        gpu_identity_preflight = probe_visible_gpu_identity(
            expected_gpu_uuid=args.gpu_uuid,
            expected_pci_bus_id=str(policy_doc.get("pci_bus_id") or ""),
            expected_gpu_name=str(hardware.get("expected_gpu_name") or ""),
            python_executable=Path(args.python_executable),
            nvidia_smi_executable=Path(expected_nvidia_smi),
            startup_environment=expected_environment,
            max_used_memory_mib=int(hardware.get("preflight_max_used_memory_mib", 1024)),
            min_free_memory_mib=int(hardware.get("preflight_min_free_memory_mib", 20_000)),
            max_utilization_percent=int(hardware.get("preflight_max_utilization_percent", 5)),
            max_temperature_c=int(hardware.get("gpu_stop_temperature_c", 80)),
            max_clock_mhz=int(hardware.get("graphics_clock_upper_mhz", 1500)),
        )
    runner_argv = [
        args.runner,
        "--root", args.root,
        "--repo", args.repo,
        "--derived", args.derived,
        "--derived-sha", args.derived_sha,
        "--manifest", args.manifest,
        "--auth", str(auth_path),
        "--auth-id", args.auth_id,
        "--auth-sha", args.auth_sha,
        "--exec-source-sha", args.exec_source_sha,
        "--exec-contract-sha", args.exec_contract_sha,
        "--execution-contract", str(execution_path),
        "--bridge-binding-sha", args.bridge_binding_sha,
        "--bridge-manifest-sha", args.bridge_manifest_sha,
        "--training-contract-sha", args.training_contract_sha,
        "--gpu-uuid", args.gpu_uuid,
        "--policy-authority", str(authority_path),
        "--policy-authority-id", args.policy_authority_id,
        "--policy-authority-identity-sha", args.policy_authority_identity_sha,
    ]
    if args.cpu_rehearsal:
        runner_argv += ["--cpu-rehearsal"]
    environment = dict(expected_environment)
    environment.update({"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(repo / "src")})
    runtime_objects = {
        "policy_authority": {"path": str(authority_path), "authority_id": args.policy_authority_id, "identity_sha256": args.policy_authority_identity_sha},
        "policy": {"resolved_path": resolved_policy["file"]["path"], "sha256": resolved_policy["file"]["sha256"], "size_bytes": resolved_policy["file"]["size_bytes"]},
        "bridge_checkpoint": {"path": str(Path(args.derived)), "sha256": args.derived_sha},
        "bridge_manifest": {"path": str(Path(args.manifest)), "sha256": args.bridge_manifest_sha},
        "authorization": {"path": str(auth_path), "id": args.auth_id, "sha256": args.auth_sha},
        "execution_contract": {"path": str(execution_path), "sha256": args.exec_contract_sha},
    }
    if gpu_identity_preflight is not None:
        runtime_objects["gpu_identity_preflight"] = gpu_identity_preflight
    plan = build_plan(
        launch_id=f"rev1-gpu-smoke-{args.auth_id}" + ("-cpu-rehearsal" if args.cpu_rehearsal else ""),
        evidence_root=Path(args.root), repo_root=repo,
        python_executable=Path(args.python_executable), runner_argv=runner_argv,
        tmux_executable=Path("tmux"), session_name=args.session,
        environment=environment, runtime_objects=runtime_objects,
        ready_timeout=300.0, exit_timeout=300.0,
    )
    plan["launch_plan_sha256"] = hashlib.sha256(canonical(plan)).hexdigest()
    verify_final_launch_plan(plan)
    result = launch(plan)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result.get("status") == "COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
