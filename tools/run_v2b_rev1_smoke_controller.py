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

from sparse_rtdetr.baseline.training_v2b_smoke_controller import build_plan, launch


RUNTIME_KEYS = {"runtime_locator", "runtime_file_count", "runtime_locators", "state_runtime_locator"}


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
    if value.get("execution_contract_id") != "v2b-execution-contract-005":
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
    parser.add_argument("--policy", required=True)
    parser.add_argument("--policy-sha", required=True)
    parser.add_argument("--policy-size", type=int)
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
    expected_environment = startup_environment(contract, gpu_uuid=args.gpu_uuid, cpu_rehearsal=args.cpu_rehearsal)
    runner_argv = [
        args.runner,
        "--root", args.root,
        "--repo", args.repo,
        "--derived", args.derived,
        "--derived-sha", args.derived_sha,
        "--manifest", args.manifest,
        "--policy", args.policy,
        "--policy-sha", args.policy_sha,
        "--auth", args.auth,
        "--auth-id", args.auth_id,
        "--auth-sha", args.auth_sha,
        "--exec-source-sha", args.exec_source_sha,
        "--exec-contract-sha", args.exec_contract_sha,
        "--execution-contract", args.exec_contract,
        "--bridge-binding-sha", args.bridge_binding_sha,
        "--bridge-manifest-sha", args.bridge_manifest_sha,
        "--training-contract-sha", args.training_contract_sha,
        "--gpu-uuid", args.gpu_uuid,
    ]
    if args.policy_size is not None:
        runner_argv += ["--policy-size", str(args.policy_size)]
    runner_argv += ["--cpu-rehearsal" if args.cpu_rehearsal else ""]
    runner_argv = [item for item in runner_argv if item]
    environment = dict(expected_environment)
    environment.update({
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(Path(args.repo) / "src"),
    })
    plan = build_plan(
        launch_id="rev1-gpu-smoke-bridge007-cpu-rehearsal" if args.cpu_rehearsal else "rev1-gpu-smoke-bridge007",
        evidence_root=Path(args.root),
        repo_root=Path(args.repo),
        python_executable=Path(args.python_executable),
        runner_argv=runner_argv,
        tmux_executable=Path("tmux"),
        session_name=args.session,
        environment=environment,
        ready_timeout=300.0,
        exit_timeout=300.0,
    )
    result = launch(plan)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result.get("status") == "COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
