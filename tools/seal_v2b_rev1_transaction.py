#!/usr/bin/env python3
"""Create immutable external plan/auth/invocation artifacts after Git freeze."""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import re
from pathlib import Path
from typing import Any


AUTH_MARKER = "__BOUND_AUTHORIZATION_SHA256__"
RUNTIME_KEYS = {"runtime_locator", "runtime_file_count", "runtime_locators", "state_runtime_locator"}
TRANSACTION_ROOT_RE = re.compile(
    r"SparseRTDETR_VisDrone_v2b_rev1_gpu_smoke_bridge[0-9]{3}_[0-9]{8}"
)


def canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def without_runtime(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: without_runtime(item)
            for key, item in value.items()
            if key not in RUNTIME_KEYS
        }
    if isinstance(value, list):
        return [without_runtime(item) for item in value]
    return value


def declared_source_digest(source: dict[str, Any]) -> str:
    return digest({key: value for key, value in source.items() if key != "execution_source_sha256"})


def declared_contract_digest(contract: dict[str, Any]) -> str:
    body = {key: value for key, value in contract.items() if key != "execution_contract_sha256"}
    return digest(without_runtime(body))


def write_new(path: Path, value: Any) -> str:
    if path.exists() or path.is_symlink():
        raise ValueError(f"refusing to overwrite {path}")
    path.write_bytes(canonical(value))
    return digest(value)


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def strict_absolute(path: Path, *, label: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute() or ".." in candidate.parts or "." in candidate.parts:
        raise ValueError(f"{label} must be a canonical absolute path")
    current = Path(candidate.anchor)
    for component in candidate.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"{label} contains a symlink component")
    if candidate.resolve(strict=False) != candidate:
        raise ValueError(f"{label} path alias is forbidden")
    return candidate


def plan_identity(plan: dict[str, Any]) -> dict[str, Any]:
    body = {k: v for k, v in plan.items() if k not in {"authorization_sha256", "frozen_plan_sha256"}}
    for field in ("controller_argv", "exact_argv"):
        argv = list(body[field])
        argv[argv.index("--auth-sha") + 1] = AUTH_MARKER
        body[field] = argv
    body["exact_launch_command"] = shlex.join(body["exact_argv"])
    return body


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--plan-id", required=True)
    parser.add_argument("--authorization-id", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--entrypoint", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha", required=True)
    parser.add_argument("--contract-id", required=True)
    parser.add_argument("--revision-verifier", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha", required=True)
    parser.add_argument("--binding-sha", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha", required=True)
    parser.add_argument("--policy-authority", type=Path, required=True)
    parser.add_argument("--policy-id", required=True)
    parser.add_argument("--policy-sha", required=True)
    parser.add_argument("--training-contract-sha", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    args = parser.parse_args()
    repo = args.repo.resolve(strict=True)
    if git(repo, "status", "--porcelain") or git(repo, "rev-parse", "HEAD") != git(repo, "rev-parse", "@{upstream}"):
        raise ValueError("Git must be clean and HEAD must equal upstream before freeze")
    contract_path = args.contract.resolve(strict=True)
    if contract_path != repo / "contracts" / "v2b" / "rev001" / "execution_contract_r32.json":
        raise ValueError("unexpected execution contract path")
    contract = json.loads(contract_path.read_text())
    if contract.get("authorization_mode") != "external_transaction":
        raise ValueError("external transaction contract required")
    parent = strict_absolute(
        Path(str((contract.get("runtime_locator") or {}).get("external_transaction_parent", ""))),
        label="approved transaction parent",
    ).resolve(strict=True)
    root = strict_absolute(args.root, label="transaction root")
    if root.exists() or root.is_symlink() or root.parent != parent:
        raise ValueError("transaction root must be new and directly under approved parent")
    if TRANSACTION_ROOT_RE.fullmatch(root.name) is None:
        raise ValueError("invalid transaction root name")
    source = json.loads(args.source.read_text())
    if source.get("execution_source_id") != "v2b-exec-032":
        raise ValueError("unexpected execution source identity")
    if source.get("execution_source_sha256") != args.source_sha or declared_source_digest(source) != args.source_sha:
        raise ValueError("execution source digest mismatch")
    if contract.get("execution_contract_id") != args.contract_id:
        raise ValueError("execution contract identity mismatch")
    if contract.get("execution_contract_sha256") != args.contract_sha or declared_contract_digest(contract) != args.contract_sha:
        raise ValueError("execution contract digest mismatch")
    if contract.get("execution_source_sha256") != args.source_sha:
        raise ValueError("execution source binding mismatch")
    if args.plan_id != args.authorization_id:
        raise ValueError("plan and authorization identities must match")
    head = git(repo, "rev-parse", "HEAD")
    upstream = git(repo, "rev-parse", "@{upstream}")
    tree = git(repo, "rev-parse", "HEAD^{tree}")
    root.mkdir(mode=0o700)
    evidence = root / "evidence"
    output = root / "output"
    evidence.mkdir(mode=0o700)
    output.mkdir(mode=0o700)
    auth_path = root / "authorization.json"
    plan_path = root / "frozen_plan.json"
    invocation_path = root / "structured_invocation.json"
    controller_argv = [args.python, str(args.controller), "--root", str(evidence), "--session", args.session, "--repo", str(repo), "--python", args.python, "--runner", str(args.runner), "--derived", str(args.checkpoint), "--derived-sha", args.checkpoint_sha, "--manifest", str(args.manifest), "--policy-authority", str(args.policy_authority), "--policy-authority-id", args.policy_id, "--policy-authority-identity-sha", args.policy_sha, "--auth", str(auth_path), "--auth-id", args.authorization_id, "--auth-sha", AUTH_MARKER, "--exec-source-sha", args.source_sha, "--exec-contract-sha", args.contract_sha, "--exec-contract", str(args.contract), "--bridge-binding-sha", args.binding_sha, "--bridge-manifest-sha", args.manifest_sha, "--training-contract-sha", args.training_contract_sha, "--gpu-uuid", args.gpu_uuid, "--expected-execution-contract-id", args.contract_id, "--execution-source", str(args.source), "--revision-verifier", str(args.revision_verifier), "--structured-invocation", str(invocation_path)]
    plan = {"schema_version": 3, "plan_id": args.plan_id, "transaction_id": args.authorization_id, "authorization_id": args.authorization_id, "authorization_sha256": AUTH_MARKER, "frozen_plan_sha256": "", "execution_source_id": source["execution_source_id"], "execution_source_sha256": args.source_sha, "execution_contract_id": args.contract_id, "execution_contract_sha256": args.contract_sha, "checkpoint_sha256": args.checkpoint_sha, "bridge_binding_sha256": args.binding_sha, "bridge_manifest_sha256": args.manifest_sha, "training_contract_sha256": args.training_contract_sha, "repository_root": str(repo), "working_directory": str(repo), "git": {"commit": head, "upstream": upstream, "tree": tree, "worktree": "clean"}, "authorization_path": str(auth_path), "structured_invocation": str(invocation_path), "external_evidence_root": str(evidence), "local_output_root": str(output), "controller_python": args.python, "controller_launcher": str(args.controller), "entrypoint": str(args.entrypoint), "runner": str(args.runner), "revision_verifier": str(args.revision_verifier), "tmux_session_name": args.session, "controller_argv": controller_argv, "exact_argv": controller_argv, "exact_launch_command": shlex.join(controller_argv), "tmux_argv": ["tmux", "new-session", "-d", "-s", args.session, "-c", str(repo), str(root / "controller" / "runner-wrapper.sh")], "policy_authority_id": args.policy_id, "policy_authority_identity_sha": args.policy_sha, "resolved_checkpoint_locator": str(args.checkpoint), "resolved_manifest_locator": str(args.manifest), "execution_source_path": str(args.source), "execution_contract_path": str(args.contract), "cuda_mapping": {"cuda_visible_devices": args.gpu_uuid}, "startup_environment": {"static": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": "0", "PYTHONNOUSERSITE": "1", "MKL_NUM_THREADS": "2", "MKL_THREADING_LAYER": "GNU", "OMP_NUM_THREADS": "2"}, "gpu": {"CUDA_VISIBLE_DEVICES": args.gpu_uuid}}, "runtime_locator": {"transaction_root": str(root), "authorization": str(auth_path), "plan": str(plan_path), "invocation": str(invocation_path)}}
    plan["frozen_plan_sha256"] = digest(plan_identity(plan))
    auth = {"schema_version": 3, "authorization_mode": "external_transaction", "smoke_authorization_id": args.authorization_id, "transaction_id": args.authorization_id, "status": "PENDING_NOT_EXECUTED", "consumed": False, "formal_launch_permitted": False, "frozen_plan_sha256": plan["frozen_plan_sha256"], "structured_invocation_id": args.authorization_id, "execution_source_id": source["execution_source_id"], "execution_source_sha256": args.source_sha, "execution_contract_id": args.contract_id, "execution_contract_sha256": args.contract_sha, "checkpoint_sha256": args.checkpoint_sha, "bridge_binding_sha256": args.binding_sha, "bridge_manifest_sha256": args.manifest_sha, "repository_commit": head, "upstream_commit": upstream, "repository_tree": tree, "policy_authority_id": args.policy_id, "policy_authority_identity_sha": args.policy_sha, "runtime_locator": {"transaction_root": str(root), "authorization": str(auth_path)}, "reason": "external immutable transaction authorization; no GPU smoke or formal continuation launched"}
    auth["authorization_sha256"] = digest({k: v for k, v in auth.items() if k != "authorization_sha256"})
    plan["authorization_sha256"] = auth["authorization_sha256"]
    actual_controller_argv = [
        auth["authorization_sha256"] if item == AUTH_MARKER else item
        for item in controller_argv
    ]
    invocation = {"schema_version": 3, "kind": "structured_smoke_invocation_v2", "invocation_id": args.authorization_id, "authorization_id": args.authorization_id, "authorization_sha256": auth["authorization_sha256"], "plan_sha256": plan["frozen_plan_sha256"], "execution_source_id": source["execution_source_id"], "execution_source_sha256": args.source_sha, "execution_contract_id": args.contract_id, "execution_contract_sha256": args.contract_sha, "repository_root": str(repo), "working_directory": str(repo), "controller_python": args.python, "controller_launcher": str(args.controller), "controller_argv": actual_controller_argv, "entrypoint": str(args.entrypoint), "entrypoint_argv": [args.python, str(args.entrypoint), "--invocation", str(invocation_path)], "transaction_root": str(root), "authorization_path": str(auth_path), "plan_path": str(plan_path), "evidence_root": str(evidence), "output_root": str(output), "controller_env": {"PYTHONPATH": str(repo / "src"), "PYTHONHASHSEED": "0", "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONNOUSERSITE": "1", "CUDA_VISIBLE_DEVICES": args.gpu_uuid}, "runner_env": {"PYTHONPATH": str(repo / "src"), "PYTHONHASHSEED": "0", "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONNOUSERSITE": "1", "CUDA_VISIBLE_DEVICES": args.gpu_uuid}}
    write_new(plan_path, plan)
    write_new(auth_path, auth)
    write_new(invocation_path, invocation)
    if git(repo, "rev-parse", "HEAD") != head or git(repo, "rev-parse", "@{upstream}") != upstream or git(repo, "rev-parse", "HEAD^{tree}") != tree or git(repo, "status", "--porcelain"):
        raise ValueError("TRANSACTION_CREATION_MUTATED_TRACKED_REPOSITORY")
    print(json.dumps({"status": "REV1_EXTERNAL_TRANSACTION_SEALED", "root": str(root), "plan": str(plan_path), "authorization": str(auth_path), "invocation": str(invocation_path), "plan_sha256": plan["frozen_plan_sha256"], "authorization_sha256": auth["authorization_sha256"], "invocation_sha256": sha_file(invocation_path), "git": {"commit": head, "upstream": upstream, "tree": tree}}, sort_keys=True, indent=2))
    return 0


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
