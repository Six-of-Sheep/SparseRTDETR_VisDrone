#!/usr/bin/env python3
"""Fail-closed verifier for REV1 external transaction artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

from sparse_rtdetr.baseline.training_v2b_runtime_locator import (
    RuntimeLocatorError,
    validate_external_transaction_authorization,
)

RUNTIME_KEYS = {"runtime_locator", "runtime_file_count", "runtime_locators", "state_runtime_locator"}
AUTH_MARKER = "__BOUND_AUTHORIZATION_SHA256__"
SHA_RE = re.compile(r"[0-9a-f]{64}")


def canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def load(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError(f"symlink is forbidden: {path}")
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if raw != canonical(value) or not isinstance(value, dict):
        raise ValueError(f"non-canonical JSON object: {path}")
    return value


def without_runtime(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: without_runtime(v) for k, v in value.items() if k not in RUNTIME_KEYS}
    if isinstance(value, list):
        return [without_runtime(v) for v in value]
    return value


def source_digest(source: dict[str, Any]) -> str:
    body = {k: v for k, v in source.items() if k != "execution_source_sha256"}
    return sha_bytes(canonical(without_runtime(body)))


def contract_digest(contract: dict[str, Any]) -> str:
    body = {k: v for k, v in contract.items() if k != "execution_contract_sha256"}
    return sha_bytes(canonical(without_runtime(body)))


def plan_identity(plan: dict[str, Any]) -> dict[str, Any]:
    body = {k: v for k, v in plan.items() if k not in {"authorization_sha256", "frozen_plan_sha256"}}
    for field in ("controller_argv", "exact_argv"):
        argv = list(body[field])
        index = argv.index("--auth-sha") + 1
        argv[index] = AUTH_MARKER
        body[field] = argv
    body["exact_launch_command"] = shlex.join(body["exact_argv"])
    return body


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def verify(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo_root.resolve(strict=True)
    contract = load(args.execution_contract.resolve(strict=True))
    source = load(args.execution_source.resolve(strict=True))
    invocation = load(args.invocation.resolve(strict=True))
    plan_path = args.plan.resolve(strict=True) if args.plan is not None else Path(str(invocation.get("plan_path", ""))).resolve(strict=True)
    plan = load(plan_path)
    plan_sha = args.plan_sha or str(invocation.get("plan_sha256", ""))
    if len(plan_sha) != 64 or any(ch not in "0123456789abcdef" for ch in plan_sha):
        raise ValueError("plan SHA is invalid")
    if invocation.get("plan_sha256") != plan_sha:
        raise ValueError("invocation plan SHA mismatch")
    auth = load(args.authorization.resolve(strict=True))
    if contract.get("execution_contract_id") != "v2b-execution-contract-032":
        raise ValueError("execution contract revision mismatch")
    if contract.get("authorization_mode") != "external_transaction":
        raise ValueError("external authorization mode missing")
    if source.get("execution_source_id") != "v2b-exec-032":
        raise ValueError("execution source revision mismatch")
    if source.get("execution_source_sha256") != args.execution_source_sha or source_digest(source) != args.execution_source_sha:
        raise ValueError("execution source digest mismatch")
    if contract.get("execution_contract_sha256") != args.execution_contract_sha or contract_digest(contract) != args.execution_contract_sha:
        raise ValueError("execution contract digest mismatch")
    if contract.get("execution_source_sha256") != args.execution_source_sha:
        raise ValueError("execution source binding mismatch")
    if plan.get("frozen_plan_sha256") != plan_sha or sha_bytes(canonical(plan_identity(plan))) != plan_sha:
        raise ValueError("frozen plan digest mismatch")
    if invocation.get("plan_sha256") != plan_sha:
        raise ValueError("invocation plan binding mismatch")
    if invocation.get("authorization_sha256") != args.authorization_sha:
        raise ValueError("invocation authorization binding mismatch")
    if auth.get("authorization_sha256") != args.authorization_sha:
        raise ValueError("authorization SHA mismatch")
    if auth.get("authorization_sha256") != sha_bytes(canonical({k: v for k, v in auth.items() if k != "authorization_sha256"})):
        raise ValueError("authorization digest mismatch")
    if auth.get("frozen_plan_sha256") != plan_sha or plan.get("frozen_plan_sha256") != plan_sha:
        raise ValueError("authorization frozen-plan binding mismatch")
    if auth.get("smoke_authorization_id") != args.authorization_id:
        raise ValueError("authorization ID mismatch")
    if auth.get("status") != "PENDING_NOT_EXECUTED" or auth.get("consumed") is not False or auth.get("formal_launch_permitted") is not False:
        raise ValueError("authorization is not pending fail-closed")
    if auth.get("execution_source_sha256") != args.execution_source_sha or auth.get("execution_contract_sha256") != args.execution_contract_sha:
        raise ValueError("authorization execution binding mismatch")
    if plan.get("authorization_sha256") != args.authorization_sha or plan.get("execution_source_sha256") != args.execution_source_sha or plan.get("execution_contract_sha256") != args.execution_contract_sha:
        raise ValueError("plan execution binding mismatch")
    if plan.get("plan_id") != args.authorization_id or plan.get("transaction_id") != args.authorization_id or plan.get("authorization_id") != args.authorization_id:
        raise ValueError("plan transaction identity mismatch")
    if plan.get("repository_root") != str(repo) or plan.get("working_directory") != str(repo):
        raise ValueError("plan repository/cwd mismatch")
    if plan.get("git", {}).get("commit") != git(repo, "rev-parse", "HEAD") or plan["git"].get("upstream") != git(repo, "rev-parse", "@{upstream}") or plan["git"].get("tree") != git(repo, "rev-parse", "HEAD^{tree}"):
        raise ValueError("plan Git identity mismatch")
    if git(repo, "status", "--porcelain"):
        raise ValueError("worktree dirty")
    evidence = Path(plan["external_evidence_root"]).resolve(strict=False)
    auth_path = validate_external_transaction_authorization(
        args.authorization, evidence_root=evidence, execution_contract=contract,
    )
    if auth_path != args.authorization.resolve():
        raise ValueError("authorization locator mismatch")
    if invocation.get("invocation_id") != args.authorization_id or auth.get("structured_invocation_id") != invocation.get("invocation_id"):
        raise ValueError("invocation identity mismatch")
    if invocation.get("schema_version") != 3 or invocation.get("kind") != "structured_smoke_invocation_v2":
        raise ValueError("structured invocation schema mismatch")
    for key, expected in (
        ("invocation_id", args.authorization_id),
        ("authorization_id", args.authorization_id),
        ("execution_source_id", source["execution_source_id"]),
        ("execution_source_sha256", args.execution_source_sha),
        ("execution_contract_id", contract["execution_contract_id"]),
        ("execution_contract_sha256", args.execution_contract_sha),
        ("repository_root", str(repo)),
        ("working_directory", str(repo)),
        ("authorization_path", str(args.authorization.resolve())),
        ("plan_path", str(plan_path)),
        ("evidence_root", str(evidence)),
    ):
        if invocation.get(key) != expected:
            raise ValueError(f"structured invocation binding mismatch: {key}")
    expected_argv = list(plan["controller_argv"])
    expected_argv[expected_argv.index("--auth-sha") + 1] = args.authorization_sha
    if invocation.get("controller_argv") != expected_argv:
        raise ValueError("structured invocation controller argv mismatch")
    for key, expected in (
        ("repository_commit", plan["git"]["commit"]),
        ("upstream_commit", plan["git"]["upstream"]),
        ("repository_tree", plan["git"]["tree"]),
    ):
        if auth.get(key) != expected:
            raise ValueError(f"authorization Git binding mismatch: {key}")
    for key, expected in (("checkpoint_sha256", args.checkpoint_sha), ("bridge_binding_sha256", args.binding_sha), ("bridge_manifest_sha256", args.manifest_sha)):
        if plan.get(key) != expected or auth.get(key) != expected:
            raise ValueError(f"{key} mismatch")
    if sha_file(args.bridge) != args.checkpoint_sha or sha_file(args.bridge_manifest) != args.manifest_sha:
        raise ValueError("checkpoint or manifest bytes mismatch")
    return {"status": "PASS", "mode": "external_transaction", "plan_sha256": plan_sha, "authorization_sha256": args.authorization_sha, "git": plan["git"], "authorization_path": str(auth_path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=False)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--invocation", type=Path, required=True)
    parser.add_argument("--authorization-id", required=True)
    parser.add_argument("--authorization-sha", required=True)
    parser.add_argument("--plan-sha", required=False)
    parser.add_argument("--execution-source", type=Path, required=True)
    parser.add_argument("--execution-source-sha", required=True)
    parser.add_argument("--execution-contract", type=Path, required=True)
    parser.add_argument("--execution-contract-sha", required=True)
    parser.add_argument("--checkpoint", "--bridge", dest="bridge", type=Path, required=True)
    parser.add_argument("--checkpoint-sha", dest="checkpoint_sha", required=True)
    parser.add_argument("--binding-sha", required=True)
    parser.add_argument("--manifest", "--bridge-manifest", dest="bridge_manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha", dest="manifest_sha", required=True)
    parser.add_argument("--contract-dir", type=Path, required=False)
    parser.add_argument("--policy-authority", type=Path, required=False)
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args), sort_keys=True, indent=2))
        return 0
    except (OSError, ValueError, RuntimeLocatorError, subprocess.CalledProcessError) as exc:
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
