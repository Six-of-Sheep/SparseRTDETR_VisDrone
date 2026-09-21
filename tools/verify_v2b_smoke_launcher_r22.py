#!/usr/bin/env python3
"""Read-only verifier for the REV1 preflight evidence execution revision 022."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
from pathlib import Path
from typing import Any

RUNTIME_KEYS = {"runtime_locator", "runtime_file_count", "runtime_locators", "state_runtime_locator"}
SCIENTIFIC_SHA = "15f4ce16e373f08cf192b73adbb09b2de90f64ef640a5d8f3b1df44c586e92f8"
BRIDGE_SHA = "3d33e1ddc598e21180cb193665f785803ee26af3829162b8bcfc70c93858b4ab"
BRIDGE_BINDING_SHA = "211be3dc210c8d644e18465c8c50df6369b64160212f9fc2ac4d37a993f550e1"
BRIDGE_MANIFEST_SHA = "22921fb4ad27050e2415d7fc78d9dc286b09f4d7f36e0dc1250b49c91317166c"
TRAINING_SHA = "bafdcbfbd219de288c3947595902d47bf37031bfdc2e9b01d36cd9f1ee2177c4"
BRIDGE_GPU_UUID = "GPU-1faee6f0-1da7-4ede-2475-67a5a00274a8"
SOURCE_ID = "v2b-exec-022"
CONTRACT_ID = "v2b-execution-contract-022"


def canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if raw != canonical(value) or not isinstance(value, dict):
        raise ValueError(f"non-canonical object: {path}")
    return value


def without_runtime(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: without_runtime(v) for k, v in value.items() if k not in RUNTIME_KEYS}
    if isinstance(value, list):
        return [without_runtime(v) for v in value]
    return value


def regular(path: Path) -> None:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError(f"regular non-symlink required: {path}")


def versioned(root: Path, candidate: Path, prefix: str) -> Path:
    root = root.resolve(strict=True)
    raw = Path(candidate)
    if raw.is_symlink():
        raise ValueError("versioned symlink forbidden")
    path = raw.resolve(strict=True)
    if path.parent != root or not path.is_file() or not re.fullmatch(re.escape(prefix) + r"r[0-9]+\.json", path.name):
        raise ValueError(f"invalid versioned path: {path}")
    return path


def verify_invocation(path: Path, *, plan_sha: str, auth_id: str, auth_sha: str,
                      source_sha: str, contract_sha: str, repo: Path) -> dict[str, Any]:
    value = load(path)
    if value.get("schema_version") != 2:
        raise ValueError("invocation schema mismatch")
    if value.get("invocation_id") != auth_id:
        raise ValueError("invocation id mismatch")
    for key, expected in {
        "execution_source_sha256": source_sha,
        "execution_contract_sha256": contract_sha,
        "authorization_sha256": auth_sha,
        "plan_sha256": plan_sha,
    }.items():
        if value.get(key) != expected:
            raise ValueError(f"invocation binding mismatch: {key}")
    if Path(value.get("repository_root", "")).resolve() != repo or Path(value.get("working_directory", "")).resolve() != repo:
        raise ValueError("invocation repository/cwd mismatch")
    envs = {name: value.get(name) for name in ("controller_env", "tmux_env", "runner_env", "restore_env")}
    if any(not isinstance(env, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in env.items()) for env in envs.values()):
        raise ValueError("invocation environment schema mismatch")
    controller_env = envs["controller_env"]
    expected_src = str((repo / "src").resolve(strict=True))
    for name, expected in {
        "PYTHONPATH": expected_src,
        "PYTHONHASHSEED": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }.items():
        if controller_env.get(name) != expected:
            raise ValueError(f"controller environment mismatch: {name}")
    argv = value.get("controller_argv")
    interpreter = value.get("controller_python")
    launcher = value.get("controller_launcher")
    if not isinstance(argv, list) or argv[:2] != [interpreter, launcher]:
        raise ValueError("controller argv binding mismatch")
    try:
        invocation_index = argv.index("--structured-invocation")
    except ValueError as exc:
        raise ValueError("controller argv missing structured invocation") from exc
    if invocation_index + 1 >= len(argv) or Path(argv[invocation_index + 1]).resolve() != path.resolve():
        raise ValueError("controller structured invocation binding mismatch")
    if value.get("entrypoint_argv", [])[:2] != [interpreter, value.get("entrypoint")]:
        raise ValueError("entrypoint argv binding mismatch")
    return {
        "invocation_id": value["invocation_id"],
        "controller_env": controller_env,
        "controller_argv": argv,
        "entrypoint_argv": value["entrypoint_argv"],
    }


def verify(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo_root.resolve(strict=True)
    contract_dir = args.contract_dir.resolve(strict=True)
    source_path = versioned(contract_dir, args.execution_source, "execution_source_")
    source = load(source_path)
    if source.get("execution_source_id") != SOURCE_ID or source.get("execution_source_sha256") != args.execution_source_sha:
        raise ValueError("execution source identity mismatch")
    if source.get("scientific_source_excluded") != SCIENTIFIC_SHA:
        raise ValueError("scientific source binding changed")
    body = {k: v for k, v in source.items() if k != "execution_source_sha256"}
    if digest(without_runtime(body)) != args.execution_source_sha:
        raise ValueError("execution source digest mismatch")
    for row in source["files"]:
        path = repo.joinpath(*row["relative_path"].split("/"))
        regular(path)
        raw = path.read_bytes()
        if len(raw) != row["size_bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError(f"execution source file drift: {row['relative_path']}")
    contract_path = versioned(contract_dir, args.execution_contract, "execution_contract_")
    contract = load(contract_path)
    if contract.get("execution_contract_id") != CONTRACT_ID or contract.get("execution_contract_sha256") != args.execution_contract_sha:
        raise ValueError("execution contract identity mismatch")
    if contract.get("execution_source_sha256") != args.execution_source_sha or contract.get("scientific_source_sha256") != SCIENTIFIC_SHA:
        raise ValueError("execution contract source binding mismatch")
    if contract.get("training_contract_sha256") != TRAINING_SHA:
        raise ValueError("training contract binding mismatch")
    body = {k: v for k, v in contract.items() if k != "execution_contract_sha256"}
    if digest(without_runtime(body)) != args.execution_contract_sha:
        raise ValueError("execution contract digest mismatch")
    startup = contract.get("startup_environment") or {}
    expected_static = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "MKL_NUM_THREADS": "2", "MKL_THREADING_LAYER": "GNU", "OMP_NUM_THREADS": "2", "PYTHONHASHSEED": "0", "PYTHONNOUSERSITE": "1"}
    if startup.get("static") != expected_static or startup.get("gpu", {}).get("CUDA_VISIBLE_DEVICES") != BRIDGE_GPU_UUID or startup.get("cpu_rehearsal", {}).get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("startup environment authority mismatch")
    auth_path = versioned(contract_dir, args.authorization, "gpu_smoke_authorization_")
    auth = load(auth_path)
    if auth.get("smoke_authorization_id") != args.authorization_id or auth.get("authorization_sha256") != args.authorization_sha:
        raise ValueError("authorization identity mismatch")
    if auth.get("execution_source_sha256") != args.execution_source_sha or auth.get("execution_contract_sha256") != args.execution_contract_sha:
        raise ValueError("authorization execution binding mismatch")
    if auth.get("checkpoint_sha256") != BRIDGE_SHA or auth.get("bridge_binding_sha256") != BRIDGE_BINDING_SHA or auth.get("bridge_manifest_sha256") != BRIDGE_MANIFEST_SHA:
        raise ValueError("bridge identity mismatch")
    if auth.get("status") != "PENDING_NOT_EXECUTED" or auth.get("consumed") is not False or auth.get("formal_launch_permitted") is not False:
        raise ValueError("authorization not pending fail-closed")
    auth_body = {k: v for k, v in auth.items() if k != "authorization_sha256"}
    if digest(without_runtime(auth_body)) != args.authorization_sha:
        raise ValueError("authorization digest mismatch")
    bridge = args.bridge.resolve(strict=True)
    manifest = args.bridge_manifest.resolve(strict=True)
    if sha_file(bridge) != BRIDGE_SHA or sha_file(manifest) != BRIDGE_MANIFEST_SHA:
        raise ValueError("bridge artifact identity mismatch")
    policy = load(args.policy_authority.resolve(strict=True))
    if policy.get("authority_id") != "hardware-policy:t8b:paired-smoke-r2" or policy.get("identity", {}).get("sha256") != policy.get("runtime_locator", {}).get("sha256"):
        raise ValueError("policy authority mismatch")
    invocation = verify_invocation(args.invocation.resolve(strict=True), plan_sha=args.plan_sha, auth_id=args.authorization_id, auth_sha=args.authorization_sha, source_sha=args.execution_source_sha, contract_sha=args.execution_contract_sha, repo=repo)
    return {
        "status": "PASS",
        "execution_source_id": SOURCE_ID,
        "execution_source_sha256": args.execution_source_sha,
        "execution_contract_id": CONTRACT_ID,
        "execution_contract_sha256": args.execution_contract_sha,
        "authorization_id": args.authorization_id,
        "authorization_sha256": args.authorization_sha,
        "invocation": invocation,
        "scientific_source_sha256": SCIENTIFIC_SHA,
        "checkpoint_sha256": BRIDGE_SHA,
        "bridge_manifest_sha256": BRIDGE_MANIFEST_SHA,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--bridge-manifest", type=Path, required=True)
    parser.add_argument("--policy-authority", type=Path, required=True)
    parser.add_argument("--execution-source", type=Path, required=True)
    parser.add_argument("--execution-source-sha", required=True)
    parser.add_argument("--execution-contract", type=Path, required=True)
    parser.add_argument("--execution-contract-sha", required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--authorization-id", required=True)
    parser.add_argument("--authorization-sha", required=True)
    parser.add_argument("--invocation", type=Path, required=True)
    parser.add_argument("--plan-sha", required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args), sort_keys=True, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}:{exc}"}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
