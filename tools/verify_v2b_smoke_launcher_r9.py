#!/usr/bin/env python3
"""Read-only verifier for the versioned REV1 smoke launch surface."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
from pathlib import Path
from typing import Any

RUNTIME_KEYS = {"runtime_locator", "runtime_file_count", "runtime_locators"}
SCIENTIFIC_SHA = "15f4ce16e373f08cf192b73adbb09b2de90f64ef640a5d8f3b1df44c586e92f8"
BRIDGE_SHA = "6523c4fcd7a9570e719813cad92ee1edb82afa717281ae096274cc28cf30c357"
BRIDGE_BINDING_SHA = "e93b26fd447b24f27c3374fddacf6cbd6c0dc1fb1c517fd87d3a8e13ad1380a5"
BRIDGE_MANIFEST_SHA = "df2e63abd63782d337866e008b1fbf17b8adfbbd4e90f9fee03d74abac054419"
TRAINING_SHA = "bafdcbfbd219de288c3947595902d47bf37031bfdc2e9b01d36cd9f1ee2177c4"
BRIDGE_GPU_UUID = "GPU-1faee6f0-1da7-4ede-2475-67a5a00274a8"
EXECUTION_SOURCE_ID = "v2b-exec-009"
EXECUTION_CONTRACT_ID = "v2b-execution-contract-009"


def canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def load(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if raw != canonical(value):
        raise ValueError(f"non-canonical JSON: {path}")
    if not isinstance(value, dict):
        raise ValueError(f"object required: {path}")
    return value


def without_runtime(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: without_runtime(item) for key, item in value.items() if key not in RUNTIME_KEYS}
    if isinstance(value, list):
        return [without_runtime(item) for item in value]
    return value


def regular(path: Path) -> None:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValueError(f"regular non-symlink required: {path}")


def versioned(contract_dir: Path, candidate: Path, prefix: str) -> Path:
    root = contract_dir.resolve(strict=True)
    raw = Path(candidate)
    if raw.is_symlink():
        raise ValueError(f"versioned contract symlink forbidden: {raw}")
    path = raw.resolve(strict=True)
    if path.parent != root or not path.is_file():
        raise ValueError(f"versioned contract must be under rev001: {path}")
    if not re.fullmatch(re.escape(prefix) + r"r[0-9]+\.json", path.name):
        raise ValueError(f"unexpected versioned contract path: {path.name}")
    return path


def verify(
    root: Path,
    contract_dir: Path,
    bridge: Path,
    bridge_manifest: Path,
    policy_authority: Path,
    execution_source: Path,
    execution_source_sha: str,
    execution_contract: Path,
    execution_contract_sha: str,
    authorization: Path,
    authorization_id: str,
    authorization_sha: str,
) -> dict[str, Any]:
    source_path = versioned(contract_dir, execution_source, "execution_source_")
    execution = load(source_path)
    if execution["execution_source_id"] != EXECUTION_SOURCE_ID:
        raise ValueError("unexpected execution source id")
    if execution["execution_source_sha256"] != execution_source_sha:
        raise ValueError("execution source SHA argument mismatch")
    if execution["scientific_source_excluded"] != SCIENTIFIC_SHA:
        raise ValueError("scientific source binding changed")
    body = {k: v for k, v in execution.items() if k != "execution_source_sha256"}
    if execution["execution_source_sha256"] != sha_bytes(canonical(without_runtime(body))):
        raise ValueError("execution source digest mismatch")
    for row in execution["files"]:
        logical = row["relative_path"]
        if "\\" in logical or ".." in Path(logical).parts or Path(logical).is_absolute():
            raise ValueError(f"unsafe execution path: {logical}")
        path = root.joinpath(*logical.split("/"))
        regular(path)
        raw = path.read_bytes()
        if len(raw) != row["size_bytes"] or sha_bytes(raw) != row["sha256"]:
            raise ValueError(f"execution source file drift: {logical}")

    contract_path = versioned(contract_dir, execution_contract, "execution_contract_")
    contract = load(contract_path)
    if contract["execution_contract_id"] != EXECUTION_CONTRACT_ID:
        raise ValueError("unexpected execution contract id")
    if contract["execution_contract_sha256"] != execution_contract_sha:
        raise ValueError("execution contract SHA argument mismatch")
    if contract["execution_source_sha256"] != execution["execution_source_sha256"]:
        raise ValueError("execution source binding mismatch")
    if contract["training_contract_sha256"] != TRAINING_SHA:
        raise ValueError("training contract binding mismatch")
    if contract["scientific_source_sha256"] != SCIENTIFIC_SHA:
        raise ValueError("scientific source binding mismatch")
    body = {k: v for k, v in contract.items() if k != "execution_contract_sha256"}
    if contract["execution_contract_sha256"] != sha_bytes(canonical(without_runtime(body))):
        raise ValueError("execution contract digest mismatch")
    startup = contract.get("startup_environment")
    if not isinstance(startup, dict):
        raise ValueError("startup environment authority missing")
    static = startup.get("static")
    gpu = startup.get("gpu")
    cpu = startup.get("cpu_rehearsal")
    expected_static = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "MKL_NUM_THREADS": "2", "MKL_THREADING_LAYER": "GNU", "OMP_NUM_THREADS": "2", "PYTHONHASHSEED": "0", "PYTHONNOUSERSITE": "1"}
    if static != expected_static or not isinstance(gpu, dict) or not isinstance(cpu, dict):
        raise ValueError("startup environment authority mismatch")
    if gpu.get("CUDA_VISIBLE_DEVICES") != BRIDGE_GPU_UUID or cpu.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("startup CUDA binding mismatch")

    auth_path = versioned(contract_dir, authorization, "gpu_smoke_authorization_")
    auth = load(auth_path)
    if auth["smoke_authorization_id"] != authorization_id:
        raise ValueError("authorization ID argument mismatch")
    if auth["authorization_sha256"] != authorization_sha:
        raise ValueError("authorization SHA argument mismatch")
    if auth["checkpoint_sha256"] != BRIDGE_SHA or auth["bridge_binding_sha256"] != BRIDGE_BINDING_SHA or auth["bridge_manifest_sha256"] != BRIDGE_MANIFEST_SHA:
        raise ValueError("bridge identity mismatch")
    if auth["execution_source_sha256"] != execution["execution_source_sha256"] or auth["execution_contract_sha256"] != contract["execution_contract_sha256"]:
        raise ValueError("authorization execution identity mismatch")
    if auth["consumed"] is not False or auth["formal_launch_permitted"] is not False or auth["status"] != "PENDING_NOT_EXECUTED":
        raise ValueError("authorization is not pending fail-closed")
    body = {k: v for k, v in auth.items() if k != "authorization_sha256"}
    if auth["authorization_sha256"] != sha_bytes(canonical(without_runtime(body))):
        raise ValueError("authorization digest mismatch")

    regular(bridge); regular(bridge_manifest); regular(policy_authority)
    if sha_file(bridge) != BRIDGE_SHA or sha_file(bridge_manifest) != BRIDGE_MANIFEST_SHA:
        raise ValueError("bridge artifact drift")
    policy_doc = load(policy_authority)
    identity = policy_doc["identity"]
    policy_path = Path(policy_doc["runtime_locator"]["path"])
    if policy_doc["authority_id"] != "hardware-policy:t8b:paired-smoke-r2":
        raise ValueError("policy authority ID mismatch")
    if policy_doc["identity_sha256"] != sha_bytes(canonical(identity)):
        raise ValueError("policy authority identity digest mismatch")
    if identity["sha256"] != policy_doc["runtime_locator"]["sha256"] or identity["size_bytes"] != policy_doc["runtime_locator"]["size_bytes"]:
        raise ValueError("policy authority self-binding mismatch")
    if not policy_path.is_file() or sha_file(policy_path) != identity["sha256"]:
        raise ValueError("policy authority runtime locator drift")
    return {
        "status": "PASS",
        "execution_source_id": execution["execution_source_id"],
        "execution_source_sha256": execution["execution_source_sha256"],
        "execution_contract_id": contract["execution_contract_id"],
        "execution_contract_sha256": contract["execution_contract_sha256"],
        "authorization_id": auth["smoke_authorization_id"],
        "authorization_sha256": auth["authorization_sha256"],
        "checkpoint_sha256": BRIDGE_SHA,
        "bridge_manifest_sha256": BRIDGE_MANIFEST_SHA,
        "policy_authority_id": policy_doc["authority_id"],
        "policy_locator": str(policy_path),
        "scientific_source_sha256": SCIENTIFIC_SHA,
        "startup_environment": startup,
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
    args = parser.parse_args()
    try:
        result = verify(
            args.repo_root.resolve(strict=True), args.contract_dir.resolve(strict=True),
            args.bridge.resolve(strict=True), args.bridge_manifest.resolve(strict=True),
            args.policy_authority.resolve(strict=True), args.execution_source,
            args.execution_source_sha, args.execution_contract, args.execution_contract_sha,
            args.authorization, args.authorization_id, args.authorization_sha,
        )
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
