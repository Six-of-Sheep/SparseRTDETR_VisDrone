#!/usr/bin/env python3
"""Read-only verifier for the REV1 sealed final-launch-plan execution revision."""
from __future__ import annotations

import argparse
import hashlib
import json
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


def verify(root: Path, contract_dir: Path, bridge: Path, bridge_manifest: Path, policy_authority: Path) -> dict[str, Any]:
    source_path = contract_dir / "execution_source_r6.json"
    execution = load(source_path)
    if execution["execution_source_id"] != "v2b-exec-006":
        raise ValueError("unexpected execution source id")
    if execution["scientific_source_excluded"] != SCIENTIFIC_SHA:
        raise ValueError("scientific source binding changed")
    if execution["execution_source_sha256"] != sha_bytes(canonical(without_runtime({k: v for k, v in execution.items() if k != "execution_source_sha256"}))):
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
    contract = load(contract_dir / "execution_contract_r6.json")
    if contract["execution_contract_id"] != "v2b-execution-contract-006":
        raise ValueError("unexpected execution contract id")
    if contract["execution_source_sha256"] != execution["execution_source_sha256"]:
        raise ValueError("execution source binding mismatch")
    if contract["training_contract_sha256"] != TRAINING_SHA:
        raise ValueError("training contract binding mismatch")
    if contract["scientific_source_sha256"] != SCIENTIFIC_SHA:
        raise ValueError("scientific source binding mismatch")
    if contract["execution_contract_sha256"] != sha_bytes(canonical(without_runtime({k: v for k, v in contract.items() if k != "execution_contract_sha256"}))):
        raise ValueError("execution contract digest mismatch")
    startup = contract.get("startup_environment")
    if not isinstance(startup, dict):
        raise ValueError("startup environment authority missing")
    static = startup.get("static")
    gpu = startup.get("gpu")
    cpu = startup.get("cpu_rehearsal")
    if not isinstance(static, dict) or not isinstance(gpu, dict) or not isinstance(cpu, dict):
        raise ValueError("startup environment authority schema invalid")
    expected_static = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "MKL_NUM_THREADS": "2", "MKL_THREADING_LAYER": "GNU", "OMP_NUM_THREADS": "2", "PYTHONHASHSEED": "0", "PYTHONNOUSERSITE": "1"}
    if static != expected_static:
        raise ValueError("startup static environment authority mismatch")
    if gpu.get("CUDA_VISIBLE_DEVICES") != BRIDGE_GPU_UUID:
        raise ValueError("GPU startup environment binding mismatch")
    if cpu.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU rehearsal startup environment binding mismatch")
    auth = load(contract_dir / "gpu_smoke_authorization_r8.json")
    if auth["smoke_authorization_id"] != "smoke-v2b-rev1-s2-r896-bridge-008":
        raise ValueError("unexpected bridge-007 id")
    if auth["checkpoint_sha256"] != BRIDGE_SHA or auth["bridge_binding_sha256"] != BRIDGE_BINDING_SHA or auth["bridge_manifest_sha256"] != BRIDGE_MANIFEST_SHA:
        raise ValueError("bridge identity mismatch")
    if auth["execution_source_sha256"] != execution["execution_source_sha256"] or auth["execution_contract_sha256"] != contract["execution_contract_sha256"]:
        raise ValueError("authorization execution identity mismatch")
    if auth["consumed"] is not False or auth["formal_launch_permitted"] is not False or auth["status"] != "PENDING_NOT_EXECUTED":
        raise ValueError("bridge-008 is not pending fail-closed")
    if auth["authorization_sha256"] != sha_bytes(canonical(without_runtime({k: v for k, v in auth.items() if k != "authorization_sha256"}))):
        raise ValueError("authorization digest mismatch")
    regular(bridge); regular(bridge_manifest); regular(policy_authority)
    if sha_file(bridge) != BRIDGE_SHA or sha_file(bridge_manifest) != BRIDGE_MANIFEST_SHA:
        raise ValueError("bridge artifact drift")
    policy_doc = load(policy_authority)
    identity = policy_doc["identity"]
    policy_path = Path(policy_doc["runtime_locator"]["path"])
    if identity["sha256"] != policy_doc["runtime_locator"]["sha256"]:
        raise ValueError("policy authority self-binding mismatch")
    if identity["size_bytes"] != policy_doc["runtime_locator"]["size_bytes"]:
        raise ValueError("policy authority size self-binding mismatch")
    if policy_doc["identity_sha256"] != sha_bytes(canonical(identity)):
        raise ValueError("policy authority identity digest mismatch")
    if not policy_path.is_file() or sha_file(policy_path) != identity["sha256"]:
        raise ValueError("policy authority runtime locator drift")
    result = {
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
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--bridge", type=Path, required=True)
    parser.add_argument("--bridge-manifest", type=Path, required=True)
    parser.add_argument("--policy-authority", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args.repo_root.resolve(strict=True), args.contract_dir.resolve(strict=True), args.bridge.resolve(strict=True), args.bridge_manifest.resolve(strict=True), args.policy_authority.resolve(strict=True)), sort_keys=True, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
