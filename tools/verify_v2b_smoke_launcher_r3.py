#!/usr/bin/env python3
"""Verify the REV1 directory-ownership smoke execution identity."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def identity_sha(value: dict[str, Any], field: str) -> str:
    body = dict(value)
    body.pop(field, None)
    return hashlib.sha256(canonical_bytes(body)).hexdigest()


def file_sha(path: Path) -> tuple[str, int]:
    raw = path.read_bytes()
    return hashlib.sha256(raw).hexdigest(), len(raw)


def verify(repo_root: Path) -> dict[str, Any]:
    contract_root = repo_root / "contracts/v2b/rev001"
    source = json.loads((contract_root / "execution_source_r3.json").read_text(encoding="utf-8"))
    execution = json.loads((contract_root / "execution_contract_r3.json").read_text(encoding="utf-8"))
    auth = json.loads((contract_root / "gpu_smoke_authorization_r5.json").read_text(encoding="utf-8"))
    errors: list[str] = []
    if source.get("execution_source_id") != "v2b-exec-003":
        errors.append("execution source id mismatch")
    if identity_sha(source, "execution_source_sha256") != source.get("execution_source_sha256"):
        errors.append("execution source canonical SHA mismatch")
    for item in source.get("files", []):
        path = repo_root / item["relative_path"]
        if not path.is_file():
            errors.append("missing execution source: " + item["relative_path"])
            continue
        digest, size = file_sha(path)
        if digest != item.get("sha256") or size != item.get("size_bytes"):
            errors.append("execution source identity mismatch: " + item["relative_path"])
    if execution.get("execution_contract_id") != "v2b-execution-contract-003":
        errors.append("execution contract id mismatch")
    if identity_sha(execution, "execution_contract_sha256") != execution.get("execution_contract_sha256"):
        errors.append("execution contract canonical SHA mismatch")
    if execution.get("execution_source_sha256") != source.get("execution_source_sha256"):
        errors.append("execution source binding mismatch")
    expected = {
        "smoke_authorization_id": "smoke-v2b-rev1-s2-r896-bridge-005",
        "checkpoint_sha256": "6523c4fcd7a9570e719813cad92ee1edb82afa717281ae096274cc28cf30c357",
        "bridge_binding_sha256": "e93b26fd447b24f27c3374fddacf6cbd6c0dc1fb1c517fd87d3a8e13ad1380a5",
        "bridge_manifest_sha256": "df2e63abd63782d337866e008b1fbf17b8adfbbd4e90f9fee03d74abac054419",
        "training_contract_sha256": "bafdcbfbd219de288c3947595902d47bf37031bfdc2e9b01d36cd9f1ee2177c4",
    }
    if identity_sha(auth, "authorization_sha256") != auth.get("authorization_sha256"):
        errors.append("authorization canonical SHA mismatch")
    if auth.get("execution_source_sha256") != source.get("execution_source_sha256"):
        errors.append("authorization execution source binding mismatch")
    if auth.get("execution_contract_sha256") != execution.get("execution_contract_sha256"):
        errors.append("authorization execution contract binding mismatch")
    for key, value in expected.items():
        if auth.get(key) != value:
            errors.append("authorization " + key + " mismatch")
    if auth.get("scope") != {
        "formal_campaign": False, "gpu_training": True,
        "max_microsteps": 2, "max_optimizer_updates": 1, "max_windows": 1,
    }:
        errors.append("authorization scope mismatch")
    if auth.get("consumed") is not False or auth.get("formal_launch_permitted") is not False:
        errors.append("authorization is not pending")
    handshake = source.get("handshake", {})
    if handshake.get("layout_protocol") != "controller-created-launch-root-runner-created-child-workspace":
        errors.append("directory ownership handshake mismatch")
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "execution_source_id": source.get("execution_source_id"),
        "execution_source_sha256": source.get("execution_source_sha256"),
        "execution_contract_id": execution.get("execution_contract_id"),
        "execution_contract_sha256": execution.get("execution_contract_sha256"),
        "authorization_sha256": auth.get("authorization_sha256"),
        "smoke_authorization_id": auth.get("smoke_authorization_id"),
        "scientific_source_sha256": source.get("scientific_source_excluded"),
    }


def main(argv: list[str] | None = None) -> int:
    root = Path(argv[0]) if argv else Path.cwd()
    try:
        report = verify(root.resolve(strict=True))
    except Exception as exc:
        report = {"status": "FAIL", "errors": [type(exc).__name__ + ": " + str(exc)]}
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
