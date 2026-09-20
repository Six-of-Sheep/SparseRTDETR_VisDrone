#!/usr/bin/env python3
"""Independently verify an immutable R35 -> REV1 checkpoint bridge."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from bridge_v2b_checkpoint_revision import (
    PARENT_CAMPAIGN, PARENT_EPOCH, canonical_bytes, sha_bytes, state_digest,
)


def read_json(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if canonical_bytes(value) != raw or type(value) is not dict:
        raise ValueError(f"non-canonical JSON object: {path}")
    return value


def verify(parent_path: Path, parent_sha: str, derived_path: Path,
           derived_sha: str, manifest_path: Path, manifest_sha: str,
           contract_dir: Path) -> dict[str, Any]:
    import torch

    parent_raw = parent_path.read_bytes()
    derived_raw = derived_path.read_bytes()
    manifest_raw = manifest_path.read_bytes()
    if sha_bytes(parent_raw) != parent_sha:
        raise ValueError("parent SHA mismatch")
    if sha_bytes(derived_raw) != derived_sha:
        raise ValueError("derived SHA mismatch")
    if sha_bytes(manifest_raw) != manifest_sha:
        raise ValueError("bridge manifest SHA mismatch")
    parent = torch.load(parent_path, map_location="cpu", weights_only=True)
    derived = torch.load(derived_path, map_location="cpu", weights_only=True)
    if type(parent) is not dict or type(derived) is not dict:
        raise ValueError("checkpoint objects must be dicts")
    if set(parent) != set(derived):
        raise ValueError("checkpoint schema keys differ")
    if parent["format"] != "sparse_rtdetr.training_v2b_checkpoint" or parent["schema_version"] != 2:
        raise ValueError("parent checkpoint schema mismatch")
    allowed = {"binding", "binding_sha256"}
    changed = {
        key for key in parent
        if key not in allowed and state_digest(parent[key]) != state_digest(derived[key])
    }
    if changed:
        raise ValueError("derived checkpoint changed non-binding keys: " + repr(sorted(changed)))
    binding = derived["binding"]
    if derived["binding_sha256"] != binding.get("binding_sha256"):
        raise ValueError("derived binding digest field mismatch")
    body = {key: value for key, value in binding.items() if key != "binding_sha256"}
    if hashlib.sha256(canonical_bytes(body)).hexdigest() != binding["binding_sha256"]:
        raise ValueError("derived binding canonical digest mismatch")
    if "revision_bridge" in binding.get("config", {}):
        raise ValueError("bridge-v1 binding schema is superseded")
    provenance = binding.get("provenance")
    bridge = provenance.get("revision_bridge") if type(provenance) is dict else None
    if type(bridge) is not dict or bridge.get("kind") != "revision_bridge":
        raise ValueError("top-level revision_bridge provenance missing")
    if bridge.get("bridge_id") != "v2b-r35-e053-to-rev1-s2-r896-003":
        raise ValueError("unexpected bridge-v2 identity")
    if (bridge.get("parent_campaign") != PARENT_CAMPAIGN
            or bridge.get("parent_epoch") != PARENT_EPOCH
            or bridge.get("parent_checkpoint_sha256") != parent_sha
            or bridge.get("old_binding_sha256") != parent["binding_sha256"]):
        raise ValueError("parent bridge identity mismatch")
    if bridge.get("semantic_delta") != "ownership_boundary_hardening":
        raise ValueError("unexpected semantic delta")
    failure = bridge.get("preserved_failure", {})
    if (failure.get("preserved") is not True or failure.get("replayed") is not False
            or failure.get("reinterpreted") is not False):
        raise ValueError("historical failure preservation is invalid")
    engine = derived["engine"]
    if (engine.get("epoch") != 53 or engine.get("optimizer_updates") != 16112
            or engine.get("microsteps") != 32224 or engine.get("epoch_active") is not False):
        raise ValueError("derived resume boundary is not epoch53")
    manifest = read_json(manifest_path)
    if manifest.get("kind") != "revision_bridge":
        raise ValueError("manifest kind mismatch")
    if manifest.get("parent", {}).get("checkpoint_sha256") != parent_sha:
        raise ValueError("manifest parent checkpoint mismatch")
    if manifest.get("derived", {}).get("checkpoint_sha256") != derived_sha:
        raise ValueError("manifest derived checkpoint mismatch")
    if manifest.get("binding_transition", {}).get("new_binding_sha256") != binding["binding_sha256"]:
        raise ValueError("manifest binding transition mismatch")
    for name in (
        "scientific_source.json", "training_contract.json",
        "execution_source.json", "execution_contract.json", "lineage.json",
    ):
        if not (contract_dir / name).is_file():
            raise ValueError("missing REV1 contract: " + name)
    auth = read_json(contract_dir / "gpu_smoke_authorization_r3.json")
    if (auth.get("status") != "PENDING_NOT_EXECUTED"
            or auth.get("consumed") is not False
            or auth.get("formal_launch_permitted") is not False):
        raise ValueError("GPU smoke authorization is not fail-closed")
    scope = auth.get("scope", {})
    if (scope.get("gpu") is not True or scope.get("max_windows") != 1
            or scope.get("max_optimizer_updates") != 1
            or scope.get("formal_campaign") is not False):
        raise ValueError("GPU smoke scope is not minimal")
    if auth.get("checkpoint_sha256") != derived_sha:
        raise ValueError("GPU smoke authorization points at wrong checkpoint")
    return {
        "status": "PASS",
        "classification": "REV1_CHECKPOINT_BRIDGE_VERIFIED",
        "parent_checkpoint_sha256": parent_sha,
        "derived_checkpoint_sha256": derived_sha,
        "derived_binding_sha256": binding["binding_sha256"],
        "changed_payload_keys": sorted(allowed),
        "resume_boundary": "epoch54/window1",
        "gpu_smoke_authorization": auth["smoke_authorization_id"],
        "gpu_smoke_executed": False,
        "formal_continuation_executed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--parent-sha256", required=True)
    parser.add_argument("--derived-checkpoint", type=Path, required=True)
    parser.add_argument("--derived-sha256", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--contract-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(verify(
            args.parent_checkpoint.resolve(strict=True), args.parent_sha256,
            args.derived_checkpoint.resolve(strict=True), args.derived_sha256,
            args.manifest.resolve(strict=True), args.manifest_sha256,
            args.contract_dir.resolve(strict=True),
        ), sort_keys=True, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({
            "status": "FAIL",
            "classification": "REV1_CHECKPOINT_BRIDGE_VERIFICATION_FAILED",
            "error": f"{type(exc).__name__}: {exc}",
        }, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
