"""Local smoke tests for the runtime locator implementation.

This file is copied to the remote repository as tests/test_v2b_runtime_locator.py.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sparse_rtdetr.baseline.training_v2b_runtime_locator import (
    RuntimeLocatorError, resolve_bridge, resolve_file, resolve_policy, resolve_policy_authority,
    validate_external_transaction_authorization,
)


def _write(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def test_correct_locator_and_mount_change(tmp_path: Path) -> None:
    first = tmp_path / "mount-a" / "policy.json"
    raw = b'{"gpu_uuid":"GPU-X"}\n'
    digest = _write(first, raw)
    assert resolve_file(first, expected_sha256=digest, expected_size_bytes=len(raw))["sha256"] == digest
    second = tmp_path / "mount-b" / "policy.json"
    _write(second, raw)
    assert resolve_file(second, expected_sha256=digest, expected_size_bytes=len(raw))["sha256"] == digest


@pytest.mark.parametrize("kind", ["missing", "wrong_sha", "same_name_wrong_content", "traversal", "symlink"])
def test_locator_fail_closed(tmp_path: Path, kind: str) -> None:
    target = tmp_path / "real.json"
    digest = _write(target, b"good")
    if kind == "missing":
        candidate = tmp_path / "missing.json"
        with pytest.raises(RuntimeLocatorError, match="RUNTIME_LOCATOR_UNAVAILABLE"):
            resolve_file(candidate, expected_sha256=digest)
    elif kind == "wrong_sha":
        with pytest.raises(RuntimeLocatorError, match="sha256 mismatch"):
            resolve_file(target, expected_sha256="0" * 64)
    elif kind == "same_name_wrong_content":
        candidate = tmp_path / "other" / "real.json"
        _write(candidate, b"bad")
        with pytest.raises(RuntimeLocatorError, match="sha256 mismatch"):
            resolve_file(candidate, expected_sha256=digest)
    elif kind == "traversal":
        with pytest.raises(RuntimeLocatorError, match="locator escapes root"):
            resolve_file(tmp_path / ".." / "real.json", expected_sha256=digest, root=tmp_path)
    else:
        link = tmp_path / "link.json"
        try:
            link.symlink_to(target)
        except (OSError, NotImplementedError):
            pytest.skip("symlink unavailable")
        with pytest.raises(RuntimeLocatorError, match="symlink"):
            resolve_file(link, expected_sha256=digest)


def test_policy_size_and_json_identity(tmp_path: Path) -> None:
    policy = tmp_path / "policy.json"
    raw = (json.dumps({"gpu_uuid": "GPU-X"}, sort_keys=True, separators=(",", ":")) + "\n").encode()
    digest = _write(policy, raw)
    result = resolve_policy(policy, expected_sha256=digest, expected_size_bytes=len(raw), expected_gpu_uuid="GPU-X")
    assert result["policy"]["gpu_uuid"] == "GPU-X"


def test_policy_missing_and_wrong_identity_fail_closed(tmp_path: Path) -> None:
    policy = tmp_path / "policy.json"
    raw = (json.dumps({"gpu_uuid": "GPU-X"}, sort_keys=True, separators=(",", ":")) + "\n").encode()
    digest = _write(policy, raw)
    with pytest.raises(RuntimeLocatorError, match="sha256 mismatch"):
        resolve_policy(policy, expected_sha256="0" * 64, expected_size_bytes=len(raw))
    with pytest.raises(RuntimeLocatorError, match="missing path"):
        resolve_policy(tmp_path / "missing-policy.json", expected_sha256=digest)


def test_bridge_wrong_artifact_and_manifest_fail_closed(tmp_path: Path) -> None:
    bridge = tmp_path / "bridge.pt"
    manifest = tmp_path / "bridge.json"
    bridge_sha = _write(bridge, b"bridge")
    manifest_raw = b'{"checkpoint_sha256":"' + bridge_sha.encode() + b'"}\n'
    manifest_sha = _write(manifest, manifest_raw)
    with pytest.raises(RuntimeLocatorError, match="sha256 mismatch"):
        resolve_bridge(bridge, checkpoint_sha256="0" * 64, manifest=manifest, manifest_sha256=manifest_sha)
    with pytest.raises(RuntimeLocatorError, match="sha256 mismatch"):
        resolve_bridge(bridge, checkpoint_sha256=bridge_sha, manifest=manifest, manifest_sha256="0" * 64)


def _external_transaction(tmp_path: Path) -> tuple[Path, Path, dict]:
    parent = tmp_path / "artifacts"
    root = parent / "SparseRTDETR_VisDrone_v2b_rev1_gpu_smoke_bridge035_20260922"
    evidence = root / "evidence"
    evidence.mkdir(parents=True)
    auth = root / "authorization.json"
    auth.write_bytes(b"{}\n")
    contract = {
        "authorization_mode": "external_transaction",
        "runtime_locator": {"external_transaction_parent": str(parent)},
    }
    return auth, evidence, contract


def test_external_transaction_authorization_is_root_scoped(tmp_path: Path) -> None:
    auth, evidence, contract = _external_transaction(tmp_path)
    assert validate_external_transaction_authorization(
        auth, evidence_root=evidence, execution_contract=contract,
    ) == auth


@pytest.mark.parametrize("kind", ["wrong_root", "wrong_name", "symlink"])
def test_external_transaction_authorization_fails_closed(tmp_path: Path, kind: str) -> None:
    auth, evidence, contract = _external_transaction(tmp_path)
    if kind == "wrong_root":
        candidate = tmp_path / "outside" / "authorization.json"
        candidate.parent.mkdir()
        candidate.write_bytes(b"{}\n")
    elif kind == "wrong_name":
        candidate = auth.parent / "other.json"
        candidate.write_bytes(b"{}\n")
    else:
        candidate = auth.parent / "link.json"
        candidate.symlink_to(auth)
    with pytest.raises(RuntimeLocatorError):
        validate_external_transaction_authorization(
            candidate, evidence_root=evidence, execution_contract=contract,
        )


def _write_authority(tmp_path: Path, policy_path: Path, raw: bytes) -> tuple[Path, str, str]:
    digest = _write(policy_path, raw)
    identity = {"gpu_uuid": "GPU-X", "sha256": digest, "size_bytes": len(raw)}
    canonical = (json.dumps(identity, sort_keys=True, separators=(",", ":")) + "\n").encode()
    identity_sha = hashlib.sha256(canonical).hexdigest()
    authority = {
        "authority_id": "hardware-policy:test",
        "identity": identity,
        "identity_sha256": identity_sha,
        "runtime_locator": {"path": str(policy_path), "sha256": digest, "size_bytes": len(raw)},
        "schema_version": 1,
    }
    authority_path = tmp_path / "authority.json"
    authority_path.write_bytes((json.dumps(authority, sort_keys=True, separators=(",", ":")) + "\n").encode())
    return authority_path, identity_sha, digest


def test_policy_authority_resolves_sealed_locator_and_mount_override(tmp_path: Path) -> None:
    raw = b'{"gpu_uuid":"GPU-X"}\n'
    authority_path, identity_sha, digest = _write_authority(tmp_path, tmp_path / "mount-a" / "policy.json", raw)
    result = resolve_policy_authority(authority_path, authority_id="hardware-policy:test", expected_identity_sha256=identity_sha)
    assert result["file"]["sha256"] == digest
    relocated = tmp_path / "mount-b" / "policy.json"
    relocated.parent.mkdir()
    relocated.write_bytes(raw)
    moved = resolve_policy_authority(authority_path, authority_id="hardware-policy:test", expected_identity_sha256=identity_sha, locator_override=relocated)
    assert moved["locator_source"] == "override"
    assert moved["file"]["path"] == str(relocated)


@pytest.mark.parametrize("kind", ["wrong_authority_sha", "wrong_content", "wrong_id"])
def test_policy_authority_fail_closed(tmp_path: Path, kind: str) -> None:
    raw = b'{"gpu_uuid":"GPU-X"}\n'
    authority_path, identity_sha, digest = _write_authority(tmp_path, tmp_path / "policy.json", raw)
    if kind == "wrong_authority_sha":
        with pytest.raises(RuntimeLocatorError, match="identity mismatch"):
            resolve_policy_authority(authority_path, authority_id="hardware-policy:test", expected_identity_sha256="0" * 64)
    elif kind == "wrong_id":
        with pytest.raises(RuntimeLocatorError, match="authority id mismatch"):
            resolve_policy_authority(authority_path, authority_id="hardware-policy:wrong", expected_identity_sha256=identity_sha)
    else:
        bad = tmp_path / "bad" / "policy.json"
        bad.parent.mkdir()
        bad.write_bytes(b"wrong\n")
        with pytest.raises(RuntimeLocatorError, match="(size mismatch|sha256 mismatch)"):
            resolve_policy_authority(authority_path, authority_id="hardware-policy:test", expected_identity_sha256=identity_sha, locator_override=bad)
