"""Local smoke tests for the runtime locator implementation.

This file is copied to the remote repository as tests/test_v2b_runtime_locator.py.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sparse_rtdetr.baseline.training_v2b_runtime_locator import RuntimeLocatorError, resolve_bridge, resolve_file, resolve_policy


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
