"""Runtime-only authority and locator resolution for REV1 execution.

Physical mount points are deliberately kept out of canonical identities.  This
module verifies the bytes found at a runtime locator against an already sealed
logical authority and fails closed instead of guessing alternate paths.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any


class RuntimeLocatorError(ValueError):
    """A runtime locator could not be proven to satisfy its authority."""

    prefix = "RUNTIME_LOCATOR_UNAVAILABLE"

    def __init__(self, detail: str):
        super().__init__(f"{self.prefix}: {detail}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise RuntimeLocatorError(f"cannot read {path}: {exc}") from exc
    return digest.hexdigest()


def _check_logical_path(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise RuntimeLocatorError("invalid logical path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or "\x00" in value:
        raise RuntimeLocatorError(f"unsafe logical path: {value!r}")
    return value


def _regular_non_symlink(path: Path) -> os.stat_result:
    try:
        info = path.lstat()
    except OSError as exc:
        raise RuntimeLocatorError(f"missing path {path}: {exc}") from exc
    if stat.S_ISLNK(info.st_mode):
        raise RuntimeLocatorError(f"symlink is not allowed: {path}")
    if not stat.S_ISREG(info.st_mode):
        raise RuntimeLocatorError(f"regular file required: {path}")
    return info


def resolve_file(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_size_bytes: int | None = None,
    logical_id: str = "file",
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve one physical locator and verify exact bytes and type."""
    candidate = Path(path)
    if root is not None:
        root_path = Path(root).resolve(strict=True)
        try:
            candidate.resolve(strict=False).relative_to(root_path)
        except ValueError as exc:
            raise RuntimeLocatorError(f"locator escapes root for {logical_id}") from exc
    info = _regular_non_symlink(candidate)
    if expected_size_bytes is not None and info.st_size != int(expected_size_bytes):
        raise RuntimeLocatorError(
            f"size mismatch for {logical_id}: {info.st_size} != {expected_size_bytes}"
        )
    actual = _sha256(candidate)
    if actual != expected_sha256:
        raise RuntimeLocatorError(f"sha256 mismatch for {logical_id}: {actual} != {expected_sha256}")
    return {
        "logical_id": logical_id,
        "path": str(candidate),
        "size_bytes": info.st_size,
        "sha256": actual,
        "symlink": False,
    }


def _load_json(path: str | Path, *, require_canonical: bool) -> dict[str, Any]:
    candidate = Path(path)
    _regular_non_symlink(candidate)
    try:
        raw = candidate.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeLocatorError(f"invalid JSON authority {candidate}: {exc}") from exc
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    if require_canonical and raw != canonical.encode("utf-8"):
        raise RuntimeLocatorError(f"non-canonical authority JSON: {candidate}")
    if not isinstance(value, dict):
        raise RuntimeLocatorError(f"authority object required: {candidate}")
    return value


def load_canonical_json(path: str | Path) -> dict[str, Any]:
    return _load_json(path, require_canonical=True)


def load_runtime_json(path: str | Path) -> dict[str, Any]:
    """Load bytes already protected by an external SHA/manifest."""
    return _load_json(path, require_canonical=False)


def resolve_bridge(
    checkpoint: str | Path,
    *,
    checkpoint_sha256: str,
    manifest: str | Path,
    manifest_sha256: str,
) -> dict[str, Any]:
    """Verify the bridge checkpoint and its separately sealed manifest."""
    cp = resolve_file(checkpoint, expected_sha256=checkpoint_sha256, logical_id="bridge-v2-checkpoint")
    mf = resolve_file(manifest, expected_sha256=manifest_sha256, logical_id="bridge-v2-manifest")
    manifest_doc = load_runtime_json(manifest)
    if manifest_doc.get("checkpoint_sha256") not in (None, checkpoint_sha256):
        raise RuntimeLocatorError("bridge manifest checkpoint identity mismatch")
    return {"checkpoint": cp, "manifest": mf, "manifest_document": manifest_doc}


def resolve_policy(
    policy: str | Path,
    *,
    expected_sha256: str,
    expected_size_bytes: int | None = None,
    expected_gpu_uuid: str | None = None,
) -> dict[str, Any]:
    """Verify a hardware policy locator without initializing CUDA."""
    row = resolve_file(
        policy,
        expected_sha256=expected_sha256,
        expected_size_bytes=expected_size_bytes,
        logical_id="native-hardware-policy",
    )
    doc = load_runtime_json(policy)
    if expected_gpu_uuid and doc.get("gpu_uuid") not in (None, expected_gpu_uuid):
        raise RuntimeLocatorError("native hardware policy GPU identity mismatch")
    return {"file": row, "policy": doc}


def resolve_authority_file(
    authority_document: str | Path,
    *,
    authority_id: str,
    locator: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve an authority by logical id, never by a filename search."""
    document = load_canonical_json(authority_document)
    authorities = document.get("authorities")
    if not isinstance(authorities, list):
        raise RuntimeLocatorError("authority list missing")
    matches = [row for row in authorities if row.get("authority_id") == authority_id]
    if len(matches) != 1:
        raise RuntimeLocatorError(f"authority id is not unique: {authority_id}")
    authority = matches[0]
    runtime = authority.get("runtime_locator") or {}
    chosen = Path(locator) if locator is not None else Path(runtime.get("path", ""))
    if not str(chosen):
        raise RuntimeLocatorError(f"runtime locator missing for {authority_id}")
    inventory = authority.get("inventory") or {}
    rows = inventory.get("rows") or []
    if not rows:
        return {"authority_id": authority_id, "path": str(chosen), "authority": authority}
    # Inventory authorities resolve a directory; each declared logical row is
    # checked.  Unknown ignored cache is intentionally outside this identity.
    root = chosen
    if not root.is_dir():
        raise RuntimeLocatorError(f"authority root is not a directory: {root}")
    actual_rows = []
    for row in rows:
        logical = _check_logical_path(row.get("relative_path"))
        actual_rows.append(
            resolve_file(
                root / Path(*logical.split("/")),
                expected_sha256=row["sha256"],
                expected_size_bytes=row.get("size_bytes"),
                logical_id=f"{authority_id}:{logical}",
            )
        )
    return {"authority_id": authority_id, "path": str(root), "authority": authority, "files": actual_rows}
