"""Stable, path-independent identities used by the VisDrone protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import PurePosixPath


class ProtocolContractError(ValueError):
    """Raised when a value violates a frozen protocol contract."""


def canonical_json_bytes(value: object) -> bytes:
    """Serialize JSON with the protocol's byte-stable settings."""

    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _hex_digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def ensure_allowed_dataset_path(relative_path: str, split: str | None = None) -> str:
    """Validate a relative train/val path without touching the filesystem."""

    if not isinstance(relative_path, str) or not relative_path:
        raise ProtocolContractError("dataset path must be a non-empty string")
    path = PurePosixPath(relative_path)
    if path.is_absolute() or ".." in path.parts:
        raise ProtocolContractError("dataset path must be relative and contained")
    parts = {part.casefold() for part in path.parts}
    if split is not None and split.casefold() == "test":
        raise ProtocolContractError("test split access is disabled")
    if "test" in parts or "test-dev" in parts or "test_dev" in parts:
        raise ProtocolContractError("test split path is rejected")
    if split is not None and split.casefold() not in {"train", "val", "development", "confirmatory"}:
        raise ProtocolContractError(f"unsupported dataset split: {split}")
    return path.as_posix()


def stable_image_id(split: str, relative_path: str) -> str:
    """Return an ID derived only from split and normalized relative path."""

    normalized = ensure_allowed_dataset_path(relative_path, split)
    payload = canonical_json_bytes({"relative_path": normalized, "split": split})
    return _hex_digest(b"visdrone:image:v1\0" + payload)


def stable_annotation_id(image_id: str, physical_line_number: int, raw_line_sha256: str) -> str:
    """Return an annotation ID tied to its source-line identity."""

    if not image_id or physical_line_number < 1:
        raise ProtocolContractError("invalid annotation identity inputs")
    if len(raw_line_sha256) != 64 or any(c not in "0123456789abcdef" for c in raw_line_sha256):
        raise ProtocolContractError("raw line hash must be lowercase SHA-256")
    payload = canonical_json_bytes({
        "image_id": image_id,
        "physical_line_number": physical_line_number,
        "raw_line_sha256": raw_line_sha256,
    })
    return _hex_digest(b"visdrone:annotation:v1\0" + payload)
