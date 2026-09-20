#!/usr/bin/env python3
"""Deterministically seal the minimal P3 v2b REV1 authority graph.

The tool intentionally supports only external authorities, scientific source,
and read-only evaluation contracts. Runtime locators are carried as operational
metadata and are never included in authority/source identity digests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA_VERSION = 1
HEX = set("0123456789abcdef")


class SealError(ValueError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha(value: Any) -> str:
    return sha_bytes(canonical_bytes(value))


def identity_without_runtime(value: Any) -> Any:
    """Remove operational locators from a canonical identity projection."""
    if isinstance(value, dict):
        return {key: identity_without_runtime(child)
                for key, child in value.items()
                if key not in {"runtime_locator", "runtime_file_count"}}
    if isinstance(value, list):
        return [identity_without_runtime(child) for child in value]
    return value


def _sha(value: Any, field: str) -> str:
    if type(value) is not str or len(value) != 64 or set(value) - HEX:
        raise SealError(f"{field} must be a lowercase SHA-256")
    return value


def _posix_relative(value: Any, field: str) -> str:
    if type(value) is not str or not value or "\\" in value or "\x00" in value:
        raise SealError(f"{field} must be a relative POSIX path")
    parts = value.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise SealError(f"{field} is not lexically relative")
    p = PurePosixPath(value)
    if p.is_absolute() or p.parts != tuple(parts):
        raise SealError(f"{field} is not a canonical POSIX path")
    return value


def _root(value: Any, field: str) -> Path:
    if type(value) is not str or not value or "\x00" in value:
        raise SealError(f"{field} must be an absolute runtime path")
    p = Path(value).resolve(strict=True)
    return p


def _regular(path: Path, label: str) -> None:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise SealError(f"{label} must be a regular non-symlink file")


def file_row(root: Path, relative: str) -> dict[str, Any]:
    relative = _posix_relative(relative, "relative_path")
    path = root / Path(*relative.split("/"))
    if path.is_symlink():
        raise SealError(f"{relative} must not be a symlink")
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise SealError(f"missing authority file: {relative}") from exc
    if not resolved.is_relative_to(root):
        raise SealError("authority file escapes its runtime root")
    _regular(path, relative)
    raw = path.read_bytes()
    return {"relative_path": relative, "size_bytes": len(raw),
            "sha256": sha_bytes(raw)}


def inventory(root: Path, include_files: list[str]) -> dict[str, Any]:
    rows = [file_row(root, value) for value in include_files]
    if len({row["relative_path"] for row in rows}) != len(rows):
        raise SealError("authority file list contains duplicates")
    rows.sort(key=lambda row: row["relative_path"])
    content = hashlib.sha256()
    for row in rows:
        content.update(row["relative_path"].encode("utf-8"))
        content.update(b"\0")
        content.update((root / Path(*row["relative_path"].split("/"))).read_bytes())
    body = {"file_count": len(rows),
            "total_size_bytes": sum(row["size_bytes"] for row in rows),
            "rows": rows}
    return {**body, "inventory_sha256": canonical_sha(body),
            "content_sha256": content.hexdigest()}


def _authority_identity(authority_id: str, logical_type: str,
                        inv: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "authority_id": authority_id,
            "logical_type": logical_type, "inventory": inv}


def _authority(spec: dict[str, Any]) -> dict[str, Any]:
    authority_id = spec.get("authority_id")
    logical_type = spec.get("logical_type")
    if type(authority_id) is not str or not authority_id:
        raise SealError("authority_id is required")
    if type(logical_type) is not str or not logical_type:
        raise SealError("logical_type is required")
    root = _root(spec.get("root"), f"{authority_id}.root")
    files = spec.get("include_files")
    if type(files) is not list or not files or any(type(v) is not str for v in files):
        raise SealError(f"{authority_id}.include_files must be a non-empty list")
    inv = inventory(root, files)
    identity = _authority_identity(authority_id, logical_type, inv)
    return {**identity, "identity_sha256": canonical_sha(identity),
            "runtime_locator": {"path": str(root)},
            "runtime_file_count": inv["file_count"]}


def _source(repo_root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    files = spec.get("files")
    if type(files) is not list or not files or any(type(v) is not str for v in files):
        raise SealError("scientific_source.files must be a non-empty list")
    rows = [file_row(repo_root, value) for value in files]
    rows.sort(key=lambda row: row["relative_path"])
    identity = {"schema_version": SCHEMA_VERSION,
                "source_id": spec.get("source_id", "v2b-sci-001"),
                "scope": spec.get("scope", "v2b_evaluation_and_training_math"),
                "files": rows}
    identity["scientific_source_sha256"] = canonical_sha(identity)
    return identity


def _checkpoint(spec: dict[str, Any]) -> dict[str, Any]:
    path = _root(spec.get("path"), "checkpoint.path")
    _regular(path, "checkpoint")
    raw = path.read_bytes()
    actual = sha_bytes(raw)
    declared = spec.get("sha256", actual)
    if declared != actual:
        raise SealError(f"checkpoint SHA mismatch: {path}")
    epoch = spec.get("epoch")
    if type(epoch) is not int or epoch < 1:
        raise SealError("checkpoint epoch must be a positive integer")
    base = {"checkpoint_id": spec.get("checkpoint_id"),
            "source_campaign": spec.get("source_campaign"),
            "seed": spec.get("seed"), "resolution": spec.get("resolution"),
            "epoch": epoch, "sha256": actual, "size_bytes": len(raw)}
    if (type(base["checkpoint_id"]) is not str or not base["checkpoint_id"]
            or type(base["source_campaign"]) is not str or not base["source_campaign"]
            or type(base["seed"]) is not int or type(base["resolution"]) is not int):
        raise SealError("checkpoint identity fields are incomplete")
    identity = {**base, "runtime_locator": {"path": str(path)}}
    return identity


def _evaluation(repo_root: Path, spec: dict[str, Any], scientific: dict[str, Any],
                authorities: dict[str, Any]) -> dict[str, Any]:
    authority_id = spec.get("development_authority_id")
    if authority_id not in authorities:
        raise SealError("evaluation references an unknown development authority")
    evaluator_files = spec.get("evaluator_source_files")
    if type(evaluator_files) is not list or not evaluator_files:
        raise SealError("evaluator_source_files must be non-empty")
    evaluator_rows = [file_row(repo_root, value) for value in evaluator_files]
    evaluator_rows.sort(key=lambda row: row["relative_path"])
    checkpoints = spec.get("checkpoints")
    if type(checkpoints) is not list or not checkpoints:
        raise SealError("evaluation checkpoints must be non-empty")
    entries = []
    for item in checkpoints:
        cp = _checkpoint(item)
        for selector in ("raw", "ema"):
            entries.append({"checkpoint": {k: v for k, v in cp.items()
                                             if k != "runtime_locator"},
                            "weight_selector": selector,
                            "runtime_locator": cp["runtime_locator"]})
    body = {
        "schema_version": SCHEMA_VERSION,
        "evaluation_id": spec.get("evaluation_id", "v2b-rev1-development-readonly"),
        "mode": "read_only_development_evaluation",
        "training_authorized": False,
        "continuation_authorized": False,
        "scientific_source_sha256": scientific["scientific_source_sha256"],
        "development_authority_id": authority_id,
        "evaluator_source": {"files": evaluator_rows,
                             "sha256": canonical_sha(evaluator_rows)},
        "protocol": spec.get("protocol", {}),
        "model": spec.get("model", {}),
        "postprocessor": spec.get("postprocessor", {}),
        "metrics": spec.get("metrics", []),
        "checkpoints": entries,
    }
    body["evaluation_contract_sha256"] = canonical_sha(identity_without_runtime(body))
    return body


def _write(path: Path, value: Any) -> str:
    raw = canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise SealError(f"refusing to overwrite {path}")
    path.write_bytes(raw)
    return sha_bytes(raw)


def seal(spec_path: Path, output_dir: Path, repo_root: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SealError("output directory must be absent or empty")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    raw_authorities = spec.get("external_authorities")
    if type(raw_authorities) is not list or not raw_authorities:
        raise SealError("external_authorities are required")
    authorities_list = [_authority(value) for value in raw_authorities]
    authorities = {item["authority_id"]: item for item in authorities_list}
    if len(authorities) != len(authorities_list):
        raise SealError("duplicate authority_id")
    scientific = _source(repo_root, spec.get("scientific_source", {}))
    evaluation = _evaluation(repo_root, spec.get("evaluation", {}), scientific, authorities)
    auth_doc = {"schema_version": SCHEMA_VERSION,
                "revision_id": spec.get("revision_id", "P3-V2B-CONTRACT-REV1"),
                "authorities": authorities_list}
    auth_doc["manifest_sha256"] = canonical_sha(identity_without_runtime(auth_doc))
    outputs = {}
    outputs["external_authorities.json"] = _write(output_dir / "external_authorities.json", auth_doc)
    outputs["scientific_source.json"] = _write(output_dir / "scientific_source.json", scientific)
    outputs["evaluation_contract.json"] = _write(output_dir / "evaluation_contract.json", evaluation)
    revision_body = {
        "schema_version": SCHEMA_VERSION,
        "revision_id": spec.get("revision_id", "P3-V2B-CONTRACT-REV1"),
        "parent_revision": spec.get("parent_revision"),
        "stages": {name: {"path": name, "sha256": digest}
                   for name, digest in outputs.items()},
        "graph": {"nodes": ["external_authorities", "scientific_source",
                             "evaluation_contract"],
                  "edges": [["external_authorities", "scientific_source"],
                            ["external_authorities", "evaluation_contract"],
                            ["scientific_source", "evaluation_contract"]]},
        "training_contract": {"status": "DEFERRED"},
        "execution_contract": {"status": "DEFERRED"},
    }
    revision = {**revision_body, "revision_sha256": canonical_sha(revision_body)}
    outputs["revision.json"] = _write(output_dir / "revision.json", revision)
    return {"status": "PASS", "revision_id": revision["revision_id"],
            "output_dir": str(output_dir.resolve()), "files": outputs,
            "scientific_source_sha256": scientific["scientific_source_sha256"],
            "evaluation_contract_sha256": evaluation["evaluation_contract_sha256"],
            "authority_count": len(authorities_list)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = seal(args.spec.resolve(strict=True), args.output_dir.resolve(),
                      args.repo_root.resolve(strict=True))
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"},
                         sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
