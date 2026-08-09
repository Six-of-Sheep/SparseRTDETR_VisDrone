"""Read-only binding to the certified R3 conversion metadata."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..data_protocol.converter import _OFFICIAL_IMAGE_NAME
from .contract import (
    BaselineContractError,
    R3_ARTIFACT_INVENTORY_SHA256,
    R3_ARTIFACT_RELATIVE,
    R3_CATEGORY_CONTRACT_SHA256,
    R3_COMPLETION_SHA256,
    R3_CONFIG_SHA256,
    R3_ENTRY_CANONICAL_INVENTORY_SHA256,
    R3_SOURCE_IDENTITY_SHA256,
)


ROLE_ANNOTATIONS = {
    "train_core": "train_core_coco.json",
    "development": "development_coco.json",
}
FORBIDDEN_ROLES = frozenset({"confirmatory", "test", "raw_train", "raw_val"})
RUNTIME_IMAGE_MAPPING = {
    "train_core": ("train/images", "VisDrone2019-DET-train/images"),
    "development": ("val/images", "VisDrone2019-DET-val/images"),
}


@dataclass(frozen=True)
class R3ArtifactBinding:
    artifact_root: Path
    completion: Path
    artifact_inventory: Path
    config: Path
    category_contract: Path
    source_identity: Path


@dataclass(frozen=True)
class RuntimePaths:
    role: str
    data_root: Path
    annotation_file: Path
    artifact_binding: R3ArtifactBinding


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise BaselineContractError(f"cannot read required R3 metadata: {path}") from exc


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BaselineContractError(f"invalid R3 metadata: {path}") from exc
    if not isinstance(value, dict):
        raise BaselineContractError(f"R3 metadata must be a JSON object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BaselineContractError(message)


def _verify_inventory_metadata(inventory: dict[str, Any]) -> None:
    _require(inventory.get("schema_version") == 1, "R3 artifact inventory schema drift")
    _require(
        inventory.get("canonical_inventory_sha256") == R3_ENTRY_CANONICAL_INVENTORY_SHA256,
        "R3 artifact inventory canonical SHA drift",
    )
    rows = inventory.get("artifacts")
    _require(isinstance(rows, list), "R3 artifact inventory rows missing")
    row_paths = {
        row.get("relative_path")
        for row in rows
        if isinstance(row, dict)
    }
    for name in ("train_core_coco.json", "development_coco.json"):
        _require(name in row_paths, f"R3 inventory does not bind {name}")


def _verify_completion_metadata(completion: dict[str, Any]) -> None:
    _require(completion.get("schema_version") == 1, "R3 completion schema drift")
    _require(completion.get("status") == "COMPLETED", "R3 completion is not COMPLETED")
    _require(
        completion.get("artifact_inventory_sha256") == R3_ARTIFACT_INVENTORY_SHA256,
        "R3 completion inventory SHA drift",
    )
    _require(completion.get("selection_allowed") is False, "R3 selection must remain disabled")
    _require(completion.get("metrics_access_allowed") is False, "R3 metrics must remain disabled")
    _require(completion.get("confirmatory_metrics_accessed") is False, "R3 confirmatory metrics were accessed")
    _require(completion.get("dataset_test_accessed_by_this_process") is False, "R3 process accessed test data")
    _require(completion.get("production_conversion_executed") is True, "R3 production conversion missing")


def _verify_protocol_metadata(config: dict[str, Any], category_contract: dict[str, Any], source: dict[str, Any]) -> None:
    _require(config.get("protocol_id") == "P3-VISDRONE-DATA-PROTOCOL-V2", "R3 protocol ID drift")
    _require(config.get("test_access_allowed") is False, "R3 test access must remain disabled")
    _require(config.get("confirmatory_metrics_accessed") is False, "R3 confirmatory metrics flag drift")
    split = config.get("split")
    _require(isinstance(split, dict), "R3 split metadata missing")
    _require(split.get("selection_allowed") is False, "R3 split selection must remain disabled")
    _require(split.get("metrics_access_allowed") is False, "R3 split metrics must remain disabled")
    _require(category_contract.get("num_classes") == 10, "R3 category count drift")
    categories = category_contract.get("categories")
    _require(
        isinstance(categories, list)
        and [item.get("id") for item in categories if isinstance(item, dict)] == list(range(1, 11)),
        "R3 category IDs drift",
    )
    _require(source.get("dataset_test_accessed_by_this_process") is False, "R3 source identity test flag drift")


def verify_r3_binding(repo_root: str | Path, artifact_root: str | Path | None = None) -> R3ArtifactBinding:
    """Verify only the frozen R3 metadata files and return their bound paths."""

    root = Path(repo_root).resolve()
    expected_root = (root / R3_ARTIFACT_RELATIVE).resolve()
    actual_root = expected_root if artifact_root is None else Path(artifact_root).resolve()
    _require(actual_root == expected_root, "runtime artifact root is not the certified R3 root")

    paths = {
        "completion": actual_root / "completion.json",
        "artifact_inventory": actual_root / "artifact_inventory.json",
        "config": actual_root / "config.json",
        "category_contract": actual_root / "category_contract.json",
        "source_identity": actual_root / "source_identity.json",
    }
    expected_shas = {
        "completion": R3_COMPLETION_SHA256,
        "artifact_inventory": R3_ARTIFACT_INVENTORY_SHA256,
        "config": R3_CONFIG_SHA256,
        "category_contract": R3_CATEGORY_CONTRACT_SHA256,
        "source_identity": R3_SOURCE_IDENTITY_SHA256,
    }
    for name, path in paths.items():
        _require(path.is_file(), f"missing certified R3 metadata: {path}")
        _require(_sha256(path) == expected_shas[name], f"R3 {name} SHA drift")

    completion = _load_json(paths["completion"])
    inventory = _load_json(paths["artifact_inventory"])
    config = _load_json(paths["config"])
    category_contract = _load_json(paths["category_contract"])
    source = _load_json(paths["source_identity"])
    _verify_completion_metadata(completion)
    _verify_inventory_metadata(inventory)
    _verify_protocol_metadata(config, category_contract, source)

    return R3ArtifactBinding(
        artifact_root=actual_root,
        completion=paths["completion"],
        artifact_inventory=paths["artifact_inventory"],
        config=paths["config"],
        category_contract=paths["category_contract"],
        source_identity=paths["source_identity"],
    )


def validate_runtime_role(role: Any) -> str:
    if type(role) is not str:
        raise BaselineContractError("runtime role must be an explicit string")
    if role in FORBIDDEN_ROLES:
        raise BaselineContractError(f"runtime role is forbidden: {role}")
    if role not in ROLE_ANNOTATIONS:
        raise BaselineContractError(f"unknown runtime role: {role}")
    return role


def _reject_test_path(path: Path) -> None:
    if any(part.casefold() == "test" for part in path.parts):
        raise BaselineContractError("runtime data_root must not contain an independent test path segment")


def resolve_runtime_paths(
    repo_root: str | Path,
    data_root: str | Path,
    role: str,
    artifact_root: str | Path | None = None,
) -> RuntimePaths:
    """Resolve one allowed runtime role without discovering sibling splits."""

    selected_role = validate_runtime_role(role)
    if not isinstance(data_root, (str, Path)):
        raise BaselineContractError("data_root must be an explicit path")
    data_path = Path(data_root)
    _reject_test_path(data_path)
    binding = verify_r3_binding(repo_root, artifact_root)
    annotation = binding.artifact_root / ROLE_ANNOTATIONS[selected_role]
    return RuntimePaths(
        role=selected_role,
        data_root=data_path,
        annotation_file=annotation,
        artifact_binding=binding,
    )


def resolve_runtime_image_path(
    data_root: Path,
    role: str,
    logical_relative_path: str,
) -> Path:
    """Map one certified logical COCO image path to an official raw image."""

    if not isinstance(data_root, Path) or not data_root.is_absolute():
        raise BaselineContractError("runtime image data_root must be an absolute Path")
    selected_role = validate_runtime_role(role)
    if selected_role not in RUNTIME_IMAGE_MAPPING:
        raise BaselineContractError("runtime image role has no frozen path mapping")
    if type(logical_relative_path) is not str or not logical_relative_path:
        raise BaselineContractError("runtime image logical path must be a built-in non-empty string")
    if logical_relative_path.startswith("/") or "\\" in logical_relative_path:
        raise BaselineContractError("runtime image logical path must be a relative POSIX path")
    parts = logical_relative_path.split("/")
    logical_prefix, raw_relative_dir = RUNTIME_IMAGE_MAPPING[selected_role]
    if len(parts) != 3 or "/".join(parts[:2]) != logical_prefix or any(part in {"", ".", ".."} for part in parts):
        raise BaselineContractError("runtime image logical path has an invalid frozen structure")
    filename = parts[2]
    if _OFFICIAL_IMAGE_NAME.fullmatch(filename) is None:
        raise BaselineContractError("runtime image basename is not an official VisDrone JPG")
    if data_root.is_symlink():
        raise BaselineContractError("runtime image data_root may not be a symlink")
    try:
        canonical_root = data_root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise BaselineContractError("runtime image data_root is missing or invalid") from exc
    if canonical_root.is_symlink() or not canonical_root.is_dir():
        raise BaselineContractError("runtime image data_root must be a regular directory")
    _reject_test_path(canonical_root)
    raw_images = canonical_root / raw_relative_dir
    if raw_images.is_symlink() or not raw_images.is_dir():
        raise BaselineContractError("official raw VisDrone image directory is missing or invalid")
    candidate = raw_images / filename
    if candidate.is_symlink() or not candidate.is_file():
        raise BaselineContractError("official raw VisDrone image is missing or invalid")
    try:
        canonical_images = raw_images.resolve(strict=True)
        canonical_candidate = candidate.resolve(strict=True)
        relative_candidate = canonical_candidate.relative_to(canonical_images)
    except (OSError, RuntimeError, ValueError) as exc:
        raise BaselineContractError("runtime image path escapes the official raw image directory") from exc
    if len(relative_candidate.parts) != 1 or canonical_candidate.is_symlink():
        raise BaselineContractError("runtime image path is outside the official raw image directory")
    return canonical_candidate
