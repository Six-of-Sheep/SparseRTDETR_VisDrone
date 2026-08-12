"""Atomic evidence and inventory primitives for Baseline Smoke V1."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import tempfile
from pathlib import Path
from typing import Any

from ..data_protocol.schema import ProtocolContractError
from .contract import VISDRONE_BASELINE_PARAMETER_COUNT


class SmokeEvidenceError(ProtocolContractError):
    """Raised when smoke evidence ownership or binding is invalid."""


ENTRY_INVENTORY_EXCLUDED = frozenset({"artifact_inventory.json", "completion.json"})
ENTRY_PARTIAL_EXCLUDED = frozenset({"partial_inventory.json", "completion.json"})
ENTRY_REQUIRED_ARTIFACTS = frozenset({
    "config.json",
    "invocation.json",
    "source_identity.json",
    "data_binding_audit.json",
    "image_selection_audit.json",
    "cuda_runtime_identity.json",
    "model_identity.json",
    "call_audit.json",
    "input_batch_audit.json",
    "model_output_audit.json",
    "postprocess_audit.json",
    "rng_audit.json",
})
SCIENTIFIC_SCHEMA_VERSION = 3
SCIENTIFIC_SOURCE_ALLOWLIST = (
    "src/sparse_rtdetr/__init__.py",
    "src/sparse_rtdetr/baseline/__init__.py",
    "src/sparse_rtdetr/baseline/artifacts.py",
    "src/sparse_rtdetr/baseline/categories.py",
    "src/sparse_rtdetr/baseline/config.py",
    "src/sparse_rtdetr/baseline/contract.py",
    "src/sparse_rtdetr/baseline/dataset.py",
    "src/sparse_rtdetr/baseline/postprocessor.py",
    "src/sparse_rtdetr/baseline/smoke.py",
    "src/sparse_rtdetr/baseline/smoke_evidence.py",
    "src/sparse_rtdetr/baseline/smoke_launcher.py",
    "src/sparse_rtdetr/baseline/smoke_outer_launcher.py",
    "src/sparse_rtdetr/data_protocol/__init__.py",
    "src/sparse_rtdetr/data_protocol/categories.py",
    "src/sparse_rtdetr/data_protocol/converter.py",
    "src/sparse_rtdetr/data_protocol/evaluation.py",
    "src/sparse_rtdetr/data_protocol/lineage.py",
    "src/sparse_rtdetr/data_protocol/parser.py",
    "src/sparse_rtdetr/data_protocol/protocol.py",
    "src/sparse_rtdetr/data_protocol/schema.py",
    "src/sparse_rtdetr/data_protocol/split.py",
)
SCIENTIFIC_ENVIRONMENT_ALLOWLIST = (
    "environment/conda-packages.json",
    "environment/conda-linux-64.explicit.txt",
)
SCIENTIFIC_REAL_CUDA_IDENTITY = {
    "cuda_visible_devices": "0",
    "device_count": 1,
    "current_device": 0,
    "device": "cuda:0",
    "device_name": "NVIDIA GeForce RTX 4090 D",
    "device_capability": [8, 9],
    "torch_version": "2.4.1",
    "torch_cuda_version": "12.4",
    "cpu_fallback": False,
    "runtime_capture_after_cuda_initialization": True,
}
SCIENTIFIC_SYNTHETIC_CUDA_IDENTITY = {
    "cuda_visible_devices": "",
    "device_count": 0,
    "current_device": None,
    "device": "cpu",
    "device_name": "cpu",
    "device_capability": None,
    "torch_version": "synthetic",
    "torch_cuda_version": None,
    "cpu_fallback": True,
    "runtime_capture_after_cuda_initialization": False,
}
SCIENTIFIC_PREPROCESSING_EXPECTED = {
    "compose": {"class": "Compose", "module": "src.data.transforms.container"},
    "ordered_pipeline": [
        {
            "class": "Resize",
            "module": "torchvision.transforms.v2._geometry",
            "size": [640, 640],
            "interpolation": "bilinear",
            "antialias": True,
            "max_size": None,
        },
        {
            "class": "ConvertPILImage",
            "module": "src.data.transforms._transforms",
            "dtype": "float32",
            "scale": True,
        },
    ],
    "resize": {
        "size": [640, 640],
        "interpolation": "bilinear",
        "antialias": True,
        "max_size": None,
    },
    "convert_pil_image": {"dtype": "float32", "scale": True},
    "scale_divisor": 255,
    "normalization": {"used": False, "mean": None, "std": None},
    "padding": {"used": False, "mode": None},
    "color": {"mode": "RGB", "channel_order": "RGB"},
}
SCIENTIFIC_BASELINE_CONFIG = "configs/baseline/rtdetrv2_r18_visdrone_baseline_v1.json"
SCIENTIFIC_MANIFEST = "manifests/rtdetrv2_upstream.json"
SCIENTIFIC_UPSTREAM_COMMIT = "1c8ac3f7ba84f14bd5651ab7b1b70d69a5f55f47"
SCIENTIFIC_UPSTREAM_ROOT_TREE = "b6de37e186373fc59b91d23c846bb0eda35b6986"
SCIENTIFIC_UPSTREAM_SUBTREE = "96a3b3e7e015d5e548e2917df2fec9641375e96e"
SCIENTIFIC_VENDOR_INVENTORY_SHA256 = "2312c80d5b0fba88d43ffc6807c3fc150ae74b77740f2ab65072f044e033d6d7"
SCIENTIFIC_VENDOR_FILE_COUNT = 124
SCIENTIFIC_VENDOR_TOTAL_BYTES = 373735
SCIENTIFIC_R3_BINDING = {
    "completion_sha256": "46734010937168ac65bf27c3a6f4bf3f234554d5d4a99407903dec4b3ae9e817",
    "artifact_inventory_sha256": "aaee01ce4e00749b9db8ac5e6e49cf35875a8b266e9c3efc237bdf0b8b109ae7",
    "entry_canonical_inventory_sha256": "ddf32745302d6095e1c3dde89b1a85f05db53e6b4cc8ee85638e0d45bfa8d982",
    "config_sha256": "58be8659d6d7ae6faace82b36d45523f80cd48bae1bbcc6da4586c53056ec0d0",
    "category_contract_sha256": "b4b309f357cbe130a505a610dff340cc498dc74f766384c4559b2acd900728a0",
    "source_identity_sha256": "f0f16ba4438b51a09a6203f78884b199f8e2a2d3fb46309c357368a47a552bb9",
}
COMPLETION_COUNTERS = {
    "images_requested": 2,
    "images_loaded": 2,
    "batches_requested": 1,
    "batches_processed": 1,
    "dataset_next_calls": 1,
    "model_forward_calls": 1,
    "postprocessor_calls": 1,
    "evaluator_calls": 0,
    "criterion_calls": 0,
    "backward_calls": 0,
    "optimizer_steps": 0,
    "scheduler_steps": 0,
    "checkpoint_loads": 0,
    "network_calls": 0,
}
COMPLETION_BOOLEAN_FIELDS = {
    "smoke_pass_candidate": True,
    "non_training": True,
    "inference_only": True,
    "model_eval": True,
    "torch_no_grad": True,
    "non_selection": True,
    "non_reportable": True,
    "formal_resume_eligible": False,
    "formal_training_eligible": False,
    "baseline_training_ready": False,
    "confirmatory_metrics_accessed": False,
    "dataset_test_accessed_by_this_process": False,
    "speed_measurement": False,
}
COMPLETION_KEYS = frozenset({
    "schema_version",
    "status",
    "mode",
    "config_sha256",
    "total_predictions",
    "artifact_inventory_sha256",
    *COMPLETION_COUNTERS,
    *COMPLETION_BOOLEAN_FIELDS,
})
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SmokeEvidenceError("evidence value is not canonical JSON") from exc


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_schema_version(value: Any, field: str) -> None:
    if type(value) is not int or value != SCIENTIFIC_SCHEMA_VERSION:
        raise SmokeEvidenceError(f"{field} must be schema version 3")


def _descriptor_sha(value: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _data_binding_sha(value: dict[str, Any]) -> str:
    return _descriptor_sha({key: child for key, child in value.items() if key != "context"})


def _identity_rows(root: Path, relatives: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = []
    for relative in relatives:
        path = root / relative
        if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
            raise SmokeEvidenceError(f"identity file is not a regular unlinked file: {relative}")
        rows.append({"relative_path": relative, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return rows


def build_scientific_context(
    repo_root: str | Path,
    *,
    runtime_mode: str,
    expected_device: str,
    preprocessing: dict[str, Any],
    data_binding: dict[str, Any],
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    source_rows = _identity_rows(root, SCIENTIFIC_SOURCE_ALLOWLIST)
    environment_rows = _identity_rows(root, SCIENTIFIC_ENVIRONMENT_ALLOWLIST)
    return _context(
        runtime_mode=runtime_mode,
        expected_device=expected_device,
        preprocessing=preprocessing,
        data_binding=data_binding,
        source_rows_sha256=sha256_bytes(canonical_json_bytes(source_rows)),
        environment_rows_sha256=sha256_bytes(canonical_json_bytes(environment_rows)),
    )


def _validate_file_identity(root: Path, value: Any, field: str) -> None:
    if not isinstance(value, dict) or set(value) != {"relative_path", "size_bytes", "sha256"}:
        raise SmokeEvidenceError(f"{field} file identity schema drift")
    relative = value["relative_path"]
    if type(relative) is not str or not relative or relative.startswith("/") or ".." in Path(relative).parts:
        raise SmokeEvidenceError(f"{field} file path drift")
    path = root / relative
    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        raise SmokeEvidenceError(f"{field} file is not regular")
    _strict_int(value["size_bytes"], f"{field}.size_bytes")
    _strict_sha(value["sha256"], f"{field}.sha256")
    if value["size_bytes"] != path.stat().st_size or value["sha256"] != sha256_file(path):
        raise SmokeEvidenceError(f"{field} file identity mismatch")


def _validate_data_binding(value: Any, config: dict[str, Any], repo_root: Path, context: dict[str, Any]) -> None:
    required = {"schema_version", "context", "role", "manifest_relative_path", "records", "metadata", "portable", "nonportable", "real_data_accessed", "runtime_artifacts_accessed", "artifact_validation_mode"}
    if not isinstance(value, dict) or set(value) != required:
        raise SmokeEvidenceError("data_binding_audit schema fields are invalid")
    _strict_schema_version(value["schema_version"], "data_binding_audit.schema_version")
    if value["context"] != context or value["role"] != "train_core":
        raise SmokeEvidenceError("data_binding_audit context drift")
    if value["records"] != config["image_selection"]["records"]:
        raise SmokeEvidenceError("data_binding_audit records drift")
    if context["runtime_mode"] == "synthetic":
        expected = {"manifest_relative_path": None, "metadata": [], "portable": True, "nonportable": False, "real_data_accessed": False, "runtime_artifacts_accessed": False, "artifact_validation_mode": "synthetic_tracked_contract_only"}
        if any(value[key] != expected[key] for key in expected):
            raise SmokeEvidenceError("synthetic data binding drift")
        return
    expected_manifest = "artifacts/data/visdrone_protocol_v2_conversion_r3/train_core_manifest.json"
    if value["manifest_relative_path"] != expected_manifest or value["portable"] is not False or value["nonportable"] is not True or value["real_data_accessed"] is not True or value["runtime_artifacts_accessed"] is not True or value["artifact_validation_mode"] != "real_r3_metadata":
        raise SmokeEvidenceError("real data binding flags drift")
    metadata = value["metadata"]
    if type(metadata) is not list or len(metadata) != 6:
        raise SmokeEvidenceError("real R3 metadata identity is incomplete")
    for item in metadata:
        _validate_file_identity(repo_root, item, "data_binding_audit.metadata")
    names = [item["relative_path"] for item in metadata]
    expected_names = [
        "artifacts/data/visdrone_protocol_v2_conversion_r3/completion.json",
        "artifacts/data/visdrone_protocol_v2_conversion_r3/artifact_inventory.json",
        "artifacts/data/visdrone_protocol_v2_conversion_r3/config.json",
        "artifacts/data/visdrone_protocol_v2_conversion_r3/category_contract.json",
        "artifacts/data/visdrone_protocol_v2_conversion_r3/source_identity.json",
        expected_manifest,
    ]
    if names != expected_names:
        raise SmokeEvidenceError("real R3 metadata order drift")
    expected_r3_shas = {
        "artifacts/data/visdrone_protocol_v2_conversion_r3/completion.json": SCIENTIFIC_R3_BINDING["completion_sha256"],
        "artifacts/data/visdrone_protocol_v2_conversion_r3/artifact_inventory.json": SCIENTIFIC_R3_BINDING["artifact_inventory_sha256"],
        "artifacts/data/visdrone_protocol_v2_conversion_r3/config.json": SCIENTIFIC_R3_BINDING["config_sha256"],
        "artifacts/data/visdrone_protocol_v2_conversion_r3/category_contract.json": SCIENTIFIC_R3_BINDING["category_contract_sha256"],
        "artifacts/data/visdrone_protocol_v2_conversion_r3/source_identity.json": SCIENTIFIC_R3_BINDING["source_identity_sha256"],
    }
    for item in metadata:
        expected_sha = expected_r3_shas.get(item["relative_path"])
        if expected_sha is not None and item["sha256"] != expected_sha:
            raise SmokeEvidenceError("real R3 metadata is not bound to the baseline contract")
    try:
        from .artifacts import verify_r3_binding
        verify_r3_binding(repo_root)
    except Exception as exc:
        raise SmokeEvidenceError("real R3 metadata validation failed") from exc
    inventory_value = json.loads((repo_root / expected_names[1]).read_text(encoding="utf-8"))
    inventory_rows = inventory_value.get("artifacts") if isinstance(inventory_value, dict) else None
    manifest_row = next(
        (row for row in inventory_rows if isinstance(row, dict) and row.get("relative_path") == "train_core_manifest.json"),
        None,
    ) if isinstance(inventory_rows, list) else None
    manifest_item = metadata[-1]
    if not isinstance(manifest_row, dict) or manifest_item["size_bytes"] != manifest_row.get("size_bytes") or manifest_item["sha256"] != manifest_row.get("sha256"):
        raise SmokeEvidenceError("train_core manifest is not bound to the certified R3 inventory")
    _validate_train_core_manifest(repo_root / expected_manifest, config["image_selection"]["records"])


def _validate_train_core_manifest(path: Path, expected_records: list[dict[str, Any]]) -> None:
    """Re-read the certified manifest instead of trusting entry metadata alone."""

    if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
        raise SmokeEvidenceError("train_core manifest is not a regular file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SmokeEvidenceError("train_core manifest is invalid") from exc
    if not isinstance(payload, dict) or set(payload) != {"records"} or not isinstance(payload["records"], list):
        raise SmokeEvidenceError("train_core manifest schema drift")
    by_id: dict[int, dict[str, Any]] = {}
    for record in payload["records"]:
        if not isinstance(record, dict) or type(record.get("coco_image_id")) is not int:
            raise SmokeEvidenceError("train_core manifest record schema drift")
        if record["coco_image_id"] in by_id:
            raise SmokeEvidenceError("train_core manifest contains duplicate image IDs")
        by_id[record["coco_image_id"]] = record
    for expected in expected_records:
        image_id = expected.get("coco_image_id") if isinstance(expected, dict) else None
        actual = by_id.get(image_id)
        if actual != expected:
            raise SmokeEvidenceError("train_core manifest selected record drift")
    selected_in_manifest_order = [
        record for record in payload["records"]
        if isinstance(record, dict) and record.get("coco_image_id") in {item.get("coco_image_id") for item in expected_records}
    ]
    if selected_in_manifest_order != expected_records:
        raise SmokeEvidenceError("train_core manifest selected record order drift")


def describe_preprocessing_pipeline(pipeline: Any) -> dict[str, Any]:
    """Describe the actual frozen transform instances, including enum values."""

    transforms = getattr(pipeline, "transforms", None)
    if type(transforms) is not list or len(transforms) != 2:
        raise SmokeEvidenceError("preprocessing pipeline must contain exactly two transforms")
    resize, convert = transforms
    resize_descriptor = {
        "class": type(resize).__name__,
        "module": type(resize).__module__,
        "size": list(getattr(resize, "size", ())),
        "interpolation": getattr(getattr(resize, "interpolation", None), "value", getattr(resize, "interpolation", None)),
        "antialias": getattr(resize, "antialias", None),
        "max_size": getattr(resize, "max_size", None),
    }
    convert_descriptor = {
        "class": type(convert).__name__,
        "module": type(convert).__module__,
        "dtype": getattr(convert, "dtype", None),
        "scale": getattr(convert, "scale", None),
    }
    return {
        "compose": {"class": type(pipeline).__name__, "module": type(pipeline).__module__},
        "ordered_pipeline": [resize_descriptor, convert_descriptor],
        "resize": {key: resize_descriptor[key] for key in ("size", "interpolation", "antialias", "max_size")},
        "convert_pil_image": {key: convert_descriptor[key] for key in ("dtype", "scale")},
        "scale_divisor": 255,
        "normalization": {"used": False, "mean": None, "std": None},
        "padding": {"used": False, "mode": None},
        "color": {"mode": "RGB", "channel_order": "RGB"},
    }


def _context(*, runtime_mode: str, expected_device: str, preprocessing: dict[str, Any], data_binding: dict[str, Any], source_rows_sha256: str, environment_rows_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": SCIENTIFIC_SCHEMA_VERSION,
        "runtime_mode": runtime_mode,
        "expected_device": expected_device,
        "preprocessing_sha256": _descriptor_sha(preprocessing),
        "data_binding_sha256": _data_binding_sha(data_binding),
        "source_rows_canonical_sha256": source_rows_sha256,
        "environment_rows_canonical_sha256": environment_rows_sha256,
    }


def _strict_bool(value: Any, field: str) -> None:
    if type(value) is not bool:
        raise SmokeEvidenceError(f"{field} must be bool")


def _strict_int(value: Any, field: str) -> None:
    if type(value) is not int:
        raise SmokeEvidenceError(f"{field} must be int")


def _strict_sha(value: Any, field: str) -> None:
    if type(value) is not str or not SHA256_RE.fullmatch(value):
        raise SmokeEvidenceError(f"{field} must be lowercase SHA-256")


def _empty_state_sha() -> str:
    return sha256_bytes(canonical_json_bytes([]))


def _state_inventory(model: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    """Return the ordered parameter/buffer inventory without running the model."""

    parameter_inventory: list[dict[str, Any]] = []
    buffer_inventory: list[dict[str, Any]] = []
    all_finite = True
    if not hasattr(model, "named_parameters") and not hasattr(model, "named_buffers"):
        return parameter_inventory, buffer_inventory, all_finite
    import torch

    for name, parameter in model.named_parameters() if hasattr(model, "named_parameters") else ():
        detached = parameter.detach()
        finite = bool(torch.isfinite(detached).all())
        all_finite = all_finite and finite
        parameter_inventory.append({
            "name": name,
            "shape": list(detached.shape),
            "dtype": str(detached.dtype),
            "numel": int(detached.numel()),
            "requires_grad": bool(parameter.requires_grad),
            "kind": "parameter",
            "logical_sha256": sha256_bytes(detached.cpu().contiguous().numpy().tobytes()) if finite else None,
        })
    for name, buffer in model.named_buffers() if hasattr(model, "named_buffers") else ():
        detached = buffer.detach()
        finite = bool(torch.isfinite(detached).all())
        all_finite = all_finite and finite
        buffer_inventory.append({
            "name": name,
            "shape": list(detached.shape),
            "dtype": str(detached.dtype),
            "numel": int(detached.numel()),
            "requires_grad": False,
            "kind": "buffer",
            "logical_sha256": sha256_bytes(detached.cpu().contiguous().numpy().tobytes()) if finite else None,
        })
    return parameter_inventory, buffer_inventory, all_finite


def _state_hashes_from_inventory(
    parameter_inventory: list[dict[str, Any]],
    buffer_inventory: list[dict[str, Any]],
) -> tuple[str, str]:
    rows = parameter_inventory + buffer_inventory
    schema_rows = [
        {key: row[key] for key in ("name", "shape", "dtype", "numel", "requires_grad", "kind")}
        for row in rows
    ]
    value_rows = [
        {
            **{key: row[key] for key in ("name", "shape", "dtype", "numel", "requires_grad", "kind")},
            "logical_sha256": row["logical_sha256"],
        }
        for row in rows
    ]
    return sha256_bytes(canonical_json_bytes(schema_rows)), sha256_bytes(canonical_json_bytes(value_rows))


def _canonical_real_state_inventory(repo_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Construct and compare two seed-zero CPU R18 identities without forward."""

    import numpy as np
    import torch

    from .config import build_r18_cpu_model

    def build_once() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state().clone()
        try:
            random.seed(0)
            np.random.seed(0)
            torch.manual_seed(0)
            model = build_r18_cpu_model(repo_root)
            parameters, buffers, _ = _state_inventory(model)
            return parameters, buffers
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)

    first = build_once()
    second = build_once()
    if first != second:
        raise SmokeEvidenceError("MODEL_STATE_IDENTITY_NOT_DETERMINISTIC")
    return first


def _strict_shape(value: Any, expected: list[int], field: str) -> None:
    if type(value) is not list or value != expected or any(type(item) is not int for item in value):
        raise SmokeEvidenceError(f"{field} shape drift")


def _strict_tensor_audit(value: Any, field: str, *, device: str | None = None) -> None:
    if not isinstance(value, dict) or set(value) != {"shape", "dtype", "device", "finite", "logical_sha256"}:
        raise SmokeEvidenceError(f"{field} tensor schema drift")
    if type(value["dtype"]) is not str or type(value["device"]) is not str:
        raise SmokeEvidenceError(f"{field} tensor type drift")
    _strict_bool(value["finite"], f"{field}.finite")
    if type(value["shape"]) is not list or any(type(item) is not int or item < 0 for item in value["shape"]):
        raise SmokeEvidenceError(f"{field}.shape contains an invalid dimension")
    if device is not None and value["device"] != device:
        raise SmokeEvidenceError(f"{field}.device drift")
    if value["finite"]:
        _strict_sha(value["logical_sha256"], f"{field}.logical_sha256")
    elif value["logical_sha256"] is not None:
        raise SmokeEvidenceError(f"{field}.logical_sha256 must be null for nonfinite tensor")


def _strict_finite_range(value: Any, field: str) -> None:
    if not isinstance(value, dict) or set(value) != {"min", "max", "finite", "within_0_1"}:
        raise SmokeEvidenceError(f"{field} range schema drift")
    for name in ("min", "max"):
        if type(value[name]) not in (int, float) or isinstance(value[name], bool):
            raise SmokeEvidenceError(f"{field}.{name} type drift")
    _strict_bool(value["finite"], f"{field}.finite")
    _strict_bool(value["within_0_1"], f"{field}.within_0_1")
    if not value["finite"] or not (0 <= float(value["min"]) <= float(value["max"]) <= 1):
        raise SmokeEvidenceError(f"{field} range is not finite [0,1]")


def _validate_input_batch_audit(value: Any, stable_ids: list[str], expected_device: str, context: dict[str, Any]) -> None:
    required = {"schema_version", "context", "runtime_mode", "batch_count", "image_count", "stable_image_ids", "target_labels_model_space", "loader_images", "model_images", "loader_orig_target_sizes", "model_orig_target_sizes", "labels", "boxes", "preprocessing"}
    if not isinstance(value, dict) or set(value) != required:
        raise SmokeEvidenceError("input_batch_audit schema fields are invalid")
    _strict_schema_version(value["schema_version"], "input_batch_audit.schema_version")
    if value["context"] != context or value["runtime_mode"] != context["runtime_mode"]:
        raise SmokeEvidenceError("input_batch_audit context drift")
    if value["runtime_mode"] not in {"synthetic", "real"}:
        raise SmokeEvidenceError("input_batch_audit.runtime_mode is invalid")
    for name, expected in (("batch_count", 1), ("image_count", 2)):
        _strict_int(value[name], f"input_batch_audit.{name}")
        if value[name] != expected:
            raise SmokeEvidenceError(f"input_batch_audit.{name} drift")
    if value["stable_image_ids"] != stable_ids or value["target_labels_model_space"] is not True:
        raise SmokeEvidenceError("input_batch_audit image identity drift")
    for name, device in (("loader_images", "cpu"), ("model_images", expected_device), ("loader_orig_target_sizes", "cpu"), ("model_orig_target_sizes", expected_device)):
        _strict_tensor_audit(value[name], f"input_batch_audit.{name}", device=device)
        if value[name]["finite"] is not True:
            raise SmokeEvidenceError(f"input_batch_audit.{name} must be finite")
    _strict_shape(value["loader_images"]["shape"], [2, 3, 640, 640], "input_batch_audit.loader_images")
    _strict_shape(value["model_images"]["shape"], [2, 3, 640, 640], "input_batch_audit.model_images")
    if value["loader_images"]["dtype"] != "torch.float32" or value["model_images"]["dtype"] != "torch.float32":
        raise SmokeEvidenceError("input_batch_audit image dtype drift")
    for name in ("loader_orig_target_sizes", "model_orig_target_sizes"):
        _strict_shape(value[name]["shape"], [2, 2], f"input_batch_audit.{name}")
        if value[name]["dtype"] != "torch.int64" or value[name]["finite"] is not True:
            raise SmokeEvidenceError(f"input_batch_audit.{name} dtype/finite drift")
    if value["loader_images"]["logical_sha256"] != value["model_images"]["logical_sha256"]:
        raise SmokeEvidenceError("input_batch_audit image logical identity drift")
    if value["loader_orig_target_sizes"]["logical_sha256"] != value["model_orig_target_sizes"]["logical_sha256"]:
        raise SmokeEvidenceError("input_batch_audit target-size logical identity drift")
    labels = value["labels"]
    boxes = value["boxes"]
    if type(labels) is not list or len(labels) != 2 or type(boxes) is not list or len(boxes) != 2:
        raise SmokeEvidenceError("input_batch_audit target count drift")
    for index, row in enumerate(labels):
        required_row = {"index", "dtype", "device", "shape", "count", "min", "max", "finite", "logical_sha256"}
        if not isinstance(row, dict) or set(row) != required_row or row["index"] != index or row["dtype"] != "torch.int64":
            raise SmokeEvidenceError("input_batch_audit label schema drift")
        _strict_shape(row["shape"], [row["count"]], "input_batch_audit.labels")
        if row["device"] != "cpu":
            raise SmokeEvidenceError("input_batch_audit label device drift")
        _strict_int(row["count"], "input_batch_audit.labels.count")
        for field in ("min", "max"):
            _strict_int(row[field], f"input_batch_audit.labels.{field}")
        _strict_bool(row["finite"], "input_batch_audit.labels.finite")
        _strict_sha(row["logical_sha256"], "input_batch_audit.labels.logical_sha256")
        if not row["finite"] or row["min"] < 0 or row["max"] > 9:
            raise SmokeEvidenceError("input_batch_audit label range drift")
    for index, row in enumerate(boxes):
        if not isinstance(row, dict) or set(row) != {"index", "shape", "dtype", "device", "finite", "logical_sha256"} or row["index"] != index:
            raise SmokeEvidenceError("input_batch_audit box schema drift")
        if type(row["dtype"]) is not str or not row["dtype"].startswith("torch.float"):
            raise SmokeEvidenceError("input_batch_audit box dtype drift")
        if row["device"] != "cpu":
            raise SmokeEvidenceError("input_batch_audit box device drift")
        if any(type(item) is not int or item < 0 for item in row["shape"]):
            raise SmokeEvidenceError("input_batch_audit.boxes shape drift")
        if len(row["shape"]) != 2 or row["shape"][1] != 4:
            raise SmokeEvidenceError("input_batch_audit box shape drift")
        if row["shape"][0] != labels[index]["count"]:
            raise SmokeEvidenceError("input_batch_audit box/label count drift")
        _strict_bool(row["finite"], "input_batch_audit.boxes.finite")
        _strict_sha(row["logical_sha256"], "input_batch_audit.boxes.logical_sha256")
    prep = value["preprocessing"]
    required_prep = {"compose", "ordered_pipeline", "resize", "convert_pil_image", "scale_divisor", "normalization", "padding", "color", "value_range", "descriptor_sha256"}
    if not isinstance(prep, dict) or set(prep) != required_prep:
        raise SmokeEvidenceError("input_batch_audit preprocessing schema drift")
    expected_prep = dict(SCIENTIFIC_PREPROCESSING_EXPECTED)
    expected_prep["value_range"] = prep["value_range"]
    expected_prep["descriptor_sha256"] = _descriptor_sha({key: prep[key] for key in expected_prep if key != "descriptor_sha256" and key != "value_range"})
    if {key: prep[key] for key in expected_prep} != expected_prep:
        raise SmokeEvidenceError("input_batch_audit preprocessing drift")
    if prep["descriptor_sha256"] != context["preprocessing_sha256"]:
        raise SmokeEvidenceError("input_batch_audit resize drift")
    _strict_finite_range(prep["value_range"], "input_batch_audit.value_range")


def _validate_cuda_runtime_identity(value: Any, expected_device: str, context: dict[str, Any]) -> None:
    required = {"schema_version", "context", "runtime_mode", "cuda_visible_devices", "device_count", "current_device", "device", "device_name", "device_capability", "torch_version", "torch_cuda_version", "cpu_fallback", "runtime_capture_after_cuda_initialization"}
    if not isinstance(value, dict) or set(value) != required:
        raise SmokeEvidenceError("cuda_runtime_identity schema fields are invalid")
    _strict_schema_version(value["schema_version"], "cuda_runtime_identity.schema_version")
    if value["context"] != context:
        raise SmokeEvidenceError("cuda runtime context drift")
    if value["runtime_mode"] != context["runtime_mode"]:
        raise SmokeEvidenceError("cuda runtime mode cross-binding drift")
    if value["runtime_mode"] not in {"synthetic", "real"} or type(value["cuda_visible_devices"]) is not str:
        raise SmokeEvidenceError("cuda_runtime_identity mode drift")
    _strict_int(value["device_count"], "cuda_runtime_identity.device_count")
    _strict_bool(value["cpu_fallback"], "cuda_runtime_identity.cpu_fallback")
    _strict_bool(value["runtime_capture_after_cuda_initialization"], "cuda_runtime_identity.runtime_capture_after_cuda_initialization")
    expected = SCIENTIFIC_REAL_CUDA_IDENTITY if value["runtime_mode"] == "real" else SCIENTIFIC_SYNTHETIC_CUDA_IDENTITY
    if value["runtime_mode"] not in {"synthetic", "real"} or any(value.get(key) != expected[key] for key in expected):
        raise SmokeEvidenceError("cuda_runtime_identity exact runtime drift")
    if expected_device != value["device"]:
        raise SmokeEvidenceError("cuda_runtime_identity device cross-binding drift")


def _validate_model_identity(
    value: Any,
    config: dict[str, Any],
    smoke_config: dict[str, Any],
    expected_device: str,
    output_audit: dict[str, Any],
    context: dict[str, Any],
    repo_root: Path,
) -> None:
    required = {"schema_version", "context", "runtime_mode", "baseline_id", "model_type", "model_class", "model_module", "backbone", "encoder", "decoder", "decoder_layers", "num_classes", "num_queries", "num_feature_levels", "feature_strides", "hidden_dim", "deformable_sampling_points", "contract_parameters", "parameters", "trainable_parameters", "parameter_tensor_count", "buffer_count", "parameter_inventory", "buffer_inventory", "training", "eval", "device", "pretrained", "checkpoint", "all_parameters_finite", "parameter_state_schema_sha256", "parameter_state_value_sha256", "postprocessor", "output_tensors"}
    if not isinstance(value, dict) or set(value) != required:
        raise SmokeEvidenceError("model_identity schema fields are invalid")
    _strict_schema_version(value["schema_version"], "model_identity.schema_version")
    if value["context"] != context:
        raise SmokeEvidenceError("model identity context drift")
    if value["runtime_mode"] != context["runtime_mode"]:
        raise SmokeEvidenceError("model runtime mode cross-binding drift")
    contract = config["baseline_contract"]
    model_config = smoke_config["model"]
    expected = {"baseline_id": contract["baseline_id"], "backbone": contract["backbone"], "encoder": contract["encoder"], "decoder": contract["decoder"], "decoder_layers": contract["decoder_layers"], "num_classes": contract["num_classes"], "num_queries": contract["num_queries"], "num_feature_levels": contract["num_feature_levels"], "feature_strides": contract["feature_strides"], "hidden_dim": contract["hidden_dim"], "deformable_sampling_points": contract["deformable_sampling_points"], "contract_parameters": contract["visdrone_parameter_count"]}
    for field, expected_value in expected.items():
        if value[field] != expected_value:
            raise SmokeEvidenceError(f"model_identity {field} drift")
    for field in ("decoder_layers", "num_classes", "num_queries", "num_feature_levels", "hidden_dim", "contract_parameters", "parameters", "trainable_parameters", "parameter_tensor_count", "buffer_count"):
        _strict_int(value[field], f"model_identity.{field}")
    for field in ("baseline_id", "num_classes", "num_queries"):
        if model_config.get(field) != expected[field]:
            raise SmokeEvidenceError(f"model identity is not bound to smoke config: {field}")
    _strict_int(model_config.get("parameters"), "smoke_config.model.parameters")
    if model_config.get("parameters") != value["contract_parameters"]:
        raise SmokeEvidenceError("model identity contract parameter binding drift")
    if model_config.get("pretrained") is not False or model_config.get("checkpoint") is not None or model_config.get("nms") is not False or model_config.get("num_top_queries") != 300:
        raise SmokeEvidenceError("model identity is not bound to frozen inference config")
    if value["feature_strides"] != [8, 16, 32] or value["deformable_sampling_points"] != [4, 4, 4] or value["training"] is not False or value["eval"] is not True or value["device"] != expected_device or value["pretrained"] is not False or value["checkpoint"] is not None or value["all_parameters_finite"] is not True:
        raise SmokeEvidenceError("model_identity runtime drift")
    if value["runtime_mode"] == "real":
        if value["model_type"] != "RTDETR" or value["model_class"] != "RTDETR" or value["model_module"] != "src.zoo.rtdetr.rtdetr" or value["postprocessor"]["type"] != "VisDronePostProcessor" or value["postprocessor"]["module"] != "sparse_rtdetr.baseline.postprocessor":
            raise SmokeEvidenceError("real model identity class drift")
    elif value["model_type"] != "synthetic-contract-model" or value["model_class"] != "SyntheticSmokeModel" or value["model_module"] != "sparse_rtdetr.baseline.smoke":
        raise SmokeEvidenceError("synthetic model identity class drift")
    if type(value["parameter_inventory"]) is not list or type(value["buffer_inventory"]) is not list:
        raise SmokeEvidenceError("model state inventory must contain lists")
    rows = value["parameter_inventory"] + value["buffer_inventory"]
    for row, kind in [
        *[(row, "parameter") for row in value["parameter_inventory"]],
        *[(row, "buffer") for row in value["buffer_inventory"]],
    ]:
        if not isinstance(row, dict) or set(row) != {"name", "shape", "dtype", "numel", "requires_grad", "kind", "logical_sha256"} or row["kind"] != kind:
            raise SmokeEvidenceError("model state inventory schema drift")
        if type(row["name"]) is not str or not row["name"] or type(row["shape"]) is not list or any(type(item) is not int or item < 0 for item in row["shape"]):
            raise SmokeEvidenceError("model state inventory shape drift")
        if type(row["dtype"]) is not str or type(row["numel"]) is not int or row["numel"] < 0 or type(row["requires_grad"]) is not bool:
            raise SmokeEvidenceError("model state inventory type drift")
        product = 1
        for dimension in row["shape"]:
            product *= dimension
        if product != row["numel"]:
            raise SmokeEvidenceError("model state inventory numel drift")
        _strict_sha(row["logical_sha256"], "model state logical_sha256")
    if value["parameters"] != sum(row["numel"] for row in value["parameter_inventory"]):
        raise SmokeEvidenceError("model observed parameter count drift")
    if value["trainable_parameters"] != sum(row["numel"] for row in value["parameter_inventory"] if row["requires_grad"]):
        raise SmokeEvidenceError("model trainable parameter count drift")
    if value["parameter_tensor_count"] != len(value["parameter_inventory"]) or value["buffer_count"] != len(value["buffer_inventory"]):
        raise SmokeEvidenceError("model tensor count drift")
    names = [row["name"] for row in rows]
    if len(names) != len(set(names)):
        raise SmokeEvidenceError("model state inventory names are not unique")
    if value["runtime_mode"] == "synthetic":
        empty_sha = _empty_state_sha()
        if value["parameters"] != 0 or value["trainable_parameters"] != 0 or value["parameter_tensor_count"] != 0 or value["buffer_count"] != 0 or value["parameter_inventory"] != [] or value["buffer_inventory"] != []:
            raise SmokeEvidenceError("synthetic model state must be empty")
        if value["parameter_state_schema_sha256"] != empty_sha or value["parameter_state_value_sha256"] != empty_sha:
            raise SmokeEvidenceError("synthetic model empty state hash drift")
    else:
        expected_parameters = (
            value["contract_parameters"],
            contract["visdrone_parameter_count"],
            config["baseline_contract"]["visdrone_parameter_count"],
            model_config["parameters"],
            VISDRONE_BASELINE_PARAMETER_COUNT,
        )
        if any(type(expected) is not int or expected != VISDRONE_BASELINE_PARAMETER_COUNT for expected in expected_parameters):
            raise SmokeEvidenceError("real model parameter contract is not frozen")
        if any(value["parameters"] != expected for expected in expected_parameters):
            raise SmokeEvidenceError("real model observed parameter count is not bound to baseline contract")
        expected_parameter_inventory, expected_buffer_inventory = _canonical_real_state_inventory(repo_root)
        if value["parameter_inventory"] != expected_parameter_inventory or value["buffer_inventory"] != expected_buffer_inventory:
            raise SmokeEvidenceError("real model state inventory is not the frozen R18 identity")
        if value["parameter_tensor_count"] != len(expected_parameter_inventory) or value["buffer_count"] != len(expected_buffer_inventory):
            raise SmokeEvidenceError("real model state tensor count is not frozen")
    schema_sha, value_sha = _state_hashes_from_inventory(value["parameter_inventory"], value["buffer_inventory"])
    if value["parameter_state_schema_sha256"] != schema_sha or value["parameter_state_value_sha256"] != value_sha:
        raise SmokeEvidenceError("model state inventory hash drift")
    _strict_sha(value["parameter_state_schema_sha256"], "model_identity.parameter_state_schema_sha256")
    _strict_sha(value["parameter_state_value_sha256"], "model_identity.parameter_state_value_sha256")
    post = value["postprocessor"]
    if not isinstance(post, dict) or set(post) != {"type", "module", "num_top_queries", "nms"} or type(post["type"]) is not str or type(post["module"]) is not str or type(post["num_top_queries"]) is not int or type(post["nms"]) is not bool or post["num_top_queries"] != 300 or post["nms"] is not False:
        raise SmokeEvidenceError("model_identity postprocessor drift")
    if value["runtime_mode"] == "real":
        if post["type"] != "VisDronePostProcessor":
            raise SmokeEvidenceError("real model postprocessor type drift")
    elif post["type"] != "SyntheticSmokePostProcessor":
        raise SmokeEvidenceError("synthetic model postprocessor type drift")
    if value["output_tensors"] != output_audit:
        raise SmokeEvidenceError("model_identity output tensor binding drift")


def _validate_source_identity(value: Any, config: dict[str, Any], repo_root: Path, context: dict[str, Any]) -> None:
    required = {"schema_version", "context", "baseline_id", "implementation", "source_allowlist", "source_rows_canonical_sha256", "environment_lock", "environment_rows_canonical_sha256", "baseline_config", "upstream_manifest", "vendor", "r3_binding"}
    if not isinstance(value, dict) or set(value) != required:
        raise SmokeEvidenceError("source_identity schema fields are invalid")
    _strict_schema_version(value["schema_version"], "source_identity.schema_version")
    if value["context"] != context:
        raise SmokeEvidenceError("source identity context drift")
    if value["baseline_id"] != config["baseline_contract"]["baseline_id"] or value["implementation"] != "vendored RT-DETRv2 PyTorch":
        raise SmokeEvidenceError("source_identity baseline drift")
    rows = value["source_allowlist"]
    if not isinstance(rows, list) or [x.get("relative_path") for x in rows] != list(SCIENTIFIC_SOURCE_ALLOWLIST):
        raise SmokeEvidenceError("source_identity allowlist drift")
    for row, relative in zip(rows, SCIENTIFIC_SOURCE_ALLOWLIST):
        if set(row) != {"relative_path", "size_bytes", "sha256"} or row["relative_path"] != relative:
            raise SmokeEvidenceError("source_identity row schema drift")
        path = repo_root / relative
        if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1 or row["size_bytes"] != path.stat().st_size or row["sha256"] != sha256_file(path):
            raise SmokeEvidenceError(f"source_identity source drift: {relative}")
        _strict_int(row["size_bytes"], "source_identity.size_bytes"); _strict_sha(row["sha256"], "source_identity.sha256")
    if value["source_rows_canonical_sha256"] != sha256_bytes(canonical_json_bytes(rows)):
        raise SmokeEvidenceError("source_identity source rows SHA drift")
    environment_rows = value["environment_lock"]
    if not isinstance(environment_rows, list) or [row.get("relative_path") for row in environment_rows] != list(SCIENTIFIC_ENVIRONMENT_ALLOWLIST):
        raise SmokeEvidenceError("source_identity environment allowlist drift")
    for row, relative in zip(environment_rows, SCIENTIFIC_ENVIRONMENT_ALLOWLIST):
        path = repo_root / relative
        if set(row) != {"relative_path", "size_bytes", "sha256"} or row["relative_path"] != relative or path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1 or row["size_bytes"] != path.stat().st_size or row["sha256"] != sha256_file(path):
            raise SmokeEvidenceError(f"source_identity environment drift: {relative}")
        _strict_int(row["size_bytes"], "source_identity.environment.size_bytes")
        _strict_sha(row["sha256"], "source_identity.environment.sha256")
    if value["environment_rows_canonical_sha256"] != sha256_bytes(canonical_json_bytes(environment_rows)):
        raise SmokeEvidenceError("source_identity environment rows SHA drift")
    baseline = value["baseline_config"]
    baseline_path = repo_root / SCIENTIFIC_BASELINE_CONFIG
    if not baseline_path.is_file():
        raise SmokeEvidenceError("source_identity baseline config is missing")
    if baseline != {"relative_path": SCIENTIFIC_BASELINE_CONFIG, "size_bytes": baseline_path.stat().st_size, "sha256": sha256_file(baseline_path)}:
        raise SmokeEvidenceError("source_identity baseline config drift")
    manifest = value["upstream_manifest"]
    manifest_path = repo_root / SCIENTIFIC_MANIFEST
    if context["runtime_mode"] == "synthetic" and manifest is not None:
        raise SmokeEvidenceError("synthetic source identity must not claim a manifest")
    if context["runtime_mode"] == "real" and not manifest_path.is_file():
        raise SmokeEvidenceError("source_identity upstream manifest is missing")
    if context["runtime_mode"] == "real":
        manifest_value = json.loads(manifest_path.read_bytes().decode("utf-8"))
        expected_manifest = {"relative_path": SCIENTIFIC_MANIFEST, "size_bytes": manifest_path.stat().st_size, "sha256": sha256_file(manifest_path), "upstream_commit": SCIENTIFIC_UPSTREAM_COMMIT, "upstream_root_tree": SCIENTIFIC_UPSTREAM_ROOT_TREE, "upstream_subtree": SCIENTIFIC_UPSTREAM_SUBTREE}
        if manifest != expected_manifest or manifest_value.get("canonical_inventory_sha256") != SCIENTIFIC_VENDOR_INVENTORY_SHA256 or manifest_value.get("file_count") != SCIENTIFIC_VENDOR_FILE_COUNT or manifest_value.get("total_size_bytes") != SCIENTIFIC_VENDOR_TOTAL_BYTES:
            raise SmokeEvidenceError("source_identity upstream manifest drift")
    vendor = value["vendor"]
    expected_vendor = {"r18_config": config["vendor"]["r18_config"], "r18_config_sha256": config["vendor"]["r18_config_sha256"], "r18_include": config["vendor"]["r18_include"], "r18_include_sha256": config["vendor"]["r18_include_sha256"]}
    if context["runtime_mode"] == "real":
        expected_vendor.update({"canonical_inventory_sha256": SCIENTIFIC_VENDOR_INVENTORY_SHA256, "file_count": SCIENTIFIC_VENDOR_FILE_COUNT, "total_size_bytes": SCIENTIFIC_VENDOR_TOTAL_BYTES})
    if vendor != expected_vendor:
        raise SmokeEvidenceError("source_identity vendor drift")
    for path_key, sha_key in (("r18_config", "r18_config_sha256"), ("r18_include", "r18_include_sha256")):
        vendor_path = repo_root / vendor[path_key]
        if vendor_path.is_symlink() or not vendor_path.is_file() or sha256_file(vendor_path) != vendor[sha_key]:
            raise SmokeEvidenceError(f"source_identity vendor file drift: {path_key}")
    expected_r3 = {
        "artifact_root": config["r3_binding"]["artifact_root"],
        "runtime_mode": context["runtime_mode"],
        "artifact_validation_mode": "real_r3_metadata" if context["runtime_mode"] == "real" else "synthetic_tracked_contract_only",
        **{key: config["r3_binding"][key] for key in (
            "completion_sha256", "artifact_inventory_sha256", "entry_canonical_inventory_sha256",
            "config_sha256", "category_contract_sha256", "source_identity_sha256",
        )},
    }
    if value["r3_binding"] != expected_r3:
        raise SmokeEvidenceError("source_identity R3 binding drift")


def _validate_model_output_audit(value: Any, expected_device: str) -> None:
    required = {"pred_logits", "pred_boxes"}
    if not isinstance(value, dict) or set(value) != required:
        raise SmokeEvidenceError("model_output_audit schema fields are invalid")
    _strict_tensor_audit(value["pred_logits"], "model_output_audit.pred_logits", device=expected_device)
    _strict_tensor_audit(value["pred_boxes"], "model_output_audit.pred_boxes", device=expected_device)
    _strict_shape(value["pred_logits"]["shape"], [2, 300, 10], "model_output_audit.pred_logits")
    _strict_shape(value["pred_boxes"]["shape"], [2, 300, 4], "model_output_audit.pred_boxes")
    if value["pred_logits"]["dtype"] != "torch.float32" or value["pred_boxes"]["dtype"] != "torch.float32":
        raise SmokeEvidenceError("model output dtype drift")
    if value["pred_logits"]["finite"] is not True or value["pred_boxes"]["finite"] is not True:
        raise SmokeEvidenceError("model output finite drift")


def _validate_postprocess_audit(value: Any, stable_ids: list[str], expected_device: str, total_predictions: int) -> None:
    if not isinstance(value, dict) or set(value) != {"images", "total_predictions", "nms", "threshold"}:
        raise SmokeEvidenceError("postprocess_audit schema fields are invalid")
    if value["total_predictions"] != total_predictions or total_predictions != 600 or value["nms"] is not False or value["threshold"] is not None:
        raise SmokeEvidenceError("postprocess contract drift")
    images = value["images"]
    if type(images) is not list or len(images) != 2:
        raise SmokeEvidenceError("postprocess image count drift")
    for index, row in enumerate(images):
        required = {"stable_image_id", "labels_min", "labels_max", "prediction_count", "labels", "boxes", "scores"}
        if not isinstance(row, dict) or set(row) != required or row["stable_image_id"] != stable_ids[index]:
            raise SmokeEvidenceError("postprocess image schema drift")
        for field in ("labels_min", "labels_max", "prediction_count"):
            _strict_int(row[field], f"postprocess_audit.images[{index}].{field}")
        if row["prediction_count"] != 300 or not 1 <= row["labels_min"] <= row["labels_max"] <= 10:
            raise SmokeEvidenceError("postprocess prediction contract drift")
        _strict_tensor_audit(row["labels"], f"postprocess_audit.images[{index}].labels", device=expected_device)
        _strict_tensor_audit(row["boxes"], f"postprocess_audit.images[{index}].boxes", device=expected_device)
        _strict_tensor_audit(row["scores"], f"postprocess_audit.images[{index}].scores", device=expected_device)
        _strict_shape(row["labels"]["shape"], [300], "postprocess labels")
        _strict_shape(row["boxes"]["shape"], [300, 4], "postprocess boxes")
        _strict_shape(row["scores"]["shape"], [300], "postprocess scores")
        if row["labels"]["dtype"] != "torch.int64" or not row["boxes"]["dtype"].startswith("torch.float") or not row["scores"]["dtype"].startswith("torch.float"):
            raise SmokeEvidenceError("postprocess dtype drift")


def _validate_call_audit(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != set(COMPLETION_COUNTERS) or any(type(value.get(field)) is not int for field in COMPLETION_COUNTERS):
        raise SmokeEvidenceError("call_audit schema fields are invalid")
    if value != COMPLETION_COUNTERS:
        raise SmokeEvidenceError("call_audit counter drift")


def _tensor_range(tensor: Any) -> dict[str, object]:
    import torch

    detached = tensor.detach()
    finite = bool(torch.isfinite(detached).all())
    if not finite or detached.numel() == 0:
        return {"min": 0.0, "max": 0.0, "finite": finite, "within_0_1": False}
    minimum = float(detached.min().item())
    maximum = float(detached.max().item())
    return {"min": minimum, "max": maximum, "finite": True, "within_0_1": 0.0 <= minimum <= maximum <= 1.0}


def _target_tensor_audit(tensor: Any, index: int) -> dict[str, object]:
    audit = tensor_audit(tensor)
    audit.update({
        "index": index,
        "count": int(tensor.numel()),
        "min": int(tensor.min().item()) if tensor.numel() else 0,
        "max": int(tensor.max().item()) if tensor.numel() else 0,
    })
    return audit


def tensor_audit(tensor: Any) -> dict[str, object]:
    """Record a tensor without retaining tensor objects in evidence."""

    import torch

    if not torch.is_tensor(tensor):
        raise SmokeEvidenceError("scientific evidence value is not a tensor")
    detached = tensor.detach()
    finite = bool(torch.isfinite(detached).all())
    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "finite": finite,
        "logical_sha256": sha256_bytes(detached.cpu().contiguous().numpy().tobytes()) if finite else None,
    }


def build_input_batch_audit(
    batch: dict[str, Any],
    model_images: Any,
    model_orig_target_sizes: Any,
    *,
    runtime_mode: str,
    preprocessing: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, object]:
    images = batch["images"]
    sizes = batch["orig_target_sizes"]
    targets = batch["targets"]
    labels = [_target_tensor_audit(target["labels"], index) for index, target in enumerate(targets)]
    boxes = []
    for index, target in enumerate(targets):
        target_boxes = target.get("boxes")
        if target_boxes is None:
            import torch
            target_boxes = torch.empty((0, 4), dtype=torch.float32, device=images.device)
        audit = tensor_audit(target_boxes)
        audit["index"] = index
        boxes.append(audit)
    return {
        "schema_version": SCIENTIFIC_SCHEMA_VERSION,
        "context": context,
        "runtime_mode": runtime_mode,
        "batch_count": 1,
        "image_count": 2,
        "stable_image_ids": list(batch["stable_image_ids"]),
        "target_labels_model_space": True,
        "loader_images": tensor_audit(images),
        "model_images": tensor_audit(model_images),
        "loader_orig_target_sizes": tensor_audit(sizes),
        "model_orig_target_sizes": tensor_audit(model_orig_target_sizes),
        "labels": labels,
        "boxes": boxes,
        "preprocessing": {**preprocessing, "value_range": _tensor_range(images), "descriptor_sha256": _descriptor_sha(preprocessing)},
    }


def _state_hashes(model: Any, *, synthetic: bool) -> tuple[int, int, int, str, str, bool]:
    parameter_inventory, buffer_inventory, all_finite = _state_inventory(model)
    if synthetic and (parameter_inventory or buffer_inventory):
        raise SmokeEvidenceError("synthetic model state must be empty")
    schema_sha, value_sha = _state_hashes_from_inventory(parameter_inventory, buffer_inventory)
    return (
        sum(row["numel"] for row in parameter_inventory),
        sum(row["numel"] for row in parameter_inventory if row["requires_grad"]),
        len(parameter_inventory),
        schema_sha,
        value_sha,
        all_finite,
    )


def _synthetic_state_hashes(contract: dict[str, Any]) -> tuple[int, int, int, str, str, bool]:
    rows: list[dict[str, Any]] = []
    digest = sha256_bytes(canonical_json_bytes(rows))
    return 0, 0, 0, digest, digest, True


def build_model_identity(repo_root: str | Path, smoke_config: dict[str, Any], model: Any, postprocessor: Any, outputs: dict[str, Any], *, runtime_mode: str, device: str, context: dict[str, Any]) -> dict[str, object]:
    baseline_path = Path(repo_root).resolve() / SCIENTIFIC_BASELINE_CONFIG
    baseline = json.loads(baseline_path.read_bytes().decode("utf-8"))
    contract = baseline["baseline_contract"]
    parameters, trainable, parameter_tensors, schema_sha, value_sha, finite = _state_hashes(model, synthetic=runtime_mode == "synthetic")
    parameter_inventory, buffer_inventory, _ = _state_inventory(model)
    if runtime_mode == "synthetic" and (parameter_inventory or buffer_inventory):
        raise SmokeEvidenceError("synthetic model state must be empty")
    return {
        "schema_version": SCIENTIFIC_SCHEMA_VERSION,
        "context": context,
        "runtime_mode": runtime_mode,
        "baseline_id": contract["baseline_id"],
        "model_type": "synthetic-contract-model" if runtime_mode == "synthetic" else baseline["model"]["type"],
        "model_class": type(model).__name__,
        "model_module": type(model).__module__,
        "backbone": contract["backbone"],
        "encoder": contract["encoder"],
        "decoder": contract["decoder"],
        "decoder_layers": contract["decoder_layers"],
        "num_classes": contract["num_classes"],
        "num_queries": contract["num_queries"],
        "num_feature_levels": contract["num_feature_levels"],
        "feature_strides": list(contract["feature_strides"]),
        "hidden_dim": contract["hidden_dim"],
        "deformable_sampling_points": list(contract["deformable_sampling_points"]),
        "contract_parameters": contract["visdrone_parameter_count"],
        "parameters": parameters,
        "trainable_parameters": trainable,
        "parameter_tensor_count": parameter_tensors,
        "buffer_count": sum(1 for _ in model.named_buffers()) if hasattr(model, "named_buffers") else 0,
        "training": bool(getattr(model, "training", False)),
        "eval": not bool(getattr(model, "training", False)),
        "device": device,
        "pretrained": contract["pretrained"],
        "checkpoint": contract["checkpoint"],
        "all_parameters_finite": finite,
        "parameter_state_schema_sha256": schema_sha,
        "parameter_state_value_sha256": value_sha,
        "parameter_inventory": parameter_inventory,
        "buffer_inventory": buffer_inventory,
        "postprocessor": {"type": type(postprocessor).__name__, "module": type(postprocessor).__module__, "num_top_queries": int(getattr(postprocessor, "num_top_queries", 300)), "nms": bool(getattr(postprocessor, "nms", False))},
        "output_tensors": outputs,
    }


def build_source_identity(repo_root: str | Path, smoke_config: dict[str, Any], *, runtime_mode: str, context: dict[str, Any]) -> dict[str, object]:
    root = Path(repo_root).resolve()
    rows = []
    for relative in SCIENTIFIC_SOURCE_ALLOWLIST:
        path = root / relative
        rows.append({"relative_path": relative, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    environment_rows = []
    for relative in SCIENTIFIC_ENVIRONMENT_ALLOWLIST:
        path = root / relative
        environment_rows.append({"relative_path": relative, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    baseline_path = root / SCIENTIFIC_BASELINE_CONFIG
    manifest_path = root / SCIENTIFIC_MANIFEST
    manifest = json.loads(manifest_path.read_bytes().decode("utf-8")) if runtime_mode == "real" else None
    baseline = json.loads((root / SCIENTIFIC_BASELINE_CONFIG).read_bytes().decode("utf-8"))
    vendor = baseline["vendor"]
    return {
        "schema_version": SCIENTIFIC_SCHEMA_VERSION,
        "context": context,
        "baseline_id": baseline["baseline_contract"]["baseline_id"],
        "implementation": "vendored RT-DETRv2 PyTorch",
        "source_allowlist": rows,
        "source_rows_canonical_sha256": sha256_bytes(canonical_json_bytes(rows)),
        "environment_lock": environment_rows,
        "environment_rows_canonical_sha256": sha256_bytes(canonical_json_bytes(environment_rows)),
        "baseline_config": {"relative_path": SCIENTIFIC_BASELINE_CONFIG, "size_bytes": baseline_path.stat().st_size, "sha256": sha256_file(baseline_path)},
        "upstream_manifest": {"relative_path": SCIENTIFIC_MANIFEST, "size_bytes": manifest_path.stat().st_size, "sha256": sha256_file(manifest_path), "upstream_commit": manifest["upstream_commit"], "upstream_root_tree": manifest["upstream_root_tree"], "upstream_subtree": manifest["upstream_subtree"]} if runtime_mode == "real" else None,
        "vendor": {"canonical_inventory_sha256": manifest["canonical_inventory_sha256"], "file_count": manifest["file_count"], "total_size_bytes": manifest["total_size_bytes"], "r18_config": vendor["r18_config"], "r18_config_sha256": vendor["r18_config_sha256"], "r18_include": vendor["r18_include"], "r18_include_sha256": vendor["r18_include_sha256"]} if runtime_mode == "real" else {"r18_config": vendor["r18_config"], "r18_config_sha256": vendor["r18_config_sha256"], "r18_include": vendor["r18_include"], "r18_include_sha256": vendor["r18_include_sha256"]},
        "r3_binding": {
            "artifact_root": baseline["r3_binding"]["artifact_root"],
            "runtime_mode": runtime_mode,
            "artifact_validation_mode": "real_r3_metadata" if runtime_mode == "real" else "synthetic_tracked_contract_only",
            **{key: baseline["r3_binding"][key] for key in (
                "completion_sha256", "artifact_inventory_sha256", "entry_canonical_inventory_sha256",
                "config_sha256", "category_contract_sha256", "source_identity_sha256",
            )},
        },
    }


def build_cuda_runtime_identity(runtime: dict[str, Any], *, runtime_mode: str, context: dict[str, Any]) -> dict[str, object]:
    if runtime_mode == "synthetic":
        return {"schema_version": SCIENTIFIC_SCHEMA_VERSION, "context": context, "runtime_mode": "synthetic", **SCIENTIFIC_SYNTHETIC_CUDA_IDENTITY}
    import torch
    actual = {"cuda_visible_devices": "0", "device_count": int(torch.cuda.device_count()), "current_device": int(torch.cuda.current_device()), "device": "cuda:0", "device_name": torch.cuda.get_device_name(0), "device_capability": list(torch.cuda.get_device_capability(0)), "torch_version": torch.__version__, "torch_cuda_version": torch.version.cuda, "cpu_fallback": False, "runtime_capture_after_cuda_initialization": True}
    return {"schema_version": SCIENTIFIC_SCHEMA_VERSION, "context": context, "runtime_mode": "real", **actual}


def canonical_argv(argv: list[str]) -> list[str]:
    if not isinstance(argv, list) or not argv or any(type(item) is not str or not item for item in argv):
        raise SmokeEvidenceError("child argv must be a non-empty list of non-empty strings")
    return [str(Path(argv[0]).resolve()), *argv[1:]]


def argv_sha256(argv: list[str]) -> str:
    return sha256_bytes(canonical_json_bytes(canonical_argv(argv)))


def atomic_write(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise SmokeEvidenceError(f"evidence file already exists: {path.name}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def file_ref(root: Path, name: str) -> dict[str, object]:
    path = root / name
    if path.is_symlink() or not path.is_file():
        return {"present": False, "relative_path": name}
    return {
        "present": True,
        "relative_path": name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def inventory(root: Path, excluded: frozenset[str]) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if path.name in excluded:
            continue
        if path.name.startswith(".") or path.is_symlink() or not path.is_file():
            raise SmokeEvidenceError(f"invalid smoke evidence entry: {path.name}")
        if path.stat().st_nlink != 1:
            raise SmokeEvidenceError(f"hardlink smoke evidence entry: {path.name}")
        rows.append({
            "relative_path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return {
        "schema_version": 1,
        "excluded_from_inventory": sorted(excluded),
        "artifacts": rows,
        "canonical_inventory_sha256": sha256_bytes(canonical_json_bytes(rows)),
    }


def _read_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SmokeEvidenceError(f"missing regular evidence file: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SmokeEvidenceError(f"invalid evidence JSON: {path.name}") from exc
    if not isinstance(value, dict):
        raise SmokeEvidenceError(f"evidence JSON must be an object: {path.name}")
    return value


def _infer_repo_root(output_root: Path) -> Path:
    for candidate in (output_root, *output_root.parents):
        if (candidate / SCIENTIFIC_BASELINE_CONFIG).is_file():
            return candidate.resolve()
    return Path(__file__).resolve().parents[3]


def _strict_sha(value: Any, field: str) -> None:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None:
        raise SmokeEvidenceError(f"{field} must be a lowercase SHA-256 string")


def _validate_entry_invocation(
    root: Path,
    config_sha256: str,
    expected_identity: dict[str, Any] | None,
    config_value: dict[str, Any],
) -> dict[str, Any]:
    invocation = _read_object(root / "invocation.json")
    if type(invocation.get("schema_version")) is not int or invocation["schema_version"] != 1:
        raise SmokeEvidenceError("entry invocation schema version is invalid")
    expected_smoke_id = config_value.get("smoke_id", "rtdetrv2_r18_visdrone_baseline_smoke_v1")
    if invocation.get("mode") not in {"synthetic", "real"} or invocation.get("smoke_id") != expected_smoke_id:
        raise SmokeEvidenceError("entry invocation identity is invalid")
    _strict_sha(invocation.get("config_sha256"), "entry invocation config_sha256")
    if invocation.get("config_sha256") != config_sha256:
        raise SmokeEvidenceError("entry invocation config SHA mismatch")
    if expected_identity is not None:
        required_identity = {"smoke_id", "config_relative_path", "config_canonical_sha256"}
        if not required_identity.issubset(expected_identity):
            raise SmokeEvidenceError("entry expected config identity is incomplete")
        if type(expected_identity.get("smoke_id")) is not str or not expected_identity["smoke_id"]:
            raise SmokeEvidenceError("entry expected smoke_id is invalid")
        if type(expected_identity.get("config_relative_path")) is not str or not expected_identity["config_relative_path"]:
            raise SmokeEvidenceError("entry expected config relative path is invalid")
        _strict_sha(expected_identity.get("config_canonical_sha256"), "entry expected config_canonical_sha256")
        if invocation.get("smoke_id") != expected_identity["smoke_id"]:
            raise SmokeEvidenceError("entry smoke_id binding mismatch")
        if invocation.get("config_relative_path") != expected_identity["config_relative_path"]:
            raise SmokeEvidenceError("entry config relative path binding mismatch")
        if invocation.get("config_sha256") != expected_identity["config_canonical_sha256"]:
            raise SmokeEvidenceError("entry canonical config SHA binding mismatch")
        for field in (
            "nonce",
            "launcher_pid",
            "child_pid",
            "child_ppid",
            "child_argv_sha256",
            "handoff_receipt_sha256",
            "handoff_receipt_relative_path",
        ):
            if field in expected_identity and invocation.get(field) != expected_identity.get(field):
                raise SmokeEvidenceError(f"entry invocation identity mismatch: {field}")
        if "handoff_receipt_sha256" in expected_identity:
            _strict_sha(invocation.get("handoff_receipt_sha256"), "entry invocation handoff_receipt_sha256")
    if invocation.get("mode") == "real" or (
        "config_relative_path" in invocation
        and isinstance(config_value.get("runtime"), dict)
        and config_value["runtime"].get("config_relative_path")
    ):
        if type(config_value.get("smoke_id")) is not str or not config_value["smoke_id"]:
            raise SmokeEvidenceError("entry smoke_id is invalid")
        runtime = config_value.get("runtime")
        if not isinstance(runtime, dict):
            raise SmokeEvidenceError("entry runtime binding is missing")
        if type(runtime.get("config_relative_path")) is not str or not runtime["config_relative_path"]:
            raise SmokeEvidenceError("entry config relative path is invalid")
        if type(invocation.get("config_relative_path")) is not str or not invocation["config_relative_path"]:
            raise SmokeEvidenceError("entry invocation config relative path is invalid")
        if invocation.get("config_relative_path") != runtime.get("config_relative_path"):
            raise SmokeEvidenceError("entry config relative path mismatch")
        if type(invocation.get("config_size_bytes")) is not int or invocation["config_size_bytes"] != root.joinpath("config.json").stat().st_size:
            raise SmokeEvidenceError("entry config size binding mismatch")
    return invocation


def validate_entry_output(
    root: Path,
    expected_identity: dict[str, Any] | None = None,
    *,
    repo_root: Path | None = None,
) -> dict[str, object]:
    """Validate a completed child output without accepting exit code alone."""

    completion_path = root / "completion.json"
    inventory_path = root / "artifact_inventory.json"
    completion = _read_object(completion_path)
    inventory_value = _read_object(inventory_path)
    if set(completion) != COMPLETION_KEYS:
        raise SmokeEvidenceError("completion schema fields are invalid")
    if type(completion.get("schema_version")) is not int or completion["schema_version"] != 1:
        raise SmokeEvidenceError("completion schema version is invalid")
    if completion.get("status") != "COMPLETED" or completion.get("mode") != "smoke":
        raise SmokeEvidenceError("completion status or mode is invalid")
    for field, expected in COMPLETION_BOOLEAN_FIELDS.items():
        if type(completion.get(field)) is not bool or completion[field] is not expected:
            raise SmokeEvidenceError(f"completion boolean field is invalid: {field}")
    for field, expected in COMPLETION_COUNTERS.items():
        if type(completion.get(field)) is not int or completion[field] != expected:
            raise SmokeEvidenceError(f"completion counter is invalid: {field}")
    if type(completion.get("total_predictions")) is not int or completion["total_predictions"] != 600:
        raise SmokeEvidenceError("completion total_predictions is invalid")
    _strict_sha(completion.get("config_sha256"), "completion config_sha256")
    config_path = root / "config.json"
    if config_path.is_symlink() or not config_path.is_file():
        raise SmokeEvidenceError("entry config is missing or not regular")
    config_value = _read_object(config_path)
    config_sha256 = sha256_file(config_path)
    if completion["config_sha256"] != config_sha256:
        raise SmokeEvidenceError("completion config SHA mismatch")
    invocation = _validate_entry_invocation(root, config_sha256, expected_identity, config_value)
    if completion.get("artifact_inventory_sha256") != sha256_file(inventory_path):
        raise SmokeEvidenceError("completion inventory SHA mismatch")
    if inventory_value.get("schema_version") != 1 or inventory_value.get("excluded_from_inventory") != sorted(ENTRY_INVENTORY_EXCLUDED):
        raise SmokeEvidenceError("entry inventory schema mismatch")
    rows = inventory_value.get("artifacts")
    if not isinstance(rows, list) or rows != sorted(rows, key=lambda row: row.get("relative_path", "") if isinstance(row, dict) else ""):
        raise SmokeEvidenceError("entry inventory rows are invalid")
    if inventory_value.get("canonical_inventory_sha256") != sha256_bytes(canonical_json_bytes(rows)):
        raise SmokeEvidenceError("entry inventory canonical SHA mismatch")
    names: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"relative_path", "size_bytes", "sha256"}:
            raise SmokeEvidenceError("entry inventory row schema is invalid")
        name = row.get("relative_path")
        if not isinstance(name, str) or not name or "/" in name or "\\" in name or name in ENTRY_INVENTORY_EXCLUDED or name in names:
            raise SmokeEvidenceError("entry inventory path is invalid")
        artifact = root / name
        if artifact.is_symlink() or not artifact.is_file() or artifact.stat().st_nlink != 1:
            raise SmokeEvidenceError(f"entry inventory artifact is not regular: {name}")
        if type(row.get("size_bytes")) is not int or row["size_bytes"] != artifact.stat().st_size:
            raise SmokeEvidenceError(f"entry inventory size binding mismatch: {name}")
        _strict_sha(row.get("sha256"), f"entry inventory sha256: {name}")
        if row["sha256"] != sha256_file(artifact):
            raise SmokeEvidenceError(f"entry inventory binding mismatch: {name}")
        names.add(name)
    actual = {path.name for path in root.iterdir() if path.name not in ENTRY_INVENTORY_EXCLUDED}
    if actual != names:
        raise SmokeEvidenceError("entry inventory does not cover every output file")
    if not ENTRY_REQUIRED_ARTIFACTS.issubset(names):
        raise SmokeEvidenceError("entry inventory is missing required evidence")
    config_row = next(row for row in rows if row["relative_path"] == "config.json")
    if config_row["sha256"] != config_sha256 or invocation.get("config_sha256") != config_sha256:
        raise SmokeEvidenceError("entry config SHA binding is incomplete")
    repo_root = _infer_repo_root(root) if repo_root is None else Path(repo_root).resolve()
    baseline_path = repo_root / SCIENTIFIC_BASELINE_CONFIG
    if not baseline_path.is_file():
        raise SmokeEvidenceError("baseline config is unavailable for scientific validation")
    baseline_config = _read_object(baseline_path)
    scientific = {
        name: _read_object(root / name)
        for name in (
            "input_batch_audit.json",
            "cuda_runtime_identity.json",
            "model_identity.json",
            "source_identity.json",
            "data_binding_audit.json",
            "model_output_audit.json",
            "postprocess_audit.json",
            "call_audit.json",
            "image_selection_audit.json",
        )
    }
    input_audit = scientific["input_batch_audit.json"]
    stable_ids = input_audit.get("stable_image_ids")
    if type(stable_ids) is not list or any(type(item) is not str or not item for item in stable_ids):
        raise SmokeEvidenceError("input batch stable IDs are invalid")
    expected_config_records = config_value.get("image_selection", {}).get("records")
    if not isinstance(expected_config_records, list) or [record.get("stable_image_id") for record in expected_config_records] != stable_ids:
        raise SmokeEvidenceError("input batch IDs are not bound to config selection")
    expected_device = "cpu" if invocation["mode"] == "synthetic" else "cuda:0"
    preprocessing = input_audit.get("preprocessing")
    data_binding = scientific["data_binding_audit.json"]
    source_value = scientific["source_identity.json"]
    context = input_audit.get("context")
    if not isinstance(context, dict) or set(context) != {"schema_version", "runtime_mode", "expected_device", "preprocessing_sha256", "data_binding_sha256", "source_rows_canonical_sha256", "environment_rows_canonical_sha256"}:
        raise SmokeEvidenceError("scientific context schema is invalid")
    if context["schema_version"] != SCIENTIFIC_SCHEMA_VERSION or context["runtime_mode"] != invocation["mode"] or context["expected_device"] != expected_device:
        raise SmokeEvidenceError("scientific context runtime drift")
    if context["data_binding_sha256"] != _data_binding_sha(data_binding):
        raise SmokeEvidenceError("scientific context data binding drift")
    _validate_data_binding(data_binding, config_value, repo_root, context)
    _validate_input_batch_audit(input_audit, stable_ids, expected_device, context)
    _validate_cuda_runtime_identity(scientific["cuda_runtime_identity.json"], expected_device, context)
    _validate_model_output_audit(scientific["model_output_audit.json"], expected_device)
    _validate_model_identity(scientific["model_identity.json"], baseline_config, config_value, expected_device, scientific["model_output_audit.json"], context, repo_root)
    _validate_source_identity(source_value, baseline_config, repo_root, context)
    if source_value["source_rows_canonical_sha256"] != context["source_rows_canonical_sha256"] or source_value["environment_rows_canonical_sha256"] != context["environment_rows_canonical_sha256"]:
        raise SmokeEvidenceError("scientific context source binding drift")
    _validate_call_audit(scientific["call_audit.json"])
    _validate_postprocess_audit(scientific["postprocess_audit.json"], stable_ids, expected_device, completion["total_predictions"])
    image_selection = scientific["image_selection_audit.json"]
    if not isinstance(image_selection, dict) or set(image_selection) != {"records", "verified_from_manifest", "selection_policy"} or image_selection["verified_from_manifest"] is not (invocation["mode"] == "real") or image_selection["selection_policy"] != "explicit_coco_image_ids_only":
        raise SmokeEvidenceError("image selection audit schema drift")
    records = image_selection["records"]
    if records != expected_config_records or not isinstance(records, list) or [record.get("stable_image_id") for record in records if isinstance(record, dict)] != stable_ids:
        raise SmokeEvidenceError("image selection stable ID binding drift")
    for record in records:
        if not isinstance(record, dict) or set(record) != {"annotation_relative_path", "annotation_sha256", "annotation_size_bytes", "coco_image_id", "filtered_rows", "height", "ignore_rows", "image_sha256", "image_size_bytes", "keep_rows", "raw_annotation_rows", "relative_path", "sequence_key", "split", "stable_image_id", "width"}:
            raise SmokeEvidenceError("image selection record schema drift")
    if input_audit["stable_image_ids"] != [record["stable_image_id"] for record in records]:
        raise SmokeEvidenceError("input/image selection stable ID mismatch")
    if (root / "error.json").exists() or (root / "partial_inventory.json").exists():
        raise SmokeEvidenceError("successful entry contains failure evidence")
    return {
        "entry_success_accepted": True,
        "completion": file_ref(root, "completion.json"),
        "artifact_inventory": file_ref(root, "artifact_inventory.json"),
        "config": file_ref(root, "config.json"),
    }


class SmokeEvidence:
    """Child-owned output directory with atomic config and terminal evidence."""

    def __init__(self, output_dir: Path, *, repo_root: Path | None = None) -> None:
        if not output_dir.is_absolute():
            raise SmokeEvidenceError("smoke output directory must be absolute")
        if output_dir.exists() or output_dir.is_symlink():
            raise SmokeEvidenceError("smoke output directory already exists")
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir()
        self.root = output_dir
        self.repo_root = None if repo_root is None else Path(repo_root).resolve()
        self._config_written = False

    def write_json(self, name: str, value: Any) -> str:
        if "/" in name or "\\" in name or name in {"artifact_inventory.json", "completion.json"}:
            raise SmokeEvidenceError(f"invalid evidence filename: {name}")
        payload = canonical_json_bytes(value)
        atomic_write(self.root / name, payload)
        return sha256_bytes(payload)

    def write_config(self, value: Any) -> str:
        if self._config_written:
            raise SmokeEvidenceError("config.json may only be written once")
        digest = self.write_json("config.json", value)
        self._config_written = True
        return digest

    def finalize_success(self, completion: dict[str, Any]) -> dict[str, object]:
        if not self._config_written:
            raise SmokeEvidenceError("config.json must be written before completion")
        inventory_value = inventory(self.root, ENTRY_INVENTORY_EXCLUDED)
        inventory_bytes = canonical_json_bytes(inventory_value)
        atomic_write(self.root / "artifact_inventory.json", inventory_bytes)
        terminal = dict(completion)
        terminal["artifact_inventory_sha256"] = sha256_bytes(inventory_bytes)
        atomic_write(self.root / "completion.json", canonical_json_bytes(terminal))
        return validate_entry_output(self.root, repo_root=self.repo_root)

    def finalize_failure(self, error: BaseException, completion: dict[str, Any]) -> None:
        if not (self.root / "error.json").exists():
            self.write_json("error.json", {
                "schema_version": 1,
                "status": "FAILED",
                "exception_type": type(error).__name__,
                "message": str(error),
                "original_exception_preserved": True,
            })
        partial = inventory(self.root, ENTRY_PARTIAL_EXCLUDED)
        if not (self.root / "partial_inventory.json").exists():
            atomic_write(self.root / "partial_inventory.json", canonical_json_bytes(partial))
        terminal = dict(completion)
        terminal.update({
            "status": "FAILED",
            "artifact_inventory_sha256": None,
            "partial_inventory_sha256": sha256_file(self.root / "partial_inventory.json"),
        })
        if not (self.root / "completion.json").exists():
            atomic_write(self.root / "completion.json", canonical_json_bytes(terminal))
