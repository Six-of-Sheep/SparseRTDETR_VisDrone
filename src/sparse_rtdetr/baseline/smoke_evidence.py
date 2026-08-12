"""Atomic evidence and inventory primitives for Baseline Smoke V1."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from ..data_protocol.schema import ProtocolContractError


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
SCIENTIFIC_SCHEMA_VERSION = 2
SCIENTIFIC_SOURCE_ALLOWLIST = (
    "src/sparse_rtdetr/baseline/artifacts.py",
    "src/sparse_rtdetr/baseline/smoke.py",
    "src/sparse_rtdetr/baseline/smoke_evidence.py",
    "src/sparse_rtdetr/baseline/smoke_launcher.py",
    "src/sparse_rtdetr/baseline/smoke_outer_launcher.py",
)
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
        raise SmokeEvidenceError(f"{field} must be schema version 2")


def _strict_bool(value: Any, field: str) -> None:
    if type(value) is not bool:
        raise SmokeEvidenceError(f"{field} must be bool")


def _strict_int(value: Any, field: str) -> None:
    if type(value) is not int:
        raise SmokeEvidenceError(f"{field} must be int")


def _strict_sha(value: Any, field: str) -> None:
    if type(value) is not str or not SHA256_RE.fullmatch(value):
        raise SmokeEvidenceError(f"{field} must be lowercase SHA-256")


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


def _validate_input_batch_audit(value: Any, stable_ids: list[str], expected_device: str) -> None:
    required = {"schema_version", "runtime_mode", "batch_count", "image_count", "stable_image_ids", "target_labels_model_space", "loader_images", "model_images", "loader_orig_target_sizes", "model_orig_target_sizes", "labels", "boxes", "preprocessing"}
    if not isinstance(value, dict) or set(value) != required:
        raise SmokeEvidenceError("input_batch_audit schema fields are invalid")
    _strict_schema_version(value["schema_version"], "input_batch_audit.schema_version")
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
    _strict_shape(value["loader_images"]["shape"], [2, 3, 640, 640], "input_batch_audit.loader_images")
    _strict_shape(value["model_images"]["shape"], [2, 3, 640, 640], "input_batch_audit.model_images")
    for name in ("loader_orig_target_sizes", "model_orig_target_sizes"):
        _strict_shape(value[name]["shape"], [2, 2], f"input_batch_audit.{name}")
        if value[name]["dtype"] != "torch.int64" or value[name]["finite"] is not True:
            raise SmokeEvidenceError(f"input_batch_audit.{name} dtype/finite drift")
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
    required_prep = {"ordered_pipeline", "resize", "convert_pil_image", "scale_divisor", "normalization", "padding", "color", "value_range"}
    if not isinstance(prep, dict) or set(prep) != required_prep or prep["ordered_pipeline"] != ["Resize", "ConvertPILImage"]:
        raise SmokeEvidenceError("input_batch_audit preprocessing schema drift")
    if prep["resize"] != {"size": [640, 640], "interpolation": "default", "antialias": None}:
        raise SmokeEvidenceError("input_batch_audit resize drift")
    if prep["convert_pil_image"] != {"dtype": "float32", "scale": True} or prep["scale_divisor"] != 255:
        raise SmokeEvidenceError("input_batch_audit conversion drift")
    if prep["normalization"] != {"used": False, "mean": None, "std": None} or prep["padding"] != {"used": False, "mode": None}:
        raise SmokeEvidenceError("input_batch_audit normalization/padding drift")
    if prep["color"] != {"mode": "RGB", "channel_order": "RGB"}:
        raise SmokeEvidenceError("input_batch_audit color drift")
    _strict_finite_range(prep["value_range"], "input_batch_audit.value_range")


def _validate_cuda_runtime_identity(value: Any, expected_device: str) -> None:
    required = {"schema_version", "runtime_mode", "cuda_visible_devices", "device_count", "current_device", "device", "device_name", "device_capability", "torch_version", "torch_cuda_version", "cpu_fallback", "runtime_capture_after_cuda_initialization"}
    if not isinstance(value, dict) or set(value) != required:
        raise SmokeEvidenceError("cuda_runtime_identity schema fields are invalid")
    _strict_schema_version(value["schema_version"], "cuda_runtime_identity.schema_version")
    if value["runtime_mode"] not in {"synthetic", "real"} or type(value["cuda_visible_devices"]) is not str:
        raise SmokeEvidenceError("cuda_runtime_identity mode drift")
    _strict_int(value["device_count"], "cuda_runtime_identity.device_count")
    _strict_bool(value["cpu_fallback"], "cuda_runtime_identity.cpu_fallback")
    _strict_bool(value["runtime_capture_after_cuda_initialization"], "cuda_runtime_identity.runtime_capture_after_cuda_initialization")
    if value["runtime_mode"] == "real":
        if value["cuda_visible_devices"] != "0" or value["device_count"] != 1 or value["current_device"] != 0 or value["device"] != "cuda:0" or value["cpu_fallback"] is not False or value["runtime_capture_after_cuda_initialization"] is not True:
            raise SmokeEvidenceError("cuda_runtime_identity real runtime drift")
        if type(value["device_name"]) is not str or type(value["torch_version"]) is not str or type(value["torch_cuda_version"]) is not str:
            raise SmokeEvidenceError("cuda_runtime_identity version drift")
        if type(value["device_capability"]) is not list or len(value["device_capability"]) != 2 or any(type(x) is not int or x < 0 for x in value["device_capability"]):
            raise SmokeEvidenceError("cuda_runtime_identity capability drift")
    else:
        if value["device"] != "cpu" or value["device_count"] != 0 or value["current_device"] is not None or value["device_capability"] is not None or value["cpu_fallback"] is not True or value["runtime_capture_after_cuda_initialization"] is not False:
            raise SmokeEvidenceError("cuda_runtime_identity synthetic runtime drift")
    if expected_device != value["device"]:
        raise SmokeEvidenceError("cuda_runtime_identity device cross-binding drift")


def _validate_model_identity(value: Any, config: dict[str, Any], smoke_config: dict[str, Any], expected_device: str, output_audit: dict[str, Any]) -> None:
    required = {"schema_version", "runtime_mode", "baseline_id", "model_type", "backbone", "encoder", "decoder", "decoder_layers", "num_classes", "num_queries", "num_feature_levels", "feature_strides", "hidden_dim", "deformable_sampling_points", "parameters", "trainable_parameters", "parameter_tensor_count", "buffer_count", "training", "eval", "device", "pretrained", "checkpoint", "all_parameters_finite", "parameter_state_schema_sha256", "parameter_state_value_sha256", "postprocessor", "output_tensors"}
    if not isinstance(value, dict) or set(value) != required:
        raise SmokeEvidenceError("model_identity schema fields are invalid")
    _strict_schema_version(value["schema_version"], "model_identity.schema_version")
    contract = config["baseline_contract"]
    model_config = smoke_config["model"]
    expected = {"baseline_id": contract["baseline_id"], "backbone": contract["backbone"], "encoder": contract["encoder"], "decoder": contract["decoder"], "decoder_layers": contract["decoder_layers"], "num_classes": contract["num_classes"], "num_queries": contract["num_queries"], "num_feature_levels": contract["num_feature_levels"], "feature_strides": contract["feature_strides"], "hidden_dim": contract["hidden_dim"], "deformable_sampling_points": contract["deformable_sampling_points"], "parameters": contract["visdrone_parameter_count"]}
    for field, expected_value in expected.items():
        if value[field] != expected_value:
            raise SmokeEvidenceError(f"model_identity {field} drift")
    for field in ("decoder_layers", "num_classes", "num_queries", "num_feature_levels", "hidden_dim", "parameters", "trainable_parameters", "parameter_tensor_count", "buffer_count"):
        _strict_int(value[field], f"model_identity.{field}")
    for field in ("baseline_id", "num_classes", "num_queries", "parameters"):
        if model_config.get(field) != expected[field]:
            raise SmokeEvidenceError(f"model identity is not bound to smoke config: {field}")
    if model_config.get("pretrained") is not False or model_config.get("checkpoint") is not None or model_config.get("nms") is not False or model_config.get("num_top_queries") != 300:
        raise SmokeEvidenceError("model identity is not bound to frozen inference config")
    if value["feature_strides"] != [8, 16, 32] or value["deformable_sampling_points"] != [4, 4, 4] or value["training"] is not False or value["eval"] is not True or value["device"] != expected_device or value["pretrained"] is not False or value["checkpoint"] is not None or value["all_parameters_finite"] is not True:
        raise SmokeEvidenceError("model_identity runtime drift")
    _strict_sha(value["parameter_state_schema_sha256"], "model_identity.parameter_state_schema_sha256")
    _strict_sha(value["parameter_state_value_sha256"], "model_identity.parameter_state_value_sha256")
    if value["runtime_mode"] == "synthetic":
        _, _, _, expected_schema_sha, expected_value_sha, _ = _synthetic_state_hashes(contract)
        if value["parameter_state_schema_sha256"] != expected_schema_sha or value["parameter_state_value_sha256"] != expected_value_sha:
            raise SmokeEvidenceError("model_identity synthetic state hash drift")
    post = value["postprocessor"]
    if not isinstance(post, dict) or set(post) != {"type", "num_top_queries", "nms"} or type(post["type"]) is not str or post["num_top_queries"] != 300 or post["nms"] is not False:
        raise SmokeEvidenceError("model_identity postprocessor drift")
    if value["output_tensors"] != output_audit:
        raise SmokeEvidenceError("model_identity output tensor binding drift")


def _validate_source_identity(value: Any, config: dict[str, Any], repo_root: Path) -> None:
    required = {"schema_version", "baseline_id", "implementation", "source_allowlist", "source_rows_canonical_sha256", "baseline_config", "upstream_manifest", "vendor", "r3_binding"}
    if not isinstance(value, dict) or set(value) != required:
        raise SmokeEvidenceError("source_identity schema fields are invalid")
    _strict_schema_version(value["schema_version"], "source_identity.schema_version")
    if value["baseline_id"] != config["baseline_contract"]["baseline_id"] or value["implementation"] != "vendored RT-DETRv2 PyTorch":
        raise SmokeEvidenceError("source_identity baseline drift")
    rows = value["source_allowlist"]
    if not isinstance(rows, list) or [x.get("relative_path") for x in rows] != list(SCIENTIFIC_SOURCE_ALLOWLIST):
        raise SmokeEvidenceError("source_identity allowlist drift")
    for row, relative in zip(rows, SCIENTIFIC_SOURCE_ALLOWLIST):
        if set(row) != {"relative_path", "size_bytes", "sha256"} or row["relative_path"] != relative:
            raise SmokeEvidenceError("source_identity row schema drift")
        path = repo_root / relative
        if not path.is_file() or row["size_bytes"] != path.stat().st_size or row["sha256"] != sha256_file(path):
            raise SmokeEvidenceError(f"source_identity source drift: {relative}")
        _strict_int(row["size_bytes"], "source_identity.size_bytes"); _strict_sha(row["sha256"], "source_identity.sha256")
    if value["source_rows_canonical_sha256"] != sha256_bytes(canonical_json_bytes(rows)):
        raise SmokeEvidenceError("source_identity source rows SHA drift")
    baseline = value["baseline_config"]
    baseline_path = repo_root / SCIENTIFIC_BASELINE_CONFIG
    if not baseline_path.is_file():
        raise SmokeEvidenceError("source_identity baseline config is missing")
    if baseline != {"relative_path": SCIENTIFIC_BASELINE_CONFIG, "size_bytes": baseline_path.stat().st_size, "sha256": sha256_file(baseline_path)}:
        raise SmokeEvidenceError("source_identity baseline config drift")
    manifest = value["upstream_manifest"]
    manifest_path = repo_root / SCIENTIFIC_MANIFEST
    if not manifest_path.is_file():
        raise SmokeEvidenceError("source_identity upstream manifest is missing")
    manifest_value = json.loads(manifest_path.read_bytes().decode("utf-8"))
    expected_manifest = {"relative_path": SCIENTIFIC_MANIFEST, "size_bytes": manifest_path.stat().st_size, "sha256": sha256_file(manifest_path), "upstream_commit": SCIENTIFIC_UPSTREAM_COMMIT, "upstream_root_tree": SCIENTIFIC_UPSTREAM_ROOT_TREE, "upstream_subtree": SCIENTIFIC_UPSTREAM_SUBTREE}
    if manifest != expected_manifest or manifest_value.get("canonical_inventory_sha256") != SCIENTIFIC_VENDOR_INVENTORY_SHA256 or manifest_value.get("file_count") != SCIENTIFIC_VENDOR_FILE_COUNT or manifest_value.get("total_size_bytes") != SCIENTIFIC_VENDOR_TOTAL_BYTES:
        raise SmokeEvidenceError("source_identity upstream manifest drift")
    vendor = value["vendor"]
    expected_vendor = {"canonical_inventory_sha256": SCIENTIFIC_VENDOR_INVENTORY_SHA256, "file_count": SCIENTIFIC_VENDOR_FILE_COUNT, "total_size_bytes": SCIENTIFIC_VENDOR_TOTAL_BYTES, "r18_config": config["vendor"]["r18_config"], "r18_config_sha256": config["vendor"]["r18_config_sha256"], "r18_include": config["vendor"]["r18_include"], "r18_include_sha256": config["vendor"]["r18_include_sha256"]}
    if vendor != expected_vendor:
        raise SmokeEvidenceError("source_identity vendor drift")
    for path_key, sha_key in (("r18_config", "r18_config_sha256"), ("r18_include", "r18_include_sha256")):
        vendor_path = repo_root / vendor[path_key]
        if vendor_path.is_symlink() or not vendor_path.is_file() or sha256_file(vendor_path) != vendor[sha_key]:
            raise SmokeEvidenceError(f"source_identity vendor file drift: {path_key}")
    if value["r3_binding"] != {"artifact_root": config["r3_binding"]["artifact_root"], **SCIENTIFIC_R3_BINDING}:
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
        "preprocessing": {
            "ordered_pipeline": ["Resize", "ConvertPILImage"],
            "resize": {"size": [640, 640], "interpolation": "default", "antialias": None},
            "convert_pil_image": {"dtype": "float32", "scale": True},
            "scale_divisor": 255,
            "normalization": {"used": False, "mean": None, "std": None},
            "padding": {"used": False, "mode": None},
            "color": {"mode": "RGB", "channel_order": "RGB"},
            "value_range": _tensor_range(images),
        },
    }


def _state_hashes(model: Any, *, synthetic: bool) -> tuple[int, int, int, str, str, bool]:
    if not hasattr(model, "named_parameters"):
        empty = sha256_bytes(canonical_json_bytes([]))
        return 0, 0, 0, empty, empty, True
    import torch

    schema_rows = []
    value_rows = []
    all_finite = True
    for name, parameter in model.named_parameters():
        detached = parameter.detach()
        finite = bool(torch.isfinite(detached).all())
        all_finite = all_finite and finite
        row = {"name": name, "shape": list(detached.shape), "dtype": str(detached.dtype), "requires_grad": bool(parameter.requires_grad)}
        schema_rows.append(row)
        value_rows.append({**row, "logical_sha256": sha256_bytes(detached.cpu().contiguous().numpy().tobytes()) if finite else None})
    for name, buffer in model.named_buffers():
        detached = buffer.detach()
        finite = bool(torch.isfinite(detached).all())
        all_finite = all_finite and finite
        row = {"name": name, "shape": list(detached.shape), "dtype": str(detached.dtype), "requires_grad": False, "buffer": True}
        schema_rows.append(row)
        value_rows.append({**row, "logical_sha256": sha256_bytes(detached.cpu().contiguous().numpy().tobytes()) if finite else None})
    return (
        sum(parameter.numel() for parameter in model.parameters()),
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        sum(1 for _ in model.named_parameters()),
        sha256_bytes(canonical_json_bytes(schema_rows)),
        sha256_bytes(canonical_json_bytes(value_rows)),
        all_finite,
    )


def _synthetic_state_hashes(contract: dict[str, Any]) -> tuple[int, int, int, str, str, bool]:
    rows = [{
        "model": "synthetic-contract-model",
        "baseline_id": contract["baseline_id"],
        "parameters": contract["visdrone_parameter_count"],
        "trainable_parameters": contract["visdrone_parameter_count"],
        "parameter_tensor_count": 0,
        "buffer_count": 0,
    }]
    digest = sha256_bytes(canonical_json_bytes(rows))
    return contract["visdrone_parameter_count"], contract["visdrone_parameter_count"], 0, digest, digest, True


def build_model_identity(repo_root: str | Path, smoke_config: dict[str, Any], model: Any, postprocessor: Any, outputs: dict[str, Any], *, runtime_mode: str, device: str) -> dict[str, object]:
    baseline_path = Path(repo_root).resolve() / SCIENTIFIC_BASELINE_CONFIG
    baseline = json.loads(baseline_path.read_bytes().decode("utf-8"))
    contract = baseline["baseline_contract"]
    if runtime_mode == "synthetic":
        parameters, trainable, parameter_tensors, schema_sha, value_sha, finite = _synthetic_state_hashes(contract)
    else:
        parameters, trainable, parameter_tensors, schema_sha, value_sha, finite = _state_hashes(model, synthetic=False)
    return {
        "schema_version": SCIENTIFIC_SCHEMA_VERSION,
        "runtime_mode": runtime_mode,
        "baseline_id": contract["baseline_id"],
        "model_type": baseline["model"]["type"],
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
        "postprocessor": {"type": type(postprocessor).__name__, "num_top_queries": int(getattr(postprocessor, "num_top_queries", 300)), "nms": bool(getattr(postprocessor, "nms", False))},
        "output_tensors": outputs,
    }


def build_source_identity(repo_root: str | Path, smoke_config: dict[str, Any]) -> dict[str, object]:
    root = Path(repo_root).resolve()
    rows = []
    for relative in SCIENTIFIC_SOURCE_ALLOWLIST:
        path = root / relative
        rows.append({"relative_path": relative, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    baseline_path = root / SCIENTIFIC_BASELINE_CONFIG
    manifest_path = root / SCIENTIFIC_MANIFEST
    manifest = json.loads(manifest_path.read_bytes().decode("utf-8"))
    baseline = json.loads((root / SCIENTIFIC_BASELINE_CONFIG).read_bytes().decode("utf-8"))
    vendor = baseline["vendor"]
    return {
        "schema_version": SCIENTIFIC_SCHEMA_VERSION,
        "baseline_id": baseline["baseline_contract"]["baseline_id"],
        "implementation": "vendored RT-DETRv2 PyTorch",
        "source_allowlist": rows,
        "source_rows_canonical_sha256": sha256_bytes(canonical_json_bytes(rows)),
        "baseline_config": {"relative_path": SCIENTIFIC_BASELINE_CONFIG, "size_bytes": baseline_path.stat().st_size, "sha256": sha256_file(baseline_path)},
        "upstream_manifest": {"relative_path": SCIENTIFIC_MANIFEST, "size_bytes": manifest_path.stat().st_size, "sha256": sha256_file(manifest_path), "upstream_commit": manifest["upstream_commit"], "upstream_root_tree": manifest["upstream_root_tree"], "upstream_subtree": manifest["upstream_subtree"]},
        "vendor": {"canonical_inventory_sha256": manifest["canonical_inventory_sha256"], "file_count": manifest["file_count"], "total_size_bytes": manifest["total_size_bytes"], "r18_config": vendor["r18_config"], "r18_config_sha256": vendor["r18_config_sha256"], "r18_include": vendor["r18_include"], "r18_include_sha256": vendor["r18_include_sha256"]},
        "r3_binding": {"artifact_root": baseline["r3_binding"]["artifact_root"], **SCIENTIFIC_R3_BINDING},
    }


def build_cuda_runtime_identity(runtime: dict[str, Any], *, runtime_mode: str) -> dict[str, object]:
    if runtime_mode == "synthetic":
        return {"schema_version": SCIENTIFIC_SCHEMA_VERSION, "runtime_mode": "synthetic", "cuda_visible_devices": "", "device_count": 0, "current_device": None, "device": "cpu", "device_name": "cpu", "device_capability": None, "torch_version": "synthetic", "torch_cuda_version": None, "cpu_fallback": True, "runtime_capture_after_cuda_initialization": False}
    import torch
    return {"schema_version": SCIENTIFIC_SCHEMA_VERSION, "runtime_mode": "real", "cuda_visible_devices": "0", "device_count": 1, "current_device": int(torch.cuda.current_device()), "device": "cuda:0", "device_name": torch.cuda.get_device_name(0), "device_capability": list(torch.cuda.get_device_capability(0)), "torch_version": torch.__version__, "torch_cuda_version": torch.version.cuda, "cpu_fallback": False, "runtime_capture_after_cuda_initialization": True}


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
        if (candidate / SCIENTIFIC_BASELINE_CONFIG).is_file() and (candidate / SCIENTIFIC_MANIFEST).is_file():
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
    _validate_input_batch_audit(input_audit, stable_ids, expected_device)
    _validate_cuda_runtime_identity(scientific["cuda_runtime_identity.json"], expected_device)
    _validate_model_output_audit(scientific["model_output_audit.json"], expected_device)
    _validate_model_identity(scientific["model_identity.json"], baseline_config, config_value, expected_device, scientific["model_output_audit.json"])
    _validate_source_identity(scientific["source_identity.json"], baseline_config, repo_root)
    _validate_call_audit(scientific["call_audit.json"])
    _validate_postprocess_audit(scientific["postprocess_audit.json"], stable_ids, expected_device, completion["total_predictions"])
    image_selection = scientific["image_selection_audit.json"]
    if not isinstance(image_selection, dict) or set(image_selection) != {"records", "verified_from_manifest", "selection_policy"} or image_selection["verified_from_manifest"] is not True or image_selection["selection_policy"] != "explicit_coco_image_ids_only":
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
