"""CPU-only prepared trainer boundary for the frozen T1 training contract.

This module is deliberately a port adapter.  It observes injected pure-Python
component ports and a single synthetic batch; it never constructs a model,
optimizer, data loader, evaluator, or CUDA object.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import sparse_rtdetr.baseline.primary_evaluator as _primary_evaluator
import sparse_rtdetr.baseline.training_contract as _training_contract
import sparse_rtdetr.baseline.training_evidence as _training_evidence
import sparse_rtdetr.baseline.training_runtime as _training_runtime


ADAPTER_SCHEMA_VERSION = 1
ADAPTER_ID = "rtdetrv2_r18_visdrone_prepared_trainer_adapter_t5c_v1"
ADAPTER_MODE = "synthetic"
COMPONENT_PROTOCOL = "t5c.synthetic_component_port.v1"
EVIDENCE_CONTRACT_MODULE_RELATIVE_PATH = "src/sparse_rtdetr/baseline/training_evidence.py"
EVIDENCE_CONTRACT_MODULE_SHA256 = "5efad5ea963f2b08e52cc7b7811ca64472dd36a5186cd5c476c7657adb5bc1b0"
CHECKPOINT_AUTHORITY_HELPER_LINE = 1440
CHECKPOINT_AUTHORITY_HELPER_SHA256 = "fd91ed1dd36374253a985d6daa5b25efafa3ff7aa188ea3d707b79a89f068c40"

COMPONENT_ROLES = (
    "model",
    "criterion",
    "optimizer",
    "lr_scheduler",
    "amp_scaler",
    "ema",
    "primary_evaluator",
    "train_batch_source",
    "development_batch_source",
    "checkpoint_serializer",
)
STATE_SEQUENCE = (
    "CREATED",
    "AUTHORITIES_BOUND",
    "COMPONENTS_PREPARED",
    "TRAIN_BATCH_ACQUIRED",
    "FORWARD_COMPLETED",
    "LOSS_VALIDATED",
    "BACKWARD_COMPLETED",
    "OPTIMIZER_STEP_COMPLETED",
    "SCHEDULER_OBSERVED",
    "EMA_UPDATED",
    "DEVELOPMENT_EVALUATION_COMPLETED",
    "CHECKPOINT_SERIALIZED",
    "EVIDENCE_TERMINALIZED",
    "COMPLETED",
)

_FORBIDDEN_ROLE_WORDS = (
    "test",
    "confirmatory",
    "tuning",
    "speed",
    "benchmark",
    "validation",
    "checkpoint",
    "dataset",
    "dataloader",
)
_SHA_FIELDS = {
    "raw_sha256",
    "canonical_sha256",
    "sha256",
    "compact_inventory_sha256",
    "manifest_raw_sha256",
    "manifest_inventory_sha256",
    "completion_sha256",
    "artifact_inventory_sha256",
    "entry_canonical_inventory_sha256",
    "config_sha256",
    "category_contract_sha256",
    "source_identity_sha256",
    "authority_file_sha256",
}

__all__ = (
    "PreparedTrainerError",
    "prepare_training_adapter",
    "validate_prepared_training_adapter",
    "run_synthetic_prepared_batch",
)


class PreparedTrainerError(ValueError):
    """Raised when the prepared adapter or its one-batch transaction drifts."""


def _fail(message: str) -> None:
    raise PreparedTrainerError(message)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical(value: Any) -> bytes:
    _assert_builtin_json(value, "canonical value")
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise PreparedTrainerError("value is not canonical JSON") from exc


def _digest(value: Any) -> str:
    return _sha_bytes(_canonical(value))


def _assert_builtin_json(value: Any, field: str = "value") -> None:
    if type(value) is dict:
        for key, child in value.items():
            if type(key) is not str:
                _fail(f"{field} has a non-string key")
            _assert_builtin_json(child, f"{field}.{key}")
        return
    if type(value) is list:
        for index, child in enumerate(value):
            _assert_builtin_json(child, f"{field}[{index}]")
        return
    if type(value) is float and not math.isfinite(value):
        _fail(f"{field} is non-finite")
    if type(value) not in {str, int, float, bool, type(None)}:
        _fail(f"{field} is not a builtin JSON value")


def _exact(value: Any, keys: set[str], field: str) -> dict[str, Any]:
    if type(value) is not dict:
        _fail(f"{field} must be a builtin dict")
    actual = set(value)
    if actual != keys:
        _fail(f"{field} key set drift: missing={sorted(keys - actual)} extra={sorted(actual - keys)}")
    return value


def _strict_equal(actual: Any, expected: Any, field: str) -> None:
    """Compare builtin JSON values without Python bool/int coercion."""

    if type(actual) is not type(expected):
        _fail(f"{field} type drift")
    if type(actual) is dict:
        if set(actual) != set(expected):
            _fail(f"{field} key set drift")
        for key in expected:
            _strict_equal(actual[key], expected[key], f"{field}.{key}")
        return
    if type(actual) is list:
        if len(actual) != len(expected):
            _fail(f"{field} length drift")
        for index, (observed, frozen) in enumerate(zip(actual, expected)):
            _strict_equal(observed, frozen, f"{field}[{index}]")
        return
    if actual != expected:
        _fail(f"{field} value drift")


def _string(value: Any, field: str, *, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value):
        _fail(f"{field} must be a builtin string")
    return value


def _integer(value: Any, field: str, *, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        _fail(f"{field} must be a builtin integer")
    return value


def _boolean(value: Any, field: str) -> bool:
    if type(value) is not bool:
        _fail(f"{field} must be a builtin boolean")
    return value


def _number(value: Any, field: str, *, positive: bool = False) -> float:
    if type(value) not in {int, float} or isinstance(value, bool) or not math.isfinite(float(value)):
        _fail(f"{field} must be a finite number")
    if positive and float(value) <= 0.0:
        _fail(f"{field} must be positive")
    return float(value)


def _float(value: Any, field: str, *, positive: bool = False) -> float:
    if type(value) is not float or not math.isfinite(value) or (positive and value <= 0.0):
        _fail(f"{field} must be a builtin finite float")
    return value


def _sha(value: Any, field: str) -> str:
    value = _string(value, field)
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        _fail(f"{field} is not a lowercase SHA-256")
    return value


def _relative_path(value: Any, field: str) -> str:
    value = _string(value, field)
    parts = value.split("/")
    if value.startswith("/") or "\\" in value or "//" in value or any(part in {"", ".", ".."} for part in parts):
        _fail(f"{field} is not a portable relative path")
    return value


def _copy(value: Any) -> Any:
    try:
        return copy.deepcopy(value)
    except Exception as exc:
        raise PreparedTrainerError("authority value could not be detached") from exc


def _expected_primary_runtime_binding() -> dict[str, Any]:
    return {
        "evaluator": {
            "evaluator_id": "visdrone_official_primary_evaluator_v1",
            "protocol_id": "visdrone_official_style_v1",
        },
        "implementation": {
            "commit": "036cca4d127ddd9e10e3cc7900c3eb759b55f59f",
            "tree": "fbe931976f6bac7d8e4b3bb319ff1905e99c5444",
        },
        "config": {
            "relative_path": "configs/baseline/visdrone_official_evaluator_v1.json",
            "raw_size_bytes": 3857,
            "raw_sha256": "36cfa69b0ff645c47c871283f917577ca27c760daf424e9544ab363dbf080ff5",
            "canonical_size_bytes": 3295,
            "canonical_sha256": "355de90bdb6007ed42ed65b3f653a1b8ea57f2184ab4fd48a9561b692b993998",
        },
        "authority_manifest": {
            "relative_path": "manifests/visdrone_det_toolkit_005445.json",
            "raw_size_bytes": 4166,
            "raw_sha256": "71168baf15d6d945fd5a4a6c5605b5ba533524efeede2a9d4020127576c1b36f",
            "canonical_size_bytes": 3351,
            "canonical_sha256": "5bad9faf7622fe4542aa3b46561d6d41fb2e4ee34f35551ecfd821c9577159e3",
        },
        "authority_archive_inventory": {
            "archive_size_bytes": 40960,
            "archive_sha256": "bf19dd9477210adf106c7cbf2a72370ed4af22dedb577f361f3dc9e77e99baa4",
            "file_count": 11,
            "inventory_sha256": "35a14a021509b82f1238912e5c77ebb3559f9ee6daa3cc6c92db810b5dce5da0",
        },
        "source_files": [
            {
                "role": "primary_evaluator",
                "relative_path": "src/sparse_rtdetr/baseline/primary_evaluator.py",
                "sha256": "d831fc641ac930822e693f99fbbcdd48abbe76738a618525135dd099963901bd",
            },
            {
                "role": "evaluation_protocol",
                "relative_path": "src/sparse_rtdetr/data_protocol/evaluation.py",
                "sha256": "e70ad71bb834b4cfc5d25441a78d2caa5982a86ae782918488924f67a832d216",
            },
        ],
        "independent_audit": {
            "stage": "P3_BASELINE_PRIMARY_EVALUATOR_V1_INDEPENDENT_AUDIT_R1",
            "classification": "PRIMARY_EVALUATOR_V1_CERTIFIED",
            "formal_return_code": 0,
            "script_size_bytes": 51244,
            "script_sha256": "4182f4082756fb2e52d6969e93889f24ae9af398cc98d67a1b9d990041b5524a",
            "mutation_cases_rejected": 129,
            "independent_oracle_cases": 100,
            "evaluator_tests_passed": 46,
            "full_cpu_tests_passed": 1116,
            "clean_archive_tests_passed": 46,
        },
    }


def _expected_vendor_binding() -> dict[str, Any]:
    return {
        "relative_path": "vendor/rtdetrv2_pytorch",
        "file_count": 124,
        "directory_count_excluding_root": 25,
        "total_size_bytes": 373735,
        "compact_inventory_sha256": "0fc6803665bc4b5720e983345b2cacb0147eceead9f882588880b6f8a0e68051",
        "manifest_relative_path": "manifests/rtdetrv2_upstream.json",
        "manifest_size_bytes": 33264,
        "manifest_raw_sha256": "f65a2d475365346a5dd5ce4f46b022a135b187e21421e922cc41eee4d28d20ae",
        "manifest_inventory_sha256": "2312c80d5b0fba88d43ffc6807c3fc150ae74b77740f2ab65072f044e033d6d7",
    }


def _expected_conversion_binding() -> dict[str, Any]:
    return {
        "relative_path": "artifacts/data/visdrone_protocol_v2_conversion_r3",
        "file_count": 25,
        "directory_count_excluding_root": 0,
        "total_size_bytes": 304418794,
        "completion_sha256": "46734010937168ac65bf27c3a6f4bf3f234554d5d4a99407903dec4b3ae9e817",
        "artifact_inventory_sha256": "aaee01ce4e00749b9db8ac5e6e49cf35875a8b266e9c3efc237bdf0b8b109ae7",
        "entry_canonical_inventory_sha256": "ddf32745302d6095e1c3dde89b1a85f05db53e6b4cc8ee85638e0d45bfa8d982",
        "config_sha256": "58be8659d6d7ae6faace82b36d45523f80cd48bae1bbcc6da4586c53056ec0d0",
        "category_contract_sha256": "b4b309f357cbe130a505a610dff340cc498dc74f766384c4559b2acd900728a0",
        "source_identity_sha256": "f0f16ba4438b51a09a6203f78884b199f8e2a2d3fb46309c357368a47a552bb9",
    }


def _expected_training_binding() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "training_contract_id": "rtdetrv2_r18_visdrone_baseline_training_v1",
        "baseline_id": "rtdetrv2_r18_visdrone_baseline_v1",
        "relative_path": "configs/baseline/rtdetrv2_r18_visdrone_training_v1.json",
        "raw_size_bytes": 10907,
        "raw_sha256": "0f9ddb2e6d8ec6d2b21e42f6c419511f5bd70df287457c2c2b6109eaf8a93297",
        "canonical_size_bytes": 9117,
        "canonical_sha256": "a20c71ef90cb4ccc091a717286a1c49be8a1ebffb178156942b77798d3d7f868",
        "vendor_runtime_binding": _expected_vendor_binding(),
        "primary_evaluator_runtime_binding": _expected_primary_runtime_binding(),
        "conversion_r3_runtime_binding": _expected_conversion_binding(),
    }


def _validate_full_training_binding(value: Any) -> dict[str, Any]:
    expected = _expected_training_binding()
    actual = _exact(
        value,
        {
            "schema_version",
            "training_contract_id",
            "baseline_id",
            "relative_path",
            "raw_size_bytes",
            "raw_sha256",
            "canonical_size_bytes",
            "canonical_sha256",
            "vendor_runtime_binding",
            "primary_evaluator_runtime_binding",
            "conversion_r3_runtime_binding",
        },
        "training contract binding",
    )
    _assert_builtin_json(actual, "training contract binding")
    _strict_equal(actual, expected, "training contract binding")
    return _copy(actual)


def _expected_runtime_identity() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "runtime_plan_id": "rtdetrv2_r18_visdrone_baseline_training_runtime_v1",
        "relative_path": "configs/baseline/rtdetrv2_r18_visdrone_training_runtime_v1.json",
        "raw_size_bytes": 12957,
        "raw_sha256": "cb6af1abae9351b4a268681587db82ad059745f1d7e198d7d8c829b4cee41aef",
        "canonical_size_bytes": 10888,
        "canonical_sha256": "3812b04d957e1cd7c3990c8458a540651c95c6bd551c631fc51717f2f7b386ac",
        "training_contract_identity": {
            "relative_path": "configs/baseline/rtdetrv2_r18_visdrone_training_v1.json",
            "raw_size_bytes": 10907,
            "raw_sha256": "0f9ddb2e6d8ec6d2b21e42f6c419511f5bd70df287457c2c2b6109eaf8a93297",
            "canonical_size_bytes": 9117,
            "canonical_sha256": "a20c71ef90cb4ccc091a717286a1c49be8a1ebffb178156942b77798d3d7f868",
            "module_relative_path": "src/sparse_rtdetr/baseline/training_contract.py",
            "module_sha256": "21599fae8312a32ddf833bf2f53557ed5b4d4122f0be167f2071f7f82008deef",
        },
        "vendor_runtime_binding": _expected_vendor_binding(),
        "conversion_r3_runtime_binding": _expected_conversion_binding(),
        "primary_evaluator_runtime_binding": _expected_primary_runtime_binding(),
        "plan": None,
    }


def _expected_runtime_plan() -> dict[str, Any]:
    """Return the detached semantic plan expected by the certified runtime API."""

    training = _expected_training_binding()
    primary = training["primary_evaluator_runtime_binding"]
    return {
        "schema_version": 1,
        "training_contract_id": "rtdetrv2_r18_visdrone_baseline_training_v1",
        "runtime_plan_id": "rtdetrv2_r18_visdrone_baseline_training_runtime_v1",
        "runtime_stage": "pre_cuda_plan_only",
        "model": {
            "type": "RTDETR",
            "backbone": "PResNet-18",
            "encoder": "HybridEncoder",
            "decoder": "RTDETRTransformerv2",
            "decoder_layers": 3,
            "num_classes": 10,
            "num_queries": 300,
            "input_size": [640, 640],
            "parameter_count": 20094584,
        },
        "optimizer": {
            "type": "AdamW",
            "betas": [0.9, 0.999],
            "gradient_clip_max_norm": 0.1,
            "parameter_name_match_mode": "case_sensitive_literal_substring",
            "parameter_groups": _optimizer_descriptor()["groups"],
            "mutually_exclusive": True,
            "cover_all_trainable_parameters": True,
            "parameter_identity_audit_required": True,
        },
        "amp": {
            key: value
            for key, value in _amp_descriptor().items()
            if key != "exact_observation_calls"
        },
        "ema": {
            "enabled": True,
            "decay": 0.9999,
            "warmups": 2000,
            "warmup_unit": "optimizer_updates",
            "development_evaluation_weights": "ema",
            "model_selection_weights": "ema",
            "preserve_raw": True,
            "preserve_ema": True,
        },
        "schedule": {
            "epochs": 120,
            "scheduler_type": "MultiStepLR",
            "scheduler_step_unit": "epoch",
            "milestones": [1000],
            "gamma": 0.1,
            "expected_decay_events": 0,
            "warmup_type": "LinearWarmup",
            "warmup_optimizer_steps": 2000,
            "augmentation_stop_epoch": 117,
            "checkpoint_every_epoch": 1,
            "periodic_checkpoint_every_epochs": 10,
            "development_evaluation_every_epochs": 1,
            "multiscale_enabled": False,
        },
        "augmentation": {
            "multiscale_enabled": False,
            "stop_epoch": 117,
            "stopped_transforms": [
                "RandomPhotometricDistort",
                "RandomZoomOut",
                "RandomIoUCrop",
            ],
            "transforms": [
                {"type": "RandomPhotometricDistort", "p": 0.5},
                {"type": "RandomZoomOut", "fill": 0},
                {"type": "RandomIoUCrop", "p": 0.8},
                {"type": "SanitizeBoundingBoxes", "min_size": 1},
                {"type": "RandomHorizontalFlip"},
                {"type": "Resize", "size": [640, 640]},
                {"type": "SanitizeBoundingBoxes", "min_size": 1},
                {"type": "ConvertPILImage", "scale": True, "dtype": "float32"},
                {"type": "ConvertBoxes", "fmt": "cxcywh", "normalize": True},
            ],
        },
        "data_roles": {
            "train_role": "train_core",
            "development_role": "development",
            "test_role": "forbidden",
            "confirmatory_role": "sealed_and_forbidden",
            "train_batch": 16,
            "development_batch": 32,
            "train_workers": 4,
            "development_workers": 4,
            "drop_last_train": True,
            "drop_last_development": False,
        },
        "evaluation_and_selection": {
            "primary_evaluator_id": "visdrone_official_primary_evaluator_v1",
            "primary_protocol_id": "visdrone_official_style_v1",
            "primary_evaluator_independently_certified": True,
            "primary_evaluator_required_before_training": True,
            "development_only": True,
            "evaluation_frequency_epochs": 1,
            "selection_metric": "AP@[0.50:0.95,maxDets=500]",
            "selection_direction": "maximize",
            "metric_comparison_precision": "unrounded_float64",
            "tie_breakers": ["AP50", "AR500", "earlier_epoch"],
            "model_selection_weights": "ema",
            "secondary_evaluator": "coco_secondary_vendor_v1",
            "secondary_diagnostic_only": True,
            "secondary_cannot_certify_or_select": True,
            "accuracy_threshold": None,
        },
        "checkpoint_policy": {
            "atomic_write_required": True,
            "atomic_rename_required": True,
            "file_fsync_required": True,
            "directory_fsync_required": True,
            "readback_required": True,
            "loadability_check_required": True,
            "inventory_required": True,
            "sha256_required": True,
            "overwrite": False,
            "resume": False,
            "retry": False,
            "last": "every epoch",
            "periodic": "every 10 epochs",
            "best": "on deterministic development selection improvement",
            "final": "epoch 120",
            "interrupted_status": "PERMANENT_FAIL",
            "required_state": list(_training_evidence.CHECKPOINT_REQUIRED_STATES),
        },
        "invocation_policy": {
            "formal_run_count": 1,
            "visible_devices": "0",
            "device": "cuda:0",
            "world_size": 1,
            "distributed": False,
            "seed": 0,
            "pretrained": False,
            "sync_bn": False,
            "resume_allowed": False,
            "retry_allowed": False,
            "overwrite_allowed": False,
            "test_only_allowed": False,
            "tuning_allowed": False,
            "network_download_allowed": False,
            "external_checkpoint_allowed": False,
            "checkpoint": None,
            "arbitrary_update_allowed": False,
        },
        "evidence_and_readiness": {
            "training_evidence_required": True,
            "checkpoint_integrity_required": True,
            "runtime_integrity_required": True,
            "source_identity_required": True,
            "data_identity_required": True,
            "environment_identity_required": True,
            "primary_evaluator_result_binding_required": True,
            "exactly_once_required": True,
            "required_epochs_complete": 120,
            "required_exit_code": 0,
            "confirmatory_access": False,
            "test_access": False,
            "speed_measurement": False,
            "model_selection_certified": False,
            "training_implementation_ready": False,
            "training_ready": False,
        },
        "source_bindings": {
            "training_contract": {
                "relative_path": training["relative_path"],
                "raw_size_bytes": training["raw_size_bytes"],
                "raw_sha256": training["raw_sha256"],
                "canonical_size_bytes": training["canonical_size_bytes"],
                "canonical_sha256": training["canonical_sha256"],
                "module_relative_path": "src/sparse_rtdetr/baseline/training_contract.py",
                "module_sha256": "21599fae8312a32ddf833bf2f53557ed5b4d4122f0be167f2071f7f82008deef",
            },
            "vendor_runtime": {
                "relative_path": "vendor/rtdetrv2_pytorch",
                "file_count": 124,
                "directory_count_excluding_root": 25,
                "total_size_bytes": 373735,
                "compact_inventory_sha256": "0fc6803665bc4b5720e983345b2cacb0147eceead9f882588880b6f8a0e68051",
                "manifest_relative_path": "manifests/rtdetrv2_upstream.json",
                "manifest_size_bytes": 33264,
                "manifest_raw_sha256": "f65a2d475365346a5dd5ce4f46b022a135b187e21421e922cc41eee4d28d20ae",
                "manifest_inventory_sha256": "2312c80d5b0fba88d43ffc6807c3fc150ae74b77740f2ab65072f044e033d6d7",
            },
            "conversion_r3": _expected_conversion_binding(),
            "primary_evaluator": {
                "evaluator_id": primary["evaluator"]["evaluator_id"],
                "protocol_id": primary["evaluator"]["protocol_id"],
                "implementation_commit": primary["implementation"]["commit"],
                "implementation_tree": primary["implementation"]["tree"],
                "config_relative_path": primary["config"]["relative_path"],
                "config_raw_size_bytes": primary["config"]["raw_size_bytes"],
                "config_raw_sha256": primary["config"]["raw_sha256"],
                "config_canonical_size_bytes": primary["config"]["canonical_size_bytes"],
                "config_canonical_sha256": primary["config"]["canonical_sha256"],
                "authority_manifest_relative_path": primary["authority_manifest"]["relative_path"],
                "authority_manifest_raw_size_bytes": primary["authority_manifest"]["raw_size_bytes"],
                "authority_manifest_raw_sha256": primary["authority_manifest"]["raw_sha256"],
                "authority_manifest_canonical_size_bytes": primary["authority_manifest"]["canonical_size_bytes"],
                "authority_manifest_canonical_sha256": primary["authority_manifest"]["canonical_sha256"],
                "authority_archive_size_bytes": primary["authority_archive_inventory"]["archive_size_bytes"],
                "authority_archive_sha256": primary["authority_archive_inventory"]["archive_sha256"],
                "authority_file_count": primary["authority_archive_inventory"]["file_count"],
                "authority_inventory_sha256": primary["authority_archive_inventory"]["inventory_sha256"],
                "source_files": primary["source_files"],
                "independent_audit": primary["independent_audit"],
            },
        },
        "vendor_conflicts": [
            {"id": "cli_resume", "vendor_observation": "vendor CLI exposes --resume", "formal_override": "forbidden"},
            {"id": "cli_tuning", "vendor_observation": "vendor CLI exposes --tuning", "formal_override": "forbidden"},
            {"id": "cli_update", "vendor_observation": "vendor CLI exposes --update", "formal_override": "forbidden"},
            {"id": "cli_test_only", "vendor_observation": "vendor CLI exposes --test-only", "formal_override": "forbidden"},
            {"id": "solver_cuda_fallback", "vendor_observation": "vendor solver falls back through torch.cuda.is_available()", "formal_override": "exact cuda:0 required later"},
            {"id": "solver_output_directory", "vendor_observation": "vendor solver creates output directories directly", "formal_override": "prepared runtime owns output creation later"},
            {"id": "solver_checkpoint_loading", "vendor_observation": "vendor solver permits network/local checkpoint loading", "formal_override": "random initialization and external checkpoints forbidden"},
            {"id": "checkpoint_logging", "vendor_observation": "vendor checkpoints use ordinary save_on_master and append-only logs", "formal_override": "atomic identity-bound checkpoint protocol required"},
            {"id": "yaml_evaluator", "vendor_observation": "vendor YAML builder supports only CocoEvaluator", "formal_override": "certified project primary evaluator required"},
            {"id": "best_stat", "vendor_observation": "vendor best_stat does not implement the frozen metric/tie-break policy", "formal_override": "unrounded AP with AP50 AR500 earlier_epoch tie-breakers"},
            {"id": "upstream_pretrained", "vendor_observation": "upstream recipe contains pretrained: True", "formal_override": "pretrained false and random initialization"},
            {"id": "upstream_sync_bn", "vendor_observation": "upstream runtime contains sync_bn: True", "formal_override": "sync_bn false"},
            {"id": "vendor_scaler_defaults", "vendor_observation": "vendor GradScaler YAML omits the four certified scaler parameters", "formal_override": "all four AMP scaler parameters frozen explicitly"},
            {"id": "vendor_multiscale_stop", "vendor_observation": "upstream dataloader include contains multiscale and stop epoch 71 defaults", "formal_override": "multiscale disabled and augmentation stop epoch 117"},
        ],
    }


def _validate_runtime_plan_semantics(plan: Any) -> dict[str, Any]:
    _assert_builtin_json(plan, "runtime plan")
    expected = _expected_runtime_plan()
    _strict_equal(plan, expected, "runtime plan")
    canonical = _training_runtime.canonical_training_runtime_plan_bytes(plan)
    if len(canonical) != 10888 or _sha_bytes(canonical) != "3812b04d957e1cd7c3990c8458a540651c95c6bd551c631fc51717f2f7b386ac":
        _fail("runtime plan canonical identity drift")
    return _copy(plan)


def _validate_full_runtime_binding(value: Any) -> dict[str, Any]:
    expected = _expected_runtime_identity()
    actual = _exact(
        value,
        {
            "schema_version",
            "runtime_plan_id",
            "relative_path",
            "raw_size_bytes",
            "raw_sha256",
            "canonical_size_bytes",
            "canonical_sha256",
            "training_contract_identity",
            "vendor_runtime_binding",
            "conversion_r3_runtime_binding",
            "primary_evaluator_runtime_binding",
            "plan",
        },
        "runtime plan binding",
    )
    _assert_builtin_json(actual, "runtime plan binding")
    for key, frozen in expected.items():
        if key != "plan":
            _strict_equal(actual[key], frozen, f"runtime plan binding.{key}")
    _validate_runtime_plan_semantics(actual["plan"])
    if actual["canonical_size_bytes"] != len(_training_runtime.canonical_training_runtime_plan_bytes(actual["plan"])):
        _fail("runtime plan binding canonical size drift")
    if actual["canonical_sha256"] != _sha_bytes(_training_runtime.canonical_training_runtime_plan_bytes(actual["plan"])):
        _fail("runtime plan binding canonical SHA drift")
    return _copy(actual)


def _validate_primary_binding(value: Any) -> dict[str, Any]:
    actual = _exact(
        value,
        {
            "config",
            "authority_manifest",
            "config_raw_size_bytes",
            "config_raw_sha256",
            "config_canonical_size_bytes",
            "config_canonical_sha256",
            "authority_manifest_raw_size_bytes",
            "authority_manifest_raw_sha256",
            "authority_manifest_canonical_size_bytes",
            "authority_manifest_canonical_sha256",
        },
        "primary evaluator binding",
    )
    _assert_builtin_json(actual, "primary evaluator binding")
    _primary_evaluator.validate_primary_evaluator_contract(actual["config"], actual["authority_manifest"])
    expected_config = {
        "relative_path": "configs/baseline/visdrone_official_evaluator_v1.json",
        "raw_size_bytes": 3857,
        "raw_sha256": "36cfa69b0ff645c47c871283f917577ca27c760daf424e9544ab363dbf080ff5",
        "canonical_size_bytes": 3295,
        "canonical_sha256": "355de90bdb6007ed42ed65b3f653a1b8ea57f2184ab4fd48a9561b692b993998",
    }
    expected_manifest = {
        "relative_path": "manifests/visdrone_det_toolkit_005445.json",
        "raw_size_bytes": 4166,
        "raw_sha256": "71168baf15d6d945fd5a4a6c5605b5ba533524efeede2a9d4020127576c1b36f",
        "canonical_size_bytes": 3351,
        "canonical_sha256": "5bad9faf7622fe4542aa3b46561d6d41fb2e4ee34f35551ecfd821c9577159e3",
    }
    canonical_config = _canonical(actual["config"])
    canonical_manifest = _canonical(actual["authority_manifest"])
    if len(canonical_config) != expected_config["canonical_size_bytes"] or _sha_bytes(canonical_config) != expected_config["canonical_sha256"]:
        _fail("primary evaluator config canonical identity drift")
    if len(canonical_manifest) != expected_manifest["canonical_size_bytes"] or _sha_bytes(canonical_manifest) != expected_manifest["canonical_sha256"]:
        _fail("primary evaluator manifest canonical identity drift")
    _strict_equal(
        {
            "relative_path": expected_config["relative_path"],
            "raw_size_bytes": actual["config_raw_size_bytes"],
            "raw_sha256": actual["config_raw_sha256"],
            "canonical_size_bytes": actual["config_canonical_size_bytes"],
            "canonical_sha256": actual["config_canonical_sha256"],
        },
        expected_config,
        "primary evaluator config binding",
    )
    _strict_equal(
        {
            "relative_path": expected_manifest["relative_path"],
            "raw_size_bytes": actual["authority_manifest_raw_size_bytes"],
            "raw_sha256": actual["authority_manifest_raw_sha256"],
            "canonical_size_bytes": actual["authority_manifest_canonical_size_bytes"],
            "canonical_sha256": actual["authority_manifest_canonical_sha256"],
        },
        expected_manifest,
        "primary evaluator manifest binding",
    )
    return _copy(actual)


def _authority_descriptor(
    training_binding: Any,
    runtime_binding: Any,
    evidence_contract: Any,
    primary_binding: Any,
) -> dict[str, Any]:
    training = _validate_full_training_binding(training_binding)
    runtime = _validate_full_runtime_binding(runtime_binding)
    primary = _validate_primary_binding(primary_binding)
    _assert_builtin_json(evidence_contract, "evidence contract")
    evidence = _training_evidence.validate_training_evidence_contract(
        evidence_contract,
        training,
        runtime,
    )
    if evidence.get("schema_version") != 1 or evidence.get("training_evidence_contract_id") != "rtdetrv2_r18_visdrone_baseline_training_evidence_v1":
        _fail("T5B evidence contract identity drift")
    if _digest(evidence) != "3184707c6cfa115477d007ac0be5eb142a054f78e6309a079ad1e9fab0db9bd6":
        _fail("T5B evidence contract canonical identity drift")
    _strict_equal(runtime["vendor_runtime_binding"], training["vendor_runtime_binding"], "vendor authority cross-binding")
    _strict_equal(runtime["conversion_r3_runtime_binding"], training["conversion_r3_runtime_binding"], "Conversion R3 cross-binding")
    _strict_equal(runtime["primary_evaluator_runtime_binding"], training["primary_evaluator_runtime_binding"], "primary evaluator cross-binding")
    _strict_equal(training["vendor_runtime_binding"], _expected_vendor_binding(), "vendor runtime authority")
    _strict_equal(training["conversion_r3_runtime_binding"], _expected_conversion_binding(), "Conversion R3 authority")
    _strict_equal(training["primary_evaluator_runtime_binding"], _expected_primary_runtime_binding(), "primary evaluator certification")
    if evidence["training_contract"]["canonical_sha256"] != training["canonical_sha256"]:
        _fail("T5B/T4 canonical binding drift")
    if evidence["runtime_plan"]["canonical_sha256"] != runtime["canonical_sha256"]:
        _fail("T5B/T5A canonical binding drift")
    identity = {
        "training_contract": {
            "id": training["training_contract_id"],
            "relative_path": training["relative_path"],
            "raw_size_bytes": training["raw_size_bytes"],
            "raw_sha256": training["raw_sha256"],
            "canonical_size_bytes": training["canonical_size_bytes"],
            "canonical_sha256": training["canonical_sha256"],
            "module_relative_path": runtime["training_contract_identity"]["module_relative_path"],
            "module_sha256": runtime["training_contract_identity"]["module_sha256"],
        },
        "runtime_plan": {
            "id": runtime["runtime_plan_id"],
            "relative_path": runtime["relative_path"],
            "raw_size_bytes": runtime["raw_size_bytes"],
            "raw_sha256": runtime["raw_sha256"],
            "canonical_size_bytes": runtime["canonical_size_bytes"],
            "canonical_sha256": runtime["canonical_sha256"],
            "module_relative_path": "src/sparse_rtdetr/baseline/training_runtime.py",
            "module_sha256": "735d75f612f59e6cec22dc53002980471c34c8b18601cd58704440f85741c7c9",
        },
        "evidence_contract": {
            "id": evidence["training_evidence_contract_id"],
            "relative_path": "configs/baseline/rtdetrv2_r18_visdrone_training_evidence_v1.json",
            "raw_size_bytes": 7139,
            "raw_sha256": "4d6bad4afbede236f1169796aa640f9d82bdcf91d06b799d39c183b20c230c9e",
            "canonical_size_bytes": 6069,
            "canonical_sha256": "3184707c6cfa115477d007ac0be5eb142a054f78e6309a079ad1e9fab0db9bd6",
            "module_relative_path": EVIDENCE_CONTRACT_MODULE_RELATIVE_PATH,
            "module_sha256": EVIDENCE_CONTRACT_MODULE_SHA256,
            "checkpoint_authority_helper_line": CHECKPOINT_AUTHORITY_HELPER_LINE,
            "checkpoint_authority_helper_source_sha256": CHECKPOINT_AUTHORITY_HELPER_SHA256,
            "required_state_order": list(_training_evidence.CHECKPOINT_REQUIRED_STATES),
        },
        "primary_evaluator": {
            "evaluator_id": primary["config"]["evaluator_id"],
            "protocol_id": primary["config"]["protocol_id"],
            "implementation_commit": training["primary_evaluator_runtime_binding"]["implementation"]["commit"],
            "implementation_tree": training["primary_evaluator_runtime_binding"]["implementation"]["tree"],
            "certified": True,
            "independent_audit_classification": "PRIMARY_EVALUATOR_V1_CERTIFIED",
        },
        "vendor_runtime": _copy(training["vendor_runtime_binding"]),
        "conversion_r3": _copy(training["conversion_r3_runtime_binding"]),
    }
    return {
        "identity": identity,
        "payloads": {
            "training_contract_binding": training,
            "runtime_plan_binding": runtime,
            "evidence_contract": evidence,
            "primary_evaluator_binding": primary,
        },
    }


def _validate_authorities(value: Any) -> dict[str, Any]:
    authorities = _exact(value, {"identity", "payloads"}, "prepared.authorities")
    _assert_builtin_json(authorities, "prepared.authorities")
    identity = _exact(
        authorities["identity"],
        {"training_contract", "runtime_plan", "evidence_contract", "primary_evaluator", "vendor_runtime", "conversion_r3"},
        "prepared.authorities.identity",
    )
    payloads = _exact(
        authorities["payloads"],
        {"training_contract_binding", "runtime_plan_binding", "evidence_contract", "primary_evaluator_binding"},
        "prepared.authorities.payloads",
    )
    _validate_full_training_binding(payloads["training_contract_binding"])
    runtime = _validate_full_runtime_binding(payloads["runtime_plan_binding"])
    evidence = _training_evidence.validate_training_evidence_contract(payloads["evidence_contract"])
    if _digest(evidence) != "3184707c6cfa115477d007ac0be5eb142a054f78e6309a079ad1e9fab0db9bd6":
        _fail("prepared T5B evidence contract digest drift")
    _validate_primary_binding(payloads["primary_evaluator_binding"])
    expected = _authority_descriptor(
        payloads["training_contract_binding"],
        runtime,
        evidence,
        payloads["primary_evaluator_binding"],
    )
    _strict_equal(identity, expected["identity"], "prepared authority identity")
    return _copy(authorities)


def _port_descriptor() -> dict[str, Any]:
    signatures = {
        "model": {
            "forward": ["batch"],
            "backward": ["scaled_loss"],
            "parameter_descriptors": [],
            "state_bytes": [],
        },
        "criterion": {"compute": ["forward_output", "batch"]},
        "optimizer": {
            "zero_grad": [],
            "step": [],
            "parameter_groups": [],
            "state_bytes": [],
        },
        "lr_scheduler": {"observe": [], "state_bytes": []},
        "amp_scaler": {
            "scale": ["loss"],
            "step": [],
            "update": [],
            "state_bytes": [],
        },
        "ema": {"update": ["model"], "state_bytes": []},
        "primary_evaluator": {"evaluate": ["development_batch", "weights", "context"]},
        "train_batch_source": {"next_batch": []},
        "development_batch_source": {"next_batch": []},
        "checkpoint_serializer": {"serialize": ["states"]},
    }
    ports = {}
    for role in COMPONENT_ROLES:
        ports[role] = {
            "role": role,
            "protocol": COMPONENT_PROTOCOL,
            "methods": [
                {
                    "name": name,
                    "parameters": list(parameters),
                    "expected_calls": 1,
                }
                for name, parameters in signatures[role].items()
            ],
            "identity_keys": ["role", "protocol", "implementation", "source_sha256"],
        }
    return {
        "protocol": COMPONENT_PROTOCOL,
        "roles": list(COMPONENT_ROLES),
        "ports": ports,
        "exact_role_count": len(COMPONENT_ROLES),
        "extra_callables_forbidden": True,
        "async_and_generator_ports_forbidden": True,
    }


def _optimizer_descriptor() -> dict[str, Any]:
    return {
        "type": "AdamW",
        "parameter_name_match_mode": "case_sensitive_literal_substring",
        "group_count": 3,
        "groups": [
            {
                "name": "backbone_non_norm",
                "parameter_scope": "backbone",
                "include_substrings": [],
                "exclude_substrings": ["norm", "bn"],
                "learning_rate": 0.00001,
                "weight_decay": 0.0001,
            },
            {
                "name": "norm_or_bn",
                "parameter_scope": "remaining_trainable",
                "include_substrings": ["norm", "bn"],
                "exclude_substrings": [],
                "learning_rate": 0.0001,
                "weight_decay": 0.0,
            },
            {
                "name": "default",
                "parameter_scope": "remaining_trainable",
                "include_substrings": [],
                "exclude_substrings": [],
                "learning_rate": 0.0001,
                "weight_decay": 0.0001,
            },
        ],
        "mutually_exclusive": True,
        "cover_all_trainable_parameters": True,
        "non_trainable_assignment_forbidden": True,
        "duplicate_assignment_forbidden": True,
        "parameter_identity_audit_required": True,
    }


def _amp_descriptor() -> dict[str, Any]:
    return {
        "enabled": True,
        "scaler_type": "GradScaler",
        "init_scale": 65536.0,
        "growth_factor": 2.0,
        "backoff_factor": 0.5,
        "growth_interval": 2000,
        "nonfinite_loss_allowed": 0,
        "nonfinite_gradient_allowed": 0,
        "optimizer_skipped_steps_allowed": 0,
        "overflow_events_allowed": 0,
        "exact_observation_calls": {"scale": 1, "step": 1, "update": 1, "state_bytes": 1},
    }


def _ema_descriptor() -> dict[str, Any]:
    return {
        "enabled": True,
        "decay": 0.9999,
        "warmups": 2000,
        "warmup_unit": "optimizer_updates",
        "development_evaluation_weights": "ema",
        "model_selection_weights": "ema",
        "preserve_raw": True,
        "preserve_ema": True,
        "exact_update_calls": 1,
        "update_after": "OPTIMIZER_STEP_COMPLETED",
    }


def _selection_descriptor() -> dict[str, Any]:
    return {
        "primary_evaluator_id": "visdrone_official_primary_evaluator_v1",
        "development_only": True,
        "weights": "ema",
        "metric": "AP@[0.50:0.95,maxDets=500]",
        "direction": "maximize",
        "precision": "unrounded_float64",
        "tie_breakers": ["AP50", "AR500", "earlier_epoch"],
        "secondary_can_certify_or_select": False,
    }


def _prepared_body(
    *,
    run_id: str,
    nonce: str,
    authorities: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": ADAPTER_SCHEMA_VERSION,
        "adapter_id": ADAPTER_ID,
        "mode": ADAPTER_MODE,
        "run_id": run_id,
        "nonce": nonce,
        "authorities": _copy(authorities),
        "component_roles": _port_descriptor(),
        "state_machine": {
            "states": list(STATE_SEQUENCE),
            "transitions": [
                {
                    "sequence": index,
                    "from": STATE_SEQUENCE[index - 1],
                    "to": STATE_SEQUENCE[index],
                    "predecessor_sha256_required": True,
                }
                for index in range(1, len(STATE_SEQUENCE))
            ],
            "exactly_once": True,
            "retry_resume_fallback_forbidden": True,
        },
        "data_roles": {
            "allowed": ["train_core", "development"],
            "train_role": "train_core",
            "development_role": "development",
            "forbidden_roles": ["test", "confirmatory", "tuning", "speed", "benchmark", "validation"],
            "host_absolute_paths_allowed": False,
            "batch_observation_keys": ["role", "batch_id", "batch_size", "payload_sha256"],
        },
        "optimizer_observation": _optimizer_descriptor(),
        "amp_observation": _amp_descriptor(),
        "ema_observation": _ema_descriptor(),
        "evaluator_integration": {
            "evaluator_id": "visdrone_official_primary_evaluator_v1",
            "protocol_id": "visdrone_official_style_v1",
            "development_only": True,
            "exact_calls": 1,
            "weights": "ema",
            "binds_run_batch_epoch_step": True,
        },
        "selection_policy": _selection_descriptor(),
        "evidence_checkpoint_handoff": {
            "evidence_contract_id": "rtdetrv2_r18_visdrone_baseline_training_evidence_v1",
            "training_contract_id": "rtdetrv2_r18_visdrone_baseline_training_v1",
            "runtime_plan_id": "rtdetrv2_r18_visdrone_baseline_training_runtime_v1",
            "required_state_order": list(_training_evidence.CHECKPOINT_REQUIRED_STATES),
            "checkpoint_role": "last",
            "checkpoint_epoch": 1,
            "atomic_write_required": True,
            "public_validator_required": True,
            "classifier_required": "SYNTHETIC_TERMINAL_COMPLETE",
            "production_root_creation_forbidden": True,
            "raw_ema_alias_forbidden": True,
        },
        "execution_policy": {
            "mode": "synthetic",
            "one_batch_only": True,
            "one_epoch_record_only": True,
            "one_checkpoint_only": True,
            "resume": False,
            "retry": False,
            "overwrite": False,
            "production_training_authorized": False,
            "real_model_authorized": False,
            "real_data_authorized": False,
            "cuda_authorized": False,
        },
    }


def _validate_state_machine(value: Any) -> dict[str, Any]:
    actual = _exact(
        value,
        {"states", "transitions", "exactly_once", "retry_resume_fallback_forbidden"},
        "prepared.state_machine",
    )
    _strict_equal(
        actual,
        _prepared_body(run_id="x", nonce="y", authorities={"identity": {}, "payloads": {}})["state_machine"],
        "prepared.state_machine",
    )
    return _copy(actual)


def _validate_prepared(value: Any) -> dict[str, Any]:
    _assert_builtin_json(value, "prepared adapter")
    prepared = _exact(
        value,
        {
            "schema_version",
            "adapter_id",
            "mode",
            "run_id",
            "nonce",
            "authorities",
            "component_roles",
            "state_machine",
            "data_roles",
            "optimizer_observation",
            "amp_observation",
            "ema_observation",
            "evaluator_integration",
            "selection_policy",
            "evidence_checkpoint_handoff",
            "execution_policy",
            "aggregate_sha256",
        },
        "prepared adapter",
    )
    if _integer(prepared["schema_version"], "prepared.schema_version", minimum=1) != ADAPTER_SCHEMA_VERSION:
        _fail("prepared adapter schema identity drift")
    if _string(prepared["adapter_id"], "prepared.adapter_id") != ADAPTER_ID:
        _fail("prepared adapter identity drift")
    if _string(prepared["mode"], "prepared.mode") != ADAPTER_MODE:
        _fail("prepared adapter is not synthetic")
    _string(prepared["run_id"], "prepared.run_id")
    _string(prepared["nonce"], "prepared.nonce")
    _validate_authorities(prepared["authorities"])
    _strict_equal(prepared["component_roles"], _port_descriptor(), "prepared.component_roles")
    _validate_state_machine(prepared["state_machine"])
    expected_body = _prepared_body(
        run_id=prepared["run_id"],
        nonce=prepared["nonce"],
        authorities=prepared["authorities"],
    )
    for key in (
        "data_roles",
        "optimizer_observation",
        "amp_observation",
        "ema_observation",
        "evaluator_integration",
        "selection_policy",
        "evidence_checkpoint_handoff",
        "execution_policy",
    ):
        _strict_equal(prepared[key], expected_body[key], f"prepared.{key}")
    aggregate = prepared["aggregate_sha256"]
    _sha(aggregate, "prepared.aggregate_sha256")
    body = _copy(prepared)
    del body["aggregate_sha256"]
    if _digest(body) != aggregate:
        _fail("prepared aggregate identity drift")
    return _copy(prepared)


def _make_prepared(*, run_id: str, nonce: str, authorities: dict[str, Any]) -> dict[str, Any]:
    body = _prepared_body(run_id=run_id, nonce=nonce, authorities=authorities)
    result = {**body, "aggregate_sha256": _digest(body)}
    return _validate_prepared(result)


def _validate_path_for_synthetic_root(value: str | os.PathLike[str]) -> Path:
    try:
        raw = os.fspath(value)
    except TypeError as exc:
        raise PreparedTrainerError("evidence root must be an absolute path") from exc
    if type(raw) is not str or not raw or not raw.startswith("/") or "\x00" in raw:
        _fail("evidence root must be an absolute path")
    if raw.endswith("/") or "//" in raw or "/./" in raw:
        _fail("evidence root must be normalized")
    parts = tuple(part for part in raw.split("/") if part)
    if ".." in parts:
        _fail("evidence root may not contain parent traversal")
    production = tuple(_training_evidence.EVIDENCE_ROOT_RELATIVE_PATH.split("/"))
    if len(parts) >= len(production) and parts[-len(production) :] == production:
        _fail("production evidence root is not accepted by T5C")
    path = Path(raw)
    if not path.parent.is_dir() or path.parent.is_symlink():
        _fail("synthetic evidence root parent must be a regular directory")
    if path.exists() or path.is_symlink():
        _fail("synthetic evidence root must be absent")
    return path


def _validate_identity_descriptor(value: Any, role: str) -> dict[str, Any]:
    _assert_builtin_json(value, f"{role}.identity")
    identity = _exact(value, {"role", "protocol", "implementation", "source_sha256"}, f"{role}.identity")
    if identity["role"] != role or identity["protocol"] != COMPONENT_PROTOCOL:
        _fail(f"{role} component identity drift")
    implementation = _string(identity["implementation"], f"{role}.identity.implementation")
    source_sha = _sha(identity["source_sha256"], f"{role}.identity.source_sha256")
    if source_sha != _digest({"role": role, "protocol": COMPONENT_PROTOCOL, "implementation": implementation}):
        _fail(f"{role} component source identity drift")
    return _copy(identity)


def _validate_ports(components: Any) -> tuple[dict[str, Any], dict[str, dict[str, int]]]:
    if type(components) is not dict or set(components) != set(COMPONENT_ROLES):
        _fail("component ports must contain exactly the ten frozen roles")
    descriptor = _port_descriptor()
    identities: dict[str, Any] = {}
    counts: dict[str, dict[str, int]] = {}
    for role in COMPONENT_ROLES:
        port = components[role]
        if port is None or callable(port):
            _fail(f"{role} port is not an object")
        expected_methods = {
            row["name"]: tuple(row["parameters"])
            for row in descriptor["ports"][role]["methods"]
        }
        public_callables = set()
        for name in dir(port):
            if name.startswith("_"):
                continue
            try:
                attribute = getattr(port, name)
            except Exception as exc:
                raise PreparedTrainerError(f"{role} port attribute access failed") from exc
            if callable(attribute):
                public_callables.add(name)
        if public_callables != set(expected_methods):
            _fail(f"{role} port has missing or extra callable methods")
        identities[role] = _validate_identity_descriptor(getattr(port, "identity", None), role)
        counts[role] = {name: 0 for name in expected_methods}
        for name, parameters in expected_methods.items():
            method = getattr(port, name)
            if inspect.iscoroutinefunction(method) or inspect.isgeneratorfunction(method):
                _fail(f"{role}.{name} may not be async or a generator")
            try:
                signature = inspect.signature(method)
            except (TypeError, ValueError) as exc:
                raise PreparedTrainerError(f"{role}.{name} signature is unavailable") from exc
            observed = list(signature.parameters.values())
            if any(parameter.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD for parameter in observed):
                _fail(f"{role}.{name} has an unsupported parameter kind")
            if [parameter.name for parameter in observed] != list(parameters):
                _fail(f"{role}.{name} parameter protocol drift")
            if any(parameter.default is not inspect.Parameter.empty for parameter in observed):
                _fail(f"{role}.{name} may not have optional parameters")
    return identities, counts


def _validate_port_identity_stability(components: dict[str, Any], identities: dict[str, Any]) -> None:
    for role in COMPONENT_ROLES:
        current = _validate_identity_descriptor(getattr(components[role], "identity", None), role)
        _strict_equal(current, identities[role], f"{role}.identity after transaction")


def _call(
    counts: dict[str, dict[str, int]],
    role: str,
    method_name: str,
    method: Any,
    *args: Any,
) -> Any:
    counts[role][method_name] += 1
    try:
        result = method(*args)
    except Exception as exc:
        raise PreparedTrainerError(f"{role}.{method_name} failed") from exc
    if inspect.isawaitable(result) or inspect.isgenerator(result):
        _fail(f"{role}.{method_name} returned an asynchronous or generator value")
    return result


def _validate_batch(value: Any, role: str, batch_size: int) -> dict[str, Any]:
    _assert_builtin_json(value, f"{role} batch")
    batch = _exact(value, {"role", "batch_id", "batch_size", "payload_sha256"}, f"{role} batch")
    if _string(batch["role"], f"{role}.role") != role:
        _fail(f"{role} batch role drift")
    _string(batch["batch_id"], f"{role}.batch_id")
    _integer(batch["batch_size"], f"{role}.batch_size", minimum=1)
    if batch["batch_size"] != batch_size:
        _fail(f"{role} batch size drift")
    _sha(batch["payload_sha256"], f"{role}.payload_sha256")
    for key, item in batch.items():
        lowered = f"{key}:{item}".casefold()
        if any(word in lowered for word in _FORBIDDEN_ROLE_WORDS):
            _fail(f"{role} batch contains a forbidden data-role token")
        if isinstance(item, str) and (item.startswith("/") or "\\" in item or ".." in item.split("/")):
            _fail(f"{role} batch contains a nonportable path")
    return _copy(batch)


def _validate_parameters_and_groups(model: Any, optimizer: Any, counts: dict[str, dict[str, int]]) -> dict[str, Any]:
    descriptors = _call(counts, "model", "parameter_descriptors", model.parameter_descriptors)
    if type(descriptors) is not list or not descriptors:
        _fail("model parameter descriptors must be a nonempty list")
    parameter_order: list[str] = []
    parameters: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(descriptors):
        _assert_builtin_json(item, f"parameter[{index}]")
        row = _exact(item, {"name", "trainable", "numel", "dtype"}, f"parameter[{index}]")
        name = _string(row["name"], f"parameter[{index}].name")
        if name in parameters:
            _fail("parameter names are duplicated")
        trainable = _boolean(row["trainable"], f"parameter[{index}].trainable")
        numel = _integer(row["numel"], f"parameter[{index}].numel", minimum=1)
        dtype = _string(row["dtype"], f"parameter[{index}].dtype")
        parameters[name] = {"trainable": trainable, "numel": numel, "dtype": dtype}
        parameter_order.append(name)
    groups = _call(counts, "optimizer", "parameter_groups", optimizer.parameter_groups)
    if type(groups) is not list or len(groups) != 3:
        _fail("optimizer must expose exactly three parameter groups")
    expected_groups = _optimizer_descriptor()["groups"]
    assigned: list[str] = []
    summaries = []
    for index, (row, expected) in enumerate(zip(groups, expected_groups)):
        _assert_builtin_json(row, f"optimizer group[{index}]")
        group = _exact(
            row,
            {"name", "parameter_names", "parameter_scope", "include_substrings", "exclude_substrings", "learning_rate", "weight_decay", "numel"},
            f"optimizer group[{index}]",
        )
        if _string(group["name"], f"optimizer group[{index}].name") != expected["name"] or _string(group["parameter_scope"], f"optimizer group[{index}].parameter_scope") != expected["parameter_scope"]:
            _fail("optimizer group role or order drift")
        _strict_equal(group["include_substrings"], expected["include_substrings"], f"optimizer group[{index}].include_substrings")
        _strict_equal(group["exclude_substrings"], expected["exclude_substrings"], f"optimizer group[{index}].exclude_substrings")
        if type(group["parameter_names"]) is not list or any(type(name) is not str for name in group["parameter_names"]):
            _fail("optimizer group parameter names are invalid")
        if len(set(group["parameter_names"])) != len(group["parameter_names"]):
            _fail("optimizer group contains duplicate parameters")
        _float(group["learning_rate"], f"optimizer group[{index}].learning_rate", positive=True)
        _float(group["weight_decay"], f"optimizer group[{index}].weight_decay")
        if group["learning_rate"] != expected["learning_rate"] or group["weight_decay"] != expected["weight_decay"]:
            _fail("optimizer group learning-rate or weight-decay drift")
        expected_numel = 0
        for name in group["parameter_names"]:
            if name not in parameters:
                _fail("optimizer group contains an extra or renamed parameter")
            if not parameters[name]["trainable"]:
                _fail("non-trainable parameter was assigned")
            if name in assigned:
                _fail("trainable parameter was assigned more than once")
            if expected["name"] == "backbone_non_norm" and (not name.startswith("backbone.") or any(token in name for token in ("norm", "bn"))):
                _fail("backbone optimizer selector did not match parameter")
            if expected["name"] == "norm_or_bn" and not any(token in name for token in ("norm", "bn")):
                _fail("normalization optimizer selector did not match parameter")
            if expected["name"] == "default" and (name.startswith("backbone.") and not any(token in name for token in ("norm", "bn")) or any(token in name for token in ("norm", "bn"))):
                _fail("default optimizer selector overlapped a prior group")
            assigned.append(name)
            expected_numel += parameters[name]["numel"]
        if _integer(group["numel"], f"optimizer group[{index}].numel", minimum=0) != expected_numel:
            _fail("optimizer group numel observation drift")
        summaries.append({
            "name": group["name"],
            "parameter_names": list(group["parameter_names"]),
            "parameter_count": len(group["parameter_names"]),
            "numel": expected_numel,
            "learning_rate": group["learning_rate"],
            "weight_decay": group["weight_decay"],
        })
    trainable = [name for name in parameter_order if parameters[name]["trainable"]]
    if assigned != [name for name in parameter_order if name in assigned] or set(assigned) != set(trainable):
        _fail("optimizer groups do not cover trainable parameters in model order")
    if any(parameters[name]["trainable"] and name not in assigned for name in parameter_order):
        _fail("optimizer groups omit a trainable parameter")
    return {
        "group_count": len(groups),
        "trainable_parameter_count": len(trainable),
        "trainable_numel": sum(parameters[name]["numel"] for name in trainable),
        "groups": summaries,
    }


def _select_development_candidate(candidates: Any) -> dict[str, Any]:
    if type(candidates) is not list or not candidates:
        _fail("selection candidates must be a nonempty list")
    normalized = []
    for index, candidate in enumerate(candidates):
        row = _exact(candidate, {"epoch", "AP", "AP50", "AR500", "weights"}, f"selection candidate[{index}]")
        epoch = _integer(row["epoch"], f"selection candidate[{index}].epoch", minimum=1)
        for field in ("AP", "AP50", "AR500"):
            _float(row[field], f"selection candidate[{index}].{field}")
        if row["weights"] != "ema":
            _fail("selection candidate is not EMA-backed")
        normalized.append(_copy(row))
    return max(normalized, key=lambda row: (row["AP"], row["AP50"], row["AR500"], -row["epoch"]))


def _selection_self_check() -> dict[str, Any]:
    candidates = [
        {"epoch": 1, "AP": 0.5000000000001, "AP50": 0.5, "AR500": 0.5, "weights": "ema"},
        {"epoch": 2, "AP": 0.5, "AP50": 0.99, "AR500": 0.99, "weights": "ema"},
        {"epoch": 3, "AP": 0.5, "AP50": 0.99, "AR500": 0.99, "weights": "ema"},
    ]
    selected = _select_development_candidate(candidates)
    if selected["epoch"] != 1:
        _fail("AP selection is not the unrounded primary comparison")
    selected = _select_development_candidate([
        {"epoch": 1, "AP": 0.5, "AP50": 0.7, "AR500": 0.6, "weights": "ema"},
        {"epoch": 2, "AP": 0.5, "AP50": 0.7, "AR500": 0.6, "weights": "ema"},
    ])
    if selected["epoch"] != 1:
        _fail("earlier-epoch tie-break drift")
    return {"cases": 2, "tie_breakers": ["AP", "AP50", "AR500", "earlier_epoch"], "pass": True}


def _validate_checkpoint_summary(checkpoint: Any, expected: dict[str, Any]) -> dict[str, Any]:
    if type(checkpoint) is not dict:
        _fail("checkpoint serializer must return a dict")
    if set(checkpoint) != set(_training_evidence.CHECKPOINT_REQUIRED_STATES):
        _fail("checkpoint serializer state set drift")
    for name in _training_evidence.CHECKPOINT_REQUIRED_STATES:
        if type(checkpoint[name]) is not bytes or not checkpoint[name]:
            _fail(f"checkpoint state {name} must be nonempty bytes")
    if checkpoint["raw_model"] is checkpoint["ema"] or checkpoint["raw_model"] == checkpoint["ema"]:
        _fail("raw model and EMA checkpoint payloads are aliased")
    return {name: {"size_bytes": len(checkpoint[name]), "sha256": _sha_bytes(checkpoint[name])} for name in checkpoint}


class _StateMachine:
    """Record each successful transition while enforcing the frozen order."""

    def __init__(self) -> None:
        self._trace: list[dict[str, Any]] = [
            {"sequence": 0, "state": STATE_SEQUENCE[0], "predecessor_sha256": None}
        ]

    @property
    def current(self) -> str:
        return self._trace[-1]["state"]

    @property
    def predecessor_sha256(self) -> str:
        return _digest(self._trace[-1])

    def advance(self, state: str) -> None:
        if state not in STATE_SEQUENCE:
            _fail("unknown state transition")
        expected_index = len(self._trace)
        if expected_index >= len(STATE_SEQUENCE) or STATE_SEQUENCE[expected_index] != state:
            _fail("state transition order or duplicate drift")
        previous = self._trace[-1]
        self._trace.append(
            {
                "sequence": expected_index,
                "state": state,
                "predecessor_sha256": _digest(previous),
            }
        )

    def snapshot(self) -> tuple[list[dict[str, Any]], str]:
        if tuple(item["state"] for item in self._trace) != STATE_SEQUENCE:
            _fail("state transition sequence is incomplete")
        trace = _copy(self._trace)
        return trace, _digest(trace)


def _identity_object(identity_id: str) -> dict[str, str]:
    return {"identity_id": identity_id, "sha256": _digest({"identity_id": identity_id})}


def _assert_state_bytes_unchanged(actual: Any, expected: Any, field: str) -> None:
    if type(actual) is not dict or type(expected) is not dict or set(actual) != set(expected):
        _fail(f"{field} state mapping was mutated")
    for name in expected:
        if type(actual[name]) is not bytes or type(expected[name]) is not bytes or actual[name] != expected[name]:
            _fail(f"{field}.{name} was mutated")


def _synthetic_loadability_probe(payload: bytes) -> bool:
    return type(payload) is bytes and bool(payload)


def _run_once(prepared: dict[str, Any], components: dict[str, Any], evidence_root: Path) -> dict[str, Any]:
    machine = _StateMachine()
    machine.advance("AUTHORITIES_BOUND")
    identities, counts = _validate_ports(components)
    model = components["model"]
    criterion = components["criterion"]
    optimizer = components["optimizer"]
    scheduler = components["lr_scheduler"]
    scaler = components["amp_scaler"]
    ema = components["ema"]
    evaluator = components["primary_evaluator"]
    train_source = components["train_batch_source"]
    development_source = components["development_batch_source"]
    serializer = components["checkpoint_serializer"]

    authority_payloads = prepared["authorities"]["payloads"]
    training_binding = authority_payloads["training_contract_binding"]
    runtime_binding = authority_payloads["runtime_plan_binding"]
    source_identity = _identity_object("t5c.synthetic.source.v1")
    data_identity = _identity_object("t5c.synthetic.train-development.v1")
    environment_identity = _identity_object("t5c.synthetic.cpu-pre-cuda.v1")
    optimizer_observation = _validate_parameters_and_groups(model, optimizer, counts)
    _call(counts, "optimizer", "zero_grad", optimizer.zero_grad)
    machine.advance("COMPONENTS_PREPARED")

    train_batch = _validate_batch(
        _call(counts, "train_batch_source", "next_batch", train_source.next_batch),
        "train_core",
        16,
    )
    machine.advance("TRAIN_BATCH_ACQUIRED")
    forward_input = _copy(train_batch)
    forward_output = _call(counts, "model", "forward", model.forward, forward_input)
    _strict_equal(forward_input, train_batch, "model.forward input")
    _assert_builtin_json(forward_output, "model forward output")
    forward = _exact(forward_output, {"batch_id", "output_sha256", "finite"}, "model forward output")
    if _string(forward["batch_id"], "forward.batch_id") != train_batch["batch_id"] or _sha(forward["output_sha256"], "forward.output_sha256") != forward["output_sha256"] or forward["finite"] is not True:
        _fail("model forward observation drift")
    machine.advance("FORWARD_COMPLETED")
    criterion_forward = _copy(forward)
    criterion_batch = _copy(train_batch)
    loss = _call(counts, "criterion", "compute", criterion.compute, criterion_forward, criterion_batch)
    _strict_equal(criterion_forward, forward, "criterion.forward_output input")
    _strict_equal(criterion_batch, train_batch, "criterion.batch input")
    _float(loss, "loss", positive=True)
    machine.advance("LOSS_VALIDATED")
    scaled_loss = _call(counts, "amp_scaler", "scale", scaler.scale, loss)
    _float(scaled_loss, "scaled_loss", positive=True)
    if scaled_loss != loss * 65536.0:
        _fail("AMP scale observation drift")
    gradient_norm = _call(counts, "model", "backward", model.backward, scaled_loss)
    _float(gradient_norm, "gradient_norm", positive=True)
    machine.advance("BACKWARD_COMPLETED")
    amp_step = _exact(_call(counts, "amp_scaler", "step", scaler.step), {"skipped", "overflow"}, "AMP step observation")
    _assert_builtin_json(amp_step, "AMP step observation")
    if amp_step["skipped"] is not False or amp_step["overflow"] is not False:
        _fail("AMP step was skipped or overflowed")
    _call(counts, "optimizer", "step", optimizer.step)
    amp_update = _exact(
        _call(counts, "amp_scaler", "update", scaler.update),
        {"scale", "growth_factor", "backoff_factor", "growth_interval", "growth_tracker"},
        "AMP update observation",
    )
    _assert_builtin_json(amp_update, "AMP update observation")
    if (
        type(amp_update["scale"]) is not float
        or type(amp_update["growth_factor"]) is not float
        or type(amp_update["backoff_factor"]) is not float
        or type(amp_update["growth_interval"]) is not int
        or type(amp_update["growth_tracker"]) is not int
        or amp_update["scale"] != 65536.0
        or amp_update["growth_factor"] != 2.0
        or amp_update["backoff_factor"] != 0.5
        or amp_update["growth_interval"] != 2000
        or amp_update["growth_tracker"] != 1
    ):
        _fail("AMP scaler literal or counter drift")
    machine.advance("OPTIMIZER_STEP_COMPLETED")
    scheduler_observation = _exact(
        _call(counts, "lr_scheduler", "observe", scheduler.observe),
        {"epoch", "step_unit", "decay_events"},
        "scheduler observation",
    )
    _assert_builtin_json(scheduler_observation, "scheduler observation")
    _strict_equal(scheduler_observation, {"epoch": 1, "step_unit": "epoch", "decay_events": 0}, "scheduler observation")
    if scheduler_observation != {"epoch": 1, "step_unit": "epoch", "decay_events": 0}:
        _fail("scheduler observation drift")
    machine.advance("SCHEDULER_OBSERVED")
    ema_model = model
    ema_observation = _exact(
        _call(counts, "ema", "update", ema.update, ema_model),
        {"updates", "decay", "warmups"},
        "EMA observation",
    )
    _assert_builtin_json(ema_observation, "EMA observation")
    _strict_equal(ema_observation, {"updates": 1, "decay": 0.9999, "warmups": 2000}, "EMA observation")
    if ema_observation != {"updates": 1, "decay": 0.9999, "warmups": 2000}:
        _fail("EMA policy or update counter drift")
    machine.advance("EMA_UPDATED")
    development_batch = _validate_batch(
        _call(counts, "development_batch_source", "next_batch", development_source.next_batch),
        "development",
        32,
    )
    evaluation_context = {
        "run_id": prepared["run_id"],
        "nonce": prepared["nonce"],
        "epoch": 1,
        "global_optimizer_step": 1,
        "ema_updates": 1,
        "training_contract_sha256": training_binding["canonical_sha256"],
        "runtime_plan_sha256": runtime_binding["canonical_sha256"],
        "ema_identity_sha256": _digest(identities["ema"]),
        "state_predecessor_sha256": machine.predecessor_sha256,
    }
    evaluator_batch = _copy(development_batch)
    evaluator_context = _copy(evaluation_context)
    evaluation = _exact(
        _call(counts, "primary_evaluator", "evaluate", evaluator.evaluate, evaluator_batch, "ema", evaluator_context),
        {"evaluator_id", "protocol_id", "role", "weights", "run_id", "nonce", "batch_id", "epoch", "global_optimizer_step", "ema_updates", "training_contract_sha256", "runtime_plan_sha256", "ema_identity_sha256", "state_predecessor_sha256", "AP", "AP50", "AR500"},
        "primary evaluator observation",
    )
    _strict_equal(evaluator_batch, development_batch, "primary evaluator batch input")
    _strict_equal(evaluator_context, evaluation_context, "primary evaluator context input")
    _assert_builtin_json(evaluation, "primary evaluator observation")
    if (
        _string(evaluation["evaluator_id"], "evaluation.evaluator_id") != "visdrone_official_primary_evaluator_v1"
        or _string(evaluation["protocol_id"], "evaluation.protocol_id") != "visdrone_official_style_v1"
        or _string(evaluation["role"], "evaluation.role") != "development"
        or _string(evaluation["weights"], "evaluation.weights") != "ema"
        or _string(evaluation["run_id"], "evaluation.run_id") != prepared["run_id"]
        or _string(evaluation["nonce"], "evaluation.nonce") != prepared["nonce"]
        or _string(evaluation["batch_id"], "evaluation.batch_id") != development_batch["batch_id"]
        or _integer(evaluation["epoch"], "evaluation.epoch", minimum=1) != 1
        or _integer(evaluation["global_optimizer_step"], "evaluation.global_optimizer_step", minimum=1) != 1
        or _integer(evaluation["ema_updates"], "evaluation.ema_updates", minimum=0) != 1
        or _sha(evaluation["training_contract_sha256"], "evaluation.training_contract_sha256") != training_binding["canonical_sha256"]
        or _sha(evaluation["runtime_plan_sha256"], "evaluation.runtime_plan_sha256") != runtime_binding["canonical_sha256"]
        or _sha(evaluation["ema_identity_sha256"], "evaluation.ema_identity_sha256") != evaluation_context["ema_identity_sha256"]
        or _sha(evaluation["state_predecessor_sha256"], "evaluation.state_predecessor_sha256") != evaluation_context["state_predecessor_sha256"]
    ):
        _fail("development evaluator binding drift")
    for field in ("AP", "AP50", "AR500"):
        _float(evaluation[field], f"evaluation.{field}")
    selection = _selection_self_check()
    selected = _select_development_candidate([{
        "epoch": 1,
        "AP": evaluation["AP"],
        "AP50": evaluation["AP50"],
        "AR500": evaluation["AR500"],
        "weights": "ema",
    }])
    if selected["epoch"] != 1:
        _fail("single development selection did not select epoch one")
    machine.advance("DEVELOPMENT_EVALUATION_COMPLETED")
    evaluator_result_sha256 = _digest(evaluation)
    epoch_observation = {
        "epoch": 1,
        "global_optimizer_step": 1,
        "loss": loss,
        "gradient_norm": gradient_norm,
        "amp_scale": amp_update["scale"],
        "amp_skipped_steps": 0,
        "amp_overflow_events": 0,
        "nonfinite_loss_count": 0,
        "nonfinite_gradient_count": 0,
        "ema_updates": 1,
        "evaluator_result_sha256": evaluator_result_sha256,
    }
    with _training_evidence.TrainingEvidenceWriter(
        str(evidence_root),
        run_id=prepared["run_id"],
        nonce=prepared["nonce"],
        mode="synthetic",
        training_contract_binding=training_binding,
        runtime_plan_binding=runtime_binding,
        source_identity=source_identity,
        data_identity=data_identity,
        environment_identity=environment_identity,
    ) as writer:
        epoch_record = writer.write_epoch_record(epoch_observation)
        state_input = {
            "raw_model": _call(counts, "model", "state_bytes", model.state_bytes),
            "ema": _call(counts, "ema", "state_bytes", ema.state_bytes),
            "optimizer": _call(counts, "optimizer", "state_bytes", optimizer.state_bytes),
            "scheduler": _call(counts, "lr_scheduler", "state_bytes", scheduler.state_bytes),
            "warmup": b"warmup:optimizer_steps=2000",
            "grad_scaler": _call(counts, "amp_scaler", "state_bytes", scaler.state_bytes),
            "epoch": b"1",
            "global_optimizer_step": b"1",
            "rng_states": b"synthetic-rng-state-v1",
            "config_identity": _canonical(prepared["authorities"]["identity"]),
            "source_identity": _canonical(source_identity),
            "environment_identity": _canonical(environment_identity),
        }
        _validate_checkpoint_summary(state_input, {})
        serializer_input = _copy(state_input)
        serialized = _call(counts, "checkpoint_serializer", "serialize", serializer.serialize, serializer_input)
        _assert_state_bytes_unchanged(serializer_input, state_input, "checkpoint serializer input")
        state_summary = _validate_checkpoint_summary(serialized, {})
        checkpoint_record = writer.write_checkpoint(
            serialized,
            role="last",
            loadability_probe=_synthetic_loadability_probe,
        )
        checkpoint_path = evidence_root / "checkpoints" / "checkpoint-last-0001.ckpt"
        validated_checkpoint = _training_evidence.validate_training_checkpoint(str(checkpoint_path))
        checkpoint_predecessor = _digest(epoch_record)
        expected_checkpoint = {
            "training_contract_id": "rtdetrv2_r18_visdrone_baseline_training_v1",
            "runtime_plan_id": "rtdetrv2_r18_visdrone_baseline_training_runtime_v1",
            "mode": "synthetic",
            "run_id": prepared["run_id"],
            "nonce": prepared["nonce"],
            "training_contract_sha256": training_binding["canonical_sha256"],
            "runtime_plan_sha256": runtime_binding["canonical_sha256"],
            "epoch": 1,
            "global_optimizer_step": 1,
            "role": "last",
            "relative_path": "checkpoints/checkpoint-last-0001.ckpt",
            "evaluator_id": "visdrone_official_primary_evaluator_v1",
            "predecessor_evidence_sha256": checkpoint_predecessor,
        }
        _training_evidence.validate_training_checkpoint(str(checkpoint_path), expected=expected_checkpoint)
        if checkpoint_record["predecessor_evidence_sha256"] != checkpoint_predecessor:
            _fail("checkpoint predecessor binding drift")
        machine.advance("CHECKPOINT_SERIALIZED")
        evidence_result = writer.complete()
    classification = _training_evidence.classify_training_evidence(str(evidence_root))
    if classification != "SYNTHETIC_TERMINAL_COMPLETE" or evidence_result["status"] != classification:
        _fail("T5B synthetic evidence did not terminalize successfully")
    machine.advance("EVIDENCE_TERMINALIZED")
    machine.advance("COMPLETED")
    _validate_port_identity_stability(components, identities)
    if counts["primary_evaluator"]["evaluate"] != 1:
        _fail("primary evaluator was called more than once")
    for role in COMPONENT_ROLES:
        for method, observed in counts[role].items():
            if observed != 1:
                _fail(f"{role}.{method} call count is not exactly one")
    final_trace, final_transition_sha = machine.snapshot()
    result = {
        "schema_version": ADAPTER_SCHEMA_VERSION,
        "adapter_id": ADAPTER_ID,
        "mode": "synthetic",
        "run_id": prepared["run_id"],
        "nonce": prepared["nonce"],
        "prepared_aggregate_sha256": prepared["aggregate_sha256"],
        "authority_identity": _copy(prepared["authorities"]["identity"]),
        "component_identities": identities,
        "call_counts": _copy(counts),
        "state_sequence": _copy(final_trace),
        "state_transition_sha256": final_transition_sha,
        "input_output_identity": {
            "train_batch": train_batch,
            "development_batch": development_batch,
            "forward_output": _copy(forward),
            "loss": loss,
            "gradient_norm": gradient_norm,
            "evaluator_result_sha256": evaluator_result_sha256,
        },
        "optimizer_observation": optimizer_observation,
        "amp_observation": {"step": amp_step, "update": amp_update, "scaled_loss": scaled_loss},
        "ema_observation": ema_observation,
        "evaluator_observation": {
            "evaluator_id": evaluation["evaluator_id"],
            "protocol_id": evaluation["protocol_id"],
            "role": evaluation["role"],
            "weights": evaluation["weights"],
            "result_sha256": evaluator_result_sha256,
        },
        "selection_observation": selection,
        "checkpoint": {
            "relative_path": checkpoint_record["relative_path"],
            "size_bytes": checkpoint_record["checkpoint_size_bytes"],
            "sha256": checkpoint_record["checkpoint_sha256"],
            "state_inventory_sha256": checkpoint_record["state_inventory_sha256"],
            "validated": True,
            "required_state_order": list(validated_checkpoint["required_state_order"]),
            "state_count": len(validated_checkpoint["states"]),
            "state_summary": state_summary,
        },
        "evidence": {
            "status": evidence_result["status"],
            "classification": classification,
            "epoch_count": evidence_result["epoch_count"],
            "checkpoint_count": evidence_result["checkpoint_count"],
            "terminalized_once": True,
        },
        "production_training_authorized": False,
        "aggregate_result_sha256": "",
    }
    result["aggregate_result_sha256"] = _digest(result)
    return _validate_result(result)


def _validate_result(value: Any) -> dict[str, Any]:
    _assert_builtin_json(value, "synthetic result")
    result = _exact(
        value,
        {
            "schema_version",
            "adapter_id",
            "mode",
            "run_id",
            "nonce",
            "prepared_aggregate_sha256",
            "authority_identity",
            "component_identities",
            "call_counts",
            "state_sequence",
            "state_transition_sha256",
            "input_output_identity",
            "optimizer_observation",
            "amp_observation",
            "ema_observation",
            "evaluator_observation",
            "selection_observation",
            "checkpoint",
            "evidence",
            "production_training_authorized",
            "aggregate_result_sha256",
        },
        "synthetic result",
    )
    if _integer(result["schema_version"], "result.schema_version", minimum=1) != ADAPTER_SCHEMA_VERSION:
        _fail("synthetic result schema version drift")
    if _string(result["adapter_id"], "result.adapter_id") != ADAPTER_ID or _string(result["mode"], "result.mode") != ADAPTER_MODE:
        _fail("synthetic result identity drift")
    _string(result["run_id"], "result.run_id")
    _string(result["nonce"], "result.nonce")
    _sha(result["prepared_aggregate_sha256"], "result.prepared_aggregate_sha256")
    if type(result["authority_identity"]) is not dict:
        _fail("result authority identity is not a dict")
    identities = _exact(result["component_identities"], set(COMPONENT_ROLES), "result.component_identities")
    for role in COMPONENT_ROLES:
        _validate_identity_descriptor(identities[role], role)
    counts = _exact(result["call_counts"], set(COMPONENT_ROLES), "result.call_counts")
    for role in COMPONENT_ROLES:
        method_counts = _exact(
            counts[role],
            {row["name"] for row in _port_descriptor()["ports"][role]["methods"]},
            f"result.call_counts.{role}",
        )
        for method, count in method_counts.items():
            if _integer(count, f"result.call_counts.{role}.{method}", minimum=0) != 1:
                _fail(f"result call count drift: {role}.{method}")

    trace = result["state_sequence"]
    if type(trace) is not list or len(trace) != len(STATE_SEQUENCE):
        _fail("result state sequence length drift")
    for index, row in enumerate(trace):
        row = _exact(row, {"sequence", "state", "predecessor_sha256"}, f"result.state_sequence[{index}]")
        if _integer(row["sequence"], f"result.state_sequence[{index}].sequence", minimum=0) != index:
            _fail("result state sequence number drift")
        if _string(row["state"], f"result.state_sequence[{index}].state") != STATE_SEQUENCE[index]:
            _fail("result state sequence state drift")
        if index == 0:
            if row["predecessor_sha256"] is not None:
                _fail("result initial state predecessor drift")
        elif _sha(row["predecessor_sha256"], f"result.state_sequence[{index}].predecessor_sha256") != _digest(trace[index - 1]):
            _fail("result state predecessor chain drift")
    _sha(result["state_transition_sha256"], "result.state_transition_sha256")
    if result["state_transition_sha256"] != _digest(trace):
        _fail("result state transition digest drift")

    io = _exact(
        result["input_output_identity"],
        {"train_batch", "development_batch", "forward_output", "loss", "gradient_norm", "evaluator_result_sha256"},
        "result.input_output_identity",
    )
    _validate_batch(io["train_batch"], "train_core", 16)
    _validate_batch(io["development_batch"], "development", 32)
    _assert_builtin_json(io["forward_output"], "result.forward_output")
    forward = _exact(io["forward_output"], {"batch_id", "output_sha256", "finite"}, "result.forward_output")
    if _string(forward["batch_id"], "result.forward_output.batch_id") != io["train_batch"]["batch_id"] or forward["finite"] is not True:
        _fail("result forward identity drift")
    _sha(forward["output_sha256"], "result.forward_output.output_sha256")
    _float(io["loss"], "result.loss", positive=True)
    _float(io["gradient_norm"], "result.gradient_norm", positive=True)
    _sha(io["evaluator_result_sha256"], "result.evaluator_result_sha256")

    _assert_builtin_json(result["optimizer_observation"], "result.optimizer_observation")
    _assert_builtin_json(result["amp_observation"], "result.amp_observation")
    _assert_builtin_json(result["ema_observation"], "result.ema_observation")
    evaluator = _exact(
        result["evaluator_observation"],
        {"evaluator_id", "protocol_id", "role", "weights", "result_sha256"},
        "result.evaluator_observation",
    )
    if _string(evaluator["evaluator_id"], "result.evaluator_observation.evaluator_id") != "visdrone_official_primary_evaluator_v1" or _string(evaluator["protocol_id"], "result.evaluator_observation.protocol_id") != "visdrone_official_style_v1" or _string(evaluator["role"], "result.evaluator_observation.role") != "development" or _string(evaluator["weights"], "result.evaluator_observation.weights") != "ema":
        _fail("result evaluator observation drift")
    if _sha(evaluator["result_sha256"], "result.evaluator_observation.result_sha256") != io["evaluator_result_sha256"]:
        _fail("result evaluator digest drift")
    _exact(result["selection_observation"], {"cases", "tie_breakers", "pass"}, "result.selection_observation")

    checkpoint = _exact(
        result["checkpoint"],
        {"relative_path", "size_bytes", "sha256", "state_inventory_sha256", "validated", "required_state_order", "state_count", "state_summary"},
        "result.checkpoint",
    )
    if _string(checkpoint["relative_path"], "result.checkpoint.relative_path") != "checkpoints/checkpoint-last-0001.ckpt":
        _fail("result checkpoint path drift")
    _integer(checkpoint["size_bytes"], "result.checkpoint.size_bytes", minimum=1)
    _sha(checkpoint["sha256"], "result.checkpoint.sha256")
    _sha(checkpoint["state_inventory_sha256"], "result.checkpoint.state_inventory_sha256")
    if checkpoint["validated"] is not True or _strict_state_order(checkpoint["required_state_order"]) is not True:
        _fail("result checkpoint validation drift")
    if _integer(checkpoint["state_count"], minimum=0, field="result.checkpoint.state_count") != len(_training_evidence.CHECKPOINT_REQUIRED_STATES):
        _fail("result checkpoint state count drift")
    _assert_builtin_json(checkpoint["state_summary"], "result.checkpoint.state_summary")

    evidence = _exact(
        result["evidence"],
        {"status", "classification", "epoch_count", "checkpoint_count", "terminalized_once"},
        "result.evidence",
    )
    if _string(evidence["status"], "result.evidence.status") != "SYNTHETIC_TERMINAL_COMPLETE" or _string(evidence["classification"], "result.evidence.classification") != "SYNTHETIC_TERMINAL_COMPLETE" or _integer(evidence["epoch_count"], "result.evidence.epoch_count", minimum=0) != 1 or _integer(evidence["checkpoint_count"], "result.evidence.checkpoint_count", minimum=0) != 1 or evidence["terminalized_once"] is not True:
        _fail("result evidence terminal identity drift")
    if result["production_training_authorized"] is not False:
        _fail("synthetic result authorizes production training")
    aggregate = result["aggregate_result_sha256"]
    _sha(aggregate, "result.aggregate_result_sha256")
    body = _copy(result)
    body["aggregate_result_sha256"] = ""
    if _digest(body) != aggregate:
        _fail("synthetic result aggregate identity drift")
    return _copy(result)


def _strict_state_order(value: Any) -> bool:
    if type(value) is not list or len(value) != len(_training_evidence.CHECKPOINT_REQUIRED_STATES):
        return False
    return all(type(item) is str for item in value) and value == list(_training_evidence.CHECKPOINT_REQUIRED_STATES)


def prepare_training_adapter(
    repo_root: str | os.PathLike[str],
    *,
    run_id: str = "t5c-synthetic-run-001",
    nonce: str = "0123456789abcdef0123456789abcdef",
    mode: str = "synthetic",
) -> dict[str, Any]:
    """Bind the four certified authorities into a detached synthetic descriptor."""

    if _string(mode, "mode") != ADAPTER_MODE:
        _fail("T5C only supports synthetic mode")
    _string(run_id, "run_id")
    _string(nonce, "nonce")
    try:
        training_binding = _training_contract.training_contract_binding(repo_root)
        runtime_binding = _training_runtime.training_runtime_plan_binding(repo_root)
        evidence_contract = _training_evidence.load_training_evidence_contract(repo_root)
        primary_binding = _primary_evaluator.primary_evaluator_contract_binding(repo_root)
        training_contract = _training_contract.load_training_contract(repo_root)
        validated_plan = _training_runtime.validate_training_runtime_plan(
            runtime_binding["plan"],
            training_contract,
            training_binding,
        )
    except Exception as exc:
        raise PreparedTrainerError("certified authority binding failed") from exc
    _strict_equal(validated_plan, runtime_binding["plan"], "runtime plan API binding")
    authority_inputs = {
        "training": _copy(training_binding),
        "runtime": _copy(runtime_binding),
        "evidence": _copy(evidence_contract),
        "primary": _copy(primary_binding),
    }
    try:
        authorities = _authority_descriptor(training_binding, runtime_binding, evidence_contract, primary_binding)
    except PreparedTrainerError:
        raise
    except Exception as exc:
        raise PreparedTrainerError("certified authority descriptor failed") from exc
    _strict_equal(training_binding, authority_inputs["training"], "training authority input")
    _strict_equal(runtime_binding, authority_inputs["runtime"], "runtime authority input")
    _strict_equal(evidence_contract, authority_inputs["evidence"], "evidence authority input")
    _strict_equal(primary_binding, authority_inputs["primary"], "primary authority input")
    return _make_prepared(run_id=run_id, nonce=nonce, authorities=authorities)


def validate_prepared_training_adapter(prepared: Any) -> dict[str, Any]:
    """Validate and return a detached prepared descriptor without side effects."""

    try:
        return _validate_prepared(prepared)
    except PreparedTrainerError:
        raise
    except Exception as exc:
        raise PreparedTrainerError("prepared adapter validation failed") from exc


_USED_PREPARED_ADAPTERS: set[str] = set()


def run_synthetic_prepared_batch(
    prepared: Any,
    components: Mapping[str, Any],
    evidence_root: str | os.PathLike[str],
) -> dict[str, Any]:
    """Run exactly one injected synthetic batch and publish one temporary chain."""

    try:
        validated = _validate_prepared(prepared)
    except PreparedTrainerError:
        raise
    except Exception as exc:
        raise PreparedTrainerError("prepared adapter validation failed") from exc
    key = validated["aggregate_sha256"]
    if key in _USED_PREPARED_ADAPTERS:
        _fail("prepared adapter has already been consumed")
    _USED_PREPARED_ADAPTERS.add(key)
    root = _validate_path_for_synthetic_root(evidence_root)
    try:
        if type(components) is not dict:
            _fail("components must be a builtin dict")
        return _run_once(validated, components, root)
    except PreparedTrainerError:
        raise
    except Exception as exc:
        raise PreparedTrainerError("synthetic prepared batch transaction failed") from exc
