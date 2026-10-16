"""Portable, standard-library-only validation for the frozen T7B v2a contract.

This module describes the next baseline without importing the v1 validator or
any framework, dataset, evaluator, launcher, or training implementation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path
from typing import Any, Iterable


V2A_CONFIG_RELATIVE_PATH = "configs/baseline/rtdetrv2_r18_visdrone_baseline_v2a.json"
V2A_CONTRACT_ID = "rtdetrv2_r18_visdrone_baseline_training_v2a"
V2A_BASELINE_ID = "rtdetrv2_r18_visdrone_baseline_v2a"
V2A_STAGE = "T7B_BASELINE_V2A_FROZEN_CONTRACT"

PARENT_TRAINING_CONFIG_RELATIVE_PATH = "configs/baseline/rtdetrv2_r18_visdrone_training_v1.json"
PARENT_TRAINING_CONFIG_RAW_SIZE_BYTES = 10907
PARENT_TRAINING_CONFIG_RAW_SHA256 = "0f9ddb2e6d8ec6d2b21e42f6c419511f5bd70df287457c2c2b6109eaf8a93297"
PARENT_TRAINING_CONFIG_CANONICAL_SIZE_BYTES = 9117
PARENT_TRAINING_CONFIG_CANONICAL_SHA256 = "a20c71ef90cb4ccc091a717286a1c49be8a1ebffb178156942b77798d3d7f868"
PARENT_BASELINE_ID = "rtdetrv2_r18_visdrone_baseline_v1"
PARENT_TRAINING_CONTRACT_ID = "rtdetrv2_r18_visdrone_baseline_training_v1"
PARENT_DEVELOPMENT_AP = 22.9469

AUTHORITY_ROOT_RELATIVE_PATH = "artifacts/pretrained_rtdetrv2_presnet18_imagenet_v1"
AUTHORITY_ARTIFACT_ID = "rtdetrv2_presnet18_imagenet_v1"
AUTHORITY_CLASSIFICATION = "OFFLINE_PRETRAINED_BACKBONE_AUTHORITY"
AUTHORITY_WEIGHT_RELATIVE_PATH = (
    f"{AUTHORITY_ROOT_RELATIVE_PATH}/ResNet18_vd_pretrained_from_paddle.pth"
)
AUTHORITY_WEIGHT_SIZE_BYTES = 44878642
AUTHORITY_WEIGHT_SHA256 = "911a745b62e173c8f4b9af513c2ea295428cf23f1bbfe9048381500f140fd720"
AUTHORITY_MANIFEST_RELATIVE_PATH = f"{AUTHORITY_ROOT_RELATIVE_PATH}/pretrained_authority.json"
AUTHORITY_MANIFEST_SIZE_BYTES = 1608
AUTHORITY_MANIFEST_SHA256 = "f290281d7c9f1589e2f6da3aca2251e351df48dbad82317ee59e4381ecb01dd0"
AUTHORITY_ROOT_MODE = 0o700
AUTHORITY_FILE_MODE = 0o600
AUTHORITY_WEIGHT_NLINK = 1
AUTHORITY_MANIFEST_NLINK = 1
AUTHORITY_STATE_KEY_COUNT = 115
AUTHORITY_TENSOR_NUMEL = 11209824
AUTHORITY_STATE_INVENTORY_SHA256 = "0dca09c370c8ae6ee4f82db357c3a3e1c4c2fdb398577e205dd5a318ac1ef6fc"
AUTHORITY_VENDOR_CONFIG_RELATIVE_PATH = "vendor/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml"
AUTHORITY_VENDOR_CONFIG_SIZE_BYTES = 643
AUTHORITY_VENDOR_CONFIG_SHA256 = "93182eb778c58346e31b947dec4cb7e673373d3d93210d271a10cc1c6043a1e9"
AUTHORITY_VENDOR_SOURCE_RELATIVE_PATH = "vendor/rtdetrv2_pytorch/src/nn/backbone/presnet.py"
AUTHORITY_VENDOR_SOURCE_SIZE_BYTES = 7431
AUTHORITY_VENDOR_SOURCE_SHA256 = "7b07405d6c3e79ebb50b7940b4b235a8aacea71f1e38018f2d0ffd5b7a10afc8"
AUTHORITY_VENDOR_OPTIMIZER_RELATIVE_PATH = "vendor/rtdetrv2_pytorch/configs/rtdetrv2/include/optimizer.yml"
AUTHORITY_VENDOR_OPTIMIZER_SIZE_BYTES = 514
AUTHORITY_VENDOR_OPTIMIZER_SHA256 = "30bc23da028bf8c451ad46f1800c40864e364403f6eb0ea95394d7cce465fc3a"
AUTHORITY_DECLARED_URL = (
    "https://github.com/lyuwenyu/storage/releases/download/v0.1/"
    "ResNet18_vd_pretrained_from_paddle.pth"
)

# Filled after the checked-in JSON is written.  Raw identity is deliberately
# separate from the canonical digest so whitespace repacks remain rejected.
V2A_CONFIG_RAW_SIZE_BYTES = 12751
V2A_CONFIG_RAW_SHA256 = "40800725fa0409ac440f463d1df08ce647714aa7fa715a05f7e98878403ae407"
V2A_CONFIG_CANONICAL_SIZE_BYTES = 10198
V2A_CONFIG_CANONICAL_SHA256 = "96db5c69286775fb2b2b6a1a6997e58fe9d6019040eeca7328c02c7c2426ab0e"

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_OID = re.compile(r"[0-9a-f]{40}\Z")
_MISSING = object()

__all__ = (
    "TrainingV2AContractError",
    "load_training_v2a_contract",
    "validate_training_v2a_contract",
    "canonical_training_v2a_contract_bytes",
    "training_v2a_contract_binding",
    "audit_optimizer_parameter_groups",
    "resolve_optimizer_parameter_group",
)


class TrainingV2AContractError(ValueError):
    """Raised when the v2a contract or a bound authority is invalid."""


def _object(**children: Any) -> tuple[str, dict[str, Any]]:
    return ("object", children)


def _list(*children: Any) -> tuple[str, tuple[Any, ...]]:
    return ("list", children)


def _build_frozen_contract() -> dict[str, Any]:
    """Return the reviewable expected contract used by all local layers."""

    return {
        "schema_version": 1,
        "contract_id": V2A_CONTRACT_ID,
        "baseline_id": V2A_BASELINE_ID,
        "stage": V2A_STAGE,
        "parent_baseline_v1": {
            "training_contract_id": PARENT_TRAINING_CONTRACT_ID,
            "baseline_id": PARENT_BASELINE_ID,
            "relative_path": PARENT_TRAINING_CONFIG_RELATIVE_PATH,
            "raw_size_bytes": PARENT_TRAINING_CONFIG_RAW_SIZE_BYTES,
            "raw_sha256": PARENT_TRAINING_CONFIG_RAW_SHA256,
            "canonical_size_bytes": PARENT_TRAINING_CONFIG_CANONICAL_SIZE_BYTES,
            "canonical_sha256": PARENT_TRAINING_CONFIG_CANONICAL_SHA256,
            "development_metric": "AP@[0.50:0.95,maxDets=500]",
            "development_ap": PARENT_DEVELOPMENT_AP,
            "development_ap_role": "comparison_only_not_acceptance_threshold",
        },
        "source_bindings": {
            "pretrained_authority": {
                "root_relative_path": AUTHORITY_ROOT_RELATIVE_PATH,
                "root_type": "directory",
                "root_mode": AUTHORITY_ROOT_MODE,
                "artifact_id": AUTHORITY_ARTIFACT_ID,
                "classification": AUTHORITY_CLASSIFICATION,
                "local_path_required": True,
                "network_download_forbidden": True,
                "weight": {
                    "relative_path": AUTHORITY_WEIGHT_RELATIVE_PATH,
                    "size_bytes": AUTHORITY_WEIGHT_SIZE_BYTES,
                    "sha256": AUTHORITY_WEIGHT_SHA256,
                    "mode": AUTHORITY_FILE_MODE,
                    "nlink": AUTHORITY_WEIGHT_NLINK,
                    "regular_file": True,
                    "symlink": False,
                },
                "manifest": {
                    "relative_path": AUTHORITY_MANIFEST_RELATIVE_PATH,
                    "size_bytes": AUTHORITY_MANIFEST_SIZE_BYTES,
                    "sha256": AUTHORITY_MANIFEST_SHA256,
                    "mode": AUTHORITY_FILE_MODE,
                    "nlink": AUTHORITY_MANIFEST_NLINK,
                    "regular_file": True,
                    "symlink": False,
                },
                "load_validation": {
                    "architecture": "PResNet-18-vd",
                    "state_key_count": AUTHORITY_STATE_KEY_COUNT,
                    "tensor_numel": AUTHORITY_TENSOR_NUMEL,
                    "state_inventory_sha256": AUTHORITY_STATE_INVENTORY_SHA256,
                    "missing_keys": [],
                    "unexpected_keys": [],
                    "strict_load_pass": True,
                    "cuda_initialized": False,
                },
                "declared_url": AUTHORITY_DECLARED_URL,
                "transfer": "owner-provided-byte-exact-upload",
            },
            "vendor_r18": {
                "upstream_commit": "1c8ac3f7ba84f14bd5651ab7b1b70d69a5f55f47",
                "recipe": {
                    "relative_path": AUTHORITY_VENDOR_CONFIG_RELATIVE_PATH,
                    "size_bytes": AUTHORITY_VENDOR_CONFIG_SIZE_BYTES,
                    "sha256": AUTHORITY_VENDOR_CONFIG_SHA256,
                },
                "presnet_source": {
                    "relative_path": AUTHORITY_VENDOR_SOURCE_RELATIVE_PATH,
                    "size_bytes": AUTHORITY_VENDOR_SOURCE_SIZE_BYTES,
                    "sha256": AUTHORITY_VENDOR_SOURCE_SHA256,
                },
                "optimizer_include": {
                    "relative_path": AUTHORITY_VENDOR_OPTIMIZER_RELATIVE_PATH,
                    "size_bytes": AUTHORITY_VENDOR_OPTIMIZER_SIZE_BYTES,
                    "sha256": AUTHORITY_VENDOR_OPTIMIZER_SHA256,
                },
                "effective_optimizer": {
                    "type": "AdamW",
                    "default_lr": 0.0001,
                    "betas": [0.9, 0.999],
                    "default_weight_decay": 0.0001,
                    "norm_bn_weight_decay": 0.0,
                    "explicit_selector": "norm|bn -> weight_decay=0",
                    "default_scope": "remaining_trainable",
                    "backbone_non_norm_lr_override": "forbidden",
                },
            },
            "data_roles": {
                "train_core": {
                    "role": "train_core",
                    "annotation": "train_core_coco.json",
                    "status": "allowed",
                    "identity_source": "parent_baseline_v1",
                },
                "development": {
                    "role": "development",
                    "annotation": "development_coco.json",
                    "status": "allowed",
                    "identity_source": "parent_baseline_v1",
                },
                "confirmatory": {
                    "role": "confirmatory",
                    "annotation": None,
                    "status": "sealed_and_forbidden",
                    "identity_read": "forbidden",
                },
                "test": {
                    "role": "test",
                    "annotation": None,
                    "status": "forbidden",
                    "identity_read": "forbidden",
                },
            },
        },
        "model": {
            "type": "RTDETR",
            "backbone": "PResNet-18",
            "encoder": "HybridEncoder",
            "decoder": "RTDETRTransformerv2",
            "num_classes": 10,
            "parameter_count": 20094584,
            "input_size": [640, 640],
            "num_queries": 300,
            "decoder_layers": 3,
            "nms": False,
        },
        "data_roles": {
            "train": "train_core",
            "development": "development",
            "confirmatory": "sealed_and_forbidden",
            "test": "forbidden",
        },
        "initialization": {
            "pretrained": True,
            "checkpoint": None,
            "initialization": "pretrained_authority",
            "authority_required": True,
            "random_initialization_forbidden": True,
            "seed": 0,
            "formal_run_count": 1,
            "seed_selection_forbidden": True,
            "network_download_forbidden": True,
            "external_checkpoint_forbidden": True,
        },
        "topology": {
            "device_type": "cuda",
            "visible_devices": "0",
            "world_size": 1,
            "distributed": False,
            "sync_bn": False,
            "train_micro_batch": 16,
            "gradient_accumulation_steps": 1,
            "effective_train_batch": 16,
            "train_workers": 4,
            "development_batch": 32,
            "development_workers": 4,
            "drop_last_train": True,
            "drop_last_development": False,
            "runtime_batch_adaptation_forbidden": True,
        },
        "schedule": {
            "epochs": 120,
            "steps_per_epoch_source": "train_core_dataset_length_and_drop_last",
            "checkpoint_frequency_epochs": 1,
            "development_evaluation_frequency_epochs": 1,
            "augmentation_stop_epoch": 117,
            "multiscale_enabled": False,
        },
        "optimizer": {
            "type": "AdamW",
            "parameter_name_match_mode": "case_sensitive_literal_substring",
            "parameter_groups": [
                {
                    "name": "norm_or_bn",
                    "parameter_scope": "all_trainable",
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
            "parameter_identity_audit_required": True,
            "backbone_non_norm_lr_override_forbidden": True,
            "betas": [0.9, 0.999],
            "default_lr": 0.0001,
            "default_weight_decay": 0.0001,
            "norm_bn_weight_decay": 0.0,
            "gradient_clip_max_norm": 0.1,
        },
        "learning_rate": {
            "warmup_type": "LinearWarmup",
            "warmup_optimizer_steps": 2000,
            "scheduler_type": "MultiStepLR",
            "scheduler_step_unit": "epoch",
            "milestones": [1000],
            "gamma": 0.1,
            "expected_decay_events_within_120_epochs": 0,
        },
        "amp": {
            "enabled": True,
            "autocast_dtype": "bfloat16",
            "scaler_policy": "disabled_absent",
            "grad_scaler_enabled": False,
            "nonfinite_loss_allowed": 0,
            "nonfinite_gradient_allowed": 0,
            "optimizer_skipped_steps_allowed": 0,
            "overflow_events_allowed": 0,
        },
        "ema": {
            "enabled": True,
            "decay": 0.9999,
            "warmups": 2000,
            "warmup_unit": "optimizer_updates",
            "development_evaluation_weights": "ema",
            "model_selection_weights": "ema",
            "raw_weights_must_be_preserved": True,
            "ema_weights_must_be_preserved": True,
        },
        "augmentation": {
            "transforms": [
                {"type": "RandomPhotometricDistort", "p": 0.5},
                {"type": "RandomZoomOut", "fill": 0},
                {"type": "RandomIoUCrop", "p": 0.8},
                {"type": "SanitizeBoundingBoxes", "min_size": 1},
                {"type": "RandomHorizontalFlip"},
                {"type": "Resize", "size": [640, 640]},
                {"type": "SanitizeBoundingBoxes", "min_size": 1},
                {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
                {"type": "ConvertBoxes", "fmt": "cxcywh", "normalize": True},
            ],
            "stop_epoch": 117,
            "stopped_transforms": ["RandomPhotometricDistort", "RandomZoomOut", "RandomIoUCrop"],
            "multiscale_enabled": False,
            "runtime_transform_identity_and_rng_binding_required": True,
        },
        "evaluation_and_selection": {
            "primary_evaluator": "visdrone_official_primary_evaluator_v1",
            "primary_protocol": "visdrone_official_style_v1",
            "primary_evaluator_required_before_training": True,
            "primary_evaluator_independently_certified": True,
            "secondary_evaluator": "coco_secondary_vendor_v1",
            "secondary_diagnostic_only": True,
            "secondary_cannot_certify_or_select": True,
            "development_only": True,
            "evaluation_frequency_epochs": 1,
            "selection_metric": "AP@[0.50:0.95,maxDets=500]",
            "selection_direction": "maximize",
            "tie_breakers": ["AP50", "AR500", "earlier_epoch"],
            "metric_comparison_precision": "unrounded_float64",
            "accuracy_threshold": None,
        },
        "checkpoint_policy": {
            "last_checkpoint": "every epoch",
            "best_checkpoint": "on deterministic development selection improvement",
            "periodic_checkpoint": "every 10 epochs",
            "final_checkpoint": "epoch 120",
            "resume": False,
            "retry": False,
            "overwrite": False,
            "exactly_once_run_identity": True,
            "interrupted_run_status": "PERMANENT_FAIL",
            "required_state": [
                "raw_model",
                "ema",
                "optimizer",
                "scheduler",
                "warmup",
                "amp_state",
                "epoch",
                "global_optimizer_step",
                "rng_states",
                "config_identity",
                "source_identity",
                "environment_identity",
            ],
            "last_checkpoint_every_epoch": True,
            "best_checkpoint_on_selection_improvement": True,
            "periodic_checkpoint_frequency_epochs": 10,
            "final_epoch_checkpoint": True,
            "atomic_write_required": True,
            "file_fsync_required": True,
            "directory_fsync_required": True,
            "readback_required": True,
            "loadability_check_required": True,
            "sha256_required": True,
            "inventory_required": True,
        },
        "acceptance": {
            "separate_states": [
                "training_runtime_integrity",
                "training_evidence_integrity",
                "checkpoint_integrity",
                "development_accuracy_reported",
                "model_selection_certified",
                "confirmatory_access",
                "test_access",
                "speed_measurement",
            ],
            "required_epochs_complete": 120,
            "expected_optimizer_steps_complete": True,
            "loss_and_gradients_finite": True,
            "amp_skip_or_overflow_allowed": False,
            "required_exit_code": 0,
            "completion_evidence_and_checkpoints_bound": True,
            "checkpoints_loadable": True,
            "development_accuracy_reporting_only": True,
            "confirmatory_metrics_accessed": False,
            "test_accessed": False,
            "speed_measured": False,
        },
        "environment_identity": {
            "gpu_name": "NVIDIA GeForce RTX 4090 D",
            "cuda_version": "12.4",
            "graphics_clock_max_mhz": 1500,
            "exclusive_host_observation": "required_before_training",
            "other_training_load_observation": "required_zero_before_training",
            "observation_source": "hardware_preflight",
            "probe_executed_in_t7b": False,
        },
        "hardware_launch_gate": {
            "required": True,
            "exclusive_host_required": True,
            "no_other_training_load_required": True,
            "gpu_name": "NVIDIA GeForce RTX 4090 D",
            "cuda_version": "12.4",
            "graphics_clock_upper_limit_mhz": 1500,
            "probe_policy": "record_and_validate_contract_only",
            "probe_executed_in_t7b": False,
        },
        "runtime_closure": {
            "process_stdout_terminal_json": {
                "required": True,
                "implemented": False,
                "independently_certified": False,
            },
            "preflight_training_parent_identity": {
                "required": True,
                "implemented": False,
                "independently_certified": False,
            },
            "owner_authorization_data_roots": {
                "required": True,
                "implemented": False,
                "independently_certified": False,
            },
            "environment_identity_binding": {
                "required": True,
                "implemented": False,
                "independently_certified": False,
            },
            "epoch_progress_evidence": {
                "required": True,
                "implemented": False,
                "independently_certified": False,
            },
        },
        "readiness": {
            "contract_implemented": True,
            "pretrained_authority_binding_implemented": True,
            "bf16_no_gradscaler_contract_implemented": True,
            "vendor_r18_optimizer_contract_implemented": True,
            "runtime_closure_requirements_recorded": True,
            "training_implementation_ready": False,
            "training_ready": False,
            "model_selection_certified": False,
            "independent_audit_pass": False,
        },
    }


_FROZEN_CONTRACT = _build_frozen_contract()


def _pointer(parent: str, child: str | int) -> str:
    token = str(child).replace("~", "~0").replace("/", "~1")
    return f"{parent}/{token}" if parent else f"/{token}"


def _walk_leaves(value: Any, pointer: str = "") -> list[tuple[str, Any]]:
    if type(value) is dict:
        rows: list[tuple[str, Any]] = []
        for key, child in value.items():
            rows.extend(_walk_leaves(child, _pointer(pointer, key)))
        return rows
    if type(value) is list:
        rows = []
        for index, child in enumerate(value):
            rows.extend(_walk_leaves(child, _pointer(pointer, index)))
        return rows
    return [(pointer, value)]


def _schema_for(value: Any) -> tuple[str, Any]:
    if type(value) is dict:
        return ("object", {key: _schema_for(child) for key, child in value.items()})
    if type(value) is list:
        return ("list", tuple(_schema_for(child) for child in value))
    return ("scalar", type(value))


_SCHEMA = _schema_for(_FROZEN_CONTRACT)
_FROZEN_LEAVES = dict(_walk_leaves(_FROZEN_CONTRACT))


def _numeric_rule(value: int | float) -> dict[str, Any]:
    return {
        "expected_type": type(value),
        "expected": value,
        "finite": type(value) is float,
        "lower": 0,
    }


_NUMERIC_CONSTRAINTS = {
    pointer: _numeric_rule(value)
    for pointer, value in _FROZEN_LEAVES.items()
    if type(value) in {int, float}
}


def _is_path_pointer(pointer: str) -> bool:
    key = pointer.rsplit("/", 1)[-1]
    return key.endswith("path") or key.endswith("_path") or key in {"artifact_root"}


def _is_sha_pointer(pointer: str) -> bool:
    key = pointer.rsplit("/", 1)[-1]
    return key == "sha256" or key.endswith("_sha256")


def _is_git_pointer(pointer: str) -> bool:
    key = pointer.rsplit("/", 1)[-1]
    return key.endswith("_commit") or key.endswith("_tree")


_PATH_SHA_LEAVES = {
    pointer: value
    for pointer, value in _FROZEN_LEAVES.items()
    if _is_path_pointer(pointer) or _is_sha_pointer(pointer) or _is_git_pointer(pointer)
}
_SEMANTIC_LEAVES = {
    pointer: value
    for pointer, value in _FROZEN_LEAVES.items()
    if pointer not in _NUMERIC_CONSTRAINTS and pointer not in _PATH_SHA_LEAVES
}

_RELATION_RULES = (
    "parent_identity_role",
    "authority_path_membership",
    "data_role_partition",
    "batch_arithmetic",
    "schedule_alignment",
    "optimizer_partition",
    "amp_policy",
    "hardware_binding",
    "runtime_closure_fail_closed",
    "readiness_fail_closed",
)


def _error(message: str) -> None:
    raise TrainingV2AContractError(message)


def _assert_builtin_json(value: Any, field: str = "v2a contract") -> None:
    if type(value) is dict:
        for key, child in value.items():
            if type(key) is not str:
                _error(f"{field} contains a non-string key")
            _assert_builtin_json(child, f"{field}.{key}")
        return
    if type(value) is list:
        for index, child in enumerate(value):
            _assert_builtin_json(child, f"{field}[{index}]")
        return
    if type(value) not in {str, int, float, bool, type(None)}:
        _error(f"{field} has a non-builtin JSON scalar")


def _validate_schema(value: Any, schema: tuple[str, Any], pointer: str = "") -> None:
    kind, detail = schema
    location = pointer or "/"
    if kind == "object":
        if type(value) is not dict:
            _error(f"closed v2a schema object type mismatch at {location}")
        actual = set(value)
        expected = set(detail)
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing and extra:
            _error(f"closed v2a schema renamed keys at {location}: missing={missing} extra={extra}")
        if missing:
            _error(f"closed v2a schema missing keys at {location}: {missing}")
        if extra:
            _error(f"closed v2a schema extra keys at {location}: {extra}")
        for key, child_schema in detail.items():
            _validate_schema(value[key], child_schema, _pointer(pointer, key))
        return
    if kind == "list":
        if type(value) is not list:
            _error(f"closed v2a schema list type mismatch at {location}")
        if len(value) != len(detail):
            _error(f"closed v2a schema list length mismatch at {location}")
        for index, child_schema in enumerate(detail):
            _validate_schema(value[index], child_schema, _pointer(pointer, index))
        return
    if type(value) is not detail:
        _error(f"closed v2a schema scalar type mismatch at {location}: expected={detail.__name__}")


def _validate_owner_registry() -> None:
    leaves = set(_FROZEN_LEAVES)
    numeric = set(_NUMERIC_CONSTRAINTS)
    path_sha = set(_PATH_SHA_LEAVES)
    semantic = set(_SEMANTIC_LEAVES)
    owners = numeric | path_sha | semantic
    if len(owners) != len(numeric) + len(path_sha) + len(semantic) or owners != leaves:
        _error(
            "v2a scalar owner registry drift: "
            f"missing={sorted(leaves - owners)} extra={sorted(owners - leaves)}"
        )


def _validate_numeric_constraints(config: dict[str, Any]) -> None:
    actual = {
        pointer: value
        for pointer, value in _walk_leaves(config)
        if type(value) in {int, float}
    }
    if set(actual) != set(_NUMERIC_CONSTRAINTS):
        _error(
            "v2a numeric registry coverage drift: "
            f"missing={sorted(set(_NUMERIC_CONSTRAINTS) - set(actual))} "
            f"extra={sorted(set(actual) - set(_NUMERIC_CONSTRAINTS))}"
        )
    for pointer, rule in _NUMERIC_CONSTRAINTS.items():
        value = actual[pointer]
        if type(value) is not rule["expected_type"]:
            _error(f"v2a numeric type mismatch at {pointer}")
        if rule["finite"] and not math.isfinite(value):
            _error(f"v2a numeric nonfinite at {pointer}")
        if value < rule["lower"] or value != rule["expected"]:
            _error(f"v2a numeric frozen literal mismatch at {pointer}")


def _assert_relative_path(value: Any, field: str) -> str:
    if type(value) is not str or not value or "\x00" in value or "\\" in value:
        _error(f"invalid relative path at {field}")
    if value.startswith("/") or value.endswith("/") or "//" in value:
        _error(f"invalid relative path at {field}")
    components = value.split("/")
    if any(not component or component in {".", ".."} for component in components):
        _error(f"invalid relative path at {field}")
    return value


def _assert_sha(value: Any, field: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        _error(f"invalid SHA-256 at {field}")
    return value


def _assert_git_oid(value: Any, field: str) -> str:
    if type(value) is not str or _GIT_OID.fullmatch(value) is None:
        _error(f"invalid Git identity at {field}")
    return value


def _validate_path_sha_leaves(config: dict[str, Any]) -> None:
    actual = dict(_walk_leaves(config))
    if set(actual) & set(_PATH_SHA_LEAVES) != set(_PATH_SHA_LEAVES):
        _error("v2a path/SHA registry coverage drift")
    for pointer, expected in _PATH_SHA_LEAVES.items():
        value = actual[pointer]
        if _is_path_pointer(pointer):
            _assert_relative_path(value, f"v2a {pointer}")
        if _is_sha_pointer(pointer):
            _assert_sha(value, f"v2a {pointer}")
        if _is_git_pointer(pointer):
            _assert_git_oid(value, f"v2a {pointer}")
        if type(value) is not type(expected) or value != expected:
            _error(f"v2a path/SHA frozen literal mismatch at {pointer}")


def _validate_semantic_leaves(config: dict[str, Any]) -> None:
    actual = dict(_walk_leaves(config))
    for pointer, expected in _SEMANTIC_LEAVES.items():
        value = actual[pointer]
        if type(value) is not type(expected) or value != expected:
            _error(f"v2a semantic frozen literal mismatch at {pointer}")


def _value(config: dict[str, Any], pointer: str) -> Any:
    current: Any = config
    for token in pointer.strip("/").split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if type(current) is dict and token in current:
            current = current[token]
        elif type(current) is list and token.isdigit() and int(token) < len(current):
            current = current[int(token)]
        else:
            return _MISSING
    return current


def _validate_relations(config: dict[str, Any]) -> None:
    parent = config["parent_baseline_v1"]
    if parent["training_contract_id"] != PARENT_TRAINING_CONTRACT_ID or parent["baseline_id"] != PARENT_BASELINE_ID:
        _error("v2a parent baseline identity drift")
    if parent["development_ap_role"] != "comparison_only_not_acceptance_threshold":
        _error("v2a development AP is not comparison-only")

    authority = config["source_bindings"]["pretrained_authority"]
    if not authority["local_path_required"] or not authority["network_download_forbidden"]:
        _error("v2a pretrained authority is not local-only")
    if not authority["weight"]["relative_path"].startswith(AUTHORITY_ROOT_RELATIVE_PATH + "/"):
        _error("v2a weight is outside authority root")
    if not authority["manifest"]["relative_path"].startswith(AUTHORITY_ROOT_RELATIVE_PATH + "/"):
        _error("v2a manifest is outside authority root")
    validation = authority["load_validation"]
    if (
        validation["missing_keys"]
        or validation["unexpected_keys"]
        or validation["strict_load_pass"] is not True
        or validation["cuda_initialized"] is not False
    ):
        _error("v2a authority strict-load declaration is not certified")

    roles = config["source_bindings"]["data_roles"]
    if roles["train_core"]["role"] == roles["development"]["role"]:
        _error("v2a train and development roles overlap")
    if roles["confirmatory"]["status"] != "sealed_and_forbidden" or roles["test"]["status"] != "forbidden":
        _error("v2a sealed data role drift")
    if config["data_roles"] != {
        "train": "train_core",
        "development": "development",
        "confirmatory": "sealed_and_forbidden",
        "test": "forbidden",
    }:
        _error("v2a logical data role drift")

    topology = config["topology"]
    if topology["train_micro_batch"] * topology["gradient_accumulation_steps"] * topology["world_size"] != topology["effective_train_batch"]:
        _error("v2a effective batch arithmetic drift")
    if topology["train_micro_batch"] != 16 or topology["development_batch"] != 32:
        _error("v2a B1 batch policy drift")
    if topology["runtime_batch_adaptation_forbidden"] is not True:
        _error("v2a runtime batch adaptation is not forbidden")

    schedule = config["schedule"]
    if schedule["epochs"] != config["acceptance"]["required_epochs_complete"]:
        _error("v2a schedule/acceptance epoch drift")
    if schedule["augmentation_stop_epoch"] != config["augmentation"]["stop_epoch"]:
        _error("v2a augmentation stop relation drift")
    if schedule["development_evaluation_frequency_epochs"] != config["evaluation_and_selection"]["evaluation_frequency_epochs"]:
        _error("v2a development evaluation frequency drift")
    lr = config["learning_rate"]
    if sum(item < schedule["epochs"] for item in lr["milestones"]) != lr["expected_decay_events_within_120_epochs"]:
        _error("v2a scheduler decay event drift")
    if schedule["epochs"] % config["checkpoint_policy"]["periodic_checkpoint_frequency_epochs"] != 0:
        _error("v2a checkpoint interval drift")

    optimizer = config["optimizer"]
    groups = optimizer["parameter_groups"]
    if tuple(group["name"] for group in groups) != ("norm_or_bn", "default"):
        _error("v2a optimizer group order drift")
    if optimizer["mutually_exclusive"] is not True or optimizer["cover_all_trainable_parameters"] is not True or optimizer["parameter_identity_audit_required"] is not True:
        _error("v2a optimizer coverage policy drift")
    if groups[0]["include_substrings"] != ["norm", "bn"] or groups[0]["weight_decay"] != 0.0:
        _error("v2a optimizer explicit norm/bn selector drift")
    if groups[1]["parameter_scope"] != "remaining_trainable" or groups[1]["include_substrings"] or groups[1]["exclude_substrings"]:
        _error("v2a optimizer default scope drift")
    if optimizer["backbone_non_norm_lr_override_forbidden"] is not True:
        _error("v2a backbone learning-rate override is not forbidden")

    amp = config["amp"]
    if amp["enabled"] is not True or amp["autocast_dtype"] != "bfloat16" or amp["grad_scaler_enabled"] is not False or amp["scaler_policy"] != "disabled_absent":
        _error("v2a AMP/scaler policy drift")
    forbidden_scaler_keys = {"init_scale", "growth_factor", "backoff_factor", "growth_interval", "scaler_type"}
    if any(key in amp for key in forbidden_scaler_keys):
        _error("v2a AMP retains forbidden scaler parameters")
    if any(amp[key] != 0 for key in ("nonfinite_loss_allowed", "nonfinite_gradient_allowed", "optimizer_skipped_steps_allowed", "overflow_events_allowed")):
        _error("v2a AMP failure allowance drift")

    hardware = config["hardware_launch_gate"]
    environment = config["environment_identity"]
    if hardware["gpu_name"] != environment["gpu_name"] or hardware["cuda_version"] != environment["cuda_version"]:
        _error("v2a hardware/environment identity drift")
    if hardware["graphics_clock_upper_limit_mhz"] != environment["graphics_clock_max_mhz"]:
        _error("v2a graphics clock binding drift")
    if hardware["probe_executed_in_t7b"] is not False or environment["probe_executed_in_t7b"] is not False:
        _error("v2a T7B hardware probe was declared executed")
    if hardware["exclusive_host_required"] is not True or hardware["no_other_training_load_required"] is not True:
        _error("v2a exclusive-host gate drift")

    for requirement in config["runtime_closure"].values():
        if requirement["required"] is not True or requirement["implemented"] is not False or requirement["independently_certified"] is not False:
            _error("v2a runtime closure is not fail-closed")
    readiness = config["readiness"]
    if any(readiness[field] is not False for field in ("training_implementation_ready", "training_ready", "model_selection_certified", "independent_audit_pass")):
        _error("v2a readiness is not fail-closed")


def _validate_owner_and_relations(config: dict[str, Any]) -> None:
    _validate_owner_registry()
    _validate_numeric_constraints(config)
    _validate_path_sha_leaves(config)
    _validate_semantic_leaves(config)
    _validate_relations(config)


def canonical_training_v2a_contract_bytes(config: Any) -> bytes:
    """Return deterministic JSON bytes without a trailing LF."""

    _assert_builtin_json(config)
    try:
        return json.dumps(
            config,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TrainingV2AContractError("v2a contract is not finite JSON data") from exc


def _validate_frozen_digest(config: dict[str, Any]) -> None:
    canonical = canonical_training_v2a_contract_bytes(config)
    if V2A_CONFIG_CANONICAL_SIZE_BYTES <= 0 or len(canonical) != V2A_CONFIG_CANONICAL_SIZE_BYTES:
        _error("v2a canonical size identity drift")
    if hashlib.sha256(canonical).hexdigest() != V2A_CONFIG_CANONICAL_SHA256:
        _error("v2a canonical identity drift")


def _validate_parent_config(parent_config: Any) -> None:
    _assert_builtin_json(parent_config, "parent baseline v1")
    if type(parent_config) is not dict:
        _error("parent baseline v1 must be an exact dict")
    canonical = canonical_training_v2a_contract_bytes(parent_config)
    if len(canonical) != PARENT_TRAINING_CONFIG_CANONICAL_SIZE_BYTES or hashlib.sha256(canonical).hexdigest() != PARENT_TRAINING_CONFIG_CANONICAL_SHA256:
        _error("parent baseline v1 canonical identity drift")
    if parent_config.get("training_contract_id") != PARENT_TRAINING_CONTRACT_ID or parent_config.get("baseline_id") != PARENT_BASELINE_ID:
        _error("parent baseline v1 semantic identity drift")


def validate_training_v2a_contract(config: Any, parent_config: Any | None = None) -> dict[str, Any]:
    """Validate detached v2a content and return a deep copy."""

    _assert_builtin_json(config)
    if type(config) is not dict:
        _error("v2a contract must be an exact dict")
    _validate_schema(config, _SCHEMA)
    _validate_owner_and_relations(config)
    _validate_frozen_digest(config)
    if parent_config is not None:
        _validate_parent_config(parent_config)
    return copy.deepcopy(config)


def _assert_repository_root(value: str | Path) -> str:
    try:
        root = os.fspath(value)
    except TypeError as exc:
        raise TrainingV2AContractError("repo_root must be an absolute lexical path") from exc
    if type(root) is not str or not root or "\x00" in root or "\\" in root:
        _error("repo_root must be an absolute lexical path")
    if root != "/" and (not root.startswith("/") or root.endswith("/") or "//" in root):
        _error("repo_root must be an absolute lexical path")
    components = [] if root == "/" else root[1:].split("/")
    if any(not component or component in {".", ".."} for component in components):
        _error("repo_root must be an absolute lexical path")
    return root


def _snapshot(info: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _identity(info: os.stat_result) -> dict[str, int]:
    return {
        "device": info.st_dev,
        "inode": info.st_ino,
        "mode": stat.S_IMODE(info.st_mode),
        "nlink": info.st_nlink,
        "size_bytes": info.st_size,
        "uid": info.st_uid,
        "gid": info.st_gid,
    }


class _VerifiedRepository:
    """Descriptor-relative read boundary for v2a repository inputs."""

    def __init__(self, repo_root: str | Path) -> None:
        self.lexical_root = _assert_repository_root(repo_root)
        required = ("O_DIRECTORY", "O_NOFOLLOW", "O_CLOEXEC", "O_NONBLOCK")
        if any(not hasattr(os, name) or not getattr(os, name) for name in required):
            _error("secure repository access is unavailable")
        self.directory_flags = os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        self.file_flags = os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
        self.fds: list[int] = []
        self.root_fd: int | None = None
        self.root_device: int | None = None

    def __enter__(self) -> "_VerifiedRepository":
        try:
            self.root_fd = os.open("/", self.directory_flags)
            self.fds.append(self.root_fd)
            parent_fd = self.root_fd
            for component in ([] if self.lexical_root == "/" else self.lexical_root[1:].split("/")):
                observed = os.lstat(component, dir_fd=parent_fd)
                if not stat.S_ISDIR(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
                    _error("repository root contains a non-directory or symlink component")
                child_fd = os.open(component, self.directory_flags, dir_fd=parent_fd)
                self.fds.append(child_fd)
                opened = os.fstat(child_fd)
                if (observed.st_dev, observed.st_ino, stat.S_IFMT(observed.st_mode)) != (opened.st_dev, opened.st_ino, stat.S_IFMT(opened.st_mode)):
                    _error("repository root component changed during open")
                parent_fd = child_fd
            self.root_fd = parent_fd
            self.root_device = os.fstat(parent_fd).st_dev
            return self
        except BaseException:
            self._close()
            raise

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        self._close()
        return False

    def _close(self) -> None:
        for fd in reversed(self.fds):
            try:
                os.close(fd)
            except OSError:
                pass
        self.fds.clear()

    def _parts(self, relative: str | Path) -> list[str]:
        value = os.fspath(relative)
        if type(value) is not str:
            _error("repository relative path must be text")
        _assert_relative_path(value, "repository relative path")
        return value.split("/")

    def _open_path(self, relative: str | Path, final_directory: bool = False) -> tuple[int, list[int]]:
        if self.root_fd is None or self.root_device is None:
            _error("repository boundary is not open")
        parts = self._parts(relative)
        owned: list[int] = []
        parent_fd = self.root_fd
        try:
            for component in parts[:-1] if not final_directory else parts:
                observed = os.lstat(component, dir_fd=parent_fd)
                if not stat.S_ISDIR(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
                    _error(f"repository path component is not a real directory: {component}")
                if observed.st_dev != self.root_device:
                    _error("repository path crossed devices")
                fd = os.open(component, self.directory_flags, dir_fd=parent_fd)
                owned.append(fd)
                opened = os.fstat(fd)
                if (observed.st_dev, observed.st_ino, stat.S_IFMT(observed.st_mode)) != (opened.st_dev, opened.st_ino, stat.S_IFMT(opened.st_mode)):
                    _error("repository directory changed during open")
                parent_fd = fd
            if final_directory:
                return parent_fd, owned
            final_name = parts[-1]
            observed = os.lstat(final_name, dir_fd=parent_fd)
            if observed.st_dev != self.root_device:
                _error("repository file crossed devices")
            fd = os.open(final_name, self.file_flags, dir_fd=parent_fd)
            owned.append(fd)
            opened = os.fstat(fd)
            return fd, owned
        except TrainingV2AContractError:
            for fd in reversed(owned):
                try:
                    os.close(fd)
                except OSError:
                    pass
            raise
        except OSError as exc:
            for fd in reversed(owned):
                try:
                    os.close(fd)
                except OSError:
                    pass
            raise TrainingV2AContractError(f"secure repository path access failed: {relative}") from exc
        except BaseException:
            for fd in reversed(owned):
                try:
                    os.close(fd)
                except OSError:
                    pass
            raise

    def read_file(self, relative: str | Path) -> tuple[bytes, dict[str, int]]:
        fd, owned = self._open_path(relative)
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode) or before.st_nlink != 1:
                _error(f"repository input is not a regular single-link file: {relative}")
            digest = hashlib.sha256()
            chunks: list[bytes] = []
            while True:
                chunk = os.read(fd, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
                digest.update(chunk)
            after = os.fstat(fd)
            if _snapshot(before) != _snapshot(after):
                _error(f"repository file changed during read: {relative}")
            return b"".join(chunks), _identity(after)
        finally:
            for item in reversed(owned):
                try:
                    os.close(item)
                except OSError:
                    pass

    def observe_directory(self, relative: str | Path) -> dict[str, Any]:
        fd, owned = self._open_path(relative, final_directory=True)
        try:
            info = os.fstat(fd)
            if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
                _error(f"repository authority root is not a directory: {relative}")
            names = sorted(os.listdir(fd))
            rows: list[dict[str, Any]] = []
            for name in names:
                item = os.lstat(name, dir_fd=fd)
                rows.append({"name": name, "identity": _identity(item), "type_mode": stat.S_IFMT(item.st_mode)})
            after = os.fstat(fd)
            if _snapshot(info) != _snapshot(after):
                _error(f"repository directory changed during observation: {relative}")
            return {"identity": _identity(after), "rows": rows}
        finally:
            for item in reversed(owned):
                try:
                    os.close(item)
                except OSError:
                    pass


def _strict_json(raw: bytes, label: str, trailing_lf: bool) -> dict[str, Any]:
    if b"\x00" in raw or b"\r" in raw or raw.startswith(b"\xef\xbb\xbf"):
        _error(f"{label} has non-portable bytes")
    if trailing_lf:
        if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
            _error(f"{label} must have exactly one trailing LF")
        payload = raw[:-1]
    else:
        payload = raw

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                _error(f"{label} contains a duplicate key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda name: (_ for _ in ()).throw(ValueError(name)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise TrainingV2AContractError(f"{label} is not strict UTF-8 JSON") from exc
    _assert_builtin_json(value, label)
    if type(value) is not dict:
        _error(f"{label} root must be an exact dict")
    return value


def _read_config(repo: _VerifiedRepository, relative: str) -> tuple[dict[str, Any], bytes, dict[str, int]]:
    raw, identity = repo.read_file(relative)
    if relative == V2A_CONFIG_RELATIVE_PATH:
        if len(raw) != V2A_CONFIG_RAW_SIZE_BYTES or hashlib.sha256(raw).hexdigest() != V2A_CONFIG_RAW_SHA256:
            _error("v2a config raw identity drift")
        return _strict_json(raw, "v2a config", trailing_lf=True), raw, identity
    return _strict_json(raw, "parent baseline v1", trailing_lf=True), raw, identity


def load_training_v2a_contract(
    repo_root: str | Path,
    config_path: str | Path = V2A_CONFIG_RELATIVE_PATH,
) -> dict[str, Any]:
    """Load and validate only portable contract bytes; authority is separate."""

    relative = os.fspath(config_path)
    _assert_relative_path(relative, "v2a config_path")
    if relative != V2A_CONFIG_RELATIVE_PATH:
        _error("v2a config path identity drift")
    with _VerifiedRepository(repo_root) as repository:
        config, _raw, _identity_value = _read_config(repository, relative)
        parent, parent_raw, _parent_identity = _read_config(repository, PARENT_TRAINING_CONFIG_RELATIVE_PATH)
    if len(parent_raw) != PARENT_TRAINING_CONFIG_RAW_SIZE_BYTES or hashlib.sha256(parent_raw).hexdigest() != PARENT_TRAINING_CONFIG_RAW_SHA256:
        _error("parent baseline v1 raw identity drift")
    validated = validate_training_v2a_contract(config, parent)
    return copy.deepcopy(validated)


def _expected_authority_manifest() -> dict[str, Any]:
    return {
        "artifact": {
            "mode": AUTHORITY_FILE_MODE,
            "relative_path": AUTHORITY_WEIGHT_RELATIVE_PATH,
            "sha256": AUTHORITY_WEIGHT_SHA256,
            "size_bytes": AUTHORITY_WEIGHT_SIZE_BYTES,
        },
        "artifact_id": AUTHORITY_ARTIFACT_ID,
        "classification": AUTHORITY_CLASSIFICATION,
        "load_validation": {
            "architecture": "PResNet-18-vd",
            "constructor": {
                "depth": 18,
                "freeze_at": -1,
                "freeze_norm": False,
                "num_stages": 4,
                "pretrained": False,
                "return_idx": [1, 2, 3],
                "variant": "d",
            },
            "cuda_initialized": False,
            "missing_keys": [],
            "state_inventory_sha256": AUTHORITY_STATE_INVENTORY_SHA256,
            "state_key_count": AUTHORITY_STATE_KEY_COUNT,
            "strict_load_pass": True,
            "tensor_numel": AUTHORITY_TENSOR_NUMEL,
            "torch_version": "2.4.1",
            "unexpected_keys": [],
        },
        "repository": {
            "branch": "codex/p3-rtdetrv2-baseline-training-t6b-vendor-component-config-r10-fix",
            "head": "12e54cdd04b3e50285f9890a81840e2d527d2d8f",
            "tree": "3485642b24b1eecfeb37c952517daff95b44d214",
        },
        "runtime_policy": {"local_path_required": True, "network_download": False},
        "schema_version": 1,
        "source": {"declared_url": AUTHORITY_DECLARED_URL, "transfer": "owner-provided-byte-exact-upload"},
        "vendor_config": {
            "relative_path": AUTHORITY_VENDOR_CONFIG_RELATIVE_PATH,
            "sha256": AUTHORITY_VENDOR_CONFIG_SHA256,
            "size_bytes": AUTHORITY_VENDOR_CONFIG_SIZE_BYTES,
        },
        "vendor_source": {
            "relative_path": AUTHORITY_VENDOR_SOURCE_RELATIVE_PATH,
            "sha256": AUTHORITY_VENDOR_SOURCE_SHA256,
            "size_bytes": AUTHORITY_VENDOR_SOURCE_SIZE_BYTES,
        },
    }


def _validate_authority_manifest(raw: bytes) -> dict[str, Any]:
    try:
        value = _strict_json(raw, "pretrained authority manifest", trailing_lf=False)
    except TrainingV2AContractError:
        raise
    if value != _expected_authority_manifest():
        _error("pretrained authority manifest content drift")
    return copy.deepcopy(value)


def _bound_file_identity(
    identity: dict[str, int], expected_size: int, expected_sha: str, expected_mode: int, expected_nlink: int, label: str
) -> dict[str, Any]:
    if identity["size_bytes"] != expected_size or identity["mode"] != expected_mode or identity["nlink"] != expected_nlink:
        _error(f"{label} metadata identity drift")
    if identity["uid"] != os.getuid() or identity["gid"] != os.getgid():
        _error(f"{label} ownership identity drift")
    return {"size_bytes": expected_size, "sha256": expected_sha, **identity}


def _authority_binding(repository: _VerifiedRepository, contract: dict[str, Any]) -> dict[str, Any]:
    authority = contract["source_bindings"]["pretrained_authority"]
    before = repository.observe_directory(AUTHORITY_ROOT_RELATIVE_PATH)
    expected_names = sorted(
        [AUTHORITY_WEIGHT_RELATIVE_PATH.rsplit("/", 1)[-1], AUTHORITY_MANIFEST_RELATIVE_PATH.rsplit("/", 1)[-1]]
    )
    if [row["name"] for row in before["rows"]] != expected_names:
        _error("pretrained authority directory entry set drift")
    weight_raw, weight_identity = repository.read_file(AUTHORITY_WEIGHT_RELATIVE_PATH)
    manifest_raw, manifest_identity = repository.read_file(AUTHORITY_MANIFEST_RELATIVE_PATH)
    if len(weight_raw) != AUTHORITY_WEIGHT_SIZE_BYTES or hashlib.sha256(weight_raw).hexdigest() != AUTHORITY_WEIGHT_SHA256:
        _error("pretrained authority weight content identity drift")
    if len(manifest_raw) != AUTHORITY_MANIFEST_SIZE_BYTES or hashlib.sha256(manifest_raw).hexdigest() != AUTHORITY_MANIFEST_SHA256:
        _error("pretrained authority manifest raw identity drift")
    manifest = _validate_authority_manifest(manifest_raw)
    after = repository.observe_directory(AUTHORITY_ROOT_RELATIVE_PATH)
    if before != after:
        _error("pretrained authority changed during binding")
    if before["identity"]["mode"] != AUTHORITY_ROOT_MODE or before["identity"]["uid"] != os.getuid() or before["identity"]["gid"] != os.getgid():
        _error("pretrained authority root metadata identity drift")
    return {
        "root_relative_path": AUTHORITY_ROOT_RELATIVE_PATH,
        "root_identity": before["identity"],
        "before_after_identity_pass": True,
        "weight": _bound_file_identity(weight_identity, AUTHORITY_WEIGHT_SIZE_BYTES, AUTHORITY_WEIGHT_SHA256, AUTHORITY_FILE_MODE, AUTHORITY_WEIGHT_NLINK, "pretrained authority weight"),
        "manifest": _bound_file_identity(manifest_identity, AUTHORITY_MANIFEST_SIZE_BYTES, AUTHORITY_MANIFEST_SHA256, AUTHORITY_FILE_MODE, AUTHORITY_MANIFEST_NLINK, "pretrained authority manifest"),
        "manifest_load_validation": copy.deepcopy(manifest["load_validation"]),
    }


def _source_binding(repository: _VerifiedRepository, relative: str, size: int, digest: str, label: str) -> dict[str, Any]:
    raw, identity = repository.read_file(relative)
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != digest:
        _error(f"{label} content identity drift")
    return {"relative_path": relative, "size_bytes": size, "sha256": digest, "identity": identity}


def training_v2a_contract_binding(
    repo_root: str | Path,
    config_path: str | Path = V2A_CONFIG_RELATIVE_PATH,
) -> dict[str, Any]:
    """Bind the validated contract to its local authority and vendor inputs."""

    validated = load_training_v2a_contract(repo_root, config_path)
    with _VerifiedRepository(repo_root) as repository:
        vendor = validated["source_bindings"]["vendor_r18"]
        recipe = vendor["recipe"]
        presnet = vendor["presnet_source"]
        optimizer = vendor["optimizer_include"]
        recipe_binding = _source_binding(repository, recipe["relative_path"], recipe["size_bytes"], recipe["sha256"], "vendor R18 recipe")
        presnet_binding = _source_binding(repository, presnet["relative_path"], presnet["size_bytes"], presnet["sha256"], "vendor PResNet source")
        optimizer_binding = _source_binding(repository, optimizer["relative_path"], optimizer["size_bytes"], optimizer["sha256"], "vendor optimizer include")
        authority_binding = _authority_binding(repository, validated)
    return {
        "schema_version": 1,
        "contract_id": V2A_CONTRACT_ID,
        "baseline_id": V2A_BASELINE_ID,
        "relative_path": V2A_CONFIG_RELATIVE_PATH,
        "raw_size_bytes": V2A_CONFIG_RAW_SIZE_BYTES,
        "raw_sha256": V2A_CONFIG_RAW_SHA256,
        "canonical_size_bytes": V2A_CONFIG_CANONICAL_SIZE_BYTES,
        "canonical_sha256": V2A_CONFIG_CANONICAL_SHA256,
        "parent_baseline_v1": copy.deepcopy(validated["parent_baseline_v1"]),
        "authority_binding": authority_binding,
        "vendor_r18_binding": {
            "upstream_commit": vendor["upstream_commit"],
            "recipe": recipe_binding,
            "presnet_source": presnet_binding,
            "optimizer_include": optimizer_binding,
            "effective_optimizer": copy.deepcopy(vendor["effective_optimizer"]),
        },
        "data_roles": copy.deepcopy(validated["source_bindings"]["data_roles"]),
        "contract": copy.deepcopy(validated),
    }


def resolve_optimizer_parameter_group(parameter_name: str) -> str:
    """Resolve one parameter name using the sole explicit v2a selector."""

    if type(parameter_name) is not str or not parameter_name:
        _error("parameter name must be a non-empty exact string")
    if "norm" in parameter_name or "bn" in parameter_name:
        return "norm_or_bn"
    return "default"


def audit_optimizer_parameter_groups(parameter_names: Iterable[str]) -> dict[str, Any]:
    """Return an identity-auditable, mutually-exclusive parameter partition."""

    try:
        names = list(parameter_names)
    except TypeError as exc:
        raise TrainingV2AContractError("parameter names must be iterable") from exc
    if any(type(name) is not str or not name for name in names):
        _error("parameter names must be unique non-empty exact strings")
    if len(set(names)) != len(names):
        _error("parameter names contain duplicate identities")
    rows = [{"parameter_name": name, "group": resolve_optimizer_parameter_group(name)} for name in names]
    counts = {"norm_or_bn": sum(row["group"] == "norm_or_bn" for row in rows), "default": sum(row["group"] == "default" for row in rows)}
    return {
        "parameter_rows": rows,
        "group_counts": counts,
        "parameter_count": len(rows),
        "mutually_exclusive": True,
        "covered_all_parameters": len(rows) == sum(counts.values()),
        "identity_key": "parameter_name",
    }
