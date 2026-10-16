"""Portable, pre-CUDA validation for the frozen T1 runtime plan."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from typing import Any

import sparse_rtdetr.baseline.training_contract as _training_contract


RUNTIME_PLAN_CONFIG_RELATIVE_PATH = "configs/baseline/rtdetrv2_r18_visdrone_training_runtime_v1.json"
RUNTIME_PLAN_ID = "rtdetrv2_r18_visdrone_baseline_training_runtime_v1"
TRAINING_CONTRACT_ID = "rtdetrv2_r18_visdrone_baseline_training_v1"
RUNTIME_STAGE = "pre_cuda_plan_only"
TRAINING_CONTRACT_MODULE_RELATIVE_PATH = "src/sparse_rtdetr/baseline/training_contract.py"
TRAINING_CONTRACT_MODULE_SHA256 = "21599fae8312a32ddf833bf2f53557ed5b4d4122f0be167f2071f7f82008deef"

# These values are fixed after the checked-in JSON document is written.  They
# are deliberately checked independently from the semantic registry below.
RUNTIME_PLAN_RAW_SIZE_BYTES = 12957
RUNTIME_PLAN_RAW_SHA256 = "cb6af1abae9351b4a268681587db82ad059745f1d7e198d7d8c829b4cee41aef"
RUNTIME_PLAN_CANONICAL_SIZE_BYTES = 10888
RUNTIME_PLAN_CANONICAL_SHA256 = "3812b04d957e1cd7c3990c8458a540651c95c6bd551c631fc51717f2f7b386ac"

__all__ = (
    "load_training_runtime_plan",
    "validate_training_runtime_plan",
    "canonical_training_runtime_plan_bytes",
    "training_runtime_plan_binding",
)


def _evaluator_binding() -> dict[str, Any]:
    return {
        "evaluator_id": "visdrone_official_primary_evaluator_v1",
        "protocol_id": "visdrone_official_style_v1",
        "implementation_commit": "036cca4d127ddd9e10e3cc7900c3eb759b55f59f",
        "implementation_tree": "fbe931976f6bac7d8e4b3bb319ff1905e99c5444",
        "config_relative_path": "configs/baseline/visdrone_official_evaluator_v1.json",
        "config_raw_size_bytes": 3857,
        "config_raw_sha256": "36cfa69b0ff645c47c871283f917577ca27c760daf424e9544ab363dbf080ff5",
        "config_canonical_size_bytes": 3295,
        "config_canonical_sha256": "355de90bdb6007ed42ed65b3f653a1b8ea57f2184ab4fd48a9561b692b993998",
        "authority_manifest_relative_path": "manifests/visdrone_det_toolkit_005445.json",
        "authority_manifest_raw_size_bytes": 4166,
        "authority_manifest_raw_sha256": "71168baf15d6d945fd5a4a6c5605b5ba533524efeede2a9d4020127576c1b36f",
        "authority_manifest_canonical_size_bytes": 3351,
        "authority_manifest_canonical_sha256": "5bad9faf7622fe4542aa3b46561d6d41fb2e4ee34f35551ecfd821c9577159e3",
        "authority_archive_size_bytes": 40960,
        "authority_archive_sha256": "bf19dd9477210adf106c7cbf2a72370ed4af22dedb577f361f3dc9e77e99baa4",
        "authority_file_count": 11,
        "authority_inventory_sha256": "35a14a021509b82f1238912e5c77ebb3559f9ee6daa3cc6c92db810b5dce5da0",
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


def _build_frozen_plan() -> dict[str, Any]:
    source_bindings = {
        "training_contract": {
            "relative_path": "configs/baseline/rtdetrv2_r18_visdrone_training_v1.json",
            "raw_size_bytes": 10907,
            "raw_sha256": "0f9ddb2e6d8ec6d2b21e42f6c419511f5bd70df287457c2c2b6109eaf8a93297",
            "canonical_size_bytes": 9117,
            "canonical_sha256": "a20c71ef90cb4ccc091a717286a1c49be8a1ebffb178156942b77798d3d7f868",
            "module_relative_path": TRAINING_CONTRACT_MODULE_RELATIVE_PATH,
            "module_sha256": TRAINING_CONTRACT_MODULE_SHA256,
        },
        "primary_evaluator": _evaluator_binding(),
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
        "conversion_r3": {
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
        },
    }
    return {
        "schema_version": 1,
        "runtime_plan_id": RUNTIME_PLAN_ID,
        "training_contract_id": TRAINING_CONTRACT_ID,
        "runtime_stage": RUNTIME_STAGE,
        "source_bindings": source_bindings,
        "invocation_policy": {
            "formal_run_count": 1,
            "device": "cuda:0",
            "visible_devices": "0",
            "world_size": 1,
            "distributed": False,
            "sync_bn": False,
            "resume_allowed": False,
            "retry_allowed": False,
            "overwrite_allowed": False,
            "tuning_allowed": False,
            "test_only_allowed": False,
            "arbitrary_update_allowed": False,
            "network_download_allowed": False,
            "external_checkpoint_allowed": False,
            "pretrained": False,
            "checkpoint": None,
            "seed": 0,
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
        },
        "data_roles": {
            "train_role": "train_core",
            "development_role": "development",
            "confirmatory_role": "sealed_and_forbidden",
            "test_role": "forbidden",
            "train_batch": 16,
            "development_batch": 32,
            "train_workers": 4,
            "development_workers": 4,
            "drop_last_train": True,
            "drop_last_development": False,
        },
        "optimizer": {
            "type": "AdamW",
            "parameter_name_match_mode": "case_sensitive_literal_substring",
            "parameter_groups": [
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
            "parameter_identity_audit_required": True,
            "betas": [0.9, 0.999],
            "gradient_clip_max_norm": 0.1,
        },
        "schedule": {
            "epochs": 120,
            "warmup_type": "LinearWarmup",
            "warmup_optimizer_steps": 2000,
            "scheduler_type": "MultiStepLR",
            "scheduler_step_unit": "epoch",
            "milestones": [1000],
            "gamma": 0.1,
            "expected_decay_events": 0,
            "checkpoint_every_epoch": 1,
            "periodic_checkpoint_every_epochs": 10,
            "development_evaluation_every_epochs": 1,
            "augmentation_stop_epoch": 117,
            "multiscale_enabled": False,
        },
        "amp": {
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
        },
        "evaluation_and_selection": {
            "primary_evaluator_id": "visdrone_official_primary_evaluator_v1",
            "primary_protocol_id": "visdrone_official_style_v1",
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
            "model_selection_weights": "ema",
        },
        "checkpoint_policy": {
            "last": "every epoch",
            "best": "on deterministic development selection improvement",
            "periodic": "every 10 epochs",
            "final": "epoch 120",
            "resume": False,
            "retry": False,
            "overwrite": False,
            "interrupted_status": "PERMANENT_FAIL",
            "required_state": [
                "raw_model",
                "ema",
                "optimizer",
                "scheduler",
                "warmup",
                "grad_scaler",
                "epoch",
                "global_optimizer_step",
                "rng_states",
                "config_identity",
                "source_identity",
                "environment_identity",
            ],
            "atomic_write_required": True,
            "file_fsync_required": True,
            "atomic_rename_required": True,
            "directory_fsync_required": True,
            "readback_required": True,
            "loadability_check_required": True,
            "sha256_required": True,
            "inventory_required": True,
        },
        "evidence_and_readiness": {
            "runtime_integrity_required": True,
            "training_evidence_required": True,
            "checkpoint_integrity_required": True,
            "primary_evaluator_result_binding_required": True,
            "source_identity_required": True,
            "environment_identity_required": True,
            "data_identity_required": True,
            "exactly_once_required": True,
            "required_exit_code": 0,
            "required_epochs_complete": 120,
            "confirmatory_access": False,
            "test_access": False,
            "speed_measurement": False,
            "model_selection_certified": False,
            "training_implementation_ready": False,
            "training_ready": False,
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


_FROZEN_PLAN = _build_frozen_plan()


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


_SCHEMA = _schema_for(_FROZEN_PLAN)
_FROZEN_LEAVES = dict(_walk_leaves(_FROZEN_PLAN))
_NUMERIC_CONSTRAINTS = {
    pointer: {
        "expected_type": type(value),
        "expected": value,
        "finite": type(value) is float,
        "lower": 0,
    }
    for pointer, value in _FROZEN_LEAVES.items()
    if type(value) in {int, float}
}
_SEMANTIC_LEAVES = {
    pointer: value
    for pointer, value in _FROZEN_LEAVES.items()
    if type(value) not in {int, float}
}


def _error(message: str) -> None:
    raise _training_contract.TrainingContractError(message)


def _assert_builtin_json(value: Any, field: str = "runtime plan") -> None:
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
            _error(f"closed runtime schema object type mismatch at {location}")
        actual = set(value)
        expected = set(detail)
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing and extra:
            _error(f"closed runtime schema renamed keys at {location}: missing={missing} extra={extra}")
        if missing:
            _error(f"closed runtime schema missing keys at {location}: {missing}")
        if extra:
            _error(f"closed runtime schema extra keys at {location}: {extra}")
        for key, child_schema in detail.items():
            _validate_schema(value[key], child_schema, _pointer(pointer, key))
        return
    if kind == "list":
        if type(value) is not list:
            _error(f"closed runtime schema list type mismatch at {location}")
        if len(value) != len(detail):
            _error(f"closed runtime schema list length mismatch at {location}")
        for index, child_schema in enumerate(detail):
            _validate_schema(value[index], child_schema, _pointer(pointer, index))
        return
    if type(value) is not detail:
        _error(f"closed runtime schema scalar type mismatch at {location}: expected={detail.__name__}")


def _validate_owner_registry() -> None:
    leaves = set(_FROZEN_LEAVES)
    numeric = set(_NUMERIC_CONSTRAINTS)
    semantic = set(_SEMANTIC_LEAVES)
    if numeric & semantic or numeric | semantic != leaves:
        _error(
            "runtime scalar owner registry drift: "
            f"missing={sorted(leaves - numeric - semantic)} "
            f"extra={sorted(numeric | semantic - leaves)} "
            f"overlap={sorted(numeric & semantic)}"
        )


def _validate_path_and_sha_syntax(plan: dict[str, Any]) -> None:
    for pointer, value in _walk_leaves(plan):
        key = pointer.rsplit("/", 1)[-1]
        if key.endswith("path") or key in {"relative_path", "module_relative_path", "artifact_root"}:
            _training_contract._assert_relative_path(value, f"runtime plan {pointer}")
        if key == "sha256" or key.endswith("_sha256"):
            _training_contract._assert_sha(value, f"runtime plan {pointer}")
        if key.endswith("_commit") or key.endswith("_tree"):
            _training_contract._assert_git_oid(value, f"runtime plan {pointer}")


def _validate_numeric_constraints(plan: dict[str, Any]) -> None:
    actual = {
        pointer: value
        for pointer, value in _walk_leaves(plan)
        if type(value) in {int, float}
    }
    if set(actual) != set(_NUMERIC_CONSTRAINTS):
        _error(
            "runtime numeric registry coverage drift: "
            f"missing={sorted(set(_NUMERIC_CONSTRAINTS) - set(actual))} "
            f"extra={sorted(set(actual) - set(_NUMERIC_CONSTRAINTS))}"
        )
    for pointer, rule in _NUMERIC_CONSTRAINTS.items():
        value = actual[pointer]
        if type(value) is not rule["expected_type"]:
            _error(f"runtime numeric type mismatch at {pointer}")
        if rule["finite"] and not math.isfinite(value):
            _error(f"runtime numeric nonfinite at {pointer}")
        if value < rule["lower"]:
            _error(f"runtime numeric out of range at {pointer}")
        if value != rule["expected"]:
            _error(f"runtime numeric frozen literal mismatch at {pointer}")


def _validate_semantic_leaves(plan: dict[str, Any]) -> None:
    actual = {
        pointer: value
        for pointer, value in _walk_leaves(plan)
        if type(value) not in {int, float}
    }
    if set(actual) != set(_SEMANTIC_LEAVES):
        _error(
            "runtime semantic registry coverage drift: "
            f"missing={sorted(set(_SEMANTIC_LEAVES) - set(actual))} "
            f"extra={sorted(set(actual) - set(_SEMANTIC_LEAVES))}"
        )
    for pointer, expected in _SEMANTIC_LEAVES.items():
        observed = actual[pointer]
        if type(observed) is not type(expected) or observed != expected:
            _error(f"runtime semantic frozen literal mismatch at {pointer}")


def _validate_relations(plan: dict[str, Any]) -> None:
    source = plan["source_bindings"]
    if plan["runtime_plan_id"] != RUNTIME_PLAN_ID or plan["training_contract_id"] != TRAINING_CONTRACT_ID:
        _error("runtime plan identity drift")
    if source["training_contract"]["relative_path"] != _training_contract.TRAINING_CONFIG_RELATIVE_PATH:
        _error("runtime plan training contract path drift")
    if plan["invocation_policy"]["formal_run_count"] != 1:
        _error("runtime formal run count drift")
    if plan["model"]["input_size"] != [640, 640]:
        _error("runtime input size drift")
    data = plan["data_roles"]
    if data["train_role"] == data["development_role"]:
        _error("runtime train/development roles overlap")
    if data["confirmatory_role"] != "sealed_and_forbidden" or data["test_role"] != "forbidden":
        _error("runtime sealed role drift")
    optimizer = plan["optimizer"]
    groups = optimizer["parameter_groups"]
    if tuple(group["name"] for group in groups) != (
        "backbone_non_norm",
        "norm_or_bn",
        "default",
    ):
        _error("runtime optimizer group order drift")
    if optimizer["mutually_exclusive"] is not True or optimizer["cover_all_trainable_parameters"] is not True:
        _error("runtime optimizer coverage policy drift")
    if groups[0]["parameter_scope"] != "backbone" or groups[0]["exclude_substrings"] != ["norm", "bn"]:
        _error("runtime backbone optimizer selector drift")
    if groups[1]["include_substrings"] != ["norm", "bn"] or groups[1]["parameter_scope"] != "remaining_trainable":
        _error("runtime norm optimizer selector drift")
    if groups[2]["include_substrings"] or groups[2]["exclude_substrings"]:
        _error("runtime default optimizer selector drift")
    if plan["schedule"]["epochs"] != plan["evidence_and_readiness"]["required_epochs_complete"]:
        _error("runtime schedule/acceptance epoch drift")
    if plan["schedule"]["augmentation_stop_epoch"] != plan["augmentation"]["stop_epoch"]:
        _error("runtime augmentation stop relation drift")
    if plan["schedule"]["development_evaluation_every_epochs"] != plan["evaluation_and_selection"]["evaluation_frequency_epochs"]:
        _error("runtime evaluation frequency drift")
    if plan["amp"]["enabled"] is not True or plan["ema"]["enabled"] is not True:
        _error("runtime AMP/EMA required-state drift")
    if plan["evaluation_and_selection"]["primary_evaluator_id"] != source["primary_evaluator"]["evaluator_id"]:
        _error("runtime primary evaluator identity drift")
    if plan["evaluation_and_selection"]["primary_protocol_id"] != source["primary_evaluator"]["protocol_id"]:
        _error("runtime primary protocol identity drift")
    if plan["evaluation_and_selection"]["primary_evaluator_independently_certified"] is not True:
        _error("runtime primary evaluator certification gate drift")
    if plan["checkpoint_policy"]["required_state"] != [
        "raw_model",
        "ema",
        "optimizer",
        "scheduler",
        "warmup",
        "grad_scaler",
        "epoch",
        "global_optimizer_step",
        "rng_states",
        "config_identity",
        "source_identity",
        "environment_identity",
    ]:
        _error("runtime checkpoint state order drift")
    if len(plan["vendor_conflicts"]) != 14:
        _error("runtime vendor conflict inventory length drift")


def _validate_local_plan(plan: Any, *, verify_digest: bool = True) -> dict[str, Any]:
    _assert_builtin_json(plan)
    if type(plan) is not dict:
        _error("runtime plan must be an exact dict")
    _validate_schema(plan, _SCHEMA)
    _validate_owner_registry()
    _validate_numeric_constraints(plan)
    _validate_path_and_sha_syntax(plan)
    _validate_semantic_leaves(plan)
    _validate_relations(plan)
    canonical = canonical_training_runtime_plan_bytes(plan)
    if verify_digest:
        if len(canonical) != RUNTIME_PLAN_CANONICAL_SIZE_BYTES or hashlib.sha256(canonical).hexdigest() != RUNTIME_PLAN_CANONICAL_SHA256:
            _error("runtime plan frozen canonical identity drift")
    return copy.deepcopy(plan)


def canonical_training_runtime_plan_bytes(plan: Any) -> bytes:
    """Return compact deterministic UTF-8 JSON bytes without a trailing LF."""

    _assert_builtin_json(plan)
    try:
        return json.dumps(
            plan,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise _training_contract.TrainingContractError("runtime plan is not finite JSON data") from exc


def _expected_training_binding(plan: dict[str, Any]) -> dict[str, Any]:
    source = plan["source_bindings"]
    contract = source["training_contract"]
    evaluator = source["primary_evaluator"]
    return {
        "schema_version": 1,
        "training_contract_id": TRAINING_CONTRACT_ID,
        "baseline_id": "rtdetrv2_r18_visdrone_baseline_v1",
        "relative_path": contract["relative_path"],
        "raw_size_bytes": contract["raw_size_bytes"],
        "raw_sha256": contract["raw_sha256"],
        "canonical_size_bytes": contract["canonical_size_bytes"],
        "canonical_sha256": contract["canonical_sha256"],
        "vendor_runtime_binding": copy.deepcopy(source["vendor_runtime"]),
        "primary_evaluator_runtime_binding": {
            "evaluator": {
                "evaluator_id": evaluator["evaluator_id"],
                "protocol_id": evaluator["protocol_id"],
            },
            "implementation": {
                "commit": evaluator["implementation_commit"],
                "tree": evaluator["implementation_tree"],
            },
            "config": {
                "relative_path": evaluator["config_relative_path"],
                "raw_size_bytes": evaluator["config_raw_size_bytes"],
                "raw_sha256": evaluator["config_raw_sha256"],
                "canonical_size_bytes": evaluator["config_canonical_size_bytes"],
                "canonical_sha256": evaluator["config_canonical_sha256"],
            },
            "authority_manifest": {
                "relative_path": evaluator["authority_manifest_relative_path"],
                "raw_size_bytes": evaluator["authority_manifest_raw_size_bytes"],
                "raw_sha256": evaluator["authority_manifest_raw_sha256"],
                "canonical_size_bytes": evaluator["authority_manifest_canonical_size_bytes"],
                "canonical_sha256": evaluator["authority_manifest_canonical_sha256"],
            },
            "authority_archive_inventory": {
                "archive_size_bytes": evaluator["authority_archive_size_bytes"],
                "archive_sha256": evaluator["authority_archive_sha256"],
                "file_count": evaluator["authority_file_count"],
                "inventory_sha256": evaluator["authority_inventory_sha256"],
            },
            "source_files": copy.deepcopy(evaluator["source_files"]),
            "independent_audit": copy.deepcopy(evaluator["independent_audit"]),
        },
        "conversion_r3_runtime_binding": copy.deepcopy(source["conversion_r3"]),
    }


def _validate_authority_bindings(
    plan: dict[str, Any],
    training_contract: Any,
    training_binding: Any,
) -> None:
    _assert_builtin_json(training_contract, "training_contract")
    _assert_builtin_json(training_binding, "training_binding")
    if type(training_contract) is not dict or type(training_binding) is not dict:
        _error("runtime authority inputs must be exact dicts")
    try:
        contract_canonical = _training_contract.canonical_training_contract_bytes(training_contract)
    except Exception as exc:
        raise _training_contract.TrainingContractError("training contract authority is not canonical") from exc
    if len(contract_canonical) != 9117 or hashlib.sha256(contract_canonical).hexdigest() != plan["source_bindings"]["training_contract"]["canonical_sha256"]:
        _error("runtime training contract canonical binding drift")
    if training_contract.get("training_contract_id") != TRAINING_CONTRACT_ID:
        _error("runtime training contract ID binding drift")
    if training_contract.get("baseline_id") != "rtdetrv2_r18_visdrone_baseline_v1":
        _error("runtime baseline ID binding drift")
    expected_binding = _expected_training_binding(plan)
    if training_binding != expected_binding:
        _error("runtime observed training-contract binding drift")
    source = training_contract.get("source_bindings")
    if type(source) is not dict:
        _error("runtime training contract source binding is missing")
    if source.get("vendor_runtime") != {
        "relative_path": "vendor/rtdetrv2_pytorch",
        "file_count": 124,
        "directory_count_excluding_root": 25,
        "total_size_bytes": 373735,
        "compact_inventory_sha256": "0fc6803665bc4b5720e983345b2cacb0147eceead9f882588880b6f8a0e68051",
    }:
        _error("runtime vendor source binding drift")
    plan_conversion = plan["source_bindings"]["conversion_r3"]
    source_conversion = source.get("conversion_r3")
    conversion_pairs = (
        ("artifact_root", "relative_path"),
        ("completion_sha256", "completion_sha256"),
        ("artifact_inventory_sha256", "artifact_inventory_sha256"),
        ("entry_canonical_inventory_sha256", "entry_canonical_inventory_sha256"),
        ("config_sha256", "config_sha256"),
        ("category_contract_sha256", "category_contract_sha256"),
        ("source_identity_sha256", "source_identity_sha256"),
    )
    if type(source_conversion) is not dict or any(
        source_conversion.get(source_key) != plan_conversion[plan_key]
        for source_key, plan_key in conversion_pairs
    ):
        _error("runtime Conversion R3 source binding drift")
    evaluator = source.get("primary_evaluator_certification")
    plan_evaluator = plan["source_bindings"]["primary_evaluator"]
    if type(evaluator) is not dict:
        _error("runtime primary evaluator source binding is missing")
    if evaluator.get("evaluator") != {
        "evaluator_id": plan_evaluator["evaluator_id"],
        "protocol_id": plan_evaluator["protocol_id"],
    }:
        _error("runtime primary evaluator source identity drift")


def load_training_runtime_plan(
    repo_root: str | os.PathLike[str],
    config_path: str | os.PathLike[str] = RUNTIME_PLAN_CONFIG_RELATIVE_PATH,
) -> dict[str, Any]:
    """Load the checked-in plan through the training-contract fd boundary."""

    relative = _training_contract._assert_relative_path(str(config_path), "runtime config_path")
    if relative != RUNTIME_PLAN_CONFIG_RELATIVE_PATH:
        _error("runtime config path identity drift")
    with _training_contract._VerifiedRepository(repo_root) as repository:
        raw = repository.read_file(relative)
    if len(raw) != RUNTIME_PLAN_RAW_SIZE_BYTES or hashlib.sha256(raw).hexdigest() != RUNTIME_PLAN_RAW_SHA256:
        _error("runtime plan raw identity drift")
    plan = _training_contract._parse_portable_json(raw, label="training runtime plan")
    return _validate_local_plan(plan)


def validate_training_runtime_plan(
    plan: Any,
    training_contract: Any,
    training_binding: Any,
) -> dict[str, Any]:
    """Validate a detached plan against detached certified T1 authorities."""

    validated = _validate_local_plan(plan)
    _validate_authority_bindings(validated, training_contract, training_binding)
    return copy.deepcopy(validated)


def training_runtime_plan_binding(
    repo_root: str | os.PathLike[str],
    config_path: str | os.PathLike[str] = RUNTIME_PLAN_CONFIG_RELATIVE_PATH,
) -> dict[str, Any]:
    """Bind the plan to the certified contract and all observed authorities."""

    # The certified binding is intentionally the first repository operation.
    training_binding = _training_contract.training_contract_binding(repo_root)
    relative = _training_contract._assert_relative_path(str(config_path), "runtime config_path")
    if relative != RUNTIME_PLAN_CONFIG_RELATIVE_PATH:
        _error("runtime config path identity drift")
    with _training_contract._VerifiedRepository(repo_root) as repository:
        training_contract, _ = _training_contract._load_training_contract_from_boundary(
            repository, _training_contract.TRAINING_CONFIG_RELATIVE_PATH
        )
        raw = repository.read_file(relative)
        module_raw = repository.read_file(TRAINING_CONTRACT_MODULE_RELATIVE_PATH)
    if len(raw) != RUNTIME_PLAN_RAW_SIZE_BYTES or hashlib.sha256(raw).hexdigest() != RUNTIME_PLAN_RAW_SHA256:
        _error("runtime plan raw identity drift")
    if hashlib.sha256(module_raw).hexdigest() != TRAINING_CONTRACT_MODULE_SHA256:
        _error("training contract module identity drift")
    plan = _training_contract._parse_portable_json(raw, label="training runtime plan")
    validated = validate_training_runtime_plan(plan, training_contract, training_binding)
    canonical = canonical_training_runtime_plan_bytes(validated)
    return {
        "schema_version": 1,
        "runtime_plan_id": RUNTIME_PLAN_ID,
        "relative_path": RUNTIME_PLAN_CONFIG_RELATIVE_PATH,
        "raw_size_bytes": len(raw),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "canonical_size_bytes": len(canonical),
        "canonical_sha256": hashlib.sha256(canonical).hexdigest(),
        "training_contract_identity": copy.deepcopy(validated["source_bindings"]["training_contract"]),
        "vendor_runtime_binding": copy.deepcopy(training_binding["vendor_runtime_binding"]),
        "conversion_r3_runtime_binding": copy.deepcopy(training_binding["conversion_r3_runtime_binding"]),
        "primary_evaluator_runtime_binding": copy.deepcopy(training_binding["primary_evaluator_runtime_binding"]),
        "plan": copy.deepcopy(validated),
    }
