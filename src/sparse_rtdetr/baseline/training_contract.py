"""Portable, standard-library-only validation for the frozen T1 contract."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath
from typing import Any


TRAINING_CONFIG_RELATIVE_PATH = "configs/baseline/rtdetrv2_r18_visdrone_training_v1.json"
BASELINE_CONFIG_RELATIVE_PATH = "configs/baseline/rtdetrv2_r18_visdrone_baseline_v1.json"
TRAINING_CONFIG_SIZE_BYTES = 8058
TRAINING_CONFIG_RAW_SHA256 = "8ce30e636caad84ee6e3e0ec10d384730a00861ccf179c1e765c095465c6cd5d"
TRAINING_CONTRACT_CANONICAL_SHA256 = "753991ca118267f571efda6a602c1347323707d074d9d7b2e4ed64fb7a17c4f6"
BASELINE_CONFIG_SIZE_BYTES = 4316
BASELINE_CONFIG_RAW_SHA256 = "38702c3483efcd3c3855552d087bbfd0fcfe3628fa9b1582847039f17eb083dd"
BASELINE_CONFIG_CANONICAL_SHA256 = "c392efd44de7738401c1136261c8ca628dea3d3b0fe355b79d0d69d6ed91bfe0"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_INT = ("scalar", int)
_FLOAT = ("scalar", float)
_BOOL = ("scalar", bool)
_STR = ("scalar", str)
_NULL = ("scalar", type(None))
_AMP_PLACEHOLDER = ("literal", "IMPLEMENTATION_MUST_FREEZE_EXPLICITLY")


def _object(**children: Any) -> tuple[str, dict[str, Any]]:
    return ("object", children)


def _list(*children: Any) -> tuple[str, tuple[Any, ...]]:
    return ("list", children)


_VENDOR_INCLUDE_ROW = _object(role=_STR, relative_path=_STR, sha256=_STR)
_TRAINING_CONTRACT_SCHEMA = _object(
    schema_version=_INT,
    training_contract_id=_STR,
    baseline_id=_STR,
    owner_decision=_STR,
    source_bindings=_object(
        baseline_config=_object(relative_path=_STR, size_bytes=_INT, raw_sha256=_STR, canonical_sha256=_STR),
        vendor_upstream_commit=_STR,
        vendor_recipe=_object(relative_path=_STR, sha256=_STR),
        vendor_includes=_list(*(_VENDOR_INCLUDE_ROW for _ in range(4))),
        conversion_r3=_object(
            artifact_root=_STR,
            completion_sha256=_STR,
            artifact_inventory_sha256=_STR,
            entry_canonical_inventory_sha256=_STR,
            config_sha256=_STR,
            category_contract_sha256=_STR,
            source_identity_sha256=_STR,
        ),
    ),
    model=_object(
        type=_STR, backbone=_STR, encoder=_STR, decoder=_STR, num_classes=_INT,
        parameter_count=_INT, input_size=_list(_INT, _INT), num_queries=_INT,
        decoder_layers=_INT, nms=_BOOL,
    ),
    data_roles=_object(train=_STR, development=_STR, confirmatory=_STR, test=_STR),
    initialization=_object(
        pretrained=_BOOL, checkpoint=_NULL, initialization=_STR, seed=_INT,
        formal_run_count=_INT, seed_selection_forbidden=_BOOL,
        network_download_forbidden=_BOOL, external_checkpoint_forbidden=_BOOL,
    ),
    topology=_object(
        device_type=_STR, visible_devices=_STR, world_size=_INT, distributed=_BOOL,
        sync_bn=_BOOL, train_micro_batch=_INT, gradient_accumulation_steps=_INT,
        effective_train_batch=_INT, train_workers=_INT, development_batch=_INT,
        development_workers=_INT, drop_last_train=_BOOL,
        drop_last_development=_BOOL, runtime_batch_adaptation_forbidden=_BOOL,
    ),
    schedule=_object(
        epochs=_INT, steps_per_epoch_source=_STR, checkpoint_frequency_epochs=_INT,
        development_evaluation_frequency_epochs=_INT, augmentation_stop_epoch=_INT,
        multiscale_enabled=_BOOL,
    ),
    optimizer=_object(
        type=_STR, default_lr=_FLOAT, backbone_non_norm_lr=_FLOAT,
        betas=_list(_FLOAT, _FLOAT), default_weight_decay=_FLOAT,
        norm_bn_weight_decay=_FLOAT, clip_max_norm=_FLOAT,
        parameter_groups_must_be_mutually_exclusive=_BOOL,
        parameter_groups_must_cover_all_trainable_parameters=_BOOL,
        parameter_identity_audit_required=_BOOL,
    ),
    learning_rate=_object(
        warmup_type=_STR, warmup_optimizer_steps=_INT, scheduler_type=_STR,
        scheduler_step_unit=_STR, milestones=_list(_INT), gamma=_FLOAT,
        expected_decay_events_within_120_epochs=_INT,
    ),
    amp=_object(
        enabled=_BOOL, scaler_type=_STR, init_scale=_AMP_PLACEHOLDER,
        growth_factor=_AMP_PLACEHOLDER, backoff_factor=_AMP_PLACEHOLDER,
        growth_interval=_AMP_PLACEHOLDER, nonfinite_loss_allowed=_INT,
        nonfinite_gradient_allowed=_INT, optimizer_skipped_steps_allowed=_INT,
        overflow_events_allowed=_INT,
    ),
    ema=_object(
        enabled=_BOOL, decay=_FLOAT, warmups=_INT, warmup_unit=_STR,
        development_evaluation_weights=_STR, model_selection_weights=_STR,
        raw_weights_must_be_preserved=_BOOL, ema_weights_must_be_preserved=_BOOL,
    ),
    augmentation=_object(
        transforms=_list(
            _object(type=_STR, p=_FLOAT),
            _object(type=_STR, fill=_INT),
            _object(type=_STR, p=_FLOAT),
            _object(type=_STR, min_size=_INT),
            _object(type=_STR),
            _object(type=_STR, size=_list(_INT, _INT)),
            _object(type=_STR, min_size=_INT),
            _object(type=_STR, dtype=_STR, scale=_BOOL),
            _object(type=_STR, fmt=_STR, normalize=_BOOL),
        ),
        stop_epoch=_INT,
        stopped_transforms=_list(_STR, _STR, _STR),
        multiscale_enabled=_BOOL,
        runtime_transform_identity_and_rng_binding_required=_BOOL,
    ),
    evaluation_and_selection=_object(
        primary_evaluator=_STR, primary_evaluator_required_before_training=_BOOL,
        primary_evaluator_independently_certified=_BOOL, training_launch_blocked=_BOOL,
        secondary_evaluator=_STR, secondary_diagnostic_only=_BOOL,
        secondary_cannot_certify_or_select=_BOOL, development_only=_BOOL,
        evaluation_frequency_epochs=_INT, selection_metric=_STR,
        selection_direction=_STR, tie_breakers=_list(_STR, _STR, _STR),
        metric_comparison_precision=_STR, accuracy_threshold=_NULL,
    ),
    checkpoint_policy=_object(
        resume=_BOOL, retry=_BOOL, overwrite=_BOOL, exactly_once_run_identity=_BOOL,
        interrupted_run_status=_STR, last_checkpoint_every_epoch=_BOOL,
        best_checkpoint_on_selection_improvement=_BOOL,
        periodic_checkpoint_frequency_epochs=_INT, final_epoch_checkpoint=_BOOL,
        atomic_write_required=_BOOL, file_fsync_required=_BOOL,
        directory_fsync_required=_BOOL, readback_required=_BOOL,
        loadability_check_required=_BOOL, sha256_required=_BOOL, inventory_required=_BOOL,
        required_state=_list(*(_STR for _ in range(12))),
    ),
    acceptance=_object(
        separate_states=_list(*(_STR for _ in range(8))),
        required_epochs_complete=_INT, expected_optimizer_steps_complete=_BOOL,
        loss_and_gradients_finite=_BOOL, amp_skip_or_overflow_allowed=_BOOL,
        required_exit_code=_INT, completion_evidence_and_checkpoints_bound=_BOOL,
        checkpoints_loadable=_BOOL, development_accuracy_reporting_only=_BOOL,
        confirmatory_metrics_accessed=_BOOL, test_accessed=_BOOL, speed_measured=_BOOL,
    ),
)


def _numeric(
    expected_type: type,
    exact: int | float,
    role: str,
    *,
    lower: int | float | None = None,
    lower_inclusive: bool = True,
    upper: int | float | None = None,
    upper_inclusive: bool = True,
) -> dict[str, Any]:
    return {
        "expected_type": expected_type,
        "finite": True,
        "lower": lower,
        "lower_inclusive": lower_inclusive,
        "upper": upper,
        "upper_inclusive": upper_inclusive,
        "exact": exact,
        "role": role,
    }


_TRAINING_CONTRACT_NUMERIC_CONSTRAINTS = {
    "/schema_version": _numeric(int, 1, "schema_version", lower=1),
    "/source_bindings/baseline_config/size_bytes": _numeric(int, 4316, "size_bytes", lower=1),
    "/model/num_classes": _numeric(int, 10, "count", lower=1),
    "/model/parameter_count": _numeric(int, 20094584, "count", lower=1),
    "/model/input_size/0": _numeric(int, 640, "size", lower=1),
    "/model/input_size/1": _numeric(int, 640, "size", lower=1),
    "/model/num_queries": _numeric(int, 300, "count", lower=1),
    "/model/decoder_layers": _numeric(int, 3, "count", lower=1),
    "/initialization/seed": _numeric(int, 0, "seed", lower=0),
    "/initialization/formal_run_count": _numeric(int, 1, "count", lower=1),
    "/topology/world_size": _numeric(int, 1, "count", lower=1),
    "/topology/train_micro_batch": _numeric(int, 16, "batch_size", lower=1),
    "/topology/gradient_accumulation_steps": _numeric(int, 1, "step_count", lower=1),
    "/topology/effective_train_batch": _numeric(int, 16, "batch_size", lower=1),
    "/topology/train_workers": _numeric(int, 4, "worker_count", lower=0),
    "/topology/development_batch": _numeric(int, 32, "batch_size", lower=1),
    "/topology/development_workers": _numeric(int, 4, "worker_count", lower=0),
    "/schedule/epochs": _numeric(int, 120, "epoch_count", lower=1),
    "/schedule/checkpoint_frequency_epochs": _numeric(int, 1, "epoch_interval", lower=1),
    "/schedule/development_evaluation_frequency_epochs": _numeric(int, 1, "epoch_interval", lower=1),
    "/schedule/augmentation_stop_epoch": _numeric(int, 117, "epoch_index", lower=0),
    "/optimizer/default_lr": _numeric(float, 0.0001, "learning_rate", lower=0.0, lower_inclusive=False),
    "/optimizer/backbone_non_norm_lr": _numeric(float, 0.00001, "learning_rate", lower=0.0, lower_inclusive=False),
    "/optimizer/betas/0": _numeric(float, 0.9, "adam_beta", lower=0.0, upper=1.0, upper_inclusive=False),
    "/optimizer/betas/1": _numeric(float, 0.999, "adam_beta", lower=0.0, upper=1.0, upper_inclusive=False),
    "/optimizer/default_weight_decay": _numeric(float, 0.0001, "weight_decay", lower=0.0),
    "/optimizer/norm_bn_weight_decay": _numeric(float, 0.0, "weight_decay", lower=0.0),
    "/optimizer/clip_max_norm": _numeric(float, 0.1, "gradient_norm", lower=0.0, lower_inclusive=False),
    "/learning_rate/warmup_optimizer_steps": _numeric(int, 2000, "step_count", lower=1),
    "/learning_rate/milestones/0": _numeric(int, 1000, "epoch_milestone", lower=1),
    "/learning_rate/gamma": _numeric(float, 0.1, "scheduler_factor", lower=0.0, lower_inclusive=False, upper=1.0),
    "/learning_rate/expected_decay_events_within_120_epochs": _numeric(int, 0, "count", lower=0),
    "/amp/nonfinite_loss_allowed": _numeric(int, 0, "allowed_event_count", lower=0),
    "/amp/nonfinite_gradient_allowed": _numeric(int, 0, "allowed_event_count", lower=0),
    "/amp/optimizer_skipped_steps_allowed": _numeric(int, 0, "allowed_event_count", lower=0),
    "/amp/overflow_events_allowed": _numeric(int, 0, "allowed_event_count", lower=0),
    "/ema/decay": _numeric(float, 0.9999, "ema_decay", lower=0.0, lower_inclusive=False, upper=1.0, upper_inclusive=False),
    "/ema/warmups": _numeric(int, 2000, "update_count", lower=1),
    "/augmentation/transforms/0/p": _numeric(float, 0.5, "probability", lower=0.0, upper=1.0),
    "/augmentation/transforms/1/fill": _numeric(int, 0, "pixel_fill", lower=0, upper=255),
    "/augmentation/transforms/2/p": _numeric(float, 0.8, "probability", lower=0.0, upper=1.0),
    "/augmentation/transforms/3/min_size": _numeric(int, 1, "size", lower=1),
    "/augmentation/transforms/5/size/0": _numeric(int, 640, "size", lower=1),
    "/augmentation/transforms/5/size/1": _numeric(int, 640, "size", lower=1),
    "/augmentation/transforms/6/min_size": _numeric(int, 1, "size", lower=1),
    "/augmentation/stop_epoch": _numeric(int, 117, "epoch_index", lower=0),
    "/evaluation_and_selection/evaluation_frequency_epochs": _numeric(int, 1, "epoch_interval", lower=1),
    "/checkpoint_policy/periodic_checkpoint_frequency_epochs": _numeric(int, 10, "epoch_interval", lower=1),
    "/acceptance/required_epochs_complete": _numeric(int, 120, "epoch_count", lower=1),
    "/acceptance/required_exit_code": _numeric(int, 0, "process_exit_code", lower=0),
}


# These fields are validated by the path/SHA and external-binding layers.  Their
# container siblings (for example vendor include roles) remain semantic data.
_SEMANTIC_SYNTAX_OR_BINDING_POINTERS = frozenset(
    {
        "/source_bindings/baseline_config/relative_path",
        "/source_bindings/baseline_config/raw_sha256",
        "/source_bindings/baseline_config/canonical_sha256",
        "/source_bindings/vendor_upstream_commit",
        "/source_bindings/vendor_recipe/relative_path",
        "/source_bindings/vendor_recipe/sha256",
        "/source_bindings/conversion_r3/artifact_root",
        "/source_bindings/conversion_r3/completion_sha256",
        "/source_bindings/conversion_r3/artifact_inventory_sha256",
        "/source_bindings/conversion_r3/entry_canonical_inventory_sha256",
        "/source_bindings/conversion_r3/config_sha256",
        "/source_bindings/conversion_r3/category_contract_sha256",
        "/source_bindings/conversion_r3/source_identity_sha256",
        *(f"/source_bindings/vendor_includes/{index}/{field}" for index in range(4) for field in ("relative_path", "sha256")),
    }
)


def _semantic_leaf_rule(
    rule_id: str,
    pointer: str,
    expected: Any,
    role: str,
    *,
    kind: str = "literal",
    allowed: tuple[Any, ...] | None = None,
) -> dict[str, Any]:
    return {
        "rule_id": rule_id,
        "kind": kind,
        "pointer": pointer,
        "expected": expected,
        "expected_type": type(expected),
        "allowed": allowed,
        "role": role,
    }


def _semantic_sequence_rule(
    rule_id: str,
    pointer: str,
    expected: tuple[Any, ...],
    role: str,
    *,
    member_field: str | None = None,
) -> dict[str, Any]:
    return {
        "rule_id": rule_id,
        "kind": "ordered_list",
        "pointer": pointer,
        "expected": expected,
        "member_field": member_field,
        "role": role,
    }


def _semantic_relation_rule(rule_id: str, frozen: dict[str, Any], role: str) -> dict[str, Any]:
    return {
        "rule_id": rule_id,
        "kind": "required_relation",
        "pointers": tuple(frozen),
        "frozen": frozen,
        "role": role,
    }


_TRAINING_CONTRACT_SEMANTIC_RULES = (
    # Every in-scope non-numeric leaf has one entry.  The expected value is
    # intentionally duplicated here as a reviewable semantic decision rather
    # than being recovered from the frozen digest or from the filesystem.
    _semantic_leaf_rule("SEM_TRAINING_CONTRACT_ID", "/training_contract_id", "rtdetrv2_r18_visdrone_baseline_training_v1", "contract identity"),
    _semantic_leaf_rule("SEM_BASELINE_ID", "/baseline_id", "rtdetrv2_r18_visdrone_baseline_v1", "baseline identity"),
    _semantic_leaf_rule("SEM_OWNER_DECISION", "/owner_decision", "T1_RANDOM_INITIALIZATION", "owner decision"),
    _semantic_leaf_rule("SEM_VENDOR_ROLE_0", "/source_bindings/vendor_includes/0/role", "dataloader", "vendor source role", kind="enum", allowed=("dataloader", "optimizer", "model", "runtime")),
    _semantic_leaf_rule("SEM_VENDOR_ROLE_1", "/source_bindings/vendor_includes/1/role", "optimizer", "vendor source role", kind="enum", allowed=("dataloader", "optimizer", "model", "runtime")),
    _semantic_leaf_rule("SEM_VENDOR_ROLE_2", "/source_bindings/vendor_includes/2/role", "model", "vendor source role", kind="enum", allowed=("dataloader", "optimizer", "model", "runtime")),
    _semantic_leaf_rule("SEM_VENDOR_ROLE_3", "/source_bindings/vendor_includes/3/role", "runtime", "vendor source role", kind="enum", allowed=("dataloader", "optimizer", "model", "runtime")),
    _semantic_leaf_rule("SEM_MODEL_TYPE", "/model/type", "RTDETR", "model type", kind="enum", allowed=("RTDETR", "RTDETRv2")),
    _semantic_leaf_rule("SEM_MODEL_BACKBONE", "/model/backbone", "PResNet-18", "model backbone", kind="enum", allowed=("PResNet-18", "PResNet-50")),
    _semantic_leaf_rule("SEM_MODEL_ENCODER", "/model/encoder", "HybridEncoder", "model encoder", kind="enum", allowed=("HybridEncoder", "TransformerEncoder")),
    _semantic_leaf_rule("SEM_MODEL_DECODER", "/model/decoder", "RTDETRTransformerv2", "model decoder", kind="enum", allowed=("RTDETRTransformerv2", "RTDETRTransformer")),
    _semantic_leaf_rule("SEM_MODEL_NMS", "/model/nms", False, "model postprocessing mode"),
    _semantic_leaf_rule("SEM_DATA_TRAIN", "/data_roles/train", "train_core", "training data role", kind="enum", allowed=("train_core", "development")),
    _semantic_leaf_rule("SEM_DATA_DEVELOPMENT", "/data_roles/development", "development", "development data role", kind="enum", allowed=("development", "train_core")),
    _semantic_leaf_rule("SEM_DATA_CONFIRMATORY", "/data_roles/confirmatory", "sealed_and_forbidden", "confirmatory data role", kind="enum", allowed=("sealed_and_forbidden", "allowed")),
    _semantic_leaf_rule("SEM_DATA_TEST", "/data_roles/test", "forbidden", "test data role", kind="enum", allowed=("forbidden", "allowed")),
    _semantic_leaf_rule("SEM_INIT_PRETRAINED", "/initialization/pretrained", False, "initialization source"),
    _semantic_leaf_rule("SEM_INIT_CHECKPOINT", "/initialization/checkpoint", None, "initialization source"),
    _semantic_leaf_rule("SEM_INIT_MODE", "/initialization/initialization", "random", "initialization mode", kind="enum", allowed=("random", "pretrained", "checkpoint")),
    _semantic_leaf_rule("SEM_INIT_SEED_SELECTION", "/initialization/seed_selection_forbidden", True, "seed policy"),
    _semantic_leaf_rule("SEM_INIT_NETWORK_DOWNLOAD", "/initialization/network_download_forbidden", True, "network policy"),
    _semantic_leaf_rule("SEM_INIT_EXTERNAL_CHECKPOINT", "/initialization/external_checkpoint_forbidden", True, "checkpoint source policy"),
    _semantic_leaf_rule("SEM_TOPOLOGY_DEVICE", "/topology/device_type", "cuda", "device mode", kind="enum", allowed=("cuda", "cpu")),
    _semantic_leaf_rule("SEM_TOPOLOGY_VISIBLE_DEVICES", "/topology/visible_devices", "0", "single-device selection"),
    _semantic_leaf_rule("SEM_TOPOLOGY_DISTRIBUTED", "/topology/distributed", False, "distributed mode"),
    _semantic_leaf_rule("SEM_TOPOLOGY_SYNC_BN", "/topology/sync_bn", False, "synchronized batch normalization mode"),
    _semantic_leaf_rule("SEM_TOPOLOGY_DROP_LAST_TRAIN", "/topology/drop_last_train", True, "train batch policy"),
    _semantic_leaf_rule("SEM_TOPOLOGY_DROP_LAST_DEVELOPMENT", "/topology/drop_last_development", False, "development batch policy"),
    _semantic_leaf_rule("SEM_TOPOLOGY_RUNTIME_ADAPTATION", "/topology/runtime_batch_adaptation_forbidden", True, "runtime batch policy"),
    _semantic_leaf_rule("SEM_SCHEDULE_STEPS_SOURCE", "/schedule/steps_per_epoch_source", "train_core_dataset_length_and_drop_last", "epoch step source"),
    _semantic_leaf_rule("SEM_SCHEDULE_MULTISCALE", "/schedule/multiscale_enabled", False, "schedule multiscale mode"),
    _semantic_leaf_rule("SEM_OPTIMIZER_TYPE", "/optimizer/type", "AdamW", "optimizer type", kind="enum", allowed=("AdamW", "SGD")),
    _semantic_leaf_rule("SEM_OPTIMIZER_GROUP_EXCLUSIVE", "/optimizer/parameter_groups_must_be_mutually_exclusive", True, "optimizer parameter-group role"),
    _semantic_leaf_rule("SEM_OPTIMIZER_GROUP_COVERAGE", "/optimizer/parameter_groups_must_cover_all_trainable_parameters", True, "optimizer parameter-group role"),
    _semantic_leaf_rule("SEM_OPTIMIZER_IDENTITY_AUDIT", "/optimizer/parameter_identity_audit_required", True, "optimizer parameter-group role"),
    _semantic_leaf_rule("SEM_WARMUP_TYPE", "/learning_rate/warmup_type", "LinearWarmup", "warmup policy", kind="enum", allowed=("LinearWarmup", "ConstantWarmup")),
    _semantic_leaf_rule("SEM_SCHEDULER_TYPE", "/learning_rate/scheduler_type", "MultiStepLR", "scheduler policy", kind="enum", allowed=("MultiStepLR", "StepLR")),
    _semantic_leaf_rule("SEM_SCHEDULER_UNIT", "/learning_rate/scheduler_step_unit", "epoch", "scheduler step unit", kind="enum", allowed=("epoch", "iteration")),
    _semantic_leaf_rule("SEM_AMP_ENABLED", "/amp/enabled", True, "AMP requirement"),
    _semantic_leaf_rule("SEM_AMP_SCALER", "/amp/scaler_type", "GradScaler", "AMP scaler", kind="enum", allowed=("GradScaler", "NoScaler")),
    _semantic_leaf_rule("SEM_AMP_INIT_SCALE", "/amp/init_scale", "IMPLEMENTATION_MUST_FREEZE_EXPLICITLY", "AMP unresolved placeholder"),
    _semantic_leaf_rule("SEM_AMP_GROWTH_FACTOR", "/amp/growth_factor", "IMPLEMENTATION_MUST_FREEZE_EXPLICITLY", "AMP unresolved placeholder"),
    _semantic_leaf_rule("SEM_AMP_BACKOFF_FACTOR", "/amp/backoff_factor", "IMPLEMENTATION_MUST_FREEZE_EXPLICITLY", "AMP unresolved placeholder"),
    _semantic_leaf_rule("SEM_AMP_GROWTH_INTERVAL", "/amp/growth_interval", "IMPLEMENTATION_MUST_FREEZE_EXPLICITLY", "AMP unresolved placeholder"),
    _semantic_leaf_rule("SEM_EMA_ENABLED", "/ema/enabled", True, "EMA requirement"),
    _semantic_leaf_rule("SEM_EMA_WARMUP_UNIT", "/ema/warmup_unit", "optimizer_updates", "EMA warmup unit", kind="enum", allowed=("optimizer_updates", "epochs")),
    _semantic_leaf_rule("SEM_EMA_DEVELOPMENT_WEIGHTS", "/ema/development_evaluation_weights", "ema", "development evaluation weights", kind="enum", allowed=("ema", "raw")),
    _semantic_leaf_rule("SEM_EMA_SELECTION_WEIGHTS", "/ema/model_selection_weights", "ema", "model selection weights", kind="enum", allowed=("ema", "raw")),
    _semantic_leaf_rule("SEM_EMA_RAW_PRESERVED", "/ema/raw_weights_must_be_preserved", True, "EMA state preservation"),
    _semantic_leaf_rule("SEM_EMA_WEIGHTS_PRESERVED", "/ema/ema_weights_must_be_preserved", True, "EMA state preservation"),
    _semantic_leaf_rule("SEM_AUGMENTATION_TYPE_0", "/augmentation/transforms/0/type", "RandomPhotometricDistort", "R18 augmentation identity", kind="enum", allowed=("RandomPhotometricDistort", "RandomZoomOut", "RandomIoUCrop")),
    _semantic_leaf_rule("SEM_AUGMENTATION_TYPE_1", "/augmentation/transforms/1/type", "RandomZoomOut", "R18 augmentation identity", kind="enum", allowed=("RandomPhotometricDistort", "RandomZoomOut", "RandomIoUCrop")),
    _semantic_leaf_rule("SEM_AUGMENTATION_TYPE_2", "/augmentation/transforms/2/type", "RandomIoUCrop", "R18 augmentation identity", kind="enum", allowed=("RandomPhotometricDistort", "RandomZoomOut", "RandomIoUCrop")),
    _semantic_leaf_rule("SEM_AUGMENTATION_TYPE_3", "/augmentation/transforms/3/type", "SanitizeBoundingBoxes", "R18 augmentation identity", kind="enum", allowed=("SanitizeBoundingBoxes", "RandomHorizontalFlip")),
    _semantic_leaf_rule("SEM_AUGMENTATION_TYPE_4", "/augmentation/transforms/4/type", "RandomHorizontalFlip", "R18 augmentation identity", kind="enum", allowed=("RandomHorizontalFlip", "SanitizeBoundingBoxes")),
    _semantic_leaf_rule("SEM_AUGMENTATION_TYPE_5", "/augmentation/transforms/5/type", "Resize", "R18 augmentation identity", kind="enum", allowed=("Resize", "RandomResize")),
    _semantic_leaf_rule("SEM_AUGMENTATION_TYPE_6", "/augmentation/transforms/6/type", "SanitizeBoundingBoxes", "R18 augmentation identity", kind="enum", allowed=("SanitizeBoundingBoxes", "RandomHorizontalFlip")),
    _semantic_leaf_rule("SEM_AUGMENTATION_TYPE_7", "/augmentation/transforms/7/type", "ConvertPILImage", "R18 augmentation identity", kind="enum", allowed=("ConvertPILImage", "ConvertBoxes")),
    _semantic_leaf_rule("SEM_AUGMENTATION_DTYPE", "/augmentation/transforms/7/dtype", "float32", "R18 image conversion", kind="enum", allowed=("float32", "float16")),
    _semantic_leaf_rule("SEM_AUGMENTATION_SCALE", "/augmentation/transforms/7/scale", True, "R18 image conversion"),
    _semantic_leaf_rule("SEM_AUGMENTATION_TYPE_8", "/augmentation/transforms/8/type", "ConvertBoxes", "R18 augmentation identity", kind="enum", allowed=("ConvertBoxes", "ConvertPILImage")),
    _semantic_leaf_rule("SEM_AUGMENTATION_FORMAT", "/augmentation/transforms/8/fmt", "cxcywh", "R18 box conversion", kind="enum", allowed=("cxcywh", "xyxy")),
    _semantic_leaf_rule("SEM_AUGMENTATION_NORMALIZE", "/augmentation/transforms/8/normalize", True, "R18 box conversion"),
    _semantic_leaf_rule("SEM_STOPPED_TRANSFORM_0", "/augmentation/stopped_transforms/0", "RandomPhotometricDistort", "train-only augmentation stop"),
    _semantic_leaf_rule("SEM_STOPPED_TRANSFORM_1", "/augmentation/stopped_transforms/1", "RandomZoomOut", "train-only augmentation stop"),
    _semantic_leaf_rule("SEM_STOPPED_TRANSFORM_2", "/augmentation/stopped_transforms/2", "RandomIoUCrop", "train-only augmentation stop"),
    _semantic_leaf_rule("SEM_AUGMENTATION_MULTISCALE", "/augmentation/multiscale_enabled", False, "augmentation multiscale mode"),
    _semantic_leaf_rule("SEM_AUGMENTATION_RUNTIME_BINDING", "/augmentation/runtime_transform_identity_and_rng_binding_required", True, "augmentation runtime binding"),
    _semantic_leaf_rule("SEM_PRIMARY_EVALUATOR", "/evaluation_and_selection/primary_evaluator", "visdrone_official_style_v1", "primary evaluator", kind="enum", allowed=("visdrone_official_style_v1", "coco_secondary_vendor_v1")),
    _semantic_leaf_rule("SEM_PRIMARY_REQUIRED", "/evaluation_and_selection/primary_evaluator_required_before_training", True, "primary evaluator gate"),
    _semantic_leaf_rule("SEM_PRIMARY_CERTIFIED", "/evaluation_and_selection/primary_evaluator_independently_certified", False, "primary evaluator certification"),
    _semantic_leaf_rule("SEM_TRAINING_LAUNCH_BLOCKED", "/evaluation_and_selection/training_launch_blocked", True, "training launch gate"),
    _semantic_leaf_rule("SEM_SECONDARY_EVALUATOR", "/evaluation_and_selection/secondary_evaluator", "coco_secondary_vendor_v1", "secondary evaluator", kind="enum", allowed=("coco_secondary_vendor_v1", "visdrone_official_style_v1")),
    _semantic_leaf_rule("SEM_SECONDARY_DIAGNOSTIC", "/evaluation_and_selection/secondary_diagnostic_only", True, "secondary evaluator role"),
    _semantic_leaf_rule("SEM_SECONDARY_NO_CERTIFY", "/evaluation_and_selection/secondary_cannot_certify_or_select", True, "secondary evaluator role"),
    _semantic_leaf_rule("SEM_DEVELOPMENT_ONLY", "/evaluation_and_selection/development_only", True, "development-only selection"),
    _semantic_leaf_rule("SEM_SELECTION_METRIC", "/evaluation_and_selection/selection_metric", "AP@[0.50:0.95,maxDets=500]", "selection metric"),
    _semantic_leaf_rule("SEM_SELECTION_DIRECTION", "/evaluation_and_selection/selection_direction", "maximize", "selection direction", kind="enum", allowed=("maximize", "minimize")),
    _semantic_leaf_rule("SEM_TIE_BREAKER_0", "/evaluation_and_selection/tie_breakers/0", "AP50", "selection tie-breaker"),
    _semantic_leaf_rule("SEM_TIE_BREAKER_1", "/evaluation_and_selection/tie_breakers/1", "AR500", "selection tie-breaker"),
    _semantic_leaf_rule("SEM_TIE_BREAKER_2", "/evaluation_and_selection/tie_breakers/2", "earlier_epoch", "selection tie-breaker"),
    _semantic_leaf_rule("SEM_METRIC_PRECISION", "/evaluation_and_selection/metric_comparison_precision", "unrounded_float64", "metric precision", kind="enum", allowed=("unrounded_float64", "rounded")),
    _semantic_leaf_rule("SEM_ACCURACY_THRESHOLD", "/evaluation_and_selection/accuracy_threshold", None, "accuracy threshold policy"),
    _semantic_leaf_rule("SEM_CHECKPOINT_RESUME", "/checkpoint_policy/resume", False, "exactly-once checkpoint policy"),
    _semantic_leaf_rule("SEM_CHECKPOINT_RETRY", "/checkpoint_policy/retry", False, "exactly-once checkpoint policy"),
    _semantic_leaf_rule("SEM_CHECKPOINT_OVERWRITE", "/checkpoint_policy/overwrite", False, "exactly-once checkpoint policy"),
    _semantic_leaf_rule("SEM_CHECKPOINT_EXACTLY_ONCE", "/checkpoint_policy/exactly_once_run_identity", True, "exactly-once checkpoint policy"),
    _semantic_leaf_rule("SEM_CHECKPOINT_INTERRUPTED", "/checkpoint_policy/interrupted_run_status", "PERMANENT_FAIL", "interrupted checkpoint status", kind="enum", allowed=("PERMANENT_FAIL", "RETRYABLE")),
    _semantic_leaf_rule("SEM_CHECKPOINT_LAST", "/checkpoint_policy/last_checkpoint_every_epoch", True, "checkpoint ownership policy"),
    _semantic_leaf_rule("SEM_CHECKPOINT_BEST", "/checkpoint_policy/best_checkpoint_on_selection_improvement", True, "checkpoint ownership policy"),
    _semantic_leaf_rule("SEM_CHECKPOINT_FINAL", "/checkpoint_policy/final_epoch_checkpoint", True, "checkpoint ownership policy"),
    _semantic_leaf_rule("SEM_CHECKPOINT_ATOMIC", "/checkpoint_policy/atomic_write_required", True, "checkpoint integrity policy"),
    _semantic_leaf_rule("SEM_CHECKPOINT_FILE_FSYNC", "/checkpoint_policy/file_fsync_required", True, "checkpoint integrity policy"),
    _semantic_leaf_rule("SEM_CHECKPOINT_DIRECTORY_FSYNC", "/checkpoint_policy/directory_fsync_required", True, "checkpoint integrity policy"),
    _semantic_leaf_rule("SEM_CHECKPOINT_READBACK", "/checkpoint_policy/readback_required", True, "checkpoint integrity policy"),
    _semantic_leaf_rule("SEM_CHECKPOINT_LOADABILITY", "/checkpoint_policy/loadability_check_required", True, "checkpoint integrity policy"),
    _semantic_leaf_rule("SEM_CHECKPOINT_SHA", "/checkpoint_policy/sha256_required", True, "checkpoint integrity policy"),
    _semantic_leaf_rule("SEM_CHECKPOINT_INVENTORY", "/checkpoint_policy/inventory_required", True, "checkpoint integrity policy"),
    _semantic_leaf_rule("SEM_CHECKPOINT_STATE_0", "/checkpoint_policy/required_state/0", "raw_model", "checkpoint state ownership"),
    _semantic_leaf_rule("SEM_CHECKPOINT_STATE_1", "/checkpoint_policy/required_state/1", "ema", "checkpoint state ownership"),
    _semantic_leaf_rule("SEM_CHECKPOINT_STATE_2", "/checkpoint_policy/required_state/2", "optimizer", "checkpoint state ownership"),
    _semantic_leaf_rule("SEM_CHECKPOINT_STATE_3", "/checkpoint_policy/required_state/3", "scheduler", "checkpoint state ownership"),
    _semantic_leaf_rule("SEM_CHECKPOINT_STATE_4", "/checkpoint_policy/required_state/4", "warmup", "checkpoint state ownership"),
    _semantic_leaf_rule("SEM_CHECKPOINT_STATE_5", "/checkpoint_policy/required_state/5", "grad_scaler", "checkpoint state ownership"),
    _semantic_leaf_rule("SEM_CHECKPOINT_STATE_6", "/checkpoint_policy/required_state/6", "epoch", "checkpoint state ownership"),
    _semantic_leaf_rule("SEM_CHECKPOINT_STATE_7", "/checkpoint_policy/required_state/7", "global_optimizer_step", "checkpoint state ownership"),
    _semantic_leaf_rule("SEM_CHECKPOINT_STATE_8", "/checkpoint_policy/required_state/8", "rng_states", "checkpoint state ownership"),
    _semantic_leaf_rule("SEM_CHECKPOINT_STATE_9", "/checkpoint_policy/required_state/9", "config_identity", "checkpoint state ownership"),
    _semantic_leaf_rule("SEM_CHECKPOINT_STATE_10", "/checkpoint_policy/required_state/10", "source_identity", "checkpoint state ownership"),
    _semantic_leaf_rule("SEM_CHECKPOINT_STATE_11", "/checkpoint_policy/required_state/11", "environment_identity", "checkpoint state ownership"),
    _semantic_leaf_rule("SEM_ACCEPTANCE_STATE_0", "/acceptance/separate_states/0", "training_runtime_integrity", "acceptance state"),
    _semantic_leaf_rule("SEM_ACCEPTANCE_STATE_1", "/acceptance/separate_states/1", "training_evidence_integrity", "acceptance state"),
    _semantic_leaf_rule("SEM_ACCEPTANCE_STATE_2", "/acceptance/separate_states/2", "checkpoint_integrity", "acceptance state"),
    _semantic_leaf_rule("SEM_ACCEPTANCE_STATE_3", "/acceptance/separate_states/3", "development_accuracy_reported", "acceptance state"),
    _semantic_leaf_rule("SEM_ACCEPTANCE_STATE_4", "/acceptance/separate_states/4", "model_selection_certified", "acceptance state"),
    _semantic_leaf_rule("SEM_ACCEPTANCE_STATE_5", "/acceptance/separate_states/5", "confirmatory_access", "acceptance state"),
    _semantic_leaf_rule("SEM_ACCEPTANCE_STATE_6", "/acceptance/separate_states/6", "test_access", "acceptance state"),
    _semantic_leaf_rule("SEM_ACCEPTANCE_STATE_7", "/acceptance/separate_states/7", "speed_measurement", "acceptance state"),
    _semantic_leaf_rule("SEM_ACCEPTANCE_OPTIMIZER_STEPS", "/acceptance/expected_optimizer_steps_complete", True, "acceptance completion"),
    _semantic_leaf_rule("SEM_ACCEPTANCE_FINITE", "/acceptance/loss_and_gradients_finite", True, "acceptance finiteness"),
    _semantic_leaf_rule("SEM_ACCEPTANCE_AMP_EVENTS", "/acceptance/amp_skip_or_overflow_allowed", False, "acceptance AMP policy"),
    _semantic_leaf_rule("SEM_ACCEPTANCE_BOUND", "/acceptance/completion_evidence_and_checkpoints_bound", True, "acceptance evidence binding"),
    _semantic_leaf_rule("SEM_ACCEPTANCE_LOADABLE", "/acceptance/checkpoints_loadable", True, "acceptance checkpoint integrity"),
    _semantic_leaf_rule("SEM_ACCEPTANCE_DEVELOPMENT_ONLY", "/acceptance/development_accuracy_reporting_only", True, "acceptance development policy"),
    _semantic_leaf_rule("SEM_ACCEPTANCE_CONFIRMATORY", "/acceptance/confirmatory_metrics_accessed", False, "acceptance confirmatory policy"),
    _semantic_leaf_rule("SEM_ACCEPTANCE_TEST", "/acceptance/test_accessed", False, "acceptance test policy"),
    _semantic_leaf_rule("SEM_ACCEPTANCE_SPEED", "/acceptance/speed_measured", False, "acceptance speed policy"),
    _semantic_sequence_rule(
        "SEM_VENDOR_INCLUDE_ORDER",
        "/source_bindings/vendor_includes",
        ("dataloader", "optimizer", "model", "runtime"),
        "vendor include role order",
        member_field="role",
    ),
    _semantic_sequence_rule(
        "SEM_R18_AUGMENTATION_ORDER",
        "/augmentation/transforms",
        (
            "RandomPhotometricDistort",
            "RandomZoomOut",
            "RandomIoUCrop",
            "SanitizeBoundingBoxes",
            "RandomHorizontalFlip",
            "Resize",
            "SanitizeBoundingBoxes",
            "ConvertPILImage",
            "ConvertBoxes",
        ),
        "R18 augmentation ordered identity",
        member_field="type",
    ),
    _semantic_sequence_rule(
        "SEM_R18_TRAIN_ONLY_STOP_ORDER",
        "/augmentation/stopped_transforms",
        ("RandomPhotometricDistort", "RandomZoomOut", "RandomIoUCrop"),
        "train-only augmentation stop order",
    ),
    _semantic_sequence_rule(
        "SEM_SELECTION_TIE_BREAKER_ORDER",
        "/evaluation_and_selection/tie_breakers",
        ("AP50", "AR500", "earlier_epoch"),
        "selection tie-breaker order",
    ),
    _semantic_sequence_rule(
        "SEM_CHECKPOINT_STATE_ORDER",
        "/checkpoint_policy/required_state",
        (
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
        ),
        "checkpoint state order and set",
    ),
    _semantic_sequence_rule(
        "SEM_ACCEPTANCE_STATE_ORDER",
        "/acceptance/separate_states",
        (
            "training_runtime_integrity",
            "training_evidence_integrity",
            "checkpoint_integrity",
            "development_accuracy_reported",
            "model_selection_certified",
            "confirmatory_access",
            "test_access",
            "speed_measurement",
        ),
        "acceptance state order and set",
    ),
    _semantic_relation_rule(
        "REL_EMA_DEVELOPMENT_WEIGHTS",
        {
            "/ema/enabled": True,
            "/ema/development_evaluation_weights": "ema",
            "/ema/model_selection_weights": "ema",
        },
        "EMA required state and evaluation weights",
    ),
    _semantic_relation_rule(
        "REL_PRIMARY_EVALUATOR_LAUNCH_GATE",
        {
            "/evaluation_and_selection/primary_evaluator_required_before_training": True,
            "/evaluation_and_selection/primary_evaluator_independently_certified": False,
            "/evaluation_and_selection/training_launch_blocked": True,
        },
        "uncertified primary evaluator launch gate",
    ),
    _semantic_relation_rule(
        "REL_DEVELOPMENT_ONLY_SELECTION",
        {
            "/evaluation_and_selection/development_only": True,
            "/data_roles/development": "development",
            "/data_roles/confirmatory": "sealed_and_forbidden",
            "/data_roles/test": "forbidden",
            "/acceptance/development_accuracy_reporting_only": True,
            "/acceptance/confirmatory_metrics_accessed": False,
            "/acceptance/test_accessed": False,
        },
        "development-only selection and access seal",
    ),
    _semantic_relation_rule(
        "REL_RANDOM_INITIALIZATION_NO_SOURCE",
        {
            "/initialization/initialization": "random",
            "/initialization/pretrained": False,
            "/initialization/checkpoint": None,
            "/initialization/network_download_forbidden": True,
            "/initialization/external_checkpoint_forbidden": True,
        },
        "random initialization source exclusion",
    ),
    _semantic_relation_rule(
        "REL_SINGLE_FORMAL_RUN_NO_RETRY",
        {
            "/initialization/formal_run_count": 1,
            "/initialization/seed_selection_forbidden": True,
            "/checkpoint_policy/resume": False,
            "/checkpoint_policy/retry": False,
            "/checkpoint_policy/overwrite": False,
            "/checkpoint_policy/exactly_once_run_identity": True,
        },
        "single formal run exactly-once policy",
    ),
    _semantic_relation_rule(
        "REL_AMP_PLACEHOLDER_GATE",
        {
            "/amp/enabled": True,
            "/amp/scaler_type": "GradScaler",
            "/amp/init_scale": "IMPLEMENTATION_MUST_FREEZE_EXPLICITLY",
            "/amp/growth_factor": "IMPLEMENTATION_MUST_FREEZE_EXPLICITLY",
            "/amp/backoff_factor": "IMPLEMENTATION_MUST_FREEZE_EXPLICITLY",
            "/amp/growth_interval": "IMPLEMENTATION_MUST_FREEZE_EXPLICITLY",
            "/acceptance/amp_skip_or_overflow_allowed": False,
        },
        "AMP required state and unresolved implementation placeholders",
    ),
    _semantic_relation_rule(
        "REL_ADAMW_PARAMETER_ROLES",
        {
            "/optimizer/type": "AdamW",
            "/optimizer/parameter_groups_must_be_mutually_exclusive": True,
            "/optimizer/parameter_groups_must_cover_all_trainable_parameters": True,
            "/optimizer/parameter_identity_audit_required": True,
        },
        "AdamW parameter-group identity roles",
    ),
    _semantic_relation_rule(
        "REL_EPOCH_SCHEDULER_POLICY",
        {
            "/learning_rate/warmup_type": "LinearWarmup",
            "/learning_rate/scheduler_type": "MultiStepLR",
            "/learning_rate/scheduler_step_unit": "epoch",
            "/schedule/steps_per_epoch_source": "train_core_dataset_length_and_drop_last",
        },
        "epoch scheduler and warmup policy",
    ),
    _semantic_relation_rule(
        "REL_R18_AUGMENTATION_TRAIN_ONLY",
        {
            "/augmentation/multiscale_enabled": False,
            "/augmentation/runtime_transform_identity_and_rng_binding_required": True,
            "/schedule/multiscale_enabled": False,
        },
        "R18 augmentation runtime and train-only policy",
    ),
    _semantic_relation_rule(
        "REL_CHECKPOINT_EXACTLY_ONCE_OWNERSHIP",
        {
            "/checkpoint_policy/exactly_once_run_identity": True,
            "/checkpoint_policy/interrupted_run_status": "PERMANENT_FAIL",
            "/checkpoint_policy/last_checkpoint_every_epoch": True,
            "/checkpoint_policy/best_checkpoint_on_selection_improvement": True,
            "/checkpoint_policy/final_epoch_checkpoint": True,
            "/checkpoint_policy/atomic_write_required": True,
            "/checkpoint_policy/readback_required": True,
            "/checkpoint_policy/loadability_check_required": True,
            "/checkpoint_policy/sha256_required": True,
            "/checkpoint_policy/inventory_required": True,
        },
        "checkpoint owner and exactly-once evidence policy",
    ),
    _semantic_relation_rule(
        "REL_EVIDENCE_ACCEPTANCE_STATES",
        {
            "/acceptance/expected_optimizer_steps_complete": True,
            "/acceptance/loss_and_gradients_finite": True,
            "/acceptance/completion_evidence_and_checkpoints_bound": True,
            "/acceptance/checkpoints_loadable": True,
            "/acceptance/speed_measured": False,
        },
        "evidence and acceptance decision states",
    ),
)


class TrainingContractError(ValueError):
    """Raised when the portable formal-training contract fails closed."""


def canonical_training_contract_bytes(config: Any) -> bytes:
    """Return deterministic compact JSON bytes without a trailing newline."""

    try:
        return json.dumps(
            config, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TrainingContractError("training contract is not finite JSON data") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TrainingContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise TrainingContractError(f"non-finite JSON number: {value}")


def _parse_portable_json(raw: bytes, *, label: str) -> dict[str, Any]:
    if raw.startswith(b"\xef\xbb\xbf"):
        raise TrainingContractError(f"{label} must not contain a UTF-8 BOM")
    if b"\r" in raw or not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise TrainingContractError(f"{label} must use LF and exactly one trailing LF")
    try:
        text = raw.decode("utf-8")
        value = json.loads(
            text, object_pairs_hook=_strict_pairs, parse_constant=_reject_constant
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise TrainingContractError(f"cannot parse {label}") from exc
    if type(value) is not dict:
        raise TrainingContractError(f"{label} must be a JSON object")
    return value


def _assert_json_types(value: Any, field: str = "contract") -> None:
    if type(value) is dict:
        for key, child in value.items():
            if type(key) is not str:
                raise TrainingContractError(f"{field} contains a non-string key")
            _assert_json_types(child, f"{field}.{key}")
    elif type(value) is list:
        for index, child in enumerate(value):
            _assert_json_types(child, f"{field}[{index}]")
    elif type(value) not in {str, int, float, bool, type(None)}:
        raise TrainingContractError(f"{field} has a non-JSON or non-strict scalar type")


def _pointer_child(pointer: str, child: str | int) -> str:
    token = str(child).replace("~", "~0").replace("/", "~1")
    return f"{pointer}/{token}" if pointer else f"/{token}"


def _validate_closed_schema(value: Any, schema: Any = _TRAINING_CONTRACT_SCHEMA, pointer: str = "") -> None:
    """Validate every container, exact key set, and leaf role before frozen identity."""

    kind, detail = schema
    location = pointer or "/"
    if kind == "object":
        if type(value) is not dict:
            raise TrainingContractError(f"closed schema object type mismatch at {location}")
        actual = set(value)
        required = set(detail)
        missing = sorted(required - actual)
        extra = sorted(actual - required, key=str)
        if missing and extra:
            raise TrainingContractError(f"closed schema renamed keys at {location}: missing={missing} extra={extra}")
        if missing:
            raise TrainingContractError(f"closed schema missing keys at {location}: {missing}")
        if extra:
            raise TrainingContractError(f"closed schema extra keys at {location}: {extra}")
        for key, child_schema in detail.items():
            _validate_closed_schema(value[key], child_schema, _pointer_child(pointer, key))
        return
    if kind == "list":
        if type(value) is not list:
            raise TrainingContractError(f"closed schema list type mismatch at {location}")
        if len(value) != len(detail):
            raise TrainingContractError(
                f"closed schema list length mismatch at {location}: expected={len(detail)} actual={len(value)}"
            )
        for index, child_schema in enumerate(detail):
            _validate_closed_schema(value[index], child_schema, _pointer_child(pointer, index))
        return
    if kind == "literal":
        if type(value) is not type(detail) or value != detail:
            raise TrainingContractError(f"closed schema literal mismatch at {location}")
        return
    if kind != "scalar":
        raise RuntimeError(f"unknown closed schema kind: {kind}")
    if type(value) is not detail:
        raise TrainingContractError(f"closed schema scalar type mismatch at {location}: expected={detail.__name__}")


def _numeric_values(value: Any, pointer: str = "") -> dict[str, int | float]:
    rows: dict[str, int | float] = {}
    if type(value) is dict:
        for key, child in value.items():
            rows.update(_numeric_values(child, _pointer_child(pointer, key)))
    elif type(value) is list:
        for index, child in enumerate(value):
            rows.update(_numeric_values(child, _pointer_child(pointer, index)))
    elif type(value) in {int, float}:
        rows[pointer] = value
    return rows


def _validate_numeric_constraints(config: Any) -> None:
    """Validate every numeric leaf independently of whole-contract identity."""

    values = _numeric_values(config)
    expected = set(_TRAINING_CONTRACT_NUMERIC_CONSTRAINTS)
    actual = set(values)
    if actual != expected:
        raise TrainingContractError(
            f"numeric constraint registry coverage drift: missing={sorted(actual - expected)} orphan={sorted(expected - actual)}"
        )
    for pointer, constraint in _TRAINING_CONTRACT_NUMERIC_CONSTRAINTS.items():
        value = values[pointer]
        expected_type = constraint["expected_type"]
        if type(value) is not expected_type:
            raise TrainingContractError(
                f"numeric type mismatch at {pointer}: observed={value!r} expected={expected_type.__name__}"
            )
        if constraint["finite"] and type(value) is float and not math.isfinite(value):
            raise TrainingContractError(f"numeric nonfinite at {pointer}: observed={value!r}")
        lower = constraint["lower"]
        if lower is not None and (value < lower or (value == lower and not constraint["lower_inclusive"])):
            operator = ">=" if constraint["lower_inclusive"] else ">"
            raise TrainingContractError(
                f"numeric out_of_range at {pointer}: observed={value!r} constraint={operator}{lower!r}"
            )
        upper = constraint["upper"]
        if upper is not None and (value > upper or (value == upper and not constraint["upper_inclusive"])):
            operator = "<=" if constraint["upper_inclusive"] else "<"
            raise TrainingContractError(
                f"numeric out_of_range at {pointer}: observed={value!r} constraint={operator}{upper!r}"
            )
        if value != constraint["exact"]:
            raise TrainingContractError(
                f"numeric frozen_literal_mismatch at {pointer}: observed={value!r} expected={constraint['exact']!r}"
            )


_MISSING = object()


def _value_at_pointer(value: Any, pointer: str) -> Any:
    current = value
    if pointer == "/":
        return current
    for raw_token in pointer.lstrip("/").split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if type(current) is dict:
            if token not in current:
                return _MISSING
            current = current[token]
        elif type(current) is list and token.isdigit():
            index = int(token)
            if index >= len(current):
                return _MISSING
            current = current[index]
        else:
            return _MISSING
    return current


def _semantic_leaf_inventory(config: Any) -> dict[str, Any]:
    """Enumerate semantic leaves without consulting files or the digest."""

    rows: dict[str, Any] = {}

    def walk(value: Any, pointer: str = "") -> None:
        if type(value) is dict:
            for key, child in value.items():
                walk(child, _pointer_child(pointer, key))
        elif type(value) is list:
            for index, child in enumerate(value):
                walk(child, _pointer_child(pointer, index))
        elif pointer not in _TRAINING_CONTRACT_NUMERIC_CONSTRAINTS and pointer not in _SEMANTIC_SYNTAX_OR_BINDING_POINTERS:
            rows[pointer] = value

    walk(config)
    return rows


def _semantic_leaf_rules() -> tuple[dict[str, Any], ...]:
    return tuple(rule for rule in _TRAINING_CONTRACT_SEMANTIC_RULES if rule["kind"] in {"literal", "enum"})


def _semantic_sequence_rules() -> tuple[dict[str, Any], ...]:
    return tuple(rule for rule in _TRAINING_CONTRACT_SEMANTIC_RULES if rule["kind"] == "ordered_list")


def _semantic_relation_rules() -> tuple[dict[str, Any], ...]:
    return tuple(rule for rule in _TRAINING_CONTRACT_SEMANTIC_RULES if rule["kind"] == "required_relation")


def semantic_contract_inventory(config: Any) -> dict[str, Any]:
    """Return deterministic coverage data for independent semantic audits."""

    actual = tuple(_semantic_leaf_inventory(config))
    registered = tuple(rule["pointer"] for rule in _semantic_leaf_rules())
    duplicates = tuple(sorted({pointer for pointer in registered if registered.count(pointer) > 1}))
    return {
        "leaf_count": len(actual),
        "registered_leaf_count": len(registered),
        "leaf_pointers": actual,
        "registered_leaf_pointers": registered,
        "missing": tuple(sorted(set(actual) - set(registered))),
        "extra": tuple(sorted(set(registered) - set(actual))),
        "duplicate": duplicates,
        "sequence_rule_count": len(_semantic_sequence_rules()),
        "relation_rule_count": len(_semantic_relation_rules()),
        "excluded_pointers": tuple(sorted(_SEMANTIC_SYNTAX_OR_BINDING_POINTERS)),
    }


def _semantic_error(rule: dict[str, Any], pointer: str, observed: Any, expected: Any, category: str) -> None:
    raise TrainingContractError(
        f"semantic {category} mismatch: rule_id={rule['rule_id']} pointer={pointer} "
        f"observed={observed!r} expected={expected!r}"
    )


def _validate_semantic_registry_coverage(config: Any) -> None:
    inventory = semantic_contract_inventory(config)
    rule_ids = [rule["rule_id"] for rule in _TRAINING_CONTRACT_SEMANTIC_RULES]
    duplicate_rule_ids = sorted({rule_id for rule_id in rule_ids if rule_ids.count(rule_id) > 1})
    if duplicate_rule_ids or inventory["missing"] or inventory["extra"] or inventory["duplicate"]:
        raise TrainingContractError(
            "semantic registry coverage drift: "
            f"missing={list(inventory['missing'])} extra={list(inventory['extra'])} "
            f"duplicate={list(inventory['duplicate'])} duplicate_rule_ids={duplicate_rule_ids}"
        )


def _validate_semantic_leaf_rules(config: Any) -> None:
    for rule in _semantic_leaf_rules():
        pointer = rule["pointer"]
        observed = _value_at_pointer(config, pointer)
        expected = rule["expected"]
        if observed is _MISSING:
            _semantic_error(rule, pointer, observed, expected, "missing")
        if type(observed) is not rule["expected_type"]:
            _semantic_error(rule, pointer, observed, rule["expected_type"].__name__, "type")
        allowed = rule["allowed"]
        if allowed is not None and observed not in allowed:
            _semantic_error(rule, pointer, observed, {"allowed": allowed}, "enum")
        if observed != expected:
            category = "enum_literal" if allowed is not None else "literal"
            _semantic_error(rule, pointer, observed, expected, category)


def _validate_semantic_sequence_rules(config: Any) -> None:
    for rule in _semantic_sequence_rules():
        pointer = rule["pointer"]
        observed_value = _value_at_pointer(config, pointer)
        expected = rule["expected"]
        if type(observed_value) is not list:
            _semantic_error(rule, pointer, observed_value, list, "type")
        member_field = rule["member_field"]
        if member_field is None:
            observed = tuple(observed_value)
        else:
            observed = tuple(
                item.get(member_field, _MISSING) if type(item) is dict else _MISSING
                for item in observed_value
            )
        try:
            same_set = set(observed) == set(expected)
        except TypeError:
            same_set = False
        if not same_set:
            _semantic_error(rule, pointer, observed, {"set": expected}, "set")
        if observed != expected:
            _semantic_error(rule, pointer, observed, expected, "order")


def _validate_semantic_relations(config: dict[str, Any]) -> None:
    for rule in _semantic_relation_rules():
        _validate_semantic_relation_rule(config, rule)


def _validate_semantic_relation_rule(config: dict[str, Any], rule: dict[str, Any]) -> None:
    for pointer, expected in rule["frozen"].items():
        observed = _value_at_pointer(config, pointer)
        if observed is _MISSING or type(observed) is not type(expected) or observed != expected:
            _semantic_error(rule, pointer, observed, expected, "relation")


def _validate_semantic_rules(config: Any) -> None:
    """Validate all non-numeric literals, ordered collections, and relations."""

    _validate_semantic_registry_coverage(config)
    _validate_semantic_leaf_rules(config)
    _validate_semantic_sequence_rules(config)
    _validate_semantic_relations(config)


def _closed_schema_object_roles(schema: Any = _TRAINING_CONTRACT_SCHEMA, pointer: str = "") -> tuple[str, ...]:
    """Return every closed object role in deterministic descriptor order."""

    kind, detail = schema
    rows: list[str] = []
    if kind == "object":
        rows.append(pointer or "/")
        for key, child in detail.items():
            rows.extend(_closed_schema_object_roles(child, _pointer_child(pointer, key)))
    elif kind == "list":
        for index, child in enumerate(detail):
            rows.extend(_closed_schema_object_roles(child, _pointer_child(pointer, index)))
    return tuple(rows)


def _assert_relative_path(value: Any, field: str) -> str:
    if type(value) is not str or not value or "\x00" in value or "\\" in value:
        raise TrainingContractError(f"{field} is not a portable POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise TrainingContractError(f"{field} is not a repository-relative POSIX path")
    return value


def _assert_sha(value: Any, field: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise TrainingContractError(f"{field} must be 64 lowercase hex characters")
    return value


def _validate_path_and_sha_fields(config: dict[str, Any]) -> None:
    sources = config["source_bindings"]
    baseline = sources["baseline_config"]
    _assert_relative_path(baseline["relative_path"], "baseline_config.relative_path")
    _assert_sha(baseline["raw_sha256"], "baseline_config.raw_sha256")
    _assert_sha(baseline["canonical_sha256"], "baseline_config.canonical_sha256")
    recipe = sources["vendor_recipe"]
    _assert_relative_path(recipe["relative_path"], "vendor_recipe.relative_path")
    _assert_sha(recipe["sha256"], "vendor_recipe.sha256")
    for index, item in enumerate(sources["vendor_includes"]):
        _assert_relative_path(item["relative_path"], f"vendor_includes[{index}].relative_path")
        _assert_sha(item["sha256"], f"vendor_includes[{index}].sha256")
    conversion = sources["conversion_r3"]
    _assert_relative_path(conversion["artifact_root"], "conversion_r3.artifact_root")
    for key, value in conversion.items():
        if key.endswith("sha256"):
            _assert_sha(value, f"conversion_r3.{key}")


def _validate_cross_fields(config: dict[str, Any]) -> None:
    # Keep this helper independently fail-closed for callers that audit the
    # cross-field layer directly, including synchronized two-sided mutations.
    _validate_semantic_relations(config)
    topology = config["topology"]
    if topology["train_micro_batch"] * topology["gradient_accumulation_steps"] * topology["world_size"] != topology["effective_train_batch"]:
        raise TrainingContractError("effective train batch arithmetic drift")
    initialization = config["initialization"]
    if initialization["formal_run_count"] != 1 or initialization["seed_selection_forbidden"] is not True:
        raise TrainingContractError("single formal run policy drift")
    schedule = config["schedule"]
    if schedule["augmentation_stop_epoch"] >= schedule["epochs"]:
        raise TrainingContractError("augmentation stop must precede completion")
    lr = config["learning_rate"]
    expected_events = sum(item < schedule["epochs"] for item in lr["milestones"])
    if lr["scheduler_step_unit"] != "epoch" or expected_events != lr["expected_decay_events_within_120_epochs"]:
        raise TrainingContractError("scheduler decay semantics drift")
    checkpoint = config["checkpoint_policy"]
    if schedule["epochs"] % checkpoint["periodic_checkpoint_frequency_epochs"] != 0 or not checkpoint["final_epoch_checkpoint"]:
        raise TrainingContractError("periodic/final checkpoint policy drift")
    if config["augmentation"]["stop_epoch"] != schedule["augmentation_stop_epoch"]:
        raise TrainingContractError("augmentation stop fields disagree")
    evaluation = config["evaluation_and_selection"]
    if evaluation["evaluation_frequency_epochs"] != schedule["development_evaluation_frequency_epochs"]:
        raise TrainingContractError("development evaluation frequency drift")
    if evaluation["primary_evaluator_independently_certified"] or not evaluation["training_launch_blocked"]:
        raise TrainingContractError("primary evaluator launch gate must remain closed")
    if config["optimizer"]["type"] != "AdamW":
        raise TrainingContractError("optimizer type must remain AdamW")
    if config["amp"]["enabled"] is not True or config["ema"]["enabled"] is not True:
        raise TrainingContractError("AMP/EMA required-state drift")
    if config["acceptance"]["required_epochs_complete"] != schedule["epochs"]:
        raise TrainingContractError("required epoch completion drift")
    if config["checkpoint_policy"]["exactly_once_run_identity"] is not True or any(
        config["checkpoint_policy"][key] is not False for key in ("resume", "retry", "overwrite")
    ):
        raise TrainingContractError("exactly-once checkpoint ownership drift")


def _validate_baseline_binding(config: dict[str, Any], baseline_config: dict[str, Any]) -> None:
    _assert_json_types(baseline_config, "baseline_config")
    source = config["source_bindings"]["baseline_config"]
    if source != {
        "relative_path": BASELINE_CONFIG_RELATIVE_PATH,
        "size_bytes": BASELINE_CONFIG_SIZE_BYTES,
        "raw_sha256": BASELINE_CONFIG_RAW_SHA256,
        "canonical_sha256": BASELINE_CONFIG_CANONICAL_SHA256,
    }:
        raise TrainingContractError("baseline source binding drift")
    if _sha256_bytes(canonical_training_contract_bytes(baseline_config)) != BASELINE_CONFIG_CANONICAL_SHA256:
        raise TrainingContractError("baseline canonical identity drift")
    contract = baseline_config.get("baseline_contract", {})
    model = baseline_config.get("model", {})
    roles = baseline_config.get("roles", {})
    if contract.get("baseline_id") != config["baseline_id"] or model.get("num_classes") != 10:
        raise TrainingContractError("baseline ID/class binding drift")
    if model.get("PResNet") != {"depth": 18, "pretrained": False} or model.get("num_queries") != 300:
        raise TrainingContractError("baseline architecture/initialization binding drift")
    if config["initialization"]["pretrained"] is not False or config["initialization"]["checkpoint"] is not None:
        raise TrainingContractError("T1 random initialization drift")
    if roles.get("train_core", {}).get("status") != "allowed" or roles.get("development", {}).get("status") != "allowed":
        raise TrainingContractError("baseline allowed-role binding drift")
    if roles.get("confirmatory", {}).get("status") != "sealed_and_forbidden" or roles.get("test", {}).get("status") != "forbidden":
        raise TrainingContractError("baseline forbidden-role binding drift")


def validate_training_contract(config: Any, baseline_config: Any) -> dict[str, Any]:
    """Validate exact T1 content and return a detached portable object."""

    _assert_json_types(config)
    _validate_closed_schema(config)
    _validate_numeric_constraints(config)
    _validate_path_and_sha_fields(config)
    _validate_semantic_rules(config)
    _validate_cross_fields(config)
    digest = _sha256_bytes(canonical_training_contract_bytes(config))
    if digest != TRAINING_CONTRACT_CANONICAL_SHA256:
        raise TrainingContractError("training contract frozen content drift")
    if type(baseline_config) is not dict:
        raise TrainingContractError("baseline config must be an exact dict")
    _validate_baseline_binding(config, baseline_config)
    return copy.deepcopy(config)


def load_training_contract(repo_root: str | Path, config_path: str | Path = TRAINING_CONFIG_RELATIVE_PATH) -> dict[str, Any]:
    """Load and validate the checked-in contract without runtime side effects."""

    root = Path(repo_root).resolve()
    relative = _assert_relative_path(str(config_path), "config_path")
    if relative != TRAINING_CONFIG_RELATIVE_PATH:
        raise TrainingContractError("training config path identity drift")
    training_raw = (root / relative).read_bytes()
    baseline_raw = (root / BASELINE_CONFIG_RELATIVE_PATH).read_bytes()
    baseline = _parse_portable_json(baseline_raw, label="baseline config")
    if len(baseline_raw) != BASELINE_CONFIG_SIZE_BYTES or _sha256_bytes(baseline_raw) != BASELINE_CONFIG_RAW_SHA256:
        raise TrainingContractError("baseline raw identity drift")
    if len(training_raw) != TRAINING_CONFIG_SIZE_BYTES or _sha256_bytes(training_raw) != TRAINING_CONFIG_RAW_SHA256:
        raise TrainingContractError("training config raw identity drift")
    config = _parse_portable_json(training_raw, label="training config")
    return validate_training_contract(config, baseline)


def _validate_vendor_source_files(root: Path, config: dict[str, Any]) -> None:
    bindings = [config["source_bindings"]["vendor_recipe"]]
    bindings.extend(config["source_bindings"]["vendor_includes"])
    for binding in bindings:
        relative = _assert_relative_path(binding["relative_path"], "vendor source path")
        path = root / relative
        if not path.is_file() or path.is_symlink() or _sha256_bytes(path.read_bytes()) != binding["sha256"]:
            raise TrainingContractError(f"vendor source identity drift: {relative}")


def training_contract_binding(repo_root: str | Path, config_path: str | Path = TRAINING_CONFIG_RELATIVE_PATH) -> dict[str, Any]:
    """Return raw/canonical identities after complete validation."""

    root = Path(repo_root).resolve()
    config = load_training_contract(root, config_path)
    _validate_vendor_source_files(root, config)
    raw = (root / TRAINING_CONFIG_RELATIVE_PATH).read_bytes()
    canonical = canonical_training_contract_bytes(config)
    return {
        "schema_version": 1,
        "training_contract_id": config["training_contract_id"],
        "baseline_id": config["baseline_id"],
        "relative_path": TRAINING_CONFIG_RELATIVE_PATH,
        "raw_size_bytes": len(raw),
        "raw_sha256": _sha256_bytes(raw),
        "canonical_size_bytes": len(canonical),
        "canonical_sha256": _sha256_bytes(canonical),
    }
