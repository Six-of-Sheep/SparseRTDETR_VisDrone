"""Portable, standard-library-only validation for the frozen T1 contract."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path, PurePosixPath
from typing import Any


TRAINING_CONFIG_RELATIVE_PATH = "configs/baseline/rtdetrv2_r18_visdrone_training_v1.json"
BASELINE_CONFIG_RELATIVE_PATH = "configs/baseline/rtdetrv2_r18_visdrone_baseline_v1.json"
VENDOR_RUNTIME_RELATIVE_PATH = "vendor/rtdetrv2_pytorch"
VENDOR_MANIFEST_RELATIVE_PATH = "manifests/rtdetrv2_upstream.json"
VENDOR_UPSTREAM_REPOSITORY = "https://github.com/lyuwenyu/RT-DETR.git"
VENDOR_UPSTREAM_BRANCH = "main"
VENDOR_UPSTREAM_COMMIT = "1c8ac3f7ba84f14bd5651ab7b1b70d69a5f55f47"
VENDOR_UPSTREAM_COMMIT_TIME = "2026-06-15T13:50:44+09:00"
VENDOR_UPSTREAM_ROOT_TREE = "b6de37e186373fc59b91d23c846bb0eda35b6986"
VENDOR_UPSTREAM_SUBTREE = "96a3b3e7e015d5e548e2917df2fec9641375e96e"
VENDOR_LICENSE_NAME = "Apache-2.0"
VENDOR_LICENSE_SHA256 = "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
VENDOR_MANIFEST_INVENTORY_ALGORITHM = "sha256(canonical_json(files))"
VENDOR_RUNTIME_FILE_COUNT = 124
VENDOR_RUNTIME_DIRECTORY_COUNT = 25
VENDOR_RUNTIME_TOTAL_SIZE_BYTES = 373735
VENDOR_RUNTIME_COMPACT_INVENTORY_SHA256 = "0fc6803665bc4b5720e983345b2cacb0147eceead9f882588880b6f8a0e68051"
VENDOR_MANIFEST_SIZE_BYTES = 33264
VENDOR_MANIFEST_RAW_SHA256 = "f65a2d475365346a5dd5ce4f46b022a135b187e21421e922cc41eee4d28d20ae"
VENDOR_MANIFEST_INVENTORY_SHA256 = "2312c80d5b0fba88d43ffc6807c3fc150ae74b77740f2ab65072f044e033d6d7"
CONVERSION_R3_RELATIVE_PATH = "artifacts/data/visdrone_protocol_v2_conversion_r3"
CONVERSION_R3_FILE_COUNT = 25
CONVERSION_R3_DIRECTORY_COUNT = 0
CONVERSION_R3_TOTAL_SIZE_BYTES = 304418794
CONVERSION_R3_COMPLETION_SHA256 = "46734010937168ac65bf27c3a6f4bf3f234554d5d4a99407903dec4b3ae9e817"
CONVERSION_R3_ARTIFACT_INVENTORY_SHA256 = "aaee01ce4e00749b9db8ac5e6e49cf35875a8b266e9c3efc237bdf0b8b109ae7"
CONVERSION_R3_ENTRY_INVENTORY_SHA256 = "ddf32745302d6095e1c3dde89b1a85f05db53e6b4cc8ee85638e0d45bfa8d982"
CONVERSION_R3_CONFIG_SHA256 = "58be8659d6d7ae6faace82b36d45523f80cd48bae1bbcc6da4586c53056ec0d0"
CONVERSION_R3_CATEGORY_CONTRACT_SHA256 = "b4b309f357cbe130a505a610dff340cc498dc74f766384c4559b2acd900728a0"
CONVERSION_R3_SOURCE_IDENTITY_SHA256 = "f0f16ba4438b51a09a6203f78884b199f8e2a2d3fb46309c357368a47a552bb9"
CONVERSION_R3_EXCLUDED_FILES = ("artifact_inventory.json", "completion.json")
CONVERSION_R3_AUTHORITY_FILES = frozenset(
    {
        "artifact_inventory.json",
        "completion.json",
        "config.json",
        "category_contract.json",
        "source_identity.json",
    }
)
CONVERSION_R3_FILE_MODE = 0o600
TRAINING_CONFIG_SIZE_BYTES = 10907
TRAINING_CONFIG_RAW_SHA256 = "0f9ddb2e6d8ec6d2b21e42f6c419511f5bd70df287457c2c2b6109eaf8a93297"
TRAINING_CONTRACT_CANONICAL_SHA256 = "a20c71ef90cb4ccc091a717286a1c49be8a1ebffb178156942b77798d3d7f868"
BASELINE_CONFIG_SIZE_BYTES = 4316
BASELINE_CONFIG_RAW_SHA256 = "38702c3483efcd3c3855552d087bbfd0fcfe3628fa9b1582847039f17eb083dd"
BASELINE_CONFIG_CANONICAL_SHA256 = "c392efd44de7738401c1136261c8ca628dea3d3b0fe355b79d0d69d6ed91bfe0"
PRIMARY_EVALUATOR_ID = "visdrone_official_primary_evaluator_v1"
PRIMARY_EVALUATOR_PROTOCOL_ID = "visdrone_official_style_v1"
PRIMARY_EVALUATOR_IMPLEMENTATION_COMMIT = "036cca4d127ddd9e10e3cc7900c3eb759b55f59f"
PRIMARY_EVALUATOR_IMPLEMENTATION_TREE = "fbe931976f6bac7d8e4b3bb319ff1905e99c5444"
PRIMARY_EVALUATOR_CONFIG_RELATIVE_PATH = "configs/baseline/visdrone_official_evaluator_v1.json"
PRIMARY_EVALUATOR_CONFIG_RAW_SIZE_BYTES = 3857
PRIMARY_EVALUATOR_CONFIG_RAW_SHA256 = "36cfa69b0ff645c47c871283f917577ca27c760daf424e9544ab363dbf080ff5"
PRIMARY_EVALUATOR_CONFIG_CANONICAL_SIZE_BYTES = 3295
PRIMARY_EVALUATOR_CONFIG_CANONICAL_SHA256 = "355de90bdb6007ed42ed65b3f653a1b8ea57f2184ab4fd48a9561b692b993998"
PRIMARY_EVALUATOR_MANIFEST_RELATIVE_PATH = "manifests/visdrone_det_toolkit_005445.json"
PRIMARY_EVALUATOR_MANIFEST_RAW_SIZE_BYTES = 4166
PRIMARY_EVALUATOR_MANIFEST_RAW_SHA256 = "71168baf15d6d945fd5a4a6c5605b5ba533524efeede2a9d4020127576c1b36f"
PRIMARY_EVALUATOR_MANIFEST_CANONICAL_SIZE_BYTES = 3351
PRIMARY_EVALUATOR_MANIFEST_CANONICAL_SHA256 = "5bad9faf7622fe4542aa3b46561d6d41fb2e4ee34f35551ecfd821c9577159e3"
PRIMARY_EVALUATOR_ARCHIVE_SIZE_BYTES = 40960
PRIMARY_EVALUATOR_ARCHIVE_SHA256 = "bf19dd9477210adf106c7cbf2a72370ed4af22dedb577f361f3dc9e77e99baa4"
PRIMARY_EVALUATOR_AUTHORITY_FILE_COUNT = 11
PRIMARY_EVALUATOR_AUTHORITY_INVENTORY_SHA256 = "35a14a021509b82f1238912e5c77ebb3559f9ee6daa3cc6c92db810b5dce5da0"
PRIMARY_EVALUATOR_AUTHORITY_FILE_SHA256 = {
    "utils/VOCap.m": "95dd1e02c956124e777caf6f44b50b0986a2a37785bd11982ef775e696c17945",
    "utils/calcAccuracy.m": "285508f54903acd75eeda346c11c8e53267e72facd648f2e22bce121d4f55d7f",
    "utils/compOas.m": "3e5c2d473c07bf2ddbe0d902284af98ef7104c609e867b2ed59083d31f3ed2af",
    "utils/createIntImg.m": "7ce3de4fc105be4f088cae5d1fcfc8f4383f3d7ffd4a551bf1afe711eff09f72",
    "utils/dropObjectsInIgr.m": "30dee2713d76a537f5c98ec5d0804e17cfd3993d83d8a80a70b90ef9461fa4de",
    "utils/evalRes.m": "610f0d078f1af8d7987e360e6e88665c0290587191d5542b469a9792402f539b",
    "utils/saveAnnoRes.m": "3210fb8fd98aed19cab61c996e0daf1dbfdb23fd5eb2fafff29afc71bb45876d",
}
PRIMARY_EVALUATOR_AUDIT_STAGE = "P3_BASELINE_PRIMARY_EVALUATOR_V1_INDEPENDENT_AUDIT_R1"
PRIMARY_EVALUATOR_AUDIT_CLASSIFICATION = "PRIMARY_EVALUATOR_V1_CERTIFIED"
PRIMARY_EVALUATOR_AUDIT_RETURN_CODE = 0
PRIMARY_EVALUATOR_AUDIT_SCRIPT_SIZE_BYTES = 51244
PRIMARY_EVALUATOR_AUDIT_SCRIPT_SHA256 = "4182f4082756fb2e52d6969e93889f24ae9af398cc98d67a1b9d990041b5524a"
PRIMARY_EVALUATOR_AUDIT_MUTATION_CASES_REJECTED = 129
PRIMARY_EVALUATOR_AUDIT_INDEPENDENT_ORACLE_CASES = 100
PRIMARY_EVALUATOR_AUDIT_EVALUATOR_TESTS_PASSED = 46
PRIMARY_EVALUATOR_AUDIT_FULL_CPU_TESTS_PASSED = 1116
PRIMARY_EVALUATOR_AUDIT_CLEAN_ARCHIVE_TESTS_PASSED = 46
PRIMARY_EVALUATOR_SOURCE_FILES = (
    ("primary_evaluator", "src/sparse_rtdetr/baseline/primary_evaluator.py", "d831fc641ac930822e693f99fbbcdd48abbe76738a618525135dd099963901bd"),
    ("evaluation_protocol", "src/sparse_rtdetr/data_protocol/evaluation.py", "e70ad71bb834b4cfc5d25441a78d2caa5982a86ae782918488924f67a832d216"),
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_OID = re.compile(r"[0-9a-f]{40}\Z")
_INT = ("scalar", int)
_FLOAT = ("scalar", float)
_BOOL = ("scalar", bool)
_STR = ("scalar", str)
_NULL = ("scalar", type(None))


def _object(**children: Any) -> tuple[str, dict[str, Any]]:
    return ("object", children)


def _list(*children: Any) -> tuple[str, tuple[Any, ...]]:
    return ("list", children)


_PRIMARY_EVALUATOR_SOURCE_FILE_SCHEMA = _object(role=_STR, relative_path=_STR, sha256=_STR)
_PRIMARY_EVALUATOR_CERTIFICATION_SCHEMA = _object(
    evaluator=_object(evaluator_id=_STR, protocol_id=_STR),
    implementation=_object(commit=_STR, tree=_STR),
    config=_object(
        relative_path=_STR,
        raw_size_bytes=_INT,
        raw_sha256=_STR,
        canonical_size_bytes=_INT,
        canonical_sha256=_STR,
    ),
    authority_manifest=_object(
        relative_path=_STR,
        raw_size_bytes=_INT,
        raw_sha256=_STR,
        canonical_size_bytes=_INT,
        canonical_sha256=_STR,
    ),
    authority_archive_inventory=_object(
        archive_size_bytes=_INT,
        archive_sha256=_STR,
        file_count=_INT,
        inventory_sha256=_STR,
    ),
    source_files=_list(_PRIMARY_EVALUATOR_SOURCE_FILE_SCHEMA, _PRIMARY_EVALUATOR_SOURCE_FILE_SCHEMA),
    independent_audit=_object(
        stage=_STR,
        classification=_STR,
        formal_return_code=_INT,
        script_size_bytes=_INT,
        script_sha256=_STR,
        mutation_cases_rejected=_INT,
        independent_oracle_cases=_INT,
        evaluator_tests_passed=_INT,
        full_cpu_tests_passed=_INT,
        clean_archive_tests_passed=_INT,
    ),
)


_VENDOR_INCLUDE_ROW = _object(role=_STR, relative_path=_STR, sha256=_STR)
_VENDOR_RUNTIME_BINDING = _object(
    relative_path=_STR,
    file_count=_INT,
    directory_count_excluding_root=_INT,
    total_size_bytes=_INT,
    compact_inventory_sha256=_STR,
)
_VENDOR_MANIFEST_BINDING = _object(
    relative_path=_STR,
    size_bytes=_INT,
    raw_sha256=_STR,
    canonical_inventory_sha256=_STR,
)
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
        vendor_runtime=_VENDOR_RUNTIME_BINDING,
        vendor_manifest=_VENDOR_MANIFEST_BINDING,
        primary_evaluator_certification=_PRIMARY_EVALUATOR_CERTIFICATION_SCHEMA,
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
        enabled=_BOOL, scaler_type=_STR, init_scale=_FLOAT,
        growth_factor=_FLOAT, backoff_factor=_FLOAT,
        growth_interval=_INT, nonfinite_loss_allowed=_INT,
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
    "/source_bindings/primary_evaluator_certification/config/raw_size_bytes": _numeric(int, PRIMARY_EVALUATOR_CONFIG_RAW_SIZE_BYTES, "size_bytes", lower=1),
    "/source_bindings/primary_evaluator_certification/config/canonical_size_bytes": _numeric(int, PRIMARY_EVALUATOR_CONFIG_CANONICAL_SIZE_BYTES, "size_bytes", lower=1),
    "/source_bindings/primary_evaluator_certification/authority_manifest/raw_size_bytes": _numeric(int, PRIMARY_EVALUATOR_MANIFEST_RAW_SIZE_BYTES, "size_bytes", lower=1),
    "/source_bindings/primary_evaluator_certification/authority_manifest/canonical_size_bytes": _numeric(int, PRIMARY_EVALUATOR_MANIFEST_CANONICAL_SIZE_BYTES, "size_bytes", lower=1),
    "/source_bindings/primary_evaluator_certification/authority_archive_inventory/archive_size_bytes": _numeric(int, PRIMARY_EVALUATOR_ARCHIVE_SIZE_BYTES, "size_bytes", lower=1),
    "/source_bindings/primary_evaluator_certification/authority_archive_inventory/file_count": _numeric(int, PRIMARY_EVALUATOR_AUTHORITY_FILE_COUNT, "count", lower=1),
    "/source_bindings/primary_evaluator_certification/independent_audit/formal_return_code": _numeric(int, PRIMARY_EVALUATOR_AUDIT_RETURN_CODE, "process_exit_code", lower=0),
    "/source_bindings/primary_evaluator_certification/independent_audit/script_size_bytes": _numeric(int, PRIMARY_EVALUATOR_AUDIT_SCRIPT_SIZE_BYTES, "size_bytes", lower=1),
    "/source_bindings/primary_evaluator_certification/independent_audit/mutation_cases_rejected": _numeric(int, PRIMARY_EVALUATOR_AUDIT_MUTATION_CASES_REJECTED, "case_count", lower=0),
    "/source_bindings/primary_evaluator_certification/independent_audit/independent_oracle_cases": _numeric(int, PRIMARY_EVALUATOR_AUDIT_INDEPENDENT_ORACLE_CASES, "case_count", lower=0),
    "/source_bindings/primary_evaluator_certification/independent_audit/evaluator_tests_passed": _numeric(int, PRIMARY_EVALUATOR_AUDIT_EVALUATOR_TESTS_PASSED, "test_count", lower=0),
    "/source_bindings/primary_evaluator_certification/independent_audit/full_cpu_tests_passed": _numeric(int, PRIMARY_EVALUATOR_AUDIT_FULL_CPU_TESTS_PASSED, "test_count", lower=0),
    "/source_bindings/primary_evaluator_certification/independent_audit/clean_archive_tests_passed": _numeric(int, PRIMARY_EVALUATOR_AUDIT_CLEAN_ARCHIVE_TESTS_PASSED, "test_count", lower=0),
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
    "/amp/init_scale": _numeric(float, 65536.0, "amp_init_scale", lower=0.0, lower_inclusive=False),
    "/amp/growth_factor": _numeric(float, 2.0, "amp_growth_factor", lower=1.0, lower_inclusive=False),
    "/amp/backoff_factor": _numeric(float, 0.5, "amp_backoff_factor", lower=0.0, lower_inclusive=False, upper=1.0, upper_inclusive=False),
    "/amp/growth_interval": _numeric(int, 2000, "amp_growth_interval", lower=1),
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
        "/source_bindings/primary_evaluator_certification/implementation/commit",
        "/source_bindings/primary_evaluator_certification/implementation/tree",
        "/source_bindings/primary_evaluator_certification/config/relative_path",
        "/source_bindings/primary_evaluator_certification/config/raw_sha256",
        "/source_bindings/primary_evaluator_certification/config/canonical_sha256",
        "/source_bindings/primary_evaluator_certification/authority_manifest/relative_path",
        "/source_bindings/primary_evaluator_certification/authority_manifest/raw_sha256",
        "/source_bindings/primary_evaluator_certification/authority_manifest/canonical_sha256",
        "/source_bindings/primary_evaluator_certification/authority_archive_inventory/archive_sha256",
        "/source_bindings/primary_evaluator_certification/authority_archive_inventory/inventory_sha256",
        *(f"/source_bindings/primary_evaluator_certification/source_files/{index}/{field}" for index in range(2) for field in ("relative_path", "sha256")),
        "/source_bindings/primary_evaluator_certification/independent_audit/script_sha256",
    }
)


# These vendor runtime and manifest observations belong to the external
# binding, whose exact values are checked atomically by dedicated layers.
_EXTERNAL_BINDING_POINTERS = frozenset(
    {
        "/source_bindings/vendor_runtime/relative_path",
        "/source_bindings/vendor_runtime/compact_inventory_sha256",
        "/source_bindings/vendor_runtime/file_count",
        "/source_bindings/vendor_runtime/directory_count_excluding_root",
        "/source_bindings/vendor_runtime/total_size_bytes",
        "/source_bindings/vendor_manifest/relative_path",
        "/source_bindings/vendor_manifest/size_bytes",
        "/source_bindings/vendor_manifest/raw_sha256",
        "/source_bindings/vendor_manifest/canonical_inventory_sha256",
    }
)
_EXTERNAL_BINDING_NUMERIC_POINTERS = frozenset(
    pointer for pointer in _EXTERNAL_BINDING_POINTERS if pointer.endswith(("/file_count", "/directory_count_excluding_root", "/total_size_bytes", "/size_bytes"))
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
    _semantic_leaf_rule("SEM_PRIMARY_CERTIFICATION_EVALUATOR", "/source_bindings/primary_evaluator_certification/evaluator/evaluator_id", PRIMARY_EVALUATOR_ID, "certified evaluator identity"),
    _semantic_leaf_rule("SEM_PRIMARY_CERTIFICATION_PROTOCOL", "/source_bindings/primary_evaluator_certification/evaluator/protocol_id", PRIMARY_EVALUATOR_PROTOCOL_ID, "certified protocol identity"),
    _semantic_leaf_rule("SEM_PRIMARY_CERTIFICATION_SOURCE_ROLE_0", "/source_bindings/primary_evaluator_certification/source_files/0/role", "primary_evaluator", "certified evaluator source role"),
    _semantic_leaf_rule("SEM_PRIMARY_CERTIFICATION_SOURCE_ROLE_1", "/source_bindings/primary_evaluator_certification/source_files/1/role", "evaluation_protocol", "certified evaluator source role"),
    _semantic_leaf_rule("SEM_PRIMARY_CERTIFICATION_AUDIT_STAGE", "/source_bindings/primary_evaluator_certification/independent_audit/stage", PRIMARY_EVALUATOR_AUDIT_STAGE, "independent audit identity"),
    _semantic_leaf_rule("SEM_PRIMARY_CERTIFICATION_AUDIT_CLASSIFICATION", "/source_bindings/primary_evaluator_certification/independent_audit/classification", PRIMARY_EVALUATOR_AUDIT_CLASSIFICATION, "independent audit identity"),
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
    _semantic_leaf_rule("SEM_PRIMARY_CERTIFIED", "/evaluation_and_selection/primary_evaluator_independently_certified", True, "primary evaluator certification"),
    _semantic_leaf_rule("SEM_TRAINING_LAUNCH_BLOCKED", "/evaluation_and_selection/training_launch_blocked", False, "training launch gate"),
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
        "SEM_PRIMARY_EVALUATOR_SOURCE_FILE_ORDER",
        "/source_bindings/primary_evaluator_certification/source_files",
        ("primary_evaluator", "evaluation_protocol"),
        "certified evaluator source-file order",
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
            "/evaluation_and_selection/primary_evaluator_independently_certified": True,
            "/evaluation_and_selection/training_launch_blocked": False,
        },
        "certified primary evaluator launch gate",
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
        "REL_AMP_EXECUTABLE_PARAMETERS",
        {
            "/amp/enabled": True,
            "/amp/scaler_type": "GradScaler",
            "/amp/init_scale": 65536.0,
            "/amp/growth_factor": 2.0,
            "/amp/backoff_factor": 0.5,
            "/amp/growth_interval": 2000,
            "/amp/nonfinite_loss_allowed": 0,
            "/amp/nonfinite_gradient_allowed": 0,
            "/amp/optimizer_skipped_steps_allowed": 0,
            "/amp/overflow_events_allowed": 0,
            "/acceptance/amp_skip_or_overflow_allowed": False,
        },
        "AMP executable parameters and required state",
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
    _semantic_relation_rule(
        "REL_VENDOR_RUNTIME_MANIFEST_IDENTITY",
        {
            "/source_bindings/vendor_upstream_commit": VENDOR_UPSTREAM_COMMIT,
            "/source_bindings/vendor_runtime/relative_path": VENDOR_RUNTIME_RELATIVE_PATH,
            "/source_bindings/vendor_manifest/relative_path": VENDOR_MANIFEST_RELATIVE_PATH,
        },
        "vendor runtime and manifest path identity",
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


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TrainingContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise TrainingContractError(f"non-finite JSON number: {value}")


def _parse_strict_json(raw: bytes, *, label: str, trailing_lf: bool) -> dict[str, Any]:
    if raw.startswith(b"\xef\xbb\xbf") or b"\x00" in raw or b"\r" in raw:
        raise TrainingContractError(f"{label} raw format is invalid")
    if trailing_lf and (not raw.endswith(b"\n") or raw.endswith(b"\n\n")):
        raise TrainingContractError(f"{label} must use LF and exactly one trailing LF")
    if not trailing_lf and raw.endswith(b"\n"):
        raise TrainingContractError(f"{label} must not have a trailing LF")
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


def _parse_portable_json(raw: bytes, *, label: str) -> dict[str, Any]:
    return _parse_strict_json(raw, label=label, trailing_lf=True)


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
    elif type(value) in {int, float} and pointer not in _EXTERNAL_BINDING_NUMERIC_POINTERS:
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
        elif (
            pointer not in _TRAINING_CONTRACT_NUMERIC_CONSTRAINTS
            and pointer not in _SEMANTIC_SYNTAX_OR_BINDING_POINTERS
            and pointer not in _EXTERNAL_BINDING_POINTERS
        ):
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
        "external_binding_pointers": tuple(sorted(_EXTERNAL_BINDING_POINTERS)),
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


def _assert_git_oid(value: Any, field: str) -> str:
    if type(value) is not str or _GIT_OID.fullmatch(value) is None:
        raise TrainingContractError(f"{field} must be 40 lowercase hex characters")
    return value


def _validate_primary_evaluator_certification(config: dict[str, Any]) -> None:
    certification = config["source_bindings"]["primary_evaluator_certification"]
    evaluator = certification["evaluator"]
    if evaluator != {
        "evaluator_id": PRIMARY_EVALUATOR_ID,
        "protocol_id": PRIMARY_EVALUATOR_PROTOCOL_ID,
    }:
        raise TrainingContractError("primary evaluator certification identity drift")
    if config["evaluation_and_selection"]["primary_evaluator"] != evaluator["protocol_id"]:
        raise TrainingContractError("primary evaluator protocol binding drift")

    implementation = certification["implementation"]
    _assert_git_oid(implementation["commit"], "primary evaluator implementation.commit")
    _assert_git_oid(implementation["tree"], "primary evaluator implementation.tree")
    if implementation != {
        "commit": PRIMARY_EVALUATOR_IMPLEMENTATION_COMMIT,
        "tree": PRIMARY_EVALUATOR_IMPLEMENTATION_TREE,
    }:
        raise TrainingContractError("primary evaluator implementation provenance drift")

    evaluator_config = certification["config"]
    _assert_relative_path(evaluator_config["relative_path"], "primary evaluator config.relative_path")
    _assert_sha(evaluator_config["raw_sha256"], "primary evaluator config.raw_sha256")
    _assert_sha(evaluator_config["canonical_sha256"], "primary evaluator config.canonical_sha256")
    if evaluator_config != {
        "relative_path": PRIMARY_EVALUATOR_CONFIG_RELATIVE_PATH,
        "raw_size_bytes": PRIMARY_EVALUATOR_CONFIG_RAW_SIZE_BYTES,
        "raw_sha256": PRIMARY_EVALUATOR_CONFIG_RAW_SHA256,
        "canonical_size_bytes": PRIMARY_EVALUATOR_CONFIG_CANONICAL_SIZE_BYTES,
        "canonical_sha256": PRIMARY_EVALUATOR_CONFIG_CANONICAL_SHA256,
    }:
        raise TrainingContractError("primary evaluator config certification identity drift")

    authority_manifest = certification["authority_manifest"]
    _assert_relative_path(authority_manifest["relative_path"], "primary evaluator authority_manifest.relative_path")
    _assert_sha(authority_manifest["raw_sha256"], "primary evaluator authority_manifest.raw_sha256")
    _assert_sha(authority_manifest["canonical_sha256"], "primary evaluator authority_manifest.canonical_sha256")
    if authority_manifest != {
        "relative_path": PRIMARY_EVALUATOR_MANIFEST_RELATIVE_PATH,
        "raw_size_bytes": PRIMARY_EVALUATOR_MANIFEST_RAW_SIZE_BYTES,
        "raw_sha256": PRIMARY_EVALUATOR_MANIFEST_RAW_SHA256,
        "canonical_size_bytes": PRIMARY_EVALUATOR_MANIFEST_CANONICAL_SIZE_BYTES,
        "canonical_sha256": PRIMARY_EVALUATOR_MANIFEST_CANONICAL_SHA256,
    }:
        raise TrainingContractError("primary evaluator authority manifest certification identity drift")

    archive_inventory = certification["authority_archive_inventory"]
    _assert_sha(archive_inventory["archive_sha256"], "primary evaluator archive.sha256")
    _assert_sha(archive_inventory["inventory_sha256"], "primary evaluator inventory.sha256")
    if archive_inventory != {
        "archive_size_bytes": PRIMARY_EVALUATOR_ARCHIVE_SIZE_BYTES,
        "archive_sha256": PRIMARY_EVALUATOR_ARCHIVE_SHA256,
        "file_count": PRIMARY_EVALUATOR_AUTHORITY_FILE_COUNT,
        "inventory_sha256": PRIMARY_EVALUATOR_AUTHORITY_INVENTORY_SHA256,
    }:
        raise TrainingContractError("primary evaluator archive/inventory certification identity drift")

    source_files = certification["source_files"]
    expected_source_files = [
        {"role": role, "relative_path": relative, "sha256": sha256}
        for role, relative, sha256 in PRIMARY_EVALUATOR_SOURCE_FILES
    ]
    for index, row in enumerate(source_files):
        _assert_relative_path(row["relative_path"], f"primary evaluator source_files[{index}].relative_path")
        _assert_sha(row["sha256"], f"primary evaluator source_files[{index}].sha256")
    if source_files != expected_source_files:
        raise TrainingContractError("primary evaluator source-file certification identity drift")

    audit = certification["independent_audit"]
    _assert_sha(audit["script_sha256"], "primary evaluator audit.script_sha256")
    if audit != {
        "stage": PRIMARY_EVALUATOR_AUDIT_STAGE,
        "classification": PRIMARY_EVALUATOR_AUDIT_CLASSIFICATION,
        "formal_return_code": PRIMARY_EVALUATOR_AUDIT_RETURN_CODE,
        "script_size_bytes": PRIMARY_EVALUATOR_AUDIT_SCRIPT_SIZE_BYTES,
        "script_sha256": PRIMARY_EVALUATOR_AUDIT_SCRIPT_SHA256,
        "mutation_cases_rejected": PRIMARY_EVALUATOR_AUDIT_MUTATION_CASES_REJECTED,
        "independent_oracle_cases": PRIMARY_EVALUATOR_AUDIT_INDEPENDENT_ORACLE_CASES,
        "evaluator_tests_passed": PRIMARY_EVALUATOR_AUDIT_EVALUATOR_TESTS_PASSED,
        "full_cpu_tests_passed": PRIMARY_EVALUATOR_AUDIT_FULL_CPU_TESTS_PASSED,
        "clean_archive_tests_passed": PRIMARY_EVALUATOR_AUDIT_CLEAN_ARCHIVE_TESTS_PASSED,
    }:
        raise TrainingContractError("primary evaluator independent-audit certification identity drift")


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
    vendor_runtime = sources["vendor_runtime"]
    _assert_relative_path(vendor_runtime["relative_path"], "vendor_runtime.relative_path")
    _assert_sha(vendor_runtime["compact_inventory_sha256"], "vendor_runtime.compact_inventory_sha256")
    if vendor_runtime != {
        "relative_path": VENDOR_RUNTIME_RELATIVE_PATH,
        "file_count": VENDOR_RUNTIME_FILE_COUNT,
        "directory_count_excluding_root": VENDOR_RUNTIME_DIRECTORY_COUNT,
        "total_size_bytes": VENDOR_RUNTIME_TOTAL_SIZE_BYTES,
        "compact_inventory_sha256": VENDOR_RUNTIME_COMPACT_INVENTORY_SHA256,
    }:
        raise TrainingContractError("vendor runtime source binding drift")
    vendor_manifest = sources["vendor_manifest"]
    _assert_relative_path(vendor_manifest["relative_path"], "vendor_manifest.relative_path")
    _assert_sha(vendor_manifest["raw_sha256"], "vendor_manifest.raw_sha256")
    _assert_sha(vendor_manifest["canonical_inventory_sha256"], "vendor_manifest.canonical_inventory_sha256")
    if vendor_manifest != {
        "relative_path": VENDOR_MANIFEST_RELATIVE_PATH,
        "size_bytes": VENDOR_MANIFEST_SIZE_BYTES,
        "raw_sha256": VENDOR_MANIFEST_RAW_SHA256,
        "canonical_inventory_sha256": VENDOR_MANIFEST_INVENTORY_SHA256,
    }:
        raise TrainingContractError("vendor manifest source binding drift")
    conversion = sources["conversion_r3"]
    _assert_relative_path(conversion["artifact_root"], "conversion_r3.artifact_root")
    for key, value in conversion.items():
        if key.endswith("sha256"):
            _assert_sha(value, f"conversion_r3.{key}")
    if conversion != {
        "artifact_root": CONVERSION_R3_RELATIVE_PATH,
        "completion_sha256": CONVERSION_R3_COMPLETION_SHA256,
        "artifact_inventory_sha256": CONVERSION_R3_ARTIFACT_INVENTORY_SHA256,
        "entry_canonical_inventory_sha256": CONVERSION_R3_ENTRY_INVENTORY_SHA256,
        "config_sha256": CONVERSION_R3_CONFIG_SHA256,
        "category_contract_sha256": CONVERSION_R3_CATEGORY_CONTRACT_SHA256,
        "source_identity_sha256": CONVERSION_R3_SOURCE_IDENTITY_SHA256,
    }:
        raise TrainingContractError("conversion R3 source binding drift")
    _validate_primary_evaluator_certification(config)


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
    if (
        evaluation["primary_evaluator_required_before_training"] is not True
        or evaluation["primary_evaluator_independently_certified"] is not True
        or evaluation["training_launch_blocked"] is not False
    ):
        raise TrainingContractError("primary evaluator certification gate is not open")
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


def _assert_repository_root(value: str | Path) -> str:
    try:
        root = os.fspath(value)
    except TypeError as exc:
        raise TrainingContractError("repo_root must be an absolute portable path") from exc
    if type(root) is not str or not root or "\x00" in root or "\\" in root:
        raise TrainingContractError("repo_root must be an absolute portable path")
    if root == "/":
        return root
    if not root.startswith("/") or root.endswith("/") or "//" in root:
        raise TrainingContractError("repo_root must be an absolute lexically normalized path")
    components = root[1:].split("/")
    if any(not component or component in {".", ".."} for component in components):
        raise TrainingContractError("repo_root must be an absolute lexically normalized path")
    return root


def _require_secure_fd_support() -> tuple[int, int, int, int]:
    required = ("O_DIRECTORY", "O_NOFOLLOW", "O_CLOEXEC", "O_NONBLOCK")
    if any(not hasattr(os, name) or not getattr(os, name) for name in required):
        raise TrainingContractError("secure repository file access is unavailable")
    supports_dir_fd = getattr(os, "supports_dir_fd", ())
    if os.open not in supports_dir_fd or os.stat not in supports_dir_fd:
        raise TrainingContractError("secure repository dir_fd access is unavailable")
    return (
        os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
        os.O_NOFOLLOW | os.O_CLOEXEC,
        os.O_CLOEXEC,
    )


def _object_identity(value: os.stat_result) -> tuple[int, int, int]:
    return value.st_dev, value.st_ino, stat.S_IFMT(value.st_mode)


def _file_snapshot(value: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _lstat_at(name: str, parent_fd: int) -> os.stat_result:
    while True:
        try:
            return os.lstat(name, dir_fd=parent_fd)
        except InterruptedError:
            continue
        except OSError as exc:
            raise TrainingContractError(f"secure repository lstat failed: {name}") from exc


def _open_at(name: str, flags: int, parent_fd: int | None = None) -> int:
    while True:
        try:
            if parent_fd is None:
                return os.open(name, flags)
            return os.open(name, flags, dir_fd=parent_fd)
        except InterruptedError:
            continue
        except OSError as exc:
            raise TrainingContractError(f"secure repository open failed: {name}") from exc


def _fstat_fd(fd: int) -> os.stat_result:
    while True:
        try:
            return os.fstat(fd)
        except InterruptedError:
            continue
        except OSError as exc:
            raise TrainingContractError("secure repository descriptor became invalid") from exc


def _readlink_fd(fd: int) -> str:
    while True:
        try:
            return os.readlink(f"/proc/self/fd/{fd}")
        except InterruptedError:
            continue
        except OSError as exc:
            raise TrainingContractError("secure repository descriptor path is unavailable") from exc


def _listdir_fd(fd: int) -> list[str]:
    while True:
        try:
            return list(os.listdir(fd))
        except InterruptedError:
            continue
        except (OSError, TypeError) as exc:
            raise TrainingContractError("secure repository directory enumeration failed") from exc


def _vendor_source_role(relative: str) -> str:
    if relative in {"LICENSE", "UPSTREAM.md"}:
        return "p3_additional"
    if relative.endswith((".yml", ".yaml")):
        return "config"
    if relative == "Dockerfile" or relative.endswith(("requirements.txt", "docker-compose.yml")):
        return "environment"
    if relative.startswith("references/deploy/"):
        return "deployment"
    if relative.startswith("tools/"):
        return "tool"
    if relative.startswith("src/data/"):
        return "data_adapter"
    if relative.startswith(("src/nn/", "src/zoo/")):
        return "model"
    if relative.startswith("src/optim/"):
        return "optimization"
    if relative.startswith("src/solver/"):
        return "training"
    if relative.startswith(("src/core/", "src/misc/")):
        return "runtime"
    return "package_or_documentation"


class _DirectoryRecord:
    __slots__ = ("name", "parent_fd", "fd", "identity", "snapshot")

    def __init__(
        self,
        name: str,
        parent_fd: int,
        fd: int,
        identity: tuple[int, int, int],
        snapshot: tuple[int, int, int, int, int, int, int] | None = None,
    ) -> None:
        self.name = name
        self.parent_fd = parent_fd
        self.fd = fd
        self.identity = identity
        self.snapshot = snapshot


class _VerifiedRepository:
    """One verified directory-fd boundary for all checked-in contract inputs."""

    def __init__(self, repo_root: str | Path) -> None:
        self.lexical_root = _assert_repository_root(repo_root)
        self._directory_flags, self._file_flags, _, _ = _require_secure_fd_support()
        self._directory_fds: list[int] = []
        self._component_records: list[_DirectoryRecord] = []
        self._root_fd: int | None = None
        self._root_identity: tuple[int, int, int] | None = None
        self._root_device: int | None = None

    def __enter__(self) -> "_VerifiedRepository":
        try:
            self._open_root()
            return self
        except BaseException:
            self._close_all()
            raise

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        self._close_all()
        return False

    def _close_all(self) -> None:
        for fd in reversed(self._directory_fds):
            try:
                os.close(fd)
            except OSError:
                pass
        self._directory_fds.clear()

    def _open_root(self) -> None:
        root_fd = _open_at("/", self._directory_flags)
        self._directory_fds.append(root_fd)
        parent_fd = root_fd
        components = [] if self.lexical_root == "/" else self.lexical_root[1:].split("/")
        for name in components:
            observed = _lstat_at(name, parent_fd)
            if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
                raise TrainingContractError("repo_root contains a non-directory or symlink component")
            if observed.st_dev < 0:
                raise TrainingContractError("repo_root component identity is invalid")
            child_fd = _open_at(name, self._directory_flags, parent_fd)
            self._directory_fds.append(child_fd)
            opened = _fstat_fd(child_fd)
            if _object_identity(observed) != _object_identity(opened):
                raise TrainingContractError("repo_root component identity changed during open")
            self._component_records.append(
                _DirectoryRecord(name, parent_fd, child_fd, _object_identity(observed))
            )
            parent_fd = child_fd
        root_stat = _fstat_fd(parent_fd)
        self._root_fd = parent_fd
        self._root_identity = _object_identity(root_stat)
        self._root_device = root_stat.st_dev
        if _readlink_fd(parent_fd) != self.lexical_root:
            raise TrainingContractError("repo_root canonical identity drift")
        self._assert_root_stable()

    def _assert_root_stable(self) -> None:
        if self._root_fd is None or self._root_identity is None:
            raise TrainingContractError("repository boundary is not open")
        if _object_identity(_fstat_fd(self._root_fd)) != self._root_identity:
            raise TrainingContractError("repo_root descriptor identity drift")
        if _readlink_fd(self._root_fd) != self.lexical_root:
            raise TrainingContractError("repo_root canonical identity drift")
        for record in self._component_records:
            if _object_identity(_lstat_at(record.name, record.parent_fd)) != record.identity:
                raise TrainingContractError("repo_root path component identity drift")
            if _object_identity(_fstat_fd(record.fd)) != record.identity:
                raise TrainingContractError("repo_root descriptor identity drift")

    def _assert_descendant_stable(
        self,
        records: list[_DirectoryRecord],
        final_parent_fd: int,
        final_name: str,
        expected_final: tuple[int, int, int, int, int, int, int],
    ) -> None:
        self._assert_root_stable()
        for record in records:
            observed = _lstat_at(record.name, record.parent_fd)
            if _object_identity(observed) != record.identity:
                raise TrainingContractError("repository file path component identity drift")
            if record.snapshot is not None and _file_snapshot(observed) != record.snapshot:
                raise TrainingContractError("repository file path component metadata drift")
            opened = _fstat_fd(record.fd)
            if _object_identity(opened) != record.identity:
                raise TrainingContractError("repository file descriptor identity drift")
            if record.snapshot is not None and _file_snapshot(opened) != record.snapshot:
                raise TrainingContractError("repository file descriptor metadata drift")
        if _file_snapshot(_lstat_at(final_name, final_parent_fd)) != expected_final:
            raise TrainingContractError("repository file path identity drift")

    def _assert_directory_records_stable(self, records: list[_DirectoryRecord]) -> None:
        self._assert_root_stable()
        for record in records:
            observed = _lstat_at(record.name, record.parent_fd)
            if _object_identity(observed) != record.identity:
                raise TrainingContractError("repository directory path identity drift")
            if record.snapshot is not None and _file_snapshot(observed) != record.snapshot:
                raise TrainingContractError("repository directory path metadata drift")
            opened = _fstat_fd(record.fd)
            if _object_identity(opened) != record.identity:
                raise TrainingContractError("repository directory descriptor identity drift")
            if record.snapshot is not None and _file_snapshot(opened) != record.snapshot:
                raise TrainingContractError("repository directory descriptor metadata drift")

    def read_file(self, relative: str) -> bytes:
        relative = _assert_relative_path(relative, "repository file path")
        if self._root_fd is None or self._root_device is None:
            raise TrainingContractError("repository boundary is not open")
        self._assert_root_stable()
        components = relative.split("/")
        parent_fd = self._root_fd
        opened_directories: list[int] = []
        file_fd: int | None = None
        descendant_records: list[_DirectoryRecord] = []
        try:
            for index, name in enumerate(components):
                observed = _lstat_at(name, parent_fd)
                if index != len(components) - 1:
                    if (
                        stat.S_ISLNK(observed.st_mode)
                        or not stat.S_ISDIR(observed.st_mode)
                        or observed.st_dev != self._root_device
                    ):
                        raise TrainingContractError(f"repository path component is not a local directory: {relative}")
                    child_fd = _open_at(name, self._directory_flags, parent_fd)
                    opened_directories.append(child_fd)
                    opened = _fstat_fd(child_fd)
                    if _object_identity(observed) != _object_identity(opened):
                        raise TrainingContractError(f"repository path component identity drift: {relative}")
                    descendant_records.append(
                        _DirectoryRecord(name, parent_fd, child_fd, _object_identity(observed))
                    )
                    parent_fd = child_fd
                    continue
                if (
                    stat.S_ISLNK(observed.st_mode)
                    or not stat.S_ISREG(observed.st_mode)
                    or observed.st_nlink != 1
                    or observed.st_dev != self._root_device
                ):
                    raise TrainingContractError(f"repository file is not a stable ordinary file: {relative}")
                file_fd = _open_at(name, self._file_flags, parent_fd)
                opened = _fstat_fd(file_fd)
                if (
                    _file_snapshot(observed) != _file_snapshot(opened)
                    or not stat.S_ISREG(opened.st_mode)
                    or opened.st_nlink != 1
                    or opened.st_dev != self._root_device
                ):
                    raise TrainingContractError(f"repository file identity changed during open: {relative}")
                expected = _file_snapshot(opened)
                self._assert_descendant_stable(descendant_records, parent_fd, name, expected)
                chunks: list[bytes] = []
                while True:
                    try:
                        chunk = os.read(file_fd, 1024 * 1024)
                    except InterruptedError:
                        continue
                    except OSError as exc:
                        raise TrainingContractError(f"repository file read failed: {relative}") from exc
                    if not chunk:
                        break
                    chunks.append(chunk)
                after = _fstat_fd(file_fd)
                if _file_snapshot(after) != expected:
                    raise TrainingContractError(f"repository file changed while reading: {relative}")
                self._assert_descendant_stable(descendant_records, parent_fd, name, expected)
                return b"".join(chunks)
            raise TrainingContractError(f"repository file path is empty: {relative}")
        finally:
            if file_fd is not None:
                try:
                    os.close(file_fd)
                except OSError:
                    pass
            for fd in reversed(opened_directories):
                try:
                    os.close(fd)
                except OSError:
                    pass

    def _read_open_file(
        self,
        parent_fd: int,
        name: str,
        observed: os.stat_result,
        relative: str,
        directory_records: list[_DirectoryRecord],
    ) -> tuple[bytes, tuple[int, int, int, int, int, int, int]]:
        if self._root_device is None:
            raise TrainingContractError("repository boundary is not open")
        if (
            stat.S_ISLNK(observed.st_mode)
            or not stat.S_ISREG(observed.st_mode)
            or observed.st_nlink != 1
            or observed.st_dev != self._root_device
        ):
            raise TrainingContractError(f"repository file is not a stable ordinary file: {relative}")
        file_fd = _open_at(name, self._file_flags, parent_fd)
        try:
            opened = _fstat_fd(file_fd)
            if (
                _file_snapshot(observed) != _file_snapshot(opened)
                or not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or opened.st_dev != self._root_device
            ):
                raise TrainingContractError(f"repository file identity changed during open: {relative}")
            expected = _file_snapshot(opened)
            self._assert_descendant_stable(directory_records, parent_fd, name, expected)
            chunks: list[bytes] = []
            while True:
                try:
                    chunk = os.read(file_fd, 1024 * 1024)
                except InterruptedError:
                    continue
                except OSError as exc:
                    raise TrainingContractError(f"repository file read failed: {relative}") from exc
                if not chunk:
                    break
                chunks.append(chunk)
            raw = b"".join(chunks)
            after = _fstat_fd(file_fd)
            if _file_snapshot(after) != expected or len(raw) != expected[4]:
                raise TrainingContractError(f"repository file changed while reading: {relative}")
            self._assert_descendant_stable(directory_records, parent_fd, name, expected)
            return raw, expected
        finally:
            try:
                os.close(file_fd)
            except OSError:
                pass

    def _read_open_file_digest(
        self,
        parent_fd: int,
        name: str,
        observed: os.stat_result,
        relative: str,
        directory_records: list[_DirectoryRecord],
        *,
        capture: bool,
        expected_mode: int | None = None,
    ) -> tuple[int, str, bytes | None, tuple[int, int, int, int, int, int, int]]:
        if self._root_device is None:
            raise TrainingContractError("repository boundary is not open")
        if (
            stat.S_ISLNK(observed.st_mode)
            or not stat.S_ISREG(observed.st_mode)
            or observed.st_nlink != 1
            or observed.st_dev != self._root_device
        ):
            raise TrainingContractError(f"repository file is not a stable ordinary file: {relative}")
        if expected_mode is not None and stat.S_IMODE(observed.st_mode) != expected_mode:
            raise TrainingContractError(f"repository file mode drift: {relative}")
        file_fd = _open_at(name, self._file_flags, parent_fd)
        try:
            opened = _fstat_fd(file_fd)
            if (
                _file_snapshot(observed) != _file_snapshot(opened)
                or not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or opened.st_dev != self._root_device
                or (expected_mode is not None and stat.S_IMODE(opened.st_mode) != expected_mode)
            ):
                raise TrainingContractError(f"repository file identity changed during open: {relative}")
            expected = _file_snapshot(opened)
            self._assert_descendant_stable(directory_records, parent_fd, name, expected)
            digest = hashlib.sha256()
            captured = bytearray() if capture else None
            total_size = 0
            while True:
                try:
                    chunk = os.read(file_fd, 1024 * 1024)
                except InterruptedError:
                    continue
                except OSError as exc:
                    raise TrainingContractError(f"repository file read failed: {relative}") from exc
                if not chunk:
                    break
                digest.update(chunk)
                total_size += len(chunk)
                if captured is not None:
                    captured.extend(chunk)
            after = _fstat_fd(file_fd)
            if (
                _file_snapshot(after) != expected
                or total_size != expected[4]
                or (expected_mode is not None and stat.S_IMODE(after.st_mode) != expected_mode)
            ):
                raise TrainingContractError(f"repository file changed while reading: {relative}")
            self._assert_descendant_stable(directory_records, parent_fd, name, expected)
            return total_size, digest.hexdigest(), bytes(captured) if captured is not None else None, expected
        finally:
            try:
                os.close(file_fd)
            except OSError:
                pass

    def _open_inventory_directory(
        self, relative: str
    ) -> tuple[int, list[_DirectoryRecord], list[int]]:
        relative = _assert_relative_path(relative, "inventory directory path")
        if self._root_fd is None or self._root_device is None:
            raise TrainingContractError("repository boundary is not open")
        self._assert_root_stable()
        parent_fd = self._root_fd
        records: list[_DirectoryRecord] = []
        opened_directories: list[int] = []
        try:
            for name in relative.split("/"):
                observed = _lstat_at(name, parent_fd)
                if (
                    stat.S_ISLNK(observed.st_mode)
                    or not stat.S_ISDIR(observed.st_mode)
                    or observed.st_dev != self._root_device
                ):
                    raise TrainingContractError(f"inventory directory is not a local directory: {relative}")
                child_fd = _open_at(name, self._directory_flags, parent_fd)
                opened_directories.append(child_fd)
                opened = _fstat_fd(child_fd)
                if _file_snapshot(observed) != _file_snapshot(opened):
                    raise TrainingContractError(f"inventory directory identity changed during open: {relative}")
                records.append(
                    _DirectoryRecord(
                        name,
                        parent_fd,
                        child_fd,
                        _object_identity(observed),
                        _file_snapshot(observed),
                    )
                )
                parent_fd = child_fd
            self._assert_directory_records_stable(records)
            return parent_fd, records, opened_directories
        except BaseException:
            for fd in reversed(opened_directories):
                try:
                    os.close(fd)
                except OSError:
                    pass
            raise

    def observe_flat_directory(
        self,
        relative: str,
        *,
        capture_names: frozenset[str],
        expected_mode: int,
    ) -> dict[str, Any]:
        directory_fd, root_records, opened_directories = self._open_inventory_directory(relative)
        rows: list[dict[str, Any]] = []
        captured: dict[str, bytes] = {}
        try:
            before_directory = _file_snapshot(_fstat_fd(directory_fd))
            names = sorted(_listdir_fd(directory_fd))
            observed_entries: list[tuple[str, tuple[int, int, int, int, int, int, int]]] = []
            for name in names:
                observed = _lstat_at(name, directory_fd)
                if stat.S_ISLNK(observed.st_mode):
                    raise TrainingContractError(f"conversion R3 symlink: {name}")
                if stat.S_ISDIR(observed.st_mode):
                    raise TrainingContractError(f"conversion R3 nested directory: {name}")
                if not stat.S_ISREG(observed.st_mode):
                    raise TrainingContractError(f"conversion R3 special object: {name}")
                if observed.st_nlink != 1 or observed.st_dev != self._root_device:
                    raise TrainingContractError(f"conversion R3 file identity drift: {name}")
                size, sha256, raw, snapshot = self._read_open_file_digest(
                    directory_fd,
                    name,
                    observed,
                    f"{relative}/{name}",
                    root_records,
                    capture=name in capture_names,
                    expected_mode=expected_mode,
                )
                rows.append({"relative_path": name, "size_bytes": size, "sha256": sha256})
                if raw is not None:
                    captured[name] = raw
                observed_entries.append((name, snapshot))
            if sorted(_listdir_fd(directory_fd)) != names:
                raise TrainingContractError("conversion R3 directory entries changed during enumeration")
            if _file_snapshot(_fstat_fd(directory_fd)) != before_directory:
                raise TrainingContractError("conversion R3 directory metadata changed during enumeration")
            for name, expected in observed_entries:
                if _file_snapshot(_lstat_at(name, directory_fd)) != expected:
                    raise TrainingContractError(f"conversion R3 entry changed during enumeration: {name}")
            self._assert_directory_records_stable(root_records)
            rows.sort(key=lambda row: str(row["relative_path"]))
            return {
                "file_count": len(rows),
                "directory_count_excluding_root": 0,
                "total_size_bytes": sum(int(row["size_bytes"]) for row in rows),
                "rows": rows,
                "captured": captured,
            }
        finally:
            for fd in reversed(opened_directories):
                try:
                    os.close(fd)
                except OSError:
                    pass

    def inventory_directory(self, relative: str) -> dict[str, Any]:
        directory_fd, root_records, opened_directories = self._open_inventory_directory(relative)
        compact_rows: list[dict[str, Any]] = []
        manifest_rows: list[dict[str, Any]] = []
        directory_count = 0
        total_size = 0

        def visit(fd: int, prefix: str, records: list[_DirectoryRecord]) -> None:
            nonlocal directory_count, total_size
            before_directory = _file_snapshot(_fstat_fd(fd))
            names = sorted(_listdir_fd(fd))
            observed_entries: list[tuple[str, tuple[int, int, int, int, int, int, int]]] = []
            for name in names:
                observed = _lstat_at(name, fd)
                child_relative = f"{prefix}/{name}" if prefix else name
                if stat.S_ISLNK(observed.st_mode):
                    raise TrainingContractError(f"vendor inventory symlink: {child_relative}")
                if stat.S_ISDIR(observed.st_mode):
                    if observed.st_dev != self._root_device:
                        raise TrainingContractError(f"vendor inventory device drift: {child_relative}")
                    child_fd = _open_at(name, self._directory_flags, fd)
                    opened_directories.append(child_fd)
                    opened = _fstat_fd(child_fd)
                    if _file_snapshot(observed) != _file_snapshot(opened):
                        raise TrainingContractError(f"vendor directory identity changed during open: {child_relative}")
                    child_record = _DirectoryRecord(
                        name,
                        fd,
                        child_fd,
                        _object_identity(observed),
                        _file_snapshot(observed),
                    )
                    directory_count += 1
                    visit(child_fd, child_relative, [*records, child_record])
                    observed_entries.append((name, _file_snapshot(observed)))
                    continue
                if not stat.S_ISREG(observed.st_mode):
                    raise TrainingContractError(f"vendor inventory special object: {child_relative}")
                raw, snapshot = self._read_open_file(fd, name, observed, child_relative, records)
                total_size += len(raw)
                compact_rows.append(
                    {
                        "relative_path": child_relative,
                        "size_bytes": len(raw),
                        "sha256": _sha256_bytes(raw),
                    }
                )
                manifest_relative = f"{VENDOR_RUNTIME_RELATIVE_PATH}/{child_relative}"
                manifest_rows.append(
                    {
                        "relative_path": manifest_relative,
                        "size_bytes": len(raw),
                        "sha256": _sha256_bytes(raw),
                        "executable": bool(snapshot[2] & 0o111),
                        "source_role": _vendor_source_role(child_relative),
                    }
                )
                observed_entries.append((name, snapshot))
            if sorted(_listdir_fd(fd)) != names:
                raise TrainingContractError(f"vendor directory entries changed during enumeration: {prefix}")
            if _file_snapshot(_fstat_fd(fd)) != before_directory:
                raise TrainingContractError(f"vendor directory metadata changed during enumeration: {prefix}")
            for name, expected in observed_entries:
                if _file_snapshot(_lstat_at(name, fd)) != expected:
                    raise TrainingContractError(f"vendor entry changed during enumeration: {prefix}/{name}")
            self._assert_directory_records_stable(records)

        try:
            visit(directory_fd, "", root_records)
            compact_rows.sort(key=lambda row: str(row["relative_path"]))
            manifest_rows.sort(key=lambda row: str(row["relative_path"]))
            return {
                "file_count": len(compact_rows),
                "directory_count_excluding_root": directory_count,
                "total_size_bytes": total_size,
                "compact_inventory_sha256": _sha256_bytes(_canonical_json_bytes(compact_rows)),
                "manifest_inventory_sha256": _sha256_bytes(_canonical_json_bytes(manifest_rows)),
                "compact_rows": compact_rows,
                "manifest_rows": manifest_rows,
            }
        finally:
            for fd in reversed(opened_directories):
                try:
                    os.close(fd)
                except OSError:
                    pass


_VENDOR_MANIFEST_KEYS = {
    "schema_version",
    "upstream_repository",
    "upstream_branch",
    "upstream_commit",
    "upstream_commit_time",
    "upstream_root_tree",
    "upstream_subtree",
    "license_name",
    "license_sha256",
    "implementation",
    "vendor_relative_path",
    "file_count",
    "total_size_bytes",
    "inventory_algorithm",
    "canonical_inventory_sha256",
    "files",
}
_VENDOR_MANIFEST_ROW_KEYS = {"relative_path", "size_bytes", "sha256", "executable", "source_role"}
_VENDOR_SOURCE_ROLES = {
    "p3_additional",
    "config",
    "environment",
    "deployment",
    "tool",
    "data_adapter",
    "model",
    "optimization",
    "training",
    "runtime",
    "package_or_documentation",
}


def _parse_vendor_manifest(raw: bytes) -> dict[str, Any]:
    if b"\x00" in raw:
        raise TrainingContractError("vendor manifest contains NUL bytes")
    manifest = _parse_portable_json(raw, label="vendor manifest")
    if set(manifest) != _VENDOR_MANIFEST_KEYS:
        raise TrainingContractError("vendor manifest root schema drift")
    exact_strings = {
        "upstream_repository": VENDOR_UPSTREAM_REPOSITORY,
        "upstream_branch": VENDOR_UPSTREAM_BRANCH,
        "upstream_commit": VENDOR_UPSTREAM_COMMIT,
        "upstream_commit_time": VENDOR_UPSTREAM_COMMIT_TIME,
        "upstream_root_tree": VENDOR_UPSTREAM_ROOT_TREE,
        "upstream_subtree": VENDOR_UPSTREAM_SUBTREE,
        "license_name": VENDOR_LICENSE_NAME,
        "implementation": "rtdetrv2_pytorch",
        "vendor_relative_path": VENDOR_RUNTIME_RELATIVE_PATH,
        "inventory_algorithm": VENDOR_MANIFEST_INVENTORY_ALGORITHM,
    }
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise TrainingContractError("vendor manifest schema version drift")
    for key, expected in exact_strings.items():
        if type(manifest[key]) is not str or manifest[key] != expected:
            raise TrainingContractError(f"vendor manifest {key} drift")
    if type(manifest["license_sha256"]) is not str or manifest["license_sha256"] != VENDOR_LICENSE_SHA256:
        raise TrainingContractError("vendor manifest license SHA drift")
    if type(manifest["file_count"]) is not int or manifest["file_count"] != VENDOR_RUNTIME_FILE_COUNT:
        raise TrainingContractError("vendor manifest file count drift")
    if type(manifest["total_size_bytes"]) is not int or manifest["total_size_bytes"] != VENDOR_RUNTIME_TOTAL_SIZE_BYTES:
        raise TrainingContractError("vendor manifest total size drift")
    if type(manifest["canonical_inventory_sha256"]) is not str or _SHA256.fullmatch(manifest["canonical_inventory_sha256"]) is None:
        raise TrainingContractError("vendor manifest inventory SHA syntax drift")
    if manifest["canonical_inventory_sha256"] != VENDOR_MANIFEST_INVENTORY_SHA256:
        raise TrainingContractError("vendor manifest inventory SHA drift")
    rows = manifest["files"]
    if type(rows) is not list or len(rows) != VENDOR_RUNTIME_FILE_COUNT or rows != sorted(rows, key=lambda row: row.get("relative_path", "") if type(row) is dict else ""):
        raise TrainingContractError("vendor manifest file rows are not canonical")
    previous = None
    for row in rows:
        if type(row) is not dict or set(row) != _VENDOR_MANIFEST_ROW_KEYS:
            raise TrainingContractError("vendor manifest row schema drift")
        relative = _assert_relative_path(row["relative_path"], "vendor manifest relative_path")
        if not relative.startswith(VENDOR_RUNTIME_RELATIVE_PATH + "/"):
            raise TrainingContractError("vendor manifest path escapes vendor root")
        if previous is not None and relative <= previous:
            raise TrainingContractError("vendor manifest rows are duplicated or unsorted")
        previous = relative
        if type(row["size_bytes"]) is not int or row["size_bytes"] < 0:
            raise TrainingContractError("vendor manifest row size type drift")
        _assert_sha(row["sha256"], "vendor manifest row sha256")
        if type(row["executable"]) is not bool or type(row["source_role"]) is not str or row["source_role"] not in _VENDOR_SOURCE_ROLES:
            raise TrainingContractError("vendor manifest row metadata drift")
    return manifest


def _validate_vendor_inventory(
    manifest: dict[str, Any],
    manifest_raw: bytes,
    inventory: dict[str, Any],
    config: dict[str, Any],
) -> None:
    source = config["source_bindings"]
    vendor_runtime = source["vendor_runtime"]
    vendor_manifest = source["vendor_manifest"]
    if len(manifest_raw) != vendor_manifest["size_bytes"] or _sha256_bytes(manifest_raw) != vendor_manifest["raw_sha256"]:
        raise TrainingContractError("vendor manifest raw identity drift")
    if manifest["upstream_commit"] != source["vendor_upstream_commit"]:
        raise TrainingContractError("vendor upstream commit binding drift")
    if manifest["file_count"] != vendor_runtime["file_count"] or inventory["file_count"] != vendor_runtime["file_count"]:
        raise TrainingContractError("vendor runtime file count binding drift")
    if inventory["directory_count_excluding_root"] != vendor_runtime["directory_count_excluding_root"]:
        raise TrainingContractError("vendor runtime directory count binding drift")
    if manifest["total_size_bytes"] != vendor_runtime["total_size_bytes"] or inventory["total_size_bytes"] != vendor_runtime["total_size_bytes"]:
        raise TrainingContractError("vendor runtime total size binding drift")
    if inventory["compact_inventory_sha256"] != vendor_runtime["compact_inventory_sha256"]:
        raise TrainingContractError("vendor compact inventory binding drift")
    if inventory["manifest_inventory_sha256"] != vendor_manifest["canonical_inventory_sha256"]:
        raise TrainingContractError("vendor manifest inventory binding drift")
    if manifest["canonical_inventory_sha256"] != vendor_manifest["canonical_inventory_sha256"]:
        raise TrainingContractError("vendor manifest declared inventory drift")
    if inventory["manifest_rows"] != manifest["files"]:
        raise TrainingContractError("vendor manifest rows do not equal observed inventory")


_CONVERSION_INVENTORY_KEYS = {
    "schema_version",
    "artifacts",
    "canonical_inventory_sha256",
    "excluded_from_inventory",
}
_CONVERSION_INVENTORY_ROW_KEYS = {"relative_path", "size_bytes", "sha256"}
_CONVERSION_COMPLETION_KEYS = {
    "artifact_inventory_sha256",
    "completion_self_hash_included",
    "config_sha256",
    "confirmatory_metrics_accessed",
    "dataset_test_accessed_by_this_process",
    "input_protocol_config_sha256",
    "metrics_access_allowed",
    "output_path_recorded",
    "production_conversion_executed",
    "project_test_split_historically_observed",
    "protocol_id",
    "run_nonce",
    "schema_version",
    "selection_allowed",
    "selection_policy",
    "single_final_access_only",
    "split_plan_sha256",
    "status",
}
_CONVERSION_CONFIG_KEYS = {
    "categories",
    "confirmatory_metrics_accessed",
    "converter_schema_version",
    "data_identity",
    "evaluators",
    "parser",
    "production_split_manifest_generated",
    "protocol_id",
    "real_conversion_outputs_generated",
    "schema_version",
    "split",
    "test_access_allowed",
}
_CONVERSION_CATEGORY_KEYS = {"background_explicit", "categories", "num_classes", "raw_mappings", "schema_version"}
_CONVERSION_SOURCE_KEYS = {
    "audited_protocol_data_identity",
    "dataset_test_accessed_by_this_process",
    "project_test_split_historically_observed",
    "protocol_config_sha256",
    "protocol_id",
    "schema_version",
    "train_annotation_count",
    "train_image_count",
    "train_raw_manifest_sha256",
    "val_annotation_count",
    "val_image_count",
    "val_raw_manifest_sha256",
}
_CONVERSION_DATA_IDENTITY_KEYS = {
    "cross_split_exact_image_sha_overlap",
    "cross_split_shared_sequence_keys",
    "source",
    "test_access_allowed",
    "train",
    "val",
}
_CONVERSION_SPLIT_KEYS = {
    "confirmatory_source",
    "development",
    "distribution_tolerance_percentage_points",
    "evaluated_prefix_count",
    "feasible_prefix_count",
    "group_unit",
    "metrics_access_allowed",
    "planner",
    "salt",
    "seed",
    "selection_allowed",
    "selection_policy",
    "single_final_access_only",
    "target_confirmatory_images",
    "test",
}
_CONVERSION_FLAT_FILE_NAMES = frozenset(
    {
        "artifact_inventory.json",
        "category_contract.json",
        "completion.json",
        "config.json",
        "confirmatory_lineage.jsonl",
        "confirmatory_sealed_manifest.json",
        "conversion_audit.json",
        "development_coco.json",
        "development_ignore_regions.jsonl",
        "development_lineage.jsonl",
        "development_manifest.json",
        "invocation.json",
        "raw_train_manifest.part-00001.jsonl",
        "raw_train_manifest.part-00002.jsonl",
        "raw_train_manifest.part-00003.jsonl",
        "raw_train_manifest.part-00004.jsonl",
        "raw_val_manifest.jsonl",
        "round_trip_audit.json",
        "sequence_groups.json",
        "source_identity.json",
        "split_plan.json",
        "train_core_coco.json",
        "train_core_ignore_regions.jsonl",
        "train_core_lineage.jsonl",
        "train_core_manifest.json",
    }
)


def _assert_flat_filename(value: Any, field: str) -> str:
    if (
        type(value) is not str
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\x00" in value
    ):
        raise TrainingContractError(f"{field} is not a flat portable filename")
    return value


def _assert_exact_keys(value: Any, expected: set[str], field: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        raise TrainingContractError(f"{field} exact schema drift")
    return value


def _conversion_observed_row(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for row in rows:
        if row["relative_path"] == name:
            return row
    raise TrainingContractError(f"conversion R3 file missing: {name}")


def _validate_conversion_inventory_document(
    raw: bytes,
    observed_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    inventory = _parse_strict_json(raw, label="Conversion R3 artifact inventory", trailing_lf=False)
    _assert_exact_keys(inventory, _CONVERSION_INVENTORY_KEYS, "Conversion R3 artifact inventory")
    if type(inventory["schema_version"]) is not int or inventory["schema_version"] != 1:
        raise TrainingContractError("Conversion R3 inventory schema version drift")
    if inventory["excluded_from_inventory"] != list(CONVERSION_R3_EXCLUDED_FILES):
        raise TrainingContractError("Conversion R3 inventory exclusions drift")
    _assert_sha(inventory["canonical_inventory_sha256"], "Conversion R3 inventory canonical SHA")
    if inventory["canonical_inventory_sha256"] != CONVERSION_R3_ENTRY_INVENTORY_SHA256:
        raise TrainingContractError("Conversion R3 inventory canonical SHA drift")
    artifacts = inventory["artifacts"]
    if type(artifacts) is not list or len(artifacts) != CONVERSION_R3_FILE_COUNT - len(CONVERSION_R3_EXCLUDED_FILES):
        raise TrainingContractError("Conversion R3 inventory rows drift")
    previous: str | None = None
    for row in artifacts:
        _assert_exact_keys(row, _CONVERSION_INVENTORY_ROW_KEYS, "Conversion R3 inventory row")
        relative = _assert_flat_filename(row["relative_path"], "Conversion R3 inventory relative_path")
        if relative in CONVERSION_R3_EXCLUDED_FILES or relative not in _CONVERSION_FLAT_FILE_NAMES:
            raise TrainingContractError("Conversion R3 inventory path drift")
        if previous is not None and relative <= previous:
            raise TrainingContractError("Conversion R3 inventory rows are not unique and sorted")
        previous = relative
        if type(row["size_bytes"]) is not int or row["size_bytes"] < 0:
            raise TrainingContractError("Conversion R3 inventory size type drift")
        _assert_sha(row["sha256"], "Conversion R3 inventory row SHA")
    if _sha256_bytes(_canonical_json_bytes(artifacts)) != CONVERSION_R3_ENTRY_INVENTORY_SHA256:
        raise TrainingContractError("Conversion R3 inventory canonical rows drift")
    observed_artifacts = [
        row for row in observed_rows if row["relative_path"] not in CONVERSION_R3_EXCLUDED_FILES
    ]
    if artifacts != observed_artifacts:
        raise TrainingContractError("Conversion R3 observed rows differ from artifact inventory")
    source = config["source_bindings"]["conversion_r3"]
    inventory_row = _conversion_observed_row(observed_rows, "artifact_inventory.json")
    if inventory_row["sha256"] != source["artifact_inventory_sha256"]:
        raise TrainingContractError("Conversion R3 artifact inventory raw SHA binding drift")
    return inventory


def _validate_conversion_completion_document(
    raw: bytes,
    observed_rows: list[dict[str, Any]],
    config: dict[str, Any],
    source_identity: dict[str, Any],
) -> dict[str, Any]:
    completion = _parse_strict_json(raw, label="Conversion R3 completion", trailing_lf=False)
    _assert_exact_keys(completion, _CONVERSION_COMPLETION_KEYS, "Conversion R3 completion")
    fixed = {
        "schema_version": 1,
        "status": "COMPLETED",
        "protocol_id": "P3-VISDRONE-DATA-PROTOCOL-V2",
        "selection_policy": "feasibility_first_nearest_hash_prefix_v2",
        "selection_allowed": False,
        "metrics_access_allowed": False,
        "single_final_access_only": True,
        "confirmatory_metrics_accessed": False,
        "dataset_test_accessed_by_this_process": False,
        "completion_self_hash_included": False,
        "output_path_recorded": False,
        "production_conversion_executed": True,
        "project_test_split_historically_observed": True,
    }
    for key, expected in fixed.items():
        if type(completion[key]) is not type(expected) or completion[key] != expected:
            raise TrainingContractError(f"Conversion R3 completion {key} drift")
    for key in ("artifact_inventory_sha256", "config_sha256", "input_protocol_config_sha256", "run_nonce", "split_plan_sha256"):
        _assert_sha(completion[key], f"Conversion R3 completion {key}")
    source = config["source_bindings"]["conversion_r3"]
    if completion["artifact_inventory_sha256"] != source["artifact_inventory_sha256"]:
        raise TrainingContractError("Conversion R3 completion inventory binding drift")
    if completion["config_sha256"] != source["config_sha256"]:
        raise TrainingContractError("Conversion R3 completion config binding drift")
    if completion["input_protocol_config_sha256"] != source_identity["protocol_config_sha256"]:
        raise TrainingContractError("Conversion R3 completion protocol binding drift")
    if completion["split_plan_sha256"] != _conversion_observed_row(observed_rows, "split_plan.json")["sha256"]:
        raise TrainingContractError("Conversion R3 completion split binding drift")
    return completion


def _validate_conversion_category_document(raw: bytes) -> dict[str, Any]:
    category = _parse_strict_json(raw, label="Conversion R3 category contract", trailing_lf=False)
    _assert_exact_keys(category, _CONVERSION_CATEGORY_KEYS, "Conversion R3 category contract")
    if type(category["schema_version"]) is not int or category["schema_version"] != 1:
        raise TrainingContractError("Conversion R3 category schema version drift")
    if type(category["background_explicit"]) is not bool or category["background_explicit"] is not False:
        raise TrainingContractError("Conversion R3 category background drift")
    if type(category["num_classes"]) is not int or category["num_classes"] != 10:
        raise TrainingContractError("Conversion R3 category count drift")
    categories = category["categories"]
    mappings = category["raw_mappings"]
    if type(categories) is not list or len(categories) != 10 or type(mappings) is not list or len(mappings) != 12:
        raise TrainingContractError("Conversion R3 category rows drift")
    expected_names = ("pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor")
    for index, row in enumerate(categories, 1):
        _assert_exact_keys(row, {"id", "name"}, "Conversion R3 category row")
        if type(row["id"]) is not int or row["id"] != index or type(row["name"]) is not str or row["name"] != expected_names[index - 1]:
            raise TrainingContractError("Conversion R3 category row drift")
    for row in mappings:
        _assert_exact_keys(row, {"coco_category_id", "enters_matching", "name", "raw_category_id", "reason_code", "training_category_id"}, "Conversion R3 mapping row")
        if type(row["raw_category_id"]) is not int or type(row["enters_matching"]) is not bool or type(row["name"]) is not str or type(row["reason_code"]) is not str:
            raise TrainingContractError("Conversion R3 mapping type drift")
    return category


def _validate_conversion_source_document(raw: bytes) -> dict[str, Any]:
    source = _parse_strict_json(raw, label="Conversion R3 source identity", trailing_lf=False)
    _assert_exact_keys(source, _CONVERSION_SOURCE_KEYS, "Conversion R3 source identity")
    expected = {
        "schema_version": 1,
        "protocol_id": "P3-VISDRONE-DATA-PROTOCOL-V2",
        "dataset_test_accessed_by_this_process": False,
        "project_test_split_historically_observed": True,
        "train_annotation_count": 353550,
        "train_image_count": 6471,
        "val_annotation_count": 40169,
        "val_image_count": 548,
        "protocol_config_sha256": "f338102200c9b953ec2a047acb84c386b00c5c80da65de39320b3085abd7172b",
        "train_raw_manifest_sha256": "8d59c5a163cf1c49f60957872768df31fcb658fbee8e27edac95701fa9d195e9",
        "val_raw_manifest_sha256": "d28e87e61812159eb914befd181529a915ed4120d2b70f2be8d76700f01583c1",
    }
    for key, value in expected.items():
        if type(source[key]) is not type(value) or source[key] != value:
            raise TrainingContractError(f"Conversion R3 source identity {key} drift")
    _assert_exact_keys(source["audited_protocol_data_identity"], _CONVERSION_DATA_IDENTITY_KEYS, "Conversion R3 audited identity")
    return source


def _validate_conversion_config_document(
    raw: bytes,
    category: dict[str, Any],
    source_identity: dict[str, Any],
) -> dict[str, Any]:
    config = _parse_strict_json(raw, label="Conversion R3 config", trailing_lf=False)
    _assert_exact_keys(config, _CONVERSION_CONFIG_KEYS, "Conversion R3 config")
    fixed = {
        "schema_version": 1,
        "converter_schema_version": 1,
        "protocol_id": "P3-VISDRONE-DATA-PROTOCOL-V2",
        "confirmatory_metrics_accessed": False,
        "production_split_manifest_generated": True,
        "real_conversion_outputs_generated": True,
        "test_access_allowed": False,
    }
    for key, expected in fixed.items():
        if type(config[key]) is not type(expected) or config[key] != expected:
            raise TrainingContractError(f"Conversion R3 config {key} drift")
    data_identity = config["data_identity"]
    _assert_exact_keys(data_identity, _CONVERSION_DATA_IDENTITY_KEYS, "Conversion R3 config data identity")
    if data_identity != source_identity["audited_protocol_data_identity"]:
        raise TrainingContractError("Conversion R3 config/source data identity drift")
    categories = config["categories"]
    if type(categories) is not dict or set(categories) != {"background_explicit", "category_0", "category_11", "num_classes", "raw_to_training", "training_to_coco"}:
        raise TrainingContractError("Conversion R3 config category schema drift")
    if categories["num_classes"] != category["num_classes"] or categories["background_explicit"] is not category["background_explicit"]:
        raise TrainingContractError("Conversion R3 config/category binding drift")
    split = config["split"]
    _assert_exact_keys(split, _CONVERSION_SPLIT_KEYS, "Conversion R3 config split")
    expected_split = {
        "confirmatory_source": "official_train_only",
        "development": "official_val",
        "distribution_tolerance_percentage_points": 5,
        "evaluated_prefix_count": 184,
        "feasible_prefix_count": 64,
        "group_unit": "sequence_key_connected_by_duplicate_image_sha",
        "metrics_access_allowed": False,
        "planner": "plan_confirmatory_split_v2",
        "salt": "P3-confirmatory-v1",
        "seed": 20260808,
        "selection_allowed": False,
        "selection_policy": "feasibility_first_nearest_hash_prefix_v2",
        "single_final_access_only": True,
        "target_confirmatory_images": 647,
        "test": "disabled",
    }
    if split != expected_split:
        raise TrainingContractError("Conversion R3 config split drift")
    return config


def _validate_conversion_r3_runtime(
    repository: _VerifiedRepository,
    config: dict[str, Any],
) -> dict[str, Any]:
    source = config["source_bindings"]["conversion_r3"]
    observed = repository.observe_flat_directory(
        source["artifact_root"],
        capture_names=CONVERSION_R3_AUTHORITY_FILES,
        expected_mode=CONVERSION_R3_FILE_MODE,
    )
    if (
        observed["file_count"] != CONVERSION_R3_FILE_COUNT
        or observed["directory_count_excluding_root"] != CONVERSION_R3_DIRECTORY_COUNT
        or observed["total_size_bytes"] != CONVERSION_R3_TOTAL_SIZE_BYTES
    ):
        raise TrainingContractError("Conversion R3 observed count/size binding drift")
    rows = observed["rows"]
    if set(row["relative_path"] for row in rows) != _CONVERSION_FLAT_FILE_NAMES:
        raise TrainingContractError("Conversion R3 observed file set drift")
    raw = observed["captured"]
    inventory = _validate_conversion_inventory_document(raw["artifact_inventory.json"], rows, config)
    source_identity = _validate_conversion_source_document(raw["source_identity.json"])
    category = _validate_conversion_category_document(raw["category_contract.json"])
    conversion_config = _validate_conversion_config_document(raw["config.json"], category, source_identity)
    completion = _validate_conversion_completion_document(raw["completion.json"], rows, config, source_identity)
    fixed_rows = {
        "completion_sha256": "completion.json",
        "artifact_inventory_sha256": "artifact_inventory.json",
        "config_sha256": "config.json",
        "category_contract_sha256": "category_contract.json",
        "source_identity_sha256": "source_identity.json",
    }
    for field, filename in fixed_rows.items():
        if _conversion_observed_row(rows, filename)["sha256"] != source[field]:
            raise TrainingContractError(f"Conversion R3 {field} binding drift")
    if completion["config_sha256"] != _conversion_observed_row(rows, "config.json")["sha256"]:
        raise TrainingContractError("Conversion R3 completion/config cross-file drift")
    return {
        "relative_path": source["artifact_root"],
        "file_count": observed["file_count"],
        "directory_count_excluding_root": observed["directory_count_excluding_root"],
        "total_size_bytes": observed["total_size_bytes"],
        "completion_sha256": _conversion_observed_row(rows, "completion.json")["sha256"],
        "artifact_inventory_sha256": _conversion_observed_row(rows, "artifact_inventory.json")["sha256"],
        "entry_canonical_inventory_sha256": inventory["canonical_inventory_sha256"],
        "config_sha256": _conversion_observed_row(rows, "config.json")["sha256"],
        "category_contract_sha256": _conversion_observed_row(rows, "category_contract.json")["sha256"],
        "source_identity_sha256": _conversion_observed_row(rows, "source_identity.json")["sha256"],
    }


def _load_training_contract_from_boundary(
    repository: _VerifiedRepository, config_path: str | Path
) -> tuple[dict[str, Any], bytes]:
    relative = _assert_relative_path(str(config_path), "config_path")
    if relative != TRAINING_CONFIG_RELATIVE_PATH:
        raise TrainingContractError("training config path identity drift")
    training_raw = repository.read_file(relative)
    baseline_raw = repository.read_file(BASELINE_CONFIG_RELATIVE_PATH)
    baseline = _parse_portable_json(baseline_raw, label="baseline config")
    if len(baseline_raw) != BASELINE_CONFIG_SIZE_BYTES or _sha256_bytes(baseline_raw) != BASELINE_CONFIG_RAW_SHA256:
        raise TrainingContractError("baseline raw identity drift")
    if len(training_raw) != TRAINING_CONFIG_SIZE_BYTES or _sha256_bytes(training_raw) != TRAINING_CONFIG_RAW_SHA256:
        raise TrainingContractError("training config raw identity drift")
    config = _parse_portable_json(training_raw, label="training config")
    return validate_training_contract(config, baseline), training_raw


def load_training_contract(repo_root: str | Path, config_path: str | Path = TRAINING_CONFIG_RELATIVE_PATH) -> dict[str, Any]:
    """Load and validate the checked-in contract without runtime side effects."""

    with _VerifiedRepository(repo_root) as repository:
        config, _ = _load_training_contract_from_boundary(repository, config_path)
        return config


def _validate_vendor_source_files(repository: _VerifiedRepository, config: dict[str, Any]) -> None:
    bindings = [config["source_bindings"]["vendor_recipe"]]
    bindings.extend(config["source_bindings"]["vendor_includes"])
    for binding in bindings:
        relative = _assert_relative_path(binding["relative_path"], "vendor source path")
        if _sha256_bytes(repository.read_file(relative)) != binding["sha256"]:
            raise TrainingContractError(f"vendor source identity drift: {relative}")


_PRIMARY_EVALUATOR_CONFIG_KEYS = {
    "schema_version",
    "evaluator_id",
    "protocol_id",
    "schema_id",
    "authority",
    "input_contract",
    "algorithm",
    "output_contract",
    "policy",
}
_PRIMARY_EVALUATOR_CONFIG_AUTHORITY_KEYS = {
    "repository_url",
    "branch",
    "commit_oid",
    "tree_oid",
    "manifest_relative_path",
    "manifest_raw_size_bytes",
    "manifest_raw_sha256",
    "manifest_canonical_size_bytes",
    "manifest_canonical_sha256",
    "inventory_canonical_sha256",
    "authority_file_sha256",
}
_PRIMARY_EVALUATOR_CONFIG_POLICY_KEYS = {
    "development_only",
    "confirmatory_access_allowed",
    "test_access_allowed",
    "implementation_present",
    "independent_audit_pass",
    "training_gate_open",
    "secondary_evaluator_can_certify",
    "secondary_evaluator_can_select",
}


def _validate_primary_evaluator_config_document(
    raw: bytes,
    certification: dict[str, Any],
) -> dict[str, Any]:
    document = _parse_portable_json(raw, label="primary evaluator config")
    _assert_exact_keys(document, _PRIMARY_EVALUATOR_CONFIG_KEYS, "primary evaluator config")
    evaluator = certification["evaluator"]
    if (
        document["evaluator_id"] != evaluator["evaluator_id"]
        or document["protocol_id"] != evaluator["protocol_id"]
        or document["evaluator_id"] != PRIMARY_EVALUATOR_ID
        or document["protocol_id"] != PRIMARY_EVALUATOR_PROTOCOL_ID
        or document["schema_id"] != "primary_evaluator_input_v2"
    ):
        raise TrainingContractError("primary evaluator config evaluator/protocol identity drift")
    authority = _assert_exact_keys(document["authority"], _PRIMARY_EVALUATOR_CONFIG_AUTHORITY_KEYS, "primary evaluator config authority")
    if authority["repository_url"] != "https://github.com/VisDrone/VisDrone2018-DET-toolkit.git" or authority["branch"] != "master":
        raise TrainingContractError("primary evaluator config authority repository drift")
    _assert_git_oid(authority["commit_oid"], "primary evaluator config authority.commit_oid")
    _assert_git_oid(authority["tree_oid"], "primary evaluator config authority.tree_oid")
    if authority["authority_file_sha256"] != PRIMARY_EVALUATOR_AUTHORITY_FILE_SHA256:
        raise TrainingContractError("primary evaluator config authority file identity drift")
    manifest = certification["authority_manifest"]
    if (
        authority["commit_oid"] != "005445782213e20cb91bc50a597db3dd949e749a"
        or authority["tree_oid"] != "038b9e68c6e9a93a64662a4d7a39be2cd2c0654e"
        or authority["manifest_relative_path"] != manifest["relative_path"]
        or authority["manifest_raw_size_bytes"] != manifest["raw_size_bytes"]
        or authority["manifest_raw_sha256"] != manifest["raw_sha256"]
        or authority["manifest_canonical_size_bytes"] != manifest["canonical_size_bytes"]
        or authority["manifest_canonical_sha256"] != manifest["canonical_sha256"]
        or authority["inventory_canonical_sha256"] != certification["authority_archive_inventory"]["inventory_sha256"]
    ):
        raise TrainingContractError("primary evaluator config authority manifest declaration drift")
    policy = _assert_exact_keys(document["policy"], _PRIMARY_EVALUATOR_CONFIG_POLICY_KEYS, "primary evaluator config policy")
    expected_policy = {
        "development_only": True,
        "confirmatory_access_allowed": False,
        "test_access_allowed": False,
        "implementation_present": True,
        "independent_audit_pass": False,
        "training_gate_open": False,
        "secondary_evaluator_can_certify": False,
        "secondary_evaluator_can_select": False,
    }
    if policy != expected_policy:
        raise TrainingContractError("primary evaluator config historical policy drift")
    return document


_PRIMARY_EVALUATOR_MANIFEST_KEYS = {
    "schema_version",
    "authority_id",
    "repository_url",
    "branch",
    "commit_oid",
    "tree_oid",
    "toolkit_version",
    "algorithm_semantics_version",
    "license_and_use_notice",
    "archive",
    "inventory",
}
_PRIMARY_EVALUATOR_MANIFEST_ARCHIVE_KEYS = {"filename", "prefix", "size_bytes", "sha256"}
_PRIMARY_EVALUATOR_MANIFEST_INVENTORY_KEYS = {
    "file_count",
    "total_size_bytes",
    "canonical_inventory_sha256",
    "canonicalization",
    "rows",
}


def _validate_primary_evaluator_manifest_document(
    raw: bytes,
    certification: dict[str, Any],
) -> dict[str, Any]:
    document = _parse_portable_json(raw, label="primary evaluator authority manifest")
    _assert_exact_keys(document, _PRIMARY_EVALUATOR_MANIFEST_KEYS, "primary evaluator authority manifest")
    if (
        document["schema_version"] != 1
        or document["authority_id"] != "visdrone2018_det_toolkit_005445782213e20c"
        or document["repository_url"] != "https://github.com/VisDrone/VisDrone2018-DET-toolkit.git"
        or document["branch"] != "master"
        or document["toolkit_version"] != "1.0.4"
        or document["algorithm_semantics_version"] != "1.0.3"
    ):
        raise TrainingContractError("primary evaluator authority manifest identity drift")
    _assert_git_oid(document["commit_oid"], "primary evaluator authority manifest.commit_oid")
    _assert_git_oid(document["tree_oid"], "primary evaluator authority manifest.tree_oid")
    if document["commit_oid"] != "005445782213e20cb91bc50a597db3dd949e749a" or document["tree_oid"] != "038b9e68c6e9a93a64662a4d7a39be2cd2c0654e":
        raise TrainingContractError("primary evaluator authority manifest Git identity drift")
    archive = _assert_exact_keys(document["archive"], _PRIMARY_EVALUATOR_MANIFEST_ARCHIVE_KEYS, "primary evaluator authority manifest archive")
    archive_identity = certification["authority_archive_inventory"]
    if (
        archive["filename"] != "visdrone_det_toolkit_005445782213e20c.tar"
        or archive["prefix"] != "VisDrone2018-DET-toolkit-005445782213e20c/"
        or archive["size_bytes"] != archive_identity["archive_size_bytes"]
        or archive["sha256"] != archive_identity["archive_sha256"]
    ):
        raise TrainingContractError("primary evaluator authority archive declaration drift")
    inventory = _assert_exact_keys(document["inventory"], _PRIMARY_EVALUATOR_MANIFEST_INVENTORY_KEYS, "primary evaluator authority manifest inventory")
    if (
        inventory["file_count"] != archive_identity["file_count"]
        or inventory["total_size_bytes"] != 22093
        or inventory["canonical_inventory_sha256"] != archive_identity["inventory_sha256"]
        or inventory["canonicalization"] != "UTF-8 JSON of rows with ensure_ascii=true, sort_keys=true, separators=(',',':'), no trailing LF"
        or type(inventory["rows"]) is not list
        or len(inventory["rows"]) != archive_identity["file_count"]
    ):
        raise TrainingContractError("primary evaluator authority inventory declaration drift")
    previous: str | None = None
    for row in inventory["rows"]:
        if type(row) is not dict or set(row) != {"relative_path", "git_mode", "git_blob_oid", "size_bytes", "sha256"}:
            raise TrainingContractError("primary evaluator authority inventory row schema drift")
        relative = _assert_relative_path(row["relative_path"], "primary evaluator authority inventory relative_path")
        if previous is not None and relative <= previous:
            raise TrainingContractError("primary evaluator authority inventory rows are not sorted")
        previous = relative
        _assert_git_oid(row["git_blob_oid"], "primary evaluator authority inventory git_blob_oid")
        _assert_sha(row["sha256"], "primary evaluator authority inventory sha256")
        if type(row["git_mode"]) is not str or type(row["size_bytes"]) is not int or row["size_bytes"] < 0:
            raise TrainingContractError("primary evaluator authority inventory row type drift")
    if sum(row["size_bytes"] for row in inventory["rows"]) != inventory["total_size_bytes"]:
        raise TrainingContractError("primary evaluator authority inventory total drift")
    return document


def _validate_primary_evaluator_runtime(
    repository: _VerifiedRepository,
    config: dict[str, Any],
) -> dict[str, Any]:
    certification = config["source_bindings"]["primary_evaluator_certification"]
    config_identity = certification["config"]
    config_raw = repository.read_file(config_identity["relative_path"])
    if len(config_raw) != config_identity["raw_size_bytes"] or _sha256_bytes(config_raw) != config_identity["raw_sha256"]:
        raise TrainingContractError("primary evaluator config raw identity drift")
    config_document = _validate_primary_evaluator_config_document(config_raw, certification)
    config_canonical = _canonical_json_bytes(config_document)
    if len(config_canonical) != config_identity["canonical_size_bytes"] or _sha256_bytes(config_canonical) != config_identity["canonical_sha256"]:
        raise TrainingContractError("primary evaluator config canonical identity drift")

    manifest_identity = certification["authority_manifest"]
    manifest_raw = repository.read_file(manifest_identity["relative_path"])
    if len(manifest_raw) != manifest_identity["raw_size_bytes"] or _sha256_bytes(manifest_raw) != manifest_identity["raw_sha256"]:
        raise TrainingContractError("primary evaluator authority manifest raw identity drift")
    manifest_document = _validate_primary_evaluator_manifest_document(manifest_raw, certification)
    manifest_rows = {
        row["relative_path"]: row["sha256"]
        for row in manifest_document["inventory"]["rows"]
    }
    authority_file_sha256 = config_document["authority"]["authority_file_sha256"]
    if {
        relative: manifest_rows.get(relative)
        for relative in authority_file_sha256
    } != authority_file_sha256:
        raise TrainingContractError("primary evaluator config/manifest authority file drift")
    manifest_canonical = _canonical_json_bytes(manifest_document)
    if len(manifest_canonical) != manifest_identity["canonical_size_bytes"] or _sha256_bytes(manifest_canonical) != manifest_identity["canonical_sha256"]:
        raise TrainingContractError("primary evaluator authority manifest canonical identity drift")

    source_files: list[dict[str, str]] = []
    for source in certification["source_files"]:
        raw = repository.read_file(source["relative_path"])
        if _sha256_bytes(raw) != source["sha256"]:
            raise TrainingContractError(f"primary evaluator source identity drift: {source['relative_path']}")
        source_files.append(copy.deepcopy(source))

    return {
        "evaluator": copy.deepcopy(certification["evaluator"]),
        "implementation": copy.deepcopy(certification["implementation"]),
        "config": {
            "relative_path": config_identity["relative_path"],
            "raw_size_bytes": len(config_raw),
            "raw_sha256": _sha256_bytes(config_raw),
            "canonical_size_bytes": len(config_canonical),
            "canonical_sha256": _sha256_bytes(config_canonical),
        },
        "authority_manifest": {
            "relative_path": manifest_identity["relative_path"],
            "raw_size_bytes": len(manifest_raw),
            "raw_sha256": _sha256_bytes(manifest_raw),
            "canonical_size_bytes": len(manifest_canonical),
            "canonical_sha256": _sha256_bytes(manifest_canonical),
        },
        "authority_archive_inventory": copy.deepcopy(certification["authority_archive_inventory"]),
        "source_files": source_files,
        "independent_audit": copy.deepcopy(certification["independent_audit"]),
    }


def training_contract_binding(repo_root: str | Path, config_path: str | Path = TRAINING_CONFIG_RELATIVE_PATH) -> dict[str, Any]:
    """Return raw/canonical identities after complete validation."""

    with _VerifiedRepository(repo_root) as repository:
        config, raw = _load_training_contract_from_boundary(repository, config_path)
        manifest_raw = repository.read_file(VENDOR_MANIFEST_RELATIVE_PATH)
        manifest = _parse_vendor_manifest(manifest_raw)
        inventory = repository.inventory_directory(VENDOR_RUNTIME_RELATIVE_PATH)
        _validate_vendor_inventory(manifest, manifest_raw, inventory, config)
        primary_evaluator_binding = _validate_primary_evaluator_runtime(repository, config)
        conversion_binding = _validate_conversion_r3_runtime(repository, config)
        _validate_vendor_source_files(repository, config)
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
            "vendor_runtime_binding": {
                "relative_path": VENDOR_RUNTIME_RELATIVE_PATH,
                "file_count": inventory["file_count"],
                "directory_count_excluding_root": inventory["directory_count_excluding_root"],
                "total_size_bytes": inventory["total_size_bytes"],
                "compact_inventory_sha256": inventory["compact_inventory_sha256"],
                "manifest_relative_path": VENDOR_MANIFEST_RELATIVE_PATH,
                "manifest_size_bytes": len(manifest_raw),
                "manifest_raw_sha256": _sha256_bytes(manifest_raw),
                "manifest_inventory_sha256": inventory["manifest_inventory_sha256"],
            },
            "primary_evaluator_runtime_binding": primary_evaluator_binding,
            "conversion_r3_runtime_binding": conversion_binding,
        }
