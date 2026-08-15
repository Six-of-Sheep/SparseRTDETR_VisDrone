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
    elif type(value) is float:
        if not math.isfinite(value):
            raise TrainingContractError(f"{field} must be finite")
    elif type(value) not in {str, int, bool, type(None)}:
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
    if detail is float and not math.isfinite(value):
        raise TrainingContractError(f"closed schema non-finite float at {location}")


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
    topology = config["topology"]
    if topology["train_micro_batch"] * topology["gradient_accumulation_steps"] != topology["effective_train_batch"]:
        raise TrainingContractError("effective train batch arithmetic drift")
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
    _validate_path_and_sha_fields(config)
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
