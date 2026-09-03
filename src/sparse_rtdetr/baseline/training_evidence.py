"""Portable evidence and checkpoint primitives for the formal T1 training contract."""

from __future__ import annotations

import copy
import errno
import hashlib
import json
import math
import os
import stat
from pathlib import Path
from typing import Any, Callable, Mapping

import sparse_rtdetr.baseline.training_contract as _training_contract
import sparse_rtdetr.baseline.training_runtime as _training_runtime


EVIDENCE_CONTRACT_CONFIG_RELATIVE_PATH = "configs/baseline/rtdetrv2_r18_visdrone_training_evidence_v1.json"
EVIDENCE_CONTRACT_ID = "rtdetrv2_r18_visdrone_baseline_training_evidence_v1"
TRAINING_CONTRACT_ID = "rtdetrv2_r18_visdrone_baseline_training_v1"
RUNTIME_PLAN_ID = "rtdetrv2_r18_visdrone_baseline_training_runtime_v1"
EVIDENCE_CONTRACT_SCHEMA_VERSION = 1
TRAINING_CONTRACT_CANONICAL_SHA256 = "a20c71ef90cb4ccc091a717286a1c49be8a1ebffb178156942b77798d3d7f868"
RUNTIME_PLAN_CANONICAL_SHA256 = "3812b04d957e1cd7c3990c8458a540651c95c6bd551c631fc51717f2f7b386ac"

# Filled from the checked-in document after its bytes are frozen.
EVIDENCE_CONTRACT_RAW_SIZE_BYTES = 7139
EVIDENCE_CONTRACT_RAW_SHA256 = "4d6bad4afbede236f1169796aa640f9d82bdcf91d06b799d39c183b20c230c9e"
EVIDENCE_CONTRACT_CANONICAL_SIZE_BYTES = 6069
EVIDENCE_CONTRACT_CANONICAL_SHA256 = "3184707c6cfa115477d007ac0be5eb142a054f78e6309a079ad1e9fab0db9bd6"

EVIDENCE_ROOT_RELATIVE_PATH = "artifacts/training/rtdetrv2_r18_visdrone_training_evidence_v1"
CHECKPOINT_ROOT_RELATIVE_PATH = "artifacts/training/rtdetrv2_r18_visdrone_training_evidence_v1/checkpoints"
PREPARED_NAME = "prepared.json"
EPOCH_RECORDS_NAME = "epoch_records.jsonl"
CHECKPOINT_REFERENCES_NAME = "checkpoint_references.jsonl"
PARTIAL_INVENTORY_NAME = "partial_inventory.json"
ARTIFACT_INVENTORY_NAME = "artifact_inventory.json"
COMPLETION_NAME = "completion.json"
FAILURE_NAME = "failure.json"
CHECKPOINT_EXTENSION = ".ckpt"
FINAL_EXCLUDED = (ARTIFACT_INVENTORY_NAME, COMPLETION_NAME)
PARTIAL_EXCLUDED = (PARTIAL_INVENTORY_NAME, COMPLETION_NAME)
CHECKPOINT_REQUIRED_STATES = (
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
)
CHECKPOINT_ROLES = ("last", "best", "periodic", "final")
CLASSIFIER_STATES = (
    "ABSENT",
    "IN_PROGRESS",
    "SYNTHETIC_TERMINAL_COMPLETE",
    "TERMINAL_COMPLETE",
    "TERMINAL_FAILED",
    "UNKNOWN",
)

__all__ = (
    "load_training_evidence_contract",
    "validate_training_evidence_contract",
    "canonical_training_evidence_contract_bytes",
    "training_evidence_contract_binding",
    "TrainingEvidenceWriter",
    "write_atomic_checkpoint",
    "validate_training_evidence",
    "validate_training_checkpoint",
    "classify_training_evidence",
    "TrainingEvidenceError",
)


class TrainingEvidenceError(_training_contract.TrainingContractError):
    """Raised when training evidence or checkpoint identity is invalid."""


def _error(message: str) -> None:
    raise TrainingEvidenceError(message)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise TrainingEvidenceError("value is not canonical JSON") from exc


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TrainingEvidenceError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise TrainingEvidenceError(f"non-finite JSON constant is forbidden: {value}")


def _parse_json(raw: bytes, label: str, *, trailing_lf: bool) -> dict[str, Any]:
    if type(raw) is not bytes:
        _error(f"{label} must be bytes")
    if b"\x00" in raw or b"\r" in raw or raw.startswith(b"\xef\xbb\xbf"):
        _error(f"{label} has forbidden raw bytes")
    if trailing_lf:
        if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
            _error(f"{label} must have exactly one trailing LF")
    elif raw.endswith(b"\n"):
        _error(f"{label} must not have a trailing LF")
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, TrainingEvidenceError) as exc:
        raise TrainingEvidenceError(f"cannot parse {label}") from exc
    if type(value) is not dict:
        _error(f"{label} must be an object")
    return value


def _assert_builtin_json(value: Any, field: str = "value") -> None:
    if type(value) is dict:
        for key, child in value.items():
            if type(key) is not str:
                _error(f"{field} has a non-string key")
            _assert_builtin_json(child, f"{field}.{key}")
        return
    if type(value) is list:
        for index, child in enumerate(value):
            _assert_builtin_json(child, f"{field}[{index}]")
        return
    if type(value) not in {str, int, float, bool, type(None)}:
        _error(f"{field} has a non-builtin JSON scalar")
    if type(value) is float and not math.isfinite(value):
        _error(f"{field} is non-finite")


def _object(**children: Any) -> tuple[str, dict[str, Any]]:
    return ("object", children)


def _list(*children: Any) -> tuple[str, tuple[Any, ...]]:
    return ("list", children)


def _schema_for(value: Any) -> tuple[str, Any]:
    if type(value) is dict:
        return ("object", {key: _schema_for(child) for key, child in value.items()})
    if type(value) is list:
        return ("list", tuple(_schema_for(child) for child in value))
    return ("scalar", type(value))


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


def _build_frozen_contract() -> dict[str, Any]:
    """Return the detached semantic template for the checked-in contract."""

    return {
        "schema_version": 1,
        "training_evidence_contract_id": EVIDENCE_CONTRACT_ID,
        "training_contract": {
            "id": TRAINING_CONTRACT_ID,
            "relative_path": "configs/baseline/rtdetrv2_r18_visdrone_training_v1.json",
            "canonical_size_bytes": 9117,
            "canonical_sha256": TRAINING_CONTRACT_CANONICAL_SHA256,
            "module_relative_path": "src/sparse_rtdetr/baseline/training_contract.py",
            "module_sha256": "21599fae8312a32ddf833bf2f53557ed5b4d4122f0be167f2071f7f82008deef",
            "primary_evaluator": {
                "evaluator_id": "visdrone_official_primary_evaluator_v1",
                "protocol_id": "visdrone_official_style_v1",
                "implementation_commit": "036cca4d127ddd9e10e3cc7900c3eb759b55f59f",
                "implementation_tree": "fbe931976f6bac7d8e4b3bb319ff1905e99c5444",
                "config_canonical_sha256": "355de90bdb6007ed42ed65b3f653a1b8ea57f2184ab4fd48a9561b692b993998",
                "authority_manifest_canonical_sha256": "5bad9faf7622fe4542aa3b46561d6d41fb2e4ee34f35551ecfd821c9577159e3",
                "authority_inventory_sha256": "35a14a021509b82f1238912e5c77ebb3559f9ee6daa3cc6c92db810b5dce5da0",
            },
        },
        "runtime_plan": {
            "id": RUNTIME_PLAN_ID,
            "relative_path": "configs/baseline/rtdetrv2_r18_visdrone_training_runtime_v1.json",
            "raw_size_bytes": 12957,
            "raw_sha256": "cb6af1abae9351b4a268681587db82ad059745f1d7e198d7d8c829b4cee41aef",
            "canonical_size_bytes": 10888,
            "canonical_sha256": RUNTIME_PLAN_CANONICAL_SHA256,
            "module_relative_path": "src/sparse_rtdetr/baseline/training_runtime.py",
            "module_sha256": "735d75f612f59e6cec22dc53002980471c34c8b18601cd58704440f85741c7c9",
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
            "primary_evaluator": {
                "evaluator_id": "visdrone_official_primary_evaluator_v1",
                "protocol_id": "visdrone_official_style_v1",
            },
        },
        "execution_policy": {
            "formal_run_count": 1,
            "resume": False,
            "retry": False,
            "overwrite": False,
            "interrupted_status": "PERMANENT_FAIL",
            "synthetic_mode_allowed_for_t5b": True,
            "synthetic_mode_can_certify_real_training": False,
            "real_success_requires_epochs": 120,
            "real_success_requires_exit_code": 0,
            "model_selection_enabled": False,
            "confirmatory_access": False,
            "test_access": False,
            "speed_measurement": False,
        },
        "roots": {
            "evidence_root": EVIDENCE_ROOT_RELATIVE_PATH,
            "checkpoint_root": CHECKPOINT_ROOT_RELATIVE_PATH,
            "roots_are_future_targets": True,
            "t5b_must_not_create_production_roots": True,
        },
        "files": {
            "prepared": PREPARED_NAME,
            "epoch_records": EPOCH_RECORDS_NAME,
            "checkpoint_references": CHECKPOINT_REFERENCES_NAME,
            "partial_inventory": PARTIAL_INVENTORY_NAME,
            "artifact_inventory": ARTIFACT_INVENTORY_NAME,
            "completion": COMPLETION_NAME,
            "failure": FAILURE_NAME,
            "checkpoint_extension": CHECKPOINT_EXTENSION,
            "final_excluded_from_inventory": list(FINAL_EXCLUDED),
            "partial_excluded_from_inventory": list(PARTIAL_EXCLUDED),
        },
        "terminal_policy": {
            "classifier_states": list(CLASSIFIER_STATES),
            "terminal_statuses": ["SYNTHETIC_TERMINAL_COMPLETE", "TERMINAL_COMPLETE", "PERMANENT_FAIL"],
            "success_mode": "real",
            "synthetic_success_mode": "synthetic",
            "permanent_failure_is_immutable": True,
            "terminalization_exactly_once": True,
        },
        "checkpoint_policy": {
            "roles": list(CHECKPOINT_ROLES),
            "required_state_order": list(CHECKPOINT_REQUIRED_STATES),
            "deterministic_name_pattern": "checkpoint-{role}-{epoch:04d}.ckpt",
            "raw_model_and_ema_must_be_separate": True,
            "atomicity": {
                "exclusive_ownership": True,
                "parent_must_exist": True,
                "parent_creation_forbidden": True,
                "same_device_temporary": True,
                "complete_write_required": True,
                "file_fsync_required": True,
                "atomic_publication_required": True,
                "directory_fsync_required": True,
                "readback_required": True,
                "size_sha256_required": True,
                "loadability_probe_required": True,
                "inventory_binding_required": True,
                "partial_failure_evidence_required": True,
            },
        },
        "evidence_categories": [
            "run_identity",
            "runtime_plan_binding",
            "data_identity",
            "source_identity",
            "environment_identity",
            "epoch_and_step_counters",
            "finite_loss_and_gradient_observations",
            "amp_scale_skip_overflow_observations",
            "ema_update_observations",
            "development_evaluator_input_result_binding",
            "checkpoint_binding",
            "terminal_exit_and_completion",
            "permanent_failure",
        ],
        "creation_order": [
            "evidence_root",
            "prepared.json",
            "epoch_records.jsonl",
            "checkpoint_references.jsonl",
            "checkpoint-{role}-{epoch:04d}.ckpt",
            "partial_inventory.json",
            "artifact_inventory.json",
            "completion.json",
            "failure.json",
        ],
        "readiness": {
            "runtime_observation_certified": False,
            "training_implementation_ready": False,
            "training_ready": False,
            "model_selection_certified": False,
            "confirmatory_metrics_accessed": False,
            "test_access": False,
            "speed_measurement": False,
            "production_training_authorized": False,
        },
    }


_FROZEN_CONTRACT = _build_frozen_contract()
_SCHEMA = _schema_for(_FROZEN_CONTRACT)
_FROZEN_LEAVES = dict(_walk_leaves(_FROZEN_CONTRACT))
_NUMERIC_CONSTRAINTS = {
    pointer: {"expected_type": type(value), "expected": value, "finite": type(value) is float}
    for pointer, value in _FROZEN_LEAVES.items()
    if type(value) in {int, float}
}
_PATH_SHA_LEAVES = {
    pointer: value
    for pointer, value in _FROZEN_LEAVES.items()
    if pointer.rsplit("/", 1)[-1] in {"relative_path", "module_relative_path", "evidence_root", "checkpoint_root"}
    or pointer.rsplit("/", 1)[-1].endswith("_sha256")
}
_SEMANTIC_LEAVES = {
    pointer: value
    for pointer, value in _FROZEN_LEAVES.items()
    if pointer not in _NUMERIC_CONSTRAINTS and pointer not in _PATH_SHA_LEAVES
}
_SCALAR_OWNER_REGISTRY = {
    "numeric": frozenset(_NUMERIC_CONSTRAINTS),
    "path_sha": frozenset(_PATH_SHA_LEAVES),
    "semantic": frozenset(_SEMANTIC_LEAVES),
    "external_runtime_observation": frozenset(),
}


def _validate_schema(value: Any, schema: tuple[str, Any], pointer: str = "") -> None:
    kind, detail = schema
    location = pointer or "/"
    if kind == "object":
        if type(value) is not dict:
            _error(f"closed evidence schema object type mismatch at {location}")
        actual = set(value)
        expected = set(detail)
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing and extra:
            _error(f"closed evidence schema renamed keys at {location}: missing={missing} extra={extra}")
        if missing:
            _error(f"closed evidence schema missing keys at {location}: {missing}")
        if extra:
            _error(f"closed evidence schema extra keys at {location}: {extra}")
        for key, child_schema in detail.items():
            _validate_schema(value[key], child_schema, _pointer(pointer, key))
        return
    if kind == "list":
        if type(value) is not list:
            _error(f"closed evidence schema list type mismatch at {location}")
        if len(value) != len(detail):
            _error(f"closed evidence schema list length mismatch at {location}")
        for index, child_schema in enumerate(detail):
            _validate_schema(value[index], child_schema, _pointer(pointer, index))
        return
    if type(value) is not detail:
        _error(f"closed evidence schema scalar type mismatch at {location}")


def _validate_owner_registry() -> None:
    leaves = set(_FROZEN_LEAVES)
    owners = list(_SCALAR_OWNER_REGISTRY.values())
    union: set[str] = set()
    for owner in owners:
        if union & set(owner):
            _error("evidence scalar owner registry overlaps")
        union.update(owner)
    if union != leaves:
        _error(
            "evidence scalar owner registry is incomplete: "
            f"missing={sorted(leaves - union)} extra={sorted(union - leaves)}"
        )


def _assert_relative_path(value: Any, field: str) -> None:
    if type(value) is not str or not value or "\x00" in value or "\\" in value:
        _error(f"{field} is not a portable relative path")
    if value.startswith("/") or "//" in value:
        _error(f"{field} is not a portable relative path")
    parts = value.split("/")
    if any(not part or part in {".", ".."} for part in parts):
        _error(f"{field} is not a portable relative path")


def _assert_sha(value: Any, field: str) -> None:
    if type(value) is not str or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        _error(f"{field} is not a lowercase SHA-256")


def _validate_path_sha_fields(contract: dict[str, Any]) -> None:
    for pointer, value in _PATH_SHA_LEAVES.items():
        key = pointer.rsplit("/", 1)[-1]
        if key.endswith("_sha256"):
            _assert_sha(_value_at_pointer(contract, pointer), pointer)
        else:
            _assert_relative_path(_value_at_pointer(contract, pointer), pointer)


def _value_at_pointer(value: Any, pointer: str) -> Any:
    current = value
    for token in pointer.lstrip("/").split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        current = current[int(token)] if type(current) is list else current[token]
    return current


def _validate_numeric_fields(contract: dict[str, Any]) -> None:
    actual = {
        pointer: value
        for pointer, value in _walk_leaves(contract)
        if type(value) in {int, float}
    }
    if set(actual) != set(_NUMERIC_CONSTRAINTS):
        _error("evidence numeric registry coverage drift")
    for pointer, rule in _NUMERIC_CONSTRAINTS.items():
        value = actual[pointer]
        if type(value) is not rule["expected_type"]:
            _error(f"evidence numeric type drift at {pointer}")
        if rule["finite"] and not math.isfinite(value):
            _error(f"evidence numeric non-finite at {pointer}")
        if value != rule["expected"]:
            _error(f"evidence numeric frozen literal drift at {pointer}")


def _validate_semantic_fields(contract: dict[str, Any]) -> None:
    actual = {
        pointer: value
        for pointer, value in _walk_leaves(contract)
        if pointer not in _NUMERIC_CONSTRAINTS and pointer not in _PATH_SHA_LEAVES
    }
    if set(actual) != set(_SEMANTIC_LEAVES):
        _error("evidence semantic registry coverage drift")
    for pointer, expected in _SEMANTIC_LEAVES.items():
        observed = actual[pointer]
        if type(observed) is not type(expected) or observed != expected:
            _error(f"evidence semantic frozen literal drift at {pointer}")


def _validate_contract_relations(contract: dict[str, Any]) -> None:
    if contract["training_contract"]["id"] != TRAINING_CONTRACT_ID:
        _error("training-contract identity drift")
    if contract["runtime_plan"]["id"] != RUNTIME_PLAN_ID:
        _error("runtime-plan identity drift")
    if contract["roots"]["t5b_must_not_create_production_roots"] is not True:
        _error("production root policy drift")
    if contract["execution_policy"]["real_success_requires_epochs"] != 120:
        _error("real completion epoch policy drift")
    if contract["checkpoint_policy"]["required_state_order"] != list(CHECKPOINT_REQUIRED_STATES):
        _error("checkpoint state order drift")
    if contract["checkpoint_policy"]["roles"] != list(CHECKPOINT_ROLES):
        _error("checkpoint role order drift")
    if contract["terminal_policy"]["classifier_states"] != list(CLASSIFIER_STATES):
        _error("classifier state order drift")
    if contract["files"]["final_excluded_from_inventory"] != list(FINAL_EXCLUDED):
        _error("final inventory exclusion drift")
    if contract["files"]["partial_excluded_from_inventory"] != list(PARTIAL_EXCLUDED):
        _error("partial inventory exclusion drift")


def _validate_detached_contract(contract: Any, *, verify_digest: bool = True) -> dict[str, Any]:
    _assert_builtin_json(contract, "training evidence contract")
    _validate_schema(contract, _SCHEMA)
    _validate_owner_registry()
    _validate_numeric_fields(contract)
    _validate_path_sha_fields(contract)
    _validate_semantic_fields(contract)
    _validate_contract_relations(contract)
    canonical = canonical_training_evidence_contract_bytes(contract)
    if verify_digest:
        if (
            len(canonical) != EVIDENCE_CONTRACT_CANONICAL_SIZE_BYTES
            or _sha256_bytes(canonical) != EVIDENCE_CONTRACT_CANONICAL_SHA256
        ):
            _error("training evidence contract canonical identity drift")
    return copy.deepcopy(contract)


def canonical_training_evidence_contract_bytes(contract: Any) -> bytes:
    _assert_builtin_json(contract, "training evidence contract")
    _validate_schema(contract, _SCHEMA)
    return _canonical_json_bytes(contract)


def _assert_authority_bindings(
    contract: dict[str, Any],
    training_binding: Any,
    runtime_binding: Any,
) -> None:
    _assert_builtin_json(training_binding, "training contract binding")
    _assert_builtin_json(runtime_binding, "runtime plan binding")
    if type(training_binding) is not dict or type(runtime_binding) is not dict:
        _error("authority bindings must be dictionaries")
    training_expected = contract["training_contract"]
    if (
        training_binding.get("training_contract_id") != training_expected["id"]
        or training_binding.get("canonical_size_bytes") != training_expected["canonical_size_bytes"]
        or training_binding.get("canonical_sha256") != training_expected["canonical_sha256"]
    ):
        _error("T4 training-contract binding drift")
    runtime_expected = contract["runtime_plan"]
    if (
        runtime_binding.get("runtime_plan_id") != runtime_expected["id"]
        or runtime_binding.get("raw_size_bytes") != runtime_expected["raw_size_bytes"]
        or runtime_binding.get("raw_sha256") != runtime_expected["raw_sha256"]
        or runtime_binding.get("canonical_size_bytes") != runtime_expected["canonical_size_bytes"]
        or runtime_binding.get("canonical_sha256") != runtime_expected["canonical_sha256"]
    ):
        _error("T5A runtime-plan binding drift")
    if runtime_binding.get("vendor_runtime_binding") != runtime_expected["vendor_runtime"]:
        _error("vendor runtime binding drift")
    if runtime_binding.get("conversion_r3_runtime_binding") != runtime_expected["conversion_r3"]:
        _error("Conversion R3 binding drift")
    evaluator = runtime_binding.get("primary_evaluator_runtime_binding")
    expected_evaluator = runtime_expected["primary_evaluator"]
    if type(evaluator) is not dict or evaluator.get("evaluator") != expected_evaluator:
        _error("primary evaluator runtime binding drift")


def load_training_evidence_contract(
    repo_root: str | os.PathLike[str],
    config_path: str | os.PathLike[str] = EVIDENCE_CONTRACT_CONFIG_RELATIVE_PATH,
) -> dict[str, Any]:
    relative = _training_contract._assert_relative_path(str(config_path), "evidence contract config path")
    if relative != EVIDENCE_CONTRACT_CONFIG_RELATIVE_PATH:
        _error("evidence contract config path drift")
    with _training_contract._VerifiedRepository(repo_root) as repository:
        raw = repository.read_file(relative)
    if len(raw) != EVIDENCE_CONTRACT_RAW_SIZE_BYTES or _sha256_bytes(raw) != EVIDENCE_CONTRACT_RAW_SHA256:
        _error("training evidence contract raw identity drift")
    return _validate_detached_contract(
        _parse_json(raw, "training evidence contract", trailing_lf=True)
    )


def validate_training_evidence_contract(
    contract: Any,
    training_contract_binding: Any | None = None,
    runtime_plan_binding: Any | None = None,
) -> dict[str, Any]:
    validated = _validate_detached_contract(contract)
    if training_contract_binding is not None or runtime_plan_binding is not None:
        if training_contract_binding is None or runtime_plan_binding is None:
            _error("both authority bindings are required")
        _assert_authority_bindings(validated, training_contract_binding, runtime_plan_binding)
    return copy.deepcopy(validated)


def training_evidence_contract_binding(
    repo_root: str | os.PathLike[str],
    config_path: str | os.PathLike[str] = EVIDENCE_CONTRACT_CONFIG_RELATIVE_PATH,
) -> dict[str, Any]:
    training_binding = _training_contract.training_contract_binding(repo_root)
    runtime_binding = _training_runtime.training_runtime_plan_binding(repo_root)
    contract = load_training_evidence_contract(repo_root, config_path)
    _assert_authority_bindings(contract, training_binding, runtime_binding)
    canonical = canonical_training_evidence_contract_bytes(contract)
    return {
        "schema_version": EVIDENCE_CONTRACT_SCHEMA_VERSION,
        "training_evidence_contract_id": EVIDENCE_CONTRACT_ID,
        "relative_path": EVIDENCE_CONTRACT_CONFIG_RELATIVE_PATH,
        "raw_size_bytes": EVIDENCE_CONTRACT_RAW_SIZE_BYTES,
        "raw_sha256": EVIDENCE_CONTRACT_RAW_SHA256,
        "canonical_size_bytes": len(canonical),
        "canonical_sha256": _sha256_bytes(canonical),
        "training_contract_binding": copy.deepcopy(training_binding),
        "runtime_plan_binding": copy.deepcopy(runtime_binding),
        "contract": copy.deepcopy(contract),
    }


def _absolute_path(value: str | os.PathLike[str], field: str) -> str:
    try:
        path = os.fspath(value)
    except TypeError as exc:
        raise TrainingEvidenceError(f"{field} must be an absolute path") from exc
    if type(path) is not str or not path or "\x00" in path or "\\" in path:
        _error(f"{field} must be an absolute path")
    if not path.startswith("/") or path == "/" or path.endswith("/") or "//" in path:
        _error(f"{field} must be a normalized absolute path")
    if any(part in {"", ".", ".."} for part in path[1:].split("/")):
        _error(f"{field} must be a normalized absolute path")
    return path


def _directory_flags() -> int:
    required = ("O_DIRECTORY", "O_NOFOLLOW", "O_CLOEXEC")
    if any(not hasattr(os, name) for name in required):
        _error("secure directory operations are unavailable")
    return os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC


def _file_flags() -> int:
    required = ("O_NOFOLLOW", "O_CLOEXEC", "O_NONBLOCK")
    if any(not hasattr(os, name) for name in required):
        _error("secure file operations are unavailable")
    return os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK


def _lstat_at(name: str, parent_fd: int) -> os.stat_result:
    while True:
        try:
            return os.lstat(name, dir_fd=parent_fd)
        except InterruptedError:
            continue
        except OSError as exc:
            raise TrainingEvidenceError(f"secure lstat failed: {name}") from exc


def _open_at(name: str, flags: int, parent_fd: int | None = None, mode: int = 0o600) -> int:
    while True:
        try:
            if parent_fd is None:
                return os.open(name, flags, mode)
            return os.open(name, flags, mode, dir_fd=parent_fd)
        except InterruptedError:
            continue
        except OSError as exc:
            raise TrainingEvidenceError(f"secure open failed: {name}") from exc


def _close_fd(fd: int | None) -> None:
    if fd is not None:
        try:
            os.close(fd)
        except OSError:
            pass


def _same_object(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino, stat.S_IFMT(left.st_mode)) == (
        right.st_dev,
        right.st_ino,
        stat.S_IFMT(right.st_mode),
    )


def _snapshot(value: os.stat_result) -> tuple[int, int, int, int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_uid,
        value.st_gid,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _open_existing_directory(path: str | os.PathLike[str], field: str) -> tuple[int, list[int]]:
    lexical = _absolute_path(path, field)
    flags = _directory_flags()
    fds: list[int] = []
    try:
        root_fd = _open_at("/", flags)
        fds.append(root_fd)
        parent_fd = root_fd
        for name in lexical[1:].split("/"):
            observed = _lstat_at(name, parent_fd)
            if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
                _error(f"{field} contains a non-directory or symlink component")
            child_fd = _open_at(name, flags, parent_fd)
            fds.append(child_fd)
            opened = os.fstat(child_fd)
            if not _same_object(observed, opened):
                _error(f"{field} component identity changed during open")
            parent_fd = child_fd
        final = os.fstat(parent_fd)
        if not stat.S_ISDIR(final.st_mode) or stat.S_ISLNK(final.st_mode):
            _error(f"{field} is not a real directory")
        return parent_fd, fds
    except BaseException:
        for fd in reversed(fds):
            _close_fd(fd)
        raise


def _open_existing_parent(path: str | os.PathLike[str], field: str) -> tuple[int, str, list[int]]:
    lexical = _absolute_path(path, field)
    parent = lexical.rsplit("/", 1)[0] or "/"
    name = lexical.rsplit("/", 1)[1]
    fd, fds = _open_existing_directory(parent, f"{field} parent")
    return fd, name, fds


def _mkdir_child(parent_fd: int, name: str, field: str) -> tuple[int, list[int]]:
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        _error(f"{field} final component is invalid")
    try:
        os.mkdir(name, 0o700, dir_fd=parent_fd)
    except FileExistsError as exc:
        raise TrainingEvidenceError(f"{field} already exists") from exc
    except OSError as exc:
        raise TrainingEvidenceError(f"{field} creation failed") from exc
    try:
        os.fsync(parent_fd)
    except OSError as exc:
        raise TrainingEvidenceError(f"{field} parent fsync failed") from exc
    flags = _directory_flags()
    child_fd = _open_at(name, flags, parent_fd)
    observed = _lstat_at(name, parent_fd)
    opened = os.fstat(child_fd)
    if not _same_object(observed, opened):
        _close_fd(child_fd)
        _error(f"{field} identity changed during creation")
    if stat.S_IMODE(opened.st_mode) != 0o700 or opened.st_uid != os.getuid() or opened.st_gid != os.getgid():
        _close_fd(child_fd)
        _error(f"{field} owner or mode is not exclusive")
    return child_fd, [child_fd]


def _read_fd(fd: int, expected: os.stat_result | None = None) -> bytes:
    chunks: list[bytes] = []
    while True:
        try:
            chunk = os.read(fd, 1024 * 1024)
        except InterruptedError:
            continue
        except OSError as exc:
            raise TrainingEvidenceError("secure read failed") from exc
        if not chunk:
            break
        chunks.append(chunk)
    raw = b"".join(chunks)
    observed = os.fstat(fd)
    if expected is not None and _snapshot(observed) != _snapshot(expected):
        _error("file metadata changed during read")
    if expected is not None and len(raw) != expected.st_size:
        _error("file size changed during read")
    return raw


def _read_regular_at(parent_fd: int, name: str, relative: str) -> tuple[bytes, os.stat_result]:
    observed = _lstat_at(name, parent_fd)
    if (
        stat.S_ISLNK(observed.st_mode)
        or not stat.S_ISREG(observed.st_mode)
        or observed.st_nlink != 1
        or observed.st_uid != os.getuid()
        or observed.st_gid != os.getgid()
        or stat.S_IMODE(observed.st_mode) != 0o600
    ):
        _error(f"unstable evidence file: {relative}")
    fd = _open_at(name, _file_flags(), parent_fd)
    try:
        opened = os.fstat(fd)
        if _snapshot(opened) != _snapshot(observed):
            _error(f"file identity changed during open: {relative}")
        raw = _read_fd(fd, opened)
        after = os.fstat(fd)
        if _snapshot(after) != _snapshot(opened):
            _error(f"file identity changed after read: {relative}")
        current = _lstat_at(name, parent_fd)
        if _snapshot(current) != _snapshot(opened):
            _error(f"file path was replaced during read: {relative}")
        return raw, opened
    finally:
        _close_fd(fd)


def _write_all(fd: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        try:
            written = os.write(fd, payload[offset:])
        except InterruptedError:
            continue
        except OSError as exc:
            raise TrainingEvidenceError("complete write failed") from exc
        if type(written) is not int or written <= 0:
            _error("short or empty write")
        offset += written


def _temporary_name(name: str) -> str:
    token = f".{name}.{os.getpid()}.{id(name)}.tmp"
    if len(token) > 240:
        _error("temporary filename is too long")
    return token


def _atomic_publish_at(parent_fd: int, name: str, payload: bytes, field: str) -> tuple[int, str]:
    if type(name) is not str or not name or "/" in name or "\\" in name or name in {".", ".."}:
        _error(f"{field} filename is invalid")
    try:
        existing = _lstat_at(name, parent_fd)
    except TrainingEvidenceError as exc:
        if not isinstance(exc.__cause__, FileNotFoundError) and getattr(exc.__cause__, "errno", None) != errno.ENOENT:
            raise
        existing = None
    if existing is not None:
        _error(f"{field} already exists")
    temp_name: str | None = _temporary_name(name)
    temp_fd: int | None = None
    try:
        temp_fd = _open_at(
            temp_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | _file_flags(),
            parent_fd,
            0o600,
        )
        created = os.fstat(temp_fd)
        if (
            not stat.S_ISREG(created.st_mode)
            or created.st_nlink != 1
            or created.st_dev != os.fstat(parent_fd).st_dev
            or created.st_uid != os.getuid()
            or created.st_gid != os.getgid()
            or stat.S_IMODE(created.st_mode) != 0o600
        ):
            _error(f"{field} temporary object identity is invalid")
        _write_all(temp_fd, payload)
        try:
            os.fsync(temp_fd)
        except OSError as exc:
            raise TrainingEvidenceError(f"{field} temporary file fsync failed") from exc
        after_write = os.fstat(temp_fd)
        if (
            after_write.st_size != len(payload)
            or after_write.st_nlink != 1
            or not _same_object(created, after_write)
            or after_write.st_uid != os.getuid()
            or after_write.st_gid != os.getgid()
            or stat.S_IMODE(after_write.st_mode) != 0o600
        ):
            _error(f"{field} temporary object is incomplete")
        _close_fd(temp_fd)
        temp_fd = None
        try:
            os.link(temp_name, name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd, follow_symlinks=False)
        except FileExistsError as exc:
            raise TrainingEvidenceError(f"{field} publication target appeared") from exc
        except OSError as exc:
            raise TrainingEvidenceError(f"{field} atomic publication failed") from exc
        try:
            os.unlink(temp_name, dir_fd=parent_fd)
        except OSError as exc:
            raise TrainingEvidenceError(f"{field} temporary cleanup failed") from exc
        temp_name = None
        try:
            os.fsync(parent_fd)
        except OSError as exc:
            raise TrainingEvidenceError(f"{field} directory fsync failed") from exc
        readback, readback_stat = _read_regular_at(parent_fd, name, field)
        if readback != payload or readback_stat.st_size != len(payload) or _sha256_bytes(readback) != _sha256_bytes(payload):
            _error(f"{field} readback identity mismatch")
        return len(payload), _sha256_bytes(payload)
    finally:
        _close_fd(temp_fd)
        if temp_name is not None:
            try:
                leftover = _lstat_at(temp_name, parent_fd)
            except TrainingEvidenceError as exc:
                if getattr(exc.__cause__, "errno", None) == errno.ENOENT:
                    leftover = None
                else:
                    raise
            if leftover is not None:
                if (
                    stat.S_ISLNK(leftover.st_mode)
                    or not stat.S_ISREG(leftover.st_mode)
                    or leftover.st_nlink != 1
                ):
                    _error(f"{field} temporary cleanup object is unsafe")
                try:
                    os.unlink(temp_name, dir_fd=parent_fd)
                except OSError as exc:
                    raise TrainingEvidenceError(f"{field} temporary cleanup failed") from exc


def _append_line_at(parent_fd: int, name: str, payload: bytes, field: str) -> tuple[int, str]:
    if not payload.endswith(b"\n") or b"\x00" in payload or b"\r" in payload:
        _error(f"{field} is not a portable line")
    observed = _lstat_at(name, parent_fd)
    if (
        stat.S_ISLNK(observed.st_mode)
        or not stat.S_ISREG(observed.st_mode)
        or observed.st_nlink != 1
        or observed.st_uid != os.getuid()
        or observed.st_gid != os.getgid()
        or stat.S_IMODE(observed.st_mode) != 0o600
    ):
        _error(f"{field} is not a stable regular file")
    fd = _open_at(name, os.O_WRONLY | os.O_APPEND | _file_flags(), parent_fd)
    try:
        opened = os.fstat(fd)
        if _snapshot(opened) != _snapshot(observed):
            _error(f"{field} identity changed before append")
        _write_all(fd, payload)
        os.fsync(fd)
        after = os.fstat(fd)
        if (
            not _same_object(opened, after)
            or after.st_nlink != 1
            or after.st_uid != os.getuid()
            or after.st_gid != os.getgid()
            or stat.S_IMODE(after.st_mode) != 0o600
            or after.st_size != opened.st_size + len(payload)
        ):
            _error(f"{field} append identity mismatch")
        current = _lstat_at(name, parent_fd)
        if _snapshot(current) != _snapshot(after):
            _error(f"{field} path was replaced during append")
        os.fsync(parent_fd)
        return after.st_size, _sha256_bytes(payload[:-1])
    finally:
        _close_fd(fd)


def _read_json_file(parent_fd: int, name: str, label: str) -> tuple[dict[str, Any], bytes]:
    raw, _ = _read_regular_at(parent_fd, name, label)
    value = _parse_json(raw, label, trailing_lf=False)
    if raw != _canonical_json_bytes(value):
        _error(f"{label} is repacked or not canonical")
    return value, raw


def _json_file(parent_fd: int, name: str, label: str) -> dict[str, Any]:
    value, _ = _read_json_file(parent_fd, name, label)
    return value


def _validate_identity_object(value: Any, field: str) -> dict[str, Any]:
    if type(value) is not dict:
        _error(f"{field} must be an object")
    if set(value) != {"identity_id", "sha256"}:
        _error(f"{field} has an invalid key set")
    if type(value["identity_id"]) is not str or not value["identity_id"]:
        _error(f"{field}.identity_id is invalid")
    _assert_sha(value["sha256"], f"{field}.sha256")
    return copy.deepcopy(value)


def _authority_identity(binding: Any, field: str, *, runtime: bool) -> tuple[str, str, str]:
    if type(binding) is not dict:
        _error(f"{field} must be a binding object")
    if runtime:
        identity = binding.get("runtime_plan_id")
        raw_sha = binding.get("raw_sha256")
        canonical_sha = binding.get("canonical_sha256")
    else:
        identity = binding.get("training_contract_id")
        raw_sha = binding.get("raw_sha256")
        canonical_sha = binding.get("canonical_sha256")
    if type(identity) is not str or not identity:
        _error(f"{field} identity is missing")
    _assert_sha(raw_sha, f"{field}.raw_sha256")
    _assert_sha(canonical_sha, f"{field}.canonical_sha256")
    return identity, raw_sha, canonical_sha


_PREPARED_KEYS = {
    "schema_version",
    "evidence_contract_id",
    "training_contract_id",
    "runtime_plan_id",
    "mode",
    "run_id",
    "nonce",
    "source_identity",
    "data_identity",
    "environment_identity",
    "training_contract_sha256",
    "runtime_plan_sha256",
    "evaluator_id",
    "sequence",
    "predecessor_sha256",
    "status",
}
_COMMON_CONTEXT_KEYS = {
    "schema_version",
    "evidence_contract_id",
    "training_contract_id",
    "runtime_plan_id",
    "mode",
    "run_id",
    "nonce",
    "training_contract_sha256",
    "runtime_plan_sha256",
    "source_identity_sha256",
    "data_identity_sha256",
    "environment_identity_sha256",
    "evaluator_id",
}
_EPOCH_KEYS = {
    "schema_version",
    "evidence_contract_id",
    "training_contract_id",
    "runtime_plan_id",
    "mode",
    "run_id",
    "nonce",
    "training_contract_sha256",
    "runtime_plan_sha256",
    "sequence",
    "epoch",
    "global_optimizer_step",
    "predecessor_sha256",
    "source_identity_sha256",
    "data_identity_sha256",
    "environment_identity_sha256",
    "loss",
    "gradient_norm",
    "amp_scale",
    "amp_skipped_steps",
    "amp_overflow_events",
    "nonfinite_loss_count",
    "nonfinite_gradient_count",
    "ema_updates",
    "evaluator_id",
    "evaluator_result_sha256",
    "checkpoint_sha256",
}
_CHECKPOINT_REFERENCE_KEYS = {
    "schema_version",
    "evidence_contract_id",
    "training_contract_id",
    "runtime_plan_id",
    "mode",
    "run_id",
    "nonce",
    "training_contract_sha256",
    "runtime_plan_sha256",
    "source_identity_sha256",
    "data_identity_sha256",
    "environment_identity_sha256",
    "evaluator_id",
    "sequence",
    "epoch",
    "global_optimizer_step",
    "predecessor_sha256",
    "role",
    "relative_path",
    "checkpoint_size_bytes",
    "checkpoint_sha256",
    "state_inventory_sha256",
    "loadability_pass",
}
_INVENTORY_ROW_KEYS = {"relative_path", "size_bytes", "sha256", "mode", "uid", "gid", "nlink"}
_INVENTORY_KEYS = {
    "schema_version",
    "evidence_contract_id",
    "run_id",
    "nonce",
    "excluded_from_inventory",
    "artifacts",
    "canonical_inventory_sha256",
}
_COMPLETION_KEYS = {
    "schema_version",
    "evidence_contract_id",
    "training_contract_id",
    "runtime_plan_id",
    "mode",
    "run_id",
    "nonce",
    "training_contract_sha256",
    "runtime_plan_sha256",
    "source_identity_sha256",
    "data_identity_sha256",
    "environment_identity_sha256",
    "evaluator_id",
    "status",
    "exit_code",
    "epoch_count",
    "final_epoch",
    "global_optimizer_step",
    "predecessor_sha256",
    "artifact_inventory_sha256",
    "checkpoint_count",
}
_FAILURE_KEYS = {
    "schema_version",
    "evidence_contract_id",
    "training_contract_id",
    "runtime_plan_id",
    "mode",
    "run_id",
    "nonce",
    "training_contract_sha256",
    "runtime_plan_sha256",
    "source_identity_sha256",
    "data_identity_sha256",
    "environment_identity_sha256",
    "evaluator_id",
    "status",
    "failure_type",
    "exception_type",
    "message",
    "epoch_count",
    "predecessor_sha256",
    "partial_inventory_sha256",
}
_CHECKPOINT_STATE_KEYS = {
    "name",
    "type",
    "format",
    "size_bytes",
    "sha256",
    "loadability_pass",
    "payload_hex",
}
_CHECKPOINT_KEYS = {
    "schema_version",
    "evidence_contract_id",
    "training_contract_id",
    "runtime_plan_id",
    "mode",
    "run_id",
    "nonce",
    "training_contract_sha256",
    "runtime_plan_sha256",
    "epoch",
    "global_optimizer_step",
    "role",
    "relative_path",
    "required_state_order",
    "states",
    "source_identity",
    "data_identity",
    "environment_identity",
    "evaluator_id",
    "predecessor_evidence_sha256",
    "state_inventory_sha256",
    "publication_status",
}


def _require_keys(value: Any, expected: set[str], field: str) -> dict[str, Any]:
    if type(value) is not dict:
        _error(f"{field} must be an object")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing and extra:
            _error(f"{field} renamed keys: missing={missing} extra={extra}")
        if missing:
            _error(f"{field} missing keys: {missing}")
        _error(f"{field} extra keys: {extra}")
    _assert_builtin_json(value, field)
    return value


def _require_string(value: Any, field: str, *, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value):
        _error(f"{field} must be a string")
    return value


def _require_nonnegative_int(value: Any, field: str) -> int:
    if type(value) is not int or value < 0:
        _error(f"{field} must be a nonnegative builtin int")
    return value


def _require_positive_int(value: Any, field: str) -> int:
    if type(value) is not int or value <= 0:
        _error(f"{field} must be a positive builtin int")
    return value


def _require_finite_float(value: Any, field: str, *, positive: bool = False) -> float:
    if type(value) is not float or not math.isfinite(value) or (positive and value <= 0.0):
        _error(f"{field} must be a finite builtin float")
    return value


def _require_optional_sha(value: Any, field: str) -> None:
    if value is not None:
        _assert_sha(value, field)


def _identity_digest(value: dict[str, Any]) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _validate_prepared(value: Any) -> dict[str, Any]:
    prepared = _require_keys(value, _PREPARED_KEYS, "prepared record")
    if prepared["schema_version"] != EVIDENCE_CONTRACT_SCHEMA_VERSION:
        _error("prepared schema version drift")
    if prepared["evidence_contract_id"] != EVIDENCE_CONTRACT_ID:
        _error("prepared evidence contract identity drift")
    if prepared["training_contract_id"] != TRAINING_CONTRACT_ID or prepared["runtime_plan_id"] != RUNTIME_PLAN_ID:
        _error("prepared authority identity drift")
    if type(prepared["mode"]) is not str or prepared["mode"] not in {"synthetic", "real"}:
        _error("prepared mode drift")
    for field in ("run_id", "nonce"):
        _require_string(prepared[field], f"prepared.{field}")
    for field in ("source_identity", "data_identity", "environment_identity"):
        _validate_identity_object(prepared[field], f"prepared.{field}")
    _assert_sha(prepared["training_contract_sha256"], "prepared.training_contract_sha256")
    _assert_sha(prepared["runtime_plan_sha256"], "prepared.runtime_plan_sha256")
    if prepared["evaluator_id"] != "visdrone_official_primary_evaluator_v1":
        _error("prepared evaluator identity drift")
    if (
        type(prepared["sequence"]) is not int
        or prepared["sequence"] != 0
        or prepared["predecessor_sha256"] is not None
        or prepared["status"] != "PREPARED"
    ):
        _error("prepared sequence or status drift")
    return copy.deepcopy(prepared)


def _validate_common_record(value: dict[str, Any], expected: dict[str, Any], field: str) -> None:
    for key in _COMMON_CONTEXT_KEYS:
        if value[key] != expected[key]:
            _error(f"{field} {key} binding drift")


def _validate_epoch(
    value: Any,
    expected: dict[str, Any],
    *,
    expected_sequence: int,
    expected_predecessor: str,
    previous_epoch: int | None,
    previous_step: int | None,
    previous_ema: int | None,
) -> dict[str, Any]:
    record = _require_keys(value, _EPOCH_KEYS, "epoch record")
    _validate_common_record(record, expected, "epoch record")
    if record["sequence"] != expected_sequence or record["predecessor_sha256"] != expected_predecessor:
        _error("epoch record chain drift")
    epoch = _require_positive_int(record["epoch"], "epoch")
    step = _require_positive_int(record["global_optimizer_step"], "global_optimizer_step")
    if previous_epoch is not None and epoch != previous_epoch + 1:
        _error("epoch sequence is missing, duplicated, or out of order")
    if previous_step is not None and step <= previous_step:
        _error("global optimizer step regressed or duplicated")
    for field in ("loss", "gradient_norm"):
        _require_finite_float(record[field], field)
    _require_finite_float(record["amp_scale"], "amp_scale", positive=True)
    for field in (
        "amp_skipped_steps",
        "amp_overflow_events",
        "nonfinite_loss_count",
        "nonfinite_gradient_count",
        "ema_updates",
    ):
        _require_nonnegative_int(record[field], field)
    if any(record[field] != 0 for field in ("amp_skipped_steps", "amp_overflow_events", "nonfinite_loss_count", "nonfinite_gradient_count")):
        _error("epoch record contains a forbidden AMP or nonfinite event")
    if previous_ema is not None and record["ema_updates"] < previous_ema:
        _error("EMA update counter regressed")
    for field in ("source_identity_sha256", "data_identity_sha256", "environment_identity_sha256", "evaluator_result_sha256"):
        _assert_sha(record[field], f"epoch.{field}")
    _require_string(record["evaluator_id"], "epoch.evaluator_id")
    _require_optional_sha(record["checkpoint_sha256"], "epoch.checkpoint_sha256")
    return copy.deepcopy(record)


def _validate_checkpoint_reference(
    value: Any,
    expected: dict[str, Any],
    *,
    expected_sequence: int,
    expected_predecessor: str,
    previous_epoch: int | None,
    previous_step: int | None,
) -> dict[str, Any]:
    reference = _require_keys(value, _CHECKPOINT_REFERENCE_KEYS, "checkpoint reference")
    _validate_common_record(reference, expected, "checkpoint reference")
    if reference["sequence"] != expected_sequence or reference["predecessor_sha256"] != expected_predecessor:
        _error("checkpoint reference chain drift")
    epoch = _require_positive_int(reference["epoch"], "checkpoint reference epoch")
    step = _require_positive_int(reference["global_optimizer_step"], "checkpoint reference step")
    if previous_epoch is not None and epoch < previous_epoch:
        _error("checkpoint reference epoch regressed")
    if previous_step is not None and step < previous_step:
        _error("checkpoint reference step regressed")
    if type(reference["role"]) is not str or reference["role"] not in CHECKPOINT_ROLES:
        _error("checkpoint reference role drift")
    _assert_relative_path(reference["relative_path"], "checkpoint reference relative_path")
    expected_name = f"checkpoints/checkpoint-{reference['role']}-{reference['epoch']:04d}{CHECKPOINT_EXTENSION}"
    if reference["relative_path"] != expected_name:
        _error("checkpoint reference path is outside the checkpoint root")
    _require_positive_int(reference["checkpoint_size_bytes"], "checkpoint_size_bytes")
    _assert_sha(reference["checkpoint_sha256"], "checkpoint_sha256")
    _assert_sha(reference["state_inventory_sha256"], "state_inventory_sha256")
    if reference["loadability_pass"] is not True:
        _error("checkpoint loadability was not affirmed")
    return copy.deepcopy(reference)


def _validate_inventory(value: Any, expected: dict[str, Any], *, partial: bool) -> dict[str, Any]:
    inventory = _require_keys(value, _INVENTORY_KEYS, "evidence inventory")
    if inventory["schema_version"] != EVIDENCE_CONTRACT_SCHEMA_VERSION or inventory["evidence_contract_id"] != EVIDENCE_CONTRACT_ID:
        _error("evidence inventory contract identity drift")
    if inventory["run_id"] != expected["run_id"] or inventory["nonce"] != expected["nonce"]:
        _error("evidence inventory run identity drift")
    required_excluded = list(PARTIAL_EXCLUDED if partial else FINAL_EXCLUDED)
    if inventory["excluded_from_inventory"] != required_excluded:
        _error("evidence inventory exclusions drift")
    if type(inventory["excluded_from_inventory"]) is not list:
        _error("evidence inventory exclusions are not a list")
    rows = inventory["artifacts"]
    if type(rows) is not list or rows != sorted(rows, key=lambda row: row.get("relative_path", "") if type(row) is dict else ""):
        _error("evidence inventory rows are not canonical")
    previous: str | None = None
    for row in rows:
        _require_keys(row, _INVENTORY_ROW_KEYS, "evidence inventory row")
        _assert_relative_path(row["relative_path"], "evidence inventory relative_path")
        if previous is not None and row["relative_path"] <= previous:
            _error("evidence inventory rows are duplicated or unsorted")
        previous = row["relative_path"]
        _require_nonnegative_int(row["size_bytes"], "evidence inventory size_bytes")
        _assert_sha(row["sha256"], "evidence inventory sha256")
        if (
            type(row["mode"]) is not int
            or row["mode"] != 0o600
            or type(row["uid"]) is not int
            or row["uid"] != os.getuid()
            or type(row["gid"]) is not int
            or row["gid"] != os.getgid()
            or row["nlink"] != 1
        ):
            _error("evidence inventory metadata drift")
    _assert_sha(inventory["canonical_inventory_sha256"], "evidence inventory canonical SHA")
    rows_bytes = _canonical_json_bytes(rows)
    if _sha256_bytes(rows_bytes) != inventory["canonical_inventory_sha256"]:
        _error("evidence inventory canonical rows drift")
    return copy.deepcopy(inventory)


def _validate_completion(value: Any, expected: dict[str, Any], *, epoch_count: int, final_epoch: int, final_step: int, predecessor: str, inventory_sha: str, checkpoint_count: int) -> dict[str, Any]:
    completion = _require_keys(value, _COMPLETION_KEYS, "completion record")
    _validate_common_record(completion, expected, "completion record")
    if completion["status"] not in {"SYNTHETIC_TERMINAL_COMPLETE", "TERMINAL_COMPLETE"}:
        _error("completion status drift")
    if completion["mode"] == "synthetic" and completion["status"] != "SYNTHETIC_TERMINAL_COMPLETE":
        _error("synthetic evidence cannot have a real terminal status")
    if completion["mode"] == "real" and completion["status"] != "TERMINAL_COMPLETE":
        _error("real evidence cannot have a synthetic terminal status")
    if (
        epoch_count <= 0
        or checkpoint_count <= 0
        or final_epoch <= 0
        or final_step <= 0
        or completion["exit_code"] != 0
        or completion["epoch_count"] != epoch_count
        or completion["final_epoch"] != final_epoch
        or completion["global_optimizer_step"] != final_step
        or completion["predecessor_sha256"] != predecessor
        or completion["artifact_inventory_sha256"] != inventory_sha
        or completion["checkpoint_count"] != checkpoint_count
    ):
        _error("completion counters or predecessor binding drift")
    if completion["mode"] == "real" and completion["epoch_count"] != 120:
        _error("real terminal completion requires exactly 120 epochs")
    _require_nonnegative_int(completion["exit_code"], "completion.exit_code")
    return copy.deepcopy(completion)


def _validate_failure(value: Any, expected: dict[str, Any], *, epoch_count: int, predecessor: str, partial_sha: str) -> dict[str, Any]:
    failure = _require_keys(value, _FAILURE_KEYS, "failure record")
    _validate_common_record(failure, expected, "failure record")
    if failure["status"] != "PERMANENT_FAIL" or failure["failure_type"] != "PERMANENT_FAIL":
        _error("failure terminal status drift")
    for field in ("exception_type", "message"):
        _require_string(failure[field], f"failure.{field}", nonempty=False)
    if failure["epoch_count"] != epoch_count or failure["predecessor_sha256"] != predecessor or failure["partial_inventory_sha256"] != partial_sha:
        _error("failure predecessor binding drift")
    _require_nonnegative_int(failure["epoch_count"], "failure.epoch_count")
    return copy.deepcopy(failure)


def _state_inventory(states: list[dict[str, Any]]) -> str:
    rows = [
        {
            "name": state["name"],
            "type": state["type"],
            "format": state["format"],
            "size_bytes": state["size_bytes"],
            "sha256": state["sha256"],
            "loadability_pass": state["loadability_pass"],
        }
        for state in states
    ]
    return _sha256_bytes(_canonical_json_bytes(rows))


def _validate_checkpoint_payload(value: Any, *, expected: dict[str, Any] | None = None) -> dict[str, Any]:
    checkpoint = _require_keys(value, _CHECKPOINT_KEYS, "checkpoint")
    if checkpoint["schema_version"] != EVIDENCE_CONTRACT_SCHEMA_VERSION or checkpoint["evidence_contract_id"] != EVIDENCE_CONTRACT_ID:
        _error("checkpoint contract identity drift")
    if checkpoint["training_contract_id"] != TRAINING_CONTRACT_ID:
        _error("checkpoint training-contract identity drift")
    if checkpoint["runtime_plan_id"] != RUNTIME_PLAN_ID:
        _error("checkpoint runtime-plan identity drift")
    if checkpoint["training_contract_sha256"] != TRAINING_CONTRACT_CANONICAL_SHA256:
        _error("checkpoint training-contract SHA drift")
    if checkpoint["runtime_plan_sha256"] != RUNTIME_PLAN_CANONICAL_SHA256:
        _error("checkpoint runtime-plan SHA drift")
    if expected is not None:
        for field in (
            "training_contract_id",
            "runtime_plan_id",
            "mode",
            "run_id",
            "nonce",
            "epoch",
            "global_optimizer_step",
            "role",
            "relative_path",
            "training_contract_sha256",
            "runtime_plan_sha256",
            "evaluator_id",
            "predecessor_evidence_sha256",
        ):
            if checkpoint[field] != expected[field]:
                _error(f"checkpoint {field} binding drift")
    _require_string(checkpoint["run_id"], "checkpoint.run_id")
    _require_string(checkpoint["nonce"], "checkpoint.nonce")
    _require_positive_int(checkpoint["epoch"], "checkpoint.epoch")
    _require_positive_int(checkpoint["global_optimizer_step"], "checkpoint.global_optimizer_step")
    if (
        type(checkpoint["mode"]) is not str
        or checkpoint["mode"] not in {"synthetic", "real"}
        or type(checkpoint["role"]) is not str
        or checkpoint["role"] not in CHECKPOINT_ROLES
    ):
        _error("checkpoint mode or role drift")
    if checkpoint["evaluator_id"] != "visdrone_official_primary_evaluator_v1":
        _error("checkpoint evaluator identity drift")
    _assert_relative_path(checkpoint["relative_path"], "checkpoint.relative_path")
    expected_name = f"checkpoints/checkpoint-{checkpoint['role']}-{checkpoint['epoch']:04d}{CHECKPOINT_EXTENSION}"
    if checkpoint["relative_path"] != expected_name:
        _error("checkpoint path escapes checkpoint root")
    if checkpoint["required_state_order"] != list(CHECKPOINT_REQUIRED_STATES):
        _error("checkpoint required state order drift")
    states = checkpoint["states"]
    if type(states) is not list or len(states) != len(CHECKPOINT_REQUIRED_STATES):
        _error("checkpoint state count drift")
    names = []
    for state in states:
        row = _require_keys(state, _CHECKPOINT_STATE_KEYS, "checkpoint state")
        names.append(row["name"])
        if row["type"] != "bytes" or row["format"] != "synthetic_bytes" or type(row["payload_hex"]) is not str:
            _error("checkpoint state format drift")
        _assert_sha(row["sha256"], "checkpoint state sha256")
        try:
            payload = bytes.fromhex(row["payload_hex"])
        except (TypeError, ValueError) as exc:
            raise TrainingEvidenceError("checkpoint state payload is not hexadecimal") from exc
        if type(row["name"]) is not str or row["name"] not in CHECKPOINT_REQUIRED_STATES or row["size_bytes"] != len(payload):
            _error("checkpoint state payload drift")
        if _sha256_bytes(payload) != row["sha256"] or row["loadability_pass"] is not True:
            _error("checkpoint state identity or loadability drift")
        _require_nonnegative_int(row["size_bytes"], "checkpoint state size_bytes")
    if names != list(CHECKPOINT_REQUIRED_STATES) or len(set(names)) != len(names):
        _error("checkpoint state order, duplicate, or missing-state drift")
    for field in ("source_identity", "data_identity", "environment_identity"):
        _validate_identity_object(checkpoint[field], f"checkpoint.{field}")
    _assert_sha(checkpoint["training_contract_sha256"], "checkpoint.training_contract_sha256")
    _assert_sha(checkpoint["runtime_plan_sha256"], "checkpoint.runtime_plan_sha256")
    _assert_sha(checkpoint["predecessor_evidence_sha256"], "checkpoint.predecessor_evidence_sha256")
    _assert_sha(checkpoint["state_inventory_sha256"], "checkpoint.state_inventory_sha256")
    if checkpoint["state_inventory_sha256"] != _state_inventory(states):
        _error("checkpoint state inventory drift")
    if checkpoint["publication_status"] != "PUBLISHED":
        _error("checkpoint publication is not complete")
    return copy.deepcopy(checkpoint)


def _record_hash(value: dict[str, Any]) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _line_records(raw: bytes, field: str) -> list[dict[str, Any]]:
    if type(raw) is not bytes or b"\x00" in raw or b"\r" in raw:
        _error(f"{field} contains forbidden bytes")
    if raw and not raw.endswith(b"\n"):
        _error(f"{field} has a partial final line")
    records: list[dict[str, Any]] = []
    for line in raw.splitlines():
        if not line:
            _error(f"{field} contains an empty record")
        value = _parse_json(line, field, trailing_lf=False)
        if line != _canonical_json_bytes(value):
            _error(f"{field} contains a repacked record")
        records.append(value)
    return records


def _inventory_root(
    root_fd: int,
    *,
    excluded: tuple[str, ...],
    expected: dict[str, Any],
    extra_excluded: tuple[str, ...] = (),
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    skipped = set(excluded) | set(extra_excluded)

    def walk(directory_fd: int, prefix: str) -> None:
        for name in sorted(os.listdir(directory_fd)):
            relative = f"{prefix}/{name}" if prefix else name
            if not prefix and name in skipped:
                continue
            observed = _lstat_at(name, directory_fd)
            if stat.S_ISLNK(observed.st_mode):
                _error(f"unexpected symlink in evidence inventory: {relative}")
            if stat.S_ISDIR(observed.st_mode):
                if (
                    observed.st_uid != os.getuid()
                    or observed.st_gid != os.getgid()
                    or stat.S_IMODE(observed.st_mode) != 0o700
                ):
                    _error(f"evidence directory metadata drift: {relative}")
                child_fd = _open_at(name, _directory_flags(), directory_fd)
                try:
                    opened = os.fstat(child_fd)
                    if not _same_object(observed, opened):
                        _error(f"evidence directory identity changed: {relative}")
                    walk(child_fd, relative)
                finally:
                    _close_fd(child_fd)
                continue
            if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
                _error(f"unexpected evidence object: {relative}")
            raw, opened = _read_regular_at(directory_fd, name, f"evidence/{relative}")
            rows.append({
                "relative_path": relative,
                "size_bytes": len(raw),
                "sha256": _sha256_bytes(raw),
                "mode": stat.S_IMODE(opened.st_mode),
                "uid": opened.st_uid,
                "gid": opened.st_gid,
                "nlink": opened.st_nlink,
            })

    walk(root_fd, "")
    rows.sort(key=lambda row: row["relative_path"])
    value = {
        "schema_version": EVIDENCE_CONTRACT_SCHEMA_VERSION,
        "evidence_contract_id": EVIDENCE_CONTRACT_ID,
        "run_id": expected["run_id"],
        "nonce": expected["nonce"],
        "excluded_from_inventory": list(excluded),
        "artifacts": rows,
        "canonical_inventory_sha256": _sha256_bytes(_canonical_json_bytes(rows)),
    }
    return value


def _checkpoint_metadata(metadata: Any, explicit: dict[str, Any]) -> dict[str, Any]:
    if metadata is None:
        metadata = {}
    if type(metadata) is not dict:
        _error("checkpoint metadata must be a dictionary")
    merged = copy.deepcopy(metadata)
    for key, value in explicit.items():
        if value is not None:
            if key in merged and merged[key] != value:
                _error(f"checkpoint metadata conflict: {key}")
            merged[key] = value
    if "evaluator_id" not in merged:
        merged["evaluator_id"] = "visdrone_official_primary_evaluator_v1"
    required = {
        "evidence_contract_id",
        "training_contract_id",
        "runtime_plan_id",
        "mode",
        "run_id",
        "nonce",
        "epoch",
        "global_optimizer_step",
        "role",
        "predecessor_evidence_sha256",
    "source_identity",
    "data_identity",
    "environment_identity",
    "training_contract_sha256",
    "runtime_plan_sha256",
        "evaluator_id",
    }
    allowed = required | {"serialized_state_payloads"}
    if not required.issubset(merged) or not set(merged).issubset(allowed):
        _error(
            "checkpoint metadata key set drift: "
            f"missing={sorted(required - set(merged))} extra={sorted(set(merged) - required)}"
        )
    if merged["evidence_contract_id"] != EVIDENCE_CONTRACT_ID or merged["training_contract_id"] != TRAINING_CONTRACT_ID or merged["runtime_plan_id"] != RUNTIME_PLAN_ID:
        _error("checkpoint metadata authority identity drift")
    if (
        type(merged["mode"]) is not str
        or merged["mode"] not in {"synthetic", "real"}
        or type(merged["role"]) is not str
        or merged["role"] not in CHECKPOINT_ROLES
    ):
        _error("checkpoint metadata mode or role drift")
    _require_string(merged["run_id"], "checkpoint metadata.run_id")
    _require_string(merged["nonce"], "checkpoint metadata.nonce")
    _require_positive_int(merged["epoch"], "checkpoint metadata.epoch")
    _require_positive_int(merged["global_optimizer_step"], "checkpoint metadata.global_optimizer_step")
    _assert_sha(merged["predecessor_evidence_sha256"], "checkpoint metadata.predecessor_evidence_sha256")
    for field in ("source_identity", "data_identity", "environment_identity"):
        _validate_identity_object(merged[field], f"checkpoint metadata.{field}")
    _assert_sha(merged["training_contract_sha256"], "checkpoint metadata.training_contract_sha256")
    _assert_sha(merged["runtime_plan_sha256"], "checkpoint metadata.runtime_plan_sha256")
    if merged["evaluator_id"] != "visdrone_official_primary_evaluator_v1":
        _error("checkpoint metadata evaluator identity drift")
    return merged


def _state_payloads(state_input: Any, metadata: dict[str, Any]) -> dict[str, bytes]:
    if type(state_input) is dict:
        if set(state_input) != set(CHECKPOINT_REQUIRED_STATES):
            _error("checkpoint state mapping must contain exactly the 12 required states")
        result: dict[str, bytes] = {}
        for name in CHECKPOINT_REQUIRED_STATES:
            value = state_input[name]
            if type(value) is not bytes:
                _error(f"checkpoint state {name} must be caller-supplied bytes")
            result[name] = bytes(value)
        return result
    if type(state_input) is bytes:
        supplied = metadata.get("serialized_state_payloads")
        if type(supplied) is not dict or set(supplied) != set(CHECKPOINT_REQUIRED_STATES) or any(type(value) is not bytes for value in supplied.values()):
            _error("serialized checkpoint bytes require an explicit 12-state byte mapping")
        return {name: bytes(supplied[name]) for name in CHECKPOINT_REQUIRED_STATES}
    _error("checkpoint input must be a detached state mapping or explicit bytes")
    return {}


def _probe_loadability(payload: bytes, probe: Callable[[bytes], Any] | None) -> None:
    if not callable(probe):
        _error("checkpoint loadability probe is required")
    before = _sha256_bytes(payload)
    try:
        result = probe(bytes(payload))
    except Exception as exc:
        raise TrainingEvidenceError("checkpoint loadability probe failed") from exc
    if _sha256_bytes(payload) != before:
        _error("checkpoint loadability probe mutated input bytes")
    if result is True:
        return
    if type(result) is dict and set(result) == {"schema_version", "loadable"} and result["schema_version"] == EVIDENCE_CONTRACT_SCHEMA_VERSION and result["loadable"] is True:
        return
    _error("checkpoint loadability probe did not affirm the frozen schema")


def _make_checkpoint_payload(state_input: Any, metadata: dict[str, Any]) -> tuple[bytes, dict[str, Any]]:
    state_payloads = _state_payloads(state_input, metadata)
    states: list[dict[str, Any]] = []
    for name in CHECKPOINT_REQUIRED_STATES:
        payload = state_payloads[name]
        states.append({
            "name": name,
            "type": "bytes",
            "format": "synthetic_bytes",
            "size_bytes": len(payload),
            "sha256": _sha256_bytes(payload),
            "loadability_pass": True,
            "payload_hex": payload.hex(),
        })
    relative_path = f"checkpoints/checkpoint-{metadata['role']}-{metadata['epoch']:04d}{CHECKPOINT_EXTENSION}"
    checkpoint = {
        "schema_version": EVIDENCE_CONTRACT_SCHEMA_VERSION,
        "evidence_contract_id": EVIDENCE_CONTRACT_ID,
        "training_contract_id": metadata["training_contract_id"],
        "runtime_plan_id": metadata["runtime_plan_id"],
        "mode": metadata["mode"],
        "run_id": metadata["run_id"],
        "nonce": metadata["nonce"],
        "epoch": metadata["epoch"],
        "global_optimizer_step": metadata["global_optimizer_step"],
        "role": metadata["role"],
        "relative_path": relative_path,
        "required_state_order": list(CHECKPOINT_REQUIRED_STATES),
        "states": states,
        "source_identity": copy.deepcopy(metadata["source_identity"]),
        "data_identity": copy.deepcopy(metadata["data_identity"]),
        "environment_identity": copy.deepcopy(metadata["environment_identity"]),
        "evaluator_id": metadata["evaluator_id"],
        "training_contract_sha256": metadata["training_contract_sha256"],
        "runtime_plan_sha256": metadata["runtime_plan_sha256"],
        "predecessor_evidence_sha256": metadata["predecessor_evidence_sha256"],
        "state_inventory_sha256": _state_inventory(states),
        "publication_status": "PUBLISHED",
    }
    _validate_checkpoint_payload(checkpoint)
    return _canonical_json_bytes(checkpoint), checkpoint


def write_atomic_checkpoint(
    checkpoint_root: str | os.PathLike[str],
    state_input: Mapping[str, bytes] | bytes,
    metadata: Mapping[str, Any] | None = None,
    *,
    loadability_probe: Callable[[bytes], Any] | None = None,
    evidence_contract_id: str | None = None,
    training_contract_id: str | None = None,
    runtime_plan_id: str | None = None,
    mode: str | None = None,
    run_id: str | None = None,
    nonce: str | None = None,
    epoch: int | None = None,
    global_optimizer_step: int | None = None,
    role: str | None = None,
    predecessor_evidence_sha256: str | None = None,
    source_identity: Mapping[str, Any] | None = None,
    data_identity: Mapping[str, Any] | None = None,
    environment_identity: Mapping[str, Any] | None = None,
    evaluator_id: str | None = None,
) -> dict[str, Any]:
    """Publish one identity-bound synthetic checkpoint without overwriting."""

    explicit = {
        "evidence_contract_id": evidence_contract_id,
        "training_contract_id": training_contract_id,
        "runtime_plan_id": runtime_plan_id,
        "mode": mode,
        "run_id": run_id,
        "nonce": nonce,
        "epoch": epoch,
        "global_optimizer_step": global_optimizer_step,
        "role": role,
        "predecessor_evidence_sha256": predecessor_evidence_sha256,
        "source_identity": None if source_identity is None else dict(source_identity),
        "data_identity": None if data_identity is None else dict(data_identity),
        "environment_identity": None if environment_identity is None else dict(environment_identity),
        "evaluator_id": evaluator_id,
    }
    supplied_metadata = {} if metadata is None else dict(metadata)
    if type(state_input) is bytes and "serialized_state_payloads" not in supplied_metadata:
        supplied_metadata["serialized_state_payloads"] = None
    checkpoint_metadata = _checkpoint_metadata(supplied_metadata, explicit)
    payload, checkpoint = _make_checkpoint_payload(state_input, checkpoint_metadata)
    _probe_loadability(payload, loadability_probe)

    checkpoint_path = f"{checkpoint_root}/checkpoint-{checkpoint_metadata['role']}-{checkpoint_metadata['epoch']:04d}{CHECKPOINT_EXTENSION}"
    root_fd, fds = _open_existing_directory(checkpoint_root, "checkpoint root")
    try:
        _atomic_publish_at(
            root_fd,
            f"checkpoint-{checkpoint_metadata['role']}-{checkpoint_metadata['epoch']:04d}{CHECKPOINT_EXTENSION}",
            payload,
            checkpoint_path,
        )
        readback, _ = _read_regular_at(
            root_fd,
            f"checkpoint-{checkpoint_metadata['role']}-{checkpoint_metadata['epoch']:04d}{CHECKPOINT_EXTENSION}",
            checkpoint_path,
        )
        if readback != payload:
            _error("checkpoint final readback mismatch")
    finally:
        for fd in reversed(fds):
            _close_fd(fd)
    result = copy.deepcopy(checkpoint)
    result["checkpoint_size_bytes"] = len(payload)
    result["checkpoint_sha256"] = _sha256_bytes(payload)
    return result


def _expected_context(
    *,
    mode: str,
    run_id: str,
    nonce: str,
    source_identity: dict[str, Any],
    data_identity: dict[str, Any],
    environment_identity: dict[str, Any],
    training_contract_sha256: str,
    runtime_plan_sha256: str,
) -> dict[str, Any]:
    _assert_sha(training_contract_sha256, "training contract binding SHA")
    _assert_sha(runtime_plan_sha256, "runtime plan binding SHA")
    return {
        "schema_version": EVIDENCE_CONTRACT_SCHEMA_VERSION,
        "evidence_contract_id": EVIDENCE_CONTRACT_ID,
        "training_contract_id": TRAINING_CONTRACT_ID,
        "runtime_plan_id": RUNTIME_PLAN_ID,
        "mode": mode,
        "run_id": run_id,
        "nonce": nonce,
        "training_contract_sha256": training_contract_sha256,
        "runtime_plan_sha256": runtime_plan_sha256,
        "source_identity_sha256": _identity_digest(source_identity),
        "data_identity_sha256": _identity_digest(data_identity),
        "environment_identity_sha256": _identity_digest(environment_identity),
        "evaluator_id": "visdrone_official_primary_evaluator_v1",
    }


def _ensure_record_file(root_fd: int, name: str) -> None:
    observed = _lstat_at(name, root_fd)
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
        _error(f"record file is not a stable regular file: {name}")


def _checkpoint_expected(context: dict[str, Any], checkpoint: dict[str, Any]) -> dict[str, Any]:
    return {
        "training_contract_id": context["training_contract_id"],
        "runtime_plan_id": context["runtime_plan_id"],
        "mode": context["mode"],
        "run_id": context["run_id"],
        "nonce": context["nonce"],
        "training_contract_sha256": context["training_contract_sha256"],
        "runtime_plan_sha256": context["runtime_plan_sha256"],
        "epoch": checkpoint["epoch"],
        "global_optimizer_step": checkpoint["global_optimizer_step"],
        "role": checkpoint["role"],
        "relative_path": checkpoint["relative_path"],
        "evaluator_id": context["evaluator_id"],
    }


class TrainingEvidenceWriter:
    """Own one new evidence root and publish an immutable ordered evidence chain."""

    def __init__(
        self,
        evidence_root: str | os.PathLike[str],
        *,
        run_id: str,
        nonce: str,
        mode: str,
        training_contract_binding: Mapping[str, Any],
        runtime_plan_binding: Mapping[str, Any],
        source_identity: Mapping[str, Any],
        data_identity: Mapping[str, Any],
        environment_identity: Mapping[str, Any],
        checkpoint_root: str | os.PathLike[str] | None = None,
    ) -> None:
        if mode not in {"synthetic", "real"}:
            _error("evidence writer mode must be synthetic or real")
        _require_string(run_id, "writer.run_id")
        _require_string(nonce, "writer.nonce")
        contract = copy.deepcopy(_FROZEN_CONTRACT)
        _assert_authority_bindings(contract, dict(training_contract_binding), dict(runtime_plan_binding))
        self._training_binding = copy.deepcopy(dict(training_contract_binding))
        self._runtime_binding = copy.deepcopy(dict(runtime_plan_binding))
        self.root_path = _absolute_path(evidence_root, "evidence root")
        self._checkpoint_root_explicit = None if checkpoint_root is None else _absolute_path(checkpoint_root, "checkpoint root")
        if self._checkpoint_root_explicit is not None:
            fd, fds = _open_existing_directory(self._checkpoint_root_explicit, "checkpoint root")
            for item in reversed(fds):
                _close_fd(item)
        self._context = _expected_context(
            mode=mode,
            run_id=run_id,
            nonce=nonce,
            source_identity=_validate_identity_object(dict(source_identity), "writer.source_identity"),
            data_identity=_validate_identity_object(dict(data_identity), "writer.data_identity"),
            environment_identity=_validate_identity_object(dict(environment_identity), "writer.environment_identity"),
            training_contract_sha256=self._training_binding["canonical_sha256"],
            runtime_plan_sha256=self._runtime_binding["canonical_sha256"],
        )
        self._source_identity = copy.deepcopy(dict(source_identity))
        self._data_identity = copy.deepcopy(dict(data_identity))
        self._environment_identity = copy.deepcopy(dict(environment_identity))
        self._root_fds: list[int] = []
        self._root_parent_fd: int | None = None
        self._root_fd: int | None = None
        self._root_name = ""
        self._checkpoint_fd: int | None = None
        self._checkpoint_fds: list[int] = []
        self._closed = False
        self._terminal = False
        self._epoch_count = 0
        self._checkpoint_count = 0
        self._last_epoch: int | None = None
        self._last_step: int | None = None
        self._last_ema: int | None = None
        self._chain_head = ""
        parent_fd, root_name, parent_fds = _open_existing_parent(self.root_path, "evidence root")
        self._root_parent_fd = parent_fd
        self._root_name = root_name
        self._root_fds.extend(parent_fds)
        try:
            root_fd, root_fds = _mkdir_child(parent_fd, root_name, "evidence root")
            self._root_fd = root_fd
            self._root_fds.extend(root_fds)
            prepared = {
                "schema_version": EVIDENCE_CONTRACT_SCHEMA_VERSION,
                "evidence_contract_id": EVIDENCE_CONTRACT_ID,
                "training_contract_id": TRAINING_CONTRACT_ID,
                "runtime_plan_id": RUNTIME_PLAN_ID,
                "mode": mode,
                "run_id": run_id,
                "nonce": nonce,
                "source_identity": copy.deepcopy(self._source_identity),
                "data_identity": copy.deepcopy(self._data_identity),
                "environment_identity": copy.deepcopy(self._environment_identity),
                "training_contract_sha256": self._training_binding["canonical_sha256"],
                "runtime_plan_sha256": self._runtime_binding["canonical_sha256"],
                "evaluator_id": self._context["evaluator_id"],
                "sequence": 0,
                "predecessor_sha256": None,
                "status": "PREPARED",
            }
            _validate_prepared(prepared)
            _atomic_publish_at(self._require_root_fd(), PREPARED_NAME, _canonical_json_bytes(prepared), PREPARED_NAME)
            self._chain_head = _record_hash(prepared)
            _atomic_publish_at(self._require_root_fd(), EPOCH_RECORDS_NAME, b"", EPOCH_RECORDS_NAME)
            _atomic_publish_at(self._require_root_fd(), CHECKPOINT_REFERENCES_NAME, b"", CHECKPOINT_REFERENCES_NAME)
            if self._checkpoint_root_explicit is None:
                child_fd, child_fds = _mkdir_child(self._require_root_fd(), "checkpoints", "checkpoint root")
                self._checkpoint_fd = child_fd
                self._checkpoint_fds.extend(child_fds)
                self._checkpoint_root_explicit = f"{self.root_path}/checkpoints"
        except BaseException:
            self.close()
            raise

    def _require_root_fd(self) -> int:
        if self._closed or self._root_fd is None or self._root_parent_fd is None:
            _error("evidence writer is closed")
        observed = _lstat_at(self._root_name, self._root_parent_fd)
        opened = os.fstat(self._root_fd)
        if not _same_object(observed, opened) or not stat.S_ISDIR(opened.st_mode) or stat.S_IMODE(opened.st_mode) != 0o700 or opened.st_uid != os.getuid() or opened.st_gid != os.getgid():
            _error("evidence root identity drift")
        return self._root_fd

    def _ensure_open_nonterminal(self) -> int:
        if self._terminal:
            _error("evidence writer has already been terminalized")
        return self._require_root_fd()

    def close(self) -> None:
        if self._closed:
            return
        for fd in reversed(self._root_fds + self._checkpoint_fds):
            _close_fd(fd)
        self._root_fds.clear()
        self._checkpoint_fds.clear()
        self._root_fd = None
        self._root_parent_fd = None
        self._checkpoint_fd = None
        self._closed = True

    def __enter__(self) -> "TrainingEvidenceWriter":
        self._require_root_fd()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        self.close()
        return False

    def _complete_epoch_record(self, value: Mapping[str, Any]) -> dict[str, Any]:
        if type(value) is not dict:
            _error("epoch observation must be a dictionary")
        observation_keys = {
            "epoch",
            "global_optimizer_step",
            "loss",
            "gradient_norm",
            "amp_scale",
            "amp_skipped_steps",
            "amp_overflow_events",
            "nonfinite_loss_count",
            "nonfinite_gradient_count",
            "ema_updates",
            "evaluator_result_sha256",
            "checkpoint_sha256",
        }
        if not set(value).issubset(observation_keys | {"sequence", "predecessor_sha256", "evaluator_id"}):
            _error("epoch observation contains unexpected keys")
        required_observations = observation_keys - {"checkpoint_sha256"}
        if not required_observations.issubset(value):
            _error("epoch observation is incomplete")
        record = {
            **self._context,
            "sequence": self._epoch_count + 1,
            "epoch": value["epoch"],
            "global_optimizer_step": value["global_optimizer_step"],
            "predecessor_sha256": self._chain_head,
            "loss": value["loss"],
            "gradient_norm": value["gradient_norm"],
            "amp_scale": value["amp_scale"],
            "amp_skipped_steps": value["amp_skipped_steps"],
            "amp_overflow_events": value["amp_overflow_events"],
            "nonfinite_loss_count": value["nonfinite_loss_count"],
            "nonfinite_gradient_count": value["nonfinite_gradient_count"],
            "ema_updates": value["ema_updates"],
            "evaluator_id": value.get("evaluator_id", self._context["evaluator_id"]),
            "evaluator_result_sha256": value["evaluator_result_sha256"],
            "checkpoint_sha256": value.get("checkpoint_sha256"),
        }
        if "sequence" in value and value["sequence"] != record["sequence"]:
            _error("epoch sequence is writer-owned")
        if "predecessor_sha256" in value and value["predecessor_sha256"] != record["predecessor_sha256"]:
            _error("epoch predecessor is writer-owned")
        if "evaluator_id" in value and value["evaluator_id"] != record["evaluator_id"]:
            _error("epoch evaluator is writer-owned")
        return record

    def write_epoch_record(self, value: Mapping[str, Any]) -> dict[str, Any]:
        root_fd = self._ensure_open_nonterminal()
        record = self._complete_epoch_record(value)
        previous_epoch = self._last_epoch
        previous_step = self._last_step
        previous_ema = self._last_ema
        _validate_epoch(
            record,
            self._context,
            expected_sequence=self._epoch_count + 1,
            expected_predecessor=self._chain_head,
            previous_epoch=previous_epoch,
            previous_step=previous_step,
            previous_ema=previous_ema,
        )
        if record["source_identity_sha256"] != self._context["source_identity_sha256"] or record["data_identity_sha256"] != self._context["data_identity_sha256"] or record["environment_identity_sha256"] != self._context["environment_identity_sha256"] or record["evaluator_id"] != self._context["evaluator_id"]:
            _error("epoch observation authority drift")
        payload = _canonical_json_bytes(record) + b"\n"
        _append_line_at(root_fd, EPOCH_RECORDS_NAME, payload, EPOCH_RECORDS_NAME)
        self._chain_head = _record_hash(record)
        self._epoch_count += 1
        self._last_epoch = record["epoch"]
        self._last_step = record["global_optimizer_step"]
        self._last_ema = record["ema_updates"]
        return copy.deepcopy(record)

    append_epoch_record = write_epoch_record
    record_epoch = write_epoch_record

    def write_checkpoint(
        self,
        state_input: Mapping[str, bytes] | bytes,
        *,
        role: str,
        loadability_probe: Callable[[bytes], Any],
        epoch: int | None = None,
        global_optimizer_step: int | None = None,
    ) -> dict[str, Any]:
        self._ensure_open_nonterminal()
        if self._epoch_count == 0:
            _error("checkpoint requires an epoch record")
        actual_epoch = self._last_epoch if epoch is None else epoch
        actual_step = self._last_step if global_optimizer_step is None else global_optimizer_step
        if actual_epoch != self._last_epoch or actual_step != self._last_step:
            _error("checkpoint must bind the latest epoch observation")
        metadata = {
            "evidence_contract_id": EVIDENCE_CONTRACT_ID,
            "training_contract_id": TRAINING_CONTRACT_ID,
            "runtime_plan_id": RUNTIME_PLAN_ID,
            "mode": self._context["mode"],
            "run_id": self._context["run_id"],
            "nonce": self._context["nonce"],
            "epoch": actual_epoch,
            "global_optimizer_step": actual_step,
            "role": role,
            "predecessor_evidence_sha256": self._chain_head,
            "source_identity": copy.deepcopy(self._source_identity),
            "data_identity": copy.deepcopy(self._data_identity),
            "environment_identity": copy.deepcopy(self._environment_identity),
            "training_contract_sha256": self._training_binding["canonical_sha256"],
            "runtime_plan_sha256": self._runtime_binding["canonical_sha256"],
            "evaluator_id": self._context["evaluator_id"],
        }
        checkpoint = write_atomic_checkpoint(
            self._checkpoint_root_explicit,
            state_input,
            metadata,
            loadability_probe=loadability_probe,
        )
        reference = {
            **self._context,
            "sequence": self._checkpoint_count + 1,
            "epoch": actual_epoch,
            "global_optimizer_step": actual_step,
            "predecessor_sha256": self._chain_head,
            "role": role,
            "relative_path": checkpoint["relative_path"],
            "checkpoint_size_bytes": checkpoint["checkpoint_size_bytes"],
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "state_inventory_sha256": checkpoint["state_inventory_sha256"],
            "loadability_pass": True,
        }
        _validate_checkpoint_reference(
            reference,
            self._context,
            expected_sequence=self._checkpoint_count + 1,
            expected_predecessor=self._chain_head,
            previous_epoch=self._last_epoch,
            previous_step=self._last_step,
        )
        payload = _canonical_json_bytes(reference) + b"\n"
        _append_line_at(self._require_root_fd(), CHECKPOINT_REFERENCES_NAME, payload, CHECKPOINT_REFERENCES_NAME)
        self._chain_head = _record_hash(reference)
        self._checkpoint_count += 1
        return copy.deepcopy(checkpoint)

    def complete(self, *, exit_code: int = 0) -> dict[str, Any]:
        root_fd = self._ensure_open_nonterminal()
        if type(exit_code) is not int or exit_code != 0:
            _error("successful completion requires exit code zero")
        if self._epoch_count == 0 or self._checkpoint_count == 0:
            _error("terminal success requires ordered epochs and checkpoints")
        if self._context["mode"] == "real" and self._epoch_count != 120:
            _error("real terminal success requires exactly 120 epochs")
        inventory = _inventory_root(root_fd, excluded=FINAL_EXCLUDED, expected=self._context)
        inventory_payload = _canonical_json_bytes(inventory)
        _atomic_publish_at(root_fd, ARTIFACT_INVENTORY_NAME, inventory_payload, ARTIFACT_INVENTORY_NAME)
        status = "SYNTHETIC_TERMINAL_COMPLETE" if self._context["mode"] == "synthetic" else "TERMINAL_COMPLETE"
        completion = {
            **self._context,
            "status": status,
            "exit_code": 0,
            "epoch_count": self._epoch_count,
            "final_epoch": self._last_epoch,
            "global_optimizer_step": self._last_step,
            "predecessor_sha256": self._chain_head,
            "artifact_inventory_sha256": _sha256_bytes(inventory_payload),
            "checkpoint_count": self._checkpoint_count,
        }
        _validate_completion(completion, self._context, epoch_count=self._epoch_count, final_epoch=self._last_epoch or 0, final_step=self._last_step or 0, predecessor=self._chain_head, inventory_sha=completion["artifact_inventory_sha256"], checkpoint_count=self._checkpoint_count)
        _atomic_publish_at(root_fd, COMPLETION_NAME, _canonical_json_bytes(completion), COMPLETION_NAME)
        self._terminal = True
        return validate_training_evidence(self.root_path)

    finalize_success = complete

    def fail(self, error: BaseException, *, failure_type: str = "PERMANENT_FAIL") -> dict[str, Any]:
        root_fd = self._ensure_open_nonterminal()
        if failure_type != "PERMANENT_FAIL":
            _error("failure type must be PERMANENT_FAIL")
        exception_type = type(error).__name__
        message = str(error)
        _require_string(exception_type, "failure.exception_type")
        if type(message) is not str:
            _error("failure message must be a string")
        partial = _inventory_root(root_fd, excluded=PARTIAL_EXCLUDED, expected=self._context)
        partial_payload = _canonical_json_bytes(partial)
        _atomic_publish_at(root_fd, PARTIAL_INVENTORY_NAME, partial_payload, PARTIAL_INVENTORY_NAME)
        failure = {
            **self._context,
            "status": "PERMANENT_FAIL",
            "failure_type": "PERMANENT_FAIL",
            "exception_type": exception_type,
            "message": message,
            "epoch_count": self._epoch_count,
            "predecessor_sha256": self._chain_head,
            "partial_inventory_sha256": _sha256_bytes(partial_payload),
        }
        _validate_failure(failure, self._context, epoch_count=self._epoch_count, predecessor=self._chain_head, partial_sha=failure["partial_inventory_sha256"])
        _atomic_publish_at(root_fd, FAILURE_NAME, _canonical_json_bytes(failure), FAILURE_NAME)
        self._terminal = True
        return validate_training_evidence(self.root_path)

    finalize_failure = fail


def _context_from_prepared(prepared: dict[str, Any]) -> dict[str, Any]:
    _validate_prepared(prepared)
    return {
        "schema_version": prepared["schema_version"],
        "evidence_contract_id": prepared["evidence_contract_id"],
        "training_contract_id": prepared["training_contract_id"],
        "runtime_plan_id": prepared["runtime_plan_id"],
        "mode": prepared["mode"],
        "run_id": prepared["run_id"],
        "nonce": prepared["nonce"],
        "training_contract_sha256": prepared["training_contract_sha256"],
        "runtime_plan_sha256": prepared["runtime_plan_sha256"],
        "source_identity_sha256": _identity_digest(prepared["source_identity"]),
        "data_identity_sha256": _identity_digest(prepared["data_identity"]),
        "environment_identity_sha256": _identity_digest(prepared["environment_identity"]),
        "evaluator_id": prepared["evaluator_id"],
    }


def _open_checkpoint_directory(root_fd: int) -> tuple[int, os.stat_result]:
    observed = _lstat_at("checkpoints", root_fd)
    if (
        stat.S_ISLNK(observed.st_mode)
        or not stat.S_ISDIR(observed.st_mode)
        or observed.st_uid != os.getuid()
        or observed.st_gid != os.getgid()
        or stat.S_IMODE(observed.st_mode) != 0o700
    ):
        _error("checkpoint directory identity is invalid")
    fd = _open_at("checkpoints", _directory_flags(), root_fd)
    opened = os.fstat(fd)
    if not _same_object(observed, opened):
        _close_fd(fd)
        _error("checkpoint directory identity changed")
    return fd, opened


def _validate_chain(
    prepared: dict[str, Any],
    epoch_records: list[dict[str, Any]],
    checkpoint_references: list[dict[str, Any]],
    expected: dict[str, Any],
) -> tuple[str, int | None, int | None, int | None]:
    previous_epoch: int | None = None
    previous_step: int | None = None
    previous_ema: int | None = None
    for index, record in enumerate(epoch_records, 1):
        _assert_sha(record.get("predecessor_sha256"), f"epoch record {index} predecessor")
        _validate_epoch(
            record,
            expected,
            expected_sequence=index,
            expected_predecessor=record["predecessor_sha256"],
            previous_epoch=previous_epoch,
            previous_step=previous_step,
            previous_ema=previous_ema,
        )
        previous_epoch = record["epoch"]
        previous_step = record["global_optimizer_step"]
        previous_ema = record["ema_updates"]

    previous_reference_epoch: int | None = None
    previous_reference_step: int | None = None
    for index, reference in enumerate(checkpoint_references, 1):
        _assert_sha(reference.get("predecessor_sha256"), f"checkpoint reference {index} predecessor")
        _validate_checkpoint_reference(
            reference,
            expected,
            expected_sequence=index,
            expected_predecessor=reference["predecessor_sha256"],
            previous_epoch=previous_reference_epoch,
            previous_step=previous_reference_step,
        )
        previous_reference_epoch = reference["epoch"]
        previous_reference_step = reference["global_optimizer_step"]

    events: dict[str, tuple[str, dict[str, Any]]] = {}
    for kind, records in (("epoch", epoch_records), ("checkpoint", checkpoint_references)):
        for record in records:
            predecessor = record["predecessor_sha256"]
            if predecessor in events:
                _error("evidence hash chain has multiple successors")
            events[predecessor] = (kind, record)

    current = _record_hash(prepared)
    visited = 0
    chain_epoch: int | None = None
    chain_step: int | None = None
    while current in events:
        kind, record = events[current]
        record_hash = _record_hash(record)
        visited += 1
        if kind == "epoch":
            if chain_epoch is not None and record["epoch"] != chain_epoch + 1:
                _error("evidence hash chain epoch order drift")
            if chain_step is not None and record["global_optimizer_step"] <= chain_step:
                _error("evidence hash chain step order drift")
            chain_epoch = record["epoch"]
            chain_step = record["global_optimizer_step"]
        else:
            if chain_epoch is None or record["epoch"] > chain_epoch or record["global_optimizer_step"] > (chain_step or 0):
                _error("checkpoint reference precedes its epoch evidence")
        current = record_hash
    if visited != len(epoch_records) + len(checkpoint_references):
        _error("evidence hash chain is disconnected or cyclic")
    return current, previous_epoch, previous_step, previous_ema


def _read_checkpoint_payloads(
    root_fd: int,
    expected: dict[str, Any],
    prepared: dict[str, Any],
    references: list[dict[str, Any]],
) -> dict[str, tuple[dict[str, Any], bytes]]:
    checkpoint_fd, _ = _open_checkpoint_directory(root_fd)
    try:
        names = sorted(os.listdir(checkpoint_fd))
        if any(name.startswith(".") for name in names):
            _error("checkpoint directory contains a temporary object")
        payloads: dict[str, tuple[dict[str, Any], bytes]] = {}
        for name in names:
            observed = _lstat_at(name, checkpoint_fd)
            if stat.S_ISLNK(observed.st_mode) or not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
                _error(f"checkpoint object is not a stable regular file: {name}")
            raw, _ = _read_regular_at(checkpoint_fd, name, f"checkpoint/{name}")
            payload = _parse_json(raw, f"checkpoint/{name}", trailing_lf=False)
            if raw != _canonical_json_bytes(payload):
                _error(f"checkpoint/{name} is repacked or not canonical")
            relative_path = payload.get("relative_path")
            if type(relative_path) is not str or relative_path != f"checkpoints/{name}":
                _error("checkpoint relative path binding drift")
            payloads[relative_path] = (payload, raw)

        if len(payloads) != len(names):
            _error("checkpoint relative paths are duplicated")
        if set(payloads) != {reference["relative_path"] for reference in references}:
            _error("checkpoint files and references are disjoint")
        for reference in references:
            payload, raw = payloads[reference["relative_path"]]
            expected_checkpoint = _checkpoint_expected(expected, payload)
            expected_checkpoint["predecessor_evidence_sha256"] = reference["predecessor_sha256"]
            _validate_checkpoint_payload(payload, expected=expected_checkpoint)
            if (
                payload["source_identity"] != prepared["source_identity"]
                or payload["data_identity"] != prepared["data_identity"]
                or payload["environment_identity"] != prepared["environment_identity"]
            ):
                _error("checkpoint source or data identity drift")
            if (
                reference["checkpoint_size_bytes"] != len(raw)
                or reference["checkpoint_sha256"] != _sha256_bytes(raw)
                or reference["state_inventory_sha256"] != payload["state_inventory_sha256"]
            ):
                _error("checkpoint reference artifact identity drift")
        return payloads
    finally:
        _close_fd(checkpoint_fd)


def _validate_evidence_root(root_path: str) -> dict[str, Any]:
    root_fd, fds = _open_existing_directory(root_path, "evidence root")
    try:
        root_stat = os.fstat(root_fd)
        if (
            not stat.S_ISDIR(root_stat.st_mode)
            or stat.S_IMODE(root_stat.st_mode) != 0o700
            or root_stat.st_uid != os.getuid()
            or root_stat.st_gid != os.getgid()
        ):
            _error("evidence root metadata is invalid")
        names = set(os.listdir(root_fd))
        base = {PREPARED_NAME, EPOCH_RECORDS_NAME, CHECKPOINT_REFERENCES_NAME, "checkpoints"}
        if not base.issubset(names):
            _error("evidence root is missing a required file")
        if names - (base | {ARTIFACT_INVENTORY_NAME, COMPLETION_NAME, PARTIAL_INVENTORY_NAME, FAILURE_NAME}):
            _error("evidence root contains an unexpected object")
        prepared, _ = _read_json_file(root_fd, PREPARED_NAME, PREPARED_NAME)
        prepared = _validate_prepared(prepared)
        expected = _context_from_prepared(prepared)
        epochs_raw, _ = _read_regular_at(root_fd, EPOCH_RECORDS_NAME, EPOCH_RECORDS_NAME)
        references_raw, _ = _read_regular_at(root_fd, CHECKPOINT_REFERENCES_NAME, CHECKPOINT_REFERENCES_NAME)
        epoch_records = _line_records(epochs_raw, EPOCH_RECORDS_NAME)
        checkpoint_references = _line_records(references_raw, CHECKPOINT_REFERENCES_NAME)
        chain_head, last_epoch, last_step, last_ema = _validate_chain(
            prepared, epoch_records, checkpoint_references, expected
        )
        payloads = _read_checkpoint_payloads(root_fd, expected, prepared, checkpoint_references)

        terminal_names = names - base
        if terminal_names == set():
            status = "IN_PROGRESS"
            terminal = None
            inventory = None
        elif terminal_names == {ARTIFACT_INVENTORY_NAME, COMPLETION_NAME}:
            inventory, inventory_raw = _read_json_file(root_fd, ARTIFACT_INVENTORY_NAME, ARTIFACT_INVENTORY_NAME)
            inventory = _validate_inventory(inventory, expected, partial=False)
            actual_inventory = _inventory_root(root_fd, excluded=FINAL_EXCLUDED, expected=expected)
            if inventory != actual_inventory:
                _error("final artifact inventory does not match evidence files")
            completion, _ = _read_json_file(root_fd, COMPLETION_NAME, COMPLETION_NAME)
            _validate_completion(
                completion,
                expected,
                epoch_count=len(epoch_records),
                final_epoch=last_epoch or 0,
                final_step=last_step or 0,
                predecessor=chain_head,
                inventory_sha=_sha256_bytes(inventory_raw),
                checkpoint_count=len(checkpoint_references),
            )
            status = completion["status"]
            terminal = completion
        elif terminal_names == {PARTIAL_INVENTORY_NAME, FAILURE_NAME}:
            inventory, inventory_raw = _read_json_file(root_fd, PARTIAL_INVENTORY_NAME, PARTIAL_INVENTORY_NAME)
            inventory = _validate_inventory(inventory, expected, partial=True)
            actual_inventory = _inventory_root(
                root_fd,
                excluded=PARTIAL_EXCLUDED,
                extra_excluded=(FAILURE_NAME,),
                expected=expected,
            )
            if inventory != actual_inventory:
                _error("partial artifact inventory does not match evidence files")
            failure, _ = _read_json_file(root_fd, FAILURE_NAME, FAILURE_NAME)
            _validate_failure(
                failure,
                expected,
                epoch_count=len(epoch_records),
                predecessor=chain_head,
                partial_sha=_sha256_bytes(inventory_raw),
            )
            status = "TERMINAL_FAILED"
            terminal = failure
        else:
            _error("evidence root terminal files are incomplete or contradictory")

        return {
            "status": status,
            "schema_version": expected["schema_version"],
            "evidence_contract_id": expected["evidence_contract_id"],
            "training_contract_id": expected["training_contract_id"],
            "runtime_plan_id": expected["runtime_plan_id"],
            "mode": expected["mode"],
            "run_id": expected["run_id"],
            "nonce": expected["nonce"],
            "training_contract_sha256": expected["training_contract_sha256"],
            "runtime_plan_sha256": expected["runtime_plan_sha256"],
            "source_identity_sha256": expected["source_identity_sha256"],
            "data_identity_sha256": expected["data_identity_sha256"],
            "environment_identity_sha256": expected["environment_identity_sha256"],
            "evaluator_id": expected["evaluator_id"],
            "epoch_count": len(epoch_records),
            "checkpoint_count": len(checkpoint_references),
            "last_epoch": last_epoch,
            "last_global_optimizer_step": last_step,
            "last_ema_updates": last_ema,
            "chain_head": chain_head,
            "prepared": copy.deepcopy(prepared),
            "epoch_records": copy.deepcopy(epoch_records),
            "checkpoint_references": copy.deepcopy(checkpoint_references),
            "checkpoints": copy.deepcopy({path: payload for path, (payload, _) in payloads.items()}),
            "inventory": copy.deepcopy(inventory),
            "terminal": copy.deepcopy(terminal),
        }
    finally:
        for fd in reversed(fds):
            _close_fd(fd)


def validate_training_checkpoint(
    checkpoint: str | os.PathLike[str] | Mapping[str, Any],
    expected: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate one detached checkpoint payload or a published checkpoint path."""

    if type(checkpoint) is dict:
        return _validate_checkpoint_payload(
            copy.deepcopy(checkpoint),
            expected=None if expected is None else dict(expected),
        )
    path = _absolute_path(checkpoint, "checkpoint path")
    parent_fd, name, fds = _open_existing_parent(path, "checkpoint path")
    try:
        if not path.rsplit("/", 1)[0].endswith("/checkpoints"):
            _error("checkpoint path is outside a checkpoint directory")
        raw, _ = _read_regular_at(parent_fd, name, path)
        payload = _parse_json(raw, path, trailing_lf=False)
        return _validate_checkpoint_payload(
            payload,
            expected=None if expected is None else dict(expected),
        )
    finally:
        for fd in reversed(fds):
            _close_fd(fd)


def validate_training_evidence(evidence_root: str | os.PathLike[str]) -> dict[str, Any]:
    """Read and validate an evidence root without creating or modifying files."""

    return _validate_evidence_root(_absolute_path(evidence_root, "evidence root"))


def classify_training_evidence(evidence_root: str | os.PathLike[str]) -> str:
    """Return a conservative state for a training evidence root."""

    path = _absolute_path(evidence_root, "evidence root")
    try:
        os.lstat(path)
    except FileNotFoundError:
        return "ABSENT"
    except OSError:
        return "UNKNOWN"
    try:
        return validate_training_evidence(path)["status"]
    except Exception:
        return "UNKNOWN"
