"""Pure validation helpers for the detached T6A launch contract.

This module deliberately has no launch capability.  It neither creates nor
consumes an owner authorization, writes files, imports a model runtime, or
reads training data.  A later launch stage supplies all launch-time facts.
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
from typing import Any, Mapping


LAUNCH_CONTRACT_CONFIG_RELATIVE_PATH = (
    "configs/baseline/rtdetrv2_r18_visdrone_training_launch_t6_v1.json"
)
LAUNCH_CONTRACT_ID = "rtdetrv2_r18_visdrone_baseline_training_launch_t6_v1"
LAUNCH_CONTRACT_SCHEMA_VERSION = 1

# These values are filled from the checked-in canonical config before release.
LAUNCH_CONTRACT_RAW_SIZE_BYTES = 11730
LAUNCH_CONTRACT_RAW_SHA256 = "539abe556edd00a5bb0ecf0ed350195ee28e55a0af0b804404a0d92ea9830ff9"
LAUNCH_CONTRACT_CANONICAL_SIZE_BYTES = 11729
LAUNCH_CONTRACT_CANONICAL_SHA256 = "13a5923aa62f3048b2baefe32aaa163b9aa44d3b3e11d2a374ebceec5034ae94"
_T6A_LAUNCH_BRANCH = "codex/p3-rtdetrv2-baseline-training-t6-authorization-contract-r1"

_SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
_GIT_RE = re.compile(r"[0-9a-f]{40}\Z")
_UUID_RE = re.compile(r"GPU-[0-9a-f-]{8,}\Z")
_NONCE_RE = re.compile(r"[0-9a-f]{32,}\Z")


def _file(relative_path: str, size_bytes: int, sha256: str) -> dict[str, Any]:
    return {
        "relative_path": relative_path,
        "size_bytes": size_bytes,
        "sha256": sha256,
    }


def _config_file(
    relative_path: str,
    raw_size_bytes: int,
    raw_sha256: str,
    canonical_size_bytes: int,
    canonical_sha256: str,
) -> dict[str, Any]:
    return {
        "relative_path": relative_path,
        "raw_size_bytes": raw_size_bytes,
        "raw_sha256": raw_sha256,
        "canonical_size_bytes": canonical_size_bytes,
        "canonical_sha256": canonical_sha256,
    }


_FROZEN_REPOSITORY_REFERENCE = {
    "repo_root_policy": "absolute_canonical_path_required_at_launch",
    "branch": "codex/p3-rtdetrv2-baseline-training-process-launcher-t5d-r1",
    "head": "0bd5d982b4c5e5cc94af445bea05f4d125630a03",
    "tree": "907d563b1b584659960f6de72861f5183733bdfc",
    "parent": "e5d1535af8e9eca6a9d093066b58846242630ccc",
    "upstream": "origin/codex/p3-rtdetrv2-baseline-training-process-launcher-t5d-r1",
    "upstream_sha": "0bd5d982b4c5e5cc94af445bea05f4d125630a03",
}

_FROZEN_SOURCE_BINDINGS = {
    "training_contract": {
        "config": _config_file(
            "configs/baseline/rtdetrv2_r18_visdrone_training_v1.json",
            10907,
            "0f9ddb2e6d8ec6d2b21e42f6c419511f5bd70df287457c2c2b6109eaf8a93297",
            9117,
            "a20c71ef90cb4ccc091a717286a1c49be8a1ebffb178156942b77798d3d7f868",
        ),
        "module": _file(
            "src/sparse_rtdetr/baseline/training_contract.py",
            151656,
            "21599fae8312a32ddf833bf2f53557ed5b4d4122f0be167f2071f7f82008deef",
        ),
        "document": _file(
            "docs/contracts/RTDETR_BASELINE_FORMAL_TRAINING_V1.md",
            16298,
            "94d90a5b411d8d9462d81b5b9be434ff15b39c5787d10d42b6f3524fe4d6d2ba",
        ),
    },
    "runtime_plan": {
        "config": _config_file(
            "configs/baseline/rtdetrv2_r18_visdrone_training_runtime_v1.json",
            12957,
            "cb6af1abae9351b4a268681587db82ad059745f1d7e198d7d8c829b4cee41aef",
            10888,
            "3812b04d957e1cd7c3990c8458a540651c95c6bd551c631fc51717f2f7b386ac",
        ),
        "module": _file(
            "src/sparse_rtdetr/baseline/training_runtime.py",
            36824,
            "735d75f612f59e6cec22dc53002980471c34c8b18601cd58704440f85741c7c9",
        ),
        "document": _file(
            "docs/contracts/RTDETR_BASELINE_FORMAL_TRAINING_RUNTIME_V1.md",
            7531,
            "6fe20eeda7c7a14c2cb694b8f07f890899ee54f5754ff7c49d7c98c1ae73cba6",
        ),
    },
    "training_evidence": {
        "config": _config_file(
            "configs/baseline/rtdetrv2_r18_visdrone_training_evidence_v1.json",
            7139,
            "4d6bad4afbede236f1169796aa640f9d82bdcf91d06b799d39c183b20c230c9e",
            6069,
            "3184707c6cfa115477d007ac0be5eb142a054f78e6309a079ad1e9fab0db9bd6",
        ),
        "module": _file(
            "src/sparse_rtdetr/baseline/training_evidence.py",
            108891,
            "5efad5ea963f2b08e52cc7b7811ca64472dd36a5186cd5c476c7657adb5bc1b0",
        ),
        "document": _file(
            "docs/contracts/RTDETR_BASELINE_FORMAL_TRAINING_EVIDENCE_V1.md",
            7747,
            "8bd7ecabb2940901ca508092108959209c1b3a8c6305f96cc3d01da9b2eda931",
        ),
    },
    "prepared_adapter": {
        "module": _file(
            "src/sparse_rtdetr/baseline/training_adapter.py",
            90937,
            "3dca63c67fc2263b9f3f3006b05c8667e4457a5d1344db41f08e720e85016fd3",
        ),
        "document": _file(
            "docs/contracts/RTDETR_BASELINE_PREPARED_TRAINER_ADAPTER_V1.md",
            6790,
            "a5e9dafe6d4bdd608e4a83e82dd5d47a320e25c08a6e725d2f073ac3d9a01100",
        ),
    },
    "process": {
        "config": _config_file(
            "configs/baseline/rtdetrv2_r18_visdrone_training_process_v1.json",
            10468,
            "a16ff1c31e0a935f0ab6ab9224b7c3f70c377006d00562dc5b2bc1b44f70ed89",
            9330,
            "8469bde5b8e87e47ef8df0b9674aa1e685bf993f90ef3bc99aa3354bbe95f19c",
        ),
        "entry_module": _file(
            "src/sparse_rtdetr/baseline/training_entry.py",
            67560,
            "6990a1a0f2d47b034db5a0aa3928fdb1820819f8272c85f2b86c78832d464892",
        ),
        "launcher_module": _file(
            "src/sparse_rtdetr/baseline/training_process_launcher.py",
            61791,
            "7543e20ea23e9e9b526d4ca89c6c3db6f4d77127a9ba48e48cc926edb5ea54de",
        ),
        "document": _file(
            "docs/contracts/RTDETR_BASELINE_FORMAL_TRAINING_PROCESS_LAUNCH_V1.md",
            9202,
            "ff1d3c8b170b63e20fb62e0737a7167a87864a7b3b0e67fb49b874d3a5736523",
        ),
    },
    "primary_evaluator": {
        "evaluator_id": "visdrone_official_primary_evaluator_v1",
        "protocol_id": "visdrone_official_style_v1",
        "implementation_commit": "036cca4d127ddd9e10e3cc7900c3eb759b55f59f",
        "implementation_tree": "fbe931976f6bac7d8e4b3bb319ff1905e99c5444",
        "implementation": _file(
            "src/sparse_rtdetr/baseline/primary_evaluator.py",
            69468,
            "d831fc641ac930822e693f99fbbcdd48abbe76738a618525135dd099963901bd",
        ),
        "test": _file(
            "tests/test_rtdetr_baseline_primary_evaluator.py",
            45357,
            "ed70a23bac8feeb0fcfa66d0c2ccfed9b02ddf8ad16cf152f98c9acb2c05bdf2",
        ),
        "document": _file(
            "docs/contracts/RTDETR_BASELINE_PRIMARY_EVALUATOR_V1.md",
            5759,
            "270587ed1022ecbdab151b025b4d23ba641b0e07cb3d1f11cbe17ead0798369e",
        ),
        "config": _config_file(
            "configs/baseline/visdrone_official_evaluator_v1.json",
            3857,
            "36cfa69b0ff645c47c871283f917577ca27c760daf424e9544ab363dbf080ff5",
            3295,
            "355de90bdb6007ed42ed65b3f653a1b8ea57f2184ab4fd48a9561b692b993998",
        ),
        "authority_manifest": _config_file(
            "manifests/visdrone_det_toolkit_005445.json",
            4166,
            "71168baf15d6d945fd5a4a6c5605b5ba533524efeede2a9d4020127576c1b36f",
            3351,
            "5bad9faf7622fe4542aa3b46561d6d41fb2e4ee34f35551ecfd821c9577159e3",
        ),
        "authority_archive_size_bytes": 40960,
        "authority_archive_sha256": "bf19dd9477210adf106c7cbf2a72370ed4af22dedb577f361f3dc9e77e99baa4",
        "authority_file_count": 11,
        "authority_inventory_sha256": "35a14a021509b82f1238912e5c77ebb3559f9ee6daa3cc6c92db810b5dce5da0",
    },
    "vendor_runtime": {
        "relative_path": "vendor/rtdetrv2_pytorch",
        "file_count": 124,
        "directory_count_excluding_root": 25,
        "total_size_bytes": 373735,
        "compact_inventory_sha256": "0fc6803665bc4b5720e983345b2cacb0147eceead9f882588880b6f8a0e68051",
        "manifest": _file(
            "manifests/rtdetrv2_upstream.json",
            33264,
            "f65a2d475365346a5dd5ce4f46b022a135b187e21421e922cc41eee4d28d20ae",
        ),
        "manifest_inventory_sha256": "2312c80d5b0fba88d43ffc6807c3fc150ae74b77740f2ab65072f044e033d6d7",
    },
    "conversion_r3": {
        "relative_path": "artifacts/data/visdrone_protocol_v2_conversion_r3",
        "file_count": 25,
        "total_size_bytes": 304418794,
        "completion_sha256": "46734010937168ac65bf27c3a6f4bf3f234554d5d4a99407903dec4b3ae9e817",
        "artifact_inventory_sha256": "aaee01ce4e00749b9db8ac5e6e49cf35875a8b266e9c3efc237bdf0b8b109ae7",
        "entry_canonical_inventory_sha256": "ddf32745302d6095e1c3dde89b1a85f05db53e6b4cc8ee85638e0d45bfa8d982",
        "config_sha256": "58be8659d6d7ae6faace82b36d45523f80cd48bae1bbcc6da4586c53056ec0d0",
        "category_contract_sha256": "b4b309f357cbe130a505a610dff340cc498dc74f766384c4559b2acd900728a0",
        "source_identity_sha256": "f0f16ba4438b51a09a6203f78884b199f8e2a2d3fb46309c357368a47a552bb9",
    },
}

_FROZEN_ENVIRONMENT_POLICY = {
    "python": {
        "path_policy": "absolute_external_path_required",
        "realpath_policy": "regular_non_symlink_binary_required",
        "version": "3.10.16",
        "sha256": "7ed96f9b2f4d3da2c7c7233d9cb968163cfad01b02b2f2e6300cf036d769cfeb",
    },
    "tmux": {
        "path_policy": "absolute_external_path_required",
        "realpath_policy": "regular_non_symlink_binary_required",
        "version_policy": "exact_external_identity_required",
        "sha256_policy": "exact_external_identity_required",
    },
    "gpu": {
        "index": 0,
        "name": "NVIDIA GeForce RTX 4090 D",
        "uuid": "GPU-1faee6f0-1da7-4ede-2475-67a5a00274a8",
        "driver_policy": "exact_external_identity_required",
        "cuda_version": "12.4",
        "power_limit_watts": 425.0,
    },
    "filesystem": {
        "device_identity_required": True,
        "mount_identity_required": True,
        "target_parent_must_exist": True,
    },
}

_FROZEN_DATA_ROLE_POLICY = {
    "train_core": {
        "role": "train_core",
        "access": "allowed",
        "identity_source": "detached_owner_authorization",
        "manifest_required": True,
    },
    "development": {
        "role": "development",
        "access": "allowed",
        "identity_source": "detached_owner_authorization",
        "manifest_required": True,
    },
    "confirmatory": {
        "role": "confirmatory",
        "access": "sealed_and_forbidden",
        "identity_read": "forbidden",
    },
    "test": {
        "role": "test",
        "access": "forbidden",
        "identity_read": "forbidden",
    },
}

_FROZEN_TRAINING_POLICY = {
    "initialization": {
        "random_initialization": True,
        "pretrained": False,
        "checkpoint": None,
        "seed": 0,
    },
    "topology": {
        "world_size": 1,
        "device": "cuda:0",
        "train_batch_size": 16,
        "development_batch_size": 32,
    },
    "schedule": {
        "epochs": 120,
        "development_evaluation_every_epoch": True,
        "checkpoint_every_epoch": True,
    },
    "amp": {
        "enabled": True,
        "scaler_type": "GradScaler",
        "init_scale": 65536.0,
        "growth_factor": 2.0,
        "backoff_factor": 0.5,
        "growth_interval": 2000,
        "nonfinite_events_allowed": 0,
        "skipped_optimizer_steps_allowed": 0,
        "overflow_events_allowed": 0,
    },
    "ema": {
        "enabled": True,
        "decay": 0.9999,
        "warmup_optimizer_updates": 2000,
        "development_weights": "ema",
        "selection_weights": "ema",
    },
    "evaluator": {
        "primary_id": "visdrone_official_primary_evaluator_v1",
        "protocol_id": "visdrone_official_style_v1",
        "role": "development_only",
        "secondary_can_certify": False,
    },
    "checkpoint": {
        "roles": ["last", "best", "periodic", "final"],
        "required_atomicity": True,
        "required_loadability": True,
        "append_only": True,
    },
    "execution": {
        "overwrite": False,
        "resume": False,
        "retry": False,
        "fallback": False,
    },
}

_FROZEN_TARGETS = {
    "training_evidence_root": "artifacts/training/rtdetrv2_r18_visdrone_training_evidence_v1",
    "process_evidence_root": "artifacts/process_evidence/rtdetrv2_r18_visdrone_training_process_v1",
    "outer_evidence_root": "artifacts/outer_launch_evidence/rtdetrv2_r18_visdrone_training_process_v1",
    "all_targets_must_be_absent": True,
    "receipt_policy": "adjacent_exclusive_mode_0600",
    "lock_policy": "adjacent_exclusive_mode_0600",
}

_FROZEN_RUN_IDENTITY = {
    "training_run_id": "rtdetrv2_r18_visdrone_baseline_training_v1",
    "tmux_session_name": "p3_rtdetrv2_r18_visdrone_baseline_training_v1",
    "nonce_policy": {
        "field_is_external_only": True,
        "minimum_hex_characters": 32,
        "lowercase_hex_only": True,
        "all_zero_forbidden": True,
        "placeholder_and_low_entropy_forbidden": True,
        "must_be_unique_per_authorization": True,
    },
}

_FROZEN_AUTHORIZATION_POLICY = {
    "required": True,
    "source": "external_detached_owner_artifact",
    "in_git": False,
    "static_authorization_field": "production authorization requires detached owner artifact",
    "raw_encoding": "UTF-8",
    "raw_trailing_lf_count": 1,
    "canonical_trailing_lf": False,
    "wall_clock_expiry": False,
    "owner_file_policy": {
        "regular": True,
        "symlink": False,
        "mode": 384,
        "uid": 1000,
        "gid": 1000,
        "nlink": 1,
        "external_raw_size_sha_required": True,
    },
}

_FROZEN_LAUNCH_PROTOCOL = {
    "target_absence_required": True,
    "session_absence_required": True,
    "receipt_absence_required": True,
    "lock_absence_required": True,
    "outer_invocation_count": 1,
    "tmux_new_session_count": 1,
    "child_binding": ["pid", "nonce", "training_run_id", "argv", "environment", "cwd"],
    "stream_binding": ["outer_stdout", "outer_stderr", "child_stdout", "child_stderr"],
    "raw_bytes_not_text": True,
    "immediate_snapshot_required": True,
    "snapshot_before_terminal_observation": True,
    "no_polling_in_authorization_phase": True,
}

_FROZEN_STATE_MACHINE = {
    "states": [
        "DESIGN_ONLY",
        "OWNER_AUTHORIZED",
        "PREFLIGHT_PASS",
        "LAUNCH_ACCEPTED",
        "RUNNING",
        "TERMINAL_COMPLETE",
        "PERMANENT_FAIL",
        "INDEPENDENT_TERMINAL_AUDIT_PASS",
        "TRAINING_CERTIFIED",
    ],
    "terminal_success": "TERMINAL_COMPLETE",
    "terminal_failure": "PERMANENT_FAIL",
    "no_resume_after": "LAUNCH_ACCEPTED",
    "no_retry_after": "LAUNCH_ACCEPTED",
    "no_overwrite_after": "LAUNCH_ACCEPTED",
    "independent_audit_required": True,
    "training_certification_requires_terminal_complete": True,
}

_FROZEN_FAILURE_POLICY = {
    "permanent_failure_class": "PERMANENT_FAIL",
    "immutable": True,
    "repair_in_place": False,
    "resume": False,
    "retry": False,
    "overwrite": False,
    "fallback": False,
    "events": [
        "signal",
        "timeout",
        "host_reboot",
        "nonzero_exit",
        "partial_write",
        "identity_drift",
        "disk_full",
        "incomplete_evidence",
        "checkpoint_not_loadable",
        "checkpoint_inventory_drift",
    ],
}

_FROZEN_READINESS = {
    "owner_authorization_required": True,
    "launch_acceptance_separate": True,
    "terminal_completion_separate": True,
    "independent_audit_required": True,
    "training_certification_separate": True,
    "static_config_authorizes_production": False,
}


def _expected_config() -> dict[str, Any]:
    return {
        "schema_version": LAUNCH_CONTRACT_SCHEMA_VERSION,
        "contract_id": LAUNCH_CONTRACT_ID,
        "stage": "T6A",
        "status": "DETACHED_AUTHORIZATION_REQUIRED",
        "repository": copy.deepcopy(_FROZEN_REPOSITORY_REFERENCE),
        "run_identity": copy.deepcopy(_FROZEN_RUN_IDENTITY),
        "targets": copy.deepcopy(_FROZEN_TARGETS),
        "authorization": copy.deepcopy(_FROZEN_AUTHORIZATION_POLICY),
        "source_bindings": copy.deepcopy(_FROZEN_SOURCE_BINDINGS),
        "environment_identity": copy.deepcopy(_FROZEN_ENVIRONMENT_POLICY),
        "data_roles": copy.deepcopy(_FROZEN_DATA_ROLE_POLICY),
        "training_policy": copy.deepcopy(_FROZEN_TRAINING_POLICY),
        "launch_protocol": copy.deepcopy(_FROZEN_LAUNCH_PROTOCOL),
        "state_machine": copy.deepcopy(_FROZEN_STATE_MACHINE),
        "failure_policy": copy.deepcopy(_FROZEN_FAILURE_POLICY),
        "readiness": copy.deepcopy(_FROZEN_READINESS),
    }


_FROZEN_CONFIG = _expected_config()

__all__ = (
    "TrainingLaunchContractError",
    "canonical_training_launch_contract_bytes",
    "load_training_launch_contract",
    "validate_training_launch_contract",
    "training_launch_contract_binding",
    "canonical_owner_authorization_bytes",
    "validate_owner_authorization",
    "owner_authorization_binding",
)


class TrainingLaunchContractError(ValueError):
    """Raised when a T6A contract or detached authorization drifts."""


def _fail(message: str) -> None:
    raise TrainingLaunchContractError(message)


def _assert_builtin_json(value: Any, field: str = "value") -> None:
    if type(value) is dict:
        for key, child in value.items():
            if type(key) is not str:
                _fail(f"{field} contains a non-string key")
            _assert_builtin_json(child, f"{field}.{key}")
        return
    if type(value) is list:
        for index, child in enumerate(value):
            _assert_builtin_json(child, f"{field}[{index}]")
        return
    if type(value) not in {str, int, float, bool, type(None)}:
        _fail(f"{field} contains a non-builtin JSON scalar")
    if type(value) is float and not math.isfinite(value):
        _fail(f"{field} contains a non-finite float")


def _canonical(value: Any) -> bytes:
    _assert_builtin_json(value)
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise TrainingLaunchContractError("value is not canonical JSON") from exc


def _parse_json(raw: bytes, field: str, *, trailing_lf: bool) -> dict[str, Any]:
    if type(raw) is not bytes:
        _fail(f"{field} raw value must be bytes")
    if raw.startswith(b"\xef\xbb\xbf") or b"\x00" in raw or b"\r" in raw:
        _fail(f"{field} contains forbidden portable bytes")
    if trailing_lf:
        if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
            _fail(f"{field} must have exactly one trailing LF")
        payload = raw[:-1]
    else:
        if raw.endswith(b"\n"):
            _fail(f"{field} must not have a trailing LF")
        payload = raw

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in items:
            if key in result:
                _fail(f"{field} contains a duplicate key: {key}")
            result[key] = item
        return result

    def constant(value: str) -> None:
        _fail(f"{field} contains a non-finite JSON constant: {value}")

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=constant,
        )
    except TrainingLaunchContractError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise TrainingLaunchContractError(f"{field} is not strict UTF-8 JSON") from exc
    if type(value) is not dict:
        _fail(f"{field} root must be an object")
    _assert_builtin_json(value, field)
    return value


def _schema_for(value: Any) -> tuple[str, Any]:
    if type(value) is dict:
        return ("dict", {key: _schema_for(child) for key, child in value.items()})
    if type(value) is list:
        return ("list", tuple(_schema_for(child) for child in value))
    if value is None:
        return ("none", None)
    return ("scalar", type(value))


def _validate_schema(value: Any, schema: tuple[str, Any], field: str) -> None:
    kind, detail = schema
    if kind == "dict":
        if type(value) is not dict:
            _fail(f"{field} must be a builtin dict")
        expected = set(detail)
        actual = set(value)
        if expected - actual:
            _fail(f"{field} is missing keys: {sorted(expected - actual)}")
        if actual - expected:
            _fail(f"{field} has extra keys: {sorted(actual - expected)}")
        for key, child_schema in detail.items():
            _validate_schema(value[key], child_schema, f"{field}.{key}")
        return
    if kind == "list":
        if type(value) is not list or len(value) != len(detail):
            _fail(f"{field} has the wrong list shape")
        for index, child_schema in enumerate(detail):
            _validate_schema(value[index], child_schema, f"{field}[{index}]")
        return
    if kind == "none":
        if value is not None:
            _fail(f"{field} must be null")
        return
    if type(value) is not detail:
        _fail(f"{field} has type drift")


def _sha(value: Any, field: str) -> str:
    if type(value) is not str or _SHA_RE.fullmatch(value) is None:
        _fail(f"{field} is not a lowercase SHA-256")
    return value


def _git_oid(value: Any, field: str) -> str:
    if type(value) is not str or _GIT_RE.fullmatch(value) is None:
        _fail(f"{field} is not a Git object ID")
    return value


def _relative_path(value: Any, field: str) -> str:
    if type(value) is not str or not value or value.startswith("/") or "\\" in value:
        _fail(f"{field} is not a relative POSIX path")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        _fail(f"{field} contains a control character")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        _fail(f"{field} contains an unsafe path component")
    return value


def _absolute_path(value: Any, field: str) -> str:
    if type(value) is not str or not value.startswith("/") or "\\" in value or "//" in value:
        _fail(f"{field} is not an absolute canonical path")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        _fail(f"{field} contains a control character")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts[1:]):
        _fail(f"{field} contains an unsafe path component")
    return value


def _branch(value: Any, field: str) -> str:
    if type(value) is not str or not value or value.startswith("/") or value.endswith("/"):
        _fail(f"{field} is invalid")
    if any(part in {"", ".", ".."} for part in value.split("/")):
        _fail(f"{field} contains an unsafe component")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        _fail(f"{field} contains a control character")
    return value


def _nonce(value: Any, field: str) -> str:
    if type(value) is not str or _NONCE_RE.fullmatch(value) is None or len(value) % 2:
        _fail(f"{field} is not lowercase hexadecimal nonce data")
    if len(set(value)) < 8 or len(set(value[i : i + 4] for i in range(0, len(value), 4))) <= 2:
        _fail(f"{field} is low entropy")
    if set(value) == {"0"}:
        _fail(f"{field} is all zero")
    if len(set(value[i : i + 2] for i in range(0, len(value), 2))) == 1:
        _fail(f"{field} is repetitive")
    placeholders = {
        "0123456789abcdef" * (len(value) // 16),
        "abcdef0123456789" * (len(value) // 16),
    }
    if value in placeholders:
        _fail(f"{field} is a placeholder nonce")
    return value


def _validate_file_identity(value: Any, field: str) -> dict[str, Any]:
    expected = {"relative_path", "size_bytes", "sha256"}
    if type(value) is not dict or set(value) != expected:
        _fail(f"{field} key set drift")
    _relative_path(value["relative_path"], f"{field}.relative_path")
    if type(value["size_bytes"]) is not int or not 0 < value["size_bytes"] <= 2**63 - 1:
        _fail(f"{field}.size_bytes is invalid")
    _sha(value["sha256"], f"{field}.sha256")
    return copy.deepcopy(value)


def _validate_config_identity(value: Any, field: str) -> dict[str, Any]:
    expected = {
        "relative_path",
        "raw_size_bytes",
        "raw_sha256",
        "canonical_size_bytes",
        "canonical_sha256",
    }
    if type(value) is not dict or set(value) != expected:
        _fail(f"{field} key set drift")
    _relative_path(value["relative_path"], f"{field}.relative_path")
    for key in ("raw_size_bytes", "canonical_size_bytes"):
        if type(value[key]) is not int or not 0 < value[key] <= 2**63 - 1:
            _fail(f"{field}.{key} is invalid")
    _sha(value["raw_sha256"], f"{field}.raw_sha256")
    _sha(value["canonical_sha256"], f"{field}.canonical_sha256")
    return copy.deepcopy(value)


def _validate_config(value: Any) -> dict[str, Any]:
    _assert_builtin_json(value, "training launch contract")
    _validate_schema(value, _schema_for(_FROZEN_CONFIG), "training launch contract")
    if value != _FROZEN_CONFIG:
        _fail("training launch contract frozen semantic identity drift")
    if value["readiness"]["static_config_authorizes_production"] is not False:
        _fail("static config cannot authorize production")
    return copy.deepcopy(value)


def canonical_training_launch_contract_bytes(config: Any) -> bytes:
    """Return deterministic compact JSON bytes without a trailing LF."""

    return _canonical(_validate_config(config))


def _repo_root(value: str | os.PathLike[str]) -> Path:
    try:
        raw = os.fspath(value)
    except TypeError as exc:
        raise TrainingLaunchContractError("repo_root is not a path") from exc
    if type(raw) is not str or not raw.startswith("/") or "//" in raw or "/../" in raw or raw.endswith("/.."):
        _fail("repo_root must be an absolute canonical path")
    root = Path(raw)
    try:
        resolved = root.resolve(strict=True)
    except OSError as exc:
        raise TrainingLaunchContractError("repo_root does not exist") from exc
    if str(resolved) != raw or not resolved.is_dir():
        _fail("repo_root is not a canonical directory")
    return resolved


def _regular_file(root: Path, relative: str, field: str) -> Path:
    _relative_path(relative, field)
    path = root / relative
    try:
        observed = path.lstat()
    except OSError as exc:
        raise TrainingLaunchContractError(f"{field} is unavailable") from exc
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
        _fail(f"{field} is not a regular non-symlink file")
    return path


def _verify_file_identity(root: Path, identity: Mapping[str, Any], field: str) -> None:
    checked = _validate_file_identity(dict(identity), field)
    path = _regular_file(root, checked["relative_path"], field)
    raw = path.read_bytes()
    if len(raw) != checked["size_bytes"] or hashlib.sha256(raw).hexdigest() != checked["sha256"]:
        _fail(f"{field} identity drift")


def _verify_config_file_identity(root: Path, identity: Mapping[str, Any], field: str) -> None:
    checked = _validate_config_identity(dict(identity), field)
    path = _regular_file(root, checked["relative_path"], field)
    raw = path.read_bytes()
    if len(raw) != checked["raw_size_bytes"] or hashlib.sha256(raw).hexdigest() != checked["raw_sha256"]:
        _fail(f"{field} raw identity drift")
    parsed = _parse_json(raw, field, trailing_lf=True)
    canonical = _canonical(parsed)
    if len(canonical) != checked["canonical_size_bytes"] or hashlib.sha256(canonical).hexdigest() != checked["canonical_sha256"]:
        _fail(f"{field} canonical identity drift")


def _verify_source_bindings(root: Path, bindings: Mapping[str, Any]) -> None:
    for role in ("training_contract", "runtime_plan", "training_evidence", "process"):
        item = bindings[role]
        _verify_config_file_identity(root, item["config"], f"source_bindings.{role}.config")
        children = ("entry_module", "launcher_module", "document") if role == "process" else ("module", "document")
        for child in children:
            _verify_file_identity(root, item[child], f"source_bindings.{role}.{child}")
    item = bindings["prepared_adapter"]
    _verify_file_identity(root, item["module"], "source_bindings.prepared_adapter.module")
    _verify_file_identity(root, item["document"], "source_bindings.prepared_adapter.document")
    evaluator = bindings["primary_evaluator"]
    for child in ("implementation", "test", "document"):
        _verify_file_identity(root, evaluator[child], f"source_bindings.primary_evaluator.{child}")
    _verify_config_file_identity(root, evaluator["config"], "source_bindings.primary_evaluator.config")
    _verify_config_file_identity(root, evaluator["authority_manifest"], "source_bindings.primary_evaluator.authority_manifest")
    _verify_file_identity(root, bindings["vendor_runtime"]["manifest"], "source_bindings.vendor_runtime.manifest")


def load_training_launch_contract(
    repo_root: str | os.PathLike[str],
    config_path: str | os.PathLike[str] = LAUNCH_CONTRACT_CONFIG_RELATIVE_PATH,
) -> dict[str, Any]:
    """Load and validate the checked-in T6A contract without side effects."""

    try:
        relative = os.fspath(config_path)
    except TypeError as exc:
        raise TrainingLaunchContractError("training launch config path is invalid") from exc
    if type(relative) is not str or relative != LAUNCH_CONTRACT_CONFIG_RELATIVE_PATH:
        _fail("training launch config path drift")
    root = _repo_root(repo_root)
    path = _regular_file(root, relative, "training launch config")
    raw = path.read_bytes()
    if len(raw) != LAUNCH_CONTRACT_RAW_SIZE_BYTES or hashlib.sha256(raw).hexdigest() != LAUNCH_CONTRACT_RAW_SHA256:
        _fail("training launch config raw identity drift")
    value = _parse_json(raw, "training launch config", trailing_lf=True)
    return _validate_config(value)


def validate_training_launch_contract(config: Any) -> dict[str, Any]:
    """Validate detached static T6A contract data."""

    return _validate_config(config)


def training_launch_contract_binding(
    repo_root: str | os.PathLike[str],
    config_path: str | os.PathLike[str] = LAUNCH_CONTRACT_CONFIG_RELATIVE_PATH,
) -> dict[str, Any]:
    """Return the contract and its checked-in dependency identities."""

    root = _repo_root(repo_root)
    contract = load_training_launch_contract(root, config_path)
    _verify_source_bindings(root, contract["source_bindings"])
    canonical = _canonical(contract)
    identity = {
        "relative_path": LAUNCH_CONTRACT_CONFIG_RELATIVE_PATH,
        "raw_size_bytes": LAUNCH_CONTRACT_RAW_SIZE_BYTES,
        "raw_sha256": LAUNCH_CONTRACT_RAW_SHA256,
        "canonical_size_bytes": len(canonical),
        "canonical_sha256": hashlib.sha256(canonical).hexdigest(),
    }
    return {
        "schema_version": LAUNCH_CONTRACT_SCHEMA_VERSION,
        "contract_id": LAUNCH_CONTRACT_ID,
        "relative_path": LAUNCH_CONTRACT_CONFIG_RELATIVE_PATH,
        "raw_size_bytes": LAUNCH_CONTRACT_RAW_SIZE_BYTES,
        "raw_sha256": LAUNCH_CONTRACT_RAW_SHA256,
        "canonical_size_bytes": len(canonical),
        "canonical_sha256": hashlib.sha256(canonical).hexdigest(),
        "contract_identity": identity,
        "source_bindings": copy.deepcopy(contract["source_bindings"]),
        "targets": copy.deepcopy(contract["targets"]),
        "run_identity": copy.deepcopy(contract["run_identity"]),
        "repository_reference": copy.deepcopy(contract["repository"]),
        "contract": copy.deepcopy(contract),
    }


def _validate_target_absence(value: Any, field: str) -> dict[str, Any]:
    expected = {
        "training_evidence_root",
        "process_evidence_root",
        "outer_evidence_root",
        "tmux_session",
        "receipt",
        "lock",
    }
    if type(value) is not dict or set(value) != expected:
        _fail(f"{field} key set drift")
    if any(type(value[key]) is not bool or value[key] is not True for key in expected):
        _fail(f"{field} must prove every target is absent")
    return copy.deepcopy(value)


def _validate_repository_observation(value: Any, field: str) -> dict[str, Any]:
    expected = {"repo_root", "branch", "head", "tree", "parent", "upstream", "upstream_sha"}
    if type(value) is not dict or set(value) != expected:
        _fail(f"{field} key set drift")
    _absolute_path(value["repo_root"], f"{field}.repo_root")
    _branch(value["branch"], f"{field}.branch")
    for key in ("head", "tree", "parent", "upstream_sha"):
        _git_oid(value[key], f"{field}.{key}")
    if type(value["upstream"]) is not str or not value["upstream"].startswith("origin/"):
        _fail(f"{field}.upstream is invalid")
    if value["branch"] != _T6A_LAUNCH_BRANCH:
        _fail(f"{field} still identifies the pre-T6 branch")
    if value["head"] == _FROZEN_REPOSITORY_REFERENCE["head"] or value["tree"] == _FROZEN_REPOSITORY_REFERENCE["tree"]:
        _fail(f"{field} still identifies the pre-T6 reference")
    return copy.deepcopy(value)


def _validate_environment_observation(value: Any) -> dict[str, Any]:
    expected = {"python", "tmux", "gpu", "filesystem"}
    if type(value) is not dict or set(value) != expected:
        _fail("environment_identity key set drift")
    for name in ("python", "tmux"):
        executable = value[name]
        if type(executable) is not dict or set(executable) != {"path", "realpath", "version", "sha256"}:
            _fail(f"environment_identity.{name} key set drift")
        _absolute_path(executable["path"], f"environment_identity.{name}.path")
        _absolute_path(executable["realpath"], f"environment_identity.{name}.realpath")
        if type(executable["version"]) is not str or not executable["version"]:
            _fail(f"environment_identity.{name}.version is invalid")
        _sha(executable["sha256"], f"environment_identity.{name}.sha256")
    if value["python"]["version"] != _FROZEN_ENVIRONMENT_POLICY["python"]["version"]:
        _fail("environment_identity.python.version drift")
    gpu = value["gpu"]
    if type(gpu) is not dict or set(gpu) != {"index", "name", "uuid", "driver_version", "cuda_version", "power_limit_watts"}:
        _fail("environment_identity.gpu key set drift")
    if type(gpu["index"]) is not int or gpu["index"] != 0:
        _fail("environment_identity.gpu.index drift")
    if type(gpu["name"]) is not str or gpu["name"] != _FROZEN_ENVIRONMENT_POLICY["gpu"]["name"]:
        _fail("environment_identity.gpu.name drift")
    if type(gpu["uuid"]) is not str or _UUID_RE.fullmatch(gpu["uuid"]) is None or gpu["uuid"] != _FROZEN_ENVIRONMENT_POLICY["gpu"]["uuid"]:
        _fail("environment_identity.gpu.uuid drift")
    if type(gpu["driver_version"]) is not str or not gpu["driver_version"]:
        _fail("environment_identity.gpu.driver_version is invalid")
    if type(gpu["cuda_version"]) is not str or gpu["cuda_version"] != "12.4":
        _fail("environment_identity.gpu.cuda_version drift")
    if type(gpu["power_limit_watts"]) is not float or gpu["power_limit_watts"] != 425.0:
        _fail("environment_identity.gpu.power_limit_watts drift")
    filesystem = value["filesystem"]
    if type(filesystem) is not dict or set(filesystem) != {"device", "mount"}:
        _fail("environment_identity.filesystem key set drift")
    if type(filesystem["device"]) is not int or filesystem["device"] < 0:
        _fail("environment_identity.filesystem.device is invalid")
    _absolute_path(filesystem["mount"], "environment_identity.filesystem.mount")
    return copy.deepcopy(value)


def _validate_data_observation(value: Any) -> dict[str, Any]:
    expected = {"train_core", "development", "confirmatory", "test"}
    if type(value) is not dict or set(value) != expected:
        _fail("data_roles key set drift")
    paths: list[str] = []
    for role in ("train_core", "development"):
        item = value[role]
        if type(item) is not dict or set(item) != {"role", "root", "manifest_sha256"}:
            _fail(f"data_roles.{role} key set drift")
        if item["role"] != role:
            _fail(f"data_roles.{role}.role drift")
        _absolute_path(item["root"], f"data_roles.{role}.root")
        parts = {part.casefold() for part in item["root"].split("/") if part}
        if (
            parts & {"test", "confirmatory", "raw", "raw_annotation", "annotations"}
            or any(
                part.startswith(("test_", "test-", "confirmatory_", "confirmatory-"))
                or "annotation" in part
                for part in parts
            )
        ):
            _fail(f"data_roles.{role} points into sealed data")
        _sha(item["manifest_sha256"], f"data_roles.{role}.manifest_sha256")
        paths.append(item["root"])
    if paths[0] == paths[1]:
        _fail("train_core and development roots must be distinct")
    for role, expected_item in (
        ("confirmatory", {"role": "confirmatory", "access": "sealed_and_forbidden", "identity_read": "forbidden"}),
        ("test", {"role": "test", "access": "forbidden", "identity_read": "forbidden"}),
    ):
        if type(value[role]) is not dict or value[role] != expected_item:
            _fail(f"data_roles.{role} is not sealed")
    return copy.deepcopy(value)


def _validate_authorization_file_identity(value: Any) -> dict[str, Any]:
    expected = {"regular", "symlink", "mode", "uid", "gid", "nlink"}
    if type(value) is not dict or set(value) != expected:
        _fail("authorization file identity key set drift")
    if type(value["regular"]) is not bool or value["regular"] is not True:
        _fail("authorization artifact is not regular")
    if type(value["symlink"]) is not bool or value["symlink"] is not False:
        _fail("authorization artifact is a symlink")
    for key, expected_value in (("mode", 384), ("uid", 1000), ("gid", 1000), ("nlink", 1)):
        if type(value[key]) is not int or value[key] != expected_value:
            _fail(f"authorization artifact {key} drift")
    return copy.deepcopy(value)


def _validate_owner_authorization_shape(value: Any) -> dict[str, Any]:
    if type(value) is not dict:
        _fail("owner authorization must be a builtin dict")
    expected = {
        "schema_version",
        "authorization_id",
        "authorized",
        "contract_id",
        "training_run_id",
        "nonce",
        "tmux_session_name",
        "targets",
        "target_absence",
        "repository",
        "contract_identity",
        "source_bindings",
        "environment_identity",
        "data_roles",
        "training_policy",
        "state_machine",
        "owner",
    }
    if set(value) != expected:
        _fail("owner authorization key set drift")
    _assert_builtin_json(value, "owner authorization")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        _fail("owner authorization schema version drift")
    if value["authorization_id"] != "rtdetrv2_r18_visdrone_training_owner_authorization_v1":
        _fail("owner authorization ID drift")
    if type(value["authorized"]) is not bool or value["authorized"] is not True:
        _fail("owner authorization is not explicitly authorized")
    if type(value["contract_id"]) is not str or value["contract_id"] != LAUNCH_CONTRACT_ID:
        _fail("owner authorization contract ID drift")
    if value["training_run_id"] != _FROZEN_RUN_IDENTITY["training_run_id"]:
        _fail("owner authorization run ID drift")
    _nonce(value["nonce"], "owner authorization nonce")
    if value["tmux_session_name"] != _FROZEN_RUN_IDENTITY["tmux_session_name"]:
        _fail("owner authorization session drift")
    if type(value["targets"]) is not dict or value["targets"] != _FROZEN_TARGETS:
        _fail("owner authorization target drift")
    _validate_target_absence(value["target_absence"], "owner authorization target_absence")
    _validate_repository_observation(value["repository"], "owner authorization repository")
    _validate_config_identity(value["contract_identity"], "owner authorization contract_identity")
    if type(value["source_bindings"]) is not dict or value["source_bindings"] != _FROZEN_SOURCE_BINDINGS:
        _fail("owner authorization source binding drift")
    _validate_environment_observation(value["environment_identity"])
    _validate_data_observation(value["data_roles"])
    if value["training_policy"] != _FROZEN_TRAINING_POLICY:
        _fail("owner authorization training policy drift")
    if value["state_machine"] != _FROZEN_STATE_MACHINE:
        _fail("owner authorization state machine drift")
    owner = value["owner"]
    expected_owner = {
        "owner_id": "project_owner",
        "uid": 1000,
        "gid": 1000,
        "detached_external": True,
        "self_issued": False,
    }
    if type(owner) is not dict or set(owner) != set(expected_owner) or owner != expected_owner:
        _fail("owner authorization owner identity drift")
    return copy.deepcopy(value)


def _authorization_raw(value: bytes | Mapping[str, Any]) -> tuple[dict[str, Any], bytes, bytes]:
    if type(value) is bytes:
        parsed = _parse_json(value, "owner authorization", trailing_lf=True)
        canonical = _canonical(_validate_owner_authorization_shape(parsed))
        if value != canonical + b"\n":
            _fail("owner authorization raw bytes are not canonical plus one LF")
        return parsed, value, canonical
    if type(value) is dict:
        parsed = _validate_owner_authorization_shape(copy.deepcopy(value))
        canonical = _canonical(parsed)
        return parsed, canonical + b"\n", canonical
    _fail("owner authorization must be bytes or a builtin dict")


def canonical_owner_authorization_bytes(authorization: Mapping[str, Any]) -> bytes:
    """Return canonical detached authorization bytes without a trailing LF."""

    if type(authorization) is not dict:
        _fail("owner authorization must be a builtin dict")
    return _canonical(_validate_owner_authorization_shape(copy.deepcopy(authorization)))


def _validate_training_launch_contract_binding(value: Any) -> dict[str, Any]:
    """Validate the complete immutable binding emitted by the factory."""

    expected = {
        "schema_version",
        "contract_id",
        "relative_path",
        "raw_size_bytes",
        "raw_sha256",
        "canonical_size_bytes",
        "canonical_sha256",
        "contract_identity",
        "source_bindings",
        "targets",
        "run_identity",
        "repository_reference",
        "contract",
    }
    if type(value) is not dict or set(value) != expected:
        _fail("contract binding key set drift")
    _assert_builtin_json(value, "contract binding")
    if type(value["schema_version"]) is not int or value["schema_version"] != LAUNCH_CONTRACT_SCHEMA_VERSION:
        _fail("contract binding schema version drift")
    if type(value["contract_id"]) is not str or value["contract_id"] != LAUNCH_CONTRACT_ID:
        _fail("contract binding contract ID drift")

    flattened_identity = {
        "relative_path": value["relative_path"],
        "raw_size_bytes": value["raw_size_bytes"],
        "raw_sha256": value["raw_sha256"],
        "canonical_size_bytes": value["canonical_size_bytes"],
        "canonical_sha256": value["canonical_sha256"],
    }
    checked_identity = _validate_config_identity(
        value["contract_identity"], "contract binding contract_identity"
    )
    expected_identity = {
        "relative_path": LAUNCH_CONTRACT_CONFIG_RELATIVE_PATH,
        "raw_size_bytes": LAUNCH_CONTRACT_RAW_SIZE_BYTES,
        "raw_sha256": LAUNCH_CONTRACT_RAW_SHA256,
        "canonical_size_bytes": LAUNCH_CONTRACT_CANONICAL_SIZE_BYTES,
        "canonical_sha256": LAUNCH_CONTRACT_CANONICAL_SHA256,
    }
    if checked_identity != expected_identity:
        _fail("contract binding contract identity drift")
    if flattened_identity != checked_identity:
        _fail("contract binding flattened identity drift")

    contract = _validate_config(copy.deepcopy(value["contract"]))
    for field, frozen in (
        ("source_bindings", _FROZEN_SOURCE_BINDINGS),
        ("targets", _FROZEN_TARGETS),
        ("run_identity", _FROZEN_RUN_IDENTITY),
        ("repository_reference", _FROZEN_REPOSITORY_REFERENCE),
    ):
        child = value[field]
        if type(child) is not dict:
            _fail(f"contract binding {field} must be a builtin dict")
        _assert_builtin_json(child, f"contract binding.{field}")
        if child != frozen:
            _fail(f"contract binding {field} drift")

    if value["source_bindings"] != contract["source_bindings"]:
        _fail("contract binding source_bindings disagree with contract")
    if value["targets"] != contract["targets"]:
        _fail("contract binding targets disagree with contract")
    if value["run_identity"] != contract["run_identity"]:
        _fail("contract binding run_identity disagrees with contract")
    if value["repository_reference"] != contract["repository"]:
        _fail("contract binding repository_reference disagrees with contract")
    return copy.deepcopy(value)


def _contract_identity_from_binding(value: Any) -> dict[str, Any]:
    return _validate_training_launch_contract_binding(value)["contract_identity"]


def _validate_observed_binding(value: Any) -> dict[str, Any]:
    expected = {
        "repository",
        "targets",
        "target_absence",
        "contract_identity",
        "source_bindings",
        "environment_identity",
        "data_roles",
        "training_policy",
        "state_machine",
        "training_run_id",
        "tmux_session_name",
        "nonce",
    }
    if type(value) is not dict or set(value) != expected:
        _fail("observed launch binding key set drift")
    repository = _validate_repository_observation(value["repository"], "observed.repository")
    if value["targets"] != _FROZEN_TARGETS:
        _fail("observed target binding drift")
    target_absence = _validate_target_absence(value["target_absence"], "observed.target_absence")
    contract_identity = _validate_config_identity(value["contract_identity"], "observed.contract_identity")
    expected_contract_identity = {
        "relative_path": LAUNCH_CONTRACT_CONFIG_RELATIVE_PATH,
        "raw_size_bytes": LAUNCH_CONTRACT_RAW_SIZE_BYTES,
        "raw_sha256": LAUNCH_CONTRACT_RAW_SHA256,
        "canonical_size_bytes": LAUNCH_CONTRACT_CANONICAL_SIZE_BYTES,
        "canonical_sha256": LAUNCH_CONTRACT_CANONICAL_SHA256,
    }
    if contract_identity != expected_contract_identity:
        _fail("observed contract identity drift")
    if value["source_bindings"] != _FROZEN_SOURCE_BINDINGS:
        _fail("observed source binding drift")
    environment = _validate_environment_observation(value["environment_identity"])
    data_roles = _validate_data_observation(value["data_roles"])
    if value["training_policy"] != _FROZEN_TRAINING_POLICY or value["state_machine"] != _FROZEN_STATE_MACHINE:
        _fail("observed policy binding drift")
    if value["training_run_id"] != _FROZEN_RUN_IDENTITY["training_run_id"]:
        _fail("observed run identity drift")
    if value["tmux_session_name"] != _FROZEN_RUN_IDENTITY["tmux_session_name"]:
        _fail("observed session identity drift")
    nonce = _nonce(value["nonce"], "observed nonce")
    return {
        "repository": repository,
        "targets": copy.deepcopy(value["targets"]),
        "target_absence": target_absence,
        "contract_identity": contract_identity,
        "source_bindings": copy.deepcopy(value["source_bindings"]),
        "environment_identity": environment,
        "data_roles": data_roles,
        "training_policy": copy.deepcopy(value["training_policy"]),
        "state_machine": copy.deepcopy(value["state_machine"]),
        "training_run_id": value["training_run_id"],
        "tmux_session_name": value["tmux_session_name"],
        "nonce": nonce,
    }


def validate_owner_authorization(
    authorization: bytes | Mapping[str, Any],
    *,
    expected_raw_size_bytes: int,
    expected_raw_sha256: str,
    contract_binding: Mapping[str, Any],
    observed_binding: Mapping[str, Any],
    authorization_file_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate detached authorization against external launch-time facts."""

    if type(expected_raw_size_bytes) is not int or expected_raw_size_bytes <= 0:
        _fail("expected authorization raw size is invalid")
    _sha(expected_raw_sha256, "expected authorization raw SHA-256")
    if type(contract_binding) is not dict:
        _fail("contract binding must be a builtin dict")
    if type(observed_binding) is not dict:
        _fail("observed binding must be a builtin dict")
    if type(authorization_file_identity) is not dict:
        _fail("authorization file identity must be a builtin dict")
    validated_binding = _validate_training_launch_contract_binding(copy.deepcopy(contract_binding))
    observed = _validate_observed_binding(copy.deepcopy(observed_binding))
    file_identity = _validate_authorization_file_identity(copy.deepcopy(authorization_file_identity))
    parsed, raw, _ = _authorization_raw(authorization)
    checked = _validate_owner_authorization_shape(parsed)
    if len(raw) != expected_raw_size_bytes or hashlib.sha256(raw).hexdigest() != expected_raw_sha256:
        _fail("authorization external raw identity mismatch")
    if checked["contract_id"] != validated_binding["contract_id"]:
        _fail("authorization contract ID does not match external contract binding")
    if checked["contract_identity"] != validated_binding["contract_identity"]:
        _fail("authorization contract identity does not match external contract binding")
    if checked["source_bindings"] != validated_binding["source_bindings"]:
        _fail("authorization source bindings do not match external contract binding")
    if checked["targets"] != validated_binding["targets"]:
        _fail("authorization targets do not match external contract binding")
    if checked["training_run_id"] != validated_binding["run_identity"]["training_run_id"]:
        _fail("authorization run ID does not match external contract binding")
    if checked["tmux_session_name"] != validated_binding["run_identity"]["tmux_session_name"]:
        _fail("authorization session does not match external contract binding")
    if checked["training_policy"] != validated_binding["contract"]["training_policy"]:
        _fail("authorization training policy does not match external contract binding")
    if checked["state_machine"] != validated_binding["contract"]["state_machine"]:
        _fail("authorization state machine does not match external contract binding")
    for field in (
        "repository",
        "targets",
        "target_absence",
        "source_bindings",
        "environment_identity",
        "data_roles",
        "training_policy",
        "state_machine",
        "training_run_id",
        "tmux_session_name",
        "nonce",
    ):
        if checked[field] != observed[field]:
            _fail(f"authorization {field} does not match launch-time observation")
    # Keep this validation explicit: the caller must prove the detached object,
    # while this pure module never stats or opens the authorization path.
    if file_identity != _validate_authorization_file_identity(file_identity):
        _fail("authorization file identity is unstable")
    return copy.deepcopy(checked)


def owner_authorization_binding(
    authorization: bytes | Mapping[str, Any],
    *,
    expected_raw_size_bytes: int,
    expected_raw_sha256: str,
    contract_binding: Mapping[str, Any],
    observed_binding: Mapping[str, Any],
    authorization_file_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the verified detached authorization identity."""

    validated = validate_owner_authorization(
        authorization,
        expected_raw_size_bytes=expected_raw_size_bytes,
        expected_raw_sha256=expected_raw_sha256,
        contract_binding=contract_binding,
        observed_binding=observed_binding,
        authorization_file_identity=authorization_file_identity,
    )
    canonical = canonical_owner_authorization_bytes(validated)
    raw = canonical + b"\n"
    return {
        "schema_version": 1,
        "authorization_id": validated["authorization_id"],
        "authorized": True,
        "raw_size_bytes": len(raw),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "canonical_size_bytes": len(canonical),
        "canonical_sha256": hashlib.sha256(canonical).hexdigest(),
        "authorization": copy.deepcopy(validated),
        "authorization_file_identity": copy.deepcopy(dict(authorization_file_identity)),
    }
