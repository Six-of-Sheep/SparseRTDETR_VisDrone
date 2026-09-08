"""T6B production entry boundary.

This module is deliberately inert at import time.  It validates a detached
T6A authorization, consumes that authorization through an exclusive receipt,
and runs the production engine only after the process boundary has published
its invocation record.  The public API is also usable with CPU fake ports;
those ports never replace the detached authorization or source gates.
"""

from __future__ import annotations

import copy
import datetime as _datetime
import hashlib
import json
import math
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from sparse_rtdetr.baseline import training_t6_authorization as _t6a
from sparse_rtdetr.baseline import training_t6_engine as _engine


T6B_SCHEMA_VERSION = 1
T6B_CONTRACT_ID = "rtdetrv2_r18_visdrone_baseline_training_t6b_v1"
ENTRY_ID = "rtdetrv2_r18_visdrone_training_t6_entry_v1"
ENTRY_MODE = "production"
ENTRY_MODULE_RELATIVE_PATH = "src/sparse_rtdetr/baseline/training_t6_entry.py"
PROCESS_MODULE_RELATIVE_PATH = "src/sparse_rtdetr/baseline/training_t6_process_launcher.py"
OUTER_MODULE_RELATIVE_PATH = "src/sparse_rtdetr/baseline/training_t6_outer_launcher.py"
ENGINE_MODULE_RELATIVE_PATH = "src/sparse_rtdetr/baseline/training_t6_engine.py"
CONFIG_RELATIVE_PATH = "configs/baseline/rtdetrv2_r18_visdrone_training_t6b_v1.json"
PROCESS_RECEIPT_SUFFIX = ".t6b-process-receipt.json"
AUTHORIZATION_RECEIPT_SUFFIX = ".t6b-consumed.json"
EVIDENCE_RECEIPT_SUFFIX = ".t6b-entry-evidence-receipt.json"
PERSISTENCE_FAILURE_MARKER_SUFFIX = ".t6b-entry-persistence-failure.json"
ENTRY_EVIDENCE_FILE_NAMES = (
    "entry_consumption.json",
    "entry_invocation.json",
    "engine_execution_claim.json",
    "entry_result.json",
    "entry_artifact_inventory.json",
    "entry_completion.json",
)
ENTRY_REQUIRED_EVIDENCE_FILE_NAMES = (
    "entry_consumption.json",
    "entry_invocation.json",
    "entry_result.json",
    "entry_artifact_inventory.json",
    "entry_completion.json",
)
ENTRY_CHECKPOINT_DIRECTORY = "checkpoints"

# Updated after the config is added.  Keeping these constants beside the
# loader makes a checked-in raw/canonical identity part of the API contract.
CONFIG_RAW_SIZE_BYTES = 6925
CONFIG_RAW_SHA256 = "a152ccb7caefd42531ee2126ca194495e4af8d4a660d907c5a9acf796fefcc85"
CONFIG_CANONICAL_SIZE_BYTES = 5565
CONFIG_CANONICAL_SHA256 = "dfeecc7b002db9ce10b33afa9d166fab4c7f88a5f922b4862fa0d1d99ff755a1"

TARGET_RELATIVE_PATHS = {
    "training_evidence_root": "artifacts/training/rtdetrv2_r18_visdrone_training_t6b_v1",
    "process_evidence_root": "artifacts/process_evidence/rtdetrv2_r18_visdrone_training_t6b_v1",
    "outer_evidence_root": "artifacts/outer_launch_evidence/rtdetrv2_r18_visdrone_training_t6b_v1",
}
ENVIRONMENT_KEYS = ("CUDA_VISIBLE_DEVICES", "PYTHONNOUSERSITE", "PYTHONDONTWRITEBYTECODE", "PYTHONPATH")
STATE_SEQUENCE = (
    "DESIGN_ONLY",
    "OWNER_AUTHORIZED",
    "PREFLIGHT_PASS",
    "LAUNCH_ACCEPTED",
    "RUNNING",
    "TERMINAL_COMPLETE",
    "PERMANENT_FAIL",
    "INDEPENDENT_TERMINAL_AUDIT_PASS",
    "TRAINING_CERTIFIED",
)
PERMANENT_FAILURE_CLASSES = (
    "AUTHORIZATION_FAILURE",
    "IDENTITY_DRIFT",
    "TARGET_ABSENCE_FAILURE",
    "RUNNER_EXCEPTION",
    "RUNNER_TIMEOUT",
    "RUNNER_SIGNAL",
    "RUNNER_NONZERO_EXIT",
    "ENGINE_FAILURE",
    "EVIDENCE_FAILURE",
    "CHECKPOINT_FAILURE",
    "PARTIAL_WRITE",
)

DESCRIPTOR_KEYS = {
    "schema_version",
    "contract_id",
    "entry_id",
    "mode",
    "training_run_id",
    "nonce",
    "tmux_session_name",
    "repository",
    "t6a_contract_binding",
    "authorization_path",
    "authorization_binding",
    "authorization_binding_sha256",
    "authorization_receipt_path",
    "evidence_receipt_path",
    "config_identity",
    "source_bindings",
    "environment",
    "data_roles",
    "training_policy",
    "targets",
    "cwd",
    "argv",
    "child_environment",
    "state_machine",
    "invocation_policy",
    "evidence_root",
    "process_evidence_root",
    "outer_evidence_root",
    "aggregate_sha256",
}
RESULT_KEYS = {
    "schema_version",
    "contract_id",
    "entry_id",
    "mode",
    "training_run_id",
    "nonce",
    "descriptor_sha256",
    "authorization_binding_sha256",
    "authorization_receipt_path",
    "evidence_receipt_path",
    "repository",
    "source_bindings",
    "environment",
    "data_roles",
    "training_policy",
    "training_evidence_root",
    "process_evidence_root",
    "outer_evidence_root",
    "return_code",
    "engine_result",
    "status",
    "classification",
    "failure_class",
    "production_training_authorized",
    "real_process_launch_authorized",
    "entry_receipt_sha256",
    "aggregate_result_sha256",
}

__all__ = (
    "TrainingEntryError",
    "load_t6b_config",
    "t6b_config_binding",
    "current_source_bindings",
    "authorization_file_identity",
    "validate_detached_authorization",
    "consume_owner_authorization",
    "validate_consumed_authorization_receipt",
    "build_production_entry_descriptor",
    "validate_entry_descriptor",
    "run_production_entry",
    "validate_entry_result",
    "classify_entry",
)


class TrainingEntryError(ValueError):
    """Raised when the T6B entry contract or evidence is invalid."""


def _fail(message: str) -> None:
    raise TrainingEntryError(message)


def _assert_builtin(value: Any, field: str = "value") -> None:
    if type(value) is dict:
        for key, child in value.items():
            if type(key) is not str:
                _fail(f"{field} has a non-string key")
            _assert_builtin(child, f"{field}.{key}")
        return
    if type(value) is list:
        for index, child in enumerate(value):
            _assert_builtin(child, f"{field}[{index}]")
        return
    if type(value) is float and not math.isfinite(value):
        _fail(f"{field} is non-finite")
    if type(value) not in {str, int, float, bool, type(None)}:
        _fail(f"{field} is not a builtin JSON value")


def _canonical(value: Any) -> bytes:
    _assert_builtin(value, "canonical value")
    try:
        return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise TrainingEntryError("value is not canonical JSON") from exc


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _digest(value: Any) -> str:
    return _sha(_canonical(value))


def _copy(value: Any) -> Any:
    return copy.deepcopy(value)


def _exact(value: Any, keys: set[str], field: str) -> dict[str, Any]:
    if type(value) is not dict:
        _fail(f"{field} must be a builtin dict")
    actual = set(value)
    if actual != keys:
        _fail(f"{field} key set drift: missing={sorted(keys - actual)} extra={sorted(actual - keys)}")
    return value


def _strict_equal(actual: Any, expected: Any, field: str) -> None:
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
        for index, (left, right) in enumerate(zip(actual, expected)):
            _strict_equal(left, right, f"{field}[{index}]")
        return
    if actual != expected:
        _fail(f"{field} value drift")


def _string(value: Any, field: str, *, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value):
        _fail(f"{field} must be a builtin string")
    return value


def _sha_string(value: Any, field: str) -> str:
    value = _string(value, field)
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        _fail(f"{field} is not a lowercase SHA-256")
    return value


def _integer(value: Any, field: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    if type(value) is not int:
        _fail(f"{field} must be a builtin int")
    if minimum is not None and value < minimum:
        _fail(f"{field} is below its minimum")
    if maximum is not None and value > maximum:
        _fail(f"{field} is above its maximum")
    return value


def _bool(value: Any, field: str) -> bool:
    if type(value) is not bool:
        _fail(f"{field} must be a builtin bool")
    return value


def _absolute_path(value: Any, field: str) -> str:
    value = _string(value, field)
    if not value.startswith("/") or "\\" in value or "//" in value or "\x00" in value:
        _fail(f"{field} is not an absolute canonical path")
    if any(part in {"", ".", ".."} for part in value.split("/")[1:]):
        _fail(f"{field} has unsafe path components")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        _fail(f"{field} has a control character")
    return value


def _relative_path(value: Any, field: str) -> str:
    value = _string(value, field)
    if value.startswith("/") or "\\" in value or "//" in value or "\x00" in value or value.endswith("/"):
        _fail(f"{field} is not a normalized relative path")
    if any(part in {"", ".", ".."} for part in value.split("/")):
        _fail(f"{field} has unsafe path components")
    return value


def _canonical_root(value: str | os.PathLike[str], field: str) -> Path:
    raw = _absolute_path(os.fspath(value), field)
    path = Path(raw)
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise TrainingEntryError(f"{field} is unavailable") from exc
    if str(resolved) != raw or not resolved.is_dir():
        _fail(f"{field} is not a canonical directory")
    return resolved


def _regular_file(path: Path, field: str, *, expected_mode: int | None = None, expected_nlink: int | None = 1) -> os.stat_result:
    try:
        parent = path.parent.resolve(strict=True)
        if parent != path.parent:
            _fail(f"{field} parent is not canonical")
        observed = path.lstat()
    except OSError as exc:
        raise TrainingEntryError(f"{field} is unavailable") from exc
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISREG(observed.st_mode):
        _fail(f"{field} is not a regular non-symlink file")
    if expected_nlink is not None and observed.st_nlink != expected_nlink:
        _fail(f"{field} nlink drift")
    if expected_mode is not None and stat.S_IMODE(observed.st_mode) != expected_mode:
        _fail(f"{field} mode drift")
    return observed


def _directory(path: Path, field: str) -> os.stat_result:
    try:
        observed = path.lstat()
    except OSError as exc:
        raise TrainingEntryError(f"{field} is unavailable") from exc
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
        _fail(f"{field} is not a directory")
    return observed


def _read_json(path: Path, field: str, *, trailing_lf: bool = False) -> tuple[dict[str, Any], bytes]:
    observed = _regular_file(path, field)
    raw = path.read_bytes()
    if trailing_lf:
        if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
            _fail(f"{field} must end with exactly one LF")
        payload = raw[:-1]
    else:
        if raw.endswith(b"\n"):
            _fail(f"{field} must not end with LF")
        payload = raw
    if payload.startswith(b"\xef\xbb\xbf") or b"\x00" in payload or b"\r" in payload:
        _fail(f"{field} contains forbidden bytes")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                _fail(f"{field} has duplicate key {key}")
            result[key] = value
        return result

    def constant(value: str) -> None:
        _fail(f"{field} contains non-finite constant {value}")

    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    except TrainingEntryError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise TrainingEntryError(f"{field} is not strict UTF-8 JSON") from exc
    if type(value) is not dict:
        _fail(f"{field} root must be a builtin dict")
    _assert_builtin(value, field)
    if not trailing_lf and raw != _canonical(value):
        _fail(f"{field} is not canonical JSON")
    return value, raw


def _file_identity(root: Path, relative: str, field: str) -> dict[str, Any]:
    _relative_path(relative, f"{field}.relative_path")
    path = root / relative
    observed = _regular_file(path, field)
    raw = path.read_bytes()
    return {
        "relative_path": relative,
        "size_bytes": len(raw),
        "sha256": _sha(raw),
        "mode": stat.S_IMODE(observed.st_mode),
    }


def _config_identity(root: Path) -> dict[str, Any]:
    path = root / CONFIG_RELATIVE_PATH
    _, raw = _read_json(path, "T6B config", trailing_lf=True)
    canonical = _canonical(_read_json(path, "T6B config", trailing_lf=True)[0])
    return {
        "relative_path": CONFIG_RELATIVE_PATH,
        "raw_size_bytes": len(raw),
        "raw_sha256": _sha(raw),
        "canonical_size_bytes": len(canonical),
        "canonical_sha256": _sha(canonical),
        "mode": stat.S_IMODE(_regular_file(path, "T6B config").st_mode),
    }


def _parse_config_semantics(value: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "schema_version",
        "contract_id",
        "stage",
        "status",
        "t6a_binding",
        "repository",
        "source_policy",
        "run_identity",
        "targets",
        "environment_identity",
        "data_roles",
        "training_policy",
        "evidence_policy",
        "state_machine",
        "failure_policy",
        "invocation_policy",
        "readiness",
    }
    config = _exact(value, expected, "T6B config")
    if config["schema_version"] != T6B_SCHEMA_VERSION or config["contract_id"] != T6B_CONTRACT_ID or config["stage"] != "T6B":
        _fail("T6B config identity drift")
    if config["status"] != "PRODUCTION_BOUNDARY_IMPLEMENTED_DETACHED_AUTH_REQUIRED":
        _fail("T6B config status drift")
    t6a = _exact(config["t6a_binding"], {"contract_id", "config_relative_path", "authorization_source", "validation_api", "binding_required"}, "T6B t6a_binding")
    _strict_equal(t6a, {"contract_id": _t6a.LAUNCH_CONTRACT_ID, "config_relative_path": _t6a.LAUNCH_CONTRACT_CONFIG_RELATIVE_PATH, "authorization_source": "external_detached_owner_artifact", "validation_api": "owner_authorization_binding", "binding_required": True}, "T6B t6a_binding")
    repository = _exact(config["repository"], {"repo_root_policy", "git_identity", "upstream_required", "launch_time_reobserve"}, "T6B repository")
    _strict_equal(repository, {"repo_root_policy": "absolute_canonical_path_required", "git_identity": ["branch", "head", "tree", "parent", "upstream", "upstream_sha"], "upstream_required": True, "launch_time_reobserve": True}, "T6B repository")
    source = _exact(config["source_policy"], {"t6b_modules", "t6a_binding_required", "launch_time_sha_required"}, "T6B source_policy")
    if type(source["t6b_modules"]) is not list or len(source["t6b_modules"]) != 4 or source["t6b_modules"] != [ENTRY_MODULE_RELATIVE_PATH, PROCESS_MODULE_RELATIVE_PATH, OUTER_MODULE_RELATIVE_PATH, ENGINE_MODULE_RELATIVE_PATH]:
        _fail("T6B source module policy drift")
    if source["t6a_binding_required"] is not True or source["launch_time_sha_required"] is not True:
        _fail("T6B source binding policy drift")
    run_identity = _exact(config["run_identity"], {"run_id_source", "nonce_source", "session_source", "all_unique", "targets_absent_before_launch"}, "T6B run_identity")
    _strict_equal(run_identity, {"run_id_source": "detached_owner_authorization", "nonce_source": "detached_owner_authorization", "session_source": "detached_owner_authorization", "all_unique": True, "targets_absent_before_launch": True}, "T6B run_identity")
    targets = _exact(config["targets"], {"training_evidence_root", "process_evidence_root", "outer_evidence_root", "receipt_policy", "lock_policy"}, "T6B targets")
    if targets["training_evidence_root"] != TARGET_RELATIVE_PATHS["training_evidence_root"] or targets["process_evidence_root"] != TARGET_RELATIVE_PATHS["process_evidence_root"] or targets["outer_evidence_root"] != TARGET_RELATIVE_PATHS["outer_evidence_root"]:
        _fail("T6B target path drift")
    if targets["receipt_policy"] != "adjacent_exclusive_mode_0600" or targets["lock_policy"] != "adjacent_exclusive_mode_0600":
        _fail("T6B target claim policy drift")
    environment = _exact(config["environment_identity"], {"python", "tmux", "gpu", "filesystem"}, "T6B environment_identity")
    for name in ("python", "tmux"):
        executable = _exact(environment[name], {"path_policy", "realpath_policy", "version_policy", "sha256_policy"}, f"T6B environment_identity.{name}")
        if executable["path_policy"] != "absolute_external_path_required" or executable["realpath_policy"] != "regular_non_symlink_binary_required":
            _fail(f"T6B {name} policy drift")
    gpu = _exact(environment["gpu"], {"index", "name", "uuid", "driver_policy", "cuda_version", "power_limit_watts"}, "T6B gpu")
    if gpu != {"index": 0, "name": "NVIDIA GeForce RTX 4090 D", "uuid": "GPU-1faee6f0-1da7-4ede-2475-67a5a00274a8", "driver_policy": "exact_external_identity_required", "cuda_version": "12.4", "power_limit_watts": 425.0}:
        _fail("T6B GPU policy drift")
    filesystem = _exact(environment["filesystem"], {"device_identity_required", "mount_identity_required", "target_parent_must_exist"}, "T6B filesystem")
    if filesystem != {"device_identity_required": True, "mount_identity_required": True, "target_parent_must_exist": True}:
        _fail("T6B filesystem policy drift")
    data = _exact(config["data_roles"], {"train_core", "development", "test", "confirmatory"}, "T6B data_roles")
    if data["train_core"] != {"role": "train_core", "access": "allowed", "identity_source": "detached_owner_authorization", "manifest_required": True} or data["development"] != {"role": "development", "access": "allowed", "identity_source": "detached_owner_authorization", "manifest_required": True}:
        _fail("T6B train/development role drift")
    if data["test"] != {"role": "test", "access": "forbidden", "identity_read": "forbidden"} or data["confirmatory"] != {"role": "confirmatory", "access": "sealed_and_forbidden", "identity_read": "forbidden"}:
        _fail("T6B sealed data role drift")
    _engine.validate_training_policy(config["training_policy"])
    evidence = _exact(config["evidence_policy"], {"raw_byte_streams", "immediate_snapshot", "append_only", "checkpoint_roles", "atomicity", "loadability"}, "T6B evidence_policy")
    if evidence != {"raw_byte_streams": ["outer_stdout", "outer_stderr", "child_stdout", "child_stderr"], "immediate_snapshot": True, "append_only": True, "checkpoint_roles": ["last", "best", "periodic", "final"], "atomicity": True, "loadability": True}:
        _fail("T6B evidence policy drift")
    machine = _exact(config["state_machine"], {"states", "terminal_success", "terminal_failure", "no_resume_after", "no_retry_after", "no_overwrite_after", "audit_required", "certification_separate"}, "T6B state_machine")
    _strict_equal(machine, {"states": list(STATE_SEQUENCE), "terminal_success": "TERMINAL_COMPLETE", "terminal_failure": "PERMANENT_FAIL", "no_resume_after": "LAUNCH_ACCEPTED", "no_retry_after": "LAUNCH_ACCEPTED", "no_overwrite_after": "LAUNCH_ACCEPTED", "audit_required": True, "certification_separate": True}, "T6B state_machine")
    failure = _exact(config["failure_policy"], {"immutable", "resume", "retry", "overwrite", "fallback", "permanent_events"}, "T6B failure_policy")
    if failure["immutable"] is not True or failure["resume"] is not False or failure["retry"] is not False or failure["overwrite"] is not False or failure["fallback"] is not False or type(failure["permanent_events"]) is not list:
        _fail("T6B failure policy drift")
    invocation = _exact(config["invocation_policy"], {"outer_calls", "tmux_new_session_calls", "child_calls", "shell", "argv_sequence", "network", "speed_measurement"}, "T6B invocation_policy")
    if invocation != {"outer_calls": 1, "tmux_new_session_calls": 1, "child_calls": 1, "shell": False, "argv_sequence": True, "network": False, "speed_measurement": False}:
        _fail("T6B invocation policy drift")
    readiness = _exact(config["readiness"], {"static_config_authorizes_production", "owner_authorization_required", "launch_acceptance_separate", "terminal_completion_separate", "independent_audit_required", "training_certification_separate"}, "T6B readiness")
    if readiness != {"static_config_authorizes_production": False, "owner_authorization_required": True, "launch_acceptance_separate": True, "terminal_completion_separate": True, "independent_audit_required": True, "training_certification_separate": True}:
        _fail("T6B readiness policy drift")
    return _copy(config)


def load_t6b_config(repo_root: str | os.PathLike[str]) -> dict[str, Any]:
    """Load the checked-in T6B config and verify raw/canonical identity."""

    root = _canonical_root(repo_root, "repo_root")
    path = root / CONFIG_RELATIVE_PATH
    value, raw = _read_json(path, "T6B config", trailing_lf=True)
    canonical = _canonical(value)
    if CONFIG_RAW_SIZE_BYTES and (len(raw) != CONFIG_RAW_SIZE_BYTES or _sha(raw) != CONFIG_RAW_SHA256):
        _fail("T6B config raw identity drift")
    if CONFIG_CANONICAL_SIZE_BYTES and (len(canonical) != CONFIG_CANONICAL_SIZE_BYTES or _sha(canonical) != CONFIG_CANONICAL_SHA256):
        _fail("T6B config canonical identity drift")
    return _parse_config_semantics(value)


def t6b_config_binding(repo_root: str | os.PathLike[str]) -> dict[str, Any]:
    root = _canonical_root(repo_root, "repo_root")
    config = load_t6b_config(root)
    raw = (root / CONFIG_RELATIVE_PATH).read_bytes()
    canonical = _canonical(config)
    observed = _regular_file(root / CONFIG_RELATIVE_PATH, "T6B config")
    return {
        "relative_path": CONFIG_RELATIVE_PATH,
        "raw_size_bytes": len(raw),
        "raw_sha256": _sha(raw),
        "canonical_size_bytes": len(canonical),
        "canonical_sha256": _sha(canonical),
        "mode": stat.S_IMODE(observed.st_mode),
    }


def current_source_bindings(repo_root: str | os.PathLike[str]) -> dict[str, Any]:
    root = _canonical_root(repo_root, "repo_root")
    rows = []
    for role, relative in (
        ("entry", ENTRY_MODULE_RELATIVE_PATH),
        ("process", PROCESS_MODULE_RELATIVE_PATH),
        ("outer", OUTER_MODULE_RELATIVE_PATH),
        ("engine", ENGINE_MODULE_RELATIVE_PATH),
    ):
        identity = _file_identity(root, relative, f"T6B {role} module")
        rows.append({"role": role, **identity})
    return {"t6b_modules": rows}


def authorization_file_identity(path: str | os.PathLike[str]) -> dict[str, Any]:
    target = Path(_absolute_path(os.fspath(path), "authorization_path"))
    observed = _regular_file(target, "authorization artifact", expected_mode=0o600, expected_nlink=1)
    return {
        "regular": True,
        "symlink": False,
        "mode": stat.S_IMODE(observed.st_mode),
        "uid": observed.st_uid,
        "gid": observed.st_gid,
        "nlink": observed.st_nlink,
    }


def validate_detached_authorization(
    authorization: bytes | Mapping[str, Any],
    *,
    expected_raw_size_bytes: int,
    expected_raw_sha256: str,
    contract_binding: Mapping[str, Any],
    observed_binding: Mapping[str, Any],
    authorization_file_identity_value: Mapping[str, Any],
) -> dict[str, Any]:
    """Use the existing T6A owner validator without duplicating its schema."""

    try:
        return _copy(
            _t6a.owner_authorization_binding(
                authorization,
                expected_raw_size_bytes=expected_raw_size_bytes,
                expected_raw_sha256=expected_raw_sha256,
                contract_binding=dict(contract_binding),
                observed_binding=dict(observed_binding),
                authorization_file_identity=dict(authorization_file_identity_value),
            )
        )
    except _t6a.TrainingLaunchContractError as exc:
        raise TrainingEntryError(str(exc)) from exc


def _validate_authorization_binding(value: Any) -> dict[str, Any]:
    if type(value) is not dict:
        _fail("authorization binding must be a builtin dict")
    expected = {"schema_version", "authorization_id", "authorized", "raw_size_bytes", "raw_sha256", "canonical_size_bytes", "canonical_sha256", "authorization", "authorization_file_identity"}
    binding = _exact(value, expected, "authorization binding")
    _assert_builtin(binding, "authorization binding")
    if binding["schema_version"] != 1 or binding["authorized"] is not True:
        _fail("authorization binding is not authorized")
    _string(binding["authorization_id"], "authorization binding.authorization_id")
    for field in ("raw_size_bytes", "canonical_size_bytes"):
        _integer(binding[field], f"authorization binding.{field}", minimum=1)
    for field in ("raw_sha256", "canonical_sha256"):
        _sha_string(binding[field], f"authorization binding.{field}")
    authorization = binding["authorization"]
    if type(authorization) is not dict or authorization.get("authorized") is not True:
        _fail("authorization binding payload is not authorized")
    try:
        canonical_authorization = _t6a.canonical_owner_authorization_bytes(authorization)
    except _t6a.TrainingLaunchContractError as exc:
        raise TrainingEntryError("authorization binding payload schema drift") from exc
    if binding["canonical_size_bytes"] != len(canonical_authorization) or binding["canonical_sha256"] != _sha(canonical_authorization):
        _fail("authorization binding canonical identity drift")
    raw_authorization = canonical_authorization + b"\n"
    if binding["raw_size_bytes"] != len(raw_authorization) or binding["raw_sha256"] != _sha(raw_authorization):
        _fail("authorization binding raw identity drift")
    if binding["authorization_id"] != authorization.get("authorization_id"):
        _fail("authorization binding authorization ID drift")
    _string(authorization.get("training_run_id"), "authorization.training_run_id")
    _string(authorization.get("tmux_session_name"), "authorization.tmux_session_name")
    _string(authorization.get("nonce"), "authorization.nonce")
    identity = _exact(binding["authorization_file_identity"], {"regular", "symlink", "mode", "uid", "gid", "nlink"}, "authorization file identity")
    if identity != {"regular": True, "symlink": False, "mode": 384, "uid": 1000, "gid": 1000, "nlink": 1}:
        _fail("authorization file identity drift")
    return _copy(binding)


def _authorization_receipt_path(path: Path) -> Path:
    return path.parent / f".{path.name}{AUTHORIZATION_RECEIPT_SUFFIX}"


def _read_exact_fd(fd: int, size: int, field: str) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        try:
            chunk = os.read(fd, remaining)
        except InterruptedError:
            continue
        if not chunk:
            _fail(f"{field} premature EOF")
        chunks.append(chunk)
        remaining -= len(chunk)
    while True:
        try:
            extra = os.read(fd, 1)
            break
        except InterruptedError:
            continue
    if extra:
        _fail(f"{field} has trailing bytes")
    return b"".join(chunks)


def _read_stable_file(path: Path, field: str, *, mode: int | None = None) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd: int | None = None
    try:
        fd = os.open(path, flags)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or (mode is not None and stat.S_IMODE(before.st_mode) != mode):
            _fail(f"{field} metadata drift")
        raw = _read_exact_fd(fd, before.st_size, field)
        after = os.fstat(fd)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_nlink, stat.S_IMODE(before.st_mode)) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_nlink, stat.S_IMODE(after.st_mode)):
            _fail(f"{field} changed during read")
        return raw
    except OSError as exc:
        raise TrainingEntryError(f"{field} could not be read stably") from exc
    finally:
        if fd is not None:
            os.close(fd)


def _write_new(path: Path, payload: bytes, field: str, *, mode: int = 0o600) -> os.stat_result:
    if path.name in {".", ".."} or "/" in path.name:
        _fail(f"{field} name is invalid")
    try:
        parent = path.parent.resolve(strict=True)
    except OSError as exc:
        raise TrainingEntryError(f"{field} parent is unavailable") from exc
    if parent != path.parent:
        _fail(f"{field} parent is not canonical")
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd: int | None = None
    try:
        fd = os.open(path, flags, mode)
        offset = 0
        while offset < len(payload):
            try:
                written = os.write(fd, payload[offset:])
            except InterruptedError:
                continue
            if written <= 0:
                _fail(f"{field} short write")
            offset += written
        while True:
            try:
                os.fsync(fd)
                break
            except InterruptedError:
                continue
        observed = os.fstat(fd)
        if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1 or stat.S_IMODE(observed.st_mode) != mode or observed.st_size != len(payload):
            _fail(f"{field} metadata drift")
        os.lseek(fd, 0, os.SEEK_SET)
        actual = _read_exact_fd(fd, len(payload), field)
        if actual != payload:
            _fail(f"{field} readback bytes drift")
        return observed
    except OSError as exc:
        raise TrainingEntryError(f"{field} could not be created durably") from exc
    finally:
        if fd is not None:
            os.close(fd)


def _fsync_directory(path: Path, field: str) -> None:
    fd: int | None = None
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0))
        while True:
            try:
                os.fsync(fd)
                return
            except InterruptedError:
                continue
    except OSError as exc:
        raise TrainingEntryError(f"{field} directory fsync failed") from exc
    finally:
        if fd is not None:
            os.close(fd)


def consume_owner_authorization(
    authorization_path: str | os.PathLike[str],
    *,
    expected_raw_size_bytes: int,
    expected_raw_sha256: str,
    contract_binding: Mapping[str, Any],
    observed_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and consume one detached authorization with one receipt."""

    path = Path(_absolute_path(os.fspath(authorization_path), "authorization_path"))
    file_identity = authorization_file_identity(path)
    receipt_path = _authorization_receipt_path(path)
    if receipt_path.exists() or receipt_path.is_symlink():
        _fail("detached authorization was already consumed")
    raw = _read_stable_file(path, "authorization artifact", mode=0o600)
    if len(raw) != expected_raw_size_bytes or _sha(raw) != expected_raw_sha256:
        _fail("authorization external raw identity mismatch")
    binding = validate_detached_authorization(
        raw,
        expected_raw_size_bytes=expected_raw_size_bytes,
        expected_raw_sha256=expected_raw_sha256,
        contract_binding=contract_binding,
        observed_binding=observed_binding,
        authorization_file_identity_value=file_identity,
    )
    receipt = {
        "schema_version": T6B_SCHEMA_VERSION,
        "kind": "T6B_DETACHED_AUTHORIZATION_CONSUMED",
        "authorization_id": binding["authorization_id"],
        "training_run_id": binding["authorization"]["training_run_id"],
        "tmux_session_name": binding["authorization"]["tmux_session_name"],
        "nonce": binding["authorization"]["nonce"],
        "authorization_path": str(path),
        "authorization_binding_sha256": _digest(binding),
        "raw_size_bytes": binding["raw_size_bytes"],
        "raw_sha256": binding["raw_sha256"],
        "canonical_size_bytes": binding["canonical_size_bytes"],
        "canonical_sha256": binding["canonical_sha256"],
        "consumed": True,
    }
    _write_new(receipt_path, _canonical(receipt), "authorization consumption receipt")
    _fsync_directory(receipt_path.parent, "authorization receipt parent")
    return {"binding": binding, "receipt_path": str(receipt_path), "receipt_sha256": _sha(_canonical(receipt))}


def validate_consumed_authorization_receipt(
    receipt_path: str | os.PathLike[str],
    *,
    authorization_binding: Mapping[str, Any],
    authorization_path: str | os.PathLike[str],
) -> dict[str, Any]:
    path = Path(_absolute_path(os.fspath(receipt_path), "authorization_receipt_path"))
    value, raw = _read_json(path, "authorization consumption receipt")
    _regular_file(path, "authorization consumption receipt", expected_mode=0o600, expected_nlink=1)
    expected = {
        "schema_version",
        "kind",
        "authorization_id",
        "training_run_id",
        "tmux_session_name",
        "nonce",
        "authorization_path",
        "authorization_binding_sha256",
        "raw_size_bytes",
        "raw_sha256",
        "canonical_size_bytes",
        "canonical_sha256",
        "consumed",
    }
    receipt = _exact(value, expected, "authorization consumption receipt")
    binding = _validate_authorization_binding(authorization_binding)
    if raw != _canonical(receipt) or receipt["schema_version"] != T6B_SCHEMA_VERSION or receipt["kind"] != "T6B_DETACHED_AUTHORIZATION_CONSUMED" or receipt["consumed"] is not True:
        _fail("authorization consumption receipt bytes drift")
    authorization = binding["authorization"]
    if receipt["authorization_path"] != _absolute_path(os.fspath(authorization_path), "authorization_path") or receipt["authorization_binding_sha256"] != _digest(binding) or receipt["authorization_id"] != binding["authorization_id"] or receipt["training_run_id"] != authorization["training_run_id"] or receipt["tmux_session_name"] != authorization["tmux_session_name"] or receipt["nonce"] != authorization["nonce"]:
        _fail("authorization consumption receipt binding drift")
    for field in ("raw_size_bytes", "canonical_size_bytes"):
        if receipt[field] != binding[field]:
            _fail(f"authorization receipt {field} drift")
    for field in ("raw_sha256", "canonical_sha256"):
        if receipt[field] != binding[field]:
            _fail(f"authorization receipt {field} drift")
    return {"receipt": _copy(receipt), "receipt_sha256": _sha(raw)}


def _validate_repository(value: Any) -> dict[str, Any]:
    repository = _exact(value, {"repo_root", "branch", "head", "tree", "parent", "upstream", "upstream_sha"}, "repository")
    _absolute_path(repository["repo_root"], "repository.repo_root")
    _string(repository["branch"], "repository.branch")
    for field in ("head", "tree", "parent", "upstream_sha"):
        value = repository[field]
        if type(value) is not str or len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
            _fail(f"repository.{field} is not a Git object ID")
    _string(repository["upstream"], "repository.upstream")
    return _copy(repository)


def _git_scalar(root: Path, arguments: list[str], field: str) -> str:
    """Run one fixed Git query and accept exactly one ASCII output line."""

    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=str(root),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            shell=False,
        )
    except OSError as exc:
        raise TrainingEntryError(f"{field} Git query could not run") from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise TrainingEntryError(f"{field} Git query failed: returncode={completed.returncode} stderr={detail}")
    raw = completed.stdout
    if type(raw) is not bytes:
        _fail(f"{field} Git stdout is not bytes")
    lines = raw.splitlines()
    if len(lines) != 1 or not raw.endswith(b"\n"):
        _fail(f"{field} Git stdout is not one line")
    try:
        value = lines[0].decode("ascii")
    except UnicodeDecodeError as exc:
        raise TrainingEntryError(f"{field} Git stdout is not ASCII") from exc
    if not value or value != value.strip() or any(ord(char) < 32 or ord(char) == 127 for char in value):
        _fail(f"{field} Git stdout is not a normalized scalar")
    return value


def _observe_git_identity(root: Path) -> dict[str, Any]:
    branch = _git_scalar(root, ["rev-parse", "--abbrev-ref", "HEAD"], "branch")
    head = _git_scalar(root, ["rev-parse", "HEAD"], "HEAD")
    tree = _git_scalar(root, ["rev-parse", "HEAD^{tree}"], "tree")
    parents = _git_scalar(root, ["rev-list", "--parents", "-n", "1", "HEAD"], "parent")
    parent_parts = parents.split(" ")
    if len(parent_parts) != 2 or parent_parts[0] != head:
        _fail("HEAD must have exactly one parent")
    upstream = _git_scalar(root, ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"], "upstream")
    upstream_sha = _git_scalar(root, ["rev-parse", "--verify", "@{upstream}"], "upstream SHA")
    return _validate_repository(
        {
            "repo_root": str(root),
            "branch": branch,
            "head": head,
            "tree": tree,
            "parent": parent_parts[1],
            "upstream": upstream,
            "upstream_sha": upstream_sha,
        }
    )


def _validate_launch_git_identity(descriptor: dict[str, Any]) -> None:
    root = _canonical_root(descriptor["repository"]["repo_root"], "descriptor.repository.repo_root")
    observed = _observe_git_identity(root)
    if observed != descriptor["repository"]:
        _fail("launch-time Git identity drift")


def _validate_module_identity(value: Any, field: str, expected_path: str | None = None) -> dict[str, Any]:
    row = _exact(value, {"role", "relative_path", "size_bytes", "sha256", "mode"}, field)
    _string(row["role"], f"{field}.role")
    _relative_path(row["relative_path"], f"{field}.relative_path")
    if expected_path is not None and row["relative_path"] != expected_path:
        _fail(f"{field} path drift")
    _integer(row["size_bytes"], f"{field}.size_bytes", minimum=1)
    _sha_string(row["sha256"], f"{field}.sha256")
    _integer(row["mode"], f"{field}.mode", minimum=1)
    return _copy(row)


def _validate_data_roles(value: Any) -> dict[str, Any]:
    data = _exact(value, {"train_core", "development", "test", "confirmatory"}, "data_roles")
    for role in ("train_core", "development"):
        item = _exact(data[role], {"role", "root", "manifest_sha256"}, f"data_roles.{role}")
        if item["role"] != role:
            _fail(f"data_roles.{role}.role drift")
        _absolute_path(item["root"], f"data_roles.{role}.root")
        _sha_string(item["manifest_sha256"], f"data_roles.{role}.manifest_sha256")
        folded = {part.casefold() for part in item["root"].split("/") if part}
        if folded & {"test", "confirmatory", "raw", "annotations", "raw_annotation"} or any("annotation" in part or part.startswith(("test_", "test-", "confirmatory_", "confirmatory-")) for part in folded):
            _fail(f"data_roles.{role} points into sealed data")
    if data["train_core"]["root"] == data["development"]["root"]:
        _fail("train_core and development roots must be distinct")
    if data["test"] != {"role": "test", "access": "forbidden", "identity_read": "forbidden"} or data["confirmatory"] != {"role": "confirmatory", "access": "sealed_and_forbidden", "identity_read": "forbidden"}:
        _fail("sealed data role drift")
    return _copy(data)


def _validate_environment(value: Any) -> dict[str, Any]:
    environment = _exact(value, {"python", "tmux", "gpu", "filesystem", "variables"}, "environment")
    for name in ("python", "tmux"):
        item = _exact(environment[name], {"path", "realpath", "version", "sha256"}, f"environment.{name}")
        _absolute_path(item["path"], f"environment.{name}.path")
        _absolute_path(item["realpath"], f"environment.{name}.realpath")
        _string(item["version"], f"environment.{name}.version")
        _sha_string(item["sha256"], f"environment.{name}.sha256")
    gpu = _exact(environment["gpu"], {"index", "name", "uuid", "driver_version", "cuda_version", "power_limit_watts"}, "environment.gpu")
    if gpu["index"] != 0 or gpu["name"] != "NVIDIA GeForce RTX 4090 D" or gpu["uuid"] != "GPU-1faee6f0-1da7-4ede-2475-67a5a00274a8" or gpu["cuda_version"] != "12.4" or type(gpu["power_limit_watts"]) is not float or gpu["power_limit_watts"] != 425.0:
        _fail("environment GPU identity drift")
    _string(gpu["driver_version"], "environment.gpu.driver_version")
    filesystem = _exact(environment["filesystem"], {"device", "mount"}, "environment.filesystem")
    _integer(filesystem["device"], "environment.filesystem.device", minimum=0)
    _absolute_path(filesystem["mount"], "environment.filesystem.mount")
    variables = _exact(environment["variables"], set(ENVIRONMENT_KEYS), "environment.variables")
    for key in ENVIRONMENT_KEYS:
        _string(variables[key], f"environment.variables.{key}", nonempty=False)
    return _copy(environment)


def _validate_targets(value: Any) -> dict[str, str]:
    targets = _exact(value, set(TARGET_RELATIVE_PATHS), "targets")
    result: dict[str, str] = {}
    for key in TARGET_RELATIVE_PATHS:
        result[key] = _absolute_path(targets[key], f"targets.{key}")
    if len(set(result.values())) != 3:
        _fail("T6B targets are not unique")
    return result


def _state_trace(states: list[str]) -> list[dict[str, Any]]:
    if not states or states[0] != "DESIGN_ONLY":
        _fail("state trace must start at DESIGN_ONLY")
    trace: list[dict[str, Any]] = [{"sequence": 0, "state": states[0], "predecessor_sha256": None}]
    for index, state in enumerate(states[1:], 1):
        if state not in STATE_SEQUENCE:
            _fail("state trace contains unknown state")
        previous = trace[-1]
        trace.append({"sequence": index, "state": state, "predecessor_sha256": _digest(previous)})
    return trace


def _validate_state_machine(value: Any) -> dict[str, Any]:
    machine = _exact(value, {"states", "trace", "exactly_once", "durable_evidence_required"}, "state_machine")
    if machine["states"] != list(STATE_SEQUENCE) or machine["exactly_once"] is not True or machine["durable_evidence_required"] is not True:
        _fail("entry state machine policy drift")
    trace = machine["trace"]
    if type(trace) is not list or len(trace) != 4 or [row.get("state") for row in trace] != ["DESIGN_ONLY", "OWNER_AUTHORIZED", "PREFLIGHT_PASS", "LAUNCH_ACCEPTED"]:
        _fail("entry prepared state trace drift")
    for index, row in enumerate(trace):
        row = _exact(row, {"sequence", "state", "predecessor_sha256"}, f"state_machine.trace[{index}]")
        if row["sequence"] != index or (index == 0 and row["predecessor_sha256"] is not None) or (index > 0 and row["predecessor_sha256"] != _digest(trace[index - 1])):
            _fail("entry state predecessor drift")
    return _copy(machine)


def _validate_policy(value: Any) -> dict[str, Any]:
    try:
        return _engine.validate_training_policy(value)
    except _engine.TrainingEngineError as exc:
        raise TrainingEntryError(str(exc)) from exc


def _validate_descriptor_shape(value: Any) -> dict[str, Any]:
    _assert_builtin(value, "entry descriptor")
    descriptor = _exact(value, DESCRIPTOR_KEYS, "entry descriptor")
    if descriptor["schema_version"] != T6B_SCHEMA_VERSION or descriptor["contract_id"] != T6B_CONTRACT_ID or descriptor["entry_id"] != ENTRY_ID or descriptor["mode"] != ENTRY_MODE:
        _fail("entry descriptor identity drift")
    _string(descriptor["training_run_id"], "descriptor.training_run_id")
    _string(descriptor["nonce"], "descriptor.nonce")
    _string(descriptor["tmux_session_name"], "descriptor.tmux_session_name")
    repository = _validate_repository(descriptor["repository"])
    _absolute_path(repository["repo_root"], "repository.repo_root")
    authorization_binding = _validate_authorization_binding(descriptor["authorization_binding"])
    if descriptor["authorization_binding_sha256"] != _digest(descriptor["authorization_binding"]):
        _fail("descriptor authorization binding digest drift")
    authorization = authorization_binding["authorization"]
    if repository != authorization["repository"]:
        _fail("descriptor repository is not bound to detached authorization")
    if descriptor["training_run_id"] != authorization["training_run_id"] or descriptor["nonce"] != authorization["nonce"] or descriptor["tmux_session_name"] != authorization["tmux_session_name"]:
        _fail("descriptor authorization run identity drift")
    _absolute_path(descriptor["authorization_path"], "descriptor.authorization_path")
    authorization_path = Path(descriptor["authorization_path"])
    authorization_receipt_path = Path(_absolute_path(descriptor["authorization_receipt_path"], "descriptor.authorization_receipt_path"))
    if authorization_receipt_path != _authorization_receipt_path(authorization_path):
        _fail("descriptor authorization receipt path drift")
    evidence_root = Path(_absolute_path(descriptor["evidence_root"], "descriptor.evidence_root"))
    evidence_receipt_path = Path(_absolute_path(descriptor["evidence_receipt_path"], "descriptor.evidence_receipt_path"))
    if evidence_receipt_path != _evidence_receipt_path(evidence_root):
        _fail("descriptor evidence receipt path drift")
    source = _exact(descriptor["source_bindings"], {"t6b_modules", "t6a_source_binding"}, "descriptor.source_bindings")
    modules = source["t6b_modules"]
    if type(modules) is not list or len(modules) != 4:
        _fail("descriptor T6B source module count drift")
    expected_paths = [ENTRY_MODULE_RELATIVE_PATH, PROCESS_MODULE_RELATIVE_PATH, OUTER_MODULE_RELATIVE_PATH, ENGINE_MODULE_RELATIVE_PATH]
    for index, expected_path in enumerate(expected_paths):
        _validate_module_identity(modules[index], f"descriptor.source_bindings.t6b_modules[{index}]", expected_path)
    if type(source["t6a_source_binding"]) is not dict:
        _fail("descriptor T6A source binding is not a dict")
    config = _exact(descriptor["config_identity"], {"relative_path", "raw_size_bytes", "raw_sha256", "canonical_size_bytes", "canonical_sha256", "mode"}, "descriptor.config_identity")
    _string(config["relative_path"], "descriptor.config_identity.relative_path")
    if config["relative_path"] != CONFIG_RELATIVE_PATH or config["raw_size_bytes"] <= 0 or config["canonical_size_bytes"] <= 0 or config["mode"] not in {0o644, 0o664}:
        _fail("descriptor config identity drift")
    _sha_string(config["raw_sha256"], "descriptor.config_identity.raw_sha256")
    _sha_string(config["canonical_sha256"], "descriptor.config_identity.canonical_sha256")
    _validate_environment(descriptor["environment"])
    if descriptor["child_environment"] != descriptor["environment"]["variables"]:
        _fail("descriptor child environment drift")
    data = _validate_data_roles(descriptor["data_roles"])
    policy = _validate_policy(descriptor["training_policy"])
    targets = _validate_targets(descriptor["targets"])
    for key, target in targets.items():
        if target == repository["repo_root"] or target.startswith(repository["repo_root"] + "/configs") or target.startswith(repository["repo_root"] + "/src"):
            _fail(f"descriptor target {key} is unsafe")
    _absolute_path(descriptor["cwd"], "descriptor.cwd")
    argv = descriptor["argv"]
    if type(argv) is not list or not argv or any(type(item) is not str or not item for item in argv):
        _fail("descriptor argv is not a string sequence")
    if not argv[0].startswith("/"):
        _fail("descriptor python argv is not absolute")
    _validate_state_machine(descriptor["state_machine"])
    invocation = _exact(descriptor["invocation_policy"], {"outer_calls", "tmux_new_session_calls", "child_calls", "shell", "argv_sequence", "network", "retry", "resume", "overwrite", "fallback"}, "invocation_policy")
    if invocation != {"outer_calls": 1, "tmux_new_session_calls": 1, "child_calls": 1, "shell": False, "argv_sequence": True, "network": False, "retry": False, "resume": False, "overwrite": False, "fallback": False}:
        _fail("descriptor invocation policy drift")
    for field in ("evidence_root", "process_evidence_root", "outer_evidence_root"):
        _absolute_path(descriptor[field], f"descriptor.{field}")
    if len({descriptor["evidence_root"], descriptor["process_evidence_root"], descriptor["outer_evidence_root"]}) != 3:
        _fail("T6B evidence roots are not distinct")
    roots = [Path(descriptor[field]) for field in ("evidence_root", "process_evidence_root", "outer_evidence_root")]
    if any(left in right.parents or right in left.parents for index, left in enumerate(roots) for right in roots[index + 1 :]):
        _fail("T6B evidence roots must not overlap")
    aggregate = _sha_string(descriptor["aggregate_sha256"], "descriptor.aggregate_sha256")
    body = _copy(descriptor)
    del body["aggregate_sha256"]
    if _digest(body) != aggregate:
        _fail("entry descriptor aggregate identity drift")
    return _copy(descriptor)


def _live_source_check(descriptor: dict[str, Any]) -> None:
    root = _canonical_root(descriptor["repository"]["repo_root"], "descriptor.repository.repo_root")
    current = current_source_bindings(root)
    if current != {"t6b_modules": descriptor["source_bindings"]["t6b_modules"]}:
        _fail("T6B source identity drift")
    observed_config = t6b_config_binding(root)
    if observed_config != descriptor["config_identity"]:
        _fail("T6B config identity drift")
    try:
        t6a_binding = _t6a.training_launch_contract_binding(root)
    except _t6a.TrainingLaunchContractError as exc:
        raise TrainingEntryError(str(exc)) from exc
    if _digest(t6a_binding) != _digest(descriptor["t6a_contract_binding"]):
        _fail("T6A contract binding drift")


def build_production_entry_descriptor(
    repo_root: str | os.PathLike[str],
    *,
    authorization_binding: Mapping[str, Any],
    authorization_path: str | os.PathLike[str],
    authorization_receipt_path: str | os.PathLike[str],
    t6a_contract_binding: Mapping[str, Any],
    repository: Mapping[str, Any],
    environment: Mapping[str, Any],
    data_roles: Mapping[str, Any],
    cwd: str | os.PathLike[str],
    argv: list[str],
    targets: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    root = _canonical_root(repo_root, "repo_root")
    config = t6b_config_binding(root)
    source = current_source_bindings(root)
    auth = _validate_authorization_binding(dict(authorization_binding))
    authorization = auth["authorization"]
    checked_repository = _validate_repository(dict(repository))
    if checked_repository != authorization["repository"] or checked_repository["repo_root"] != str(root):
        _fail("descriptor repository is not bound to repo root and authorization")
    actual_targets = dict(targets) if targets is not None else {key: str(root / relative) for key, relative in TARGET_RELATIVE_PATHS.items()}
    body = {
        "schema_version": T6B_SCHEMA_VERSION,
        "contract_id": T6B_CONTRACT_ID,
        "entry_id": ENTRY_ID,
        "mode": ENTRY_MODE,
        "training_run_id": authorization["training_run_id"],
        "nonce": authorization["nonce"],
        "tmux_session_name": authorization["tmux_session_name"],
        "repository": _copy(dict(repository)),
        "t6a_contract_binding": _copy(dict(t6a_contract_binding)),
        "authorization_path": _absolute_path(os.fspath(authorization_path), "authorization_path"),
        "authorization_binding": auth,
        "authorization_binding_sha256": _digest(auth),
        "authorization_receipt_path": _absolute_path(os.fspath(authorization_receipt_path), "authorization_receipt_path"),
        "evidence_receipt_path": str(Path(actual_targets["training_evidence_root"]).parent / f".{Path(actual_targets['training_evidence_root']).name}{EVIDENCE_RECEIPT_SUFFIX}"),
        "config_identity": config,
        "source_bindings": {"t6b_modules": source["t6b_modules"], "t6a_source_binding": _copy(dict(t6a_contract_binding["source_bindings"]))},
        "environment": _copy(dict(environment)),
        "data_roles": _copy(dict(data_roles)),
        "training_policy": _validate_policy(load_t6b_config(root)["training_policy"]),
        "targets": actual_targets,
        "cwd": _absolute_path(os.fspath(cwd), "cwd"),
        "argv": list(argv),
        "child_environment": _copy(dict(environment)["variables"]),
        "state_machine": {"states": list(STATE_SEQUENCE), "trace": _state_trace(["DESIGN_ONLY", "OWNER_AUTHORIZED", "PREFLIGHT_PASS", "LAUNCH_ACCEPTED"]), "exactly_once": True, "durable_evidence_required": True},
        "invocation_policy": {"outer_calls": 1, "tmux_new_session_calls": 1, "child_calls": 1, "shell": False, "argv_sequence": True, "network": False, "retry": False, "resume": False, "overwrite": False, "fallback": False},
        "evidence_root": actual_targets["training_evidence_root"],
        "process_evidence_root": actual_targets["process_evidence_root"],
        "outer_evidence_root": actual_targets["outer_evidence_root"],
    }
    return _validate_descriptor_shape({**body, "aggregate_sha256": _digest(body)})


def validate_entry_descriptor(value: Any) -> dict[str, Any]:
    descriptor = _validate_descriptor_shape(value)
    _live_source_check(descriptor)
    return _copy(descriptor)


def _entry_receipt_payload(descriptor: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": T6B_SCHEMA_VERSION,
        "kind": "T6B_ENTRY_INVOCATION",
        "descriptor_sha256": descriptor["aggregate_sha256"],
        "authorization_binding_sha256": descriptor["authorization_binding_sha256"],
        "training_run_id": descriptor["training_run_id"],
        "nonce": descriptor["nonce"],
        "status": "CLAIMED",
    }


def _build_engine_execution_context(
    descriptor: dict[str, Any],
    consumed: Mapping[str, Any],
    evidence_claim: Mapping[str, Any],
    root: Path,
) -> dict[str, Any]:
    """Bind all pre-engine entry bytes into one immutable in-memory context."""

    authorization_path = Path(descriptor["authorization_receipt_path"])
    evidence_path = Path(descriptor["evidence_receipt_path"])
    consumption_path = root / "entry_consumption.json"
    invocation_path = root / "entry_invocation.json"
    authorization_raw = _read_stable_file(authorization_path, "authorization consumption receipt", mode=0o600)
    evidence_raw = _read_stable_file(evidence_path, "training evidence claim receipt", mode=0o600)
    consumption_raw = _read_stable_file(consumption_path, "entry consumption", mode=0o600)
    invocation_raw = _read_stable_file(invocation_path, "entry invocation", mode=0o600)
    if consumed["receipt_sha256"] != _sha(authorization_raw) or evidence_claim["receipt_sha256"] != _sha(evidence_raw):
        _fail("engine execution context source receipt drift")
    return {
        "schema_version": _engine.ENGINE_CONTEXT_SCHEMA_VERSION,
        "kind": _engine.ENGINE_CONTEXT_KIND,
        "descriptor": _copy(descriptor),
        "descriptor_sha256": descriptor["aggregate_sha256"],
        "training_run_id": descriptor["training_run_id"],
        "nonce": descriptor["nonce"],
        "authorization": {
            "path": descriptor["authorization_path"],
            "binding": _copy(descriptor["authorization_binding"]),
            "binding_sha256": descriptor["authorization_binding_sha256"],
            "receipt_path": descriptor["authorization_receipt_path"],
            "receipt_size_bytes": len(authorization_raw),
            "receipt_sha256": _sha(authorization_raw),
            "receipt_bytes_hex": authorization_raw.hex(),
        },
        "evidence": {
            "root": descriptor["evidence_root"],
            "receipt_path": descriptor["evidence_receipt_path"],
            "receipt_size_bytes": len(evidence_raw),
            "receipt_sha256": _sha(evidence_raw),
            "receipt_bytes_hex": evidence_raw.hex(),
        },
        "entry": {
            "consumption_path": str(consumption_path),
            "consumption_size_bytes": len(consumption_raw),
            "consumption_sha256": _sha(consumption_raw),
            "consumption_bytes_hex": consumption_raw.hex(),
            "invocation_path": str(invocation_path),
            "invocation_size_bytes": len(invocation_raw),
            "invocation_sha256": _sha(invocation_raw),
            "invocation_bytes_hex": invocation_raw.hex(),
        },
        "repository": _copy(descriptor["repository"]),
        "t6a_contract_binding": _copy(descriptor["t6a_contract_binding"]),
        "source_bindings": _copy(descriptor["source_bindings"]),
        "config_identity": _copy(descriptor["config_identity"]),
        "environment": _copy(descriptor["environment"]),
        "data_roles": _copy(descriptor["data_roles"]),
        "training_policy": _copy(descriptor["training_policy"]),
        "targets": _copy(descriptor["targets"]),
        "cwd": descriptor["cwd"],
        "argv": _copy(descriptor["argv"]),
        "child_environment": _copy(descriptor["child_environment"]),
        "evidence_root": descriptor["evidence_root"],
        "process_evidence_root": descriptor["process_evidence_root"],
        "outer_evidence_root": descriptor["outer_evidence_root"],
        "engine_claim_path": str(root / _engine.ENGINE_CLAIM_FILE_NAME),
    }


def _ensure_process_root(root: Path) -> None:
    _directory(root, "process evidence root")
    if stat.S_IMODE(root.lstat().st_mode) not in {0o700, 0o755}:
        _fail("process evidence root mode drift")


def _evidence_receipt_path(root: Path) -> Path:
    return root.parent / f".{root.name}{EVIDENCE_RECEIPT_SUFFIX}"


def _persistence_failure_marker_path(root: Path) -> Path:
    return root.parent / f".{root.name}{PERSISTENCE_FAILURE_MARKER_SUFFIX}"


def _record_persistence_failure(
    root: Path,
    descriptor: dict[str, Any],
    phase: str,
    error: BaseException,
) -> None:
    """Leave an external terminal marker when evidence publication breaks."""

    marker = _persistence_failure_marker_path(root)
    payload = {
        "schema_version": T6B_SCHEMA_VERSION,
        "kind": "T6B_ENTRY_PERSISTENCE_FAILURE",
        "descriptor_sha256": descriptor["aggregate_sha256"],
        "training_run_id": descriptor["training_run_id"],
        "nonce": descriptor["nonce"],
        "evidence_root": str(root),
        "phase": phase,
        "exception_type": type(error).__name__,
        "exception_message": str(error),
        "failure_class": "EVIDENCE_FAILURE",
        "status": "PERMANENT_FAIL",
    }
    try:
        if not marker.exists() and not marker.is_symlink():
            _write_new(marker, _canonical(payload), "entry persistence failure marker")
            _fsync_directory(marker.parent, "entry persistence failure marker parent")
    except Exception:
        # The original publication exception remains the authoritative error.
        return


def _has_valid_persistence_failure_marker(root: Path) -> bool:
    marker = _persistence_failure_marker_path(root)
    if not marker.exists() and not marker.is_symlink():
        return False
    try:
        observed = _regular_file(marker, "entry persistence failure marker", expected_mode=0o600, expected_nlink=1)
        value, raw = _read_json(marker, "entry persistence failure marker")
        expected = {
            "schema_version": T6B_SCHEMA_VERSION,
            "kind": "T6B_ENTRY_PERSISTENCE_FAILURE",
            "descriptor_sha256": value.get("descriptor_sha256"),
            "training_run_id": value.get("training_run_id"),
            "nonce": value.get("nonce"),
            "evidence_root": str(root),
            "phase": value.get("phase"),
            "exception_type": value.get("exception_type"),
            "exception_message": value.get("exception_message"),
            "failure_class": "EVIDENCE_FAILURE",
            "status": "PERMANENT_FAIL",
        }
        if observed is None or raw != _canonical(value) or set(value) != set(expected) or value != expected:
            return False
        return _sha_string(value["descriptor_sha256"], "entry marker descriptor") is not None and all(
            type(value[field]) is str and bool(value[field]) for field in ("training_run_id", "nonce", "phase", "exception_type")
        )
    except Exception:
        return False


def _claim_evidence_root(descriptor: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    root = Path(descriptor["evidence_root"])
    receipt = Path(descriptor["evidence_receipt_path"])
    _canonical_root(str(root.parent), "training evidence parent")
    if root.exists() or root.is_symlink() or receipt.exists() or receipt.is_symlink():
        _fail("training evidence target or receipt already exists")
    payload = {
        "schema_version": T6B_SCHEMA_VERSION,
        "kind": "T6B_ENTRY_EVIDENCE_CLAIM",
        "descriptor_sha256": descriptor["aggregate_sha256"],
        "descriptor": _copy(descriptor),
        "training_run_id": descriptor["training_run_id"],
        "nonce": descriptor["nonce"],
        "status": "CLAIMED",
    }
    raw = _canonical(payload)
    _write_new(receipt, raw, "training evidence receipt")
    _fsync_directory(receipt.parent, "training evidence receipt parent")
    try:
        os.mkdir(root, 0o700)
        observed = root.lstat()
        if not stat.S_ISDIR(observed.st_mode) or stat.S_IMODE(observed.st_mode) != 0o700 or observed.st_nlink < 2 or observed.st_uid != os.getuid() or observed.st_gid != os.getgid():
            _fail("training evidence root metadata drift")
        _fsync_directory(root.parent, "training evidence parent")
    except OSError as exc:
        raise TrainingEntryError("training evidence root could not be created") from exc
    return root, {"receipt": payload, "receipt_sha256": _sha(raw)}


def _publish_json(root: Path, name: str, value: Mapping[str, Any]) -> os.stat_result:
    raw = _canonical(dict(value))
    return _write_new(root / name, raw, name)


def _entry_failure(
    descriptor: dict[str, Any],
    *,
    return_code: int,
    failure_class: str,
    engine_result: Any = None,
    entry_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    if failure_class not in PERMANENT_FAILURE_CLASSES:
        failure_class = "ENGINE_FAILURE"
    body = {
        "schema_version": T6B_SCHEMA_VERSION,
        "contract_id": T6B_CONTRACT_ID,
        "entry_id": ENTRY_ID,
        "mode": ENTRY_MODE,
        "training_run_id": descriptor["training_run_id"],
        "nonce": descriptor["nonce"],
        "descriptor_sha256": descriptor["aggregate_sha256"],
        "authorization_binding_sha256": descriptor["authorization_binding_sha256"],
        "authorization_receipt_path": descriptor["authorization_receipt_path"],
        "evidence_receipt_path": descriptor["evidence_receipt_path"],
        "repository": _copy(descriptor["repository"]),
        "source_bindings": _copy(descriptor["source_bindings"]),
        "environment": _copy(descriptor["environment"]),
        "data_roles": _copy(descriptor["data_roles"]),
        "training_policy": _copy(descriptor["training_policy"]),
        "training_evidence_root": descriptor["evidence_root"],
        "process_evidence_root": descriptor["process_evidence_root"],
        "outer_evidence_root": descriptor["outer_evidence_root"],
        "return_code": return_code,
        "engine_result": _copy(engine_result),
        "status": "PERMANENT_FAIL",
        "classification": "PERMANENT_FAIL",
        "failure_class": failure_class,
        "production_training_authorized": False,
        "real_process_launch_authorized": False,
        "entry_receipt_sha256": entry_receipt_sha256,
    }
    return {**body, "aggregate_result_sha256": _digest(body)}


def validate_entry_result(value: Any, expected_descriptor: Any | None = None) -> dict[str, Any]:
    result = _exact(value, RESULT_KEYS, "entry result")
    _assert_builtin(result, "entry result")
    if result["schema_version"] != T6B_SCHEMA_VERSION or result["contract_id"] != T6B_CONTRACT_ID or result["entry_id"] != ENTRY_ID or result["mode"] != ENTRY_MODE:
        _fail("entry result identity drift")
    _string(result["training_run_id"], "result.training_run_id")
    _string(result["nonce"], "result.nonce")
    _sha_string(result["descriptor_sha256"], "result.descriptor_sha256")
    _sha_string(result["authorization_binding_sha256"], "result.authorization_binding_sha256")
    _absolute_path(result["authorization_receipt_path"], "result.authorization_receipt_path")
    _absolute_path(result["evidence_receipt_path"], "result.evidence_receipt_path")
    _validate_repository(result["repository"])
    _validate_environment(result["environment"])
    _validate_data_roles(result["data_roles"])
    _validate_policy(result["training_policy"])
    if result["status"] == "TERMINAL_COMPLETE":
        try:
            _engine._validate_engine_result(result["engine_result"])
        except _engine.TrainingEngineError as exc:
            raise TrainingEntryError(str(exc)) from exc
    elif result["engine_result"] is not None:
        _assert_builtin(result["engine_result"], "result.engine_result")
    if result["entry_receipt_sha256"] is not None:
        _sha_string(result["entry_receipt_sha256"], "result.entry_receipt_sha256")
    for field in ("training_evidence_root", "process_evidence_root", "outer_evidence_root"):
        _absolute_path(result[field], f"result.{field}")
    if result["evidence_receipt_path"] != str(_evidence_receipt_path(Path(result["training_evidence_root"]))):
        _fail("result evidence receipt path drift")
    _integer(result["return_code"], "result.return_code", minimum=0, maximum=255)
    if result["status"] not in {"TERMINAL_COMPLETE", "PERMANENT_FAIL"} or result["classification"] != result["status"]:
        _fail("entry result terminal state drift")
    if result["status"] == "TERMINAL_COMPLETE":
        if result["failure_class"] != "NONE" or result["return_code"] != 0 or result["production_training_authorized"] is not True or result["real_process_launch_authorized"] is not True or result["entry_receipt_sha256"] is None:
            _fail("entry successful result semantics drift")
    else:
        if result["failure_class"] not in PERMANENT_FAILURE_CLASSES or result["production_training_authorized"] is not False or result["real_process_launch_authorized"] is not False:
            _fail("entry permanent failure semantics drift")
    _sha_string(result["aggregate_result_sha256"], "result.aggregate_result_sha256")
    body = _copy(result)
    del body["aggregate_result_sha256"]
    if _digest(body) != result["aggregate_result_sha256"]:
        _fail("entry result aggregate identity drift")
    if expected_descriptor is not None:
        descriptor = validate_entry_descriptor(expected_descriptor)
        if result["descriptor_sha256"] != descriptor["aggregate_sha256"] or result["training_run_id"] != descriptor["training_run_id"] or result["nonce"] != descriptor["nonce"]:
            _fail("entry result descriptor binding drift")
        if result["evidence_receipt_path"] != descriptor["evidence_receipt_path"]:
            _fail("entry result evidence receipt binding drift")
        if result["training_evidence_root"] != descriptor["evidence_root"]:
            _fail("entry result training evidence root binding drift")
        for field in (
            "authorization_binding_sha256",
            "authorization_receipt_path",
            "repository",
            "source_bindings",
            "environment",
            "data_roles",
            "training_policy",
            "process_evidence_root",
            "outer_evidence_root",
        ):
            descriptor_field = field
            if field == "repository":
                expected_value = descriptor["repository"]
            elif field == "source_bindings":
                expected_value = descriptor["source_bindings"]
            elif field == "environment":
                expected_value = descriptor["environment"]
            elif field == "data_roles":
                expected_value = descriptor["data_roles"]
            elif field == "training_policy":
                expected_value = descriptor["training_policy"]
            else:
                expected_value = descriptor[descriptor_field]
            if result[field] != expected_value:
                _fail(f"entry result {field} binding drift")
    return _copy(result)


def run_production_entry(
    descriptor: Mapping[str, Any],
    *,
    engine_ports: Mapping[str, Any] | None = None,
    process_pid: int = 0,
) -> dict[str, Any]:
    """Run one authorized child entry and publish a terminal result."""

    checked = validate_entry_descriptor(dict(descriptor))
    if engine_ports is not None:
        _fail("injected production engine ports are forbidden")
    _validate_launch_git_identity(checked)
    consumed = validate_consumed_authorization_receipt(
        checked["authorization_receipt_path"],
        authorization_binding=checked["authorization_binding"],
        authorization_path=checked["authorization_path"],
    )
    root, evidence_claim = _claim_evidence_root(checked)
    entry_receipt = {
        **_entry_receipt_payload(checked),
        "receipt_path": str(root / "entry_consumption.json"),
        "consumed_authorization_receipt_sha256": consumed["receipt_sha256"],
        "evidence_claim_receipt_sha256": evidence_claim["receipt_sha256"],
        "process_pid": process_pid,
    }
    try:
        _publish_json(root, "entry_consumption.json", entry_receipt)
        _publish_json(root, "entry_invocation.json", {"schema_version": T6B_SCHEMA_VERSION, "status": "RUNNING", "descriptor_sha256": checked["aggregate_sha256"], "process_pid": process_pid})
    except Exception as exc:
        _record_persistence_failure(root, checked, "entry_invocation", exc)
        raise TrainingEntryError("entry invocation evidence could not be published") from exc

    result: dict[str, Any]
    try:
        ports = dict(_engine.resolve_production_ports(
            repo_root=checked["repository"]["repo_root"],
            data_roles=checked["data_roles"],
            output_root=checked["evidence_root"],
        ))
        ports["checkpoint_identity"] = {
            "training_run_id": checked["training_run_id"],
            "nonce": checked["nonce"],
            "repository": _copy(checked["repository"]),
            "source_bindings": _copy(checked["source_bindings"]),
            "config_identity": _copy(checked["config_identity"]),
            "data_roles": _copy(checked["data_roles"]),
            "evaluator": "visdrone_official_primary_evaluator_v1/development_only",
        }
        authorization_context = _build_engine_execution_context(checked, consumed, evidence_claim, root)
        engine_result = _engine.run_training_engine(
            checked["training_policy"],
            ports,
            mode="production",
            authorization_context=authorization_context,
        )
        body = {
            "schema_version": T6B_SCHEMA_VERSION,
            "contract_id": T6B_CONTRACT_ID,
            "entry_id": ENTRY_ID,
            "mode": ENTRY_MODE,
            "training_run_id": checked["training_run_id"],
            "nonce": checked["nonce"],
            "descriptor_sha256": checked["aggregate_sha256"],
            "authorization_binding_sha256": checked["authorization_binding_sha256"],
            "authorization_receipt_path": checked["authorization_receipt_path"],
            "evidence_receipt_path": checked["evidence_receipt_path"],
            "repository": _copy(checked["repository"]),
            "source_bindings": _copy(checked["source_bindings"]),
            "environment": _copy(checked["environment"]),
            "data_roles": _copy(checked["data_roles"]),
            "training_policy": _copy(checked["training_policy"]),
            "training_evidence_root": checked["evidence_root"],
            "process_evidence_root": checked["process_evidence_root"],
            "outer_evidence_root": checked["outer_evidence_root"],
            "return_code": 0,
            "engine_result": engine_result,
            "status": "TERMINAL_COMPLETE",
            "classification": "TERMINAL_COMPLETE",
            "failure_class": "NONE",
            "production_training_authorized": True,
            "real_process_launch_authorized": True,
            "entry_receipt_sha256": _sha(_canonical(entry_receipt)),
        }
        result = {**body, "aggregate_result_sha256": _digest(body)}
    except TimeoutError:
        result = _entry_failure(checked, return_code=124, failure_class="RUNNER_TIMEOUT", entry_receipt_sha256=_sha(_canonical(entry_receipt)))
    except KeyboardInterrupt:
        result = _entry_failure(checked, return_code=130, failure_class="RUNNER_SIGNAL", entry_receipt_sha256=_sha(_canonical(entry_receipt)))
    except Exception as exc:
        result = _entry_failure(checked, return_code=5, failure_class="ENGINE_FAILURE", engine_result={"exception_type": type(exc).__name__, "exception_message": str(exc)}, entry_receipt_sha256=_sha(_canonical(entry_receipt)))
    try:
        checked_result = validate_entry_result(result, checked)
        _publish_json(root, "entry_result.json", checked_result)
        rows = _entry_inventory(root)
        inventory = {
            "schema_version": T6B_SCHEMA_VERSION,
            "excluded": ["entry_artifact_inventory.json", "entry_completion.json"],
            "files": rows,
            "canonical_inventory_sha256": _digest(rows),
        }
        _publish_json(root, "entry_artifact_inventory.json", inventory)
        _publish_json(
            root,
            "entry_completion.json",
            {
                "schema_version": T6B_SCHEMA_VERSION,
                "status": checked_result["status"],
                "classification": checked_result["classification"],
                "result_sha256": _sha(_canonical(checked_result)),
                "inventory_sha256": _sha(_canonical(inventory)),
                "evidence_receipt_sha256": evidence_claim["receipt_sha256"],
                "failure_class": checked_result["failure_class"],
            },
        )
        _fsync_directory(root, "process evidence root")
        return checked_result
    except Exception as exc:
        _record_persistence_failure(root, checked, "entry_terminal", exc)
        if isinstance(exc, TrainingEntryError):
            raise
        raise TrainingEntryError("entry terminal evidence could not be published") from exc


def _entry_inventory(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    names = set(ENTRY_REQUIRED_EVIDENCE_FILE_NAMES) - {"entry_artifact_inventory.json", "entry_completion.json"}
    if (root / _engine.ENGINE_CLAIM_FILE_NAME).exists() or (root / _engine.ENGINE_CLAIM_FILE_NAME).is_symlink():
        names.add(_engine.ENGINE_CLAIM_FILE_NAME)
    for name in sorted(names):
        path = root / name
        observed = _regular_file(path, name, expected_mode=0o600, expected_nlink=1)
        raw = _read_stable_file(path, name, mode=0o600)
        rows.append(
            {
                "relative_path": name,
                "size_bytes": len(raw),
                "sha256": _sha(raw),
                "mode": stat.S_IMODE(observed.st_mode),
                "nlink": observed.st_nlink,
            }
        )
    checkpoint_root = root / ENTRY_CHECKPOINT_DIRECTORY
    if checkpoint_root.exists() or checkpoint_root.is_symlink():
        _directory(checkpoint_root, "checkpoint directory")
        for path in sorted(checkpoint_root.iterdir(), key=lambda item: item.name):
            observed = _regular_file(path, f"checkpoint {path.name}", expected_mode=0o600, expected_nlink=1)
            raw = _read_stable_file(path, f"checkpoint {path.name}", mode=0o600)
            rows.append(
                {
                    "relative_path": f"{ENTRY_CHECKPOINT_DIRECTORY}/{path.name}",
                    "size_bytes": len(raw),
                    "sha256": _sha(raw),
                    "mode": stat.S_IMODE(observed.st_mode),
                    "nlink": observed.st_nlink,
                }
            )
    return rows


def classify_entry(process_evidence_root: str | os.PathLike[str]) -> str:
    try:
        root = Path(_absolute_path(os.fspath(process_evidence_root), "process_evidence_root"))
        if _has_valid_persistence_failure_marker(root):
            return "PERMANENT_FAIL"
        if not root.exists() or root.is_symlink():
            return "ABSENT"
        _ensure_process_root(root)
        present = {path.name for path in root.iterdir()}
        if present == {"entry_consumption.json", "entry_invocation.json"}:
            return "RUNNING"
        expected_present = set(ENTRY_REQUIRED_EVIDENCE_FILE_NAMES)
        allowed_present = expected_present | {ENTRY_CHECKPOINT_DIRECTORY, _engine.ENGINE_CLAIM_FILE_NAME}
        if not present.issubset(allowed_present) or not expected_present.issubset(present):
            return "UNKNOWN"
        if ENTRY_CHECKPOINT_DIRECTORY in present:
            try:
                _directory(root / ENTRY_CHECKPOINT_DIRECTORY, "checkpoint directory")
            except Exception:
                return "UNKNOWN"
        consumption, consumption_raw = _read_json(root / "entry_consumption.json", "entry consumption")
        consumption_keys = {"schema_version", "kind", "descriptor_sha256", "authorization_binding_sha256", "training_run_id", "nonce", "status", "receipt_path", "consumed_authorization_receipt_sha256", "evidence_claim_receipt_sha256", "process_pid"}
        if set(consumption) != consumption_keys or consumption_raw != _canonical(consumption):
            return "UNKNOWN"
        if type(consumption.get("process_pid")) is not int or consumption["process_pid"] < 0:
            return "UNKNOWN"
        result, result_raw = _read_json(root / "entry_result.json", "entry result")
        validate_entry_result(result)
        if result["training_evidence_root"] != str(root) or result["evidence_receipt_path"] != str(_evidence_receipt_path(root)):
            return "UNKNOWN"
        if result["status"] == "TERMINAL_COMPLETE":
            claim_path = root / _engine.ENGINE_CLAIM_FILE_NAME
            if not claim_path.exists() or claim_path.is_symlink():
                return "UNKNOWN"
            claim_info = _engine._validate_engine_claim(
                claim_path,
                expected_context_sha256=result["engine_result"]["execution_context_sha256"],
                expected_descriptor_sha256=result["descriptor_sha256"],
                expected_root=str(root),
            )
            if claim_info["sha256"] != result["engine_result"]["engine_claim_sha256"] or len(claim_info["raw"]) != result["engine_result"]["engine_claim_size_bytes"]:
                return "UNKNOWN"
        if consumption["schema_version"] != T6B_SCHEMA_VERSION or consumption["kind"] != "T6B_ENTRY_INVOCATION" or consumption["descriptor_sha256"] != result["descriptor_sha256"] or consumption["authorization_binding_sha256"] != result["authorization_binding_sha256"] or consumption["training_run_id"] != result["training_run_id"] or consumption["nonce"] != result["nonce"] or consumption["status"] != "CLAIMED" or consumption["receipt_path"] != str(root / "entry_consumption.json") or result["entry_receipt_sha256"] != _sha(consumption_raw):
            return "UNKNOWN"
        claim, claim_raw = _read_json(Path(result["evidence_receipt_path"]), "training evidence claim receipt")
        expected_claim = {
            "schema_version": T6B_SCHEMA_VERSION,
            "kind": "T6B_ENTRY_EVIDENCE_CLAIM",
            "descriptor_sha256": result["descriptor_sha256"],
            "descriptor": claim.get("descriptor"),
            "training_run_id": result["training_run_id"],
            "nonce": result["nonce"],
            "status": "CLAIMED",
        }
        if claim_raw != _canonical(claim) or consumption["evidence_claim_receipt_sha256"] != _sha(claim_raw):
            return "UNKNOWN"
        if set(claim) != {"schema_version", "kind", "descriptor_sha256", "descriptor", "training_run_id", "nonce", "status"}:
            return "UNKNOWN"
        try:
            claim_descriptor = validate_entry_descriptor(claim["descriptor"])
        except Exception:
            return "UNKNOWN"
        if (
            claim != {**expected_claim, "descriptor": claim_descriptor}
            or claim_descriptor["aggregate_sha256"] != result["descriptor_sha256"]
            or claim_descriptor["training_run_id"] != result["training_run_id"]
            or claim_descriptor["nonce"] != result["nonce"]
        ):
            return "UNKNOWN"
        try:
            validate_entry_result(result, claim_descriptor)
        except Exception:
            return "UNKNOWN"
        authorization_receipt, authorization_receipt_raw = _read_json(Path(result["authorization_receipt_path"]), "authorization consumption receipt")
        if authorization_receipt_raw != _canonical(authorization_receipt) or authorization_receipt.get("authorization_binding_sha256") != result["authorization_binding_sha256"] or authorization_receipt.get("consumed") is not True or consumption["consumed_authorization_receipt_sha256"] != _sha(authorization_receipt_raw):
            return "UNKNOWN"
        invocation, invocation_raw = _read_json(root / "entry_invocation.json", "entry invocation")
        if invocation_raw != _canonical(invocation) or invocation != {"schema_version": T6B_SCHEMA_VERSION, "status": "RUNNING", "descriptor_sha256": result["descriptor_sha256"], "process_pid": consumption["process_pid"]}:
            return "UNKNOWN"
        inventory, inventory_raw = _read_json(root / "entry_artifact_inventory.json", "entry artifact inventory")
        if type(inventory) is not dict or set(inventory) != {"schema_version", "excluded", "files", "canonical_inventory_sha256"} or inventory["canonical_inventory_sha256"] != _digest(inventory["files"]):
            return "UNKNOWN"
        if inventory["files"] != _entry_inventory(root):
            return "UNKNOWN"
        receipt_raw = _read_stable_file(Path(result["evidence_receipt_path"]), "training evidence receipt", mode=0o600)
        if result["entry_receipt_sha256"] != _sha(_read_stable_file(root / "entry_consumption.json", "entry consumption", mode=0o600)):
            return "UNKNOWN"
        completion, completion_raw = _read_json(root / "entry_completion.json", "entry completion")
        if completion_raw != _canonical(completion):
            return "UNKNOWN"
        expected_completion = {
            "schema_version": T6B_SCHEMA_VERSION,
            "status": result["status"],
            "classification": result["classification"],
            "result_sha256": _sha(result_raw),
            "inventory_sha256": _sha(inventory_raw),
            "evidence_receipt_sha256": _sha(receipt_raw),
            "failure_class": result["failure_class"],
        }
        if completion != expected_completion:
            return "UNKNOWN"
        return result["status"]
    except Exception:
        return "UNKNOWN"


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2 or args[0] != "--descriptor":
        return 2
    descriptor_path = Path(_absolute_path(args[1], "descriptor path"))
    descriptor, _ = _read_json(descriptor_path, "entry descriptor")
    result = run_production_entry(descriptor, process_pid=os.getpid())
    sys.stdout.buffer.write(_canonical(result) + b"\n")
    sys.stdout.buffer.flush()
    return result["return_code"]


if __name__ == "__main__":
    raise SystemExit(main())
