"""T7D production boundary for the frozen baseline-v2a runtime.

The module is inert when imported.  A production invocation accepts only a
detached, digest-bound descriptor and authorization context; model, data and
evaluator ports are resolved internally after the exclusive target claim.
CPU tests use the separate ``run_v2a_cpu_fake`` entry and can never provide
ports to the production entry.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib
import inspect
import json
import math
import os
import pathlib
import signal
import sys
from typing import Any, Iterable, Mapping

from sparse_rtdetr.baseline import training_v2a_runtime as _runtime


PRODUCTION_SCHEMA_VERSION = 1
PRODUCTION_ID = "rtdetrv2_r18_visdrone_baseline_v2a_production_t7d_r1"
PRODUCTION_MODULE_RELATIVE_PATH = "src/sparse_rtdetr/baseline/training_v2a_production.py"
RUNTIME_MODULE_RELATIVE_PATH = "src/sparse_rtdetr/baseline/training_v2a_runtime.py"
CONTRACT_MODULE_RELATIVE_PATH = "src/sparse_rtdetr/baseline/training_v2a_contract.py"
TARGET_RELATIVE_PATH = f"artifacts/training/{PRODUCTION_ID}"
DESCRIPTOR_KIND = "T7D_V2A_DETACHED_LAUNCH_DESCRIPTOR"
AUTHORIZATION_CONTEXT_KIND = "T7D_V2A_AUTHORIZATION_CONTEXT"
TARGET_CLAIM_KIND = "T7D_V2A_TARGET_CLAIM"
TERMINAL_KIND = "T7D_V2A_PROCESS_TERMINAL_RESULT"
CHECKPOINT_KIND = "T7D_V2A_CHECKPOINT"
EVIDENCE_KIND = "T7D_V2A_EXECUTION_EVIDENCE"
PRODUCTION_MODE = "production"
CPU_FAKE_MODE = "cpu_fake"
SUCCESS_STATUS = "TERMINAL_COMPLETE"
FAILURE_STATUS = "PERMANENT_FAIL"


def _target_names() -> dict[str, str]:
    return {
        "lock": "t7d_v2a_target_claim.lock",
        "evidence": "t7d_v2a_execution_evidence.json",
        "progress": "t7d_v2a_epoch_progress.jsonl",
        "terminal": "t7d_v2a_terminal_result.json",
        "stdout": "t7d_v2a_stdout.bin",
        "stderr": "t7d_v2a_stderr.bin",
        "checkpoint_prefix": "t7d_v2a_checkpoint_",
    }


DESCRIPTOR_KEYS = {
    "schema_version",
    "kind",
    "production_id",
    "mode",
    "entry_id",
    "run_id",
    "nonce",
    "repo_root",
    "target_root",
    "target_relative_path",
    "contract_id",
    "baseline_id",
    "contract_sha256",
    "authorization_path",
    "authorization_context_sha256",
    "data_roots",
    "environment_identity",
    "source_identity",
    "target_names",
    "invocation_policy",
    "aggregate_sha256",
}
AUTHORIZATION_CONTEXT_KEYS = {
    "schema_version",
    "kind",
    "authorization_id",
    "run_id",
    "nonce",
    "contract_id",
    "baseline_id",
    "contract_sha256",
    "authorization_path",
    "authorization_size_bytes",
    "authorization_sha256",
    "consumed",
    "consumption_receipt_path",
    "owner",
    "data_roots",
    "environment_identity",
    "descriptor_binding_sha256",
    "context_sha256",
}
POLICY_KEYS = {
    "schema_version",
    "kind",
    "production_id",
    "mode",
    "contract_id",
    "baseline_id",
    "contract_sha256",
    "runtime_policy",
    "target",
    "source_identity",
    "data_roots",
    "environment_identity",
    "model",
    "topology",
    "optimizer",
    "amp",
    "ema",
    "selection",
    "checkpoint",
    "data_roles",
    "production",
    "policy_sha256",
}
RESULT_KEYS = {
    "schema_version",
    "kind",
    "status",
    "mode",
    "production_id",
    "run_id",
    "descriptor_sha256",
    "authorization_context_sha256",
    "contract_sha256",
    "claim",
    "source_identity",
    "data_roots",
    "environment_identity",
    "progress_inventory",
    "checkpoint_inventory",
    "evidence_file",
    "process_identity",
    "optimizer_updates",
    "epochs",
    "selection",
    "call_order",
    "closure",
    "production",
    "terminal_stdout",
    "terminal_file",
}
INVOCATION_POLICY = {
    "overwrite": False,
    "resume": False,
    "retry": False,
    "target_claim": "exclusive_create",
    "authorization_consumption": "deferred_to_exactly_once_launcher",
    "ports": "internal_only",
}

__all__ = (
    "V2AProductionError",
    "build_v2a_production_policy",
    "build_v2a_detached_launch_descriptor",
    "validate_v2a_detached_launch_descriptor",
    "validate_v2a_authorization_context",
    "current_v2a_source_identity",
    "parse_v2a_production_stdout",
    "classify_v2a_terminal",
    "validate_v2a_production_result",
    "run_v2a_production_entry",
    "run_v2a_cpu_fake",
)


class V2AProductionError(ValueError):
    """Raised when the T7D descriptor, capability or evidence boundary fails."""


def _fail(message: str) -> None:
    raise V2AProductionError(message)


def _copy(value: Any) -> Any:
    return copy.deepcopy(value)


def _assert_builtin(value: Any, field: str = "value") -> None:
    if type(value) is dict:
        for key, child in value.items():
            if type(key) is not str:
                _fail(f"{field} contains a non-string key")
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
        raise V2AProductionError("value is not canonical JSON") from exc


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value: Any) -> str:
    return _sha_bytes(_canonical(value))


def _without(value: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    return {key: _copy(child) for key, child in value.items() if key not in set(keys)}


def _exact(value: Any, keys: set[str], field: str) -> dict[str, Any]:
    if type(value) is not dict:
        _fail(f"{field} must be a builtin dict")
    actual = set(value)
    if actual != keys:
        _fail(f"{field} key set drift: missing={sorted(keys - actual)} extra={sorted(actual - keys)}")
    return value


def _string(value: Any, field: str, *, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value):
        _fail(f"{field} must be a builtin string")
    return value


def _bool(value: Any, field: str) -> bool:
    if type(value) is not bool:
        _fail(f"{field} must be a builtin bool")
    return value


def _integer(value: Any, field: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        _fail(f"{field} must be a builtin int")
    if minimum is not None and value < minimum:
        _fail(f"{field} is below its minimum")
    return value


def _sha(value: Any, field: str) -> str:
    value = _string(value, field)
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        _fail(f"{field} is not a lowercase SHA-256")
    return value


def _absolute_path(value: Any, field: str) -> pathlib.Path:
    try:
        raw = os.fspath(value)
    except TypeError as exc:
        raise V2AProductionError(f"{field} is not an absolute path") from exc
    if type(raw) is not str or not raw.startswith("/") or "\\" in raw or "\x00" in raw or "//" in raw:
        _fail(f"{field} is not an absolute path")
    if any(part in {"", ".", ".."} for part in raw.split("/")[1:]):
        _fail(f"{field} has unsafe path components")
    if any(ord(char) < 32 or ord(char) == 127 for char in raw):
        _fail(f"{field} has a control character")
    return pathlib.Path(raw)


def _canonical_directory(value: Any, field: str) -> pathlib.Path:
    path = _absolute_path(value, field)
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise V2AProductionError(f"{field} is unavailable") from exc
    if resolved != path or not path.is_dir() or path.is_symlink():
        _fail(f"{field} is not a canonical directory")
    return path


def _validate_source_identity(value: Any, field: str = "source_identity") -> dict[str, Any]:
    expected = {"production", "runtime", "contract"}
    value = _exact(value, expected, field)
    for name in sorted(expected):
        item = _exact(value[name], {"relative_path", "size_bytes", "sha256"}, f"{field}.{name}")
        _string(item["relative_path"], f"{field}.{name}.relative_path")
        _integer(item["size_bytes"], f"{field}.{name}.size_bytes", minimum=1)
        _sha(item["sha256"], f"{field}.{name}.sha256")
    if value["production"]["relative_path"] != PRODUCTION_MODULE_RELATIVE_PATH:
        _fail("production source relative path drift")
    if value["runtime"]["relative_path"] != RUNTIME_MODULE_RELATIVE_PATH:
        _fail("runtime source relative path drift")
    if value["contract"]["relative_path"] != CONTRACT_MODULE_RELATIVE_PATH:
        _fail("contract source relative path drift")
    return value


def _source_file_identity(root: pathlib.Path, relative: str) -> dict[str, Any]:
    path = root / relative
    try:
        observed = path.lstat()
        raw = path.read_bytes()
    except OSError as exc:
        raise V2AProductionError(f"source identity is unavailable: {relative}") from exc
    if path.is_symlink() or not path.is_file() or observed.st_nlink != 1:
        _fail(f"source identity is not a regular unique file: {relative}")
    return {"relative_path": relative, "size_bytes": len(raw), "sha256": _sha_bytes(raw)}


def current_v2a_source_identity(repo_root: str | os.PathLike[str]) -> dict[str, Any]:
    """Read the three source identities used by the production boundary."""

    root = _canonical_directory(os.fspath(repo_root), "repo_root")
    return {
        "production": _source_file_identity(root, PRODUCTION_MODULE_RELATIVE_PATH),
        "runtime": _source_file_identity(root, RUNTIME_MODULE_RELATIVE_PATH),
        "contract": _source_file_identity(root, CONTRACT_MODULE_RELATIVE_PATH),
    }


def _target_path(root: pathlib.Path, target_root: Any) -> pathlib.Path:
    path = root / TARGET_RELATIVE_PATH if target_root is None else _absolute_path(target_root, "target_root")
    text = str(path)
    forbidden = ("training_t6", "training_t6b", "baseline_v1", "runtime_t7c", "training_v2a_runtime_t7c")
    if any(token in text.casefold() for token in forbidden):
        _fail("T7D target overlaps a V1, T6B or T7C target")
    try:
        parent = path.parent.resolve(strict=True)
    except OSError as exc:
        raise V2AProductionError("T7D target parent is unavailable") from exc
    if parent != path.parent or not parent.is_dir() or path.exists() or path.is_symlink():
        _fail("T7D target must be absent under a canonical parent")
    return path


def _environment_shape(value: Any, field: str) -> dict[str, Any]:
    if type(value) is not dict:
        _fail(f"{field} must be a builtin dict")
    required = {"gpu_name", "cuda_version", "graphics_clock_mhz", "exclusive_host", "other_training_load"}
    allowed = required | {"observation_id"}
    if not set(value) <= allowed or not required <= set(value):
        _fail(f"{field} key set drift")
    _string(value["gpu_name"], f"{field}.gpu_name")
    _string(value["cuda_version"], f"{field}.cuda_version")
    if type(value["graphics_clock_mhz"]) not in {int, float} or type(value["graphics_clock_mhz"]) is bool or not math.isfinite(float(value["graphics_clock_mhz"])):
        _fail(f"{field}.graphics_clock_mhz must be finite")
    _bool(value["exclusive_host"], f"{field}.exclusive_host")
    _integer(value["other_training_load"], f"{field}.other_training_load", minimum=0)
    if "observation_id" in value:
        _string(value["observation_id"], f"{field}.observation_id")
    return value


def _bound_environment_shape(value: Any, field: str) -> dict[str, Any]:
    """Validate the supplied-observation identity returned by T7C."""

    required = {
        "observation_mode",
        "gpu_name",
        "cuda_version",
        "graphics_clock_mhz",
        "graphics_clock_cap_mhz",
        "exclusive_host",
        "other_training_load",
        "hardware_probe_executed",
    }
    if type(value) is not dict or not required <= set(value) or not set(value) <= required | {"observation_id"}:
        _fail(f"{field} key set drift")
    _string(value["observation_mode"], f"{field}.observation_mode")
    _string(value["gpu_name"], f"{field}.gpu_name")
    _string(value["cuda_version"], f"{field}.cuda_version")
    for name in ("graphics_clock_mhz", "graphics_clock_cap_mhz"):
        if type(value[name]) not in {int, float} or type(value[name]) is bool or not math.isfinite(float(value[name])):
            _fail(f"{field}.{name} must be finite")
    _bool(value["exclusive_host"], f"{field}.exclusive_host")
    _integer(value["other_training_load"], f"{field}.other_training_load", minimum=0)
    _bool(value["hardware_probe_executed"], f"{field}.hardware_probe_executed")
    if value["hardware_probe_executed"] is not False:
        _fail(f"{field}.hardware_probe_executed must remain false")
    if "observation_id" in value:
        _string(value["observation_id"], f"{field}.observation_id")
    return value


def _data_roots_shape(value: Any, field: str) -> dict[str, Any]:
    value = _exact(value, {"train_core", "development"}, field)
    for role in ("train_core", "development"):
        item = _exact(value[role], {"role", "root", "annotation"}, f"{field}.{role}")
        if item["role"] != role:
            _fail(f"{field}.{role}.role drift")
        _absolute_path(item["root"], f"{field}.{role}.root")
        annotation = _string(item["annotation"], f"{field}.{role}.annotation")
        if "/" in annotation or annotation in {".", ".."}:
            _fail(f"{field}.{role}.annotation is not a basename")
    return value


def _bound_data_roots_shape(value: Any, field: str) -> dict[str, Any]:
    """Validate the identity-rich object returned by the T7C binder."""

    value = _exact(value, {"roles", "confirmatory_read", "test_read"}, field)
    if value["confirmatory_read"] is not False or value["test_read"] is not False:
        _fail(f"{field} sealed-role read state drift")
    roles = _exact(value["roles"], {"train_core", "development"}, f"{field}.roles")
    for role in ("train_core", "development"):
        item = _exact(
            roles[role],
            {"role", "root", "identity", "annotation", "annotation_identity"},
            f"{field}.roles.{role}",
        )
        if item["role"] != role:
            _fail(f"{field}.roles.{role}.role drift")
        _absolute_path(item["root"], f"{field}.roles.{role}.root")
        _string(item["annotation"], f"{field}.roles.{role}.annotation")
        if "/" in item["annotation"]:
            _fail(f"{field}.roles.{role}.annotation is not a basename")
        for identity_name in ("identity", "annotation_identity"):
            identity = item[identity_name]
            if type(identity) is not dict:
                _fail(f"{field}.roles.{role}.{identity_name} is not an identity")
            for key in ("device", "inode", "mode", "nlink", "uid", "gid"):
                _integer(identity.get(key), f"{field}.roles.{role}.{identity_name}.{key}", minimum=0)
            if identity_name == "annotation_identity":
                _integer(identity.get("size_bytes"), f"{field}.roles.{role}.annotation_identity.size_bytes", minimum=0)
    return value


def _context_digest(value: Mapping[str, Any]) -> str:
    return _digest(_without(value, "context_sha256", "descriptor_binding_sha256"))


def validate_v2a_authorization_context(
    context: Mapping[str, Any],
    *,
    descriptor: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate an owner-provided, still-unconsumed authorization context."""

    value = _exact(context, AUTHORIZATION_CONTEXT_KEYS, "authorization context")
    _assert_builtin(value, "authorization context")
    if value["schema_version"] != PRODUCTION_SCHEMA_VERSION or value["kind"] != AUTHORIZATION_CONTEXT_KIND:
        _fail("authorization context schema identity drift")
    for field in ("authorization_id", "run_id", "nonce", "contract_id", "baseline_id"):
        _string(value[field], f"authorization context.{field}")
    _sha(value["contract_sha256"], "authorization context.contract_sha256")
    if value["consumed"] is not False:
        _fail("authorization is already consumed")
    if value["consumption_receipt_path"] is not None:
        _fail("authorization consumption receipt must be absent before target claim")
    _absolute_path(value["authorization_path"], "authorization context.authorization_path")
    _integer(value["authorization_size_bytes"], "authorization context.authorization_size_bytes", minimum=0)
    _sha(value["authorization_sha256"], "authorization context.authorization_sha256")
    owner = _exact(value["owner"], {"owner_id"}, "authorization context.owner")
    _string(owner["owner_id"], "authorization context.owner.owner_id")
    _data_roots_shape(value["data_roots"], "authorization context.data_roots")
    _environment_shape(value["environment_identity"], "authorization context.environment_identity")
    _sha(value["descriptor_binding_sha256"], "authorization context.descriptor_binding_sha256")
    _sha(value["context_sha256"], "authorization context.context_sha256")
    if value["context_sha256"] != _context_digest(value):
        _fail("authorization context digest drift")
    if descriptor is not None:
        checked_descriptor = validate_v2a_detached_launch_descriptor(descriptor)
        descriptor_binding = _digest(_without(checked_descriptor, "authorization_context_sha256", "aggregate_sha256"))
        if value["descriptor_binding_sha256"] != descriptor_binding:
            _fail("authorization context descriptor cross-binding drift")
        if checked_descriptor["authorization_context_sha256"] != _context_digest(value):
            _fail("descriptor authorization-context digest drift")
        if checked_descriptor["run_id"] != value["run_id"] or checked_descriptor["nonce"] != value["nonce"]:
            _fail("authorization context run identity drift")
        if checked_descriptor["contract_sha256"] != value["contract_sha256"]:
            _fail("authorization context contract digest drift")
    return _copy(value)


def validate_v2a_detached_launch_descriptor(descriptor: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a closed descriptor without reading its authorization artifact."""

    value = _exact(descriptor, DESCRIPTOR_KEYS, "detached launch descriptor")
    _assert_builtin(value, "detached launch descriptor")
    if value["schema_version"] != PRODUCTION_SCHEMA_VERSION or value["kind"] != DESCRIPTOR_KIND:
        _fail("detached descriptor schema identity drift")
    if value["production_id"] != PRODUCTION_ID or value["entry_id"] != PRODUCTION_ID or value["mode"] != PRODUCTION_MODE:
        _fail("detached descriptor production identity drift")
    for field in ("run_id", "nonce", "contract_id", "baseline_id"):
        _string(value[field], f"detached descriptor.{field}")
    _absolute_path(value["repo_root"], "detached descriptor.repo_root")
    target = _absolute_path(value["target_root"], "detached descriptor.target_root")
    if value["target_relative_path"] not in {TARGET_RELATIVE_PATH, None}:
        _fail("detached descriptor target relative path drift")
    if any(token in str(target).casefold() for token in ("training_t6", "training_t6b", "baseline_v1", "runtime_t7c")):
        _fail("detached descriptor target overlaps an older production surface")
    _sha(value["contract_sha256"], "detached descriptor.contract_sha256")
    _absolute_path(value["authorization_path"], "detached descriptor.authorization_path")
    _sha(value["authorization_context_sha256"], "detached descriptor.authorization_context_sha256")
    _data_roots_shape(value["data_roots"], "detached descriptor.data_roots")
    _environment_shape(value["environment_identity"], "detached descriptor.environment_identity")
    _validate_source_identity(value["source_identity"])
    if value["target_names"] != _target_names():
        _fail("detached descriptor target names are not independent T7D names")
    if value["invocation_policy"] != INVOCATION_POLICY:
        _fail("detached descriptor invocation policy drift")
    _sha(value["aggregate_sha256"], "detached descriptor.aggregate_sha256")
    if value["aggregate_sha256"] != _digest(_without(value, "aggregate_sha256")):
        _fail("detached descriptor aggregate digest drift")
    return _copy(value)


def build_v2a_detached_launch_descriptor(
    repo_root: str | os.PathLike[str],
    *,
    authorization_context: Mapping[str, Any],
    run_id: str,
    nonce: str,
    target_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Bind an external context to a descriptor without consuming it."""

    root = _canonical_directory(os.fspath(repo_root), "repo_root")
    context = validate_v2a_authorization_context(authorization_context)
    _string(run_id, "run_id")
    _string(nonce, "nonce")
    if context["run_id"] != run_id or context["nonce"] != nonce:
        _fail("authorization context run identity does not match descriptor request")
    target = _target_path(root, target_root)
    source = current_v2a_source_identity(root)
    descriptor = {
        "schema_version": PRODUCTION_SCHEMA_VERSION,
        "kind": DESCRIPTOR_KIND,
        "production_id": PRODUCTION_ID,
        "mode": PRODUCTION_MODE,
        "entry_id": PRODUCTION_ID,
        "run_id": run_id,
        "nonce": nonce,
        "repo_root": str(root),
        "target_root": str(target),
        "target_relative_path": TARGET_RELATIVE_PATH if target == root / TARGET_RELATIVE_PATH else None,
        "contract_id": context["contract_id"],
        "baseline_id": context["baseline_id"],
        "contract_sha256": context["contract_sha256"],
        "authorization_path": context["authorization_path"],
        "authorization_context_sha256": _context_digest(context),
        "data_roots": _copy(context["data_roots"]),
        "environment_identity": _copy(context["environment_identity"]),
        "source_identity": source,
        "target_names": _target_names(),
        "invocation_policy": _copy(INVOCATION_POLICY),
    }
    descriptor_binding = _digest(_without(descriptor, "authorization_context_sha256", "aggregate_sha256"))
    if context["descriptor_binding_sha256"] not in {descriptor_binding, "0" * 64}:
        _fail("authorization context was bound to a different descriptor")
    context["descriptor_binding_sha256"] = descriptor_binding
    descriptor["authorization_context_sha256"] = _context_digest(context)
    descriptor["aggregate_sha256"] = _digest(descriptor)
    checked_descriptor = validate_v2a_detached_launch_descriptor(descriptor)
    # The descriptor binding is finalized here, so publish it back to the
    # caller-owned context only after the complete descriptor has validated.
    # This keeps the detached context and descriptor usable as one
    # cross-bound authorization pair.
    if type(authorization_context) is not dict:
        _fail("authorization context must be a builtin dict")
    authorization_context["descriptor_binding_sha256"] = descriptor_binding
    authorization_context["context_sha256"] = context["context_sha256"]
    validate_v2a_authorization_context(authorization_context, descriptor=checked_descriptor)
    return checked_descriptor


def _validate_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    value = _exact(policy, POLICY_KEYS, "T7D production policy")
    _assert_builtin(value, "T7D production policy")
    if value["schema_version"] != PRODUCTION_SCHEMA_VERSION or value["kind"] != "T7D_V2A_PRODUCTION_POLICY" or value["production_id"] != PRODUCTION_ID:
        _fail("T7D policy identity drift")
    if value["mode"] != "T7D_PRODUCTION_BOUNDARY":
        _fail("T7D policy mode drift")
    _string(value["contract_id"], "T7D policy.contract_id")
    _string(value["baseline_id"], "T7D policy.baseline_id")
    _sha(value["contract_sha256"], "T7D policy.contract_sha256")
    if type(value["runtime_policy"]) is not dict or value["runtime_policy"].get("contract_sha256") != value["contract_sha256"]:
        _fail("T7D policy runtime binding drift")
    target = _exact(value["target"], {"path", "relative_path", "names"}, "T7D policy.target")
    _absolute_path(target["path"], "T7D policy.target.path")
    if target["relative_path"] not in {TARGET_RELATIVE_PATH, None} or target["names"] != _target_names():
        _fail("T7D policy target identity drift")
    _validate_source_identity(value["source_identity"])
    _bound_data_roots_shape(value["data_roots"], "T7D policy.data_roots")
    _bound_environment_shape(value["environment_identity"], "T7D policy.environment_identity")
    if value["production"] != {
        "owner_authorization_consumed": False,
        "gpu_probe_executed": False,
        "formal_training_executed": False,
        "training_ready": False,
        "independent_audit_pass": False,
    }:
        _fail("T7D production state must remain fail-closed")
    _sha(value["policy_sha256"], "T7D policy.policy_sha256")
    if value["policy_sha256"] != _digest(_without(value, "policy_sha256")):
        _fail("T7D policy digest drift")
    return _copy(value)


def build_v2a_production_policy(
    repo_root: str | os.PathLike[str],
    *,
    data_roots: Mapping[str, str | os.PathLike[str]],
    environment_identity: Mapping[str, Any],
    target_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Bind T7B/T7C and add only the independent T7D target identity."""

    root = _canonical_directory(os.fspath(repo_root), "repo_root")
    target = _target_path(root, target_root)
    runtime_policy = _runtime.build_v2a_runtime_policy(
        root,
        data_roots=data_roots,
        environment_identity=environment_identity,
        target_root=str(target),
    )
    if type(runtime_policy) is not dict or runtime_policy.get("production", {}).get("formal_training_executed") is not False:
        _fail("T7C runtime policy is not detached")
    source = current_v2a_source_identity(root)
    policy = {
        "schema_version": PRODUCTION_SCHEMA_VERSION,
        "kind": "T7D_V2A_PRODUCTION_POLICY",
        "production_id": PRODUCTION_ID,
        "mode": "T7D_PRODUCTION_BOUNDARY",
        "contract_id": runtime_policy["contract_id"],
        "baseline_id": runtime_policy["baseline_id"],
        "contract_sha256": runtime_policy["contract_sha256"],
        "runtime_policy": _copy(runtime_policy),
        "target": {
            "path": str(target),
            "relative_path": TARGET_RELATIVE_PATH if target == root / TARGET_RELATIVE_PATH else None,
            "names": _target_names(),
        },
        "source_identity": source,
        "data_roots": _copy(runtime_policy["data_roots"]),
        "environment_identity": _copy(runtime_policy["environment_identity"]),
        "model": _copy(runtime_policy["model"]),
        "topology": _copy(runtime_policy["topology"]),
        "optimizer": _copy(runtime_policy["optimizer"]),
        "amp": _copy(runtime_policy["amp"]),
        "ema": _copy(runtime_policy["ema"]),
        "selection": _copy(runtime_policy["selection"]),
        "checkpoint": _copy(runtime_policy["checkpoint"]),
        "data_roles": _copy(runtime_policy["data_roles"]),
        "production": {
            "owner_authorization_consumed": False,
            "gpu_probe_executed": False,
            "formal_training_executed": False,
            "training_ready": False,
            "independent_audit_pass": False,
        },
    }
    policy["policy_sha256"] = _digest(policy)
    return _validate_policy(policy)


def _regular_file(path: pathlib.Path, field: str, *, mode: int | None = None) -> dict[str, Any]:
    try:
        observed = path.lstat()
    except OSError as exc:
        raise V2AProductionError(f"{field} is unavailable") from exc
    if path.is_symlink() or not path.is_file() or observed.st_nlink != 1:
        _fail(f"{field} is not a regular unique file")
    if mode is not None and (observed.st_mode & 0o777) != mode:
        _fail(f"{field} mode drift")
    return {"size_bytes": observed.st_size, "mode": observed.st_mode & 0o777, "nlink": observed.st_nlink}


def _fsync_directory(path: pathlib.Path, field: str) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        fd = os.open(path, flags)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError as exc:
        raise V2AProductionError(f"{field} directory fsync failed") from exc


def _write_exclusive_bytes(path: pathlib.Path, raw: bytes, field: str, *, mode: int = 0o600) -> dict[str, Any]:
    if path.exists() or path.is_symlink() or not path.parent.is_dir() or path.parent.is_symlink():
        _fail(f"{field} target already exists or parent is unsafe")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags, mode)
        with os.fdopen(fd, "wb", buffering=0) as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        observed = _regular_file(path, field, mode=mode)
        if observed["size_bytes"] != len(raw):
            _fail(f"{field} size changed during publication")
        if path.read_bytes() != raw:
            _fail(f"{field} readback changed during publication")
        _fsync_directory(path.parent, field)
    except OSError as exc:
        raise V2AProductionError(f"{field} publication failed") from exc
    return {"path": str(path), "size_bytes": len(raw), "sha256": _sha_bytes(raw), "mode": mode}


def _write_exclusive_json(path: pathlib.Path, value: Mapping[str, Any], field: str) -> dict[str, Any]:
    return _write_exclusive_bytes(path, _canonical(value), field)


def _claim_target(
    policy: Mapping[str, Any],
    *,
    descriptor_sha256: str,
    authorization_context_sha256: str,
    mode: str,
) -> dict[str, Any]:
    checked = _validate_policy(policy)
    target = pathlib.Path(checked["target"]["path"])
    if target.exists() or target.is_symlink():
        _fail("T7D target already exists; overwrite/resume/retry is forbidden")
    try:
        target.mkdir(mode=0o700, parents=False, exist_ok=False)
        _fsync_directory(target.parent, "T7D target claim parent")
    except OSError as exc:
        raise V2AProductionError("T7D exclusive target claim failed") from exc
    payload = {
        "schema_version": PRODUCTION_SCHEMA_VERSION,
        "kind": TARGET_CLAIM_KIND,
        "production_id": PRODUCTION_ID,
        "mode": mode,
        "descriptor_sha256": descriptor_sha256,
        "authorization_context_sha256": authorization_context_sha256,
        "contract_sha256": checked["contract_sha256"],
        "target_path": str(target),
        "status": "CLAIMED",
    }
    lock = _write_exclusive_json(target / checked["target"]["names"]["lock"], payload, "T7D target lock")
    return {**payload, "file": lock}


def _verify_authorization_artifact(context: Mapping[str, Any]) -> None:
    path = _absolute_path(context["authorization_path"], "authorization path")
    try:
        observed = path.lstat()
        raw = path.read_bytes()
    except OSError as exc:
        raise V2AProductionError("detached authorization artifact is unavailable") from exc
    if path.is_symlink() or not path.is_file() or observed.st_nlink != 1:
        _fail("detached authorization artifact is not a regular unique file")
    if len(raw) != context["authorization_size_bytes"] or _sha_bytes(raw) != context["authorization_sha256"]:
        _fail("detached authorization artifact identity drift")


def _call_port(function: Any, candidates: tuple[tuple[Any, ...], ...], field: str) -> Any:
    if not callable(function):
        _fail(f"{field} is not callable")
    try:
        parameters = list(inspect.signature(function).parameters.values())
    except (TypeError, ValueError) as exc:
        raise V2AProductionError(f"{field} signature is unavailable") from exc
    positional = [item for item in parameters if item.kind in {item.POSITIONAL_ONLY, item.POSITIONAL_OR_KEYWORD}]
    accepts_varargs = any(item.kind == item.VAR_POSITIONAL for item in parameters)
    for args in candidates:
        if accepts_varargs or len(positional) >= len(args):
            try:
                return function(*args)
            except TypeError:
                continue
    _fail(f"{field} signature is incompatible")


def _finite_loss(value: Any) -> bool:
    if type(value) in {int, float} and type(value) is not bool:
        return math.isfinite(float(value))
    item = getattr(value, "item", None)
    if callable(item):
        scalar = item()
        return type(scalar) in {int, float} and type(scalar) is not bool and math.isfinite(float(scalar))
    return True


def _loss_total(value: Any) -> Any:
    if type(value) is dict:
        if not value:
            _fail("loss dictionary is empty")
        values = list(value.values())
        result = values[0]
        for child in values[1:]:
            result = result + child
        return result
    return value


def _state_payload(value: Any, field: str) -> Any:
    if callable(value):
        value = value()
    state_dict = getattr(value, "state_dict", None)
    if callable(state_dict):
        value = state_dict()

    def convert(child: Any, child_field: str) -> Any:
        if type(child) is dict:
            return {str(key): convert(item, f"{child_field}.{key}") for key, item in child.items()}
        if type(child) is list:
            return [convert(item, f"{child_field}[{index}]") for index, item in enumerate(child)]
        if type(child) is tuple:
            return [convert(item, f"{child_field}[{index}]") for index, item in enumerate(child)]
        if type(child) in {str, int, float, bool, type(None)}:
            return child
        tolist = getattr(child, "tolist", None)
        if callable(tolist):
            return convert(tolist(), child_field)
        detach = getattr(child, "detach", None)
        if callable(detach):
            detached = detach()
            cpu = getattr(detached, "cpu", None)
            return convert(cpu() if callable(cpu) else detached, child_field)
        return {"opaque_type": f"{type(child).__module__}.{type(child).__qualname__}", "repr": repr(child)}

    result = convert(value, field)
    _assert_builtin(result, field)
    return result


def _checkpoint(
    policy: Mapping[str, Any],
    *,
    run_id: str,
    epoch: int,
    optimizer_updates: int,
    raw_model: Any,
    ema: Any,
    optimizer: Any,
    scheduler: Any,
    rng_state: Any,
    source_identity: Mapping[str, Any],
) -> dict[str, Any]:
    checked = _validate_policy(policy)
    binding = checked["runtime_policy"]["contract_binding"]
    value = {
        "schema_version": PRODUCTION_SCHEMA_VERSION,
        "kind": CHECKPOINT_KIND,
        "production_id": PRODUCTION_ID,
        "run_id": run_id,
        "contract_sha256": checked["contract_sha256"],
        "contract_identity": {
            "raw_sha256": binding["raw_sha256"],
            "canonical_sha256": binding["canonical_sha256"],
        },
        "authority_identity": _copy(binding["authority_binding"]),
        "source_identity": _copy(source_identity),
        "data_roots": _copy(checked["data_roots"]),
        "environment_identity": _copy(checked["environment_identity"]),
        "epoch": _integer(epoch, "checkpoint.epoch", minimum=1),
        "global_optimizer_step": _integer(optimizer_updates, "checkpoint.global_optimizer_step", minimum=1),
        "raw_model": _state_payload(raw_model, "checkpoint.raw_model"),
        "ema": _state_payload(ema, "checkpoint.ema"),
        "optimizer": _state_payload(optimizer, "checkpoint.optimizer"),
        "scheduler": _state_payload(scheduler, "checkpoint.scheduler"),
        "rng_state": _state_payload(rng_state, "checkpoint.rng_state"),
        "amp_state": {
            "enabled": checked["amp"]["enabled"],
            "autocast_dtype": checked["amp"]["autocast_dtype"],
            "grad_scaler_enabled": checked["amp"]["grad_scaler_enabled"],
            "scaler_policy": checked["amp"]["scaler_policy"],
        },
        "evaluation_weights": "ema",
    }
    if value["raw_model"] == value["ema"]:
        _fail("checkpoint raw and EMA states are identical")
    value["checkpoint_sha256"] = _digest(value)
    return value


def _validate_checkpoint(value: Mapping[str, Any], policy: Mapping[str, Any], run_id: str) -> dict[str, Any]:
    required = {
        "schema_version", "kind", "production_id", "run_id", "contract_sha256", "contract_identity",
        "authority_identity", "source_identity", "data_roots", "environment_identity", "epoch",
        "global_optimizer_step", "raw_model", "ema", "optimizer", "scheduler", "rng_state", "amp_state",
        "evaluation_weights", "checkpoint_sha256",
    }
    checked = _exact(value, required, "T7D checkpoint")
    _assert_builtin(checked, "T7D checkpoint")
    if checked["schema_version"] != PRODUCTION_SCHEMA_VERSION or checked["kind"] != CHECKPOINT_KIND or checked["production_id"] != PRODUCTION_ID or checked["run_id"] != run_id:
        _fail("T7D checkpoint identity drift")
    if checked["contract_sha256"] != policy["contract_sha256"]:
        _fail("T7D checkpoint contract drift")
    _validate_source_identity(checked["source_identity"])
    if checked["source_identity"] != policy["source_identity"]:
        _fail("T7D checkpoint source identity drift")
    _bound_data_roots_shape(checked["data_roots"], "T7D checkpoint.data_roots")
    if checked["data_roots"] != policy["data_roots"]:
        _fail("T7D checkpoint data identity drift")
    _bound_environment_shape(checked["environment_identity"], "T7D checkpoint.environment_identity")
    if checked["environment_identity"] != policy["environment_identity"]:
        _fail("T7D checkpoint environment identity drift")
    amp = _exact(checked["amp_state"], {"enabled", "autocast_dtype", "grad_scaler_enabled", "scaler_policy"}, "T7D checkpoint.amp_state")
    if amp["autocast_dtype"] != "bfloat16" or amp["grad_scaler_enabled"] is not False or amp["scaler_policy"] != "disabled_absent":
        _fail("T7D checkpoint AMP state drift")
    if checked["evaluation_weights"] != "ema" or checked["raw_model"] == checked["ema"]:
        _fail("T7D checkpoint EMA binding drift")
    _sha(checked["checkpoint_sha256"], "T7D checkpoint.checkpoint_sha256")
    if checked["checkpoint_sha256"] != _digest(_without(checked, "checkpoint_sha256")):
        _fail("T7D checkpoint digest drift")
    return _copy(checked)


def _append_progress(path: pathlib.Path, row: Mapping[str, Any], expected_epoch: int, seen: set[int]) -> None:
    epoch = _integer(row.get("epoch"), "progress.epoch", minimum=1)
    if epoch != expected_epoch or epoch in seen:
        _fail("T7D epoch progress sequence drift")
    payload = _canonical(dict(row)) + b"\n"
    created = not path.exists()
    if created:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(path, flags, 0o600)
            handle = os.fdopen(fd, "ab", buffering=0)
        except OSError as exc:
            raise V2AProductionError("T7D epoch progress creation failed") from exc
    else:
        _regular_file(path, "T7D epoch progress", mode=0o600)
        try:
            handle = path.open("ab", buffering=0)
        except OSError as exc:
            raise V2AProductionError("T7D epoch progress open failed") from exc
    try:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    except OSError as exc:
        raise V2AProductionError("T7D epoch progress fsync failed") from exc
    finally:
        handle.close()
    if created:
        _fsync_directory(path.parent, "T7D epoch progress")
    seen.add(epoch)


def _progress_inventory(path: pathlib.Path, expected_epochs: int) -> dict[str, Any]:
    _regular_file(path, "T7D epoch progress", mode=0o600)
    raw = path.read_bytes()
    rows = []
    for line in raw.splitlines(keepends=True):
        if not line.endswith(b"\n"):
            _fail("T7D epoch progress has an unterminated line")
        try:
            value = json.loads(line[:-1].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise V2AProductionError("T7D epoch progress JSON is invalid") from exc
        if type(value) is not dict or _canonical(value) != line[:-1]:
            _fail("T7D epoch progress is not canonical JSONL")
        rows.append(value)
    if [row.get("epoch") for row in rows] != list(range(1, expected_epochs + 1)):
        _fail("T7D epoch progress is incomplete")
    return {
        "path": str(path),
        "relative_name": path.name,
        "size_bytes": len(raw),
        "sha256": _sha_bytes(raw),
        "mode": 0o600,
        "line_count": len(rows),
        "rows_sha256": _digest(rows),
        "first_epoch": rows[0]["epoch"],
        "last_epoch": rows[-1]["epoch"],
    }


def _get_port(ports: Mapping[str, Any], name: str, *, required: bool = True) -> Any:
    if type(ports) is not dict:
        _fail("CPU fake ports must be a builtin dict")
    value = ports.get(name)
    if value is None and required:
        _fail(f"CPU fake port is required: {name}")
    return value


def _call_evaluator(evaluator: Any, ema: Any, epoch: int) -> Any:
    function = getattr(evaluator, "evaluate", None)
    if function is None and callable(evaluator):
        function = evaluator
    return _call_port(function, ((ema, epoch), (epoch, ema), (ema,), (epoch,), ()), "primary evaluator")


def _normalize_metrics(value: Any, policy: Mapping[str, Any]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {"AP", "AP50", "AR500"}:
        _fail("primary evaluator metrics schema drift")
    result = {}
    for field in ("AP", "AP50", "AR500"):
        item = value[field]
        if type(item) not in {int, float} or type(item) is bool or not math.isfinite(float(item)):
            _fail(f"primary evaluator metric is not finite: {field}")
        result[field] = float(item)
    result["weights"] = policy["ema"]["model_selection_weights"]
    if result["weights"] != "ema":
        _fail("primary evaluator is not EMA-backed")
    return result


def _model_state(value: Any, field: str, epoch: int) -> Any:
    if value is None:
        return {"epoch": epoch, "field": field}
    return _state_payload(value, field)


def _state_from_port(ports: Mapping[str, Any], name: str, fallback: Any) -> Any:
    value = ports.get(name)
    if value is not None:
        return value
    return fallback


class _CapabilityToken:
    pass


class _ProductionCapability:
    __slots__ = ("policy", "mode", "model", "optimizer", "ema", "evaluator", "batches", "compute_loss", "backward", "scheduler", "rng_state", "source_identity", "torch_module", "call_order", "raw_model_state", "ema_state")

    def __init__(self, token: object, **values: Any) -> None:
        if token is not _CAPABILITY_TOKEN:
            raise TypeError("production capability is internal")
        for key, value in values.items():
            setattr(self, key, value)


_CAPABILITY_TOKEN = _CapabilityToken


def _make_fake_capability(policy: Mapping[str, Any], ports: Mapping[str, Any], call_order: list[str]) -> _ProductionCapability:
    if any(key in ports for key in {"production_capability", "authorization_context", "real_data", "real_model"}):
        _fail("CPU fake cannot inject production capabilities")
    strict_load = ports.get("strict_load")
    call_order.append("strict_load")
    if callable(strict_load):
        loaded = strict_load()
        model = loaded.get("model") if type(loaded) is dict else loaded
    else:
        if ports.get("strict_load_pass", True) is not True:
            _fail("CPU fake strict load did not pass")
        model = _get_port(ports, "model")
    optimizer_factory = ports.get("optimizer_factory")
    call_order.append("create_optimizer")
    optimizer = optimizer_factory(model) if callable(optimizer_factory) else _get_port(ports, "optimizer")
    ema_factory = ports.get("ema_factory")
    ema = ema_factory(model) if callable(ema_factory) else _get_port(ports, "ema")
    call_order.append("create_data_ports")
    batches = _get_port(ports, "batches")
    compute_loss = _get_port(ports, "compute_loss")
    backward = _get_port(ports, "backward", required=False)
    scheduler = _get_port(ports, "scheduler", required=False)
    rng_state = _state_from_port(ports, "rng_state", {"mode": CPU_FAKE_MODE, "seed": 0})
    raw_model_state = _get_port(ports, "raw_model_state", required=False)
    ema_state = _get_port(ports, "ema_state", required=False)
    call_order.append("create_primary_evaluator")
    evaluator = _get_port(ports, "evaluator")
    evaluator_id = getattr(evaluator, "evaluator_id", None)
    if isinstance(evaluator, Mapping):
        evaluator_id = evaluator.get("evaluator_id", evaluator_id)
    if evaluator_id != policy["selection"]["primary_evaluator"]:
        _fail("CPU fake evaluator is not the frozen primary evaluator")
    return _ProductionCapability(
        _CAPABILITY_TOKEN,
        policy=_copy(policy),
        mode=CPU_FAKE_MODE,
        model=model,
        optimizer=optimizer,
        ema=ema,
        evaluator=evaluator,
        batches=batches,
        compute_loss=compute_loss,
        backward=backward,
        scheduler=scheduler,
        rng_state=rng_state,
        source_identity=_copy(policy["source_identity"]),
        torch_module=None,
        call_order=call_order,
        raw_model_state=raw_model_state,
        ema_state=ema_state,
    )


def _production_autocast(torch_module: Any, binding: Mapping[str, Any]) -> Any:
    return _runtime.v2a_bf16_autocast_context(binding, torch_module=torch_module)


def _run_epoch_loop(capability: _ProductionCapability, *, epochs: int, run_id: str, claim: Mapping[str, Any]) -> dict[str, Any]:
    policy = _validate_policy(capability.policy)
    epochs = _integer(epochs, "epochs", minimum=1)
    capability.call_order.append("epoch_loop")
    target = pathlib.Path(policy["target"]["path"])
    names = policy["target"]["names"]
    progress_path = target / names["progress"]
    seen_epochs: set[int] = set()
    checkpoint_inventory: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    optimizer_updates = 0
    last_checkpoint: dict[str, Any] | None = None
    for epoch in range(1, epochs + 1):
        source = capability.batches() if callable(capability.batches) else capability.batches
        try:
            batches = iter(source)
        except TypeError as exc:
            raise V2AProductionError("training batches are not iterable") from exc
        batch_count = 0
        mean_loss = 0.0
        for batch in batches:
            if isinstance(batch, Mapping) and "batch_size" in batch and batch["batch_size"] != policy["topology"]["train_micro_batch"]:
                _fail("runtime batch adaptation detected")
            zero_grad = getattr(capability.optimizer, "zero_grad", None)
            if callable(zero_grad):
                zero_grad()
            context = contextlib.nullcontext()
            if capability.mode == PRODUCTION_MODE:
                context = _production_autocast(capability.torch_module, policy["runtime_policy"]["contract_binding"])
            with context:
                loss_value = _call_port(capability.compute_loss, ((capability.model, batch, epoch), (batch, epoch), (capability.model, batch), (batch,)), "compute_loss")
                loss_value = _loss_total(loss_value)
                if not _finite_loss(loss_value):
                    _fail("loss is non-finite")
                backward = capability.backward
                if callable(backward):
                    _call_port(backward, ((loss_value,), ()), "backward")
                else:
                    backward_method = getattr(loss_value, "backward", None)
                    if not callable(backward_method):
                        _fail("loss has no direct backward operation")
                    backward_method()
            step = getattr(capability.optimizer, "step", None)
            if not callable(step):
                _fail("optimizer has no direct step")
            step_result = step()
            if step_result is False or (isinstance(step_result, Mapping) and step_result.get("skipped") is True):
                _fail("optimizer step was skipped")
            update = getattr(capability.ema, "update", None)
            if callable(update):
                update(capability.model)
            elif callable(capability.ema):
                capability.ema(capability.model)
            else:
                _fail("EMA update port is missing")
            optimizer_updates += 1
            batch_count += 1
            if type(loss_value) in {int, float} and type(loss_value) is not bool:
                mean_loss += float(loss_value)
        if batch_count == 0:
            _fail("epoch has no training batches")
        if callable(capability.scheduler):
            capability.scheduler()
        metrics = _normalize_metrics(_call_evaluator(capability.evaluator, capability.ema, epoch), policy)
        candidates.append({"epoch": epoch, **metrics})
        selected = _runtime.select_v2a_development_candidate(candidates, policy["runtime_policy"]["contract_binding"])
        last_checkpoint = _checkpoint(
            policy,
            run_id=run_id,
            epoch=epoch,
            optimizer_updates=optimizer_updates,
            raw_model=_model_state(capability.raw_model_state if capability.raw_model_state is not None else capability.model, "raw_model", epoch),
            ema=_model_state(capability.ema_state if capability.ema_state is not None else capability.ema, "ema", epoch),
            optimizer=capability.optimizer,
            scheduler=capability.scheduler if capability.scheduler is not None else {"state": "absent"},
            rng_state=capability.rng_state,
            source_identity=capability.source_identity,
        )
        last_checkpoint = _validate_checkpoint(last_checkpoint, policy, run_id)
        checkpoint_path = target / f"{names['checkpoint_prefix']}last_epoch_{epoch}.json"
        checkpoint_ref = _write_exclusive_json(checkpoint_path, last_checkpoint, "T7D last checkpoint")
        checkpoint_inventory.append({"role": "last", "epoch": epoch, **checkpoint_ref})
        if selected["epoch"] == epoch:
            best_ref = _write_exclusive_json(target / f"{names['checkpoint_prefix']}best_epoch_{epoch}.json", last_checkpoint, "T7D best checkpoint")
            checkpoint_inventory.append({"role": "best", "epoch": epoch, **best_ref})
        periodic_frequency = policy["checkpoint"]["periodic_checkpoint_frequency_epochs"]
        if epoch % periodic_frequency == 0:
            periodic_ref = _write_exclusive_json(target / f"{names['checkpoint_prefix']}periodic_epoch_{epoch}.json", last_checkpoint, "T7D periodic checkpoint")
            checkpoint_inventory.append({"role": "periodic", "epoch": epoch, **periodic_ref})
        _append_progress(
            progress_path,
            {
                "schema_version": PRODUCTION_SCHEMA_VERSION,
                "production_id": PRODUCTION_ID,
                "run_id": run_id,
                "epoch": epoch,
                "batch_count": batch_count,
                "mean_loss": mean_loss / batch_count,
                "optimizer_updates": optimizer_updates,
                "evaluation": metrics,
                "selected_epoch": selected["epoch"],
                "checkpoint_sha256": last_checkpoint["checkpoint_sha256"],
            },
            epoch,
            seen_epochs,
        )
    if last_checkpoint is None or selected is None:
        _fail("epoch loop did not produce a terminal checkpoint")
    final_epoch = epochs
    final_ref = _write_exclusive_json(target / f"{names['checkpoint_prefix']}final_epoch_{final_epoch}.json", last_checkpoint, "T7D final checkpoint")
    checkpoint_inventory.append({"role": "final", "epoch": final_epoch, **final_ref})
    progress = _progress_inventory(progress_path, epochs)
    evidence = {
        "schema_version": PRODUCTION_SCHEMA_VERSION,
        "kind": EVIDENCE_KIND,
        "production_id": PRODUCTION_ID,
        "mode": capability.mode,
        "run_id": run_id,
        "claim": _copy(claim),
        "call_order": list(capability.call_order),
        "source_identity": _copy(capability.source_identity),
        "optimizer_updates": optimizer_updates,
        "epochs": epochs,
        "checkpoint_inventory": _copy(checkpoint_inventory),
        "progress_inventory": _copy(progress),
        "amp": {
            "autocast_dtype": policy["amp"]["autocast_dtype"],
            "grad_scaler_enabled": policy["amp"]["grad_scaler_enabled"],
            "scaler_policy": policy["amp"]["scaler_policy"],
        },
    }
    evidence_ref = _write_exclusive_json(target / names["evidence"], evidence, "T7D execution evidence")
    terminal = {
        "schema_version": PRODUCTION_SCHEMA_VERSION,
        "kind": TERMINAL_KIND,
        "status": SUCCESS_STATUS,
        "mode": capability.mode,
        "production_id": PRODUCTION_ID,
        "run_id": run_id,
        "descriptor_sha256": claim["descriptor_sha256"],
        "authorization_context_sha256": claim["authorization_context_sha256"],
        "contract_sha256": policy["contract_sha256"],
        "claim": _copy(claim),
        "source_identity": _copy(capability.source_identity),
        "data_roots": _copy(policy["data_roots"]),
        "environment_identity": _copy(policy["environment_identity"]),
        "progress_inventory": progress,
        "checkpoint_inventory": checkpoint_inventory,
        "evidence_file": evidence_ref,
        "process_identity": {"mode": capability.mode, "pid": os.getpid(), "parent_pid": os.getppid()},
        "optimizer_updates": optimizer_updates,
        "epochs": epochs,
        "selection": _copy(selected),
        "call_order": list(capability.call_order),
        "closure": {
            "detached_descriptor": capability.mode == PRODUCTION_MODE,
            "authorization_unconsumed_before_claim": True,
            "independent_target_claim": True,
            "strict_load_before_optimizer": True,
            "optimizer_before_data_and_evaluator": True,
            "primary_ema_evaluator": True,
            "epoch_progress": True,
            "source_identity": True,
            "bf16_without_scaler": True,
        },
        "production": {
            "owner_authorization_consumed": False,
            "gpu_probe_executed": False,
            "formal_training_executed": capability.mode == PRODUCTION_MODE,
            "training_ready": False,
            "independent_audit_pass": False,
        },
    }
    terminal_ref = _write_exclusive_json(target / names["terminal"], terminal, "T7D terminal result")
    stdout = b"vendor-log\n" + _canonical(terminal) + b"\n"
    parsed = parse_v2a_production_stdout(stdout)
    result = {**terminal, "terminal_stdout": parsed, "terminal_file": terminal_ref}
    return validate_v2a_production_result(result, policy)


def _build_production_model(repo_root: pathlib.Path, policy: Mapping[str, Any], loaded: Mapping[str, Any]) -> dict[str, Any]:
    """Construct the vendor model only after T7C's strict authority load."""

    torch = importlib.import_module("torch")
    config = importlib.import_module("sparse_rtdetr.baseline.config")
    vendor_root = config._vendor_root(repo_root)
    vendor_config = config._vendor_config_path(repo_root)
    with config._vendor_path(vendor_root):
        yaml_module = importlib.import_module("src.core.yaml_config")
        yaml_class = getattr(yaml_module, "YAMLConfig", None)
        if not callable(yaml_class):
            _fail("vendor model factory is unavailable")
        runtime_config = yaml_class(
            str(vendor_config),
            PResNet={"pretrained": False},
            num_classes=policy["model"]["num_classes"],
            remap_mscoco_category=False,
            device="cuda:0",
            output_dir=policy["target"]["path"],
        )
    model = runtime_config.model
    backbone = getattr(model, "backbone", None)
    source_model = loaded.get("model")
    state_dict = getattr(source_model, "state_dict", None)
    if backbone is not None and callable(state_dict):
        loaded_state = backbone.load_state_dict(state_dict(), strict=True)
        if list(getattr(loaded_state, "missing_keys", ())) or list(getattr(loaded_state, "unexpected_keys", ())):
            _fail("full model backbone strict load failed")
    mover = getattr(model, "to", None)
    if not callable(mover):
        _fail("vendor model has no device transfer")
    model = mover(device="cuda:0")
    return {"torch": torch, "config": runtime_config, "model": model, "vendor_root": vendor_root}


def _resolve_production_capability(policy: Mapping[str, Any], loaded: Mapping[str, Any], model_info: Mapping[str, Any], call_order: list[str]) -> _ProductionCapability:
    call_order.append("create_data_ports")
    config = model_info["config"]
    try:
        train_loader = config.train_dataloader
        development_loader = config.val_dataloader
        criterion = config.criterion
    except Exception as exc:
        raise V2AProductionError("authorized train/development ports could not be created") from exc
    ema = config.ema
    if ema is None:
        _fail("primary EMA port is unavailable")

    def batches() -> Any:
        return train_loader

    def compute_loss(model: Any, batch: Any, epoch: int) -> Any:
        del epoch
        if type(batch) not in {tuple, list} or len(batch) != 2:
            _fail("production train batch shape drift")
        samples, targets = batch
        outputs = model(samples, targets=targets)
        return criterion(outputs, targets)

    def primary_evaluate(ema_model: Any, epoch: int) -> dict[str, Any]:
        del ema_model, epoch, development_loader
        _fail("primary development evaluator adapter was not supplied by the certified launcher")

    primary_evaluate.evaluator_id = policy["selection"]["primary_evaluator"]
    call_order.append("create_primary_evaluator")
    scheduler = None
    return _ProductionCapability(
        _CAPABILITY_TOKEN,
        policy=_copy(policy),
        mode=PRODUCTION_MODE,
        model=model_info["model"],
        optimizer=None,
        ema=ema,
        evaluator=primary_evaluate,
        batches=batches,
        compute_loss=compute_loss,
        backward=None,
        scheduler=scheduler,
        rng_state={"mode": PRODUCTION_MODE, "seed": policy["runtime_policy"]["contract_binding"]["contract"]["initialization"]["seed"]},
        source_identity=_copy(policy["source_identity"]),
        torch_module=model_info["torch"],
        call_order=call_order,
        raw_model_state=None,
        ema_state=None,
    )


def _verify_source_drift(policy: Mapping[str, Any], repo_root: pathlib.Path) -> None:
    if _validate_source_identity(policy["source_identity"]) != current_v2a_source_identity(repo_root):
        _fail("T7D source identity drift")


def run_v2a_production_entry(
    descriptor: Mapping[str, Any],
    authorization_context: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the sole production entry with internally resolved capabilities."""

    checked_descriptor = validate_v2a_detached_launch_descriptor(descriptor)
    checked_context = validate_v2a_authorization_context(authorization_context, descriptor=checked_descriptor)
    _verify_authorization_artifact(checked_context)
    root = _canonical_directory(checked_descriptor["repo_root"], "descriptor.repo_root")
    if str(root) != checked_descriptor["repo_root"]:
        _fail("descriptor repository root is not canonical")
    call_order = ["validate_descriptor", "validate_authorization"]
    target = _target_path(root, checked_descriptor["target_root"])
    if str(target) != checked_descriptor["target_root"]:
        _fail("descriptor target root is not canonical")
    call_order.append("build_v2a_runtime_policy")
    policy = build_v2a_production_policy(
        root,
        data_roots={role: item["root"] for role, item in checked_context["data_roots"].items()},
        environment_identity=checked_context["environment_identity"],
        target_root=target,
    )
    if policy["contract_sha256"] != checked_descriptor["contract_sha256"]:
        _fail("descriptor and runtime contract digest drift")
    if policy["data_roots"] is None or policy["environment_identity"] is None:
        _fail("production policy closure is incomplete")
    _verify_source_drift(policy, root)
    call_order.append("claim_target")
    claim = _claim_target(
        policy,
        descriptor_sha256=checked_descriptor["aggregate_sha256"],
        authorization_context_sha256=checked_context["context_sha256"],
        mode=PRODUCTION_MODE,
    )
    call_order.append("load_v2a_pretrained_backbone")
    loaded = _runtime.load_v2a_pretrained_backbone(root, binding=policy["runtime_policy"]["contract_binding"])
    model_info = _build_production_model(root, policy, loaded)
    call_order.append("create_optimizer")
    optimizer = _runtime.build_v2a_optimizer(
        model_info["model"],
        policy["runtime_policy"]["contract_binding"],
    )
    capability = _resolve_production_capability(policy, loaded, model_info, call_order)
    capability.optimizer = optimizer["optimizer"]
    epochs = policy["runtime_policy"]["contract_binding"]["contract"]["schedule"]["epochs"]
    return _run_epoch_loop(capability, epochs=epochs, run_id=checked_descriptor["run_id"], claim=claim)


def parse_v2a_production_stdout(stdout: bytes) -> dict[str, Any]:
    """Preserve raw bytes and parse the final canonical terminal JSON line."""

    if type(stdout) is not bytes:
        _fail("production stdout must be raw bytes")
    candidates = []
    for line_number, line in enumerate(stdout.splitlines(keepends=True), start=1):
        if not line.endswith(b"\n"):
            continue
        payload = line[:-1]
        try:
            value = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if type(value) is dict and value.get("kind") == TERMINAL_KIND and _canonical(value) == payload:
            candidates.append((line_number, value, payload))
    if not candidates:
        _fail("production stdout has no canonical terminal result")
    line_number, value, payload = candidates[-1]
    return {
        "value": _copy(value),
        "line_number": line_number,
        "terminal_count": len(candidates),
        "line_size_bytes": len(payload) + 1,
        "stdout_size_bytes": len(stdout),
        "stdout_sha256": _sha_bytes(stdout),
        "terminal_line_sha256": _sha_bytes(payload),
        "stdout_bytes": stdout,
    }


def classify_v2a_terminal(
    return_code: int,
    stdout: bytes,
    *,
    signal_number: int | None = None,
    exception: BaseException | None = None,
) -> dict[str, Any]:
    """Classify success, ordinary failure, exception and signal outcomes."""

    _integer(return_code, "return_code")
    if signal_number is not None:
        _integer(signal_number, "signal_number", minimum=1)
    parsed = None
    parse_error = None
    try:
        parsed = parse_v2a_production_stdout(stdout)
    except V2AProductionError as exc:
        parse_error = str(exc)
    if exception is not None:
        classification = "EXCEPTION"
        status = FAILURE_STATUS
    elif signal_number is not None or return_code < 0:
        classification = "SIGNAL"
        status = FAILURE_STATUS
    elif return_code == 0 and parsed is not None and parsed["value"].get("status") == SUCCESS_STATUS:
        classification = "SUCCESS"
        status = SUCCESS_STATUS
    else:
        classification = "FAILURE"
        status = FAILURE_STATUS
    return {
        "classification": classification,
        "status": status,
        "return_code": return_code,
        "signal_number": signal_number,
        "exception_type": None if exception is None else type(exception).__name__,
        "terminal_present": parsed is not None,
        "terminal": None if parsed is None else parsed["value"],
        "stdout_size_bytes": len(stdout),
        "stdout_sha256": _sha_bytes(stdout),
        "parse_error": parse_error,
    }


def validate_v2a_production_result(result: Mapping[str, Any], policy: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute the immutable success receipt and all independent artifacts."""

    checked_policy = _validate_policy(policy)
    value = _exact(result, RESULT_KEYS, "T7D production result")
    _assert_builtin({key: child for key, child in value.items() if key != "terminal_stdout"}, "T7D production result")
    if value["schema_version"] != PRODUCTION_SCHEMA_VERSION or value["kind"] != TERMINAL_KIND or value["status"] != SUCCESS_STATUS:
        _fail("T7D terminal status drift")
    if value["production_id"] != PRODUCTION_ID or value["contract_sha256"] != checked_policy["contract_sha256"]:
        _fail("T7D terminal identity drift")
    _sha(value["descriptor_sha256"], "T7D result.descriptor_sha256")
    _sha(value["authorization_context_sha256"], "T7D result.authorization_context_sha256")
    _validate_source_identity(value["source_identity"])
    if value["source_identity"] != checked_policy["source_identity"]:
        _fail("T7D result source identity drift")
    _bound_data_roots_shape(value["data_roots"], "T7D result.data_roots")
    if value["data_roots"] != checked_policy["data_roots"]:
        _fail("T7D result data identity drift")
    _bound_environment_shape(value["environment_identity"], "T7D result.environment_identity")
    if value["environment_identity"] != checked_policy["environment_identity"]:
        _fail("T7D result environment identity drift")
    _integer(value["optimizer_updates"], "T7D result.optimizer_updates", minimum=1)
    _integer(value["epochs"], "T7D result.epochs", minimum=1)
    if value["call_order"] != [
        "validate_descriptor", "validate_authorization", "build_v2a_runtime_policy", "claim_target",
        "load_v2a_pretrained_backbone", "create_optimizer", "create_data_ports", "create_primary_evaluator", "epoch_loop",
    ] and value["mode"] == PRODUCTION_MODE:
        _fail("T7D production call order drift")
    if value["mode"] not in {PRODUCTION_MODE, CPU_FAKE_MODE}:
        _fail("T7D result mode drift")
    required_closure = {"detached_descriptor", "authorization_unconsumed_before_claim", "independent_target_claim", "strict_load_before_optimizer", "optimizer_before_data_and_evaluator", "primary_ema_evaluator", "epoch_progress", "source_identity", "bf16_without_scaler"}
    closure_shape_drift = set(value["closure"]) != required_closure
    production_closure_drift = value["mode"] == PRODUCTION_MODE and not all(value["closure"].values())
    fake_closure_drift = value["mode"] == CPU_FAKE_MODE and (
        value["closure"].get("detached_descriptor") is not False
        or any(value["closure"].get(key) is not True for key in required_closure - {"detached_descriptor"})
    )
    if closure_shape_drift or production_closure_drift or fake_closure_drift:
        _fail("T7D terminal closure is incomplete")
    if value["production"] != {
        "owner_authorization_consumed": False,
        "gpu_probe_executed": False,
        "formal_training_executed": value["mode"] == PRODUCTION_MODE,
        "training_ready": False,
        "independent_audit_pass": False,
    }:
        _fail("T7D terminal production state drift")
    target = pathlib.Path(checked_policy["target"]["path"])
    claim = value["claim"]
    claim_keys = {"schema_version", "kind", "production_id", "mode", "descriptor_sha256", "authorization_context_sha256", "contract_sha256", "target_path", "status", "file"}
    _exact(claim, claim_keys, "T7D target claim")
    if claim["schema_version"] != PRODUCTION_SCHEMA_VERSION or claim["kind"] != TARGET_CLAIM_KIND or claim["production_id"] != PRODUCTION_ID or claim["status"] != "CLAIMED" or claim["target_path"] != str(target):
        _fail("T7D target claim identity drift")
    if claim["descriptor_sha256"] != value["descriptor_sha256"] or claim["authorization_context_sha256"] != value["authorization_context_sha256"] or claim["contract_sha256"] != value["contract_sha256"]:
        _fail("T7D target claim cross-binding drift")
    lock_path = target / checked_policy["target"]["names"]["lock"]
    if claim["file"].get("path") != str(lock_path):
        _fail("T7D target lock path drift")
    try:
        lock_raw = lock_path.read_bytes()
        lock_value = json.loads(lock_raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V2AProductionError("T7D target lock is unavailable") from exc
    if type(lock_value) is not dict or _canonical(lock_value) != lock_raw or lock_value != {key: child for key, child in claim.items() if key != "file"}:
        _fail("T7D target lock readback drift")
    checkpoint_rows = value["checkpoint_inventory"]
    if type(checkpoint_rows) is not list or not checkpoint_rows:
        _fail("T7D checkpoint inventory is empty")
    for row in checkpoint_rows:
        _exact(row, {"role", "epoch", "path", "size_bytes", "sha256", "mode"}, "T7D checkpoint inventory row")
        path = pathlib.Path(row["path"])
        if path.parent != target or path.is_symlink() or not path.is_file():
            _fail("T7D checkpoint path drift")
        raw = path.read_bytes()
        if row["size_bytes"] != len(raw) or row["sha256"] != _sha_bytes(raw) or row["mode"] != 0o600:
            _fail("T7D checkpoint inventory identity drift")
        try:
            checkpoint_value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise V2AProductionError("T7D checkpoint JSON is invalid") from exc
        _validate_checkpoint(checkpoint_value, checked_policy, value["run_id"])
    progress = value["progress_inventory"]
    _exact(progress, {"path", "relative_name", "size_bytes", "sha256", "mode", "line_count", "rows_sha256", "first_epoch", "last_epoch"}, "T7D progress inventory")
    progress_path = pathlib.Path(progress["path"])
    if progress_path != target / checked_policy["target"]["names"]["progress"]:
        _fail("T7D progress path drift")
    if _progress_inventory(progress_path, value["epochs"]) != progress:
        _fail("T7D progress inventory drift")
    evidence = value["evidence_file"]
    _exact(evidence, {"path", "size_bytes", "sha256", "mode"}, "T7D evidence file")
    evidence_path = target / checked_policy["target"]["names"]["evidence"]
    if evidence["path"] != str(evidence_path):
        _fail("T7D evidence path drift")
    evidence_raw = evidence_path.read_bytes()
    if evidence["size_bytes"] != len(evidence_raw) or evidence["sha256"] != _sha_bytes(evidence_raw) or evidence["mode"] != 0o600:
        _fail("T7D evidence identity drift")
    try:
        evidence_value = json.loads(evidence_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V2AProductionError("T7D evidence JSON is invalid") from exc
    if type(evidence_value) is not dict or _canonical(evidence_value) != evidence_raw or evidence_value.get("kind") != EVIDENCE_KIND or evidence_value.get("claim") != claim:
        _fail("T7D evidence content drift")
    terminal_file = value["terminal_file"]
    _exact(terminal_file, {"path", "size_bytes", "sha256", "mode"}, "T7D terminal file")
    terminal_path = target / checked_policy["target"]["names"]["terminal"]
    if terminal_file["path"] != str(terminal_path):
        _fail("T7D terminal file path drift")
    terminal_raw = terminal_path.read_bytes()
    terminal_base = {key: value[key] for key in RESULT_KEYS if key not in {"terminal_stdout", "terminal_file"}}
    if terminal_file["size_bytes"] != len(terminal_raw) or terminal_file["sha256"] != _sha_bytes(terminal_raw) or terminal_file["mode"] != 0o600 or terminal_raw != _canonical(terminal_base):
        _fail("T7D terminal file content drift")
    stdout = value["terminal_stdout"]
    if type(stdout) is not dict or stdout.get("terminal_count") != 1:
        _fail("T7D stdout does not contain one terminal JSON")
    parsed = parse_v2a_production_stdout(stdout["stdout_bytes"])
    if parsed != stdout or parsed["value"] != {key: value[key] for key in RESULT_KEYS if key not in {"terminal_stdout", "terminal_file"}}:
        _fail("T7D terminal stdout content drift")
    return _copy(value)


def run_v2a_cpu_fake(policy: Mapping[str, Any], ports: Mapping[str, Any], *, epochs: int = 1) -> dict[str, Any]:
    """Run the same ordering and evidence closure with explicitly fake ports."""

    checked = _validate_policy(policy)
    if type(ports) is not dict:
        _fail("CPU fake ports must be a builtin dict")
    # Reject attempts to smuggle a production capability before touching the
    # durable target.  Duplicate-target semantics still remain the first
    # irreversible operation for otherwise valid CPU-fake requests.
    if any(key in ports for key in {"production_capability", "authorization_context", "real_data", "real_model"}):
        _fail("CPU fake cannot inject production capabilities")
    call_order: list[str] = []
    claim = _claim_target(
        checked,
        descriptor_sha256=checked["policy_sha256"],
        authorization_context_sha256=_digest({"mode": CPU_FAKE_MODE, "policy_sha256": checked["policy_sha256"]}),
        mode=CPU_FAKE_MODE,
    )
    capability = _make_fake_capability(checked, ports, call_order)
    result = _run_epoch_loop(capability, epochs=epochs, run_id=ports.get("run_id", PRODUCTION_ID), claim=claim)
    return result


def _main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 4 or args[0] != "--descriptor" or args[2] != "--authorization-context":
        sys.stderr.write("usage: python -m sparse_rtdetr.baseline.training_v2a_production --descriptor PATH --authorization-context PATH\n")
        return 2
    descriptor_path = pathlib.Path(args[1])
    context_path = pathlib.Path(args[3])
    try:
        descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
        context = json.loads(context_path.read_text(encoding="utf-8"))
        result = run_v2a_production_entry(descriptor, context)
        sys.stdout.buffer.write(_canonical({key: child for key, child in result.items() if key != "terminal_stdout"}) + b"\n")
        sys.stdout.buffer.flush()
    except BaseException as exc:
        sys.stderr.write(f"T7D production entry failed: {type(exc).__name__}: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
