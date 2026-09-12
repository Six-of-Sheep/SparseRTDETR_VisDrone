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
            {"role", "root", "identity", "annotation", "annotation_path", "annotation_identity"},
            f"{field}.roles.{role}",
        )
        if item["role"] != role:
            _fail(f"{field}.roles.{role}.role drift")
        _absolute_path(item["root"], f"{field}.roles.{role}.root")
        _string(item["annotation"], f"{field}.roles.{role}.annotation")
        if "/" in item["annotation"]:
            _fail(f"{field}.roles.{role}.annotation is not a basename")
        annotation_path = _absolute_path(item["annotation_path"], f"{field}.roles.{role}.annotation_path")
        if annotation_path.name != item["annotation"]:
            _fail(f"{field}.roles.{role}.annotation path binding drift")
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
    try:
        return math.isfinite(_loss_scalar(value))
    except V2AProductionError:
        return False


def _loss_scalar(value: Any) -> float:
    """Return the detached scalar used for durable epoch-loss evidence."""

    scalar = value
    if type(value) not in {int, float} or type(value) is bool:
        detach = getattr(value, "detach", None)
        if callable(detach):
            scalar = detach()
        item = getattr(scalar, "item", None)
        if not callable(item):
            _fail("loss does not expose a scalar observation")
        scalar = item()
    if type(scalar) not in {int, float} or type(scalar) is bool:
        _fail("loss scalar observation has invalid type")
    return float(scalar)


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


def _move_to_device(value: Any, device: Any) -> Any:
    if type(value) is dict:
        return {key: _move_to_device(child, device) for key, child in value.items()}
    if type(value) is list:
        return [_move_to_device(child, device) for child in value]
    if type(value) is tuple:
        return tuple(_move_to_device(child, device) for child in value)
    mover = getattr(value, "to", None)
    return mover(device) if callable(mover) else value


def _weighted_loss(criterion: Any, losses: Any) -> Any:
    """Apply the vendor criterion's frozen weight dictionary exactly once."""

    if type(losses) is not dict:
        return losses
    weights = getattr(criterion, "weight_dict", None)
    if type(weights) is not dict or not weights:
        _fail("production criterion weight dictionary is unavailable")
    selected = []
    for name, value in losses.items():
        if name in weights:
            weight = weights[name]
            if type(weight) not in {int, float} or type(weight) is bool or not math.isfinite(float(weight)):
                _fail("production criterion weight is not finite")
            selected.append(value * weight)
    if not selected:
        _fail("production criterion returned no weighted losses")
    total = selected[0]
    for value in selected[1:]:
        total = total + value
    return total


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
    __slots__ = (
        "policy", "mode", "model", "optimizer", "ema", "evaluator", "batches",
        "compute_loss", "backward", "scheduler", "warmup", "checkpoint_writer",
        "rng_state", "source_identity", "torch_module", "call_order",
        "raw_model_state", "ema_state",
    )

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
        warmup=None,
        checkpoint_writer=None,
        rng_state=rng_state,
        source_identity=_copy(policy["source_identity"]),
        torch_module=None,
        call_order=call_order,
        raw_model_state=raw_model_state,
        ema_state=ema_state,
    )


def _production_autocast(torch_module: Any, binding: Mapping[str, Any]) -> Any:
    return _runtime.v2a_bf16_autocast_context(binding, torch_module=torch_module)


def _required_state_dict(value: Any, field: str) -> Any:
    state_dict = getattr(value, "state_dict", None)
    if not callable(state_dict):
        _fail(f"{field} does not expose state_dict")
    try:
        return state_dict()
    except Exception as exc:
        raise V2AProductionError(f"{field} state_dict failed") from exc


def _production_runtime_states(capability: _ProductionCapability) -> dict[str, Any]:
    torch_module = capability.torch_module
    random_module = importlib.import_module("random")
    cuda = getattr(torch_module, "cuda", None)
    cuda_states = []
    get_cuda_states = getattr(cuda, "get_rng_state_all", None)
    if callable(get_cuda_states):
        cuda_states = get_cuda_states()
    scheduler_state = (
        _required_state_dict(capability.scheduler, "production scheduler")
        if capability.scheduler is not None
        else {"state": "absent"}
    )
    warmup_state = (
        _required_state_dict(capability.warmup, "production warmup")
        if capability.warmup is not None
        else {"state": "absent"}
    )
    return {
        "raw_model": _required_state_dict(capability.model, "production model"),
        "ema": _required_state_dict(capability.ema, "production EMA"),
        "optimizer": _required_state_dict(capability.optimizer, "production optimizer"),
        "scheduler": scheduler_state,
        "warmup": warmup_state,
        "rng_states": {
            "python": random_module.getstate(),
            "torch_cpu": torch_module.get_rng_state(),
            "torch_cuda": cuda_states,
        },
    }


def _save_production_checkpoint(
    capability: _ProductionCapability,
    role: str,
    epoch: int,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    writer = capability.checkpoint_writer
    save_atomic = getattr(writer, "save_atomic", None)
    is_loadable = getattr(writer, "is_loadable", None)
    if not callable(save_atomic) or not callable(is_loadable):
        _fail("production checkpoint writer is unavailable")
    state_value = {**_copy(metadata), "runtime_states": _production_runtime_states(capability)}
    reference = save_atomic(role, epoch, state_value)
    if not is_loadable(reference):
        _fail("production checkpoint is not loadable after publication")
    return {
        "role": role,
        "epoch": epoch,
        "path": reference["path"],
        "size_bytes": reference["file_size_bytes"],
        "sha256": reference["file_sha256"],
        "mode": 0o600,
        "format": "torch_save_v1",
        "state_sha256": reference["state_sha256"],
        "runtime_state_keys": reference["runtime_state_keys"],
        "inventory_path": reference["inventory_path"],
        "inventory_sha256": reference["inventory_sha256"],
    }


def _write_checkpoint(
    capability: _ProductionCapability,
    policy: Mapping[str, Any],
    role: str,
    epoch: int,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    if capability.mode == PRODUCTION_MODE:
        return _save_production_checkpoint(capability, role, epoch, metadata)
    path = pathlib.Path(policy["target"]["path"]) / f"{policy['target']['names']['checkpoint_prefix']}{role}_epoch_{epoch}.json"
    reference = _write_exclusive_json(path, metadata, f"T7D {role} checkpoint")
    return {"role": role, "epoch": epoch, **reference}


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
        train_mode = getattr(capability.model, "train", None)
        if capability.mode == PRODUCTION_MODE and not callable(train_mode):
            _fail("production model does not expose train mode")
        if callable(train_mode):
            train_mode()
        source = capability.batches() if callable(capability.batches) else capability.batches
        set_epoch = getattr(source, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(epoch - 1)
        else:
            dataset_set_epoch = getattr(getattr(source, "dataset", None), "set_epoch", None)
            if callable(dataset_set_epoch):
                dataset_set_epoch(epoch - 1)
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
                loss_scalar = _loss_scalar(loss_value)
                if not math.isfinite(loss_scalar):
                    _fail("loss is non-finite")
                backward = capability.backward
                if callable(backward):
                    _call_port(backward, ((loss_value,), ()), "backward")
                else:
                    backward_method = getattr(loss_value, "backward", None)
                    if not callable(backward_method):
                        _fail("loss has no direct backward operation")
                    backward_method()
            if capability.mode == PRODUCTION_MODE:
                parameters = getattr(capability.model, "parameters", None)
                if not callable(parameters):
                    _fail("production model does not expose parameters")
                capability.torch_module.nn.utils.clip_grad_norm_(
                    parameters(),
                    policy["optimizer"]["gradient_clip_max_norm"],
                )
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
            warmup_step = getattr(capability.warmup, "step", None)
            if callable(warmup_step):
                warmup_step()
            batch_count += 1
            mean_loss += loss_scalar
        if batch_count == 0:
            _fail("epoch has no training batches")
        warmup_finished = getattr(capability.warmup, "finished", None)
        if capability.scheduler is not None and (
            not callable(warmup_finished) or bool(warmup_finished())
        ):
            scheduler_step = getattr(capability.scheduler, "step", None)
            if not callable(scheduler_step):
                _fail("production scheduler has no step operation")
            scheduler_step()
        metrics = _normalize_metrics(_call_evaluator(capability.evaluator, capability.ema, epoch), policy)
        candidates.append({"epoch": epoch, **metrics})
        selected = _runtime.select_v2a_development_candidate(candidates, policy["runtime_policy"]["contract_binding"])
        production_binary = capability.mode == PRODUCTION_MODE
        last_checkpoint = _checkpoint(
            policy,
            run_id=run_id,
            epoch=epoch,
            optimizer_updates=optimizer_updates,
            raw_model={"binary_state": "raw_model"} if production_binary else _model_state(capability.raw_model_state if capability.raw_model_state is not None else capability.model, "raw_model", epoch),
            ema={"binary_state": "ema"} if production_binary else _model_state(capability.ema_state if capability.ema_state is not None else capability.ema, "ema", epoch),
            optimizer={"binary_state": "optimizer"} if production_binary else capability.optimizer,
            scheduler={"binary_state": "scheduler_and_warmup"} if production_binary else (capability.scheduler if capability.scheduler is not None else {"state": "absent"}),
            rng_state={"binary_state": "rng_states"} if production_binary else capability.rng_state,
            source_identity=capability.source_identity,
        )
        last_checkpoint = _validate_checkpoint(last_checkpoint, policy, run_id)
        checkpoint_inventory.append(
            _write_checkpoint(capability, policy, "last", epoch, last_checkpoint)
        )
        if selected["epoch"] == epoch:
            checkpoint_inventory.append(
                _write_checkpoint(capability, policy, "best", epoch, last_checkpoint)
            )
        periodic_frequency = policy["checkpoint"]["periodic_checkpoint_frequency_epochs"]
        if epoch % periodic_frequency == 0:
            checkpoint_inventory.append(
                _write_checkpoint(capability, policy, "periodic", epoch, last_checkpoint)
            )
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
    checkpoint_inventory.append(
        _write_checkpoint(capability, policy, "final", final_epoch, last_checkpoint)
    )
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
    return {"torch": torch, "config": runtime_config, "model": model, "vendor_root": vendor_root, "repo_root": repo_root}


class _ProductionCheckpointWriter:
    def __init__(self, root: pathlib.Path, torch_module: Any) -> None:
        self._torch = torch_module
        self._root = root / "checkpoints"
        try:
            self._root.mkdir(mode=0o700)
        except OSError as exc:
            raise V2AProductionError("production checkpoint root creation failed") from exc
        self._records: list[dict[str, Any]] = []
        _fsync_directory(root, "production checkpoint parent")

    def save_atomic(self, role: str, epoch: int, state_value: dict[str, Any]) -> dict[str, Any]:
        tempfile_module = importlib.import_module("tempfile")
        target = self._root / f"checkpoint-{role}-{epoch:04d}.pth"
        if target.exists() or target.is_symlink():
            _fail("production checkpoint target already exists")
        fd, temporary = tempfile_module.mkstemp(prefix=f".{target.name}.", dir=str(self._root))
        try:
            with os.fdopen(fd, "wb") as stream:
                self._torch.save(state_value, stream)
                stream.flush()
                os.fsync(stream.fileno())
            if target.exists() or target.is_symlink():
                _fail("production checkpoint target appeared during publication")
            os.link(temporary, target)
            os.unlink(temporary)
            _fsync_directory(self._root, "production checkpoint root")
        except Exception:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
        raw = target.read_bytes()
        observed = target.lstat()
        if target.is_symlink() or not target.is_file() or observed.st_nlink != 1 or (observed.st_mode & 0o777) != 0o600:
            _fail("production checkpoint metadata drift")
        runtime_states = state_value.get("runtime_states")
        if type(runtime_states) is not dict or not runtime_states:
            _fail("production checkpoint lacks runtime states")
        json_state = {key: value for key, value in state_value.items() if key != "runtime_states"}
        reference = {
            "role": role,
            "epoch": epoch,
            "path": str(target),
            "state_sha256": _digest(json_state),
            "runtime_state_keys": sorted(runtime_states),
            "file_size_bytes": len(raw),
            "file_sha256": _sha_bytes(raw),
        }
        self._records.append(_copy(reference))
        inventory = {
            "schema_version": PRODUCTION_SCHEMA_VERSION,
            "records": _copy(self._records),
        }
        inventory["inventory_sha256"] = _digest(inventory["records"])
        inventory_path = self._root / f"checkpoint-inventory-{len(self._records):04d}.json"
        inventory_ref = _write_exclusive_json(inventory_path, inventory, "production checkpoint inventory")
        reference["inventory_path"] = inventory_ref["path"]
        reference["inventory_sha256"] = inventory_ref["sha256"]
        return reference

    def is_loadable(self, reference: Mapping[str, Any]) -> bool:
        try:
            path = pathlib.Path(reference["path"])
            raw = path.read_bytes()
            if path.parent != self._root or path.is_symlink() or not path.is_file():
                return False
            if reference["file_size_bytes"] != len(raw) or reference["file_sha256"] != _sha_bytes(raw):
                return False
            loaded = self._torch.load(path, map_location="cpu", weights_only=False)
            if type(loaded) is not dict or type(loaded.get("runtime_states")) is not dict:
                return False
            json_state = {key: value for key, value in loaded.items() if key != "runtime_states"}
            return reference["state_sha256"] == _digest(json_state)
        except Exception:
            return False


def _build_runtime_ports(model_info: Mapping[str, Any], policy: Mapping[str, Any]) -> dict[str, Any]:
    repo_root = model_info["repo_root"]
    vendor_root = model_info["vendor_root"]
    runtime_config = model_info["config"]
    torch = model_info["torch"]
    multiprocessing = getattr(torch, "multiprocessing", None)
    set_sharing_strategy = getattr(multiprocessing, "set_sharing_strategy", None)
    if not callable(set_sharing_strategy):
        _fail("production torch sharing-strategy API is unavailable")
    set_sharing_strategy("file_system")
    baseline_config = importlib.import_module("sparse_rtdetr.baseline.config")
    baseline_dataset = importlib.import_module("sparse_rtdetr.baseline.dataset")
    baseline_postprocessor = importlib.import_module("sparse_rtdetr.baseline.postprocessor")
    primary_evaluator = importlib.import_module("sparse_rtdetr.baseline.primary_evaluator")
    with baseline_config._vendor_path(vendor_root):
        workspace = importlib.import_module("src.core.workspace")
        dataloader_module = importlib.import_module("src.data.dataloader")
        warmup_module = importlib.import_module("src.optim.warmup")
        importlib.import_module("src.data")
        importlib.import_module("src.nn")
        importlib.import_module("src.zoo.rtdetr")
    dataset_type = getattr(baseline_dataset, "VisDroneCocoDetection", None)
    postprocessor_type = getattr(baseline_postprocessor, "VisDronePostProcessor", None)
    loader_type = getattr(dataloader_module, "DataLoader", None)
    warmup_type = getattr(warmup_module, "LinearWarmup", None)
    if not all(callable(value) for value in (dataset_type, postprocessor_type, loader_type, warmup_type)):
        _fail("production data/evaluator runtime API is incomplete")
    state: dict[str, Any] = {}

    def configured_component(value: Any, field: str) -> Any:
        if type(value) is not dict:
            _fail(f"production {field} configuration is invalid")
        payload = _copy(value)
        component_type = payload.pop("type", None)
        if type(component_type) is not str or not component_type:
            _fail(f"production {field} type is missing")
        descriptor = runtime_config.global_cfg.get(component_type)
        if type(descriptor) is not dict or "_kwargs" not in descriptor:
            _fail(f"production {field} type is not registered")
        for key in [key for key in descriptor if not key.startswith("_")]:
            del descriptor[key]
        descriptor.update(_copy(descriptor["_kwargs"]))
        descriptor.update(payload)
        return workspace.create(component_type, runtime_config.global_cfg)

    def dataset(role: str) -> Any:
        cached = state.setdefault("datasets", {}).get(role)
        if cached is not None:
            return cached
        loader_name = "train_dataloader" if role == "train_core" else "val_dataloader"
        loader_config = runtime_config.yaml_cfg.get(loader_name)
        if type(loader_config) is not dict or type(loader_config.get("dataset")) is not dict:
            _fail(f"production {role} loader configuration is invalid")
        transform = configured_component(loader_config["dataset"].get("transforms"), f"{role} transforms")
        role_binding = policy["data_roots"]["roles"][role]
        result = dataset_type(
            img_folder=role_binding["root"],
            ann_file=role_binding["annotation_path"],
            transforms=transform,
            return_masks=False,
            remap_mscoco_category=False,
            vendor_root=vendor_root,
            role=role,
        )
        state.setdefault("datasets", {})[role] = result
        return result

    def loader(role: str) -> Any:
        cached = state.setdefault("loaders", {}).get(role)
        if cached is not None:
            return cached
        loader_name = "train_dataloader" if role == "train_core" else "val_dataloader"
        loader_config = runtime_config.yaml_cfg.get(loader_name)
        if type(loader_config) is not dict:
            _fail(f"production {role} loader configuration is invalid")
        collate = configured_component(loader_config.get("collate_fn"), f"{role} collate")
        topology = policy["topology"]
        result = loader_type(
            dataset=dataset(role),
            batch_size=topology["train_micro_batch"] if role == "train_core" else topology["development_batch"],
            num_workers=topology["train_workers"] if role == "train_core" else topology["development_workers"],
            drop_last=topology["drop_last_train"] if role == "train_core" else topology["drop_last_development"],
            collate_fn=collate,
            shuffle=role == "train_core",
        )
        result.shuffle = role == "train_core"
        state.setdefault("loaders", {})[role] = result
        return result

    def postprocessor_factory() -> Any:
        value = state.get("postprocessor")
        if value is None:
            value = postprocessor_type(vendor_root=vendor_root)
            state["postprocessor"] = value
        return value

    return {
        "torch": model_info["torch"],
        "repo_root": str(repo_root),
        "primary_evaluator": primary_evaluator,
        "train_loader": loader("train_core"),
        "development_loader": loader("development"),
        "postprocessor_factory": postprocessor_factory,
        "checkpoint_writer": _ProductionCheckpointWriter(pathlib.Path(policy["target"]["path"]), model_info["torch"]),
        "warmup_type": warmup_type,
    }


def _evaluate_primary_metrics(
    runtime_ports: Mapping[str, Any],
    weights: Any,
    batches: Any,
) -> dict[str, float]:
    """Convert development batches to the certified primary evaluator schema."""

    torch_module = runtime_ports["torch"]
    primary = runtime_ports["primary_evaluator"]
    postprocessor = runtime_ports["postprocessor_factory"]()
    evaluator = getattr(primary, "evaluate_primary_v1", None)
    validate_result = getattr(primary, "validate_primary_evaluator_result", None)
    if not callable(evaluator) or not callable(validate_result):
        _fail("certified primary evaluator API is unavailable")
    contract = primary.primary_evaluator_contract_binding(runtime_ports["repo_root"])
    protocol = importlib.import_module("sparse_rtdetr.data_protocol.evaluation")
    model = getattr(weights, "module", weights)
    if not callable(model):
        _fail("EMA evaluation weights are not callable")
    evaluate_mode = getattr(model, "eval", None)
    if callable(evaluate_mode):
        evaluate_mode()
    dataset = getattr(batches, "dataset", None)
    images = []
    detections = []
    ground_truth = []
    seen_images: set[str] = set()

    def scalar(value: Any, field: str) -> int:
        item = getattr(value, "item", None)
        if callable(item):
            value = item()
        if type(value) is not int:
            _fail(f"{field} is not an integer")
        return value

    with torch_module.no_grad():
        for batch in batches:
            if type(batch) not in {tuple, list} or len(batch) != 2:
                _fail("development loader yielded an invalid batch")
            samples, targets = batch
            if type(targets) is not list:
                _fail("development targets must be a list")
            parameters = getattr(model, "parameters", None)
            first_parameter = next(iter(parameters()), None) if callable(parameters) else None
            device = getattr(first_parameter, "device", "cuda:0")
            outputs = model(_move_to_device(samples, device))
            sizes = torch_module.stack([target["orig_size"] for target in targets], dim=0)
            output_values = outputs.values() if type(outputs) is dict else ()
            first_output = next(iter(output_values), None)
            if first_output is not None:
                sizes = sizes.to(getattr(first_output, "device", sizes.device))
            processed = postprocessor(outputs, sizes)
            stable_ids = []
            for target in targets:
                numeric_id = scalar(target.get("image_id"), "development image identity")
                vendor = getattr(dataset, "_vendor", None)
                coco = getattr(vendor, "coco", None)
                image_record = None
                if coco is not None and hasattr(coco, "loadImgs"):
                    loaded_images = coco.loadImgs(numeric_id)
                    if type(loaded_images) is list and len(loaded_images) == 1 and type(loaded_images[0]) is dict:
                        image_record = loaded_images[0]
                stable = target.get("stable_image_id")
                if type(stable) is not str or not stable:
                    stable = image_record.get("stable_image_id") if image_record is not None else None
                if type(stable) is not str or not stable:
                    stable = str(numeric_id)
                if stable in seen_images:
                    _fail("development image IDs are duplicated")
                seen_images.add(stable)
                orig_size = target.get("orig_size")
                if orig_size is None or len(orig_size) != 2:
                    _fail("development target orig_size is invalid")
                height = scalar(orig_size[0], "development image height")
                width = scalar(orig_size[1], "development image width")
                if width <= 0 or height <= 0:
                    _fail("development target dimensions are invalid")
                images.append(protocol.PrimaryImageV2(stable, width, height))
                stable_ids.append(stable)
                raw_annotations = getattr(coco, "imgToAnns", {}).get(numeric_id, []) if coco is not None else []
                if type(raw_annotations) is not list:
                    _fail("development annotation rows are invalid")
                for index, annotation in enumerate(raw_annotations):
                    if type(annotation) is not dict:
                        _fail("development annotation row is invalid")
                    bbox = annotation.get("bbox")
                    if type(bbox) is not list or len(bbox) != 4:
                        _fail("development annotation box is invalid")
                    x, y, box_width, box_height = (float(item) for item in bbox)
                    if box_width <= 0 or box_height <= 0:
                        continue
                    category = annotation.get("category_id")
                    if type(category) is not int or category < 0 or category > 11:
                        _fail("development annotation category is invalid")
                    ground_truth.append(
                        protocol.PrimaryGroundTruth(
                            annotation_id=str(annotation.get("stable_annotation_id", f"{stable}:{index}")),
                            image_id=stable,
                            category_id=category,
                            bbox_xyxy=(x, y, x + box_width, y + box_height),
                            area=float(annotation.get("area", box_width * box_height)),
                            ignore_region=category == 0,
                            ignored=bool(annotation.get("ignore", False) or annotation.get("iscrowd", 0)),
                        )
                    )
            detections.extend(postprocessor.to_detections(processed, stable_ids))
    if not images:
        _fail("development loader yielded no images")
    input_value = protocol.PrimaryEvaluatorInputV2(tuple(images), tuple(detections), tuple(ground_truth))
    result = evaluator(input_value, contract)
    validate_result(result, input_value, contract)
    return {"AP": float(result.AP), "AP50": float(result.AP50), "AR500": float(result.AR500)}


def _resolve_production_capability(
    policy: Mapping[str, Any],
    loaded: Mapping[str, Any],
    model_info: Mapping[str, Any],
    optimizer: Any,
    call_order: list[str],
) -> _ProductionCapability:
    del loaded
    call_order.append("create_data_ports")
    config = model_info["config"]
    try:
        criterion = config.criterion
        runtime_ports = _build_runtime_ports(model_info, policy)
        train_loader = runtime_ports["train_loader"]
        development_loader = runtime_ports["development_loader"]
        checkpoint_writer = runtime_ports["checkpoint_writer"]
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
        parameters = getattr(model, "parameters", None)
        first_parameter = next(iter(parameters()), None) if callable(parameters) else None
        device = getattr(first_parameter, "device", "cuda:0")
        samples = _move_to_device(samples, device)
        targets = _move_to_device(targets, device)
        outputs = model(samples, targets=targets)
        return _weighted_loss(criterion, criterion(outputs, targets))

    def primary_evaluate(ema_model: Any, epoch: int) -> dict[str, Any]:
        del epoch
        return _evaluate_primary_metrics(runtime_ports, ema_model, development_loader)

    primary_evaluate.evaluator_id = policy["selection"]["primary_evaluator"]
    call_order.append("create_primary_evaluator")
    scheduler = model_info["torch"].optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[1000],
        gamma=0.1,
    )
    warmup_type = runtime_ports.get("warmup_type")
    if not callable(warmup_type):
        _fail("production linear warmup is unavailable")
    warmup = warmup_type(scheduler, warmup_duration=2000)
    return _ProductionCapability(
        _CAPABILITY_TOKEN,
        policy=_copy(policy),
        mode=PRODUCTION_MODE,
        model=model_info["model"],
        optimizer=optimizer,
        ema=ema,
        evaluator=primary_evaluate,
        batches=batches,
        compute_loss=compute_loss,
        backward=None,
        scheduler=scheduler,
        warmup=warmup,
        checkpoint_writer=checkpoint_writer,
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
    capability = _resolve_production_capability(
        policy,
        loaded,
        model_info,
        optimizer["optimizer"],
        call_order,
    )
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
        common_keys = {"role", "epoch", "path", "size_bytes", "sha256", "mode"}
        binary_keys = common_keys | {"format", "state_sha256", "runtime_state_keys", "inventory_path", "inventory_sha256"}
        if row.get("format") == "torch_save_v1":
            _exact(row, binary_keys, "T7D checkpoint inventory row")
        else:
            _exact(row, common_keys, "T7D checkpoint inventory row")
        path = pathlib.Path(row["path"])
        allowed_parent = target / "checkpoints" if row.get("format") == "torch_save_v1" else target
        if path.parent != allowed_parent or path.is_symlink() or not path.is_file():
            _fail("T7D checkpoint path drift")
        raw = path.read_bytes()
        if row["size_bytes"] != len(raw) or row["sha256"] != _sha_bytes(raw) or row["mode"] != 0o600:
            _fail("T7D checkpoint inventory identity drift")
        if row.get("format") == "torch_save_v1":
            try:
                loaded = importlib.import_module("torch").load(path, map_location="cpu", weights_only=False)
            except Exception as exc:
                raise V2AProductionError("T7D production checkpoint is not loadable") from exc
            if type(loaded) is not dict or type(loaded.get("runtime_states")) is not dict:
                _fail("T7D production checkpoint state schema drift")
            runtime_states = loaded.pop("runtime_states")
            required_runtime = {"raw_model", "ema", "optimizer", "scheduler", "warmup", "rng_states"}
            if set(runtime_states) != required_runtime or sorted(required_runtime) != row["runtime_state_keys"]:
                _fail("T7D production checkpoint runtime state drift")
            if row["state_sha256"] != _digest(loaded):
                _fail("T7D production checkpoint metadata digest drift")
            _validate_checkpoint(loaded, checked_policy, value["run_id"])
            inventory_path = pathlib.Path(row["inventory_path"])
            if inventory_path.parent != allowed_parent or inventory_path.is_symlink() or not inventory_path.is_file():
                _fail("T7D production checkpoint inventory path drift")
            inventory_raw = inventory_path.read_bytes()
            if row["inventory_sha256"] != _sha_bytes(inventory_raw):
                _fail("T7D production checkpoint inventory digest drift")
        else:
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
