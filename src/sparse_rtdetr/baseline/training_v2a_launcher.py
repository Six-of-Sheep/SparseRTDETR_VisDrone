"""T7E exactly-once launcher boundary for the frozen baseline-v2a CLI.

The module is inert when imported.  CPU tests use the separate fake entry and
never consume a disk authorization, create a production target, or call tmux.
Only the private owner implementation can execute the two fixed tmux calls.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import pathlib
import stat
import subprocess
import sys
from typing import Any, Mapping

from sparse_rtdetr.baseline import training_v2a_production as _production


SCHEMA_VERSION = 1
LAUNCH_ID = "rtdetrv2_r18_visdrone_baseline_v2a_exactly_once_launcher_t7e_r1"
PRODUCTION_MODULE_NAME = "sparse_rtdetr.baseline.training_v2a_production"
MODULE_RELATIVE_PATH = "src/sparse_rtdetr/baseline/training_v2a_launcher.py"
AUTHORIZATION_KIND = "T7E_V2A_OWNER_AUTHORIZATION"
PLAN_KIND = "T7E_V2A_LAUNCH_PLAN"
RECEIPT_KIND = "T7E_V2A_AUTHORIZATION_CONSUMPTION_RECEIPT"
SNAPSHOT_KIND = "T7E_V2A_IMMEDIATE_LAUNCH_SNAPSHOT"
RESULT_KIND = "T7E_V2A_IMMEDIATE_LAUNCH_RESULT"
TMUX_ACCEPTED = "TMUX_ACCEPTED"
PERMANENT_FAIL = "PERMANENT_FAIL"
_ZERO_SHA = "0" * 64

AUTHORIZATION_KEYS = {
    "schema_version",
    "kind",
    "launch_id",
    "authorization_id",
    "run_id",
    "nonce",
    "authorization_path",
    "repo_root",
    "git_identity",
    "t7d_policy",
    "t7d_policy_sha256",
    "t7d_descriptor",
    "t7d_authorization_context",
    "t7a_authority",
    "t7a_authority_sha256",
    "host_gpu_policy",
    "target_identity",
    "owner",
    "consumed",
    "consumption_receipt_path",
    "plan_binding_sha256",
    "authorization_sha256",
}
PLAN_KEYS = {
    "schema_version",
    "kind",
    "launch_id",
    "authorization_id",
    "run_id",
    "nonce",
    "repo_root",
    "target_root",
    "python_path",
    "authorization_path",
    "authorization_sha256",
    "descriptor_path",
    "descriptor_sha256",
    "authorization_context_path",
    "authorization_context_sha256",
    "t7d_contract_sha256",
    "t7d_source_identity",
    "t7d_data_roots",
    "t7d_environment_identity",
    "t7a_authority_sha256",
    "host_gpu_policy",
    "target_identity",
    "parameter_layers",
    "production_argv",
    "tmux_session_name",
    "invocation_policy",
    "plan_sha256",
}
RECEIPT_KEYS = {
    "schema_version",
    "kind",
    "launch_id",
    "authorization_id",
    "run_id",
    "nonce",
    "authorization_path",
    "authorization_sha256",
    "authorization_file_identity",
    "descriptor_sha256",
    "authorization_context_sha256",
    "plan_sha256",
    "t7d_contract_sha256",
    "t7a_authority_sha256",
    "receipt_path",
    "consumed",
    "mode",
    "receipt_sha256",
}
RESULT_KEYS = {
    "schema_version",
    "kind",
    "status",
    "classification",
    "launch_id",
    "authorization_id",
    "run_id",
    "nonce",
    "plan_sha256",
    "receipt_sha256",
    "descriptor_sha256",
    "authorization_context_sha256",
    "repo_root",
    "tmux_session_name",
    "production_argv",
    "tmux_new_session_count",
    "immediate_snapshot_count",
    "launch_observation",
    "immediate_snapshot",
    "stdout_file",
    "stderr_file",
    "snapshot_file",
    "production",
    "training_certified",
    "call_counts",
    "result_sha256",
    "result_file",
}
TARGET_IDENTITY_KEYS = {
    "control_root",
    "plan_path",
    "descriptor_path",
    "authorization_context_path",
    "receipt_path",
    "result_path",
    "stdout_path",
    "stderr_path",
    "snapshot_path",
    "lock_path",
    "t7d_target_root",
    "t7d_process_root",
    "t7d_outer_root",
    "protected_paths",
    "tmux_session_name",
    "session_absent",
}
GIT_IDENTITY_KEYS = {
    "branch",
    "head",
    "parent",
    "tree",
    "upstream",
    "upstream_sha",
    "worktree_status",
    "cache_counts",
}
HOST_GPU_POLICY_KEYS = {
    "policy_id",
    "tmux_path",
    "cuda_visible_devices",
    "gpu_index",
    "gpu_probe_executed",
    "cuda_initialized",
    "exclusive_host_required",
    "training_allowed",
    "formal_training_executed",
    "network_download",
}
INVOCATION_POLICY = {
    "authorization_consumption": "exclusive_once",
    "descriptor_persistence": "O_EXCL_fsync_file_parent",
    "overwrite": False,
    "resume": False,
    "retry": False,
    "shell": False,
    "tmux_new_session_count": 1,
    "tmux_immediate_snapshot_count": 1,
    "ports": "internal_owner_only",
}

__all__ = (
    "V2ALauncherError",
    "canonical_v2a_launcher_bytes",
    "build_v2a_owner_authorization",
    "validate_v2a_owner_authorization",
    "build_v2a_launch_plan",
    "validate_v2a_launch_plan",
    "consume_v2a_authorization",
    "validate_v2a_consumption_receipt",
    "validate_v2a_immediate_launch_result",
    "run_v2a_cpu_fake_launch",
    "run_v2a_production_launch",
)


class V2ALauncherError(ValueError):
    """Raised when a T7E authorization, plan, or evidence boundary drifts."""


def _fail(message: str) -> None:
    raise V2ALauncherError(message)


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
        raise V2ALauncherError("value is not canonical JSON") from exc


def canonical_v2a_launcher_bytes(value: Any) -> bytes:
    """Return canonical bytes for a T7E JSON value without a trailing LF."""

    return _canonical(value)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value: Any) -> str:
    return _sha_bytes(_canonical(value))


def _without(value: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    excluded = set(keys)
    return {key: _copy(child) for key, child in value.items() if key not in excluded}


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


def _git_oid(value: Any, field: str) -> str:
    value = _string(value, field)
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        _fail(f"{field} is not a Git object ID")
    return value


def _absolute_path(value: Any, field: str) -> pathlib.Path:
    try:
        raw = os.fspath(value)
    except TypeError as exc:
        raise V2ALauncherError(f"{field} is not an absolute path") from exc
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
        raise V2ALauncherError(f"{field} is unavailable") from exc
    if resolved != path or path.is_symlink() or not path.is_dir():
        _fail(f"{field} is not a canonical directory")
    return path


def _parent_directory(path: pathlib.Path, field: str) -> pathlib.Path:
    try:
        parent = path.parent.resolve(strict=True)
    except OSError as exc:
        raise V2ALauncherError(f"{field} parent is unavailable") from exc
    if parent != path.parent or path.parent.is_symlink() or not path.parent.is_dir():
        _fail(f"{field} parent is not a canonical directory")
    return path.parent


def _file_identity(path: pathlib.Path, field: str, *, mode: int | None = None) -> dict[str, Any]:
    try:
        observed = path.lstat()
        raw = path.read_bytes()
    except OSError as exc:
        raise V2ALauncherError(f"{field} is unavailable") from exc
    if path.is_symlink() or not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
        _fail(f"{field} is not a regular unique file")
    observed_mode = stat.S_IMODE(observed.st_mode)
    if mode is not None and observed_mode != mode:
        _fail(f"{field} mode drift")
    return {
        "device": observed.st_dev,
        "inode": observed.st_ino,
        "mode": observed_mode,
        "nlink": observed.st_nlink,
        "uid": observed.st_uid,
        "gid": observed.st_gid,
        "size_bytes": len(raw),
        "sha256": _sha_bytes(raw),
    }


def _validate_file_ref(value: Any, field: str, *, path: pathlib.Path | None = None) -> dict[str, Any]:
    expected = {"path", "size_bytes", "sha256", "mode", "nlink"}
    value = _exact(value, expected, field)
    actual_path = _absolute_path(value["path"], f"{field}.path")
    _integer(value["size_bytes"], f"{field}.size_bytes", minimum=0)
    _sha(value["sha256"], f"{field}.sha256")
    _integer(value["mode"], f"{field}.mode", minimum=0)
    _integer(value["nlink"], f"{field}.nlink", minimum=1)
    if value["mode"] != 0o600 or value["nlink"] != 1:
        _fail(f"{field} metadata drift")
    if path is not None and actual_path != path:
        _fail(f"{field}.path drift")
    return value


def _validate_git_identity(value: Any, field: str = "git_identity") -> dict[str, Any]:
    value = _exact(value, GIT_IDENTITY_KEYS, field)
    _string(value["branch"], f"{field}.branch")
    for name in ("head", "parent", "tree", "upstream_sha"):
        _git_oid(value[name], f"{field}.{name}")
    _string(value["upstream"], f"{field}.upstream")
    if value["worktree_status"] != "":
        _fail(f"{field}.worktree_status is not clean")
    cache = _exact(value["cache_counts"], {"pycache", "pyc", "pytest_cache"}, f"{field}.cache_counts")
    for name in cache:
        _integer(cache[name], f"{field}.cache_counts.{name}", minimum=0)
        if cache[name] != 0:
            _fail(f"{field}.cache_counts.{name} is nonzero")
    return value


def _validate_host_gpu_policy(value: Any, field: str = "host_gpu_policy") -> dict[str, Any]:
    value = _exact(value, HOST_GPU_POLICY_KEYS, field)
    _string(value["policy_id"], f"{field}.policy_id")
    tmux_path = _absolute_path(value["tmux_path"], f"{field}.tmux_path")
    if not tmux_path.name:
        _fail(f"{field}.tmux_path is empty")
    _string(value["cuda_visible_devices"], f"{field}.cuda_visible_devices", nonempty=False)
    _integer(value["gpu_index"], f"{field}.gpu_index", minimum=0)
    for name in ("gpu_probe_executed", "cuda_initialized", "exclusive_host_required", "training_allowed", "formal_training_executed", "network_download"):
        _bool(value[name], f"{field}.{name}")
    if value["gpu_probe_executed"] or value["cuda_initialized"] or value["formal_training_executed"]:
        _fail(f"{field} is no longer fail-closed")
    if value["network_download"]:
        _fail(f"{field}.network_download must be false")
    return value


def _validate_target_identity(value: Any, field: str = "target_identity") -> dict[str, Any]:
    value = _exact(value, TARGET_IDENTITY_KEYS, field)
    path_fields = tuple(key for key in TARGET_IDENTITY_KEYS if key not in {"protected_paths", "tmux_session_name", "session_absent"})
    paths = {}
    for name in path_fields:
        paths[name] = _absolute_path(value[name], f"{field}.{name}")
    _string(value["tmux_session_name"], f"{field}.tmux_session_name")
    if not value["tmux_session_name"].startswith("t7e_"):
        _fail(f"{field}.tmux_session_name is not independent")
    _bool(value["session_absent"], f"{field}.session_absent")
    if value["session_absent"] is not True:
        _fail(f"{field}.session_absent must be true before launch")
    protected = value["protected_paths"]
    if type(protected) is not list or not protected or any(type(item) is not str for item in protected):
        _fail(f"{field}.protected_paths must be a nonempty string list")
    protected_paths = [_absolute_path(item, f"{field}.protected_paths[{index}]") for index, item in enumerate(protected)]
    if len(set(protected_paths)) != len(protected_paths):
        _fail(f"{field}.protected_paths contains duplicates")
    all_paths = list(paths.values()) + protected_paths
    if len(set(all_paths)) != len(all_paths):
        _fail(f"{field} contains colliding paths")
    for name in ("plan_path", "descriptor_path", "authorization_context_path", "receipt_path", "result_path", "stdout_path", "stderr_path", "snapshot_path"):
        if not pathlib.Path(value[name]).name.startswith("t7e_"):
            _fail(f"{field}.{name} is not a T7E artifact")
    _canonical_directory(value["control_root"], f"{field}.control_root")
    return value


def _flat_data_roots(bound: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    roles = bound.get("roles") if type(bound) is dict else None
    if type(roles) is not dict or set(roles) != {"train_core", "development"}:
        _fail("T7D data binding role drift")
    result = {}
    for role in ("train_core", "development"):
        item = roles[role]
        if type(item) is not dict:
            _fail(f"T7D data binding {role} is not an object")
        result[role] = {"role": item["role"], "root": item["root"], "annotation": item["annotation"]}
    return result


def _validate_t7d_policy_snapshot(policy: Mapping[str, Any], descriptor: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, Any]:
    if type(policy) is not dict:
        _fail("T7D policy must be a builtin dict")
    _assert_builtin(policy, "T7D policy")
    if policy.get("policy_sha256") != _digest(_without(policy, "policy_sha256")):
        _fail("T7D policy digest drift")
    if policy.get("contract_sha256") != descriptor["contract_sha256"]:
        _fail("T7D policy contract binding drift")
    if policy.get("source_identity") != descriptor["source_identity"]:
        _fail("T7D policy source binding drift")
    target = policy.get("target")
    if type(target) is not dict or target.get("path") != descriptor["target_root"]:
        _fail("T7D policy target binding drift")
    data_roots = policy.get("data_roots")
    if type(data_roots) is not dict or _flat_data_roots(data_roots) != context["data_roots"]:
        _fail("T7D policy data binding drift")
    if policy.get("production") != {
        "owner_authorization_consumed": False,
        "gpu_probe_executed": False,
        "formal_training_executed": False,
        "training_ready": False,
        "independent_audit_pass": False,
    }:
        _fail("T7D policy production state is not fail-closed")
    runtime_policy = policy.get("runtime_policy")
    if type(runtime_policy) is not dict or type(runtime_policy.get("contract_binding")) is not dict:
        _fail("T7D runtime policy binding is unavailable")
    contract_binding = runtime_policy["contract_binding"]
    if type(contract_binding.get("authority_binding")) is not dict:
        _fail("T7A authority binding is unavailable")
    return _copy(policy)


def _validate_authority(value: Any, expected: Mapping[str, Any], field: str = "t7a_authority") -> dict[str, Any]:
    if type(value) is not dict:
        _fail(f"{field} must be a builtin dict")
    _assert_builtin(value, field)
    if value != expected:
        _fail(f"{field} drifted from the T7D public binding")
    return _copy(value)


def _authorization_digest(value: Mapping[str, Any]) -> str:
    return _digest(_without(value, "authorization_sha256", "plan_binding_sha256"))


def _validate_t7d_pair(descriptor: Mapping[str, Any], context: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        checked_descriptor = _production.validate_v2a_detached_launch_descriptor(descriptor)
        checked_context = _production.validate_v2a_authorization_context(context, descriptor=checked_descriptor)
    except Exception as exc:
        if isinstance(exc, V2ALauncherError):
            raise
        raise V2ALauncherError(f"T7D descriptor/context validation failed: {exc}") from exc
    return checked_descriptor, checked_context


def validate_v2a_owner_authorization(
    authorization: Mapping[str, Any],
    *,
    t7d_policy: Mapping[str, Any] | None = None,
    expected_git_identity: Mapping[str, Any] | None = None,
    expected_authorization_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Validate the closed, still-unconsumed T7E owner authorization."""

    value = _exact(authorization, AUTHORIZATION_KEYS, "T7E owner authorization")
    _assert_builtin(value, "T7E owner authorization")
    if value["schema_version"] != SCHEMA_VERSION or value["kind"] != AUTHORIZATION_KIND or value["launch_id"] != LAUNCH_ID:
        _fail("T7E owner authorization identity drift")
    for name in ("authorization_id", "run_id", "nonce", "owner"):
        _string(value[name], f"authorization.{name}")
    if not value["owner"].startswith("owner:"):
        _fail("authorization.owner is not an owner identity")
    auth_path = _absolute_path(value["authorization_path"], "authorization.authorization_path")
    if expected_authorization_path is not None and auth_path != _absolute_path(expected_authorization_path, "expected_authorization_path"):
        _fail("authorization path drift")
    repo_root = _canonical_directory(value["repo_root"], "authorization.repo_root")
    descriptor, context = _validate_t7d_pair(value["t7d_descriptor"], value["t7d_authorization_context"])
    if descriptor["repo_root"] != str(repo_root) or descriptor["run_id"] != value["run_id"] or descriptor["nonce"] != value["nonce"]:
        _fail("T7E and T7D run identity drift")
    checked_policy = _validate_t7d_policy_snapshot(value["t7d_policy"], descriptor, context)
    if t7d_policy is not None:
        supplied_policy = _validate_t7d_policy_snapshot(t7d_policy, descriptor, context)
        if supplied_policy != checked_policy:
            _fail("authorization T7D policy does not match supplied public binding")
    _sha(value["t7d_policy_sha256"], "authorization.t7d_policy_sha256")
    if value["t7d_policy_sha256"] != checked_policy["policy_sha256"]:
        _fail("authorization T7D policy digest drift")
    expected_authority = checked_policy["runtime_policy"]["contract_binding"]["authority_binding"]
    _validate_authority(value["t7a_authority"], expected_authority)
    _sha(value["t7a_authority_sha256"], "authorization.t7a_authority_sha256")
    if value["t7a_authority_sha256"] != _digest(expected_authority):
        _fail("authorization T7A authority digest drift")
    _validate_git_identity(value["git_identity"], "authorization.git_identity")
    if expected_git_identity is not None and value["git_identity"] != _validate_git_identity(expected_git_identity, "expected_git_identity"):
        _fail("authorization Git identity drift")
    _validate_host_gpu_policy(value["host_gpu_policy"])
    target = _validate_target_identity(value["target_identity"])
    if auth_path in {pathlib.Path(target[name]) for name in TARGET_IDENTITY_KEYS if name.endswith("_path")}:
        _fail("authorization path collides with a T7E target")
    if target["t7d_target_root"] != descriptor["target_root"]:
        _fail("authorization T7D target drift")
    if value["consumed"] is not False or value["consumption_receipt_path"] is not None:
        _fail("T7E authorization is already consumed")
    _sha(value["plan_binding_sha256"], "authorization.plan_binding_sha256")
    if value["plan_binding_sha256"] not in {_ZERO_SHA}:
        _sha(value["plan_binding_sha256"], "authorization.plan_binding_sha256")
    _sha(value["authorization_sha256"], "authorization.authorization_sha256")
    if value["authorization_sha256"] != _authorization_digest(value):
        _fail("T7E owner authorization digest drift")
    return _copy(value)


def build_v2a_owner_authorization(
    authorization_path: str | os.PathLike[str],
    *,
    t7d_policy: Mapping[str, Any],
    descriptor: Mapping[str, Any],
    authorization_context: Mapping[str, Any],
    git_identity: Mapping[str, Any],
    host_gpu_policy: Mapping[str, Any],
    target_identity: Mapping[str, Any],
    owner: str,
    authorization_id: str,
) -> dict[str, Any]:
    """Build a detached T7E authorization from the T7D public binding."""

    auth_path = _absolute_path(authorization_path, "authorization_path")
    _string(owner, "owner")
    if not owner.startswith("owner:"):
        _fail("owner must use the owner: namespace")
    _string(authorization_id, "authorization_id")
    checked_descriptor, checked_context = _validate_t7d_pair(descriptor, authorization_context)
    checked_policy = _validate_t7d_policy_snapshot(t7d_policy, checked_descriptor, checked_context)
    checked_git = _validate_git_identity(git_identity)
    checked_host = _validate_host_gpu_policy(host_gpu_policy)
    checked_target = _validate_target_identity(target_identity)
    value = {
        "schema_version": SCHEMA_VERSION,
        "kind": AUTHORIZATION_KIND,
        "launch_id": LAUNCH_ID,
        "authorization_id": authorization_id,
        "run_id": checked_descriptor["run_id"],
        "nonce": checked_descriptor["nonce"],
        "authorization_path": str(auth_path),
        "repo_root": checked_descriptor["repo_root"],
        "git_identity": _copy(checked_git),
        "t7d_policy": _copy(checked_policy),
        "t7d_policy_sha256": checked_policy["policy_sha256"],
        "t7d_descriptor": checked_descriptor,
        "t7d_authorization_context": checked_context,
        "t7a_authority": _copy(checked_policy["runtime_policy"]["contract_binding"]["authority_binding"]),
        "t7a_authority_sha256": _digest(checked_policy["runtime_policy"]["contract_binding"]["authority_binding"]),
        "host_gpu_policy": _copy(checked_host),
        "target_identity": _copy(checked_target),
        "owner": owner,
        "consumed": False,
        "consumption_receipt_path": None,
        "plan_binding_sha256": _ZERO_SHA,
    }
    value["authorization_sha256"] = _authorization_digest(value)
    return validate_v2a_owner_authorization(value)


def _python_identity(value: str | os.PathLike[str]) -> pathlib.Path:
    path = _absolute_path(value, "python_path")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise V2ALauncherError("python_path is unavailable") from exc
    if path.is_symlink():
        path = resolved
    if path != resolved or not path.is_file() or path.is_symlink():
        _fail("python_path must be a regular non-symlink executable")
    observed = path.lstat()
    if not stat.S_ISREG(observed.st_mode) or not os.access(path, os.X_OK):
        _fail("python_path is not an executable regular file")
    return path


def _plan_digest(value: Mapping[str, Any]) -> str:
    return _digest(_without(value, "plan_sha256"))


def _production_argv(plan: Mapping[str, Any]) -> list[str]:
    return [
        plan["python_path"],
        "-m",
        PRODUCTION_MODULE_NAME,
        "--descriptor",
        plan["descriptor_path"],
        "--authorization-context",
        plan["authorization_context_path"],
    ]


def build_v2a_launch_plan(
    authorization: Mapping[str, Any],
    *,
    python_path: str | os.PathLike[str] | None = None,
    t7d_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the three-layer fixed production argv and its digest closure."""

    checked = validate_v2a_owner_authorization(authorization, t7d_policy=t7d_policy)
    target = checked["target_identity"]
    python = _python_identity(sys.executable if python_path is None else python_path)
    descriptor = checked["t7d_descriptor"]
    context = checked["t7d_authorization_context"]
    descriptor_sha = _digest(descriptor)
    context_sha = _digest(context)
    plan_base = {
        "schema_version": SCHEMA_VERSION,
        "kind": PLAN_KIND,
        "launch_id": LAUNCH_ID,
        "authorization_id": checked["authorization_id"],
        "run_id": checked["run_id"],
        "nonce": checked["nonce"],
        "repo_root": checked["repo_root"],
        "target_root": descriptor["target_root"],
        "authorization_path": checked["authorization_path"],
        "authorization_sha256": checked["authorization_sha256"],
        "descriptor_path": target["descriptor_path"],
        "descriptor_sha256": descriptor_sha,
        "authorization_context_path": target["authorization_context_path"],
        "authorization_context_sha256": context_sha,
        "t7d_contract_sha256": descriptor["contract_sha256"],
        "t7d_source_identity": _copy(descriptor["source_identity"]),
        "t7d_data_roots": _copy(checked["t7d_policy"]["data_roots"]),
        "t7d_environment_identity": _copy(checked["t7d_policy"]["environment_identity"]),
        "t7a_authority_sha256": checked["t7a_authority_sha256"],
        "host_gpu_policy": _copy(checked["host_gpu_policy"]),
        "target_identity": _copy(target),
        "parameter_layers": {
            "descriptor": {"path": target["descriptor_path"], "sha256": descriptor_sha, "argv_fragment": ["--descriptor", target["descriptor_path"]]},
            "authorization_context": {"path": target["authorization_context_path"], "sha256": context_sha, "argv_fragment": ["--authorization-context", target["authorization_context_path"]]},
        },
        "python_path": str(python),
        "production_argv": [],
        "tmux_session_name": target["tmux_session_name"],
        "invocation_policy": _copy(INVOCATION_POLICY),
    }
    plan_base["production_argv"] = _production_argv(plan_base)
    plan_base["parameter_layers"]["production"] = {
        "argv": _copy(plan_base["production_argv"]),
        "sha256": _digest(plan_base["production_argv"]),
    }
    plan = {**plan_base, "plan_sha256": _plan_digest(plan_base)}
    return validate_v2a_launch_plan(plan, checked)


def validate_v2a_launch_plan(plan: Mapping[str, Any], authorization: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute every T7E plan digest and argv cross-binding."""

    value = _exact(plan, PLAN_KEYS, "T7E launch plan")
    _assert_builtin(value, "T7E launch plan")
    checked_auth = validate_v2a_owner_authorization(authorization)
    if value["schema_version"] != SCHEMA_VERSION or value["kind"] != PLAN_KIND or value["launch_id"] != LAUNCH_ID:
        _fail("T7E launch plan identity drift")
    for name in ("authorization_id", "run_id", "nonce", "repo_root", "target_root", "authorization_path", "descriptor_path", "authorization_context_path", "tmux_session_name"):
        _string(value[name], f"plan.{name}")
    if value["authorization_id"] != checked_auth["authorization_id"] or value["run_id"] != checked_auth["run_id"] or value["nonce"] != checked_auth["nonce"]:
        _fail("T7E plan run identity drift")
    if value["repo_root"] != checked_auth["repo_root"] or value["authorization_path"] != checked_auth["authorization_path"]:
        _fail("T7E plan repository/authorization path drift")
    _absolute_path(value["target_root"], "plan.target_root")
    python_path = _absolute_path(value["python_path"], "plan.python_path")
    _python_identity(python_path)
    for name in ("descriptor_path", "authorization_context_path"):
        _absolute_path(value[name], f"plan.{name}")
    _sha(value["authorization_sha256"], "plan.authorization_sha256")
    if value["authorization_sha256"] != checked_auth["authorization_sha256"]:
        _fail("T7E plan authorization digest drift")
    _sha(value["descriptor_sha256"], "plan.descriptor_sha256")
    _sha(value["authorization_context_sha256"], "plan.authorization_context_sha256")
    if value["descriptor_sha256"] != _digest(checked_auth["t7d_descriptor"]):
        _fail("T7E descriptor digest drift")
    if value["authorization_context_sha256"] != _digest(checked_auth["t7d_authorization_context"]):
        _fail("T7E context digest drift")
    if value["target_root"] != checked_auth["t7d_descriptor"]["target_root"] or value["tmux_session_name"] != checked_auth["target_identity"]["tmux_session_name"]:
        _fail("T7E plan target/session drift")
    if value["t7d_contract_sha256"] != checked_auth["t7d_descriptor"]["contract_sha256"] or value["t7d_source_identity"] != checked_auth["t7d_descriptor"]["source_identity"]:
        _fail("T7E plan T7D source drift")
    if value["t7d_data_roots"] != checked_auth["t7d_policy"]["data_roots"] or value["t7d_environment_identity"] != checked_auth["t7d_policy"]["environment_identity"]:
        _fail("T7E plan T7D runtime identity drift")
    if value["t7a_authority_sha256"] != checked_auth["t7a_authority_sha256"]:
        _fail("T7E plan T7A authority drift")
    _validate_host_gpu_policy(value["host_gpu_policy"], "plan.host_gpu_policy")
    if value["host_gpu_policy"] != checked_auth["host_gpu_policy"]:
        _fail("T7E plan host/GPU policy drift")
    if value["target_identity"] != checked_auth["target_identity"]:
        _fail("T7E plan target identity drift")
    layers = _exact(value["parameter_layers"], {"descriptor", "authorization_context", "production"}, "plan.parameter_layers")
    for name, path, digest in (("descriptor", value["descriptor_path"], value["descriptor_sha256"]), ("authorization_context", value["authorization_context_path"], value["authorization_context_sha256"])):
        item = _exact(layers[name], {"path", "sha256", "argv_fragment"}, f"plan.parameter_layers.{name}")
        expected_fragment = ["--descriptor", path] if name == "descriptor" else ["--authorization-context", path]
        if item["path"] != path or item["sha256"] != digest or item["argv_fragment"] != expected_fragment:
            _fail(f"plan parameter layer {name} drift")
    production_layer = _exact(layers["production"], {"argv", "sha256"}, "plan.parameter_layers.production")
    _assert_builtin(value["production_argv"], "plan.production_argv")
    if type(value["production_argv"]) is not list or any(type(item) is not str for item in value["production_argv"]):
        _fail("plan.production_argv must be a string list")
    if len(value["production_argv"]) != 7 or value["production_argv"][0] != str(python_path) or value["production_argv"][1:] != ["-m", PRODUCTION_MODULE_NAME, "--descriptor", value["descriptor_path"], "--authorization-context", value["authorization_context_path"]]:
        _fail("T7E production argv shape drift")
    if production_layer["argv"] != value["production_argv"] or production_layer["sha256"] != _digest(value["production_argv"]):
        _fail("T7E production argv layer drift")
    if value["invocation_policy"] != INVOCATION_POLICY:
        _fail("T7E invocation policy drift")
    _sha(value["plan_sha256"], "plan.plan_sha256")
    if value["plan_sha256"] != _plan_digest(value):
        _fail("T7E plan digest drift")
    if checked_auth["plan_binding_sha256"] not in {_ZERO_SHA, value["plan_sha256"]}:
        _fail("T7E authorization plan cross-binding drift")
    return _copy(value)


def _write_new_bytes(path: pathlib.Path, raw: bytes, field: str, *, mode: int = 0o600) -> dict[str, Any]:
    _parent_directory(path, field)
    if path.exists() or path.is_symlink():
        _fail(f"{field} already exists")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = None
    try:
        fd = os.open(path, flags, mode)
        offset = 0
        while offset < len(raw):
            offset += os.write(fd, raw[offset:])
        os.fsync(fd)
    except OSError as exc:
        raise V2ALauncherError(f"{field} exclusive write failed") from exc
    finally:
        if fd is not None:
            os.close(fd)
    try:
        dir_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError as exc:
        raise V2ALauncherError(f"{field} parent fsync failed") from exc
    return _file_identity(path, field, mode=mode)


def _write_new_json(path: pathlib.Path, value: Mapping[str, Any], field: str) -> dict[str, Any]:
    return _write_new_bytes(path, _canonical(value), field)


def _read_canonical_json(path: pathlib.Path, field: str) -> tuple[dict[str, Any], bytes]:
    identity = _file_identity(path, field, mode=0o600)
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V2ALauncherError(f"{field} is not canonical JSON") from exc
    if type(value) is not dict or _canonical(value) != raw or identity["sha256"] != _sha_bytes(raw):
        _fail(f"{field} bytes drift")
    return value, raw


def _assert_absent(path: pathlib.Path, field: str) -> None:
    _parent_directory(path, field)
    if path.exists() or path.is_symlink():
        _fail(f"{field} must be absent")


def _preflight(
    authorization: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    observed_git_identity: Mapping[str, Any] | None = None,
    t7d_policy: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    checked_auth = validate_v2a_owner_authorization(authorization, t7d_policy=t7d_policy, expected_git_identity=observed_git_identity)
    checked_plan = validate_v2a_launch_plan(plan, checked_auth)
    current_source = _production.current_v2a_source_identity(checked_auth["repo_root"])
    if current_source != checked_auth["t7d_descriptor"]["source_identity"]:
        _fail("T7D source identity changed before T7E consumption")
    target = checked_auth["target_identity"]
    for name in ("plan_path", "descriptor_path", "authorization_context_path", "receipt_path", "result_path", "stdout_path", "stderr_path", "snapshot_path", "lock_path"):
        _assert_absent(pathlib.Path(target[name]), f"T7E target.{name}")
    for index, raw_path in enumerate(target["protected_paths"]):
        _assert_absent(pathlib.Path(raw_path), f"protected target[{index}]")
    _assert_absent(pathlib.Path(target["t7d_target_root"]), "T7D target root")
    _assert_absent(pathlib.Path(target["t7d_process_root"]), "T7D process target root")
    _assert_absent(pathlib.Path(target["t7d_outer_root"]), "T7D outer target root")
    return checked_auth, checked_plan


def _authorization_file_identity(path: pathlib.Path) -> dict[str, Any]:
    return _file_identity(path, "T7E owner authorization", mode=0o600)


def _receipt_digest(value: Mapping[str, Any]) -> str:
    return _digest(_without(value, "receipt_sha256"))


def consume_v2a_authorization(
    authorization: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    mode: str = "cpu_fake",
    owner_file_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create the one immutable T7E consumption receipt with O_EXCL."""

    if mode not in {"cpu_fake", "production"}:
        _fail("T7E consumption mode is invalid")
    checked_auth = validate_v2a_owner_authorization(authorization)
    checked_plan = validate_v2a_launch_plan(plan, checked_auth)
    target = checked_auth["target_identity"]
    auth_path = pathlib.Path(checked_auth["authorization_path"])
    if mode == "production":
        on_disk, _ = _read_canonical_json(auth_path, "T7E owner authorization")
        if on_disk != checked_auth:
            _fail("owner authorization bytes changed before consumption")
        observed = _authorization_file_identity(auth_path)
        if owner_file_identity is not None and observed != _validate_file_ref(owner_file_identity, "owner_file_identity"):
            _fail("owner authorization file identity drift")
        file_identity: dict[str, Any] | None = observed
    else:
        if owner_file_identity is not None:
            file_identity = _validate_file_ref(owner_file_identity, "owner_file_identity")
        else:
            file_identity = None
    receipt_base = {
        "schema_version": SCHEMA_VERSION,
        "kind": RECEIPT_KIND,
        "launch_id": LAUNCH_ID,
        "authorization_id": checked_auth["authorization_id"],
        "run_id": checked_auth["run_id"],
        "nonce": checked_auth["nonce"],
        "authorization_path": checked_auth["authorization_path"],
        "authorization_sha256": checked_auth["authorization_sha256"],
        "authorization_file_identity": file_identity,
        "descriptor_sha256": checked_plan["descriptor_sha256"],
        "authorization_context_sha256": checked_plan["authorization_context_sha256"],
        "plan_sha256": checked_plan["plan_sha256"],
        "t7d_contract_sha256": checked_plan["t7d_contract_sha256"],
        "t7a_authority_sha256": checked_plan["t7a_authority_sha256"],
        "receipt_path": target["receipt_path"],
        "consumed": True,
        "mode": mode,
    }
    receipt = {**receipt_base, "receipt_sha256": _receipt_digest(receipt_base)}
    path = pathlib.Path(target["receipt_path"])
    _write_new_json(path, receipt, "T7E consumption receipt")
    checked, raw = _read_canonical_json(path, "T7E consumption receipt")
    if checked != receipt or raw != _canonical(receipt):
        _fail("T7E consumption receipt readback drift")
    return validate_v2a_consumption_receipt(receipt, checked_plan, checked_auth)


def validate_v2a_consumption_receipt(
    receipt: Mapping[str, Any],
    plan: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the durable, single-consumption T7E receipt."""

    value = _exact(receipt, RECEIPT_KEYS, "T7E consumption receipt")
    _assert_builtin(value, "T7E consumption receipt")
    checked_auth = validate_v2a_owner_authorization(authorization)
    checked_plan = validate_v2a_launch_plan(plan, checked_auth)
    if value["schema_version"] != SCHEMA_VERSION or value["kind"] != RECEIPT_KIND or value["launch_id"] != LAUNCH_ID:
        _fail("T7E receipt identity drift")
    for name in ("authorization_id", "run_id", "nonce", "authorization_path", "receipt_path", "mode"):
        _string(value[name], f"receipt.{name}")
    if value["mode"] not in {"cpu_fake", "production"} or value["consumed"] is not True:
        _fail("T7E receipt consumption state drift")
    if value["authorization_id"] != checked_auth["authorization_id"] or value["run_id"] != checked_auth["run_id"] or value["nonce"] != checked_auth["nonce"]:
        _fail("T7E receipt run identity drift")
    if value["authorization_path"] != checked_auth["authorization_path"] or value["receipt_path"] != checked_auth["target_identity"]["receipt_path"]:
        _fail("T7E receipt path drift")
    for name, expected in (("authorization_sha256", checked_auth["authorization_sha256"]), ("descriptor_sha256", checked_plan["descriptor_sha256"]), ("authorization_context_sha256", checked_plan["authorization_context_sha256"]), ("plan_sha256", checked_plan["plan_sha256"]), ("t7d_contract_sha256", checked_plan["t7d_contract_sha256"]), ("t7a_authority_sha256", checked_plan["t7a_authority_sha256"])):
        _sha(value[name], f"receipt.{name}")
        if value[name] != expected:
            _fail(f"T7E receipt {name} drift")
    if value["authorization_file_identity"] is not None:
        _validate_file_ref(value["authorization_file_identity"], "receipt.authorization_file_identity")
    _sha(value["receipt_sha256"], "receipt.receipt_sha256")
    if value["receipt_sha256"] != _receipt_digest(value):
        _fail("T7E receipt digest drift")
    return _copy(value)


def _validate_launch_observation(value: Any, field: str = "launch_observation") -> dict[str, Any]:
    value = _exact(value, {"return_code", "pid", "stdout_size_bytes", "stdout_sha256", "stderr_size_bytes", "stderr_sha256", "exception_type"}, field)
    _integer(value["return_code"], f"{field}.return_code")
    _integer(value["pid"], f"{field}.pid", minimum=0)
    for name in ("stdout_size_bytes", "stderr_size_bytes"):
        _integer(value[name], f"{field}.{name}", minimum=0)
    for name in ("stdout_sha256", "stderr_sha256"):
        _sha(value[name], f"{field}.{name}")
    if value["exception_type"] is not None:
        _string(value["exception_type"], f"{field}.exception_type")
    return value


def _validate_snapshot(value: Any, field: str = "immediate_snapshot") -> dict[str, Any]:
    value = _exact(value, {"kind", "session_name", "return_code", "present", "stdout_size_bytes", "stdout_sha256", "stderr_size_bytes", "stderr_sha256", "exception_type"}, field)
    if value["kind"] != SNAPSHOT_KIND:
        _fail(f"{field}.kind drift")
    _string(value["session_name"], f"{field}.session_name")
    _integer(value["return_code"], f"{field}.return_code")
    _bool(value["present"], f"{field}.present")
    for name in ("stdout_size_bytes", "stderr_size_bytes"):
        _integer(value[name], f"{field}.{name}", minimum=0)
    for name in ("stdout_sha256", "stderr_sha256"):
        _sha(value[name], f"{field}.{name}")
    if value["exception_type"] is not None:
        _string(value["exception_type"], f"{field}.exception_type")
    return value


def _validate_production_state(value: Any, field: str = "production") -> dict[str, Any]:
    value = _exact(value, {"owner_authorization_consumed", "gpu_probe_executed", "formal_training_executed", "training_ready", "independent_audit_pass"}, field)
    for name in value:
        _bool(value[name], f"{field}.{name}")
    if value["gpu_probe_executed"] or value["formal_training_executed"] or value["training_ready"] or value["independent_audit_pass"]:
        _fail(f"{field} is not fail-closed")
    return value


def _result_digest(value: Mapping[str, Any]) -> str:
    return _digest(_without(value, "result_sha256", "result_file"))


def validate_v2a_immediate_launch_result(
    result: Mapping[str, Any],
    plan: Mapping[str, Any],
    receipt: Mapping[str, Any],
    authorization: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate TMUX_ACCEPTED as a request result, never as training readiness."""

    value = _exact(result, RESULT_KEYS, "T7E immediate launch result")
    _assert_builtin(value, "T7E immediate launch result")
    if authorization is None:
        authorization = plan.get("_authorization_for_validation") if type(plan) is dict else None
    if type(authorization) is not dict:
        _fail("T7E result validation requires the bound authorization")
    checked_plan = validate_v2a_launch_plan(plan, authorization)
    checked_receipt = validate_v2a_consumption_receipt(receipt, checked_plan, authorization)
    if value["schema_version"] != SCHEMA_VERSION or value["kind"] != RESULT_KIND or value["launch_id"] != LAUNCH_ID:
        _fail("T7E immediate result identity drift")
    if value["status"] not in {TMUX_ACCEPTED, PERMANENT_FAIL} or value["classification"] != value["status"]:
        _fail("T7E immediate result status drift")
    for name in ("authorization_id", "run_id", "nonce", "repo_root", "tmux_session_name"):
        _string(value[name], f"result.{name}")
    if value["authorization_id"] != checked_receipt["authorization_id"] or value["run_id"] != checked_receipt["run_id"] or value["nonce"] != checked_receipt["nonce"]:
        _fail("T7E immediate result run drift")
    for name, expected in (("plan_sha256", checked_receipt["plan_sha256"]), ("receipt_sha256", checked_receipt["receipt_sha256"]), ("descriptor_sha256", checked_receipt["descriptor_sha256"]), ("authorization_context_sha256", checked_receipt["authorization_context_sha256"])):
        _sha(value[name], f"result.{name}")
        if value[name] != expected:
            _fail(f"T7E result {name} drift")
    if value["tmux_session_name"] != checked_plan["tmux_session_name"] or value["production_argv"] != checked_plan["production_argv"]:
        _fail("T7E result launch binding drift")
    if value["tmux_new_session_count"] != 1 or value["immediate_snapshot_count"] != 1:
        _fail("T7E exactly-once call count drift")
    _validate_launch_observation(value["launch_observation"])
    snapshot = _validate_snapshot(value["immediate_snapshot"])
    if snapshot["session_name"] != value["tmux_session_name"]:
        _fail("T7E snapshot session drift")
    if value["repo_root"] != checked_plan["repo_root"]:
        _fail("T7E result repository drift")
    for name in ("stdout_file", "stderr_file", "snapshot_file", "result_file"):
        _validate_file_ref(value[name], f"result.{name}", path=pathlib.Path(checked_plan["target_identity"][name.replace("_file", "_path")]))
    stdout_raw = pathlib.Path(value["stdout_file"]["path"]).read_bytes()
    stderr_raw = pathlib.Path(value["stderr_file"]["path"]).read_bytes()
    snapshot_raw = pathlib.Path(value["snapshot_file"]["path"]).read_bytes()
    if _sha_bytes(stdout_raw) != value["stdout_file"]["sha256"] or _sha_bytes(stderr_raw) != value["stderr_file"]["sha256"] or _sha_bytes(snapshot_raw) != value["snapshot_file"]["sha256"]:
        _fail("T7E raw evidence identity drift")
    for name in ("stdout_file", "stderr_file", "snapshot_file", "result_file"):
        observed = _file_identity(pathlib.Path(value[name]["path"]), f"result.{name}", mode=0o600)
        if any(observed[key] != value[name][key] for key in ("size_bytes", "sha256", "mode", "nlink")):
            _fail(f"T7E result.{name} metadata drift")
    try:
        snapshot_value = json.loads(snapshot_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V2ALauncherError("T7E snapshot is not JSON") from exc
    if snapshot_value != snapshot:
        _fail("T7E snapshot content drift")
    if value["launch_observation"]["stdout_size_bytes"] != len(stdout_raw) or value["launch_observation"]["stderr_size_bytes"] != len(stderr_raw):
        _fail("T7E stream size drift")
    if value["launch_observation"]["stdout_sha256"] != value["stdout_file"]["sha256"] or value["launch_observation"]["stderr_sha256"] != value["stderr_file"]["sha256"]:
        _fail("T7E stream digest drift")
    expected_status = TMUX_ACCEPTED if value["launch_observation"]["return_code"] == 0 and snapshot["return_code"] == 0 and snapshot["present"] is True else PERMANENT_FAIL
    if value["status"] != expected_status:
        _fail("T7E immediate classification drift")
    _validate_production_state(value["production"])
    if value["production"]["owner_authorization_consumed"] != (checked_receipt["mode"] == "production"):
        _fail("T7E result authorization consumption mode drift")
    if value["training_certified"] is not False:
        _fail("T7E TMUX acceptance was relabeled as training")
    _bool(value["training_certified"], "result.training_certified")
    counts = _exact(value["call_counts"], {"preflight", "persist", "consume", "launcher", "immediate_snapshot"}, "result.call_counts")
    for name in counts:
        if counts[name] != 1:
            _fail(f"T7E result call count drift: {name}")
    _sha(value["result_sha256"], "result.result_sha256")
    if value["result_sha256"] != _result_digest(value):
        _fail("T7E result digest drift")
    result_path = pathlib.Path(checked_plan["target_identity"]["result_path"])
    result_raw = result_path.read_bytes()
    base = _without(value, "result_file", "result_sha256")
    if result_raw != _canonical(base) or _sha_bytes(result_raw) != value["result_file"]["sha256"]:
        _fail("T7E result file content drift")
    return _copy(value)


def _observation_from_result(value: Any, field: str) -> tuple[dict[str, Any], bytes, bytes]:
    if type(value) is not dict:
        _fail(f"{field} must be a builtin dict")
    expected = {"return_code", "stdout", "stderr", "pid"}
    if set(value) != expected:
        _fail(f"{field} key set drift")
    _integer(value["return_code"], f"{field}.return_code")
    _integer(value["pid"], f"{field}.pid", minimum=0)
    if type(value["stdout"]) is not bytes or type(value["stderr"]) is not bytes:
        _fail(f"{field} streams must be bytes")
    observation = {
        "return_code": value["return_code"],
        "pid": value["pid"],
        "stdout_size_bytes": len(value["stdout"]),
        "stdout_sha256": _sha_bytes(value["stdout"]),
        "stderr_size_bytes": len(value["stderr"]),
        "stderr_sha256": _sha_bytes(value["stderr"]),
        "exception_type": None,
    }
    return observation, value["stdout"], value["stderr"]


def _snapshot_from_result(value: Any, session_name: str) -> tuple[dict[str, Any], bytes, bytes]:
    if type(value) is not dict or set(value) != {"return_code", "present", "stdout", "stderr"}:
        _fail("tmux immediate snapshot fake result shape drift")
    _integer(value["return_code"], "snapshot.return_code")
    _bool(value["present"], "snapshot.present")
    if type(value["stdout"]) is not bytes or type(value["stderr"]) is not bytes:
        _fail("snapshot streams must be bytes")
    snapshot = {
        "kind": SNAPSHOT_KIND,
        "session_name": session_name,
        "return_code": value["return_code"],
        "present": value["present"],
        "stdout_size_bytes": len(value["stdout"]),
        "stdout_sha256": _sha_bytes(value["stdout"]),
        "stderr_size_bytes": len(value["stderr"]),
        "stderr_sha256": _sha_bytes(value["stderr"]),
        "exception_type": None,
    }
    return snapshot, value["stdout"], value["stderr"]


def _build_and_publish_result(
    plan: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    authorization: Mapping[str, Any],
    launch_value: Mapping[str, Any],
    snapshot_value: Mapping[str, Any],
    stdout: bytes,
    stderr: bytes,
    snapshot_stdout: bytes,
    snapshot_stderr: bytes,
    production_mode: bool,
    counts: Mapping[str, int],
) -> dict[str, Any]:
    checked_plan = validate_v2a_launch_plan(plan, authorization)
    checked_receipt = validate_v2a_consumption_receipt(receipt, checked_plan, authorization)
    launch_observation, _, _ = _observation_from_result(launch_value, "launch result")
    _validate_snapshot(snapshot_value)
    snapshot_raw = _canonical(snapshot_value)
    target = checked_plan["target_identity"]
    stdout_identity = _write_new_bytes(pathlib.Path(target["stdout_path"]), stdout, "T7E stdout evidence")
    stderr_identity = _write_new_bytes(pathlib.Path(target["stderr_path"]), stderr, "T7E stderr evidence")
    del snapshot_stdout, snapshot_stderr
    snapshot_identity = _write_new_bytes(pathlib.Path(target["snapshot_path"]), snapshot_raw, "T7E immediate snapshot")
    def public_ref(path: pathlib.Path, identity: Mapping[str, Any]) -> dict[str, Any]:
        return {"path": str(path), "size_bytes": identity["size_bytes"], "sha256": identity["sha256"], "mode": identity["mode"], "nlink": identity["nlink"]}
    accepted = launch_observation["return_code"] == 0 and snapshot_value["return_code"] == 0 and snapshot_value["present"] is True
    base = {
        "schema_version": SCHEMA_VERSION,
        "kind": RESULT_KIND,
        "status": TMUX_ACCEPTED if accepted else PERMANENT_FAIL,
        "classification": TMUX_ACCEPTED if accepted else PERMANENT_FAIL,
        "launch_id": LAUNCH_ID,
        "authorization_id": checked_receipt["authorization_id"],
        "run_id": checked_receipt["run_id"],
        "nonce": checked_receipt["nonce"],
        "plan_sha256": checked_receipt["plan_sha256"],
        "receipt_sha256": checked_receipt["receipt_sha256"],
        "descriptor_sha256": checked_receipt["descriptor_sha256"],
        "authorization_context_sha256": checked_receipt["authorization_context_sha256"],
        "repo_root": checked_plan["repo_root"],
        "tmux_session_name": checked_plan["tmux_session_name"],
        "production_argv": _copy(checked_plan["production_argv"]),
        "tmux_new_session_count": 1,
        "immediate_snapshot_count": 1,
        "launch_observation": launch_observation,
        "immediate_snapshot": _copy(snapshot_value),
        "stdout_file": public_ref(pathlib.Path(target["stdout_path"]), stdout_identity),
        "stderr_file": public_ref(pathlib.Path(target["stderr_path"]), stderr_identity),
        "snapshot_file": public_ref(pathlib.Path(target["snapshot_path"]), snapshot_identity),
        "production": {
            "owner_authorization_consumed": production_mode,
            "gpu_probe_executed": False,
            "formal_training_executed": False,
            "training_ready": False,
            "independent_audit_pass": False,
        },
        "training_certified": False,
        "call_counts": dict(counts),
    }
    base["result_sha256"] = _result_digest(base)
    result_raw = _canonical(_without(base, "result_sha256"))
    result_identity = _write_new_bytes(pathlib.Path(target["result_path"]), result_raw, "T7E immediate launch result")
    result = {**base, "result_file": public_ref(pathlib.Path(target["result_path"]), result_identity)}
    return validate_v2a_immediate_launch_result(result, checked_plan, receipt, authorization)


def _default_fake_launch(argv: list[str], cwd: str, policy: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    del argv, cwd, policy, plan
    return {"return_code": 0, "stdout": b"fake tmux new-session accepted", "stderr": b"", "pid": 0}


def _default_fake_snapshot(session_name: str, plan: Mapping[str, Any]) -> dict[str, Any]:
    del plan
    return {"return_code": 0, "present": True, "stdout": session_name.encode("ascii"), "stderr": b""}


def run_v2a_cpu_fake_launch(
    authorization: Mapping[str, Any],
    plan: Mapping[str, Any],
    ports: Mapping[str, Any],
    *,
    t7d_policy: Mapping[str, Any] | None = None,
    observed_git_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Exercise exactly-once ordering with explicit fake launcher ports."""

    if type(ports) is not dict:
        _fail("CPU fake launcher ports must be a builtin dict")
    if any(key in ports for key in {"subprocess", "tmux", "real_data", "real_model", "production_capability", "authorization_path"}):
        _fail("CPU fake launcher cannot inject production capabilities or disk authorization")
    new_session = ports.get("new_session", _default_fake_launch)
    snapshot = ports.get("has_session", _default_fake_snapshot)
    if not callable(new_session) or not callable(snapshot):
        _fail("CPU fake launcher requires new_session and has_session callables")
    checked_auth, checked_plan = _preflight(authorization, plan, observed_git_identity=observed_git_identity, t7d_policy=t7d_policy)
    repo_root = pathlib.Path(checked_plan["repo_root"])
    control_root = pathlib.Path(checked_plan["target_identity"]["control_root"])
    if control_root == repo_root or repo_root in control_root.parents:
        _fail("CPU fake control root must be outside the repository")
    counts = {"preflight": 1, "persist": 0, "consume": 0, "launcher": 0, "immediate_snapshot": 0}
    target = checked_auth["target_identity"]
    plan_to_persist = _copy(checked_plan)
    _write_new_json(pathlib.Path(target["plan_path"]), plan_to_persist, "T7E launch plan")
    _write_new_json(pathlib.Path(target["descriptor_path"]), checked_auth["t7d_descriptor"], "T7E T7D descriptor")
    _write_new_json(pathlib.Path(target["authorization_context_path"]), checked_auth["t7d_authorization_context"], "T7E T7D authorization context")
    plan_value, _ = _read_canonical_json(pathlib.Path(target["plan_path"]), "T7E launch plan")
    descriptor_value, _ = _read_canonical_json(pathlib.Path(target["descriptor_path"]), "T7E T7D descriptor")
    context_value, _ = _read_canonical_json(pathlib.Path(target["authorization_context_path"]), "T7E T7D authorization context")
    if plan_value != checked_plan or descriptor_value != checked_auth["t7d_descriptor"] or context_value != checked_auth["t7d_authorization_context"]:
        _fail("T7E persisted descriptor/context readback drift")
    counts["persist"] = 1
    receipt = consume_v2a_authorization(checked_auth, checked_plan, mode="cpu_fake")
    counts["consume"] = 1
    launch_error = None
    try:
        launch_raw = new_session(checked_plan["production_argv"], checked_plan["repo_root"], checked_plan["host_gpu_policy"], checked_plan)
        launch_observation, launch_stdout, launch_stderr = _observation_from_result(launch_raw, "CPU fake launcher")
    except Exception as exc:
        launch_error = exc
        launch_observation = {"return_code": 255, "pid": 0, "stdout_size_bytes": 0, "stdout_sha256": _sha_bytes(b""), "stderr_size_bytes": 0, "stderr_sha256": _sha_bytes(b""), "exception_type": type(exc).__name__}
        launch_stdout = b""
        launch_stderr = b""
    del launch_observation
    counts["launcher"] = 1
    snapshot_error = None
    try:
        snapshot_raw = snapshot(checked_plan["tmux_session_name"], checked_plan)
        snapshot_value, snapshot_stdout, snapshot_stderr = _snapshot_from_result(snapshot_raw, checked_plan["tmux_session_name"])
    except Exception as exc:
        snapshot_error = exc
        snapshot_value = {"kind": SNAPSHOT_KIND, "session_name": checked_plan["tmux_session_name"], "return_code": 255, "present": False, "stdout_size_bytes": 0, "stdout_sha256": _sha_bytes(b""), "stderr_size_bytes": 0, "stderr_sha256": _sha_bytes(b""), "exception_type": type(exc).__name__}
        snapshot_stdout = b""
        snapshot_stderr = b""
    counts["immediate_snapshot"] = 1
    if launch_error is not None:
        launch_raw = {"return_code": 255, "stdout": launch_stdout, "stderr": launch_stderr, "pid": 0}
    if snapshot_error is not None:
        snapshot_raw = {"return_code": 255, "present": False, "stdout": snapshot_stdout, "stderr": snapshot_stderr}
    result = _build_and_publish_result(
        checked_plan,
        receipt,
        authorization=checked_auth,
        launch_value=launch_raw,
        snapshot_value=snapshot_value,
        stdout=launch_stdout,
        stderr=launch_stderr,
        snapshot_stdout=snapshot_stdout,
        snapshot_stderr=snapshot_stderr,
        production_mode=False,
        counts=counts,
    )
    return result


def _git_command(root: pathlib.Path, arguments: list[str]) -> str:
    try:
        completed = subprocess.run(["git", *arguments], cwd=str(root), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, shell=False)
    except OSError as exc:
        raise V2ALauncherError("Git observation failed") from exc
    if completed.returncode != 0:
        raise V2ALauncherError(f"Git observation failed: {' '.join(arguments)}")
    try:
        return completed.stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise V2ALauncherError("Git observation was not UTF-8") from exc


def _observe_git(root: pathlib.Path) -> dict[str, Any]:
    status = _git_command(root, ["status", "--porcelain=v1"])
    ignored = _git_command(root, ["status", "--ignored", "--porcelain=v1", "--untracked-files=all"])
    cache_counts = {"pycache": 0, "pyc": 0, "pytest_cache": 0}
    for line in ignored.splitlines():
        if "__pycache__" in line:
            cache_counts["pycache"] += 1
        if line.endswith(".pyc"):
            cache_counts["pyc"] += 1
        if ".pytest_cache" in line:
            cache_counts["pytest_cache"] += 1
    return {
        "branch": _git_command(root, ["rev-parse", "--abbrev-ref", "HEAD"]).rstrip("\n"),
        "head": _git_command(root, ["rev-parse", "HEAD"]).rstrip("\n"),
        "parent": _git_command(root, ["rev-parse", "HEAD^"]).rstrip("\n"),
        "tree": _git_command(root, ["rev-parse", "HEAD^{tree}"]).rstrip("\n"),
        "upstream": _git_command(root, ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"]).rstrip("\n"),
        "upstream_sha": _git_command(root, ["rev-parse", "@{upstream}"]).rstrip("\n"),
        "worktree_status": status,
        "cache_counts": cache_counts,
    }


def _owner_tmux_new_session(argv: list[str], root: str, policy: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    tmux = policy["tmux_path"]
    environment = os.environ.copy()
    environment.update({"PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": policy["cuda_visible_devices"], "PYTHONPATH": str(pathlib.Path(root) / "src")})
    command = [tmux, "new-session", "-d", "-s", plan["tmux_session_name"], "--", *argv]
    completed = subprocess.run(command, cwd=root, env=environment, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, shell=False)
    return {"return_code": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr, "pid": 0}


def _owner_tmux_snapshot(session_name: str, plan: Mapping[str, Any]) -> dict[str, Any]:
    policy = plan["host_gpu_policy"]
    completed = subprocess.run([policy["tmux_path"], "has-session", "-t", session_name], cwd=plan["repo_root"], env=os.environ.copy(), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, shell=False)
    return {"return_code": completed.returncode, "present": completed.returncode == 0, "stdout": completed.stdout, "stderr": completed.stderr}


def run_v2a_production_launch(authorization_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Consume one owner authorization and perform one tmux call plus snapshot."""

    path = _absolute_path(authorization_path, "authorization_path")
    authorization, _ = _read_canonical_json(path, "T7E owner authorization")
    checked_auth = validate_v2a_owner_authorization(authorization, expected_authorization_path=path)
    root = pathlib.Path(checked_auth["repo_root"])
    observed_git = _observe_git(root)
    if observed_git != checked_auth["git_identity"]:
        _fail("current Git identity does not match owner authorization")
    flat_roots = {role: item["root"] for role, item in checked_auth["t7d_policy"]["data_roots"]["roles"].items()}
    current_policy = _production.build_v2a_production_policy(
        root,
        data_roots=flat_roots,
        environment_identity=checked_auth["t7d_descriptor"]["environment_identity"],
        target_root=checked_auth["t7d_descriptor"]["target_root"],
    )
    _validate_t7d_policy_snapshot(current_policy, checked_auth["t7d_descriptor"], checked_auth["t7d_authorization_context"])
    if current_policy["policy_sha256"] != checked_auth["t7d_policy_sha256"]:
        _fail("current T7D policy does not match owner authorization")
    plan = build_v2a_launch_plan(checked_auth, t7d_policy=current_policy)
    checked_auth, checked_plan = _preflight(checked_auth, plan, observed_git_identity=observed_git, t7d_policy=current_policy)
    target = checked_auth["target_identity"]
    _write_new_json(pathlib.Path(target["plan_path"]), checked_plan, "T7E launch plan")
    _write_new_json(pathlib.Path(target["descriptor_path"]), checked_auth["t7d_descriptor"], "T7E T7D descriptor")
    _write_new_json(pathlib.Path(target["authorization_context_path"]), checked_auth["t7d_authorization_context"], "T7E T7D authorization context")
    plan_value, _ = _read_canonical_json(pathlib.Path(target["plan_path"]), "T7E launch plan")
    descriptor_value, _ = _read_canonical_json(pathlib.Path(target["descriptor_path"]), "T7E T7D descriptor")
    context_value, _ = _read_canonical_json(pathlib.Path(target["authorization_context_path"]), "T7E T7D authorization context")
    if plan_value != checked_plan or descriptor_value != checked_auth["t7d_descriptor"] or context_value != checked_auth["t7d_authorization_context"]:
        _fail("T7E persisted descriptor/context readback drift")
    receipt = consume_v2a_authorization(checked_auth, checked_plan, mode="production", owner_file_identity=_authorization_file_identity(path))
    launch_error = None
    try:
        launch_raw = _owner_tmux_new_session(checked_plan["production_argv"], checked_plan["repo_root"], checked_plan["host_gpu_policy"], checked_plan)
    except Exception as exc:
        launch_error = exc
        launch_raw = {"return_code": 255, "stdout": b"", "stderr": b"", "pid": 0}
    try:
        snapshot_raw = _owner_tmux_snapshot(checked_plan["tmux_session_name"], checked_plan)
        snapshot_value, snapshot_stdout, snapshot_stderr = _snapshot_from_result(snapshot_raw, checked_plan["tmux_session_name"])
    except Exception as exc:
        snapshot_value = {"kind": SNAPSHOT_KIND, "session_name": checked_plan["tmux_session_name"], "return_code": 255, "present": False, "stdout_size_bytes": 0, "stdout_sha256": _sha_bytes(b""), "stderr_size_bytes": 0, "stderr_sha256": _sha_bytes(b""), "exception_type": type(exc).__name__}
        snapshot_stdout = b""
        snapshot_stderr = b""
    del launch_error
    return _build_and_publish_result(
        checked_plan,
        receipt,
        authorization=checked_auth,
        launch_value=launch_raw,
        snapshot_value=snapshot_value,
        stdout=launch_raw["stdout"],
        stderr=launch_raw["stderr"],
        snapshot_stdout=snapshot_stdout,
        snapshot_stderr=snapshot_stderr,
        production_mode=True,
        counts={"preflight": 1, "persist": 1, "consume": 1, "launcher": 1, "immediate_snapshot": 1},
    )
