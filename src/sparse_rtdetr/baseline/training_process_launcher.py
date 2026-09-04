"""Synthetic-only outer process boundary for the formal T5D contract.

The module deliberately has no subprocess, shell, terminal multiplexer, model,
data-loader, or CUDA path.  Its only executable path is an injected synchronous
fake runner.  Synthetic evidence is published through directory descriptors so
that a path replacement cannot turn a successful result into an authority.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import os
import stat
from pathlib import Path
from typing import Any, Callable, Mapping

from sparse_rtdetr.baseline import training_entry as _entry


LAUNCHER_SCHEMA_VERSION = 1
PROCESS_CONTRACT_ID = _entry.PROCESS_CONTRACT_ID
LAUNCHER_ID = "rtdetrv2_r18_visdrone_training_process_launcher_t5d_v1"
LAUNCHER_MODE = "synthetic"
LAUNCHER_MODULE_RELATIVE_PATH = "src/sparse_rtdetr/baseline/training_process_launcher.py"
LAUNCH_RECEIPT_SUFFIX = ".t5d-launch-receipt.json"
STATE_SEQUENCE = (
    "ABSENT",
    "PREPARED",
    "ACCEPTED",
    "SYNTHETIC_TERMINAL_COMPLETE",
    "TERMINAL_COMPLETE",
    "TERMINAL_FAILED",
    "UNKNOWN",
)
RUNNER_PARAMETERS = ("argv", "cwd", "environment", "entry_descriptor")
RUNNER_RESULT_KEYS = {"return_code", "stdout", "stderr", "entry_result"}
JSON_EVIDENCE_FILE_NAMES = ("invocation.json", "result.json", "completion.json", "artifact_inventory.json")
BINARY_ANCHOR_FILE_NAMES = ("stdout.bin", "stderr.bin")
EVIDENCE_FILE_NAMES = JSON_EVIDENCE_FILE_NAMES + BINARY_ANCHOR_FILE_NAMES
INVENTORY_EXCLUDED = {"artifact_inventory.json", "completion.json"}
FAILURE_CLASSES = (
    "NONE",
    "RUNNER_NONZERO_RETURN_CODE",
    "RUNNER_EXCEPTION",
    "RUNNER_TIMEOUT",
    "RUNNER_ASYNC_OR_GENERATOR_RESULT",
    "RUNNER_SCHEMA_OR_ENTRY_FAILURE",
    "RUNNER_MUTATED_INVOCATION",
    "PERSISTENCE_FAILURE",
)
RETURN_CODE_MIN = 0
RETURN_CODE_MAX = 255
SAFE_FAILURE_RETURN_CODE = 2
FAILURE_RETURN_CODES = {
    "RUNNER_EXCEPTION": 2,
    "RUNNER_TIMEOUT": 3,
    "RUNNER_ASYNC_OR_GENERATOR_RESULT": 4,
    "RUNNER_SCHEMA_OR_ENTRY_FAILURE": 5,
    "RUNNER_MUTATED_INVOCATION": 6,
}

DESCRIPTOR_KEYS = {
    "schema_version",
    "process_contract_id",
    "launcher_id",
    "mode",
    "run_id",
    "nonce",
    "entry_descriptor",
    "entry_descriptor_sha256",
    "argv",
    "cwd",
    "environment",
    "config_identity",
    "source_bindings",
    "future_targets",
    "evidence_root",
    "receipt_path",
    "state_machine",
    "runner_contract",
    "aggregate_sha256",
}
RESULT_KEYS = {
    "schema_version",
    "process_contract_id",
    "launcher_id",
    "mode",
    "run_id",
    "nonce",
    "launch_descriptor_sha256",
    "entry_descriptor_sha256",
    "argv",
    "cwd",
    "environment",
    "config_identity",
    "source_bindings",
    "future_targets",
    "evidence_root",
    "receipt_path",
    "evidence_root_identity",
    "receipt_size_bytes",
    "receipt_sha256",
    "return_code",
    "stdout_size_bytes",
    "stdout_sha256",
    "stderr_size_bytes",
    "stderr_sha256",
    "stdout_anchor_relative_path",
    "stderr_anchor_relative_path",
    "entry_result",
    "entry_result_sha256",
    "status",
    "classification",
    "failure_class",
    "state_sequence",
    "state_transition_sha256",
    "production_training_authorized",
    "real_process_launch_authorized",
    "aggregate_result_sha256",
}
INVOCATION_KEYS = {"schema_version", "status", "descriptor_sha256", "descriptor"}
INVENTORY_KEYS = {"schema_version", "excluded", "files", "canonical_inventory_sha256"}
COMPLETION_KEYS = {
    "schema_version",
    "status",
    "classification",
    "exit_code",
    "descriptor_sha256",
    "result_sha256",
    "inventory_sha256",
    "stdout_size_bytes",
    "stdout_sha256",
    "stderr_size_bytes",
    "stderr_sha256",
    "entry_result_sha256",
    "state_transition_sha256",
    "evidence_root_identity",
    "production_training_authorized",
}

__all__ = (
    "TrainingProcessLauncherError",
    "build_synthetic_launch_descriptor",
    "validate_launch_descriptor",
    "run_synthetic_launch",
    "validate_launch_result",
    "classify_launch",
)


class TrainingProcessLauncherError(ValueError):
    """Raised when the process boundary or durable evidence drifts."""


class _RunnerAsyncResult(TrainingProcessLauncherError):
    pass


class _RunnerSchemaFailure(TrainingProcessLauncherError):
    pass


class _RunnerInputMutation(TrainingProcessLauncherError):
    pass


def _fail(message: str) -> None:
    raise TrainingProcessLauncherError(message)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _assert_builtin_json(value: Any, field: str = "value") -> None:
    if type(value) is dict:
        for key, child in value.items():
            if type(key) is not str:
                _fail(f"{field} has a non-string key")
            _assert_builtin_json(child, f"{field}.{key}")
        return
    if type(value) is list:
        for index, child in enumerate(value):
            _assert_builtin_json(child, f"{field}[{index}]")
        return
    if type(value) is float and not math.isfinite(value):
        _fail(f"{field} is non-finite")
    if type(value) not in {str, int, float, bool, type(None)}:
        _fail(f"{field} is not builtin JSON")


def _canonical(value: Any) -> bytes:
    _assert_builtin_json(value, "canonical value")
    try:
        return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise TrainingProcessLauncherError("value is not canonical JSON") from exc


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


def _absolute_path(value: Any, field: str) -> str:
    try:
        return _entry._absolute_path(value, field)
    except _entry.TrainingEntryError as exc:
        raise TrainingProcessLauncherError(str(exc)) from exc


def _relative_path(value: Any, field: str) -> str:
    value = _string(value, field)
    if value.startswith("/") or "\x00" in value or "\\" in value or "//" in value or value.endswith("/"):
        _fail(f"{field} is not a normalized relative path")
    if any(part in {"", ".", ".."} for part in value.split("/")):
        _fail(f"{field} contains traversal or empty components")
    return value


def _validate_config_identity(value: Any, field: str) -> dict[str, Any]:
    identity = _exact(value, {"relative_path", "raw_size_bytes", "raw_sha256", "canonical_size_bytes", "canonical_sha256", "mode"}, field)
    _relative_path(identity["relative_path"], f"{field}.relative_path")
    for name in ("raw_size_bytes", "canonical_size_bytes"):
        _integer(identity[name], f"{field}.{name}", minimum=1)
    for name in ("raw_sha256", "canonical_sha256"):
        _sha_string(identity[name], f"{field}.{name}")
    _integer(identity["mode"], f"{field}.mode", minimum=1)
    if identity["mode"] not in {0o644, 0o664}:
        _fail(f"{field}.mode drift")
    return _copy(identity)


def _state_trace(states: list[str]) -> list[dict[str, Any]]:
    if not states or states[0] != "ABSENT":
        _fail("state trace must begin at ABSENT")
    trace: list[dict[str, Any]] = [{"sequence": 0, "state": "ABSENT", "predecessor_sha256": None}]
    for index, state in enumerate(states[1:], 1):
        if state not in STATE_SEQUENCE or state == "ABSENT":
            _fail("unknown state in trace")
        previous = trace[-1]
        trace.append({"sequence": index, "state": state, "predecessor_sha256": _digest(previous)})
    return trace


def _validate_module_identity(value: Any, field: str, expected_path: str) -> dict[str, Any]:
    row = _exact(value, {"relative_path", "size_bytes", "sha256", "mode"}, field)
    if row["relative_path"] != expected_path:
        _fail(f"{field} path drift")
    _integer(row["size_bytes"], f"{field}.size_bytes", minimum=1)
    _sha_string(row["sha256"], f"{field}.sha256")
    _integer(row["mode"], f"{field}.mode", minimum=1)
    return _copy(row)


def _validate_descriptor_shape(value: Any) -> dict[str, Any]:
    _assert_builtin_json(value, "launch descriptor")
    descriptor = _exact(value, DESCRIPTOR_KEYS, "launch descriptor")
    if (
        _integer(descriptor["schema_version"], "descriptor.schema_version", minimum=1) != LAUNCHER_SCHEMA_VERSION
        or descriptor["process_contract_id"] != PROCESS_CONTRACT_ID
        or descriptor["launcher_id"] != LAUNCHER_ID
        or descriptor["mode"] != LAUNCHER_MODE
    ):
        _fail("launch descriptor identity drift")
    _string(descriptor["run_id"], "descriptor.run_id")
    _string(descriptor["nonce"], "descriptor.nonce")
    entry = _entry.validate_entry_descriptor(descriptor["entry_descriptor"])
    entry_digest = _sha_string(descriptor["entry_descriptor_sha256"], "descriptor.entry_descriptor_sha256")
    if entry_digest != entry["aggregate_sha256"]:
        _fail("entry descriptor digest drift")
    if descriptor["run_id"] != entry["run_id"] or descriptor["nonce"] != entry["nonce"]:
        _fail("launch and entry run identity drift")
    argv = descriptor["argv"]
    if type(argv) is not list or any(type(item) is not str for item in argv) or argv != entry["argv"]:
        _fail("launch argv is not the detached entry argv sequence")
    cwd = _absolute_path(descriptor["cwd"], "descriptor.cwd")
    if cwd != entry["working_directory"]:
        _fail("launch cwd drift")
    environment = descriptor["environment"]
    if type(environment) is not dict:
        _fail("launch environment is not a builtin dict")
    _strict_equal(environment, entry["environment"], "launch environment")
    config = _validate_config_identity(descriptor["config_identity"], "descriptor.config_identity")
    _strict_equal(config, entry["source_bindings"]["process_config"], "launch process config identity")
    source = _exact(descriptor["source_bindings"], {"entry_module", "launcher_module", "repository_modules"}, "descriptor.source_bindings")
    _validate_module_identity(source["entry_module"], "descriptor.source_bindings.entry_module", _entry.ENTRY_MODULE_RELATIVE_PATH)
    _validate_module_identity(source["launcher_module"], "descriptor.source_bindings.launcher_module", LAUNCHER_MODULE_RELATIVE_PATH)
    modules = source["repository_modules"]
    if type(modules) is not list or len(modules) != len(_entry.SOURCE_MODULES):
        _fail("launch source module inventory length drift")
    for index, row in enumerate(modules):
        row = _exact(row, {"role", "relative_path", "size_bytes", "sha256", "mode"}, f"repository_modules[{index}]")
        _string(row["role"], f"repository_modules[{index}].role")
        _relative_path(row["relative_path"], f"repository_modules[{index}].relative_path")
        _integer(row["size_bytes"], f"repository_modules[{index}].size_bytes", minimum=1)
        _sha_string(row["sha256"], f"repository_modules[{index}].sha256")
        _integer(row["mode"], f"repository_modules[{index}].mode", minimum=1)
    expected_modules = entry["source_bindings"]["repository_modules"]
    _strict_equal(modules, expected_modules, "launch repository module identity")
    future = _exact(descriptor["future_targets"], set(_entry.FUTURE_TARGETS), "descriptor.future_targets")
    _strict_equal(future, _entry.FUTURE_TARGETS, "launch future target identity")
    evidence_root = _absolute_path(descriptor["evidence_root"], "descriptor.evidence_root")
    _entry._reject_production_root(evidence_root, "descriptor.evidence_root")
    expected_receipt = str(Path(evidence_root).parent / _entry._receipt_name(evidence_root, LAUNCH_RECEIPT_SUFFIX))
    if descriptor["receipt_path"] != expected_receipt:
        _fail("launch receipt path drift")
    parent, name = _entry._bound_target_parts(evidence_root, "descriptor.evidence_root")
    try:
        if not name:
            _fail("launch evidence root final component drift")
        parent.revalidate()
    finally:
        parent.close()
    machine = _exact(descriptor["state_machine"], {"states", "trace", "exactly_once", "durable_evidence_required"}, "descriptor.state_machine")
    if machine["states"] != list(STATE_SEQUENCE) or machine["exactly_once"] is not True or machine["durable_evidence_required"] is not True:
        _fail("launch state-machine policy drift")
    trace = machine["trace"]
    if type(trace) is not list or len(trace) != 2 or [row.get("state") for row in trace] != ["ABSENT", "PREPARED"]:
        _fail("launch descriptor state trace drift")
    for index, row in enumerate(trace):
        row = _exact(row, {"sequence", "state", "predecessor_sha256"}, f"descriptor.state_machine.trace[{index}]")
        if row["sequence"] != index or (index == 0 and row["predecessor_sha256"] is not None) or (index == 1 and row["predecessor_sha256"] != _digest(trace[0])):
            _fail("launch descriptor predecessor chain drift")
    runner = _exact(descriptor["runner_contract"], {"parameters", "return_keys", "synchronous", "max_calls", "argv_must_be_sequence"}, "descriptor.runner_contract")
    if runner["parameters"] != list(RUNNER_PARAMETERS) or runner["return_keys"] != sorted(RUNNER_RESULT_KEYS) or runner["synchronous"] is not True or runner["max_calls"] != 1 or runner["argv_must_be_sequence"] is not True:
        _fail("launch runner contract drift")
    aggregate = _sha_string(descriptor["aggregate_sha256"], "descriptor.aggregate_sha256")
    body = _copy(descriptor)
    del body["aggregate_sha256"]
    if _digest(body) != aggregate:
        _fail("launch descriptor aggregate identity drift")
    return _copy(descriptor)


def _actual_module_identity(root: Path, role: str, relative: str) -> dict[str, Any]:
    try:
        raw, observed = _entry._read_relative_file(root, relative, f"{role} module")
    except _entry.TrainingEntryError as exc:
        raise TrainingProcessLauncherError(str(exc)) from exc
    if stat.S_IMODE(observed.st_mode) == 0 or observed.st_nlink != 1:
        _fail(f"{role} module metadata drift")
    return {"role": role, "relative_path": relative, "size_bytes": len(raw), "sha256": _sha(raw), "mode": stat.S_IMODE(observed.st_mode)}


def _current_source_bindings(root: Path) -> dict[str, Any]:
    entry_identity = _actual_module_identity(root, "entry", _entry.ENTRY_MODULE_RELATIVE_PATH)
    launcher_identity = _actual_module_identity(root, "launcher", LAUNCHER_MODULE_RELATIVE_PATH)
    source_modules = [_actual_module_identity(root, role, relative) for role, relative in _entry.SOURCE_MODULES]
    return {
        "entry_module": {key: entry_identity[key] for key in ("relative_path", "size_bytes", "sha256", "mode")},
        "launcher_module": {key: launcher_identity[key] for key in ("relative_path", "size_bytes", "sha256", "mode")},
        "repository_modules": [
            {key: row[key] for key in ("role", "relative_path", "size_bytes", "sha256", "mode")}
            for row in source_modules
        ],
    }


def build_synthetic_launch_descriptor(entry_descriptor: Any, evidence_root: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Build one detached outer descriptor around a validated T5D entry."""

    entry = _entry.validate_entry_descriptor(entry_descriptor)
    root = Path(entry["repo_root"])
    if evidence_root is None:
        evidence_root = root.parent / f".t5d-launch-{entry['run_id']}"
    bound_root = _absolute_path(os.fspath(evidence_root), "evidence_root")
    _entry._validate_bound_root_shape(bound_root, "evidence_root", LAUNCH_RECEIPT_SUFFIX, require_absent=True)
    source = _current_source_bindings(root)
    body = {
        "schema_version": LAUNCHER_SCHEMA_VERSION,
        "process_contract_id": PROCESS_CONTRACT_ID,
        "launcher_id": LAUNCHER_ID,
        "mode": LAUNCHER_MODE,
        "run_id": entry["run_id"],
        "nonce": entry["nonce"],
        "entry_descriptor": _copy(entry),
        "entry_descriptor_sha256": entry["aggregate_sha256"],
        "argv": list(entry["argv"]),
        "cwd": entry["working_directory"],
        "environment": _copy(entry["environment"]),
        "config_identity": _copy(entry["source_bindings"]["process_config"]),
        "source_bindings": source,
        "future_targets": _copy(_entry.FUTURE_TARGETS),
        "evidence_root": bound_root,
        "receipt_path": str(Path(bound_root).parent / _entry._receipt_name(bound_root, LAUNCH_RECEIPT_SUFFIX)),
        "state_machine": {
            "states": list(STATE_SEQUENCE),
            "trace": _state_trace(["ABSENT", "PREPARED"]),
            "exactly_once": True,
            "durable_evidence_required": True,
        },
        "runner_contract": {
            "parameters": list(RUNNER_PARAMETERS),
            "return_keys": sorted(RUNNER_RESULT_KEYS),
            "synchronous": True,
            "max_calls": 1,
            "argv_must_be_sequence": True,
        },
    }
    return _validate_descriptor_shape({**body, "aggregate_sha256": _digest(body)})


def validate_launch_descriptor(value: Any) -> dict[str, Any]:
    """Validate a descriptor and reconcile it with the current source closure."""

    descriptor = _validate_descriptor_shape(value)
    current = _current_source_bindings(Path(descriptor["entry_descriptor"]["repo_root"]))
    _strict_equal(descriptor["source_bindings"], current, "launch source closure")
    return _copy(descriptor)


def _validate_runner(runner: Any) -> None:
    call = getattr(runner, "__call__", None)
    if not callable(runner) or inspect.iscoroutinefunction(runner) or inspect.isgeneratorfunction(runner) or inspect.iscoroutinefunction(call) or inspect.isgeneratorfunction(call):
        _fail("fake runner must be a synchronous callable")
    try:
        signature = inspect.signature(runner)
    except (TypeError, ValueError) as exc:
        raise TrainingProcessLauncherError("fake runner signature is unavailable") from exc
    parameters = list(signature.parameters.values())
    if [parameter.name for parameter in parameters] != list(RUNNER_PARAMETERS) or any(parameter.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD for parameter in parameters) or any(parameter.default is not inspect.Parameter.empty for parameter in parameters):
        _fail("fake runner signature drift")


def _assert_runner_inputs_unchanged(
    argv: Any,
    cwd: Any,
    environment: Any,
    entry_descriptor: Any,
    before: tuple[Any, Any, Any, Any],
) -> None:
    try:
        _strict_equal(argv, before[0], "runner argv")
        _strict_equal(cwd, before[1], "runner cwd")
        _strict_equal(environment, before[2], "runner environment")
        _strict_equal(entry_descriptor, before[3], "runner entry descriptor")
    except Exception as exc:
        raise _RunnerInputMutation("fake runner mutated its invocation inputs") from exc


def _entry_receipt_name(root: str) -> str:
    return _entry._receipt_name(root, LAUNCH_RECEIPT_SUFFIX)


def _launch_receipt_payload(descriptor: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": LAUNCHER_SCHEMA_VERSION,
        "kind": "T5D_LAUNCH_CLAIM",
        "descriptor_sha256": descriptor["aggregate_sha256"],
        "run_id": descriptor["run_id"],
        "nonce": descriptor["nonce"],
        "evidence_root": descriptor["evidence_root"],
        "receipt_path": descriptor["receipt_path"],
        "status": "CLAIMED",
    }


def _claim_launch_receipt(descriptor: dict[str, Any]) -> tuple[_entry._DirectoryLease, str, os.stat_result]:
    parent, root_name = _entry._bound_target_parts(descriptor["evidence_root"], "launch evidence root")
    receipt_name = Path(descriptor["receipt_path"]).name
    try:
        if receipt_name != _entry_receipt_name(descriptor["evidence_root"]):
            _fail("launch receipt path drift")
        _entry._require_absent(parent.fd, root_name, "launch evidence root")
        _entry._require_absent(parent.fd, receipt_name, "launch receipt")
        receipt_stat = _entry._create_exclusive_file(parent.fd, receipt_name, _canonical(_launch_receipt_payload(descriptor)), "launch receipt")
        current = _entry._stat_at(parent.fd, receipt_name)
        if current is None or not _entry._same_file_identity(receipt_stat, current):
            _fail("launch receipt path identity drift")
        _entry._fsync(parent.fd)
        _entry._require_absent(parent.fd, root_name, "launch evidence root")
        parent.revalidate()
        return parent, receipt_name, receipt_stat
    except _entry.TrainingEntryError as exc:
        parent.close()
        raise TrainingProcessLauncherError(str(exc)) from exc
    except Exception:
        parent.close()
        raise


def _verify_launch_receipt_at(parent: _entry._DirectoryLease, descriptor: dict[str, Any], name: str) -> os.stat_result:
    fd: int | None = None
    try:
        if name != _entry_receipt_name(descriptor["evidence_root"]):
            _fail("launch receipt path drift")
        expected = _canonical(_launch_receipt_payload(descriptor))
        fd = _entry._open_fd(name, _entry._file_flags(), dir_fd=parent.fd)
        observed = _entry._fstat(fd)
        _entry._require_regular_file(observed, "launch receipt")
        if stat.S_IMODE(observed.st_mode) != 0o600 or observed.st_size != len(expected):
            _fail("launch receipt metadata drift")
        while True:
            try:
                os.lseek(fd, 0, os.SEEK_SET)
                break
            except InterruptedError:
                continue
        if _entry._read_fd_exact(fd, observed.st_size, "launch receipt") != expected:
            _fail("launch receipt bytes drift")
        after = _entry._fstat(fd)
        if not _entry._same_file_identity(observed, after):
            _fail("launch receipt metadata drift")
        current = _entry._stat_at(parent.fd, name)
        if current is None or not _entry._same_file_identity(observed, current):
            _fail("launch receipt path identity drift")
        parent.revalidate()
        return after
    except OSError as exc:
        raise TrainingProcessLauncherError("launch receipt could not be verified") from exc
    except _entry.TrainingEntryError as exc:
        raise TrainingProcessLauncherError(str(exc)) from exc
    finally:
        if fd is not None:
            _entry._close_fd(fd)


def _verify_launch_receipt(descriptor: dict[str, Any]) -> os.stat_result:
    parent, name = _entry._bound_target_parts(descriptor["receipt_path"], "launch receipt")
    try:
        expected_name = _entry_receipt_name(descriptor["evidence_root"])
        if name != expected_name:
            _fail("launch receipt path drift")
        return _verify_launch_receipt_at(parent, descriptor, name)
    finally:
        parent.close()


def _open_root_lease(root: str) -> _entry._DirectoryLease:
    try:
        lease = _entry._open_directory_chain(root)
        observed = lease.snapshot
        if not stat.S_ISDIR(observed.st_mode) or observed.st_uid != os.getuid() or observed.st_gid != os.getgid() or stat.S_IMODE(observed.st_mode) != 0o700 or observed.st_nlink < 2:
            lease.close()
            _fail("launch evidence root metadata drift")
        return lease
    except _entry.TrainingEntryError as exc:
        raise TrainingProcessLauncherError(str(exc)) from exc


def _create_root(parent: _entry._DirectoryLease, name: str) -> os.stat_result:
    created = _entry._mkdir_exclusive(parent.fd, name, "launch evidence root")
    _entry._fsync(parent.fd)
    return created


def _open_root_from_parent(
    parent: _entry._DirectoryLease,
    name: str,
    expected: os.stat_result | None = None,
) -> _entry._DirectoryLease:
    """Open the created root from the same held parent used for mkdir-at."""

    fd: int | None = None
    lease: _entry._DirectoryLease | None = None
    try:
        fd, observed = _entry._open_created_root(parent, name, "launch evidence root", expected=expected)
        lease = _entry._DirectoryLease([fd], [observed])
        fd = None
        _assert_root_binding(parent, name, lease)
        return lease
    except _entry.TrainingEntryError as exc:
        if lease is not None:
            lease.close()
        raise TrainingProcessLauncherError(str(exc)) from exc
    except Exception:
        if lease is not None:
            lease.close()
        raise
    finally:
        if fd is not None:
            _entry._close_fd(fd)


def _assert_root_binding(parent: _entry._DirectoryLease, name: str, root: _entry._DirectoryLease) -> None:
    parent.revalidate()
    root.revalidate()
    named = _entry._stat_at(parent.fd, name)
    observed = _entry._fstat(root.fd)
    if named is None or not stat.S_ISDIR(named.st_mode) or not stat.S_ISDIR(observed.st_mode):
        _fail("launch evidence root path binding drift")
    fields = ("st_dev", "st_ino", "st_mode", "st_uid", "st_gid", "st_nlink")
    if any(getattr(named, field) != getattr(observed, field) for field in fields):
        _fail("launch evidence root path binding drift")


def _read_file_at(lease: _entry._DirectoryLease, name: str, field: str, *, expected_mode: int = 0o600) -> tuple[bytes, os.stat_result]:
    _relative_path(name, field)
    if "/" in name:
        _fail(f"{field} must be a direct evidence child")
    fd: int | None = None
    try:
        lease.revalidate(include_nlink=True)
        fd = _entry._open_fd(name, _entry._file_flags(), dir_fd=lease.fd)
        before = _entry._fstat(fd)
        _entry._require_regular_file(before, field)
        if stat.S_IMODE(before.st_mode) != expected_mode or before.st_dev != lease.snapshot.st_dev:
            _fail(f"{field} metadata drift")
        current_before = _entry._stat_at(lease.fd, name)
        if current_before is None or not _entry._same_file_identity(before, current_before):
            _fail(f"{field} path identity drift")
        while True:
            try:
                os.lseek(fd, 0, os.SEEK_SET)
                break
            except InterruptedError:
                continue
        payload = _entry._read_fd_exact(fd, before.st_size, field)
        after = _entry._fstat(fd)
        _entry._require_regular_file(after, field)
        if not _entry._same_file_identity(before, after):
            _fail(f"{field} metadata drift")
        current_after = _entry._stat_at(lease.fd, name)
        if current_after is None or not _entry._same_file_identity(after, current_after):
            _fail(f"{field} path identity drift")
        lease.revalidate(include_nlink=True)
        return payload, after
    except OSError as exc:
        raise TrainingProcessLauncherError(f"{field} is unreadable") from exc
    except _entry.TrainingEntryError as exc:
        raise TrainingProcessLauncherError(str(exc)) from exc
    finally:
        if fd is not None:
            _entry._close_fd(fd)


def _parse_json(raw: bytes, field: str) -> dict[str, Any]:
    if not raw or b"\r" in raw or b"\x00" in raw or raw.startswith(b"\xef\xbb\xbf"):
        _fail(f"{field} has non-portable bytes")

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                _fail(f"{field} contains duplicate keys")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=lambda token: _fail(f"{field} has {token}"))
    except TrainingProcessLauncherError:
        raise
    except (UnicodeDecodeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise TrainingProcessLauncherError(f"{field} is invalid JSON") from exc
    _assert_builtin_json(value, field)
    if type(value) is not dict or raw != _canonical(value):
        _fail(f"{field} is repacked")
    return value


def _read_json_at(lease: _entry._DirectoryLease, name: str) -> tuple[dict[str, Any], bytes]:
    raw, _ = _read_file_at(lease, name, name)
    return _parse_json(raw, name), raw


def _link(src: str, dst: str, directory_fd: int) -> None:
    while True:
        try:
            os.link(src, dst, src_dir_fd=directory_fd, dst_dir_fd=directory_fd, follow_symlinks=False)
            return
        except InterruptedError:
            continue


def _unlink(name: str, directory_fd: int) -> None:
    while True:
        try:
            os.unlink(name, dir_fd=directory_fd)
            return
        except InterruptedError:
            continue


def _publish_at(lease: _entry._DirectoryLease, name: str, payload: bytes) -> None:
    if name not in EVIDENCE_FILE_NAMES:
        _fail("unexpected evidence filename")
    if type(payload) is not bytes:
        _fail("evidence payload must be exact bytes")
    temporary = f".{name}.tmp"
    temp_stat: os.stat_result | None = None
    linked = False
    try:
        lease.revalidate()
        _entry._require_absent(lease.fd, name, name)
        _entry._require_absent(lease.fd, temporary, f"{name} temporary")
        temp_stat = _entry._create_exclusive_file(lease.fd, temporary, payload, f"{name} temporary")
        _link(temporary, name, lease.fd)
        linked = True
        linked_stat = _entry._stat_at(lease.fd, name)
        if linked_stat is None or linked_stat.st_dev != temp_stat.st_dev or linked_stat.st_ino != temp_stat.st_ino:
            _fail(f"{name} publication identity drift")
        _entry._fsync(lease.fd)
        _unlink(temporary, lease.fd)
        temp_stat = None
        _entry._fsync(lease.fd)
        actual, _ = _read_file_at(lease, name, name)
        if actual != payload or _sha(actual) != _sha(payload):
            _fail(f"{name} readback bytes drift")
        lease.revalidate()
    except OSError as exc:
        raise TrainingProcessLauncherError(f"{name} could not be durably published") from exc
    except _entry.TrainingEntryError as exc:
        raise TrainingProcessLauncherError(str(exc)) from exc
    finally:
        if temp_stat is not None:
            current = _entry._stat_at(lease.fd, temporary)
            if current is not None:
                try:
                    _unlink(temporary, lease.fd)
                except OSError:
                    pass
        if linked and temp_stat is not None:
            current = _entry._stat_at(lease.fd, name)
            if current is not None and current.st_dev == temp_stat.st_dev and current.st_ino == temp_stat.st_ino:
                try:
                    _unlink(name, lease.fd)
                except OSError:
                    pass


def _inventory_at(lease: _entry._DirectoryLease) -> list[dict[str, Any]]:
    try:
        names = set(_entry._listdir_fd(lease.fd))
    except (OSError, TrainingProcessLauncherError, _entry.TrainingEntryError) as exc:
        raise TrainingProcessLauncherError("launch inventory enumeration failed") from exc
    rows: list[dict[str, Any]] = []
    for name in sorted(names):
        if name in INVENTORY_EXCLUDED:
            continue
        if name not in EVIDENCE_FILE_NAMES:
            _fail("launch inventory contains an unexpected file")
        raw, _ = _read_file_at(lease, name, name)
        rows.append({"relative_path": name, "size_bytes": len(raw), "sha256": _sha(raw)})
    return rows


def _root_identity(observed: os.stat_result) -> dict[str, int]:
    return {
        "device": observed.st_dev,
        "inode": observed.st_ino,
        "mode": stat.S_IMODE(observed.st_mode),
        "uid": observed.st_uid,
        "gid": observed.st_gid,
        "nlink": observed.st_nlink,
        "size_bytes": observed.st_size,
        "mtime_ns": observed.st_mtime_ns,
        "ctime_ns": observed.st_ctime_ns,
    }


def _stable_root_identity(value: dict[str, int]) -> dict[str, int]:
    return {key: value[key] for key in ("device", "inode", "mode", "uid", "gid", "nlink")}


def _validate_invocation(value: Any) -> dict[str, Any]:
    invocation = _exact(value, INVOCATION_KEYS, "invocation")
    if _integer(invocation["schema_version"], "invocation.schema_version", minimum=1) != LAUNCHER_SCHEMA_VERSION or invocation["status"] != "ACCEPTED":
        _fail("invocation state drift")
    descriptor = validate_launch_descriptor(invocation["descriptor"])
    if _sha_string(invocation["descriptor_sha256"], "invocation.descriptor_sha256") != descriptor["aggregate_sha256"]:
        _fail("invocation descriptor binding drift")
    return _copy(invocation)


def _validate_result_shape(value: Any, expected_descriptor: Any | None = None) -> dict[str, Any]:
    _assert_builtin_json(value, "launch result")
    result = _exact(value, RESULT_KEYS, "launch result")
    if _integer(result["schema_version"], "result.schema_version", minimum=1) != LAUNCHER_SCHEMA_VERSION or result["process_contract_id"] != PROCESS_CONTRACT_ID or result["launcher_id"] != LAUNCHER_ID or result["mode"] != LAUNCHER_MODE:
        _fail("launch result identity drift")
    _string(result["run_id"], "result.run_id")
    _entry._nonce(result["nonce"])
    for field in ("launch_descriptor_sha256", "entry_descriptor_sha256", "receipt_sha256", "state_transition_sha256", "aggregate_result_sha256"):
        _sha_string(result[field], f"result.{field}")
    if type(result["argv"]) is not list or any(type(item) is not str for item in result["argv"]):
        _fail("result argv drift")
    _absolute_path(result["cwd"], "result.cwd")
    _exact(result["environment"], set(_entry.ENVIRONMENT_KEYS), "result.environment")
    for key, child in result["environment"].items():
        _string(key, "result.environment.key")
        _string(child, f"result.environment.{key}", nonempty=False)
    _validate_config_identity(result["config_identity"], "result.config_identity")
    source = _exact(result["source_bindings"], {"entry_module", "launcher_module", "repository_modules"}, "result.source_bindings")
    _validate_module_identity(source["entry_module"], "result.source_bindings.entry_module", _entry.ENTRY_MODULE_RELATIVE_PATH)
    _validate_module_identity(source["launcher_module"], "result.source_bindings.launcher_module", LAUNCHER_MODULE_RELATIVE_PATH)
    if type(source["repository_modules"]) is not list or len(source["repository_modules"]) != len(_entry.SOURCE_MODULES):
        _fail("result repository module identity drift")
    for index, row in enumerate(source["repository_modules"]):
        row = _exact(row, {"role", "relative_path", "size_bytes", "sha256", "mode"}, f"result.repository_modules[{index}]")
        _string(row["role"], f"result.repository_modules[{index}].role")
        _relative_path(row["relative_path"], f"result.repository_modules[{index}].relative_path")
        _integer(row["size_bytes"], f"result.repository_modules[{index}].size_bytes", minimum=1)
        _sha_string(row["sha256"], f"result.repository_modules[{index}].sha256")
        _integer(row["mode"], f"result.repository_modules[{index}].mode", minimum=1)
    future = _exact(result["future_targets"], set(_entry.FUTURE_TARGETS), "result.future_targets")
    _strict_equal(future, _entry.FUTURE_TARGETS, "result.future_targets")
    evidence_root = _absolute_path(result["evidence_root"], "result.evidence_root")
    _entry._reject_production_root(evidence_root, "result.evidence_root")
    receipt_path = _absolute_path(result["receipt_path"], "result.receipt_path")
    expected_receipt = str(Path(evidence_root).parent / _entry._receipt_name(evidence_root, LAUNCH_RECEIPT_SUFFIX))
    if receipt_path != expected_receipt:
        _fail("result receipt path drift")
    root_identity = _exact(result["evidence_root_identity"], {"device", "inode", "mode", "uid", "gid", "nlink", "size_bytes", "mtime_ns", "ctime_ns"}, "result.evidence_root_identity")
    for field in root_identity:
        _integer(root_identity[field], f"result.evidence_root_identity.{field}", minimum=0)
    _integer(result["receipt_size_bytes"], "result.receipt_size_bytes", minimum=1)
    for field in ("stdout_size_bytes", "stderr_size_bytes"):
        _integer(result[field], f"result.{field}", minimum=0)
    for field in ("stdout_sha256", "stderr_sha256"):
        _sha_string(result[field], f"result.{field}")
    if result["stdout_anchor_relative_path"] != "stdout.bin" or result["stderr_anchor_relative_path"] != "stderr.bin":
        _fail("result byte anchor path drift")
    _integer(result["return_code"], "result.return_code", minimum=RETURN_CODE_MIN, maximum=RETURN_CODE_MAX)
    status = _string(result["status"], "result.status")
    if status not in {"SYNTHETIC_TERMINAL_COMPLETE", "TERMINAL_FAILED"} or result["classification"] != status:
        _fail("result terminal status drift")
    failure_class = _string(result["failure_class"], "result.failure_class")
    if failure_class not in FAILURE_CLASSES:
        _fail("result failure class is not allowlisted")
    if status == "SYNTHETIC_TERMINAL_COMPLETE":
        if result["return_code"] != 0 or failure_class != "NONE" or type(result["entry_result"]) is not dict:
            _fail("synthetic success result is not fully bound")
        _sha_string(result["entry_result_sha256"], "result.entry_result_sha256")
        entry_result = _entry.validate_entry_result(result["entry_result"], None if expected_descriptor is None else expected_descriptor["entry_descriptor"])
        if result["entry_result_sha256"] != _digest(entry_result):
            _fail("entry result digest drift")
    else:
        if result["entry_result"] is not None or result["entry_result_sha256"] is not None or failure_class == "NONE" or result["return_code"] == 0:
            _fail("terminal failure result payload drift")
        if failure_class == "PERSISTENCE_FAILURE":
            _fail("persistence failure has no complete durable terminal record")
        if failure_class == "RUNNER_NONZERO_RETURN_CODE":
            if result["return_code"] == 0:
                _fail("nonzero runner failure has a zero return code")
        elif result["return_code"] != FAILURE_RETURN_CODES.get(failure_class):
            _fail("failure class and return code are not bound")
    trace = result["state_sequence"]
    if type(trace) is not list or len(trace) != 4 or [row.get("state") for row in trace[:3]] != ["ABSENT", "PREPARED", "ACCEPTED"] or trace[-1].get("state") != status:
        _fail("result state sequence drift")
    for index, row in enumerate(trace):
        row = _exact(row, {"sequence", "state", "predecessor_sha256"}, f"result.state_sequence[{index}]")
        if row["sequence"] != index or (index == 0 and row["predecessor_sha256"] is not None) or (index > 0 and row["predecessor_sha256"] != _digest(trace[index - 1])):
            _fail("result state predecessor drift")
    if result["state_transition_sha256"] != _digest(trace) or result["production_training_authorized"] is not False or result["real_process_launch_authorized"] is not False:
        _fail("result state or readiness drift")
    if expected_descriptor is not None:
        expected = validate_launch_descriptor(expected_descriptor)
        expected_fields = (
            "process_contract_id", "launcher_id", "mode", "run_id", "nonce", "argv", "cwd", "environment",
            "config_identity", "source_bindings", "future_targets", "evidence_root", "receipt_path",
        )
        for field in expected_fields:
            _strict_equal(result[field], expected[field], f"result.{field}")
        if result["launch_descriptor_sha256"] != expected["aggregate_sha256"] or result["entry_descriptor_sha256"] != expected["entry_descriptor_sha256"]:
            _fail("result descriptor binding drift")
        if result["status"] == "SYNTHETIC_TERMINAL_COMPLETE" and result["entry_result"] is not None and result["entry_result"]["invocation_descriptor_sha256"] != expected["entry_descriptor_sha256"]:
            _fail("result entry descriptor binding drift")
    body = _copy(result)
    aggregate = body.pop("aggregate_result_sha256")
    if _digest(body) != aggregate:
        _fail("launch result aggregate identity drift")
    return _copy(result)


def validate_launch_result(value: Any, expected_descriptor: Any | None = None) -> dict[str, Any]:
    """Validate a detached result, optionally against its exact descriptor."""

    return _validate_result_shape(value, expected_descriptor)


def _validate_published_root(
    root: str | os.PathLike[str],
    expected_descriptor: Any | None = None,
    *,
    lease: _entry._DirectoryLease | None = None,
    bound_parent: _entry._DirectoryLease | None = None,
    bound_name: str | None = None,
) -> dict[str, Any]:
    raw_root = _absolute_path(os.fspath(root), "launch evidence root")
    expected = None if expected_descriptor is None else validate_launch_descriptor(expected_descriptor)
    owns_lease = lease is None
    if lease is None:
        lease = _open_root_lease(raw_root)
    try:
        if bound_parent is not None:
            if bound_name is None:
                _fail("launch root binding name is missing")
            _assert_root_binding(bound_parent, bound_name, lease)
        names = set(_entry._listdir_fd(lease.fd))
        if names != set(EVIDENCE_FILE_NAMES):
            _fail("launch evidence root file set drift")
        invocation, invocation_raw = _read_json_at(lease, "invocation.json")
        invocation = _validate_invocation(invocation)
        descriptor = invocation["descriptor"]
        if expected is not None:
            _strict_equal(descriptor, expected, "published invocation descriptor")
        if descriptor["evidence_root"] != raw_root:
            _fail("published launch root binding drift")
        if bound_parent is None:
            receipt_stat = _verify_launch_receipt(descriptor)
        else:
            receipt_name = Path(descriptor["receipt_path"]).name
            receipt_stat = _verify_launch_receipt_at(bound_parent, descriptor, receipt_name)
        result, result_raw = _read_json_at(lease, "result.json")
        validated_result = validate_launch_result(result, expected_descriptor=descriptor)
        stdout, _ = _read_file_at(lease, "stdout.bin", "stdout anchor")
        stderr, _ = _read_file_at(lease, "stderr.bin", "stderr anchor")
        if len(stdout) != validated_result["stdout_size_bytes"] or _sha(stdout) != validated_result["stdout_sha256"] or len(stderr) != validated_result["stderr_size_bytes"] or _sha(stderr) != validated_result["stderr_sha256"]:
            _fail("published byte anchor identity drift")
        inventory, inventory_raw = _read_json_at(lease, "artifact_inventory.json")
        inventory = _exact(inventory, INVENTORY_KEYS, "artifact inventory")
        if inventory["excluded"] != sorted(INVENTORY_EXCLUDED):
            _fail("published inventory exclusion drift")
        files = inventory["files"]
        if type(files) is not list:
            _fail("published inventory rows are not a list")
        for index, row in enumerate(files):
            row = _exact(row, {"relative_path", "size_bytes", "sha256"}, f"artifact inventory.files[{index}]")
            _relative_path(row["relative_path"], f"artifact inventory.files[{index}].relative_path")
            _integer(row["size_bytes"], f"artifact inventory.files[{index}].size_bytes", minimum=0)
            _sha_string(row["sha256"], f"artifact inventory.files[{index}].sha256")
        if files != _inventory_at(lease) or inventory["canonical_inventory_sha256"] != _digest(files):
            _fail("published inventory rows or digest drift")
        completion, completion_raw = _read_json_at(lease, "completion.json")
        completion = _exact(completion, COMPLETION_KEYS, "completion")
        expected_completion = {
            "schema_version": LAUNCHER_SCHEMA_VERSION,
            "status": validated_result["status"],
            "classification": validated_result["classification"],
            "exit_code": validated_result["return_code"],
            "descriptor_sha256": descriptor["aggregate_sha256"],
            "result_sha256": _sha(result_raw),
            "inventory_sha256": _sha(inventory_raw),
            "stdout_size_bytes": validated_result["stdout_size_bytes"],
            "stdout_sha256": validated_result["stdout_sha256"],
            "stderr_size_bytes": validated_result["stderr_size_bytes"],
            "stderr_sha256": validated_result["stderr_sha256"],
            "entry_result_sha256": validated_result["entry_result_sha256"],
            "state_transition_sha256": validated_result["state_transition_sha256"],
            "evidence_root_identity": validated_result["evidence_root_identity"],
            "production_training_authorized": False,
        }
        _strict_equal(completion, expected_completion, "completion binding")
        if receipt_stat.st_size != validated_result["receipt_size_bytes"] or _sha(_canonical(_launch_receipt_payload(descriptor))) != validated_result["receipt_sha256"]:
            _fail("published receipt identity drift")
        observed_root = _root_identity(lease.snapshot)
        if _stable_root_identity(observed_root) != _stable_root_identity(validated_result["evidence_root_identity"]):
            _fail("published root identity drift")
        if bound_parent is not None:
            _assert_root_binding(bound_parent, bound_name, lease)
        return {"descriptor": _copy(descriptor), "result": _copy(validated_result), "invocation_raw": invocation_raw, "result_raw": result_raw, "inventory_raw": inventory_raw, "completion_raw": completion_raw}
    finally:
        if owns_lease:
            lease.close()


def _make_result(
    descriptor: dict[str, Any],
    trace: list[dict[str, Any]],
    *,
    root_identity: dict[str, int],
    receipt_stat: os.stat_result,
    return_code: int,
    stdout: bytes,
    stderr: bytes,
    entry_result: dict[str, Any] | None,
    status: str,
    failure_class: str,
) -> dict[str, Any]:
    if status not in {"SYNTHETIC_TERMINAL_COMPLETE", "TERMINAL_FAILED"} or failure_class not in FAILURE_CLASSES:
        _fail("invalid terminal result classification")
    state_trace = [*_copy(trace), {"sequence": len(trace), "state": status, "predecessor_sha256": _digest(trace[-1])}]
    body = {
        "schema_version": LAUNCHER_SCHEMA_VERSION,
        "process_contract_id": PROCESS_CONTRACT_ID,
        "launcher_id": LAUNCHER_ID,
        "mode": LAUNCHER_MODE,
        "run_id": descriptor["run_id"],
        "nonce": descriptor["nonce"],
        "launch_descriptor_sha256": descriptor["aggregate_sha256"],
        "entry_descriptor_sha256": descriptor["entry_descriptor_sha256"],
        "argv": list(descriptor["argv"]),
        "cwd": descriptor["cwd"],
        "environment": _copy(descriptor["environment"]),
        "config_identity": _copy(descriptor["config_identity"]),
        "source_bindings": _copy(descriptor["source_bindings"]),
        "future_targets": _copy(descriptor["future_targets"]),
        "evidence_root": descriptor["evidence_root"],
        "receipt_path": descriptor["receipt_path"],
        "evidence_root_identity": _copy(root_identity),
        "receipt_size_bytes": receipt_stat.st_size,
        "receipt_sha256": _sha(_canonical(_launch_receipt_payload(descriptor))),
        "return_code": return_code,
        "stdout_size_bytes": len(stdout),
        "stdout_sha256": _sha(stdout),
        "stderr_size_bytes": len(stderr),
        "stderr_sha256": _sha(stderr),
        "stdout_anchor_relative_path": "stdout.bin",
        "stderr_anchor_relative_path": "stderr.bin",
        "entry_result": _copy(entry_result),
        "entry_result_sha256": None if entry_result is None else _digest(entry_result),
        "status": status,
        "classification": status,
        "failure_class": failure_class,
        "state_sequence": state_trace,
        "state_transition_sha256": _digest(state_trace),
        "production_training_authorized": False,
        "real_process_launch_authorized": False,
    }
    return {**body, "aggregate_result_sha256": _digest(body)}


def _runner_result(value: Any) -> dict[str, Any]:
    if inspect.isawaitable(value) or inspect.isgenerator(value):
        raise _RunnerAsyncResult("fake runner returned an awaitable or generator")
    try:
        result = _exact(value, RUNNER_RESULT_KEYS, "fake runner result")
    except TrainingProcessLauncherError as exc:
        raise _RunnerSchemaFailure(str(exc)) from exc
    if type(result["return_code"]) is not int or not RETURN_CODE_MIN <= result["return_code"] <= RETURN_CODE_MAX:
        raise _RunnerSchemaFailure("fake runner return_code is outside its frozen range")
    if type(result["stdout"]) is not bytes or type(result["stderr"]) is not bytes:
        raise _RunnerSchemaFailure("fake runner stdout and stderr must be exact bytes")
    if result["entry_result"] is not None:
        try:
            _entry._assert_builtin_json(result["entry_result"], "fake runner entry_result")
        except _entry.TrainingEntryError as exc:
            raise _RunnerSchemaFailure("fake runner entry_result is not builtin JSON") from exc
        if type(result["entry_result"]) is not dict:
            raise _RunnerSchemaFailure("fake runner entry_result must be a dict or null")
    return {"return_code": result["return_code"], "stdout": bytes(result["stdout"]), "stderr": bytes(result["stderr"]), "entry_result": _copy(result["entry_result"])}


def _read_result_status(lease: _entry._DirectoryLease) -> str | None:
    fd: int | None = None
    try:
        lease.revalidate()
        fd = _entry._open_fd("result.json", _entry._file_flags(), dir_fd=lease.fd)
        observed = _entry._fstat(fd)
        _entry._require_regular_file(observed, "forged result")
        while True:
            try:
                os.lseek(fd, 0, os.SEEK_SET)
                break
            except InterruptedError:
                continue
        raw = _entry._read_fd_exact(fd, observed.st_size, "forged result")
        after = _entry._fstat(fd)
        if not _entry._same_file_identity(observed, after):
            return None
        current = _entry._stat_at(lease.fd, "result.json")
        if current is None or not _entry._same_file_identity(after, current):
            return None
        lease.revalidate()
        result = _parse_json(raw, "forged result")
    except Exception:
        return None
    finally:
        if fd is not None:
            try:
                _entry._close_fd(fd)
            except OSError:
                pass
    if type(result) is dict and result.get("status") == "TERMINAL_COMPLETE":
        return "TERMINAL_FAILED"
    return None


def run_synthetic_launch(
    descriptor: Any,
    runner: Callable[[list[str], str, dict[str, str], dict[str, Any]], Mapping[str, Any]],
    evidence_root: str | os.PathLike[str],
) -> dict[str, Any]:
    """Run one synchronous fake launch and publish immutable synthetic evidence."""

    validated = validate_launch_descriptor(descriptor)
    _validate_runner(runner)
    root = _absolute_path(os.fspath(evidence_root), "launch evidence root")
    if root != validated["evidence_root"]:
        _fail("launch evidence root is not the descriptor-bound root")
    claim_parent, receipt_name, receipt_stat = _claim_launch_receipt(validated)
    root_lease: _entry._DirectoryLease | None = None
    try:
        root_path = Path(root)
        created_root = _create_root(claim_parent, root_path.name)
        root_lease = _open_root_from_parent(claim_parent, root_path.name, expected=created_root)
        _assert_root_binding(claim_parent, root_path.name, root_lease)
        receipt_stat = _verify_launch_receipt_at(claim_parent, validated, receipt_name)
        invocation = {
            "schema_version": LAUNCHER_SCHEMA_VERSION,
            "status": "ACCEPTED",
            "descriptor_sha256": validated["aggregate_sha256"],
            "descriptor": _copy(validated),
        }
        _publish_at(root_lease, "invocation.json", _canonical(invocation))
        _assert_root_binding(claim_parent, root_path.name, root_lease)
        argv = list(validated["argv"])
        cwd = validated["cwd"]
        environment = _copy(validated["environment"])
        entry_descriptor = _copy(validated["entry_descriptor"])
        before = (_copy(argv), cwd, _copy(environment), _copy(entry_descriptor))
        stdout = b""
        stderr = b""
        return_code = SAFE_FAILURE_RETURN_CODE
        entry_result: dict[str, Any] | None = None
        failure_class = "RUNNER_EXCEPTION"
        status = "TERMINAL_FAILED"
        try:
            raw_result = runner(argv, cwd, environment, entry_descriptor)
        except TimeoutError:
            return_code = FAILURE_RETURN_CODES["RUNNER_TIMEOUT"]
            failure_class = "RUNNER_TIMEOUT"
        except Exception:
            return_code = FAILURE_RETURN_CODES["RUNNER_EXCEPTION"]
            failure_class = "RUNNER_EXCEPTION"
        else:
            try:
                _assert_runner_inputs_unchanged(argv, cwd, environment, entry_descriptor, before)
                normalized = _runner_result(raw_result)
                return_code = normalized["return_code"]
                stdout = normalized["stdout"]
                stderr = normalized["stderr"]
                entry_result = normalized["entry_result"]
                if return_code == 0:
                    if entry_result is None:
                        raise _RunnerSchemaFailure("zero return code without entry result")
                    entry_result = _entry.validate_entry_result(entry_result, expected_descriptor=validated["entry_descriptor"])
                    failure_class = "NONE"
                    status = "SYNTHETIC_TERMINAL_COMPLETE"
                else:
                    if entry_result is not None:
                        raise _RunnerSchemaFailure("nonzero return code carried an entry result")
                    failure_class = "RUNNER_NONZERO_RETURN_CODE"
            except _RunnerAsyncResult:
                stdout = b""
                stderr = b""
                return_code = FAILURE_RETURN_CODES["RUNNER_ASYNC_OR_GENERATOR_RESULT"]
                failure_class = "RUNNER_ASYNC_OR_GENERATOR_RESULT"
                entry_result = None
            except _RunnerInputMutation:
                stdout = b""
                stderr = b""
                return_code = FAILURE_RETURN_CODES["RUNNER_MUTATED_INVOCATION"]
                failure_class = "RUNNER_MUTATED_INVOCATION"
                entry_result = None
            except Exception:
                stdout = b""
                stderr = b""
                return_code = FAILURE_RETURN_CODES["RUNNER_SCHEMA_OR_ENTRY_FAILURE"]
                failure_class = "RUNNER_SCHEMA_OR_ENTRY_FAILURE"
                entry_result = None
        _assert_root_binding(claim_parent, root_path.name, root_lease)
        _publish_at(root_lease, "stdout.bin", stdout)
        _publish_at(root_lease, "stderr.bin", stderr)
        stdout, _ = _read_file_at(root_lease, "stdout.bin", "stdout anchor")
        stderr, _ = _read_file_at(root_lease, "stderr.bin", "stderr anchor")
        _assert_root_binding(claim_parent, root_path.name, root_lease)
        root_identity = _root_identity(root_lease.snapshot)
        result = _make_result(
            validated,
            _state_trace(["ABSENT", "PREPARED", "ACCEPTED"]),
            root_identity=root_identity,
            receipt_stat=receipt_stat,
            return_code=return_code,
            stdout=stdout,
            stderr=stderr,
            entry_result=entry_result,
            status=status,
            failure_class=failure_class,
        )
        result = validate_launch_result(result, expected_descriptor=validated)
        _publish_at(root_lease, "result.json", _canonical(result))
        _assert_root_binding(claim_parent, root_path.name, root_lease)
        rows = _inventory_at(root_lease)
        inventory = {
            "schema_version": LAUNCHER_SCHEMA_VERSION,
            "excluded": sorted(INVENTORY_EXCLUDED),
            "files": rows,
            "canonical_inventory_sha256": _digest(rows),
        }
        _publish_at(root_lease, "artifact_inventory.json", _canonical(inventory))
        result_raw = _read_json_at(root_lease, "result.json")[1]
        inventory_raw = _read_json_at(root_lease, "artifact_inventory.json")[1]
        completion = {
            "schema_version": LAUNCHER_SCHEMA_VERSION,
            "status": result["status"],
            "classification": result["classification"],
            "exit_code": result["return_code"],
            "descriptor_sha256": validated["aggregate_sha256"],
            "result_sha256": _sha(result_raw),
            "inventory_sha256": _sha(inventory_raw),
            "stdout_size_bytes": result["stdout_size_bytes"],
            "stdout_sha256": result["stdout_sha256"],
            "stderr_size_bytes": result["stderr_size_bytes"],
            "stderr_sha256": result["stderr_sha256"],
            "entry_result_sha256": result["entry_result_sha256"],
            "state_transition_sha256": result["state_transition_sha256"],
            "evidence_root_identity": result["evidence_root_identity"],
            "production_training_authorized": False,
        }
        _publish_at(root_lease, "completion.json", _canonical(completion))
        _assert_root_binding(claim_parent, root_path.name, root_lease)
        return _copy(
            _validate_published_root(
                root,
                expected_descriptor=validated,
                lease=root_lease,
                bound_parent=claim_parent,
                bound_name=root_path.name,
            )["result"]
        )
    finally:
        if root_lease is not None:
            root_lease.close()
        claim_parent.close()


def classify_launch(evidence_root: str | os.PathLike[str]) -> str:
    """Classify a durable launch root conservatively without modifying it."""

    try:
        raw = os.fspath(evidence_root)
        if type(raw) is not str:
            return "UNKNOWN"
        raw = _absolute_path(raw, "launch evidence root")
        parent, name = _entry._bound_target_parts(raw, "launch evidence root")
        lease: _entry._DirectoryLease | None = None
        try:
            observed = _entry._stat_at(parent.fd, name)
            receipt_observed = _entry._stat_at(parent.fd, _entry_receipt_name(raw))
            if observed is None:
                return "UNKNOWN" if receipt_observed is not None else "ABSENT"
            if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
                return "UNKNOWN"
            lease = _open_root_from_parent(parent, name)
            _assert_root_binding(parent, name, lease)
            names = set(_entry._listdir_fd(lease.fd))
            if names == {"invocation.json"}:
                try:
                    invocation, _ = _read_json_at(lease, "invocation.json")
                    checked = _validate_invocation(invocation)
                    _verify_launch_receipt_at(parent, checked["descriptor"], _entry_receipt_name(raw))
                    _assert_root_binding(parent, name, lease)
                    return "IN_PROGRESS"
                except Exception:
                    return "UNKNOWN"
            if names == set(EVIDENCE_FILE_NAMES):
                try:
                    published = _validate_published_root(raw, lease=lease, bound_parent=parent, bound_name=name)
                    status = published["result"]["status"]
                    if status in {"SYNTHETIC_TERMINAL_COMPLETE", "TERMINAL_FAILED"}:
                        return status
                except Exception:
                    forged = _read_result_status(lease)
                    if forged is not None:
                        return forged
                    return "UNKNOWN"
            forged = _read_result_status(lease)
            return forged if forged is not None else "UNKNOWN"
        finally:
            if lease is not None:
                lease.close()
            parent.close()
    except Exception:
        return "UNKNOWN"
