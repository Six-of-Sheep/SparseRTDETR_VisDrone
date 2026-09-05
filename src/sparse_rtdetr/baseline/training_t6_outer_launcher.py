"""T6B outer launch boundary and immediate snapshot.

The outer layer owns the one-shot tmux request and the outer evidence root.
It never interprets ``TMUX_ACCEPTED`` as training completion.  Tests inject a
synchronous tmux port; production uses one absolute ``Popen`` call with
``shell=False``.  No polling, retry, resume, or fallback is implemented.
"""

from __future__ import annotations

import copy
import datetime as _datetime
import hashlib
import inspect
import json
import math
import os
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from sparse_rtdetr.baseline import training_t6_entry as _entry
from sparse_rtdetr.baseline import training_t6_process_launcher as _process


OUTER_SCHEMA_VERSION = 1
OUTER_CONTRACT_ID = _entry.T6B_CONTRACT_ID
OUTER_LAUNCHER_ID = "rtdetrv2_r18_visdrone_training_t6_outer_launcher_v1"
OUTER_MODE = "production"
OUTER_MODULE_RELATIVE_PATH = _entry.OUTER_MODULE_RELATIVE_PATH
OUTER_RECEIPT_SUFFIX = ".t6b-outer-receipt.json"
OUTER_STREAM_RECEIPT_SUFFIX = ".t6b-outer-streams.json"
PERSISTENCE_FAILURE_MARKER_SUFFIX = ".t6b-outer-persistence-failure.json"
OUTER_FILE_NAMES = ("invocation.json", "snapshot.json", "stdout.bin", "stderr.bin", "result.json", "artifact_inventory.json", "completion.json")
OUTER_JSON_NAMES = ("invocation.json", "snapshot.json", "result.json", "artifact_inventory.json", "completion.json")
OUTER_BINARY_NAMES = ("stdout.bin", "stderr.bin")
TMUX_RESULT_KEYS = {"return_code", "stdout", "stderr", "pid"}

DESCRIPTOR_KEYS = {
    "schema_version",
    "contract_id",
    "launcher_id",
    "mode",
    "training_run_id",
    "nonce",
    "tmux_session_name",
    "process_descriptor",
    "process_descriptor_sha256",
    "authorization_path",
    "authorization_receipt_path",
    "authorization_binding",
    "authorization_binding_sha256",
    "repository",
    "environment",
    "data_roles",
    "targets",
    "outer_evidence_root",
    "outer_receipt_path",
    "cwd",
    "tmux_argv",
    "state_machine",
    "invocation_policy",
    "aggregate_sha256",
}
RESULT_KEYS = {
    "schema_version",
    "contract_id",
    "launcher_id",
    "mode",
    "training_run_id",
    "nonce",
    "tmux_session_name",
    "descriptor_sha256",
    "process_descriptor_sha256",
    "authorization_binding_sha256",
    "repository",
    "environment",
    "data_roles",
    "targets",
    "outer_evidence_root",
    "outer_receipt_path",
    "tmux_argv",
    "pid",
    "return_code",
    "stdout_size_bytes",
    "stdout_sha256",
    "stderr_size_bytes",
    "stderr_sha256",
    "stdout_anchor_relative_path",
    "stderr_anchor_relative_path",
    "snapshot_monotonic_ns",
    "snapshot_utc",
    "status",
    "classification",
    "failure_class",
    "outer_invocation_count",
    "tmux_new_session_count",
    "shell_used",
    "production_training_authorized",
    "real_process_launch_authorized",
    "training_certified",
    "state_sequence",
    "state_transition_sha256",
    "aggregate_result_sha256",
}

__all__ = (
    "TrainingOuterError",
    "build_production_outer_descriptor",
    "validate_outer_descriptor",
    "run_outer_once",
    "validate_outer_result",
    "classify_outer",
)


class TrainingOuterError(ValueError):
    """Raised when outer launch identity or durable evidence drifts."""


def _fail(message: str) -> None:
    raise TrainingOuterError(message)


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
        return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise TrainingOuterError("value is not canonical JSON") from exc


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


def _string(value: Any, field: str, *, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value):
        _fail(f"{field} must be a builtin string")
    return value


def _absolute(value: Any, field: str) -> str:
    try:
        return _entry._absolute_path(value, field)
    except _entry.TrainingEntryError as exc:
        raise TrainingOuterError(str(exc)) from exc


def _integer(value: Any, field: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    if type(value) is not int:
        _fail(f"{field} must be a builtin int")
    if minimum is not None and value < minimum:
        _fail(f"{field} is below minimum")
    if maximum is not None and value > maximum:
        _fail(f"{field} is above maximum")
    return value


def _state_trace(states: list[str]) -> list[dict[str, Any]]:
    if not states or states[0] != "DESIGN_ONLY":
        _fail("outer state trace must start at DESIGN_ONLY")
    trace = [{"sequence": 0, "state": states[0], "predecessor_sha256": None}]
    for index, state in enumerate(states[1:], 1):
        if state not in _entry.STATE_SEQUENCE:
            _fail("outer state trace contains unknown state")
        trace.append({"sequence": index, "state": state, "predecessor_sha256": _digest(trace[-1])})
    return trace


def _validate_environment(value: Any) -> dict[str, Any]:
    try:
        return _entry._validate_environment(value)
    except _entry.TrainingEntryError as exc:
        raise TrainingOuterError(str(exc)) from exc


def _validate_data(value: Any) -> dict[str, Any]:
    try:
        return _entry._validate_data_roles(value)
    except _entry.TrainingEntryError as exc:
        raise TrainingOuterError(str(exc)) from exc


def _validate_targets(value: Any) -> dict[str, str]:
    try:
        return _entry._validate_targets(value)
    except _entry.TrainingEntryError as exc:
        raise TrainingOuterError(str(exc)) from exc


def _validate_tmux_contract(value: Any) -> dict[str, Any]:
    contract = _exact(value, {"argv_sequence", "shell", "max_calls", "synchronous", "new_session_once"}, "tmux contract")
    if contract != {"argv_sequence": True, "shell": False, "max_calls": 1, "synchronous": True, "new_session_once": True}:
        _fail("tmux contract drift")
    return _copy(contract)


def _validate_descriptor_shape(value: Any) -> dict[str, Any]:
    _assert_builtin(value, "outer descriptor")
    descriptor = _exact(value, DESCRIPTOR_KEYS, "outer descriptor")
    if descriptor["schema_version"] != OUTER_SCHEMA_VERSION or descriptor["contract_id"] != OUTER_CONTRACT_ID or descriptor["launcher_id"] != OUTER_LAUNCHER_ID or descriptor["mode"] != OUTER_MODE:
        _fail("outer descriptor identity drift")
    _string(descriptor["training_run_id"], "descriptor.training_run_id")
    _string(descriptor["nonce"], "descriptor.nonce")
    _string(descriptor["tmux_session_name"], "descriptor.tmux_session_name")
    process = _process.validate_process_descriptor(descriptor["process_descriptor"])
    if descriptor["process_descriptor_sha256"] != process["aggregate_sha256"]:
        _fail("outer process descriptor digest drift")
    entry = process["entry_descriptor"]
    if descriptor["training_run_id"] != entry["training_run_id"] or descriptor["nonce"] != entry["nonce"] or descriptor["tmux_session_name"] != entry["tmux_session_name"]:
        _fail("outer run identity drift")
    if descriptor["authorization_binding"] != entry["authorization_binding"] or descriptor["authorization_binding_sha256"] != entry["authorization_binding_sha256"]:
        _fail("outer authorization binding drift")
    _absolute(descriptor["authorization_path"], "descriptor.authorization_path")
    _absolute(descriptor["authorization_receipt_path"], "descriptor.authorization_receipt_path")
    _validate_environment(descriptor["environment"])
    if descriptor["environment"] != entry["environment"]:
        _fail("outer environment drift")
    _validate_data(descriptor["data_roles"])
    if descriptor["data_roles"] != entry["data_roles"]:
        _fail("outer data role drift")
    _validate_targets(descriptor["targets"])
    _absolute(descriptor["outer_evidence_root"], "descriptor.outer_evidence_root")
    if descriptor["outer_evidence_root"] != entry["outer_evidence_root"]:
        _fail("outer evidence root drift")
    if descriptor["outer_evidence_root"] in {entry["evidence_root"], entry["process_evidence_root"]}:
        _fail("outer evidence root is not distinct")
    receipt = _absolute(descriptor["outer_receipt_path"], "descriptor.outer_receipt_path")
    expected_receipt = str(Path(descriptor["outer_evidence_root"]).parent / f".{Path(descriptor['outer_evidence_root']).name}{OUTER_RECEIPT_SUFFIX}")
    if receipt != expected_receipt:
        _fail("outer receipt path drift")
    _absolute(descriptor["cwd"], "descriptor.cwd")
    if descriptor["cwd"] != entry["cwd"]:
        _fail("outer cwd drift")
    argv = descriptor["tmux_argv"]
    if type(argv) is not list or not argv or any(type(item) is not str or not item for item in argv):
        _fail("outer tmux argv drift")
    if not argv[0].startswith("/") or "new-session" not in argv:
        _fail("outer tmux argv is not the exact sequence")
    machine = _exact(descriptor["state_machine"], {"states", "trace", "exactly_once", "durable_evidence_required"}, "outer state_machine")
    if machine["states"] != list(_entry.STATE_SEQUENCE) or machine["exactly_once"] is not True or machine["durable_evidence_required"] is not True:
        _fail("outer state machine policy drift")
    trace = machine["trace"]
    if type(trace) is not list or len(trace) != 3 or [row.get("state") for row in trace] != ["DESIGN_ONLY", "OWNER_AUTHORIZED", "PREFLIGHT_PASS"]:
        _fail("outer prepared state trace drift")
    for index, row in enumerate(trace):
        row = _exact(row, {"sequence", "state", "predecessor_sha256"}, f"outer trace[{index}]")
        if row["sequence"] != index or (index == 0 and row["predecessor_sha256"] is not None) or (index > 0 and row["predecessor_sha256"] != _digest(trace[index - 1])):
            _fail("outer state predecessor drift")
    invocation = _validate_tmux_contract(descriptor["invocation_policy"])
    if invocation != {"argv_sequence": True, "shell": False, "max_calls": 1, "synchronous": True, "new_session_once": True}:
        _fail("outer invocation policy drift")
    aggregate = _string(descriptor["aggregate_sha256"], "descriptor.aggregate_sha256")
    if len(aggregate) != 64 or any(char not in "0123456789abcdef" for char in aggregate):
        _fail("outer descriptor aggregate is not SHA-256")
    body = _copy(descriptor)
    del body["aggregate_sha256"]
    if _digest(body) != aggregate:
        _fail("outer descriptor aggregate drift")
    return _copy(descriptor)


def build_production_outer_descriptor(process_descriptor: Mapping[str, Any]) -> dict[str, Any]:
    process = _process.validate_process_descriptor(dict(process_descriptor))
    entry = process["entry_descriptor"]
    root = Path(entry["repository"]["repo_root"])
    outer_root = entry["outer_evidence_root"]
    outer_receipt = str(Path(outer_root).parent / f".{Path(outer_root).name}{OUTER_RECEIPT_SUFFIX}")
    tmux_argv = ["/usr/bin/tmux", "new-session", "-d", "-s", entry["tmux_session_name"], *process["argv"]]
    body = {
        "schema_version": OUTER_SCHEMA_VERSION,
        "contract_id": OUTER_CONTRACT_ID,
        "launcher_id": OUTER_LAUNCHER_ID,
        "mode": OUTER_MODE,
        "training_run_id": entry["training_run_id"],
        "nonce": entry["nonce"],
        "tmux_session_name": entry["tmux_session_name"],
        "process_descriptor": process,
        "process_descriptor_sha256": process["aggregate_sha256"],
        "authorization_path": entry["authorization_path"],
        "authorization_receipt_path": entry["authorization_receipt_path"],
        "authorization_binding": _copy(entry["authorization_binding"]),
        "authorization_binding_sha256": entry["authorization_binding_sha256"],
        "repository": _copy(entry["repository"]),
        "environment": _copy(entry["environment"]),
        "data_roles": _copy(entry["data_roles"]),
        "targets": _copy(entry["targets"]),
        "outer_evidence_root": outer_root,
        "outer_receipt_path": outer_receipt,
        "cwd": entry["cwd"],
        "tmux_argv": tmux_argv,
        "state_machine": {"states": list(_entry.STATE_SEQUENCE), "trace": _state_trace(["DESIGN_ONLY", "OWNER_AUTHORIZED", "PREFLIGHT_PASS"]), "exactly_once": True, "durable_evidence_required": True},
        "invocation_policy": {"argv_sequence": True, "shell": False, "max_calls": 1, "synchronous": True, "new_session_once": True},
    }
    return _validate_descriptor_shape({**body, "aggregate_sha256": _digest(body)})


def validate_outer_descriptor(value: Any) -> dict[str, Any]:
    return _validate_descriptor_shape(value)


def _claim_root(descriptor: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    root = Path(descriptor["outer_evidence_root"])
    receipt = Path(descriptor["outer_receipt_path"])
    parent = root.parent
    try:
        _entry._canonical_root(str(parent), "outer evidence parent")
    except _entry.TrainingEntryError as exc:
        raise TrainingOuterError(str(exc)) from exc
    stream_receipt = root.parent / f".{root.name}{OUTER_STREAM_RECEIPT_SUFFIX}"
    if (
        root.exists()
        or root.is_symlink()
        or receipt.exists()
        or receipt.is_symlink()
        or stream_receipt.exists()
        or stream_receipt.is_symlink()
    ):
        _fail("outer evidence target or receipt already exists")
    receipt_payload = {
        "schema_version": OUTER_SCHEMA_VERSION,
        "kind": "T6B_OUTER_CLAIM",
        "descriptor_sha256": descriptor["aggregate_sha256"],
        "descriptor": _copy(descriptor),
        "training_run_id": descriptor["training_run_id"],
        "nonce": descriptor["nonce"],
        "status": "CLAIMED",
    }
    receipt_raw = _canonical(receipt_payload)
    _entry._write_new(receipt, receipt_raw, "outer receipt")
    _entry._fsync_directory(parent, "outer receipt parent")
    try:
        os.mkdir(root, 0o700)
        observed = root.lstat()
        if not stat.S_ISDIR(observed.st_mode) or stat.S_IMODE(observed.st_mode) != 0o700 or observed.st_nlink < 2 or observed.st_uid != os.getuid() or observed.st_gid != os.getgid():
            _fail("outer evidence root metadata drift")
        _entry._fsync_directory(parent, "outer evidence parent")
    except OSError as exc:
        raise TrainingOuterError("outer evidence root could not be created") from exc
    return root, {"receipt": receipt_payload, "receipt_sha256": _sha(receipt_raw)}


def _publish(root: Path, name: str, value: Any) -> None:
    _entry._write_new(root / name, _canonical(value), name)


def _persistence_failure_marker_path(root: Path) -> Path:
    return root.parent / f".{root.name}{PERSISTENCE_FAILURE_MARKER_SUFFIX}"


def _record_persistence_failure(
    root: Path,
    descriptor: dict[str, Any],
    phase: str,
    error: BaseException,
) -> None:
    payload = {
        "schema_version": OUTER_SCHEMA_VERSION,
        "kind": "T6B_OUTER_PERSISTENCE_FAILURE",
        "descriptor_sha256": descriptor["aggregate_sha256"],
        "training_run_id": descriptor["training_run_id"],
        "nonce": descriptor["nonce"],
        "outer_evidence_root": str(root),
        "phase": phase,
        "exception_type": type(error).__name__,
        "exception_message": str(error),
        "failure_class": "PERSISTENCE_FAILURE",
        "status": "PERMANENT_FAIL",
    }
    marker = _persistence_failure_marker_path(root)
    try:
        if not marker.exists() and not marker.is_symlink():
            _entry._write_new(marker, _canonical(payload), "outer persistence failure marker")
            _entry._fsync_directory(marker.parent, "outer persistence failure marker parent")
    except Exception:
        return


def _has_valid_persistence_failure_marker(root: Path) -> bool:
    marker = _persistence_failure_marker_path(root)
    if not marker.exists() and not marker.is_symlink():
        return False
    try:
        _entry._regular_file(marker, "outer persistence failure marker", expected_mode=0o600, expected_nlink=1)
        value, raw = _entry._read_json(marker, "outer persistence failure marker")
        expected = {
            "schema_version": OUTER_SCHEMA_VERSION,
            "kind": "T6B_OUTER_PERSISTENCE_FAILURE",
            "descriptor_sha256": value.get("descriptor_sha256"),
            "training_run_id": value.get("training_run_id"),
            "nonce": value.get("nonce"),
            "outer_evidence_root": str(root),
            "phase": value.get("phase"),
            "exception_type": value.get("exception_type"),
            "exception_message": value.get("exception_message"),
            "failure_class": "PERSISTENCE_FAILURE",
            "status": "PERMANENT_FAIL",
        }
        if raw != _canonical(value) or set(value) != set(expected) or value != expected:
            return False
        return all(type(value[field]) is str and bool(value[field]) for field in ("descriptor_sha256", "training_run_id", "nonce", "phase", "exception_type")) and len(value["descriptor_sha256"]) == 64
    except Exception:
        return False


def _validate_tmux_result(value: Any) -> dict[str, Any]:
    result = _exact(value, TMUX_RESULT_KEYS, "tmux result")
    if type(result["return_code"]) is not int or not 0 <= result["return_code"] <= 255 or type(result["stdout"]) is not bytes or type(result["stderr"]) is not bytes or type(result["pid"]) is not int or result["pid"] < 0:
        _fail("tmux result schema drift")
    return _copy(result)


def _invoke_real(descriptor: dict[str, Any]) -> dict[str, Any]:
    process = subprocess.Popen(
        list(descriptor["tmux_argv"]),
        cwd=descriptor["cwd"],
        env=dict(descriptor["environment"]["variables"]),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
    )
    stdout, stderr = process.communicate()
    return {"return_code": process.returncode, "stdout": stdout, "stderr": stderr, "pid": process.pid}


def _make_result(descriptor: dict[str, Any], payload: dict[str, Any], snapshot: dict[str, Any], receipt: dict[str, Any], failure_class: str) -> dict[str, Any]:
    accepted = payload["return_code"] == 0 and payload["pid"] > 0 and failure_class == "NONE"
    status = "TMUX_ACCEPTED" if accepted else "PERMANENT_FAIL"
    effective_failure = "NONE" if accepted else (failure_class if failure_class != "NONE" else "RUNNER_NONZERO_EXIT")
    stdout_size, stdout_sha = len(payload["stdout"]), _sha(payload["stdout"])
    stderr_size, stderr_sha = len(payload["stderr"]), _sha(payload["stderr"])
    trace = _state_trace(["DESIGN_ONLY", "OWNER_AUTHORIZED", "PREFLIGHT_PASS", "LAUNCH_ACCEPTED"])
    body = {
        "schema_version": OUTER_SCHEMA_VERSION,
        "contract_id": OUTER_CONTRACT_ID,
        "launcher_id": OUTER_LAUNCHER_ID,
        "mode": OUTER_MODE,
        "training_run_id": descriptor["training_run_id"],
        "nonce": descriptor["nonce"],
        "tmux_session_name": descriptor["tmux_session_name"],
        "descriptor_sha256": descriptor["aggregate_sha256"],
        "process_descriptor_sha256": descriptor["process_descriptor_sha256"],
        "authorization_binding_sha256": descriptor["authorization_binding_sha256"],
        "repository": _copy(descriptor["repository"]),
        "environment": _copy(descriptor["environment"]),
        "data_roles": _copy(descriptor["data_roles"]),
        "targets": _copy(descriptor["targets"]),
        "outer_evidence_root": descriptor["outer_evidence_root"],
        "outer_receipt_path": descriptor["outer_receipt_path"],
        "tmux_argv": list(descriptor["tmux_argv"]),
        "pid": payload["pid"],
        "return_code": payload["return_code"],
        "stdout_size_bytes": stdout_size,
        "stdout_sha256": stdout_sha,
        "stderr_size_bytes": stderr_size,
        "stderr_sha256": stderr_sha,
        "stdout_anchor_relative_path": "stdout.bin",
        "stderr_anchor_relative_path": "stderr.bin",
        "snapshot_monotonic_ns": snapshot["monotonic_ns"],
        "snapshot_utc": snapshot["utc"],
        "status": status,
        "classification": status,
        "failure_class": effective_failure,
        "outer_invocation_count": 1,
        "tmux_new_session_count": 1,
        "shell_used": False,
        "production_training_authorized": False,
        "real_process_launch_authorized": False,
        "training_certified": False,
        "state_sequence": trace,
        "state_transition_sha256": _digest(trace),
    }
    return {**body, "aggregate_result_sha256": _digest(body)}


def validate_outer_result(value: Any, expected_descriptor: Any | None = None) -> dict[str, Any]:
    result = _exact(value, RESULT_KEYS, "outer result")
    _assert_builtin(result, "outer result")
    if result["schema_version"] != OUTER_SCHEMA_VERSION or result["contract_id"] != OUTER_CONTRACT_ID or result["launcher_id"] != OUTER_LAUNCHER_ID or result["mode"] != OUTER_MODE:
        _fail("outer result identity drift")
    for field in ("training_run_id", "nonce", "tmux_session_name"):
        _string(result[field], f"result.{field}")
    for field in ("descriptor_sha256", "process_descriptor_sha256", "authorization_binding_sha256", "stdout_sha256", "stderr_sha256", "state_transition_sha256", "aggregate_result_sha256"):
        value = result[field]
        if type(value) is not str or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            _fail(f"result.{field} is not SHA-256")
    _integer(result["pid"], "result.pid", minimum=0)
    _integer(result["return_code"], "result.return_code", minimum=0, maximum=255)
    _integer(result["stdout_size_bytes"], "result.stdout_size_bytes", minimum=0)
    _integer(result["stderr_size_bytes"], "result.stderr_size_bytes", minimum=0)
    _integer(result["snapshot_monotonic_ns"], "result.snapshot_monotonic_ns", minimum=1)
    _string(result["snapshot_utc"], "result.snapshot_utc")
    if result["stdout_anchor_relative_path"] != "stdout.bin" or result["stderr_anchor_relative_path"] != "stderr.bin":
        _fail("outer binary anchor path drift")
    _absolute(result["outer_evidence_root"], "result.outer_evidence_root")
    _absolute(result["outer_receipt_path"], "result.outer_receipt_path")
    if type(result["tmux_argv"]) is not list or not result["tmux_argv"] or any(type(item) is not str for item in result["tmux_argv"]):
        _fail("outer result argv drift")
    if result["tmux_argv"][0] != "/usr/bin/tmux" or "new-session" not in result["tmux_argv"]:
        _fail("outer result tmux command drift")
    _validate_environment(result["environment"])
    _validate_data(result["data_roles"])
    _validate_targets(result["targets"])
    if result["snapshot_monotonic_ns"] <= 0 or not result["snapshot_utc"].endswith("+00:00"):
        _fail("outer snapshot timing identity drift")
    if result["status"] not in {"TMUX_ACCEPTED", "PERMANENT_FAIL"} or result["classification"] != result["status"]:
        _fail("outer terminal state drift")
    if result["status"] == "TMUX_ACCEPTED":
        if result["return_code"] != 0 or result["failure_class"] != "NONE" or result["production_training_authorized"] is not False or result["real_process_launch_authorized"] is not False or result["training_certified"] is not False:
            _fail("TMUX acceptance semantics drift")
    else:
        if result["failure_class"] not in {"RUNNER_NONZERO_EXIT", "RUNNER_EXCEPTION", "RUNNER_TIMEOUT", "RUNNER_SIGNAL", "RUNNER_SCHEMA_FAILURE", "RUNNER_MUTATED_INVOCATION", "PERSISTENCE_FAILURE"} or result["production_training_authorized"] is not False or result["real_process_launch_authorized"] is not False:
            _fail("outer failure semantics drift")
    trace = result["state_sequence"]
    if type(trace) is not list or len(trace) != 4:
        _fail("outer state sequence drift")
    for index, row in enumerate(trace):
        row = _exact(row, {"sequence", "state", "predecessor_sha256"}, f"result.state_sequence[{index}]")
        if row["sequence"] != index or (index == 0 and row["predecessor_sha256"] is not None) or (index > 0 and row["predecessor_sha256"] != _digest(trace[index - 1])):
            _fail("outer state predecessor drift")
    if result["state_transition_sha256"] != _digest(trace) or result["outer_invocation_count"] != 1 or result["tmux_new_session_count"] != 1 or result["shell_used"] is not False:
        _fail("outer invocation semantics drift")
    body = _copy(result)
    del body["aggregate_result_sha256"]
    if _digest(body) != result["aggregate_result_sha256"]:
        _fail("outer result aggregate drift")
    if expected_descriptor is not None:
        descriptor = validate_outer_descriptor(expected_descriptor)
        if result["descriptor_sha256"] != descriptor["aggregate_sha256"] or result["process_descriptor_sha256"] != descriptor["process_descriptor_sha256"]:
            _fail("outer result descriptor binding drift")
        if result["authorization_binding_sha256"] != descriptor["authorization_binding_sha256"] or result["outer_evidence_root"] != descriptor["outer_evidence_root"]:
            _fail("outer result authorization or root binding drift")
        if result["tmux_argv"] != descriptor["tmux_argv"] or result["environment"] != descriptor["environment"] or result["data_roles"] != descriptor["data_roles"]:
            _fail("outer result invocation binding drift")
    return _copy(result)


def _inventory(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in sorted(set(OUTER_FILE_NAMES) - {"artifact_inventory.json", "completion.json"}):
        path = root / name
        observed = _entry._regular_file(path, name, expected_mode=0o600, expected_nlink=1)
        raw = _entry._read_stable_file(path, name, mode=0o600)
        rows.append({"relative_path": name, "size_bytes": len(raw), "sha256": _sha(raw), "mode": stat.S_IMODE(observed.st_mode), "nlink": observed.st_nlink})
    return rows


def run_outer_once(
    descriptor: Mapping[str, Any],
    tmux_runner: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Claim the outer target, issue one tmux request, and snapshot it."""

    checked = validate_outer_descriptor(dict(descriptor))
    # This read-only receipt check is the authorization gate immediately before
    # any outer target is claimed.
    _entry.validate_consumed_authorization_receipt(
        checked["authorization_receipt_path"],
        authorization_binding=checked["authorization_binding"],
        authorization_path=checked["authorization_path"],
    )
    for target in checked["targets"].values():
        path = Path(target)
        if path.exists() or path.is_symlink():
            _fail("a production target already exists")
        _entry._directory(path.parent, "target parent")
    root, receipt = _claim_root(checked)
    try:
        _publish(root, "invocation.json", {"schema_version": OUTER_SCHEMA_VERSION, "status": "PREPARED", "descriptor_sha256": checked["aggregate_sha256"], "tmux_new_session_count": 0})
    except Exception as exc:
        _record_persistence_failure(root, checked, "outer_invocation", exc)
        if isinstance(exc, TrainingOuterError):
            raise
        raise TrainingOuterError("outer invocation evidence could not be published") from exc
    argv = list(checked["tmux_argv"])
    cwd = checked["cwd"]
    environment = _copy(checked["environment"]["variables"])
    failure_class = "NONE"
    try:
        if tmux_runner is not None:
            try:
                inspect.signature(tmux_runner).bind(argv, cwd, environment, _copy(checked))
            except (TypeError, ValueError) as exc:
                failure_class = "RUNNER_SCHEMA_FAILURE"
                raise TrainingOuterError("tmux runner signature is not the closed ABI") from exc
            original = (list(argv), cwd, _copy(environment))
            returned = tmux_runner(argv, cwd, environment, _copy(checked))
            if argv != original[0] or cwd != original[1] or environment != original[2]:
                failure_class = "RUNNER_MUTATED_INVOCATION"
                raise TrainingOuterError("tmux runner mutated invocation inputs")
            payload = _validate_tmux_result(returned)
        else:
            payload = _validate_tmux_result(_invoke_real(checked))
        if payload["return_code"] != 0:
            failure_class = "RUNNER_NONZERO_EXIT"
        elif payload["pid"] <= 0:
            failure_class = "RUNNER_SCHEMA_FAILURE"
    except TrainingOuterError:
        if failure_class == "NONE":
            failure_class = "RUNNER_SCHEMA_FAILURE"
        payload = {"return_code": 5, "stdout": b"", "stderr": b"", "pid": 0}
    except TimeoutError:
        failure_class = "RUNNER_TIMEOUT"
        payload = {"return_code": 124, "stdout": b"", "stderr": b"", "pid": 0}
    except KeyboardInterrupt:
        failure_class = "RUNNER_SIGNAL"
        payload = {"return_code": 130, "stdout": b"", "stderr": b"", "pid": 0}
    except Exception:
        failure_class = "RUNNER_EXCEPTION"
        payload = {"return_code": 2, "stdout": b"", "stderr": b"", "pid": 0}
    try:
        snapshot = {"schema_version": OUTER_SCHEMA_VERSION, "monotonic_ns": time.monotonic_ns(), "utc": _datetime.datetime.now(_datetime.timezone.utc).isoformat(), "tmux_return_code": payload["return_code"], "tmux_pid": payload["pid"], "stdout_size_bytes": len(payload["stdout"]), "stdout_sha256": _sha(payload["stdout"]), "stderr_size_bytes": len(payload["stderr"]), "stderr_sha256": _sha(payload["stderr"])}
        _publish(root, "snapshot.json", snapshot)
        stream_receipt = {
            "schema_version": OUTER_SCHEMA_VERSION,
            "kind": "T6B_OUTER_STREAMS",
            "descriptor_sha256": checked["aggregate_sha256"],
            "pid": payload["pid"],
            "return_code": payload["return_code"],
            "stdout_size_bytes": len(payload["stdout"]),
            "stdout_sha256": _sha(payload["stdout"]),
            "stderr_size_bytes": len(payload["stderr"]),
            "stderr_sha256": _sha(payload["stderr"]),
        }
        stream_receipt_raw = _canonical(stream_receipt)
        stream_receipt_path = Path(checked["outer_receipt_path"]).parent / f".{Path(checked['outer_evidence_root']).name}{OUTER_STREAM_RECEIPT_SUFFIX}"
        _entry._write_new(stream_receipt_path, stream_receipt_raw, "outer stream receipt")
        _entry._fsync_directory(stream_receipt_path.parent, "outer stream receipt parent")
        result = _make_result(checked, payload, snapshot, receipt, failure_class)
        validated = validate_outer_result(result, checked)
        _entry._write_new(root / "stdout.bin", payload["stdout"], "outer stdout")
        _entry._write_new(root / "stderr.bin", payload["stderr"], "outer stderr")
        _publish(root, "result.json", validated)
        rows = _inventory(root)
        inventory = {"schema_version": OUTER_SCHEMA_VERSION, "excluded": ["artifact_inventory.json", "completion.json"], "files": rows, "canonical_inventory_sha256": _digest(rows)}
        _publish(root, "artifact_inventory.json", inventory)
        completion = {"schema_version": OUTER_SCHEMA_VERSION, "status": validated["status"], "classification": validated["classification"], "exit_code": validated["return_code"], "descriptor_sha256": checked["aggregate_sha256"], "result_sha256": _sha(_canonical(validated)), "inventory_sha256": _sha(_canonical(inventory)), "snapshot_sha256": _sha(_canonical(snapshot)), "stdout_size_bytes": validated["stdout_size_bytes"], "stdout_sha256": validated["stdout_sha256"], "stderr_size_bytes": validated["stderr_size_bytes"], "stderr_sha256": validated["stderr_sha256"], "receipt_sha256": receipt["receipt_sha256"], "stream_receipt_sha256": _sha(stream_receipt_raw)}
        _publish(root, "completion.json", completion)
        _entry._fsync_directory(root, "outer evidence root")
        return validated
    except Exception as exc:
        _record_persistence_failure(root, checked, "outer_terminal", exc)
        if isinstance(exc, TrainingOuterError):
            raise
        raise TrainingOuterError("outer terminal evidence could not be published") from exc


def classify_outer(outer_evidence_root: str | os.PathLike[str]) -> str:
    try:
        root = Path(_absolute(os.fspath(outer_evidence_root), "outer_evidence_root"))
        if _has_valid_persistence_failure_marker(root):
            return "PERMANENT_FAIL"
        if not root.exists():
            return "ABSENT"
        if root.is_symlink() or not root.is_dir():
            return "UNKNOWN"
        if {path.name for path in root.iterdir()} != set(OUTER_FILE_NAMES):
            return "UNKNOWN"
        result_raw = _entry._read_stable_file(root / "result.json", "outer result")
        result, _ = _entry._read_json(root / "result.json", "outer result")
        validate_outer_result(result)
        if result["outer_evidence_root"] != str(root) or result["outer_receipt_path"] != str(root.parent / f".{root.name}{OUTER_RECEIPT_SUFFIX}"):
            return "UNKNOWN"
        claim, claim_raw = _entry._read_json(Path(result["outer_receipt_path"]), "outer receipt")
        if set(claim) != {"schema_version", "kind", "descriptor_sha256", "descriptor", "training_run_id", "nonce", "status"} or claim_raw != _canonical(claim):
            return "UNKNOWN"
        claim_descriptor = validate_outer_descriptor(claim["descriptor"])
        expected_claim = {
            "schema_version": OUTER_SCHEMA_VERSION,
            "kind": "T6B_OUTER_CLAIM",
            "descriptor_sha256": result["descriptor_sha256"],
            "descriptor": claim_descriptor,
            "training_run_id": result["training_run_id"],
            "nonce": result["nonce"],
            "status": "CLAIMED",
        }
        if claim != expected_claim or claim_descriptor["aggregate_sha256"] != result["descriptor_sha256"]:
            return "UNKNOWN"
        validate_outer_result(result, claim_descriptor)
        stream_receipt_path = root.parent / f".{root.name}{OUTER_STREAM_RECEIPT_SUFFIX}"
        stream_receipt, stream_receipt_raw = _entry._read_json(stream_receipt_path, "outer stream receipt")
        _entry._regular_file(stream_receipt_path, "outer stream receipt", expected_mode=0o600, expected_nlink=1)
        expected_stream_receipt = {
            "schema_version": OUTER_SCHEMA_VERSION,
            "kind": "T6B_OUTER_STREAMS",
            "descriptor_sha256": result["descriptor_sha256"],
            "pid": result["pid"],
            "return_code": result["return_code"],
            "stdout_size_bytes": result["stdout_size_bytes"],
            "stdout_sha256": result["stdout_sha256"],
            "stderr_size_bytes": result["stderr_size_bytes"],
            "stderr_sha256": result["stderr_sha256"],
        }
        if stream_receipt != expected_stream_receipt or stream_receipt_raw != _canonical(stream_receipt):
            return "UNKNOWN"
        stdout = _entry._read_stable_file(root / "stdout.bin", "outer stdout")
        stderr = _entry._read_stable_file(root / "stderr.bin", "outer stderr")
        if len(stdout) != result["stdout_size_bytes"] or _sha(stdout) != result["stdout_sha256"] or len(stderr) != result["stderr_size_bytes"] or _sha(stderr) != result["stderr_sha256"]:
            return "UNKNOWN"
        snapshot, snapshot_raw = _entry._read_json(root / "snapshot.json", "outer snapshot")
        if set(snapshot) != {"schema_version", "monotonic_ns", "utc", "tmux_return_code", "tmux_pid", "stdout_size_bytes", "stdout_sha256", "stderr_size_bytes", "stderr_sha256"} or snapshot["schema_version"] != OUTER_SCHEMA_VERSION:
            return "UNKNOWN"
        if snapshot["monotonic_ns"] != result["snapshot_monotonic_ns"] or snapshot["utc"] != result["snapshot_utc"] or snapshot["stdout_size_bytes"] != result["stdout_size_bytes"] or snapshot["stdout_sha256"] != result["stdout_sha256"] or snapshot["stderr_size_bytes"] != result["stderr_size_bytes"] or snapshot["stderr_sha256"] != result["stderr_sha256"]:
            return "UNKNOWN"
        if snapshot["tmux_return_code"] != result["return_code"] or snapshot["tmux_pid"] != result["pid"]:
            return "UNKNOWN"
        invocation, invocation_raw = _entry._read_json(root / "invocation.json", "outer invocation")
        if invocation_raw != _canonical(invocation) or invocation != {"schema_version": OUTER_SCHEMA_VERSION, "status": "PREPARED", "descriptor_sha256": result["descriptor_sha256"], "tmux_new_session_count": 0}:
            return "UNKNOWN"
        inventory, _ = _entry._read_json(root / "artifact_inventory.json", "outer inventory")
        if type(inventory) is not dict or set(inventory) != {"schema_version", "excluded", "files", "canonical_inventory_sha256"} or inventory["canonical_inventory_sha256"] != _digest(inventory["files"]):
            return "UNKNOWN"
        if inventory["files"] != _inventory(root):
            return "UNKNOWN"
        completion, _ = _entry._read_json(root / "completion.json", "outer completion")
        receipt_raw = _entry._read_stable_file(Path(result["outer_receipt_path"]), "outer receipt", mode=0o600)
        if completion != {"schema_version": OUTER_SCHEMA_VERSION, "status": result["status"], "classification": result["classification"], "exit_code": result["return_code"], "descriptor_sha256": result["descriptor_sha256"], "result_sha256": _sha(result_raw), "inventory_sha256": _sha(_canonical(inventory)), "snapshot_sha256": _sha(snapshot_raw), "stdout_size_bytes": result["stdout_size_bytes"], "stdout_sha256": result["stdout_sha256"], "stderr_size_bytes": result["stderr_size_bytes"], "stderr_sha256": result["stderr_sha256"], "receipt_sha256": _sha(receipt_raw), "stream_receipt_sha256": _sha(stream_receipt_raw) }:
            return "UNKNOWN"
        return "LAUNCH_ACCEPTED" if result["status"] == "TMUX_ACCEPTED" else "PERMANENT_FAIL"
    except Exception:
        return "UNKNOWN"


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2 or args[0] != "--descriptor":
        return 2
    descriptor_path = Path(_absolute(args[1], "outer descriptor path"))
    descriptor, _ = _entry._read_json(descriptor_path, "outer descriptor")
    result = run_outer_once(descriptor)
    sys.stdout.buffer.write(_canonical(result) + b"\n")
    sys.stdout.buffer.flush()
    return 0 if result["status"] == "TMUX_ACCEPTED" else result["return_code"]


if __name__ == "__main__":
    raise SystemExit(main())
