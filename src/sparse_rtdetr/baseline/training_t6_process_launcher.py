"""T6B exactly-once child process boundary.

The process layer owns process evidence and raw child stdout/stderr.  It has
one injected synchronous fake seam for CPU tests and one production
``subprocess.Popen`` path.  Neither path retries, resumes, overwrites, or
uses shell evaluation.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

from sparse_rtdetr.baseline import training_t6_entry as _entry


PROCESS_SCHEMA_VERSION = 1
PROCESS_CONTRACT_ID = _entry.T6B_CONTRACT_ID
PROCESS_LAUNCHER_ID = "rtdetrv2_r18_visdrone_training_t6_process_launcher_v1"
PROCESS_MODE = "production"
PROCESS_MODULE_RELATIVE_PATH = _entry.PROCESS_MODULE_RELATIVE_PATH
PROCESS_RECEIPT_SUFFIX = _entry.PROCESS_RECEIPT_SUFFIX
PROCESS_STREAM_RECEIPT_SUFFIX = ".t6b-process-streams.json"
PERSISTENCE_FAILURE_MARKER_SUFFIX = ".t6b-process-persistence-failure.json"
PROCESS_EVIDENCE_FILE_NAMES = (
    "invocation.json",
    "stdout.bin",
    "stderr.bin",
    "result.json",
    "artifact_inventory.json",
    "completion.json",
)
JSON_FILE_NAMES = (
    "invocation.json",
    "entry_consumption.json",
    "entry_invocation.json",
    "entry_result.json",
    "entry_completion.json",
    "result.json",
    "artifact_inventory.json",
    "completion.json",
)
BINARY_FILE_NAMES = ("stdout.bin", "stderr.bin")
RUNNER_PARAMETERS = ("argv", "cwd", "environment", "entry_descriptor")
RUNNER_RETURN_KEYS = {"return_code", "stdout", "stderr", "pid", "entry_result"}
FAILURE_CLASSES = (
    "NONE",
    "RUNNER_NONZERO_EXIT",
    "RUNNER_EXCEPTION",
    "RUNNER_TIMEOUT",
    "RUNNER_SIGNAL",
    "RUNNER_SCHEMA_FAILURE",
    "RUNNER_MUTATED_INVOCATION",
    "ENTRY_TERMINAL_FAILURE",
    "PERSISTENCE_FAILURE",
)
FAILURE_RETURN_CODES = {
    "RUNNER_EXCEPTION": 2,
    "RUNNER_TIMEOUT": 124,
    "RUNNER_SIGNAL": 130,
    "RUNNER_SCHEMA_FAILURE": 5,
    "RUNNER_MUTATED_INVOCATION": 6,
    "ENTRY_TERMINAL_FAILURE": 5,
    "PERSISTENCE_FAILURE": 7,
}

DESCRIPTOR_KEYS = {
    "schema_version",
    "contract_id",
    "launcher_id",
    "mode",
    "training_run_id",
    "nonce",
    "tmux_session_name",
    "entry_descriptor",
    "entry_evidence_root",
    "entry_descriptor_sha256",
    "argv",
    "cwd",
    "environment",
    "process_evidence_root",
    "process_receipt_path",
    "runner_contract",
    "state_machine",
    "aggregate_sha256",
}
RESULT_KEYS = {
    "schema_version",
    "contract_id",
    "launcher_id",
    "mode",
    "training_run_id",
    "nonce",
    "descriptor_sha256",
    "entry_descriptor_sha256",
    "authorization_binding_sha256",
    "argv",
    "cwd",
    "environment",
    "process_evidence_root",
    "entry_evidence_root",
    "process_receipt_path",
    "pid",
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
    "child_invocation_count",
    "shell_used",
    "production_training_authorized",
    "real_process_launch_authorized",
    "aggregate_result_sha256",
}

__all__ = (
    "TrainingProcessError",
    "build_production_process_descriptor",
    "validate_process_descriptor",
    "run_process_once",
    "validate_process_result",
    "classify_process",
)


class TrainingProcessError(ValueError):
    """Raised when the T6B process boundary or evidence is invalid."""


def _fail(message: str) -> None:
    raise TrainingProcessError(message)


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
        raise TrainingProcessError("value is not canonical JSON") from exc


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
        raise TrainingProcessError(str(exc)) from exc


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
        _fail("process state trace must start at DESIGN_ONLY")
    trace = [{"sequence": 0, "state": states[0], "predecessor_sha256": None}]
    for index, state in enumerate(states[1:], 1):
        if state not in _entry.STATE_SEQUENCE:
            _fail("process state trace contains an unknown state")
        trace.append({"sequence": index, "state": state, "predecessor_sha256": _digest(trace[-1])})
    return trace


def _module_identity(root: Path) -> dict[str, Any]:
    return _entry._file_identity(root, PROCESS_MODULE_RELATIVE_PATH, "T6B process module")


def _validate_runner_contract(value: Any) -> dict[str, Any]:
    contract = _exact(value, {"parameters", "return_keys", "synchronous", "max_calls", "argv_sequence", "shell"}, "runner_contract")
    if contract["parameters"] != list(RUNNER_PARAMETERS) or contract["return_keys"] != sorted(RUNNER_RETURN_KEYS) or contract["synchronous"] is not True or contract["max_calls"] != 1 or contract["argv_sequence"] is not True or contract["shell"] is not False:
        _fail("runner contract drift")
    return _copy(contract)


def _validate_descriptor_shape(value: Any) -> dict[str, Any]:
    _assert_builtin(value, "process descriptor")
    descriptor = _exact(value, DESCRIPTOR_KEYS, "process descriptor")
    if descriptor["schema_version"] != PROCESS_SCHEMA_VERSION or descriptor["contract_id"] != PROCESS_CONTRACT_ID or descriptor["launcher_id"] != PROCESS_LAUNCHER_ID or descriptor["mode"] != PROCESS_MODE:
        _fail("process descriptor identity drift")
    _string(descriptor["training_run_id"], "descriptor.training_run_id")
    _string(descriptor["nonce"], "descriptor.nonce")
    _string(descriptor["tmux_session_name"], "descriptor.tmux_session_name")
    try:
        entry = _entry.validate_entry_descriptor(descriptor["entry_descriptor"])
    except _entry.TrainingEntryError as exc:
        raise TrainingProcessError(str(exc)) from exc
    if descriptor["entry_descriptor_sha256"] != entry["aggregate_sha256"]:
        _fail("process entry descriptor digest drift")
    if descriptor["training_run_id"] != entry["training_run_id"] or descriptor["nonce"] != entry["nonce"] or descriptor["tmux_session_name"] != entry["tmux_session_name"]:
        _fail("process run identity drift")
    if descriptor["entry_evidence_root"] != entry["evidence_root"]:
        _fail("process entry evidence root drift")
    argv = descriptor["argv"]
    if type(argv) is not list or not argv or any(type(item) is not str or not item for item in argv) or argv != entry["argv"]:
        _fail("process argv drift")
    cwd = _absolute(descriptor["cwd"], "descriptor.cwd")
    if cwd != entry["cwd"]:
        _fail("process cwd drift")
    environment = descriptor["environment"]
    if type(environment) is not dict or environment != entry["environment"]["variables"]:
        _fail("process environment drift")
    root = _absolute(descriptor["process_evidence_root"], "descriptor.process_evidence_root")
    if root != entry["process_evidence_root"]:
        _fail("process evidence root drift")
    receipt = _absolute(descriptor["process_receipt_path"], "descriptor.process_receipt_path")
    expected_receipt = str(Path(root).parent / f".{Path(root).name}{PROCESS_RECEIPT_SUFFIX}")
    if receipt != expected_receipt:
        _fail("process receipt path drift")
    _validate_runner_contract(descriptor["runner_contract"])
    machine = _exact(descriptor["state_machine"], {"states", "trace", "exactly_once", "durable_evidence_required"}, "process state_machine")
    if machine["states"] != list(_entry.STATE_SEQUENCE) or machine["exactly_once"] is not True or machine["durable_evidence_required"] is not True:
        _fail("process state machine policy drift")
    trace = machine["trace"]
    if type(trace) is not list or len(trace) != 4 or [row.get("state") for row in trace] != ["DESIGN_ONLY", "OWNER_AUTHORIZED", "PREFLIGHT_PASS", "LAUNCH_ACCEPTED"]:
        _fail("process prepared trace drift")
    for index, row in enumerate(trace):
        row = _exact(row, {"sequence", "state", "predecessor_sha256"}, f"process trace[{index}]")
        if row["sequence"] != index or (index == 0 and row["predecessor_sha256"] is not None) or (index > 0 and row["predecessor_sha256"] != _digest(trace[index - 1])):
            _fail("process state predecessor drift")
    aggregate = _string(descriptor["aggregate_sha256"], "descriptor.aggregate_sha256")
    if len(aggregate) != 64 or any(char not in "0123456789abcdef" for char in aggregate):
        _fail("process descriptor aggregate is not SHA-256")
    body = _copy(descriptor)
    del body["aggregate_sha256"]
    if _digest(body) != aggregate:
        _fail("process descriptor aggregate drift")
    return _copy(descriptor)


def build_production_process_descriptor(entry_descriptor: Mapping[str, Any]) -> dict[str, Any]:
    entry = _entry.validate_entry_descriptor(dict(entry_descriptor))
    root = Path(entry["repository"]["repo_root"])
    source = _module_identity(root)
    process_root = entry["process_evidence_root"]
    receipt = str(Path(process_root).parent / f".{Path(process_root).name}{PROCESS_RECEIPT_SUFFIX}")
    body = {
        "schema_version": PROCESS_SCHEMA_VERSION,
        "contract_id": PROCESS_CONTRACT_ID,
        "launcher_id": PROCESS_LAUNCHER_ID,
        "mode": PROCESS_MODE,
        "training_run_id": entry["training_run_id"],
        "nonce": entry["nonce"],
        "tmux_session_name": entry["tmux_session_name"],
        "entry_descriptor": entry,
        "entry_evidence_root": entry["evidence_root"],
        "entry_descriptor_sha256": entry["aggregate_sha256"],
        "argv": list(entry["argv"]),
        "cwd": entry["cwd"],
        "environment": _copy(entry["environment"]["variables"]),
        "process_evidence_root": process_root,
        "process_receipt_path": receipt,
        "runner_contract": {"parameters": list(RUNNER_PARAMETERS), "return_keys": sorted(RUNNER_RETURN_KEYS), "synchronous": True, "max_calls": 1, "argv_sequence": True, "shell": False},
        "state_machine": {"states": list(_entry.STATE_SEQUENCE), "trace": _state_trace(["DESIGN_ONLY", "OWNER_AUTHORIZED", "PREFLIGHT_PASS", "LAUNCH_ACCEPTED"]), "exactly_once": True, "durable_evidence_required": True},
    }
    return _validate_descriptor_shape({**body, "aggregate_sha256": _digest(body)})


def validate_process_descriptor(value: Any) -> dict[str, Any]:
    return _validate_descriptor_shape(value)


def _claim_process_root(descriptor: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    root = Path(descriptor["process_evidence_root"])
    receipt = Path(descriptor["process_receipt_path"])
    parent = root.parent
    try:
        _entry._canonical_root(str(parent), "process evidence parent")
    except _entry.TrainingEntryError as exc:
        raise TrainingProcessError(str(exc)) from exc
    stream_receipt = root.parent / f".{root.name}{PROCESS_STREAM_RECEIPT_SUFFIX}"
    if (
        root.exists()
        or root.is_symlink()
        or receipt.exists()
        or receipt.is_symlink()
        or stream_receipt.exists()
        or stream_receipt.is_symlink()
    ):
        _fail("process evidence target or receipt already exists")
    receipt_payload = {
        "schema_version": PROCESS_SCHEMA_VERSION,
        "kind": "T6B_PROCESS_CLAIM",
        "descriptor_sha256": descriptor["aggregate_sha256"],
        "descriptor": _copy(descriptor),
        "training_run_id": descriptor["training_run_id"],
        "nonce": descriptor["nonce"],
        "status": "CLAIMED",
    }
    receipt_raw = _canonical(receipt_payload)
    _entry._write_new(receipt, receipt_raw, "process receipt")
    _entry._fsync_directory(parent, "process receipt parent")
    try:
        os.mkdir(root, 0o700)
        observed = root.lstat()
        if stat.S_IMODE(observed.st_mode) != 0o700 or observed.st_nlink < 2 or observed.st_uid != os.getuid() or observed.st_gid != os.getgid() or not stat.S_ISDIR(observed.st_mode):
            _fail("process evidence root metadata drift")
        _entry._fsync_directory(parent, "process evidence parent")
    except OSError as exc:
        raise TrainingProcessError("process evidence root could not be created") from exc
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
        "schema_version": PROCESS_SCHEMA_VERSION,
        "kind": "T6B_PROCESS_PERSISTENCE_FAILURE",
        "descriptor_sha256": descriptor["aggregate_sha256"],
        "training_run_id": descriptor["training_run_id"],
        "nonce": descriptor["nonce"],
        "process_evidence_root": str(root),
        "phase": phase,
        "exception_type": type(error).__name__,
        "exception_message": str(error),
        "failure_class": "PERSISTENCE_FAILURE",
        "status": "PERMANENT_FAIL",
    }
    marker = _persistence_failure_marker_path(root)
    try:
        if not marker.exists() and not marker.is_symlink():
            _entry._write_new(marker, _canonical(payload), "process persistence failure marker")
            _entry._fsync_directory(marker.parent, "process persistence failure marker parent")
    except Exception:
        return


def _has_valid_persistence_failure_marker(root: Path) -> bool:
    marker = _persistence_failure_marker_path(root)
    if not marker.exists() and not marker.is_symlink():
        return False
    try:
        _entry._regular_file(marker, "process persistence failure marker", expected_mode=0o600, expected_nlink=1)
        value, raw = _entry._read_json(marker, "process persistence failure marker")
        expected = {
            "schema_version": PROCESS_SCHEMA_VERSION,
            "kind": "T6B_PROCESS_PERSISTENCE_FAILURE",
            "descriptor_sha256": value.get("descriptor_sha256"),
            "training_run_id": value.get("training_run_id"),
            "nonce": value.get("nonce"),
            "process_evidence_root": str(root),
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


def _binary_identity(raw: bytes) -> tuple[int, str]:
    return len(raw), _sha(raw)


def _validate_runner_payload(value: Any) -> dict[str, Any]:
    payload = _exact(value, RUNNER_RETURN_KEYS, "runner result")
    if type(payload["return_code"]) is not int or not 0 <= payload["return_code"] <= 255:
        _fail("runner return code drift")
    for field in ("stdout", "stderr"):
        if type(payload[field]) is not bytes:
            _fail(f"runner {field} must be raw bytes")
    _integer(payload["pid"], "runner pid", minimum=0)
    if payload["entry_result"] is not None and type(payload["entry_result"]) is not dict:
        _fail("runner entry_result must be a builtin dict or null")
    if payload["entry_result"] is not None:
        _assert_builtin(payload["entry_result"], "runner entry_result")
    return _copy(payload)


def _parse_entry_result_stdout(raw: bytes) -> dict[str, Any] | None:
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        return None
    payload = raw[:-1]
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return None
    if type(value) is not dict:
        return None
    try:
        _assert_builtin(value, "entry result stdout")
        if _canonical(value) != payload:
            return None
    except TrainingProcessError:
        return None
    return value


def _invoke_real(descriptor: dict[str, Any]) -> dict[str, Any]:
    process = subprocess.Popen(
        list(descriptor["argv"]),
        cwd=descriptor["cwd"],
        env=dict(descriptor["environment"]),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
    )
    stdout, stderr = process.communicate()
    return {"return_code": process.returncode, "stdout": stdout, "stderr": stderr, "pid": process.pid, "entry_result": _parse_entry_result_stdout(stdout)}


def _failure_result(descriptor: dict[str, Any], *, payload: dict[str, Any] | None, failure_class: str, entry_result: Any = None) -> dict[str, Any]:
    stdout = b"" if payload is None else payload["stdout"]
    stderr = b"" if payload is None else payload["stderr"]
    return_code = FAILURE_RETURN_CODES.get(failure_class, 5) if payload is None else payload["return_code"]
    if failure_class == "RUNNER_NONZERO_EXIT" and payload is not None and payload["return_code"] == 0:
        return_code = 5
    stdout_size, stdout_sha = _binary_identity(stdout)
    stderr_size, stderr_sha = _binary_identity(stderr)
    return {
        "return_code": return_code,
        "stdout": stdout,
        "stderr": stderr,
        "pid": 0 if payload is None else payload["pid"],
        "entry_result": entry_result if entry_result is not None else (None if payload is None else payload["entry_result"]),
        "failure_class": failure_class,
        "stdout_size_bytes": stdout_size,
        "stdout_sha256": stdout_sha,
        "stderr_size_bytes": stderr_size,
        "stderr_sha256": stderr_sha,
    }


def _make_result(descriptor: dict[str, Any], payload: dict[str, Any], receipt: dict[str, Any], failure_class: str) -> dict[str, Any]:
    entry_result = payload["entry_result"]
    entry_result_sha = None if entry_result is None else _digest(entry_result)
    if entry_result is not None:
        try:
            entry_result = _entry.validate_entry_result(entry_result, descriptor["entry_descriptor"])
            consumption, _ = _entry._read_json(
                Path(entry_result["training_evidence_root"]) / "entry_consumption.json",
                "entry consumption",
            )
            if type(consumption.get("process_pid")) is not int or consumption["process_pid"] != payload["pid"]:
                raise TrainingProcessError("child PID is not bound to entry evidence")
        except Exception as exc:
            failure_class = "RUNNER_SCHEMA_FAILURE"
            entry_result = None
            entry_result_sha = None
    terminal_success = payload["return_code"] == 0 and payload["pid"] > 0 and failure_class == "NONE" and type(entry_result) is dict and entry_result.get("status") == "TERMINAL_COMPLETE"
    if terminal_success:
        status = "TERMINAL_COMPLETE"
        classification = "TERMINAL_COMPLETE"
        effective_failure = "NONE"
        authorized = True
    else:
        status = "PERMANENT_FAIL"
        classification = "PERMANENT_FAIL"
        effective_failure = "ENTRY_TERMINAL_FAILURE" if failure_class == "NONE" else failure_class
        authorized = False
    effective_return_code = payload["return_code"]
    if not terminal_success and effective_return_code == 0:
        effective_return_code = FAILURE_RETURN_CODES.get(effective_failure, 5)
    trace = _state_trace(["DESIGN_ONLY", "OWNER_AUTHORIZED", "PREFLIGHT_PASS", "LAUNCH_ACCEPTED", "RUNNING", status])
    stdout_size, stdout_sha = _binary_identity(payload["stdout"])
    stderr_size, stderr_sha = _binary_identity(payload["stderr"])
    body = {
        "schema_version": PROCESS_SCHEMA_VERSION,
        "contract_id": PROCESS_CONTRACT_ID,
        "launcher_id": PROCESS_LAUNCHER_ID,
        "mode": PROCESS_MODE,
        "training_run_id": descriptor["training_run_id"],
        "nonce": descriptor["nonce"],
        "descriptor_sha256": descriptor["aggregate_sha256"],
        "entry_descriptor_sha256": descriptor["entry_descriptor_sha256"],
        "entry_evidence_root": descriptor["entry_evidence_root"],
        "authorization_binding_sha256": descriptor["entry_descriptor"]["authorization_binding_sha256"],
        "argv": list(descriptor["argv"]),
        "cwd": descriptor["cwd"],
        "environment": _copy(descriptor["environment"]),
        "process_evidence_root": descriptor["process_evidence_root"],
        "process_receipt_path": descriptor["process_receipt_path"],
        "pid": payload["pid"],
        "return_code": effective_return_code,
        "stdout_size_bytes": stdout_size,
        "stdout_sha256": stdout_sha,
        "stderr_size_bytes": stderr_size,
        "stderr_sha256": stderr_sha,
        "stdout_anchor_relative_path": "stdout.bin",
        "stderr_anchor_relative_path": "stderr.bin",
        "entry_result": _copy(entry_result),
        "entry_result_sha256": entry_result_sha,
        "status": status,
        "classification": classification,
        "failure_class": effective_failure,
        "state_sequence": trace,
        "state_transition_sha256": _digest(trace),
        "child_invocation_count": 1,
        "shell_used": False,
        "production_training_authorized": authorized,
        "real_process_launch_authorized": authorized,
    }
    return {**body, "aggregate_result_sha256": _digest(body)}


def validate_process_result(value: Any, expected_descriptor: Any | None = None) -> dict[str, Any]:
    result = _exact(value, RESULT_KEYS, "process result")
    _assert_builtin(result, "process result")
    if result["schema_version"] != PROCESS_SCHEMA_VERSION or result["contract_id"] != PROCESS_CONTRACT_ID or result["launcher_id"] != PROCESS_LAUNCHER_ID or result["mode"] != PROCESS_MODE:
        _fail("process result identity drift")
    _string(result["training_run_id"], "result.training_run_id")
    _string(result["nonce"], "result.nonce")
    for field in ("descriptor_sha256", "entry_descriptor_sha256", "authorization_binding_sha256", "stdout_sha256", "stderr_sha256", "state_transition_sha256", "aggregate_result_sha256"):
        value = result[field]
        if type(value) is not str or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            _fail(f"result.{field} is not SHA-256")
    _integer(result["pid"], "result.pid", minimum=0)
    _integer(result["return_code"], "result.return_code", minimum=0, maximum=255)
    _integer(result["stdout_size_bytes"], "result.stdout_size_bytes", minimum=0)
    _integer(result["stderr_size_bytes"], "result.stderr_size_bytes", minimum=0)
    if result["stdout_anchor_relative_path"] != "stdout.bin" or result["stderr_anchor_relative_path"] != "stderr.bin":
        _fail("result binary anchor path drift")
    _absolute(result["cwd"], "result.cwd")
    _absolute(result["process_evidence_root"], "result.process_evidence_root")
    _absolute(result["entry_evidence_root"], "result.entry_evidence_root")
    _absolute(result["process_receipt_path"], "result.process_receipt_path")
    if type(result["argv"]) is not list or any(type(item) is not str for item in result["argv"]):
        _fail("result argv drift")
    if type(result["environment"]) is not dict or any(type(key) is not str or type(value) is not str for key, value in result["environment"].items()):
        _fail("result environment drift")
    if result["entry_result"] is not None:
        if type(result["entry_result"] ) is not dict:
            _fail("result entry_result drift")
        try:
            _entry.validate_entry_result(result["entry_result"], expected_descriptor["entry_descriptor"] if expected_descriptor is not None else None)
        except _entry.TrainingEntryError as exc:
            _fail(str(exc))
        if result["entry_result_sha256"] != _digest(result["entry_result"]):
            _fail("result entry_result digest drift")
        if result["entry_result"]["training_evidence_root"] != result["entry_evidence_root"]:
            _fail("result entry evidence root binding drift")
    elif result["entry_result_sha256"] is not None:
        _fail("result null entry digest drift")
    if result["status"] not in {"TERMINAL_COMPLETE", "PERMANENT_FAIL"} or result["classification"] != result["status"]:
        _fail("process terminal state drift")
    if result["status"] == "TERMINAL_COMPLETE":
        if result["return_code"] != 0 or result["failure_class"] != "NONE" or result["entry_result"] is None or result["entry_result"].get("status") != "TERMINAL_COMPLETE" or result["production_training_authorized"] is not True or result["real_process_launch_authorized"] is not True:
            _fail("process success semantics drift")
    else:
        if result["failure_class"] not in FAILURE_CLASSES[1:] or result["production_training_authorized"] is not False or result["real_process_launch_authorized"] is not False:
            _fail("process failure semantics drift")
        if result["entry_result"] is not None and result["entry_result"].get("status") == "TERMINAL_COMPLETE":
            _fail("process failure contradicts terminal entry success")
    trace = result["state_sequence"]
    if type(trace) is not list or len(trace) != 6:
        _fail("process state sequence length drift")
    for index, row in enumerate(trace):
        row = _exact(row, {"sequence", "state", "predecessor_sha256"}, f"result.state_sequence[{index}]")
        if row["sequence"] != index or (index == 0 and row["predecessor_sha256"] is not None) or (index > 0 and row["predecessor_sha256"] != _digest(trace[index - 1])):
            _fail("process state predecessor drift")
    if result["state_transition_sha256"] != _digest(trace) or result["child_invocation_count"] != 1 or result["shell_used"] is not False:
        _fail("process invocation semantics drift")
    body = _copy(result)
    del body["aggregate_result_sha256"]
    if _digest(body) != result["aggregate_result_sha256"]:
        _fail("process result aggregate drift")
    if expected_descriptor is not None:
        descriptor = validate_process_descriptor(expected_descriptor)
        if result["descriptor_sha256"] != descriptor["aggregate_sha256"] or result["entry_descriptor_sha256"] != descriptor["entry_descriptor_sha256"]:
            _fail("process result descriptor binding drift")
        if result["entry_evidence_root"] != descriptor["entry_evidence_root"]:
            _fail("process result entry evidence binding drift")
        if result["authorization_binding_sha256"] != descriptor["entry_descriptor"]["authorization_binding_sha256"] or result["argv"] != descriptor["argv"] or result["cwd"] != descriptor["cwd"] or result["environment"] != descriptor["environment"] or result["process_evidence_root"] != descriptor["process_evidence_root"] or result["process_receipt_path"] != descriptor["process_receipt_path"]:
            _fail("process result invocation binding drift")
        if result["training_run_id"] != descriptor["training_run_id"] or result["nonce"] != descriptor["nonce"]:
            _fail("process result run identity drift")
    return _copy(result)


def _inventory(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    allowed = set(JSON_FILE_NAMES + BINARY_FILE_NAMES) - {"artifact_inventory.json", "completion.json"}
    present = {path.name for path in root.iterdir()}
    unexpected = present - set(PROCESS_EVIDENCE_FILE_NAMES) - {"artifact_inventory.json", "completion.json"}
    if unexpected:
        _fail(f"process evidence has unexpected files: {sorted(unexpected)}")
    for name in sorted((present & allowed)):
        path = root / name
        observed = _entry._regular_file(path, name, expected_mode=0o600, expected_nlink=1)
        raw = _entry._read_stable_file(path, name, mode=0o600)
        rows.append({"relative_path": name, "size_bytes": len(raw), "sha256": _sha(raw), "mode": stat.S_IMODE(observed.st_mode), "nlink": observed.st_nlink})
    return rows


def _publish_terminal(
    root: Path,
    descriptor: dict[str, Any],
    receipt: dict[str, Any],
    result: dict[str, Any],
    stdout: bytes,
    stderr: bytes,
    stream_receipt_sha256: str,
) -> None:
    _entry._write_new(root / "stdout.bin", stdout, "stdout.bin")
    _entry._write_new(root / "stderr.bin", stderr, "stderr.bin")
    _publish(root, "result.json", result)
    rows = _inventory(root)
    inventory = {"schema_version": PROCESS_SCHEMA_VERSION, "excluded": ["artifact_inventory.json", "completion.json"], "files": rows, "canonical_inventory_sha256": _digest(rows)}
    _publish(root, "artifact_inventory.json", inventory)
    completion = {
        "schema_version": PROCESS_SCHEMA_VERSION,
        "status": result["status"],
        "classification": result["classification"],
        "exit_code": result["return_code"],
        "descriptor_sha256": descriptor["aggregate_sha256"],
        "result_sha256": _sha(_canonical(result)),
        "inventory_sha256": _sha(_canonical(inventory)),
        "stdout_size_bytes": result["stdout_size_bytes"],
        "stdout_sha256": result["stdout_sha256"],
        "stderr_size_bytes": result["stderr_size_bytes"],
        "stderr_sha256": result["stderr_sha256"],
        "entry_result_sha256": result["entry_result_sha256"],
        "state_transition_sha256": result["state_transition_sha256"],
        "receipt_sha256": receipt["receipt_sha256"],
        "stream_receipt_sha256": stream_receipt_sha256,
    }
    _publish(root, "completion.json", completion)
    _entry._fsync_directory(root, "process evidence root")


def run_process_once(
    descriptor: Mapping[str, Any],
    runner: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Invoke exactly one child and persist one immutable terminal record."""

    checked = validate_process_descriptor(dict(descriptor))
    try:
        _entry.validate_consumed_authorization_receipt(
            checked["entry_descriptor"]["authorization_receipt_path"],
            authorization_binding=checked["entry_descriptor"]["authorization_binding"],
            authorization_path=checked["entry_descriptor"]["authorization_path"],
        )
    except _entry.TrainingEntryError as exc:
        raise TrainingProcessError(str(exc)) from exc
    root, receipt = _claim_process_root(checked)
    try:
        _publish(
            root,
            "invocation.json",
            {
                "schema_version": PROCESS_SCHEMA_VERSION,
                "status": "RUNNING",
                "descriptor_sha256": checked["aggregate_sha256"],
                "entry_descriptor_sha256": checked["entry_descriptor_sha256"],
                "child_invocation_count": 0,
            },
        )
    except Exception as exc:
        _record_persistence_failure(root, checked, "process_invocation", exc)
        if isinstance(exc, TrainingProcessError):
            raise
        raise TrainingProcessError("process invocation evidence could not be published") from exc
    argv = list(checked["argv"])
    cwd = checked["cwd"]
    environment = _copy(checked["environment"])
    entry_descriptor = _copy(checked["entry_descriptor"])
    payload: dict[str, Any] | None = None
    failure_class = "NONE"
    try:
        if runner is not None:
            try:
                inspect.signature(runner).bind(argv, cwd, environment, entry_descriptor)
            except (TypeError, ValueError) as exc:
                failure_class = "RUNNER_SCHEMA_FAILURE"
                raise TrainingProcessError("runner signature is not the closed process ABI") from exc
            original = (list(argv), _copy(cwd), _copy(environment), _copy(entry_descriptor))
            returned = runner(argv, cwd, environment, entry_descriptor)
            if argv != original[0] or cwd != original[1] or environment != original[2] or entry_descriptor != original[3]:
                failure_class = "RUNNER_MUTATED_INVOCATION"
                raise TrainingProcessError("runner mutated its invocation inputs")
            payload = _validate_runner_payload(returned)
        else:
            payload = _validate_runner_payload(_invoke_real(checked))
        if payload["return_code"] != 0:
            failure_class = "RUNNER_NONZERO_EXIT"
        if payload["entry_result"] is not None and payload["return_code"] == 0 and payload["entry_result"].get("status") == "PERMANENT_FAIL":
            failure_class = "ENTRY_TERMINAL_FAILURE"
    except TimeoutError:
        failure_class = "RUNNER_TIMEOUT"
    except KeyboardInterrupt:
        failure_class = "RUNNER_SIGNAL"
    except TrainingProcessError:
        if failure_class == "NONE":
            failure_class = "RUNNER_SCHEMA_FAILURE"
        if payload is None:
            payload = {"return_code": FAILURE_RETURN_CODES.get(failure_class, 5), "stdout": b"", "stderr": b"", "pid": 0, "entry_result": None}
    except Exception:
        failure_class = "RUNNER_EXCEPTION"
        payload = {"return_code": FAILURE_RETURN_CODES[failure_class], "stdout": b"", "stderr": b"", "pid": 0, "entry_result": None}
    if payload is None:
        payload = {"return_code": FAILURE_RETURN_CODES.get(failure_class, 5), "stdout": b"", "stderr": b"", "pid": 0, "entry_result": None}
    if failure_class == "NONE" and payload["return_code"] != 0:
        failure_class = "RUNNER_NONZERO_EXIT"
    stream_receipt = {
        "schema_version": PROCESS_SCHEMA_VERSION,
        "kind": "T6B_PROCESS_STREAMS",
        "descriptor_sha256": checked["aggregate_sha256"],
        "pid": payload["pid"],
        "return_code": payload["return_code"],
        "stdout_size_bytes": len(payload["stdout"]),
        "stdout_sha256": _sha(payload["stdout"]),
        "stderr_size_bytes": len(payload["stderr"]),
        "stderr_sha256": _sha(payload["stderr"]),
    }
    try:
        stream_receipt_raw = _canonical(stream_receipt)
        stream_receipt_path = Path(checked["process_receipt_path"]).parent / f".{Path(checked['process_evidence_root']).name}{PROCESS_STREAM_RECEIPT_SUFFIX}"
        _entry._write_new(stream_receipt_path, stream_receipt_raw, "process stream receipt")
        _entry._fsync_directory(stream_receipt_path.parent, "process stream receipt parent")
        result = _make_result(checked, payload, receipt, failure_class)
        validated = validate_process_result(result, checked)
        _publish_terminal(root, checked, receipt, validated, payload["stdout"], payload["stderr"], _sha(stream_receipt_raw))
        return validated
    except Exception as exc:
        _record_persistence_failure(root, checked, "process_terminal", exc)
        if isinstance(exc, TrainingProcessError):
            raise
        raise TrainingProcessError("process terminal evidence could not be published") from exc


def _read_bytes(root: Path, name: str) -> bytes:
    return _entry._read_stable_file(root / name, name)


def classify_process(process_evidence_root: str | os.PathLike[str]) -> str:
    try:
        root = Path(_absolute(os.fspath(process_evidence_root), "process_evidence_root"))
        if _has_valid_persistence_failure_marker(root):
            return "PERMANENT_FAIL"
        if not root.exists():
            return "ABSENT"
        if root.is_symlink() or not root.is_dir():
            return "UNKNOWN"
        names = {path.name for path in root.iterdir()}
        if names == {"invocation.json"}:
            return "RUNNING"
        mandatory = {"invocation.json", "stdout.bin", "stderr.bin", "result.json", "artifact_inventory.json", "completion.json"}
        if names != mandatory:
            return "UNKNOWN"
        result_raw = _read_bytes(root, "result.json")
        result, _ = _entry._read_json(root / "result.json", "result.json")
        if result["process_evidence_root"] != str(root) or result["process_receipt_path"] != str(root.parent / f".{root.name}{PROCESS_RECEIPT_SUFFIX}"):
            return "UNKNOWN"
        claim, claim_raw = _entry._read_json(Path(result["process_receipt_path"]), "process receipt")
        if set(claim) != {"schema_version", "kind", "descriptor_sha256", "descriptor", "training_run_id", "nonce", "status"} or claim_raw != _canonical(claim):
            return "UNKNOWN"
        claim_descriptor = validate_process_descriptor(claim["descriptor"])
        expected_claim = {
            "schema_version": PROCESS_SCHEMA_VERSION,
            "kind": "T6B_PROCESS_CLAIM",
            "descriptor_sha256": result["descriptor_sha256"],
            "descriptor": claim_descriptor,
            "training_run_id": result["training_run_id"],
            "nonce": result["nonce"],
            "status": "CLAIMED",
        }
        if claim != expected_claim or claim_descriptor["aggregate_sha256"] != result["descriptor_sha256"]:
            return "UNKNOWN"
        validate_process_result(result, claim_descriptor)
        stream_receipt_path = root.parent / f".{root.name}{PROCESS_STREAM_RECEIPT_SUFFIX}"
        stream_receipt, stream_receipt_raw = _entry._read_json(stream_receipt_path, "process stream receipt")
        if _entry._regular_file(stream_receipt_path, "process stream receipt", expected_mode=0o600, expected_nlink=1) is None:
            return "UNKNOWN"
        expected_stream_receipt = {
            "schema_version": PROCESS_SCHEMA_VERSION,
            "kind": "T6B_PROCESS_STREAMS",
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
        stdout = _read_bytes(root, "stdout.bin")
        stderr = _read_bytes(root, "stderr.bin")
        if len(stdout) != result["stdout_size_bytes"] or _sha(stdout) != result["stdout_sha256"] or len(stderr) != result["stderr_size_bytes"] or _sha(stderr) != result["stderr_sha256"]:
            return "UNKNOWN"
        invocation, invocation_raw = _entry._read_json(root / "invocation.json", "invocation.json")
        if invocation_raw != _canonical(invocation) or invocation != {"schema_version": PROCESS_SCHEMA_VERSION, "status": "RUNNING", "descriptor_sha256": result["descriptor_sha256"], "entry_descriptor_sha256": result["entry_descriptor_sha256"], "child_invocation_count": 0}:
            return "UNKNOWN"
        completion, _ = _entry._read_json(root / "completion.json", "completion.json")
        inventory, _ = _entry._read_json(root / "artifact_inventory.json", "artifact_inventory.json")
        if type(inventory) is not dict or set(inventory) != {"schema_version", "excluded", "files", "canonical_inventory_sha256"} or inventory["canonical_inventory_sha256"] != _digest(inventory["files"]):
            return "UNKNOWN"
        if inventory["files"] != _inventory(root):
            return "UNKNOWN"
        if result["entry_result"] is not None:
            entry_result, entry_result_raw = _entry._read_json(Path(result["entry_evidence_root"]) / "entry_result.json", "entry result")
            if entry_result_raw != _entry._canonical(entry_result) or entry_result != result["entry_result"]:
                return "UNKNOWN"
            if _entry.classify_entry(result["entry_evidence_root"]) != result["entry_result"]["status"]:
                return "UNKNOWN"
        expected_completion = {
            "schema_version": PROCESS_SCHEMA_VERSION,
            "status": result["status"],
            "classification": result["classification"],
            "exit_code": result["return_code"],
            "descriptor_sha256": result["descriptor_sha256"],
            "result_sha256": _sha(result_raw),
            "inventory_sha256": _sha(_canonical(inventory)),
            "stdout_size_bytes": result["stdout_size_bytes"],
            "stdout_sha256": result["stdout_sha256"],
            "stderr_size_bytes": result["stderr_size_bytes"],
            "stderr_sha256": result["stderr_sha256"],
            "entry_result_sha256": result["entry_result_sha256"],
            "state_transition_sha256": result["state_transition_sha256"],
            "receipt_sha256": None,
            "stream_receipt_sha256": _sha(stream_receipt_raw),
        }
        receipt_raw = _entry._read_stable_file(Path(result["process_receipt_path"]), "process receipt", mode=0o600)
        expected_completion["receipt_sha256"] = _sha(receipt_raw)
        if completion != expected_completion:
            return "UNKNOWN"
        return result["status"]
    except Exception:
        return "UNKNOWN"


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2 or args[0] != "--descriptor":
        return 2
    descriptor_path = Path(_absolute(args[1], "process descriptor path"))
    descriptor, _ = _entry._read_json(descriptor_path, "process descriptor")
    result = run_process_once(descriptor)
    sys.stdout.buffer.write(_canonical(result) + b"\n")
    sys.stdout.buffer.flush()
    return result["return_code"]


if __name__ == "__main__":
    raise SystemExit(main())
