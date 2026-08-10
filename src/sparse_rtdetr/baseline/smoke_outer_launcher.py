"""Durable, single-call outer launcher for Baseline Smoke V2.

The outer launcher owns only outer and pane evidence. It does not import
torch, construct a dataset, or infer scientific completion from tmux status.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import signal
import stat
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .smoke import (
    SMOKE_V2_CONFIG_RELATIVE,
    SMOKE_V2_ID,
    SMOKE_V2_TMUX_SESSION,
    SMOKE_V2_TMUX_TIMEOUT_SECONDS,
    load_smoke_config,
)
from .smoke_evidence import (
    SmokeEvidenceError,
    _read_object,
    atomic_write,
    argv_sha256,
    canonical_json_bytes,
    file_ref,
    inventory,
    sha256_bytes,
    sha256_file,
    validate_entry_output,
)


class OuterLaunchError(Exception):
    """Raised when the outer evidence contract cannot be established."""


OUTER_COMPLETION = "outer_completion.json"
OUTER_INVENTORY = "outer_inventory.json"
OUTER_PARTIAL_INVENTORY = "outer_partial_inventory.json"
OUTER_SECONDARY_FAILURE = "outer_secondary_finalization_failure.json"
OUTER_EXCLUDED = frozenset({OUTER_COMPLETION, OUTER_INVENTORY, OUTER_PARTIAL_INVENTORY, OUTER_SECONDARY_FAILURE})
PANE_COMPLETION = "pane_completion.json"
PANE_INVENTORY = "pane_inventory.json"
PANE_PARTIAL_INVENTORY = "pane_partial_inventory.json"
PANE_SECONDARY_FAILURE = "pane_secondary_finalization_failure.json"
PANE_LOCK = "pane_consume.lock"
PANE_RECEIPT = "pane_receipt.json"
PANE_START = "pane_start.json"
PANE_EXIT = "pane_exit_code.txt"
PANE_TIMING = "pane_timing.json"
PANE_ERROR = "pane_error.json"
PANE_CONSOLE = "pane_console.log"
PANE_WRAPPER = "pane_wrapper.sh"
PANE_FINALIZER_EXIT = "pane_finalizer_exit_code.txt"
PANE_RUNTIME_EVIDENCE = (
    PANE_LOCK,
    PANE_CONSOLE,
    PANE_START,
    PANE_EXIT,
    PANE_RECEIPT,
    PANE_TIMING,
    PANE_ERROR,
    PANE_INVENTORY,
    PANE_PARTIAL_INVENTORY,
    PANE_COMPLETION,
    PANE_SECONDARY_FAILURE,
    PANE_FINALIZER_EXIT,
)
PANE_EXCLUDED = frozenset({PANE_COMPLETION, PANE_INVENTORY, PANE_PARTIAL_INVENTORY, PANE_SECONDARY_FAILURE, PANE_FINALIZER_EXIT})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_NONCE_RE = re.compile(r"^[0-9a-f]{32}$")
PANE_TIMING_TOLERANCE_SECONDS = 1e-6
PANE_FAILURE_MESSAGES = {
    "PREINNER_BINDING_FAILURE": "pane binding failed before the inner command.",
    "INNER_NONZERO_EXIT": "inner command exited with a nonzero return code.",
    "PANE_FINALIZATION_FAILURE": "pane finalization failed after inner evidence was persisted.",
}
_EXECUTABLE_IDENTITY_KEYS = {"canonical_path", "size_bytes", "sha256", "mode", "regular_file", "executable"}
_PANE_ERROR_KEYS = {"schema_version", "status", "failure_class", "message", "inner_started", "inner_returncode", "original_error_preserved", "console", "nonce", "pane_pid", "plan_sha256", "wrapper_sha256", "created_at_utc"}
_PANE_SECONDARY_KEYS = {"schema_version", "status", "failure_class", "pane_pid", "nonce", "plan_sha256", "wrapper_sha256", "inner_started", "original_inner_returncode", "finalizer_returncode", "finalizer_executable_identity", "original_error_preserved", "console", "created_at_utc"}
_MONITORED_SIGNALS = tuple(value for value in (signal.SIGHUP, signal.SIGTERM, signal.SIGINT, signal.SIGQUIT) if value is not None)
_MONITORED_SIGNAL_NAMES = ["SIGHUP", "SIGTERM", "SIGINT", "SIGQUIT"]
_PLAN_KEYS = {"schema_version", "state", "single_use", "nonce", "smoke_id", "repo_root", "data_root", "output_dir", "process_evidence_dir", "outer_evidence_dir", "config_binding", "child_python", "child_python_identity", "pane_evidence_python", "pane_evidence_python_identity", "inner_argv", "inner_argv_sha256", "tmux_session", "created_at_utc", "nonportable"}
_INVOCATION_KEYS = {"schema_version", "mode", "status", "nonportable", "created_at_utc", "nonce", "smoke_id", "repo_root", "data_root", "config_path", "output_dir", "process_evidence_dir", "outer_evidence_dir", "child_python", "child_python_identity", "pane_evidence_python", "pane_evidence_python_identity", "tmux_executable", "tmux_session", "environment"}
_SIGNAL_EVENT_KEYS = {"signal_number", "signal_name", "received_at_utc", "forwarded_to_tmux_client", "forwarding_result"}
_FORWARD_FAILURE_RE = re.compile(r"^failed:[A-Za-z_][A-Za-z0-9_]*$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _strict_sha(value: Any, field: str) -> None:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise OuterLaunchError(f"{field} must be a lowercase SHA-256 string")


def _strict_nonce(value: Any, field: str = "nonce") -> None:
    if type(value) is not str or _NONCE_RE.fullmatch(value) is None:
        raise OuterLaunchError(f"{field} must be 32 lowercase hexadecimal characters")


def _strict_int(value: Any, field: str, minimum: int | None = None) -> None:
    if type(value) is not int or (minimum is not None and value < minimum):
        raise OuterLaunchError(f"{field} must be an integer")


def _strict_bool(value: Any, field: str) -> None:
    if type(value) is not bool:
        raise OuterLaunchError(f"{field} must be a bool")


def _strict_finite_number(value: Any, field: str, minimum: float | None = None) -> None:
    if type(value) not in (int, float) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise OuterLaunchError(f"{field} must be a finite number")
    if minimum is not None and float(value) < minimum:
        raise OuterLaunchError(f"{field} must be >= {minimum}")


def _parse_utc(value: Any, field: str) -> datetime:
    if type(value) is not str or not value:
        raise OuterLaunchError(f"{field} must be a non-empty UTC ISO-8601 string")
    if not (value.endswith("Z") or value.endswith("+00:00")):
        raise OuterLaunchError(f"{field} must contain an explicit UTC timezone")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    except ValueError as exc:
        raise OuterLaunchError(f"{field} is not valid ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise OuterLaunchError(f"{field} must contain a UTC timezone")
    return parsed


def _validate_executable_identity(value: Any, field: str = "executable identity") -> None:
    if not isinstance(value, dict) or set(value) != _EXECUTABLE_IDENTITY_KEYS:
        raise OuterLaunchError(f"{field} schema is invalid")
    if type(value.get("canonical_path")) is not str or not value["canonical_path"] or not Path(value["canonical_path"]).is_absolute():
        raise OuterLaunchError(f"{field} canonical path is invalid")
    _strict_int(value.get("size_bytes"), f"{field} size_bytes", 1)
    _strict_sha(value.get("sha256"), f"{field} sha256")
    _strict_int(value.get("mode"), f"{field} mode", 0)
    _strict_bool(value.get("regular_file"), f"{field} regular_file")
    _strict_bool(value.get("executable"), f"{field} executable")
    if value["regular_file"] is not True or value["executable"] is not True:
        raise OuterLaunchError(f"{field} does not describe an executable regular file")


def _pane_error_payload(*, failure_class: str, message: str, inner_started: bool, inner_returncode: int, console: dict[str, Any], nonce: str, pane_pid: int, plan_sha256: str, wrapper_sha256: str, created_at_utc: str) -> dict[str, Any]:
    """Build the exact pane failure schema used by the Python finalizer."""
    if failure_class not in PANE_FAILURE_MESSAGES or message != PANE_FAILURE_MESSAGES[failure_class]:
        raise OuterLaunchError("pane error payload classification is invalid")
    _strict_bool(inner_started, "pane error inner_started")
    _strict_int(inner_returncode, "pane error inner_returncode")
    _strict_nonce(nonce, "pane error nonce")
    _strict_int(pane_pid, "pane error pane_pid", 1)
    _strict_sha(plan_sha256, "pane error plan_sha256")
    _strict_sha(wrapper_sha256, "pane error wrapper_sha256")
    _parse_utc(created_at_utc, "pane error created_at_utc")
    return {
        "schema_version": 1,
        "status": "PANE_FAILED",
        "failure_class": failure_class,
        "message": message,
        "inner_started": inner_started,
        "inner_returncode": inner_returncode,
        "original_error_preserved": True,
        "console": console,
        "nonce": nonce,
        "pane_pid": pane_pid,
        "plan_sha256": plan_sha256,
        "wrapper_sha256": wrapper_sha256,
        "created_at_utc": created_at_utc,
    }


def _pane_timing_payload(*, started_at_utc: str, finished_at_utc: str, inner_started: bool, inner_returncode: int, signal_name: str | None) -> dict[str, Any]:
    """Build pane timing with a real, non-negative, recomputable duration."""
    started = _parse_utc(started_at_utc, "pane timing started_at_utc")
    finished = _parse_utc(finished_at_utc, "pane timing finished_at_utc")
    elapsed = (finished - started).total_seconds()
    if elapsed < 0 or not math.isfinite(elapsed):
        raise OuterLaunchError("pane timing chronology is invalid")
    _strict_bool(inner_started, "pane timing inner_started")
    _strict_int(inner_returncode, "pane timing inner_returncode")
    if signal_name is not None and signal_name not in _MONITORED_SIGNAL_NAMES:
        raise OuterLaunchError("pane timing signal_name is invalid")
    return {
        "schema_version": 1,
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "elapsed_seconds": elapsed,
        "inner_started": inner_started,
        "inner_returncode": inner_returncode,
        "signal_name": signal_name,
    }


def _pane_secondary_payload(*, pane_pid: int, nonce: str, plan_sha256: str, wrapper_sha256: str, inner_started: bool, original_inner_returncode: int, finalizer_returncode: int, finalizer_executable_identity: dict[str, Any], console: dict[str, Any], created_at_utc: str) -> dict[str, Any]:
    """Build durable secondary evidence without exposing arbitrary exception text."""
    _strict_int(pane_pid, "pane secondary pane_pid", 1)
    _strict_nonce(nonce, "pane secondary nonce")
    _strict_sha(plan_sha256, "pane secondary plan_sha256")
    _strict_sha(wrapper_sha256, "pane secondary wrapper_sha256")
    _strict_bool(inner_started, "pane secondary inner_started")
    _strict_int(original_inner_returncode, "pane secondary original_inner_returncode")
    _strict_int(finalizer_returncode, "pane secondary finalizer_returncode")
    _validate_executable_identity(finalizer_executable_identity, "pane secondary finalizer identity")
    _parse_utc(created_at_utc, "pane secondary created_at_utc")
    return {
        "schema_version": 1,
        "status": "FAILED_FINALIZATION",
        "failure_class": "PANE_FINALIZATION_FAILURE",
        "pane_pid": pane_pid,
        "nonce": nonce,
        "plan_sha256": plan_sha256,
        "wrapper_sha256": wrapper_sha256,
        "inner_started": inner_started,
        "original_inner_returncode": original_inner_returncode,
        "finalizer_returncode": finalizer_returncode,
        "finalizer_executable_identity": finalizer_executable_identity,
        "original_error_preserved": True,
        "console": console,
        "created_at_utc": created_at_utc,
    }


def _pane_preexisting_evidence_gate(pane: Path) -> None:
    """Fail closed if any runtime evidence belongs to an earlier consumer."""
    for name in PANE_RUNTIME_EVIDENCE:
        path = pane / name
        if path.exists() or path.is_symlink():
            raise OuterLaunchError("pane runtime evidence already exists")


def executable_identity(path: Path) -> dict[str, Any]:
    """Return the byte identity of a canonical executable target.

    Conda commonly exposes Python through a final symlink. Parent symlinks and
    canonical escapes remain forbidden, while that final symlink is resolved to
    the ordinary executable whose bytes are actually invoked.
    """
    requested = Path(path)
    if not requested.is_absolute() or any(part == ".." for part in requested.parts):
        raise OuterLaunchError(f"executable path is not a trusted absolute path: {requested}")
    current = Path(os.sep)
    parts = requested.parts[1:]
    for index, component in enumerate(parts):
        current /= component
        try:
            info = current.lstat()
        except OSError as exc:
            raise OuterLaunchError(f"cannot inspect executable path component: {current}") from exc
        if stat.S_ISLNK(info.st_mode) and index != len(parts) - 1:
            raise OuterLaunchError(f"executable parent symlink is forbidden: {current}")
        if index != len(parts) - 1 and not stat.S_ISDIR(info.st_mode):
            raise OuterLaunchError(f"executable path component is not a directory: {current}")
    try:
        canonical = Path(os.path.realpath(requested))
    except OSError as exc:
        raise OuterLaunchError(f"cannot canonicalize executable path: {requested}") from exc
    _validate_path_components(canonical, final_kind="file")
    info = canonical.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or not os.access(canonical, os.X_OK):
        raise OuterLaunchError(f"executable target is not a regular executable: {canonical}")
    identity = {
        "canonical_path": str(canonical),
        "size_bytes": info.st_size,
        "sha256": sha256_file(canonical),
        "mode": stat.S_IMODE(info.st_mode),
        "regular_file": True,
        "executable": True,
    }
    _validate_executable_identity(identity)
    return identity


def _validate_signal_events(runner: dict[str, Any], status: str) -> None:
    handlers = runner.get("signal_handlers_installed")
    if type(handlers) is not bool:
        raise OuterLaunchError("signal_handlers_installed must be a bool")
    events = runner.get("signal_events")
    if not isinstance(events, list):
        raise OuterLaunchError("signal_events must be a list")
    count = runner.get("observed_signal_count")
    if type(count) is not int or count < 0 or count != len(events):
        raise OuterLaunchError("observed_signal_count does not bind signal_events")
    if runner.get("monitored_signals") != _MONITORED_SIGNAL_NAMES:
        raise OuterLaunchError("monitored_signals drift")
    if status == "PREFLIGHT_FAILED" and not handlers:
        if events or count != 0:
            raise OuterLaunchError("uninstalled preflight runner contains signal events")
    elif not handlers:
        raise OuterLaunchError("tmux runner signal handlers were not installed")
    previous_time: datetime | None = None
    for event in events:
        if not isinstance(event, dict) or set(event) != _SIGNAL_EVENT_KEYS:
            raise OuterLaunchError("signal event schema is invalid")
        number = event.get("signal_number")
        if type(number) is not int or number not in _MONITORED_SIGNALS:
            raise OuterLaunchError("signal event number is invalid")
        if event.get("signal_name") != signal.Signals(number).name or type(event.get("signal_name")) is not str:
            raise OuterLaunchError("signal event name is invalid")
        timestamp = _parse_utc(event.get("received_at_utc"), "signal event received_at_utc")
        if previous_time is not None and timestamp < previous_time:
            raise OuterLaunchError("signal event timestamps are not non-decreasing")
        previous_time = timestamp
        forwarded = event.get("forwarded_to_tmux_client")
        if type(forwarded) is not bool:
            raise OuterLaunchError("signal event forwarded flag is invalid")
        forwarding_result = event.get("forwarding_result")
        if type(forwarding_result) is not str or not forwarding_result:
            raise OuterLaunchError("signal event forwarding result is invalid")
        if forwarded and forwarding_result != "forwarded":
            raise OuterLaunchError("forwarded signal event result is invalid")
        if not forwarded and forwarding_result != "not_running" and _FORWARD_FAILURE_RE.fullmatch(forwarding_result) is None:
            raise OuterLaunchError("non-forwarded signal event result is invalid")
    for field in ("timeout_triggered", "term_sent", "kill_sent"):
        if type(runner.get(field)) is not bool:
            raise OuterLaunchError(f"{field} must be a bool")
    original = runner.get("original_exception")
    if original is not None:
        if not isinstance(original, dict) or set(original) != {"type", "message"} or type(original.get("type")) is not str or not original["type"] or type(original.get("message")) is not str:
            raise OuterLaunchError("original_exception schema is invalid")


def _validate_outer_invocation(invocation: dict[str, Any], root: Path) -> None:
    if set(invocation) != _INVOCATION_KEYS:
        raise OuterLaunchError("outer invocation schema fields are invalid")
    if invocation.get("schema_version") != 1 or invocation.get("mode") != "outer_launch" or invocation.get("status") != "OUTER_PREPARED" or invocation.get("nonportable") is not True or invocation.get("smoke_id") != SMOKE_V2_ID:
        raise OuterLaunchError("outer invocation identity is invalid")
    if invocation.get("outer_evidence_dir") != str(root):
        raise OuterLaunchError("outer invocation root binding is invalid")
    for field in ("created_at_utc", "repo_root", "data_root", "config_path", "output_dir", "process_evidence_dir", "outer_evidence_dir", "child_python", "pane_evidence_python", "tmux_executable"):
        if type(invocation.get(field)) is not str or not invocation[field]:
            raise OuterLaunchError(f"outer invocation string field is invalid: {field}")
    _parse_utc(invocation["created_at_utc"], "outer invocation created_at_utc")
    _strict_nonce(invocation.get("nonce"), "outer invocation nonce")
    if invocation.get("child_python_identity") is not None:
        _validate_executable_identity(invocation.get("child_python_identity"), "outer child Python identity")
    _validate_executable_identity(invocation.get("pane_evidence_python_identity"), "outer pane evidence Python identity")
    if invocation.get("child_python_identity") is not None and invocation["child_python"] != invocation["child_python_identity"]["canonical_path"]:
        raise OuterLaunchError("outer child executable path does not bind its identity")
    if invocation["pane_evidence_python"] != invocation["pane_evidence_python_identity"]["canonical_path"]:
        raise OuterLaunchError("outer executable paths do not bind their identities")
    environment = invocation.get("environment")
    if not isinstance(environment, dict) or set(environment) != {"CUDA_VISIBLE_DEVICES", "PYTHONDONTWRITEBYTECODE", "PYTHONNOUSERSITE", "P3_RTDETR_BASELINE_SMOKE_AUTHORIZED", "PYTHONPATH", "P3_SMOKE_TMUX_SESSION", "PANE_EVIDENCE_PYTHON"}:
        raise OuterLaunchError("outer invocation environment schema is invalid")
    expected_environment = {
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "P3_RTDETR_BASELINE_SMOKE_AUTHORIZED": "1",
        "PYTHONPATH": str(Path(invocation["repo_root"]) / "src"),
        "P3_SMOKE_TMUX_SESSION": SMOKE_V2_TMUX_SESSION,
    }
    for field, expected in expected_environment.items():
        if type(environment.get(field)) is not str or environment[field] != expected:
            raise OuterLaunchError(f"outer invocation environment drift: {field}")
    if type(environment.get("PANE_EVIDENCE_PYTHON")) is not str or environment["PANE_EVIDENCE_PYTHON"] != invocation["pane_evidence_python_identity"]["canonical_path"]:
        raise OuterLaunchError("outer invocation evidence Python is invalid")


def _validate_pane_plan_identity(
    pane: Path,
    expected_plan: dict[str, Any] | None = None,
    expected_wrapper: bytes | None = None,
) -> tuple[dict[str, Any], str, str]:
    plan_path = pane / "pane_plan.json"
    wrapper_path = pane / PANE_WRAPPER
    plan = _read_json_object(plan_path)
    if plan is None or set(plan) != _PLAN_KEYS or plan.get("schema_version") != 1 or plan.get("state") != "PREPARED" or plan.get("single_use") is not True:
        raise OuterLaunchError("pane plan schema is invalid")
    _strict_nonce(plan.get("nonce"))
    _validate_exact_session(plan.get("tmux_session"))
    if plan.get("smoke_id") != SMOKE_V2_ID or plan.get("nonportable") is not True or plan.get("inner_argv_sha256") != argv_sha256(plan.get("inner_argv")):
        raise OuterLaunchError("pane plan identity or argv binding is invalid")
    _validate_executable_identity(plan.get("child_python_identity"), "pane child Python identity")
    _validate_executable_identity(plan.get("pane_evidence_python_identity"), "pane evidence Python identity")
    if plan.get("child_python") != plan["child_python_identity"]["canonical_path"] or plan.get("pane_evidence_python") != plan["pane_evidence_python_identity"]["canonical_path"]:
        raise OuterLaunchError("pane plan executable path binding is invalid")
    _parse_utc(plan.get("created_at_utc"), "pane plan created_at_utc")
    config_binding = plan.get("config_binding")
    if not isinstance(config_binding, dict) or set(config_binding) != {"smoke_id", "config_relative_path", "config_path", "config_size_bytes", "config_file_sha256", "config_canonical_sha256"} or config_binding.get("config_relative_path") != SMOKE_V2_CONFIG_RELATIVE:
        raise OuterLaunchError("pane plan config binding is invalid")
    _strict_sha(config_binding.get("config_file_sha256"), "pane plan config_file_sha256")
    _strict_sha(config_binding.get("config_canonical_sha256"), "pane plan config_canonical_sha256")
    _strict_int(config_binding.get("config_size_bytes"), "pane plan config_size_bytes", 1)
    _strict_file(wrapper_path, executable=True)
    plan_sha = sha256_file(plan_path)
    wrapper_sha = sha256_file(wrapper_path)
    if expected_plan is not None:
        expected_plan_bytes = canonical_json_bytes(expected_plan)
        if plan != expected_plan or plan_sha != sha256_bytes(expected_plan_bytes) or plan_path.read_bytes() != expected_plan_bytes:
            raise OuterLaunchError("pane plan is not the deterministic expected plan")
    if expected_wrapper is not None:
        if wrapper_path.read_bytes() != expected_wrapper or wrapper_sha != sha256_bytes(expected_wrapper) or stat.S_IMODE(wrapper_path.stat().st_mode) != 0o755:
            raise OuterLaunchError("pane wrapper is not the deterministic expected wrapper")
    return plan, plan_sha, wrapper_sha


def _validate_path_components(path: Path, *, final_kind: str, allow_missing_final: bool = False) -> None:
    """Inspect every POSIX component with lstat before accepting a path."""

    path = Path(path)
    if not path.is_absolute() or any(part == ".." for part in path.parts):
        raise OuterLaunchError(f"path is not a trusted absolute path: {path}")
    if not path.parts or path.parts[0] != os.sep:
        raise OuterLaunchError(f"path is not POSIX absolute: {path}")
    current = Path(os.sep)
    components = path.parts[1:]
    for index, component in enumerate(components):
        current /= component
        final = index == len(components) - 1
        try:
            info = current.lstat()
        except FileNotFoundError:
            if final and allow_missing_final:
                return
            raise OuterLaunchError(f"missing path component: {current}")
        except OSError as exc:
            raise OuterLaunchError(f"cannot inspect path component: {current}") from exc
        if stat.S_ISLNK(info.st_mode):
            raise OuterLaunchError(f"symlink path component is forbidden: {current}")
        if not final and not stat.S_ISDIR(info.st_mode):
            raise OuterLaunchError(f"non-directory path component: {current}")
        if final and final_kind == "dir" and not stat.S_ISDIR(info.st_mode):
            raise OuterLaunchError(f"path is not a directory: {current}")
        if final and final_kind == "file" and not stat.S_ISREG(info.st_mode):
            raise OuterLaunchError(f"path is not a regular file: {current}")
    try:
        if Path(os.path.realpath(path)) != path:
            raise OuterLaunchError(f"path canonical escape: {path}")
    except OSError as exc:
        raise OuterLaunchError(f"cannot canonicalize path: {path}") from exc


def _strict_file(path: Path, *, executable: bool = False) -> None:
    _validate_path_components(path, final_kind="file")
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise OuterLaunchError(f"not a regular non-hardlinked file: {path}")
    if executable and not os.access(path, os.X_OK):
        raise OuterLaunchError(f"file is not executable: {path}")


def _validate_outer_parent(outer_root: Path) -> None:
    parent = Path(outer_root).parent
    _validate_path_components(parent, final_kind="dir")
    info = parent.lstat()
    if stat.S_IMODE(info.st_mode) != 0o775 or info.st_uid != os.geteuid() or info.st_gid != os.getegid():
        raise OuterLaunchError("outer evidence parent owner/group/mode mismatch")
    _validate_path_components(Path(outer_root), final_kind="dir", allow_missing_final=True)
    if outer_root.exists() or outer_root.is_symlink():
        raise OuterLaunchError("outer evidence root already exists")


def _validate_exact_session(value: Any) -> None:
    if type(value) is not str or value != SMOKE_V2_TMUX_SESSION:
        raise OuterLaunchError("tmux session identity is not the frozen R2 value")


def _validate_timeout(value: Any) -> None:
    if type(value) is not int or value != SMOKE_V2_TMUX_TIMEOUT_SECONDS:
        raise OuterLaunchError("tmux client timeout is not the frozen R2 value")


def _bind_config(repo_root: Path, config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(repo_root)
    requested = Path(config_path)
    if not requested.is_absolute():
        requested = root / requested
    if requested != root / SMOKE_V2_CONFIG_RELATIVE:
        raise OuterLaunchError("outer launcher requires the frozen R2 config path")
    _strict_file(requested)
    config = load_smoke_config(root, requested)
    if config.get("smoke_id") != SMOKE_V2_ID:
        raise OuterLaunchError("outer launcher requires explicit R2 config")
    _validate_exact_session(config.get("runtime", {}).get("tmux_session_name"))
    _validate_timeout(config.get("runtime", {}).get("tmux_client_timeout_seconds"))
    payload = canonical_json_bytes(config)
    return config, {
        "smoke_id": SMOKE_V2_ID,
        "config_relative_path": requested.relative_to(root).as_posix(),
        "config_path": str(requested),
        "config_size_bytes": requested.stat().st_size,
        "config_file_sha256": sha256_file(requested),
        "config_canonical_sha256": sha256_bytes(payload),
    }


def _validate_runtime_paths(repo_root: Path, config: dict[str, Any], data_root: Path, output: Path, process: Path, outer: Path) -> None:
    root = Path(repo_root)
    _validate_path_components(root, final_kind="dir")
    runtime = config["runtime"]
    if output != root / runtime["output_relative_path"] or process != root / runtime["process_evidence_relative_path"] or outer != root / runtime["outer_launch_evidence_relative_path"]:
        raise OuterLaunchError("R2 runtime path drift")
    _validate_path_components(data_root, final_kind="dir")
    if any(part.casefold() == "test" for part in data_root.parts):
        raise OuterLaunchError("runtime data root may not contain a test component")
    _validate_path_components(output.parent, final_kind="dir")
    _validate_path_components(process.parent, final_kind="dir")
    if output.exists() or output.is_symlink() or process.exists() or process.is_symlink():
        raise OuterLaunchError("R2 child evidence path already exists")


def _write_json(path: Path, value: Any) -> str:
    payload = canonical_json_bytes(value)
    atomic_write(path, payload)
    return sha256_bytes(payload)


def _file_ref_strict(root: Path, name: str, *, required: bool = False) -> dict[str, Any]:
    path = root / name
    if path.is_symlink() or not path.is_file():
        if required:
            raise OuterLaunchError(f"missing evidence file: {name}")
        return {"present": False, "relative_path": name}
    if path.stat().st_nlink != 1:
        raise OuterLaunchError(f"hardlinked evidence file: {name}")
    return {"present": True, "relative_path": name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


_FINALIZER_EXIT_BYTES_RE = re.compile(r"^(0|[1-9][0-9]*)\n$")


def _validate_pane_finalizer_exit(pane: Path) -> int:
    """Read and strictly validate the shell-captured finalizer return code."""
    marker = pane / PANE_FINALIZER_EXIT
    _strict_file(marker)
    try:
        raw = marker.read_bytes()
        decoded = raw.decode("ascii")
    except (OSError, UnicodeError) as exc:
        raise OuterLaunchError("pane finalizer exit code bytes are invalid") from exc
    if _FINALIZER_EXIT_BYTES_RE.fullmatch(decoded) is None:
        raise OuterLaunchError("pane finalizer exit code bytes are invalid")
    try:
        parsed = int(decoded[:-1], 10)
    except ValueError as exc:
        raise OuterLaunchError("pane finalizer exit code bytes are invalid") from exc
    if type(parsed) is not int or parsed < 0 or parsed > 255:
        raise OuterLaunchError("pane finalizer exit code is outside the process range")
    if raw != (str(parsed) + "\n").encode("ascii"):
        raise OuterLaunchError("pane finalizer exit code is not canonical")
    return parsed


def _inventory_value(root: Path, excluded: frozenset[str]) -> dict[str, Any]:
    value = inventory(root, excluded)
    rows = value.get("artifacts")
    if not isinstance(rows, list) or rows != sorted(rows, key=lambda row: row["relative_path"]):
        raise OuterLaunchError("inventory rows are not canonically sorted")
    if value.get("canonical_inventory_sha256") != sha256_bytes(canonical_json_bytes(rows)):
        raise OuterLaunchError("inventory canonical SHA mismatch")
    return value


def _inner_argv(repo: Path, data: Path, output: Path, process: Path, config: Path, child: Path) -> list[str]:
    return [
        str(child), "-u", "-m", "sparse_rtdetr.baseline.smoke_launcher", "smoke",
        "--repo-root", str(repo), "--data-root", str(data), "--output-dir", str(output),
        "--process-evidence-dir", str(process), "--config", str(config), "--child-python", str(child),
    ]


def _build_pane_plan(
    *,
    repo_root: Path,
    data_root: Path,
    output_dir: Path,
    process_evidence_dir: Path,
    outer_evidence_dir: Path,
    config_binding: dict[str, Any],
    nonce: str,
    child_python_identity: dict[str, Any],
    pane_evidence_python_identity: dict[str, Any],
    created_at_utc: str,
) -> dict[str, Any]:
    """Build the sole pane plan representation used by production and validation."""
    _strict_nonce(nonce)
    _validate_executable_identity(child_python_identity, "child Python identity")
    _validate_executable_identity(pane_evidence_python_identity, "pane evidence Python identity")
    inner = _inner_argv(
        repo_root,
        data_root,
        output_dir,
        process_evidence_dir,
        Path(config_binding["config_path"]),
        Path(child_python_identity["canonical_path"]),
    )
    return {
        "schema_version": 1,
        "state": "PREPARED",
        "single_use": True,
        "nonce": nonce,
        "smoke_id": SMOKE_V2_ID,
        "repo_root": str(repo_root),
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "process_evidence_dir": str(process_evidence_dir),
        "outer_evidence_dir": str(outer_evidence_dir),
        "config_binding": config_binding,
        "child_python": child_python_identity["canonical_path"],
        "child_python_identity": child_python_identity,
        "pane_evidence_python": pane_evidence_python_identity["canonical_path"],
        "pane_evidence_python_identity": pane_evidence_python_identity,
        "inner_argv": inner,
        "inner_argv_sha256": argv_sha256(inner),
        "tmux_session": SMOKE_V2_TMUX_SESSION,
        "created_at_utc": created_at_utc,
        "nonportable": True,
    }


_PANE_FINALIZER_CODE = r'''
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from sparse_rtdetr.baseline.smoke_evidence import atomic_write, canonical_json_bytes, file_ref, inventory, sha256_bytes, sha256_file
from sparse_rtdetr.baseline.smoke_outer_launcher import _pane_error_payload, _pane_secondary_payload, _pane_timing_payload

pane = Path(os.environ["P3_PANE_DIR"])
plan_path = pane / "pane_plan.json"
receipt_path = pane / "pane_receipt.json"
lock_path = pane / "pane_consume.lock"
wrapper_path = pane / "pane_wrapper.sh"
config_path = Path(os.environ["P3_PANE_CONFIG"])
inner_started = os.environ.get("P3_PANE_INNER_STARTED") == "true"
inner_rc = int(os.environ.get("P3_PANE_INNER_RC", "125"))
pane_pid = int(os.environ.get("P3_PANE_PID", "0"))
failure_class = os.environ.get("P3_PANE_FAILURE_CLASS", "")
failure_message = os.environ.get("P3_PANE_FAILURE_MESSAGE", "")

def ref(name):
    return file_ref(pane, name)

def secondary(finalizer_rc):
    now = datetime.now(timezone.utc).isoformat()
    value = _pane_secondary_payload(
        pane_pid=pane_pid,
        nonce=os.environ.get("P3_PANE_NONCE", ""),
        plan_sha256=os.environ.get("P3_PANE_PLAN_SHA", ""),
        wrapper_sha256=sha256_file(wrapper_path),
        inner_started=inner_started,
        original_inner_returncode=inner_rc,
        finalizer_returncode=finalizer_rc,
        finalizer_executable_identity=json.loads(os.environ["P3_PANE_FINALIZER_IDENTITY_JSON"]),
        console=ref("pane_console.log"),
        created_at_utc=now,
    )
    if not (pane / "pane_secondary_finalization_failure.json").exists():
        atomic_write(pane / "pane_secondary_finalization_failure.json", canonical_json_bytes(value))

try:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan_bytes = plan_path.read_bytes()
    lock_text = lock_path.read_text(encoding="ascii")
    lock = dict(line.split("=", 1) for line in lock_text.splitlines() if "=" in line)
    started = json.loads((pane / "pane_start.json").read_text(encoding="utf-8"))
    started_at = started["started_at_utc"]
    exit_bytes = (pane / "pane_exit_code.txt").read_bytes()
    if exit_bytes != (str(inner_rc) + "\n").encode("ascii") or started["pane_pid"] != pane_pid:
        raise RuntimeError("pane runtime binding mismatch")
    wrapper_sha = sha256_file(wrapper_path)
    consumed_at = datetime.now(timezone.utc).isoformat()
    receipt = {
        "schema_version": 1,
        "state": "CONSUMED",
        "consumed": True,
        "single_use": True,
        "nonce": plan["nonce"],
        "pane_pid": pane_pid,
        "parent_pid": os.getppid(),
        "plan_relative_path": "pane_plan.json",
        "plan_size_bytes": len(plan_bytes),
        "plan_sha256": sha256_bytes(plan_bytes),
        "wrapper_relative_path": "pane_wrapper.sh",
        "wrapper_size_bytes": wrapper_path.stat().st_size,
        "wrapper_sha256": wrapper_sha,
        "consume_lock_relative_path": "pane_consume.lock",
        "consume_lock_size_bytes": lock_path.stat().st_size,
        "consume_lock_sha256": sha256_file(lock_path),
        "config_relative_path": plan["config_binding"]["config_relative_path"],
        "config_absolute_path": str(config_path),
        "config_size_bytes": plan["config_binding"]["config_size_bytes"],
        "config_file_sha256": plan["config_binding"]["config_file_sha256"],
        "config_canonical_sha256": plan["config_binding"]["config_canonical_sha256"],
        "smoke_id": plan["smoke_id"],
        "tmux_session": plan["tmux_session"],
        "repo_root": plan["repo_root"],
        "runtime_data_root": plan["data_root"],
        "output_path": plan["output_dir"],
        "process_path": plan["process_evidence_dir"],
        "outer_path": plan["outer_evidence_dir"],
        "child_python": plan["child_python"],
        "child_python_identity": plan["child_python_identity"],
        "pane_evidence_python": plan["pane_evidence_python"],
        "pane_evidence_python_identity": plan["pane_evidence_python_identity"],
        "inner_argv_sha256": plan["inner_argv_sha256"],
        "created_at_utc": plan["created_at_utc"],
        "consumed_at_utc": consumed_at,
    }
    if lock.get("nonce") != receipt["nonce"] or lock.get("plan_sha256") != receipt["plan_sha256"] or lock.get("wrapper_sha256") != receipt["wrapper_sha256"] or lock.get("tmux_session") != receipt["tmux_session"]:
        raise RuntimeError("pane consume lock binding mismatch")
    atomic_write(receipt_path, canonical_json_bytes(receipt))
    if not failure_class:
        failure_class = "PREINNER_BINDING_FAILURE" if not inner_started else "INNER_NONZERO_EXIT" if inner_rc != 0 else ""
        failure_message = "pane binding failed before the inner command." if not inner_started else "inner command exited with a nonzero return code."
    if failure_class:
        error = _pane_error_payload(failure_class=failure_class, message=failure_message, inner_started=inner_started, inner_returncode=inner_rc, console=ref("pane_console.log"), nonce=plan["nonce"], pane_pid=pane_pid, plan_sha256=receipt["plan_sha256"], wrapper_sha256=wrapper_sha, created_at_utc=datetime.now(timezone.utc).isoformat())
        error_path = pane / "pane_error.json"
        if error_path.exists():
            existing_error = json.loads(error_path.read_text(encoding="utf-8"))
            if any(existing_error.get(key) != error[key] for key in ("schema_version", "status", "failure_class", "message", "inner_started", "inner_returncode", "original_error_preserved", "console", "nonce", "pane_pid", "plan_sha256", "wrapper_sha256")):
                raise RuntimeError("pane error evidence binding mismatch")
        else:
            atomic_write(error_path, canonical_json_bytes(error))
    finished_at = datetime.now(timezone.utc).isoformat()
    timing = _pane_timing_payload(started_at_utc=started_at, finished_at_utc=finished_at, inner_started=inner_started, inner_returncode=inner_rc, signal_name=None)
    atomic_write(pane / "pane_timing.json", canonical_json_bytes(timing))
    pane_inventory = inventory(pane, frozenset({"pane_completion.json", "pane_inventory.json", "pane_partial_inventory.json", "pane_secondary_finalization_failure.json", "pane_finalizer_exit_code.txt"}))
    atomic_write(pane / "pane_inventory.json", canonical_json_bytes(pane_inventory))
    status = "PANE_COMPLETED" if inner_started and inner_rc == 0 and not failure_class else "PANE_FAILED"
    completion = {
        "schema_version": 1,
        "status": status,
        "inner_started": inner_started,
        "inner_returncode": inner_rc,
        "pane_pid": pane_pid,
        "inner_argv_sha256": plan["inner_argv_sha256"],
        "plan_sha256": receipt["plan_sha256"],
        "wrapper_sha256": wrapper_sha,
        "nonce": receipt["nonce"],
        "tmux_session": receipt["tmux_session"],
        "child_python_identity": plan["child_python_identity"],
        "pane_evidence_python_identity": plan["pane_evidence_python_identity"],
        "pane_receipt": ref("pane_receipt.json"),
        "consume_lock": ref("pane_consume.lock"),
        "pane_inventory": ref("pane_inventory.json"),
        "pane_timing": ref("pane_timing.json"),
        "error": ref("pane_error.json"),
    }
    atomic_write(pane / "pane_completion.json", canonical_json_bytes(completion))
except BaseException:
    try:
        secondary(1)
    except BaseException:
        pass
    raise
'''


def _build_pane_wrapper_bytes(
    pane: Path,
    plan: Path,
    plan_sha: str,
    config: Path,
    config_sha: str,
    inner: list[str],
    environment: dict[str, str],
    pane_evidence_python_identity: dict[str, Any],
) -> bytes:
    """Build deterministic wrapper bytes from validated launch inputs."""
    _validate_executable_identity(pane_evidence_python_identity, "pane evidence Python identity")
    env_lines = "\n".join(f"export {key}={shlex.quote(value)}" for key, value in sorted(environment.items()))
    inner_cmd = shlex.join(inner)
    finalizer_identity_json = canonical_json_bytes(pane_evidence_python_identity).decode("ascii")
    value = f'''#!/bin/sh
set -u
PANE={shlex.quote(str(pane))}
PLAN={shlex.quote(str(plan))}
CONFIG={shlex.quote(str(config))}
PLAN_SHA={shlex.quote(plan_sha)}
CONFIG_SHA={shlex.quote(config_sha)}
LOCK="$PANE/{PANE_LOCK}"
CONSOLE="$PANE/{PANE_CONSOLE}"
START="$PANE/{PANE_START}"
EXIT="$PANE/{PANE_EXIT}"
ERROR="$PANE/{PANE_ERROR}"
SECONDARY="$PANE/{PANE_SECONDARY_FAILURE}"
FINALIZER_EXIT="$PANE/{PANE_FINALIZER_EXIT}"
INNER_STARTED=false
INNER_RC=125
FAILURE_CLASS=''
FAILURE_MESSAGE=''
STARTED_AT=$(date -u +%Y-%m-%dT%H:%M:%S.%N 2>/dev/null)
STARTED_AT="${{STARTED_AT%???}}Z"
{env_lines}
export P3_PANE_DIR="$PANE"
export P3_PANE_CONFIG="$CONFIG"
export P3_PANE_PID="$$"
export P3_PANE_STARTED_AT="$STARTED_AT"
export P3_PANE_INNER_STARTED=false
export P3_PANE_INNER_RC=125
export P3_PANE_ERROR=
export P3_PANE_FAILURE_CLASS=
export P3_PANE_FAILURE_MESSAGE=
export P3_PANE_PLAN_SHA="$PLAN_SHA"
export P3_PANE_FINALIZER_IDENTITY_JSON={shlex.quote(finalizer_identity_json)}

# The ownership gate is before every write, including lock, console, and start.
for RUNTIME_NAME in {shlex.join(list(PANE_RUNTIME_EVIDENCE))}; do
  if test -e "$PANE/$RUNTIME_NAME" || test -L "$PANE/$RUNTIME_NAME"; then
    printf '%s\n' 'pane runtime evidence already exists' >&2
    exit 73
  fi
done

# noclobber is the POSIX exclusive-create operation used for single consume.
set -C
if ! (umask 022; printf 'nonce=%s\nplan_sha256=%s\nwrapper_sha256=%s\ntmux_session=%s\n' "$P3_PANE_NONCE" "$PLAN_SHA" "$(sha256sum "$0" 2>/dev/null | awk '{{print $1}}')" "$P3_SMOKE_TMUX_SESSION" > "$LOCK"); then
  exit 73
fi
set +C
: > "$CONSOLE"
printf '{{"schema_version":1,"state":"STARTED","nonce":"%s","pane_pid":%s,"plan_relative_path":"pane_plan.json","plan_sha256":"%s","started_at_utc":"%s"}}\n' "$P3_PANE_NONCE" "$$" "$PLAN_SHA" "$STARTED_AT" > "$START"
printf '%s\n' 'P3_SMOKE_PANE_SHELL_STARTED' >> "$CONSOLE"
if test ! -f "$PLAN" || test "$(sha256sum "$PLAN" 2>/dev/null | awk '{{print $1}}')" != "$PLAN_SHA"; then
  FAILURE_CLASS='PREINNER_BINDING_FAILURE'
  FAILURE_MESSAGE='pane binding failed before the inner command.'
elif test ! -f "$CONFIG" || test "$(sha256sum "$CONFIG" 2>/dev/null | awk '{{print $1}}')" != "$CONFIG_SHA"; then
  FAILURE_CLASS='PREINNER_BINDING_FAILURE'
  FAILURE_MESSAGE='pane binding failed before the inner command.'
fi
if test -z "$FAILURE_CLASS"; then
  INNER_STARTED=true
  export P3_PANE_INNER_STARTED=true
  set +e
  {inner_cmd} >> "$CONSOLE" 2>&1
  INNER_RC=$?
  set -u
  export P3_PANE_INNER_RC="$INNER_RC"
  if test "$INNER_RC" -ne 0; then
    FAILURE_CLASS='INNER_NONZERO_EXIT'
    FAILURE_MESSAGE='inner command exited with a nonzero return code.'
  fi
else
  export P3_PANE_FAILURE_CLASS="$FAILURE_CLASS"
  export P3_PANE_FAILURE_MESSAGE="$FAILURE_MESSAGE"
fi
if test "$INNER_RC" -ne 0 && test -z "$FAILURE_CLASS"; then
  FAILURE_CLASS='INNER_NONZERO_EXIT'
  FAILURE_MESSAGE='inner command exited with a nonzero return code.'
fi
export P3_PANE_FAILURE_CLASS="$FAILURE_CLASS"
export P3_PANE_FAILURE_MESSAGE="$FAILURE_MESSAGE"
if test -n "$FAILURE_CLASS"; then
  CONSOLE_SIZE=$(wc -c < "$CONSOLE" 2>/dev/null || printf '0')
  CONSOLE_SHA=$(sha256sum "$CONSOLE" 2>/dev/null | awk '{{print $1}}' || printf '%s' '')
  ERROR_NOW=$(date -u +%Y-%m-%dT%H:%M:%S.%N 2>/dev/null)
  ERROR_NOW="${{ERROR_NOW%???}}Z"
  printf '{{"schema_version":1,"status":"PANE_FAILED","failure_class":"%s","message":"%s","inner_started":%s,"inner_returncode":%s,"original_error_preserved":true,"console":{{"present":true,"relative_path":"pane_console.log","size_bytes":%s,"sha256":"%s"}},"nonce":"%s","pane_pid":%s,"plan_sha256":"%s","wrapper_sha256":"%s","created_at_utc":"%s"}}\n' "$FAILURE_CLASS" "$FAILURE_MESSAGE" "$INNER_STARTED" "$INNER_RC" "$CONSOLE_SIZE" "$CONSOLE_SHA" "$P3_PANE_NONCE" "$$" "$PLAN_SHA" "$(sha256sum "$0" 2>/dev/null | awk '{{print $1}}')" "$ERROR_NOW" > "$ERROR"
fi
printf '%s\n' "$INNER_RC" > "$EXIT"
set +e
{shlex.quote(environment["PANE_EVIDENCE_PYTHON"])} -c {shlex.quote(_PANE_FINALIZER_CODE)} >/dev/null 2>&1
FINALIZER_RC=$?
if ! (umask 022; set -C; printf '%s\n' "$FINALIZER_RC" > "$FINALIZER_EXIT") 2>/dev/null; then
  set -u
  exit 74
fi
set -u
if test "$FINALIZER_RC" -ne 0; then
  if test ! -e "$SECONDARY" && test ! -L "$SECONDARY"; then
    CONSOLE_SIZE=$(wc -c < "$CONSOLE" 2>/dev/null || printf '0')
    CONSOLE_SHA=$(sha256sum "$CONSOLE" 2>/dev/null | awk '{{print $1}}' || printf '%s' '')
    FINALIZER_NOW=$(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || printf '%s' "$STARTED_AT")
    SECONDARY_PAYLOAD=$(printf '{{"schema_version":1,"status":"FAILED_FINALIZATION","failure_class":"PANE_FINALIZATION_FAILURE","pane_pid":%s,"nonce":"%s","plan_sha256":"%s","wrapper_sha256":"%s","inner_started":%s,"original_inner_returncode":%s,"finalizer_returncode":%s,"finalizer_executable_identity":%s,"original_error_preserved":true,"console":{{"present":true,"relative_path":"pane_console.log","size_bytes":%s,"sha256":"%s"}},"created_at_utc":"%s"}}' "$$" "$P3_PANE_NONCE" "$PLAN_SHA" "$(sha256sum "$0" 2>/dev/null | awk '{{print $1}}')" "$INNER_STARTED" "$INNER_RC" "$FINALIZER_RC" "$P3_PANE_FINALIZER_IDENTITY_JSON" "$CONSOLE_SIZE" "$CONSOLE_SHA" "$FINALIZER_NOW")
    (set -C; printf '%s\n' "$SECONDARY_PAYLOAD" > "$SECONDARY") 2>/dev/null || true
  fi
  exit 74
fi
exit "$INNER_RC"
'''
    return value.encode("ascii")


def _wrapper(pane: Path, plan: Path, plan_sha: str, config: Path, config_sha: str, inner: list[str], environment: dict[str, str], pane_evidence_python_identity: dict[str, Any] | None = None) -> str:
    if pane_evidence_python_identity is None:
        pane_evidence_python_identity = executable_identity(Path(environment["PANE_EVIDENCE_PYTHON"]))
    return _build_pane_wrapper_bytes(pane, plan, plan_sha, config, config_sha, inner, environment, pane_evidence_python_identity).decode("ascii")


def _run_tmux(*, argv: list[str], cwd: Path, env: dict[str, str], timeout_seconds: int) -> dict[str, Any]:
    state: dict[str, Any] = {"process": None, "events": [], "term_sent": False, "kill_sent": False}
    original_exception: BaseException | None = None
    timeout_triggered = False
    stdout: bytes = b""
    stderr: bytes = b""
    returncode: int | None = None
    previous = {signum: signal.getsignal(signum) for signum in _MONITORED_SIGNALS}

    def handle_signal(signum: int, _frame: Any) -> None:
        process = state["process"]
        event: dict[str, Any] = {
            "signal_number": signum,
            "signal_name": signal.Signals(signum).name,
            "received_at_utc": _utc_now(),
            "forwarded_to_tmux_client": False,
            "forwarding_result": "not_running",
        }
        if process is not None and getattr(process, "poll", lambda: 0)() is None:
            try:
                os.killpg(process.pid, signum)
                event["forwarded_to_tmux_client"] = True
                event["forwarding_result"] = "forwarded"
            except OSError as exc:
                event["forwarding_result"] = f"failed:{type(exc).__name__}"
        state["events"].append(event)

    installed = False
    try:
        for signum in _MONITORED_SIGNALS:
            signal.signal(signum, handle_signal)
        installed = True
        process = subprocess.Popen(
            argv,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        state["process"] = process
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            timeout_triggered = True
            stdout = exc.output or b""
            stderr = exc.stderr or b""
            try:
                os.killpg(process.pid, signal.SIGTERM)
                state["term_sent"] = True
            except OSError:
                pass
            try:
                more_out, more_err = process.communicate(timeout=2.0)
                stdout = (stdout or b"") + (more_out or b"")
                stderr = (stderr or b"") + (more_err or b"")
            except subprocess.TimeoutExpired as second:
                stdout = (stdout or b"") + (second.output or b"")
                stderr = (stderr or b"") + (second.stderr or b"")
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                    state["kill_sent"] = True
                except OSError:
                    pass
                more_out, more_err = process.communicate()
                stdout = (stdout or b"") + (more_out or b"")
                stderr = (stderr or b"") + (more_err or b"")
        returncode = process.returncode
    except BaseException as exc:
        original_exception = exc
        process = state["process"]
        if process is not None and getattr(process, "poll", lambda: 0)() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                state["term_sent"] = True
            except OSError:
                pass
            try:
                stdout, stderr = process.communicate(timeout=2.0)
            except BaseException:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                    state["kill_sent"] = True
                except OSError:
                    pass
                try:
                    stdout, stderr = process.communicate()
                except BaseException:
                    pass
        if process is not None:
            returncode = getattr(process, "returncode", None)
    finally:
        try:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
        except BaseException as exc:
            if original_exception is None:
                original_exception = exc
    return {
        "stdout": stdout if isinstance(stdout, bytes) else str(stdout or "").encode(),
        "stderr": stderr if isinstance(stderr, bytes) else str(stderr or "").encode(),
        "returncode": returncode,
        "signal_events": state["events"],
        "signal_handlers_installed": installed,
        "monitored_signals": list(_MONITORED_SIGNAL_NAMES),
        "observed_signal_count": len(state["events"]),
        "timeout_triggered": timeout_triggered,
        "term_sent": bool(state["term_sent"]),
        "kill_sent": bool(state["kill_sent"]),
        "original_exception": None if original_exception is None else {"type": type(original_exception).__name__, "message": str(original_exception)},
    }


def _runner_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "signal_handlers_installed": result["signal_handlers_installed"],
        "monitored_signals": result["monitored_signals"],
        "observed_signal_count": result["observed_signal_count"],
        "signal_events": result["signal_events"],
        "timeout_triggered": result["timeout_triggered"],
        "term_sent": result["term_sent"],
        "kill_sent": result["kill_sent"],
        "actual_returncode": result["returncode"],
        "original_exception": result["original_exception"],
    }


def _safe_secondary(root: Path, original: BaseException | None, failures: list[dict[str, str]], name: str) -> None:
    first_failure = failures[0] if failures else {"type": "UnknownError", "message": "unknown finalization failure"}
    payload = {
        "schema_version": 1,
        "status": "FAILED_FINALIZATION",
        "original_exception_type": type(original).__name__ if original is not None else first_failure["type"],
        "original_exception_message": str(original) if original is not None else first_failure["message"],
        "finalization_failures": failures,
        "original_error_preserved": True,
    }
    try:
        _write_json(root / name, payload)
    except BaseException:
        pass


def _finish_outer(outer: Path, launcher: Path, invocation: dict[str, Any], status: str, runner: dict[str, Any], original: BaseException | None = None) -> dict[str, Any]:
    failures: list[dict[str, str]] = []

    def attempt(label: str, action):
        try:
            return action()
        except BaseException as exc:
            failures.append({"step": label, "type": type(exc).__name__, "message": str(exc)})
            return None

    error_record = None if original is None else {"type": type(original).__name__, "message": str(original)}
    if error_record is None and runner.get("original_exception") is not None:
        error_record = runner["original_exception"]
    if error_record is not None:
        attempt("outer_error", lambda: _write_json(launcher / "outer_error.json", {
            "schema_version": 1,
            "status": "FAILED",
            "exception_type": error_record["type"],
            "message": error_record["message"],
            "original_error_preserved": True,
        }))
    attempt("outer_timing", lambda: _write_json(launcher / "outer_timing.json", {
        "schema_version": 1,
        "started_at_utc": invocation["created_at_utc"],
        "finished_at_utc": _utc_now(),
        "runner": _runner_summary(runner),
    }))
    attempt("outer_partial_inventory", lambda: _write_json(launcher / OUTER_PARTIAL_INVENTORY, _inventory_value(launcher, OUTER_EXCLUDED)))
    inventory_sha = None
    inventory_ref: dict[str, Any] = {"present": False, "relative_path": OUTER_INVENTORY}
    if not failures:
        def write_inventory() -> str:
            value = _inventory_value(launcher, OUTER_EXCLUDED)
            payload = canonical_json_bytes(value)
            atomic_write(launcher / OUTER_INVENTORY, payload)
            return sha256_bytes(payload)
        inventory_sha = attempt("outer_inventory", write_inventory)
        if inventory_sha is not None:
            inventory_ref = _file_ref_strict(launcher, OUTER_INVENTORY, required=True)
    if failures:
        _safe_secondary(launcher, original, failures, OUTER_SECONDARY_FAILURE)
        return {"status": "FINALIZATION_FAILED", "outer_evidence_dir": str(outer), "finalization_failures": failures}
    completion = {
        "schema_version": 1,
        "status": status,
        "smoke_id": invocation["smoke_id"],
        "tmux_session": invocation["tmux_session"],
        "nonce": invocation.get("nonce"),
        "tmux_called": (launcher / "tmux_invocation.json").is_file(),
        "tmux_returncode": runner["returncode"],
        "runner": _runner_summary(runner),
        "invocation": _file_ref_strict(launcher, "outer_invocation.json", required=True),
        "config_binding": _file_ref_strict(launcher, "outer_config_binding.json"),
        "preflight": _file_ref_strict(launcher, "preflight_audit.json"),
        "tmux_invocation": _file_ref_strict(launcher, "tmux_invocation.json"),
        "tmux_stdout": _file_ref_strict(launcher, "tmux_stdout.log"),
        "tmux_stderr": _file_ref_strict(launcher, "tmux_stderr.log"),
        "outer_inventory": inventory_ref,
        "outer_partial_inventory": _file_ref_strict(launcher, OUTER_PARTIAL_INVENTORY, required=True),
        "outer_timing": _file_ref_strict(launcher, "outer_timing.json", required=True),
        "error": _file_ref_strict(launcher, "outer_error.json"),
        "failure": error_record,
        "finished_at_utc": _utc_now(),
    }
    completion_sha = attempt("outer_completion", lambda: _write_json(launcher / OUTER_COMPLETION, completion))
    if failures:
        _safe_secondary(launcher, original, failures, OUTER_SECONDARY_FAILURE)
        return {"status": "FINALIZATION_FAILED", "outer_evidence_dir": str(outer), "finalization_failures": failures}
    return {"status": status, "completion_sha256": completion_sha, "outer_evidence_dir": str(outer)}


def run_outer_launch(*, repo_root: Path, data_root: Path, config_path: Path, output_dir: Path, process_evidence_dir: Path, outer_evidence_dir: Path, child_python: Path, tmux_executable: Path, tmux_session: str) -> dict[str, Any]:
    """Prepare evidence and issue exactly one bounded tmux client call."""

    repo_root = Path(repo_root)
    data_root = Path(data_root)
    output = Path(output_dir)
    process = Path(process_evidence_dir)
    outer = Path(outer_evidence_dir)
    child = Path(child_python)
    tmux = Path(tmux_executable)
    _validate_outer_parent(outer)
    outer.mkdir(mode=0o775)
    outer.chmod(0o775)
    launcher = outer / "launcher"
    pane = outer / "pane"
    launcher.mkdir(mode=0o775)
    pane.mkdir(mode=0o775)
    _pane_preexisting_evidence_gate(pane)
    created_at = _utc_now()
    config_arg = Path(config_path)
    if not config_arg.is_absolute():
        config_arg = repo_root / config_arg
    try:
        pane_evidence_identity = executable_identity(Path(sys.executable))
    except BaseException:
        pane_evidence_identity = None
    try:
        child_identity = executable_identity(child)
    except BaseException:
        child_identity = None
    try:
        tmux_identity = executable_identity(tmux)
    except BaseException:
        tmux_identity = None
    child_for_invocation = child_identity["canonical_path"] if child_identity is not None else str(child)
    tmux_for_invocation = tmux_identity["canonical_path"] if tmux_identity is not None else str(tmux)
    nonce_inner = _inner_argv(repo_root, data_root, output, process, config_arg, Path(child_for_invocation))
    nonce = sha256_bytes(canonical_json_bytes({"created_at_utc": created_at, "inner_argv_sha256": argv_sha256(nonce_inner), "tmux_session": tmux_session}))[:32]
    pane_evidence_path = pane_evidence_identity["canonical_path"] if pane_evidence_identity is not None else str(Path(sys.executable).resolve())
    invocation = {
        "schema_version": 1,
        "mode": "outer_launch",
        "status": "OUTER_PREPARED",
        "nonportable": True,
        "created_at_utc": created_at,
        "nonce": nonce,
        "smoke_id": SMOKE_V2_ID,
        "repo_root": str(repo_root),
        "data_root": str(data_root),
        "config_path": str(config_arg),
        "output_dir": str(output),
        "process_evidence_dir": str(process),
        "outer_evidence_dir": str(outer),
        "child_python": child_for_invocation,
        "child_python_identity": child_identity,
        "pane_evidence_python": pane_evidence_path,
        "pane_evidence_python_identity": pane_evidence_identity,
        "tmux_executable": tmux_for_invocation,
        "tmux_session": tmux_session,
        "environment": {
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "P3_RTDETR_BASELINE_SMOKE_AUTHORIZED": "1",
            "PYTHONPATH": str(repo_root / "src"),
            "P3_SMOKE_TMUX_SESSION": SMOKE_V2_TMUX_SESSION,
            "PANE_EVIDENCE_PYTHON": pane_evidence_path,
        },
    }
    _write_json(launcher / "outer_invocation.json", invocation)
    runner: dict[str, Any] = {"stdout": b"", "stderr": b"", "returncode": None, "signal_events": [], "signal_handlers_installed": False, "monitored_signals": list(_MONITORED_SIGNAL_NAMES), "observed_signal_count": 0, "timeout_triggered": False, "term_sent": False, "kill_sent": False, "original_exception": None}
    try:
        _validate_exact_session(tmux_session)
        _validate_path_components(repo_root, final_kind="dir")
        config, binding = _bind_config(repo_root, config_arg)
        _validate_runtime_paths(repo_root, config, data_root, output, process, outer)
        if child_identity is None or pane_evidence_identity is None or tmux_identity is None:
            raise OuterLaunchError("executable identity preflight failed")
        child = Path(child_identity["canonical_path"])
        tmux = Path(tmux_identity["canonical_path"])
        for path, field in ((repo_root, "repo_root"), (data_root, "data_root"), (output, "output_dir"), (process, "process_evidence_dir"), (outer, "outer_evidence_dir"), (child, "child_python"), (tmux, "tmux_executable")):
            if any(ord(char) > 127 for char in str(path)):
                raise OuterLaunchError(f"{field} must be ASCII")
        invocation["child_python"] = child_identity["canonical_path"]
        invocation["child_python_identity"] = child_identity
        invocation["tmux_executable"] = tmux_identity["canonical_path"]
        invocation["pane_evidence_python"] = pane_evidence_identity["canonical_path"]
        invocation["pane_evidence_python_identity"] = pane_evidence_identity
        _write_json(launcher / "outer_config_binding.json", binding)
        inner = _inner_argv(repo_root, data_root, output, process, Path(binding["config_path"]), child)
        inner_sha = argv_sha256(inner)
        if inner_sha != argv_sha256(nonce_inner):
            raise OuterLaunchError("inner argv changed during executable canonicalization")
        plan = _build_pane_plan(
            repo_root=repo_root,
            data_root=data_root,
            output_dir=output,
            process_evidence_dir=process,
            outer_evidence_dir=outer,
            config_binding=binding,
            nonce=nonce,
            child_python_identity=child_identity,
            pane_evidence_python_identity=pane_evidence_identity,
            created_at_utc=created_at,
        )
        plan_sha = _write_json(pane / "pane_plan.json", plan)
        pane_environment = dict(invocation["environment"])
        pane_environment.update({"P3_PANE_NONCE": nonce, "P3_SMOKE_TMUX_SESSION": tmux_session})
        wrapper_path = pane / PANE_WRAPPER
        wrapper_bytes = _build_pane_wrapper_bytes(pane, pane / "pane_plan.json", plan_sha, Path(binding["config_path"]), binding["config_file_sha256"], inner, pane_environment, pane_evidence_identity)
        atomic_write(wrapper_path, wrapper_bytes)
        wrapper_path.chmod(0o755)
        tmux_argv = [str(tmux), "new-session", "-d", "-s", tmux_session, "-c", str(repo_root), str(wrapper_path)]
        _write_json(launcher / "preflight_audit.json", {"schema_version": 1, "status": "PASS", "config_binding": binding, "data_root": str(data_root), "output_process_not_created_by_outer": True, "tmux_executable_identity": tmux_identity, "child_python_identity": child_identity, "pane_evidence_python_identity": pane_evidence_identity, "environment": invocation["environment"], "wrapper_relative_path": "../pane/pane_wrapper.sh", "wrapper_sha256": sha256_bytes(wrapper_bytes), "pane_plan_relative_path": "../pane/pane_plan.json", "pane_plan_sha256": plan_sha, "inner_argv_sha256": inner_sha, "nonce": nonce, "tmux_session": tmux_session, "tmux_client_timeout_seconds": config["runtime"]["tmux_client_timeout_seconds"], "created_at_utc": created_at})
        _write_json(launcher / "tmux_invocation.json", {"schema_version": 1, "argv": tmux_argv, "argv_sha256": argv_sha256(tmux_argv), "wrapper_sha256": sha256_bytes(wrapper_bytes), "pane_plan_sha256": plan_sha, "nonce": nonce, "tmux_session": tmux_session, "timeout_seconds": config["runtime"]["tmux_client_timeout_seconds"], "called_at_utc": _utc_now(), "tmux_executable_identity": tmux_identity, "child_python_identity": child_identity, "pane_evidence_python_identity": pane_evidence_identity})
        runner = _run_tmux(argv=tmux_argv, cwd=repo_root, env={**os.environ, **pane_environment}, timeout_seconds=config["runtime"]["tmux_client_timeout_seconds"])
        atomic_write(launcher / "tmux_stdout.log", runner["stdout"])
        atomic_write(launcher / "tmux_stderr.log", runner["stderr"])
        status = "TMUX_ACCEPTED" if runner["returncode"] == 0 and not runner["timeout_triggered"] and runner["original_exception"] is None else "TMUX_REJECTED" if runner["returncode"] is not None and not runner["timeout_triggered"] else "INTERRUPTED_WITH_EVIDENCE"
        return _finish_outer(outer, launcher, invocation | {"nonce": nonce}, status, runner)
    except BaseException as error:
        if runner["original_exception"] is None:
            runner["original_exception"] = {"type": type(error).__name__, "message": str(error)}
        status = "PREFLIGHT_FAILED" if not (launcher / "tmux_invocation.json").is_file() else "INTERRUPTED_WITH_EVIDENCE"
        return _finish_outer(outer, launcher, invocation, status, runner, error)


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _verify_ref(root: Path, value: Any, name: str, *, required: bool = True) -> None:
    if not isinstance(value, dict):
        raise OuterLaunchError(f"{name} reference is invalid")
    expected = _file_ref_strict(root, name, required=required)
    if value != expected:
        raise OuterLaunchError(f"{name} reference binding mismatch")


def validate_outer_evidence(outer_evidence_dir: Path, output_dir: Path | None = None, process_evidence_dir: Path | None = None) -> dict[str, Any]:
    root = Path(outer_evidence_dir)
    _validate_path_components(root, final_kind="dir")
    launcher = root / "launcher"
    pane = root / "pane"
    _validate_path_components(launcher, final_kind="dir")
    _validate_path_components(pane, final_kind="dir")
    completion = _read_json_object(launcher / OUTER_COMPLETION)
    invocation = _read_json_object(launcher / "outer_invocation.json")
    if completion is None or invocation is None or completion.get("schema_version") != 1 or invocation.get("schema_version") != 1 or invocation.get("smoke_id") != SMOKE_V2_ID:
        raise OuterLaunchError("outer invocation/completion schema is invalid")
    _validate_outer_invocation(invocation, root)
    _verify_ref(launcher, completion.get("invocation"), "outer_invocation.json")
    status = completion.get("status")
    if status not in {"PREFLIGHT_FAILED", "TMUX_REJECTED", "TMUX_ACCEPTED", "INTERRUPTED_WITH_EVIDENCE", "FINALIZATION_FAILED"}:
        raise OuterLaunchError("outer status is invalid")
    runner = completion.get("runner")
    if not isinstance(runner, dict):
        raise OuterLaunchError("outer runner evidence is missing")
    _validate_signal_events(runner, status)
    if completion.get("tmux_returncode") != runner.get("actual_returncode"):
        raise OuterLaunchError("outer return code binding mismatch")
    if completion.get("tmux_session") != invocation.get("tmux_session"):
        raise OuterLaunchError("outer session binding mismatch")
    if status != "PREFLIGHT_FAILED":
        _validate_exact_session(invocation.get("tmux_session"))
        if completion.get("nonce") is None:
            raise OuterLaunchError("outer nonce is missing")
        _strict_nonce(completion.get("nonce"))
        _verify_ref(launcher, completion.get("config_binding"), "outer_config_binding.json")
        config_binding = _read_json_object(launcher / "outer_config_binding.json")
        if config_binding is None or config_binding.get("smoke_id") != SMOKE_V2_ID or config_binding.get("config_relative_path") != SMOKE_V2_CONFIG_RELATIVE:
            raise OuterLaunchError("outer config binding is invalid")
        _verify_ref(launcher, completion.get("preflight"), "preflight_audit.json")
        preflight = _read_json_object(launcher / "preflight_audit.json")
        _verify_ref(launcher, completion.get("tmux_invocation"), "tmux_invocation.json")
        tmux_invocation = _read_json_object(launcher / "tmux_invocation.json")
        if preflight is None or tmux_invocation is None:
            raise OuterLaunchError("outer launch binding files are invalid")
        config_root = Path(invocation["repo_root"])
        config_path = Path(invocation["config_path"])
        _validate_path_components(config_root, final_kind="dir")
        _validate_path_components(Path(invocation["data_root"]), final_kind="dir")
        config, actual_binding = _bind_config(config_root, config_path)
        if config_binding != actual_binding:
            raise OuterLaunchError("outer config binding does not match config bytes")
        expected_runtime_paths = {
            "output_dir": str(config_root / config["runtime"]["output_relative_path"]),
            "process_evidence_dir": str(config_root / config["runtime"]["process_evidence_relative_path"]),
            "outer_evidence_dir": str(config_root / config["runtime"]["outer_launch_evidence_relative_path"]),
        }
        for field, expected_path in expected_runtime_paths.items():
            if invocation[field] != expected_path:
                raise OuterLaunchError(f"outer runtime path binding mismatch: {field}")
        _validate_path_components(Path(expected_runtime_paths["output_dir"]).parent, final_kind="dir")
        _validate_path_components(Path(expected_runtime_paths["process_evidence_dir"]).parent, final_kind="dir")
        if output_dir is not None and str(output_dir) != expected_runtime_paths["output_dir"]:
            raise OuterLaunchError("expected output path binding mismatch")
        if process_evidence_dir is not None and str(process_evidence_dir) != expected_runtime_paths["process_evidence_dir"]:
            raise OuterLaunchError("expected process path binding mismatch")
        if completion.get("nonce") != invocation.get("nonce"):
            raise OuterLaunchError("outer completion nonce does not bind invocation")
        child_identity = executable_identity(Path(invocation["child_python"]))
        pane_evidence_identity = _bound_pane_evidence_identity(pane, invocation)
        if invocation["child_python_identity"] != child_identity:
            raise OuterLaunchError("outer executable identity has drifted")
        if invocation["config_path"] != actual_binding["config_path"] or config_binding != actual_binding:
            raise OuterLaunchError("outer config path binding mismatch")
        expected_plan = _build_pane_plan(
            repo_root=config_root,
            data_root=Path(invocation["data_root"]),
            output_dir=Path(expected_runtime_paths["output_dir"]),
            process_evidence_dir=Path(expected_runtime_paths["process_evidence_dir"]),
            outer_evidence_dir=root,
            config_binding=actual_binding,
            nonce=invocation["nonce"],
            child_python_identity=child_identity,
            pane_evidence_python_identity=pane_evidence_identity,
            created_at_utc=invocation["created_at_utc"],
        )
        expected_nonce = sha256_bytes(canonical_json_bytes({"created_at_utc": invocation["created_at_utc"], "inner_argv_sha256": expected_plan["inner_argv_sha256"], "tmux_session": invocation["tmux_session"]}))[:32]
        if invocation["nonce"] != expected_nonce:
            raise OuterLaunchError("outer nonce is not deterministically bound to the plan inputs")
        expected_plan_sha = sha256_bytes(canonical_json_bytes(expected_plan))
        expected_environment = dict(invocation["environment"])
        expected_environment.update({"P3_PANE_NONCE": invocation["nonce"], "P3_SMOKE_TMUX_SESSION": invocation["tmux_session"]})
        expected_wrapper = _build_pane_wrapper_bytes(
            pane,
            pane / "pane_plan.json",
            expected_plan_sha,
            Path(actual_binding["config_path"]),
            actual_binding["config_file_sha256"],
            expected_plan["inner_argv"],
            expected_environment,
            pane_evidence_identity,
        )
        plan, plan_sha, wrapper_sha = _validate_pane_plan_identity(pane, expected_plan, expected_wrapper)
        if preflight.get("pane_plan_sha256") != plan_sha or preflight.get("wrapper_sha256") != wrapper_sha:
            raise OuterLaunchError("preflight does not bind actual pane files")
        if preflight.get("pane_plan_relative_path") != "../pane/pane_plan.json" or preflight.get("wrapper_relative_path") != "../pane/pane_wrapper.sh":
            raise OuterLaunchError("preflight pane paths are invalid")
        for value, field in ((preflight.get("nonce"), "preflight nonce"), (tmux_invocation.get("nonce"), "tmux invocation nonce")):
            _strict_nonce(value, field)
            if value != completion["nonce"]:
                raise OuterLaunchError(f"{field} mismatch")
        if preflight.get("tmux_session") != invocation["tmux_session"] or tmux_invocation.get("tmux_session") != invocation["tmux_session"]:
            raise OuterLaunchError("outer launch session binding mismatch")
        if preflight.get("environment") != invocation["environment"]:
            raise OuterLaunchError("preflight environment binding mismatch")
        if preflight.get("child_python_identity") != child_identity or preflight.get("pane_evidence_python_identity") != pane_evidence_identity:
            raise OuterLaunchError("preflight executable identity binding mismatch")
        if tmux_invocation.get("child_python_identity") != child_identity or tmux_invocation.get("pane_evidence_python_identity") != pane_evidence_identity:
            raise OuterLaunchError("tmux executable identity binding mismatch")
        _validate_timeout(preflight.get("tmux_client_timeout_seconds"))
        _validate_timeout(tmux_invocation.get("timeout_seconds"))
        if tmux_invocation.get("argv_sha256") != argv_sha256(tmux_invocation.get("argv")):
            raise OuterLaunchError("tmux argv SHA mismatch")
        if preflight.get("pane_plan_sha256") != tmux_invocation.get("pane_plan_sha256") or preflight.get("wrapper_sha256") != tmux_invocation.get("wrapper_sha256") or tmux_invocation.get("pane_plan_sha256") != plan_sha or tmux_invocation.get("wrapper_sha256") != wrapper_sha:
            raise OuterLaunchError("outer plan/wrapper binding mismatch")
        tmux_path = Path(invocation["tmux_executable"])
        tmux_identity = executable_identity(tmux_path)
        if invocation["tmux_executable"] != tmux_invocation.get("argv", [None])[0] or tmux_invocation.get("tmux_executable_identity") != tmux_identity or preflight.get("tmux_executable_identity") != tmux_identity:
            raise OuterLaunchError("tmux executable identity binding mismatch")
        expected_tmux_argv = [str(tmux_path), "new-session", "-d", "-s", invocation["tmux_session"], "-c", invocation["repo_root"], str(pane / PANE_WRAPPER)]
        if tmux_invocation.get("argv") != expected_tmux_argv or tmux_invocation.get("argv_sha256") != argv_sha256(expected_tmux_argv):
            raise OuterLaunchError("tmux invocation argv binding mismatch")
    _verify_ref(launcher, completion.get("outer_timing"), "outer_timing.json")
    _verify_ref(launcher, completion.get("outer_partial_inventory"), OUTER_PARTIAL_INVENTORY)
    if status != "FINALIZATION_FAILED":
        _verify_ref(launcher, completion.get("outer_inventory"), OUTER_INVENTORY)
        value = _read_json_object(launcher / OUTER_INVENTORY)
        if value is None or value.get("canonical_inventory_sha256") != sha256_bytes(canonical_json_bytes(value.get("artifacts"))):
            raise OuterLaunchError("outer inventory is not canonical")
        if value != _inventory_value(launcher, OUTER_EXCLUDED):
            raise OuterLaunchError("outer inventory does not bind current files")
    if status == "PREFLIGHT_FAILED":
        if completion.get("tmux_called") is not False or completion.get("tmux_returncode") is not None:
            raise OuterLaunchError("preflight failure falsely claims tmux execution")
    else:
        if completion.get("tmux_called") is not True:
            raise OuterLaunchError("tmux call evidence is missing")
        if status in {"TMUX_ACCEPTED", "TMUX_REJECTED"} and type(completion.get("tmux_returncode")) is not int:
            raise OuterLaunchError("tmux return code must be a real integer")
        if status == "TMUX_ACCEPTED" and completion.get("tmux_returncode") != 0:
            raise OuterLaunchError("accepted tmux status has a nonzero return code")
        if status == "TMUX_REJECTED" and completion.get("tmux_returncode") == 0:
            raise OuterLaunchError("rejected tmux status has a zero return code")
        _verify_ref(launcher, completion.get("tmux_stdout"), "tmux_stdout.log")
        _verify_ref(launcher, completion.get("tmux_stderr"), "tmux_stderr.log")
    return {"status": status, "invocation": invocation, "completion": completion}


def _parse_lock(path: Path) -> dict[str, str]:
    _strict_file(path)
    text = path.read_text(encoding="ascii")
    if not text or not text.endswith("\n"):
        raise OuterLaunchError("pane consume lock is empty or incomplete")
    result = dict(line.split("=", 1) for line in text.splitlines() if "=" in line)
    if set(result) != {"nonce", "plan_sha256", "wrapper_sha256", "tmux_session"}:
        raise OuterLaunchError("pane consume lock schema is invalid")
    _strict_nonce(result["nonce"])
    _strict_sha(result["plan_sha256"], "pane lock plan_sha256")
    _strict_sha(result["wrapper_sha256"], "pane lock wrapper_sha256")
    _validate_exact_session(result["tmux_session"])
    return result


def _validate_inventory(root: Path, name: str, excluded: frozenset[str]) -> dict[str, Any]:
    value = _read_json_object(root / name)
    if value is None or value.get("schema_version") != 1 or value.get("excluded_from_inventory") != sorted(excluded):
        raise OuterLaunchError(f"{name} schema is invalid")
    rows = value.get("artifacts")
    if not isinstance(rows, list) or rows != sorted(rows, key=lambda row: row.get("relative_path", "") if isinstance(row, dict) else ""):
        raise OuterLaunchError(f"{name} rows are invalid")
    if value.get("canonical_inventory_sha256") != sha256_bytes(canonical_json_bytes(rows)):
        raise OuterLaunchError(f"{name} canonical SHA mismatch")
    actual = _inventory_value(root, excluded)
    if value != actual:
        raise OuterLaunchError(f"{name} does not bind current files")
    return value


def _validate_pane_error(
    pane: Path,
    value: Any,
    plan: dict[str, Any],
    plan_sha: str,
    wrapper_sha: str,
    pane_pid: int,
    start: dict[str, Any],
    timing: dict[str, Any],
    completion: dict[str, Any] | None = None,
) -> None:
    if not isinstance(value, dict) or set(value) != _PANE_ERROR_KEYS:
        raise OuterLaunchError("pane error schema is invalid")
    if value.get("schema_version") != 1 or value.get("status") != "PANE_FAILED" or value.get("failure_class") not in PANE_FAILURE_MESSAGES:
        raise OuterLaunchError("pane error identity is invalid")
    if type(value.get("message")) is not str or value["message"] != PANE_FAILURE_MESSAGES[value["failure_class"]] or not value["message"]:
        raise OuterLaunchError("pane error message is invalid")
    _strict_bool(value.get("inner_started"), "pane error inner_started")
    _strict_int(value.get("inner_returncode"), "pane error inner_returncode")
    if value.get("original_error_preserved") is not True:
        raise OuterLaunchError("pane error original_error_preserved is invalid")
    _strict_nonce(value.get("nonce"), "pane error nonce")
    _strict_int(value.get("pane_pid"), "pane error pane_pid", 1)
    _strict_sha(value.get("plan_sha256"), "pane error plan_sha256")
    _strict_sha(value.get("wrapper_sha256"), "pane error wrapper_sha256")
    created_at = _parse_utc(value.get("created_at_utc"), "pane error created_at_utc")
    plan_created_at = _parse_utc(plan["created_at_utc"], "pane plan created_at_utc")
    started_at = _parse_utc(start["started_at_utc"], "pane start started_at_utc")
    finished_at = _parse_utc(timing["finished_at_utc"], "pane timing finished_at_utc")
    if created_at < plan_created_at or created_at < started_at or created_at > finished_at:
        raise OuterLaunchError("pane error chronology is invalid")
    if value["nonce"] != plan["nonce"] or value["plan_sha256"] != plan_sha or value["wrapper_sha256"] != wrapper_sha or value["pane_pid"] != pane_pid:
        raise OuterLaunchError("pane error identity binding mismatch")
    expected_console = _file_ref_strict(pane, PANE_CONSOLE, required=True)
    if value.get("console") != expected_console:
        raise OuterLaunchError("pane error console binding mismatch")
    if completion is not None:
        if value["inner_started"] != completion["inner_started"] or value["inner_returncode"] != completion["inner_returncode"]:
            raise OuterLaunchError("pane error exit binding mismatch")
        expected_class = "PREINNER_BINDING_FAILURE" if not completion["inner_started"] else "INNER_NONZERO_EXIT" if completion["inner_returncode"] != 0 else "PANE_FINALIZATION_FAILURE"
        if value["failure_class"] != expected_class:
            raise OuterLaunchError("pane error failure class does not bind pane state")


def _validate_pane_timing(pane: Path, value: Any, start: dict[str, Any], receipt: dict[str, Any], completion: dict[str, Any]) -> None:
    required = {"schema_version", "started_at_utc", "finished_at_utc", "elapsed_seconds", "inner_started", "inner_returncode", "signal_name"}
    if not isinstance(value, dict) or set(value) != required or value.get("schema_version") != 1:
        raise OuterLaunchError("pane timing schema is invalid")
    started = _parse_utc(value.get("started_at_utc"), "pane timing started_at_utc")
    finished = _parse_utc(value.get("finished_at_utc"), "pane timing finished_at_utc")
    consumed = _parse_utc(receipt.get("consumed_at_utc"), "pane receipt consumed_at_utc")
    plan_created = _parse_utc(receipt.get("created_at_utc"), "pane receipt created_at_utc")
    if value["started_at_utc"] != start["started_at_utc"] or started != _parse_utc(start["started_at_utc"], "pane start started_at_utc"):
        raise OuterLaunchError("pane timing start does not bind pane start")
    if finished < started or finished < consumed or finished < _parse_utc(start["started_at_utc"], "pane start started_at_utc") or consumed < plan_created:
        raise OuterLaunchError("pane timing chronology is invalid")
    elapsed = value.get("elapsed_seconds")
    _strict_finite_number(elapsed, "pane timing elapsed_seconds", 0.0)
    expected_elapsed = (finished - started).total_seconds()
    if abs(float(elapsed) - expected_elapsed) > PANE_TIMING_TOLERANCE_SECONDS:
        raise OuterLaunchError("pane timing elapsed_seconds does not match UTC chronology")
    _strict_bool(value.get("inner_started"), "pane timing inner_started")
    _strict_int(value.get("inner_returncode"), "pane timing inner_returncode")
    if value["inner_started"] != completion["inner_started"] or value["inner_returncode"] != completion["inner_returncode"]:
        raise OuterLaunchError("pane timing inner state binding mismatch")
    if value.get("signal_name") is not None and (type(value["signal_name"]) is not str or value["signal_name"] not in _MONITORED_SIGNAL_NAMES):
        raise OuterLaunchError("pane timing signal_name is invalid")


def _validate_pane_secondary(
    pane: Path,
    value: Any,
    plan: dict[str, Any],
    plan_sha: str,
    wrapper_sha: str,
    pane_pid: int | None = None,
    finalizer_exit_code: int | None = None,
) -> None:
    if not isinstance(value, dict) or set(value) != _PANE_SECONDARY_KEYS or value.get("schema_version") != 1 or value.get("status") != "FAILED_FINALIZATION" or value.get("failure_class") != "PANE_FINALIZATION_FAILURE":
        raise OuterLaunchError("pane secondary finalization schema is invalid")
    _strict_int(value.get("pane_pid"), "pane secondary pane_pid", 1)
    _strict_nonce(value.get("nonce"), "pane secondary nonce")
    _strict_sha(value.get("plan_sha256"), "pane secondary plan_sha256")
    _strict_sha(value.get("wrapper_sha256"), "pane secondary wrapper_sha256")
    _strict_bool(value.get("inner_started"), "pane secondary inner_started")
    _strict_int(value.get("original_inner_returncode"), "pane secondary original_inner_returncode")
    _strict_int(value.get("finalizer_returncode"), "pane secondary finalizer_returncode", 0)
    if finalizer_exit_code is None or value["finalizer_returncode"] != finalizer_exit_code or finalizer_exit_code == 0:
        raise OuterLaunchError("pane secondary finalizer return code does not bind raw marker")
    _validate_executable_identity(value.get("finalizer_executable_identity"), "pane secondary finalizer identity")
    if value.get("original_error_preserved") is not True:
        raise OuterLaunchError("pane secondary original_error_preserved is invalid")
    _parse_utc(value.get("created_at_utc"), "pane secondary created_at_utc")
    if value["nonce"] != plan["nonce"] or value["plan_sha256"] != plan_sha or value["wrapper_sha256"] != wrapper_sha or pane_pid is not None and value["pane_pid"] != pane_pid:
        raise OuterLaunchError("pane secondary plan/wrapper binding mismatch")
    if value["finalizer_executable_identity"] != plan["pane_evidence_python_identity"]:
        raise OuterLaunchError("pane secondary finalizer identity mismatch")
    expected_console = _file_ref_strict(pane, PANE_CONSOLE, required=True)
    if value.get("console") != expected_console:
        raise OuterLaunchError("pane secondary console binding mismatch")


def _bound_pane_evidence_identity(pane: Path, invocation: dict[str, Any]) -> dict[str, Any]:
    expected = invocation["pane_evidence_python_identity"]
    try:
        actual = executable_identity(Path(invocation["pane_evidence_python"]))
    except OuterLaunchError:
        if (pane / PANE_SECONDARY_FAILURE).is_file():
            return expected
        raise
    if actual != expected:
        if (pane / PANE_SECONDARY_FAILURE).is_file():
            return expected
        raise OuterLaunchError("pane evidence executable identity has drifted")
    return actual


def _expected_pane_context(pane: Path, expected_outer: dict[str, Any] | None) -> tuple[dict[str, Any], bytes]:
    if expected_outer is None:
        invocation = _read_json_object(pane.parent / "launcher" / "outer_invocation.json")
        if invocation is None:
            raise OuterLaunchError("pane has no trusted outer invocation")
    else:
        invocation = expected_outer.get("invocation")
    if not isinstance(invocation, dict):
        raise OuterLaunchError("pane outer invocation is invalid")
    _validate_outer_invocation(invocation, pane.parent)
    root = Path(invocation["outer_evidence_dir"])
    if root != pane.parent:
        raise OuterLaunchError("pane outer root binding is invalid")
    config_root = Path(invocation["repo_root"])
    data_root = Path(invocation["data_root"])
    config, binding = _bind_config(config_root, Path(invocation["config_path"]))
    if config_root != Path(invocation["repo_root"]):
        raise OuterLaunchError("pane repo binding is invalid")
    expected_paths = {
        "output_dir": config_root / config["runtime"]["output_relative_path"],
        "process_evidence_dir": config_root / config["runtime"]["process_evidence_relative_path"],
        "outer_evidence_dir": root,
    }
    for field, expected in expected_paths.items():
        if Path(invocation[field]) != expected:
            raise OuterLaunchError(f"pane runtime path drift: {field}")
    child_identity = executable_identity(Path(invocation["child_python"]))
    pane_identity = _bound_pane_evidence_identity(pane, invocation)
    if invocation["child_python_identity"] != child_identity:
        raise OuterLaunchError("pane outer executable identity drift")
    expected_plan = _build_pane_plan(
        repo_root=config_root,
        data_root=data_root,
        output_dir=expected_paths["output_dir"],
        process_evidence_dir=expected_paths["process_evidence_dir"],
        outer_evidence_dir=root,
        config_binding=binding,
        nonce=invocation["nonce"],
        child_python_identity=child_identity,
        pane_evidence_python_identity=pane_identity,
        created_at_utc=invocation["created_at_utc"],
    )
    expected_nonce = sha256_bytes(canonical_json_bytes({"created_at_utc": invocation["created_at_utc"], "inner_argv_sha256": expected_plan["inner_argv_sha256"], "tmux_session": invocation["tmux_session"]}))[:32]
    if invocation["nonce"] != expected_nonce:
        raise OuterLaunchError("pane nonce is not deterministically bound to the plan inputs")
    plan_sha = sha256_bytes(canonical_json_bytes(expected_plan))
    environment = dict(invocation["environment"])
    environment.update({"P3_PANE_NONCE": invocation["nonce"], "P3_SMOKE_TMUX_SESSION": invocation["tmux_session"]})
    wrapper = _build_pane_wrapper_bytes(
        pane,
        pane / "pane_plan.json",
        plan_sha,
        Path(binding["config_path"]),
        binding["config_file_sha256"],
        expected_plan["inner_argv"],
        environment,
        pane_identity,
    )
    return expected_plan, wrapper


def validate_pane_evidence(pane_dir: Path, expected_outer: dict[str, Any] | None = None) -> dict[str, Any]:
    pane = Path(pane_dir)
    _validate_path_components(pane, final_kind="dir")
    plan_path = pane / "pane_plan.json"
    wrapper_path = pane / PANE_WRAPPER
    expected_plan, expected_wrapper = _expected_pane_context(pane, expected_outer)
    plan, plan_sha, wrapper_sha = _validate_pane_plan_identity(pane, expected_plan, expected_wrapper)
    lock = _parse_lock(pane / PANE_LOCK)
    if lock["nonce"] != plan["nonce"] or lock["plan_sha256"] != plan_sha or lock["wrapper_sha256"] != wrapper_sha or lock["tmux_session"] != plan["tmux_session"]:
        raise OuterLaunchError("pane consume lock binding mismatch")
    receipt = _read_json_object(pane / PANE_RECEIPT)
    required = {"schema_version", "state", "consumed", "single_use", "nonce", "pane_pid", "parent_pid", "plan_relative_path", "plan_size_bytes", "plan_sha256", "wrapper_relative_path", "wrapper_size_bytes", "wrapper_sha256", "consume_lock_relative_path", "consume_lock_size_bytes", "consume_lock_sha256", "config_relative_path", "config_absolute_path", "config_size_bytes", "config_file_sha256", "config_canonical_sha256", "smoke_id", "tmux_session", "repo_root", "runtime_data_root", "output_path", "process_path", "outer_path", "child_python", "child_python_identity", "pane_evidence_python", "pane_evidence_python_identity", "inner_argv_sha256", "created_at_utc", "consumed_at_utc"}
    if receipt is None or set(receipt) != required or receipt.get("schema_version") != 1 or receipt.get("state") != "CONSUMED" or receipt.get("consumed") is not True or receipt.get("single_use") is not True:
        raise OuterLaunchError("pane receipt schema is invalid")
    _strict_nonce(receipt.get("nonce"))
    for field in ("plan_relative_path", "wrapper_relative_path", "consume_lock_relative_path", "config_relative_path", "config_absolute_path", "smoke_id", "tmux_session", "repo_root", "runtime_data_root", "output_path", "process_path", "outer_path", "child_python", "pane_evidence_python", "created_at_utc", "consumed_at_utc"):
        if type(receipt.get(field)) is not str or not receipt[field]:
            raise OuterLaunchError(f"pane receipt string field is invalid: {field}")
    for field in ("pane_pid", "parent_pid", "plan_size_bytes", "wrapper_size_bytes", "consume_lock_size_bytes", "config_size_bytes"):
        _strict_int(receipt.get(field), f"pane receipt {field}", 1 if field.endswith("size_bytes") else 1)
    for field in ("plan_sha256", "wrapper_sha256", "consume_lock_sha256", "config_file_sha256", "config_canonical_sha256", "inner_argv_sha256"):
        _strict_sha(receipt.get(field), f"pane receipt {field}")
    refs = (("plan", plan_path, receipt["plan_relative_path"], receipt["plan_size_bytes"], receipt["plan_sha256"]), ("wrapper", wrapper_path, receipt["wrapper_relative_path"], receipt["wrapper_size_bytes"], receipt["wrapper_sha256"]), ("lock", pane / PANE_LOCK, receipt["consume_lock_relative_path"], receipt["consume_lock_size_bytes"], receipt["consume_lock_sha256"]))
    for label, path, rel, size, digest in refs:
        if rel != path.name or size != path.stat().st_size or digest != sha256_file(path):
            raise OuterLaunchError(f"pane receipt {label} binding mismatch")
    config = Path(receipt["config_absolute_path"])
    _strict_file(config)
    if receipt["config_relative_path"] != plan["config_binding"]["config_relative_path"] or receipt["config_size_bytes"] != config.stat().st_size or receipt["config_file_sha256"] != sha256_file(config):
        raise OuterLaunchError("pane receipt config binding mismatch")
    if receipt["config_canonical_sha256"] != plan["config_binding"]["config_canonical_sha256"]:
        raise OuterLaunchError("pane receipt config canonical binding mismatch")
    try:
        loaded_config = load_smoke_config(Path(plan["repo_root"]), config)
    except Exception as exc:
        raise OuterLaunchError("pane receipt config cannot be revalidated") from exc
    if sha256_bytes(canonical_json_bytes(loaded_config)) != receipt["config_canonical_sha256"]:
        raise OuterLaunchError("pane receipt config canonical SHA mismatch")
    expected = {"nonce": plan["nonce"], "smoke_id": plan["smoke_id"], "tmux_session": plan["tmux_session"], "repo_root": plan["repo_root"], "runtime_data_root": plan["data_root"], "output_path": plan["output_dir"], "process_path": plan["process_evidence_dir"], "outer_path": plan["outer_evidence_dir"], "child_python": plan["child_python"], "pane_evidence_python": plan["pane_evidence_python"], "inner_argv_sha256": plan["inner_argv_sha256"], "created_at_utc": plan["created_at_utc"]}
    for field, expected_value in expected.items():
        if receipt[field] != expected_value:
            raise OuterLaunchError(f"pane receipt {field} binding mismatch")
    _validate_executable_identity(receipt.get("child_python_identity"), "pane receipt child Python identity")
    _validate_executable_identity(receipt.get("pane_evidence_python_identity"), "pane receipt evidence Python identity")
    if receipt["child_python_identity"] != plan["child_python_identity"] or receipt["pane_evidence_python_identity"] != plan["pane_evidence_python_identity"]:
        raise OuterLaunchError("pane receipt executable identity binding mismatch")
    _strict_file(pane / PANE_START)
    _strict_file(pane / PANE_EXIT)
    try:
        pane_exit = (pane / PANE_EXIT).read_bytes()
        if not pane_exit.endswith(b"\n") or pane_exit.count(b"\n") != 1:
            raise OuterLaunchError("pane exit code bytes are invalid")
        actual_exit = int(pane_exit[:-1].decode("ascii"))
    except (UnicodeError, ValueError) as exc:
        raise OuterLaunchError("pane exit code bytes are invalid") from exc
    start = _read_json_object(pane / PANE_START)
    if start is None or set(start) != {"schema_version", "state", "nonce", "pane_pid", "plan_relative_path", "plan_sha256", "started_at_utc"} or start.get("schema_version") != 1 or start.get("state") != "STARTED" or start.get("nonce") != plan["nonce"] or start.get("plan_sha256") != plan_sha or start.get("plan_relative_path") != "pane_plan.json":
        raise OuterLaunchError("pane start binding is invalid")
    _strict_int(start.get("pane_pid"), "pane start pane_pid", 1)
    _parse_utc(start.get("started_at_utc"), "pane start started_at_utc")
    if start["pane_pid"] != receipt["pane_pid"]:
        raise OuterLaunchError("pane start PID binding mismatch")
    finalizer_exit_code = _validate_pane_finalizer_exit(pane)
    _validate_inventory(pane, PANE_INVENTORY, PANE_EXCLUDED)
    completion = _read_json_object(pane / PANE_COMPLETION)
    required_completion = {"schema_version", "status", "inner_started", "inner_returncode", "pane_pid", "inner_argv_sha256", "plan_sha256", "wrapper_sha256", "nonce", "tmux_session", "child_python_identity", "pane_evidence_python_identity", "pane_receipt", "consume_lock", "pane_inventory", "pane_timing", "error"}
    if completion is None or set(completion) != required_completion or completion.get("schema_version") != 1:
        raise OuterLaunchError("pane completion schema is invalid")
    if type(completion.get("status")) is not str or completion["status"] not in {"PANE_COMPLETED", "PANE_FAILED"}:
        raise OuterLaunchError("pane completion status is invalid")
    if completion.get("inner_argv_sha256") != plan["inner_argv_sha256"] or completion.get("plan_sha256") != plan_sha or completion.get("wrapper_sha256") != wrapper_sha or completion.get("nonce") != plan["nonce"] or completion.get("tmux_session") != plan["tmux_session"]:
        raise OuterLaunchError("pane completion binding mismatch")
    if type(completion.get("inner_started")) is not bool or type(completion.get("inner_returncode")) is not int or type(completion.get("pane_pid")) is not int:
        raise OuterLaunchError("pane completion status fields are invalid")
    if completion["pane_pid"] != receipt["pane_pid"] or completion["pane_pid"] != start["pane_pid"]:
        raise OuterLaunchError("pane completion PID binding mismatch")
    if completion["child_python_identity"] != plan["child_python_identity"] or completion["pane_evidence_python_identity"] != plan["pane_evidence_python_identity"]:
        raise OuterLaunchError("pane completion executable identity binding mismatch")
    if completion.get("inner_returncode") != actual_exit:
        raise OuterLaunchError("pane completion exit code mismatch")
    timing = _read_json_object(pane / PANE_TIMING)
    if timing is None:
        raise OuterLaunchError("pane timing is missing")
    _validate_pane_timing(pane, timing, start, receipt, completion)
    if expected_outer is not None:
        outer_completion = expected_outer.get("completion", {})
        if outer_completion.get("nonce") != plan["nonce"] or outer_completion.get("tmux_session") != plan["tmux_session"]:
            raise OuterLaunchError("pane and outer identity binding mismatch")
    _verify_ref(pane, completion.get("pane_receipt"), PANE_RECEIPT)
    _verify_ref(pane, completion.get("consume_lock"), PANE_LOCK)
    _verify_ref(pane, completion.get("pane_inventory"), PANE_INVENTORY)
    _verify_ref(pane, completion.get("pane_timing"), PANE_TIMING)
    error_ref = completion.get("error")
    if error_ref != _file_ref_strict(pane, PANE_ERROR):
        raise OuterLaunchError("pane completion error binding mismatch")
    error_present = error_ref.get("present") is True
    inner_started = completion["inner_started"]
    inner_returncode = completion["inner_returncode"]
    if completion["status"] == "PANE_COMPLETED":
        if finalizer_exit_code != 0 or inner_started is not True or inner_returncode != 0 or (pane / PANE_EXIT).read_bytes() != b"0\n" or error_present or (pane / PANE_SECONDARY_FAILURE).exists() or (pane / PANE_SECONDARY_FAILURE).is_symlink():
            raise OuterLaunchError("PANE_COMPLETED status semantics are invalid")
    else:
        if finalizer_exit_code != 0 or not error_present or (pane / PANE_SECONDARY_FAILURE).exists() or (pane / PANE_SECONDARY_FAILURE).is_symlink():
            raise OuterLaunchError("PANE_FAILED status has no failure evidence")
        error = _read_json_object(pane / PANE_ERROR)
        _validate_pane_error(pane, error, plan, plan_sha, wrapper_sha, receipt["pane_pid"], start, timing, completion)
    return {"plan": plan, "receipt": receipt, "completion": completion, "finalizer_returncode": finalizer_exit_code}


def validate_process_evidence(process_dir: Path, expected_identity: dict[str, Any] | None = None) -> dict[str, Any]:
    process = Path(process_dir)
    _validate_path_components(process, final_kind="dir")
    completion = _read_json_object(process / "process_completion.json")
    if completion is None or completion.get("schema_version") != 1 or completion.get("status") != "COMPLETED":
        raise OuterLaunchError("process completion is not terminal success")
    if type(completion.get("child_returncode")) is not int or type(completion.get("launcher_exit_code")) is not int or completion.get("child_returncode") != 0 or completion.get("launcher_exit_code") != 0:
        raise OuterLaunchError("process exit contract is not zero")
    exit_path = process / "process_exit_code.txt"
    _strict_file(exit_path)
    if exit_path.read_bytes() != b"0\n":
        raise OuterLaunchError("process exit code bytes are not exactly zero")
    for field in ("nonce", "smoke_id", "config_relative_path", "config_path", "config_size_bytes", "config_file_sha256", "runtime_data_root"):
        if field not in completion:
            raise OuterLaunchError(f"process completion field missing: {field}")
    _strict_nonce(completion.get("nonce"), "process completion nonce")
    _strict_int(completion.get("config_size_bytes"), "process completion config_size_bytes", 1)
    if expected_identity is not None and completion.get("nonce") != expected_identity.get("nonce"):
        raise OuterLaunchError("process nonce does not bind pane")
    prepared = _read_json_object(process / "handoff_prepared.json")
    receipt = _read_json_object(process / "handoff_receipt.json")
    invocation = _read_json_object(process / "process_invocation.json")
    if prepared is None or receipt is None or invocation is None:
        raise OuterLaunchError("process handoff evidence is incomplete")
    if prepared.get("nonce") != completion["nonce"] or receipt.get("nonce") != completion["nonce"] or invocation.get("nonce") != completion["nonce"]:
        raise OuterLaunchError("process handoff nonce binding mismatch")
    if receipt.get("consumed") is not True or prepared.get("consumed") is not False:
        raise OuterLaunchError("process handoff consumption state is invalid")
    for field in ("child_pid", "child_parent_pid"):
        _strict_int(completion.get(field), f"process completion {field}", 1)
    for field in ("child_pid", "child_ppid", "launcher_pid"):
        _strict_int(receipt.get(field), f"process receipt {field}", 1)
    if completion.get("child_pid") != receipt.get("child_pid") or completion.get("child_parent_pid") != receipt.get("child_ppid"):
        raise OuterLaunchError("process PID binding mismatch")
    if completion.get("handoff_receipt_sha256") != sha256_file(process / "handoff_receipt.json"):
        raise OuterLaunchError("process receipt SHA mismatch")
    process_inventory = _validate_inventory(process, "process_inventory.json", frozenset({"process_inventory.json", "process_completion.json", "process_partial_inventory.json"}))
    process_inventory_ref = completion.get("process_inventory")
    expected_process_inventory_ref = {"relative_path": "process_inventory.json", "size_bytes": (process / "process_inventory.json").stat().st_size, "sha256": sha256_file(process / "process_inventory.json")}
    if process_inventory_ref != expected_process_inventory_ref:
        raise OuterLaunchError("process inventory reference mismatch")
    process_exit_ref = completion.get("process_exit_code")
    expected_process_exit_ref = {"present": True, "relative_path": "process_exit_code.txt", "size_bytes": 2, "sha256": sha256_file(exit_path)}
    if process_exit_ref != expected_process_exit_ref:
        raise OuterLaunchError("process exit reference mismatch")
    for field in ("nonce", "runtime_data_root", "config_relative_path", "config_path", "config_size_bytes", "config_file_sha256"):
        if invocation.get(field) != completion.get(field):
            raise OuterLaunchError(f"process invocation binding mismatch: {field}")
    for field in ("runtime_data_root", "config_relative_path", "config_path", "config_size_bytes", "config_file_sha256"):
        if receipt.get(field) != completion.get(field):
            raise OuterLaunchError(f"process receipt binding mismatch: {field}")
    if prepared.get("runtime_data_root") != completion.get("runtime_data_root") or prepared.get("output_dir") != invocation.get("output_dir") or prepared.get("process_evidence_dir") != invocation.get("process_evidence_dir"):
        raise OuterLaunchError("prepared handoff path binding mismatch")
    entry = completion.get("entry")
    if not isinstance(entry, dict) or entry.get("entry_success_accepted") is not True:
        raise OuterLaunchError("process entry acceptance is not proven")
    return {"completion": completion, "invocation": invocation, "receipt": receipt}


def validate_entry_evidence(output_dir: Path, expected_identity: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        return validate_entry_output(Path(output_dir), expected_identity=expected_identity)
    except (OSError, SmokeEvidenceError, ValueError, StopIteration) as exc:
        raise OuterLaunchError(f"entry evidence is invalid: {type(exc).__name__}: {exc}") from exc


def classify_outer_evidence(outer_evidence_dir: Path, output_dir: Path, process_evidence_dir: Path) -> str:
    """Classify evidence without writing, deleting, or trusting status alone."""
    try:
        outer_value = validate_outer_evidence(outer_evidence_dir, output_dir, process_evidence_dir)
    except (OSError, OuterLaunchError, SmokeEvidenceError, ValueError):
        return "UNKNOWN"
    status = outer_value["status"]
    if status in {"PREFLIGHT_FAILED", "TMUX_REJECTED", "FINALIZATION_FAILED"}:
        return status
    if status == "INTERRUPTED_WITH_EVIDENCE":
        return status
    pane = Path(outer_evidence_dir) / "pane"
    if not (pane / PANE_START).is_file():
        return "TMUX_ACCEPTED_PANE_NOT_STARTED"
    if (pane / PANE_SECONDARY_FAILURE).is_file():
        try:
            expected_plan, expected_wrapper = _expected_pane_context(pane, outer_value)
            plan, plan_sha, wrapper_sha = _validate_pane_plan_identity(pane, expected_plan, expected_wrapper)
            secondary = _read_json_object(pane / PANE_SECONDARY_FAILURE)
            start = _read_json_object(pane / PANE_START)
            if start is None or type(start.get("pane_pid")) is not int:
                raise OuterLaunchError("pane secondary start binding is missing")
            finalizer_exit_code = _validate_pane_finalizer_exit(pane)
            _validate_pane_secondary(pane, secondary, plan, plan_sha, wrapper_sha, start["pane_pid"], finalizer_exit_code)
        except (OSError, OuterLaunchError, SmokeEvidenceError, ValueError):
            return "UNKNOWN"
        return "PANE_FINALIZATION_FAILED"
    if not (pane / PANE_EXIT).is_file() or not (pane / PANE_COMPLETION).is_file():
        return "PANE_STARTED_NOT_TERMINAL"
    try:
        pane_value = validate_pane_evidence(pane, outer_value)
    except (OSError, OuterLaunchError, SmokeEvidenceError, ValueError):
        return "PANE_STARTED_NOT_TERMINAL"
    if pane_value["completion"].get("inner_started") is not True:
        return "PANE_STARTED_NOT_TERMINAL"
    if pane_value["completion"].get("inner_returncode") != 0:
        return "TERMINAL_FAILED"
    if not Path(process_evidence_dir).is_dir():
        return "INNER_STARTED_PROCESS_EVIDENCE_ABSENT"
    try:
        process_value = validate_process_evidence(Path(process_evidence_dir), pane_value["receipt"])
        receipt = process_value["receipt"]
        expected = {"nonce": receipt.get("nonce"), "launcher_pid": receipt.get("launcher_pid"), "child_pid": receipt.get("child_pid"), "child_ppid": receipt.get("child_ppid"), "child_argv_sha256": receipt.get("child_argv_sha256"), "handoff_receipt_sha256": sha256_file(Path(process_evidence_dir) / "handoff_receipt.json"), "handoff_receipt_relative_path": "handoff_receipt.json"}
        validate_entry_evidence(Path(output_dir), expected)
    except (OSError, OuterLaunchError, SmokeEvidenceError, ValueError):
        return "TERMINAL_FAILED"
    return "TERMINAL_COMPLETE"


def contract_check(repo_root: Path, config_path: Path) -> dict[str, Any]:
    root = Path(repo_root)
    requested = Path(config_path)
    if not requested.is_absolute():
        requested = root / requested
    config, binding = _bind_config(root, requested)
    return {
        "status": "PASS",
        "mode": "outer-contract-check",
        "smoke_id": config["smoke_id"],
        "config_binding": binding,
        "torch_imported": False,
        "real_data_accessed": False,
        "dataset_or_dataloader_constructed": False,
        "model_constructed": False,
        "output_directory_created": False,
        "process_evidence_created": False,
        "outer_evidence_created": False,
        "tmux_called": False,
        "confirmatory_metrics_accessed": False,
        "dataset_test_accessed_by_this_process": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rtdetr-baseline-smoke-outer")
    sub = parser.add_subparsers(dest="mode", required=True)
    check = sub.add_parser("contract-check")
    check.add_argument("--repo-root", type=Path, required=True)
    check.add_argument("--config", type=Path, required=True)
    classify = sub.add_parser("classify")
    classify.add_argument("--outer-evidence-dir", type=Path, required=True)
    classify.add_argument("--output-dir", type=Path, required=True)
    classify.add_argument("--process-evidence-dir", type=Path, required=True)
    launch = sub.add_parser("launch")
    for name in ("repo-root", "data-root", "config", "output-dir", "process-evidence-dir", "outer-evidence-dir", "child-python", "tmux-executable", "tmux-session"):
        launch.add_argument("--" + name, required=True, type=str if name == "tmux-session" else Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.mode == "contract-check":
            print(canonical_json_bytes(contract_check(args.repo_root, args.config)).decode("utf-8"))
            return 0
        if args.mode == "classify":
            print(classify_outer_evidence(args.outer_evidence_dir, args.output_dir, args.process_evidence_dir))
            return 0
        result = run_outer_launch(
            repo_root=args.repo_root,
            data_root=args.data_root,
            config_path=args.config,
            output_dir=args.output_dir,
            process_evidence_dir=args.process_evidence_dir,
            outer_evidence_dir=args.outer_evidence_dir,
            child_python=args.child_python,
            tmux_executable=args.tmux_executable,
            tmux_session=args.tmux_session,
        )
        print(canonical_json_bytes(result).decode("utf-8"))
        return 0
    except Exception as error:
        print(f"rtdetr-baseline-smoke-outer: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
