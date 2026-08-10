"""Durable, single-call outer launcher for Baseline Smoke V2.

The outer launcher owns only outer and pane evidence. It does not import
torch, construct a dataset, or infer scientific completion from tmux status.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import stat
import subprocess
import sys
from datetime import datetime, timezone
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
PANE_EXCLUDED = frozenset({PANE_COMPLETION, PANE_INVENTORY, PANE_PARTIAL_INVENTORY, PANE_SECONDARY_FAILURE})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_NONCE_RE = re.compile(r"^[0-9a-f]{32}$")
_MONITORED_SIGNALS = tuple(value for value in (signal.SIGHUP, signal.SIGTERM, signal.SIGINT, signal.SIGQUIT) if value is not None)
_MONITORED_SIGNAL_NAMES = ["SIGHUP", "SIGTERM", "SIGINT", "SIGQUIT"]


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


_PANE_FINALIZER_CODE = r'''
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from sparse_rtdetr.baseline.smoke_evidence import atomic_write, canonical_json_bytes, file_ref, inventory, sha256_bytes, sha256_file

pane = Path(os.environ["P3_PANE_DIR"])
plan_path = pane / "pane_plan.json"
receipt_path = pane / "pane_receipt.json"
lock_path = pane / "pane_consume.lock"
wrapper_path = pane / "pane_wrapper.sh"
config_path = Path(os.environ["P3_PANE_CONFIG"])
inner_started = os.environ.get("P3_PANE_INNER_STARTED") == "true"
inner_rc = int(os.environ.get("P3_PANE_INNER_RC", "125"))
pane_pid = int(os.environ.get("P3_PANE_PID", "0"))
started_at = os.environ.get("P3_PANE_STARTED_AT", "unknown")
error_message = os.environ.get("P3_PANE_ERROR", "")

def ref(name):
    return file_ref(pane, name)

try:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan_bytes = plan_path.read_bytes()
    lock_text = lock_path.read_text(encoding="ascii")
    lock = dict(line.split("=", 1) for line in lock_text.splitlines() if "=" in line)
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
        "wrapper_sha256": sha256_file(wrapper_path),
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
        "inner_argv_sha256": plan["inner_argv_sha256"],
        "created_at_utc": plan["created_at_utc"],
        "consumed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if lock.get("nonce") != receipt["nonce"] or lock.get("plan_sha256") != receipt["plan_sha256"] or lock.get("wrapper_sha256") != receipt["wrapper_sha256"] or lock.get("tmux_session") != receipt["tmux_session"]:
        raise RuntimeError("pane consume lock binding mismatch")
    atomic_write(receipt_path, canonical_json_bytes(receipt))
    timing = {
        "schema_version": 1,
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "inner_started": inner_started,
        "inner_returncode": inner_rc,
        "signal_name": None,
    }
    atomic_write(pane / "pane_timing.json", canonical_json_bytes(timing))
    if error_message:
        atomic_write(pane / "pane_error.json", canonical_json_bytes({"schema_version": 1, "status": "FAILED", "message": error_message, "original_error_preserved": True}))
    pane_inventory = inventory(pane, frozenset({"pane_completion.json", "pane_inventory.json", "pane_partial_inventory.json", "pane_secondary_finalization_failure.json"}))
    inventory_bytes = canonical_json_bytes(pane_inventory)
    atomic_write(pane / "pane_inventory.json", inventory_bytes)
    status = "PANE_COMPLETED" if inner_started and inner_rc == 0 and not error_message else "PANE_FAILED"
    completion = {
        "schema_version": 1,
        "status": status,
        "inner_started": inner_started,
        "inner_returncode": inner_rc,
        "inner_argv_sha256": plan["inner_argv_sha256"],
        "plan_sha256": receipt["plan_sha256"],
        "nonce": receipt["nonce"],
        "tmux_session": receipt["tmux_session"],
        "pane_receipt": ref("pane_receipt.json"),
        "consume_lock": ref("pane_consume.lock"),
        "pane_inventory": ref("pane_inventory.json"),
        "pane_timing": ref("pane_timing.json"),
        "error": ref("pane_error.json"),
    }
    atomic_write(pane / "pane_completion.json", canonical_json_bytes(completion))
except BaseException as exc:
    try:
        atomic_write(pane / "pane_secondary_finalization_failure.json", canonical_json_bytes({"schema_version": 1, "status": "FAILED_FINALIZATION", "original_exception_type": type(exc).__name__, "original_exception_message": str(exc), "original_error_preserved": True}))
    except BaseException:
        pass
    raise
'''


def _wrapper(pane: Path, plan: Path, plan_sha: str, config: Path, config_sha: str, inner: list[str], environment: dict[str, str]) -> str:
    env_lines = "\n".join(f"export {key}={shlex.quote(value)}" for key, value in sorted(environment.items()))
    inner_cmd = shlex.join(inner)
    return f'''#!/bin/sh
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
INNER_STARTED=false
INNER_RC=125
ERROR_MESSAGE=''
STARTED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || printf '%s' unknown)
{env_lines}
export P3_PANE_DIR="$PANE"
export P3_PANE_CONFIG="$CONFIG"
export P3_PANE_PID="$$"
export P3_PANE_STARTED_AT="$STARTED_AT"
export P3_PANE_INNER_STARTED=false
export P3_PANE_INNER_RC=125
export P3_PANE_ERROR=

# noclobber is the POSIX exclusive-create operation used for single consume.
set -C
if ! (umask 022; printf 'nonce=%s\nplan_sha256=%s\nwrapper_sha256=%s\ntmux_session=%s\n' "$P3_PANE_NONCE" "$PLAN_SHA" "$(sha256sum "$0" 2>/dev/null | awk '{{print $1}}')" "$P3_SMOKE_TMUX_SESSION" > "$LOCK"); then
  exit 73
fi
set +C
: > "$CONSOLE"
printf '{{"schema_version":1,"state":"STARTED","nonce":"%s","pane_pid":%s,"plan_relative_path":"pane_plan.json","plan_sha256":"%s"}}\n' "$P3_PANE_NONCE" "$$" "$PLAN_SHA" > "$START"
printf '%s\n' 'P3_SMOKE_PANE_SHELL_STARTED' >> "$CONSOLE"
if test ! -f "$PLAN" || test "$(sha256sum "$PLAN" 2>/dev/null | awk '{{print $1}}')" != "$PLAN_SHA"; then
  ERROR_MESSAGE='pane plan binding failed'
elif test ! -f "$CONFIG" || test "$(sha256sum "$CONFIG" 2>/dev/null | awk '{{print $1}}')" != "$CONFIG_SHA"; then
  ERROR_MESSAGE='config binding failed before inner launcher'
fi
if test -z "$ERROR_MESSAGE"; then
  INNER_STARTED=true
  export P3_PANE_INNER_STARTED=true
  set +e
  {inner_cmd} >> "$CONSOLE" 2>&1
  INNER_RC=$?
  set -u
  export P3_PANE_INNER_RC="$INNER_RC"
else
  export P3_PANE_ERROR="$ERROR_MESSAGE"
fi
printf '%s\n' "$INNER_RC" > "$EXIT"
set +e
{shlex.quote(environment["PANE_EVIDENCE_PYTHON"])} -c {shlex.quote(_PANE_FINALIZER_CODE)} >> "$CONSOLE" 2>&1
FINALIZER_RC=$?
set -u
if test "$FINALIZER_RC" -ne 0; then
  exit "$FINALIZER_RC"
fi
exit "$INNER_RC"
'''


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
    created_at = _utc_now()
    invocation = {
        "schema_version": 1,
        "mode": "outer_launch",
        "status": "OUTER_PREPARED",
        "nonportable": True,
        "created_at_utc": created_at,
        "smoke_id": SMOKE_V2_ID,
        "repo_root": str(repo_root),
        "data_root": str(data_root),
        "config_path": str(config_path),
        "output_dir": str(output),
        "process_evidence_dir": str(process),
        "outer_evidence_dir": str(outer),
        "child_python": str(child),
        "tmux_executable": str(tmux),
        "tmux_session": tmux_session,
        "environment": {
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "P3_RTDETR_BASELINE_SMOKE_AUTHORIZED": "1",
            "PYTHONPATH": str(repo_root / "src"),
            "P3_SMOKE_TMUX_SESSION": SMOKE_V2_TMUX_SESSION,
            "PANE_EVIDENCE_PYTHON": str(Path(sys.executable).resolve()),
        },
    }
    _write_json(launcher / "outer_invocation.json", invocation)
    runner: dict[str, Any] = {"stdout": b"", "stderr": b"", "returncode": None, "signal_events": [], "signal_handlers_installed": False, "monitored_signals": list(_MONITORED_SIGNAL_NAMES), "observed_signal_count": 0, "timeout_triggered": False, "term_sent": False, "kill_sent": False, "original_exception": None}
    try:
        _validate_exact_session(tmux_session)
        _validate_path_components(repo_root, final_kind="dir")
        config_arg = Path(config_path)
        if not config_arg.is_absolute():
            config_arg = repo_root / config_arg
        config, binding = _bind_config(repo_root, config_arg)
        _validate_runtime_paths(repo_root, config, data_root, output, process, outer)
        _strict_file(child, executable=True)
        _strict_file(tmux, executable=True)
        for path, field in ((repo_root, "repo_root"), (data_root, "data_root"), (output, "output_dir"), (process, "process_evidence_dir"), (outer, "outer_evidence_dir"), (child, "child_python"), (tmux, "tmux_executable")):
            if any(ord(char) > 127 for char in str(path)):
                raise OuterLaunchError(f"{field} must be ASCII")
        _write_json(launcher / "outer_config_binding.json", binding)
        inner = _inner_argv(repo_root, data_root, output, process, Path(binding["config_path"]), child)
        inner_sha = argv_sha256(inner)
        nonce = sha256_bytes(canonical_json_bytes({"created_at_utc": created_at, "inner_argv_sha256": inner_sha, "tmux_session": tmux_session}))[:32]
        plan = {
            "schema_version": 1,
            "state": "PREPARED",
            "single_use": True,
            "nonce": nonce,
            "smoke_id": SMOKE_V2_ID,
            "repo_root": str(repo_root),
            "data_root": str(data_root),
            "output_dir": str(output),
            "process_evidence_dir": str(process),
            "outer_evidence_dir": str(outer),
            "config_binding": binding,
            "child_python": str(child),
            "inner_argv": inner,
            "inner_argv_sha256": inner_sha,
            "tmux_session": tmux_session,
            "created_at_utc": created_at,
            "nonportable": True,
        }
        plan_sha = _write_json(pane / "pane_plan.json", plan)
        pane_environment = dict(invocation["environment"])
        pane_environment.update({"P3_PANE_NONCE": nonce, "P3_SMOKE_TMUX_SESSION": tmux_session})
        wrapper_path = pane / PANE_WRAPPER
        wrapper_bytes = _wrapper(pane, pane / "pane_plan.json", plan_sha, Path(binding["config_path"]), binding["config_file_sha256"], inner, pane_environment).encode("ascii")
        atomic_write(wrapper_path, wrapper_bytes)
        wrapper_path.chmod(0o755)
        tmux_argv = [str(tmux), "new-session", "-d", "-s", tmux_session, "-c", str(repo_root), str(wrapper_path)]
        tmux_identity = {"path": str(tmux), "size_bytes": tmux.stat().st_size, "sha256": sha256_file(tmux), "mode": stat.S_IMODE(tmux.stat().st_mode)}
        _write_json(launcher / "preflight_audit.json", {"schema_version": 1, "status": "PASS", "config_binding": binding, "data_root": str(data_root), "output_process_not_created_by_outer": True, "tmux_executable_identity": tmux_identity, "wrapper_relative_path": "../pane/pane_wrapper.sh", "wrapper_sha256": sha256_bytes(wrapper_bytes), "pane_plan_relative_path": "../pane/pane_plan.json", "pane_plan_sha256": plan_sha, "inner_argv_sha256": inner_sha, "nonce": nonce, "tmux_session": tmux_session, "tmux_client_timeout_seconds": config["runtime"]["tmux_client_timeout_seconds"], "created_at_utc": created_at})
        _write_json(launcher / "tmux_invocation.json", {"schema_version": 1, "argv": tmux_argv, "argv_sha256": argv_sha256(tmux_argv), "wrapper_sha256": sha256_bytes(wrapper_bytes), "pane_plan_sha256": plan_sha, "nonce": nonce, "tmux_session": tmux_session, "timeout_seconds": config["runtime"]["tmux_client_timeout_seconds"], "called_at_utc": _utc_now(), "tmux_executable_identity": tmux_identity})
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
    _verify_ref(launcher, completion.get("invocation"), "outer_invocation.json")
    status = completion.get("status")
    if status not in {"PREFLIGHT_FAILED", "TMUX_REJECTED", "TMUX_ACCEPTED", "INTERRUPTED_WITH_EVIDENCE", "FINALIZATION_FAILED"}:
        raise OuterLaunchError("outer status is invalid")
    runner = completion.get("runner")
    if not isinstance(runner, dict):
        raise OuterLaunchError("outer runner evidence is missing")
    handlers_ok = runner.get("signal_handlers_installed") is True or (status == "PREFLIGHT_FAILED" and runner.get("signal_handlers_installed") is False)
    if not isinstance(runner.get("signal_events"), list) or not handlers_ok or runner.get("monitored_signals") != _MONITORED_SIGNAL_NAMES or type(runner.get("observed_signal_count")) is not int or runner["observed_signal_count"] != len(runner.get("signal_events", [])):
        raise OuterLaunchError("outer runner evidence is incomplete")
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
        for value, field in ((preflight.get("nonce"), "preflight nonce"), (tmux_invocation.get("nonce"), "tmux invocation nonce")):
            _strict_nonce(value, field)
            if value != completion["nonce"]:
                raise OuterLaunchError(f"{field} mismatch")
        if preflight.get("tmux_session") != invocation["tmux_session"] or tmux_invocation.get("tmux_session") != invocation["tmux_session"]:
            raise OuterLaunchError("outer launch session binding mismatch")
        _validate_timeout(preflight.get("tmux_client_timeout_seconds"))
        _validate_timeout(tmux_invocation.get("timeout_seconds"))
        if tmux_invocation.get("argv_sha256") != argv_sha256(tmux_invocation.get("argv")):
            raise OuterLaunchError("tmux argv SHA mismatch")
        if preflight.get("pane_plan_sha256") != tmux_invocation.get("pane_plan_sha256") or preflight.get("wrapper_sha256") != tmux_invocation.get("wrapper_sha256"):
            raise OuterLaunchError("outer plan/wrapper binding mismatch")
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


def validate_pane_evidence(pane_dir: Path, expected_outer: dict[str, Any] | None = None) -> dict[str, Any]:
    pane = Path(pane_dir)
    _validate_path_components(pane, final_kind="dir")
    plan_path = pane / "pane_plan.json"
    wrapper_path = pane / PANE_WRAPPER
    plan = _read_json_object(plan_path)
    plan_keys = {"schema_version", "state", "single_use", "nonce", "smoke_id", "repo_root", "data_root", "output_dir", "process_evidence_dir", "outer_evidence_dir", "config_binding", "child_python", "inner_argv", "inner_argv_sha256", "tmux_session", "created_at_utc", "nonportable"}
    if plan is None or set(plan) != plan_keys or plan.get("schema_version") != 1 or plan.get("state") != "PREPARED" or plan.get("single_use") is not True:
        raise OuterLaunchError("pane plan schema is invalid")
    _strict_nonce(plan.get("nonce"))
    _validate_exact_session(plan.get("tmux_session"))
    if plan.get("smoke_id") != SMOKE_V2_ID or plan.get("nonportable") is not True or plan.get("inner_argv_sha256") != argv_sha256(plan.get("inner_argv")):
        raise OuterLaunchError("pane plan identity or argv binding is invalid")
    config_binding = plan.get("config_binding")
    if not isinstance(config_binding, dict) or config_binding.get("config_relative_path") != SMOKE_V2_CONFIG_RELATIVE:
        raise OuterLaunchError("pane plan config binding is invalid")
    _strict_sha(config_binding.get("config_file_sha256"), "pane plan config_file_sha256")
    _strict_sha(config_binding.get("config_canonical_sha256"), "pane plan config_canonical_sha256")
    _strict_file(wrapper_path, executable=True)
    plan_sha = sha256_file(plan_path)
    wrapper_sha = sha256_file(wrapper_path)
    lock = _parse_lock(pane / PANE_LOCK)
    if lock["nonce"] != plan["nonce"] or lock["plan_sha256"] != plan_sha or lock["wrapper_sha256"] != wrapper_sha or lock["tmux_session"] != plan["tmux_session"]:
        raise OuterLaunchError("pane consume lock binding mismatch")
    receipt = _read_json_object(pane / PANE_RECEIPT)
    required = {"schema_version", "state", "consumed", "single_use", "nonce", "pane_pid", "parent_pid", "plan_relative_path", "plan_size_bytes", "plan_sha256", "wrapper_relative_path", "wrapper_size_bytes", "wrapper_sha256", "consume_lock_relative_path", "consume_lock_size_bytes", "consume_lock_sha256", "config_relative_path", "config_absolute_path", "config_size_bytes", "config_file_sha256", "config_canonical_sha256", "smoke_id", "tmux_session", "repo_root", "runtime_data_root", "output_path", "process_path", "outer_path", "child_python", "inner_argv_sha256", "created_at_utc", "consumed_at_utc"}
    if receipt is None or set(receipt) != required or receipt.get("schema_version") != 1 or receipt.get("state") != "CONSUMED" or receipt.get("consumed") is not True or receipt.get("single_use") is not True:
        raise OuterLaunchError("pane receipt schema is invalid")
    _strict_nonce(receipt.get("nonce"))
    for field in ("plan_relative_path", "wrapper_relative_path", "consume_lock_relative_path", "config_relative_path", "config_absolute_path", "smoke_id", "tmux_session", "repo_root", "runtime_data_root", "output_path", "process_path", "outer_path", "child_python", "created_at_utc", "consumed_at_utc"):
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
    expected = {"nonce": plan["nonce"], "smoke_id": plan["smoke_id"], "tmux_session": plan["tmux_session"], "repo_root": plan["repo_root"], "runtime_data_root": plan["data_root"], "output_path": plan["output_dir"], "process_path": plan["process_evidence_dir"], "outer_path": plan["outer_evidence_dir"], "child_python": plan["child_python"], "inner_argv_sha256": plan["inner_argv_sha256"]}
    for field, expected_value in expected.items():
        if receipt[field] != expected_value:
            raise OuterLaunchError(f"pane receipt {field} binding mismatch")
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
    if start is None or start.get("schema_version") != 1 or start.get("state") != "STARTED" or start.get("nonce") != plan["nonce"] or start.get("plan_sha256") != plan_sha:
        raise OuterLaunchError("pane start binding is invalid")
    _validate_inventory(pane, PANE_INVENTORY, PANE_EXCLUDED)
    completion = _read_json_object(pane / PANE_COMPLETION)
    required_completion = {"schema_version", "status", "inner_started", "inner_returncode", "inner_argv_sha256", "plan_sha256", "nonce", "tmux_session", "pane_receipt", "consume_lock", "pane_inventory", "pane_timing", "error"}
    if completion is None or set(completion) != required_completion or completion.get("schema_version") != 1:
        raise OuterLaunchError("pane completion schema is invalid")
    if completion.get("inner_argv_sha256") != plan["inner_argv_sha256"] or completion.get("plan_sha256") != plan_sha or completion.get("nonce") != plan["nonce"] or completion.get("tmux_session") != plan["tmux_session"]:
        raise OuterLaunchError("pane completion binding mismatch")
    if type(completion.get("inner_started")) is not bool or type(completion.get("inner_returncode")) is not int:
        raise OuterLaunchError("pane completion status fields are invalid")
    if completion.get("inner_returncode") != actual_exit:
        raise OuterLaunchError("pane completion exit code mismatch")
    timing = _read_json_object(pane / PANE_TIMING)
    if timing is None or timing.get("schema_version") != 1 or timing.get("inner_started") != completion["inner_started"] or timing.get("inner_returncode") != completion["inner_returncode"]:
        raise OuterLaunchError("pane timing binding is invalid")
    if expected_outer is not None:
        outer_completion = expected_outer.get("completion", {})
        if outer_completion.get("nonce") != plan["nonce"] or outer_completion.get("tmux_session") != plan["tmux_session"]:
            raise OuterLaunchError("pane and outer identity binding mismatch")
    _verify_ref(pane, completion.get("pane_receipt"), PANE_RECEIPT)
    _verify_ref(pane, completion.get("consume_lock"), PANE_LOCK)
    _verify_ref(pane, completion.get("pane_inventory"), PANE_INVENTORY)
    _verify_ref(pane, completion.get("pane_timing"), PANE_TIMING)
    if completion.get("error") != _file_ref_strict(pane, PANE_ERROR):
        raise OuterLaunchError("pane completion error binding mismatch")
    return {"plan": plan, "receipt": receipt, "completion": completion}


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
