"""Authorized Baseline Smoke V1 CLI and process-evidence launcher."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import signal
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .smoke import (
    SMOKE_AUTH_ENV,
    SMOKE_CONFIG_RELATIVE,
    SMOKE_NONCE_ENV,
    SmokeContractError,
    contract_check,
    get_smoke_runtime_spec,
    load_smoke_config,
    run_authorized_smoke,
    validate_real_smoke_environment,
)
from .smoke_evidence import (
    SmokeEvidenceError,
    atomic_write,
    canonical_json_bytes,
    file_ref,
    inventory,
    _read_object,
    sha256_bytes,
    sha256_file,
    validate_entry_output,
)


class SmokeLauncherError(SmokeContractError):
    """Raised when process evidence ownership or launch identity fails."""


PROCESS_INVENTORY_EXCLUDED = frozenset({"process_inventory.json", "process_completion.json", "process_partial_inventory.json"})
HANDOFF_PREPARED_NAME = "handoff_prepared.json"
HANDOFF_RECEIPT_NAME = "handoff_receipt.json"
SIGNAL_GRACE_SECONDS = 2.0
_SIGNALS = tuple(value for value in (signal.SIGHUP, signal.SIGTERM, signal.SIGINT, signal.SIGQUIT) if value is not None)
_CHILD_MODULE = "sparse_rtdetr.baseline.smoke_launcher"
_CHILD_SUBCOMMAND = "_child"
_CHILD_REPO_FLAG = "--repo-root"
_PROC_CMDLINE_PATH = Path("/proc/self/cmdline")
_PROC_EXE_PATH = Path("/proc/self/exe")
_MAX_PROC_CMDLINE_BYTES = 64 * 1024
_MAX_PROC_CMDLINE_TOKENS = 64
_EXECUTABLE_IDENTITY_KEYS = frozenset({"canonical_path", "size_bytes", "sha256", "mode", "regular_file", "executable"})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _signal_name(signum: int) -> str:
    try:
        return signal.Signals(signum).name
    except ValueError:
        return f"SIG{signum}"


def _validate_launch_paths(output_dir: Path, process_evidence_dir: Path, child_python: Path) -> None:
    if not output_dir.is_absolute() or not process_evidence_dir.is_absolute() or not child_python.is_absolute():
        raise SmokeLauncherError("smoke launcher paths must be absolute")
    if any(part == ".." for part in output_dir.parts + process_evidence_dir.parts):
        raise SmokeLauncherError("smoke launcher paths may not contain ..")
    if output_dir.exists() or output_dir.is_symlink():
        raise SmokeLauncherError("child output directory already exists")
    if process_evidence_dir.exists() or process_evidence_dir.is_symlink():
        raise SmokeLauncherError("process evidence directory already exists")
    if child_python.is_symlink() or not child_python.is_file() or not os.access(child_python, os.X_OK):
        raise SmokeLauncherError("child Python must be an executable regular file")


def _config_binding(repo_root: Path, config_path: Path | None) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    root = repo_root.resolve()
    requested = root / SMOKE_CONFIG_RELATIVE if config_path is None else Path(config_path)
    if not requested.is_absolute():
        requested = root / requested
    if requested.is_symlink() or not requested.is_file() or requested.stat().st_nlink != 1:
        raise SmokeLauncherError("smoke config must be a regular non-hardlinked file")
    resolved = requested.resolve()
    config = load_smoke_config(root, resolved)
    relative = resolved.relative_to(root).as_posix()
    expected_relative = get_smoke_runtime_spec(config["smoke_id"]).config_relative_path
    if relative != expected_relative:
        raise SmokeLauncherError("smoke config relative path is not bound to its identity")
    binding = {
        "config_relative_path": relative,
        "config_path": str(resolved),
        "config_size_bytes": resolved.stat().st_size,
        "config_file_sha256": sha256_file(resolved),
        "config_canonical_sha256": sha256_bytes(canonical_json_bytes(config)),
        "smoke_id": config["smoke_id"],
    }
    return config, resolved, binding


def _canonical_runtime_data_root(data_root: Path) -> Path:
    if not isinstance(data_root, Path) or not data_root.is_absolute():
        raise SmokeLauncherError("smoke runtime data root must be absolute")
    if data_root.is_symlink():
        raise SmokeLauncherError("smoke runtime data root may not be a symlink")
    try:
        canonical = data_root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise SmokeLauncherError("smoke runtime data root is missing or invalid") from exc
    if canonical.is_symlink() or not canonical.is_dir():
        raise SmokeLauncherError("smoke runtime data root must be a regular directory")
    if any(part.casefold() == "test" for part in canonical.parts):
        raise SmokeLauncherError("smoke runtime data root may not contain an independent test component")
    return canonical


def _validate_frozen_runtime_paths(
    repo_root: Path,
    output_dir: Path,
    process_evidence_dir: Path,
    *,
    config: dict[str, Any] | None = None,
    config_path: Path | None = None,
) -> None:
    """Validate paths against the already-bound versioned config identity."""

    root = repo_root.resolve()
    if config is None:
        config = load_smoke_config(root, config_path)
    try:
        spec = get_smoke_runtime_spec(config["smoke_id"])
    except (KeyError, SmokeContractError) as exc:
        raise SmokeLauncherError("smoke runtime identity is unknown") from exc
    if config_path is None:
        config_path = root / spec.config_relative_path
    resolved_config = Path(config_path).resolve()
    if resolved_config != root / spec.config_relative_path:
        raise SmokeLauncherError("smoke config path drift from the frozen identity")
    expected_output = root / spec.output_relative_path
    expected_process = root / spec.process_relative_path
    if output_dir.resolve() != expected_output or process_evidence_dir.resolve() != expected_process:
        raise SmokeLauncherError("smoke output/evidence paths drift from the frozen contract")


def _atomic_json(path: Path, value: Any) -> str:
    payload = canonical_json_bytes(value)
    atomic_write(path, payload)
    return sha256_bytes(payload)


def _validate_nonce(value: Any) -> None:
    if type(value) is not str or len(value) != 32 or any(character not in "0123456789abcdef" for character in value):
        raise SmokeLauncherError("smoke nonce must be exactly 32 lowercase hexadecimal characters")


def _validate_sha(value: Any, field: str) -> None:
    if type(value) is not str or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise SmokeLauncherError(f"{field} must be a lowercase SHA-256 string")


def _canonical_repo_root(repo_root: Path) -> Path:
    if not isinstance(repo_root, Path) or not repo_root.is_absolute() or any(part == ".." for part in repo_root.parts):
        raise SmokeLauncherError("child repo root must be an absolute canonical path")
    try:
        canonical = repo_root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise SmokeLauncherError("child repo root cannot be resolved") from exc
    if canonical != repo_root or not canonical.is_dir() or canonical.is_symlink():
        raise SmokeLauncherError("child repo root is not a canonical regular directory")
    return canonical


def _executable_identity(path: Path) -> dict[str, Any]:
    if not isinstance(path, Path) or not path.is_absolute() or any(part == ".." for part in path.parts):
        raise SmokeLauncherError("child executable path is not absolute and canonical")
    if path.is_symlink():
        raise SmokeLauncherError("child executable may not be a symlink")
    try:
        canonical = path.resolve(strict=True)
        info = canonical.lstat()
    except (OSError, RuntimeError) as exc:
        raise SmokeLauncherError("child executable cannot be resolved") from exc
    if canonical != path or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or not os.access(canonical, os.X_OK):
        raise SmokeLauncherError("child executable must be a regular executable file")
    return {
        "canonical_path": str(canonical),
        "size_bytes": info.st_size,
        "sha256": sha256_file(canonical),
        "mode": stat.S_IMODE(info.st_mode),
        "regular_file": True,
        "executable": True,
    }


def _validate_executable_identity(value: Any, field: str = "child executable identity") -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _EXECUTABLE_IDENTITY_KEYS:
        raise SmokeLauncherError(f"{field} schema is invalid")
    path = value.get("canonical_path")
    if type(path) is not str or not path or not Path(path).is_absolute():
        raise SmokeLauncherError(f"{field} path is invalid")
    if type(value.get("size_bytes")) is not int or value["size_bytes"] < 1:
        raise SmokeLauncherError(f"{field} size is invalid")
    _validate_sha(value.get("sha256"), f"{field} sha256")
    if type(value.get("mode")) is not int or value["mode"] < 0 or value["mode"] > 0o7777:
        raise SmokeLauncherError(f"{field} mode is invalid")
    if value.get("regular_file") is not True or value.get("executable") is not True:
        raise SmokeLauncherError(f"{field} type flags are invalid")
    actual = _executable_identity(Path(path))
    if actual != value:
        raise SmokeLauncherError(f"{field} does not match the executable")
    return value


def _canonical_child_argv(child_python: Path, repo_root: Path) -> list[str]:
    executable = _executable_identity(child_python)
    root = _canonical_repo_root(repo_root)
    return [executable["canonical_path"], "-m", _CHILD_MODULE, _CHILD_SUBCOMMAND, _CHILD_REPO_FLAG, str(root)]


def _child_argv_sha256(argv: list[str]) -> str:
    if not isinstance(argv, list) or not argv or any(type(token) is not str or not token or "\x00" in token for token in argv):
        raise SmokeLauncherError("child argv must be a non-empty list of non-empty strings")
    return hashlib.sha256(canonical_json_bytes(argv)).hexdigest()


def _read_proc_cmdline() -> list[str]:
    if sys.platform != "linux":
        raise SmokeLauncherError("Linux /proc child argv evidence is required")
    try:
        info = _PROC_CMDLINE_PATH.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise SmokeLauncherError("/proc/self/cmdline is not a regular file")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(str(_PROC_CMDLINE_PATH), flags)
    except SmokeLauncherError:
        raise
    except OSError as exc:
        raise SmokeLauncherError("cannot open /proc/self/cmdline") from exc
    try:
        raw = bytearray()
        while True:
            chunk = os.read(fd, min(4096, _MAX_PROC_CMDLINE_BYTES + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
            if len(raw) > _MAX_PROC_CMDLINE_BYTES:
                raise SmokeLauncherError("/proc/self/cmdline is too long")
    finally:
        os.close(fd)
    if not raw or raw[-1] != 0:
        raise SmokeLauncherError("/proc/self/cmdline must end with one NUL")
    content = bytes(raw[:-1])
    if not content:
        raise SmokeLauncherError("/proc/self/cmdline contains no tokens")
    encoded_tokens = content.split(b"\x00")
    if len(encoded_tokens) > _MAX_PROC_CMDLINE_TOKENS or any(not token for token in encoded_tokens):
        raise SmokeLauncherError("/proc/self/cmdline token schema is invalid")
    try:
        tokens = [token.decode("utf-8", errors="strict") for token in encoded_tokens]
    except UnicodeDecodeError as exc:
        raise SmokeLauncherError("/proc/self/cmdline is not valid UTF-8") from exc
    if any(not token or "\x00" in token for token in tokens):
        raise SmokeLauncherError("child argv contains an empty token")
    return tokens


def _actual_child_executable_identity() -> dict[str, Any]:
    if sys.platform != "linux":
        raise SmokeLauncherError("Linux /proc executable evidence is required")
    try:
        proc_executable = Path(os.readlink(_PROC_EXE_PATH)).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise SmokeLauncherError("cannot resolve /proc/self/exe") from exc
    return _executable_identity(proc_executable)


def _actual_child_argv(*, require_module: bool = True) -> list[str]:
    actual = _read_proc_cmdline()
    original = getattr(sys, "orig_argv", None)
    if not isinstance(original, list) or actual != original:
        raise SmokeLauncherError("sys.orig_argv does not match /proc/self/cmdline")
    if not isinstance(sys.executable, str) or not sys.executable or actual[0] != sys.executable:
        raise SmokeLauncherError("sys.executable does not match process argv0")
    if not isinstance(sys.argv, list) or not sys.argv:
        raise SmokeLauncherError("sys.argv is invalid")
    if require_module:
        if len(actual) < 4 or actual[1] != "-m" or actual[2] != _CHILD_MODULE or actual[3:] != sys.argv[1:]:
            raise SmokeLauncherError("sys.argv and module process argv are inconsistent")
    return actual


def _consume_handoff(repo_root: Path) -> tuple[dict[str, Any], str]:
    """Consume the launcher-owned handoff before output or model-side work."""

    if os.environ.get(SMOKE_AUTH_ENV) != "1" or os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or os.environ.get("PYTHONNOUSERSITE") != "1":
        raise SmokeLauncherError("handoff consumer environment is not exact")
    handoff_path_value = os.environ.get("P3_SMOKE_HANDOFF_PATH")
    output_value = os.environ.get("P3_SMOKE_OUTPUT_DIR")
    process_value = os.environ.get("P3_SMOKE_PROCESS_EVIDENCE_DIR")
    data_root_value = os.environ.get("P3_SMOKE_DATA_ROOT")
    nonce_value = os.environ.get(SMOKE_NONCE_ENV)
    if not all(isinstance(value, str) and value for value in (handoff_path_value, output_value, process_value, data_root_value, nonce_value)):
        raise SmokeLauncherError("handoff consumer environment is incomplete")
    _validate_nonce(nonce_value)
    runtime_data_root = _canonical_runtime_data_root(Path(data_root_value))
    handoff_path = Path(handoff_path_value)
    output_dir = Path(output_value)
    process_dir = Path(process_value)
    if not handoff_path.is_absolute() or not output_dir.is_absolute() or not process_dir.is_absolute():
        raise SmokeLauncherError("handoff paths must be absolute")
    if handoff_path != process_dir / HANDOFF_PREPARED_NAME or not process_dir.is_dir() or process_dir.is_symlink():
        raise SmokeLauncherError("handoff path ownership is invalid")
    if handoff_path.is_symlink() or not handoff_path.is_file() or handoff_path.stat().st_nlink != 1:
        raise SmokeLauncherError("prepared handoff is not a regular file")
    receipt_path = process_dir / HANDOFF_RECEIPT_NAME
    if receipt_path.exists() or receipt_path.is_symlink():
        raise SmokeLauncherError("prepared handoff has already been consumed")
    prepared = _read_object(handoff_path)
    required = {
        "schema_version",
        "state",
        "single_use",
        "consumed",
        "nonce",
        "launcher_pid",
        "repo_root",
        "output_dir",
        "process_evidence_dir",
        "runtime_data_root",
        "smoke_id",
        "config_relative_path",
        "config_path",
        "config_size_bytes",
        "config_file_sha256",
        "config_canonical_sha256",
        "child_argv",
        "child_argv_sha256",
        "child_executable_identity",
        "child_argv_schema",
        "created_at_utc",
    }
    if set(prepared) != required or prepared.get("schema_version") != 1 or prepared.get("state") != "PREPARED":
        raise SmokeLauncherError("prepared handoff schema is invalid")
    if prepared.get("single_use") is not True or prepared.get("consumed") is not False:
        raise SmokeLauncherError("prepared handoff is not single-use")
    if prepared.get("nonce") != nonce_value or prepared.get("repo_root") != str(repo_root.resolve()):
        raise SmokeLauncherError("prepared handoff nonce or repository binding mismatch")
    if prepared.get("output_dir") != str(output_dir) or prepared.get("process_evidence_dir") != str(process_dir):
        raise SmokeLauncherError("prepared handoff path binding mismatch")
    config_path = Path(prepared.get("config_path", ""))
    if not config_path.is_absolute() or config_path != repo_root.resolve() / prepared.get("config_relative_path", ""):
        raise SmokeLauncherError("prepared handoff config path binding mismatch")
    if config_path.is_symlink() or not config_path.is_file() or config_path.stat().st_nlink != 1:
        raise SmokeLauncherError("prepared handoff config is not a regular file")
    if type(prepared.get("config_size_bytes")) is not int or prepared["config_size_bytes"] != config_path.stat().st_size:
        raise SmokeLauncherError("prepared handoff config size mismatch")
    _validate_sha(prepared.get("config_file_sha256"), "prepared handoff config_file_sha256")
    _validate_sha(prepared.get("config_canonical_sha256"), "prepared handoff config_canonical_sha256")
    if prepared["config_file_sha256"] != sha256_file(config_path):
        raise SmokeLauncherError("prepared handoff config SHA mismatch")
    prepared_config = load_smoke_config(repo_root, config_path)
    if prepared.get("smoke_id") != prepared_config.get("smoke_id"):
        raise SmokeLauncherError("prepared handoff config identity mismatch")
    if prepared["config_canonical_sha256"] != sha256_bytes(canonical_json_bytes(prepared_config)):
        raise SmokeLauncherError("prepared handoff canonical config SHA mismatch")
    if prepared.get("runtime_data_root") != str(runtime_data_root):
        raise SmokeLauncherError("prepared handoff runtime data root mismatch")
    if type(prepared.get("launcher_pid")) is not int or prepared["launcher_pid"] != os.getppid():
        raise SmokeLauncherError("prepared handoff launcher PID mismatch")
    child_argv = prepared.get("child_argv")
    if not isinstance(child_argv, list) or not child_argv or any(type(token) is not str or not token or "\x00" in token for token in child_argv):
        raise SmokeLauncherError("prepared handoff child argv is invalid")
    if prepared.get("child_argv_schema") == "module-v1":
        expected_child_argv = _canonical_child_argv(Path(child_argv[0]), repo_root)
    elif prepared.get("child_argv_schema") == "synthetic-test-v1" and os.environ.get("P3_SMOKE_SYNTHETIC_CHILD") == "1":
        expected_child_argv = [str(Path(child_argv[0]).resolve()), *child_argv[1:]]
    else:
        raise SmokeLauncherError("prepared handoff child argv schema is invalid")
    if child_argv != expected_child_argv:
        raise SmokeLauncherError("prepared handoff child argv schema mismatch")
    _validate_executable_identity(prepared.get("child_executable_identity"))
    expected_identity = _executable_identity(Path(expected_child_argv[0]))
    if prepared.get("child_executable_identity") != expected_identity:
        raise SmokeLauncherError("prepared handoff executable identity mismatch")
    actual_argv = _actual_child_argv(require_module=prepared["child_argv_schema"] == "module-v1")
    actual_identity = _actual_child_executable_identity()
    if actual_identity != expected_identity:
        raise SmokeLauncherError("actual child executable identity mismatch")
    if prepared.get("child_argv") != actual_argv or prepared.get("child_argv_sha256") != _child_argv_sha256(actual_argv):
        raise SmokeLauncherError("prepared handoff child argv mismatch")
    if prepared.get("child_argv_sha256") != _child_argv_sha256(prepared["child_argv"]):
        raise SmokeLauncherError("prepared handoff child argv SHA mismatch")
    prepared_sha256 = sha256_file(handoff_path)
    receipt = {
        "schema_version": 1,
        "consumed": True,
        "nonce": nonce_value,
        "launcher_pid": prepared["launcher_pid"],
        "child_pid": os.getpid(),
        "child_ppid": os.getppid(),
        "child_argv": prepared["child_argv"],
        "child_argv_sha256": prepared["child_argv_sha256"],
        "child_executable_identity": prepared["child_executable_identity"],
        "child_argv_schema": prepared["child_argv_schema"],
        "prepared_relative_path": HANDOFF_PREPARED_NAME,
        "prepared_sha256": prepared_sha256,
        "output_dir": str(output_dir),
        "process_evidence_dir": str(process_dir),
        "runtime_data_root": prepared["runtime_data_root"],
        "smoke_id": prepared["smoke_id"],
        "config_relative_path": prepared["config_relative_path"],
        "config_path": prepared["config_path"],
        "config_size_bytes": prepared["config_size_bytes"],
        "config_file_sha256": prepared["config_file_sha256"],
        "config_canonical_sha256": prepared["config_canonical_sha256"],
        "consumed_at_utc": _utc_now(),
    }
    receipt_sha256 = _atomic_json(receipt_path, receipt)
    return receipt, receipt_sha256


def _validate_handoff_receipt(
    process_dir: Path,
    invocation: dict[str, Any],
    child_identity: dict[str, int | None],
) -> tuple[dict[str, Any], str]:
    prepared_path = process_dir / HANDOFF_PREPARED_NAME
    receipt_path = process_dir / HANDOFF_RECEIPT_NAME
    if prepared_path.is_symlink() or not prepared_path.is_file() or prepared_path.stat().st_nlink != 1:
        raise SmokeLauncherError("prepared handoff evidence is missing or invalid")
    receipt = _read_object(receipt_path)
    required = {
        "schema_version",
        "consumed",
        "nonce",
        "launcher_pid",
        "child_pid",
        "child_ppid",
        "child_argv",
        "child_argv_sha256",
        "child_executable_identity",
        "child_argv_schema",
        "prepared_relative_path",
        "prepared_sha256",
        "output_dir",
        "process_evidence_dir",
        "runtime_data_root",
        "smoke_id",
        "config_relative_path",
        "config_path",
        "config_size_bytes",
        "config_file_sha256",
        "config_canonical_sha256",
        "consumed_at_utc",
    }
    if set(receipt) != required or receipt.get("schema_version") != 1 or receipt.get("consumed") is not True:
        raise SmokeLauncherError("handoff receipt schema is invalid")
    if receipt.get("prepared_relative_path") != HANDOFF_PREPARED_NAME:
        raise SmokeLauncherError("handoff receipt prepared path is invalid")
    for field in ("launcher_pid", "child_pid", "child_ppid"):
        if type(receipt.get(field)) is not int:
            raise SmokeLauncherError(f"handoff receipt PID field is invalid: {field}")
    _validate_nonce(receipt.get("nonce"))
    if not isinstance(receipt.get("child_argv"), list) or any(type(token) is not str or not token or "\x00" in token for token in receipt["child_argv"]):
        raise SmokeLauncherError("handoff receipt child argv is invalid")
    _validate_sha(receipt.get("child_argv_sha256"), "handoff receipt child_argv_sha256")
    if receipt.get("child_argv_sha256") != _child_argv_sha256(receipt["child_argv"]):
        raise SmokeLauncherError("handoff receipt child argv SHA mismatch")
    _validate_executable_identity(receipt.get("child_executable_identity"), "handoff receipt executable identity")
    _validate_sha(receipt.get("prepared_sha256"), "handoff receipt prepared_sha256")
    if not all(type(receipt.get(field)) is str and receipt[field] for field in ("output_dir", "process_evidence_dir", "runtime_data_root", "consumed_at_utc")):
        raise SmokeLauncherError("handoff receipt string field is invalid")
    if type(receipt.get("smoke_id")) is not str or type(receipt.get("config_relative_path")) is not str or type(receipt.get("config_path")) is not str:
        raise SmokeLauncherError("handoff receipt config identity fields are invalid")
    if type(receipt.get("config_size_bytes")) is not int or receipt["config_size_bytes"] < 1:
        raise SmokeLauncherError("handoff receipt config size is invalid")
    _validate_sha(receipt.get("config_file_sha256"), "handoff receipt config_file_sha256")
    _validate_sha(receipt.get("config_canonical_sha256"), "handoff receipt config_canonical_sha256")
    if receipt.get("nonce") != invocation["nonce"] or receipt.get("launcher_pid") != invocation["launcher_pid"]:
        raise SmokeLauncherError("handoff receipt nonce or launcher PID mismatch")
    if receipt.get("child_pid") != child_identity.get("pid") or receipt.get("child_ppid") != child_identity.get("parent_pid"):
        raise SmokeLauncherError("handoff receipt child PID or PPID mismatch")
    if receipt.get("child_argv_sha256") != invocation["child_argv_sha256"]:
        raise SmokeLauncherError("handoff receipt argv SHA mismatch")
    if receipt.get("child_argv") != invocation["child_argv"] or receipt.get("child_executable_identity") != invocation["child_executable_identity"]:
        raise SmokeLauncherError("handoff receipt argv identity mismatch")
    if receipt.get("child_argv_schema") != invocation["child_argv_schema"]:
        raise SmokeLauncherError("handoff receipt argv schema mismatch")
    if receipt.get("output_dir") != invocation["output_dir"] or receipt.get("process_evidence_dir") != invocation["process_evidence_dir"]:
        raise SmokeLauncherError("handoff receipt path mismatch")
    if receipt.get("runtime_data_root") != invocation["runtime_data_root"]:
        raise SmokeLauncherError("handoff receipt runtime data root mismatch")
    for field in ("smoke_id", "config_relative_path", "config_path", "config_size_bytes", "config_file_sha256", "config_canonical_sha256"):
        if receipt.get(field) != invocation[field]:
            raise SmokeLauncherError(f"handoff receipt config binding mismatch: {field}")
    prepared = _read_object(prepared_path)
    if prepared.get("runtime_data_root") != receipt.get("runtime_data_root"):
        raise SmokeLauncherError("prepared and receipt runtime data root mismatch")
    if (prepared.get("child_argv") != receipt.get("child_argv")
            or prepared.get("child_executable_identity") != receipt.get("child_executable_identity")
            or prepared.get("child_argv_schema") != receipt.get("child_argv_schema")):
        raise SmokeLauncherError("prepared and receipt argv identity mismatch")
    if receipt.get("prepared_sha256") != sha256_file(prepared_path):
        raise SmokeLauncherError("handoff prepared SHA mismatch")
    if receipt.get("prepared_sha256") != invocation.get("handoff_prepared_sha256"):
        raise SmokeLauncherError("handoff prepared SHA binding mismatch")
    config_path = Path(receipt["config_path"])
    if config_path.is_symlink() or not config_path.is_file() or config_path.stat().st_nlink != 1:
        raise SmokeLauncherError("handoff receipt config is not a regular file")
    if receipt["config_size_bytes"] != config_path.stat().st_size or receipt["config_file_sha256"] != sha256_file(config_path):
        raise SmokeLauncherError("handoff receipt config file binding mismatch")
    loaded_config = load_smoke_config(Path(invocation["repo_root"]), config_path)
    spec = get_smoke_runtime_spec(loaded_config.get("smoke_id"))
    if receipt["smoke_id"] != loaded_config["smoke_id"] or receipt["config_relative_path"] != spec.config_relative_path:
        raise SmokeLauncherError("handoff receipt config identity mismatch")
    if receipt["config_canonical_sha256"] != sha256_bytes(canonical_json_bytes(loaded_config)):
        raise SmokeLauncherError("handoff receipt canonical config SHA mismatch")
    receipt_sha256 = sha256_file(receipt_path)
    return receipt, receipt_sha256


def _send_group_signal(child: subprocess.Popen[bytes], signum: int) -> bool:
    try:
        os.killpg(child.pid, signum)
        return True
    except OSError:
        return False


def _run_child(child_argv: list[str], env: dict[str, str], console) -> tuple[int, list[dict[str, Any]], bool, dict[str, int | None]]:
    state: dict[str, Any] = {"child": None, "deadline": None, "kill_sent": False, "events": []}

    def handle_signal(signum: int, _frame) -> None:
        child = state["child"]
        event: dict[str, Any] = {
            "signal_number": signum,
            "signal_name": _signal_name(signum),
            "received_at_utc": _utc_now(),
            "forwarded_to_child_process_group": False,
        }
        if isinstance(child, subprocess.Popen) and child.poll() is None:
            event["forwarded_to_child_process_group"] = _send_group_signal(child, signum)
            state["deadline"] = time.monotonic() + SIGNAL_GRACE_SECONDS
        state["events"].append(event)

    previous = {signum: signal.getsignal(signum) for signum in _SIGNALS}
    child: subprocess.Popen[bytes] | None = None
    session_id: int | None = None
    process_group_id: int | None = None
    try:
        for signum in _SIGNALS:
            signal.signal(signum, handle_signal)
        child = subprocess.Popen(
            child_argv,
            env=env,
            cwd=env.get("P3_SMOKE_REPO_ROOT") or None,
            stdout=console,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        state["child"] = child
        session_id = os.getsid(child.pid)
        process_group_id = os.getpgid(child.pid)
        for event in state["events"]:
            if not event["forwarded_to_child_process_group"] and child.poll() is None:
                event["forwarded_to_child_process_group"] = _send_group_signal(child, event["signal_number"])
                state["deadline"] = time.monotonic() + SIGNAL_GRACE_SECONDS
        while child.poll() is None:
            deadline = state["deadline"]
            if isinstance(deadline, float) and time.monotonic() >= deadline:
                if _send_group_signal(child, signal.SIGKILL):
                    state["kill_sent"] = True
                state["deadline"] = None
            time.sleep(0.05)
        returncode = child.wait()
    except BaseException:
        if child is not None and child.poll() is None:
            _send_group_signal(child, signal.SIGTERM)
            try:
                child.wait(timeout=SIGNAL_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                _send_group_signal(child, signal.SIGKILL)
                child.wait()
        raise
    finally:
        for signum, previous_handler in previous.items():
            signal.signal(signum, previous_handler)
        console.flush()
        os.fsync(console.fileno())
    return returncode, list(state["events"]), bool(state["kill_sent"]), {
        "pid": child.pid,
        "sid": session_id,
        "pgid": process_group_id,
        "parent_pid": os.getpid(),
    }


def _entry_summary(output_dir: Path, expected_identity: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        return validate_entry_output(output_dir, expected_identity=expected_identity)
    except (OSError, SmokeEvidenceError, ValueError) as exc:
        return {"entry_success_accepted": False, "entry_status": "FAILED_ENTRY_CONTRACT", "entry_validation_error": f"{type(exc).__name__}: {exc}"}


def _finalize_process(
    process_dir: Path,
    output_dir: Path,
    *,
    invocation: dict[str, Any],
    returncode: int | None,
    signal_events: list[dict[str, Any]],
    kill_sent: bool,
    process_started_at: str,
    process_elapsed_seconds: float,
    child_identity: dict[str, int | None],
    internal_error: BaseException | None = None,
) -> int:
    exit_ref: dict[str, Any] = {"present": False, "relative_path": "process_exit_code.txt"}
    if isinstance(returncode, int):
        atomic_write(process_dir / "process_exit_code.txt", f"{returncode}\n".encode("ascii"))
        exit_ref = file_ref(process_dir, "process_exit_code.txt")
    _atomic_json(process_dir / "process_timing.json", {
        "schema_version": 1,
        "started_at_utc": process_started_at,
        "finished_at_utc": _utc_now(),
        "elapsed_seconds": process_elapsed_seconds,
    })
    handoff_receipt: dict[str, Any] | None = None
    handoff_receipt_sha256: str | None = None
    handoff_error: str | None = None
    try:
        process_invocation = _read_object(process_dir / "process_invocation.json")
        if process_invocation != invocation:
            raise SmokeLauncherError("process invocation evidence drift")
        handoff_receipt, handoff_receipt_sha256 = _validate_handoff_receipt(process_dir, invocation, child_identity)
    except (OSError, SmokeEvidenceError, SmokeLauncherError, ValueError) as exc:
        handoff_error = f"{type(exc).__name__}: {exc}"
    expected_identity = None if handoff_receipt is None else {
        "nonce": invocation["nonce"],
        "launcher_pid": invocation["launcher_pid"],
        "child_pid": child_identity.get("pid"),
        "child_ppid": child_identity.get("parent_pid"),
        "child_argv_sha256": invocation["child_argv_sha256"],
        "handoff_receipt_sha256": handoff_receipt_sha256,
        "handoff_receipt_relative_path": HANDOFF_RECEIPT_NAME,
        "smoke_id": handoff_receipt["smoke_id"],
        "config_relative_path": handoff_receipt["config_relative_path"],
        "config_canonical_sha256": handoff_receipt["config_canonical_sha256"],
    }
    if output_dir.is_dir():
        entry = _entry_summary(output_dir, expected_identity=expected_identity)
    else:
        entry = {
            "entry_success_accepted": False,
            "entry_status": "FAILED_ENTRY_MISSING",
            "entry_validation_error": "child output directory is missing",
        }
    if handoff_error is not None:
        entry = {
            "entry_success_accepted": False,
            "entry_status": "FAILED_HANDOFF_CONTRACT",
            "entry_validation_error": handoff_error,
        }
    success = returncode == 0 and handoff_receipt is not None and entry.get("entry_success_accepted") is True and internal_error is None
    if not success:
        _atomic_json(process_dir / "process_partial_inventory.json", inventory(
            process_dir,
            frozenset({"process_partial_inventory.json", "process_inventory.json", "process_completion.json"}),
        ))
    process_inventory = inventory(process_dir, PROCESS_INVENTORY_EXCLUDED)
    process_inventory_bytes = canonical_json_bytes(process_inventory)
    atomic_write(process_dir / "process_inventory.json", process_inventory_bytes)
    status = "COMPLETED" if success else "FAILED_ENTRY_CONTRACT" if returncode == 0 else "FAILED_CHILD"
    completion = {
        "schema_version": 1,
        "status": status,
        "child_returncode": returncode,
        "launcher_exit_code": 0 if success else 2,
        "signal_events": signal_events,
        "child_kill_escalated_to_sigkill": kill_sent,
        "child_pid": child_identity.get("pid"),
        "child_sid": child_identity.get("sid"),
        "child_pgid": child_identity.get("pgid"),
        "child_parent_pid": child_identity.get("parent_pid"),
        "nonce": invocation["nonce"],
        "child_argv": invocation["child_argv"],
        "process_exit_code": exit_ref,
        "process_inventory": {"relative_path": "process_inventory.json", "size_bytes": len(process_inventory_bytes), "sha256": sha256_bytes(process_inventory_bytes)},
        "entry": entry,
        "handoff_prepared": file_ref(process_dir, HANDOFF_PREPARED_NAME),
        "handoff_receipt": file_ref(process_dir, HANDOFF_RECEIPT_NAME),
        "handoff_receipt_sha256": handoff_receipt_sha256,
        "runtime_data_root": invocation["runtime_data_root"],
        "smoke_id": invocation["smoke_id"],
        "config_relative_path": invocation["config_relative_path"],
        "config_path": invocation["config_path"],
        "config_size_bytes": invocation["config_size_bytes"],
        "config_file_sha256": invocation["config_file_sha256"],
        "config_canonical_sha256": invocation["config_canonical_sha256"],
        "data_role": "train_core",
        "handoff_validation_error": handoff_error,
        "internal_error": None if internal_error is None else {"type": type(internal_error).__name__, "message": str(internal_error)},
        "non_training": True,
        "non_selection": True,
        "non_reportable": True,
        "confirmatory_metrics_accessed": False,
        "dataset_test_accessed_by_this_process": False,
    }
    _atomic_json(process_dir / "process_completion.json", completion)
    return 0 if success else 2


def launch_smoke(
    child_argv: list[str],
    output_dir: Path,
    process_evidence_dir: Path,
    *,
    data_root: Path,
    child_python: Path,
    repo_root: Path,
    env: dict[str, str] | None = None,
    nonce: str | None = None,
    config_path: Path | None = None,
    allow_noncanonical_child: bool = True,
) -> int:
    """Launch a child without creating or reserving its output directory.

    The production CLI always supplies the frozen module argv. The optional
    synthetic schema preserves the lower-level CPU harness used by historical
    tests; it is explicitly marked in the handoff and is never produced by
    ``main``.
    """

    canonical_repo_root = _canonical_repo_root(repo_root)
    _config, _resolved_config_path, config_binding = _config_binding(canonical_repo_root, config_path)
    canonical_data_root = _canonical_runtime_data_root(data_root)
    _validate_launch_paths(output_dir, process_evidence_dir, child_python)
    if os.environ.get(SMOKE_AUTH_ENV) != "1" or os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or os.environ.get("PYTHONNOUSERSITE") != "1":
        raise SmokeLauncherError("authorized smoke environment is not exact")
    actual_nonce = nonce or secrets.token_hex(16)
    _validate_nonce(actual_nonce)
    child_executable_identity = _executable_identity(child_python)
    module_child_argv = _canonical_child_argv(child_python, canonical_repo_root)
    if child_argv == module_child_argv:
        canonical_child_argv = module_child_argv
        child_argv_schema = "module-v1"
    elif allow_noncanonical_child and isinstance(child_argv, list) and child_argv and child_argv[0] == child_executable_identity["canonical_path"]:
        canonical_child_argv = [str(Path(child_argv[0]).resolve()), *child_argv[1:]]
        if any(type(token) is not str or not token or "\x00" in token for token in canonical_child_argv):
            raise SmokeLauncherError("synthetic child argv is invalid")
        child_argv_schema = "synthetic-test-v1"
    else:
        raise SmokeLauncherError("child argv is not the frozen module handoff")
    process_evidence_dir.parent.mkdir(parents=True, exist_ok=True)
    process_evidence_dir.mkdir()
    child_env = dict(os.environ if env is None else env)
    child_env.update({
        "P3_SMOKE_REPO_ROOT": str(canonical_repo_root),
        "P3_SMOKE_OUTPUT_DIR": str(output_dir),
        "P3_SMOKE_PROCESS_EVIDENCE_DIR": str(process_evidence_dir),
        "P3_SMOKE_HANDOFF_PATH": str(process_evidence_dir / HANDOFF_PREPARED_NAME),
        "P3_SMOKE_DATA_ROOT": str(canonical_data_root),
        SMOKE_NONCE_ENV: actual_nonce,
        "PYTHONNOUSERSITE": "1",
        "CUDA_VISIBLE_DEVICES": "0",
        "P3_SMOKE_SYNTHETIC_CHILD": "1" if child_argv_schema == "synthetic-test-v1" else "0",
    })
    handoff = {
        "schema_version": 1,
        "state": "PREPARED",
        "single_use": True,
        "consumed": False,
        "nonce": actual_nonce,
        "launcher_pid": os.getpid(),
        "repo_root": str(canonical_repo_root),
        "output_dir": str(output_dir),
        "process_evidence_dir": str(process_evidence_dir),
        "runtime_data_root": str(canonical_data_root),
        **config_binding,
        "child_argv": canonical_child_argv,
        "child_argv_sha256": _child_argv_sha256(canonical_child_argv),
        "child_executable_identity": child_executable_identity,
        "child_argv_schema": child_argv_schema,
        "created_at_utc": _utc_now(),
    }
    handoff_path = process_evidence_dir / HANDOFF_PREPARED_NAME
    handoff_sha256 = _atomic_json(handoff_path, handoff)
    invocation = {
        "schema_version": 1,
        "mode": "smoke",
        "nonce": actual_nonce,
        "launcher_pid": os.getpid(),
        "child_argv": canonical_child_argv,
        "child_argv_sha256": _child_argv_sha256(canonical_child_argv),
        "child_executable_identity": child_executable_identity,
        "child_argv_schema": child_argv_schema,
        "child_python": child_executable_identity["canonical_path"],
        "repo_root": str(canonical_repo_root),
        "output_dir": str(output_dir),
        "process_evidence_dir": str(process_evidence_dir),
        "runtime_data_root": str(canonical_data_root),
        **config_binding,
        "data_role": "train_core",
        "handoff_prepared_relative_path": HANDOFF_PREPARED_NAME,
        "handoff_prepared_sha256": handoff_sha256,
        "output_dir_not_created_by_launcher": True,
        "process_evidence_owner": "launcher",
        "nonportable": True,
        "confirmatory_metrics_accessed": False,
        "dataset_test_accessed_by_this_process": False,
        "cuda_visible_devices": "0",
        "python_no_user_site": "1",
    }
    _atomic_json(process_evidence_dir / "process_invocation.json", invocation)
    console_path = process_evidence_dir / "process_console.log"
    start = time.monotonic()
    started_at = _utc_now()
    child_identity: dict[str, int | None] = {"sid": None, "pgid": None}
    signal_events: list[dict[str, Any]] = []
    kill_sent = False
    returncode: int | None = None
    internal_error: BaseException | None = None
    try:
        with console_path.open("xb") as console:
            returncode, signal_events, kill_sent, child_identity = _run_child(canonical_child_argv, child_env, console)
    except BaseException as error:
        internal_error = error
    return _finalize_process(
        process_evidence_dir,
        output_dir,
        invocation=invocation,
        returncode=returncode,
        signal_events=signal_events,
        kill_sent=kill_sent,
        process_started_at=started_at,
        process_elapsed_seconds=time.monotonic() - start,
        child_identity=child_identity,
        internal_error=internal_error,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rtdetr-baseline-smoke")
    sub = parser.add_subparsers(dest="mode", required=True)
    check = sub.add_parser("contract-check")
    check.add_argument("--repo-root", type=Path, required=True)
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--repo-root", type=Path, required=True)
    smoke.add_argument("--data-root", type=Path, required=True)
    smoke.add_argument("--output-dir", type=Path, required=True)
    smoke.add_argument("--process-evidence-dir", type=Path, required=True)
    smoke.add_argument("--config", type=Path, default=None)
    smoke.add_argument("--child-python", type=Path, default=Path(sys.executable))
    sub.add_parser("_child").add_argument("--repo-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.mode == "contract-check":
            print(canonical_json_bytes(contract_check(args.repo_root)).decode("utf-8"))
            return 0
        if args.mode == "_child":
            handoff_receipt, handoff_receipt_sha256 = _consume_handoff(args.repo_root)
            child_kwargs = {
                "handoff_receipt": handoff_receipt,
                "handoff_receipt_sha256": handoff_receipt_sha256,
            }
            spec = get_smoke_runtime_spec(handoff_receipt.get("smoke_id"))
            if spec.allow_outer_launch:
                child_kwargs["config_path"] = Path(handoff_receipt["config_path"])
            run_authorized_smoke(
                args.repo_root,
                Path(handoff_receipt["runtime_data_root"]),
                Path(os.environ["P3_SMOKE_OUTPUT_DIR"]),
                **child_kwargs,
            )
            return 0
        validate_real_smoke_environment()
        repo_root = _canonical_repo_root(args.repo_root)
        _config, config_path, _binding = _config_binding(repo_root, args.config)
        if args.data_root is None or any(part.casefold() == "test" for part in args.data_root.parts):
            raise SmokeLauncherError("real smoke data root is invalid")
        _validate_frozen_runtime_paths(
            repo_root,
            args.output_dir,
            args.process_evidence_dir,
            config=_config,
            config_path=config_path,
        )
        child_argv = _canonical_child_argv(args.child_python, repo_root)
        env = dict(os.environ)
        env["P3_SMOKE_OUTPUT_DIR"] = str(args.output_dir)
        return launch_smoke(
            child_argv,
            args.output_dir,
            args.process_evidence_dir,
            data_root=args.data_root,
            child_python=args.child_python,
            repo_root=repo_root,
            env=env,
            config_path=config_path,
        )
    except (SmokeContractError, SmokeEvidenceError, SmokeLauncherError) as error:
        print(f"rtdetr-baseline-smoke: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    except Exception as error:
        print(f"rtdetr-baseline-smoke: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
