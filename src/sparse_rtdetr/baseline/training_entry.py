"""Fail-closed inner entry boundary for the synthetic T5D process contract.

The entry boundary is intentionally a pure contract adapter.  Its executable
path accepts only the injected CPU ports already certified by T5C; production
entry execution remains disabled by the contract and by this module.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any, Mapping

from sparse_rtdetr.baseline import training_adapter as _adapter


PROCESS_SCHEMA_VERSION = 1
PROCESS_CONTRACT_ID = "rtdetrv2_r18_visdrone_baseline_training_process_v1"
ENTRY_ID = "rtdetrv2_r18_visdrone_training_entry_t5d_v1"
ENTRY_MODE = "synthetic"
PROCESS_CONFIG_RELATIVE_PATH = "configs/baseline/rtdetrv2_r18_visdrone_training_process_v1.json"
ENTRY_MODULE_RELATIVE_PATH = "src/sparse_rtdetr/baseline/training_entry.py"
LAUNCHER_MODULE_RELATIVE_PATH = "src/sparse_rtdetr/baseline/training_process_launcher.py"
TRAINING_EVIDENCE_ROOT_RELATIVE_PATH = "artifacts/training/rtdetrv2_r18_visdrone_training_evidence_v1"
PROCESS_EVIDENCE_ROOT_RELATIVE_PATH = "artifacts/process_evidence/rtdetrv2_r18_visdrone_training_process_v1"
LAUNCH_EVIDENCE_ROOT_RELATIVE_PATH = "artifacts/outer_launch_evidence/rtdetrv2_r18_visdrone_training_process_v1"
ENTRY_RECEIPT_SUFFIX = ".t5d-entry-receipt.json"

# These values are filled after the config's bytes are frozen.  They do not
# include either new module, avoiding a self-referential source digest.
PROCESS_CONFIG_RAW_SIZE_BYTES = 10468
PROCESS_CONFIG_RAW_SHA256 = "a16ff1c31e0a935f0ab6ab9224b7c3f70c377006d00562dc5b2bc1b44f70ed89"
PROCESS_CONFIG_CANONICAL_SIZE_BYTES = 9330
PROCESS_CONFIG_CANONICAL_SHA256 = "8469bde5b8e87e47ef8df0b9674aa1e685bf993f90ef3bc99aa3354bbe95f19c"

SOURCE_MODULES = (
    ("package", "src/sparse_rtdetr/__init__.py"),
    ("package", "src/sparse_rtdetr/baseline/__init__.py"),
    ("entry", ENTRY_MODULE_RELATIVE_PATH),
    ("launcher", LAUNCHER_MODULE_RELATIVE_PATH),
    ("prepared_adapter", "src/sparse_rtdetr/baseline/training_adapter.py"),
    ("training_contract", "src/sparse_rtdetr/baseline/training_contract.py"),
    ("runtime_plan", "src/sparse_rtdetr/baseline/training_runtime.py"),
    ("training_evidence", "src/sparse_rtdetr/baseline/training_evidence.py"),
    ("primary_evaluator", "src/sparse_rtdetr/baseline/primary_evaluator.py"),
    ("package", "src/sparse_rtdetr/data_protocol/__init__.py"),
    ("evaluation_protocol", "src/sparse_rtdetr/data_protocol/evaluation.py"),
    ("protocol_schema", "src/sparse_rtdetr/data_protocol/schema.py"),
)
ENVIRONMENT_KEYS = ("CUDA_VISIBLE_DEVICES", "PYTHONNOUSERSITE", "PYTHONDONTWRITEBYTECODE", "PYTHONPATH")
FUTURE_TARGETS = {
    "training_evidence_root": TRAINING_EVIDENCE_ROOT_RELATIVE_PATH,
    "process_evidence_root": PROCESS_EVIDENCE_ROOT_RELATIVE_PATH,
    "launch_evidence_root": LAUNCH_EVIDENCE_ROOT_RELATIVE_PATH,
}
FORBIDDEN_WORDS = ("test", "confirmatory", "tuning", "speed", "benchmark", "validation", "dataset", "dataloader")
HEX_RE = re.compile(r"[0-9a-f]+\Z")

__all__ = (
    "TrainingEntryError",
    "build_synthetic_entry_descriptor",
    "validate_entry_descriptor",
    "run_synthetic_entry",
    "validate_entry_result",
)


class TrainingEntryError(ValueError):
    """Raised when an invocation descriptor or synthetic result drifts."""


def _fail(message: str) -> None:
    raise TrainingEntryError(message)


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
        raise TrainingEntryError("value is not canonical JSON") from exc


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


def _relative_path(value: Any, field: str) -> str:
    value = _string(value, field)
    if value.startswith("/") or "\x00" in value or "//" in value or value.endswith("/"):
        _fail(f"{field} is not a normalized relative path")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        _fail(f"{field} contains traversal or empty components")
    return value


def _absolute_path(value: Any, field: str, *, require_directory: bool = False) -> str:
    value = _string(value, field)
    if not value.startswith("/") or "\x00" in value or value.endswith("/") or "//" in value or "/./" in value:
        _fail(f"{field} must be a normalized absolute path")
    path = Path(value)
    if any(part == ".." for part in path.parts):
        _fail(f"{field} contains traversal")
    if path.resolve(strict=False) != path:
        _fail(f"{field} is not canonical")
    return value


def _token(value: Any, field: str) -> str:
    value = _string(value, field)
    lowered = value.casefold()
    if "/" in value or "\\" in value or any(word in lowered for word in FORBIDDEN_WORDS):
        _fail(f"{field} contains a forbidden role or path token")
    return value


def _nonce(value: Any) -> str:
    value = _string(value, "nonce")
    if len(value) < 32 or HEX_RE.fullmatch(value) is None:
        _fail("nonce must be a high-entropy lowercase hexadecimal token")
    return value


class _DirectoryLease:
    def __init__(self, fds: list[int], snapshots: list[os.stat_result], components: tuple[str, ...] = ()) -> None:
        self.fds = fds
        self.snapshots = snapshots
        self.components = components
        self._closed = False

    @property
    def fd(self) -> int:
        return self.fds[-1]

    @property
    def snapshot(self) -> os.stat_result:
        return self.snapshots[-1]

    def revalidate(self, *, include_nlink: bool = False) -> None:
        if self._closed:
            _fail("directory lease is closed")
        for fd, expected in zip(self.fds, self.snapshots):
            observed = _fstat(fd)
            fields = ("st_dev", "st_ino", "st_mode", "st_uid", "st_gid")
            if include_nlink:
                fields += ("st_nlink",)
            if any(getattr(observed, field) != getattr(expected, field) for field in fields):
                _fail("directory identity drift")
        if self.components:
            if len(self.components) != len(self.fds) - 1:
                _fail("directory path component drift")
            for index, component in enumerate(self.components, 1):
                observed = _stat_at(self.fds[index - 1], component)
                expected = self.snapshots[index]
                if observed is None or not stat.S_ISDIR(observed.st_mode):
                    _fail("directory path component drift")
                fields = ("st_dev", "st_ino", "st_mode", "st_uid", "st_gid")
                if include_nlink:
                    fields += ("st_nlink",)
                if any(getattr(observed, field) != getattr(expected, field) for field in fields):
                    _fail("directory path component drift")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        first_error: OSError | None = None
        for fd in reversed(self.fds):
            try:
                _close_fd(fd)
            except OSError as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error


def _fstat(fd: int) -> os.stat_result:
    while True:
        try:
            return os.fstat(fd)
        except InterruptedError:
            continue


def _close_fd(fd: int) -> None:
    while True:
        try:
            os.close(fd)
            return
        except InterruptedError:
            continue


def _listdir_fd(fd: int) -> list[str]:
    while True:
        try:
            names = os.listdir(fd)
            break
        except InterruptedError:
            continue
    if type(names) is not list or any(type(name) is not str for name in names):
        _fail("directory enumeration returned invalid names")
    return names


def _open_fd(name: str, flags: int, *, mode: int = 0, dir_fd: int | None = None) -> int:
    kwargs = {} if dir_fd is None else {"dir_fd": dir_fd}
    while True:
        try:
            return os.open(name, flags, mode, **kwargs)
        except InterruptedError:
            continue


def _directory_flags() -> int:
    return os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)


def _file_flags() -> int:
    return os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)


def _directory_snapshot(observed: os.stat_result, field: str) -> os.stat_result:
    if not stat.S_ISDIR(observed.st_mode) or observed.st_nlink < 2:
        _fail(f"{field} is not a stable directory")
    return observed


def _open_directory_chain(value: str | os.PathLike[str]) -> _DirectoryLease:
    raw = _absolute_path(os.fspath(value), "directory path")
    fds: list[int] = []
    snapshots: list[os.stat_result] = []
    components = tuple(Path(raw).parts[1:])
    try:
        fd = _open_fd("/", _directory_flags())
        fds.append(fd)
        snapshots.append(_directory_snapshot(_fstat(fd), "root directory"))
        for component in components:
            fd = _open_fd(component, _directory_flags(), dir_fd=fds[-1])
            fds.append(fd)
            snapshots.append(_directory_snapshot(_fstat(fd), f"directory component {component}"))
        lease = _DirectoryLease(fds, snapshots, components)
        lease.revalidate(include_nlink=True)
        return lease
    except OSError as exc:
        for fd in reversed(fds):
            try:
                _close_fd(fd)
            except OSError:
                pass
        raise TrainingEntryError("directory path could not be opened stably") from exc
    except TrainingEntryError:
        for fd in reversed(fds):
            try:
                _close_fd(fd)
            except OSError:
                pass
        raise


def _open_relative_parent(root_fd: int, relative: str, field: str) -> tuple[int, list[tuple[int, int, str, os.stat_result]]]:
    parts = _relative_path(relative, field).split("/")
    records: list[tuple[int, int, str, os.stat_result]] = []
    parent_fd = root_fd
    for component in parts[:-1]:
        try:
            current = _open_fd(component, _directory_flags(), dir_fd=parent_fd)
        except OSError as exc:
            for fd, _, _, _ in reversed(records):
                _close_fd(fd)
            raise TrainingEntryError(f"{field} parent is not a stable directory") from exc
        try:
            observed = _directory_snapshot(_fstat(current), f"{field} parent component")
        except Exception:
            _close_fd(current)
            for fd, _, _, _ in reversed(records):
                _close_fd(fd)
            raise
        records.append((current, parent_fd, component, observed))
        parent_fd = current
    return parent_fd, records


def _revalidate_relative_parent(records: list[tuple[int, int, str, os.stat_result]], *, include_nlink: bool = False) -> None:
    for fd, parent_fd, component, expected in records:
        observed = _fstat(fd)
        if not stat.S_ISDIR(observed.st_mode):
            _fail("relative parent directory drift")
        path_observed = _stat_at(parent_fd, component)
        fields = ("st_dev", "st_ino", "st_mode", "st_uid", "st_gid")
        if include_nlink:
            fields += ("st_nlink",)
        if path_observed is None or any(getattr(observed, field) != getattr(expected, field) for field in fields) or any(getattr(path_observed, field) != getattr(expected, field) for field in fields):
            _fail("relative parent directory drift")


def _read_fd_exact(fd: int, expected_size: int, field: str) -> bytes:
    if type(expected_size) is not int or expected_size < 0:
        _fail(f"{field} has an invalid declared size")
    chunks: list[bytes] = []
    remaining = expected_size
    while remaining:
        try:
            chunk = os.read(fd, remaining)
        except InterruptedError:
            continue
        if not chunk:
            _fail(f"{field} ended before its declared size")
        if len(chunk) > remaining:
            _fail(f"{field} returned more bytes than declared")
        chunks.append(chunk)
        remaining -= len(chunk)
    while True:
        try:
            extra = os.read(fd, 1)
            break
        except InterruptedError:
            continue
    if extra:
        _fail(f"{field} grew while being read")
    return b"".join(chunks)


def _require_regular_file(observed: os.stat_result, field: str) -> None:
    if not stat.S_ISREG(observed.st_mode) or observed.st_nlink != 1:
        _fail(f"{field} is not a stable regular file")
    if observed.st_uid != os.getuid() or observed.st_gid != os.getgid():
        _fail(f"{field} owner drift")
    if stat.S_IMODE(observed.st_mode) & 0o002:
        _fail(f"{field} is group/world writable")


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return all(
        getattr(left, field) == getattr(right, field)
        for field in ("st_dev", "st_ino", "st_mode", "st_uid", "st_gid", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns")
    )


def _read_relative_file(root: Path, relative: str, field: str) -> tuple[bytes, os.stat_result]:
    chain = _open_directory_chain(root)
    extra_fds: list[int] = []
    file_fd: int | None = None
    try:
        parent_fd, relative_fds = _open_relative_parent(chain.fd, relative, field)
        extra_fds.extend(fd for fd, _, _, _ in relative_fds)
        filename = _relative_path(relative, field).split("/")[-1]
        _revalidate_relative_parent(relative_fds, include_nlink=True)
        file_fd = _open_fd(filename, _file_flags(), dir_fd=parent_fd)
        before = _fstat(file_fd)
        _require_regular_file(before, field)
        if before.st_dev != chain.snapshot.st_dev:
            _fail(f"{field} is on a different device")
        current_before = _stat_at(parent_fd, filename)
        if current_before is None or not _same_file_identity(before, current_before):
            _fail(f"{field} path or inode drift")
        payload = _read_fd_exact(file_fd, before.st_size, field)
        after = _fstat(file_fd)
        _require_regular_file(after, field)
        if not _same_file_identity(before, after):
            _fail(f"{field} metadata or inode drift")
        current = _stat_at(parent_fd, filename)
        if current is None or not _same_file_identity(before, current):
            _fail(f"{field} path or inode drift")
        _revalidate_relative_parent(relative_fds, include_nlink=True)
        chain.revalidate(include_nlink=True)
        return payload, before
    except OSError as exc:
        raise TrainingEntryError(f"{field} is unreadable") from exc
    finally:
        if file_fd is not None:
            _close_fd(file_fd)
        for fd in reversed(extra_fds):
            _close_fd(fd)
        chain.close()


def _read_regular(root: Path, relative: str, field: str) -> bytes:
    return _read_relative_file(root, relative, field)[0]


def _validate_existing_executable(value: str) -> str:
    path = Path(_absolute_path(value, "python_executable"))
    parent = _open_directory_chain(path.parent)
    fd: int | None = None
    try:
        fd = _open_fd(path.name, _file_flags(), dir_fd=parent.fd)
        before = _fstat(fd)
        _require_regular_file(before, "python_executable")
        if before.st_dev != parent.snapshot.st_dev:
            _fail("python_executable device drift")
        current = _stat_at(parent.fd, path.name)
        if current is None or not _same_file_identity(before, current):
            _fail("python_executable path identity drift")
        after = _fstat(fd)
        if not _same_file_identity(before, after):
            _fail("python_executable identity drift")
        parent.revalidate(include_nlink=True)
    except OSError as exc:
        raise TrainingEntryError("python_executable is not a stable file") from exc
    finally:
        if fd is not None:
            _close_fd(fd)
        parent.close()
    return str(path)


def _bound_target_parts(value: str, field: str) -> tuple[_DirectoryLease, str]:
    path = Path(_absolute_path(value, field))
    name = path.name
    if not name or name in {".", ".."}:
        _fail(f"{field} has an invalid final component")
    return _open_directory_chain(path.parent), name


def _stat_at(parent_fd: int, name: str) -> os.stat_result | None:
    while True:
        try:
            return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return None
        except InterruptedError:
            continue


def _require_absent(parent_fd: int, name: str, field: str) -> None:
    if _stat_at(parent_fd, name) is not None:
        _fail(f"{field} already exists")


def _fsync(fd: int) -> None:
    while True:
        try:
            os.fsync(fd)
            return
        except InterruptedError:
            continue


def _write_all(fd: int, payload: bytes, field: str) -> None:
    offset = 0
    while offset < len(payload):
        try:
            written = os.write(fd, payload[offset:])
        except InterruptedError:
            continue
        if written <= 0:
            _fail(f"{field} short write")
        offset += written


def _readback_fd(fd: int, payload: bytes, field: str, *, mode: int = 0o600) -> os.stat_result:
    observed = _fstat(fd)
    _require_regular_file(observed, field)
    if stat.S_IMODE(observed.st_mode) != mode or observed.st_size != len(payload):
        _fail(f"{field} metadata or size drift")
    while True:
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            break
        except InterruptedError:
            continue
    actual = _read_fd_exact(fd, len(payload), field)
    if actual != payload:
        _fail(f"{field} readback bytes drift")
    after = _fstat(fd)
    _require_regular_file(after, field)
    if not _same_file_identity(observed, after):
        _fail(f"{field} readback metadata drift")
    return after


def _create_exclusive_file(parent_fd: int, name: str, payload: bytes, field: str) -> os.stat_result:
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd: int | None = None
    try:
        parent_before = _fstat(parent_fd)
        if not stat.S_ISDIR(parent_before.st_mode):
            _fail(f"{field} parent is not a directory")
        fd = _open_fd(name, flags, mode=0o600, dir_fd=parent_fd)
        _write_all(fd, payload, field)
        _fsync(fd)
        result = _readback_fd(fd, payload, field)
        if result.st_dev != parent_before.st_dev:
            _fail(f"{field} device drift")
        return result
    except OSError as exc:
        raise TrainingEntryError(f"{field} could not be durably created") from exc
    finally:
        if fd is not None:
            _close_fd(fd)


def _mkdir_exclusive(parent_fd: int, name: str, field: str) -> os.stat_result:
    while True:
        try:
            os.mkdir(name, 0o700, dir_fd=parent_fd)
            observed = _stat_at(parent_fd, name)
            if observed is None or not stat.S_ISDIR(observed.st_mode) or observed.st_nlink < 2:
                _fail(f"{field} was not created as a stable directory")
            parent = _fstat(parent_fd)
            if observed.st_dev != parent.st_dev or observed.st_uid != os.getuid() or observed.st_gid != os.getgid() or stat.S_IMODE(observed.st_mode) != 0o700:
                _fail(f"{field} metadata drift during creation")
            return observed
        except InterruptedError:
            continue
        except OSError as exc:
            raise TrainingEntryError(f"{field} could not be created exclusively") from exc


def _stable_root_identity(value: dict[str, int]) -> dict[str, int]:
    return {key: value[key] for key in ("device", "inode", "mode", "uid", "gid", "nlink")}


def _same_root_identity(left: dict[str, int], right: dict[str, int]) -> bool:
    return _stable_root_identity(left) == _stable_root_identity(right)


def _root_operation_identity(value: dict[str, int]) -> dict[str, int]:
    return {key: value[key] for key in ("device", "inode", "mode", "uid", "gid")}


def _receipt_name(root: str, suffix: str) -> str:
    return f".{Path(root).name}{suffix}"


def _reject_production_root(value: str, field: str) -> None:
    for relative in FUTURE_TARGETS.values():
        suffix = "/" + relative
        if value.endswith(suffix) or value == relative:
            _fail(f"{field} is a production evidence target")


def _validate_bound_root_shape(value: str, field: str, receipt_suffix: str, *, require_absent: bool = False) -> str:
    raw = _absolute_path(value, field)
    _reject_production_root(raw, field)
    receipt = str(Path(raw).parent / _receipt_name(raw, receipt_suffix))
    _absolute_path(receipt, f"{field} receipt")
    parent, name = _bound_target_parts(raw, field)
    try:
        if require_absent:
            _require_absent(parent.fd, name, field)
            _require_absent(parent.fd, Path(receipt).name, f"{field} receipt")
        parent.revalidate()
    finally:
        parent.close()
    return receipt


def _entry_receipt_payload(descriptor: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": PROCESS_SCHEMA_VERSION,
        "kind": "T5D_ENTRY_CLAIM",
        "descriptor_sha256": descriptor["aggregate_sha256"],
        "run_id": descriptor["run_id"],
        "nonce": descriptor["nonce"],
        "evidence_root": descriptor["evidence_root"],
        "receipt_path": descriptor["receipt_path"],
        "status": "CLAIMED",
    }


def _claim_entry_receipt(descriptor: dict[str, Any]) -> tuple[_DirectoryLease, str, os.stat_result]:
    parent, root_name = _bound_target_parts(descriptor["evidence_root"], "entry evidence root")
    receipt_name = Path(descriptor["receipt_path"]).name
    try:
        if receipt_name != _receipt_name(descriptor["evidence_root"], ENTRY_RECEIPT_SUFFIX):
            _fail("entry receipt path drift")
        _require_absent(parent.fd, root_name, "entry evidence root")
        _require_absent(parent.fd, receipt_name, "entry receipt")
        receipt = _entry_receipt_payload(descriptor)
        receipt_raw = _canonical(receipt)
        receipt_stat = _create_exclusive_file(parent.fd, receipt_name, receipt_raw, "entry receipt")
        current = _stat_at(parent.fd, receipt_name)
        if current is None or not _same_file_identity(receipt_stat, current):
            _fail("entry receipt path identity drift")
        _fsync(parent.fd)
        _require_absent(parent.fd, root_name, "entry evidence root")
        parent.revalidate()
        return parent, receipt_name, receipt_stat
    except Exception:
        parent.close()
        raise


def _verify_entry_receipt(parent: _DirectoryLease, descriptor: dict[str, Any], receipt_name: str) -> os.stat_result:
    fd: int | None = None
    try:
        fd = _open_fd(receipt_name, _file_flags(), dir_fd=parent.fd)
        expected = _canonical(_entry_receipt_payload(descriptor))
        observed = _fstat(fd)
        _require_regular_file(observed, "entry receipt")
        if stat.S_IMODE(observed.st_mode) != 0o600:
            _fail("entry receipt mode drift")
        while True:
            try:
                os.lseek(fd, 0, os.SEEK_SET)
                break
            except InterruptedError:
                continue
        actual = _read_fd_exact(fd, observed.st_size, "entry receipt")
        if actual != expected:
            _fail("entry receipt bytes drift")
        after = _fstat(fd)
        if not _same_file_identity(observed, after):
            _fail("entry receipt metadata drift")
        current = _stat_at(parent.fd, receipt_name)
        if current is None or not _same_file_identity(observed, current):
            _fail("entry receipt path identity drift")
        return after
    except OSError as exc:
        raise TrainingEntryError("entry receipt could not be verified") from exc
    finally:
        if fd is not None:
            _close_fd(fd)


def _open_created_root(
    parent: _DirectoryLease,
    name: str,
    field: str,
    expected: os.stat_result | None = None,
) -> tuple[int, os.stat_result]:
    fd: int | None = None
    try:
        fd = _open_fd(name, _directory_flags(), dir_fd=parent.fd)
        observed = _fstat(fd)
        if not stat.S_ISDIR(observed.st_mode) or observed.st_nlink < 2:
            _fail(f"{field} is not a directory")
        if expected is not None and not _same_file_identity(observed, expected):
            _fail(f"{field} identity changed during open")
        if observed.st_dev != parent.snapshot.st_dev:
            _fail(f"{field} device drift")
        if observed.st_uid != os.getuid() or observed.st_gid != os.getgid() or stat.S_IMODE(observed.st_mode) != 0o700:
            _fail(f"{field} metadata drift")
        named = _stat_at(parent.fd, name)
        if named is None or not _same_file_identity(observed, named):
            _fail(f"{field} path identity drift")
        parent.revalidate()
        return fd, observed
    except OSError as exc:
        if fd is not None:
            _close_fd(fd)
        raise TrainingEntryError(f"{field} could not be opened stably") from exc
    except TrainingEntryError:
        if fd is not None:
            _close_fd(fd)
        raise


def _create_and_open_root(parent: _DirectoryLease, name: str, field: str) -> tuple[int, os.stat_result]:
    """Create the final bound directory through the held parent descriptor."""

    created = _mkdir_exclusive(parent.fd, name, field)
    _fsync(parent.fd)
    return _open_created_root(parent, name, field, expected=created)


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


def _parse_strict_json(raw: bytes, field: str) -> dict[str, Any]:
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n") or b"\r" in raw or b"\x00" in raw or raw.startswith(b"\xef\xbb\xbf"):
        _fail(f"{field} has non-portable raw bytes")
    try:
        text = raw[:-1].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TrainingEntryError(f"{field} is not UTF-8") from exc

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                _fail(f"{field} contains duplicate keys")
            result[key] = value
        return result

    try:
        value = json.loads(text, object_pairs_hook=pairs, parse_constant=lambda token: _fail(f"{field} has {token}"))
    except TrainingEntryError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise TrainingEntryError(f"{field} is invalid JSON") from exc
    _assert_builtin_json(value, field)
    if type(value) is not dict:
        _fail(f"{field} root must be a dict")
    return value


def _validate_config(config: Any) -> dict[str, Any]:
    _assert_builtin_json(config, "process config")
    top = _exact(
        config,
        {"schema_version", "process_contract_id", "entry_id", "launcher_id", "mode", "source_bindings", "invocation_policy", "entry_boundary", "process_boundary", "evidence_policy", "readiness"},
        "process config",
    )
    if _integer(top["schema_version"], "config.schema_version", minimum=1) != PROCESS_SCHEMA_VERSION:
        _fail("process config schema version drift")
    if _string(top["process_contract_id"], "config.process_contract_id") != PROCESS_CONTRACT_ID:
        _fail("process contract ID drift")
    if _string(top["entry_id"], "config.entry_id") != ENTRY_ID or _string(top["mode"], "config.mode") != ENTRY_MODE:
        _fail("entry config identity drift")
    if _string(top["launcher_id"], "config.launcher_id") != "rtdetrv2_r18_visdrone_training_process_launcher_t5d_v1":
        _fail("launcher config identity drift")

    source = _exact(top["source_bindings"], {"training_contract", "runtime_plan", "training_evidence", "prepared_adapter", "primary_evaluator", "vendor_runtime", "conversion_r3", "repository_modules"}, "config.source_bindings")
    for name, expected_id in (
        ("training_contract", "rtdetrv2_r18_visdrone_baseline_training_v1"),
        ("runtime_plan", "rtdetrv2_r18_visdrone_baseline_training_runtime_v1"),
        ("training_evidence", "rtdetrv2_r18_visdrone_baseline_training_evidence_v1"),
    ):
        item = source[name]
        if type(item) is not dict or item.get("id") != expected_id:
            _fail(f"config source binding {name} identity drift")
        _relative_path(item.get("relative_path"), f"config.source_bindings.{name}.relative_path")
        _sha_string(item.get("canonical_sha256", item.get("module_sha256")), f"config.source_bindings.{name}.digest")
    evaluator = _exact(source["primary_evaluator"], {"evaluator_id", "protocol_id", "implementation_commit", "implementation_tree", "config_canonical_sha256", "authority_manifest_canonical_sha256", "authority_inventory_sha256"}, "config.primary_evaluator")
    if evaluator["evaluator_id"] != "visdrone_official_primary_evaluator_v1" or evaluator["protocol_id"] != "visdrone_official_style_v1":
        _fail("config primary evaluator identity drift")
    for field in ("implementation_commit", "implementation_tree"):
        if type(evaluator[field]) is not str or len(evaluator[field]) != 40 or any(char not in "0123456789abcdef" for char in evaluator[field]):
            _fail(f"config primary evaluator {field} drift")
    for field in ("config_canonical_sha256", "authority_manifest_canonical_sha256", "authority_inventory_sha256"):
        _sha_string(evaluator[field], f"config.primary_evaluator.{field}")
    vendor_binding = _exact(source["vendor_runtime"], {"relative_path", "file_count", "directory_count_excluding_root", "total_size_bytes", "compact_inventory_sha256", "manifest_relative_path", "manifest_size_bytes", "manifest_raw_sha256", "manifest_inventory_sha256"}, "config.vendor_runtime")
    if vendor_binding["relative_path"] != "vendor/rtdetrv2_pytorch" or vendor_binding["manifest_relative_path"] != "manifests/rtdetrv2_upstream.json":
        _fail("config vendor path identity drift")
    if vendor_binding["file_count"] != 124 or vendor_binding["directory_count_excluding_root"] != 25 or vendor_binding["total_size_bytes"] != 373735 or vendor_binding["manifest_size_bytes"] != 33264:
        _fail("config vendor inventory counts drift")
    for field in ("compact_inventory_sha256", "manifest_raw_sha256", "manifest_inventory_sha256"):
        _sha_string(vendor_binding[field], f"config.vendor_runtime.{field}")
    conversion = _exact(source["conversion_r3"], {"relative_path", "entry_canonical_inventory_sha256", "artifact_inventory_sha256", "completion_sha256", "config_sha256", "category_contract_sha256", "source_identity_sha256"}, "config.conversion_r3")
    if conversion["relative_path"] != "artifacts/data/visdrone_protocol_v2_conversion_r3":
        _fail("config Conversion R3 path drift")
    for field in set(conversion) - {"relative_path"}:
        _sha_string(conversion[field], f"config.conversion_r3.{field}")
    modules = source["repository_modules"]
    if type(modules) is not list or len(modules) != len(SOURCE_MODULES):
        _fail("config repository module inventory length drift")
    expected_modules = [{"role": role, "relative_path": path} for role, path in SOURCE_MODULES]
    _strict_equal(modules, expected_modules, "config.repository_modules")

    invocation = _exact(top["invocation_policy"], {"mode", "formal_run_count", "repo_root_required", "working_directory_must_equal_repo_root", "python_executable_must_be_absolute", "argv_is_sequence", "argv_shell_evaluation", "environment_is_closed", "environment_keys", "cuda_visible_devices", "real_entry_execution_authorized", "real_process_launch_authorized", "production_training_authorized", "external_effects_authorized", "subprocess_authorized", "tmux_authorized", "network_authorized", "resume_allowed", "retry_allowed", "overwrite_allowed", "fallback_allowed"}, "config.invocation_policy")
    if invocation["mode"] != "synthetic" or invocation["formal_run_count"] != 1 or invocation["environment_keys"] != list(ENVIRONMENT_KEYS) or invocation["cuda_visible_devices"] != "":
        _fail("config invocation policy drift")
    for field in set(invocation) - {"mode", "formal_run_count", "environment_keys", "cuda_visible_devices"}:
        if type(invocation[field]) is not bool:
            _fail(f"config invocation policy {field} is not bool")
    for field in ("real_entry_execution_authorized", "real_process_launch_authorized", "production_training_authorized", "external_effects_authorized", "subprocess_authorized", "tmux_authorized", "network_authorized", "resume_allowed", "retry_allowed", "overwrite_allowed", "fallback_allowed"):
        if invocation[field] is not False:
            _fail(f"config invocation policy {field} is open")

    entry = _exact(top["entry_boundary"], {"entry_module_relative_path", "entry_callable", "entry_public_api", "required_descriptor_keys", "required_result_keys", "synthetic_injected_ports_only", "production_root_forbidden", "real_model_forbidden", "real_data_forbidden"}, "config.entry_boundary")
    if entry["entry_module_relative_path"] != ENTRY_MODULE_RELATIVE_PATH or entry["entry_callable"] != "run_synthetic_entry" or entry["entry_public_api"] != list(__all__):
        _fail("config entry public surface drift")
    required_descriptor_keys = {"schema_version", "process_contract_id", "entry_id", "mode", "run_id", "nonce", "repo_root", "working_directory", "python_executable", "module_identity", "argv", "environment", "authorities", "future_targets", "evidence_root", "receipt_path", "handoff", "source_bindings", "aggregate_sha256"}
    if type(entry["required_descriptor_keys"]) is not list or set(entry["required_descriptor_keys"]) != required_descriptor_keys:
        _fail("config entry descriptor key contract drift")
    required_result_keys = {"schema_version", "entry_id", "mode", "run_id", "nonce", "invocation_descriptor_sha256", "prepared_aggregate_sha256", "result", "production_training_authorized", "evidence_root", "receipt_path", "evidence_root_identity", "receipt_size_bytes", "receipt_sha256", "aggregate_result_sha256"}
    if type(entry["required_result_keys"]) is not list or set(entry["required_result_keys"]) != required_result_keys:
        _fail("config entry result key contract drift")
    for field in ("synthetic_injected_ports_only", "production_root_forbidden", "real_model_forbidden", "real_data_forbidden"):
        if entry[field] is not True:
            _fail(f"config entry boundary {field} is not enforced")

    process = _exact(top["process_boundary"], {"launcher_module_relative_path", "launcher_callable", "launcher_public_api", "launch_descriptor_keys", "launch_result_keys", "state_order", "failure_classes", "return_code_range", "evidence_file_names", "binary_anchor_files", "exactly_once", "fake_runner_signature", "runner_return_keys", "runner_must_be_synchronous", "real_subprocess_forbidden", "real_success_forbidden"}, "config.process_boundary")
    if process["launcher_module_relative_path"] != LAUNCHER_MODULE_RELATIVE_PATH or process["launcher_callable"] != "run_synthetic_launch":
        _fail("config launcher identity drift")
    if process["state_order"] != ["ABSENT", "PREPARED", "ACCEPTED", "SYNTHETIC_TERMINAL_COMPLETE", "TERMINAL_COMPLETE", "TERMINAL_FAILED", "UNKNOWN"] or process["fake_runner_signature"] != ["argv", "cwd", "environment", "entry_descriptor"] or process["runner_return_keys"] != ["return_code", "stdout", "stderr", "entry_result"]:
        _fail("config process protocol drift")
    if type(process["launch_descriptor_keys"]) is not list or set(process["launch_descriptor_keys"]) != {"schema_version", "process_contract_id", "launcher_id", "mode", "run_id", "nonce", "entry_descriptor", "entry_descriptor_sha256", "argv", "cwd", "environment", "config_identity", "source_bindings", "future_targets", "evidence_root", "receipt_path", "state_machine", "runner_contract", "aggregate_sha256"}:
        _fail("config launch descriptor key contract drift")
    if type(process["launch_result_keys"]) is not list or set(process["launch_result_keys"]) != {"schema_version", "process_contract_id", "launcher_id", "mode", "run_id", "nonce", "launch_descriptor_sha256", "entry_descriptor_sha256", "argv", "cwd", "environment", "config_identity", "source_bindings", "future_targets", "evidence_root", "receipt_path", "evidence_root_identity", "receipt_size_bytes", "receipt_sha256", "return_code", "stdout_size_bytes", "stdout_sha256", "stderr_size_bytes", "stderr_sha256", "stdout_anchor_relative_path", "stderr_anchor_relative_path", "entry_result", "entry_result_sha256", "status", "classification", "failure_class", "state_sequence", "state_transition_sha256", "production_training_authorized", "real_process_launch_authorized", "aggregate_result_sha256"}:
        _fail("config launch result key contract drift")
    if process["failure_classes"] != ["NONE", "RUNNER_NONZERO_RETURN_CODE", "RUNNER_EXCEPTION", "RUNNER_TIMEOUT", "RUNNER_ASYNC_OR_GENERATOR_RESULT", "RUNNER_SCHEMA_OR_ENTRY_FAILURE", "RUNNER_MUTATED_INVOCATION", "PERSISTENCE_FAILURE"] or process["return_code_range"] != [0, 255] or process["evidence_file_names"] != ["invocation.json", "result.json", "completion.json", "artifact_inventory.json", "stdout.bin", "stderr.bin"] or process["binary_anchor_files"] != ["stdout.bin", "stderr.bin"]:
        _fail("config process failure or evidence schema drift")
    for field in ("exactly_once", "runner_must_be_synchronous", "real_subprocess_forbidden", "real_success_forbidden"):
        if process[field] is not True:
            _fail(f"config process boundary {field} is not enforced")

    evidence = _exact(top["evidence_policy"], {"evidence_root", "process_evidence_root", "launch_evidence_root", "target_roots_are_future_only", "records", "exclusive_creation", "atomic_publication", "file_fsync", "directory_fsync", "readback", "inventory_sha256", "production_targets_must_remain_absent"}, "config.evidence_policy")
    if evidence["evidence_root"] != TRAINING_EVIDENCE_ROOT_RELATIVE_PATH or evidence["process_evidence_root"] != PROCESS_EVIDENCE_ROOT_RELATIVE_PATH or evidence["launch_evidence_root"] != LAUNCH_EVIDENCE_ROOT_RELATIVE_PATH or evidence["records"] != ["invocation.json", "result.json", "completion.json", "artifact_inventory.json", "stdout.bin", "stderr.bin"]:
        _fail("config evidence path or record drift")
    for field in ("target_roots_are_future_only", "exclusive_creation", "atomic_publication", "file_fsync", "directory_fsync", "readback", "inventory_sha256", "production_targets_must_remain_absent"):
        if evidence[field] is not True:
            _fail(f"config evidence policy {field} is not enforced")

    readiness = _exact(top["readiness"], {"production_training_authorized", "real_entry_execution_authorized", "real_process_launch_authorized", "training_implementation_ready", "training_ready", "model_selection_certified", "confirmatory_access", "test_access", "speed_measurement"}, "config.readiness")
    for field, value in readiness.items():
        if value is not False:
            _fail(f"config readiness {field} is open")
    return _copy(top)


def _load_process_config(repo_root: str | os.PathLike[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(_absolute_path(os.fspath(repo_root), "repo_root", require_directory=True))
    raw, metadata = _read_relative_file(root, PROCESS_CONFIG_RELATIVE_PATH, "process config")
    mode = stat.S_IMODE(metadata.st_mode)
    if mode not in {0o644, 0o664}:
        _fail("process config mode drift")
    if len(raw) != PROCESS_CONFIG_RAW_SIZE_BYTES or _sha(raw) != PROCESS_CONFIG_RAW_SHA256:
        _fail("process config raw identity drift")
    config = _validate_config(_parse_strict_json(raw, "process config"))
    canonical = _canonical(config)
    if len(canonical) != PROCESS_CONFIG_CANONICAL_SIZE_BYTES or _sha(canonical) != PROCESS_CONFIG_CANONICAL_SHA256:
        _fail("process config canonical identity drift")
    return config, {
        "relative_path": PROCESS_CONFIG_RELATIVE_PATH,
        "raw_size_bytes": len(raw),
        "raw_sha256": _sha(raw),
        "canonical_size_bytes": len(canonical),
        "canonical_sha256": _sha(canonical),
        "mode": mode,
    }


def _file_identity(root: Path, relative: str, role: str) -> dict[str, Any]:
    raw, metadata = _read_relative_file(root, relative, role)
    mode = stat.S_IMODE(metadata.st_mode)
    # Git records these repository files as 0644; the shared checkout can
    # expose the same blobs as 0664 under its group-writable umask.
    if mode not in {0o644, 0o664}:
        _fail(f"{role} mode drift")
    return {"role": role, "relative_path": relative, "size_bytes": len(raw), "sha256": _sha(raw), "mode": mode}


def _source_bindings(root: Path, config: dict[str, Any], config_identity: dict[str, Any]) -> dict[str, Any]:
    rows = [_file_identity(root, relative, role) for role, relative in SOURCE_MODULES]
    return {"process_config": _copy(config_identity), "repository_modules": rows}


def _validate_authority_identity(prepared: dict[str, Any], config: dict[str, Any]) -> None:
    authorities = _exact(prepared["authorities"], {"identity", "payloads"}, "prepared.authorities")
    identity = _exact(authorities["identity"], {"training_contract", "runtime_plan", "evidence_contract", "primary_evaluator", "vendor_runtime", "conversion_r3"}, "prepared.authorities.identity")
    source = config["source_bindings"]
    for identity_name, source_name in (
        ("training_contract", "training_contract"),
        ("runtime_plan", "runtime_plan"),
        ("evidence_contract", "training_evidence"),
    ):
        expected = source[source_name]
        observed = identity[identity_name]
        if observed["id"] != expected["id"] or observed["canonical_sha256"] != expected["canonical_sha256"]:
            _fail(f"authority identity drift: {identity_name}")
    primary = identity["primary_evaluator"]
    expected_primary = source["primary_evaluator"]
    if primary["evaluator_id"] != expected_primary["evaluator_id"] or primary["protocol_id"] != expected_primary["protocol_id"] or primary["implementation_commit"] != expected_primary["implementation_commit"] or primary["implementation_tree"] != expected_primary["implementation_tree"] or primary["certified"] is not True:
        _fail("primary evaluator authority identity drift")
    vendor_binding = identity["vendor_runtime"]
    expected_vendor = source["vendor_runtime"]
    for field in ("relative_path", "file_count", "directory_count_excluding_root", "total_size_bytes", "compact_inventory_sha256", "manifest_relative_path", "manifest_size_bytes", "manifest_raw_sha256", "manifest_inventory_sha256"):
        if vendor_binding[field] != expected_vendor[field]:
            _fail(f"vendor authority identity drift: {field}")
    conversion = identity["conversion_r3"]
    expected_conversion = source["conversion_r3"]
    for field in set(expected_conversion):
        if conversion[field] != expected_conversion[field]:
            _fail(f"Conversion R3 authority identity drift: {field}")


def _validate_descriptor_shape(value: Any) -> dict[str, Any]:
    _assert_builtin_json(value, "entry descriptor")
    descriptor = _exact(value, {"schema_version", "process_contract_id", "entry_id", "mode", "run_id", "nonce", "repo_root", "working_directory", "python_executable", "module_identity", "argv", "environment", "authorities", "future_targets", "evidence_root", "receipt_path", "handoff", "source_bindings", "aggregate_sha256"}, "entry descriptor")
    if _integer(descriptor["schema_version"], "descriptor.schema_version", minimum=1) != PROCESS_SCHEMA_VERSION or _string(descriptor["process_contract_id"], "descriptor.process_contract_id") != PROCESS_CONTRACT_ID or _string(descriptor["entry_id"], "descriptor.entry_id") != ENTRY_ID or _string(descriptor["mode"], "descriptor.mode") != ENTRY_MODE:
        _fail("entry descriptor identity drift")
    _token(descriptor["run_id"], "run_id")
    _nonce(descriptor["nonce"])
    root = _absolute_path(descriptor["repo_root"], "descriptor.repo_root", require_directory=True)
    if _absolute_path(descriptor["working_directory"], "descriptor.working_directory") != root:
        _fail("working directory is not the repository root")
    executable = _absolute_path(descriptor["python_executable"], "descriptor.python_executable")
    if str(Path(executable)) != executable:
        _fail("python executable path is not canonical")
    _validate_existing_executable(executable)
    evidence_root = _absolute_path(descriptor["evidence_root"], "descriptor.evidence_root")
    _reject_production_root(evidence_root, "descriptor.evidence_root")
    expected_receipt = str(Path(evidence_root).parent / _receipt_name(evidence_root, ENTRY_RECEIPT_SUFFIX))
    if descriptor["receipt_path"] != expected_receipt:
        _fail("entry receipt path drift")
    root_parent, root_name = _bound_target_parts(evidence_root, "descriptor.evidence_root")
    try:
        if root_name in {".", ".."}:
            _fail("descriptor.evidence_root final component drift")
        root_parent.revalidate()
    finally:
        root_parent.close()
    module = _exact(descriptor["module_identity"], {"relative_path", "size_bytes", "sha256", "mode"}, "descriptor.module_identity")
    if module["relative_path"] != ENTRY_MODULE_RELATIVE_PATH:
        _fail("entry module path drift")
    _integer(module["size_bytes"], "module_identity.size_bytes", minimum=1)
    _sha_string(module["sha256"], "module_identity.sha256")
    _integer(module["mode"], "module_identity.mode", minimum=1)
    argv = descriptor["argv"]
    if type(argv) is not list or not argv or any(type(item) is not str for item in argv):
        _fail("argv must be a builtin string list")
    if argv != [descriptor["python_executable"], "-m", "sparse_rtdetr.baseline.training_entry", "--mode", "synthetic", "--run-id", descriptor["run_id"], "--nonce", descriptor["nonce"]]:
        _fail("entry argv identity drift")
    environment = _exact(descriptor["environment"], set(ENVIRONMENT_KEYS), "descriptor.environment")
    expected_environment = {
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": f"{root}/src",
    }
    _strict_equal(environment, expected_environment, "descriptor.environment")
    future = _exact(descriptor["future_targets"], set(FUTURE_TARGETS), "descriptor.future_targets")
    _strict_equal(future, FUTURE_TARGETS, "descriptor.future_targets")
    handoff = _exact(descriptor["handoff"], {"from_stage", "status", "prepared_aggregate_sha256", "predecessor_sha256"}, "descriptor.handoff")
    if handoff["from_stage"] != "T5C_PREPARED_TRAINER_ADAPTER" or handoff["status"] != "PASS":
        _fail("T5C handoff identity drift")
    _sha_string(handoff["prepared_aggregate_sha256"], "handoff.prepared_aggregate_sha256")
    _sha_string(handoff["predecessor_sha256"], "handoff.predecessor_sha256")
    source = _exact(descriptor["source_bindings"], {"process_config", "repository_modules"}, "descriptor.source_bindings")
    config_identity = _exact(source["process_config"], {"relative_path", "raw_size_bytes", "raw_sha256", "canonical_size_bytes", "canonical_sha256", "mode"}, "descriptor.process_config")
    if config_identity["relative_path"] != PROCESS_CONFIG_RELATIVE_PATH:
        _fail("descriptor process config path drift")
    for field in ("raw_size_bytes", "canonical_size_bytes"):
        _integer(config_identity[field], f"descriptor.process_config.{field}", minimum=1)
    for field in ("raw_sha256", "canonical_sha256"):
        _sha_string(config_identity[field], f"descriptor.process_config.{field}")
    _integer(config_identity["mode"], "descriptor.process_config.mode", minimum=1)
    if config_identity["mode"] not in {0o644, 0o664}:
        _fail("descriptor process config mode drift")
    modules = source["repository_modules"]
    if type(modules) is not list or len(modules) != len(SOURCE_MODULES):
        _fail("descriptor module inventory length drift")
    for index, row in enumerate(modules):
        row = _exact(row, {"role", "relative_path", "size_bytes", "sha256", "mode"}, f"descriptor.repository_modules[{index}]")
        _string(row["role"], f"module[{index}].role")
        _relative_path(row["relative_path"], f"module[{index}].relative_path")
        _integer(row["size_bytes"], f"module[{index}].size_bytes", minimum=1)
        _sha_string(row["sha256"], f"module[{index}].sha256")
        _integer(row["mode"], f"module[{index}].mode", minimum=1)
    authorities = _exact(descriptor["authorities"], {"identity", "prepared"}, "descriptor.authorities")
    _assert_builtin_json(authorities["identity"], "descriptor.authorities.identity")
    prepared = _adapter.validate_prepared_training_adapter(authorities["prepared"])
    if prepared["aggregate_sha256"] != handoff["prepared_aggregate_sha256"]:
        _fail("prepared authority handoff digest drift")
    if _digest(authorities["identity"]) != _digest(prepared["authorities"]["identity"]):
        _fail("prepared authority identity was repacked")
    aggregate = _sha_string(descriptor["aggregate_sha256"], "descriptor.aggregate_sha256")
    body = _copy(descriptor)
    del body["aggregate_sha256"]
    if _digest(body) != aggregate:
        _fail("entry descriptor aggregate identity drift")
    return _copy(descriptor)


def _validate_repository_identity(descriptor: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(descriptor["repo_root"])
    config, config_identity = _load_process_config(root)
    _validate_authority_identity(descriptor["authorities"]["prepared"], config)
    _strict_equal(descriptor["source_bindings"]["process_config"], config_identity, "descriptor process config identity")
    actual_module = _file_identity(root, ENTRY_MODULE_RELATIVE_PATH, "entry module")
    if actual_module["size_bytes"] != descriptor["module_identity"]["size_bytes"] or actual_module["sha256"] != descriptor["module_identity"]["sha256"] or actual_module["mode"] != descriptor["module_identity"]["mode"]:
        _fail("entry module source identity drift")
    expected_rows = _source_bindings(root, config, config_identity)["repository_modules"]
    _strict_equal(descriptor["source_bindings"]["repository_modules"], expected_rows, "descriptor repository module identity")
    return config, config_identity


def build_synthetic_entry_descriptor(
    repo_root: str | os.PathLike[str],
    *,
    run_id: str,
    nonce: str,
    evidence_root: str | os.PathLike[str] | None = None,
    python_executable: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Build one detached synthetic invocation descriptor without launching."""

    root = _absolute_path(os.fspath(repo_root), "repo_root", require_directory=True)
    _token(run_id, "run_id")
    _nonce(nonce)
    executable_value = os.fspath(sys.executable if python_executable is None else python_executable)
    if python_executable is None:
        executable_value = str(Path(executable_value).resolve(strict=True))
    executable = _validate_existing_executable(executable_value)
    if evidence_root is None:
        evidence_root = Path(root).parent / f".t5d-entry-{run_id}"
    bound_evidence_root = _absolute_path(os.fspath(evidence_root), "evidence_root")
    receipt_path = _validate_bound_root_shape(bound_evidence_root, "evidence_root", ENTRY_RECEIPT_SUFFIX, require_absent=True)
    config, config_identity = _load_process_config(root)
    try:
        prepared = _adapter.prepare_training_adapter(root, run_id=run_id, nonce=nonce, mode=ENTRY_MODE)
    except Exception as exc:
        raise TrainingEntryError("certified T5C prepared adapter could not be bound") from exc
    prepared = _adapter.validate_prepared_training_adapter(prepared)
    _validate_authority_identity({"authorities": prepared["authorities"]}, config)
    module_identity = _file_identity(Path(root), ENTRY_MODULE_RELATIVE_PATH, "entry module")
    module_identity = {key: module_identity[key] for key in ("relative_path", "size_bytes", "sha256", "mode")}
    source = _source_bindings(Path(root), config, config_identity)
    source["repository_modules"] = [
        {key: row[key] for key in ("role", "relative_path", "size_bytes", "sha256", "mode")}
        for row in source["repository_modules"]
    ]
    handoff = {
        "from_stage": "T5C_PREPARED_TRAINER_ADAPTER",
        "status": "PASS",
        "prepared_aggregate_sha256": prepared["aggregate_sha256"],
        "predecessor_sha256": _digest({"stage": "T5C", "aggregate_sha256": prepared["aggregate_sha256"]}),
    }
    body = {
        "schema_version": PROCESS_SCHEMA_VERSION,
        "process_contract_id": PROCESS_CONTRACT_ID,
        "entry_id": ENTRY_ID,
        "mode": ENTRY_MODE,
        "run_id": run_id,
        "nonce": nonce,
        "repo_root": root,
        "working_directory": root,
        "python_executable": executable,
        "module_identity": module_identity,
        "argv": [executable, "-m", "sparse_rtdetr.baseline.training_entry", "--mode", "synthetic", "--run-id", run_id, "--nonce", nonce],
        "environment": {
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": f"{root}/src",
        },
        "authorities": {"identity": _copy(prepared["authorities"]["identity"]), "prepared": _copy(prepared)},
        "future_targets": _copy(FUTURE_TARGETS),
        "evidence_root": bound_evidence_root,
        "receipt_path": receipt_path,
        "handoff": handoff,
        "source_bindings": source,
    }
    descriptor = {**body, "aggregate_sha256": _digest(body)}
    return _validate_descriptor_shape(descriptor)


def validate_entry_descriptor(value: Any) -> dict[str, Any]:
    """Validate and detach an entry descriptor without launching it."""

    descriptor = _validate_descriptor_shape(value)
    _validate_repository_identity(descriptor)
    return _copy(descriptor)


def _entry_root_stat(path: str, expected: dict[str, int]) -> os.stat_result:
    parent, name = _bound_target_parts(path, "entry evidence root")
    fd: int | None = None
    try:
        fd = _open_fd(name, _directory_flags(), dir_fd=parent.fd)
        observed = _fstat(fd)
        if not stat.S_ISDIR(observed.st_mode) or observed.st_nlink < 2:
            _fail("entry evidence root is not a stable directory")
        if observed.st_uid != os.getuid() or observed.st_gid != os.getgid() or stat.S_IMODE(observed.st_mode) != 0o700:
            _fail("entry evidence root metadata drift")
        if observed.st_dev != parent.snapshot.st_dev:
            _fail("entry evidence root device drift")
        if _root_identity(observed) != expected:
            _fail("entry evidence root identity drift")
        current = _stat_at(parent.fd, name)
        if current is None or not _same_file_identity(observed, current):
            _fail("entry evidence root path identity drift")
        parent.revalidate()
        return observed
    except OSError as exc:
        raise TrainingEntryError("entry evidence root could not be verified") from exc
    finally:
        if fd is not None:
            _close_fd(fd)
        parent.close()


def _result_receipt_payload(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": PROCESS_SCHEMA_VERSION,
        "kind": "T5D_ENTRY_CLAIM",
        "descriptor_sha256": result["invocation_descriptor_sha256"],
        "run_id": result["run_id"],
        "nonce": result["nonce"],
        "evidence_root": result["evidence_root"],
        "receipt_path": result["receipt_path"],
        "status": "CLAIMED",
    }


def _validate_result_receipt(result: dict[str, Any]) -> os.stat_result:
    root = _absolute_path(result["evidence_root"], "result.evidence_root")
    _reject_production_root(root, "result.evidence_root")
    expected_path = str(Path(root).parent / _receipt_name(root, ENTRY_RECEIPT_SUFFIX))
    if result["receipt_path"] != expected_path:
        _fail("result receipt path drift")
    parent, receipt_name = _bound_target_parts(result["receipt_path"], "result.receipt_path")
    fd: int | None = None
    try:
        fd = _open_fd(receipt_name, _file_flags(), dir_fd=parent.fd)
        observed = _fstat(fd)
        _require_regular_file(observed, "entry receipt")
        if stat.S_IMODE(observed.st_mode) != 0o600:
            _fail("entry receipt mode drift")
        while True:
            try:
                os.lseek(fd, 0, os.SEEK_SET)
                break
            except InterruptedError:
                continue
        expected = _canonical(_result_receipt_payload(result))
        actual = _read_fd_exact(fd, observed.st_size, "entry receipt")
        if actual != expected or observed.st_size != len(expected):
            _fail("entry receipt bytes drift")
        after = _fstat(fd)
        if not _same_file_identity(observed, after):
            _fail("entry receipt metadata drift")
        if result["receipt_size_bytes"] != len(actual) or result["receipt_sha256"] != _sha(actual):
            _fail("entry receipt digest drift")
        parent.revalidate()
        return after
    except OSError as exc:
        raise TrainingEntryError("entry receipt could not be verified") from exc
    finally:
        if fd is not None:
            _close_fd(fd)
        parent.close()


def validate_entry_result(value: Any, expected_descriptor: Any | None = None) -> dict[str, Any]:
    """Validate a detached result and, when supplied, its exact invocation."""

    _assert_builtin_json(value, "entry result")
    result = _exact(value, {"schema_version", "entry_id", "mode", "run_id", "nonce", "invocation_descriptor_sha256", "prepared_aggregate_sha256", "result", "production_training_authorized", "evidence_root", "receipt_path", "evidence_root_identity", "receipt_size_bytes", "receipt_sha256", "aggregate_result_sha256"}, "entry result")
    if _integer(result["schema_version"], "result.schema_version", minimum=1) != PROCESS_SCHEMA_VERSION or result["entry_id"] != ENTRY_ID or result["mode"] != ENTRY_MODE:
        _fail("entry result identity drift")
    _token(result["run_id"], "result.run_id")
    _nonce(result["nonce"])
    _sha_string(result["invocation_descriptor_sha256"], "result.invocation_descriptor_sha256")
    _sha_string(result["prepared_aggregate_sha256"], "result.prepared_aggregate_sha256")
    if type(result["production_training_authorized"]) is not bool or result["production_training_authorized"] is not False:
        _fail("entry result authorizes production training")
    _absolute_path(result["evidence_root"], "result.evidence_root")
    _absolute_path(result["receipt_path"], "result.receipt_path")
    root_identity = _exact(result["evidence_root_identity"], {"device", "inode", "mode", "uid", "gid", "nlink", "size_bytes", "mtime_ns", "ctime_ns"}, "result.evidence_root_identity")
    for field in root_identity:
        _integer(root_identity[field], f"result.evidence_root_identity.{field}", minimum=0)
    _integer(result["receipt_size_bytes"], "result.receipt_size_bytes", minimum=1)
    _sha_string(result["receipt_sha256"], "result.receipt_sha256")
    adapter_result = result["result"]
    _assert_builtin_json(adapter_result, "entry adapter result")
    if type(adapter_result) is not dict:
        _fail("T5C result identity drift")
    try:
        adapter_result = _adapter._validate_result(adapter_result)
    except Exception as exc:
        raise TrainingEntryError("T5C result identity drift") from exc
    if adapter_result["mode"] != ENTRY_MODE or adapter_result["run_id"] != result["run_id"] or adapter_result["nonce"] != result["nonce"] or adapter_result["prepared_aggregate_sha256"] != result["prepared_aggregate_sha256"] or adapter_result["production_training_authorized"] is not False:
        _fail("T5C result binding drift")
    if expected_descriptor is not None:
        expected = validate_entry_descriptor(expected_descriptor)
        for field in ("run_id", "nonce", "mode", "evidence_root", "receipt_path"):
            if result[field] != expected[field]:
                _fail(f"entry result {field} drift")
        if result["invocation_descriptor_sha256"] != expected["aggregate_sha256"] or result["prepared_aggregate_sha256"] != expected["handoff"]["prepared_aggregate_sha256"]:
            _fail("entry result descriptor binding drift")
    _validate_result_receipt(result)
    _entry_root_stat(result["evidence_root"], root_identity)
    aggregate = _sha_string(result["aggregate_result_sha256"], "result.aggregate_result_sha256")
    body = _copy(result)
    del body["aggregate_result_sha256"]
    if _digest(body) != aggregate:
        _fail("entry result aggregate identity drift")
    return _copy(result)


def run_synthetic_entry(
    descriptor: Any,
    components: Mapping[str, Any],
    evidence_root: str | os.PathLike[str],
) -> dict[str, Any]:
    """Execute exactly one injected CPU synthetic entry transaction."""

    try:
        validated = validate_entry_descriptor(descriptor)
        root = _absolute_path(os.fspath(evidence_root), "entry evidence root")
        if root != validated["evidence_root"]:
            _fail("entry evidence root is not the descriptor-bound root")
        if type(components) is not dict:
            _fail("components must be a builtin dict")
        parent, receipt_name, _ = _claim_entry_receipt(validated)
        root_fd: int | None = None
        try:
            root_path = Path(root)
            root_fd, root_stat = _create_and_open_root(parent, root_path.name, "entry evidence root")
            receipt_stat = _verify_entry_receipt(parent, validated, receipt_name)
            prepared = _copy(validated["authorities"]["prepared"])
            detached_components = dict(components)
            adapter_root = root_path / "adapter-evidence"
            try:
                observed = _adapter.run_synthetic_prepared_batch(prepared, detached_components, adapter_root)
            except Exception as exc:
                raise TrainingEntryError("synthetic T5C entry transaction failed") from exc
            parent.revalidate()
            after_root = _fstat(root_fd)
            if not stat.S_ISDIR(after_root.st_mode) or after_root.st_dev != parent.snapshot.st_dev:
                _fail("entry evidence root identity drift")
            if _root_operation_identity(_root_identity(after_root)) != _root_operation_identity(_root_identity(root_stat)):
                _fail("entry evidence root identity drift")
            named_root = _stat_at(parent.fd, root_path.name)
            if named_root is None or not _same_file_identity(named_root, after_root):
                _fail("entry evidence root name binding drift")
            observed = _adapter._validate_result(observed)
            body = {
                "schema_version": PROCESS_SCHEMA_VERSION,
                "entry_id": ENTRY_ID,
                "mode": ENTRY_MODE,
                "run_id": validated["run_id"],
                "nonce": validated["nonce"],
                "invocation_descriptor_sha256": validated["aggregate_sha256"],
                "prepared_aggregate_sha256": prepared["aggregate_sha256"],
                "result": _copy(observed),
                "production_training_authorized": False,
                "evidence_root": validated["evidence_root"],
                "receipt_path": validated["receipt_path"],
                "evidence_root_identity": _root_identity(after_root),
                "receipt_size_bytes": receipt_stat.st_size,
                "receipt_sha256": _sha(_canonical(_entry_receipt_payload(validated))),
            }
            result = {**body, "aggregate_result_sha256": _digest(body)}
            return validate_entry_result(result, expected_descriptor=validated)
        finally:
            if root_fd is not None:
                _close_fd(root_fd)
            parent.close()
    except TrainingEntryError:
        raise
    except Exception as exc:
        raise TrainingEntryError("synthetic entry boundary failed closed") from exc
