"""Pure CPU implementation of the frozen VisDrone official-style evaluator.

The public evaluation path is data-free: it consumes immutable schema objects
and returns detached evidence.  Contract loading is deliberately separate and
is the only part that reads the repository's committed JSON identities.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import errno
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from ..data_protocol.evaluation import (
    Detection,
    PrimaryEvaluatorInputV2,
    PrimaryEvaluatorResultV2,
    PrimaryGroundTruth,
    PrimaryImageV2,
)
from ..data_protocol.schema import ProtocolContractError, canonical_json_bytes


class PrimaryEvaluatorContractError(ProtocolContractError):
    """Raised when evaluator input, contract, or result evidence drifts."""


PRIMARY_EVALUATOR_ID = "visdrone_official_primary_evaluator_v1"
PRIMARY_PROTOCOL_ID = "visdrone_official_style_v1"
PRIMARY_SCHEMA_ID = "primary_evaluator_input_v2"
PRIMARY_RESULT_SCHEMA_ID = "primary_evaluator_result_v2"
PRIMARY_CONFIG_RELATIVE_PATH = "configs/baseline/visdrone_official_evaluator_v1.json"
PRIMARY_MANIFEST_RELATIVE_PATH = "manifests/visdrone_det_toolkit_005445.json"
AUTHORITY_REPOSITORY_URL = "https://github.com/VisDrone/VisDrone2018-DET-toolkit.git"
AUTHORITY_BRANCH = "master"
AUTHORITY_COMMIT = "005445782213e20cb91bc50a597db3dd949e749a"
AUTHORITY_TREE = "038b9e68c6e9a93a64662a4d7a39be2cd2c0654e"
AUTHORITY_MANIFEST_RAW_SIZE = 4166
AUTHORITY_MANIFEST_RAW_SHA256 = "71168baf15d6d945fd5a4a6c5605b5ba533524efeede2a9d4020127576c1b36f"
AUTHORITY_MANIFEST_CANONICAL_SIZE = 3351
AUTHORITY_MANIFEST_CANONICAL_SHA256 = "5bad9faf7622fe4542aa3b46561d6d41fb2e4ee34f35551ecfd821c9577159e3"
AUTHORITY_INVENTORY_SHA256 = "35a14a021509b82f1238912e5c77ebb3559f9ee6daa3cc6c92db810b5dce5da0"
AUTHORITY_FILE_SHA256 = {
    "utils/saveAnnoRes.m": "3210fb8fd98aed19cab61c996e0daf1dbfdb23fd5eb2fafff29afc71bb45876d",
    "utils/dropObjectsInIgr.m": "30dee2713d76a537f5c98ec5d0804e17cfd3993d83d8a80a70b90ef9461fa4de",
    "utils/createIntImg.m": "7ce3de4fc105be4f088cae5d1fcfc8f4383f3d7ffd4a551bf1afe711eff09f72",
    "utils/compOas.m": "3e5c2d473c07bf2ddbe0d902284af98ef7104c609e867b2ed59083d31f3ed2af",
    "utils/evalRes.m": "610f0d078f1af8d7987e360e6e88665c0290587191d5542b469a9792402f539b",
    "utils/VOCap.m": "95dd1e02c956124e777caf6f44b50b0986a2a37785bd11982ef775e696c17945",
    "utils/calcAccuracy.m": "285508f54903acd75eeda346c11c8e53267e72facd648f2e22bce121d4f55d7f",
}
IOU_THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
MAX_DETS = (1, 10, 100, 500)
METRIC_NAMES = ("AP", "AP50", "AP75", "AR1", "AR10", "AR100", "AR500")
_SHA256_RE = set("0123456789abcdef")
_GIT_OID_RE = set("0123456789abcdef")

# These are filled from the committed canonical config identity.  Keeping the
# whole-file identity in code avoids an impossible self-referential JSON hash.
PRIMARY_CONFIG_RAW_SIZE = 3857
PRIMARY_CONFIG_RAW_SHA256 = "36cfa69b0ff645c47c871283f917577ca27c760daf424e9544ab363dbf080ff5"
PRIMARY_CONFIG_CANONICAL_SIZE = 3295
PRIMARY_CONFIG_CANONICAL_SHA256 = "355de90bdb6007ed42ed65b3f653a1b8ea57f2184ab4fd48a9561b692b993998"

_MANIFEST_KEYS = {
    "schema_version", "authority_id", "repository_url", "branch", "commit_oid",
    "tree_oid", "toolkit_version", "algorithm_semantics_version",
    "license_and_use_notice", "archive", "inventory",
}
_MANIFEST_ARCHIVE_KEYS = {"filename", "prefix", "size_bytes", "sha256"}
_MANIFEST_INVENTORY_KEYS = {"file_count", "total_size_bytes", "canonical_inventory_sha256", "canonicalization", "rows"}
_MANIFEST_ROW_KEYS = {"relative_path", "git_mode", "git_blob_oid", "size_bytes", "sha256"}
_CONFIG_KEYS = {"schema_version", "evaluator_id", "protocol_id", "schema_id", "authority", "input_contract", "algorithm", "output_contract", "policy"}
_AUTHORITY_KEYS = {"repository_url", "branch", "commit_oid", "tree_oid", "manifest_relative_path", "manifest_raw_size_bytes", "manifest_raw_sha256", "manifest_canonical_size_bytes", "manifest_canonical_sha256", "inventory_canonical_sha256", "authority_file_sha256"}
_INPUT_KEYS = {"coordinate_system", "box_conversion", "category_domain", "image_id_order", "detection_order", "max_dets_prefix", "width_height_required", "input_immutability"}
_INPUT_CATEGORY_KEYS = {"detections", "ground_truth", "scored", "spatial_ignore", "others"}
_ALGORITHM_KEYS = {"iou_thresholds", "max_dets", "nms", "ignore_threshold", "ignore_fraction_rule", "standard_overlap", "ignored_overlap", "rounding", "clipping", "matching", "threshold_comparison", "recall_denominator", "ap_source", "voc_ap", "aggregation", "metrics_units"}
_OUTPUT_KEYS = {"metric_names", "evidence", "ap_small_emitted", "result_immutable_detached", "float_dtype"}
_POLICY_KEYS = {"development_only", "confirmatory_access_allowed", "test_access_allowed", "implementation_present", "independent_audit_pass", "training_gate_open", "secondary_evaluator_can_certify", "secondary_evaluator_can_select"}
_BINDING_KEYS = {
    "config", "authority_manifest", "config_raw_size_bytes", "config_raw_sha256",
    "config_canonical_size_bytes", "config_canonical_sha256", "authority_manifest_raw_size_bytes",
    "authority_manifest_raw_sha256", "authority_manifest_canonical_size_bytes",
    "authority_manifest_canonical_sha256",
}


def _fail(message: str) -> None:
    raise PrimaryEvaluatorContractError(message)


def _exact_keys(value: object, expected: set[str], path: str) -> dict[str, Any]:
    if type(value) is not dict:
        _fail(f"{path} must be a builtin object")
    if set(value) != expected:
        _fail(f"{path} key set drift")
    return value


def _exact_int(value: object, path: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        _fail(f"{path} must be a builtin integer")
    if minimum is not None and value < minimum:
        _fail(f"{path} is below its lower bound")
    return value


def _exact_bool(value: object, path: str) -> bool:
    if type(value) is not bool:
        _fail(f"{path} must be a builtin boolean")
    return value


def _exact_str(value: object, path: str, *, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value):
        _fail(f"{path} must be a builtin non-empty string")
    return value


def _exact_float(value: object, path: str, *, minimum: float | None = None) -> float:
    if type(value) not in (int, float) or type(value) is bool:
        _fail(f"{path} must be a builtin number")
    number = float(value)
    if not math.isfinite(number):
        _fail(f"{path} must be finite")
    if minimum is not None and number < minimum:
        _fail(f"{path} is below its lower bound")
    return number


def _validate_builtin_json(value: object, path: str) -> None:
    """Reject JSON-equivalent Python subclasses before semantic validation."""

    value_type = type(value)
    if value_type is dict:
        for key, item in value.items():
            if type(key) is not str:
                _fail(f"{path} contains a non-builtin object key")
            _validate_builtin_json(item, f"{path}/{key}")
        return
    if value_type is list:
        for index, item in enumerate(value):
            _validate_builtin_json(item, f"{path}/{index}")
        return
    if value_type is str or value_type is bool or value_type is int or value_type is type(None):
        return
    if value_type is float:
        if not math.isfinite(value):
            _fail(f"{path} must be finite")
        return
    _fail(f"{path} contains a non-builtin JSON value")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha(value: object) -> bool:
    return type(value) is str and len(value) == 64 and set(value) <= _SHA256_RE


def _is_git_oid(value: object) -> bool:
    return type(value) is str and len(value) == 40 and set(value) <= _GIT_OID_RE


def _exact_sha(value: object, path: str) -> str:
    if not _is_sha(value):
        _fail(f"{path} must be a lowercase SHA-256")
    return value


def _exact_git_oid(value: object, path: str) -> str:
    if not _is_git_oid(value):
        _fail(f"{path} must be a lowercase Git blob OID")
    return value


def _strict_json_bytes(raw: bytes, label: str) -> tuple[dict[str, Any], bytes]:
    if raw.startswith(b"\xef\xbb\xbf") or b"\x00" in raw or b"\r" in raw:
        _fail(f"{label} contains forbidden BOM, NUL, or CR")

    def duplicate_guard(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                _fail(f"{label} contains duplicate JSON keys")
            result[key] = item
        return result

    def reject_constant(value: str) -> None:
        _fail(f"{label} contains non-finite JSON constant {value}")

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=duplicate_guard,
            parse_constant=reject_constant,
        )
    except PrimaryEvaluatorContractError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        _fail(f"{label} JSON parse failure: {type(exc).__name__}")
    if type(value) is not dict:
        _fail(f"{label} must be a JSON object")
    return value, canonical_json_bytes(value)


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _repo_root_path(repo_root: Path | str) -> tuple[Path, tuple[str, ...]]:
    try:
        raw = os.fspath(repo_root)
    except (TypeError, ValueError) as exc:
        _fail(f"repository root is invalid: {type(exc).__name__}")
    if type(raw) is not str or not raw:
        _fail("repository root must be a non-empty builtin string path")
    if "\x00" in raw or "\\" in raw:
        _fail("repository root contains an invalid separator or NUL")
    if not raw.startswith("/"):
        _fail("repository root must be absolute")
    if raw == "/":
        return Path(raw), ()
    if raw.startswith("//") or raw.endswith("/") or "//" in raw:
        _fail("repository root is not a canonical absolute path")
    components = tuple(raw[1:].split("/"))
    if any(not component or component in (".", "..") for component in components):
        _fail("repository root contains an invalid lexical component")
    if os.path.normpath(raw) != raw:
        _fail("repository root is not a canonical absolute path")
    return Path(raw), components


def _relative_components(relative_path: str) -> tuple[str, ...]:
    if type(relative_path) is not str or not relative_path:
        _fail("repository relative path must be a non-empty builtin string")
    if "\x00" in relative_path or "\\" in relative_path:
        _fail("repository relative path contains an invalid separator")
    raw_parts = tuple(relative_path.split("/"))
    if any(part in ("", ".", "..") for part in raw_parts):
        _fail("repository relative path is not lexically contained")
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or pure.parts != raw_parts:
        _fail("repository relative path is not a contained POSIX path")
    return raw_parts


def _open_directory(path: str | bytes, *, dir_fd: int | None = None) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        return os.open(path, flags, dir_fd=dir_fd)
    except OSError as exc:
        _fail(f"repository directory open failure: {type(exc).__name__}")


def _read_fd_complete(fd: int, expected_size: int, label: str) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        try:
            chunk = os.read(fd, 65536)
        except OSError as exc:
            if exc.errno == errno.EINTR:
                continue
            _fail(f"{label} read failure: {type(exc).__name__}")
        if chunk == b"":
            if total != expected_size:
                _fail(f"{label} premature EOF")
            break
        chunks.append(chunk)
        total += len(chunk)
    return b"".join(chunks)


@dataclass(frozen=True)
class _DescriptorComponent:
    parent_fd: int
    name: str
    child_fd: int
    identity: tuple[int, int, int, int, int, int, int, int, int]


@dataclass(frozen=True)
class _ReadEvidence:
    directories: tuple[_DescriptorComponent, ...]
    parent_fd: int
    final_name: str
    file_fd: int
    identity: tuple[int, int, int, int, int, int, int, int, int]


class _RepositoryBoundary:
    """A stable descriptor-rooted reader for committed evaluator JSON."""

    def __init__(self, repo_root: Path | str):
        self.root, root_components = _repo_root_path(repo_root)
        self._owned_fds: list[int] = []
        self._closed = False
        self._root_chain: tuple[_DescriptorComponent, ...] = ()
        self._reads: list[_ReadEvidence] = []
        try:
            self._open_root_chain(root_components)
        except BaseException:
            self._close_owned()
            raise

    def __enter__(self) -> "_RepositoryBoundary":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        close_error = self._close_owned()
        if exc_type is None and close_error is not None:
            raise close_error

    def _close_owned(self) -> OSError | None:
        if self._closed:
            return None
        self._closed = True
        first_error: OSError | None = None
        for fd in reversed(self._owned_fds):
            try:
                os.close(fd)
            except OSError as exc:
                if first_error is None:
                    first_error = exc
        return first_error

    def _open_root_chain(self, components: tuple[str, ...]) -> None:
        current_fd = _open_directory("/")
        self._owned_fds.append(current_fd)
        evidence: list[_DescriptorComponent] = []
        for component in components:
            try:
                before = os.stat(component, dir_fd=current_fd, follow_symlinks=False)
            except OSError as exc:
                _fail(f"repository root component stat failure: {type(exc).__name__}")
            if not stat.S_ISDIR(before.st_mode) or stat.S_ISLNK(before.st_mode):
                _fail("repository root component must be a real directory")
            next_fd = _open_directory(component, dir_fd=current_fd)
            self._owned_fds.append(next_fd)
            try:
                after = os.fstat(next_fd)
            except OSError as exc:
                _fail(f"repository root component fstat failure: {type(exc).__name__}")
            if _stat_identity(after) != _stat_identity(before):
                _fail("repository root component identity changed during open")
            evidence.append(_DescriptorComponent(current_fd, component, next_fd, _stat_identity(before)))
            current_fd = next_fd
        self._root_chain = tuple(evidence)
        self._root_fd = current_fd
        try:
            self._root_fstat = os.fstat(self._root_fd)
        except OSError as exc:
            _fail(f"repository root final fstat failure: {type(exc).__name__}")
        if not stat.S_ISDIR(self._root_fstat.st_mode) or stat.S_ISLNK(self._root_fstat.st_mode):
            _fail("repository root must be a real directory")
        self._root_device = self._root_fstat.st_dev

    def assert_stable(self) -> None:
        if self._closed:
            _fail("repository boundary is already closed")
        for component in self._root_chain:
            try:
                current_path = os.stat(component.name, dir_fd=component.parent_fd, follow_symlinks=False)
                current_fd = os.fstat(component.child_fd)
            except OSError as exc:
                _fail(f"repository root terminal revalidation failure: {type(exc).__name__}")
            if _stat_identity(current_path) != component.identity or _stat_identity(current_fd) != component.identity:
                _fail("repository root descriptor path identity changed")
        try:
            current = os.fstat(self._root_fd)
        except OSError as exc:
            _fail(f"repository root final stat failure: {type(exc).__name__}")
        if _stat_identity(current) != _stat_identity(self._root_fstat):
            _fail("repository root identity changed during transaction")
        for read in self._reads:
            for component in read.directories:
                try:
                    current_path = os.stat(component.name, dir_fd=component.parent_fd, follow_symlinks=False)
                    current_fd = os.fstat(component.child_fd)
                except OSError as exc:
                    _fail(f"repository descendant terminal revalidation failure: {type(exc).__name__}")
                if _stat_identity(current_path) != component.identity or _stat_identity(current_fd) != component.identity:
                    _fail("repository descendant descriptor path identity changed")
            try:
                current_path = os.stat(read.final_name, dir_fd=read.parent_fd, follow_symlinks=False)
                current_fd = os.fstat(read.file_fd)
            except OSError as exc:
                _fail(f"repository file terminal revalidation failure: {type(exc).__name__}")
            if _stat_identity(current_path) != read.identity or _stat_identity(current_fd) != read.identity:
                _fail("repository file descriptor path identity changed")

    def _open_parent(self, components: tuple[str, ...]) -> tuple[int, str, tuple[_DescriptorComponent, ...], list[int]]:
        if not components:
            _fail("repository relative path is empty")
        current_fd = self._root_fd
        opened_fds: list[int] = []
        evidence: list[_DescriptorComponent] = []
        try:
            for component in components[:-1]:
                try:
                    before = os.stat(component, dir_fd=current_fd, follow_symlinks=False)
                except OSError as exc:
                    _fail(f"repository path component stat failure: {type(exc).__name__}")
                if not stat.S_ISDIR(before.st_mode) or stat.S_ISLNK(before.st_mode):
                    _fail("repository path component is not a real directory")
                if before.st_dev != self._root_device:
                    _fail("repository path component crossed devices")
                next_fd = _open_directory(component, dir_fd=current_fd)
                opened_fds.append(next_fd)
                try:
                    after = os.fstat(next_fd)
                    if _stat_identity(after) != _stat_identity(before):
                        _fail("repository path component identity changed during open")
                except BaseException:
                    raise
                evidence.append(_DescriptorComponent(current_fd, component, next_fd, _stat_identity(before)))
                current_fd = next_fd
            return current_fd, components[-1], tuple(evidence), opened_fds
        except BaseException:
            for fd in reversed(opened_fds):
                try:
                    os.close(fd)
                except OSError:
                    pass
            raise

    def read_bytes(self, relative_path: str, label: str) -> bytes:
        components = _relative_components(relative_path)
        parent_fd, final_name, directories, opened_fds = self._open_parent(components)
        committed = False
        try:
            try:
                before = os.stat(final_name, dir_fd=parent_fd, follow_symlinks=False)
            except OSError as exc:
                _fail(f"{label} stat failure: {type(exc).__name__}")
            if (
                not stat.S_ISREG(before.st_mode)
                or stat.S_ISLNK(before.st_mode)
                or before.st_nlink != 1
                or before.st_dev != self._root_device
                or before.st_uid != self._root_fstat.st_uid
                or before.st_gid != self._root_fstat.st_gid
            ):
                _fail(f"{label} must be a regular owned repository file")
            flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
            try:
                file_fd = os.open(final_name, flags, dir_fd=parent_fd)
            except OSError as exc:
                _fail(f"{label} open failure: {type(exc).__name__}")
            opened_fds.append(file_fd)
            opened = os.fstat(file_fd)
            if _stat_identity(opened) != _stat_identity(before):
                _fail(f"{label} identity changed during open")
            raw = _read_fd_complete(file_fd, before.st_size, label)
            after = os.fstat(file_fd)
            try:
                path_after = os.stat(final_name, dir_fd=parent_fd, follow_symlinks=False)
            except OSError as exc:
                _fail(f"{label} final path stat failure: {type(exc).__name__}")
            if _stat_identity(after) != _stat_identity(before) or _stat_identity(path_after) != _stat_identity(before):
                _fail(f"{label} identity changed during read")
            self._reads.append(_ReadEvidence(directories, parent_fd, final_name, file_fd, _stat_identity(before)))
            self._owned_fds.extend(opened_fds)
            committed = True
            return raw
        finally:
            if not committed:
                for fd in reversed(opened_fds):
                    try:
                        os.close(fd)
                    except OSError:
                        pass


def _load_strict_json(boundary: _RepositoryBoundary, relative_path: str, label: str) -> tuple[dict[str, Any], bytes, bytes]:
    raw = boundary.read_bytes(relative_path, label)
    value, canonical = _strict_json_bytes(raw, label)
    _validate_builtin_json(value, label)
    return value, raw, canonical


def _validate_authority_manifest(manifest: dict[str, Any]) -> None:
    _validate_builtin_json(manifest, "authority_manifest")
    _exact_keys(manifest, _MANIFEST_KEYS, "authority_manifest")
    _exact_int(manifest["schema_version"], "authority_manifest/schema_version", minimum=1)
    for key in ("authority_id", "repository_url", "branch", "toolkit_version", "algorithm_semantics_version", "license_and_use_notice"):
        _exact_str(manifest[key], f"authority_manifest/{key}")
    _exact_git_oid(manifest["commit_oid"], "authority_manifest/commit_oid")
    _exact_git_oid(manifest["tree_oid"], "authority_manifest/tree_oid")
    if manifest["repository_url"] != AUTHORITY_REPOSITORY_URL or manifest["branch"] != AUTHORITY_BRANCH:
        _fail("authority repository identity drift")
    if manifest["commit_oid"] != AUTHORITY_COMMIT or manifest["tree_oid"] != AUTHORITY_TREE:
        _fail("authority commit or tree drift")
    archive = _exact_keys(manifest["archive"], _MANIFEST_ARCHIVE_KEYS, "authority_manifest/archive")
    _exact_str(archive["filename"], "authority_manifest/archive/filename")
    _exact_str(archive["prefix"], "authority_manifest/archive/prefix")
    _exact_int(archive["size_bytes"], "authority_manifest/archive/size_bytes", minimum=1)
    _exact_sha(archive["sha256"], "authority_manifest/archive/sha256")
    inventory = _exact_keys(manifest["inventory"], _MANIFEST_INVENTORY_KEYS, "authority_manifest/inventory")
    file_count = _exact_int(inventory["file_count"], "authority_manifest/inventory/file_count", minimum=1)
    _exact_int(inventory["total_size_bytes"], "authority_manifest/inventory/total_size_bytes", minimum=1)
    _exact_str(inventory["canonicalization"], "authority_manifest/inventory/canonicalization")
    _exact_sha(inventory["canonical_inventory_sha256"], "authority_manifest/inventory/canonical_inventory_sha256")
    rows = inventory["rows"]
    if type(rows) is not list or len(rows) != file_count:
        _fail("authority inventory row count drift")
    previous = ""
    canonical_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        row_value = _exact_keys(row, _MANIFEST_ROW_KEYS, f"authority_manifest/inventory/rows/{index}")
        relative = _exact_str(row_value["relative_path"], f"authority_manifest/inventory/rows/{index}/relative_path")
        if relative <= previous or PurePosixPath(relative).is_absolute() or ".." in PurePosixPath(relative).parts:
            _fail("authority inventory paths are not sorted and contained")
        previous = relative
        _exact_str(row_value["git_mode"], f"authority_manifest/inventory/rows/{index}/git_mode")
        _exact_git_oid(row_value["git_blob_oid"], f"authority_manifest/inventory/rows/{index}/git_blob_oid")
        _exact_int(row_value["size_bytes"], f"authority_manifest/inventory/rows/{index}/size_bytes", minimum=0)
        _exact_sha(row_value["sha256"], f"authority_manifest/inventory/rows/{index}/sha256")
        canonical_rows.append(row_value)
    if sum(row["size_bytes"] for row in canonical_rows) != inventory["total_size_bytes"]:
        _fail("authority inventory total size drift")
    if _sha(canonical_json_bytes(canonical_rows)) != inventory["canonical_inventory_sha256"]:
        _fail("authority inventory canonical SHA mismatch")
    if inventory["canonical_inventory_sha256"] != AUTHORITY_INVENTORY_SHA256:
        _fail("authority inventory frozen SHA mismatch")


def _validate_contract(config: dict[str, Any], authority_manifest: dict[str, Any] | None) -> None:
    _validate_builtin_json(config, "primary_evaluator_contract")
    _exact_keys(config, _CONFIG_KEYS, "primary_evaluator_contract")
    if _exact_int(config["schema_version"], "/schema_version", minimum=1) != 1:
        _fail("primary evaluator schema version drift")
    if _exact_str(config["evaluator_id"], "/evaluator_id") != PRIMARY_EVALUATOR_ID:
        _fail("primary evaluator ID drift")
    if _exact_str(config["protocol_id"], "/protocol_id") != PRIMARY_PROTOCOL_ID:
        _fail("primary protocol ID drift")
    if _exact_str(config["schema_id"], "/schema_id") != PRIMARY_SCHEMA_ID:
        _fail("primary input schema ID drift")

    authority = _exact_keys(config["authority"], _AUTHORITY_KEYS, "/authority")
    for key in ("repository_url", "branch", "manifest_relative_path", "manifest_raw_sha256", "manifest_canonical_sha256", "inventory_canonical_sha256"):
        _exact_str(authority[key], f"/authority/{key}")
    _exact_git_oid(authority["commit_oid"], "/authority/commit_oid")
    _exact_git_oid(authority["tree_oid"], "/authority/tree_oid")
    if authority["repository_url"] != AUTHORITY_REPOSITORY_URL or authority["branch"] != AUTHORITY_BRANCH:
        _fail("contract authority repository drift")
    if authority["commit_oid"] != AUTHORITY_COMMIT or authority["tree_oid"] != AUTHORITY_TREE:
        _fail("contract authority commit drift")
    if authority["manifest_relative_path"] != PRIMARY_MANIFEST_RELATIVE_PATH:
        _fail("contract manifest path drift")
    _exact_int(authority["manifest_raw_size_bytes"], "/authority/manifest_raw_size_bytes", minimum=1)
    _exact_int(authority["manifest_canonical_size_bytes"], "/authority/manifest_canonical_size_bytes", minimum=1)
    if authority["manifest_raw_size_bytes"] != AUTHORITY_MANIFEST_RAW_SIZE:
        _fail("contract manifest raw size drift")
    _exact_sha(authority["manifest_raw_sha256"], "/authority/manifest_raw_sha256")
    _exact_sha(authority["manifest_canonical_sha256"], "/authority/manifest_canonical_sha256")
    _exact_sha(authority["inventory_canonical_sha256"], "/authority/inventory_canonical_sha256")
    if authority["manifest_raw_sha256"] != AUTHORITY_MANIFEST_RAW_SHA256:
        _fail("contract manifest raw SHA drift")
    if authority["manifest_canonical_size_bytes"] != AUTHORITY_MANIFEST_CANONICAL_SIZE:
        _fail("contract manifest canonical size drift")
    if authority["manifest_canonical_sha256"] != AUTHORITY_MANIFEST_CANONICAL_SHA256:
        _fail("contract manifest canonical SHA drift")
    if authority["inventory_canonical_sha256"] != AUTHORITY_INVENTORY_SHA256:
        _fail("contract inventory SHA drift")
    authority_files = authority["authority_file_sha256"]
    if type(authority_files) is not dict or authority_files != AUTHORITY_FILE_SHA256:
        _fail("contract authority source hash map drift")
    for relative, digest in authority_files.items():
        _exact_str(relative, "/authority/authority_file_sha256/path")
        _exact_sha(digest, "/authority/authority_file_sha256/value")

    input_contract = _exact_keys(config["input_contract"], _INPUT_KEYS, "/input_contract")
    if input_contract["coordinate_system"] != "xyxy" or input_contract["box_conversion"] != "xywh: w=x2-x1; h=y2-y1; no_plus_one":
        _fail("contract coordinate binding drift")
    category_domain = _exact_keys(input_contract["category_domain"], _INPUT_CATEGORY_KEYS, "/input_contract/category_domain")
    for key in ("detections", "ground_truth", "scored"):
        if type(category_domain[key]) is not list or any(type(item) is not int for item in category_domain[key]):
            _fail("contract category domain must use builtin integer lists")
    _exact_int(category_domain["spatial_ignore"], "/input_contract/category_domain/spatial_ignore", minimum=0)
    _exact_int(category_domain["others"], "/input_contract/category_domain/others", minimum=0)
    if category_domain != {"detections": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "ground_truth": list(range(12)), "scored": list(range(1, 11)), "spatial_ignore": 0, "others": 11}:
        _fail("contract category domain drift")
    if input_contract["image_id_order"] != "unique_in_declared_order" or input_contract["detection_order"] != "per_image_non_increasing_score_stable_ties":
        _fail("contract ordering binding drift")
    if input_contract["max_dets_prefix"] != "before_category_filter" or input_contract["width_height_required"] is not True or input_contract["input_immutability"] != "required":
        _fail("contract input policy drift")

    algorithm = _exact_keys(config["algorithm"], _ALGORITHM_KEYS, "/algorithm")
    if type(algorithm["iou_thresholds"]) is not list or any(type(item) is not float for item in algorithm["iou_thresholds"]):
        _fail("contract IoU thresholds must be a builtin float list")
    if type(algorithm["max_dets"]) is not list or any(type(item) is not int for item in algorithm["max_dets"]):
        _fail("contract maxDets must be a builtin integer list")
    _exact_bool(algorithm["nms"], "/algorithm/nms")
    _exact_float(algorithm["ignore_threshold"], "/algorithm/ignore_threshold", minimum=0.0)
    if algorithm["iou_thresholds"] != list(IOU_THRESHOLDS) or algorithm["max_dets"] != list(MAX_DETS):
        _fail("contract threshold or maxDets drift")
    if algorithm["nms"] is not False or algorithm["ignore_threshold"] != 0.5:
        _fail("contract NMS or ignore threshold drift")
    expected_algorithm = {
        "ignore_fraction_rule": "remove_when_fraction_greater_or_equal",
        "standard_overlap": "intersection_over_union",
        "ignored_overlap": "intersection_over_detection_area",
        "rounding": "matlab_half_away_from_zero",
        "clipping": "one_based_inclusive",
        "matching": "stable_score_descending_standard_before_ignored",
        "threshold_comparison": "inclusive",
        "recall_denominator": "max(1,numel(gtMatch))",
        "ap_source": "maxDets=500",
        "voc_ap": "monotonic_precision_envelope_continuous_recall",
        "aggregation": "evalClass_occurrence_per_image",
        "metrics_units": "percentage_0_100_float64",
    }
    for key, expected in expected_algorithm.items():
        if algorithm[key] != expected:
            _fail(f"contract algorithm field drift: {key}")

    output_contract = _exact_keys(config["output_contract"], _OUTPUT_KEYS, "/output_contract")
    if type(output_contract["metric_names"]) is not list or any(type(item) is not str for item in output_contract["metric_names"]):
        _fail("contract metric names must be a builtin string list")
    if type(output_contract["evidence"]) is not list or any(type(item) is not str for item in output_contract["evidence"]):
        _fail("contract evidence names must be a builtin string list")
    _exact_bool(output_contract["ap_small_emitted"], "/output_contract/ap_small_emitted")
    _exact_bool(output_contract["result_immutable_detached"], "/output_contract/result_immutable_detached")
    if output_contract["metric_names"] != list(METRIC_NAMES) or output_contract["ap_small_emitted"] is not False:
        _fail("contract output metric drift")
    if output_contract["result_immutable_detached"] is not True or output_contract["float_dtype"] != "float64":
        _fail("contract result policy drift")
    if output_contract["evidence"] != [
        "contract_raw_and_canonical_identity", "authority_manifest_raw_and_canonical_identity",
        "canonical_input_sha256", "evaluated_counts", "eval_class_occurrence_vector",
        "per_class_iou_ap", "per_class_iou_max_dets_ar", "match_counts", "canonical_result_sha256",
    ]:
        _fail("contract evidence field drift")

    policy = _exact_keys(config["policy"], _POLICY_KEYS, "/policy")
    expected_policy = {
        "development_only": True, "confirmatory_access_allowed": False, "test_access_allowed": False,
        "implementation_present": True, "independent_audit_pass": False, "training_gate_open": False,
        "secondary_evaluator_can_certify": False, "secondary_evaluator_can_select": False,
    }
    if any(type(policy[key]) is not bool for key in _POLICY_KEYS):
        _fail("contract policy values must be builtin booleans")
    if policy != expected_policy:
        _fail("contract policy drift")
    if authority_manifest is not None:
        _validate_authority_manifest(authority_manifest)

    canonical = canonical_json_bytes(config)
    if PRIMARY_CONFIG_CANONICAL_SHA256 and _sha(canonical) != PRIMARY_CONFIG_CANONICAL_SHA256:
        _fail("primary evaluator canonical config identity drift")


def validate_primary_evaluator_contract(config: object, authority_manifest: object) -> None:
    """Validate the exact closed contract and its committed source manifest."""

    if type(config) is not dict or type(authority_manifest) is not dict:
        _fail("contract and authority manifest must be builtin objects")
    _validate_contract(config, authority_manifest)


def load_primary_evaluator_authority_manifest(repo_root: Path | str) -> dict[str, Any]:
    with _RepositoryBoundary(repo_root) as boundary:
        manifest, raw, canonical = _load_strict_json(boundary, PRIMARY_MANIFEST_RELATIVE_PATH, "authority manifest")
        if len(raw) != AUTHORITY_MANIFEST_RAW_SIZE or _sha(raw) != AUTHORITY_MANIFEST_RAW_SHA256:
            _fail("authority manifest raw identity drift")
        if len(canonical) != AUTHORITY_MANIFEST_CANONICAL_SIZE or _sha(canonical) != AUTHORITY_MANIFEST_CANONICAL_SHA256:
            _fail("authority manifest canonical identity drift")
        _validate_authority_manifest(manifest)
        boundary.assert_stable()
        return manifest


def load_primary_evaluator_contract(repo_root: Path | str) -> dict[str, Any]:
    with _RepositoryBoundary(repo_root) as boundary:
        config, raw, canonical = _load_strict_json(boundary, PRIMARY_CONFIG_RELATIVE_PATH, "primary evaluator contract")
        manifest, manifest_raw, manifest_canonical = _load_strict_json(boundary, PRIMARY_MANIFEST_RELATIVE_PATH, "authority manifest")
        if PRIMARY_CONFIG_RAW_SIZE and (len(raw) != PRIMARY_CONFIG_RAW_SIZE or _sha(raw) != PRIMARY_CONFIG_RAW_SHA256):
            _fail("primary evaluator contract raw identity drift")
        if PRIMARY_CONFIG_CANONICAL_SIZE and (len(canonical) != PRIMARY_CONFIG_CANONICAL_SIZE or _sha(canonical) != PRIMARY_CONFIG_CANONICAL_SHA256):
            _fail("primary evaluator contract canonical identity drift")
        if len(manifest_raw) != AUTHORITY_MANIFEST_RAW_SIZE or _sha(manifest_raw) != AUTHORITY_MANIFEST_RAW_SHA256:
            _fail("authority manifest raw identity drift")
        if len(manifest_canonical) != AUTHORITY_MANIFEST_CANONICAL_SIZE or _sha(manifest_canonical) != AUTHORITY_MANIFEST_CANONICAL_SHA256:
            _fail("authority manifest canonical identity drift")
        _validate_contract(config, manifest)
        boundary.assert_stable()
        return config


def primary_evaluator_contract_binding(repo_root: Path | str) -> dict[str, Any]:
    with _RepositoryBoundary(repo_root) as boundary:
        config, config_raw, config_canonical = _load_strict_json(boundary, PRIMARY_CONFIG_RELATIVE_PATH, "primary evaluator contract")
        manifest, manifest_raw, manifest_canonical = _load_strict_json(boundary, PRIMARY_MANIFEST_RELATIVE_PATH, "authority manifest")
        if len(config_raw) != PRIMARY_CONFIG_RAW_SIZE or _sha(config_raw) != PRIMARY_CONFIG_RAW_SHA256:
            _fail("primary evaluator contract raw identity drift")
        if len(config_canonical) != PRIMARY_CONFIG_CANONICAL_SIZE or _sha(config_canonical) != PRIMARY_CONFIG_CANONICAL_SHA256:
            _fail("primary evaluator contract canonical identity drift")
        if len(manifest_raw) != AUTHORITY_MANIFEST_RAW_SIZE or _sha(manifest_raw) != AUTHORITY_MANIFEST_RAW_SHA256:
            _fail("authority manifest raw identity drift")
        if len(manifest_canonical) != AUTHORITY_MANIFEST_CANONICAL_SIZE or _sha(manifest_canonical) != AUTHORITY_MANIFEST_CANONICAL_SHA256:
            _fail("authority manifest canonical identity drift")
        _validate_contract(config, manifest)
        boundary.assert_stable()
        return {
            "config": config,
            "authority_manifest": manifest,
            "config_raw_size_bytes": len(config_raw),
            "config_raw_sha256": _sha(config_raw),
            "config_canonical_size_bytes": len(config_canonical),
            "config_canonical_sha256": _sha(config_canonical),
            "authority_manifest_raw_size_bytes": len(manifest_raw),
            "authority_manifest_raw_sha256": _sha(manifest_raw),
            "authority_manifest_canonical_size_bytes": len(manifest_canonical),
            "authority_manifest_canonical_sha256": _sha(manifest_canonical),
        }


def _real(value: object, path: str) -> float:
    if type(value) not in (int, float) or type(value) is bool:
        _fail(f"{path} must be a builtin numeric scalar")
    value_float = float(value)
    if not math.isfinite(value_float):
        _fail(f"{path} must be finite")
    return value_float


def _box_xywh(box: tuple[float, float, float, float], path: str) -> tuple[float, float, float, float]:
    if type(box) is not tuple or len(box) != 4:
        _fail(f"{path} must be an exact four-tuple")
    values = tuple(_real(item, f"{path}/{index}") for index, item in enumerate(box))
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        _fail(f"{path} must have positive extent")
    return x1, y1, x2 - x1, y2 - y1


def _validate_input(value: object) -> PrimaryEvaluatorInputV2:
    if type(value) is not PrimaryEvaluatorInputV2:
        _fail("primary evaluator requires PrimaryEvaluatorInputV2")
    if type(value.images) is not tuple or type(value.detections) is not tuple or type(value.ground_truth) is not tuple:
        _fail("input containers must be exact tuples")
    if type(value.protocol) is not str or value.protocol != PRIMARY_PROTOCOL_ID:
        _fail("input protocol drift")
    if type(value.iou_thresholds) is not tuple or len(value.iou_thresholds) != len(IOU_THRESHOLDS) or any(type(item) is not float or not math.isfinite(item) for item in value.iou_thresholds) or value.iou_thresholds != IOU_THRESHOLDS:
        _fail("input IoU thresholds drift")
    if type(value.max_dets) is not tuple or any(type(item) is not int for item in value.max_dets) or value.max_dets != MAX_DETS:
        _fail("input maxDets drift")
    if type(value.nms) is not bool or value.nms is not False:
        _fail("NMS is forbidden")
    if type(value.ignore_threshold) is not float or not math.isfinite(value.ignore_threshold) or value.ignore_threshold != 0.5:
        _fail("input ignored-region threshold drift")
    if type(value.standard_overlap) is not str or type(value.ignored_overlap) is not str or value.standard_overlap != "intersection_over_union" or value.ignored_overlap != "intersection_over_detection_area":
        _fail("input overlap rule drift")

    image_ids: set[str] = set()
    for index, image in enumerate(value.images):
        if type(image) is not PrimaryImageV2:
            _fail(f"images/{index} has an invalid exact type")
        if type(image.image_id) is not str or not image.image_id or image.image_id in image_ids:
            _fail("image IDs must be unique non-empty builtin strings")
        if type(image.image_id) is not str or type(image.width) is not int or type(image.height) is not int or image.width <= 0 or image.height <= 0:
            _fail("image dimensions must be positive builtin integers")
        image_ids.add(image.image_id)

    last_scores: dict[str, float] = {}
    for index, detection in enumerate(value.detections):
        if type(detection) is not Detection:
            _fail(f"detections/{index} has an invalid exact type")
        if type(detection.image_id) is not str or detection.image_id not in image_ids:
            _fail("detection image ID is absent from image table")
        if type(detection.category_id) is not int or detection.category_id not in range(1, 11):
            _fail("detections must use categories 1..10")
        _box_xywh(detection.bbox_xyxy, f"detections/{index}/bbox_xyxy")
        score = _real(detection.score, f"detections/{index}/score")
        if detection.image_id in last_scores and score > last_scores[detection.image_id]:
            _fail("detections for each image must be non-increasing by score")
        last_scores[detection.image_id] = score

    annotation_ids: set[str] = set()
    for index, ground_truth in enumerate(value.ground_truth):
        if type(ground_truth) is not PrimaryGroundTruth:
            _fail(f"ground_truth/{index} has an invalid exact type")
        if type(ground_truth.annotation_id) is not str or not ground_truth.annotation_id or ground_truth.annotation_id in annotation_ids:
            _fail("ground-truth annotation IDs must be non-empty builtin strings")
        annotation_ids.add(ground_truth.annotation_id)
        if type(ground_truth.image_id) is not str or ground_truth.image_id not in image_ids:
            _fail("ground-truth image ID is absent from image table")
        if type(ground_truth.category_id) is not int or ground_truth.category_id not in range(12):
            _fail("ground-truth category is outside 0..11")
        if type(ground_truth.ignore_region) is not bool or type(ground_truth.ignored) is not bool:
            _fail("ground-truth ignore flags must be builtin booleans")
        if ground_truth.ignore_region is not (ground_truth.category_id == 0):
            _fail("ignore_region must be true exactly for category 0")
        x, y, width, height = _box_xywh(ground_truth.bbox_xyxy, f"ground_truth/{index}/bbox_xyxy")
        area = _real(ground_truth.area, f"ground_truth/{index}/area")
        if area <= 0 or area != width * height:
            _fail("ground-truth stored area does not match its xyxy extent")
        _ = (x, y)
    return value


def _input_payload(value: PrimaryEvaluatorInputV2) -> dict[str, Any]:
    return {
        "images": [{"image_id": image.image_id, "width": image.width, "height": image.height} for image in value.images],
        "detections": [
            {"image_id": item.image_id, "category_id": item.category_id, "bbox_xyxy": list(item.bbox_xyxy), "score": item.score}
            for item in value.detections
        ],
        "ground_truth": [
            {"annotation_id": item.annotation_id, "image_id": item.image_id, "category_id": item.category_id,
             "bbox_xyxy": list(item.bbox_xyxy), "area": item.area, "ignore_region": item.ignore_region, "ignored": item.ignored}
            for item in value.ground_truth
        ],
        "protocol": value.protocol,
        "iou_thresholds": list(value.iou_thresholds),
        "max_dets": list(value.max_dets),
        "nms": value.nms,
        "ignore_threshold": value.ignore_threshold,
        "standard_overlap": value.standard_overlap,
        "ignored_overlap": value.ignored_overlap,
    }


def _matlab_round(value: float) -> int:
    return math.floor(value + 0.5) if value >= 0 else math.ceil(value - 0.5)


def _official_xywh(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x, y, width, height = _box_xywh(box, "box")
    return x, y, width, height


def _rounded_positive_xywh(box: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    x, y, width, height = _official_xywh(box)
    return max(1, _matlab_round(x)), max(1, _matlab_round(y)), max(1, _matlab_round(width)), max(1, _matlab_round(height))


def _build_integral(image: PrimaryImageV2, regions: list[PrimaryGroundTruth]) -> np.ndarray:
    canvas = np.zeros((image.height, image.width), dtype=np.float64)
    for region in regions:
        x, y, width, height = _rounded_positive_xywh(region.bbox_xyxy)
        x = max(1, min(image.width, x))
        y = max(1, min(image.height, y))
        x_end = min(image.width, x + width)
        y_end = min(image.height, y + height)
        canvas[y - 1:y_end, x - 1:x_end] = 1.0
    return np.cumsum(np.cumsum(canvas, axis=0), axis=1)


def _integral_value(integral: np.ndarray, y: int, x: int) -> float:
    return float(integral[y - 1, x - 1])


def _ignored_fraction(integral: np.ndarray, image: PrimaryImageV2, box: tuple[float, float, float, float]) -> float:
    x, y, width, height = _rounded_positive_xywh(box)
    x = max(1, min(image.width, x))
    y = max(1, min(image.height, y))
    x_end = max(1, min(image.width, x + width))
    y_end = max(1, min(image.height, y + height))
    tl = _integral_value(integral, y, x)
    tr = _integral_value(integral, y, x_end)
    bl = _integral_value(integral, y_end, x)
    br = _integral_value(integral, y_end, x_end)
    return (tl + br - tr - bl) / float(width * height)


def _overlap(det: tuple[float, float, float, float], gt: tuple[float, float, float, float], ignored: bool) -> float:
    dx, dy, dw, dh = det
    gx, gy, gw, gh = gt
    width = min(dx + dw, gx + gw) - max(dx, gx)
    height = min(dy + dh, gy + gh) - max(dy, gy)
    if width <= 0 or height <= 0:
        return 0.0
    intersection = width * height
    denominator = dw * dh if ignored else dw * dh + gw * gh - intersection
    return intersection / denominator


def _match(
    detections: list[tuple[float, tuple[float, float, float, float]]],
    ground_truth: list[tuple[tuple[float, float, float, float], bool]],
    threshold: float,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    ordered_gt = sorted(enumerate(ground_truth), key=lambda item: (item[1][1], item[0]))
    gt_boxes = [item[1][0] for item in ordered_gt]
    gt_ignored = [item[1][1] for item in ordered_gt]
    gt_state = [-1 if ignored else 0 for ignored in gt_ignored]
    det_state = [0] * len(detections)
    for det_index, (score, det_box) in enumerate(detections):
        _ = score
        best_overlap = threshold
        best_gt = -1
        best_match = 0
        for gt_index, gt_box in enumerate(gt_boxes):
            state = gt_state[gt_index]
            if state == 1:
                continue
            if best_match != 0 and state == -1:
                break
            overlap = _overlap(det_box, gt_box, state == -1)
            if overlap < best_overlap:
                continue
            best_overlap = overlap
            best_gt = gt_index
            best_match = -1 if state == -1 else 1
        if best_match == -1:
            det_state[det_index] = -1
        elif best_match == 1:
            gt_state[best_gt] = 1
            det_state[det_index] = 1
    original_state = [0] * len(ground_truth)
    for ordered_index, (original_index, _) in enumerate(ordered_gt):
        original_state[original_index] = gt_state[ordered_index]
    return tuple(original_state), tuple(det_state)


def _voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    if recall.size == 0:
        return 0.0
    mrec = np.concatenate((np.array([0.0]), recall.astype(np.float64), np.array([1.0])))
    mpre = np.concatenate((np.array([0.0]), precision.astype(np.float64), np.array([0.0])))
    for index in range(mpre.size - 2, -1, -1):
        mpre[index] = max(mpre[index], mpre[index + 1])
    changes = np.nonzero(mrec[1:] != mrec[:-1])[0] + 1
    return float(np.sum((mrec[changes] - mrec[changes - 1]) * mpre[changes]))


def _nested_lists(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_nested_lists(item) for item in value]
    return value


def _binding_identity(contract: object) -> tuple[dict[str, Any], str, str, str, str]:
    if type(contract) is not dict:
        _fail("evaluate_primary_v1 requires an authoritative binding dictionary")
    _validate_builtin_json(contract, "contract_binding")
    _exact_keys(contract, _BINDING_KEYS, "contract_binding")
    config = contract["config"]
    manifest = contract["authority_manifest"]
    if type(config) is not dict or type(manifest) is not dict:
        _fail("contract binding objects must be builtin dictionaries")
    for key in (
        "config_raw_size_bytes", "config_canonical_size_bytes",
        "authority_manifest_raw_size_bytes", "authority_manifest_canonical_size_bytes",
    ):
        _exact_int(contract[key], f"contract_binding/{key}", minimum=0)
    for key in (
        "config_raw_sha256", "config_canonical_sha256",
        "authority_manifest_raw_sha256", "authority_manifest_canonical_sha256",
    ):
        _exact_sha(contract[key], f"contract_binding/{key}")
    expected = {
        "config_raw_size_bytes": PRIMARY_CONFIG_RAW_SIZE,
        "config_raw_sha256": PRIMARY_CONFIG_RAW_SHA256,
        "config_canonical_size_bytes": PRIMARY_CONFIG_CANONICAL_SIZE,
        "config_canonical_sha256": PRIMARY_CONFIG_CANONICAL_SHA256,
        "authority_manifest_raw_size_bytes": AUTHORITY_MANIFEST_RAW_SIZE,
        "authority_manifest_raw_sha256": AUTHORITY_MANIFEST_RAW_SHA256,
        "authority_manifest_canonical_size_bytes": AUTHORITY_MANIFEST_CANONICAL_SIZE,
        "authority_manifest_canonical_sha256": AUTHORITY_MANIFEST_CANONICAL_SHA256,
    }
    for key, frozen in expected.items():
        if contract[key] != frozen:
            _fail(f"contract binding identity drift: {key}")
    _validate_contract(config, manifest)
    return (
        config,
        contract["config_raw_sha256"],
        contract["config_canonical_sha256"],
        contract["authority_manifest_raw_sha256"],
        contract["authority_manifest_canonical_sha256"],
    )


def evaluate_primary_v1(input_v2: PrimaryEvaluatorInputV2, contract: object) -> PrimaryEvaluatorResultV2:
    """Evaluate one immutable V2 input using the frozen official semantics."""

    config, config_raw_sha, config_canonical_sha, manifest_raw_sha, manifest_canonical_sha = _binding_identity(contract)
    value = _validate_input(input_v2)
    _ = config
    image_by_id = {image.image_id: image for image in value.images}
    spatial_by_image: dict[str, list[PrimaryGroundTruth]] = {image.image_id: [] for image in value.images}
    for ground_truth in value.ground_truth:
        if ground_truth.category_id == 0 or ground_truth.ignore_region:
            spatial_by_image[ground_truth.image_id].append(ground_truth)
    integral_by_image = {image_id: _build_integral(image_by_id[image_id], regions) for image_id, regions in spatial_by_image.items()}
    filtered_gt: dict[str, list[PrimaryGroundTruth]] = {image.image_id: [] for image in value.images}
    for ground_truth in value.ground_truth:
        if ground_truth.ignore_region or ground_truth.category_id in (0, 11) or ground_truth.category_id is None:
            continue
        image = image_by_id[ground_truth.image_id]
        fraction = _ignored_fraction(integral_by_image[ground_truth.image_id], image, ground_truth.bbox_xyxy)
        if fraction < value.ignore_threshold:
            filtered_gt[ground_truth.image_id].append(ground_truth)
    filtered_det: dict[str, list[Detection]] = {image.image_id: [] for image in value.images}
    for detection in value.detections:
        image = image_by_id[detection.image_id]
        fraction = _ignored_fraction(integral_by_image[detection.image_id], image, detection.bbox_xyxy)
        if fraction < value.ignore_threshold:
            filtered_det[detection.image_id].append(detection)

    occurrences: list[int] = []
    for image in value.images:
        present = {item.category_id for item in filtered_gt[image.image_id] if item.category_id in range(1, 11)}
        occurrences.extend(sorted(present))

    ap_table = np.zeros((10, 10), dtype=np.float64)
    ar_table = np.zeros((10, 10, 4), dtype=np.float64)
    match_counts = np.zeros((10, 10, 4, 4), dtype=np.int64)
    for category_index, category_id in enumerate(range(1, 11)):
        for iou_index, threshold in enumerate(value.iou_thresholds):
            ap_records: list[tuple[float, int]] = []
            for max_index, max_dets in enumerate(value.max_dets):
                gt_match_all: list[int] = []
                det_records: list[tuple[float, int]] = []
                for image in value.images:
                    image_id = image.image_id
                    gt_items = [item for item in filtered_gt[image_id] if item.category_id == category_id]
                    det_items = [item for item in filtered_det[image_id]][:max_dets]
                    det_items = [item for item in det_items if item.category_id == category_id]
                    gt_boxes = [(_official_xywh(item.bbox_xyxy), item.ignored) for item in gt_items]
                    det_boxes = [(item.score, _official_xywh(item.bbox_xyxy)) for item in det_items]
                    gt_states, det_states = _match(det_boxes, gt_boxes, threshold)
                    gt_match_all.extend(gt_states)
                    det_records.extend((float(item.score), state) for item, state in zip(det_items, det_states))
                ordered = sorted(enumerate(det_records), key=lambda item: (-item[1][0], item[0]))
                tp = sum(state == 1 for _, (_, state) in ordered)
                fp = sum(state == 0 for _, (_, state) in ordered)
                ignored = sum(state == -1 for _, (_, state) in ordered)
                match_counts[category_index, iou_index, max_index] = (len(gt_match_all), tp, fp, ignored)
                if gt_match_all:
                    running_tp = 0
                    max_recall = 0.0
                    denominator = max(1, len(gt_match_all))
                    for _, (_, state) in ordered:
                        running_tp += state == 1
                        max_recall = max(max_recall, running_tp / denominator)
                    ar_table[category_index, iou_index, max_index] = max_recall * 100.0
                if max_index == len(value.max_dets) - 1:
                    ap_records = [record for _, record in ordered]
            if ap_records:
                tp_values = np.cumsum(np.array([state == 1 for _, state in ap_records], dtype=np.int64))
                fp_values = np.cumsum(np.array([state == 0 for _, state in ap_records], dtype=np.int64))
                denominator = max(1, sum(len([item for item in filtered_gt[image.image_id] if item.category_id == category_id]) for image in value.images))
                recall = tp_values.astype(np.float64) / denominator
                precision = tp_values.astype(np.float64) / np.maximum(1, tp_values + fp_values)
                ap_table[category_index, iou_index] = _voc_ap(recall, precision) * 100.0

    occurrence_rows = [category - 1 for category in occurrences]
    if occurrence_rows:
        selected_ap = ap_table[occurrence_rows, :]
        selected_ar = ar_table[occurrence_rows, :, :]
        aggregate_ap = float(np.mean(selected_ap))
        aggregate_ap50 = float(np.mean(selected_ap[:, 0]))
        aggregate_ap75 = float(np.mean(selected_ap[:, 5]))
        aggregate_ar = [float(np.mean(selected_ar[:, :, index])) for index in range(4)]
    else:
        aggregate_ap = aggregate_ap50 = aggregate_ap75 = 0.0
        aggregate_ar = [0.0, 0.0, 0.0, 0.0]

    ap_tuple = tuple(tuple(float(item) for item in row) for row in ap_table.tolist())
    ar_tuple = tuple(tuple(tuple(float(item) for item in row) for row in category) for category in ar_table.tolist())
    count_tuple = tuple(tuple(tuple(tuple(int(item) for item in counts) for counts in iou) for iou in category) for category in match_counts.tolist())
    metrics = (aggregate_ap, aggregate_ap50, aggregate_ap75, *aggregate_ar)
    result_payload = {
        "protocol": PRIMARY_PROTOCOL_ID,
        "metric_names": list(METRIC_NAMES),
        "metrics": list(metrics),
        "AP": aggregate_ap,
        "AP50": aggregate_ap50,
        "AP75": aggregate_ap75,
        "AR1": aggregate_ar[0],
        "AR10": aggregate_ar[1],
        "AR100": aggregate_ar[2],
        "AR500": aggregate_ar[3],
        "evaluated_image_count": len(value.images),
        "evaluated_detection_count": sum(len(items) for items in filtered_det.values()),
        "evaluated_ground_truth_count": sum(len(items) for items in filtered_gt.values()),
        "canonical_input_sha256": _sha(canonical_json_bytes(_input_payload(value))),
        "contract_raw_sha256": config_raw_sha,
        "contract_canonical_sha256": config_canonical_sha,
        "authority_manifest_raw_sha256": manifest_raw_sha,
        "authority_manifest_canonical_sha256": manifest_canonical_sha,
        "eval_class_occurrences": occurrences,
        "ap_by_class_iou": _nested_lists(ap_tuple),
        "ar_by_class_iou_max_dets": _nested_lists(ar_tuple),
        "match_counts_by_class_iou_max_dets": _nested_lists(count_tuple),
    }
    result_sha = _sha(canonical_json_bytes(result_payload))
    return PrimaryEvaluatorResultV2(
        protocol=PRIMARY_PROTOCOL_ID,
        metric_names=METRIC_NAMES,
        metrics=tuple(float(item) for item in metrics),
        AP=aggregate_ap,
        AP50=aggregate_ap50,
        AP75=aggregate_ap75,
        AR1=aggregate_ar[0],
        AR10=aggregate_ar[1],
        AR100=aggregate_ar[2],
        AR500=aggregate_ar[3],
        evaluated_image_count=len(value.images),
        evaluated_detection_count=sum(len(items) for items in filtered_det.values()),
        evaluated_ground_truth_count=sum(len(items) for items in filtered_gt.values()),
        canonical_input_sha256=result_payload["canonical_input_sha256"],
        contract_raw_sha256=config_raw_sha,
        contract_canonical_sha256=config_canonical_sha,
        authority_manifest_raw_sha256=manifest_raw_sha,
        authority_manifest_canonical_sha256=manifest_canonical_sha,
        eval_class_occurrences=tuple(occurrences),
        ap_by_class_iou=ap_tuple,
        ar_by_class_iou_max_dets=ar_tuple,
        match_counts_by_class_iou_max_dets=count_tuple,
        canonical_result_sha256=result_sha,
    )


def _result_payload(result: PrimaryEvaluatorResultV2) -> dict[str, Any]:
    return {
        "protocol": result.protocol,
        "metric_names": list(result.metric_names),
        "metrics": list(result.metrics),
        "AP": result.AP, "AP50": result.AP50, "AP75": result.AP75,
        "AR1": result.AR1, "AR10": result.AR10, "AR100": result.AR100, "AR500": result.AR500,
        "evaluated_image_count": result.evaluated_image_count,
        "evaluated_detection_count": result.evaluated_detection_count,
        "evaluated_ground_truth_count": result.evaluated_ground_truth_count,
        "canonical_input_sha256": result.canonical_input_sha256,
        "contract_raw_sha256": result.contract_raw_sha256,
        "contract_canonical_sha256": result.contract_canonical_sha256,
        "authority_manifest_raw_sha256": result.authority_manifest_raw_sha256,
        "authority_manifest_canonical_sha256": result.authority_manifest_canonical_sha256,
        "eval_class_occurrences": list(result.eval_class_occurrences),
        "ap_by_class_iou": _nested_lists(result.ap_by_class_iou),
        "ar_by_class_iou_max_dets": _nested_lists(result.ar_by_class_iou_max_dets),
        "match_counts_by_class_iou_max_dets": _nested_lists(result.match_counts_by_class_iou_max_dets),
    }


def _result_float(value: object, path: str) -> float:
    if type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 100.0:
        _fail(f"{path} must be a finite builtin float in [0,100]")
    return value


def _validate_result_schema(result: object) -> PrimaryEvaluatorResultV2:
    if type(result) is not PrimaryEvaluatorResultV2:
        _fail("primary evaluator result has the wrong exact type")
    if type(result.protocol) is not str or result.protocol != PRIMARY_PROTOCOL_ID:
        _fail("primary evaluator result protocol drift")
    if type(result.metric_names) is not tuple or len(result.metric_names) != 7 or any(type(item) is not str for item in result.metric_names) or result.metric_names != METRIC_NAMES:
        _fail("primary evaluator result metric names drift")
    if type(result.metrics) is not tuple or len(result.metrics) != 7:
        _fail("primary evaluator result metrics shape drift")
    metrics = tuple(_result_float(item, f"result/metrics/{index}") for index, item in enumerate(result.metrics))
    named = tuple(
        _result_float(getattr(result, field), f"result/{field}")
        for field in ("AP", "AP50", "AP75", "AR1", "AR10", "AR100", "AR500")
    )
    if metrics != named:
        _fail("primary evaluator result named metrics are not bound to metrics")
    for field in ("evaluated_image_count", "evaluated_detection_count", "evaluated_ground_truth_count"):
        _exact_int(getattr(result, field), f"result/{field}", minimum=0)
    for field in (
        "canonical_input_sha256", "contract_raw_sha256", "contract_canonical_sha256",
        "authority_manifest_raw_sha256", "authority_manifest_canonical_sha256", "canonical_result_sha256",
    ):
        _exact_sha(getattr(result, field), f"result/{field}")

    occurrences = result.eval_class_occurrences
    if type(occurrences) is not tuple or any(type(item) is not int or item not in range(1, 11) for item in occurrences):
        _fail("primary evaluator result occurrence vector drift")

    ap = result.ap_by_class_iou
    if type(ap) is not tuple or len(ap) != 10:
        _fail("primary evaluator result AP table shape drift")
    for class_index, row in enumerate(ap):
        if type(row) is not tuple or len(row) != 10:
            _fail(f"result/ap_by_class_iou/{class_index} shape drift")
        for iou_index, item in enumerate(row):
            _result_float(item, f"result/ap_by_class_iou/{class_index}/{iou_index}")

    ar = result.ar_by_class_iou_max_dets
    if type(ar) is not tuple or len(ar) != 10:
        _fail("primary evaluator result AR table shape drift")
    for class_index, category in enumerate(ar):
        if type(category) is not tuple or len(category) != 10:
            _fail(f"result/ar_by_class_iou_max_dets/{class_index} shape drift")
        for iou_index, row in enumerate(category):
            if type(row) is not tuple or len(row) != 4:
                _fail(f"result/ar_by_class_iou_max_dets/{class_index}/{iou_index} shape drift")
            for max_index, item in enumerate(row):
                _result_float(item, f"result/ar_by_class_iou_max_dets/{class_index}/{iou_index}/{max_index}")

    counts = result.match_counts_by_class_iou_max_dets
    if type(counts) is not tuple or len(counts) != 10:
        _fail("primary evaluator result match-count shape drift")
    for class_index, category in enumerate(counts):
        if type(category) is not tuple or len(category) != 10:
            _fail(f"result/match_counts/{class_index} shape drift")
        for iou_index, rows in enumerate(category):
            if type(rows) is not tuple or len(rows) != 4:
                _fail(f"result/match_counts/{class_index}/{iou_index} shape drift")
            for max_index, row in enumerate(rows):
                if type(row) is not tuple or len(row) != 4:
                    _fail(f"result/match_counts/{class_index}/{iou_index}/{max_index} shape drift")
                gt_count, true_positive, false_positive, ignored = row
                for field, item in zip(("gt", "tp", "fp", "ignored"), row):
                    _exact_int(item, f"result/match_counts/{class_index}/{iou_index}/{max_index}/{field}", minimum=0)
                if true_positive > gt_count or true_positive + false_positive + ignored > result.evaluated_detection_count:
                    _fail("primary evaluator result match-count range drift")
                if gt_count > result.evaluated_ground_truth_count:
                    _fail("primary evaluator result ground-truth count range drift")
    return result


def validate_primary_evaluator_result(result: object, input_v2: PrimaryEvaluatorInputV2, contract: object) -> None:
    """Recompute every detached field and reject repacked or stale evidence."""

    _validate_result_schema(result)
    expected = evaluate_primary_v1(input_v2, contract)
    if result != expected:
        _fail("primary evaluator result evidence does not recompute")
    payload = _result_payload(result)
    if _sha(canonical_json_bytes(payload)) != result.canonical_result_sha256:
        _fail("primary evaluator result canonical SHA mismatch")


__all__ = [
    "AUTHORITY_COMMIT", "AUTHORITY_FILE_SHA256", "AUTHORITY_INVENTORY_SHA256", "AUTHORITY_MANIFEST_CANONICAL_SHA256",
    "AUTHORITY_MANIFEST_RAW_SHA256", "AUTHORITY_REPOSITORY_URL", "AUTHORITY_TREE", "IOU_THRESHOLDS", "MAX_DETS",
    "METRIC_NAMES", "PRIMARY_CONFIG_RELATIVE_PATH", "PRIMARY_EVALUATOR_ID", "PRIMARY_MANIFEST_RELATIVE_PATH",
    "PRIMARY_PROTOCOL_ID", "PRIMARY_SCHEMA_ID", "PrimaryEvaluatorContractError", "evaluate_primary_v1",
    "load_primary_evaluator_authority_manifest", "load_primary_evaluator_contract", "primary_evaluator_contract_binding",
    "validate_primary_evaluator_contract", "validate_primary_evaluator_result",
]
