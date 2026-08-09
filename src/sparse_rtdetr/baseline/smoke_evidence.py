"""Atomic evidence and inventory primitives for Baseline Smoke V1."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from ..data_protocol.schema import ProtocolContractError


class SmokeEvidenceError(ProtocolContractError):
    """Raised when smoke evidence ownership or binding is invalid."""


ENTRY_INVENTORY_EXCLUDED = frozenset({"artifact_inventory.json", "completion.json"})
ENTRY_PARTIAL_EXCLUDED = frozenset({"partial_inventory.json", "completion.json"})
ENTRY_REQUIRED_ARTIFACTS = frozenset({
    "config.json",
    "invocation.json",
    "source_identity.json",
    "data_binding_audit.json",
    "image_selection_audit.json",
    "cuda_runtime_identity.json",
    "model_identity.json",
    "call_audit.json",
    "input_batch_audit.json",
    "model_output_audit.json",
    "postprocess_audit.json",
    "rng_audit.json",
})
COMPLETION_COUNTERS = {
    "images_requested": 2,
    "images_loaded": 2,
    "batches_requested": 1,
    "batches_processed": 1,
    "dataset_next_calls": 1,
    "model_forward_calls": 1,
    "postprocessor_calls": 1,
    "evaluator_calls": 0,
    "criterion_calls": 0,
    "backward_calls": 0,
    "optimizer_steps": 0,
    "scheduler_steps": 0,
    "checkpoint_loads": 0,
    "network_calls": 0,
}
COMPLETION_BOOLEAN_FIELDS = {
    "smoke_pass_candidate": True,
    "non_training": True,
    "inference_only": True,
    "model_eval": True,
    "torch_no_grad": True,
    "non_selection": True,
    "non_reportable": True,
    "formal_resume_eligible": False,
    "formal_training_eligible": False,
    "baseline_training_ready": False,
    "confirmatory_metrics_accessed": False,
    "dataset_test_accessed_by_this_process": False,
    "speed_measurement": False,
}
COMPLETION_KEYS = frozenset({
    "schema_version",
    "status",
    "mode",
    "config_sha256",
    "total_predictions",
    "artifact_inventory_sha256",
    *COMPLETION_COUNTERS,
    *COMPLETION_BOOLEAN_FIELDS,
})
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SmokeEvidenceError("evidence value is not canonical JSON") from exc


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_argv(argv: list[str]) -> list[str]:
    if not isinstance(argv, list) or not argv or any(type(item) is not str or not item for item in argv):
        raise SmokeEvidenceError("child argv must be a non-empty list of non-empty strings")
    return [str(Path(argv[0]).resolve()), *argv[1:]]


def argv_sha256(argv: list[str]) -> str:
    return sha256_bytes(canonical_json_bytes(canonical_argv(argv)))


def atomic_write(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise SmokeEvidenceError(f"evidence file already exists: {path.name}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def file_ref(root: Path, name: str) -> dict[str, object]:
    path = root / name
    if path.is_symlink() or not path.is_file():
        return {"present": False, "relative_path": name}
    return {
        "present": True,
        "relative_path": name,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def inventory(root: Path, excluded: frozenset[str]) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if path.name in excluded:
            continue
        if path.name.startswith(".") or path.is_symlink() or not path.is_file():
            raise SmokeEvidenceError(f"invalid smoke evidence entry: {path.name}")
        if path.stat().st_nlink != 1:
            raise SmokeEvidenceError(f"hardlink smoke evidence entry: {path.name}")
        rows.append({
            "relative_path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return {
        "schema_version": 1,
        "excluded_from_inventory": sorted(excluded),
        "artifacts": rows,
        "canonical_inventory_sha256": sha256_bytes(canonical_json_bytes(rows)),
    }


def _read_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SmokeEvidenceError(f"missing regular evidence file: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SmokeEvidenceError(f"invalid evidence JSON: {path.name}") from exc
    if not isinstance(value, dict):
        raise SmokeEvidenceError(f"evidence JSON must be an object: {path.name}")
    return value


def _strict_sha(value: Any, field: str) -> None:
    if type(value) is not str or SHA256_RE.fullmatch(value) is None:
        raise SmokeEvidenceError(f"{field} must be a lowercase SHA-256 string")


def _validate_entry_invocation(
    root: Path,
    config_sha256: str,
    expected_identity: dict[str, Any] | None,
) -> dict[str, Any]:
    invocation = _read_object(root / "invocation.json")
    if type(invocation.get("schema_version")) is not int or invocation["schema_version"] != 1:
        raise SmokeEvidenceError("entry invocation schema version is invalid")
    if invocation.get("mode") not in {"synthetic", "real"} or invocation.get("smoke_id") != "rtdetrv2_r18_visdrone_baseline_smoke_v1":
        raise SmokeEvidenceError("entry invocation identity is invalid")
    if invocation.get("config_sha256") != config_sha256:
        raise SmokeEvidenceError("entry invocation config SHA mismatch")
    if expected_identity is not None:
        for field in (
            "nonce",
            "launcher_pid",
            "child_pid",
            "child_ppid",
            "child_argv_sha256",
            "handoff_receipt_sha256",
            "handoff_receipt_relative_path",
        ):
            if invocation.get(field) != expected_identity.get(field):
                raise SmokeEvidenceError(f"entry invocation identity mismatch: {field}")
        _strict_sha(invocation.get("handoff_receipt_sha256"), "entry invocation handoff_receipt_sha256")
    return invocation


def validate_entry_output(root: Path, expected_identity: dict[str, Any] | None = None) -> dict[str, object]:
    """Validate a completed child output without accepting exit code alone."""

    completion_path = root / "completion.json"
    inventory_path = root / "artifact_inventory.json"
    completion = _read_object(completion_path)
    inventory_value = _read_object(inventory_path)
    if set(completion) != COMPLETION_KEYS:
        raise SmokeEvidenceError("completion schema fields are invalid")
    if type(completion.get("schema_version")) is not int or completion["schema_version"] != 1:
        raise SmokeEvidenceError("completion schema version is invalid")
    if completion.get("status") != "COMPLETED" or completion.get("mode") != "smoke":
        raise SmokeEvidenceError("completion status or mode is invalid")
    for field, expected in COMPLETION_BOOLEAN_FIELDS.items():
        if type(completion.get(field)) is not bool or completion[field] is not expected:
            raise SmokeEvidenceError(f"completion boolean field is invalid: {field}")
    for field, expected in COMPLETION_COUNTERS.items():
        if type(completion.get(field)) is not int or completion[field] != expected:
            raise SmokeEvidenceError(f"completion counter is invalid: {field}")
    if type(completion.get("total_predictions")) is not int or completion["total_predictions"] != 600:
        raise SmokeEvidenceError("completion total_predictions is invalid")
    _strict_sha(completion.get("config_sha256"), "completion config_sha256")
    config_path = root / "config.json"
    if config_path.is_symlink() or not config_path.is_file():
        raise SmokeEvidenceError("entry config is missing or not regular")
    config_sha256 = sha256_file(config_path)
    if completion["config_sha256"] != config_sha256:
        raise SmokeEvidenceError("completion config SHA mismatch")
    invocation = _validate_entry_invocation(root, config_sha256, expected_identity)
    if completion.get("artifact_inventory_sha256") != sha256_file(inventory_path):
        raise SmokeEvidenceError("completion inventory SHA mismatch")
    if inventory_value.get("schema_version") != 1 or inventory_value.get("excluded_from_inventory") != sorted(ENTRY_INVENTORY_EXCLUDED):
        raise SmokeEvidenceError("entry inventory schema mismatch")
    rows = inventory_value.get("artifacts")
    if not isinstance(rows, list) or rows != sorted(rows, key=lambda row: row.get("relative_path", "") if isinstance(row, dict) else ""):
        raise SmokeEvidenceError("entry inventory rows are invalid")
    if inventory_value.get("canonical_inventory_sha256") != sha256_bytes(canonical_json_bytes(rows)):
        raise SmokeEvidenceError("entry inventory canonical SHA mismatch")
    names: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"relative_path", "size_bytes", "sha256"}:
            raise SmokeEvidenceError("entry inventory row schema is invalid")
        name = row.get("relative_path")
        if not isinstance(name, str) or not name or "/" in name or "\\" in name or name in ENTRY_INVENTORY_EXCLUDED or name in names:
            raise SmokeEvidenceError("entry inventory path is invalid")
        artifact = root / name
        if artifact.is_symlink() or not artifact.is_file() or artifact.stat().st_nlink != 1:
            raise SmokeEvidenceError(f"entry inventory artifact is not regular: {name}")
        if type(row.get("size_bytes")) is not int or row["size_bytes"] != artifact.stat().st_size:
            raise SmokeEvidenceError(f"entry inventory size binding mismatch: {name}")
        _strict_sha(row.get("sha256"), f"entry inventory sha256: {name}")
        if row["sha256"] != sha256_file(artifact):
            raise SmokeEvidenceError(f"entry inventory binding mismatch: {name}")
        names.add(name)
    actual = {path.name for path in root.iterdir() if path.name not in ENTRY_INVENTORY_EXCLUDED}
    if actual != names:
        raise SmokeEvidenceError("entry inventory does not cover every output file")
    if not ENTRY_REQUIRED_ARTIFACTS.issubset(names):
        raise SmokeEvidenceError("entry inventory is missing required evidence")
    config_row = next(row for row in rows if row["relative_path"] == "config.json")
    if config_row["sha256"] != config_sha256 or invocation.get("config_sha256") != config_sha256:
        raise SmokeEvidenceError("entry config SHA binding is incomplete")
    if (root / "error.json").exists() or (root / "partial_inventory.json").exists():
        raise SmokeEvidenceError("successful entry contains failure evidence")
    return {
        "entry_success_accepted": True,
        "completion": file_ref(root, "completion.json"),
        "artifact_inventory": file_ref(root, "artifact_inventory.json"),
        "config": file_ref(root, "config.json"),
    }


class SmokeEvidence:
    """Child-owned output directory with atomic config and terminal evidence."""

    def __init__(self, output_dir: Path) -> None:
        if not output_dir.is_absolute():
            raise SmokeEvidenceError("smoke output directory must be absolute")
        if output_dir.exists() or output_dir.is_symlink():
            raise SmokeEvidenceError("smoke output directory already exists")
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir()
        self.root = output_dir
        self._config_written = False

    def write_json(self, name: str, value: Any) -> str:
        if "/" in name or "\\" in name or name in {"artifact_inventory.json", "completion.json"}:
            raise SmokeEvidenceError(f"invalid evidence filename: {name}")
        payload = canonical_json_bytes(value)
        atomic_write(self.root / name, payload)
        return sha256_bytes(payload)

    def write_config(self, value: Any) -> str:
        if self._config_written:
            raise SmokeEvidenceError("config.json may only be written once")
        digest = self.write_json("config.json", value)
        self._config_written = True
        return digest

    def finalize_success(self, completion: dict[str, Any]) -> dict[str, object]:
        if not self._config_written:
            raise SmokeEvidenceError("config.json must be written before completion")
        inventory_value = inventory(self.root, ENTRY_INVENTORY_EXCLUDED)
        inventory_bytes = canonical_json_bytes(inventory_value)
        atomic_write(self.root / "artifact_inventory.json", inventory_bytes)
        terminal = dict(completion)
        terminal["artifact_inventory_sha256"] = sha256_bytes(inventory_bytes)
        atomic_write(self.root / "completion.json", canonical_json_bytes(terminal))
        return validate_entry_output(self.root)

    def finalize_failure(self, error: BaseException, completion: dict[str, Any]) -> None:
        if not (self.root / "error.json").exists():
            self.write_json("error.json", {
                "schema_version": 1,
                "status": "FAILED",
                "exception_type": type(error).__name__,
                "message": str(error),
                "original_exception_preserved": True,
            })
        partial = inventory(self.root, ENTRY_PARTIAL_EXCLUDED)
        if not (self.root / "partial_inventory.json").exists():
            atomic_write(self.root / "partial_inventory.json", canonical_json_bytes(partial))
        terminal = dict(completion)
        terminal.update({
            "status": "FAILED",
            "artifact_inventory_sha256": None,
            "partial_inventory_sha256": sha256_file(self.root / "partial_inventory.json"),
        })
        if not (self.root / "completion.json").exists():
            atomic_write(self.root / "completion.json", canonical_json_bytes(terminal))
