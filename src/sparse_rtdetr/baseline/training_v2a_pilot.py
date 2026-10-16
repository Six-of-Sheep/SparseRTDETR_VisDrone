"""Detached T7H loss-repair diagnostic-pilot contract and CPU fake adapter.

The module is deliberately inert when imported.  It validates the checked-in
T7H contract, binds the frozen T7G/T7A identities, and delegates fake epoch
mechanics to the already-certified T7G production implementation only after
all synthetic evidence inputs have been validated.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import inspect
import json
import math
import os
import pathlib
import shutil
import stat
import tempfile
from typing import Any, Mapping

from sparse_rtdetr.baseline import training_v2a_contract as _parent_contract


PILOT_SCHEMA_VERSION = 1
PILOT_ID = "rtdetrv2_r18_visdrone_baseline_v2a_t7h_loss_repair_pilot_r1"
PILOT_CONTRACT_RELATIVE_PATH = "configs/baseline/rtdetrv2_r18_visdrone_baseline_v2a_t7h_pilot.json"
PARENT_CONFIG_RELATIVE_PATH = "configs/baseline/rtdetrv2_r18_visdrone_baseline_v2a.json"
PARENT_CONTRACT_ID = "rtdetrv2_r18_visdrone_baseline_training_v2a"
PARENT_BASELINE_ID = "rtdetrv2_r18_visdrone_baseline_v2a"
PILOT_STAGE = "T7H_LOSS_REPAIR_DIAGNOSTIC_PILOT"
TARGET_RELATIVE_PATH = f"artifacts/training/{PILOT_ID}"
PILOT_MODULE_RELATIVE_PATH = "src/sparse_rtdetr/baseline/training_v2a_pilot.py"
PILOT_MODE = "cpu_fake"
SUCCESS_STATUS = "T7H_CPU_FAKE_COMPLETE"
TERMINAL_KIND = "T7H_V2A_LOSS_REPAIR_PILOT_RESULT"
CHECKPOINT_EPOCHS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
ENCODER_PARAMETER_NAMES = (
    "decoder.enc_output.norm.weight",
    "decoder.enc_output.norm.bias",
    "decoder.enc_output.proj.weight",
    "decoder.enc_output.proj.bias",
    "decoder.enc_score_head.weight",
    "decoder.enc_score_head.bias",
    "decoder.enc_bbox_head.layers.0.weight",
    "decoder.enc_bbox_head.layers.0.bias",
    "decoder.enc_bbox_head.layers.1.weight",
    "decoder.enc_bbox_head.layers.1.bias",
    "decoder.enc_bbox_head.layers.2.weight",
    "decoder.enc_bbox_head.layers.2.bias",
)
BASE_LOSS_KEYS = {"loss_vfl", "loss_bbox", "loss_giou"}
LOSS_FAMILY_COUNTS = {"base": 3, "aux": 6, "dn": 6, "encoder": 6}
T7G_SOURCE_IDENTITIES = {
    "src/sparse_rtdetr/baseline/training_v2a_production.py": (100085, "c3bbc7738e2db12c6f47b93d8d87d80d6d0766b711033382b8ce28a6933edf10"),
    "tests/test_rtdetr_baseline_training_v2a_production.py": (17453, "3880b71f50f013003b8d85384b7e456f17b676c7cee52fced485e1aef6e447ab"),
    "docs/contracts/RTDETR_BASELINE_V2A_PRODUCTION_BOUNDARY_T7D.md": (4954, "4d487ae89dc382ba5b12f55d1980572a0f5b06d3010a1d4e9630fc48c3aa5d23"),
}
T7G_WEIGHTED_LOSS_SOURCE_SHA256 = "c50be6f2ed177c8c0a97a1d1bde8cb2675ac5a60d293abf3607b37fb4d8d0bf2"
T7G_HEAD = "476e224cadebf8787a10c38dd9a0823808b56d48"
T7G_TREE = "cf85724186e9c7180f2f6f00cdc400d31c145ee9"

TARGET_NAMES = {
    "lock": "t7h_loss_repair_pilot_target_claim.lock",
    "evidence": "t7h_loss_repair_pilot_execution_evidence.json",
    "progress": "t7h_loss_repair_pilot_epoch_progress.jsonl",
    "terminal": "t7h_loss_repair_pilot_terminal_result.json",
    "stdout": "t7h_loss_repair_pilot_stdout.bin",
    "stderr": "t7h_loss_repair_pilot_stderr.bin",
    "checkpoint_prefix": "t7h_loss_repair_pilot_checkpoint_",
}

CONTRACT_KEYS = {
    "schema_version", "contract_id", "baseline_id", "stage", "parent_binding", "t7g_binding",
    "authority_binding", "overlay", "requirements", "loss_policy", "evidence_policy", "readiness",
}
PARENT_BINDING_KEYS = {
    "relative_path", "contract_id", "baseline_id", "raw_size_bytes", "raw_sha256",
    "canonical_size_bytes", "canonical_sha256", "parent_delta_allowlist",
}
T7G_BINDING_KEYS = {
    "head_sha", "tree_sha", "parent_sha", "subject", "production_module", "production_tests",
    "production_document", "weighted_loss_source_sha256", "independent_audit_pass",
}
AUTHORITY_BINDING_KEYS = {
    "root_relative_path", "artifact_id", "weight_relative_path", "weight_size_bytes", "weight_sha256",
    "manifest_relative_path", "manifest_size_bytes", "manifest_sha256",
}
OVERLAY_KEYS = {
    "run_namespace", "stage_namespace", "artifact_namespace", "session_namespace", "executed_epochs",
    "required_complete_epochs", "checkpoint_epochs", "development_evaluation_epochs", "augmentation_stop_epoch",
    "lr_milestones", "terminal_classification", "development_metric_role", "model_selection_certified",
    "training_ready", "diagnostic_complete", "independent_audit_pass",
}
REQUIREMENT_KEYS = {
    "fresh_run", "resume_forbidden", "retry_forbidden", "confirmatory_access_forbidden", "test_access_forbidden",
    "model_selection_forbidden", "seed_selection_forbidden", "hyperparameter_selection_forbidden",
    "runtime_batch_adaptation_forbidden", "random_initialization_forbidden", "external_checkpoint_forbidden",
    "network_download_forbidden", "owner_authorization_creation_forbidden", "owner_authorization_consumption_forbidden",
    "gpu_probe_forbidden", "tmux_forbidden", "training_forbidden_in_this_stage", "real_data_access_forbidden",
}
LOSS_POLICY_KEYS = {
    "aggregation_owner", "aggregation_source_sha256", "preweighted_terms", "term_count", "family_counts",
    "aggregate_exactly_once", "weight_dict_read_forbidden", "key_name_filtering_forbidden",
    "encoder_parameter_count", "encoder_gradient_nonzero_finite_required", "encoder_optimizer_state_after_first_update_required",
}
EVIDENCE_POLICY_KEYS = {
    "per_epoch_records_required", "total_loss_required", "family_subtotals_required", "family_recomputation_required",
    "development_ap_recorded", "development_ap_threshold", "checkpoint_every_epoch", "evaluation_every_epoch",
    "checkpoint_selection_forbidden", "comparison_only",
}
READINESS_KEYS = {
    "contract_layer_implemented", "cpu_fake_verified", "archive_verified", "published", "independent_audit_pass",
    "owner_authorization_created", "production_launch_executed", "diagnostic_complete", "training_ready",
    "model_selection_certified", "test_access_ready", "confirmatory_metrics_accessed",
    "dataset_test_split_accessed_by_this_stage",
}
POLICY_KEYS = {
    "schema_version", "kind", "pilot_id", "repo_root", "contract", "parent_contract", "effective_parent_contract",
    "parent_contract_sha256", "contract_sha256", "runtime_binding", "t7g_binding", "authority_binding",
    "source_identity", "target", "model", "topology", "optimizer", "amp", "ema", "selection", "checkpoint",
    "data_roles", "loss_policy", "evidence_policy", "production", "policy_sha256",
}
DESCRIPTOR_KEYS = {
    "schema_version", "kind", "pilot_id", "mode", "run_id", "session_namespace", "repo_root", "target_root",
    "target_relative_path", "contract_id", "baseline_id", "policy_sha256", "authorization_required",
    "authorization_created", "authorization_consumed", "data_accessed", "gpu_started", "tmux_started",
    "training_started", "aggregate_sha256",
}
RESULT_KEYS = {
    "schema_version", "kind", "status", "mode", "pilot_id", "run_id", "session_namespace", "policy_sha256",
    "target_root", "epochs", "checkpoint_epochs", "evaluation_epochs", "progress_inventory",
    "checkpoint_inventory", "loss_records", "encoder_gradient_evidence", "encoder_optimizer_state_evidence",
    "development_ap_role", "terminal_classification", "production", "forbidden_operations", "call_order",
    "result_sha256",
}
PROGRESS_ROW_KEYS = {
    "schema_version", "production_id", "run_id", "epoch", "batch_count", "mean_loss", "optimizer_updates",
    "evaluation", "selected_epoch", "checkpoint_sha256", "loss_total", "loss_family_subtotals",
}

__all__ = (
    "T7HPilotError",
    "canonical_t7h_pilot_bytes",
    "load_t7h_pilot_contract",
    "validate_t7h_pilot_contract",
    "build_t7h_pilot_policy",
    "validate_t7h_pilot_policy",
    "build_t7h_pilot_descriptor",
    "validate_t7h_pilot_descriptor",
    "validate_t7h_pilot_result",
    "run_t7h_cpu_fake",
)


class T7HPilotError(ValueError):
    """Raised when the detached T7H contract or fake evidence is invalid."""


def _fail(message: str) -> None:
    raise T7HPilotError(message)


def _copy(value: Any) -> Any:
    return copy.deepcopy(value)


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
        _fail(f"{field} is not builtin JSON data")


def _canonical(value: Any) -> bytes:
    _assert_builtin(value)
    try:
        return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise T7HPilotError("value is not canonical JSON") from exc


def canonical_t7h_pilot_bytes(value: Any) -> bytes:
    """Return canonical T7H JSON without a trailing newline."""

    checked = validate_t7h_pilot_contract(value)
    return _canonical(checked)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest(value: Any) -> str:
    return _sha_bytes(_canonical(value))


def _exact(value: Any, keys: set[str], field: str) -> dict[str, Any]:
    if type(value) is not dict:
        _fail(f"{field} must be a builtin object")
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


def _git(value: Any, field: str) -> str:
    value = _string(value, field)
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        _fail(f"{field} is not a lowercase Git object ID")
    return value


def _relative(value: Any, field: str) -> str:
    value = _string(value, field)
    if value.startswith("/") or "\\" in value or "//" in value or "\x00" in value:
        _fail(f"{field} is not a safe relative path")
    if any(part in {"", ".", ".."} for part in value.split("/")):
        _fail(f"{field} contains an unsafe path component")
    return value


def _path(value: Any, field: str) -> pathlib.Path:
    raw = _string(value, field)
    if not raw.startswith("/") or "\\" in raw or "//" in raw or "\x00" in raw:
        _fail(f"{field} is not an absolute path")
    if any(part in {"", ".", ".."} for part in raw.split("/")[1:]):
        _fail(f"{field} contains an unsafe path component")
    return pathlib.Path(raw)


def _finite_number(value: Any, field: str) -> float:
    if type(value) not in {int, float} or type(value) is bool or not math.isfinite(float(value)):
        _fail(f"{field} must be finite")
    return float(value)


def _int_list(value: Any, expected: tuple[int, ...], field: str) -> list[int]:
    if type(value) is not list or tuple(value) != expected or any(type(item) is not int for item in value):
        _fail(f"{field} must be the contiguous sequence {list(expected)}")
    return value


def _validate_identity(value: Any, field: str) -> dict[str, Any]:
    item = _exact(value, {"relative_path", "size_bytes", "sha256"}, field)
    _relative(item["relative_path"], f"{field}.relative_path")
    _integer(item["size_bytes"], f"{field}.size_bytes", minimum=1)
    _sha(item["sha256"], f"{field}.sha256")
    return item


def _validate_contract(value: Any) -> dict[str, Any]:
    value = _exact(value, CONTRACT_KEYS, "T7H contract")
    _assert_builtin(value, "T7H contract")
    if value["schema_version"] != PILOT_SCHEMA_VERSION or value["contract_id"] != PILOT_ID or value["baseline_id"] != PARENT_BASELINE_ID or value["stage"] != PILOT_STAGE:
        _fail("T7H contract identity drift")

    parent = _exact(value["parent_binding"], PARENT_BINDING_KEYS, "T7H parent binding")
    if parent["relative_path"] != PARENT_CONFIG_RELATIVE_PATH or parent["contract_id"] != PARENT_CONTRACT_ID or parent["baseline_id"] != PARENT_BASELINE_ID:
        _fail("T7H parent identity drift")
    _integer(parent["raw_size_bytes"], "T7H parent raw size", minimum=1)
    _sha(parent["raw_sha256"], "T7H parent raw SHA")
    _integer(parent["canonical_size_bytes"], "T7H parent canonical size", minimum=1)
    _sha(parent["canonical_sha256"], "T7H parent canonical SHA")
    if parent["raw_size_bytes"] != 12751 or parent["raw_sha256"] != "40800725fa0409ac440f463d1df08ce647714aa7fa715a05f7e98878403ae407" or parent["canonical_size_bytes"] != 10198 or parent["canonical_sha256"] != "96db5c69286775fb2b2b6a1a6997e58fe9d6019040eeca7328c02c7c2426ab0e":
        _fail("T7H parent baseline-v2a identity drift")
    if parent["parent_delta_allowlist"] != ["/schedule/epochs", "/acceptance/required_epochs_complete", "/checkpoint_policy/final_checkpoint"]:
        _fail("T7H parent-delta allowlist drift")

    t7g = _exact(value["t7g_binding"], T7G_BINDING_KEYS, "T7H T7G binding")
    _git(t7g["head_sha"], "T7H T7G head")
    _git(t7g["tree_sha"], "T7H T7G tree")
    _git(t7g["parent_sha"], "T7H T7G parent")
    _string(t7g["subject"], "T7H T7G subject")
    if (t7g["head_sha"], t7g["tree_sha"], t7g["parent_sha"], t7g["subject"], t7g["independent_audit_pass"]) != (T7G_HEAD, T7G_TREE, "c1efd152897bd478e97b1faf66413e959eb4139c", "fix(training): preserve all preweighted v2a losses", True):
        _fail("T7H frozen T7G Git identity drift")
    for field, relative in (("production_module", "src/sparse_rtdetr/baseline/training_v2a_production.py"), ("production_tests", "tests/test_rtdetr_baseline_training_v2a_production.py"), ("production_document", "docs/contracts/RTDETR_BASELINE_V2A_PRODUCTION_BOUNDARY_T7D.md")):
        item = _validate_identity(t7g[field], f"T7H T7G {field}")
        if item["relative_path"] != relative or (item["size_bytes"], item["sha256"]) != T7G_SOURCE_IDENTITIES[relative]:
            _fail(f"T7H T7G {field} identity drift")
    _sha(t7g["weighted_loss_source_sha256"], "T7H weighted-loss source SHA")
    if t7g["weighted_loss_source_sha256"] != T7G_WEIGHTED_LOSS_SOURCE_SHA256:
        _fail("T7H weighted-loss source binding drift")
    _bool(t7g["independent_audit_pass"], "T7H T7G independent audit flag")

    authority = _exact(value["authority_binding"], AUTHORITY_BINDING_KEYS, "T7H authority binding")
    for field in ("root_relative_path", "weight_relative_path", "manifest_relative_path"):
        _relative(authority[field], f"T7H authority.{field}")
    _string(authority["artifact_id"], "T7H authority.artifact_id")
    _integer(authority["weight_size_bytes"], "T7H authority weight size", minimum=1)
    _sha(authority["weight_sha256"], "T7H authority weight SHA")
    _integer(authority["manifest_size_bytes"], "T7H authority manifest size", minimum=1)
    _sha(authority["manifest_sha256"], "T7H authority manifest SHA")
    if authority != {
        "root_relative_path": "artifacts/pretrained_rtdetrv2_presnet18_imagenet_v1",
        "artifact_id": "rtdetrv2_r18_visdrone_presnet18_imagenet_v1",
        "weight_relative_path": "artifacts/pretrained_rtdetrv2_presnet18_imagenet_v1/ResNet18_vd_pretrained_from_paddle.pth",
        "weight_size_bytes": 44878642,
        "weight_sha256": _parent_contract.AUTHORITY_WEIGHT_SHA256,
        "manifest_relative_path": "artifacts/pretrained_rtdetrv2_presnet18_imagenet_v1/pretrained_authority.json",
        "manifest_size_bytes": 1608,
        "manifest_sha256": _parent_contract.AUTHORITY_MANIFEST_SHA256,
    }:
        _fail("T7H T7A authority identity drift")

    overlay = _exact(value["overlay"], OVERLAY_KEYS, "T7H overlay")
    for field in ("run_namespace", "stage_namespace", "artifact_namespace", "session_namespace", "terminal_classification", "development_metric_role"):
        _string(overlay[field], f"T7H overlay.{field}")
    if (overlay["run_namespace"], overlay["stage_namespace"], overlay["artifact_namespace"], overlay["session_namespace"], overlay["terminal_classification"], overlay["development_metric_role"]) != ("t7h_loss_repair_pilot_r1", PILOT_STAGE, PILOT_ID, "t7h_loss_repair_pilot_r1", "DIAGNOSTIC_ONLY", "comparison_only_not_acceptance_threshold"):
        _fail("T7H namespace or terminal policy drift")
    for field in ("executed_epochs", "required_complete_epochs", "augmentation_stop_epoch"):
        _integer(overlay[field], f"T7H overlay.{field}", minimum=1)
    if overlay["executed_epochs"] != 10 or overlay["required_complete_epochs"] != 10 or overlay["augmentation_stop_epoch"] != 117:
        _fail("T7H 10-epoch or augmentation policy drift")
    _int_list(overlay["checkpoint_epochs"], CHECKPOINT_EPOCHS, "T7H checkpoint epochs")
    _int_list(overlay["development_evaluation_epochs"], CHECKPOINT_EPOCHS, "T7H evaluation epochs")
    if overlay["lr_milestones"] != [1000] or any(type(item) is not int for item in overlay["lr_milestones"]):
        _fail("T7H learning-rate milestone drift")
    for field in ("model_selection_certified", "training_ready", "diagnostic_complete", "independent_audit_pass"):
        if _bool(overlay[field], f"T7H overlay.{field}") is not False:
            _fail(f"T7H overlay.{field} must remain false")

    requirements = _exact(value["requirements"], REQUIREMENT_KEYS, "T7H requirements")
    if any(_bool(requirements[field], f"T7H requirements.{field}") is not expected for field, expected in {
        "fresh_run": True,
        "resume_forbidden": True,
        "retry_forbidden": True,
        "confirmatory_access_forbidden": True,
        "test_access_forbidden": True,
        "model_selection_forbidden": True,
        "seed_selection_forbidden": True,
        "hyperparameter_selection_forbidden": True,
        "runtime_batch_adaptation_forbidden": True,
        "random_initialization_forbidden": True,
        "external_checkpoint_forbidden": True,
        "network_download_forbidden": True,
        "owner_authorization_creation_forbidden": True,
        "owner_authorization_consumption_forbidden": True,
        "gpu_probe_forbidden": True,
        "tmux_forbidden": True,
        "training_forbidden_in_this_stage": True,
        "real_data_access_forbidden": True,
    }.items()):
        _fail("T7H forbidden-operation policy drift")

    loss = _exact(value["loss_policy"], LOSS_POLICY_KEYS, "T7H loss policy")
    if loss["aggregation_owner"] != "src/sparse_rtdetr/baseline/training_v2a_production.py::_weighted_loss" or loss["aggregation_source_sha256"] != T7G_WEIGHTED_LOSS_SOURCE_SHA256:
        _fail("T7H loss aggregation owner drift")
    if loss["family_counts"] != LOSS_FAMILY_COUNTS or loss["term_count"] != 21 or loss["encoder_parameter_count"] != 12:
        _fail("T7H loss family cardinality drift")
    for field in ("preweighted_terms", "aggregate_exactly_once", "weight_dict_read_forbidden", "key_name_filtering_forbidden", "encoder_gradient_nonzero_finite_required", "encoder_optimizer_state_after_first_update_required"):
        if _bool(loss[field], f"T7H loss.{field}") is not True:
            _fail(f"T7H loss.{field} must be true")

    evidence = _exact(value["evidence_policy"], EVIDENCE_POLICY_KEYS, "T7H evidence policy")
    if evidence["family_subtotals_required"] != ["base", "aux", "dn", "encoder"] or evidence["development_ap_threshold"] is not None:
        _fail("T7H evidence family or threshold policy drift")
    for field in ("per_epoch_records_required", "total_loss_required", "family_recomputation_required", "development_ap_recorded", "checkpoint_every_epoch", "evaluation_every_epoch", "checkpoint_selection_forbidden", "comparison_only"):
        if _bool(evidence[field], f"T7H evidence.{field}") is not True:
            _fail(f"T7H evidence.{field} must be true")

    readiness = _exact(value["readiness"], READINESS_KEYS, "T7H readiness")
    expected_readiness = {field: False for field in READINESS_KEYS}
    expected_readiness["contract_layer_implemented"] = True
    for field, expected in expected_readiness.items():
        if _bool(readiness[field], f"T7H readiness.{field}") is not expected:
            _fail(f"T7H readiness.{field} drift")
    return _copy(value)


def validate_t7h_pilot_contract(value: Any) -> dict[str, Any]:
    """Validate the closed, detached T7H JSON contract."""

    return _validate_contract(value)


def _regular_identity(path: pathlib.Path, field: str, *, modes: set[int] | None = None) -> dict[str, Any]:
    try:
        info = path.lstat()
        raw = path.read_bytes()
    except OSError as exc:
        raise T7HPilotError(f"{field} identity unavailable") from exc
    if path.is_symlink() or not path.is_file() or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        _fail(f"{field} is not a regular unique file")
    mode = stat.S_IMODE(info.st_mode)
    if modes is not None and mode not in modes:
        _fail(f"{field} mode drift")
    return {"relative_path": field, "size_bytes": len(raw), "sha256": _sha_bytes(raw), "mode": mode, "uid": info.st_uid, "gid": info.st_gid, "nlink": info.st_nlink}


def _canonical_directory(value: Any, field: str) -> pathlib.Path:
    path = _path(os.fspath(value), field)
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise T7HPilotError(f"{field} is unavailable") from exc
    if resolved != path or not path.is_dir() or path.is_symlink():
        _fail(f"{field} is not a canonical directory")
    return path


def _read_json(path: pathlib.Path, field: str, *, config: bool = False) -> dict[str, Any]:
    _regular_identity(path, field, modes={0o644, 0o664} if config else None)
    try:
        raw = path.read_bytes()
        if not raw.endswith(b"\n") or raw.endswith(b"\n\n") or b"\r" in raw or b"\x00" in raw:
            _fail(f"{field} has non-portable bytes")
        value = json.loads(raw[:-1].decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise T7HPilotError(f"{field} is not valid JSON") from exc
    if type(value) is not dict or _canonical(value) != raw[:-1]:
        _fail(f"{field} is not canonical JSON")
    return value


def load_t7h_pilot_contract(repo_root: str | os.PathLike[str]) -> dict[str, Any]:
    """Load and validate the checked-in canonical T7H contract."""

    root = _canonical_directory(repo_root, "repo_root")
    return validate_t7h_pilot_contract(_read_json(root / PILOT_CONTRACT_RELATIVE_PATH, "T7H pilot contract", config=True))


def _file_identity(root: pathlib.Path, relative: str) -> dict[str, Any]:
    path = root / relative
    try:
        info = path.lstat()
        raw = path.read_bytes()
    except OSError as exc:
        raise T7HPilotError(f"source identity unavailable: {relative}") from exc
    if path.is_symlink() or not path.is_file() or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        _fail(f"source identity is not a regular unique file: {relative}")
    return {"relative_path": relative, "size_bytes": len(raw), "sha256": _sha_bytes(raw)}


def _load_parent(root: pathlib.Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    parent = _parent_contract.load_training_v2a_contract(root)
    raw = (root / PARENT_CONFIG_RELATIVE_PATH).read_bytes()
    binding = contract["parent_binding"]
    canonical = _parent_contract.canonical_training_v2a_contract_bytes(parent)
    if len(raw) != binding["raw_size_bytes"] or _sha_bytes(raw) != binding["raw_sha256"] or len(canonical) != binding["canonical_size_bytes"] or _sha_bytes(canonical) != binding["canonical_sha256"]:
        _fail("T7H parent baseline-v2a identity does not match the frozen binding")
    if parent["contract_id"] != PARENT_CONTRACT_ID or parent["baseline_id"] != PARENT_BASELINE_ID:
        _fail("T7H parent baseline-v2a semantic identity drift")
    return parent


def _check_t7g_sources(root: pathlib.Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    t7g = contract["t7g_binding"]
    observed = {}
    for field in ("production_module", "production_tests", "production_document"):
        item = t7g[field]
        identity = _file_identity(root, item["relative_path"])
        if identity["size_bytes"] != item["size_bytes"] or identity["sha256"] != item["sha256"]:
            _fail(f"T7G source identity drift: {item['relative_path']}")
        observed[field] = identity
    module = importlib.import_module("sparse_rtdetr.baseline.training_v2a_production")
    source = inspect.getsource(module._weighted_loss)
    if _sha_bytes(source.encode("utf-8")) != t7g["weighted_loss_source_sha256"]:
        _fail("T7G weighted-loss source identity drift")
    return observed


def _check_authority(root: pathlib.Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    binding = _parent_contract.training_v2a_contract_binding(root)
    authority = contract["authority_binding"]
    observed = binding["authority_binding"]
    if observed["root_relative_path"] != authority["root_relative_path"]:
        _fail("T7A authority semantic identity drift")
    for name, size_key, sha_key in (("weight", "weight_size_bytes", "weight_sha256"), ("manifest", "manifest_size_bytes", "manifest_sha256")):
        if observed[name]["size_bytes"] != authority[size_key] or observed[name]["sha256"] != authority[sha_key]:
            _fail(f"T7A authority {name} identity drift")
    return {
        "root_relative_path": observed["root_relative_path"],
        "artifact_id": authority["artifact_id"],
        "weight_relative_path": authority["weight_relative_path"],
        "weight_size_bytes": observed["weight"]["size_bytes"],
        "weight_sha256": observed["weight"]["sha256"],
        "manifest_relative_path": authority["manifest_relative_path"],
        "manifest_size_bytes": observed["manifest"]["size_bytes"],
        "manifest_sha256": observed["manifest"]["sha256"],
    }


def _effective_parent(parent: Mapping[str, Any]) -> dict[str, Any]:
    value = _copy(parent)
    value["schedule"]["epochs"] = 10
    value["acceptance"]["required_epochs_complete"] = 10
    value["checkpoint_policy"]["final_checkpoint"] = "epoch 10"
    return value


def _changed_pointers(left: Any, right: Any, pointer: str = "") -> set[str]:
    if type(left) is dict and type(right) is dict and set(left) == set(right):
        result: set[str] = set()
        for key in left:
            result |= _changed_pointers(left[key], right[key], f"{pointer}/{key}")
        return result
    if type(left) is list and type(right) is list and len(left) == len(right):
        result = set()
        for index, (lchild, rchild) in enumerate(zip(left, right)):
            result |= _changed_pointers(lchild, rchild, f"{pointer}/{index}")
        return result
    return set() if left == right and type(left) is type(right) else {pointer or "/"}


def _target_path(root: pathlib.Path, target_root: Any) -> tuple[pathlib.Path, str | None]:
    target = root / TARGET_RELATIVE_PATH if target_root is None else _path(os.fspath(target_root), "target_root")
    try:
        parent = target.parent.resolve(strict=True)
    except OSError as exc:
        raise T7HPilotError("target parent is unavailable") from exc
    if parent != target.parent or not target.parent.is_dir() or target.exists() or target.is_symlink():
        _fail("T7H target must be absent under a canonical parent")
    if any(token in str(target).casefold() for token in ("training_t6", "baseline_v1", "runtime_t7c", "production_t7d")):
        _fail("T7H target overlaps an earlier target")
    return target, TARGET_RELATIVE_PATH if target == root / TARGET_RELATIVE_PATH else None


def _pilot_source_identity(root: pathlib.Path) -> dict[str, Any]:
    return {"pilot_module": _file_identity(root, PILOT_MODULE_RELATIVE_PATH)}


def validate_t7h_pilot_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a detached policy and its complete parent-delta closure."""

    value = _exact(policy, POLICY_KEYS, "T7H pilot policy")
    _assert_builtin(value, "T7H pilot policy")
    if value["schema_version"] != PILOT_SCHEMA_VERSION or value["kind"] != "T7H_V2A_LOSS_REPAIR_PILOT_POLICY" or value["pilot_id"] != PILOT_ID:
        _fail("T7H policy identity drift")
    _path(value["repo_root"], "T7H policy.repo_root")
    contract = validate_t7h_pilot_contract(value["contract"])
    parent = _parent_contract.validate_training_v2a_contract(value["parent_contract"])
    effective = value["effective_parent_contract"]
    _assert_builtin(effective, "T7H effective parent contract")
    if type(effective) is not dict:
        _fail("T7H effective parent contract must be an object")
    if _changed_pointers(parent, effective) != set(contract["parent_binding"]["parent_delta_allowlist"]):
        _fail("T7H effective parent delta is outside the allowlist")
    if effective["schedule"]["epochs"] != 10 or effective["acceptance"]["required_epochs_complete"] != 10 or effective["checkpoint_policy"]["final_checkpoint"] != "epoch 10":
        _fail("T7H effective parent overlay drift")
    if value["parent_contract_sha256"] != _digest(parent) or value["contract_sha256"] != _digest(contract):
        _fail("T7H policy contract digest drift")
    if value["runtime_binding"].get("contract_id") != PARENT_CONTRACT_ID or value["runtime_binding"].get("baseline_id") != PARENT_BASELINE_ID:
        _fail("T7H runtime binding parent identity drift")
    _exact(value["t7g_binding"], T7G_BINDING_KEYS, "T7H policy T7G binding")
    _exact(value["authority_binding"], AUTHORITY_BINDING_KEYS, "T7H policy authority binding")
    _validate_identity(value["source_identity"]["pilot_module"], "T7H policy pilot source")
    target = _exact(value["target"], {"path", "relative_path", "names"}, "T7H policy target")
    _path(target["path"], "T7H policy.target.path")
    if target["relative_path"] not in {TARGET_RELATIVE_PATH, None} or target["names"] != TARGET_NAMES:
        _fail("T7H policy target identity drift")
    inherited_fields = {
        "model": "model",
        "topology": "topology",
        "optimizer": "optimizer",
        "amp": "amp",
        "ema": "ema",
        "selection": "evaluation_and_selection",
        "checkpoint": "checkpoint_policy",
        "data_roles": "data_roles",
    }
    for policy_field, parent_field in inherited_fields.items():
        if value[policy_field] != effective[parent_field]:
            _fail(f"T7H inherited policy field drift: {policy_field}")
    if value["loss_policy"] != contract["loss_policy"] or value["evidence_policy"] != contract["evidence_policy"]:
        _fail("T7H policy evidence binding drift")
    expected_production = {"owner_authorization_created": False, "owner_authorization_consumed": False, "gpu_probe_executed": False, "tmux_started": False, "formal_training_executed": False, "training_ready": False, "independent_audit_pass": False, "diagnostic_complete": False}
    if value["production"] != expected_production:
        _fail("T7H policy readiness is not fail-closed")
    _sha(value["policy_sha256"], "T7H policy SHA")
    if value["policy_sha256"] != _digest({key: child for key, child in value.items() if key != "policy_sha256"}):
        _fail("T7H policy digest mismatch")
    return _copy(value)


def build_t7h_pilot_policy(repo_root: str | os.PathLike[str], *, target_root: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Validate the parent first, then apply only the closed 10-epoch overlay."""

    root = _canonical_directory(repo_root, "repo_root")
    contract = load_t7h_pilot_contract(root)
    parent = _load_parent(root, contract)
    effective = _effective_parent(parent)
    if _changed_pointers(parent, effective) != set(contract["parent_binding"]["parent_delta_allowlist"]):
        _fail("T7H parent overlay changed an unapproved field")
    t7g_sources = _check_t7g_sources(root, contract)
    authority = _check_authority(root, contract)
    target, relative = _target_path(root, target_root)
    policy = {
        "schema_version": PILOT_SCHEMA_VERSION,
        "kind": "T7H_V2A_LOSS_REPAIR_PILOT_POLICY",
        "pilot_id": PILOT_ID,
        "repo_root": str(root),
        "contract": contract,
        "parent_contract": _copy(parent),
        "effective_parent_contract": effective,
        "parent_contract_sha256": _digest(parent),
        "contract_sha256": _digest(contract),
        "runtime_binding": _parent_contract.training_v2a_contract_binding(root),
        "t7g_binding": _copy(contract["t7g_binding"]),
        "authority_binding": authority,
        "source_identity": {"pilot_module": _file_identity(root, PILOT_MODULE_RELATIVE_PATH), "t7g_sources": t7g_sources},
        "target": {"path": str(target), "relative_path": relative, "names": _copy(TARGET_NAMES)},
        "model": _copy(effective["model"]),
        "topology": _copy(effective["topology"]),
        "optimizer": _copy(effective["optimizer"]),
        "amp": _copy(effective["amp"]),
        "ema": _copy(effective["ema"]),
        "selection": _copy(effective["evaluation_and_selection"]),
        "checkpoint": _copy(effective["checkpoint_policy"]),
        "data_roles": _copy(effective["data_roles"]),
        "loss_policy": _copy(contract["loss_policy"]),
        "evidence_policy": _copy(contract["evidence_policy"]),
        "production": {"owner_authorization_created": False, "owner_authorization_consumed": False, "gpu_probe_executed": False, "tmux_started": False, "formal_training_executed": False, "training_ready": False, "independent_audit_pass": False, "diagnostic_complete": False},
    }
    policy["policy_sha256"] = _digest(policy)
    return validate_t7h_pilot_policy(policy)


def validate_t7h_pilot_descriptor(descriptor: Mapping[str, Any]) -> dict[str, Any]:
    value = _exact(descriptor, DESCRIPTOR_KEYS, "T7H pilot descriptor")
    _assert_builtin(value, "T7H pilot descriptor")
    if value["schema_version"] != PILOT_SCHEMA_VERSION or value["kind"] != "T7H_V2A_LOSS_REPAIR_PILOT_DESCRIPTOR" or value["pilot_id"] != PILOT_ID or value["mode"] != PILOT_MODE:
        _fail("T7H descriptor identity drift")
    for field in ("run_id", "session_namespace", "contract_id", "baseline_id"):
        _string(value[field], f"T7H descriptor.{field}")
    if value["run_id"] != "t7h_loss_repair_pilot_r1" or value["session_namespace"] != value["run_id"] or value["contract_id"] != PILOT_ID or value["baseline_id"] != PARENT_BASELINE_ID:
        _fail("T7H descriptor namespace drift")
    _path(value["repo_root"], "T7H descriptor.repo_root")
    target = _path(value["target_root"], "T7H descriptor.target_root")
    if value["target_relative_path"] not in {TARGET_RELATIVE_PATH, None}:
        _fail("T7H descriptor target relative path drift")
    _sha(value["policy_sha256"], "T7H descriptor.policy_sha256")
    for field in ("authorization_required", "authorization_created", "authorization_consumed", "data_accessed", "gpu_started", "tmux_started", "training_started"):
        _bool(value[field], f"T7H descriptor.{field}")
    if value["authorization_required"] is not True or any(value[field] is not False for field in ("authorization_created", "authorization_consumed", "data_accessed", "gpu_started", "tmux_started", "training_started")):
        _fail("T7H descriptor forbidden-operation state drift")
    _sha(value["aggregate_sha256"], "T7H descriptor aggregate SHA")
    if value["aggregate_sha256"] != _digest({key: child for key, child in value.items() if key != "aggregate_sha256"}):
        _fail("T7H descriptor digest drift")
    if target.exists() or target.is_symlink():
        _fail("T7H descriptor target must remain absent")
    return _copy(value)


def build_t7h_pilot_descriptor(policy: Mapping[str, Any]) -> dict[str, Any]:
    checked = validate_t7h_pilot_policy(policy)
    target = pathlib.Path(checked["target"]["path"])
    descriptor = {
        "schema_version": PILOT_SCHEMA_VERSION,
        "kind": "T7H_V2A_LOSS_REPAIR_PILOT_DESCRIPTOR",
        "pilot_id": PILOT_ID,
        "mode": PILOT_MODE,
        "run_id": checked["contract"]["overlay"]["run_namespace"],
        "session_namespace": checked["contract"]["overlay"]["session_namespace"],
        "repo_root": checked["repo_root"],
        "target_root": str(target),
        "target_relative_path": checked["target"]["relative_path"],
        "contract_id": PILOT_ID,
        "baseline_id": PARENT_BASELINE_ID,
        "policy_sha256": checked["policy_sha256"],
        "authorization_required": True,
        "authorization_created": False,
        "authorization_consumed": False,
        "data_accessed": False,
        "gpu_started": False,
        "tmux_started": False,
        "training_started": False,
    }
    descriptor["aggregate_sha256"] = _digest(descriptor)
    return validate_t7h_pilot_descriptor(descriptor)


def _scalar(value: Any, field: str) -> float:
    item = value
    detach = getattr(item, "detach", None)
    if callable(detach):
        item = detach()
    getter = getattr(item, "item", None)
    if callable(getter):
        item = getter()
    if type(item) not in {int, float} or type(item) is bool or not math.isfinite(float(item)):
        _fail(f"{field} is not a finite scalar")
    return float(item)


def _loss_family_subtotals(losses: Any) -> dict[str, float]:
    if type(losses) is not dict or len(losses) != 21 or set(losses) & {"weight_dict"}:
        _fail("CPU fake loss must contain exactly 21 preweighted terms")
    families = {"base": [], "aux": [], "dn": [], "encoder": []}
    for key, value in losses.items():
        if type(key) is not str:
            _fail("CPU fake loss key is not a string")
        if key in BASE_LOSS_KEYS:
            family = "base"
        elif "_aux_" in key:
            family = "aux"
        elif "_dn_" in key:
            family = "dn"
        elif "_enc_" in key:
            family = "encoder"
        else:
            _fail(f"CPU fake loss key is outside the four families: {key}")
        families[family].append(_scalar(value, f"loss.{key}"))
    if {key: len(items) for key, items in families.items()} != LOSS_FAMILY_COUNTS:
        _fail("CPU fake loss family cardinality drift")
    return {family: sum(items) for family, items in families.items()}


def _validate_gradient_evidence(value: Any) -> dict[str, Any]:
    checked = _exact(value, {"parameter_count", "rows"}, "encoder gradient evidence")
    if _integer(checked["parameter_count"], "encoder gradient parameter count", minimum=0) != 12:
        _fail("encoder gradient parameter count drift")
    rows = checked["rows"]
    if type(rows) is not dict or set(rows) != set(ENCODER_PARAMETER_NAMES):
        _fail("encoder gradient parameter identity drift")
    for name in ENCODER_PARAMETER_NAMES:
        row = _exact(rows[name], {"exists", "finite", "nonzero"}, f"encoder gradient.{name}")
        if any(_bool(row[field], f"encoder gradient.{name}.{field}") is not True for field in ("exists", "finite", "nonzero")):
            _fail(f"encoder gradient evidence is incomplete: {name}")
    return _copy(checked)


def _validate_optimizer_state_evidence(value: Any) -> dict[str, Any]:
    checked = _exact(value, {"parameter_count", "after_first_optimizer_update", "all_state_entries_present"}, "encoder optimizer state evidence")
    if _integer(checked["parameter_count"], "encoder optimizer parameter count", minimum=0) != 12:
        _fail("encoder optimizer parameter count drift")
    if _bool(checked["after_first_optimizer_update"], "encoder optimizer after-first-update evidence") is not True or _bool(checked["all_state_entries_present"], "encoder optimizer state entries evidence") is not True:
        _fail("encoder optimizer state evidence is incomplete")
    return _copy(checked)


def _validate_loss_records(rows: Any) -> list[dict[str, Any]]:
    if type(rows) is not list or len(rows) != 10:
        _fail("T7H loss record count drift")
    checked = []
    for expected_epoch, row in zip(CHECKPOINT_EPOCHS, rows):
        value = _exact(row, PROGRESS_ROW_KEYS, f"T7H loss record {expected_epoch}")
        if _integer(value["epoch"], "T7H loss record epoch", minimum=1) != expected_epoch:
            _fail("T7H loss record sequence drift")
        total = _finite_number(value["loss_total"], "T7H loss total")
        subtotals = _exact(value["loss_family_subtotals"], set(LOSS_FAMILY_COUNTS), "T7H loss family subtotals")
        for family in LOSS_FAMILY_COUNTS:
            _finite_number(subtotals[family], f"T7H loss family {family}")
        if not math.isclose(total, sum(float(subtotals[field]) for field in LOSS_FAMILY_COUNTS), rel_tol=0.0, abs_tol=1e-9) or not math.isclose(total, _finite_number(value["mean_loss"], "T7H mean loss"), rel_tol=0.0, abs_tol=1e-9):
            _fail("T7H loss family recomputation drift")
        evaluation = _exact(value["evaluation"], {"AP", "AP50", "AR500", "weights"}, "T7H development evaluation")
        for field in ("AP", "AP50", "AR500"):
            _finite_number(evaluation[field], f"T7H development.{field}")
        if evaluation["weights"] != "ema":
            _fail("T7H development evaluation is not EMA-backed")
        checked.append(_copy(value))
    return checked


def validate_t7h_pilot_result(result: Mapping[str, Any], policy: Mapping[str, Any]) -> dict[str, Any]:
    checked_policy = validate_t7h_pilot_policy(policy)
    value = _exact(result, RESULT_KEYS, "T7H pilot result")
    _assert_builtin(value, "T7H pilot result")
    if value["schema_version"] != PILOT_SCHEMA_VERSION or value["kind"] != TERMINAL_KIND or value["status"] != SUCCESS_STATUS or value["mode"] != PILOT_MODE or value["pilot_id"] != PILOT_ID:
        _fail("T7H pilot result identity drift")
    if value["run_id"] != checked_policy["contract"]["overlay"]["run_namespace"] or value["session_namespace"] != checked_policy["contract"]["overlay"]["session_namespace"] or value["policy_sha256"] != checked_policy["policy_sha256"]:
        _fail("T7H pilot result binding drift")
    _path(value["target_root"], "T7H result.target_root")
    if _integer(value["epochs"], "T7H result.epochs", minimum=1) != 10:
        _fail("T7H result epoch count drift")
    _int_list(value["checkpoint_epochs"], CHECKPOINT_EPOCHS, "T7H result checkpoint epochs")
    _int_list(value["evaluation_epochs"], CHECKPOINT_EPOCHS, "T7H result evaluation epochs")
    _validate_loss_records(value["loss_records"])
    _validate_gradient_evidence(value["encoder_gradient_evidence"])
    _validate_optimizer_state_evidence(value["encoder_optimizer_state_evidence"])
    if value["development_ap_role"] != "comparison_only_not_acceptance_threshold" or value["terminal_classification"] != "DIAGNOSTIC_ONLY":
        _fail("T7H diagnostic classification drift")
    production = _exact(value["production"], {"owner_authorization_created", "owner_authorization_consumed", "gpu_probe_executed", "tmux_started", "formal_training_executed", "training_ready", "independent_audit_pass", "diagnostic_complete"}, "T7H result production state")
    if any(_bool(production[field], f"T7H result.production.{field}") is not False for field in production):
        _fail("T7H result production state is not fail-closed")
    forbidden = _exact(value["forbidden_operations"], {"owner_authorization_created", "owner_authorization_consumed", "data_access", "gpu_cuda", "tmux", "training", "independent_audit"}, "T7H forbidden operation counters")
    if any(_integer(forbidden[field], f"T7H forbidden operation {field}", minimum=0) != 0 for field in forbidden):
        _fail("T7H forbidden operation counter drift")
    if type(value["call_order"]) is not list or not value["call_order"] or "epoch_loop" not in value["call_order"]:
        _fail("T7H fake call order evidence is incomplete")
    _sha(value["result_sha256"], "T7H result SHA")
    if value["result_sha256"] != _digest({key: child for key, child in value.items() if key != "result_sha256"}):
        _fail("T7H result digest drift")
    return _copy(value)


def _fake_environment() -> dict[str, Any]:
    return {"gpu_name": "NVIDIA GeForce RTX 4090 D", "cuda_version": "12.4", "graphics_clock_mhz": 1500, "exclusive_host": True, "other_training_load": 0, "observation_id": "t7h-cpu-fake-only"}


def _call_user_loss(function: Any, args: tuple[Any, ...], production: Any) -> Any:
    candidates = (args, args[1:], (args[0], args[1]), (args[1],), ())
    return production._call_port(function, candidates, "T7H CPU fake compute_loss")


def run_t7h_cpu_fake(policy: Mapping[str, Any], ports: Mapping[str, Any], *, epochs: int = 10) -> dict[str, Any]:
    """Run exactly ten synthetic epochs through T7G mechanics, never real training."""

    checked = validate_t7h_pilot_policy(policy)
    if _integer(epochs, "T7H CPU fake epochs", minimum=1) != 10:
        _fail("T7H CPU fake requires exactly 10 epochs")
    if type(ports) is not dict:
        _fail("T7H CPU fake ports must be a builtin dict")
    forbidden_ports = {"production_capability", "authorization_context", "real_data", "real_model", "data_loader", "filesystem", "evaluator_path"}
    if forbidden_ports & set(ports):
        _fail("T7H CPU fake cannot inject production capabilities")
    for required in ("compute_loss", "encoder_gradient_evidence", "encoder_optimizer_state_evidence"):
        if required not in ports:
            _fail(f"T7H CPU fake port is required: {required}")
    gradient = _validate_gradient_evidence(ports["encoder_gradient_evidence"])
    optimizer_state = _validate_optimizer_state_evidence(ports["encoder_optimizer_state_evidence"])
    if pathlib.Path(checked["target"]["path"]).is_relative_to(pathlib.Path(checked["repo_root"]) / "artifacts"):
        _fail("T7H CPU fake target must be detached from repository artifacts")
    target = pathlib.Path(checked["target"]["path"])
    family_rows: dict[int, list[dict[str, float]]] = {}
    original_loss = ports["compute_loss"]
    if not callable(original_loss):
        _fail("T7H CPU fake compute_loss is not callable")
    production = importlib.import_module("sparse_rtdetr.baseline.training_v2a_production")
    fixture_root = pathlib.Path(tempfile.mkdtemp(prefix="p3-t7h-cpu-fake-", dir="/tmp"))
    train_root = fixture_root / "train-core"
    development_root = fixture_root / "development"
    train_root.mkdir()
    development_root.mkdir()
    (train_root / "train_core_coco.json").write_bytes(b"{}")
    (development_root / "development_coco.json").write_bytes(b"{}")

    def wrapped_loss(*args: Any) -> Any:
        raw = _call_user_loss(original_loss, args, production)
        subtotals = _loss_family_subtotals(raw)
        epoch = args[2] if len(args) >= 3 and type(args[2]) is int else 1
        family_rows.setdefault(epoch, []).append(subtotals)
        return production._weighted_loss(None, raw)

    production_ports = dict(ports)
    production_ports["compute_loss"] = wrapped_loss
    production_ports["run_id"] = checked["contract"]["overlay"]["run_namespace"]
    saved = {name: getattr(production, name) for name in ("PRODUCTION_ID", "CPU_FAKE_MODE", "SUCCESS_STATUS", "TERMINAL_KIND", "CHECKPOINT_KIND", "EVIDENCE_KIND", "TARGET_CLAIM_KIND", "_target_names", "_append_progress")}

    def append_progress(path: pathlib.Path, row: Mapping[str, Any], expected_epoch: int, seen: set[int]) -> None:
        batches = family_rows.get(expected_epoch)
        if not batches:
            _fail("T7H family evidence was not observed before progress publication")
        count = len(batches)
        subtotals = {family: sum(item[family] for item in batches) / count for family in LOSS_FAMILY_COUNTS}
        enriched = dict(row)
        enriched["loss_total"] = float(row["mean_loss"])
        enriched["loss_family_subtotals"] = subtotals
        saved["_append_progress"](path, enriched, expected_epoch, seen)

    production.PRODUCTION_ID = PILOT_ID
    production.CPU_FAKE_MODE = PILOT_MODE
    production.SUCCESS_STATUS = SUCCESS_STATUS
    production.TERMINAL_KIND = "T7H_V2A_PROCESS_TERMINAL_RESULT"
    production.CHECKPOINT_KIND = "T7H_V2A_CHECKPOINT"
    production.EVIDENCE_KIND = "T7H_V2A_EXECUTION_EVIDENCE"
    production.TARGET_CLAIM_KIND = "T7H_V2A_TARGET_CLAIM"
    production._target_names = lambda: _copy(TARGET_NAMES)
    production._append_progress = append_progress
    target_absent_before = not target.exists()
    try:
        internal = production.build_v2a_production_policy(
            checked["repo_root"],
            data_roots={"train_core": str(train_root), "development": str(development_root)},
            environment_identity=_fake_environment(),
            target_root=str(target),
        )
        internal["production_id"] = PILOT_ID
        internal["target"]["names"] = _copy(TARGET_NAMES)
        internal["policy_sha256"] = production._digest({key: child for key, child in internal.items() if key != "policy_sha256"})
        raw_result = production.run_v2a_cpu_fake(internal, production_ports, epochs=epochs)
        progress_path = pathlib.Path(raw_result["progress_inventory"]["path"])
        progress_rows = []
        for line in progress_path.read_bytes().splitlines(keepends=True):
            if not line.endswith(b"\n"):
                _fail("T7H progress has an unterminated line")
            progress_rows.append(json.loads(line[:-1].decode("utf-8")))
        loss_records = _validate_loss_records(progress_rows)
        last_epochs = [row["epoch"] for row in raw_result["checkpoint_inventory"] if row["role"] == "last"]
        if last_epochs != list(CHECKPOINT_EPOCHS):
            _fail("T7H checkpoint sequence is not exactly 1..10")
        result = {
            "schema_version": PILOT_SCHEMA_VERSION,
            "kind": TERMINAL_KIND,
            "status": SUCCESS_STATUS,
            "mode": PILOT_MODE,
            "pilot_id": PILOT_ID,
            "run_id": checked["contract"]["overlay"]["run_namespace"],
            "session_namespace": checked["contract"]["overlay"]["session_namespace"],
            "policy_sha256": checked["policy_sha256"],
            "target_root": str(target),
            "epochs": 10,
            "checkpoint_epochs": list(CHECKPOINT_EPOCHS),
            "evaluation_epochs": list(CHECKPOINT_EPOCHS),
            "progress_inventory": _copy(raw_result["progress_inventory"]),
            "checkpoint_inventory": _copy(raw_result["checkpoint_inventory"]),
            "loss_records": loss_records,
            "encoder_gradient_evidence": gradient,
            "encoder_optimizer_state_evidence": optimizer_state,
            "development_ap_role": "comparison_only_not_acceptance_threshold",
            "terminal_classification": "DIAGNOSTIC_ONLY",
            "production": {"owner_authorization_created": False, "owner_authorization_consumed": False, "gpu_probe_executed": False, "tmux_started": False, "formal_training_executed": False, "training_ready": False, "independent_audit_pass": False, "diagnostic_complete": False},
            "forbidden_operations": {"owner_authorization_created": 0, "owner_authorization_consumed": 0, "data_access": 0, "gpu_cuda": 0, "tmux": 0, "training": 0, "independent_audit": 0},
            "call_order": _copy(raw_result["call_order"]),
        }
        result["result_sha256"] = _digest(result)
        return validate_t7h_pilot_result(result, checked)
    except Exception as exc:
        if target_absent_before and target.exists() and target.is_dir() and target.parent != pathlib.Path(checked["repo_root"]) / "artifacts":
            shutil.rmtree(target)
        if isinstance(exc, T7HPilotError):
            raise
        raise T7HPilotError(str(exc)) from exc
    finally:
        for name, value in saved.items():
            setattr(production, name, value)
        shutil.rmtree(fixture_root, ignore_errors=True)
