"""Frozen epoch-30 to epoch-60 continuation contracts.

This module is deliberately CPU-only and contains no model, CUDA, data-loader,
subprocess, or evaluator import.  The epoch-30 checkpoint keeps its original
run binding.  A continuation receives a separate execution identity and a
separately frozen orchestration inventory; it never relabels the historical
checkpoint as a product of the new campaign.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Mapping


SCHEMA_VERSION = 1
FREEZE_KIND = "v2b_epoch60_continuation_freeze"
CONTRACT_KIND = "v2b_epoch60_continuation_worker_contract"
RESULT_KIND = "v2b_epoch60_continuation_worker_result"
CELLS = ("s1_r640", "s1_r896", "s2_r640", "s2_r896")
STAGES = ("smoke", "smoke_replay", "formal60")
SOURCE_EPOCH = 30
TARGET_EPOCH = 60
BATCHES_PER_EPOCH = 304
SOURCE_UPDATES = 9120
TARGET_UPDATES = 18240
ADDITIONAL_UPDATES = TARGET_UPDATES - SOURCE_UPDATES
ORCHESTRATION_FILES = (
    "docs/contracts/RTDETR_BASELINE_V2B_EPOCH60_CONTINUATION_T8A.md",
    "src/sparse_rtdetr/baseline/training_v2b_epoch60_continuation.py",
    "tests/test_training_v2b_epoch60_continuation.py",
    "tests/test_training_v2b_epoch60_worker.py",
    "tools/run_training_v2b_epoch60_continuation_campaign.py",
    "tools/run_training_v2b_epoch60_continuation_worker.py",
    "tools/repository_contract_check.py",
)
EXECUTION_POLICY = {
    "source_checkpoint_immutable": True,
    "source_run_binding_reused_exactly": True,
    "outer_execution_identity_distinct": True,
    "historical_run_not_relabelled": True,
    "source_artifacts_never_overwritten": True,
    "training_core_semantics_change_forbidden": True,
    "resume_from_complete_epoch30_boundary": True,
    "continuation_first_epoch": 31,
    "continuation_last_epoch": 60,
    "fresh_zero_to60_required": False,
    "formal_retry_or_resume_after_failure": False,
    "automatic_batch_or_precision_fallback": False,
    "smoke_replay_required_before_formal": True,
    "development_endpoints": [45, 60],
    "confirmatory_or_test_access": False,
}
_DIGEST = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}")
_GPU_UUID = re.compile(
    r"GPU-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
    r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}"
)
WORKER_ENVIRONMENT_STATIC = {
    "MKL_THREADING_LAYER": "GNU",
    "PYTHONNOUSERSITE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "OMP_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "PYTHONHASHSEED": "0",
}
WORKER_CUDA_BACKEND_POLICY = {
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
    "cudnn_allow_tf32": False,
    "matmul_allow_tf32": False,
    "deterministic_algorithms": True,
    "deterministic_warn_only": False,
    "cublas_workspace_config": ":4096:8",
}


class Epoch60ContinuationError(RuntimeError):
    """A continuation contract, lineage, identity, or stage is invalid."""


def worker_startup_environment(expected_gpu_uuid: Any) -> dict[str, str]:
    """Return the complete frozen child-process environment overlay.

    This is the single authority used by both the campaign controller and the
    worker-side gate. It deliberately matches the environment used by the
    already-certified 30-epoch production campaigns.
    """
    _require(type(expected_gpu_uuid) is str
             and _GPU_UUID.fullmatch(expected_gpu_uuid) is not None,
             "worker GPU UUID must be complete")
    return {"CUDA_VISIBLE_DEVICES": expected_gpu_uuid, **WORKER_ENVIRONMENT_STATIC}


def validate_worker_process_environment(contract: Mapping[str, Any]) -> dict[str, str]:
    """Fail before project imports or output creation if startup drifted."""
    _require(isinstance(contract, Mapping), "worker contract environment is missing")
    expected = worker_startup_environment(contract.get("expected_gpu_uuid"))
    _same(contract.get("startup_environment"), expected,
          "worker declared startup environment")
    observed = {name: os.environ.get(name) for name in expected}
    _same(observed, expected, "worker process startup environment")
    _require(sys.dont_write_bytecode, "worker bytecode generation must be disabled at startup")
    return observed


def _require(value: Any, message: str) -> None:
    if not value:
        raise Epoch60ContinuationError(message)


def _typed_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(_typed_equal(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_typed_equal(a, b) for a, b in zip(left, right))
    return left == right


def _same(left: Any, right: Any, label: str) -> None:
    _require(_typed_equal(left, right), label + " drift")


def _schema(value: Any, keys: set[str], label: str) -> dict:
    _require(type(value) is dict and set(value) == keys, label + " schema drift")
    return value


def _identifier(value: Any, label: str) -> str:
    _require(type(value) is str and _IDENTIFIER.fullmatch(value) is not None,
             label + " must be a bounded identifier")
    return value


def _absolute_path(value: Any, label: str) -> Path:
    _require(type(value) is str and value != "", label + " must be a path string")
    path = Path(value)
    _require(path.is_absolute() and path == path.resolve(), label + " must be canonical and absolute")
    _require(not any(part.casefold() == "test" or part.casefold().startswith("confirmatory")
                     for part in path.parts), label + " crosses a forbidden data role")
    return path


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                     allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_reference(path: str | os.PathLike[str]) -> dict[str, Any]:
    source = Path(path).resolve()
    fd = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(fd)
        _require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1,
                 "authority must be a single-link regular file")
        digest = hashlib.sha256()
        total = 0
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            total += len(block)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    _require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
             == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
             "authority changed while it was read")
    _require(total == before.st_size, "authority short read")
    return {"path": str(source), "sha256": digest.hexdigest(), "size_bytes": total,
            "sha256_scope": "complete_file_bytes"}


def validate_reference(value: Any, label: str = "reference", *, verify: bool = False) -> dict:
    keys = {"path", "sha256", "size_bytes", "sha256_scope"}
    _schema(value, keys, label)
    path = _absolute_path(value["path"], label + " path")
    _require(type(value["sha256"]) is str and _DIGEST.fullmatch(value["sha256"]) is not None,
             label + " SHA-256 invalid")
    _require(type(value["size_bytes"]) is int and value["size_bytes"] >= 0,
             label + " size invalid")
    _same(value["sha256_scope"], "complete_file_bytes", label + " SHA scope")
    if verify:
        _same(file_reference(path), value, label + " current identity")
    return copy.deepcopy(value)


def _reference_core(value: Any, label: str) -> dict:
    """Normalize a historical checkpoint descriptor to its file identity.

    Checkpoint ledgers add schema/binding/epoch fields to the three stable file
    identity fields.  This presentation difference is not a byte-identity
    difference and was the already-reconciled T8A-R1 harness issue.
    """
    _require(type(value) is dict, label + " must be an object")
    core = {name: value.get(name) for name in ("path", "sha256", "size_bytes")}
    core["sha256_scope"] = value.get("sha256_scope", "complete_file_bytes")
    return validate_reference(core, label, verify=False)


def read_json_reference(value: Any, label: str, *, verify: bool = True) -> dict:
    reference = validate_reference(value, label, verify=verify)
    try:
        result = json.loads(Path(reference["path"]).read_text(encoding="utf-8"),
                            parse_constant=lambda token: (_ for _ in ()).throw(
                                ValueError("nonfinite JSON constant: " + token)))
    except Exception as exc:
        raise Epoch60ContinuationError(label + " is not strict JSON") from exc
    _require(type(result) is dict, label + " must contain one JSON object")
    return result


def _cell_dimensions(cell_id: str) -> tuple[int, int]:
    _require(cell_id in CELLS, "unknown continuation cell")
    return int(cell_id[1]), int(cell_id[4:])


def _order_schema(value: Any, seed: int) -> dict[str, str]:
    expected = {str(epoch) for epoch in range(SOURCE_EPOCH + 1, TARGET_EPOCH + 1)}
    _require(type(value) is dict and set(value) == expected,
             f"seed {seed} future epoch order coverage drift")
    for epoch, digest in value.items():
        _require(type(digest) is str and _DIGEST.fullmatch(digest) is not None,
                 f"seed {seed} epoch {epoch} order digest invalid")
    return copy.deepcopy(value)


def validate_t8a_report(report: Any) -> dict:
    _require(type(report) is dict, "T8A report must be an object")
    for name in (
        "P3_T8A_EPOCH30_TO_60_CONTINUATION_ELIGIBILITY_AUDIT_PASS",
        "P3_T8A_EPOCH30_TO_60_SCIENTIFIC_CONTINUATION_ELIGIBLE",
        "P3_T8A_FOUR_PROSPECTIVE_EPOCH30_CHECKPOINTS_CERTIFIED",
    ):
        _same(report.get(name), True, name)
    _same(report.get("P3_T8A_CURRENT_PRODUCTION_CONTINUATION_PATH_READY"), False,
          "T8A pre-engineering path readiness")
    _same(report.get("status"), "PASS", "T8A status")
    _same(report.get("classification"),
          "T8A_SCIENTIFIC_CONTINUATION_ELIGIBLE_ENGINEERING_GATE_REQUIRED",
          "T8A classification")
    endpoints = report.get("audited_endpoints")
    _require(type(endpoints) is dict and set(endpoints) == set(CELLS),
             "T8A endpoint set drift")
    schedule = report.get("schedule")
    _require(type(schedule) is dict, "T8A schedule missing")
    for key, expected in (
        ("continuation_start_epoch", 31), ("proposed_endpoint_epoch", 60),
        ("historical_observation_endpoint_epoch", 30),
        ("optimizer_updates_at_epoch30", SOURCE_UPDATES),
        ("optimizer_updates_at_epoch60", TARGET_UPDATES),
        ("additional_optimizer_updates", ADDITIONAL_UPDATES),
        ("physical_microsteps_at_epoch60", TARGET_UPDATES * 2),
        ("original_resolved_horizon_epochs", 120),
        ("learning_rate_transition_between_epochs31_and60", False),
        ("augmentation_policy_unchanged_through_epoch60", True),
    ):
        _same(schedule.get(key), expected, "T8A schedule " + key)
    future = schedule.get("future_epoch_order_sha256")
    _require(type(future) is dict and set(future) == {"seed1", "seed2"},
             "T8A future order seed set drift")
    for seed in (1, 2):
        _order_schema(future[f"seed{seed}"], seed)
    for cell_id, endpoint in endpoints.items():
        seed, size = _cell_dimensions(cell_id)
        _same(endpoint.get("seed"), seed, cell_id + " seed")
        _same(endpoint.get("input_size"), size, cell_id + " input size")
        _same(endpoint.get("epoch"), SOURCE_EPOCH, cell_id + " source epoch")
        _same(endpoint.get("optimizer_updates"), SOURCE_UPDATES, cell_id + " source updates")
        _same(endpoint.get("microsteps"), SOURCE_UPDATES * 2, cell_id + " source microsteps")
        _same(endpoint.get("complete_epoch_boundary"), True, cell_id + " complete boundary")
        for name in ("checkpoint", "contract", "run_binding"):
            validate_reference(endpoint.get(name), cell_id + " " + name, verify=False)
        _require(type(endpoint.get("checkpoint_binding_sha256")) is str
                 and _DIGEST.fullmatch(endpoint["checkpoint_binding_sha256"]) is not None,
                 cell_id + " checkpoint binding invalid")
    return copy.deepcopy(report)


def _orchestration_inventory(root: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for relative in ORCHESTRATION_FILES:
        path = root / relative
        _require(path.is_file(), "missing continuation orchestration file: " + relative)
        result[relative] = file_reference(path)
    return result


def _historical_result(source_contract: Mapping[str, Any]) -> tuple[dict, dict]:
    path = Path(source_contract["output_dir"]) / "worker-result.json"
    reference = file_reference(path)
    result = read_json_reference(reference, "historical worker result")
    _same(result.get("status"), "PASS", "historical worker status")
    _same(result.get("stage"), "control30", "historical worker stage")
    return result, reference


def _cell_from_report(cell_id: str, endpoint: Mapping[str, Any], report: Mapping[str, Any]) -> dict:
    seed, input_size = _cell_dimensions(cell_id)
    contract_ref = validate_reference(endpoint["contract"], cell_id + " source contract", verify=True)
    binding_ref = validate_reference(endpoint["run_binding"], cell_id + " source binding", verify=True)
    checkpoint_ref = validate_reference(endpoint["checkpoint"], cell_id + " source checkpoint", verify=True)
    source_contract = read_json_reference(contract_ref, cell_id + " source contract")
    source_binding = read_json_reference(binding_ref, cell_id + " source binding")
    _same(source_contract.get("stage"), "control30", cell_id + " source stage")
    _same(source_contract.get("cell_id"), cell_id, cell_id + " source cell")
    _same(source_contract.get("config", {}).get("seed"), seed, cell_id + " contract seed")
    _same(source_contract.get("config", {}).get("input_size"), input_size,
          cell_id + " contract input size")
    _same(source_binding.get("run_id"), source_contract.get("run_id"), cell_id + " run identity")
    _same(source_binding.get("binding_sha256"), endpoint["checkpoint_binding_sha256"],
          cell_id + " checkpoint binding")
    startup_environment = worker_startup_environment(source_contract["expected_gpu_uuid"])
    binding_config = source_binding.get("config", {})
    _same(binding_config.get("cuda_gpu_uuid"), source_contract["expected_gpu_uuid"],
          cell_id + " source CUDA UUID")
    _same(binding_config.get("sampling_backend"), "deterministic_gather",
          cell_id + " source sampling backend")
    _same(binding_config.get("cuda_backend_policy"), WORKER_CUDA_BACKEND_POLICY,
          cell_id + " source CUDA backend policy")
    _same(startup_environment["CUBLAS_WORKSPACE_CONFIG"],
          binding_config["cuda_backend_policy"]["cublas_workspace_config"],
          cell_id + " source cuBLAS startup policy")
    historical, historical_ref = _historical_result(source_contract)
    checkpoint_row = historical.get("checkpoints", {}).get("epoch_30")
    _require(type(checkpoint_row) is dict, cell_id + " historical epoch30 checkpoint missing")
    _same(_reference_core(checkpoint_row.get("checkpoint"), cell_id + " historical checkpoint"),
          checkpoint_ref,
          cell_id + " historical checkpoint reference")
    boundary_ref = validate_reference(checkpoint_row.get("state_reference"),
                                      cell_id + " epoch30 boundary state", verify=True)
    _same(historical.get("run_id"), source_contract["run_id"], cell_id + " historical result run")
    source_root = _absolute_path(source_contract.get("repo_root"), cell_id + " source repo")
    _require(source_root.is_dir(), cell_id + " source repository missing")
    core = report.get("training_core_identities")
    _require(type(core) is dict and core, "T8A training core identities missing")
    for relative, identity in core.items():
        _same(file_reference(source_root / relative),
              {**identity, "path": str((source_root / relative).resolve()),
               "sha256_scope": "complete_file_bytes"},
              cell_id + " training core " + relative)
    return {
        "cell_id": cell_id, "seed": seed, "input_size": input_size,
        "source_repo_root": str(source_root),
        "source_campaign_id": source_contract["campaign_id"],
        "source_run_id": source_contract["run_id"],
        "source_contract_reference": contract_ref,
        "source_binding_reference": binding_ref,
        "source_checkpoint_reference": checkpoint_ref,
        "source_boundary_state_reference": boundary_ref,
        "source_worker_result_reference": historical_ref,
        "source_binding_sha256": source_binding["binding_sha256"],
        "config": copy.deepcopy(source_contract["config"]),
        "pretrained": copy.deepcopy(source_contract["pretrained"]),
        "train_core": copy.deepcopy(source_contract["train_core"]),
        "development_binding": copy.deepcopy(source_contract["development_binding"]),
        "expected_gpu_uuid": source_contract["expected_gpu_uuid"],
        "startup_environment": startup_environment,
        "num_workers": source_contract["num_workers"],
        "prefetch_factor": source_contract["prefetch_factor"],
        "epoch_order_sha256": _order_schema(
            report["schedule"]["future_epoch_order_sha256"][f"seed{seed}"], seed),
    }


def make_epoch60_continuation_freeze(*, t8a_report_reference: Mapping[str, Any],
                                     repo_root: str | os.PathLike[str], campaign_id: str,
                                     output_root: str | os.PathLike[str]) -> dict:
    root = Path(repo_root).resolve()
    _require(root.is_dir(), "continuation checkout missing")
    _identifier(campaign_id, "continuation campaign_id")
    output = Path(output_root).resolve()
    _require(output.is_absolute() and output != root and not output.is_relative_to(root),
             "continuation output must be a separate absolute tree")
    report_ref = validate_reference(t8a_report_reference, "T8A report", verify=True)
    report = validate_t8a_report(read_json_reference(report_ref, "T8A report"))
    cells = {cell_id: _cell_from_report(cell_id, report["audited_endpoints"][cell_id], report)
             for cell_id in CELLS}
    source_campaigns = {cell["source_campaign_id"] for cell in cells.values()}
    _require(campaign_id not in source_campaigns, "continuation campaign reuses a historical campaign id")
    freeze = {
        "schema_version": SCHEMA_VERSION, "kind": FREEZE_KIND,
        "campaign_id": campaign_id, "repo_root": str(root), "output_root": str(output),
        "source_epoch": SOURCE_EPOCH, "target_epoch": TARGET_EPOCH,
        "t8a_report_reference": report_ref,
        "training_core_identities": copy.deepcopy(report["training_core_identities"]),
        "execution_policy": copy.deepcopy(EXECUTION_POLICY),
        "orchestration_sources": _orchestration_inventory(root), "cells": cells,
    }
    return validate_epoch60_continuation_freeze(freeze, verify_files=True)


def validate_epoch60_continuation_freeze(value: Any, *, verify_files: bool = False) -> dict:
    keys = {"schema_version", "kind", "campaign_id", "repo_root", "output_root",
            "source_epoch", "target_epoch", "t8a_report_reference", "training_core_identities",
            "execution_policy", "orchestration_sources", "cells"}
    freeze = _schema(value, keys, "continuation freeze")
    _same(freeze["schema_version"], SCHEMA_VERSION, "freeze schema version")
    _same(freeze["kind"], FREEZE_KIND, "freeze kind")
    _identifier(freeze["campaign_id"], "freeze campaign_id")
    root = _absolute_path(freeze["repo_root"], "freeze repo_root")
    output = _absolute_path(freeze["output_root"], "freeze output_root")
    _require(output != root and not output.is_relative_to(root), "freeze output overlaps checkout")
    _same(freeze["source_epoch"], SOURCE_EPOCH, "freeze source epoch")
    _same(freeze["target_epoch"], TARGET_EPOCH, "freeze target epoch")
    _same(freeze["execution_policy"], EXECUTION_POLICY, "freeze execution policy")
    validate_reference(freeze["t8a_report_reference"], "freeze T8A report", verify=verify_files)
    _require(type(freeze["training_core_identities"]) is dict
             and freeze["training_core_identities"], "freeze training core inventory missing")
    sources = freeze["orchestration_sources"]
    _require(type(sources) is dict and set(sources) == set(ORCHESTRATION_FILES),
             "freeze orchestration source set drift")
    for relative, reference in sources.items():
        checked = validate_reference(reference, "orchestration " + relative, verify=verify_files)
        _same(checked["path"], str((root / relative).resolve()), "orchestration path " + relative)
    cells = freeze["cells"]
    _require(type(cells) is dict and set(cells) == set(CELLS), "freeze cell set drift")
    historical_outputs = []
    source_campaigns = set()
    for cell_id in CELLS:
        cell = cells[cell_id]
        keys = {"cell_id", "seed", "input_size", "source_repo_root", "source_campaign_id",
                "source_run_id", "source_contract_reference", "source_binding_reference",
                "source_checkpoint_reference", "source_boundary_state_reference",
                "source_worker_result_reference", "source_binding_sha256", "config", "pretrained",
                "train_core", "development_binding", "expected_gpu_uuid", "num_workers",
                "startup_environment", "prefetch_factor", "epoch_order_sha256"}
        _schema(cell, keys, cell_id + " freeze cell")
        seed, size = _cell_dimensions(cell_id)
        _same((cell["cell_id"], cell["seed"], cell["input_size"]),
              (cell_id, seed, size), cell_id + " dimensions")
        _absolute_path(cell["source_repo_root"], cell_id + " source repo")
        _identifier(cell["source_campaign_id"], cell_id + " source campaign")
        _identifier(cell["source_run_id"], cell_id + " source run")
        source_campaigns.add(cell["source_campaign_id"])
        _require(freeze["campaign_id"] != cell["source_campaign_id"],
                 cell_id + " continuation reuses source campaign")
        _require(freeze["campaign_id"] not in cell["source_run_id"],
                 cell_id + " source run was relabelled")
        for name in ("source_contract_reference", "source_binding_reference",
                     "source_checkpoint_reference", "source_boundary_state_reference",
                     "source_worker_result_reference"):
            reference = validate_reference(cell[name], cell_id + " " + name,
                                           verify=verify_files)
            historical_outputs.append(Path(reference["path"]))
        _require(type(cell["source_binding_sha256"]) is str
                 and _DIGEST.fullmatch(cell["source_binding_sha256"]) is not None,
                 cell_id + " binding SHA invalid")
        _require(type(cell["config"]) is dict
                 and cell["config"].get("seed") == seed
                 and cell["config"].get("input_size") == size,
                 cell_id + " configuration drift")
        _require(type(cell["expected_gpu_uuid"]) is str
                 and _GPU_UUID.fullmatch(cell["expected_gpu_uuid"]) is not None,
                 cell_id + " GPU UUID invalid")
        _same(cell["startup_environment"],
              worker_startup_environment(cell["expected_gpu_uuid"]),
              cell_id + " startup environment")
        _require(type(cell["num_workers"]) is int and cell["num_workers"] >= 0,
                 cell_id + " worker count invalid")
        _require(type(cell["prefetch_factor"]) is int and cell["prefetch_factor"] > 0,
                 cell_id + " prefetch factor invalid")
        _order_schema(cell["epoch_order_sha256"], seed)
    for path in historical_outputs:
        _require(not path.is_relative_to(output), "historical authority is inside new output tree")
    if verify_files:
        report = validate_t8a_report(read_json_reference(
            freeze["t8a_report_reference"], "freeze T8A report"))
        _same(report["training_core_identities"], freeze["training_core_identities"],
              "freeze/T8A training core inventory")
    return copy.deepcopy(freeze)


def _result_identity(reference: Mapping[str, Any], *, cell_id: str, stage: str,
                     freeze_reference: Mapping[str, Any], verify_files: bool) -> dict:
    result = read_json_reference(reference, f"{cell_id} {stage} result", verify=verify_files)
    _same(result.get("schema_version"), SCHEMA_VERSION, "prerequisite result schema")
    _same(result.get("kind"), RESULT_KIND, "prerequisite result kind")
    _same(result.get("status"), "PASS", "prerequisite result status")
    _same(result.get("cell_id"), cell_id, "prerequisite result cell")
    _same(result.get("stage"), stage, "prerequisite result stage")
    _same(result.get("freeze_reference"), freeze_reference, "prerequisite result freeze")
    return result


def continuation_worker_contract(freeze: Mapping[str, Any], freeze_reference: Mapping[str, Any],
                                 *, cell_id: str, stage: str,
                                 policy_bundle: Mapping[str, Any],
                                 smoke_reference: Mapping[str, Any] | None = None,
                                 replay_reference: Mapping[str, Any] | None = None,
                                 matched_reference: Mapping[str, Any] | None = None) -> dict:
    checked = validate_epoch60_continuation_freeze(freeze, verify_files=False)
    freeze_ref = validate_reference(freeze_reference, "continuation freeze reference", verify=True)
    _same(read_json_reference(freeze_ref, "continuation freeze"), checked,
          "continuation freeze bytes")
    _require(cell_id in CELLS and stage in STAGES, "unknown continuation worker target")
    cell = checked["cells"][cell_id]
    execution_id = f"{checked['campaign_id']}-{cell_id}-{stage}"
    output = Path(checked["output_root"]) / "execution" / f"{cell_id}-{stage}"
    contract = {
        "schema_version": SCHEMA_VERSION, "kind": CONTRACT_KIND,
        "campaign_id": checked["campaign_id"], "execution_id": execution_id,
        "invocation_id": execution_id, "cell_id": cell_id, "stage": stage,
        "repo_root": checked["repo_root"], "output_dir": str(output),
        "freeze_reference": freeze_ref, "source_repo_root": cell["source_repo_root"],
        "source_campaign_id": cell["source_campaign_id"], "source_run_id": cell["source_run_id"],
        "source_contract_reference": copy.deepcopy(cell["source_contract_reference"]),
        "source_binding_reference": copy.deepcopy(cell["source_binding_reference"]),
        "source_checkpoint_reference": copy.deepcopy(cell["source_checkpoint_reference"]),
        "source_boundary_state_reference": copy.deepcopy(cell["source_boundary_state_reference"]),
        "source_binding_sha256": cell["source_binding_sha256"],
        "config": copy.deepcopy(cell["config"]), "pretrained": copy.deepcopy(cell["pretrained"]),
        "train_core": copy.deepcopy(cell["train_core"]),
        "development_binding": copy.deepcopy(cell["development_binding"]),
        "expected_gpu_uuid": cell["expected_gpu_uuid"], "num_workers": cell["num_workers"],
        "prefetch_factor": cell["prefetch_factor"],
        "startup_environment": copy.deepcopy(cell["startup_environment"]),
        "epoch_order_sha256": copy.deepcopy(cell["epoch_order_sha256"]),
        "policy_bundle": copy.deepcopy(policy_bundle),
        "smoke_reference": copy.deepcopy(smoke_reference),
        "replay_reference": copy.deepcopy(replay_reference),
        "matched_reference": copy.deepcopy(matched_reference),
        "execution_policy": copy.deepcopy(EXECUTION_POLICY),
        "orchestration_sources": copy.deepcopy(checked["orchestration_sources"]),
    }
    return validate_continuation_worker_contract(contract, checked, verify_files=False)


def validate_continuation_worker_contract(value: Any, freeze: Mapping[str, Any], *,
                                          verify_files: bool = False) -> dict:
    checked_freeze = validate_epoch60_continuation_freeze(freeze, verify_files=False)
    keys = {"schema_version", "kind", "campaign_id", "execution_id", "invocation_id",
            "cell_id", "stage", "repo_root", "output_dir", "freeze_reference",
            "source_repo_root", "source_campaign_id", "source_run_id",
            "source_contract_reference", "source_binding_reference",
            "source_checkpoint_reference", "source_boundary_state_reference",
            "source_binding_sha256", "config", "pretrained", "train_core",
            "development_binding", "expected_gpu_uuid", "num_workers", "prefetch_factor",
            "startup_environment",
            "epoch_order_sha256", "policy_bundle", "smoke_reference", "replay_reference",
            "matched_reference", "execution_policy", "orchestration_sources"}
    contract = _schema(value, keys, "continuation worker contract")
    _same(contract["schema_version"], SCHEMA_VERSION, "worker schema version")
    _same(contract["kind"], CONTRACT_KIND, "worker kind")
    _same(contract["campaign_id"], checked_freeze["campaign_id"], "worker campaign")
    _require(contract["cell_id"] in CELLS and contract["stage"] in STAGES,
             "worker cell or stage invalid")
    cell_id, stage = contract["cell_id"], contract["stage"]
    cell = checked_freeze["cells"][cell_id]
    expected_id = f"{checked_freeze['campaign_id']}-{cell_id}-{stage}"
    _same(contract["execution_id"], expected_id, "outer continuation execution identity")
    _same(contract["invocation_id"], expected_id, "continuation invocation identity")
    _require(expected_id != cell["source_run_id"] and expected_id != cell["source_campaign_id"],
             "continuation identity aliases historical identity")
    _same(contract["repo_root"], checked_freeze["repo_root"], "worker orchestration repo")
    expected_output = str(Path(checked_freeze["output_root"]) / "execution" / f"{cell_id}-{stage}")
    _same(contract["output_dir"], expected_output, "worker output")
    _require(not Path(expected_output).exists(), "worker output already exists")
    for name in ("source_repo_root", "source_campaign_id", "source_run_id",
                 "source_contract_reference", "source_binding_reference",
                 "source_checkpoint_reference", "source_boundary_state_reference",
                 "source_binding_sha256", "config", "pretrained", "train_core",
                 "development_binding", "expected_gpu_uuid", "num_workers",
                 "startup_environment", "prefetch_factor", "epoch_order_sha256"):
        _same(contract[name], cell[name], "worker lineage " + name)
    _same(contract["execution_policy"], EXECUTION_POLICY, "worker execution policy")
    _same(contract["orchestration_sources"], checked_freeze["orchestration_sources"],
          "worker orchestration inventory")
    _same(contract["startup_environment"],
          worker_startup_environment(cell["expected_gpu_uuid"]),
          "worker startup environment")
    validate_reference(contract["freeze_reference"], "worker freeze reference",
                       verify=verify_files)
    _require(type(contract["policy_bundle"]) is dict and contract["policy_bundle"],
             "worker native policy bundle missing")
    seed, size = _cell_dimensions(cell_id)
    _same((contract["config"]["seed"], contract["config"]["input_size"]),
          (seed, size), "worker dimensions")
    if stage == "smoke":
        _same((contract["smoke_reference"], contract["replay_reference"]), (None, None),
              "smoke prerequisites")
    elif stage == "smoke_replay":
        _require(type(contract["smoke_reference"]) is dict, "replay requires original smoke")
        _result_identity(contract["smoke_reference"], cell_id=cell_id, stage="smoke",
                         freeze_reference=contract["freeze_reference"], verify_files=verify_files)
        _same(contract["replay_reference"], None, "replay formal prerequisite")
    else:
        _require(type(contract["smoke_reference"]) is dict
                 and type(contract["replay_reference"]) is dict,
                 "formal60 requires passed smoke and replay")
        _result_identity(contract["smoke_reference"], cell_id=cell_id, stage="smoke",
                         freeze_reference=contract["freeze_reference"], verify_files=verify_files)
        _result_identity(contract["replay_reference"], cell_id=cell_id, stage="smoke_replay",
                         freeze_reference=contract["freeze_reference"], verify_files=verify_files)
    if size == 640:
        _same(contract["matched_reference"], None, "640 matched reference")
    else:
        _require(type(contract["matched_reference"]) is dict, "896 requires matched 640 result")
        matched_stage = stage
        _result_identity(contract["matched_reference"], cell_id=f"s{seed}_r640",
                         stage=matched_stage, freeze_reference=contract["freeze_reference"],
                         verify_files=verify_files)
    if verify_files:
        validate_epoch60_continuation_freeze(checked_freeze, verify_files=True)
        source_binding = read_json_reference(contract["source_binding_reference"],
                                             "worker source binding")
        _same(source_binding.get("binding_sha256"), contract["source_binding_sha256"],
              "worker source binding digest")
    return copy.deepcopy(contract)


def validate_continuation_result(value: Any, contract: Mapping[str, Any], *,
                                 verify_files: bool = False) -> dict:
    _require(type(value) is dict, "continuation result must be an object")
    for name, expected in (
        ("schema_version", SCHEMA_VERSION), ("kind", RESULT_KIND), ("status", "PASS"),
        ("campaign_id", contract["campaign_id"]), ("execution_id", contract["execution_id"]),
        ("cell_id", contract["cell_id"]), ("stage", contract["stage"]),
        ("source_run_id", contract["source_run_id"]),
        ("source_binding_sha256", contract["source_binding_sha256"]),
        ("freeze_reference", contract["freeze_reference"]),
    ):
        _same(value.get(name), expected, "continuation result " + name)
    _same(value.get("startup_environment"), contract["startup_environment"],
          "continuation result startup environment")
    historical_environment = {
        name: contract["startup_environment"][name]
        for name in ("CUDA_VISIBLE_DEVICES", "MKL_THREADING_LAYER", "PYTHONNOUSERSITE",
                     "OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG")
    }
    _same(value.get("historical_startup_environment"), historical_environment,
          "continuation result historical startup environment")
    expected_receipts = (ADDITIONAL_UPDATES if contract["stage"] == "formal60"
                         else 4 if contract["stage"] == "smoke" else 2)
    _same(value.get("receipt_count"), expected_receipts, "continuation receipt count")
    clocks = value.get("final_clocks")
    _require(type(clocks) is dict, "continuation final clocks missing")
    if contract["stage"] == "formal60":
        expected = {"epoch": TARGET_EPOCH, "epoch_active": False,
                    "optimizer_updates": TARGET_UPDATES, "microsteps": TARGET_UPDATES * 2}
    else:
        expected = {"epoch": 31, "epoch_active": True,
                    "optimizer_updates": SOURCE_UPDATES + 4,
                    "microsteps": (SOURCE_UPDATES + 4) * 2}
    for name, wanted in expected.items():
        _same(clocks.get(name), wanted, "continuation final clock " + name)
    for name in ("contract_reference", "restored_boundary_reference", "final_state_reference",
                 "monitor_reference"):
        validate_reference(value.get(name), "continuation result " + name,
                           verify=verify_files)
    return copy.deepcopy(value)
