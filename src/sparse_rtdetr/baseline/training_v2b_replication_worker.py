"""Seed 1/2 resolution replication under a separately frozen workload contract.

This module reuses the frozen training, checkpoint and development implementations.
It does not reuse the historical seed-zero workers or grant GPU admission. Capacity
has its own disposable worker; the controller owns sequencing and guardian closure.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import functools
import math
import os
from pathlib import Path
import re
from typing import Any

import torch

from . import training_v2b_control as control
from .training_v2b import V2BConfig, build_v2b_components, logical_batch_indices
from .training_v2b_data import (
    TrainCoreDataConfig, build_train_core_loader, validate_worker_environment,
)
from .training_v2b_device import prepare_runtime
from .training_v2b_evidence import (
    canonical_json_bytes, canonical_sha256, file_reference, strict_json_loads,
    write_exclusive_json,
)
from .training_v2b_runtime import V2BTrainingSession, build_train_core_run_binding


KIND = "v2b_seed_resolution_replication_worker"
RESULT_KIND = KIND + "_result"
FREEZE_KIND = "v2b_seed_resolution_replication_freeze"
GATE_KIND = "v2b_seed_resolution_replication_stage_gate"
COMPLETION_KIND = "v2b_seed_resolution_replication_completion"
COMPARISON_KIND = "v2b_seed_resolution_replication_smoke_comparison"
STAGES = ("capacity", "smoke", "smoke_replay", "control30")
CELLS = ("s1_r640", "s1_r896", "s2_r640", "s2_r896")
REF_KEYS = {"path", "sha256", "size_bytes", "sha256_scope"}
COMMON_KEYS = {
    "torch_version", "code_files", "train_core", "pretrained", "policy_bundle",
    "expected_gpu_uuid", "num_workers", "prefetch_factor",
}
CELL_KEYS = {
    "config", "development_binding", "epoch_order_sha256", "capacity_plan", "invocations",
}
FREEZE_KEYS = {
    "schema_version", "kind", "campaign_id", "repo_root", "output_root", "source",
    "clean_archive_reference", "qualification_reference", "authorization_reference",
    "foundation_reference", "common", "cells", "formal_order",
}
CONTRACT_KEYS = {
    "schema_version", "kind", "campaign_id", "cell_id", "stage", "run_id",
    "invocation_id", "repo_root", "output_dir", "freeze_reference",
    "qualification_reference", "stage_gate_reference", "capacity_plan",
    "config", "num_workers", "prefetch_factor", "torch_version", "code_files",
    "train_core", "pretrained", "development_binding", "epoch_order_sha256",
    "policy_bundle", "expected_gpu_uuid", "matched_reference", "replay_source",
}
GATE_KEYS = {
    "schema_version", "kind", "status", "campaign_id", "freeze_reference",
    "cell_id", "stage", "prerequisites", "smoke_comparisons",
}
COMPLETION_KEYS = {
    "schema_version", "kind", "status", "campaign_id", "cell_id", "stage",
    "freeze_reference", "contract_reference", "result_reference", "launch_reference",
    "monitor", "elapsed_seconds", "capacity_memory_audit_reference",
}
_SOURCE_KEYS = {"repo_root", "commit", "tree", "branch", "code_files", "inventory_sha256"}
_MONITOR_IDENTITY_KEYS = (
    "run_id", "run_binding_sha256", "policy_sha256", "monitor_pid",
    "owner", "guardian_identity", "gpu_uuid",
)
_MONITOR_LEAVES = ("samples", "setter_receipt", "startup_evidence", "guardian_heartbeat")


class ReplicationWorkerError(control.ControlWorkerError):
    """Stop the current invocation without retry, resume or configuration fallback."""


def _fail(message):
    raise ReplicationWorkerError(message)


def _same(left, right, label):
    if canonical_sha256(left) != canonical_sha256(right):
        _fail(label + " mismatch")


def _identifier(value, label):
    if type(value) is not str or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}", value) is None:
        _fail(label + " must be a bounded safe identifier")
    return value


def _schema(value, keys, kind, label):
    if type(value) is not dict or set(value) != keys:
        _fail(label + " schema keys mismatch")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1 or value["kind"] != kind:
        _fail(label + " schema version/kind mismatch")


def _reference(value, label, *, verify=False):
    if type(value) is not dict or set(value) != REF_KEYS:
        _fail(label + " requires exactly four complete-file reference fields")
    if value["sha256_scope"] != "complete_file_bytes":
        _fail(label + " must bind complete_file_bytes")
    return control._reference(value, label, verify=verify)


def _read_reference(value, label):
    return control._read_json_reference(_reference(value, label), label)


@functools.lru_cache(maxsize=2)
def _frozen_seed_orders(seed):
    if type(seed) is not int or seed not in (1, 2):
        _fail("replication seed must be the builtin integer 1 or 2")
    return tuple(
        (str(epoch), canonical_sha256([
            list(batch) for batch in logical_batch_indices(
                4869, seed=seed, epoch=epoch, logical_batch_size=16)
        ]))
        for epoch in range(1, 31)
    )


def frozen_seed_orders(seed):
    """Return a detached plan; algorithm epochs are one-based, exactly 1..30."""
    if type(seed) is not int or seed not in (1, 2):
        _fail("replication seed must be the builtin integer 1 or 2")
    return dict(_frozen_seed_orders(seed))


def _configuration(value, cell_id):
    if type(value) is not dict:
        _fail("complete replication model configuration is required")
    seed, size = value.get("seed"), value.get("input_size")
    if type(seed) is not int or seed not in (1, 2) or type(size) is not int or size not in (640, 896):
        _fail("replication is restricted to seed 1/2 and resolution 640/896")
    if cell_id != f"s{seed}_r{size}":
        _fail("cell identity differs from its seed/resolution")
    expected = V2BConfig(
        seed=seed, input_size=size, physical_batch_size=8, accumulation_steps=2,
        sampling_backend="deterministic_gather")
    _same(value, asdict(expected), "frozen replication configuration")
    return expected


def _development_metadata(value, size):
    if type(value) is not dict:
        _fail("existing development binding is required")
    # Development's established internal references are deliberately unchanged.
    for name, filename in (("annotation", "development_coco.json"),
                           ("manifest", "development_manifest.json")):
        control._reference(value.get(name), "development " + name, expected_name=filename)
    control._path(value.get("image_root"), "development image_root")
    control._sha(value.get("binding_sha256"), "development binding")
    if value.get("policy", {}).get("input_size") != [size, size]:
        _fail("development geometry must equal this cell's input resolution")


def validate_replication_worker_contract(contract, *, verify_files=False):
    """Reject metadata drift before opening authority inputs or constructing a model."""
    _schema(contract, CONTRACT_KEYS, KIND, "replication worker contract")
    if contract["cell_id"] not in CELLS or contract["stage"] not in STAGES:
        _fail("unknown replication cell/stage")
    for key in ("campaign_id", "run_id", "invocation_id"):
        _identifier(contract[key], key)
    root = control._path(contract["repo_root"], "repo_root")
    output = control._path(contract["output_dir"], "output_dir")
    if root == output or root.is_relative_to(output):
        _fail("output may not contain the source checkout")
    config = _configuration(contract["config"], contract["cell_id"])
    for key in ("num_workers", "prefetch_factor"):
        if type(contract[key]) is not int or contract[key] != 2:
            _fail(key + " must remain exactly two")
    if type(contract["torch_version"]) is not str or contract["torch_version"] != str(torch.__version__):
        _fail("PyTorch version differs from the frozen construction/order authority")
    if (type(contract["expected_gpu_uuid"]) is not str or
            re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}",
                         contract["expected_gpu_uuid"]) is None):
        _fail("the complete authorized GPU UUID is required")
    code = contract["code_files"]
    if type(code) is not dict or not code:
        _fail("complete frozen code inventory is required")
    for name, ref in code.items():
        if (type(name) is not str or Path(name).is_absolute() or str(Path(name)) != name
                or ".." in Path(name).parts):
            _fail("code inventory names must be checkout-relative")
        if Path(_reference(ref, "code " + name)["path"]) != root / name:
            _fail("code reference belongs to another checkout")
    data = contract["train_core"]
    if type(data) is not dict or set(data) != {
            "annotation", "manifest", "image_root", "expected_samples", "expected_batches_per_epoch"}:
        _fail("train_core authority schema mismatch")
    for name, filename, sha, size in (
        ("annotation", "train_core_coco.json", control.TRAIN_ANNOTATION_SHA256, 51148042),
        ("manifest", "train_core_manifest.json", control.TRAIN_MANIFEST_SHA256, 2955809),
    ):
        ref = _reference(data[name], "train_core " + name)
        if Path(ref["path"]).name != filename or ref["sha256"] != sha or ref["size_bytes"] != size:
            _fail("train_core " + name + " authority changed")
    control._path(data["image_root"], "train_core image_root")
    if (type(data["expected_samples"]) is not int or data["expected_samples"] != 4869
            or type(data["expected_batches_per_epoch"]) is not int
            or data["expected_batches_per_epoch"] != 304):
        _fail("the full train_core/drop-five epoch must remain 4869 samples and 304 batches")
    pretrained = _reference(contract["pretrained"], "pretrained")
    if (Path(pretrained["path"]).name != "ResNet18_vd_pretrained_from_paddle.pth"
            or pretrained["sha256"] != control.PRETRAINED_SHA256
            or pretrained["size_bytes"] != 44878642):
        _fail("pretrained authority changed")
    _development_metadata(contract["development_binding"], config.input_size)
    _same(contract["epoch_order_sha256"], frozen_seed_orders(config.seed), "seed-specific 30 epoch order")
    policy = contract["policy_bundle"]
    if type(policy) is not dict or policy.get("setter_mode") != "external_admin_acknowledged":
        _fail("replication must preserve the external administrator clock lock")
    _reference(policy.get("authorization_reference"), "hardware authorization")
    for key in ("freeze_reference", "qualification_reference", "capacity_plan"):
        _reference(contract[key], key)
    if contract["stage"] == "capacity":
        if any(contract[key] is not None for key in ("stage_gate_reference", "matched_reference", "replay_source")):
            _fail("capacity has no prior-stage gate, matched source or replay source")
    else:
        _reference(contract["stage_gate_reference"], "stage gate")
        if config.input_size == 896:
            _reference(contract["matched_reference"], "same-seed matched 640 source")
        elif contract["matched_reference"] is not None:
            _fail("640 must not consume a matched 896 source")
        if contract["stage"] == "smoke_replay":
            _reference(contract["replay_source"], "original smoke source")
        elif contract["replay_source"] is not None:
            _fail("only smoke_replay may restore a checkpoint")
    checked = strict_json_loads(canonical_json_bytes(contract))
    if verify_files:
        verify_replication_worker_authorities(checked)
    return checked


def validate_replication_freeze(freeze, contract):
    """Validate the complete four-cell definition without file reads."""
    _schema(freeze, FREEZE_KEYS, FREEZE_KIND, "replication freeze")
    for key in ("campaign_id", "repo_root", "qualification_reference"):
        _same(freeze[key], contract[key], "freeze " + key)
    output_root = control._path(freeze["output_root"], "frozen output root")
    root = control._path(freeze["repo_root"], "frozen source root")
    if root == output_root or root.is_relative_to(output_root):
        _fail("frozen output root may not contain the source checkout")
    for key in ("qualification_reference", "authorization_reference", "clean_archive_reference", "foundation_reference"):
        _reference(freeze[key], "freeze " + key)
    common = freeze["common"]
    if type(common) is not dict or set(common) != COMMON_KEYS:
        _fail("freeze common schema keys mismatch")
    for key in COMMON_KEYS:
        _same(common[key], contract[key], "freeze common " + key)
    _same(common["policy_bundle"]["authorization_reference"], freeze["authorization_reference"],
          "frozen hardware authorization")
    source = freeze["source"]
    if type(source) is not dict or set(source) != _SOURCE_KEYS:
        _fail("freeze source_snapshot schema mismatch")
    _same(source["repo_root"], freeze["repo_root"], "source snapshot checkout")
    for name in ("commit", "tree"):
        if type(source[name]) is not str or re.fullmatch(r"[0-9a-f]{40}", source[name]) is None:
            _fail("source " + name + " must be a complete Git object identity")
    if type(source["branch"]) is not str:
        _fail("source branch must be recorded")
    _same(source["code_files"], common["code_files"], "complete source/common code inventory")
    _same(source["inventory_sha256"], canonical_sha256(source["code_files"]), "source inventory digest")
    if type(freeze["cells"]) is not dict or set(freeze["cells"]) != set(CELLS):
        _fail("freeze must contain exactly the four predefined replication cells")
    _same(freeze["formal_order"], list(CELLS), "formal cell order")
    output_paths, invocation_ids, logical_runs = [], [], []
    for cell_id in CELLS:
        cell = freeze["cells"][cell_id]
        if type(cell) is not dict or set(cell) != CELL_KEYS:
            _fail("frozen cell schema mismatch")
        config = _configuration(cell["config"], cell_id)
        _development_metadata(cell["development_binding"], config.input_size)
        _same(cell["epoch_order_sha256"], frozen_seed_orders(config.seed), cell_id + " frozen orders")
        _reference(cell["capacity_plan"], cell_id + " frozen capacity plan")
        invocations = cell["invocations"]
        if type(invocations) is not dict or set(invocations) != set(STAGES):
            _fail("every cell requires all four predefined invocations")
        for stage, invocation in invocations.items():
            if type(invocation) is not dict or set(invocation) != {"run_id", "invocation_id", "output_dir"}:
                _fail("frozen invocation schema mismatch")
            _identifier(invocation["run_id"], "frozen run id")
            invocation_ids.append(_identifier(invocation["invocation_id"], "frozen invocation id"))
            path = control._path(invocation["output_dir"], "frozen invocation output")
            if path == output_root or not path.is_relative_to(output_root):
                _fail("invocation output escaped its frozen output root")
            output_paths.append(path)
        if invocations["smoke"]["run_id"] != invocations["smoke_replay"]["run_id"]:
            _fail("original smoke/replay must share their checkpoint run identity")
        logical_runs.extend(invocations[stage]["run_id"] for stage in ("capacity", "smoke", "control30"))
    if len(set(invocation_ids)) != 16 or len(set(logical_runs)) != 12:
        _fail("frozen invocation/logical run identities collide")
    if any(a == b or a.is_relative_to(b) or b.is_relative_to(a)
           for i, a in enumerate(output_paths) for b in output_paths[i + 1:]):
        _fail("frozen invocation output directories overlap")
    selected = freeze["cells"][contract["cell_id"]]
    for key in CELL_KEYS - {"invocations"}:
        _same(selected[key], contract[key], "selected frozen cell " + key)
    for key, value in selected["invocations"][contract["stage"]].items():
        _same(value, contract[key], "selected frozen invocation " + key)
    return copy.deepcopy(freeze)


def _qualified_document(reference):
    from .training_v2b_replication_gate import verify_replication_qualification_binding
    qualified = verify_replication_qualification_binding(reference)
    if (type(qualified) is not dict
            or qualified.get("kind") != "v2b_seed_resolution_replication_qualification"
            or qualified.get("status") != "PASS"
            or qualified.get("eligible_for_fixed_paired_seed1_and_seed2") is not True
            or qualified.get("gpu_admission_granted") is not False
            or qualified.get("requires_fresh_native_admission") is not True):
        _fail("CPU evidence audit PASS is insufficient: explicit replication eligibility is required")
    return qualified


def _expected_prerequisites(contract, freeze):
    cell, stage = contract["cell_id"], contract["stage"]
    if stage == "capacity":
        return set()
    expected = {(cell, "capacity")}
    if stage == "smoke_replay":
        expected.add((cell, "smoke"))
    if stage == "control30":
        expected = {(c, s) for c in CELLS for s in ("capacity", "smoke", "smoke_replay")}
        position = freeze["formal_order"].index(cell)
        if position:
            expected.add((freeze["formal_order"][position - 1], "control30"))
    if contract["config"]["input_size"] == 896:
        peer = f"s{contract['config']['seed']}_r640"
        expected.add((peer, "control30" if stage == "control30" else "smoke"))
    return expected


def _successful_cell_source(reference, *, cell_id, stage, contract, freeze):
    result = _read_reference(reference, "completed replication worker")
    if (type(result) is not dict or result.get("kind") != RESULT_KIND
            or result.get("status") != "PASS" or result.get("cell_id") != cell_id
            or result.get("stage") != stage or result.get("campaign_id") != contract["campaign_id"]):
        _fail("source is not the required successful replication cell/stage")
    _same(result.get("freeze_reference"), contract["freeze_reference"], "source frozen package")
    _same(result.get("qualification_reference"), contract["qualification_reference"], "source qualification")
    expected = freeze["cells"][cell_id]["invocations"][stage]
    for name in ("run_id", "invocation_id"):
        _same(result.get(name), expected[name], "source " + name)
    if type(result.get("worker_pid")) is not int or result["worker_pid"] <= 0:
        _fail("source worker PID is absent")
    _same(result.get("seed"), freeze["cells"][cell_id]["config"]["seed"], "source seed")
    _same(result.get("input_size"), freeze["cells"][cell_id]["config"]["input_size"], "source resolution")
    _same(result.get("capacity_plan"), freeze["cells"][cell_id]["capacity_plan"], "source capacity plan")
    if (result.get("guardian_clearance_required_after_worker_exit") is not True
            or result.get("scientific_certified") is not False
            or result.get("automatic_retry_or_batch_fallback") is not False):
        _fail("source result scope/guardian declarations differ")
    source_contract = _read_reference(result.get("contract_reference"), "source worker contract")
    validate_replication_worker_contract(source_contract, verify_files=False)
    _same(source_contract["freeze_reference"], contract["freeze_reference"], "source contract freeze reference")
    validate_replication_freeze(freeze, source_contract)
    if source_contract["cell_id"] != cell_id or source_contract["stage"] != stage:
        _fail("source contract cell/stage differs from its result")
    if Path(reference["path"]) != Path(expected["output_dir"]) / "worker-result.json":
        _fail("source worker result is outside its frozen invocation output")
    return result, source_contract


def _verify_completion(reference, result_reference, result, source_contract, contract, freeze):
    completion = _read_reference(reference, "replication completion")
    _schema(completion, COMPLETION_KEYS, COMPLETION_KIND, "replication completion")
    if completion["status"] != "PASS":
        _fail("prerequisite completion is not PASS")
    for key in ("campaign_id", "cell_id", "stage", "freeze_reference"):
        _same(completion[key], result[key], "completion " + key)
    _same(completion["result_reference"], result_reference, "completed worker bytes")
    _same(completion["contract_reference"], result["contract_reference"], "completed contract bytes")
    if (type(completion["elapsed_seconds"]) not in (int, float)
            or not math.isfinite(completion["elapsed_seconds"]) or completion["elapsed_seconds"] < 0):
        _fail("completion elapsed time is invalid")
    launch = _read_reference(completion["launch_reference"], "worker launch")
    if (type(launch) is not dict or type(launch.get("pid")) is not int
            or launch["pid"] != result["worker_pid"]):
        _fail("completion launch PID differs from its worker result")
    _same(launch.get("contract"), result["contract_reference"], "launch contract")
    monitor = completion["monitor"]
    if (type(monitor) is not dict or monitor.get("status") != "PASS"
            or monitor.get("worker_exited") is not True):
        _fail("native monitor has not closed after owner exit")
    monitor_reference = result.get("monitor_reference")
    if (type(monitor_reference) is not dict or monitor_reference.get("worker_must_exit") is not True
            or any(key not in monitor_reference or key not in monitor for key in _MONITOR_IDENTITY_KEYS)):
        _fail("source result lacks complete native finish identities")
    for key in _MONITOR_IDENTITY_KEYS:
        _same(monitor[key], monitor_reference[key], "native completion " + key)
    for name in ("run_binding_sha256", "policy_sha256"):
        control._sha(monitor[name], "native " + name)
    if (type(monitor["monitor_pid"]) is not int or monitor["monitor_pid"] <= 0
            or monitor["monitor_pid"] == result["worker_pid"]):
        _fail("native guardian PID is invalid")
    owner = _native_identity(monitor["owner"], "native owner")
    guardian = _native_identity(monitor["guardian_identity"], "native guardian")
    if (monitor["run_id"] != result["run_id"]
            or monitor["gpu_uuid"] != contract["expected_gpu_uuid"]
            or owner["pid"] != result["worker_pid"] or guardian["pid"] != monitor["monitor_pid"]):
        _fail("native monitor owner/run/GPU identity differs")
    _same(launch.get("owner"), owner, "launch/native owner identity")
    binding = _read_reference(result.get("binding_reference"), "completed run binding")
    _same(binding.get("binding_sha256"),
          canonical_sha256({key: value for key, value in binding.items() if key != "binding_sha256"}),
          "completed run binding digest")
    _same(binding["binding_sha256"], monitor["run_binding_sha256"], "native actual run binding")
    _same(binding.get("run_id"), result["run_id"], "native bound run id")
    _same(binding.get("code"), source_contract["code_files"], "native frozen source inventory")
    for name in ("seed", "input_size", "physical_batch_size", "accumulation_steps"):
        _same(binding.get("config", {}).get(name), source_contract["config"][name], "native bound " + name)
    scope, limit = (("train_core_30epoch", 43200) if source_contract["stage"] == "control30"
                    else ("paired_smoke", 600))
    expected_policy = copy.deepcopy(source_contract["policy_bundle"])
    expected_policy.update(authorized_scope=scope, workload_deadline_seconds=limit,
                           minimum_loaded_clock_samples=3)
    _same(monitor["policy_sha256"], canonical_sha256(expected_policy), "native frozen scoped policy")
    native_root = Path(source_contract["output_dir"]) / "native-hardware"
    final_reference = _reference(monitor.get("final_report_reference"), "native final report")
    if (final_reference["path"] != monitor_reference.get("monitor_final_report")
            or Path(final_reference["path"]) != native_root / "monitor-final.json"):
        _fail("native final report path differs from its owned invocation")
    final = _read_reference(final_reference, "native final report")
    _same(final, {key: value for key, value in monitor.items() if key != "final_report_reference"},
          "native closed report bytes")
    if final.get("sampled_clock_compliance") is not True:
        _fail("native final report lacks sampled 1500 MHz compliance")
    for key in _MONITOR_LEAVES:
        ref = _reference(final.get(key), "native " + key, verify=True)
        if Path(ref["path"]).parent != native_root:
            _fail("native leaf escaped its owned evidence directory")
    audit_ref = completion["capacity_memory_audit_reference"]
    if source_contract["stage"] == "capacity":
        audit = _read_reference(audit_ref, "capacity native memory audit")
        if (type(audit) is not dict or audit.get("status") != "PASS"
                or audit.get("kind") != "v2b_seed_resolution_replication_capacity_memory_audit"
                or audit.get("required_margin_mib") != 3072):
            _fail("capacity completion lacks the fixed 3 GiB memory acceptance")
        _same(audit.get("result_reference"), result_reference, "capacity audit result")
        _same(audit.get("monitor_final_report_reference"), final_reference, "capacity audit native report")
        for key in ("minimum_native_free_mib", "minimum_conservative_free_mib"):
            value = audit.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 3072:
                _fail("capacity audit memory margin is below 3 GiB")
    elif audit_ref is not None:
        _fail("only capacity completion may supply a capacity memory audit")
    return completion


def validate_replication_stage_gate(gate, contract, freeze, *, verify_files=True):
    """Authenticate prerequisite summaries and native leaves, without replaying old work."""
    _schema(gate, GATE_KEYS, GATE_KIND, "replication stage gate")
    if gate["status"] != "PASS":
        _fail("stage gate is not PASS")
    for key in ("campaign_id", "cell_id", "stage", "freeze_reference"):
        _same(gate[key], contract[key], "stage gate " + key)
    if type(gate["prerequisites"]) is not list:
        _fail("stage gate prerequisites must be an explicit list")
    indexed, results = {}, {}
    for row in gate["prerequisites"]:
        if type(row) is not dict or set(row) != {
                "cell_id", "stage", "worker_result_reference", "completion_reference"}:
            _fail("stage prerequisite schema mismatch")
        key = (row["cell_id"], row["stage"])
        if key in indexed or key[0] not in CELLS or key[1] not in STAGES:
            _fail("stage prerequisite identity is duplicate or unknown")
        for name in ("worker_result_reference", "completion_reference"):
            _reference(row[name], "prerequisite " + name)
        indexed[key] = row
    if set(indexed) != _expected_prerequisites(contract, freeze):
        _fail("stage gate prerequisites do not equal the frozen required dependency set")
    comparisons = gate["smoke_comparisons"]
    expected_comparisons = set(CELLS) if contract["stage"] == "control30" else set()
    if type(comparisons) is not dict or set(comparisons) != expected_comparisons:
        _fail("formal admission requires all four full smoke/replay comparisons")
    for cell, ref in comparisons.items():
        _reference(ref, cell + " smoke comparison")
    if contract["matched_reference"] is not None:
        peer = (f"s{contract['config']['seed']}_r640",
                "control30" if contract["stage"] == "control30" else "smoke")
        _same(indexed[peer]["worker_result_reference"], contract["matched_reference"], "matched prerequisite")
    if contract["replay_source"] is not None:
        _same(indexed[contract["cell_id"], "smoke"]["worker_result_reference"],
              contract["replay_source"], "replay prerequisite")
    if verify_files:
        for key, row in indexed.items():
            result, source_contract = _successful_cell_source(
                row["worker_result_reference"], cell_id=key[0], stage=key[1],
                contract=contract, freeze=freeze)
            _verify_completion(row["completion_reference"], row["worker_result_reference"],
                               result, source_contract, contract, freeze)
            results[key] = result
        for cell, ref in comparisons.items():
            comparison = _read_reference(ref, cell + " smoke comparison")
            if (type(comparison) is not dict or comparison.get("kind") != COMPARISON_KIND
                    or comparison.get("status") != "PASS" or comparison.get("cell_id") != cell
                    or comparison.get("campaign_id") != contract["campaign_id"]):
                _fail("smoke comparison is not a successful member of this campaign")
            _same(comparison.get("freeze_reference"), contract["freeze_reference"], "smoke comparison freeze")
            for field, stage in (("original_reference", "smoke"), ("replay_reference", "smoke_replay")):
                _same(comparison.get(field), indexed[cell, stage]["worker_result_reference"],
                      "smoke comparison " + field)
            _same(comparison.get("replay_tolerances"), control.REPLAY_TOLERANCES, "smoke comparison tolerances")
            if (comparison.get("restore_boundary_exact") is not True
                    or comparison.get("continuation_checkpoint_numerics", {}).get("status") != "PASS"
                    or len(comparison.get("windows", [])) != 2
                    or any(row.get("status") != "PASS" for row in comparison["windows"])):
                _fail("smoke comparison lacks the complete declared restoration acceptance")
    return copy.deepcopy(gate)


def verify_replication_worker_authorities(contract):
    """Qualify first; native admission is constructed only later by the worker."""
    qualified = _qualified_document(contract["qualification_reference"])
    freeze = validate_replication_freeze(
        _read_reference(contract["freeze_reference"], "replication freeze"), contract)
    _same(qualified.get("conditional_authorization_reference"), freeze["authorization_reference"],
          "qualification/experiment authorization")
    for key in ("authorization_reference", "foundation_reference", "clean_archive_reference"):
        _reference(freeze[key], "freeze " + key, verify=True)
    if Path(__file__).resolve() != Path(contract["repo_root"]) / "src/sparse_rtdetr/baseline/training_v2b_replication_worker.py":
        _fail("replication worker was imported from another checkout")
    for name, ref in contract["code_files"].items():
        _reference(ref, "code " + name, verify=True)
    from .training_v2b_replication_capacity import validate_replication_capacity_plan
    plan = _read_reference(contract["capacity_plan"], "replication capacity plan")
    validate_replication_capacity_plan(
        plan, seed=contract["config"]["seed"], input_size=contract["config"]["input_size"],
        train_core=contract["train_core"], verify_files=True)
    if contract["stage"] != "capacity":
        validate_replication_stage_gate(
            _read_reference(contract["stage_gate_reference"], "replication stage gate"),
            contract, freeze, verify_files=True)
    for name in ("annotation", "manifest"):
        _reference(contract["train_core"][name], "train_core " + name, verify=True)
    _reference(contract["pretrained"], "pretrained", verify=True)
    from .training_v2b_development import validate_development_binding
    validate_development_binding(contract["development_binding"], verify_files=True)
    return {"qualification": qualified, "freeze": freeze, "capacity_plan": plan}


def _common_identity(contract, binding, loader):
    return {
        "campaign_id": contract["campaign_id"], "cell_id": contract["cell_id"],
        "freeze_reference": copy.deepcopy(contract["freeze_reference"]),
        "qualification_reference": copy.deepcopy(contract["qualification_reference"]),
        "config": copy.deepcopy(binding["config"]), "code": copy.deepcopy(binding["code"]),
        "initial_weights": copy.deepcopy(binding["initial_weights"]),
        "train_core": copy.deepcopy(binding["data"]), "torch_version": str(torch.__version__),
        "worker_environment": validate_worker_environment(loader.config),
        "epoch_order_sha256": copy.deepcopy(contract["epoch_order_sha256"]),
        "development_binding_sha256": contract["development_binding"]["binding_sha256"],
        "policy_bundle_sha256": canonical_sha256(contract["policy_bundle"]),
        "capacity_plan": copy.deepcopy(contract["capacity_plan"]),
    }


def _integrations():
    # Complete integration imports precede both CPU and CUDA model construction.
    from .training_v2b_admission import MonitoredHardwareSession
    from .training_v2b_development import evaluate_development, preview_development
    from . import training_v2b_replication_capacity, training_v2b_replication_gate
    from . import training_v2b_resolution
    return MonitoredHardwareSession, evaluate_development, preview_development


def _new_report(contract, contract_reference):
    return {
        "schema_version": 1, "kind": RESULT_KIND, "status": "RUNNING",
        "stage": contract["stage"], "cell_id": contract["cell_id"],
        "seed": contract["config"]["seed"], "input_size": contract["config"]["input_size"],
        "campaign_id": contract["campaign_id"], "run_id": contract["run_id"],
        "invocation_id": contract["invocation_id"], "worker_pid": os.getpid(),
        "contract_reference": copy.deepcopy(contract_reference),
        "freeze_reference": copy.deepcopy(contract["freeze_reference"]),
        "qualification_reference": copy.deepcopy(contract["qualification_reference"]),
        "capacity_plan": copy.deepcopy(contract["capacity_plan"]),
        "receipts": [], "checkpoints": {}, "evaluations": [], "replay_checks": [],
        "guardian_clearance_required_after_worker_exit": True,
        "scientific_certified": False, "automatic_retry_or_batch_fallback": False,
    }


def _native_identity(value, label):
    if (type(value) is not dict or value.get("readable") is not True
            or type(value.get("pid")) is not int or value["pid"] <= 0
            or type(value.get("start_ticks")) is not int or value["start_ticks"] < 0):
        _fail(label + " lacks a readable PID/start-time identity")
    _reference(value.get("executable"), label + " executable")
    return value


def _invocation_sources(contract, freeze, report, binding, initialization):
    """Read only the sources already authenticated by the complete stage gate."""
    matched = original = own_smoke = None
    matched_index, replay_index = None, {}
    if contract["matched_reference"] is not None:
        from .training_v2b_resolution import compare_matched_initialization
        stage = "control30" if contract["stage"] == "control30" else "smoke"
        matched, _ = _successful_cell_source(
            contract["matched_reference"], cell_id=f"s{contract['config']['seed']}_r640",
            stage=stage, contract=contract, freeze=freeze)
        report["matched_reference"] = copy.deepcopy(contract["matched_reference"])
        report["matched_initialization"] = compare_matched_initialization(matched, initialization)
        matched_index = control._receipt_index(matched)
        expected = ({(e, i) for e in range(1, 31) for i in range(304)}
                    if stage == "control30" else {(1, i) for i in range(4)})
        if set(matched_index) != expected:
            _fail("same-seed matched 640 source does not cover the complete declared stage")
    if contract["replay_source"] is not None:
        original, _ = _successful_cell_source(
            contract["replay_source"], cell_id=contract["cell_id"], stage="smoke",
            contract=contract, freeze=freeze)
        if original["run_id"] != contract["run_id"] or original["worker_pid"] == os.getpid():
            _fail("smoke replay requires a different process and the same checkpoint run identity")
        _same(original["common_identity"], report["common_identity"], "same-cell replay common identity")
        _same(original["initialization"], initialization, "same-cell replay CPU initialization")
        _same(_read_reference(original["binding_reference"], "original smoke binding"),
              binding, "same-cell checkpoint run binding")
        replay_index = control._receipt_index(original)
        if set(replay_index) != {(1, i) for i in range(4)}:
            _fail("original smoke must contain exactly the four declared windows")
    if contract["stage"] == "control30":
        gate = _read_reference(contract["stage_gate_reference"], "formal stage gate")
        row = next(row for row in gate["prerequisites"]
                   if row["cell_id"] == contract["cell_id"] and row["stage"] == "smoke")
        own_smoke, _ = _successful_cell_source(
            row["worker_result_reference"], cell_id=contract["cell_id"], stage="smoke",
            contract=contract, freeze=freeze)
        _same(own_smoke["common_identity"], report["common_identity"], "formal/smoke common identity")
        _same(own_smoke["initialization"], initialization, "fresh formal/smoke CPU initialization")
        report["fresh_initialization_smoke_reference"] = copy.deepcopy(row["worker_result_reference"])
    return matched, original, own_smoke, matched_index, replay_index


def _execute_training_stage(contract, session, monitor, output, report, *,
                            original, matched_index, replay_index,
                            evaluate_development, preview_development):
    """One frozen invocation; each epoch owns exactly one loader iterator."""
    components, loader = session.components, session.loader
    checkpoints, receipts = report["checkpoints"], report["receipts"]
    replay_checks, evaluations = report["replay_checks"], report["evaluations"]

    def evaluation_gate():
        torch.cuda.synchronize(0)
        return monitor.check(stage="development_batch")

    component_map = {
        name: getattr(components, name) for name in
        ("model", "ema", "postprocessor", "engine", "runtime", "optimizer", "scheduler", "warmup")
    }

    def start_epoch(epoch):
        session.begin_epoch(epoch)
        if loader.state_dict()["order_sha256"] != contract["epoch_order_sha256"][str(epoch)]:
            _fail("actual seed-specific sampler differs from its frozen complete epoch plan")

    if contract["stage"] in ("smoke", "smoke_replay"):
        if original is None:
            start_epoch(1)
        else:
            midpoint = original["checkpoints"]["window_2"]
            control._reference(midpoint["checkpoint"], "smoke midpoint checkpoint", verify=True,
                               expected_name="checkpoint-window-02.pt")
            monitor.check(stage="before_restore")
            session.restore(midpoint["checkpoint"]["path"], midpoint["checkpoint"]["sha256"])
            torch.cuda.synchronize(0)
            monitor.check(stage="after_restore")
            restored = control.capture_control_state(components, loader, full=True)
            _same(_read_reference(midpoint["state_reference"], "checkpoint boundary state"),
                  restored, "complete checkpoint restored state")
            if (components.engine.optimizer_updates != 2 or components.engine.epoch != 1
                    or not components.engine.epoch_active):
                _fail("smoke replay checkpoint is not the active second-window boundary")
            report["restore_boundary_reference"] = write_exclusive_json(
                output / "restored-boundary-state.json", restored)
            report["restore_boundary_bn_snapshot_reference"] = control._publish_bn_snapshot(
                output / "bn-state-restored-boundary-02.pt", components,
                {"run_binding_sha256": session.binding["binding_sha256"], "state": restored,
                 "window": {"epoch": components.engine.epoch,
                            "optimizer_updates": components.engine.optimizer_updates}})
            report["restored_from"] = copy.deepcopy(midpoint)
        control._run_active_windows(
            session, monitor, output, limit=4 if original is None else 2,
            paired_index={}, replay_index=replay_index, save_midpoint=original is None,
            snapshots=True, checkpoints=checkpoints, receipts=receipts,
            replay_checks=replay_checks, matched_index=matched_index)
        if components.engine.optimizer_updates != 4:
            _fail("smoke must end after exactly four successful logical updates")
        checkpoints["window_4"] = control._write_checkpoint(
            session, output, "checkpoint-window-04", monitor)
        before = control.capture_control_state(components, loader, full=True)
        preview_dir = output / "development-preview"
        preview_dir.mkdir()
        evaluations.append(preview_development(
            component_map, data_binding=contract["development_binding"],
            output_dir=preview_dir, hardware_gate=evaluation_gate))
        _same(before, control.capture_control_state(components, loader, full=True),
              "development preview preserved complete training state")
        expected_receipts = 4 if original is None else 2
        expected_checks = 0 if original is None else 2
        if len(receipts) != expected_receipts or len(replay_checks) != expected_checks or len(evaluations) != 1:
            _fail("smoke/replay coverage or preview count differs")
    elif contract["stage"] == "control30":
        if original is not None or replay_index:
            _fail("formal replication starts from fresh pretrained initialization without resume")
        for epoch in range(1, 31):
            start_epoch(epoch)
            control._run_active_windows(
                session, monitor, output, limit=304, paired_index={}, replay_index={},
                save_midpoint=False, snapshots=False, checkpoints=checkpoints,
                receipts=receipts, replay_checks=replay_checks, matched_index=matched_index)
            epoch_record = session.finish_epoch()
            checkpoints[f"epoch_{epoch}"] = control._write_checkpoint(
                session, output, f"checkpoint-epoch-{epoch:03d}", monitor)
            write_exclusive_json(output / f"epoch-{epoch:03d}.json", epoch_record)
            if epoch in (10, 20, 30):
                evaluation_dir = output / f"development-epoch-{epoch:03d}"
                evaluation_dir.mkdir()
                before = control.capture_control_state(components, loader, full=True)
                evaluations.append(evaluate_development(
                    component_map, data_binding=contract["development_binding"],
                    output_dir=evaluation_dir, logged_epoch=epoch, hardware_gate=evaluation_gate))
                _same(before, control.capture_control_state(components, loader, full=True),
                      "development evaluation preserved complete training state")
        if (components.engine.optimizer_updates != 9120 or components.engine.epoch_active
                or len(receipts) != 9120 or len(evaluations) != 3 or len(checkpoints) != 30
                or replay_checks):
            _fail("the complete thirty-epoch execution ledger differs")
    else:
        _fail("capacity must use its independent disposable worker")


def _final_ledger(contract, report, final):
    stage = contract["stage"]
    expected_updates = 9120 if stage == "control30" else 4
    expected_epoch = 30 if stage == "control30" else 1
    state = final["clocks"]["engine"]
    for key, expected in (
        ("epoch", expected_epoch), ("epoch_active", stage != "control30"),
        ("optimizer_updates", expected_updates), ("microsteps", expected_updates * 2),
    ):
        _same(state[key], expected, "final execution clock " + key)
    expected_positions = (
        {(e, i) for e in range(1, 31) for i in range(304)} if stage == "control30"
        else {(1, 2), (1, 3)} if stage == "smoke_replay"
        else {(1, i) for i in range(4)}
    )
    if set(control._receipt_index(report)) != expected_positions:
        _fail("final receipt positions differ from the exact invocation scope")
    return expected_updates


def run_replication_worker(contract, *, contract_reference):
    """Execute one separately authorized invocation; this never launches another worker."""
    checked = validate_replication_worker_contract(contract, verify_files=False)
    _same(_read_reference(contract_reference, "worker contract"), checked, "invoked contract bytes")
    if checked["stage"] == "capacity":
        from .training_v2b_replication_capacity import run_replication_capacity_worker
        return run_replication_capacity_worker(checked, contract_reference=contract_reference)
    output = Path(checked["output_dir"])
    output.mkdir(parents=False, exist_ok=False)
    report, monitor = _new_report(checked, contract_reference), None
    try:
        MonitoredHardwareSession, evaluate_development, preview_development = _integrations()
        report["startup_environment"] = control._environment(checked)
        authorities = verify_replication_worker_authorities(checked)
        write_exclusive_json(output / "worker-start.json", report)
        config = V2BConfig(**checked["config"])
        data = checked["train_core"]
        loader = build_train_core_loader(
            TrainCoreDataConfig(
                seed=config.seed, input_size=config.input_size, logical_batch_size=16,
                num_workers=checked["num_workers"], prefetch_factor=checked["prefetch_factor"]),
            annotation_file=data["annotation"]["path"], annotation_sha256=data["annotation"]["sha256"],
            manifest_file=data["manifest"]["path"], manifest_sha256=data["manifest"]["sha256"],
            image_root=data["image_root"], repo_root=checked["repo_root"])
        if len(loader.dataset) != 4869 or loader.batches_per_epoch != 304:
            _fail("actual train_core loader differs from the frozen full epoch")
        arguments = {
            "repo_root": checked["repo_root"], "pretrained_path": checked["pretrained"]["path"],
            "pretrained_sha256": checked["pretrained"]["sha256"],
        }
        cpu = build_v2b_components(config, **arguments)
        binding = build_train_core_run_binding(
            cpu, loader, run_id=checked["run_id"], repo_root=checked["repo_root"],
            additional_code_paths={name: ref["path"] for name, ref in checked["code_files"].items()},
            requested_device="cuda:0", cuda_gpu_uuid=checked["expected_gpu_uuid"])
        _same(binding["code"], checked["code_files"], "actual complete imported source inventory")
        report["binding_reference"] = write_exclusive_json(output / "run-binding.json", binding)
        report["initialization"] = cpu.initialization
        report["common_identity"] = _common_identity(checked, binding, loader)
        matched, original, own_smoke, matched_index, replay_index = _invocation_sources(
            checked, authorities["freeze"], report, binding, cpu.initialization)
        monitor = MonitoredHardwareSession(
            binding, checked["policy_bundle"], output / "native-hardware",
            "train_core_30epoch" if checked["stage"] == "control30" else "paired_smoke",
            43200 if checked["stage"] == "control30" else 600)
        monitor.start()
        runtime = prepare_runtime(
            device="cuda:0", seed=config.seed, binding=binding,
            gpu_probe=monitor.admission(), expected_gpu_uuid=checked["expected_gpu_uuid"])
        components = build_v2b_components(config, runtime=runtime, **arguments)
        _same(cpu.initialization, components.initialization, "actual CPU/CUDA initialized model")
        del cpu
        session = V2BTrainingSession(components, loader, binding)
        report["runtime"] = runtime.identity
        initial = control.capture_control_state(components, loader, full=True)
        report["initial_state_reference"] = write_exclusive_json(output / "initial-state.json", initial)
        if matched is not None:
            from .training_v2b_resolution import compare_matched_initial_state
            report["matched_cuda_initialization"] = compare_matched_initial_state(
                matched, initial, runtime.identity)
        for source in (original, own_smoke):
            if source is not None:
                _same(source["runtime"], report["runtime"], "same-cell actual CUDA runtime")
                control._verify_common_initial_state(
                    _read_reference(source["initial_state_reference"], "same-cell initial state"), initial)
        _execute_training_stage(
            checked, session, monitor, output, report, original=original,
            matched_index=matched_index, replay_index=replay_index,
            evaluate_development=evaluate_development, preview_development=preview_development)
        final = control.capture_control_state(components, loader, full=True)
        updates = _final_ledger(checked, report, final)
        report["final_state_reference"] = write_exclusive_json(output / "final-state.json", final)
        report["receipt_count"] = len(report["receipts"])
        report["logical_samples_committed"] = updates * 16
        report["logical_samples_executed_this_invocation"] = len(report["receipts"]) * 16
        report["replay_tolerances"] = copy.deepcopy(control.REPLAY_TOLERANCES)
        if checked["stage"] == "control30":
            report.update(primary_endpoint_epoch=30, health_only_epochs=[10, 20],
                          formal_started_from_pretrained_update_zero=True,
                          formal_checkpoint_resume_used=False)
        report["monitor_reference"] = monitor.finish()
        report["status"] = "PASS"
        write_exclusive_json(output / "worker-result.json", report)
        return report
    except BaseException as exc:
        report["status"] = "STOP_NO_RETRY"
        chain, cause, seen = [], exc, set()
        while cause is not None and id(cause) not in seen:
            seen.add(id(cause))
            row = {"type": type(cause).__name__, "message": str(cause)}
            observation = getattr(cause, "replay_observation_reference", None)
            if observation is not None:
                row["observation_reference"] = copy.deepcopy(observation)
            chain.append(row)
            cause = cause.__cause__
        report["failure"] = chain
        try:
            write_exclusive_json(output / "worker-result.json", report)
        finally:
            if monitor is not None:
                monitor.abort("replication worker failure: " + type(exc).__name__ + ": " + str(exc))
        raise


def compare_replication_smoke(original_reference, replay_reference, *, freeze_reference):
    """Authenticate one same-cell replay on CPU; the controller closes each guardian."""
    try:
        base = _read_reference(original_reference, "original replication smoke")
        contract = _read_reference(base["contract_reference"], "original smoke contract")
        validate_replication_worker_contract(contract, verify_files=False)
        _same(contract["freeze_reference"], freeze_reference, "compared freeze")
        freeze = validate_replication_freeze(_read_reference(freeze_reference, "replication freeze"), contract)
        base, _ = _successful_cell_source(
            original_reference, cell_id=contract["cell_id"], stage="smoke", contract=contract, freeze=freeze)
        replay, replay_contract = _successful_cell_source(
            replay_reference, cell_id=contract["cell_id"], stage="smoke_replay",
            contract=contract, freeze=freeze)
        _same(replay_contract["replay_source"], original_reference, "restored original smoke source")
        if (base["run_id"] != replay["run_id"] or base["worker_pid"] == replay["worker_pid"]
                or base["invocation_id"] == replay["invocation_id"]):
            _fail("same-cell replay must have a fresh invocation/PID and unchanged checkpoint run identity")
        for name in ("common_identity", "runtime", "initialization"):
            _same(base[name], replay[name], "same-cell smoke " + name)
        for result in (base, replay):
            _same(result["replay_tolerances"], control.REPLAY_TOLERANCES, "frozen replay tolerances")
        control._verify_common_initial_state(
            _read_reference(base["initial_state_reference"], "original initialization"),
            _read_reference(replay["initial_state_reference"], "replay initialization"))
        midpoint = base["checkpoints"]["window_2"]
        control._reference(midpoint["checkpoint"], "original midpoint checkpoint", verify=True,
                           expected_name="checkpoint-window-02.pt")
        _same(replay["restored_from"], midpoint, "restored checkpoint reference")
        _same(_read_reference(midpoint["state_reference"], "original checkpoint boundary"),
              _read_reference(replay["restore_boundary_reference"], "restored checkpoint boundary"),
              "restored complete checkpoint boundary")
        base_index, replay_index = control._receipt_index(base), control._receipt_index(replay)
        if set(base_index) != {(1, i) for i in range(4)} or set(replay_index) != {(1, 2), (1, 3)}:
            _fail("smoke/replay did not execute exactly four/two declared windows")
        windows = []
        for key in sorted(replay_index):
            comparison = control.compare_replay_records(
                control._read_json_reference(base_index[key], "original logical window"),
                control._read_json_reference(replay_index[key], "replayed logical window"))
            windows.append({**comparison, "epoch": key[0], "logical_batch_index": key[1]})
        base_final = _read_reference(base["final_state_reference"], "original final state")
        replay_final = _read_reference(replay["final_state_reference"], "replay final state")
        _final_ledger(contract, base, base_final)
        _final_ledger(replay_contract, replay, replay_final)
        for name in ("clocks", "rng", "raw_bn", "ema_bn"):
            _same(base_final[name], replay_final[name], "same-cell final " + name)
        binding = _read_reference(base["binding_reference"], "original smoke binding")
        _same(binding, _read_reference(replay["binding_reference"], "replay binding"),
              "smoke continuation checkpoint binding")
        continuation = control.compare_continuation_checkpoints(
            base["checkpoints"]["window_4"]["checkpoint"],
            replay["checkpoints"]["window_4"]["checkpoint"], binding=binding)
        preview = control._compare_previews(control._preview_evidence(base), control._preview_evidence(replay))
        return {
            "schema_version": 1, "kind": COMPARISON_KIND, "status": "PASS",
            "campaign_id": contract["campaign_id"], "cell_id": contract["cell_id"],
            "freeze_reference": copy.deepcopy(freeze_reference),
            "original_reference": copy.deepcopy(original_reference),
            "replay_reference": copy.deepcopy(replay_reference),
            "restore_boundary_exact": True, "windows": windows,
            "continuation_checkpoint_numerics": continuation,
            "replay_tolerances": copy.deepcopy(control.REPLAY_TOLERANCES),
            "development_preview": preview,
            "ema_final_state_exact": base_final["ema_state_sha256"] == replay_final["ema_state_sha256"],
            "optimizer_final_state_exact": base_final["optimizer_state_sha256"] == replay_final["optimizer_state_sha256"],
            "development_numerical_equalities_are_observations_not_acceptance_thresholds": True,
            "hardware_clearance": "controller_authenticates_native_final_after_each_owner_exit",
            "scientific_certified": False, "primary_endpoint_epoch": 30, "health_only_epochs": [10, 20],
        }
    except Exception as exc:
        return {
            "schema_version": 1, "kind": COMPARISON_KIND, "status": "STOP_NO_RETRY",
            "freeze_reference": copy.deepcopy(freeze_reference),
            "original_reference": copy.deepcopy(original_reference),
            "replay_reference": copy.deepcopy(replay_reference),
            "failure_type": type(exc).__name__, "failure": str(exc), "scientific_certified": False,
        }
