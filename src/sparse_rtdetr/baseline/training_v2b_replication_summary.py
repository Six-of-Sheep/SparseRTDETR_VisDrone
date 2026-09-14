"""Fixed epoch-30 seed replication summary from immutable saved evidence.

The primary point estimates use the existing complete-lineage official GT adapter
and full primary evaluator, with an independent cache point check. COCO small
metrics retain the converted COCO GT contract. This module has no model, image,
checkpoint-deserialization, training, clock-setting or GPU-admission entry point.
"""
from __future__ import annotations

from collections import Counter
import copy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import statistics
import time

from .training_v2b_evidence import canonical_json_bytes, canonical_sha256, strict_json_loads, write_exclusive_json


KIND = "v2b_seed_resolution_replication_fixed_epoch_summary"
CELLS = ("s1_r640", "s1_r896", "s2_r640", "s2_r896")
SIZES = (640, 896)
SEEDS = (0, 1, 2)
PRIMARY_METRICS = ("AP", "AP50", "AP75", "AR1", "AR10", "AR100", "AR500")
COCO_METRICS = ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
                "AR100", "AR_small", "AR_medium", "AR_large")
REF_KEYS = {"path", "sha256", "size_bytes", "sha256_scope"}
NATIVE_KEYS = ("run_id", "run_binding_sha256", "policy_sha256", "monitor_pid",
               "owner", "guardian_identity", "gpu_uuid")
NATIVE_LEAVES = ("samples", "setter_receipt", "startup_evidence", "guardian_heartbeat")
POINT_TOLERANCE = 1e-10


class ReplicationSummaryError(ValueError):
    """Incomplete fixed-epoch evidence or a changed pairing/metric authority."""


def _require(condition, message):
    if not condition:
        raise ReplicationSummaryError(message)


def _same(actual, expected, message):
    _require(canonical_sha256(actual) == canonical_sha256(expected), message + " differs")


def _number(value, name, *, minimum=None, maximum=None):
    _require(type(value) in (int, float) and math.isfinite(value), name + " must be finite")
    _require(minimum is None or value >= minimum, name + " is below its bound")
    _require(maximum is None or value <= maximum, name + " exceeds its bound")
    return float(value)


def _integer(value, name, *, minimum=0):
    _require(type(value) is int and value >= minimum, name + " must be a builtin integer")
    return value


def _path(value):
    _require(type(value) is str and value and Path(value).is_absolute(), "explicit absolute path required")
    path = Path(value)
    forbidden = {"confirmatory", "test", "visdrone2019-det-test-dev", "visdrone2019-det-test-challenge"}
    _require(not any(part.casefold() in forbidden or part.casefold().startswith("confirmatory_")
                     for part in path.parts), "path crosses a forbidden data role")
    _require(str(path) == value and ".." not in path.parts and path.resolve() == path,
             "path must be canonical without symlinks")
    return path


def _reference(reference, *, embedded=False):
    required = REF_KEYS - {"sha256_scope"} if embedded else REF_KEYS
    _require(type(reference) is dict and set(reference) in (required, REF_KEYS),
             "complete-file reference schema differs")
    normalized = copy.deepcopy(reference)
    normalized.setdefault("sha256_scope", "complete_file_bytes")
    _path(normalized["path"])
    _require(normalized["sha256_scope"] == "complete_file_bytes"
             and type(normalized["sha256"]) is str
             and re.fullmatch("[0-9a-f]{64}", normalized["sha256"]) is not None,
             "complete-file SHA-256 is required")
    _integer(normalized["size_bytes"], "reference byte size")
    return normalized


def _checkpoint_reference(value):
    _require(type(value) is dict and {"path", "sha256", "size_bytes"}.issubset(value),
             "checkpoint reference is incomplete")
    return _reference({key: value.get(key, "complete_file_bytes") for key in REF_KEYS})


def _byte_identity(reference):
    value = _reference(reference, embedded=True)
    return {"sha256": value["sha256"], "size_bytes": value["size_bytes"],
            "sha256_scope": value["sha256_scope"]}


class _Reader:
    """Hash every consumed immutable file; do not retain thousands of receipt bytes."""

    def __init__(self):
        self.references = {}

    def read(self, reference, *, embedded=False, retain=True):
        ref = _reference(reference, embedded=embedded)
        path = _path(ref["path"])
        if ref["path"] in self.references:
            _same(ref, self.references[ref["path"]], "one path's evidence identity")
        directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for part in path.parent.parts[1:]:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
                os.close(directory)
                directory = child
            descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        finally:
            os.close(directory)
        chunks, count, digest = [], 0, hashlib.sha256()
        with os.fdopen(descriptor, "rb") as stream:
            before = os.fstat(stream.fileno())
            _require(stat.S_ISREG(before.st_mode) and before.st_size == ref["size_bytes"],
                     "evidence is not the bound regular file")
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                count += len(chunk)
                digest.update(chunk)
                if retain:
                    chunks.append(chunk)
            after = os.fstat(stream.fileno())
        for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"):
            _require(getattr(before, key) == getattr(after, key), "evidence changed while being read")
        _require(count == ref["size_bytes"] and digest.hexdigest() == ref["sha256"],
                 "complete-file SHA or byte-size mismatch: " + path.name)
        self.references[ref["path"]] = ref
        return b"".join(chunks) if retain else None

    def json(self, reference, *, embedded=False):
        return strict_json_loads(self.read(reference, embedded=embedded))

    def discover(self, path):
        """Only the caller's exact checkpoint-derived historical evidence path is accepted."""
        path = _path(str(path))
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            _require(stat.S_ISREG(before.st_mode), "discovered evidence is not a regular file")
            digest = hashlib.file_digest(stream, "sha256") if hasattr(hashlib, "file_digest") else None
            if digest is None:
                stream.seek(0)
                digest = hashlib.sha256()
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            after = os.fstat(stream.fileno())
        _require((before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
                 (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
                 "discovered evidence changed while being hashed")
        reference = {"path": str(path), "sha256": digest.hexdigest(), "size_bytes": before.st_size,
                     "sha256_scope": "complete_file_bytes"}
        self.read(reference, retain=False)
        return reference

    def inventory(self):
        return [self.references[key] for key in sorted(self.references)]


def descriptive_statistics(values):
    """Descriptive sample statistics only; n=3 does not establish significance."""
    numbers = [_number(value, "paired seed value") for value in values]
    _require(len(numbers) >= 2, "sample SD requires at least two observations")
    return {"n": len(numbers), "values": numbers, "mean": statistics.mean(numbers),
            "sample_sd": statistics.stdev(numbers), "sample_sd_ddof": 1,
            "minimum": min(numbers), "maximum": max(numbers), "range": max(numbers) - min(numbers),
            "statistical_significance_claimed": False}


def _state_mapping(value):
    _require(type(value) is dict and value.get("mapping_type") == "dict"
             and type(value.get("items")) is list, "saved scalar-state mapping is incomplete")
    rows = value["items"]
    _require(all(type(row) is list and len(row) == 3 and row[0] == "str" for row in rows),
             "saved scalar-state mapping has an invalid key")
    _require(len({row[1] for row in rows}) == len(rows), "saved scalar-state keys repeat")
    return {row[1]: row[2] for row in rows}


def _state_clocks(state, *, input_size, initial=False):
    clocks, expected_updates = state["clocks"], 0 if initial else 9120
    engine, loader = clocks["engine"], clocks["loader"]
    expected_epoch = 0 if initial else 30
    for name, expected in (
        ("epoch", expected_epoch), ("epoch_active", False), ("optimizer_updates", expected_updates),
        ("microsteps", 2 * expected_updates), ("epoch_start_optimizer_updates", 0 if initial else 8816),
        ("phase", 0), ("failed", False),
    ):
        _same(engine[name], expected, "fixed endpoint engine " + name)
    for name, expected in (("physical_batch_size", 8), ("accumulation_steps", 2),
                           ("expected_input_size", [input_size, input_size])):
        _same(engine["config"][name], expected, "fixed endpoint engine configuration " + name)
    for name, expected in (
        ("epoch", expected_epoch), ("epoch_active", False), ("optimizer_updates", expected_updates),
        ("logical_batch_size", 16), ("dataset_size", 4869), ("batches_per_epoch", 304),
        ("dropped_samples", 5), ("completed_batches", 0 if initial else 304),
    ):
        _same(loader[name], expected, "fixed endpoint loader " + name)
    _same(clocks["ema_updates"], expected_updates, "fixed endpoint EMA clock")
    warmup, scheduler = _state_mapping(clocks["warmup"]), _state_mapping(clocks["scheduler"])
    _same(warmup["last_step"], expected_updates, "fixed endpoint warmup clock")
    _same(warmup["warmup_duration"], 2000, "fixed warmup duration")
    _same(scheduler["last_epoch"], 0 if initial else 24, "fixed endpoint scheduler epoch")
    _same(scheduler["_step_count"], 1 if initial else 25, "fixed endpoint scheduler steps")
    # Per-parameter Adam steps may legitimately lag when a branch had no gradient.
    for name, step in clocks.get("adam_steps_by_parameter", {}).items():
        _integer(step, "Adam step " + name)
        _require(step <= expected_updates, "Adam step exceeds committed updates")


def _owned(reference, output):
    path = _path(_reference(reference, embedded=True)["path"])
    _require(path.is_relative_to(output) and path != output, "evidence leaf escaped its invocation")


def _evaluation(reader, result, contract):
    evaluations = result.get("evaluations")
    _require(type(evaluations) is list and [row.get("logged_epoch") for row in evaluations] == [10, 20, 30],
             "training must retain exactly fixed epoch 10/20/30 evaluations")
    evaluation = evaluations[-1]
    for key, expected in (
        ("role", "development"), ("logged_epoch", 30), ("full_development_evaluation", True),
        ("checkpoint_selection_performed", False), ("protocol_certification_claimed", False),
        ("scope", "development_full_fixed_checkpoint"), ("scientific_comparison_eligible", True),
    ):
        _same(evaluation.get(key), expected, "fixed development " + key)
    _same(evaluation.get("rng_restored"),
          ["python", "numpy", "torch_cpu", "torch_cuda_bound_device"], "fixed development RNG restoration")
    unchanged = evaluation.get("training_state_unchanged")
    _require(type(unchanged) is dict, "development complete-state preservation evidence is absent")
    final = reader.json(result["final_state_reference"])
    _same(unchanged.get("engine"), final["clocks"]["engine"], "development preserved final engine")
    _same(unchanged.get("ema_updates"), 9120, "development preserved EMA update clock")
    _same(unchanged.get("ema_sha256"), evaluation["weights"]["state_sha256"],
          "development evaluated the preserved EMA state")
    auxiliary = unchanged.get("auxiliary_clocks", {})
    _same(auxiliary.get("warmup", {}).get("last_step"), 9120, "development preserved warmup clock")
    _same(auxiliary.get("scheduler", {}).get("last_epoch"), 24, "development preserved scheduler clock")
    if "geometry" in final:
        _same(unchanged.get("geometry"), final["geometry"], "development preserved final geometry")
    _same(evaluation["weights"].get("kind"), "ema", "fixed development weights")
    _same(evaluation["weights"].get("ema_updates"), 9120, "fixed development EMA updates")
    _same(evaluation["data_binding_sha256"], contract["development_binding"]["binding_sha256"],
          "development binding")
    _same(evaluation.get("policy"), contract["development_binding"]["policy"],
          "fixed development evaluation policy")
    _same(evaluation["policy"].get("input_size"), [contract["config"]["input_size"]] * 2,
          "development evaluation resolution matches its labeled training endpoint")
    _same(evaluation["runtime"], result["runtime"], "development/training runtime")
    for name in ("evaluated_image_count", "bound_image_count"):
        _same(evaluation["data"][name], 548, "development image coverage")
    _same(evaluation["data"]["evaluated_ground_truth_count"], 38759, "converted development GT count")
    image_receipts = evaluation["data"]["image_receipts"]
    _require(type(image_receipts) is list and len(image_receipts) == 548,
             "development image receipt coverage differs")
    _same(canonical_sha256(image_receipts), contract["development_binding"]["image_manifest_sha256"],
          "actual development image receipt identities")
    published = reader.json(evaluation["result_artifact"])
    _same(published, {k: v for k, v in evaluation.items() if k != "result_artifact"},
          "published fixed development result")
    _owned(evaluation["prediction_artifact"], Path(contract["output_dir"]))
    _owned(evaluation["result_artifact"], Path(contract["output_dir"]))
    return evaluation


def _native(reader, result, contract, binding, *, completion=None):
    finish = result["monitor_reference"]
    path = Path(contract["output_dir"]) / "native-hardware/monitor-final.json"
    _same(finish.get("monitor_final_report"), str(path), "owned native final path")
    final_ref = (completion["monitor"]["final_report_reference"] if completion is not None
                 else reader.discover(path))
    final = reader.json(final_ref)
    for key in NATIVE_KEYS:
        _require(key in final and key in finish, "native identity field is absent")
        _same(final[key], finish[key], "native finish " + key)
    for key, expected in (
        ("status", "PASS"), ("worker_exited", True), ("sampled_clock_compliance", True),
        ("authorized_scope", "train_core_30epoch"), ("settings_after_exit", "keep_1500_1500"),
    ):
        _same(final.get(key), expected, "native fixed workload " + key)
    _same(final["owner"].get("pid"), result["worker_pid"], "native owner PID")
    for identity in (final["owner"], final["guardian_identity"]):
        _require(identity.get("readable") is True, "native PID identity is unreadable")
        _integer(identity.get("pid"), "native PID", minimum=1)
        _integer(identity.get("start_ticks"), "native process start ticks")
        _reference(identity["executable"])
    _same(final["monitor_pid"], final["guardian_identity"]["pid"], "native guardian PID")
    _same(final["run_id"], binding["run_id"], "native workload run")
    _same(final["run_binding_sha256"], binding["binding_sha256"], "native workload binding")
    _same(final["gpu_uuid"], contract["expected_gpu_uuid"], "native GPU UUID")
    policy = copy.deepcopy(contract["policy_bundle"])
    policy.update(authorized_scope="train_core_30epoch", workload_deadline_seconds=43200,
                  minimum_loaded_clock_samples=3)
    _same(final["policy_sha256"], canonical_sha256(policy), "native scoped policy")
    for key in NATIVE_LEAVES:
        _owned(final[key], path.parent)
        reader.read(final[key], retain=False)
    if completion is not None:
        _same(completion["monitor"], {**final, "final_report_reference": final_ref},
              "controller/native terminal report")
    return {"final_report_reference": final_ref,
            "source_native_status": final["status"], "sampled_clock_compliance": True,
            "gpu_uuid": final["gpu_uuid"], "owner": final["owner"],
            "guardian_identity": final["guardian_identity"],
            "policy_sha256": final["policy_sha256"],
            "native_leaves": {name: final[name] for name in NATIVE_LEAVES},
            "complete_leaf_bytes_authenticated": True, "per_sample_hardware_reaudit_performed": False,
            "live_hardware_preflight_performed": False}


def _training_source(reader, result_reference, *, seed, size, expected_config,
                     expected_checkpoint=None, expected_binding_sha=None, completion=None):
    from . import training_v2b_control as control
    from .training_v2b_replication_worker import frozen_seed_orders
    result = reader.json(result_reference)
    _require(result.get("status") == "PASS" and result.get("stage") == "control30",
             "a complete successful formal training result is required")
    for key, value in (("receipt_count", 9120), ("logical_samples_committed", 145920),
                       ("logical_samples_executed_this_invocation", 145920),
                       ("scientific_certified", False), ("automatic_retry_or_batch_fallback", False),
                       ("guardian_clearance_required_after_worker_exit", True)):
        _same(result.get(key), value, "formal worker " + key)
    _integer(result.get("worker_pid"), "formal worker PID", minimum=1)
    contract = reader.json(result["contract_reference"])
    _same(contract.get("config"), expected_config, "seed/resolution training configuration")
    _same(contract.get("stage"), "control30", "formal source stage")
    _same(result["run_id"], contract["run_id"], "formal source run")
    _same(result["campaign_id"], contract["campaign_id"], "formal source campaign")
    output = _path(contract["output_dir"])
    _same(result_reference["path"], str(output / "worker-result.json"), "owned formal result")
    binding = reader.json(result["binding_reference"])
    _same(binding.get("binding_sha256"), canonical_sha256(
        {k: v for k, v in binding.items() if k != "binding_sha256"}), "training binding digest")
    _same(binding["run_id"], result["run_id"], "training binding run")
    if expected_binding_sha is not None:
        _same(binding["binding_sha256"], expected_binding_sha, "qualified seed-zero checkpoint binding")
    for key in ("seed", "input_size", "physical_batch_size", "accumulation_steps", "sampling_backend"):
        _same(binding["config"][key], expected_config[key], "actual bound training " + key)
    plans = control.frozen_epoch_orders() if seed == 0 else frozen_seed_orders(seed)
    _same(contract["epoch_order_sha256"], plans, "seed-specific frozen thirty-epoch plan")
    _same(result["common_identity"]["epoch_order_sha256"], plans, "actual frozen thirty-epoch plan")
    _same(result["initialization"].get("seed"), seed, "actual initialization seed")
    _require(result["initialization"].get("seed_initialized_before_model") is True,
             "model was not initialized after its actual seed")
    checkpoints = result["checkpoints"]
    _same(sorted(checkpoints), sorted(f"epoch_{epoch}" for epoch in range(1, 31)),
          "exact fixed checkpoint epoch set")
    for epoch in range(1, 31):
        checkpoint = checkpoints[f"epoch_{epoch}"]["checkpoint"]
        _same(checkpoint.get("epoch"), epoch, "fixed checkpoint epoch")
        _same(checkpoint.get("optimizer_updates"), epoch * 304, "fixed checkpoint update")
        _same(checkpoint.get("binding_sha256"), binding["binding_sha256"], "checkpoint run binding")
        _same(checkpoint.get("path"), str(output / f"checkpoint-epoch-{epoch:03d}.pt"),
              "owned fixed checkpoint path")
    checkpoint = checkpoints["epoch_30"]["checkpoint"]
    checkpoint_ref = _checkpoint_reference(checkpoint)
    if expected_checkpoint is not None:
        _same(checkpoint_ref, expected_checkpoint, "qualified seed-zero epoch-30 checkpoint bytes")
    reader.read(checkpoint_ref, retain=False)  # Integrity only; never deserialize model tensors.
    initial = reader.json(result["initial_state_reference"])
    final = reader.json(result["final_state_reference"])
    _state_clocks(initial, input_size=size, initial=True)
    _state_clocks(final, input_size=size)
    _same(binding["data"].get("kind"), "train_core_runtime", "bound loader role")
    loader_sha = binding["data"]["loader_binding_sha256"]
    _same(binding["data"]["loader"]["loader_binding_sha256"], loader_sha,
          "nested training loader binding")
    for label, state in (("initial", initial), ("final", final)):
        _same(state["clocks"]["loader"]["binding_sha256"], loader_sha, label + " loader binding")
    _same(reader.json(checkpoints["epoch_30"]["state_reference"]), final,
          "epoch-30 checkpoint and final state")
    _same(final["clocks"]["loader"]["order_sha256"], plans["30"], "final sampler order")
    references = result["receipts"]
    _require(type(references) is list and len(references) == 9120, "full logical receipt sequence is missing")
    for position, reference in enumerate(references):
        epoch, index = divmod(position, 304)
        _same(reference.get("epoch"), epoch + 1, "ordered receipt epoch")
        _same(reference.get("logical_batch_index"), index, "ordered logical batch")
        expected_path = output / "receipts" / f"epoch-{epoch+1:03d}" / f"window-{index+1:04d}.json"
        _same(reference["path"], str(expected_path), "owned logical receipt path")
        _reference({key: reference[key] for key in REF_KEYS})
    evaluation = _evaluation(reader, result, contract)
    native = _native(reader, result, contract, binding, completion=completion)
    return {
        "seed": seed, "input_size": size, "result_reference": result_reference, "result": result,
        "contract": contract, "binding": binding, "initial": initial, "final": final,
        "checkpoint_reference": checkpoint_ref, "evaluation": evaluation, "native": native,
        "completion_reference": None, "completion": completion,
    }


def _resources():
    return {"logical_windows": 0, "synchronized_train_window_wall_seconds": 0.,
            "logical_input_wait_seconds": 0., "maximum_window_wall_seconds": 0.,
            "maximum_window_peak_allocated_bytes": 0, "maximum_window_peak_reserved_bytes": 0}


def _consume_window(record, source, epoch, index, resources):
    from .training_v2b_control import verify_input_receipt
    _same(record["run_binding_sha256"], source["binding"]["binding_sha256"], "window run binding")
    input_receipt = record["input"]
    _same(input_receipt.get("binding_sha256"), source["binding"]["data"]["loader_binding_sha256"],
          "actual input receipt loader binding")
    indices = [sample["index"] for sample in input_receipt["samples"]]
    _require(len(indices) == 16 and len(set(indices)) == 16
             and all(type(value) is int and 0 <= value < 4869 for value in indices),
             "logical batch sample indices differ")
    verify_input_receipt(input_receipt, expected_indices=indices, epoch=epoch, logical_batch_index=index)
    for key, expected in (("epoch", epoch), ("optimizer_updates", (epoch - 1) * 304 + index + 1),
                           ("microsteps", 2 * ((epoch - 1) * 304 + index + 1)),
                           ("logical_batch_size", 16), ("phase", 0)):
        _same(record["window"][key], expected, "logical window " + key)
    _number(record["window"]["loss"], "logical loss")
    engine = record["state"]["clocks"]["engine"]
    _same(engine["epoch_active"], True, "window epoch activity")
    _same(engine["optimizer_updates"], record["window"]["optimizer_updates"], "window state clock")
    _same(engine["microsteps"], record["window"]["microsteps"], "window microstep clock")
    seconds = _number(record["synchronized_train_window_seconds"], "synchronized window time", minimum=0)
    resources["logical_windows"] += 1
    resources["synchronized_train_window_wall_seconds"] += seconds
    resources["logical_input_wait_seconds"] += _number(
        record["logical_input_wait_seconds"], "logical input wait time", minimum=0)
    resources["maximum_window_wall_seconds"] = max(resources["maximum_window_wall_seconds"], seconds)
    for key in ("peak_allocated_bytes", "peak_reserved_bytes"):
        value = _integer(record["memory"][key], "window memory " + key)
        target = "maximum_window_" + key
        resources[target] = max(resources[target], value)
    return indices


def audit_training_pair(reader, left, right, *, progress=None):
    """Hash all 18,240 receipts and compare source identity; do not equate evolved losses/BN/DN."""
    from .training_v2b_resolution import (
        compare_matched_initialization, compare_matched_initial_state, verify_matched_input,
    )
    _require(left["seed"] == right["seed"] and (left["input_size"], right["input_size"]) == SIZES,
             "pair must be the same seed at 640 and 896")
    initialization = compare_matched_initialization(left["result"], right["result"]["initialization"])
    initial_state = compare_matched_initial_state(left["result"], right["initial"], right["result"]["runtime"])
    _same(left["contract"]["epoch_order_sha256"], right["contract"]["epoch_order_sha256"], "paired epoch plans")
    resources = {"640": _resources(), "896": _resources()}
    pair_digest = hashlib.sha256()
    for epoch in range(1, 31):
        epoch_indices = []
        for index in range(304):
            position = (epoch - 1) * 304 + index
            refs = [source["result"]["receipts"][position] for source in (left, right)]
            normalized = [{key: reference[key] for key in REF_KEYS} for reference in refs]
            rows = [reader.json(reference) for reference in normalized]
            for source, record in zip((left, right), rows):
                indices = _consume_window(record, source, epoch, index, resources[str(source["input_size"])])
            verify_matched_input(rows[0]["input"], rows[1]["input"])
            epoch_indices.append(indices)
            pair_digest.update(canonical_json_bytes(
                {"epoch": epoch, "logical_batch_index": index, "references": normalized}) + b"\n")
        _same(canonical_sha256(epoch_indices), left["contract"]["epoch_order_sha256"][str(epoch)],
              "actual complete frozen epoch order")
        _require(len({value for batch in epoch_indices for value in batch}) == 4864,
                 "one epoch did not commit exactly 4864 distinct samples")
        if progress is not None:
            progress("paired_epoch_audited", seed=left["seed"], epoch=epoch)
    for size, resource in resources.items():
        resource["measured_train_window_wall_hours"] = resource["synchronized_train_window_wall_seconds"] / 3600.
        resource["scope"] = "synchronized training-window wall time; excludes checkpoint/evaluation and other wall overhead"
    return {
        "status": "PASS", "seed": left["seed"], "logical_windows_per_arm": 9120,
        "logical_samples_per_arm": 145920, "physical_microsteps_per_arm": 18240,
        "recorded_source_image_sha256_order_and_augmentation_seeds_matched": True,
        "source_image_bytes_reread": False,
        "actual_epoch_orders_equal_frozen_plan": True,
        "initialization": initialization, "initial_state": initial_state,
        "paired_receipt_inventory_sha256": pair_digest.hexdigest(),
        "resources": resources,
        "cross_resolution_loss_denominator_BN_DN_rng_or_parameter_equality_required": False,
    }




def _completed_invocations(reader, completed, freeze_ref, freeze):
    """Authenticate all capacity/smoke/formal completions, then expose formal sources."""
    from . import training_v2b_replication_worker as replication
    _require(type(completed) is list and len(completed) == 16, "campaign must contain all sixteen invocations")
    index = {}
    for row in completed:
        _require(type(row) is dict and set(row) == {
            "cell_id", "stage", "worker_result_reference", "completion_reference"},
            "completed invocation schema differs")
        key = row["cell_id"], row["stage"]
        _require(key not in index, "duplicate completed invocation")
        index[key] = row
    _same([list(key) for key in sorted(index)],
          sorted([cell, stage] for cell in CELLS for stage in replication.STAGES),
          "complete campaign invocation set")
    checked = {}
    for cell in CELLS:
        for stage in replication.STAGES:
            entry = index[cell, stage]
            document = reader.json(entry["worker_result_reference"])
            contract = reader.json(document["contract_reference"])
            replication.validate_replication_worker_contract(contract, verify_files=False)
            replication.validate_replication_freeze(freeze, contract)
            _same(contract["freeze_reference"], freeze_ref, "invocation contract freeze")
            result, source_contract = replication._successful_cell_source(
                entry["worker_result_reference"], cell_id=cell, stage=stage, contract=contract, freeze=freeze)
            completion = replication._verify_completion(
                entry["completion_reference"], entry["worker_result_reference"],
                result, source_contract, contract, freeze)
            # The reusable validators authenticate these leaves as well; record every
            # consumed identity in this summary's complete-file inventory.
            for reference in (entry["completion_reference"], completion["launch_reference"],
                              result["binding_reference"], completion["monitor"]["final_report_reference"]):
                reader.read(reference, retain=False)
            for key in NATIVE_LEAVES:
                reader.read(completion["monitor"][key], retain=False)
            if completion["capacity_memory_audit_reference"] is not None:
                reader.read(completion["capacity_memory_audit_reference"], retain=False)
            checked[cell, stage] = {
                "entry": entry, "result": result, "contract": source_contract, "completion": completion}
    return checked


def _load_context(reader, campaign_result_reference):
    from .training_v2b_campaign import validate_frozen_source
    from .training_v2b_replication_gate import verify_replication_qualification_binding
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "summary requires empty CUDA_VISIBLE_DEVICES")
    terminal = reader.json(campaign_result_reference)
    _require(terminal.get("kind") == "v2b_seed_resolution_replication_campaign_result"
             and terminal.get("status") == "PASS" and terminal.get("formal_started") is True,
             "all four formal replication cells must have completed")
    _same(terminal.get("primary_endpoint_epoch"), 30, "campaign primary endpoint")
    _same(terminal.get("health_only_epochs"), [10, 20], "campaign intermediate checkpoints")
    _same(terminal.get("training_extension_authorized"), False, "campaign extension authority")
    freeze_ref = terminal["freeze_reference"]
    freeze = reader.json(freeze_ref)
    _same(terminal["campaign_id"], freeze["campaign_id"], "campaign identity")
    _same(freeze["formal_order"], list(CELLS), "fixed formal order")
    _require(Path(__file__).resolve() ==
             Path(freeze["repo_root"]) / "src/sparse_rtdetr/baseline/training_v2b_replication_summary.py",
             "summary imported from a different frozen checkout")
    validate_frozen_source(freeze["source"])
    qualification = verify_replication_qualification_binding(freeze["qualification_reference"])
    reader.read(freeze["qualification_reference"], retain=False)
    _require(qualification.get("status") == "PASS"
             and qualification.get("eligible_for_fixed_paired_seed1_and_seed2") is True
             and qualification.get("gpu_admission_granted") is False,
             "replication qualification did not establish the fixed eligible scope")
    _same(qualification["conditional_authorization_reference"], freeze["authorization_reference"],
          "frozen qualification authorization")
    completed = _completed_invocations(reader, terminal.get("completed"), freeze_ref, freeze)
    sources = {}
    for cell in CELLS:
        seed, size = int(cell[1]), int(cell.split("_r")[1])
        checked = completed[cell, "control30"]
        entry, completion = checked["entry"], checked["completion"]
        source = _training_source(
            reader, entry["worker_result_reference"], seed=seed, size=size,
            expected_config=freeze["cells"][cell]["config"], completion=completion)
        source["completion_reference"] = entry["completion_reference"]
        source["prediction_result"] = source["evaluation"]
        source["prediction_role"] = "saved_formal_training_epoch_30_development_predictions"
        sources[seed, size] = source
    analysis = reader.json(qualification["analysis_reference"])
    seed0_campaign = reader.json(qualification["campaign_reference"])
    seed0_terminal = reader.json(qualification["campaign_result_reference"])
    _same(analysis["campaign_reference"], qualification["campaign_reference"], "qualified seed-zero campaign")
    _same(analysis["campaign_result_reference"], qualification["campaign_result_reference"],
          "qualified seed-zero terminal report")
    _require(analysis.get("status") == seed0_terminal.get("status") == "PASS"
             and set(analysis["cells"]) == {"t640_e640", "t640_e896", "t896_e640", "t896_e896"},
             "qualified seed-zero four-cell analysis is incomplete")
    for size in SIZES:
        cell = f"t{size}_e{size}"
        row = analysis["cells"][cell]
        entry = seed0_terminal["completed"][cell]
        _same(row["worker_result"], entry["result_reference"], "qualified diagonal worker")
        result = reader.json(row["worker_result"])
        preparation = reader.json(result["cpu_preparation_reference"])
        checkpoint = seed0_campaign["checkpoints"][str(size)]
        _same(preparation["checkpoint"], checkpoint, "qualified fixed seed-zero checkpoint")
        _same(preparation["weights"], "ema", "qualified diagonal inference weights")
        for key, expected in (("seed", 0), ("input_size", size), ("physical_batch_size", 8),
                               ("accumulation_steps", 2), ("sampling_backend", "deterministic_gather")):
            _same(preparation["source_training_config"][key], expected, "qualified seed-zero source " + key)
        for key, expected in (("epoch", 30), ("optimizer_updates", 9120), ("microsteps", 18240),
                               ("epoch_active", False), ("failed", False), ("phase", 0)):
            _same(preparation["checkpoint_engine"][key], expected, "qualified checkpoint clock " + key)
        _same(result["source_training_size"], size, "qualified source resolution")
        _same(result["evaluation_size"], size, "qualified diagonal evaluation resolution")
        expected_config = copy.deepcopy(freeze["cells"][f"s1_r{size}"]["config"])
        expected_config["seed"] = 0
        # These are the only exact historical paths discovered outside a prebound Ref.
        # Their epoch-30 checkpoint and run binding are independently anchored by C.
        historical_ref = reader.discover(Path(checkpoint["path"]).parent / "worker-result.json")
        source = _training_source(
            reader, historical_ref, seed=0, size=size, expected_config=expected_config,
            expected_checkpoint=checkpoint, expected_binding_sha=preparation["checkpoint_binding_sha256"])
        for name in ("prediction_artifact", "primary_official_gt", "primary_legacy_formal_gt", "coco_secondary"):
            _same(row[name], result[name], "qualified diagonal saved " + name)
        _same(result["full_development_coverage"]["images"], 548, "qualified diagonal image count")
        _same(result["full_development_coverage"]["predictions"], 548 * 300, "qualified diagonal predictions")
        source["prediction_result"] = result
        source["prediction_role"] = "qualified_seed_zero_diagonal_cross_eval_of_fixed_epoch_30_checkpoint"
        source["prediction_worker_reference"] = row["worker_result"]
        source["prediction_completion_reference"] = entry["completion_reference"]
        source["prediction_cpu_preparation_reference"] = result["cpu_preparation_reference"]
        sources[0, size] = source
    return {
        "campaign_result_reference": _reference(campaign_result_reference), "terminal": terminal,
        "freeze_reference": freeze_ref, "freeze": freeze, "qualification": qualification,
        "seed0_campaign": seed0_campaign, "analysis": analysis, "sources": sources,
    }


def validate_replication_summary_inputs(campaign_result_reference):
    """Authenticate endpoint/input metadata; full receipt pairing runs before scoring."""
    context = _load_context(_Reader(), campaign_result_reference)
    return {
        "status": "PASS", "campaign_id": context["freeze"]["campaign_id"],
        "campaign_result_reference": context["campaign_result_reference"],
        "freeze_reference": context["freeze_reference"],
        "qualification_reference": context["freeze"]["qualification_reference"],
        "primary_endpoint_epoch": 30, "seeds": list(SEEDS), "resolutions": list(SIZES),
        "all_16_capacity_smoke_formal_completions_authenticated": True,
        "endpoints": {
            f"s{seed}_r{size}": {"training_result_reference": source["result_reference"],
                               "checkpoint_reference": source["checkpoint_reference"],
                               "prediction_artifact": source["prediction_result"]["prediction_artifact"]}
            for (seed, size), source in context["sources"].items()
        },
        "full_logical_receipt_pairing_still_required_before_summary": True,
        "gpu_admission_granted": False,
    }


def _official_context(reader, context):
    from .training_v2b_official_gt import _make_binding
    from .training_v2b_replication_gate import rebind_development_for_consumer
    campaign, sources = context["seed0_campaign"], context["sources"]
    development = campaign["development_bindings"]["640"]
    original = campaign["official_gt_bindings"]["640"]
    _require(original["raw_annotation_root"] is None and original["raw_bytes_independently_verified"] is False,
             "summary must retain the registered lineage-only evidence boundary")
    reader.read(development["annotation"], embedded=True, retain=False)
    reader.read(development["manifest"], embedded=True, retain=False)
    reader.read(original["lineage"], embedded=True, retain=False)
    consumer_development, development_provenance = rebind_development_for_consumer(development, reader=reader)
    rebuilt, official, image_map, attributes = _make_binding(consumer_development, original["lineage"], None)
    _same(_byte_identity(rebuilt["adapter_source"]), _byte_identity(original["adapter_source"]),
          "official GT adapter code bytes")
    actual_adapter = copy.deepcopy(rebuilt["adapter_source"])
    rebuilt["adapter_source"] = copy.deepcopy(original["adapter_source"])
    rebuilt["binding_sha256"] = canonical_sha256({k: v for k, v in rebuilt.items() if k != "binding_sha256"})
    _same(rebuilt, original, "official GT reconstruction beyond adapter provenance path")
    for source in sources.values():
        binding = source["contract"]["development_binding"]
        for role in ("annotation", "manifest"):
            _same(_byte_identity(binding[role]), _byte_identity(development[role]), "uniform development " + role)
        for key in ("image_count", "annotation_count", "image_order_sha256", "image_manifest_sha256"):
            _same(binding[key], development[key], "uniform development membership " + key)
        _same(binding["source"]["primary_contract"], development["source"]["primary_contract"],
              "uniform frozen primary evaluator contract")
    image_ids = tuple(image_map)
    _require(len(image_ids) == 548 and image_ids == tuple(sorted(image_ids)),
             "official development integer image order differs")
    return {
        "official": official, "image_map": image_map, "image_ids": image_ids,
        "primary_contract": development["source"]["primary_contract"],
        "coco": reader.json(development["annotation"], embedded=True),
        "binding": original, "adapter_current": actual_adapter,
        "development_rebinding": development_provenance,
    }


def _point_check(observed, expected, names):
    deltas = {}
    for name in names:
        actual = _number(observed[name], name + " observed metric", minimum=0, maximum=100)
        reference = _number(expected[name], name + " reference metric", minimum=0, maximum=100)
        _require(abs(actual - reference) <= POINT_TOLERANCE, "cache/full point discrepancy: " + name)
        deltas[name] = actual - reference
    return {"status": "PASS", "absolute_tolerance_percent": POINT_TOLERANCE, "delta_pp": deltas}


def _score_endpoint(reader, source, official, output):
    from .training_v2b_official_gt import with_predictions
    from .training_v2b_primary_cache import prepare_primary_cache
    from . import primary_evaluator
    from .training_v2b_resolution_diagnostics import prepare_coco_cache
    from .training_v2b_replication_gate import _tensor_metrics
    prediction_result = source["prediction_result"]
    predictions = reader.json(prediction_result["prediction_artifact"])
    _require(type(predictions) is list and len(predictions) == 548 * 300,
             "fixed endpoint must retain all 300 predictions for all 548 development images")
    counts = Counter()
    for row in predictions:
        _require(type(row) is dict and set(row) == {"image_id", "category_id", "bbox", "score"},
                 "saved prediction schema differs")
        _integer(row["image_id"], "prediction image id")
        _require(row["image_id"] in official["image_map"], "prediction escaped development membership")
        _integer(row["category_id"], "prediction category", minimum=1)
        _require(row["category_id"] <= 10, "prediction category differs")
        _number(row["score"], "prediction score", minimum=0, maximum=1)
        _require(type(row["bbox"]) is list and len(row["bbox"]) == 4, "prediction box differs")
        for value in row["bbox"]:
            _number(value, "prediction box coordinate")
        counts[row["image_id"]] += 1
    _same([[key, counts[key]] for key in sorted(counts)], [[image_id, 300] for image_id in official["image_ids"]],
          "per-image prediction coverage")
    value = with_predictions(official["official"], official["image_map"], predictions)
    full = primary_evaluator.evaluate_primary_v1(value, official["primary_contract"])
    primary_evaluator.validate_primary_evaluator_result(full, value, official["primary_contract"])
    full_dict = strict_json_loads(json.dumps(asdict(full), ensure_ascii=False, allow_nan=False).encode("utf-8"))
    metrics = {name: full_dict[name] for name in PRIMARY_METRICS}
    cache = prepare_primary_cache(value, official["primary_contract"], official["image_map"])
    cached = cache.score(official["image_ids"])
    _same(full_dict["canonical_input_sha256"], cache.input_sha256, "uniform primary input digest")
    primary_check = _point_check(cached, metrics, ("AP", "AP50", "AP75", "AR500"))
    legacy = copy.deepcopy(prediction_result["primary_legacy_formal_gt"])
    legacy_artifact = reader.json(legacy["result_artifact"])
    _same({name: legacy_artifact[name] for name in PRIMARY_METRICS}, legacy["metrics"],
          "separately retained legacy primary artifact")
    for name in PRIMARY_METRICS:
        _number(legacy["metrics"][name], "legacy " + name, minimum=0, maximum=100)
    coco_metrics = _tensor_metrics(reader, prediction_result)
    coco_cache = prepare_coco_cache(official["coco"], predictions, max_dets=100)
    coco_check = _point_check(coco_cache.score(official["image_ids"]), coco_metrics, COCO_METRICS)
    if source["seed"] == 0:
        _point_check(metrics, prediction_result["primary_official_gt"]["metrics"], PRIMARY_METRICS)
    label = f"s{source['seed']}_r{source['input_size']}"
    full_reference = write_exclusive_json(output / (label + ".primary-official-gt.json"), full_dict)
    return {
        "seed": source["seed"], "input_size": source["input_size"], "epoch": 30,
        "weights": "ema_at_9120_updates", "prediction_role": source["prediction_role"],
        "prediction_artifact": prediction_result["prediction_artifact"],
        "training_result_reference": source["result_reference"],
        "training_contract_reference": source["result"]["contract_reference"],
        "training_binding_reference": source["result"]["binding_reference"],
        "training_initial_state_reference": source["result"]["initial_state_reference"],
        "training_final_state_reference": source["result"]["final_state_reference"],
        "checkpoint_reference": source["checkpoint_reference"],
        "completion_reference": source["completion_reference"],
        "prediction_worker_reference": source.get("prediction_worker_reference"),
        "prediction_completion_reference": source.get("prediction_completion_reference"),
        "uniform_official_primary": {
            "metrics": metrics, "units": "percent", "result_artifact": full_reference,
            "ground_truth_binding_sha256": official["binding"]["binding_sha256"],
            "entry": "sparse_rtdetr.baseline.primary_evaluator.evaluate_primary_v1",
            "cache_point_check": primary_check, "cache_identity": cache.describe(),
            "AP_small_defined_by_this_primary_contract": False,
        },
        "legacy_primary_reported_separately": legacy,
        "converted_COCO_reported_separately": {
            "metrics": {name: coco_metrics[name] for name in COCO_METRICS}, "units": "percent",
            "saved_report": prediction_result["coco_secondary"], "cache_point_check": coco_check,
            "GT_contract": "unchanged_converted_development_formal_GT",
            "AP_small_is_not_uniform_official_primary_AP_small": True,
        },
        "historical_training_epoch30_evaluation_reference": source["evaluation"]["result_artifact"],
        "native_training_evidence": source["native"],
        "controller_elapsed_seconds": (None if source["completion"] is None
                                       else source["completion"]["elapsed_seconds"]),
        "prediction_execution_elapsed_seconds": prediction_result.get("execution_elapsed_seconds"),
        "prediction_inference_elapsed_seconds": prediction_result.get("inference_elapsed_seconds"),
        "historical_controller_wall_time_not_inferred_from_training_window_sum": source["completion"] is None,
    }


def aggregate_fixed_epoch_results(endpoints):
    _same(sorted(endpoints), sorted(f"s{seed}_r{size}" for seed in SEEDS for size in SIZES),
          "all six fixed endpoint cells")
    for seed in SEEDS:
        for size in SIZES:
            row = endpoints[f"s{seed}_r{size}"]
            for key, expected in (("seed", seed), ("input_size", size), ("epoch", 30)):
                _same(row.get(key), expected, "fixed endpoint cell " + key)
    families = {
        "uniform_official_primary": ("uniform_official_primary", PRIMARY_METRICS),
        "converted_COCO": ("converted_COCO_reported_separately", COCO_METRICS),
        "legacy_primary": ("legacy_primary_reported_separately", PRIMARY_METRICS),
    }
    pairs, aggregates = [], {}
    for seed in SEEDS:
        differences = {}
        for family, (key, names) in families.items():
            differences[family] = {
                metric: endpoints[f"s{seed}_r896"][key]["metrics"][metric]
                - endpoints[f"s{seed}_r640"][key]["metrics"][metric] for metric in names}
        pairs.append({"seed": seed, "epoch": 30, "difference_direction": "896_minus_640",
                      "units": "percentage_points", "differences": differences})
    for family, (key, names) in families.items():
        aggregates[family] = {}
        for metric in names:
            aggregates[family][metric] = {
                "r640": descriptive_statistics([endpoints[f"s{seed}_r640"][key]["metrics"][metric] for seed in SEEDS]),
                "r896": descriptive_statistics([endpoints[f"s{seed}_r896"][key]["metrics"][metric] for seed in SEEDS]),
                "paired_difference_896_minus_640_pp": descriptive_statistics(
                    [row["differences"][family][metric] for row in pairs]),
                "additional_seeds_1_and_2_paired_difference_pp": descriptive_statistics(
                    [row["differences"][family][metric] for row in pairs if row["seed"] in (1, 2)]),
            }
    return {
        "paired_seed_results": pairs, "mean_sample_sd_and_range": aggregates,
        "seed0_was_used_for_conditional_replication_qualification": True,
        "three_seed_statistics_are_descriptive_and_conditioned_on_seed0_gate": True,
        "statistical_significance_claimed": False, "new_confidence_interval_or_p_value_computed": False,
    }


def build_replication_summary(*, campaign_result_reference, output_dir):
    """Create one new immutable summary; no best-epoch, seed, threshold or data overrides."""
    from .training_v2b_campaign import validate_frozen_source
    reader = _Reader()
    context = _load_context(reader, campaign_result_reference)
    output = _path(str(output_dir))
    parent = Path(context["freeze"]["output_root"])
    _require(output.parent == parent and output != parent, "summary must be a new direct child of its frozen campaign")
    _require(all(not output.is_relative_to(Path(source["contract"]["output_dir"]))
                 for source in context["sources"].values()), "summary may not enter a historical invocation")
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    started = time.monotonic()
    report = {
        "schema_version": 1, "kind": KIND, "status": "RUNNING",
        "campaign_id": context["freeze"]["campaign_id"],
        "campaign_result_reference": context["campaign_result_reference"],
        "freeze_reference": context["freeze_reference"],
        "qualification_reference": context["freeze"]["qualification_reference"],
        "seed0_analysis_reference": context["qualification"]["analysis_reference"],
        "summary_source": context["freeze"]["source"],
        "primary_endpoint_epoch": 30, "intermediate_health_epochs_excluded_from_comparison": [10, 20],
        "pairing_audits": {}, "endpoints": {},
        "all_16_capacity_smoke_formal_completions_authenticated": True,
        "scope": {"gpu_work_launched": False, "image_bytes_opened": False, "checkpoint_tensors_deserialized": False,
                  "confirmatory_or_test_accessed": False, "best_checkpoint_selected": False,
                  "model_selection_certification": False, "automatic_experiment_authorization": False,
                  "epoch60_authorized": False, "resolution960_authorized": False, "sparse_authorized": False},
    }
    progress_path = output / "progress.jsonl"
    with progress_path.open("xb", buffering=0) as progress_file:
        def progress(stage, **fields):
            progress_file.write(canonical_json_bytes(
                {"stage": stage, "elapsed_seconds": time.monotonic() - started, **fields}) + b"\n")
            os.fsync(progress_file.fileno())
        try:
            for seed in SEEDS:
                audit = audit_training_pair(reader, context["sources"][seed, 640], context["sources"][seed, 896],
                                            progress=progress)
                report["pairing_audits"][str(seed)] = audit
            official = _official_context(reader, context)
            report["official_GT_provenance"] = {
                "binding": official["binding"], "consumer_adapter": official["adapter_current"],
                "development_rebinding": official["development_rebinding"],
                "raw_bytes_independently_verified": False,
                "evidence_boundary": "complete previously saved development raw lineage and manifest; raw annotation bytes not independently reread",
            }
            for seed in SEEDS:
                for size in SIZES:
                    endpoint = _score_endpoint(reader, context["sources"][seed, size], official, output)
                    endpoint["training_resources"] = report["pairing_audits"][str(seed)]["resources"][str(size)]
                    report["endpoints"][f"s{seed}_r{size}"] = endpoint
                    progress("fixed_epoch_rescored", seed=seed, input_size=size,
                             primary_AP=endpoint["uniform_official_primary"]["metrics"]["AP"],
                             coco_AP_small=endpoint["converted_COCO_reported_separately"]["metrics"]["AP_small"])
            report["statistics"] = aggregate_fixed_epoch_results(report["endpoints"])
            validate_frozen_source(context["freeze"]["source"])
            inventory = reader.inventory()
            report["consumed_leaf_inventory_reference"] = write_exclusive_json(
                output / "consumed-leaf-inventory.json",
                {"schema_version": 1, "references": inventory, "inventory_sha256": canonical_sha256(inventory)})
            report.update(status="PASS", elapsed_seconds=time.monotonic() - started)
            progress("finished", endpoint_count=6, paired_seed_count=3)
        except BaseException as exc:
            report.update(status="STOP_NO_RETRY", elapsed_seconds=time.monotonic() - started,
                          failure={"type": type(exc).__name__, "message": str(exc)})
            progress("failed", **report["failure"])
            write_exclusive_json(output / "summary.json", report)
            raise
    report["progress_reference"] = reader.discover(progress_path)
    summary_reference = write_exclusive_json(output / "summary.json", report)
    return {"summary": report, "summary_reference": summary_reference}
