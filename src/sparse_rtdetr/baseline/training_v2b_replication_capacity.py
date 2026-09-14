"""Disposable seed 1/2 capacity admission for frozen 640/896, physical 8 x 2.

This is a diagnostic process, never a training prefix or a checkpoint source.
Original metadata authorities and historical implementations remain immutable.
"""
from __future__ import annotations

import copy
import gzip
import hashlib
import io
import math
import os
from pathlib import Path
import time

import torch

from . import training_v2b_control as control
from .training_v2b import V2BConfig, build_v2b_components, logical_batch_indices
from .training_v2b_data import (
    TrainCoreDataConfig, build_train_core_loader, augmented_tensor_sha256, _sample_seed,
)
from .training_v2b_development import evaluate_development_capacity
from .training_v2b_device import prepare_runtime
from .training_v2b_engine import validate_engine_state_dict
from .training_v2b_evidence import (
    canonical_sha256, file_reference, strict_json_loads, write_exclusive_json,
)
from .training_v2b_geometry import snapshot_model_geometry
from .training_v2b_official_gt import read_regular_reference
from .training_v2b_resolution import _capacity_memory, _selected_capacity_batches
from .training_v2b_runtime import build_train_core_run_binding

PLAN_KIND = "v2b_seed_resolution_replication_capacity_plan"
RESULT_KIND = "v2b_seed_resolution_replication_worker_result"
METADATA_REPORT_SHA256 = "5866b2f00f31fac61428a67db9868238d59d9c6f487321abdaf5f5c24fe4299c"
METADATA_PLAN_SHA256 = {
    1: "cc281e8ba15adb4e698c495348cff4378e739d66b1ee751e86af11ecfb51c9a2",
    2: "f8f5077a80a95ed3756dafbcb038afbf453588b48c20a9f0572225571dc49012",
}
STRESS_POSITIONS = {1: ((13, 68), (18, 52), (27, 36)),
                    2: ((10, 67), (17, 255), (23, 273))}
JOINT_COUNTS = (462, 98, 98, 98, 98, 98, 98, 97) * 2
DN_PARAMETER = "decoder.denoising_class_embed.weight"
SEED_BOUNDS = {
    1: {"max_per_image": 462, "max_per_physical": 1139, "max_per_logical": 1612,
        "max_DN_padded_queries_after_removal": 924},
    2: {"max_per_image": 462, "max_per_physical": 1147, "max_per_logical": 1610,
        "max_DN_padded_queries_after_removal": 924},
}
_PLAN_KEYS = {
    "schema_version", "kind", "seed", "input_size", "physical_batch_size",
    "accumulation_steps", "logical_batch_size", "dataset_size", "epochs",
    "updates_per_epoch", "drop_last_samples_per_epoch", "metadata_report_reference",
    "metadata_plan_reference", "annotation", "manifest", "metadata_sources",
    "counts_in_loader_order_sha256", "epoch_order_sha256", "fixed_windows",
    "bounds", "synthetic_envelope", "resource_gate", "execution_policy", "plan_sha256",
}


class ReplicationCapacityError(ValueError):
    """A changed plan, unexplained clock, lost evidence, or failed capacity gate."""


def _require(value, message):
    if not value:
        raise ReplicationCapacityError(message)


def _same(left, right, message):
    _require(canonical_sha256(left) == canonical_sha256(right), message)


def _integer(value, label, minimum=0):
    _require(type(value) is int and value >= minimum, label + " must be an exact integer")
    return value


def _reference(reference):
    _require(type(reference) is dict and set(reference) == {
        "path", "sha256", "size_bytes", "sha256_scope"}, "complete-file reference schema differs")
    path = Path(reference["path"])
    _require(path.is_absolute() and ".." not in path.parts
             and not any(p.casefold() in {"test", "confirmatory"}
                         or p.casefold().startswith("confirmatory_") for p in path.parts),
             "capacity reference crosses an unapproved path")
    _require(reference["sha256_scope"] == "complete_file_bytes"
             and type(reference["sha256"]) is str and len(reference["sha256"]) == 64
             and all(c in "0123456789abcdef" for c in reference["sha256"]),
             "capacity reference SHA/scope differs")
    _integer(reference["size_bytes"], "reference bytes")
    return reference


def _json(reference):
    return strict_json_loads(read_regular_reference(_reference(reference)))


def _identity(reference):
    return reference["sha256"], reference["size_bytes"]


def _metadata_authority(seed, report_reference, plan_reference):
    _require(report_reference["sha256"] == METADATA_REPORT_SHA256
             and plan_reference["sha256"] == METADATA_PLAN_SHA256[seed],
             "capacity metadata authority differs")
    report, metadata = _json(report_reference), _json(plan_reference)
    _require(report.get("status") == "PASS" and report.get("integer_roundtrip_verified") is True
             and report.get("cuda_initialized") is False and report.get("images_opened") is False
             and metadata.get("kind") == "metadata_only_seed_replication_capacity_plan"
             and metadata.get("seed") == seed
             and metadata.get("all_augmentation_seeds_roundtrip_exact") is True,
             "capacity metadata did not pass its exact integer/CPU audit")
    _require(_identity(report["seed_plans"][str(seed)]) == _identity(plan_reference),
             "metadata report and seed plan do not name the same complete bytes")
    _same(report["sources"], metadata["sources"], "metadata source authorities disagree")
    root = Path(__file__).resolve().parents[3]
    for name, reference in metadata["sources"].items():
        _require(not Path(name).is_absolute() and ".." not in Path(name).parts,
                 "metadata source name escapes the checkout")
        actual = file_reference(root / name)
        _require(_identity(actual) == _identity(reference),
                 "capacity order/augmentation/DN source differs from the metadata audit: " + name)
    return report, metadata


def physical_bounds(counts, slot):
    _require(type(slot) is int and slot in (0, 1), "physical slot must be zero or one")
    _require(type(counts) is list and len(counts) == 8, "a physical count vector must have eight images")
    for count in counts:
        _integer(count, "raw target count")
    maximum, total = max(counts), sum(counts)
    groups = max(1, 100 // maximum) if maximum else None
    queries = 0 if groups is None else 2 * maximum * groups
    # Raw DN queries are not monotone under cropping (m=60 -> 50: 120 -> 200).
    query_upper = 2 * max(100, maximum) if maximum else 0
    active_upper = max([total] + [
        (100 // m) * sum(min(n, m) for n in counts)
        for m in range(1, min(maximum, 99) + 1)
    ]) if maximum else 0
    return {
        "physical_index": slot, "raw_counts": list(counts), "max_gt": maximum, "sum_gt": total,
        "dn_groups_raw": groups, "dn_padded_queries_raw": queries,
        "dn_padded_query_slots_raw": 8 * queries,
        "dn_active_positive_indices_raw": 0 if groups is None else groups * total,
        "dn_active_positive_indices_after_removal_upper": active_upper,
        "dn_padded_queries_after_removal_upper": query_upper,
        "decoder_queries_after_removal_upper": 300 + query_upper,
        "attention_mask_elements_after_removal_upper": (300 + query_upper) ** 2,
    }


def _derive_plan(seed, input_size, report_reference, metadata_reference, report, metadata):
    windows = []
    for row in metadata["fixed_windows"]:
        value = copy.deepcopy(row)
        value["epoch"] = row["epoch_logged"]  # actual order/augmentation algorithm epoch
        value["case_id"] = f"source-epoch-{value['epoch']:03d}-window-{row['logical_batch_index']:04d}"
        windows.append(value)
    summary = report["seed_summaries"][str(seed)]
    plan = {
        "schema_version": 1, "kind": PLAN_KIND, "seed": seed, "input_size": input_size,
        "physical_batch_size": 8, "accumulation_steps": 2, "logical_batch_size": 16,
        "dataset_size": 4869, "epochs": 30, "updates_per_epoch": 304,
        "drop_last_samples_per_epoch": 5,
        "metadata_report_reference": copy.deepcopy(report_reference),
        "metadata_plan_reference": copy.deepcopy(metadata_reference),
        "annotation": copy.deepcopy(metadata["annotation"]), "manifest": copy.deepcopy(metadata["manifest"]),
        "metadata_sources": copy.deepcopy(metadata["sources"]),
        "counts_in_loader_order_sha256": metadata["counts_in_loader_order_sha256"],
        "epoch_order_sha256": {str(row["epoch_logged"]): row["logical_order_sha256"]
                               for row in metadata["epoch_orders"]},
        "fixed_windows": windows,
        "bounds": {key: summary[key] for key in
                   ("max_per_image", "max_per_physical", "max_per_logical", "max_DN_padded_queries_after_removal")},
        "synthetic_envelope": {
            "target_counts": list(JOINT_COUNTS), "cpu_fixture_seed": 896, "repeat_windows": 2,
            "physical": [physical_bounds(list(JOINT_COUNTS[i:i+8]), i // 8) for i in (0, 8)],
            "source_pixels": "last_fixed_real_window_without_another_image_read",
            "claim": "joint_maxGT_sumGT_DN_shape_bound_not_componentwise_dataset_envelope",
        },
        "resource_gate": {
            "minimum_conservative_free_mib": 3072, "minimum_sampled_native_free_mib": 3072,
            "clock_ceiling_mhz": 1500, "scope": "paired_smoke", "deadline_seconds": 600,
            "native_memory_audit_after_worker_exit_required": True,
        },
        "execution_policy": {
            "real_windows": 7, "synthetic_windows": 2, "diagnostic_engine_epoch": 1,
            "formal_loader_cursor_must_remain_initial": True, "formal_training_samples": 0,
            "checkpoints_created": False, "capacity_checkpoint_cannot_seed_training": True,
            "full_development_images": 548, "development_batch_size": 4,
            "development_forward_dtype": "float32", "development_autocast": False,
            "development_is_capacity_only": True, "source_epoch_is_one_based": True,
            "worker_generator_seed": 896, "num_workers": 2, "prefetch_factor": 2,
            "no_empty_cache_or_offload": True, "no_retry_or_batch_resolution_fallback": True,
            "conditional_grad_none_parameter": DN_PARAMETER,
            "conditional_grad_none_requires_both_physical_DN_disabled": True,
        },
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    return plan


def make_replication_capacity_plan(*, seed, input_size, metadata_report_reference,
                                   metadata_plan_reference):
    """Read only authenticated metadata and source bytes; no model/image/CUDA work."""
    _require(type(seed) is int and seed in (1, 2)
             and type(input_size) is int and input_size in (640, 896), "capacity seed/size differs")
    _reference(metadata_report_reference)
    _reference(metadata_plan_reference)
    report, metadata = _metadata_authority(seed, metadata_report_reference, metadata_plan_reference)
    plan = _derive_plan(seed, input_size, metadata_report_reference, metadata_plan_reference, report, metadata)
    return validate_replication_capacity_plan(plan, verify_files=False)


def validate_replication_capacity_plan(plan, *, seed=None, input_size=None, train_core=None,
                                       verify_files=True):
    _require(type(plan) is dict and set(plan) == _PLAN_KEYS, "capacity plan schema differs")
    value = copy.deepcopy(plan)
    actual_seed, actual_size = value["seed"], value["input_size"]
    _require(type(actual_seed) is int and actual_seed in (1, 2)
             and type(actual_size) is int and actual_size in (640, 896), "capacity seed/size differs")
    if seed is not None:
        _require(type(seed) is int and seed == actual_seed, "capacity plan seed differs from worker")
    if input_size is not None:
        _require(type(input_size) is int and input_size == actual_size, "capacity plan size differs from worker")
    _same({key: value[key] for key in ("schema_version", "kind", "physical_batch_size",
                                      "accumulation_steps", "logical_batch_size", "dataset_size",
                                      "epochs", "updates_per_epoch", "drop_last_samples_per_epoch")},
          {"schema_version": 1, "kind": PLAN_KIND, "physical_batch_size": 8, "accumulation_steps": 2,
           "logical_batch_size": 16, "dataset_size": 4869, "epochs": 30, "updates_per_epoch": 304,
           "drop_last_samples_per_epoch": 5}, "capacity fixed dimensions differ")
    _same(value["plan_sha256"], canonical_sha256({k: v for k, v in value.items() if k != "plan_sha256"}),
          "capacity plan digest differs")
    for field in ("metadata_report_reference", "metadata_plan_reference", "annotation", "manifest"):
        _reference(value[field])
    _require(value["metadata_report_reference"]["sha256"] == METADATA_REPORT_SHA256
             and value["metadata_plan_reference"]["sha256"] == METADATA_PLAN_SHA256[actual_seed],
             "capacity plan no longer names the audited seed metadata")
    _same(value["bounds"], SEED_BOUNDS[actual_seed], "capacity seed-specific audited bounds differ")
    expected_positions = [(1, i) for i in range(4)] + list(STRESS_POSITIONS[actual_seed])
    windows = value["fixed_windows"]
    _require(type(windows) is list and [(x["epoch"], x["logical_batch_index"]) for x in windows]
             == expected_positions, "capacity warmup/stress windows changed")
    orders = {}
    for epoch in range(1, 31):
        orders[str(epoch)] = canonical_sha256([list(batch) for batch in logical_batch_indices(
            4869, seed=actual_seed, epoch=epoch, logical_batch_size=16)])
    _same(value["epoch_order_sha256"], orders, "capacity actual one-based epoch order differs")
    for row in windows:
        epoch, index = row["epoch"], row["logical_batch_index"]
        _require(type(epoch) is int and type(index) is int
                 and row["epoch_logged"] == epoch and row["epoch_index"] == epoch - 1
                 and row["seed"] == actual_seed and row["position_start"] == index * 16
                 and row["update_ordinal"] == (epoch - 1) * 304 + index + 1,
                 "capacity source epoch/index semantics differ")
        expected = list(logical_batch_indices(4869, seed=actual_seed, epoch=epoch, logical_batch_size=16))[index]
        _require(len(row["samples"]) == 16, "capacity window lacks sixteen source samples")
        for offset, sample in enumerate(row["samples"]):
            position = index * 16 + offset
            _require(type(sample["dataset_index"]) is int and sample["dataset_index"] == expected[offset]
                     and type(sample["augmentation_seed"]) is int
                     and sample["augmentation_seed"] == _sample_seed(
                         actual_seed, epoch, position, sample["stable_image_id"])
                     and sample["order_position"] == position,
                     "capacity source order or full-precision augmentation seed differs")
            _integer(sample["raw_gt_count"], "raw GT")
            _require(sample["raw_gt_count"] <= 462, "capacity source exceeds audited max GT")
        counts = [x["raw_gt_count"] for x in row["samples"]]
        _same(row["physical"], [physical_bounds(counts[:8], 0), physical_bounds(counts[8:], 1)],
              "capacity removal-safe DN/count bounds differ")
        _require(row["logical_sum_gt"] == sum(counts), "capacity logical GT sum differs")
    if train_core is not None:
        for field in ("annotation", "manifest"):
            _same(value[field], train_core[field], "capacity and worker train_core bytes differ")
    if verify_files:
        report, metadata = _metadata_authority(actual_seed, value["metadata_report_reference"],
                                              value["metadata_plan_reference"])
        expected = _derive_plan(actual_seed, actual_size, value["metadata_report_reference"],
                                value["metadata_plan_reference"], report, metadata)
        _same(value, expected, "capacity plan differs from exact immutable metadata derivation")
    else:
        # Structural callers still cannot weaken a safety gate, envelope or execution policy.
        stub = {"fixed_windows": [], "annotation": value["annotation"], "manifest": value["manifest"],
                "sources": value["metadata_sources"], "counts_in_loader_order_sha256":
                    value["counts_in_loader_order_sha256"], "epoch_orders": []}
        template = _derive_plan(actual_seed, actual_size, value["metadata_report_reference"],
                                value["metadata_plan_reference"],
                                {"seed_summaries": {str(actual_seed): value["bounds"]}}, stub)
        for key in ("synthetic_envelope", "resource_gate", "execution_policy"):
            _same(value[key], template[key], "capacity invariant differs: " + key)
    return value


def capture_adam_state(model, optimizer, updates):
    """Bind every parameter by name; a missing gradient is not a zero gradient."""
    _integer(updates, "optimizer updates")
    _require(type(optimizer) is torch.optim.AdamW, "capacity requires the actual AdamW implementation")
    _require(updates != 0 or not optimizer.state, "fresh capacity optimizer already has parameter state")
    names = {id(p): name for name, p in model.named_parameters()}
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    ordered, groups, by_name = [], [], {}
    for index, group in enumerate(optimizer.param_groups):
        group_names = []
        for parameter in group["params"]:
            _require(id(parameter) in names and id(parameter) not in ordered,
                     "optimizer has an unknown/duplicate parameter")
            ordered.append(id(parameter))
            name = names[id(parameter)]
            group_names.append(name)
            by_name[name] = parameter
        if "param_names" in group:
            _same(group["param_names"], group_names, "optimizer param_names do not match live owners")
        options = {k: v for k, v in group.items() if k not in {"params", "param_names", "lr"}}
        groups.append({"index": index, "names": group_names, "options": control._state_identity(options)})
    _require(set(ordered) == trainable and trainable, "optimizer no longer covers all trainable parameters")
    _require(all(id(p) in trainable for p in optimizer.state), "optimizer state has an unowned parameter")
    layout = {"groups": groups, "parameters": {
        name: {"shape": list(p.shape), "dtype": str(p.dtype), "requires_grad": p.requires_grad}
        for name, p in sorted(by_name.items())}}
    states = {}
    for name, parameter in sorted(by_name.items()):
        values = optimizer.state.get(parameter, {})
        step = None
        if values:
            group = next(g for g in optimizer.param_groups if any(p is parameter for p in g["params"]))
            required = {"step", "exp_avg", "exp_avg_sq"}
            if group["amsgrad"]:
                required.add("max_exp_avg_sq")
            _require(set(values) == required, "incomplete/unknown AdamW state: " + name)
            clock = values["step"]
            _require(isinstance(clock, torch.Tensor) and clock.ndim == 0
                     and clock.dtype in (torch.float32, torch.float64), "invalid AdamW scalar clock: " + name)
            number = float(clock.item())
            _require(math.isfinite(number) and number == int(number) and 0 <= number <= updates,
                     "AdamW clock is noninteger/nonfinite/outside logical updates: " + name)
            step = int(number)
            for key in required - {"step"}:
                moment = values[key]
                _require(isinstance(moment, torch.Tensor) and moment.shape == parameter.shape
                         and moment.dtype == parameter.dtype and moment.device == parameter.device,
                         "AdamW moment layout differs: " + name + "." + key)
        _require(updates != 0 or not values, "fresh capacity optimizer already has state")
        states[name] = {"state_present": bool(values), "step": step, "grad_present": parameter.grad is not None}
    return {"layout": layout, "layout_sha256": canonical_sha256(layout), "parameters": states,
            "parameter_count": len(states), "state_count": sum(x["state_present"] for x in states.values()),
            "learning_rates": [float(group["lr"]) for group in optimizer.param_groups]}


def _validate_adam_snapshot(value):
    _require(type(value) is dict and set(value) == {
        "layout", "layout_sha256", "parameters", "parameter_count", "state_count", "learning_rates"},
        "named Adam snapshot schema differs")
    layout, parameters = value["layout"], value["parameters"]
    _same(value["layout_sha256"], canonical_sha256(layout), "named Adam layout digest differs")
    _require(type(layout) is dict and set(layout) == {"groups", "parameters"}
             and type(parameters) is dict and bool(parameters),
             "named Adam layout or parameter inventory is absent")
    _same(sorted(parameters), sorted(layout["parameters"]), "named Adam parameter inventory is incomplete")
    names = []
    for index, group in enumerate(layout["groups"]):
        _require(type(group) is dict and set(group) == {"index", "names", "options"}
                 and type(group["index"]) is int and group["index"] == index
                 and type(group["names"]) is list, "named Adam group layout differs")
        names.extend(group["names"])
    _require(len(names) == len(set(names)) and set(names) == set(parameters),
             "named Adam groups do not cover the actual parameter inventory")
    _require(type(value["parameter_count"]) is int and value["parameter_count"] == len(parameters)
             and type(value["state_count"]) is int
             and value["state_count"] == sum(row["state_present"] is True for row in parameters.values()),
             "named Adam state/parameter counts differ")
    _require(type(value["learning_rates"]) is list
             and len(value["learning_rates"]) == len(layout["groups"])
             and all(type(rate) in (int, float) and math.isfinite(rate) and rate >= 0
                     for rate in value["learning_rates"]), "named Adam learning rates differ")
    for name, row in parameters.items():
        _require(type(row) is dict and set(row) == {"state_present", "step", "grad_present"}
                 and type(row["state_present"]) is bool and type(row["grad_present"]) is bool,
                 "named Adam participation evidence differs: " + name)
        _require((row["state_present"] and type(row["step"]) is int and row["step"] >= 0)
                 or (not row["state_present"] and row["step"] is None),
                 "named Adam absent state/step semantics differ: " + name)
    return value


def verify_adam_transition(before, after, *, dn_groups):
    """Only absent DN work can explain this frozen architecture's missing gradient."""
    _validate_adam_snapshot(before)
    _validate_adam_snapshot(after)
    _same(before["layout"], after["layout"], "capacity optimizer layout changed")
    _require(type(dn_groups) is list and len(dn_groups) == 2,
             "capacity requires both physical DN observations")
    _require(all(x is None or (type(x) is int and x > 0) for x in dn_groups),
             "DN groups must be positive integers or explicitly disabled")
    _same(sorted(before["parameters"]), sorted(after["parameters"]), "Adam parameter coverage changed")
    _same(sorted(after["parameters"]), sorted(after["layout"]["parameters"]), "Adam parameter evidence is incomplete")
    no_dn = all(x is None for x in dn_groups)
    _require(DN_PARAMETER in after["parameters"]
             and after["parameters"][DN_PARAMETER]["grad_present"] is not no_dn,
             "DN embedding gradient presence disagrees with actual physical DN execution")
    exempt = []
    for name, row in after["parameters"].items():
        previous = before["parameters"][name]
        present = row["grad_present"]
        _require(type(present) is bool, "gradient presence must be an observed boolean")
        if not present:
            _require(name == DN_PARAMETER and no_dn, "unexplained unused trainable parameter: " + name)
            exempt.append(name)
        expected_present = previous["state_present"] or present
        expected_step = (previous["step"] or 0) + int(present) if expected_present else None
        _require(row["state_present"] == expected_present and row["step"] == expected_step,
                 "AdamW step increment differs from actual gradient participation: " + name)
    return {"status": "PASS", "named_parameter_count": len(after["parameters"]),
            "conditional_DN_absence": exempt, "every_step_delta_matches_grad_presence": True}


def _state(components, loader, *, full=False):
    engine = validate_engine_state_dict(components.engine.state_dict())
    updates = engine["optimizer_updates"]
    _require(components.ema.updates == updates and components.warmup.last_step == updates,
             "capacity EMA/warmup/optimizer clocks differ")
    _require(engine["epoch"] in (0, 1) and (engine["epoch"] == 0 or engine["epoch_active"])
             and components.scheduler.last_epoch == 0 and components.scheduler._step_count == 1,
             "disposable capacity engine/scheduler must not invent completed epochs")
    cursor = loader.state_dict()
    _require(cursor["epoch"] == 0 and cursor["optimizer_updates"] == 0
             and cursor["completed_batches"] == 0 and cursor["epoch_active"] is False
             and not loader.has_active_iterator, "capacity changed the formal loader cursor")
    adam = capture_adam_state(components.model, components.optimizer, updates)
    expected_lr = components.config.learning_rate * min(1., (updates + 1) / components.config.warmup_steps)
    _require(all(math.isclose(x, expected_lr, rel_tol=0., abs_tol=1e-15)
                 for x in adam["learning_rates"]), "capacity actual warmup learning rate differs")
    value = {
        "engine": engine, "formal_loader_cursor": cursor, "adam": adam,
        "ema_updates": components.ema.updates,
        "warmup": control._state_identity(components.warmup.state_dict()),
        "scheduler": control._state_identity(components.scheduler.state_dict()),
        "rng": control._rng_identity(components.runtime),
        "raw_bn": control._bn_state(components.model),
        "ema_bn": control._bn_state(components.ema.module),
        "geometry": {name: snapshot_model_geometry(model, expected_input_size=components.config.input_size)
                     for name, model in (("raw", components.model), ("ema", components.ema.module))},
        "auxiliary_numerical_health": control._finite_auxiliary_state(components),
    }
    if full:
        value["full_state_hashes"] = {
            name: canonical_sha256(control._state_identity(owner.state_dict()))
            for name, owner in (("raw", components.model), ("ema", components.ema),
                                ("optimizer", components.optimizer))}
    return value


def _memory_gate(memory):
    fields = {"allocated_bytes", "reserved_bytes", "peak_allocated_bytes", "peak_reserved_bytes",
              "device_free_bytes", "device_total_bytes", "observed_nonallocator_bytes",
              "conservative_free_at_reserved_peak_bytes"}
    _require(type(memory) is dict and set(memory) == fields
             and all(type(memory[name]) is int and memory[name] >= 0 for name in fields),
             "capacity allocator/native memory evidence is incomplete")
    total, free = memory["device_total_bytes"], memory["device_free_bytes"]
    reserved, peak = memory["reserved_bytes"], memory["peak_reserved_bytes"]
    overhead = max(0, total - free - reserved)
    _require(0 < total and free <= total and memory["allocated_bytes"] <= reserved <= peak <= total
             and memory["allocated_bytes"] <= memory["peak_allocated_bytes"] <= peak
             and memory["observed_nonallocator_bytes"] == overhead
             and memory["conservative_free_at_reserved_peak_bytes"] == total - peak - overhead,
             "capacity allocator/native memory arithmetic differs")
    for name in ("conservative_free_at_reserved_peak_bytes", "device_free_bytes"):
        _require(memory[name] >= 3072 * 1024**2,
                 "capacity headroom below frozen 3072 MiB or unavailable: " + name)


def _verify_source_batch(batch, planned, loader):
    receipt = batch.evidence()
    control.verify_input_receipt(receipt, expected_indices=[x["dataset_index"] for x in planned["samples"]],
                                 epoch=planned["epoch"], logical_batch_index=planned["logical_batch_index"])
    _require(receipt["binding_sha256"] == loader.binding_sha256, "capacity receipt changed loader binding")
    for actual, expected, target in zip(receipt["samples"], planned["samples"], batch.targets):
        for key in ("augmentation_seed", "image_id", "stable_image_id", "relative_path",
                    "image_sha256", "order_position"):
            _same(actual[key], expected[key], "capacity actual source sample differs: " + key)
        index = expected["dataset_index"]
        image = loader.dataset.images[index]
        raw_count = len(loader.dataset.annotations[image["id"]])
        _require(raw_count == expected["raw_gt_count"] and len(target["labels"]) <= raw_count,
                 "capacity raw GT or removal-only transformed GT bound differs")
    return receipt


def _update(components, loader, monitor, plan, images, targets, *, case_id, bounds, source_input):
    size = plan["input_size"]
    _require(list(images.shape) == [16, 3, size, size], "capacity logical input geometry differs")
    before = _state(components, loader)
    hardware_before = monitor.check(stage="replication_capacity_before_window")
    torch.cuda.synchronize(0)
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    with control._ForwardObservation(components) as observation:
        window = components.engine.train_window(images, targets)
    torch.cuda.synchronize(0)
    elapsed = time.perf_counter() - started
    hardware_after = monitor.check(stage="replication_capacity_after_window")
    memory = _capacity_memory()
    control.verify_window_ledger(
        window, targets, config=components.config, epoch=1,
        logical_batch_index=before["engine"]["optimizer_updates"], batches_per_epoch=304)
    counts = [len(t["labels"]) for t in targets]
    for slot, upper in enumerate(bounds):
        actual = counts[slot * 8:(slot + 1) * 8]
        group = window["dn_num_groups"][slot]
        queries = window["dn_query_counts"][slot] or 0
        _require(all(a <= b for a, b in zip(actual, upper["raw_counts"]))
                 and queries <= upper["dn_padded_queries_after_removal_upper"]
                 and (0 if group is None else group * sum(actual))
                     <= upper["dn_active_positive_indices_after_removal_upper"],
                 "actual GT/DN load exceeds the audited removal-safe bound")
    after = _state(components, loader)
    clocks = verify_adam_transition(before["adam"], after["adam"], dn_groups=window["dn_num_groups"])
    _same(before["formal_loader_cursor"], after["formal_loader_cursor"], "capacity committed formal data progress")
    _same(before["geometry"]["raw"], after["geometry"]["raw"], "capacity raw anchors/positions changed")
    for name, row in after["raw_bn"].items():
        _require(row["num_batches_tracked"] - before["raw_bn"][name]["num_batches_tracked"] == 2,
                 "capacity named BN counter did not advance by two physical forwards")
    for name, row in after["ema_bn"].items():
        _require(row["num_batches_tracked"] == before["ema_bn"][name]["num_batches_tracked"],
                 "vendor EMA integer BN buffer changed")
    _memory_gate(memory)
    return {
        "case_id": case_id, "diagnostic_only": True, "formal_training_samples": 0,
        "source_input": source_input, "input_shape": list(images.shape),
        "input_tensor_sha256": augmented_tensor_sha256(images, targets),
        "target_counts_per_image": counts, "audited_physical_bounds": bounds,
        "window": window, "forward_observation": observation.evidence(),
        "adam_transition": clocks, "state_before": before, "state_after": after,
        "memory": memory, "synchronized_train_window_seconds": elapsed,
        "hardware_before": hardware_before, "hardware_after": hardware_after,
        "frozen_margin_pass": True,
    }


def run_replication_capacity_worker(contract, *, contract_reference):
    """One independent, supervised process; authorization/freeze checks precede work."""
    from .training_v2b_replication_worker import validate_replication_worker_contract
    from .training_v2b_admission import MonitoredHardwareSession

    checked = validate_replication_worker_contract(contract, verify_files=True)
    _same(_json(contract_reference), checked, "capacity contract bytes differ")
    _require(checked["stage"] == "capacity" and checked["stage_gate_reference"] is None
             and checked["matched_reference"] is None and checked["replay_source"] is None,
             "capacity cannot resume, match a historical seed, or become formal training")
    _require(checked["num_workers"] == 2 and checked["prefetch_factor"] == 2,
             "capacity must use the frozen two-worker/prefetch path")
    config = V2BConfig(**checked["config"])
    plan = validate_replication_capacity_plan(
        _json(checked["capacity_plan"]), seed=config.seed, input_size=config.input_size,
        train_core=checked["train_core"])
    _same(checked["epoch_order_sha256"], plan["epoch_order_sha256"], "worker and capacity epoch orders differ")
    output = Path(checked["output_dir"])
    output.mkdir(parents=False, exist_ok=False)
    monitor = None
    report = {
        "schema_version": 1, "kind": RESULT_KIND, "status": "RUNNING", "stage": "capacity",
        "cell_id": checked["cell_id"], "seed": config.seed, "input_size": config.input_size,
        "campaign_id": checked["campaign_id"], "run_id": checked["run_id"],
        "invocation_id": checked["invocation_id"], "worker_pid": os.getpid(),
        "contract_reference": copy.deepcopy(contract_reference),
        "freeze_reference": copy.deepcopy(checked["freeze_reference"]),
        "qualification_reference": copy.deepcopy(checked["qualification_reference"]),
        "capacity_plan": copy.deepcopy(checked["capacity_plan"]),
        "guardian_clearance_required_after_worker_exit": True, "scientific_certified": False,
        "automatic_retry_or_batch_fallback": False, "diagnostic_only": True,
        "formal_training_samples": 0, "checkpoints_created": False, "cases": [],
    }
    try:
        report["startup_environment"] = control._environment(checked)
        data = checked["train_core"]
        loader = build_train_core_loader(
            TrainCoreDataConfig(seed=config.seed, input_size=config.input_size, logical_batch_size=16,
                                num_workers=checked["num_workers"], prefetch_factor=checked["prefetch_factor"]),
            annotation_file=data["annotation"]["path"], annotation_sha256=data["annotation"]["sha256"],
            manifest_file=data["manifest"]["path"], manifest_sha256=data["manifest"]["sha256"],
            image_root=data["image_root"], repo_root=checked["repo_root"])
        _require(len(loader.dataset) == 4869 and loader.batches_per_epoch == 304,
                 "capacity actual train_core cardinality differs")
        counts = [len(loader.dataset.annotations[image["id"]]) for image in loader.dataset.images]
        _same(canonical_sha256(counts), plan["counts_in_loader_order_sha256"], "capacity actual raw GT inventory differs")
        for epoch in range(1, 31):
            _same(canonical_sha256([list(batch) for batch in loader._plan(epoch)]), plan["epoch_order_sha256"][str(epoch)],
                  "capacity loader actual seed/order differs")
        arguments = {"repo_root": checked["repo_root"], "pretrained_path": checked["pretrained"]["path"],
                     "pretrained_sha256": checked["pretrained"]["sha256"]}
        cpu = build_v2b_components(config, **arguments)
        binding = build_train_core_run_binding(
            cpu, loader, run_id=checked["run_id"], repo_root=checked["repo_root"],
            additional_code_paths={name: ref["path"] for name, ref in checked["code_files"].items()},
            requested_device="cuda:0", cuda_gpu_uuid=checked["expected_gpu_uuid"])
        _require(set(binding["code"]).issubset(checked["code_files"]),
                 "capacity imported code escaped the frozen inventory")
        report["binding_reference"] = write_exclusive_json(output / "run-binding.json", binding)
        report["initialization"] = cpu.initialization
        write_exclusive_json(output / "worker-start.json", report)
        monitor = MonitoredHardwareSession(binding, checked["policy_bundle"], output / "native-hardware",
                                            "paired_smoke", 600)
        monitor.start()
        runtime = prepare_runtime(device="cuda:0", seed=config.seed, binding=binding,
                                  gpu_probe=monitor.admission(), expected_gpu_uuid=checked["expected_gpu_uuid"])
        components = build_v2b_components(config, runtime=runtime, **arguments)
        _same(cpu.initialization, components.initialization, "capacity CPU/CUDA initialization differs")
        del cpu
        report["runtime"] = runtime.identity
        initial = _state(components, loader, full=True)
        report["initial_state_reference"] = write_exclusive_json(output / "initial-state.json", initial)
        components.engine.begin_epoch(1)
        positions = [(row["epoch"], row["logical_batch_index"]) for row in plan["fixed_windows"]]
        last_images = last_receipt = None
        with _selected_capacity_batches(loader, positions) as batches:
            for planned in plan["fixed_windows"]:
                batch = next(batches)
                receipt = _verify_source_batch(batch, planned, loader)
                row = _update(components, loader, monitor, plan, batch.images, batch.targets,
                              case_id=planned["case_id"], bounds=planned["physical"], source_input=receipt)
                report["cases"].append(write_exclusive_json(output / (row["case_id"] + ".json"), row))
                last_images, last_receipt = batch.images, receipt
        fixture = torch.Generator(device="cpu").manual_seed(plan["synthetic_envelope"]["cpu_fixture_seed"])
        targets = []
        for count in plan["synthetic_envelope"]["target_counts"]:
            centers = .1 + .8 * torch.rand((count, 2), generator=fixture)
            sizes = .02 + .08 * torch.rand((count, 2), generator=fixture)
            targets.append({"labels": torch.arange(count, dtype=torch.int64) % 10,
                            "boxes": torch.cat((centers, sizes), dim=1)})
        for repeat in range(2):
            row = _update(components, loader, monitor, plan, last_images, targets,
                          case_id=f"synthetic-envelope-{repeat + 1:02d}",
                          bounds=plan["synthetic_envelope"]["physical"],
                          source_input={"kind": "synthetic_targets_on_last_bound_real_pixels",
                                        "base_pixels_receipt": last_receipt,
                                        "fixture": plan["synthetic_envelope"]})
            report["cases"].append(write_exclusive_json(output / (row["case_id"] + ".json"), row))
        before = _state(components, loader, full=True)
        report["before_development_state_reference"] = write_exclusive_json(
            output / "before-development-state.json", before)
        evaluation_dir = output / "development-capacity"
        evaluation_dir.mkdir()
        def evaluation_gate():
            torch.cuda.synchronize(0)
            return monitor.check(stage="replication_capacity_development_batch")
        torch.cuda.reset_peak_memory_stats(0)
        development = evaluate_development_capacity(
            {name: getattr(components, name) for name in
             ("model", "ema", "postprocessor", "engine", "runtime", "optimizer", "scheduler", "warmup")},
            data_binding=checked["development_binding"], output_dir=evaluation_dir, hardware_gate=evaluation_gate)
        torch.cuda.synchronize(0)
        memory = _capacity_memory()
        _memory_gate(memory)
        after = _state(components, loader, full=True)
        _same(before, after, "capacity full development changed training/Adam/BN/geometry/RNG state")
        _require(development["scope"] == "development_capacity_only"
                 and development["scientific_comparison_eligible"] is False
                 and development["data"]["evaluated_image_count"] == 548,
                 "capacity development scope/coverage differs")
        _require(after["engine"]["optimizer_updates"] == 9 and len(report["cases"]) == 9,
                 "capacity did not execute exactly seven source and two synthetic windows")
        _same(initial["formal_loader_cursor"], after["formal_loader_cursor"], "capacity advanced formal loader")
        report["development_capacity"] = development
        report["development_memory"] = memory
        report["development_memory_reference"] = write_exclusive_json(output / "development-memory.json", memory)
        report["final_state_reference"] = write_exclusive_json(output / "capacity-final-state.json", after)
        report["diagnostic_optimizer_updates"] = 9
        report["loader_cursor_unchanged"] = after["formal_loader_cursor"]
        report["capacity_acceptance"] = {
            "status": "PASS", "minimum_conservative_free_mib": 3072, "native_minimum_free_mib": 3072,
            "full_development_images": 548, "no_empty_cache_or_model_offload": True,
            "same_arm_replay_requires_separate_smoke_comparison": True,
            "sampled_native_margin_requires_controller_final_audit": True,
            "named_grad_presence_and_Adam_steps_audited": True,
            "formal_training_samples": 0, "checkpoints_created": False,
        }
        validate_replication_worker_contract(checked, verify_files=True)
        report["monitor_reference"] = monitor.finish()
        report["status"] = "PASS"
        write_exclusive_json(output / "worker-result.json", report)
        return report
    except BaseException as exc:
        report["status"] = "STOP_NO_RETRY"
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        try:
            write_exclusive_json(output / "worker-result.json", report)
        finally:
            if monitor is not None:
                monitor.abort("replication capacity failure: " + type(exc).__name__ + ": " + str(exc))
        raise


def audit_replication_capacity_native_memory(completion):
    """Read complete native evidence after worker exit; never grant a live capability."""
    from .training_v2b_admission import wait_for_monitored_finish
    monitor = completion["monitor"]
    result = _json(completion["result_reference"])
    contract = _json(result["contract_reference"])
    launch = _json(completion["launch_reference"])
    plan = validate_replication_capacity_plan(_json(result["capacity_plan"]),
                                              seed=result["seed"], input_size=result["input_size"],
                                              train_core=contract["train_core"])
    _require(result.get("kind") == RESULT_KIND and result.get("status") == "PASS"
             and result.get("stage") == "capacity" and contract.get("stage") == "capacity"
             and contract.get("kind") == "v2b_seed_resolution_replication_worker"
             and result.get("capacity_acceptance", {}).get("status") == "PASS"
             and result.get("formal_training_samples") == 0 and result.get("checkpoints_created") is False,
             "capacity completion belongs to another scope or did not pass")
    _same(wait_for_monitored_finish(result["monitor_reference"], timeout_seconds=1.), monitor,
          "capacity native final report or leaf evidence changed")
    _require(monitor.get("status") == "PASS" and monitor.get("worker_exited") is True
             and monitor.get("sampled_clock_compliance") is True, "capacity native guardian did not clear")
    for key in ("cell_id", "campaign_id", "run_id", "invocation_id", "freeze_reference",
                "qualification_reference", "capacity_plan"):
        _same(result[key], contract[key], "capacity result/contract identity differs: " + key)
    _same({k: contract["config"][k] for k in ("seed", "input_size", "physical_batch_size", "accumulation_steps")},
          {k: plan[k] for k in ("seed", "input_size", "physical_batch_size", "accumulation_steps")},
          "capacity worker and plan design differ")
    _same(result["contract_reference"], launch["contract"], "capacity launch contract differs")
    _same(monitor["owner"], launch["owner"], "capacity native/launch owner differs")
    _require(launch["pid"] == result["worker_pid"] == monitor["owner"]["pid"], "capacity PID binding differs")
    binding = _json(result["binding_reference"])
    _require(monitor["run_id"] == result["run_id"] == binding["run_id"]
             and monitor["run_binding_sha256"] == binding["binding_sha256"]
             and monitor["gpu_uuid"] == contract["expected_gpu_uuid"], "capacity native run/GPU binding differs")
    _require(binding["config"]["seed"] == plan["seed"]
             and binding["config"]["input_size"] == plan["input_size"]
             and binding["config"]["physical_batch_size"] == 8
             and binding["config"]["accumulation_steps"] == 2, "capacity actual bound seed/geometry differs")
    output = Path(contract["output_dir"])
    _require(result["monitor_reference"]["monitor_final_report"] == str(output / "native-hardware/monitor-final.json"),
             "capacity monitor path escaped its owned output")
    expected_cases = [x["case_id"] for x in plan["fixed_windows"]] + [
        "synthetic-envelope-01", "synthetic-envelope-02"]
    _require(len(result["cases"]) == 9 and result["diagnostic_optimizer_updates"] == 9,
             "capacity case coverage is incomplete")
    minimum_conservative_free = math.inf
    previous_after = None
    for index, (reference, expected_case) in enumerate(zip(result["cases"], expected_cases)):
        _require(Path(reference["path"]) == output / (expected_case + ".json"), "capacity case path differs")
        case = _json(reference)
        _require(case["case_id"] == expected_case and case["adam_transition"]["status"] == "PASS",
                 "capacity case identity or named Adam audit failed")
        _same(case["adam_transition"], verify_adam_transition(
            case["state_before"]["adam"], case["state_after"]["adam"],
            dn_groups=case["window"]["dn_num_groups"]), "capacity saved Adam transition is inconsistent")
        before_engine = validate_engine_state_dict(case["state_before"]["engine"])
        after_engine = validate_engine_state_dict(case["state_after"]["engine"])
        _require(before_engine["optimizer_updates"] == index and after_engine["optimizer_updates"] == index + 1
                 and before_engine["epoch"] == after_engine["epoch"] == 1
                 and before_engine["epoch_active"] is True and after_engine["epoch_active"] is True,
                 "capacity case update sequence differs")
        if previous_after is not None:
            _same(previous_after, case["state_before"], "capacity state changed between source windows")
        previous_after = case["state_after"]
        _memory_gate(case["memory"])
        minimum_conservative_free = min(minimum_conservative_free,
                                        case["memory"]["conservative_free_at_reserved_peak_bytes"])
    development_memory = _json(result["development_memory_reference"])
    _same(development_memory, result["development_memory"], "capacity embedded development memory differs")
    _memory_gate(development_memory)
    minimum_conservative_free = min(minimum_conservative_free,
                                    development_memory["conservative_free_at_reserved_peak_bytes"])
    before_development = _json(result["before_development_state_reference"])
    _same(previous_after, {k: v for k, v in before_development.items() if k != "full_state_hashes"},
          "capacity state changed before development")
    _same(_json(result["before_development_state_reference"]), _json(result["final_state_reference"]),
          "capacity evaluation did not preserve complete state")
    stream = _audit_native_memory_samples(monitor)
    return {
        "schema_version": 1,
        "kind": "v2b_seed_resolution_replication_capacity_memory_audit",
        "status": "PASS", "cell_id": result["cell_id"], "seed": result["seed"],
        "input_size": result["input_size"], "capacity_plan": result["capacity_plan"],
        "result_reference": copy.deepcopy(completion["result_reference"]),
        "monitor_final_report_reference": monitor["final_report_reference"],
        "native_sample_reference": monitor["samples"],
        "required_margin_mib": 3072,
        "minimum_conservative_free_mib": minimum_conservative_free / 1024**2,
        **stream,
        "whole_native_file_and_chain_authenticated": True,
        "all_nine_cases_and_evaluation_memory_authenticated": True,
        "clock_and_hardware_clearance_from_final_independent_guardian": True,
        "separate_smoke_and_checkpoint_resume_still_required": True,
    }


def _audit_native_memory_samples(monitor):
    """Authenticate every native memory observation, including the faster clock stream."""
    raw = read_regular_reference(_reference(monitor["samples"]))
    previous, count = None, 0
    sample_counts = {"health": 0, "clock": 0}
    last_stamp = {"health": None, "clock": None}
    maximum_gaps = {"health": 0., "clock": 0.}
    minimum_free, maximum_used, maximum_clock = math.inf, 0., 0.
    with gzip.GzipFile(fileobj=io.BytesIO(raw), mode="rb") as stream:
        for line in stream:
            entry = strict_json_loads(line)
            _require(type(entry) is dict and set(entry) == {"sequence", "previous_sha256", "record", "sha256"}
                     and type(entry["sequence"]) is int and entry["sequence"] == count
                     and entry["previous_sha256"] == previous, "capacity native chain is incomplete")
            _same(canonical_sha256({k: v for k, v in entry.items() if k != "sha256"}), entry["sha256"],
                  "capacity native chain hash differs")
            previous, count = entry["sha256"], count + 1
            row = entry["record"]
            _require(type(row) is dict and row.get("kind") in {"health", "clock"},
                     "capacity native stream contains an unrecognized observation")
            kind = row["kind"]
            stamp = _integer(row["started_ns"], "native sample timestamp")
            last = last_stamp[kind]
            limit = 2_000_000_000 if kind == "health" else 1_000_000_000
            if last is not None:
                _require(stamp > last and stamp - last <= limit,
                         "capacity native " + kind + " sampling gap")
                maximum_gaps[kind] = max(maximum_gaps[kind], (stamp - last) / 1e9)
            last_stamp[kind] = stamp
            if kind == "health":
                gpu = row["gpu"]
                _require(gpu["uuid"] == monitor["gpu_uuid"], "capacity native GPU identity changed")
            else:
                gpu = row
                for field in ("graphics_clock_mhz", "sm_clock_mhz"):
                    value = row[field]
                    _require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1500,
                             "capacity sampled clock exceeds 1500 MHz or is unavailable")
                    maximum_clock = max(maximum_clock, value)
                _same(row["clock_mhz"], row["graphics_clock_mhz"], "capacity sampled graphics clock differs")
            free, used = gpu["memory_free_mib"], gpu["memory_used_mib"]
            _require(type(free) in (int, float) and math.isfinite(free) and free >= 3072
                     and type(used) in (int, float) and math.isfinite(used) and used >= 0,
                     "capacity sampled native headroom below 3072 MiB or unavailable")
            minimum_free, maximum_used = min(minimum_free, free), max(maximum_used, used)
            sample_counts[kind] += 1
    _same({"records": count, "last_sha256": previous}, monitor["sample_hash_chain"],
          "capacity full native stream differs from final guardian hash")
    _require(all(n >= 3 for n in sample_counts.values()), "capacity native memory observations are insufficient")
    return {
        "records": count, "health_samples": sample_counts["health"], "clock_samples": sample_counts["clock"],
        "minimum_native_free_mib": minimum_free, "maximum_native_used_mib": maximum_used,
        "maximum_sampled_clock_mhz": maximum_clock,
        "maximum_health_gap_seconds": maximum_gaps["health"],
        "maximum_clock_gap_seconds": maximum_gaps["clock"],
    }
