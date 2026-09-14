"""The explicitly authorized 896 x 8 x 2 experiment against sealed 640 B.

Resolution changes geometry and post-resize box survival. Matched source-image
identities and augmentation seeds are exact; resized pixels, target counts and
DN padding are measured, not forced equal. Capacity updates are diagnostics and
are never a checkpoint or a prefix of the formal thirty-epoch experiment.
"""
from __future__ import annotations

import contextlib
import copy
from dataclasses import asdict
import math
import os
from pathlib import Path
import re
import time

import torch
from torch.utils.data import DataLoader

from .training_v2b import V2BConfig, build_v2b_components, logical_batch_indices, seed_worker
from .training_v2b_data import (
    TrainCoreDataConfig, build_train_core_loader, _LogicalCollator, augmented_tensor_sha256,
)
from .training_v2b_device import prepare_runtime
from .training_v2b_evidence import canonical_sha256, write_exclusive_json
from .training_v2b_geometry import snapshot_model_geometry
from .training_v2b_campaign import _json_reference
from .training_v2b_runtime import build_train_core_run_binding

RESOLUTION_AUTHORIZATION_SHA256 = "ece816c4bfc93b62c403a00986a738f853b86e91a05953531ab93c0998538819"
MATCHED_COMPLETION_SHA256 = "8bf192eb292ff6cc2687a09879ff670632b360dca45c4e2ab6003807db53aaf9"
_GEOMETRY_BUFFERS = {"decoder.anchors", "decoder.valid_mask"}
_SAMPLE_KEYS = {
    "augmentation_seed", "binding_sha256", "epoch", "image_id", "image_sha256",
    "index", "order_position", "relative_path", "source_path", "stable_image_id",
}
_FIXED_STRESS = ((14, 82), (25, 21), (27, 128))


def _control():
    from . import training_v2b_control
    return training_v2b_control


def _require(value, message):
    if not value:
        _control()._fail(message)


def load_matched_control(completion_reference: dict) -> dict:
    """Authenticate the particular completed B, including its final guardian."""
    _require(completion_reference.get("sha256") == MATCHED_COMPLETION_SHA256,
             "896 must bind the sealed completed 640 x 8 x 2 B control")
    completion = _json_reference(completion_reference)
    _require(completion.get("monitor", {}).get("status") == "PASS",
             "matched B lacks final independent hardware clearance")
    result_reference = completion["result_reference"]
    result = _json_reference(result_reference)
    contract = _json_reference(result["contract_reference"])
    _require(result.get("status") == "PASS" and result.get("stage") == "control30"
             and result.get("arm") == "B" and result.get("receipt_count") == 9120,
             "matched B is not a complete fixed thirty-epoch control")
    _require(contract.get("kind") == "v2b_640_control_worker"
             and contract.get("stage") == "control30" and contract.get("arm") == "B",
             "matched B worker contract differs")
    _control()._same(contract["config"], asdict(V2BConfig(
        physical_batch_size=8, accumulation_steps=2, sampling_backend="deterministic_gather",
    )), "historical B scientific configuration")
    _require(completion["monitor"]["run_id"] == result["run_id"]
             and completion["monitor"]["owner"]["pid"] == result["worker_pid"],
             "matched B guardian belongs to a different worker")
    return {"completion": completion, "result": result, "contract": contract,
            "result_reference": result_reference}


def validate_matched_definition(contract: dict, matched: dict) -> None:
    """Permit only the explicit resolution intervention in scientific settings."""
    c = _control()
    old = matched["contract"]
    expected = {**old["config"], "input_size": 896}
    c._same(contract["config"], expected, "640 B to 896 scientific configuration")
    for name in ("num_workers", "prefetch_factor", "torch_version", "train_core",
                 "pretrained", "epoch_order_sha256", "expected_gpu_uuid"):
        c._same(contract[name], old[name], "matched resolution " + name)
    left, right = old["development_binding"], contract["development_binding"]
    for name in ("annotation", "manifest", "image_root", "image_count", "annotation_count",
                 "image_order_sha256", "image_manifest_sha256"):
        c._same(left[name], right[name], "matched development " + name)
    policy = copy.deepcopy(left["policy"])
    policy["input_size"] = [896, 896]
    resize = [op for op in policy["transform"] if op.get("type") == "Resize"]
    _require(len(resize) == 1, "historical development resize is not unique")
    resize[0]["size"] = [896, 896]
    c._same(policy, right["policy"], "development policy changes only resolution")
    # Core math, optimizer/checkpoint/runtime, sampling and evaluator are frozen.
    protected = {
        "src/sparse_rtdetr/baseline/training_v2b_engine.py",
        "src/sparse_rtdetr/baseline/training_v2b_checkpoint.py",
        "src/sparse_rtdetr/baseline/training_v2b_runtime.py",
        "src/sparse_rtdetr/baseline/training_v2b_device.py",
        "src/sparse_rtdetr/baseline/training_v2b_deterministic_sampling.py",
        "src/sparse_rtdetr/baseline/primary_evaluator.py",
        "src/sparse_rtdetr/baseline/postprocessor.py",
        "src/sparse_rtdetr/data_protocol/evaluation.py",
        "src/sparse_rtdetr/data_protocol/protocol.py",
    }
    protected.update(name for name in old["code_files"] if name.startswith("vendor/"))
    for name in protected:
        _require(name in contract["code_files"] and
                 contract["code_files"][name]["sha256"] == old["code_files"][name]["sha256"],
                 "matched scientific core source changed: " + name)


def compare_matched_initialization(old_result: dict, initialization: dict) -> dict:
    c = _control()
    expected = old_result["initialization"]
    for name in ("seed", "seed_initialized_before_model", "parameters", "pretrained", "sampling"):
        c._same(expected[name], initialization[name], "matched model initialization " + name)
    def invariant_inventory(value):
        return [row for row in value["inventory"] if row["name"] not in _GEOMETRY_BUFFERS]
    c._same(invariant_inventory(expected["model_state"]),
            invariant_inventory(initialization["model_state"]),
            "initialized model state excluding only resolution-sized decoder geometry")
    changed = sorted(row["name"] for row in initialization["model_state"]["inventory"]
                     if row["name"] in _GEOMETRY_BUFFERS)
    _require(set(changed) == _GEOMETRY_BUFFERS, "resolution geometry inventory is incomplete")
    return {"status": "PASS", "trainable_parameters_exact": True,
            "seed_and_pretrained_exact": True,
            "all_other_initial_state_dict_tensors_exact": True,
            "resolution_sized_state_dict_buffers": changed,
            "plain_position_cache_verified_separately": True}


def compare_matched_initial_state(old_result: dict, state: dict, runtime: dict) -> dict:
    c = _control()
    expected = c._read_json_reference(old_result["initial_state_reference"], "matched initial state")
    c._same(old_result["runtime"], runtime, "matched actual CUDA runtime")
    for name in ("rng", "raw_bn", "ema_bn", "raw_parameters", "optimizer_state_sha256",
                 "model_training", "ema_training"):
        c._same(expected[name], state[name], "matched CUDA initialized " + name)
    _require("geometry" in state, "896 initialized geometry snapshot is absent")
    return {"status": "PASS", "initial_parameters_BN_RNG_optimizer_modes_exact": True,
            "actual_runtime_exact": True, "full_geometry_hash_equality_required": False}


def verify_matched_input(expected: dict, actual: dict) -> None:
    """Fail before forward if source order/bytes or augmentation RNG differ."""
    c = _control()
    indices = [sample["index"] for sample in actual.get("samples", [])]
    _require(len(indices) == 16, "matched receipt must contain one complete logical batch of sixteen")
    for receipt in (expected, actual):
        c.verify_input_receipt(receipt, expected_indices=indices,
                               epoch=actual.get("epoch"),
                               logical_batch_index=actual.get("logical_batch_index"))
        for sample in receipt["samples"]:
            _require(set(sample) == _SAMPLE_KEYS, "matched sample receipt schema differs")
            _require(type(sample["augmentation_seed"]) is int and
                     0 <= sample["augmentation_seed"] < 2**63,
                     "augmentation seed must retain its full integer identity")
    def source_samples(receipt):
        return [{k: v for k, v in sample.items() if k != "binding_sha256"}
                for sample in receipt["samples"]]
    c._same(source_samples(expected), source_samples(actual),
            "matched source samples, bytes, order and augmentation seeds")


def _envelope_counts(maximum: int, total: int) -> list[int]:
    _require(type(maximum) is int and type(total) is int
             and 0 < maximum <= total <= 8 * maximum,
             "invalid raw physical-batch capacity envelope")
    residual = total - maximum
    counts = [maximum] + [residual // 7 + int(i < residual % 7) for i in range(7)]
    _require(sum(counts) == total and max(counts) == maximum, "invalid envelope distribution")
    return counts * 2


def make_capacity_plan(loader) -> dict:
    """Compute an annotation-only upper bound for the fixed thirty-epoch order."""
    _require(loader.config.input_size == 896 and loader.config.logical_batch_size == 16
             and loader.config.seed == 0 and len(loader.dataset) == 4869,
             "capacity plan requires the exact authorized 896 train_core loader")
    dataset = loader.dataset
    counts = [len(dataset.annotations[row["id"]]) for row in dataset.images]
    _require(sum(counts) == 261886, "raw train_core annotation cardinality differs")
    maximum, micro_max, logical_max = max(counts), -1, -1
    micro_position = logical_position = None
    for epoch in range(1, 31):
        for index, batch in enumerate(logical_batch_indices(
                len(dataset), seed=0, epoch=epoch, logical_batch_size=16)):
            row = [counts[i] for i in batch]
            if sum(row) > logical_max:
                logical_max, logical_position = sum(row), (epoch, index)
            for offset in (0, 8):
                if sum(row[offset:offset + 8]) > micro_max:
                    micro_max = sum(row[offset:offset + 8])
                    micro_position = (epoch, index)
    positions = sorted(set(_FIXED_STRESS) | {micro_position, logical_position})
    value = {
        "schema_version": 1, "kind": "896_fixed_order_capacity_plan",
        "input_size": 896, "physical_batch_size": 8, "accumulation_steps": 2,
        "logical_batch_size": 16, "seed": 0,
        "warmup_positions": [[1, i] for i in range(4)],
        "stress_positions": [list(row) for row in positions],
        "raw_bounds": {
            "image_count": 4869, "annotation_count": sum(counts),
            "counts_in_loader_order_sha256": canonical_sha256(counts),
            "max_per_image": maximum, "max_per_physical_microbatch": micro_max,
            "max_per_logical_batch": logical_max,
            "raw_microbatch_max_position": list(micro_position),
            "raw_logical_max_position": list(logical_position),
            "proof": "no_mosaic_or_copy; converter_crop_sanitize_only_remove_boxes",
        },
        "synthetic_envelope": {
            "target_counts": _envelope_counts(maximum, micro_max), "repeat_windows": 2,
            "cpu_fixture_seed": 896, "scientific_samples": False,
            "purpose": "bound_DN_padding_and_GT_cost_in_both_physical_microbatch_positions",
        },
        "development": {"role": "development", "image_count": 548, "batch_size": 4,
                        "autocast": False, "weights": "ema", "scientific_eligible": False},
        "resource_gate": {"minimum_conservative_free_mib": 3072,
                          "native_minimum_free_mib": 3072,
                          "no_empty_cache_or_model_offload": True},
    }
    validate_capacity_plan(value)
    return value


def validate_capacity_plan(plan: dict) -> None:
    fields = {"schema_version", "kind", "input_size", "physical_batch_size",
              "accumulation_steps", "logical_batch_size", "seed", "warmup_positions",
              "stress_positions", "raw_bounds", "synthetic_envelope", "development",
              "resource_gate"}
    _require(type(plan) is dict and set(plan) == fields, "capacity plan schema differs")
    for name, value in (("schema_version", 1), ("input_size", 896),
                        ("physical_batch_size", 8), ("accumulation_steps", 2),
                        ("logical_batch_size", 16), ("seed", 0)):
        _require(type(plan[name]) is int and plan[name] == value, "capacity plan differs: " + name)
    _require(plan["kind"] == "896_fixed_order_capacity_plan"
             and plan["warmup_positions"] == [[1, i] for i in range(4)],
             "capacity kind or warmup positions differ")
    positions = plan["stress_positions"]
    _require(type(positions) is list and 3 <= len(positions) <= 5,
             "capacity stress coverage differs")
    for row in positions:
        _require(type(row) is list and len(row) == 2 and all(type(i) is int for i in row)
                 and 1 <= row[0] <= 30 and 0 <= row[1] < 304,
                 "invalid capacity position")
    _require(sorted(set(map(tuple, positions))) == list(map(tuple, positions))
             and set(_FIXED_STRESS).issubset(map(tuple, positions)),
             "capacity stress set omitted a reviewed B extreme")
    bounds = plan["raw_bounds"]
    _require(type(bounds) is dict and set(bounds) == {
        "image_count", "annotation_count", "counts_in_loader_order_sha256",
        "max_per_image", "max_per_physical_microbatch", "max_per_logical_batch",
        "raw_microbatch_max_position", "raw_logical_max_position", "proof",
    }, "capacity raw bound schema differs")
    _require(bounds["image_count"] == 4869 and bounds["annotation_count"] == 261886
             and re.fullmatch(r"[0-9a-f]{64}", bounds["counts_in_loader_order_sha256"]) is not None
             and bounds["proof"] == "no_mosaic_or_copy; converter_crop_sanitize_only_remove_boxes",
             "raw capacity authority differs")
    for name in ("max_per_image", "max_per_physical_microbatch", "max_per_logical_batch"):
        _require(type(bounds[name]) is int and bounds[name] > 0, "capacity count is invalid")
    _require(bounds["max_per_logical_batch"] <= 2 * bounds["max_per_physical_microbatch"],
             "logical target upper bound exceeds the two microbatch bounds")
    for name in ("raw_microbatch_max_position", "raw_logical_max_position"):
        _require(bounds[name] in positions, "raw target extreme omitted from pilot")
    _require(plan["synthetic_envelope"] == {
        "target_counts": _envelope_counts(bounds["max_per_image"], bounds["max_per_physical_microbatch"]),
        "repeat_windows": 2, "cpu_fixture_seed": 896, "scientific_samples": False,
        "purpose": "bound_DN_padding_and_GT_cost_in_both_physical_microbatch_positions",
    }, "synthetic envelope differs")
    _require(plan["development"] == {
        "role": "development", "image_count": 548, "batch_size": 4,
        "autocast": False, "weights": "ema", "scientific_eligible": False,
    }, "capacity evaluation must cover full FP32 batch-four development")
    _require(plan["resource_gate"] == {
        "minimum_conservative_free_mib": 3072, "native_minimum_free_mib": 3072,
        "no_empty_cache_or_model_offload": True,
    }, "capacity margin may not be relaxed")


def compare_resolution_smoke(smoke_dir, replay_dir) -> dict:
    """Reuse the original unchanged replay tolerances, including EMA and AdamW."""
    c = _control()
    (base, base_ref), (replay, replay_ref) = (
        c._result_from_dir(path) for path in (smoke_dir, replay_dir)
    )
    _require(base["stage"] == "smoke" and replay["stage"] == "smoke_replay"
             and base["run_id"] == replay["run_id"] and base["worker_pid"] != replay["worker_pid"]
             and base["arm"] == replay["arm"] == "B", "896 replay process/arm/stage differs")
    for name in ("common_identity", "runtime", "matched_control_reference"):
        c._same(base[name], replay[name], "896 replay " + name)
    c._verify_common_initial_state(
        c._read_json_reference(base["initial_state_reference"], "896 initial state"),
        c._read_json_reference(replay["initial_state_reference"], "896 replay initial state"),
    )
    c._same(c._read_json_reference(base["checkpoints"]["window_2"]["state_reference"], "896 midpoint"),
            c._read_json_reference(replay["restore_boundary_reference"], "896 restored boundary"),
            "896 exact complete checkpoint boundary, including geometry")
    left, right = c._receipt_index(base), c._receipt_index(replay)
    _require(set(left) == {(1, i) for i in range(4)}
             and set(right) == {(1, 2), (1, 3)}, "896 replay logical coverage differs")
    windows = [c.compare_replay_records(c._read_json_reference(left[key], "896 original"),
                                       c._read_json_reference(right[key], "896 replay"))
               for key in sorted(right)]
    a = c._read_json_reference(base["final_state_reference"], "896 final state")
    b = c._read_json_reference(replay["final_state_reference"], "896 replay final state")
    for name in ("clocks", "rng", "raw_bn", "ema_bn", "geometry"):
        c._same(a[name], b[name], "896 final replay " + name)
    binding = c._read_json_reference(base["binding_reference"], "896 smoke binding")
    c._same(binding, c._read_json_reference(replay["binding_reference"], "896 replay binding"),
            "896 same-arm binding")
    continuation = c.compare_continuation_checkpoints(
        base["checkpoints"]["window_4"]["checkpoint"],
        replay["checkpoints"]["window_4"]["checkpoint"], binding=binding,
    )
    preview = c._compare_previews(c._preview_evidence(base), c._preview_evidence(replay))
    return {"schema_version": 1, "status": "PASS", "scope": "896_same_arm_engineering_smoke",
            "workers": [base_ref, replay_ref], "windows": windows,
            "continuation_checkpoint_numerics": continuation, "development_preview": preview,
            "replay_tolerances": copy.deepcopy(c.REPLAY_TOLERANCES),
            "full_restore_geometry_BN_RNG_inputs_clocks_exact": True,
            "scientific_comparison_eligible": False,
            "hardware_clearance_requires_controller_final_guardian": True}


@contextlib.contextmanager
def _selected_capacity_batches(loader, positions):
    """Same dataset/transform/collate/spawn path, with no invented training cursor."""
    batches = []
    for epoch, index in positions:
        plan = loader._plan(epoch)
        batches.append([(item, epoch, index * 16 + offset)
                        for offset, item in enumerate(plan[index])])
    generator = torch.Generator(device="cpu").manual_seed(896)
    data = DataLoader(
        loader.dataset, batch_sampler=batches, num_workers=2, prefetch_factor=2,
        multiprocessing_context="spawn", persistent_workers=False, pin_memory=False,
        generator=generator, worker_init_fn=seed_worker,
        collate_fn=_LogicalCollator(loader.dataset.root, loader.dataset.transform_config, 16),
    )
    iterator = iter(data)
    try:
        yield iterator
    finally:
        if hasattr(iterator, "_shutdown_workers"):
            iterator._shutdown_workers()


def _capacity_memory() -> dict:
    """Conservative allocator peak plus contemporaneous non-allocator overhead."""
    free, total = map(int, torch.cuda.mem_get_info(0))
    reserved = int(torch.cuda.memory_reserved(0))
    peak_reserved = int(torch.cuda.max_memory_reserved(0))
    overhead = max(0, total - free - reserved)
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(0)), "reserved_bytes": reserved,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "peak_reserved_bytes": peak_reserved, "device_free_bytes": free,
        "device_total_bytes": total, "observed_nonallocator_bytes": overhead,
        "conservative_free_at_reserved_peak_bytes": total - peak_reserved - overhead,
    }


def _check_capacity_memory(memory: dict, plan: dict) -> None:
    margin = plan["resource_gate"]["minimum_conservative_free_mib"] * 1024**2
    _require(memory["conservative_free_at_reserved_peak_bytes"] >= margin
             and memory["device_free_bytes"] >= margin,
             "896 capacity headroom below frozen 3072 MiB; no resolution or batch fallback")


def _capacity_state(components) -> dict:
    c = _control()
    engine = components.engine.state_dict()
    updates = engine["optimizer_updates"]
    _require(components.ema.updates == updates and components.warmup.last_step == updates,
             "capacity EMA/warmup clock differs from actual optimizer updates")
    steps = {str(i): int(state["step"].item())
             for i, state in enumerate(components.optimizer.state.values()) if state}
    _require((updates == 0 and not steps) or
             (len(steps) == 326 and set(steps.values()) == {updates}),
             "capacity AdamW parameter clocks differ")
    return {
        "engine": engine, "optimizer_parameter_steps": steps, "ema_updates": components.ema.updates,
        "warmup": c._state_identity(components.warmup.state_dict()),
        "scheduler": c._state_identity(components.scheduler.state_dict()),
        "rng": c._rng_identity(components.runtime),
        "raw_bn": c._bn_state(components.model), "ema_bn": c._bn_state(components.ema.module),
        "geometry": {
            "raw": snapshot_model_geometry(components.model, expected_input_size=896),
            "ema": snapshot_model_geometry(components.ema.module, expected_input_size=896),
        },
    }


def _capacity_update(components, images, targets, monitor, plan, *, case_id, source_input=None,
                     matched_record=None, matched_reference=None) -> dict:
    c = _control()
    _require(list(images.shape) == [16, 3, 896, 896], "capacity image geometry differs")
    if source_input is not None:
        verify_matched_input(matched_record["input"], source_input)
    before = _capacity_state(components)
    before_gate = monitor.check(stage="capacity_before_window")
    torch.cuda.synchronize(0)
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    with c._ForwardObservation(components) as observation:
        window = components.engine.train_window(images, targets)
    torch.cuda.synchronize(0)
    elapsed = time.perf_counter() - started
    after_gate = monitor.check(stage="capacity_after_window")
    memory = _capacity_memory()
    c.verify_window_ledger(
        window, targets, config=components.config, epoch=1,
        logical_batch_index=before["engine"]["optimizer_updates"], batches_per_epoch=304,
    )
    after = _capacity_state(components)
    for name, row in after["raw_bn"].items():
        _require(row["num_batches_tracked"] - before["raw_bn"][name]["num_batches_tracked"] == 2,
                 "capacity BatchNorm clock differs")
    c._same(before["geometry"]["raw"], after["geometry"]["raw"], "capacity raw geometry immutability")
    row = {
        "case_id": case_id, "diagnostic_only": True, "formal_training_samples": 0,
        "source_input": source_input, "matched_receipt": matched_reference,
        "input_shape": list(images.shape), "input_tensor_sha256": augmented_tensor_sha256(images, targets),
        "target_counts_per_image": [len(t["labels"]) for t in targets],
        "window": window, "forward_observation": observation.evidence(),
        "state_after": after, "auxiliary_numerical_health": c._finite_auxiliary_state(components),
        "memory": memory, "synchronized_train_window_seconds": elapsed,
        "hardware_before": before_gate, "hardware_after": after_gate,
        "frozen_margin_pass": memory["conservative_free_at_reserved_peak_bytes"] >= 3072 * 1024**2,
    }
    return row


def run_capacity_worker(contract: dict, *, contract_reference: dict) -> dict:
    """One supervised disposable capacity process; never resume it as formal training."""
    c = _control()
    from .training_v2b_admission import MonitoredHardwareSession
    from .training_v2b_development import evaluate_development_capacity
    checked = c.validate_worker_contract(contract, verify_files=True)
    c._same(c._read_json_reference(contract_reference, "capacity contract"), checked,
            "capacity contract bytes")
    _require(checked["kind"] == "v2b_896_control_worker" and checked["stage"] == "capacity",
             "capacity entry requires its exact 896 contract")
    output = Path(checked["output_dir"])
    output.mkdir(parents=False, exist_ok=False)
    monitor = None
    report = {
        "schema_version": 1, "status": "RUNNING", "stage": "capacity", "arm": "B",
        "run_id": checked["run_id"], "campaign_id": checked["campaign_id"],
        "contract_reference": copy.deepcopy(contract_reference), "worker_pid": os.getpid(),
        "guardian_clearance_required_after_worker_exit": True, "diagnostic_only": True,
        "scientific_certified": False, "formal_training_samples": 0,
        "automatic_retry_or_batch_fallback": False, "cases": [],
    }
    try:
        report["startup_environment"] = c._environment(checked)
        matched = load_matched_control(checked["matched_control_reference"])
        validate_matched_definition(checked, matched)
        report["matched_control_reference"] = checked["matched_control_reference"]
        config = V2BConfig(**checked["config"])
        data = checked["train_core"]
        loader = build_train_core_loader(
            TrainCoreDataConfig(seed=0, input_size=896, logical_batch_size=16,
                                num_workers=2, prefetch_factor=2),
            annotation_file=data["annotation"]["path"], annotation_sha256=data["annotation"]["sha256"],
            manifest_file=data["manifest"]["path"], manifest_sha256=data["manifest"]["sha256"],
            image_root=data["image_root"], repo_root=checked["repo_root"],
        )
        plan = make_capacity_plan(loader)
        c._same(plan, checked["capacity_plan"], "runtime recomputed capacity plan")
        arguments = {"repo_root": checked["repo_root"],
                     "pretrained_path": checked["pretrained"]["path"],
                     "pretrained_sha256": checked["pretrained"]["sha256"]}
        cpu = build_v2b_components(config, **arguments)
        binding = build_train_core_run_binding(
            cpu, loader, run_id=checked["run_id"], repo_root=checked["repo_root"],
            additional_code_paths={name: ref["path"] for name, ref in checked["code_files"].items()},
            requested_device="cuda:0", cuda_gpu_uuid=checked["expected_gpu_uuid"],
        )
        _require(set(binding["code"]).issubset(checked["code_files"]),
                 "capacity imported code lies outside frozen inventory")
        report["binding_reference"] = write_exclusive_json(output / "run-binding.json", binding)
        report["initialization"] = cpu.initialization
        report["matched_initialization"] = compare_matched_initialization(matched["result"], cpu.initialization)
        write_exclusive_json(output / "worker-start.json", report)
        monitor = MonitoredHardwareSession(binding, checked["policy_bundle"],
                                            output / "native-hardware", "paired_smoke", 600)
        monitor.start()
        runtime = prepare_runtime(device="cuda:0", seed=0, binding=binding,
                                  gpu_probe=monitor.admission(), expected_gpu_uuid=checked["expected_gpu_uuid"])
        components = build_v2b_components(config, runtime=runtime, **arguments)
        c._same(cpu.initialization, components.initialization, "capacity CPU/CUDA initialization")
        del cpu
        report["runtime"] = runtime.identity
        report["initial_state_reference"] = write_exclusive_json(
            output / "initial-state.json", _capacity_state(components))
        components.engine.begin_epoch(1)
        index = c._receipt_index(matched["result"])
        positions = plan["warmup_positions"] + plan["stress_positions"]
        images = None
        with _selected_capacity_batches(loader, positions) as batches:
            for epoch, logical_index in positions:
                batch = next(batches)
                _require((batch.epoch, batch.logical_batch_index) == (epoch, logical_index),
                         "capacity loader delivered the wrong planned sample window")
                reference = index[(epoch, logical_index)]
                old = c._read_json_reference(reference, "matched capacity window")
                row = _capacity_update(
                    components, batch.images, batch.targets, monitor, plan,
                    case_id=f"source-epoch-{epoch:03d}-window-{logical_index + 1:04d}",
                    source_input=batch.evidence(), matched_record=old, matched_reference=reference,
                )
                artifact = write_exclusive_json(output / (row["case_id"] + ".json"), row)
                report["cases"].append(artifact)
                _check_capacity_memory(row["memory"], plan)
                images = batch.images
        fixture = torch.Generator(device="cpu").manual_seed(plan["synthetic_envelope"]["cpu_fixture_seed"])
        targets = []
        for count in plan["synthetic_envelope"]["target_counts"]:
            centers = .1 + .8 * torch.rand((count, 2), generator=fixture)
            sizes = .02 + .08 * torch.rand((count, 2), generator=fixture)
            targets.append({"labels": torch.arange(count, dtype=torch.int64) % 10,
                            "boxes": torch.cat((centers, sizes), dim=1)})
        for repeat in range(plan["synthetic_envelope"]["repeat_windows"]):
            row = _capacity_update(
                components, images, targets, monitor, plan, case_id=f"synthetic-envelope-{repeat + 1:02d}",
            )
            artifact = write_exclusive_json(output / (row["case_id"] + ".json"), row)
            report["cases"].append(artifact)
            _check_capacity_memory(row["memory"], plan)
        before = _capacity_state(components)
        component_map = {name: getattr(components, name) for name in
                         ("model", "ema", "postprocessor", "engine", "runtime", "optimizer", "scheduler", "warmup")}
        evaluation_dir = output / "development-capacity"
        evaluation_dir.mkdir()
        def evaluation_gate():
            torch.cuda.synchronize(0)
            return monitor.check(stage="capacity_development_batch")
        torch.cuda.reset_peak_memory_stats(0)
        development = evaluate_development_capacity(
            component_map, data_binding=checked["development_binding"],
            output_dir=evaluation_dir, hardware_gate=evaluation_gate,
        )
        torch.cuda.synchronize(0)
        memory = _capacity_memory()
        report["development_capacity"] = development
        report["development_memory"] = memory
        write_exclusive_json(output / "development-memory.json", memory)
        c._same(before, _capacity_state(components), "full capacity evaluation preserves training state and RNG")
        _require(development["scope"] == "development_capacity_only"
                 and development["scientific_comparison_eligible"] is False
                 and development["data"]["evaluated_image_count"] == 548,
                 "capacity development coverage or eligibility differs")
        _check_capacity_memory(memory, plan)
        report["final_state_reference"] = write_exclusive_json(
            output / "capacity-final-state.json", _capacity_state(components))
        report["diagnostic_optimizer_updates"] = components.engine.optimizer_updates
        report["loader_cursor_unchanged"] = loader.state_dict()
        _require(loader.state_dict()["optimizer_updates"] == 0,
                 "diagnostic windows must not invent committed formal-loader progress")
        report["capacity_acceptance"] = {
            "status": "PASS", "minimum_conservative_free_mib": 3072,
            "native_minimum_free_mib": 3072, "full_development_images": 548,
            "no_empty_cache_or_model_offload": True,
            "sampled_native_margin_requires_controller_final_audit": True,
            "same_arm_replay_requires_separate_smoke_comparison": True,
            "all_case_artifacts_published_before_acceptance": True,
        }
        report["monitor_reference"] = monitor.finish()
        report["status"] = "PASS"
        write_exclusive_json(output / "worker-result.json", report)
        return report
    except BaseException as exc:
        report["status"] = "STOP_NO_RETRY"
        chain, cause = [], exc
        while cause is not None:
            chain.append({"type": type(cause).__name__, "message": str(cause)})
            cause = cause.__cause__
        report["failure"] = chain
        try:
            write_exclusive_json(output / "worker-result.json", report)
        finally:
            if monitor is not None:
                monitor.abort("896 capacity failure: " + type(exc).__name__ + ": " + str(exc))
        raise


def audit_capacity_native_memory(completion: dict) -> dict:
    """Authenticate the final native stream before admitting any formal work."""
    import gzip
    import io
    from .training_v2b_campaign import read_reference
    from .training_v2b_evidence import strict_json_loads
    c = _control()
    monitor = completion.get("monitor", {})
    _require(monitor.get("status") == "PASS" and monitor.get("worker_exited") is True
             and monitor.get("sampled_clock_compliance") is True,
             "capacity guardian did not close with native clearance")
    published = _json_reference(monitor["final_report_reference"])
    c._same(published, {key: value for key, value in monitor.items()
                        if key != "final_report_reference"}, "capacity final guardian bytes")
    result = _json_reference(completion["result_reference"])
    contract = _json_reference(result["contract_reference"])
    launch = _json_reference(completion["launch_reference"])
    binding = _json_reference(result["binding_reference"])
    _require(result.get("status") == "PASS" and result.get("stage") == "capacity"
             and result.get("arm") == "B" and result.get("capacity_acceptance", {}).get("status") == "PASS"
             and contract.get("kind") == "v2b_896_control_worker" and contract.get("stage") == "capacity"
             and contract.get("arm") == "B" and contract.get("campaign_id") == result.get("campaign_id")
             and contract.get("run_id") == result.get("run_id"),
             "native capacity evidence belongs to another stage or campaign")
    validate_capacity_plan(contract["capacity_plan"])
    for name, expected in (("input_size", 896), ("physical_batch_size", 8), ("accumulation_steps", 2)):
        _require(type(contract["config"].get(name)) is int and contract["config"][name] == expected
                 and binding["config"].get(name) == expected,
                 "capacity native evidence binds a different resolution or batch")
    c._same(result["contract_reference"], launch["contract"], "capacity launch contract identity")
    c._same(monitor["owner"], launch["owner"], "capacity launched owner identity")
    _require(launch["pid"] == result["worker_pid"] == monitor["owner"]["pid"],
             "capacity native monitor belongs to another worker")
    for name in ("run_id", "run_binding_sha256", "policy_sha256", "owner",
                 "gpu_uuid", "monitor_pid", "guardian_identity"):
        c._same(monitor[name], result["monitor_reference"][name], "capacity monitor linkage " + name)
    _require(monitor["run_id"] == result["run_id"] == binding["run_id"]
             and monitor["run_binding_sha256"] == binding["binding_sha256"]
             and monitor["gpu_uuid"] == contract["expected_gpu_uuid"],
             "capacity native stream does not bind the actual 896 worker initialization")
    _require(result["monitor_reference"]["monitor_final_report"] ==
             monitor["final_report_reference"]["path"] ==
             str(Path(contract["output_dir"]) / "native-hardware/monitor-final.json"),
             "capacity final monitor path belongs to another execution")
    raw = read_reference(monitor["samples"])
    previous, count, health_count, last_health = None, 0, 0, None
    minimum_free, maximum_used, max_health_gap = math.inf, 0., 0.
    with gzip.GzipFile(fileobj=io.BytesIO(raw), mode="rb") as stream:
        for line in stream:
            entry = strict_json_loads(line)
            _require(set(entry) == {"sequence", "previous_sha256", "record", "sha256"}
                     and type(entry["sequence"]) is int and entry["sequence"] == count
                     and entry["previous_sha256"] == previous,
                     "capacity native sample chain is incomplete or replayed")
            body = {key: value for key, value in entry.items() if key != "sha256"}
            _require(canonical_sha256(body) == entry["sha256"], "capacity native sample hash differs")
            previous, count = entry["sha256"], count + 1
            row = entry["record"]
            if row.get("kind") != "health":
                continue
            stamp = row.get("started_ns")
            _require(type(stamp) is int and (last_health is None or stamp > last_health),
                     "capacity native health order differs")
            if last_health is not None:
                gap = (stamp - last_health) / 1e9
                max_health_gap = max(max_health_gap, gap)
                _require(gap <= 2., "capacity native health sampling gap")
            last_health = stamp
            gpu = row["gpu"]
            _require(gpu["uuid"] == monitor["gpu_uuid"], "capacity native GPU identity changed")
            free, used = gpu["memory_free_mib"], gpu["memory_used_mib"]
            _require(type(free) in (int, float) and math.isfinite(free) and free >= 3072
                     and type(used) in (int, float) and math.isfinite(used) and used >= 0,
                     "896 sampled native headroom below 3072 MiB or unavailable")
            minimum_free, maximum_used = min(minimum_free, free), max(maximum_used, used)
            health_count += 1
    c._same({"records": count, "last_sha256": previous}, monitor["sample_hash_chain"],
            "capacity complete native sample chain")
    _require(health_count >= 3, "capacity has insufficient native memory observations")
    return {"schema_version": 1, "status": "PASS", "native_sample_reference": monitor["samples"],
            "guardian_final_reference": monitor["final_report_reference"],
            "records": count, "health_samples": health_count,
            "sampled_minimum_free_mib": minimum_free, "sampled_maximum_used_mib": maximum_used,
            "maximum_health_gap_seconds": max_health_gap, "minimum_required_free_mib": 3072,
            "whole_native_file_and_chain_authenticated": True,
            "clock_and_hardware_clearance_from_final_independent_guardian": True}
