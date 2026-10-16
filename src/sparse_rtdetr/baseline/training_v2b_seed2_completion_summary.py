"""CPU fixed-epoch summary for a new seed-2/896 completion and historical controls.

The failed replication campaign remains STOP_NO_RETRY.  The new completion gate
authenticates exactly the new smoke, replay and formal endpoint; it never
manufactures the missing sixteenth completion of the old campaign.  Scoring,
pairing and descriptive statistics use the unchanged replication helpers.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import copy
import io
import os
from pathlib import Path
import sys
import time

from .training_v2b_evidence import (
    canonical_json_bytes, canonical_sha256, initial_parameter_reference,
    write_exclusive_json,
)
from . import training_v2b_replication_summary as original
from .training_v2b_replication_summary import (
    _Reader, _require, _same, _reference, _path, _checkpoint_reference,
    _training_source, audit_training_pair, _official_context, _score_endpoint,
    aggregate_fixed_epoch_results,
)

KIND = "v2b_seed2_historical_control_completion_fixed_epoch_summary"
SEEDS = (0, 1, 2)
SIZES = (640, 896)
NEW_RESULT_KIND = "v2b_seed2_historical_control_completion_worker_result"
NEW_COMPLETION_KIND = "v2b_seed2_historical_control_completion_stage_completion"
_AUDIT_SCOPES = []
_AUDIT_INSTALLED = False


def _cpu_audit(event, args):
    if not _AUDIT_SCOPES:
        return
    scope = _AUDIT_SCOPES[-1]
    forbidden = event.startswith(("socket.", "subprocess.", "os.exec", "os.spawn"))
    forbidden = forbidden or event in {"os.system", "os.fork", "os.forkpty"}
    if event == "open":
        path, mode, flags = args
        allowed = isinstance(path, (str, bytes, os.PathLike)) and os.fsdecode(path) == scope["checkpoint_path"]
        if type(path) is int and scope["checkpoint_identity"] is not None:
            info = os.fstat(path)
            allowed = (info.st_dev, info.st_ino, info.st_size) == scope["checkpoint_identity"]
        writes = flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        forbidden = not allowed or bool(writes)
    if forbidden:
        scope["forbidden_calls"].append(event)
        raise original.ReplicationSummaryError("CPU checkpoint bridge forbids " + event)


@contextmanager
def _checkpoint_cpu_guard(checkpoint_path):
    """Allow only the pinned checkpoint's read; forbid devices and model work.

    The audit hook is inert outside this bounded scope.  Python modules needed
    by the native checkpoint inspector are imported before entering it.
    """
    import torch
    global _AUDIT_INSTALLED
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU bridge requires empty CUDA_VISIBLE_DEVICES")
    _require(torch.cuda.is_initialized() is False, "CPU bridge cannot enter an initialized CUDA process")
    _require(not _AUDIT_SCOPES, "CPU checkpoint guards cannot overlap")
    if not _AUDIT_INSTALLED:
        sys.addaudithook(_cpu_audit)
        _AUDIT_INSTALLED = True
    try:
        identity = os.stat(checkpoint_path, follow_symlinks=False)
        identity = (identity.st_dev, identity.st_ino, identity.st_size)
    except FileNotFoundError:
        identity = None
    runtime = {"checkpoint_path": str(checkpoint_path), "checkpoint_identity": identity, "forbidden_calls": [],
               "cuda_initialized_before": False, "cuda_visible_devices": "",
               "map_location": "cpu", "weights_only": True,
               "torch_version": str(torch.__version__), "torch_cpu_threads": torch.get_num_threads()}
    patches = []

    def block(name):
        def rejected(*args, **kwargs):
            runtime["forbidden_calls"].append(name)
            raise original.ReplicationSummaryError("CPU checkpoint bridge forbids " + name)
        return rejected

    def patch(target, name, label):
        if hasattr(target, name):
            # Preserve staticmethod descriptors such as Parameter.__new__.
            old = vars(target).get(name, getattr(target, name))
            patches.append((target, name, old))
            setattr(target, name, block(label))

    try:
        for name in (
            "_lazy_init", "init", "is_available", "device_count", "current_device",
            "get_device_properties", "get_device_capability", "synchronize", "set_device",
            "get_rng_state", "get_rng_state_all", "set_rng_state", "set_rng_state_all",
            "manual_seed", "manual_seed_all", "memory_allocated", "memory_reserved",
        ):
            patch(torch.cuda, name, "torch.cuda." + name)
        for target, name, label in (
            (torch.nn.Module, "__init__", "model construction"),
            (torch.nn.Module, "_call_impl", "model forward"),
            (torch.nn.Module, "register_parameter", "parameter registration"),
            (torch.nn.Module, "register_buffer", "buffer registration"),
            (torch.nn.Parameter, "__new__", "Parameter construction"),
            (torch.Tensor, "cuda", "Tensor.cuda"),
            (torch.Tensor, "pin_memory", "Tensor.pin_memory"),
        ):
            patch(target, name, label)
        _AUDIT_SCOPES.append(runtime)
        yield runtime
        runtime["cuda_initialized_after"] = torch.cuda.is_initialized()
        _require(runtime["cuda_initialized_after"] is False and runtime["forbidden_calls"] == [],
                 "CPU checkpoint guard was violated")
        _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU visibility changed during checkpoint bridge")
    finally:
        if _AUDIT_SCOPES and _AUDIT_SCOPES[-1] is runtime:
            _AUDIT_SCOPES.pop()
        for target, name, old in reversed(patches):
            setattr(target, name, old)


def _all_cpu_tensors(value):
    import torch
    if isinstance(value, torch.Tensor):
        _require(value.device.type == "cpu" and value.layout == torch.strided and not value.is_quantized,
                 "checkpoint must contain only dense unquantized CPU tensors")
        return 1
    if isinstance(value, dict):
        return sum(_all_cpu_tensors(child) for child in value.values())
    if type(value) in (list, tuple):
        return sum(_all_cpu_tensors(child) for child in value)
    _require(value is None or type(value) in (str, bool, int, float),
             "checkpoint contains an unsupported non-tensor value")
    return 0


def _bind_checkpoint_clocks_and_rng(payload, final):
    """Compare stored scalar clocks/RNG bytes to the actual final-state receipt."""
    from . import training_v2b_checkpoint as checkpoint
    from .training_v2b_control import _state_identity, _tensor_identity
    clocks = final["clocks"]
    scheduler = copy.deepcopy(payload["scheduler"]["state"])
    if payload["scheduler"]["type"] == "torch.optim.lr_scheduler.MultiStepLR":
        # Native serialization intentionally converts Counter to its safe dict.
        scheduler["milestones"] = Counter(scheduler["milestones"])
    _same(_state_identity(scheduler), clocks["scheduler"], "payload/final scheduler state")
    _same(_state_identity(payload["warmup"]["state"]), clocks["warmup"], "payload/final warmup state")
    _same([float(group["lr"]) for group in payload["optimizer"]["param_groups"]],
          clocks["learning_rates"], "payload/final learning rates")
    names = {index: name for group in payload["optimizer_layout"]["groups"]
             for index, name in zip(group["ids"], group["names"])}
    steps = {}
    for index, values in payload["optimizer"]["state"].items():
        if values:
            value = values["step"]
            steps[names[index]] = int(value.item() if hasattr(value, "item") else value)
    _same(steps, clocks["adam_steps_by_parameter"], "payload/final per-parameter Adam steps")
    _same(len(steps), clocks["adam_parameter_count_with_state"], "payload/final Adam participation count")
    _same(sorted(set(steps.values())), clocks["adam_step_values"], "payload/final Adam step set")
    _same(payload["ema"]["updates"], clocks["ema_updates"], "payload/final EMA clock")
    rng = payload["rng"]
    identity = {
        "python": canonical_sha256(_state_identity(rng["python"])),
        "numpy": canonical_sha256(_state_identity(checkpoint._numpy_state(rng["numpy"]))),
        "torch_cpu": _tensor_identity(rng["torch_cpu"]),
        "cuda": {name: _tensor_identity(value) for name, value in rng["cuda"].items()},
    }
    _same(identity, final["rng"], "payload/final RNG identity")
    parameters = {row["name"]: payload["model"][row["name"]]
                  for row in payload["model_layout"]["parameters"]}
    _same(initial_parameter_reference(parameters), final["raw_parameters"], "payload/final raw parameters")
    return {"all_saved_update_clocks_match_final": True, "all_saved_RNG_bytes_match_final": True,
            "raw_parameter_tensor_inventory_matches_final": True, "CUDA_RNG_restored": False}


def audit_checkpoint_ema_evaluation(reader, source):
    """Reopen a trusted complete checkpoint on CPU and bridge every EMA tensor.

    The unchanged inspect_checkpoint first validates the payload's model,
    optimizer, RNG, sampler, EMA and scheduler semantics.  A second weights-only
    CPU load of independently authenticated immutable bytes supplies the actual
    EMA module inventory.  No model is instantiated and no RNG is restored.
    """
    import torch
    from . import training_v2b_checkpoint as checkpoint
    # Complete lazy imports before the narrowly scoped file/device guard.
    from . import training_v2b_data, training_v2b_device, training_v2b_engine, training_v2b_control  # noqa: F401

    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU bridge requires empty CUDA_VISIBLE_DEVICES")
    _require(type(source["seed"]) is int and source["seed"] in SEEDS
             and type(source["input_size"]) is int and source["input_size"] in SIZES,
             "unexpected checkpoint cell")
    checkpoint_ref = _reference(source["checkpoint_reference"])
    _require(Path(checkpoint_ref["path"]).suffix == ".pt", "explicit checkpoint .pt file required")
    _same(checkpoint_ref, _checkpoint_reference(source["result"]["checkpoints"]["epoch_30"]["checkpoint"]),
          "bridge endpoint checkpoint")
    evaluation = source["evaluation"]
    published = reader.json(evaluation["result_artifact"])
    _same(published, {k: v for k, v in evaluation.items() if k != "result_artifact"},
          "bridge published development result")
    for key, expected in (("role", "development"), ("logged_epoch", 30),
                          ("full_development_evaluation", True)):
        _same(evaluation.get(key), expected, "bridge development " + key)
    _same(evaluation["policy"]["input_size"], [source["input_size"]] * 2, "bridge evaluation size")
    _same(evaluation["weights"]["kind"], "ema", "bridge evaluation weights")
    _same(evaluation["weights"]["ema_updates"], 9120, "bridge evaluation update count")
    _same(evaluation["training_state_unchanged"]["ema_updates"], 9120, "bridge preserved EMA updates")
    original._state_clocks(source["final"], input_size=source["input_size"])
    raw = reader.read(checkpoint_ref)
    started = time.monotonic()
    with _checkpoint_cpu_guard(checkpoint_ref["path"]) as runtime:
        inspection = checkpoint.inspect_checkpoint(
            checkpoint_ref["path"], expected_sha256=checkpoint_ref["sha256"],
            expected_binding=source["binding"])
        _same({key: inspection[key] for key in ("path", "sha256", "size_bytes")},
              {key: checkpoint_ref[key] for key in ("path", "sha256", "size_bytes")},
              "inspected checkpoint bytes")
        _same(inspection["schema_version"], 2, "current native checkpoint schema")
        _same(inspection["engine"], source["final"]["clocks"]["engine"], "payload/final engine")
        _same(inspection["sampler"]["state"], source["final"]["clocks"]["loader"], "payload/final loader")
        _same(inspection["runtime"], source["result"]["runtime"], "payload/worker runtime")
        payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        payload_tensor_count = _all_cpu_tensors(payload)
        state_binding = _bind_checkpoint_clocks_and_rng(payload, source["final"])
        ema = payload["ema"]
        _same(ema["updates"], 9120, "checkpoint EMA updates")
        _same(payload["binding"], source["binding"], "loaded checkpoint binding")
        _same(payload["engine"], inspection["engine"], "independent CPU load engine")
        state = ema["module"]
        digest = initial_parameter_reference(state)
        expected_weights = evaluation["weights"]["state_sha256"]
        expected_preserved = evaluation["training_state_unchanged"]["ema_sha256"]
        _same(digest["sha256"], expected_weights, "checkpoint/evaluation EMA module tensors")
        _same(digest["sha256"], expected_preserved, "checkpoint/preserved EMA module tensors")
        inventory = digest["inventory"]
        anchors = [row for row in inventory if row["name"].endswith(".anchors")]
        masks = [row for row in inventory if row["name"].endswith(".valid_mask")]
        bn = {suffix: sum(row["name"].endswith("." + suffix) for row in inventory)
              for suffix in ("running_mean", "running_var", "num_batches_tracked")}
        _require(anchors and masks and all(bn.values()), "EMA geometry or BatchNorm buffers are absent")
        _require(len(set(bn.values())) == 1, "EMA BatchNorm buffer inventory is incomplete")
        row = {
            "cell_id": f"s{source['seed']}_r{source['input_size']}", "input_size": source["input_size"],
            "status": "PASS", "checkpoint_reference": checkpoint_ref,
            "development_result_reference": evaluation["result_artifact"],
            "checkpoint_ema_updates": ema["updates"], "module_state_reference": digest,
            "weights_state_sha256": expected_weights,
            "training_state_unchanged_ema_sha256": expected_preserved,
            "both_development_digests_match": True, "all_tensors_cpu": True,
            "tensor_count": len(inventory), "payload_tensor_count": payload_tensor_count,
            "total_tensor_elements": sum(t.numel() for t in state.values()),
            "dtype_counts": dict(sorted(Counter(str(t.dtype) for t in state.values()).items())),
            "bn_buffer_counts": bn, "anchors": anchors, "valid_masks": masks,
            "checkpoint_top_level_keys": sorted(payload),
            "checkpoint_bytes_verified_before_and_after_load": True,
        }
        del payload, ema, state
    del raw
    reader.read(checkpoint_ref, retain=False)
    runtime.pop("checkpoint_path")
    runtime.pop("checkpoint_identity")
    return {
        "schema_version": 1, "kind": "completed_checkpoint_ema_evaluation_binding_cpu",
        "status": "PASS_CHECKPOINT_EMA_EVALUATION_BINDING",
        "digest_domain": "payload['ema']['module']; canonical_tensor_inventory_v1, all tensors",
        "ema_wrapper_digest_compared": False, "cells": [row], "runtime": runtime,
        "native_payload_semantics": {
            "entry": "sparse_rtdetr.baseline.training_v2b_checkpoint.inspect_checkpoint",
            "scope": inspection["inspection_scope"],
            "payload_engine_sampler_and_runtime_match_final_evidence": True,
            **state_binding,
            "live_restore_or_numerical_trajectory_certified": False,
        },
        "no_model_optimizer_loader_or_evaluator_constructed": True,
        "no_dataset_or_prediction_files_read": True,
        "elapsed_seconds": time.monotonic() - started,
    }


def _record_completion(reader, result, completion, completion_reference):
    """Record leaves consumed by unchanged completion validators in this ledger."""
    for reference in (completion_reference, completion["launch_reference"],
                      result["binding_reference"], completion["monitor"]["final_report_reference"]):
        reader.read(reference, retain=False)
    for name in original.NATIVE_LEAVES:
        reader.read(completion["monitor"][name], retain=False)
    if completion.get("capacity_memory_audit_reference") is not None:
        reader.read(completion["capacity_memory_audit_reference"], retain=False)
    for key in ("log_capture_reference", "exit_reference"):
        if key in completion:
            document = reader.json(completion[key])
            if key == "log_capture_reference":
                for name in ("stdout", "stderr"):
                    reader.read(document["references"][name], retain=False)


def _record_new_chain(reader, checked):
    """Inventory the new smoke/replay chain already admitted by the new gate."""
    gate = reader.json(checked["contract"]["stage_gate_reference"])
    for reference in checked.get("verified_leaf_references", []):
        reader.read(reference, retain=False)
    for entry in gate["prerequisites"]:
        result = reader.json(entry["worker_result_reference"])
        contract = reader.json(result["contract_reference"])
        reader.read(contract["stage_gate_reference"], retain=False)
        completion = reader.json(entry["completion_reference"])
        _record_completion(reader, result, completion, entry["completion_reference"])
        for key in ("initial_state_reference", "final_state_reference"):
            reader.read(result[key], retain=False)
        for reference in result["receipts"]:
            reader.read({key: reference[key] for key in original.REF_KEYS}, retain=False)
        for saved in result["checkpoints"].values():
            reader.read(_checkpoint_reference(saved["checkpoint"]), retain=False)
            reader.read(saved["state_reference"], retain=False)
    for reference in gate["smoke_comparisons"].values():
        reader.read(reference, retain=False)


def _historical_source(reader, entry, old_freeze):
    from . import training_v2b_replication_worker as worker
    cell = entry["cell_id"]
    _require(cell in {"s1_r640", "s1_r896", "s2_r640"} and entry["stage"] == "control30",
             "only complete historical fixed controls are accepted")
    document = reader.json(entry["worker_result_reference"])
    contract = reader.json(document["contract_reference"])
    result, contract = worker._successful_cell_source(
        entry["worker_result_reference"], cell_id=cell, stage="control30",
        contract=contract, freeze=old_freeze)
    completion = worker._verify_completion(
        entry["completion_reference"], entry["worker_result_reference"], result,
        contract, contract, old_freeze)
    _record_completion(reader, result, completion, entry["completion_reference"])
    seed, size = int(cell[1]), int(cell.split("_r")[1])
    source = _training_source(
        reader, entry["worker_result_reference"], seed=seed, size=size,
        expected_config=old_freeze["cells"][cell]["config"], completion=completion)
    source.update(completion_reference=entry["completion_reference"],
                  prediction_result=source["evaluation"],
                  prediction_role="saved_formal_training_epoch_30_development_predictions")
    return source


def _seed0_sources(reader, qualification, old_freeze):
    """Reconstruct qualified diagonal sources without invoking the old PASS loader."""
    analysis = reader.json(qualification["analysis_reference"])
    campaign = reader.json(qualification["campaign_reference"])
    terminal = reader.json(qualification["campaign_result_reference"])
    _same(analysis["campaign_reference"], qualification["campaign_reference"], "qualified seed-zero campaign")
    _same(analysis["campaign_result_reference"], qualification["campaign_result_reference"],
          "qualified seed-zero terminal")
    _require(analysis.get("status") == terminal.get("status") == "PASS"
             and set(analysis["cells"]) == {"t640_e640", "t640_e896", "t896_e640", "t896_e896"},
             "qualified seed-zero four-cell analysis is incomplete")
    sources = {}
    for size in SIZES:
        cell = f"t{size}_e{size}"
        row, entry = analysis["cells"][cell], terminal["completed"][cell]
        _same(row["worker_result"], entry["result_reference"], "qualified diagonal worker")
        result = reader.json(row["worker_result"])
        reader.read(entry["completion_reference"], retain=False)
        preparation = reader.json(result["cpu_preparation_reference"])
        checkpoint = campaign["checkpoints"][str(size)]
        _same(preparation["checkpoint"], checkpoint, "qualified fixed seed-zero checkpoint")
        _same(preparation["weights"], "ema", "qualified diagonal inference weights")
        for key, expected in (("seed", 0), ("input_size", size), ("physical_batch_size", 8),
                              ("accumulation_steps", 2), ("sampling_backend", "deterministic_gather")):
            _same(preparation["source_training_config"][key], expected, "qualified seed-zero " + key)
        for key, expected in (("epoch", 30), ("optimizer_updates", 9120), ("microsteps", 18240),
                              ("epoch_active", False), ("failed", False), ("phase", 0)):
            _same(preparation["checkpoint_engine"][key], expected, "qualified checkpoint " + key)
        _same(result["source_training_size"], size, "qualified source resolution")
        _same(result["evaluation_size"], size, "qualified diagonal evaluation resolution")
        config = copy.deepcopy(old_freeze["cells"][f"s1_r{size}"]["config"])
        config["seed"] = 0
        historical_ref = reader.discover(Path(checkpoint["path"]).parent / "worker-result.json")
        source = _training_source(reader, historical_ref, seed=0, size=size, expected_config=config,
                                  expected_checkpoint=checkpoint,
                                  expected_binding_sha=preparation["checkpoint_binding_sha256"])
        for key in ("prediction_artifact", "primary_official_gt", "primary_legacy_formal_gt", "coco_secondary"):
            _same(row[key], result[key], "qualified diagonal saved " + key)
        _same(result["full_development_coverage"]["images"], 548, "qualified diagonal images")
        _same(result["full_development_coverage"]["predictions"], 548 * 300, "qualified diagonal predictions")
        source.update(prediction_result=result,
                      prediction_role="qualified_seed_zero_diagonal_cross_eval_of_fixed_epoch_30_checkpoint",
                      prediction_worker_reference=row["worker_result"],
                      prediction_completion_reference=entry["completion_reference"],
                      prediction_cpu_preparation_reference=result["cpu_preparation_reference"])
        sources[0, size] = source
    return sources, campaign, analysis


def _historical_ema_binding(reader, source, partial):
    from .training_v2b_evidence import _validate_initial
    label = f"s{source['seed']}_r{source['input_size']}"
    bound = partial["ema_binding_audits"][label]
    _same(bound["checkpoint_reference"], source["checkpoint_reference"], "historical EMA partial checkpoint")
    report = reader.json(bound["report_reference"])
    _same(report.get("status"), "PASS_CHECKPOINT_EMA_EVALUATION_BINDING", "historical EMA audit status")
    _same(report.get("kind"), "completed_checkpoint_ema_evaluation_binding_cpu", "historical EMA audit kind")
    _same(report.get("digest_domain"), "payload['ema']['module']; canonical_tensor_inventory_v1, all tensors",
          "historical EMA digest domain")
    _same(report.get("ema_wrapper_digest_compared"), False, "historical EMA wrapper distinction")
    rows = [row for row in report["cells"] if row.get("cell_id") == label]
    _require(len(rows) == 1, "historical EMA audit cell missing or duplicated")
    row = rows[0]
    for key, expected in (("input_size", source["input_size"]), ("status", "PASS"),
                          ("checkpoint_ema_updates", 9120), ("both_development_digests_match", True),
                          ("all_tensors_cpu", True), ("checkpoint_bytes_verified_before_and_after_load", True)):
        _same(row.get(key), expected, "historical EMA " + key)
    _same(row["checkpoint_reference"], source["checkpoint_reference"], "historical EMA checkpoint")
    _same(row["development_result_reference"], source["evaluation"]["result_artifact"],
          "historical EMA evaluation")
    digest = row["module_state_reference"]
    _validate_initial(digest)
    _same(row["tensor_count"], len(digest["inventory"]), "historical EMA full tensor coverage")
    for value in (bound["module_sha256"], row["weights_state_sha256"],
                  row["training_state_unchanged_ema_sha256"], source["evaluation"]["weights"]["state_sha256"],
                  source["evaluation"]["training_state_unchanged"]["ema_sha256"]):
        _same(value, digest["sha256"], "historical full EMA module identity")
    runtime = report["runtime"]
    _require(runtime.get("cuda_initialized_before") is False and runtime.get("cuda_initialized_after") is False
             and runtime.get("forbidden_cuda_calls") == [] and runtime.get("map_location") == "cpu",
             "historical EMA audit CPU scope differs")
    for key in ("manifest_reference", "script_reference"):
        _same(report[key], bound[key], "historical EMA " + key)
        reader.read(bound[key], retain=False)
    for reference in report["input_references"]:
        reader.read(reference, retain=False)
    return {**copy.deepcopy(bound), "source_report_and_full_tensor_inventory_authenticated": True,
            "new_checkpoint_deserialization": False}


def _load_context(reader, completion_reference):
    from .training_v2b_campaign import validate_frozen_source
    from .training_v2b_seed2_completion_worker import validate_completed_seed2_896
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "summary requires empty CUDA_VISIBLE_DEVICES")
    checked = validate_completed_seed2_896(completion_reference, expected_stage="control30", verify_files=True)
    freeze, historical = checked["freeze"], checked["historical"]
    for key, ref_key in (("freeze", "freeze_reference"), ("result", "result_reference"),
                         ("contract", "contract_reference"), ("completion", "completion_reference")):
        _same(reader.json(checked[ref_key]), checked[key], "new authenticated " + key)
    _same(checked["completion_reference"], completion_reference, "requested new completion")
    _same(checked["result"].get("kind"), NEW_RESULT_KIND, "new worker result kind")
    _same(checked["completion"].get("kind"), NEW_COMPLETION_KIND, "new completion kind")
    _same(checked["completion"].get("status"), "PASS", "new completion terminal")
    _same(checked["result"].get("stage"), "control30", "new fixed formal stage")
    _same(checked["result"].get("historical_control_reference"), freeze["historical"]["control_result_reference"],
          "new result's original control")
    _same(historical["failed_campaign"].get("status"), "STOP_NO_RETRY", "original campaign remains failed")
    _same(historical["partial_summary"].get("three_pairs_complete"), False, "historical partial was incomplete")
    _require(Path(__file__).resolve() ==
             Path(freeze["repo_root"]) / "src/sparse_rtdetr/baseline/training_v2b_seed2_completion_summary.py",
             "summary imported from a different frozen checkout")
    validate_frozen_source(freeze["source"])
    for reference in historical["verified_leaf_references"]:
        reader.read(reference, retain=False)
    old = historical["freeze"]
    _same(reader.json(freeze["historical"]["freeze_reference"]), old, "historical frozen source")
    qualification = reader.json(freeze["qualification_reference"])
    _same(qualification, historical["qualification"], "original admitted qualification")
    _require(qualification.get("status") == "PASS"
             and qualification.get("eligible_for_fixed_paired_seed1_and_seed2") is True
             and qualification.get("gpu_admission_granted") is False, "original seed qualification is absent")
    _same(qualification["conditional_authorization_reference"], old["authorization_reference"],
          "original qualification authorization")
    for reference in qualification.get("leaf_references", []):
        reader.read(reference, retain=False)
    _record_new_chain(reader, checked)
    completed = historical["failed_campaign"]["completed"]
    _require(type(completed) is list and len(completed) == 15, "old campaign's actual 15 completions required")
    index = {}
    for entry in completed:
        key = entry["cell_id"], entry["stage"]
        _require(key not in index, "duplicate historical completion")
        index[key] = entry
    expected = {(cell, stage) for cell in original.CELLS
                for stage in ("capacity", "smoke", "smoke_replay", "control30")} - {("s2_r896", "control30")}
    _same([list(key) for key in sorted(index)], [list(key) for key in sorted(expected)],
          "original incomplete invocation set")
    sources = {}
    for cell in ("s1_r640", "s1_r896", "s2_r640"):
        source = _historical_source(reader, index[cell, "control30"], old)
        sources[source["seed"], source["input_size"]] = source
    _same(sources[2, 640]["result_reference"], freeze["historical"]["control_result_reference"],
          "historical A retains its original identity")
    _record_completion(reader, checked["result"], checked["completion"], completion_reference)
    new_source = _training_source(
        reader, checked["result_reference"], seed=2, size=896,
        expected_config=freeze["cells"]["s2_r896"]["config"], completion=checked["completion"])
    new_source.update(completion_reference=completion_reference, prediction_result=new_source["evaluation"],
                      prediction_role="new_seed2_completion_fixed_epoch_30_development_predictions")
    sources[2, 896] = new_source
    seed0, seed0_campaign, analysis = _seed0_sources(reader, qualification, old)
    sources.update(seed0)
    ema = {f"s{seed}_r{size}": _historical_ema_binding(reader, source, historical["partial_summary"])
           for (seed, size), source in sources.items() if (seed, size) != (2, 896)}
    return {
        "freeze": freeze, "freeze_reference": checked["freeze_reference"],
        "completion_reference": completion_reference, "historical": historical,
        "qualification": qualification, "seed0_campaign": seed0_campaign, "analysis": analysis,
        "sources": sources, "historical_ema_bindings": ema,
    }


def validate_seed2_completion_summary_inputs(completion_reference):
    """Input admission only; does not claim receipt pairing or new EMA decoding."""
    context = _load_context(_Reader(), completion_reference)
    return {
        "status": "PASS", "completion_reference": context["completion_reference"],
        "freeze_reference": context["freeze_reference"], "seeds": list(SEEDS), "resolutions": list(SIZES),
        "primary_endpoint_epoch": 30, "historical_campaign_status": "STOP_NO_RETRY",
        "historical_control_is_new_campaign_product": False,
        "full_pairing_and_new_checkpoint_EMA_bridge_still_required": True,
        "gpu_admission_granted": False,
    }


def build_seed2_completion_summary(*, completion_reference, output_dir):
    """Create a new immutable three-pair result without extending any experiment."""
    from .training_v2b_campaign import validate_frozen_source
    reader = _Reader()
    context = _load_context(reader, completion_reference)
    output, freeze = _path(str(output_dir)), context["freeze"]
    parent = Path(freeze["output_root"])
    _require(output.parent == parent and output != parent, "summary must be a new child of the new campaign")
    _require(all(not output.is_relative_to(Path(source["contract"]["output_dir"]))
                 for source in context["sources"].values()), "summary may not enter any training invocation")
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    started = time.monotonic()
    report = {
        "schema_version": 1, "kind": KIND, "status": "RUNNING", "campaign_id": freeze["campaign_id"],
        "completion_reference": completion_reference, "freeze_reference": context["freeze_reference"],
        "qualification_reference": freeze["qualification_reference"],
        "historical_campaign_status": "STOP_NO_RETRY",
        "historical_failed_campaign_reference": freeze["historical"]["failed_campaign_reference"],
        "historical_partial_summary_reference": freeze["historical"]["partial_summary_reference"],
        "historical_capacity_reference": freeze["historical"]["historical_capacity_reference"],
        "historical_control_reference": freeze["historical"]["control_result_reference"],
        "historical_control_is_new_campaign_product": False, "old_16_invocation_PASS_gate_invoked": False,
        "new_formal_cells": ["s2_r896"], "new_engineering_stages": ["smoke", "smoke_replay"],
        "leaf_inventory_scope": "summary reads plus historical admission and qualification leaves; new smoke/replay gate, result, completion, checkpoint-byte, receipt, native and settled-log references reauthenticated",
        "primary_endpoint_epoch": 30, "intermediate_health_epochs_excluded_from_comparison": [10, 20],
        "summary_source": freeze["source"], "pairing_audits": {}, "endpoints": {},
        "ema_binding_audits": copy.deepcopy(context["historical_ema_bindings"]),
        "scope": {
            "gpu_work_launched": False, "image_bytes_opened": False,
            "new_checkpoint_tensor_deserialization": "only_new_s2_r896_on_CPU",
            "confirmatory_or_test_accessed": False, "best_checkpoint_selected": False,
            "model_selection_certification": False, "automatic_experiment_authorization": False,
            "epoch60_authorized": False, "resolution960_authorized": False, "sparse_authorized": False,
        },
        "verification_limits": [
            "No live restoration or numerical trajectory reproduction is claimed by the CPU EMA bridge.",
            "Native completion/leaf bytes are authenticated; per-sample monitor chains are not independently replayed.",
            "Full receipts check inputs and endpoint/update clocks; per-window LR/EMA arithmetic is not independently recomputed.",
            "Cross-resolution losses, BN, DN, evolved RNG and parameters need not be equal.",
        ],
    }
    progress_path = output / "progress.jsonl"
    with progress_path.open("xb", buffering=0) as stream:
        def progress(stage, **fields):
            stream.write(canonical_json_bytes(
                {"stage": stage, "elapsed_seconds": time.monotonic() - started, **fields}) + b"\n")
            os.fsync(stream.fileno())
        try:
            ema = audit_checkpoint_ema_evaluation(reader, context["sources"][2, 896])
            ema_ref = write_exclusive_json(output / "new-s2-r896-checkpoint-ema.json", ema)
            report["ema_binding_audits"]["s2_r896"] = {
                "status": ema["status"], "report_reference": ema_ref,
                "checkpoint_reference": context["sources"][2, 896]["checkpoint_reference"],
                "module_sha256": ema["cells"][0]["module_state_reference"]["sha256"],
                "new_checkpoint_deserialization": True,
            }
            progress("new_checkpoint_ema_bridge_authenticated")
            for seed in SEEDS:
                report["pairing_audits"][str(seed)] = audit_training_pair(
                    reader, context["sources"][seed, 640], context["sources"][seed, 896], progress=progress)
            official = _official_context(reader, context)
            report["official_GT_provenance"] = {
                "binding": official["binding"], "consumer_adapter": official["adapter_current"],
                "development_rebinding": official["development_rebinding"],
                "raw_bytes_independently_verified": False,
                "evidence_boundary": "complete saved development raw lineage; raw annotation bytes not independently reread",
            }
            for seed in SEEDS:
                for size in SIZES:
                    source = context["sources"][seed, size]
                    endpoint = _score_endpoint(reader, source, official, output)
                    endpoint["training_resources"] = report["pairing_audits"][str(seed)]["resources"][str(size)]
                    endpoint["origin_campaign_id"] = source["result"]["campaign_id"]
                    endpoint["is_new_campaign_training_product"] = (seed, size) == (2, 896)
                    endpoint["seed_role"] = "exploratory_conditioning_seed" if seed == 0 else "prospective_replication"
                    report["endpoints"][f"s{seed}_r{size}"] = endpoint
                    progress("fixed_epoch_rescored", seed=seed, input_size=size,
                             primary_AP=endpoint["uniform_official_primary"]["metrics"]["AP"],
                             coco_AP_small=endpoint["converted_COCO_reported_separately"]["metrics"]["AP_small"])
            report["statistics"] = aggregate_fixed_epoch_results(report["endpoints"])
            # End-of-run source and historical failure checks cannot promote old status.
            validate_frozen_source(freeze["source"])
            _same(reader.json(freeze["historical"]["failed_campaign_reference"]),
                  context["historical"]["failed_campaign"], "unchanged original failed campaign")
            inventory = reader.inventory()
            report["consumed_leaf_inventory_reference"] = write_exclusive_json(
                output / "consumed-leaf-inventory.json",
                {"schema_version": 1, "references": inventory, "inventory_sha256": canonical_sha256(inventory)})
            report.update(status="PASS", three_pairs_complete=True, elapsed_seconds=time.monotonic() - started)
            progress("finished", endpoint_count=6, paired_seed_count=3)
        except BaseException as exc:
            report.update(status="STOP_NO_RETRY", three_pairs_complete=False,
                          elapsed_seconds=time.monotonic() - started,
                          failure={"type": type(exc).__name__, "message": str(exc)})
            progress("failed", **report["failure"])
            write_exclusive_json(output / "summary.json", report)
            raise
    report["progress_reference"] = reader.discover(progress_path)
    return {"summary": report, "summary_reference": write_exclusive_json(output / "summary.json", report)}
