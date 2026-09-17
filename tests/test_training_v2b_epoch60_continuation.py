from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest

from sparse_rtdetr.baseline import training_v2b_epoch60_continuation as gate


ROOT = Path(__file__).resolve().parents[1]


def test_orchestration_inventory_binds_runtime_device_capability_owner():
    assert "src/sparse_rtdetr/baseline/training_v2b_device.py" in gate.ORCHESTRATION_FILES


def write_json(path: Path, value: dict) -> dict:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
                    encoding="utf-8")
    return gate.file_reference(path)


def dummy_reference(tmp_path: Path, name: str, value: dict | None = None) -> dict:
    return write_json(tmp_path / name, value or {"name": name})


@pytest.fixture()
def definition(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "new-output"
    report_ref = dummy_reference(tmp_path, "t8a.json")
    sources = {relative: gate.file_reference(ROOT / relative) for relative in gate.ORCHESTRATION_FILES}
    cells = {}
    orders = {str(epoch): (f"{epoch:064x}")[-64:] for epoch in range(31, 61)}
    for cell_id in gate.CELLS:
        seed, size = int(cell_id[1]), int(cell_id[4:])
        cells[cell_id] = {
            "cell_id": cell_id, "seed": seed, "input_size": size,
            "source_repo_root": str(source.resolve()),
            "source_campaign_id": "historical-campaign-" + str(seed),
            "source_run_id": "historical-run-" + cell_id,
            "source_contract_reference": dummy_reference(tmp_path, cell_id + "-contract.json"),
            "source_binding_reference": dummy_reference(tmp_path, cell_id + "-binding.json"),
            "source_checkpoint_reference": dummy_reference(tmp_path, cell_id + "-checkpoint.json"),
            "source_boundary_state_reference": dummy_reference(tmp_path, cell_id + "-state.json"),
            "source_worker_result_reference": dummy_reference(tmp_path, cell_id + "-result.json"),
            "source_binding_sha256": (str(seed) + str(size))[:1] * 64,
            "config": {"seed": seed, "input_size": size},
            "pretrained": {"authority": True}, "train_core": {"role": "train_core"},
            "development_binding": {"role": "development"},
            "expected_gpu_uuid": "GPU-00000000-0000-0000-0000-000000000000",
            "startup_environment": gate.worker_startup_environment(
                "GPU-00000000-0000-0000-0000-000000000000"),
            "num_workers": 2, "prefetch_factor": 2,
            "epoch_order_sha256": copy.deepcopy(orders),
        }
    freeze = {
        "schema_version": 1, "kind": gate.FREEZE_KIND,
        "campaign_id": "epoch60-new-campaign", "repo_root": str(ROOT.resolve()),
        "output_root": str(output.resolve()), "source_epoch": 30, "target_epoch": 60,
        "t8a_report_reference": report_ref,
        "training_core_identities": {"core.py": {"sha256": "a" * 64, "size_bytes": 1}},
        "execution_policy": copy.deepcopy(gate.EXECUTION_POLICY),
        "loader_runtime_policy": copy.deepcopy(gate.CONTINUATION_LOADER_RUNTIME_POLICY),
        "orchestration_sources": sources, "cells": cells,
    }
    gate.validate_epoch60_continuation_freeze(freeze, verify_files=False)
    freeze_ref = write_json(tmp_path / "freeze.json", freeze)
    return freeze, freeze_ref, tmp_path


def result_reference(tmp_path, freeze_ref, cell_id, stage, *, suffix=""):
    value = {"schema_version": 1, "kind": gate.RESULT_KIND, "status": "PASS",
             "cell_id": cell_id, "stage": stage, "freeze_reference": freeze_ref}
    return write_json(tmp_path / f"{cell_id}-{stage}{suffix}.json", value)


def contract(definition, cell_id="s1_r640", stage="smoke", *, smoke=None,
             replay=None, matched=None):
    freeze, freeze_ref, _ = definition
    return gate.continuation_worker_contract(
        freeze, freeze_ref, cell_id=cell_id, stage=stage,
        policy_bundle={"native": "policy"}, smoke_reference=smoke,
        replay_reference=replay, matched_reference=matched)


def monitor_finish_descriptor(selected, *, worker_pid=1234, monitor_pid=1235):
    identity = lambda pid: {"pid": pid, "start_ticks": pid * 10, "readable": True,
                            "executable": {"path": "/frozen/python", "sha256": "f" * 64,
                                           "size_bytes": 1,
                                           "sha256_scope": "complete_file_bytes"}}
    return {
        "monitor_final_report": str(
            Path(selected["output_dir"]) / "native-hardware/monitor-final.json"),
        "monitor_pid": monitor_pid, "run_id": selected["source_run_id"],
        "guardian_identity": identity(monitor_pid),
        "gpu_uuid": selected["expected_gpu_uuid"],
        "run_binding_sha256": selected["source_binding_sha256"],
        "policy_sha256": "e" * 64, "owner": identity(worker_pid),
        "worker_must_exit": True,
    }


def test_freeze_separates_historical_training_identity_from_new_execution(definition):
    freeze, _, _ = definition
    checked = gate.validate_epoch60_continuation_freeze(freeze, verify_files=False)
    for cell_id, cell in checked["cells"].items():
        assert checked["campaign_id"] != cell["source_campaign_id"]
        assert checked["campaign_id"] not in cell["source_run_id"]
        assert set(cell["epoch_order_sha256"]) == {str(epoch) for epoch in range(31, 61)}
    assert checked["execution_policy"]["historical_run_not_relabelled"] is True
    assert checked["execution_policy"]["training_core_semantics_change_forbidden"] is True


@pytest.mark.parametrize("field,value", [
    ("source_checkpoint_immutable", False),
    ("source_run_binding_reused_exactly", False),
    ("historical_run_not_relabelled", False),
    ("formal_retry_or_resume_after_failure", True),
    ("automatic_batch_or_precision_fallback", True),
    ("confirmatory_or_test_access", True),
])
def test_execution_policy_is_closed(definition, field, value):
    freeze, _, _ = definition
    mutated = copy.deepcopy(freeze)
    mutated["execution_policy"][field] = value
    with pytest.raises(gate.Epoch60ContinuationError, match="execution policy drift"):
        gate.validate_epoch60_continuation_freeze(mutated)


@pytest.mark.parametrize("field", [
    "source_run_id", "source_binding_sha256", "source_checkpoint_reference",
    "source_boundary_state_reference", "epoch_order_sha256",
])
def test_worker_cannot_mutate_source_lineage(definition, field):
    value = contract(definition)
    mutated = copy.deepcopy(value)
    if field == "epoch_order_sha256":
        mutated[field]["31"] = "f" * 64
    elif field.endswith("_reference"):
        mutated[field]["sha256"] = "f" * 64
    else:
        mutated[field] = "f" * 64
    with pytest.raises(gate.Epoch60ContinuationError, match="worker lineage"):
        gate.validate_continuation_worker_contract(mutated, definition[0])


def test_outer_execution_identity_is_new_but_source_binding_is_unchanged(definition):
    value = contract(definition)
    assert value["execution_id"] == "epoch60-new-campaign-s1_r640-smoke"
    assert value["execution_id"] != value["source_run_id"]
    assert value["source_binding_sha256"] == definition[0]["cells"]["s1_r640"]["source_binding_sha256"]


def test_worker_contract_binds_the_complete_historical_startup_environment(definition):
    value = contract(definition)
    assert value["startup_environment"] == {
        "CUDA_VISIBLE_DEVICES": "GPU-00000000-0000-0000-0000-000000000000",
        "MKL_THREADING_LAYER": "GNU",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONHASHSEED": "0",
    }


def test_freeze_rejects_startup_environment_drift(definition):
    freeze = copy.deepcopy(definition[0])
    freeze["cells"]["s1_r640"]["startup_environment"]["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    with pytest.raises(gate.Epoch60ContinuationError, match="startup environment drift"):
        gate.validate_epoch60_continuation_freeze(freeze)


@pytest.mark.parametrize("name", [
    "CUDA_VISIBLE_DEVICES", "MKL_THREADING_LAYER", "PYTHONNOUSERSITE",
    "PYTHONDONTWRITEBYTECODE", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED",
])
def test_worker_contract_rejects_each_startup_environment_mutation(definition, name):
    value = contract(definition)
    value["startup_environment"][name] += "-drift"
    with pytest.raises(gate.Epoch60ContinuationError, match=r"startup.?environment drift"):
        gate.validate_continuation_worker_contract(value, definition[0])


def test_process_environment_gate_closes_all_keys_before_imports(definition, monkeypatch):
    value = contract(definition)
    for name, expected in value["startup_environment"].items():
        monkeypatch.setenv(name, expected)
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    assert gate.validate_worker_process_environment(value) == value["startup_environment"]

    for missing in value["startup_environment"]:
        with monkeypatch.context() as scoped:
            scoped.delenv(missing, raising=False)
            with pytest.raises(gate.Epoch60ContinuationError,
                               match="process startup environment drift"):
                gate.validate_worker_process_environment(value)


def test_process_environment_gate_requires_interpreter_bytecode_disable(definition, monkeypatch):
    value = contract(definition)
    for name, expected in value["startup_environment"].items():
        monkeypatch.setenv(name, expected)
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    with pytest.raises(gate.Epoch60ContinuationError, match="bytecode generation"):
        gate.validate_worker_process_environment(value)


def test_smoke_replay_and_formal_require_exact_prerequisites(definition):
    freeze_ref, tmp = definition[1], definition[2]
    smoke = result_reference(tmp, freeze_ref, "s1_r640", "smoke")
    replay = result_reference(tmp, freeze_ref, "s1_r640", "smoke_replay")
    replay_contract = contract(definition, stage="smoke_replay", smoke=smoke)
    assert replay_contract["smoke_reference"] == smoke
    formal = contract(definition, stage="formal60", smoke=smoke, replay=replay)
    assert formal["replay_reference"] == replay
    with pytest.raises(gate.Epoch60ContinuationError, match="passed smoke and replay"):
        contract(definition, stage="formal60", smoke=smoke)


def test_896_requires_same_seed_640_result_for_each_stage(definition):
    freeze_ref, tmp = definition[1], definition[2]
    correct = result_reference(tmp, freeze_ref, "s1_r640", "smoke")
    value = contract(definition, cell_id="s1_r896", matched=correct)
    assert value["matched_reference"] == correct
    wrong = result_reference(tmp, freeze_ref, "s2_r640", "smoke")
    with pytest.raises(gate.Epoch60ContinuationError, match="result cell drift"):
        contract(definition, cell_id="s1_r896", matched=wrong)


def test_640_rejects_a_matched_reference(definition):
    freeze_ref, tmp = definition[1], definition[2]
    matched = result_reference(tmp, freeze_ref, "s1_r640", "smoke")
    with pytest.raises(gate.Epoch60ContinuationError, match="640 matched reference drift"):
        contract(definition, matched=matched)


def test_existing_output_is_rejected_before_worker_execution(definition):
    freeze, _, _ = definition
    output = Path(freeze["output_root"]) / "execution" / "s1_r640-smoke"
    output.mkdir(parents=True)
    value = {
        "schema_version": 1, "kind": gate.CONTRACT_KIND,
    }
    # Build through the public constructor so all other fields are authoritative.
    output.rmdir()
    output.parent.rmdir()
    value = contract(definition)
    output.mkdir(parents=True)
    with pytest.raises(gate.Epoch60ContinuationError, match="output already exists"):
        gate.validate_continuation_worker_contract(value, freeze)


@pytest.mark.parametrize("stage,receipt_count,clocks", [
    ("smoke", 4, {"epoch": 31, "epoch_active": True, "optimizer_updates": 9124, "microsteps": 18248}),
    ("smoke_replay", 2, {"epoch": 31, "epoch_active": True, "optimizer_updates": 9124, "microsteps": 18248}),
    ("formal60", 9120, {"epoch": 60, "epoch_active": False, "optimizer_updates": 18240, "microsteps": 36480}),
])
def test_result_validator_enforces_exact_stage_clocks(definition, stage, receipt_count, clocks):
    freeze_ref, tmp = definition[1], definition[2]
    smoke = result_reference(tmp, freeze_ref, "s1_r640", "smoke", suffix="-pre")
    replay = result_reference(tmp, freeze_ref, "s1_r640", "smoke_replay", suffix="-pre")
    kwargs = ({"smoke": smoke} if stage == "smoke_replay" else
              {"smoke": smoke, "replay": replay} if stage == "formal60" else {})
    selected = contract(definition, stage=stage, **kwargs)
    refs = {name: dummy_reference(tmp, f"{stage}-{name}.json") for name in
            ("contract_reference", "restored_boundary_reference", "final_state_reference")}
    result = {**refs, "schema_version": 1, "kind": gate.RESULT_KIND, "status": "PASS",
              "campaign_id": selected["campaign_id"], "execution_id": selected["execution_id"],
              "cell_id": selected["cell_id"], "stage": stage,
              "source_run_id": selected["source_run_id"],
              "source_binding_sha256": selected["source_binding_sha256"],
              "freeze_reference": selected["freeze_reference"], "receipt_count": receipt_count,
              "loader_runtime_policy": selected["loader_runtime_policy"],
              "loader_runtime_observation": {
                  "num_workers": 0, "prefetch_factor": 2,
                  "multiprocessing_context": "spawn", "worker_processes": 0,
                  "automatic_batch_or_precision_fallback": False,
              },
              "startup_environment": selected["startup_environment"],
              "historical_startup_environment": {
                  name: selected["startup_environment"][name]
                  for name in ("CUDA_VISIBLE_DEVICES", "MKL_THREADING_LAYER",
                               "PYTHONNOUSERSITE", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                               "CUBLAS_WORKSPACE_CONFIG")
              },
              "final_clocks": clocks, "worker_pid": 1234,
              "monitor_reference": monitor_finish_descriptor(selected)}
    gate.validate_continuation_result(result, selected)
    mutated = copy.deepcopy(result)
    mutated["final_clocks"]["optimizer_updates"] += 1
    with pytest.raises(gate.Epoch60ContinuationError, match="final clock optimizer_updates drift"):
        gate.validate_continuation_result(mutated, selected)


def test_result_monitor_finish_descriptor_is_not_a_file_reference(definition):
    selected = contract(definition)
    descriptor = monitor_finish_descriptor(selected)
    assert gate.validate_monitor_finish_descriptor(
        descriptor, selected, worker_pid=1234) == descriptor
    mutated = copy.deepcopy(descriptor)
    mutated["monitor_final_report"] = str(Path(selected["output_dir"]) / "elsewhere.json")
    with pytest.raises(gate.Epoch60ContinuationError, match="final report path drift"):
        gate.validate_monitor_finish_descriptor(mutated, selected, worker_pid=1234)


def test_post_exit_completion_requires_passing_bound_monitor(definition):
    freeze_ref, tmp = definition[1], definition[2]
    selected = contract(definition)
    output = Path(selected["output_dir"])
    native = output / "native-hardware"
    native.mkdir(parents=True)
    contract_ref = dummy_reference(tmp, "completion-contract.json")
    restored = dummy_reference(tmp, "completion-restored.json")
    final_state = dummy_reference(tmp, "completion-final-state.json")
    descriptor = monitor_finish_descriptor(selected)
    result = {
        "schema_version": 1, "kind": gate.RESULT_KIND, "status": "PASS",
        "campaign_id": selected["campaign_id"], "execution_id": selected["execution_id"],
        "cell_id": selected["cell_id"], "stage": selected["stage"],
        "source_run_id": selected["source_run_id"],
        "source_binding_sha256": selected["source_binding_sha256"],
        "freeze_reference": freeze_ref, "receipt_count": 4,
        "loader_runtime_policy": selected["loader_runtime_policy"],
        "loader_runtime_observation": {
            "num_workers": 0, "prefetch_factor": 2,
            "multiprocessing_context": "spawn", "worker_processes": 0,
            "automatic_batch_or_precision_fallback": False,
        },
        "startup_environment": selected["startup_environment"],
        "historical_startup_environment": {
            name: selected["startup_environment"][name]
            for name in ("CUDA_VISIBLE_DEVICES", "MKL_THREADING_LAYER", "PYTHONNOUSERSITE",
                         "OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUBLAS_WORKSPACE_CONFIG")
        },
        "final_clocks": {"epoch": 31, "epoch_active": True,
                         "optimizer_updates": 9124, "microsteps": 18248},
        "contract_reference": contract_ref,
        "restored_boundary_reference": restored,
        "final_state_reference": final_state,
        "worker_pid": 1234, "monitor_reference": descriptor,
    }
    result_ref = write_json(output / "worker-result.json", result)
    leaves = {name: dummy_reference(tmp, "monitor-" + name + ".json") for name in
              ("samples", "setter_receipt", "startup_evidence", "guardian_heartbeat")}
    monitor = {
        **{name: descriptor[name] for name in (
            "run_id", "run_binding_sha256", "policy_sha256", "monitor_pid", "owner",
            "guardian_identity", "gpu_uuid")},
        **leaves, "status": "PASS", "worker_exited": True,
        "sampled_clock_compliance": True, "loaded_clock_samples": 3,
    }
    monitor_ref = write_json(native / "monitor-final.json", monitor)
    launch_ref = write_json(tmp / "completion-launch.json", {
        "pid": 1234, "owner": descriptor["owner"], "contract": contract_ref})
    stdout_ref = dummy_reference(tmp, "completion.stdout")
    stderr_ref = dummy_reference(tmp, "completion.stderr")
    exit_ref = write_json(tmp / "completion-exit.json", {
        "returncode": 0, "pid": 1234, "contract": contract_ref,
        "stdout": stdout_ref, "stderr": stderr_ref})
    completion = {
        "schema_version": 1, "kind": gate.COMPLETION_KIND, "status": "PASS",
        "campaign_id": selected["campaign_id"], "execution_id": selected["execution_id"],
        "cell_id": selected["cell_id"], "stage": selected["stage"],
        "contract_reference": contract_ref, "result_reference": result_ref,
        "monitor_final_reference": monitor_ref, "launch_reference": launch_ref,
        "exit_reference": exit_ref, "elapsed_seconds": 1.0, "automatic_retry": False,
    }
    assert gate.validate_continuation_completion(
        completion, selected, verify_files=True) == completion
    mutations = (
        ("status", "FAIL", "monitor status drift"),
        ("worker_exited", False, "monitor worker exit drift"),
        ("sampled_clock_compliance", False, "sampled clock compliance drift"),
        ("loaded_clock_samples", 2, "loaded clock coverage missing"),
        ("policy_sha256", "d" * 64, "monitor policy_sha256 drift"),
    )
    for field, value, message in mutations:
        changed = copy.deepcopy(monitor)
        changed[field] = value
        (native / "monitor-final.json").write_text(
            json.dumps(changed, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8")
        mutated_completion = copy.deepcopy(completion)
        mutated_completion["monitor_final_reference"] = gate.file_reference(
            native / "monitor-final.json")
        with pytest.raises(gate.Epoch60ContinuationError, match=message):
            gate.validate_continuation_completion(
                mutated_completion, selected, verify_files=True)


def test_freeze_rejects_future_order_coverage_or_digest_drift(definition):
    freeze = copy.deepcopy(definition[0])
    del freeze["cells"]["s2_r896"]["epoch_order_sha256"]["60"]
    with pytest.raises(gate.Epoch60ContinuationError, match="future epoch order coverage"):
        gate.validate_epoch60_continuation_freeze(freeze)
    freeze = copy.deepcopy(definition[0])
    freeze["cells"]["s2_r896"]["epoch_order_sha256"]["31"] = "ABC"
    with pytest.raises(gate.Epoch60ContinuationError, match="order digest invalid"):
        gate.validate_epoch60_continuation_freeze(freeze)
