"""CPU contract/orchestration tests; no images, native GPU admission or GPU work."""
from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from sparse_rtdetr.baseline import training_v2b_control as control
from sparse_rtdetr.baseline import training_v2b_replication_worker as worker
from sparse_rtdetr.baseline.training_v2b import V2BConfig, logical_batch_indices
from sparse_rtdetr.baseline.training_v2b_evidence import canonical_sha256, file_reference, write_exclusive_json


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail("CPU replication tests must not initialize or observe a GPU")
    monkeypatch.setattr(torch.cuda, "_lazy_init", blocked)
    monkeypatch.setattr(torch.cuda, "init", blocked)
    monkeypatch.setattr(torch.cuda, "is_available", blocked)
    monkeypatch.setattr(worker, "prepare_runtime", blocked)
    monkeypatch.setattr(worker, "build_v2b_components", blocked)
    monkeypatch.setattr(worker, "build_train_core_loader", blocked)


def ref(path, sha="a" * 64, size=1):
    return {"path": str(path), "sha256": sha, "size_bytes": size,
            "sha256_scope": "complete_file_bytes"}


@pytest.fixture
def definition(tmp_path):
    root, data, output = tmp_path / "checkout", tmp_path / "metadata", tmp_path / "outputs"
    output.mkdir()
    authorization = ref(tmp_path / "authorization.json")
    code = {"src/file.py": ref(root / "src/file.py")}
    policy = {"setter_mode": "external_admin_acknowledged", "authorization_reference": authorization}
    common = {
        "torch_version": str(torch.__version__), "code_files": code,
        "train_core": {
            "annotation": ref(data / "train_core_coco.json", control.TRAIN_ANNOTATION_SHA256, 51148042),
            "manifest": ref(data / "train_core_manifest.json", control.TRAIN_MANIFEST_SHA256, 2955809),
            "image_root": str(tmp_path / "allowed_train_images"),
            "expected_samples": 4869, "expected_batches_per_epoch": 304,
        },
        "pretrained": ref(tmp_path / "ResNet18_vd_pretrained_from_paddle.pth",
                          control.PRETRAINED_SHA256, 44878642),
        "policy_bundle": policy, "expected_gpu_uuid": "GPU-1faee6f0-1da7-4ede-2475-67a5a00274a8",
        "num_workers": 2, "prefetch_factor": 2,
    }
    freeze = {
        "schema_version": 1, "kind": worker.FREEZE_KIND, "campaign_id": "cpu-replication",
        "repo_root": str(root), "output_root": str(output),
        "source": {"repo_root": str(root), "commit": "a" * 40, "tree": "b" * 40, "branch": "cpu-fixture",
                   "code_files": code, "inventory_sha256": canonical_sha256(code)},
        "clean_archive_reference": ref(tmp_path / "archive.json"),
        "qualification_reference": ref(tmp_path / "qualification.json"),
        "authorization_reference": authorization, "foundation_reference": ref(tmp_path / "foundation.json"),
        "common": common, "cells": {}, "formal_order": list(worker.CELLS),
    }
    for cell in worker.CELLS:
        seed, size = int(cell[1]), int(cell.split("r")[1])
        invocations = {}
        for stage in worker.STAGES:
            run_stage = "smoke" if stage == "smoke_replay" else stage
            invocations[stage] = {
                "run_id": f"cpu-{cell}-{run_stage}", "invocation_id": f"invocation-{cell}-{stage}",
                "output_dir": str(output / f"{cell}-{stage}"),
            }
        freeze["cells"][cell] = {
            "config": asdict(V2BConfig(seed=seed, input_size=size, physical_batch_size=8,
                                       accumulation_steps=2, sampling_backend="deterministic_gather")),
            "development_binding": {
                "annotation": {k: v for k, v in ref(data / "development_coco.json").items() if k != "sha256_scope"},
                "manifest": {k: v for k, v in ref(data / "development_manifest.json").items() if k != "sha256_scope"},
                "image_root": str(tmp_path / "allowed_dev_images"), "binding_sha256": "b" * 64,
                "policy": {"input_size": [size, size]},
            },
            "epoch_order_sha256": worker.frozen_seed_orders(seed),
            "capacity_plan": ref(tmp_path / (cell + "-capacity-plan.json")), "invocations": invocations,
        }
    return freeze


def select_contract(freeze, cell="s1_r640", stage="smoke"):
    selected = freeze["cells"][cell]
    output = Path(freeze["output_root"])
    peer_stage = "control30" if stage == "control30" else "smoke"
    peer = cell.replace("r896", "r640")
    matched = (ref(Path(freeze["cells"][peer]["invocations"][peer_stage]["output_dir"]) / "worker-result.json")
               if cell.endswith("r896") and stage != "capacity" else None)
    replay = (ref(Path(selected["invocations"]["smoke"]["output_dir"]) / "worker-result.json")
              if stage == "smoke_replay" else None)
    return copy.deepcopy({
        "schema_version": 1, "kind": worker.KIND, "campaign_id": freeze["campaign_id"],
        "cell_id": cell, "stage": stage, "repo_root": freeze["repo_root"],
        **selected["invocations"][stage], **freeze["common"],
        **{name: selected[name] for name in worker.CELL_KEYS - {"invocations"}},
        "freeze_reference": ref(output / "freeze.json"),
        "qualification_reference": freeze["qualification_reference"],
        "stage_gate_reference": None if stage == "capacity" else ref(output / (cell + "-gate.json")),
        "matched_reference": matched, "replay_source": replay,
    })


def make_gate(contract, freeze):
    rows = []
    for cell, stage in sorted(worker._expected_prerequisites(contract, freeze)):
        result = ref(Path(freeze["cells"][cell]["invocations"][stage]["output_dir"]) / "worker-result.json")
        rows.append({"cell_id": cell, "stage": stage, "worker_result_reference": result,
                     "completion_reference": ref(Path(freeze["output_root"]) / f"{cell}-{stage}-completion.json")})
    return {
        "schema_version": 1, "kind": worker.GATE_KIND, "status": "PASS",
        "campaign_id": contract["campaign_id"], "freeze_reference": contract["freeze_reference"],
        "cell_id": contract["cell_id"], "stage": contract["stage"], "prerequisites": rows,
        "smoke_comparisons": ({cell: ref(Path(freeze["output_root"]) / (cell + "-comparison.json"))
                               for cell in worker.CELLS} if contract["stage"] == "control30" else {}),
    }


@pytest.mark.parametrize("cell", worker.CELLS)
@pytest.mark.parametrize("stage", worker.STAGES)
def test_all_frozen_contracts_are_metadata_only(definition, cell, stage, monkeypatch):
    contract = select_contract(definition, cell, stage)
    monkeypatch.setattr(control, "file_reference", lambda *a, **k: pytest.fail("metadata validator read bytes"))
    monkeypatch.setattr(Path, "read_bytes", lambda *a: pytest.fail("metadata validator opened inputs"))
    assert worker.validate_replication_worker_contract(contract) == contract
    assert worker.validate_replication_freeze(definition, contract) == definition


def test_seed_orders_are_complete_one_based_and_detached():
    plans = [worker.frozen_seed_orders(seed) for seed in (1, 2)]
    assert all(set(plan) == {str(epoch) for epoch in range(1, 31)} for plan in plans)
    assert all(plans[0][str(epoch)] != plans[1][str(epoch)] for epoch in range(1, 31))
    for seed in (1, 2):
        batches = [list(batch) for batch in logical_batch_indices(4869, seed=seed, epoch=30, logical_batch_size=16)]
        assert len(batches) == 304 and len(set(sum(batches, []))) == 4864
        assert canonical_sha256(batches) == plans[seed - 1]["30"]
    plans[0]["1"] = "changed"
    assert worker.frozen_seed_orders(1)["1"] != "changed"
    with pytest.raises(worker.ReplicationWorkerError):
        worker.frozen_seed_orders(True)


@pytest.mark.parametrize("keys,value", [
    (("schema_version",), True), (("kind",), "v2b_640_control_worker"),
    (("cell_id",), "s0_r640"), (("stage",), "train120"), (("run_id",), "../reuse"),
    (("invocation_id",), "bad name"), (("config", "seed"), 0), (("config", "seed"), True),
    (("config", "input_size"), 960), (("config", "physical_batch_size"), 16),
    (("config", "accumulation_steps"), 1), (("config", "sampling_backend"), "native"),
    (("config", "amp_dtype"), "float32"), (("config", "bn_statistics"), "frozen"),
    (("config", "num_denoising"), 0), (("config", "warmup_steps"), 20),
    (("num_workers",), 0), (("prefetch_factor",), True), (("torch_version",), "unknown"),
    (("expected_gpu_uuid",), "0"), (("epoch_order_sha256", "10"), "b" * 64),
    (("train_core", "expected_samples"), 4864), (("train_core", "expected_batches_per_epoch"), 305),
    (("train_core", "annotation", "size_bytes"), 1), (("pretrained", "sha256"), "b" * 64),
    (("policy_bundle", "setter_mode"), "sudo_n"), (("freeze_reference", "sha256_scope"), "prefix"),
    (("development_binding", "policy", "input_size"), [640, 896]),
])
def test_contract_rejects_scope_drift_before_io(definition, keys, value, monkeypatch):
    contract = select_contract(definition)
    target = contract
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    monkeypatch.setattr(control, "file_reference", lambda *a, **k: pytest.fail("unexpected byte read"))
    with pytest.raises((ValueError, RuntimeError)):
        worker.validate_replication_worker_contract(contract)


@pytest.mark.parametrize("key", ["freeze_reference", "qualification_reference", "capacity_plan"])
def test_new_references_require_exact_four_fields(definition, key):
    contract = select_contract(definition)
    del contract[key]["sha256_scope"]
    with pytest.raises(worker.ReplicationWorkerError, match="four"):
        worker.validate_replication_worker_contract(contract)


@pytest.mark.parametrize("key", ["stage_gate_reference", "matched_reference", "replay_source"])
def test_capacity_has_no_historical_source_or_replay(definition, key, tmp_path):
    contract = select_contract(definition, stage="capacity")
    contract[key] = ref(tmp_path / "unexpected.json")
    with pytest.raises(worker.ReplicationWorkerError, match="capacity has no"):
        worker.validate_replication_worker_contract(contract)


def test_validation_does_not_reject_existing_capacity_output(definition):
    contract = select_contract(definition, stage="capacity")
    Path(contract["output_dir"]).mkdir()
    assert worker.validate_replication_worker_contract(contract)["stage"] == "capacity"


@pytest.mark.parametrize("role", ["train_core", "pretrained", "development", "source", "output", "authority"])
def test_forbidden_roles_rejected_without_content_reads(definition, role, tmp_path, monkeypatch):
    contract = select_contract(definition)
    forbidden = tmp_path / "confirmatory" / "unopened"
    if role == "train_core":
        contract["train_core"]["annotation"]["path"] = str(forbidden / "train_core_coco.json")
    elif role == "pretrained":
        contract["pretrained"]["path"] = str(forbidden / "ResNet18_vd_pretrained_from_paddle.pth")
    elif role == "development":
        contract["development_binding"]["image_root"] = str(forbidden)
    elif role == "source":
        contract["code_files"]["src/file.py"]["path"] = str(forbidden / "file.py")
    elif role == "output":
        contract["output_dir"] = str(forbidden)
    else:
        contract["qualification_reference"]["path"] = str(forbidden / "qualification.json")
    monkeypatch.setattr(control, "file_reference", lambda *a, **k: pytest.fail("forbidden role was read"))
    with pytest.raises(control.ControlWorkerError):
        worker.validate_replication_worker_contract(contract)


@pytest.mark.parametrize("mutation", ["output_collision", "nested_output", "invocation_collision",
                                      "logical_run_collision", "different_replay_run", "source_digest", "formal_order"])
def test_freeze_rejects_identity_or_order_collisions(definition, mutation):
    contract = select_contract(definition)
    freeze = copy.deepcopy(definition)
    left = freeze["cells"]["s1_r640"]["invocations"]
    right = freeze["cells"]["s2_r640"]["invocations"]
    if mutation == "output_collision":
        right["capacity"]["output_dir"] = left["capacity"]["output_dir"]
    elif mutation == "nested_output":
        right["capacity"]["output_dir"] = left["capacity"]["output_dir"] + "/child"
    elif mutation == "invocation_collision":
        right["capacity"]["invocation_id"] = left["capacity"]["invocation_id"]
    elif mutation == "logical_run_collision":
        right["control30"]["run_id"] = left["control30"]["run_id"]
    elif mutation == "different_replay_run":
        right["smoke_replay"]["run_id"] = "different-replay"
    elif mutation == "source_digest":
        freeze["source"]["inventory_sha256"] = "c" * 64
    else:
        freeze["formal_order"] = list(reversed(freeze["formal_order"]))
    with pytest.raises(worker.ReplicationWorkerError):
        worker.validate_replication_freeze(freeze, contract)


@pytest.mark.parametrize("cell", worker.CELLS)
@pytest.mark.parametrize("stage", ["smoke", "smoke_replay", "control30"])
def test_stage_gates_require_exact_dependencies(definition, cell, stage):
    contract = select_contract(definition, cell, stage)
    gate = make_gate(contract, definition)
    assert worker.validate_replication_stage_gate(gate, contract, definition, verify_files=False) == gate
    if stage == "control30":
        previous = worker.CELLS.index(cell)
        assert len(gate["prerequisites"]) == 12 + int(previous > 0)
        assert set(gate["smoke_comparisons"]) == set(worker.CELLS)
    bad = copy.deepcopy(gate)
    bad["prerequisites"].pop()
    with pytest.raises(worker.ReplicationWorkerError, match="dependency set"):
        worker.validate_replication_stage_gate(bad, contract, definition, verify_files=False)
    bad = copy.deepcopy(gate)
    bad["prerequisites"].append(copy.deepcopy(bad["prerequisites"][0]))
    with pytest.raises(worker.ReplicationWorkerError, match="duplicate"):
        worker.validate_replication_stage_gate(bad, contract, definition, verify_files=False)


def test_integrity_pass_without_scientific_eligibility_stops_before_reads(definition, monkeypatch):
    from sparse_rtdetr.baseline import training_v2b_replication_gate as gate
    monkeypatch.setattr(gate, "verify_replication_qualification_binding", lambda reference: {
        "kind": "v2b_seed_resolution_replication_qualification", "status": "PASS",
        "eligible_for_fixed_paired_seed1_and_seed2": False,
        "gpu_admission_granted": False, "requires_fresh_native_admission": True,
    })
    monkeypatch.setattr(worker, "_read_reference", lambda *a, **k: pytest.fail("unqualified freeze was read"))
    with pytest.raises(worker.ReplicationWorkerError, match="explicit replication eligibility"):
        worker.verify_replication_worker_authorities(select_contract(definition))


def test_failed_qualification_persists_stop_without_constructing_model(definition, tmp_path, monkeypatch):
    contract = select_contract(definition)
    reference = write_exclusive_json(tmp_path / "worker-contract.json", contract)
    monkeypatch.setattr(worker, "_integrations", lambda: (None, None, None))
    monkeypatch.setattr(control, "_environment", lambda c: {"CPU_fixture": True})
    monkeypatch.setattr(worker, "_qualified_document", lambda ref: worker._fail("scientific eligibility absent"))
    with pytest.raises(worker.ReplicationWorkerError, match="eligibility"):
        worker.run_replication_worker(contract, contract_reference=reference)
    report = json.loads((Path(contract["output_dir"]) / "worker-result.json").read_text())
    assert report["status"] == "STOP_NO_RETRY" and report["scientific_certified"] is False
    assert report["contract_reference"] == reference and report["stage"] == "smoke"
    assert report["guardian_clearance_required_after_worker_exit"] is True


def test_capacity_dispatch_authenticates_bytes_and_bypasses_old_worker(definition, tmp_path, monkeypatch):
    from sparse_rtdetr.baseline import training_v2b_replication_capacity as capacity
    contract = select_contract(definition, stage="capacity")
    reference = write_exclusive_json(tmp_path / "capacity-contract.json", contract)
    observed = []
    monkeypatch.setattr(capacity, "run_replication_capacity_worker",
                        lambda value, *, contract_reference: observed.append((value, contract_reference)) or {"dispatch": True})
    monkeypatch.setattr(worker, "verify_replication_worker_authorities",
                        lambda *a: pytest.fail("capacity must own its one full validation"))
    monkeypatch.setattr(control, "validate_worker_contract", lambda *a, **k: pytest.fail("historical validator used"))
    assert worker.run_replication_worker(contract, contract_reference=reference) == {"dispatch": True}
    assert observed == [(contract, reference)] and not Path(contract["output_dir"]).exists()
    bad = {**reference, "sha256": "b" * 64}
    with pytest.raises(control.ControlWorkerError, match="reference mismatch"):
        worker.run_replication_worker(contract, contract_reference=bad)
    assert len(observed) == 1


def completion_fixture(tmp_path, contract):
    output = Path(contract["output_dir"])
    native = output / "native-hardware"
    native.mkdir(parents=True)
    binding = {"run_id": contract["run_id"], "config": contract["config"], "code": contract["code_files"]}
    binding["binding_sha256"] = canonical_sha256(binding)
    binding_ref = write_exclusive_json(output / "run-binding.json", binding)
    contract_ref = write_exclusive_json(tmp_path / "source-contract.json", contract)
    executable = ref(tmp_path / "python")
    owner = {"pid": 12001, "start_ticks": 81, "readable": True, "executable": executable}
    guardian = {"pid": 12002, "start_ticks": 82, "readable": True, "executable": executable}
    scope, limit = ("train_core_30epoch", 43200) if contract["stage"] == "control30" else ("paired_smoke", 600)
    policy = {**contract["policy_bundle"], "authorized_scope": scope, "workload_deadline_seconds": limit,
              "minimum_loaded_clock_samples": 3}
    identity = {
        "run_id": contract["run_id"], "run_binding_sha256": binding["binding_sha256"],
        "policy_sha256": canonical_sha256(policy), "monitor_pid": guardian["pid"],
        "owner": owner, "guardian_identity": guardian, "gpu_uuid": contract["expected_gpu_uuid"],
    }
    leaves = {}
    for name in worker._MONITOR_LEAVES:
        path = native / (name + ".json")
        path.write_text("{}\n")
        leaves[name] = file_reference(path)
    final = {**identity, **leaves, "status": "PASS", "worker_exited": True, "sampled_clock_compliance": True}
    final_ref = write_exclusive_json(native / "monitor-final.json", final)
    monitor = {**final, "final_report_reference": final_ref}
    result = worker._new_report(contract, contract_ref)
    result.update(status="PASS", worker_pid=owner["pid"], binding_reference=binding_ref,
                  monitor_reference={**identity, "monitor_final_report": final_ref["path"], "worker_must_exit": True})
    result_ref = write_exclusive_json(output / "worker-result.json", result)
    launch_ref = write_exclusive_json(tmp_path / "launch.json",
                                     {"pid": owner["pid"], "owner": owner, "contract": contract_ref})
    completion = {
        "schema_version": 1, "kind": worker.COMPLETION_KIND, "status": "PASS",
        "campaign_id": contract["campaign_id"], "cell_id": contract["cell_id"], "stage": contract["stage"],
        "freeze_reference": contract["freeze_reference"], "contract_reference": contract_ref,
        "result_reference": result_ref, "launch_reference": launch_ref, "monitor": monitor,
        "elapsed_seconds": 1.5, "capacity_memory_audit_reference": None,
    }
    if contract["stage"] == "capacity":
        completion["capacity_memory_audit_reference"] = write_exclusive_json(tmp_path / "capacity-memory-audit.json", {
            "schema_version": 1, "kind": "v2b_seed_resolution_replication_capacity_memory_audit", "status": "PASS",
            "result_reference": result_ref, "monitor_final_report_reference": final_ref,
            "required_margin_mib": 3072, "minimum_native_free_mib": 3072, "minimum_conservative_free_mib": 4096,
        })
    return completion, result_ref, result


@pytest.mark.parametrize("stage", ["capacity", "smoke", "control30"])
def test_completion_authenticates_real_leaf_bytes_and_process_identities(definition, tmp_path, stage):
    contract = select_contract(definition, stage=stage)
    completion, result_ref, result = completion_fixture(tmp_path, contract)
    completion_ref = write_exclusive_json(tmp_path / "completion.json", completion)
    assert worker._verify_completion(completion_ref, result_ref, result, contract, contract, definition) == completion
    loaded, source = worker._successful_cell_source(
        result_ref, cell_id=contract["cell_id"], stage=stage, contract=contract, freeze=definition)
    assert loaded == result and source == contract


@pytest.mark.parametrize("mutation", ["missing_both_sha", "missing_both_guardian", "unreadable_owner",
                                      "no_start_ticks", "wrong_monitor_pid", "false_worker_exit",
                                      "wrong_policy", "changed_leaf", "changed_final", "wrong_gpu"])
def test_completion_refuses_identity_and_leaf_tampering(definition, tmp_path, mutation):
    contract = select_contract(definition)
    completion, result_ref, result = completion_fixture(tmp_path, contract)
    monitor, finish = completion["monitor"], result["monitor_reference"]
    if mutation == "missing_both_sha":
        del monitor["run_binding_sha256"], finish["run_binding_sha256"]
    elif mutation == "missing_both_guardian":
        del monitor["guardian_identity"], finish["guardian_identity"]
    elif mutation == "unreadable_owner":
        monitor["owner"]["readable"] = finish["owner"]["readable"] = False
    elif mutation == "no_start_ticks":
        del monitor["owner"]["start_ticks"]
    elif mutation == "wrong_monitor_pid":
        monitor["monitor_pid"] = finish["monitor_pid"] = True
    elif mutation == "false_worker_exit":
        monitor["worker_exited"] = False
    elif mutation == "wrong_policy":
        monitor["policy_sha256"] = finish["policy_sha256"] = "b" * 64
    elif mutation == "wrong_gpu":
        monitor["gpu_uuid"] = finish["gpu_uuid"] = "GPU-" + "0" * 36
    elif mutation == "changed_leaf":
        Path(monitor["samples"]["path"]).write_text('{"changed": true}\n')
    else:
        Path(monitor["final_report_reference"]["path"]).write_text('{"changed": true}\n')
    completion_ref = write_exclusive_json(tmp_path / "completion.json", completion)
    with pytest.raises((ValueError, RuntimeError)):
        worker._verify_completion(completion_ref, result_ref, result, contract, contract, definition)


@pytest.mark.parametrize("margin", [3071.999, -1, True])
def test_capacity_completion_requires_full_three_gibibyte_margin(definition, tmp_path, margin):
    contract = select_contract(definition, stage="capacity")
    completion, result_ref, result = completion_fixture(tmp_path, contract)
    audit = worker._read_reference(completion["capacity_memory_audit_reference"], "fixture")
    audit["minimum_conservative_free_mib"] = margin
    completion["capacity_memory_audit_reference"] = write_exclusive_json(tmp_path / "bad-capacity-audit.json", audit)
    completion_ref = write_exclusive_json(tmp_path / "completion.json", completion)
    with pytest.raises(worker.ReplicationWorkerError, match="margin"):
        worker._verify_completion(completion_ref, result_ref, result, contract, contract, definition)


def stage_fixture(contract, monkeypatch):
    events = []
    engine = SimpleNamespace(epoch=0, epoch_active=False, optimizer_updates=0, microsteps=0)
    components = SimpleNamespace(engine=engine, config=V2BConfig(**contract["config"]))
    for name in ("model", "ema", "postprocessor", "runtime", "optimizer", "scheduler", "warmup"):
        setattr(components, name, SimpleNamespace())
    loader = SimpleNamespace(batches_per_epoch=304, dataset=list(range(4869)),
                             state_dict=lambda: {"order_sha256": contract["epoch_order_sha256"][str(engine.epoch)]})
    def begin(epoch):
        assert epoch == engine.epoch + 1 and not engine.epoch_active
        engine.epoch, engine.epoch_active = epoch, True
        events.append(("begin", epoch))
    def finish():
        assert engine.optimizer_updates == engine.epoch * 304
        events.append(("finish", engine.epoch))
        engine.epoch_active = False
        return {"epoch": engine.epoch}
    def restore(path, sha):
        events.append(("restore", path))
        engine.epoch, engine.epoch_active, engine.optimizer_updates, engine.microsteps = 1, True, 2, 4
    session = SimpleNamespace(components=components, loader=loader, binding={"binding_sha256": "b" * 64},
                              begin_epoch=begin, finish_epoch=finish, restore=restore)
    def state(*args, **kwargs):
        return {"clocks": {"engine": dict(vars(engine))}, "rng": "fixture", "raw_bn": {}, "ema_bn": {}}
    def windows(session, monitor, output, **kwargs):
        assert kwargs["paired_index"] == {}
        assert kwargs["matched_index"] is None or isinstance(kwargs["matched_index"], dict)
        start = engine.optimizer_updates % 304
        events.append(("windows", engine.epoch, kwargs["limit"], kwargs["save_midpoint"], kwargs["snapshots"]))
        for index in range(start, start + kwargs["limit"]):
            engine.optimizer_updates += 1
            engine.microsteps += 2
            kwargs["receipts"].append({**ref(output / f"receipt-{engine.epoch}-{index}.json"),
                                       "epoch": engine.epoch, "logical_batch_index": index})
            if kwargs["replay_index"]:
                kwargs["replay_checks"].append({"status": "PASS"})
            if kwargs["save_midpoint"] and engine.optimizer_updates == 2:
                kwargs["checkpoints"]["window_2"] = {"fixture": 2}
    def checkpoint(session, output, name, monitor):
        events.append(("checkpoint", name))
        return {"fixture": name}
    def evaluate(components, **kwargs):
        assert kwargs["data_binding"] == contract["development_binding"]
        events.append(("evaluate", kwargs["logged_epoch"]))
        return {"logged_epoch": kwargs["logged_epoch"]}
    def preview(components, **kwargs):
        assert kwargs["data_binding"] == contract["development_binding"]
        events.append(("preview", engine.optimizer_updates))
        return {"scope": "development_preview_only"}
    monkeypatch.setattr(control, "capture_control_state", state)
    monkeypatch.setattr(control, "_run_active_windows", windows)
    monkeypatch.setattr(control, "_write_checkpoint", checkpoint)
    monkeypatch.setattr(control, "_publish_bn_snapshot", lambda *a, **k: {"fixture": "bn"})
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: events.append(("synchronize",)))
    monitor = SimpleNamespace(check=lambda **kwargs: events.append(("gate", kwargs["stage"])))
    return session, monitor, events, state, evaluate, preview


@pytest.mark.parametrize("cell", worker.CELLS)
def test_exact_thirty_epoch_cadence_and_update_totals(definition, tmp_path, monkeypatch, cell):
    contract = select_contract(definition, cell, "control30")
    output = Path(contract["output_dir"]); output.mkdir()
    report = worker._new_report(contract, ref(tmp_path / "contract.json"))
    session, monitor, events, state, evaluate, preview = stage_fixture(contract, monkeypatch)
    worker._execute_training_stage(contract, session, monitor, output, report,
        original=None, matched_index=None, replay_index={},
        evaluate_development=evaluate, preview_development=preview)
    assert [x[1] for x in events if x[0] == "begin"] == list(range(1, 31))
    assert [x[1] for x in events if x[0] == "evaluate"] == [10, 20, 30]
    windows = [x for x in events if x[0] == "windows"]
    assert len(windows) == 30 and all(x[2:] == (304, False, False) for x in windows)
    assert len(report["checkpoints"]) == 30 and len(report["receipts"]) == 9120
    assert session.components.engine.microsteps == 18240
    assert worker._final_ledger(contract, report, state()) == 9120
    assert len(report["receipts"]) * 16 == 145920


def test_smoke_and_fresh_midpoint_replay_execute_only_declared_windows(definition, tmp_path, monkeypatch):
    for stage in ("smoke", "smoke_replay"):
        contract = select_contract(definition, stage=stage)
        output = Path(contract["output_dir"]); output.mkdir()
        report = worker._new_report(contract, ref(tmp_path / "contract.json"))
        session, monitor, events, state, evaluate, preview = stage_fixture(contract, monkeypatch)
        original = None
        if stage == "smoke_replay":
            source = tmp_path / "source"; source.mkdir()
            checkpoint = source / "checkpoint-window-02.pt"; checkpoint.write_bytes(b"CPU fixture only")
            session.restore(str(checkpoint), "fixture")
            expected = write_exclusive_json(source / "midpoint.json", state())
            session.components.engine.epoch = 0
            session.components.engine.epoch_active = False
            session.components.engine.optimizer_updates = session.components.engine.microsteps = 0
            events.clear()
            original = {"checkpoints": {"window_2": {"checkpoint": file_reference(checkpoint), "state_reference": expected}}}
        worker._execute_training_stage(contract, session, monitor, output, report,
            original=original, matched_index=None, replay_index={} if original is None else {(1, 2): {}},
            evaluate_development=evaluate, preview_development=preview)
        assert len(report["receipts"]) == (4 if original is None else 2)
        assert [x[1] for x in events if x[0] == "preview"] == [4]
        assert ("checkpoint", "checkpoint-window-04") in events
        assert len(report["replay_checks"]) == (0 if original is None else 2)
        assert worker._final_ledger(contract, report, state()) == 4
        if original is not None:
            assert not any(x[0] == "begin" for x in events)
            assert report["restored_from"] == original["checkpoints"]["window_2"]


def test_development_state_mutation_stops_before_later_epochs(definition, tmp_path, monkeypatch):
    contract = select_contract(definition, stage="control30")
    output = Path(contract["output_dir"]); output.mkdir()
    report = worker._new_report(contract, ref(tmp_path / "contract.json"))
    session, monitor, events, state, evaluate, preview = stage_fixture(contract, monkeypatch)
    def mutating(*args, **kwargs):
        session.components.engine.microsteps += 1
        return {"logged_epoch": kwargs["logged_epoch"]}
    with pytest.raises(worker.ReplicationWorkerError, match="preserved complete training state"):
        worker._execute_training_stage(contract, session, monitor, output, report,
            original=None, matched_index=None, replay_index={},
            evaluate_development=mutating, preview_development=preview)
    assert session.components.engine.epoch == 10
    assert len(report["checkpoints"]) == 10 and len(report["receipts"]) == 3040


@pytest.mark.parametrize("field", ["epoch", "epoch_active", "optimizer_updates", "microsteps"])
def test_final_ledger_rejects_clock_drift(definition, tmp_path, monkeypatch, field):
    contract = select_contract(definition)
    output = Path(contract["output_dir"]); output.mkdir()
    report = worker._new_report(contract, ref(tmp_path / "contract.json"))
    session, monitor, events, state, evaluate, preview = stage_fixture(contract, monkeypatch)
    worker._execute_training_stage(contract, session, monitor, output, report,
        original=None, matched_index=None, replay_index={}, evaluate_development=evaluate, preview_development=preview)
    final = state()
    final["clocks"]["engine"][field] = False if field == "epoch_active" else 999
    with pytest.raises(worker.ReplicationWorkerError, match="clock"):
        worker._final_ledger(contract, report, final)


@pytest.mark.parametrize("case", ["invalid_sha", "wrong_sha", "forbidden_path", "unknown_override",
                                "duplicate_json", "nonfinite_json"])
def test_fresh_cli_refuses_bad_inputs_before_torch_or_model_import(tmp_path, case):
    cli = Path(worker.__file__).resolve().parents[3] / "tools/run_training_v2b_replication_worker.py"
    path = tmp_path / "contract.json"
    raw = b'{"schema_version":1}'
    if case == "duplicate_json":
        raw = b'{"schema_version":1,"schema_version":1}'
    if case == "nonfinite_json":
        raw = b'{"value":NaN}'
    path.write_bytes(raw)
    sha = hashlib.sha256(raw).hexdigest()
    args = ["--contract", str(path), "--contract-sha256", sha]
    if case == "invalid_sha":
        args[-1] = "short"
    elif case == "wrong_sha":
        args[-1] = "b" * 64
    elif case == "forbidden_path":
        args[1] = str(tmp_path / "confirmatory" / "never-open.json")
    elif case == "unknown_override":
        args += ["--seed", "2"]
    wrapper = (
        "import runpy,sys\n"
        "class Block:\n"
        " def find_spec(self,name,path=None,target=None):\n"
        "  if name=='torch' or name.startswith('sparse_rtdetr'):\n"
        "   raise AssertionError('FORBIDDEN_IMPORT_REACHED')\n"
        "sys.meta_path.insert(0,Block())\n"
        "sys.argv=sys.argv[1:]\n"
        "runpy.run_path(sys.argv[0],run_name='__main__')\n"
    )
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-B", "-c", wrapper, str(cli), *args],
                            env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 2
    assert "FORBIDDEN_IMPORT_REACHED" not in result.stdout + result.stderr
    if case != "unknown_override":
        assert json.loads(result.stdout)["status"] == "STOP_NO_RETRY"



@pytest.mark.parametrize("failure", [False, True])
def test_public_worker_connects_admission_before_runtime_and_preserves_stop_evidence(
        definition, tmp_path, monkeypatch, failure):
    contract = select_contract(definition)
    contract_reference = write_exclusive_json(tmp_path / "live-contract.json", contract)
    session, unused_monitor, events, state, evaluate, preview = stage_fixture(contract, monkeypatch)
    session.components.initialization = {"cpu_fixture_initialization": True}
    runtime = SimpleNamespace(identity={"cpu_fixture_runtime": True})
    binding = {"run_id": contract["run_id"], "code": contract["code_files"], "config": contract["config"],
               "initial_weights": {"fixture": True}, "data": {"fixture": True}}
    binding["binding_sha256"] = canonical_sha256(binding)
    def authorize(value):
        events.append(("qualified",))
        return {"freeze": definition}
    def build(config, **kwargs):
        events.append(("construct", "runtime" in kwargs))
        return session.components
    def prepare(**kwargs):
        assert ("monitor_start",) in events and ("qualified",) in events
        events.append(("prepare",))
        if failure:
            raise RuntimeError("CPU fixture runtime failure")
        return runtime
    class Monitor:
        def __init__(self, actual_binding, policy, output, scope, limit):
            assert actual_binding == binding and policy == contract["policy_bundle"]
            assert (scope, limit) == ("paired_smoke", 600)
        def start(self):
            events.append(("monitor_start",))
        def admission(self):
            return "CPU fixture capability"
        def check(self, **kwargs):
            events.append(("gate", kwargs["stage"]))
        def finish(self):
            events.append(("monitor_finish",))
            return {"worker_must_exit": True, "CPU_fixture_only": True}
        def abort(self, reason):
            events.append(("monitor_abort", reason))
    def connect(components, loader, actual_binding):
        session.binding = actual_binding
        return session
    monkeypatch.setattr(worker, "_integrations", lambda: (Monitor, evaluate, preview))
    monkeypatch.setattr(control, "_environment", lambda value: {"CPU_fixture": True})
    monkeypatch.setattr(worker, "verify_replication_worker_authorities", authorize)
    monkeypatch.setattr(worker, "build_v2b_components", build)
    monkeypatch.setattr(worker, "build_train_core_loader", lambda *a, **k: session.loader)
    monkeypatch.setattr(worker, "build_train_core_run_binding", lambda *a, **k: binding)
    monkeypatch.setattr(worker, "_common_identity", lambda *a: {"CPU_fixture": True})
    monkeypatch.setattr(worker, "prepare_runtime", prepare)
    monkeypatch.setattr(worker, "V2BTrainingSession", connect)
    if failure:
        with pytest.raises(RuntimeError, match="fixture runtime failure"):
            worker.run_replication_worker(contract, contract_reference=contract_reference)
        saved = json.loads((Path(contract["output_dir"]) / "worker-result.json").read_text())
        assert saved["status"] == "STOP_NO_RETRY"
        assert len([x for x in events if x[0] == "monitor_abort"]) == 1
        assert not any(x[0] in {"windows", "monitor_finish"} for x in events)
    else:
        result = worker.run_replication_worker(contract, contract_reference=contract_reference)
        assert result["status"] == "PASS" and result["receipt_count"] == 4
        assert result["logical_samples_committed"] == result["logical_samples_executed_this_invocation"] == 64
        assert events.index(("qualified",)) < events.index(("construct", False)) < events.index(("monitor_start",))
        assert events.index(("monitor_start",)) < events.index(("prepare",)) < events.index(("construct", True))
        assert events.count(("monitor_finish",)) == 1
        assert not any(x[0] == "monitor_abort" for x in events)
        assert result["guardian_clearance_required_after_worker_exit"] is True
        assert result["scientific_certified"] is False


def comparison_fixture(definition, tmp_path, *, mutation=None):
    freeze_ref = write_exclusive_json(tmp_path / "frozen-package.json", definition)
    original_contract = select_contract(definition)
    replay_contract = select_contract(definition, stage="smoke_replay")
    original_contract["freeze_reference"] = replay_contract["freeze_reference"] = freeze_ref
    binding = {"run_id": original_contract["run_id"], "config": original_contract["config"],
               "code": original_contract["code_files"]}
    binding["binding_sha256"] = canonical_sha256(binding)
    initial = {
        "rng": {"CPU_fixture": "rng0"}, "raw_bn": {}, "ema_bn": {},
        "raw_state_sha256": "a" * 64, "raw_parameters": {"CPU_fixture": True},
        "ema_state_sha256": "b" * 64, "optimizer_state_sha256": "c" * 64,
        "model_training": {"": True}, "ema_training": {"": False},
    }
    def full_state(updates):
        return {**copy.deepcopy(initial), "clocks": {"engine": {
            "epoch": 1, "epoch_active": True, "optimizer_updates": updates, "microsteps": updates * 2}}}
    def checkpoint(output, name, updates):
        path = output / (name + ".pt")
        path.write_bytes(b"CPU checkpoint comparator call fixture only")
        return {"checkpoint": file_reference(path),
                "state_reference": write_exclusive_json(output / (name + "-state.json"), full_state(updates))}
    def make_preview(result, output, is_replay):
        empty = write_exclusive_json(output / "empty-metrics.json", {"CPU_fixture": True})
        prediction = write_exclusive_json(output / "predictions.json", {"CPU_fixture_prediction": int(is_replay)})
        preview = {
            "role": "development", "scope": "development_preview_only", "logged_epoch": None,
            "full_development_evaluation": False, "scientific_comparison_eligible": False,
            "data": {"evaluated_image_count": 4, "image_receipts": [{"CPU_fixture": i} for i in range(4)]},
            "weights": {"kind": "ema", "ema_updates": 4, "state_sha256": ("d" if is_replay else "e") * 64},
            "data_binding_sha256": result["common_identity"]["development_binding_sha256"],
            "runtime": result["runtime"], "policy": {"CPU_fixture_policy": True},
            "prediction_artifact": prediction,
            "coco_secondary": {"tensor_artifact": empty, "axes": {"CPU_fixture_axes": True},
                               "metrics": {"AP": int(is_replay)}},
            "primary_legacy_formal_gt": {"algorithm_id": "CPU_fixture_algorithm",
                                        "result_artifact": empty, "metrics": {"AP": int(is_replay)}},
            "native_gate_callbacks": [{"phase": phase, "batch": 0} for phase in ("before_batch", "after_batch")],
        }
        preview["result_artifact"] = write_exclusive_json(output / "preview-result.json", preview)
        return preview
    results = []
    for is_replay, contract in ((False, original_contract), (True, replay_contract)):
        output = Path(contract["output_dir"]); output.mkdir()
        if is_replay:
            contract["replay_source"] = results[0][1]
            if mutation == "replay_source":
                contract["replay_source"] = ref(tmp_path / "other-smoke.json")
        contract_ref = write_exclusive_json(tmp_path / ("replay-contract.json" if is_replay else "original-contract.json"), contract)
        result = worker._new_report(contract, contract_ref)
        result.update(status="PASS", worker_pid=17001 + int(is_replay),
                      common_identity={"development_binding_sha256": "b" * 64},
                      initialization={"CPU_fixture_initialization": True}, runtime={"CPU_fixture_runtime": True},
                      replay_tolerances=copy.deepcopy(control.REPLAY_TOLERANCES))
        if is_replay and mutation == "same_pid":
            result["worker_pid"] = 17001
        if is_replay and mutation == "tolerance":
            result["replay_tolerances"]["logical_loss_relative"] *= 2
        result["binding_reference"] = write_exclusive_json(output / "run-binding.json", binding)
        result["initial_state_reference"] = write_exclusive_json(output / "initial-state.json", initial)
        if not is_replay:
            result["checkpoints"]["window_2"] = checkpoint(output, "checkpoint-window-02", 2)
        else:
            result["restored_from"] = results[0][0]["checkpoints"]["window_2"]
            restored = full_state(2)
            if mutation == "restore":
                restored["rng"] = {"changed": True}
            result["restore_boundary_reference"] = write_exclusive_json(output / "restored.json", restored)
        for index in (range(2, 4) if is_replay else range(4)):
            values = torch.ones(2, dtype=torch.float64)
            if is_replay and mutation == "parameters":
                values += .001
            parameters = control._publish_tensor_evidence(
                output / f"parameters-{index}.pt", {"schema_version": 1, "parameters": {"weight": values}})
            window = {
                "epoch": 1, "optimizer_updates": index + 1, "microsteps": 2 * (index + 1),
                "logical_batch_size": 16, "target_total": 3, "denominator": 3,
                "microbatch_target_counts": [1, 2], "coefficients": [1 / 3, 2 / 3],
                "dn_num_groups": [100, 50], "dn_query_counts": [200, 200],
                "phase": "ready", "loss": 1.,
            }
            observed_state = full_state(index + 1)
            if is_replay and mutation == "rng":
                observed_state["rng"] = {"changed": True}
            if is_replay and mutation == "dn":
                window["dn_num_groups"][0] = 50
            if is_replay and mutation == "loss":
                window["loss"] += .01
            record = {
                "run_binding_sha256": binding["binding_sha256"],
                "input": {"CPU_fixture_logical_batch": index, "samples": list(range(16))},
                "window": window, "state": observed_state, "forward_observation": {"physical_forwards": 2},
                "raw_parameters_reference": parameters,
            }
            record_ref = write_exclusive_json(output / f"receipt-{index}.json", record)
            result["receipts"].append({**record_ref, "epoch": 1, "logical_batch_index": index})
        final = full_state(4)
        if is_replay and mutation == "final_clock":
            final["clocks"]["engine"]["microsteps"] = 7
        result["final_state_reference"] = write_exclusive_json(output / "final-state.json", final)
        result["checkpoints"]["window_4"] = checkpoint(output, "checkpoint-window-04", 4)
        result["evaluations"] = [make_preview(result, output, is_replay)]
        result_ref = write_exclusive_json(output / "worker-result.json", result)
        results.append((result, result_ref))
    return results[0][1], results[1][1], freeze_ref


def test_comparator_authenticates_replay_records_and_keeps_preview_numerics_observational(
        definition, tmp_path, monkeypatch):
    original, replay, freeze_ref = comparison_fixture(definition, tmp_path)
    calls = []
    def continuation(left, right, *, binding):
        calls.append((left, right, binding))
        return {"status": "PASS", "CPU_fixture_continuation": True}
    monkeypatch.setattr(control, "compare_continuation_checkpoints", continuation)
    result = worker.compare_replication_smoke(original, replay, freeze_reference=freeze_ref)
    assert result["status"] == "PASS", result
    assert result["restore_boundary_exact"] is True and len(result["windows"]) == 2
    assert len(calls) == 1 and all(Path(ref["path"]).name == "checkpoint-window-04.pt" for ref in calls[0][:2])
    assert result["replay_tolerances"] == control.REPLAY_TOLERANCES
    assert result["development_preview"]["prediction_bytes_exact"] is False
    assert result["development_preview"]["primary_metrics_exact"] is False
    assert result["development_preview"]["numerical_equalities_are_observations_not_new_replay_tolerances"] is True
    assert result["scientific_certified"] is False


@pytest.mark.parametrize("mutation", ["restore", "rng", "dn", "loss", "parameters", "same_pid",
                                      "tolerance", "final_clock", "replay_source", "continuation"])
def test_smoke_comparator_rejects_checkpoint_replay_and_numerical_drift(
        definition, tmp_path, monkeypatch, mutation):
    original, replay, freeze_ref = comparison_fixture(definition, tmp_path, mutation=mutation)
    def continuation(*args, **kwargs):
        if mutation == "continuation":
            raise control.ControlWorkerError("CPU fixture continuation moment mismatch")
        return {"status": "PASS"}
    monkeypatch.setattr(control, "compare_continuation_checkpoints", continuation)
    result = worker.compare_replication_smoke(original, replay, freeze_reference=freeze_ref)
    assert result["status"] == "STOP_NO_RETRY" and result["scientific_certified"] is False
    assert result["failure_type"] in {"ReplicationWorkerError", "ControlWorkerError"}
