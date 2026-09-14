from pathlib import Path
from types import SimpleNamespace
import copy
import json

import pytest

from sparse_rtdetr.baseline import training_v2b_replication_campaign as campaign


def test_replay_has_same_run_binding_identity_but_unique_invocation_and_directory(tmp_path):
    identities = campaign.invocation_plan("replication-fixture", tmp_path)
    directories, invocations, run_ids = [], [], []
    for cell in campaign.CELLS:
        rows = identities[cell]
        assert rows["smoke"]["run_id"] == rows["smoke_replay"]["run_id"]
        assert rows["capacity"]["run_id"] != rows["control30"]["run_id"]
        for stage in campaign.STAGES:
            directories.append(rows[stage]["output_dir"])
            invocations.append(rows[stage]["invocation_id"])
            run_ids.append(rows[stage]["run_id"])
            assert Path(rows[stage]["output_dir"]).is_relative_to(tmp_path/"execution")
    assert len(set(directories)) == len(set(invocations)) == 16
    assert len(set(run_ids)) == 12


@pytest.mark.parametrize("seed,size", [(1,640),(1,896),(2,640),(2,896)])
def test_fixed_cell_preserves_matched_training_semantics(seed, size):
    config = campaign.fixed_config(seed, size)
    assert config["seed"] == seed and config["input_size"] == size
    assert config["physical_batch_size"] == 8 and config["accumulation_steps"] == 2
    assert config["amp_dtype"] == "bfloat16" and config["bn_statistics"] == "train"
    assert config["pretrained_required"] is True and config["num_denoising"] == 100
    assert config["sampling_backend"] == "deterministic_gather"
    a = campaign.fixed_config(seed, 640)
    b = campaign.fixed_config(seed, 896)
    assert {k:v for k,v in a.items() if k != "input_size"} == {
        k:v for k,v in b.items() if k != "input_size"}


@pytest.mark.parametrize("seed,size", [(0,640),(3,896),(1,960),(2,1024),(1,800)])
def test_unapproved_seed_or_resolution_is_rejected(seed, size):
    with pytest.raises(campaign.CampaignError):
        campaign.fixed_config(seed, size)


def test_epoch_orders_use_real_one_based_epochs_and_preserve_seed_pairing():
    from sparse_rtdetr.baseline.training_v2b import logical_batch_indices
    from sparse_rtdetr.baseline.training_v2b_evidence import canonical_sha256
    a, b = campaign.epoch_orders(1), campaign.epoch_orders(2)
    assert set(a) == {str(e) for e in range(1,31)}
    assert all(a[key] != b[key] for key in a)
    for seed, orders in ((1,a),(2,b)):
        for epoch in (1,10,20,30):
            actual = [list(x) for x in logical_batch_indices(
                4869, seed=seed, epoch=epoch, logical_batch_size=16)]
            assert orders[str(epoch)] == canonical_sha256(actual)
            assert len(actual) == 304 and len({i for batch in actual for i in batch}) == 4864


def completed():
    return {(cell, stage): {} for cell in campaign.CELLS for stage in campaign.STAGES}


def test_resolution_smoke_and_replay_require_same_seed_original_640_smoke():
    rows = completed()
    expected = {("s2_r896","capacity"),("s2_r640","smoke")}
    assert set(campaign.prerequisite_keys("s2_r896","smoke",rows)) == expected
    assert set(campaign.prerequisite_keys("s2_r896","smoke_replay",rows)) == expected | {("s2_r896","smoke")}
    del rows[("s2_r640","smoke")]
    with pytest.raises(campaign.CampaignError):
        campaign.prerequisite_keys("s2_r896","smoke_replay",rows)


def test_formal_requires_all_four_capacity_and_replay_closures_and_immediate_predecessor():
    rows = completed()
    common = {(c,s) for c in campaign.CELLS for s in ("capacity","smoke","smoke_replay")}
    assert set(campaign.prerequisite_keys("s1_r640","control30",rows)) == common
    assert set(campaign.prerequisite_keys("s2_r896","control30",rows)) == common | {("s2_r640","control30")}
    del rows[("s1_r896","smoke_replay")]
    with pytest.raises(campaign.CampaignError):
        campaign.prerequisite_keys("s2_r896","control30",rows)


def test_missing_formal_predecessor_cannot_be_replaced_by_another_seed_result():
    rows = completed()
    del rows[("s2_r640","control30")]
    with pytest.raises(campaign.CampaignError):
        campaign.prerequisite_keys("s2_r896","control30",rows)


def failure_harness(tmp_path, monkeypatch, *, constructor_fails=False):
    from sparse_rtdetr.baseline import training_v2b_admission as admission
    output = tmp_path/"run";(output/"execution").mkdir(parents=True)
    contract = {"cell_id":"s1_r640","stage":"smoke","run_id":"fixture-smoke",
                "invocation_id":"fixture-smoke","output_dir":str(output/"execution"/"s1_r640-smoke")}
    freeze = {"repo_root":str(tmp_path),"output_root":str(output),"source":{},
              "common":{"policy_bundle":{},"expected_gpu_uuid":"GPU-fixture"}}
    events = []
    class Child:
        pid = 123456
        returncode = 17
        def poll(self): return self.returncode
        def wait(self, timeout): events.append(("wait",timeout));return self.returncode
        def kill(self):events.append("kill");self.returncode=-9
    child = Child()
    monkeypatch.setattr(campaign.subprocess,"Popen",lambda *a,**kw:(events.append("spawn") or child))
    monkeypatch.setattr(admission,"_validate_policy",lambda *a:{"policy_sha256":"fixture-policy"})
    class Guardian:
        def __init__(self,*a,**kw):
            events.append("guardian")
            if constructor_fails: raise RuntimeError("owned guardian construction failed")
            self.owner={"pid":child.pid,"start_ticks":321}
        def stop_owned_worker(self):events.append("stop_owned")
        def close_failed_guardian(self):events.append("guardian_cleanup");return {"closed":True}
        def close(self):events.append("close")
    monkeypatch.setattr(campaign,"GuardianSupervisor",Guardian)
    return freeze,contract,child,events


def test_failed_worker_stops_once_preserves_exit_receipt_and_cleans_only_owned_guardian(tmp_path,monkeypatch):
    freeze,contract,child,events = failure_harness(tmp_path,monkeypatch)
    with pytest.raises(campaign.CampaignError,match="worker failed"):
        campaign._run_stage(freeze,{},contract,{"path":"/fixture","sha256":"f"*64,"size_bytes":1,"sha256_scope":"complete_file_bytes"})
    assert events.count("spawn") == events.count("stop_owned") == events.count("guardian_cleanup") == events.count("close") == 1
    root=Path(freeze["output_root"])/"execution"
    failure=json.loads((root/"s1_r640-smoke.controller-failure.json").read_text())
    exit_record=json.loads((root/"s1_r640-smoke.exit.json").read_text())
    assert failure["status"] == "STOP_NO_RETRY" and failure["only_owned_campaign_processes_signalled"] is True
    assert exit_record["returncode"] == 17 and exit_record["pid"] == child.pid


def test_guardian_construction_failure_can_only_kill_its_unreaped_direct_child(tmp_path,monkeypatch):
    freeze,contract,child,events=failure_harness(tmp_path,monkeypatch,constructor_fails=True)
    child.returncode=None
    with pytest.raises(RuntimeError,match="construction failed"):
        campaign._run_stage(freeze,{},contract,{"path":"/fixture","sha256":"f"*64,"size_bytes":1,"sha256_scope":"complete_file_bytes"})
    assert events.count("spawn") == events.count("kill") == 1
    assert "stop_owned" not in events and "guardian_cleanup" not in events
    assert any(isinstance(event,tuple) and event[0]=="wait" for event in events)


def test_gate_dependencies_cannot_use_future_formal_stages_as_a_substitute():
    rows={("s2_r896","control30"):{}}
    with pytest.raises(campaign.CampaignError):
        campaign.prerequisite_keys("s1_r640","control30",rows)


def terminal_success_harness(tmp_path, monkeypatch, *, mutation=None):
    """A synthetic final formal stage exercises the real completed-evidence verifier.

    Source/contract qualification is tested separately. No hardware capability or
    worker process is constructed here; all monitor leaves are explicit fixtures.
    """
    from sparse_rtdetr.baseline import training_v2b_admission as admission
    from sparse_rtdetr.baseline import training_v2b_replication_worker as worker
    from sparse_rtdetr.baseline.training_v2b_evidence import canonical_sha256
    output = tmp_path / "campaign"
    execution = output / "execution"
    execution.mkdir(parents=True)
    worker_output = execution / "s2_r896-control30"
    worker_output.mkdir()
    native = worker_output / "native-hardware"
    native.mkdir()
    source_path = tmp_path / "source.py"
    source_path.write_text("synthetic source fixture\n")
    binary = tmp_path / "synthetic-python.bin"
    binary.write_bytes(b"synthetic executable identity; never executed")
    code = {"source.py": campaign.file_reference(source_path)}
    config = {"seed": 2, "input_size": 896, "physical_batch_size": 8, "accumulation_steps": 2}
    run_id, invocation_id = "fixture-s2-r896-formal", "fixture-s2-r896-control30"
    gpu = "GPU-1faee6f0-1da7-4ede-2475-67a5a00274a8"
    policy_bundle = {"setter_mode": "external_admin_acknowledged", "synthetic_only": True}
    policy = {**policy_bundle, "authorized_scope": "train_core_30epoch",
              "workload_deadline_seconds": 43200, "minimum_loaded_clock_samples": 3}
    policy["policy_sha256"] = canonical_sha256(policy)
    owner = {"pid": 123456, "start_ticks": 101, "readable": True,
             "executable": campaign.file_reference(binary)}
    guardian_identity = {"pid": 123457, "start_ticks": 103, "readable": True,
                         "executable": campaign.file_reference(binary)}
    qualification = campaign._write_json(tmp_path / "qualification.json", {"synthetic_only": True})
    plan = campaign._write_json(tmp_path / "capacity-plan.json", {"synthetic_only": True})
    freeze = {
        "repo_root": str(tmp_path), "output_root": str(output), "source": {},
        "campaign_id": "fixture-campaign",
        "common": {"expected_gpu_uuid": gpu, "policy_bundle": policy_bundle},
        "cells": {"s2_r896": {"config": config, "capacity_plan": plan,
            "invocations": {"control30": {
                "run_id": run_id, "invocation_id": invocation_id, "output_dir": str(worker_output)}}}},
    }
    freeze_reference = campaign._write_json(output / "freeze.json", freeze)
    contract = {
        "cell_id": "s2_r896", "stage": "control30", "run_id": run_id, "invocation_id": invocation_id,
        "output_dir": str(worker_output), "freeze_reference": freeze_reference,
        "campaign_id": freeze["campaign_id"], "qualification_reference": qualification,
        "config": config, "capacity_plan": plan, "expected_gpu_uuid": gpu, "code_files": code,
        "policy_bundle": policy_bundle,
    }
    contract_reference = campaign._write_json(execution / "contract.json", contract)
    binding = {"run_id": run_id, "code": code, "config": config}
    if mutation == "binding_run":
        binding["run_id"] = "different-bound-run"
    binding["binding_sha256"] = canonical_sha256(binding)
    binding_reference = campaign._write_json(worker_output / "run-binding.json", binding)
    monitor = {
        "schema_version": 1, "status": "PASS", "worker_exited": True,
        "sampled_clock_compliance": True, "run_id": run_id, "gpu_uuid": gpu,
        "run_binding_sha256": binding["binding_sha256"], "policy_sha256": policy["policy_sha256"],
        "monitor_pid": guardian_identity["pid"], "owner": copy.deepcopy(owner),
        "guardian_identity": guardian_identity,
    }
    for field in ("samples", "setter_receipt", "startup_evidence", "guardian_heartbeat"):
        monitor[field] = campaign._write_json(native / (field + ".json"), {"synthetic_only": True})
    if mutation == "owner":
        monitor["owner"]["pid"] += 1000
    elif mutation == "run":
        monitor["run_id"] = "different-native-run"
    elif mutation == "gpu":
        monitor["gpu_uuid"] = "GPU-00000000-0000-0000-0000-000000000000"
    elif mutation == "policy":
        monitor["policy_sha256"] = "f" * 64
    elif mutation == "binding":
        monitor["run_binding_sha256"] = "f" * 64
    elif mutation == "clock":
        monitor["sampled_clock_compliance"] = False
    final_reference = campaign._write_json(native / "monitor-final.json", monitor)
    monitor_reference = {key: monitor[key] for key in (
        "run_id", "run_binding_sha256", "policy_sha256", "monitor_pid",
        "owner", "guardian_identity", "gpu_uuid")}
    monitor_reference.update(monitor_final_report=final_reference["path"], worker_must_exit=True)
    result = {
        "schema_version": 1, "kind": worker.RESULT_KIND if mutation != "kind" else "unrelated-worker-result",
        "status": "PASS", "cell_id": "s2_r896", "stage": "control30", "seed": 2, "input_size": 896,
        "campaign_id": freeze["campaign_id"], "run_id": run_id, "invocation_id": invocation_id,
        "freeze_reference": freeze_reference, "qualification_reference": qualification,
        "capacity_plan": plan, "worker_pid": owner["pid"], "contract_reference": contract_reference,
        "guardian_clearance_required_after_worker_exit": True,
        "scientific_certified": False, "automatic_retry_or_batch_fallback": False,
        "binding_reference": binding_reference, "monitor_reference": monitor_reference,
    }
    campaign._write_json(worker_output / "worker-result.json", result)
    # Keep real source-result and completion validators, including all native
    # file references and the actual wait_for_monitored_finish leaf checks.
    monkeypatch.setattr(worker, "validate_replication_worker_contract",
                        lambda value, **kwargs: copy.deepcopy(value))
    monkeypatch.setattr(worker, "validate_replication_freeze",
                        lambda value, contract: copy.deepcopy(value))
    monkeypatch.setattr(campaign, "validate_frozen_source", lambda source: None)
    monkeypatch.setattr(admission, "_validate_policy", lambda *args: copy.deepcopy(policy))
    events = []
    class Child:
        pid = owner["pid"]
        returncode = 0
        def poll(self): return self.returncode
        def wait(self, timeout): events.append("wait");return self.returncode
        def kill(self): raise AssertionError("no real child may be signalled in this fixture")
    child = Child()
    monkeypatch.setattr(campaign.subprocess, "Popen",
                        lambda *args, **kwargs: (events.append("spawn") or child))
    class Guardian:
        def __init__(self, *args, **kwargs): self.owner = copy.deepcopy(owner)
        def check(self): raise AssertionError("the synthetic child already exited")
        def stop_owned_worker(self): events.append("stop_owned")
        def close_failed_guardian(self): events.append("guardian_cleanup");return {"synthetic_only": True}
        def close(self): events.append("close")
    monkeypatch.setattr(campaign, "GuardianSupervisor", Guardian)
    return freeze, freeze_reference, contract, contract_reference, events


def test_last_formal_completion_is_published_after_real_worker_verifier_and_exit(tmp_path, monkeypatch):
    from unittest.mock import patch
    from sparse_rtdetr.baseline import training_v2b_replication_worker as worker
    freeze, freeze_ref, contract, contract_ref, events = terminal_success_harness(tmp_path, monkeypatch)
    with patch.object(worker, "_verify_completion", wraps=worker._verify_completion) as verifier:
        entry = campaign._run_stage(freeze, freeze_ref, contract, contract_ref)
    assert verifier.call_count == 1
    root = Path(freeze["output_root"]) / "execution"
    candidate = root / "s2_r896-control30.completion-candidate.json"
    public = root / "s2_r896-control30.completion.json"
    exit_path = root / "s2_r896-control30.exit.json"
    assert candidate.is_file() and public.is_file() and exit_path.is_file()
    assert candidate.read_bytes() == public.read_bytes()
    assert entry["completion_reference"] == campaign.file_reference(public)
    assert events == ["spawn", "close"]
    assert not (root / "s2_r896-control30.controller-failure.json").exists()


@pytest.mark.parametrize("mutation", ["owner", "run", "gpu", "policy", "binding", "kind", "binding_run", "clock"])
def test_last_formal_stage_refuses_cross_bound_native_or_worker_result(tmp_path, monkeypatch, mutation):
    from sparse_rtdetr.baseline.training_v2b_replication_worker import ReplicationWorkerError
    freeze, freeze_ref, contract, contract_ref, events = terminal_success_harness(
        tmp_path, monkeypatch, mutation=mutation)
    with pytest.raises((campaign.CampaignError, ReplicationWorkerError)):
        campaign._run_stage(freeze, freeze_ref, contract, contract_ref)
    root = Path(freeze["output_root"]) / "execution"
    assert not (root / "s2_r896-control30.completion.json").exists()
    assert (root / "s2_r896-control30.completion-candidate.json").is_file()
    failure = json.loads((root / "s2_r896-control30.controller-failure.json").read_text())
    assert failure["status"] == "STOP_NO_RETRY"
    assert events.count("spawn") == events.count("stop_owned") == events.count("close") == 1


def controller_failure_documents(freeze):
    root = Path(freeze["output_root"]) / "execution"
    failure = json.loads((root / "s1_r640-smoke.controller-failure.json").read_text())
    summary = json.loads((root / "s1_r640-smoke.cleanup-summary.json").read_text())
    rows = []
    for item in summary["cleanup_records"]:
        if item["reference"] is not None:
            raw = campaign.read_reference(item["reference"])
            rows.append(json.loads(raw))
    return root, failure, summary, rows


@pytest.mark.parametrize("broken", ["stop", "wait", "guardian_cleanup", "close"])
def test_cleanup_exception_keeps_first_worker_failure_and_persists_each_step(tmp_path, monkeypatch, broken):
    import subprocess
    freeze, contract, child, events = failure_harness(tmp_path, monkeypatch)
    base = campaign.GuardianSupervisor
    class FaultyGuardian(base):
        def stop_owned_worker(self):
            super().stop_owned_worker()
            if broken == "stop": raise OSError("synthetic stop failure")
        def close_failed_guardian(self):
            result = super().close_failed_guardian()
            if broken == "guardian_cleanup": raise OSError("synthetic guardian cleanup failure")
            return result
        def close(self):
            super().close()
            if broken == "close": raise OSError("synthetic supervisor close failure")
    monkeypatch.setattr(campaign, "GuardianSupervisor", FaultyGuardian)
    if broken == "wait":
        def wait(timeout):
            events.append(("wait", timeout))
            raise subprocess.TimeoutExpired("synthetic owned child", timeout)
        child.wait = wait
    reference = {"path": "/fixture", "sha256": "f"*64, "size_bytes": 1, "sha256_scope": "complete_file_bytes"}
    with pytest.raises(campaign.CampaignError, match="worker failed"):
        campaign._run_stage(freeze, {}, contract, reference)
    root, failure, summary, steps = controller_failure_documents(freeze)
    assert "worker failed" in failure["failure"]["message"]
    assert summary["first_failure"] == failure["failure"]
    bad_steps = [row for row in steps if row["status"] == "ERROR"]
    assert len(bad_steps) == 1
    assert "synthetic" in bad_steps[0]["error"]["message"]
    assert all(row["first_failure"] == failure["failure"] for row in steps)
    assert (root / "s1_r640-smoke.exit.json").is_file()
    assert events.count("spawn") == events.count("close") == 1


@pytest.mark.parametrize("broken", ["exit", "cleanup_receipt", "failure_receipt"])
def test_cleanup_evidence_write_error_never_masks_first_cause(tmp_path, monkeypatch, capsys, broken):
    freeze, contract, child, events = failure_harness(tmp_path, monkeypatch)
    original_write = campaign._write_json
    def write(path, value):
        if ((broken == "exit" and path.name.endswith(".exit.json"))
                or (broken == "cleanup_receipt" and ".cleanup-01-" in path.name)
                or (broken == "failure_receipt" and path.name.endswith(".controller-failure.json"))):
            raise OSError("synthetic " + broken + " publication failure")
        return original_write(path, value)
    monkeypatch.setattr(campaign, "_write_json", write)
    reference = {"path": "/fixture", "sha256": "f"*64, "size_bytes": 1, "sha256_scope": "complete_file_bytes"}
    with pytest.raises(campaign.CampaignError, match="worker failed"):
        campaign._run_stage(freeze, {}, contract, reference)
    root = Path(freeze["output_root"]) / "execution"
    summary = json.loads((root / "s1_r640-smoke.cleanup-summary.json").read_text())
    assert "worker failed" in summary["first_failure"]["message"]
    if broken == "exit":
        failed = [row for row in summary["cleanup_records"] if row["status"] == "ERROR"]
        assert [row["operation"] for row in failed] == ["publish_exit_receipt"]
    else:
        assert len(summary["evidence_errors"]) == 1
        fallback = json.loads(capsys.readouterr().err.splitlines()[-1])
        assert fallback["first_failure"] == summary["first_failure"]
        assert "synthetic" in fallback["evidence_error"]["error"]["message"]
    if broken != "failure_receipt":
        failure = json.loads((root / "s1_r640-smoke.controller-failure.json").read_text())
        assert failure["failure"] == summary["first_failure"]
    assert not (root / "s1_r640-smoke.completion.json").exists()


def test_successful_worker_close_error_cannot_publish_formal_completion(tmp_path, monkeypatch):
    freeze, freeze_ref, contract, contract_ref, events = terminal_success_harness(tmp_path, monkeypatch)
    base = campaign.GuardianSupervisor
    class FaultyGuardian(base):
        def close(self):
            super().close()
            raise OSError("synthetic final close failure")
    monkeypatch.setattr(campaign, "GuardianSupervisor", FaultyGuardian)
    with pytest.raises(OSError, match="synthetic final close failure"):
        campaign._run_stage(freeze, freeze_ref, contract, contract_ref)
    root = Path(freeze["output_root"]) / "execution"
    assert not (root / "s2_r896-control30.completion.json").exists()
    assert (root / "s2_r896-control30.exit.json").is_file()
    failure = json.loads((root / "s2_r896-control30.controller-failure.json").read_text())
    assert failure["failure"] == {"type": "OSError", "message": "synthetic final close failure"}
