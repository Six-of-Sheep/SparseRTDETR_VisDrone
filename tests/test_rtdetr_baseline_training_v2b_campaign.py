"""CPU-only checks of irreversible campaign transitions and frozen inputs."""
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from sparse_rtdetr.baseline import training_v2b_campaign as campaign
from sparse_rtdetr.baseline.training_v2b import V2BConfig


def _fixture(tmp_path):
    value = {"schema_version": 1, "kind": "v2b_640_paired_campaign", "campaign_id": "unit-campaign",
             "output_root": str(tmp_path), "source": {"repo_root": str(tmp_path), "code_files": {}},
             "common_config": asdict(V2BConfig(sampling_backend="deterministic_gather")),
             "arms": {"A": {"physical_batch_size": 16, "accumulation_steps": 1},
                      "B": {"physical_batch_size": 8, "accumulation_steps": 2}},
             "num_workers": 2, "prefetch_factor": 2, "torch_version": "fixture",
             "train_core": {}, "pretrained": {}, "development_binding": {},
             "epoch_order_sha256": {}, "policy_bundle": {}, "expected_gpu_uuid": "fixture"}
    ref = campaign._write_json(tmp_path / "campaign.json", value)
    return value, ref


def test_smoke_replay_keeps_run_identity_with_new_output(tmp_path):
    value, _ = _fixture(tmp_path)
    first = campaign.worker_contract(value, stage="smoke", arm="A", output_dir=tmp_path / "first")
    replay = campaign.worker_contract(value, stage="smoke_replay", arm="A", output_dir=tmp_path / "replay")
    formal = campaign.worker_contract(value, stage="control30", arm="A", output_dir=tmp_path / "formal")
    assert first["run_id"] == replay["run_id"]
    assert first["output_dir"] != replay["output_dir"]
    assert formal["run_id"] != first["run_id"]


def test_only_physical_batch_and_accumulation_change_between_arms(tmp_path):
    value, _ = _fixture(tmp_path)
    a = campaign.worker_contract(value, stage="control30", arm="A", output_dir=tmp_path / "a")
    b = campaign.worker_contract(value, stage="control30", arm="B", output_dir=tmp_path / "b")
    differences = {key for key in a["config"] if a["config"][key] != b["config"][key]}
    assert differences == {"physical_batch_size", "accumulation_steps"}
    for item in (a, b):
        assert item["config"]["physical_batch_size"] * item["config"]["accumulation_steps"] == 16
    before = campaign.scientific_template(b)
    b["paired_reference"] = {"fresh_result": "A"}
    assert campaign.scientific_template(b) == before
    b["config"]["learning_rate"] *= 2
    assert campaign.scientific_template(b) != before


def test_bound_file_mutation_is_rejected(tmp_path):
    path = tmp_path / "evidence.json"
    path.write_bytes(b"original")
    ref = campaign.file_reference(path)
    path.write_bytes(b"changed")
    with pytest.raises(campaign.CampaignError, match="bound file changed"):
        campaign.read_reference(ref)


def test_native_campaign_cannot_launch_smoke_or_formal_workers(tmp_path, monkeypatch):
    value, _ = _fixture(tmp_path)
    value["common_config"]["sampling_backend"] = "native"
    reference = campaign._write_json(tmp_path / "native-campaign.json", value)
    monkeypatch.setattr(campaign, "_run_worker", lambda *a: pytest.fail("native campaign launched a worker"))
    monkeypatch.setattr(campaign, "validate_frozen_source", lambda *a: pytest.fail("native campaign passed entry gate"))
    with pytest.raises(campaign.CampaignError, match="native is diagnostic only"):
        campaign.run_campaign(reference)
    assert not (tmp_path / "execution").exists()


@pytest.mark.parametrize("role", ["test", "confirmatory", "confirmatory_coco.json"])
def test_protected_data_path_refused_before_read(tmp_path, role):
    with pytest.raises(campaign.CampaignError, match="authorized role"):
        campaign._data_path(str(tmp_path / role / "train_core_coco.json"), expected_name="train_core_coco.json")


def test_frozen_source_detects_post_commit_change(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "a.py").write_text("before\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "a.py"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=CPU Test", "-c", "user.email=cpu@invalid",
                    "commit", "-qm", "fixture"], check=True)
    snap = campaign.source_snapshot(tmp_path)
    (tmp_path / "a.py").write_text("after\n")
    with pytest.raises(campaign.CampaignError):
        campaign.validate_frozen_source(snap)


@pytest.mark.parametrize("fail_at", ["smoke-A", "smoke_replay-A", "smoke-B", "smoke_replay-B", "control30-A"])
def test_failure_never_starts_later_worker(tmp_path, monkeypatch, fail_at):
    from sparse_rtdetr.baseline import training_v2b_control as control
    value, reference = _fixture(tmp_path)
    calls = []
    def worker(_, contract, contract_ref, execution):
        label = contract["stage"] + "-" + contract["arm"]
        calls.append(label)
        if label == fail_at:
            raise campaign.CampaignError("deliberate CPU transition failure")
        return {"result_reference": {"synthetic": label}, "monitor": {"status": "PASS"}}
    monkeypatch.setattr(campaign, "validate_frozen_source", lambda source: None)
    monkeypatch.setattr(campaign, "_run_worker", worker)
    monkeypatch.setattr(control, "compare_smoke_outputs", lambda *paths: {"status": "PASS"})
    with pytest.raises(campaign.CampaignError):
        campaign.run_campaign(reference)
    sequence = ["smoke-A", "smoke_replay-A", "smoke-B", "smoke_replay-B", "control30-A", "control30-B"]
    assert calls == sequence[:sequence.index(fail_at) + 1]
    result = json.loads((tmp_path / "execution/campaign-result.json").read_bytes())
    assert result["status"] == "STOP_NO_RETRY"
    assert result["continue_B_or_splice_comparison_allowed"] is False


def test_smoke_mismatch_prevents_formal_freeze_and_training(tmp_path, monkeypatch):
    from sparse_rtdetr.baseline import training_v2b_control as control
    value, reference = _fixture(tmp_path)
    calls = []
    def worker(_, contract, *args):
        calls.append(contract["stage"])
        return {"result_reference": {"synthetic": True}}
    monkeypatch.setattr(campaign, "validate_frozen_source", lambda source: None)
    monkeypatch.setattr(campaign, "_run_worker", worker)
    monkeypatch.setattr(control, "compare_smoke_outputs", lambda *paths: {"status": "STOP_NO_RETRY"})
    with pytest.raises(campaign.CampaignError, match="paired smoke"):
        campaign.run_campaign(reference)
    assert "control30" not in calls
    assert not (tmp_path / "execution/formal-freeze.json").exists()


def test_second_invocation_cannot_overwrite_existing_scene(tmp_path, monkeypatch):
    value, reference = _fixture(tmp_path)
    (tmp_path / "execution").mkdir()
    sentinel = tmp_path / "execution/original"
    sentinel.write_text("preserve")
    monkeypatch.setattr(campaign, "validate_frozen_source", lambda source: None)
    with pytest.raises(FileExistsError):
        campaign.run_campaign(reference)
    assert sentinel.read_text() == "preserve"


def test_change_after_formal_a_prevents_b(tmp_path, monkeypatch):
    from sparse_rtdetr.baseline import training_v2b_control as control
    _, reference = _fixture(tmp_path)
    changed = False
    calls = []
    def validate(source):
        if changed:
            raise campaign.CampaignError("source changed after formal A")
    def worker(_, contract, *args):
        nonlocal changed
        calls.append((contract["stage"], contract["arm"]))
        if contract["stage"] == "control30":
            changed = True
        return {"result_reference": {"synthetic": True}}
    monkeypatch.setattr(campaign, "validate_frozen_source", validate)
    monkeypatch.setattr(campaign, "_run_worker", worker)
    monkeypatch.setattr(control, "compare_smoke_outputs", lambda *paths: {"status": "PASS"})
    with pytest.raises(campaign.CampaignError, match="source changed"):
        campaign.run_campaign(reference)
    assert ("control30", "B") not in calls
    result = json.loads((tmp_path / "execution/campaign-result.json").read_bytes())
    assert result["formal_started"] is True


_CPU_HEARTBEAT = r'''
import json, os, sys, time
from pathlib import Path
from sparse_rtdetr.baseline.training_v2b_hardware import _process_identity
root, base = Path(sys.argv[1]), json.loads(sys.argv[2])
base["guardian_identity"] = _process_identity(os.getpid())
fd = os.open(root / "guardian-heartbeat.jsonl", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
while True:
    stamp = time.monotonic_ns()
    row = dict(base, schema_version=1, monotonic_ns=stamp, last_clock_started_ns=stamp,
               last_health_started_ns=stamp, clock_samples=3, health_observed=True, phase="running")
    os.write(fd, (json.dumps(row) + "\n").encode())
    time.sleep(.025)
'''


@pytest.fixture
def supervised_cpu_worker(tmp_path):
    """Three real owned CPU processes; no native GPU query or setter exists here."""
    from sparse_rtdetr.baseline.training_v2b_hardware import _process_identity
    worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    base = {"run_id": "cpu-only", "gpu_uuid": "CPU-fixture-no-GPU", "policy_sha256": "a" * 64,
            "run_binding_sha256": "b" * 64, "owner": _process_identity(worker.pid)}
    guardian = subprocess.Popen([sys.executable, "-c", _CPU_HEARTBEAT, str(tmp_path), json.dumps(base)])
    supervisor = None
    try:
        heartbeat = tmp_path / "guardian-heartbeat.jsonl"
        end = time.monotonic() + 5
        first = None
        while time.monotonic() < end:
            if heartbeat.exists():
                raw = heartbeat.read_bytes()
                if b"\n" in raw:
                    first = json.loads(raw.split(b"\n", 1)[0])
                    break
            time.sleep(.01)
        assert first is not None, "CPU heartbeat child did not start"
        descriptor = dict(base, guardian_identity=first["guardian_identity"], monitor_pid=guardian.pid,
                          heartbeat_path=str(heartbeat), heartbeat_period_seconds=.1,
                          heartbeat_max_gap_seconds=1., clock_max_gap_seconds=1., health_max_gap_seconds=2.)
        campaign._write_json(tmp_path / "admission-start.json", descriptor)
        supervisor = campaign.GuardianSupervisor(worker.pid, tmp_path, run_id=base["run_id"],
                                                 gpu_uuid=base["gpu_uuid"], policy_sha256=base["policy_sha256"])
        supervisor.check()
        yield worker, guardian, unrelated, supervisor
    finally:
        if supervisor is not None:
            supervisor.close()
        for proc in (guardian, worker, unrelated):
            if proc.poll() is None:
                os.kill(proc.pid, signal.SIGCONT)
                proc.terminate()
            proc.wait(timeout=5)


@pytest.mark.parametrize("failure", ["SIGSTOP", "exit"])
def test_independent_controller_stops_only_owned_worker_when_guardian_fails(supervised_cpu_worker, failure):
    worker, guardian, unrelated, supervisor = supervised_cpu_worker
    if failure == "SIGSTOP":
        os.kill(guardian.pid, signal.SIGSTOP)
    else:
        guardian.terminate()
        guardian.wait(timeout=5)
    detected = None
    end = time.monotonic() + 2.5
    while time.monotonic() < end:
        try:
            supervisor.check()
        except campaign.CampaignError as exc:
            detected = str(exc)
            supervisor.stop_owned_worker()
            break
        time.sleep(.025)
    assert detected is not None
    assert worker.wait(timeout=5) != 0
    cleanup = supervisor.close_failed_guardian(grace_seconds=.05)
    assert cleanup["guardian_exited"] is True
    assert cleanup["owner_exit_confirmed_before_guardian_cleanup"] is True
    guardian.wait(timeout=5)
    assert unrelated.poll() is None
    if failure == "SIGSTOP":
        assert "gap" in detected
    else:
        assert "exited" in detected


def test_live_owner_prevents_guardian_cleanup(supervised_cpu_worker):
    worker, guardian, unrelated, supervisor = supervised_cpu_worker
    with pytest.raises(campaign.CampaignError, match="confirmed worker exit"):
        supervisor.close_failed_guardian(grace_seconds=.01)
    assert all(proc.poll() is None for proc in (worker, guardian, unrelated))


def test_heartbeat_partial_tail_does_not_refresh_liveness(supervised_cpu_worker):
    worker, guardian, unrelated, supervisor = supervised_cpu_worker
    os.kill(guardian.pid, signal.SIGSTOP)
    supervisor.check()
    before = supervisor.last_heartbeat_ns
    with open(supervisor.root / "guardian-heartbeat.jsonl", "ab") as stream:
        stream.write(b'{"incomplete":')
    supervisor.check()
    assert supervisor.last_heartbeat_ns == before
    assert supervisor.buffer == b'{"incomplete":'
    assert worker.poll() is None and unrelated.poll() is None


def test_supervisor_constructor_failure_stops_its_cpu_worker(tmp_path, monkeypatch):
    from sparse_rtdetr.baseline import training_v2b_hardware as hw
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        monkeypatch.setattr(hw, "_process_identity", lambda pid: {"pid": pid, "readable": False})
        with pytest.raises(campaign.CampaignError, match="bind its worker"):
            campaign.GuardianSupervisor(proc.pid, tmp_path, run_id="cpu", gpu_uuid="fixture", policy_sha256="fixture")
        assert proc.wait(timeout=5) != 0
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=5)


def test_bootstrap_error_cannot_leave_a_launched_worker_running(tmp_path, monkeypatch):
    from sparse_rtdetr.baseline import training_v2b_admission as admission
    value, _ = _fixture(tmp_path)
    scripts = tmp_path / "tools"
    scripts.mkdir()
    (scripts / "run_training_v2b_control.py").write_text("import time\ntime.sleep(60)\n")
    contract = campaign.worker_contract(value, stage="smoke", arm="A", output_dir=tmp_path / "output")
    contract_ref = campaign._write_json(tmp_path / "worker-contract.json", contract)
    monkeypatch.setattr(admission, "_validate_policy", lambda *args: {"policy_sha256": "a" * 64})
    def broken(*args, **kwargs):
        raise campaign.CampaignError("CPU injection: supervisor bootstrap failed")
    monkeypatch.setattr(campaign, "GuardianSupervisor", broken)
    with pytest.raises(campaign.CampaignError, match="bootstrap failed"):
        campaign._run_worker(value, contract, contract_ref, tmp_path)
    exit_report = json.loads((tmp_path / "smoke-A.exit.json").read_bytes())
    assert exit_report["returncode"] != 0
    assert not Path("/proc", str(exit_report["pid"])).exists()
    stop = json.loads((tmp_path / "smoke-A.supervisor-stop.json").read_bytes())
    assert stop["only_owned_campaign_processes_signalled"] is True
    assert stop["pidfd_supervision_established"] is False


@pytest.mark.parametrize("failure", ["worker_exit", "monitor_final"])
def test_post_worker_failure_also_cleans_paused_guardian(tmp_path, monkeypatch, supervised_cpu_worker, failure):
    from sparse_rtdetr.baseline import training_v2b_admission as admission
    worker, guardian, unrelated, supervisor = supervised_cpu_worker
    value, _ = _fixture(tmp_path)
    contract = campaign.worker_contract(value, stage="smoke", arm="A", output_dir=tmp_path)
    contract_ref = campaign._write_json(tmp_path / "worker-contract.json", contract)
    os.kill(guardian.pid, signal.SIGSTOP)
    worker.terminate()
    worker.wait(timeout=5)
    if failure == "monitor_final":
        # CPU fixture represents a worker that exited with a successful result;
        # the real independently spawned owner is already confirmed dead.
        worker.returncode = 0
        campaign._write_json(tmp_path / "worker-result.json", {
            "status": "PASS", "run_id": contract["run_id"], "contract_reference": contract_ref,
            "monitor_reference": {"CPU_fixture": True},
        })
    monkeypatch.setattr(campaign.subprocess, "Popen", lambda *args, **kwargs: worker)
    monkeypatch.setattr(campaign, "GuardianSupervisor", lambda *args, **kwargs: supervisor)
    monkeypatch.setattr(admission, "_validate_policy", lambda *args: {"policy_sha256": "a" * 64})
    def missing_final(*args, **kwargs):
        raise admission.MonitoredHardwareError("CPU injection: final monitor unavailable")
    monkeypatch.setattr(admission, "wait_for_monitored_finish", missing_final)
    cleanup = supervisor.close_failed_guardian
    monkeypatch.setattr(supervisor, "close_failed_guardian", lambda: cleanup(grace_seconds=.01))
    with pytest.raises((campaign.CampaignError, admission.MonitoredHardwareError)):
        campaign._run_worker(value, contract, contract_ref, tmp_path)
    assert guardian.wait(timeout=5) != 0
    assert unrelated.poll() is None
    stop = json.loads((tmp_path / "smoke-A.supervisor-stop.json").read_bytes())
    assert stop["guardian_cleanup"]["guardian_exited"] is True
    assert stop["guardian_cleanup"]["owner_exit_confirmed_before_guardian_cleanup"] is True


@pytest.fixture
def campaign_cli(tmp_path, monkeypatch):
    """CLI admission choices are checked before constructing a workload."""
    import importlib.util
    from sparse_rtdetr.baseline import training_v2b_admission as admission
    source = Path(__file__).resolve().parents[1] / "tools" / "run_training_v2b_campaign.py"
    spec = importlib.util.spec_from_file_location("unit_v2b_campaign_cli", source)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    auth = tmp_path / "auth.txt"
    auth.write_text("CPU fixture authorization")
    external = tmp_path / "external.json"
    external.write_text('{"CPU_fixture": true}\n')
    captured = {}
    def policy(*args, **kwargs):
        captured["policy"] = kwargs
        return {"CPU_fixture_policy": True}
    def prepare(**kwargs):
        captured["prepare"] = kwargs
        return {"campaign_reference": {"CPU_fixture": True}}
    monkeypatch.setattr(admission, "build_policy_bundle", policy)
    monkeypatch.setattr(campaign, "make_campaign", prepare)
    args = ["prepare", "--campaign-id", "cpu-fixture", "--output-root", str(tmp_path / "new"),
            "--input-spec", str(tmp_path / "input.json"), "--policy-evidence-dir", str(tmp_path),
            "--authorization", str(auth), "--authorization-sha256", campaign.file_reference(auth)["sha256"]]
    (tmp_path / "input.json").write_text("{}")
    return cli, args, external, captured


def test_cli_external_receipt_is_explicit_and_bound(campaign_cli):
    cli, args, external, captured = campaign_cli
    reference = campaign.file_reference(external)
    assert cli.main(args + ["--setter-mode", "external_admin_acknowledged",
                            "--external-clock-receipt", str(external),
                            "--external-clock-receipt-sha256", reference["sha256"]]) == 0
    assert captured["policy"]["setter_mode"] == "external_admin_acknowledged"
    assert captured["policy"]["external_clock_receipt"] == reference
    assert captured["prepare"]["policy_bundle"] == {"CPU_fixture_policy": True}
    assert captured["prepare"]["sampling_backend"] == "deterministic_gather"


@pytest.mark.parametrize("provided", ["neither", "path", "sha"])
def test_cli_missing_external_reference_stops_before_policy(campaign_cli, provided):
    cli, args, external, captured = campaign_cli
    extra = []
    if provided == "path":
        extra = ["--external-clock-receipt", str(external)]
    elif provided == "sha":
        extra = ["--external-clock-receipt-sha256", campaign.file_reference(external)["sha256"]]
    with pytest.raises(SystemExit) as failure:
        cli.main(args + ["--setter-mode", "external_admin_acknowledged"] + extra)
    assert failure.value.code == 2
    assert not captured


def test_cli_external_hash_mismatch_stops_before_policy(campaign_cli):
    cli, args, external, captured = campaign_cli
    with pytest.raises(campaign.CampaignError, match="receipt bytes differ"):
        cli.main(args + ["--setter-mode", "external_admin_acknowledged",
                        "--external-clock-receipt", str(external),
                        "--external-clock-receipt-sha256", "0" * 64])
    assert not captured


@pytest.mark.parametrize("mode", ["direct", "sudo_n"])
def test_cli_rejects_implicit_external_mode(campaign_cli, mode):
    cli, args, external, captured = campaign_cli
    with pytest.raises(SystemExit) as failure:
        cli.main(args + ["--setter-mode", mode, "--external-clock-receipt", str(external),
                        "--external-clock-receipt-sha256", campaign.file_reference(external)["sha256"]])
    assert failure.value.code == 2
    assert not captured
