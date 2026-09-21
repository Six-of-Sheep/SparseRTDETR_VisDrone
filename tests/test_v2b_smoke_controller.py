from __future__ import annotations

import json
import os
import stat
import sys
import time
from pathlib import Path

import pytest

from sparse_rtdetr.baseline import training_v2b_smoke_controller as controller


def _plan(tmp_path: Path, **overrides):
    values = dict(
        launch_id="test-launch",
        evidence_root=tmp_path / "evidence",
        repo_root=tmp_path,
        python_executable=Path(sys.executable),
        runner_argv=["-c", "pass"],
        tmux_executable=Path("/missing/tmux"),
        session_name="test-session",
        environment={"CUDA_VISIBLE_DEVICES": "", "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": "0"},
        ready_timeout=0.1,
        exit_timeout=0.1,
    )
    values.update(overrides)
    return controller.build_plan(**values)


def test_plan_has_explicit_argv_and_no_shell_string(tmp_path):
    plan = _plan(tmp_path)
    assert plan["argv"][0] == sys.executable
    assert plan["tmux_argv"][-1] == str(tmp_path)
    assert "shell" not in plan
    assert plan["environment"]["CUDA_VISIBLE_DEVICES"] == ""


def test_canonical_json_is_deterministic():
    value = {"b": 2, "a": [1, True]}
    assert controller.canonical_bytes(value) == controller.canonical_bytes({"a": [1, True], "b": 2})
    assert controller.canonical_sha256(value) == controller.canonical_sha256({"a": [1, True], "b": 2})


def test_invalid_tmux_executable_is_reported(tmp_path):
    result = controller.launch(_plan(tmp_path))
    assert result["status"] == "TMUX_CREATE_FAILED"
    assert (tmp_path / "evidence" / "controller" / "launch-evidence.json").is_file()
    assert (tmp_path / "evidence" / "handshake").is_dir()
    assert not (tmp_path / "evidence" / "runner").exists()


def test_invalid_cwd_is_reported_before_tmux(tmp_path):
    result = controller.launch(_plan(tmp_path, repo_root=tmp_path / "does-not-exist", tmux_executable=Path("/missing/tmux")))
    assert result["status"] == "INVALID_CWD"


def test_invalid_python_is_reported_before_tmux(tmp_path):
    result = controller.launch(_plan(tmp_path, python_executable=tmp_path / "missing-python", tmux_executable=Path("/missing/tmux")))
    assert result["status"] == "INVALID_PYTHON_EXECUTABLE"


@pytest.mark.parametrize(
    "ready,exit_code,expected",
    [
        (None, None, "READY_NOT_OBSERVED"),
        (None, 0, "RUNNER_EXITED_BEFORE_READY"),
        (None, 2, "RUNNER_EXITED_BEFORE_READY"),
        ({"event": "READY"}, 0, "COMPLETED"),
        ({"event": "READY"}, 2, "RUNNER_NONZERO_EXIT"),
        ({"event": "READY"}, None, "EXIT_TIMEOUT_AFTER_READY"),
    ],
)
def test_marker_classification(ready, exit_code, expected):
    exit_data = None if exit_code is None else {"exit_code": exit_code}
    assert controller._status_from_files(Path("."), ready, exit_data) == expected


def test_wrapper_contains_durable_markers_and_redirection(tmp_path):
    plan = _plan(tmp_path, tmux_executable=Path("tmux"))
    root = tmp_path / "evidence"
    text = controller._wrapper_text(plan, root / "runner-wrapper.sh", root / "stdout.log", root / "stderr.log")
    assert "started.json" in text
    assert "runner-started.json" in text
    assert "exit.json" in text
    assert ">" in text and "2>" in text
    assert "tmux" not in text


def test_import_failure_and_runner_exception_keep_traceable_exit(tmp_path):
    root = tmp_path / "failure-evidence"
    root.mkdir()
    (root / "stderr.log").write_text("ModuleNotFoundError: missing_package\n", encoding="utf-8")
    (root / "exit.json").write_text(json.dumps({"event": "EXIT", "exit_code": 1}), encoding="utf-8")
    assert controller._status_from_files(root, None, {"exit_code": 1}) == "RUNNER_EXITED_BEFORE_READY"
    assert "ModuleNotFoundError" in (root / "stderr.log").read_text(encoding="utf-8")
    (root / "stderr.log").write_text("RuntimeError: runner boom\n", encoding="utf-8")
    assert controller._status_from_files(root, {"event": "READY"}, {"exit_code": 1}) == "RUNNER_NONZERO_EXIT"
    assert "runner boom" in (root / "stderr.log").read_text(encoding="utf-8")


def _fake_tmux(tmp_path: Path) -> Path:
    fake = tmp_path / "fake-tmux"
    fake.write_text("#!/bin/sh\nlast=\nfor arg in \"$@\"; do last=\"$arg\"; done\n\"$last\" >/dev/null 2>&1 &\nexit 0\n", encoding="utf-8")
    fake.chmod(0o700)
    return fake


def test_real_wrapper_negative_outcomes_preserve_stderr_and_exit(tmp_path):
    fake = _fake_tmux(tmp_path)
    import_failure = _plan(tmp_path, evidence_root=tmp_path / "import-failure", tmux_executable=fake, runner_argv=["-c", "import module_that_does_not_exist_for_rev1"])
    result = controller.launch(import_failure)
    assert result["status"] == "RUNNER_EXITED_BEFORE_READY"
    assert "ModuleNotFoundError" in result["stderr"]

    exception = _plan(tmp_path, evidence_root=tmp_path / "runner-exception", tmux_executable=fake, runner_argv=["-c", "raise RuntimeError('runner boom')"])
    result = controller.launch(exception)
    assert result["status"] == "RUNNER_EXITED_BEFORE_READY"
    assert "runner boom" in result["stderr"]

    ready_root = tmp_path / "quick-normal"
    ready_code = "from pathlib import Path; Path(%r).write_text('{\\\"event\\\":\\\"READY\\\"}\\n')" % str(ready_root / "handshake" / "ready.json")
    result = controller.launch(_plan(tmp_path, evidence_root=ready_root, tmux_executable=fake, runner_argv=["-c", ready_code]))
    assert result["status"] == "COMPLETED"
    assert result["exit"]["exit_code"] == 0

    timeout = _plan(tmp_path, evidence_root=tmp_path / "ready-timeout", tmux_executable=fake, runner_argv=["-c", "import time; time.sleep(0.25)"], ready_timeout=0.05, exit_timeout=0.05)
    result = controller.launch(timeout)
    assert result["status"] == "READY_NOT_OBSERVED"
    time.sleep(0.3)


def test_directory_ownership_contract_is_explicit(tmp_path):
    root = tmp_path / "launch"
    root.mkdir()
    layout = controller._layout(root)
    layout["controller"].mkdir()
    layout["handshake"].mkdir()
    runner = controller._create_runner_workspace(root)
    assert runner == layout["runner"]
    assert layout["controller"].is_dir()
    assert layout["handshake"].is_dir()
    with pytest.raises(controller.LaunchDirectoryError, match="RUNNER_WORKSPACE_ALREADY_EXISTS"):
        controller._create_runner_workspace(root)
    assert layout["controller"].is_dir()


def test_runner_requires_controller_created_root(tmp_path):
    with pytest.raises(controller.LaunchDirectoryError, match="LAUNCH_ROOT_MISSING"):
        controller._create_runner_workspace(tmp_path / "missing-root")


def test_controller_root_and_handshake_are_not_runner_workspace(tmp_path):
    fake = _fake_tmux(tmp_path)
    root = tmp_path / "ownership-positive"
    code = (
        "import os; from pathlib import Path; "
        "runner=Path(os.environ['REV1_RUNNER_DIR']); runner.mkdir(parents=False, exist_ok=False); "
        "Path(os.environ['REV1_HANDSHAKE_DIR'], 'ready.json').write_text('{\\\"event\\\":\\\"READY\\\"}\\n')"
    )
    result = controller.launch(_plan(tmp_path, evidence_root=root, tmux_executable=fake, runner_argv=["-c", code]))
    assert result["status"] == "COMPLETED"
    assert (root / "controller" / "launch-evidence.json").is_file()
    assert (root / "handshake" / "started.json").is_file()
    assert (root / "handshake" / "runner-started.json").is_file()
    assert (root / "handshake" / "ready.json").is_file()
    assert (root / "handshake" / "exit.json").is_file()
    assert (root / "runner").is_dir()
def test_startup_environment_exact_match_and_parent_override():
    expected = {
        "CUDA_VISIBLE_DEVICES": "",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "MKL_THREADING_LAYER": "GNU",
        "PYTHONHASHSEED": "0",
    }
    observed = controller.verify_startup_environment(
        expected,
        environ={
            "CUDA_VISIBLE_DEVICES": "",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "MKL_THREADING_LAYER": "GNU",
            "PYTHONHASHSEED": "0",
        },
    )
    assert observed == expected


@pytest.mark.parametrize(
    "environment",
    [
        {
            "CUDA_VISIBLE_DEVICES": "",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "MKL_THREADING_LAYER": "GNU",
        },
        {
            "CUDA_VISIBLE_DEVICES": "",
            "CUBLAS_WORKSPACE_CONFIG": ":16:8",
            "MKL_THREADING_LAYER": "GNU",
            "PYTHONHASHSEED": "0",
        },
    ],
)
def test_startup_environment_missing_or_wrong_is_fail_closed(environment):
    expected = {
        "CUDA_VISIBLE_DEVICES": "",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "MKL_THREADING_LAYER": "GNU",
        "PYTHONHASHSEED": "0",
    }
    with pytest.raises(controller.StartupEnvironmentError, match="STARTUP_ENVIRONMENT_MISMATCH"):
        controller.verify_startup_environment(expected, environ=environment)


def test_wrapper_explicitly_propagates_startup_environment(tmp_path):
    plan = _plan(
        tmp_path,
        environment={
            "CUDA_VISIBLE_DEVICES": "GPU-test",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "MKL_THREADING_LAYER": "GNU",
            "PYTHONHASHSEED": "0",
        },
    )
    text = controller._wrapper_text(
        plan,
        tmp_path / "wrapper.sh",
        tmp_path / "stdout.log",
        tmp_path / "stderr.log",
    )
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in text
    assert "PYTHONHASHSEED=0" in text
    assert "CUDA_VISIBLE_DEVICES=GPU-test" in text


def _sealed_plan(tmp_path: Path, *, runner_argv=None):
    objects = {
        "policy_authority": {"path": str(tmp_path / "authority.json"), "authority_id": "hardware-policy:test", "identity_sha256": "a" * 64},
        "policy": {"resolved_path": str(tmp_path / "runtime" / "policy.json"), "sha256": "b" * 64, "size_bytes": 12},
        "bridge_checkpoint": {"path": str(tmp_path / "bridge.pt"), "sha256": "c" * 64},
        "bridge_manifest": {"path": str(tmp_path / "bridge.json"), "sha256": "d" * 64},
        "authorization": {"path": str(tmp_path / "auth.json"), "id": "smoke-test", "sha256": "e" * 64},
        "execution_contract": {"path": str(tmp_path / "execution.json"), "sha256": "f" * 64},
    }
    argv = runner_argv or [
        "runner.py", "--root", str(tmp_path / "evidence"),
        "--derived", str(tmp_path / "bridge.pt"), "--manifest", str(tmp_path / "bridge.json"),
        "--auth", str(tmp_path / "auth.json"), "--execution-contract", str(tmp_path / "execution.json"),
        "--policy-authority", str(tmp_path / "authority.json"), "--policy-authority-id", "hardware-policy:test",
        "--policy-authority-identity-sha", "a" * 64,
    ]
    plan = controller.build_plan(
        launch_id="sealed", evidence_root=tmp_path / "evidence", repo_root=tmp_path,
        python_executable=Path(sys.executable), runner_argv=argv, tmux_executable=Path("tmux"),
        session_name="sealed", environment={}, runtime_objects=objects,
    )
    plan["launch_plan_sha256"] = controller.canonical_sha256({k: v for k, v in plan.items() if k != "launch_plan_sha256"})
    return plan


def test_final_launch_plan_matches_sealed_runtime_objects(tmp_path):
    plan = _sealed_plan(tmp_path)
    assert controller.verify_final_launch_plan(plan) == plan["launch_plan_sha256"]


def test_final_launch_plan_rejects_old_policy_path_before_tmux(tmp_path):
    plan = _sealed_plan(tmp_path, runner_argv=_sealed_plan(tmp_path)["runner_argv"] + ["--policy", "/old/policy.json"])
    plan["launch_plan_sha256"] = controller.canonical_sha256({k: v for k, v in plan.items() if k != "launch_plan_sha256"})
    with pytest.raises(Exception, match="physical policy argv is forbidden"):
        controller.verify_final_launch_plan(plan)


def test_final_launch_plan_rejects_mutation_after_seal(tmp_path):
    plan = _sealed_plan(tmp_path)
    plan["runner_argv"][plan["runner_argv"].index("--manifest") + 1] = "/old/manifest.json"
    with pytest.raises(Exception, match="does not match sealed plan"):
        controller.verify_final_launch_plan(plan)


def test_final_launch_plan_rejects_digest_drift(tmp_path):
    plan = _sealed_plan(tmp_path)
    plan["runtime_objects"]["policy"]["resolved_path"] = "/different/policy.json"
    with pytest.raises(Exception, match="launch plan digest drift"):
        controller.verify_final_launch_plan(plan)


def test_versioned_contract_path_accepts_new_authorization_revision(tmp_path):
    contract_dir = tmp_path / "contracts" / "v2b" / "rev001"
    contract_dir.mkdir(parents=True)
    candidate = contract_dir / "gpu_smoke_authorization_r9.json"
    candidate.write_text("{}\n", encoding="utf-8")
    assert controller.validate_versioned_contract_path(tmp_path, candidate, "gpu_smoke_authorization_") == candidate


def test_versioned_contract_path_rejects_old_or_external_path(tmp_path):
    contract_dir = tmp_path / "contracts" / "v2b" / "rev001"
    contract_dir.mkdir(parents=True)
    old = contract_dir / "gpu_smoke_authorization.json"
    old.write_text("{}\n", encoding="utf-8")
    with pytest.raises(Exception, match="VERSIONED_CONTRACT_PATH_MISMATCH"):
        controller.validate_versioned_contract_path(tmp_path, old, "gpu_smoke_authorization_")
    external = tmp_path / "gpu_smoke_authorization_r9.json"
    external.write_text("{}\n", encoding="utf-8")
    with pytest.raises(Exception, match="VERSIONED_CONTRACT_PATH_MISMATCH"):
        controller.validate_versioned_contract_path(tmp_path, external, "gpu_smoke_authorization_")


def _gpu_csv(uuid="GPU-11111111-2222-3333-4444-555555555555", pci="00000000:01:00.0",
             name="NVIDIA GeForce RTX 4090 D", used="15", free="24067",
             util="0", temp="51", clock="1500"):
    return f"0, {uuid}, {pci}, {name}, {used}, {free}, {util}, {temp}, {clock}\n"


def test_gpu_identity_parser_accepts_bound_physical_gpu():
    result = controller.parse_gpu_identity_query(
        _gpu_csv(),
        "",
        expected_gpu_uuid="GPU-11111111-2222-3333-4444-555555555555",
        expected_pci_bus_id="00000000:01:00.0",
        expected_gpu_name="NVIDIA GeForce RTX 4090 D",
    )
    assert result["status"] == "PASS"
    assert result["physical_gpu"]["uuid"].startswith("GPU-")
    assert result["telemetry"]["memory_free_mib"] == 24067.0


@pytest.mark.parametrize(
    "kwargs,compute,reason",
    [
        ({"expected_gpu_uuid": "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"}, "", "GPU_UUID_MISSING_OR_AMBIGUOUS"),
        ({"expected_pci_bus_id": "00000000:02:00.0"}, "", "GPU_PCI_BUS_MISMATCH"),
        ({"expected_gpu_name": "Other GPU"}, "", "GPU_NAME_MISMATCH"),
        ({}, "123, python, 100 MiB\n", "EXTERNAL_GPU_COMPUTE_PROCESS"),
        ({"max_clock_mhz": 1400}, "", "GPU_HEALTH_ADMISSION_MISMATCH"),
    ],
)
def test_gpu_identity_parser_rejects_identity_or_admission_drift(kwargs, compute, reason):
    values = {
        "expected_gpu_uuid": "GPU-11111111-2222-3333-4444-555555555555",
        "expected_pci_bus_id": "00000000:01:00.0",
        "expected_gpu_name": "NVIDIA GeForce RTX 4090 D",
    }
    values.update(kwargs)
    with pytest.raises(controller.GPUIdentityPreflightError, match=reason):
        controller.parse_gpu_identity_query(_gpu_csv(), compute, **values)


def test_gpu_identity_probe_uses_nvidia_smi_and_logical_cuda_mapping(monkeypatch, tmp_path):
    calls = []
    class Completed:
        def __init__(self, stdout, returncode=0, stderr=""):
            self.stdout, self.returncode, self.stderr = stdout, returncode, stderr
    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        if "--query-gpu=" in " ".join(argv):
            return Completed(_gpu_csv())
        if "--query-compute-apps=" in " ".join(argv):
            return Completed("")
        return Completed('{"cuda_available": true, "cuda_initialized": false, "device_count": 1, "device_name": "NVIDIA GeForce RTX 4090 D"}\n')
    monkeypatch.setattr(controller.subprocess, "run", fake_run)
    result = controller.probe_visible_gpu_identity(
        expected_gpu_uuid="GPU-11111111-2222-3333-4444-555555555555",
        expected_pci_bus_id="00000000:01:00.0",
        expected_gpu_name="NVIDIA GeForce RTX 4090 D",
        python_executable=tmp_path / "python",
        nvidia_smi_executable=tmp_path / "nvidia-smi",
        startup_environment={"CUDA_VISIBLE_DEVICES": "GPU-11111111-2222-3333-4444-555555555555"},
    )
    assert result["logical_cuda"]["device"] == "cuda:0"
    assert result["logical_cuda"]["cuda_visible_devices"].startswith("GPU-")
    assert len(calls) == 3
    assert all(".uuid" not in " ".join(call[0]) for call in calls)
