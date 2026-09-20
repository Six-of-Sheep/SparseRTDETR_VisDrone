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
        environment={"CUDA_VISIBLE_DEVICES": ""},
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
