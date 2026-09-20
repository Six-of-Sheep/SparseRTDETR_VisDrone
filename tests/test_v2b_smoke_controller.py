from __future__ import annotations

import json
import os
import stat
import sys
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
    assert (tmp_path / "evidence" / "launch-evidence.json").is_file()


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
