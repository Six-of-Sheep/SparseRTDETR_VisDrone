from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys


MODULE_PATH = Path(__file__).parents[1] / "tools" / "run_v2b_rev1_gpu_smoke_lifecycle.py"
_spec = importlib.util.spec_from_file_location("v2b_smoke_lifecycle", MODULE_PATH)
assert _spec and _spec.loader
_lifecycle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lifecycle)


def _child(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def test_run_child_waits_reaps_and_captures_stdout(tmp_path: Path) -> None:
    evidence = tmp_path / "ok"
    result = _lifecycle.run_child(
        "ok", _child("print('READY')"), cwd=tmp_path,
        env=dict(os.environ), evidence_root=evidence,
    )
    assert result["returncode"] == 0
    assert result["reaped"] is True
    assert result["pid_absent"] is True
    assert (evidence / "stdout.log").read_text(encoding="utf-8").strip() == "READY"
    assert (evidence / "exit.json").is_file()


def test_run_child_preserves_exit_code_and_stderr(tmp_path: Path) -> None:
    evidence = tmp_path / "failed"
    try:
        _lifecycle.run_child(
            "failed", _child("import sys; print('ROOT_CAUSE', file=sys.stderr); sys.exit(7)"),
            cwd=tmp_path, env=dict(os.environ), evidence_root=evidence,
        )
    except RuntimeError as exc:
        assert "rc=7" in str(exc)
    else:
        raise AssertionError("failed child was accepted")
    assert "ROOT_CAUSE" in (evidence / "stderr.log").read_text(encoding="utf-8")
    exit_record = _lifecycle.read_json(evidence / "exit.json")
    assert exit_record["returncode"] == 7
    assert exit_record["pid_absent"] is True


def test_run_child_records_exact_argv_and_cwd(tmp_path: Path) -> None:
    evidence = tmp_path / "identity"
    command = _child("pass")
    result = _lifecycle.run_child(
        "identity", command, cwd=tmp_path, env=dict(os.environ),
        evidence_root=evidence,
    )
    assert result["argv"] == command
    assert result["cwd"] == str(tmp_path)


def test_run_child_does_not_reuse_evidence_root(tmp_path: Path) -> None:
    evidence = tmp_path / "single"
    _lifecycle.run_child("one", _child("pass"), cwd=tmp_path,
                         env=dict(os.environ), evidence_root=evidence)
    try:
        _lifecycle.run_child("two", _child("pass"), cwd=tmp_path,
                             env=dict(os.environ), evidence_root=evidence)
    except FileExistsError:
        pass
    else:
        raise AssertionError("evidence root was overwritten")


def test_pid_absent_rejects_live_process(tmp_path: Path) -> None:
    import subprocess
    process = subprocess.Popen(_child("import time; time.sleep(1)"))
    try:
        assert _lifecycle.pid_absent(process.pid) is False
    finally:
        process.terminate()
        process.wait()
    assert _lifecycle.pid_absent(process.pid) is True


def test_parse_stdout_requires_one_json_object(tmp_path: Path) -> None:
    path = tmp_path / "stdout.log"
    path.write_text('{"status":"PASS"}', encoding="utf-8")
    assert _lifecycle.parse_stdout_json(path)["status"] == "PASS"
