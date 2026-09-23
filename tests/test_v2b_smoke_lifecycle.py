from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace


MODULE_PATH = Path(__file__).parents[1] / "tools" / "run_v2b_rev1_gpu_smoke_lifecycle.py"
_spec = importlib.util.spec_from_file_location("v2b_smoke_lifecycle", MODULE_PATH)
assert _spec and _spec.loader
_lifecycle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lifecycle)


def _child(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def test_lifecycle_accepts_controller_invocation_flag() -> None:
    parser = _lifecycle.build_parser()
    argv = []
    for action in parser._actions:
        if action.required:
            option = "--invocation" if action.dest == "structured_invocation" else action.option_strings[0]
            argv.extend((option, "bound-value"))
    assert parser.parse_args(argv).structured_invocation == "bound-value"


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


def test_child_environment_applies_contract_startup_environment(tmp_path: Path) -> None:
    contract = tmp_path / "execution-contract.json"
    _lifecycle.write_json(contract, {
        "startup_environment": {
            "static": {
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "MKL_NUM_THREADS": "2",
                "MKL_THREADING_LAYER": "GNU",
                "OMP_NUM_THREADS": "2",
                "PYTHONHASHSEED": "0",
                "PYTHONNOUSERSITE": "1",
            },
            "cpu_rehearsal": {"CUDA_VISIBLE_DEVICES": ""},
            "gpu": {"CUDA_VISIBLE_DEVICES": "GPU-test"},
        }
    })
    args = SimpleNamespace(
        execution_contract=str(contract), gpu_uuid="GPU-test", repo=str(tmp_path)
    )
    env = _lifecycle.child_environment(args, cpu=True)
    assert env["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert env["MKL_NUM_THREADS"] == "2"
    assert env["PYTHONHASHSEED"] == "0"
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["PYTHONPATH"] == str(tmp_path / "src")


def _monitor_report(status: str = "PASS", *, worker_exited: bool = True) -> dict:
    return {
        "status": "PASS", "monitor_final": {
            "status": status, "worker_exited": worker_exited,
            "transaction_id": "smoke-test", "monitor_phase": "restore",
        },
    }


def test_require_monitor_final_accepts_logical_pass_only_with_monitor_pass() -> None:
    final = _lifecycle.require_monitor_final(_monitor_report(), "restore", "smoke-test")
    assert final["status"] == "PASS"


def test_require_monitor_final_rejects_logical_pass_with_monitor_failure() -> None:
    try:
        _lifecycle.require_monitor_final(_monitor_report("FAIL"), "restore", "smoke-test")
    except RuntimeError as exc:
        assert "monitor-final failed" in str(exc)
    else:
        raise AssertionError("monitor failure was accepted")


def test_require_monitor_final_rejects_missing_or_live_monitor() -> None:
    for report in ({"status": "PASS"}, _monitor_report("PASS", worker_exited=False)):
        try:
            _lifecycle.require_monitor_final(report, "restore", "smoke-test")
        except RuntimeError:
            pass
        else:
            raise AssertionError("incomplete monitor final was accepted")


def test_finalize_child_report_waits_for_reaped_owner_and_publishes_final(tmp_path, monkeypatch):
    report_path = tmp_path / "quiescence-report.json"
    pending = {
        "status": "PASS_PENDING_MONITOR",
        "monitor_finish": {"monitor_final_report": str(tmp_path / "monitor-final.json"),
                           "transaction_id": "tx", "monitor_phase": "quiescence"},
    }
    observed = []
    def fake_wait(reference):
        observed.append(reference)
        return {"status": "PASS", "worker_exited": True,
                "transaction_id": "tx", "monitor_phase": "quiescence"}
    monkeypatch.setattr(_lifecycle, "_wait_for_monitored_finish", fake_wait)
    finalized = _lifecycle.finalize_child_report(report_path, pending, "preflight", "tx")
    assert observed == [pending["monitor_finish"]]
    assert finalized["status"] == "PASS"
    assert finalized["monitor_final_pending"] is False
    assert finalized["monitor_final"]["worker_exited"] is True
    assert _lifecycle.read_json(report_path)["status"] == "PASS"


def test_deferred_child_result_is_not_accepted_without_monitor_reference(tmp_path):
    try:
        _lifecycle.finalize_child_report(
            tmp_path / "missing.json", {"status": "PASS_PENDING_MONITOR"},
            "training", "tx",
        )
    except RuntimeError as exc:
        assert "missing deferred monitor reference" in str(exc)
    else:
        raise AssertionError("missing monitor reference was accepted")


def test_lifecycle_closes_monitor_only_after_child_reap():
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "finalize_child_report" in source
    assert source.index("training = run_child") < source.index("finalize_child_report(", source.index("training = run_child"))
