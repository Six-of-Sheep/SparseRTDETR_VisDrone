from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import pytest

from sparse_rtdetr.baseline.smoke import load_smoke_config
from sparse_rtdetr.baseline.smoke_evidence import canonical_json_bytes, sha256_bytes
from sparse_rtdetr.baseline.smoke_outer_launcher import (
    OuterLaunchError,
    _finish_outer,
    _run_tmux,
    classify_outer_evidence,
    run_outer_launch,
    validate_outer_evidence,
    validate_pane_evidence,
    validate_process_evidence,
)


ROOT = Path(__file__).resolve().parents[1]
V1_CONFIG = ROOT / "configs/baseline/rtdetrv2_r18_visdrone_smoke_v1.json"
V2_CONFIG = ROOT / "configs/baseline/rtdetrv2_r18_visdrone_smoke_v2.json"
SESSION = "p3_rtdetrv2_r18_visdrone_baseline_smoke_r2"


def _script(path: Path, body: str) -> Path:
    path.write_text("#!/bin/sh\n" + body + "\n", encoding="ascii")
    path.chmod(0o755)
    return path


def _repo(tmp_path: Path, *, child_rc: int = 0) -> tuple[dict[str, Path | str], Path, Path]:
    root = tmp_path / "repo"
    shutil.copytree(ROOT / "src", root / "src")
    (root / "configs/baseline").mkdir(parents=True)
    (root / "artifacts/data/visdrone_protocol_v2_conversion_r3").mkdir(parents=True)
    shutil.copyfile(V2_CONFIG, root / "configs/baseline/rtdetrv2_r18_visdrone_smoke_v2.json")
    config = json.loads(V2_CONFIG.read_text(encoding="utf-8"))
    (root / "artifacts/data/visdrone_protocol_v2_conversion_r3/train_core_manifest.json").write_text(
        json.dumps({"records": config["image_selection"]["records"]}), encoding="utf-8"
    )
    (root / "artifacts/runs").mkdir(parents=True)
    (root / "artifacts/process_evidence").mkdir(parents=True)
    outer_parent = root / "artifacts/outer_launch_evidence"
    outer_parent.mkdir(parents=True)
    outer_parent.chmod(0o775)
    data = tmp_path / "data"
    data.mkdir()
    child_count = tmp_path / "child.count"
    child = _script(tmp_path / "child", f"echo child >> {child_count}\nexit {child_rc}")
    tmux_count = tmp_path / "tmux.count"
    tmux = _script(tmp_path / "fake-tmux", f"echo tmux >> {tmux_count}\n\"$7\"\nexit 0")
    output = root / "artifacts/runs/rtdetrv2_r18_visdrone_baseline_smoke_r2"
    process = root / "artifacts/process_evidence/rtdetrv2_r18_visdrone_baseline_smoke_r2"
    outer = outer_parent / "rtdetrv2_r18_visdrone_baseline_smoke_r2"
    args: dict[str, Path | str] = {
        "repo_root": root,
        "data_root": data,
        "config_path": root / "configs/baseline/rtdetrv2_r18_visdrone_smoke_v2.json",
        "output_dir": output,
        "process_evidence_dir": process,
        "outer_evidence_dir": outer,
        "child_python": child,
        "tmux_executable": tmux,
        "tmux_session": SESSION,
    }
    return args, child_count, tmux_count


def _digests(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_v1_config_is_preserved_and_scientific_contract_matches_v2():
    assert hashlib.sha256(V1_CONFIG.read_bytes()).hexdigest() == "76562afe145a7ec1049497176da6a91f6513eb4244a737335c1c12ea7c5f55d2"
    v1 = load_smoke_config(ROOT, V1_CONFIG)
    v2 = load_smoke_config(ROOT, V2_CONFIG)
    for field in ("execution", "model", "image_selection", "evidence", "mode", "split_role"):
        assert v1[field] == v2[field]
    for field in ("authorized_env", "visible_devices", "device_count", "cpu_fallback"):
        assert v1["runtime"][field] == v2["runtime"][field]


def test_contract_check_is_data_free_and_does_not_create_outer_evidence(tmp_path):
    args, _child_count, _tmux_count = _repo(tmp_path)
    from sparse_rtdetr.baseline.smoke_outer_launcher import contract_check

    result = contract_check(args["repo_root"], args["config_path"])
    assert result["status"] == "PASS"
    assert result["tmux_called"] is False
    assert not args["outer_evidence_dir"].exists()


def test_exact_session_rejects_all_variants_before_fake_tmux(tmp_path):
    invalid = [
        "p3_rtdetrv2_r18_visdrone_baseline_smoke_r1",
        "other",
        "",
        "bad;tmux",
        "p3_rtdetrv2_r18_visdrone_baseline_smoke_r2 ",
        "p3_rtdetrv2_r18_visdrone_baseline_smoke_r2\n",
        "x" * 200,
        "非ascii",
    ]
    for index, value in enumerate(invalid):
        args, _child_count, tmux_count = _repo(tmp_path / str(index))
        args["tmux_session"] = value
        result = run_outer_launch(**args)
        assert result["status"] == "PREFLIGHT_FAILED"
        assert not tmux_count.exists()
        assert classify_outer_evidence(args["outer_evidence_dir"], args["output_dir"], args["process_evidence_dir"]) == "PREFLIGHT_FAILED"


def test_first_wrapper_execution_has_canonical_receipt_and_inventory(tmp_path):
    args, child_count, tmux_count = _repo(tmp_path)
    result = run_outer_launch(**args)
    assert result["status"] == "TMUX_ACCEPTED"
    assert tmux_count.read_text(encoding="ascii").splitlines() == ["tmux"]
    assert child_count.read_text(encoding="ascii").splitlines() == ["child"]
    pane = args["outer_evidence_dir"] / "pane"
    pane_value = validate_pane_evidence(pane)
    assert pane_value["receipt"]["nonce"] == pane_value["plan"]["nonce"]
    inventory_value = json.loads((pane / "pane_inventory.json").read_text(encoding="utf-8"))
    assert inventory_value["canonical_inventory_sha256"] == sha256_bytes(canonical_json_bytes(inventory_value["artifacts"]))
    validate_outer_evidence(args["outer_evidence_dir"])
    assert classify_outer_evidence(args["outer_evidence_dir"], args["output_dir"], args["process_evidence_dir"]) == "INNER_STARTED_PROCESS_EVIDENCE_ABSENT"


def test_wrapper_replay_is_exclusive_and_does_not_change_first_evidence(tmp_path):
    args, child_count, _tmux_count = _repo(tmp_path)
    run_outer_launch(**args)
    pane = args["outer_evidence_dir"] / "pane"
    before = _digests(pane)
    replay = subprocess.run([str(pane / "pane_wrapper.sh")], cwd=args["repo_root"], capture_output=True, check=False)
    assert replay.returncode == 73
    assert child_count.read_text(encoding="ascii").splitlines() == ["child"]
    assert _digests(pane) == before


def test_concurrent_wrapper_execution_has_one_consumer(tmp_path):
    args, child_count, _tmux_count = _repo(tmp_path)
    run_outer_launch(**args)
    wrapper = args["outer_evidence_dir"] / "pane/pane_wrapper.sh"
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _index: subprocess.run([str(wrapper)], cwd=args["repo_root"], capture_output=True, check=False).returncode, (1, 2)))
    assert 73 in results
    assert child_count.read_text(encoding="ascii").splitlines() == ["child"]


def test_nonzero_inner_is_terminal_failure_not_success(tmp_path):
    args, _child_count, _tmux_count = _repo(tmp_path, child_rc=99)
    result = run_outer_launch(**args)
    assert result["status"] == "TMUX_ACCEPTED"
    assert classify_outer_evidence(args["outer_evidence_dir"], args["output_dir"], args["process_evidence_dir"]) == "TERMINAL_FAILED"


def test_forged_process_completed_with_exit_99_is_rejected(tmp_path):
    process = tmp_path / "process"
    process.mkdir()
    (process / "process_exit_code.txt").write_bytes(b"99\n")
    (process / "process_completion.json").write_text(json.dumps({"schema_version": 1, "status": "COMPLETED", "child_returncode": 0, "launcher_exit_code": 0}), encoding="utf-8")
    with pytest.raises(OuterLaunchError):
        validate_process_evidence(process)


def test_intermediate_symlink_is_rejected_before_tmux(tmp_path):
    args, _child_count, tmux_count = _repo(tmp_path)
    real_parent = tmp_path / "real-data"
    real_parent.mkdir()
    link_parent = tmp_path / "data-link"
    link_parent.symlink_to(real_parent, target_is_directory=True)
    args["data_root"] = link_parent / "nested"
    (real_parent / "nested").mkdir()
    result = run_outer_launch(**args)
    assert result["status"] == "PREFLIGHT_FAILED"
    assert not tmux_count.exists()


def test_pane_validator_rejects_nonce_config_and_path_drift(tmp_path):
    args, _child_count, _tmux_count = _repo(tmp_path)
    run_outer_launch(**args)
    pane = args["outer_evidence_dir"] / "pane"
    receipt_path = pane / "pane_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["nonce"] = "0" * 32
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(OuterLaunchError):
        validate_pane_evidence(pane)


def test_classifier_is_read_only_and_rejects_forged_tmux_return_code(tmp_path):
    args, _child_count, _tmux_count = _repo(tmp_path)
    run_outer_launch(**args)
    root = args["outer_evidence_dir"]
    before = _digests(root)
    mtimes = {str(path.relative_to(root)): path.stat().st_mtime_ns for path in root.rglob("*") if path.is_file()}
    assert classify_outer_evidence(root, args["output_dir"], args["process_evidence_dir"]) == "INNER_STARTED_PROCESS_EVIDENCE_ABSENT"
    assert _digests(root) == before
    assert {str(path.relative_to(root)): path.stat().st_mtime_ns for path in root.rglob("*") if path.is_file()} == mtimes
    completion_path = root / "launcher/outer_completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["tmux_returncode"] = True
    completion_path.write_text(json.dumps(completion), encoding="utf-8")
    assert classify_outer_evidence(root, args["output_dir"], args["process_evidence_dir"]) == "UNKNOWN"


def test_popen_exception_keeps_original_error_and_does_not_retry(tmp_path):
    args, _child_count, _tmux_count = _repo(tmp_path)
    with mock.patch("sparse_rtdetr.baseline.smoke_outer_launcher.subprocess.Popen", side_effect=RuntimeError("popen sentinel")) as popen:
        result = run_outer_launch(**args)
    assert popen.call_count == 1
    assert result["status"] == "INTERRUPTED_WITH_EVIDENCE"
    error = json.loads((args["outer_evidence_dir"] / "launcher/outer_error.json").read_text(encoding="utf-8"))
    assert error["message"] == "popen sentinel"


def test_tmux_runner_is_bounded_and_records_timeout():
    result = _run_tmux(argv=["/bin/sh", "-c", "sleep 1"], cwd=Path("/tmp"), env={"PATH": "/usr/bin:/bin"}, timeout_seconds=0.01)
    assert result["timeout_triggered"] is True
    assert result["term_sent"] is True
    assert result["signal_handlers_installed"] is True
    assert result["monitored_signals"] == ["SIGHUP", "SIGTERM", "SIGINT", "SIGQUIT"]


@pytest.mark.parametrize("signum", [1, 2, 3, 15])
def test_tmux_runner_records_each_monitored_signal(signum):
    handlers = {}

    class FakeProcess:
        pid = 99999999
        returncode = 0

        def poll(self):
            return None

        def communicate(self, **_kwargs):
            handlers[signum](signum, None)
            return b"out", b"err"

    def fake_signal(number, handler):
        if callable(handler):
            handlers[number] = handler
        return signal.SIG_DFL

    import signal

    with mock.patch("sparse_rtdetr.baseline.smoke_outer_launcher.signal.signal", side_effect=fake_signal), mock.patch("sparse_rtdetr.baseline.smoke_outer_launcher.subprocess.Popen", return_value=FakeProcess()):
        result = _run_tmux(argv=["fake-tmux"], cwd=Path("/tmp"), env={}, timeout_seconds=10)
    assert result["observed_signal_count"] == 1
    assert result["signal_events"][0]["signal_number"] == signum


def test_outer_secondary_finalization_preserves_original_exception(tmp_path):
    args, _child_count, _tmux_count = _repo(tmp_path)
    with mock.patch("sparse_rtdetr.baseline.smoke_outer_launcher._inventory_value", side_effect=ValueError("inventory sentinel")):
        result = run_outer_launch(**args)
    assert result["status"] == "FINALIZATION_FAILED"
    secondary = args["outer_evidence_dir"] / "launcher/outer_secondary_finalization_failure.json"
    value = json.loads(secondary.read_text(encoding="utf-8"))
    assert value["original_exception_type"] == "ValueError"
    assert value["original_exception_message"] == "inventory sentinel"
