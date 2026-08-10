from __future__ import annotations

import hashlib
import json
import os
import signal
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import pytest

from sparse_rtdetr.baseline.smoke import load_smoke_config
from sparse_rtdetr.baseline.smoke_evidence import canonical_json_bytes, inventory, sha256_bytes, sha256_file
from sparse_rtdetr.baseline.smoke_outer_launcher import (
    OUTER_EXCLUDED,
    PANE_EXCLUDED,
    OuterLaunchError,
    executable_identity,
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


def _ref(root: Path, name: str) -> dict[str, object]:
    path = root / name
    return {"present": True, "relative_path": name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _repack_pane(pane: Path) -> None:
    inventory_path = pane / "pane_inventory.json"
    inventory_path.write_bytes(canonical_json_bytes(inventory(pane, PANE_EXCLUDED)))
    completion_path = pane / "pane_completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["pane_inventory"] = _ref(pane, "pane_inventory.json")
    completion_path.write_bytes(canonical_json_bytes(completion))


def _repack_outer(root: Path) -> None:
    launcher = root / "launcher"
    outer_inventory = launcher / "outer_inventory.json"
    outer_partial = launcher / "outer_partial_inventory.json"
    value = inventory(launcher, OUTER_EXCLUDED)
    payload = canonical_json_bytes(value)
    outer_inventory.write_bytes(payload)
    outer_partial.write_bytes(payload)
    completion_path = launcher / "outer_completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["invocation"] = _ref(launcher, "outer_invocation.json")
    completion["outer_inventory"] = _ref(launcher, "outer_inventory.json")
    completion["outer_partial_inventory"] = _ref(launcher, "outer_partial_inventory.json")
    completion_path.write_bytes(canonical_json_bytes(completion))


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


@pytest.mark.parametrize("status", ["COMPLETED", "FAILED", "SUCCESS", "", None, True, 1, "PANE_FAILED"])
def test_pane_completion_status_is_strict_after_inventory_repack(tmp_path, status):
    args, _child_count, _tmux_count = _repo(tmp_path)
    run_outer_launch(**args)
    pane = args["outer_evidence_dir"] / "pane"
    completion_path = pane / "pane_completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["status"] = status
    completion_path.write_bytes(canonical_json_bytes(completion))
    _repack_pane(pane)
    with pytest.raises(OuterLaunchError):
        validate_pane_evidence(pane)


def test_pane_failed_status_requires_failure_evidence(tmp_path):
    args, _child_count, _tmux_count = _repo(tmp_path, child_rc=99)
    run_outer_launch(**args)
    pane = args["outer_evidence_dir"] / "pane"
    completion_path = pane / "pane_completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    assert completion["status"] == "PANE_FAILED"
    completion["inner_returncode"] = 0
    (pane / "pane_exit_code.txt").write_bytes(b"0\n")
    completion_path.write_bytes(canonical_json_bytes(completion))
    _repack_pane(pane)
    with pytest.raises(OuterLaunchError):
        validate_pane_evidence(pane)


def _signal_event() -> dict[str, object]:
    return {
        "signal_number": signal.SIGHUP,
        "signal_name": "SIGHUP",
        "received_at_utc": "2026-08-10T12:00:00+00:00",
        "forwarded_to_tmux_client": True,
        "forwarding_result": "forwarded",
    }


@pytest.mark.parametrize("mutation", [
    lambda event, runner: {},
    lambda event, runner: {key: value for key, value in event.items() if key != "signal_number"},
    lambda event, runner: {key: value for key, value in event.items() if key != "signal_name"},
    lambda event, runner: {key: value for key, value in event.items() if key != "received_at_utc"},
    lambda event, runner: {key: value for key, value in event.items() if key != "forwarded_to_tmux_client"},
    lambda event, runner: {key: value for key, value in event.items() if key != "forwarding_result"},
    lambda event, runner: {**event, "extra": True},
    lambda event, runner: {**event, "signal_number": True},
    lambda event, runner: {**event, "signal_number": signal.SIGKILL},
    lambda event, runner: {**event, "signal_name": "SIGTERM"},
    lambda event, runner: {**event, "received_at_utc": "bad"},
    lambda event, runner: {**event, "received_at_utc": "2026-08-10T12:00:00"},
    lambda event, runner: {**event, "received_at_utc": "2026-08-10T12:00:00+01:00"},
    lambda event, runner: {**event, "forwarded_to_tmux_client": "true"},
    lambda event, runner: {**event, "forwarding_result": "failed:Nope"},
    lambda event, runner: {**event, "forwarded_to_tmux_client": False, "forwarding_result": "forwarded"},
    lambda event, runner: {**event, "forwarded_to_tmux_client": False, "forwarding_result": "free text"},
    lambda event, runner: event,
])
def test_outer_signal_event_mutations_are_rejected_after_inventory_repack(tmp_path, mutation):
    args, _child_count, _tmux_count = _repo(tmp_path)
    run_outer_launch(**args)
    root = args["outer_evidence_dir"]
    completion_path = root / "launcher/outer_completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    event = _signal_event()
    runner = completion["runner"]
    runner["signal_events"] = [mutation(dict(event), runner)]
    runner["observed_signal_count"] = 1
    if mutation(event, runner) == event:
        runner["observed_signal_count"] = True
    completion_path.write_bytes(canonical_json_bytes(completion))
    _repack_outer(root)
    with pytest.raises(OuterLaunchError):
        validate_outer_evidence(root)


def test_outer_signal_event_timestamp_order_is_rejected_after_repack(tmp_path):
    args, _child_count, _tmux_count = _repo(tmp_path)
    run_outer_launch(**args)
    root = args["outer_evidence_dir"]
    completion_path = root / "launcher/outer_completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    first = _signal_event()
    second = dict(first)
    second["received_at_utc"] = "2026-08-10T11:59:59+00:00"
    completion["runner"]["signal_events"] = [first, second]
    completion["runner"]["observed_signal_count"] = 2
    completion_path.write_bytes(canonical_json_bytes(completion))
    _repack_outer(root)
    with pytest.raises(OuterLaunchError):
        validate_outer_evidence(root)


@pytest.mark.parametrize("field", ["repo_root", "data_root", "config_path", "output_dir", "process_evidence_dir", "outer_evidence_dir", "child_python", "tmux_executable", "tmux_session"])
def test_outer_invocation_path_and_identity_drift_is_rejected_after_repack(tmp_path, field):
    args, _child_count, _tmux_count = _repo(tmp_path)
    run_outer_launch(**args)
    root = args["outer_evidence_dir"]
    invocation_path = root / "launcher/outer_invocation.json"
    invocation = json.loads(invocation_path.read_text(encoding="utf-8"))
    invocation[field] = "other" if field == "tmux_session" else "/tmp/forged-" + field
    invocation_path.write_bytes(canonical_json_bytes(invocation))
    _repack_outer(root)
    with pytest.raises(OuterLaunchError):
        validate_outer_evidence(root, args["output_dir"], args["process_evidence_dir"])


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


def test_inner_failures_write_strict_pane_error_and_bind_console(tmp_path):
    for child_rc in (1, 99):
        args, _child_count, _tmux_count = _repo(tmp_path / str(child_rc), child_rc=child_rc)
        run_outer_launch(**args)
        pane = args["outer_evidence_dir"] / "pane"
        value = validate_pane_evidence(pane)
        error = json.loads((pane / "pane_error.json").read_text(encoding="utf-8"))
        assert value["completion"]["status"] == "PANE_FAILED"
        assert error["status"] == "PANE_FAILED"
        assert error["failure_class"] == "INNER_NONZERO_EXIT"
        assert error["message"] == "inner command exited with a nonzero return code."
        assert error["console"] == _ref(pane, "pane_console.log")
        assert type(error["inner_returncode"]) is int
        assert error["inner_returncode"] == child_rc
        assert error["pane_pid"] == value["receipt"]["pane_pid"]


def test_pane_failed_requires_present_error_and_completed_forbids_it(tmp_path):
    args, _child_count, _tmux_count = _repo(tmp_path, child_rc=99)
    run_outer_launch(**args)
    pane = args["outer_evidence_dir"] / "pane"
    error_path = pane / "pane_error.json"
    completion_path = pane / "pane_completion.json"
    error = json.loads(error_path.read_text(encoding="utf-8"))
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    error_path.unlink()
    completion["error"] = {"present": False, "relative_path": "pane_error.json"}
    completion_path.write_bytes(canonical_json_bytes(completion))
    _repack_pane(pane)
    with pytest.raises(OuterLaunchError):
        validate_pane_evidence(pane)

    error_path.write_bytes(canonical_json_bytes(error))
    completion["status"] = "PANE_COMPLETED"
    completion["inner_started"] = True
    completion["inner_returncode"] = 0
    (pane / "pane_exit_code.txt").write_bytes(b"0\n")
    completion["error"] = _ref(pane, "pane_error.json")
    completion_path.write_bytes(canonical_json_bytes(completion))
    _repack_pane(pane)
    with pytest.raises(OuterLaunchError):
        validate_pane_evidence(pane)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: {**value, "elapsed_seconds": -1.0},
        lambda value: {**value, "elapsed_seconds": True},
        lambda value: {**value, "elapsed_seconds": float("nan")},
        lambda value: {**value, "elapsed_seconds": float("inf")},
        lambda value: {**value, "finished_at_utc": "2020-01-01T00:00:00+00:00"},
        lambda value: {**value, "started_at_utc": "2020-01-01T00:00:00"},
        lambda value: {**value, "extra": True},
    ],
)
def test_pane_timing_semantics_are_strict(tmp_path, mutation):
    args, _child_count, _tmux_count = _repo(tmp_path)
    run_outer_launch(**args)
    pane = args["outer_evidence_dir"] / "pane"
    timing_path = pane / "pane_timing.json"
    timing = json.loads(timing_path.read_text(encoding="utf-8"))
    timing_path.write_bytes(canonical_json_bytes(mutation(timing)))
    _repack_pane(pane)
    with pytest.raises(OuterLaunchError):
        validate_pane_evidence(pane)


@pytest.mark.parametrize("filename", ["pane_start.json", "pane_receipt.json", "pane_completion.json"])
def test_pane_pid_binding_rejects_each_layer_drift(tmp_path, filename):
    args, _child_count, _tmux_count = _repo(tmp_path)
    run_outer_launch(**args)
    pane = args["outer_evidence_dir"] / "pane"
    path = pane / filename
    value = json.loads(path.read_text(encoding="utf-8"))
    value["pane_pid"] += 1
    path.write_bytes(canonical_json_bytes(value))
    _repack_pane(pane)
    with pytest.raises(OuterLaunchError):
        validate_pane_evidence(pane)


def test_executable_identity_rechecks_bytes_mode_and_canonical_path(tmp_path):
    executable = _script(tmp_path / "python", "exit 0")
    identity = executable_identity(executable)
    assert identity["canonical_path"] == str(executable)
    assert identity["regular_file"] is True
    assert identity["executable"] is True
    executable.write_text("#!/bin/sh\nexit 99\n", encoding="ascii")
    executable.chmod(0o755)
    assert executable_identity(executable)["sha256"] != identity["sha256"]
    executable.chmod(0o644)
    with pytest.raises(OuterLaunchError):
        executable_identity(executable)


def test_child_and_finalizer_identity_drift_is_rejected_after_inventory_repack(tmp_path):
    args, _child_count, _tmux_count = _repo(tmp_path)
    run_outer_launch(**args)
    root = args["outer_evidence_dir"]
    invocation_path = root / "launcher/outer_invocation.json"
    invocation = json.loads(invocation_path.read_text(encoding="utf-8"))
    invocation["environment"]["PANE_EVIDENCE_PYTHON"] = "/tmp/other-python"
    invocation_path.write_bytes(canonical_json_bytes(invocation))
    _repack_outer(root)
    with pytest.raises(OuterLaunchError):
        validate_outer_evidence(root, args["output_dir"], args["process_evidence_dir"])


def test_plan_and_wrapper_rebuild_reject_repacked_semantic_drift(tmp_path):
    args, _child_count, _tmux_count = _repo(tmp_path)
    run_outer_launch(**args)
    root = args["outer_evidence_dir"]
    pane = root / "pane"
    plan_path = pane / "pane_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["inner_argv"] = [*plan["inner_argv"], "--forged"]
    plan["inner_argv_sha256"] = sha256_bytes(canonical_json_bytes(plan["inner_argv"]))
    plan_path.write_bytes(canonical_json_bytes(plan))
    wrapper = pane / "pane_wrapper.sh"
    wrapper.write_bytes(wrapper.read_bytes() + b"\n# forged\n")
    _repack_pane(pane)
    _repack_outer(root)
    with pytest.raises(OuterLaunchError):
        validate_outer_evidence(root, args["output_dir"], args["process_evidence_dir"])


def test_preexisting_runtime_evidence_is_not_overwritten_or_consumed(tmp_path):
    args, child_count, _tmux_count = _repo(tmp_path)
    run_outer_launch(**args)
    pane = args["outer_evidence_dir"] / "pane"
    before = b"preexisting receipt\n"
    receipt = pane / "pane_receipt.json"
    receipt.write_bytes(before)
    replay = subprocess.run([str(pane / "pane_wrapper.sh")], cwd=args["repo_root"], capture_output=True, check=False)
    assert replay.returncode == 73
    assert receipt.read_bytes() == before
    assert child_count.read_text(encoding="ascii").splitlines() == ["child"]


def test_finalizer_internal_failure_keeps_original_exit_and_secondary_evidence(tmp_path):
    from sparse_rtdetr.baseline import smoke_outer_launcher as launcher

    args, child_count, _tmux_count = _repo(tmp_path, child_rc=99)
    finalizer = _script(tmp_path / "synthetic-finalizer", f'exec {launcher.sys.executable} -c "raise RuntimeError(\\"synthetic finalizer failure\\")"')
    with mock.patch.object(launcher.sys, "executable", str(finalizer)):
        run_outer_launch(**args)
    pane = args["outer_evidence_dir"] / "pane"
    secondary = json.loads((pane / "pane_secondary_finalization_failure.json").read_text(encoding="utf-8"))
    assert (pane / "pane_exit_code.txt").read_bytes() == b"99\n"
    assert (pane / "pane_error.json").is_file()
    assert secondary["status"] == "FAILED_FINALIZATION"
    assert secondary["failure_class"] == "PANE_FINALIZATION_FAILURE"
    assert secondary["original_inner_returncode"] == 99
    assert not (pane / "pane_completion.json").exists()
    assert child_count.read_text(encoding="ascii").splitlines() == ["child"]
    assert classify_outer_evidence(args["outer_evidence_dir"], args["output_dir"], args["process_evidence_dir"]) == "PANE_FINALIZATION_FAILED"


def test_missing_finalizer_executable_keeps_original_exit_and_secondary_evidence(tmp_path):
    from sparse_rtdetr.baseline import smoke_outer_launcher as launcher

    args, child_count, _tmux_count = _repo(tmp_path, child_rc=99)
    finalizer = _script(tmp_path / "synthetic-finalizer", "exit 0")
    tmux = args["tmux_executable"]
    tmux.write_text(
        "#!/bin/sh\n"
        f"echo tmux >> {tmp_path / 'tmux-missing.count'}\n"
        f"mv {finalizer} {finalizer}.missing\n"
        '"$7"\n'
        "exit 0\n",
        encoding="ascii",
    )
    tmux.chmod(0o755)
    with mock.patch.object(launcher.sys, "executable", str(finalizer)):
        run_outer_launch(**args)
    pane = args["outer_evidence_dir"] / "pane"
    secondary = json.loads((pane / "pane_secondary_finalization_failure.json").read_text(encoding="utf-8"))
    assert secondary["finalizer_returncode"] == 127
    assert secondary["original_inner_returncode"] == 99
    assert (pane / "pane_error.json").is_file()
    assert (pane / "pane_exit_code.txt").read_bytes() == b"99\n"
    assert not (pane / "pane_completion.json").exists()
    assert child_count.read_text(encoding="ascii").splitlines() == ["child"]
    assert classify_outer_evidence(args["outer_evidence_dir"], args["output_dir"], args["process_evidence_dir"]) == "PANE_FINALIZATION_FAILED"


def test_preinner_binding_failure_and_finalizer_failure_are_both_durable(tmp_path):
    from sparse_rtdetr.baseline import smoke_outer_launcher as launcher

    args, child_count, _tmux_count = _repo(tmp_path, child_rc=99)
    finalizer = _script(tmp_path / "synthetic-finalizer", f'exec {launcher.sys.executable} -c "raise RuntimeError(\\"synthetic finalizer failure\\")"')
    tmux = args["tmux_executable"]
    tmux.write_text(
        "#!/bin/sh\n"
        'PANE_DIR=$(dirname "$7")\n'
        'PLAN_MODE=$(stat -c %a "$PANE_DIR/pane_plan.json")\n'
        'chmod 000 "$PANE_DIR/pane_plan.json"\n'
        '"$7"\n'
        'RC=$?\n'
        'chmod "$PLAN_MODE" "$PANE_DIR/pane_plan.json"\n'
        "exit 0\n",
        encoding="ascii",
    )
    tmux.chmod(0o755)
    with mock.patch.object(launcher.sys, "executable", str(finalizer)):
        run_outer_launch(**args)
    pane = args["outer_evidence_dir"] / "pane"
    error = json.loads((pane / "pane_error.json").read_text(encoding="utf-8"))
    secondary = json.loads((pane / "pane_secondary_finalization_failure.json").read_text(encoding="utf-8"))
    assert error["failure_class"] == "PREINNER_BINDING_FAILURE"
    assert error["inner_started"] is False
    assert error["inner_returncode"] == 125
    assert secondary["original_inner_returncode"] == 125
    assert secondary["inner_started"] is False
    assert (pane / "pane_exit_code.txt").read_bytes() == b"125\n"
    assert not (pane / "pane_completion.json").exists()
    assert not child_count.exists()
    assert classify_outer_evidence(args["outer_evidence_dir"], args["output_dir"], args["process_evidence_dir"]) == "PANE_FINALIZATION_FAILED"
