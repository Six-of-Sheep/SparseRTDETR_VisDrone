from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from sparse_rtdetr.baseline.smoke import load_smoke_config
from sparse_rtdetr.baseline.smoke_outer_launcher import (
    classify_outer_evidence,
    contract_check,
    run_outer_launch,
)


ROOT = Path(__file__).resolve().parents[1]
V1_CONFIG = ROOT / "configs/baseline/rtdetrv2_r18_visdrone_smoke_v1.json"
V2_CONFIG = ROOT / "configs/baseline/rtdetrv2_r18_visdrone_smoke_v2.json"


def _repo(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path, Path]:
    root = tmp_path / "repo"
    (root / "configs/baseline").mkdir(parents=True)
    (root / "artifacts/data/visdrone_protocol_v2_conversion_r3").mkdir(parents=True)
    shutil.copyfile(V2_CONFIG, root / V2_CONFIG.relative_to(ROOT))
    config = json.loads(V2_CONFIG.read_text(encoding="utf-8"))
    (root / "artifacts/data/visdrone_protocol_v2_conversion_r3/train_core_manifest.json").write_text(
        json.dumps({"records": config["image_selection"]["records"]}), encoding="utf-8"
    )
    data = tmp_path / "data"
    data.mkdir()
    output = root / "artifacts/runs/rtdetrv2_r18_visdrone_baseline_smoke_r2"
    process = root / "artifacts/process_evidence/rtdetrv2_r18_visdrone_baseline_smoke_r2"
    outer_parent = root / "artifacts/outer_launch_evidence"
    output.parent.mkdir(parents=True)
    process.parent.mkdir(parents=True)
    outer_parent.mkdir(parents=True)
    outer_parent.chmod(0o775)
    return root, data, root / V2_CONFIG.relative_to(ROOT), output, process, outer_parent / "rtdetrv2_r18_visdrone_baseline_smoke_r2"


def _executable(path: Path, body: str = "#!/bin/sh\nexit 37\n") -> Path:
    path.write_text(body, encoding="ascii")
    path.chmod(0o755)
    return path


def _launch_args(tmp_path: Path) -> tuple[dict[str, Path | str], Path]:
    root, data, config, output, process, outer = _repo(tmp_path)
    child = _executable(tmp_path / "fake-child")
    tmux = _executable(tmp_path / "fake-tmux")
    return {
        "repo_root": root,
        "data_root": data,
        "config_path": config,
        "output_dir": output,
        "process_evidence_dir": process,
        "outer_evidence_dir": outer,
        "child_python": child,
        "tmux_executable": tmux,
        "tmux_session": "p3_rtdetrv2_r18_visdrone_baseline_smoke_r2",
    }, child


def test_v1_config_is_preserved_and_scientific_contract_matches_v2():
    v1_bytes = V1_CONFIG.read_bytes()
    assert hashlib.sha256(v1_bytes).hexdigest() == "76562afe145a7ec1049497176da6a91f6513eb4244a737335c1c12ea7c5f55d2"
    v1 = load_smoke_config(ROOT, V1_CONFIG)
    v2 = load_smoke_config(ROOT, V2_CONFIG)
    for field in ("execution", "model", "image_selection", "evidence", "mode", "split_role"):
        assert v1[field] == v2[field]
    for field in ("authorized_env", "visible_devices", "device_count", "cpu_fallback"):
        assert v1["runtime"][field] == v2["runtime"][field]
    assert v1["smoke_id"] != v2["smoke_id"]
    assert v1["runtime"]["output_relative_path"] != v2["runtime"]["output_relative_path"]


def test_v1_and_v2_config_paths_cannot_be_mixed(tmp_path):
    root, _data, _config, _output, _process, _outer = _repo(tmp_path)
    mixed = root / "configs/baseline/rtdetrv2_r18_visdrone_smoke_v2.json"
    mixed.write_bytes(V1_CONFIG.read_bytes())
    with pytest.raises(Exception):
        load_smoke_config(root, mixed)


def test_outer_contract_check_is_data_free_and_does_not_create_evidence(tmp_path):
    args, _child = _launch_args(tmp_path)
    result = contract_check(args["repo_root"], args["config_path"])
    assert result["status"] == "PASS"
    assert result["tmux_called"] is False
    assert result["outer_evidence_created"] is False
    assert not args["outer_evidence_dir"].exists()


def test_tmux_nonzero_result_is_persisted_once_and_rejected(tmp_path):
    args, _child = _launch_args(tmp_path)
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=17, stdout=b"tmux-out\n", stderr=b"tmux-err\n")

    with mock.patch("sparse_rtdetr.baseline.smoke_outer_launcher.subprocess.run", side_effect=fake_run):
        result = run_outer_launch(**args)
    launcher = args["outer_evidence_dir"] / "launcher"
    assert result["status"] == "TMUX_REJECTED"
    assert len(calls) == 1
    assert calls[0][0][1:4] == ["new-session", "-d", "-s"]
    assert "-c" in calls[0][0]
    assert len(calls[0][0][-1]) < 300
    assert "sparse_rtdetr.baseline.smoke_launcher" not in calls[0][0][-1]
    assert "tmux-out\n" == (launcher / "tmux_stdout.log").read_text()
    assert "tmux-err\n" == (launcher / "tmux_stderr.log").read_text()
    assert not args["output_dir"].exists()
    assert not args["process_evidence_dir"].exists()
    assert classify_outer_evidence(args["outer_evidence_dir"], args["output_dir"], args["process_evidence_dir"]) == "TMUX_REJECTED"


def test_tmux_zero_without_pane_is_distinct_from_inner_start(tmp_path):
    args, _child = _launch_args(tmp_path)

    with mock.patch(
        "sparse_rtdetr.baseline.smoke_outer_launcher.subprocess.run",
        return_value=SimpleNamespace(returncode=0, stdout=b"", stderr=b""),
    ) as run:
        result = run_outer_launch(**args)
    assert result["status"] == "TMUX_ACCEPTED"
    assert run.call_count == 1
    assert classify_outer_evidence(args["outer_evidence_dir"], args["output_dir"], args["process_evidence_dir"]) == "TMUX_ACCEPTED_PANE_NOT_STARTED"


def test_pane_writes_start_and_inner_exit_before_process_evidence(tmp_path):
    args, fake_child = _launch_args(tmp_path)
    real_run = subprocess.run

    def fake_tmux(argv, **kwargs):
        wrapper = Path(argv[-1])
        pane_result = real_run([str(wrapper)], cwd=kwargs["cwd"], env=kwargs["env"], capture_output=True, check=False)
        return SimpleNamespace(returncode=0, stdout=pane_result.stdout, stderr=pane_result.stderr)

    with mock.patch("sparse_rtdetr.baseline.smoke_outer_launcher.subprocess.run", side_effect=fake_tmux):
        result = run_outer_launch(**args)
    pane = args["outer_evidence_dir"] / "pane"
    assert result["status"] == "TMUX_ACCEPTED"
    assert (pane / "pane_start.json").is_file()
    assert (pane / "pane_receipt.json").is_file()
    assert (pane / "pane_exit_code.txt").read_text() == "37\n"
    assert "P3_SMOKE_PANE_SHELL_STARTED" in (pane / "pane_console.log").read_text()
    assert (pane / "pane_inventory.json").is_file()
    assert (args["outer_evidence_dir"] / "launcher" / "outer_timing.json").is_file()
    assert not set(p.name for p in (args["outer_evidence_dir"] / "launcher").iterdir()) & set(p.name for p in pane.iterdir())
    assert classify_outer_evidence(args["outer_evidence_dir"], args["output_dir"], args["process_evidence_dir"]) == "INNER_STARTED_PROCESS_EVIDENCE_ABSENT"
    assert not args["output_dir"].exists()
    assert not args["process_evidence_dir"].exists()
    assert fake_child.exists()


def test_preflight_failure_persists_without_tmux_call(tmp_path):
    args, _child = _launch_args(tmp_path)
    args["child_python"] = tmp_path / "missing-child"
    with mock.patch("sparse_rtdetr.baseline.smoke_outer_launcher.subprocess.run") as run:
        result = run_outer_launch(**args)
    launcher = args["outer_evidence_dir"] / "launcher"
    assert result["status"] == "PREFLIGHT_FAILED"
    assert run.call_count == 0
    assert (launcher / "outer_invocation.json").is_file()
    assert (launcher / "outer_error.json").is_file()
    assert (launcher / "outer_completion.json").is_file()
    assert classify_outer_evidence(args["outer_evidence_dir"], args["output_dir"], args["process_evidence_dir"]) == "PREFLIGHT_FAILED"


def test_prepared_config_mutation_fails_in_pane_before_inner(tmp_path):
    args, _child = _launch_args(tmp_path)
    real_run = subprocess.run

    def fake_tmux(argv, **kwargs):
        config = Path(args["config_path"])
        config.write_bytes(config.read_bytes() + b" ")
        pane_result = real_run([argv[-1]], cwd=kwargs["cwd"], env=kwargs["env"], capture_output=True, check=False)
        return SimpleNamespace(returncode=0, stdout=pane_result.stdout, stderr=pane_result.stderr)

    with mock.patch("sparse_rtdetr.baseline.smoke_outer_launcher.subprocess.run", side_effect=fake_tmux):
        run_outer_launch(**args)
    pane = args["outer_evidence_dir"] / "pane"
    assert (pane / "pane_start.json").is_file()
    assert (pane / "pane_error.json").is_file()
    assert (pane / "pane_exit_code.txt").read_text() == "125\n"
    assert classify_outer_evidence(args["outer_evidence_dir"], args["output_dir"], args["process_evidence_dir"]) == "INNER_STARTED_PROCESS_EVIDENCE_ABSENT"
