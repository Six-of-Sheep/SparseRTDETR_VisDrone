from __future__ import annotations

import hashlib
import json
import os
import signal
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from unittest import mock

import pytest

from sparse_rtdetr.baseline.smoke import (
    SMOKE_V2_ID,
    SMOKE_V3_ID,
    SMOKE_V4_ID,
    SMOKE_V5_ID,
    load_smoke_config,
)
from sparse_rtdetr.baseline.smoke_evidence import canonical_json_bytes, inventory, sha256_bytes, sha256_file
from sparse_rtdetr.baseline.smoke_launcher import launch_smoke
from sparse_rtdetr.baseline.smoke_outer_launcher import (
    OUTER_EXCLUDED,
    PANE_EXCLUDED,
    PANE_FINALIZER_EXIT,
    OuterLaunchError,
    executable_identity,
    _finish_outer,
    _run_tmux,
    classify_outer_evidence,
    run_outer_launch,
    validate_outer_evidence,
    validate_pane_evidence,
    validate_process_evidence,
    contract_check,
)


ROOT = Path(__file__).resolve().parents[1]
V1_CONFIG = ROOT / "configs/baseline/rtdetrv2_r18_visdrone_smoke_v1.json"
V2_CONFIG = ROOT / "configs/baseline/rtdetrv2_r18_visdrone_smoke_v2.json"
V3_CONFIG = ROOT / "configs/baseline/rtdetrv2_r18_visdrone_smoke_v3.json"
V4_CONFIG = ROOT / "configs/baseline/rtdetrv2_r18_visdrone_smoke_v4.json"
V5_CONFIG = ROOT / "configs/baseline/rtdetrv2_r18_visdrone_smoke_v5.json"
SESSION = "p3_rtdetrv2_r18_visdrone_baseline_smoke_r2"
V3_SESSION = "p3_rtdetrv2_r18_visdrone_baseline_smoke_r3"
V4_SESSION = "p3_rtdetrv2_r18_visdrone_baseline_smoke_r4"
PYTHON = Path(sys.executable).resolve()


def _script(path: Path, body: str) -> Path:
    path.write_text("#!/bin/sh\n" + body + "\n", encoding="ascii")
    path.chmod(0o755)
    return path


def _repo(tmp_path: Path, *, child_rc: int = 0, version: int = 2) -> tuple[dict[str, Path | str], Path, Path]:
    root = tmp_path / "repo"
    shutil.copytree(ROOT / "src", root / "src")
    (root / "configs/baseline").mkdir(parents=True)
    (root / "artifacts/data/visdrone_protocol_v2_conversion_r3").mkdir(parents=True)
    config_source = {2: V2_CONFIG, 3: V3_CONFIG, 4: V4_CONFIG, 5: V5_CONFIG}[version]
    config_name = config_source.name
    shutil.copyfile(config_source, root / "configs/baseline" / config_name)
    config = json.loads(config_source.read_text(encoding="utf-8"))
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
    suffix = {2: "r2", 3: "r3", 4: "r4", 5: "r5"}[version]
    output = root / f"artifacts/runs/rtdetrv2_r18_visdrone_baseline_smoke_{suffix}"
    process = root / f"artifacts/process_evidence/rtdetrv2_r18_visdrone_baseline_smoke_{suffix}"
    outer = outer_parent / f"rtdetrv2_r18_visdrone_baseline_smoke_{suffix}"
    args: dict[str, Path | str] = {
        "repo_root": root,
        "data_root": data,
        "config_path": root / "configs/baseline" / config_name,
        "output_dir": output,
        "process_evidence_dir": process,
        "outer_evidence_dir": outer,
        "child_python": child,
        "tmux_executable": tmux,
        "tmux_session": {2: SESSION, 3: V3_SESSION, 4: V4_SESSION, 5: "p3_rtdetrv2_r18_visdrone_baseline_smoke_r5"}[version],
    }
    return args, child_count, tmux_count


def _entry_completion() -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "COMPLETED",
        "mode": "smoke",
        "smoke_pass_candidate": True,
        "non_training": True,
        "inference_only": True,
        "model_eval": True,
        "torch_no_grad": True,
        "non_selection": True,
        "non_reportable": True,
        "formal_resume_eligible": False,
        "formal_training_eligible": False,
        "baseline_training_ready": False,
        "confirmatory_metrics_accessed": False,
        "dataset_test_accessed_by_this_process": False,
        "speed_measurement": False,
        "config_sha256": "",
        "total_predictions": 600,
        "images_requested": 2,
        "images_loaded": 2,
        "batches_requested": 1,
        "batches_processed": 1,
        "dataset_next_calls": 1,
        "model_forward_calls": 1,
        "postprocessor_calls": 1,
        "evaluator_calls": 0,
        "criterion_calls": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "scheduler_steps": 0,
        "checkpoint_loads": 0,
        "network_calls": 0,
    }


def _write_real_child(path: Path) -> None:
    completion = repr(_entry_completion())
    path.write_text(
        "\n".join([
            "import json, os",
            "from pathlib import Path",
            "from sparse_rtdetr.baseline.smoke_evidence import SmokeEvidence",
            "from sparse_rtdetr.baseline.smoke_launcher import _consume_handoff",
            "output = Path(os.environ['P3_SMOKE_OUTPUT_DIR'])",
            "receipt, receipt_sha = _consume_handoff(Path(os.environ['P3_SMOKE_REPO_ROOT']))",
            "config = json.loads(Path(receipt['config_path']).read_text(encoding='utf-8'))",
            "evidence = SmokeEvidence(output)",
            "config_sha = evidence.write_config(config)",
            "invocation = {'schema_version': 1, 'mode': 'real', 'smoke_id': receipt['smoke_id'], 'config_sha256': config_sha, 'config_relative_path': receipt['config_relative_path'], 'config_size_bytes': (output / 'config.json').stat().st_size, 'nonce': receipt['nonce'], 'launcher_pid': receipt['launcher_pid'], 'child_pid': receipt['child_pid'], 'child_ppid': receipt['child_ppid'], 'child_argv_sha256': receipt['child_argv_sha256'], 'handoff_receipt_sha256': receipt_sha, 'handoff_receipt_relative_path': 'handoff_receipt.json'}",
            "evidence.write_json('invocation.json', invocation)",
            "[evidence.write_json(name, {}) for name in ('source_identity.json', 'data_binding_audit.json', 'image_selection_audit.json', 'cuda_runtime_identity.json', 'model_identity.json', 'call_audit.json', 'input_batch_audit.json', 'model_output_audit.json', 'postprocess_audit.json', 'rng_audit.json')]",
            "completion = " + completion,
            "completion['config_sha256'] = config_sha",
            "evidence.finalize_success(completion)",
        ]) + "\n",
        encoding="utf-8",
    )


def _repack_process(process: Path) -> None:
    excluded = frozenset({"process_inventory.json", "process_completion.json", "process_partial_inventory.json"})
    value = inventory(process, excluded)
    payload = canonical_json_bytes(value)
    inventory_path = process / "process_inventory.json"
    inventory_path.write_bytes(payload)
    completion_path = process / "process_completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["process_inventory"] = {"relative_path": "process_inventory.json", "size_bytes": inventory_path.stat().st_size, "sha256": sha256_file(inventory_path)}
    completion_path.write_bytes(canonical_json_bytes(completion))


def _complete_v3_chain(tmp_path: Path, monkeypatch, *, version: int = 3) -> dict[str, Path | str]:
    args, _child_count, _tmux_count = _repo(tmp_path, version=version)
    assert run_outer_launch(**args)["status"] == "TMUX_ACCEPTED"
    monkeypatch.setenv("P3_RTDETR_BASELINE_SMOKE_AUTHORIZED", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    child = tmp_path / "real-entry-child.py"
    _write_real_child(child)
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "P3_RTDETR_BASELINE_SMOKE_AUTHORIZED": "1",
        "PYTHONPATH": str(ROOT / "src"),
    }
    pane = args["outer_evidence_dir"] / "pane"
    pane_receipt = json.loads((pane / "pane_receipt.json").read_text(encoding="utf-8"))
    assert launch_smoke(
        [str(PYTHON), str(child)],
        args["output_dir"],
        args["process_evidence_dir"],
        data_root=args["data_root"],
        child_python=PYTHON,
        repo_root=args["repo_root"],
        env=env,
        nonce=pane_receipt["nonce"],
        config_path=args["config_path"],
    ) == 0
    return args


def _complete_v4_chain(tmp_path: Path, monkeypatch) -> dict[str, Path | str]:
    return _complete_v3_chain(tmp_path, monkeypatch, version=4)


def _complete_v5_chain(tmp_path: Path, monkeypatch) -> dict[str, Path | str]:
    return _complete_v3_chain(tmp_path, monkeypatch, version=5)


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


def test_v3_outer_synthetic_full_path_uses_only_v3_identity(tmp_path):
    args, _child_count, tmux_count = _repo(tmp_path, version=3)
    result = run_outer_launch(**args)
    assert result["status"] == "TMUX_ACCEPTED"
    assert tmux_count.read_text(encoding="ascii").splitlines() == ["tmux"]
    outer = validate_outer_evidence(args["outer_evidence_dir"], args["output_dir"], args["process_evidence_dir"])
    assert outer["invocation"]["smoke_id"] == SMOKE_V3_ID
    assert outer["invocation"]["tmux_session"] == V3_SESSION
    assert outer["invocation"]["output_dir"] == str(args["output_dir"])
    assert outer["invocation"]["process_evidence_dir"] == str(args["process_evidence_dir"])
    assert classify_outer_evidence(args["outer_evidence_dir"], args["output_dir"], args["process_evidence_dir"]) == "INNER_STARTED_PROCESS_EVIDENCE_ABSENT"


def test_v4_outer_contract_check_is_data_free_and_uses_only_v4_identity(tmp_path):
    args, _child_count, _tmux_count = _repo(tmp_path, version=4)
    result = contract_check(args["repo_root"], args["config_path"])
    assert result["status"] == "PASS"
    assert result["smoke_id"] == SMOKE_V4_ID
    assert result["tmux_called"] is False
    assert result["torch_imported"] is False
    assert not args["outer_evidence_dir"].exists()


def test_v5_outer_contract_check_is_data_free_and_uses_only_v5_identity(tmp_path):
    args, _child_count, _tmux_count = _repo(tmp_path, version=5)
    result = contract_check(args["repo_root"], args["config_path"])
    assert result["status"] == "PASS"
    assert result["smoke_id"] == SMOKE_V5_ID
    assert result["config_binding"]["config_relative_path"] == "configs/baseline/rtdetrv2_r18_visdrone_smoke_v5.json"
    assert result["tmux_called"] is False
    assert result["torch_imported"] is False
    assert not args["outer_evidence_dir"].exists()


def test_v4_outer_synthetic_full_path_uses_only_v4_identity(tmp_path):
    args, _child_count, tmux_count = _repo(tmp_path, version=4)
    result = run_outer_launch(**args)
    assert result["status"] == "TMUX_ACCEPTED"
    assert tmux_count.read_text(encoding="ascii").splitlines() == ["tmux"]
    outer = validate_outer_evidence(args["outer_evidence_dir"], args["output_dir"], args["process_evidence_dir"])
    assert outer["invocation"]["smoke_id"] == SMOKE_V4_ID
    assert outer["invocation"]["tmux_session"] == V4_SESSION
    assert outer["invocation"]["output_dir"] == str(args["output_dir"])
    assert outer["invocation"]["process_evidence_dir"] == str(args["process_evidence_dir"])
    assert classify_outer_evidence(args["outer_evidence_dir"], args["output_dir"], args["process_evidence_dir"]) == "INNER_STARTED_PROCESS_EVIDENCE_ABSENT"


def test_v5_outer_synthetic_full_path_uses_only_v5_identity(tmp_path):
    args, _child_count, tmux_count = _repo(tmp_path, version=5)
    result = run_outer_launch(**args)
    assert result["status"] == "TMUX_ACCEPTED"
    assert tmux_count.read_text(encoding="ascii").splitlines() == ["tmux"]
    outer = validate_outer_evidence(args["outer_evidence_dir"], args["output_dir"], args["process_evidence_dir"])
    assert outer["invocation"]["smoke_id"] == SMOKE_V5_ID
    assert outer["invocation"]["tmux_session"] == "p3_rtdetrv2_r18_visdrone_baseline_smoke_r5"
    binding = json.loads((args["outer_evidence_dir"] / "launcher/outer_config_binding.json").read_text(encoding="utf-8"))
    assert binding["config_relative_path"] == "configs/baseline/rtdetrv2_r18_visdrone_smoke_v5.json"
    assert classify_outer_evidence(args["outer_evidence_dir"], args["output_dir"], args["process_evidence_dir"]) == "INNER_STARTED_PROCESS_EVIDENCE_ABSENT"


def test_v3_classifier_accepts_complete_synthetic_entry_and_process(tmp_path, monkeypatch):
    args = _complete_v3_chain(tmp_path, monkeypatch)
    assert classify_outer_evidence(args["outer_evidence_dir"], args["output_dir"], args["process_evidence_dir"]) == "TERMINAL_COMPLETE"


def test_v4_classifier_accepts_complete_synthetic_entry_and_process(tmp_path, monkeypatch):
    args = _complete_v4_chain(tmp_path, monkeypatch)
    assert classify_outer_evidence(args["outer_evidence_dir"], args["output_dir"], args["process_evidence_dir"]) == "TERMINAL_COMPLETE"
    completion = json.loads((args["process_evidence_dir"] / "process_completion.json").read_text(encoding="utf-8"))
    assert completion["smoke_id"] == SMOKE_V4_ID
    assert completion["config_relative_path"] == "configs/baseline/rtdetrv2_r18_visdrone_smoke_v4.json"


def test_v5_classifier_accepts_complete_synthetic_entry_and_process(tmp_path, monkeypatch):
    args = _complete_v5_chain(tmp_path, monkeypatch)
    assert classify_outer_evidence(args["outer_evidence_dir"], args["output_dir"], args["process_evidence_dir"]) == "TERMINAL_COMPLETE"
    completion = json.loads((args["process_evidence_dir"] / "process_completion.json").read_text(encoding="utf-8"))
    assert completion["smoke_id"] == SMOKE_V5_ID
    assert completion["config_relative_path"] == "configs/baseline/rtdetrv2_r18_visdrone_smoke_v5.json"


def test_v4_main_propagates_pane_nonce_through_complete_synthetic_chain(tmp_path, monkeypatch):
    from sparse_rtdetr.baseline import smoke_launcher as launcher

    args, _child_count, _tmux_count = _repo(tmp_path, version=4)
    assert run_outer_launch(**args)["status"] == "TMUX_ACCEPTED"
    pane_receipt = json.loads((args["outer_evidence_dir"] / "pane/pane_receipt.json").read_text(encoding="utf-8"))
    child = tmp_path / "real-entry-child.py"
    _write_real_child(child)
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "P3_RTDETR_BASELINE_SMOKE_AUTHORIZED": "1",
        "PYTHONPATH": str(ROOT / "src"),
    }
    monkeypatch.setenv("P3_PANE_NONCE", pane_receipt["nonce"])
    monkeypatch.setenv("P3_RTDETR_BASELINE_SMOKE_AUTHORIZED", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    monkeypatch.setattr(launcher, "validate_real_smoke_environment", lambda: {"cpu_fallback": False})
    real_launch = launcher.launch_smoke
    captured = {}

    def launch_with_real_entry(child_argv, output_dir, process_dir, **kwargs):
        captured["nonce"] = kwargs["nonce"]
        return real_launch(
            [str(PYTHON), str(child)],
            output_dir,
            process_dir,
            data_root=args["data_root"],
            child_python=PYTHON,
            repo_root=args["repo_root"],
            env=env,
            nonce=kwargs["nonce"],
            config_path=args["config_path"],
        )

    monkeypatch.setattr(launcher, "launch_smoke", launch_with_real_entry)
    assert launcher.main([
        "smoke",
        "--repo-root", str(args["repo_root"]),
        "--data-root", str(args["data_root"]),
        "--output-dir", str(args["output_dir"]),
        "--process-evidence-dir", str(args["process_evidence_dir"]),
        "--config", str(args["config_path"]),
        "--child-python", str(PYTHON),
    ]) == 0
    outer = validate_outer_evidence(args["outer_evidence_dir"], args["output_dir"], args["process_evidence_dir"])
    pane = validate_pane_evidence(args["outer_evidence_dir"] / "pane", outer)
    process = validate_process_evidence(args["process_evidence_dir"], pane["receipt"])
    entry_invocation = json.loads((args["output_dir"] / "invocation.json").read_text(encoding="utf-8"))
    prepared = json.loads((args["process_evidence_dir"] / "handoff_prepared.json").read_text(encoding="utf-8"))
    nonces = {
        outer["invocation"]["nonce"],
        outer["completion"]["nonce"],
        pane["plan"]["nonce"],
        pane["receipt"]["nonce"],
        pane["completion"]["nonce"],
        prepared["nonce"],
        process["receipt"]["nonce"],
        process["invocation"]["nonce"],
        process["completion"]["nonce"],
        entry_invocation["nonce"],
        captured["nonce"],
    }
    assert nonces == {pane_receipt["nonce"]}
    assert classify_outer_evidence(args["outer_evidence_dir"], args["output_dir"], args["process_evidence_dir"]) == "TERMINAL_COMPLETE"


def test_v4_identity_drift_is_terminal_failure(tmp_path, monkeypatch):
    args = _complete_v4_chain(tmp_path, monkeypatch)
    completion_path = args["process_evidence_dir"] / "process_completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["smoke_id"] = SMOKE_V3_ID
    completion_path.write_bytes(canonical_json_bytes(completion))
    assert classify_outer_evidence(args["outer_evidence_dir"], args["output_dir"], args["process_evidence_dir"]) == "TERMINAL_FAILED"


def _repack_entry(output: Path) -> None:
    excluded = frozenset({"artifact_inventory.json", "completion.json"})
    inventory_path = output / "artifact_inventory.json"
    payload = canonical_json_bytes(inventory(output, excluded))
    inventory_path.write_bytes(payload)
    completion_path = output / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["artifact_inventory_sha256"] = sha256_bytes(payload)
    completion_path.write_bytes(canonical_json_bytes(completion))


@pytest.mark.parametrize("field", [
    "outer_smoke_id", "pane_smoke_id", "process_smoke_id", "entry_smoke_id",
    "config_relative_path", "outer_path", "process_path", "tmux_session",
    "source_config_sha", "canonical_config_sha", "nonce", "receipt_sha",
])
def test_v5_identity_drift_never_reaches_terminal_complete(tmp_path, monkeypatch, field):
    args = _complete_v5_chain(tmp_path, monkeypatch)
    if field == "outer_smoke_id":
        path = args["outer_evidence_dir"] / "launcher/outer_invocation.json"
        value = json.loads(path.read_text(encoding="utf-8")); value["smoke_id"] = SMOKE_V4_ID; path.write_bytes(canonical_json_bytes(value)); _repack_outer(args["outer_evidence_dir"])
    elif field == "pane_smoke_id":
        path = args["outer_evidence_dir"] / "pane/pane_receipt.json"
        value = json.loads(path.read_text(encoding="utf-8")); value["smoke_id"] = SMOKE_V4_ID; path.write_bytes(canonical_json_bytes(value)); _repack_pane(args["outer_evidence_dir"] / "pane")
    elif field == "entry_smoke_id":
        path = args["output_dir"] / "invocation.json"
        value = json.loads(path.read_text(encoding="utf-8")); value["smoke_id"] = SMOKE_V4_ID; path.write_bytes(canonical_json_bytes(value)); _repack_entry(args["output_dir"])
    elif field in {"outer_path", "process_path", "tmux_session"}:
        path = args["outer_evidence_dir"] / "launcher/outer_invocation.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        mutations = {
            "outer_path": ("outer_evidence_dir", "/tmp/forged-outer"),
            "process_path": ("process_evidence_dir", "/tmp/forged-process"),
            "tmux_session": ("tmux_session", "p3_rtdetrv2_r18_visdrone_baseline_smoke_r4"),
        }
        key, forged = mutations[field]; value[key] = forged; path.write_bytes(canonical_json_bytes(value)); _repack_outer(args["outer_evidence_dir"])
    else:
        path = args["process_evidence_dir"] / "process_completion.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        mutations = {
            "process_smoke_id": ("smoke_id", SMOKE_V4_ID),
            "config_relative_path": ("config_relative_path", "configs/baseline/rtdetrv2_r18_visdrone_smoke_v4.json"),
            "outer_path": ("outer_evidence_dir", "/tmp/forged-outer"),
            "process_path": ("process_evidence_dir", "/tmp/forged-process"),
            "tmux_session": ("tmux_session", "p3_rtdetrv2_r18_visdrone_baseline_smoke_r4"),
            "source_config_sha": ("config_file_sha256", "0" * 64),
            "canonical_config_sha": ("config_canonical_sha256", "0" * 64),
            "nonce": ("nonce", "0" * 32),
            "receipt_sha": ("handoff_receipt_sha256", "0" * 64),
        }
        key, forged = mutations[field]; value[key] = forged; path.write_bytes(canonical_json_bytes(value)); _repack_process(args["process_evidence_dir"])
    assert classify_outer_evidence(args["outer_evidence_dir"], args["output_dir"], args["process_evidence_dir"]) != "TERMINAL_COMPLETE"


def test_v4_preexisting_output_is_rejected_before_outer_creation(tmp_path):
    args, _child_count, _tmux_count = _repo(tmp_path, version=4)
    args["output_dir"].mkdir()
    result = run_outer_launch(**args)
    assert result["status"] == "PREFLIGHT_FAILED"
    assert args["outer_evidence_dir"].is_dir()
    assert not args["process_evidence_dir"].exists()
    assert classify_outer_evidence(args["outer_evidence_dir"], args["output_dir"], args["process_evidence_dir"]) == "PREFLIGHT_FAILED"


@pytest.mark.parametrize("kind", ["output", "process", "outer", "symlink", "file", "directory"])
def test_v5_preexisting_targets_are_rejected_before_fake_tmux(tmp_path, kind):
    args, _child_count, tmux_count = _repo(tmp_path, version=5)
    target = {"output": args["output_dir"], "process": args["process_evidence_dir"], "outer": args["outer_evidence_dir"]}.get(kind, args["output_dir"])
    if kind == "symlink":
        real = tmp_path / "existing-output"
        real.mkdir()
        target.symlink_to(real, target_is_directory=True)
    elif kind == "file":
        target.write_text("existing", encoding="ascii")
    elif kind == "directory":
        target.mkdir()
    else:
        target.mkdir()
    if kind == "outer":
        with pytest.raises(OuterLaunchError):
            run_outer_launch(**args)
        assert not tmux_count.exists()
        return
    result = run_outer_launch(**args)
    assert result["status"] == "PREFLIGHT_FAILED"
    assert not tmux_count.exists()


@pytest.mark.parametrize("mutation", [
    "prepared_canonical",
    "receipt_canonical",
    "invocation_canonical",
    "completion_canonical",
    "prepared_smoke_id",
    "receipt_smoke_id",
    "invocation_smoke_id",
    "completion_smoke_id",
    "prepared_relative",
    "receipt_relative",
    "invocation_relative",
    "completion_relative",
    "swap_source_and_canonical",
    "prepared_bool_size",
    "completion_invalid_sha",
])
def test_v3_process_config_binding_mutations_never_complete(tmp_path, monkeypatch, mutation):
    args = _complete_v3_chain(tmp_path, monkeypatch)
    process = args["process_evidence_dir"]
    if mutation == "completion_canonical":
        path = process / "process_completion.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["config_canonical_sha256"] = "0" * 64
        path.write_bytes(canonical_json_bytes(value))
    elif mutation == "completion_invalid_sha":
        path = process / "process_completion.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["config_file_sha256"] = "invalid"
        path.write_bytes(canonical_json_bytes(value))
    elif mutation == "swap_source_and_canonical":
        path = process / "handoff_receipt.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["config_file_sha256"], value["config_canonical_sha256"] = value["config_canonical_sha256"], value["config_file_sha256"]
        path.write_bytes(canonical_json_bytes(value))
        _repack_process(process)
    elif mutation == "prepared_bool_size":
        path = process / "handoff_prepared.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["config_size_bytes"] = True
        path.write_bytes(canonical_json_bytes(value))
        _repack_process(process)
    else:
        layer, field = mutation.split("_", 1)
        filename = {
            "prepared": "handoff_prepared.json",
            "receipt": "handoff_receipt.json",
            "invocation": "process_invocation.json",
            "completion": "process_completion.json",
        }[layer]
        path = process / filename
        value = json.loads(path.read_text(encoding="utf-8"))
        if field == "canonical":
            value["config_canonical_sha256"] = "0" * 64
        elif field == "smoke_id":
            value["smoke_id"] = SMOKE_V2_ID
        elif field == "relative":
            value["config_relative_path"] = V2_CONFIG.relative_to(ROOT).as_posix()
        path.write_bytes(canonical_json_bytes(value))
        if layer != "completion":
            _repack_process(process)
    assert classify_outer_evidence(args["outer_evidence_dir"], args["output_dir"], process) == "TERMINAL_FAILED"


def test_process_expected_identity_requires_pane_config_fields(tmp_path, monkeypatch):
    args = _complete_v3_chain(tmp_path, monkeypatch)
    process = args["process_evidence_dir"]
    pane_receipt = json.loads((args["outer_evidence_dir"] / "pane/pane_receipt.json").read_text(encoding="utf-8"))
    pane_receipt.pop("config_canonical_sha256")
    with pytest.raises(OuterLaunchError):
        validate_process_evidence(process, pane_receipt)


@pytest.mark.parametrize("version,wrong_version", [(2, 3), (3, 2)])
def test_outer_cross_version_paths_fail_before_fake_tmux(tmp_path, version, wrong_version):
    args, _child_count, tmux_count = _repo(tmp_path, version=version)
    root = args["repo_root"]
    wrong_suffix = f"r{wrong_version}"
    args["output_dir"] = root / f"artifacts/runs/rtdetrv2_r18_visdrone_baseline_smoke_{wrong_suffix}"
    args["process_evidence_dir"] = root / f"artifacts/process_evidence/rtdetrv2_r18_visdrone_baseline_smoke_{wrong_suffix}"
    args["outer_evidence_dir"] = root / f"artifacts/outer_launch_evidence/rtdetrv2_r18_visdrone_baseline_smoke_{wrong_suffix}"
    result = run_outer_launch(**args)
    assert result["status"] == "PREFLIGHT_FAILED"
    assert not tmux_count.exists()


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


@pytest.mark.parametrize("child_rc", [0, 99])
def test_finalizer_marker_binds_successful_finalizer_and_inner_exit(tmp_path, child_rc):
    args, _child_count, _tmux_count = _repo(tmp_path, child_rc=child_rc)
    run_outer_launch(**args)
    pane = args["outer_evidence_dir"] / "pane"
    assert (pane / PANE_FINALIZER_EXIT).read_bytes() == b"0\n"
    assert (pane / "pane_exit_code.txt").read_bytes() == f"{child_rc}\n".encode("ascii")
    value = validate_pane_evidence(pane)
    assert value["finalizer_returncode"] == 0
    assert value["completion"]["status"] == ("PANE_COMPLETED" if child_rc == 0 else "PANE_FAILED")
    expected = "TERMINAL_FAILED" if child_rc else "INNER_STARTED_PROCESS_EVIDENCE_ABSENT"
    assert classify_outer_evidence(args["outer_evidence_dir"], args["output_dir"], args["process_evidence_dir"]) == expected


@pytest.mark.parametrize("child_rc", [0, 99])
def test_finalizer_failure_persists_actual_raw_and_secondary_codes(tmp_path, child_rc):
    from sparse_rtdetr.baseline import smoke_outer_launcher as launcher

    args, _child_count, _tmux_count = _repo(tmp_path, child_rc=child_rc)
    finalizer = _script(tmp_path / "synthetic-finalizer", "exit 1")
    with mock.patch.object(launcher.sys, "executable", str(finalizer)):
        run_outer_launch(**args)
    pane = args["outer_evidence_dir"] / "pane"
    secondary = json.loads((pane / "pane_secondary_finalization_failure.json").read_text(encoding="utf-8"))
    assert (pane / PANE_FINALIZER_EXIT).read_bytes() == b"1\n"
    assert secondary["finalizer_returncode"] == 1
    assert (pane / "pane_exit_code.txt").read_bytes() == f"{child_rc}\n".encode("ascii")
    assert not (pane / "pane_completion.json").exists()
    assert classify_outer_evidence(args["outer_evidence_dir"], args["output_dir"], args["process_evidence_dir"]) == "PANE_FINALIZATION_FAILED"


def test_finalizer_marker_mutations_are_never_terminal(tmp_path):
    mutations = [
        ("empty", b""),
        ("missing_lf", b"0"),
        ("multiple_lf", b"0\n\n"),
        ("space", b" 0\n"),
        ("plus", b"+0\n"),
        ("leading_zero", b"00\n"),
        ("negative", b"-1\n"),
        ("too_large", b"256\n"),
        ("non_ascii", "零\n".encode("utf-8")),
        ("success_nonzero", b"1\n"),
    ]
    for name, raw in mutations:
        args, _child_count, _tmux_count = _repo(tmp_path / name)
        run_outer_launch(**args)
        pane = args["outer_evidence_dir"] / "pane"
        (pane / PANE_FINALIZER_EXIT).write_bytes(raw)
        with pytest.raises(OuterLaunchError):
            validate_pane_evidence(pane)
        assert classify_outer_evidence(args["outer_evidence_dir"], args["output_dir"], args["process_evidence_dir"]) not in {
            "TERMINAL_COMPLETE",
            "TERMINAL_FAILED",
            "PANE_FINALIZATION_FAILED",
        }


@pytest.mark.parametrize("kind", ["missing", "symlink", "directory"])
def test_finalizer_marker_missing_or_nonregular_is_not_terminal(tmp_path, kind):
    args, _child_count, _tmux_count = _repo(tmp_path / kind)
    run_outer_launch(**args)
    pane = args["outer_evidence_dir"] / "pane"
    marker = pane / PANE_FINALIZER_EXIT
    marker.unlink()
    if kind == "symlink":
        target = tmp_path / kind / "marker-target"
        target.write_bytes(b"0\n")
        marker.symlink_to(target)
    elif kind == "directory":
        marker.mkdir()
    with pytest.raises(OuterLaunchError):
        validate_pane_evidence(pane)
    assert classify_outer_evidence(args["outer_evidence_dir"], args["output_dir"], args["process_evidence_dir"]) not in {
        "TERMINAL_COMPLETE",
        "TERMINAL_FAILED",
        "PANE_FINALIZATION_FAILED",
    }


def test_secondary_return_code_must_equal_raw_finalizer_marker(tmp_path):
    from sparse_rtdetr.baseline import smoke_outer_launcher as launcher

    args, _child_count, _tmux_count = _repo(tmp_path, child_rc=99)
    finalizer = _script(tmp_path / "synthetic-finalizer", "exit 1")
    with mock.patch.object(launcher.sys, "executable", str(finalizer)):
        run_outer_launch(**args)
    pane = args["outer_evidence_dir"] / "pane"
    secondary_path = pane / "pane_secondary_finalization_failure.json"
    secondary = json.loads(secondary_path.read_text(encoding="utf-8"))
    secondary["finalizer_returncode"] = 0
    secondary_path.write_bytes(canonical_json_bytes(secondary))
    assert classify_outer_evidence(args["outer_evidence_dir"], args["output_dir"], args["process_evidence_dir"]) == "UNKNOWN"


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
        start = json.loads((pane / "pane_start.json").read_text(encoding="utf-8"))
        timing = json.loads((pane / "pane_timing.json").read_text(encoding="utf-8"))
        plan = json.loads((pane / "pane_plan.json").read_text(encoding="utf-8"))
        created = datetime.fromisoformat(error["created_at_utc"].replace("Z", "+00:00"))
        assert datetime.fromisoformat(plan["created_at_utc"].replace("Z", "+00:00")) <= created
        assert datetime.fromisoformat(start["started_at_utc"].replace("Z", "+00:00")) <= created
        assert created <= datetime.fromisoformat(timing["finished_at_utc"].replace("Z", "+00:00"))


@pytest.mark.parametrize(
    "label,value",
    [
        ("before_plan", "2000-01-01T00:00:00Z"),
        ("before_start", "2001-01-01T00:00:00Z"),
        ("after_finish", "2099-01-01T00:00:00Z"),
        ("non_utc", "2026-08-10T12:00:00+08:00"),
        ("no_timezone", "2026-08-10T12:00:00"),
        ("invalid", "not-a-time"),
    ],
)
def test_pane_error_chronology_mutations_are_rejected(tmp_path, label, value):
    args, _child_count, _tmux_count = _repo(tmp_path / label, child_rc=99)
    run_outer_launch(**args)
    pane = args["outer_evidence_dir"] / "pane"
    error_path = pane / "pane_error.json"
    error = json.loads(error_path.read_text(encoding="utf-8"))
    error["created_at_utc"] = value
    error_path.write_bytes(canonical_json_bytes(error))
    completion_path = pane / "pane_completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["error"] = _ref(pane, "pane_error.json")
    completion_path.write_bytes(canonical_json_bytes(completion))
    _repack_pane(pane)
    with pytest.raises(OuterLaunchError):
        validate_pane_evidence(pane)


def test_pane_error_chronology_boundaries_are_accepted(tmp_path):
    args, _child_count, _tmux_count = _repo(tmp_path, child_rc=99)
    run_outer_launch(**args)
    pane = args["outer_evidence_dir"] / "pane"
    start = json.loads((pane / "pane_start.json").read_text(encoding="utf-8"))
    timing = json.loads((pane / "pane_timing.json").read_text(encoding="utf-8"))
    start_at = datetime.fromisoformat(start["started_at_utc"].replace("Z", "+00:00"))
    finish_at = datetime.fromisoformat(timing["finished_at_utc"].replace("Z", "+00:00"))
    candidates = [start_at, start_at + (finish_at - start_at) / 2, finish_at]
    for candidate in candidates:
        error_path = pane / "pane_error.json"
        error = json.loads(error_path.read_text(encoding="utf-8"))
        error["created_at_utc"] = candidate.isoformat()
        error_path.write_bytes(canonical_json_bytes(error))
        completion_path = pane / "pane_completion.json"
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        completion["error"] = _ref(pane, "pane_error.json")
        completion_path.write_bytes(canonical_json_bytes(completion))
        _repack_pane(pane)
        validate_pane_evidence(pane)


def test_preinner_writer_uses_failure_time_within_valid_pane_interval(tmp_path):
    args, _child_count, _tmux_count = _repo(tmp_path)
    config = args["config_path"]
    hidden = Path(str(config) + ".hidden")
    args["tmux_executable"].write_text(
        "#!/bin/sh\n"
        f"mv {config} {hidden}\n"
        '"$7"\n'
        f"mv {hidden} {config}\n"
        "exit 0\n",
        encoding="ascii",
    )
    args["tmux_executable"].chmod(0o755)
    run_outer_launch(**args)
    pane = args["outer_evidence_dir"] / "pane"
    value = validate_pane_evidence(pane)
    assert value["completion"]["status"] == "PANE_FAILED"
    error = json.loads((pane / "pane_error.json").read_text(encoding="utf-8"))
    start = json.loads((pane / "pane_start.json").read_text(encoding="utf-8"))
    timing = json.loads((pane / "pane_timing.json").read_text(encoding="utf-8"))
    created = datetime.fromisoformat(error["created_at_utc"].replace("Z", "+00:00"))
    assert datetime.fromisoformat(start["started_at_utc"].replace("Z", "+00:00")) <= created
    assert created <= datetime.fromisoformat(timing["finished_at_utc"].replace("Z", "+00:00"))


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
    assert (pane / PANE_FINALIZER_EXIT).read_bytes() == b"1\n"
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
    assert (pane / PANE_FINALIZER_EXIT).read_bytes() == b"127\n"
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
    assert (pane / PANE_FINALIZER_EXIT).read_bytes() == b"1\n"
    assert error["failure_class"] == "PREINNER_BINDING_FAILURE"
    assert error["inner_started"] is False
    assert error["inner_returncode"] == 125
    assert secondary["original_inner_returncode"] == 125
    assert secondary["inner_started"] is False
    assert (pane / "pane_exit_code.txt").read_bytes() == b"125\n"
    assert not (pane / "pane_completion.json").exists()
    assert not child_count.exists()
    assert classify_outer_evidence(args["outer_evidence_dir"], args["output_dir"], args["process_evidence_dir"]) == "PANE_FINALIZATION_FAILED"
