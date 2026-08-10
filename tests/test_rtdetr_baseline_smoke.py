from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

torch = importlib.import_module("torch")

from sparse_rtdetr.baseline.smoke import (
    FROZEN_IMAGE_RECORDS,
    SMOKE_ID,
    SMOKE_V2_ID,
    SMOKE_V3_ID,
    SMOKE_V3_CONFIG_RELATIVE,
    SMOKE_V3_OUTPUT_RELATIVE,
    SMOKE_V3_PROCESS_RELATIVE,
    SMOKE_V3_OUTER_RELATIVE,
    SMOKE_V3_TMUX_SESSION,
    SmokeContractError,
    get_smoke_runtime_spec,
    SyntheticOneBatchLoader,
    SyntheticSmokeModel,
    SyntheticSmokePostProcessor,
    _validate_batch,
    contract_check,
    load_frozen_image_selection,
    load_smoke_config,
    run_authorized_smoke,
    run_synthetic_smoke,
    synthetic_batch,
    validate_real_smoke_environment,
)
from sparse_rtdetr.baseline.smoke_evidence import (
    argv_sha256,
    canonical_json_bytes,
    SmokeEvidence,
    SmokeEvidenceError,
    _read_object,
    validate_entry_output,
)
from sparse_rtdetr.baseline.config import canonical_config_bytes
from sparse_rtdetr.baseline.smoke_launcher import (
    SmokeLauncherError,
    _consume_handoff,
    _config_binding,
    _validate_frozen_runtime_paths,
    launch_smoke,
    main as launcher_main,
)


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable).resolve()
V1_CONFIG = ROOT / "configs/baseline/rtdetrv2_r18_visdrone_smoke_v1.json"
V2_CONFIG = ROOT / "configs/baseline/rtdetrv2_r18_visdrone_smoke_v2.json"
V3_CONFIG = ROOT / SMOKE_V3_CONFIG_RELATIVE


def _base_env() -> dict[str, str]:
    return {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "P3_RTDETR_BASELINE_SMOKE_AUTHORIZED": "1",
        "PYTHONPATH": str(ROOT / "src"),
    }


def _runtime_data_root(tmp_path: Path, name: str = "data") -> Path:
    path = tmp_path / name
    path.mkdir()
    return path


def _assert_no_absolute_posix_paths(value: object) -> None:
    if isinstance(value, dict):
        for child in value.values():
            _assert_no_absolute_posix_paths(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_absolute_posix_paths(child)
    elif isinstance(value, str):
        assert not value.startswith("/")


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


def _write_success_child(
    path: Path,
    invocation_mutation: dict[str, object] | None = None,
    receipt_mutation: dict[str, object] | None = None,
    process_invocation_mutation: dict[str, object] | None = None,
) -> None:
    mutation = {} if invocation_mutation is None else invocation_mutation
    receipt_drift = {} if receipt_mutation is None else receipt_mutation
    process_invocation_drift = {} if process_invocation_mutation is None else process_invocation_mutation
    path.write_text(
        "\n".join(
            [
                "import os",
                "from pathlib import Path",
                "import json",
                "from sparse_rtdetr.baseline.smoke_evidence import SmokeEvidence",
                "from sparse_rtdetr.baseline.smoke_launcher import _consume_handoff",
                "output = Path(os.environ['P3_SMOKE_OUTPUT_DIR'])",
                "receipt, receipt_sha = _consume_handoff(Path(os.environ['P3_SMOKE_REPO_ROOT']))",
                "receipt.update(" + repr(receipt_drift) + ")",
                "Path(os.environ['P3_SMOKE_PROCESS_EVIDENCE_DIR'], 'handoff_receipt.json').write_text(json.dumps(receipt), encoding='utf-8') if " + repr(bool(receipt_drift)) + " else None",
                "process_invocation_path = Path(os.environ['P3_SMOKE_PROCESS_EVIDENCE_DIR'], 'process_invocation.json')",
                "process_invocation = json.loads(process_invocation_path.read_text(encoding='utf-8'))",
                "process_invocation.update(" + repr(process_invocation_drift) + ")",
                "process_invocation_path.write_text(json.dumps(process_invocation), encoding='utf-8') if " + repr(bool(process_invocation_drift)) + " else None",
                "evidence = SmokeEvidence(output)",
                "config_sha = evidence.write_config({'synthetic_launcher_child': True})",
                "invocation = {'schema_version': 1, 'mode': 'synthetic', 'smoke_id': 'rtdetrv2_r18_visdrone_baseline_smoke_v1', 'config_sha256': config_sha, 'nonce': receipt['nonce'], 'launcher_pid': receipt['launcher_pid'], 'child_pid': receipt['child_pid'], 'child_ppid': receipt['child_ppid'], 'child_argv_sha256': receipt['child_argv_sha256'], 'handoff_receipt_sha256': receipt_sha, 'handoff_receipt_relative_path': 'handoff_receipt.json'}",
                "invocation.update(" + repr(mutation) + ")",
                "evidence.write_json('invocation.json', invocation)",
                "[evidence.write_json(name, {}) for name in ('source_identity.json', 'data_binding_audit.json', 'image_selection_audit.json', 'cuda_runtime_identity.json', 'model_identity.json', 'call_audit.json', 'input_batch_audit.json', 'model_output_audit.json', 'postprocess_audit.json', 'rng_audit.json')]",
                "completion = " + repr(_entry_completion()),
                "completion['config_sha256'] = config_sha",
                "evidence.finalize_success(completion)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_unconsuming_child(path: Path) -> None:
    path.write_text(
        "\n".join([
            "import os",
            "from pathlib import Path",
            "from sparse_rtdetr.baseline.smoke_evidence import SmokeEvidence",
            "output = Path(os.environ['P3_SMOKE_OUTPUT_DIR'])",
            "evidence = SmokeEvidence(output)",
            "config_sha = evidence.write_config({'unconsuming_child': True})",
            "evidence.write_json('invocation.json', {'schema_version': 1, 'mode': 'synthetic', 'smoke_id': 'rtdetrv2_r18_visdrone_baseline_smoke_v1', 'config_sha256': config_sha})",
            "[evidence.write_json(name, {}) for name in ('source_identity.json', 'data_binding_audit.json', 'image_selection_audit.json', 'cuda_runtime_identity.json', 'model_identity.json', 'call_audit.json', 'input_batch_audit.json', 'model_output_audit.json', 'postprocess_audit.json', 'rng_audit.json')]",
            "completion = " + repr(_entry_completion()),
            "completion['config_sha256'] = config_sha",
            "evidence.finalize_success(completion)",
        ]) + "\n",
        encoding="utf-8",
    )


def test_smoke_config_and_manifest_are_strict_and_frozen():
    config = load_smoke_config(ROOT)
    assert config["model"]["seed"] == 0
    assert config["execution"]["expected_batches"] == 1
    assert load_frozen_image_selection(ROOT) == FROZEN_IMAGE_RECORDS

    drifted = copy.deepcopy(config)
    drifted["execution"]["expected_batches"] = True
    with pytest.raises(SmokeContractError):
        importlib.import_module("sparse_rtdetr.baseline.smoke").validate_smoke_config(drifted)


def test_manifest_field_drift_fails_closed(monkeypatch, tmp_path):
    manifest = {
        "records": [dict(record) for record in FROZEN_IMAGE_RECORDS],
    }
    manifest["records"][0]["image_sha256"] = "0" * 64
    path = tmp_path / "train_core_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    smoke = importlib.import_module("sparse_rtdetr.baseline.smoke")
    monkeypatch.setattr(smoke, "_manifest_path", lambda _root: path)
    with pytest.raises(SmokeContractError):
        load_frozen_image_selection(ROOT)


def test_contract_check_is_data_free_and_does_not_import_torch():
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(ROOT / "src"),
    }
    result = subprocess.run(
        [str(PYTHON), "-m", "sparse_rtdetr.baseline.smoke_launcher", "contract-check", "--repo-root", str(ROOT)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    evidence = json.loads(result.stdout)
    assert evidence["status"] == "PASS"
    assert evidence["torch_imported"] is False
    assert evidence["output_directory_created"] is False
    assert evidence["dataset_or_dataloader_constructed"] is False


def test_real_smoke_prehash_uses_shared_resolver_before_cuda(monkeypatch, tmp_path):
    smoke = importlib.import_module("sparse_rtdetr.baseline.smoke")
    dataset_module = importlib.import_module("sparse_rtdetr.baseline.dataset")
    data_root = _runtime_data_root(tmp_path)
    output = tmp_path / "entry"
    calls = []

    def fail_resolver(root, role, logical_path):
        calls.append((root, role, logical_path))
        raise SmokeContractError("resolver sentinel")

    def forbidden_cuda_gate():
        raise AssertionError("invalid runtime image path must fail before CUDA validation")

    assert smoke.resolve_runtime_image_path is dataset_module.resolve_runtime_image_path
    monkeypatch.setattr(smoke, "resolve_runtime_image_path", fail_resolver)
    monkeypatch.setattr(smoke, "validate_real_smoke_environment", forbidden_cuda_gate)
    receipt = {
        "nonce": "a" * 32,
        "launcher_pid": 1,
        "child_pid": 2,
        "child_ppid": 3,
        "child_argv_sha256": "0" * 64,
    }
    with pytest.raises(SmokeContractError, match="resolver sentinel"):
        run_authorized_smoke(
            ROOT,
            data_root,
            output,
            handoff_receipt=receipt,
            handoff_receipt_sha256="0" * 64,
        )
    assert calls == [(data_root, "train_core", FROZEN_IMAGE_RECORDS[0]["relative_path"])]
    assert _read_object(output / "completion.json")["status"] == "FAILED"


def test_real_smoke_environment_rejects_unauthorized_cpu(monkeypatch):
    monkeypatch.setenv("P3_RTDETR_BASELINE_SMOKE_AUTHORIZED", "")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    with pytest.raises(SmokeContractError):
        validate_real_smoke_environment()


def test_real_smoke_environment_rejects_hidden_cuda(monkeypatch):
    monkeypatch.setenv("P3_RTDETR_BASELINE_SMOKE_AUTHORIZED", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    with pytest.raises(SmokeContractError):
        validate_real_smoke_environment()


def test_launcher_rejects_frozen_path_drift(tmp_path):
    with pytest.raises(SmokeLauncherError):
        _validate_frozen_runtime_paths(ROOT, tmp_path / "entry", tmp_path / "process")


@pytest.mark.parametrize(
    ("config_path", "output_relative", "process_relative"),
    [
        (V1_CONFIG, "artifacts/runs/rtdetrv2_r18_visdrone_baseline_smoke_r1", "artifacts/process_evidence/rtdetrv2_r18_visdrone_baseline_smoke_r1"),
        (V2_CONFIG, "artifacts/runs/rtdetrv2_r18_visdrone_baseline_smoke_r2", "artifacts/process_evidence/rtdetrv2_r18_visdrone_baseline_smoke_r2"),
        (V3_CONFIG, SMOKE_V3_OUTPUT_RELATIVE, SMOKE_V3_PROCESS_RELATIVE),
    ],
)
def test_versioned_runtime_registry_accepts_only_matching_inner_paths(config_path, output_relative, process_relative):
    config = load_smoke_config(ROOT, config_path)
    _validate_frozen_runtime_paths(
        ROOT,
        ROOT / output_relative,
        ROOT / process_relative,
        config=config,
        config_path=config_path,
    )


@pytest.mark.parametrize(
    ("config_path", "wrong_output", "wrong_process"),
    [
        (V1_CONFIG, "artifacts/runs/rtdetrv2_r18_visdrone_baseline_smoke_r2", "artifacts/process_evidence/rtdetrv2_r18_visdrone_baseline_smoke_r2"),
        (V2_CONFIG, "artifacts/runs/rtdetrv2_r18_visdrone_baseline_smoke_r1", "artifacts/process_evidence/rtdetrv2_r18_visdrone_baseline_smoke_r1"),
        (V2_CONFIG, SMOKE_V3_OUTPUT_RELATIVE, SMOKE_V3_PROCESS_RELATIVE),
        (V3_CONFIG, "artifacts/runs/rtdetrv2_r18_visdrone_baseline_smoke_r2", "artifacts/process_evidence/rtdetrv2_r18_visdrone_baseline_smoke_r2"),
        (V3_CONFIG, "artifacts/runs/rtdetrv2_r18_visdrone_baseline_smoke_r1", "artifacts/process_evidence/rtdetrv2_r18_visdrone_baseline_smoke_r1"),
    ],
)
def test_versioned_runtime_registry_rejects_cross_version_inner_paths(config_path, wrong_output, wrong_process):
    config = load_smoke_config(ROOT, config_path)
    with pytest.raises(SmokeLauncherError):
        _validate_frozen_runtime_paths(
            ROOT,
            ROOT / wrong_output,
            ROOT / wrong_process,
            config=config,
            config_path=config_path,
        )


def test_runtime_registry_is_explicit_and_unknown_ids_fail_closed():
    assert tuple(get_smoke_runtime_spec(value).smoke_id for value in (SMOKE_ID, SMOKE_V2_ID, SMOKE_V3_ID)) == (SMOKE_ID, SMOKE_V2_ID, SMOKE_V3_ID)
    with pytest.raises(SmokeContractError):
        get_smoke_runtime_spec("rtdetrv2_r18_visdrone_baseline_smoke_v99")


@pytest.mark.parametrize("config_path, forged_id", [(V2_CONFIG, SMOKE_V3_ID), (V3_CONFIG, SMOKE_V2_ID)])
def test_config_identity_forgery_is_rejected(config_path, forged_id):
    forged = json.loads(config_path.read_text(encoding="utf-8"))
    forged["smoke_id"] = forged_id
    with pytest.raises(SmokeContractError):
        importlib.import_module("sparse_rtdetr.baseline.smoke").validate_smoke_config(forged)


def test_v2_v3_scientific_fields_differ_only_by_versioned_runtime_identity():
    v2 = json.loads(V2_CONFIG.read_text(encoding="utf-8"))
    v3 = json.loads(V3_CONFIG.read_text(encoding="utf-8"))
    assert v2["smoke_id"] != v3["smoke_id"]
    allowed_runtime = {
        "config_relative_path",
        "output_relative_path",
        "process_evidence_relative_path",
        "outer_launch_evidence_relative_path",
        "tmux_session_name",
    }
    assert {key for key in v2["runtime"] if v2["runtime"].get(key) != v3["runtime"].get(key)} == allowed_runtime
    for field in ("execution", "model", "image_selection", "evidence", "mode", "split_role"):
        assert v2[field] == v3[field]
    assert v2["runtime"]["tmux_client_timeout_seconds"] == v3["runtime"]["tmux_client_timeout_seconds"] == 10


@pytest.mark.parametrize(
    ("config_path", "output_relative", "process_relative"),
    [
        (V1_CONFIG, "artifacts/runs/rtdetrv2_r18_visdrone_baseline_smoke_r1", "artifacts/process_evidence/rtdetrv2_r18_visdrone_baseline_smoke_r1"),
        (V2_CONFIG, "artifacts/runs/rtdetrv2_r18_visdrone_baseline_smoke_r2", "artifacts/process_evidence/rtdetrv2_r18_visdrone_baseline_smoke_r2"),
        (V3_CONFIG, SMOKE_V3_OUTPUT_RELATIVE, SMOKE_V3_PROCESS_RELATIVE),
    ],
)
def test_main_passes_bound_version_config_to_fake_launch_without_creating_artifacts(monkeypatch, tmp_path, config_path, output_relative, process_relative):
    launcher = importlib.import_module("sparse_rtdetr.baseline.smoke_launcher")
    captured = {}
    monkeypatch.setattr(launcher, "validate_real_smoke_environment", lambda: {"cpu_fallback": False})

    def fake_launch(child_argv, output_dir, process_dir, **kwargs):
        captured.update({"argv": child_argv, "output": output_dir, "process": process_dir, **kwargs})
        return 0

    monkeypatch.setattr(launcher, "launch_smoke", fake_launch)
    data_root = tmp_path / "synthetic-data"
    data_root.mkdir()
    output = ROOT / output_relative
    process = ROOT / process_relative
    assert not output.exists()
    assert not process.exists()
    smoke_argv = [
        "smoke",
        "--repo-root", str(ROOT),
        "--data-root", str(data_root),
        "--output-dir", str(output),
        "--process-evidence-dir", str(process),
        "--config", str(config_path),
        "--child-python", str(PYTHON),
    ]
    assert launcher.main(smoke_argv) == 0
    assert captured["output"] == output
    assert captured["process"] == process
    assert captured["config_path"] == config_path.resolve()
    assert "--config" in smoke_argv
    assert "--config" not in captured["argv"]
    parsed = launcher._parser().parse_args(captured["argv"][3:])
    assert parsed.mode == "_child"
    assert parsed.repo_root == ROOT
    assert not output.exists()
    assert not process.exists()


def test_old_child_argv_with_config_remains_argparse_failure():
    launcher = importlib.import_module("sparse_rtdetr.baseline.smoke_launcher")
    with pytest.raises(SystemExit) as raised:
        launcher._parser().parse_args([
            "_child",
            "--repo-root", str(ROOT),
            "--config", str(V3_CONFIG),
        ])
    assert raised.value.code == 2


def test_real_child_argv_roundtrip_dispatches_from_v3_receipt(monkeypatch, tmp_path):
    launcher = importlib.import_module("sparse_rtdetr.baseline.smoke_launcher")
    captured = {}
    monkeypatch.setattr(launcher, "validate_real_smoke_environment", lambda: {"stub": True})

    def fake_launch(child_argv, output_dir, process_dir, **kwargs):
        captured.update({"argv": list(child_argv), "output": output_dir, "process": process_dir, "kwargs": kwargs})
        return 0

    monkeypatch.setattr(launcher, "launch_smoke", fake_launch)
    data_root = _runtime_data_root(tmp_path)
    output = ROOT / SMOKE_V3_OUTPUT_RELATIVE
    process = ROOT / SMOKE_V3_PROCESS_RELATIVE
    smoke_argv = [
        "smoke",
        "--repo-root", str(ROOT),
        "--data-root", str(data_root),
        "--output-dir", str(output),
        "--process-evidence-dir", str(process),
        "--config", str(V3_CONFIG),
        "--child-python", str(PYTHON),
    ]
    assert launcher.main(smoke_argv) == 0
    child_argv = captured["argv"]
    assert "--config" not in child_argv
    parsed = launcher._parser().parse_args(child_argv[3:])
    assert parsed.mode == "_child"
    assert parsed.repo_root == ROOT

    receipt = {
        "schema_version": 1,
        "consumed": True,
        "nonce": "a" * 32,
        "launcher_pid": 1,
        "child_pid": 2,
        "child_ppid": 1,
        "child_argv_sha256": "0" * 64,
        "prepared_relative_path": "handoff_prepared.json",
        "prepared_sha256": "1" * 64,
        "output_dir": str(output),
        "process_evidence_dir": str(process),
        "runtime_data_root": str(data_root),
        "smoke_id": SMOKE_V3_ID,
        "config_relative_path": V3_CONFIG.relative_to(ROOT).as_posix(),
        "config_path": str(V3_CONFIG),
        "config_size_bytes": V3_CONFIG.stat().st_size,
        "config_file_sha256": hashlib.sha256(V3_CONFIG.read_bytes()).hexdigest(),
        "config_canonical_sha256": hashlib.sha256(canonical_json_bytes(json.loads(V3_CONFIG.read_text(encoding="utf-8")))).hexdigest(),
        "consumed_at_utc": "2026-08-10T00:00:00+00:00",
    }
    consume_calls = []
    entry_calls = []

    def fake_consume(repo_root):
        consume_calls.append(repo_root)
        return receipt, "2" * 64

    def fake_entry(repo_root, actual_data_root, actual_output, *, handoff_receipt, handoff_receipt_sha256, config_path=None):
        entry_calls.append({
            "repo_root": repo_root,
            "data_root": actual_data_root,
            "output": actual_output,
            "config_path": config_path,
            "receipt": handoff_receipt,
        })
        return {}

    monkeypatch.setattr(launcher, "_consume_handoff", fake_consume)
    monkeypatch.setattr(launcher, "run_authorized_smoke", fake_entry)
    monkeypatch.setenv("P3_SMOKE_OUTPUT_DIR", str(output))
    assert launcher.main(child_argv[3:]) == 0
    assert consume_calls == [ROOT]
    assert len(entry_calls) == 1
    assert entry_calls[0]["config_path"] == V3_CONFIG
    assert entry_calls[0]["data_root"] == data_root
    assert entry_calls[0]["output"] == output
    assert not output.exists()
    assert not process.exists()


def test_v3_child_passes_receipt_config_and_ignores_environment_override(monkeypatch, tmp_path):
    launcher = importlib.import_module("sparse_rtdetr.baseline.smoke_launcher")
    monkeypatch.setenv("P3_RTDETR_BASELINE_SMOKE_AUTHORIZED", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    data_root = _runtime_data_root(tmp_path)
    output = tmp_path / "v3-entry"
    process = tmp_path / "v3-process"
    capture = tmp_path / "captured-config"
    child = tmp_path / "v3-child.py"
    child.write_text(
        "import os\n"
        "from pathlib import Path\n"
        "import sparse_rtdetr.baseline.smoke_launcher as launcher\n"
        "def sentinel(repo_root, data_root, output_dir, *, handoff_receipt, handoff_receipt_sha256, config_path=None):\n"
        "    Path(os.environ['P3_SMOKE_CAPTURE']).write_text(str(config_path), encoding='utf-8')\n"
        "    return {}\n"
        "launcher.run_authorized_smoke = sentinel\n"
        "os.environ['P3_SMOKE_CONFIG_OVERRIDE'] = os.environ['P3_SMOKE_V2_CONFIG']\n"
        "raise SystemExit(launcher.main(['_child', '--repo-root', os.environ['P3_SMOKE_REPO_ROOT']]))\n",
        encoding="utf-8",
    )
    env = _base_env()
    env.update({
        "P3_SMOKE_CAPTURE": str(capture),
        "P3_SMOKE_V2_CONFIG": str(V2_CONFIG),
    })
    result = launch_smoke(
        [str(PYTHON), str(child)],
        output,
        process,
        data_root=data_root,
        child_python=PYTHON,
        repo_root=ROOT,
        env=env,
        config_path=V3_CONFIG,
        nonce="b" * 32,
    )
    assert result == 2
    assert capture.read_text(encoding="utf-8") == str(V3_CONFIG.resolve())
    assert not output.exists()


def test_synthetic_smoke_has_exact_one_batch_and_complete_evidence(tmp_path):
    output = tmp_path / "entry"
    completion = run_synthetic_smoke(ROOT, output)
    assert completion["status"] == "COMPLETED"
    assert completion["smoke_pass_candidate"] is True
    assert completion["batches_processed"] == 1
    assert completion["model_forward_calls"] == 1
    assert completion["postprocessor_calls"] == 1
    assert completion["total_predictions"] == 600
    assert _read_object(output / "rng_audit.json")["restored_equal"] is True
    assert validate_entry_output(output)["entry_success_accepted"] is True


@pytest.mark.parametrize("field", ["inference_only", "model_eval", "torch_no_grad", "speed_measurement", "total_predictions", "config_sha256"])
def test_completion_required_fields_fail_closed(tmp_path, field):
    output = tmp_path / field
    run_synthetic_smoke(ROOT, output)
    completion_path = output / "completion.json"
    completion = _read_object(completion_path)
    completion.pop(field)
    completion_path.write_bytes(canonical_json_bytes(completion))
    with pytest.raises(SmokeEvidenceError):
        validate_entry_output(output)


def test_completion_rejects_boolean_counter(tmp_path):
    output = tmp_path / "bool_counter"
    run_synthetic_smoke(ROOT, output)
    completion_path = output / "completion.json"
    completion = _read_object(completion_path)
    completion["images_requested"] = True
    completion_path.write_bytes(canonical_json_bytes(completion))
    with pytest.raises(SmokeEvidenceError):
        validate_entry_output(output)


def test_config_three_way_byte_binding_fails_closed(tmp_path):
    output = tmp_path / "config_binding"
    run_synthetic_smoke(ROOT, output)
    config_path = output / "config.json"
    config_path.write_bytes(config_path.read_bytes() + b" ")
    with pytest.raises(SmokeEvidenceError):
        validate_entry_output(output)


def test_config_can_only_be_written_once(tmp_path):
    evidence = SmokeEvidence(tmp_path / "entry")
    evidence.write_config({"once": True})
    with pytest.raises(SmokeEvidenceError):
        evidence.write_config({"twice": True})


@pytest.mark.parametrize("mutation", [
    {"nonce": "0" * 32},
    {"child_pid": 1},
    {"child_ppid": 1},
    {"child_argv_sha256": "0" * 64},
    {"handoff_receipt_sha256": "0" * 64},
    {"handoff_receipt_relative_path": "wrong.json"},
])
def test_launcher_rejects_entry_identity_drift(tmp_path, monkeypatch, mutation):
    monkeypatch.setenv("P3_RTDETR_BASELINE_SMOKE_AUTHORIZED", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    child = tmp_path / "identity_child.py"
    _write_success_child(child, mutation)
    output = tmp_path / "entry"
    process = tmp_path / "process"
    result = launch_smoke([str(PYTHON), str(child)], output, process, data_root=_runtime_data_root(tmp_path), child_python=PYTHON, repo_root=ROOT, env=_base_env(), nonce="a" * 32)
    assert result == 2
    assert _read_object(process / "process_completion.json")["status"] == "FAILED_ENTRY_CONTRACT"


@pytest.mark.parametrize("mutation", [
    {"child_pid": 1},
    {"child_ppid": 1},
    {"child_argv_sha256": "0" * 64},
    {"output_dir": "/wrong/output"},
    {"process_evidence_dir": "/wrong/process"},
    {"config_canonical_sha256": "0" * 64},
])
def test_launcher_rejects_receipt_identity_drift(tmp_path, monkeypatch, mutation):
    monkeypatch.setenv("P3_RTDETR_BASELINE_SMOKE_AUTHORIZED", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    child = tmp_path / "receipt_child.py"
    _write_success_child(child, receipt_mutation=mutation)
    output = tmp_path / "entry"
    process = tmp_path / "process"
    result = launch_smoke([str(PYTHON), str(child)], output, process, data_root=_runtime_data_root(tmp_path), child_python=PYTHON, repo_root=ROOT, env=_base_env(), nonce="9" * 32)
    assert result == 2
    assert _read_object(process / "process_completion.json")["status"] == "FAILED_ENTRY_CONTRACT"


@pytest.mark.parametrize(
    "mutation",
    [
        {"smoke_id": SMOKE_V2_ID},
        {"config_relative_path": V2_CONFIG.relative_to(ROOT).as_posix()},
        {"config_path": str(V2_CONFIG)},
    ],
)
def test_v3_receipt_cannot_bind_v2_identity(tmp_path, monkeypatch, mutation):
    monkeypatch.setenv("P3_RTDETR_BASELINE_SMOKE_AUTHORIZED", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    child = tmp_path / "v3_receipt_cross_version_child.py"
    _write_success_child(child, receipt_mutation=mutation)
    output = tmp_path / "entry"
    process = tmp_path / "process"
    result = launch_smoke(
        [str(PYTHON), str(child)],
        output,
        process,
        data_root=_runtime_data_root(tmp_path),
        child_python=PYTHON,
        repo_root=ROOT,
        env=_base_env(),
        config_path=V3_CONFIG,
        nonce="c" * 32,
    )
    assert result == 2
    assert _read_object(process / "process_completion.json")["status"] == "FAILED_ENTRY_CONTRACT"


def test_unconsuming_success_child_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("P3_RTDETR_BASELINE_SMOKE_AUTHORIZED", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    child = tmp_path / "unconsuming_child.py"
    _write_unconsuming_child(child)
    output = tmp_path / "entry"
    process = tmp_path / "process"
    result = launch_smoke([str(PYTHON), str(child)], output, process, data_root=_runtime_data_root(tmp_path), child_python=PYTHON, repo_root=ROOT, env=_base_env(), nonce="1" * 32)
    assert result == 2
    completion = _read_object(process / "process_completion.json")
    assert completion["status"] == "FAILED_ENTRY_CONTRACT"
    assert completion["entry"]["entry_status"] == "FAILED_HANDOFF_CONTRACT"


def test_prepared_handoff_is_single_use(tmp_path, monkeypatch):
    monkeypatch.setenv("P3_RTDETR_BASELINE_SMOKE_AUTHORIZED", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    child = tmp_path / "double_consume.py"
    child.write_text(
        "from pathlib import Path\n"
        "from sparse_rtdetr.baseline.smoke_launcher import _consume_handoff\n"
        "receipt, _ = _consume_handoff(Path(__import__('os').environ['P3_SMOKE_REPO_ROOT']))\n"
        "_consume_handoff(Path(__import__('os').environ['P3_SMOKE_REPO_ROOT']))\n",
        encoding="utf-8",
    )
    output = tmp_path / "entry"
    process = tmp_path / "process"
    result = launch_smoke([str(PYTHON), str(child)], output, process, data_root=_runtime_data_root(tmp_path), child_python=PYTHON, repo_root=ROOT, env=_base_env(), nonce="2" * 32)
    assert result == 2
    assert _read_object(process / "handoff_receipt.json")["consumed"] is True
    assert not output.exists()


def test_direct_child_without_handoff_fails_before_output(tmp_path):
    env = _base_env()
    env.pop("P3_SMOKE_HANDOFF_PATH", None)
    output = tmp_path / "entry"
    env["P3_SMOKE_OUTPUT_DIR"] = str(output)
    result = subprocess.run(
        [str(PYTHON), "-m", "sparse_rtdetr.baseline.smoke_launcher", "_child", "--repo-root", str(ROOT)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert not output.exists()
    assert "torch" not in result.stdout + result.stderr


@pytest.mark.parametrize("case", [
    "image_rank",
    "image_channel",
    "image_dtype",
    "image_nan",
    "labels_float",
    "labels_bool",
    "labels_nan",
    "labels_out_of_range",
    "target_boxes_shape",
    "target_boxes_nan",
    "sizes_float",
    "sizes_negative",
    "sizes_zero",
    "sizes_nan",
    "sizes_shape",
])
def test_batch_schema_rejects_each_invalid_case(case):
    batch = synthetic_batch()
    if case == "image_rank":
        batch["images"] = batch["images"][:, 0]
    elif case == "image_channel":
        batch["images"] = batch["images"][:, :1]
    elif case == "image_dtype":
        batch["images"] = batch["images"].double()
    elif case == "image_nan":
        batch["images"][0, 0, 0, 0] = float("nan")
    elif case == "labels_float":
        batch["targets"][0]["labels"] = torch.tensor([1.0])
    elif case == "labels_bool":
        batch["targets"][0]["labels"] = torch.tensor([True])
    elif case == "labels_nan":
        batch["targets"][0]["labels"] = torch.tensor([float("nan")])
    elif case == "labels_out_of_range":
        batch["targets"][0]["labels"] = torch.tensor([10], dtype=torch.int64)
    elif case == "target_boxes_shape":
        batch["targets"][0]["boxes"] = torch.zeros((2, 4), dtype=torch.float32)
    elif case == "target_boxes_nan":
        batch["targets"][0]["boxes"] = torch.full((1, 4), float("nan"), dtype=torch.float32)
    elif case == "sizes_float":
        batch["orig_target_sizes"] = batch["orig_target_sizes"].float()
    elif case == "sizes_negative":
        batch["orig_target_sizes"][0, 0] = -1
    elif case == "sizes_zero":
        batch["orig_target_sizes"][0, 0] = 0
    elif case == "sizes_nan":
        batch["orig_target_sizes"] = batch["orig_target_sizes"].float()
        batch["orig_target_sizes"][0, 0] = float("nan")
    elif case == "sizes_shape":
        batch["orig_target_sizes"] = batch["orig_target_sizes"][:, :1]
    with pytest.raises(SmokeContractError):
        _validate_batch(batch, torch)


@pytest.mark.parametrize("label_tensor", [torch.ones((300,), dtype=torch.float32), torch.ones((300,), dtype=torch.bool)])
def test_postprocessor_schema_rejects_float_and_bool_labels(tmp_path, label_tensor):
    class BadPostprocessor(SyntheticSmokePostProcessor):
        def __call__(self, outputs, orig_target_sizes):
            results = super().__call__(outputs, orig_target_sizes)
            results[0]["labels"] = label_tensor
            return results

    with pytest.raises(SmokeContractError):
        run_synthetic_smoke(ROOT, tmp_path / "bad_post", postprocessor_factory=BadPostprocessor)


def test_entry_artifacts_are_portable(tmp_path):
    output = tmp_path / "entry"
    run_synthetic_smoke(ROOT, output)
    media_prefix = "/" + "media/"
    home_prefix = "/" + "home/"
    for path in output.glob("*.json"):
        text = path.read_text(encoding="utf-8")
        assert media_prefix not in text
        assert home_prefix not in text
        assert "VisDrone2019-DET-train" not in text
        assert "VisDrone2019-DET-val" not in text
    binding = _read_object(output / "data_binding_audit.json")
    assert binding["portable"] is True
    assert binding["nonportable"] is False
    assert "data_root" not in binding


def test_config_is_written_before_any_injected_component(tmp_path):
    output = tmp_path / "entry"
    seen: dict[str, bytes] = {}

    def check_config(name: str):
        def check():
            path = output / "config.json"
            assert path.is_file()
            seen[name] = path.read_bytes()
            if name == "model":
                return SyntheticSmokeModel()
            if name == "loader":
                return SyntheticOneBatchLoader(synthetic_batch())
            return SyntheticSmokePostProcessor()

        return check

    run_synthetic_smoke(
        ROOT,
        output,
        loader_factory=check_config("loader"),
        model_factory=check_config("model"),
        postprocessor_factory=check_config("postprocessor"),
    )
    assert set(seen) == {"loader", "model", "postprocessor"}
    assert len({value for value in seen.values()}) == 1
    assert seen["model"] == (output / "config.json").read_bytes()


def test_loader_second_next_is_never_requested(tmp_path):
    class SentinelLoader:
        def __init__(self):
            self.next_calls = 0

        def __iter__(self):
            return self

        def __next__(self):
            self.next_calls += 1
            if self.next_calls > 1:
                raise AssertionError("second batch requested")
            return synthetic_batch()

    loader = SentinelLoader()
    run_synthetic_smoke(ROOT, tmp_path / "entry", loader_factory=lambda: loader)
    assert loader.next_calls == 1


def test_forward_failure_does_not_request_second_batch(tmp_path):
    loader = SyntheticOneBatchLoader(synthetic_batch())

    class FailingModel(SyntheticSmokeModel):
        def __call__(self, images):
            self.forward_calls += 1
            raise RuntimeError("forward sentinel")

    with pytest.raises(RuntimeError, match="forward sentinel"):
        run_synthetic_smoke(ROOT, tmp_path / "entry", loader_factory=lambda: loader, model_factory=FailingModel)
    assert loader.next_calls == 1


def test_postprocessor_failure_stops_after_one_forward(tmp_path):
    model = SyntheticSmokeModel()
    postprocessor = SyntheticSmokePostProcessor()

    class FailingPostprocessor(SyntheticSmokePostProcessor):
        def __call__(self, outputs, orig_target_sizes):
            self.calls += 1
            raise RuntimeError("postprocessor sentinel")

    postprocessor = FailingPostprocessor()
    with pytest.raises(RuntimeError, match="postprocessor sentinel"):
        run_synthetic_smoke(
            ROOT,
            tmp_path / "entry",
            model_factory=lambda: model,
            postprocessor_factory=lambda: postprocessor,
        )
    assert model.forward_calls == 1
    assert postprocessor.calls == 1


def test_nonfinite_input_and_output_fail_with_failure_evidence(tmp_path):
    batch = synthetic_batch()
    batch["images"][0, 0, 0, 0] = float("nan")
    output = tmp_path / "bad_input"
    with pytest.raises(SmokeContractError):
        run_synthetic_smoke(ROOT, output, loader_factory=lambda: SyntheticOneBatchLoader(batch))
    assert _read_object(output / "completion.json")["status"] == "FAILED"
    assert (output / "error.json").is_file()
    assert (output / "partial_inventory.json").is_file()

    class NaNModel(SyntheticSmokeModel):
        def __call__(self, images):
            values = super().__call__(images)
            values["pred_boxes"][0, 0, 0] = float("nan")
            return values

    with pytest.raises(SmokeContractError):
        run_synthetic_smoke(ROOT, tmp_path / "bad_output", model_factory=NaNModel)


def test_success_inventory_rejects_unbound_files(tmp_path):
    output = tmp_path / "entry"
    run_synthetic_smoke(ROOT, output)
    (output / "unbound.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SmokeEvidenceError):
        validate_entry_output(output)


def test_synthetic_failure_preserves_config_bytes(tmp_path):
    output = tmp_path / "entry"

    class FailingPostprocessor(SyntheticSmokePostProcessor):
        def __call__(self, outputs, orig_target_sizes):
            raise RuntimeError("config preservation sentinel")

    with pytest.raises(RuntimeError):
        run_synthetic_smoke(ROOT, output, postprocessor_factory=FailingPostprocessor)
    config = output / "config.json"
    expected = canonical_config_bytes(json.loads(
        (ROOT / "configs/baseline/rtdetrv2_r18_visdrone_smoke_v1.json").read_text(encoding="utf-8")
    ))
    assert config.read_bytes() == expected


def test_launcher_rejects_unauthorized_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("P3_RTDETR_BASELINE_SMOKE_AUTHORIZED", "")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    process = tmp_path / "process"
    with pytest.raises(SmokeLauncherError):
        launch_smoke(
            [str(PYTHON), "-c", "pass"],
            tmp_path / "entry",
            process,
            data_root=_runtime_data_root(tmp_path),
            child_python=PYTHON,
            repo_root=ROOT,
        )
    assert not process.exists()


def test_launcher_does_not_take_existing_output(tmp_path, monkeypatch):
    monkeypatch.setenv("P3_RTDETR_BASELINE_SMOKE_AUTHORIZED", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    output = tmp_path / "entry"
    output.mkdir()
    with pytest.raises(SmokeLauncherError):
        launch_smoke(
            [str(PYTHON), "-c", "pass"],
            output,
            tmp_path / "process",
            data_root=_runtime_data_root(tmp_path),
            child_python=PYTHON,
            repo_root=ROOT,
        )


@pytest.mark.parametrize("kind", ["relative", "missing", "file", "symlink", "test_component"])
def test_launcher_rejects_invalid_runtime_data_root_before_process_evidence(tmp_path, monkeypatch, kind):
    monkeypatch.setenv("P3_RTDETR_BASELINE_SMOKE_AUTHORIZED", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    if kind == "relative":
        data_root = Path("relative-data-root")
    elif kind == "missing":
        data_root = tmp_path / "missing"
    elif kind == "file":
        data_root = tmp_path / "file"
        data_root.write_text("not a directory", encoding="utf-8")
    elif kind == "symlink":
        target = _runtime_data_root(tmp_path, "symlink_target")
        data_root = tmp_path / "symlink"
        data_root.symlink_to(target, target_is_directory=True)
    else:
        parent = tmp_path / "Test"
        parent.mkdir()
        data_root = parent / "train_core"
        data_root.mkdir()
    output = tmp_path / "entry"
    process = tmp_path / "process"
    with pytest.raises(SmokeLauncherError):
        launch_smoke(
            [str(PYTHON), "-c", "pass"],
            output,
            process,
            data_root=data_root,
            child_python=PYTHON,
            repo_root=ROOT,
        )
    assert not output.exists()
    assert not process.exists()


def test_child_rejects_runtime_data_root_drift_before_receipt_or_output(tmp_path, monkeypatch):
    monkeypatch.setenv("P3_RTDETR_BASELINE_SMOKE_AUTHORIZED", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    root_a = _runtime_data_root(tmp_path, "root_a")
    root_b = _runtime_data_root(tmp_path, "root_b")
    child = tmp_path / "drifting_child.py"
    child.write_text(
        "import os\n"
        "from pathlib import Path\n"
        "os.environ['P3_SMOKE_DATA_ROOT'] = os.environ['P3_SMOKE_DRIFT_ROOT']\n"
        "from sparse_rtdetr.baseline.smoke_launcher import _consume_handoff\n"
        "_consume_handoff(Path(os.environ['P3_SMOKE_REPO_ROOT']))\n",
        encoding="utf-8",
    )
    output = tmp_path / "entry"
    process = tmp_path / "process"
    env = _base_env()
    env["P3_SMOKE_DRIFT_ROOT"] = str(root_b)
    result = launch_smoke(
        [str(PYTHON), str(child)],
        output,
        process,
        data_root=root_a,
        child_python=PYTHON,
        repo_root=ROOT,
        env=env,
        nonce="f" * 32,
    )
    assert result != 0
    assert not output.exists()
    assert not (process / "handoff_receipt.json").exists()
    assert "prepared handoff runtime data root mismatch" in (process / "process_console.log").read_text(encoding="utf-8")
    assert _read_object(process / "process_completion.json")["child_returncode"] != 0


def test_child_rejects_missing_runtime_data_root_before_output(tmp_path, monkeypatch):
    monkeypatch.setenv("P3_RTDETR_BASELINE_SMOKE_AUTHORIZED", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    child = tmp_path / "missing_root_child.py"
    child.write_text(
        "import os\n"
        "from pathlib import Path\n"
        "os.environ.pop('P3_SMOKE_DATA_ROOT', None)\n"
        "from sparse_rtdetr.baseline.smoke_launcher import _consume_handoff\n"
        "_consume_handoff(Path(os.environ['P3_SMOKE_REPO_ROOT']))\n",
        encoding="utf-8",
    )
    output = tmp_path / "entry"
    process = tmp_path / "process"
    result = launch_smoke(
        [str(PYTHON), str(child)],
        output,
        process,
        data_root=_runtime_data_root(tmp_path),
        child_python=PYTHON,
        repo_root=ROOT,
        env=_base_env(),
    )
    assert result != 0
    assert not output.exists()
    assert not (process / "handoff_receipt.json").exists()
    assert "handoff consumer environment is incomplete" in (process / "process_console.log").read_text(encoding="utf-8")


def test_child_rejects_modified_prepared_runtime_data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("P3_RTDETR_BASELINE_SMOKE_AUTHORIZED", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    root_a = _runtime_data_root(tmp_path, "root_a")
    root_b = _runtime_data_root(tmp_path, "root_b")
    child = tmp_path / "modified_prepared_child.py"
    child.write_text(
        "import json, os\n"
        "from pathlib import Path\n"
        "path = Path(os.environ['P3_SMOKE_HANDOFF_PATH'])\n"
        "prepared = json.loads(path.read_text(encoding='utf-8'))\n"
        "prepared['runtime_data_root'] = os.environ['P3_SMOKE_DRIFT_ROOT']\n"
        "path.write_text(json.dumps(prepared), encoding='utf-8')\n"
        "from sparse_rtdetr.baseline.smoke_launcher import _consume_handoff\n"
        "_consume_handoff(Path(os.environ['P3_SMOKE_REPO_ROOT']))\n",
        encoding="utf-8",
    )
    output = tmp_path / "entry"
    process = tmp_path / "process"
    env = _base_env()
    env["P3_SMOKE_DRIFT_ROOT"] = str(root_b)
    result = launch_smoke(
        [str(PYTHON), str(child)],
        output,
        process,
        data_root=root_a,
        child_python=PYTHON,
        repo_root=ROOT,
        env=env,
    )
    assert result != 0
    assert not output.exists()
    assert not (process / "handoff_receipt.json").exists()


def test_launcher_requires_entry_contract_even_when_child_exits_zero(tmp_path, monkeypatch):
    monkeypatch.setenv("P3_RTDETR_BASELINE_SMOKE_AUTHORIZED", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    output = tmp_path / "entry"
    process = tmp_path / "process"
    result = launch_smoke(
        [str(PYTHON), "-c", "pass"],
        output,
        process,
        data_root=_runtime_data_root(tmp_path),
        child_python=PYTHON,
        repo_root=ROOT,
        env=_base_env(),
        nonce="b" * 32,
    )
    assert result == 2
    completion = _read_object(process / "process_completion.json")
    assert completion["status"] == "FAILED_ENTRY_CONTRACT"
    assert completion["child_returncode"] == 0
    assert completion["entry"]["entry_status"] == "FAILED_HANDOFF_CONTRACT"
    assert (process / "process_partial_inventory.json").is_file()
    assert not output.exists()


def test_launcher_success_owns_process_evidence_and_handoffs_nonce(tmp_path, monkeypatch):
    monkeypatch.setenv("P3_RTDETR_BASELINE_SMOKE_AUTHORIZED", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    child = tmp_path / "success_child.py"
    _write_success_child(child)
    output = tmp_path / "entry"
    process = tmp_path / "process"
    runtime_data_root = _runtime_data_root(tmp_path)
    result = launch_smoke(
        [str(PYTHON), str(child)],
        output,
        process,
        data_root=runtime_data_root,
        child_python=PYTHON,
        repo_root=ROOT,
        env=_base_env(),
    )
    assert result == 0
    completion = _read_object(process / "process_completion.json")
    assert completion["status"] == "COMPLETED"
    assert completion["child_returncode"] == 0
    nonce = completion["nonce"]
    assert type(nonce) is str and len(nonce) == 32 and all(character in "0123456789abcdef" for character in nonce)
    assert type(completion["child_pid"]) is int
    assert completion["child_sid"] == completion["child_pgid"]
    assert completion["handoff_receipt"]["present"] is True
    assert completion["runtime_data_root"] == str(runtime_data_root)
    assert completion["data_role"] == "train_core"
    receipt = _read_object(process / "handoff_receipt.json")
    assert receipt["child_pid"] == completion["child_pid"]
    assert receipt["runtime_data_root"] == str(runtime_data_root)
    prepared = _read_object(process / "handoff_prepared.json")
    assert prepared["runtime_data_root"] == str(runtime_data_root)
    process_invocation = _read_object(process / "process_invocation.json")
    assert process_invocation["runtime_data_root"] == str(runtime_data_root)
    assert process_invocation["data_role"] == "train_core"
    assert process_invocation["nonportable"] is True
    assert process_invocation["confirmatory_metrics_accessed"] is False
    assert process_invocation["dataset_test_accessed_by_this_process"] is False
    entry_invocation = _read_object(output / "invocation.json")
    assert entry_invocation["nonce"] == completion["nonce"]
    assert entry_invocation["child_pid"] == completion["child_pid"]
    assert entry_invocation["handoff_receipt_sha256"] == completion["handoff_receipt_sha256"]
    assert entry_invocation["handoff_receipt_sha256"] == hashlib.sha256((process / "handoff_receipt.json").read_bytes()).hexdigest()
    assert _read_object(output / "completion.json")["status"] == "COMPLETED"
    for artifact in output.glob("*.json"):
        _assert_no_absolute_posix_paths(_read_object(artifact))
        assert str(runtime_data_root) not in artifact.read_text(encoding="utf-8")
    assert (process / "process_exit_code.txt").read_bytes() == b"0\n"
    assert not (process / "process_partial_inventory.json").exists()
    assert not (output / "error.json").exists()


def test_launcher_rejects_receipt_runtime_data_root_drift(tmp_path, monkeypatch):
    monkeypatch.setenv("P3_RTDETR_BASELINE_SMOKE_AUTHORIZED", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    root_a = _runtime_data_root(tmp_path, "root_a")
    root_b = _runtime_data_root(tmp_path, "root_b")
    child = tmp_path / "receipt_drift_child.py"
    _write_success_child(child, receipt_mutation={"runtime_data_root": str(root_b)})
    output = tmp_path / "entry"
    process = tmp_path / "process"
    result = launch_smoke(
        [str(PYTHON), str(child)],
        output,
        process,
        data_root=root_a,
        child_python=PYTHON,
        repo_root=ROOT,
        env=_base_env(),
    )
    assert result == 2
    completion = _read_object(process / "process_completion.json")
    assert completion["status"] == "FAILED_ENTRY_CONTRACT"
    assert completion["entry"]["entry_status"] == "FAILED_HANDOFF_CONTRACT"
    assert "runtime data root mismatch" in completion["handoff_validation_error"]


def test_launcher_rejects_process_invocation_runtime_data_root_drift(tmp_path, monkeypatch):
    monkeypatch.setenv("P3_RTDETR_BASELINE_SMOKE_AUTHORIZED", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    root_a = _runtime_data_root(tmp_path, "root_a")
    root_b = _runtime_data_root(tmp_path, "root_b")
    child = tmp_path / "process_invocation_drift_child.py"
    _write_success_child(child, process_invocation_mutation={"runtime_data_root": str(root_b)})
    output = tmp_path / "entry"
    process = tmp_path / "process"
    result = launch_smoke(
        [str(PYTHON), str(child)],
        output,
        process,
        data_root=root_a,
        child_python=PYTHON,
        repo_root=ROOT,
        env=_base_env(),
    )
    assert result == 2
    completion = _read_object(process / "process_completion.json")
    assert completion["status"] == "FAILED_ENTRY_CONTRACT"
    assert completion["entry"]["entry_status"] == "FAILED_HANDOFF_CONTRACT"
    assert "process invocation evidence drift" in completion["handoff_validation_error"]


def test_child_uses_receipt_runtime_data_root_after_environment_drift(tmp_path, monkeypatch):
    monkeypatch.setenv("P3_RTDETR_BASELINE_SMOKE_AUTHORIZED", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    root_a = _runtime_data_root(tmp_path, "root_a")
    root_b = _runtime_data_root(tmp_path, "root_b")
    capture = tmp_path / "captured_root"
    child = tmp_path / "receipt_runtime_child.py"
    child.write_text(
        "import os\n"
        "from pathlib import Path\n"
        "import sparse_rtdetr.baseline.smoke_launcher as launcher\n"
        "original_consume = launcher._consume_handoff\n"
        "def consume(repo_root):\n"
        "    result = original_consume(repo_root)\n"
        "    os.environ['P3_SMOKE_DATA_ROOT'] = os.environ['P3_SMOKE_DRIFT_ROOT']\n"
        "    return result\n"
        "def sentinel(repo_root, data_root, output_dir, *, handoff_receipt, handoff_receipt_sha256):\n"
        "    Path(os.environ['P3_SMOKE_CAPTURE']).write_text(str(data_root), encoding='utf-8')\n"
        "    return {}\n"
        "launcher._consume_handoff = consume\n"
        "launcher.run_authorized_smoke = sentinel\n"
        "raise SystemExit(launcher.main(['_child', '--repo-root', os.environ['P3_SMOKE_REPO_ROOT']]))\n",
        encoding="utf-8",
    )
    output = tmp_path / "entry"
    process = tmp_path / "process"
    env = _base_env()
    env.update({"P3_SMOKE_DRIFT_ROOT": str(root_b), "P3_SMOKE_CAPTURE": str(capture)})
    result = launch_smoke(
        [str(PYTHON), str(child)],
        output,
        process,
        data_root=root_a,
        child_python=PYTHON,
        repo_root=ROOT,
        env=env,
    )
    assert result == 2
    assert capture.read_text(encoding="utf-8") == str(root_a)
    assert not output.exists()


def test_launcher_forwards_signal_and_escalates_to_sigkill(tmp_path):
    child = tmp_path / "ignore_term.py"
    child.write_text(
        "import os, signal, time\n"
        "from pathlib import Path\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "Path(os.environ['P3_SMOKE_MARKER']).write_text('ready', encoding='utf-8')\n"
        "while True: time.sleep(1)\n",
        encoding="utf-8",
    )
    output = tmp_path / "entry"
    process = tmp_path / "process"
    runtime_data_root = _runtime_data_root(tmp_path)
    env = _base_env()
    env.update({
        "P3_SMOKE_CHILD": str(child),
        "P3_SMOKE_OUTPUT": str(output),
        "P3_SMOKE_PROCESS": str(process),
        "P3_SMOKE_DATA_ROOT_FOR_TEST": str(runtime_data_root),
        "P3_SMOKE_MARKER": str(tmp_path / "child_ready"),
    })
    wrapper = (
        "import os, sys\n"
        "from pathlib import Path\n"
        "from sparse_rtdetr.baseline.smoke_launcher import launch_smoke\n"
        "raise SystemExit(launch_smoke([sys.executable, os.environ['P3_SMOKE_CHILD']], "
        "Path(os.environ['P3_SMOKE_OUTPUT']), Path(os.environ['P3_SMOKE_PROCESS']), "
        "data_root=Path(os.environ['P3_SMOKE_DATA_ROOT_FOR_TEST']), "
        "child_python=Path(sys.executable), repo_root=Path(" + repr(str(ROOT)) + "), "
        "env=dict(os.environ), nonce='d'*32))\n"
    )
    runner = subprocess.Popen([str(PYTHON), "-c", wrapper], cwd=ROOT, env=env)
    try:
        deadline = time.monotonic() + 5
        while not (process / "process_invocation.json").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert (process / "process_invocation.json").exists()
        while not Path(env["P3_SMOKE_MARKER"]).exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert Path(env["P3_SMOKE_MARKER"]).exists()
        runner.send_signal(signal.SIGTERM)
        assert runner.wait(timeout=8) == 2
    finally:
        if runner.poll() is None:
            runner.kill()
            runner.wait()
    completion = _read_object(process / "process_completion.json")
    assert completion["child_kill_escalated_to_sigkill"] is True
    assert completion["signal_events"][0]["signal_name"] == "SIGTERM"
    assert completion["signal_events"][0]["forwarded_to_child_process_group"] is True
    assert completion["status"] == "FAILED_CHILD"
    assert not output.exists()


def test_process_inventory_binds_all_launcher_files(tmp_path, monkeypatch):
    monkeypatch.setenv("P3_RTDETR_BASELINE_SMOKE_AUTHORIZED", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    process = tmp_path / "process"
    launch_smoke(
        [str(PYTHON), "-c", "pass"],
        tmp_path / "entry",
        process,
        data_root=_runtime_data_root(tmp_path),
        child_python=PYTHON,
        repo_root=ROOT,
        env=_base_env(),
        nonce="e" * 32,
    )
    completion = _read_object(process / "process_completion.json")
    inventory = _read_object(process / "process_inventory.json")
    assert completion["process_inventory"]["sha256"] == __import__("hashlib").sha256(
        (process / "process_inventory.json").read_bytes()
    ).hexdigest()
    assert "process_partial_inventory.json" not in {
        row["relative_path"] for row in inventory["artifacts"]
    }
