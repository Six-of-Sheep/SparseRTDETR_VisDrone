from __future__ import annotations

import copy
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
    SmokeContractError,
    SyntheticOneBatchLoader,
    SyntheticSmokeModel,
    SyntheticSmokePostProcessor,
    contract_check,
    load_frozen_image_selection,
    load_smoke_config,
    run_synthetic_smoke,
    synthetic_batch,
    validate_real_smoke_environment,
)
from sparse_rtdetr.baseline.smoke_evidence import (
    SmokeEvidence,
    SmokeEvidenceError,
    _read_object,
    validate_entry_output,
)
from sparse_rtdetr.baseline.config import canonical_config_bytes
from sparse_rtdetr.baseline.smoke_launcher import (
    SmokeLauncherError,
    _validate_frozen_runtime_paths,
    launch_smoke,
)


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable).resolve()


def _base_env() -> dict[str, str]:
    return {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "P3_RTDETR_BASELINE_SMOKE_AUTHORIZED": "1",
        "PYTHONPATH": str(ROOT / "src"),
    }


def _entry_completion() -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "COMPLETED",
        "mode": "smoke",
        "smoke_pass_candidate": True,
        "non_training": True,
        "non_selection": True,
        "non_reportable": True,
        "formal_resume_eligible": False,
        "formal_training_eligible": False,
        "baseline_training_ready": False,
        "confirmatory_metrics_accessed": False,
        "dataset_test_accessed_by_this_process": False,
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


def _write_success_child(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "import os",
                "from pathlib import Path",
                "from sparse_rtdetr.baseline.smoke_evidence import SmokeEvidence",
                "output = Path(os.environ['P3_SMOKE_OUTPUT_DIR'])",
                "evidence = SmokeEvidence(output)",
                "evidence.write_config({'synthetic_launcher_child': True})",
                f"evidence.finalize_success({_entry_completion()!r})",
            ]
        )
        + "\n",
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
            child_python=PYTHON,
            repo_root=ROOT,
        )


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
        child_python=PYTHON,
        repo_root=ROOT,
        env=_base_env(),
        nonce="b" * 32,
    )
    assert result == 2
    completion = _read_object(process / "process_completion.json")
    assert completion["status"] == "FAILED_ENTRY_CONTRACT"
    assert completion["child_returncode"] == 0
    assert completion["entry"]["entry_status"] == "FAILED_ENTRY_MISSING"
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
    nonce = "c" * 32
    result = launch_smoke(
        [str(PYTHON), str(child)],
        output,
        process,
        child_python=PYTHON,
        repo_root=ROOT,
        env=_base_env(),
        nonce=nonce,
    )
    assert result == 0
    completion = _read_object(process / "process_completion.json")
    assert completion["status"] == "COMPLETED"
    assert completion["child_returncode"] == 0
    assert completion["nonce"] == nonce
    assert completion["child_sid"] == completion["child_pgid"]
    assert _read_object(output / "completion.json")["status"] == "COMPLETED"
    assert (process / "process_exit_code.txt").read_bytes() == b"0\n"
    assert not (process / "process_partial_inventory.json").exists()
    assert not (output / "error.json").exists()


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
    env = _base_env()
    env.update({
        "P3_SMOKE_CHILD": str(child),
        "P3_SMOKE_OUTPUT": str(output),
        "P3_SMOKE_PROCESS": str(process),
        "P3_SMOKE_MARKER": str(tmp_path / "child_ready"),
    })
    wrapper = (
        "import os, sys\n"
        "from pathlib import Path\n"
        "from sparse_rtdetr.baseline.smoke_launcher import launch_smoke\n"
        "raise SystemExit(launch_smoke([sys.executable, os.environ['P3_SMOKE_CHILD']], "
        "Path(os.environ['P3_SMOKE_OUTPUT']), Path(os.environ['P3_SMOKE_PROCESS']), "
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
