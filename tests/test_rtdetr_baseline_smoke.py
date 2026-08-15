from __future__ import annotations

import copy
import builtins
import argparse
import hashlib
import importlib
import inspect
import json
import os
import random
import signal
import stat
import subprocess
import sys
import tempfile
import time
import ast
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = importlib.import_module("torch")

from sparse_rtdetr.baseline.smoke import (
    FROZEN_IMAGE_RECORDS,
    SMOKE_ID,
    SMOKE_V2_ID,
    SMOKE_V2_OUTPUT_RELATIVE,
    SMOKE_V2_PROCESS_RELATIVE,
    SMOKE_V3_ID,
    SMOKE_V4_ID,
    SMOKE_V5_ID,
    SMOKE_V6_ID,
    SMOKE_V7_ID,
    SMOKE_V3_CONFIG_RELATIVE,
    SMOKE_V3_OUTPUT_RELATIVE,
    SMOKE_V3_PROCESS_RELATIVE,
    SMOKE_V3_OUTER_RELATIVE,
    SMOKE_V3_TMUX_SESSION,
    SMOKE_V4_CONFIG_RELATIVE,
    SMOKE_V5_CONFIG_RELATIVE,
    SMOKE_V4_OUTPUT_RELATIVE,
    SMOKE_V4_PROCESS_RELATIVE,
    SMOKE_V4_OUTER_RELATIVE,
    SMOKE_V4_TMUX_SESSION,
    SMOKE_V5_OUTPUT_RELATIVE,
    SMOKE_V5_PROCESS_RELATIVE,
    SMOKE_V5_OUTER_RELATIVE,
    SMOKE_V5_TMUX_SESSION,
    SMOKE_V6_CONFIG_RELATIVE,
    SMOKE_V6_OUTPUT_RELATIVE,
    SMOKE_V6_PROCESS_RELATIVE,
    SMOKE_V6_OUTER_RELATIVE,
    SMOKE_V6_TMUX_SESSION,
    SMOKE_V7_CONFIG_RELATIVE,
    SMOKE_V7_OUTPUT_RELATIVE,
    SMOKE_V7_PROCESS_RELATIVE,
    SMOKE_V7_OUTER_RELATIVE,
    SMOKE_V7_TMUX_SESSION,
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
    ENTRY_INVENTORY_EXCLUDED,
    argv_sha256,
    canonical_json_bytes,
    inventory,
    SmokeEvidence,
    SmokeEvidenceError,
    _canonical_real_state_inventory,
    _read_object,
    _state_hashes_from_inventory,
    _validate_and_hash_state,
    _validate_model_identity,
    build_model_identity,
    validate_entry_output,
)
from sparse_rtdetr.baseline.config import build_r18_cpu_model, canonical_config_bytes
from sparse_rtdetr.baseline.postprocessor import VisDronePostProcessor
from sparse_rtdetr.baseline.smoke_launcher import (
    SmokeLauncherError,
    _SingleOccurrenceAction,
    _create_exclusive_durable_handoff_receipt,
    _consume_handoff,
    _config_binding,
    _parser,
    _validate_frozen_runtime_paths,
    launch_smoke,
    main as launcher_main,
)

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable).resolve()
V1_CONFIG = ROOT / "configs/baseline/rtdetrv2_r18_visdrone_smoke_v1.json"
V2_CONFIG = ROOT / "configs/baseline/rtdetrv2_r18_visdrone_smoke_v2.json"
V3_CONFIG = ROOT / SMOKE_V3_CONFIG_RELATIVE
V4_CONFIG = ROOT / SMOKE_V4_CONFIG_RELATIVE
V5_CONFIG = ROOT / SMOKE_V5_CONFIG_RELATIVE
V6_CONFIG = ROOT / SMOKE_V6_CONFIG_RELATIVE
V7_CONFIG = ROOT / SMOKE_V7_CONFIG_RELATIVE
CHECKER_PATH = ROOT / "tools" / "repository_contract_check.py"
CHECKER_SPEC = importlib.util.spec_from_file_location("repository_contract_check_for_smoke", CHECKER_PATH)
CHECKER = importlib.util.module_from_spec(CHECKER_SPEC)
assert CHECKER_SPEC.loader is not None
CHECKER_SPEC.loader.exec_module(CHECKER)


def _directory_fingerprint(path: Path) -> tuple[tuple[str, int, str], ...] | None:
    if not path.exists() and not path.is_symlink():
        return None
    rows = []
    for item in sorted(path.rglob("*")):
        relative = item.relative_to(path).as_posix()
        if item.is_symlink():
            rows.append((relative, -1, os.readlink(item)))
        elif item.is_file():
            rows.append((relative, item.stat().st_size, hashlib.sha256(item.read_bytes()).hexdigest()))
        elif item.is_dir():
            rows.append((relative + "/", 0, "directory"))
    return tuple(rows)


R5_FROZEN_FINGERPRINTS = {
    relative: _directory_fingerprint(ROOT / relative)
    for relative in (SMOKE_V5_OUTPUT_RELATIVE, SMOKE_V5_PROCESS_RELATIVE, SMOKE_V5_OUTER_RELATIVE)
}


def _assert_r5_frozen():
    assert {
        relative: _directory_fingerprint(ROOT / relative)
        for relative in R5_FROZEN_FINGERPRINTS
    } == R5_FROZEN_FINGERPRINTS


class _PreCudaSentinel(RuntimeError):
    pass


def _real_entry_handoff(config_path: Path, output: Path, data_root: Path) -> tuple[dict[str, object], str]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    child_argv = [
        str(PYTHON),
        "-m",
        "sparse_rtdetr.baseline.smoke_launcher",
        "_child",
        "--repo-root",
        str(ROOT),
    ]
    python_stat = PYTHON.stat()
    receipt = {
        "schema_version": 1,
        "state": "CONSUMED",
        "single_use": True,
        "consumed": True,
        "nonce": "a" * 32,
        "launcher_pid": 101,
        "child_pid": 202,
        "child_ppid": 303,
        "repo_root": str(ROOT),
        "output_dir": str(output),
        "process_evidence_dir": str(output.parent / "process"),
        "runtime_data_root": str(data_root),
        "smoke_id": config["smoke_id"],
        "config_relative_path": config_path.relative_to(ROOT).as_posix(),
        "config_path": str(config_path.resolve()),
        "config_size_bytes": config_path.stat().st_size,
        "config_file_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "config_canonical_sha256": hashlib.sha256(canonical_json_bytes(config)).hexdigest(),
        "child_argv": child_argv,
        "child_argv_schema": "module-v1",
        "child_argv_sha256": argv_sha256(child_argv),
        "child_executable_identity": {
            "canonical_path": str(PYTHON),
            "size_bytes": python_stat.st_size,
            "sha256": hashlib.sha256(PYTHON.read_bytes()).hexdigest(),
            "mode": stat.S_IMODE(python_stat.st_mode),
            "regular_file": True,
            "executable": True,
        },
        "prepared_relative_path": "handoff_prepared.json",
        "prepared_sha256": "2" * 64,
        "created_at_utc": "2026-08-11T00:00:00+00:00",
        "consumed_at_utc": "2026-08-11T00:00:01+00:00",
    }
    return receipt, hashlib.sha256(canonical_json_bytes(receipt)).hexdigest()


def _patch_synthetic_selection(monkeypatch):
    smoke = importlib.import_module("sparse_rtdetr.baseline.smoke")
    def synthetic_selection(_repo_root, *, runtime_mode="real", config_path=None):
        assert runtime_mode in {"real", "synthetic"}
        if runtime_mode == "real":
            return FROZEN_IMAGE_RECORDS
        assert config_path is not None
        return FROZEN_IMAGE_RECORDS

    monkeypatch.setattr(smoke, "load_frozen_image_selection", synthetic_selection)
    return smoke


def _base_env() -> dict[str, str]:
    return {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "P3_RTDETR_BASELINE_SMOKE_AUTHORIZED": "1",
        "PYTHONPATH": str(ROOT / "src"),
    }


def _write_module_child_sitecustomize(path: Path) -> None:
    path.write_text(
        "import json, os\n"
        "from pathlib import Path\n"
        "if os.environ.get('P3_SMOKE_TEST_EXCLUSIVE_STUB') == '1':\n"
        "    import time\n"
        "    _barrier_root = Path(os.environ['P3_SMOKE_EXCLUSIVE_BARRIER_DIR'])\n"
        "    _barrier_count = int(os.environ['P3_SMOKE_EXCLUSIVE_BARRIER_COUNT'])\n"
        "    _original_open = os.open\n"
        "    def _barrier_open(path, flags, mode=0o777, *args, **kwargs):\n"
        "        if Path(path).name == 'handoff_receipt.json' and flags & os.O_EXCL:\n"
        "            marker = _barrier_root / ('ready.' + str(os.getpid()))\n"
        "            try:\n"
        "                fd = _original_open(str(marker), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)\n"
        "                os.close(fd)\n"
        "            except FileExistsError:\n"
        "                pass\n"
        "            deadline = time.monotonic() + 10\n"
        "            while len(list(_barrier_root.glob('ready.*'))) < _barrier_count:\n"
        "                if time.monotonic() >= deadline:\n"
        "                    raise RuntimeError('exclusive receipt barrier timeout')\n"
        "                time.sleep(0.001)\n"
        "        return _original_open(path, flags, mode, *args, **kwargs)\n"
        "    os.open = _barrier_open\n"
        "mutation = os.environ.get('P3_SMOKE_TEST_MUTATION', '')\n"
        "handoff_path = Path(os.environ.get('P3_SMOKE_HANDOFF_PATH', '/missing'))\n"
        "if mutation:\n"
        "    value = json.loads(handoff_path.read_text(encoding='utf-8'))\n"
        "    if mutation == 'extra': value['child_argv'].append('--extra')\n"
        "    elif mutation == 'missing': value['child_argv'].pop()\n"
        "    elif mutation == 'reordered': value['child_argv'][2], value['child_argv'][3] = value['child_argv'][3], value['child_argv'][2]\n"
        "    elif mutation == 'duplicate': value['child_argv'].insert(4, '--repo-root')\n"
        "    elif mutation == 'config': value['child_argv'].insert(3, '--config')\n"
        "    elif mutation == 'module': value['child_argv'][2] = 'wrong.module'\n"
        "    elif mutation == 'subcommand': value['child_argv'][3] = 'smoke'\n"
        "    elif mutation == 'repo': value['child_argv'][-1] = '.'\n"
        "    elif mutation == 'sha': value['child_argv_sha256'] = '0' * 64\n"
        "    elif mutation == 'identity': value['child_executable_identity']['sha256'] = '0' * 64\n"
        "    handoff_path.write_text(json.dumps(value), encoding='utf-8')\n"
        "if mutation == 'orig_argv': __import__('sys').orig_argv = list(__import__('sys').orig_argv) + ['forged']\n"
        "if os.environ.get('P3_SMOKE_TEST_STUB') == '1':\n"
        "    import sparse_rtdetr.baseline.smoke as smoke\n"
        "    def stub(repo_root, data_root, output_dir, **kwargs):\n"
        "        Path(os.environ['P3_SMOKE_TEST_MARKER']).write_text('called\\n', encoding='utf-8')\n"
        "        completion = smoke.run_synthetic_smoke(repo_root, output_dir, config_path=Path(repo_root, 'configs/baseline/rtdetrv2_r18_visdrone_smoke_v1.json'))\n"
        "        receipt = kwargs['handoff_receipt']\n"
        "        invocation_path = Path(output_dir, 'invocation.json')\n"
        "        invocation = json.loads(invocation_path.read_text(encoding='utf-8'))\n"
        "        invocation.update({'nonce': receipt['nonce'], 'launcher_pid': receipt['launcher_pid'], 'child_pid': receipt['child_pid'], 'child_ppid': receipt['child_ppid'], 'child_argv_sha256': receipt['child_argv_sha256'], 'handoff_receipt_sha256': kwargs['handoff_receipt_sha256'], 'handoff_receipt_relative_path': 'handoff_receipt.json'})\n"
        "        from sparse_rtdetr.baseline.smoke_evidence import canonical_json_bytes\n"
        "        invocation_path.write_bytes(canonical_json_bytes(invocation))\n"
        "        from sparse_rtdetr.baseline.smoke_evidence import ENTRY_INVENTORY_EXCLUDED, inventory, sha256_file\n"
        "        inventory_path = Path(output_dir, 'artifact_inventory.json')\n"
        "        inventory_path.write_bytes(canonical_json_bytes(inventory(Path(output_dir), ENTRY_INVENTORY_EXCLUDED)))\n"
        "        completion_path = Path(output_dir, 'completion.json')\n"
        "        completion_value = json.loads(completion_path.read_text(encoding='utf-8'))\n"
        "        completion_value['artifact_inventory_sha256'] = sha256_file(inventory_path)\n"
        "        completion_path.write_bytes(canonical_json_bytes(completion_value))\n"
        "        return completion\n"
        "    smoke.run_authorized_smoke = stub\n"
        "if os.environ.get('P3_SMOKE_TEST_EXCLUSIVE_STUB') == '1':\n"
        "    import sparse_rtdetr.baseline.smoke as smoke\n"
        "    def exclusive_stub(repo_root, data_root, output_dir, **kwargs):\n"
        "        marker = Path(os.environ['P3_SMOKE_EXCLUSIVE_ENTRY_DIR']) / ('entry.' + str(os.getpid()))\n"
        "        fd = os.open(str(marker), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)\n"
        "        os.write(fd, b'ENTRY\\n')\n"
        "        os.close(fd)\n"
        "    smoke.run_authorized_smoke = exclusive_stub\n",
        encoding="utf-8",
    )


def _module_child_env(tmp_path: Path, sitecustomize: Path) -> dict[str, str]:
    env = _base_env()
    env["PYTHONPATH"] = f"{sitecustomize.parent}:{ROOT / 'src'}"
    return env


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
                "from sparse_rtdetr.baseline.smoke import run_synthetic_smoke",
                "from sparse_rtdetr.baseline.smoke_launcher import _consume_handoff",
                "output = Path(os.environ['P3_SMOKE_OUTPUT_DIR'])",
                "receipt, receipt_sha = _consume_handoff(Path(os.environ['P3_SMOKE_REPO_ROOT']))",
                "receipt.update(" + repr(receipt_drift) + ")",
                "Path(os.environ['P3_SMOKE_PROCESS_EVIDENCE_DIR'], 'handoff_receipt.json').write_text(json.dumps(receipt), encoding='utf-8') if " + repr(bool(receipt_drift)) + " else None",
                "process_invocation_path = Path(os.environ['P3_SMOKE_PROCESS_EVIDENCE_DIR'], 'process_invocation.json')",
                "process_invocation = json.loads(process_invocation_path.read_text(encoding='utf-8'))",
                "process_invocation.update(" + repr(process_invocation_drift) + ")",
                "process_invocation_path.write_text(json.dumps(process_invocation), encoding='utf-8') if " + repr(bool(process_invocation_drift)) + " else None",
                "run_synthetic_smoke(Path(os.environ['P3_SMOKE_REPO_ROOT']), output, config_path=receipt['config_path'])",
                "invocation_path = output / 'invocation.json'",
                "invocation = json.loads(invocation_path.read_text(encoding='utf-8'))",
                "invocation.update({'nonce': receipt['nonce'], 'launcher_pid': receipt['launcher_pid'], 'child_pid': receipt['child_pid'], 'child_ppid': receipt['child_ppid'], 'child_argv_sha256': receipt['child_argv_sha256'], 'handoff_receipt_sha256': receipt_sha, 'handoff_receipt_relative_path': 'handoff_receipt.json'})",
                "invocation.update(" + repr(mutation) + ")",
                "from sparse_rtdetr.baseline.smoke_evidence import ENTRY_INVENTORY_EXCLUDED, inventory, sha256_file, canonical_json_bytes",
                "invocation_path.write_bytes(canonical_json_bytes(invocation))",
                "inventory_path = output / 'artifact_inventory.json'",
                "inventory_path.write_bytes(canonical_json_bytes(inventory(output, ENTRY_INVENTORY_EXCLUDED)))",
                "completion_path = output / 'completion.json'",
                "completion = json.loads(completion_path.read_text(encoding='utf-8'))",
                "completion['artifact_inventory_sha256'] = sha256_file(inventory_path)",
                "completion_path.write_bytes(canonical_json_bytes(completion))",
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
            "from sparse_rtdetr.baseline.smoke import run_synthetic_smoke",
            "output = Path(os.environ['P3_SMOKE_OUTPUT_DIR'])",
            "from pathlib import Path",
            "run_synthetic_smoke(Path(os.environ['P3_SMOKE_REPO_ROOT']), output)",
        ]) + "\n",
        encoding="utf-8",
    )


def _write_real_entry(output: Path, config_path: Path) -> dict[str, object]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_synthetic_smoke(ROOT, output, config_path=config_path)
    invocation_path = output / "invocation.json"
    invocation = _read_object(invocation_path)
    receipt = {
        "nonce": "a" * 32,
        "launcher_pid": 11,
        "child_pid": 22,
        "child_ppid": 33,
        "child_argv_sha256": "1" * 64,
        "handoff_receipt_sha256": "2" * 64,
        "handoff_receipt_relative_path": "handoff_receipt.json",
    }
    invocation.update({"mode": "synthetic", **receipt})
    invocation_path.write_bytes(canonical_json_bytes(invocation))
    _repack_entry(output)
    return {
        **receipt,
        "smoke_id": config["smoke_id"],
        "config_relative_path": config_path.relative_to(ROOT).as_posix(),
        "config_canonical_sha256": hashlib.sha256(canonical_json_bytes(config)).hexdigest(),
    }


def _repack_entry(output: Path) -> None:
    value = inventory(output, ENTRY_INVENTORY_EXCLUDED)
    payload = canonical_json_bytes(value)
    (output / "artifact_inventory.json").write_bytes(payload)
    completion = _read_object(output / "completion.json")
    completion["artifact_inventory_sha256"] = hashlib.sha256(payload).hexdigest()
    (output / "completion.json").write_bytes(canonical_json_bytes(completion))


def _repack_model_identity(output: Path, mutate) -> None:
    path = output / "model_identity.json"
    value = _read_object(path)
    mutate(value)
    try:
        schema_sha, value_sha = _state_hashes_from_inventory(value["parameter_inventory"], value["buffer_inventory"])
    except (KeyError, TypeError):
        schema_sha = value_sha = value["parameter_state_schema_sha256"]
    value["parameter_state_schema_sha256"] = schema_sha
    value["parameter_state_value_sha256"] = value_sha
    path.write_bytes(canonical_json_bytes(value))
    _repack_entry(output)


def _forged_parameter_row(*, name: str = "forged.weight", numel: int = 1) -> dict[str, object]:
    return {
        "name": name,
        "shape": [numel],
        "dtype": "torch.float32",
        "numel": numel,
        "requires_grad": True,
        "kind": "parameter",
        "logical_sha256": hashlib.sha256(b"forged").hexdigest(),
    }


def test_smoke_config_and_manifest_are_strict_and_frozen():
    config = load_smoke_config(ROOT)
    assert config["model"]["seed"] == 0
    assert config["execution"]["expected_batches"] == 1
    assert load_frozen_image_selection(ROOT) == FROZEN_IMAGE_RECORDS

    drifted = copy.deepcopy(config)
    drifted["execution"]["expected_batches"] = True
    with pytest.raises(SmokeContractError):
        importlib.import_module("sparse_rtdetr.baseline.smoke").validate_smoke_config(drifted)


@pytest.mark.parametrize("config_path", [V1_CONFIG, V2_CONFIG, V3_CONFIG, V4_CONFIG, V5_CONFIG, V6_CONFIG, V7_CONFIG])
def test_schema_version_accepts_only_builtin_int_one(config_path):
    smoke = importlib.import_module("sparse_rtdetr.baseline.smoke")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    smoke.validate_smoke_config(config)
    invalid_values = [True, False, 1.0, "1", None, [], {}, 0, -1, 2]
    try:
        import numpy as np
    except ImportError:
        np = None

    class IntSubclass(int):
        pass

    invalid_values.extend([IntSubclass(1)])
    if np is not None:
        invalid_values.append(np.int64(1))
    for value in invalid_values:
        drifted = copy.deepcopy(config)
        drifted["schema_version"] = value
        with pytest.raises(SmokeContractError):
            smoke.validate_smoke_config(drifted)
    missing = copy.deepcopy(config)
    del missing["schema_version"]
    with pytest.raises(SmokeContractError):
        smoke.validate_smoke_config(missing)


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


def _real_contract_cli(
    config_path: Path | None = None,
    *,
    repo_root: Path = ROOT,
    python: Path = PYTHON,
    canonicalize_config: bool = True,
):
    argv = [
        str(python),
        "-m",
        "sparse_rtdetr.baseline.smoke_launcher",
        "contract-check",
        "--repo-root",
        str(repo_root.resolve()),
    ]
    if config_path is not None:
        config_token = config_path.resolve() if canonicalize_config else config_path
        argv.extend(("--config", str(config_token)))
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(repo_root.resolve() / "src"),
    }
    result = subprocess.run(argv, cwd=repo_root, env=env, capture_output=True, text=True, check=False)
    return argv, result


def _assert_contract_cli_pass(result, smoke_id):
    assert result.returncode == 0, result.stdout + result.stderr
    evidence = json.loads(result.stdout)
    assert evidence["status"] == "PASS"
    assert evidence["smoke_id"] == smoke_id
    assert evidence["torch_imported"] is False
    assert evidence["real_data_accessed"] is False
    assert evidence["real_image_accessed"] is False
    assert evidence["model_constructed"] is False
    assert evidence["dataset_or_dataloader_constructed"] is False
    assert evidence["cuda_object_created"] is False
    assert evidence["network_requested"] is False
    assert evidence["output_directory_created"] is False
    assert evidence["process_evidence_created"] is False
    assert evidence["confirmatory_metrics_accessed"] is False
    assert evidence["dataset_test_accessed_by_this_process"] is False
    return evidence


@pytest.mark.parametrize(
    ("config_path", "smoke_id"),
    [
        (None, SMOKE_ID),
        (V1_CONFIG, SMOKE_ID),
        (V2_CONFIG, SMOKE_V2_ID),
        (V3_CONFIG, SMOKE_V3_ID),
        (V4_CONFIG, SMOKE_V4_ID),
        (V5_CONFIG, SMOKE_V5_ID),
        (V6_CONFIG, SMOKE_V6_ID),
        (V7_CONFIG, SMOKE_V7_ID),
    ],
)
def test_real_contract_check_cli_binds_explicit_config_across_versions(config_path, smoke_id):
    argv, result = _real_contract_cli(config_path)
    _assert_contract_cli_pass(result, smoke_id)
    if config_path is None:
        assert "--config" not in argv
    else:
        assert argv[-2:] == ["--config", str(config_path.resolve())]
        assert argv[0] == str(PYTHON)
        assert len(argv) == 8
        assert argv_sha256(argv) == hashlib.sha256(canonical_json_bytes(argv)).hexdigest()
        assert "P3_SMOKE_CONFIG" not in os.environ


@pytest.mark.parametrize(
    "argv_tail",
    [
        ["contract-check"],
        ["contract-check", "--repo-root", str(ROOT), "--config"],
        ["contract-check", "--repo-root", str(ROOT), "--unknown"],
        ["contract-check", "--repo-root", str(ROOT), "--config", str(ROOT / "missing.json")],
        ["contract-check", "--repo-root", str(ROOT), "--config", str(ROOT / "configs")],
    ],
)
def test_real_contract_check_cli_rejects_invalid_arguments(argv_tail):
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(ROOT / "src"),
    }
    result = subprocess.run(
        [str(PYTHON), "-m", "sparse_rtdetr.baseline.smoke_launcher", *argv_tail],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert '"status": "PASS"' not in result.stdout


def test_contract_check_cli_rejects_symlink_and_unregistered_config(tmp_path):
    symlink = tmp_path / "v5-link.json"
    symlink.symlink_to(V5_CONFIG)
    for config_path in (symlink, tmp_path / "v5-copy.json"):
        if config_path.name == "v5-copy.json":
            config_path.write_bytes(V5_CONFIG.read_bytes())
        _, result = _real_contract_cli(config_path, canonicalize_config=False)
        assert result.returncode == 2
        assert "PASS" not in result.stdout


def test_contract_check_cli_rejects_mutated_v5_unknown_identity_and_cross_version_path(tmp_path):
    mutated = tmp_path / "mutated.json"
    payload = json.loads(V5_CONFIG.read_text(encoding="utf-8"))
    payload["model"]["seed"] = 1
    mutated.write_text(json.dumps(payload), encoding="utf-8")
    unknown = tmp_path / "unknown.json"
    unknown_payload = json.loads(V5_CONFIG.read_text(encoding="utf-8"))
    unknown_payload["smoke_id"] = "unknown"
    unknown.write_text(json.dumps(unknown_payload), encoding="utf-8")
    cross_version = tmp_path / "cross-version.json"
    cross_payload = json.loads(V5_CONFIG.read_text(encoding="utf-8"))
    cross_payload["runtime"]["output_relative_path"] = "artifacts/runs/rtdetrv2_r18_visdrone_baseline_smoke_r4"
    cross_version.write_text(json.dumps(cross_payload), encoding="utf-8")
    for config_path in (mutated, unknown, cross_version):
        _, result = _real_contract_cli(config_path)
        assert result.returncode == 2
        assert "PASS" not in result.stdout


@pytest.mark.parametrize(
    "config_tokens",
    [
        ["--config", str(V5_CONFIG), "--config", str(V1_CONFIG)],
        ["--config", str(V1_CONFIG), "--config", str(V5_CONFIG)],
        ["--config", str(V5_CONFIG), "--config", str(V5_CONFIG)],
        ["--config", str(V4_CONFIG), "--config", str(V5_CONFIG)],
        ["--config=" + str(V5_CONFIG), "--config=" + str(V1_CONFIG)],
        ["--config", str(V5_CONFIG), "--config=" + str(V1_CONFIG)],
    ],
)
def test_contract_check_cli_rejects_duplicate_config_without_v1_fallback(config_tokens):
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(ROOT / "src"),
    }
    result = subprocess.run(
        [
            str(PYTHON), "-m", "sparse_rtdetr.baseline.smoke_launcher", "contract-check",
            "--repo-root", str(ROOT), *config_tokens,
        ],
        cwd=ROOT, env=env, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert "PASS" not in result.stdout
    _assert_r5_frozen()


def test_contract_check_parser_rejects_duplicate_config_and_defaults_to_none(monkeypatch):
    launcher = importlib.import_module("sparse_rtdetr.baseline.smoke_launcher")
    parser = _parser()
    default_args = parser.parse_args(["contract-check", "--repo-root", str(ROOT)])
    assert default_args.config is None
    single_args = parser.parse_args(["contract-check", "--repo-root", str(ROOT), "--config", str(V5_CONFIG)])
    assert single_args.config == V5_CONFIG
    subparsers = next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction))
    check_parser = subparsers.choices["contract-check"]
    config_action = next(action for action in check_parser._actions if "--config" in action.option_strings)
    assert type(config_action) is _SingleOccurrenceAction
    assert config_action.type is Path
    assert config_action.default is None
    calls = []
    monkeypatch.setattr(launcher, "contract_check", lambda *args: calls.append(args))
    with pytest.raises(SystemExit) as raised:
        launcher.main([
            "contract-check", "--repo-root", str(ROOT),
            "--config", str(V5_CONFIG), "--config", str(V1_CONFIG),
        ])
    assert raised.value.code == 2
    assert calls == []


@pytest.mark.parametrize(
    "option_tokens",
    [
        ["--c", str(V5_CONFIG)],
        ["--co", str(V5_CONFIG)],
        ["--con", str(V5_CONFIG)],
        ["--conf", str(V5_CONFIG)],
        ["--confi", str(V5_CONFIG)],
        ["--c=" + str(V5_CONFIG)],
        ["--co=" + str(V5_CONFIG)],
        ["--con=" + str(V5_CONFIG)],
        ["--conf=" + str(V5_CONFIG)],
        ["--confi=" + str(V5_CONFIG)],
        ["--r", str(ROOT)],
        ["--re", str(ROOT)],
        ["--rep", str(ROOT)],
        ["--repo", str(ROOT)],
        ["--repo-", str(ROOT)],
        ["--repo-r", str(ROOT)],
        ["--repo-ro", str(ROOT)],
        ["--repo-roo", str(ROOT)],
        ["--configuration", str(V5_CONFIG)],
        ["--configs", str(V5_CONFIG)],
        ["--config-path", str(V5_CONFIG)],
    ],
)
def test_contract_check_parser_rejects_abbreviated_or_lookalike_options(option_tokens):
    parser = _parser()
    with pytest.raises(SystemExit) as raised:
        parser.parse_args(["contract-check", "--repo-root", str(ROOT), *option_tokens])
    assert raised.value.code == 2


def test_contract_check_parser_disables_abbreviation_only_for_contract_check():
    parser = _parser()
    subparsers = next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction))
    check_parser = subparsers.choices["contract-check"]
    smoke_parser = subparsers.choices["smoke"]
    child_parser = subparsers.choices["_child"]
    assert check_parser.allow_abbrev is False
    assert smoke_parser.allow_abbrev is True
    assert child_parser.allow_abbrev is True


def test_contract_check_parser_and_child_argv_static_guards():
    launcher_path = ROOT / "src/sparse_rtdetr/baseline/smoke_launcher.py"
    tree = ast.parse(launcher_path.read_text(encoding="utf-8"))
    parser_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"]
    config_calls = [node for node in parser_calls if any(isinstance(arg, ast.Constant) and arg.value == "--config" for arg in node.args)]
    assert len(config_calls) == 2
    check_source = ast.get_source_segment(launcher_path.read_text(encoding="utf-8"), config_calls[0])
    assert check_source is not None and "type=Path" in check_source and "default=None" in check_source
    child_parser = _parser().parse_args(["_child", "--repo-root", str(ROOT)])
    assert child_parser.mode == "_child"
    with pytest.raises(SystemExit) as raised:
        _parser().parse_args(["_child", "--repo-root", str(ROOT), "--config", str(V5_CONFIG)])
    assert raised.value.code == 2
    child_argv = [str(PYTHON), "-m", "sparse_rtdetr.baseline.smoke_launcher", "_child", "--repo-root", str(ROOT)]
    assert "--config" not in child_argv
    assert "P3_SMOKE_CONFIG" not in {
        node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    contract_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "contract_check"
    ]
    assert any(
        len(node.args) == 2
        and isinstance(node.args[1], ast.Attribute)
        and node.args[1].attr == "config"
        for node in contract_calls
    )
    check_config_call = next(
        node for node in config_calls
        if any(isinstance(arg, ast.Constant) and arg.value == "--config" for arg in node.args)
        and any(keyword.arg == "action" for keyword in node.keywords)
    )
    action_keyword = next(keyword for keyword in check_config_call.keywords if keyword.arg == "action")
    assert isinstance(action_keyword.value, ast.Name)
    assert action_keyword.value.id == "_SingleOccurrenceAction"


def test_real_contract_check_cli_preserves_r5_absence(tmp_path):
    before = R5_FROZEN_FINGERPRINTS.copy()
    _assert_r5_frozen()
    _assert_contract_cli_pass(_real_contract_cli(V5_CONFIG)[1], SMOKE_V5_ID)
    assert {relative: _directory_fingerprint(ROOT / relative) for relative in before} == before


def test_clean_archive_real_contract_cli_matrix():
    export = Path(tempfile.mkdtemp(prefix="p3_inner_cli_export_", dir="/tmp"))
    try:
        subprocess.run(["git", "archive", "HEAD", "-o", str(export / "repo.tar")], cwd=ROOT, check=True)
        subprocess.run(["tar", "-xf", str(export / "repo.tar"), "-C", str(export)], check=True)
        for relative in (
            "src/sparse_rtdetr/baseline/smoke.py",
            "src/sparse_rtdetr/baseline/smoke_launcher.py",
            "tests/test_rtdetr_baseline_smoke.py",
            "docs/contracts/RTDETR_BASELINE_SMOKE_OUTER_LAUNCH_V2.md",
            "configs/baseline/rtdetrv2_r18_visdrone_smoke_v6.json",
            "configs/baseline/rtdetrv2_r18_visdrone_smoke_v7.json",
        ):
            destination = export / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, destination)
        r3_relative = "artifacts/data/visdrone_protocol_v2_conversion_r3"
        for name in (
            "train_core_manifest.json",
            "completion.json",
            "artifact_inventory.json",
            "config.json",
            "category_contract.json",
            "source_identity.json",
        ):
            destination = export / r3_relative / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / r3_relative / name, destination)
        archive_root = export
        if (archive_root / "artifacts/runs").exists():
            assert not any(path.name.startswith("rtdetrv2_r18_visdrone_baseline_smoke_r") for path in (archive_root / "artifacts/runs").glob("*"))
        for config_path, smoke_id in ((None, SMOKE_ID), (archive_root / "configs/baseline/rtdetrv2_r18_visdrone_smoke_v2.json", SMOKE_V2_ID), (archive_root / "configs/baseline/rtdetrv2_r18_visdrone_smoke_v3.json", SMOKE_V3_ID), (archive_root / "configs/baseline/rtdetrv2_r18_visdrone_smoke_v4.json", SMOKE_V4_ID), (archive_root / "configs/baseline/rtdetrv2_r18_visdrone_smoke_v5.json", SMOKE_V5_ID), (archive_root / "configs/baseline/rtdetrv2_r18_visdrone_smoke_v6.json", SMOKE_V6_ID), (archive_root / "configs/baseline/rtdetrv2_r18_visdrone_smoke_v7.json", SMOKE_V7_ID)):
            _, result = _real_contract_cli(config_path, repo_root=archive_root)
            _assert_contract_cli_pass(result, smoke_id)
        duplicate = subprocess.run(
            [
                str(PYTHON), "-m", "sparse_rtdetr.baseline.smoke_launcher", "contract-check",
                "--repo-root", str(archive_root),
                "--config", str(archive_root / "configs/baseline/rtdetrv2_r18_visdrone_smoke_v5.json"),
                "--config", str(archive_root / "configs/baseline/rtdetrv2_r18_visdrone_smoke_v1.json"),
            ],
            cwd=archive_root,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": str(archive_root / "src"),
            },
            capture_output=True,
            text=True,
            check=False,
        )
        assert duplicate.returncode == 2
        assert "PASS" not in duplicate.stdout
        for option_tokens in (
            ["--c", str(archive_root / "configs/baseline/rtdetrv2_r18_visdrone_smoke_v5.json")],
            ["--co", str(archive_root / "configs/baseline/rtdetrv2_r18_visdrone_smoke_v5.json")],
            ["--con", str(archive_root / "configs/baseline/rtdetrv2_r18_visdrone_smoke_v5.json")],
            ["--conf", str(archive_root / "configs/baseline/rtdetrv2_r18_visdrone_smoke_v5.json")],
            ["--confi", str(archive_root / "configs/baseline/rtdetrv2_r18_visdrone_smoke_v5.json")],
            ["--c=" + str(archive_root / "configs/baseline/rtdetrv2_r18_visdrone_smoke_v5.json")],
            ["--co=" + str(archive_root / "configs/baseline/rtdetrv2_r18_visdrone_smoke_v5.json")],
            ["--con=" + str(archive_root / "configs/baseline/rtdetrv2_r18_visdrone_smoke_v5.json")],
            ["--conf=" + str(archive_root / "configs/baseline/rtdetrv2_r18_visdrone_smoke_v5.json")],
            ["--confi=" + str(archive_root / "configs/baseline/rtdetrv2_r18_visdrone_smoke_v5.json")],
            ["--r", str(archive_root)],
            ["--re", str(archive_root)],
            ["--rep", str(archive_root)],
            ["--repo", str(archive_root)],
            ["--repo-", str(archive_root)],
            ["--repo-r", str(archive_root)],
            ["--repo-ro", str(archive_root)],
            ["--repo-roo", str(archive_root)],
            ["--configuration", str(archive_root / "configs/baseline/rtdetrv2_r18_visdrone_smoke_v5.json")],
            ["--configs", str(archive_root / "configs/baseline/rtdetrv2_r18_visdrone_smoke_v5.json")],
            ["--config-path", str(archive_root / "configs/baseline/rtdetrv2_r18_visdrone_smoke_v5.json")],
        ):
            abbreviated = subprocess.run(
                [
                    str(PYTHON), "-m", "sparse_rtdetr.baseline.smoke_launcher", "contract-check",
                    "--repo-root", str(archive_root), *option_tokens,
                ],
                cwd=archive_root,
                env={
                    **os.environ,
                    "CUDA_VISIBLE_DEVICES": "",
                    "PYTHONNOUSERSITE": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONPATH": str(archive_root / "src"),
                },
                capture_output=True,
                text=True,
                check=False,
            )
            assert abbreviated.returncode == 2
            assert "PASS" not in abbreviated.stdout
    finally:
        shutil.rmtree(export)


def test_v5_inner_contract_check_is_explicit_and_data_free(monkeypatch):
    smoke = _patch_synthetic_selection(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    monkeypatch.setattr(smoke, "verify_r3_binding", lambda *_args: SimpleNamespace(artifact_root=ROOT / "synthetic-r3"))
    result = contract_check(ROOT, V5_CONFIG)
    assert result["status"] == "PASS"
    assert result["smoke_id"] == SMOKE_V5_ID
    assert result["real_data_accessed"] is False
    assert result["dataset_or_dataloader_constructed"] is False
    assert result["model_constructed"] is False
    assert result["output_directory_created"] is False
    assert result["process_evidence_created"] is False
    code = (
        "import json, types\n"
        "import sparse_rtdetr.baseline.smoke as smoke\n"
        "smoke.load_frozen_image_selection = lambda _root, **kwargs: smoke.FROZEN_IMAGE_RECORDS\n"
        "smoke.verify_r3_binding = lambda *_args: types.SimpleNamespace(artifact_root='synthetic-r3')\n"
        f"result = smoke.contract_check({str(ROOT)!r}, {str(V5_CONFIG)!r})\n"
        "assert result['status'] == 'PASS' and result['smoke_id'] == 'rtdetrv2_r18_visdrone_baseline_smoke_v5'\n"
        "assert result['torch_imported'] is False\n"
        "print(json.dumps(result, sort_keys=True))\n"
    )
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(ROOT / "src")}
    completed = subprocess.run([str(PYTHON), "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr


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
        (V4_CONFIG, SMOKE_V4_OUTPUT_RELATIVE, SMOKE_V4_PROCESS_RELATIVE),
        (V5_CONFIG, SMOKE_V5_OUTPUT_RELATIVE, SMOKE_V5_PROCESS_RELATIVE),
        (V6_CONFIG, SMOKE_V6_OUTPUT_RELATIVE, SMOKE_V6_PROCESS_RELATIVE),
        (V7_CONFIG, SMOKE_V7_OUTPUT_RELATIVE, SMOKE_V7_PROCESS_RELATIVE),
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
        (V3_CONFIG, SMOKE_V4_OUTPUT_RELATIVE, SMOKE_V4_PROCESS_RELATIVE),
        (V4_CONFIG, SMOKE_V3_OUTPUT_RELATIVE, SMOKE_V3_PROCESS_RELATIVE),
        (V4_CONFIG, SMOKE_V2_OUTPUT_RELATIVE, SMOKE_V2_PROCESS_RELATIVE),
        (V5_CONFIG, SMOKE_V4_OUTPUT_RELATIVE, SMOKE_V4_PROCESS_RELATIVE),
        (V4_CONFIG, SMOKE_V5_OUTPUT_RELATIVE, SMOKE_V5_PROCESS_RELATIVE),
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
    ids = (SMOKE_ID, SMOKE_V2_ID, SMOKE_V3_ID, SMOKE_V4_ID, SMOKE_V5_ID, SMOKE_V6_ID, SMOKE_V7_ID)
    assert tuple(get_smoke_runtime_spec(value).smoke_id for value in ids) == ids
    assert len({get_smoke_runtime_spec(value).config_relative_path for value in ids}) == 7
    assert get_smoke_runtime_spec(SMOKE_V4_ID).output_relative_path.endswith("_r4")
    assert get_smoke_runtime_spec(SMOKE_V5_ID).output_relative_path.endswith("_r5")
    assert get_smoke_runtime_spec(SMOKE_V6_ID).output_relative_path.endswith("_r6")
    assert get_smoke_runtime_spec(SMOKE_V7_ID).output_relative_path.endswith("_r7")
    for value in ("unknown_smoke", "", True, None):
        with pytest.raises(SmokeContractError):
            get_smoke_runtime_spec(value)


@pytest.mark.parametrize("config_path, forged_id", [(V2_CONFIG, SMOKE_V3_ID), (V3_CONFIG, SMOKE_V2_ID), (V3_CONFIG, SMOKE_V4_ID), (V4_CONFIG, SMOKE_V3_ID)])
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


def test_v4_config_is_frozen_identity_variant_of_v3():
    v3 = json.loads(V3_CONFIG.read_text(encoding="utf-8"))
    v4 = json.loads(V4_CONFIG.read_text(encoding="utf-8"))
    assert hashlib.sha256(V4_CONFIG.read_bytes()).hexdigest() == "564da04b483a4bfad6f640010f49f157ee97a4bde5f5f1c7f1db926390057b47"
    assert hashlib.sha256(canonical_json_bytes(v4)).hexdigest() == "9a0d8a304b35b955bec24ab13940002a2660b2c9653c567526bd1d0addc47d5e"
    assert hashlib.sha256(V3_CONFIG.read_bytes()).hexdigest() == "70d181632987f63edf2d63362594549a36a7dc2ef94c4f94f02ea9f170e055f9"
    assert hashlib.sha256(V2_CONFIG.read_bytes()).hexdigest() == "a5f751eb39e29df36972f66b52c9517c50558b075873659509b2450711cbd11c"
    allowed_runtime = {
        "config_relative_path",
        "output_relative_path",
        "process_evidence_relative_path",
        "outer_launch_evidence_relative_path",
        "tmux_session_name",
    }
    assert {key for key in v3["runtime"] if v3["runtime"].get(key) != v4["runtime"].get(key)} == allowed_runtime
    for field in ("execution", "model", "image_selection", "evidence", "mode", "split_role"):
        assert v3[field] == v4[field]
    config, config_path, binding = _config_binding(ROOT, V4_CONFIG)
    assert config_path == V4_CONFIG
    assert config["smoke_id"] == SMOKE_V4_ID
    assert binding["config_file_sha256"] == hashlib.sha256(V4_CONFIG.read_bytes()).hexdigest()
    assert binding["config_canonical_sha256"] == hashlib.sha256(canonical_json_bytes(v4)).hexdigest()


def _json_pointer_diffs(left, right, path=""):
    if isinstance(left, dict) and isinstance(right, dict):
        differences = []
        for key in sorted(set(left) | set(right)):
            child_path = f"{path}/{key}"
            if key not in left or key not in right:
                differences.append(child_path)
            else:
                differences.extend(_json_pointer_diffs(left[key], right[key], child_path))
        return differences
    if isinstance(left, list) and isinstance(right, list):
        differences = []
        if len(left) != len(right):
            differences.append(path)
        for index, (left_value, right_value) in enumerate(zip(left, right)):
            differences.extend(_json_pointer_diffs(left_value, right_value, f"{path}/{index}"))
        return differences
    return [] if type(left) is type(right) and left == right else [path]


def test_v5_scientific_identity_matches_v4_at_exactly_six_pointers():
    v4 = json.loads(V4_CONFIG.read_text(encoding="utf-8"))
    v5 = json.loads(V5_CONFIG.read_text(encoding="utf-8"))
    expected = {
        "/smoke_id",
        "/runtime/config_relative_path",
        "/runtime/output_relative_path",
        "/runtime/process_evidence_relative_path",
        "/runtime/outer_launch_evidence_relative_path",
        "/runtime/tmux_session_name",
    }
    assert set(_json_pointer_diffs(v4, v5)) == expected
    assert v5["smoke_id"] == SMOKE_V5_ID
    assert v5["runtime"] == {
        **v4["runtime"],
        "config_relative_path": SMOKE_V5_CONFIG_RELATIVE,
        "output_relative_path": SMOKE_V5_OUTPUT_RELATIVE,
        "process_evidence_relative_path": SMOKE_V5_PROCESS_RELATIVE,
        "outer_launch_evidence_relative_path": SMOKE_V5_OUTER_RELATIVE,
        "tmux_session_name": SMOKE_V5_TMUX_SESSION,
    }
    assert v5["runtime"]["tmux_client_timeout_seconds"] == 10

    seventh = copy.deepcopy(v5)
    seventh["model"]["seed"] = 1
    assert "/model/seed" in _json_pointer_diffs(v4, seventh)
    typed = copy.deepcopy(v5)
    typed["runtime"]["tmux_client_timeout_seconds"] = 10.0
    assert "/runtime/tmux_client_timeout_seconds" in _json_pointer_diffs(v4, typed)


def test_v5_raw_and_canonical_config_identity_is_frozen_from_bytes():
    raw = V5_CONFIG.read_bytes()
    value = json.loads(raw)
    canonical = canonical_json_bytes(value)
    assert len(raw) == 3631
    assert hashlib.sha256(raw).hexdigest() == "18c425805f33726d0bbe1834e295366a4aa39e5ad830f1f1f3cf92b3d1509716"
    assert len(canonical) == 2922
    assert hashlib.sha256(canonical).hexdigest() == "ed39e40602b716fcc7e94bb521dbcf66ce6f57436b9cbc99ed7ed778462f672b"


def test_v6_is_a_v5_identity_pointer_variant_with_stable_bytes():
    v5 = json.loads(V5_CONFIG.read_text(encoding="utf-8"))
    v6 = json.loads(V6_CONFIG.read_text(encoding="utf-8"))
    allowed = {
        "/smoke_id",
        "/runtime/config_relative_path",
        "/runtime/output_relative_path",
        "/runtime/process_evidence_relative_path",
        "/runtime/outer_launch_evidence_relative_path",
        "/runtime/tmux_session_name",
    }
    assert set(_json_pointer_diffs(v5, v6)) == allowed
    assert v6["smoke_id"] == SMOKE_V6_ID
    assert v6["runtime"] == {
        **v5["runtime"],
        "config_relative_path": SMOKE_V6_CONFIG_RELATIVE,
        "output_relative_path": SMOKE_V6_OUTPUT_RELATIVE,
        "process_evidence_relative_path": SMOKE_V6_PROCESS_RELATIVE,
        "outer_launch_evidence_relative_path": SMOKE_V6_OUTER_RELATIVE,
        "tmux_session_name": SMOKE_V6_TMUX_SESSION,
    }
    raw = V6_CONFIG.read_bytes()
    assert raw.startswith(b"{") and b"\r" not in raw and not raw.startswith(b"\xef\xbb\xbf")
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert hashlib.sha256(raw).hexdigest() == hashlib.sha256(V6_CONFIG.read_bytes()).hexdigest()
    assert hashlib.sha256(canonical_json_bytes(v6)).hexdigest() == hashlib.sha256(canonical_json_bytes(json.loads(raw))).hexdigest()


def test_v7_is_v6_identity_variant_plus_frozen_dual_gate_policy():
    v6 = json.loads(V6_CONFIG.read_text(encoding="utf-8"))
    v7 = json.loads(V7_CONFIG.read_text(encoding="utf-8"))
    assert set(_json_pointer_diffs(v6, v7)) == {
        "/smoke_id", "/runtime/config_relative_path", "/runtime/output_relative_path",
        "/runtime/process_evidence_relative_path", "/runtime/outer_launch_evidence_relative_path",
        "/runtime/tmux_session_name", "/evidence_policy",
    }
    assert load_smoke_config(ROOT, V7_CONFIG) == v7
    assert v7["evidence_policy"] == {
        "policy_version": 1,
        "certification_policy": "DUAL_GATE",
        "durable_terminal_evidence_required": True,
        "immediate_snapshot": {
            "required": True, "owner": "outer_launcher", "capture_count": 1,
            "max_delay_seconds": 1.0, "observer_timeout_seconds": 0.25,
            "max_observer_elapsed_seconds": 0.5, "response_binding_required": True,
            "timing_tolerance_seconds": 0.01,
        },
    }
    assert get_smoke_runtime_spec(SMOKE_V7_ID).output_relative_path == SMOKE_V7_OUTPUT_RELATIVE


@pytest.mark.parametrize("mutation", [
    "v5_id_v4_path",
    "v4_id_v5_path",
    "relative_path",
    "output_path",
    "process_path",
    "outer_path",
    "session",
    "timeout_bool",
    "timeout_float",
    "timeout_string",
])
def test_v5_config_mutation_matrix_fails_closed(mutation):
    smoke = importlib.import_module("sparse_rtdetr.baseline.smoke")
    config = json.loads(V5_CONFIG.read_text(encoding="utf-8"))
    if mutation == "v5_id_v4_path":
        config["runtime"]["config_relative_path"] = SMOKE_V4_CONFIG_RELATIVE
    elif mutation == "v4_id_v5_path":
        config["smoke_id"] = SMOKE_V4_ID
    elif mutation == "relative_path":
        config["runtime"]["config_relative_path"] = "configs/baseline/changed.json"
    elif mutation == "output_path":
        config["runtime"]["output_relative_path"] = SMOKE_V4_OUTPUT_RELATIVE
    elif mutation == "process_path":
        config["runtime"]["process_evidence_relative_path"] = SMOKE_V4_PROCESS_RELATIVE
    elif mutation == "outer_path":
        config["runtime"]["outer_launch_evidence_relative_path"] = SMOKE_V4_OUTER_RELATIVE
    elif mutation == "session":
        config["runtime"]["tmux_session_name"] = "p3_rtdetrv2_r18_visdrone_baseline_smoke_r4"
    elif mutation == "timeout_bool":
        config["runtime"]["tmux_client_timeout_seconds"] = True
    elif mutation == "timeout_float":
        config["runtime"]["tmux_client_timeout_seconds"] = 10.0
    else:
        config["runtime"]["tmux_client_timeout_seconds"] = "10"
    with pytest.raises(SmokeContractError):
        smoke.validate_smoke_config(config)


def test_v5_duplicate_and_symlink_config_files_fail_closed(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    with pytest.raises(SmokeContractError):
        load_smoke_config(ROOT, duplicate)
    link = tmp_path / "v5-link.json"
    link.symlink_to(V5_CONFIG)
    with pytest.raises(SmokeContractError):
        load_smoke_config(ROOT, link)


@pytest.mark.parametrize(
    ("config_path", "output_relative", "process_relative"),
    [
        (V1_CONFIG, "artifacts/runs/rtdetrv2_r18_visdrone_baseline_smoke_r1", "artifacts/process_evidence/rtdetrv2_r18_visdrone_baseline_smoke_r1"),
        (V2_CONFIG, "artifacts/runs/rtdetrv2_r18_visdrone_baseline_smoke_r2", "artifacts/process_evidence/rtdetrv2_r18_visdrone_baseline_smoke_r2"),
        (V3_CONFIG, SMOKE_V3_OUTPUT_RELATIVE, SMOKE_V3_PROCESS_RELATIVE),
        (V4_CONFIG, SMOKE_V4_OUTPUT_RELATIVE, SMOKE_V4_PROCESS_RELATIVE),
    ],
)
def test_main_passes_bound_version_config_to_fake_launch_without_creating_artifacts(monkeypatch, tmp_path, config_path, output_relative, process_relative):
    launcher = importlib.import_module("sparse_rtdetr.baseline.smoke_launcher")
    _patch_synthetic_selection(monkeypatch)
    captured = {}
    monkeypatch.setattr(launcher, "validate_real_smoke_environment", lambda: {"cpu_fallback": False})
    if config_path == V1_CONFIG:
        monkeypatch.delenv("P3_PANE_NONCE", raising=False)
    else:
        monkeypatch.setenv("P3_PANE_NONCE", "a" * 32)

    def fake_launch(child_argv, output_dir, process_dir, **kwargs):
        captured.update({"argv": child_argv, "output": output_dir, "process": process_dir, **kwargs})
        return 0

    monkeypatch.setattr(launcher, "launch_smoke", fake_launch)
    data_root = tmp_path / "synthetic-data"
    data_root.mkdir()
    output = ROOT / output_relative
    process = ROOT / process_relative
    output_exists_before = output.exists()
    process_exists_before = process.exists()
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
    assert captured["nonce"] == (None if config_path == V1_CONFIG else "a" * 32)
    assert "--config" in smoke_argv
    assert "--config" not in captured["argv"]
    assert "P3_PANE_NONCE" not in captured["argv"]
    parsed = launcher._parser().parse_args(captured["argv"][3:])
    assert parsed.mode == "_child"
    assert parsed.repo_root == ROOT
    assert output.exists() == output_exists_before
    assert process.exists() == process_exists_before


@pytest.mark.parametrize("pane_nonce", [None, "", "a" * 31, "a" * 33, "A" * 32, "g" * 32, "a" * 31 + "!", "a" * 31 + " ", "a" * 31 + "\n"])
def test_outer_enabled_main_rejects_missing_or_invalid_pane_nonce_before_launch(monkeypatch, tmp_path, pane_nonce):
    launcher = importlib.import_module("sparse_rtdetr.baseline.smoke_launcher")
    _patch_synthetic_selection(monkeypatch)
    monkeypatch.setattr(launcher, "validate_real_smoke_environment", lambda: {"cpu_fallback": False})
    monkeypatch.setattr(launcher.secrets, "token_hex", lambda _size: pytest.fail("outer-enabled smoke used nonce fallback"))
    launch_calls = []
    monkeypatch.setattr(launcher, "launch_smoke", lambda *args, **kwargs: launch_calls.append((args, kwargs)) or 0)
    if pane_nonce is None:
        monkeypatch.delenv("P3_PANE_NONCE", raising=False)
    else:
        monkeypatch.setenv("P3_PANE_NONCE", pane_nonce)
    data_root = tmp_path / "synthetic-data"
    data_root.mkdir()
    output = ROOT / SMOKE_V4_OUTPUT_RELATIVE
    process = ROOT / SMOKE_V4_PROCESS_RELATIVE
    output_exists_before = output.exists()
    process_exists_before = process.exists()
    result = launcher.main([
        "smoke",
        "--repo-root", str(ROOT),
        "--data-root", str(data_root),
        "--output-dir", str(output),
        "--process-evidence-dir", str(process),
        "--config", str(V4_CONFIG),
        "--child-python", str(PYTHON),
    ])
    assert result == 2
    assert launch_calls == []
    assert output.exists() == output_exists_before
    assert process.exists() == process_exists_before


def test_old_child_argv_with_config_remains_argparse_failure():
    launcher = importlib.import_module("sparse_rtdetr.baseline.smoke_launcher")
    with pytest.raises(SystemExit) as raised:
        launcher._parser().parse_args([
            "_child",
            "--repo-root", str(ROOT),
            "--config", str(V3_CONFIG),
        ])
    assert raised.value.code == 2


def test_real_module_child_handoff_consumes_and_calls_cpu_stub_once(tmp_path, monkeypatch):
    launcher = importlib.import_module("sparse_rtdetr.baseline.smoke_launcher")
    monkeypatch.setenv("P3_RTDETR_BASELINE_SMOKE_AUTHORIZED", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    sitecustomize = tmp_path / "sitecustomize.py"
    _write_module_child_sitecustomize(sitecustomize)
    marker = tmp_path / "stub_calls"
    env = _module_child_env(tmp_path, sitecustomize)
    env.update({"P3_SMOKE_TEST_STUB": "1", "P3_SMOKE_TEST_MARKER": str(marker)})
    output = tmp_path / "entry"
    process = tmp_path / "process"
    result = launch_smoke(
        launcher._canonical_child_argv(PYTHON, ROOT),
        output,
        process,
        data_root=_runtime_data_root(tmp_path),
        child_python=PYTHON,
        repo_root=ROOT,
        env=env,
        nonce="6" * 32,
    )
    assert result == 0
    assert marker.read_text(encoding="utf-8") == "called\n"
    prepared = _read_object(process / "handoff_prepared.json")
    receipt = _read_object(process / "handoff_receipt.json")
    expected = launcher._canonical_child_argv(PYTHON, ROOT)
    assert prepared["child_argv_schema"] == "module-v1"
    assert prepared["child_argv"] == expected
    assert prepared["child_argv_sha256"] == launcher._child_argv_sha256(expected)
    assert receipt["child_argv"] == expected
    assert receipt["child_argv_sha256"] == prepared["child_argv_sha256"]
    assert receipt["child_executable_identity"] == prepared["child_executable_identity"]
    assert receipt["child_pid"] == _read_object(process / "process_completion.json")["child_pid"]


@pytest.mark.parametrize("mutation", [
    "extra", "missing", "reordered", "duplicate", "config", "module", "subcommand", "repo", "sha", "identity", "orig_argv",
])
def test_real_module_child_handoff_rejects_argv_binding_matrix(tmp_path, monkeypatch, mutation):
    launcher = importlib.import_module("sparse_rtdetr.baseline.smoke_launcher")
    monkeypatch.setenv("P3_RTDETR_BASELINE_SMOKE_AUTHORIZED", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    sitecustomize = tmp_path / "sitecustomize.py"
    _write_module_child_sitecustomize(sitecustomize)
    env = _module_child_env(tmp_path, sitecustomize)
    env["P3_SMOKE_TEST_MUTATION"] = mutation
    process = tmp_path / "process"
    output = tmp_path / "entry"
    result = launch_smoke(
        launcher._canonical_child_argv(PYTHON, ROOT),
        output,
        process,
        data_root=_runtime_data_root(tmp_path),
        child_python=PYTHON,
        repo_root=ROOT,
        env=env,
        nonce="7" * 32,
    )
    assert result == 2
    assert not (process / "handoff_receipt.json").exists()
    assert not output.exists()
    assert _read_object(process / "handoff_prepared.json")["consumed"] is False
    assert "SmokeLauncherError" in (process / "process_console.log").read_text(encoding="utf-8")


@pytest.mark.parametrize("raw", [b"one", b"one\x00\x00", b"\xff\x00", b""])
def test_proc_cmdline_parser_rejects_corrupt_bytes(tmp_path, monkeypatch, raw):
    launcher = importlib.import_module("sparse_rtdetr.baseline.smoke_launcher")
    path = tmp_path / "cmdline"
    path.write_bytes(raw)
    monkeypatch.setattr(launcher, "_PROC_CMDLINE_PATH", path)
    with pytest.raises(SmokeLauncherError):
        launcher._read_proc_cmdline()


def test_real_child_argv_roundtrip_dispatches_from_v3_receipt(monkeypatch, tmp_path):
    launcher = importlib.import_module("sparse_rtdetr.baseline.smoke_launcher")
    captured = {}
    monkeypatch.setattr(launcher, "validate_real_smoke_environment", lambda: {"stub": True})
    monkeypatch.setenv("P3_PANE_NONCE", "a" * 32)

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
    assert captured["kwargs"]["nonce"] == "a" * 32

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
    assert process.is_dir()


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
        allow_noncanonical_child=True,
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


def test_scientific_source_allowlist_is_exact_ast_dependency_closure():
    assert CHECKER._scientific_dependency_closure(ROOT) == set(CHECKER.SCIENTIFIC_SOURCE_ALLOWLIST)


def test_scientific_dependency_mutation_is_detected(tmp_path):
    root = tmp_path / "repo"
    source = root / "src/sparse_rtdetr/baseline/smoke.py"
    source.parent.mkdir(parents=True)
    source.write_text("from sparse_rtdetr.data_protocol.new_semantics import value\n", encoding="utf-8")
    dependency = root / "src/sparse_rtdetr/data_protocol/new_semantics.py"
    dependency.parent.mkdir(parents=True)
    dependency.write_text("value = 1\n", encoding="utf-8")
    for relative in CHECKER.SCIENTIFIC_ENTRY_ROOTS[1:]:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    failures = CHECKER._scientific_dependency_failures(root)
    assert "scientific source dependency missing from allowlist: src/sparse_rtdetr/data_protocol/new_semantics.py" in failures


def test_synthetic_writer_is_independent_of_ignored_r3_artifact(tmp_path):
    archive = tmp_path / "archive"
    subprocess.run(["git", "archive", "HEAD", "-o", str(tmp_path / "repo.tar")], cwd=ROOT, check=True)
    archive.mkdir()
    subprocess.run(["tar", "-xf", str(tmp_path / "repo.tar"), "-C", str(archive)], check=True)
    archive_root = archive
    for relative in ("src/sparse_rtdetr/baseline/smoke.py", "src/sparse_rtdetr/baseline/smoke_evidence.py"):
        destination = archive_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    output = tmp_path / "synthetic-entry"
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(archive_root / "src"),
    }
    script = (
        "from pathlib import Path\n"
        "from sparse_rtdetr.baseline.smoke import run_synthetic_smoke\n"
        "from sparse_rtdetr.baseline.smoke_evidence import validate_entry_output\n"
        f"root = Path({str(archive_root)!r})\n"
        f"output = Path({str(output)!r})\n"
        "run_synthetic_smoke(root, output)\n"
        "assert validate_entry_output(output, repo_root=root)['entry_success_accepted']\n"
    )
    result = subprocess.run([str(PYTHON), "-c", script], cwd=archive_root, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    binding = json.loads((output / "data_binding_audit.json").read_text(encoding="utf-8"))
    selection = json.loads((output / "image_selection_audit.json").read_text(encoding="utf-8"))
    assert binding["runtime_artifacts_accessed"] is False
    assert selection["verified_from_manifest"] is False


def test_real_mode_cannot_use_synthetic_selection_bypass(tmp_path):
    config = tmp_path / "smoke.json"
    config.write_bytes(V1_CONFIG.read_bytes())
    with pytest.raises(SmokeContractError):
        load_smoke_config(tmp_path, config, runtime_mode="real")


def test_v2_scientific_evidence_is_rejected(tmp_path):
    output = tmp_path / "entry"
    run_synthetic_smoke(ROOT, output)
    path = output / "input_batch_audit.json"
    value = _read_object(path)
    value["schema_version"] = 2
    path.write_bytes(canonical_json_bytes(value))
    _repack_entry(output)
    with pytest.raises(SmokeEvidenceError):
        validate_entry_output(output)


@pytest.mark.parametrize("field", ["device_name", "device_capability", "torch_version", "torch_cuda_version", "current_device", "device_count", "cuda_visible_devices", "runtime_capture_after_cuda_initialization", "runtime_mode"])
def test_synthetic_cuda_identity_exact_matrix_rejects_drift(tmp_path, field):
    output = tmp_path / field
    run_synthetic_smoke(ROOT, output)
    path = output / "cuda_runtime_identity.json"
    value = _read_object(path)
    value[field] = {
        "device_name": "other",
        "device_capability": [9, 0],
        "torch_version": "other",
        "torch_cuda_version": "other",
        "current_device": 0,
        "device_count": 1,
        "cuda_visible_devices": "0",
        "runtime_capture_after_cuda_initialization": True,
        "runtime_mode": "real",
    }[field]
    path.write_bytes(canonical_json_bytes(value))
    _repack_entry(output)
    with pytest.raises(SmokeEvidenceError):
        validate_entry_output(output)


def test_model_identity_forged_parameter_rejected_after_full_repack(tmp_path):
    output = tmp_path / "forged-parameter"
    run_synthetic_smoke(ROOT, output)

    def mutate(value):
        row = _forged_parameter_row()
        value["parameters"] = 1
        value["trainable_parameters"] = 1
        value["parameter_tensor_count"] = 1
        value["parameter_inventory"] = [row]
        value["buffer_count"] = 0
        value["buffer_inventory"] = []

    _repack_model_identity(output, mutate)
    with pytest.raises(SmokeEvidenceError):
        validate_entry_output(output)


@pytest.mark.parametrize("parameter_count", [20094583, 20094585])
def test_model_identity_parameter_count_boundary_rejected_after_repack(tmp_path, parameter_count):
    output = tmp_path / str(parameter_count)
    run_synthetic_smoke(ROOT, output)

    def mutate(value):
        row = _forged_parameter_row(numel=parameter_count)
        value["parameters"] = parameter_count
        value["trainable_parameters"] = parameter_count
        value["parameter_tensor_count"] = 1
        value["parameter_inventory"] = [row]

    _repack_model_identity(output, mutate)
    with pytest.raises(SmokeEvidenceError):
        validate_entry_output(output)


def test_model_identity_forged_single_row_contract_size_rejected_after_repack(tmp_path):
    output = tmp_path / "forged-contract-size"
    run_synthetic_smoke(ROOT, output)

    def mutate(value):
        row = _forged_parameter_row(numel=20094584)
        value["parameters"] = 20094584
        value["trainable_parameters"] = 20094584
        value["parameter_tensor_count"] = 1
        value["parameter_inventory"] = [row]

    _repack_model_identity(output, mutate)
    with pytest.raises(SmokeEvidenceError):
        validate_entry_output(output)


@pytest.mark.parametrize("mutation", [
    lambda value: value["parameter_inventory"].append(_forged_parameter_row(name="added.weight")),
    lambda value: value["parameter_inventory"].pop(),
    lambda value: value["parameter_inventory"].reverse(),
    lambda value: value["parameter_inventory"].__setitem__(0, {**value["parameter_inventory"][0], "name": "renamed.weight"}),
    lambda value: value["parameter_inventory"].__setitem__(0, {**value["parameter_inventory"][0], "logical_sha256": "0" * 64}),
    lambda value: value["parameter_inventory"].__setitem__(0, {**value["parameter_inventory"][0], "requires_grad": False}),
    lambda value: value["parameter_inventory"].__setitem__(0, {**value["parameter_inventory"][0], "dtype": "torch.float64"}),
    lambda value: value["parameter_inventory"].__setitem__(0, {**value["parameter_inventory"][0], "shape": [1], "numel": 1}),
    lambda value: value["parameter_inventory"].__setitem__(0, {**value["parameter_inventory"][0], "kind": "buffer"}),
    lambda value: value["parameter_inventory"].__setitem__(0, {**value["parameter_inventory"][0], "numel": True}),
    lambda value: value["parameter_inventory"].__setitem__(0, {**value["parameter_inventory"][0], "name": ""}),
    lambda value: value["parameter_inventory"].__setitem__(0, {}),
    lambda value: value["parameter_inventory"].__setitem__(0, {key: child for key, child in value["parameter_inventory"][0].items() if key != "dtype"}),
    lambda value: value["parameter_inventory"].__setitem__(0, {**value["parameter_inventory"][0], "unexpected": 1}),
])
def test_model_identity_inventory_mutations_rejected_after_repack(tmp_path, mutation):
    output = tmp_path / "inventory-mutation"
    run_synthetic_smoke(ROOT, output)
    base_rows = [_forged_parameter_row(name="base.one"), _forged_parameter_row(name="base.two")]

    def seed_inventory(value):
        value["parameter_inventory"] = [dict(row) for row in base_rows]
        value["parameters"] = value["trainable_parameters"] = 2
        value["parameter_tensor_count"] = 2
        value["buffer_inventory"] = []
        value["buffer_count"] = 0

    def apply_mutation(value):
        seed_inventory(value)
        mutation(value)

    _repack_model_identity(output, apply_mutation)
    with pytest.raises(SmokeEvidenceError):
        validate_entry_output(output)


def test_model_identity_duplicate_parameter_and_buffer_names_rejected_after_repack(tmp_path):
    output = tmp_path / "duplicate-names"
    run_synthetic_smoke(ROOT, output)

    def mutate(value):
        row = _forged_parameter_row(name="duplicate")
        value["parameter_inventory"] = [row]
        value["buffer_inventory"] = [{**row, "kind": "buffer", "requires_grad": False}]
        value["parameters"] = value["trainable_parameters"] = 1
        value["parameter_tensor_count"] = value["buffer_count"] = 1

    _repack_model_identity(output, mutate)
    with pytest.raises(SmokeEvidenceError):
        validate_entry_output(output)


def test_model_identity_duplicate_parameter_names_rejected_after_repack(tmp_path):
    output = tmp_path / "duplicate-parameters"
    run_synthetic_smoke(ROOT, output)

    def mutate(value):
        first = _forged_parameter_row(name="duplicate")
        second = _forged_parameter_row(name="duplicate")
        value["parameter_inventory"] = [first, second]
        value["buffer_inventory"] = []
        value["parameters"] = value["trainable_parameters"] = 2
        value["parameter_tensor_count"] = 2
        value["buffer_count"] = 0

    _repack_model_identity(output, mutate)
    with pytest.raises(SmokeEvidenceError):
        validate_entry_output(output)


@pytest.mark.parametrize("field", ["parameters", "trainable_parameters", "parameter_tensor_count", "buffer_count"])
def test_synthetic_model_state_counters_cannot_be_forged_after_repack(tmp_path, field):
    output = tmp_path / field
    run_synthetic_smoke(ROOT, output)
    _repack_model_identity(output, lambda value: value.update({field: 1}))
    with pytest.raises(SmokeEvidenceError):
        validate_entry_output(output)


def test_synthetic_nonempty_model_writer_fails_closed(tmp_path):
    nn = importlib.import_module("torch.nn")

    class NonemptySyntheticModel(SyntheticSmokeModel, nn.Module):
        def __init__(self):
            nn.Module.__init__(self)
            SyntheticSmokeModel.__init__(self)
            self.forged = nn.Parameter(torch.zeros(1))

    with pytest.raises(SmokeEvidenceError):
        run_synthetic_smoke(ROOT, tmp_path / "nonempty-model", model_factory=NonemptySyntheticModel)


def test_synthetic_nonempty_buffer_writer_fails_closed(tmp_path):
    nn = importlib.import_module("torch.nn")

    class NonemptySyntheticBufferModel(SyntheticSmokeModel, nn.Module):
        def __init__(self):
            nn.Module.__init__(self)
            SyntheticSmokeModel.__init__(self)
            self.register_buffer("forged_buffer", torch.zeros(1))

    with pytest.raises(SmokeEvidenceError):
        run_synthetic_smoke(ROOT, tmp_path / "nonempty-buffer", model_factory=NonemptySyntheticBufferModel)


def test_real_cpu_model_identity_is_deterministic_without_forward():
    first = _canonical_real_state_inventory(ROOT)
    second = _canonical_real_state_inventory(ROOT)
    assert first == second
    assert sum(row["numel"] for row in first[0]) == 20094584
    assert len(first[0]) == 326
    assert len(first[1]) == 212


def _fresh_real_cpu_model():
    import numpy as np

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    model = build_r18_cpu_model(ROOT)
    model.eval()
    return model


def _state_parent_and_leaf(model, name):
    parent_name, leaf_name = name.rsplit(".", 1)
    return model.get_submodule(parent_name), leaf_name


def _delete_state_entry(model, collection, name):
    parent, leaf_name = _state_parent_and_leaf(model, name)
    getattr(parent, collection).pop(leaf_name)


def _rename_state_entry(model, collection, name, new_name):
    parent, leaf_name = _state_parent_and_leaf(model, name)
    registry = getattr(parent, collection)
    registry[new_name] = registry.pop(leaf_name)


def _first_ordinary_parameter(model):
    return next(iter(model.named_parameters()))


def _first_ordinary_buffer(model):
    return next(name_and_value for name_and_value in model.named_buffers() if name_and_value[0] not in {"decoder.anchors", "decoder.valid_mask"})


def _set_first_element(model, collection, name, value):
    _, leaf_name = _state_parent_and_leaf(model, name)
    parent, _ = _state_parent_and_leaf(model, name)
    getattr(parent, collection)[leaf_name].data.view(-1)[0] = value


def _replace_buffer_with_forged_entry(model):
    name, _ = _first_ordinary_buffer(model)
    _delete_state_entry(model, "_buffers", name)
    model.register_buffer("forged_finite_buffer", torch.ones(1))


def _replace_parameter_with_forged_entry(model):
    import torch.nn as nn

    name, _ = _first_ordinary_parameter(model)
    _delete_state_entry(model, "_parameters", name)
    model.register_parameter("forged_parameter", nn.Parameter(torch.ones(1)))


def _replace_buffer_same_shape(model):
    name, old = _first_ordinary_buffer(model)
    parent, leaf_name = _state_parent_and_leaf(model, name)
    parent._buffers[leaf_name] = torch.ones_like(old)


def _replace_parameter_same_shape(model):
    import torch.nn as nn

    name, old = _first_ordinary_parameter(model)
    parent, leaf_name = _state_parent_and_leaf(model, name)
    parent._parameters[leaf_name] = nn.Parameter(torch.ones_like(old), requires_grad=old.requires_grad)


def _mutate_anchor(model):
    index = int(torch.nonzero(model.decoder.valid_mask.squeeze(-1), as_tuple=False)[0, 1].item())
    model.decoder.anchors.data[0, index, 0] = float("nan")


def _mutate_valid_mask(model):
    index = int(torch.nonzero(model.decoder.valid_mask.squeeze(-1), as_tuple=False)[0, 1].item())
    model.decoder.valid_mask.data[0, index, 0] = False


def _misalign_anchor_and_mask(model):
    valid_index = int(torch.nonzero(model.decoder.valid_mask.squeeze(-1), as_tuple=False)[0, 1].item())
    invalid_index = int(torch.nonzero(~model.decoder.valid_mask.squeeze(-1), as_tuple=False)[0, 1].item())
    model.decoder.anchors.data[0, valid_index, 0] = float("inf")
    model.decoder.anchors.data[0, invalid_index, 0] = 0.0


@pytest.mark.parametrize("mutation", [
    lambda model: model.register_buffer("forged_inf_buffer", torch.tensor([float("inf")])),
    lambda model: model.register_buffer("forged_finite_buffer", torch.tensor([1.0])),
    lambda model: model.register_buffer("forged_nan_buffer", torch.tensor([float("nan")])),
    lambda model: _set_first_element(model, "_parameters", _first_ordinary_parameter(model)[0], float("nan")),
    lambda model: _set_first_element(model, "_parameters", _first_ordinary_parameter(model)[0], float("inf")),
    lambda model: _set_first_element(model, "_parameters", _first_ordinary_parameter(model)[0], float("-inf")),
    lambda model: _delete_state_entry(model, "_parameters", _first_ordinary_parameter(model)[0]),
    lambda model: _delete_state_entry(model, "_buffers", _first_ordinary_buffer(model)[0]),
    lambda model: _set_first_element(model, "_parameters", _first_ordinary_parameter(model)[0], 0.0),
    lambda model: _set_first_element(model, "_buffers", _first_ordinary_buffer(model)[0], 1.0),
    lambda model: _rename_state_entry(model, "_parameters", _first_ordinary_parameter(model)[0], "forged_parameter"),
    lambda model: _rename_state_entry(model, "_buffers", _first_ordinary_buffer(model)[0], "forged_buffer"),
    _replace_buffer_with_forged_entry,
    _replace_parameter_with_forged_entry,
    _replace_buffer_same_shape,
    _replace_parameter_same_shape,
    _mutate_anchor,
    _mutate_valid_mask,
    _misalign_anchor_and_mask,
    lambda model: _delete_state_entry(model, "_buffers", "decoder.anchors"),
    lambda model: _delete_state_entry(model, "_buffers", "decoder.valid_mask"),
])
def test_frozen_real_state_helper_rejects_direct_mutations(mutation):
    model = _fresh_real_cpu_model()
    mutation(model)
    with pytest.raises(SmokeEvidenceError):
        _validate_and_hash_state(model, require_frozen_real_state=True)


def test_frozen_real_state_helper_accepts_only_frozen_identity():
    model = _fresh_real_cpu_model()
    rows = _validate_and_hash_state(model, require_frozen_real_state=True)
    assert sum(row["numel"] for row in rows[0]) == 20094584
    assert sum(row["numel"] for row in rows[0] if row["requires_grad"]) == 20094584
    assert len(rows[0]) == 326
    assert len(rows[1]) == 212
    assert rows[2] is True
    assert rows[3] is False
    assert rows[4] == []
    assert rows[5] == ["decoder.anchors"]
    assert rows[6] == "97ac7e46d699f7bef247eef213307b1e5d930cf6037c2c12e1d20cd31b40d1ff"
    assert rows[7] == "7fe2d718eea8e38e04fedc924e1b82883e07897a0635f7d1da949f8c5da6ab80"


def test_real_model_identity_accepts_expected_anchor_sentinel_without_forward():
    import numpy as np

    config = json.loads(V1_CONFIG.read_text(encoding="utf-8"))
    baseline = json.loads((ROOT / "configs/baseline/rtdetrv2_r18_visdrone_baseline_v1.json").read_text(encoding="utf-8"))
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    try:
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        model = build_r18_cpu_model(ROOT)
        model.eval()
        identity = build_model_identity(
            ROOT,
            config,
            model,
            VisDronePostProcessor(vendor_root=ROOT / "vendor/rtdetrv2_pytorch"),
            {},
            runtime_mode="real",
            device="cuda:0",
            context={"runtime_mode": "real"},
        )
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
    _validate_model_identity(identity, baseline, config, "cuda:0", {}, {"runtime_mode": "real"}, ROOT)
    assert identity["schema_version"] == 4
    assert identity["all_parameters_finite"] is True
    assert identity["all_buffers_finite"] is False
    assert identity["nonfinite_parameter_names"] == []
    assert identity["nonfinite_buffer_names"] == ["decoder.anchors"]
    assert identity["parameter_state_schema_sha256"] == "97ac7e46d699f7bef247eef213307b1e5d930cf6037c2c12e1d20cd31b40d1ff"
    assert identity["parameter_state_value_sha256"] == "7fe2d718eea8e38e04fedc924e1b82883e07897a0635f7d1da949f8c5da6ab80"
    assert torch.cuda.is_initialized() is False


def test_real_model_identity_rows_bind_complete_finiteness_and_raw_bytes():
    model = _fresh_real_cpu_model()
    config = json.loads(V1_CONFIG.read_text(encoding="utf-8"))
    identity = build_model_identity(
        ROOT,
        config,
        model,
        VisDronePostProcessor(vendor_root=ROOT / "vendor/rtdetrv2_pytorch"),
        {},
        runtime_mode="real",
        device="cuda:0",
        context={"runtime_mode": "real"},
    )
    rows = identity["parameter_inventory"] + identity["buffer_inventory"]
    assert all(set(row) == {"name", "shape", "dtype", "numel", "requires_grad", "kind", "finite", "finite_count", "nan_count", "posinf_count", "neginf_count", "raw_bytes_sha256"} for row in rows)
    assert all(isinstance(row["raw_bytes_sha256"], str) and len(row["raw_bytes_sha256"]) == 64 for row in rows)
    anchors = next(row for row in identity["buffer_inventory"] if row["name"] == "decoder.anchors")
    mask = next(row for row in identity["buffer_inventory"] if row["name"] == "decoder.valid_mask")
    assert anchors["finite"] is False
    assert anchors["finite_count"] == 32336
    assert anchors["nan_count"] == 0
    assert anchors["posinf_count"] == 1264
    assert anchors["neginf_count"] == 0
    assert anchors["raw_bytes_sha256"] == "fbe13addb417ce9eeff40c344f0e6b930dc25927fb459f74747efa853060d021"
    assert mask["finite"] is True
    assert mask["finite_count"] == 8400
    assert mask["raw_bytes_sha256"] == "e6fc2c623b214eb283567c918be21e9cebf2b23dd95c2c26f2076610d3e5f093"


@pytest.mark.parametrize("field", ["all_parameters_finite", "all_buffers_finite", "finite", "finite_count", "nan_count", "posinf_count", "neginf_count", "raw_bytes_sha256"])
def test_model_state_finiteness_mutations_rejected_after_repack(tmp_path, field):
    output = tmp_path / field
    run_synthetic_smoke(ROOT, output)
    value = _read_object(output / "model_identity.json")
    row = {
        "name": "forged.buffer",
        "shape": [1],
        "dtype": "torch.float32",
        "numel": 1,
        "requires_grad": False,
        "kind": "buffer",
        "finite": False,
        "finite_count": 0,
        "nan_count": 1,
        "posinf_count": 0,
        "neginf_count": 0,
        "raw_bytes_sha256": "0" * 64,
    }
    value["buffer_inventory"] = [row]
    value["buffer_count"] = 1
    value["all_buffers_finite"] = False
    value["nonfinite_buffer_names"] = ["forged.buffer"]
    if field in value:
        value[field] = False if field in {"all_parameters_finite", "all_buffers_finite", "finite"} else ("0" * 64 if field == "raw_bytes_sha256" else 1)
    _repack_model_identity(output, lambda current: current.update(value))
    with pytest.raises(SmokeEvidenceError):
        validate_entry_output(output)


def _seeded_real_model_identity():
    import numpy as np

    config = json.loads(V1_CONFIG.read_text(encoding="utf-8"))
    baseline = json.loads((ROOT / "configs/baseline/rtdetrv2_r18_visdrone_baseline_v1.json").read_text(encoding="utf-8"))
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    try:
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        model = build_r18_cpu_model(ROOT)
        model.eval()
        identity = build_model_identity(
            ROOT,
            config,
            model,
            VisDronePostProcessor(vendor_root=ROOT / "vendor/rtdetrv2_pytorch"),
            {},
            runtime_mode="real",
            device="cuda:0",
            context={"runtime_mode": "real"},
        )
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
    return identity, baseline, config


def _repack_real_identity_fixture(output: Path, identity: dict[str, object]) -> dict[str, object]:
    run_synthetic_smoke(ROOT, output)
    path = output / "model_identity.json"
    path.write_bytes(canonical_json_bytes(identity))
    _repack_entry(output)
    return _read_object(path)


def _sync_v4_identity_hashes(value: dict[str, object]) -> None:
    value["parameter_state_schema_sha256"], value["parameter_state_value_sha256"] = _state_hashes_from_inventory(value["parameter_inventory"], value["buffer_inventory"])


def _update_named_identity_row(value: dict[str, object], collection: str, name: str, updates: dict[str, object]) -> None:
    rows = value[collection]
    index = next(index for index, row in enumerate(rows) if row["name"] == name)
    rows[index] = {**rows[index], **updates}


@pytest.mark.parametrize("mutation", [
    lambda value: value.update({"all_parameters_finite": False}),
    lambda value: value["parameter_inventory"].__setitem__(0, {**value["parameter_inventory"][0], "finite": False, "finite_count": 0, "nan_count": 1}),
    lambda value: value["parameter_inventory"].__setitem__(0, {**value["parameter_inventory"][0], "finite": False, "finite_count": 0, "posinf_count": 1}),
    lambda value: _update_named_identity_row(value, "buffer_inventory", "decoder.anchors", {"finite": False, "finite_count": 0, "nan_count": 1}),
    lambda value: _update_named_identity_row(value, "buffer_inventory", "decoder.anchors", {"finite": True, "finite_count": 33600, "posinf_count": 0}),
    lambda value: _update_named_identity_row(value, "buffer_inventory", "decoder.anchors", {"finite_count": 32335, "posinf_count": 1265}),
    lambda value: _update_named_identity_row(value, "buffer_inventory", "decoder.anchors", {"raw_bytes_sha256": "0" * 64}),
    lambda value: _update_named_identity_row(value, "buffer_inventory", "decoder.anchors", {"posinf_count": 1263}),
    lambda value: _update_named_identity_row(value, "buffer_inventory", "decoder.anchors", {"posinf_count": 1265}),
    lambda value: _update_named_identity_row(value, "buffer_inventory", "decoder.anchors", {"nan_count": 1, "finite": False}),
    lambda value: _update_named_identity_row(value, "buffer_inventory", "decoder.anchors", {"neginf_count": 1, "finite": False}),
    lambda value: value["buffer_inventory"].__setitem__(0, {**value["buffer_inventory"][0], "name": "decoder.other"}),
    lambda value: value["buffer_inventory"].pop(),
])
def test_real_model_finiteness_mutations_rejected_after_full_repack(tmp_path, mutation):
    identity, baseline, config = _seeded_real_model_identity()
    value = copy.deepcopy(identity)
    mutation(value)
    _sync_v4_identity_hashes(value)
    output = tmp_path / "real-finiteness"
    observed = _repack_real_identity_fixture(output, value)
    with pytest.raises(SmokeEvidenceError):
        _validate_model_identity(observed, baseline, config, "cuda:0", {}, {"runtime_mode": "real"}, ROOT)


@pytest.mark.parametrize("mutation", [
    lambda value: _update_named_identity_row(value, "buffer_inventory", "decoder.anchors", {"raw_bytes_sha256": "0" * 64}),
    lambda value: _update_named_identity_row(value, "buffer_inventory", "decoder.anchors", {"finite_count": 32335, "posinf_count": 1265}),
    lambda value: _update_named_identity_row(value, "buffer_inventory", "decoder.anchors", {"name": "decoder.valid_mask"}),
])
def test_anchor_mask_repack_mutations_rejected_after_aggregate_resync(tmp_path, mutation):
    identity, baseline, config = _seeded_real_model_identity()
    value = copy.deepcopy(identity)
    mutation(value)
    _sync_v4_identity_hashes(value)
    output = tmp_path / "anchor-mask"
    observed = _repack_real_identity_fixture(output, value)
    with pytest.raises(SmokeEvidenceError):
        _validate_model_identity(observed, baseline, config, "cuda:0", {}, {"runtime_mode": "real"}, ROOT)


@pytest.mark.parametrize("artifact, mutation", [
    ("input_batch_audit.json", lambda value: value["preprocessing"].update({"padding": {"used": True, "mode": "zero"}})),
    ("cuda_runtime_identity.json", lambda value: value.update({"cpu_fallback": False})),
    ("model_identity.json", lambda value: value.update({"parameter_state_value_sha256": "0" * 64})),
    ("source_identity.json", lambda value: value["vendor"].update({"canonical_inventory_sha256": "0" * 64})),
    ("model_output_audit.json", lambda value: value["pred_logits"].update({"shape": [2, 300, 9]})),
    ("postprocess_audit.json", lambda value: value.update({"total_predictions": 599})),
    ("call_audit.json", lambda value: value.update({"model_forward_calls": 2})),
])
def test_scientific_semantic_mutation_rejected_after_inventory_repack(tmp_path, artifact, mutation):
    output = tmp_path / "entry"
    run_synthetic_smoke(ROOT, output)
    path = output / artifact
    value = _read_object(path)
    mutation(value)
    path.write_bytes(canonical_json_bytes(value))
    _repack_entry(output)
    with pytest.raises(SmokeEvidenceError):
        validate_entry_output(output)


@pytest.mark.parametrize("artifact", [
    "input_batch_audit.json",
    "cuda_runtime_identity.json",
    "model_identity.json",
    "source_identity.json",
])
@pytest.mark.parametrize("operation", ["missing", "extra"])
def test_scientific_exact_keys_reject_missing_and_extra_after_repack(tmp_path, artifact, operation):
    output = tmp_path / f"{artifact}-{operation}"
    run_synthetic_smoke(ROOT, output)
    path = output / artifact
    value = _read_object(path)
    if operation == "missing":
        value.pop(next(iter(value)))
    else:
        value["unexpected"] = 1
    path.write_bytes(canonical_json_bytes(value))
    _repack_entry(output)
    with pytest.raises(SmokeEvidenceError):
        validate_entry_output(output)


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


@pytest.mark.parametrize("config_path", [V2_CONFIG, V3_CONFIG])
def test_real_entry_v2_and_v3_share_canonical_config_binding(tmp_path, config_path):
    output = tmp_path / config_path.stem
    expected = _write_real_entry(output, config_path)
    result = validate_entry_output(output, expected_identity=expected)
    assert result["entry_success_accepted"] is True
    invocation = _read_object(output / "invocation.json")
    assert invocation["config_sha256"] == expected["config_canonical_sha256"]
    assert invocation["config_size_bytes"] == (output / "config.json").stat().st_size


def _run_real_entry_precuda_path(tmp_path, monkeypatch, config_path):
    smoke = importlib.import_module("sparse_rtdetr.baseline.smoke")
    monkeypatch.setattr(smoke, "resolve_runtime_paths", lambda *_args: None)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    data_root = tmp_path / "synthetic_train_core"
    data_root.mkdir()
    output = tmp_path / config_path.stem
    receipt, receipt_sha = _real_entry_handoff(config_path, output, data_root)
    selection_calls = []

    def selection_sentinel(_repo_root, *, runtime_mode="real", config_path=None):
        selection_calls.append((runtime_mode, config_path))
        assert runtime_mode == "real"
        if len(selection_calls) == 1:
            assert config_path == config_path_under_test
            return config["image_selection"]["records"]
        assert config_path is None
        assert (output / "config.json").is_file()
        assert (output / "invocation.json").is_file()
        assert not (output / "completion.json").exists()
        raise _PreCudaSentinel("pre-CUDA entry boundary")

    def forbidden_runtime_access(*_args, **_kwargs):
        raise AssertionError("pre-CUDA regression reached forbidden runtime access")

    config_path_under_test = config_path
    monkeypatch.setattr(smoke, "load_frozen_image_selection", selection_sentinel)
    monkeypatch.setattr(smoke, "resolve_runtime_image_path", forbidden_runtime_access)
    monkeypatch.setattr(smoke, "validate_real_smoke_environment", forbidden_runtime_access)
    monkeypatch.setattr(smoke, "build_r18_cpu_model", forbidden_runtime_access)

    imported = []
    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        imported.append(name)
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("pre-CUDA regression imported torch")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(_PreCudaSentinel, match="pre-CUDA entry boundary"):
        run_authorized_smoke(
            ROOT,
            data_root,
            output,
            handoff_receipt=receipt,
            handoff_receipt_sha256=receipt_sha,
            config_path=config_path,
        )

    assert selection_calls == [("real", config_path), ("real", None)]
    assert all(name != "torch" and not name.startswith("torch.") for name in imported)
    config_bytes = (output / "config.json").read_bytes()
    expected_bytes = canonical_json_bytes(config)
    assert config_bytes == expected_bytes
    assert config_bytes == canonical_config_bytes(config)
    assert len(config_bytes) == (output / "config.json").stat().st_size
    config_sha = hashlib.sha256(config_bytes).hexdigest()
    assert config_sha == hashlib.sha256(expected_bytes).hexdigest()
    invocation = _read_object(output / "invocation.json")
    assert invocation["smoke_id"] == config["smoke_id"]
    assert invocation["config_relative_path"] == config["runtime"]["config_relative_path"]
    assert invocation["config_size_bytes"] == len(expected_bytes)
    assert invocation["config_sha256"] == config_sha
    for field in (
        "nonce",
        "launcher_pid",
        "child_pid",
        "child_ppid",
        "child_argv_sha256",
    ):
        assert invocation[field] == receipt[field]
    assert invocation["handoff_receipt_sha256"] == receipt_sha
    assert invocation["handoff_receipt_relative_path"] == "handoff_receipt.json"
    assert receipt["config_size_bytes"] == config_path.stat().st_size
    assert receipt["config_size_bytes"] != invocation["config_size_bytes"]
    completion = _read_object(output / "completion.json")
    assert completion["status"] == "FAILED"
    assert _read_object(output / "error.json")["exception_type"] == "_PreCudaSentinel"
    assert (output / "partial_inventory.json").is_file()
    return config, output, invocation


def test_real_v4_entry_precuda_config_invocation_path(tmp_path, monkeypatch):
    config, output, invocation = _run_real_entry_precuda_path(tmp_path, monkeypatch, V4_CONFIG)
    assert config["smoke_id"] == SMOKE_V4_ID
    assert invocation["config_relative_path"] == SMOKE_V4_CONFIG_RELATIVE
    assert invocation["config_size_bytes"] == 2922
    assert invocation["config_sha256"] == "9a0d8a304b35b955bec24ab13940002a2660b2c9653c567526bd1d0addc47d5e"
    assert (output / "config.json").stat().st_size == 2922


def test_real_v5_entry_precuda_config_invocation_path(tmp_path, monkeypatch):
    config, output, invocation = _run_real_entry_precuda_path(tmp_path, monkeypatch, V5_CONFIG)
    assert config["smoke_id"] == SMOKE_V5_ID
    assert invocation["config_relative_path"] == SMOKE_V5_CONFIG_RELATIVE
    assert invocation["config_size_bytes"] == 2922
    assert invocation["config_sha256"] == "ed39e40602b716fcc7e94bb521dbcf66ce6f57436b9cbc99ed7ed778462f672b"
    assert (output / "config.json").stat().st_size == 2922


@pytest.mark.parametrize(
    "config_path, expected_id, expected_relative",
    [
        (V2_CONFIG, SMOKE_V2_ID, "configs/baseline/rtdetrv2_r18_visdrone_smoke_v2.json"),
        (V3_CONFIG, SMOKE_V3_ID, SMOKE_V3_CONFIG_RELATIVE),
        (V4_CONFIG, SMOKE_V4_ID, SMOKE_V4_CONFIG_RELATIVE),
        (V5_CONFIG, SMOKE_V5_ID, SMOKE_V5_CONFIG_RELATIVE),
    ],
)
def test_real_versioned_entry_precuda_path_v2_v3_v4(
    tmp_path, monkeypatch, config_path, expected_id, expected_relative
):
    _config, _output, invocation = _run_real_entry_precuda_path(tmp_path, monkeypatch, config_path)
    assert invocation["smoke_id"] == expected_id
    assert invocation["config_relative_path"] == expected_relative


def test_real_entry_canonical_helper_has_no_unresolved_global():
    closure = inspect.getclosurevars(run_authorized_smoke)
    assert "canonical_config_bytes" not in closure.unbound
    assert callable(closure.globals["canonical_json_bytes"])


def test_process_failure_keeps_child_marker_and_launcher_exit_semantics(tmp_path, monkeypatch):
    _patch_synthetic_selection(monkeypatch)
    monkeypatch.setenv("P3_RTDETR_BASELINE_SMOKE_AUTHORIZED", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    child = tmp_path / "return_one_child.py"
    child.write_text("raise SystemExit(1)\n", encoding="utf-8")
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
        nonce="c" * 32,
        allow_noncanonical_child=True,
    )
    assert result == 2
    assert (process / "process_exit_code.txt").read_bytes() == b"1\n"
    completion = _read_object(process / "process_completion.json")
    assert completion["child_returncode"] == 1
    assert completion["launcher_exit_code"] == 2
    assert completion["status"] == "FAILED_CHILD"


@pytest.mark.parametrize("mutation", [
    "delete_relative",
    "v2_relative",
    "arbitrary_relative",
    "delete_size",
    "bool_size",
    "increment_size",
    "delete_sha",
    "source_file_sha",
    "other_sha",
    "v2_smoke_id",
    "config_smoke_id_v2",
    "config_content",
    "expected_canonical",
    "expected_smoke_id",
    "expected_relative",
])
def test_v3_real_entry_config_binding_mutations_fail_closed(tmp_path, mutation):
    output = tmp_path / "entry"
    expected = _write_real_entry(output, V3_CONFIG)
    invocation_path = output / "invocation.json"
    invocation = _read_object(invocation_path)
    if mutation == "delete_relative":
        invocation.pop("config_relative_path")
    elif mutation == "v2_relative":
        invocation["config_relative_path"] = V2_CONFIG.relative_to(ROOT).as_posix()
    elif mutation == "arbitrary_relative":
        invocation["config_relative_path"] = "configs/other.json"
    elif mutation == "delete_size":
        invocation.pop("config_size_bytes")
    elif mutation == "bool_size":
        invocation["config_size_bytes"] = True
    elif mutation == "increment_size":
        invocation["config_size_bytes"] += 1
    elif mutation == "delete_sha":
        invocation.pop("config_sha256")
    elif mutation == "source_file_sha":
        invocation["config_sha256"] = hashlib.sha256(V3_CONFIG.read_bytes()).hexdigest()
    elif mutation == "other_sha":
        invocation["config_sha256"] = "0" * 64
    elif mutation == "v2_smoke_id":
        invocation["smoke_id"] = SMOKE_V2_ID
    elif mutation in {"config_smoke_id_v2", "config_content"}:
        config = json.loads((output / "config.json").read_text(encoding="utf-8"))
        if mutation == "config_smoke_id_v2":
            config["smoke_id"] = SMOKE_V2_ID
        else:
            config["runtime"]["config_relative_path"] = "configs/changed.json"
        (output / "config.json").write_bytes(canonical_json_bytes(config))
        config_sha = hashlib.sha256((output / "config.json").read_bytes()).hexdigest()
        invocation["config_sha256"] = config_sha
        completion = _read_object(output / "completion.json")
        completion["config_sha256"] = config_sha
        (output / "completion.json").write_bytes(canonical_json_bytes(completion))
    elif mutation == "expected_canonical":
        expected["config_canonical_sha256"] = "0" * 64
    elif mutation == "expected_smoke_id":
        expected["smoke_id"] = SMOKE_V2_ID
    elif mutation == "expected_relative":
        expected["config_relative_path"] = V2_CONFIG.relative_to(ROOT).as_posix()
    invocation_path.write_bytes(canonical_json_bytes(invocation))
    _repack_entry(output)
    with pytest.raises(SmokeEvidenceError):
        validate_entry_output(output, expected_identity=expected)


def test_real_entry_requires_complete_expected_config_identity(tmp_path):
    output = tmp_path / "entry"
    expected = _write_real_entry(output, V3_CONFIG)
    expected.pop("config_canonical_sha256")
    with pytest.raises(SmokeEvidenceError):
        validate_entry_output(output, expected_identity=expected)


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
    result = launch_smoke([str(PYTHON), str(child)], output, process, data_root=_runtime_data_root(tmp_path), child_python=PYTHON, repo_root=ROOT, env=_base_env(), nonce="a" * 32, allow_noncanonical_child=True)
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
    result = launch_smoke([str(PYTHON), str(child)], output, process, data_root=_runtime_data_root(tmp_path), child_python=PYTHON, repo_root=ROOT, env=_base_env(), nonce="9" * 32, allow_noncanonical_child=True)
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
        allow_noncanonical_child=True,
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
    result = launch_smoke([str(PYTHON), str(child)], output, process, data_root=_runtime_data_root(tmp_path), child_python=PYTHON, repo_root=ROOT, env=_base_env(), nonce="1" * 32, allow_noncanonical_child=True)
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
    result = launch_smoke([str(PYTHON), str(child)], output, process, data_root=_runtime_data_root(tmp_path), child_python=PYTHON, repo_root=ROOT, env=_base_env(), nonce="2" * 32, allow_noncanonical_child=True)
    assert result == 2
    assert _read_object(process / "handoff_receipt.json")["consumed"] is True
    assert not output.exists()


def _receipt_test_value() -> dict[str, object]:
    return {"schema_version": 1, "consumed": True, "nonce": "a" * 32}


def test_exclusive_receipt_writer_uses_final_path_and_readback(tmp_path, monkeypatch):
    launcher = importlib.import_module("sparse_rtdetr.baseline.smoke_launcher")
    process = tmp_path / "process"
    process.mkdir()
    receipt_path = process / "handoff_receipt.json"
    observed = {}
    original_open = launcher.os.open

    def capture_open(path, flags, mode=0o777, *args, **kwargs):
        if Path(path) == receipt_path and flags & os.O_EXCL:
            observed.update({"flags": flags, "mode": mode})
        return original_open(path, flags, mode, *args, **kwargs)

    monkeypatch.setattr(launcher.os, "open", capture_open)
    value = _receipt_test_value()
    receipt_sha = launcher._create_exclusive_durable_handoff_receipt(receipt_path, value)
    payload = canonical_json_bytes(value)
    assert receipt_sha == hashlib.sha256(payload).hexdigest()
    assert receipt_path.read_bytes() == payload
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o600
    assert receipt_path.stat().st_nlink == 1
    assert observed["flags"] & os.O_WRONLY
    assert observed["flags"] & os.O_CREAT
    assert observed["flags"] & os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        assert observed["flags"] & os.O_NOFOLLOW
    assert observed["mode"] == 0o600
    with pytest.raises(SmokeLauncherError, match="already been consumed"):
        launcher._create_exclusive_durable_handoff_receipt(receipt_path, value)
    assert receipt_path.read_bytes() == payload


def test_exclusive_receipt_writer_handles_partial_write_and_eintr(tmp_path, monkeypatch):
    launcher = importlib.import_module("sparse_rtdetr.baseline.smoke_launcher")
    process = tmp_path / "process"
    process.mkdir()
    receipt_path = process / "handoff_receipt.json"
    original_open = launcher.os.open
    original_write = launcher.os.write
    state = {"receipt_fd": None, "writes": 0}

    def capture_open(path, flags, mode=0o777, *args, **kwargs):
        fd = original_open(path, flags, mode, *args, **kwargs)
        if Path(path) == receipt_path and flags & os.O_EXCL:
            state["receipt_fd"] = fd
        return fd

    def partial_write(fd, payload):
        if fd != state["receipt_fd"]:
            return original_write(fd, payload)
        state["writes"] += 1
        if state["writes"] == 1:
            raise InterruptedError()
        if state["writes"] == 2:
            partial = payload[:max(1, len(payload) // 2)]
            original_write(fd, partial)
            return len(partial)
        return original_write(fd, payload)

    monkeypatch.setattr(launcher.os, "open", capture_open)
    monkeypatch.setattr(launcher.os, "write", partial_write)
    value = _receipt_test_value()
    launcher._create_exclusive_durable_handoff_receipt(receipt_path, value)
    assert receipt_path.read_bytes() == canonical_json_bytes(value)
    assert state["writes"] >= 3


@pytest.mark.parametrize("fault", ["open", "zero", "write", "file_fsync", "directory_open", "directory_fsync", "readback"])
def test_exclusive_receipt_claim_failures_are_fail_closed(tmp_path, monkeypatch, fault):
    launcher = importlib.import_module("sparse_rtdetr.baseline.smoke_launcher")
    process = tmp_path / "process"
    process.mkdir()
    receipt_path = process / "handoff_receipt.json"
    original_open = launcher.os.open
    original_write = launcher.os.write
    original_fsync = launcher.os.fsync
    original_read = launcher.os.read
    state = {"receipt_fd": None, "directory_fd": None, "readback_fd": None}

    def capture_open(path, flags, mode=0o777, *args, **kwargs):
        if fault == "open" and Path(path) == receipt_path and flags & os.O_EXCL:
            raise PermissionError("injected claim open failure")
        if fault == "directory_open" and Path(path) == process and not flags & os.O_WRONLY:
            raise PermissionError("injected directory open failure")
        fd = original_open(path, flags, mode, *args, **kwargs)
        if Path(path) == receipt_path and flags & os.O_EXCL:
            state["receipt_fd"] = fd
        elif Path(path) == receipt_path and not flags & os.O_WRONLY:
            state["readback_fd"] = fd
        elif Path(path) == process:
            state["directory_fd"] = fd
        return fd

    def injected_write(fd, payload):
        if fd != state["receipt_fd"]:
            return original_write(fd, payload)
        if fault == "zero":
            return 0
        if fault == "write":
            raise OSError("injected write failure")
        return original_write(fd, payload)

    def injected_fsync(fd):
        if fault in {"file_fsync", "directory_fsync"} and (fd == state["receipt_fd"] or fd == state["directory_fd"]):
            raise OSError("injected fsync failure")
        return original_fsync(fd)

    def injected_read(fd, size):
        if fault == "readback" and fd == state["readback_fd"]:
            return b"x"
        return original_read(fd, size)

    monkeypatch.setattr(launcher.os, "open", capture_open)
    monkeypatch.setattr(launcher.os, "write", injected_write)
    monkeypatch.setattr(launcher.os, "fsync", injected_fsync)
    monkeypatch.setattr(launcher.os, "read", injected_read)
    value = _receipt_test_value()
    with pytest.raises(SmokeLauncherError):
        launcher._create_exclusive_durable_handoff_receipt(receipt_path, value)
    if fault == "open":
        assert not receipt_path.exists()
    else:
        assert receipt_path.exists()
        with pytest.raises(SmokeLauncherError, match="already been consumed"):
            launcher._create_exclusive_durable_handoff_receipt(receipt_path, value)


@pytest.mark.parametrize("kind", ["file", "directory", "symlink", "hardlink", "fifo", "parent_symlink", "escape"])
def test_preexisting_receipt_objects_are_never_repaired(tmp_path, kind):
    launcher = importlib.import_module("sparse_rtdetr.baseline.smoke_launcher")
    process = tmp_path / "process"
    process.mkdir()
    receipt_path = process / "handoff_receipt.json"
    target = tmp_path / "target"
    if kind == "file":
        receipt_path.write_bytes(b"PREEXISTING\n")
    elif kind == "directory":
        receipt_path.mkdir()
    elif kind == "symlink":
        target.write_bytes(b"TARGET\n")
        receipt_path.symlink_to(target)
    elif kind == "hardlink":
        target.write_bytes(b"HARDLINK\n")
        os.link(target, receipt_path)
    elif kind == "fifo":
        os.mkfifo(receipt_path)
    elif kind == "parent_symlink":
        alias = tmp_path / "alias"
        alias.symlink_to(process, target_is_directory=True)
        receipt_path = alias / "handoff_receipt.json"
    else:
        receipt_path = process / ".." / "escape" / "handoff_receipt.json"
    with pytest.raises(SmokeLauncherError):
        launcher._create_exclusive_durable_handoff_receipt(receipt_path, _receipt_test_value())
    if kind == "file":
        assert receipt_path.read_bytes() == b"PREEXISTING\n"
    elif kind == "symlink":
        assert receipt_path.is_symlink() and target.read_bytes() == b"TARGET\n"
    elif kind == "hardlink":
        assert receipt_path.stat().st_nlink == 2 and target.read_bytes() == b"HARDLINK\n"
    elif kind == "fifo":
        assert stat.S_ISFIFO(receipt_path.lstat().st_mode)
    elif kind == "directory":
        assert receipt_path.is_dir() and not any(receipt_path.iterdir())
    elif kind == "parent_symlink":
        assert not (process / "handoff_receipt.json").exists()


@pytest.mark.parametrize("child_count", [2, 4])
def test_exclusive_receipt_claim_is_cross_process_exactly_once(tmp_path, monkeypatch, child_count):
    launcher = importlib.import_module("sparse_rtdetr.baseline.smoke_launcher")
    monkeypatch.setenv("P3_RTDETR_BASELINE_SMOKE_AUTHORIZED", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    for round_index in range(2):
        root = tmp_path / f"competition_{child_count}_{round_index}"
        root.mkdir()
        barrier = root / "barrier"
        entries = root / "entries"
        data = root / "data"
        barrier.mkdir()
        entries.mkdir()
        data.mkdir()
        sitecustomize = root / "sitecustomize.py"
        _write_module_child_sitecustomize(sitecustomize)
        env = _module_child_env(root, sitecustomize)
        env.update({
            "P3_SMOKE_TEST_EXCLUSIVE_STUB": "1",
            "P3_SMOKE_EXCLUSIVE_BARRIER_DIR": str(barrier),
            "P3_SMOKE_EXCLUSIVE_BARRIER_COUNT": str(child_count),
            "P3_SMOKE_EXCLUSIVE_ENTRY_DIR": str(entries),
        })
        output = root / "output"
        process = root / "process"
        statuses = []
        pids = []
        snapshots = {}

        def run_children(child_argv, child_env, console):
            prepared = process / "handoff_prepared.json"
            before = prepared.read_bytes()
            snapshots["before"] = hashlib.sha256(before).hexdigest()
            children = [subprocess.Popen(child_argv, cwd=ROOT, env=child_env) for _ in range(child_count)]
            pids.extend(child.pid for child in children)
            child_identity = {"pid": children[0].pid, "sid": os.getsid(children[0].pid), "pgid": os.getpgid(children[0].pid), "parent_pid": os.getpid()}
            statuses.extend(child.wait() for child in children)
            snapshots["after"] = hashlib.sha256(prepared.read_bytes()).hexdigest()
            return max(statuses), [], False, child_identity

        monkeypatch.setattr(launcher, "_run_child", run_children)
        result = launcher.launch_smoke(
            launcher._canonical_child_argv(PYTHON, ROOT),
            output,
            process,
            data_root=data,
            child_python=PYTHON,
            repo_root=ROOT,
            env=env,
            nonce=("a" * 31) + str(round_index),
        )
        assert result == 2
        assert statuses.count(0) == 1
        assert statuses.count(2) == child_count - 1
        entry_files = sorted(entries.glob("entry.*"))
        assert len(entry_files) == 1
        receipt = _read_object(process / "handoff_receipt.json")
        assert receipt["child_pid"] == pids[statuses.index(0)]
        assert receipt["consumed"] is True
        assert snapshots["before"] == snapshots["after"]
        assert _read_object(process / "handoff_prepared.json")["consumed"] is False
        assert not list(process.glob(".handoff_receipt.*.tmp"))


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
        allow_noncanonical_child=True,
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
        allow_noncanonical_child=True,
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
        allow_noncanonical_child=True,
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
        allow_noncanonical_child=True,
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
        allow_noncanonical_child=True,
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
        allow_noncanonical_child=True,
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
        allow_noncanonical_child=True,
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
        allow_noncanonical_child=True,
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
        "env=dict(os.environ), nonce='d'*32, allow_noncanonical_child=True))\n"
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
        allow_noncanonical_child=True,
    )
    completion = _read_object(process / "process_completion.json")
    inventory = _read_object(process / "process_inventory.json")
    assert completion["process_inventory"]["sha256"] == __import__("hashlib").sha256(
        (process / "process_inventory.json").read_bytes()
    ).hexdigest()
    assert "process_partial_inventory.json" not in {
        row["relative_path"] for row in inventory["artifacts"]
    }
