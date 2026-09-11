from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import pytest

from sparse_rtdetr.baseline import training_v2a_contract as contract


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / contract.V2A_CONFIG_RELATIVE_PATH
AUTHORITY_ROOT = ROOT / contract.AUTHORITY_ROOT_RELATIVE_PATH
CHECKER_PATH = ROOT / "tools" / "repository_contract_check.py"
CHECKER_SPEC = importlib.util.spec_from_file_location("repository_contract_check_for_v2a", CHECKER_PATH)
CHECKER = importlib.util.module_from_spec(CHECKER_SPEC)
assert CHECKER_SPEC.loader is not None
CHECKER_SPEC.loader.exec_module(CHECKER)
TRAINING_CONTRACT_PATH = ROOT / "src" / "sparse_rtdetr" / "baseline" / "training_contract.py"
TRAINING_CONTRACT_SPEC = importlib.util.spec_from_file_location("training_contract_for_v2a", TRAINING_CONTRACT_PATH)
V1_CONTRACT = importlib.util.module_from_spec(TRAINING_CONTRACT_SPEC)
assert TRAINING_CONTRACT_SPEC.loader is not None
TRAINING_CONTRACT_SPEC.loader.exec_module(V1_CONTRACT)


def _config() -> dict:
    return json.loads(CONFIG.read_bytes())


def _set_pointer(value: dict, pointer: str, replacement: object) -> None:
    current: object = value
    tokens = pointer.strip("/").split("/")
    for token in tokens[:-1]:
        token = token.replace("~1", "/").replace("~0", "~")
        current = current[int(token)] if type(current) is list else current[token]
    token = tokens[-1].replace("~1", "/").replace("~0", "~")
    if type(current) is list:
        current[int(token)] = replacement
    else:
        current[token] = replacement


def _different(value: object) -> object:
    if value is None:
        return "not-null"
    if type(value) is bool:
        return not value
    if type(value) is int:
        return value + 1
    if type(value) is float:
        return value + 0.125
    if type(value) is str:
        return value + "_drift"
    raise AssertionError(type(value))


def _portable_root(tmp_path: Path, *, include_authority: bool) -> Path:
    root = tmp_path / "portable-repo"
    root.mkdir(parents=True)
    for relative in (contract.V2A_CONFIG_RELATIVE_PATH, contract.PARENT_TRAINING_CONFIG_RELATIVE_PATH):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    for relative in (
        contract.AUTHORITY_VENDOR_CONFIG_RELATIVE_PATH,
        contract.AUTHORITY_VENDOR_SOURCE_RELATIVE_PATH,
        contract.AUTHORITY_VENDOR_OPTIMIZER_RELATIVE_PATH,
    ):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    if include_authority:
        shutil.copytree(AUTHORITY_ROOT, root / contract.AUTHORITY_ROOT_RELATIVE_PATH)
    return root


def test_config_raw_and_canonical_identity_are_frozen():
    raw = CONFIG.read_bytes()
    value = _config()
    canonical = contract.canonical_training_v2a_contract_bytes(value)
    assert len(raw) == contract.V2A_CONFIG_RAW_SIZE_BYTES
    assert hashlib.sha256(raw).hexdigest() == contract.V2A_CONFIG_RAW_SHA256
    assert len(canonical) == contract.V2A_CONFIG_CANONICAL_SIZE_BYTES
    assert hashlib.sha256(canonical).hexdigest() == contract.V2A_CONFIG_CANONICAL_SHA256
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")
    assert b"\r" not in raw
    assert b"\x00" not in raw
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert not canonical.endswith(b"\n")


def test_detached_validation_is_closed_and_detached():
    value = _config()
    validated = contract.validate_training_v2a_contract(value)
    assert validated == value
    assert validated is not value
    assert validated["source_bindings"] is not value["source_bindings"]
    validated["readiness"]["training_ready"] = True
    assert value["readiness"]["training_ready"] is False
    assert set(value) == {
        "schema_version",
        "contract_id",
        "baseline_id",
        "stage",
        "parent_baseline_v1",
        "source_bindings",
        "model",
        "data_roles",
        "initialization",
        "topology",
        "schedule",
        "optimizer",
        "learning_rate",
        "amp",
        "ema",
        "augmentation",
        "evaluation_and_selection",
        "checkpoint_policy",
        "acceptance",
        "environment_identity",
        "hardware_launch_gate",
        "runtime_closure",
        "readiness",
    }


def test_registry_covers_every_scalar_without_overlap():
    leaves = set(contract._FROZEN_LEAVES)
    numeric = set(contract._NUMERIC_CONSTRAINTS)
    path_sha = set(contract._PATH_SHA_LEAVES)
    semantic = set(contract._SEMANTIC_LEAVES)
    assert leaves == numeric | path_sha | semantic
    assert not numeric & path_sha
    assert not numeric & semantic
    assert not path_sha & semantic
    assert len(contract._RELATION_RULES) == 10
    contract._validate_owner_registry()


def test_every_frozen_leaf_mutation_is_rejected():
    value = _config()
    for pointer, original in contract._FROZEN_LEAVES.items():
        candidate = copy.deepcopy(value)
        _set_pointer(candidate, pointer, _different(original))
        with pytest.raises(contract.TrainingV2AContractError):
            contract.validate_training_v2a_contract(candidate)


@pytest.mark.parametrize(
    ("pointer", "replacement"),
    [
        ("/model/num_classes", True),
        ("/optimizer/default_lr", 0.0),
        ("/parent_baseline_v1/development_ap", float("nan")),
        ("/parent_baseline_v1/development_ap", float("inf")),
        ("/topology/train_micro_batch", "16"),
        ("/source_bindings/pretrained_authority/weight/sha256", "A" * 64),
        ("/source_bindings/data_roles/train_core/annotation", "../train.json"),
    ],
)
def test_type_range_nonfinite_and_path_mutations_fail_closed(pointer, replacement):
    candidate = _config()
    _set_pointer(candidate, pointer, replacement)
    with pytest.raises(contract.TrainingV2AContractError):
        contract.validate_training_v2a_contract(candidate)


def test_v2a_frozen_decisions_are_explicit():
    value = _config()
    assert value["baseline_id"] == contract.V2A_BASELINE_ID
    assert value["contract_id"] == contract.V2A_CONTRACT_ID
    assert value["parent_baseline_v1"]["development_ap"] == 22.9469
    assert value["parent_baseline_v1"]["development_ap_role"] == "comparison_only_not_acceptance_threshold"
    assert value["source_bindings"]["pretrained_authority"]["weight"]["sha256"] == contract.AUTHORITY_WEIGHT_SHA256
    assert value["source_bindings"]["pretrained_authority"]["manifest"]["sha256"] == contract.AUTHORITY_MANIFEST_SHA256
    assert value["model"]["input_size"] == [640, 640]
    assert value["topology"]["train_micro_batch"] == 16
    assert value["topology"]["development_batch"] == 32
    assert value["topology"]["gradient_accumulation_steps"] == 1
    assert value["data_roles"] == {
        "train": "train_core",
        "development": "development",
        "confirmatory": "sealed_and_forbidden",
        "test": "forbidden",
    }


def test_bf16_has_no_gradscaler_parameters():
    amp = _config()["amp"]
    assert amp["enabled"] is True
    assert amp["autocast_dtype"] == "bfloat16"
    assert amp["grad_scaler_enabled"] is False
    assert amp["scaler_policy"] == "disabled_absent"
    assert not {"init_scale", "growth_factor", "backoff_factor", "growth_interval", "scaler_type"} & set(amp)
    assert all(amp[field] == 0 for field in ("nonfinite_loss_allowed", "nonfinite_gradient_allowed", "optimizer_skipped_steps_allowed", "overflow_events_allowed"))


def test_optimizer_policy_is_vendor_bound_and_identity_auditable():
    value = _config()
    optimizer = value["optimizer"]
    assert [group["name"] for group in optimizer["parameter_groups"]] == ["norm_or_bn", "default"]
    assert optimizer["parameter_groups"][0]["include_substrings"] == ["norm", "bn"]
    assert optimizer["parameter_groups"][0]["weight_decay"] == 0.0
    assert optimizer["parameter_groups"][1]["parameter_scope"] == "remaining_trainable"
    assert optimizer["parameter_groups"][1]["include_substrings"] == []
    assert optimizer["parameter_groups"][1]["exclude_substrings"] == []
    assert optimizer["backbone_non_norm_lr_override_forbidden"] is True
    audit = contract.audit_optimizer_parameter_groups(
        ["backbone.conv1.weight", "backbone.bn1.weight", "encoder.norm.weight", "decoder.class_embed.weight"]
    )
    assert audit["mutually_exclusive"] is True
    assert audit["covered_all_parameters"] is True
    assert audit["group_counts"] == {"norm_or_bn": 2, "default": 2}
    assert contract.resolve_optimizer_parameter_group("backbone.conv1.weight") == "default"
    assert contract.resolve_optimizer_parameter_group("backbone.bn1.weight") == "norm_or_bn"
    with pytest.raises(contract.TrainingV2AContractError):
        contract.audit_optimizer_parameter_groups(["x", "x"])


def test_runtime_closure_and_hardware_gate_remain_fail_closed():
    value = _config()
    for requirement in value["runtime_closure"].values():
        assert requirement == {"required": True, "implemented": False, "independently_certified": False}
    assert value["environment_identity"]["graphics_clock_max_mhz"] == 1500
    assert value["hardware_launch_gate"]["graphics_clock_upper_limit_mhz"] == 1500
    assert value["environment_identity"]["exclusive_host_observation"] == "required_before_training"
    assert value["hardware_launch_gate"]["probe_executed_in_t7b"] is False
    assert value["readiness"]["training_implementation_ready"] is False
    assert value["readiness"]["training_ready"] is False


def test_v1_and_t7a_identities_are_not_changed():
    parent_raw = (ROOT / contract.PARENT_TRAINING_CONFIG_RELATIVE_PATH).read_bytes()
    authority_weight = ROOT / contract.AUTHORITY_WEIGHT_RELATIVE_PATH
    authority_manifest = ROOT / contract.AUTHORITY_MANIFEST_RELATIVE_PATH
    assert len(parent_raw) == contract.PARENT_TRAINING_CONFIG_RAW_SIZE_BYTES
    assert hashlib.sha256(parent_raw).hexdigest() == contract.PARENT_TRAINING_CONFIG_RAW_SHA256
    assert authority_weight.stat().st_size == contract.AUTHORITY_WEIGHT_SIZE_BYTES
    assert hashlib.sha256(authority_weight.read_bytes()).hexdigest() == contract.AUTHORITY_WEIGHT_SHA256
    assert authority_manifest.stat().st_size == contract.AUTHORITY_MANIFEST_SIZE_BYTES
    assert hashlib.sha256(authority_manifest.read_bytes()).hexdigest() == contract.AUTHORITY_MANIFEST_SHA256


def test_portable_load_does_not_require_t7a_authority(tmp_path):
    portable = _portable_root(tmp_path, include_authority=False)
    loaded = contract.load_training_v2a_contract(portable)
    assert loaded == contract.validate_training_v2a_contract(_config())
    with pytest.raises(contract.TrainingV2AContractError):
        contract.training_v2a_contract_binding(portable)


def test_current_authority_binding_is_detached_and_complete():
    binding = contract.training_v2a_contract_binding(ROOT)
    assert binding["contract_id"] == contract.V2A_CONTRACT_ID
    assert binding["authority_binding"]["before_after_identity_pass"] is True
    assert binding["authority_binding"]["weight"]["size_bytes"] == contract.AUTHORITY_WEIGHT_SIZE_BYTES
    assert binding["authority_binding"]["weight"]["sha256"] == contract.AUTHORITY_WEIGHT_SHA256
    assert binding["authority_binding"]["manifest_load_validation"]["state_key_count"] == 115
    assert binding["vendor_r18_binding"]["effective_optimizer"]["type"] == "AdamW"
    snapshot = copy.deepcopy(binding)
    binding["contract"]["readiness"]["training_ready"] = True
    assert contract.training_v2a_contract_binding(ROOT) == snapshot


@pytest.mark.parametrize(
    ("relative", "kind"),
    [
        (contract.AUTHORITY_WEIGHT_RELATIVE_PATH, "symlink"),
        (contract.AUTHORITY_WEIGHT_RELATIVE_PATH, "hardlink"),
        (contract.AUTHORITY_WEIGHT_RELATIVE_PATH, "directory"),
        (contract.AUTHORITY_WEIGHT_RELATIVE_PATH, "fifo"),
        (contract.AUTHORITY_MANIFEST_RELATIVE_PATH, "symlink"),
        (contract.AUTHORITY_MANIFEST_RELATIVE_PATH, "hardlink"),
        (contract.AUTHORITY_MANIFEST_RELATIVE_PATH, "directory"),
        (contract.AUTHORITY_MANIFEST_RELATIVE_PATH, "fifo"),
    ],
)
def test_authority_object_matrix_fails_closed(tmp_path, relative, kind):
    portable = _portable_root(tmp_path, include_authority=True)
    path = portable / relative
    if kind == "symlink":
        path.unlink()
        path.symlink_to(ROOT / relative)
    elif kind == "hardlink":
        target = tmp_path / "outside-hardlink-target"
        target.write_bytes(path.read_bytes())
        path.unlink()
        path.hardlink_to(target)
    elif kind == "directory":
        path.unlink()
        path.mkdir()
    elif kind == "fifo":
        path.unlink()
        os.mkfifo(path)
    else:
        raise AssertionError(kind)
    with pytest.raises((contract.TrainingV2AContractError, OSError)):
        contract.training_v2a_contract_binding(portable)


def test_import_has_no_framework_gpu_or_filesystem_side_effect(tmp_path):
    probe = (
        "import sys; before=set(sys.modules); "
        "import sparse_rtdetr.baseline.training_v2a_contract; "
        "print({'torch': 'torch' in sys.modules, 'numpy': 'numpy' in sys.modules, "
        "'unchanged': set(sys.modules) >= before})"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", probe],
        cwd=ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": "src"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert "'torch': False" in result.stdout
    assert "'numpy': False" in result.stdout
    assert not list(tmp_path.iterdir())


def _clean_candidate_archive(tmp_path: Path) -> Path:
    archive_root = tmp_path / "clean-candidate-archive"
    archive_root.mkdir()
    archive_file = tmp_path / "clean-candidate.tar"
    subprocess.run(["git", "archive", "HEAD", "-o", str(archive_file)], cwd=ROOT, check=True)
    with tarfile.open(archive_file) as handle:
        handle.extractall(archive_root)
    for relative in (
        "configs/baseline/rtdetrv2_r18_visdrone_baseline_v2a.json",
        "docs/contracts/RTDETR_BASELINE_V2A_FORMAL_TRAINING_V1.md",
        "src/sparse_rtdetr/baseline/training_v2a_contract.py",
        "tests/test_rtdetr_baseline_training_v2a_contract.py",
        "docs/contracts/RTDETR_BASELINE_V2A_RUNTIME_INTEGRATION_T7C.md",
        "src/sparse_rtdetr/baseline/training_v2a_runtime.py",
        "tests/test_rtdetr_baseline_training_v2a_runtime.py",
        "tools/repository_contract_check.py",
    ):
        destination = archive_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    return archive_root


def test_repository_checker_does_not_own_t6b_runtime_target_absence():
    source = CHECKER_PATH.read_text(encoding="utf-8")
    assert "T6B production target must remain absent" not in source


def test_repository_checker_accepts_checkout_with_completed_v1_targets():
    for relative in (
        "artifacts/training/rtdetrv2_r18_visdrone_training_t6b_v1",
        "artifacts/process_evidence/rtdetrv2_r18_visdrone_training_t6b_v1",
        "artifacts/outer_launch_evidence/rtdetrv2_r18_visdrone_training_t6b_v1",
    ):
        assert (ROOT / relative).exists()
    assert CHECKER.check_repository(ROOT)


def test_repository_checker_accepts_clean_candidate_archive_without_v1_targets(tmp_path, capsys):
    archive_root = _clean_candidate_archive(tmp_path)
    assert not (archive_root / "artifacts").exists()
    for relative in (
        "artifacts/training/rtdetrv2_r18_visdrone_training_t6b_v1",
        "artifacts/process_evidence/rtdetrv2_r18_visdrone_training_t6b_v1",
        "artifacts/outer_launch_evidence/rtdetrv2_r18_visdrone_training_t6b_v1",
    ):
        assert not (archive_root / relative).exists()
    v1_loaded = V1_CONTRACT.load_training_contract(archive_root)
    assert v1_loaded["training_contract_id"] == "rtdetrv2_r18_visdrone_baseline_training_v1"
    with pytest.raises(V1_CONTRACT.TrainingContractError, match="secure repository lstat failed: artifacts"):
        V1_CONTRACT.training_contract_binding(archive_root)

    v2a_loaded = contract.load_training_v2a_contract(archive_root)
    v2a_validated = contract.validate_training_v2a_contract(_config())
    assert v2a_loaded == v2a_validated
    v2a_canonical = contract.canonical_training_v2a_contract_bytes(v2a_loaded)
    assert len(v2a_canonical) == contract.V2A_CONFIG_CANONICAL_SIZE_BYTES
    assert hashlib.sha256(v2a_canonical).hexdigest() == contract.V2A_CONFIG_CANONICAL_SHA256
    with pytest.raises(contract.TrainingV2AContractError, match="secure repository path access failed"):
        contract.training_v2a_contract_binding(archive_root)

    portable = _portable_root(tmp_path / "portable-positive", include_authority=True)
    binding = contract.training_v2a_contract_binding(portable)
    assert binding["authority_binding"]["before_after_identity_pass"] is True

    spec = importlib.util.spec_from_file_location("repository_contract_check_for_v2a_archive", archive_root / "tools/repository_contract_check.py")
    checker = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(checker)
    assert checker.check_repository(archive_root) is False
    assert capsys.readouterr().out.splitlines() == [
        "FAIL: formal training contract validation failure: TrainingContractError: secure repository lstat failed: artifacts"
    ]


def test_launch_time_target_absence_remains_production_preflight_owned():
    outer_source = (ROOT / "src/sparse_rtdetr/baseline/training_t6_outer_launcher.py").read_text(encoding="utf-8")
    ast_tree = ast.parse(outer_source)
    run_outer_once = next(node for node in ast_tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_outer_once")
    assert 'for target in checked["targets"].values()' in outer_source
    assert 'if path.exists() or path.is_symlink()' in outer_source
    assert '_fail("a production target already exists")' in outer_source
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_fail"
        and any(isinstance(argument, ast.Constant) and argument.value == "a production target already exists" for argument in node.args)
        for node in ast.walk(run_outer_once)
    )
