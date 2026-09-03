from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import os
import stat
from pathlib import Path

import pytest

from sparse_rtdetr.baseline import training_contract
from sparse_rtdetr.baseline import training_runtime as runtime


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "sparse_rtdetr" / "baseline" / "training_runtime.py"
CONFIG = ROOT / runtime.RUNTIME_PLAN_CONFIG_RELATIVE_PATH


def _plan() -> dict:
    return runtime.load_training_runtime_plan(ROOT)


def _training() -> dict:
    return training_contract.load_training_contract(ROOT)


def _binding_for(plan: dict) -> dict:
    return runtime._expected_training_binding(plan)


def _runtime_fixture(tmp_path: Path, raw: bytes | None = None) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    path = root / runtime.RUNTIME_PLAN_CONFIG_RELATIVE_PATH
    path.parent.mkdir(parents=True)
    path.write_bytes(CONFIG.read_bytes() if raw is None else raw)
    return root, path


def _replace_with_hardlink(path: Path, target: Path) -> None:
    target.write_bytes(path.read_bytes())
    path.unlink()
    path.hardlink_to(target)


def test_runtime_config_raw_and_canonical_identity_are_frozen():
    raw = CONFIG.read_bytes()
    plan = _plan()
    canonical = runtime.canonical_training_runtime_plan_bytes(plan)
    assert len(raw) == runtime.RUNTIME_PLAN_RAW_SIZE_BYTES
    assert hashlib.sha256(raw).hexdigest() == runtime.RUNTIME_PLAN_RAW_SHA256
    assert len(canonical) == runtime.RUNTIME_PLAN_CANONICAL_SIZE_BYTES
    assert hashlib.sha256(canonical).hexdigest() == runtime.RUNTIME_PLAN_CANONICAL_SHA256
    assert raw.endswith(b"\n")
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert b"\x00" not in raw
    assert b"\r" not in raw
    assert not canonical.endswith(b"\n")
    assert canonical.decode("ascii")


def test_runtime_schema_has_closed_root_and_nested_objects():
    plan = _plan()
    assert set(plan) == {
        "schema_version",
        "runtime_plan_id",
        "training_contract_id",
        "runtime_stage",
        "source_bindings",
        "invocation_policy",
        "model",
        "data_roles",
        "optimizer",
        "schedule",
        "amp",
        "ema",
        "augmentation",
        "evaluation_and_selection",
        "checkpoint_policy",
        "evidence_and_readiness",
        "vendor_conflicts",
    }
    for group in (
        "source_bindings",
        "invocation_policy",
        "model",
        "data_roles",
        "optimizer",
        "schedule",
        "amp",
        "ema",
        "augmentation",
        "evaluation_and_selection",
        "checkpoint_policy",
        "evidence_and_readiness",
    ):
        assert type(plan[group]) is dict
    assert len(plan["vendor_conflicts"]) == 14
    assert all(type(row) is dict and set(row) == {"id", "vendor_observation", "formal_override"} for row in plan["vendor_conflicts"])


@pytest.mark.parametrize(
    ("pointer", "value"),
    [
        (("runtime_plan_id",), "renamed"),
        (("invocation_policy", "retry_allowed"), True),
        (("model", "num_classes"), 11),
        (("optimizer", "parameter_groups", 1, "name"), "default"),
        (("schedule", "milestones"), []),
        (("augmentation", "transforms", 0, "p"), 0.75),
        (("checkpoint_policy", "required_state", 0), "ema"),
        (("vendor_conflicts", 0, "id"), "other"),
    ],
)
def test_schema_and_frozen_semantics_reject_mutations(pointer, value):
    candidate = _plan()
    target = candidate
    for token in pointer[:-1]:
        target = target[token]
    target[pointer[-1]] = value
    with pytest.raises(training_contract.TrainingContractError):
        runtime._validate_local_plan(candidate)


def test_schema_rejects_missing_extra_renamed_and_container_kind_mutations():
    candidate = _plan()
    candidate.pop("runtime_stage")
    with pytest.raises(training_contract.TrainingContractError, match="missing"):
        runtime._validate_local_plan(candidate)

    candidate = _plan()
    candidate["unexpected"] = True
    with pytest.raises(training_contract.TrainingContractError, match="extra"):
        runtime._validate_local_plan(candidate)

    candidate = _plan()
    candidate["optimizer"]["parameter_groups"] = {}
    with pytest.raises(training_contract.TrainingContractError, match="list type"):
        runtime._validate_local_plan(candidate)

    candidate = _plan()
    candidate["model"]["num_queries"] = [300]
    with pytest.raises(training_contract.TrainingContractError, match="scalar type"):
        runtime._validate_local_plan(candidate)


def test_strict_builtin_scalars_and_nonfinite_numbers_reject():
    class IntSubclass(int):
        pass

    class DictSubclass(dict):
        pass

    candidate = _plan()
    candidate["model"]["num_classes"] = IntSubclass(10)
    with pytest.raises(training_contract.TrainingContractError):
        runtime._validate_local_plan(candidate)

    candidate = DictSubclass(_plan())
    with pytest.raises(training_contract.TrainingContractError):
        runtime._validate_local_plan(candidate)

    for value in (math.nan, math.inf, -math.inf):
        candidate = _plan()
        candidate["amp"]["init_scale"] = value
        with pytest.raises(training_contract.TrainingContractError):
            runtime._validate_local_plan(candidate)
        with pytest.raises(training_contract.TrainingContractError):
            runtime.canonical_training_runtime_plan_bytes(candidate)


def test_scalar_owner_registry_is_complete_and_nonoverlapping():
    leaves = set(runtime._FROZEN_LEAVES)
    numeric = set(runtime._NUMERIC_CONSTRAINTS)
    semantic = set(runtime._SEMANTIC_LEAVES)
    assert leaves == numeric | semantic
    assert not numeric & semantic
    runtime._validate_owner_registry()


def test_all_frozen_runtime_mapping_values_are_explicit():
    plan = _plan()
    assert plan["runtime_stage"] == "pre_cuda_plan_only"
    assert plan["model"] == {
        "type": "RTDETR",
        "backbone": "PResNet-18",
        "encoder": "HybridEncoder",
        "decoder": "RTDETRTransformerv2",
        "num_classes": 10,
        "parameter_count": 20094584,
        "input_size": [640, 640],
        "num_queries": 300,
        "decoder_layers": 3,
    }
    assert plan["data_roles"] == {
        "train_role": "train_core",
        "development_role": "development",
        "confirmatory_role": "sealed_and_forbidden",
        "test_role": "forbidden",
        "train_batch": 16,
        "development_batch": 32,
        "train_workers": 4,
        "development_workers": 4,
        "drop_last_train": True,
        "drop_last_development": False,
    }
    assert plan["schedule"] == {
        "epochs": 120,
        "warmup_type": "LinearWarmup",
        "warmup_optimizer_steps": 2000,
        "scheduler_type": "MultiStepLR",
        "scheduler_step_unit": "epoch",
        "milestones": [1000],
        "gamma": 0.1,
        "expected_decay_events": 0,
        "checkpoint_every_epoch": 1,
        "periodic_checkpoint_every_epochs": 10,
        "development_evaluation_every_epochs": 1,
        "augmentation_stop_epoch": 117,
        "multiscale_enabled": False,
    }
    assert plan["amp"] == {
        "enabled": True,
        "scaler_type": "GradScaler",
        "init_scale": 65536.0,
        "growth_factor": 2.0,
        "backoff_factor": 0.5,
        "growth_interval": 2000,
        "nonfinite_loss_allowed": 0,
        "nonfinite_gradient_allowed": 0,
        "optimizer_skipped_steps_allowed": 0,
        "overflow_events_allowed": 0,
    }
    assert plan["ema"]["decay"] == 0.9999
    assert plan["ema"]["warmups"] == 2000
    assert plan["ema"]["development_evaluation_weights"] == "ema"
    assert plan["ema"]["model_selection_weights"] == "ema"
    assert plan["augmentation"]["stopped_transforms"] == [
        "RandomPhotometricDistort",
        "RandomZoomOut",
        "RandomIoUCrop",
    ]
    assert plan["augmentation"]["transforms"] == [
        {"type": "RandomPhotometricDistort", "p": 0.5},
        {"type": "RandomZoomOut", "fill": 0},
        {"type": "RandomIoUCrop", "p": 0.8},
        {"type": "SanitizeBoundingBoxes", "min_size": 1},
        {"type": "RandomHorizontalFlip"},
        {"type": "Resize", "size": [640, 640]},
        {"type": "SanitizeBoundingBoxes", "min_size": 1},
        {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
        {"type": "ConvertBoxes", "fmt": "cxcywh", "normalize": True},
    ]


def test_optimizer_groups_are_ordered_mutually_exclusive_and_full_coverage_declared():
    optimizer = _plan()["optimizer"]
    assert optimizer["type"] == "AdamW"
    assert optimizer["parameter_name_match_mode"] == "case_sensitive_literal_substring"
    assert [group["name"] for group in optimizer["parameter_groups"]] == [
        "backbone_non_norm",
        "norm_or_bn",
        "default",
    ]
    assert optimizer["parameter_groups"][0]["exclude_substrings"] == ["norm", "bn"]
    assert optimizer["parameter_groups"][1]["include_substrings"] == ["norm", "bn"]
    assert optimizer["parameter_groups"][2]["include_substrings"] == []
    assert optimizer["mutually_exclusive"] is True
    assert optimizer["cover_all_trainable_parameters"] is True
    assert optimizer["parameter_identity_audit_required"] is True
    assert optimizer["betas"] == [0.9, 0.999]
    assert optimizer["gradient_clip_max_norm"] == 0.1


def test_evaluation_checkpoint_and_readiness_policy_is_closed():
    plan = _plan()
    evaluation = plan["evaluation_and_selection"]
    assert evaluation["primary_evaluator_id"] == "visdrone_official_primary_evaluator_v1"
    assert evaluation["primary_protocol_id"] == "visdrone_official_style_v1"
    assert evaluation["secondary_diagnostic_only"] is True
    assert evaluation["secondary_cannot_certify_or_select"] is True
    assert evaluation["selection_metric"] == "AP@[0.50:0.95,maxDets=500]"
    assert evaluation["tie_breakers"] == ["AP50", "AR500", "earlier_epoch"]
    assert evaluation["metric_comparison_precision"] == "unrounded_float64"
    assert plan["checkpoint_policy"]["required_state"] == [
        "raw_model",
        "ema",
        "optimizer",
        "scheduler",
        "warmup",
        "grad_scaler",
        "epoch",
        "global_optimizer_step",
        "rng_states",
        "config_identity",
        "source_identity",
        "environment_identity",
    ]
    assert plan["checkpoint_policy"]["interrupted_status"] == "PERMANENT_FAIL"
    readiness = plan["evidence_and_readiness"]
    assert readiness["confirmatory_access"] is False
    assert readiness["test_access"] is False
    assert readiness["speed_measurement"] is False
    assert readiness["model_selection_certified"] is False
    assert readiness["training_implementation_ready"] is False
    assert readiness["training_ready"] is False


def test_vendor_conflict_inventory_covers_each_override():
    rows = _plan()["vendor_conflicts"]
    ids = [row["id"] for row in rows]
    assert ids == [
        "cli_resume",
        "cli_tuning",
        "cli_update",
        "cli_test_only",
        "solver_cuda_fallback",
        "solver_output_directory",
        "solver_checkpoint_loading",
        "checkpoint_logging",
        "yaml_evaluator",
        "best_stat",
        "upstream_pretrained",
        "upstream_sync_bn",
        "vendor_scaler_defaults",
        "vendor_multiscale_stop",
    ]
    assert all(row["vendor_observation"] and row["formal_override"] for row in rows)


def test_positive_detached_validation_chain_and_no_input_mutation():
    plan = _plan()
    training = _training()
    binding = _binding_for(plan)
    original_plan = copy.deepcopy(plan)
    original_training = copy.deepcopy(training)
    original_binding = copy.deepcopy(binding)
    validated = runtime.validate_training_runtime_plan(plan, training, binding)
    assert validated == plan
    assert validated is not plan
    assert validated["source_bindings"] is not plan["source_bindings"]
    validated["model"]["input_size"][0] = 1
    assert plan == original_plan
    assert training == original_training
    assert binding == original_binding


@pytest.mark.parametrize(
    "field",
    [
        "runtime_plan_id",
        "training_contract_id",
        "source_bindings",
        "invocation_policy",
        "model",
        "optimizer",
        "schedule",
        "amp",
        "ema",
        "augmentation",
        "evaluation_and_selection",
        "checkpoint_policy",
        "evidence_and_readiness",
        "vendor_conflicts",
    ],
)
def test_training_contract_and_runtime_authority_are_cross_bound(field):
    plan = _plan()
    training = _training()
    binding = _binding_for(plan)
    candidate = copy.deepcopy(plan)
    if field == "runtime_plan_id":
        candidate[field] = "other"
    elif field == "training_contract_id":
        candidate[field] = "other"
    elif field == "source_bindings":
        candidate[field]["vendor_runtime"]["file_count"] = 1
    elif field == "vendor_conflicts":
        candidate[field][0]["formal_override"] = "allowed"
    elif field == "augmentation":
        candidate[field]["stop_epoch"] = 118
    elif type(candidate[field]) is dict:
        first = next(iter(candidate[field]))
        value = candidate[field][first]
        if type(value) is bool:
            candidate[field][first] = not value
        elif type(value) is int:
            candidate[field][first] = value + 1
        elif type(value) is str:
            candidate[field][first] = "other"
        else:
            candidate[field][first] = copy.deepcopy(value)
    else:
        candidate[field] = copy.deepcopy(candidate[field])
    with pytest.raises(training_contract.TrainingContractError):
        runtime.validate_training_runtime_plan(candidate, training, binding)

    bad_binding = copy.deepcopy(binding)
    bad_binding["vendor_runtime_binding"]["total_size_bytes"] = 0
    with pytest.raises(training_contract.TrainingContractError):
        runtime.validate_training_runtime_plan(plan, training, bad_binding)


def test_training_runtime_plan_binding_returns_complete_detached_observed_identity():
    if not (ROOT / training_contract.CONVERSION_R3_RELATIVE_PATH).exists():
        before = sorted(path.relative_to(ROOT).as_posix() for path in ROOT.rglob("*"))
        with pytest.raises(training_contract.TrainingContractError):
            runtime.training_runtime_plan_binding(ROOT)
        after = sorted(path.relative_to(ROOT).as_posix() for path in ROOT.rglob("*"))
        assert before == after
        assert _plan()["runtime_stage"] == "pre_cuda_plan_only"
        assert runtime.canonical_training_runtime_plan_bytes(_plan())
        return
    result = runtime.training_runtime_plan_binding(ROOT)
    assert set(result) == {
        "schema_version",
        "runtime_plan_id",
        "relative_path",
        "raw_size_bytes",
        "raw_sha256",
        "canonical_size_bytes",
        "canonical_sha256",
        "training_contract_identity",
        "vendor_runtime_binding",
        "conversion_r3_runtime_binding",
        "primary_evaluator_runtime_binding",
        "plan",
    }
    assert result["runtime_plan_id"] == runtime.RUNTIME_PLAN_ID
    assert result["raw_size_bytes"] == runtime.RUNTIME_PLAN_RAW_SIZE_BYTES
    assert result["canonical_sha256"] == runtime.RUNTIME_PLAN_CANONICAL_SHA256
    assert result["training_contract_identity"]["module_sha256"] == runtime.TRAINING_CONTRACT_MODULE_SHA256
    plan = result["plan"]
    expected = _binding_for(plan)
    assert result["vendor_runtime_binding"] == expected["vendor_runtime_binding"]
    assert result["conversion_r3_runtime_binding"] == expected["conversion_r3_runtime_binding"]
    assert result["primary_evaluator_runtime_binding"] == expected["primary_evaluator_runtime_binding"]
    result["plan"]["model"]["num_classes"] = 1
    assert runtime.load_training_runtime_plan(ROOT)["model"]["num_classes"] == 10


def test_path_and_sha_mutations_are_rejected_before_authority_use():
    plan = _plan()
    for pointer, value in (
        (("source_bindings", "training_contract", "relative_path"), "../escape"),
        (("source_bindings", "training_contract", "raw_sha256"), "0"),
        (("source_bindings", "primary_evaluator", "implementation_commit"), "0" * 39),
        (("source_bindings", "conversion_r3", "artifact_root"), "/absolute"),
    ):
        candidate = copy.deepcopy(plan)
        target = candidate
        for token in pointer[:-1]:
            target = target[token]
        target[pointer[-1]] = value
        with pytest.raises(training_contract.TrainingContractError):
            runtime._validate_local_plan(candidate)


def test_raw_parser_rejects_bom_nul_cr_duplicate_and_wrong_lf():
    bad_values = [
        b"\xef\xbb\xbf{}\n",
        b"{\"x\": \"\x00\"}\n",
        b"{\"x\": 1}\r\n",
        b"{\"x\": 1, \"x\": 2}\n",
        b"{}",
        b"{}\n\n",
    ]
    for raw in bad_values:
        with pytest.raises(training_contract.TrainingContractError):
            training_contract._parse_portable_json(raw, label="runtime test")


def test_runtime_file_repack_and_format_drift_reject(tmp_path):
    raw = CONFIG.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    repacked = json.dumps(value, ensure_ascii=True, sort_keys=True, indent=4).encode("utf-8") + b"\n"
    root, _ = _runtime_fixture(tmp_path, repacked)
    with pytest.raises(training_contract.TrainingContractError):
        runtime.load_training_runtime_plan(root)


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "directory", "fifo", "missing"])
def test_runtime_file_object_matrix_fails_closed(tmp_path, kind):
    root, path = _runtime_fixture(tmp_path)
    external = tmp_path / "external"
    try:
        if kind == "symlink":
            path.unlink()
            path.symlink_to(CONFIG)
        elif kind == "hardlink":
            _replace_with_hardlink(path, external)
        elif kind == "directory":
            path.unlink()
            path.mkdir()
        elif kind == "fifo":
            path.unlink()
            os.mkfifo(path)
        elif kind == "missing":
            path.unlink()
        else:
            raise AssertionError(kind)
        with pytest.raises(training_contract.TrainingContractError):
            runtime.load_training_runtime_plan(root)
    finally:
        if path.is_symlink() or path.is_file() or path.is_fifo():
            path.unlink()
        elif path.is_dir():
            path.rmdir()
        if external.exists():
            external.unlink()


def test_runtime_root_and_config_path_containment_fail_closed(tmp_path):
    root, _ = _runtime_fixture(tmp_path)
    with pytest.raises(training_contract.TrainingContractError):
        runtime.load_training_runtime_plan(root, "../runtime.json")
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(training_contract.TrainingContractError):
        runtime.load_training_runtime_plan(alias)


def test_module_imports_only_portable_contract_dependencies():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert "sparse_rtdetr.baseline.training_contract" in imported
    forbidden = ("torch", "numpy", "PIL", "cv2", "vendor", "dataset", "evaluator", "subprocess")
    assert not any(any(name == item or name.startswith(item + ".") for item in imported) for name in forbidden)
    assert set(runtime.__all__) == {
        "load_training_runtime_plan",
        "validate_training_runtime_plan",
        "canonical_training_runtime_plan_bytes",
        "training_runtime_plan_binding",
    }


def test_public_operations_do_not_mutate_environment_or_create_files(monkeypatch, tmp_path):
    before = dict(os.environ)
    candidate = _plan()
    training = _training()
    binding = _binding_for(candidate)
    runtime.validate_training_runtime_plan(candidate, training, binding)
    assert dict(os.environ) == before
    assert not list(tmp_path.iterdir())


def test_authority_plan_contains_no_host_absolute_data_paths():
    payload = json.dumps(_plan(), ensure_ascii=True, sort_keys=True)
    assert "media" + os.sep not in payload
    assert "home" + os.sep not in payload
    assert "data_root" not in payload
