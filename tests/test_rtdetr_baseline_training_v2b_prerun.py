"""CPU-only pre-run policy and identity guards; no real images or training.

The tool itself is not executed. Policy validation uses its committed JSON
recipes; pretrained checks read only tiny synthetic byte files. Binding tests
use nonexistent synthetic metadata paths and trap every evidence file read.
"""

from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

from sparse_rtdetr.baseline import training_v2b_evidence as evidence
from sparse_rtdetr.baseline.training_v2b import V2BConfig


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tool():
    # Import the helper definitions without calling verify() or main(). Restore
    # the tool's startup environment edits when this test module is finished.
    keys = ("CUDA_VISIBLE_DEVICES", "MKL_THREADING_LAYER", "PYTHONNOUSERSITE",
            "OMP_NUM_THREADS", "MKL_NUM_THREADS")
    environment = {key: os.environ.get(key) for key in keys}
    original_path, original_bytecode = list(sys.path), sys.dont_write_bytecode
    spec = importlib.util.spec_from_file_location(
        "v2b_prerun_policy_review", ROOT / "tools/verify_training_v2b_prerun.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        yield module
    finally:
        sys.path[:] = original_path
        sys.dont_write_bytecode = original_bytecode
        for key, value in environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@pytest.fixture
def policies():
    model = json.loads((ROOT / "configs/baseline/rtdetrv2_r18_visdrone_baseline_v2b_engineering.json").read_text())
    runtime = json.loads((ROOT / "configs/baseline/rtdetrv2_r18_visdrone_baseline_v2b_runtime.json").read_text())
    return model, runtime


def _set(document, path, value):
    node = document
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value


def test_committed_prerun_policy_is_validated_without_model_or_data_execution(tool, policies, monkeypatch):
    model, runtime = policies
    before = deepcopy(policies)
    def forbidden(*args, **kwargs):
        raise AssertionError("policy validation cannot construct a model, read files, or initialize CUDA")
    monkeypatch.setattr(tool, "build_v2b_components", forbidden)
    monkeypatch.setattr(tool, "build_train_core_loader", forbidden)
    monkeypatch.setattr(tool, "file_reference", forbidden)
    monkeypatch.setattr(tool.torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(tool.torch.cuda, "init", forbidden)
    result = tool._validate_policy(model, runtime)
    assert isinstance(result, V2BConfig)
    assert (result.seed, result.input_size, result.physical_batch_size,
            result.accumulation_steps, result.logical_batch_size) == (0, 640, 16, 1, 16)
    assert policies == before


@pytest.mark.parametrize("path,value", [
    (("current_scope",), "formal_training"),
    (("next_gpu_stage", "authorized"), True),
    (("next_gpu_stage", "authorized"), 0),
    (("scientific_control_after_smoke", "authorized"), True),
    (("data", "role"), "confirmatory"),
    (("data", "logical_batch_size"), 8),
    (("data", "multiprocessing_context"), "fork"),
    (("data", "MKL_THREADING_LAYER"), "INTEL"),
    (("data", "persistent_workers"), True),
    (("data", "pin_memory"), True),
    (("data", "workers_to_verify"), []),
    (("data", "workers_to_verify"), [0]),
    (("data", "workers_to_verify"), [0, 2]),
    (("data", "workers_to_verify"), [0, 2, 4, 8]),
    (("data", "workers_to_verify"), [0, 2, 2]),
    (("data", "workers_to_verify"), [0, 4, 2]),
    (("next_gpu_stage", "logical_windows_per_arm"), 1),
    (("next_gpu_stage", "checkpoint_after_logical_window"), 1),
    (("scientific_control_after_smoke", "early_checkpoint_epoch"), 9),
    (("scientific_control_after_smoke", "primary_comparison_endpoint_epoch"), 10),
])
def test_changed_runtime_policy_cannot_still_claim_the_required_prerun(tool, policies, path, value):
    model, runtime = policies
    _set(runtime, path, value)
    with pytest.raises((ValueError, TypeError, KeyError)):
        tool._validate_policy(model, runtime)


@pytest.mark.parametrize("key", [
    "formal_training_authorized", "GPU_smoke_authorized",
    "high_resolution_896_960_authorized", "confirmatory_test_access_authorized",
    "scientific_certified",
])
def test_runtime_readiness_cannot_grant_any_unlaunched_action(tool, policies, key):
    model, runtime = policies
    runtime["readiness"][key] = True
    with pytest.raises((ValueError, TypeError, KeyError)):
        tool._validate_policy(model, runtime)


@pytest.mark.parametrize("path,value", [
    (("scope",), "train_core_runtime_engineering"),
    (("config", "seed"), 1),
    (("config", "input_size"), 128),
    (("config", "physical_batch_size"), 8),
    (("config", "accumulation_steps"), 2),
    (("readiness", "gpu_training_authorized"), True),
    (("readiness", "high_resolution_experiment_authorized"), True),
    (("readiness", "confirmatory_test_access_authorized"), True),
    (("readiness", "scientific_certified"), True),
])
def test_changed_model_policy_is_rejected_before_construction(tool, policies, path, value):
    model, runtime = policies
    _set(model, path, value)
    with pytest.raises((ValueError, TypeError, KeyError)):
        tool._validate_policy(model, runtime)


@pytest.mark.parametrize("damage", [
    "same_physical_arm_twice", "alternate_split_same_logical_batch",
    "reversed", "one", "extra", "empty", "path_escape_id", "absolute_id",
    "wrong_resolution", "wrong_second_name",
])
def test_exact_requested_pair_and_safe_output_identifiers_are_required(tool, policies, damage):
    model, runtime = policies
    arms = runtime["next_gpu_stage"]["arms"]
    if damage == "same_physical_arm_twice":
        arms[1].update(physical_batch_size=16, accumulation_steps=1)
    elif damage == "alternate_split_same_logical_batch":
        arms[1].update(physical_batch_size=4, accumulation_steps=4)
    elif damage == "reversed":
        arms.reverse()
    elif damage == "one":
        arms.pop()
    elif damage == "extra":
        arms.append(deepcopy(arms[0]))
    elif damage == "empty":
        arms.clear()
    elif damage == "path_escape_id":
        arms[0]["id"] = "../outside-approved-output"
    elif damage == "absolute_id":
        arms[0]["id"] = "/outside-approved-output"
    elif damage == "wrong_resolution":
        arms[1]["input_size"] = 128
    else:
        arms[1]["id"] = "640x16x1"
    with pytest.raises((ValueError, TypeError, KeyError)):
        tool._validate_policy(model, runtime)


def _authority(raw):
    return {
        "weight_relative_path": "artifacts/pretrained_rtdetrv2_presnet18_imagenet_v1/ResNet18_vd_pretrained_from_paddle.pth",
        "weight_sha256": hashlib.sha256(raw).hexdigest(),
        "weight_size_bytes": len(raw),
    }


def _weight_file(root, raw=b"synthetic authority bytes; not a torch checkpoint"):
    root.mkdir(parents=True, exist_ok=True)
    path = root / "ResNet18_vd_pretrained_from_paddle.pth"
    path.write_bytes(raw)
    return path, _authority(raw)


def test_explicit_pretrained_reference_verifies_actual_synthetic_file_bytes(tool, tmp_path):
    path, authority = _weight_file(tmp_path / "approved")
    reference = tool._pretrained_reference(str(path), authority)
    assert reference["sha256"] == authority["weight_sha256"]
    assert reference["size_bytes"] == authority["weight_size_bytes"]
    assert reference["sha256_scope"] == "complete_file_bytes"


@pytest.mark.parametrize("field,value", [
    ("weight_sha256", "a" * 64), ("weight_size_bytes", 1),
])
def test_pretrained_authority_byte_mismatch_is_rejected(tool, tmp_path, field, value):
    path, authority = _weight_file(tmp_path / "approved")
    authority[field] = value
    with pytest.raises((ValueError, RuntimeError)):
        tool._pretrained_reference(path, authority)


@pytest.mark.parametrize("component", [
    "test", "confirmatory", "confirmatory_holdout", "development", "val",
    "VisDrone2019-DET-val", "VisDrone2019-DET-test-dev",
    "VisDrone2019-DET-test-challenge",
])
def test_forbidden_raw_pretrained_path_is_rejected_before_any_content_read(tool, tmp_path, monkeypatch, component):
    # These are synthetic paths only; the test never creates or reads held-out
    # experiment data. A spy proves rejection precedes file_reference.
    path = tmp_path / component / "ResNet18_vd_pretrained_from_paddle.pth"
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(args)
        raise AssertionError("a forbidden pretrained path must be rejected before reading bytes")
    monkeypatch.setattr(tool, "file_reference", forbidden)
    with pytest.raises((ValueError, RuntimeError)):
        tool._pretrained_reference(path, _authority(b"synthetic"))
    assert calls == []


@pytest.mark.parametrize("kind", ["wrong_basename", "resolved_forbidden_parent",
                                   "raw_forbidden_alias", "resolved_wrong_basename"])
def test_pretrained_resolution_cannot_bypass_path_or_authority_guards(tool, tmp_path, monkeypatch, kind):
    good, authority = _weight_file(tmp_path / "approved")
    if kind == "wrong_basename":
        path = good.with_name("unrelated-data.pth")
        path.write_bytes(good.read_bytes())
    elif kind == "resolved_forbidden_parent":
        target, _ = _weight_file(tmp_path / "confirmatory")
        alias = tmp_path / "approved-alias"
        alias.symlink_to(target.parent, target_is_directory=True)
        path = alias / target.name
    elif kind == "raw_forbidden_alias":
        alias = tmp_path / "test"
        alias.symlink_to(good.parent, target_is_directory=True)
        path = alias / good.name
    else:
        target = tmp_path / "another-file.pth"
        target.write_bytes(good.read_bytes())
        directory = tmp_path / "resolved-name"
        directory.mkdir()
        path = directory / good.name
        path.symlink_to(target)
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(args)
        raise AssertionError("path rejection must precede any file_reference call")
    monkeypatch.setattr(tool, "file_reference", forbidden)
    with pytest.raises((ValueError, RuntimeError)):
        tool._pretrained_reference(path, authority)
    assert calls == []


def _reference(path, sha, size):
    return {"path": str(path), "sha256": sha, "size_bytes": size,
            "sha256_scope": "complete_file_bytes"}


def _rehash(binding):
    loader = binding["data"]["loader"]
    body = {key: value for key, value in loader.items() if key != "loader_binding_sha256"}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode()).hexdigest()
    loader["loader_binding_sha256"] = digest
    binding["data"]["loader_binding_sha256"] = digest
    binding["binding_sha256"] = evidence.canonical_sha256(
        {key: value for key, value in binding.items() if key != "binding_sha256"},
    )
    return binding


def _synthetic_binding(tmp_path):
    # All referenced paths are intentionally nonexistent. verify_files=False
    # validates structure and binding identity, not actual source/data contents.
    source_name, source_sha = "synthetic_source.py", "a" * 64
    semantic = {
        "role": "train_core", "seed": 0, "input_size": 640,
        "logical_batch_size": 16, "augmentation_stop_internal_epoch": 117,
        "tail_policy": "drop_incomplete_logical_batch",
        "sample_rng": "sha256(seed,logged_epoch,order_position,stable_image_id)",
        "epoch_numbering": "logged_one_based_vendor_zero_based",
        "multiscale": False, "category_mapping": "1..10_to_0..9",
        "source_verification": "manifest_sha256_before_decode",
    }
    loader = {
        "schema_version": 1, "role": "train_core",
        "annotation": _reference(tmp_path / "train_core_coco.json", "b" * 64, 200),
        "manifest": _reference(tmp_path / "train_core_manifest.json", "c" * 64, 300),
        "image_root": str(tmp_path / "approved_images"),
        "sample_count": 32, "annotation_count": 48,
        "config": semantic, "transforms": {"fixture": "identity_only_no_transform_execution"},
        "source": {"sources": [{"path": source_name, "sha256": source_sha}],
                   "torch": "synthetic_version", "torchvision": "synthetic_version",
                   "numpy": "synthetic_version", "pillow": "synthetic_version"},
    }
    inventory = [{
        "name": "synthetic_parameter", "dtype": "torch.float32", "shape": [1],
        "size_bytes": 4, "sha256": hashlib.sha256(bytes(4)).hexdigest(),
    }]
    initial = {
        "sha256": evidence.canonical_sha256(inventory),
        "sha256_scope": "canonical_tensor_inventory_v1", "inventory": inventory,
    }
    binding = {
        "schema_version": 1, "run_id": "synthetic-prerun-reference-validation",
        "code": {source_name: _reference(tmp_path / source_name, source_sha, 12)},
        "config": {
            "scope": "train_core_runtime_engineering", "device": "cpu",
            "seed": 0, "input_size": 640, "logical_batch_size": 16,
            "augmentation_stop_internal_epoch": 117,
        },
        "initial_weights": initial,
        "data": {"kind": "train_core_runtime", "loader": loader},
    }
    return _rehash(binding)


def _forbid_evidence_io(monkeypatch):
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(args)
        raise AssertionError("verify_files=False must not read or resolve filesystem references")
    monkeypatch.setattr(evidence, "_read_regular", forbidden)
    monkeypatch.setattr(Path, "resolve", forbidden)
    return calls


def test_metadata_structure_can_be_validated_without_any_file_reads(tmp_path, monkeypatch):
    binding = _synthetic_binding(tmp_path)
    with monkeypatch.context() as no_io:
        calls = _forbid_evidence_io(no_io)
        assert evidence.validate_run_binding(binding, verify_files=False) == binding
        assert calls == []


@pytest.mark.parametrize("role", ["annotation", "manifest"])
@pytest.mark.parametrize("field,damage", [
    ("sha256", "bad"), ("sha256", "A" * 64), ("sha256", None),
    ("sha256", "REMOVE"), ("size_bytes", -1), ("size_bytes", True),
    ("size_bytes", "200"), ("size_bytes", "REMOVE"),
    ("sha256_scope", "prefix_only"), ("sha256_scope", None),
])
def test_rehashed_invalid_metadata_references_are_rejected_even_without_file_io(
        tmp_path, monkeypatch, role, field, damage):
    binding = _synthetic_binding(tmp_path)
    reference = binding["data"]["loader"][role]
    if damage == "REMOVE":
        del reference[field]
    else:
        reference[field] = damage
    _rehash(binding)  # Prove this is schema validation, not merely digest mismatch.
    with monkeypatch.context() as no_io:
        calls = _forbid_evidence_io(no_io)
        with pytest.raises(evidence.EvidenceError):
            evidence.validate_run_binding(binding, verify_files=False)
        assert calls == []
