"""CPU admission boundaries for development evaluation and paired seed replication.

All hardware commands and CUDA queries are mocked or forbidden. Fixtures contain
only temporary synthetic metadata and CPU tensor bytes; no real data is opened.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sparse_rtdetr.baseline import training_v2b_admission as ad
from sparse_rtdetr.baseline import training_v2b_evidence as ev
from test_rtdetr_baseline_training_v2b_hardware import (
    UUID, bound, native_fixture, forbid_unmocked_gpu_queries,
)
from test_rtdetr_baseline_training_v2b_admission import (
    observation, policy_bundle, external_receipt, no_native_gpu_or_setter,
    repair_policy_bundle, resolution_896_policy_bundle,
)


def _rehash(binding):
    binding["binding_sha256"] = ev.canonical_sha256(
        {key: value for key, value in binding.items() if key != "binding_sha256"})
    return binding


@pytest.fixture
def evidence_policy_bundle(tmp_path, monkeypatch, policy_bundle, external_receipt):
    auth = tmp_path / "resolution-evidence-authorization.txt"
    auth.write_text("CPU fixture: development evaluation and seed 1/2; no GPU authorization\n")
    reference = ev.file_reference(auth)
    monkeypatch.setattr(ad, "_RESOLUTION_EVIDENCE_AUTHORIZATION_SHA", reference["sha256"])
    return ad.build_policy_bundle(
        Path(policy_bundle["authority"]["path"]).parent,
        authorization_reference=reference, setter_mode=ad._EXTERNAL_ADMIN_MODE,
        external_clock_receipt=external_receipt["reference"])


def _evaluation_binding(bound, tmp_path, policy, size=640):
    result = copy.deepcopy(bound)
    manifest = tmp_path / "development_manifest.json"
    manifest.write_text('{"role":"development","fixture":"CPU only","records":[]}\n')
    result["config"].update(scope="development_cross_eval", input_size=size,
                            cuda_gpu_uuid=UUID,
                            authorization_reference=copy.deepcopy(policy["authorization_reference"]))
    result["data"] = {"kind": "manifest_binding_only",
                      "manifests": {"development": ev.file_reference(manifest)},
                      "dataset_executed": False}
    return _rehash(result)


def _replication_binding(bound, tmp_path, *, size=640, seed=1, batch=8, accumulation=2):
    result = copy.deepcopy(bound)
    annotation = tmp_path / "train_core_coco.json"
    annotation.write_text('{"fixture":"CPU only","images":[],"annotations":[]}\n')
    manifest = tmp_path / "train_core_manifest.json"
    manifest.write_text('{"role":"train_core","fixture":"CPU only","records":[]}\n')
    source = next(iter(result["code"].values()))
    result["code"] = {source["path"]: source}
    result["config"].update(scope="train_core_runtime_engineering", input_size=size,
                            seed=seed, physical_batch_size=batch,
                            accumulation_steps=accumulation, logical_batch_size=16,
                            augmentation_stop_internal_epoch=117, cuda_gpu_uuid=UUID)
    semantic = {"role": "train_core", **{key: result["config"][key] for key in (
        "seed", "input_size", "logical_batch_size", "augmentation_stop_internal_epoch")}}
    loader = {"schema_version": 1, "role": "train_core",
              "annotation": ev.file_reference(annotation), "manifest": ev.file_reference(manifest),
              "image_root": str(tmp_path / "not_loaded_images"), "sample_count": 16,
              "annotation_count": 0, "config": semantic, "transforms": {"CPU_fixture": True},
              "source": {"sources": [{"path": source["path"], "sha256": source["sha256"]}]}}
    digest = hashlib.sha256(json.dumps(loader, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode()).hexdigest()
    loader["loader_binding_sha256"] = digest
    result["data"] = {"kind": "train_core_runtime", "loader": loader,
                      "loader_binding_sha256": digest}
    return _rehash(result)


def test_new_authority_adds_only_bounded_evaluation_without_hardware_changes(
        evidence_policy_bundle, external_receipt):
    old = external_receipt["build"](external_receipt["reference"])
    assert evidence_policy_bundle["authorized_scope_limits_seconds"] == {
        "paired_smoke": 600, "train_core_30epoch": 43200, "development_cross_eval": 1800}
    different = {"authorization_reference", "authorized_scope_limits_seconds", "bundle_sha256"}
    assert {k: v for k, v in old.items() if k not in different} == {
        k: v for k, v in evidence_policy_bundle.items() if k not in different}
    selected = ad._validate_policy(evidence_policy_bundle, "development_cross_eval", 1800)
    assert selected["minimum_loaded_clock_samples"] == 3
    assert selected["clock_period_seconds"] == .2
    assert selected["clock_max_gap_seconds"] == 1.
    assert selected["health_period_seconds"] == 1.
    assert selected["health_max_gap_seconds"] == 2.
    assert selected["guardian_heartbeat_period_seconds"] == .1
    assert selected["guardian_heartbeat_max_gap_seconds"] == 1.


@pytest.mark.parametrize("deadline", [0, -1, 1801, 1800.0, True, "1800", None])
def test_evaluation_deadline_is_strict_and_has_no_fallback(evidence_policy_bundle, deadline):
    with pytest.raises(ad.MonitoredHardwareError, match="not authorized"):
        ad._validate_policy(evidence_policy_bundle, "development_cross_eval", deadline)


@pytest.mark.parametrize("scope,deadline", [
    ("paired_smoke", 601), ("train_core_30epoch", 43201),
    ("synthetic_operator_diagnostic", 600), ("production", 1),
    ("train_core_120epoch", 1), ("resolution_960", 1),
])
def test_new_authority_cannot_expand_other_scopes(evidence_policy_bundle, scope, deadline):
    with pytest.raises(ad.MonitoredHardwareError, match="not authorized"):
        ad._validate_policy(evidence_policy_bundle, scope, deadline)


@pytest.mark.parametrize("mode", ["direct", "sudo_n"])
def test_new_authority_never_permits_a_native_setter(evidence_policy_bundle, mode):
    with pytest.raises(ad.MonitoredHardwareError, match="external_admin_acknowledged"):
        ad.build_policy_bundle(
            Path(evidence_policy_bundle["authority"]["path"]).parent,
            authorization_reference=evidence_policy_bundle["authorization_reference"],
            setter_mode=mode)


def test_authorization_requires_exact_complete_file_bytes(evidence_policy_bundle):
    reference = evidence_policy_bundle["authorization_reference"]
    Path(reference["path"]).write_text("changed fixture authority\n")
    with pytest.raises(ad.MonitoredHardwareError, match="authorization reference differs"):
        ad._validate_policy(evidence_policy_bundle, "development_cross_eval", 1800)


@pytest.mark.parametrize("authority", ["original", "repair", "896"])
def test_prior_authority_cannot_inherit_development_scope(
        policy_bundle, repair_policy_bundle, resolution_896_policy_bundle, authority):
    original = {"original": policy_bundle, "repair": repair_policy_bundle,
                "896": resolution_896_policy_bundle}[authority]
    with pytest.raises(ad.MonitoredHardwareError, match="not authorized"):
        ad._validate_policy(original, "development_cross_eval", 1800)
    forged = copy.deepcopy(original)
    forged["authorized_scope_limits_seconds"]["development_cross_eval"] = 1800
    forged["bundle_sha256"] = ev.canonical_sha256(
        {k: v for k, v in forged.items() if k != "bundle_sha256"})
    with pytest.raises(ad.MonitoredHardwareError, match="differs"):
        ad._validate_policy(forged, "development_cross_eval", 1800)


@pytest.mark.parametrize("size", [640, 896])
def test_real_development_role_is_bound_before_any_native_access(
        evidence_policy_bundle, bound, tmp_path, size):
    binding = _evaluation_binding(bound, tmp_path, evidence_policy_bundle, size)
    output = tmp_path / "never-started"
    session = ad.MonitoredHardwareSession(
        binding, evidence_policy_bundle, output, "development_cross_eval", 1800)
    assert session._state == "new"
    assert session.binding["data"]["kind"] == "manifest_binding_only"
    assert set(session.binding["data"]["manifests"]) == {"development"}
    assert session.binding["config"]["authorization_reference"] == evidence_policy_bundle["authorization_reference"]
    assert not output.exists()


@pytest.mark.parametrize("size", [512, 960, 1024, 640.0, True, None])
def test_evaluation_geometry_is_exactly_640_or_896(evidence_policy_bundle, bound, tmp_path, size):
    binding = _evaluation_binding(bound, tmp_path, evidence_policy_bundle, size)
    with pytest.raises(ad.MonitoredHardwareError, match="input_size 640 or 896"):
        ad.MonitoredHardwareSession(
            binding, evidence_policy_bundle, tmp_path / "never-started", "development_cross_eval", 1800)


@pytest.mark.parametrize("change", ["scope", "missing_authority", "authority_sha", "authority_path"])
def test_evaluation_scope_and_authority_are_both_bound(
        evidence_policy_bundle, bound, tmp_path, change):
    binding = _evaluation_binding(bound, tmp_path, evidence_policy_bundle)
    if change == "scope":
        binding["config"]["scope"] = "train_core_runtime_engineering"
    elif change == "missing_authority":
        del binding["config"]["authorization_reference"]
    else:
        key = "sha256" if change == "authority_sha" else "path"
        binding["config"]["authorization_reference"][key] = "different"
    _rehash(binding)
    with pytest.raises(ad.MonitoredHardwareError, match="scope and exact authorization"):
        ad.MonitoredHardwareSession(
            binding, evidence_policy_bundle, tmp_path / "never-started", "development_cross_eval", 1800)


@pytest.mark.parametrize("role", ["train_core", "both", "synthetic"])
def test_wrong_evaluation_role_is_rejected_before_any_bound_data_read(
        evidence_policy_bundle, bound, tmp_path, monkeypatch, role):
    binding = _evaluation_binding(bound, tmp_path, evidence_policy_bundle)
    forbidden_reference = copy.deepcopy(binding["data"]["manifests"]["development"])
    if role == "synthetic":
        binding["data"] = copy.deepcopy(bound["data"])
    elif role == "train_core":
        binding["data"]["manifests"] = {"train_core": forbidden_reference}
    else:
        binding["data"]["manifests"]["train_core"] = forbidden_reference
    _rehash(binding)
    verify = ev._verify_reference
    def guarded(reference):
        assert reference != forbidden_reference, "rejected data was opened before its scope check"
        return verify(reference)
    monkeypatch.setattr(ev, "_verify_reference", guarded)
    output = tmp_path / "never-started"
    with pytest.raises(ad.MonitoredHardwareError, match="only its development manifest"):
        ad.MonitoredHardwareSession(
            binding, evidence_policy_bundle, output, "development_cross_eval", 1800)
    assert not output.exists()


@pytest.mark.parametrize("path", [
    "confirmatory/development_manifest.json", "confirmatory_extra/development_manifest.json",
    "test/development_manifest.json", "train_core/development_manifest.json",
    "development_coco.json",
])
def test_labeled_development_reference_cannot_escape_its_role(
        evidence_policy_bundle, bound, tmp_path, monkeypatch, path):
    binding = _evaluation_binding(bound, tmp_path, evidence_policy_bundle)
    reference = binding["data"]["manifests"]["development"]
    reference["path"] = str(tmp_path / path)  # Deliberately nonexistent; never opened.
    _rehash(binding)
    verify = ev._verify_reference
    def guarded(candidate):
        assert candidate != reference, "forbidden data path was opened"
        return verify(candidate)
    monkeypatch.setattr(ev, "_verify_reference", guarded)
    with pytest.raises(ad.MonitoredHardwareError, match="permitted data role"):
        ad.MonitoredHardwareSession(
            binding, evidence_policy_bundle, tmp_path / "never-started", "development_cross_eval", 1800)


@pytest.mark.parametrize("scope,deadline", [("paired_smoke", 600), ("train_core_30epoch", 43200)])
@pytest.mark.parametrize("seed", [1, 2])
@pytest.mark.parametrize("size", [640, 896])
def test_seed_replication_keeps_actual_train_core_binding_and_fixed_dimensions(
        evidence_policy_bundle, bound, tmp_path, scope, deadline, seed, size):
    binding = _replication_binding(bound, tmp_path, size=size, seed=seed)
    output = tmp_path / "never-started"
    session = ad.MonitoredHardwareSession(binding, evidence_policy_bundle, output, scope, deadline)
    assert session._state == "new"
    assert session.binding["data"]["kind"] == "train_core_runtime"
    assert session.binding["config"]["seed"] == seed
    assert session.binding["config"]["physical_batch_size"] == 8
    assert session.binding["config"]["accumulation_steps"] == 2
    assert not output.exists()


@pytest.mark.parametrize("seed", [0, 3])
def test_new_training_authority_only_permits_predetermined_seed_one_and_two(
        evidence_policy_bundle, bound, tmp_path, seed):
    binding = _replication_binding(bound, tmp_path, seed=seed)
    with pytest.raises(ad.MonitoredHardwareError, match="seed 1 or 2"):
        ad.MonitoredHardwareSession(
            binding, evidence_policy_bundle, tmp_path / "never-started", "paired_smoke", 600)


@pytest.mark.parametrize("batch,accumulation", [(16, 1), (4, 4), (8.0, 2), (8, 2.0), (True, 2), (8, True)])
def test_new_training_authority_cannot_change_physical_batch_or_accumulation(
        evidence_policy_bundle, bound, tmp_path, batch, accumulation):
    binding = _replication_binding(bound, tmp_path, batch=batch, accumulation=accumulation)
    with pytest.raises(ad.MonitoredHardwareError, match="physical batch 8 accumulation 2"):
        ad.MonitoredHardwareSession(
            binding, evidence_policy_bundle, tmp_path / "never-started", "paired_smoke", 600)


def test_replication_cannot_use_a_manifest_or_synthetic_substitute(
        evidence_policy_bundle, bound, tmp_path):
    binding = copy.deepcopy(bound)
    binding["config"].update(scope="train_core_runtime_engineering", seed=1, input_size=640,
                            physical_batch_size=8, accumulation_steps=2, cuda_gpu_uuid=UUID)
    _rehash(binding)
    with pytest.raises(ad.MonitoredHardwareError, match="actual train_core runtime"):
        ad.MonitoredHardwareSession(
            binding, evidence_policy_bundle, tmp_path / "never-started", "paired_smoke", 600)
