"""CPU tests for fixed-epoch summarization, never a model or GPU experiment."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from sparse_rtdetr.baseline import training_v2b_replication_summary as summary
from sparse_rtdetr.baseline.training_v2b import logical_batch_indices
from sparse_rtdetr.baseline.training_v2b_evidence import canonical_sha256, file_reference, write_exclusive_json


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    def blocked(*args, **kwargs):
        pytest.fail("fixed-epoch summary must not initialize or observe a GPU")
    monkeypatch.setattr(torch.cuda, "_lazy_init", blocked)
    monkeypatch.setattr(torch.cuda, "init", blocked)
    monkeypatch.setattr(torch.cuda, "is_available", blocked)
    from PIL import Image
    monkeypatch.setattr(Image, "open", lambda *a, **k: pytest.fail("summary opened image bytes"))


def ref(path, sha="a" * 64):
    return {"path": str(path), "sha256": sha, "size_bytes": 1, "sha256_scope": "complete_file_bytes"}


def endpoint(seed, size):
    direction = (1., -2., 3.)[seed] if size == 896 else 0.
    return {
        "seed": seed, "input_size": size, "epoch": 30,
        "uniform_official_primary": {"metrics": {name: 40. + direction for name in summary.PRIMARY_METRICS}},
        "legacy_primary_reported_separately": {"metrics": {name: 30. - direction for name in summary.PRIMARY_METRICS}},
        "converted_COCO_reported_separately": {"metrics": {
            name: (10. + 2 * direction if name == "AP_small" else 20. + direction) for name in summary.COCO_METRICS}},
    }


def endpoints():
    return {f"s{seed}_r{size}": endpoint(seed, size) for seed in summary.SEEDS for size in summary.SIZES}


def test_paired_statistics_use_sample_sd_and_keep_primary_and_coco_small_separate():
    result = summary.aggregate_fixed_epoch_results(endpoints())
    rows = result["paired_seed_results"]
    assert [row["differences"]["uniform_official_primary"]["AP"] for row in rows] == [1., -2., 3.]
    assert [row["differences"]["converted_COCO"]["AP_small"] for row in rows] == [2., -4., 6.]
    stats = result["mean_sample_sd_and_range"]["uniform_official_primary"]["AP"]
    assert stats["paired_difference_896_minus_640_pp"]["mean"] == pytest.approx(2 / 3)
    assert stats["paired_difference_896_minus_640_pp"]["sample_sd"] == pytest.approx((19 / 3) ** .5)
    assert stats["paired_difference_896_minus_640_pp"]["range"] == 5
    assert stats["paired_difference_896_minus_640_pp"]["sample_sd_ddof"] == 1
    assert stats["additional_seeds_1_and_2_paired_difference_pp"]["values"] == [-2., 3.]
    assert "AP_small" not in result["mean_sample_sd_and_range"]["uniform_official_primary"]
    assert result["seed0_was_used_for_conditional_replication_qualification"] is True
    assert result["three_seed_statistics_are_descriptive_and_conditioned_on_seed0_gate"] is True
    assert result["statistical_significance_claimed"] is False


@pytest.mark.parametrize("values", [[1.], [1., True], [1., float("nan")], [1., float("inf")]])
def test_bad_descriptive_inputs_are_rejected(values):
    with pytest.raises(summary.ReplicationSummaryError):
        summary.descriptive_statistics(values)


@pytest.mark.parametrize("change", ["missing", "epoch", "seed", "resolution"])
def test_summary_cannot_mix_missing_or_wrong_epoch_seed_cells(change):
    values = endpoints()
    if change == "missing":
        del values["s2_r896"]
    else:
        values["s2_r896"][{"epoch": "epoch", "seed": "seed", "resolution": "input_size"}[change]] = 99
    with pytest.raises(summary.ReplicationSummaryError):
        summary.aggregate_fixed_epoch_results(values)


def test_reader_hashes_complete_bytes_and_rejects_changed_content(tmp_path):
    path = tmp_path / "evidence.json"
    path.write_text('{"value":1}')
    reference = file_reference(path)
    reader = summary._Reader()
    assert reader.json(reference) == {"value": 1}
    assert reader.inventory() == [reference]
    path.write_text('{"value":2}')
    with pytest.raises(summary.ReplicationSummaryError, match="SHA"):
        reader.json(reference)


def test_reader_streaming_mode_does_not_keep_receipt_contents(tmp_path):
    path = tmp_path / "leaf.json"
    path.write_bytes(b" " * (2 * 1024 * 1024))
    reader = summary._Reader()
    assert reader.read(file_reference(path), retain=False) is None
    assert set(vars(reader)) == {"references"}
    assert reader.discover(path) == file_reference(path)


@pytest.mark.parametrize("role", ["test", "confirmatory", "confirmatory_extra", "VisDrone2019-DET-test-dev"])
def test_forbidden_reference_is_rejected_before_open(tmp_path, role, monkeypatch):
    monkeypatch.setattr(os, "open", lambda *a, **k: pytest.fail("forbidden evidence path was opened"))
    with pytest.raises(summary.ReplicationSummaryError, match="forbidden"):
        summary._Reader().read(ref(tmp_path / role / "data.json"))


def test_role_symlink_and_reference_scope_are_rejected(tmp_path):
    target = tmp_path / "real.json"; target.write_text("{}")
    link = tmp_path / "alias.json"; link.symlink_to(target)
    with pytest.raises(summary.ReplicationSummaryError, match="canonical"):
        summary._Reader().read({**file_reference(target), "path": str(link)})
    with pytest.raises(summary.ReplicationSummaryError, match="SHA"):
        summary._reference({**file_reference(target), "sha256_scope": "prefix"})


def scalar_state(values):
    return {"mapping_type": "dict", "items": [["str", key, value] for key, value in sorted(values.items())]}


def fixed_state(size=640, *, initial=False):
    epoch, updates = (0, 0) if initial else (30, 9120)
    return {"clocks": {
        "engine": {"epoch": epoch, "epoch_active": False, "optimizer_updates": updates,
                   "microsteps": updates * 2, "epoch_start_optimizer_updates": 0 if initial else 8816,
                   "phase": 0, "failed": False,
                   "config": {"physical_batch_size": 8, "accumulation_steps": 2,
                              "expected_input_size": [size, size]}},
        "loader": {"epoch": epoch, "epoch_active": False, "optimizer_updates": updates,
                   "logical_batch_size": 16, "dataset_size": 4869, "batches_per_epoch": 304,
                   "dropped_samples": 5, "completed_batches": 0 if initial else 304},
        "ema_updates": updates,
        "warmup": scalar_state({"last_step": updates, "warmup_duration": 2000}),
        "scheduler": scalar_state({"last_epoch": 0 if initial else 24, "_step_count": 1 if initial else 25}),
        "adam_steps_by_parameter": {} if initial else {"ordinary": 9120, "conditional_DN": 9119},
    }}


def test_fixed_clocks_accept_legitimate_conditional_adam_lag():
    summary._state_clocks(fixed_state(), input_size=640)
    summary._state_clocks(fixed_state(initial=True), input_size=640, initial=True)


@pytest.mark.parametrize("family,key,value", [
    ("engine", "epoch", 29), ("engine", "epoch_active", True), ("engine", "optimizer_updates", 9000),
    ("engine", "microsteps", 9120), ("engine", "failed", True), ("loader", "dropped_samples", 0),
    ("loader", "completed_batches", 303), ("loader", "epoch", 31),
])
def test_incomplete_or_changed_fixed_endpoint_clock_is_rejected(family, key, value):
    state = fixed_state()
    state["clocks"][family][key] = value
    with pytest.raises(summary.ReplicationSummaryError):
        summary._state_clocks(state, input_size=640)


def test_adam_future_step_is_rejected_without_requiring_all_parameters_equal():
    state = fixed_state()
    state["clocks"]["adam_steps_by_parameter"]["conditional_DN"] = 9121
    with pytest.raises(summary.ReplicationSummaryError, match="Adam"):
        summary._state_clocks(state, input_size=640)



def evaluation_fixture(tmp_path):
    final = fixed_state(896)
    policy = {"input_size": [896, 896], "nms": False, "num_top_queries": 300,
              "transform": [{"type": "Resize", "size": [896, 896]}]}
    receipts = [{"fixture_image_id": i} for i in range(548)]
    contract = {"output_dir": str(tmp_path), "config": {"input_size": 896},
                "development_binding": {
                    "binding_sha256": "b" * 64, "policy": copy.deepcopy(policy),
                    "image_manifest_sha256": canonical_sha256(receipts)}}
    runtime = {"cpu_fixture": True}
    evaluation = {
        "role": "development", "logged_epoch": 30, "full_development_evaluation": True,
        "checkpoint_selection_performed": False, "protocol_certification_claimed": False,
        "scope": "development_full_fixed_checkpoint", "scientific_comparison_eligible": True,
        "rng_restored": ["python", "numpy", "torch_cpu", "torch_cuda_bound_device"],
        "training_state_unchanged": {
            "engine": copy.deepcopy(final["clocks"]["engine"]), "ema_updates": 9120,
            "ema_sha256": "e" * 64,
            "auxiliary_clocks": {"warmup": {"last_step": 9120}, "scheduler": {"last_epoch": 24}}},
        "weights": {"kind": "ema", "ema_updates": 9120, "state_sha256": "e" * 64},
        "data_binding_sha256": "b" * 64, "policy": policy, "runtime": runtime,
        "data": {"evaluated_image_count": 548, "bound_image_count": 548,
                 "evaluated_ground_truth_count": 38759, "image_receipts": receipts},
        "prediction_artifact": ref(tmp_path / "predictions.json"),
        "result_artifact": ref(tmp_path / "evaluation.json"),
    }
    result = {"runtime": runtime, "final_state_reference": ref(tmp_path / "final-state.json"),
              "evaluations": [{"logged_epoch": 10}, {"logged_epoch": 20}, evaluation]}
    class Reader:
        def json(self, reference):
            if reference == result["final_state_reference"]:
                return final
            return {k: v for k, v in evaluation.items() if k != "result_artifact"}
    return Reader(), result, contract, evaluation


def test_saved_evaluation_is_bound_to_actual_policy_and_training_resolution(tmp_path):
    reader, result, contract, evaluation = evaluation_fixture(tmp_path)
    assert summary._evaluation(reader, result, contract) is evaluation


@pytest.mark.parametrize("mutation", ["size", "transform", "nms", "bound_wrong_size"])
def test_changed_saved_evaluation_policy_cannot_be_reported_as_labeled_resolution(tmp_path, mutation):
    reader, result, contract, evaluation = evaluation_fixture(tmp_path)
    if mutation in ("size", "bound_wrong_size"):
        evaluation["policy"]["input_size"] = [640, 640]
    elif mutation == "transform":
        evaluation["policy"]["transform"][0]["size"] = [640, 640]
    else:
        evaluation["policy"]["nms"] = True
    if mutation == "bound_wrong_size":
        contract["development_binding"]["policy"] = copy.deepcopy(evaluation["policy"])
    with pytest.raises(summary.ReplicationSummaryError, match="policy|resolution"):
        summary._evaluation(reader, result, contract)


def paired_fixture(monkeypatch, mutation=None):
    from sparse_rtdetr.baseline import training_v2b_resolution as resolution
    monkeypatch.setattr(resolution, "compare_matched_initialization", lambda *a: {"status": "PASS"})
    monkeypatch.setattr(resolution, "compare_matched_initial_state", lambda *a: {"status": "PASS"})
    plans = {epoch: [list(batch) for batch in logical_batch_indices(4869, seed=1, epoch=epoch, logical_batch_size=16)]
             for epoch in range(1, 31)}
    sources = []
    for size in (640, 896):
        references = [{**ref(Path(f"/cpu-evidence/{size}/{epoch}/{index}.json")),
                       "epoch": epoch, "logical_batch_index": index}
                      for epoch in range(1, 31) for index in range(304)]
        sources.append({
            "seed": 1, "input_size": size,
            "binding": {"binding_sha256": str(size)[0] * 64,
                        "data": {"loader_binding_sha256": ("a" if size == 640 else "b") * 64}},
            "result": {"initialization": {}, "runtime": {}, "receipts": references}, "initial": {},
            "contract": {"epoch_order_sha256": {str(e): canonical_sha256(plans[e]) for e in plans}},
        })
    class Reader:
        count = 0
        def json(self, reference):
            self.count += 1
            path = Path(reference["path"])
            size, epoch, index = int(path.parts[-3]), int(path.parts[-2]), int(path.stem)
            indices = list(plans[epoch][index])
            if mutation == "order" and epoch == 1 and index == 0:
                indices[0], indices[1] = indices[1], indices[0]
            binding = ("a" if size == 640 else "b") * 64
            if mutation == "loader_binding" and size == 896 and epoch == 1 and index == 0:
                binding = "f" * 64
            samples = [{
                "index": value, "epoch": epoch, "order_position": index * 16 + offset,
                "binding_sha256": binding, "image_id": value + 1,
                "stable_image_id": f"{value:064x}", "image_sha256": f"{value+1:064x}",
                "relative_path": f"train/images/{value}.jpg",
                "source_path": f"/cpu-fixture/train_core/images/{value}.jpg",
                "augmentation_seed": epoch * 4869 + value,
            } for offset, value in enumerate(indices)]
            if mutation == "last_augmentation" and size == 896 and epoch == 30 and index == 303:
                samples[0]["augmentation_seed"] += 1
            receipt = {"epoch": epoch, "logical_batch_index": index, "binding_sha256": binding,
                       "augmented_sha256": str(size)[0] * 64, "samples": samples}
            receipt["receipt_sha256"] = canonical_sha256(receipt)
            update = (epoch - 1) * 304 + index + 1
            return {
                "input": receipt, "run_binding_sha256": str(size)[0] * 64,
                "window": {"epoch": epoch, "optimizer_updates": update, "microsteps": update * 2,
                           "logical_batch_size": 16, "phase": 0, "loss": 20. + size / 1000},
                "state": {"clocks": {"engine": {"epoch_active": True, "optimizer_updates": update,
                                              "microsteps": update * 2}}},
                "synchronized_train_window_seconds": 1. if size == 640 else 2.,
                "logical_input_wait_seconds": .1,
                "memory": {"peak_allocated_bytes": size * 1024, "peak_reserved_bytes": size * 2048},
            }
    return Reader(), sources


def test_all_thirty_epochs_and_source_identity_are_paired_without_equating_losses(monkeypatch):
    reader, sources = paired_fixture(monkeypatch)
    progress = []
    result = summary.audit_training_pair(reader, *sources, progress=lambda *a, **k: progress.append(k))
    assert reader.count == 18240 and len(progress) == 30 and progress[-1]["epoch"] == 30
    assert result["logical_windows_per_arm"] == 9120 and result["logical_samples_per_arm"] == 145920
    assert result["resources"]["640"]["synchronized_train_window_wall_seconds"] == 9120
    assert result["resources"]["896"]["synchronized_train_window_wall_seconds"] == 18240
    assert result["cross_resolution_loss_denominator_BN_DN_rng_or_parameter_equality_required"] is False


@pytest.mark.parametrize("mutation", ["last_augmentation", "order", "loader_binding"])
def test_pairing_checks_last_window_and_frozen_order(monkeypatch, mutation):
    reader, sources = paired_fixture(monkeypatch, mutation)
    with pytest.raises((summary.ReplicationSummaryError, RuntimeError, ValueError)):
        summary.audit_training_pair(reader, *sources)
    if mutation == "last_augmentation":
        assert reader.count == 18240


@dataclass
class PrimaryPoint:
    AP: float = 40.
    AP50: float = 50.
    AP75: float = 30.
    AR1: float = 5.
    AR10: float = 15.
    AR100: float = 45.
    AR500: float = 55.
    canonical_input_sha256: str = "a" * 64


def score_fixture(monkeypatch, tmp_path, *, cache_error=False):
    from sparse_rtdetr.baseline import primary_evaluator as primary
    from sparse_rtdetr.baseline import training_v2b_official_gt as gt
    from sparse_rtdetr.baseline import training_v2b_primary_cache as cache
    from sparse_rtdetr.baseline import training_v2b_resolution_diagnostics as diagnostics
    from sparse_rtdetr.baseline import training_v2b_replication_gate as gate
    point = PrimaryPoint()
    metrics = {name: getattr(point, name) for name in summary.PRIMARY_METRICS}
    coco = {name: 11. if name == "AP_small" else 20. for name in summary.COCO_METRICS}
    predictions = [row for image in range(548) for row in
                   [{"image_id": image, "category_id": 1, "bbox": [-2., 0., 1., 1.], "score": .5}] * 300]
    prediction_ref, legacy_ref = ref(tmp_path / "predictions.json"), ref(tmp_path / "legacy.json")
    class Reader:
        def json(self, reference):
            return predictions if reference == prediction_ref else metrics
    observed = []
    def combine(official, image_map, rows):
        assert rows is predictions
        observed.append(("saved_unclipped_predictions", len(rows), rows[0]["bbox"][0]))
        return "CPU_fixture_input"
    monkeypatch.setattr(gt, "with_predictions", combine)
    monkeypatch.setattr(primary, "evaluate_primary_v1", lambda value, contract: point)
    monkeypatch.setattr(primary, "validate_primary_evaluator_result", lambda *a: None)
    cached = {name: metrics[name] for name in ("AP", "AP50", "AP75", "AR500")}
    if cache_error:
        cached["AP"] += .01
    monkeypatch.setattr(cache, "prepare_primary_cache", lambda *a: SimpleNamespace(
        input_sha256="a" * 64, score=lambda ids: cached, describe=lambda: {"CPU_fixture": True}))
    monkeypatch.setattr(gate, "_tensor_metrics", lambda *a: coco)
    monkeypatch.setattr(diagnostics, "prepare_coco_cache", lambda *a, **k: SimpleNamespace(score=lambda ids: coco))
    source = {
        "seed": 1, "input_size": 640, "result_reference": ref(tmp_path / "worker.json"),
        "result": {name: ref(tmp_path / (name + ".json")) for name in (
            "contract_reference", "binding_reference", "initial_state_reference", "final_state_reference")},
        "checkpoint_reference": ref(tmp_path / "checkpoint-epoch-030.pt"),
        "completion_reference": None, "completion": None, "native": {"CPU_fixture": True},
        "prediction_role": "saved_formal_training_epoch_30_development_predictions",
        "evaluation": {"result_artifact": ref(tmp_path / "evaluation.json")},
        "prediction_result": {
            "prediction_artifact": prediction_ref,
            "primary_legacy_formal_gt": {"metrics": metrics, "result_artifact": legacy_ref},
            "coco_secondary": {"CPU_fixture": True},
        },
    }
    official = {"official": "CPU_fixture_official", "image_map": {i: str(i) for i in range(548)},
                "image_ids": tuple(range(548)), "primary_contract": {}, "coco": {},
                "binding": {"binding_sha256": "b" * 64}}
    return Reader(), source, official, predictions, observed


def test_point_scoring_uses_full_primary_and_cache_and_preserves_separate_coco(monkeypatch, tmp_path):
    reader, source, official, predictions, observed = score_fixture(monkeypatch, tmp_path)
    result = summary._score_endpoint(reader, source, official, tmp_path)
    assert result["uniform_official_primary"]["metrics"]["AP"] == 40
    assert result["converted_COCO_reported_separately"]["metrics"]["AP_small"] == 11
    assert result["uniform_official_primary"]["AP_small_defined_by_this_primary_contract"] is False
    assert observed == [("saved_unclipped_predictions", 164400, -2.)]
    assert result["uniform_official_primary"]["cache_point_check"]["status"] == "PASS"


def test_primary_cache_point_disagreement_is_not_hidden_by_statistics(monkeypatch, tmp_path):
    reader, source, official, predictions, observed = score_fixture(monkeypatch, tmp_path, cache_error=True)
    with pytest.raises(summary.ReplicationSummaryError, match="point discrepancy"):
        summary._score_endpoint(reader, source, official, tmp_path)


def test_missing_prediction_is_rejected_before_full_scoring(monkeypatch, tmp_path):
    reader, source, official, predictions, observed = score_fixture(monkeypatch, tmp_path)
    predictions.pop()
    with pytest.raises(summary.ReplicationSummaryError, match="300 predictions"):
        summary._score_endpoint(reader, source, official, tmp_path)
    assert not observed




def official_context_fixture(monkeypatch, *, mutation=None):
    from sparse_rtdetr.baseline import training_v2b_official_gt as gt
    from sparse_rtdetr.baseline import training_v2b_replication_gate as gate
    development = {
        "repo_root": "/cpu-producer",
        "annotation": ref(Path("/cpu-evidence/development_coco.json")),
        "manifest": ref(Path("/cpu-evidence/development_manifest.json")),
        "image_count": 548, "annotation_count": 38759,
        "image_order_sha256": "a" * 64, "image_manifest_sha256": "b" * 64,
        "source": {"primary_contract": {"CPU_fixture": True}},
    }
    producer_adapter = ref(Path("/cpu-producer/official_gt.py"))
    consumer_adapter = ref(Path("/cpu-consumer/official_gt.py"))
    original = {
        "raw_annotation_root": None, "raw_bytes_independently_verified": False,
        "lineage": ref(Path("/cpu-evidence/development_lineage.jsonl")),
        "adapter_source": producer_adapter, "ground_truth_sha256": "c" * 64,
    }
    original["binding_sha256"] = canonical_sha256(original)
    consumer = {**development, "repo_root": "/cpu-consumer"}
    provenance = {"producer_repo_root": "/cpu-producer", "consumer_repo_root": "/cpu-consumer",
                  "changed_fields": ["repo_root", "binding_sha256"]}
    observed = []
    def rebind(producer, *, reader=None):
        assert producer is development and reader is not None
        observed.append("strict_shared_rebinding")
        return consumer, provenance
    def make(bound, lineage, raw_root):
        assert bound is consumer and lineage == original["lineage"] and raw_root is None
        observed.append("official_rebuild")
        rebuilt = {**original, "adapter_source": dict(consumer_adapter)}
        if mutation == "adapter_bytes":
            rebuilt["adapter_source"]["sha256"] = "d" * 64
        elif mutation == "truth":
            rebuilt["ground_truth_sha256"] = "e" * 64
        rebuilt["binding_sha256"] = canonical_sha256({k: v for k, v in rebuilt.items() if k != "binding_sha256"})
        return rebuilt, "CPU_fixture_official", {i: str(i) for i in range(548)}, []
    monkeypatch.setattr(gate, "rebind_development_for_consumer", rebind)
    monkeypatch.setattr(gt, "_make_binding", make)
    class Reader:
        def read(self, *a, **k):
            return b"CPU_fixture"
        def json(self, *a, **k):
            return {"CPU_fixture_coco": True}
    if mutation == "raw":
        original["raw_annotation_root"] = "/cpu-evidence/raw"
    context = {
        "seed0_campaign": {"development_bindings": {"640": development},
                           "official_gt_bindings": {"640": original}},
        "sources": {(0, 640): {"contract": {"development_binding": development}}},
    }
    return Reader(), context, development, original, provenance, observed


def test_official_context_uses_strict_shared_rebinding_and_preserves_producer_identity(monkeypatch):
    reader, context, development, original, provenance, observed = official_context_fixture(monkeypatch)
    saved_development = copy.deepcopy(development)
    result = summary._official_context(reader, context)
    assert observed == ["strict_shared_rebinding", "official_rebuild"]
    assert development == saved_development and result["binding"] == original
    assert result["development_rebinding"] == provenance
    assert result["adapter_current"]["path"] == "/cpu-consumer/official_gt.py"


@pytest.mark.parametrize("mutation", ["adapter_bytes", "truth", "raw"])
def test_only_provenance_paths_can_change_during_official_gt_reconstruction(monkeypatch, mutation):
    reader, context, development, original, provenance, observed = official_context_fixture(monkeypatch, mutation=mutation)
    with pytest.raises(summary.ReplicationSummaryError):
        summary._official_context(reader, context)
    if mutation == "raw":
        assert observed == []


def invocation_fixture(monkeypatch, tmp_path, *, failed_stage=None):
    from sparse_rtdetr.baseline import training_v2b_replication_worker as worker
    freeze_ref = write_exclusive_json(tmp_path / "freeze.json", {"cpu_fixture": True})
    reader, entries, validated = summary._Reader(), [], []
    def artifact(directory, name, payload=None):
        return write_exclusive_json(directory / name, {"cpu_fixture": True} if payload is None else payload)
    for cell in summary.CELLS:
        for stage in worker.STAGES:
            directory = tmp_path / cell / stage
            directory.mkdir(parents=True)
            contract = {"freeze_reference": freeze_ref, "cell_id": cell, "stage": stage}
            contract_ref = artifact(directory, "contract.json", contract)
            result = {"contract_reference": contract_ref, "binding_reference": artifact(directory, "binding.json"),
                      "cell_id": cell, "stage": stage, "status": "PASS"}
            result_ref = artifact(directory, "worker-result.json", result)
            monitor = {key: artifact(directory, key + ".json") for key in summary.NATIVE_LEAVES}
            monitor["final_report_reference"] = artifact(directory, "monitor-final.json")
            completion = {
                "status": "STOP_NO_RETRY" if stage == failed_stage else "PASS",
                "launch_reference": artifact(directory, "launch.json"), "monitor": monitor,
                "capacity_memory_audit_reference": artifact(directory, "memory-audit.json") if stage == "capacity" else None,
                "cell_id": cell, "stage": stage,
            }
            completion_ref = artifact(directory, "completion.json", completion)
            entries.append({"cell_id": cell, "stage": stage, "worker_result_reference": result_ref,
                            "completion_reference": completion_ref})
    monkeypatch.setattr(worker, "validate_replication_worker_contract", lambda *a, **k: None)
    monkeypatch.setattr(worker, "validate_replication_freeze", lambda *a, **k: None)
    def successful(reference, *, cell_id, stage, contract, freeze):
        result = reader.json(reference)
        assert (result["cell_id"], result["stage"]) == (cell_id, stage)
        return result, contract
    def completion(reference, result_reference, result, source_contract, contract, freeze):
        document = reader.json(reference)
        if document["status"] != "PASS":
            raise summary.ReplicationSummaryError("prerequisite completion failed")
        validated.append((document["cell_id"], document["stage"]))
        return document
    monkeypatch.setattr(worker, "_successful_cell_source", successful)
    monkeypatch.setattr(worker, "_verify_completion", completion)
    return reader, entries, freeze_ref, {}, validated


def test_all_sixteen_invocations_consume_existing_completion_validators(monkeypatch, tmp_path):
    from sparse_rtdetr.baseline import training_v2b_replication_worker as worker
    reader, entries, freeze_ref, freeze, validated = invocation_fixture(monkeypatch, tmp_path)
    checked = summary._completed_invocations(reader, entries, freeze_ref, freeze)
    assert set(checked) == {(cell, stage) for cell in summary.CELLS for stage in worker.STAGES}
    assert validated == [(cell, stage) for cell in summary.CELLS for stage in worker.STAGES]
    paths = {row["path"] for row in reader.inventory()}
    assert str(tmp_path / "s2_r896" / "capacity" / "memory-audit.json") in paths
    assert all(row["completion_reference"]["path"] in paths for row in entries)


@pytest.mark.parametrize("failed_stage", ["capacity", "smoke", "smoke_replay"])
def test_nonformal_failure_cannot_be_hidden_by_pass_campaign_terminal(monkeypatch, tmp_path, failed_stage):
    reader, entries, freeze_ref, freeze, validated = invocation_fixture(
        monkeypatch, tmp_path, failed_stage=failed_stage)
    with pytest.raises(summary.ReplicationSummaryError, match="completion failed"):
        summary._completed_invocations(reader, entries, freeze_ref, freeze)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "other_stage"])
def test_completed_invocation_set_cannot_omit_or_replace_a_stage(monkeypatch, tmp_path, mutation):
    reader, entries, freeze_ref, freeze, validated = invocation_fixture(monkeypatch, tmp_path)
    if mutation == "missing":
        entries.pop()
    elif mutation == "duplicate":
        entries[-1] = entries[0]
    else:
        entries[-1] = {**entries[-1], "stage": "epoch60"}
    with pytest.raises(summary.ReplicationSummaryError):
        summary._completed_invocations(reader, entries, freeze_ref, freeze)
    assert validated == []


def test_builder_writes_one_immutable_summary_and_never_authorizes_experiments(monkeypatch, tmp_path):
    from sparse_rtdetr.baseline import training_v2b_campaign as campaign
    root = tmp_path / "campaign"; root.mkdir()
    sources = {(seed, size): {"seed": seed, "input_size": size,
                              "contract": {"output_dir": str(root / f"s{seed}-r{size}")}}
               for seed in summary.SEEDS for size in summary.SIZES}
    context = {
        "freeze": {"output_root": str(root), "campaign_id": "cpu_fixture",
                   "qualification_reference": ref(root / "qualification.json"), "source": {"CPU_fixture": True}},
        "freeze_reference": ref(root / "freeze.json"), "campaign_result_reference": ref(root / "terminal.json"),
        "qualification": {"analysis_reference": ref(root / "analysis.json")}, "sources": sources,
    }
    monkeypatch.setattr(summary, "_load_context", lambda *a: context)
    monkeypatch.setattr(summary, "audit_training_pair", lambda *a, **k: {
        "status": "PASS", "resources": {"640": {"CPU_fixture": True}, "896": {"CPU_fixture": True}}})
    monkeypatch.setattr(summary, "_official_context", lambda *a: {
        "binding": {"raw_bytes_independently_verified": False}, "adapter_current": ref(root / "adapter.py"),
        "development_rebinding": {"CPU_fixture": True}})
    monkeypatch.setattr(summary, "_score_endpoint", lambda reader, source, *a: endpoint(source["seed"], source["input_size"]))
    monkeypatch.setattr(campaign, "validate_frozen_source", lambda *a: None)
    output = root / "fixed-epoch-summary"
    result = summary.build_replication_summary(campaign_result_reference=context["campaign_result_reference"],
                                               output_dir=str(output))
    saved = json.loads((output / "summary.json").read_text())
    assert saved == result["summary"] and result["summary_reference"] == file_reference(output / "summary.json")
    assert saved["status"] == "PASS" and len(saved["endpoints"]) == 6
    assert saved["official_GT_provenance"]["raw_bytes_independently_verified"] is False
    assert all(value is False for value in saved["scope"].values())
    with pytest.raises(FileExistsError):
        summary.build_replication_summary(campaign_result_reference=context["campaign_result_reference"],
                                          output_dir=str(output))


@pytest.mark.parametrize("case", ["bad_sha", "forbidden", "epoch_override", "visible_cuda"])
def test_cli_rejects_scope_overrides_before_research_imports(tmp_path, case):
    cli = Path(summary.__file__).resolve().parents[3] / "tools/summarize_training_v2b_replication.py"
    path = tmp_path / "campaign-result.json"; path.write_text("{}")
    args = ["--campaign-result", str(path), "--campaign-result-sha256", file_reference(path)["sha256"],
            "--output-dir", str(tmp_path / "new-output")]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    if case == "bad_sha":
        args[3] = "bad"
    elif case == "forbidden":
        args[1] = str(tmp_path / "confirmatory" / "never-open.json")
    elif case == "epoch_override":
        args += ["--epoch", "60"]
    else:
        env["CUDA_VISIBLE_DEVICES"] = "0"
    wrapper = (
        "import runpy,sys\n"
        "class Block:\n"
        " def find_spec(self,name,path=None,target=None):\n"
        "  if name=='torch' or name.startswith('sparse_rtdetr'):\n"
        "   raise AssertionError('UNAUTHORIZED_IMPORT')\n"
        "sys.meta_path.insert(0,Block())\n"
        "sys.argv=sys.argv[1:]\n"
        "runpy.run_path(sys.argv[0],run_name='__main__')\n"
    )
    result = subprocess.run([sys.executable, "-B", "-c", wrapper, str(cli), *args],
                            env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 2
    assert "UNAUTHORIZED_IMPORT" not in result.stdout + result.stderr
