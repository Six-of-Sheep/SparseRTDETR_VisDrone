"""Synthetic CPU coverage for raw GT repair, geometry-only adaptation and query provenance."""

from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from sparse_rtdetr.baseline import primary_evaluator
from sparse_rtdetr.baseline import training_v2b_cross_eval as cross
from sparse_rtdetr.baseline import training_v2b_official_gt as official
from sparse_rtdetr.baseline import training_v2b_development as development
from sparse_rtdetr.baseline.training_v2b import V2BConfig, _build_vendor_objects, _configure_model_sampling
from sparse_rtdetr.baseline.training_v2b_evidence import file_reference
from sparse_rtdetr.baseline.training_v2b_geometry import snapshot_model_geometry
from sparse_rtdetr.baseline.postprocessor import VisDronePostProcessor
from sparse_rtdetr.data_protocol.parser import parse_annotation_bytes
from sparse_rtdetr.data_protocol.schema import stable_annotation_id
from test_rtdetr_baseline_training_v2b_development import files, write_json


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def raw_fixture(files, tmp_path):
    coco = json.loads(files["annotation_file"].read_text())
    manifest = json.loads(files["manifest_file"].read_text())
    raw_root = tmp_path / "raw_development"
    rows = []
    for image, record, annotation in zip(coco["images"], manifest["records"], coco["annotations"]):
        raw = b"40,10,20,20,1,1,0,0\n0,0,20,20,0,0,0,0\n80,20,20,20,0,11,0,0\n"
        parsed = parse_annotation_bytes(raw)
        relative = image["file_name"].replace("images", "annotations").replace(".jpg", ".txt")
        path = raw_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        record.update(annotation_relative_path=relative,
                      annotation_sha256=hashlib.sha256(raw).hexdigest(),
                      annotation_size_bytes=len(raw), raw_annotation_rows=3)
        for item in parsed:
            fields = item.fields
            aid = stable_annotation_id(image["stable_image_id"], item.physical_line_number,
                                       item.raw_line_sha256)
            rows.append({
                "split": "val", "source_relative_path": image["file_name"],
                "image_id": image["stable_image_id"], "annotation_id": aid,
                "raw_fields": list(item.raw_fields), "raw_line_sha256": item.raw_line_sha256,
                "physical_line_number": item.physical_line_number,
                "raw_category_id": fields.category, "bbox_xyxy": list(fields.bbox_xyxy),
                "area": fields.area, "ignore_region": False,
                "reason_code": "MATCHABLE_FORMAL_GT" if fields.score else "SCORE_ZERO_IGNORED",
            })
            if item.physical_line_number == 1:
                annotation["stable_annotation_id"] = aid
    files = dict(files, annotation_sha256=write_json(files["annotation_file"], coco),
                 manifest_sha256=write_json(files["manifest_file"], manifest))
    lineage = tmp_path / "development_lineage.jsonl"
    lineage.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return files, lineage, raw_root, rows


def bound(raw_fixture, *, raw_bytes=False, size=640):
    files, lineage, raw_root, _ = raw_fixture
    development_binding = development.build_development_binding(**files, input_size=size)
    binding = official.build_official_gt_binding(
        development_binding=development_binding, lineage_reference=file_reference(lineage),
        raw_annotation_root=str(raw_root) if raw_bytes else None)
    return binding, development_binding


def test_score_zero_class_zero_is_spatial_class_eleven_is_not(raw_fixture):
    binding, data = bound(raw_fixture)
    value, image_map, attributes = official.load_official_ground_truth(binding, data)
    assert len(value.ground_truth) == 15 and len(image_map) == 5
    assert binding["spatial_ignore_region_count"] == 5
    assert binding["unscored_category_11_count"] == 5
    assert binding["historical_ignore_region_count"] == 0
    assert binding["raw_bytes_independently_verified"] is False
    assert all(row.ignore_region == (row.category_id == 0) for row in value.ground_truth)
    assert all(not row.ignore_region for row in value.ground_truth if row.category_id == 11)
    assert len(attributes) == 15


def test_full_raw_bytes_and_lineage_are_checked_when_explicitly_supplied(raw_fixture):
    binding, data = bound(raw_fixture, raw_bytes=True)
    assert binding["raw_bytes_independently_verified"] is True
    assert len(official.load_official_ground_truth(binding, data)[0].ground_truth) == 15
    raw_root = raw_fixture[2]
    path = next(raw_root.glob("val/annotations/*.txt"))
    path.write_bytes(path.read_bytes().replace(b"40,10", b"41,10", 1))
    with pytest.raises(official.OfficialGroundTruthError, match="SHA"):
        official.load_official_ground_truth(binding, data)


def test_gt_binding_does_not_confuse_evaluation_resize_with_gt_coordinates(raw_fixture):
    first, _ = bound(raw_fixture, size=640)
    second, _ = bound(raw_fixture, size=896)
    assert first == second


def test_spatial_ignore_removes_false_positive_but_category_eleven_does_not(raw_fixture):
    binding, data = bound(raw_fixture)
    value, image_map, _ = official.load_official_ground_truth(binding, data)
    predictions = []
    for integer in image_map:
        predictions.extend([
            {"image_id": integer, "category_id": 1, "bbox": [2., 2., 5., 5.], "score": .99},
            {"image_id": integer, "category_id": 1, "bbox": [40., 10., 20., 20.], "score": .90},
        ])
    primary_binding = data["source"]["primary_contract"]
    result = primary_evaluator.evaluate_primary_v1(
        official.with_predictions(value, image_map, predictions), primary_binding)
    assert result.AP == pytest.approx(100.0)
    predictions = [{**row, "bbox": [82., 22., 5., 5.]}
                   if row["score"] == .99 else row for row in predictions]
    result = primary_evaluator.evaluate_primary_v1(
        official.with_predictions(value, image_map, predictions), primary_binding)
    assert result.AP == pytest.approx(50.0)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "category", "geometry", "score"])
def test_incomplete_or_inconsistent_raw_lineage_fails_closed(raw_fixture, mutation):
    files, lineage, _, rows = raw_fixture
    rows = copy.deepcopy(rows)
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(rows[0])
    elif mutation == "category":
        rows[0]["raw_category_id"] = 2
    elif mutation == "geometry":
        rows[0]["bbox_xyxy"][0] += 1
    else:
        rows[0]["raw_fields"][4] = "0.5"
    lineage.write_text("".join(json.dumps(row)+"\n" for row in rows))
    with pytest.raises(official.OfficialGroundTruthError):
        bound(raw_fixture)


def test_reference_accepts_only_complete_byte_schemas_and_refuses_symlinks(tmp_path):
    path = tmp_path / "authority.json"
    path.write_bytes(b'{"ok":true}\n')
    reference = file_reference(path)
    assert official.read_regular_reference(reference) == path.read_bytes()
    three = {key: reference[key] for key in ("path", "sha256", "size_bytes")}
    assert official.read_regular_reference(three) == path.read_bytes()
    with pytest.raises(official.OfficialGroundTruthError, match="scope"):
        official.read_regular_reference({**reference, "sha256_scope": "prefix"})
    with pytest.raises(official.OfficialGroundTruthError, match="schema"):
        official.read_regular_reference({**reference, "unknown": 1})
    alias = tmp_path / "alias.json"
    alias.symlink_to(path)
    with pytest.raises(OSError):
        official.read_regular_reference({**reference, "path": str(alias)})


def test_class_specific_ignored_and_duplicate_raw_rows_are_preserved(raw_fixture):
    files, lineage, _, rows = raw_fixture
    coco = json.loads(files["annotation_file"].read_text())
    manifest = json.loads(files["manifest_file"].read_text())
    first = coco["images"][0]
    template = copy.deepcopy(rows[0])
    for line, raw in ((4, b"10,40,10,10,0,2,1,2\n"), (5, b"40,10,20,20,1,1,0,0\n")):
        item = parse_annotation_bytes(raw)[0]
        row = copy.deepcopy(template)
        row.update(raw_fields=list(item.raw_fields), raw_line_sha256=item.raw_line_sha256,
                   physical_line_number=line, raw_category_id=item.fields.category,
                   bbox_xyxy=list(item.fields.bbox_xyxy), area=item.fields.area,
                   annotation_id=stable_annotation_id(first["stable_image_id"], line, item.raw_line_sha256))
        rows.append(row)
    manifest["records"][0]["raw_annotation_rows"] = 5
    files["manifest_sha256"] = write_json(files["manifest_file"], manifest)
    lineage.write_text("".join(json.dumps(row)+"\n" for row in rows))
    binding, data = bound(raw_fixture)
    value, _, attrs = official.load_official_ground_truth(binding, data)
    assert len(value.ground_truth) == 17
    assert binding["class_specific_ignored_count"] == 1
    assert sum(row.category_id == 1 and row.image_id == first["stable_image_id"]
               for row in value.ground_truth) == 2
    assert next(row for row in attrs if row["raw_category_id"] == 2)["occlusion"] == 2.0


@pytest.fixture
def geometry_objects():
    def make(size=640):
        config = V2BConfig(seed=13, input_size=size, physical_batch_size=8,
                           accumulation_steps=2, pretrained_required=False,
                           sampling_backend="deterministic_gather")
        model, criterion, postprocessor, _, ema_type, _ = _build_vendor_objects(ROOT, config)
        _configure_model_sampling(model, config)
        ema = ema_type(model, decay=.9999, warmups=2000)
        for _ in range(3):
            ema.update(model)
        return model, ema.module, postprocessor
    return make


@pytest.mark.parametrize("source,target", [(640, 640), (640, 896), (896, 640), (896, 896)])
def test_real_geometry_changes_only_caches_and_preserves_vendor_ema_history(geometry_objects, source, target):
    raw, model, _ = geometry_objects(source)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    result = cross.adapt_evaluation_geometry(
        model, source_training_size=source, evaluation_size=target,
        updates=3, decay=.9999, warmups=2000,
        expected_raw_anchors=raw.decoder.anchors, expected_raw_mask=raw.decoder.valid_mask)
    assert result["source_replay"]["source_ema_anchors_byte_exact"] is True
    assert "pos_embed2" not in model.encoder._buffers
    assert "decoder.anchors" in model.state_dict()
    for key, value in before.items():
        if key not in ("decoder.anchors", "decoder.valid_mask"):
            assert torch.equal(model.state_dict()[key], value), key
    assert snapshot_model_geometry(model, expected_input_size=target)["input_size"] == [target, target]
    if source == target:
        assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())


def test_cache_mismatch_is_not_silently_rebuilt(geometry_objects):
    raw, model, _ = geometry_objects()
    mask = model.decoder.valid_mask.expand_as(model.decoder.anchors)
    index = mask.nonzero()[0].tolist()
    model.decoder.anchors[tuple(index)] += 1.0
    with pytest.raises(cross.CrossEvaluationError, match="not byte-exact"):
        cross.adapt_evaluation_geometry(
            model, source_training_size=640, evaluation_size=896, updates=3,
            decay=.9999, warmups=2000,
            expected_raw_anchors=raw.decoder.anchors, expected_raw_mask=raw.decoder.valid_mask)
    assert model.encoder.eval_spatial_size == [640, 640]


def test_query_origin_matches_real_vendor_topk_even_with_ties_and_multiple_classes_per_query():
    wrapper = VisDronePostProcessor(vendor_root=ROOT / "vendor/rtdetrv2_pytorch").eval()
    logits = torch.full((2, 300, 10), -5.)
    logits[:, 17, :] = 5.
    boxes = torch.full((2, 300, 4), .25)
    boxes[..., 0:2] = .5
    outputs = {"pred_logits": logits, "pred_boxes": boxes}
    sizes = torch.tensor([[200, 100], [900, 600]])
    processed = wrapper(outputs, sizes)
    indices, runtime_probabilities = cross.prediction_origin_indices(outputs, sizes, processed)
    assert runtime_probabilities.dtype == torch.float32
    assert torch.equal(runtime_probabilities, logits.sigmoid())
    assert indices.shape == (2, 300)
    assert (indices[0, :10] // 10 == 17).all()
    altered = copy.deepcopy(processed)
    altered[0]["scores"][0] = 0.
    with pytest.raises(cross.CrossEvaluationError, match="provenance"):
        cross.prediction_origin_indices(outputs, sizes, altered)


@pytest.mark.parametrize("size", [True, 640.0, 800, 960, 1024, [640,640]])
def test_unapproved_sizes_fail_before_checkpoint_access(monkeypatch, size):
    monkeypatch.setattr(cross, "read_regular_reference", lambda *a: pytest.fail("read before size validation"))
    with pytest.raises(cross.CrossEvaluationError, match="integer"):
        cross.prepare_cross_eval_model(checkpoint={}, source_training_size=size,
                                       evaluation_size=640, repo_root=ROOT)


def test_raw_query_artifact_is_exclusive_and_roundtrips_without_pickle(tmp_path):
    path = tmp_path / "raw-queries.npz"
    arrays = {"pred_logits": np.zeros((1,300,10), dtype=np.float32),
              "pred_boxes": np.full((1,300,4), .5, dtype=np.float32),
              "image_ids": np.array([7], dtype=np.int64)}
    reference = cross._write_npz(path, **arrays)
    with np.load(io.BytesIO(official.read_regular_reference(reference)), allow_pickle=False) as loaded:
        assert set(loaded.files) == set(arrays)
        for key in arrays:
            assert np.array_equal(loaded[key], arrays[key])
    with pytest.raises(FileExistsError):
        cross._write_npz(path, **arrays)


def test_real_development_transforms_do_not_consume_rng(files):
    binding = development.build_development_binding(**files, input_size=896)
    dataset = development._DevelopmentDataset(binding)
    before = cross._cpu_rng()
    for image in dataset.images:
        pixels, target, receipt = dataset.item(image)
        assert list(pixels.shape) == [3,896,896]
    assert cross._rng_equal(before, cross._cpu_rng())


@pytest.fixture
def worker_contract(raw_fixture, tmp_path):
    gt, data = bound(raw_fixture)
    authorization = {"path": str(tmp_path / "authorization.json"), "sha256": "a"*64,
                     "size_bytes": 12, "sha256_scope": "complete_file_bytes"}
    gpu = "GPU-1faee6f0-1da7-4ede-2475-67a5a00274a8"
    return {
        "schema_version": 1, "kind": "v2b_development_cross_eval_worker",
        "run_id": "unit-cross-eval", "campaign_id": "unit-campaign", "cell_key": "t640_e640",
        "repo_root": str(ROOT), "output_dir": str(ROOT/"artifacts/unit-synthetic-cross-eval"),
        "source_training_size": 640, "evaluation_size": 640,
        "checkpoint": {"path": str(tmp_path/"checkpoint-epoch-030.pt"),
                       "sha256": cross._EPOCH30_IDENTITIES[640][0],
                       "size_bytes": cross._EPOCH30_IDENTITIES[640][1]},
        "development_binding": data, "official_gt_binding": gt,
        "code_files": {"unit": {"path": str(ROOT/"unit"), "sha256": "b"*64, "size_bytes": 1}},
        "policy_bundle": {"authorization_reference": authorization,
                          "gpu_uuid": gpu, "setter_mode": "external_admin_acknowledged",
                          "authorized_scope_limits_seconds": {"development_cross_eval":1800}},
        "expected_gpu_uuid": gpu, "authorization_reference": authorization,
        "cache_policy": cross.CACHE_POLICY, "evaluator_evidence_reference": authorization,
    }


def test_contract_metadata_is_detached_and_validated_before_gpu(worker_contract, monkeypatch):
    monkeypatch.setattr(cross, "read_regular_reference", lambda *a: pytest.fail("metadata-only validation read bytes"))
    result = cross.validate_worker_contract(worker_contract, verify_files=False)
    assert result == worker_contract and result is not worker_contract
    result["run_id"] = "changed"
    assert worker_contract["run_id"] == "unit-cross-eval"


@pytest.mark.parametrize("mutation", [
    "float_size", "unsupported_size", "cell_key", "cache_policy", "checkpoint_sha",
    "checkpoint_role", "metadata_size", "scope", "setter", "authorization", "gt_membership",
])
def test_changed_cell_contract_fails_before_any_input_access(worker_contract, monkeypatch, mutation):
    value = copy.deepcopy(worker_contract)
    if mutation == "float_size":
        value["evaluation_size"] = 640.0
    elif mutation == "unsupported_size":
        value["evaluation_size"] = 960
    elif mutation == "cell_key":
        value["cell_key"] = "t896_e640"
    elif mutation == "cache_policy":
        value["cache_policy"] = "canonical_for_off_diagonal_only"
    elif mutation == "checkpoint_sha":
        value["checkpoint"]["sha256"] = "c"*64
    elif mutation == "checkpoint_role":
        value["checkpoint"]["path"] = "/not-authorized/confirmatory/checkpoint-epoch-030.pt"
    elif mutation == "metadata_size":
        value["evaluation_size"], value["cell_key"] = 896, "t640_e896"
    elif mutation == "scope":
        value["policy_bundle"]["authorized_scope_limits_seconds"] = {"paired_smoke":1800}
    elif mutation == "setter":
        value["policy_bundle"]["setter_mode"] = "direct"
    elif mutation == "authorization":
        value["authorization_reference"] = {**value["authorization_reference"], "sha256":"d"*64}
    else:
        value["official_gt_binding"]["manifest"] = {**value["official_gt_binding"]["manifest"],"sha256":"e"*64}
    monkeypatch.setattr(cross, "read_regular_reference", lambda *a: pytest.fail("invalid metadata read bytes"))
    with pytest.raises(cross.CrossEvaluationError):
        cross.validate_worker_contract(value, verify_files=True)
