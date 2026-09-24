"""Development-only evaluation tests on synthetic JPEGs and real vendor math."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

from sparse_rtdetr.baseline import training_v2b_development as development
from sparse_rtdetr.baseline.postprocessor import VisDronePostProcessor
from sparse_rtdetr.baseline.training_v2b import V2BConfig, build_v2b_components
from sparse_rtdetr.baseline.training_v2b_device import prepare_runtime
from sparse_rtdetr.baseline.training_v2b_engine import AccumulationEngine


ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    raw = json.dumps(value, sort_keys=True).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def files(tmp_path):
    image_root = tmp_path / "approved_development_images"
    image_root.mkdir()
    images, annotations, records = [], [], []
    for index in range(5):
        image_id = index + 1
        name = f"0000001_{index:05d}_d_{index:07d}.jpg"
        width, height = 200, 100
        pixels = np.zeros((height, width, 3), dtype=np.uint8)
        pixels[:, :, index % 3] = 20 + index * 20
        path = image_root / name
        Image.fromarray(pixels).save(path, format="JPEG", quality=90)
        raw = path.read_bytes()
        stable = hashlib.sha256(f"synthetic-development-{image_id}".encode()).hexdigest()
        logical = f"val/images/{name}"
        images.append({"id": image_id, "file_name": logical, "split": "val",
                       "stable_image_id": stable, "width": width, "height": height})
        annotations.append({
            "id": image_id, "image_id": image_id, "category_id": 1,
            "bbox": [40.0, 10.0, 20.0, 20.0], "area": 400.0, "iscrowd": 0,
            "stable_annotation_id": hashlib.sha256(f"gt-{image_id}".encode()).hexdigest(),
        })
        records.append({"coco_image_id": image_id, "relative_path": logical,
                        "stable_image_id": stable, "split": "val",
                        "width": width, "height": height,
                        "image_sha256": hashlib.sha256(raw).hexdigest(),
                        "image_size_bytes": len(raw)})
    coco = {"images": images, "annotations": annotations,
            "categories": [{"id": i, "name": f"category_{i}"} for i in range(1, 11)]}
    annotation = tmp_path / "development_coco.json"
    manifest = tmp_path / "development_manifest.json"
    return {
        "repo_root": ROOT, "image_root": image_root,
        "annotation_file": annotation, "annotation_sha256": write_json(annotation, coco),
        "manifest_file": manifest, "manifest_sha256": write_json(manifest, {"records": records}),
    }


class Predictor(nn.Module):
    """CPU fixture with real BN modes and deliberately consumed random streams."""

    def __init__(self, *, correct=True, input_size=640):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))
        self.bn = nn.BatchNorm2d(3)
        self.drop = nn.Dropout(0.4)
        self.correct = correct
        self.calls = []
        self.fail = False
        # Explicit synthetic geometry exercises the production cache guard.
        # The predictor still supplies fixed boxes independently of these tensors.
        self.encoder = nn.Module()
        self.encoder.eval_spatial_size = [input_size, input_size]
        self.encoder.feat_strides = [8, 16, 32]
        self.encoder.hidden_dim = 256
        self.encoder.use_encoder_idx = [2]
        self.encoder.pos_embed2 = torch.zeros(1, (input_size // 32) ** 2, 256)
        self.decoder = nn.Module()
        self.decoder.eval_spatial_size = [input_size, input_size]
        self.decoder.feat_strides = [8, 16, 32]
        self.decoder.hidden_dim = 256
        self.decoder.num_levels = 3
        count = sum((input_size // stride) ** 2 for stride in (8, 16, 32))
        self.decoder.register_buffer("anchors", torch.zeros(1, count, 4))
        self.decoder.register_buffer("valid_mask", torch.ones(1, count, 1, dtype=torch.bool))

    def forward(self, samples):
        self.calls.append((list(samples.shape), samples.dtype, self.training,
                           random.random(), float(np.random.random()),
                           float(torch.rand(()))))
        pixels = self.bn(samples)
        assert not self.training and not self.bn.training
        assert not torch.is_grad_enabled()
        assert pixels.shape == samples.shape
        if self.fail:
            raise RuntimeError("synthetic forward failure")
        logits = torch.full((len(samples), 300, 10), -20.0, device=samples.device)
        logits[:, 0, 0] = 8.0
        center = [0.25, 0.2, 0.1, 0.2] if self.correct else [0.8, 0.8, 0.1, 0.1]
        boxes = torch.tensor(center, device=samples.device).expand(len(samples), 300, 4).clone()
        return {"pred_logits": logits, "pred_boxes": boxes}


def set_clock(engine, epoch=10, *, active=False):
    # These are synthetic controller clocks, not evidence of real training.
    state = engine.state_dict()
    state.update(epoch=epoch, epoch_active=active, epoch_start_optimizer_updates=epoch - 1,
                 optimizer_updates=epoch, microsteps=epoch, phase=0, failed=False)
    engine.load_state_dict(state)


@pytest.fixture
def components():
    runtime = prepare_runtime(device="cpu", seed=321)
    model = Predictor(correct=False)
    ema = SimpleNamespace(module=Predictor(correct=True), updates=10,
                          update=lambda *args: pytest.fail("evaluation must not update EMA"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    engine = AccumulationEngine(model, nn.MSELoss(), optimizer, physical_batch_size=1,
                                accumulation_steps=1, device="cpu", amp_dtype="bfloat16",
                                expected_input_size=640, ema=ema)
    set_clock(engine)
    postprocessor = VisDronePostProcessor(vendor_root=ROOT / "vendor/rtdetrv2_pytorch")._vendor
    model.train()
    model.bn.eval()
    ema.module.train()
    ema.module.drop.eval()
    return {"model": model, "ema": ema, "engine": engine,
            "postprocessor": postprocessor, "runtime": runtime}


def rng_snapshot():
    return random.getstate(), np.random.get_state(), torch.random.default_generator.get_state().clone()


def assert_rng_equal(first, second):
    assert first[0] == second[0]
    assert first[1][0] == second[1][0]
    assert np.array_equal(first[1][1], second[1][1])
    assert first[1][2:] == second[1][2:]
    assert torch.equal(first[2], second[2])


def mode_snapshot(components):
    return [module.training for key in ("model", "postprocessor")
            for module in components[key].modules()] + [
                module.training for module in components["ema"].module.modules()
            ]


def output_dir(tmp_path, name="evaluation"):
    path = tmp_path / name
    path.mkdir()
    return path


def test_binding_only_reads_explicit_development_metadata_and_source(files, monkeypatch):
    original = Path.read_bytes
    reads = []

    def record(path):
        reads.append(path)
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", record)
    binding = development.build_development_binding(**files)
    assert binding["image_count"] == 5
    assert binding["annotation_count"] == 5
    assert binding["role"] == "development"
    assert binding["policy"]["model_weights"] == "ema"
    assert binding["policy"]["input_size"] == [640, 640]
    assert binding["policy"]["secondary_max_dets"] == [1, 10, 100]
    assert binding["policy"]["ignore_regions_loaded"] is False
    assert not any(path.suffix == ".jpg" for path in reads)
    assert not any("lineage" in path.name or "ignore_regions" in path.name for path in reads)
    assert not any("confirmatory" in path.name for path in reads)
    assert development.validate_development_binding(binding) == binding
    binding["policy"]["batch_size"] = 8
    with pytest.raises(development.DevelopmentEvaluationError, match="digest"):
        development.validate_development_binding(binding)


@pytest.mark.parametrize("part", ["confirmatory", "test", "train_core", "VisDrone2019-DET-test-dev"])
def test_other_role_paths_are_rejected_before_read(files, tmp_path, monkeypatch, part):
    bad = dict(files, annotation_file=tmp_path / part / "development_coco.json")
    original = Path.read_bytes

    def prohibit(path):
        assert part not in path.parts, "forbidden input was read"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", prohibit)
    with pytest.raises(development.DevelopmentEvaluationError, match="data role"):
        development.build_development_binding(**bad)


@pytest.mark.parametrize("mutation", [
    "wrong_split", "wrong_filename", "duplicate_image", "missing_manifest_member",
    "wrong_manifest_geometry", "category_zero", "negative_box", "wrong_area",
])
def test_metadata_membership_geometry_and_category_drift(files, mutation):
    coco = json.loads(files["annotation_file"].read_text())
    manifest = json.loads(files["manifest_file"].read_text())
    if mutation == "wrong_split":
        coco["images"][0]["split"] = "train"
    elif mutation == "wrong_filename":
        coco["images"][0]["file_name"] = "test/images/forbidden.jpg"
    elif mutation == "duplicate_image":
        coco["images"].append(copy.deepcopy(coco["images"][0]))
    elif mutation == "missing_manifest_member":
        manifest["records"].pop()
    elif mutation == "wrong_manifest_geometry":
        manifest["records"][0]["width"] += 1
    elif mutation == "category_zero":
        coco["annotations"][0]["category_id"] = 0
    elif mutation == "negative_box":
        coco["annotations"][0]["bbox"][2] = -1
    elif mutation == "wrong_area":
        coco["annotations"][0]["area"] = 800
    candidate = dict(files)
    candidate["annotation_sha256"] = write_json(candidate["annotation_file"], coco)
    candidate["manifest_sha256"] = write_json(candidate["manifest_file"], manifest)
    with pytest.raises(development.DevelopmentEvaluationError):
        development.build_development_binding(**candidate)


def test_success_uses_ema_original_geometry_all_axes_and_preserves_training(files, components, tmp_path, monkeypatch):
    binding = development.build_development_binding(**files)
    before_rng, before_modes = rng_snapshot(), mode_snapshot(components)
    before_engine = copy.deepcopy(components["engine"].state_dict())
    gates = []

    def gate():
        gates.append(len(gates))
        random.random()
        np.random.random()
        torch.rand(())

    def no_cuda(*args, **kwargs):
        raise AssertionError("CPU evaluation touched a CUDA API")

    for name in ("is_available", "is_initialized", "get_rng_state", "set_rng_state", "init"):
        monkeypatch.setattr(torch.cuda, name, no_cuda)
    result = development.evaluate_development(
        components, data_binding=binding, output_dir=output_dir(tmp_path),
        logged_epoch=10, hardware_gate=gate,
    )
    assert components["model"].calls == []
    assert [row[0][0] for row in components["ema"].module.calls] == [4, 1]
    assert all(row[1] == torch.float32 for row in components["ema"].module.calls)
    assert mode_snapshot(components) == before_modes
    assert_rng_equal(before_rng, rng_snapshot())
    assert components["engine"].state_dict() == before_engine
    assert components["ema"].updates == 10
    assert result["data"]["evaluated_image_count"] == 5
    assert result["data"]["observed_images_sha256"] == binding["image_manifest_sha256"]
    assert result["full_development_evaluation"] is True
    assert result["checkpoint_selection_performed"] is False
    assert result["protocol_certification_claimed"] is False
    metrics = result["coco_secondary"]["metrics"]
    assert len(metrics) == 12
    for name in ("AP", "AP50", "AP75", "AP_small", "AR1", "AR10", "AR100", "AR_small"):
        assert metrics[name]["value"] == pytest.approx(100.0)
    # Original 20x20 boxes are small; their 640-resized area would be medium.
    assert metrics["AP_medium"]["value"] is None
    assert metrics["AP_medium"]["available"] is False
    assert result["coco_secondary"]["axes"]["area_coordinate_space"] == "original_image_pixels"
    assert result["primary_legacy_formal_gt"]["metrics"]["AP"] == pytest.approx(100.0)
    assert result["primary_legacy_formal_gt"]["ignore_regions_loaded"] is False
    assert result["primary_legacy_formal_gt"]["full_official_ignore_compliance_claimed"] is False
    predictions = json.loads(Path(result["prediction_artifact"]["path"]).read_text())
    assert len(predictions) == 5 * 300
    assert predictions[0]["category_id"] == 1
    assert predictions[0]["bbox"] == pytest.approx([40, 10, 20, 20])
    tensor_ref = result["coco_secondary"]["tensor_artifact"]
    raw = Path(tensor_ref["path"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == tensor_ref["sha256"]
    with np.load(tensor_ref["path"], allow_pickle=False) as arrays:
        assert arrays["precision"].shape == (10, 101, 10, 4, 3)
        assert arrays["recall"].shape == (10, 10, 4, 3)
    assert [row["phase"] for row in result["native_gate_callbacks"]].count("before_batch") == 2
    assert [row["phase"] for row in result["native_gate_callbacks"]].count("after_batch") == 2
    assert len(gates) == len(result["native_gate_callbacks"])


def test_preview_is_exactly_four_images_and_never_full_metric(files, components, tmp_path):
    binding = development.build_development_binding(**files)
    set_clock(components["engine"], 1, active=True)
    result = development.preview_development(
        components, data_binding=binding, output_dir=output_dir(tmp_path),
        hardware_gate=lambda: None,
    )
    assert result["scope"] == "development_preview_only"
    assert result["logged_epoch"] is None
    assert result["data"]["evaluated_image_count"] == 4
    assert result["data"]["bound_image_count"] == 5
    assert result["full_development_evaluation"] is False
    assert result["scientific_comparison_eligible"] is False
    assert [row["image_id"] for row in result["data"]["image_receipts"]] == [1, 2, 3, 4]
    assert components["engine"].epoch_active is True


@pytest.mark.parametrize("failure", ["gate_first", "gate_after_forward", "model"])
def test_failure_restores_rng_modes_and_publishes_no_success(files, components, tmp_path, failure):
    binding = development.build_development_binding(**files)
    directory = output_dir(tmp_path)
    before_rng, before_modes = rng_snapshot(), mode_snapshot(components)
    calls = []
    components["ema"].module.fail = failure == "model"

    def gate():
        calls.append(None)
        random.random()
        np.random.random()
        torch.rand(())
        if (failure == "gate_first" and len(calls) == 1
                or failure == "gate_after_forward" and len(calls) == 3):
            raise RuntimeError("synthetic live gate stopped")

    with pytest.raises(RuntimeError, match="synthetic"):
        development.evaluate_development(components, data_binding=binding,
                                         output_dir=directory, logged_epoch=10,
                                         hardware_gate=gate)
    assert_rng_equal(before_rng, rng_snapshot())
    assert mode_snapshot(components) == before_modes
    assert not list(directory.iterdir())


@pytest.mark.parametrize("epoch,active", [(9, False), (10, True), (20, False), (30, False)])
def test_only_the_live_completed_fixed_epoch_is_allowed(files, components, tmp_path, epoch, active):
    binding = development.build_development_binding(**files)
    if active:
        set_clock(components["engine"], 10, active=True)
    with pytest.raises(development.DevelopmentEvaluationError, match="completed epoch"):
        development.evaluate_development(components, data_binding=binding,
                                         output_dir=output_dir(tmp_path), logged_epoch=epoch,
                                         hardware_gate=lambda: None)
    assert components["ema"].module.calls == []


def test_metadata_and_image_drift_fail_before_prediction_publication(files, components, tmp_path):
    binding = development.build_development_binding(**files)
    raw = files["annotation_file"].read_bytes()
    files["annotation_file"].write_bytes(raw + b" ")
    with pytest.raises(development.DevelopmentEvaluationError, match="SHA-256"):
        development.validate_development_binding(binding)
    files["annotation_file"].write_bytes(raw)
    first_image = files["image_root"] / "0000001_00000_d_0000000.jpg"
    first_image.write_bytes(first_image.read_bytes() + b"changed")
    directory = output_dir(tmp_path)
    with pytest.raises(development.DevelopmentEvaluationError, match="SHA-256/size"):
        development.evaluate_development(components, data_binding=binding,
                                         output_dir=directory, logged_epoch=10,
                                         hardware_gate=lambda: None)
    assert not list(directory.iterdir())
    assert components["ema"].module.calls == []


def test_image_symlink_escape_is_rejected_without_reading_target(files, components, tmp_path, monkeypatch):
    binding = development.build_development_binding(**files)
    first_image = files["image_root"] / "0000001_00000_d_0000000.jpg"
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(first_image.read_bytes())
    first_image.unlink()
    first_image.symlink_to(outside)
    original = Path.read_bytes

    def read(path):
        assert path != outside, "unbound image target was read"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    with pytest.raises(development.DevelopmentEvaluationError, match="escapes"):
        development.evaluate_development(components, data_binding=binding,
                                         output_dir=output_dir(tmp_path), logged_epoch=10,
                                         hardware_gate=lambda: None)


def test_existing_artifact_and_missing_ema_never_fall_back(files, components, tmp_path):
    binding = development.build_development_binding(**files)
    directory = output_dir(tmp_path)
    (directory / "predictions.json").write_text("existing")
    with pytest.raises(development.DevelopmentEvaluationError, match="overwrite"):
        development.evaluate_development(components, data_binding=binding,
                                         output_dir=directory, logged_epoch=10,
                                         hardware_gate=lambda: None)
    assert (directory / "predictions.json").read_text() == "existing"
    assert components["ema"].module.calls == []
    broken = dict(components, ema=None)
    with pytest.raises(development.DevelopmentEvaluationError):
        development.evaluate_development(broken, data_binding=binding,
                                         output_dir=output_dir(tmp_path, "missing-ema"),
                                         logged_epoch=10, hardware_gate=lambda: None)


def coco_result(coco, predictions):
    from faster_coco_eval import COCO
    evaluator_type = development._vendor_tools(ROOT)[3]
    evaluator = evaluator_type(COCO(coco), ["bbox"])
    evaluator.world_size = 1
    evaluator.update(predictions)
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    return development._coco_metrics(evaluator.coco_eval["bbox"])


def test_real_vendor_maxdets_truncation_is_distinct_from_300_output_topk():
    annotations = []
    boxes = []
    for index in range(120):
        box = [index * 10.0, 0.0, index * 10.0 + 5.0, 5.0]
        boxes.append(box)
        annotations.append({"id": index + 1, "image_id": 1, "category_id": 1,
                            "bbox": [box[0], 0.0, 5.0, 5.0], "area": 25.0, "iscrowd": 0})
    coco = {"images": [{"id": 1, "width": 2000, "height": 1000}],
            "annotations": annotations,
            "categories": [{"id": i, "name": str(i)} for i in range(1, 11)]}
    metrics, axes, _, _ = coco_result(coco, {
        1: {"boxes": torch.tensor(boxes), "labels": torch.ones(120, dtype=torch.int64),
            "scores": torch.linspace(0.99, 0.5, 120)},
    })
    assert axes["maxDets"] == [1, 10, 100]
    assert metrics["AR1"]["value"] == pytest.approx(100 / 120)
    assert metrics["AR10"]["value"] == pytest.approx(1000 / 120)
    assert metrics["AR100"]["value"] == pytest.approx(10000 / 120)
    assert metrics["AR_small"]["value"] < 100


def test_real_vendor_iscrowd_ignored_detection_does_not_become_false_positive():
    coco = {"images": [{"id": 1, "width": 200, "height": 100}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 10, 10],
                 "area": 100, "iscrowd": 0},
                {"id": 2, "image_id": 1, "category_id": 1, "bbox": [80, 10, 40, 40],
                 "area": 1600, "iscrowd": 1},
            ], "categories": [{"id": i, "name": str(i)} for i in range(1, 11)]}
    metrics, _, _, _ = coco_result(coco, {
        1: {"boxes": torch.tensor([[90., 20., 100., 30.], [10., 10., 20., 20.]]),
            "labels": torch.tensor([1, 1]), "scores": torch.tensor([0.99, 0.8])},
    })
    assert metrics["AP"]["value"] == pytest.approx(100.0)
    assert metrics["AP_small"]["value"] == pytest.approx(100.0)
    assert metrics["AP_medium"]["available"] is False


def test_actual_r18_640_ema_forward_and_vendor_metrics_on_synthetic_jpeg(files, tmp_path):
    # A real R18 forward is required; the completed clock is synthetic, not a
    # claim that this test trained ten epochs or used development performance.
    coco = json.loads(files["annotation_file"].read_text())
    manifest = json.loads(files["manifest_file"].read_text())
    coco["images"] = coco["images"][:1]
    coco["annotations"] = coco["annotations"][:1]
    manifest["records"] = manifest["records"][:1]
    candidate = dict(files, annotation_sha256=write_json(files["annotation_file"], coco),
                     manifest_sha256=write_json(files["manifest_file"], manifest))
    binding = development.build_development_binding(**candidate)
    components = build_v2b_components(
        V2BConfig(seed=509, input_size=640, physical_batch_size=1,
                  accumulation_steps=1, pretrained_required=False),
        repo_root=ROOT,
    )
    set_clock(components.engine)
    before_rng = rng_snapshot()
    result = development.evaluate_development(
        components, data_binding=binding, output_dir=output_dir(tmp_path),
        logged_epoch=10, hardware_gate=lambda: None,
    )
    assert_rng_equal(before_rng, rng_snapshot())
    assert result["data"]["evaluated_image_count"] == 1
    assert result["weights"]["eval_spatial_size"] == [640, 640]
    assert result["runtime"]["device"] == "cpu"
    assert result["coco_secondary"]["precision_shape"] == [10, 101, 10, 4, 3]
    predictions = json.loads(Path(result["prediction_artifact"]["path"]).read_text())
    assert len(predictions) == 300
    assert all(1 <= row["category_id"] <= 10 for row in predictions)


def test_live_gate_false_stops_before_forward(files, components, tmp_path):
    binding = development.build_development_binding(**files)
    directory = output_dir(tmp_path)
    with pytest.raises(development.DevelopmentEvaluationError, match="live gate refused"):
        development.evaluate_development(
            components, data_binding=binding, output_dir=directory, logged_epoch=10,
            hardware_gate=lambda: False,
        )
    assert not components["ema"].module.calls
    assert not list(directory.iterdir())


@pytest.mark.parametrize("state", ["learning_rate", "batchnorm_buffer"])
def test_unexpected_training_state_mutation_cannot_publish_metrics(files, components, tmp_path, state):
    binding = development.build_development_binding(**files)
    directory = output_dir(tmp_path)
    calls = []

    def gate():
        calls.append(None)
        if len(calls) == 3:
            if state == "learning_rate":
                components["engine"].optimizer.param_groups[0]["lr"] *= 2
            else:
                components["ema"].module.bn.running_mean.add_(1)

    before_rng, before_modes = rng_snapshot(), mode_snapshot(components)
    with pytest.raises(development.DevelopmentEvaluationError, match="changed model/EMA state or update clocks"):
        development.evaluate_development(
            components, data_binding=binding, output_dir=directory, logged_epoch=10,
            hardware_gate=gate,
        )
    assert_rng_equal(before_rng, rng_snapshot())
    assert mode_snapshot(components) == before_modes
    assert not list(directory.iterdir())


def test_development_geometry_admits_1024_and_keeps_square_policy():
    assert development._input_size(1024) == 1024
    assert development._policy_input_size({'input_size': [1024, 1024]}) == 1024
    for bad in (960, 1024.0, True):
        with pytest.raises(development.DevelopmentEvaluationError):
            development._input_size(bad)
    with pytest.raises(development.DevelopmentEvaluationError, match='square'):
        development._policy_input_size({'input_size': [768, 1344]})
