"""CPU-only resolution wiring and capacity boundaries on synthetic inputs."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sparse_rtdetr.baseline import training_v2b_development as development
from sparse_rtdetr.baseline.config import load_isolated_vendor_config_dict
from sparse_rtdetr.baseline.postprocessor import VisDronePostProcessor
from sparse_rtdetr.baseline.training_v2b import V2BConfig, build_v2b_components
from sparse_rtdetr.baseline.training_v2b_device import prepare_runtime
from sparse_rtdetr.baseline.training_v2b_engine import AccumulationEngine
from sparse_rtdetr.baseline.training_v2b_geometry import (
    ModelGeometryError, snapshot_model_geometry,
)

from test_rtdetr_baseline_training_v2b_development import (
    Predictor, assert_rng_equal, files, mode_snapshot, output_dir, rng_snapshot,
    set_clock, write_json,
)


ROOT = Path(__file__).resolve().parents[1]
# Canonical _policy() from frozen commit 858cf2b25cad8fc5e2efed60925c0ee06050dcaa.
FROZEN_640_POLICY_SHA256 = "26e39af4ac70f8f7a42d7a22dead2c58f3cb62c47ba3c2cac6f96d7f69a98cd5"


def synthetic_components(input_size=896):
    runtime = prepare_runtime(device="cpu", seed=321)
    model = Predictor(correct=False, input_size=input_size)
    ema = SimpleNamespace(module=Predictor(input_size=input_size), updates=1,
                          update=lambda *args: pytest.fail("evaluation updated EMA"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    engine = AccumulationEngine(model, nn.MSELoss(), optimizer, physical_batch_size=1,
                                accumulation_steps=1, device="cpu", amp_dtype="bfloat16",
                                expected_input_size=input_size, ema=ema)
    set_clock(engine, 1, active=True)
    postprocessor = VisDronePostProcessor(vendor_root=ROOT / "vendor/rtdetrv2_pytorch")._vendor
    model.train()
    model.bn.eval()
    ema.module.train()
    ema.module.drop.eval()
    return {"model": model, "ema": ema, "engine": engine,
            "postprocessor": postprocessor, "runtime": runtime}


def rehash(binding):
    binding["binding_sha256"] = development._digest({
        key: value for key, value in binding.items() if key != "binding_sha256"
    })
    return binding


def test_640_policy_is_the_frozen_policy_and_896_changes_only_resize():
    assert hashlib.sha256(development._canonical(development._policy())).hexdigest() == FROZEN_640_POLICY_SHA256
    assert development._policy() == development._policy(640)
    enlarged = development._policy(896)
    assert enlarged["input_size"] == enlarged["transform"][0]["size"] == [896, 896]
    enlarged["input_size"] = enlarged["transform"][0]["size"] = [640, 640]
    assert enlarged == development._policy()


@pytest.mark.parametrize("input_size", [True, 640.0, 128, 800, 960, [896, 896], None])
def test_unsupported_evaluation_sizes_fail_before_any_input_read(files, monkeypatch, input_size):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid size reached a file read")

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    with pytest.raises(development.DevelopmentEvaluationError, match="input_size"):
        development.build_development_binding(**files, input_size=input_size)


def test_binding_keeps_data_identity_and_revalidates_896(files):
    legacy = development.build_development_binding(**files)
    assert legacy == development.build_development_binding(**files, input_size=640)
    enlarged = development.build_development_binding(**files, input_size=896)
    assert development.validate_development_binding(enlarged) == enlarged
    for name in ("annotation", "manifest", "image_root", "image_count", "annotation_count",
                 "image_manifest_sha256", "image_order_sha256", "source"):
        assert enlarged[name] == legacy[name]
    assert enlarged["binding_sha256"] != legacy["binding_sha256"]
    assert any(row["path"].endswith("training_v2b_geometry.py") for row in enlarged["source"]["sources"])


@pytest.mark.parametrize("mutation", [
    "unsupported_size", "rectangle", "float_size", "wrong_resize", "batch",
    "dtype", "autocast", "extra_eligibility",
])
def test_rehashed_policy_cannot_change_other_semantics(files, mutation):
    binding = development.build_development_binding(**files, input_size=896)
    policy = binding["policy"]
    if mutation == "unsupported_size":
        policy["input_size"] = [800, 800]
    elif mutation == "rectangle":
        policy["input_size"] = [896, 640]
    elif mutation == "float_size":
        policy["input_size"] = [896.0, 896.0]
    elif mutation == "wrong_resize":
        policy["transform"][0]["size"] = [640, 640]
    elif mutation == "batch":
        policy["batch_size"] = 2
    elif mutation == "dtype":
        policy["forward_dtype"] = "bfloat16"
    elif mutation == "autocast":
        policy["autocast"] = True
    else:
        policy["scientific_comparison_eligible"] = True
    with pytest.raises(development.DevelopmentEvaluationError, match="policy|input_size"):
        development.validate_development_binding(rehash(binding), verify_files=False)


def test_real_transform_640_896_640_preserves_registry_and_original_coordinates(files):
    first = None
    for size in (640, 896, 640):
        binding = development.build_development_binding(**files, input_size=size)
        dataset = development._DevelopmentDataset(binding)
        pixels, target, _ = dataset.item(dataset.images[0])
        assert list(pixels.shape) == [3, size, size]
        assert pixels.dtype == torch.float32
        assert target["orig_size"].tolist() == [200, 100]
        if first is None:
            first = pixels.clone()
        elif size == 640:
            assert torch.equal(first, pixels)
    recipe = load_isolated_vendor_config_dict(ROOT)["val_dataloader"]["dataset"]["transforms"]
    assert recipe["ops"] == development._policy(640)["transform"]


@pytest.mark.parametrize("size,count,positions", [(640, 8400, 400), (896, 16464, 784)])
def test_real_cpu_model_geometry_and_ema_rounding_are_independent(size, count, positions):
    components = build_v2b_components(
        V2BConfig(seed=509, input_size=size, physical_batch_size=8,
                  accumulation_steps=2, pretrained_required=False), repo_root=ROOT,
    )
    raw = snapshot_model_geometry(components.model, expected_input_size=size)
    ema = snapshot_model_geometry(components.ema.module, expected_input_size=size)
    assert raw == ema
    assert raw["decoder"]["anchors"]["shape"] == [1, count, 4]
    assert raw["decoder"]["valid_mask"]["shape"] == [1, count, 1]
    assert raw["encoder"]["position_caches"]["pos_embed2"]["shape"] == [1, positions, 256]
    assert "encoder.pos_embed2" not in components.model.state_dict()
    assert "decoder.anchors" in components.model.state_dict()
    components.model.train()
    assert snapshot_model_geometry(components.model, expected_input_size=size) == raw
    components.model.eval()
    assert snapshot_model_geometry(components.model, expected_input_size=size) == raw

    # Legitimate EMA rounding must remain distinct from immutable raw geometry.
    anchor = components.ema.module.decoder.anchors
    index = int(components.ema.module.decoder.valid_mask.reshape(-1).nonzero()[0].item())
    anchor[0, index, 0] = torch.nextafter(anchor[0, index, 0], torch.tensor(float("inf")))
    rounded = snapshot_model_geometry(components.ema.module, expected_input_size=size)
    assert rounded["decoder"]["anchors"]["sha256"] != raw["decoder"]["anchors"]["sha256"]
    assert snapshot_model_geometry(components.model, expected_input_size=size) == raw


def test_actual_896_ema_fp32_capacity_forward_on_one_synthetic_image(files, tmp_path):
    coco = json.loads(files["annotation_file"].read_text())
    manifest = json.loads(files["manifest_file"].read_text())
    coco["images"], coco["annotations"] = coco["images"][:1], coco["annotations"][:1]
    manifest["records"] = manifest["records"][:1]
    candidate = dict(files, annotation_sha256=write_json(files["annotation_file"], coco),
                     manifest_sha256=write_json(files["manifest_file"], manifest))
    binding = development.build_development_binding(**candidate, input_size=896)
    components = build_v2b_components(
        V2BConfig(seed=509, input_size=896, physical_batch_size=8,
                  accumulation_steps=2, pretrained_required=False), repo_root=ROOT,
    )
    components.engine.begin_epoch(1)
    before = rng_snapshot()
    result = development.evaluate_development_capacity(
        components, data_binding=binding, output_dir=output_dir(tmp_path),
        hardware_gate=lambda: None,
    )
    assert_rng_equal(before, rng_snapshot())
    assert result["weights"]["eval_spatial_size"] == [896, 896]
    assert result["scope"] == "development_capacity_only"
    assert result["scientific_comparison_eligible"] is False
    assert result["training_state_unchanged"]["engine"]["epoch_active"] is True
    assert result["data"]["evaluated_image_count"] == 1
    predictions = json.loads(Path(result["prediction_artifact"]["path"]).read_text())
    assert len(predictions) == 300


def test_capacity_covers_all_batches_without_formal_epoch_or_scientific_eligibility(files, tmp_path):
    binding = development.build_development_binding(**files, input_size=896)
    components = synthetic_components()
    before_rng, before_modes = rng_snapshot(), mode_snapshot(components)
    before_engine = copy.deepcopy(components["engine"].state_dict())
    result = development.evaluate_development_capacity(
        components, data_binding=binding, output_dir=output_dir(tmp_path),
        hardware_gate=lambda: None,
    )
    assert components["model"].calls == []
    assert [row[0] for row in components["ema"].module.calls] == [[4, 3, 896, 896], [1, 3, 896, 896]]
    assert all(row[1] == torch.float32 for row in components["ema"].module.calls)
    assert result["scope"] == "development_capacity_only"
    assert result["logged_epoch"] is None
    assert result["capacity_only"] is True
    assert result["full_development_coverage"] is True
    assert result["full_development_evaluation"] is False
    assert result["scientific_comparison_eligible"] is False
    assert result["checkpoint_selection_performed"] is False
    assert result["data"]["evaluated_image_count"] == 5
    assert result["data"]["observed_images_sha256"] == binding["image_manifest_sha256"]
    assert result["weights"]["eval_spatial_size"] == [896, 896]
    assert result["coco_secondary"]["metrics"]["AP"]["value"] == pytest.approx(100.0)
    predictions = json.loads(Path(result["prediction_artifact"]["path"]).read_text())
    assert predictions[0]["bbox"] == pytest.approx([40, 10, 20, 20])
    assert_rng_equal(before_rng, rng_snapshot())
    assert mode_snapshot(components) == before_modes
    assert components["engine"].state_dict() == before_engine

    with pytest.raises(development.DevelopmentEvaluationError, match="completed epoch"):
        development.evaluate_development(
            components, data_binding=binding, output_dir=output_dir(tmp_path, "formal"),
            logged_epoch=10, hardware_gate=lambda: None,
        )


def test_capacity_cannot_impersonate_an_epoch_or_preview():
    with pytest.raises(TypeError, match="logged_epoch"):
        development.evaluate_development_capacity(
            {}, data_binding={}, output_dir="unused", hardware_gate=lambda: None, logged_epoch=10,
        )
    for options in (
        {"preview": True, "capacity": True, "logged_epoch": None},
        {"preview": False, "capacity": True, "logged_epoch": 10},
        {"preview": False, "capacity": 1, "logged_epoch": None},
    ):
        with pytest.raises(development.DevelopmentEvaluationError):
            development._evaluate({}, data_binding={}, output_dir="unused",
                                  hardware_gate=lambda: None, **options)


def test_preview_stays_four_images_at_896(files, tmp_path):
    binding = development.build_development_binding(**files, input_size=896)
    components = synthetic_components()
    result = development.preview_development(
        components, data_binding=binding, output_dir=output_dir(tmp_path),
        hardware_gate=lambda: None,
    )
    assert result["scope"] == "development_preview_only"
    assert result["data"]["evaluated_image_count"] == 4
    assert result["full_development_coverage"] is False
    assert result["scientific_comparison_eligible"] is False
    assert len(components["ema"].module.calls) == 1


def test_capacity_requires_the_complete_observed_manifest(files, tmp_path, monkeypatch):
    binding = development.build_development_binding(**files, input_size=896)
    components = synthetic_components()
    original = development._DevelopmentDataset.item

    def corrupt_receipt(dataset, image):
        pixels, target, receipt = original(dataset, image)
        if image["id"] == 5:
            receipt["stable_image_id"] = "0" * 64
        return pixels, target, receipt

    monkeypatch.setattr(development._DevelopmentDataset, "item", corrupt_receipt)
    directory = output_dir(tmp_path)
    with pytest.raises(development.DevelopmentEvaluationError, match="observed image identities"):
        development.evaluate_development_capacity(
            components, data_binding=binding, output_dir=directory, hardware_gate=lambda: None,
        )
    assert not list(directory.iterdir())


@pytest.mark.parametrize("mutation", ["engine", "raw_size", "ema_size", "pos", "anchors", "mask", "missing_encoder"])
def test_stale_geometry_fails_before_any_image_or_forward(files, tmp_path, monkeypatch, mutation):
    binding = development.build_development_binding(**files, input_size=896)
    components = synthetic_components()
    if mutation == "engine":
        components["engine"].expected_input_size = (640, 640)
    elif mutation == "raw_size":
        components["model"].encoder.eval_spatial_size = [640, 640]
    elif mutation == "ema_size":
        components["ema"].module.decoder.eval_spatial_size = [640, 640]
    elif mutation == "pos":
        components["ema"].module.encoder.pos_embed2 = torch.zeros(1, 400, 256)
    elif mutation == "anchors":
        components["ema"].module.decoder.anchors = torch.zeros(1, 8400, 4)
    elif mutation == "mask":
        components["ema"].module.decoder.valid_mask = torch.ones(1, 8400, 1, dtype=torch.bool)
    else:
        del components["model"].encoder

    def forbidden(*args, **kwargs):
        raise AssertionError("stale model geometry reached an image")

    monkeypatch.setattr(development._DevelopmentDataset, "item", forbidden)
    directory = output_dir(tmp_path)
    with pytest.raises(development.DevelopmentEvaluationError, match="geometry"):
        development.evaluate_development_capacity(
            components, data_binding=binding, output_dir=directory, hardware_gate=lambda: None,
        )
    assert components["ema"].module.calls == []
    assert not list(directory.iterdir())


@pytest.mark.parametrize("gate_index", [1, 3, 8])
@pytest.mark.parametrize("mutation", ["plain_pos", "size_attribute"])
def test_geometry_mutation_is_rejected_including_before_publication(files, tmp_path, gate_index, mutation):
    binding = development.build_development_binding(**files, input_size=896)
    components = synthetic_components()
    before_rng, before_modes = rng_snapshot(), mode_snapshot(components)
    calls = []

    def gate():
        calls.append(None)
        if len(calls) == gate_index:
            if mutation == "plain_pos":
                components["ema"].module.encoder.pos_embed2.add_(1)
            else:
                components["ema"].module.encoder.eval_spatial_size = [640, 640]

    directory = output_dir(tmp_path)
    with pytest.raises(development.DevelopmentEvaluationError, match="geometry"):
        development.evaluate_development_capacity(
            components, data_binding=binding, output_dir=directory, hardware_gate=gate,
        )
    assert_rng_equal(before_rng, rng_snapshot())
    assert mode_snapshot(components) == before_modes
    assert not list(directory.iterdir())


def test_geometry_hashes_values_and_enforces_buffer_registration():
    model = Predictor(input_size=896)
    original = snapshot_model_geometry(model, expected_input_size=896)
    model.encoder.pos_embed2.add_(1)
    changed = snapshot_model_geometry(model, expected_input_size=896)
    assert changed["encoder"]["position_caches"] != original["encoder"]["position_caches"]
    mask = model.decoder.valid_mask
    del model.decoder.valid_mask
    model.decoder.valid_mask = mask
    with pytest.raises(ModelGeometryError, match="persistent"):
        snapshot_model_geometry(model, expected_input_size=896)
