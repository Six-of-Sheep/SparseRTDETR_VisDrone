"""CPU-only 896 training wiring; every image and annotation below is synthetic."""

from __future__ import annotations

import contextlib
import copy
import gc
import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from sparse_rtdetr.baseline.training_v2b import (
    V2BConfig, V2BConfigurationError, build_v2b_components,
)
from sparse_rtdetr.baseline.training_v2b_data import (
    TrainCoreDataConfig, TrainCoreDataError, build_train_core_loader,
)
from sparse_rtdetr.baseline.training_v2b_evidence import (
    EvidenceError, build_run_binding, canonical_sha256, initial_parameter_reference,
    validate_run_binding,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def cpu_without_tensor_archives(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    torch.set_num_threads(2)

    def forbidden(*args, **kwargs):
        raise AssertionError("this synthetic resolution test cannot initialize CUDA or read weights")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", forbidden)


def _write_json(path, value):
    raw = json.dumps(value, sort_keys=True).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def synthetic_files(tmp_path):
    image_root = tmp_path / "synthetic_images"
    image_root.mkdir()
    images, annotations, records = [], [], []
    for image_id in range(1, 6):
        width, height = 1024, 128
        name = f"synthetic_{image_id}.jpg"
        path = image_root / name
        Image.new("RGB", (width, height), (image_id * 20, 70, 130)).save(path, format="JPEG")
        raw = path.read_bytes()
        stable_id = hashlib.sha256(f"resolution-fixture-{image_id}".encode()).hexdigest()
        logical_name = f"train/images/{name}"
        images.append({
            "id": image_id, "file_name": logical_name, "split": "train",
            "stable_image_id": stable_id, "width": width, "height": height,
        })
        # Width 1.3 survives the original-image min_size=1 check, but becomes
        # 0.8125 pixels at 640 and 1.1375 pixels at 896 after the real Resize.
        for category, bbox in ((1, [10.0, 10.0, 1.3, 8.0]), (2, [40.0, 20.0, 16.0, 8.0])):
            annotations.append({
                "id": len(annotations) + 1, "image_id": image_id,
                "category_id": category, "bbox": bbox,
                "area": bbox[2] * bbox[3], "iscrowd": 0,
            })
        records.append({
            "coco_image_id": image_id, "relative_path": logical_name,
            "stable_image_id": stable_id, "split": "train", "width": width, "height": height,
            "image_sha256": hashlib.sha256(raw).hexdigest(), "image_size_bytes": len(raw),
        })
    annotation_file, manifest_file = tmp_path / "train_core_coco.json", tmp_path / "train_core_manifest.json"
    return {
        "annotation_file": annotation_file,
        "annotation_sha256": _write_json(annotation_file, {
            "images": images, "annotations": annotations,
            "categories": [{"id": index, "name": f"class_{index}"} for index in range(1, 11)],
        }),
        "manifest_file": manifest_file,
        "manifest_sha256": _write_json(manifest_file, {"records": records}),
        "image_root": image_root, "repo_root": ROOT,
    }


def _loader(files, size=896, **overrides):
    values = dict(seed=0, input_size=size, logical_batch_size=2, num_workers=0)
    values.update(overrides)
    return build_train_core_loader(TrainCoreDataConfig(**values), **files)


def _engine_state(loader, completed=0):
    return {
        "schema_version": 1,
        "config": {
            "physical_batch_size": 1, "accumulation_steps": 2, "amp_dtype": "bfloat16",
            "clip_max_norm": 0.1, "bn_statistics": "train",
            "expected_input_size": [loader.config.input_size] * 2,
            "bn_backward_layout": "cpu_contiguous_cuda_native",
        },
        "epoch": 1, "epoch_active": True, "epoch_start_optimizer_updates": 0,
        "optimizer_updates": completed, "microsteps": completed * 2,
        "phase": 0, "failed": False,
    }


def _binding(loader):
    config = V2BConfig(input_size=loader.config.input_size, physical_batch_size=1,
                       accumulation_steps=2, pretrained_required=False)
    return build_run_binding(
        run_id=f"synthetic-resolution-{config.input_size}",
        code_paths={row["path"]: ROOT / row["path"] for row in loader.binding["source"]["sources"]},
        config=config.binding_config(scope="train_core_runtime_engineering"),
        initial_parameters=initial_parameter_reference({"synthetic_weight": torch.ones(1)}),
        train_core_input=loader.binding,
    )


def test_defaults_remain_640_and_explicit_896_does_not_change_other_fields():
    default = V2BConfig()
    assert (default.seed, default.input_size, default.physical_batch_size,
            default.accumulation_steps, default.logical_batch_size) == (0, 640, 16, 1, 16)
    assert TrainCoreDataConfig().input_size == 640
    explicit = asdict(V2BConfig(input_size=896))
    explicit["input_size"] = 640
    assert explicit == asdict(default)


@pytest.mark.parametrize("size", [*range(128, 641, 32), 896])
def test_preserves_every_legacy_size_and_only_adds_896(size):
    assert V2BConfig(input_size=size).input_size == size
    assert TrainCoreDataConfig(input_size=size).input_size == size


@pytest.mark.parametrize("size", [96, 129, 672, 800, 864, 928, 960, 1056, 1152, True, 896.0, "896"])
def test_other_resolutions_and_wrong_types_still_fail(size):
    with pytest.raises(V2BConfigurationError):
        V2BConfig(input_size=size)
    with pytest.raises(TrainCoreDataError):
        TrainCoreDataConfig(input_size=size)


def test_actual_factory_640_896_640_preserves_parameters_and_resolves_geometry():
    observations = []
    for size, anchors, positions in ((640, 8400, 400), (896, 16464, 784), (640, 8400, 400)):
        components = build_v2b_components(
            V2BConfig(input_size=size, physical_batch_size=8, accumulation_steps=2,
                      pretrained_required=False, sampling_backend="deterministic_gather"),
            repo_root=ROOT,
        )
        assert components.resolved_config["eval_spatial_size"] == [size, size]
        assert list(components.model.encoder.eval_spatial_size) == [size, size]
        assert list(components.model.decoder.eval_spatial_size) == [size, size]
        assert components.geometry["anchors"] == [1, anchors, 4]
        assert components.geometry["valid_mask"] == [1, anchors, 1]
        assert components.geometry["position_caches"] == {"pos_embed2": [1, positions, 256]}
        assert components.engine.expected_input_size == (size, size)
        assert components.engine.logical_batch_size == 16
        assert components.resolved_config["train_dataloader"]["collate_fn"]["scales"] is None
        for role in ("train_dataloader", "val_dataloader"):
            operations = components.resolved_config[role]["dataset"]["transforms"]["ops"]
            assert [op["size"] for op in operations if op["type"] == "Resize"] == [[size, size]]
        observations.append({
            "parameters": components.initialization["parameters"],
            "model_state": components.initialization["model_state"],
            "geometry": components.geometry, "rng": torch.get_rng_state().clone(),
        })
        del components
        gc.collect()
    assert observations[0]["parameters"] == observations[1]["parameters"] == observations[2]["parameters"]
    assert observations[0]["model_state"] == observations[2]["model_state"]
    assert observations[0]["geometry"] == observations[2]["geometry"]
    assert observations[0]["model_state"] != observations[1]["model_state"]
    assert torch.equal(observations[0]["rng"], observations[1]["rng"])
    assert torch.equal(observations[0]["rng"], observations[2]["rng"])


@pytest.mark.parametrize("size", [640, 896])
def test_actual_loader_resize_and_public_evidence_binding_use_requested_size(synthetic_files, size):
    loader = _loader(synthetic_files, size)
    binding = _binding(loader)
    assert validate_run_binding(binding) == binding
    bound_loader = binding["data"]["loader"]
    assert binding["config"]["input_size"] == size
    assert bound_loader["config"]["input_size"] == size
    assert [op["size"] for op in bound_loader["transforms"]["ops"]
            if op["type"] == "Resize"] == [[size, size]]
    assert bound_loader["config"]["multiscale"] is False
    assert loader.batches_per_epoch == 2
    assert loader.state_dict()["dropped_samples"] == 1
    batch = next(loader.preview_batches(1, epoch=1))
    assert tuple(batch.images.shape) == (2, 3, size, size)


@pytest.mark.parametrize("size", [800, 960])
def test_passive_evidence_validator_rejects_repacked_unsupported_size(synthetic_files, size):
    binding = copy.deepcopy(_binding(_loader(synthetic_files)))
    binding["config"]["input_size"] = size
    data, loader = binding["data"], binding["data"]["loader"]
    loader["config"]["input_size"] = size
    for op in loader["transforms"]["ops"]:
        if op["type"] == "Resize":
            op["size"] = [size, size]
    loader["loader_binding_sha256"] = canonical_sha256({
        key: value for key, value in loader.items() if key != "loader_binding_sha256"
    })
    data["loader_binding_sha256"] = loader["loader_binding_sha256"]
    binding["binding_sha256"] = canonical_sha256({
        key: value for key, value in binding.items() if key != "binding_sha256"
    })
    with pytest.raises(EvidenceError, match="input geometry is out of scope"):
        validate_run_binding(binding, verify_files=False)


def test_real_post_resize_sanitize_changes_retained_gt_without_changing_source_order(synthetic_files):
    # Disabling the three stochastic photometric/crop operators isolates the
    # existing Resize/Sanitize mechanism; this is not a proposed training recipe.
    low = _loader(synthetic_files, 640, augmentation_stop_internal_epoch=0)
    high = _loader(synthetic_files, 896, augmentation_stop_internal_epoch=0)
    a = next(low.preview_batches(1, epoch=1))
    b = next(high.preview_batches(1, epoch=1))
    common = lambda samples: [{key: value for key, value in sample.items()
                               if key != "binding_sha256"} for sample in samples]
    assert common(a.samples) == common(b.samples)
    assert a.binding_sha256 != b.binding_sha256
    assert a.augmented_sha256 != b.augmented_sha256
    assert a.receipt_sha256 != b.receipt_sha256
    assert [target["labels"].tolist() for target in a.targets] == [[1], [1]]
    assert [target["labels"].tolist() for target in b.targets] == [[0, 1], [0, 1]]
    assert sum(len(target["labels"]) for target in a.targets) == 2
    assert sum(len(target["labels"]) for target in b.targets) == 4
    for target in b.targets:
        narrow_width = float(target["boxes"][target["labels"] == 0, 2].item()) * 896
        assert narrow_width == pytest.approx(1.3 * 896 / 1024, rel=1e-4)


def test_896_spawn_workers_and_committed_cursor_reproduce_exact_augmented_receipts(synthetic_files):
    reference = _loader(synthetic_files)
    workers = _loader(synthetic_files, num_workers=2)
    assert reference.binding_sha256 == workers.binding_sha256
    before = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    expected = list(reference.preview_batches(2, epoch=1))
    assert random.getstate() == before[0]
    after_numpy = np.random.get_state()
    assert after_numpy[0] == before[1][0] and np.array_equal(after_numpy[1], before[1][1])
    assert after_numpy[2:] == before[1][2:]
    assert torch.equal(torch.get_rng_state(), before[2])
    observed = list(workers.preview_batches(2, epoch=1))
    assert [batch.evidence() for batch in observed] == [batch.evidence() for batch in expected]
    workers.begin_epoch(1, _engine_state(workers))
    with contextlib.closing(iter(workers)) as iterator:
        first = next(iterator)
        workers.commit_batch(first, _engine_state(workers, completed=1))
    committed = workers.state_dict()
    resumed = _loader(synthetic_files)
    resumed.load_state_dict(committed)
    remaining = list(resumed.preview_batches(1))
    assert remaining[0].evidence() == expected[1].evidence()
    assert resumed.state_dict() == committed


def test_896_loader_refuses_640_resume_before_mutating_cursor(synthetic_files):
    low, high = _loader(synthetic_files, 640), _loader(synthetic_files, 896)
    low.begin_epoch(1, _engine_state(low))
    before = high.state_dict()
    with pytest.raises(TrainCoreDataError, match="binding_sha256"):
        high.load_state_dict(low.state_dict())
    assert high.state_dict() == before
