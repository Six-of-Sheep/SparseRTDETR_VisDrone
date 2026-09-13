"""Actual CPU R18 runtime wiring on synthetic train_core JPEG/COCO fixtures.

These are engineering updates, not a scientific training run. No actual
train_core images, evaluation, GPU, development or held-out data are opened.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from sparse_rtdetr.baseline.training_v2b import V2BConfig, build_v2b_components
from sparse_rtdetr.baseline.training_v2b_data import (
    TrainCoreDataConfig, build_train_core_loader,
)
from sparse_rtdetr.baseline.training_v2b_runtime import (
    V2BTrainingSession, build_train_core_run_binding,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    torch.set_num_threads(2)

    def forbidden(*args, **kwargs):
        raise AssertionError("CUDA initialization is forbidden in runtime CPU tests")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)


def make_files(tmp_path, count=7):
    root = tmp_path / "approved_images"
    root.mkdir()
    images, annotations, records = [], [], []
    generator = np.random.default_rng(711)
    for index in range(count):
        image_id = index + 1
        name = f"0000003_{index:05d}_d_{index:07d}.jpg"
        logical = f"train/images/{name}"
        stable = hashlib.sha256(f"runtime-fixture-{image_id}".encode()).hexdigest()
        width, height = 100 + index % 3 * 4, 88
        pixels = generator.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        path = root / name
        Image.fromarray(pixels).save(path, format="JPEG", quality=85)
        raw = path.read_bytes()
        images.append({"id": image_id, "file_name": logical, "split": "train",
                       "stable_image_id": stable, "width": width, "height": height})
        for box_index in range(2):
            annotations.append({"id": len(annotations) + 1, "image_id": image_id,
                                "category_id": (image_id + box_index) % 10 + 1,
                                "bbox": [11 + box_index * 32, 14, 26, 31],
                                "area": 26 * 31, "iscrowd": 0})
        records.append({"coco_image_id": image_id, "relative_path": logical, "split": "train",
                        "stable_image_id": stable, "width": width, "height": height,
                        "image_sha256": hashlib.sha256(raw).hexdigest(), "image_size_bytes": len(raw)})
    coco = {"images": images, "annotations": annotations,
            "categories": [{"id": index, "name": f"class_{index}"} for index in range(1, 11)]}
    result = {"image_root": root, "repo_root": ROOT}
    for key, filename, document in (
        ("annotation", "train_core_coco.json", coco),
        ("manifest", "train_core_manifest.json", {"records": records}),
    ):
        raw = json.dumps(document, sort_keys=True).encode()
        path = tmp_path / filename
        path.write_bytes(raw)
        result[key + "_file"] = path
        result[key + "_sha256"] = hashlib.sha256(raw).hexdigest()
    return result


@pytest.fixture
def files(tmp_path):
    return make_files(tmp_path)


def components(**overrides):
    values = {"seed": 17, "input_size": 128, "physical_batch_size": 1,
              "accumulation_steps": 2, "pretrained_required": False,
              "warmup_steps": 2, "ema_warmups": 2}
    values.update(overrides)
    return build_v2b_components(V2BConfig(**values), repo_root=ROOT)


def data_loader(files, model, **overrides):
    values = {"seed": model.config.seed, "input_size": model.config.input_size,
              "logical_batch_size": model.config.logical_batch_size, "num_workers": 0}
    values.update(overrides)
    return build_train_core_loader(TrainCoreDataConfig(**values), **files)


def session(files, *, workers=0, run_id="synthetic-runtime-replay", **overrides):
    model = components(**overrides)
    loader = data_loader(files, model, num_workers=workers)
    binding = build_train_core_run_binding(model, loader, run_id=run_id, repo_root=ROOT)
    return V2BTrainingSession(model, loader, binding), model, loader, binding


def assert_nested_equal(first, second):
    if isinstance(first, torch.Tensor):
        assert isinstance(second, torch.Tensor)
        assert first.dtype == second.dtype and first.shape == second.shape
        assert torch.equal(first, second)
    elif isinstance(first, np.ndarray):
        assert isinstance(second, np.ndarray) and first.dtype == second.dtype
        assert np.array_equal(first, second)
    elif isinstance(first, dict):
        assert type(first) is type(second) and first.keys() == second.keys()
        for key in first:
            assert_nested_equal(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert type(first) is type(second) and len(first) == len(second)
        for left, right in zip(first, second):
            assert_nested_equal(left, right)
    else:
        assert first == second


def rng():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def test_real_r18_resume_through_dataset_workers_checkpoint_and_runtime(files, tmp_path):
    current, model, loader, binding = session(files)
    assert sum(parameter.numel() for parameter in model.model.parameters()) == 20_094_584
    assert binding["data"]["kind"] == "train_core_runtime"
    assert binding["data"]["loader_binding_sha256"] == loader.binding_sha256
    current.begin_epoch(1)
    iterator = iter(loader)
    first = next(iterator)
    assert loader.state_dict()["completed_batches"] == 0
    first_record = current.train_batch(first)
    assert first_record["input"] == first.evidence()
    assert model.engine.state_dict()["optimizer_updates"] == 1
    assert loader.state_dict()["completed_batches"] == 1
    assert loader.has_active_iterator
    checkpoint = tmp_path / "runtime-after-one.pt"
    reference = current.save(checkpoint)
    assert checkpoint.is_file()
    second = next(iterator)
    expected_record = current.train_batch(second)
    iterator.close()
    expected_rng = rng()

    restored, other, other_loader, other_binding = session(files, workers=2)
    assert binding == other_binding
    inspection = restored.restore(checkpoint, expected_sha256=reference["sha256"])
    assert isinstance(inspection, dict)
    assert other.engine.state_dict()["optimizer_updates"] == 1
    assert other_loader.state_dict()["completed_batches"] == 1
    other_iterator = iter(other_loader)
    replay = next(other_iterator)
    assert replay.evidence() == second.evidence()
    actual_record = restored.train_batch(replay)
    other_iterator.close()
    assert_nested_equal(actual_record, expected_record)
    assert_nested_equal(other.model.state_dict(), model.model.state_dict())
    assert_nested_equal(other.ema.module.state_dict(), model.ema.module.state_dict())
    assert other.ema.updates == model.ema.updates
    assert_nested_equal(other.optimizer.state_dict(), model.optimizer.state_dict())
    assert_nested_equal(other.scheduler.state_dict(), model.scheduler.state_dict())
    assert other.scheduler._get_closed_form_lr() == model.scheduler._get_closed_form_lr()
    assert_nested_equal(other.warmup.state_dict(), model.warmup.state_dict())
    assert_nested_equal(other.engine.state_dict(), model.engine.state_dict())
    assert other_loader.state_dict() == loader.state_dict()
    assert_nested_equal(rng(), expected_rng)
    assert other.runtime.identity["device"] == "cpu"
    assert not torch.cuda.is_initialized()


def test_mutated_pending_batch_is_refused_before_model_update(files):
    current, model, loader, _ = session(files)
    current.begin_epoch(1)
    iterator = iter(loader)
    batch = next(iterator)
    initial_bn = {name: module.num_batches_tracked.clone()
                  for name, module in model.model.named_modules()
                  if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)}
    original = batch.images.clone()
    batch.images[0, 0, 0, 0] += 0.125
    with pytest.raises((ValueError, RuntimeError), match="changed|batch"):
        current.train_batch(batch)
    assert loader.state_dict()["completed_batches"] == 0
    assert model.engine.state_dict()["optimizer_updates"] == 0
    for name, module in model.model.named_modules():
        if name in initial_bn:
            assert torch.equal(module.num_batches_tracked, initial_bn[name])
    batch.images.copy_(original)
    current.train_batch(batch)
    assert loader.state_dict()["completed_batches"] == 1
    iterator.close()


def test_post_forward_failure_poisoned_engine_never_acknowledges_or_saves(files, tmp_path, monkeypatch):
    current, model, loader, _ = session(files)
    current.begin_epoch(1)
    iterator = iter(loader)
    batch = next(iterator)

    def broken(*args, **kwargs):
        raise RuntimeError("injected criterion failure after model forward")

    monkeypatch.setattr(model.criterion, "forward", broken)
    with pytest.raises(RuntimeError, match="fail|injected") as caught:
        current.train_batch(batch)
    cause = caught.value
    messages = []
    while cause is not None:
        messages.append(str(cause))
        cause = cause.__cause__
    assert any("injected criterion failure" in message for message in messages)
    assert loader.state_dict()["completed_batches"] == 0
    assert model.engine.failed
    with pytest.raises((RuntimeError, ValueError), match="fail|poison|rebuild"):
        current.train_batch(batch)
    path = tmp_path / "must-not-save.pt"
    with pytest.raises((RuntimeError, ValueError)):
        current.save(path)
    assert not path.exists()
    iterator.close()


def test_unacknowledged_engine_update_cannot_be_checkpointed(files, tmp_path):
    current, model, loader, _ = session(files)
    current.begin_epoch(1)
    iterator = iter(loader)
    batch = next(iterator)
    model.engine.train_window(batch.images, batch.targets)
    assert model.engine.state_dict()["optimizer_updates"] == 1
    assert loader.state_dict()["optimizer_updates"] == 0
    path = tmp_path / "unacknowledged.pt"
    with pytest.raises((ValueError, RuntimeError), match="mismatch|differ|cursor"):
        current.save(path)
    assert not path.exists()
    iterator.close()


@pytest.mark.parametrize("override", [
    {"seed": 18}, {"input_size": 160}, {"logical_batch_size": 1},
    {"augmentation_stop_internal_epoch": 0},
])
def test_data_model_configuration_mismatch_is_rejected(files, override):
    model = components()
    loader = data_loader(files, model, **override)
    with pytest.raises((RuntimeError, ValueError), match="seed|size|logical|config|differ"):
        binding = build_train_core_run_binding(model, loader, run_id="mismatched-config", repo_root=ROOT)
        V2BTrainingSession(model, loader, binding)


def test_initial_weight_mutation_cannot_claim_original_binding(files):
    model = components()
    loader = data_loader(files, model)
    binding = build_train_core_run_binding(model, loader, run_id="changed-initial-weights", repo_root=ROOT)
    with torch.no_grad():
        next(model.model.parameters()).add_(0.1)
    with pytest.raises((ValueError, RuntimeError), match="initial|weight|state|parameter"):
        V2BTrainingSession(model, loader, binding)


def test_binding_tamper_cannot_enter_session(files):
    model = components()
    loader = data_loader(files, model)
    binding = build_train_core_run_binding(model, loader, run_id="altered-binding", repo_root=ROOT)
    forged = copy.deepcopy(binding)
    forged["data"]["loader_binding_sha256"] = "0" * 64
    with pytest.raises((ValueError, RuntimeError), match="binding|loader|data"):
        V2BTrainingSession(model, loader, forged)


def test_source_drift_since_model_construction_is_rejected(files, monkeypatch):
    model = components()
    loader = data_loader(files, model)
    from sparse_rtdetr.baseline import training_v2b_evidence as evidence
    original = evidence._read_regular
    engine_source = ROOT / "src/sparse_rtdetr/baseline/training_v2b_engine.py"

    def changed(path):
        actual, raw = original(path)
        if actual.resolve() == engine_source:
            raw += b"\n# synthetic source drift\n"
        return actual, raw

    monkeypatch.setattr(evidence, "_read_regular", changed)
    with pytest.raises((RuntimeError, ValueError), match="source|code|identity|drift"):
        binding = build_train_core_run_binding(model, loader, run_id="changed-source", repo_root=ROOT)
        V2BTrainingSession(model, loader, binding)


def test_finish_epoch_requires_all_logical_batches_and_closed_iterator(files):
    current, model, loader, _ = session(files)
    current.begin_epoch(1)
    iterator = iter(loader)
    batch = next(iterator)
    with pytest.raises((ValueError, RuntimeError), match="iterator|complete|epoch"):
        current.finish_epoch()
    current.train_batch(batch)
    iterator.close()
    with pytest.raises((ValueError, RuntimeError), match="complete|epoch|batch"):
        current.finish_epoch()
    assert loader.state_dict()["epoch_active"]


def test_logical16_physical16_or8_share_exact_augmented_input_without_batch_equivalence_claim(tmp_path):
    paths = make_files(tmp_path, count=33)
    first = components(seed=24, physical_batch_size=16, accumulation_steps=1)
    second = components(seed=24, physical_batch_size=8, accumulation_steps=2)
    first_loader = data_loader(paths, first, num_workers=0)
    second_loader = data_loader(paths, second, num_workers=2)
    assert first.initialization["parameters"] == second.initialization["parameters"]
    assert first_loader.binding_sha256 == second_loader.binding_sha256
    a = list(first_loader.preview_batches(1, epoch=1))[0]
    b = list(second_loader.preview_batches(1, epoch=1))[0]
    assert a.evidence() == b.evidence()
    assert torch.equal(a.images, b.images)
    assert_nested_equal(a.targets, b.targets)
    assert first.engine.state_dict()["optimizer_updates"] == second.engine.state_dict()["optimizer_updates"] == 0
    assert first.config.bn_statistics == second.config.bn_statistics == "train"
    assert first.config.num_denoising == second.config.num_denoising == 100
    # Physical batches have different native BN/DN behavior; no loss/AP equivalence asserted.


def test_real_session_complete_epoch_and_next_epoch_keep_engine_loader_clocks_aligned(tmp_path):
    paths = make_files(tmp_path, count=2)
    current, model, loader, _ = session(paths)
    current.begin_epoch(1)
    for batch in loader:
        current.train_batch(batch)
    record = current.finish_epoch()
    assert record["epoch"]["epoch"] == 1
    assert record["epoch"]["epoch_optimizer_steps"] == 1
    assert record["epoch"]["optimizer_updates"] == 1
    assert record["epoch"]["microsteps"] == 2
    assert record["epoch"]["scheduler_stepped"] is False
    assert not loader.has_active_iterator
    assert not loader.state_dict()["epoch_active"]
    assert not model.engine.state_dict()["epoch_active"]
    assert loader.state_dict()["completed_batches"] == 1
    current.begin_epoch(2)
    assert loader.state_dict()["completed_batches"] == 0
    assert loader.state_dict()["epoch_start_optimizer_updates"] == 1
    assert model.engine.state_dict()["epoch_start_optimizer_updates"] == 1
