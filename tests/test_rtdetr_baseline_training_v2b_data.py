"""Real vendored augmentation and spawn-worker recovery on synthetic JPEG files."""

from __future__ import annotations

import copy
import hashlib
import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from sparse_rtdetr.baseline.training_v2b_data import (
    TrainCoreDataConfig,
    TrainCoreDataError,
    augmented_tensor_sha256,
    build_train_core_loader,
    validate_loader_state,
    validate_worker_environment,
)

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    raw = json.dumps(value, sort_keys=True).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def files(tmp_path):
    image_root = tmp_path / "approved_images"
    image_root.mkdir()
    images, annotations, records = [], [], []
    generator = np.random.default_rng(901)
    for index in range(13):
        image_id = index + 1
        name = f"0000001_{index:05d}_d_{index:07d}.jpg"
        width, height = 112 + (index % 3) * 4, 80 + (index % 2) * 8
        pixels = generator.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        path = image_root / name
        Image.fromarray(pixels).save(path, format="JPEG", quality=85)
        raw = path.read_bytes()
        stable = hashlib.sha256(f"synthetic-train-core-{image_id}".encode()).hexdigest()
        logical = f"train/images/{name}"
        images.append({"id": image_id, "file_name": logical, "split": "train",
                       "stable_image_id": stable, "width": width, "height": height})
        if index != 0:
            for box_index in range(2):
                annotations.append({"id": len(annotations) + 1, "image_id": image_id,
                                    "category_id": (index + box_index) % 10 + 1,
                                    "bbox": [12 + 18 * box_index, 11, 23, 29],
                                    "area": 23 * 29, "iscrowd": 0})
        records.append({"coco_image_id": image_id, "relative_path": logical,
                        "stable_image_id": stable, "split": "train",
                        "width": width, "height": height,
                        "image_sha256": hashlib.sha256(raw).hexdigest(),
                        "image_size_bytes": len(raw)})
    coco = {"images": images, "annotations": annotations,
            "categories": [{"id": index, "name": f"class_{index}"} for index in range(1, 11)]}
    manifest = {"records": records}
    annotation_path = tmp_path / "train_core_coco.json"
    manifest_path = tmp_path / "train_core_manifest.json"
    return {"annotation_file": annotation_path, "annotation_sha256": write_json(annotation_path, coco),
            "manifest_file": manifest_path, "manifest_sha256": write_json(manifest_path, manifest),
            "image_root": image_root, "repo_root": ROOT}


def build(files, **overrides):
    config = TrainCoreDataConfig(seed=41, input_size=128, logical_batch_size=4, num_workers=0)
    return build_train_core_loader(replace(config, **overrides), **files)


def engine_state(loader, *, epoch=1, completed=0, active=True, accumulation=2):
    start = (epoch - 1) * loader.batches_per_epoch if epoch else 0
    updates = start + completed
    return {
        "schema_version": 1,
        "config": {"physical_batch_size": loader.config.logical_batch_size // accumulation,
                   "accumulation_steps": accumulation, "amp_dtype": "bfloat16",
                   "clip_max_norm": 0.1, "bn_statistics": "train",
                   "expected_input_size": [128, 128], "bn_backward_layout": "cpu_contiguous_cuda_native"},
        "epoch": epoch, "epoch_active": active,
        "epoch_start_optimizer_updates": start, "optimizer_updates": updates,
        "microsteps": updates * accumulation, "phase": 0, "failed": False,
    }


def start(loader, epoch=1):
    loader.begin_epoch(epoch, engine_state(loader, epoch=epoch))


def consume(loader):
    rows = []
    for batch in loader:
        rows.append(batch.evidence())
        loader.commit_batch(batch, engine_state(loader, epoch=batch.epoch,
                                               completed=batch.logical_batch_index + 1))
    return rows


def rng_snapshot():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def assert_rng_equal(first, second):
    assert first[0] == second[0]
    assert first[1][0] == second[1][0]
    assert np.array_equal(first[1][1], second[1][1])
    assert first[1][2:] == second[1][2:]
    assert torch.equal(first[2], second[2])


def test_metadata_only_construction_and_category_mapping(files, monkeypatch):
    original_read = Path.read_bytes
    reads = []

    def record(path):
        reads.append(path)
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", record)
    loader = build(files)
    assert not any(path.suffix == ".jpg" for path in reads)
    assert loader.binding["role"] == "train_core"
    assert loader.binding["sample_count"] == 13
    assert loader.state_dict()["dropped_samples"] == 1
    assert loader.state_dict()["batches_per_epoch"] == 3
    start(loader)
    batches = list(loader.preview_batches(1))
    assert len(batches) == 1
    batch = batches[0]
    assert batch.images.shape == (4, 3, 128, 128)
    assert batch.images.dtype == torch.float32
    assert torch.isfinite(batch.images).all()
    assert batch.augmented_sha256 == augmented_tensor_sha256(batch.images, batch.targets)
    for target in batch.targets:
        assert target["boxes"].shape == (len(target["labels"]), 4)
        assert torch.isfinite(target["boxes"]).all()
        assert torch.all((target["labels"] >= 0) & (target["labels"] < 10))
        assert torch.all((target["boxes"] >= 0) & (target["boxes"] <= 1))
    assert sum(path.suffix == ".jpg" for path in reads) == 4
    assert loader.state_dict()["completed_batches"] == 0
    assert not torch.cuda.is_initialized()


def test_main_process_loader_preserves_model_and_dn_rng(files):
    loader = build(files)
    start(loader)
    random.seed(93)
    np.random.seed(94)
    torch.random.default_generator.manual_seed(95)
    before = rng_snapshot()
    rows = list(loader.preview_batches(3))
    after = rng_snapshot()
    assert_rng_equal(before, after)
    assert len(rows) == 3
    assert len({batch.augmented_sha256 for batch in rows}) == 3


@pytest.mark.parametrize("workers", [2, 4])
def test_real_spawn_workers_equal_main_process_augmentation(files, workers):
    parent = build(files, num_workers=0)
    child = build(files, num_workers=workers, prefetch_factor=3)
    assert parent.binding_sha256 == child.binding_sha256
    start(parent)
    start(child)
    reference = list(parent.preview_batches(3))
    random.random()
    np.random.rand(9)
    torch.rand(7)
    observed = list(child.preview_batches(3))
    assert [batch.evidence() for batch in reference] == [batch.evidence() for batch in observed]
    for first, second in zip(reference, observed):
        assert torch.equal(first.images, second.images)
        for t1, t2 in zip(first.targets, second.targets):
            assert t1.keys() == t2.keys()
            assert all(torch.equal(t1[key], t2[key]) for key in t1)


def test_prefetch_cursor_recovery_uses_committed_window_only(files):
    uninterrupted = build(files, num_workers=0)
    interrupted = build(files, num_workers=2, prefetch_factor=4)
    start(uninterrupted)
    start(interrupted)
    expected = consume(uninterrupted)
    iterator = iter(interrupted)
    first = next(iterator)
    assert first.evidence() == expected[0]
    # Workers may already have produced every batch; only successful updates count.
    assert interrupted.state_dict()["completed_batches"] == 0
    interrupted.commit_batch(first, engine_state(interrupted, completed=1))
    saved = interrupted.state_dict()
    assert saved["completed_batches"] == 1
    assert saved["optimizer_updates"] == 1
    iterator.close()
    restored = build(files, num_workers=4, prefetch_factor=1)
    restored.load_state_dict(saved)
    restored.validate_engine_state(saved, engine_state(restored, completed=1))
    assert consume(restored) == expected[1:]


def test_yield_without_commit_is_replayed_after_restore(files):
    loader = build(files)
    start(loader)
    iterator = iter(loader)
    pending = next(iterator)
    state = loader.state_dict()
    iterator.close()
    restored = build(files)
    restored.load_state_dict(state)
    assert list(restored.preview_batches(1))[0].evidence() == pending.evidence()
    with pytest.raises(TrainCoreDataError, match="must be committed"):
        iterator = iter(restored)
        next(iterator)
        next(iterator)


def test_failed_engine_cannot_advance_or_commit_cursor(files):
    loader = build(files)
    start(loader)
    iterator = iter(loader)
    batch = next(iterator)
    previous = loader.state_dict()
    failed = engine_state(loader, completed=1)
    failed["failed"] = True
    with pytest.raises(Exception, match="successful window"):
        loader.commit_batch(batch, failed)
    assert loader.state_dict() == previous
    with pytest.raises(TrainCoreDataError, match="optimizer_updates mismatch"):
        loader.validate_engine_state(previous, engine_state(loader, completed=1))
    iterator.close()


def test_acknowledgment_detects_mutation_and_double_commit(files):
    loader = build(files)
    start(loader)
    iterator = iter(loader)
    batch = next(iterator)
    original = batch.images.clone()
    batch.images[0, 0, 0, 0] += 1
    with pytest.raises(TrainCoreDataError, match="changed after yield"):
        loader.commit_batch(batch, engine_state(loader, completed=1))
    batch.images.copy_(original)
    loader.commit_batch(batch, engine_state(loader, completed=1))
    with pytest.raises(TrainCoreDataError, match="stale"):
        loader.commit_batch(batch, engine_state(loader, completed=1))
    iterator.close()


def test_complete_epoch_and_new_epoch_order_and_recovery(files):
    loader = build(files)
    start(loader)
    first = consume(loader)
    before_finish = loader.state_dict()
    loader.finish_epoch(engine_state(loader, completed=3, active=False))
    finished = loader.state_dict()
    assert finished["optimizer_updates"] == 3
    assert not finished["epoch_active"]
    restored = build(files, num_workers=0)
    restored.load_state_dict(finished)
    start(loader, 2)
    start(restored, 2)
    second = consume(loader)
    assert consume(restored) == second
    assert {tuple(x["image_id"] for x in row["samples"]) for row in first} != {
        tuple(x["image_id"] for x in row["samples"]) for row in second}
    assert before_finish["order_sha256"] != loader.state_dict()["order_sha256"]
    assert loader.state_dict()["optimizer_updates"] == 6


def test_cannot_end_incomplete_epoch(files):
    loader = build(files)
    start(loader)
    with pytest.raises(TrainCoreDataError, match="all complete"):
        loader.finish_epoch(engine_state(loader, active=False, completed=1))


@pytest.mark.parametrize("role", ["test", "confirmatory", "development", "raw_train", None])
def test_role_rejected_before_any_file_io(files, monkeypatch, role):
    monkeypatch.setattr(Path, "read_bytes", lambda _: pytest.fail("forbidden role caused file I/O"))
    with pytest.raises(TrainCoreDataError):
        build(files, role=role)


@pytest.mark.parametrize("field,name", [
    ("annotation_file", "confirmatory_coco.json"),
    ("annotation_file", "test_coco.json"),
    ("manifest_file", "confirmatory_sealed_manifest.json"),
])
def test_forbidden_metadata_name_rejected_before_io(files, monkeypatch, field, name):
    files[field] = files[field].with_name(name)
    monkeypatch.setattr(Path, "read_bytes", lambda _: pytest.fail("forbidden filename caused file I/O"))
    with pytest.raises(TrainCoreDataError, match="named train_core|forbidden data role"):
        build(files)


@pytest.mark.parametrize("segment", ["confirmatory", "test", "VisDrone2019-DET-val",
                                     "VisDrone2019-DET-test-dev"])
def test_forbidden_image_root_rejected_before_metadata_io(files, monkeypatch, segment):
    files["image_root"] = files["image_root"].parent / segment
    monkeypatch.setattr(Path, "read_bytes", lambda _: pytest.fail("forbidden root caused metadata I/O"))
    with pytest.raises(TrainCoreDataError, match="forbidden"):
        build(files)


@pytest.mark.parametrize("logical", ["../image.jpg", "train/images/../image.jpg",
                                      "test/images/a.jpg", "train/images/a/b.jpg",
                                      "train\\images\\a.jpg"])
def test_manifest_cannot_cross_train_core_path_scope(files, logical):
    coco = json.loads(files["annotation_file"].read_bytes())
    coco["images"][0]["file_name"] = logical
    files["annotation_sha256"] = write_json(files["annotation_file"], coco)
    with pytest.raises(TrainCoreDataError, match="train/images"):
        build(files)


def test_manifest_membership_must_match_exactly(files):
    manifest = json.loads(files["manifest_file"].read_bytes())
    manifest["records"].pop()
    files["manifest_sha256"] = write_json(files["manifest_file"], manifest)
    with pytest.raises(TrainCoreDataError, match="incomplete"):
        build(files)


def test_metadata_sha_mismatch_fails_before_raw_image_read(files):
    files["annotation_sha256"] = "0" * 64
    with pytest.raises(TrainCoreDataError, match="SHA-256 mismatch"):
        build(files)


def test_image_bytes_checked_before_decode(files):
    loader = build(files)
    start(loader)
    index = loader._plan(1)[0][0]
    filename = loader.dataset.images[index]["file_name"].split("/")[-1]
    (files["image_root"] / filename).write_bytes(b"not-a-jpeg")
    with pytest.raises(TrainCoreDataError, match="bytes differ"):
        list(loader.preview_batches(1))


def test_safe_symlink_root_is_allowed_and_image_escape_is_not(files, tmp_path):
    original_root = files["image_root"]
    linked_root = tmp_path / "approved_link"
    linked_root.symlink_to(original_root, target_is_directory=True)
    files["image_root"] = linked_root
    loader = build(files)
    assert loader.dataset.image_root == original_root
    start(loader)
    index = loader._plan(1)[0][0]
    filename = loader.dataset.images[index]["file_name"].split("/")[-1]
    source = original_root / filename
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(source.read_bytes())
    source.unlink()
    source.symlink_to(outside)
    with pytest.raises(TrainCoreDataError, match="escapes"):
        list(loader.preview_batches(1))


def test_preview_cap_excludes_later_images_even_with_worker_prefetch(files):
    loader = build(files, num_workers=2, prefetch_factor=4)
    start(loader)
    later_index = loader._plan(1)[1][0]
    later_filename = loader.dataset.images[later_index]["file_name"].split("/")[-1]
    (files["image_root"] / later_filename).write_bytes(b"would-fail-if-prefetched")
    result = list(loader.preview_batches(1))
    assert len(result) == 1
    assert len(result[0].samples) == 4
    assert loader.state_dict()["completed_batches"] == 0


@pytest.mark.parametrize("field,value", [
    ("completed_batches", 4), ("dropped_samples", 0), ("dataset_size", 14),
    ("optimizer_updates", 1), ("epoch_start_optimizer_updates", 1),
    ("epoch_active", 1), ("schema_version", True), ("binding_sha256", "bad"),
])
def test_detached_state_rejects_inconsistent_fields(files, field, value):
    loader = build(files)
    start(loader)
    state = loader.state_dict()
    state[field] = value
    with pytest.raises(TrainCoreDataError):
        validate_loader_state(state)


def test_restore_rejects_other_seed_geometry_and_order(files):
    loader = build(files)
    start(loader)
    state = loader.state_dict()
    for overrides in ({"seed": 42}, {"input_size": 160}, {"logical_batch_size": 2}):
        other = build(files, **overrides)
        with pytest.raises(TrainCoreDataError, match="differs"):
            other.load_state_dict(state)
    state["order_sha256"] = "0" * 64
    with pytest.raises(TrainCoreDataError, match="order hash"):
        loader.load_state_dict(state)


def test_restore_into_live_iterator_rejected_without_mutation(files):
    loader = build(files)
    start(loader)
    state = loader.state_dict()
    iterator = iter(loader)
    next(iterator)
    with pytest.raises(TrainCoreDataError, match="close active"):
        loader.load_state_dict(state)
    assert loader.state_dict() == state
    iterator.close()


def test_epoch_stop_preserves_remaining_flip_and_final_geometry(files):
    active = build(files, augmentation_stop_internal_epoch=117)
    stopped = build(files, augmentation_stop_internal_epoch=0)
    start(active)
    start(stopped)
    a = list(active.preview_batches(3))
    b = list(stopped.preview_batches(3))
    assert any(x.augmented_sha256 != y.augmented_sha256 for x, y in zip(a, b))
    assert all(tuple(batch.images.shape) == (4, 3, 128, 128) for batch in b)
    assert stopped.dataset.epoch == 0


@pytest.mark.parametrize("change", [
    {"input_size": 896}, {"input_size": 129}, {"num_workers": -1},
    {"prefetch_factor": 0}, {"logical_batch_size": 0},
    {"seed": -1}, {"seed": 2**32}, {"multiprocessing_context": "fork"},
])
def test_invalid_data_configuration(change):
    with pytest.raises(TrainCoreDataError):
        TrainCoreDataConfig(**change)


def test_mkl_spawn_conflict_rejected_before_startup(monkeypatch):
    monkeypatch.setattr(np.__config__, "CONFIG", {"Build Dependencies": {"blas": {"name": "mkl-sdl"}}})
    monkeypatch.setenv("MKL_THREADING_LAYER", "INTEL")
    with pytest.raises(TrainCoreDataError, match="explicit MKL_THREADING_LAYER=GNU"):
        validate_worker_environment(TrainCoreDataConfig(num_workers=2))
    monkeypatch.delenv("MKL_THREADING_LAYER")
    with pytest.raises(TrainCoreDataError, match="process startup"):
        validate_worker_environment(TrainCoreDataConfig(num_workers=4))
    monkeypatch.setenv("MKL_THREADING_LAYER", "GNU")
    assert validate_worker_environment(TrainCoreDataConfig())["MKL_THREADING_LAYER"] == "GNU"


def test_fresh_loader_preview_needs_no_engine_and_changes_no_cursor(files):
    loader = build(files)
    fresh = loader.state_dict()
    batch = list(loader.preview_batches(1, epoch=1))[0]
    assert loader.state_dict() == fresh
    assert batch.epoch == 1
    start(loader)
    assert list(loader.preview_batches(1))[0].evidence() == batch.evidence()


def test_metadata_symlink_cannot_redirect_to_another_role_before_read(files, tmp_path, monkeypatch):
    original = files["annotation_file"]
    other = tmp_path / "test_coco.json"
    # All files here are synthetic fixtures, never real test data.
    other.write_bytes(original.read_bytes())
    original.unlink()
    original.symlink_to(other)
    monkeypatch.setattr(Path, "read_bytes", lambda _: pytest.fail("redirected metadata was read"))
    with pytest.raises(TrainCoreDataError, match="named train_core"):
        build(files)


def test_sample_receipt_mutation_rejected(files):
    loader = build(files)
    start(loader)
    iterator = iter(loader)
    batch = next(iterator)
    batch.samples[0]["image_id"] += 1000
    with pytest.raises(TrainCoreDataError, match="receipt changed"):
        loader.commit_batch(batch, engine_state(loader, completed=1))
    assert loader.state_dict()["completed_batches"] == 0
    iterator.close()


def test_wrong_engine_geometry_rejected_before_raw_loading(files):
    loader = build(files)
    state = engine_state(loader)
    state["config"]["expected_input_size"] = [160, 160]
    with pytest.raises(TrainCoreDataError, match="geometry"):
        loader.begin_epoch(1, state)
    assert loader.state_dict()["epoch"] == 0


@pytest.mark.parametrize("categories", [None, [None] * 10,
    [{"id": index} for index in range(1, 10)] + [{"id": True}]])
def test_invalid_category_types_fail_with_data_error(files, categories):
    coco = json.loads(files["annotation_file"].read_bytes())
    coco["categories"] = categories
    files["annotation_sha256"] = write_json(files["annotation_file"], coco)
    with pytest.raises(TrainCoreDataError):
        build(files)
