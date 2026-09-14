"""CPU-only synthetic evidence for checkpoint continuity and rejection gates."""

from copy import deepcopy
import functools
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from sparse_rtdetr.baseline import training_v2b_checkpoint as checkpoint
from sparse_rtdetr.baseline.training_v2b_checkpoint import (
    CheckpointError, inspect_checkpoint, restore_checkpoint, save_checkpoint,
)
from sparse_rtdetr.baseline.training_v2b_engine import AccumulationEngine


class TinyDetector(nn.Module):
    def __init__(self, *, unused=False):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)
        self.bn = nn.BatchNorm2d(4)
        self.dropout = nn.Dropout(0.2)
        self.head = nn.Linear(4, 2)
        self.register_buffer("position_cache", torch.arange(4.0), persistent=False)
        # RT-DETR anchors can legitimately contain infinity.
        self.anchors = torch.tensor([0.5, float("inf")])
        if unused:
            self.unused = nn.Parameter(torch.tensor([1.0]))

    def forward(self, images, targets=None):
        values = self.dropout(self.bn(self.conv(images)).mean(dim=(2, 3)))
        return {"pred_logits": self.head(values)}


class TinyCriterion:
    def __call__(self, outputs, targets):
        labels = torch.stack([target["labels"][0] for target in targets])
        target_shift = torch.stack([target["boxes"].mean() for target in targets])
        logits = outputs["pred_logits"]
        return {
            "classification": F.cross_entropy(logits, labels),
            "box_proxy": 0.2 * F.mse_loss(logits[:, 0], target_shift),
        }


class VendorStyleEMA:
    def __init__(self, model):
        self.module = deepcopy(model).eval()
        self.module.requires_grad_(False)
        self.updates = 0
        self.decay = 0.9
        self.warmups = 2

    def update(self, model):
        self.updates += 1
        with torch.no_grad():
            current = model.state_dict()
            for name, value in self.module.state_dict().items():
                if value.dtype.is_floating_point:
                    value.mul_(self.decay).add_(current[name], alpha=1 - self.decay)

    def state_dict(self):
        return {"module": self.module.state_dict(), "updates": self.updates}

    def load_state_dict(self, state, strict=True):
        self.module.load_state_dict(state["module"], strict=strict)
        self.updates = state["updates"]


class CountedWarmup:
    def __init__(self, duration=2):
        self.duration = duration
        self.steps_done = 0

    def step(self):
        self.steps_done += 1

    def finished(self):
        return self.steps_done >= self.duration

    def state_dict(self):
        return {"duration": self.duration, "steps_done": self.steps_done}

    def load_state_dict(self, state):
        self.duration = state["duration"]
        self.steps_done = state["steps_done"]


BINDING = {
    "initial_weights": {"kind": "synthetic_initial_tensor_identity", "sha256": "1" * 64},
    "data": {"kind": "synthetic_only", "fixture_sha256": "2" * 64},
    "code": {"engineering_source_identity": "3" * 64},
    "config": {
        "schema": "checkpoint_cpu_engineering_v1",
        "group_initial_lrs": [0.01, 0.005],
        "sampler": "explicit_fixture_generator_and_window_cursor",
        "warmup_duration": 2, "scheduler_T_max": 8,
        "amp_dtype": "float32", "physical_batch_size": 4,
        "bn_backward_layout": "cpu_contiguous_cuda_native",
        "accumulation_steps": 2,
    },
}


def make_state(seed=17, *, optimizer_kind="adamw", amsgrad=False, unused=False, multistep=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.default_generator.manual_seed(seed)
    model = TinyDetector(unused=unused)
    params = list(model.parameters())
    optimizer_class = torch.optim.AdamW if optimizer_kind == "adamw" else torch.optim.Adam
    optimizer = optimizer_class(
        [{"params": params[:4], "lr": 0.01, "weight_decay": 0.02},
         {"params": params[4:], "lr": 0.005, "weight_decay": 0.0}],
        lr=0.01, foreach=False, amsgrad=amsgrad,
    )
    ema = VendorStyleEMA(model)
    scheduler = (
        torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000], gamma=0.1)
        if multistep else torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)
    )
    warmup = CountedWarmup()
    engine = AccumulationEngine(
        model, TinyCriterion(), optimizer, physical_batch_size=4,
        accumulation_steps=2, amp_dtype="float32", clip_max_norm=1.0,
        ema=ema, scheduler=scheduler, warmup=warmup,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed + 5)
    return dict(
        model=model, optimizer=optimizer, ema=ema, engine=engine,
        scheduler=scheduler, warmup=warmup,
        extra_rng_generators={"fixture": generator},
    )


def batch(state):
    generator = state["extra_rng_generators"]["fixture"]
    images = torch.randn(8, 3, 4, 4)
    images = images + random.random() + float(np.random.normal())
    images = images + torch.rand(8, 1, 1, 1, generator=generator)
    targets = [
        {"labels": torch.tensor([random.randrange(2)]),
         "boxes": torch.tensor(np.random.rand(1, 4), dtype=torch.float32)}
        for _ in range(8)
    ]
    return images, targets


def advance(state, windows):
    engine = state["engine"]
    logs = []
    for _ in range(windows):
        if not engine.epoch_active:
            engine.begin_epoch(engine.epoch + 1)
        logs.append(engine.train_window(*batch(state)))
        if engine.optimizer_updates % 2 == 0:
            engine.finish_epoch(expected_optimizer_steps=2)
    return logs


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert left.dtype == right.dtype
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    else:
        assert left == right


def trained_checkpoint(tmp_path):
    state = make_state()
    advance(state, 3)
    path = tmp_path / "window-3.pt"
    reference = save_checkpoint(path, **state, binding=BINDING)
    return state, path, reference


def restore(path, reference, state, binding=BINDING):
    return restore_checkpoint(
        path, **state, expected_binding=binding,
        expected_sha256=reference["sha256"],
    )


def test_continuous_equals_rebuild_restore_with_real_adamw_bn_ema_and_all_rng(tmp_path):
    continuous, path, reference = trained_checkpoint(tmp_path)
    assert reference["optimizer_updates"] == 3
    assert reference["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert reference["size_bytes"] == path.stat().st_size
    continuous_logs = advance(continuous, 3)
    expected_random = (
        random.random(), np.random.normal(), torch.rand(5),
        torch.rand(5, generator=continuous["extra_rng_generators"]["fixture"]),
    )
    reconstructed = make_state(seed=999)
    metadata = restore(path, reference, reconstructed)
    assert metadata["epoch_active"] is True
    assert metadata["epoch"] == 2
    assert metadata["microsteps"] == 6
    assert reconstructed["warmup"].steps_done == 3
    assert reconstructed["scheduler"].last_epoch == 1
    assert advance(reconstructed, 3) == continuous_logs
    actual_random = (
        random.random(), np.random.normal(), torch.rand(5),
        torch.rand(5, generator=reconstructed["extra_rng_generators"]["fixture"]),
    )
    for name in ("model", "optimizer", "ema", "engine", "scheduler", "warmup"):
        assert_tree_equal(continuous[name].state_dict(), reconstructed[name].state_dict())
    assert_tree_equal(expected_random, actual_random)
    assert reconstructed["ema"].updates == 6
    assert reconstructed["model"].bn.num_batches_tracked.item() == 12
    assert reconstructed["ema"].module.bn.num_batches_tracked.item() == 0


@pytest.mark.parametrize("boundary_windows", [0, 2])
def test_fresh_and_completed_epoch_boundaries_roundtrip(tmp_path, boundary_windows):
    state = make_state()
    advance(state, boundary_windows)
    path = tmp_path / "boundary.pt"
    reference = save_checkpoint(path, **state, binding=BINDING)
    target = make_state(101)
    metadata = restore(path, reference, target)
    assert metadata["epoch_active"] is False
    assert_tree_equal(state["engine"].state_dict(), target["engine"].state_dict())


@pytest.mark.parametrize("field,value", [("phase", 1), ("failed", True)])
def test_partial_or_failed_engine_is_never_published(tmp_path, field, value):
    state = make_state()
    setattr(state["engine"], field, value)
    path = tmp_path / "invalid.pt"
    with pytest.raises(CheckpointError, match="engine|window|failed"):
        save_checkpoint(path, **state, binding=BINDING)
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_hash_is_checked_before_deserialization_or_mutation(tmp_path, monkeypatch):
    _, path, reference = trained_checkpoint(tmp_path)
    target = make_state()
    original = deepcopy(target["model"].state_dict())
    path.write_bytes(path.read_bytes() + b"corruption")
    def forbidden_load(*args, **kwargs):
        raise AssertionError("torch.load must not run on digest mismatch")
    monkeypatch.setattr(torch, "load", forbidden_load)
    with pytest.raises(CheckpointError, match="SHA256 mismatch"):
        restore(path, reference, target)
    assert_tree_equal(original, target["model"].state_dict())


def test_binding_mismatch_rejected_before_state_load(tmp_path):
    _, path, reference = trained_checkpoint(tmp_path)
    target = make_state()
    original = deepcopy(target["model"].state_dict())
    changed = deepcopy(BINDING)
    changed["data"]["fixture_sha256"] = "4" * 64
    with pytest.raises(CheckpointError, match="binding differs"):
        restore(path, reference, target, binding=changed)
    assert_tree_equal(original, target["model"].state_dict())


@pytest.mark.parametrize("kind", ["shape", "dtype", "key", "cache_shape", "cache_value", "parameter_order", "optimizer_options", "ema_config", "warmup_config", "scheduler_config"])
def test_incompatible_reconstruction_rejected_before_load(tmp_path, kind):
    _, path, reference = trained_checkpoint(tmp_path)
    target = make_state()
    if kind == "shape":
        target["model"].head.weight = nn.Parameter(torch.zeros(3, 4))
    elif kind == "dtype":
        target["model"].double()
    elif kind == "key":
        target["model"].register_buffer("unexpected", torch.ones(1))
    elif kind == "cache_shape":
        target["model"].position_cache = torch.arange(5.0)
    elif kind == "cache_value":
        target["model"].anchors[0] = 0.75
    elif kind == "parameter_order":
        target["optimizer"].param_groups[0]["params"].reverse()
    elif kind == "optimizer_options":
        target["optimizer"].param_groups[0]["weight_decay"] = 0.3
    elif kind == "ema_config":
        target["ema"].decay = 0.8
    elif kind == "warmup_config":
        target["warmup"].duration = 4
    elif kind == "scheduler_config":
        target["scheduler"].T_max = 9
    original = deepcopy(target["model"].state_dict())
    with pytest.raises(CheckpointError):
        restore(path, reference, target)
    assert_tree_equal(original, target["model"].state_dict())


@pytest.mark.parametrize("kind", ["version", "model_key", "model_shape", "model_dtype", "optimizer_ids", "optimizer_moment", "optimizer_step", "ema_updates", "engine_steps", "scheduler_state", "numpy_state", "generator_names", "engine_phase"])
def test_malformed_safe_payload_rejected_even_with_matching_external_digest(tmp_path, kind):
    _, path, _ = trained_checkpoint(tmp_path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if kind == "version":
        payload["schema_version"] = 77
    elif kind == "model_key":
        del payload["model"]["bn.running_mean"]
    elif kind == "model_shape":
        payload["model"]["bn.running_mean"] = torch.zeros(5)
    elif kind == "model_dtype":
        payload["model"]["bn.running_mean"] = torch.zeros(4, dtype=torch.float64)
    elif kind == "optimizer_ids":
        payload["optimizer"]["param_groups"][0]["params"].reverse()
    elif kind == "optimizer_moment":
        first = next(iter(payload["optimizer"]["state"].values()))
        first["exp_avg"] = torch.zeros(999)
    elif kind == "optimizer_step":
        first = next(iter(payload["optimizer"]["state"].values()))
        first["step"] = torch.tensor(1000.0)
    elif kind == "ema_updates":
        payload["ema"]["updates"] += 1
    elif kind == "engine_steps":
        payload["engine"]["microsteps"] += 1
    elif kind == "scheduler_state":
        payload["scheduler"]["state"]["T_max"] += 1
    elif kind == "numpy_state":
        payload["rng"]["numpy"]["keys"] = torch.ones(10, dtype=torch.int64)
    elif kind == "generator_names":
        payload["rng"]["generators"] = {}
    elif kind == "engine_phase":
        payload["engine"]["phase"] = 1
    damaged = tmp_path / "malformed.pt"
    torch.save(payload, damaged)
    reference = {"sha256": hashlib.sha256(damaged.read_bytes()).hexdigest()}
    target = make_state()
    original = deepcopy(target["model"].state_dict())
    with pytest.raises(CheckpointError):
        restore(damaged, reference, target)
    assert_tree_equal(original, target["model"].state_dict())
    assert target["engine"].optimizer_updates == 0


def test_generator_must_be_supplied_and_cpu_checkpoint_never_calls_cuda(tmp_path, monkeypatch):
    source = make_state()
    target = make_state()
    def forbidden_cuda(*args, **kwargs):
        raise AssertionError("CPU checkpoint must not inspect or initialize CUDA")
    for name in ("get_rng_state", "set_rng_state", "get_rng_state_all", "set_rng_state_all", "is_available", "is_initialized", "current_device", "device_count"):
        monkeypatch.setattr(torch.cuda, name, forbidden_cuda)
    path = tmp_path / "cpu.pt"
    reference = save_checkpoint(path, **source, binding=BINDING)
    restore(path, reference, target)
    target["extra_rng_generators"] = {}
    with pytest.raises(CheckpointError, match="generator names"):
        restore(path, reference, target)


def test_no_overwrite_and_atomic_publish_race_cleanup(tmp_path, monkeypatch):
    state = make_state()
    existing = tmp_path / "existing.pt"
    existing.write_bytes(b"historical sentinel")
    with pytest.raises(CheckpointError, match="overwrite"):
        save_checkpoint(existing, **state, binding=BINDING)
    assert existing.read_bytes() == b"historical sentinel"
    original_link = checkpoint.os.link
    def race(source, destination):
        destination.write_bytes(b"concurrent writer")
        original_link(source, destination)
    monkeypatch.setattr(checkpoint.os, "link", race)
    target = tmp_path / "raced.pt"
    with pytest.raises(CheckpointError, match="save failed"):
        save_checkpoint(target, **state, binding=BINDING)
    assert target.read_bytes() == b"concurrent writer"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["existing.pt", "raced.pt"]


def test_binding_digest_matches_evidence_canonicalization(tmp_path):
    state = make_state()
    binding = deepcopy(BINDING)
    binding["binding_sha256"] = hashlib.sha256(json.dumps(
        binding, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode()).hexdigest()
    reference = save_checkpoint(tmp_path / "bound.pt", **state, binding=binding)
    assert reference["binding_sha256"] == binding["binding_sha256"]
    binding["config"]["group_initial_lrs"][0] = 0.2
    with pytest.raises(CheckpointError, match="declared digest"):
        save_checkpoint(tmp_path / "bad.pt", **state, binding=binding)


class UnsupportedPickle:
    pass


def test_safe_loader_rejects_arbitrary_python_objects(tmp_path):
    path = tmp_path / "untrusted.pt"
    torch.save({"object": UnsupportedPickle()}, path)
    reference = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    with pytest.raises(CheckpointError, match="restore failed"):
        restore(path, reference, make_state())


def test_ema_counter_mismatch_is_rejected_at_save(tmp_path):
    state = make_state()
    state["ema"].updates = 1
    with pytest.raises(CheckpointError, match="EMA update"):
        save_checkpoint(tmp_path / "mismatch.pt", **state, binding=BINDING)


def test_missing_binding_identity_is_rejected(tmp_path):
    with pytest.raises(CheckpointError, match="binding requires"):
        save_checkpoint(tmp_path / "unbound.pt", **make_state(), binding={"config": {"x": 1}})


@pytest.mark.parametrize("component", ["model", "optimizer", "ema", "scheduler", "warmup"])
def test_engine_component_identity_is_required(tmp_path, component):
    state = make_state()
    unrelated = make_state(104)
    state[component] = unrelated[component]
    with pytest.raises(CheckpointError, match="supplied checkpoint component"):
        save_checkpoint(tmp_path / "unrelated.pt", **state, binding=BINDING)


def test_legitimate_zero_gradient_update_can_be_saved_and_restored(tmp_path):
    state = make_state()
    state["engine"].criterion = lambda outputs, targets: {"zero": outputs["pred_logits"].sum() * 0}
    state["engine"].begin_epoch(1)
    observation = state["engine"].train_window(*batch(state))
    assert observation["gradient_norm_before_clip"] == 0.0
    reference = save_checkpoint(tmp_path / "zero.pt", **state, binding=BINDING)
    restored = make_state(44)
    restore(tmp_path / "zero.pt", reference, restored)
    assert restored["engine"].optimizer_updates == restored["ema"].updates == 1
    assert_tree_equal(state["optimizer"].state_dict(), restored["optimizer"].state_dict())


def test_engine_input_size_identity_is_required(tmp_path):
    _, path, reference = trained_checkpoint(tmp_path)
    restored = make_state(45)
    restored["engine"].expected_input_size = (4, 4)
    with pytest.raises(CheckpointError, match="engine type/config"):
        restore(path, reference, restored)


@pytest.mark.parametrize("operation", ["save", "restore"])
def test_actual_vendor_linear_warmup_clock_is_cross_checked(tmp_path, monkeypatch, operation):
    vendor = Path(__file__).resolve().parents[1] / "vendor" / "rtdetrv2_pytorch"
    monkeypatch.syspath_prepend(str(vendor))
    from src.optim.warmup import LinearWarmup

    state = make_state()
    state["warmup"] = LinearWarmup(state["scheduler"], warmup_duration=2)
    state["engine"].warmup = state["warmup"]
    advance(state, 3)
    assert state["warmup"].last_step == state["engine"].optimizer_updates == 3
    path = tmp_path / "linear-warmup.pt"
    if operation == "save":
        state["warmup"].last_step = 2
        with pytest.raises(CheckpointError, match="LinearWarmup last_step"):
            save_checkpoint(path, **state, binding=BINDING)
        assert not path.exists()
        return
    reference = save_checkpoint(path, **state, binding=BINDING)
    target = make_state(99)
    target["warmup"] = LinearWarmup(target["scheduler"], warmup_duration=2)
    target["engine"].warmup = target["warmup"]
    restore(path, reference, target)
    assert target["warmup"].last_step == 3
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["warmup"]["state"]["last_step"] = 2
    bad = tmp_path / "linear-warmup-corrupt.pt"
    torch.save(payload, bad)
    with pytest.raises(CheckpointError, match="LinearWarmup last_step"):
        restore(bad, {"sha256": hashlib.sha256(bad.read_bytes()).hexdigest()}, target)


@pytest.mark.parametrize("optimizer_kind,amsgrad,damage", [
    ("adamw", False, "step"), ("adamw", False, "exp_avg"),
    ("adamw", False, "exp_avg_sq"), ("adamw", True, "max_exp_avg_sq"),
    ("adam", False, "step"), ("adam", True, "max_exp_avg_sq"),
    ("adamw", False, "step_nan"), ("adamw", False, "step_fraction"),
    ("adamw", False, "step_shape"), ("adamw", False, "step_dtype"),
    ("adamw", False, "moment_nan"), ("adamw", False, "moment_dtype"),
    ("adamw", False, "moment_shape"), ("adamw", False, "variance_negative"),
    ("adamw", True, "max_moment_infinite"),
])
def test_adam_state_corruption_rejected_before_any_live_loader(tmp_path, monkeypatch, optimizer_kind, amsgrad, damage):
    state = make_state(optimizer_kind=optimizer_kind, amsgrad=amsgrad)
    advance(state, 3)
    good = tmp_path / "adam-good.pt"
    save_checkpoint(good, **state, binding=BINDING)
    payload = torch.load(good, map_location="cpu", weights_only=True)
    first = next(value for value in payload["optimizer"]["state"].values() if value)
    if damage in {"step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"}:
        del first[damage]
    elif damage == "step_nan":
        first["step"] = torch.tensor(float("nan"))
    elif damage == "step_fraction":
        first["step"] = torch.tensor(1.5)
    elif damage == "step_shape":
        first["step"] = torch.tensor([3.0])
    elif damage == "step_dtype":
        first["step"] = torch.tensor(3, dtype=torch.int64)
    elif damage == "moment_nan":
        first["exp_avg"].reshape(-1)[0] = float("nan")
    elif damage == "moment_dtype":
        first["exp_avg"] = first["exp_avg"].double()
    elif damage == "moment_shape":
        first["exp_avg"] = torch.zeros(999)
    elif damage == "variance_negative":
        first["exp_avg_sq"].reshape(-1)[0] = -1.0
    elif damage == "max_moment_infinite":
        first["max_exp_avg_sq"].reshape(-1)[0] = float("inf")
    bad = tmp_path / "adam-corrupt.pt"
    torch.save(payload, bad)
    target = make_state(93, optimizer_kind=optimizer_kind, amsgrad=amsgrad)
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("no live loader may run on invalid optimizer state")
    for name in ("model", "ema", "optimizer", "scheduler", "warmup", "engine"):
        monkeypatch.setattr(type(target[name]), "load_state_dict", forbidden)
    with pytest.raises(CheckpointError):
        restore(bad, {"sha256": hashlib.sha256(bad.read_bytes()).hexdigest()}, target)
    assert calls == []
    assert target["engine"].failed is False


@pytest.mark.parametrize("optimizer_kind", ["adam", "adamw"])
@pytest.mark.parametrize("amsgrad", [False, True])
@pytest.mark.parametrize("explicit_empty", [False, True])
def test_unused_parameters_may_have_absent_or_empty_adam_state(tmp_path, optimizer_kind, amsgrad, explicit_empty):
    state = make_state(optimizer_kind=optimizer_kind, amsgrad=amsgrad, unused=True)
    if explicit_empty:
        state["optimizer"].state[state["model"].unused]
    advance(state, 3)
    assert not state["optimizer"].state.get(state["model"].unused)
    path = tmp_path / "unused.pt"
    reference = save_checkpoint(path, **state, binding=BINDING)
    target = make_state(92, optimizer_kind=optimizer_kind, amsgrad=amsgrad, unused=True)
    restore(path, reference, target)
    assert_tree_equal(state["optimizer"].state_dict(), target["optimizer"].state_dict())
    assert advance(target, 1)


def test_save_rejects_incomplete_nonempty_adam_state(tmp_path):
    state = make_state()
    advance(state, 1)
    del next(iter(state["optimizer"].state.values()))["step"]
    with pytest.raises(CheckpointError, match="missing or unexpected fields"):
        save_checkpoint(tmp_path / "incomplete.pt", **state, binding=BINDING)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("damage", ["inactive_zero_updates", "too_many_epochs"])
def test_engine_epoch_corruption_rejected_before_any_live_loader(tmp_path, monkeypatch, damage):
    _, path, _ = trained_checkpoint(tmp_path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["engine"]["epoch_active"] = False
    if damage == "inactive_zero_updates":
        payload["engine"]["epoch_start_optimizer_updates"] = payload["engine"]["optimizer_updates"]
    else:
        payload["engine"]["epoch"] = 5
        payload["engine"]["epoch_start_optimizer_updates"] = 0
    bad = tmp_path / "engine-corrupt.pt"
    torch.save(payload, bad)
    reference = {"sha256": hashlib.sha256(bad.read_bytes()).hexdigest()}
    target = make_state(91)
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("no live loader may run on invalid engine state")
    for name in ("model", "ema", "optimizer", "scheduler", "warmup", "engine"):
        monkeypatch.setattr(type(target[name]), "load_state_dict", forbidden)
    with pytest.raises(CheckpointError, match="engine state is invalid"):
        restore(bad, reference, target)
    assert calls == []
    with pytest.raises(CheckpointError, match="engine state is invalid"):
        inspect_checkpoint(bad, expected_sha256=reference["sha256"], expected_binding=BINDING)


def test_unexpected_loader_failure_invalidates_engine(tmp_path, monkeypatch):
    _, path, reference = trained_checkpoint(tmp_path)
    target = make_state(90)
    def unexpected(*args, **kwargs):
        raise RuntimeError("synthetic unexpected custom loader failure")
    monkeypatch.setattr(target["optimizer"], "load_state_dict", unexpected)
    with pytest.raises(CheckpointError, match="reconstruction invalidated"):
        restore(path, reference, target)
    assert target["engine"].failed is True


def test_inspector_reads_authenticated_internal_counts_without_live_construction_or_cuda(tmp_path, monkeypatch):
    _, path, reference = trained_checkpoint(tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError("inspection must not construct live components or touch CUDA")
    monkeypatch.setattr(torch.nn.Module, "__init__", forbidden)
    monkeypatch.setattr(torch.optim.Optimizer, "__init__", forbidden)
    for name in ("get_rng_state", "set_rng_state", "is_available", "is_initialized", "current_device", "device_count"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    inspected = inspect_checkpoint(path, expected_sha256=reference["sha256"], expected_binding=BINDING)
    assert inspected["engine"]["epoch"] == 2
    assert inspected["engine"]["epoch_active"] is True
    assert inspected["optimizer_updates"] == 3
    assert inspected["microsteps"] == 6
    assert inspected["samples_seen"] == 24
    assert inspected["binding"] == BINDING
    assert inspected["binding_sha256"] == reference["binding_sha256"]
    assert inspected["inspection_scope"] == "stored_checkpoint_identity_and_counters"


@pytest.mark.parametrize("damage", ["negative_epoch", "future_epoch", "step_count", "unfinished_warmup"])
def test_multistep_clock_corruption_rejected_before_live_load(tmp_path, monkeypatch, damage):
    vendor = Path(__file__).resolve().parents[1] / "vendor" / "rtdetrv2_pytorch"
    monkeypatch.syspath_prepend(str(vendor))
    from src.optim.warmup import LinearWarmup

    duration = 100 if damage == "unfinished_warmup" else 2
    state = make_state(multistep=True)
    state["warmup"] = LinearWarmup(state["scheduler"], warmup_duration=duration)
    state["engine"].warmup = state["warmup"]
    advance(state, 3)
    good = tmp_path / "multistep.pt"
    reference = save_checkpoint(good, **state, binding=BINDING)
    inspected = inspect_checkpoint(good, expected_sha256=reference["sha256"], expected_binding=BINDING)
    assert inspected["optimizer_updates"] == 3
    payload = torch.load(good, map_location="cpu", weights_only=True)
    scheduler = payload["scheduler"]["state"]
    if damage == "negative_epoch":
        scheduler["last_epoch"] = -1
        scheduler["_step_count"] = 0
    elif damage == "future_epoch":
        scheduler["last_epoch"] = 2
        scheduler["_step_count"] = 3
    elif damage == "step_count":
        scheduler["_step_count"] = 99
    else:
        scheduler["last_epoch"] = 1
        scheduler["_step_count"] = 2
    bad = tmp_path / "multistep-corrupt.pt"
    torch.save(payload, bad)
    target = make_state(89, multistep=True)
    target["warmup"] = LinearWarmup(target["scheduler"], warmup_duration=duration)
    target["engine"].warmup = target["warmup"]
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("scheduler clock preflight must precede all live loads")
    for name in ("model", "ema", "optimizer", "scheduler", "warmup", "engine"):
        monkeypatch.setattr(type(target[name]), "load_state_dict", forbidden)
    with pytest.raises(CheckpointError, match="MultiStepLR"):
        restore(bad, {"sha256": hashlib.sha256(bad.read_bytes()).hexdigest()}, target)
    assert calls == []


@pytest.mark.parametrize("legacy_policy", [None, "cpu_native_cuda_native"])
def test_legacy_bn_backward_policy_is_rejected_before_live_load(tmp_path, monkeypatch, legacy_policy):
    _, path, reference = trained_checkpoint(tmp_path)
    inspected = inspect_checkpoint(path, expected_sha256=reference["sha256"], expected_binding=BINDING)
    assert inspected["engine"]["config"]["bn_backward_layout"] == "cpu_contiguous_cuda_native"
    assert inspected["binding"]["config"]["bn_backward_layout"] == "cpu_contiguous_cuda_native"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if legacy_policy is None:
        del payload["engine"]["config"]["bn_backward_layout"]
    else:
        payload["engine"]["config"]["bn_backward_layout"] = legacy_policy
    bad = tmp_path / "legacy-bn-policy.pt"
    torch.save(payload, bad)
    bad_sha = hashlib.sha256(bad.read_bytes()).hexdigest()
    target = make_state(88)
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("legacy BN semantics must not load any live state")
    for name in ("model", "ema", "optimizer", "scheduler", "warmup", "engine"):
        monkeypatch.setattr(type(target[name]), "load_state_dict", forbidden)
    with pytest.raises(CheckpointError, match="engine state is invalid"):
        restore(bad, {"sha256": bad_sha}, target)
    assert calls == []
    assert target["engine"].failed is False
    with pytest.raises(CheckpointError, match="engine state is invalid"):
        inspect_checkpoint(bad, expected_sha256=bad_sha, expected_binding=BINDING)


# Schema 2 adds runtime/topology and acknowledged-loader identity. These tests
# use only CPU tensors; CUDA behavior below is explicitly simulated.
from test_rtdetr_baseline_training_v2b_device import (
    GPU_UUID, fake_cuda, forbid_cuda,
)


def write_payload(tmp_path, payload, name="edited.pt"):
    path = tmp_path / name
    torch.save(payload, path)
    return path, {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def test_new_schema_records_cpu_runtime_and_legacy_cpu_artifact_remains_readable(tmp_path, monkeypatch):
    source, path, reference = trained_checkpoint(tmp_path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["schema_version"] == 2
    assert payload["runtime"]["device"] == "cpu"
    assert payload["sampler"] is None
    legacy = deepcopy(payload)
    legacy["schema_version"] = 1
    del legacy["runtime"], legacy["sampler"]
    old_path, old_ref = write_payload(tmp_path, legacy, "legacy-cpu.pt")
    target = make_state(87)
    forbid_cuda(monkeypatch)
    inspected = inspect_checkpoint(old_path, expected_sha256=old_ref["sha256"], expected_binding=BINDING)
    restored = restore(old_path, old_ref, target)
    assert inspected["schema_version"] == restored["schema_version"] == 1
    for key in ("model", "optimizer", "ema", "engine", "scheduler", "warmup"):
        assert_tree_equal(source[key].state_dict(), target[key].state_dict())


def test_legacy_cuda_without_uuid_binding_is_explicitly_rejected(tmp_path):
    _, path, _ = trained_checkpoint(tmp_path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["schema_version"] = 1
    del payload["runtime"], payload["sampler"]
    payload["rng"]["cuda"] = {"0": torch.zeros(16, dtype=torch.uint8)}
    old_path, old_ref = write_payload(tmp_path, payload, "legacy-cuda.pt")
    with pytest.raises(CheckpointError, match="legacy CUDA.*UUID"):
        inspect_checkpoint(old_path, expected_sha256=old_ref["sha256"], expected_binding=BINDING)
    with pytest.raises(CheckpointError, match="legacy CUDA.*UUID"):
        restore(old_path, old_ref, make_state())


def test_entire_cpu_checkpoint_cycle_has_no_cuda_api_calls(tmp_path, monkeypatch):
    source = make_state()
    advance(source, 1)
    target = make_state(99)
    from sparse_rtdetr.baseline.training_v2b_device import prepare_runtime
    forbid_cuda(monkeypatch)
    runtime = prepare_runtime(seed=17)
    path = tmp_path / "cuda-forbidden.pt"
    reference = save_checkpoint(path, **source, binding=BINDING, runtime=runtime)
    inspect_checkpoint(path, expected_sha256=reference["sha256"], expected_binding=BINDING)
    restore_checkpoint(
        path, **target, expected_sha256=reference["sha256"],
        expected_binding=BINDING, runtime=runtime,
    )
    assert_tree_equal(source["model"].state_dict(), target["model"].state_dict())


class AcknowledgedSyntheticSampler:
    """State-contract double; no datasets, image access, or workers are involved."""
    binding_sha256 = "e" * 64

    def __init__(self, engine_state):
        self.record(engine_state)

    def record(self, engine_state):
        epoch = engine_state["epoch"]
        start = engine_state["epoch_start_optimizer_updates"]
        self.state = {
            "schema_version": 1, "binding_sha256": self.binding_sha256,
            "epoch": epoch, "epoch_active": engine_state["epoch_active"],
            "epoch_start_optimizer_updates": start,
            "optimizer_updates": engine_state["optimizer_updates"],
            "completed_batches": engine_state["optimizer_updates"] - start,
            "logical_batch_size": 8, "dataset_size": 17,
            "batches_per_epoch": 2, "dropped_samples": 1,
            "order_sha256": hashlib.sha256(b"[]").hexdigest() if epoch == 0 else "f" * 64,
        }

    def state_dict(self):
        return deepcopy(self.state)

    def validate_state_dict(self, state):
        from sparse_rtdetr.baseline.training_v2b_data import validate_loader_state
        validate_loader_state(state)
        if state["binding_sha256"] != self.binding_sha256:
            raise ValueError("synthetic sampler binding mismatch")

    def validate_engine_state(self, state, engine_state):
        from sparse_rtdetr.baseline.training_v2b_data import validate_loader_state
        self.validate_state_dict(state)
        validate_loader_state(state, engine_state)

    def load_state_dict(self, state):
        self.validate_state_dict(state)
        self.state = deepcopy(state)


def sampler_binding():
    bound = deepcopy(BINDING)
    bound["data"]["loader_binding_sha256"] = AcknowledgedSyntheticSampler.binding_sha256
    return bound


def test_checkpoint_inspects_and_restores_acknowledged_sampler_cursor(tmp_path, monkeypatch):
    source = make_state()
    advance(source, 3)
    source["sampler"] = AcknowledgedSyntheticSampler(source["engine"].state_dict())
    bound = sampler_binding()
    path = tmp_path / "acknowledged-cursor.pt"
    reference = save_checkpoint(path, **source, binding=bound)
    expected_log = advance(source, 1)
    target = make_state(19)
    target["sampler"] = AcknowledgedSyntheticSampler(target["engine"].state_dict())
    with monkeypatch.context() as safety:
        forbid_cuda(safety)
        info = inspect_checkpoint(path, expected_sha256=reference["sha256"], expected_binding=bound)
        assert info["sampler"]["state"]["completed_batches"] == 1
        assert info["sampler"]["state"]["optimizer_updates"] == info["optimizer_updates"] == 3
        result = restore(path, reference, target, binding=bound)
        assert result["sampler_restored"] is True
        assert target["sampler"].state_dict() == source["sampler"].state_dict()
    assert expected_log == advance(target, 1)


@pytest.mark.parametrize("kind", ["missing", "unbound", "wrong_binding", "unacknowledged_update"])
def test_bound_sampler_must_match_acknowledged_engine_state_at_save(tmp_path, kind):
    state = make_state()
    state["sampler"] = AcknowledgedSyntheticSampler(state["engine"].state_dict())
    bound = sampler_binding()
    if kind == "missing":
        del state["sampler"]
    elif kind == "unbound":
        del bound["data"]["loader_binding_sha256"]
    elif kind == "wrong_binding":
        bound["data"]["loader_binding_sha256"] = "a" * 64
    else:
        advance(state, 1)  # Model update happened, loader cursor was not committed.
    path = tmp_path / "invalid-cursor.pt"
    with pytest.raises(CheckpointError, match="loader|sampler"):
        save_checkpoint(path, **state, binding=bound)
    assert not path.exists()


@pytest.mark.parametrize("kind", ["missing_target", "tampered_cursor", "tampered_identity"])
def test_sampler_restore_preflight_rejects_before_live_model_mutation(tmp_path, monkeypatch, kind):
    source = make_state()
    advance(source, 1)
    source["sampler"] = AcknowledgedSyntheticSampler(source["engine"].state_dict())
    bound = sampler_binding()
    path = tmp_path / "sampler-good.pt"
    reference = save_checkpoint(path, **source, binding=bound)
    if kind != "missing_target":
        payload = torch.load(path, map_location="cpu", weights_only=True)
        key = "completed_batches" if kind == "tampered_cursor" else "binding_sha256"
        payload["sampler"]["state"][key] = 0 if kind == "tampered_cursor" else "a" * 64
        path, reference = write_payload(tmp_path, payload, "sampler-bad.pt")
    target = make_state()
    if kind != "missing_target":
        target["sampler"] = AcknowledgedSyntheticSampler(target["engine"].state_dict())
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("invalid cursor must precede every live load")
    for key in ("model", "optimizer", "ema", "engine", "scheduler", "warmup"):
        monkeypatch.setattr(type(target[key]), "load_state_dict", forbidden)
    with pytest.raises(CheckpointError):
        restore(path, reference, target, binding=bound)
    assert calls == []


def test_optimizer_post_restore_placement_failure_poisoned_before_resume(tmp_path, monkeypatch):
    _, path, reference = trained_checkpoint(tmp_path)
    target = make_state()
    loader = target["optimizer"].load_state_dict
    def misplaced(state):
        loader(state)
        first = next(iter(target["optimizer"].state.values()))
        first["exp_avg"] = torch.empty_like(first["exp_avg"], device="meta")
    monkeypatch.setattr(target["optimizer"], "load_state_dict", misplaced)
    with pytest.raises(CheckpointError, match="reconstruction invalidated"):
        restore(path, reference, target)
    assert target["engine"].failed is True


def prepare_checkpoint_fake_cuda(fake_cuda, monkeypatch, binding):
    from sparse_rtdetr.baseline import training_v2b_hardware as hardware
    from sparse_rtdetr.baseline.training_v2b_device import prepare_runtime
    def admission(*args, **kwargs):
        assert kwargs["binding"] == binding
        assert kwargs["expected_gpu_uuid"] == GPU_UUID
        fake_cuda.calls.append(("admission",))
        return {"gpu": {"uuid": GPU_UUID}}
    monkeypatch.setattr(hardware, "require_native_hardware_admission", admission)
    return prepare_runtime(
        device="cuda:0", seed=17, binding=binding,
        gpu_probe="explicit-test-double", expected_gpu_uuid=GPU_UUID,
    )


def gpu_mock_binding():
    bound = deepcopy(BINDING)
    bound["config"].update(device="cuda:0", cuda_gpu_uuid=GPU_UUID)
    return bound


def test_mock_cuda_checkpoint_serializes_cpu_rng_bytes_and_restores_before_resumed_work(tmp_path, fake_cuda, monkeypatch):
    # Model/optimizer execution remains CPU. Device placement is isolated from
    # this test of checkpoint serialization, RNG control flow, and restoration.
    source = make_state()
    advance(source, 3)
    bound = gpu_mock_binding()
    runtime = prepare_checkpoint_fake_cuda(fake_cuda, monkeypatch, bound)
    monkeypatch.setattr(checkpoint, "_check_placement", lambda *args: None)
    path = tmp_path / "mock-cuda-rng.pt"
    reference = save_checkpoint(path, **source, binding=bound, runtime=runtime)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["runtime"]["cuda_devices"][0]["uuid"] == GPU_UUID
    assert payload["rng"]["cuda"]["0"].device.type == "cpu"
    assert payload["rng"]["cuda"]["0"].dtype == torch.uint8
    expected_cuda_draw = torch.rand(9, generator=fake_cuda.default_generators[0].generator)
    target = make_state(82)
    result = restore_checkpoint(
        path, **target, expected_binding=bound,
        expected_sha256=reference["sha256"], runtime=runtime,
    )
    assert result["runtime"] == runtime.identity
    replay_cuda_draw = torch.rand(9, generator=fake_cuda.default_generators[0].generator)
    assert torch.equal(expected_cuda_draw, replay_cuda_draw)
    assert_tree_equal(source["optimizer"].state_dict(), target["optimizer"].state_dict())


def test_mock_cuda_artifact_inspection_uses_no_cuda_api(tmp_path, fake_cuda, monkeypatch):
    source = make_state()
    bound = gpu_mock_binding()
    runtime = prepare_checkpoint_fake_cuda(fake_cuda, monkeypatch, bound)
    monkeypatch.setattr(checkpoint, "_check_placement", lambda *args: None)
    path = tmp_path / "mock-cuda-inspect.pt"
    reference = save_checkpoint(path, **source, binding=bound, runtime=runtime)
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU inspection of a CUDA artifact must never initialize/query CUDA")
    for name in ("is_initialized", "get_rng_state", "set_rng_state", "current_device", "device_count", "init"):
        monkeypatch.setattr(fake_cuda, name, forbidden)
    inspected = inspect_checkpoint(path, expected_sha256=reference["sha256"], expected_binding=bound)
    assert inspected["runtime"]["device"] == "cuda:0"


@pytest.mark.parametrize("kind", ["uuid", "ordinal", "build", "backend", "rng_shape"])
def test_mock_cuda_checkpoint_remapping_or_runtime_mismatch_rejected_before_live_load(tmp_path, fake_cuda, monkeypatch, kind):
    source = make_state()
    bound = gpu_mock_binding()
    runtime = prepare_checkpoint_fake_cuda(fake_cuda, monkeypatch, bound)
    monkeypatch.setattr(checkpoint, "_check_placement", lambda *args: None)
    path = tmp_path / "mock-cuda-good.pt"
    save_checkpoint(path, **source, binding=bound, runtime=runtime)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    identity = payload["runtime"]
    if kind == "uuid":
        identity["cuda_devices"][0]["uuid"] = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        identity["cuda_visible_devices"] = identity["cuda_devices"][0]["uuid"]
    elif kind == "ordinal":
        identity["device"] = "cuda:1"
        identity["cuda_devices"][0]["index"] = 1
    elif kind == "build":
        identity["cuda_build_version"] = "another-build"
    elif kind == "backend":
        identity["backend_policy"]["deterministic_algorithms"] = not identity["backend_policy"]["deterministic_algorithms"]
    else:
        payload["rng"]["cuda"]["0"] = torch.zeros(8, dtype=torch.uint8)
    path, reference = write_payload(tmp_path, payload, "mock-cuda-bad.pt")
    target = make_state(81)
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("runtime mismatch must precede live model mutation")
    for key in ("model", "optimizer", "ema", "engine", "scheduler", "warmup"):
        monkeypatch.setattr(type(target[key]), "load_state_dict", forbidden)
    with pytest.raises(CheckpointError):
        restore_checkpoint(
            path, **target, expected_binding=bound,
            expected_sha256=reference["sha256"], runtime=runtime,
        )
    assert calls == []


def test_mock_cuda_restore_api_failure_poisoned_before_a_forward_can_continue(tmp_path, fake_cuda, monkeypatch):
    source = make_state()
    bound = gpu_mock_binding()
    runtime = prepare_checkpoint_fake_cuda(fake_cuda, monkeypatch, bound)
    monkeypatch.setattr(checkpoint, "_check_placement", lambda *args: None)
    path = tmp_path / "mock-cuda-late-failure.pt"
    reference = save_checkpoint(path, **source, binding=bound, runtime=runtime)
    target = make_state(80)
    def failed(*args, **kwargs):
        raise RuntimeError("simulated native CUDA set_rng_state failure")
    monkeypatch.setattr(fake_cuda, "set_rng_state", failed)
    with pytest.raises(CheckpointError, match="reconstruction invalidated"):
        restore_checkpoint(
            path, **target, expected_binding=bound,
            expected_sha256=reference["sha256"], runtime=runtime,
        )
    assert target["engine"].failed is True



def persistent_geometry(state):
    for model in (state["model"], state["ema"].module):
        anchors = model.anchors
        del model.anchors
        model.register_buffer("anchors", anchors)
        model.register_buffer("valid_mask", torch.tensor([True, False]))
        model.register_buffer("num_points_scale", torch.tensor([0.25, 0.25]))
    return state


def test_persistent_geometry_identity_is_checked_and_legacy_cpu_migration_is_exact(tmp_path):
    source = persistent_geometry(make_state())
    advance(source, 1)
    path = tmp_path / "geometry.pt"
    reference = save_checkpoint(path, **source, binding=BINDING)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert ":persistent_cache:anchors" in payload["model_layout"]["caches"]
    legacy = deepcopy(payload)
    legacy["schema_version"] = 1
    del legacy["runtime"], legacy["sampler"]
    for layout in (legacy["model_layout"], legacy["ema_layout"]["model"]):
        layout["caches"] = {key: value for key, value in layout["caches"].items()
                            if ":persistent_cache:" not in key}
    old, old_ref = write_payload(tmp_path, legacy, "legacy-geometry.pt")
    target = persistent_geometry(make_state(79))
    restore(old, old_ref, target)
    assert_tree_equal(source["model"].state_dict(), target["model"].state_dict())
    incompatible = persistent_geometry(make_state(78))
    incompatible["model"].anchors[0] = 0.75
    with pytest.raises(CheckpointError, match="cache identity"):
        restore(old, old_ref, incompatible)


@pytest.mark.parametrize("family", ["model", "ema"])
@pytest.mark.parametrize("cache", ["anchors", "valid_mask", "num_points_scale"])
def test_persistent_cache_payload_cannot_silently_replace_configured_geometry(tmp_path, family, cache):
    source = persistent_geometry(make_state())
    path = tmp_path / "geometry-good.pt"
    save_checkpoint(path, **source, binding=BINDING)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload["model"] if family == "model" else payload["ema"]["module"]
    state[cache][0] = False if cache == "valid_mask" else 0.75
    bad, reference = write_payload(tmp_path, payload, "geometry-bad.pt")
    with pytest.raises(CheckpointError, match="persistent geometry"):
        inspect_checkpoint(bad, expected_sha256=reference["sha256"], expected_binding=BINDING)
    with pytest.raises(CheckpointError, match="persistent geometry"):
        restore(bad, reference, persistent_geometry(make_state(77)))


@pytest.mark.parametrize("family,name,damage", [
    ("model", "conv.weight", float("nan")),
    ("model", "bn.running_mean", float("inf")),
    ("ema", "head.weight", float("-inf")),
    ("ema", "bn.running_var", float("nan")),
    ("model", "bn.num_batches_tracked", -1),
    ("model", "bn.running_var", -0.1),
])
def test_nonfinite_parameters_mutable_buffers_and_negative_bn_counters_are_rejected(tmp_path, family, name, damage):
    _, path, _ = trained_checkpoint(tmp_path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload["model"] if family == "model" else payload["ema"]["module"]
    state[name].reshape(-1)[0] = damage
    bad, reference = write_payload(tmp_path, payload, "nonfinite.pt")
    with pytest.raises(CheckpointError, match="non-finite|negative BatchNorm"):
        inspect_checkpoint(bad, expected_sha256=reference["sha256"], expected_binding=BINDING)
    target = make_state(76)
    before = deepcopy(target["model"].state_dict())
    with pytest.raises(CheckpointError, match="non-finite|negative BatchNorm"):
        restore(bad, reference, target)
    assert_tree_equal(before, target["model"].state_dict())


def test_loader_restore_specific_preflight_runs_before_any_live_component_mutation(tmp_path, monkeypatch):
    source = make_state()
    advance(source, 1)
    source["sampler"] = AcknowledgedSyntheticSampler(source["engine"].state_dict())
    bound = sampler_binding()
    path = tmp_path / "active-cursor.pt"
    reference = save_checkpoint(path, **source, binding=bound)
    target = make_state(75)
    target["sampler"] = AcknowledgedSyntheticSampler(target["engine"].state_dict())
    def active_iterator(state):
        raise ValueError("close active loader iterator before restoring")
    target["sampler"].validate_restore_state_dict = active_iterator
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("active target loader must be rejected before model mutation")
    for key in ("model", "optimizer", "ema", "engine", "scheduler", "warmup"):
        monkeypatch.setattr(type(target[key]), "load_state_dict", forbidden)
    with pytest.raises(CheckpointError, match="active loader iterator"):
        restore(path, reference, target, binding=bound)
    assert calls == []


def test_cpu_checkpoint_rejects_outside_generator_device_before_get_state(tmp_path, monkeypatch):
    from types import SimpleNamespace
    source = make_state()
    calls = []
    def forbidden_state():
        calls.append(True)
        raise AssertionError("CPU checkpoint must not inspect an outside CUDA generator")
    synthetic_generator = SimpleNamespace(device=torch.device("cuda:0"), get_state=forbidden_state)
    monkeypatch.setattr(checkpoint, "_generators", lambda value: {"outside": synthetic_generator})
    with pytest.raises(CheckpointError, match="outside the bound runtime"):
        save_checkpoint(tmp_path / "outside.pt", **source, binding=BINDING)
    assert calls == []



@pytest.mark.parametrize("version", [1, 2])
def test_multistep_counter_codec_preserves_repeated_milestones_and_closed_form_api(tmp_path, version):
    from collections import Counter
    source = make_state(multistep=True)
    source["scheduler"].milestones = Counter([2, 2, 5])
    advance(source, 3)
    path = tmp_path / "counter.pt"
    reference = save_checkpoint(path, **source, binding=BINDING)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert type(payload["scheduler"]["state"]["milestones"]) is dict
    if version == 1:
        payload["schema_version"] = 1
        del payload["runtime"], payload["sampler"]
        path, reference = write_payload(tmp_path, payload, "legacy-counter.pt")
    target = make_state(74, multistep=True)
    target["scheduler"].milestones = Counter([2, 2, 5])
    restore(path, reference, target)
    assert type(target["scheduler"].milestones) is Counter
    assert target["scheduler"].milestones == Counter({2: 2, 5: 1})
    for epoch in (1, 2, 5):
        source["scheduler"].last_epoch = target["scheduler"].last_epoch = epoch
        assert source["scheduler"]._get_closed_form_lr() == target["scheduler"]._get_closed_form_lr()


@pytest.mark.parametrize("milestones", [{-1: 1}, {2: -1}, {2: 0}, {2: True}, {"2": 1}])
def test_inspection_rejects_invalid_multistep_counter_encoding(tmp_path, milestones):
    state = make_state(multistep=True)
    path = tmp_path / "milestone-good.pt"
    save_checkpoint(path, **state, binding=BINDING)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    component = payload["scheduler"]
    component["state"]["milestones"] = milestones
    component["config"]["milestones"] = milestones
    component["structure"] = checkpoint._structure(component["state"])
    bad, reference = write_payload(tmp_path, payload, "milestone-bad.pt")
    with pytest.raises(CheckpointError, match="MultiStepLR milestones"):
        inspect_checkpoint(bad, expected_sha256=reference["sha256"], expected_binding=BINDING)


def test_saved_ema_geometry_rounding_is_preserved_without_relaxing_raw_geometry(tmp_path):
    state = persistent_geometry(make_state())
    # Mimic a legitimate single-ULP consequence of averaging constant floating
    # EMA buffers. Raw model geometry stays exactly the configured constant.
    ema_anchor = state["ema"].module.anchors
    ema_anchor[0] = torch.nextafter(ema_anchor[0], torch.tensor(float("inf")))
    path = tmp_path / "ema-cache-rounding.pt"
    reference = save_checkpoint(path, **state, binding=BINDING)
    target = persistent_geometry(make_state(73))
    restore(path, reference, target)
    assert torch.equal(target["model"].anchors, state["model"].anchors)
    assert torch.equal(target["ema"].module.anchors, state["ema"].module.anchors)
    assert not torch.equal(target["model"].anchors, target["ema"].module.anchors)


def _sampling_cpu_guard(monkeypatch):
    torch.set_num_threads(2)
    forbid_cuda(monkeypatch)
    # PyTorch 2.4 checks this CUDA-availability predicate while constructing even
    # a CPU autocast context. Its CPU branch must remain observation-free here.
    monkeypatch.setattr(torch.cuda.amp.common, "amp_definitely_not_available", lambda: True)
    # AdamW also queries CUDA availability in this graph-capture health check,
    # including for CPU-only parameters. Keep its real update arithmetic while
    # proving that bypassing this CUDA-only check is confined to CPU tensors.
    def cpu_graph_capture_health_check(optimizer):
        assert all(parameter.device.type == "cpu"
                   for group in optimizer.param_groups for parameter in group["params"])

    monkeypatch.setattr(torch.optim.Optimizer, "_cuda_graph_capture_health_check",
                        cpu_graph_capture_health_check)


@pytest.fixture
def sampling_cpu_only(monkeypatch):
    _sampling_cpu_guard(monkeypatch)


def _real_sampling_state(*, amp_dtype="float32", sampling_backend="deterministic_gather"):
    from sparse_rtdetr.baseline.training_v2b import V2BConfig, build_v2b_components
    return build_v2b_components(
        V2BConfig(input_size=128, physical_batch_size=1, accumulation_steps=2,
                  pretrained_required=False, warmup_steps=2, ema_warmups=2,
                  amp_dtype=amp_dtype, sampling_backend=sampling_backend),
        repo_root=Path(__file__).resolve().parents[1],
    )


def _real_sampling_parts(components):
    return {key: getattr(components, key) for key in
            ("model", "optimizer", "ema", "engine", "scheduler", "warmup", "runtime")}


def _real_sampling_binding(components):
    # This is a real model with generated CPU tensors, not a train_core run.
    # The same initialization snapshot is copied by the production run builder.
    return {
        "initial_weights": {"parameters": components.initialization["parameters"],
                            "model_state": components.initialization["model_state"]},
        "data": {"kind": "synthetic_cpu_sampling_checkpoint", "fixture_seeds": [930, 931]},
        "code": {key: components.initialization[key] for key in ("vendor_sources", "package_sources")},
        "config": {**components.config.binding_config(),
                   "sampling": deepcopy(components.initialization["sampling"])},
    }


def _real_sampling_batch(seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    images = torch.rand(2, 3, 128, 128, generator=generator)
    targets = []
    for count in (1, 3):
        centers = .25 + .5 * torch.rand(count, 2, generator=generator)
        sizes = .05 + .1 * torch.rand(count, 2, generator=generator)
        targets.append({"labels": torch.arange(count, dtype=torch.int64),
                        "boxes": torch.cat((centers, sizes), dim=1)})
    return images, targets


def _cpu_rng_snapshot():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())


@pytest.fixture(scope="module")
def real_candidate_checkpoint(tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        _sampling_cpu_guard(patch)
        source = _real_sampling_state()
        bound = _real_sampling_binding(source)
        path = tmp_path_factory.mktemp("real-candidate-checkpoint") / "untrained.pt"
        reference = save_checkpoint(path, **_real_sampling_parts(source), binding=bound)
    return path, reference, bound


@pytest.mark.parametrize("amp_dtype", ["float32", "bfloat16"])
def test_real_candidate_cpu_checkpoint_preserves_callable_and_exact_continuation(
        tmp_path, sampling_cpu_only, amp_dtype):
    from sparse_rtdetr.baseline.training_v2b import validate_model_sampling
    source = _real_sampling_state(amp_dtype=amp_dtype)
    bound = _real_sampling_binding(source)
    source.engine.begin_epoch(1)
    source.engine.train_window(*_real_sampling_batch(930))
    path = tmp_path / "candidate-boundary.pt"
    reference = save_checkpoint(path, **_real_sampling_parts(source), binding=bound)
    metadata = inspect_checkpoint(path, expected_sha256=reference["sha256"], expected_binding=bound)
    assert metadata["schema_version"] == 2 and metadata["optimizer_updates"] == 1
    assert metadata["binding"]["config"]["sampling"] == source.initialization["sampling"]
    expected_window = source.engine.train_window(*_real_sampling_batch(931))
    expected_rng = _cpu_rng_snapshot()
    rebuilt = _real_sampling_state(amp_dtype=amp_dtype)
    restored = restore_checkpoint(
        path, **_real_sampling_parts(rebuilt), expected_binding=bound,
        expected_sha256=reference["sha256"],
    )
    assert restored["schema_version"] == 2 and restored["optimizer_updates"] == 1
    actual_window = rebuilt.engine.train_window(*_real_sampling_batch(931))
    assert_tree_equal(expected_window, actual_window)
    assert_tree_equal(expected_rng, _cpu_rng_snapshot())
    for key in ("model", "optimizer", "ema", "engine", "scheduler", "warmup"):
        assert_tree_equal(getattr(source, key).state_dict(), getattr(rebuilt, key).state_dict())
    for model in (rebuilt.model, rebuilt.ema.module):
        assert validate_model_sampling(model, sampling_backend="deterministic_gather") == bound["config"]["sampling"]


def _forbid_checkpoint_mutation(monkeypatch, components):
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("sampling rejection must precede live state/RNG mutation")
    for key in ("model", "optimizer", "ema", "engine", "scheduler", "warmup"):
        monkeypatch.setattr(getattr(components, key), "load_state_dict", forbidden)
    monkeypatch.setattr(checkpoint, "_restore_rng", forbidden)
    monkeypatch.setattr(checkpoint, "_capture_rng", forbidden)
    return calls


@pytest.mark.parametrize("operation", ["save", "restore"])
@pytest.mark.parametrize("family,damage", [
    ("raw", "native_callable"), ("EMA", "native_callable"),
    ("raw", "partial_args"), ("EMA", "partial_keywords"),
    ("raw", "method"), ("EMA", "missing_module"), ("raw", "points"),
])
def test_actual_sampling_drift_rejected_before_checkpoint_state_or_rng_mutation(
        tmp_path, monkeypatch, sampling_cpu_only, real_candidate_checkpoint, operation, family, damage):
    import importlib
    path, reference, bound = real_candidate_checkpoint
    target = _real_sampling_state()
    model = target.model if family == "raw" else target.ema.module
    attention = model.decoder.decoder.layers[0].cross_attn
    core = attention.ms_deformable_attn_core
    if damage == "native_callable":
        native = importlib.import_module("src.zoo.rtdetr.rtdetrv2_decoder").deformable_attention_core_func_v2
        attention.ms_deformable_attn_core = functools.partial(native, method="default")
    elif damage == "partial_args":
        attention.ms_deformable_attn_core = functools.partial(core.func, 0, method="default")
    elif damage == "partial_keywords":
        attention.ms_deformable_attn_core = functools.partial(core.func, method="default", extra=None)
    elif damage == "method":
        attention.method = "discrete"
    elif damage == "missing_module":
        model.decoder.decoder.layers[0].cross_attn = nn.Identity()
    else:
        attention.num_points_list[0] += 1
    before_model = deepcopy(target.model.state_dict())
    before_ema = deepcopy(target.ema.state_dict())
    before_rng = _cpu_rng_snapshot()
    calls = _forbid_checkpoint_mutation(monkeypatch, target)
    destination = tmp_path / "must-not-exist.pt"
    with pytest.raises(CheckpointError, match="sampling"):
        if operation == "save":
            save_checkpoint(destination, **_real_sampling_parts(target), binding=bound)
        else:
            restore_checkpoint(path, **_real_sampling_parts(target), expected_binding=bound,
                               expected_sha256=reference["sha256"])
    assert calls == [] and not destination.exists()
    assert_tree_equal(before_model, target.model.state_dict())
    assert_tree_equal(before_ema, target.ema.state_dict())
    assert_tree_equal(before_rng, _cpu_rng_snapshot())
    assert target.engine.failed is False


@pytest.mark.parametrize("damage", [
    "missing_snapshot", "wrong_backend", "snapshot_numeric_type", "removed_declarations", "wrong_cuda_policy",
])
def test_sampling_initialization_snapshot_and_runtime_policy_cannot_be_omitted_or_redeclared(
        tmp_path, monkeypatch, sampling_cpu_only, damage):
    target = _real_sampling_state()
    bound = _real_sampling_binding(target)
    config = bound["config"]
    if damage == "missing_snapshot":
        del config["sampling"]
    elif damage == "wrong_backend":
        config["sampling_backend"] = "native"
    elif damage == "snapshot_numeric_type":
        config["sampling"]["modules"][0]["num_heads"] = 8.0
    elif damage == "removed_declarations":
        for key in ("sampling", "sampling_backend", "cuda_backend_policy"):
            del config[key]
    else:
        config["cuda_backend_policy"]["deterministic_algorithms"] = False
    before_rng = _cpu_rng_snapshot()
    calls = _forbid_checkpoint_mutation(monkeypatch, target)
    path = tmp_path / "invalid-declaration.pt"
    with pytest.raises(CheckpointError, match="sampling"):
        save_checkpoint(path, **_real_sampling_parts(target), binding=bound)
    assert calls == [] and not path.exists()
    assert_tree_equal(before_rng, _cpu_rng_snapshot())


def test_candidate_checkpoint_cannot_restore_into_native_model_with_same_tensor_state(
        monkeypatch, sampling_cpu_only, real_candidate_checkpoint):
    path, reference, bound = real_candidate_checkpoint
    target = _real_sampling_state(sampling_backend="native")
    assert target.initialization["parameters"] == bound["initial_weights"]["parameters"]
    assert target.initialization["model_state"] == bound["initial_weights"]["model_state"]
    before_rng = _cpu_rng_snapshot()
    calls = _forbid_checkpoint_mutation(monkeypatch, target)
    with pytest.raises(CheckpointError, match="raw sampling callable"):
        restore_checkpoint(path, **_real_sampling_parts(target), expected_binding=bound,
                           expected_sha256=reference["sha256"])
    assert calls == [] and target.engine.failed is False
    assert_tree_equal(before_rng, _cpu_rng_snapshot())


def test_checkpoint_inspection_checks_sampling_backend_policy_without_a_live_model_or_cuda(
        tmp_path, sampling_cpu_only, real_candidate_checkpoint):
    path, _, bound = real_candidate_checkpoint
    payload = torch.load(path, map_location="cpu", weights_only=True)
    incompatible = deepcopy(bound)
    incompatible["config"]["cuda_backend_policy"]["deterministic_algorithms"] = False
    payload["binding"], payload["binding_sha256"] = checkpoint._binding(incompatible)
    damaged, reference = write_payload(tmp_path, payload, "incorrect-candidate-policy.pt")
    with pytest.raises(CheckpointError, match="sampling/runtime backend policy"):
        inspect_checkpoint(damaged, expected_sha256=reference["sha256"], expected_binding=incompatible)
