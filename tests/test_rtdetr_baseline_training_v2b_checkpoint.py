"""CPU-only synthetic evidence for checkpoint continuity and rejection gates."""

from copy import deepcopy
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
    torch.manual_seed(seed)
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
