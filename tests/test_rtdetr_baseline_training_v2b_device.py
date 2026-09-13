"""CPU tests only; CUDA control flow uses a fake API backed by CPU generators."""

from copy import deepcopy
import hashlib
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from sparse_rtdetr.baseline import training_v2b_device as device
from sparse_rtdetr.baseline.training_v2b_device import (
    RuntimeDeviceError, capture_cuda_rng, cpu_runtime_identity, place_module,
    preflight_cuda_rng_restore, prepare_runtime, restore_cuda_rng,
    validate_component_placement, validate_cuda_rng_states,
    validate_module_placement, validate_optimizer_placement,
    validate_prepared_runtime, validate_runtime_identity,
)


GPU_UUID = "GPU-01234567-89ab-cdef-0123-456789abcdef"
FAKE_BINDING = {"config": {"device": "cuda:0", "cuda_gpu_uuid": GPU_UUID}}


def forbid_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU path must not call any CUDA API")
    for name in (
        "init", "_lazy_init", "is_available", "is_initialized", "current_device",
        "device_count", "get_device_properties", "set_device", "get_rng_state",
        "set_rng_state", "get_rng_state_all", "set_rng_state_all", "manual_seed",
        "manual_seed_all", "_lazy_call", "_is_in_bad_fork", "synchronize",
        "get_device_capability", "is_bf16_supported", "memory_allocated",
        "memory_reserved", "max_memory_allocated", "mem_get_info",
    ):
        monkeypatch.setattr(torch.cuda, name, forbidden)


class CacheModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)
        self.register_buffer("running", torch.zeros(2))
        self.register_buffer("nonpersistent", torch.ones(2), persistent=False)
        self.anchors = torch.tensor([0.5, float("inf")])
        self.nested = {"positions": [torch.arange(4), (torch.zeros(2),)]}
        self.parameter_alias = {"weight": self.linear.weight}


def test_cpu_seed_precedes_initialization_and_never_touches_cuda(monkeypatch):
    forbid_cuda(monkeypatch)
    runtime = prepare_runtime(seed=91)
    first = (random.random(), np.random.rand(3), torch.rand(3), nn.Linear(3, 2).weight.detach().clone())
    random.random()
    np.random.rand(5)
    torch.rand(11)
    second_runtime = prepare_runtime(device="cpu", seed=91)
    second = (random.random(), np.random.rand(3), torch.rand(3), nn.Linear(3, 2).weight.detach().clone())
    assert first[0] == second[0]
    np.testing.assert_array_equal(first[1], second[1])
    assert torch.equal(first[2], second[2])
    assert torch.equal(first[3], second[3])
    assert runtime.identity == second_runtime.identity == cpu_runtime_identity()
    assert runtime.admitted_binding_sha256 is None
    assert capture_cuda_rng(runtime) == {}
    preflight_cuda_rng_restore({}, runtime)
    restore_cuda_rng({}, runtime)


def test_real_cpu_module_optimizer_and_unregistered_caches_are_validated(monkeypatch):
    runtime = prepare_runtime(seed=12)
    model = CacheModule()
    ema = SimpleNamespace(module=deepcopy(model))
    optimizer = torch.optim.AdamW(model.parameters())
    # Populate actual AdamW moments, including its CPU step scalar.
    model.linear(torch.ones(2, 3)).sum().backward()
    optimizer.step()
    forbid_cuda(monkeypatch)
    placed = place_module(model, runtime)
    assert placed is model
    assert placed.parameter_alias["weight"] is placed.linear.weight
    assert torch.isinf(placed.anchors[1])
    assert placed.nested["positions"][1][0].device.type == "cpu"
    validate_component_placement(
        model=model, optimizer=optimizer, ema=ema, runtime=runtime,
    )


@pytest.mark.parametrize("kind", ["registered_buffer", "cache", "nested_cache", "ema", "optimizer"])
def test_cpu_placement_rejects_missing_cache_or_state_movement(kind):
    runtime = prepare_runtime(seed=14)
    model = CacheModule()
    ema = SimpleNamespace(module=deepcopy(model))
    optimizer = torch.optim.AdamW(model.parameters())
    if kind == "registered_buffer":
        model.running = torch.empty(2, device="meta")
    elif kind == "cache":
        model.anchors = torch.empty(2, device="meta")
    elif kind == "nested_cache":
        model.nested["positions"][1] = (torch.empty(2, device="meta"),)
    elif kind == "ema":
        ema.module.anchors = torch.empty(2, device="meta")
    else:
        optimizer.state[model.linear.weight]["exp_avg"] = torch.empty_like(model.linear.weight, device="meta")
    with pytest.raises(RuntimeDeviceError, match="placement"):
        validate_component_placement(model=model, optimizer=optimizer, ema=ema, runtime=runtime)


class LogicalDeviceTensor(torch.Tensor):
    """Metadata-only tensor double. Underlying storage always stays on CPU."""
    @staticmethod
    def __new__(cls, value, logical_device):
        result = torch.Tensor._make_subclass(cls, value, False)
        result._logical_device = torch.device(logical_device)
        return result

    @property
    def device(self):
        return self._logical_device


@pytest.mark.parametrize("capturable,fused", [(False, False), (True, False), (False, True)])
def test_adam_step_placement_semantics_use_cpu_scalar_unless_capturable_or_fused(capturable, fused):
    parameter = LogicalDeviceTensor(torch.ones(3), "cuda:0")
    expected_step_device = "cuda:0" if capturable or fused else "cpu"
    values = {
        "step": LogicalDeviceTensor(torch.tensor(1.0), expected_step_device),
        "exp_avg": LogicalDeviceTensor(torch.zeros(3), "cuda:0"),
        "exp_avg_sq": LogicalDeviceTensor(torch.ones(3), "cuda:0"),
    }
    optimizer = SimpleNamespace(
        param_groups=[{"params": [parameter], "capturable": capturable, "fused": fused}],
        state={parameter: values},
    )
    validate_optimizer_placement(optimizer, "cuda:0")
    wrong = "cpu" if expected_step_device == "cuda:0" else "cuda:0"
    values["step"] = LogicalDeviceTensor(torch.tensor(1.0), wrong)
    with pytest.raises(RuntimeDeviceError, match="optimizer step"):
        validate_optimizer_placement(optimizer, "cuda:0")


@pytest.mark.parametrize("seed", [-1, 2**32, True, 1.5])
def test_invalid_seed_fails_before_any_cuda_call(monkeypatch, seed):
    forbid_cuda(monkeypatch)
    with pytest.raises(RuntimeDeviceError, match="seed"):
        prepare_runtime(seed=seed)


@pytest.mark.parametrize("choice", ["cuda", "cuda:1", "mps", "cpu:0"])
def test_ambiguous_or_unsupported_device_never_initializes_cuda(monkeypatch, choice):
    forbid_cuda(monkeypatch)
    with pytest.raises(RuntimeDeviceError):
        prepare_runtime(device=choice, seed=0)


@pytest.mark.parametrize("uuid", [None, "0", "GPU-short", "GPU-01234567-89ab-cdef-0123-456789abcdeg"])
def test_cuda_requires_full_uuid_without_querying_cuda(monkeypatch, uuid):
    forbid_cuda(monkeypatch)
    with pytest.raises(RuntimeDeviceError, match="UUID"):
        prepare_runtime(device="cuda:0", seed=0, expected_gpu_uuid=uuid, binding=FAKE_BINDING)


class FakeGenerator:
    def __init__(self, calls):
        self.generator = torch.Generator(device="cpu")
        self.calls = calls

    def manual_seed(self, seed):
        self.calls.append(("cuda_seed", seed))
        self.generator.manual_seed(seed)
        return self

    def get_state(self):
        return self.generator.get_state()

    def set_state(self, state):
        self.generator.set_state(state)
        return self


class FakeCUDA:
    def __init__(self):
        self.calls = []
        self.initialized = False
        self.visible_count = 1
        self.index = 0
        self.properties = SimpleNamespace(
            uuid=GPU_UUID, name="CPU-backed fake GPU API",
            total_memory=24 * 1024**3, major=8, minor=6,
        )
        self.default_generators = [FakeGenerator(self.calls)]

    def is_available(self):
        # Stock CPU AdamW checks availability even for CPU-only parameters.
        return False

    def is_initialized(self):
        self.calls.append(("is_initialized",))
        return self.initialized

    def init(self):
        self.calls.append(("init",))
        self.initialized = True

    def device_count(self):
        self.calls.append(("device_count",))
        return self.visible_count

    def set_device(self, index):
        self.calls.append(("set_device", index))
        self.index = index

    def current_device(self):
        self.calls.append(("current_device",))
        return self.index

    def get_device_properties(self, index):
        self.calls.append(("get_device_properties", index))
        return self.properties

    def get_rng_state(self, index):
        self.calls.append(("get_rng_state", index))
        return self.default_generators[index].get_state()

    def set_rng_state(self, state, index):
        self.calls.append(("set_rng_state", index))
        self.default_generators[index].set_state(state)


@pytest.fixture
def fake_cuda(monkeypatch):
    from sparse_rtdetr.baseline import training_v2b_hardware as hardware
    # Restore numerical flags after each fake CUDA control-flow test.
    for component, key in (
        (torch.backends.cudnn, "benchmark"), (torch.backends.cudnn, "deterministic"),
        (torch.backends.cudnn, "allow_tf32"), (torch.backends.cuda.matmul, "allow_tf32"),
    ):
        monkeypatch.setattr(component, key, getattr(component, key))
    api = FakeCUDA()
    monkeypatch.setattr(torch, "cuda", api)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU_UUID)
    monkeypatch.setattr(torch.version, "cuda", "12.1-FAKE-CPU-TEST")
    def fake_admission(probe, *, binding, expected_gpu_uuid, **kwargs):
        api.calls.append(("admission",))
        assert probe == "explicit-test-double"
        assert binding == FAKE_BINDING
        assert expected_gpu_uuid == GPU_UUID
        return {"gpu": {"uuid": GPU_UUID}}
    monkeypatch.setattr(hardware, "require_native_hardware_admission", fake_admission)
    monkeypatch.setattr(device, "_scratch_cuda_generator", lambda index: torch.Generator(device="cpu"))
    return api


def prepared_fake_cuda():
    return prepare_runtime(
        device="cuda:0", seed=37, binding=FAKE_BINDING,
        gpu_probe="explicit-test-double", expected_gpu_uuid=GPU_UUID,
    )


def test_mock_cuda_admission_happens_before_initialization_and_seed_before_any_forward(fake_cuda):
    runtime = prepared_fake_cuda()
    calls = fake_cuda.calls
    assert calls[:3] == [("admission",), ("is_initialized",), ("init",)]
    assert ("cuda_seed", 37) in calls
    assert calls.index(("cuda_seed", 37)) < calls.index(("get_rng_state", 0))
    assert str(runtime.device) == "cuda:0"
    assert runtime.identity["cuda_devices"][0]["uuid"] == GPU_UUID
    assert runtime.identity["remapping_policy"] == "strict_no_device_or_uuid_remap"
    assert runtime.admitted_binding_sha256 == hashlib.sha256(
        b'{"config":{"cuda_gpu_uuid":"GPU-01234567-89ab-cdef-0123-456789abcdef","device":"cuda:0"}}'
    ).hexdigest()


def test_mock_cuda_rng_is_portable_cpu_bytes_and_replays_exactly(fake_cuda):
    runtime = prepared_fake_cuda()
    states = capture_cuda_rng(runtime)
    assert set(states) == {"0"}
    assert states["0"].device.type == "cpu" and states["0"].dtype == torch.uint8
    expected = torch.rand(13, generator=fake_cuda.default_generators[0].generator)
    preflight_cuda_rng_restore(states, runtime)
    restore_cuda_rng(states, runtime)
    replay = torch.rand(13, generator=fake_cuda.default_generators[0].generator)
    assert torch.equal(expected, replay)


def test_native_admission_failure_precedes_every_cuda_api(fake_cuda, monkeypatch):
    from sparse_rtdetr.baseline import training_v2b_hardware as hardware
    def blocked(*args, **kwargs):
        raise hardware.HardwareGateError("native cap proof missing")
    monkeypatch.setattr(hardware, "require_native_hardware_admission", blocked)
    with pytest.raises(hardware.HardwareGateError, match="cap proof"):
        prepared_fake_cuda()
    assert fake_cuda.calls == []


@pytest.mark.parametrize("kind", ["already_initialized", "two_devices", "wrong_uuid"])
def test_mock_cuda_topology_must_match_native_admission(fake_cuda, kind):
    if kind == "already_initialized":
        fake_cuda.initialized = True
    elif kind == "two_devices":
        fake_cuda.visible_count = 2
    else:
        fake_cuda.properties.uuid = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    with pytest.raises(RuntimeDeviceError):
        prepared_fake_cuda()
    assert ("cuda_seed", 37) not in fake_cuda.calls


def test_mock_cuda_old_torch_uuid_property_omission_uses_exact_uuid_visibility(fake_cuda):
    del fake_cuda.properties.uuid
    runtime = prepared_fake_cuda()
    assert runtime.identity["cuda_visible_devices"] == GPU_UUID
    assert runtime.identity["cuda_devices"][0]["uuid"] == GPU_UUID


@pytest.mark.parametrize("kind", ["visibility", "backend", "identity"])
def test_prepared_runtime_cannot_silently_change(fake_cuda, monkeypatch, kind):
    runtime = prepared_fake_cuda()
    if kind == "visibility":
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    elif kind == "backend":
        torch.backends.cudnn.benchmark = True
    else:
        runtime._identity["cuda_devices"][0]["name"] = "altered"
    with pytest.raises(RuntimeDeviceError):
        capture_cuda_rng(runtime)


@pytest.mark.parametrize("kind", ["topology", "dtype", "shape", "device", "zero_length"])
def test_mock_cuda_restore_preflight_rejects_bad_byte_state_without_live_rng_mutation(fake_cuda, kind):
    runtime = prepared_fake_cuda()
    states = capture_cuda_rng(runtime)
    before = fake_cuda.default_generators[0].get_state().clone()
    if kind == "topology":
        states = {"1": states["0"]}
    elif kind == "dtype":
        states["0"] = states["0"].to(torch.int64)
    elif kind == "shape":
        states["0"] = states["0"].reshape(1, -1)
    elif kind == "device":
        states["0"] = torch.empty_like(states["0"], device="meta")
    else:
        states["0"] = torch.empty(0, dtype=torch.uint8)
    with pytest.raises(RuntimeDeviceError):
        preflight_cuda_rng_restore(states, runtime)
    assert torch.equal(before, fake_cuda.default_generators[0].get_state())
    assert not any(call[0] == "set_rng_state" for call in fake_cuda.calls)


def test_cpu_identity_cannot_be_relabeled_cuda():
    identity = cpu_runtime_identity()
    identity["device"] = "cuda:0"
    with pytest.raises(RuntimeDeviceError, match="topology"):
        validate_runtime_identity(identity)


def test_runtime_identity_property_is_a_copy():
    runtime = prepare_runtime(seed=3)
    copied = runtime.identity
    copied["device"] = "cuda:0"
    assert runtime.identity["device"] == "cpu"
    validate_prepared_runtime(runtime)
