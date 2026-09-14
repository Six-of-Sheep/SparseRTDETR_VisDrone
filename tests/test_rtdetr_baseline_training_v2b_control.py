"""CPU engineering tests for the control harness; no native GPU or real data.

The small network uses actual BatchNorm, CPU BF16, backward and AdamW. Native
admission and CUDA timing APIs are trapped or replaced only inside individual
tests. These checks do not certify CUDA numerical replay or hardware stability.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from sparse_rtdetr.baseline import training_v2b_control as control
from sparse_rtdetr.baseline.training_v2b import V2BConfig, seed_cpu_sources, logical_batch_indices
from sparse_rtdetr.baseline.training_v2b_device import prepare_runtime
from sparse_rtdetr.baseline.training_v2b_engine import AccumulationEngine
from sparse_rtdetr.baseline.training_v2b_evidence import (
    canonical_sha256, file_reference, write_exclusive_json,
)


@pytest.fixture(autouse=True)
def never_initialize_cuda(monkeypatch):
    torch.set_num_threads(2)
    def forbidden(*args, **kwargs):
        raise AssertionError("native CUDA work is forbidden in CPU control tests")
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(torch.cuda, "init", forbidden)


def dummy_reference(path, sha="a" * 64, size=1):
    return {"path": str(path), "sha256": sha, "size_bytes": size,
            "sha256_scope": "complete_file_bytes"}


@pytest.fixture
def contract(tmp_path):
    root = tmp_path / "checkout"
    data = tmp_path / "approved_metadata"
    return {
        "schema_version": 1, "kind": "v2b_640_control_worker",
        "campaign_id": "cpu-fixture-campaign", "stage": "smoke", "arm": "A",
        "run_id": "cpu-fixture-smoke-A", "repo_root": str(root),
        "output_dir": str(tmp_path / "worker-output"),
        "config": asdict(V2BConfig()), "num_workers": 2, "prefetch_factor": 2,
        "torch_version": str(torch.__version__),
        "code_files": {"src/file.py": dummy_reference(root / "src/file.py")},
        "train_core": {
            "annotation": dummy_reference(data / "train_core_coco.json", control.TRAIN_ANNOTATION_SHA256, 51148042),
            "manifest": dummy_reference(data / "train_core_manifest.json", control.TRAIN_MANIFEST_SHA256, 2955809),
            "image_root": str(tmp_path / "allowed_train_images"),
            "expected_samples": 4869, "expected_batches_per_epoch": 304,
        },
        "pretrained": dummy_reference(tmp_path / "ResNet18_vd_pretrained_from_paddle.pth",
                                      control.PRETRAINED_SHA256, 44878642),
        "development_binding": {
            "annotation": dummy_reference(data / "development_coco.json"),
            "manifest": dummy_reference(data / "development_manifest.json"),
            "image_root": str(tmp_path / "allowed_dev_images"), "binding_sha256": "b" * 64,
        },
        "epoch_order_sha256": control.frozen_epoch_orders(),
        "policy_bundle": {"synthetic_test_only": True},
        "expected_gpu_uuid": "GPU-1faee6f0-1da7-4ede-2475-67a5a00274a8",
        "paired_reference": None, "replay_source": None,
    }


def test_contract_validation_is_cpu_pure_and_preserves_frozen_order(contract, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("pure contract check must not open inputs or build a model")
    monkeypatch.setattr(control, "file_reference", forbidden)
    monkeypatch.setattr(control, "build_v2b_components", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    result = control.validate_worker_contract(contract, verify_files=False)
    assert result == contract
    result["config"]["seed"] = 9
    assert contract["config"]["seed"] == 0
    assert len(control.frozen_epoch_orders()) == 30


@pytest.mark.parametrize("path,value", [
    (("schema_version",), True), (("kind",), "unfrozen"), (("stage",), "train120"),
    (("arm",), "C"), (("run_id",), "../overwrite"),
    (("config", "seed"), 1), (("config", "seed"), False),
    (("config", "input_size"), 896), (("config", "amp_dtype"), "float32"),
    (("config", "bn_statistics"), "frozen"), (("config", "num_denoising"), 0),
    (("config", "learning_rate"), 0.01), (("config", "warmup_steps"), 20),
    (("config", "physical_batch_size"), 8), (("num_workers",), 0),
    (("num_workers",), True), (("prefetch_factor",), 1),
    (("torch_version",), "unknown"), (("expected_gpu_uuid",), "0"),
    (("train_core", "expected_samples"), 4868), (("train_core", "expected_batches_per_epoch"), 305),
    (("train_core", "annotation", "sha256"), "b" * 64),
    (("train_core", "manifest", "size_bytes"), 2),
    (("pretrained", "size_bytes"), 1), (("policy_bundle",), {}),
    (("epoch_order_sha256", "7"), "0" * 64),
])
def test_scope_configuration_and_order_drift_is_rejected_without_io(contract, path, value, monkeypatch):
    document = copy.deepcopy(contract)
    target = document
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    monkeypatch.setattr(control, "file_reference", lambda *a, **k: pytest.fail("input read before scope check"))
    with pytest.raises((ValueError, RuntimeError)):
        control.validate_worker_contract(document, verify_files=False)


@pytest.mark.parametrize("field", ["train", "pretrained", "development", "code", "output", "peer"])
def test_forbidden_paths_fail_before_any_file_read(contract, field, tmp_path, monkeypatch):
    forbidden = tmp_path / "confirmatory" / "forbidden"
    if field == "train":
        contract["train_core"]["annotation"]["path"] = str(forbidden / "train_core_coco.json")
    elif field == "pretrained":
        contract["pretrained"]["path"] = str(forbidden / "ResNet18_vd_pretrained_from_paddle.pth")
    elif field == "development":
        contract["development_binding"]["manifest"]["path"] = str(forbidden / "development_manifest.json")
    elif field == "code":
        contract["code_files"]["src/file.py"]["path"] = str(forbidden / "file.py")
    elif field == "output":
        contract["output_dir"] = str(forbidden)
    else:
        contract["arm"], contract["config"] = "B", asdict(V2BConfig(physical_batch_size=8, accumulation_steps=2))
        contract["paired_reference"] = dummy_reference(forbidden / "worker-result.json")
    monkeypatch.setattr(control, "file_reference", lambda *a, **k: pytest.fail("forbidden content read"))
    with pytest.raises(control.ControlWorkerError):
        control.validate_worker_contract(contract, verify_files=False)


def test_b_and_replay_require_correct_external_authorities(contract, tmp_path):
    contract["arm"], contract["config"] = "B", asdict(V2BConfig(physical_batch_size=8, accumulation_steps=2))
    with pytest.raises(control.ControlWorkerError, match="B must bind"):
        control.validate_worker_contract(contract, verify_files=False)
    contract["paired_reference"] = dummy_reference(tmp_path / "A" / "worker-result.json")
    assert control.validate_worker_contract(contract, verify_files=False)["arm"] == "B"
    contract["stage"] = "smoke_replay"
    with pytest.raises(control.ControlWorkerError, match="smoke_replay"):
        control.validate_worker_contract(contract, verify_files=False)
    contract["replay_source"] = dummy_reference(tmp_path / "B" / "worker-result.json")
    assert control.validate_worker_contract(contract, verify_files=False)["stage"] == "smoke_replay"


def test_reference_hash_checked_before_json_parse_and_snapshot_deserialization(tmp_path, monkeypatch):
    path = tmp_path / "broken.pt"
    path.write_bytes(b"not a pickle or JSON")
    reference = file_reference(path)
    reference["sha256"] = "0" * 64
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("unverified deserialization"))
    with pytest.raises(control.ControlWorkerError, match="reference mismatch"):
        control._load_parameters(reference)
    with pytest.raises(control.ControlWorkerError, match="reference mismatch"):
        control._read_json_reference(reference, "synthetic")


def test_role_symlink_is_rejected_without_opening_target(tmp_path):
    target = tmp_path / "confirmatory"
    target.mkdir()
    link = tmp_path / "approved_alias"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(control.ControlWorkerError, match="forbidden"):
        control._reference(dummy_reference(link / "file.json"), "synthetic")


def receipt(indices=(3, 1), index=0, epoch=1):
    binding = "a" * 64
    result = {
        "epoch": epoch, "logical_batch_index": index, "binding_sha256": binding,
        "augmented_sha256": "b" * 64,
        "samples": [
            {"index": value, "epoch": epoch, "order_position": index * len(indices) + offset,
             "binding_sha256": binding, "source_path": f"/synthetic_train/{value}.jpg"}
            for offset, value in enumerate(indices)
        ],
    }
    result["receipt_sha256"] = canonical_sha256(result)
    return result


def test_actual_receipts_check_augmented_bytes_source_path_and_predefined_order():
    original = receipt()
    control.verify_input_receipt(original, expected_indices=[3, 1], epoch=1, logical_batch_index=0,
                                 paired_receipt=copy.deepcopy(original))
    for field in ("augmented_sha256", "source_path", "order"):
        altered = copy.deepcopy(original)
        if field == "augmented_sha256":
            altered[field] = "c" * 64
        elif field == "source_path":
            altered["samples"][0][field] = "/synthetic_train/another.jpg"
        else:
            altered["samples"][0]["index"] = 99
        altered["receipt_sha256"] = canonical_sha256({k: v for k, v in altered.items() if k != "receipt_sha256"})
        with pytest.raises(control.ControlWorkerError, match="input|order"):
            control.verify_input_receipt(altered, expected_indices=[3, 1], epoch=1,
                                         logical_batch_index=0, paired_receipt=original)


def test_replay_parameter_comparison_uses_global_l2_and_absolute_bound():
    expected = {"small": torch.tensor([1.0], dtype=torch.float64),
                "large": torch.full((10,), 100.0, dtype=torch.float64)}
    actual = {name: tensor.clone() for name, tensor in expected.items()}
    actual["small"] += 9e-6
    result = control.compare_parameter_maps(expected, actual)
    assert result["max_absolute"] == pytest.approx(9e-6)
    actual["small"] += 2e-6
    with pytest.raises(control.ControlWorkerError, match="tolerances"):
        control.compare_parameter_maps(expected, actual)
    with pytest.raises(control.ControlWorkerError, match="tolerances"):
        control.compare_parameter_maps({"zero": torch.zeros(1)}, {"zero": torch.tensor([1e-8])})
    with pytest.raises(control.ControlWorkerError, match="nonfinite"):
        control.compare_parameter_maps({"x": torch.ones(1)}, {"x": torch.tensor([float("nan")])})


class SmallDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 1)
        self.bn = nn.BatchNorm2d(2)
        self.linear = nn.Linear(2, 1)

    def forward(self, images, targets=None):
        values = self.linear(self.bn(self.conv(images)).mean((2, 3))).flatten()
        # Consume actual CPU RNG, so successful replay must restore it too.
        values = values + 0.01 * torch.rand_like(values)
        maximum = max(len(target["labels"]) for target in targets)
        groups = max(100 // maximum, 1) if maximum else None
        return {"predictions": values,
                "dn_meta": None if groups is None else {"dn_num_group": groups, "dn_num_split": [2 * maximum * groups, 300]}}


class SmallLoss(nn.Module):
    def forward(self, outputs, targets):
        values = outputs["predictions"].float()
        denominator = max(sum(len(target["labels"]) for target in targets), 1)
        return {"loss_vfl": (values - 0.4).square().sum() / denominator,
                "loss_bbox_aux_0": values.abs().sum() / denominator,
                "loss_vfl_dn_0": 0.2 * values.square().sum() / denominator}


class SmallEMA:
    def __init__(self, model):
        self.module = copy.deepcopy(model).eval()
        self.updates = 0
    def update(self, model):
        self.updates += 1
        with torch.no_grad():
            for name, value in self.module.state_dict().items():
                if value.is_floating_point():
                    value.mul_(0.9).add_(model.state_dict()[name], alpha=0.1)
    def state_dict(self):
        return {"module": self.module.state_dict(), "updates": self.updates}


class SmallWarmup:
    def __init__(self, duration=2000):
        self.last_step, self.warmup_duration = 0, duration
    def step(self):
        self.last_step += 1
    def finished(self):
        return self.last_step >= self.warmup_duration
    def state_dict(self):
        return {"last_step": self.last_step, "warmup_duration": self.warmup_duration}


def components():
    runtime = prepare_runtime(seed=17)
    config = V2BConfig(seed=17, input_size=128, physical_batch_size=1,
                       accumulation_steps=2, pretrained_required=False)
    model, criterion = SmallDetector(), SmallLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    ema, warmup = SmallEMA(model), SmallWarmup()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000], gamma=0.1)
    engine = AccumulationEngine(model, criterion, optimizer, physical_batch_size=1,
                                accumulation_steps=2, amp_dtype="bfloat16",
                                ema=ema, warmup=warmup, scheduler=scheduler)
    return SimpleNamespace(config=config, runtime=runtime, model=model, criterion=criterion,
                           optimizer=optimizer, ema=ema, warmup=warmup, scheduler=scheduler, engine=engine)


def targets():
    return [{"labels": torch.zeros(count, dtype=torch.long),
             "boxes": torch.full((count, 4), 0.25)} for count in (1, 2)]


class SmallLoader:
    def __init__(self, c):
        self.c, self.batches_per_epoch, self.dataset = c, 4, list(range(8))
        self.has_active_iterator = False
    def state_dict(self):
        return {"optimizer_updates": self.c.engine.optimizer_updates,
                "epoch": self.c.engine.epoch, "epoch_active": self.c.engine.epoch_active}
    def __iter__(self):
        self.has_active_iterator = True
        try:
            plan = list(logical_batch_indices(8, seed=17, epoch=1, logical_batch_size=2))
            for i in range(self.c.engine.optimizer_updates, 4):
                evidence = receipt(plan[i], index=i)
                yield SimpleNamespace(epoch=1, logical_batch_index=i, images=torch.arange(32).float().reshape(2, 1, 4, 4) / 32,
                                      targets=targets(), evidence=lambda evidence=evidence: copy.deepcopy(evidence))
        finally:
            self.has_active_iterator = False


class SmallSession:
    def __init__(self, c):
        self.components, self.loader, self.calls = c, SmallLoader(c), 0
        self.binding = {"binding_sha256": "c" * 64}
    def train_batch(self, batch, hardware_probe=None):
        self.calls += 1
        return {"input": batch.evidence(), "window": self.components.engine.train_window(batch.images, batch.targets)}


class FakeMonitor:
    """A test double is never consumed by a production prepare_runtime."""
    def __init__(self):
        self.events = []
    def check(self, stage):
        self.events.append(stage)
        return {"synthetic_cpu_test": True, "stage": stage}
    def admission(self):
        return None


@pytest.fixture
def fake_timing(monkeypatch):
    for name in ("synchronize", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, lambda *args: None)
    for name in ("memory_allocated", "memory_reserved", "max_memory_allocated", "max_memory_reserved"):
        monkeypatch.setattr(torch.cuda, name, lambda *args: 0)


def test_actual_cpu_bf16_bn_dn_denominator_and_clock_observations(fake_timing):
    c = components()
    session = SmallSession(c)
    c.engine.begin_epoch(1)
    iterator = iter(session.loader)
    batch = next(iterator)
    record = control._measure_window(session, batch, FakeMonitor(), expected_indices=[s["index"] for s in batch.evidence()["samples"]])
    iterator.close()
    assert record["window"]["denominator"] == 3
    assert record["window"]["coefficients"] == [1 / 3, 2 / 3]
    assert record["window"]["dn_num_groups"] == [100, 50]
    assert record["window"]["dn_query_counts"] == [200, 200]
    assert record["forward_observation"]["bn_forward_calls"]["bn"] == 2
    assert record["state"]["raw_bn"]["bn"]["num_batches_tracked"] == 2
    assert record["state"]["ema_bn"]["bn"]["num_batches_tracked"] == 0
    assert record["state"]["clocks"]["ema_updates"] == 1
    assert record["state"]["clocks"]["adam_step_values"] == [1]
    assert set(record["loss_families_weighted"]["denoising"]) == {"loss_vfl_dn_0"}
    assert record["auxiliary_numerical_health"]["finite"]


def test_pair_mismatch_is_rejected_before_monitor_or_model_update(fake_timing):
    c = components()
    session, monitor = SmallSession(c), FakeMonitor()
    c.engine.begin_epoch(1)
    iterator = iter(session.loader)
    batch = next(iterator)
    peer = batch.evidence()
    peer["augmented_sha256"] = "c" * 64
    with pytest.raises(control.ControlWorkerError, match="paired actual"):
        control._measure_window(session, batch, monitor, expected_indices=[s["index"] for s in batch.evidence()["samples"]],
                                paired_record={"input": peer})
    iterator.close()
    assert session.calls == 0 and c.engine.optimizer_updates == 0 and monitor.events == []


def test_failed_window_closes_iterator_and_never_retries(tmp_path, fake_timing):
    c = components()
    session = SmallSession(c)
    c.engine.begin_epoch(1)
    attempts = []
    def failure(*args, **kwargs):
        attempts.append(1)
        raise RuntimeError("injected ordinary test failure")
    session.train_batch = failure
    with pytest.raises(RuntimeError, match="injected ordinary"):
        control._run_active_windows(session, FakeMonitor(), tmp_path, limit=4,
                                    paired_index={}, replay_index={}, save_midpoint=False,
                                    snapshots=False, checkpoints={}, receipts=[], replay_checks=[])
    assert attempts == [1]
    assert not session.loader.has_active_iterator
    assert c.engine.optimizer_updates == 0


def test_full_state_hashes_capture_rng_bn_optimizer_and_preserve_all_state():
    c = components()
    loader = SmallLoader(c)
    before = control.capture_control_state(c, loader, full=True)
    assert before == control.capture_control_state(c, loader, full=True)
    torch.rand(1)
    after = control.capture_control_state(c, loader, full=True)
    assert before["rng"] != after["rng"]
    assert before["raw_state_sha256"] == after["raw_state_sha256"]
    with torch.no_grad():
        c.model.bn.running_mean.add_(1)
    changed = control.capture_control_state(c, loader, full=True)
    assert after["raw_bn"] != changed["raw_bn"]


def make_replay_record(tmp_path, stem):
    c = components()
    loader = SmallLoader(c)
    c.engine.begin_epoch(1)
    window = c.engine.train_window(torch.arange(32).float().reshape(2, 1, 4, 4) / 32, targets())
    return {
        "input": receipt(), "window": window, "run_binding_sha256": "c" * 64,
        "state": control.capture_control_state(c, loader),
        "forward_observation": {"synthetic": "fixed"},
        "raw_parameters_reference": control._publish_parameters(tmp_path / (stem + ".pt"), c.model),
    }


def test_replay_rejects_rng_bn_clock_loss_and_parameter_drift(tmp_path):
    expected = make_replay_record(tmp_path, "original")
    actual = make_replay_record(tmp_path, "fresh")
    assert control.compare_replay_records(expected, actual)["status"] == "PASS"
    for field in ("rng", "raw_bn", "clocks"):
        altered = copy.deepcopy(actual)
        altered["state"][field]["injected"] = 1
        with pytest.raises(control.ControlWorkerError, match="same-arm replay"):
            control.compare_replay_records(expected, altered)
    altered = copy.deepcopy(actual)
    altered["window"]["loss"] *= 1.001
    with pytest.raises(control.ControlWorkerError, match="logical loss"):
        control.compare_replay_records(expected, altered)


def test_forward_hooks_are_removed_after_exception():
    c = components()
    before = [(len(module._forward_hooks), len(module._forward_pre_hooks)) for module in c.model.modules()]
    with pytest.raises(RuntimeError):
        with control._ForwardObservation(c):
            raise RuntimeError("fail inside observer")
    assert before == [(len(module._forward_hooks), len(module._forward_pre_hooks)) for module in c.model.modules()]
    assert len(c.criterion._forward_pre_hooks) == 0


@pytest.fixture
def real_sampling_components():
    root = Path(__file__).resolve().parents[1]
    return control.build_v2b_components(
        V2BConfig(pretrained_required=False, input_size=128), repo_root=root,
    )


@pytest.mark.parametrize("autocast", [False, True])
def test_real_sampling_proxy_preserves_objects_outputs_gradients_layout_and_rng(
        real_sampling_components, autocast):
    c = real_sampling_components
    modules = [(name, module) for name, module in c.model.named_modules()
               if type(module).__name__ == "MSDeformableAttention"]
    name, module = modules[0]
    originals = [item.ms_deformable_attn_core for _, item in modules]
    core = originals[0]
    shapes, points = [(2, 2), (1, 1), (1, 1)], module.num_points_list
    generator = torch.Generator(device="cpu").manual_seed(919)
    dtype = torch.bfloat16 if autocast else torch.float32
    # Non-dense views with nonzero offsets must be observed before any copy.
    value = torch.randn(2, 7, 2, 3, generator=generator, dtype=dtype)[:, 1:].requires_grad_(True)
    locations = torch.rand(2, 4, 2, sum(points), 4, generator=generator)[..., 1:3].requires_grad_(True)
    weights = torch.rand(2, 4, 2, sum(points), 2, generator=generator, dtype=dtype)[..., 1].requires_grad_(True)
    inputs = (value, locations, weights)
    versions = [item._version for item in inputs]
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        expected = core(value, shapes, locations, weights, points)
    expected_gradients = torch.autograd.grad(expected, inputs, torch.ones_like(expected))
    seen = []
    def original_call(*args, **kwargs):
        assert len(args) == 1 and args[0] is value
        assert kwargs["value_spatial_shapes"] is shapes
        assert kwargs["sampling_locations"] is locations
        assert kwargs["attention_weights"] is weights
        assert kwargs["num_points_list"] is points
        result = core(*args, **kwargs)
        seen.append(result)
        return result
    module.ms_deformable_attn_core = original_call
    before_rng = torch.get_rng_state().clone()
    with control._ForwardObservation(c, sampling=True) as observation:
        assert module.ms_deformable_attn_core is not original_call
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            actual = module.ms_deformable_attn_core(
                value, value_spatial_shapes=shapes, sampling_locations=locations,
                attention_weights=weights, num_points_list=points,
            )
        assert len(seen) == 1 and actual is seen[0]
        actual_gradients = torch.autograd.grad(actual, inputs, torch.ones_like(actual))
    assert module.ms_deformable_attn_core is original_call
    assert [item.ms_deformable_attn_core for _, item in modules[1:]] == originals[1:]
    assert torch.equal(torch.get_rng_state(), before_rng)
    assert [item._version for item in inputs] == versions
    assert torch.equal(actual, expected)
    assert all(torch.equal(left, right) for left, right in zip(expected_gradients, actual_gradients))
    evidence = observation.sampling_evidence()
    assert evidence["acceptance_decision"] is False
    assert evidence["derived_grid_tensor_directly_observed"] is False
    assert evidence["modules"] == [key for key, _ in modules]
    row, = evidence["calls"]
    assert row["module_name"] == name and row["call_completed"] is True
    assert row["autocast_enabled"] is autocast and row["autocast_dtype"] == "torch.bfloat16"
    for key, tensor in zip(("value", "sampling_locations", "attention_weights"), inputs):
        metadata = row["inputs"][key]
        assert metadata["dtype"] == str(tensor.dtype) and metadata["shape"] == list(tensor.shape)
        assert metadata["stride"] == list(tensor.stride())
        assert metadata["storage_offset"] == tensor.storage_offset() > 0
    assert row["output"] == control._tensor_layout_observation(actual)


def test_real_attention_forward_observes_actual_cpu_amp_core_inputs(real_sampling_components):
    c = real_sampling_components
    module = next(item for item in c.model.modules() if type(item).__name__ == "MSDeformableAttention")
    core = module.ms_deformable_attn_core
    generator = torch.Generator(device="cpu").manual_seed(920)
    query = torch.randn(2, 4, module.embed_dim, generator=generator)
    value = torch.randn(2, 6, module.embed_dim, generator=generator)
    references = torch.rand(2, 4, 1, 4, generator=generator)
    with control._ForwardObservation(c, sampling=True) as observation:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = module(query, references, value, [(2, 2), (1, 1), (1, 1)])
    assert output.shape == (2, 4, module.embed_dim)
    assert module.ms_deformable_attn_core is core
    row, = observation.sampling_evidence()["calls"]
    assert row["core_identity"]["callable_name"] == "deformable_attention_core_func_v2"
    assert row["inputs"]["value"]["dtype"] == "torch.bfloat16"
    assert row["inputs"]["sampling_locations"]["dtype"] == "torch.float32"
    assert row["inputs"]["attention_weights"]["dtype"] == "torch.bfloat16"
    assert row["output"]["dtype"] == "torch.float32"


@pytest.mark.parametrize("failure", ["body", "second_install", "core"])
def test_sampling_proxies_and_forward_hooks_restore_after_all_exceptions(
        real_sampling_components, monkeypatch, failure):
    c = real_sampling_components
    modules = [item for item in c.model.modules() if type(item).__name__ == "MSDeformableAttention"]
    hooks = [(len(item._forward_hooks), len(item._forward_pre_hooks)) for item in c.model.modules()]
    if failure == "core":
        def broken_core(*args, **kwargs):
            raise RuntimeError("core failed")
        modules[0].ms_deformable_attn_core = broken_core
    originals = [item.ms_deformable_attn_core for item in modules]
    if failure == "second_install":
        factory = control._ForwardObservation._sampling_proxy
        calls = []
        def broken_install(self, name, module, core):
            calls.append(name)
            if len(calls) == 2:
                raise RuntimeError("installation failed")
            return factory(self, name, module, core)
        monkeypatch.setattr(control._ForwardObservation, "_sampling_proxy", broken_install)
    with pytest.raises(RuntimeError):
        with control._ForwardObservation(c, sampling=True):
            if failure == "core":
                modules[0].ms_deformable_attn_core()
            raise RuntimeError("body failed")
    assert [item.ms_deformable_attn_core for item in modules] == originals
    assert hooks == [(len(item._forward_hooks), len(item._forward_pre_hooks)) for item in c.model.modules()]
    assert len(c.criterion._forward_pre_hooks) == 0


def test_nonfinite_ema_or_adam_state_fails_health_gate():
    c = components()
    c.engine.begin_epoch(1)
    c.engine.train_window(torch.ones(2, 1, 4, 4), targets())
    state = next(iter(c.optimizer.state.values()))
    state["exp_avg"][0] = float("inf")
    with pytest.raises(control.ControlWorkerError, match="nonfinite"):
        control._finite_auxiliary_state(c)


def test_output_parameter_snapshot_is_exclusive_and_cpu_weights_only(tmp_path):
    c = components()
    destination = tmp_path / "raw.pt"
    reference = control._publish_parameters(destination, c.model)
    actual = control._load_parameters(reference)
    assert set(actual) == dict(c.model.named_parameters()).keys()
    with pytest.raises(FileExistsError):
        control._publish_parameters(destination, c.model)
    assert file_reference(destination) == reference


def test_complete_bn_snapshot_is_atomic_exclusive_and_preserves_state(tmp_path, monkeypatch):
    c = components()
    c.engine.begin_epoch(1)
    window = c.engine.train_window(torch.ones(2, 1, 4, 4), targets())
    before = control.capture_control_state(c, SmallLoader(c), full=True)
    record = {"run_binding_sha256": "c" * 64, "window": window, "state": before}
    destination = tmp_path / "bn.pt"
    original_link = control.os.link
    def atomic_link(source, target):
        assert not Path(target).exists()
        value = torch.load(source, map_location="cpu", weights_only=True)
        assert value["raw_bn"]["bn"]["running_mean"].device.type == "cpu"
        assert value["ema_bn"]["bn"]["num_batches_tracked"].dtype == torch.int64
        return original_link(source, target)
    monkeypatch.setattr(control.os, "link", atomic_link)
    reference = control._publish_bn_snapshot(destination, c, record)
    snapshot = control._load_bn_snapshot(reference, record)
    assert before == control.capture_control_state(c, SmallLoader(c), full=True)
    for family, model in (("raw_bn", c.model), ("ema_bn", c.ema.module)):
        assert snapshot[family]["bn"]["training"] == model.bn.training
        for name in ("running_mean", "running_var", "num_batches_tracked"):
            assert torch.equal(snapshot[family]["bn"][name], getattr(model.bn, name))
    with torch.no_grad():
        c.model.bn.running_mean.add_(0.25)
    assert not torch.equal(snapshot["raw_bn"]["bn"]["running_mean"], c.model.bn.running_mean)
    monkeypatch.setattr(control.os, "link", original_link)
    with pytest.raises(FileExistsError):
        control._publish_tensor_evidence(destination, snapshot)
    assert file_reference(destination) == reference
    assert list(tmp_path.iterdir()) == [destination]
    bad = {**reference, "sha256": "0" * 64}
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("unverified BN snapshot deserialized"))
    with pytest.raises(control.ControlWorkerError, match="reference mismatch"):
        control._load_bn_snapshot(bad, record)


def test_live_layout_is_observed_before_copy_and_does_not_relax_or_add_replay_gates(tmp_path):
    c = components()
    c.engine.begin_epoch(1)
    window = c.engine.train_window(torch.ones(2, 1, 4, 4), targets())
    expected = {
        "run_binding_sha256": "c" * 64, "input": receipt(), "window": window,
        "state": control.capture_control_state(c, SmallLoader(c)),
        "forward_observation": {"synthetic": "fixed"},
        "sampling_core_observation": {"synthetic_layout": {"stride": [2], "storage_offset": 1}},
        "raw_parameters_reference": control._publish_parameters(tmp_path / "expected-raw.pt", c.model),
    }
    expected["bn_snapshot_reference"] = control._publish_bn_snapshot(tmp_path / "expected-bn.pt", c, expected)
    with torch.no_grad():
        # Preserve every BN/parameter/moment value while changing live storage
        # offset/stride; the CPU clone in the evidence will be contiguous.
        source = torch.empty(6)
        source[1:5:2].copy_(c.model.bn.running_mean)
        c.model.bn.running_mean = source[1:5:2]
        moment = c.optimizer.state[c.model.bn.weight]["exp_avg"]
        storage = torch.empty(4)
        storage[::2].copy_(moment)
        c.optimizer.state[c.model.bn.weight]["exp_avg"] = storage[::2]
    actual = {**copy.deepcopy(expected), "state": control.capture_control_state(c, SmallLoader(c))}
    actual["sampling_core_observation"]["synthetic_layout"]["stride"] = [1]
    actual["raw_parameters_reference"] = control._publish_parameters(tmp_path / "actual-raw.pt", c.model)
    actual["bn_snapshot_reference"] = control._publish_bn_snapshot(tmp_path / "actual-bn.pt", c, actual)
    payload = control._load_bn_snapshot(actual["bn_snapshot_reference"], actual)
    layout = payload["live_tensor_layout"]
    observed = layout["raw_bn"]["bn"]["running_mean"]
    assert observed["stride"] == [2] and observed["storage_offset"] == 1
    assert observed["contiguous"] is False and observed["channels_last"] is None
    assert payload["raw_bn"]["bn"]["running_mean"].is_contiguous()
    assert payload["raw_bn"]["bn"]["running_mean"].storage_offset() == 0
    assert layout["optimizer_moments"]["bn.weight"]["exp_avg"]["stride"] == [2]
    assert layout["raw_model_parameters"]["conv.weight"]["channels_last"] is not None
    assert layout["raw_model_buffers"]["bn.running_mean"]["storage_offset"] == 1
    left = write_exclusive_json(tmp_path / "expected-receipt.json", expected)
    right = write_exclusive_json(tmp_path / "actual-receipt.json", actual)
    report_ref = control._publish_replay_observation(
        tmp_path / "layout-observation.json", expected, actual, expected_reference=left, actual_reference=right,
    )
    report = control._read_json_reference(report_ref, "layout observation")
    assert report["layout_field_differences"] and report["exact_field_differences"] == []
    assert report["layout_differences_are_observations_not_an_acceptance_gate"] is True
    assert report["sampling_core_observation"]["field_differences"]
    assert report["sampling_core_observation"]["acceptance_decision"] is False
    assert control.compare_replay_records(expected, actual)["status"] == "PASS"


def test_formal_window_mode_has_no_complete_bn_snapshot_cost(tmp_path, fake_timing, monkeypatch):
    c = components()
    session = SmallSession(c)
    c.engine.begin_epoch(1)
    monkeypatch.setattr(control, "_publish_bn_snapshot", lambda *a, **k: pytest.fail("formal BN snapshot"))
    observation_type = control._ForwardObservation
    def formal_observation(components, *, sampling=False):
        assert sampling is False
        return observation_type(components, sampling=sampling)
    monkeypatch.setattr(control, "_ForwardObservation", formal_observation)
    monkeypatch.setattr(observation_type, "sampling_evidence", lambda *a: pytest.fail("formal sampling snapshot"))
    receipts = []
    control._run_active_windows(
        session, FakeMonitor(), tmp_path, limit=1, paired_index={}, replay_index={},
        save_midpoint=False, snapshots=False, checkpoints={}, receipts=receipts, replay_checks=[],
    )
    record = control._read_json_reference(receipts[0], "formal window observation")
    assert "bn_snapshot_reference" not in record
    assert "sampling_core_observation" not in record
    assert session.calls == 1 and not session.loader.has_active_iterator


def _synthetic_replay_pair(tmp_path):
    original_dir, prelude_dir, replay_dir = (tmp_path / name for name in ("original", "prelude", "replay"))
    for path in (original_dir, prelude_dir, replay_dir):
        path.mkdir()
    original = SmallSession(components())
    original.components.engine.begin_epoch(1)
    original_receipts = []
    control._run_active_windows(
        original, FakeMonitor(), original_dir, limit=4, paired_index={}, replay_index={},
        save_midpoint=False, snapshots=True, checkpoints={}, receipts=original_receipts, replay_checks=[],
    )
    replay = SmallSession(components())
    replay.components.engine.begin_epoch(1)
    # Establish the same deterministic CPU second-window boundary. Checkpoint
    # serialization/restore itself is covered by the real session tests.
    control._run_active_windows(
        replay, FakeMonitor(), prelude_dir, limit=2, paired_index={}, replay_index={},
        save_midpoint=False, snapshots=True, checkpoints={}, receipts=[], replay_checks=[],
    )
    index = {(row["epoch"], row["logical_batch_index"]): row for row in original_receipts}
    return replay, replay_dir, index


@pytest.mark.parametrize("failed_window", [3, 4])
def test_replay_bn_failure_preserves_actual_receipt_values_and_diff_before_stop(
        tmp_path, fake_timing, monkeypatch, failed_window):
    replay, output, index = _synthetic_replay_pair(tmp_path)
    train = replay.train_batch
    def changed_bn(batch, hardware_probe=None):
        record = train(batch, hardware_probe=hardware_probe)
        if replay.components.engine.optimizer_updates == failed_window:
            with torch.no_grad():
                replay.components.model.bn.running_mean[0].add_(0.25)
                replay.components.ema.module.bn.running_var[1].add_(0.125)
        return record
    replay.train_batch = changed_bn
    comparator, writer = control.compare_replay_records, control.write_exclusive_json
    rejecting = False
    def observed_compare(expected, actual):
        nonlocal rejecting
        window = actual["window"]["optimizer_updates"]
        assert (output / "receipts" / "epoch-001" / f"window-{window:04d}.json").exists()
        assert (output / f"bn-state-window-{window:02d}.pt").exists()
        assert (output / f"replay-observation-window-{window:02d}.json").exists()
        rejecting = window == failed_window
        return comparator(expected, actual)
    def write_before_rejection(*args, **kwargs):
        assert not rejecting, "failure handler added diagnostic I/O before abort"
        return writer(*args, **kwargs)
    monkeypatch.setattr(control, "compare_replay_records", observed_compare)
    monkeypatch.setattr(control, "write_exclusive_json", write_before_rejection)
    receipts, checks = [], []
    with pytest.raises(control.ControlWorkerError, match="raw_bn mismatch") as caught:
        control._run_active_windows(
            replay, FakeMonitor(), output, limit=2, paired_index={}, replay_index=index,
            save_midpoint=False, snapshots=True, checkpoints={}, receipts=receipts, replay_checks=checks,
        )
    assert replay.calls == failed_window
    assert not replay.loader.has_active_iterator
    assert len(receipts) == failed_window - 2 and len(checks) == failed_window - 3
    actual = control._read_json_reference(receipts[-1], "failed actual receipt")
    full = control._load_bn_snapshot(actual["bn_snapshot_reference"], actual)
    report = control._read_json_reference(caught.value.replay_observation_reference, "failed comparison evidence")
    assert report["status"] == "OBSERVATION_ONLY" and report["acceptance_decision"] is False
    assert report["BN_values"]["available"] is True
    assert report["actual_receipt"]["sha256"] == receipts[-1]["sha256"]
    paths = {tuple(row["path"]) for row in report["exact_field_differences"]}
    assert ("state", "raw_bn", "bn", "running_mean", "sha256") in paths
    assert ("state", "ema_bn", "bn", "running_var", "sha256") in paths
    by_family = {row["family"]: row for row in report["BN_values"]["layers"]}
    assert by_family["raw_bn"]["running_mean"]["max_absolute"] == pytest.approx(0.25, abs=1e-7)
    assert by_family["ema_bn"]["running_var"]["max_absolute"] == pytest.approx(0.125, abs=1e-7)
    assert torch.equal(full["raw_bn"]["bn"]["running_mean"], replay.components.model.bn.running_mean)
    assert not (output / "checkpoint-window-04.pt").exists()


@pytest.mark.parametrize("family,key,value", [
    ("rng", "python", "0" * 64),
    ("clocks", "ema_updates", 999),
])
def test_replay_rng_or_clock_failure_retains_exact_field_evidence(
        tmp_path, fake_timing, monkeypatch, family, key, value):
    replay, output, index = _synthetic_replay_pair(tmp_path)
    measure = control._measure_window
    def changed_observation(*args, **kwargs):
        record = measure(*args, **kwargs)
        record["state"][family][key] = value
        return record
    monkeypatch.setattr(control, "_measure_window", changed_observation)
    receipts = []
    with pytest.raises(control.ControlWorkerError, match="same-arm replay " + family) as caught:
        control._run_active_windows(
            replay, FakeMonitor(), output, limit=2, paired_index={}, replay_index=index,
            save_midpoint=False, snapshots=True, checkpoints={}, receipts=receipts, replay_checks=[],
        )
    report = control._read_json_reference(caught.value.replay_observation_reference, "failure fields")
    changed = next(row for row in report["exact_field_differences"] if row["path"] == ["state", family, key])
    assert changed["actual"] == value and changed["actual"] != changed["expected"]
    assert replay.calls == 3 and len(receipts) == 1 and not replay.loader.has_active_iterator
    assert not (output / "receipts" / "epoch-001" / "window-0004.json").exists()


def test_missing_bn_diagnostic_is_unavailable_and_cannot_be_reported_as_zero(tmp_path):
    expected, actual = make_replay_record(tmp_path, "expected"), make_replay_record(tmp_path, "actual")
    left = write_exclusive_json(tmp_path / "expected.json", expected)
    right = write_exclusive_json(tmp_path / "actual.json", actual)
    with pytest.raises(control.ControlWorkerError, match="BN diagnostic evidence is unavailable") as caught:
        control._publish_replay_observation(
            tmp_path / "observation.json", expected, actual, expected_reference=left, actual_reference=right,
        )
    report = control._read_json_reference(caught.value.replay_observation_reference, "unavailable diagnosis")
    assert report["BN_values"]["available"] is False
    assert report["BN_values"]["layers"] is None
    assert report["BN_values"]["error"]["type"] == "KeyError"
    assert report["acceptance_decision"] is False


def test_worker_failure_is_durable_before_hardware_abort(contract, tmp_path, monkeypatch):
    # Fail before input construction/admission; a syntactically valid contract
    # can never turn failed full verification into PASS.
    path = tmp_path / "worker-contract.json"
    reference = write_exclusive_json(path, contract)
    observation = write_exclusive_json(tmp_path / "observation.json", {"status": "OBSERVATION_ONLY"})
    original = control.validate_worker_contract
    def validate(value, *, verify_files=True):
        if verify_files:
            error = control.ControlWorkerError("injected frozen source mismatch")
            error.replay_observation_reference = observation
            raise error
        return original(value, verify_files=False)
    monkeypatch.setattr(control, "validate_worker_contract", validate)
    monkeypatch.setattr(control, "_environment", lambda value: {"synthetic": "cpu"})
    monkeypatch.setattr(control, "build_v2b_components", lambda *a, **k: pytest.fail("model constructed after failed contract"))
    with pytest.raises(control.ControlWorkerError, match="source mismatch"):
        control.run_worker(contract, contract_reference=reference)
    result = json.loads((Path(contract["output_dir"]) / "worker-result.json").read_text())
    assert result["status"] == "STOP_NO_RETRY"
    assert result["receipts"] == [] and result["checkpoints"] == {}
    assert result["automatic_retry_or_batch_fallback"] is False
    assert result["failure"][0]["observation_reference"] == observation


def test_smoke_comparison_of_incomplete_artifacts_returns_stop(tmp_path):
    paths = [tmp_path / name for name in ("A", "Ar", "B", "Br")]
    for path in paths:
        path.mkdir()
        write_exclusive_json(path / "worker-result.json", {"status": "STOP_NO_RETRY"})
    result = control.compare_smoke_outputs(*paths)
    assert result["status"] == "STOP_NO_RETRY" and not result["scientific_certified"]


def test_cli_hash_failure_happens_before_worker_import_or_execution(tmp_path):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("v2b_control_cli_test", root / "tools/run_training_v2b_control.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "contract.json"
    path.write_bytes(b"{}")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        module.main(["--contract", str(path), "--contract-sha256", "0" * 64])


def test_native_directory_owned_by_monitor_in_complete_cpu_worker(contract, tmp_path, monkeypatch, fake_timing):
    """Actual worker flow/updates, with native APIs replaced only in this CPU test."""
    from sparse_rtdetr.baseline import training_v2b_admission as admission
    from sparse_rtdetr.baseline import training_v2b_development as development
    from sparse_rtdetr.baseline.training_v2b_data import TrainCoreDataConfig

    original_validate = control.validate_worker_contract
    monkeypatch.setattr(control, "validate_worker_contract",
                        lambda value, *, verify_files=True: original_validate(value, verify_files=False))
    monkeypatch.setattr(control, "_environment", lambda value: {"synthetic": "cpu"})
    real_cpu_prepare = prepare_runtime
    monkeypatch.setattr(control, "prepare_runtime",
                        lambda **kwargs: real_cpu_prepare(seed=kwargs["seed"]))

    def build(config, **kwargs):
        runtime = kwargs.get("runtime") or real_cpu_prepare(seed=config.seed)
        seed_cpu_sources(config.seed)
        model, criterion = SmallDetector(), SmallLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        ema, warmup = SmallEMA(model), SmallWarmup()
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000], gamma=0.1)
        engine = AccumulationEngine(
            model, criterion, optimizer, physical_batch_size=config.physical_batch_size,
            accumulation_steps=config.accumulation_steps, amp_dtype="bfloat16",
            ema=ema, warmup=warmup, scheduler=scheduler,
        )
        initialization = {"parameters": control.initial_parameter_reference(dict(model.named_parameters())),
                          "model_state": control.initial_parameter_reference(model.state_dict())}
        return SimpleNamespace(config=config, runtime=runtime, model=model, criterion=criterion,
                               optimizer=optimizer, ema=ema, warmup=warmup, scheduler=scheduler,
                               engine=engine, initialization=initialization, postprocessor=nn.Identity())

    class Loader(SmallLoader):
        def __init__(self, config):
            self.config = config
            self.c = None
            self.dataset, self.batches_per_epoch = list(range(4869)), 304
            self.has_active_iterator = False
        def state_dict(self):
            return {
                "optimizer_updates": self.c.engine.optimizer_updates,
                "epoch": self.c.engine.epoch, "epoch_active": self.c.engine.epoch_active,
                "order_sha256": contract["epoch_order_sha256"]["1"],
            }
        def __iter__(self):
            self.has_active_iterator = True
            try:
                plan = list(logical_batch_indices(4869, seed=0, epoch=1, logical_batch_size=16))
                for index in range(self.c.engine.optimizer_updates, 304):
                    evidence = receipt(plan[index], index=index)
                    yield SimpleNamespace(
                        epoch=1, logical_batch_index=index,
                        images=torch.arange(256).float().reshape(16, 1, 4, 4) / 256,
                        targets=[targets()[i % 2] for i in range(16)],
                        evidence=lambda evidence=evidence: copy.deepcopy(evidence),
                    )
            finally:
                self.has_active_iterator = False

    class Session(SmallSession):
        def __init__(self, components, loader, binding):
            self.components, self.loader, self.calls = components, loader, 0
            self.binding = binding
            loader.c = components
        def begin_epoch(self, epoch):
            self.components.engine.begin_epoch(epoch)

    class Monitor(FakeMonitor):
        instances = []
        def __init__(self, binding, policy, evidence_dir, scope, deadline):
            super().__init__()
            self.directory, self.scope, self.deadline = evidence_dir, scope, deadline
            self.instances.append(self)
        def start(self):
            # This is the real native session's exclusive directory lifecycle.
            self.directory.mkdir(parents=False, exist_ok=False)
            self.events.append("started")
        def finish(self):
            self.events.append("finished")
            return {"synthetic_cpu_only": True, "worker_must_exit": True}
        def abort(self, reason):
            self.events.append("abort")
            pytest.fail("successful CPU lifecycle unexpectedly aborted: " + reason)

    def build_binding(cpu, loader, **kwargs):
        return {
            "binding_sha256": "a" * 64,
            "config": cpu.config.binding_config(device="cuda:0",
                       scope="train_core_runtime_engineering", cuda_gpu_uuid=contract["expected_gpu_uuid"]),
            "code": contract["code_files"], "data": {"synthetic_cpu_only": True},
            "initial_weights": cpu.initialization["model_state"],
        }

    def checkpoint(session, output, name, monitor):
        checkpoint = write_exclusive_json(output / (name + ".pt"), {"synthetic_cpu_fixture": True})
        state_ref = write_exclusive_json(output / (name + "-state.json"),
                                        control.capture_control_state(session.components, session.loader, full=True))
        return {"checkpoint": checkpoint, "state_reference": state_ref}

    previews = []
    def preview(component_map, **kwargs):
        assert {"optimizer", "scheduler", "warmup", "model", "ema", "runtime"} <= set(component_map)
        assert component_map["engine"].optimizer_updates == 4
        kwargs["hardware_gate"]()
        kwargs["hardware_gate"]()
        previews.append(1)
        return {"scope": "development_preview_only", "synthetic_cpu_fixture": True}

    monkeypatch.setattr(control, "build_v2b_components", build)
    monkeypatch.setattr(control, "build_train_core_loader", lambda config, **kwargs: Loader(config))
    monkeypatch.setattr(control, "build_train_core_run_binding", build_binding)
    monkeypatch.setattr(control, "V2BTrainingSession", Session)
    monkeypatch.setattr(control, "_write_checkpoint", checkpoint)
    monkeypatch.setattr(admission, "MonitoredHardwareSession", Monitor)
    monkeypatch.setattr(development, "preview_development", preview)
    reference = write_exclusive_json(tmp_path / "contract.json", contract)
    result = control.run_worker(contract, contract_reference=reference)
    assert result["status"] == "PASS"
    assert result["receipt_count"] == 4
    assert set(result["checkpoints"]) == {"window_2", "window_4"}
    assert previews == [1] and Monitor.instances[0].events[-1] == "finished"
    assert Monitor.instances[0].scope == "paired_smoke" and Monitor.instances[0].deadline == 600
    records = [control._read_json_reference(ref, "receipt") for ref in result["receipts"]]
    assert [row["window"]["optimizer_updates"] for row in records] == [1, 2, 3, 4]
    assert [row["state"]["raw_bn"]["bn"]["num_batches_tracked"] for row in records] == [1, 2, 3, 4]


def test_real_loader_final_acknowledgment_survives_explicit_close_and_finish(tmp_path):
    """Exercise the actual generator/finally protocol with real R18 CPU updates."""
    from test_rtdetr_baseline_training_v2b_runtime import make_files, session
    files = make_files(tmp_path, count=4)
    current, model, loader, binding = session(files)
    current.begin_epoch(1)
    iterator = iter(loader)
    for _ in range(loader.batches_per_epoch):
        batch = next(iterator)
        current.train_batch(batch)
    # There is deliberately no extra next() to reach StopIteration.
    assert loader.has_active_iterator
    assert loader.state_dict()["completed_batches"] == loader.batches_per_epoch
    iterator.close()
    assert not loader.has_active_iterator
    result = current.finish_epoch()
    assert result["loader"]["epoch_active"] is False
    assert result["epoch"]["epoch_optimizer_steps"] == 2
    assert model.engine.optimizer_updates == 2


def test_preview_comparison_preserves_metric_roles_and_does_not_compare_elapsed_or_paths():
    original = {
        "data": {"evaluated_image_count": 4, "image_receipts": [1, 2, 3, 4]},
        "data_binding_sha256": "a" * 64, "policy": {"dtype": "float32", "weights": "ema"},
        "weights": {"state_sha256": "b" * 64}, "prediction_artifact": {"sha256": "c" * 64},
        "primary_legacy_formal_gt": {"algorithm_id": "primary", "metrics": {"AP": 12.3}},
        "coco_secondary": {"axes": {"area": ["all", "small"]}, "metrics": {"AP_small": {"value": 4.2}}},
        "evaluation_elapsed_seconds": 1.0, "result_artifact": {"path": "/one"},
    }
    replay = copy.deepcopy(original)
    replay["evaluation_elapsed_seconds"] = 999.0
    replay["result_artifact"]["path"] = "/two"
    result = control._compare_previews(original, replay)
    assert result["primary_metrics_exact"] and result["secondary_metrics_exact"]
    replay["coco_secondary"]["metrics"]["AP_small"]["value"] = 4.3
    result = control._compare_previews(original, replay)
    assert result["primary_metrics_exact"] and not result["secondary_metrics_exact"]
    assert not result["scientific_comparison_eligible"]
    replay["coco_secondary"]["axes"]["area"] = ["all", "large"]
    with pytest.raises(control.ControlWorkerError, match="axes"):
        control._compare_previews(original, replay)


def test_common_initial_state_requires_actual_rng_and_buffers_not_only_seed():
    c = components()
    initial = control.capture_control_state(c, SmallLoader(c), full=True)
    control._verify_common_initial_state(initial, copy.deepcopy(initial))
    altered = copy.deepcopy(initial)
    altered["rng"]["torch_cpu"]["sha256"] = "0" * 64
    with pytest.raises(control.ControlWorkerError, match="shared CUDA initialization rng"):
        control._verify_common_initial_state(initial, altered)


@pytest.fixture
def continuation_fixture(tmp_path):
    """Actual CPU AdamW/EMA checkpoints, with no model or hardware doubles."""
    from test_rtdetr_baseline_training_v2b_checkpoint import make_state, batch, BINDING
    from sparse_rtdetr.baseline.training_v2b_checkpoint import save_checkpoint
    state = make_state()
    state["engine"].begin_epoch(1)
    for _ in range(4):
        state["engine"].train_window(*batch(state))
    directory = tmp_path / "original"
    directory.mkdir()
    reference = save_checkpoint(directory / "checkpoint-window-04.pt", binding=BINDING, **state)
    payload = torch.load(reference["path"], map_location="cpu", weights_only=True)
    return reference, payload, copy.deepcopy(BINDING)


def _checkpoint_payload(tmp_path, payload, name="altered"):
    directory = tmp_path / name
    directory.mkdir()
    path = directory / "checkpoint-window-04.pt"
    with path.open("xb") as stream:
        torch.save(payload, stream)
    return file_reference(path)


def test_authenticated_ema_and_moments_continuation_accepts_only_declared_small_drift(continuation_fixture, tmp_path):
    reference, payload, binding = continuation_fixture
    ema = next(value for name, value in payload["ema"]["module"].items() if name.endswith("weight"))
    ema.flatten()[0] += 5e-7
    moment = next(iter(payload["optimizer"]["state"].values()))["exp_avg"]
    offset = int(moment.abs().flatten().argmax())
    moment.flatten()[offset] *= 1.000001
    result = control.compare_continuation_checkpoints(reference, _checkpoint_payload(tmp_path, payload), binding=binding)
    assert result["status"] == "PASS"
    for family in ("EMA", "AdamW_moments"):
        row = result[family]
        assert row["reference_L2"] > 0
        assert 0 < row["difference_L2"] and 0 < row["max_absolute"] <= 1e-5
        assert row["relative_L2"] <= 1e-5
        assert row["thresholds"] == {"relative_L2": 1e-5, "max_absolute": 1e-5}
        assert row["worst_absolute_tensor"]["path"]
    assert result["families_are_aggregated_separately"]
    assert any(path.endswith("/step") for path in result["AdamW_moments"]["exact_tensors"])


@pytest.mark.parametrize("mutation", [
    "ema_absolute", "moment_absolute", "ema_nan", "moment_inf", "missing_ema",
    "missing_moment", "moment_shape", "moment_dtype", "optimizer_step", "learning_rate",
    "bn_buffer",
])
def test_authenticated_continuation_tampering_stops(continuation_fixture, tmp_path, mutation):
    reference, payload, binding = continuation_fixture
    parameter_name = next(name for name in payload["ema"]["module"] if name.endswith("weight"))
    ema = payload["ema"]["module"][parameter_name]
    state = next(iter(payload["optimizer"]["state"].values()))
    if mutation == "ema_absolute":
        ema.flatten()[0] += 1e-3
    elif mutation == "moment_absolute":
        state["exp_avg"].flatten()[0] += 1e-3
    elif mutation == "ema_nan":
        ema.flatten()[0] = float("nan")
    elif mutation == "moment_inf":
        state["exp_avg"].flatten()[0] = float("inf")
    elif mutation == "missing_ema":
        del payload["ema"]["module"][parameter_name]
    elif mutation == "missing_moment":
        del state["exp_avg"]
    elif mutation == "moment_shape":
        state["exp_avg"] = state["exp_avg"].flatten()[:1]
    elif mutation == "moment_dtype":
        state["exp_avg"] = state["exp_avg"].double()
    elif mutation == "optimizer_step":
        state["step"] -= 1
    elif mutation == "learning_rate":
        payload["optimizer"]["param_groups"][0]["lr"] *= 1.000001
    elif mutation == "bn_buffer":
        payload["ema"]["module"]["bn.running_mean"][0] += 1e-7
    altered = _checkpoint_payload(tmp_path, payload)
    with pytest.raises((control.ControlWorkerError, ValueError, RuntimeError)):
        control.compare_continuation_checkpoints(reference, altered, binding=binding)


def test_relative_moment_error_cannot_be_hidden_by_ema_scale():
    baseline = {"state": {0: {"exp_avg": torch.full((20,), 1e-8),
                              "exp_avg_sq": torch.full((20,), 1e-12),
                              "step": torch.tensor(4.)}},
                "param_groups": [{"lr": 1e-4, "params": [0]}]}
    altered = copy.deepcopy(baseline)
    altered["state"][0]["exp_avg"][0] += 1e-9  # absolute passes; family-relative does not
    with pytest.raises(control.ControlWorkerError, match="relative_L2"):
        control._compare_state_tree(baseline, altered, family="AdamW_moments", optimizer_moments_only=True)
    # A huge EMA norm is never included in this family's denominator.
    result = control._compare_state_tree(
        {"weight": torch.tensor([1e4])}, {"weight": torch.tensor([1e4])}, family="EMA",
    )
    assert result["reference_L2"] == 1e4


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_anchor_name_alone_never_grants_a_nonfinite_exception(value):
    expected = {"module": {"invented.anchors": torch.tensor([value])}, "updates": 4}
    layout = {"model": {
        "modules": {"invented": "synthetic.UntrustedDecoder"},
        "parameters": [], "state": {"invented.anchors": {"shape": [1], "dtype": "torch.float32"}},
        "caches": {"invented:persistent_cache:anchors": control._tensor_identity(expected["module"]["invented.anchors"])},
    }}
    with pytest.raises(control.ControlWorkerError, match="vendor module identity"):
        control._ema_exact_tensor_paths(expected, layout)
    with pytest.raises(control.ControlWorkerError, match="nonfinite"):
        control._compare_state_tree(expected, copy.deepcopy(expected), family="EMA")


def test_real_r18_anchor_positive_infinity_is_only_an_exact_authenticated_geometry_exception():
    from sparse_rtdetr.baseline.training_v2b import build_v2b_components
    from sparse_rtdetr.baseline.training_v2b_checkpoint import _ema
    root = Path(__file__).resolve().parents[1]
    c = build_v2b_components(V2BConfig(pretrained_required=False), repo_root=root)
    state, layout = _ema(c.ema, updates=0)
    exact, sentinel = control._ema_exact_tensor_paths(state, layout)
    assert sentinel == {("module", "decoder.anchors")}
    anchors = state["module"]["decoder.anchors"]
    assert torch.isposinf(anchors).any()
    # Keep the real, certified geometry tensor while making the numerical
    # family small; this test does not run inference or train the real model.
    expected = {"module": {"decoder.anchors": anchors, "learned.weight": torch.ones(2)}, "updates": 4}
    actual = copy.deepcopy(expected)
    report = control._compare_state_tree(expected, actual, family="EMA",
                                        exact_tensor_paths=exact, positive_infinity_paths=sentinel)
    assert report["positive_infinity_sentinel_exception"]["tensors_with_sentinel"] == ["module/decoder.anchors"]
    assert report["reference_L2"] == pytest.approx(2**0.5)
    flat = actual["module"]["decoder.anchors"].flatten()
    index = int(torch.isposinf(flat).nonzero()[0])
    flat[index] = 0.5
    with pytest.raises(control.ControlWorkerError, match="exact tensor"):
        control._compare_state_tree(expected, actual, family="EMA",
                                    exact_tensor_paths=exact, positive_infinity_paths=sentinel)
    for invalid in (float("nan"), -float("inf")):
        actual = copy.deepcopy(expected)
        actual["module"]["decoder.anchors"].flatten()[index] = invalid
        with pytest.raises(control.ControlWorkerError, match="invalid anchor"):
            control._compare_state_tree(expected, actual, family="EMA",
                                        exact_tensor_paths=exact, positive_infinity_paths=sentinel)


def test_continuation_reference_hash_is_checked_before_any_deserialization(continuation_fixture, monkeypatch):
    reference, payload, binding = continuation_fixture
    changed = dict(reference, sha256="0" * 64)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("unverified checkpoint deserialization"))
    with pytest.raises((control.ControlWorkerError, ValueError, RuntimeError), match="SHA256 mismatch"):
        control.compare_continuation_checkpoints(changed, reference, binding=binding)


def test_smoke_comparison_cannot_turn_moment_gate_failure_into_formal_pass(continuation_fixture, tmp_path, monkeypatch):
    reference, payload, binding = continuation_fixture
    next(iter(payload["optimizer"]["state"].values()))["exp_avg"].flatten()[0] += 0.1
    altered = _checkpoint_payload(tmp_path, payload)
    base = {
        "stage": "smoke", "arm": "A", "run_id": "A-run", "worker_pid": 1,
        "common_identity": {}, "runtime": {},
        "initial_state_reference": {"kind": "initial"},
        "final_state_reference": {"kind": "final"},
        "binding_reference": {"kind": "binding"},
        "checkpoints": {"window_2": {"state_reference": {"kind": "midpoint"}},
                        "window_4": {"checkpoint": reference}},
    }
    replay = copy.deepcopy(base)
    replay.update(stage="smoke_replay", worker_pid=2, restore_boundary_reference={"kind": "midpoint"})
    replay["checkpoints"]["window_4"]["checkpoint"] = altered
    outcomes = iter([(base, {}), (replay, {}), ({}, {}), ({}, {})])
    monkeypatch.setattr(control, "_result_from_dir", lambda path: next(outcomes))
    monkeypatch.setattr(control, "_verify_common_initial_state", lambda *args: None)
    def index(value):
        return {(1, index): {"kind": "window"} for index in
                (range(4) if value["stage"] == "smoke" else (2, 3))}
    monkeypatch.setattr(control, "_receipt_index", index)
    original_reader = control._read_json_reference
    def read(ref, label):
        if ref.get("kind") == "binding":
            return binding
        if ref.get("kind") == "final":
            return {"clocks": {}, "rng": {}, "raw_bn": {}, "ema_bn": {}}
        if "kind" in ref:
            return {}
        return original_reader(ref, label)
    monkeypatch.setattr(control, "_read_json_reference", read)
    monkeypatch.setattr(control, "compare_replay_records", lambda *args: {"status": "PASS"})
    result = control.compare_smoke_outputs("/unused-A", "/unused-Ar", "/unused-B", "/unused-Br")
    assert result["status"] == "STOP_NO_RETRY"
    assert "AdamW_moments" in result["failure"]
