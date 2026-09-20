from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from sparse_rtdetr.baseline.training_v2b_engine import (
    AccumulationEngine,
    AccumulationEngineError,
    validate_engine_state_dict,
    BN_BACKWARD_LAYOUT,
    _batchnorm_backward_layout,
)


class TinyDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(1, 1, 1, bias=False)
        with torch.no_grad():
            self.projection.weight.fill_(0.3)

    def forward(self, images, targets=None):
        return self.projection(images).mean(dim=(1, 2, 3))


class TargetNormalizedLoss(nn.Module):
    """Real differentiable classification/regression, not a fake loss dictionary."""

    def __init__(self, dynamic_aux=False) -> None:
        super().__init__()
        self.dynamic_aux = dynamic_aux
        self.autocast_observations = []

    def forward(self, predictions, targets):
        self.autocast_observations.append(torch.is_autocast_enabled("cpu"))
        counts = predictions.new_tensor([len(target["labels"]) for target in targets])
        denominator = max(sum(len(target["labels"]) for target in targets), 1)
        positives = (counts > 0).to(dtype=predictions.dtype)
        losses = {
            "classification": F.binary_cross_entropy_with_logits(
                predictions, positives, reduction="sum"
            ) / denominator,
            "regression": 5 * (
                (predictions.sigmoid() - 0.4).abs() * counts
            ).sum() / denominator,
        }
        if self.dynamic_aux and counts.sum() > 0:
            losses["presence_aux"] = predictions.square().sum() / denominator
        return losses


class MutatingDetector(TinyDetector):
    """Model-side in-place writes must not mutate loader-owned inputs."""

    def forward(self, images, targets=None):
        images.add_(0.25)
        if targets:
            targets[0]["boxes"].mul_(0.5)
        return super().forward(images, targets=targets)


class NestedMutatingDetector(TinyDetector):
    """Nested mutable target tensors must also be isolated from the loader."""

    def forward(self, images, targets=None):
        images.add_(0.25)
        targets[0]["nested"]["boxes"].mul_(0.25)
        targets[0]["nested"]["deep"][0]["labels"].add_(1)
        return super().forward(images, targets=targets)


class CountingSGD(torch.optim.SGD):
    def __init__(self, parameters, lr=0.07):
        super().__init__(parameters, lr=lr)
        self.steps = 0
        self.zeroes = 0

    def step(self, closure=None):
        self.steps += 1
        return super().step(closure)

    def zero_grad(self, set_to_none=True):
        self.zeroes += 1
        return super().zero_grad(set_to_none=set_to_none)


class CountingEMA:
    def __init__(self):
        self.updates = 0

    def update(self, model):
        self.updates += 1


class CountingWarmup:
    def __init__(self, duration=2):
        self.steps = 0
        self.duration = duration

    def step(self):
        self.steps += 1

    def finished(self):
        return self.steps >= self.duration


class CountingScheduler:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1


def targets_for(counts):
    return [
        {
            "labels": torch.zeros(count, dtype=torch.long),
            "boxes": torch.full((count, 4), 0.25),
        }
        for count in counts
    ]


def images_for(batch=4):
    return torch.arange(1, batch + 1, dtype=torch.float32).reshape(batch, 1, 1, 1)


def make_engine(*, model=None, criterion=None, physical=2, accum=2, **kwargs):
    model = TinyDetector() if model is None else model
    criterion = TargetNormalizedLoss() if criterion is None else criterion
    optimizer = CountingSGD(model.parameters())
    engine = AccumulationEngine(
        model,
        criterion,
        optimizer,
        physical_batch_size=physical,
        accumulation_steps=accum,
        amp_dtype=kwargs.pop("amp_dtype", "float32"),
        clip_max_norm=kwargs.pop("clip_max_norm", 1e6),
        **kwargs,
    )
    return engine, model, criterion, optimizer


@pytest.mark.parametrize("counts", [
    [0, 1, 3, 0],
    [0, 0, 0, 0],
    [0, 0, 2, 0],
    [1, 0, 2, 0],
])
def test_logical_gt_normalization_matches_real_full_batch_gradient(counts):
    images = images_for()
    targets = targets_for(counts)
    engine, model, criterion, optimizer = make_engine()
    reference = copy.deepcopy(model)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.07)
    reference_loss = sum(TargetNormalizedLoss()(reference(images), targets).values())
    reference_loss.backward()
    reference_gradient = reference.projection.weight.grad.detach().clone()
    reference_optimizer.step()

    engine.begin_epoch(1)
    report = engine.train_window(images, targets)
    torch.testing.assert_close(
        model.projection.weight, reference.projection.weight, rtol=1e-6, atol=1e-7
    )
    torch.testing.assert_close(
        model.projection.weight.grad, reference_gradient, rtol=1e-6, atol=1e-7
    )
    assert report["loss"] == pytest.approx(reference_loss.item(), rel=2e-6)
    denominator = max(sum(counts), 1)
    assert report["coefficients"] == [
        max(sum(counts[:2]), 1) / denominator,
        max(sum(counts[2:]), 1) / denominator,
    ]
    assert optimizer.steps == 1
    assert optimizer.zeroes == 1
    assert engine.microsteps == 2 and engine.optimizer_updates == 1


def test_accumulation_one_preserves_vendor_preweighted_sum():
    engine, model, criterion, _ = make_engine(physical=4, accum=1)
    images, targets = images_for(), targets_for([0, 1, 3, 0])
    expected = sum(criterion(model(images), targets).values()).item()
    engine.begin_epoch(1)
    report = engine.train_window(images, targets)
    assert report["coefficients"] == [1.0]
    assert report["loss"] == pytest.approx(expected)


def test_engine_owns_inputs_before_vendor_forward_mutation():
    engine, _, _, _ = make_engine(model=MutatingDetector())
    images = images_for()
    targets = targets_for([0, 1, 3, 0])
    original_images = images.clone()
    original_boxes = [target["boxes"].clone() for target in targets]
    engine.begin_epoch(1)
    engine.train_window(images, targets)
    assert torch.equal(images, original_images)
    for target, boxes in zip(targets, original_boxes):
        assert torch.equal(target["boxes"], boxes)


def test_engine_deepcopies_nested_target_tensors_before_vendor_forward_mutation():
    engine, _, _, _ = make_engine(model=NestedMutatingDetector())
    images = images_for()
    targets = targets_for([0, 1, 3, 0])
    for target in targets:
        target["nested"] = {
            "boxes": target["boxes"].clone(),
            "deep": [{"labels": target["labels"].clone()}],
        }
    original_images = images.clone()
    original_nested_boxes = [target["nested"]["boxes"].clone() for target in targets]
    original_nested_labels = [
        target["nested"]["deep"][0]["labels"].clone() for target in targets
    ]
    engine.begin_epoch(1)
    engine.train_window(images, targets)
    assert torch.equal(images, original_images)
    for target, boxes, labels in zip(targets, original_nested_boxes, original_nested_labels):
        assert torch.equal(target["nested"]["boxes"], boxes)
        assert torch.equal(target["nested"]["deep"][0]["labels"], labels)


def test_dynamic_loss_keys_and_empty_microbatch_are_not_forced_to_21():
    engine, _, _, _ = make_engine(criterion=TargetNormalizedLoss(dynamic_aux=True))
    engine.begin_epoch(1)
    report = engine.train_window(images_for(), targets_for([0, 0, 1, 0]))
    assert set(report["microbatch_losses"][0]) == {"classification", "regression"}
    assert "presence_aux" in report["microbatch_losses"][1]
    assert report["coefficients"] == [1.0, 1.0]


class FailSecondLoss(TargetNormalizedLoss):
    def __init__(self, gradient=False):
        super().__init__()
        self.calls = 0
        self.gradient = gradient

    def forward(self, predictions, targets):
        self.calls += 1
        if self.calls == 2:
            if self.gradient:
                # Forward is finite zero; derivative of sqrt at zero is infinite.
                return {"bad_gradient": (predictions - predictions.detach()).sqrt().sum()}
            return {"bad_loss": predictions.sum() * float("nan")}
        return super().forward(predictions, targets)


@pytest.mark.parametrize("gradient", [False, True])
def test_nonfinite_second_microbatch_never_updates_and_invalidates_engine(gradient):
    ema, warmup = CountingEMA(), CountingWarmup()
    engine, model, _, optimizer = make_engine(
        criterion=FailSecondLoss(gradient), ema=ema, warmup=warmup
    )
    initial = copy.deepcopy(model.state_dict())
    engine.begin_epoch(1)
    with pytest.raises(AccumulationEngineError, match="nonfinite"):
        engine.train_window(images_for(), targets_for([1, 0, 2, 0]))
    assert engine.failed
    assert optimizer.steps == ema.updates == warmup.steps == 0
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, initial[name])
    assert all(parameter.grad is None for parameter in model.parameters())
    with pytest.raises(AccumulationEngineError, match="failed"):
        engine.train_window(images_for(), targets_for([1, 0, 2, 0]))
    with pytest.raises(AccumulationEngineError, match="failed"):
        engine.state_dict()


def test_clip_optimizer_ema_warmup_once_per_window_and_scheduler_after_warmup(monkeypatch):
    clip_calls = []
    real_clip = torch.nn.utils.clip_grad_norm_

    def clip(*args, **kwargs):
        clip_calls.append(kwargs)
        return real_clip(*args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
    ema, warmup, scheduler = CountingEMA(), CountingWarmup(2), CountingScheduler()
    engine, _, _, optimizer = make_engine(
        ema=ema, warmup=warmup, scheduler=scheduler, clip_max_norm=0.1
    )
    engine.begin_epoch(1)
    first = engine.train_window(images_for(), targets_for([1, 0, 2, 0]))
    assert first["gradient_norm_before_clip"] > 0.1
    assert optimizer.steps == ema.updates == warmup.steps == len(clip_calls) == 1
    assert clip_calls[0]["error_if_nonfinite"] is True
    assert engine.finish_epoch(expected_optimizer_steps=1)["scheduler_stepped"] is False
    engine.begin_epoch(2)
    engine.train_window(images_for(), targets_for([1, 0, 2, 0]))
    assert engine.finish_epoch(expected_optimizer_steps=1)["scheduler_stepped"] is True
    assert optimizer.steps == ema.updates == warmup.steps == len(clip_calls) == 2
    assert scheduler.steps == 1
    assert engine.microsteps == 4


class BNDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.Conv2d(2, 1, 1, bias=False),
            nn.BatchNorm2d(1),
        )

    def forward(self, images, targets=None):
        return self.body(images).mean(dim=(1, 2, 3))


@pytest.mark.parametrize("mode,expected_batches", [("frozen", 0), ("train", 2)])
def test_bn_statistics_policy_reapplied_after_model_train_without_freezing_affine(mode, expected_batches):
    engine, model, _, _ = make_engine(model=BNDetector(), bn_statistics=mode)
    bn_modules = [module for module in model.modules() if isinstance(module, nn.BatchNorm2d)]
    initial_statistics = [
        (module.running_mean.clone(), module.running_var.clone()) for module in bn_modules
    ]
    engine.begin_epoch(1)
    model.train()  # The engine must reapply its policy, not trust this external mode.
    engine.train_window(
        images_for().expand(4, 1, 2, 2).clone(), targets_for([0, 1, 2, 0])
    )
    for module, (mean, variance) in zip(bn_modules, initial_statistics):
        assert module.num_batches_tracked.item() == expected_batches
        assert module.weight.requires_grad and module.bias.requires_grad
        assert module.weight.grad is not None and module.bias.grad is not None
        assert module.training is (mode == "train")
        if mode == "frozen":
            torch.testing.assert_close(module.running_mean, mean)
            torch.testing.assert_close(module.running_var, variance)


def test_bfloat16_autocast_only_wraps_forward_and_criterion_not_backward():
    engine, model, criterion, _ = make_engine(amp_dtype="bfloat16")
    backward_autocast = []
    handle = model.projection.weight.register_hook(
        lambda gradient: backward_autocast.append(torch.is_autocast_enabled("cpu"))
    )
    engine.begin_epoch(1)
    engine.train_window(images_for(), targets_for([0, 1, 3, 0]))
    handle.remove()
    assert criterion.autocast_observations == [True, True]
    assert backward_autocast == [False, False]


@pytest.mark.parametrize("kind", ["tail", "image_dimensions", "target_count", "boxes", "labels"])
def test_invalid_logical_windows_fail_before_forward_or_update(kind):
    images, targets = images_for(), targets_for([0, 1, 3, 0])
    if kind == "tail":
        images, targets = images[:3], targets[:3]
    elif kind == "image_dimensions":
        images = images.reshape(4, 1)
    elif kind == "target_count":
        targets = targets[:3]
    elif kind == "boxes":
        targets[1]["boxes"] = torch.zeros(2, 4)
    else:
        targets[1]["labels"] = torch.ones(1, dtype=torch.float32)
    engine, _, criterion, optimizer = make_engine()
    engine.begin_epoch(1)
    with pytest.raises(AccumulationEngineError):
        engine.train_window(images, targets)
    assert engine.failed and optimizer.steps == 0
    assert criterion.autocast_observations == []


@pytest.mark.parametrize("kind", ["empty", "number", "vector", "not_dict"])
def test_invalid_loss_contract_is_rejected(kind):
    class InvalidLoss(nn.Module):
        def forward(self, outputs, targets):
            if kind == "empty":
                return {}
            if kind == "number":
                return {"loss": 1.0}
            if kind == "vector":
                return {"loss": outputs}
            return outputs.sum()

    engine, _, _, optimizer = make_engine(criterion=InvalidLoss())
    engine.begin_epoch(1)
    with pytest.raises(AccumulationEngineError):
        engine.train_window(images_for(), targets_for([0, 1, 3, 0]))
    assert engine.failed and optimizer.steps == 0


def test_engine_state_roundtrip_restores_only_counters_and_allows_window_boundary_resume():
    engine, _, _, _ = make_engine()
    engine.begin_epoch(1)
    engine.train_window(images_for(), targets_for([0, 1, 3, 0]))
    state = engine.state_dict()
    restored, model, _, optimizer = make_engine()
    before_model = copy.deepcopy(model.state_dict())
    restored.load_state_dict(state)
    assert restored.state_dict() == state
    assert optimizer.steps == 0
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, before_model[name])
    restored.finish_epoch(expected_optimizer_steps=1)
    restored.begin_epoch(2)
    assert restored.epoch_start_optimizer_updates == 1


@pytest.mark.parametrize("mutation", [
    {"phase": 1},
    {"failed": True},
    {"microsteps": 3},
    {"optimizer_updates": True},
    {"epoch_active": 1},
    {"epoch": 0},
    {"epoch_start_optimizer_updates": 2},
    {"schema_version": 2},
    {"unexpected": 0},
])
def test_invalid_state_is_rejected_atomically(mutation):
    engine, _, _, _ = make_engine()
    engine.begin_epoch(1)
    engine.train_window(images_for(), targets_for([0, 1, 3, 0]))
    original = engine.state_dict()
    invalid = copy.deepcopy(original)
    invalid.update(mutation)
    with pytest.raises(AccumulationEngineError):
        engine.load_state_dict(invalid)
    assert engine.state_dict() == original


def test_state_configuration_mismatch_and_partial_window_rejected():
    engine, _, _, _ = make_engine()
    state = engine.state_dict()
    state["config"]["accumulation_steps"] = 1
    with pytest.raises(AccumulationEngineError, match="configuration"):
        engine.load_state_dict(state)
    engine.phase = 1
    with pytest.raises(AccumulationEngineError, match="inside"):
        engine.state_dict()


def test_epoch_order_and_observed_step_count_must_be_verified():
    engine, _, _, _ = make_engine()
    with pytest.raises(AccumulationEngineError, match="begin_epoch"):
        engine.train_window(images_for(), targets_for([0, 1, 3, 0]))
    with pytest.raises(AccumulationEngineError, match="consecutively"):
        engine.begin_epoch(2)
    engine.begin_epoch(1)
    engine.train_window(images_for(), targets_for([0, 1, 3, 0]))
    with pytest.raises(AccumulationEngineError, match="count mismatch"):
        engine.finish_epoch(expected_optimizer_steps=304)
    assert engine.failed


@pytest.mark.parametrize("expected", [1, (1, 1), (2, 3)])
def test_expected_input_size_is_enforced_and_saved_in_canonical_state(expected):
    engine, _, _, _ = make_engine(expected_input_size=expected)
    dimensions = (expected, expected) if type(expected) is int else expected
    assert engine.expected_input_size == dimensions
    engine.begin_epoch(1)
    engine.train_window(
        images_for().expand(4, 1, *dimensions).clone(), targets_for([0, 1, 3, 0])
    )
    assert engine.state_dict()["config"]["expected_input_size"] == list(dimensions)


@pytest.mark.parametrize("expected", [0, -1, True, (2,), [2, 3], (2, 0), (2, False), "640"])
def test_invalid_expected_input_size_is_rejected_at_construction(expected):
    with pytest.raises(AccumulationEngineError, match="expected_input_size"):
        make_engine(expected_input_size=expected)


def test_spatial_size_mismatch_fails_before_model_or_criterion_execution():
    engine, _, criterion, optimizer = make_engine(expected_input_size=640)
    engine.begin_epoch(1)
    with pytest.raises(AccumulationEngineError, match="spatial size"):
        engine.train_window(images_for(), targets_for([0, 1, 3, 0]))
    assert engine.failed and optimizer.steps == 0
    assert criterion.autocast_observations == []


@pytest.mark.parametrize("size", [(1, 1), [1], [1, 0], [True, 1], "1"])
def test_invalid_state_input_size_is_rejected_atomically(size):
    engine, _, _, _ = make_engine(expected_input_size=1)
    before = engine.state_dict()
    invalid = copy.deepcopy(before)
    invalid["config"]["expected_input_size"] = size
    with pytest.raises(AccumulationEngineError, match="expected_input_size"):
        engine.load_state_dict(invalid)
    assert engine.state_dict() == before


def test_state_cannot_be_restored_under_a_different_bound_input_size():
    engine, _, _, _ = make_engine(expected_input_size=640)
    other, _, _, _ = make_engine(expected_input_size=128)
    with pytest.raises(AccumulationEngineError, match="configuration"):
        other.load_state_dict(engine.state_dict())


@pytest.mark.parametrize("label,valid", [(0, True), (9, True), (10, False), (-1, False)])
def test_labels_must_fit_actual_decoder_class_count(label, valid):
    model = TinyDetector()
    model.decoder = nn.Identity()
    model.decoder.num_classes = 10
    engine, _, _, optimizer = make_engine(model=model)
    targets = targets_for([0, 1, 3, 0])
    targets[1]["labels"][0] = label
    engine.begin_epoch(1)
    if valid:
        engine.train_window(images_for(), targets)
        assert optimizer.steps == 1
    else:
        with pytest.raises(AccumulationEngineError, match="class indices"):
            engine.train_window(images_for(), targets)
        assert engine.failed and optimizer.steps == 0


@pytest.mark.parametrize("box", [
    [-0.1, 0.25, 0.25, 0.25],
    [0.25, 1.01, 0.25, 0.25],
    [0.25, 0.25, 0.0, 0.25],
    [0.25, 0.25, 0.25, -0.1],
    [0.25, 0.25, 1.01, 0.25],
    [float("nan"), 0.25, 0.25, 0.25],
    [0.25, 0.25, float("inf"), 0.25],
])
def test_boxes_must_be_finite_normalized_cxcywh_with_positive_extents(box):
    engine, _, criterion, optimizer = make_engine()
    targets = targets_for([0, 1, 3, 0])
    targets[1]["boxes"][0] = torch.tensor(box)
    engine.begin_epoch(1)
    with pytest.raises(AccumulationEngineError, match="boxes"):
        engine.train_window(images_for(), targets)
    assert engine.failed and optimizer.steps == 0
    assert criterion.autocast_observations == []


def test_public_state_validator_is_detached_and_does_not_mutate_input_or_engine():
    engine, _, _, _ = make_engine(expected_input_size=(2, 3))
    original = engine.state_dict()
    candidate = copy.deepcopy(original)
    detached = validate_engine_state_dict(candidate)
    assert detached == candidate == original
    detached["config"]["expected_input_size"][0] = 9
    assert candidate == original
    assert engine.state_dict() == original
    assert engine.validate_state_dict(candidate) == original
    assert engine.state_dict() == original


@pytest.mark.parametrize("updates,start,epoch", [(1, 1, 1), (1, 0, 2)])
def test_public_validator_enforces_completed_epoch_invariants(updates, start, epoch):
    engine, _, _, _ = make_engine()
    state = engine.state_dict()
    state.update(
        epoch=epoch,
        epoch_active=False,
        optimizer_updates=updates,
        epoch_start_optimizer_updates=start,
        microsteps=updates * engine.accumulation_steps,
    )
    with pytest.raises(AccumulationEngineError, match="epoch counters"):
        validate_engine_state_dict(state)
    with pytest.raises(AccumulationEngineError, match="epoch counters"):
        engine.validate_state_dict(state)


@pytest.mark.parametrize("field,value", [
    ("amp_dtype", "float16"), ("bn_statistics", "unknown"),
    ("clip_max_norm", float("nan")),
])
def test_detached_validator_rejects_invalid_engine_policy(field, value):
    engine, _, _, _ = make_engine()
    state = engine.state_dict()
    state["config"][field] = value
    with pytest.raises(AccumulationEngineError):
        validate_engine_state_dict(state)


class StridedBatchNormDetector(nn.Module):
    """A cat/view chain producing the actual singleton-N BN gradient layout."""

    def __init__(self):
        super().__init__()
        self.norm = nn.BatchNorm2d(8)
        self.incoming_gradient_strides = []
        self.output_dtypes = []
        self.forward_hook_counts = []
        # This observer is installed before the engine's temporary layout hook.
        self.norm.register_forward_hook(self._observe)

    def _observe(self, module, inputs, output):
        self.output_dtypes.append(output.dtype)
        self.forward_hook_counts.append(len(module._forward_hooks))
        if output.requires_grad:
            output.register_hook(
                lambda gradient: self.incoming_gradient_strides.append(gradient.stride())
            )

    def forward(self, images, targets=None):
        features = self.norm(images)
        return torch.cat((
            features.new_zeros((len(features), 3, 8)),
            features.flatten(2).transpose(1, 2),
            features.new_zeros((len(features), 5, 8)),
        ), dim=1)


def strided_bn_fixture(dtype=torch.float32, batch=1):
    # Per-channel mean=0 and population variance=1, exactly representable even
    # in BF16. This permits a strict analytic oracle instead of loose BF16 tolerances.
    images = (
        (torch.arange(64).remainder(2) * 2 - 1)
        .reshape(1, 1, 8, 8).expand(batch, 8, 8, 8)
        .to(dtype).contiguous().requires_grad_()
    )
    generator = torch.Generator().manual_seed(51)
    coefficients = (
        torch.randint(0, 2, (1, 72, 8), generator=generator) * 2 - 1
    ).to(dtype)
    return images, coefficients


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
@pytest.mark.parametrize("mode", ["train", "frozen"])
def test_cpu_bn_strided_singleton_backward_matches_analytic_derivative(dtype, mode):
    model = StridedBatchNormDetector()
    if dtype == torch.float64:
        model.double()
    images, coefficients = strided_bn_fixture(dtype)

    def criterion(outputs, targets):
        return {"linear_probe": (outputs * coefficients).sum()}

    engine, _, _, optimizer = make_engine(
        model=model, criterion=criterion, physical=1, accum=1,
        amp_dtype="bfloat16" if dtype == torch.bfloat16 else "float32",
        bn_statistics=mode,
    )
    before_hooks = tuple(model.norm._forward_hooks)
    engine.begin_epoch(1)
    report = engine.train_window(images, targets_for([1]))

    # This is the valid but degenerate layout that broke CPU BN in PyTorch 2.4.1.
    # We assert the corrected derivative, never that an unfixed future version fails.
    gradient = coefficients[:, 3:67].transpose(1, 2).reshape(1, 8, 8, 8).double()
    assert model.incoming_gradient_strides == [(8, 1, 64, 8)]
    inverse_std = (1 + model.norm.eps) ** -0.5
    normalized = images.detach().double() * inverse_std
    expected_bias = gradient.sum((0, 2, 3))
    expected_weight = (gradient * normalized).sum((0, 2, 3))
    if mode == "frozen":
        expected_input = gradient * inverse_std
    else:
        expected_input = inverse_std * (
            gradient - expected_bias.reshape(1, -1, 1, 1) / 64
            - normalized * expected_weight.reshape(1, -1, 1, 1) / 64
        )

    tolerance = dict(rtol=1e-12, atol=1e-12) if dtype == torch.float64 else dict(rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(model.norm.bias.grad.double(), expected_bias, **tolerance)
    torch.testing.assert_close(model.norm.weight.grad.double(), expected_weight, **tolerance)
    torch.testing.assert_close(images.grad.double(), expected_input.to(dtype).double(), **tolerance)
    assert model.output_dtypes == [dtype]
    assert model.norm.num_batches_tracked.item() == int(mode == "train")
    assert tuple(model.norm._forward_hooks) == before_hooks
    assert optimizer.steps == engine.optimizer_updates == engine.microsteps == 1
    assert report["phase"] == 0


@pytest.mark.parametrize("fail", [False, True])
def test_cpu_bn_layout_hooks_do_not_accumulate_and_are_removed_after_failure(fail):
    model = StridedBatchNormDetector()
    images, coefficients = strided_bn_fixture(batch=2)
    calls = 0

    def criterion(outputs, targets):
        nonlocal calls
        calls += 1
        loss = (outputs * coefficients).sum()
        return {"linear_probe": loss * float("nan") if fail and calls == 2 else loss}

    before_hooks = tuple(model.norm._forward_hooks)
    engine, _, _, optimizer = make_engine(
        model=model, criterion=criterion, physical=1, accum=2, bn_statistics="frozen",
    )
    assert tuple(model.norm._forward_hooks) == before_hooks
    engine.begin_epoch(1)
    if fail:
        with pytest.raises(AccumulationEngineError, match="nonfinite"):
            engine.train_window(images, targets_for([1, 1]))
        assert engine.failed and optimizer.steps == 0
    else:
        for _ in range(2):
            engine.train_window(images, targets_for([1, 1]))
            assert tuple(model.norm._forward_hooks) == before_hooks
        assert optimizer.steps == engine.optimizer_updates == 2
        assert engine.microsteps == 4
    assert tuple(model.norm._forward_hooks) == before_hooks
    # One pre-existing observer plus exactly one temporary normalization hook.
    assert model.forward_hook_counts == [2] * calls


def test_cuda_layout_policy_installs_no_hooks_and_does_not_initialize_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("no CUDA runtime use is permitted")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    model = StridedBatchNormDetector()
    before_hooks = tuple(model.norm._forward_hooks)
    # Only a device descriptor is constructed; model and tensors remain on CPU.
    with _batchnorm_backward_layout(model, torch.device("cuda")):
        assert tuple(model.norm._forward_hooks) == before_hooks
    assert tuple(model.norm._forward_hooks) == before_hooks


@pytest.mark.parametrize("invalid", [None, "native", "cpu_contiguous", True])
def test_bn_backward_layout_is_fixed_and_state_policy_cannot_be_omitted_or_changed(invalid):
    engine, _, _, _ = make_engine()
    original = engine.state_dict()
    assert original["config"]["bn_backward_layout"] == BN_BACKWARD_LAYOUT == "cpu_contiguous_cuda_native"
    changed = copy.deepcopy(original)
    if invalid is None:
        del changed["config"]["bn_backward_layout"]
    else:
        changed["config"]["bn_backward_layout"] = invalid
    with pytest.raises(AccumulationEngineError, match="config|bn_backward_layout"):
        validate_engine_state_dict(changed)
    with pytest.raises(AccumulationEngineError, match="config|bn_backward_layout"):
        engine.load_state_dict(changed)
    assert engine.state_dict() == original
