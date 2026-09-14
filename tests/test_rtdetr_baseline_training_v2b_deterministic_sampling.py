"""CPU numerical/derivative oracles for the unconnected sampling candidate."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from sparse_rtdetr.baseline.training_v2b_deterministic_sampling import (
    DeterministicSamplingError, _sampling_dtype, bilinear_grid_sample,
    deformable_attention_core_func_v2,
)


@pytest.fixture(autouse=True)
def forbid_cuda_calls(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("sampling CPU validation attempted CUDA access")
    for name in ("init", "_lazy_init", "is_available", "is_initialized", "device_count", "get_device_properties"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    # PyTorch 2.4's CPU autocast constructor evaluates this CUDA availability
    # predicate before checking the requested device. Stub that guard so even
    # availability observation is excluded from these CPU tensor tests.
    monkeypatch.setattr(torch.cuda.amp.common, "amp_definitely_not_available", lambda: True)


def _inputs(shape, out_shape, *, dtype=torch.float64, layout="contiguous"):
    generator = torch.Generator(device="cpu").manual_seed(913)
    n, channels, height, width = shape
    out_height, out_width = out_shape
    if layout == "strided":
        value = torch.randn(n, channels, width, height, generator=generator, dtype=dtype).transpose(2, 3)
        grid = (torch.rand(n, out_width, out_height, 2, generator=generator, dtype=dtype) * 3.2 - 1.6).transpose(1, 2)
        gradient = torch.randn(n, channels, out_width, out_height, generator=generator, dtype=dtype).transpose(2, 3)
    elif layout == "channels_last":
        value = torch.randn(shape, generator=generator, dtype=dtype).contiguous(memory_format=torch.channels_last)
        grid = torch.rand(n, out_height, out_width, 2, generator=generator, dtype=dtype) * 3.2 - 1.6
        gradient = torch.randn(n, channels, out_height, out_width, generator=generator, dtype=dtype)
    else:
        value = torch.randn(shape, generator=generator, dtype=dtype)
        grid = torch.rand(n, out_height, out_width, 2, generator=generator, dtype=dtype) * 3.2 - 1.6
        gradient = torch.randn(n, channels, out_height, out_width, generator=generator, dtype=dtype)
    return value, grid, gradient


def _run(function, value, grid, gradient, *, autocast=False):
    value = value.detach().clone(memory_format=torch.preserve_format).requires_grad_(True)
    grid = grid.detach().clone(memory_format=torch.preserve_format).requires_grad_(True)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        output = function(value, grid)
    grad_value, grad_grid = torch.autograd.grad(output, (value, grid), gradient.to(output.dtype))
    return output.detach(), grad_value.detach(), grad_grid.detach()


def _native(value, grid):
    return F.grid_sample(value, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def _metrics(left, right, names=("forward", "grad_value", "grad_grid")):
    result = {}
    for name, native, candidate in zip(names, left, right):
        delta = candidate.double() - native.double()
        scale = max(float(torch.linalg.vector_norm(native.double())), 1e-30)
        result[name] = {"dtype": str(native.dtype), "exact": torch.equal(native, candidate),
                        "max_absolute": float(delta.abs().max()),
                        "relative_L2": float(torch.linalg.vector_norm(delta)) / scale}
    return result


def _assert_numerical(native, candidate):
    # Operator implementation comparison only. These thresholds do not change
    # checkpoint, BN, RNG, clock or campaign acceptance requirements.
    tolerance = {torch.float64: (2e-12, 2e-12), torch.float32: (2e-5, 2e-6),
                 torch.bfloat16: (2e-2, 2e-5), torch.float16: (2e-3, 2e-5)}
    for expected, actual in zip(native, candidate):
        assert expected.dtype == actual.dtype and expected.shape == actual.shape
        assert torch.isfinite(actual).all()
        relative, absolute = tolerance[actual.dtype]
        torch.testing.assert_close(actual, expected, rtol=relative, atol=absolute)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape,out_shape", [((1, 1, 1, 1), (3, 2)), ((1, 2, 1, 7), (2, 5)),
    ((2, 3, 5, 1), (4, 3)), ((2, 3, 4, 7), (5, 4))])
@pytest.mark.parametrize("layout", ["contiguous", "strided", "channels_last"])
def test_forward_and_both_gradients_match_native_cpu(dtype, shape, out_shape, layout, record_property):
    value, grid, gradient = _inputs(shape, out_shape, dtype=dtype, layout=layout)
    native = _run(_native, value, grid, gradient)
    candidate = _run(bilinear_grid_sample, value, grid, gradient)
    record_property("sampling_difference", json.dumps(_metrics(native, candidate), sort_keys=True))
    _assert_numerical(native, candidate)


@pytest.mark.parametrize("value_dtype,grid_dtype", [(torch.float32, torch.float32), (torch.float64, torch.float64),
    (torch.bfloat16, torch.bfloat16), (torch.bfloat16, torch.float32), (torch.float32, torch.bfloat16),
    (torch.float16, torch.float16), (torch.float16, torch.float32), (torch.float32, torch.float16)])
def test_cpu_autocast_preserves_native_operator_and_leaf_gradient_dtypes(value_dtype, grid_dtype, record_property):
    value, grid, gradient = _inputs((2, 3, 4, 7), (5, 4))
    value, grid = value.to(value_dtype), grid.to(grid_dtype)
    native = _run(_native, value, grid, gradient, autocast=True)
    candidate = _run(bilinear_grid_sample, value, grid, gradient, autocast=True)
    assert candidate[0].dtype == (torch.float64 if value_dtype == torch.float64 else torch.float32)
    assert candidate[1].dtype == value_dtype and candidate[2].dtype == grid_dtype
    record_property("sampling_difference", json.dumps(_metrics(native, candidate), sort_keys=True))
    _assert_numerical(native, candidate)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_zero_padding_boundaries_and_pixel_centers_match_native_derivatives(dtype, record_property):
    value = torch.arange(1, 25, dtype=dtype).reshape(1, 1, 4, 6)
    points = [[-1., -1.], [1., 1.], [-1., 1.], [1., -1.], [0., 0.],
              [-1.5, -.2], [1.5, .2], [.1, -1.5], [.1, 1.5],
              [(2 * 2 + 1) / 6 - 1, (2 * 1 + 1) / 4 - 1],
              [-1. + 1e-5, .1], [-1. - 1e-5, .1], [1. + 1e-5, -.1], [1. - 1e-5, -.1]]
    grid = torch.tensor(points, dtype=dtype).reshape(1, 1, -1, 2)
    gradient = torch.linspace(-.7, .9, len(points), dtype=dtype).reshape(1, 1, 1, -1)
    native = _run(_native, value, grid, gradient)
    candidate = _run(bilinear_grid_sample, value, grid, gradient)
    record_property("sampling_difference", json.dumps(_metrics(native, candidate), sort_keys=True))
    _assert_numerical(native, candidate)


def test_corner_weights_and_grid_derivatives_have_independent_analytic_values():
    value = torch.full((1, 1, 4, 6), 4., dtype=torch.float64)
    grid = torch.tensor([[[[-1., -1.]]]], dtype=torch.float64)
    output, grad_value, grad_grid = _run(bilinear_grid_sample, value, grid, torch.ones(1, 1, 1, 1))
    torch.testing.assert_close(output, torch.ones_like(output), rtol=0, atol=0)
    expected = torch.zeros_like(value)
    expected[0, 0, 0, 0] = .25
    torch.testing.assert_close(grad_value, expected, rtol=0, atol=0)
    torch.testing.assert_close(grad_grid, torch.tensor([[[[6., 4.]]]], dtype=torch.float64), rtol=0, atol=0)


def test_interior_linear_ramp_has_analytic_spatial_gradient():
    value = (torch.arange(3, dtype=torch.float64)[:, None] * 10 + torch.arange(4, dtype=torch.float64)).reshape(1, 1, 3, 4)
    grid = torch.tensor([[[[.2, .1]]]], dtype=torch.float64)
    output, grad_value, grad_grid = _run(bilinear_grid_sample, value, grid, torch.ones(1, 1, 1, 1))
    torch.testing.assert_close(output, torch.tensor([[[[13.4]]]], dtype=torch.float64), rtol=0, atol=1e-14)
    torch.testing.assert_close(grad_grid, torch.tensor([[[[2., 15.]]]], dtype=torch.float64), rtol=0, atol=1e-14)
    torch.testing.assert_close(grad_value.sum(), torch.tensor(1., dtype=torch.float64), rtol=0, atol=1e-14)


def test_far_outside_grid_has_exactly_zero_output_and_both_gradients():
    value = torch.ones((1, 2, 4, 6), dtype=torch.float64)
    grid = torch.tensor([[[[-9., -9.], [9., 9.], [-9., 0.], [0., 9.]]]], dtype=torch.float64)
    result = _run(bilinear_grid_sample, value, grid, torch.ones(1, 2, 1, 4))
    assert all(torch.count_nonzero(tensor) == 0 for tensor in result)


@pytest.mark.parametrize("dtype,magnitude", [(torch.float32, 1e38), (torch.float64, 1e307)])
def test_large_finite_coordinates_match_native_finite_zero_output_and_gradients(dtype, magnitude, record_property):
    # FP32 1e38 used to overflow the candidate's coordinate multiplication,
    # despite native CPU returning finite zeros for all three results.
    points = [[magnitude, 0.], [-magnitude, 0.], [0., magnitude], [0., -magnitude],
              [magnitude, magnitude], [magnitude, -magnitude],
              [-magnitude, magnitude], [-magnitude, -magnitude]]
    value = torch.ones((1, 2, 4, 6), dtype=dtype)
    grid = torch.tensor(points, dtype=dtype).reshape(1, 1, -1, 2)
    gradient = torch.ones(1, 2, 1, len(points), dtype=dtype)
    native = _run(_native, value, grid, gradient)
    candidate = _run(bilinear_grid_sample, value, grid, gradient)
    for expected, actual in zip(native, candidate):
        assert torch.isfinite(expected).all() and torch.isfinite(actual).all()
        assert torch.count_nonzero(expected) == 0
        assert torch.equal(actual, expected)
    record_property("sampling_difference", json.dumps(_metrics(native, candidate), sort_keys=True))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("height,width", [(1, 1), (1, 6), (4, 1), (4, 6)])
def test_maximum_finite_coordinates_have_analytic_zero_support(dtype, height, width):
    # Native CPU itself can return NaN at finfo.max. This is an independent
    # mathematical zeros-padding oracle, not a claim of native numerical parity.
    magnitude = torch.finfo(dtype).max
    points = [[magnitude, 0.], [-magnitude, 0.], [0., magnitude], [0., -magnitude],
              [magnitude, magnitude], [magnitude, -magnitude],
              [-magnitude, magnitude], [-magnitude, -magnitude]]
    value = torch.linspace(-2., 3., 2 * height * width, dtype=dtype).reshape(1, 2, height, width)
    grid = torch.tensor(points, dtype=dtype).reshape(1, 1, -1, 2)
    gradient = torch.linspace(-.7, .9, 2 * len(points), dtype=dtype).reshape(1, 2, 1, -1)
    for tensor in _run(bilinear_grid_sample, value, grid, gradient):
        assert torch.isfinite(tensor).all()
        assert torch.count_nonzero(tensor) == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("height,width", [(1, 1), (1, 4), (4, 1), (4, 8)])
def test_far_guard_transition_has_zero_native_output_and_derivatives(dtype, height, width):
    points = []
    for edge in (-4., 4.):
        center = torch.tensor(edge, dtype=dtype)
        for direction in (-torch.inf, torch.inf):
            neighbor = torch.nextafter(center, torch.tensor(direction, dtype=dtype)).item()
            points.extend([[neighbor, 0.], [0., neighbor], [neighbor, -neighbor]])
        points.extend([[edge, 0.], [0., edge], [edge, -edge]])
    value = torch.ones((1, 2, height, width), dtype=dtype)
    grid = torch.tensor(points, dtype=dtype).reshape(1, 1, -1, 2)
    gradient = torch.ones(1, 2, 1, len(points), dtype=dtype)
    native = _run(_native, value, grid, gradient)
    candidate = _run(bilinear_grid_sample, value, grid, gradient)
    for expected, actual in zip(native, candidate):
        assert torch.isfinite(actual).all()
        assert torch.count_nonzero(expected) == 0
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("height,width", [(1, 1), (1, 4), (4, 1), (4, 8)])
def test_true_support_boundaries_retain_native_one_sided_derivatives(dtype, height, width, record_property):
    # Power-of-two dimensions make the exact support endpoints representable.
    # The lower boundary's zero output still has a nonzero inward derivative;
    # masking at normalized +/-1 or at the support endpoint would destroy it.
    points, exact_boundaries = [], []
    for axis, size in enumerate((width, height)):
        for sign in (-1, 1):
            edge = torch.tensor(sign * (1 + 1 / size), dtype=dtype)
            candidates = (torch.nextafter(edge, torch.tensor(-torch.inf, dtype=dtype)),
                          edge, torch.nextafter(edge, torch.tensor(torch.inf, dtype=dtype)))
            for offset, point in enumerate(candidates):
                coordinates = [0., 0.]
                coordinates[axis] = point.item()
                points.append(coordinates)
                if offset == 1:
                    exact_boundaries.append((len(points) - 1, axis, size if sign == -1 else 0))
    value = torch.full((1, 1, height, width), 2., dtype=dtype)
    grid = torch.tensor(points, dtype=dtype).reshape(1, 1, -1, 2)
    gradient = torch.ones(1, 1, 1, len(points), dtype=dtype)
    native = _run(_native, value, grid, gradient)
    candidate = _run(bilinear_grid_sample, value, grid, gradient)
    _assert_numerical(native, candidate)
    for index, axis, derivative in exact_boundaries:
        assert candidate[0][0, 0, 0, index] == 0
        expected = torch.zeros(2, dtype=dtype)
        expected[axis] = derivative
        assert torch.equal(candidate[2][0, 0, index], expected)
        assert torch.equal(native[2][0, 0, index], expected)
    record_property("sampling_difference", json.dumps(_metrics(native, candidate), sort_keys=True))


def test_repeated_sampling_locations_accumulate_all_contributions():
    value = torch.arange(16, dtype=torch.float64).reshape(1, 1, 4, 4)
    grid = torch.zeros(1, 3, 7, 2, dtype=torch.float64)
    output, grad_value, grad_grid = _run(bilinear_grid_sample, value, grid, torch.ones(1, 1, 3, 7))
    assert torch.equal(output, torch.full_like(output, 7.5))
    expected = torch.zeros_like(value)
    expected[0, 0, 1:3, 1:3] = 21 / 4
    assert torch.equal(grad_value, expected)
    assert torch.equal(grad_grid[..., 0], torch.full((1, 3, 7), 2., dtype=torch.float64))
    assert torch.equal(grad_grid[..., 1], torch.full((1, 3, 7), 8., dtype=torch.float64))


def test_double_precision_gradcheck_away_from_piecewise_boundaries():
    value, grid, _ = _inputs((1, 2, 3, 4), (2, 3))
    value.requires_grad_(True)
    # Chosen interior/noninteger source coordinates avoid derivative kinks.
    grid = torch.tensor([[[[-.63, -.31], [.11, .19], [.44, .58]],
                          [[-.37, .43], [.62, -.47], [.23, -.13]]]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(bilinear_grid_sample, (value, grid), eps=1e-6, atol=2e-6, rtol=1e-4)


def test_cpu_repetition_is_exact_and_does_not_change_backend_or_autocast_flags():
    value, grid, gradient = _inputs((2, 3, 4, 7), (5, 4), dtype=torch.float32, layout="strided")
    before = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
              torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark, torch.is_autocast_enabled("cpu"))
    first = _run(bilinear_grid_sample, value, grid, gradient)
    for _ in range(3):
        assert all(torch.equal(left, right) for left, right in zip(first, _run(bilinear_grid_sample, value, grid, gradient)))
    after = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
             torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark, torch.is_autocast_enabled("cpu"))
    assert before == after


@pytest.mark.parametrize("value_dtype,grid_dtype,enabled,low,expected", [
    (torch.bfloat16, torch.float32, True, torch.bfloat16, torch.float32),
    (torch.float32, torch.bfloat16, True, torch.bfloat16, torch.float32),
    (torch.float16, torch.float32, True, torch.float16, torch.float32),
    (torch.bfloat16, torch.bfloat16, True, torch.bfloat16, torch.bfloat16),
    (torch.float32, torch.float32, False, None, torch.float32),
    (torch.float64, torch.float64, True, torch.bfloat16, torch.float64),
])
def test_cuda_promotion_policy_is_inspectable_without_cuda(value_dtype, grid_dtype, enabled, low, expected):
    assert _sampling_dtype(value_dtype, grid_dtype, device_type="cuda", autocast_enabled=enabled,
                           autocast_dtype=low) == expected


@pytest.mark.parametrize("value_dtype,grid_dtype,enabled,low", [
    (torch.bfloat16, torch.float32, False, None), (torch.float64, torch.float32, True, torch.bfloat16),
    (torch.float16, torch.bfloat16, True, torch.bfloat16), (torch.int64, torch.float32, False, None),
])
def test_native_dtype_scope_is_not_silently_broadened(value_dtype, grid_dtype, enabled, low):
    with pytest.raises(DeterministicSamplingError):
        _sampling_dtype(value_dtype, grid_dtype, device_type="cuda", autocast_enabled=enabled, autocast_dtype=low)


@pytest.mark.parametrize("change", ["rank", "grid_last", "batch", "integer", "half_without_autocast", "mixed_double"])
def test_invalid_or_unvalidated_operator_scope_fails_explicitly(change):
    value, grid, _ = _inputs((1, 2, 3, 4), (2, 3), dtype=torch.float32)
    if change == "rank": value = value[0]
    elif change == "grid_last": grid = grid[..., :1]
    elif change == "batch": grid = grid.expand(2, -1, -1, -1)
    elif change == "integer": value = value.to(torch.int64)
    elif change == "half_without_autocast": value, grid = value.half(), grid.half()
    else: value = value.double()
    with pytest.raises(DeterministicSamplingError):
        bilinear_grid_sample(value, grid)


@pytest.fixture
def native_attention_core():
    # Load the actual vendored utility directly: no model construction,
    # package-level registrations, backend wiring, data or CUDA access.
    path = Path(__file__).resolve().parents[1] / "vendor/rtdetrv2_pytorch/src/zoo/rtdetr/utils.py"
    spec = importlib.util.spec_from_file_location("_sampling_native_attention_oracle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.deformable_attention_core_func_v2


def _attention_inputs(spatial_shapes, points, *, dtype=torch.float64, layout="contiguous"):
    generator = torch.Generator(device="cpu").manual_seed(914)
    batch, heads, channels, queries = 2, 2, 3, 4
    length, count = sum(h * w for h, w in spatial_shapes), sum(points)
    if layout == "strided":
        value = torch.randn(batch, channels, heads, length, generator=generator, dtype=dtype).permute(0, 3, 2, 1)
        locations = (torch.rand(batch, count, heads, queries, 2, generator=generator, dtype=dtype) * 1.4 - .2).permute(0, 3, 2, 1, 4)
        weights = torch.randn(batch, count, heads, queries, generator=generator, dtype=dtype).softmax(dim=1).permute(0, 3, 2, 1)
        gradient = torch.randn(batch, heads * channels, queries, generator=generator, dtype=dtype).permute(0, 2, 1)
    else:
        value = torch.randn(batch, length, heads, channels, generator=generator, dtype=dtype)
        locations = torch.rand(batch, queries, heads, count, 2, generator=generator, dtype=dtype) * 1.4 - .2
        weights = torch.randn(batch, queries, heads, count, generator=generator, dtype=dtype).softmax(dim=-1)
        gradient = torch.randn(batch, queries, heads * channels, generator=generator, dtype=dtype)
    return value, locations, weights, gradient


def _run_attention(function, value, locations, weights, gradient, spatial_shapes, points, *, autocast=False):
    inputs = tuple(t.detach().clone(memory_format=torch.preserve_format).requires_grad_(True)
                   for t in (value, locations, weights))
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        output = function(inputs[0], spatial_shapes, inputs[1], inputs[2], points, method="default")
    gradients = torch.autograd.grad(output, inputs, gradient.to(output.dtype))
    return (output.detach(), *(t.detach() for t in gradients))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("spatial_shapes,points", [([(1, 1)], [3]), ([(3, 4), (2, 1), (1, 1)], [2, 3, 1])])
@pytest.mark.parametrize("layout", ["contiguous", "strided"])
@pytest.mark.parametrize("shape_container", ["list", "tensor"])
def test_attention_core_matches_vendor_output_and_all_three_input_gradients(
        dtype, spatial_shapes, points, layout, shape_container, native_attention_core, record_property):
    value, locations, weights, gradient = _attention_inputs(spatial_shapes, points, dtype=dtype, layout=layout)
    shapes = torch.tensor(spatial_shapes, dtype=torch.int64) if shape_container == "tensor" else spatial_shapes
    native = _run_attention(native_attention_core, value, locations, weights, gradient, shapes, points)
    candidate = _run_attention(deformable_attention_core_func_v2, value, locations, weights, gradient, shapes, points)
    assert candidate[0].shape == (value.shape[0], locations.shape[1], value.shape[2] * value.shape[3])
    _assert_numerical(native, candidate)
    record_property("sampling_difference", json.dumps(_metrics(
        native, candidate, ("forward", "grad_value", "grad_locations", "grad_attention")), sort_keys=True))


@pytest.mark.parametrize("value_dtype,location_dtype,attention_dtype", [
    (torch.float32, torch.float32, torch.float32),
    (torch.bfloat16, torch.float32, torch.float32),
    (torch.bfloat16, torch.float32, torch.bfloat16),
    (torch.bfloat16, torch.bfloat16, torch.bfloat16),
    (torch.float32, torch.bfloat16, torch.float32),
    (torch.float16, torch.float32, torch.float16),
    (torch.float16, torch.float16, torch.float16),
])
def test_attention_core_cpu_autocast_matches_vendor_dtypes_and_all_gradients(
        value_dtype, location_dtype, attention_dtype, native_attention_core, record_property):
    shapes, points = [(3, 4), (2, 1), (1, 1)], [2, 3, 1]
    value, locations, weights, gradient = _attention_inputs(shapes, points, layout="strided")
    value, locations, weights = value.to(value_dtype), locations.to(location_dtype), weights.to(attention_dtype)
    native = _run_attention(native_attention_core, value, locations, weights, gradient, shapes, points, autocast=True)
    candidate = _run_attention(deformable_attention_core_func_v2, value, locations, weights, gradient, shapes, points, autocast=True)
    assert tuple(t.dtype for t in candidate) == (torch.float32, value_dtype, location_dtype, attention_dtype)
    _assert_numerical(native, candidate)
    record_property("sampling_difference", json.dumps(_metrics(
        native, candidate, ("forward", "grad_value", "grad_locations", "grad_attention")), sort_keys=True))


def test_attention_core_keeps_three_analytic_input_derivatives():
    value = torch.arange(4, dtype=torch.float64).reshape(1, 4, 1, 1)
    locations = torch.full((1, 1, 1, 1, 2), .5, dtype=torch.float64)
    weights = torch.full((1, 1, 1, 1), .4, dtype=torch.float64)
    result = _run_attention(deformable_attention_core_func_v2, value, locations, weights,
                            torch.ones(1, 1, 1, dtype=torch.float64), [(2, 2)], [1])
    expected = (torch.full((1, 1, 1), .6, dtype=torch.float64), torch.full_like(value, .1),
                torch.tensor([[[[[.8, 1.6]]]]], dtype=torch.float64), torch.full_like(weights, 1.5))
    for actual, reference in zip(result, expected):
        torch.testing.assert_close(actual, reference, rtol=0, atol=1e-15)


@pytest.mark.parametrize("method", ["discrete", "unknown", None])
def test_attention_core_rejects_nondefault_sampling(method):
    shapes, points = [(2, 2)], [1]
    value, locations, weights, _ = _attention_inputs(shapes, points)
    with pytest.raises(DeterministicSamplingError, match="method='default'"):
        deformable_attention_core_func_v2(value, shapes, locations, weights, points, method=method)
