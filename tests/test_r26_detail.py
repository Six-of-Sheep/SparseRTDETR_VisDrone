"""R26 synthetic arithmetic (CPU; no data, no checkpoint, no CUDA)."""
import copy
import importlib
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sparse_rtdetr.research import r26_detail as R
from sparse_rtdetr.baseline import training_v2b as vendor


def csp(d=8, cin=16, cout=16, dtype=torch.float64, seed=1):
    with vendor._vendor_path(vendor._vendor_root(ROOT)):
        H = importlib.import_module("src.zoo.rtdetr.hybrid_encoder")
    torch.manual_seed(seed)
    b = H.CSPRepLayer(cin, cout, num_blocks=3, expansion=d / cout, act="silu").to(dtype).eval()
    return b


def randomize_T(b, seed=7):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for m in b.r26_detail:
            if isinstance(m, torch.nn.Conv2d):
                m.weight.copy_(torch.randn(m.weight.shape, generator=g, dtype=m.weight.dtype) * 0.3)


def pair(seed=1, d=8):
    a, b = csp(d, seed=seed), csp(d, seed=seed)
    R.install_block(a, "U")
    R.install_block(b, "P", copy.deepcopy(a.r26_detail))
    return a, b


def test_grid_operators_exact():
    x = torch.randn(2, 3, 6, 10, dtype=torch.float64)
    c = torch.randn(2, 3, 3, 5, dtype=torch.float64)
    assert torch.equal(R.down(R.up(c)), c)                                  # DU = I
    r = x - R.up(R.down(x))
    assert R.down(r).abs().max() < 1e-15                                    # D r = 0
    p = R.project(x)
    assert (R.project(p) - p).abs().max() < 1e-15                           # idempotent
    y = torch.randn_like(x)
    assert (R.project(2 * x - 3 * y) - (2 * R.project(x) - 3 * R.project(y))).abs().max() < 1e-13   # linear
    const = torch.full_like(x, 3.25)
    assert R.project(const).abs().max() == 0                                # constants removed
    point = torch.zeros(1, 1, 4, 4, dtype=torch.float64); point[0, 0, 1, 2] = 1.
    pp = R.project(point)
    assert pp[0, 0, 1, 2] == 0.75 and pp[0, 0, 0, 2] == -0.25 and pp.sum().abs() < 1e-15
    checker = (torch.arange(4)[:, None] + torch.arange(6)[None]).remainder(2).double()[None, None] * 2 - 1
    assert torch.equal(R.project(checker), checker)                         # zero cell mean kept


def test_odd_grid_refused():
    with pytest.raises(ValueError):
        R.down(torch.zeros(1, 1, 5, 4))
    b = csp()
    R.install_block(b, "P")
    with pytest.raises(ValueError):
        b(torch.zeros(1, 16, 6, 7, dtype=torch.float64))


def test_parameter_count_and_init():
    assert R.detail_parameters(128) == 12576
    state = torch.get_rng_state()
    t1, t2 = R.make_detail(128), R.make_detail(128)
    assert torch.equal(torch.get_rng_state(), state)                        # no global RNG consumed
    assert R.tensor_sha(t1) == R.tensor_sha(t2)
    assert sum(p.numel() for p in t1.parameters()) == 12576
    assert all(m.bias is None for m in t1 if isinstance(m, torch.nn.Conv2d))
    assert torch.count_nonzero(t1[4].weight) == 0 and torch.count_nonzero(t1[0].weight) > 0


def test_native_mode_is_bitwise_native():
    a, b = csp(), csp()
    R.install_block(b, "N")
    x = torch.randn(2, 16, 8, 12, dtype=torch.float64)
    assert torch.equal(a(x), b(x))


def test_zero_step_U_equals_P_and_differs_from_N():
    u, p = pair()
    n = csp()
    x = torch.randn(2, 16, 8, 12, dtype=torch.float64)
    assert torch.equal(u(x), p(x))
    assert not torch.allclose(u(x), n(x))                                   # B(D v) != D(B v)


@pytest.mark.parametrize("mode", ["U", "P"])
def test_constraint_with_nonzero_t(mode):
    u, p = pair()
    randomize_T(u); randomize_T(p, seed=7)
    blk = u if mode == "U" else p
    stats = []
    blk.r26_monitor = stats
    blk(torch.randn(2, 16, 8, 12, dtype=torch.float64))
    s = stats[0]
    assert s["norm_t"] > 1e-3
    if mode == "P":
        assert s["max_abs_D_z_minus_c_fp32"] < 1e-12
    else:
        assert s["max_abs_D_z_minus_c_fp32"] > 1e-4
    assert s["max_abs_r_cell_mean"] < 1e-12


@pytest.mark.parametrize("mode", ["U", "P"])
def test_gradcheck(mode):
    b = csp(d=4, cin=8, cout=8, seed=3)
    R.install_block(b, mode)
    randomize_T(b, seed=5)
    x = torch.randn(1, 8, 4, 6, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda v: b(v), (x,), eps=1e-6, atol=1e-6)
    w = b.r26_detail[4].weight
    assert torch.autograd.gradcheck(lambda wt: torch.func.functional_call(b, {"r26_detail.4.weight": wt}, (x.detach(),)),
                                    (w.detach().clone().requires_grad_(True),), eps=1e-6, atol=1e-6)


def test_P_gradient_is_projected():
    """d out/d t passes through (I-UD): a loss depending on cell means of z gives zero T gradient in P."""
    u, p = pair()
    randomize_T(u); randomize_T(p)
    x = torch.randn(1, 16, 8, 12, dtype=torch.float64)
    for blk, zero in ((u, False), (p, True)):
        captured = {}
        blk.conv3.register_forward_pre_hook(lambda m, a: captured.__setitem__("in", a[0]))
        blk(x)
        loss = R.down(captured["in"]).sum()                                  # depends only on cell means of z+s
        g = torch.autograd.grad(loss, blk.r26_detail[4].weight, allow_unused=True)[0]
        assert (g.abs().max() < 1e-12) == zero


def test_bf16_autocast_records_cast_error():
    b = csp(d=8, dtype=torch.float32)
    R.install_block(b, "P")
    randomize_T(b)
    stats = []
    b.r26_monitor = stats
    with torch.autocast("cpu", dtype=torch.bfloat16):
        b(torch.randn(1, 16, 8, 12))
    s = stats[0]
    assert s["max_abs_D_z_minus_c_fp32"] < 1e-5            # FP32 reconstruction keeps the constraint
    assert s["z_cast_dtype"] == "torch.bfloat16"            # z follows c's dtype before conv3
    assert s["max_abs_cast_error_z"] >= 0


def test_deepcopy_keeps_mode():
    u, p = pair()
    q = copy.deepcopy(p)
    assert q.r26_mode == "P" and R.tensor_sha(q.r26_detail) == R.tensor_sha(p.r26_detail)
    with pytest.raises(ValueError):
        R.install_block(q, "U")
