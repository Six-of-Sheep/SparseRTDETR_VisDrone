"""R26: coarse-grid context with a constrained detail update in the P3 top-down CSPRep block.

TASK_R26_001.  Only ``encoder.fpn_blocks[1]`` (the top-down CSPRepLayer that produces P3) changes:

    v = conv1(X); s = conv2(X)
    D = non-overlapping 2x2 mean (stride 2, no padding); U = 2x2 nearest repeat
    c = B(D(v))                      # original bottlenecks, run on the coarse grid
    r = v - U(D(v))
    t = T(concat(r, U(c)))           # T: 1x1(2d->d/4) SiLU DW3x3 SiLU 1x1(d/4->d), no bias, no BN
    z_U = U(c) + r + t
    z_P = U(c) + r + t - U(D(t))     # only difference between the two learnable arms
    out = conv3(z + s)

N is the native dense block (no change).  D, r, the projection and the reconstruction are FP32; B and T
follow the active autocast.  z is cast to the dtype of c (the dtype native B(v) would have) before conv3.
Odd grids are refused (no implicit padding).  T is initialised from its own CPU generator (seed0, Kaiming
normal fan_in/relu for the first two convs, zeros for the last), never from the training RNG.
"""
from __future__ import annotations

import copy
import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F

VERSION = "R26-constrained-detail-v1"
MODES = ("N", "U", "P")
BLOCK_PATH = "encoder.fpn_blocks.1"
DETAIL_NAME = "r26_detail"
INIT_SEED = 0
EXPECTED = {"in_channels": 512, "hidden": 128, "out_channels": 256, "num_blocks": 3}


def require(ok, message):
    if not ok:
        raise ValueError(message)


# ---------------------------------------------------------------- grid operators (exact, deterministic)
def down(x):
    """D: non-overlapping 2x2 average; even H and W only."""
    require(x.dim() == 4 and x.shape[-2] % 2 == 0 and x.shape[-1] % 2 == 0, f"R26 needs an even grid, got {tuple(x.shape)}")
    return F.avg_pool2d(x, 2, 2)


def up(x):
    """U: 2x2 nearest repeat via expand/reshape (backward is a plain sum, deterministic on CUDA)."""
    n, c, h, w = x.shape
    return x[:, :, :, None, :, None].expand(n, c, h, 2, w, 2).reshape(n, c, 2 * h, 2 * w)


def project(t):
    """(I - U D) t: remove each 2x2 cell mean."""
    return t - up(down(t))


# ---------------------------------------------------------------- detail update T
def make_detail(d, generator=None):
    q = d // 4
    require(d % 4 == 0, "R26 detail width d must be divisible by 4")
    with torch.random.fork_rng(devices=[]):      # module construction must not consume the training RNG
        t = nn.Sequential(nn.Conv2d(2 * d, q, 1, bias=False), nn.SiLU(),
                          nn.Conv2d(q, q, 3, padding=1, groups=q, bias=False), nn.SiLU(),
                          nn.Conv2d(q, d, 1, bias=False))
    if generator is None:
        generator = torch.Generator(device="cpu").manual_seed(INIT_SEED)
    with torch.no_grad():
        for i in (0, 2):
            nn.init.kaiming_normal_(t[i].weight, mode="fan_in", nonlinearity="relu", generator=generator)
        nn.init.zeros_(t[4].weight)
    return t


def detail_parameters(d):
    q = d // 4
    return 2 * d * q + 9 * q + q * d


def tensor_sha(module):
    h = hashlib.sha256()
    for name, value in module.state_dict().items():
        x = value.detach().cpu().contiguous()
        h.update(f"{name}:{x.dtype}:{tuple(x.shape)}".encode())
        h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------- block forward
def _stats(store, *, v, c, r, t, z, zc, mode):
    with torch.no_grad():
        f = lambda a: float(a.float().norm())
        dt = down(t)
        store.append({"mode": mode, "grid": list(v.shape[-2:]), "coarse_grid": list(c.shape[-2:]),
                      "v_dtype": str(v.dtype), "c_dtype": str(c.dtype), "z_cast_dtype": str(zc.dtype),
                      "norm_c": f(c), "norm_r": f(r), "norm_t": f(t), "norm_projected_t": f(project(t)),
                      "norm_D_t": f(dt), "c_over_r": f(c) * 2. / max(f(r), 1e-30),  # |U c| = 2|c|
                      "max_abs_D_z_minus_c_fp32": float((down(z) - c.to(z.dtype)).abs().max()),
                      "max_abs_D_z_minus_c_cast": float((down(zc.to(z.dtype)) - c.to(z.dtype)).abs().max()),
                      "max_abs_cast_error_z": float((zc.to(z.dtype) - z).abs().max()),
                      "max_abs_r_cell_mean": float(down(r).abs().max())})


def r26_forward(self, x):
    mode = self.r26_mode
    if mode == "N":
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)
    v = self.conv1(x)
    s = self.conv2(x)
    hi = torch.promote_types(v.dtype, torch.float32)   # FP32 (FP64 stays FP64 in the synthetic tests)
    with torch.autocast(v.device.type, enabled=False):
        v32 = v.to(hi)
        dv = down(v32)
        r = v32 - up(dv)
    c = self.bottlenecks(dv.to(v.dtype) if not torch.is_autocast_enabled(v.device.type) else dv)  # autocast casts in convs
    with torch.autocast(v.device.type, enabled=False):
        uc = up(c.to(hi))
    t = getattr(self, DETAIL_NAME)(torch.cat((r, uc), 1))
    with torch.autocast(v.device.type, enabled=False):
        t32 = t.to(hi)
        z = uc + r + t32
        if mode == "P":
            z = z - up(down(t32))
        zc = z.to(c.dtype)
    if self.r26_monitor is not None:
        _stats(self.r26_monitor, v=v, c=c, r=r, t=t32, z=z, zc=zc, mode=mode)
    return self.conv3(zc + s)


def _block(model):
    owner = model
    for part in BLOCK_PATH.split("."):
        owner = owner[int(part)] if part.isdigit() else getattr(owner, part)
    return owner


def block_identity(model):
    """Instance attributes of the target block and its place in the encoder (not the name only)."""
    enc = model.encoder
    b = _block(model)
    d = b.conv1.conv.out_channels
    ident = {"class": type(b).__mro__[0].__name__ if not getattr(b, "r26_mode", None) else type(b).__mro__[1].__name__,
             "in_channels": b.conv1.conv.in_channels, "hidden": d, "out_channels": b.conv3.conv.out_channels,
             "num_blocks": len(b.bottlenecks), "fpn_blocks": len(enc.fpn_blocks), "levels": len(enc.in_channels),
             "feat_strides": list(enc.feat_strides), "hidden_dim": enc.hidden_dim,
             "index_used_for_stride8": len(enc.in_channels) - 1 - 1,
             "conv2_in": b.conv2.conv.in_channels}
    require(ident["class"] == "CSPRepLayer" and ident["levels"] == 3 and ident["fpn_blocks"] == 2
            and ident["index_used_for_stride8"] == 1 and ident["feat_strides"] == [8, 16, 32]
            and all(ident[k] == v for k, v in EXPECTED.items()) and ident["conv2_in"] == 512,
            ("R26 target block identity differs", ident))
    return ident


def install(model, mode, detail=None):
    """Turn fpn_blocks[1] into the R26 block; N keeps native arithmetic.  Returns the T module (or None)."""
    require(mode in MODES, f"unknown R26 mode {mode}")
    block_identity(model)
    return install_block(_block(model), mode, detail)


def install_block(b, mode, detail=None):
    """Install on one CSPRepLayer instance (used directly by the synthetic tests)."""
    require(mode in MODES, f"unknown R26 mode {mode}")
    require(not hasattr(b, "r26_mode"), "R26 already installed")
    require(not getattr(b, "_s15_checkpointed", False), "install R26 before activation checkpointing")
    native = type(b)
    b.__class__ = type("R26CSPRepLayer", (native,), {"forward": r26_forward, "_r26_native": native})
    b.r26_mode = mode
    b.r26_monitor = None
    if mode == "N":
        return None
    d = b.conv1.conv.out_channels
    t = detail if detail is not None else make_detail(d)
    p = next(b.parameters())
    t = t.to(device=p.device, dtype=p.dtype)
    b.add_module(DETAIL_NAME, t)
    require(sum(x.numel() for x in t.parameters()) == detail_parameters(d), "R26 detail parameter count")
    return t


def mode_of(model):
    return getattr(_block(model), "r26_mode", None)


def set_monitor(model, store):
    _block(model).r26_monitor = store


# ---------------------------------------------------------------- engine construction from the S15 bridge
def create(parent, device, mode, *, config=None, fixture=False, activation_checkpoint=True):
    """S15 seed0 bridge of the native e060 complete state, then R26 install on online and EMA models.

    Online parameters, Adam state and EMA values of every original parameter are inherited by name; the
    new T has empty Adam state and the same initial value in online and EMA; clocks continue.  The
    optimizer is rebuilt with the native partition rule so T sits in its native group/order.
    """
    from . import scale_baseline as S
    from ..baseline.training_v2b import _optimizer
    from ..protocol_r18 import checkpoint as native
    require(mode in MODES, f"unknown R26 mode {mode}")
    e, bridge = S.create(parent, device, config, seed=0, fixture=fixture, activation_checkpoint=False)
    rng_after_bridge = native.capture_rng(e.device)
    online_before = {n: p.detach().clone() for n, p in e.model.named_parameters()}
    ident = block_identity(e.model)
    t = install(e.model, mode)
    t_ema = install(e.ema.module, mode, copy.deepcopy(t) if t is not None else None)
    info = {"version": VERSION, "mode": mode, "block": BLOCK_PATH, "block_identity": ident,
            "new_parameters": 0, "T_init_sha256": None}
    if t is not None:
        t_ema.requires_grad_(False)
        require(tensor_sha(t) == tensor_sha(t_ema), "T online/EMA initial values differ")
        old = e.optimizer
        by_name = {}
        for g in old.param_groups:
            for name, p in zip(g["param_names"], g["params"]):
                by_name[name] = old.state.get(p, {})
        opt = _optimizer(e.model, e.config)
        for g, og in zip(opt.param_groups, old.param_groups):
            require(g["name"] == og["name"], "R26 optimizer group order")
            for k, v in og.items():
                if k not in ("params", "param_names"):
                    g[k] = copy.deepcopy(v)
            for name, p in zip(g["param_names"], g["params"]):
                if name in by_name and by_name[name]:
                    opt.state[p] = by_name[name]
        new = sorted(set(n for g in opt.param_groups for n in g["param_names"]) - set(by_name))
        require(new == sorted(f"{BLOCK_PATH}.{DETAIL_NAME}.{i}.weight" for i in (0, 2, 4)), f"R26 new names {new}")
        require(all(f"{BLOCK_PATH}.{DETAIL_NAME}" in n for n in opt.param_groups[1]["param_names"] if n in new),
                "R26 T must be in the native default group")
        e.optimizer = opt
        for n in new:
            e.parameter_steps[n] = 0
        info.update(new_parameters=sum(x.numel() for x in t.parameters()), T_init_sha256=tensor_sha(t),
                    new_parameter_names=new,
                    T_group={n: g["name"] for g in opt.param_groups for n in g["param_names"] if n in new})
    covered = [id(p) for g in e.optimizer.param_groups for p in g["params"]]
    require(sorted(covered) == sorted(id(p) for p in e.model.parameters() if p.requires_grad)
            and len(covered) == len(set(covered)), "R26 optimizer coverage")
    require(all(torch.equal(online_before[n], p) for n, p in e.model.named_parameters() if n in online_before),
            "R26 install changed an inherited parameter")
    require(native.exact_equal(native.capture_rng(e.device), rng_after_bridge), "R26 install consumed training RNG")
    e.r25_arm = f"R26-{mode}"
    wrapped = S.install_activation_checkpoint(e.model) if activation_checkpoint else []
    e.s15_activation_checkpoint = bool(activation_checkpoint)
    bridge = dict(bridge, activation_checkpoint_segments=wrapped, parameters=sum(p.numel() for p in e.model.parameters()))
    return e, {"S15_bridge": bridge, "R26": info}


def macs_block(h, w, mode, d=128, cin=512, cout=256, blocks=3):
    """Analytic multiply-accumulates of the target block per image (convs only; BN/act/pool/add excluded)."""
    hi = h * w
    lo = hi // 4
    conv12 = 2 * hi * cin * d
    rep = blocks * d * d * (9 + 1)
    conv3 = hi * d * cout
    if mode == "N":
        return {"conv1_conv2": conv12, "bottlenecks": rep * hi, "T": 0, "conv3": conv3,
                "total": conv12 + rep * hi + conv3}
    q = d // 4
    t = hi * (2 * d * q + 9 * q + q * d)
    return {"conv1_conv2": conv12, "bottlenecks": rep * lo, "T": t, "conv3": conv3,
            "total": conv12 + rep * lo + t + conv3}
