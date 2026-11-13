"""R28 arms W and K: the uniform H->L reduction of L0 replaced by the budgeted separable resampling of r28_warp.

Definition: CANDIDATE_R28_001.md section 2 with TASK_R28_CODEX_AMENDMENT_002.md section 2.

The detector is the L0 detector: native RT-DETRv2-R18 built at L, native key set, the e060 parent loads strictly, no
parameter, buffer or loss key is added.  The forward receives the H tensor I_H (1152x2016, [0,1], FP32) and

  1. reads a demand map (12x21): arm W from the frozen demand network on I_H reduced to 384x672, arm K from one fixed
     map (the train_core mean).  The demand source lives in a plain context object outside the module tree; it is
     evaluated without gradient, FP32, autocast disabled, and never receives a detection loss;
  2. allocates per-axis slopes in closed form (FP64, on the CPU: the sorted-breakpoint solution uses cumsum, which
     the deterministic-algorithm switch refuses on CUDA) and builds the monotone maps;
  3. if every image of the batch has the exactly uniform plan, calls the unchanged L0 path
     native.forward(to_low(I_H), targets) and returns its output untouched;
  4. otherwise resamples I_H to the L canvas (FP32), gives the decoder a copy of the targets whose boxes are mapped
     into the warped canvas (only the denoising queries read them; the caller's targets are not modified), runs the
     native forward, and maps every predicted box family of the output (final, decoder aux, encoder aux, denoising
     aux) back to the source frame, edge by edge, FP64 arithmetic, FP32 result, linear extension outside [0, 1],
     no clipping.  Queries, reference points and anchors inside the decoder stay in warped coordinates.

The criterion and the evaluator therefore see source-frame boxes and the original targets.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from . import r27_l0_control as L0
from . import r28_warp as W
from .r27_local_memory import H_HW, L_HW, PARENT_SHA, PARENT_UPDATE, require, to_low

VERSION = "R28-001-budgeted-resampling-v1"
ARMS = ("W", "K")
FORMAL_KIND = {"W": "network", "K": "fixed"}
CHECK_KINDS = ("constant", "synthetic")
DEMAND_SCHEMA = "r28-demand-net-v1"
PRIOR_SCHEMA = "r28-fixed-demand-v1"
BOX_FAMILIES = ("aux_outputs", "enc_aux_outputs", "dn_aux_outputs")
OUTPUT_KEYS = {"pred_logits", "pred_boxes", "enc_meta", "dn_meta", *BOX_FAMILIES}
F64 = torch.float64


def _identity(arm):
    body = {
        "version": VERSION, "arm": arm, "task": "TASK_R28_001 + TASK_R28_CODEX_AMENDMENT_002",
        "H": list(H_HW), "L": list(L_HW), "definition": W.DEFINITION,
        "demand_source": {"W": "frozen demand network (r28_warp.DemandNet) on I_H reduced to 384x672; eval mode, no "
                               "gradient, FP32, autocast disabled; trained once on train_core, never on a detection "
                               "loss",
                          "K": "one fixed 12x21 map: arithmetic mean over train_core of the per-image deployment "
                               "demand map (images without GT count as zero); no network"}[arm],
        "allocation_device": "CPU FP64 (CUDA cumsum is refused by the deterministic-algorithm switch)",
        "identity_bypass": "a batch whose plans are all exactly uniform runs native.forward(to_low(I_H), targets) "
                           "and returns its output untouched",
        "forward": "native RTDETR.forward(resample(I_H, plan), targets with boxes mapped to the warped canvas); "
                   "pred_boxes of final / aux_outputs / enc_aux_outputs / dn_aux_outputs mapped back to the source "
                   "frame at the output boundary (FP64 arithmetic, FP32 result, linear extension, no clipping)",
        "decoder_internal_frame": "warped canvas (queries, reference points, anchors, denoising boxes)",
        "criterion_frame": "source frame; original targets",
        "detector": L0.IDENTITY["forward"], "detector_state_dict": L0.IDENTITY["state_dict"],
        "detector_new_learnable_parameters": 0,
        "demand_network_parameters": {"W": "about 0.34 M, frozen, outside the detector module tree", "K": 0}[arm],
        "targets": L0.IDENTITY["targets"], "anchor_position_cache_canvas": list(L_HW),
        "activation_checkpoint_segments": list(L0.CHECKPOINT_SEGMENTS),
        "parent_sha256": PARENT_SHA, "parent_update": PARENT_UPDATE,
        "is_not": {"L0": "L0 reduces I_H uniformly with to_low; this arm resamples non-uniformly",
                   "S15": "S15 runs the native network on the H canvas; this arm keeps the L canvas token count",
                   "other_arm": "W and K differ only in the demand source"},
    }
    return body


IDENTITY = {a: _identity(a) for a in ARMS}
IDENTITY_SHA256 = {a: hashlib.sha256(json.dumps(IDENTITY[a], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                   for a in ARMS}
_TYPES = {}


# ---------------------------------------------------------------- demand source (outside the module tree)
def synthetic_demand(seed):
    """A fixed non-constant 12x21 demand map for checks: two blobs whose position depends on the seed."""
    gh, gw = W.GRID
    g = torch.Generator().manual_seed(1000 + int(seed))
    c = torch.rand(2, 2, generator=g, dtype=F64)
    y = (torch.arange(gh, dtype=F64) + 0.5) / gh
    x = (torch.arange(gw, dtype=F64) + 0.5) / gw
    d = torch.zeros(gh, gw, dtype=F64)
    for (cy, cx), amp, s in zip(c, (6.0, 2.5), (0.12, 0.2)):
        d = d + amp * torch.exp(-((y[:, None] - cy) ** 2 + (x[None, :] - cx) ** 2) / (2 * s * s))
    return d


def _file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _pinned(ref):
    p = Path(ref["path"])
    require(p.is_file() and p.stat().st_size == ref["bytes"] and _file_sha(p) == ref["sha256"],
            f"R28 demand source file differs from its reference: {ref['path']}")
    return p


class Context:
    """Demand source of one engine or evaluation model.  Not an nn.Module: nothing here enters a detector state."""

    def __init__(self, source):
        from ..protocol_r18.engine import tensor_digest
        kind = source["kind"]
        require(kind in ("network", "fixed", *CHECK_KINDS), f"R28 demand source kind: {kind}")
        self.kind, self.source, self.net, self.fixed = kind, dict(source), None, None
        self.observe, self.records = False, []
        self.device = torch.device("cpu")
        if kind == "network":
            s = torch.load(_pinned(source["file"]), map_location="cpu", weights_only=True)
            require(s["schema"] == DEMAND_SCHEMA and s["definition"] == W.DEFINITION, "R28 demand network file schema")
            net = W.DemandNet()  # draws from the torch RNG; callers restore the training RNG afterwards
            net.load_state_dict(s["model"], strict=True)
            self.net = net.eval().requires_grad_(False)
            self.digest = tensor_digest(self.net.state_dict().items())
            require(self.digest == s["state_sha256"], "R28 demand network state digest")
        elif kind == "fixed":
            s = json.loads(_pinned(source["file"]).read_text())
            require(s["schema"] == PRIOR_SCHEMA and s["definition"] == W.DEFINITION, "R28 fixed demand file schema")
            self.fixed = torch.tensor(s["demand"], dtype=F64)
        elif kind == "constant":
            self.fixed = torch.zeros(W.GRID, dtype=F64)
        else:
            self.fixed = synthetic_demand(source["seed"])
        if self.fixed is not None:
            require(tuple(self.fixed.shape) == W.GRID and bool(torch.isfinite(self.fixed).all())
                    and bool((self.fixed >= 0).all()), "R28 fixed demand map")
            self.digest = tensor_digest([("demand", self.fixed)])

    def to(self, device):
        self.device = torch.device(device)
        if self.net is not None:
            self.net.to(self.device)
        return self

    def binding(self):
        return {"kind": self.kind, "file": self.source.get("file"), "seed": self.source.get("seed"),
                "state_sha256": self.digest}

    def demand(self, x):
        """(B, GH, GW) FP64 on the CPU, without eps."""
        b = x.shape[0]
        if self.net is None:
            return self.fixed.unsqueeze(0).expand(b, -1, -1)
        require(not self.net.training and not any(p.requires_grad for p in self.net.parameters()),
                "R28 demand network must be frozen in eval mode")
        d = self.net(W.demand_input(x))
        require(tuple(d.shape) == (b, *W.GRID) and d.dtype == torch.float32, "R28 demand network output")
        return d.to(F64).cpu()

    def plan(self, x):
        d = self.demand(x)
        require(bool(torch.isfinite(d).all()) and bool((d >= 0).all()), "R28 demand map is not finite / non-negative")
        return W.plan_from_demand(d).to(x.device)


# ---------------------------------------------------------------- forward
def _padded_xyxy(targets):
    """Target boxes as one (B, N, 4) FP64 xyxy tensor (padding rows are a harmless central box) and the counts."""
    n = [len(t["boxes"]) for t in targets]
    dev = targets[0]["boxes"].device
    pad = torch.tensor([0.25, 0.25, 0.75, 0.75], dtype=F64, device=dev).repeat(len(n), max(max(n), 1), 1)
    for i, t in enumerate(targets):
        if n[i]:
            pad[i, :n[i]] = W.cxcywh_to_xyxy(t["boxes"].to(F64))
    return pad, n


def warp_targets(plan, targets):
    """Copies of the target dicts whose boxes live in the warped canvas; the originals are not touched."""
    pad, n = _padded_xyxy(targets)
    if not sum(n):
        return [dict(t) for t in targets]
    x1, y1, x2, y2 = pad.unbind(-1)
    out = torch.stack([W.fwd(plan.kx, x1), W.fwd(plan.ky, y1), W.fwd(plan.kx, x2), W.fwd(plan.ky, y2)], -1)
    out = W.xyxy_to_cxcywh(out)
    return [{**t, "boxes": out[i, :n[i]].to(t["boxes"].dtype)} for i, t in enumerate(targets)]


def unwarp_outputs(plan, out):
    """A new output dict whose every pred_boxes tensor is in the source frame (one batched inverse map)."""
    require(set(out) <= OUTPUT_KEYS, f"R28 unknown detector output keys: {sorted(set(out) - OUTPUT_KEYS)}")
    for k in BOX_FAMILIES:
        for d in out.get(k, ()):
            require(set(d) == {"pred_logits", "pred_boxes"}, f"R28 unknown keys in {k}: {sorted(d)}")
    boxes = [out["pred_boxes"]] + [d["pred_boxes"] for k in BOX_FAMILIES for d in out.get(k, ())]
    sizes = [b.shape[1] for b in boxes]
    mapped = W.map_boxes(plan, torch.cat([b.float() for b in boxes], 1), inverse=True).split(sizes, 1)
    res = dict(out)
    res["pred_boxes"] = mapped[0]
    i = 1
    for k in BOX_FAMILIES:
        if k in out:
            res[k] = []
            for d in out[k]:
                res[k].append({"pred_logits": d["pred_logits"], "pred_boxes": mapped[i]})
                i += 1
    require(i == len(mapped), "R28 box family count")
    return res


def observe(plan, targets):
    """Per-image CPU readings of one training forward (no model call): g, uniform-L size and warped short side."""
    pad, n = _padded_xyxy(targets)
    ox, oy = W.magnification(plan, pad)
    w = (pad[..., 2] - pad[..., 0]) * L_HW[1]
    h = (pad[..., 3] - pad[..., 1]) * L_HW[0]
    g, size, short = torch.sqrt(ox * oy).cpu(), torch.sqrt(w * h).cpu(), torch.minimum(ox * w, oy * h).cpu()
    return {"n": n, "g": [g[i, :k] for i, k in enumerate(n)], "size": [size[i, :k] for i, k in enumerate(n)],
            "short": [short[i, :k] for i, k in enumerate(n)],
            "slope_min": float(torch.minimum(plan.mx.min(), plan.my.min())),
            "slope_max": float(torch.maximum(plan.mx.max(), plan.my.max())),
            "identity_images": int(plan.identity.sum())}


def _forward(self, x, targets=None):
    ctx, native = self.r28_ctx, self._r28_native
    require(x.dim() == 4 and tuple(x.shape[-2:]) == H_HW and x.dtype == torch.float32,
            f"R28 expects the FP32 H input, got {tuple(x.shape)} {x.dtype}")
    with torch.autocast(x.device.type, enabled=False), torch.no_grad():
        plan = ctx.plan(x)
        uniform = bool(plan.identity.all())
        if ctx.observe and targets is not None:
            ctx.records.append(observe(plan, targets))
        if not uniform:
            xw = W.resample(x, plan)
            tw = None if targets is None else warp_targets(plan, targets)
    if uniform:
        return native.forward(self, to_low(x), targets)
    out = native.forward(self, xw, tw)
    with torch.autocast(x.device.type, enabled=False):
        return unwarp_outputs(plan, out)


def is_wrapped(model):
    cls = type(model)
    return (cls.__dict__.get("_r28_native") is not None and cls.__dict__.get("forward") is _forward
            and "forward" not in vars(model) and isinstance(vars(model).get("r28_ctx"), Context)
            and not hasattr(model, "r27_ctx"))


def swap(model, ctx):
    """L0-wrapped native model -> R28 class, in place; the module layout and the state dict keys do not change."""
    require(L0.is_wrapped(model), "R28 swap expects an L0-wrapped native model")
    native = type(model)._l0_native
    before = L0._layout(model)
    if native not in _TYPES:
        _TYPES[native] = type("R28RTDETR", (native,), {"forward": _forward, "_r28_native": native})
    model.__class__ = _TYPES[native]
    vars(model)["r28_ctx"] = ctx
    require(L0._layout(model) == before and is_wrapped(model) and not L0.is_wrapped(model),
            "R28 swap changed the module layout")


def check_source(arm, source, check):
    require(arm in ARMS, f"R28 arm: {arm}")
    kind = source["kind"]
    require(kind == FORMAL_KIND[arm] or (check and kind in CHECK_KINDS),
            f"R28 arm {arm} cannot use a '{kind}' demand source outside a check")


# ---------------------------------------------------------------- engine / evaluation model
def create(parent, device, arm, source, *, fixture=False, activation_checkpoint=True, check=False):
    """The L0 construction from the e060 parent with the forward swapped.  The demand context is built first: its
    network initialiser draws from the torch RNG, and L0.create restores the parent training RNG as its last step."""
    check_source(arm, source, check)
    ctx = Context(source)
    e, checks = L0.create(parent, device, fixture=fixture, activation_checkpoint=activation_checkpoint)
    ctx.to(e.device)
    swap(e.model, ctx)
    swap(e.ema.module, ctx)
    del e.l0_arm
    e.r28_arm = e.r25_arm = arm
    e.r28_ctx = ctx
    e.r28_check = bool(check)
    checks = {**checks, "arm": arm, "r25_arm": arm, "identity_sha256": IDENTITY_SHA256[arm], "L0_identity_sha256":
              L0.IDENTITY_SHA256, "demand": ctx.binding(), "wrapped": {"online": is_wrapped(e.model), "EMA":
                                                                      is_wrapped(e.ema.module)},
              "demand_network_parameters": 0 if ctx.net is None else sum(p.numel() for p in ctx.net.parameters()),
              "detector_state_keys_equal_parent": checks["model_keys_equal_parent"] and checks["EMA_keys_equal_parent"]}
    require(all(checks["wrapped"].values()) and checks["new_parameters"] == 0, ("R28 bridge mismatch", checks))
    return e, checks


def build_eval_model(state, device, arm, source, *, check=False):
    """FP32 evaluation model from a saved module state (native key set) with the arm's demand source."""
    check_source(arm, source, check)
    ctx = Context(source).to(device)
    model = L0.build_eval_model(state, device)
    swap(model, ctx)
    model.r28_arm = arm
    return model
