"""R28 arm F: the frozen demand of arm W, mapped by the FOVEA-style separable attraction rule instead of the bounded
allocation (TASK_R28_FOVEA_MAPPING_CONTROL_004).

Arm F is a controlled adaptation of one rule of the official FOVEA code, not a reproduction of the FOVEA method:

  demand     the same frozen demand network, weights, 384x672 FP32 eval / no_grad path and 12x21 non-negative map D as
             arm W (r28_resample.Context.demand, unchanged).  No detection loss reaches it; no training RNG is drawn.
  saliency   per axis the marginal of D + eps on the 12x21 grid, exactly arm W's marginals; each is read as values at
             the cell centres (linear between centres, constant beyond the outer ones) and sampled at the official
             nodes (51 horizontal, 31 vertical).  Marginal first, sampling second.
  map        RecasensSaliencyToGridMixin.separable_saliency_to_grid of github.com/tchittesh/fovea at commit c8707bd2:
             reflect padding by 30 nodes (anti_crop), Gaussian attraction kernel of FWHM 4 nodes (the released
             plain_kde_grid.py value, not the constructor default 13), node position = kernel-weighted mean of the
             padded node coordinates, then the official clamp of 2G-1 to [-1, 1].  G maps warped -> source.
  frames     G is piecewise linear between the nodes u = i/(n-1).  The image is rendered at output pixel centres
             u = (i+0.5)/L; target boxes of the denoising queries go source -> warped through the exact inverse of
             the same piecewise linear G; every predicted box family goes warped -> source through G at the output
             boundary.  Outside [0, 1] both directions extend linearly with the first / last node interval; nothing
             clips a box edge.  Map and box arithmetic are FP64.
  renderer   arm W's tent kernel (half-width max(1, footprint/1.5) source pixels, normalised, replicate border,
             horizontal then vertical, FP32) with EVERY non-zero weight: the number of taps is the support actually
             needed.  Arm W's 4-tap limit and its slope bounds are not applied to F.

G must be finite and strictly increasing.  Nothing here sorts, accumulates a maximum, floors or clamps a slope to make
it so: a violation raises and stops the forward.  A demand map that is constant by arm W's frozen test takes the same
L0 bypass as arm W.  The mapping adds no parameter, buffer or loss key; the detector is the L0 detector.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import torch

from . import r27_l0_control as L0
from . import r28_resample as R
from . import r28_warp as W
from .r27_local_memory import H_HW, L_HW, PARENT_SHA, PARENT_UPDATE, require, to_low

VERSION = "R28-004-fovea-map-control-v1"
ARM = "F"
ARMS = (ARM,)
FORMAL_KIND = {ARM: "network"}
CHECK_KINDS = R.CHECK_KINDS
GRID_SHAPE = (31, 51)                 # official canonical grid: 31 vertical nodes, 51 horizontal nodes
FWHM = 4.0                            # configs/models/plain_kde_grid.py: attraction_fwhm=4
PAD = min(GRID_SHAPE) - 1             # official padding_size
BOX_FAMILIES = R.BOX_FAMILIES
OUTPUT_KEYS = R.OUTPUT_KEYS
F64 = torch.float64
OFFICIAL = {
    "repository": "https://github.com/tchittesh/fovea",
    "commit": "c8707bd2e4f10678f6636d24c2915a43b44a59ed",
    "files": {"fovea/models/grid_generator.py": "4d36cec017d3acf888bb9643a5746507feb817b5fcce640bc8528bcf40c1596b",
              "configs/models/plain_kde_grid.py": "efbb5581425d329440a7227a0378bbc00e0365cc1f574c25cd671bd1b82e22f2"},
    "rule": "RecasensSaliencyToGridMixin.separable_saliency_to_grid",
}
DEFINITION = {
    "name": "r28_fovea_style_separable_map_v1", "official": OFFICIAL,
    "H_HW": list(H_HW), "L_HW": list(L_HW),
    "grid_shape": list(GRID_SHAPE), "attraction_fwhm": FWHM, "anti_crop": True, "padding_mode": "reflect",
    "padding_size": PAD, "kernel": "k(r) = exp(-4 log(2) r^2 / 4^2), r = -30..30 nodes",
    "saliency": "per axis the marginal of D + eps on the 12x21 grid (dx = sum over rows, dy = sum over columns); the "
                "marginal is read as values at the cell centres (j+0.5)/n, linear between centres and constant beyond "
                "the outer centres, sampled at the official nodes x=j/50 and y=i/30 and divided by its sum; marginal "
                "first, sampling second; no extra background weight and no learnable parameter",
    "map": "G_i = sum_j k(i-j) s_pad(j) p_j / sum_j k(i-j) s_pad(j) with p_j = j/(n-1), j = -30..n-1+30, then "
           "(clamp(2G-1, -1, 1) + 1)/2 as in the official code; G maps warped -> source on the nodes u = i/(n-1) and "
           "is piecewise linear between them",
    "inverse": "source -> warped is the exact inverse of the same piecewise linear G (interval search and linear "
               "interpolation); no second grid and no fit to uniform source knots",
    "extension": "outside [0, 1] both directions extend linearly with the first / last node interval; no box edge is "
                 "clipped",
    "monotonicity": "G must be finite and strictly increasing; no sorting, running maximum, slope floor or slope clamp "
                    "is applied; a violation raises and stops the forward",
    "identity": "a demand map that is constant by arm W's frozen test (relative range of each interpolated marginal "
                "<= 1e-12) takes the L0 bypass when every image of the batch is constant; in a mixed batch such an "
                "image gets the exact uniform nodes",
    "arithmetic": "map and box arithmetic FP64 (nodes computed on the CPU), image FP32",
    "pixel_convention": "output pixel centres u = (i+0.5)/L for the image, output pixel edges i/L for the footprint",
    "resampler": "arm W's tent: source pixel centre p = n_src G(u_centre) - 0.5, footprint d = n_src (G(u_right) - "
                 "G(u_left)), half-width h = max(1, d/1.5), weights max(0, 1 - |index - p|/h) normalised, replicate "
                 "border, horizontal then vertical, FP32; every non-zero weight is used (no 4-tap limit, no slope "
                 "bound)",
    "shared_with_W": {k: W.DEFINITION[k] for k in ("grid", "demand_input_hw", "eps", "demand_net")},
    "differences_from_official": [
        "the saliency is the frozen R28 demand network's marginal, not the official KDE of boxes or a video prior",
        "attraction_fwhm is the released configuration value 4; the constructor default 13 is not used and neither "
        "value was tuned on VisDrone",
        "the image is rendered by the common R28 tent kernel at pixel centres, not by bilinear grid_sample on a grid "
        "interpolated with align_corners=True",
        "boxes are mapped by the exact piecewise linear map with linear extension, not by unwarp_bboxes "
        "(grid_sample with border padding)",
        "FP64 map arithmetic instead of FP32",
        "the detector is RT-DETRv2-R18 with every output family mapped back to the source frame before the native "
        "losses",
    ],
}


def _identity():
    return {
        "version": VERSION, "arm": ARM, "task": "TASK_R28_FOVEA_MAPPING_CONTROL_004",
        "H": list(H_HW), "L": list(L_HW), "definition": DEFINITION,
        "demand_source": R.IDENTITY["W"]["demand_source"],
        "demand_path": "r28_resample.Context.demand of arm W, unchanged: the same weights file, 384x672 input, FP32, "
                       "eval mode, no gradient, autocast disabled",
        "map_device": "CPU FP64 for the nodes; box maps and tap positions FP64 on the model device",
        "identity_bypass": "a batch whose demand maps are all constant runs native.forward(to_low(I_H), targets) and "
                           "returns its output untouched",
        "forward": "native RTDETR.forward(resample(I_H, G), targets with boxes mapped to the warped canvas by the "
                   "inverse of G); pred_boxes of final / aux_outputs / enc_aux_outputs / dn_aux_outputs mapped back "
                   "to the source frame by G at the output boundary (FP64 arithmetic, FP32 result, linear extension, "
                   "no clipping)",
        "decoder_internal_frame": "warped canvas (queries, reference points, anchors, denoising boxes)",
        "criterion_frame": "source frame; original targets",
        "detector": L0.IDENTITY["forward"], "detector_state_dict": L0.IDENTITY["state_dict"],
        "detector_new_learnable_parameters": 0, "mapping_learnable_parameters": 0,
        "demand_network_parameters": R.IDENTITY["W"]["demand_network_parameters"],
        "targets": L0.IDENTITY["targets"], "anchor_position_cache_canvas": list(L_HW),
        "activation_checkpoint_segments": list(L0.CHECKPOINT_SEGMENTS),
        "parent_sha256": PARENT_SHA, "parent_update": PARENT_UPDATE,
        "is_not": {"W": "arm W allocates bounded slopes in closed form; arm F uses the attraction map and has no "
                        "slope bound",
                   "official_FOVEA": "not the FOVEA method end to end: shared demand, common renderer, pixel-centre "
                                     "convention and linear extension are adaptations",
                   "L0": "L0 reduces I_H uniformly with to_low; this arm resamples non-uniformly"},
    }


IDENTITY = {ARM: _identity()}
IDENTITY_SHA256 = {ARM: hashlib.sha256(json.dumps(IDENTITY[ARM], sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
_TYPES, _KERNELS = {}, {}


# ---------------------------------------------------------------- saliency at the official nodes
def node_saliency(marg, nodes):
    """(B, n) marginal on the cell centres (j+0.5)/n -> (B, nodes) values at k/(nodes-1), divided by their sum."""
    marg = marg.to(F64)
    n = marg.shape[-1]
    pos = torch.arange(nodes, dtype=F64, device=marg.device) / (nodes - 1) * n - 0.5
    i0 = pos.floor().clamp(0, n - 1).long()
    i1 = (i0 + 1).clamp(max=n - 1)
    fr = (pos - i0).clamp(0, 1)
    a, b = marg[:, i0], marg[:, i1]
    s = a + fr * (b - a)
    return s / s.sum(1, keepdim=True)


def reflect_index(n):
    """Source index of the padded positions j = -PAD .. n-1+PAD under reflect padding (the edge node is not repeated)."""
    j = torch.arange(-PAD, n + PAD)
    return torch.where(j < 0, -j, torch.where(j > n - 1, 2 * (n - 1) - j, j))


def kernel_matrix(n):
    """(n, n + 2 PAD) FP64: row i holds k(i - j) for the padded positions j (zero where |i - j| > PAD)."""
    if n not in _KERNELS:
        i = torch.arange(n, dtype=F64)[:, None]
        j = torch.arange(-PAD, n + PAD, dtype=F64)[None, :]
        r = i - j
        k = torch.exp(-4.0 * math.log(2.0) * r * r / (FWHM * FWHM))
        _KERNELS[n] = torch.where(r.abs() <= PAD, k, torch.zeros_like(k))
    return _KERNELS[n]


def attraction_nodes(s):
    """(B, n) positive node saliency -> (raw, G), FP64 on the CPU: the official kernel-weighted mean of the padded
    node coordinates before and after the official clamp."""
    s = s.to(F64).cpu()
    n = s.shape[1]
    require(n > PAD, "reflect padding needs more nodes than the padding size")
    sp = s[:, reflect_index(n)]
    p = torch.arange(-PAD, n + PAD, dtype=F64) / (n - 1)
    m = kernel_matrix(n)
    num = (m[None] * (sp * p)[:, None, :]).sum(-1)
    den = (m[None] * sp[:, None, :]).sum(-1)
    raw = num / den
    return raw, (torch.clamp(raw * 2 - 1, min=-1, max=1) + 1) / 2


def _constant(v):
    hi, lo = v.max(1).values, v.min(1).values
    return (hi - lo) <= W.CONST_RTOL * hi.clamp_min(1e-30)


@dataclass
class Plan:
    gx: torch.Tensor          # (B, 51) warped -> source nodes, horizontal
    gy: torch.Tensor          # (B, 31)
    identity: torch.Tensor    # (B,) bool: constant demand by arm W's test
    stats: dict               # batch readings of the map (python numbers)

    def select(self, index):
        return Plan(self.gx[index], self.gy[index], self.identity[index], self.stats)

    def to(self, device):
        return Plan(self.gx.to(device), self.gy.to(device), self.identity.to(device), self.stats)


def plan_from_demand(demand):
    """(B, GH, GW) demand without eps -> Plan (tensors on the CPU).  Reads nothing but the demand map."""
    d = demand.detach().to(F64).cpu() + W.EPS
    marg = {"y": d.sum(2), "x": d.sum(1)}
    identity = _constant(W.interp_marginal(marg["y"], L_HW[0])) & _constant(W.interp_marginal(marg["x"], L_HW[1]))
    nodes, stats = {}, {}
    for axis, n in (("y", GRID_SHAPE[0]), ("x", GRID_SHAPE[1])):
        raw, g = attraction_nodes(node_saliency(marg[axis], n))
        uniform = torch.arange(n, dtype=F64) / (n - 1)
        g = torch.where(identity[:, None], uniform[None].expand_as(g), g)
        step = g[:, 1:] - g[:, :-1]
        if not (bool(torch.isfinite(g).all()) and bool((step > 0).all())):
            bad = (~torch.isfinite(g)).any(1) | (step <= 0).any(1)
            raise RuntimeError("R28-F map is not finite and strictly increasing on axis %s: images %s, smallest node step %.3e"
                               % (axis, bad.nonzero().flatten().tolist(), float(step.min())))
        live = ~identity
        over = (raw - raw.clamp(0, 1)).abs()[live]              # official clamp of 2G-1 to [-1, 1]: amount removed per node
        stats[axis] = {"node_slope_min": float(step.min()) * (n - 1), "node_slope_max": float(step.max()) * (n - 1),
                       "endpoint_error_max": float(torch.maximum(raw[:, 0].abs(), (raw[:, -1] - 1).abs())[live].max()) if bool(live.any()) else 0.0,
                       "clamped_nodes": int((over > 0).sum()), "clamp_max": float(over.max()) if bool(live.any()) else 0.0,
                       "clamped_interior_nodes": int((over[:, 1:-1] > 0).sum()),      # position: any clamp away from the two end nodes
                       "first_node_max": float(g[:, 0].max()), "last_node_min": float(g[:, -1].min())}
        nodes[axis] = g
    return Plan(nodes["x"], nodes["y"], identity, stats)


# ---------------------------------------------------------------- maps (differentiable in the coordinate)
def unwarp(g, v):
    """warped -> source: the piecewise linear G at v.  g (B, n+1), v (B, M); linear extension outside [0, 1]."""
    n = g.shape[-1] - 1
    t = v.to(F64) * n
    i = t.detach().floor().clamp(0, n - 1).long()
    g0, g1 = g.gather(1, i), g.gather(1, i + 1)
    return g0 + (t - i) * (g1 - g0)


def warp(g, u):
    """source -> warped: the inverse of the same piecewise linear G.  Linear extension outside [G_0, G_n]."""
    n = g.shape[-1] - 1
    u = u.to(F64)
    i = (torch.searchsorted(g, u.detach().contiguous(), right=True) - 1).clamp(0, n - 1)
    g0, g1 = g.gather(1, i), g.gather(1, i + 1)
    return (i + (u - g0) / (g1 - g0)) / n


def map_boxes(plan, boxes, inverse):
    """cxcywh boxes (B, M, 4) -> cxcywh boxes in the other frame, edge by edge, in the dtype of `boxes`.

    inverse=False: source frame -> warped canvas (targets for the denoising queries).
    inverse=True:  warped canvas -> source frame (every predicted box family at the output boundary).
    """
    f = unwarp if inverse else warp
    x1, y1, x2, y2 = W.cxcywh_to_xyxy(boxes.to(F64)).unbind(-1)
    out = torch.stack([f(plan.gx, x1), f(plan.gy, y1), f(plan.gx, x2), f(plan.gy, y2)], -1)
    return W.xyxy_to_cxcywh(out).to(boxes.dtype)


def magnification(plan, xyxy):
    """Per-box (ox, oy) of normalised source xyxy boxes (B, M, 4) with positive width and height."""
    x1, y1, x2, y2 = xyxy.to(F64).unbind(-1)
    ox = (warp(plan.gx, x2) - warp(plan.gx, x1)) / (x2 - x1)
    oy = (warp(plan.gy, y2) - warp(plan.gy, y1)) / (y2 - y1)
    return ox, oy


# ---------------------------------------------------------------- renderer with the complete tent support
def pixel_positions(g, n_out):
    """Source positions (normalised) of the n_out + 1 output pixel edges and of the n_out output pixel centres."""
    j = torch.arange(n_out + 1, dtype=F64, device=g.device)
    b = g.shape[0]
    return unwarp(g, (j / n_out).expand(b, -1)), unwarp(g, ((j[:-1] + 0.5) / n_out).expand(b, -1))


def tent_taps(edge, centre, n_src):
    """Source indices (B, n_out, T), FP32 weights, T and the footprint (source pixels).  T is the largest support any
    output pixel of the batch needs; a tap beyond a pixel's own support has weight exactly zero."""
    p = n_src * centre - 0.5
    d = n_src * (edge[:, 1:] - edge[:, :-1])
    h = (d / 1.5).clamp_min(1.0)
    base = (p - h).floor() + 1
    span = int(((p + h).floor() - base + 1).max())
    idx = base[..., None] + torch.arange(span, dtype=F64, device=p.device)
    w = (1.0 - (idx - p[..., None]).abs() / h[..., None]).clamp_min(0)
    w = w / w.sum(-1, keepdim=True)
    return idx.long().clamp(0, n_src - 1), w.to(torch.float32), span, d


def apply_taps(x, ix, wx, iy, wy):
    """(B, C, Hs, Ws) FP32 -> (B, C, LH, LW): horizontal pass over every tap of ix, then vertical pass over iy."""
    B, C, Hs, _ = x.shape
    LW, LH = ix.shape[1], iy.shape[1]
    t = None
    for a in range(ix.shape[-1]):
        g = x.gather(3, ix[:, None, None, :, a].expand(B, C, Hs, LW)) * wx[:, None, None, :, a]
        t = g if t is None else t + g
    out = None
    for a in range(iy.shape[-1]):
        g = t.gather(2, iy[:, None, :, None, a].expand(B, C, LH, LW)) * wy[:, None, :, None, a]
        out = g if out is None else out + g
    return out


def resample(images, plan, out_hw=L_HW):
    """(B, C, Hs, Ws) -> ((B, C, LH, LW) FP32, readings) under the plan."""
    x = images.to(torch.float32)
    LH, LW = out_hw
    ix, wx, sx, dx = tent_taps(*pixel_positions(plan.gx, LW), x.shape[3])
    iy, wy, sy, dy = tent_taps(*pixel_positions(plan.gy, LH), x.shape[2])
    info = {"taps_x": sx, "taps_y": sy, "footprint_max_px": float(torch.maximum(dx.max(), dy.max())),
            "footprint_min_px": float(torch.minimum(dx.min(), dy.min()))}
    return apply_taps(x, ix, wx, iy, wy), info


# ---------------------------------------------------------------- forward
def warp_targets(plan, targets):
    """Copies of the target dicts whose boxes live in the warped canvas; the originals are not touched."""
    pad, n = R._padded_xyxy(targets)
    if not sum(n):
        return [dict(t) for t in targets]
    x1, y1, x2, y2 = pad.unbind(-1)
    out = W.xyxy_to_cxcywh(torch.stack([warp(plan.gx, x1), warp(plan.gy, y1), warp(plan.gx, x2), warp(plan.gy, y2)], -1))
    return [{**t, "boxes": out[i, :n[i]].to(t["boxes"].dtype)} for i, t in enumerate(targets)]


def unwarp_outputs(plan, out):
    """A new output dict whose every pred_boxes tensor is in the source frame (one batched map)."""
    require(set(out) <= OUTPUT_KEYS, f"R28-F unknown detector output keys: {sorted(set(out) - OUTPUT_KEYS)}")
    for k in BOX_FAMILIES:
        for d in out.get(k, ()):
            require(set(d) == {"pred_logits", "pred_boxes"}, f"R28-F unknown keys in {k}: {sorted(d)}")
    boxes = [out["pred_boxes"]] + [d["pred_boxes"] for k in BOX_FAMILIES for d in out.get(k, ())]
    sizes = [b.shape[1] for b in boxes]
    mapped = map_boxes(plan, torch.cat([b.float() for b in boxes], 1), inverse=True).split(sizes, 1)
    res = dict(out)
    res["pred_boxes"] = mapped[0]
    i = 1
    for k in BOX_FAMILIES:
        if k in out:
            res[k] = []
            for d in out[k]:
                res[k].append({"pred_logits": d["pred_logits"], "pred_boxes": mapped[i]})
                i += 1
    require(i == len(mapped), "R28-F box family count")
    return res


def observe(plan, targets, render):
    """Per-image CPU readings of one training forward (no model call).  The keys of arm W's record, with the slope
    range given as magnification (1 / node slope of G), plus the map and tap readings of this arm."""
    pad, n = R._padded_xyxy(targets)
    ox, oy = magnification(plan, pad)
    w = (pad[..., 2] - pad[..., 0]) * L_HW[1]
    h = (pad[..., 3] - pad[..., 1]) * L_HW[0]
    g, size, short = torch.sqrt(ox * oy).cpu(), torch.sqrt(w * h).cpu(), torch.minimum(ox * w, oy * h).cpu()
    sx, sy = plan.stats["x"], plan.stats["y"]
    lo, hi = min(sx["node_slope_min"], sy["node_slope_min"]), max(sx["node_slope_max"], sy["node_slope_max"])
    return {"n": n, "g": [g[i, :k] for i, k in enumerate(n)], "size": [size[i, :k] for i, k in enumerate(n)],
            "short": [short[i, :k] for i, k in enumerate(n)],
            "slope_min": 1.0 / hi, "slope_max": 1.0 / lo, "identity_images": int(plan.identity.sum()),
            "fovea": {"node_slope_min": lo, "node_slope_max": hi,
                      "endpoint_error_max": max(sx["endpoint_error_max"], sy["endpoint_error_max"]),
                      "clamped_nodes": sx["clamped_nodes"] + sy["clamped_nodes"],
                      "clamped_interior_nodes": sx["clamped_interior_nodes"] + sy["clamped_interior_nodes"],
                      "clamp_max": max(sx["clamp_max"], sy["clamp_max"]),
                      "taps_x": None if render is None else render["taps_x"],
                      "taps_y": None if render is None else render["taps_y"],
                      "footprint_max_px": None if render is None else render["footprint_max_px"]}}


def _forward(self, x, targets=None):
    ctx, native = self.r28_ctx, self._r28f_native
    require(x.dim() == 4 and tuple(x.shape[-2:]) == H_HW and x.dtype == torch.float32,
            f"R28-F expects the FP32 H input, got {tuple(x.shape)} {x.dtype}")
    mark = vars(ctx).get("probe") or (lambda _name: None)
    with torch.autocast(x.device.type, enabled=False), torch.no_grad():
        d = ctx.demand(x)
        require(bool(torch.isfinite(d).all()) and bool((d >= 0).all()), "R28-F demand map is not finite / non-negative")
        mark("demand")
        plan = plan_from_demand(d).to(x.device)
        uniform = bool(plan.identity.all())
        mark("map")
        render = None
        if not uniform:
            xw, render = resample(x, plan)
            tw = None if targets is None else warp_targets(plan, targets)
        if ctx.observe and targets is not None:
            ctx.records.append(observe(plan, targets, render))
    if uniform:
        out = native.forward(self, to_low(x), targets)
        mark("detector")
        return out
    mark("render")
    out = native.forward(self, xw, tw)
    mark("detector")
    with torch.autocast(x.device.type, enabled=False):
        res = unwarp_outputs(plan, out)
    mark("unwarp")
    return res


def is_wrapped(model):
    cls = type(model)
    return (cls.__dict__.get("_r28f_native") is not None and cls.__dict__.get("forward") is _forward
            and "forward" not in vars(model) and isinstance(vars(model).get("r28_ctx"), R.Context)
            and not hasattr(model, "r27_ctx"))


def swap(model, ctx):
    """L0-wrapped native model -> arm F class, in place; the module layout and the state dict keys do not change."""
    require(L0.is_wrapped(model), "R28-F swap expects an L0-wrapped native model")
    native = type(model)._l0_native
    before = L0._layout(model)
    if native not in _TYPES:
        _TYPES[native] = type("R28FoveaRTDETR", (native,), {"forward": _forward, "_r28f_native": native})
    model.__class__ = _TYPES[native]
    vars(model)["r28_ctx"] = ctx
    require(L0._layout(model) == before and is_wrapped(model) and not L0.is_wrapped(model) and not R.is_wrapped(model),
            "R28-F swap changed the module layout")


def check_source(arm, source, check):
    require(arm == ARM, f"R28-F arm: {arm}")
    kind = source["kind"]
    require(kind == FORMAL_KIND[arm] or (check and kind in CHECK_KINDS),
            f"R28 arm {arm} cannot use a '{kind}' demand source outside a check")


# ---------------------------------------------------------------- engine / evaluation model
def create(parent, device, arm, source, *, fixture=False, activation_checkpoint=True, check=False):
    """Arm W's construction with the arm F forward: the demand context first (its network initialiser draws from the
    torch RNG), then L0.create, which restores the parent training RNG as its last step."""
    check_source(arm, source, check)
    ctx = R.Context(source)
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
              "mapping_learnable_parameters": 0,
              "detector_state_keys_equal_parent": checks["model_keys_equal_parent"] and checks["EMA_keys_equal_parent"]}
    require(all(checks["wrapped"].values()) and checks["new_parameters"] == 0, ("R28-F bridge mismatch", checks))
    return e, checks


def build_eval_model(state, device, arm, source, *, check=False):
    """FP32 evaluation model from a saved module state (native key set) with the arm's demand source."""
    check_source(arm, source, check)
    ctx = R.Context(source).to(device)
    model = L0.build_eval_model(state, device)
    swap(model, ctx)
    model.r28_arm = arm
    return model
