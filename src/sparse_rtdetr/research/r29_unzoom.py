"""R29 arms QP ("P") and QC ("C"): query-wise un-zoomed read of the warped decoder memory (TASK_R29_001).

Both arms are arm FG (r28_fovea_geo: the frozen demand network, the fixed kernel width 19.03125, the same maps, the
same renderer, warped decoder frame, source-frame losses) with one scientific change: the read operator of the three
decoder cross-attention layers.  N is arm FG itself and never runs through this module.

Notation of the task: F is source -> warped (r28_fovea.warp), G = F^-1 is warped -> source (r28_fovea.unwarp); both are
read from the plan of the same forward (gx, gy), nothing is estimated again.

For every query of every decoder layer (ordinary and denoising), with the native warped reference b_w = (c_w, s_w):
    a_o = G(c_w - s_w / 2),  b_o = G(c_w + s_w / 2),  c_o = (a_o + b_o) / 2,  s_o = b_o - a_o        (per axis, FP64)
    p_o = c_o + sampling_offsets * (1 / n_points_of_the_level) * s_o * 0.5                           (FP64)
V_l is the value tensor after the native value_proj and value_mask, split by level.
    arm QP (P, direct):  sample = bilinear(V_l, F(p_o))
    arm QC (C, dense):   U_l[i, j] = bilinear(V_l, F(q[i, j])),  q[i, j] = ((j + .5) / W_l, (i + .5) / H_l);
                         sample = bilinear(U_l, p_o)            (U_l has the size and channels of V_l, not detached)
Everything else of the layer is the native code: query_pos, reference update and detach, offset / weight networks,
softmax, value / output projection, the attention-weighted sum, self-attention and FFN.  No parameter is added.

Sampling: every bilinear above is the frozen training_v2b_deterministic_sampling.bilinear_grid_sample (bilinear, zeros,
align_corners=False, grid 2 p - 1, FP32); the P read and the C read go through the frozen attention core
deformable_attention_core_func_v2 (the model's own ms_deformable_attn_core), the C reconstruction calls the frozen
sampler directly.  Geometry is FP64; coordinates handed to the sampler are FP32.  Nothing is clipped; outside [0, 1]
the maps extend linearly as in arm FG.  The gradient of the sampling positions passes through F to the offsets; the
map nodes receive no gradient.

Identity: a batch whose maps are all the identity runs the native attention forward untouched.  In a mixed batch an
identity image uses the native sampling positions (and, in arm QC, V_l itself): no identity resampling is inserted.

The plan reaches the attention layers through a Reader object installed on the three cross-attention modules of one
model instance for the duration of one model forward and removed in a finally block; a cross-attention call without it
raises.  The decoder is outside the activation-checkpoint segments, so no backward replay calls these layers.

QC is a matched adaptation of the LZU idea (whole-feature un-zoom before the read), not a reproduction of the official
LZU detector.  The intermediate reconstruction changes bandwidth, smoothing, aliasing, borders and gradients.
"""
from __future__ import annotations

import contextlib
import functools
import hashlib
import json

import torch
import torch.nn.functional as F

from ..baseline import training_v2b_deterministic_sampling as DS
from . import r27_l0_control as L0
from . import r28_fovea as FV
from . import r28_fovea_cal as FS
from . import r28_fovea_geo as FG
from . import r28_resample as R
from .r27_local_memory import H_HW, require, to_low

VERSION = "R29-001-query-unzoom-read-v1"
ARMS = ("QP", "QC")
SHORT = {"QP": "P", "QC": "C"}
READ = {"QP": "direct", "QC": "dense"}
MODES = ("direct", "dense", "native")           # "native" only exists for the wiring test of the copied layer prefix
BASE_ARM = FG.ARM
LAYERS, LEVELS, POINTS, HEADS, OFFSET_SCALE = 3, 3, [4, 4, 4], 8, 0.5
F64, F32 = torch.float64, torch.float32
FORMAL_KIND = {a: "network" for a in ARMS}
CHECK_KINDS = R.CHECK_KINDS

DEFINITION = {
    "name": "r29_query_unzoom_read_v1",
    "base": {"arm": BASE_ARM, "module": "r28_fovea_geo (arm FG, unchanged)", "identity_sha256": FG.IDENTITY_SHA256[BASE_ARM],
             "forward": "the body of r28_fovea_cal._forward at the fixed width, with the read context installed around the "
                        "native detector call"},
    "frames": {"F": "source -> warped: r28_fovea.warp on the plan nodes of this forward",
               "G": "warped -> source: r28_fovea.unwarp on the same nodes",
               "source": "normalised coordinates of the augmented H canvas of this forward"},
    "scope": {"decoder_layers": LAYERS, "memory_levels": LEVELS, "points_per_level": list(POINTS), "queries": "ordinary and denoising",
              "reference": "four-dimensional references only; the two-dimensional branch is refused"},
    "source_reference": "a_o = G(c_w - s_w/2), b_o = G(c_w + s_w/2), c_o = (a_o + b_o)/2, s_o = b_o - a_o per axis; not G(c_w)",
    "source_points": "p_o = c_o + sampling_offsets * (1 / n_points_of_the_level) * s_o * 0.5",
    "value": "V_l = the native value_proj output with the native value_mask applied, split by level (the same place for both arms)",
    "read": {"QP": "sample = bilinear(V_l, F(p_o)); no source-uniform memory is built",
             "QC": "U_l[i, j] = bilinear(V_l, F(((j + .5)/W_l, (i + .5)/H_l))), same size and channels as V_l, each decoder "
                   "layer from its own projection, not detached; sample = bilinear(U_l, p_o)"},
    "sampler": {"function": "training_v2b_deterministic_sampling.bilinear_grid_sample at the P read, the C reconstruction and the C read",
                "geometry": {"mode": "bilinear", "padding_mode": "zeros", "align_corners": False}, "grid": "2 p - 1", "clip": None,
                "core": "the model's own ms_deformable_attn_core (deterministic_gather): split, sample, concat, multiply, sum"},
    "precision": {"geometry": "float64", "sampling_coordinates": "float32", "bilinear": "float32",
                  "attention_weights_and_projections": "native (autocast boundary unchanged)"},
    "gradient": {"offsets": "through F(p_o) (QP) or through p_o (QC)", "value": "kept", "attention": "kept",
                 "reference": "native detach unchanged", "map_nodes": "none"},
    "identity": {"all_identity_batch": "native attention forward, bit for bit",
                 "identity_image_in_a_mixed_batch": "native sampling positions; arm QC reads V_l itself"},
    "context": "one Reader per model forward, installed on the three cross-attention modules of that model instance and removed "
               "in a finally block; no global or cached plan; a cross-attention call without it raises",
    "unchanged": ["encoder pool, top-300 selection and box updates in the warped frame", "output mapping to the source frame (arm FG)",
                  "denoising noise, masks, targets, losses and counts (warped denoising targets of arm FG)",
                  "backbone, FPN, loss keys / weights / normalisers, score definition, post-processing"],
    "new_learnable_parameters": 0,
    "not": {"official_LZU": "arm QC is a matched adaptation of the whole-feature un-zoom idea, not the official LZU detector",
            "interpolation_only": "QP and QC also differ in bandwidth, smoothing, aliasing, border response and gradients",
            "claim_of_a_native_bug": "the native four-dimensional reference already scales offsets by the box size"},
}


def _identity(arm):
    base = FG.IDENTITY[BASE_ARM]
    return {**{k: v for k, v in base.items() if k not in ("version", "arm", "task", "is_not")},
            "version": VERSION, "arm": arm, "short_name": SHORT[arm], "task": "TASK_R29_001", "read_mode": READ[arm],
            "read_definition": {**{k: v for k, v in DEFINITION.items() if k != "read"}, "read": DEFINITION["read"][arm]},
            "base_arm": BASE_ARM, "base_identity_sha256": FG.IDENTITY_SHA256[BASE_ARM],
            "detector_new_learnable_parameters": 0,
            "is_not": {**base["is_not"], "FG": "arm FG (N) reads with the native warped sampling positions",
                       "the_other_arm": "QP reads the warped memory directly, QC reads a rebuilt source-uniform memory; a state "
                                        "of one is refused by the other",
                       "R27_P_or_C": "the R27 arms named P and C are local-memory arms of another task"}}


IDENTITY = {a: _identity(a) for a in ARMS}
IDENTITY_SHA256 = {a: hashlib.sha256(json.dumps(IDENTITY[a], sort_keys=True, separators=(",", ":")).encode()).hexdigest() for a in ARMS}
_TYPES, _ATTN_TYPES = {}, {}


# ---------------------------------------------------------------- read context
class Reader:
    """Read context of one model forward.  plan=None is the identity bypass (native attention forward).  Not a module;
    never part of a state."""

    def __init__(self, mode, plan, num_queries, audit=None):
        require(mode in MODES, f"R29 read mode: {mode!r}")
        self.mode, self.plan, self.num_queries, self.audit = mode, plan, int(num_queries), audit
        self.calls, self._grids = 0, {}

    def grid(self, level, h, w):
        """2 F(q) - 1 of one memory level for this forward's plan (computed once per forward, never across forwards)."""
        key = (int(level), int(h), int(w))
        if key not in self._grids:
            self._grids[key] = source_grid(self.plan, int(h), int(w))
        return self._grids[key]


def _attention_layers(model):
    return [layer.cross_attn for layer in model.decoder.decoder.layers]


@contextlib.contextmanager
def reading(model, reader):
    """Install `reader` on the cross-attention modules of this model instance for one detector call."""
    mods = _attention_layers(model)
    require(not any("r29_reader" in vars(m) for m in mods), "R29 read context is already installed on this model (nested forward)")
    for m in mods:
        vars(m)["r29_reader"] = reader
    try:
        yield reader
    finally:
        for m in mods:
            vars(m).pop("r29_reader", None)


# ---------------------------------------------------------------- geometry (FP64)
def source_grid(plan, h, w):
    """(B, h, w, 2) FP32: 2 F(q) - 1 at the source pixel centres q[i, j] = ((j + .5) / w, (i + .5) / h)."""
    b, dev = plan.gx.shape[0], plan.gx.device
    qx = ((torch.arange(w, dtype=F64, device=dev) + 0.5) / w).expand(b, -1)
    qy = ((torch.arange(h, dtype=F64, device=dev) + 0.5) / h).expand(b, -1)
    fx, fy = FV.warp(plan.gx, qx), FV.warp(plan.gy, qy)
    p = torch.stack([fx[:, None, :].expand(b, h, w), fy[:, :, None].expand(b, h, w)], -1).to(F32)
    return 2 * p - 1


def source_reference(plan, reference_points):
    """(B, Q, 1, 4) warped cxcywh -> (c_o, s_o), each (B, Q, 2) FP64: both box edges through G, then centre and size."""
    ref = reference_points[:, :, 0, :].to(F64)
    c, s = ref[..., :2], ref[..., 2:]
    lo, hi = c - s / 2, c + s / 2
    a = torch.stack([FV.unwarp(plan.gx, lo[..., 0]), FV.unwarp(plan.gy, lo[..., 1])], -1)
    b = torch.stack([FV.unwarp(plan.gx, hi[..., 0]), FV.unwarp(plan.gy, hi[..., 1])], -1)
    return (a + b) / 2, b - a


def source_points(attn, c_o, s_o, sampling_offsets):
    """(B, Q, heads, points, 2) FP64 source sampling points, in the native order of the factors."""
    scale = attn.num_points_scale.to(F64).unsqueeze(-1)
    offset = sampling_offsets.to(F64) * scale * s_o[:, :, None, None, :] * attn.offset_scale
    return c_o[:, :, None, None, :] + offset


def to_warped(plan, p):
    """F applied to points (B, ..., 2), FP64; differentiable in p."""
    shape = p.shape
    flat = p.reshape(shape[0], -1, 2)
    return torch.stack([FV.warp(plan.gx, flat[..., 0]), FV.warp(plan.gy, flat[..., 1])], -1).reshape(shape)


def native_locations(attn, query_dtype, reference_points, sampling_offsets):
    """The native four-dimensional sampling positions, expression for expression."""
    num_points_scale = attn.num_points_scale.to(dtype=query_dtype).unsqueeze(-1)
    offset = sampling_offsets * num_points_scale * reference_points[:, :, None, :, 2:] * attn.offset_scale
    return reference_points[:, :, None, :, :2] + offset


# ---------------------------------------------------------------- read operators
def _levels(value, shapes):
    """The frozen core's split of the value tensor: one (B * heads, C, h, w) tensor per level."""
    bs, _, n_head, c = value.shape
    parts = value.permute(0, 2, 3, 1).flatten(0, 1).split([h * w for h, w in shapes], dim=-1)
    return [parts[level].reshape(bs * n_head, c, h, w) for level, (h, w) in enumerate(shapes)]


def rebuild(reader, value, shapes):
    """Arm QC: the source-uniform memory U in the layout of `value` (B, tokens, heads, C), FP32, attached to V."""
    bs, _, n_head, c = value.shape
    out = []
    for level, ((h, w), value_l) in enumerate(zip(shapes, _levels(value, shapes))):
        grid = reader.grid(level, h, w)
        grid = grid[:, None].expand(bs, n_head, h, w, 2).reshape(bs * n_head, h, w, 2)
        out.append(DS.bilinear_grid_sample(value_l, grid).flatten(2))
    return torch.concat(out, dim=-1).reshape(bs, n_head, c, -1).permute(0, 3, 1, 2)


def _sample_points(value, shapes, locations, points):
    """Per level (B * heads, C, Q, points_l): the frozen core's sampling half, for the audit only."""
    grids = (2 * locations - 1).permute(0, 2, 1, 3, 4).flatten(0, 1).split(points, dim=-2)
    return [DS.bilinear_grid_sample(v, g) for v, g in zip(_levels(value, shapes), grids)]


def read(attn, reader, index, query_dtype, value, shapes, reference_points, sampling_offsets, attention_weights):
    plan, mode, core, points = reader.plan, reader.mode, attn.ms_deformable_attn_core, attn.num_points_list
    if mode == "native":
        return core(value, shapes, native_locations(attn, query_dtype, reference_points, sampling_offsets), attention_weights, points)
    audit = reader.audit is not None
    with torch.autocast(value.device.type, enabled=False):
        c_o, s_o = source_reference(plan, reference_points)
        p_o = source_points(attn, c_o, s_o, sampling_offsets)
        p_w = to_warped(plan, p_o) if (mode == "direct" or audit) else None
    identity = plan.identity
    mixed = bool(identity.any())
    native = native_locations(attn, query_dtype, reference_points, sampling_offsets) if (mixed or audit) else None
    if mixed:
        require(native.dtype == F32, f"R29 mixed identity batch expects FP32 native sampling positions, got {native.dtype}")
        keep = identity[:, None, None, None, None]
    u = None
    if mode == "direct":
        locations = p_w.to(F32)
        if mixed:
            locations = torch.where(keep, native, locations)
        out = core(value, shapes, locations, attention_weights, points)
    else:
        locations = p_o.to(F32)
        u = rebuild(reader, value, shapes)
        if mixed:
            locations = torch.where(keep, native, locations)
            u = torch.where(identity[:, None, None, None], value.to(u.dtype), u)
        out = core(u, shapes, locations, attention_weights, points)
    if audit:
        rec = audit_layer(attn, reader, index, value, u, shapes, reference_points, p_o, p_w, native, attention_weights)
        if locations.requires_grad:
            def grad_record(g, rec=rec):
                rec["location_gradient"] = {"finite": bool(torch.isfinite(g).all()), "abs_max": float(g.abs().max()),
                                            "abs_mean": float(g.abs().mean())}
            locations.register_hook(grad_record)
        reader.audit.append(rec)
    return out


def _attn_forward(self, query, reference_points, value, value_spatial_shapes, value_mask=None):
    reader = vars(self).get("r29_reader")
    require(isinstance(reader, Reader), "R29 cross-attention called without the read context of an R29 model forward")
    index = reader.calls
    reader.calls += 1
    if reader.plan is None:                     # identity bypass: the native layer, untouched
        return type(self)._r29_native.forward(self, query, reference_points, value, value_spatial_shapes, value_mask)
    require(reference_points.dim() == 4 and reference_points.shape[2] == 1 and reference_points.shape[-1] == 4,
            f"R29 read expects four-dimensional references (B, Q, 1, 4), got {tuple(reference_points.shape)}")
    # ---- the native layer up to the sampling positions, statement for statement
    bs, Len_q = query.shape[:2]
    Len_v = value.shape[1]

    value = self.value_proj(value)
    if value_mask is not None:
        value = value * value_mask.to(value.dtype).unsqueeze(-1)

    value = value.reshape(bs, Len_v, self.num_heads, self.head_dim)

    sampling_offsets: torch.Tensor = self.sampling_offsets(query)
    sampling_offsets = sampling_offsets.reshape(bs, Len_q, self.num_heads, sum(self.num_points_list), 2)

    attention_weights = self.attention_weights(query).reshape(bs, Len_q, self.num_heads, sum(self.num_points_list))
    attention_weights = F.softmax(attention_weights, dim=-1).reshape(bs, Len_q, self.num_heads, sum(self.num_points_list))
    # ---- the read
    output = read(self, reader, index, query.dtype, value, value_spatial_shapes, reference_points, sampling_offsets, attention_weights)

    output = self.output_proj(output)

    return output


# ---------------------------------------------------------------- audit of one read (no gradient, no model call)
def _classes(p, h, w):
    """interior (all four neighbours inside), edge (inside [0, 1]^2 but within half a pixel of the border), outside."""
    x, y = p[..., 0] * w - 0.5, p[..., 1] * h - 0.5
    outside = (p[..., 0] < 0) | (p[..., 0] > 1) | (p[..., 1] < 0) | (p[..., 1] > 1)
    interior = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
    return interior, ~outside & ~interior, outside


def _segment(g, v):
    n = g.shape[-1] - 1
    return (v * n).floor().clamp(0, n - 1)


def _mean(t):
    return float(t.to(F64).mean()) if t.numel() else None


def _max(t):
    return float(t.max()) if t.numel() else None


@torch.no_grad()
def audit_layer(attn, reader, index, value, u, shapes, reference_points, p_o, p_w, native, attention_weights):
    """Readings of one cross-attention call on tensors the forward already holds.  Per query group (ordinary / dn) and
    over the non-identity images: displacement of the P position from the native position of the same query, reference
    and offset (normalised and in feature pixels of the level), its attention-weighted mean, the share of references
    whose two edges lie in different linear segments of G, the border classes of the N, P and C positions, and the
    difference between the P read and the C read (the read of the other arm is computed here, detached)."""
    plan, points = reader.plan, attn.num_points_list
    bs, lq, heads, total, _ = p_o.shape
    dn = lq - reader.num_queries
    require(dn >= 0, f"R29 audit: {lq} queries, fewer than the {reader.num_queries} ordinary ones")
    groups = {"normal": slice(dn, lq), **({"dn": slice(0, dn)} if dn else {})}
    live = ~plan.identity
    p_o, p_w, nat, aw = p_o.detach(), p_w.detach(), native.detach().to(F64), attention_weights.detach().to(F64)
    level_of = torch.tensor([k for k, n in enumerate(points) for _ in range(n)], device=p_o.device)
    wh = torch.tensor([[w, h] for h, w in shapes], dtype=F64, device=p_o.device)[level_of]          # (points, 2)
    d = p_w - nat
    d_norm, d_px = d.norm(dim=-1), (d * wh).norm(dim=-1)
    weighted = (aw * d_px).sum(-1)
    ref = reference_points[:, :, 0, :].detach().to(F64)
    lo, hi = ref[..., :2] - ref[..., 2:] / 2, ref[..., :2] + ref[..., 2:] / 2
    cross_x = _segment(plan.gx, lo[..., 0]) != _segment(plan.gx, hi[..., 0])
    cross_y = _segment(plan.gy, lo[..., 1]) != _segment(plan.gy, hi[..., 1])
    sets = {"N": native.detach().to(F32), "P": p_w.to(F32), "C": p_o.to(F32)}
    # the two reads on the same value tensor (pre-attention samples), float32
    value = value.detach()
    u = rebuild(reader, value, shapes) if u is None else u.detach()
    s_p = _sample_points(value, shapes, sets["P"], points)
    s_c = _sample_points(u, shapes, sets["C"], points)
    att = aw.permute(0, 2, 1, 3).reshape(bs * heads, 1, lq, total).to(F32)
    o_p, o_c = (torch.concat(s_p, dim=-1) * att).sum(-1), (torch.concat(s_c, dim=-1) * att).sum(-1)     # (B * heads, C, Q)
    shape5 = lambda t: t.reshape(bs, heads, lq, -1).permute(0, 2, 1, 3)                                # -> (B, Q, heads, points_l)
    d2 = torch.concat([shape5(((a - b).to(F64) ** 2).sum(1)) for a, b in zip(s_p, s_c)], dim=-1)
    p2 = torch.concat([shape5((a.to(F64) ** 2).sum(1)) for a in s_p], dim=-1)
    od2 = ((o_p - o_c).to(F64) ** 2).sum(1).reshape(bs, heads, lq).permute(0, 2, 1)
    op2 = (o_p.to(F64) ** 2).sum(1).reshape(bs, heads, lq).permute(0, 2, 1)
    cls = {}
    for name, loc in sets.items():
        parts = [torch.stack(_classes(loc[..., level_of == k, :], h, w), 0) for k, (h, w) in enumerate(shapes)]
        cls[name] = torch.concat(parts, dim=-1)                                                         # (3, B, Q, heads, points)
    rel = lambda num, den: (float(torch.sqrt(num.sum() / den.sum())) if float(den.sum()) > 0 else None)
    rec = {"layer": index, "mode": reader.mode, "queries": lq, "dn_queries": dn, "images": bs, "identity_images": int(plan.identity.sum())}
    for name, sl in groups.items():
        m = live[:, None].expand(bs, lq)[:, sl]
        pick = lambda t: t[:, sl][m]
        g = {"queries": int(m.sum()),
             "reference_edges_in_different_segments": {"x": _mean(pick(cross_x)), "y": _mean(pick(cross_y)), "any": _mean(pick(cross_x | cross_y))},
             "displacement_normalised": {"mean": _mean(pick(d_norm)), "max": _max(pick(d_norm))},
             "displacement_feature_px": {"mean": _mean(pick(d_px)), "max": _max(pick(d_px)),
                                         "share_above_half_px": _mean(pick(d_px) > 0.5)},
             "attention_weighted_displacement_feature_px": {"mean": _mean(pick(weighted)), "max": _max(pick(weighted))},
             "levels": [], "border": {}, "read_difference_P_vs_C": {}}
        for k, (h, w) in enumerate(shapes):
            sel = pick(d_px)[..., level_of == k]
            g["levels"].append({"hw": [int(h), int(w)], "displacement_feature_px": {"mean": _mean(sel), "max": _max(sel)}})
        for which in sets:
            c3 = [pick(cls[which][i]) for i in range(3)]
            g["border"][which] = {"interior": _mean(c3[0]), "edge_half_pixel": _mean(c3[1]), "outside": _mean(c3[2])}
        c3 = [pick(cls["C"][i]) for i in range(3)]
        dd, pp = pick(d2), pick(p2)
        g["read_difference_P_vs_C"] = {
            "samples_relative_l2": rel(dd, pp),
            "by_border_class_of_the_source_point": {n_: {"points": int(c.sum()), "relative_l2": rel(dd[c], pp[c])}
                                                    for n_, c in zip(("interior", "edge_half_pixel", "outside"), c3)},
            "attention_output_relative_l2": rel(pick(od2), pick(op2))}
        rec[name] = g
    return rec


# ---------------------------------------------------------------- model forward
def _forward(self, x, targets=None):
    ctx, native, arm = self.r28_ctx, self._r29_native, self._r29_arm
    require(x.dim() == 4 and tuple(x.shape[-2:]) == H_HW and x.dtype == torch.float32,
            f"R29 expects the FP32 H input, got {tuple(x.shape)} {x.dtype}")
    h = vars(ctx).get("r28_fwhm")
    require(type(h) is float and h.hex() == FG.WIDTH_HEX, f"R29 context without the fixed kernel width of arm FG: {h!r}")
    mode = vars(ctx).get("r29_read")
    require(mode == READ[arm], f"R29 arm {arm} context carries the read mode {mode!r}")
    mark = vars(ctx).get("probe") or (lambda _name: None)
    queries = self.decoder.num_queries
    expected = len(self.decoder.decoder.layers) if self.training else self.decoder.decoder.eval_idx + 1
    with torch.autocast(x.device.type, enabled=False), torch.no_grad():
        d = ctx.demand(x)
        require(bool(torch.isfinite(d).all()) and bool((d >= 0).all()), "R29 demand map is not finite / non-negative")
        mark("demand")
        plan = FS.plan_from_demand(d, h).to(x.device)
        uniform = bool(plan.identity.all())
        mark("map")
        render = None
        if not uniform:
            xw, render = FV.resample(x, plan)
            tw = None if targets is None else FV.warp_targets(plan, targets)
        if ctx.observe and targets is not None:
            ctx.records.append(FS.observe(plan, targets, render, d))
    sink = vars(ctx).get("r29_audit")
    if uniform:
        with reading(self, Reader(mode, None, queries)) as reader:
            out = native.forward(self, to_low(x), targets)
        require(reader.calls == expected, f"R29 identity bypass: {reader.calls} cross-attention calls, expected {expected}")
        mark("detector")
        if sink is not None:
            sink.append({"identity_batch": True, "images": int(x.shape[0]), "layers": []})
        return out
    mark("render")
    with reading(self, Reader(mode, plan, queries, [] if sink is not None else None)) as reader:
        out = native.forward(self, xw, tw)
    require(reader.calls == expected, f"R29 read: {reader.calls} cross-attention calls, expected {expected}")
    mark("detector")
    with torch.autocast(x.device.type, enabled=False):
        res = FV.unwarp_outputs(plan, out)
    mark("unwarp")
    if sink is not None:
        sink.append({"identity_batch": False, "images": int(x.shape[0]), "layers": reader.audit})
    return res


def is_wrapped(model, arm=None):
    cls = type(model)
    a = cls.__dict__.get("_r29_arm")
    ctx = vars(model).get("r28_ctx")
    if not (cls.__dict__.get("_r29_native") is not None and cls.__dict__.get("forward") is _forward and "forward" not in vars(model)
            and a in ARMS and (arm is None or a == arm) and isinstance(ctx, R.Context) and not hasattr(model, "r27_ctx")):
        return False
    h = vars(ctx).get("r28_fwhm")
    mods = _attention_layers(model)
    return (type(h) is float and h.hex() == FG.WIDTH_HEX and vars(ctx).get("r29_read") == READ[a] and len(mods) == LAYERS
            and all(type(m).__dict__.get("forward") is _attn_forward and type(m).__dict__.get("_r29_native") is not None
                    and "forward" not in vars(m) and "r29_reader" not in vars(m) for m in mods))


def _check_attention(m, native):
    core = getattr(m, "ms_deformable_attn_core", None)
    require(type(m) is native and "forward" not in vars(m) and m.method == "default"
            and type(core) is functools.partial and core.func is DS.deformable_attention_core_func_v2 and core.args == ()
            and core.keywords == {"method": "default"} and list(m.num_points_list) == POINTS and m.num_levels == LEVELS
            and m.num_heads == HEADS and m.offset_scale == OFFSET_SCALE,
            "R29 expects the native three-level 4/4/4 deformable attention with the deterministic_gather core")


def swap(model, ctx, arm):
    """L0-wrapped native model -> arm QP / QC class, in place: the model class and the class of its three decoder
    cross-attention modules change; the module layout, every parameter, buffer and state dict key stay."""
    require(arm in ARMS, f"R29 arm: {arm}")
    require(L0.is_wrapped(model), "R29 swap expects an L0-wrapped native model")
    require(vars(ctx).get("r29_read") == READ[arm], f"R29 swap: the context read mode is not the one of arm {arm}")
    native = type(model)._l0_native
    before = L0._layout(model)
    mods = _attention_layers(model)
    attn_native = type(mods[0])
    require(len(mods) == LAYERS and attn_native.__name__ == "MSDeformableAttention" and "_r29_native" not in attn_native.__dict__,
            "R29 swap: unexpected decoder cross-attention modules")
    for m in mods:
        _check_attention(m, attn_native)
    if attn_native not in _ATTN_TYPES:
        _ATTN_TYPES[attn_native] = type("R29ReadAttention", (attn_native,), {"forward": _attn_forward, "_r29_native": attn_native})
    if (native, arm) not in _TYPES:
        _TYPES[(native, arm)] = type(f"R29Unzoom{SHORT[arm]}RTDETR", (native,), {"forward": _forward, "_r29_native": native, "_r29_arm": arm})
    for m in mods:
        m.__class__ = _ATTN_TYPES[attn_native]
    model.__class__ = _TYPES[(native, arm)]
    vars(model)["r28_ctx"] = ctx
    require(L0._layout(model) == before and is_wrapped(model, arm) and not L0.is_wrapped(model) and not R.is_wrapped(model)
            and not FV.is_wrapped(model) and not FS.is_wrapped(model) and not FG.is_wrapped(model), "R29 swap changed the module layout")


# ---------------------------------------------------------------- source / mapping (arm FG's, unchanged)
def mapping(source):
    return FG.mapping(source)


def check_source(arm, source, check):
    require(arm in ARMS, f"R29 arm: {arm}")
    return FG.check_source(BASE_ARM, source, check)


def _context(source, binding, arm):
    ctx = FG._context(source, binding)
    vars(ctx)["r29_read"] = READ[arm]
    return ctx


# ---------------------------------------------------------------- engine / evaluation model
def create(parent, device, arm, source, *, fixture=False, activation_checkpoint=True, check=False):
    """Arm FG's construction order with the R29 swap: the demand context first (its network initialiser draws from the
    torch RNG), then L0.create, which restores the parent training RNG as its last step."""
    binding = check_source(arm, source, check)
    ctx = _context(source, binding, arm)
    e, checks = L0.create(parent, device, fixture=fixture, activation_checkpoint=activation_checkpoint)
    ctx.to(e.device)
    swap(e.model, ctx, arm)
    swap(e.ema.module, ctx, arm)
    del e.l0_arm
    e.r28_arm = e.r25_arm = arm
    e.r28_ctx = ctx
    e.r28_mapping = binding
    e.r29_read = READ[arm]
    e.r28_check = bool(check)
    checks = {**checks, "arm": arm, "short_name": SHORT[arm], "read_mode": READ[arm], "r25_arm": arm,
              "identity_sha256": IDENTITY_SHA256[arm], "base_identity_sha256": FG.IDENTITY_SHA256[BASE_ARM],
              "L0_identity_sha256": L0.IDENTITY_SHA256, "demand": ctx.binding(), "mapping": binding,
              "wrapped": {"online": is_wrapped(e.model, arm), "EMA": is_wrapped(e.ema.module, arm)},
              "demand_network_parameters": 0 if ctx.net is None else sum(p.numel() for p in ctx.net.parameters()),
              "mapping_learnable_parameters": 0, "read_learnable_parameters": 0,
              "detector_state_keys_equal_parent": checks["model_keys_equal_parent"] and checks["EMA_keys_equal_parent"]}
    require(all(checks["wrapped"].values()) and checks["new_parameters"] == 0, ("R29 bridge mismatch", checks))
    return e, checks


def build_eval_model(state, device, arm, source, *, check=False):
    """FP32 evaluation model from a saved module state (native key set) with arm FG's demand source and fixed width."""
    binding = check_source(arm, source, check)
    ctx = _context(source, binding, arm).to(device)
    model = L0.build_eval_model(state, device)
    swap(model, ctx, arm)
    model.r28_arm = arm
    model.r28_mapping = binding
    model.r29_read = READ[arm]
    return model
