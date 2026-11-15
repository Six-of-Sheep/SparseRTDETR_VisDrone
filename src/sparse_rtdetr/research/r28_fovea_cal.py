"""R28 arm FS ("F*"): arm F with one calibrated global attraction-kernel width (TASK_R28_CALIBRATED_F_CONTROL_006).

Arm FS is arm F (r28_fovea, unchanged and not edited by this module) with a single scientific difference: the FWHM of
the Gaussian attraction kernel is one global scalar h* (the same on both axes) instead of the released value 4.  h* is
fixed once, before any training of this arm, by a geometric calibration on the training set (section 2 of the task):
the pooled mean log magnification of small training GT under the F rule with width h is matched to the one of arm W
on the same demand maps and the same GT.  No detector prediction and no AP enters the calibration.  At deployment the
width is a constant; nothing is fitted per image.

Unchanged from arm F: the frozen demand network and its path, eps = 0.2, marginal first and sampling second, 31 x 51
nodes, reflect padding by 30 nodes, kernel support r = -30..30, the kernel-weighted mean of the padded node
coordinates, the official clamp, the piecewise linear G with its exact inverse, linear extension outside [0, 1], the
complete tent support, source-frame losses.  Nothing sorts, accumulates a maximum, bounds a slope, rescales the end
points or adds a gain to reach the target: a map that is not finite and strictly increasing raises.

The kernel cache is keyed by (nodes, padding, h as the FP64 hex string, dtype), never by the node count alone.

This control was designed after the W, K and F results were seen; it is an exploratory control, not a preregistered
test of arm W and not a reproduction of the official FOVEA method.
"""
from __future__ import annotations

import hashlib
import json
import math

import torch

from . import r27_l0_control as L0
from . import r28_fovea as FV
from . import r28_resample as R
from . import r28_warp as W
from .r27_local_memory import H_HW, L_HW, PARENT_SHA, PARENT_UPDATE, require, to_low

VERSION = "R28-006-calibrated-fovea-control-v1"
ARM = "FS"
ARMS = (ARM,)
FORMAL_KIND = {ARM: "network"}
CHECK_KINDS = R.CHECK_KINDS
GRID_SHAPE = FV.GRID_SHAPE
PAD = FV.PAD
BOX_FAMILIES = R.BOX_FAMILIES
OUTPUT_KEYS = R.OUTPUT_KEYS
F64 = torch.float64
H_MIN, H_MAX = 4.0, 30.0
GRID_POINTS = 65
EXTRA_WIDTHS = (13.0,)                 # the constructor default of the official generator
TOLERANCE = 0.005
BISECTIONS = 16
END_TOLERANCE = 1e-12                  # arm F's numerical convention for the anti-crop end nodes and the clamp amount
STATUSES = ("matched", "calibration_invalid", "calibration_unmatched", "confirmation_invalid")
CALIBRATION = {
    "task": "TASK_R28_CALIBRATED_F_CONTROL_006 section 2",
    "data": {"epoch": 61, "windows": 304, "occurrences": 4864,
             "source": "the formal training loader (train_core, seed 0, production augmentation and audit); every "
                       "window is compared with the pinned R27 formal-P input ledger before it is used"},
    "confirmation": {"epoch": 118, "windows": 32, "occurrences": 512,
                     "use": "validity of the locked width and the reported error only; the width is never re-fitted"},
    "demand": "the frozen demand network of arm W, one forward per image occurrence on the production FP32 path "
              "(microbatches of 8 in window order, eval mode, no gradient, autocast disabled); cached with the "
              "transformed GT, raw areas and the window identities; no detector forward",
    "small": "training targets delivered by the loader with raw_area <= 1024 (the production small class on the "
             "original image area); a box of non-positive size is refused by the training engine, so a window that "
             "contains one stops the calibration",
    "g": "sqrt((warped_width / source_width) * (warped_height / source_height)) with each arm's own exact box map",
    "statistic": "L = sum(log g) / N_small pooled over every valid small observation of the calibration windows",
    "target": "L_W, recomputed for arm W on the same demand maps and the same GT (not the historical 1.267, which "
              "averages per-window geometric means)",
    "widths": {"lo": H_MIN, "hi": H_MAX, "points": GRID_POINTS, "extra": list(EXTRA_WIDTHS),
               "note": "65 equally spaced widths and the official constructor default, sorted, duplicates removed"},
    "validity": "a width is valid only if on every calibration image and both axes the nodes are finite and strictly "
                "increasing, the raw end nodes are within 1e-12 of 0 and 1, and the official clamp removes at most "
                "1e-12 and only at the two end nodes; an invalid width is never repaired and never used to "
                "interpolate across it",
    "tolerance": TOLERANCE, "match": "abs(exp(L_h - L_W) - 1) <= tolerance",
    "selection": [
        "if valid widths of the table match, take the smallest of them",
        "otherwise take the first pair of neighbours of the sorted table that are both valid and whose L_h - L_W have "
        "opposite signs; bisect that interval (at most 16 midpoints, all evaluated unless a midpoint is exact or the "
        "interval can no longer be split in FP64) and take the midpoint with the smallest absolute error, the "
        "smaller width on a tie; an invalid midpoint ends the calibration as calibration_invalid",
        "if no width is valid: calibration_invalid; if valid widths exist but no such pair: the valid width with the "
        "smallest absolute error is reported and, being outside the tolerance, the result is calibration_unmatched",
        "a result outside the tolerance is calibration_unmatched; no second interval, wider range, other target or "
        "extra parameter is tried",
    ],
    "bisections": BISECTIONS, "statuses": list(STATUSES),
    "lock": "h* is stored as its FP64 hex string in CALIBRATION.json together with every reading; the formal engine, "
            "every saved state, the evaluation and the timing are bound to that file",
}
DEFINITION = {
    **{k: v for k, v in FV.DEFINITION.items() if k not in ("name", "attraction_fwhm", "kernel", "differences_from_official")},
    "name": "r28_fovea_style_separable_map_calibrated_width_v1",
    "base": {"name": FV.DEFINITION["name"], "module": "r28_fovea (arm F, unchanged)", "identity_sha256": FV.IDENTITY_SHA256["F"]},
    "attraction_fwhm": "one global scalar h* in nodes, the same on both axes, read from the round's locked "
                       "CALIBRATION.json; a constant at deployment",
    "kernel": "k(r) = exp(-4 log(2) r^2 / h*^2), r = -30..30 nodes",
    "kernel_cache_key": "(nodes, padding, h as FP64 hex, dtype)",
    "only_difference_from_arm_F": "attraction_fwhm 4 -> h*",
    "calibration": CALIBRATION,
    "differences_from_official": [
        "the kernel width is one scalar calibrated on the training set so that the pooled mean log magnification of "
        "small GT matches arm W's; neither the released value 4 nor the constructor default 13 is used as such; the "
        "calibration was designed after the F result was seen",
        *[d for d in FV.DEFINITION["differences_from_official"] if not d.startswith("attraction_fwhm")],
    ],
    "not_matched": "the mean small magnification is matched on the training set at the start of the schedule; the "
                   "per-instance distribution, position, anisotropy, tails, medium / large objects, gradients and "
                   "sampling footprints are not matched",
}


def _identity():
    base = FV.IDENTITY["F"]
    return {
        **{k: v for k, v in base.items() if k not in ("version", "arm", "task", "definition", "is_not")},
        "version": VERSION, "arm": ARM, "task": "TASK_R28_CALIBRATED_F_CONTROL_006", "definition": DEFINITION,
        "mapping_learnable_parameters": 0, "mapping_calibrated_scalars": 1,
        "is_not": {**base["is_not"],
                   "F": "arm F uses the released width 4; this arm uses the calibrated width h* and its states carry "
                        "the calibration binding",
                   "preregistered": "designed after the W, K and F results; an exploratory control"},
    }


IDENTITY = {ARM: _identity()}
IDENTITY_SHA256 = {ARM: hashlib.sha256(json.dumps(IDENTITY[ARM], sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
_TYPES, _KERNELS = {}, {}


# ---------------------------------------------------------------- width and kernel
def check_width(h):
    require(type(h) is float and math.isfinite(h) and H_MIN <= h <= H_MAX, f"R28-FS kernel width must be a float in [{H_MIN}, {H_MAX}]: {h!r}")
    return h


def kernel_key(n, h):
    return (int(n), int(PAD), check_width(h).hex(), "float64")


def kernel_matrix(n, h):
    """(n, n + 2 PAD) FP64: row i holds k(i - j) for the padded positions j (zero where |i - j| > PAD)."""
    key = kernel_key(n, h)
    if key not in _KERNELS:
        i = torch.arange(n, dtype=F64)[:, None]
        j = torch.arange(-PAD, n + PAD, dtype=F64)[None, :]
        r = i - j
        k = torch.exp(-4.0 * math.log(2.0) * r * r / (h * h))
        _KERNELS[key] = torch.where(r.abs() <= PAD, k, torch.zeros_like(k))
    return _KERNELS[key]


def attraction_nodes(s, h):
    """(B, n) positive node saliency -> (raw, G), FP64 on the CPU: arm F's rule with the kernel of width h."""
    s = s.to(F64).cpu()
    n = s.shape[1]
    require(n > PAD, "reflect padding needs more nodes than the padding size")
    sp = s[:, FV.reflect_index(n)]
    p = torch.arange(-PAD, n + PAD, dtype=F64) / (n - 1)
    m = kernel_matrix(n, h)
    num = (m[None] * (sp * p)[:, None, :]).sum(-1)
    den = (m[None] * sp[:, None, :]).sum(-1)
    raw = num / den
    return raw, (torch.clamp(raw * 2 - 1, min=-1, max=1) + 1) / 2


def nodes_from_demand(demand, h):
    """(B, GH, GW) demand without eps -> ({axis: (raw, G)}, identity), FP64 on the CPU.  Nothing is checked or
    repaired here; an image whose demand is constant by arm W's frozen test gets the exact uniform nodes."""
    d = demand.detach().to(F64).cpu() + W.EPS
    marg = {"y": d.sum(2), "x": d.sum(1)}
    identity = FV._constant(W.interp_marginal(marg["y"], L_HW[0])) & FV._constant(W.interp_marginal(marg["x"], L_HW[1]))
    out = {}
    for axis, n in (("y", GRID_SHAPE[0]), ("x", GRID_SHAPE[1])):
        raw, g = attraction_nodes(FV.node_saliency(marg[axis], n), h)
        uniform = torch.arange(n, dtype=F64) / (n - 1)
        out[axis] = (raw, torch.where(identity[:, None], uniform[None].expand_as(g), g))
    return out, identity


def node_readings(raw, g, identity):
    """Per-image readings of one axis: finite / strictly increasing flags and arm F's end-node quantities."""
    step = g[:, 1:] - g[:, :-1]
    finite = torch.isfinite(g).all(1) & torch.isfinite(raw).all(1)
    increasing = (step > 0).all(1)
    over = (raw - raw.clamp(0, 1)).abs()
    live = ~identity
    end = torch.maximum(raw[:, 0].abs(), (raw[:, -1] - 1).abs())
    return {"finite": finite, "increasing": increasing, "step": step, "over": over, "live": live, "end": end}


def valid_images(nodes, identity):
    """(B,) bool: the calibration validity of every image (both axes), and the per-image failure flags."""
    ok = torch.ones(len(identity), dtype=torch.bool)
    flags = {"not_finite": ~ok, "not_strictly_increasing": ~ok, "end_nodes": ~ok, "clamp": ~ok}
    for axis in ("y", "x"):
        r = node_readings(*nodes[axis], identity)
        nan_safe = torch.nan_to_num(r["end"], nan=math.inf)
        over = torch.nan_to_num(r["over"], nan=math.inf)
        flags["not_finite"] = flags["not_finite"] | ~r["finite"]
        flags["not_strictly_increasing"] = flags["not_strictly_increasing"] | ~r["increasing"]
        flags["end_nodes"] = flags["end_nodes"] | (r["live"] & (nan_safe > END_TOLERANCE))
        flags["clamp"] = flags["clamp"] | (r["live"] & ((over.max(1).values > END_TOLERANCE) | (over[:, 1:-1] > 0).any(1)))
    for v in flags.values():
        ok = ok & ~v
    return ok, flags


def plan_from_demand(demand, h):
    """(B, GH, GW) demand without eps -> r28_fovea.Plan (tensors on the CPU) for the kernel width h.  The checks and
    the readings are arm F's; with h = 4.0 the result equals r28_fovea.plan_from_demand bit for bit."""
    nodes, identity = nodes_from_demand(demand, h)
    stats = {}
    for axis, n in (("y", GRID_SHAPE[0]), ("x", GRID_SHAPE[1])):
        raw, g = nodes[axis]
        step = g[:, 1:] - g[:, :-1]
        if not (bool(torch.isfinite(g).all()) and bool((step > 0).all())):
            bad = (~torch.isfinite(g)).any(1) | (step <= 0).any(1)
            raise RuntimeError("R28-FS map is not finite and strictly increasing on axis %s (width %s): images %s, smallest node step %.3e"
                               % (axis, h.hex(), bad.nonzero().flatten().tolist(), float(step.min())))
        live = ~identity
        over = (raw - raw.clamp(0, 1)).abs()[live]
        stats[axis] = {"node_slope_min": float(step.min()) * (n - 1), "node_slope_max": float(step.max()) * (n - 1),
                       "endpoint_error_max": float(torch.maximum(raw[:, 0].abs(), (raw[:, -1] - 1).abs())[live].max()) if bool(live.any()) else 0.0,
                       "clamped_nodes": int((over > 0).sum()), "clamp_max": float(over.max()) if bool(live.any()) else 0.0,
                       "clamped_interior_nodes": int((over[:, 1:-1] > 0).sum()),
                       "first_node_max": float(g[:, 0].max()), "last_node_min": float(g[:, -1].min())}
    return FV.Plan(nodes["x"][1], nodes["y"][1], identity, stats)


# ---------------------------------------------------------------- calibration rule (pure)
def candidates():
    """The sorted width table: 65 equally spaced widths on [4, 30] and the official constructor default."""
    grid = [H_MIN + (H_MAX - H_MIN) * k / (GRID_POINTS - 1) for k in range(GRID_POINTS)]
    return sorted(set(grid) | set(EXTRA_WIDTHS))


def relative_error(l_h, l_w):
    return math.expm1(l_h - l_w)


def select(evaluate, l_w):
    """The fixed selection rule.  evaluate(h) -> (valid, L_h or None, detail dict).  Returns the complete record:
    every width read (table and midpoints) in reading order, the interval, the status and the locked width."""
    rows = []

    def read(h, stage):
        valid, l_h, detail = evaluate(h)
        require(bool(valid) == (l_h is not None), "R28-FS calibration: a reading has a value exactly when it is valid")
        row = {"h": h, "h_hex": h.hex(), "stage": stage, "valid": bool(valid), "L": l_h,
               "relative_error": relative_error(l_h, l_w) if valid else None, **detail}
        rows.append(row)
        return row

    table = [read(h, "table") for h in candidates()]
    valid = [r for r in table if r["valid"]]
    out = {"L_W": l_w, "tolerance": TOLERANCE, "readings": rows, "interval": None, "h_star": None, "h_star_hex": None,
           "L_h_star": None, "relative_error": None, "rule_applied": None}

    def lock(row, rule):
        out.update(status="matched", h_star=row["h"], h_star_hex=row["h_hex"], L_h_star=row["L"],
                   relative_error=row["relative_error"], rule_applied=rule)
        return out

    def closest(points):
        return min(points, key=lambda r: (abs(r["relative_error"]), r["h"]))

    if not valid:
        return {**out, "status": "calibration_invalid", "rule_applied": "no valid width in the table"}
    hits = [r for r in valid if abs(r["relative_error"]) <= TOLERANCE]
    if hits:
        return lock(min(hits, key=lambda r: r["h"]), "smallest valid width of the table within the tolerance")
    pair = next(((a, b) for a, b in zip(table, table[1:])
                 if a["valid"] and b["valid"] and (a["relative_error"] < 0) != (b["relative_error"] < 0)), None)
    if pair is None:
        best = closest(valid)
        return {**out, "status": "calibration_unmatched", "rule_applied": "valid widths without a sign change between valid neighbours",
                "closest": {k: best[k] for k in ("h", "h_hex", "L", "relative_error")}}
    lo, hi = pair
    out["interval"] = {"lo": lo["h"], "hi": hi["h"], "lo_hex": lo["h_hex"], "hi_hex": hi["h_hex"],
                       "lo_relative_error": lo["relative_error"], "hi_relative_error": hi["relative_error"]}
    points = []
    for k in range(BISECTIONS):
        mid = (lo["h"] + hi["h"]) / 2
        if not lo["h"] < mid < hi["h"]:
            break
        row = read(mid, f"bisection-{k + 1}")
        if not row["valid"]:
            return {**out, "status": "calibration_invalid", "rule_applied": "an invalid map inside the bisection interval"}
        points.append(row)
        if row["relative_error"] == 0.0:
            break
        if (row["relative_error"] < 0) == (lo["relative_error"] < 0):
            lo = row
        else:
            hi = row
    require(bool(points), "R28-FS calibration: the interval between two table widths could not be split")
    best = closest(points)
    if abs(best["relative_error"]) <= TOLERANCE:
        return lock(best, "bisection midpoint with the smallest absolute error")
    return {**out, "status": "calibration_unmatched", "rule_applied": "bisection did not reach the tolerance",
            "closest": {k: best[k] for k in ("h", "h_hex", "L", "relative_error")}}


# ---------------------------------------------------------------- forward
def mapping(source):
    """The mapping binding of a demand source: the width (value and FP64 hex) and the calibration file reference."""
    h = check_width(source.get("fwhm"))
    return {"fwhm": h, "fwhm_hex": h.hex(), "calibration": source.get("calibration")}


def observe(plan, targets, render, demand):
    """Arm F's training record plus what the calibration description needs: the demand map already produced by this
    forward, the padded source boxes and the per-axis magnification.  No model call, no second demand forward."""
    rec = FV.observe(plan, targets, render)
    pad, _ = R._padded_xyxy(targets)
    ox, oy = FV.magnification(plan, pad)
    rec.update(demand=demand.detach().to(F64).cpu(), xyxy=pad.cpu(), ox=ox.cpu(), oy=oy.cpu())
    return rec


def _forward(self, x, targets=None):
    ctx, native = self.r28_ctx, self._r28fs_native
    require(x.dim() == 4 and tuple(x.shape[-2:]) == H_HW and x.dtype == torch.float32,
            f"R28-FS expects the FP32 H input, got {tuple(x.shape)} {x.dtype}")
    h = vars(ctx).get("r28_fwhm")
    require(type(h) is float, "R28-FS context without a kernel width")
    mark = vars(ctx).get("probe") or (lambda _name: None)
    with torch.autocast(x.device.type, enabled=False), torch.no_grad():
        d = ctx.demand(x)
        require(bool(torch.isfinite(d).all()) and bool((d >= 0).all()), "R28-FS demand map is not finite / non-negative")
        mark("demand")
        plan = plan_from_demand(d, h).to(x.device)
        uniform = bool(plan.identity.all())
        mark("map")
        render = None
        if not uniform:
            xw, render = FV.resample(x, plan)
            tw = None if targets is None else FV.warp_targets(plan, targets)
        if ctx.observe and targets is not None:
            ctx.records.append(observe(plan, targets, render, d))
    if uniform:
        out = native.forward(self, to_low(x), targets)
        mark("detector")
        return out
    mark("render")
    out = native.forward(self, xw, tw)
    mark("detector")
    with torch.autocast(x.device.type, enabled=False):
        res = FV.unwarp_outputs(plan, out)
    mark("unwarp")
    return res


def is_wrapped(model):
    cls = type(model)
    ctx = vars(model).get("r28_ctx")
    return (cls.__dict__.get("_r28fs_native") is not None and cls.__dict__.get("forward") is _forward
            and "forward" not in vars(model) and isinstance(ctx, R.Context) and type(vars(ctx).get("r28_fwhm")) is float
            and not hasattr(model, "r27_ctx"))


def swap(model, ctx):
    """L0-wrapped native model -> arm FS class, in place; the module layout and the state dict keys do not change."""
    require(L0.is_wrapped(model), "R28-FS swap expects an L0-wrapped native model")
    native = type(model)._l0_native
    before = L0._layout(model)
    if native not in _TYPES:
        _TYPES[native] = type("R28FoveaCalRTDETR", (native,), {"forward": _forward, "_r28fs_native": native})
    model.__class__ = _TYPES[native]
    vars(model)["r28_ctx"] = ctx
    require(L0._layout(model) == before and is_wrapped(model) and not L0.is_wrapped(model) and not R.is_wrapped(model)
            and not FV.is_wrapped(model), "R28-FS swap changed the module layout")


def check_source(arm, source, check):
    require(arm == ARM, f"R28-FS arm: {arm}")
    kind = source["kind"]
    require(kind == FORMAL_KIND[arm] or (check and kind in CHECK_KINDS),
            f"R28 arm {arm} cannot use a '{kind}' demand source outside a check")
    binding = mapping(source)
    cal = binding["calibration"]
    require(check or (isinstance(cal, dict) and set(cal) >= {"path", "bytes", "sha256"}),
            "R28 arm FS needs the locked calibration file reference outside a check")
    return binding


def _context(source, binding):
    ctx = R.Context({k: source[k] for k in ("kind", "file", "seed") if k in source})
    vars(ctx)["r28_fwhm"] = binding["fwhm"]
    return ctx


# ---------------------------------------------------------------- engine / evaluation model
def create(parent, device, arm, source, *, fixture=False, activation_checkpoint=True, check=False):
    """Arm F's construction with the calibrated width: the demand context first (its network initialiser draws from
    the torch RNG), then L0.create, which restores the parent training RNG as its last step."""
    binding = check_source(arm, source, check)
    ctx = _context(source, binding)
    e, checks = L0.create(parent, device, fixture=fixture, activation_checkpoint=activation_checkpoint)
    ctx.to(e.device)
    swap(e.model, ctx)
    swap(e.ema.module, ctx)
    del e.l0_arm
    e.r28_arm = e.r25_arm = arm
    e.r28_ctx = ctx
    e.r28_mapping = binding
    e.r28_check = bool(check)
    checks = {**checks, "arm": arm, "r25_arm": arm, "identity_sha256": IDENTITY_SHA256[arm], "L0_identity_sha256":
              L0.IDENTITY_SHA256, "demand": ctx.binding(), "mapping": binding,
              "wrapped": {"online": is_wrapped(e.model), "EMA": is_wrapped(e.ema.module)},
              "demand_network_parameters": 0 if ctx.net is None else sum(p.numel() for p in ctx.net.parameters()),
              "mapping_learnable_parameters": 0,
              "detector_state_keys_equal_parent": checks["model_keys_equal_parent"] and checks["EMA_keys_equal_parent"]}
    require(all(checks["wrapped"].values()) and checks["new_parameters"] == 0, ("R28-FS bridge mismatch", checks))
    return e, checks


def build_eval_model(state, device, arm, source, *, check=False):
    """FP32 evaluation model from a saved module state (native key set) with the arm's demand source and width."""
    binding = check_source(arm, source, check)
    ctx = _context(source, binding).to(device)
    model = L0.build_eval_model(state, device)
    swap(model, ctx)
    model.r28_arm = arm
    model.r28_mapping = binding
    return model
