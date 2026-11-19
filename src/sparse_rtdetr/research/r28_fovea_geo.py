"""R28 arm FG ("F_geo"): arm F with one fixed, geometry-selected attraction-kernel width (TASK_R28_STRONG_F_CONTROL_009).

Arm FG is arm F (r28_fovea, unchanged) with a single scientific difference: the FWHM of the Gaussian attraction
kernel is the constant 19.03125 nodes (FP64 hex 0x1.3080000000000p+4, the same on both axes) instead of the released
value 4.  The forward is r28_fovea_cal._forward itself, called with that constant: the demand network and its path,
eps = 0.2, 31 x 51 nodes, reflect padding by 30 nodes, the attraction formula, the clamp, the piecewise linear G with
its exact inverse, the linear extension, the complete tent support and the source-frame losses are the code arm F and
arm FS run.  Nothing sorts, accumulates a maximum, bounds a slope or adds a gain; a map that is not finite and strictly
increasing raises.

Where the width comes from: task 006 read the pooled mean log magnification of small training GT (epoch 61, 4,864
image occurrences, 137,229 small observations) for 66 fixed widths.  None matched arm W within the registered
tolerance, so that calibration ended as calibration_unmatched with h_star = null and no F* exists.  19.03125 is the
width of that table with the largest pooled small magnification, which is also the one closest to arm W (-4.98 %).
No detection result of any width was used.  This arm is therefore
  * not the equal-magnitude control F* (arm FS keeps its matched-calibration guard and cannot be built from this),
  * not an AP-selected width and not the optimum over a continuous range of widths,
  * not a reproduction of the official FOVEA method.
It is an exploratory control designed after the W, K and F results and the 006 geometry readings were seen.
"""
from __future__ import annotations

import hashlib
import json

from . import r27_l0_control as L0
from . import r28_fovea as FV
from . import r28_fovea_cal as FS
from . import r28_resample as R
from .r27_local_memory import require

VERSION = "R28-009-fixed-geometry-selected-width-fovea-control-v1"
ARM = "FG"
ARMS = (ARM,)
FORMAL_KIND = {ARM: "network"}
CHECK_KINDS = R.CHECK_KINDS
GRID_SHAPE = FS.GRID_SHAPE
PAD = FS.PAD
WIDTH_HEX = "0x1.3080000000000p+4"
WIDTH = float.fromhex(WIDTH_HEX)        # 19.03125 nodes
SELECTION_KIND = "fixed_training_geometry_selected_width"
SELECTION = {
    "kind": SELECTION_KIND, "task": "TASK_R28_STRONG_F_CONTROL_009 section 1",
    "fwhm": WIDTH, "fwhm_hex": WIDTH_HEX,
    "source": {"task": "TASK_R28_CALIBRATED_F_CONTROL_006", "commit": "f376dc973d8333252d4cf50c9ed37d4fd8b9d244",
               "release_manifest_sha256": "07247d2d2a19fca7daa14273617f178051f0c9e4a94ccd1effcb2000d8540cd3",
               "calibration_sha256": "8dcfcd5cf2cbd7ef96a54b1832d9eabd6923a5b7da860a9e634784ed0fa05c59",
               "calibration_status": "calibration_unmatched", "h_star": None,
               "table": "66 widths (65 equally spaced on [4, 30] and 13), epoch 61, 4,864 occurrences, 137,229 small GT"},
    "rule": "the width of the 006 table with the largest pooled mean log magnification of small training GT, which is "
            "the table width closest to arm W's; read from stored geometry only",
    "reading": {"W_pooled_small_magnification": 1.2682159020223975, "FG_pooled_small_magnification": 1.205055951488523,
                "FG_over_W_minus_1": -0.04980220673242978, "tolerance_of_006": 0.005, "matched": False},
    "not_used": "no detector prediction and no AP of any width; no new width, range, target or parameter was read",
    "scope": "the largest of 66 measured widths is not a bound for a continuous range, another demand map, another "
             "parameterisation or the FOVEA method family",
}
DEFINITION = {
    **{k: v for k, v in FV.DEFINITION.items() if k not in ("name", "attraction_fwhm", "kernel", "differences_from_official")},
    "name": "r28_fovea_style_separable_map_fixed_geometry_selected_width_v1",
    "base": {"name": FV.DEFINITION["name"], "module": "r28_fovea (arm F, unchanged)", "identity_sha256": FV.IDENTITY_SHA256["F"]},
    "forward": "r28_fovea_cal._forward with the constant width (arm FS's code path; arm FS's module is unchanged)",
    "attraction_fwhm": f"the constant {WIDTH!r} nodes ({WIDTH_HEX}), the same on both axes",
    "kernel": f"k(r) = exp(-4 log(2) r^2 / {WIDTH!r}^2), r = -30..30 nodes",
    "kernel_cache_key": "(nodes, padding, width as FP64 hex, dtype)",
    "only_difference_from_arm_F": f"attraction_fwhm 4 -> {WIDTH!r}",
    "selection": SELECTION,
    "differences_from_official": [
        "the kernel width is the fixed value selected on stored training-set geometry of task 006 (largest pooled small "
        "magnification of the measured table); neither the released value 4 nor the constructor default 13 is used; the "
        "selection was made after the W, K and F results were seen",
        *[d for d in FV.DEFINITION["differences_from_official"] if not d.startswith("attraction_fwhm")],
    ],
    "not_matched": "the mean small magnification is not matched to arm W (about -5 % on epoch 61); the per-instance "
                   "distribution, position, anisotropy, tails, medium / large objects, gradients and sampling footprints "
                   "are not matched either",
}


def _identity():
    base = FV.IDENTITY["F"]
    return {
        **{k: v for k, v in base.items() if k not in ("version", "arm", "task", "definition", "is_not")},
        "version": VERSION, "arm": ARM, "task": "TASK_R28_STRONG_F_CONTROL_009", "definition": DEFINITION,
        "mapping_learnable_parameters": 0, "mapping_calibrated_scalars": 0, "mapping_fixed_scalars": 1,
        "is_not": {**base["is_not"],
                   "F": "arm F uses the released width 4; this arm uses the fixed width 19.03125 and its states carry the "
                        "selection binding",
                   "FS": "arm FS (F*) needs a matched calibration; task 006 ended calibration_unmatched with h_star = null, "
                         "so no F* exists and this arm is not one",
                   "equal_magnitude_control": "the pooled small magnification stays about 5 % below arm W's",
                   "AP_selected": "no detection result of any width entered the choice",
                   "preregistered": "designed after the W, K and F results and the 006 geometry readings; exploratory"},
    }


IDENTITY = {ARM: _identity()}
IDENTITY_SHA256 = {ARM: hashlib.sha256(json.dumps(IDENTITY[ARM], sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
_TYPES = {}


# ---------------------------------------------------------------- width, map
def check_width(h):
    require(type(h) is float and h.hex() == WIDTH_HEX, f"R28-FG kernel width is the constant {WIDTH_HEX}: {h!r}")
    return h


def plan_from_demand(demand):
    """(B, GH, GW) demand without eps -> r28_fovea.Plan: arm FS's map function at the fixed width."""
    return FS.plan_from_demand(demand, WIDTH)


def mapping(source):
    """The mapping binding of a demand source: the fixed width (value and FP64 hex) and the selection reference."""
    h = check_width(source.get("fwhm"))
    return {"fwhm": h, "fwhm_hex": h.hex(), "selection": source.get("selection")}


def check_selection(sel):
    require(isinstance(sel, dict) and sel.get("kind") == SELECTION_KIND and sel.get("fwhm_hex") == WIDTH_HEX
            and sel.get("calibration_status") == "calibration_unmatched" and sel.get("h_star") is None
            and isinstance(sel.get("calibration"), dict) and set(sel["calibration"]) >= {"path", "bytes", "sha256"}
            and sel["calibration"]["sha256"] == SELECTION["source"]["calibration_sha256"],
            "R28 arm FG needs the selection reference to the unmatched 006 calibration outside a check")
    return sel


def check_source(arm, source, check):
    require(arm == ARM, f"R28-FG arm: {arm}")
    kind = source["kind"]
    require(kind == FORMAL_KIND[arm] or (check and kind in CHECK_KINDS),
            f"R28 arm {arm} cannot use a '{kind}' demand source outside a check")
    require("calibration" not in source, "R28 arm FG has no calibration lock (that is arm FS)")
    binding = mapping(source)
    if not check or binding["selection"] is not None:
        check_selection(binding["selection"])
    return binding


def _context(source, binding):
    ctx = R.Context({k: source[k] for k in ("kind", "file", "seed") if k in source})
    vars(ctx)["r28_fwhm"] = binding["fwhm"]
    return ctx


# ---------------------------------------------------------------- forward
def _forward(self, x, targets=None):
    h = vars(self.r28_ctx).get("r28_fwhm")
    require(type(h) is float and h.hex() == WIDTH_HEX, f"R28-FG context without the fixed kernel width: {h!r}")
    return FS._forward(self, x, targets)


def is_wrapped(model):
    cls = type(model)
    ctx = vars(model).get("r28_ctx")
    h = vars(ctx).get("r28_fwhm") if isinstance(ctx, R.Context) else None
    return (cls.__dict__.get("_r28fg_native") is not None and cls.__dict__.get("forward") is _forward
            and "forward" not in vars(model) and type(h) is float and h.hex() == WIDTH_HEX and not hasattr(model, "r27_ctx"))


def swap(model, ctx):
    """L0-wrapped native model -> arm FG class, in place; the module layout and the state dict keys do not change.
    The class carries _r28fs_native only because the shared forward body (r28_fovea_cal._forward) reads it; its own
    forward is this module's, so r28_fovea_cal.is_wrapped is False for it."""
    require(L0.is_wrapped(model), "R28-FG swap expects an L0-wrapped native model")
    native = type(model)._l0_native
    before = L0._layout(model)
    if native not in _TYPES:
        _TYPES[native] = type("R28FoveaGeoRTDETR", (native,), {"forward": _forward, "_r28fg_native": native, "_r28fs_native": native})
    model.__class__ = _TYPES[native]
    vars(model)["r28_ctx"] = ctx
    require(L0._layout(model) == before and is_wrapped(model) and not L0.is_wrapped(model) and not R.is_wrapped(model)
            and not FV.is_wrapped(model) and not FS.is_wrapped(model), "R28-FG swap changed the module layout")


# ---------------------------------------------------------------- engine / evaluation model
def create(parent, device, arm, source, *, fixture=False, activation_checkpoint=True, check=False, seed=0):
    """Arm FS's construction order with the fixed width: the demand context first (its network initialiser draws from
    the torch RNG), then L0.create, which restores the parent training RNG as its last step."""
    binding = check_source(arm, source, check)
    ctx = _context(source, binding)
    e, checks = L0.create(parent, device, fixture=fixture, activation_checkpoint=activation_checkpoint, seed=seed)
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
    require(all(checks["wrapped"].values()) and checks["new_parameters"] == 0, ("R28-FG bridge mismatch", checks))
    return e, checks


def build_eval_model(state, device, arm, source, *, check=False, seed=0):
    """FP32 evaluation model from a saved module state (native key set) with the arm's demand source and fixed width."""
    binding = check_source(arm, source, check)
    ctx = _context(source, binding).to(device)
    model = L0.build_eval_model(state, device, seed)
    swap(model, ctx)
    model.r28_arm = arm
    model.r28_mapping = binding
    return model
