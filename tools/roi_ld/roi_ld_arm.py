"""One installed ROI-ld arm; the caller keeps the term for its statistics.

The term is a pure function of the frozen matcher's own assignment and of the ROI boxes the
dataset attaches; it carries no parameter and no buffer, so installing it cannot change any
tensor the model computes.  The two armed modes share every constant and differ only in which
cached view set supplies the ROI boxes.
"""
from __future__ import annotations

import hashlib
import json

OBJECTIVE = "band_roi_resolution_box_distill_v1"
MODES = ("R1", "R2")
# R1 consumes the whole-image view only: the same canvas the student already saw, so it cannot
# carry new information.  R2 consumes the four raw-pixel tiles: the same weights re-encoding
# the image at true raw pixel density.  Both are read from the frozen, SHA-bound teacher cache.
VIEWS = {"R1": (0,), "R2": (1, 2, 3, 4)}
NATIVE_DECLARATION = {"installed": False, "objective": OBJECTIVE, "mode": None}


def declaration(mode):
    """The declared ROI identity of one arm; the native declaration when uninstalled."""
    if mode is None:
        return dict(NATIVE_DECLARATION)
    require(mode in MODES, "unknown roi-ld mode: " + str(mode))
    from . import roi_ld
    return {"installed": True, "objective": OBJECTIVE, "mode": mode, "views": list(VIEWS[mode]),
            "target_iou": float(roi_ld.TARGET_IOU), "targets_per_gt": int(roi_ld.TARGETS_PER_GT),
            "constants_sha256": roi_ld.constants_sha256(),
            "band_area_px": float(roi_ld.BAND_AREA_PX)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def install(model, criterion, mode):
    """Install one ROI-ld arm; returns the term (the caller keeps it for its stats)."""
    from . import roi_ld
    require(mode in MODES, "unknown roi-ld mode: " + str(mode))
    require(getattr(criterion, "roi_ld", None) is None, "a roi-ld term is already installed")
    require(getattr(model, "roi_ld", None) is None, "the model must stay untouched")
    term = roi_ld.RoiTerm(mode)
    criterion.roi_ld = term
    criterion.roi_ld_declaration = declaration(mode)
    return term


def assert_installed(model, criterion, mode, *, term=None):
    require(mode in MODES, "unknown roi-ld mode: " + str(mode))
    installed = getattr(criterion, "roi_ld", None)
    require(installed is not None, "no roi-ld term is installed")
    require(installed.mode == mode, "the installed roi-ld mode differs")
    require(getattr(criterion, "roi_ld_declaration", None) == declaration(mode),
            "the criterion does not carry the declared roi-ld identity")
    require(getattr(model, "roi_ld", None) is None, "the model was touched")
    if term is not None:
        require(installed is term, "the installed term is not the expected one")
    return True


def installed(engine):
    return getattr(engine.criterion, "roi_ld_declaration", dict(NATIVE_DECLARATION))


def declared(engine):
    return installed(engine)


def state(engine):
    term = getattr(engine.criterion, "roi_ld", None)
    if term is None:
        return dict(NATIVE_DECLARATION)
    return {"installed": True, "mode": term.mode, "targets": int(term.targets)}


def load(engine, value):
    term = getattr(engine.criterion, "roi_ld", None)
    if value is None or not value.get("installed"):
        require(term is None, "roi-ld state/term mismatch")
        return dict(NATIVE_DECLARATION)
    require(term is not None, "roi-ld state/term mismatch")
    require(term.mode == value.get("mode"), "roi-ld state mode differs")
    term.targets = int(value.get("targets", 0))
    return {"installed": True, "mode": term.mode, "targets": term.targets}


def digest(value):
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(body).hexdigest()
