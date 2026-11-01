"""Arm adapter: install the tiny-band extra-positive term into the frozen chain (CUDA entry).

Mirrors the frozen arm-install convention of the r10/r17/r18 rounds:

    install(model, criterion, mode, ...)   -> the installed term (native mode: nothing)
    declaration(mode)                      -> the identity the checkpoint behaviour carries
    assert_installed(model, criterion, mode)
    declared(engine)                       -> the declaration this engine must carry
    installed(engine) / state(engine) / load(engine, state)

The term owns no parameter and no buffer and consumes no random draw: the extra positive set is
a deterministic function of the frozen matching, the predicted boxes and the ground-truth boxes,
so the installation order relative to ``model.to('cuda:0')`` does not matter and every training
RNG stream is untouched.

The declaration is non-vacuous: the patched ``checkpoint.behavior_identity`` expects the
declaration derived from the engine's declared positives mode and refuses a criterion whose own
declaration differs, so a state saved under one arm cannot be restored under another arm or into
an unarmed engine.
"""
from __future__ import annotations

from .tiny_positives import (BAND_AREA_PX, EXTRAS_PER_GT, IOU_TAU, OBJECTIVE, REQUIRE_BEST_GT,
                             TinyPositivesTerm, band_gt, constants_sha256, observe_call)

MODES = ("A1", "A2")
NATIVE_DECLARATION = {"installed": False, "objective": None, "mode": None, "iou_tau": None,
                      "extras_per_gt": None, "constants_sha256": None, "band_area_px": None}
DECLARATION_KEYS = tuple(sorted(NATIVE_DECLARATION))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def declaration(mode):
    """The declared positives identity of one arm; the native declaration when uninstalled."""
    if mode is None:
        return dict(NATIVE_DECLARATION)
    require(mode in MODES, "unknown tiny-positives mode: " + str(mode))
    return {"installed": True, "objective": OBJECTIVE, "mode": mode, "iou_tau": float(IOU_TAU),
            "extras_per_gt": int(EXTRAS_PER_GT), "constants_sha256": constants_sha256(),
            "band_area_px": float(BAND_AREA_PX)}


def install(model, criterion, mode):
    """Install one positives arm; returns the term (the caller keeps it for its stats)."""
    require(mode in MODES, "unknown tiny-positives mode: " + str(mode))
    require(getattr(criterion, "tiny_positives", None) is None, "a positives term is already installed")
    require(getattr(model, "tiny_positives", None) is None, "the model must stay untouched")
    term = TinyPositivesTerm(mode)
    criterion.tiny_positives = term
    criterion.tiny_positives_declaration = declaration(mode)
    return term


def assert_installed(model, criterion, mode, *, term=None):
    require(mode in MODES, "unknown tiny-positives mode: " + str(mode))
    installed_term = getattr(criterion, "tiny_positives", None)
    require(installed_term is not None, "no positives term is installed")
    require(installed_term.mode == mode, "the installed positives mode differs")
    require(getattr(criterion, "tiny_positives_declaration", None) == declaration(mode),
            "the criterion does not carry the declared positives identity")
    require(getattr(model, "tiny_positives", None) is None, "the model was touched")
    if term is not None:
        require(installed_term is term, "the installed term is not the expected one")
    return True


def installed(engine):
    return getattr(getattr(engine, "criterion", None), "tiny_positives", None) is not None


def declared(engine):
    """The declaration this engine must carry, and the criterion's own declaration must match."""
    expected = declaration(getattr(engine, "pos_mode", None))
    actual = getattr(engine.criterion, "tiny_positives_declaration", None)
    if actual != expected:
        raise ValueError("criterion positives declaration differs from the engine's positives mode")
    return expected


def state(engine):
    """The term's own state for the chain checkpoint; refuses a half-installed engine."""
    mode = getattr(engine, "pos_mode", None)
    term = getattr(getattr(engine, "criterion", None), "tiny_positives", None)
    if mode is None:
        require(term is None, "an engine without a declared positives mode carries a term")
        return None
    require(term is not None and term.mode == mode, "declared positives mode without an installed term")
    return {"declaration": declaration(mode), "term": term.state_dict()}


def load(engine, value):
    """Restore the term's own state into an engine with the same declared positives identity."""
    mode = getattr(engine, "pos_mode", None)
    term = getattr(getattr(engine, "criterion", None), "tiny_positives", None)
    if value is None:
        require(mode is None and term is None,
                "an unarmed checkpoint cannot be restored into a positives-armed engine")
        return None
    require(mode is not None and term is not None,
            "a positives-armed checkpoint cannot be restored into an unarmed engine")
    require(value.get("declaration") == declaration(mode), "the checkpoint's positives declaration differs")
    term.load_state_dict(value["term"])
    return {"positives_restored": True, "mode": mode, "extras": int(value["term"]["extras"])}
