"""Arm adapter: install the tiny-band geometry term into the frozen chain (CUDA entry).

Mirrors the frozen arm-install convention of the r10/r17 rounds:

    install(model, criterion, mode, ...)   -> the installed term (native mode: nothing)
    declaration(mode)                      -> the identity the checkpoint behaviour carries
    assert_installed(model, criterion, mode)
    declared(engine)                       -> the declaration this engine must carry
    installed(engine) / state(engine) / load(engine, state)

The term owns no parameter and no buffer, every tensor it creates follows the pair
tensors' device, and its only mutable state is a CPU ``torch.Generator`` plus a draw
counter, so the installation order relative to ``model.to('cuda:0')`` does not matter and
the training RNG streams are never touched.

The declaration is non-vacuous: the patched ``checkpoint.behavior_identity`` expects the
declaration derived from the engine's declared geometry mode and refuses a criterion whose
own declaration differs, so a state saved under one geometry arm cannot be restored under
another, and an armed state cannot be restored into an unarmed engine.
"""
from __future__ import annotations

from .tiny_geometry import (CONSTANT_C, OBJECTIVE, BAND_AREA_PX, TinyGeometryTerm,
                            constants_sha256, observe_call)

MODES = ("A1", "A2")
NATIVE_DECLARATION = {"installed": False, "objective": None, "mode": None, "constant_C": None,
                      "constants_sha256": None, "band_area_px": None}
DECLARATION_KEYS = tuple(sorted(NATIVE_DECLARATION))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def declaration(mode):
    """The declared geometry identity of one arm; the native declaration when uninstalled."""
    if mode is None:
        return dict(NATIVE_DECLARATION)
    require(mode in MODES, "unknown geometry mode: " + str(mode))
    require(CONSTANT_C is not None, "the frozen constant C is not set")
    return {"installed": True, "objective": OBJECTIVE, "mode": mode, "constant_C": float(CONSTANT_C),
            "constants_sha256": constants_sha256(), "band_area_px": float(BAND_AREA_PX)}


def install(model, criterion, mode, *, seed=0):
    """Install one geometry arm; returns the term (the caller keeps it for its stats)."""
    require(mode in MODES, "unknown geometry mode: " + str(mode))
    require(getattr(criterion, "tiny_geometry", None) is None, "a geometry term is already installed")
    require(getattr(model, "tiny_geometry", None) is None, "the model must stay untouched")
    term = TinyGeometryTerm(mode, seed=seed)
    criterion.tiny_geometry = term
    criterion.tiny_geometry_declaration = declaration(mode)
    return term


def assert_installed(model, criterion, mode, *, term=None):
    require(mode in MODES, "unknown geometry mode: " + str(mode))
    installed_term = getattr(criterion, "tiny_geometry", None)
    require(installed_term is not None, "no geometry term is installed")
    require(installed_term.mode == mode, "the installed geometry mode differs")
    require(getattr(criterion, "tiny_geometry_declaration", None) == declaration(mode),
            "the criterion does not carry the declared geometry identity")
    require(getattr(model, "tiny_geometry", None) is None, "the model was touched")
    if term is not None:
        require(installed_term is term, "the installed term is not the expected one")
    return True


def installed(engine):
    return getattr(getattr(engine, "criterion", None), "tiny_geometry", None) is not None


def declared(engine):
    """The declaration this engine must carry, and the criterion's own declaration must match."""
    expected = declaration(getattr(engine, "geo_mode", None))
    actual = getattr(engine.criterion, "tiny_geometry_declaration", None)
    if actual != expected:
        raise ValueError("criterion geometry declaration differs from the engine's geometry mode")
    return expected


def state(engine):
    """The term's own state for the chain checkpoint; refuses a half-installed engine."""
    mode = getattr(engine, "geo_mode", None)
    term = getattr(getattr(engine, "criterion", None), "tiny_geometry", None)
    if mode is None:
        require(term is None, "an engine without a declared geometry mode carries a term")
        return None
    require(term is not None and term.mode == mode, "declared geometry mode without an installed term")
    return {"declaration": declaration(mode), "term": term.state_dict()}


def load(engine, value):
    """Restore the term's own state into an engine with the same declared geometry identity."""
    mode = getattr(engine, "geo_mode", None)
    term = getattr(getattr(engine, "criterion", None), "tiny_geometry", None)
    if value is None:
        require(mode is None and term is None,
                "an unarmed checkpoint cannot be restored into a geometry-armed engine")
        return None
    require(mode is not None and term is not None,
            "a geometry-armed checkpoint cannot be restored into an unarmed engine")
    require(value.get("declaration") == declaration(mode), "the checkpoint's geometry declaration differs")
    term.load_state_dict(value["term"])
    return {"geometry_restored": True, "mode": mode, "draws": int(value["term"]["draws"])}


def observe(pred, target, module):
    """Value-only native-arm statistics for the patched criterion; no arithmetic change."""
    return observe_call(pred, target, module, family=".", sink=None)
