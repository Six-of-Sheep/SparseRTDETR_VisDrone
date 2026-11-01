"""Arm adapter: install the selection-ranking term into the frozen chain (CUDA entry).

Mirrors the frozen module-install convention of the r10/r12 rounds:

    install(model, criterion, mode, ...)   -> the installed term (native mode: nothing)
    declaration(mode)                      -> the identity the checkpoint behaviour carries
    assert_installed(model, criterion, mode)
    declared(engine)                       -> the declaration this engine must carry
    installed(engine) / state(engine) / load(engine, state)

Device placement: the term owns no parameter and no buffer, every tensor it creates follows
the pool logits' device, and its only mutable state is a CPU ``torch.Generator``, so the
installation order relative to ``model.to('cuda:0')`` does not matter.  The frozen
``construct`` installs a module before the move to CUDA and builds the two EMA records
(deep copies of the model) after it; the term's ``__deepcopy__`` contract covers that copy.

The declaration is non-vacuous: ``checkpoint.behavior_identity`` (r16 patch) expects the
declaration derived from the engine's declared ranking mode and refuses a criterion whose
own declaration differs, so a state saved under one ranking arm cannot be restored under
another, and an armed state cannot be restored into an unarmed engine.
"""
from __future__ import annotations

import hashlib
import json

from ..localization import consumer_geometry
from .selection_rank import DECLARATION, SelectionRankTerm

MODES = ("A2", "A1")
NATIVE_DECLARATION = {"installed": False, "objective": None, "mode": None,
                      "objective_weight": None, "constants_sha256": None}
DECLARATION_KEYS = tuple(sorted(NATIVE_DECLARATION))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def declaration(mode):
    """The declared ranking identity of one arm; the native declaration when uninstalled."""
    if mode is None:
        return dict(NATIVE_DECLARATION)
    require(mode in MODES, "unknown ranking mode: " + str(mode))
    constants = json.dumps(DECLARATION["constants"], sort_keys=True, separators=(",", ":")).encode()
    return {"installed": True, "objective": DECLARATION["objective"], "mode": mode,
            "objective_weight": DECLARATION["total_loss"]["weight"],
            "constants_sha256": hashlib.sha256(constants).hexdigest()}


def install(model, criterion, mode, *, seed=0, consumer=consumer_geometry):
    """Install one ranking arm; returns the term (the caller keeps it for its stats)."""
    require(mode in MODES, "unknown ranking mode: " + str(mode))
    require(getattr(criterion, "selection_rank", None) is None, "a ranking term is already installed")
    term = SelectionRankTerm(mode, seed=seed, k=int(model.decoder.num_queries))
    term.attach(model, criterion, consumer=consumer)
    criterion.selection_rank_declaration = declaration(mode)
    return term


def assert_installed(model, criterion, mode, *, term=None):
    require(mode in MODES, "unknown ranking mode: " + str(mode))
    installed_term = getattr(criterion, "selection_rank", None)
    require(installed_term is not None, "no ranking term is installed")
    require(installed_term.mode == mode, "the installed ranking mode differs")
    require(getattr(criterion, "selection_rank_declaration", None) == declaration(mode),
            "the criterion does not carry the declared ranking identity")
    installed_term.assert_installed(model, criterion)
    if term is not None:
        require(installed_term is term, "the installed term is not the expected one")
    return True


def installed(engine):
    return getattr(getattr(engine, "criterion", None), "selection_rank", None) is not None


def declared(engine):
    """The declaration this engine must carry, and the criterion's own declaration must match."""
    expected = declaration(getattr(engine, "ranking_mode", None))
    actual = getattr(engine.criterion, "selection_rank_declaration", None)
    if actual != expected:
        raise ValueError("criterion ranking declaration differs from the engine's ranking mode")
    return expected


def state(engine):
    """The term's own state for the chain checkpoint; refuses a half-installed engine."""
    mode = getattr(engine, "ranking_mode", None)
    term = getattr(getattr(engine, "criterion", None), "selection_rank", None)
    if mode is None:
        require(term is None, "an engine without a declared ranking mode carries a term")
        return None
    require(term is not None and term.mode == mode, "declared ranking mode without an installed term")
    return {"declaration": declaration(mode), "term": term.state_dict()}


def load(engine, value):
    """Restore the term's own state into an engine with the same declared ranking identity."""
    mode = getattr(engine, "ranking_mode", None)
    term = getattr(getattr(engine, "criterion", None), "selection_rank", None)
    if value is None:
        require(mode is None and term is None,
                "an unarmed checkpoint cannot be restored into a ranking-armed engine")
        return None
    require(mode is not None and term is not None,
            "a ranking-armed checkpoint cannot be restored into an unarmed engine")
    require(value.get("declaration") == declaration(mode), "the checkpoint's ranking declaration differs")
    term.load_state_dict(value["term"])
    return {"ranking_restored": True, "mode": mode, "draws": int(value["term"]["draws"])}
