"""r16 item 4: the A1 state through the REAL checkpoint chain (CPU, no training).

Builds the frozen fine-tune engine on the certified e120 parent with the A1 arm installed
in the same order the frozen ``construct`` uses (install before the model moves, then the
two EMA records are deep copies of the installed model), saves one chain checkpoint
through the chain's own ``slice_finetune.save`` and restores it into a fresh engine with
``slice_finetune.restore``.  Evidence: the captured state carries the ranking declaration
and the term's own state; the restored term state is exactly equal (generator state and
draw counter, via the chain's own ``exact_equal`` over the full capture); the next draws
continue identically; a state saved under A1 is refused by an A2 engine and by an unarmed
engine, and an unarmed state is refused by an armed engine.  No update, no optimizer step,
no EMA step and no image data are involved.

usage: CUDA_VISIBLE_DEVICES= <frozen python> -I verify_chain_entry.py
       --source-root <dev-tree>/source --contract <production-contract-003.json> --out <dir>
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

PLAN_SHA = "d0a3593d14878bf0e0d400e68e35fdec40c33dd742aceae0ab7dd78c4714c11c"
FROZEN_CHECKPOINT_SHA = "13a35283a02d2b728e3449917586ded5cdf2d578ebe6847510748e0dc4d21757"
COMMON_ARM = "RT-DETRv2-R18-common-protocol-v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def content_digest(value):
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(canonical).hexdigest()


def load_tree(root):
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root))
    import sparse_rtdetr
    require(str(Path(sparse_rtdetr.__file__).resolve()).startswith(str((root / "src").resolve())),
            "sparse_rtdetr resolved outside the staged tree")
    from sparse_rtdetr.protocol_r18.model import Config, build
    from sparse_rtdetr.protocol_r18 import checkpoint as native_state
    from sparse_rtdetr.protocol_r18 import slice_finetune as ft
    from sparse_rtdetr.protocol_r18 import selection_rank_arm as arm
    from sparse_rtdetr.localization.runio import read as read_json
    return {"Config": Config, "build": build, "native_state": native_state, "ft": ft,
            "arm": arm, "read_json": read_json}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(args.source_root).resolve()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    require(not torch.cuda.is_initialized(), "CUDA was initialized before the CPU chain-entry check")
    m = load_tree(root)
    print("TREE", root, flush=True)

    contract = m["read_json"](args.contract)
    require(contract["contract_sha256"] == content_digest(
        {k: v for k, v in contract.items() if k != "contract_sha256"}), "contract content digest differs")
    require(file_sha(contract["plan"]) == contract["plan_sha256"] == PLAN_SHA, "the frozen plan differs")
    parent = contract["parent_checkpoint"]
    require(file_sha(parent["path"]) == parent["sha256"] == FROZEN_CHECKPOINT_SHA,
            "the parent is not the frozen e120 state")
    binding_ref = contract["parent_binding"]
    require(file_sha(binding_ref["path"]) == binding_ref["sha256"], "the parent binding file differs")
    parent_binding = m["read_json"](binding_ref["path"])
    config = m["Config"](input_hw=(768, 1344), physical_batch_size=1, amp_dtype="float32")

    def make_engine(mode):
        """The frozen construct order: build, install, move, then the two EMA deep copies."""
        model, criterion, info = m["build"](config, synthetic=True)
        payload = m["native_state"].read(parent["path"], parent["sha256"], parent_binding)
        initial = m["ft"].initialize_from_parent(model, payload, expected_binding=parent_binding)
        del payload
        require(initial == contract["initial_model_sha256"], "the parent EMA did not load exactly")
        term = None
        if mode is not None:
            term = m["arm"].install(model, criterion, mode)
        optimizer, ema, alt = m["ft"].make_optimizer_ema(model, config)
        engine = m["ft"].FineTuneEngine(model, criterion, optimizer, ema, config, arm="D0",
                                        plan_sha256=contract["plan_sha256"],
                                        parent_sha256=parent["sha256"], alt_ema=alt)
        engine.ranking_mode = mode
        return engine, term, info

    random.seed(0)
    np.random.seed(0)
    torch.random.default_generator.manual_seed(0)

    engine, term, info = make_engine("A1")
    require(term is not None, "the A1 arm was not installed")
    m["arm"].assert_installed(engine.model, engine.criterion, "A1", term=term)
    require(m["arm"].state(engine) is not None, "the engine does not carry a ranking state")
    copies = {"model": engine.model, "ema": engine.ema.module, "ema_alt": engine.alt_ema.module}
    for label, module in copies.items():
        bound = module.decoder._select_topk
        require(isinstance(bound, functools.partial) and bound.args[0] is module.decoder,
                ("a chain copy does not carry its own installed wrapper", label))
    require(copies["ema"].decoder._select_topk.args[1] is not term
            and copies["ema_alt"].decoder._select_topk.args[1] is not term,
            "an EMA copy shares the live term instead of its own copy")
    for _ in range(5):
        torch.randint(2 ** 31, (1,), generator=term.generator)
    state_before = term.state_dict()
    print("INSTALLED A1 draws", term.draws, flush=True)

    binding = {"arm": COMMON_ARM, "scope": "cpu_R18_synthetic",
               "run_id": "r18-selection-rank-r16-chain-entry-001",
               "config": config.binding(), "model": info}
    path = out_dir / "chain-entry-A1.pth"
    reference = m["ft"].save(engine, path)
    captured = m["ft"].capture(engine)
    require(captured["selection_rank"] is not None, "the chain capture does not carry the ranking state")
    require(captured["selection_rank"]["declaration"] == m["arm"].declaration("A1"),
            "the captured ranking declaration differs")
    require(torch.equal(captured["selection_rank"]["term"]["a1_generator"], state_before["a1_generator"])
            and int(captured["selection_rank"]["term"]["draws"]) == int(state_before["draws"]),
            "the captured term state differs from the live term state")
    require(captured["behavior"]["criterion"].get("selection_rank") == m["arm"].declaration("A1"),
            "the behaviour identity does not carry the ranking declaration")
    print("SAVED", reference["bytes"], reference["sha256"][:16], flush=True)

    clone, clone_term, _ = make_engine("A1")
    restored = m["ft"].restore(clone, reference)
    require(restored.get("full_state_RNG_exact") is True, "the chain restore is not exact")
    require(m["native_state"].exact_equal(captured, m["ft"].capture(clone)),
            "the restored full state (including the term state) differs")
    drawings = []
    for source in (term, clone_term):
        drawings.append([int(torch.randint(2 ** 31, (1,), generator=source.generator).item()) for _ in range(2)])
    require(drawings[0] == drawings[1], ("the A1 stream does not continue identically after restore", drawings))
    require(int(clone_term.draws) == int(term.draws), "the restored draw counter differs")

    armed_refusal = None
    other, other_term, _ = make_engine("A2")
    try:
        m["ft"].restore(other, reference)
    except Exception as error:  # noqa: BLE001
        armed_refusal = repr(error)
    require(armed_refusal is not None, "an A1 state was accepted by an A2 engine")
    require(other.failed is True, "the refused A2 restore did not mark the engine failed")

    unarmed, unarmed_term, _ = make_engine(None)
    unarmed_refusal = None
    try:
        m["ft"].restore(unarmed, reference)
    except Exception as error:  # noqa: BLE001
        unarmed_refusal = repr(error)
    require(unarmed_refusal is not None, "an armed state was accepted by an unarmed engine")
    require(m["arm"].state(unarmed) is None, "an unarmed engine reports a ranking state")
    require(m["arm"].load(unarmed, None) is None, "an unarmed engine refused its own (absent) state")
    reverse_refusal = None
    try:
        m["arm"].load(engine, None)
    except Exception as error:  # noqa: BLE001
        reverse_refusal = repr(error)
    require(reverse_refusal is not None, "an armed engine accepted an absent ranking state")
    print("REFUSALS", armed_refusal[:80] if armed_refusal else None, flush=True)

    report = {
        "status": "PASS",
        "scope": ("CPU check of the A1 state through the real checkpoint chain: save and restore with "
                  "slice_finetune.capture/save/restore; no training update, no optimizer step, no EMA "
                  "step, no image data, no GPU"),
        "source_root": str(root),
        "contract": {"task_id": contract["task_id"], "plan_sha256": contract["plan_sha256"],
                     "parent_checkpoint_sha256": parent["sha256"]},
        "config": {"input_hw": [768, 1344], "physical_batch_size": 1, "amp_dtype": "float32",
                   "install_order": "install -> two EMA deep copies (the frozen construct order)"},
        "checkpoint": {"path": reference["path"], "bytes": reference["bytes"], "sha256": reference["sha256"],
                       "updates": int(engine.completed_updates)},
        "ranking_state": {"declaration": m["arm"].declaration("A1"), "term_state_bytes": int(
            state_before["a1_generator"].numel()), "draws_before_save": int(state_before["draws"])},
        "restore": {"full_state_RNG_exact": bool(restored.get("full_state_RNG_exact")),
                    "capture_exact_including_term_state": True,
                    "draws_continue_identically": True,
                    "draws_after_restore": int(clone_term.draws)},
        "refusals": {"a1_state_into_a2_engine": armed_refusal,
                     "armed_state_into_unarmed_engine": unarmed_refusal,
                     "absent_state_into_armed_engine": reverse_refusal},
        "no_training": {"completed_updates": int(engine.completed_updates),
                        "optimizer_state_empty": not bool(engine.optimizer.state),
                        "ema_updates": int(engine.ema.updates),
                        "ema_alt_updates": int(engine.alt_ema.updates)},
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "cuda_initialized": bool(torch.cuda.is_initialized()),
    }
    (out_dir / "CHAIN-ENTRY.json").write_text(json.dumps(report, indent=1, ensure_ascii=False),
                                              encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("checkpoint", "ranking_state", "restore",
                                                   "no_training", "elapsed_seconds")},
                     ensure_ascii=False), flush=True)
    print("CHAIN-ENTRY-COMPLETE", report["elapsed_seconds"], "s", flush=True)


if __name__ == "__main__":
    main()
