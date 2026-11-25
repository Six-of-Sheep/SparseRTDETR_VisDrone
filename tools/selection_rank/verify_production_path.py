"""r15: verify the selection-ranking term through the real production path (CPU, no GPU).

Runs against the staged development tree (frozen dual-EMA release-003 source plus the
committed r15 overlay) with CUDA hidden and the frozen interpreter.

  build      the production r18 model/criterion on the real 768x1344 canvas with the
             declared CPU verification configuration (physical batch 1, fp32), the
             certified e120 parent EMA loaded through the chain's own
             ``initialize_from_parent``, the chain's optimizers/EMAs and FineTuneEngine,
             and real train_core images through the chain's own ``Dataset`` (manifest and
             roots taken from the frozen production contract) in the chain's own logical
             batch order, collated with the chain's own ``collate_window``.
  window 1   one full production update (``engine.train_window``) with the method arm A2
             installed and its gradient audit armed: real loss-key counts (22 with DN,
             13 without), actual pair count, zero-pair fraction and A1/A2 choice-space
             fraction, and a check that the backward pass placed exactly -coef/N on each
             pair's better (unselected) pool logit and nothing else on any unselected
             pool token.
  window 2   one full update on the same real images with the supervision emptied: the
             empty-pair path must give ``loss_rank`` exactly 0.0 and no rank gradient.
  pass 3     one more forward on the same two images, then three criterion calls on that
             single graph: native (no term), A1 (own generator) and A2.  Native values
             must be bitwise unchanged by installation (only ``loss_rank`` is added);
             A1 and A2 must see identical eligibility/counts/margins while A1's draw
             advances only its own generator; the A1 generator state must round-trip
             exactly through save/restore and torch.save/load; and the cached top-k
             indices must reproduce the decoder's gathered enc_topk logits/boxes bitwise.

CPU only.  No gate, no release, no training round, no scoring.  Declared limits: the CPU
verification configuration is not the production recipe (real canvas, batch 1, fp32; the
production recipe is batch 8 / bf16 and is unchanged); pass 3 skips the engine's guard
contexts and does not step the optimizer; GPU cost is not measured here.

usage: CUDA_VISIBLE_DEVICES= <frozen python> -I verify_production_path.py
       --source-root <dev-tree>/source --contract <production-contract-003.json>
       --out <evidence dir> [--epoch 1]
"""
from __future__ import annotations

import argparse
import copy
import functools
import hashlib
import io
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

PLAN_SHA = "d0a3593d14878bf0e0d400e68e35fdec40c33dd742aceae0ab7dd78c4714c11c"
FROZEN_CHECKPOINT_SHA = "13a35283a02d2b728e3449917586ded5cdf2d578ebe6847510748e0dc4d21757"


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
    """Import the staged package; sparse_rtdetr must resolve inside the staged tree."""
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root))
    import sparse_rtdetr
    require(str(Path(sparse_rtdetr.__file__).resolve()).startswith(str((root / "src").resolve())),
            "sparse_rtdetr resolved outside the staged tree")
    from sparse_rtdetr.protocol_r18.model import Config, build, fp32_outputs
    from sparse_rtdetr.protocol_r18 import checkpoint as native_state
    from sparse_rtdetr.protocol_r18 import slice_finetune as ft
    from sparse_rtdetr.protocol_r18 import selection_rank as sr
    from sparse_rtdetr.localization import consumer_geometry as geo
    from sparse_rtdetr.localization.data import collate_window
    from sparse_rtdetr.localization.runio import read as read_json
    from sparse_rtdetr.baseline import training_v2b_checkpoint as layout
    from sparse_rtdetr.baseline.training_v2b import logical_batch_indices
    return {"Config": Config, "build": build, "fp32_outputs": fp32_outputs,
            "native_state": native_state, "ft": ft, "sr": sr, "geo": geo,
            "collate_window": collate_window, "read_json": read_json, "layout": layout,
            "logical_batch_indices": logical_batch_indices}


def build_window(m, dataset, source_epoch, window_index):
    """Two real train_core items in the chain's own order, collated with the chain's collate."""
    order = None
    for indices in m["logical_batch_indices"](len(dataset), seed=0, epoch=source_epoch,
                                             logical_batch_size=16, completed_batches=window_index):
        order = indices
        break
    require(order is not None, "window index outside the epoch order")
    items = [dataset[(source_epoch, 0, order[0])], dataset[(source_epoch, 1, order[1])]]
    images, targets, traces = m["collate_window"](items)
    require(tuple(images.shape) == (2, 3, 768, 1344) and images.dtype == torch.float32,
            "window shape/dtype differs from the declared CPU verification window")
    require(len(targets) == 2 and len(traces) == 2, "window membership differs")
    return images, targets, traces


def check_gradient_audit(audit, coefficient):
    """Only unselected pool tokens are inspected: no native loss reaches them."""
    require(audit["finite"] is True, "the term's backward observation is not finite")
    require(audit["unselected_nonzero_truncated"] is False, "unselected gradient entries were truncated")
    better = [entry for entry in audit["expected_entries"] if entry[3] < 0]
    got = audit["unselected_nonzero"]
    require(len(got) == len(better),
            ("unselected pool entries with a gradient differ from the declared pairs", len(got), len(better)))
    table = {(entry[0], entry[1], entry[2]): entry[3] for entry in better}
    for image, token, cls, value in got:
        require((image, token, cls) in table, ("a pool gradient outside the declared pairs", image, token, cls))
        want = coefficient * table[(image, token, cls)]
        require(abs(value - want) < 1e-6, ("a pool gradient value differs from the declared term gradient", value, want))
    return {"active_pairs": len(better), "entries": len(got), "coefficient": coefficient, "values_exact": True}


def emptied(target):
    """The same real image and the same target keys, with the supervision removed."""
    value = dict(target)
    value["boxes"] = target["boxes"][:0]
    value["labels"] = target["labels"][:0]
    value["raw_area"] = target["raw_area"][:0]
    value["gt_ids"] = []
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epoch", type=int, default=1)
    args = parser.parse_args()

    root = Path(args.source_root).resolve()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    require(not torch.cuda.is_initialized(), "CUDA was initialized before the production-path verification")
    m = load_tree(root)
    print("TREE", root, flush=True)

    contract = m["read_json"](args.contract)
    require(contract.get("scope") == "PRODUCTION"
            and str(contract.get("task_id", "")).startswith("r18-dual-ema-20261031-001"),
            "unexpected contract identity")
    require(contract["contract_sha256"] == content_digest(
        {k: v for k, v in contract.items() if k != "contract_sha256"}), "contract content digest differs")
    require(file_sha(contract["plan"]) == contract["plan_sha256"] == PLAN_SHA, "the frozen plan differs")
    parent = contract["parent_checkpoint"]
    require(Path(parent["path"]).stat().st_size == parent["bytes"], "parent checkpoint size differs")
    require(file_sha(parent["path"]) == parent["sha256"] == FROZEN_CHECKPOINT_SHA,
            "the parent is not the frozen e120 state")
    binding_ref = contract["parent_binding"]
    require(file_sha(binding_ref["path"]) == binding_ref["sha256"], "the parent binding file differs")
    binding = m["read_json"](binding_ref["path"])
    print("CONTRACT", contract["task_id"], flush=True)

    config = m["Config"](input_hw=(768, 1344), physical_batch_size=1, amp_dtype="float32")
    require(config.physical_batch_size == 1 and config.amp_dtype == "float32" and config.num_denoising == 100,
            "unexpected CPU verification configuration")
    model, criterion, info = m["build"](config, synthetic=True)
    decoder = model.decoder
    require(decoder.num_queries == 300, "the selection rule is not the frozen top-300 rule")
    require(decoder.query_select_method == "default", "the frozen selection method is not the default one")
    print("BUILT", info["trainable_parameters"], flush=True)

    payload = m["native_state"].read(parent["path"], parent["sha256"], binding)
    initial = m["ft"].initialize_from_parent(model, payload, expected_binding=binding)
    del payload
    require(initial == contract["initial_model_sha256"], "the parent EMA did not load into the exact initial state")
    optimizer, ema, alt = m["ft"].make_optimizer_ema(model, config)
    engine = m["ft"].FineTuneEngine(model, criterion, optimizer, ema, config, arm="D0",
                                    plan_sha256=contract["plan_sha256"], parent_sha256=parent["sha256"],
                                    alt_ema=alt)
    random.seed(0)
    np.random.seed(0)
    torch.random.default_generator.manual_seed(0)

    data = contract["data"]["train_core"]
    dataset = m["ft"].Dataset("D0", manifest_path=data["manifest"], images_path=data["roots"]["images"],
                              annotations_path=data["roots"]["annotations"], root_binding=data["roots"])
    source_epoch = m["ft"].clock(args.epoch)
    require(source_epoch == 109, "the declared verification epoch is not the chain's first pass")
    images, targets, traces = build_window(m, dataset, source_epoch, 0)
    print("WINDOW", [int(len(row["labels"])) for row in targets],
          [row["image_id"] for row in traces], flush=True)

    layout_before = m["layout"]._model_layout(model)
    names_before = [name for name, _ in model.named_parameters()]
    numel_before = sum(p.numel() for p in model.parameters())
    state_keys_before = sorted(model.state_dict().keys())

    term = m["sr"].SelectionRankTerm("A2", seed=0, k=300, audit_gradients=True)
    term.attach(model, criterion, consumer=m["geo"])
    term.assert_installed(model, criterion)
    require(m["layout"]._model_layout(model) == layout_before, "installation changed the model layout")
    require([name for name, _ in model.named_parameters()] == names_before
            and sum(p.numel() for p in model.parameters()) == numel_before
            and sorted(model.state_dict().keys()) == state_keys_before,
            "installation changed the parameter/buffer layout")
    clone = copy.deepcopy(model)
    bound = clone.decoder._select_topk
    require(isinstance(bound, functools.partial) and bound.args[0] is clone.decoder and bound.args[1] is not term,
            "the EMA-style deepcopy did not rebind the selection wrapper to the copy")
    term.assert_installed(model, criterion)
    del clone
    print("ATTACHED", term.mode, flush=True)

    # --- window 1: one production update with the method arm -------------------------
    r1 = engine.train_window(images, targets, capture=True)
    counts1 = [len(row) for row in r1["microbatch_losses"]]
    expected1 = [22 if len(targets[index]["labels"]) else 13 for index in range(2)]
    require(counts1 == expected1, ("production loss-key counts differ", counts1, expected1))
    require(all("loss_rank" in row for row in r1["microbatch_losses"]), "loss_rank is missing from a microbatch")
    require(r1["finite"] is True and r1["gradient_norm_before_clip"] > 0.0,
            "the production update did not run through the real path")
    require(engine.completed_updates == 1 and engine.ema.updates == 1
            and engine.alt_ema.updates == 1 + m["ft"].ALT_OFFSET, "the production update clocks differ")
    for index, row in enumerate(term.history):
        require(abs(r1["microbatch_losses"][index]["loss_rank"] - row["loss"]) < 1e-12,
                "the engine's loss_rank value differs from the term's own value")
    require(len(term.grad_audits) == 2, "the gradient audit did not observe both micro-batches")
    coefficient = r1["coefficients"][0]
    require(coefficient == .5, "the frozen engine coefficient changed")
    gradient_checks = [check_gradient_audit(audit, coefficient) for audit in term.grad_audits]
    require(sum(item["active_pairs"] for item in gradient_checks)
            == sum(row["active_pairs"] for row in term.history),
            "the backward audit did not see every active pair")
    window1_history = list(term.history)
    instances = sum(row["instances"] for row in window1_history)
    pairs = sum(row["pairs"] for row in window1_history)
    zero_pairs = sum(row["zero_pair_instances"] for row in window1_history)
    choice_space = sum(row["choice_space_instances"] for row in window1_history)
    require(instances > 0, "the real window had no matched instance, the mechanism had no chance to act")
    require(pairs > 0, "the real window produced no pair at all")
    headline = {"instances": instances, "pairs": pairs, "zero_pair_instances": zero_pairs,
                "choice_space_instances": choice_space,
                "zero_pair_fraction": zero_pairs / instances,
                "choice_space_fraction": choice_space / instances}
    print("WINDOW1", json.dumps({"counts": counts1, "headline": headline,
                                 "gradient": gradient_checks}), flush=True)

    # --- window 2: the same real pipeline with the supervision emptied ---------------
    r2 = engine.train_window(images, [emptied(row) for row in targets], capture=True)
    counts2 = [len(row) for row in r2["microbatch_losses"]]
    require(counts2 == [13, 13], ("the empty-pair loss-key counts differ", counts2))
    require(all(row["loss_rank"] == 0.0 for row in r2["microbatch_losses"]),
            "the empty-pair path did not contribute exactly zero")
    empty_history = window2_history = list(term.history[len(window1_history):])
    require(all(row["pairs"] == 0 and row["loss"] == 0.0 for row in empty_history),
            "the empty-pair path produced a pair or a nonzero loss")
    for audit in term.grad_audits[2:]:
        require(audit["finite"] and audit["expected_entries"] == [] and audit["unselected_nonzero"] == [],
                "the empty-pair window touched unselected pool tokens")
    require(engine.completed_updates == 2, "the empty-pair update clock differs")
    print("WINDOW2", json.dumps({"counts": counts2}), flush=True)

    # --- pass 3: one graph, native / A1 / A2 -----------------------------------------
    model.train()
    out = model(images, targets=targets)
    cache = dict(term._cache or {})
    require(set(cache) == {"pool_logits", "pool_coord_unact", "topk_indices"},
            "no captured pool batch on the production forward")
    pool_logits = cache["pool_logits"]
    pool_coord = cache["pool_coord_unact"]
    topk = cache["topk_indices"]
    classes = pool_logits.shape[-1]
    gathered_logits = pool_logits.gather(1, topk.unsqueeze(-1).expand(-1, -1, classes))
    gathered_boxes = torch.sigmoid(pool_coord.gather(1, topk.unsqueeze(-1).expand(-1, -1, 4)))
    reference = out["enc_aux_outputs"][0]
    mapping = {"pool_tokens": int(pool_logits.shape[1]), "selected": int(topk.shape[1]),
               "logits_bitwise_equal": bool(torch.equal(gathered_logits, reference["pred_logits"])),
               "boxes_bitwise_equal": bool(torch.equal(gathered_boxes, reference["pred_boxes"]))}
    require(mapping["logits_bitwise_equal"] and mapping["boxes_bitwise_equal"],
            "the cached top-k indices do not reproduce the decoder's gathered enc_topk tensors")

    criterion.selection_rank = None
    native = criterion(m["fp32_outputs"](out), targets)
    require("loss_rank" not in native, "the native criterion call already carries loss_rank")
    require(len(native) in (12, 21), ("unexpected native loss-key count", len(native)))

    term1 = m["sr"].SelectionRankTerm("A1", seed=0, k=300).bind_manual(cache, consumer=m["geo"])
    criterion.selection_rank = term1
    rng_before = torch.random.get_rng_state()
    control = criterion(m["fp32_outputs"](out), targets)
    rng_after = torch.random.get_rng_state()
    require(torch.equal(rng_before, rng_after), "the A1 draw consumed the training RNG stream")
    require(len(control) == len(native) + 1
            and abs(float(control["loss_rank"]) - term1.last_stats["loss"]) < 1e-12,
            "the A1 term key/value differs")
    paired_instances = sum(1 for row in term1.last_stats["per_image"]
                           for record in row["records"] if record["partner"] is not None)
    require(term1.mode == "A1" and term1.draws == paired_instances,
            ("the A1 draw count does not match the instances with an eligible set",
             term1.draws, paired_instances))

    term2 = m["sr"].SelectionRankTerm("A2", seed=0, k=300).bind_manual(cache, consumer=m["geo"])
    criterion.selection_rank = term2
    rng_before = torch.random.get_rng_state()
    method = criterion(m["fp32_outputs"](out), targets)
    rng_after = torch.random.get_rng_state()
    require(torch.equal(rng_before, rng_after), "the A2 call consumed the training RNG stream")
    criterion.selection_rank = term

    for key in ("instances", "pairs", "zero_pair_instances", "choice_space_instances"):
        require(term1.last_stats[key] == term2.last_stats[key],
                ("A1 and A2 eligibility/counts differ", key, term1.last_stats[key], term2.last_stats[key]))
    require(all(abs(a["margin"] - b["margin"]) < 1e-12
                for row_a, row_b in zip(term1.last_stats["per_image"], term2.last_stats["per_image"])
                for a, b in zip(row_a["records"], row_b["records"])),
            "A1 and A2 margins differ")
    shared = [key for key in native if key in method]
    extra = sorted(set(method) - set(native))
    require(extra == ["loss_rank"], ("installation added something else", extra))
    require(all(torch.equal(native[key], method[key]) for key in shared),
            "installation changed a native loss value")
    compared = 0
    disagreements = 0
    for row_a, row_b in zip(term1.last_stats["per_image"], term2.last_stats["per_image"]):
        for a, b in zip(row_a["records"], row_b["records"]):
            if a["partner"] is None and b["partner"] is None:
                continue
            compared += 1
            if a["partner"] != b["partner"]:
                disagreements += 1

    state = term1.state_dict()
    buffer = io.BytesIO()
    torch.save(state, buffer)
    saved_bytes = len(buffer.getvalue())
    buffer.seek(0)
    reloaded = torch.load(buffer, weights_only=True)
    draw_a = int(torch.randint(2 ** 31, (1,), generator=term1.generator).item())
    term1.load_state_dict(reloaded)
    draw_b = int(torch.randint(2 ** 31, (1,), generator=term1.generator).item())
    require(draw_a == draw_b, "the A1 generator state did not round-trip exactly")
    require(reloaded["draws"] == state["draws"] and reloaded["mode"] == "A1"
            and torch.equal(reloaded["a1_generator"], state["a1_generator"]),
            "the A1 state through torch.save/load differs")
    print("PASS3", json.dumps({"native_keys": len(native), "extra": extra, "draws": term1.draws,
                               "disagreements": disagreements, "compared": compared}), flush=True)

    timing = {}
    with torch.no_grad():
        scores = pool_logits.max(dim=-1).values
        torch.topk(scores, 300, dim=1)
        tick = time.perf_counter()
        for _ in range(20):
            torch.topk(scores, 300, dim=1)
        timing["selection_topk_ms_per_call"] = round((time.perf_counter() - tick) / 20 * 1000.0, 4)
    timing["term_ms_per_microbatch"] = [row["milliseconds"] for row in window1_history + empty_history]

    report = {
        "status": "PASS",
        "scope": ("CPU production-path verification of the r15 selection-ranking term against the staged "
                  "release-003 tree; no gate, no release, no training round, no scoring"),
        "source_root": str(root),
        "contract": {"path": str(Path(args.contract).resolve()), "task_id": contract["task_id"],
                     "contract_sha256": contract["contract_sha256"], "plan_sha256": contract["plan_sha256"],
                     "parent_checkpoint_sha256": parent["sha256"]},
        "config": {"input_hw": [768, 1344], "physical_batch_size": 1, "amp_dtype": "float32",
                   "synthetic": True, "note": "declared CPU verification configuration, not the production recipe"},
        "window": {"epoch_clock": source_epoch, "images": [row["image_id"] for row in traces],
                   "gt_per_image": [int(len(row["labels"])) for row in targets]},
        "installation": {"parameters_unchanged": True, "buffers_unchanged": True, "layout_unchanged": True,
                         "deepcopy_rebinds_wrapper": True, "new_parameters": 0, "new_buffers": 0},
        "window1": {"loss_key_counts": counts1, "loss_rank_values": [row["loss"] for row in window1_history],
                    "headline": headline, "gradient_checks": gradient_checks,
                    "gradient_norm_before_clip": r1["gradient_norm_before_clip"]},
        "window2_empty": {"loss_key_counts": counts2, "loss_rank_values": [row["loss"] for row in empty_history],
                          "unselected_pool_tokens_touched": 0},
        "pass3": {"mapping": mapping, "native_key_count": len(native), "installed_extra_keys": extra,
                  "native_values_bitwise_unchanged": True,
                  "counts_identical_A1_A2": {key: term2.last_stats[key] for key in
                                             ("instances", "pairs", "zero_pair_instances", "choice_space_instances")},
                  "partner_compared": compared, "partner_disagreements_A1_vs_A2": disagreements,
                  "a1_draws": term1.draws, "a1_draws_expected": paired_instances,
                  "training_rng_untouched": True},
        "a1_state": {"generator_bytes": int(state["a1_generator"].numel()),
                     "serialized_bytes": saved_bytes, "roundtrip_exact": True, "draws": state["draws"]},
        "per_image_records": term2.last_stats["per_image"],
        "cost": timing,
        "declaration": m["sr"].DECLARATION,
        "test_criteria_correction": {
            "native_counts_observed": [len(native)],
            "installed_counts_observed": counts1 + counts2,
            "patched_checks": ["worker.py full_clock (13/22 with a term, 12/21 otherwise)",
                               "slice_ft_job.py update (same correction)"],
            "untouched_fixture": "slice_finetune_cpu.py keeps its native 21/12 assertion (not reused by a term arm)"},
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "cuda_initialized": bool(torch.cuda.is_initialized()),
    }
    payload_path = out_dir / "PRODUCTION-PATH.json"
    payload_path.write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "window1", "window2_empty", "pass3",
                                                   "a1_state", "cost", "elapsed_seconds")},
                     ensure_ascii=False, default=str), flush=True)
    print("PRODUCTION-PATH-COMPLETE", report["elapsed_seconds"], "s", flush=True)


if __name__ == "__main__":
    main()
