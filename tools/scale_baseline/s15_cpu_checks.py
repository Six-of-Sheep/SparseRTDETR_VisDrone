"""S15 preparation CPU checks (no CUDA, no development scoring, no tuning).

usage (certified interpreter, -B, CUDA hidden, lowest priority):
  python -B tools/scale_baseline/s15_cpu_checks.py --contract R25_FORMAL_CONTRACT --out NEW_DIR

Fixed inputs: R25 e060 parent, train_core epoch-61 window 0 and epoch-118 window 0 (pre-registered,
not chosen by any readout), and development image 0 for one GT-free pixel/forward schema check.
"""
from __future__ import annotations

import os
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
import argparse
import gc
import hashlib
import json
import random
import resource
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools" / "query_local"), str(ROOT)]

import numpy as np
import torch

from sparse_rtdetr.research import scale_baseline as S
from sparse_rtdetr.protocol_r18 import checkpoint as native
from sparse_rtdetr.protocol_r18.engine import learning_rate, tensor_digest
from sparse_rtdetr.protocol_r18.geometry import forward_guards, validate_geometry
from sparse_rtdetr.protocol_r18.model import Config, build
from sparse_rtdetr.baseline.training_v2b import logical_batch_indices
from sparse_rtdetr.localization.data import collate_window

FIXTURE = Config(seed=0, input_hw=S.S15_HW, physical_batch_size=1, amp_dtype="float32")
CHECK_EPOCHS = (61, 118)


def setup():
    torch.set_num_threads(int(os.environ.get("S15_CPU_THREADS", "4")))
    torch.use_deterministic_algorithms(True)
    random.seed(0); np.random.seed(0); torch.manual_seed(0)


def cost(t0, c0):
    return {"wall_seconds": time.perf_counter() - t0, "cpu_seconds": time.process_time() - c0,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def publish(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=1, allow_nan=False, sort_keys=True)
        f.write("\n")


def expect_error(fn, label):
    try:
        fn()
    except Exception as exc:  # the identity guard must reject
        return {"label": label, "rejected": True, "error": f"{type(exc).__name__}: {str(exc)[:240]}"}
    raise AssertionError("S15 guard accepted an invalid input: " + label)


# ------------------------------------------------------------------ C1 same draw, two canvases
def augmentation_consistency(binding):
    t0, c0 = time.perf_counter(), time.process_time()
    small = S.train_dataset(binding, S.PARENT_HW)
    large = S.train_dataset(binding, S.S15_HW)
    rows, seconds = [], {"768x1344": 0., "1152x2016": 0.}
    for epoch in CHECK_EPOCHS:
        indices = next(logical_batch_indices(len(small), seed=0, epoch=epoch))
        for offset, index in enumerate(indices):
            key = (epoch, offset, index)
            t = time.perf_counter(); a = small[key]; seconds["768x1344"] += time.perf_counter() - t
            t = time.perf_counter(); b = large[key]; seconds["1152x2016"] += time.perf_counter() - t
            (pa, ta, ra), (pb, tb, rb) = a, b
            S.require(tuple(pa.shape) == (3, *S.PARENT_HW) and tuple(pb.shape) == (3, *S.S15_HW), "canvas shape")
            S.require(ra["sample_seed"] == rb["sample_seed"] and ra["geometry_steps"] == rb["geometry_steps"]
                      and ra["augmentation_strong"] == rb["augmentation_strong"], "draws differ across canvas")
            common = [g for g in ta["gt_ids"] if g in set(tb["gt_ids"])]
            ia = [ta["gt_ids"].index(g) for g in common]; ib = [tb["gt_ids"].index(g) for g in common]
            box_diff = float((ta["boxes"][ia] - tb["boxes"][ib]).abs().max()) if common else 0.
            S.require(ta["labels"][ia].tolist() == tb["labels"][ib].tolist(), "labels differ")
            ign_same = ta["ignore_ids"] == tb["ignore_ids"]
            ign_diff = (float((ta["ignore_spatial_boxes"] - tb["ignore_spatial_boxes"]).abs().max())
                        if ign_same and len(ta["ignore_spatial_boxes"]) else 0.)
            rows.append({"epoch": epoch, "position": offset, "image_id": ra["image_id"],
                         "geometry_steps": ra["geometry_steps"], "GT_768": len(ta["gt_ids"]), "GT_1152": len(tb["gt_ids"]),
                         "only_768": sorted(set(ta["gt_ids"]) - set(tb["gt_ids"])),
                         "only_1152": sorted(set(tb["gt_ids"]) - set(ta["gt_ids"])),
                         "max_abs_normalized_box_diff": box_diff, "ignore_ids_equal": ign_same,
                         "max_abs_normalized_ignore_diff": ign_diff})
    return {"samples": len(rows), "rows": rows,
            "GT_only_768_total": sum(len(r["only_768"]) for r in rows),
            "draws_identical": True,
            "GT_768_total": sum(r["GT_768"] for r in rows), "GT_1152_total": sum(r["GT_1152"] for r in rows),
            "GT_only_1152_total": sum(len(r["only_1152"]) for r in rows),
            "max_abs_normalized_box_diff": max(r["max_abs_normalized_box_diff"] for r in rows),
            "max_abs_normalized_ignore_diff": max(r["max_abs_normalized_ignore_diff"] for r in rows),
            "all_ignore_ids_equal": all(r["ignore_ids_equal"] for r in rows),
            "single_process_seconds_per_image": {k: v / len(rows) for k, v in seconds.items()},
            "explanation": ("augmentation random draws happen before the final Resize; the post-resize "
                            "SanitizeBoundingBoxes(min_size=1) is in canvas pixels, so a larger canvas can keep extra tiny GT"),
            "cost": cost(t0, c0)}


# ------------------------------------------------------------------ C1b edge rounding census (no augmentation)
def edge_rounding_census(binding):
    """Exact native float32 box resize + SanitizeBoundingBoxes bound test for every train_core GT,
    as in epochs 118-120 (no zoom/crop).  Flip is symmetric for the bound test only approximately,
    so both orientations are counted."""
    import numpy as np
    from sparse_rtdetr.localization.data import parse_raw, _clip, _valid
    t0, c0 = time.perf_counter(), time.process_time()
    data = S.train_dataset(binding, S.PARENT_HW)
    out = {}
    for label, (ch, cw) in (("768x1344", S.PARENT_HW), ("1152x2016", S.S15_HW)):
        out[label] = {"GT": 0, "dropped_bound_unflipped": 0, "dropped_bound_flipped": 0,
                      "dropped_min_size_or_area": 0, "images_with_bound_drop": 0, "by_image_size": {}}
    for row in data.records:
        _, raw = data.bound_bytes(row)
        t = parse_raw(raw, row["stable_image_id"], expected_rows=row["raw_annotation_rows"])
        h, w = row["height"], row["width"]
        gt = _clip(np.array([r["box"] for r in t["gt"]], dtype=np.float64).reshape(-1, 4), (h, w))
        gt = gt[_valid(gt)]
        base = torch.tensor(gt, dtype=torch.float32)
        ws, hs = base[:, 2] - base[:, 0], base[:, 3] - base[:, 1]
        pre = (ws >= 1) & (hs >= 1) & (ws * hs >= 1)
        flipped = base.clone(); flipped[:, [0, 2]] = w - base[:, [2, 0]]
        for label, (ch, cw) in (("768x1344", S.PARENT_HW), ("1152x2016", S.S15_HW)):
            ratios = torch.tensor([cw / w, ch / h, cw / w, ch / h])
            rec = out[label]
            n_drop = []
            for name, boxes in (("unflipped", base), ("flipped", flipped)):
                b = boxes[pre].mul(ratios).to(torch.float32)
                bw, bh = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
                size_ok = (bw >= 1) & (bh >= 1) & (bw * bh >= 1) & (b >= 0).all(-1)
                bound_ok = (b[:, 0] <= cw) & (b[:, 2] <= cw) & (b[:, 1] <= ch) & (b[:, 3] <= ch)
                drop = int((size_ok & ~bound_ok).sum())
                rec["dropped_bound_" + name] += drop
                n_drop.append(drop)
                if name == "unflipped":
                    rec["dropped_min_size_or_area"] += int((~size_ok).sum())
            rec["GT"] += int(pre.sum())
            rec["images_with_bound_drop"] += int(max(n_drop) > 0)
            key = f"{w}x{h}"
            bucket = rec["by_image_size"].setdefault(key, {"images": 0, "GT": 0, "bound_drops_unflipped": 0, "bound_drops_flipped": 0})
            bucket["images"] += 1; bucket["GT"] += int(pre.sum())
            bucket["bound_drops_unflipped"] += n_drop[0]; bucket["bound_drops_flipped"] += n_drop[1]
    return {"by_canvas": out, "rule": "torchvision 0.19.1 resize_bounding_boxes float32 ratio; sanitize requires x2<=W, y2<=H",
            "cost": cost(t0, c0)}


# ------------------------------------------------------------------ C1c strong-augmentation sample (epoch 61, fixed windows 0-19)
def strong_sample(binding, windows_count=20):
    t0, c0 = time.perf_counter(), time.process_time()
    small = S.train_dataset(binding, S.PARENT_HW)
    large = S.train_dataset(binding, S.S15_HW)
    gen = logical_batch_indices(len(small), seed=0, epoch=61)
    totals = {"samples": 0, "GT_768": 0, "GT_1152": 0, "only_768": 0, "only_1152": 0}
    for wi in range(windows_count):
        for offset, index in enumerate(next(gen)):
            key = (61, wi * 16 + offset, index)
            ta, tb = small[key][1], large[key][1]
            sa, sb = set(ta["gt_ids"]), set(tb["gt_ids"])
            totals["samples"] += 1; totals["GT_768"] += len(sa); totals["GT_1152"] += len(sb)
            totals["only_768"] += len(sa - sb); totals["only_1152"] += len(sb - sa)
    totals["cost"] = cost(t0, c0)
    return totals


# ------------------------------------------------------------------ C2 real two-worker entry
def real_loader(binding, a0_first_update):
    from formal_data import PARENT_UPDATE, check_boundaries, windows
    t0, c0 = time.perf_counter(), time.process_time()
    data = S.train_dataset(binding, S.S15_HW)
    batches = list(windows(data, PARENT_UPDATE, limit=1))
    images, targets, audit = batches[0]
    S.require(tuple(images.shape) == (16, 3, *S.S15_HW) and images.dtype == torch.float32, "S15 window tensor")
    same = {k: audit[k] == a0_first_update[k] for k in ("update", "epoch", "window", "image_ids_sha256",
                                                        "augmentation_seeds_sha256", "augmentation_strong")}
    S.require(all(same.values()), ("S15 order/seed identity differs from R25 A0", same))
    differs = {k: audit[k] != a0_first_update[k] for k in ("gt_ids_sha256", "augmentation_trace_sha256",
                                                           "input_tensor_sha256", "target_sha256")}
    first = cost(t0, c0)
    t1, c1 = time.perf_counter(), time.process_time()
    boundaries = check_boundaries(data)
    return images, targets, {"first_window": audit, "equal_to_R25_A0_update_18241": same,
                             "canvas_dependent_fields_differ": differs, "first_window_cost": first,
                             "boundaries": boundaries, "boundaries_cost": cost(t1, c1)}


# ------------------------------------------------------------------ C3 production parent bridge
def production_bridge(parent):
    t0, c0 = time.perf_counter(), time.process_time()
    e, checks = S.create(parent, "cpu")
    st = e.state_dict()
    e.validate_state_dict(st)
    S.require(st["config"]["input_hw"] == list(S.S15_HW) and st["completed_updates"] == S.PARENT_UPDATE
              and st["next_epoch"] == 61 and st["next_window"] == 0, "S15 engine clock")
    S.require(e.config.physical_batch_size == 8 and e.config.accumulation_steps == 2
              and e.config.amp_dtype == "bfloat16" and e.config.clip_max_norm == .1, "S15 production math differs")
    bn = [m for m in e.model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    report = {"checks": checks, "engine_state": {k: st[k] for k in ("completed_updates", "next_epoch", "next_window")},
              "lr_update_18241": learning_rate(18241), "binding": S.binding(e.config),
              "batchnorm": {"modules": len(bn), "types": sorted({type(m).__name__ for m in bn}),
                            "affine_trainable": all(m.weight.requires_grad for m in bn if m.weight is not None),
                            "track_running_stats": all(m.track_running_stats for m in bn),
                            "frozen": False, "note": "R18 config freeze_at -1, freeze_norm False; BN uses each 8-image micro-batch"},
              "denoising": {"num_denoising": e.model.decoder.num_denoising,
                            "note": "DN groups follow the per-micro-batch max GT count; CUDA RNG draws follow the GT count"},
              "cost": cost(t0, c0)}
    del e; gc.collect()
    return report


# ------------------------------------------------------------------ C4 full-size tensors, save/restore, exact continuation
def full_size_updates(parent, images, targets, out):
    t0, c0 = time.perf_counter(), time.process_time()
    e, _ = S.create(parent, "cpu", FIXTURE, fixture=True)
    u1 = e.train_window(images[0:2], targets[0:2], capture=True)
    saved = S.save(e, out / "fixture-u1.pth")
    u2 = e.train_window(images[2:4], targets[2:4], capture=True)
    after = S.capture(e)
    del e; gc.collect()
    r, _ = S.create(parent, "cpu", FIXTURE, fixture=True)
    restored = S.restore(r, out / "fixture-u1.pth", saved["sha256"])
    u2r = r.train_window(images[2:4], targets[2:4], capture=True)
    exact = native.exact_equal(S.capture(r), after)
    S.require(exact and u2r["loss"] == u2["loss"] and u2r["gradient_sha256_before_clip"] == u2["gradient_sha256_before_clip"],
              "S15 restored continuation is not bitwise identical")
    del r; gc.collect()
    plain = cost(t0, c0)
    t1, c1 = time.perf_counter(), time.process_time()
    k, _ = S.create(parent, "cpu", FIXTURE, fixture=True, activation_checkpoint=True)
    S.restore(k, out / "fixture-u1.pth", saved["sha256"])
    u2k = k.train_window(images[2:4], targets[2:4], capture=True)
    with_checkpoint = S.capture(k)
    with_checkpoint["activation_checkpoint"] = after["activation_checkpoint"]
    exact_ckpt = native.exact_equal(with_checkpoint, after)
    S.require(exact_ckpt and u2k["gradient_sha256_before_clip"] == u2["gradient_sha256_before_clip"],
              "activation checkpoint changes the S15 update")
    del k; gc.collect()
    keep = ("update", "epoch", "window", "lr", "loss", "gradient_norm_before_clip", "gradient_sha256_before_clip",
            "output_sha256", "dn_noise_sha256", "elapsed_seconds")
    return {"fixture": FIXTURE.binding(), "u1": {k: u1[k] for k in keep}, "u2": {k: u2[k] for k in keep},
            "u2_after_restore": {k: u2r[k] for k in keep}, "u2_activation_checkpoint": {k: u2k[k] for k in keep},
            "saved_u1": saved, "restore": restored, "continuation_bitwise_exact": exact,
            "activation_checkpoint_bitwise_exact": exact_ckpt,
            "cost_plain": plain, "cost_checkpoint": cost(t1, c1)}


# ------------------------------------------------------------------ C5 activation bytes (estimate only)
def saved_bytes(parent, images, targets):
    """Autograd saved-tensor bytes of one FP32 bs1 forward at both canvases; CPU estimate, not CUDA."""
    from sparse_rtdetr.research.query_local_runtime import create as r25_create
    from sparse_rtdetr.protocol_r18.model import fp32_outputs
    from sparse_rtdetr.baseline.training_v2b_engine import _move
    results = {}
    small_cfg = Config(seed=0, input_hw=S.PARENT_HW, physical_batch_size=1, amp_dtype="float32")
    small = torch.nn.functional.interpolate(images[0:1], size=S.PARENT_HW, mode="bilinear", align_corners=False)
    for label, maker, x in (("768x1344", lambda: r25_create(parent, "A0", "cpu", config=small_cfg, synthetic=True)[0], small),
                            ("1152x2016", lambda: S.create(parent, "cpu", FIXTURE, fixture=True)[0], images[0:1])):
        e = maker(); e.model.train()
        seen, total, inside, depth = set(), [0], [0], [0]
        hooks = []
        for path in S.CHECKPOINT_SEGMENTS:
            owner = e.model
            for part in path.split("."):
                owner = getattr(owner, part)
            for block in owner:
                hooks.append(block.register_forward_pre_hook(lambda *_: depth.__setitem__(0, depth[0] + 1)))
                hooks.append(block.register_forward_hook(lambda *_: depth.__setitem__(0, depth[0] - 1)))

        def pack(t):
            s = t.untyped_storage()
            key = (s.data_ptr(), s.nbytes())
            if key not in seen:
                seen.add(key); total[0] += s.nbytes()
                if depth[0] > 0:
                    inside[0] += s.nbytes()
            return t
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            out = e.model(x, targets=_move(targets[0:1], e.device))
            losses = e.criterion(fp32_outputs(out), _move(targets[0:1], e.device))
        for h in hooks:
            h.remove()
        results[label] = {"saved_bytes": total[0], "inside_checkpoint_segments_bytes": inside[0],
                          "loss_finite": bool(torch.isfinite(sum(losses.values())))}
        del e, out, losses; gc.collect()
    ratio = results["1152x2016"]["saved_bytes"] / results["768x1344"]["saved_bytes"]
    keep = 1 - results["1152x2016"]["inside_checkpoint_segments_bytes"] / results["1152x2016"]["saved_bytes"]
    return {"by_canvas": results, "S15_over_A0_saved_ratio": ratio,
            "fraction_outside_checkpoint_segments_at_S15": keep,
            "note": "FP32 bs1 CPU autograd storage; BF16/CUDA allocator, workspaces and fragmentation are not modelled"}


# ------------------------------------------------------------------ C6 GT-free evaluation entry
def evaluation_entry(parent, binding, ids):
    from infer_seed import load_original
    from sparse_rtdetr.r0.evaluation import rank_candidates
    from torchvision.transforms.v2 import functional as TF
    from PIL import Image
    data = S.development(binding, S.S15_HW)
    S.require([r["stable_image_id"] for r in data.records] == ids, "S15 development members/order")
    row = data.records[0]
    with load_original(data, row) as image:
        x = TF.pil_to_tensor(data.transform.resize(image)).float() / 255.
        direct = TF.pil_to_tensor(TF.resize(image, list(S.S15_HW))).float() / 255.
    S.require(tuple(x.shape) == (3, *S.S15_HW) and torch.equal(x, direct), "S15 GT-free pixels differ from one direct resize")
    e, _ = S.create(parent, "cpu", FIXTURE, fixture=True)
    model = e.ema.module.eval()
    validate_geometry(model)
    with torch.inference_mode(), forward_guards(model):
        result = model(x[None])
    S.require(result["pred_boxes"].shape == (1, 300, 4) and result["pred_logits"].shape == (1, 300, 10), "S15 eval schema")
    ranked = rank_candidates(result["pred_logits"][0].float(), result["pred_boxes"][0].float(), row, 600)
    S.require(len(ranked) == 600, "S15 ranking count")
    return {"pixels_shape": list(x.shape), "pixels_equal_direct_resize": True, "eval_output_schema": True,
            "ranked_600": len(ranked), "original_hw": [row["height"], row["width"]],
            "note": "GT-free schema check on one image; no prediction file, no scoring"}


# ------------------------------------------------------------------ C7 cross-size / recipe rejection
def rejections(parent, out, saved_sha):
    from sparse_rtdetr.research.query_local_runtime import create as r25_create
    a0, _ = r25_create(parent, "A0", "cpu")  # the native 768x1344 production engine
    a0.s15_activation_checkpoint = False
    s15_model, _, _ = build(S.production_config(), synthetic=True)
    parent_state = torch.load(parent, map_location="cpu", weights_only=True)["model"]
    tampered = dict(parent_state); tampered["decoder.anchors"] = tampered["decoder.anchors"].clone()
    finite = torch.isfinite(tampered["decoder.anchors"]).reshape(-1).nonzero()[0, 0]
    tampered["decoder.anchors"].view(-1)[finite] += 1e-3  # a valid (finite) anchor; masked ones are +inf
    fixture_s15, _ = S.create(parent, "cpu", FIXTURE, fixture=True)
    rows = [
        expect_error(lambda: S.restore(a0, out / "fixture-u1.pth", saved_sha), "S15 state into native 768x1344 engine"),
        expect_error(lambda: s15_model.load_state_dict(parent_state, strict=True), "raw 768x1344 parent caches into S15 model"),
        expect_error(lambda: S._bridge_state(tampered, s15_model, "tampered"), "parent cache not native 768x1344"),
        expect_error(lambda: S.validate_production(Config(seed=0)), "native canvas as S15 production"),
        expect_error(lambda: S.validate_production(Config(seed=0, input_hw=S.S15_HW, physical_batch_size=4)),
                     "silent 4x4 micro-batch"),
        expect_error(lambda: S.validate_production(Config(seed=0, input_hw=(1024, 1792))), "other canvas"),
        expect_error(lambda: fixture_s15.train_window(torch.zeros(2, 3, *S.PARENT_HW), [{}, {}]), "768x1344 window into S15 engine"),
        expect_error(lambda: a0.load_state_dict(fixture_s15.state_dict()), "S15 engine clock/config into 768x1344 engine"),
    ]
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--contract", required=True); p.add_argument("--out", required=True)
    args = p.parse_args()
    S.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CUDA must be hidden")
    setup()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    c = json.loads(Path(args.contract).read_bytes())
    parent = c["parent"]["path"]
    a0_first = json.loads((Path(c["attempt_root"]) / "train-A0" / "UPDATES.jsonl").open().readline())["input"]
    report = {"task": "NEXT_TASK_SCALE_BASELINE_001 preparation CPU checks", "source_root": str(ROOT),
              "contract": {"path": args.contract, "sha256": native.file_sha(args.contract)},
              "parent": {"path": parent, "sha256": native.file_sha(parent)}, "torch": torch.__version__,
              "threads": torch.get_num_threads()}
    t0, c0 = time.perf_counter(), time.process_time()
    try:
        report["C1_augmentation_consistency"] = augmentation_consistency(c["data"]["train_core"])
        print("C1 PASS", flush=True)
        report["C1b_edge_rounding_census"] = edge_rounding_census(c["data"]["train_core"])
        print("C1b PASS", flush=True)
        report["C1c_strong_augmentation_sample"] = strong_sample(c["data"]["train_core"])
        print("C1c PASS", flush=True)
        images, targets, report["C2_real_loader"] = real_loader(c["data"]["train_core"], a0_first)
        print("C2 PASS", flush=True)
        report["C3_production_bridge"] = production_bridge(parent)
        print("C3 PASS", flush=True)
        report["C4_full_size_updates"] = full_size_updates(parent, images, targets, out)
        print("C4 PASS", flush=True)
        report["C5_saved_tensor_bytes"] = saved_bytes(parent, images, targets)
        print("C5 PASS", flush=True)
        report["C6_evaluation_entry"] = evaluation_entry(parent, c["data"]["development"], c["development_ids"])
        print("C6 PASS", flush=True)
        report["C7_rejections"] = rejections(parent, out, report["C4_full_size_updates"]["saved_u1"]["sha256"])
        print("C7 PASS", flush=True)
        S.require(not torch.cuda.is_initialized(), "CUDA initialized")
        report.update(status="PASS", cuda_initialized=False, model_forwards_development=1, scoring_calls=0,
                      total=cost(t0, c0))
        publish(out / "CPU_CHECKS.json", report)
    except Exception:
        report.update(status="STOP", traceback=traceback.format_exc(), total=cost(t0, c0))
        publish(out / "CPU_CHECKS_STOP.json", report)
        raise


if __name__ == "__main__":
    main()
