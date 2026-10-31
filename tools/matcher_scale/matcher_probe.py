"""Carrier-side decision diagnostic for the size-relative matching cost (CPU, read-only).

Runs the frozen common-protocol seed0 e120 endpoint on the same 16 train-side
development images the r10 D1 selection diagnostic used, computes the three
matching cost matrices (native, per-GT relative, batch-mean relative) with the
frozen 2/5/2 focal matcher weights, solves all three Hungarian assignments, and
reports how ownership changes and whether the paired IoU of the assigned tokens
moves in the declared direction.

Pre-registered screening rules (declared before this probe was run, see the
single exploration record section 15):
  F1  the fraction of small GTs (raw area <= 1024) whose owning token changes
      under 'rel' must be >= 5% -- otherwise the mechanism has no decision
      difference on this carrier and the round stops before any GPU work;
  F2  the median change of the paired IoU among those changed small GTs must be
      strictly positive -- otherwise the ownership change is not in the declared
      direction and the round stops.
The control's numbers are reported alongside; the probe is evidence, not a
science result and not a score.

usage: CUDA_VISIBLE_DEVICES= <frozen python> -I matcher_probe.py
       --source-root <e088 release-001/source> --contract <e120 execution-contract-001.json>
       --checkpoint <checkpoint-epoch-120.pth> --split <stage0>/SPLIT_SEQUENCES.json
       [--source-manifest <e088 source-manifest.json>] --images 16 --out <dir>
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import frozen_r18  # noqa: E402

matcher_scale = None  # bound inside main() from the frozen release package

FROZEN_IMAGE_PREFIXES = ("00e9facd8565", "02a34eb8a81b", "02d411276904", "03b327dd752d", "03b4f0b1dc7e",
                         "03d569ae67c6", "05759bbe3643", "0636d25253b9", "06d0c5aa00dd", "07c21db371c3",
                         "082bd0c17805", "09d9898ec2cc", "09fe8dd3d06f", "0cd346337bb1", "0dcbb8966bb9",
                         "0e7b3e8b1475")
WEIGHTS = {"cost_bbox": 5.0, "cost_class": 2.0, "cost_giou": 2.0, "alpha": 0.25, "gamma": 2.0}
SMALL_MAX_AREA = 1024.0
MEDIUM_MAX_AREA = 9216.0
MODES = ("off", "rel", "ctrl")
RULES = {"F1_small_ownership_change_frac_min": 0.05,
         "F2_median_paired_iou_change_of_changed_small_gt_gt": 0.0}


def r6(value):
    return None if value is None else round(float(value), 6)


def size_band(area):
    if area <= SMALL_MAX_AREA:
        return "small"
    return "medium" if area <= MEDIUM_MAX_AREA else "large"


def cxcywh_to_xyxy(boxes):
    return np.concatenate([boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2], axis=1)


def xyxy_iou(a, b):
    low = np.maximum(a[:, None, :2], b[None, :, :2])
    high = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(high - low, 0, None).prod(-1)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--source-manifest", default=None)
    parser.add_argument("--module", required=True,
                        help="path to the staged matcher_scale.py (the mechanism module; it is "
                             "loaded standalone because the model comes from the frozen release tree)")
    parser.add_argument("--images", type=int, default=16)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    torch.set_num_threads(2)
    started = time.monotonic()
    loaded = frozen_r18.load_frozen_cpu(args.source_root, args.contract, args.checkpoint,
                                        args.source_manifest)
    model = loaded["model"]
    dev = frozen_r18.development548(loaded["contract"])
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    assignment_roles = split["assignment"]
    train_rows = [row for row in dev.records if assignment_roles.get(row["sequence_key"]) == "train"]
    frozen_r18.require(len(train_rows) == 338, "train-side development row count differs")
    train_rows.sort(key=lambda row: row["stable_image_id"])
    rows = train_rows[:args.images]
    prefixes = [row["stable_image_id"][:12] for row in rows]
    frozen_r18.require(tuple(prefixes) == FROZEN_IMAGE_PREFIXES,
                       "probe image set differs from the frozen D1 diagnostic: " + ",".join(prefixes))

    global matcher_scale
    import importlib.util
    spec = importlib.util.spec_from_file_location("matcher_scale_standalone", args.module)
    matcher_scale = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(matcher_scale)
    vendor = importlib.import_module(type(loaded["criterion"].matcher).__module__)
    matcher_scale.bind_vendor_names(vendor)
    native_matcher = loaded["criterion"].matcher

    per_band = {band: {"n": 0, "changed": 0, "improved": 0, "worsened": 0, "delta": [],
                       "iou": {m: [] for m in MODES}, "control_changed": 0}
                for band in ("small", "medium", "large")}
    unreachable = {"n": 0, "changed": 0, "improved": 0, "delta": []}
    timing = {"cost_off": 0.0, "cost_rel": 0.0, "cost_ctrl": 0.0, "solve": 0.0, "native_matcher": 0.0}
    images_out = []

    with torch.no_grad():
        for position, row in enumerate(rows):
            pixels, target = frozen_r18.image_pixels_and_target(dev, row, position)
            outputs = model(pixels.unsqueeze(0))
            boxes = outputs["pred_boxes"][0].detach().numpy()
            gt = target["boxes"].numpy()
            area = target["raw_area"].numpy()
            gts = [{"labels": target["labels"], "boxes": target["boxes"]}]
            query_outputs = {"pred_logits": outputs["pred_logits"].detach(),
                             "pred_boxes": outputs["pred_boxes"].detach()}

            solutions = {}
            for mode in MODES:
                tick = time.perf_counter()
                indices, _, _ = matcher_scale.assignment(query_outputs, gts, WEIGHTS, mode)
                timing["cost_" + mode] += time.perf_counter() - tick
                owner = {int(j): int(i) for i, j in zip(indices[0][0].tolist(), indices[0][1].tolist())}
                solutions[mode] = owner
            tick = time.perf_counter()
            native_matcher(query_outputs, gts)
            timing["native_matcher"] += time.perf_counter() - tick

            pred_xyxy = cxcywh_to_xyxy(boxes)
            gt_xyxy = cxcywh_to_xyxy(gt)
            iou_matrix = xyxy_iou(pred_xyxy, gt_xyxy)
            image_row = {"image_id": row["stable_image_id"], "n_gt": len(area), "bands": {}}
            for g in range(len(area)):
                band = size_band(float(area[g]))
                record = per_band[band]
                record["n"] += 1
                native_owner = solutions["off"].get(g)
                rel_owner = solutions["rel"].get(g)
                ctrl_owner = solutions["ctrl"].get(g)
                iou_off = float(iou_matrix[native_owner, g]) if native_owner is not None else 0.0
                iou_rel = float(iou_matrix[rel_owner, g]) if rel_owner is not None else 0.0
                iou_ctrl = float(iou_matrix[ctrl_owner, g]) if ctrl_owner is not None else 0.0
                record["iou"]["off"].append(iou_off)
                record["iou"]["rel"].append(iou_rel)
                record["iou"]["ctrl"].append(iou_ctrl)
                changed = bool(rel_owner != native_owner)
                if changed:
                    record["changed"] += 1
                    record["delta"].append(iou_rel - iou_off)
                    if iou_rel > iou_off:
                        record["improved"] += 1
                    elif iou_rel < iou_off:
                        record["worsened"] += 1
                if ctrl_owner != native_owner:
                    record["control_changed"] += 1
                if band == "small" and iou_off < 0.75:
                    unreachable["n"] += 1
                    if changed:
                        unreachable["changed"] += 1
                        unreachable["delta"].append(iou_rel - iou_off)
                        if iou_rel > iou_off:
                            unreachable["improved"] += 1
            for band, record in per_band.items():
                image_row["bands"][band] = sum(1 for g in range(len(area))
                                               if size_band(float(area[g])) == band)
            image_row["changed_small"] = sum(
                1 for g in range(len(area))
                if size_band(float(area[g])) == "small" and solutions["rel"].get(g) != solutions["off"].get(g))
            images_out.append(image_row)
            print("image %d/%d %s gt=%d changed=%d (%.1fs)" % (
                position + 1, len(rows), row["stable_image_id"][:12], len(area),
                image_row["changed_small"], time.monotonic() - started), flush=True)

    summary = {}
    for band, record in per_band.items():
        deltas = record["delta"]
        summary[band] = {
            "gts": record["n"],
            "ownership_changed": record["changed"],
            "ownership_changed_frac": r6(record["changed"] / record["n"]) if record["n"] else None,
            "control_changed_frac": r6(record["control_changed"] / record["n"]) if record["n"] else None,
            "paired_iou_median": {m: r6(np.median(record["iou"][m])) if record["iou"][m] else None for m in MODES},
            "changed_improved": record["improved"],
            "changed_worsened": record["worsened"],
            "delta_paired_iou_median": r6(np.median(deltas)) if deltas else None,
            "delta_paired_iou_mean": r6(np.mean(deltas)) if deltas else None,
        }
    small = summary["small"]
    delta_small = [d for d in per_band["small"]["delta"]]
    f1 = bool(small["ownership_changed_frac"] is not None
              and small["ownership_changed_frac"] >= RULES["F1_small_ownership_change_frac_min"])
    f2 = bool(delta_small and float(np.median(delta_small)) > RULES["F2_median_paired_iou_change_of_changed_small_gt_gt"])
    unreachable_delta = unreachable["delta"]
    result = {
        "probe": "r18-matcher-scale-decision-cpu-v1",
        "status": "PASS",
        "frozen": {"checkpoint_sha256": loaded["checkpoint_sha256"], "release_root": loaded["release_root"],
                   "release_manifest_sha256": (frozen_r18.file_sha256(args.source_manifest)
                                               if args.source_manifest else None),
                   "binding_scope": loaded["binding"]["scope"], "ema_updates": frozen_r18.FROZEN_EMA_UPDATES},
        "weights": WEIGHTS,
        "image_set": {"count": len(rows), "split": "pdra_train_side", "order": "stable_image_id ascending",
                      "matches_frozen_d1": True, "gt_rows": "evaluation-transform target rows (all raw rows, as in D1)"},
        "rules": RULES,
        "summary": summary,
        "small_native_unreachable_0.75": {
            "gts": unreachable["n"], "ownership_changed": unreachable["changed"],
            "changed_frac": r6(unreachable["changed"] / unreachable["n"]) if unreachable["n"] else None,
            "changed_improved": unreachable["improved"],
            "delta_paired_iou_median": r6(np.median(unreachable_delta)) if unreachable_delta else None,
        },
        "decisions": {"F1_small_change_frac": f1, "F2_small_delta_median_positive": f2,
                      "screening_passed": bool(f1 and f2)},
        "timing_seconds": {k: r6(v) for k, v in timing.items()},
        "images": images_out,
        "elapsed_seconds": r6(time.monotonic() - started),
        "cuda_initialized": bool(torch.cuda.is_initialized()),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "MATCHER_PROBE.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"summary": summary, "unreachable": result["small_native_unreachable_0.75"],
                      "decisions": result["decisions"], "timing": result["timing_seconds"]},
                     indent=1, ensure_ascii=False))
    print("PROBE_COMPLETE", result["elapsed_seconds"], "s")


if __name__ == "__main__":
    main()
