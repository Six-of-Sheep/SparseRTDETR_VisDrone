"""CPU selection/coverage probe on the frozen common-protocol R18 (read-only).

For a fixed set of development images (first N of the train side of the PDRA
sequence split, ordered by stable_image_id) this probe runs the production
evaluation transform and forward and records, through the production decoder
path:

  * per-level spatial shapes, registered anchors / valid mask,
  * the encoder selection logits and unactivated boxes before top-k, plus the
    chosen top-k indices recomputed with the production rule (required to
    reproduce the produced gather bitwise),
  * final decoder outputs and the native Hungarian matcher pairs,

then reports per-GT coverage / ranking / size statistics by raw-area band and
per-level composition.  No AP/AR is computed, no training happens, and CUDA is
never initialized.

Usage (from the worktree root, frozen interpreter, CUDA hidden; flags are
--source-root, --contract, --checkpoint, --split, optional --source-manifest,
--images and --out):

  CUDA_VISIBLE_DEVICES= python -I tools/explore/selection_probe.py [flags]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import frozen_r18  # noqa: E402

SIZE_BANDS = (("small", 0.0, 1024.0), ("medium", 1024.0, 9216.0),
              ("large", 9216.0, float("inf")))
BAND_NAMES = tuple(name for name, _, _ in SIZE_BANDS)
IOU_THRESHOLDS = (0.5, 0.75)


def r6(value):
    return round(float(value), 6)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def size_band(area):
    for name, low, high in SIZE_BANDS:
        if low < area <= high:
            return name
    return "large"


def cxcywh_to_xyxy(boxes):
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], axis=1)


def xyxy_iou(pred_xyxy, gt_xyxy):
    lt = np.maximum(gt_xyxy[:, None, :2], pred_xyxy[None, :, :2])
    rb = np.minimum(gt_xyxy[:, None, 2:], pred_xyxy[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_p = (pred_xyxy[:, 2] - pred_xyxy[:, 0]) * (pred_xyxy[:, 3] - pred_xyxy[:, 1])
    area_g = (gt_xyxy[:, 2] - gt_xyxy[:, 0]) * (gt_xyxy[:, 3] - gt_xyxy[:, 1])
    union = area_g[:, None] + area_p[None, :] - inter
    return inter / np.clip(union, 1e-12, None)


def pair_iou(box_a, box_b):
    lt = np.maximum(box_a[:2], box_b[:2])
    rb = np.minimum(box_a[2:], box_b[2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = float(wh[0] * wh[1])
    area_a = float((box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = float((box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    return inter / max(area_a + area_b - inter, 1e-12)


def ranks(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ordinal = np.empty(len(values), dtype=np.float64)
    ordinal[order] = np.arange(len(values), dtype=np.float64)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts), dtype=np.float64)
    np.add.at(sums, inverse, ordinal)
    return (sums / counts)[inverse]


def spearman(x, y):
    if len(x) < 8:
        return float("nan")
    rx, ry = ranks(x), ranks(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def capture_forward(model, pixels):
    decoder = model.decoder
    captured = {}
    original_select = decoder._select_topk
    original_decoder_input = decoder._get_decoder_input

    def select(memory, outputs_logits, outputs_coords_unact, topk):
        captured["logits"] = outputs_logits.detach().clone()
        captured["coords_unact"] = outputs_coords_unact.detach().clone()
        result = original_select(memory, outputs_logits, outputs_coords_unact, topk)
        captured["gathered_coords"] = result[2].detach().clone()
        return result

    def decoder_input(*args, **kwargs):
        if len(args) >= 2:
            captured["spatial_shapes"] = [list(shape) for shape in args[1]]
        return original_decoder_input(*args, **kwargs)

    decoder._select_topk = select
    decoder._get_decoder_input = decoder_input
    try:
        with torch.no_grad():
            outputs = model(pixels.unsqueeze(0))
    finally:
        del decoder._select_topk
        del decoder._get_decoder_input
    return captured, outputs


def native_matcher(release_root):
    import importlib
    try:
        from sparse_rtdetr.baseline.config import _vendor_path
    except ImportError:
        from sparse_rtdetr.baseline.training_v2b import _vendor_path
    with _vendor_path(Path(release_root) / "vendor" / "rtdetrv2_pytorch"):
        matcher_module = importlib.import_module("src.zoo.rtdetr.matcher")
        box_ops = importlib.import_module("src.zoo.rtdetr.box_ops")
    matcher = matcher_module.HungarianMatcher({"cost_class": 2, "cost_bbox": 5, "cost_giou": 2},
                                              use_focal_loss=True, alpha=0.25, gamma=2.0)
    return matcher, box_ops


def matcher_records(matcher, box_ops, outputs, target, gt_xyxy, raw_area):
    boxes = outputs["pred_boxes"][0]
    out_xyxy = cxcywh_to_xyxy(boxes.numpy())
    gt_boxes = target["boxes"]
    result = matcher({"pred_logits": outputs["pred_logits"], "pred_boxes": outputs["pred_boxes"]},
                     [{"labels": target["labels"], "boxes": gt_boxes}])
    src, tgt = result["indices"][0]
    if len(src) == 0:
        return []
    probs = torch.sigmoid(outputs["pred_logits"][0])
    p = probs[:, target["labels"]]
    neg = (1 - 0.25) * (p ** 2.0) * (-(1 - p + 1e-8).log())
    pos = 0.25 * ((1 - p) ** 2.0) * (-(p + 1e-8).log())
    cost_class = pos - neg
    cost_bbox = torch.cdist(boxes, gt_boxes, p=1)
    cost_giou = -box_ops.generalized_box_iou(box_ops.box_cxcywh_to_xyxy(boxes),
                                             box_ops.box_cxcywh_to_xyxy(gt_boxes))
    rows = []
    for i, j in zip(src.tolist(), tgt.tolist()):
        rows.append({
            "iou": pair_iou(out_xyxy[i], gt_xyxy[j]),
            "area": float(raw_area[j]),
            "cost_class_w": 2.0 * float(cost_class[i, j]),
            "cost_bbox_w": 5.0 * float(cost_bbox[i, j]),
            "cost_giou_w": 2.0 * float(cost_giou[i, j]),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--source-manifest", default=None)
    parser.add_argument("--images", type=int, default=16)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    torch.set_num_threads(2)
    started = time.monotonic()
    loaded = frozen_r18.load_frozen_cpu(args.source_root, args.contract, args.checkpoint,
                                        args.source_manifest)
    model = loaded["model"]
    contract = loaded["contract"]
    dev = frozen_r18.development548(contract)
    split = json.loads(Path(args.split).read_text(encoding="utf-8"))
    assignment = split["assignment"]
    train_rows = [row for row in dev.records if assignment.get(row["sequence_key"]) == "train"]
    require(len(train_rows) == 338, "train-side development row count differs")
    train_rows.sort(key=lambda row: row["stable_image_id"])
    rows = train_rows[:args.images]
    matcher, box_ops = native_matcher(loaded["release_root"])

    decoder = model.decoder
    k_queries = int(getattr(decoder, "num_queries"))
    require(k_queries == 300, "unexpected query count")
    anchors = torch.sigmoid(torch.nan_to_num(decoder.anchors[0], posinf=60.0)).numpy()
    valid = decoder.valid_mask[0, :, 0].numpy().astype(bool)

    image_records = []
    shapes_expected = None
    anchor_table = None
    cov = defaultdict(lambda: defaultdict(list))
    cov_sel = defaultdict(lambda: defaultdict(list))
    lost_rows = defaultdict(list)
    dup_sel = defaultdict(list)
    size_ratio = defaultdict(list)
    center_ratio = []
    coarseness = []
    best_hist_small = np.zeros(10, dtype=np.int64)
    sel_level_counts = np.zeros(3, dtype=np.int64)
    sel_matched_group = defaultdict(int)
    sel_unmatched = 0
    spearmans = []
    pair_stats = defaultdict(lambda: {"iou": [], "wc": [], "wb": [], "wg": []})

    for position, row in enumerate(rows):
        pixels, target = frozen_r18.image_pixels_and_target(dev, row, position)
        captured, outputs = capture_forward(model, pixels)
        logits = captured["logits"][0]
        coords = captured["coords_unact"][0]
        shapes = captured["spatial_shapes"]
        if shapes_expected is None:
            shapes_expected = shapes
            require(sum(h * w for h, w in shapes) == int(logits.shape[0]),
                    "level map size differs")
        require(shapes == shapes_expected, "spatial shapes changed across images")
        L = int(logits.shape[0])
        scores_raw = logits.max(-1).values
        topk_ind = torch.topk(scores_raw, k_queries).indices
        require(torch.equal(coords[topk_ind], captured["gathered_coords"][0]),
                "recomputed top-k does not reproduce the production gather")

        level = np.zeros(L, dtype=np.int64)
        start = 0
        for lvl, (h, w) in enumerate(shapes):
            level[start:start + h * w] = lvl
            start += h * w
        if anchor_table is None:
            anchor_table = {}
            for lvl, (h, w) in enumerate(shapes):
                lvl_valid = valid & (level == lvl)
                wh = anchors[lvl_valid][:, 2:4]
                anchor_table[str(lvl)] = {
                    "grid": [int(h), int(w)],
                    "valid_tokens": int(lvl_valid.sum()),
                    "anchor_w_norm_median": r6(np.median(wh[:, 0])),
                    "anchor_h_norm_median": r6(np.median(wh[:, 1])),
                    "anchor_w_canvas_px": r6(np.median(wh[:, 0]) * 1344.0),
                    "anchor_h_canvas_px": r6(np.median(wh[:, 1]) * 768.0),
                }

        pred_xyxy = cxcywh_to_xyxy(torch.sigmoid(coords).numpy())
        scores = scores_raw.numpy()
        score_sig = 1.0 / (1.0 + np.exp(-scores))
        sel = np.zeros(L, dtype=bool)
        sel[topk_ind.numpy()] = True
        sel_idx = np.flatnonzero(sel)
        sel_level_counts += np.bincount(level[sel_idx], minlength=3)

        gt_cxcywh = target["boxes"].numpy()
        raw_area = target["raw_area"].numpy()
        gt_ids = list(target["gt_ids"])
        labels = target["labels"].numpy()
        G = len(gt_ids)
        record = {"image_id": row["stable_image_id"], "sequence": row["sequence_key"],
                  "width": int(row["width"]), "height": int(row["height"]),
                  "n_gt": G, "n_tokens": L, "n_selected": int(sel.sum()), "gts": []}
        gt_xyxy = None
        if G > 0:
            gt_xyxy = cxcywh_to_xyxy(gt_cxcywh)
            iou = xyxy_iou(pred_xyxy, gt_xyxy)
            best = iou.argmax(1)
            best_iou = iou[np.arange(G), best]
            sel_best = iou[:, sel].max(1)
            greater = (scores[None, :] > scores[best][:, None]).sum(1)
            rank_pct = greater / float(L)
            sel_ge50 = (iou[:, sel] >= 0.5).sum(1)
            sel_ge75 = (iou[:, sel] >= 0.75).sum(1)
            best_level = level[best]
            for g in range(G):
                area = float(raw_area[g])
                band = size_band(area)
                cov[band][0.5].append(bool(best_iou[g] >= 0.5))
                cov[band][0.75].append(bool(best_iou[g] >= 0.75))
                cov_sel[band][0.5].append(bool(sel_best[g] >= 0.5))
                cov_sel[band][0.75].append(bool(sel_best[g] >= 0.75))
                gt_b = gt_xyxy[g]
                pr_b = pred_xyxy[best[g]]
                size_ratio_gt = (float(pr_b[2] - pr_b[0]) / max(float(gt_b[2] - gt_b[0]), 1e-9)
                                 if best_iou[g] >= 0.3 else None)
                center_err_gt = (max(abs(float((pr_b[0] + pr_b[2]) / 2 - (gt_b[0] + gt_b[2]) / 2))
                                     / max(float(gt_b[2] - gt_b[0]), 1e-9),
                                     abs(float((pr_b[1] + pr_b[3]) / 2 - (gt_b[1] + gt_b[3]) / 2))
                                     / max(float(gt_b[3] - gt_b[1]), 1e-9))
                                 if best_iou[g] >= 0.3 else None)
                record["gts"].append({
                    "gt_id": gt_ids[g], "label": int(labels[g]), "area": r6(area), "band": band,
                    "best_iou": r6(best_iou[g]), "sel_best_iou": r6(sel_best[g]),
                    "best_rank_pct": r6(rank_pct[g]), "best_selected": bool(sel[best[g]]),
                    "best_level": int(best_level[g]),
                    "sel_ge50": int(sel_ge50[g]), "sel_ge75": int(sel_ge75[g]),
                    "center_err_over_gt": r6(center_err_gt) if center_err_gt is not None else None,
                    "pred_w_over_gt_w": r6(size_ratio_gt) if size_ratio_gt is not None else None,
                })
                if best_iou[g] >= 0.5:
                    dup_sel[band].append(int(sel_ge50[g]))
                    gt_b = gt_xyxy[g]
                    pr_b = pred_xyxy[best[g]]
                    size_ratio[band].append(float(pr_b[2] - pr_b[0]) / max(float(gt_b[2] - gt_b[0]), 1e-9))
                    if band == "small" and 0.3 <= best_iou[g] < 0.75:
                        dcx = abs(float((pr_b[0] + pr_b[2]) / 2 - (gt_b[0] + gt_b[2]) / 2))
                        dcy = abs(float((pr_b[1] + pr_b[3]) / 2 - (gt_b[1] + gt_b[3]) / 2))
                        center_ratio.append(max(dcx / max(float(gt_b[2] - gt_b[0]), 1e-9),
                                                dcy / max(float(gt_b[3] - gt_b[1]), 1e-9)))
                        coarseness.append(abs(np.log(
                            max(float(pr_b[2] - pr_b[0]), 1e-9) / max(float(gt_b[2] - gt_b[0]), 1e-9))))
                if best_iou[g] >= 0.5 and not sel_best[g] >= 0.5:
                    lost_rows[band].append(r6(rank_pct[g]))
                if band == "small":
                    best_hist_small[min(int(best_iou[g] * 10), 9)] += 1

            token_best = iou.argmax(0)
            token_best_iou = iou[token_best, np.arange(L)]
            match_count = (iou >= 0.5).sum(0)
            unique = (match_count == 1) & (token_best_iou >= 0.5)
            if int(unique.sum()) >= 8:
                spearmans.append(spearman(raw_area[token_best[unique]], score_sig[unique]))
            for i in sel_idx:
                if unique[i]:
                    sel_matched_group[size_band(float(raw_area[token_best[i]]))] += 1
                else:
                    sel_unmatched += 1

        pairs = matcher_records(matcher, box_ops, outputs, target,
                                gt_xyxy, raw_area) if G > 0 else []
        for pair in pairs:
            band = size_band(pair["area"])
            pair_stats[band]["iou"].append(pair["iou"])
            pair_stats[band]["wc"].append(pair["cost_class_w"])
            pair_stats[band]["wb"].append(pair["cost_bbox_w"])
            pair_stats[band]["wg"].append(pair["cost_giou_w"])

        image_records.append(record)
        pooled = ([g["best_iou"] >= 0.5 for g in record["gts"]] if G else [])
        selected = ([g["sel_best_iou"] >= 0.5 for g in record["gts"]] if G else [])
        print("image %d/%d %s gt=%d pool50=%.3f sel50=%.3f (%.1fs)" % (
            position + 1, len(rows), row["stable_image_id"][:12], G,
            float(np.mean(pooled)) if pooled else -1.0,
            float(np.mean(selected)) if selected else -1.0,
            time.monotonic() - started), flush=True)

    def frac(values):
        return r6(np.mean(values)) if values else None

    matcher_pairs = {}
    for band in BAND_NAMES:
        stats = pair_stats[band]
        if not stats["iou"]:
            matcher_pairs[band] = None
            continue
        shares = [(abs(b) + abs(g)) / max(abs(c) + abs(b) + abs(g), 1e-12)
                  for c, b, g in zip(stats["wc"], stats["wb"], stats["wg"])]
        matcher_pairs[band] = {
            "count": len(stats["iou"]),
            "iou_median": r6(np.median(stats["iou"])),
            "w_class_mean": r6(np.mean(stats["wc"])),
            "w_bbox_mean": r6(np.mean(stats["wb"])),
            "w_giou_mean": r6(np.mean(stats["wg"])),
            "loc_share_mean": r6(np.mean(shares)),
        }

    aggregate = {
        "gts": {band: int(len(cov[band][0.5])) for band in BAND_NAMES},
        "pool_cov": {band: {str(t): frac(cov[band][t]) for t in IOU_THRESHOLDS} for band in BAND_NAMES},
        "sel_cov": {band: {str(t): frac(cov_sel[band][t]) for t in IOU_THRESHOLDS} for band in BAND_NAMES},
        "lost_at_0.5": {band: {"count": len(lost_rows[band]),
                               "rank_pct_median": r6(np.median(lost_rows[band])) if lost_rows[band] else None,
                               "rank_pct_p90": r6(np.percentile(lost_rows[band], 90)) if lost_rows[band] else None,
                               "frac_rank_pct_le_0.03": frac([v <= 0.03 for v in lost_rows[band]])}
                        for band in BAND_NAMES},
        "dup_selected_per_gt_ge50": {band: {"mean": r6(np.mean(dup_sel[band])) if dup_sel[band] else None,
                                            "gt_with_ge2_frac": frac([v >= 2 for v in dup_sel[band]])}
                                     for band in BAND_NAMES},
        "pred_w_over_gt_w": {band: {"median": r6(np.median(size_ratio[band])) if size_ratio[band] else None,
                                    "q10": r6(np.percentile(size_ratio[band], 10)) if size_ratio[band] else None,
                                    "q90": r6(np.percentile(size_ratio[band], 90)) if size_ratio[band] else None}
                             for band in BAND_NAMES},
        "small_best03_075": {
            "count": len(center_ratio),
            "center_err_over_gt_median": r6(np.median(center_ratio)) if center_ratio else None,
            "abs_log_size_ratio_median": r6(np.median(coarseness)) if coarseness else None,
        },
        "small_best_iou_hist": [int(x) for x in best_hist_small],
        "selected_level_counts": [int(x) for x in sel_level_counts],
        "selected_matched_by_band": {band: int(sel_matched_group.get(band, 0)) for band in BAND_NAMES},
        "selected_unmatched": int(sel_unmatched),
        "score_size_spearman": {"per_image": [r6(x) for x in spearmans],
                                "mean": r6(np.nanmean(spearmans)) if spearmans else None},
        "matcher_pairs": matcher_pairs,
    }
    result = {
        "probe": "r18-selection-coverage-cpu-v1",
        "status": "PASS",
        "frozen": {"checkpoint_sha256": loaded["checkpoint_sha256"],
                   "release_root": loaded["release_root"],
                   "release_manifest_sha256": (frozen_r18.file_sha256(args.source_manifest)
                                               if args.source_manifest else None),
                   "binding_scope": loaded["binding"]["scope"],
                   "ema_updates": frozen_r18.FROZEN_EMA_UPDATES},
        "model": {"parameters": int(sum(p.numel() for p in model.parameters())),
                  "decoder_class": type(decoder).__module__ + "." + type(decoder).__name__,
                  "feat_strides": [int(s) for s in getattr(decoder, "feat_strides", [])],
                  "num_queries": k_queries,
                  "spatial_shapes": shapes_expected,
                  "criterion_recipe": loaded["criterion"].recipe()},
        "anchors": anchor_table,
        "image_set": {"count": len(image_records), "split": "pdra_train_side",
                      "order": "stable_image_id ascending"},
        "aggregate": aggregate,
        "images": image_records,
        "elapsed_seconds": r6(time.monotonic() - started),
        "cuda_initialized": bool(torch.cuda.is_initialized()),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "RESULT.json").write_text(json.dumps(result, indent=1, ensure_ascii=False),
                                     encoding="utf-8")
    print(json.dumps(aggregate, indent=1, ensure_ascii=False))
    print("PROBE_COMPLETE", r6(result["elapsed_seconds"]), "s")


if __name__ == "__main__":
    main()
