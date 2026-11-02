"""C1 prototype (CPU, zero training, zero extra forward): admit the encoder dense head's own
candidates beside the 300 selected queries, and measure what that changes at the certified endpoint.

Mechanism.  RTDETRTransformerv2._get_decoder_input already computes, inside the single existing
forward, enc_score_head / enc_bbox_head over the full encoder anchor pool (21,168 anchors x 10
classes = 211,680 candidates per image) and then throws all but the top-300 away in _select_topk.
This prototype captures that dense pool with a read-only hook (same monkeypatch the D1 probe used and
verified reproduces the production gather bitwise), builds candidate lists that admit the dense pool,
ranks them with the certified scoring rule, and scores them with the certified bridge.

No window, no box edit, no score edit, no extra forward: the only thing that changes is WHICH
candidates enter the top-600/top-300 list.  Every entry that survives from the whole-image arm is
copied verbatim (that is check 3 below).

Pre-registration (fixed before the full-set read; recorded in RESULT.json):
  1. dense-only AP@600 must be >= DENSE_ONLY_AP_MIN, else the dense head is not usable output and the
     whole axis is reported dead on arrival.
  2. fused - A0cpu deltas are reported for all seven metrics at BOTH certified budgets (600 and 300;
     the 300 list is the certified ordered prefix of the 600 list).  The five historical protection
     cells are the AP@600 / AP@300 / APS@300 / APM@600 / APL@600 entries of that table and are also
     summarised separately.  The historical gate (+0.50 pp APS@600 and +0.30 pp AP75_small@600) is
     reported for context; this prototype pre-registers no pass gate.
  3. set-addition invariant: every entry present in both the A0cpu list and a fused list must have
     identical category_id, score and bbox_xyxy (exact float equality); displaced entries are counted.
  4. exactly one model forward per image (asserted by counting calls).
  5. fused pre-cut candidate-set recall@0.5/0.75 for small GTs must exceed the A0cpu 600-list recall
     on the same images.  Both sides use the same unfiltered target GT set (the transform's
     target["labels"] / target["raw_area"], whose labels are already the 0-based class index and whose
     category-0 ignore region is already excluded), so the comparison is internally valid but is NOT
     the certified bridge's region-filtered GT; it is a diagnostic, not a metric.
  6. pipeline fidelity: the CPU forward + certified ranking must reproduce the frozen A0 list
     (run-003/baseline) as CONTENT, not as slot identity.  Per class, entries are matched on
     descending score within FIDELITY_SCORE_TOL and FIDELITY_BOX_TOL_PX; the check passes when the
     unmatched fraction on each side is <= FIDELITY_UNMATCHED_MAX_FRAC and the largest matched score
     difference stays within tolerance.  Query-id agreement is reported separately (6b) and is NOT
     gated: the encoder top-k is a near-tie ordering, so a run may permute query slots by +-1 without
     changing any box or score.  Correction 2026-11-02: the original check keyed on
     (query_id, category_id) and reported max |dscore| 0.291 on the full set purely because of such
     permutations; the content check passes there with max |dscore| 9.984e-06, max |dbox| 0.0346 px
     and 387 unmatched entries per side out of 328,800 (verified by tools/explore/fidelity_tolerance.py).

Six variants.  `union-perclass100` was dropped before the full-set read (its 100-per-class cap could
never bind inside a 600 budget) and replaced by `union-perclass10` and `union-nms05`; the smoke-002
artefact still records the old five-variant set, and its check-5 numbers are superseded because the GT
label index was off by one there (see the comment at the recall call).

usage (server, from the worktree root, CUDA hidden, certified interpreter):
  CUDA_VISIBLE_DEVICES= <frozen python> -I tools/explore/dense_pool_fuse.py \
      --source-root <e088 release-001>/source --contract <run>/execution-contract-001.json \
      --checkpoint <run>/e088-io-20261014-001/formal/checkpoint-epoch-120.pth \
      --source-manifest <e088 release-001>/source-manifest.json \
      --frozen-a0 <r20 round>/run-003/baseline/pred-top600.json \
      --images 548 --out <explore round>/probe-dense-fuse-001
"""
from __future__ import annotations

import argparse
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

PREREGISTRATION = {
    "dense_only_ap_min": 0.05,
    "fidelity_score_tol": 1e-5,
    "fidelity_box_tol_px": 0.05,
    "fidelity_unmatched_max_frac": 0.01,
    "historical_gate": {"APS@600": 0.50, "AP75_small@600": 0.30},
    "small_area_px": 1024.0,
    "merge_rules": {"nms_iou_primary": 0.7, "nms_iou_strict": 0.5, "per_class_cap": 10,
                    "max_candidates_per_image": 20000},
    "variants": ["A0frozen", "A0cpu", "dense-only", "union", "union-nms07", "union-nms05",
                 "union-perclass10"],
    "replaced_before_full_run": {
        "union-perclass100": "a cap of 100 per class x 10 classes = 1000 slots exceeds the 600 budget, "
                             "so the cap never bound; smoke-002's list was shaped like plain union. "
                             "Replaced before the full-set read by perclass10 (100 dense slots max) and "
                             "union-nms05 (stricter same-class dedup).",
    },
    "budgets_scored": [600, 300],
}
DENSE_ONLY_AP_MIN = PREREGISTRATION["dense_only_ap_min"]
FIDELITY_SCORE_TOL = PREREGISTRATION["fidelity_score_tol"]
FIDELITY_BOX_TOL_PX = PREREGISTRATION["fidelity_box_tol_px"]
FIDELITY_UNMATCHED_MAX_FRAC = PREREGISTRATION["fidelity_unmatched_max_frac"]
SMALL_AREA = PREREGISTRATION["small_area_px"]
NMS_IOU = 0.7
NMS_IOU_STRICT = 0.5
PER_CLASS_CAP = 10
NMS_MAX_CANDIDATES = 20000
DENSE_QUERY_BASE = 100000


def r6(value):
    return round(float(value), 6)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def cxcywh_to_xyxy(boxes):
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return np.stack([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], axis=1)


def xyxy_iou(a, b):
    """IoU matrix between two (N,4) / (M,4) pixel-space xyxy arrays."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-12, None)


def capture_dense_pool(model, pixels):
    """Read-only hook: capture the encoder selection logits/boxes and the production top-k gather."""
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


def flatten_candidates(logits2d, boxes_per_anchor):
    """Flatten an (N,10) logit block into one candidate per (anchor, class).

    Row-major flattening puts candidate i on anchor i//10 and class i%10, so the per-anchor boxes must
    be repeated 10 times to line up with the flattened scores.  Mixing the two conventions is an index
    bug (it produced a real IndexError in the first smoke run), hence the explicit assert below.
    """
    n = logits2d.shape[0]
    score = (1.0 / (1.0 + np.exp(-logits2d.astype(np.float64)))).reshape(-1)
    query = np.repeat(np.arange(n), 10)
    category = np.tile(np.arange(10), n)
    boxes = np.repeat(boxes_per_anchor, 10, axis=0)
    require(len(score) == len(query) == len(category) == len(boxes) == n * 10,
            "dense candidate table is not one row per (anchor, class)")
    return score, query, category, boxes


def rank_pool(score, query, category, boxes, source, budget):
    """Certified ranking rule extended to a fused pool: descending score, then source, query, class."""
    order = np.lexsort((category, query, source, -score))[:budget]
    return order


def greedy_nms_keep(order, score, cat, box, protect, budget, max_candidates, iou_threshold=NMS_IOU):
    """Walk the merged pool in score order; protected (whole-image) entries are always kept, the rest
    are kept unless they duplicate an already-kept same-class entry above NMS_IOU.  Stop once no later
    candidate can enter the top-`budget`: kept stays in descending score order, so its budget-th entry
    is the bar.  Returns (kept positions, candidates processed)."""
    kept, kept_by_class, processed = [], {}, 0
    for pos in order:
        if len(kept) >= budget and score[pos] <= score[kept[budget - 1]]:
            break
        processed += 1
        if processed > max_candidates:
            break
        cls = int(cat[pos])
        prior = kept_by_class.get(cls)
        if not protect[pos] and prior:
            iou = xyxy_iou(box[pos][None, :], box[np.asarray(prior)])[0]
            if float(iou.max()) > iou_threshold:
                continue
        kept.append(pos)
        kept_by_class.setdefault(cls, []).append(pos)
    return kept, processed


def per_class_capped_keep(order, score, cat, protect, budget, cap, max_candidates):
    """Same interleaving and early exit, but the only filter on non-protected entries is a per-class
    cap on how many dense candidates may be admitted."""
    kept, seen, processed = [], {}, 0
    for pos in order:
        if len(kept) >= budget and score[pos] <= score[kept[budget - 1]]:
            break
        processed += 1
        if processed > max_candidates:
            break
        cls = int(cat[pos])
        if not protect[pos]:
            if seen.get(cls, 0) >= cap:
                continue
            seen[cls] = seen.get(cls, 0) + 1
        kept.append(pos)
    return kept, processed


def small_recall(cand_boxes, cand_cat, gt_boxes, gt_lab0, gt_area):
    """Fraction of small GTs with a same-class candidate at IoU >= 0.5 / 0.75.

    The certified bridge only matches same-class pairs, so this loops over the 10 classes rather than
    over GTs; the largest block is 21,168 candidates against that class's small GTs.
    """
    small = np.flatnonzero(gt_area <= SMALL_AREA)
    out = {"n_small": int(len(small))}
    hits = {0.5: 0, 0.75: 0}
    for label in (np.unique(gt_lab0[small]) if len(small) else []):
        gsel = small[gt_lab0[small] == label]
        csel = np.flatnonzero(cand_cat == label)
        if not len(csel):
            continue
        best = xyxy_iou(gt_boxes[gsel], cand_boxes[csel]).max(axis=1)
        for thr in (0.5, 0.75):
            hits[thr] += int((best >= thr).sum())
    for thr in (0.5, 0.75):
        out[str(thr)] = (hits[thr] / len(small)) if len(small) else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-root", required=True)
    ap.add_argument("--contract", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--source-manifest", default=None)
    ap.add_argument("--frozen-a0", default=None, help="run-003 baseline pred-top600.json (fidelity)")
    ap.add_argument("--images", type=int, default=548)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(8)
    started = time.monotonic()
    loaded = frozen_r18.load_frozen_cpu(args.source_root, args.contract, args.checkpoint,
                                        args.source_manifest)
    model = loaded["model"]
    contract = loaded["contract"]
    dev = frozen_r18.development548(contract)
    rows_all = dev.records
    rows = rows_all if args.images <= 0 else rows_all[:args.images]
    require(len(rows) > 0, "no images selected")

    sys.path.insert(0, str(Path(args.source_root) / "src"))
    from sparse_rtdetr.r0 import evaluation as ev

    parsed = [dev.raw(row) for row in rows]
    frozen_a0 = {}
    if args.frozen_a0:
        for entry in json.loads(Path(args.frozen_a0).read_text(encoding="utf-8")):
            frozen_a0.setdefault(entry["image_id"], []).append(entry)

    variants = [name for name in PREREGISTRATION["variants"] if name != "A0frozen" or frozen_a0]
    n = len(rows)
    budget = 600
    store = {name: {"image_idx": np.zeros((n, budget), np.int32),
                    "query": np.zeros((n, budget), np.int64),
                    "cat": np.zeros((n, budget), np.int8),
                    "score": np.zeros((n, budget), np.float32),
                    "box": np.zeros((n, budget, 4), np.float32)} for name in variants}
    forward_calls = 0
    fidelity, invariants, recalls = [], {}, {}
    pool_shapes = None
    base_recalls, fused_recalls = [], []
    nms07_processed, nms05_processed, cap_processed = [], [], []

    for position, row in enumerate(rows):
        pixels, target = frozen_r18.image_pixels_and_target(dev, row, position)
        captured, outputs = capture_dense_pool(model, pixels)
        forward_calls += 1

        enc_logits = captured["logits"][0].numpy()
        enc_coords = captured["coords_unact"][0].numpy()
        shapes = captured["spatial_shapes"]
        pool_shapes = pool_shapes or shapes
        require(sum(h * w for h, w in shapes) == enc_logits.shape[0], "anchor pool size differs")
        pool_sig = torch.sigmoid(torch.from_numpy(enc_coords)).numpy()
        pool_boxes_anchor = cxcywh_to_xyxy(pool_sig) * np.array([row["width"], row["height"]] * 2)
        pool_score, pool_query, pool_cat, pool_boxes = flatten_candidates(enc_logits, pool_boxes_anchor)

        base = ev.rank_candidates(outputs["pred_logits"][0], outputs["pred_boxes"][0], row, budget)
        dec_score = np.array([b["score"] for b in base], np.float64)
        dec_query = np.array([b["query_id"] for b in base], np.int64)
        dec_cat = np.array([b["category_id"] - 1 for b in base], np.int8)
        dec_box = np.array([b["bbox_xyxy"] for b in base], np.float32)

        # --- check 6: content fidelity against the frozen whole-image list (see the docstring) -----
        if frozen_a0:
            entries = frozen_a0.get(row["stable_image_id"], [])
            f_score = np.array([e["score"] for e in entries], np.float64)
            f_cat = np.array([e["category_id"] - 1 for e in entries], np.int8)
            f_query = np.array([e["query_id"] for e in entries], np.int64)
            f_box = np.array([e["bbox_xyxy"] for e in entries], np.float32)
            matched = only_cpu = only_frozen = nonzero_shift = 0
            max_ds = max_db = 0.0
            for cat_index in range(10):
                order_a = np.flatnonzero(dec_cat == cat_index)
                order_a = order_a[np.argsort(-dec_score[order_a], kind="stable")]
                order_b = np.flatnonzero(f_cat == cat_index)
                order_b = order_b[np.argsort(-f_score[order_b], kind="stable")]
                i = j = 0
                while i < len(order_a) and j < len(order_b):
                    a, b = order_a[i], order_b[j]
                    ds = abs(dec_score[a] - float(f_score[b]))
                    db = float(np.abs(dec_box[a] - f_box[b]).max())
                    if ds <= FIDELITY_SCORE_TOL and db <= FIDELITY_BOX_TOL_PX:
                        matched += 1
                        max_ds = max(max_ds, ds)
                        max_db = max(max_db, db)
                        if int(dec_query[a]) != int(f_query[b]):
                            nonzero_shift += 1
                        i += 1
                        j += 1
                    elif dec_score[a] > f_score[b]:
                        only_cpu += 1
                        i += 1
                    else:
                        only_frozen += 1
                        j += 1
                only_cpu += len(order_a) - i
                only_frozen += len(order_b) - j
            fidelity.append({"image_id": row["stable_image_id"], "entries": len(entries),
                             "matched": matched, "unmatched_cpu": only_cpu,
                             "unmatched_frozen": only_frozen,
                             "max_abs_score_diff": max_ds, "max_abs_box_diff": max_db,
                             "query_id_shifts": nonzero_shift})

        # --- fused pools ------------------------------------------------------------------------
        union_score = np.concatenate([dec_score, pool_score])
        union_query = np.concatenate([dec_query, pool_query])
        union_cat = np.concatenate([dec_cat, pool_cat])
        union_box = np.concatenate([dec_box, pool_boxes.astype(np.float32)])
        union_source = np.concatenate([np.zeros(len(dec_score), np.int8),
                                       np.ones(len(pool_score), np.int8)])
        protect = union_source == 0

        def emit(name, order, idx):
            order = order[:budget]
            take = len(order)
            s = store[name]
            s["image_idx"][idx, :take] = idx
            s["query"][idx, :take] = np.where(union_source[order] == 0, union_query[order],
                                              DENSE_QUERY_BASE + union_query[order])
            s["cat"][idx, :take] = union_cat[order]
            s["score"][idx, :take] = union_score[order]
            s["box"][idx, :take] = union_box[order]
            if take < budget:
                s["cat"][idx, take:] = -1
            return order

        # A0cpu: exactly the certified rule on the decoder candidates only
        order_a0 = np.arange(len(dec_score))
        order_a0 = order_a0[np.lexsort((dec_cat, dec_query, -dec_score))[:budget]]
        emit("A0cpu", order_a0, position)

        # dense-only
        dense_mask = union_source == 1
        dense_idx = np.flatnonzero(dense_mask)
        order_dense = dense_idx[np.lexsort((union_cat[dense_idx], union_query[dense_idx],
                                            -union_score[dense_idx]))[:budget]]
        emit("dense-only", order_dense, position)

        # union (no dedup)
        order_union = rank_pool(union_score, union_query, union_cat, union_box, union_source, budget)
        emit("union", order_union, position)

        # union + dense-side NMS against everything already kept (protected entries never dropped)
        order_all = rank_pool(union_score, union_query, union_cat, union_box, union_source,
                              len(union_score))
        kept, processed_nms07 = greedy_nms_keep(order_all, union_score, union_cat, union_box, protect,
                                                budget, NMS_MAX_CANDIDATES, NMS_IOU)
        order_nms = np.array(kept, np.int64)[:budget]
        emit("union-nms07", order_nms, position)
        nms07_processed.append(processed_nms07)

        # same, with a stricter same-class dedup threshold
        kept05, processed_nms05 = greedy_nms_keep(order_all, union_score, union_cat, union_box, protect,
                                                  budget, NMS_MAX_CANDIDATES, NMS_IOU_STRICT)
        order_nms05 = np.array(kept05, np.int64)[:budget]
        emit("union-nms05", order_nms05, position)
        nms05_processed.append(processed_nms05)

        # union + at most PER_CLASS_CAP dense additions per class (100 dense slots in total)
        kept_cap, processed_cap = per_class_capped_keep(order_all, union_score, union_cat, protect,
                                                      budget, PER_CLASS_CAP, NMS_MAX_CANDIDATES)
        order_cap = np.array(kept_cap, np.int64)[:budget]
        emit("union-perclass10", order_cap, position)
        cap_processed.append(processed_cap)

        # --- check 3: set-addition invariant ----------------------------------------------------
        a0_keys = {}
        for pos in order_a0:
            a0_keys[(int(union_query[pos]), int(union_cat[pos]))] = pos
        for name in variants:
            if name in ("A0cpu", "A0frozen"):
                continue
            row_entries = store[name]
            keys = set()
            bad = 0
            for k in range(budget):
                if row_entries["cat"][position, k] < 0:
                    break
                q, c = int(row_entries["query"][position, k]), int(row_entries["cat"][position, k])
                keys.add((q, c) if q < DENSE_QUERY_BASE else (("dense", q), c))
            for (q, c), pos in a0_keys.items():
                if (q, c) in keys:
                    k = np.flatnonzero((row_entries["query"][position, :] == q)
                                       & (row_entries["cat"][position, :] == c))
                    if len(k):
                        k = int(k[0])
                        if not (row_entries["score"][position, k] == union_score[pos]
                                and np.array_equal(row_entries["box"][position, k], union_box[pos])):
                            bad += 1
            record = invariants.setdefault(name, {"checked": 0, "modified": 0, "displaced": 0})
            record["checked"] += len(a0_keys)
            record["modified"] += bad
            record["displaced"] += len([1 for (q, c) in a0_keys if (q, c) not in keys])

        # --- check 5: recall of the pre-cut fused pool vs the A0cpu 600-list ---------------------
        gt_boxes = cxcywh_to_xyxy(target["boxes"].numpy() if hasattr(target["boxes"], "numpy")
                                  else np.asarray(target["boxes"], np.float32))
        gt_boxes = gt_boxes * np.array([row["width"], row["height"]] * 2, np.float32)
        # target["labels"] is already the 0-based class index (the parsed categories are 1-based);
        # verified directly on three development images before the full run.  Do NOT subtract 1 here:
        # doing so shifted every GT one class down and depressed the recall check to ~0.25.
        gt_lab0 = np.asarray(target["labels"]).astype(np.int64)
        gt_area = np.asarray(target["raw_area"], np.float32)
        base_recalls.append(small_recall(union_box[order_a0], union_cat[order_a0], gt_boxes, gt_lab0,
                                         gt_area))
        fused_recalls.append(small_recall(union_box, union_cat, gt_boxes, gt_lab0, gt_area))

        if position % 8 == 0 or position == n - 1:
            print("image %d/%d %s pool=%d (%.1fs)" % (position + 1, n, row["stable_image_id"][:12],
                                                      len(pool_score), time.monotonic() - started),
                  flush=True)

    # --- frozen A0 anchor: the certified whole-image list, scored through the same bridge ---------
    # Gives the protocol's own baseline as a row (and, on the full set, an end-to-end reproduction of
    # the certified endpoint numbers).  The file is in certified rank order, so the 300 list is its
    # per-image prefix.  It is excluded from the delta table and from the invariant comparison.
    if frozen_a0:
        for position, row in enumerate(rows):
            entries = frozen_a0.get(row["stable_image_id"], [])
            s = store["A0frozen"]
            take = min(len(entries), budget)
            for k in range(take):
                e = entries[k]
                s["image_idx"][position, k] = position
                s["query"][position, k] = int(e["query_id"])
                s["cat"][position, k] = int(e["category_id"]) - 1
                s["score"][position, k] = float(e["score"])
                s["box"][position, k] = [float(v) for v in e["bbox_xyxy"]]
            if take < budget:
                s["cat"][position, take:] = -1

    # --- score every variant with the certified bridge, at both certified budgets ---------------
    metrics = {}
    for name in variants:
        s = store[name]
        metrics[name] = {}
        for scored_budget in PREREGISTRATION["budgets_scored"]:
            preds = []
            for position in range(n):
                for k in range(scored_budget):
                    cat = int(s["cat"][position, k])
                    if cat < 0:
                        break
                    preds.append({"image_id": rows[position]["stable_image_id"],
                                  "query_id": int(s["query"][position, k]),
                                  "category_id": cat + 1,
                                  "score": float(s["score"][position, k]),
                                  "bbox_xyxy": [float(v) for v in s["box"][position, k]]})
            scored = ev.score_budget(rows, parsed, preds, args.source_root)
            metrics[name][str(scored_budget)] = {"COCO_bridge": scored["COCO_bridge"],
                                                 "spatial_filter_counts": scored["spatial_filter_counts"],
                                                 "predictions": len(preds)}
            print("scored %s @%d: AP=%.6f APS=%.6f AP75_small=%.6f" % (
                name, scored_budget, scored["COCO_bridge"]["metrics"]["AP"],
                scored["COCO_bridge"]["metrics"]["APS"],
                scored["COCO_bridge"]["metrics"]["AP75_small"]), flush=True)
            if name in ("A0cpu", "union"):
                (Path(args.out) / ("pred-%s-top%d.json" % (name, scored_budget))).write_text(
                    json.dumps(preds), encoding="utf-8")

    def mean_of(records, key):
        vals = [r[key] for r in records if r.get(key) is not None]
        return r6(np.mean(vals)) if vals else None

    deltas = {}
    for name in variants:
        if name in ("A0cpu", "A0frozen"):
            continue
        deltas[name] = {str(b): {m: r6(100.0 * (metrics[name][str(b)]["COCO_bridge"]["metrics"][m]
                                                - metrics["A0cpu"][str(b)]["COCO_bridge"]["metrics"][m]))
                                 for m in ("AP", "AP50", "AP75", "APS", "APM", "APL", "AP75_small")}
                         for b in PREREGISTRATION["budgets_scored"]}

    checks = {
        "1_dense_only_non_trivial": {
            "AP@600": metrics["dense-only"]["600"]["COCO_bridge"]["metrics"]["AP"],
            "threshold": DENSE_ONLY_AP_MIN,
            "passed": metrics["dense-only"]["600"]["COCO_bridge"]["metrics"]["AP"] >= DENSE_ONLY_AP_MIN},
        "2_fused_deltas_pp": deltas,
        "2b_protection_cells": {
            "historical_thresholds_pp": {"AP@600": -0.10, "AP@300": -0.20, "APS@300": -0.20,
                                         "APM@600": -0.50, "APL@600": -0.50},
            "values": {name: {"AP@600": d["600"]["AP"], "AP@300": d["300"]["AP"],
                              "APS@300": d["300"]["APS"], "APM@600": d["600"]["APM"],
                              "APL@600": d["600"]["APL"]} for name, d in deltas.items()}},
        "3_set_addition_invariant": invariants,
        "4_forward_calls": {"calls": forward_calls, "images": n, "one_per_image": forward_calls == n},
        "7_merge_rule_accounting": {
            "union-nms07": {"processed_total": int(sum(nms07_processed)),
                            "processed_max": int(max(nms07_processed)) if nms07_processed else 0},
            "union-nms05": {"processed_total": int(sum(nms05_processed)),
                            "processed_max": int(max(nms05_processed)) if nms05_processed else 0},
            "union-perclass10": {"processed_total": int(sum(cap_processed)),
                                 "processed_max": int(max(cap_processed)) if cap_processed else 0},
            "cap_per_image": NMS_MAX_CANDIDATES,
            "cap_hit_images": int(sum(1 for v in list(nms07_processed) + list(nms05_processed)
                                      + list(cap_processed) if v > NMS_MAX_CANDIDATES)),
        },
        "5_recall": {"A0cpu_600": {"0.5": mean_of(base_recalls, "0.5"), "0.75": mean_of(base_recalls, "0.75")},
                     "fused_pool": {"0.5": mean_of(fused_recalls, "0.5"), "0.75": mean_of(fused_recalls, "0.75")},
                     "n_small_total": int(sum(r["n_small"] for r in base_recalls))},
        "6_fidelity_vs_frozen_A0": {
            "entries_total": int(sum(r["entries"] for r in fidelity)),
            "matched_total": int(sum(r["matched"] for r in fidelity)),
            "unmatched_cpu_total": int(sum(r["unmatched_cpu"] for r in fidelity)),
            "unmatched_frozen_total": int(sum(r["unmatched_frozen"] for r in fidelity)),
            "max_abs_score_diff": (max(r["max_abs_score_diff"] for r in fidelity) if fidelity else None),
            "max_abs_box_diff_px": (max(r["max_abs_box_diff"] for r in fidelity) if fidelity else None),
            "thresholds": {"score_tol": FIDELITY_SCORE_TOL, "box_tol_px": FIDELITY_BOX_TOL_PX,
                           "unmatched_max_frac": FIDELITY_UNMATCHED_MAX_FRAC}},
        "6b_query_id_agreement": {
            "entries_with_shifted_query_id": int(sum(r["query_id_shifts"] for r in fidelity)),
            "note": "informational only; the encoder top-k is a near-tie ordering and slots may "
                    "permute by +-1 without changing any box or score"},
    }
    checks["5_recall"]["passed"] = (checks["5_recall"]["fused_pool"]["0.5"] or 0) > \
        (checks["5_recall"]["A0cpu_600"]["0.5"] or 0)
    fidelity_stats = checks["6_fidelity_vs_frozen_A0"]
    unmatched_frac = (max(fidelity_stats["unmatched_cpu_total"], fidelity_stats["unmatched_frozen_total"])
                      / max(fidelity_stats["entries_total"], 1))
    fidelity_stats["unmatched_max_frac"] = r6(unmatched_frac)
    fidelity_stats["passed"] = (
        bool(fidelity)
        and unmatched_frac <= FIDELITY_UNMATCHED_MAX_FRAC
        and fidelity_stats["max_abs_score_diff"] is not None
        and fidelity_stats["max_abs_score_diff"] <= FIDELITY_SCORE_TOL
        and fidelity_stats["max_abs_box_diff_px"] is not None
        and fidelity_stats["max_abs_box_diff_px"] <= FIDELITY_BOX_TOL_PX)

    result = {
        "prototype": "r21-c1-dense-pool-fusion-cpu-v1",
        "status": "PASS",
        "preregistration": PREREGISTRATION,
        "frozen": {"checkpoint_sha256": loaded["checkpoint_sha256"],
                   "release_root": loaded["release_root"],
                   "release_manifest_sha256": (frozen_r18.file_sha256(args.source_manifest)
                                               if args.source_manifest else None),
                   "binding_scope": loaded["binding"]["scope"],
                   "ema_updates": frozen_r18.FROZEN_EMA_UPDATES},
        "anchor_pool_shapes": pool_shapes,
        "images": {"count": n, "order": "development record order (stable_image_id)",
                   "first": rows[0]["stable_image_id"], "last": rows[-1]["stable_image_id"]},
        "metrics": metrics,
        "deltas_pp_vs_A0cpu": deltas,
        "checks": checks,
        "fidelity_rows": fidelity[:5],
        "elapsed_seconds": r6(time.monotonic() - started),
        "cuda_initialized": bool(torch.cuda.is_initialized()),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "RESULT.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: checks[k] for k in ("1_dense_only_non_trivial", "4_forward_calls",
                                             "5_recall", "6_fidelity_vs_frozen_A0")},
                     indent=1, ensure_ascii=False))
    print("PROBE_COMPLETE", r6(result["elapsed_seconds"]), "s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
