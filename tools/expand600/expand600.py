"""r22 mechanism: one extra decoder pass over the encoder top-600 anchors (inference only).

The certified endpoint runs one forward per image.  Inside it `_get_decoder_input` scores the whole
encoder pool (21,168 anchors x 10 classes), keeps the top-300 anchors as the decoder queries, and the
decoder returns 300 query outputs; the certified policy then ranks the 3000 (query, class) names and
submits the top 600.  The pool's own class scores are not usable output (r21/C1: every merge rule lost
at least 1.5 pp AP at budget 600), but the model can be asked to score the extra anchors itself.

This module keeps the certified first stage byte-identical and runs one extra decoder pass over the
encoder top-600 anchors; only the NEW 300 queries are admitted, after a same-class IoU 0.7 de-duplication
against the certified 600-name prefix.  Nothing is trained, no parameter or threshold is searched:
one configuration, dedup 0.7 fixed.  The certified evaluation protocol (category-0 raster ignore
filter applied AFTER the budget truncation, no refill) is untouched.
"""
from __future__ import annotations

import contextlib

import numpy as np
import torch

NUM_QUERIES = 300
NUM_NEW = 300
NUM_CLASSES = 10
DEDUP_IOU = 0.7
NEW_QUERY_BASE = NUM_QUERIES


def require(condition, message):
    if not condition:
        raise ValueError(message)


def find_decoder_module(model):
    """The single module that owns the encoder selection and the decoder input (read-only lookup)."""
    found = [module for module in model.modules()
             if hasattr(module, "_get_decoder_input") and hasattr(module, "_select_topk")]
    require(len(found) == 1, "expected exactly one RTDETR decoder, found %d" % len(found))
    return found[0]


def select_indices(logits, count):
    """The certified query-selection rule: top-k by the per-anchor maximum class logit."""
    return torch.topk(logits.max(-1).values, int(count), dim=-1).indices


def gather_rows(tensor, indices):
    expand = indices.unsqueeze(-1).expand(-1, -1, tensor.shape[-1])
    return tensor.gather(1, expand)


@contextlib.contextmanager
def capture_selection(decoder):
    """Read-only capture of the tensors `_get_decoder_input` builds.

    Also re-derives the production top-k gather from the captured tensors and records whether it is
    bitwise identical, so a divergence between the capture and the production path stops the run
    instead of silently changing the first stage.
    """
    captured = {}
    original_encoder_input = decoder._get_encoder_input
    original_decoder_input = decoder._get_decoder_input
    original_select = decoder._select_topk

    def encoder_input(*args, **kwargs):
        result = original_encoder_input(*args, **kwargs)
        captured["memory"] = result[0]
        captured["spatial_shapes"] = [list(shape) for shape in result[1]]
        return result

    def decoder_input(*args, **kwargs):
        return original_decoder_input(*args, **kwargs)

    def select(memory, outputs_logits, outputs_coords_unact, topk):
        captured["output_memory"] = memory
        captured["enc_logits"] = outputs_logits
        captured["enc_coords_unact"] = outputs_coords_unact
        captured["production_topk"] = int(topk)
        result = original_select(memory, outputs_logits, outputs_coords_unact, topk)
        captured["production_memory"] = result[0]
        captured["production_coords_unact"] = result[2]
        recomputed = gather_rows(memory, select_indices(outputs_logits, int(topk)))
        captured["recomputed_matches_production"] = bool(torch.equal(recomputed, result[0]))
        return result

    decoder._get_encoder_input = encoder_input
    decoder._get_decoder_input = decoder_input
    decoder._select_topk = select
    try:
        yield captured
    finally:
        del decoder._get_encoder_input
        del decoder._get_decoder_input
        del decoder._select_topk


def new_query_selection(ind300, ind600, all_logits, num_new=NUM_NEW):
    """Positions of the NEW queries inside the top-600 selection, per batch row.

    The new set is the top-600 index set minus the production top-300 index set, in top-600 order.
    Ties at the boundary can make the difference larger or smaller than `num_new`; every adjustment is
    counted instead of hidden, and the selection never reuses an anchor already taken.
    """
    require(ind300.shape[1] >= num_new and ind600.shape[1] == ind300.shape[1] + num_new,
            "unexpected selection shapes")
    positions, anchors, counters = [], [], []
    pool_size = int(all_logits.shape[1])
    for row in range(ind600.shape[0]):
        top300 = set(int(v) for v in ind300[row].tolist())
        order600 = [int(v) for v in ind600[row].tolist()]
        chosen = [value for value in order600 if value not in top300]
        absent = len(top300 - set(order600))
        truncated = 0
        filled = 0
        if len(chosen) < num_new:
            used = set(order600) | top300
            for value in select_indices(all_logits[row:row + 1], pool_size)[0].tolist():
                if value in used:
                    continue
                chosen.append(value)
                used.add(value)
                filled += 1
                if len(chosen) >= num_new:
                    break
        if len(chosen) > num_new:
            truncated = len(chosen) - num_new
            chosen = chosen[:num_new]
        position = [order600.index(value) for value in chosen]
        positions.append(position)
        anchors.append(chosen)
        counters.append({"row": row, "ind300_members_absent_from_ind600": absent,
                         "truncated_over_num_new": truncated, "filled_short_of_num_new": filled,
                         "num_new": len(chosen)})
    return {"positions": positions, "anchors": anchors, "counters": counters,
            "production_topk": int(ind300.shape[1])}


def second_stage(decoder, captured, indices):
    """One extra decoder pass over the selected anchors, using the captured encoder memory."""
    stack = decoder.decoder
    content = gather_rows(captured["output_memory"], indices).detach()
    reference = gather_rows(captured["enc_coords_unact"], indices).detach()
    boxes, logits = stack(content, reference, captured["memory"], captured["spatial_shapes"],
                          decoder.dec_bbox_head, decoder.dec_score_head, decoder.query_pos_head)
    return boxes[-1].float(), logits[-1].float()


def query_names(boxes, logits, positions, width, height, query_base=NEW_QUERY_BASE):
    """(query_slot, category_id, score, box_xyxy) for the selected new queries of one image.

    The scoring rule is the certified one: sigmoid of the class logit, normalised cxcywh converted to
    pixel xyxy.  The query slot is a synthetic id 300..599 so an added name is distinguishable from
    every first-stage name.
    """
    xy = torch.cat((boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2), -1)
    xy = xy * torch.tensor([width, height] * 2, dtype=torch.float32)
    scores = logits.sigmoid()
    out = []
    for slot, row in enumerate(positions):
        box = [float(v) for v in xy[row]]
        for klass in range(NUM_CLASSES):
            out.append((int(query_base + slot), int(klass) + 1, float(scores[row, klass]), box))
    return out


def xyxy_iou_matrix(boxes_a, boxes_b):
    """IoU between (N,4) and (M,4) pixel xyxy arrays."""
    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), dtype=np.float64)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0.0, None) * np.clip(a[:, 3] - a[:, 1], 0.0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0.0, None) * np.clip(b[:, 3] - b[:, 1], 0.0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-12, None)


def dedup_new_names(new_names, reference, iou_min=DEDUP_IOU):
    """Drop a new name when a certified entry of the same class overlaps it at IoU >= iou_min."""
    by_class_ref, by_class_new = {}, {}
    for entry in reference:
        by_class_ref.setdefault(int(entry["category_id"]), []).append(entry)
    for item in new_names:
        by_class_new.setdefault(int(item[1]), []).append(item)
    kept, dropped = [], []
    for category, items in by_class_new.items():
        reference_boxes = [e["bbox_xyxy"] for e in by_class_ref.get(category, [])]
        new_boxes = [item[3] for item in items]
        best = (xyxy_iou_matrix(new_boxes, reference_boxes).max(axis=1)
                if reference_boxes else np.zeros(len(items), dtype=np.float64))
        for index, item in enumerate(items):
            if float(best[index]) >= iou_min:
                dropped.append({"query_slot": item[0], "category_id": category, "score": item[2],
                                "best_iou": float(best[index])})
            else:
                kept.append(item)
    return kept, dropped


def merge_arm(reference, kept_new, budget, tie_break="score desc, query id, class id"):
    """Rank the certified entries and the admitted new entries with the certified key, then truncate."""
    entries = [{"image_id": entry["image_id"], "query_id": int(entry["query_id"]),
                "category_id": int(entry["category_id"]), "score": float(entry["score"]),
                "bbox_xyxy": [float(v) for v in entry["bbox_xyxy"]], "source": "first_stage"}
               for entry in reference]
    for slot, category, score, box in kept_new:
        entries.append({"image_id": reference[0]["image_id"] if reference else None,
                        "query_id": int(slot), "category_id": int(category), "score": float(score),
                        "bbox_xyxy": [float(v) for v in box], "source": "new_query"})
    entries.sort(key=lambda e: (-e["score"], e["query_id"], e["category_id"]))
    return entries[:budget], entries[budget:], tie_break


def per_class_macro_metrics(eval_result):
    """Class-equal-weight aggregation, the form of the certified published numbers.

    COCO's own `stats` average over (threshold, recall, class) cells, which weights classes with more
    valid recall bins more heavily; the certified endpoint's published block (top600-metrics.json)
    averages per class instead.  Both are reported; the gate uses this one.
    """
    precision = eval_result["precision"]
    recall = eval_result["recall"]
    num_class = precision.shape[2]

    def cls_precision(threshold, area, max_det):
        values = []
        for klass in range(num_class):
            cell = precision[threshold, :, klass, area, max_det]
            cell = cell[cell > -1]
            if len(cell):
                values.append(float(cell.mean()))
        return float(np.mean(values)) if values else -1.0

    def cls_recall(area, max_det):
        values = []
        for klass in range(recall.shape[1]):
            cell = recall[:, klass, area, max_det]
            cell = cell[cell > -1]
            if len(cell):
                values.append(float(cell.mean()))
        return float(np.mean(values)) if values else -1.0

    whole = slice(None)
    return {
        "AP": cls_precision(whole, 0, 2), "AP50": cls_precision(0, 0, 2),
        "AP75": cls_precision(5, 0, 2), "APS": cls_precision(whole, 1, 2),
        "APM": cls_precision(whole, 2, 2), "APL": cls_precision(whole, 3, 2),
        "AP75_small": cls_precision(5, 1, 2),
        "AR1": cls_recall(0, 0), "AR10": cls_recall(0, 1), "AR100": cls_recall(0, 2),
        "AR_small": cls_recall(1, 2), "AR_medium": cls_recall(2, 2), "AR_large": cls_recall(3, 2),
    }


def per_class_values(eval_result, area=0, max_det=2):
    """Per-class AP at (all thresholds, given area/maxDets), for the decomposition report."""
    precision = eval_result["precision"]
    out = []
    for klass in range(precision.shape[2]):
        cell = precision[:, :, klass, area, max_det]
        cell = cell[cell > -1]
        out.append(float(cell.mean()) if len(cell) else float("nan"))
    return out


def gt_potential(entries, raw_rows, iou_floor=0.5):
    """Diagnostic: how many entries match a raw GT box of the same class at IoU >= iou_floor.

    Uses the frozen raw annotations (category 1..10), never the certified region-filtered GT; it is a
    reachability diagnostic for the added predictions, not a metric.
    """
    from collections import defaultdict
    by_class = defaultdict(list)
    for row in raw_rows:
        if int(row["category"]) in range(1, NUM_CLASSES + 1):
            by_class[int(row["category"])].append([float(v) for v in row["box"]])
    hits, total = 0, 0
    for entry in entries:
        category = int(entry["category_id"])
        boxes = by_class.get(category, [])
        total += 1
        if boxes and float(xyxy_iou_matrix([entry["bbox_xyxy"]], boxes).max()) >= iou_floor:
            hits += 1
    return {"entries": total, "matched": hits, "iou_floor": iou_floor}
