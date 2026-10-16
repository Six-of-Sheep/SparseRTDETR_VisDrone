"""CPU primary-AP resampling cache using the frozen evaluator's per-image math.

This cache does not certify the evaluator or change its GT adapter. It retains
global per-image maxDets before category filtering, spatial-ignore filtering,
ignored-GT IoA, the official denominator, and evalClass occurrence weighting.
Repeated images are independent literal copies; stable ties are ordered by
(original integer image ID, copy index, original prediction row).
"""
from __future__ import annotations

import copy
import hashlib
from collections import Counter, defaultdict
from dataclasses import replace
from typing import Mapping, Sequence

import numpy as np

from . import primary_evaluator as primary
from ..data_protocol.evaluation import PrimaryEvaluatorInputV2
from ..data_protocol.schema import canonical_json_bytes


class PrimaryCacheError(ValueError):
    """An input or resampling identity is inconsistent."""


def _mapping(value, image_id_map):
    if not isinstance(image_id_map, Mapping) or not image_id_map:
        raise PrimaryCacheError("a nonempty integer-to-primary-image mapping is required")
    if any(type(k) is not int or k < 0 or type(v) is not str or not v
           for k, v in image_id_map.items()):
        raise PrimaryCacheError("invalid integer/primary image identity")
    if len(set(image_id_map.values())) != len(image_id_map):
        raise PrimaryCacheError("primary image mapping is not one-to-one")
    if set(image_id_map.values()) != {x.image_id for x in value.images}:
        raise PrimaryCacheError("mapping must cover the input images exactly")
    return dict(sorted(image_id_map.items()))


def _instances(image_ids, allowed):
    ids = tuple(image_ids)
    if not ids or any(type(x) is not int or x not in allowed for x in ids):
        raise PrimaryCacheError("resampling IDs must be nonempty, known builtin integers")
    return tuple(sorted(ids))


def literal_resample_primary(value, image_id_map, image_ids):
    """Construct independent copied image/annotation IDs for a literal reference."""
    primary._validate_input(value)
    mapping = _mapping(value, image_id_map)
    ids = _instances(image_ids, mapping)
    images_by_id = {x.image_id: x for x in value.images}
    gt_by_id, det_by_id = defaultdict(list), defaultdict(list)
    for gt in value.ground_truth:
        gt_by_id[gt.image_id].append(gt)
    for det in value.detections:
        det_by_id[det.image_id].append(det)
    images, ground_truth, detections, copies = [], [], [], Counter()
    for original in ids:
        old = mapping[original]
        copy_index = copies[original]
        copies[original] += 1
        new = f"bootstrap:{original}:{copy_index}"
        images.append(replace(images_by_id[old], image_id=new))
        ground_truth.extend(replace(x, image_id=new,
                                    annotation_id=f"{new}:{x.annotation_id}")
                            for x in gt_by_id[old])
        detections.extend(replace(x, image_id=new) for x in det_by_id[old])
    return replace(value, images=tuple(images), detections=tuple(detections),
                   ground_truth=tuple(ground_truth))


def _vector_voc_ap(recall, precision):
    """Same integration as primary._voc_ap, with a vectorized upper envelope."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    changes = np.nonzero(mrec[1:] != mrec[:-1])[0] + 1
    return float(np.sum((mrec[changes] - mrec[changes - 1]) * mpre[changes]))


class PrimaryMetricCache:
    """Reusable per-image matching; score() recomputes dataset AP on each draw."""

    def __init__(self, value, binding, image_id_map):
        primary._binding_identity(binding)
        primary._validate_input(value)
        self.image_id_map = _mapping(value, image_id_map)
        self.image_ids = tuple(self.image_id_map)
        self.binding = copy.deepcopy(binding)
        self.input_sha256 = hashlib.sha256(
            canonical_json_bytes(primary._input_payload(value))).hexdigest()
        self.binding_sha256 = hashlib.sha256(canonical_json_bytes(binding)).hexdigest()
        self._records = {}
        image_by_id = {x.image_id: x for x in value.images}
        gt_by_id, det_by_id = defaultdict(list), defaultdict(list)
        for gt in value.ground_truth:
            gt_by_id[gt.image_id].append(gt)
        for det in value.detections:
            det_by_id[det.image_id].append(det)
        self.filtered_counts = {"ground_truth": 0, "detections": 0}
        for integer_id, string_id in self.image_id_map.items():
            image = image_by_id[string_id]
            regions = [x for x in gt_by_id[string_id] if x.ignore_region]
            integral = primary._build_integral(image, regions) if regions else None

            def retained(box):
                return (integral is None or
                        primary._ignored_fraction(integral, image, box)
                        < value.ignore_threshold)

            gt = [x for x in gt_by_id[string_id]
                  if x.category_id not in (0, 11) and retained(x.bbox_xyxy)]
            det = [x for x in det_by_id[string_id] if retained(x.bbox_xyxy)]
            self.filtered_counts["ground_truth"] += len(gt)
            self.filtered_counts["detections"] += len(det)
            # Frozen primary AP comes from maxDets=500, applied before category.
            det = det[:value.max_dets[-1]]
            for category in range(1, 11):
                targets = [x for x in gt if x.category_id == category]
                candidates = [x for x in det if x.category_id == category]
                gt_boxes = [(primary._official_xywh(x.bbox_xyxy), x.ignored)
                            for x in targets]
                det_boxes = [(x.score, primary._official_xywh(x.bbox_xyxy))
                             for x in candidates]
                states = np.asarray([
                    primary._match(det_boxes, gt_boxes, threshold)[1]
                    for threshold in value.iou_thresholds], dtype=np.int8)
                scores = np.asarray([x.score for x in candidates], dtype=np.float64)
                states.setflags(write=False)
                scores.setflags(write=False)
                self._records[integer_id, category] = (len(targets), scores, states)

    def score(self, image_ids: Sequence[int]) -> dict[str, float]:
        ids = _instances(image_ids, self.image_id_map)
        ap = np.zeros((10, 10), dtype=np.float64)
        ar = np.zeros_like(ap)
        occurrences = [category - 1 for i in ids for category in range(1, 11)
                       if self._records[i, category][0] > 0]
        for category in range(1, 11):
            rows = [self._records[i, category] for i in ids]
            denominator = max(1, sum(row[0] for row in rows))
            nonempty = [row for row in rows if row[1].size]
            if not nonempty:
                continue
            scores = np.concatenate([row[1] for row in nonempty])
            # Concatenation is image ID, copy index, original category-row order.
            order = np.argsort(-scores, kind="stable")
            states = np.concatenate([row[2] for row in nonempty], axis=1)[:, order]
            tp = np.cumsum(states == 1, axis=1, dtype=np.int64)
            fp = np.cumsum(states == 0, axis=1, dtype=np.int64)
            recall = tp.astype(np.float64) / denominator
            precision = tp.astype(np.float64) / np.maximum(1, tp + fp)
            for threshold in range(10):
                ap[category - 1, threshold] = _vector_voc_ap(
                    recall[threshold], precision[threshold]) * 100.0
                ar[category - 1, threshold] = recall[threshold, -1] * 100.0
        if not occurrences:
            return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "AR500": 0.0}
        selected = ap[occurrences]
        return {"AP": float(np.mean(selected)),
                "AP50": float(np.mean(selected[:, 0])),
                "AP75": float(np.mean(selected[:, 5])),
                "AR500": float(np.mean(ar[occurrences]))}

    def describe(self):
        return {"schema_version": 1, "input_sha256": self.input_sha256,
                "binding_sha256": self.binding_sha256,
                "image_ids": list(self.image_ids), "ap_max_dets": 500,
                "filtered_counts": dict(self.filtered_counts),
                "tie_order": "original_integer_image_id,copy_index,original_prediction_row",
                "aggregation": "dataset_VOC_AP_with_evalClass_occurrence_weighting",
                "certification_claimed": False}


def prepare_primary_cache(value: PrimaryEvaluatorInputV2, binding: dict,
                          image_id_map: Mapping[int, str]) -> PrimaryMetricCache:
    return PrimaryMetricCache(value, binding, image_id_map)
