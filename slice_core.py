"""Frozen, GT-free CPU geometry/merge core for the R18 SAHI-style adaptation.

This is not an implementation of the entire original SAHI recipe.  The caller
must obtain Q300 x 10 sigmoid probabilities in the original FP32 path and decode
normalized cxcywh with the certified FP32 decode helper before using this core.
The whole-image A branch is left untouched.  Boxes here are pixel xyxy values;
crop mapping only adds the integer crop origin.  No clipping, score threshold,
padding, GT access, model construction, or torch import occurs in this module.

All NMS geometry uses Python's IEEE-754 binary64 arithmetic.  Scores are never
recomputed or rounded.  Rank is score descending, then view/query/category IDs
ascending.  Category IDs are one-based.  Query IDs are local to a view and never
define cross-view correspondence.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

import numpy as np


CORE_VERSION = "r18-sahi-fixed-v1"
NMS_IOU = 0.5
PER_VIEW_BUDGET = 600
FINAL_BUDGETS = (600, 300)
EXPECTED_SHAPE = (300, 10)


def _integer(value: object, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}: {value!r}")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number, not bool")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite: {value!r}")
    return result


def _box64(box: Sequence[float]) -> tuple[float, float, float, float]:
    if len(box) != 4:
        raise ValueError(f"xyxy box must contain four coordinates: {box!r}")
    x0, y0, x1, y1 = tuple(_finite(v, "box coordinate") for v in box)
    width, height = x1 - x0, y1 - y0
    area = width * height
    if not (math.isfinite(width) and math.isfinite(height)
            and math.isfinite(area) and width > 0 and height > 0 and area > 0):
        raise ValueError(f"box must have finite positive width, height, area: {box!r}")
    return x0, y0, x1, y1


@dataclass(frozen=True)
class Tile:
    """An exact half-open PIL crop; view IDs 1..4 are TL, TR, BL, BR."""

    view_id: int
    x0: int
    y0: int
    x1: int
    y1: int
    image_width: int
    image_height: int

    def __post_init__(self) -> None:
        for key in ("view_id", "x0", "y0", "x1", "y1", "image_width", "image_height"):
            _integer(getattr(self, key), key)
        if not (1 <= self.view_id <= 4 and self.image_width > 0 and self.image_height > 0
                and 0 <= self.x0 < self.x1 <= self.image_width
                and 0 <= self.y0 < self.y1 <= self.image_height):
            raise ValueError(f"invalid half-open tile: {self!r}")

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def crop_box(self) -> tuple[int, int, int, int]:
        return self.x0, self.y0, self.x1, self.y1

    def as_dict(self) -> dict:
        return {
            "view_id": self.view_id, "crop_box": list(self.crop_box),
            "image_width": self.image_width, "image_height": self.image_height,
            "crop_width": self.width, "crop_height": self.height,
        }


def tile_windows(height: int, width: int) -> list[Tile]:
    """Return all four IDs even when tiny-image crop geometries coincide."""
    height = _integer(height, "height", 1)
    width = _integer(width, "width", 1)
    tile_h = min(height, (5 * height + 8) // 9)  # exact ceil(5H/9)
    tile_w = min(width, (5 * width + 8) // 9)
    return [Tile(i, x, y, x + tile_w, y + tile_h, width, height)
            for i, (y, x) in enumerate(
                ((0, 0), (0, width - tile_w),
                 (height - tile_h, 0), (height - tile_h, width - tile_w)), 1)]


def tile_geometry(height: int, width: int) -> dict:
    tiles = tile_windows(height, width)
    tile = tiles[0]
    overlap_h, overlap_w = 2 * tile.height - height, 2 * tile.width - width
    return {
        "image_height": height, "image_width": width,
        "tile_height": tile.height, "tile_width": tile.width,
        "overlap_height_px": overlap_h, "overlap_width_px": overlap_w,
        "overlap_height_fraction_of_tile": overlap_h / tile.height,
        "overlap_width_fraction_of_tile": overlap_w / tile.width,
        "tiles": [item.as_dict() for item in tiles],
    }


def map_crop_xyxy(boxes_xyxy: Sequence[Sequence[float]], tile: Tile) -> list[tuple]:
    """Map crop-local pixel boxes without clipping, including out-of-crop boxes."""
    if not isinstance(tile, Tile):
        raise ValueError("tile must be a validated Tile")
    mapped = []
    for box in boxes_xyxy:
        x0, y0, x1, y1 = _box64(box)
        mapped.append(_box64((x0 + tile.x0, y0 + tile.y0,
                              x1 + tile.x0, y1 + tile.y0)))
    return mapped


@dataclass(frozen=True)
class Candidate:
    box_xyxy: tuple[float, float, float, float]
    score: float
    view_id: int
    query_id: int
    category_id: int
    view_candidate_rank: int
    input_shape: tuple[int, int] = EXPECTED_SHAPE

    def __post_init__(self) -> None:
        box = _box64(self.box_xyxy)
        score = _finite(self.score, "score")
        if not 0 <= score <= 1:
            raise ValueError("sigmoid probability must lie in [0,1]")
        if len(self.input_shape) != 2:
            raise ValueError("input_shape must be (queries, classes)")
        shape = tuple(_integer(v, "shape dimension", 1) for v in self.input_shape)
        _integer(self.view_id, "view_id")
        _integer(self.query_id, "query_id")
        _integer(self.category_id, "category_id", 1)
        _integer(self.view_candidate_rank, "view_candidate_rank", 1)
        if self.view_id > 4 or self.query_id >= shape[0] or self.category_id > shape[1]:
            raise ValueError(f"candidate identity outside view/input shape: {self!r}")
        if self.view_candidate_rank > shape[0] * shape[1]:
            raise ValueError("view_candidate_rank exceeds input score count")
        object.__setattr__(self, "box_xyxy", box)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "input_shape", shape)

    @property
    def identity(self) -> tuple[int, int, int]:
        return self.view_id, self.query_id, self.category_id

    @property
    def sort_key(self) -> tuple:
        return -self.score, self.view_id, self.query_id, self.category_id

    def as_dict(self) -> dict:
        return {
            "box_xyxy": list(self.box_xyxy), "score": self.score,
            "view_id": self.view_id, "query_id": self.query_id,
            "category_id": self.category_id,
            "view_candidate_rank": self.view_candidate_rank,
            "input_shape": list(self.input_shape),
        }

    @classmethod
    def from_dict(cls, item: Mapping) -> "Candidate":
        return cls(
            box_xyxy=tuple(item["box_xyxy"]), score=item["score"],
            view_id=item["view_id"], query_id=item["query_id"],
            category_id=item["category_id"],
            view_candidate_rank=item["view_candidate_rank"],
            input_shape=tuple(item["input_shape"]),
        )


def select_view_candidates(
    boxes_xyxy: Sequence[Sequence[float]],
    probabilities: Sequence[Sequence[float]],
    *, view_id: int, limit: int = PER_VIEW_BUDGET,
    expected_shape: tuple[int, int] = EXPECTED_SHAPE,
) -> list[Candidate]:
    """Select cross-class top600 from original FP32 sigmoid scores; never sigmoid.

    The generic shape argument exists for synthetic tests.  The execution
    contract fixes its production value at (300,10), view IDs at 0..4, limit600.
    All boxes/scores are validated, including values outside the selected budget.
    """
    _integer(view_id, "view_id")
    if view_id > 4:
        raise ValueError("view_id must be 0..4")
    _integer(limit, "limit", 1)
    if len(expected_shape) != 2:
        raise ValueError("expected_shape must contain queries and classes")
    nq, nc = tuple(_integer(v, "shape dimension", 1) for v in expected_shape)
    if len(boxes_xyxy) != nq or len(probabilities) != nq:
        raise ValueError("box/score query dimension differs from expected_shape")
    boxes = [_box64(box) for box in boxes_xyxy]
    scored = []
    for query_id, row in enumerate(probabilities):
        if len(row) != nc:
            raise ValueError("score class dimension differs from expected_shape")
        for class_index, value in enumerate(row):
            score = _finite(value, "score")
            if not 0 <= score <= 1:
                raise ValueError("sigmoid probability must lie in [0,1]")
            scored.append((score, query_id, class_index + 1))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [Candidate(boxes[q], score, view_id, q, category, rank, (nq, nc))
            for rank, (score, q, category) in enumerate(scored[:limit], 1)]


def box_iou64(first: Sequence[float], second: Sequence[float]) -> float:
    a, b = _box64(first), _box64(second)
    intersection = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection
    if not math.isfinite(union) or union <= 0 or not math.isfinite(intersection):
        raise ValueError("IoU intersection/union is not finite and valid")
    iou = intersection / union
    if not math.isfinite(iou) or not 0 <= iou <= 1:
        raise ValueError("IoU outside [0,1]")
    return iou


def _validated_candidates(candidates: Iterable[Candidate]) -> list[Candidate]:
    items = list(candidates)
    identities = set()
    shape = None
    for item in items:
        if not isinstance(item, Candidate):
            raise ValueError("candidates must contain validated Candidate objects")
        if item.identity in identities:
            raise ValueError(f"duplicate candidate identity: {item.identity}")
        identities.add(item.identity)
        if shape is None:
            shape = item.input_shape
        elif shape != item.input_shape:
            raise ValueError("mixed model output shapes cannot be merged")
    return items


def classwise_nms(candidates: Iterable[Candidate]) -> list[Candidate]:
    """Classwise hard NMS; suppress iff IoU > 0.5, preserve exact ties stably."""
    ordered = sorted(_validated_candidates(candidates), key=lambda item: item.sort_key)
    by_class: dict[int, list[int]] = {}
    for index, candidate in enumerate(ordered):
        by_class.setdefault(candidate.category_id, []).append(index)
    kept_indices = set()
    for indices in by_class.values():
        boxes = np.asarray([ordered[index].box_xyxy for index in indices], dtype=np.float64)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        remaining = np.arange(len(indices), dtype=np.int64)
        while remaining.size:
            first = int(remaining[0])
            kept_indices.add(indices[first])
            rest = remaining[1:]
            if not rest.size:
                break
            left = np.maximum(boxes[first, :2], boxes[rest, :2])
            right = np.minimum(boxes[first, 2:], boxes[rest, 2:])
            extent = np.maximum(0.0, right - left)
            intersection = extent[:, 0] * extent[:, 1]
            union = areas[first] + areas[rest] - intersection
            if not (np.isfinite(intersection).all() and np.isfinite(union).all()
                    and (union > 0).all()):
                raise ValueError("NMS intersection/union is not finite and valid")
            iou = intersection / union
            if not (np.isfinite(iou).all() and (iou >= 0).all() and (iou <= 1).all()):
                raise ValueError("NMS IoU outside [0,1]")
            remaining = rest[iou <= NMS_IOU]
    return [item for index, item in enumerate(ordered) if index in kept_indices]


def merge_candidates(candidates: Iterable[Candidate]) -> dict:
    """B: view0 only; C: views0..4. Caller supplies the frozen branch's views."""
    items = _validated_candidates(candidates)
    kept = classwise_nms(items)
    return {
        "core_version": CORE_VERSION, "nms_iou": NMS_IOU,
        "candidate_count": len(items), "kept_count": len(kept),
        "suppressed_count": len(items) - len(kept),
        "top600": kept[:600], "top300": kept[:300],
    }


def serialize_candidates(candidates: Iterable[Candidate]) -> list[dict]:
    return [item.as_dict() for item in _validated_candidates(candidates)]


def deserialize_candidates(records: Iterable[Mapping]) -> list[Candidate]:
    return _validated_candidates(Candidate.from_dict(record) for record in records)
