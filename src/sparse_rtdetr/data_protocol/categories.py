"""Explicit VisDrone category and ignore-region mapping."""

from __future__ import annotations

from dataclasses import dataclass

from .schema import ProtocolContractError


NUM_CLASSES = 10
COCO_CATEGORY_IDS = tuple(range(1, 11))
CATEGORY_NAMES = {
    1: "pedestrian", 2: "people", 3: "bicycle", 4: "car", 5: "van",
    6: "truck", 7: "tricycle", 8: "awning-tricycle", 9: "bus", 10: "motor",
}


@dataclass(frozen=True)
class CategoryMapping:
    raw_category_id: int
    name: str | None
    training_category_id: int | None
    coco_category_id: int | None
    enters_matching: bool
    reason_code: str


def category_mapping(raw_category_id: int) -> CategoryMapping:
    if raw_category_id in CATEGORY_NAMES:
        training = raw_category_id - 1
        return CategoryMapping(raw_category_id, CATEGORY_NAMES[raw_category_id], training, training + 1, True, "FORMAL_SCORED_CATEGORY")
    if raw_category_id == 0:
        return CategoryMapping(0, "ignored_region", None, None, False, "SPATIAL_IGNORE_REGION")
    if raw_category_id == 11:
        return CategoryMapping(11, "others", None, None, False, "UNSCORED_CATEGORY_11")
    raise ProtocolContractError(f"unsupported VisDrone category: {raw_category_id}")


def raw_to_training_id(raw_category_id: int) -> int:
    mapping = category_mapping(raw_category_id)
    if mapping.training_category_id is None:
        raise ProtocolContractError("non-formal category has no training ID")
    return mapping.training_category_id


def training_to_raw_id(training_category_id: int) -> int:
    if training_category_id not in range(NUM_CLASSES):
        raise ProtocolContractError("training category must be in [0, 9]")
    return training_category_id + 1


def training_to_coco_id(training_category_id: int) -> int:
    return training_to_raw_id(training_category_id)


def coco_to_training_id(coco_category_id: int) -> int:
    if coco_category_id not in COCO_CATEGORY_IDS:
        raise ProtocolContractError("COCO category ID is outside VisDrone scored classes")
    return coco_category_id - 1
