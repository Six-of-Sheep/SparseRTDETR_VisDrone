"""Annotation disposition and source-lineage adapter for in-memory records."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .categories import CategoryMapping, category_mapping
from .parser import AnnotationFields, ParsedAnnotation
from .schema import ProtocolContractError, ensure_allowed_dataset_path, stable_annotation_id, stable_image_id


class AnnotationStatus(str, Enum):
    KEEP = "keep"
    IGNORE = "ignore"
    FILTERED = "filtered"


@dataclass(frozen=True)
class AnnotationDisposition:
    status: AnnotationStatus
    reason_code: str
    ignore_region: bool
    mapping: CategoryMapping


@dataclass(frozen=True)
class AnnotationLineage:
    split: str
    source_relative_path: str
    physical_line_number: int
    raw_fields: tuple[str, ...]
    raw_line_sha256: str
    image_id: str
    annotation_id: str
    bbox_xyxy: tuple[float, float, float, float]
    area: float
    raw_category_id: int
    training_category_id: int | None
    coco_category_id: int | None
    status: AnnotationStatus
    reason_code: str
    ignore_region: bool


def classify_annotation(fields: AnnotationFields) -> AnnotationDisposition:
    """Apply official score/category/area semantics without clipping."""

    mapping = category_mapping(fields.category)
    if fields.score == 0:
        return AnnotationDisposition(AnnotationStatus.IGNORE, "SCORE_ZERO_IGNORED", False, mapping)
    if fields.category == 0:
        return AnnotationDisposition(AnnotationStatus.IGNORE, "SPATIAL_IGNORE_REGION", True, mapping)
    if fields.category == 11:
        return AnnotationDisposition(AnnotationStatus.FILTERED, "UNSCORED_CATEGORY_11", False, mapping)
    if fields.area <= 0:
        return AnnotationDisposition(AnnotationStatus.FILTERED, "NON_POSITIVE_FORMAL_BBOX", False, mapping)
    return AnnotationDisposition(AnnotationStatus.KEEP, "MATCHABLE_FORMAL_GT", False, mapping)


def annotate_records(
    split: str,
    source_relative_path: str,
    parsed_records: tuple[ParsedAnnotation, ...] | list[ParsedAnnotation],
) -> tuple[AnnotationLineage, ...]:
    """Attach stable IDs and duplicate/filter reasons to parsed records."""

    normalized_path = ensure_allowed_dataset_path(source_relative_path, split)
    image_id = stable_image_id(split, normalized_path)
    seen: set[tuple[float | int, ...]] = set()
    output: list[AnnotationLineage] = []
    for parsed in parsed_records:
        fields = parsed.fields
        key = fields.as_tuple()
        disposition = classify_annotation(fields)
        if key in seen:
            status = AnnotationStatus.FILTERED
            reason = "FILTERED_DUPLICATE_EXACT_GT"
            ignore_region = False
        else:
            seen.add(key)
            status = disposition.status
            reason = disposition.reason_code
            ignore_region = disposition.ignore_region
        output.append(AnnotationLineage(
            split=split,
            source_relative_path=normalized_path,
            physical_line_number=parsed.physical_line_number,
            raw_fields=parsed.raw_fields,
            raw_line_sha256=parsed.raw_line_sha256,
            image_id=image_id,
            annotation_id=stable_annotation_id(image_id, parsed.physical_line_number, parsed.raw_line_sha256),
            bbox_xyxy=fields.bbox_xyxy,
            area=fields.area,
            raw_category_id=fields.category,
            training_category_id=disposition.mapping.training_category_id,
            coco_category_id=disposition.mapping.coco_category_id,
            status=status,
            reason_code=reason,
            ignore_region=ignore_region,
        ))
    return tuple(output)
