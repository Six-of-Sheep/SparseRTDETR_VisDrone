"""Separate schemas for the two non-interchangeable evaluators."""

from __future__ import annotations

from dataclasses import dataclass

from .schema import ProtocolContractError


@dataclass(frozen=True)
class Detection:
    image_id: str
    category_id: int
    bbox_xyxy: tuple[float, float, float, float]
    score: float


@dataclass(frozen=True)
class PrimaryGroundTruth:
    annotation_id: str
    image_id: str
    category_id: int | None
    bbox_xyxy: tuple[float, float, float, float]
    area: float
    ignore_region: bool
    ignored: bool


@dataclass(frozen=True)
class COCOGroundTruth:
    annotation_id: str
    image_id: str
    category_id: int
    bbox_xywh: tuple[float, float, float, float]
    area: float
    iscrowd: int = 0


@dataclass(frozen=True)
class PrimaryEvaluatorInput:
    image_ids: tuple[str, ...]
    detections: tuple[Detection, ...]
    ground_truth: tuple[PrimaryGroundTruth, ...]
    protocol: str = "visdrone_official_style_v1"
    iou_thresholds: tuple[float, ...] = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
    max_dets: tuple[int, ...] = (1, 10, 100, 500)
    nms: bool = False


@dataclass(frozen=True)
class COCODiagnosticInput:
    image_ids: tuple[str, ...]
    detections: tuple[Detection, ...]
    ground_truth: tuple[COCOGroundTruth, ...]
    protocol: str = "coco_secondary_vendor_v1"
    max_dets: tuple[int, ...] = (1, 10, 100)
    area_ranges: tuple[str, ...] = ("all", "small", "medium", "large")
    nms: bool = False


@dataclass(frozen=True)
class PrimaryEvaluatorOutput:
    protocol: str = "visdrone_official_style_v1"
    metric_names: tuple[str, ...] = ("AP", "AP50", "AP75")
    status: str = "schema_only"
    metrics_accessed: bool = False


@dataclass(frozen=True)
class PrimaryImageV2:
    """Image metadata required by the official ignored-region rasterizer."""

    image_id: str
    width: int
    height: int


@dataclass(frozen=True)
class PrimaryEvaluatorInputV2:
    """Immutable, versioned input for the source-exact primary evaluator."""

    images: tuple[PrimaryImageV2, ...]
    detections: tuple[Detection, ...]
    ground_truth: tuple[PrimaryGroundTruth, ...]
    protocol: str = "visdrone_official_style_v1"
    iou_thresholds: tuple[float, ...] = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
    max_dets: tuple[int, ...] = (1, 10, 100, 500)
    nms: bool = False
    ignore_threshold: float = 0.5
    standard_overlap: str = "intersection_over_union"
    ignored_overlap: str = "intersection_over_detection_area"


@dataclass(frozen=True)
class PrimaryEvaluatorResultV2:
    """Detached deterministic metrics and intermediate evidence."""

    protocol: str
    metric_names: tuple[str, ...]
    metrics: tuple[float, ...]
    AP: float
    AP50: float
    AP75: float
    AR1: float
    AR10: float
    AR100: float
    AR500: float
    evaluated_image_count: int
    evaluated_detection_count: int
    evaluated_ground_truth_count: int
    canonical_input_sha256: str
    contract_raw_sha256: str
    contract_canonical_sha256: str
    authority_manifest_raw_sha256: str
    authority_manifest_canonical_sha256: str
    eval_class_occurrences: tuple[int, ...]
    ap_by_class_iou: tuple[tuple[float, ...], ...]
    ar_by_class_iou_max_dets: tuple[tuple[tuple[float, ...], ...], ...]
    match_counts_by_class_iou_max_dets: tuple[tuple[tuple[tuple[int, ...], ...], ...], ...]
    canonical_result_sha256: str


@dataclass(frozen=True)
class COCODiagnosticOutput:
    protocol: str = "coco_secondary_vendor_v1"
    metric_names: tuple[str, ...] = ("AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large")
    status: str = "schema_only"
    metrics_accessed: bool = False


def assert_primary_input(value: object) -> PrimaryEvaluatorInput:
    if not isinstance(value, PrimaryEvaluatorInput):
        raise ProtocolContractError("primary evaluator cannot consume secondary COCO schema")
    return value


def assert_secondary_input(value: object) -> COCODiagnosticInput:
    if not isinstance(value, COCODiagnosticInput):
        raise ProtocolContractError("secondary evaluator cannot consume VisDrone primary schema")
    return value
