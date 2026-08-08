"""Pure-CPU VisDrone protocol contracts.

This package contains schemas and deterministic algorithms only. It does not
read a dataset directory, instantiate a model, or evaluate predictions.
"""

from .categories import (
    COCO_CATEGORY_IDS,
    NUM_CLASSES,
    CategoryMapping,
    category_mapping,
    coco_to_training_id,
    raw_to_training_id,
    training_to_coco_id,
    training_to_raw_id,
)
from .evaluation import (
    COCOGroundTruth,
    COCODiagnosticInput,
    COCODiagnosticOutput,
    Detection,
    PrimaryGroundTruth,
    PrimaryEvaluatorInput,
    PrimaryEvaluatorOutput,
    assert_primary_input,
    assert_secondary_input,
)
from .lineage import AnnotationDisposition, AnnotationLineage, AnnotationStatus, annotate_records, classify_annotation
from .parser import AnnotationFields, AnnotationParseError, ParsedAnnotation, parse_annotation_bytes, parse_annotation_line
from .protocol import PROTOCOL_SCHEMA, protocol_schema
from .schema import ProtocolContractError, canonical_json_bytes, ensure_allowed_dataset_path, stable_annotation_id, stable_image_id
from .split import AtomicGroup, ImageIdentity, SplitPlan, build_atomic_groups, plan_confirmatory_split

__all__ = [
    "AnnotationDisposition", "AnnotationFields", "AnnotationLineage", "AnnotationParseError", "AnnotationStatus",
    "AtomicGroup", "COCO_CATEGORY_IDS", "COCOGroundTruth", "COCODiagnosticInput", "COCODiagnosticOutput",
    "CategoryMapping", "Detection", "ImageIdentity", "NUM_CLASSES", "PROTOCOL_SCHEMA", "ParsedAnnotation",
    "PrimaryEvaluatorInput", "PrimaryEvaluatorOutput", "PrimaryGroundTruth", "ProtocolContractError", "SplitPlan", "annotate_records",
    "assert_primary_input", "assert_secondary_input", "build_atomic_groups", "canonical_json_bytes",
    "category_mapping", "coco_to_training_id", "ensure_allowed_dataset_path", "parse_annotation_bytes",
    "parse_annotation_line", "plan_confirmatory_split", "protocol_schema", "raw_to_training_id",
    "stable_annotation_id", "stable_image_id", "training_to_coco_id", "training_to_raw_id",
]
