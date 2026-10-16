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
from .converter import (
    ConversionBundle,
    ConversionContractError,
    RawImageRecord,
    assign_coco_image_ids,
    build_conversion_bundle,
    collect_split,
    record_from_bytes,
    run_conversion,
    write_conversion_artifacts,
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
from .protocol import PROTOCOL_SCHEMA, PROTOCOL_V2_SCHEMA, protocol_schema, protocol_v2_schema
from .schema import ProtocolContractError, canonical_json_bytes, ensure_allowed_dataset_path, stable_annotation_id, stable_image_id
from .split import AtomicGroup, ImageIdentity, ProtocolV2ContractError, SplitPlan, build_atomic_groups, plan_confirmatory_split, plan_confirmatory_split_v2

__all__ = [
    "AnnotationDisposition", "AnnotationFields", "AnnotationLineage", "AnnotationParseError", "AnnotationStatus",
    "AtomicGroup", "COCO_CATEGORY_IDS", "COCOGroundTruth", "COCODiagnosticInput", "COCODiagnosticOutput",
    "CategoryMapping", "ConversionBundle", "ConversionContractError", "Detection", "ImageIdentity", "NUM_CLASSES", "PROTOCOL_SCHEMA", "PROTOCOL_V2_SCHEMA", "ParsedAnnotation", "ProtocolV2ContractError",
    "PrimaryEvaluatorInput", "PrimaryEvaluatorOutput", "PrimaryGroundTruth", "ProtocolContractError", "SplitPlan", "annotate_records",
    "assert_primary_input", "assert_secondary_input", "assign_coco_image_ids", "build_atomic_groups", "build_conversion_bundle", "canonical_json_bytes",
    "category_mapping", "coco_to_training_id", "ensure_allowed_dataset_path", "parse_annotation_bytes",
    "parse_annotation_line", "plan_confirmatory_split", "plan_confirmatory_split_v2", "protocol_schema", "protocol_v2_schema", "raw_to_training_id",
    "record_from_bytes", "run_conversion", "stable_annotation_id", "stable_image_id", "training_to_coco_id", "training_to_raw_id",
    "RawImageRecord", "collect_split", "write_conversion_artifacts",
]
