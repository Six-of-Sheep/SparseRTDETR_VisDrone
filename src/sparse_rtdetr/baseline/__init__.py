"""Frozen RT-DETRv2 R18 VisDrone baseline adapter interfaces."""

from .artifacts import RuntimePaths, R3ArtifactBinding, resolve_runtime_paths, validate_runtime_role, verify_r3_binding
from .categories import (
    coco_categories_to_model_labels,
    coco_category_to_model_label,
    map_coco_tensor_to_model,
    map_model_tensor_to_coco,
    model_labels_to_coco_categories,
    model_label_to_coco_category,
)
from .config import (
    build_baseline_config,
    build_r18_cpu_model,
    build_runtime_config,
    build_upstream_r18_cpu_model,
    canonical_config_bytes,
    load_isolated_vendor_config_dict,
)
from .contract import BaselineContractError

__all__ = [
    "BaselineContractError",
    "R3ArtifactBinding",
    "RuntimePaths",
    "build_baseline_config",
    "build_r18_cpu_model",
    "build_runtime_config",
    "build_upstream_r18_cpu_model",
    "canonical_config_bytes",
    "coco_categories_to_model_labels",
    "coco_category_to_model_label",
    "load_isolated_vendor_config_dict",
    "map_coco_tensor_to_model",
    "map_model_tensor_to_coco",
    "model_labels_to_coco_categories",
    "model_label_to_coco_category",
    "resolve_runtime_paths",
    "validate_runtime_role",
    "verify_r3_binding",
]
