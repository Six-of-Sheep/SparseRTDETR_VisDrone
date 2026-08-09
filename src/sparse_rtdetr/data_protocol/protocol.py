"""The frozen, data-free VisDrone protocol configuration."""

from __future__ import annotations

from copy import deepcopy


PROTOCOL_SCHEMA = {
    "schema_version": 1,
    "protocol_id": "P3-VISDRONE-DATA-PROTOCOL-V1",
    "data_identity": {
        "source": "official VisDrone train and val only",
        "train": {
            "images": 6471,
            "annotations": 6471,
            "image_inventory_sha256": "390e35ee7ff3fe46b4c147230a6650f6001264d133ebbdc89f3de3c950a54e29",
            "annotation_inventory_sha256": "f03551a62514e84255bfb07c8d137740d2cd015d35e0043fb9b2137a1779dc62",
            "sequence_keys": 208,
        },
        "val": {
            "images": 548,
            "annotations": 548,
            "image_inventory_sha256": "feaa06717e35d33b85c84e2af567df90bed8eeb32ab0bd72289b0c421b2ebded",
            "annotation_inventory_sha256": "fd110fe03a8b4ab1f6446104c2a5c4c6878e42460ee60c00cf6edd43e9744b98",
            "sequence_keys": 76,
        },
        "cross_split_shared_sequence_keys": 24,
        "cross_split_exact_image_sha_overlap": 0,
        "test_access_allowed": False,
    },
    "parser": {
        "fields": ["left", "top", "width", "height", "score", "category", "truncation", "occlusion"],
        "trailing_empty_field": "allow_one_only",
        "finite_numbers": True,
        "category_integer": True,
        "bbox_xyxy": "[left, top, left + width, top + height]",
        "area": "width * height",
        "clipping": "none",
        "non_positive_formal_bbox": "retain_lineage_and_filter",
        "exact_duplicate": "first_keep_later_FILTERED_DUPLICATE_EXACT_GT",
    },
    "categories": {
        "raw_to_training": {str(i): i - 1 for i in range(1, 11)},
        "training_to_coco": {str(i - 1): i for i in range(1, 11)},
        "num_classes": 10,
        "background_explicit": False,
        "category_0": "spatial_ignore_region",
        "category_11": "unscored_others",
    },
    "evaluators": {
        "primary": {
            "id": "visdrone_official_style_v1",
            "iou_thresholds": "0.50:0.05:0.95",
            "max_dets": [1, 10, 100, 500],
            "ignore_semantics": "official_spatial_ignore_region",
            "ap": "official_VOC_style",
            "nms": False,
            "small_metric": False,
        },
        "secondary": {
            "id": "coco_secondary_vendor_v1",
            "max_dets": [1, 10, 100],
            "area_ranges": ["all", "small", "medium", "large"],
            "ignore_semantics": "COCO_iscrowd_only",
            "unmodified_vendor": True,
            "equivalent_to_primary": False,
        },
    },
    "split": {
        "development": "official_val",
        "confirmatory_source": "official_train_only",
        "test": "disabled",
        "group_unit": "sequence_key_connected_by_duplicate_image_sha",
        "seed": 20260808,
        "salt": "P3-confirmatory-v1",
        "target_confirmatory_images": 647,
        "distribution_tolerance_percentage_points": 5,
        "selection_allowed": False,
        "metrics_access_allowed": False,
        "single_final_access_only": True,
    },
}


PROTOCOL_V2_SCHEMA = deepcopy(PROTOCOL_SCHEMA)
PROTOCOL_V2_SCHEMA["protocol_id"] = "P3-VISDRONE-DATA-PROTOCOL-V2"
PROTOCOL_V2_SCHEMA["split"]["selection_policy"] = "feasibility_first_nearest_hash_prefix_v2"
PROTOCOL_V2_SCHEMA["split"]["planner"] = "plan_confirmatory_split_v2"
PROTOCOL_V2_SCHEMA["split"]["evaluated_prefix_count"] = 184
PROTOCOL_V2_SCHEMA["split"]["feasible_prefix_count"] = 64


def protocol_schema() -> dict:
    """Return a defensive copy suitable for assertions or serialization."""

    return deepcopy(PROTOCOL_SCHEMA)


def protocol_v2_schema() -> dict:
    """Return a defensive copy of the independent feasibility-first V2 schema."""

    return deepcopy(PROTOCOL_V2_SCHEMA)
