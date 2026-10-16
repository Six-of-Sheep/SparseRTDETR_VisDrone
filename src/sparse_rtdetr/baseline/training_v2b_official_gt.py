"""Official raw-lineage GT semantics for an independently named development analysis.

Historical conversion artifacts are immutable. In particular, score-zero class 0
rows must still be spatial ignore regions; class 11 is unscored, never a region.
This adapter does not certify the evaluator or claim raw-byte access when it only
authenticates the complete development lineage and manifest.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import stat

from ..data_protocol.evaluation import (
    Detection, PrimaryEvaluatorInputV2, PrimaryGroundTruth, PrimaryImageV2,
)
from ..data_protocol.parser import parse_annotation_line, parse_annotation_bytes
from ..data_protocol.schema import stable_annotation_id
from . import training_v2b_development as development
from .training_v2b_evidence import canonical_sha256, file_reference, strict_json_loads


class OfficialGroundTruthError(ValueError):
    """Incomplete development provenance or ambiguous raw annotation semantics."""


POLICY = {
    "id": "development_raw_lineage_official_gt_v1",
    "category_0": "spatial_ignore_regardless_of_score",
    "category_11": "unscored_nonspatial",
    "categories_1_to_10_score_zero": "class_specific_ignored_gt",
    "duplicates": "retain_every_raw_row_no_converter_deduplication",
    "boxes": "unclipped_original_image_xyxy_positive_area",
    "historical_conversion_modified": False,
}


def _require(value, message):
    if not value:
        raise OfficialGroundTruthError(message)


def read_regular_reference(reference: dict) -> bytes:
    """Authenticate exact bytes via non-following directory descriptors."""
    _require(type(reference) is dict and set(reference) in (
        {"path", "sha256", "size_bytes"}, {"path", "sha256", "size_bytes", "sha256_scope"}),
        "file reference schema differs")
    if "sha256_scope" in reference:
        _require(reference["sha256_scope"] == "complete_file_bytes", "file SHA scope differs")
    path = Path(reference["path"])
    _require(path.is_absolute() and ".." not in path.parts, "reference path must be absolute")
    development._sha(reference["sha256"], "reference")
    development._integer(reference["size_bytes"], "reference bytes")
    directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in path.parent.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
    finally:
        os.close(directory)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        _require(stat.S_ISREG(before.st_mode), "reference is not a regular file")
        _require(before.st_size == reference["size_bytes"], "reference size differs")
        raw = stream.read(before.st_size + 1)
        after = os.fstat(stream.fileno())
    keys = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    _require(tuple(getattr(before, k) for k in keys) == tuple(getattr(after, k) for k in keys),
             "reference changed while reading")
    _require(len(raw) == reference["size_bytes"]
             and hashlib.sha256(raw).hexdigest() == reference["sha256"], "reference SHA differs")
    return raw


def _documents(development_binding, lineage_reference, raw_annotation_root):
    checked = development.validate_development_binding(development_binding)
    lineage_path = development._role_path(lineage_reference["path"], "development lineage")
    _require(lineage_path.name == "development_lineage.jsonl", "only development lineage is allowed")
    lineage_raw = read_regular_reference(lineage_reference)
    rows = [strict_json_loads(line) for line in lineage_raw.splitlines()]
    coco = strict_json_loads(read_regular_reference(checked["annotation"]))
    manifest = strict_json_loads(read_regular_reference(checked["manifest"]))
    images, grouped, members, _ = development._validate_documents(coco, manifest)
    image_by_stable = {image["stable_image_id"]: image for image in images}
    by_image = {key: [] for key in image_by_stable}
    seen = set()
    truth, attributes = [], []
    for row in rows:
        _require(type(row) is dict and row.get("split") == "val"
                 and row.get("image_id") in by_image, "lineage crosses the bound development members")
        image = image_by_stable[row["image_id"]]
        fields = row.get("raw_fields")
        line = row.get("physical_line_number")
        _require(type(fields) is list and len(fields) == 8
                 and all(type(v) is str for v in fields)
                 and type(line) is int and line >= 1, "lineage raw fields or line number differ")
        parsed = parse_annotation_line(",".join(fields).encode("utf-8"), line).fields
        _require(parsed.category in range(12) and parsed.score in (0.0, 1.0)
                 and parsed.width > 0 and parsed.height > 0,
                 "unsupported raw category/score or nonpositive box; do not silently discard")
        _require(row.get("source_relative_path") == image["file_name"]
                 and row.get("raw_category_id") == parsed.category
                 and row.get("bbox_xyxy") == list(parsed.bbox_xyxy)
                 and row.get("area") == parsed.area, "lineage raw geometry/identity drift")
        development._sha(row.get("raw_line_sha256"), "raw line")
        annotation_id = stable_annotation_id(row["image_id"], line, row["raw_line_sha256"])
        _require(row.get("annotation_id") == annotation_id and annotation_id not in seen,
                 "lineage annotation identity missing or repeated")
        seen.add(annotation_id)
        by_image[row["image_id"]].append(row)
        box = parsed.bbox_xyxy
        truth.append(PrimaryGroundTruth(
            annotation_id, row["image_id"], parsed.category, box,
            (box[2] - box[0]) * (box[3] - box[1]),
            parsed.category == 0, parsed.score == 0.0,
        ))
        attributes.append({
            "annotation_id": annotation_id, "image_id": image["id"],
            "stable_image_id": row["image_id"], "raw_category_id": parsed.category,
            "raw_score": parsed.score, "bbox_xyxy": list(box), "area": parsed.area,
            "truncation": parsed.truncation, "occlusion": parsed.occlusion,
            "physical_line_number": line, "raw_line_sha256": row["raw_line_sha256"],
            "historical_disposition": row.get("reason_code"),
            "historical_ignore_region": row.get("ignore_region"),
        })
    raw_root = None
    if raw_annotation_root is not None:
        raw_root = development._role_path(raw_annotation_root, "raw annotation root")
        _require(raw_root.is_dir() and not raw_root.is_symlink(), "raw annotation root is not a directory")
    for image in images:
        record = members[image["id"]]
        selected = sorted(by_image[image["stable_image_id"]], key=lambda row: row["physical_line_number"])
        _require(type(record.get("raw_annotation_rows")) is int
                 and len(selected) == record["raw_annotation_rows"]
                 and [row["physical_line_number"] for row in selected] == list(range(1, len(selected) + 1)),
                 "lineage does not cover every physical line declared in development manifest")
        if raw_root is not None:
            relative = Path(record["annotation_relative_path"])
            _require(not relative.is_absolute() and ".." not in relative.parts
                     and relative.parts[:2] == ("val", "annotations") and relative.suffix == ".txt",
                     "raw annotation path crosses another data role")
            path = raw_root / relative
            ref = {"path": str(path), "sha256": record["annotation_sha256"],
                   "size_bytes": record["annotation_size_bytes"]}
            parsed_rows = parse_annotation_bytes(read_regular_reference(ref))
            _require(len(parsed_rows) == len(selected), "raw annotation completeness differs")
            for actual, row in zip(parsed_rows, selected):
                _require(actual.physical_line_number == row["physical_line_number"]
                         and actual.raw_line_sha256 == row["raw_line_sha256"]
                         and list(actual.raw_fields) == row["raw_fields"], "raw bytes differ from lineage")
    truth_by_id = {row.annotation_id: row for row in truth}
    for image in images:
        for ann in grouped[image["id"]]:
            raw = truth_by_id.get(ann.get("stable_annotation_id"))
            x, y, w, h = ann["bbox"]
            _require(raw is not None and raw.image_id == image["stable_image_id"]
                     and raw.category_id == ann["category_id"] and not raw.ignored
                     and not raw.ignore_region and raw.bbox_xyxy == (x, y, x+w, y+h),
                     "converted formal GT cannot be traced to its raw row")
    image_map = {image["id"]: image["stable_image_id"] for image in images}
    order = {stable: index for index, stable in enumerate(image_map.values())}
    line_by_id = {row["annotation_id"]: row["physical_line_number"] for row in attributes}
    truth.sort(key=lambda row: (order[row.image_id], line_by_id[row.annotation_id]))
    attributes.sort(key=lambda row: (row["image_id"], row["physical_line_number"]))
    primary_images = tuple(PrimaryImageV2(image["stable_image_id"], image["width"], image["height"])
                           for image in images)
    return checked, rows, tuple(truth), primary_images, image_map, attributes



def _make_binding(development_binding, lineage_reference, raw_annotation_root):
    checked, rows, truth, images, image_map, attributes = _documents(
        development_binding, lineage_reference, raw_annotation_root)
    # Use physical raw order explicitly for stable matching when boxes/scores tie.
    line_by_id = {row["annotation_id"]: row["physical_line_number"] for row in attributes}
    integer_by_stable = {value: key for key, value in image_map.items()}
    truth = tuple(sorted(truth, key=lambda row: (integer_by_stable[row.image_id],
                                                line_by_id[row.annotation_id])))
    value = {
        "schema_version": 1, "role": "development", "policy": copy.deepcopy(POLICY),
        "annotation": checked["annotation"], "manifest": checked["manifest"],
        "lineage": copy.deepcopy(lineage_reference),
        "raw_annotation_root": None if raw_annotation_root is None else str(raw_annotation_root),
        "raw_bytes_independently_verified": raw_annotation_root is not None,
        "image_order_sha256": checked["image_order_sha256"],
        "image_count": len(images), "raw_annotation_count": len(truth),
        "raw_category_counts": {str(key): value for key, value in sorted(
            Counter(row.category_id for row in truth).items())},
        "spatial_ignore_region_count": sum(row.ignore_region for row in truth),
        "unscored_category_11_count": sum(row.category_id == 11 for row in truth),
        "class_specific_ignored_count": sum(row.ignored and row.category_id in range(1, 11) for row in truth),
        "historical_ignore_region_count": sum(row.get("ignore_region") is True for row in rows),
        "ground_truth_sha256": canonical_sha256([
            {**asdict(row), "bbox_xyxy": list(row.bbox_xyxy)} for row in truth]),
        "diagnostic_attributes_sha256": canonical_sha256(attributes),
        "evaluator_certification_claimed": False,
        "adapter_source": file_reference(Path(__file__).absolute()),
    }
    value["binding_sha256"] = canonical_sha256(value)
    return value, PrimaryEvaluatorInputV2(images, (), truth), image_map, attributes


def build_official_gt_binding(*, development_binding: dict, lineage_reference: dict,
                              raw_annotation_root: str | None = None) -> dict:
    """Bind complete raw GT; optional root contains only authorized val/annotations paths."""
    return _make_binding(development_binding, lineage_reference, raw_annotation_root)[0]


def load_official_ground_truth(binding: dict, development_binding: dict):
    """Return (empty-detection input, integer/stable ID map, diagnostic attributes)."""
    _require(type(binding) is dict and binding.get("role") == "development",
             "official GT binding must declare development")
    rebuilt, value, image_map, attributes = _make_binding(
        development_binding, binding["lineage"], binding["raw_annotation_root"])
    _require(rebuilt == binding, "official GT/source binding changed")
    return value, image_map, attributes


def with_predictions(value: PrimaryEvaluatorInputV2, image_id_map: dict,
                     predictions: list[dict]) -> PrimaryEvaluatorInputV2:
    """Preserve every already-sorted flattened prediction and its original order."""
    detections = []
    for row in predictions:
        _require(type(row) is dict and row.get("image_id") in image_id_map,
                 "prediction does not belong to the bound development set")
        x, y, width, height = row["bbox"]
        detections.append(Detection(image_id_map[row["image_id"]], row["category_id"],
                                    (x, y, x+width, y+height), row["score"]))
    return PrimaryEvaluatorInputV2(value.images, tuple(detections), value.ground_truth)
