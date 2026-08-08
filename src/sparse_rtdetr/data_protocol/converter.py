"""Production VisDrone conversion contracts with a deliberately inert CLI.

The conversion functions are usable for an explicitly authorized future
train/val run.  They never enumerate a generic dataset root and never expose
the test split.  The current phase exercises them only with synthetic bytes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping

from .categories import CATEGORY_NAMES, category_mapping
from .lineage import AnnotationLineage, AnnotationStatus, annotate_records
from .parser import ParsedAnnotation, parse_annotation_bytes
from .protocol import protocol_schema
from .schema import ProtocolContractError, canonical_json_bytes, ensure_allowed_dataset_path, stable_image_id
from .split import AtomicGroup, ImageIdentity, SplitPlan, build_atomic_groups, plan_confirmatory_split


class ConversionContractError(ProtocolContractError):
    """Raised when a production conversion precondition is not satisfied."""


CONVERTER_SCHEMA_VERSION = 1
ALLOWED_SPLITS = ("train", "val")
SPLIT_ORDER = {"train": 0, "val": 1}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MAX_ARTIFACT_BYTES = 1024 * 1024
AUTHORIZATION_ENV = "P3_VISDRONE_PRODUCTION_CONVERSION_AUTHORIZED"
EXPECTED_AUTHORIZATION = "1"


@dataclass(frozen=True)
class RawImageRecord:
    split: str
    relative_path: str
    annotation_relative_path: str
    sequence_key: str
    image_sha256: str
    image_size_bytes: int
    annotation_sha256: str
    annotation_size_bytes: int
    width: int
    height: int
    parsed_records: tuple[ParsedAnnotation, ...]
    lineage: tuple[AnnotationLineage, ...]

    @property
    def stable_image_id(self) -> str:
        return stable_image_id(self.split, self.relative_path)

    @property
    def valid_lineage(self) -> tuple[AnnotationLineage, ...]:
        return tuple(item for item in self.lineage if item.status is AnnotationStatus.KEEP)

    @property
    def image_key(self) -> tuple[str, str]:
        return self.split, self.relative_path

    def manifest_dict(self, coco_image_id: int) -> dict[str, object]:
        return {
            "split": self.split,
            "relative_path": self.relative_path,
            "annotation_relative_path": self.annotation_relative_path,
            "sequence_key": self.sequence_key,
            "stable_image_id": self.stable_image_id,
            "coco_image_id": coco_image_id,
            "image_sha256": self.image_sha256,
            "image_size_bytes": self.image_size_bytes,
            "annotation_sha256": self.annotation_sha256,
            "annotation_size_bytes": self.annotation_size_bytes,
            "width": self.width,
            "height": self.height,
            "raw_annotation_rows": len(self.lineage),
            "keep_rows": sum(item.status is AnnotationStatus.KEEP for item in self.lineage),
            "ignore_rows": sum(item.status is AnnotationStatus.IGNORE for item in self.lineage),
            "filtered_rows": sum(item.status is AnnotationStatus.FILTERED for item in self.lineage),
        }


@dataclass(frozen=True)
class ConversionBundle:
    train_records: tuple[RawImageRecord, ...]
    val_records: tuple[RawImageRecord, ...]
    train_core_records: tuple[RawImageRecord, ...]
    confirmatory_records: tuple[RawImageRecord, ...]
    development_records: tuple[RawImageRecord, ...]
    coco_image_ids: Mapping[tuple[str, str], int]
    atomic_groups: tuple[AtomicGroup, ...]
    split_plan: SplitPlan
    files: Mapping[str, bytes]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: object) -> bytes:
    return canonical_json_bytes(value)


def _jsonl_bytes(records: Iterable[Mapping[str, object]]) -> bytes:
    return b"".join(_canonical_json(record) + b"\n" for record in records)


def _sequence_key(relative_path: str) -> str:
    stem = PurePosixPath(relative_path).stem
    if "_" in stem:
        return stem.rsplit("_", 1)[0]
    match = re.match(r"^(.*?)(?:[-.]\d+)$", stem)
    return match.group(1) if match else stem


def _image_size(image_bytes: bytes) -> tuple[int, int]:
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n") and len(image_bytes) >= 24:
        return int.from_bytes(image_bytes[16:20], "big"), int.from_bytes(image_bytes[20:24], "big")
    if image_bytes.startswith(b"\xff\xd8"):
        index = 2
        while index + 9 < len(image_bytes):
            if image_bytes[index] != 0xFF:
                index += 1
                continue
            marker = image_bytes[index + 1]
            index += 2
            if marker in {0xD8, 0xD9}:
                continue
            if index + 2 > len(image_bytes):
                break
            segment_length = int.from_bytes(image_bytes[index:index + 2], "big")
            if segment_length < 2 or index + segment_length > len(image_bytes):
                break
            if marker in set(range(0xC0, 0xC4)) | set(range(0xC5, 0xC8)) | set(range(0xC9, 0xCC)) | set(range(0xCD, 0xD0)):
                if segment_length >= 7:
                    return int.from_bytes(image_bytes[index + 5:index + 7], "big"), int.from_bytes(image_bytes[index + 3:index + 5], "big")
            index += segment_length
    raise ConversionContractError("unsupported or malformed image header")


def record_from_bytes(
    split: str,
    relative_path: str,
    image_bytes: bytes,
    annotation_bytes: bytes,
    *,
    annotation_relative_path: str | None = None,
    width: int | None = None,
    height: int | None = None,
) -> RawImageRecord:
    """Build one source record from bytes; this is the synthetic-test seam."""

    normalized = ensure_allowed_dataset_path(relative_path, split)
    if split not in ALLOWED_SPLITS:
        raise ConversionContractError(f"unsupported split: {split}")
    parsed = parse_annotation_bytes(annotation_bytes)
    lineage = annotate_records(split, normalized, parsed)
    if width is None or height is None:
        width, height = _image_size(image_bytes)
    if width <= 0 or height <= 0:
        raise ConversionContractError("image dimensions must be positive")
    annotation_path = annotation_relative_path or str(PurePosixPath(normalized).with_suffix(".txt"))
    return RawImageRecord(
        split=split,
        relative_path=normalized,
        annotation_relative_path=ensure_allowed_dataset_path(annotation_path, split),
        sequence_key=_sequence_key(normalized),
        image_sha256=_sha256_bytes(image_bytes),
        image_size_bytes=len(image_bytes),
        annotation_sha256=_sha256_bytes(annotation_bytes),
        annotation_size_bytes=len(annotation_bytes),
        width=width,
        height=height,
        parsed_records=tuple(parsed),
        lineage=tuple(lineage),
    )


def _allowed_source_directories(data_root: Path, split: str) -> tuple[Path, Path]:
    if split not in ALLOWED_SPLITS:
        raise ConversionContractError("only train and val source directories are addressable")
    dataset_dir = data_root / f"VisDrone2019-DET-{split}"
    return dataset_dir / "images", dataset_dir / "annotations"


def collect_split(data_root: Path, split: str) -> tuple[RawImageRecord, ...]:
    """Read only the explicitly named official train or val directories."""

    if not data_root.is_absolute():
        raise ConversionContractError("data-root must be absolute")
    image_dir, annotation_dir = _allowed_source_directories(data_root, split)
    if not image_dir.is_dir() or not annotation_dir.is_dir():
        raise ConversionContractError(f"missing official {split} images or annotations directory")
    image_paths = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and not path.is_symlink() and path.suffix.casefold() in IMAGE_SUFFIXES
    )
    annotation_paths = sorted(
        path for path in annotation_dir.iterdir()
        if path.is_file() and not path.is_symlink() and path.suffix.casefold() == ".txt"
    )
    image_stems = {path.stem.casefold() for path in image_paths}
    annotation_stems = {path.stem.casefold() for path in annotation_paths}
    if image_stems != annotation_stems:
        raise ConversionContractError(f"{split} image/annotation stems do not match")
    records: list[RawImageRecord] = []
    for image_path in image_paths:
        annotation_path = annotation_dir / f"{image_path.stem}.txt"
        if not annotation_path.is_file() or annotation_path.is_symlink():
            raise ConversionContractError(f"missing annotation for {image_path.name}")
        image_bytes = image_path.read_bytes()
        annotation_bytes = annotation_path.read_bytes()
        relative_image = f"{split}/images/{image_path.name}"
        relative_annotation = f"{split}/annotations/{annotation_path.name}"
        width, height = _image_size(image_bytes)
        records.append(record_from_bytes(
            split,
            relative_image,
            image_bytes,
            annotation_bytes,
            annotation_relative_path=relative_annotation,
            width=width,
            height=height,
        ))
    return tuple(sorted(records, key=lambda item: item.relative_path))


def assign_coco_image_ids(
    train_records: Iterable[RawImageRecord],
    val_records: Iterable[RawImageRecord],
) -> dict[tuple[str, str], int]:
    """Assign one global, contiguous integer ID space in train-then-val order."""

    records = tuple(train_records) + tuple(val_records)
    keys = [record.image_key for record in records]
    if len(set(keys)) != len(keys):
        raise ConversionContractError("duplicate source image identity")
    seen_hashes: dict[str, tuple[str, str]] = {}
    for record in records:
        previous = seen_hashes.setdefault(record.image_sha256, record.image_key)
        if previous != record.image_key and previous[0] != record.split:
            raise ConversionContractError("duplicate image content across source sides")
    ordered = sorted(records, key=lambda record: (SPLIT_ORDER[record.split], record.relative_path))
    return {record.image_key: index for index, record in enumerate(ordered, 1)}


def _image_identity(record: RawImageRecord, coco_image_id: int) -> dict[str, object]:
    return {
        "id": coco_image_id,
        "file_name": record.relative_path,
        "width": record.width,
        "height": record.height,
        "stable_image_id": record.stable_image_id,
        "split": record.split,
    }


def _lineage_dict(item: AnnotationLineage) -> dict[str, object]:
    return {
        "split": item.split,
        "source_relative_path": item.source_relative_path,
        "physical_line_number": item.physical_line_number,
        "raw_fields": list(item.raw_fields),
        "raw_line_sha256": item.raw_line_sha256,
        "image_id": item.image_id,
        "annotation_id": item.annotation_id,
        "bbox_xyxy": list(item.bbox_xyxy),
        "area": item.area,
        "raw_category_id": item.raw_category_id,
        "training_category_id": item.training_category_id,
        "coco_category_id": item.coco_category_id,
        "status": item.status.value,
        "reason_code": item.reason_code,
        "ignore_region": item.ignore_region,
    }


def _coco_annotation(item: AnnotationLineage, coco_image_id: int, annotation_id: int) -> dict[str, object]:
    if item.status is not AnnotationStatus.KEEP or item.coco_category_id is None or item.area <= 0:
        raise ConversionContractError("only kept formal records may enter COCO annotations")
    left, top, right, bottom = item.bbox_xyxy
    return {
        "id": annotation_id,
        "image_id": coco_image_id,
        "category_id": item.coco_category_id,
        "bbox": [left, top, right - left, bottom - top],
        "area": item.area,
        "iscrowd": 0,
        "stable_annotation_id": item.annotation_id,
    }


def _category_contract() -> dict[str, object]:
    mappings = []
    for raw_id in (0, *range(1, 12)):
        mapping = category_mapping(raw_id)
        mappings.append({
            "raw_category_id": mapping.raw_category_id,
            "name": mapping.name,
            "training_category_id": mapping.training_category_id,
            "coco_category_id": mapping.coco_category_id,
            "enters_matching": mapping.enters_matching,
            "reason_code": mapping.reason_code,
        })
    return {
        "schema_version": 1,
        "num_classes": 10,
        "background_explicit": False,
        "categories": [
            {"id": raw_id, "name": CATEGORY_NAMES[raw_id]}
            for raw_id in range(1, 11)
        ],
        "raw_mappings": mappings,
    }


def _group_dict(group: AtomicGroup) -> dict[str, object]:
    return {
        "group_id": group.group_id,
        "sequence_keys": list(group.sequence_keys),
        "image_paths": list(group.image_paths),
        "image_sha256s": list(group.image_sha256s),
        "image_count": group.image_count,
        "valid_target_count": group.valid_target_count,
        "class_counts": list(group.class_counts),
        "small_target_count": group.small_target_count,
        "identity_closed": group.identity_closed,
        "abnormal": group.abnormal,
    }


def _split_plan_dict(plan: SplitPlan) -> dict[str, object]:
    return {
        "schema_version": 1,
        "train_core_group_ids": list(plan.train_core_group_ids),
        "confirmatory_group_ids": list(plan.confirmatory_group_ids),
        "ordered_candidate_group_ids": list(plan.ordered_candidate_group_ids),
        "target_image_count": plan.target_image_count,
        "selected_image_count": plan.selected_image_count,
        "selected_group_list_sha256": plan.selected_group_list_sha256,
        "seed": plan.seed,
        "salt": plan.salt,
        "selection_allowed": plan.selection_allowed,
        "metrics_access_allowed": plan.metrics_access_allowed,
        "single_final_access_only": plan.single_final_access_only,
        "checks": [{"name": name, "passed": passed} for name, passed in plan.checks],
    }


def _manifest_records(records: Iterable[RawImageRecord], ids: Mapping[tuple[str, str], int]) -> list[dict[str, object]]:
    return [record.manifest_dict(ids[record.image_key]) for record in sorted(records, key=lambda item: item.relative_path)]


def _lineages(records: Iterable[RawImageRecord]) -> list[dict[str, object]]:
    return [
        _lineage_dict(item)
        for record in sorted(records, key=lambda value: value.relative_path)
        for item in record.lineage
    ]


def _ignore_regions(records: Iterable[RawImageRecord]) -> list[dict[str, object]]:
    return [item for item in _lineages(records) if item["ignore_region"] is True]


def _coco_json(records: Iterable[RawImageRecord], ids: Mapping[tuple[str, str], int]) -> dict[str, object]:
    records = tuple(sorted(records, key=lambda item: item.relative_path))
    images = [_image_identity(record, ids[record.image_key]) for record in records]
    candidates = [
        (record, item)
        for record in records
        for item in record.lineage
        if item.status is AnnotationStatus.KEEP
    ]
    candidates.sort(key=lambda value: (ids[value[0].image_key], value[1].physical_line_number, value[1].annotation_id))
    annotations = [
        _coco_annotation(item, ids[record.image_key], index)
        for index, (record, item) in enumerate(candidates, 1)
    ]
    return {
        "info": {"description": "VisDrone Protocol V1 converted COCO formal GT"},
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": raw_id, "name": CATEGORY_NAMES[raw_id]}
            for raw_id in range(1, 11)
        ],
    }


def _audit(records: Iterable[RawImageRecord], coco: Mapping[str, object] | None) -> dict[str, object]:
    lineage = _lineages(records)
    counts = {
        "raw_rows": len(lineage),
        "keep_rows": sum(item["status"] == "keep" for item in lineage),
        "ignore_rows": sum(item["status"] == "ignore" for item in lineage),
        "filtered_rows": sum(item["status"] == "filtered" for item in lineage),
    }
    counts["row_conservation_pass"] = counts["raw_rows"] == counts["keep_rows"] + counts["ignore_rows"] + counts["filtered_rows"]
    counts["one_terminal_state_per_row"] = len({(item["source_relative_path"], item["physical_line_number"]) for item in lineage}) == len(lineage)
    coco_count = len(coco.get("annotations", [])) if coco is not None else 0
    counts["coco_annotation_count"] = coco_count
    counts["keep_equals_coco"] = counts["keep_rows"] == coco_count
    return counts


def _round_trip_audit(
    train_core: tuple[RawImageRecord, ...],
    development: tuple[RawImageRecord, ...],
    confirmatory: tuple[RawImageRecord, ...],
    ids: Mapping[tuple[str, str], int],
) -> dict[str, object]:
    all_records = train_core + development + confirmatory
    formal = [item for record in all_records for item in record.lineage if item.status is AnnotationStatus.KEEP]
    coco_formal = [item for record in train_core + development for item in record.lineage if item.status is AnnotationStatus.KEEP]
    coco_ids = set()
    for record in train_core + development:
        coco_ids.add(ids[record.image_key])
    stable_ids = {item.annotation_id for item in formal}
    return {
        "schema_version": 1,
        "formal_lineage_count_including_confirmatory": len(formal),
        "coco_formal_lineage_count": len(coco_formal),
        "unique_stable_annotation_ids": len(stable_ids) == len(formal),
        "unique_coco_stable_annotation_ids": len({item.annotation_id for item in coco_formal}) == len(coco_formal),
        "unique_coco_image_ids": len(coco_ids) == len(train_core) + len(development),
        "confirmatory_has_no_coco_annotations": True,
        "test_access_count": 0,
        "development_is_official_val": all(record.split == "val" for record in development),
        "confirmatory_metrics_accessed": False,
        "byte_determinism_contract": "rerun_same_inputs_and_compare_all_artifact_bytes",
    }


def _chunk_jsonl(records: Iterable[Mapping[str, object]], stem: str, max_bytes: int = MAX_ARTIFACT_BYTES) -> dict[str, bytes]:
    if max_bytes <= 0:
        raise ConversionContractError("chunk max must be positive")
    lines = [_canonical_json(record) + b"\n" for record in records]
    if any(len(line) > max_bytes for line in lines):
        raise ConversionContractError(f"single {stem} manifest record exceeds chunk limit")
    chunks: dict[str, bytes] = {}
    current: list[bytes] = []
    current_size = 0
    for line in lines:
        if current and current_size + len(line) > max_bytes:
            index = len(chunks) + 1
            chunks[f"{stem}.part-{index:05d}.jsonl"] = b"".join(current)
            current = []
            current_size = 0
        current.append(line)
        current_size += len(line)
    if current or not chunks:
        if len(chunks) == 0 and current_size <= max_bytes:
            chunks[f"{stem}.jsonl"] = b"".join(current)
        else:
            index = len(chunks) + 1
            chunks[f"{stem}.part-{index:05d}.jsonl"] = b"".join(current)
    return chunks


def _config_for_output() -> dict[str, object]:
    config = protocol_schema()
    config["converter_schema_version"] = CONVERTER_SCHEMA_VERSION
    config["real_conversion_outputs_generated"] = True
    return config


def _invocation_for_output() -> dict[str, object]:
    return {
        "schema_version": 1,
        "mode": "run",
        "authorized": True,
        "overwrite": False,
        "resume": False,
        "data_root_recorded": False,
        "absolute_data_root_in_artifacts": False,
        "test_accessed": False,
    }


def build_conversion_bundle(
    train_records: Iterable[RawImageRecord],
    val_records: Iterable[RawImageRecord],
    *,
    planner_train_image_count: int = 6471,
    planner_target_ratio: float = 0.10,
    planner_tolerance: float = 0.05,
) -> ConversionBundle:
    """Build all deterministic production artifacts in memory."""

    train = tuple(sorted(train_records, key=lambda item: item.relative_path))
    val = tuple(sorted(val_records, key=lambda item: item.relative_path))
    if not train or not val:
        raise ConversionContractError("both train and val records are required")
    if any(record.split != "train" for record in train) or any(record.split != "val" for record in val):
        raise ConversionContractError("records are assigned to the wrong split")
    ids = assign_coco_image_ids(train, val)
    train_images = [
        ImageIdentity(
            relative_path=record.relative_path,
            sequence_key=record.sequence_key,
            image_sha256=record.image_sha256,
            valid_target_count=len(record.valid_lineage),
            class_counts=tuple(sum(item.training_category_id == index for item in record.valid_lineage) for index in range(10)),
            small_target_count=sum(item.area < 32 * 32 for item in record.valid_lineage),
        )
        for record in train
    ]
    groups = build_atomic_groups(train_images)
    development_sequences = frozenset(record.sequence_key for record in val)
    plan = plan_confirmatory_split(
        groups,
        train_image_count=planner_train_image_count,
        target_ratio=planner_target_ratio,
        development_sequence_keys=development_sequences,
        tolerance=planner_tolerance,
    )
    selected_ids = set(plan.confirmatory_group_ids)
    group_by_path = {
        path: group.group_id
        for group in groups
        for path in group.image_paths
    }
    confirmatory = tuple(record for record in train if group_by_path[record.relative_path] in selected_ids)
    train_core = tuple(record for record in train if record.relative_path not in {item.relative_path for item in confirmatory})
    development = val
    train_core_coco = _coco_json(train_core, ids)
    development_coco = _coco_json(development, ids)
    files: dict[str, bytes] = {
        "config.json": _canonical_json(_config_for_output()),
        "invocation.json": _canonical_json(_invocation_for_output()),
        "source_identity.json": _canonical_json({
            "schema_version": 1,
            "protocol_id": "P3-VISDRONE-DATA-PROTOCOL-V1",
            "audited_protocol_data_identity": protocol_schema()["data_identity"],
            "train_image_count": len(train),
            "val_image_count": len(val),
            "train_raw_manifest_sha256": _sha256_bytes(_jsonl_bytes(_manifest_records(train, ids))),
            "val_raw_manifest_sha256": _sha256_bytes(_jsonl_bytes(_manifest_records(val, ids))),
            "train_annotation_count": sum(len(record.lineage) for record in train),
            "val_annotation_count": sum(len(record.lineage) for record in val),
            "test_accessed": False,
        }),
        "sequence_groups.json": _canonical_json({"schema_version": 1, "groups": [_group_dict(group) for group in groups]}),
        "split_plan.json": _canonical_json(_split_plan_dict(plan)),
        "train_core_manifest.json": _canonical_json({"records": _manifest_records(train_core, ids)}),
        "development_manifest.json": _canonical_json({"records": _manifest_records(development, ids)}),
        "confirmatory_sealed_manifest.json": _canonical_json({
            "records": _manifest_records(confirmatory, ids),
            "selection_allowed": False,
            "metrics_access_allowed": False,
            "single_final_access_only": True,
        }),
        "category_contract.json": _canonical_json(_category_contract()),
        "train_core_coco.json": _canonical_json(train_core_coco),
        "development_coco.json": _canonical_json(development_coco),
        "train_core_lineage.jsonl": _jsonl_bytes(_lineages(train_core)),
        "development_lineage.jsonl": _jsonl_bytes(_lineages(development)),
        "confirmatory_lineage.jsonl": _jsonl_bytes(_lineages(confirmatory)),
        "train_core_ignore_regions.jsonl": _jsonl_bytes(_ignore_regions(train_core)),
        "development_ignore_regions.jsonl": _jsonl_bytes(_ignore_regions(development)),
        "conversion_audit.json": _canonical_json({
            "schema_version": 1,
            "train_core": _audit(train_core, train_core_coco),
            "development": _audit(development, development_coco),
            "confirmatory": _audit(confirmatory, None),
            "test_access_count": 0,
        }),
        "round_trip_audit.json": _canonical_json(_round_trip_audit(train_core, development, confirmatory, ids)),
    }
    files.update(_chunk_jsonl(_manifest_records(train, ids), "raw_train_manifest"))
    files.update(_chunk_jsonl(_manifest_records(val, ids), "raw_val_manifest"))
    return ConversionBundle(
        train_records=train,
        val_records=val,
        train_core_records=train_core,
        confirmatory_records=confirmatory,
        development_records=development,
        coco_image_ids=ids,
        atomic_groups=groups,
        split_plan=plan,
        files=files,
    )


def validate_protocol_config(config: Mapping[str, object]) -> None:
    """Validate the immutable Protocol V1 identity used by a future run."""

    if config.get("schema_version") != 1 or config.get("protocol_id") != "P3-VISDRONE-DATA-PROTOCOL-V1":
        raise ConversionContractError("protocol config identity mismatch")
    if config.get("test_access_allowed") is not False:
        raise ConversionContractError("protocol config does not disable test access")
    if config.get("real_conversion_outputs_generated") not in {False, None}:
        raise ConversionContractError("input protocol config must be the pre-run schema")
    split = config.get("split")
    if not isinstance(split, Mapping):
        raise ConversionContractError("protocol split schema is missing")
    if split.get("test") != "disabled" or split.get("seed") != 20260808 or split.get("salt") != "P3-confirmatory-v1":
        raise ConversionContractError("protocol split identity mismatch")
    if split.get("selection_allowed") is not False or split.get("metrics_access_allowed") is not False:
        raise ConversionContractError("protocol selection/metrics gates are not closed")


def load_protocol_config(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ConversionContractError("protocol config must be a regular file")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConversionContractError("protocol config cannot be parsed") from exc
    if not isinstance(config, dict):
        raise ConversionContractError("protocol config must be a JSON object")
    validate_protocol_config(config)
    return config


def validate_run_arguments(
    data_root: Path,
    output_dir: Path,
    protocol_config: Path,
    splits: tuple[str, ...],
) -> Path:
    if set(splits) != set(ALLOWED_SPLITS) or len(splits) != 2:
        raise ConversionContractError("run requires exactly train,val splits")
    if not data_root.is_absolute():
        raise ConversionContractError("data-root must be absolute")
    if any(part == ".." for part in output_dir.parts):
        raise ConversionContractError("output-dir may not escape through ..")
    resolved_output = output_dir.resolve(strict=False)
    if resolved_output.exists() or resolved_output.is_symlink():
        raise ConversionContractError("output-dir already exists; overwrite/resume are forbidden")
    if protocol_config.is_symlink() or not protocol_config.is_file():
        raise ConversionContractError("protocol-config must be a regular file")
    if os.environ.get(AUTHORIZATION_ENV) != EXPECTED_AUTHORIZATION:
        raise ConversionContractError(f"{AUTHORIZATION_ENV}=1 is required")
    return resolved_output


def _atomic_write(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise ConversionContractError(f"artifact already exists: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _inventory_records(output_dir: Path, excluded: frozenset[str]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(output_dir.iterdir(), key=lambda item: item.name):
        if path.name in excluded or path.name.startswith("."):
            continue
        if path.is_symlink() or not path.is_file():
            raise ConversionContractError(f"invalid artifact entry: {path.name}")
        payload = path.read_bytes()
        records.append({"relative_path": path.name, "size_bytes": len(payload), "sha256": _sha256_bytes(payload)})
    return records


def _inventory_payload(output_dir: Path, excluded: frozenset[str]) -> dict[str, object]:
    records = _inventory_records(output_dir, excluded)
    return {
        "schema_version": 1,
        "excluded_from_inventory": sorted(excluded),
        "artifacts": records,
        "canonical_inventory_sha256": _sha256_bytes(_canonical_json(records)),
    }


def _redact_error(message: str, redactions: Iterable[str]) -> str:
    safe = message
    for value in redactions:
        if value:
            safe = safe.replace(value, "<redacted-data-root>")
    return safe


def write_failure_evidence(
    output_dir: Path,
    error: BaseException,
    *,
    redactions: Iterable[str] = (),
) -> None:
    """Retain an original failure and a partial inventory without fake success."""

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_message = _redact_error(str(error), redactions)
    if not (output_dir / "error.json").exists():
        _atomic_write(output_dir / "error.json", _canonical_json({
            "schema_version": 1,
            "status": "FAILED",
            "exception_type": type(error).__name__,
            "message": safe_message,
            "original_exception_preserved": True,
        }))
    partial = _inventory_payload(output_dir, frozenset({"partial_inventory.json", "completion.json"}))
    if not (output_dir / "partial_inventory.json").exists():
        _atomic_write(output_dir / "partial_inventory.json", _canonical_json(partial))
    if not (output_dir / "completion.json").exists():
        _atomic_write(output_dir / "completion.json", _canonical_json({
            "schema_version": 1,
            "status": "FAILED",
            "error_artifact": "error.json",
            "partial_inventory_artifact": "partial_inventory.json",
            "selection_allowed": False,
            "metrics_access_allowed": False,
            "single_final_access_only": True,
            "confirmatory_metrics_accessed": False,
            "test_accessed": False,
            "completion_self_hash_included": False,
        }))


def _write_success_files(output_dir: Path, bundle: ConversionBundle) -> dict[str, object]:
    for name, payload in sorted(bundle.files.items()):
        if "/" in name or "\\" in name or name in {"artifact_inventory.json", "completion.json"}:
            raise ConversionContractError(f"invalid artifact filename: {name}")
        _atomic_write(output_dir / name, payload)
    excluded = frozenset({"artifact_inventory.json", "completion.json"})
    inventory = _inventory_payload(output_dir, excluded)
    inventory_bytes = _canonical_json(inventory)
    _atomic_write(output_dir / "artifact_inventory.json", inventory_bytes)
    run_nonce = _sha256_bytes(_canonical_json({
        "config_sha256": _sha256_bytes(bundle.files["config.json"]),
        "source_identity_sha256": _sha256_bytes(bundle.files["source_identity.json"]),
        "split_plan_sha256": _sha256_bytes(bundle.files["split_plan.json"]),
    }))
    completion = {
        "schema_version": 1,
        "status": "COMPLETED",
        "run_nonce": run_nonce,
        "config_sha256": _sha256_bytes(bundle.files["config.json"]),
        "artifact_inventory_sha256": _sha256_bytes(inventory_bytes),
        "output_path_recorded": False,
        "selection_allowed": False,
        "metrics_access_allowed": False,
        "single_final_access_only": True,
        "confirmatory_metrics_accessed": False,
        "test_accessed": False,
        "completion_self_hash_included": False,
        "production_conversion_executed": True,
    }
    _atomic_write(output_dir / "completion.json", _canonical_json(completion))
    return completion


def write_conversion_artifacts(output_dir: Path, bundle: ConversionBundle) -> dict[str, object]:
    """Atomically write a successful bundle; preserve evidence on failure."""

    output_dir = output_dir.resolve(strict=False)
    if output_dir.exists() or output_dir.is_symlink():
        raise ConversionContractError("output-dir already exists; no overwrite or resume")
    output_dir.mkdir(parents=True)
    try:
        return _write_success_files(output_dir, bundle)
    except Exception as error:
        write_failure_evidence(output_dir, error)
        raise


def run_conversion(
    data_root: Path,
    output_dir: Path,
    protocol_config: Path,
    splits: tuple[str, ...] = ALLOWED_SPLITS,
) -> dict[str, object]:
    """Authorized future production entry point; never called by contract-check."""

    resolved_output = validate_run_arguments(data_root, output_dir, protocol_config, splits)
    resolved_output.mkdir(parents=True)
    try:
        load_protocol_config(protocol_config)
        train = collect_split(data_root, "train")
        val = collect_split(data_root, "val")
        bundle = build_conversion_bundle(train, val)
        return _write_success_files(resolved_output, bundle)
    except Exception as error:
        write_failure_evidence(resolved_output, error, redactions=(str(data_root), str(protocol_config)))
        raise
