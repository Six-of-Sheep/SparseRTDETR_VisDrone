"""Fixed-checkpoint development evaluation for the v2b engineering baseline.

Only the explicitly bound development COCO, manifest and listed JPEGs are read.
The legacy primary algorithm consumes converted formal GT, exactly as v2a did;
its spatial-ignore sidecar is not loaded or silently repaired. COCO diagnostics
are separate and retain the unmodified vendor evaluator's category/area/maxDets
axes. No metric selects a checkpoint or certifies a historical protocol run.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib
import importlib.metadata
import inspect
import io
import json
import math
import os
from pathlib import Path
import random
import stat
import sys
import time
from typing import Any, Mapping

import numpy as np
from PIL import Image
import torch

from .categories import map_target_labels_to_model
from .config import (
    _restore_registry, _snapshot_registry, _vendor_path, _vendor_root,
    load_isolated_vendor_config_dict,
)
from .postprocessor import VisDronePostProcessor
from .training_v2b import _vendor_source_identities
from .training_v2b_device import (
    capture_cuda_rng, restore_cuda_rng, validate_component_placement,
    validate_prepared_runtime,
)
from .training_v2b_engine import validate_engine_state_dict
from .training_v2b_geometry import ModelGeometryError, snapshot_model_geometry
from .training_v2b_evidence import (
    file_reference, initial_parameter_reference, write_exclusive_json,
)


class DevelopmentEvaluationError(ValueError):
    """Unbound development input, changed semantics, or unsafe evaluation."""


EVALUATION_EPOCHS = (10, 20, 30)
INPUT_SIZE = 640
SUPPORTED_INPUT_SIZES = (640, 896, 1024)
BATCH_SIZE = 4
PREVIEW_IMAGES = 4
_BINDING_KEYS = {
    "schema_version", "role", "protocol_id", "repo_root", "annotation",
    "manifest", "image_root", "image_count", "annotation_count",
    "image_manifest_sha256", "image_order_sha256", "policy", "source",
    "binding_sha256",
}
_ARTIFACT_NAMES = (
    "predictions.json", "coco_tensors.npz", "primary_legacy_result.json",
    "result.json",
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise DevelopmentEvaluationError(f"{name} must be an integer >= {minimum}")
    return value


def _sha(value: Any, name: str) -> str:
    if (type(value) is not str or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise DevelopmentEvaluationError(f"{name} must be lowercase SHA-256")
    return value


def _role_path(value: str | Path, name: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise DevelopmentEvaluationError(f"{name} must be an explicit path")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise DevelopmentEvaluationError(f"{name} must be absolute and canonical")
    forbidden = {"confirmatory", "test", "train", "train_core",
                 "visdrone2019-det-train", "visdrone2019-det-test-dev",
                 "visdrone2019-det-test-challenge"}
    if any(part.casefold() in forbidden
           or part.casefold().startswith("confirmatory_")
           for part in path.parts):
        raise DevelopmentEvaluationError(f"{name} crosses another data role")
    return path


def _read_json(path: str | Path, expected_sha: str, name: str):
    expected_name = {"annotation": "development_coco.json",
                     "manifest": "development_manifest.json"}[name]
    candidate = _role_path(path, name)
    if candidate.name != expected_name:
        raise DevelopmentEvaluationError(f"{name} must name {expected_name}")
    _sha(expected_sha, name + " SHA-256")
    try:
        actual = _role_path(candidate.resolve(strict=True), name)
        if actual.name != expected_name or not stat.S_ISREG(actual.stat().st_mode):
            raise DevelopmentEvaluationError(f"{name} is not the named regular authority")
        raw = actual.read_bytes()
    except OSError as exc:
        raise DevelopmentEvaluationError(f"cannot read bound {name}") from exc
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise DevelopmentEvaluationError(f"{name} SHA-256 mismatch")
    try:
        document = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise DevelopmentEvaluationError(f"{name} is not valid JSON") from exc
    if type(document) is not dict:
        raise DevelopmentEvaluationError(f"{name} must be an object")
    return document, {"path": str(actual), "sha256": expected_sha,
                      "size_bytes": len(raw)}


def _filename(value: Any) -> str:
    if type(value) is not str or "\\" in value:
        raise DevelopmentEvaluationError("image must use a val/images JPEG name")
    parts = value.split("/")
    if (len(parts) != 3 or parts[:2] != ["val", "images"]
            or parts[2] in ("", ".", "..") or Path(parts[2]).suffix.casefold() != ".jpg"):
        raise DevelopmentEvaluationError("image must use one val/images JPEG name")
    return parts[2]


def _validate_documents(coco: dict, manifest: dict):
    images, annotations, categories = (coco.get(k) for k in
                                       ("images", "annotations", "categories"))
    records = manifest.get("records")
    if not all(type(v) is list for v in (images, annotations, categories, records)):
        raise DevelopmentEvaluationError("COCO arrays and manifest records are required")
    if (len(categories) != 10 or any(type(c) is not dict or type(c.get("id")) is not int
                                    for c in categories)
            or sorted(c["id"] for c in categories) != list(range(1, 11))):
        raise DevelopmentEvaluationError("development categories must be exactly 1..10")
    by_id, stable_ids, filenames = {}, set(), set()
    for image in images:
        if type(image) is not dict:
            raise DevelopmentEvaluationError("COCO image must be an object")
        image_id = _integer(image.get("id"), "image ID", 1)
        stable_id = _sha(image.get("stable_image_id"), "stable image ID")
        filename = _filename(image.get("file_name"))
        for field in ("width", "height"):
            _integer(image.get(field), field, 1)
        if image.get("split") != "val":
            raise DevelopmentEvaluationError("development image is not an official-val member")
        if image_id in by_id or stable_id in stable_ids or filename in filenames:
            raise DevelopmentEvaluationError("duplicate development image identity")
        by_id[image_id] = image
        stable_ids.add(stable_id)
        filenames.add(filename)
    if not by_id:
        raise DevelopmentEvaluationError("development cannot be empty")
    members = {}
    for record in records:
        if type(record) is not dict:
            raise DevelopmentEvaluationError("manifest record must be an object")
        image_id = _integer(record.get("coco_image_id"), "manifest image ID", 1)
        if image_id not in by_id or image_id in members:
            raise DevelopmentEvaluationError("development manifest membership mismatch")
        image = by_id[image_id]
        _filename(record.get("relative_path"))
        if (record.get("relative_path") != image["file_name"]
                or record.get("stable_image_id") != image["stable_image_id"]
                or record.get("split") != "val"
                or any(record.get(k) != image[k] for k in ("width", "height"))):
            raise DevelopmentEvaluationError("development manifest identity mismatch")
        _sha(record.get("image_sha256"), "image SHA-256")
        _integer(record.get("image_size_bytes"), "image size", 1)
        members[image_id] = record
    if set(members) != set(by_id):
        raise DevelopmentEvaluationError("development manifest is incomplete")
    grouped, annotation_ids = {key: [] for key in by_id}, set()
    for annotation in annotations:
        if type(annotation) is not dict:
            raise DevelopmentEvaluationError("COCO annotation must be an object")
        aid = _integer(annotation.get("id"), "annotation ID", 1)
        image_id = _integer(annotation.get("image_id"), "annotation image ID", 1)
        category = _integer(annotation.get("category_id"), "category ID", 1)
        bbox, area = annotation.get("bbox"), annotation.get("area")
        if aid in annotation_ids or image_id not in grouped or category > 10:
            raise DevelopmentEvaluationError("annotation identity or category mismatch")
        if (type(bbox) is not list or len(bbox) != 4
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in bbox)
                or bbox[2] <= 0 or bbox[3] <= 0
                or type(area) not in (int, float) or not math.isfinite(area) or area <= 0
                or not math.isclose(area, bbox[2] * bbox[3], rel_tol=1e-12, abs_tol=1e-9)
                or type(annotation.get("iscrowd", 0)) is not int
                or annotation.get("iscrowd", 0) not in (0, 1)):
            raise DevelopmentEvaluationError("invalid development box, area or crowd flag")
        if "stable_annotation_id" in annotation:
            _sha(annotation["stable_annotation_id"], "stable annotation ID")
        annotation_ids.add(aid)
        grouped[image_id].append(annotation)
    ordered = [by_id[key] for key in sorted(by_id)]
    inventory = [
        {"image_id": image["id"], "stable_image_id": image["stable_image_id"],
         "relative_path": image["file_name"], "width": image["width"],
         "height": image["height"],
         "sha256": members[image["id"]]["image_sha256"],
         "size_bytes": members[image["id"]]["image_size_bytes"]}
        for image in ordered
    ]
    return ordered, grouped, members, inventory


def _input_size(value: Any) -> int:
    if type(value) is not int or value not in SUPPORTED_INPUT_SIZES:
        raise DevelopmentEvaluationError("development input_size must be 640, 896 or 1024")
    return value


def _policy_input_size(policy: Any) -> int:
    if type(policy) is not dict:
        raise DevelopmentEvaluationError("development policy must be an object")
    size = policy.get("input_size")
    if (type(size) is not list or len(size) != 2
            or any(type(value) is not int for value in size) or size[0] != size[1]):
        raise DevelopmentEvaluationError("development policy requires a square integer input_size")
    return _input_size(size[0])


def _policy(input_size: int = INPUT_SIZE) -> dict:
    size = _input_size(input_size)
    return {
        "input_size": [size, size], "batch_size": BATCH_SIZE,
        "num_workers": 0, "shuffle": False, "drop_last": False,
        "model_weights": "ema", "forward_dtype": "float32", "autocast": False,
        "evaluation_epochs": list(EVALUATION_EPOCHS),
        "preview_images": PREVIEW_IMAGES, "image_order": "ascending_coco_image_id",
        "transform": [
            {"type": "Resize", "size": [size, size]},
            {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
        ],
        "num_top_queries": 300, "focal_sigmoid": True, "nms": False,
        "confidence_threshold": None, "postprocessor_size_order": "width_height",
        "prediction_coordinates": "original_image_pixels_unclipped_xywh",
        "category_mapping": "model_0..9_to_coco_1..10",
        "primary_output_name": "primary_legacy_formal_gt",
        "primary_algorithm": "visdrone_official_style_v1",
        "primary_max_dets": [1, 10, 100, 500],
        "ground_truth_source": "converted_formal_coco",
        "ignore_regions_loaded": False, "full_official_ignore_compliance_claimed": False,
        "secondary_evaluator": "coco_secondary_vendor_v1",
        "secondary_max_dets": [1, 10, 100], "secondary_ignore": "COCO_iscrowd_only",
        "area_coordinate_space": "original_image_pixels",
        "checkpoint_selection": False, "protocol_changed": False,
    }


def _vendor_tools(root: Path, input_size: int = INPUT_SIZE):
    size = _input_size(input_size)
    _vendor_source_identities(root)
    with _vendor_path(_vendor_root(root)):
        importlib.import_module("src.data")
        from src.core import GLOBAL_CONFIG
        from src.data.dataset.coco_dataset import ConvertCocoPolysToMask
        from src.data.dataset.coco_eval import CocoEvaluator
        from src.data.transforms.container import Compose
        from src.data._misc import convert_to_tv_tensor
        snapshot = _snapshot_registry(GLOBAL_CONFIG)
        try:
            resolved = load_isolated_vendor_config_dict(root)
            recipe = copy.deepcopy(resolved["val_dataloader"]["dataset"]["transforms"])
            if recipe["ops"] != _policy()["transform"]:
                raise DevelopmentEvaluationError("vendored validation transform recipe changed")
            for operation in recipe["ops"]:
                if operation["type"] == "Resize":
                    operation["size"] = [size, size]
            transforms = Compose(**{k: copy.deepcopy(v) for k, v in recipe.items()
                                    if k != "type"})
        finally:
            _restore_registry(GLOBAL_CONFIG, snapshot)
    _vendor_source_identities(root)
    return ConvertCocoPolysToMask(False), transforms, convert_to_tv_tensor, CocoEvaluator


def _source_binding(root: Path) -> dict:
    import PIL
    import torchvision
    import faster_coco_eval
    from faster_coco_eval import COCO, COCOeval_faster
    from faster_coco_eval.utils.pytorch import FasterCocoEvaluator
    from . import primary_evaluator

    if not Path(__file__).resolve().is_relative_to(root):
        raise DevelopmentEvaluationError("development module belongs to another checkout")
    _vendor_tools(root)
    relative = (
        "src/sparse_rtdetr/baseline/training_v2b_development.py",
        "src/sparse_rtdetr/baseline/training_v2b_geometry.py",
        "src/sparse_rtdetr/baseline/categories.py",
        "src/sparse_rtdetr/baseline/config.py",
        "src/sparse_rtdetr/baseline/postprocessor.py",
        "src/sparse_rtdetr/baseline/primary_evaluator.py",
        "src/sparse_rtdetr/data_protocol/evaluation.py",
        "src/sparse_rtdetr/data_protocol/protocol.py",
        "configs/visdrone_protocol_v2.json",
        "vendor/rtdetrv2_pytorch/src/data/dataset/coco_dataset.py",
        "vendor/rtdetrv2_pytorch/src/data/dataset/coco_eval.py",
        "vendor/rtdetrv2_pytorch/src/data/transforms/container.py",
        "vendor/rtdetrv2_pytorch/src/data/transforms/_transforms.py",
        "vendor/rtdetrv2_pytorch/src/data/_misc.py",
        "vendor/rtdetrv2_pytorch/src/zoo/rtdetr/rtdetr_postprocessor.py",
        "vendor/rtdetrv2_pytorch/configs/rtdetrv2/include/dataloader.yml",
        primary_evaluator.PRIMARY_CONFIG_RELATIVE_PATH,
        primary_evaluator.PRIMARY_MANIFEST_RELATIVE_PATH,
    )
    sources = []
    for name in relative:
        path = (root / name).resolve(strict=True)
        if not path.is_relative_to(root):
            raise DevelopmentEvaluationError("evaluator source escapes the checkout")
        sources.append({"path": str(path.relative_to(root)),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    installed = {Path(faster_coco_eval.__file__).resolve()}
    for cls in (COCO, COCOeval_faster, FasterCocoEvaluator):
        for base in cls.__mro__:
            if base.__module__.startswith("faster_coco_eval"):
                installed.add(Path(inspect.getfile(base)).resolve())
    for name, module in tuple(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if name.startswith("faster_coco_eval.") and filename and filename.endswith(".so"):
            installed.add(Path(filename).resolve())
    return {
        "sources": sources,
        "faster_coco_eval_sources": [
            {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in sorted(installed)
        ],
        "versions": {"torch": str(torch.__version__), "torchvision": torchvision.__version__,
                     "numpy": np.__version__, "pillow": PIL.__version__,
                     "faster_coco_eval": importlib.metadata.version("faster-coco-eval")},
        # This source/config-only binder never opens the historical R3 closure.
        "primary_contract": primary_evaluator.primary_evaluator_contract_binding(root),
    }


def build_development_binding(*, repo_root: str | Path,
                              annotation_file: str | Path, annotation_sha256: str,
                              manifest_file: str | Path, manifest_sha256: str,
                              image_root: str | Path,
                              input_size: int = INPUT_SIZE) -> dict:
    """Bind explicit metadata, evaluation size, and expected JPEG identities."""
    size = _input_size(input_size)
    root = Path(repo_root).resolve(strict=True)
    images_root = _role_path(image_root, "image root")
    images_root = _role_path(images_root.resolve(strict=True), "resolved image root")
    if not images_root.is_dir():
        raise DevelopmentEvaluationError("development image root must be a directory")
    coco, annotation = _read_json(annotation_file, annotation_sha256, "annotation")
    manifest_doc, manifest = _read_json(manifest_file, manifest_sha256, "manifest")
    images, _, _, inventory = _validate_documents(coco, manifest_doc)
    result = {
        "schema_version": 1, "role": "development",
        "protocol_id": "P3-VISDRONE-DATA-PROTOCOL-V2", "repo_root": str(root),
        "annotation": annotation, "manifest": manifest, "image_root": str(images_root),
        "image_count": len(images), "annotation_count": len(coco["annotations"]),
        "image_manifest_sha256": _digest(inventory),
        "image_order_sha256": _digest([image["id"] for image in images]),
        "policy": _policy(size), "source": _source_binding(root),
    }
    return {**result, "binding_sha256": _digest(result)}


def validate_development_binding(binding: Mapping[str, Any], *,
                                 verify_files: bool = True) -> dict:
    """Recheck the frozen development/source binding, never the full R3 directory."""
    if type(binding) is not dict or set(binding) != _BINDING_KEYS:
        raise DevelopmentEvaluationError("development binding schema mismatch")
    checked = copy.deepcopy(binding)
    body = {k: v for k, v in checked.items() if k != "binding_sha256"}
    if _digest(body) != _sha(checked["binding_sha256"], "development binding"):
        raise DevelopmentEvaluationError("development binding digest mismatch")
    size = _policy_input_size(checked["policy"])
    if (checked["schema_version"] != 1 or checked["role"] != "development"
            or checked["protocol_id"] != "P3-VISDRONE-DATA-PROTOCOL-V2"
            or checked["policy"] != _policy(size)):
        raise DevelopmentEvaluationError("development role or evaluation policy changed")
    _integer(checked["image_count"], "image count", 1)
    _integer(checked["annotation_count"], "annotation count")
    for key in ("image_manifest_sha256", "image_order_sha256"):
        _sha(checked[key], key)
    for key in ("annotation", "manifest"):
        ref = checked[key]
        if type(ref) is not dict or set(ref) != {"path", "sha256", "size_bytes"}:
            raise DevelopmentEvaluationError("metadata reference schema mismatch")
        _role_path(ref["path"], key)
        _sha(ref["sha256"], key)
        _integer(ref["size_bytes"], key + " size", 1)
    _role_path(checked["image_root"], "image root")
    if verify_files:
        rebuilt = build_development_binding(
            repo_root=checked["repo_root"],
            annotation_file=checked["annotation"]["path"],
            annotation_sha256=checked["annotation"]["sha256"],
            manifest_file=checked["manifest"]["path"],
            manifest_sha256=checked["manifest"]["sha256"],
            image_root=checked["image_root"], input_size=size,
        )
        if rebuilt != checked:
            raise DevelopmentEvaluationError("development source/data binding drift")
    return checked


class _DevelopmentDataset:
    def __init__(self, binding: dict):
        self.binding = binding
        self.input_size = _policy_input_size(binding["policy"])
        coco, _ = _read_json(binding["annotation"]["path"],
                             binding["annotation"]["sha256"], "annotation")
        manifest, _ = _read_json(binding["manifest"]["path"],
                                binding["manifest"]["sha256"], "manifest")
        self.coco = coco
        self.images, self.annotations, self.members, _ = _validate_documents(coco, manifest)
        self.image_root = Path(binding["image_root"])
        self.converter, self.transforms, self.to_tv_tensor, self.evaluator_type = (
            _vendor_tools(Path(binding["repo_root"]), self.input_size)
        )
        self.epoch = -1

    def item(self, image: dict):
        filename = _filename(image["file_name"])
        try:
            path = _role_path((self.image_root / filename).resolve(strict=True), "image")
            if not path.is_relative_to(self.image_root) or not stat.S_ISREG(path.stat().st_mode):
                raise DevelopmentEvaluationError("development JPEG escapes the bound image root")
            raw = path.read_bytes()
        except OSError as exc:
            raise DevelopmentEvaluationError("cannot read the bound development image") from exc
        record = self.members[image["id"]]
        actual_sha = hashlib.sha256(raw).hexdigest()
        if len(raw) != record["image_size_bytes"] or actual_sha != record["image_sha256"]:
            raise DevelopmentEvaluationError("development image SHA-256/size mismatch")
        with Image.open(io.BytesIO(raw)) as source:
            pixels = source.convert("RGB")
        if pixels.size != (image["width"], image["height"]):
            raise DevelopmentEvaluationError("development decoded geometry mismatch")
        pixels, target = self.converter(pixels, {
            "image_id": image["id"], "annotations": self.annotations[image["id"]],
        })
        target["boxes"] = self.to_tv_tensor(target["boxes"], key="boxes",
                                            spatial_size=pixels.size[::-1])
        target = map_target_labels_to_model(target)
        pixels, target, _ = self.transforms(pixels, target, self)
        if (pixels.dtype != torch.float32 or list(pixels.shape) != [3, self.input_size, self.input_size]
                or not bool(torch.isfinite(pixels).all())
                or target["orig_size"].tolist() != [image["width"], image["height"]]):
            raise DevelopmentEvaluationError("development resize/width-height convention drift")
        receipt = {"image_id": image["id"], "stable_image_id": image["stable_image_id"],
                   "relative_path": image["file_name"], "width": image["width"],
                   "height": image["height"], "sha256": actual_sha,
                   "size_bytes": len(raw)}
        return pixels, target, receipt


def _axes(evaluator) -> dict:
    params = evaluator.params
    result = {
        "iouType": params.iouType, "useCats": int(params.useCats),
        "iouThrs": np.asarray(params.iouThrs).tolist(),
        "recThrs": np.asarray(params.recThrs).tolist(),
        "catIds": [int(v) for v in params.catIds],
        "areaRng": [[float(v) for v in pair] for pair in params.areaRng],
        "areaRngLbl": list(params.areaRngLbl), "maxDets": list(params.maxDets),
        "area_coordinate_space": "original_image_pixels",
    }
    if (result["iouType"] != "bbox" or result["useCats"] != 1
            or result["catIds"] != list(range(1, 11))
            or result["maxDets"] != [1, 10, 100]
            or result["areaRngLbl"] != ["all", "small", "medium", "large"]
            or result["areaRng"] != [[0, 1e10], [0, 32**2], [32**2, 96**2], [96**2, 1e10]]
            or not np.array_equal(params.iouThrs, np.linspace(0.5, 0.95, 10))
            or not np.array_equal(params.recThrs, np.linspace(0.0, 1.0, 101))):
        raise DevelopmentEvaluationError("vendored COCO bbox evaluation axes changed")
    return result


def _coco_metrics(evaluator) -> tuple[dict, dict, np.ndarray, np.ndarray]:
    axes = _axes(evaluator)
    precision = np.asarray(evaluator.eval["precision"], dtype=np.float64)
    recall = np.asarray(evaluator.eval["recall"], dtype=np.float64)
    if precision.shape != (10, 101, 10, 4, 3) or recall.shape != (10, 10, 4, 3):
        raise DevelopmentEvaluationError("COCO accumulator shape differs from frozen axes")
    for values in (precision, recall):
        if not np.isfinite(values).all() or not (
            (values == -1) | ((values >= 0) & (values <= 1))
        ).all():
            raise DevelopmentEvaluationError("COCO accumulator has invalid values")
    metrics = {}
    for name, kind, area, max_dets, ious in (
        ("AP", "precision", 0, 2, list(range(10))),
        ("AP50", "precision", 0, 2, [0]),
        ("AP75", "precision", 0, 2, [5]),
        ("AP_small", "precision", 1, 2, list(range(10))),
        ("AP_medium", "precision", 2, 2, list(range(10))),
        ("AP_large", "precision", 3, 2, list(range(10))),
        ("AR1", "recall", 0, 0, list(range(10))),
        ("AR10", "recall", 0, 1, list(range(10))),
        ("AR100", "recall", 0, 2, list(range(10))),
        ("AR_small", "recall", 1, 2, list(range(10))),
        ("AR_medium", "recall", 2, 2, list(range(10))),
        ("AR_large", "recall", 3, 2, list(range(10))),
    ):
        values = (precision[ious, :, :, area, max_dets] if kind == "precision"
                  else recall[ious, :, area, max_dets])
        valid = values[values > -1]
        metrics[name] = {
            "value": float(valid.mean() * 100) if valid.size else None,
            "units": "percent", "available": bool(valid.size),
            "valid_accumulator_entries": int(valid.size), "tensor": kind,
            "iou_indices": ious, "area_index": area, "max_dets_index": max_dets,
            "max_dets": axes["maxDets"][max_dets],
            "area": axes["areaRngLbl"][area],
        }
    return metrics, axes, precision, recall


def _component(components, name):
    try:
        return components[name] if isinstance(components, Mapping) else getattr(components, name)
    except (KeyError, AttributeError) as exc:
        raise DevelopmentEvaluationError("missing runtime component: " + name) from exc


def _geometry_snapshot(model, ema, input_size: int) -> dict:
    try:
        return {
            "raw": snapshot_model_geometry(model, expected_input_size=input_size),
            "ema": snapshot_model_geometry(ema.module, expected_input_size=input_size),
        }
    except ModelGeometryError as exc:
        raise DevelopmentEvaluationError("evaluation geometry mismatch: " + str(exc)) from exc


def _state_reference(module) -> str:
    return initial_parameter_reference({
        name: value.detach().cpu() for name, value in module.state_dict().items()
    })["sha256"]


def _auxiliary_clocks(engine) -> dict:
    steps = []
    for group in engine.optimizer.param_groups:
        for parameter in group["params"]:
            step = engine.optimizer.state.get(parameter, {}).get("step")
            if isinstance(step, torch.Tensor):
                step = float(step.detach().cpu().item())
            steps.append(step)
    return {
        "optimizer_learning_rates": [float(group["lr"]) for group in engine.optimizer.param_groups],
        "optimizer_steps_sha256": _digest(steps),
        "scheduler": json.loads(_canonical(engine.scheduler.state_dict()))
                     if engine.scheduler is not None else None,
        "warmup": json.loads(_canonical(engine.warmup.state_dict()))
                  if engine.warmup is not None else None,
    }


@contextlib.contextmanager
def _preserve_training_state(model, ema, postprocessor, engine, runtime, *,
                             input_size: int, initial_geometry: dict):
    modules = {id(module): module for root in (model, ema.module, postprocessor)
               for module in root.modules()}
    modes = [(module, module.training) for module in modules.values()]
    before = {"engine": copy.deepcopy(engine.state_dict()), "ema_updates": ema.updates,
              "model_sha256": _state_reference(model),
              "ema_sha256": _state_reference(ema.module),
              "auxiliary_clocks": _auxiliary_clocks(engine),
              "geometry": _geometry_snapshot(model, ema, input_size)}
    if before["geometry"] != initial_geometry:
        raise DevelopmentEvaluationError("evaluation geometry changed before snapshot")
    try:
        yield before
    finally:
        for module, mode in modes:
            module.training = mode
        after = {"engine": engine.state_dict(), "ema_updates": ema.updates,
                 "model_sha256": _state_reference(model),
                 "ema_sha256": _state_reference(ema.module),
                 "auxiliary_clocks": _auxiliary_clocks(engine),
                 "geometry": _geometry_snapshot(model, ema, input_size)}
        if after != before:
            raise DevelopmentEvaluationError("evaluation changed model/EMA state or update clocks/geometry")


def _primary_result(dataset, selected, predictions):
    from . import primary_evaluator
    from ..data_protocol.evaluation import (
        Detection, PrimaryEvaluatorInputV2, PrimaryGroundTruth, PrimaryImageV2,
    )

    by_id = {image["id"]: image for image in selected}
    images = tuple(PrimaryImageV2(image["stable_image_id"], image["width"], image["height"])
                   for image in selected)
    detections = tuple(
        Detection(by_id[row["image_id"]]["stable_image_id"], row["category_id"],
                  (row["bbox"][0], row["bbox"][1],
                   row["bbox"][0] + row["bbox"][2], row["bbox"][1] + row["bbox"][3]),
                  row["score"])
        for row in predictions
    )
    ground_truth = []
    for image in selected:
        for index, ann in enumerate(dataset.annotations[image["id"]]):
            x, y, w, h = (float(v) for v in ann["bbox"])
            ground_truth.append(PrimaryGroundTruth(
                annotation_id=ann.get("stable_annotation_id", f'{image["stable_image_id"]}:{index}'),
                image_id=image["stable_image_id"], category_id=ann["category_id"],
                bbox_xyxy=(x, y, x + w, y + h), area=float(ann["area"]),
                ignore_region=False, ignored=bool(ann.get("ignore", False) or ann.get("iscrowd", 0)),
            ))
    value = PrimaryEvaluatorInputV2(images, detections, tuple(ground_truth))
    contract = dataset.binding["source"]["primary_contract"]
    result = primary_evaluator.evaluate_primary_v1(value, contract)
    primary_evaluator.validate_primary_evaluator_result(result, value, contract)
    from dataclasses import asdict
    return json.loads(_canonical(asdict(result)))


def _publish_tensors(path: Path, precision: np.ndarray, recall: np.ndarray) -> dict:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, precision=precision, recall=recall)
    raw = buffer.getvalue()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                 | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    reference = file_reference(path)
    if reference["sha256"] != hashlib.sha256(raw).hexdigest():
        raise DevelopmentEvaluationError("COCO tensor artifact readback mismatch")
    return reference


def _evaluate(components, *, data_binding, output_dir, logged_epoch, hardware_gate,
              preview, capacity=False):
    if type(preview) is not bool or type(capacity) is not bool or (preview and capacity):
        raise DevelopmentEvaluationError("preview and capacity must be distinct evaluation modes")
    if (preview or capacity) and logged_epoch is not None:
        raise DevelopmentEvaluationError("preview/capacity cannot represent a completed training epoch")
    if not callable(hardware_gate):
        raise DevelopmentEvaluationError("an explicit per-batch live gate callback is required")
    binding = validate_development_binding(data_binding)
    input_size = _policy_input_size(binding["policy"])
    model, ema, postprocessor, engine, runtime = (
        _component(components, name)
        for name in ("model", "ema", "postprocessor", "engine", "runtime")
    )
    identity = validate_prepared_runtime(runtime)
    state = validate_engine_state_dict(engine.state_dict())
    if (engine.model is not model or engine.ema is not ema
            or state["config"]["expected_input_size"] != [input_size, input_size]
            or engine.device != runtime.device):
        raise DevelopmentEvaluationError("evaluation runtime/model/bound input geometry mismatch")
    if not isinstance(getattr(ema, "module", None), torch.nn.Module):
        raise DevelopmentEvaluationError("EMA weights are required; raw fallback is forbidden")
    if (not preview and not capacity and
            (type(logged_epoch) is not int or logged_epoch not in EVALUATION_EPOCHS
                         or state["epoch"] != logged_epoch or state["epoch_active"])):
        raise DevelopmentEvaluationError("full evaluation requires completed epoch 10, 20 or 30")
    validate_component_placement(model=model, ema=ema, postprocessor=postprocessor,
                                 optimizer=engine.optimizer, runtime=runtime)
    if any(parameter.dtype != torch.float32 for parameter in ema.module.parameters()):
        raise DevelopmentEvaluationError("EMA inference requires FP32 model parameters")
    geometry = _geometry_snapshot(model, ema, input_size)
    output = Path(output_dir).absolute()
    if not output.is_dir() or output.is_symlink():
        raise DevelopmentEvaluationError("evaluation output directory must already exist")
    for name in _ARTIFACT_NAMES:
        target = output / name
        if target.exists() or target.is_symlink():
            raise DevelopmentEvaluationError("evaluation artifacts already exist; overwrite is forbidden")
    dataset = _DevelopmentDataset(binding)
    if preview and len(dataset.images) < PREVIEW_IMAGES:
        raise DevelopmentEvaluationError("preview requires the fixed first four development images")
    selected = dataset.images[:PREVIEW_IMAGES] if preview else dataset.images
    from faster_coco_eval import COCO

    evaluator = dataset.evaluator_type(COCO(copy.deepcopy(dataset.coco)), ["bbox"])
    evaluator.world_size = 1
    _axes(evaluator.coco_eval["bbox"])
    wrapper = VisDronePostProcessor(vendor_postprocessor=postprocessor)
    gate_events, receipts, predictions = [], [], []

    def gate(phase, batch=None):
        before = time.monotonic_ns()
        if hardware_gate() is False:
            raise DevelopmentEvaluationError("live gate refused evaluation")
        gate_events.append({"phase": phase, "batch": batch,
                            "start_monotonic_ns": before,
                            "end_monotonic_ns": time.monotonic_ns()})

    started = time.monotonic_ns()
    gate("before_snapshot")
    with _preserve_training_state(
            model, ema, postprocessor, engine, runtime,
            input_size=input_size, initial_geometry=geometry) as unchanged:
        ema.module.eval()
        wrapper.eval()
        with torch.inference_mode(), torch.autocast(device_type=runtime.device.type, enabled=False):
            for batch_index, start in enumerate(range(0, len(selected), BATCH_SIZE)):
                gate("before_batch", batch_index)
                image_rows = selected[start:start + BATCH_SIZE]
                items = [dataset.item(image) for image in image_rows]
                samples = torch.stack([item[0] for item in items]).to(device=runtime.device)
                sizes = torch.stack([item[1]["orig_size"] for item in items]).to(device=runtime.device)
                outputs = ema.module(samples)
                if (type(outputs) is not dict or "pred_logits" not in outputs or "pred_boxes" not in outputs
                        or list(outputs["pred_logits"].shape) != [len(items), 300, 10]
                        or list(outputs["pred_boxes"].shape) != [len(items), 300, 4]):
                    raise DevelopmentEvaluationError("model evaluation output geometry changed")
                for name in ("pred_logits", "pred_boxes"):
                    tensor = outputs[name]
                    if tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all()):
                        raise DevelopmentEvaluationError("model evaluation output is nonfinite or not FP32")
                if not bool((outputs["pred_boxes"][..., 2:] > 0).all()):
                    raise DevelopmentEvaluationError("model produced nonpositive predicted box extents")
                processed = wrapper(outputs, sizes)
                cpu_predictions = {}
                for image, row in zip(image_rows, processed):
                    boxes = row["boxes"]
                    if (len(row["scores"]) != 300 or not bool(torch.isfinite(boxes).all())
                            or not bool((boxes[:, 2:] > boxes[:, :2]).all())):
                        raise DevelopmentEvaluationError("postprocessed detection geometry changed")
                    cpu_predictions[image["id"]] = {
                        key: value.detach().cpu() for key, value in row.items()
                    }
                predictions.extend(evaluator.prepare_for_coco_detection(cpu_predictions))
                evaluator.update(cpu_predictions)
                receipts.extend(item[2] for item in items)
                gate("after_batch", batch_index)
        gate("before_metrics")
        evaluator.synchronize_between_processes()
        evaluator.accumulate()
        metrics, axes, precision, recall = _coco_metrics(evaluator.coco_eval["bbox"])
        primary = _primary_result(dataset, selected, predictions)
        gate("after_metrics")
    validate_development_binding(binding)
    if len(receipts) != len(selected) or len({row["image_id"] for row in receipts}) != len(selected):
        raise DevelopmentEvaluationError("development coverage is incomplete or duplicated")
    if not preview and _digest(receipts) != binding["image_manifest_sha256"]:
        raise DevelopmentEvaluationError("observed image identities differ from the bound manifest")
    gate("before_publish")
    if _geometry_snapshot(model, ema, input_size) != geometry:
        raise DevelopmentEvaluationError("evaluation geometry changed before publication")
    prediction_reference = write_exclusive_json(output / "predictions.json", predictions)
    tensor_reference = _publish_tensors(output / "coco_tensors.npz", precision, recall)
    primary_reference = write_exclusive_json(output / "primary_legacy_result.json", primary)
    result = {
        "schema_version": 1, "role": "development",
        "scope": ("development_capacity_only" if capacity else
                  "development_preview_only" if preview else "development_full_fixed_checkpoint"),
        "logged_epoch": logged_epoch,
        "full_development_evaluation": not preview and not capacity,
        "full_development_coverage": not preview, "capacity_only": capacity,
        "scientific_comparison_eligible": not preview and not capacity,
        "checkpoint_selection_performed": False, "protocol_certification_claimed": False,
        "data_binding_sha256": binding["binding_sha256"],
        "data": {"annotation": binding["annotation"], "manifest": binding["manifest"],
                 "image_root": binding["image_root"], "bound_image_count": binding["image_count"],
                 "evaluated_image_count": len(selected),
                 "evaluated_ground_truth_count": sum(len(dataset.annotations[row["id"]]) for row in selected),
                 "image_receipts": receipts, "observed_images_sha256": _digest(receipts)},
        "policy": binding["policy"], "runtime": identity,
        "weights": {"kind": "ema", "state_sha256": unchanged["ema_sha256"],
                    "ema_updates": unchanged["ema_updates"],
                    "eval_spatial_size": unchanged["geometry"]["ema"]["decoder"]["eval_spatial_size"]},
        "training_state_unchanged": unchanged,
        "rng_restored": ["python", "numpy", "torch_cpu"]
                        + (["torch_cuda_bound_device"] if runtime.device.type == "cuda" else []),
        "native_gate_callbacks": gate_events,
        "evaluation_elapsed_seconds": (time.monotonic_ns() - started) / 1e9,
        "primary_legacy_formal_gt": {
            "algorithm_id": primary["protocol"], "ground_truth_source": "converted_formal_coco",
            "ignore_regions_loaded": False, "full_official_ignore_compliance_claimed": False,
            "max_dets": [1, 10, 100, 500], "units": "percent",
            "metrics": {name: primary[name] for name in
                        ("AP", "AP50", "AP75", "AR1", "AR10", "AR100", "AR500")},
            "result_artifact": primary_reference,
        },
        "coco_secondary": {
            "evaluator_id": "coco_secondary_vendor_v1", "iou_type": "bbox",
            "axes": axes, "axes_sha256": _digest(axes),
            "precision_shape": list(precision.shape), "recall_shape": list(recall.shape),
            "metrics": metrics, "tensor_artifact": tensor_reference,
        },
        "prediction_artifact": prediction_reference,
    }
    result_reference = write_exclusive_json(output / "result.json", result)
    return {**result, "result_artifact": result_reference}


@contextlib.contextmanager
def _preserve_all_rng(runtime):
    # Include binding construction, live-gate callbacks and artifact publication,
    # not only forward(), in the stream that must leave future DN draws unchanged.
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    cpu_rng = torch.random.default_generator.get_state().clone()
    cuda_rng = capture_cuda_rng(runtime)
    try:
        yield
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.random.default_generator.set_state(cpu_rng)
        restore_cuda_rng(cuda_rng, runtime)


def evaluate_development(components, *, data_binding: Mapping[str, Any],
                         output_dir: str | Path, logged_epoch: int,
                         hardware_gate) -> dict:
    """Evaluate all bound development images at a completed fixed checkpoint."""
    with _preserve_all_rng(_component(components, "runtime")):
        return _evaluate(components, data_binding=data_binding, output_dir=output_dir,
                         logged_epoch=logged_epoch, hardware_gate=hardware_gate, preview=False)


def preview_development(components, *, data_binding: Mapping[str, Any],
                        output_dir: str | Path, hardware_gate) -> dict:
    """Smoke-only first-four-image evaluation; never a full-development result."""
    with _preserve_all_rng(_component(components, "runtime")):
        return _evaluate(components, data_binding=data_binding, output_dir=output_dir,
                         logged_epoch=None, hardware_gate=hardware_gate, preview=True)


def evaluate_development_capacity(components, *, data_binding: Mapping[str, Any],
                                  output_dir: str | Path, hardware_gate) -> dict:
    """Exercise all bound images and metrics without claiming a fixed epoch.

    The production campaign binds 548 images. Synthetic CPU fixtures may bind a
    smaller complete set; both paths require the full bound identity chain.
    """
    with _preserve_all_rng(_component(components, "runtime")):
        return _evaluate(components, data_binding=data_binding, output_dir=output_dir,
                         logged_epoch=None, hardware_gate=hardware_gate,
                         preview=False, capacity=True)
