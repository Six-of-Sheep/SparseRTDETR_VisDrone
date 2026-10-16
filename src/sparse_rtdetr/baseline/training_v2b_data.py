"""Verified train_core loading with stateless augmentation and committed cursors.

Only explicit train_core COCO/manifest paths are read. Each selected JPEG is
verified against its train_core manifest before the vendored conversion and
augmentation run. No directory discovery, evaluator, model update or CUDA call
occurs here. Worker prefetch never advances the checkpoint cursor.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib
import io
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .categories import map_target_labels_to_model
from .config import _vendor_path, _vendor_root, load_isolated_vendor_config_dict
from .training_v2b import _vendor_source_identities, logical_batch_indices, seed_worker
from .training_v2b_engine import validate_engine_state_dict


class TrainCoreDataError(ValueError):
    """Unbound data, invalid role, altered input, or inconsistent sampler state."""


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise TrainCoreDataError(f"{name} must be an integer >= {minimum}")
    return value


def _sha(value: Any, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise TrainCoreDataError(f"{name} must be a lowercase SHA-256")
    return value


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _role_path(path: str | Path, name: str) -> Path:
    if not isinstance(path, (str, Path)):
        raise TrainCoreDataError(f"{name} must be an explicit path")
    candidate = Path(path)
    if not candidate.is_absolute():
        raise TrainCoreDataError(f"{name} must be absolute")
    forbidden = {"confirmatory", "test", "development", "val",
                 "visdrone2019-det-test-dev", "visdrone2019-det-test-challenge",
                 "visdrone2019-det-val"}
    if any(part.casefold() in forbidden or part.casefold().startswith("confirmatory_")
           for part in candidate.parts):
        raise TrainCoreDataError(f"{name} crosses a forbidden data role")
    return candidate


def _read_bound_json(path: Path, expected_sha256: str, name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    _sha(expected_sha256, f"{name} SHA-256")
    try:
        actual_path = _role_path(path.resolve(strict=True), name)
        required_name = {"annotation": "train_core_coco.json", "manifest": "train_core_manifest.json"}[name]
        if actual_path.name != required_name:
            raise TrainCoreDataError("resolved metadata must remain the named train_core authority")
        raw = actual_path.read_bytes()
    except OSError as exc:
        raise TrainCoreDataError(f"cannot read explicitly bound {name}") from exc
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise TrainCoreDataError(f"{name} SHA-256 mismatch")
    try:
        document = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise TrainCoreDataError(f"{name} must be valid JSON") from exc
    if type(document) is not dict:
        raise TrainCoreDataError(f"{name} must be an object")
    return document, {"path": str(actual_path), "sha256": expected_sha256, "size_bytes": len(raw)}


def _logical_filename(value: Any) -> str:
    if type(value) is not str or "\\" in value:
        raise TrainCoreDataError("image name must be a canonical train/images path")
    parts = value.split("/")
    if (len(parts) != 3 or parts[:2] != ["train", "images"] or
            not parts[2] or parts[2] in (".", "..") or
            Path(parts[2]).suffix.casefold() != ".jpg"):
        raise TrainCoreDataError("image name must be one train/images JPEG")
    return parts[2]


@dataclass(frozen=True)
class TrainCoreDataConfig:
    seed: int = 0
    input_size: int = 640
    logical_batch_size: int = 16
    num_workers: int = 4
    prefetch_factor: int = 2
    augmentation_stop_internal_epoch: int = 117
    multiprocessing_context: str = "spawn"
    role: str = "train_core"

    def __post_init__(self) -> None:
        if self.role != "train_core":
            raise TrainCoreDataError("only the explicit train_core role is authorized")
        _integer(self.seed, "seed")
        if self.seed >= 2**32:
            raise TrainCoreDataError("seed exceeds the CPU seed range")
        _integer(self.input_size, "input_size", 128)
        if (self.input_size > 640 and self.input_size != 896) or self.input_size % 32:
            raise TrainCoreDataError("runtime wiring supports sizes 128..640 divisible by 32, plus 896")
        _integer(self.logical_batch_size, "logical_batch_size", 1)
        _integer(self.num_workers, "num_workers")
        _integer(self.prefetch_factor, "prefetch_factor", 1)
        _integer(self.augmentation_stop_internal_epoch, "augmentation_stop_internal_epoch")
        if self.multiprocessing_context != "spawn":
            raise TrainCoreDataError("spawn workers are required before future CUDA integration")

    def semantic_config(self) -> dict[str, Any]:
        value = asdict(self)
        for name in ("num_workers", "prefetch_factor", "multiprocessing_context"):
            value.pop(name)
        return {**value, "tail_policy": "drop_incomplete_logical_batch",
                "sample_rng": "sha256(seed,logged_epoch,order_position,stable_image_id)",
                "epoch_numbering": "logged_one_based_vendor_zero_based",
                "multiscale": False, "category_mapping": "1..10_to_0..9",
                "source_verification": "manifest_sha256_before_decode"}


def validate_worker_environment(config: TrainCoreDataConfig) -> dict[str, Any]:
    """Reject the known MKL/libgomp spawn conflict before workers are created.

    The process entry point must choose the threading layer before importing
    NumPy/PyTorch. This library never silently changes a running process policy.
    """
    blas = getattr(np.__config__, "CONFIG", {}).get("Build Dependencies", {}).get("blas", {})
    backend = str(blas.get("name", "unknown"))
    if (config.num_workers and sys.platform.startswith("linux") and "mkl" in backend.casefold()
            and os.environ.get("MKL_THREADING_LAYER", "").upper() != "GNU"):
        raise TrainCoreDataError(
            "spawn workers with MKL NumPy and PyTorch/libgomp require explicit "
            "MKL_THREADING_LAYER=GNU at process startup"
        )
    return {"num_workers": config.num_workers, "prefetch_factor": config.prefetch_factor,
            "multiprocessing_context": config.multiprocessing_context,
            "persistent_workers": False, "pin_memory": False,
            "numpy_blas_backend": backend,
            "MKL_THREADING_LAYER": os.environ.get("MKL_THREADING_LAYER"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS")}


@contextlib.contextmanager
def _sample_rng(seed: int):
    # Only the CPU default generator is changed, even when a CUDA model exists.
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state = torch.random.default_generator.get_state()
    try:
        random.seed(seed)
        np.random.seed(seed % 2**32)
        torch.random.default_generator.manual_seed(seed)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.default_generator.set_state(torch_state)


def _sample_seed(base_seed: int, epoch: int, position: int, stable_id: str) -> int:
    raw = f"v2b-augment:{base_seed}:{epoch}:{position}:{stable_id}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") & ((1 << 63) - 1)


def _transform_config(root: Path, config: TrainCoreDataConfig) -> dict[str, Any]:
    _vendor_source_identities(root)
    resolved = load_isolated_vendor_config_dict(root)
    result = copy.deepcopy(resolved["train_dataloader"]["dataset"]["transforms"])
    required = ["RandomPhotometricDistort", "RandomZoomOut", "RandomIoUCrop",
                "SanitizeBoundingBoxes", "RandomHorizontalFlip", "Resize",
                "SanitizeBoundingBoxes", "ConvertPILImage", "ConvertBoxes"]
    if [op["type"] for op in result["ops"]] != required:
        raise TrainCoreDataError("vendored training augmentation operation order changed")
    for op in result["ops"]:
        if op["type"] == "Resize":
            op["size"] = [config.input_size, config.input_size]
    if result["policy"]["name"] != "stop_epoch":
        raise TrainCoreDataError("stateful sample-count augmentation is not supported")
    result["policy"]["epoch"] = config.augmentation_stop_internal_epoch
    return result


def _vendor_tools(root: Path, transform_config: dict[str, Any]):
    _vendor_source_identities(root)
    with _vendor_path(_vendor_root(root)):
        importlib.import_module("src.data")
        from src.data.dataset.coco_dataset import ConvertCocoPolysToMask
        from src.data.transforms.container import Compose
        from src.data._misc import convert_to_tv_tensor
        from src.data.dataloader import BatchImageCollateFunction
        _vendor_source_identities(root)
        transforms = Compose(**{key: copy.deepcopy(value) for key, value in transform_config.items()
                                if key != "type"})
    return ConvertCocoPolysToMask(False), transforms, convert_to_tv_tensor, BatchImageCollateFunction


def _source_binding(root: Path) -> dict[str, Any]:
    import PIL
    import torchvision
    paths = [
        Path(__file__), root / "src/sparse_rtdetr/baseline/categories.py",
        root / "src/sparse_rtdetr/baseline/config.py",
        root / "src/sparse_rtdetr/baseline/training_v2b.py",
        root / "vendor/rtdetrv2_pytorch/src/data/dataset/coco_dataset.py",
        root / "vendor/rtdetrv2_pytorch/src/data/transforms/container.py",
        root / "vendor/rtdetrv2_pytorch/src/data/transforms/_transforms.py",
        root / "vendor/rtdetrv2_pytorch/src/data/_misc.py",
        root / "vendor/rtdetrv2_pytorch/src/data/dataloader.py",
    ]
    sources = []
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise TrainCoreDataError("data loader source belongs to another checkout")
        sources.append({"path": str(resolved.relative_to(root)),
                        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest()})
    return {"sources": sources, "torch": str(torch.__version__),
            "torchvision": torchvision.__version__, "numpy": np.__version__,
            "pillow": PIL.__version__}


def _validate_documents(coco: dict, manifest: dict) -> tuple[list[dict], dict[int, list[dict]], dict[int, dict]]:
    images, annotations, categories = (coco.get(k) for k in ("images", "annotations", "categories"))
    records = manifest.get("records")
    if not all(type(value) is list for value in (images, annotations, categories, records)):
        raise TrainCoreDataError("COCO arrays and manifest records are required")
    if (len(categories) != 10 or any(type(cat) is not dict or type(cat.get("id")) is not int
                                    for cat in categories) or
            sorted(cat["id"] for cat in categories) != list(range(1, 11))):
        raise TrainCoreDataError("VisDrone categories must be exactly 1..10")
    image_map, stable_ids, filenames = {}, set(), set()
    for image in images:
        if type(image) is not dict:
            raise TrainCoreDataError("COCO image must be an object")
        image_id = _integer(image.get("id"), "image id", 1)
        filename = _logical_filename(image.get("file_name"))
        stable_id = _sha(image.get("stable_image_id"), "stable_image_id")
        for name in ("width", "height"):
            _integer(image.get(name), name, 1)
        if image.get("split") != "train":
            raise TrainCoreDataError("COCO image is not a train_core source member")
        if image_id in image_map or stable_id in stable_ids or filename in filenames:
            raise TrainCoreDataError("duplicate COCO image identity")
        image_map[image_id] = image
        stable_ids.add(stable_id)
        filenames.add(filename)
    manifests = {}
    for record in records:
        if type(record) is not dict:
            raise TrainCoreDataError("manifest record must be an object")
        image_id = _integer(record.get("coco_image_id"), "manifest image id", 1)
        if image_id not in image_map or image_id in manifests:
            raise TrainCoreDataError("manifest membership does not match train_core COCO")
        image = image_map[image_id]
        _logical_filename(record.get("relative_path"))
        if (record.get("relative_path") != image["file_name"] or
                record.get("stable_image_id") != image["stable_image_id"] or
                record.get("split") != "train" or
                any(record.get(key) != image[key] for key in ("width", "height"))):
            raise TrainCoreDataError("manifest identity differs from train_core COCO")
        _sha(record.get("image_sha256"), "raw image SHA-256")
        _integer(record.get("image_size_bytes"), "raw image size", 1)
        manifests[image_id] = record
    if set(manifests) != set(image_map):
        raise TrainCoreDataError("train_core manifest is incomplete")
    grouped, annotation_ids = {key: [] for key in image_map}, set()
    for annotation in annotations:
        if type(annotation) is not dict:
            raise TrainCoreDataError("COCO annotation must be an object")
        aid = _integer(annotation.get("id"), "annotation id", 1)
        image_id = _integer(annotation.get("image_id"), "annotation image id", 1)
        category = _integer(annotation.get("category_id"), "category id", 1)
        bbox, area = annotation.get("bbox"), annotation.get("area")
        if aid in annotation_ids or image_id not in grouped or category > 10:
            raise TrainCoreDataError("annotation identity/category outside train_core")
        if (type(bbox) is not list or len(bbox) != 4 or
                any(type(x) not in (int, float) or not math.isfinite(x) for x in bbox) or
                bbox[2] <= 0 or bbox[3] <= 0 or type(area) not in (int, float) or
                not math.isfinite(area) or area <= 0 or annotation.get("iscrowd", 0) not in (0, 1)):
            raise TrainCoreDataError("invalid COCO box, area or crowd flag")
        annotation_ids.add(aid)
        grouped[image_id].append(annotation)
    return [image_map[key] for key in sorted(image_map)], grouped, manifests


def augmented_tensor_sha256(images: torch.Tensor, targets: list[dict]) -> str:
    """Hash the actual augmented image/target tensors, not only sample IDs."""
    digest = hashlib.sha256()
    items = [("images", images)]
    for index, target in enumerate(targets):
        for name, value in sorted(target.items()):
            if not isinstance(value, torch.Tensor):
                raise TrainCoreDataError("vendor target fields must remain tensors")
            items.append((f"target:{index}:{name}", value))
    for name, tensor in items:
        plain = tensor.detach().cpu().contiguous().as_subclass(torch.Tensor)
        digest.update(_canonical([name, str(plain.dtype), list(plain.shape)]))
        digest.update(plain.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class TrainCoreDataset(Dataset):
    """Picklable verified COCO dataset using the actual vendored target converter."""

    def __init__(self, *, root: Path, image_root: Path, config: TrainCoreDataConfig,
                 images: list[dict], annotations: dict[int, list[dict]],
                 manifest: dict[int, dict], transforms: dict[str, Any], binding_sha256: str):
        self.root, self.image_root, self.config = root, image_root, config
        self.images, self.annotations, self.manifest = images, annotations, manifest
        self.transform_config, self.binding_sha256 = transforms, binding_sha256
        self._tools = None
        self._epoch = -1

    def __len__(self) -> int:
        return len(self.images)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_tools"] = None
        return state

    @property
    def epoch(self) -> int:
        return self._epoch

    def __getitem__(self, key: tuple[int, int, int]):
        if type(key) is not tuple or len(key) != 3:
            raise TrainCoreDataError("sample index must carry epoch and stable order position")
        index, epoch, position = key
        _integer(index, "sample index")
        _integer(epoch, "sample epoch", 1)
        _integer(position, "sample position")
        if index >= len(self.images):
            raise TrainCoreDataError("sample index exceeds train_core")
        image_record = self.images[index]
        seed = _sample_seed(self.config.seed, epoch, position, image_record["stable_image_id"])
        with _sample_rng(seed):
            if self._tools is None:
                self._tools = _vendor_tools(self.root, self.transform_config)
            converter, transforms, to_tv_tensor, _ = self._tools
            self._epoch = epoch - 1
            filename = _logical_filename(image_record["file_name"])
            path = _role_path((self.image_root / filename).resolve(strict=True), "selected image")
            if not path.is_relative_to(self.image_root):
                raise TrainCoreDataError("selected image symlink escapes the approved image root")
            raw = path.read_bytes()
            manifest = self.manifest[image_record["id"]]
            if len(raw) != manifest["image_size_bytes"] or hashlib.sha256(raw).hexdigest() != manifest["image_sha256"]:
                raise TrainCoreDataError("selected train_core image bytes differ from the bound manifest")
            with Image.open(io.BytesIO(raw)) as source:
                image = source.convert("RGB")
            if image.size != (image_record["width"], image_record["height"]):
                raise TrainCoreDataError("decoded image geometry differs from train_core metadata")
            image, target = converter(image, {"image_id": image_record["id"],
                                              "annotations": self.annotations[image_record["id"]]})
            target["idx"] = torch.tensor([index])
            target["boxes"] = to_tv_tensor(target["boxes"], key="boxes", spatial_size=image.size[::-1])
            target = map_target_labels_to_model(target)
            image, target, _ = transforms(image, target, self)
        descriptor = {"index": index, "epoch": epoch, "order_position": position,
                      "image_id": image_record["id"], "stable_image_id": image_record["stable_image_id"],
                      "relative_path": image_record["file_name"], "source_path": str(path),
                      "image_sha256": manifest["image_sha256"], "augmentation_seed": seed,
                      "binding_sha256": self.binding_sha256}
        return image, target, descriptor


@dataclass
class LogicalBatch:
    images: torch.Tensor
    targets: list[dict]
    samples: list[dict]
    epoch: int
    logical_batch_index: int
    binding_sha256: str
    augmented_sha256: str
    receipt_sha256: str

    def evidence(self) -> dict[str, Any]:
        return {"epoch": self.epoch, "logical_batch_index": self.logical_batch_index,
                "binding_sha256": self.binding_sha256, "samples": copy.deepcopy(self.samples),
                "augmented_sha256": self.augmented_sha256, "receipt_sha256": self.receipt_sha256}


class _LogicalCollator:
    def __init__(self, root: Path, transforms: dict, logical_batch_size: int):
        self.root, self.transforms, self.logical_batch_size = root, transforms, logical_batch_size
        self._collator = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_collator"] = None
        return state

    def __call__(self, items):
        if len(items) != self.logical_batch_size:
            raise TrainCoreDataError("incomplete logical batch reached the collator")
        if self._collator is None:
            _, _, _, collator_type = _vendor_tools(self.root, self.transforms)
            self._collator = collator_type(scales=None)
        descriptors = [item[2] for item in items]
        first = descriptors[0]
        epoch, binding = first["epoch"], first["binding_sha256"]
        start = first["order_position"]
        if start % self.logical_batch_size or any(
            item["order_position"] != start + index or item["epoch"] != epoch or item["binding_sha256"] != binding
            for index, item in enumerate(descriptors)
        ):
            raise TrainCoreDataError("collated samples cross logical windows or data identities")
        self._collator.set_epoch(epoch - 1)
        images, targets = self._collator([(item[0], item[1]) for item in items])
        augmented = augmented_tensor_sha256(images, targets)
        index = start // self.logical_batch_size
        receipt = _digest({"epoch": epoch, "logical_batch_index": index, "samples": descriptors,
                           "binding_sha256": binding, "augmented_sha256": augmented})
        return LogicalBatch(images, targets, descriptors, epoch, index, binding, augmented, receipt)


_STATE_KEYS = {"schema_version", "binding_sha256", "epoch", "epoch_active",
               "epoch_start_optimizer_updates", "optimizer_updates", "completed_batches",
               "logical_batch_size", "dataset_size", "batches_per_epoch",
               "dropped_samples", "order_sha256"}


def validate_loader_state(state: Any, engine_state: Any = None) -> dict[str, Any]:
    """Validate detached cursor/engine state without reading data or constructing a loader."""
    if type(state) is not dict or set(state) != _STATE_KEYS:
        raise TrainCoreDataError("loader state schema mismatch")
    if _integer(state["schema_version"], "schema_version", 1) != 1:
        raise TrainCoreDataError("unsupported loader state version")
    _sha(state["binding_sha256"], "loader binding")
    _sha(state["order_sha256"], "sample order")
    epoch = _integer(state["epoch"], "epoch")
    batch_size = _integer(state["logical_batch_size"], "logical_batch_size", 1)
    size = _integer(state["dataset_size"], "dataset_size", batch_size)
    count = _integer(state["batches_per_epoch"], "batches_per_epoch", 1)
    dropped = _integer(state["dropped_samples"], "dropped_samples")
    completed = _integer(state["completed_batches"], "completed_batches")
    start = _integer(state["epoch_start_optimizer_updates"], "epoch_start_optimizer_updates")
    updates = _integer(state["optimizer_updates"], "optimizer_updates")
    if type(state["epoch_active"]) is not bool:
        raise TrainCoreDataError("epoch_active must be boolean")
    if count != size // batch_size or dropped != size % batch_size or completed > count:
        raise TrainCoreDataError("loader logical-tail counters mismatch")
    if epoch == 0:
        if state["epoch_active"] or completed or start or updates or state["order_sha256"] != _digest([]):
            raise TrainCoreDataError("fresh loader cannot have progress or a nonempty order")
    elif (start != (epoch - 1) * count or updates != start + completed or
          (not state["epoch_active"] and completed != count)):
        raise TrainCoreDataError("loader committed epoch/update counters mismatch")
    if engine_state is not None:
        engine = validate_engine_state_dict(engine_state)
        for key in ("epoch", "epoch_active", "epoch_start_optimizer_updates", "optimizer_updates"):
            if state[key] != engine[key]:
                raise TrainCoreDataError(f"loader/engine {key} mismatch")
        config = engine["config"]
        if config["physical_batch_size"] * config["accumulation_steps"] != batch_size:
            raise TrainCoreDataError("engine and loader logical batch sizes differ")
    return copy.deepcopy(state)


class TrainCoreLogicalLoader:
    """Common logical batches with acknowledgments after successful engine updates.

    Checkpoint state is the committed cursor. Prefetched/yielded data is safely
    regenerated after restore because augmentation is keyed to immutable sample
    identity and does not consume the model/DN random stream.
    """

    def __init__(self, dataset: TrainCoreDataset, binding: dict[str, Any]):
        self.dataset, self.config = dataset, dataset.config
        self._binding = copy.deepcopy(binding)
        self.binding_sha256 = _digest(binding)
        self._epoch = 0
        self._epoch_active = False
        self._completed = 0
        self._iterating = False
        self._pending = None

    @property
    def binding(self) -> dict[str, Any]:
        return {**copy.deepcopy(self._binding), "loader_binding_sha256": self.binding_sha256}

    @property
    def has_active_iterator(self) -> bool:
        return self._iterating

    @property
    def batches_per_epoch(self) -> int:
        return len(self.dataset) // self.config.logical_batch_size

    def _plan(self, epoch: int) -> list[tuple[int, ...]]:
        if epoch == 0:
            return []
        return list(logical_batch_indices(len(self.dataset), seed=self.config.seed, epoch=epoch,
                                          logical_batch_size=self.config.logical_batch_size))

    def state_dict(self) -> dict[str, Any]:
        start = max(self._epoch - 1, 0) * self.batches_per_epoch
        return {"schema_version": 1, "binding_sha256": self.binding_sha256,
                "epoch": self._epoch, "epoch_active": self._epoch_active,
                "epoch_start_optimizer_updates": start,
                "optimizer_updates": start + self._completed,
                "completed_batches": self._completed,
                "logical_batch_size": self.config.logical_batch_size,
                "dataset_size": len(self.dataset), "batches_per_epoch": self.batches_per_epoch,
                "dropped_samples": len(self.dataset) % self.config.logical_batch_size,
                "order_sha256": _digest(self._plan(self._epoch))}

    def validate_state_dict(self, state: Any) -> dict[str, Any]:
        checked = validate_loader_state(state)
        current = self.state_dict()
        for key in ("binding_sha256", "logical_batch_size", "dataset_size", "batches_per_epoch", "dropped_samples"):
            if checked[key] != current[key]:
                raise TrainCoreDataError(f"loader restored {key} differs from runtime data")
        if checked["order_sha256"] != _digest(self._plan(checked["epoch"])):
            raise TrainCoreDataError("loader restored sample order hash differs")
        return checked

    def validate_engine_state(self, state: Any, engine_state: Any) -> dict[str, Any]:
        self.validate_state_dict(state)
        checked = validate_loader_state(state, engine_state)
        if engine_state["config"]["expected_input_size"] != [self.config.input_size, self.config.input_size]:
            raise TrainCoreDataError("engine and loader input geometry differ")
        return checked

    def validate_restore_state_dict(self, state: Any) -> dict[str, Any]:
        checked = self.validate_state_dict(state)
        if self._iterating:
            raise TrainCoreDataError("close active loader iterator before restoring")
        return checked

    def load_state_dict(self, state: Any) -> None:
        checked = self.validate_restore_state_dict(state)
        self._epoch, self._epoch_active = checked["epoch"], checked["epoch_active"]
        self._completed, self._pending = checked["completed_batches"], None

    def begin_epoch(self, epoch: int, engine_state: Any) -> None:
        _integer(epoch, "epoch", 1)
        if self._iterating or self._epoch_active or epoch != self._epoch + 1:
            raise TrainCoreDataError("epochs must advance after completing and closing the prior loader")
        proposed = self.state_dict()
        proposed.update(epoch=epoch, epoch_active=True, completed_batches=0,
                        epoch_start_optimizer_updates=(epoch - 1) * self.batches_per_epoch,
                        optimizer_updates=(epoch - 1) * self.batches_per_epoch,
                        order_sha256=_digest(self._plan(epoch)))
        self.validate_engine_state(proposed, engine_state)
        self.load_state_dict(proposed)

    def finish_epoch(self, engine_state: Any) -> None:
        if self._iterating or not self._epoch_active or self._completed != self.batches_per_epoch:
            raise TrainCoreDataError("cannot finish before all complete logical batches are committed")
        proposed = self.state_dict()
        proposed["epoch_active"] = False
        self.validate_engine_state(proposed, engine_state)
        self.load_state_dict(proposed)

    def _raw_batches(self, *, limit: int | None = None, epoch: int | None = None):
        validate_worker_environment(self.config)
        selected_epoch = self._epoch if epoch is None else epoch
        completed = self._completed if epoch is None else 0
        plan = self._plan(selected_epoch)
        stop = len(plan) if limit is None else min(len(plan), completed + limit)
        sample_batches = [
            [(index, selected_epoch, logical_index * self.config.logical_batch_size + position)
             for position, index in enumerate(plan[logical_index])]
            for logical_index in range(completed, stop)
        ]
        # DataLoader's iterator seed uses a dedicated CPU generator, not model RNG.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_sample_seed(self.config.seed, selected_epoch, 0, self.binding_sha256))
        options = {}
        if self.config.num_workers:
            options.update(prefetch_factor=self.config.prefetch_factor,
                           multiprocessing_context=self.config.multiprocessing_context)
        loader = DataLoader(self.dataset, batch_sampler=sample_batches,
                            num_workers=self.config.num_workers, persistent_workers=False,
                            pin_memory=False, generator=generator, worker_init_fn=seed_worker,
                            collate_fn=_LogicalCollator(self.dataset.root, self.dataset.transform_config,
                                                       self.config.logical_batch_size), **options)
        iterator = iter(loader)
        try:
            yield from iterator
        finally:
            if hasattr(iterator, "_shutdown_workers"):
                iterator._shutdown_workers()

    def __iter__(self) -> Iterator[LogicalBatch]:
        if not self._epoch_active or self._iterating:
            raise TrainCoreDataError("one iterator for an active epoch is required")
        self._iterating = True
        try:
            with contextlib.closing(self._raw_batches()) as batches:
                for batch in batches:
                    self._pending = batch.receipt_sha256
                    yield batch
                    if self._completed != batch.logical_batch_index + 1:
                        raise TrainCoreDataError("logical batch must be committed after a successful update before advancing")
        finally:
            self._iterating = False
            self._pending = None

    def validate_pending_batch(self, batch: LogicalBatch) -> None:
        """Validate the yielded receipt and tensors before any model update."""
        if (not isinstance(batch, LogicalBatch) or not self._epoch_active or
                batch.binding_sha256 != self.binding_sha256 or batch.epoch != self._epoch or
                batch.logical_batch_index != self._completed or self._pending != batch.receipt_sha256):
            raise TrainCoreDataError("batch acknowledgment is stale, foreign, or was not yielded")
        if batch.augmented_sha256 != augmented_tensor_sha256(batch.images, batch.targets):
            raise TrainCoreDataError("augmented batch changed after yield")
        receipt = _digest({"epoch": batch.epoch, "logical_batch_index": batch.logical_batch_index,
                           "samples": batch.samples, "binding_sha256": batch.binding_sha256,
                           "augmented_sha256": batch.augmented_sha256})
        if receipt != batch.receipt_sha256:
            raise TrainCoreDataError("sample receipt changed after yield")

    def commit_batch(self, batch: LogicalBatch, engine_state: Any) -> None:
        self.validate_pending_batch(batch)
        proposed = self.state_dict()
        proposed["completed_batches"] += 1
        proposed["optimizer_updates"] += 1
        self.validate_engine_state(proposed, engine_state)
        self._completed += 1
        self._pending = None

    def preview_batches(self, limit: int, *, epoch: int | None = None) -> Iterator[LogicalBatch]:
        """Bounded CPU loader-only inspection; no cursor or model update occurs.

        An explicit epoch previews that epoch from its first batch without an
        engine or any cursor change. With no epoch, preview the active cursor.
        The batch sampler itself is capped, so worker prefetch cannot read images
        outside the requested number of logical batches.
        """
        _integer(limit, "preview limit", 1)
        if epoch is not None:
            _integer(epoch, "preview epoch", 1)
        if self._iterating or (epoch is None and not self._epoch_active):
            raise TrainCoreDataError("preview requires a closed loader and an active or explicit epoch")
        self._iterating = True
        try:
            yield from self._raw_batches(limit=limit, epoch=epoch)
        finally:
            self._iterating = False


def build_train_core_loader(config: TrainCoreDataConfig, *, annotation_file: str | Path,
                            annotation_sha256: str, manifest_file: str | Path,
                            manifest_sha256: str, image_root: str | Path,
                            repo_root: str | Path | None = None) -> TrainCoreLogicalLoader:
    """Bind verified train_core metadata without discovering or reading raw images."""
    if not isinstance(config, TrainCoreDataConfig) or config.role != "train_core":
        raise TrainCoreDataError("only an explicit train_core data configuration is permitted")
    annotation = _role_path(annotation_file, "annotation")
    manifest = _role_path(manifest_file, "manifest")
    raw_root = _role_path(image_root, "image_root")
    if annotation.name != "train_core_coco.json" or manifest.name != "train_core_manifest.json":
        raise TrainCoreDataError("only named train_core metadata is permitted")
    root = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[3]
    validate_worker_environment(config)
    coco, annotation_ref = _read_bound_json(annotation, annotation_sha256, "annotation")
    source_manifest, manifest_ref = _read_bound_json(manifest, manifest_sha256, "manifest")
    images, annotations, manifests = _validate_documents(coco, source_manifest)
    if len(images) < config.logical_batch_size:
        raise TrainCoreDataError("train_core must contain at least one complete logical batch")
    raw_root = _role_path(raw_root.resolve(strict=True), "image_root")
    if not raw_root.is_dir():
        raise TrainCoreDataError("approved image_root must be a directory")
    transform_config = _transform_config(root, config)
    # Construct only transforms once to validate the actual imported implementation.
    with _sample_rng(config.seed):
        _vendor_tools(root, transform_config)
    binding = {"schema_version": 1, "role": "train_core", "annotation": annotation_ref,
               "manifest": manifest_ref, "image_root": str(raw_root),
               "sample_count": len(images), "annotation_count": len(coco["annotations"]),
               "config": config.semantic_config(), "transforms": transform_config,
               "source": _source_binding(root)}
    dataset = TrainCoreDataset(root=root, image_root=raw_root, config=config,
                               images=images, annotations=annotations, manifest=manifests,
                               transforms=transform_config, binding_sha256=_digest(binding))
    return TrainCoreLogicalLoader(dataset, binding)
