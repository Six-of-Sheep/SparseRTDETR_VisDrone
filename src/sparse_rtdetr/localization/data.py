"""Role-bound raw VisDrone data and replayable, GT-driven augmentation.

Ignore regions never participate in crop sampling or positive matching. Their
geometry follows the single parameters actually used for the image/valid GT.
Physical duplicate annotation rows retain distinct stable identities.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import tv_tensors
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as TF
from torchvision.ops import box_convert

from ..data_protocol.parser import parse_annotation_bytes
from ..data_protocol.schema import stable_annotation_id, stable_image_id
from ..baseline.training_v2b import logical_batch_indices, seed_worker
from ..baseline.training_v2b_data import _sample_rng, _sample_seed
from .ignore import disjoint_union
from .material_paths import MANIFESTS,metadata_path

def digest(raw): return hashlib.sha256(raw).hexdigest()


def validate_role_roots(role, images_path, annotations_path, root_binding):
    """Admit only frozen role-specific directories, before reading any payload.

    Raw source trees must first be staged as the authorized role subset. A
    matching filename or a post-read SHA failure cannot authorize another role.
    The deploy contract freezes these absolute paths and its own SHA.
    """
    if role not in MANIFESTS or not isinstance(root_binding, dict) or set(root_binding) != {"role", "images", "annotations"} or root_binding["role"] != role:
        raise ValueError("missing or mismatched frozen data-root binding")
    roots = []
    for kind, supplied in (("images", images_path), ("annotations", annotations_path)):
        path, frozen = Path(supplied), Path(root_binding[kind])
        if not path.is_absolute() or not frozen.is_absolute():
            raise ValueError("data roots must be absolute")
        for candidate in (path, frozen, path.resolve(), frozen.resolve()):
            parts = [p.lower().replace("-", "_") for p in candidate.parts]
            if any(any(word in p for word in ("holdout", "confirmatory", "test_dev", "testdev")) for p in parts):
                raise ValueError("forbidden data role path before payload read")
            if candidate.name != kind or candidate.parent.name != role:
                raise ValueError("data root is not a staged directory of the requested role")
        if path.resolve() != frozen.resolve() or any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("data root differs from frozen path or traverses a symlink")
        if not path.is_dir():
            raise ValueError("role-bound data directory is missing")
        roots.append(path.resolve())
    return tuple(roots)


def read_bound(path, expected_sha, expected_size):
    p = Path(path)
    if not p.is_file() or p.is_symlink():
        raise ValueError("bound regular file missing or symlink: " + str(p))
    raw = p.read_bytes()
    if len(raw) != expected_size or digest(raw) != expected_sha:
        raise ValueError("bound file SHA/size mismatch: " + str(p))
    return raw


def parse_raw(raw, image_id, *, expected_rows=None):
    parsed = parse_annotation_bytes(raw)
    if expected_rows is not None and len(parsed) != expected_rows:
        raise ValueError("raw physical annotation count mismatch")
    gt, ignored0, ignored_class, rows = [], [], [], []
    counts = Counter()
    for row in parsed:
        fields = row.fields
        identity = stable_annotation_id(image_id, row.physical_line_number, row.raw_line_sha256)
        valid_area = fields.width > 0 and fields.height > 0
        if not valid_area:
            role = "nonpositive_area"
        elif fields.category == 0:
            role = "spatial_ignore"
        elif 1 <= fields.category <= 10 and fields.score == 1:
            role = "valid_gt"
        elif 1 <= fields.category <= 10 and fields.score == 0:
            role = "class_ignore"
        elif fields.category == 11:
            role = "category11_not_spatial"
        else:
            raise ValueError("unrecognized raw annotation category/score")
        item = {"id": identity, "physical_line": row.physical_line_number,
                "raw_line_sha256": row.raw_line_sha256, "box": list(fields.bbox_xyxy),
                "category": fields.category, "score": fields.score, "raw_area": fields.area,
                "role": role, "raw_fields": list(row.raw_fields)}
        rows.append(item); counts[role] += 1
        if role == "valid_gt": gt.append(item)
        if role == "spatial_ignore": ignored0.append(item)
        if role == "class_ignore": ignored_class.append(item)
    return {"gt": gt, "spatial_ignore": ignored0, "class_ignore": ignored_class,
            "rows": rows, "counts": dict(counts)}


class CaptureZoom(T.RandomZoomOut):
    def _get_params(self, flat_inputs):
        result = super()._get_params(flat_inputs)
        self.actual_params = result
        return result


class CaptureCrop(T.RandomIoUCrop):
    def __call__(self, *inputs):
        # The bound project wrapper performs this extra p=0.8 draw.
        if torch.rand(1) >= .8:
            return inputs
        return super().forward(*inputs)

    def _get_params(self, flat_inputs):
        result = super()._get_params(flat_inputs)
        self.actual_params = result
        return result


def labels_and_rows(inputs):
    return inputs[1]["labels"], inputs[1]["row_indices"]


def _clip(rects, hw):
    rects = rects.copy()
    rects[:, [0, 2]] = np.clip(rects[:, [0, 2]], 0, hw[1])
    rects[:, [1, 3]] = np.clip(rects[:, [1, 3]], 0, hw[0])
    return rects


def _valid(rects): return np.all(rects[:, 2:] > rects[:, :2], axis=1)


class RawAugmentation:
    def __init__(self, input_hw=(768, 1344)):
        self.input_hw = tuple(input_hw)
        self.photo = T.RandomPhotometricDistort(p=.5)
        self.zoom = CaptureZoom(fill=0)
        self.crop = CaptureCrop()
        self.sanitize = T.SanitizeBoundingBoxes(min_size=1, labels_getter=labels_and_rows)
        self.resize = T.Resize(list(input_hw))

    def __call__(self, image, raw_target, *, epoch, seed, position, image_id, training=True):
        h, w = image.height, image.width
        original_hw = (h, w)
        all_gt = raw_target["gt"]
        gt_np = np.array([r["box"] for r in all_gt], dtype=np.float64).reshape(-1, 4)
        gt_np = _clip(gt_np, (h, w))
        # Match the old converter's initial positive-area removal, without dedup.
        kept = np.where(_valid(gt_np))[0]
        boxes = tv_tensors.BoundingBoxes(torch.tensor(gt_np[kept], dtype=torch.float32), format="XYXY", canvas_size=(h, w))
        target = {"boxes": boxes, "labels": torch.tensor([all_gt[i]["category"]-1 for i in kept], dtype=torch.long),
                  "row_indices": torch.tensor(kept, dtype=torch.long)}
        ignored = raw_target["spatial_ignore"] + raw_target["class_ignore"]
        rectangles = _clip(np.array([r["box"] for r in ignored], dtype=np.float64).reshape(-1, 4), (h, w))
        steps = []
        sample_seed = _sample_seed(seed, epoch, position, image_id)
        with _sample_rng(sample_seed):
            if training and epoch < 118:
                image, target = self.photo(image, target)
                self.zoom.actual_params = None
                image, target = self.zoom(image, target)
                params = self.zoom.actual_params
                if params:
                    left, top, right, bottom = params["padding"]
                    rectangles += np.array([left, top, left, top], dtype=np.float64)
                    steps.append({"operation": "zoom", "padding": [int(left), int(top), int(right), int(bottom)]})
                self.crop.actual_params = None
                image, target = self.crop(image, target)
                params = self.crop.actual_params
                if params:
                    top, left, height, width = [int(params[k]) for k in ("top", "left", "height", "width")]
                    rectangles -= np.array([left, top, left, top], dtype=np.float64)
                    rectangles = _clip(rectangles, (height, width))
                    steps.append({"operation": "crop", "top": top, "left": left, "height": height, "width": width})
            if training:
                image, target = self.sanitize(image, target)
                if torch.rand(1) < .5:
                    image = TF.horizontal_flip(image)
                    target["boxes"] = TF.horizontal_flip(target["boxes"])
                    rectangles[:, [0, 2]] = image.width - rectangles[:, [2, 0]]
                    steps.append({"operation": "horizontal_flip"})
            before_hw = (image.height, image.width)
            image, target = self.resize(image, target)
            rectangles *= np.array([self.input_hw[1]/before_hw[1], self.input_hw[0]/before_hw[0]] * 2)
            rectangles = _clip(rectangles, self.input_hw)
            if training:
                image, target = self.sanitize(image, target)
            pixels = TF.pil_to_tensor(image).float() / 255.
            gt_boxes = box_convert(target["boxes"].as_subclass(torch.Tensor), "xyxy", "cxcywh")
            target["boxes"] = gt_boxes / torch.tensor(self.input_hw[::-1] * 2, dtype=torch.float32)
        retained_rows = target.pop("row_indices").tolist()
        target["gt_ids"] = [all_gt[i]["id"] for i in retained_rows]
        target["raw_area"] = torch.tensor([all_gt[i]["raw_area"] for i in retained_rows], dtype=torch.float64)
        valid = _valid(rectangles)
        n0 = len(raw_target["spatial_ignore"])
        norm = np.array(self.input_hw[::-1] * 2)
        spatial = rectangles[:n0][valid[:n0]] / norm
        class_boxes = rectangles[n0:][valid[n0:]] / norm
        target["ignore_spatial_boxes"] = torch.tensor(spatial, dtype=torch.float64)
        target["ignore_union"] = torch.tensor(disjoint_union(spatial), dtype=torch.float64)
        target["ignore_class_boxes"] = torch.tensor(class_boxes, dtype=torch.float64)
        target["ignore_class_labels"] = torch.tensor([r["category"]-1 for r, keep in zip(ignored[n0:], valid[n0:]) if keep], dtype=torch.long)
        target["ignore_ids"] = [r["id"] for r, keep in zip(ignored, valid) if keep]
        target["image_id"] = image_id
        target["original_hw"] = torch.tensor(original_hw, dtype=torch.long)
        trace = {"image_id": image_id, "sample_seed": sample_seed, "epoch": epoch, "position": position,
                 "augmentation_strong": training and epoch < 118, "geometry_steps": steps,
                 "raw_counts": raw_target["counts"], "retained_gt_count": len(target["labels"]),
                 "gt_ids": target["gt_ids"], "ignore_ids": target["ignore_ids"]}
        identity = hashlib.sha256()
        identity.update(pixels.contiguous().numpy().tobytes())
        for name in ("boxes", "labels", "raw_area", "ignore_spatial_boxes", "ignore_union", "ignore_class_boxes", "ignore_class_labels"):
            identity.update(name.encode()); identity.update(target[name].contiguous().numpy().tobytes())
        identity.update(json.dumps([target["gt_ids"], target["ignore_ids"]], separators=(",", ":")).encode())
        trace["augmented_tensor_sha256"] = identity.hexdigest()
        return pixels, target, trace


class RawRoleDataset(Dataset):
    def __init__(self, *, role, manifest_path, images_path, annotations_path, input_hw=(768,1344),
                 selection_path=None, selection_sha256=None, seed=0, root_binding=None):
        if role not in MANIFESTS or type(seed) is not int or seed not in (0,1,2):
            raise ValueError("unauthorized role or seed")
        self.images_path, self.annotations_path = validate_role_roots(role, images_path, annotations_path, root_binding)
        expected, count, split = MANIFESTS[role]
        raw = metadata_path(role,manifest_path,expected).read_bytes()
        if digest(raw) != expected:
            raise ValueError("role manifest SHA mismatch")
        records = json.loads(raw)["records"]
        if len(records) != count or len({r["stable_image_id"] for r in records}) != count:
            raise ValueError("role manifest member count/identity mismatch")
        for row in records:
            if row["split"] != split or stable_image_id(split, row["relative_path"]) != row["stable_image_id"]:
                raise ValueError("role stable identity mismatch")
            for key, sub in (("relative_path", "images"), ("annotation_relative_path", "annotations")):
                p = Path(row[key])
                if p.parts[:2] != (split, sub) or len(p.parts) != 3:
                    raise ValueError("cross-role or nested path in manifest")
        if role == "development":
            if selection_path is None or selection_sha256 is None:
                raise ValueError("development is restricted to the frozen 64 IDs")
            selection_raw = metadata_path('selection',selection_path,selection_sha256).read_bytes()
            if digest(selection_raw) != selection_sha256:
                raise ValueError("trace selection SHA mismatch")
            selection = json.loads(selection_raw)
            ordered = sorted(records, key=lambda r: (digest(("loc-trace-v1/"+r["stable_image_id"]).encode()), r["stable_image_id"]))[:64]
            if selection["source_manifest_sha256"] != expected or selection["selected"] != [dict(r, selection_key=digest(("loc-trace-v1/"+r["stable_image_id"]).encode())) for r in ordered]:
                raise ValueError("trace selection is not the frozen deterministic first64")
            records = ordered
        self.records, self.role, self.seed = records, role, seed
        self.by_id = {r["stable_image_id"]: r for r in records}
        self.root_binding = dict(root_binding)
        self.manifest_sha256 = expected
        self.transform = RawAugmentation(input_hw)

    def __len__(self): return len(self.records)

    def __getitem__(self, key):
        if self.role != "train_core":
            raise ValueError("development trace must use the separate non-augmented trace reader")
        epoch, position, index = key
        row = self.records[index]
        image_bytes, raw_annotation = self.bound_bytes(row)
        with Image.open(io.BytesIO(image_bytes)) as raw_image:
            image = raw_image.convert("RGB")
        if (image.width, image.height) != (row["width"], row["height"]):
            raise ValueError("decoded image dimensions differ from manifest")
        target = parse_raw(raw_annotation, row["stable_image_id"], expected_rows=row["raw_annotation_rows"])
        pixels, target, trace = self.transform(image, target, epoch=epoch, seed=self.seed, position=position,
                                                image_id=row["stable_image_id"], training=True)
        trace.update(image_sha256=row["image_sha256"], annotation_sha256=row["annotation_sha256"])
        return pixels, target, trace

    def bound_bytes(self, row):
        validate_role_roots(self.role, self.images_path, self.annotations_path, self.root_binding)
        if self.by_id.get(row.get("stable_image_id")) != row:
            raise ValueError("data row outside explicitly allowed membership")
        image_path = self.images_path / Path(row["relative_path"]).name
        annotation_path = self.annotations_path / Path(row["annotation_relative_path"]).name
        if not image_path.resolve().is_relative_to(self.images_path) or not annotation_path.resolve().is_relative_to(self.annotations_path):
            raise ValueError("bound path escapes role directory")
        return (read_bound(image_path, row["image_sha256"], row["image_size_bytes"]),
                read_bound(annotation_path, row["annotation_sha256"], row["annotation_size_bytes"]))


class WindowSampler:
    def __init__(self, dataset_size, *, seed, epoch, completed_windows, window_limit=None):
        self.dataset_size, self.seed, self.epoch, self.completed_windows = dataset_size, seed, epoch, completed_windows
        self.window_limit=window_limit
        if window_limit is not None and (type(window_limit) is not int or window_limit<1 or completed_windows+window_limit>dataset_size//16):
            raise ValueError("invalid finite loader window range")

    def __iter__(self):
        for window, indices in enumerate(logical_batch_indices(self.dataset_size, seed=self.seed, epoch=self.epoch,
                                      logical_batch_size=16, completed_batches=self.completed_windows), self.completed_windows):
            if self.window_limit is not None and window>=self.completed_windows+self.window_limit:break
            yield [(self.epoch, window*16+offset, index) for offset, index in enumerate(indices)]

    def __len__(self): return self.window_limit if self.window_limit is not None else self.dataset_size//16 - self.completed_windows


def collate_window(items):
    images, targets, traces = zip(*items)
    return torch.stack(images), list(targets), list(traces)


def training_loader(dataset, *, epoch, completed_windows=0, workers=2, window_limit=None):
    if dataset.role != "train_core" or len(dataset) != 4869 or type(dataset.seed) is not int or dataset.seed not in (0,1,2) or not 1 <= epoch <= 120 or not 0 <= completed_windows <= 304:
        raise ValueError("invalid F0 loader cursor")
    generator = torch.Generator().manual_seed(dataset.seed)
    return DataLoader(dataset, batch_sampler=WindowSampler(len(dataset), seed=dataset.seed, epoch=epoch, completed_windows=completed_windows,window_limit=window_limit),
                      num_workers=workers, collate_fn=collate_window, worker_init_fn=seed_worker,
                      multiprocessing_context="spawn" if workers else None,
                      prefetch_factor=2 if workers else None, persistent_workers=False, generator=generator,
                      timeout=300 if workers else 0)
