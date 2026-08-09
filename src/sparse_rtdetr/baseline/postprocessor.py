"""VisDrone postprocessor wrapper around the vendored RT-DETR math."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn

from ..data_protocol.evaluation import Detection
from .categories import map_model_tensor_to_coco
from .config import _vendor_path
from .contract import BaselineContractError, NUM_CLASSES, NUM_TOP_QUERIES


def _vendor_postprocessor(vendor_root: Path):
    with _vendor_path(vendor_root):
        from src.zoo.rtdetr.rtdetr_postprocessor import RTDETRPostProcessor

        return RTDETRPostProcessor


class VisDronePostProcessor(nn.Module):
    """Keep vendor sigmoid/top-k/box conversion and only remap final labels."""

    def __init__(
        self,
        vendor_root: str | Path | None = None,
        num_top_queries: int = NUM_TOP_QUERIES,
        vendor_postprocessor: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if type(num_top_queries) is not int or num_top_queries != NUM_TOP_QUERIES:
            raise BaselineContractError("baseline num_top_queries is frozen at 300")
        if vendor_postprocessor is None:
            if vendor_root is None:
                raise BaselineContractError("vendor_root is required for postprocessor construction")
            vendor_cls = _vendor_postprocessor(Path(vendor_root))
            vendor_postprocessor = vendor_cls(
                num_classes=NUM_CLASSES,
                use_focal_loss=True,
                num_top_queries=NUM_TOP_QUERIES,
                remap_mscoco_category=False,
            )
        if getattr(vendor_postprocessor, "remap_mscoco_category", False) is not False:
            raise BaselineContractError("vendor COCO remapping must remain false")
        self._vendor = vendor_postprocessor
        self.num_classes = NUM_CLASSES
        self.num_top_queries = NUM_TOP_QUERIES
        self.nms = False
        self._assert_vendor_identity()

    def _assert_vendor_identity(self) -> None:
        if getattr(self._vendor, "num_classes", None) != NUM_CLASSES:
            raise BaselineContractError("vendor postprocessor num_classes must be 10")
        if type(getattr(self._vendor, "num_top_queries", None)) is not int or self._vendor.num_top_queries != NUM_TOP_QUERIES:
            raise BaselineContractError("vendor postprocessor num_top_queries must be 300")
        if getattr(self._vendor, "use_focal_loss", None) is not True:
            raise BaselineContractError("vendor postprocessor must use focal loss")
        if getattr(self._vendor, "remap_mscoco_category", None) is not False:
            raise BaselineContractError("vendor COCO remapping must remain false")

    def forward(self, outputs: dict[str, torch.Tensor], orig_target_sizes: torch.Tensor):
        self._assert_vendor_identity()
        result = self._vendor(outputs, orig_target_sizes)
        if isinstance(result, tuple):
            labels, boxes, scores = result
            return map_model_tensor_to_coco(labels), boxes, scores

        mapped = []
        for item in result:
            if not isinstance(item, dict) or set(item) != {"labels", "boxes", "scores"}:
                raise BaselineContractError("vendor postprocessor returned an unexpected result")
            mapped.append(
                {
                    "labels": map_model_tensor_to_coco(item["labels"]),
                    "boxes": item["boxes"],
                    "scores": item["scores"],
                }
            )
        return mapped

    def deploy(self):
        if hasattr(self._vendor, "deploy"):
            self._vendor.deploy()
        return self

    def to_detections(
        self,
        results: Iterable[dict[str, torch.Tensor]],
        stable_image_ids: Iterable[str],
    ) -> tuple[Detection, ...]:
        """Bind per-image output rows to stable IDs and the project schema."""

        if isinstance(stable_image_ids, (str, bytes)):
            raise BaselineContractError("stable_image_ids must be an iterable of non-empty strings")
        result_list = list(results)
        image_ids = list(stable_image_ids)
        if len(result_list) != len(image_ids):
            raise BaselineContractError("stable_image_ids length must match postprocessor results")
        detections: list[Detection] = []
        for image_id, result in zip(image_ids, result_list):
            if type(image_id) is not str or not image_id:
                raise BaselineContractError("stable_image_id must be a non-empty string")
            if not isinstance(result, dict) or set(result) != {"labels", "boxes", "scores"}:
                raise BaselineContractError("postprocessor result schema is invalid")
            labels = result["labels"]
            boxes = result["boxes"]
            scores = result["scores"]
            if not torch.is_tensor(labels) or not torch.is_tensor(boxes) or not torch.is_tensor(scores):
                raise BaselineContractError("postprocessor result values must be tensors")
            if labels.ndim != 1 or boxes.ndim != 2 or boxes.shape[-1] != 4 or scores.ndim != 1:
                raise BaselineContractError("postprocessor result tensor shapes are invalid")
            if not (len(labels) == len(boxes) == len(scores) <= NUM_TOP_QUERIES):
                raise BaselineContractError("postprocessor result count violates top-k contract")
            if labels.dtype == torch.bool or labels.dtype not in {
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            }:
                raise BaselineContractError("Detection labels must use an integer tensor dtype")
            if labels.numel() and (bool((labels < 1).any()) or bool((labels > NUM_CLASSES).any())):
                raise BaselineContractError("Detection category IDs must be in [1, 10]")
            if not bool(torch.isfinite(boxes).all()) or not bool(torch.isfinite(scores).all()):
                raise BaselineContractError("Detection boxes and scores must be finite")
            for label, box, score in zip(labels, boxes, scores):
                detections.append(
                    Detection(
                        image_id=str(image_id),
                        category_id=int(label.item()),
                        bbox_xyxy=tuple(float(value.item()) for value in box),
                        score=float(score.item()),
                    )
                )
        return tuple(detections)
