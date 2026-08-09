"""Strict VisDrone COCO-artifact and RT-DETR label mappings."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .contract import BaselineContractError, COCO_CATEGORY_IDS, MODEL_LABEL_IDS, NUM_CLASSES


def _strict_int(value: Any, field: str) -> int:
    if type(value) is not int:
        raise BaselineContractError(f"{field} must be a strict integer")
    return value


def coco_category_to_model_label(category_id: Any) -> int:
    """Map a COCO artifact category ID in 1..10 to a model label in 0..9."""

    value = _strict_int(category_id, "COCO category_id")
    if value not in COCO_CATEGORY_IDS:
        raise BaselineContractError("COCO category_id must be in [1, 10]")
    return value - 1


def model_label_to_coco_category(label: Any) -> int:
    """Map an RT-DETR model label in 0..9 back to category ID 1..10."""

    value = _strict_int(label, "model label")
    if value not in MODEL_LABEL_IDS:
        raise BaselineContractError("model label must be in [0, 9]")
    return value + 1


def coco_categories_to_model_labels(category_ids: Sequence[Any]) -> tuple[int, ...]:
    """Map a finite scalar sequence while preserving order; empty is valid."""

    if isinstance(category_ids, (str, bytes)):
        raise BaselineContractError("category IDs must be a sequence of integers")
    try:
        return tuple(coco_category_to_model_label(value) for value in category_ids)
    except TypeError as exc:
        raise BaselineContractError("category IDs must be a sequence of integers") from exc


def model_labels_to_coco_categories(labels: Sequence[Any]) -> tuple[int, ...]:
    """Map a finite scalar sequence while preserving order; empty is valid."""

    if isinstance(labels, (str, bytes)):
        raise BaselineContractError("model labels must be a sequence of integers")
    try:
        return tuple(model_label_to_coco_category(value) for value in labels)
    except TypeError as exc:
        raise BaselineContractError("model labels must be a sequence of integers") from exc


def map_coco_tensor_to_model(labels: Any) -> Any:
    """Map an integer tensor without accepting float, bool, or invalid labels."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment contract supplies torch
        raise BaselineContractError("torch is required for tensor label mapping") from exc

    if not torch.is_tensor(labels):
        raise BaselineContractError("labels must be a torch tensor")
    if labels.dtype == torch.bool or labels.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise BaselineContractError("labels must use an integer tensor dtype")
    if labels.numel() == 0:
        return labels.clone()
    if bool((labels < 1).any()) or bool((labels > NUM_CLASSES).any()):
        raise BaselineContractError("COCO labels must be in [1, 10]")
    return labels.to(dtype=torch.int64) - 1


def map_model_tensor_to_coco(labels: Any) -> Any:
    """Map an integer model-label tensor without changing its shape or order."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment contract supplies torch
        raise BaselineContractError("torch is required for tensor label mapping") from exc

    if not torch.is_tensor(labels):
        raise BaselineContractError("labels must be a torch tensor")
    if labels.dtype == torch.bool or labels.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise BaselineContractError("labels must use an integer tensor dtype")
    if labels.numel() == 0:
        return labels.clone()
    if bool((labels < 0).any()) or bool((labels >= NUM_CLASSES).any()):
        raise BaselineContractError("model labels must be in [0, 9]")
    return labels.to(dtype=torch.int64) + 1


def map_target_labels_to_model(target: dict[str, Any]) -> dict[str, Any]:
    """Copy a vendor target and convert its COCO labels before transforms."""

    if not isinstance(target, dict) or "labels" not in target:
        raise BaselineContractError("target must be a dict containing labels")
    mapped = dict(target)
    mapped["labels"] = map_coco_tensor_to_model(target["labels"])
    return mapped
