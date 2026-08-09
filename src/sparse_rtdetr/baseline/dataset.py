"""VisDrone COCO-artifact dataset adapter.

The vendor dataset is deliberately composed rather than modified. Its raw
``load_item`` result is mapped before this adapter invokes transforms.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .artifacts import validate_runtime_role
from .categories import map_target_labels_to_model
from .config import _vendor_path
from .contract import BaselineContractError


def _vendor_coco_detection(vendor_root: Path):
    with _vendor_path(vendor_root):
        from src.data.dataset.coco_dataset import CocoDetection

        return CocoDetection


class VisDroneCocoDetection:
    """Compose vendor COCO loading with the VisDrone 1..10 to 0..9 mapping."""

    def __init__(
        self,
        img_folder: str | Path,
        ann_file: str | Path,
        transforms: Any = None,
        return_masks: bool = False,
        remap_mscoco_category: bool = False,
        vendor_root: str | Path | None = None,
        role: str | None = None,
        vendor_dataset: Any = None,
    ) -> None:
        if remap_mscoco_category is not False:
            raise BaselineContractError("vendor COCO remapping must remain false for VisDrone")
        self.role = validate_runtime_role(role)
        if vendor_dataset is None:
            if vendor_root is None:
                raise BaselineContractError("vendor_root is required for dataset construction")
            dataset_cls = _vendor_coco_detection(Path(vendor_root))
            vendor_dataset = dataset_cls(
                img_folder=str(img_folder),
                ann_file=str(ann_file),
                transforms=None,
                return_masks=return_masks,
                remap_mscoco_category=False,
            )
        self._vendor = vendor_dataset
        self._transforms = transforms
        self.img_folder = Path(img_folder)
        self.ann_file = Path(ann_file)

    def __len__(self) -> int:
        return len(self._vendor)

    def load_item(self, index: int):
        image, target = self._vendor.load_item(index)
        return image, map_target_labels_to_model(target)

    def __getitem__(self, index: int):
        image, target = self.load_item(index)
        if self._transforms is not None:
            image, target, _ = self._transforms(image, target, self)
        return image, target

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self._vendor, "set_epoch"):
            self._vendor.set_epoch(epoch)

    @property
    def epoch(self) -> int:
        return getattr(self._vendor, "epoch", -1)

    @property
    def categories(self):
        return self._vendor.categories

    @property
    def category2name(self):
        return self._vendor.category2name

    @property
    def category2label(self):
        return {category_id: category_id - 1 for category_id in range(1, 11)}

    @property
    def label2category(self):
        return {label: label + 1 for label in range(10)}

    def __getattr__(self, name: str):
        vendor = self.__dict__.get("_vendor")
        if vendor is None:
            raise AttributeError(name)
        return getattr(vendor, name)

    def extra_repr(self) -> str:
        return f"img_folder: {self.img_folder}\nann_file: {self.ann_file}\nrole: {self.role}"
