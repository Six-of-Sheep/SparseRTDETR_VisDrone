"""Real feature reads for PDRA (rev 4): own-view P3 strips under one frozen RoIAlign rule.

Coordinate convention (frozen; the stride is applied in exactly one place)
--------------------------------------------------------------------------
original-image pixels --``ViewAffine.to_feature``--> feature units of the box's *own* view:
subtract the integer crop origin (no clipping), stretch by ``canvas/crop`` per axis, divide
by the P3 stride ``(8, 8)``.  Strips are clipped to the crop for the visibility test and
clamped to the feature extent (``pdra_core.read_exclusive``).  The resulting feature-unit
boxes are handed to ``torchvision.ops.roi_align`` with

    spatial_scale = 1.0, aligned = True, output_size = (2, 2), sampling_ratio = 2.

``aligned=True`` subtracts 0.5 feature units from the box before sampling, so a box
``[c, c+1]`` is centred on the centre of feature cell ``c`` (cell centres lie at box
coordinates ``c + 0.5``) and the box ``[0, W]`` covers the whole map with its outer
samples clamped at the boundary.  No other scaling, rounding or half-pixel offset is
applied anywhere in this module.  Passing canvas or original pixels together with
``spatial_scale = 1/stride`` is the *equivalent* spelling that this module deliberately
does not use, so a second division by the stride can never occur in the real path.

Slot layout (frozen)
--------------------
``subtract_strips`` yields up to four disjoint strips (top, bottom, left, right) for
intersecting boxes, or the single ``full`` box for disjoint cluster members.  A side
descriptor therefore has five fixed slots in this order; an absent slot is all zeros and
its mask bit is 0.  With ``C`` channels and a 2x2 grid one side is ``5 * 4C + 5`` values.
Controls (b)/(c) read the whole box as one ``full`` slot: ``4C + 1`` values.

Nothing here depends on ground truth.  The module is torch-only; the numpy decision core
stays importable without torch.
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
import torch
from torchvision.ops import roi_align

import pdra_core as P

GRID = P.SAMPLE_GRID                      # 2 -> 2x2 pooled cells per strip
SAMPLING_RATIO = 2                        # frozen: 2x2 bilinear samples per pooled cell
ALIGNED = True
SPATIAL_SCALE = 1.0                       # coordinates are already in feature units
SLOTS = ("top", "bottom", "left", "right", "full")
SLOT_INDEX = {name: index for index, name in enumerate(SLOTS)}
P3_CHANNELS = 256                         # frozen R18 HybridEncoder hidden_dim
CONVENTION = {
    "to_feature": "ViewAffine.to_feature (crop origin, canvas/crop stretch, /stride) once",
    "roi_align": {"spatial_scale": SPATIAL_SCALE, "aligned": ALIGNED,
                  "output_size": [GRID, GRID], "sampling_ratio": SAMPLING_RATIO},
    "aligned_semantics": "box shifted by -0.5 feature units; [0,1] samples cell 0 exactly",
    "double_scaling_guard": "no spatial_scale=1/stride path exists in this module",
    "slots": list(SLOTS), "channels": P3_CHANNELS,
}


def side_dim(channels: int = P3_CHANNELS) -> int:
    """PDRA side descriptor length: five 2x2 slots plus five mask bits."""
    return len(SLOTS) * GRID * GRID * channels + len(SLOTS)


def whole_dim(channels: int = P3_CHANNELS) -> int:
    """Whole-box (control) descriptor length: one 2x2 slot plus one mask bit."""
    return GRID * GRID * channels + 1


def pair_geometry_dim() -> int:
    return 5


def rois_of_read(read: P.RegionRead) -> list[tuple[int, tuple[float, float, float, float]]]:
    """Feature-unit boxes of a read, with their frozen slot index."""
    out = []
    for name, box in read.feature_strips:
        if name not in SLOT_INDEX:
            raise ValueError(f"unknown strip name {name!r}")
        out.append((SLOT_INDEX[name], tuple(float(v) for v in box)))
    return out


def whole_box_read(box, affine: P.ViewAffine) -> P.RegionRead:
    """Control read: the whole box as one ``full`` strip under the same source rule."""
    return P.read_exclusive((("full", P.box64(box)),), affine)


class FeatureBank:
    """The five own-view P3 maps of one image, ``{view_id: Tensor[C, H, W]}``."""

    def __init__(self, maps: Mapping[int, torch.Tensor], *, stride=P.FEATURE_STRIDE):
        if sorted(maps) != [0, 1, 2, 3, 4]:
            raise ValueError("a feature bank needs exactly the five views 0..4")
        shapes = {tuple(m.shape) for m in maps.values()}
        if len(shapes) != 1:
            raise ValueError(f"views differ in feature shape: {shapes}")
        c, h, w = next(iter(shapes))
        ext_h = math.ceil(P.CANVAS_HW[0] / stride[0])
        ext_w = math.ceil(P.CANVAS_HW[1] / stride[1])
        if (h, w) != (ext_h, ext_w):
            raise ValueError(f"feature extent {(h, w)} differs from canvas/stride {(ext_h, ext_w)}")
        self.maps = {int(k): v for k, v in maps.items()}
        self.channels, self.height, self.width = int(c), int(h), int(w)
        self.device = next(iter(maps.values())).device

    def pool_boxes(self, view_id: int, boxes: Sequence[Sequence[float]]) -> torch.Tensor:
        """RoIAlign of feature-unit boxes on one view map -> ``[K, C, GRID, GRID]``."""
        if not boxes:
            return torch.zeros((0, self.channels, GRID, GRID),
                               device=self.maps[int(view_id)].device)
        fmap = self.maps[int(view_id)]
        rois = torch.tensor([[0.0, *box] for box in boxes], dtype=torch.float32,
                            device=fmap.device)
        return roi_align(fmap.unsqueeze(0).float(), rois, output_size=(GRID, GRID),
                         spatial_scale=SPATIAL_SCALE, sampling_ratio=SAMPLING_RATIO,
                         aligned=ALIGNED)


def pool_sides(bank: FeatureBank, reads: Sequence[P.RegionRead], *,
               out_dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """Side descriptors for many reads -> ``[N, side_dim]``.

    Strips are grouped by their own source view and pooled in one RoIAlign call per view;
    every strip lands in its frozen slot.  Invalid reads produce an all-zero descriptor
    with an all-zero mask; their pair already abstains mechanically in ``decide_pair``,
    so they are never fed to a trained comparator as evidence.
    """
    n, c = len(reads), bank.channels
    per_slot = GRID * GRID * c
    out = torch.zeros((n, side_dim(c)), dtype=torch.float32, device=bank.device)
    by_view: dict[int, list[tuple[int, int, tuple]]] = {}
    for row, read in enumerate(reads):
        if not read.valid:
            continue
        if read.source_view is None:
            raise ValueError("a valid read must carry its source view")
        for slot, box in rois_of_read(read):
            by_view.setdefault(read.source_view, []).append((row, slot, box))
    for view_id, items in by_view.items():
        pooled = bank.pool_boxes(view_id, [box for _, _, box in items]).reshape(len(items), -1)
        rows = torch.tensor([r for r, _, _ in items], device=bank.device)
        slots = torch.tensor([s for _, s, _ in items], device=bank.device)
        for slot in range(len(SLOTS)):
            pick = slots == slot
            if pick.any():
                start = slot * per_slot
                out[rows[pick], start:start + per_slot] = pooled[pick]
                out[rows[pick], len(SLOTS) * per_slot + slot] = 1.0
    return out.to(out_dtype)


def pool_whole(bank: FeatureBank, reads: Sequence[P.RegionRead], *,
               out_dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """Whole-box descriptors for the controls -> ``[N, whole_dim]`` (full slot + mask)."""
    n, c = len(reads), bank.channels
    out = torch.zeros((n, whole_dim(c)), dtype=torch.float32, device=bank.device)
    by_view: dict[int, list[tuple[int, tuple]]] = {}
    for row, read in enumerate(reads):
        if not read.valid:
            continue
        rois = rois_of_read(read)
        if len(rois) != 1 or rois[0][0] != SLOT_INDEX["full"]:
            raise ValueError("a whole-box read must consist of exactly one full strip")
        by_view.setdefault(read.source_view, []).append((row, rois[0][1]))
    for view_id, items in by_view.items():
        pooled = bank.pool_boxes(view_id, [box for _, box in items]).reshape(len(items), -1)
        rows = torch.tensor([r for r, _ in items], device=bank.device)
        out[rows, :GRID * GRID * c] = pooled
        out[rows, GRID * GRID * c] = 1.0
    return out.to(out_dtype)


def geometry_tensor(pairs: Sequence[P.PairInput], device) -> torch.Tensor:
    """``[N, 5]`` = (dx, dy, log w ratio, log h ratio, IoU); the first four negate on swap."""
    return torch.tensor([list(pair.geometry) for pair in pairs], dtype=torch.float32,
                        device=device)


# --- float64 reference of the frozen RoIAlign rule (for the convention self-check) ---
def _bilinear(fmap: np.ndarray, y: float, x: float) -> np.ndarray:
    """torchvision ``bilinear_interpolate`` (forward) reproduced in float64."""
    height, width = fmap.shape[1:]
    if y < -1.0 or y > height or x < -1.0 or x > width:
        return np.zeros(fmap.shape[0], dtype=np.float64)
    y = max(y, 0.0)
    x = max(x, 0.0)
    y_low, x_low = int(math.floor(y)), int(math.floor(x))
    if y_low >= height - 1:
        y_high = y_low = height - 1
        y = float(y_low)
    else:
        y_high = y_low + 1
    if x_low >= width - 1:
        x_high = x_low = width - 1
        x = float(x_low)
    else:
        x_high = x_low + 1
    ly, lx = y - y_low, x - x_low
    hy, hx = 1.0 - ly, 1.0 - lx
    return (hy * hx * fmap[:, y_low, x_low] + hy * lx * fmap[:, y_low, x_high]
            + ly * hx * fmap[:, y_high, x_low] + ly * lx * fmap[:, y_high, x_high])


def reference_roi_align(fmap: np.ndarray, box: Sequence[float], *, grid: int = GRID,
                        sampling_ratio: int = SAMPLING_RATIO, aligned: bool = ALIGNED,
                        spatial_scale: float = SPATIAL_SCALE) -> np.ndarray:
    """Float64 replica of torchvision's RoIAlign forward for one box -> ``[C, grid, grid]``."""
    fmap = np.asarray(fmap, dtype=np.float64)
    offset = 0.5 if aligned else 0.0
    x0, y0, x1, y1 = (float(v) * spatial_scale for v in box)
    x0, y0, x1, y1 = x0 - offset, y0 - offset, x1 - offset, y1 - offset
    roi_w, roi_h = x1 - x0, y1 - y0
    if not aligned:
        roi_w, roi_h = max(roi_w, 1.0), max(roi_h, 1.0)
    bin_w, bin_h = roi_w / grid, roi_h / grid
    gw = sampling_ratio if sampling_ratio > 0 else int(math.ceil(roi_w / grid))
    gh = sampling_ratio if sampling_ratio > 0 else int(math.ceil(roi_h / grid))
    count = max(gh * gw, 1)
    out = np.zeros((fmap.shape[0], grid, grid), dtype=np.float64)
    for ph in range(grid):
        for pw in range(grid):
            acc = np.zeros(fmap.shape[0], dtype=np.float64)
            for iy in range(gh):
                y = y0 + ph * bin_h + (iy + 0.5) * bin_h / gh
                for ix in range(gw):
                    x = x0 + pw * bin_w + (ix + 0.5) * bin_w / gw
                    acc += _bilinear(fmap, y, x)
            out[:, ph, pw] = acc / count
    return out


# --- descriptor cache (server-side only; never returned to the workstation) ----------
def save_cache(path, payload: Mapping) -> dict:
    """Atomic ``torch.save`` of fp16 descriptors plus their pair index."""
    import hashlib
    import os
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("xb") as stream:
        torch.save(dict(payload), stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": h.hexdigest()}


def load_cache(path, device="cpu") -> dict:
    return torch.load(str(path), map_location=device, weights_only=True)


def cache_budget_bytes(n_pairs: int, *, channels: int = P3_CHANNELS, whole: bool = False,
                       boxes: int | None = None) -> int:
    """fp16 bytes for ``n_pairs`` PDRA pairs (two sides), or for whole-box reads."""
    if whole:
        return 2 * whole_dim(channels) * (boxes if boxes is not None else 2 * n_pairs)
    return 2 * 2 * side_dim(channels) * n_pairs
