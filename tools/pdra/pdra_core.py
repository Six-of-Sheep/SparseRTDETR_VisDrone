"""PDRA decision-and-geometry core (frozen rules, rev 3).

Pure CPU.  No torch, no detector, no training loop, no randomness.  Every rule in
this module is fixed by ``PAIRWISE_ARBITRATION_DESIGN_20261030.md`` rev 3.  Anything
that needs a *real* feature map sits behind an explicit provider interface; the CPU
prototype uses a deterministic geometric provider and therefore never claims a
detector-feature result.

Implemented here
----------------
S3  cross-view feature coordinate mapping, visibility domain, out-of-bounds policy
S3  exclusive difference-region (strip) construction and read validity
S4  preference / abstention decision, Copeland tournament, cycle and tie rules
S4  one pairing domain for training and inference (cluster-internal pairs)
S4  representative replacement followed by re-ranking and budget truncation
S5  cluster-level representative replacement, original-C fallback
S6  three-way supervision labelling with explicit ambiguity routing and counts
S8  frozen module train/eval sequence split and frozen diagnostic buckets
S7  control fairness asserted on arm specifications, separated from the
    geometric region-coverage margin

Coordinate facts used (verified against the frozen inference entry, 2026-10-30)
------------------------------------------------------------------------------
* Five views: ``view_id`` 0 = whole image, 1..4 = TL/TR/BL/BR half-open crops.
* Every view (whole image and each crop) is resized to one shared canvas
  ``(768, 1344)`` (height, width) before the forward pass.
* The model emits normalised cxcywh; the frozen entry scales it by the *crop*
  pixel size, i.e. the resize is an anisotropic stretch, so crop-local pixels are
  ``normalised * crop_size`` and original-image pixels add the integer crop origin.
* Candidate fields: original-image pixel xyxy, FP32 sigmoid score, category (1-based),
  view, query, view-internal rank.  Merge is classwise hard NMS, suppress iff
  ``IoU > 0.5``, iteration key ``(-score, view_id, query_id, category_id)``.
* Feature maps are read only through a read-only hook on the frozen HybridEncoder
  output (index 0 = P3, stride 8, 256 channels for the frozen R18); the frozen entry
  itself still returns only ``pred_logits``/``pred_boxes``.  The binding is declared
  in rev 4 and is marked verified only after the GPU engineering check PASSes
  (``FEATURE_BINDING_VERIFIED``).

rev-3 corrections over rev 2 (2026-10-30 second-round authorization)
-------------------------------------------------------------------
1. **One pairing domain.**  Training pairs and inference pairs are the *same* set:
   the unordered pairs inside each original-C cluster.  rev 2 drew training pairs
   from "same-class IoU>=0.5" over the whole pool while inference judged all pairs
   inside ``{survivor} U suppressed``; those two sets differ (two suppressed members
   of one cluster may sit below 0.5, and a >=0.5 pair may straddle two clusters).
   ``cluster_pair_records`` is now the single enumeration used by both.
2. **Ordering and budget truncation after substitution.**  rev 2 truncated the
   representative list in *cluster* order, which silently kept the survivor's rank
   after its score changed.  The representatives are now re-sorted with the frozen
   sort key and truncated to the budget afterwards, with no refill, for PDRA and for
   every control.
3. **Coverage vs fairness.**  The geometric fact that control (c) reads a superset
   region (margin ``2*area(i&j) >= 0``) is a *coverage* property and is now reported
   by ``region_coverage_margin``.  Control fairness is asserted separately, on the
   four arms' specifications (``ArmSpec``/``assert_control_fairness``), where the read
   rule is the only permitted difference.
4. **Frozen split and label rules.**  ``freeze_sequence_split`` fixes the module
   train/eval sequence partition deterministically, and ``frozen_hyperparameters``
   collects every frozen numeric rule; the main scientific verdict reads only the
   module-untrained (eval) sequences.
5. **rev-4 real wiring (2026-10-30).**  ``ViewAffine`` carries its own ``view_id`` and
   every ``RegionRead`` records the ``source_view`` it was mapped from, so a real
   feature provider reads each side from its own view's map.  ``PairInput`` carries the
   two candidate identities, so the decision layer can look up precomputed comparator
   outputs by identity.  ``pair_decisions`` enumerates the *complete* inference domain
   (every cluster-internal pair, identical to the training enumeration); supervision
   labels and their exclusions never enter this code path, which takes no GT argument.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


CORE_VERSION = "pdra-rules-rev4-20261030"
DESIGN_REVISION = "rev 4 (2026-10-30)"

# --- frozen numeric rules -----------------------------------------------------
CANVAS_HW = (768, 1344)          # every view resizes to this canvas (height, width)
FEATURE_STRIDE = (8, 8)          # rev 4: bound to HybridEncoder output-list index 0 (P3)
FEATURE_LEVEL = "hybrid_encoder_out_index_0_p3"
FEATURE_BINDING_VERIFIED = True  # rev6 GPU engineering check PASS (2026-10-31): task r18-pdra-gpu-check-20261031-002/attempt-001, RESULT.json sha256 3e18792511dfa282d373f957c3eabd54cf42e47cecb3efec14d9a55e24f7046d
FEATURE_READ = ("roi_align(aligned=True, spatial_scale=1.0, output_size=2x2, "
                "sampling_ratio=2) on the box's own-view P3 map, feature-unit coordinates")
NMS_IOU = 0.5                    # mirrors the frozen merge core
PER_VIEW_BUDGET = 600
FINAL_BUDGETS = (600, 300)
SAMPLE_GRID = 2                  # 2x2 pool per strip
DELTA_PREF = 0.1                 # preference dead zone; vote iff r >= 0.5+D or <= 0.5-D
DELTA_ABSTAIN = 0.5              # abstain iff a >= this
THETA_REGION = 0.5               # a read is valid iff visible area fraction >= this
MATCH_IOU = 0.5                  # GT match threshold
TAU_GAP = 0.05                   # min |dIoU| for a same-GT preference label
TAU_CLEAR = 0.25                 # "clear pair" diagnostic bucket
LAMBDA_ABSTAIN = 1.0             # weight of the abstention loss (frozen, not searched)

STRIP_ORDER = ("top", "bottom", "left", "right")
PAIR_DESTINATIONS = ("pref", "abstain", "excluded_empty", "excluded_gap",
                     "excluded_ambiguous")

# --- frozen pairing / ordering / evidence rules (rev 3) -----------------------
PAIR_DOMAIN = "cluster_internal_pairs_of_original_C"
PAIR_DOMAIN_NOTE = (
    "Training and inference share one pairing domain: the unordered pairs inside each "
    "original-C cluster ({survivor} U suppressed).  It is NOT 'same-class IoU>=0.5 pairs "
    "from the pool'; two suppressed members may sit below 0.5 and a >=0.5 pool pair may "
    "straddle two clusters."
)
POST_REPLACEMENT_RANKING = "canonical_sort_key_then_truncate_no_refill"
CLUSTER_BUILDER = "numpy_mirror_of_frozen_classwise_nms"

# Frozen diagnostic buckets (declared, never searched).
DIOU_BUCKETS = ("gap_0.05_0.10", "gap_0.10_0.15", "gap_0.15_0.25",
                "gap_0.25_0.50", "gap_ge_0.50")
AREA_BUCKETS = ("area_le_1024", "area_1024_9216", "area_gt_9216")
GAP_EDGES = (0.05, 0.10, 0.15, 0.25, 0.50)
AREA_EDGES = (1024.0, 9216.0)
N_MIN_PAIRS = 300                # a bucket below this pair count is "insufficient evidence"
N_MIN_IMAGES = 40                # ...and below this image count as well
EVIDENCE_INSUFFICIENT = "insufficient_evidence"

# Frozen module train/eval sequence split rule (rev 3).
SPLIT_RULE = "sorted_sequence_keys_stride3_remainder2_to_eval"
SPLIT_RULE_NOTE = (
    "Sequence keys (VisDrone sequence prefix of annotation_relative_path) are sorted "
    "ascending; keys at sorted index i with i % 3 == 2 go to the module-eval split, all "
    "others to module-train.  The rule is deterministic, sequence-level (no image "
    "leakage between splits) and frozen; the main scientific verdict reads only the "
    "module-untrained (eval) sequences."
)
MAIN_VERDICT_SPLIT = "eval"


# --- small float64 box helpers (binary64, mirroring the frozen core) ----------
def _finite(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number, not bool")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite: {value!r}")
    return result


def box64(box: Sequence[float]) -> tuple[float, float, float, float]:
    if len(box) != 4:
        raise ValueError(f"xyxy box must contain four coordinates: {box!r}")
    x0, y0, x1, y1 = tuple(_finite(v, "box coordinate") for v in box)
    if not (x1 > x0 and y1 > y0):
        raise ValueError(f"box must have positive width and height: {box!r}")
    return x0, y0, x1, y1


def box_area(box: Sequence[float]) -> float:
    x0, y0, x1, y1 = box64(box)
    return (x1 - x0) * (y1 - y0)


def intersect(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float, float] | None:
    ax0, ay0, ax1, ay1 = box64(a)
    bx0, by0, bx1, by1 = box64(b)
    x0, y0 = max(ax0, bx0), max(ay0, by0)
    x1, y1 = min(ax1, bx1), min(ay1, by1)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax0, ay0, ax1, ay1 = box64(a)
    bx0, by0, bx1, by1 = box64(b)
    inter = intersect(a, b)
    ia = 0.0 if inter is None else (inter[2] - inter[0]) * (inter[3] - inter[1])
    aa = (ax1 - ax0) * (ay1 - ay0)
    ab = (bx1 - bx0) * (by1 - by0)
    union = aa + ab - ia
    if not math.isfinite(union) or union <= 0:
        raise ValueError("IoU union is not finite and valid")
    return ia / union


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Vectorised binary64 IoU using the same op order as :func:`box_iou`.

    ``boxes_a`` is ``(n, 4)`` and ``boxes_b`` is ``(m, 4)``; the result is ``(n, m)``.
    The arithmetic is written so that each entry equals :func:`box_iou` bit for bit
    (same order of min/max, products, union and division), which lets the stage-0
    materialiser match GT without paying pure-Python IoU for every candidate pair.
    """
    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    if not a.size or not b.size:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    width = x1 - x0
    height = y1 - y0
    inter = np.maximum(0.0, width) * np.maximum(0.0, height)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    if not (np.isfinite(union).all() and (union > 0).all() and np.isfinite(inter).all()):
        raise ValueError("IoU intersection/union is not finite and valid")
    iou = inter / union
    if not (np.isfinite(iou).all() and (iou >= 0).all() and (iou <= 1).all()):
        raise ValueError("IoU outside [0,1]")
    return iou


def subtract_strips(a: Sequence[float], b: Sequence[float]) -> tuple[tuple[str, tuple], ...]:
    """Exact decomposition of ``a \\ b`` into disjoint strips in the fixed order.

    With ``c = a & b`` the pieces are top ``[ax0, ay0, ax1, cy0]``,
    bottom ``[ax0, cy1, ax1, ay1]``, left ``[ax0, cy0, cx0, cy1]``,
    right ``[cx1, cy0, ax1, cy1]``.  Their union is exactly ``a \\ b`` with pairwise
    disjoint interiors.  Empty pieces are dropped; if ``c`` is empty the single piece
    is ``("full", a)``; if ``b`` covers ``a`` the result is empty.
    """
    ax0, ay0, ax1, ay1 = box64(a)
    c = intersect(a, b)
    if c is None:
        return (("full", (ax0, ay0, ax1, ay1)),)
    cx0, cy0, cx1, cy1 = c
    pieces = (
        ("top", (ax0, ay0, ax1, cy0)),
        ("bottom", (ax0, cy1, ax1, ay1)),
        ("left", (ax0, cy0, cx0, cy1)),
        ("right", (cx1, cy0, ax1, cy1)),
    )
    return tuple((name, box) for name, box in pieces if box[2] > box[0] and box[3] > box[1])


# --- cross-view coordinate mapping -------------------------------------------
@dataclass(frozen=True)
class ViewAffine:
    """Original-image pixels of one view -> that view's feature coordinates.

    ``crop_box`` is the half-open crop rectangle in original-image pixels
    (``(0, 0, W, H)`` for view 0).  ``canvas_hw`` is the resize target shared by all
    views.  ``stride`` is the (sy, sx) of the tapped feature map.
    """

    crop_box: tuple[float, float, float, float]
    canvas_hw: tuple[int, int] = CANVAS_HW
    stride: tuple[int, int] = FEATURE_STRIDE
    view_id: int = -1            # own view of this affine; -1 = unspecified

    def __post_init__(self) -> None:
        x0, y0, x1, y1 = box64(self.crop_box)
        ch, cw = self.canvas_hw
        sy, sx = self.stride
        if not (ch > 0 and cw > 0 and sy > 0 and sx > 0):
            raise ValueError("canvas and stride must be positive")
        if not isinstance(self.view_id, int) or not -1 <= self.view_id <= 4:
            raise ValueError("view_id must be an integer in -1..4")
        object.__setattr__(self, "crop_box", (x0, y0, x1, y1))

    @property
    def crop_size(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.crop_box
        return x1 - x0, y1 - y0

    @property
    def feature_extent(self) -> tuple[float, float]:
        ch, cw = self.canvas_hw
        sy, sx = self.stride
        return math.ceil(ch / sy), math.ceil(cw / sx)

    def to_crop_local(self, box: Sequence[float]) -> tuple[float, float, float, float]:
        """Original-image pixels -> crop-local pixels (no clipping)."""
        x0, y0, x1, y1 = box64(box)
        ox, oy = self.crop_box[0], self.crop_box[1]
        return (x0 - ox, y0 - oy, x1 - ox, y1 - oy)

    def to_crop_normalised(self, box: Sequence[float]) -> tuple[float, float, float, float]:
        """Crop-local pixels -> the normalised cxcywh space the model predicts in."""
        cw, chh = self.crop_size
        x0, y0, x1, y1 = self.to_crop_local(box)
        return ((x0 + x1) / 2 / cw, (y0 + y1) / 2 / chh, (x1 - x0) / cw, (y1 - y0) / chh)

    def to_feature(self, box: Sequence[float]) -> tuple[float, float, float, float]:
        """Original-image pixels -> feature-map coordinates (crop-local, then affine)."""
        cx0, cy0, cx1, cy1 = self.to_crop_local(box)
        cw, chh = self.crop_size
        canv_h, canv_w = self.canvas_hw
        str_y, str_x = self.stride
        return (cx0 * (canv_w / cw) / str_x, cy0 * (canv_h / chh) / str_y,
                cx1 * (canv_w / cw) / str_x, cy1 * (canv_h / chh) / str_y)

    def from_feature(self, box: Sequence[float]) -> tuple[float, float, float, float]:
        """Inverse of :meth:`to_feature`, back to original-image pixels."""
        fx0, fy0, fx1, fy1 = box64(box)
        cw, chh = self.crop_size
        canv_h, canv_w = self.canvas_hw
        str_y, str_x = self.stride
        ox, oy = self.crop_box[0], self.crop_box[1]
        return (fx0 * str_x * cw / canv_w + ox, fy0 * str_y * chh / canv_h + oy,
                fx1 * str_x * cw / canv_w + ox, fy1 * str_y * chh / canv_h + oy)

    def visible_fraction(self, box: Sequence[float]) -> float:
        """Fraction of ``box`` area inside this view's crop (the visibility domain)."""
        c = intersect(box, self.crop_box)
        if c is None:
            return 0.0
        return ((c[2] - c[0]) * (c[3] - c[1])) / box_area(box)


def candidate_affine(candidate: Any, image_hw: tuple[int, int],
                     tiles: Mapping[int, tuple] | None, *,
                     canvas_hw: tuple[int, int] = CANVAS_HW,
                     stride: tuple[int, int] = FEATURE_STRIDE) -> ViewAffine:
    """Affine for the candidate's *own* view: 0 = whole image, 1..4 = its crop."""
    ih, iw = image_hw
    view_id = int(candidate.view_id)
    if view_id == 0:
        return ViewAffine((0.0, 0.0, float(iw), float(ih)), canvas_hw, stride, view_id)
    if not 1 <= view_id <= 4 or not tiles or view_id not in tiles:
        raise ValueError(f"view {view_id} needs its half-open crop box")
    return ViewAffine(tuple(float(v) for v in tiles[view_id]), canvas_hw, stride, view_id)


@dataclass(frozen=True)
class RegionRead:
    """One exclusive-region read (``i\\j`` or ``j\\i``) from its own box's view."""

    strips: tuple[tuple[str, tuple[float, float, float, float]], ...]
    feature_strips: tuple[tuple[str, tuple[float, float, float, float]], ...]
    region_area: float
    visible_area: float
    valid: bool
    reason: str
    source_view: int | None = None   # own view this read was mapped from (rev 4)

    @property
    def visible_fraction(self) -> float:
        return 0.0 if self.region_area <= 0 else self.visible_area / self.region_area


def read_exclusive(region_strips: tuple[tuple[str, tuple], ...],
                   affine: ViewAffine) -> RegionRead:
    """Apply the frozen out-of-bounds policy to one exclusive region.

    The strips of ``region = box_a \\ box_b`` are clipped to the view's crop to obtain
    the visible area, and mapped (then clamped to the feature extent) for reading.  A
    read is valid iff the region has positive area, the visible fraction is at least
    ``THETA_REGION``, and at least one clipped feature strip is non-empty.  There is no
    fallback source: an invalid read makes its pair abstain.
    """
    region_area = 0.0
    visible_area = 0.0
    feature_strips: list[tuple[str, tuple[float, float, float, float]]] = []
    cropped_strips: list[tuple[str, tuple[float, float, float, float]]] = []
    ext_h, ext_w = affine.feature_extent
    for name, box in region_strips:
        region_area += box_area(box)
        clipped = intersect(box, affine.crop_box)
        if clipped is not None:
            visible_area += box_area(clipped)
            fx = affine.to_feature(clipped)
            clamped = (min(max(fx[0], 0.0), ext_w), min(max(fx[1], 0.0), ext_h),
                       min(max(fx[2], 0.0), ext_w), min(max(fx[3], 0.0), ext_h))
            cropped_strips.append((name, clipped))
            if clamped[2] > clamped[0] and clamped[3] > clamped[1]:
                feature_strips.append((name, clamped))
    source_view = affine.view_id if affine.view_id >= 0 else None
    if region_area <= 0:
        return RegionRead(region_strips, tuple(feature_strips), 0.0, 0.0, False,
                          "empty_difference_region", source_view)
    frac = visible_area / region_area
    reason = ("below_visibility_threshold" if frac < THETA_REGION
              else ("empty_feature_region" if not feature_strips else "ok"))
    valid = frac >= THETA_REGION and bool(feature_strips)
    return RegionRead(region_strips, tuple(feature_strips), region_area, visible_area,
                      valid, reason, source_view)


@dataclass(frozen=True)
class PairInput:
    """Frozen pair input: the two exclusive-region reads plus antisymmetric geometry."""

    box_i: tuple[float, float, float, float]
    box_j: tuple[float, float, float, float]
    left: RegionRead          # read of i\\j from view(i)
    right: RegionRead         # read of j\\i from view(j)
    geometry: tuple[float, ...]
    valid: bool
    reason: str
    identity_i: tuple | None = None   # (view, query, category) of box i (rev 4)
    identity_j: tuple | None = None   # (view, query, category) of box j (rev 4)

    @property
    def read_area(self) -> float:
        return self.left.region_area + self.right.region_area


def build_pair_input(box_i, view_i: ViewAffine, box_j, view_j: ViewAffine, *,
                     identity_i: tuple | None = None,
                     identity_j: tuple | None = None) -> PairInput:
    """Both exclusive regions read from their own box's view; valid only if both valid."""
    bi, bj = box64(box_i), box64(box_j)
    left = read_exclusive(subtract_strips(bi, bj), view_i)
    right = read_exclusive(subtract_strips(bj, bi), view_j)
    ax0, ay0, ax1, ay1 = bi
    bx0, by0, bx1, by1 = bj
    wi, hi, wj, hj = ax1 - ax0, ay1 - ay0, bx1 - bx0, by1 - by0
    scale = max(wi, hi, wj, hj)
    geometry = ((ax0 + ax1 - bx0 - bx1) / 2 / scale,
                (ay0 + ay1 - by0 - by1) / 2 / scale,
                math.log(wj / wi), math.log(hj / hi),
                box_iou(bi, bj))
    if not left.valid or not right.valid:
        reason = "left_" + left.reason if not left.valid else "right_" + right.reason
        return PairInput(bi, bj, left, right, geometry, False, reason,
                         identity_i, identity_j)
    return PairInput(bi, bj, left, right, geometry, True, "ok", identity_i, identity_j)


def swap_pair(pair: PairInput) -> PairInput:
    """Mirror a pair: exchange the two roles.  Odd geometry entries negate; IoU does not.

    ``PairComparator`` computes ``r`` as ``sigmoid(w . (g(pair) - g(swap_pair(pair))))``,
    which makes ``r(j, i) == 1 - r(i, j)`` exact by construction, whatever ``g`` is.
    """
    dx, dy, lw, lh, iou = pair.geometry
    return PairInput(pair.box_j, pair.box_i, pair.right, pair.left,
                     (-dx, -dy, -lw, -lh, iou), pair.valid, pair.reason,
                     pair.identity_j, pair.identity_i)


# --- deterministic CPU feature provider (prototype only) ----------------------
class GeometricFeatureProvider:
    """Prototype provider: a deterministic function of the mapped strip rectangles.

    Deliberately *not* a detector feature read.  Stage 1 replaces it with a RoIAlign
    provider over a bound feature-map tap; the call signature is identical.
    """

    def __init__(self, dim: int = 6) -> None:
        self.dim = dim

    def __call__(self, read: RegionRead | None) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float64)
        if read is None or not read.feature_strips:
            return vec
        ext_w = max(b[2] for _, b in read.feature_strips) or 1.0
        ext_h = max(b[3] for _, b in read.feature_strips) or 1.0
        acc = np.zeros(4, dtype=np.float64)
        total = 0.0
        for _, (x0, y0, x1, y1) in read.feature_strips:
            w, h = x1 - x0, y1 - y0
            acc += np.array([x0 + x1, y0 + y1, w, h], dtype=np.float64)
            total += w * h
        acc /= len(read.feature_strips)
        vec[:4] = acc
        vec[4] = total / (ext_w * ext_h)
        vec[5] = len(read.feature_strips) / 4.0
        return vec


class PairComparator:
    """Shared-encoder pairwise comparator: preference r (antisymmetric) + abstention a.

    ``r(i, j) = sigmoid(w . (g(i, j) - g(j, i)))`` is antisymmetric by construction, so
    ``r(j, i) == 1 - r(i, j)`` exactly, for any encoder ``g``.  ``a`` reads a symmetric
    pair descriptor, so ``a(i, j) == a(j, i)`` exactly.  Weights are seeded here and are
    never trained; the stage-1 comparator keeps the same interface.
    """

    def __init__(self, provider: Callable[[RegionRead | None], np.ndarray], *,
                 seed: int = 20261030) -> None:
        self.provider = provider
        probe = provider(None)
        self._dim = int(probe.shape[0])
        rng = np.random.default_rng(seed)
        self._w_pref = rng.normal(0.0, 0.5, size=2 * self._dim + 5)
        self._w_abstain = rng.normal(0.0, 0.5, size=2 * self._dim + 1)
        self._bias_pref = 0.0
        self._bias_abstain = 1.5      # prior toward "same target": low abstention
        self.seed = seed

    def g(self, pair: PairInput) -> np.ndarray:
        return np.concatenate([self.provider(pair.left), self.provider(pair.right),
                               np.asarray(pair.geometry, dtype=np.float64)])

    def h(self, pair: PairInput) -> np.ndarray:
        fa, fb = self.provider(pair.left), self.provider(pair.right)
        return np.concatenate([fa + fb, np.abs(fa - fb), [pair.geometry[-1]]])

    def r(self, pair: PairInput) -> float:
        delta = self.g(pair) - self.g(swap_pair(pair))
        return _sigmoid(float(self._w_pref @ delta) + self._bias_pref)

    def a(self, pair: PairInput) -> float:
        return _sigmoid(float(self._w_abstain @ self.h(pair)) + self._bias_abstain)


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


class ScriptedComparator:
    """Deterministic comparator returning preset values, for decision-layer tests."""

    def __init__(self, preference: Callable[[PairInput], float] | float = 0.5,
                 abstention: Callable[[PairInput], float] | float = 0.0) -> None:
        self._pref = preference
        self._abst = abstention

    def r(self, pair: PairInput) -> float:
        return float(self._pref(pair)) if callable(self._pref) else float(self._pref)

    def a(self, pair: PairInput) -> float:
        return float(self._abst(pair)) if callable(self._abst) else float(self._abst)


# --- decision layer -----------------------------------------------------------
@dataclass(frozen=True)
class PairVerdict:
    winner: int      # +1 => i, -1 => j, 0 => abstain
    r: float
    abstain_prob: float
    reason: str


def decide_pair(pair: PairInput, r_ij: float, a_ij: float) -> PairVerdict:
    """Frozen pair decision.  Abstention never promotes anything; it withholds a vote."""
    if not pair.valid:
        return PairVerdict(0, r_ij, a_ij, "infeasible_read")
    if a_ij >= DELTA_ABSTAIN:
        return PairVerdict(0, r_ij, a_ij, "abstain_head")
    if r_ij >= 0.5 + DELTA_PREF:
        return PairVerdict(1, r_ij, a_ij, "pref_i")
    if r_ij <= 0.5 - DELTA_PREF:
        return PairVerdict(-1, r_ij, a_ij, "pref_j")
    return PairVerdict(0, r_ij, a_ij, "dead_zone")


def sort_key_of(candidate: Any) -> tuple:
    """The frozen merge iteration key: score desc, then view/query/category ascending."""
    return (-float(candidate.score), int(candidate.view_id),
            int(candidate.query_id), int(candidate.category_id))


def canonical_order(candidates: Sequence[Any]) -> list[Any]:
    return sorted(candidates, key=sort_key_of)


def _identity(candidate: Any) -> tuple[int, int, int]:
    """``(view_id, query_id, category_id)``; works for frozen and minimal candidates."""
    identity = getattr(candidate, "identity", None)
    if identity is not None:
        return tuple(identity)
    return (int(candidate.view_id), int(candidate.query_id), int(candidate.category_id))


# --- clustering: the frozen core's own arithmetic, with cluster bookkeeping -----
def _frozen_iou_rows(boxes: np.ndarray, areas: np.ndarray,
                     first: int, rest: np.ndarray) -> np.ndarray:
    left = np.maximum(boxes[first, :2], boxes[rest, :2])
    right = np.minimum(boxes[first, 2:], boxes[rest, 2:])
    extent = np.maximum(0.0, right - left)
    intersection = extent[:, 0] * extent[:, 1]
    union = areas[first] + areas[rest] - intersection
    if not (np.isfinite(intersection).all() and np.isfinite(union).all()
            and (union > 0).all()):
        raise ValueError("NMS intersection/union is not finite and valid")
    iou = intersection / union
    if not (np.isfinite(iou).all() and (iou >= 0).all() and (iou <= 1).all()):
        raise ValueError("NMS IoU outside [0,1]")
    return iou


def nms_clusters_scalar(candidates: Sequence[Any], *, iou_threshold: float = NMS_IOU) -> list[dict]:
    """Scalar cluster builder (binary64 via :func:`box_iou`); kept for cross-checks."""
    ordered = canonical_order(candidates)
    by_class: dict[int, list[int]] = {}
    for index, candidate in enumerate(ordered):
        by_class.setdefault(int(candidate.category_id), []).append(index)
    clusters: list[dict] = []
    for _, indices in by_class.items():
        alive = list(indices)
        while alive:
            survivor = ordered[alive[0]]
            suppressed, keep = [], []
            for idx in alive[1:]:
                if box_iou(survivor.box_xyxy, ordered[idx].box_xyxy) > iou_threshold:
                    suppressed.append(ordered[idx])
                else:
                    keep.append(idx)
            clusters.append({"survivor": survivor, "suppressed": suppressed})
            alive = keep
    position = {id(c): i for i, c in enumerate(ordered)}
    clusters.sort(key=lambda cl: position[id(cl["survivor"])])
    return clusters


def nms_clusters(candidates: Sequence[Any], *, iou_threshold: float = NMS_IOU,
                 with_stats: bool = False) -> list[dict] | tuple[list[dict], dict]:
    """Classwise hard NMS mirroring the frozen core, additionally exposing clusters.

    The arithmetic replicates ``slice_core.classwise_nms`` step by step in numpy
    float64 (same min/max order, same ``IoU > threshold`` suppression, same iteration
    key), so the survivor sequence equals the frozen core's kept list.  Each cluster is
    ``{"survivor": c, "suppressed": [...]}`` in frozen order; clusters are returned in
    the survivor's canonical position.
    """
    ordered = canonical_order(candidates)
    identities = [_identity(c) for c in ordered]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate candidate identity")
    by_class: dict[int, list[int]] = {}
    for index, candidate in enumerate(ordered):
        by_class.setdefault(int(candidate.category_id), []).append(index)
    clusters: list[dict] = []
    comparisons = boundary = 0
    for _, indices in by_class.items():
        boxes = np.asarray([ordered[i].box_xyxy for i in indices], dtype=np.float64)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        remaining = np.arange(len(indices), dtype=np.int64)
        while remaining.size:
            first = int(remaining[0])
            survivor = ordered[indices[first]]
            rest = remaining[1:]
            suppressed: list[Any] = []
            if rest.size:
                iou = _frozen_iou_rows(boxes, areas, first, rest)
                comparisons += int(rest.size)
                boundary += int(np.sum(np.abs(iou - iou_threshold) <= 1e-9))
                removed = rest[iou > iou_threshold]
                suppressed = [ordered[indices[int(k)]] for k in removed.tolist()]
                remaining = rest[iou <= iou_threshold]
            else:
                remaining = rest
            clusters.append({"survivor": survivor, "suppressed": suppressed})
    clustered = sum(1 + len(cl["suppressed"]) for cl in clusters)
    if clustered != len(ordered):
        raise ValueError("cluster partition incomplete")
    position = {id(c): i for i, c in enumerate(ordered)}
    clusters.sort(key=lambda cl: position[id(cl["survivor"])])
    if with_stats:
        return clusters, {"comparisons": comparisons, "boundary_within_eps": boundary,
                          "boundary_eps": 1e-9, "nms_iou": iou_threshold,
                          "cluster_count": len(clusters)}
    return clusters


def cluster_members(cluster: Mapping) -> list[Any]:
    """``{survivor} U suppressed`` in frozen order (survivor first)."""
    return [cluster["survivor"], *cluster["suppressed"]]


def cluster_pairs(cluster: Mapping) -> list[tuple[Any, Any]]:
    """The frozen pairing domain *inside one cluster*: unordered member pairs.

    Member order is the frozen member order, so the pair list is canonical and the
    same list drives the inference tournament and the stage-0 training labels.
    """
    members = cluster_members(cluster)
    return [(members[i], members[j])
            for i in range(len(members)) for j in range(i + 1, len(members))]


def cluster_pair_records(candidates: Sequence[Any], *,
                         iou_threshold: float = NMS_IOU) -> list[dict]:
    """Enumerate every pair the comparator may judge, in cluster-canonical order.

    This is the single enumeration used by training-material prep and by inference; it
    replaces the rev-2 split where training used a same-class ``IoU>=0.5`` pool filter.
    """
    records = []
    for cluster_index, cluster in enumerate(nms_clusters(candidates, iou_threshold=iou_threshold)):
        for i, j in cluster_pairs(cluster):
            records.append({"cluster_index": cluster_index, "box_i": i, "box_j": j})
    return records


def copeland_votes(members: Sequence[Any],
                   affine_of: Callable[[Any], ViewAffine],
                   comparator) -> tuple[dict, dict]:
    """One Copeland vote per decided unordered pair; abstentions cast no vote.

    The pairs visited are exactly :func:`cluster_pairs` of the member list.
    """
    votes = {id(m): 0 for m in members}
    stats = {"decided": 0, "abstained": 0, "infeasible": 0, "pairs": 0}
    for a in range(len(members)):
        for b in range(a + 1, len(members)):
            mi, mj = members[a], members[b]
            stats["pairs"] += 1
            pair = build_pair_input(mi.box_xyxy, affine_of(mi), mj.box_xyxy, affine_of(mj),
                                    identity_i=_identity(mi), identity_j=_identity(mj))
            verdict = decide_pair(pair, comparator.r(pair), comparator.a(pair))
            if verdict.winner == 0:
                stats["abstained" if pair.valid else "infeasible"] += 1
                continue
            stats["decided"] += 1
            votes[id(mi if verdict.winner > 0 else mj)] += 1
    return votes, stats


def select_representative(members: Sequence[Any],
                          affine_of: Callable[[Any], ViewAffine], comparator) -> dict:
    """Pick the cluster representative by Copeland count, then the frozen tie rules.

    Cycles are handled by construction: Copeland counts wins rather than following a
    transitive chain, so a 3-cycle yields equal counts and falls through to the tie
    rules.  Tie order: votes desc, original score desc, view/query/category asc.
    """
    votes, stats = copeland_votes(members, affine_of, comparator)
    ranked = sorted(members, key=lambda m: (-votes[id(m)],) + sort_key_of(m))
    return {"representative": ranked[0], "votes": {id(m): votes[id(m)] for m in members},
            "stats": stats}


def pdra_merge(candidates: Sequence[Any],
               affine_of: Callable[[Any], ViewAffine], comparator,
               *, budgets: tuple[int, int] = FINAL_BUDGETS) -> dict:
    """Replace each NMS cluster's representative; then re-rank and truncate.

    Cluster decomposition, suppression topology, raw-ignore and the per-class
    ``maxDets`` handling stay the original C's.  What changes is *which* member of a
    cluster represents it, and therefore the score that ranks it.

    rev-3 correction: because a substituted representative carries a *different* score
    than the survivor it replaced, the representative list is re-sorted with the frozen
    sort key and only then truncated to the budget (no refill).  rev 2 truncated the
    list in cluster order, which kept the survivor's rank.  With no substitution the
    re-ranked list equals the survivor list, so an all-abstain run is still identical
    to C.
    """
    clusters = nms_clusters(candidates)
    c_kept = [cluster["survivor"] for cluster in clusters]
    kept: list[Any] = []
    substitutions = 0
    all_abstain_clusters = 0
    for cluster in clusters:
        members = cluster_members(cluster)
        if len(members) == 1:
            kept.append(cluster["survivor"])
            continue
        chosen = select_representative(members, affine_of, comparator)
        representative = chosen["representative"]
        if chosen["stats"]["decided"] == 0:
            all_abstain_clusters += 1
        if representative is not cluster["survivor"]:
            substitutions += 1
        kept.append(representative)
    ranked = canonical_order(kept)
    c_ranked = canonical_order(c_kept)
    rank_shifted = sum(1 for a, b in zip(ranked, c_ranked) if a is not b)
    budget_delta = {}
    for budget in budgets:
        before, after = set(map(id, c_ranked[:budget])), set(map(id, ranked[:budget]))
        budget_delta[str(budget)] = {"entered": len(after - before), "left": len(before - after)}
    return {
        "core_version": CORE_VERSION, "nms_iou": NMS_IOU,
        "cluster_count": len(clusters), "kept_count": len(kept),
        "substitutions": substitutions, "all_abstain_clusters": all_abstain_clusters,
        "representative_ranking": POST_REPLACEMENT_RANKING,
        "rank_shifted_members": rank_shifted,
        "budget_membership_delta_vs_C": budget_delta,
        "top600": ranked[:budgets[0]], "top300": ranked[:budgets[1]],
    }


def pair_decisions(candidates: Sequence[Any],
                   affine_of: Callable[[Any], ViewAffine], comparator) -> list[dict]:
    """Decide every cluster-internal pair of one image; ground truth is never read.

    The enumeration is exactly :func:`cluster_pair_records` -- the same domain used
    for training -- and each verdict depends only on the two boxes, their own-view
    affines and the comparator outputs.  Supervision labels and their exclusions are
    absent from this code path by construction (no GT argument exists), so label
    routing can never narrow the inference domain.  This is the stage-2 inference
    enumeration and the enumeration exercised by the GPU engineering check.
    """
    decisions = []
    for record in cluster_pair_records(candidates):
        bi, bj = record["box_i"], record["box_j"]
        pair = build_pair_input(bi.box_xyxy, affine_of(bi), bj.box_xyxy, affine_of(bj),
                                identity_i=_identity(bi), identity_j=_identity(bj))
        r_ij = comparator.r(pair)
        a_ij = comparator.a(pair)
        verdict = decide_pair(pair, r_ij, a_ij)
        decisions.append({"cluster_index": record["cluster_index"],
                          "identity_i": pair.identity_i, "identity_j": pair.identity_j,
                          "valid": pair.valid, "read_reason": pair.reason,
                          "winner": verdict.winner, "decision_reason": verdict.reason,
                          "r": r_ij, "a": a_ij})
    return decisions


# --- supervision labelling (three-way, with ambiguity routing) ----------------
@dataclass(frozen=True)
class GTItem:
    gt_id: int
    category_id: int
    box_xyxy: tuple[float, float, float, float]
    area: float = 0.0


def match_gts_iou(box: Sequence[float], category_id: int,
                  gts: Sequence[GTItem]) -> tuple[tuple[float, GTItem], ...]:
    """Same-category GTs matched at ``IoU >= MATCH_IOU``, best first (IoU desc, id asc)."""
    hits = [(box_iou(box, g.box_xyxy), g) for g in gts if g.category_id == category_id]
    hits = [(iou, g) for iou, g in hits if iou >= MATCH_IOU]
    hits.sort(key=lambda item: (-item[0], item[1].gt_id))
    return tuple(hits)


def match_gts(box: Sequence[float], category_id: int,
              gts: Sequence[GTItem]) -> tuple[GTItem, ...]:
    """Backwards-compatible wrapper returning only the matched GTs."""
    return tuple(g for _, g in match_gts_iou(box, category_id, gts))


def label_pair(box_i, box_j, category_id: int, gts: Sequence[GTItem]) -> dict:
    """Route one candidate pair to exactly one supervision destination.

    Destinations
    ------------
    ``pref``               same shared GT, unique sign, ``|dIoU| >= TAU_GAP``
    ``abstain``            disjoint GT sets, both non-empty
    ``excluded_empty``     at least one box matches no GT
    ``excluded_gap``       same GT but ``|dIoU| < TAU_GAP``
    ``excluded_ambiguous`` conflicting GT structure or a GT-match tie (see below)

    Ambiguity rule (hard): a pair whose two boxes *share* a GT and *also* each match a
    distinct additional GT is routed to ``excluded_ambiguous`` and enters **neither**
    the same-GT preference loss nor the cross-GT abstention loss.  Its would-be
    destinations and GT ids are recorded.  A box matching several GTs is also flagged
    ``multigt`` even when the pair is not excluded.
    """
    return route_pair(match_gts_iou(box_i, category_id, gts),
                      match_gts_iou(box_j, category_id, gts))


def route_pair(mi: Sequence[tuple[float, GTItem]],
               mj: Sequence[tuple[float, GTItem]]) -> dict:
    """:func:`label_pair` on precomputed matches (same IoUs, same routing)."""
    iou_i = {g.gt_id: iou for iou, g in mi}
    iou_j = {g.gt_id: iou for iou, g in mj}
    record = {"destination": None, "ambiguous": False, "would_be": [], "gt_ids": [],
              "multigt": (len(mi) > 1 or len(mj) > 1)}
    if not mi or not mj:
        record["destination"] = "excluded_empty"
        record["gt_ids"] = [g.gt_id for _, g in mi] + [g.gt_id for _, g in mj]
        return record
    shared = [(iou_i[g.gt_id], g) for _, g in mi if g.gt_id in iou_j]
    extras_i = [g for _, g in mi if g.gt_id not in iou_j]
    extras_j = [g for _, g in mj if g.gt_id not in iou_i]
    if shared and extras_i and extras_j:
        record.update(destination="excluded_ambiguous", ambiguous=True,
                      would_be=["pref", "abstain"],
                      gt_ids=[g.gt_id for _, g in shared] + [g.gt_id for g in extras_i]
                             + [g.gt_id for g in extras_j])
        return record
    for side in (mi, mj):
        if len(side) > 1 and abs(side[0][0] - side[1][0]) <= 1e-9:
            record.update(destination="excluded_ambiguous", ambiguous=True,
                          would_be=[], gt_ids=[g.gt_id for _, g in side])
            return record
    if shared:
        signs = {}
        for _, g in shared:
            delta = iou_i[g.gt_id] - iou_j[g.gt_id]
            if abs(delta) >= TAU_GAP:
                signs[g.gt_id] = 1 if delta > 0 else -1
        if 1 in signs.values() and -1 in signs.values():
            record.update(destination="excluded_ambiguous", ambiguous=True,
                          would_be=["pref"], gt_ids=[g.gt_id for _, g in shared])
            return record
        g_star = sorted((g for _, g in shared),
                        key=lambda g: (-min(iou_i[g.gt_id], iou_j[g.gt_id]), g.gt_id))[0]
        delta = iou_i[g_star.gt_id] - iou_j[g_star.gt_id]
        record["iou_i"] = iou_i[g_star.gt_id]
        record["iou_j"] = iou_j[g_star.gt_id]
        record["delta_iou"] = delta
        record["gt_area"] = g_star.area
        if abs(delta) < TAU_GAP:
            record.update(destination="excluded_gap", gt_ids=[g_star.gt_id])
            return record
        record.update(destination="pref", gt_ids=[g_star.gt_id])
        record["label"] = 1 if delta > 0 else 0
        record["iou_gap"] = abs(delta)
        return record
    record.update(destination="abstain",
                  gt_ids=[g.gt_id for _, g in mi] + [g.gt_id for _, g in mj])
    record["label"] = 1
    return record


def aggregate_labels(records: Iterable[Mapping]) -> dict:
    """Counts per destination plus the ambiguity ledger (count and would-be destination)."""
    counts = {d: 0 for d in PAIR_DESTINATIONS}
    ambiguous = 0
    ambiguous_would_be: dict[str, int] = {}
    multigt = 0
    total = 0
    for rec in records:
        total += 1
        counts[rec["destination"]] += 1
        if rec.get("multigt"):
            multigt += 1
        if rec.get("ambiguous"):
            ambiguous += 1
            for dest in (rec.get("would_be") or ["<none>"]):
                ambiguous_would_be[dest] = ambiguous_would_be.get(dest, 0) + 1
    return {"total_pairs": total, "destination_counts": counts,
            "ambiguous_pairs": ambiguous, "ambiguous_would_be": ambiguous_would_be,
            "multigt_pairs": multigt}


# --- frozen diagnostic buckets and the frozen sequence split -------------------
def delta_bucket_of(abs_delta: float) -> str:
    """|dIoU| -> frozen bucket name (``gap_ge_0.50`` is the open top bucket)."""
    edges = GAP_EDGES
    for index in range(len(edges) - 1):
        if edges[index] <= abs_delta < edges[index + 1]:
            return DIOU_BUCKETS[index]
    if abs_delta >= edges[-1]:
        return DIOU_BUCKETS[-1]
    return DIOU_BUCKETS[0]


def area_bucket_of(area: float) -> str:
    if area <= AREA_EDGES[0]:
        return AREA_BUCKETS[0]
    if area <= AREA_EDGES[1]:
        return AREA_BUCKETS[1]
    return AREA_BUCKETS[2]


def pair_buckets(record: Mapping) -> dict:
    """Frozen diagnostic keys of one labelled pair (``None`` where undefined)."""
    delta = record.get("delta_iou")
    area = record.get("gt_area")
    return {"diou": None if delta is None else delta_bucket_of(abs(delta)),
            "area": None if area is None else area_bucket_of(float(area)),
            "view": None if record.get("same_view") is None else (
                "same_view" if record["same_view"] else "cross_view")}


def freeze_sequence_split(sequence_images: Mapping[str, Sequence[Any]]) -> dict:
    """Deterministic module train/eval sequence partition (frozen rule).

    ``sequence_images`` maps a sequence key to its image identifiers.  Keys are sorted
    ascending; the key at sorted index ``i`` joins the eval split iff ``i % 3 == 2``.
    The rule uses no randomness and no per-image choice, so no sequence can leak
    between the two splits.
    """
    keys = sorted(sequence_images)
    assignment = {}
    for index, key in enumerate(keys):
        assignment[key] = "eval" if index % 3 == 2 else "train"
    train = [k for k in keys if assignment[k] == "train"]
    evaluation = [k for k in keys if assignment[k] == "eval"]
    return {"rule": SPLIT_RULE, "rule_note": SPLIT_RULE_NOTE,
            "sequence_count": len(keys), "sequences": keys, "assignment": assignment,
            "train_sequences": train, "eval_sequences": evaluation,
            "train_images": sum(len(sequence_images[k]) for k in train),
            "eval_images": sum(len(sequence_images[k]) for k in evaluation),
            "main_verdict_split": MAIN_VERDICT_SPLIT}


def frozen_hyperparameters() -> dict:
    """Every frozen numeric rule, for the stage-0 hyperparameter freeze file."""
    return {
        "design_revision": DESIGN_REVISION, "core_version": CORE_VERSION,
        "canvas_hw": list(CANVAS_HW), "feature_stride": list(FEATURE_STRIDE),
        "feature_level": FEATURE_LEVEL, "feature_read": FEATURE_READ,
        "feature_binding_verified": FEATURE_BINDING_VERIFIED,
        "nms_iou": NMS_IOU, "per_view_budget": PER_VIEW_BUDGET,
        "final_budgets": list(FINAL_BUDGETS), "sample_grid": SAMPLE_GRID,
        "delta_pref": DELTA_PREF, "delta_abstain": DELTA_ABSTAIN,
        "theta_region": THETA_REGION, "match_iou": MATCH_IOU, "tau_gap": TAU_GAP,
        "tau_clear": TAU_CLEAR, "lambda_abstain": LAMBDA_ABSTAIN,
        "strip_order": list(STRIP_ORDER), "pair_domain": PAIR_DOMAIN,
        "post_replacement_ranking": POST_REPLACEMENT_RANKING,
        "cluster_builder": CLUSTER_BUILDER,
        "diou_buckets": list(DIOU_BUCKETS), "area_buckets": list(AREA_BUCKETS),
        "gap_edges": list(GAP_EDGES), "area_edges": list(AREA_EDGES),
        "n_min_pairs": N_MIN_PAIRS, "n_min_images": N_MIN_IMAGES,
        "split_rule": SPLIT_RULE, "main_verdict_split": MAIN_VERDICT_SPLIT,
        "destinations": list(PAIR_DESTINATIONS),
        "abstain_semantics": "abstention withholds a vote; it preserves nothing across instances",
    }


# --- control specifications (declared, not trained here) ----------------------
@dataclass(frozen=True)
class ControlSpec:
    """One of the three mandatory controls.  (a) is the original C and needs no module."""

    key: str
    reads: str          # "score" | "whole_box_quality" | "whole_box_pairwise"
    note: str

    def read_area(self, box_i, box_j) -> float:
        """Region area this control consumes; used to assert (c) covers PDRA's domain."""
        if self.reads == "whole_box_pairwise":
            return box_area(box_i) + box_area(box_j)
        return 0.0


CONTROL_SCORE = ControlSpec("a_original_c", "score",
                            "original C: highest score, original NMS")
CONTROL_SINGLE_QUALITY = ControlSpec(
    "b_single_box_quality", "whole_box_quality",
    "IoU-Net-style independent per-box quality head; same features/budget; sign(q_i - q_j)")
CONTROL_WHOLE_BOX_PAIRWISE = ControlSpec(
    "c_whole_box_pairwise", "whole_box_pairwise",
    "same comparator reading whole boxes (a superset of the difference strips)")


def region_coverage_margin(pairs: Sequence[PairInput]) -> dict:
    """Geometric *coverage* property: control (c)'s read region contains PDRA's.

    This is a fact about region areas, not about experimental fairness.  Fairness is
    asserted separately by :func:`assert_control_fairness` on the arm specifications.
    """
    checks = {"pairs_checked": len(pairs), "c_covers_pdra_domain": True,
              "every_pair_uses_one_source_rule": True, "min_margin": None}
    for pair in pairs:
        pdra_area = pair.read_area
        c_area = CONTROL_WHOLE_BOX_PAIRWISE.read_area(pair.box_i, pair.box_j)
        if c_area + 1e-9 < pdra_area:
            checks["c_covers_pdra_domain"] = False
        margin = c_area - pdra_area
        checks["min_margin"] = margin if checks["min_margin"] is None else min(
            checks["min_margin"], margin)
    return checks


# --- control fairness on arm specifications (rev 3) --------------------------
#: Fields every arm must share for the comparison to be attributable to the read.
FAIRNESS_EQUAL_FIELDS = (
    "candidate_pool_sha256", "pair_domain", "feature_source_rule", "feature_level",
    "feature_stride", "canvas_hw", "theta_region", "sample_grid", "split_sha256",
    "sampling_spec_sha256", "parameter_cap", "update_cap", "eval_members_sha256",
)
#: The one field allowed to differ between arms.
FAIRNESS_FREE_FIELD = "reads"


@dataclass(frozen=True)
class ArmSpec:
    """Declared specification of one experimental arm.

    Only ``reads`` may differ between arms (``"difference_strips"`` for PDRA,
    ``"whole_box_pairwise"`` for (c), ``"whole_box_quality"`` for (b), ``"score"`` for
    (a)); every other field is a fairness invariant and must be identical.
    """

    key: str
    reads: str
    candidate_pool_sha256: str
    pair_domain: str = PAIR_DOMAIN
    feature_source_rule: str = "each box read from its own view via candidate_affine"
    feature_level: str = FEATURE_LEVEL
    feature_stride: tuple = FEATURE_STRIDE
    canvas_hw: tuple = CANVAS_HW
    theta_region: float = THETA_REGION
    sample_grid: int = SAMPLE_GRID
    split_sha256: str = "PENDING_STAGE0"
    sampling_spec_sha256: str = "PENDING_STAGE0"
    parameter_cap: int = 1_000_000
    update_cap: int = 20_000
    eval_members_sha256: str = "PENDING_STAGE0"


def assert_control_fairness(arms: Sequence[ArmSpec]) -> dict:
    """Assert the four arms differ only in what they read.

    Returns the shared specification, the per-arm read rule and any violation of a
    fairness invariant (empty list means the comparison is fair by declaration).
    """
    if len(arms) < 2:
        raise ValueError("control fairness needs at least two arms")
    violations: list[str] = []
    reads = {}
    seen_keys = set()
    for arm in arms:
        if arm.key in seen_keys:
            raise ValueError(f"duplicate arm key: {arm.key}")
        seen_keys.add(arm.key)
        if arm.reads in reads.values():
            violations.append(f"two arms share the read rule {arm.reads!r}")
        reads[arm.key] = arm.reads
    reference = arms[0]
    for arm in arms[1:]:
        for field in FAIRNESS_EQUAL_FIELDS:
            if getattr(arm, field) != getattr(reference, field):
                violations.append(
                    f"{arm.key}:{field} differs from {reference.key}: "
                    f"{getattr(arm, field)!r} != {getattr(reference, field)!r}")
    return {"arms": [arm.key for arm in arms], "reads": reads,
            "equal_fields": list(FAIRNESS_EQUAL_FIELDS),
            "free_field": FAIRNESS_FREE_FIELD,
            "violations": violations, "fair": not violations}
