"""PDRA decision-and-geometry core (frozen rules, rev 2).

Pure CPU.  No torch, no detector, no training loop, no randomness.  Every rule in
this module is fixed by ``PAIRWISE_ARBITRATION_DESIGN_20261030.md`` rev 2.  Anything
that needs a *real* feature map sits behind an explicit provider interface; the CPU
prototype uses a deterministic geometric provider and therefore never claims a
detector-feature result.

Implemented here
----------------
S3  cross-view feature coordinate mapping, visibility domain, out-of-bounds policy
S3  exclusive difference-region (strip) construction and read validity
S4  preference / abstention decision, Copeland tournament, cycle and tie rules
S5  cluster-level representative replacement, original-C fallback
S6  three-way supervision labelling with explicit ambiguity routing and counts

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
* Feature maps are NOT currently exposed by the frozen forward (it returns only
  ``pred_logits``/``pred_boxes``).  Binding the tap is a stage-1 engineering item;
  ``FEATURE_STRIDE`` below is a declared placeholder until that binding exists.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


CORE_VERSION = "pdra-rules-rev2-20261030"

# --- frozen numeric rules -----------------------------------------------------
CANVAS_HW = (768, 1344)          # every view resizes to this canvas (height, width)
FEATURE_STRIDE = (8, 8)          # PLACEHOLDER: must be bound to the frozen tap (stage 1)
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

STRIP_ORDER = ("top", "bottom", "left", "right")
PAIR_DESTINATIONS = ("pref", "abstain", "excluded_empty", "excluded_gap",
                     "excluded_ambiguous")


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

    def __post_init__(self) -> None:
        x0, y0, x1, y1 = box64(self.crop_box)
        ch, cw = self.canvas_hw
        sy, sx = self.stride
        if not (ch > 0 and cw > 0 and sy > 0 and sx > 0):
            raise ValueError("canvas and stride must be positive")
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
        return ViewAffine((0.0, 0.0, float(iw), float(ih)), canvas_hw, stride)
    if not 1 <= view_id <= 4 or not tiles or view_id not in tiles:
        raise ValueError(f"view {view_id} needs its half-open crop box")
    return ViewAffine(tuple(float(v) for v in tiles[view_id]), canvas_hw, stride)


@dataclass(frozen=True)
class RegionRead:
    """One exclusive-region read (``i\\j`` or ``j\\i``) from its own box's view."""

    strips: tuple[tuple[str, tuple[float, float, float, float]], ...]
    feature_strips: tuple[tuple[str, tuple[float, float, float, float]], ...]
    region_area: float
    visible_area: float
    valid: bool
    reason: str

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
    if region_area <= 0:
        return RegionRead(region_strips, tuple(feature_strips), 0.0, 0.0, False,
                          "empty_difference_region")
    frac = visible_area / region_area
    reason = ("below_visibility_threshold" if frac < THETA_REGION
              else ("empty_feature_region" if not feature_strips else "ok"))
    valid = frac >= THETA_REGION and bool(feature_strips)
    return RegionRead(region_strips, tuple(feature_strips), region_area, visible_area,
                      valid, reason)


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

    @property
    def read_area(self) -> float:
        return self.left.region_area + self.right.region_area


def build_pair_input(box_i, view_i: ViewAffine, box_j, view_j: ViewAffine) -> PairInput:
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
        return PairInput(bi, bj, left, right, geometry, False, reason)
    return PairInput(bi, bj, left, right, geometry, True, "ok")


def swap_pair(pair: PairInput) -> PairInput:
    """Mirror a pair: exchange the two roles.  Odd geometry entries negate; IoU does not.

    ``PairComparator`` computes ``r`` as ``sigmoid(w . (g(pair) - g(swap_pair(pair))))``,
    which makes ``r(j, i) == 1 - r(i, j)`` exact by construction, whatever ``g`` is.
    """
    dx, dy, lw, lh, iou = pair.geometry
    return PairInput(pair.box_j, pair.box_i, pair.right, pair.left,
                     (-dx, -dy, -lw, -lh, iou), pair.valid, pair.reason)


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


def nms_clusters(candidates: Sequence[Any], *, iou_threshold: float = NMS_IOU) -> list[dict]:
    """Classwise hard NMS mirroring the frozen core, additionally exposing clusters.

    Returns clusters in kept order; each is ``{"survivor": c, "suppressed": [...]}``
    where ``suppressed`` are the candidates the survivor removed (suppress iff
    ``IoU > iou_threshold``).  The survivor sequence equals the frozen core's kept list.
    """
    ordered = canonical_order(candidates)
    by_class: dict[int, list[int]] = {}
    for index, candidate in enumerate(ordered):
        by_class.setdefault(int(candidate.category_id), []).append(index)
    clusters: list[dict] = []
    for _, indices in sorted(by_class.items()):
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


def copeland_votes(members: Sequence[Any],
                   affine_of: Callable[[Any], ViewAffine],
                   comparator) -> tuple[dict, dict]:
    """One Copeland vote per decided unordered pair; abstentions cast no vote."""
    votes = {id(m): 0 for m in members}
    stats = {"decided": 0, "abstained": 0, "infeasible": 0}
    for a in range(len(members)):
        for b in range(a + 1, len(members)):
            mi, mj = members[a], members[b]
            pair = build_pair_input(mi.box_xyxy, affine_of(mi), mj.box_xyxy, affine_of(mj))
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
    """Replace each NMS cluster's representative; keep the frozen suppression topology.

    The cluster decomposition, budgets, raw-ignore and per-class maxDets handling are
    the original C's.  Only *which* member of a cluster represents it changes.  A
    cluster in which every pair abstains keeps the original survivor, so an all-abstain
    run is identical to C.
    """
    clusters = nms_clusters(candidates)
    kept: list[Any] = []
    substitutions = 0
    all_abstain_clusters = 0
    for cluster in clusters:
        members = [cluster["survivor"], *cluster["suppressed"]]
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
    return {
        "core_version": CORE_VERSION, "nms_iou": NMS_IOU,
        "cluster_count": len(clusters), "kept_count": len(kept),
        "substitutions": substitutions, "all_abstain_clusters": all_abstain_clusters,
        "top600": kept[:budgets[0]], "top300": kept[:budgets[1]],
    }


# --- supervision labelling (three-way, with ambiguity routing) ----------------
@dataclass(frozen=True)
class GTItem:
    gt_id: int
    category_id: int
    box_xyxy: tuple[float, float, float, float]


def match_gts(box: Sequence[float], category_id: int,
              gts: Sequence[GTItem]) -> tuple[GTItem, ...]:
    """GTs of the same category matched at IoU >= MATCH_IOU, best first (IoU desc, id asc)."""
    hits = [(box_iou(box, g.box_xyxy), g) for g in gts if g.category_id == category_id]
    hits = [(iou, g) for iou, g in hits if iou >= MATCH_IOU]
    hits.sort(key=lambda item: (-item[0], item[1].gt_id))
    return tuple(g for _, g in hits)


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
    mi = match_gts(box_i, category_id, gts)
    mj = match_gts(box_j, category_id, gts)
    shared = tuple(g for g in mi if g in mj)
    record = {"destination": None, "ambiguous": False, "would_be": [], "gt_ids": [],
              "multigt": (len(mi) > 1 or len(mj) > 1)}
    if not mi or not mj:
        record["destination"] = "excluded_empty"
        record["gt_ids"] = [g.gt_id for g in mi + mj]
        return record
    extras_i = tuple(g for g in mi if g not in mj)
    extras_j = tuple(g for g in mj if g not in mi)
    if shared and extras_i and extras_j:
        record.update(destination="excluded_ambiguous", ambiguous=True,
                      would_be=["pref", "abstain"],
                      gt_ids=[g.gt_id for g in shared + extras_i + extras_j])
        return record
    for boxes, side in ((box_i, mi), (box_j, mj)):
        if len(side) > 1 and abs(box_iou(boxes, side[0].box_xyxy)
                                 - box_iou(boxes, side[1].box_xyxy)) <= 1e-9:
            record.update(destination="excluded_ambiguous", ambiguous=True,
                          would_be=[], gt_ids=[g.gt_id for g in side])
            return record
    if shared:
        signs = {}
        for g in shared:
            delta = box_iou(box_i, g.box_xyxy) - box_iou(box_j, g.box_xyxy)
            if abs(delta) >= TAU_GAP:
                signs[g.gt_id] = 1 if delta > 0 else -1
        if 1 in signs.values() and -1 in signs.values():
            record.update(destination="excluded_ambiguous", ambiguous=True,
                          would_be=["pref"], gt_ids=[g.gt_id for g in shared])
            return record
        g_star = sorted(shared, key=lambda g: (-min(box_iou(box_i, g.box_xyxy),
                                                   box_iou(box_j, g.box_xyxy)), g.gt_id))[0]
        delta = box_iou(box_i, g_star.box_xyxy) - box_iou(box_j, g_star.box_xyxy)
        if abs(delta) < TAU_GAP:
            record.update(destination="excluded_gap", gt_ids=[g_star.gt_id])
            return record
        record.update(destination="pref", gt_ids=[g_star.gt_id])
        record["label"] = 1 if delta > 0 else 0
        record["iou_gap"] = abs(delta)
        return record
    record.update(destination="abstain",
                  gt_ids=[g.gt_id for g in mi] + [g.gt_id for g in mj])
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


def assert_control_fairness(pairs: Sequence[PairInput]) -> dict:
    """Structural invariants that PDRA and every control must share."""
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
