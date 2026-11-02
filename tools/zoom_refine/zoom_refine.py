"""r20 zoom-refine prototype: deterministic adaptive window planning plus substitution-only box repair.

Mechanism under test (inference side, no training, no new parameters):
  * the certified whole-image endpoint predicts boxes for small objects;
  * a deterministic planner picks up to K windows around the densest small-detection
    clusters and re-renders each window at 2x magnification into the fixed 768x1344 canvas;
  * every small-band whole-image detection has its box replaced by the best same-class
    window detection when they overlap enough; scores, classes, query ids and the number
    of predictions never change.

Pure CPU module: no model, no CUDA, no RNG.  Rendering helpers are separate so the
control arm (same windows, same compute, no new information) shares all of this code.
"""
import math
from dataclasses import dataclass

BAND_AREA = 1024.0            # COCO small upper bound on the predicted raw box area
WINDOW_RAW_HW = (540, 960)    # window size in raw pixels -> exactly 2x into the canvas
WINDOWS_PER_IMAGE = 2
CANVAS_HW = (768, 1344)
MATCH_IOU_MIN = 0.3
MAX_GT = 1e12


def _finite(value, name):
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _box4(box, name="box"):
    if len(box) != 4:
        raise ValueError(f"{name} must have four coordinates")
    x0, y0, x1, y1 = (_finite(v, name) for v in box)
    if not (x1 > x0 and y1 > y0):
        raise ValueError(f"{name} must have positive extent")
    if x1 - x0 > MAX_GT or y1 - y0 > MAX_GT:
        raise ValueError(f"{name} exceeds the guard")
    return (x0, y0, x1, y1)


def box_iou(first, second):
    a, b = _box4(first, "first"), _box4(second, "second")
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    if union <= 0:
        raise ValueError("degenerate union")
    return inter / union


def box_area(box):
    b = _box4(box)
    return (b[2] - b[0]) * (b[3] - b[1])


def in_band(box, *, band_area=BAND_AREA):
    return box_area(box) < float(band_area)


@dataclass(frozen=True)
class Window:
    """Half-open integer window in raw image pixels, ordered by view id 1..K."""

    view_id: int
    x0: int
    y0: int
    x1: int
    y1: int
    image_width: int
    image_height: int

    def __post_init__(self):
        for key in ("view_id", "x0", "y0", "x1", "y1", "image_width", "image_height"):
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{key} must be an integer")
        if self.view_id < 1 or self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("invalid window identity")
        if not (0 <= self.x0 < self.x1 <= self.image_width and 0 <= self.y0 < self.y1 <= self.image_height):
            raise ValueError("invalid half-open window")

    @property
    def width(self):
        return self.x1 - self.x0

    @property
    def height(self):
        return self.y1 - self.y0

    @property
    def crop_box(self):
        return self.x0, self.y0, self.x1, self.y1

    def as_dict(self):
        return {"view_id": self.view_id, "crop_box": list(self.crop_box),
                "image_width": self.image_width, "image_height": self.image_height,
                "crop_width": self.width, "crop_height": self.height}


def plan_windows(width, height, detections, *, windows=WINDOWS_PER_IMAGE, window_hw=WINDOW_RAW_HW,
                 band_area=BAND_AREA):
    """Deterministic greedy cover of band detections, mirroring the probe's planner.

    detections are whole-image predictions with bbox_xyxy/score/query_id/category_id.
    Ties are broken by (-score, query_id, category_id); windows are clamped to the image
    and the returned centres are a pure function of the inputs (no RNG, no data access).
    """
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise ValueError("image geometry must be positive integers")
    if windows < 0:
        raise ValueError("windows must be non-negative")
    win_h, win_w = (int(window_hw[0]), int(window_hw[1]))
    if win_h <= 0 or win_w <= 0:
        raise ValueError("window size must be positive")
    win_h, win_w = min(win_h, height), min(win_w, width)
    points = []
    for det in detections:
        box = _box4(det["bbox_xyxy"])
        if box_area(box) >= float(band_area):
            continue
        score = _finite(det["score"], "score")
        points.append({"center": ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0),
                       "box": box, "score": score,
                       "query_id": int(det["query_id"]), "category_id": int(det["category_id"])})
    points.sort(key=lambda item: (-item["score"], item["query_id"], item["category_id"],
                                  -item["center"][1], -item["center"][0]))
    covered = [False] * len(points)
    chosen = []
    while len(chosen) < windows and not all(covered):
        pick = next(i for i, hit in enumerate(covered) if not hit)
        cx, cy = points[pick]["center"]
        x0 = int(min(max(round(cx - win_w / 2.0), 0), width - win_w))
        y0 = int(min(max(round(cy - win_h / 2.0), 0), height - win_h))
        for i, item in enumerate(points):
            if not covered[i] and x0 <= item["center"][0] <= x0 + win_w and y0 <= item["center"][1] <= y0 + win_h:
                covered[i] = True
        chosen.append(Window(len(chosen) + 1, x0, y0, x0 + win_w, y0 + win_h, width, height))
    return chosen, {"band_points": len(points), "covered": sum(covered),
                    "coverage": (sum(covered) / len(points)) if points else 0.0}


def canvas_rect(window, *, canvas_hw=CANVAS_HW):
    """Half-open float rect of the window inside the resized whole-image canvas."""
    ch, cw = int(canvas_hw[0]), int(canvas_hw[1])
    sx, sy = cw / window.image_width, ch / window.image_height
    return (window.x0 * sx, window.y0 * sy, window.x1 * sx, window.y1 * sy)


def window_pixels(image, window):
    """Method rendering: true raw pixels of the window resized to the fixed canvas."""
    from torchvision.transforms.v2 import functional as TF
    crop = image.crop(window.crop_box)
    try:
        return TF.pil_to_tensor(TF.resize(crop, list(CANVAS_HW))).float() / 255.0
    finally:
        crop.close()


def control_pixels(canvas, window, *, canvas_hw=CANVAS_HW):
    """Control rendering: the already-seen canvas content, enlarged to the same canvas.

    No original-resolution information can enter here; the tensor is a function of the
    base pass output only, so any A2-A1 difference cannot come from extra compute.
    """
    import torch
    if tuple(canvas.shape[-2:]) != tuple(canvas_hw):
        raise ValueError("base canvas shape differs from CANVAS_HW")
    x0, y0, x1, y1 = canvas_rect(window, canvas_hw=canvas_hw)
    i0, j0 = int(math.floor(y0)), int(math.floor(x0))
    i1, j1 = int(math.ceil(y1)), int(math.ceil(x1))
    i0, j0 = max(i0, 0), max(j0, 0)
    i1, j1 = min(i1, tuple(canvas_hw)[0]), min(j1, tuple(canvas_hw)[1])
    patch = canvas[..., i0:i1, j0:j1]
    out = torch.nn.functional.interpolate(patch.unsqueeze(0), size=list(canvas_hw),
                                          mode="bilinear", align_corners=False)
    return out[0]


def substitute(whole_predictions, window_detections, *, band_area=BAND_AREA, iou_min=MATCH_IOU_MIN):
    """Replace the box of each band detection with the best overlapping window box.

    window_detections is a list of per-window raw-pixel detections already mapped back to
    original image coordinates.  Scores, classes, query ids, list length and list order are
    preserved bit-for-bit; only bbox_xyxy may change.  Deterministic tie-breaks:
    higher IoU, then lower window index, then higher score, then lower query_id.
    """
    repaired = list(whole_predictions)
    hits = 0
    considered = 0
    for index, det in enumerate(whole_predictions):
        box = _box4(det["bbox_xyxy"])
        if box_area(box) >= float(band_area):
            continue
        considered += 1
        best = None
        for wd in window_detections:
            cand = _box4(wd["bbox_xyxy"], "window box")
            if int(wd["category_id"]) != int(det["category_id"]):
                continue
            value = box_iou(box, cand)
            if value < float(iou_min):
                continue
            key = (-value, int(wd["window_id"]), -_finite(wd["score"], "score"), int(wd["query_id"]))
            if best is None or key < best[0]:
                best = (key, cand)
        if best is not None:
            hits += 1
            repaired[index] = dict(det)
            repaired[index]["bbox_xyxy"] = list(best[1])
    return repaired, {"band_considered": considered, "repaired": hits,
                      "unchanged": considered - hits,
                      "repair_share": (hits / considered) if considered else 0.0}


def invariants(whole_predictions, repaired):
    """Return a report that the substitution changed geometry only."""
    if len(whole_predictions) != len(repaired):
        return {"ok": False, "reason": "prediction count changed"}
    changed = 0
    for before, after in zip(whole_predictions, repaired):
        for key in ("image_id", "query_id", "category_id", "score"):
            if before[key] != after[key]:
                return {"ok": False, "reason": f"{key} changed"}
        if [float(v) for v in before["bbox_xyxy"]] != [float(v) for v in after["bbox_xyxy"]]:
            changed += 1
    return {"ok": True, "changed_boxes": changed, "total": len(repaired)}
