"""R31 external evaluation on UAVDT-M (TASK_R31_EXTERNAL_UAVDT_001): pure helpers.

No torch and no project import: parsing of the toolkit files, the official ignore rule, the vehicle candidate rule,
COCOeval plumbing, the registered label, the descriptive tables and the dHash arithmetic.  Everything here is fixed
before a UAVDT result is read; the runner (r31_external.py) only wires these functions to files and models.

Conventions
  boxes      GT rows are the toolkit's integers (left, top, width, height) in the 1024 x 540 frame and are never
             clipped or removed.  Candidate boxes are FP32 pixel corners in the same frame, not clipped.
  areas      GT area = width * height of the annotation.  The COCO ranges keep their native closed end points
             (small = [0, 32^2], medium = [32^2, 96^2], large = [96^2, 1e10]), as in the development evaluation; the
             two sub-ranges of small are area < 16^2 and 16^2 <= area <= 32^2, the split of R30-001.
  candidates one per query: the query's box, score = the largest sigmoid score of the classes car, van, truck, bus;
             all 300 are kept, ordered by score (descending) and query index.
  ignore     CalculateDetectionPR_overall.m of the UAVDT toolkit: a candidate is deleted when its four edges lie
             strictly inside one ignore rectangle of the same frame.  Nothing is refilled; GT is never deleted.
"""
import math

import numpy as np

TEST_SEQUENCES = ("M0203", "M0205", "M0208", "M0209", "M0403", "M0601", "M0602", "M0606", "M0701", "M0801",
                  "M0802", "M1001", "M1004", "M1007", "M1009", "M1101", "M1301", "M1302", "M1303", "M1401")
SEQUENCE_FRAMES = (1007, 646, 265, 1576, 514, 372, 480, 1374, 1308, 298, 1101, 1859, 269, 659, 604, 864, 1182, 719, 445, 1050)
FRAMES = 16592
# task section 5.1: the 16 test sequences with at least 1,000 small GT
SEQUENCES_GE_1000_SMALL = ("M0209", "M0403", "M0601", "M0602", "M0606", "M0701", "M0802", "M1001", "M1004", "M1007",
                           "M1009", "M1101", "M1301", "M1302", "M1303", "M1401")
DOMINANT_SEQUENCE = "M0701"
IMAGE_WH = (1024, 540)
ATTRIBUTES = ("daylight", "night", "fog", "low-alt", "medium-alt", "high-alt", "front-view", "side-view", "bird-view", "long-term")
WEATHER = ("daylight", "night", "fog")
ALTITUDE = ("low-alt", "medium-alt", "high-alt")
UAVDT_CLASSES = {1: "car", 2: "truck", 3: "bus"}
# zero-based logit columns of the detector (VisDrone category id - 1): car 4, van 5, truck 6, bus 9
VEHICLE_COLUMNS = (3, 4, 5, 8)
THREE_CLASS_COLUMNS = {1: (3, 4), 2: (5,), 3: (8,)}       # UAVDT category -> detector columns; van counted as car
THREE_CLASS_HEADER = "van 归入 car 是未经核实的假设"
QUERIES, CLASSES = 300, 10

SMALL_AREA, LARGE_AREA, SPLIT_AREA = 32.0 ** 2, 96.0 ** 2, 16.0 ** 2
AREA_RANGES = (("all", 0.0, 1e10), ("small", 0.0, SMALL_AREA), ("medium", SMALL_AREA, LARGE_AREA), ("large", LARGE_AREA, 1e10),
               ("lt16", 0.0, float(np.nextafter(SPLIT_AREA, 0.0))), ("16to32", SPLIT_AREA, SMALL_AREA))
SIDE_AREA_RANGES = AREA_RANGES[:2]
MAX_DETS = (1, 10, 100)
MAIN = ("AP", "APS", "AP75_small")
EXTRA = ("AP50", "AP75", "APM", "APL", "AR_small", "AP_lt16", "AP75_lt16", "AP_16to32", "AP75_16to32")
METRICS = MAIN + EXTRA
RULE = (("APS", 1.0), ("AP75_small", 1.0), ("AP", 0.0))      # pp, the R30-002 rule, all three at once
LABELS = ("external_gain_repeated_3of3", "external_gain_reduced", "external_gain_reversed", "external_gain_mixed")

BIN_NAMES = ("g<1.0", "1.0<=g<1.2", "1.2<=g<1.4", "g>=1.4")
THRESHOLDS = (0.5, 0.75)
AREA_GROUPS = ("small", "area<16^2", "16^2<=area<=32^2")
QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
DHASH_STEP = 10
DHASH_NEAR = 6


def require(ok, message):
    if not ok:
        raise ValueError(message)


# ================================================================ toolkit files
def parse_table(text, columns):
    """Comma separated numeric rows -> (n, columns) FP64."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            v = [float(x) for x in line.split(",")]
            require(len(v) == columns, f"a row has {len(v)} columns, expected {columns}: {line[:80]!r}")
            rows.append(v)
    return np.array(rows, dtype=np.float64).reshape(-1, columns)


def parse_gt(text):
    """<seq>_gt_whole.txt: frame, target id, left, top, width, height, out-of-view, occlusion, category -> (n, 9) int64."""
    a = parse_table(text, 9)
    require(bool((a == np.round(a)).all()), "a GT value is not an integer")
    a = a.astype(np.int64)
    require(bool((a[:, 4:6] > 0).all()) and bool(np.isin(a[:, 8], list(UAVDT_CLASSES)).all()) and bool((a[:, 0] >= 1).all()),
            "GT with a non-positive side, an unknown category or a frame index below 1")
    return a


def parse_ignore(text):
    """<seq>_gt_ignore.txt: frame, id, left, top, width, height, 1, -1, -1 -> (n, 5) int64: frame, left, top, width, height."""
    a = parse_table(text, 9)
    require(bool((a == np.round(a)).all()), "an ignore value is not an integer")
    a = a.astype(np.int64)[:, [0, 2, 3, 4, 5]]
    require(bool((a[:, 0] >= 1).all()), "ignore rectangle with a frame index below 1")
    return a


def parse_detections(text):
    """RES_DET/<detector>/<seq>.txt: frame, -1, left, top, width, height, score, ... -> (n, 6) FP64."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            v = [float(x) for x in line.split(",")]
            require(len(v) >= 7, f"a detection row has {len(v)} columns: {line[:80]!r}")
            rows.append([v[0], v[2], v[3], v[4], v[5], v[6]])
    return np.array(rows, dtype=np.float64).reshape(-1, 6)


def attribute_sequence(member):
    """'M_attr/test/M0701 _attr.txt' -> ('test', 'M0701'); None for another member."""
    parts = member.split("/")
    if len(parts) != 3 or not parts[2].endswith("_attr.txt"):
        return None
    return parts[1], parts[2].split("_attr")[0].strip()


def parse_attributes(text):
    v = [int(x) for x in text.replace("\n", "").replace("\r", "").split(",") if x.strip() != ""]
    require(len(v) == len(ATTRIBUTES) and set(v) <= {0, 1}, f"attribute row: {v}")
    return [a for a, on in zip(ATTRIBUTES, v) if on == 1]


def sequence_groups(attrs):
    """{sequence: [attribute names]} -> {group: [sequences]} for the three altitude and the three weather labels;
    every sequence carries exactly one label of each family."""
    out = {a: [] for a in ALTITUDE + WEATHER}
    for s in TEST_SEQUENCES:
        alt = [a for a in attrs[s] if a in ALTITUDE]
        wea = [a for a in attrs[s] if a in WEATHER]
        require(len(alt) == 1 and len(wea) == 1, f"sequence {s} does not carry one altitude and one weather label: {attrs[s]}")
        out[alt[0]].append(s)
        out[wea[0]].append(s)
    return out


def census(gt, ignore):
    """gt {seq: (n, 9)}, ignore {seq: (m, 5)} -> the counts of the selection census (its own definitions: lt16 is
    area < 256, 16to32 is 256 <= area < 1024, small is area < 1024) and the closed-end-point counts used by COCOeval."""
    out = {"gt": 0, "lt16": 0, "16to32": 0, "small": 0, "32to96": 0, "ge96": 0, "ignore_boxes": 0, "frames_with_gt": 0,
           "frames_gt_over_100": 0, "max_gt_per_frame": 0, "gt_center_in_ignore": 0, "gt_center_in_ignore_small": 0,
           "gt_strictly_inside_ignore": 0, "area_eq_1024": 0, "area_eq_9216": 0, "max_x2": 0, "max_y2": 0, "min_x": None, "min_y": None,
           "cat": {n: 0 for n in UAVDT_CLASSES.values()}, "per_sequence": {}}
    for s in gt:
        g, ig = gt[s], ignore[s]
        area = g[:, 4] * g[:, 5]
        frames, per_frame = np.unique(g[:, 0], return_counts=True)
        q = {"gt": int(len(g)), "small": int((area < 1024).sum()), "lt16": int((area < 256).sum()), "ignore_boxes": int(len(ig)),
             "frames_with_gt": int(len(frames)), "first_frame": int(frames.min()), "last_frame": int(frames.max()),
             "max_gt_per_frame": int(per_frame.max()), "frames_with_ignore": int(len(np.unique(ig[:, 0]))),
             "cat": {n: int((g[:, 8] == c).sum()) for c, n in UAVDT_CLASSES.items() if (g[:, 8] == c).any()}}
        out["per_sequence"][s] = q
        out["gt"] += q["gt"]
        out["lt16"] += q["lt16"]
        out["16to32"] += int(((area >= 256) & (area < 1024)).sum())
        out["small"] += q["small"]
        out["32to96"] += int(((area >= 1024) & (area < 9216)).sum())
        out["ge96"] += int((area >= 9216).sum())
        out["ignore_boxes"] += q["ignore_boxes"]
        out["frames_with_gt"] += q["frames_with_gt"]
        out["frames_gt_over_100"] += int((per_frame > 100).sum())
        out["max_gt_per_frame"] = max(out["max_gt_per_frame"], q["max_gt_per_frame"])
        out["area_eq_1024"] += int((area == 1024).sum())
        out["area_eq_9216"] += int((area == 9216).sum())
        out["max_x2"] = max(out["max_x2"], int((g[:, 2] + g[:, 4]).max()))
        out["max_y2"] = max(out["max_y2"], int((g[:, 3] + g[:, 5]).max()))
        out["min_x"] = int(g[:, 2].min()) if out["min_x"] is None else min(out["min_x"], int(g[:, 2].min()))
        out["min_y"] = int(g[:, 3].min()) if out["min_y"] is None else min(out["min_y"], int(g[:, 3].min()))
        for c, n in UAVDT_CLASSES.items():
            out["cat"][n] += int((g[:, 8] == c).sum())
        by = {}
        for r in ig:
            by.setdefault(int(r[0]), []).append((r[1], r[2], r[1] + r[3], r[2] + r[4]))
        for f in np.unique(g[:, 0]):
            rects = by.get(int(f))
            if not rects:
                continue
            rows = g[g[:, 0] == f]
            cx, cy = rows[:, 2] + rows[:, 4] / 2.0, rows[:, 3] + rows[:, 5] / 2.0
            centre = np.zeros(len(rows), dtype=bool)
            for x1, y1, x2, y2 in rects:
                centre |= (x1 <= cx) & (cx <= x2) & (y1 <= cy) & (cy <= y2)
            out["gt_center_in_ignore"] += int(centre.sum())
            out["gt_center_in_ignore_small"] += int((centre & (rows[:, 4] * rows[:, 5] < 1024)).sum())
            xyxy = np.stack([rows[:, 2], rows[:, 3], rows[:, 2] + rows[:, 4], rows[:, 3] + rows[:, 5]], 1).astype(np.float64)
            out["gt_strictly_inside_ignore"] += int(inside_ignore(xyxy, np.array([[a, b, c - a, d - b] for a, b, c, d in rects], dtype=np.float64)).sum())
    return out


# ================================================================ the official ignore rule
def inside_ignore(xyxy, rects):
    """(n, 4) corners of the candidates of one frame, (m, 4) ignore rectangles (left, top, width, height) of the same
    frame -> (n,) bool: every edge strictly inside one rectangle."""
    xyxy = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
    rects = np.asarray(rects, dtype=np.float64).reshape(-1, 4)
    if not len(xyxy) or not len(rects):
        return np.zeros(len(xyxy), dtype=bool)
    left, top = rects[:, 0][None], rects[:, 1][None]
    right, bottom = (rects[:, 0] + rects[:, 2])[None], (rects[:, 1] + rects[:, 3])[None]
    return ((xyxy[:, 0:1] > left) & (xyxy[:, 1:2] > top) & (xyxy[:, 2:3] < right) & (xyxy[:, 3:4] < bottom)).any(1)


def toolkit_deleted(det, ign):
    """Independent per-detection judgement in the toolkit's own terms (the MATLAB expression, one detection and one
    ignore row at a time; the ignore rows are only looked up by their frame number first):
    det (frame, left, top, width, height, ...), ign (frame, left, top, width, height) -> list of bool."""
    by = {}
    for g in ign:
        by.setdefault(float(g[0]), []).append(g)
    out = []
    for d in det:
        hit = False
        for g in by.get(float(d[0]), ()):
            if d[1] > g[1] and d[2] > g[2] and d[1] + d[3] < g[1] + g[3] and d[2] + d[4] < g[2] + g[4]:
                hit = True
                break
        out.append(hit)
    return out


def deleted_by_frame(det, ign):
    """The rule of the scoring path on a toolkit detection table: (n,) bool, frame by frame through inside_ignore."""
    out = np.zeros(len(det), dtype=bool)
    for f in np.unique(det[:, 0]):
        rows = np.nonzero(det[:, 0] == f)[0]
        d = det[rows]
        xyxy = np.stack([d[:, 1], d[:, 2], d[:, 1] + d[:, 3], d[:, 2] + d[:, 4]], 1)
        out[rows] = inside_ignore(xyxy, ign[ign[:, 0] == f][:, 1:5])
    return out


# ================================================================ candidates
def pixel_boxes(boxes, width, height):
    """Normalised FP32 cxcywh (..., 4) -> FP32 pixel corners: the two FP32 operations of r0.evaluation.rank_candidates."""
    b = np.asarray(boxes)
    require(b.dtype == np.float32 and b.shape[-1] == 4, "pixel_boxes takes FP32 cxcywh")
    half = b[..., 2:] / np.float32(2)
    xy = np.concatenate((b[..., :2] - half, b[..., :2] + half), -1)
    return xy * np.array([width, height, width, height], dtype=np.float32)


def class_scores(scores, columns):
    """(..., 300, 10) FP32 sigmoid scores -> (..., 300): the largest score of the named columns."""
    s = np.asarray(scores)
    require(s.dtype == np.float32 and s.shape[-1] == CLASSES, "class_scores takes the FP32 score table")
    return s[..., list(columns)].max(-1)


def candidate_order(score):
    """(..., 300) -> query indices by score (descending), ties by query index."""
    return np.argsort(-np.asarray(score), axis=-1, kind="stable")


def submission(inside, score, limit=None):
    """Query indices of one frame in submission order: by score (descending, ties by query index), without the
    candidates deleted by the ignore rule; nothing is refilled.  `limit` keeps the first rows of that list."""
    order = candidate_order(score)
    kept = order[~np.asarray(inside, dtype=bool)[order]]
    return kept if limit is None else kept[:limit]


def records_equal(mine, stored):
    """Field-for-field comparison of prediction records; returns the first difference or None."""
    if len(mine) != len(stored):
        return {"index": None, "lengths": [len(mine), len(stored)]}
    for i, (a, b) in enumerate(zip(mine, stored)):
        if a != b:
            return {"index": i, "mine": a, "stored": b}
    return None


def normalised_cxcywh(xywh, width, height):
    """Integer GT rows (left, top, width, height) -> FP32 normalised cxcywh, the form a detector emits."""
    b = np.asarray(xywh, dtype=np.float64)
    out = np.stack([(b[:, 0] + b[:, 2] / 2) / width, (b[:, 1] + b[:, 3] / 2) / height, b[:, 2] / width, b[:, 3] / height], 1)
    return out.astype(np.float32)


# ================================================================ COCOeval plumbing
def coco_dataset(n_images, gt_image, gt_xywh, gt_category, categories):
    """Every frame is an image (also the frames without GT); every GT row is scored (iscrowd 0, area = w * h)."""
    w, h = IMAGE_WH
    anns = [{"id": k + 1, "image_id": int(i) + 1, "category_id": int(c), "bbox": [float(b[0]), float(b[1]), float(b[2]), float(b[3])],
             "area": float(b[2] * b[3]), "iscrowd": 0}
            for k, (i, b, c) in enumerate(zip(gt_image, gt_xywh, gt_category))]
    return {"info": {"description": "R31 UAVDT-M test, COCO-style bridge"},
            "images": [{"id": i + 1, "width": w, "height": h} for i in range(n_images)],
            "categories": [{"id": int(c), "name": n} for c, n in categories.items()], "annotations": anns}


def detection_records(image, category, xyxy, score):
    """Parallel arrays (already in submission order) -> COCO result records (bbox = left, top, width, height)."""
    xyxy = np.asarray(xyxy, dtype=np.float64)
    return [{"image_id": int(i) + 1, "category_id": int(c), "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
             "score": float(s)} for i, c, b, s in zip(image, category, xyxy, score)]


def cocoeval(dataset, detections, area_ranges=AREA_RANGES, max_dets=MAX_DETS):
    """pycocotools COCOeval after evaluate(): IoU 0.50:0.05:0.95, 101 recall points, the given area ranges."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    coco = COCO()
    coco.dataset = dataset
    coco.createIndex()
    if detections:
        dt = coco.loadRes(detections)
    else:
        dt = COCO()
        dt.dataset = {"info": dataset["info"], "images": dataset["images"], "categories": dataset["categories"], "annotations": []}
        dt.createIndex()
    e = COCOeval(coco, dt, "bbox")
    e.params.imgIds = sorted(r["id"] for r in dataset["images"])
    e.params.catIds = sorted(c["id"] for c in dataset["categories"])
    e.params.maxDets = list(max_dets)
    e.params.areaRng = [[lo, hi] for _, lo, hi in area_ranges]
    e.params.areaRngLbl = [n for n, _, _ in area_ranges]
    require(len(e.params.iouThrs) == 10 and abs(float(e.params.iouThrs[5]) - 0.75) < 1e-12 and len(e.params.recThrs) == 101,
            "COCOeval thresholds are not 0.50:0.05:0.95 / 101 recall points")
    e.evaluate()
    return e


def accumulate(e, positions=None):
    """(precision [T, R, K, A, M], recall [T, K, A, M]) of COCOeval.accumulate over the images at `positions` of the
    evaluated image list (all images when None).  COCOeval.accumulate(p) itself cannot take an image subset: it indexes
    evalImgs with positions of the subset list, so the per-image results are re-sliced here and accumulated unchanged."""
    import contextlib
    import copy
    import io
    pe = e._paramsEval
    n_img, n_area, n_cat = len(pe.imgIds), len(pe.areaRng), len(pe.catIds)
    require(len(e.evalImgs) == n_img * n_area * n_cat, "COCOeval per-image result layout")
    positions = list(range(n_img)) if positions is None else [int(i) for i in positions]
    require(positions == sorted(set(positions)) and (not positions or (positions[0] >= 0 and positions[-1] < n_img)), "image positions")
    sub = copy.copy(e)
    sub.params, sub._paramsEval = copy.deepcopy(pe), copy.deepcopy(pe)
    ids = [pe.imgIds[i] for i in positions]
    sub.params.imgIds, sub._paramsEval.imgIds = list(ids), list(ids)
    sub.evalImgs = [e.evalImgs[k * n_area * n_img + a * n_img + i] for k in range(n_cat) for a in range(n_area) for i in positions]
    sub.eval = {}
    with contextlib.redirect_stdout(io.StringIO()):
        sub.accumulate()
    return sub.eval["precision"], sub.eval["recall"]


def _mean(x):
    v = x[x > -1]
    return float(v.mean()) if len(v) else None


def read_cells(precision, recall, labels, category=None):
    """{area label: {AP, AP50, AP75, AR}} at the last maxDets column; category = index into K or None for all."""
    k = slice(None) if category is None else slice(category, category + 1)
    return {label: {"AP": _mean(precision[:, :, k, a, -1]), "AP50": _mean(precision[0, :, k, a, -1]),
                    "AP75": _mean(precision[5, :, k, a, -1]), "AR": _mean(recall[:, k, a, -1])}
            for a, label in enumerate(labels)}


def named_metrics(cells):
    """The cells of the six registered area ranges -> the named readings of the task (0..1; None without GT)."""
    return {"AP": cells["all"]["AP"], "APS": cells["small"]["AP"], "AP75_small": cells["small"]["AP75"],
            "AP50": cells["all"]["AP50"], "AP75": cells["all"]["AP75"], "APM": cells["medium"]["AP"], "APL": cells["large"]["AP"],
            "AR_small": cells["small"]["AR"], "AP_lt16": cells["lt16"]["AP"], "AP75_lt16": cells["lt16"]["AP75"],
            "AP_16to32": cells["16to32"]["AP"], "AP75_16to32": cells["16to32"]["AP75"]}


# ================================================================ reading
def difference_pp(a, b, metrics=METRICS):
    return {m: (None if a.get(m) is None or b.get(m) is None else 100.0 * (a[m] - b[m])) for m in metrics if m in a and m in b}


def mean_std(values):
    """Mean and sample standard deviation (n - 1) of the values; None when a value is missing."""
    if any(v is None for v in values) or not values:
        return {"mean": None, "sample_std": None}
    n = len(values)
    mean = math.fsum(values) / n
    return {"mean": mean, "sample_std": math.sqrt(math.fsum((v - mean) ** 2 for v in values) / (n - 1)) if n > 1 else None}


def paired(minuend, subtrahend, metrics=METRICS):
    """{seed: metric dict} x 2 -> per-seed differences (pp) and the mean / sample std of the per-seed differences."""
    seeds = sorted(minuend)
    require(seeds == sorted(subtrahend), "paired seeds")
    rows = {s: difference_pp(minuend[s], subtrahend[s], metrics) for s in seeds}
    keys = [m for m in metrics if all(m in rows[s] for s in seeds)]
    return {"per_seed": rows, **{m: mean_std([rows[s][m] for s in seeds]) for m in keys}}


def label(diffs):
    """The registered label from the three per-seed F_geo - L0 differences (pp) of AP, APS and AP75_small."""
    require(len(diffs) == 3 and all(d[m] is not None for d in diffs for m in MAIN), "the label needs three complete seeds")
    if all(all(d[m] >= t for m, t in RULE) for d in diffs):
        return LABELS[0]
    if all(d["APS"] > 0 for d in diffs):
        return LABELS[1]
    if all(d["APS"] < 0 for d in diffs):
        return LABELS[2]
    return LABELS[3]


def rule_cells(d):
    return {m: {"difference_pp": d[m], "minimum_pp": t, "passed": d[m] >= t} for m, t in RULE}


# ================================================================ magnification tables (definitions of R30-001 A1)
def bin_index(g):
    return 0 if g < 1.0 else 1 if g < 1.2 else 2 if g < 1.4 else 3


def bin_indices(g):
    g = np.asarray(g, dtype=np.float64)
    return np.where(g < 1.0, 0, np.where(g < 1.2, 1, np.where(g < 1.4, 2, 3)))


def area_masks(area):
    """Masks of small GT rows by annotation area: every small row, area < 16^2, 16^2 <= area <= 32^2."""
    area = np.asarray(area, dtype=np.float64)
    require(bool((area <= SMALL_AREA).all()), "area_masks takes small GT only")
    return {"small": np.ones(len(area), dtype=bool), "area<16^2": area < SPLIT_AREA, "16^2<=area<=32^2": area >= SPLIT_AREA}


def bin_table(g, area, iou, pairs=()):
    """g, area: (n,) of the small GT; iou: {endpoint: (n,) best IoU}.  One row per (bin + all) x area group with the
    recall of every endpoint at both thresholds and the differences (pp) of the named pairs."""
    bins = bin_indices(g)
    masks = area_masks(area)
    rows = []
    for b, name in enumerate(BIN_NAMES + ("all",)):
        in_bin = np.ones(len(bins), dtype=bool) if name == "all" else bins == b
        for group in AREA_GROUPS:
            sel = in_bin & masks[group]
            row = {"bin": name, "area_group": group, "n": int(sel.sum())}
            for t in THRESHOLDS:
                r = {k: (float((v[sel] >= t).mean()) if sel.any() else None) for k, v in iou.items()}
                row[f"recall@{t}"] = r
                row[f"difference_pp@{t}"] = {f"{a}-{c}": (None if r[a] is None else 100.0 * (r[a] - r[c])) for a, c in pairs}
            rows.append(row)
    return rows


def geometry_reading(g, area):
    """Per size class (COCO ranges on the annotation area, closed end points): pooled magnification and shares."""
    g, area = np.asarray(g, dtype=np.float64), np.asarray(area, dtype=np.float64)
    classes = {"small": area <= SMALL_AREA, "medium": (area >= SMALL_AREA) & (area <= LARGE_AREA), "large": area >= LARGE_AREA,
               "small_area<16^2": area < SPLIT_AREA, "small_16^2<=area<=32^2": (area >= SPLIT_AREA) & (area <= SMALL_AREA)}
    out = {}
    for name, sel in classes.items():
        v = g[sel]
        out[name] = {"n": int(sel.sum())}
        if len(v):
            out[name].update(g_geomean=float(np.exp(np.log(v).mean())), g_mean=float(v.mean()), area_ratio_mean=float((v * v).mean()),
                             g_q10_50_90=[float(q) for q in np.quantile(v, [0.1, 0.5, 0.9])],
                             frac_g_lt_1=float((v < 1.0).mean()), frac_g_lt_1p2=float((v < 1.2).mean()),
                             frac_g_ge_1p2=float((v >= 1.2).mean()), frac_g_ge_1p4=float((v >= 1.4).mean()),
                             bin_counts=[int((bin_indices(v) == b).sum()) for b in range(4)])
    return out


def best_iou(gt_xyxy, cand_xyxy):
    """(n, 4), (m, 4) FP64 corners -> (n,) best IoU of every GT over the candidates (0 without candidates)."""
    gt = np.asarray(gt_xyxy, dtype=np.float64).reshape(-1, 4)
    cand = np.asarray(cand_xyxy, dtype=np.float64).reshape(-1, 4)
    if not len(gt) or not len(cand):
        return np.zeros(len(gt), dtype=np.float64)
    lt = np.maximum(gt[:, None, :2], cand[None, :, :2])
    rb = np.minimum(gt[:, None, 2:], cand[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    union = (cand[:, 2:] - cand[:, :2]).prod(1)[None] + (gt[:, 2:] - gt[:, :2]).prod(1)[:, None] - inter
    return (inter / union).max(1)


def quantiles(v):
    v = np.asarray(v, dtype=np.float64)
    return {"n": int(len(v)), **{f"q{round(100 * q):02d}": (float(np.quantile(v, q)) if len(v) else None) for q in QUANTILES},
            "mean": float(v.mean()) if len(v) else None}


def canvas_sizes(w, h, image_w, image_h, canvas_hw):
    """Box sides after the uniform (aspect-free) resize of the image to the canvas: width, height, area on the canvas."""
    w, h = np.asarray(w, dtype=np.float64), np.asarray(h, dtype=np.float64)
    cw = w * canvas_hw[1] / np.asarray(image_w, dtype=np.float64)
    ch = h * canvas_hw[0] / np.asarray(image_h, dtype=np.float64)
    return {"width": quantiles(cw), "height": quantiles(ch), "area": quantiles(cw * ch),
            "share_area_le_32^2": float((cw * ch <= SMALL_AREA).mean()) if len(w) else None,
            "share_area_lt_16^2": float((cw * ch < SPLIT_AREA).mean()) if len(w) else None}


# ================================================================ dHash
def dhash(gray):
    """(8, 9) grey levels -> 64-bit difference hash: bit = right pixel brighter than its left neighbour, row major."""
    g = np.asarray(gray)
    require(g.shape == (8, 9), "dHash takes an 8 x 9 grey image (9 columns, 8 rows)")
    bits = (g[:, 1:] > g[:, :-1]).reshape(-1)
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value


_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming(a, b):
    """(n,), (m,) uint64 -> (n, m) uint8 Hamming distances."""
    a, b = np.asarray(a, dtype=np.uint64), np.asarray(b, dtype=np.uint64)
    x = a[:, None] ^ b[None, :]
    return _POP[x.view(np.uint8).reshape(len(a), len(b), 8)].sum(-1, dtype=np.uint8)


def every_tenth(frames):
    """Frame numbers 1, 11, 21, ... of one sequence of `frames` frames."""
    return list(range(1, frames + 1, DHASH_STEP))
