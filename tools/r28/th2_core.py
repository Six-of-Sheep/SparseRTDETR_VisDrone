"""TH2 phase 1 (TASK_TH2_PHASE1_001): pure numpy helpers.  No torch, no file access, no model.

Everything that defines a reading lives here so that the tests can exercise it without data: the folds, the
perturbation draws, the candidate rule, the per-GT outcomes, the image bootstrap, the slope tables, the slope
models M0 / M1 / M2, the rank remap of the value map, the two geometry readings of E3 and the verdict.
"""
import hashlib
import math

import numpy as np

GRID = (12, 21)
L_HW = (768, 1344)
BLOCKS = ((3, 5), (5, 9))                   # block of the perturbation; the second only by the fallback rule
AMPLITUDES = (2.0, 3.0, 4.0, 6.0, 8.0)
AMP_TARGET, AMP_FLOOR = 0.15, 0.08
K_PERTURB = 48
DRAW_TAG = "TH2-PHASE1-001"
SIZE_EDGES = (16.0 ** 2, 32.0 ** 2, 64.0 ** 2, 96.0 ** 2)
SIZE_NAMES = ("area<16^2", "16^2<=area<=32^2", "32^2<area<64^2", "64^2<=area<96^2", "area>=96^2")
SMALL_AREA = 32.0 ** 2
G_NAMES = ("g<1.0", "1.0<=g<1.2", "1.2<=g<1.4", "g>=1.4")
CROWD_EDGES = (1, 4, 10)
CROWD_NAMES = ("0", "1-3", "4-9", ">=10")
CROWD_RADIUS = 64.0
AMPLIFIED = 0.10                            # dlog g of a "magnified" pair in R2
MIN_SSX = 0.02                              # smallest sum of dlog g^2 of a per-GT slope (per half)
BOOT, BOOT_SEED = 1000, 20261122
S_KNOTS = (16.0, 32.0, 64.0, 96.0)
S_CAP = 160.0
VERDICT = {"go": {"small_hit75_pp": 0.50, "aps_pp": 0.20, "ap75s_pp": 0.20, "seed_floor_pp": -0.10, "medium_hit75_pp": -0.30},
           "stop": {"divergent_share": 0.10, "small_hit75_pp": 0.20}}


def require(ok, message):
    if not ok:
        raise ValueError(message)


# ================================================================ folds and draws
def folds(ids):
    """ids in evaluation order -> (n,) int8: 0 = fold A (positions 1, 3, 5, ... of the list sorted by image id),
    1 = fold B (positions 2, 4, ...)."""
    order = sorted(range(len(ids)), key=lambda i: ids[i])
    out = np.empty(len(ids), dtype=np.int8)
    for pos, i in enumerate(order):
        out[i] = pos % 2
    return out


def draw(image_id, k):
    """Perturbation k (1..K) of one image: (centre row, centre column, up).  Depends on the image id and k only.
    The direction is fixed by k (up, down, down, up, ...) so that each image and each parity half of k has exactly
    half of either direction."""
    require(1 <= k <= K_PERTURB, "perturbation index")
    h = hashlib.sha256(f"{DRAW_TAG}|{image_id}|{k}".encode()).digest()
    r = int.from_bytes(h[:8], "big") % GRID[0]
    c = int.from_bytes(h[8:16], "big") % GRID[1]
    return r, c, (k - 1) % 4 in (0, 3)


def block_mask(r, c, block):
    """(12, 21) bool: the block of `block` cells centred on (r, c), clipped at the border."""
    bh, bw = block
    m = np.zeros(GRID, dtype=bool)
    m[max(0, r - bh // 2):r + bh // 2 + 1, max(0, c - bw // 2):c + bw // 2 + 1] = True
    return m


def perturb(demand, image_id, k, a, block):
    """One perturbed copy of a (12, 21) demand map (FP64) and its block mask."""
    r, c, up = draw(image_id, k)
    m = block_mask(r, c, block)
    out = np.array(demand, dtype=np.float64, copy=True)
    out[m] = out[m] * (a if up else 1.0 / a)
    return out, m


def cell_of(xyxy_norm):
    """(n, 4) normalised corners -> (row, col) of the box centre on the 12 x 21 grid (the rule of the label demand)."""
    b = np.asarray(xyxy_norm, dtype=np.float64).reshape(-1, 4)
    ci = np.clip(np.floor((b[:, 1] + b[:, 3]) / 2 * GRID[0]).astype(np.int64), 0, GRID[0] - 1)
    cj = np.clip(np.floor((b[:, 0] + b[:, 2]) / 2 * GRID[1]).astype(np.int64), 0, GRID[1] - 1)
    return ci, cj


def choose_amplitude(medians):
    """{a: median |dlog g|} -> (a or None, reached).  The amplitude whose median is closest to the target; None when
    no candidate reaches the floor."""
    require(sorted(medians) == sorted(AMPLITUDES), "amplitude table")
    if max(medians.values()) < AMP_FLOOR:
        return None, False
    return min(AMPLITUDES, key=lambda a: (abs(medians[a] - AMP_TARGET), a)), True


# ================================================================ candidates and outcomes
def top600(scores, boxes, width, height):
    """scores: (300, 10) FP32 sigmoid scores; boxes: (300, 4) FP32 cxcywh.  Returns (category, score, corners FP64, query)."""
    s = np.asarray(scores)
    b = np.asarray(boxes)
    require(s.dtype == np.float32 and s.shape == (300, 10) and b.dtype == np.float32 and b.shape == (300, 4), "raw output schema")
    flat = s.reshape(-1)
    q = np.repeat(np.arange(300), 10)
    cl = np.tile(np.arange(10), 300)
    order = np.lexsort((cl, q, -flat))[:600]
    half = b[:, 2:] / np.float32(2)
    xy = np.concatenate((b[:, :2] - half, b[:, :2] + half), -1) * np.array([width, height, width, height], dtype=np.float32)
    return cl[order] + 1, flat[order], xy[q[order]].astype(np.float64), q[order]


def outcomes(gt_xyxy, gt_cat, cand_cat, cand_score, cand_xyxy):
    """Per GT: (best same-class IoU, highest score among same-class candidates with IoU >= 0.5 or 0)."""
    gt = np.asarray(gt_xyxy, dtype=np.float64).reshape(-1, 4)
    n = len(gt)
    iou_max, q50 = np.zeros(n, dtype=np.float64), np.zeros(n, dtype=np.float64)
    if not n or not len(cand_cat):
        return iou_max, q50
    cand = np.asarray(cand_xyxy, dtype=np.float64).reshape(-1, 4)
    lt = np.maximum(gt[:, None, :2], cand[None, :, :2])
    rb = np.minimum(gt[:, None, 2:], cand[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    union = (cand[:, 2:] - cand[:, :2]).prod(1)[None] + (gt[:, 2:] - gt[:, :2]).prod(1)[:, None] - inter
    iou = inter / union
    same = np.asarray(gt_cat)[:, None] == np.asarray(cand_cat)[None, :]
    iou = np.where(same, iou, 0.0)
    iou_max = iou.max(1)
    q50 = np.where(iou >= 0.5, np.asarray(cand_score, dtype=np.float64)[None, :], 0.0).max(1)
    return iou_max, q50


# ================================================================ bins and features
def size_bin(area):
    a = np.asarray(area, dtype=np.float64)
    return np.where(a < SIZE_EDGES[0], 0, np.where(a <= SIZE_EDGES[1], 1, np.where(a < SIZE_EDGES[2], 2, np.where(a < SIZE_EDGES[3], 3, 4))))


def g_bin(g):
    g = np.asarray(g, dtype=np.float64)
    return np.where(g < 1.0, 0, np.where(g < 1.2, 1, np.where(g < 1.4, 2, 3)))


def crowd_bin(n):
    n = np.asarray(n)
    return np.where(n < CROWD_EDGES[0], 0, np.where(n < CROWD_EDGES[1], 1, np.where(n < CROWD_EDGES[2], 2, 3)))


def omega(s):
    return np.clip((64.0 - np.asarray(s, dtype=np.float64)) / 32.0, 0.0, 1.0)


def crowd_counts(image, cx, cy):
    """Number of other boxes of the same image whose centre lies within CROWD_RADIUS (canvas L pixels)."""
    image, cx, cy = np.asarray(image), np.asarray(cx, dtype=np.float64), np.asarray(cy, dtype=np.float64)
    out = np.zeros(len(image), dtype=np.int64)
    order = np.argsort(image, kind="stable")
    bounds = np.flatnonzero(np.diff(image[order])) + 1
    for idx in np.split(order, bounds):
        d2 = (cx[idx, None] - cx[None, idx]) ** 2 + (cy[idx, None] - cy[None, idx]) ** 2
        out[idx] = (d2 <= CROWD_RADIUS ** 2).sum(1) - 1
    return out


def m1_basis(s):
    """Piecewise linear basis in s (canvas L pixels, capped): 1, s, hinges at the knots."""
    s = np.minimum(np.asarray(s, dtype=np.float64), S_CAP) / 32.0
    cols = [np.ones_like(s), s] + [np.clip(s - k / 32.0, 0, None) for k in S_KNOTS]
    return np.stack(cols, 1)


M2_EXTRA = ("class_2..10", "occlusion_1", "occlusion_2", "truncation_1", "log_aspect", "abs_log_aspect", "log1p_crowd", "log_g0")


def m2_basis(f):
    """f: dict of per-GT arrays (s, category, occlusion, truncation, log_aspect, crowd, g0).  The M1 basis, the other
    features as main effects, and each of them times omega(s) (the interaction terms)."""
    s = np.asarray(f["s"], dtype=np.float64)
    cat = np.asarray(f["category"])
    extra = [(cat == k).astype(np.float64) for k in range(2, 11)]
    extra += [(np.asarray(f["occlusion"]) == 1).astype(np.float64), (np.asarray(f["occlusion"]) >= 2).astype(np.float64),
              (np.asarray(f["truncation"]) >= 1).astype(np.float64)]
    la = np.asarray(f["log_aspect"], dtype=np.float64)
    extra += [la, np.abs(la), np.log1p(np.asarray(f["crowd"], dtype=np.float64)), np.log(np.asarray(f["g0"], dtype=np.float64))]
    e = np.stack(extra, 1)
    return np.concatenate([m1_basis(s), e, e * omega(s)[:, None]], 1)


def fit_slope_model(phi, dx, dy):
    """Least squares of dy on (phi . theta) dx over all (GT, perturbation) pairs.  phi (n, p); dx, dy (n, K) with NaN
    where a pair is absent.  Returns theta (minimum norm)."""
    ok = np.isfinite(dx) & np.isfinite(dy)
    x = np.where(ok, dx, 0.0)
    y = np.where(ok, dy, 0.0)
    sxx, sxy = (x * x).sum(1), (x * y).sum(1)
    a = (phi * sxx[:, None]).T @ phi
    b = phi.T @ sxy
    return np.linalg.lstsq(a, b, rcond=None)[0]


# ================================================================ image bootstrap
def boot_weights(n_images, n=BOOT, seed=BOOT_SEED):
    """(n, n_images) resampling counts."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_images, np.full(n_images, 1.0 / n_images), size=n).astype(np.float64)


def by_image(values, image, n_images):
    """Sum of per-row values (n,) or (n, k) per image -> (n_images,) or (n_images, k)."""
    v = np.asarray(values, dtype=np.float64)
    out = np.zeros((n_images,) + v.shape[1:], dtype=np.float64)
    np.add.at(out, np.asarray(image), v)
    return out


def interval(samples):
    s = np.asarray(samples, dtype=np.float64)
    s = s[np.isfinite(s)]
    if not len(s):
        return [None, None]
    return [float(np.quantile(s, 0.025)), float(np.quantile(s, 0.975))]


def ratio(num, den, image, n_images, w, scale=1.0):
    """sum(num) / sum(den) with the image bootstrap interval; num and den are per-row arrays."""
    a, b = by_image(num, image, n_images), by_image(den, image, n_images)
    tb = b.sum()
    point = None if tb == 0 else float(scale * a.sum() / tb)
    with np.errstate(divide="ignore", invalid="ignore"):
        samples = scale * (w @ a) / (w @ b)
    return {"value": point, "ci95": interval(samples), "n": float(tb)}


def _corr_from_sums(s):
    n, sx, sy, sxx, syy, sxy = (s[..., i] for i in range(6))
    with np.errstate(divide="ignore", invalid="ignore"):
        cov = sxy - sx * sy / n
        vx, vy = sxx - sx * sx / n, syy - sy * sy / n
        return cov / np.sqrt(vx * vy)


def corr_sums(x, y, image, n_images, mask=None):
    """Per-image sums (n_images, 6) from which a Pearson correlation of the pooled rows follows."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if mask is not None:
        ok &= mask
    x, y = np.where(ok, x, 0.0), np.where(ok, y, 0.0)
    cols = [ok.astype(np.float64), x, y, x * x, y * y, x * y]
    if x.ndim == 2:
        cols = [c.sum(1) for c in cols]
    return by_image(np.stack(cols, 1), image, n_images)


def corr(sums, w):
    point = _corr_from_sums(sums.sum(0))
    samples = _corr_from_sums(np.einsum("bi,ik->bk", w, sums))
    return {"value": None if not np.isfinite(point) else float(point), "ci95": interval(samples), "n": float(sums[:, 0].sum())}


def corr_difference(sums_a, sums_b, w):
    """corr_a - corr_b on the same resamples."""
    point = _corr_from_sums(sums_a.sum(0)) - _corr_from_sums(sums_b.sum(0))
    samples = _corr_from_sums(np.einsum("bi,ik->bk", w, sums_a)) - _corr_from_sums(np.einsum("bi,ik->bk", w, sums_b))
    ci = interval(samples)
    return {"value": None if not np.isfinite(point) else float(point), "ci95": ci,
            "ci_contains_zero": None if ci[0] is None else bool(ci[0] <= 0.0 <= ci[1])}


# ================================================================ slopes
def slope(dx, dy, image, n_images, w, mask):
    """Through-origin regression of dy on dx over the masked pairs: slope per 0.1 log g, times 100 (pp)."""
    ok = mask & np.isfinite(dx) & np.isfinite(dy)
    x, y = np.where(ok, dx, 0.0), np.where(ok, dy, 0.0)
    out = ratio((x * y).sum(1), (x * x).sum(1), image, n_images, w, scale=10.0)
    n = int(ok.sum())
    out.update(pairs=n, mean_dx=float(x.sum() / n) if n else None, mean_dy_pp=float(100.0 * y.sum() / n) if n else None)
    out["n"] = n
    return out


def per_gt_slope(dx, dy, columns=None):
    """Per-GT through-origin slope over the chosen perturbation columns; NaN where the sum of dx^2 is below MIN_SSX."""
    if columns is not None:
        dx, dy = dx[:, columns], dy[:, columns]
    ok = np.isfinite(dx) & np.isfinite(dy)
    x, y = np.where(ok, dx, 0.0), np.where(ok, dy, 0.0)
    ssx = (x * x).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ssx >= MIN_SSX, (x * y).sum(1) / ssx, np.nan)


def auc(score, label):
    """Area under the ROC curve (ties counted half); None when a class is absent."""
    score, label = np.asarray(score, dtype=np.float64), np.asarray(label, dtype=bool)
    n1, n0 = int(label.sum()), int((~label).sum())
    if not n1 or not n0:
        return None
    order = np.argsort(score, kind="stable")
    ranks = np.empty(len(score), dtype=np.float64)
    s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def deciles(pred, obs):
    """Ten equal-count groups by the prediction: mean prediction and mean observation of each."""
    pred, obs = np.asarray(pred, dtype=np.float64).reshape(-1), np.asarray(obs, dtype=np.float64).reshape(-1)
    ok = np.isfinite(pred) & np.isfinite(obs)
    pred, obs = pred[ok], obs[ok]
    order = np.argsort(pred, kind="stable")
    rows = []
    for part in np.array_split(order, 10):
        rows.append({"n": int(len(part)), "predicted_pp": float(100.0 * pred[part].mean()) if len(part) else None,
                     "observed_pp": float(100.0 * obs[part].mean()) if len(part) else None})
    return rows


# ================================================================ value map
def rank_remap(value, target):
    """(12, 21) value map and (12, 21) target map -> a map holding exactly the values of `target`, the largest at the
    cell of the largest `value`.  Ties of `value` are ordered by `target` (descending), then by the cell index, so a
    constant value map returns `target` unchanged."""
    v = np.asarray(value, dtype=np.float64).reshape(-1)
    t = np.asarray(target, dtype=np.float64).reshape(-1)
    require(v.shape == t.shape == (GRID[0] * GRID[1],), "rank remap takes two 12 x 21 maps")
    cells = np.lexsort((np.arange(len(v)), -t, -v))
    out = np.empty_like(t)
    out[cells] = np.sort(t)[::-1]
    return out.reshape(GRID)


def weighted_demand(xyxy_norm, weight):
    """The label-demand accumulation with a given per-box weight: (12, 21) FP64 before the 3 x 3 mean."""
    ci, cj = cell_of(xyxy_norm)
    d = np.zeros(GRID[0] * GRID[1], dtype=np.float64)
    np.add.at(d, ci * GRID[1] + cj, np.asarray(weight, dtype=np.float64))
    return d.reshape(GRID)


def blur3(d):
    """3 x 3 mean with replicated border (numpy twin of r28_warp.blur3)."""
    p = np.pad(np.asarray(d, dtype=np.float64), 1, mode="edge")
    out = np.zeros(GRID, dtype=np.float64)
    for i in range(3):
        for j in range(3):
            out += p[i:i + GRID[0], j:j + GRID[1]]
    return out / 9.0


# ================================================================ E3
def waterfill_area(count, lo, hi):
    """Cell area scales a (mean 1, lo <= a <= hi) maximising sum(count * log a).  Cells with a count get
    clip(count / lam, lo, hi); cells without one carry no objective and take the lower bound, or what is left of the
    budget when every counted cell is already at the upper bound."""
    c = np.asarray(count, dtype=np.float64).reshape(-1)
    n = len(c)
    pos = c > 0
    n1, n0 = int(pos.sum()), int((~pos).sum())
    out = np.ones(n)
    if not n1:
        return out
    target = n - n0 * lo
    if n1 * hi <= target:
        out[pos] = hi
        if n0:
            out[~pos] = (n - n1 * hi) / n0
        return out
    a, b = 1e-12, c.max() / lo * 10
    for _ in range(200):
        m = math.sqrt(a * b)
        if np.clip(c[pos] / m, lo, hi).sum() > target:
            a = m
        else:
            b = m
    out[pos] = np.clip(c[pos] / math.sqrt(a * b), lo, hi)
    out[~pos] = lo
    return out


# ================================================================ verdict
def verdict(d):
    """d: the readings of the pre-registered rule.  Returns the label and every condition."""
    g, s = VERDICT["go"], VERDICT["stop"]
    small, med = d["small_hit75_pp"], d["medium_hit75_pp"]
    mean = lambda v: float(np.mean(v))
    signs = {int(np.sign(x)) for x in small}
    go = {"1_small_hit75": mean(small) >= g["small_hit75_pp"] and all(x > 0 for x in small),
          "2_APS_AP75small": (mean(d["aps_pp"]) >= g["aps_pp"] and mean(d["ap75s_pp"]) >= g["ap75s_pp"]
                              and min(list(d["aps_pp"]) + list(d["ap75s_pp"])) >= g["seed_floor_pp"]),
          "3_medium_hit75": mean(med) >= g["medium_hit75_pp"]}
    stop = {"1_no_increment_and_same_shape": bool(d["m2_increment_ci_contains_zero"] and d["m1_shape_agrees"]),
            "2_divergent_set_below_10pct": d["divergent_share"] < s["divergent_share"],
            "3_net_change_small_or_mixed_sign": mean(small) <= s["small_hit75_pp"] or len(signs) > 1}
    label = "GO" if all(go.values()) else "STOP" if any(stop.values()) else "UNDECIDED"
    if all(go.values()) and any(stop.values()):
        label = "UNDECIDED"
    return {"label": label, "GO": go, "STOP": stop,
            "means_pp": {"small_hit75": mean(small), "medium_hit75": mean(med), "APS": mean(d["aps_pp"]), "AP75_small": mean(d["ap75s_pp"])}}
