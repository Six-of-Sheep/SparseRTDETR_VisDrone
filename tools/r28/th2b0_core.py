"""TH2 B0 (TASK_TH2_B0_001): pure numpy helpers.  No torch, no file access, no model.

The non-separable ("NS") attraction map, the fold test, the geometry of a box under any node grid (magnification,
stretch, bounding expansion, round-trip IoU), the image-level readings, the statistics, the width selection and the
verdict.  A separable map is measured by the same functions on its outer-product node grid.
"""
import math

import numpy as np

GRID = (12, 21)
NODES = (31, 51)                            # vertical, horizontal nodes
PAD = 30
EPS = 0.2
L_HW = (768, 1344)
FG_WIDTH = float.fromhex("0x1.3080000000000p+4")      # 19.03125
F_WIDTH = 4.0
WIDTHS = (4.0, 6.0, 8.0, 10.0, 13.0, 16.0, FG_WIDTH, 24.0, 30.0)
SIZE_EDGES = (16.0 ** 2, 32.0 ** 2, 64.0 ** 2, 96.0 ** 2)
SIZE_NAMES = ("area<16^2", "16^2<=area<=32^2", "32^2<area<64^2", "64^2<=area<96^2", "area>=96^2")
G_NAMES = ("g<1.0", "1.0<=g<1.2", "1.2<=g<1.4", "g>=1.4")
SIDE_POINTS = 8                             # 4 corners + 7 interior points per side = 32 boundary points
INVERSE_TOLERANCE = 1e-9                    # task section 4.4 (normalised coordinates)
NEWTON_TOLERANCE = 1e-13
HOT, COLD, DILATE = 1.2, 0.9, 8.0
DLOGG = 0.10
BOOT, BOOT_SEED = 1000, 20261122
SELECT = {"medium_mean_logg_margin": 0.01, "small_iou90_share": 0.95}
VERDICT = {"go": {"G1_pp": 15.0, "G2_pp": 2.0, "G3_share": 0.95}, "stop": {"G1_pp": 8.0, "G3_share": 0.85}}


def require(ok, message):
    if not ok:
        raise ValueError(message)


# ================================================================ the non-separable map
def _axis(n, nodes):
    """Reading rule of r28_fovea.node_saliency for one axis: cell-centre values, linear between centres, constant beyond."""
    pos = np.arange(nodes, dtype=np.float64) / (nodes - 1) * n - 0.5
    i0 = np.clip(np.floor(pos), 0, n - 1).astype(np.int64)
    i1 = np.minimum(i0 + 1, n - 1)
    return i0, i1, np.clip(pos - i0, 0, 1)


def node_saliency_2d(s):
    """(B, 12, 21) positive saliency on the cells -> (B, 31, 51) on the nodes, divided by its sum."""
    s = np.asarray(s, dtype=np.float64)
    i0, i1, fr = _axis(s.shape[1], NODES[0])
    a = s[:, i0] + fr[None, :, None] * (s[:, i1] - s[:, i0])
    j0, j1, fc = _axis(s.shape[2], NODES[1])
    b = a[:, :, j0] + fc[None, None, :] * (a[:, :, j1] - a[:, :, j0])
    return b / b.sum((1, 2), keepdims=True)


def node_saliency_1d(marg, nodes):
    """numpy twin of r28_fovea.node_saliency."""
    marg = np.asarray(marg, dtype=np.float64)
    i0, i1, fr = _axis(marg.shape[1], nodes)
    s = marg[:, i0] + fr * (marg[:, i1] - marg[:, i0])
    return s / s.sum(1, keepdims=True)


def reflect_index(n):
    j = np.arange(-PAD, n + PAD)
    return np.where(j < 0, -j, np.where(j > n - 1, 2 * (n - 1) - j, j))


def kernel_matrix(n, w):
    """(n, n + 2 PAD): row i holds exp(-4 ln2 (i - j)^2 / w^2) for the padded positions j, zero where |i - j| > PAD."""
    r = np.arange(n, dtype=np.float64)[:, None] - np.arange(-PAD, n + PAD, dtype=np.float64)[None, :]
    k = np.exp(-4.0 * math.log(2.0) * r * r / (w * w))
    return np.where(np.abs(r) <= PAD, k, 0.0)


def ns_nodes(s, w):
    """(B, 31, 51) node saliency -> (raw, G), each (B, 31, 51, 2) with [..., 0] = x and [..., 1] = y: the kernel-weighted
    mean of the padded node coordinates under the two-dimensional kernel (a product of two one-dimensional kernels),
    before and after the official clamp of 2G - 1 to [-1, 1]."""
    s = np.asarray(s, dtype=np.float64)
    ny, nx = NODES
    sp = s[:, reflect_index(ny)][:, :, reflect_index(nx)]
    ky, kx = kernel_matrix(ny, w), kernel_matrix(nx, w)
    px = np.arange(-PAD, nx + PAD, dtype=np.float64) / (nx - 1)
    py = np.arange(-PAD, ny + PAD, dtype=np.float64) / (ny - 1)
    two = lambda m: np.einsum("ia,bac,jc->bij", ky, m, kx, optimize=True)
    den = two(sp)
    raw = np.stack([two(sp * px[None, None, :]) / den, two(sp * py[None, :, None]) / den], -1)
    return raw, (np.clip(raw * 2 - 1, -1, 1) + 1) / 2


def sep_grid(gx, gy):
    """Separable nodes gx (B, 51), gy (B, 31) -> the node grid (B, 31, 51, 2) of the same map."""
    g = np.empty((len(gx), NODES[0], NODES[1], 2), dtype=np.float64)
    g[..., 0] = gx[:, None, :]
    g[..., 1] = gy[:, :, None]
    return g


def corner_dets(G):
    """(B, 31, 51, 2) -> (B, 30, 50, 4): the Jacobian determinant of the bilinear cell at its four corners (node units)."""
    p00, p01, p10, p11 = G[:, :-1, :-1], G[:, :-1, 1:], G[:, 1:, :-1], G[:, 1:, 1:]
    cross = lambda a, b: a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    u0, u1, v0, v1 = p01 - p00, p11 - p10, p10 - p00, p11 - p01
    return np.stack([cross(u0, v0), cross(u0, v1), cross(u1, v0), cross(u1, v1)], -1)


def folded(G):
    """(B,) bool: any corner determinant of any cell is not positive (or not finite).  Nothing is sorted or repaired."""
    d = corner_dets(G)
    return ~(np.isfinite(d) & (d > 0)).all((1, 2, 3))


def nonseparability(s):
    """(B, 12, 21) -> (B,): 1 - sigma_1^2 / sum sigma_i^2."""
    sv = np.linalg.svd(np.asarray(s, dtype=np.float64), compute_uv=False)
    return 1.0 - sv[:, 0] ** 2 / (sv ** 2).sum(1)


# ================================================================ the map on points
def ev(G, b, u, v):
    """G (B, 31, 51, 2); image index b and canvas points (u, v), all (M,).  Returns the source point (x, y) and the
    partial derivatives xu, xv, yu, yv (normalised coordinates) of the piecewise bilinear map."""
    nv, nu = NODES[0] - 1, NODES[1] - 1
    tu, tv = u * nu, v * nv
    cj = np.clip(np.floor(tu), 0, nu - 1).astype(np.int64)
    ci = np.clip(np.floor(tv), 0, nv - 1).astype(np.int64)
    fu, fv = (tu - cj)[:, None], (tv - ci)[:, None]
    p00, p01, p10, p11 = G[b, ci, cj], G[b, ci, cj + 1], G[b, ci + 1, cj], G[b, ci + 1, cj + 1]
    p = p00 * (1 - fu) * (1 - fv) + p01 * fu * (1 - fv) + p10 * (1 - fu) * fv + p11 * fu * fv
    du = ((p01 - p00) * (1 - fv) + (p11 - p10) * fv) * nu
    dv = ((p10 - p00) * (1 - fu) + (p11 - p01) * fu) * nv
    return p[:, 0], p[:, 1], du[:, 0], dv[:, 0], du[:, 1], dv[:, 1]


def _brute(Gi, tx, ty):
    """One point, one image: Newton inside every cell at once; the in-cell solution with the smallest residual."""
    nv, nu = NODES[0] - 1, NODES[1] - 1
    p00, p01, p10, p11 = Gi[:-1, :-1], Gi[:-1, 1:], Gi[1:, :-1], Gi[1:, 1:]
    s = np.full((nv, nu), 0.5)
    t = np.full((nv, nu), 0.5)
    target = np.array([tx, ty])
    with np.errstate(all="ignore"):
        for _ in range(25):
            p = p00 * ((1 - s) * (1 - t))[..., None] + p01 * (s * (1 - t))[..., None] + p10 * ((1 - s) * t)[..., None] + p11 * (s * t)[..., None]
            r = p - target
            ds = (p01 - p00) * (1 - t)[..., None] + (p11 - p10) * t[..., None]
            dt = (p10 - p00) * (1 - s)[..., None] + (p11 - p01) * s[..., None]
            det = ds[..., 0] * dt[..., 1] - dt[..., 0] * ds[..., 1]
            s = np.clip(s - (r[..., 0] * dt[..., 1] - dt[..., 0] * r[..., 1]) / det, -0.5, 1.5)
            t = np.clip(t - (ds[..., 0] * r[..., 1] - r[..., 0] * ds[..., 1]) / det, -0.5, 1.5)
        p = p00 * ((1 - s) * (1 - t))[..., None] + p01 * (s * (1 - t))[..., None] + p10 * ((1 - s) * t)[..., None] + p11 * (s * t)[..., None]
        res = np.abs(p - target).max(-1)
    inside = np.isfinite(res) & (s >= -1e-9) & (s <= 1 + 1e-9) & (t >= -1e-9) & (t <= 1 + 1e-9)
    if not inside.any():
        return None
    res = np.where(inside, res, np.inf)
    i, j = np.unravel_index(np.argmin(res), res.shape)
    return float(np.clip((j + s[i, j]) / nu, 0, 1)), float(np.clip((i + t[i, j]) / nv, 0, 1))


def invert(G, b, tx, ty, iters=60):
    """Source points -> canvas points: numerical inverse of the piecewise bilinear map (damped Newton across cells,
    then an exhaustive per-cell search for what is left).  Returns (u, v, residual, number of points sent to the
    exhaustive search); residual = max(|G(u, v) - target|) evaluated again with the forward map."""
    u, v = np.array(tx, dtype=np.float64), np.array(ty, dtype=np.float64)
    u, v = np.clip(u, 0, 1), np.clip(v, 0, 1)
    px, py, xu, xv, yu, yv = ev(G, b, u, v)
    rx, ry = px - tx, py - ty
    err = np.maximum(np.abs(rx), np.abs(ry))
    for _ in range(iters):
        act = np.flatnonzero(err > NEWTON_TOLERANCE)
        if not len(act):
            break
        with np.errstate(all="ignore"):
            det = xu[act] * yv[act] - xv[act] * yu[act]
            su = (rx[act] * yv[act] - xv[act] * ry[act]) / det
            sv = (xu[act] * ry[act] - rx[act] * yu[act]) / det
        su, sv = np.nan_to_num(su, nan=0.0, posinf=0.0, neginf=0.0), np.nan_to_num(sv, nan=0.0, posinf=0.0, neginf=0.0)
        step = np.ones(len(act))
        todo = np.arange(len(act))
        moved = False
        for _h in range(12):
            a = act[todo]
            un, vn = np.clip(u[a] - step[todo] * su[todo], 0, 1), np.clip(v[a] - step[todo] * sv[todo], 0, 1)
            q = ev(G, b[a], un, vn)
            rxn, ryn = q[0] - tx[a], q[1] - ty[a]
            en = np.maximum(np.abs(rxn), np.abs(ryn))
            better = en < err[a]
            w = a[better]
            if len(w):
                moved = True
                u[w], v[w], rx[w], ry[w], err[w] = un[better], vn[better], rxn[better], ryn[better], en[better]
                xu[w], xv[w], yu[w], yv[w] = q[2][better], q[3][better], q[4][better], q[5][better]
            todo = todo[~better]
            if not len(todo):
                break
            step[todo] *= 0.5
        if not moved:
            break
    left = np.flatnonzero(err > 1e-12)
    for k in left:
        hit = _brute(G[b[k]], tx[k], ty[k])
        if hit is not None:
            q = ev(G, b[k:k + 1], np.array([hit[0]]), np.array([hit[1]]))
            e = max(abs(q[0][0] - tx[k]), abs(q[1][0] - ty[k]))
            if e < err[k]:
                u[k], v[k], err[k] = hit[0], hit[1], e
    return u, v, err, int(len(left))


# ================================================================ boxes
_T = np.arange(SIDE_POINTS, dtype=np.float64) / SIDE_POINTS


def box_points(xyxy):
    """(N, 4) -> (N, 32, 2): the corners and 7 interior points of every side, counter-clockwise from (x1, y1)."""
    x1, y1, x2, y2 = (np.asarray(xyxy, dtype=np.float64)[:, k:k + 1] for k in range(4))
    w, h = x2 - x1, y2 - y1
    xs = np.concatenate([x1 + w * _T, x2 + 0 * _T, x2 - w * _T, x1 + 0 * _T], 1)
    ys = np.concatenate([y1 + 0 * _T, y1 + h * _T, y2 + 0 * _T, y2 - h * _T], 1)
    return np.stack([xs, ys], -1)


def poly_area(p):
    x, y = p[..., 0], p[..., 1]
    return 0.5 * np.abs((x * np.roll(y, -1, -1) - np.roll(x, -1, -1) * y).sum(-1))


def iou(a, b):
    lt, rb = np.maximum(a[:, :2], b[:, :2]), np.minimum(a[:, 2:], b[:, 2:])
    inter = np.clip(rb - lt, 0, None).prod(1)
    return inter / ((a[:, 2:] - a[:, :2]).prod(1) + (b[:, 2:] - b[:, :2]).prod(1) - inter)


def box_metrics(G, b, xyxy):
    """Per box (image index b into G, normalised source corners): g, stretch, bounding expansion, round-trip IoU, the
    canvas polygon (N, 32, 2), the residual of the inverse and the number of points sent to the exhaustive search."""
    xyxy = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
    n = len(xyxy)
    pts = box_points(xyxy)
    cx, cy = (xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2
    tx = np.concatenate([pts[..., 0].reshape(-1), cx])
    ty = np.concatenate([pts[..., 1].reshape(-1), cy])
    bb = np.concatenate([np.repeat(b, 32), b])
    u, v, err, brute = invert(G, bb, tx, ty)
    poly = np.stack([u[:n * 32].reshape(n, 32), v[:n * 32].reshape(n, 32)], -1)
    a_c = poly_area(poly)
    a_s = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    lo, hi = poly.min(1), poly.max(1)
    rect = np.concatenate([lo, hi], 1)
    back = box_points(rect)
    q = ev(G, np.repeat(b, 32), back[..., 0].reshape(-1), back[..., 1].reshape(-1))
    bx, by = q[0].reshape(n, 32), q[1].reshape(n, 32)
    src = np.stack([bx.min(1), by.min(1), bx.max(1), by.max(1)], 1)
    c = ev(G, b, u[n * 32:], v[n * 32:])
    f = (c[2] ** 2 + c[3] ** 2 + c[4] ** 2 + c[5] ** 2) / 2
    det = np.abs(c[2] * c[5] - c[3] * c[4])
    root = np.sqrt(np.clip(f * f - det * det, 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        stretch = 0.5 * np.log((f + root) / (f - root))
        res = {"g": np.sqrt(a_c / a_s), "stretch": stretch, "expand": (hi - lo).prod(1) / a_c, "iou": iou(xyxy, src)}
    res.update(poly=poly, residual=err.reshape(-1)[:n * 32].reshape(n, 32).max(1) if n else np.zeros(0), brute=brute)
    if n:
        res["residual"] = np.maximum(res["residual"], err[n * 32:])
    return res


# ================================================================ image-level readings
def local_magnification(Gi, hw=L_HW):
    """(31, 51, 2) -> (H, W): 1 / sqrt(det J) of the canvas -> source map at every canvas pixel centre."""
    h, w = hw
    vv, uu = np.meshgrid((np.arange(h) + 0.5) / h, (np.arange(w) + 0.5) / w, indexing="ij")
    q = ev(Gi[None], np.zeros(h * w, dtype=np.int64), uu.reshape(-1), vv.reshape(-1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return (1.0 / np.sqrt(q[2] * q[5] - q[3] * q[4])).reshape(h, w)


def _hull(p):
    """Convex hull (counter-clockwise) of (n, 2) points: monotone chain."""
    p = np.unique(p, axis=0)
    if len(p) < 3:
        return p
    cross = lambda o, a, c: (a[0] - o[0]) * (c[1] - o[1]) - (a[1] - o[1]) * (c[0] - o[0])
    lower, upper = [], []
    for q in p:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    for q in p[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    return np.array(lower[:-1] + upper[:-1])


def cover_mask(polys, hw=L_HW, dilate=DILATE):
    """Canvas polygons (n, 32, 2) in normalised coordinates -> (H, W) bool: pixels that meet the polygon dilated by
    `dilate` pixels (the square structuring element of the phase-1 rule; convex hull of the dilated polygon).  A pixel
    [r, r+1) x [c, c+1) meets the dilated shape iff its centre lies in the shape dilated by a further half pixel."""
    h, w = hw
    mask = np.zeros((h, w), dtype=bool)
    d = dilate + 0.5
    off = np.array([[-d, -d], [d, -d], [d, d], [-d, d]])
    for p in polys:
        px = p * np.array([w, h], dtype=np.float64)
        hull = _hull((px[:, None, :] + off[None]).reshape(-1, 2))
        lo, hi = hull.min(0), hull.max(0)
        c0, c1 = max(0, int(math.ceil(lo[0] - 0.5))), min(w, int(math.floor(hi[0] - 0.5)) + 1)
        r0, r1 = max(0, int(math.ceil(lo[1] - 0.5))), min(h, int(math.floor(hi[1] - 0.5)) + 1)
        if c0 >= c1 or r0 >= r1:
            continue
        yy, xx = np.meshgrid(np.arange(r0, r1) + 0.5, np.arange(c0, c1) + 0.5, indexing="ij")
        a, bq = hull, np.roll(hull, -1, 0)
        ex, ey = (bq - a)[:, 0], (bq - a)[:, 1]
        inside = np.ones(yy.shape, dtype=bool)
        for k in range(len(hull)):
            inside &= ex[k] * (yy - a[k, 1]) - ey[k] * (xx - a[k, 0]) >= -1e-9
        mask[r0:r1, c0:c1] |= inside
    return mask


def image_level(Gi, polys_small, polys_all):
    """One image: pixel counts (canvas, hot, hot outside every small GT, cold, cold outside every GT)."""
    m = local_magnification(Gi)
    hot, cold = m >= HOT, m <= COLD
    out = [m.size, int(hot.sum()), 0, int(cold.sum()), 0]
    if out[1]:
        out[2] = int((hot & ~cover_mask(polys_small)).sum())
    if out[3]:
        out[4] = int((cold & ~cover_mask(polys_all)).sum())
    return out


# ================================================================ statistics
def size_bin(area):
    a = np.asarray(area, dtype=np.float64)
    return np.where(a < SIZE_EDGES[0], 0, np.where(a <= SIZE_EDGES[1], 1, np.where(a < SIZE_EDGES[2], 2, np.where(a < SIZE_EDGES[3], 3, 4))))


def g_bin(g):
    g = np.asarray(g, dtype=np.float64)
    return np.where(g < 1.0, 0, np.where(g < 1.2, 1, np.where(g < 1.4, 2, 3)))


def groups(size):
    return {"small": size <= 1, "medium": (size == 2) | (size == 3), "large96": size == 4}


def _mean(v, m):
    return float(v[m].mean()) if m.any() else None


def _quant(v, m, q):
    return float(np.quantile(v[m], q)) if m.any() else None


def row(size, m, ok):
    """One table row without intervals.  m: dict of per-GT arrays (g, stretch, expand, iou); ok: rows that were measured."""
    gr = {k: v & ok for k, v in groups(size).items()}
    with np.errstate(divide="ignore", invalid="ignore"):
        lg = np.log(m["g"])
    s, md, lg96 = gr["small"], gr["medium"], gr["large96"]
    g = m["g"]
    return {"n_small": int(s.sum()), "n_medium": int(md.sum()), "n_large96": int(lg96.sum()),
            "small_mean_logg": _mean(lg, s), "small_g_geomean": None if not s.any() else float(np.exp(lg[s].mean())),
            "small_share_g_lt_1": _mean(g < 1.0, s), "small_share_g_lt_1.2": _mean(g < 1.2, s), "small_share_g_ge_1.4": _mean(g >= 1.4, s),
            "medium_mean_logg": _mean(lg, md), "medium_share_g_lt_1": _mean(g < 1.0, md),
            "large96_g_geomean": None if not lg96.any() else float(np.exp(lg[lg96].mean())),
            "small_stretch_median": _quant(m["stretch"], s, 0.5), "small_stretch_p90": _quant(m["stretch"], s, 0.9),
            "small_share_iou_ge_0.90": _mean(m["iou"] >= 0.90, s), "small_share_iou_ge_0.75": _mean(m["iou"] >= 0.75, s),
            "small_expand_median": _quant(m["expand"], s, 0.5), "all_expand_median": _quant(m["expand"], ok, 0.5)}


def boot_weights(n_images, n=BOOT, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_images, np.full(n_images, 1.0 / n_images), size=n).astype(np.float64)


def by_image(values, image, n_images):
    out = np.zeros(n_images, dtype=np.float64)
    np.add.at(out, np.asarray(image), np.asarray(values, dtype=np.float64))
    return out


def interval(samples):
    s = np.asarray(samples, dtype=np.float64)
    s = s[np.isfinite(s)]
    return [None, None] if not len(s) else [float(np.quantile(s, 0.025)), float(np.quantile(s, 0.975))]


def ratio_ci(num, den, image, n_images, w, f=None):
    """f(sum(num) / sum(den)) with the image-bootstrap interval."""
    a, b = by_image(num, image, n_images), by_image(den, image, n_images)
    f = f or (lambda x: x)
    with np.errstate(divide="ignore", invalid="ignore"):
        return {"value": None if b.sum() == 0 else float(f(a.sum() / b.sum())), "ci95": interval(f((w @ a) / (w @ b)))}


def quantile_ci(v, mask, q, image, w):
    """Quantile of v over the masked rows; the interval from the image bootstrap (weighted step quantile)."""
    idx = np.flatnonzero(mask)
    if not len(idx):
        return {"value": None, "ci95": [None, None]}
    order = idx[np.argsort(v[idx], kind="stable")]
    vals, img = v[order], np.asarray(image)[order]
    samples = np.empty(len(w))
    for k in range(len(w)):
        cw = np.cumsum(w[k][img])
        samples[k] = np.nan if cw[-1] == 0 else vals[min(np.searchsorted(cw, q * cw[-1]), len(vals) - 1)]
    return {"value": float(np.quantile(vals, q)), "ci95": interval(samples)}


def row_ci(size, m, ok, image, n_images, w):
    """The columns of `row` with image-bootstrap intervals."""
    gr = {k: v & ok for k, v in groups(size).items()}
    with np.errstate(divide="ignore", invalid="ignore"):
        lg = np.where(ok, np.log(m["g"]), 0.0)
    g = np.where(ok, m["g"], 1.0)
    s, md, l96 = gr["small"], gr["medium"], gr["large96"]
    share = lambda cond, sel: ratio_ci(cond & sel, sel, image, n_images, w)
    io = np.where(ok, m["iou"], 0.0)
    return {"n_small": int(s.sum()), "n_medium": int(md.sum()), "n_large96": int(l96.sum()),
            "small_mean_logg": ratio_ci(lg * s, s, image, n_images, w), "small_g_geomean": ratio_ci(lg * s, s, image, n_images, w, np.exp),
            "small_share_g_lt_1": share(g < 1.0, s), "small_share_g_lt_1.2": share(g < 1.2, s), "small_share_g_ge_1.4": share(g >= 1.4, s),
            "medium_mean_logg": ratio_ci(lg * md, md, image, n_images, w), "medium_share_g_lt_1": share(g < 1.0, md),
            "large96_g_geomean": ratio_ci(lg * l96, l96, image, n_images, w, np.exp),
            "small_stretch_median": quantile_ci(m["stretch"], s, 0.5, image, w), "small_stretch_p90": quantile_ci(m["stretch"], s, 0.9, image, w),
            "small_share_iou_ge_0.90": share(io >= 0.90, s), "small_share_iou_ge_0.75": share(io >= 0.75, s),
            "small_expand_median": quantile_ci(m["expand"], s, 0.5, image, w), "all_expand_median": quantile_ci(m["expand"], ok, 0.5, image, w)}


def describe(v):
    v = np.asarray(v, dtype=np.float64)
    return {"n": int(len(v)), "mean": float(v.mean()), "p10": float(np.quantile(v, 0.1)), "p50": float(np.quantile(v, 0.5)),
            "p90": float(np.quantile(v, 0.9))}


# ================================================================ selection and verdict
def select(ns_rows, sep_reference_medium_mean_logg):
    """ns_rows: {width hex: {"folded_images", "medium_mean_logg", "small_share_iou_ge_0.90", "small_mean_logg"}} of the
    NS rows of T1 with source P.  The three removals in the order of the task, then the largest small mean log g."""
    removed, left = {}, []
    floor = sep_reference_medium_mean_logg - SELECT["medium_mean_logg_margin"]
    for h, r in ns_rows.items():
        reasons = []
        if r["folded_images"] > 0:
            reasons.append("folded")
        if r["medium_mean_logg"] is None or r["medium_mean_logg"] < floor:
            reasons.append("medium_mean_logg_more_than_0.01_below_SEP(19.03125)")
        if r["small_share_iou_ge_0.90"] is None or r["small_share_iou_ge_0.90"] < SELECT["small_iou90_share"]:
            reasons.append("small_round_trip_iou_ge_0.90_share_below_0.95")
        if reasons:
            removed[h] = {"first_reason": reasons[0], "all_reasons": reasons, "width": float.fromhex(h)}
        else:
            left.append(h)
    star = None if not left else max(left, key=lambda h: (ns_rows[h]["small_mean_logg"], -float.fromhex(h)))
    return {"w_star_hex": star, "w_star": None if star is None else float.fromhex(star), "removed": removed, "left": left,
            "medium_floor": floor, "SEP_reference_medium_mean_logg": sep_reference_medium_mean_logg}


def verdict(d):
    """d: selected (bool), G1_pp, G2_pp, G3_share, folds ({name: count})."""
    go, stop = VERDICT["go"], VERDICT["stop"]
    g = {"G1": d["G1_pp"] >= go["G1_pp"], "G2": d["G2_pp"] <= go["G2_pp"], "G3": d["G3_share"] >= go["G3_share"],
         "G4": all(v == 0 for v in d["folds"].values())}
    s = {"G1": d["G1_pp"] < stop["G1_pp"], "G3": d["G3_share"] < stop["G3_share"], "no_width_left": not d["selected"]}
    label = "STOP" if any(s.values()) else "GO" if all(g.values()) else "GREY"
    return {"label": label, "GO": g, "STOP": s}
