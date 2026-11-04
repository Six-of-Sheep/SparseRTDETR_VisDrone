"""Channel logic for the cached error analysis.  Pure numpy/pycocotools; no model, no CUDA.

Matching is always the certified one: pycocotools evalImgs on the frozen bridge inputs
(raw-ignore filter applied first, class score 0 as crowd, maxDets 100 per image per class).

Channel semantics follow tidecv v1.0.1 (quantify.TIDEExample / TIDERun._eval_image,
errors.error.BestGTMatch, errors.main_errors):
  * test order for an unmatched detection: Loc, Cls, Dupe, Bkg, otherwise Both;
  * Loc and Cls name the argmax GT as target; only the highest scoring Loc/Cls error on an
    unused GT is fixed into a TP, every other error on that GT is suppressed;
  * Dupe/Bkg/Both fixes remove the detection, Miss fixes lower the GT count;
  * each oracle starts from the unmodified run and is never accumulated or summed.

Two explicit extensions keep COCO area semantics in the small evaluation:
  * GT outside the area range stay present for IoU and "used"; an error targeting one is
    removed by its fix and yields no TP, as a re-evaluation would (never a background error);
  * unmatched detections outside the area range are evaluator-ignored: they add no FP and act
    only when a Loc/Cls fix would turn them into a TP for an unused in-range GT.
Detections matched to an ignored GT or crowd are not errors (tidecv still counts its ignored
predictions; that difference is reported beside the native run, not hidden here).
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
import io
import math

import numpy as np

T_INDEX = {0.5: 0, 0.75: 5}
BG = 0.1
AREA_ALL, AREA_SMALL = 0, 1
AREA_NAMES = {AREA_ALL: 'all', AREA_SMALL: 'small'}
REC = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
BIN_EDGES = np.array([8., 12., 17., 32.])
BIN_LABELS = ('(0,8]', '(8,12]', '(12,17]', '(17,32]')
NONE, CLS, LOC, BOTH, DUPE, BKG = 0, 1, 2, 3, 4, 5
CHANNELS = (('Cls', CLS), ('Loc', LOC), ('Both', BOTH), ('Dupe', DUPE), ('Bkg', BKG))
ORACLES = ('Cls', 'Loc', 'Both', 'Dupe', 'Bkg', 'Miss', 'FP_rank', 'FN_recall')
KEYS = ('AP', 'AP50', 'AP75', 'APS', 'APM', 'APL', 'AR1', 'AR10', 'AR100',
        'AR_small', 'AR_medium', 'AR_large')


def evaluate_certified(dataset, det):
    """Body of sparse_rtdetr.r0.evaluation.bridge_metrics, keeping the evaluator object."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO()
        coco.dataset = dataset
        coco.createIndex()
        dt = coco.loadRes(det)
        e = COCOeval(coco, dt, 'bbox')
        e.params.imgIds = sorted(r['id'] for r in dataset['images'])
        e.params.catIds = list(range(1, 11))
        e.params.maxDets = [1, 10, 100]
        e.evaluate()
        e.accumulate()
        e.summarize()
    values = {k: float(v) for k, v in zip(KEYS, e.stats)}
    small = e.eval['precision'][5, :, :, 1, 2]
    valid = small[small > -1]
    values['AP75_small'] = float(valid.mean()) if len(valid) else -1.
    return e, values


@dataclass
class GTTable:
    id: np.ndarray
    img: np.ndarray
    cat: np.ndarray
    bbox: np.ndarray
    area: np.ndarray
    crowd: np.ndarray

    @property
    def n(self):
        return len(self.id)


def gt_table(dataset):
    a = dataset['annotations']
    ids = np.array([g['id'] for g in a], dtype=np.int64)
    if not np.array_equal(ids, np.arange(1, len(a) + 1)):
        raise ValueError('bridge GT ids must be 1..N in list order')
    return GTTable(ids, np.array([g['image_id'] for g in a], dtype=np.int64),
                   np.array([g['category_id'] for g in a], dtype=np.int64),
                   np.array([g['bbox'] for g in a], dtype=np.float64).reshape(-1, 4),
                   np.array([g['area'] for g in a], dtype=np.float64),
                   np.array([bool(g['iscrowd']) for g in a], dtype=bool))


@dataclass
class DetTable:
    img: np.ndarray
    cat: np.ndarray
    bbox: np.ndarray
    score: np.ndarray


def det_table(det):
    """Call after loadRes: pycocotools assigned id = position + 1."""
    if any(d['id'] != i + 1 for i, d in enumerate(det)):
        raise ValueError('detection ids must be list position + 1')
    return DetTable(np.array([d['image_id'] for d in det], dtype=np.int64),
                    np.array([d['category_id'] for d in det], dtype=np.int64),
                    np.array([d['bbox'] for d in det], dtype=np.float64).reshape(-1, 4),
                    np.array([d['score'] for d in det], dtype=np.float64))


@dataclass
class RunTable:
    """One (area, IoU) evaluation; detections concatenated per class in accumulate order."""
    area: int
    t: float
    det_id: np.ndarray
    cat_index: np.ndarray
    img_index: np.ndarray
    score: np.ndarray
    dtm: np.ndarray
    dtig: np.ndarray
    slices: list
    npig: np.ndarray
    gt_used: np.ndarray
    gt_nonig: np.ndarray
    gt_match: np.ndarray
    et: np.ndarray = None
    fix: np.ndarray = None
    tgt: np.ndarray = None
    gt_usable: np.ndarray = None
    _rows: list = field(default=None, repr=False)

    def init_errors(self):
        self.et = np.zeros(len(self.det_id), dtype=np.int8)
        self.fix = np.zeros(len(self.det_id), dtype=bool)
        self.tgt = np.zeros(len(self.det_id), dtype=np.int64)
        self.gt_usable = np.zeros(len(self.gt_used), dtype=bool)

    def rows_of_image(self, i, n_img):
        if self._rows is None:
            order = np.argsort(self.img_index, kind='stable')
            cuts = np.searchsorted(self.img_index[order], np.arange(n_img + 1))
            self._rows = [order[cuts[k]:cuts[k + 1]] for k in range(n_img)]
        return self._rows[i]


def collect_run(e, area_index, t, n_gt):
    p = e.params
    K, I, A = len(p.catIds), len(p.imgIds), len(p.areaRng)
    t_index = T_INDEX[t]
    if p.iouThrs[t_index] != t or p.maxDets[-1] != 100:
        raise ValueError('unexpected COCOeval parameters')
    ids, cats, imgs, scores, dtm, dtig, slices = [], [], [], [], [], [], []
    npig = np.zeros(K, dtype=np.int64)
    used = np.zeros(n_gt + 1, dtype=bool)
    nonig = np.zeros(n_gt + 1, dtype=bool)
    match = np.zeros(n_gt + 1, dtype=np.int64)
    start = 0
    for k in range(K):
        count = 0
        for i in range(I):
            x = e.evalImgs[k * A * I + area_index * I + i]
            if x is None:
                continue
            D = len(x['dtIds'])
            if D:
                ids.append(np.asarray(x['dtIds'], dtype=np.int64))
                cats.append(np.full(D, k, dtype=np.int16))
                imgs.append(np.full(D, i, dtype=np.int32))
                scores.append(np.asarray(x['dtScores'], dtype=np.float64))
                dtm.append(x['dtMatches'][t_index])
                dtig.append(x['dtIgnore'][t_index])
            npig[k] += int(np.count_nonzero(x['gtIgnore'] == 0))
            gid = np.asarray(x['gtIds'], dtype=np.int64)
            if len(gid):
                gm = x['gtMatches'][t_index]
                used[gid] = gm > 0
                nonig[gid] = x['gtIgnore'] == 0
                match[gid] = gm.astype(np.int64)
            count += D
        slices.append((start, start + count))
        start += count

    def joined(xs, dtype):
        return np.concatenate(xs).astype(dtype) if xs else np.zeros(0, dtype=dtype)

    return RunTable(area_index, t, joined(ids, np.int64), joined(cats, np.int16), joined(imgs, np.int32),
                    joined(scores, np.float64), joined(dtm, np.float64), joined(dtig, bool),
                    slices, npig, used, nonig, match)


def curve(scores, matched, ignored, npig):
    """COCOeval.accumulate inner loop for one class/area/IoU/maxDet; None when npig == 0."""
    if npig == 0:
        return None
    inds = np.argsort(-scores, kind='mergesort')
    matched, ignored = matched[inds], ignored[inds]
    tp = np.cumsum(np.logical_and(matched, np.logical_not(ignored))).astype(float)
    fp = np.cumsum(np.logical_and(np.logical_not(matched), np.logical_not(ignored))).astype(float)
    q = np.zeros(len(REC))
    if len(tp) == 0:
        return q
    rc = tp / npig
    pr = tp / (fp + tp + np.spacing(1))
    pr = np.maximum.accumulate(pr[::-1])[::-1]
    idx = np.searchsorted(rc, REC, side='left')
    valid = idx < len(pr)
    # pycocotools leaves the remainder at zero once a recall threshold is unreachable
    q[valid] = pr[idx[valid]]
    return q


def macro(curves):
    """np.mean(s[s > -1]) over a (R, K) matrix; -1 when no class is evaluable."""
    P = -np.ones((len(REC), len(curves)))
    for k, q in enumerate(curves):
        if q is not None:
            P[:, k] = q
    valid = P[P > -1]
    return float(np.mean(valid)) if len(valid) else -1.


def run_curves(rt):
    return [curve(rt.score[s:t], rt.dtm[s:t] != 0, rt.dtig[s:t], int(rt.npig[k]))
            for k, (s, t) in enumerate(rt.slices)]


def check_port(e, runs):
    """Largest absolute difference between the port and the stored pycocotools precision."""
    worst = 0.0
    for (area, t), rt in runs.items():
        stored = e.eval['precision'][T_INDEX[t], :, :, area, 2]
        for k, q in enumerate(run_curves(rt)):
            if q is None:
                if not np.all(stored[:, k] == -1):
                    raise ValueError(f'port absent class differs: {area} {t} {k}')
            elif not np.array_equal(q, stored[:, k]):
                worst = max(worst, float(np.abs(q - stored[:, k]).max()))
    return worst


def classify_candidates(iou, det_cat, gt_cat, gt_used, tf, bg=BG):
    """Test order of TIDERun._eval_image over unmatched detections; argmax keeps first max."""
    D, G = len(det_cat), len(gt_cat)
    et = np.full(D, BKG, dtype=np.int8)
    tg = np.full(D, -1, dtype=np.int64)
    if D == 0 or G == 0:
        return et, tg
    same = det_cat[:, None] == gt_cat[None, :]
    cls_iou = iou * same
    non_iou = iou * np.logical_not(same)
    used_cls = cls_iou * gt_used[None, :]
    r = np.arange(D)
    i_loc = cls_iou.argmax(1)
    v_loc = cls_iou[r, i_loc]
    i_cls = non_iou.argmax(1)
    v_cls = non_iou[r, i_cls]
    v_dup = used_cls.max(1)
    v_any = iou.max(1)
    is_loc = np.logical_and(bg <= v_loc, v_loc <= tf)
    is_cls = np.logical_and(np.logical_not(is_loc), v_cls >= tf)
    rest = np.logical_not(np.logical_or(is_loc, is_cls))
    is_dup = np.logical_and(rest, v_dup >= tf)
    rest = np.logical_and(rest, np.logical_not(is_dup))
    is_bkg = np.logical_and(rest, v_any <= bg)
    is_both = np.logical_and(rest, np.logical_not(is_bkg))
    et[is_loc] = LOC
    et[is_cls] = CLS
    et[is_dup] = DUPE
    et[is_bkg] = BKG
    et[is_both] = BOTH
    tg[is_loc] = i_loc[is_loc]
    tg[is_cls] = i_cls[is_cls]
    return et, tg


def resolve_fixes(et, tg, score, gt_used, gt_nonig):
    """BestGTMatch over all Loc+Cls errors of the image; caller passes score-descending order."""
    G = len(gt_used)
    usable = np.zeros(G, dtype=bool)
    fix = np.zeros(len(et), dtype=bool)
    best_score = np.full(G, -1.0)
    best = np.full(G, -1, dtype=np.int64)
    for c in np.nonzero(np.logical_or(et == LOC, et == CLS))[0]:
        g = tg[c]
        if gt_used[g] or not gt_nonig[g]:
            continue
        usable[g] = True
        if best_score[g] < score[c]:
            best_score[g] = score[c]
            best[g] = c
    fix[best[best >= 0]] = True
    return fix, usable


def analyze_errors(gt, det, runs, n_img, classify=True):
    """Fill per-run channels and return per-GT best same-class IoU (all filtered / evaluated)."""
    from pycocotools import mask as mask_utils
    N = gt.n
    best_filtered = np.zeros(N)
    best_evaluated = np.zeros(N)
    evaluated = np.zeros(len(det.score), dtype=bool)
    first = next(iter(runs.values()))
    evaluated[first.det_id - 1] = True
    for rt in runs.values():
        if not np.array_equal(np.sort(rt.det_id), np.sort(first.det_id)):
            raise ValueError('evaluated detection set differs across runs')
        rt.init_errors()
    det_order = np.argsort(det.img, kind='stable')
    det_cuts = np.searchsorted(det.img[det_order], np.arange(1, n_img + 2))
    gt_nc = np.nonzero(np.logical_not(gt.crowd))[0]
    gt_order = gt_nc[np.argsort(gt.img[gt_nc], kind='stable')]
    gt_cuts = np.searchsorted(gt.img[gt_order], np.arange(1, n_img + 2))
    for i in range(n_img):
        d_idx = np.sort(det_order[det_cuts[i]:det_cuts[i + 1]])
        g_idx = np.sort(gt_order[gt_cuts[i]:gt_cuts[i + 1]])
        D, G = len(d_idx), len(g_idx)
        if D and G:
            iou = mask_utils.iou(det.bbox[d_idx], gt.bbox[g_idx], np.zeros(G, dtype=np.uint8))
            same = det.cat[d_idx][:, None] == gt.cat[g_idx][None, :]
            iou_same = iou * same
            best_filtered[g_idx] = iou_same.max(0)
            best_evaluated[g_idx] = (iou_same * evaluated[d_idx][:, None]).max(0)
        else:
            iou = np.zeros((D, G))
        if not classify:
            continue
        gid = gt.id[g_idx]
        for rt in runs.values():
            rows = rt.rows_of_image(i, n_img)
            cand = rows[rt.dtm[rows] == 0]
            if len(cand) == 0:
                continue
            cand = cand[np.lexsort((rt.det_id[cand], -rt.score[cand]))]
            local = np.searchsorted(d_idx, rt.det_id[cand] - 1)
            sub = iou[local] if G else np.zeros((len(cand), 0))
            used = rt.gt_used[gid]
            nonig = rt.gt_nonig[gid]
            et, tg = classify_candidates(sub, det.cat[rt.det_id[cand] - 1], gt.cat[g_idx], used, rt.t)
            fix, usable = resolve_fixes(et, tg, rt.score[cand], used, nonig)
            keep = np.logical_or(np.logical_not(rt.dtig[cand]),
                                 np.logical_and(np.logical_or(et == LOC, et == CLS), fix))
            rt.et[cand] = np.where(keep, et, NONE)
            rt.fix[cand] = fix
            has = tg >= 0
            rt.tgt[cand[has]] = gid[tg[has]]
            if G:
                rt.gt_usable[gid] = usable
    return best_filtered, best_evaluated


def oracle_curves(rt, gt, which):
    """One independent oracle from the unmodified run; returns (curves, emptied classes)."""
    K = len(rt.slices)
    extra = [[] for _ in range(K)]
    if which == 'Cls':
        for pos in np.nonzero(np.logical_and(rt.et == CLS, rt.fix))[0]:
            extra[int(gt.cat[rt.tgt[pos] - 1] - 1)].append(rt.score[pos])
    fn = np.logical_and(rt.gt_nonig, np.logical_not(rt.gt_used))
    drop = np.zeros(K, dtype=np.int64)
    if which in ('Miss', 'FN_recall'):
        mask = fn if which == 'FN_recall' else np.logical_and(fn, np.logical_not(rt.gt_usable))
        chosen = np.nonzero(mask)[0]
        if len(chosen):
            drop = np.bincount(gt.cat[chosen - 1] - 1, minlength=K).astype(np.int64)
    curves, empty = [], []
    code = dict(CHANNELS)
    for k, (s, t) in enumerate(rt.slices):
        score = rt.score[s:t]
        m = rt.dtm[s:t] != 0
        ig = rt.dtig[s:t].copy()
        fp = np.logical_and(np.logical_not(m), np.logical_not(ig))
        et = rt.et[s:t]
        fx = rt.fix[s:t]
        if which in ('Both', 'Dupe', 'Bkg'):
            ig[np.logical_and(et == code[which], fp)] = True
        elif which == 'Loc':
            this = et == LOC
            ig[np.logical_and(np.logical_and(this, fp), np.logical_not(fx))] = True
            to_tp = np.logical_and(this, fx)
            m = m.copy()
            m[to_tp] = True
            ig[to_tp] = False
        elif which == 'Cls':
            ig[np.logical_and(et == CLS, fp)] = True
        elif which == 'FP_rank':
            score = np.where(np.logical_and(m, np.logical_not(ig)), 1.0, 0.0)
        if extra[k]:
            add = np.asarray(extra[k], dtype=np.float64)
            score = np.concatenate((score, add))
            m = np.concatenate((m, np.ones(len(add), dtype=bool)))
            ig = np.concatenate((ig, np.zeros(len(add), dtype=bool)))
        n = int(rt.npig[k] - drop[k])
        if rt.npig[k] > 0 and n == 0:
            empty.append(k)
        curves.append(curve(score, m, ig, n))
    return curves, empty


def run_summary(rt, gt, per_class=True):
    m = rt.dtm != 0
    tp = np.logical_and(m, np.logical_not(rt.dtig))
    fp = np.logical_and(np.logical_not(m), np.logical_not(rt.dtig))
    used_nonig = np.logical_and(rt.gt_used, rt.gt_nonig)
    if int(tp.sum()) != int(used_nonig.sum()):
        raise ValueError('TP detections and matched in-range GT differ')
    if int(rt.gt_nonig.sum()) != int(rt.npig.sum()):
        raise ValueError('in-range GT count differs from npig')
    fn = np.logical_and(rt.gt_nonig, np.logical_not(rt.gt_used))
    miss = np.logical_and(fn, np.logical_not(rt.gt_usable))
    base = run_curves(rt)
    ap = macro(base)
    out = {'area': AREA_NAMES[rt.area], 'iou': rt.t, 'AP': ap, 'TP': int(tp.sum()), 'FP': int(fp.sum()),
           'GT': int(rt.npig.sum()), 'FN': int(fn.sum()), 'evaluator_ignored_detections': int(rt.dtig.sum()),
           'FN_targeted_by_Loc_or_Cls': int(fn.sum() - miss.sum()),
           'class_AP': [None if q is None else float(np.mean(q)) for q in base],
           'class_GT': rt.npig.tolist()}
    counts, details = {}, {}
    total = 0
    for name, code in CHANNELS:
        counts[name] = int(np.logical_and(rt.et == code, fp).sum())
        total += counts[name]
    if total != out['FP']:
        raise ValueError('FP channels do not partition FP')
    tgt_nonig = np.zeros(len(rt.et), dtype=bool)
    tgt_used = np.zeros(len(rt.et), dtype=bool)
    has = rt.tgt > 0
    tgt_nonig[has] = rt.gt_nonig[rt.tgt[has]]
    tgt_used[has] = rt.gt_used[rt.tgt[has]]
    for name, code in (('Cls', CLS), ('Loc', LOC)):
        this = rt.et == code
        this_fp = np.logical_and(this, fp)
        details[name] = {
            'FP_to_TP': int(np.logical_and(this_fp, rt.fix).sum()),
            'FP_suppressed_same_GT': int(np.logical_and.reduce(
                (this_fp, np.logical_not(rt.fix), tgt_nonig, np.logical_not(tgt_used))).sum()),
            'FP_target_already_matched': int(np.logical_and.reduce((this_fp, tgt_nonig, tgt_used)).sum()),
            'FP_target_out_of_area': int(np.logical_and.reduce((this_fp, has, np.logical_not(tgt_nonig))).sum()),
            'area_ignored_det_to_TP': int(np.logical_and.reduce((this, rt.dtig, rt.fix)).sum())}
    out['counts'] = {**counts, 'Miss': int(miss.sum())}
    out['fix_details'] = details
    delta, class_delta, empties = {}, {}, {}
    for which in ORACLES:
        cs, empty = oracle_curves(rt, gt, which)
        delta[which] = (macro(cs) - ap) * 100.
        empties[which] = empty
        if per_class:
            class_delta[which] = [None if (q is None or b is None) else (float(np.mean(q)) - float(np.mean(b))) * 100.
                                  for q, b in zip(cs, base)]
    out['delta_pp'] = delta
    if any(empties.values()):
        out['oracle_classes_emptied'] = {k: v for k, v in empties.items() if v}
    if per_class:
        out['class_delta_pp'] = class_delta
    return out


def group_metrics(e, groups):
    """AP-style point metrics on image subsets from stored evalImgs; never reweights subsets."""
    p = e.params
    K, I, A, T = len(p.catIds), len(p.imgIds), len(p.areaRng), len(p.iouThrs)
    R = len(REC)

    def mean(M):
        return float(np.mean(M[M > -1])) if np.any(M > -1) else -1.

    out = {}
    for name, images in groups.items():
        P = {a: -np.ones((T, R, K)) for a in (AREA_ALL, AREA_SMALL)}
        npig = {a: np.zeros(K, dtype=np.int64) for a in P}
        for k in range(K):
            for a in P:
                E = [e.evalImgs[k * A * I + a * I + i] for i in images]
                E = [x for x in E if x is not None]
                if not E:
                    continue
                gi = np.concatenate([x['gtIgnore'] for x in E])
                npig[a][k] = int(np.count_nonzero(gi == 0))
                if npig[a][k] == 0:
                    continue
                scores = np.concatenate([np.asarray(x['dtScores'], dtype=np.float64) for x in E])
                dtm = np.concatenate([x['dtMatches'] for x in E], axis=1)
                dti = np.concatenate([x['dtIgnore'] for x in E], axis=1)
                for t in range(T):
                    P[a][t, :, k] = curve(scores, dtm[t] != 0, dti[t], int(npig[a][k]))
        out[name] = {'images': len(images), 'AP': mean(P[AREA_ALL]), 'AP50': mean(P[AREA_ALL][0]),
                     'AP75': mean(P[AREA_ALL][5]), 'APS': mean(P[AREA_SMALL]),
                     'AP50_small': mean(P[AREA_SMALL][0]), 'AP75_small': mean(P[AREA_SMALL][5]),
                     'GT_all': int(npig[AREA_ALL].sum()), 'GT_small': int(npig[AREA_SMALL].sum()),
                     'classes_all': [k + 1 for k in range(K) if npig[AREA_ALL][k] > 0],
                     'classes_small': [k + 1 for k in range(K) if npig[AREA_SMALL][k] > 0],
                     'class_GT_small': npig[AREA_SMALL].tolist()}
    return out


def min_side_bin(w, h):
    side = np.minimum(w, h)
    if np.any(side <= 0) or np.any(side > 32.0):
        raise ValueError('small GT min side outside (0,32]')
    return np.searchsorted(BIN_EDGES, side, side='left')


def control_registration(x0, y0, x1, y1, width, height, canvas_hw=(768, 1344)):
    """Geometry of zoom_refine.control_pixels: integer patch against the exact window rect."""
    ch, cw = int(canvas_hw[0]), int(canvas_hw[1])
    sx, sy = cw / width, ch / height
    rx0, ry0, rx1, ry1 = x0 * sx, y0 * sy, x1 * sx, y1 * sy
    i0, j0 = max(int(math.floor(ry0)), 0), max(int(math.floor(rx0)), 0)
    i1, j1 = min(int(math.ceil(ry1)), ch), min(int(math.ceil(rx1)), cw)
    return {'sx': sx, 'sy': sy, 'j0': j0, 'j1': j1, 'i0': i0, 'i1': i1,
            'error_left': x0 - j0 / sx, 'error_right': x1 - j1 / sx,
            'error_top': y0 - i0 / sy, 'error_bottom': y1 - i1 / sy}


def corrected_coordinate(x, wx0, wlen, j0, j1, s):
    """Raw position actually shown at the mapped coordinate x of a control-arm window."""
    f = (x - wx0) / wlen
    return (j0 + f * (j1 - j0)) / s


def registration_error(x, wx0, wlen, j0, j1, s):
    """Mapped raw coordinate minus the raw position the control pixels actually show."""
    return x - corrected_coordinate(x, wx0, wlen, j0, j1, s)


def ols(x, y):
    n = len(x)
    if n < 2:
        return {'n': n, 'slope': None, 'intercept': None}
    mx, my = float(np.mean(x)), float(np.mean(y))
    var = float(np.sum((x - mx) ** 2))
    if var < 1e-18:  # the predicted shift is zero up to float noise (full-image windows)
        return {'n': n, 'slope': None, 'intercept': my, 'mean_abs_y': float(np.mean(np.abs(y)))}
    slope = float(np.sum((x - mx) * (y - my)) / var)
    return {'n': n, 'slope': slope, 'intercept': my - slope * mx,
            'mean_abs_y': float(np.mean(np.abs(y)))}
