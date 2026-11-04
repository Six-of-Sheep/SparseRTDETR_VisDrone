"""CPU checks for the cached error channels.  Synthetic fixtures only; no experiment data.

usage: python -B -I test_core.py --tide-root DIR --frozen-src DIR --zoom-refine FILE
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import core as C
import tide_native as N


def dataset_of(images, gts):
    anns = []
    for img, cat, box, crowd in gts:
        anns.append({'id': len(anns) + 1, 'image_id': img, 'category_id': cat, 'bbox': list(box),
                     'area': float(box[2] * box[3]), 'iscrowd': int(crowd)})
    return {'info': {}, 'images': [{'id': i, 'width': 1000, 'height': 1000} for i in range(1, images + 1)],
            'categories': [{'id': i, 'name': str(i)} for i in range(1, 11)], 'annotations': anns}


def det_of(rows):
    return [{'image_id': i, 'category_id': c, 'bbox': list(b), 'score': float(s)} for i, c, b, s in rows]


def prepared(dataset, det, n_img):
    e, values = C.evaluate_certified(dataset, det)
    gt = C.gt_table(dataset)
    dt = C.det_table(det)
    runs = {(a, t): C.collect_run(e, a, t, gt.n) for a in (C.AREA_ALL, C.AREA_SMALL) for t in (0.5, 0.75)}
    C.analyze_errors(gt, dt, runs, n_img)
    return e, values, gt, dt, runs


def synthetic(seed, crowd, extra_class=True):
    rng = np.random.default_rng(seed)
    top = 8 if extra_class else 7  # one detection-only class differs by design from tidecv
    gts, rows = [], []
    n_img = 24
    for img in range(1, n_img + 1):
        for _ in range(rng.integers(0, 30)):
            cat = int(rng.integers(1, 7))
            side = float(rng.choice([6, 9, 13, 18, 24, 31, 32, 40, 64, 120]))
            w, h = side * rng.uniform(.7, 1.3), side * rng.uniform(.7, 1.3)
            x, y = rng.uniform(0, 800), rng.uniform(0, 800)
            is_crowd = crowd and rng.random() < .05
            gts.append((img, cat, (x, y, w, h), is_crowd))
            for _ in range(rng.integers(0, 4)):
                jitter = rng.normal(0, rng.choice([.02, .1, .3]), 4) * max(w, h)
                c = cat if rng.random() < .85 else int(rng.integers(1, 7))
                b = (x + jitter[0], y + jitter[1], max(w + jitter[2], 1.), max(h + jitter[3], 1.))
                rows.append((img, c, b, rng.random()))
        for _ in range(rng.integers(0, 25)):
            s = rng.uniform(3, 60)
            rows.append((img, int(rng.integers(1, top)), (rng.uniform(0, 900), rng.uniform(0, 900), s, s), rng.random()))
    return n_img, dataset_of(n_img, gts), det_of(rows)


def test_port_and_groups():
    for crowd in (False, True):
        n_img, dataset, det = synthetic(11 + crowd, crowd)
        e, values, gt, dt, runs = prepared(dataset, det, n_img)
        assert C.check_port(e, runs) == 0.0
        for (a, t), rt in runs.items():
            assert C.macro(C.run_curves(rt)) == float(np.mean(
                e.eval['precision'][C.T_INDEX[t], :, :, a, 2][e.eval['precision'][C.T_INDEX[t], :, :, a, 2] > -1]))
            C.run_summary(rt, gt)
        full = C.group_metrics(e, {'ALL': list(range(n_img))})['ALL']
        for key in ('AP', 'AP50', 'AP75', 'APS', 'AP75_small'):
            assert full[key] == values[key], key
        subset = list(range(0, n_img, 3))
        mine = C.group_metrics(e, {'S': subset})['S']
        ds = copy.deepcopy(dataset)
        keep = {i + 1 for i in subset}
        ds['images'] = [r for r in ds['images'] if r['id'] in keep]
        ds['annotations'] = [g for g in ds['annotations'] if g['image_id'] in keep]
        for i, g in enumerate(ds['annotations']):
            g['id'] = i + 1
        sub_det = [{k: d[k] for k in ('image_id', 'category_id', 'bbox', 'score')} for d in det if d['image_id'] in keep]
        _, sub_values = C.evaluate_certified(ds, sub_det)
        for key in ('AP', 'AP50', 'AP75', 'APS', 'AP75_small'):
            assert mine[key] == sub_values[key], (key, mine[key], sub_values[key])


def test_frozen_bridge_metrics(frozen_src):
    sys.path.insert(0, str(frozen_src))
    from sparse_rtdetr.r0 import evaluation as ev
    n_img, dataset, det = synthetic(5, True)
    frozen = ev.bridge_metrics(copy.deepcopy(dataset), copy.deepcopy(det))['metrics']
    _, mine = C.evaluate_certified(copy.deepcopy(dataset), copy.deepcopy(det))
    assert frozen == mine


def test_scene_small_semantics():
    gts = [(1, 1, (100, 100, 20, 20), False),  # g1 small, TP
           (1, 1, (300, 300, 50, 50), False),  # g2 medium, out of the small range
           (1, 2, (500, 500, 20, 20), False),  # g3 small, wrong-class detection
           (1, 1, (700, 700, 20, 20), False),  # g4 small, only an out-of-range detection
           (1, 3, (800, 800, 20, 20), False),  # g5 small, nothing near
           (2, 1, (300, 300, 50, 50), False),  # g6 medium matched exactly
           (3, 1, (0, 0, 34, 34), False)]      # g7 medium with a small duplicate
    rows = [(1, 1, (100, 100, 20, 20), .9),    # d1 TP
            (1, 1, (100.5, 100.5, 20, 20), .8),  # d2 duplicate of g1
            (1, 1, (300, 300, 30, 30), .7),    # d3 small, IoU .36 with medium g2
            (1, 1, (500, 500, 20, 20), .6),    # d4 class error on g3
            (1, 1, (695, 695, 35, 35), .5),    # d5 area 1225, IoU .327 with g4
            (1, 1, (10, 10, 20, 20), .4),      # d6 background
            (1, 2, (110, 110, 20, 20), .3),    # d7 class and location error around g1
            (2, 1, (300, 300, 50, 50), .95),   # d8 matched to out-of-range g6
            (3, 1, (0, 0, 34, 34), .9),        # d9 matched to out-of-range g7
            (3, 1, (0, 0, 31, 31), .8)]        # d10 small duplicate of out-of-range g7
    n_img = 3
    e, values, gt, dt, runs = prepared(dataset_of(n_img, gts), det_of(rows), n_img)
    s = C.run_summary(runs[(C.AREA_SMALL, 0.75)], gt)
    assert s['GT'] == 4 and s['TP'] == 1 and s['FN'] == 3, s
    assert s['counts'] == {'Cls': 1, 'Loc': 1, 'Both': 1, 'Dupe': 2, 'Bkg': 1, 'Miss': 1}, s['counts']
    assert s['FN_targeted_by_Loc_or_Cls'] == 2
    assert s['fix_details']['Loc']['FP_target_out_of_area'] == 1
    assert s['fix_details']['Loc']['area_ignored_det_to_TP'] == 1
    assert s['fix_details']['Cls']['FP_to_TP'] == 1
    # hand computed macro AP over classes 1..3; class 3 has no detections
    r = 101.
    base = (51 / r + 0 + 0) / 3
    assert abs(s['AP'] - base) < 1e-12, (s['AP'], base)
    assert abs(s['delta_pp']['Cls'] - ((51 / r + 1 + 0) / 3 - base) * 100) < 1e-9
    # Loc oracle: d3 removed, d5 becomes TP behind d2, d10, d4; precision at full recall 2/5
    assert abs(s['delta_pp']['Loc'] - (((51 + 50 * .4) / r + 0 + 0) / 3 - base) * 100) < 1e-9
    assert abs(s['delta_pp']['Miss'] - ((51 / r + 0) / 2 - base) * 100) < 1e-9
    assert s['oracle_classes_emptied'] == {'Miss': [2], 'FN_recall': [2]} or 'Miss' in s['oracle_classes_emptied']
    assert s['delta_pp']['Both'] == 0 and s['delta_pp']['Bkg'] == 0
    a = C.run_summary(runs[(C.AREA_ALL, 0.75)], gt)
    assert a['GT'] == 7 and a['counts']['Bkg'] == 1
    assert a['counts']['Dupe'] == 2
    assert a['fix_details']['Loc']['area_ignored_det_to_TP'] == 0


def test_against_original_tide(tide_root):
    quantify, data = N.load_tide(tide_root)
    n_img, dataset, det = synthetic(23, False, extra_class=False)
    e, values, gt, dt, runs = prepared(dataset, det, n_img)
    g, p = N.build_data(data, n_img, gt.img, gt.cat, gt.bbox, gt.crowd, dt.img, dt.cat, dt.score, dt.bbox)
    for t in (0.5, 0.75):
        mine = C.run_summary(runs[(C.AREA_ALL, t)], gt)
        native = N.native_run(quantify, g, p, t)
        assert native['counts'] == mine['counts'], (t, native['counts'], mine['counts'])
        assert native['errors_on_ignored_predictions'] == 0
        assert abs(native['AP'] - mine['AP'] * 100) < 1e-6, (native['AP'], mine['AP'])
        for name in ('Cls', 'Loc', 'Both', 'Dupe', 'Bkg', 'Miss'):
            assert abs(native['delta_pp_signed'][name] - mine['delta_pp'][name]) < 1e-6, (
                t, name, native['delta_pp_signed'][name], mine['delta_pp'][name])
        assert abs(native['FP_rank_delta_pp'] - mine['delta_pp']['FP_rank']) < 1e-6
        assert abs(native['FN_recall_delta_pp'] - mine['delta_pp']['FN_recall']) < 1e-6


def test_registration(zoom_refine):
    spec = importlib.util.spec_from_file_location('frozen_zoom_refine', zoom_refine)
    module = importlib.util.module_from_spec(spec)
    sys.modules['frozen_zoom_refine'] = module
    spec.loader.exec_module(module)
    for W, H, x0, y0 in ((1360, 765, 200, 100), (1920, 1080, 108, 199), (960, 540, 0, 0), (1360, 765, 400, 225)):
        ww, wh = min(960, W), min(540, H)
        w = module.Window(1, x0, y0, x0 + ww, y0 + wh, W, H)
        rect = module.canvas_rect(w)
        g = C.control_registration(x0, y0, x0 + ww, y0 + wh, W, H)
        assert g['sx'] == 1344 / W and g['sy'] == 768 / H
        import math
        assert g['j0'] == max(math.floor(rect[0]), 0) and g['j1'] == min(math.ceil(rect[2]), 1344)
        assert g['i0'] == max(math.floor(rect[1]), 0) and g['i1'] == min(math.ceil(rect[3]), 768)
        # both ends of the window map through the integer patch edges
        assert abs(C.corrected_coordinate(x0, x0, ww, g['j0'], g['j1'], g['sx']) - g['j0'] / g['sx']) < 1e-9
        assert abs(C.registration_error(x0 + ww, x0, ww, g['j0'], g['j1'], g['sx']) - g['error_right']) < 1e-9
    full = C.control_registration(0, 0, 960, 540, 960, 540)
    assert full['j0'] == 0 and full['j1'] == 1344 and full['i0'] == 0 and full['i1'] == 768
    assert abs(full['error_left']) + abs(full['error_right']) + abs(full['error_top']) + abs(full['error_bottom']) < 1e-9


def test_bins():
    assert C.ols(np.array([1e-13, 1e-13, 1.1e-13]), np.array([1., 2., 3.]))['slope'] is None
    fit = C.ols(np.array([0., 1., 2.]), np.array([1., 3., 5.]))
    assert abs(fit['slope'] - 2) < 1e-12 and abs(fit['intercept'] - 1) < 1e-12
    assert C.min_side_bin(np.array([8., 8.5, 12., 12.1, 17., 17.5, 32.]), np.array([8., 40., 12., 13., 90., 20., 32.])).tolist() == [0, 1, 1, 2, 2, 3, 3]
    try:
        C.min_side_bin(np.array([33.]), np.array([33.]))
    except ValueError:
        return
    raise AssertionError('min side above 32 must be rejected')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tide-root', required=True)
    ap.add_argument('--frozen-src', required=True)
    ap.add_argument('--zoom-refine', required=True)
    args = ap.parse_args()
    test_bins()
    print('PASS bins', flush=True)
    test_port_and_groups()
    print('PASS port and groups', flush=True)
    test_scene_small_semantics()
    print('PASS small semantics', flush=True)
    test_against_original_tide(args.tide_root)
    print('PASS original tidecv agreement', flush=True)
    test_frozen_bridge_metrics(args.frozen_src)
    print('PASS frozen bridge_metrics agreement', flush=True)
    test_registration(args.zoom_refine)
    print('PASS registration geometry', flush=True)
    print('ALL PASS')


if __name__ == '__main__':
    main()
