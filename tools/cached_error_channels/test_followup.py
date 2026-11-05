"""CPU checks for the follow-up shift and the one-to-one offset helpers.  Synthetic fixtures only.

usage: python -B -I test_followup.py
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import core as C
import followup_shift as F


def test_shift_fields():
    rng = np.random.default_rng(5)
    preds = []
    for i in range(2000):
        x0 = float(np.float32(rng.uniform(-5, 1925)))
        y0 = float(np.float32(rng.uniform(-5, 1085)))
        w = float(np.float32(rng.uniform(.5, 300)))
        h = float(np.float32(rng.uniform(.5, 300)))
        preds.append({'image_id': f'im{i % 7}', 'query_id': i % 300, 'category_id': 1 + i % 10, 'score': float(rng.random()),
                      'bbox_xyxy': [x0, y0, float(np.float32(x0 + w)), float(np.float32(y0 + h))], 'view_id': i % 5})
    preds.append({'image_id': 'tiny', 'query_id': 0, 'category_id': 1, 'score': .5, 'bbox_xyxy': [1e-12, 0.0, 3e-12, 1.0]})
    shifted, inexact = F.shift_predictions(preds)
    assert len(shifted) == len(preds)
    for p, q in zip(preds, shifted):
        assert set(p) == set(q) and all(p[k] == q[k] for k in p if k != 'bbox_xyxy')
        assert q['bbox_xyxy'][1] == p['bbox_xyxy'][1] and q['bbox_xyxy'][3] == p['bbox_xyxy'][3]
        assert abs(q['bbox_xyxy'][0] - (p['bbox_xyxy'][0] - .25)) < 1e-12 and abs(q['bbox_xyxy'][2] - (p['bbox_xyxy'][2] - .25)) < 1e-12
    assert inexact == 1, inexact  # only the 1e-12 box cannot keep its width bit for bit
    assert preds[0]['bbox_xyxy'] != shifted[0]['bbox_xyxy']  # the source list is never modified in place
    back, _ = F.shift_predictions(shifted, .25)
    assert sum(p != q for p, q in zip(preds[:-1], back[:-1])) == 0
    try:
        F.shift_predictions([{'bbox': [0, 0, 1, 1]}])
        raise AssertionError('xywh cache must be refused')
    except ValueError:
        pass


def scene():
    """GT 10x10 boxes; the top-scoring detection of each GT sits at +0.3 px in x, a lower one at -2 px."""
    anns, det = [], []
    for img in range(1, 4):
        for k in range(5):
            x, y = 50. * k + 10, 40. * img
            side = 10. if k < 4 else 50.
            anns.append({'id': len(anns) + 1, 'image_id': img, 'category_id': 1 + k % 2, 'bbox': [x, y, side, side],
                         'area': side * side, 'iscrowd': 0})
            det.append({'image_id': img, 'category_id': 1 + k % 2, 'bbox': [x + .3, y, side, side], 'score': .9 - .01 * k})
            det.append({'image_id': img, 'category_id': 1 + k % 2, 'bbox': [x - 2., y, side, side], 'score': .5})
    dataset = {'info': {}, 'images': [{'id': i, 'width': 400, 'height': 300} for i in range(1, 4)],
               'categories': [{'id': i, 'name': str(i)} for i in range(1, 11)], 'annotations': anns}
    return dataset, det


def test_one_to_one_residuals():
    dataset, det = scene()
    e, values = C.evaluate_certified(dataset, det)
    gt, dt = C.gt_table(dataset), C.det_table(det)
    rt_all = C.collect_run(e, C.AREA_ALL, .5, gt.n)
    rt_small = C.collect_run(e, C.AREA_SMALL, .5, gt.n)
    r_all = F.residuals(rt_all, gt, dt)
    r_small = F.residuals(rt_small, gt, dt)
    assert not np.isnan(r_all[:, 0]).any()
    assert np.allclose(r_all[:, 0], .3) and np.allclose(r_all[:, 1:], 0.)
    small = gt.area <= 1024
    assert np.allclose(r_small[small, 0], .3) and np.isnan(r_small[~small, 0]).all()
    size = np.array(['400x300'] * gt.n)
    seq = np.array(['s1'] * 10 + ['s2'] * 5)
    summary = F.offset_summary(r_all, r_small, gt, size, seq)
    assert summary['by_area_band']['small']['n'] == 12 and summary['by_area_band']['medium']['n'] == 3
    assert abs(summary['by_area_band']['small']['cx']['mean'] - .3) < 1e-9
    assert summary['small_sequences']['with_at_least_20_pairs'] == 0


def test_flips():
    a = np.array([1, 1, 0, 0, 1], dtype=bool)
    b = np.array([1, 0, 1, 0, 1], dtype=bool)
    m = np.array([1, 1, 1, 1, 0], dtype=bool)
    assert F.flips(a, b, m) == {'gained': 1, 'lost': 1, 'both': 1, 'neither': 1}



def test_match_only_never_accumulates():
    from types import SimpleNamespace
    from pycocotools.cocoeval import COCOeval
    dataset, det = scene()
    ctx = SimpleNamespace(rows=[], parsed=[], groups={'ALL': [0,1,2]},
                          ev=SimpleNamespace(bridge_inputs=lambda rows, parsed, preds: (dataset, det, {'test': 1})))
    saved_map = F.RC.map_det_to_pred
    F.RC.map_det_to_pred = lambda preds, det, rows: np.arange(len(det))
    try:
        orig = F.evaluate_list(ctx, [])
        previous = {'certified_metrics_reproduced': orig['values'], 'groups': orig['groups']}
        saved_accumulate, saved_summarize = COCOeval.accumulate, COCOeval.summarize
        def forbidden(*args, **kwargs):
            raise AssertionError('match-only replay must not score')
        COCOeval.accumulate = forbidden
        COCOeval.summarize = forbidden
        try:
            replay = F.evaluate_list(ctx, [], previous=previous)
        finally:
            COCOeval.accumulate, COCOeval.summarize = saved_accumulate, saved_summarize
        for k in orig['used']:
            assert np.array_equal(orig['used'][k], replay['used'][k])
        assert np.array_equal(orig['res_small'], replay['res_small'], equal_nan=True)
        assert replay['values'] is previous['certified_metrics_reproduced']
        assert replay['class_ap'] == {}
    finally:
        F.RC.map_det_to_pred = saved_map

def main():
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print('PASS', name, flush=True)
    print('ALL PASS', flush=True)


if __name__ == '__main__':
    main()
